"""Result bundle dataclasses and type aliases for analyze() output.

Kept in a separate module so that both the analysis router (router.py) and
the console summary formatter (summary.py) can import these types without
creating a circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Union

from .types import BenchmarkResult, MultiModelBenchmark
from .paired import PairwiseMatrix
from .ranking import RankDistribution
from .variance import RobustnessResult, SeedVarianceResult

if TYPE_CHECKING:
    from .mixed_effects import LMMInfo, FactorialLMMInfo
    from ..alignment import AlignmentResult


# ---------------------------------------------------------------------------
# Shape descriptor
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkShape:
    """Detected structural properties of a benchmark input.

    Attributes
    ----------
    n_models : int
        Number of distinct LLM models. 1 for BenchmarkResult; ≥2 for
        MultiModelBenchmark.
    n_prompts : int
        Number of prompt templates (templates per model).
    n_input_vars : int
        Number of independent input variables. 1 when each benchmark
        input is a single value; >1 when input_labels are tuples
        representing a cross-product of variables.
    n_evaluators : int
        Number of evaluators/scorers.
    n_runs : int
        Number of repeated runs (seeds) per cell. 1 means no seed dimension.
    """

    n_models: int
    n_prompts: int
    n_input_vars: int
    n_evaluators: int
    n_runs: int = 1

    def __repr__(self) -> str:
        runs_str = f", runs={self.n_runs}" if self.n_runs > 1 else ""
        return (
            f"BenchmarkShape(models={self.n_models}, prompts={self.n_prompts}, "
            f"input_vars={self.n_input_vars}, evaluators={self.n_evaluators}"
            f"{runs_str})"
        )


# ---------------------------------------------------------------------------
# Result bundles
# ---------------------------------------------------------------------------

@dataclass
class AnalysisBundle:
    """Consolidated results from a single-model benchmark analysis run.

    Attributes
    ----------
    benchmark : BenchmarkResult
        The underlying benchmark data.
    shape : BenchmarkShape
        Detected structural properties used for routing.
    pairwise : PairwiseMatrix
        All pairwise statistical comparisons between templates.
    robustness : RobustnessResult
        Per-template absolute performance and robustness metrics
        (on cell means), including marginal CI bounds.
    rank_dist : RankDistribution
        Bootstrap distribution over template rankings.
    seed_variance : SeedVarianceResult or None
        Seed-variance decomposition (instability scores).  Present only
        when the benchmark carries R >= 3 repeated runs.
    lmm_info : LMMInfo or None
        Variance components and ICC from a standard one-factor LMM.
        Present only when method='lmm' was used.
    factorial_lmm_info : FactorialLMMInfo or None
        Factor tests and marginal means from a factorial LMM.  Present
        only when analyze_factorial() was used.
    ppi_applied : bool
        True when ``compare(..., alignment=...)`` overrode this bundle's
        robustness/pairwise/rank_dist with a Prediction-Powered Inference
        correction (see ``evalstats.api._run_alignment_ppi``).
    alignment_result : AlignmentResult or None
        The :class:`~evalstats.alignment.AlignmentResult` the correction
        above was computed from -- set together with ``ppi_applied``,
        ``None`` otherwise. Lets the summary printer show the full
        alignment/representativeness report inline instead of just the
        boolean flag.
    resolved_score_range : tuple[float, float] or None
        The ``(lo, hi)`` bounds actually used to rescale data for
        ``resolved_method='logit_t'`` / ``resolved_ci_method='logit_t'``
        (user-declared via ``score_range``, or auto-detected/approximated —
        see ``analyze()``'s ``score_range`` parameter). ``None`` when
        logit-t wasn't used.
    resolved_data_kind : str or None
        The data kind (``"binary"``/``"bounded_01"``/``"likert"``/
        ``"unbounded"``) the ``method="auto"`` router actually resolved for
        this data -- see ``evalstats.core.router.resolve_auto_robustness_method``.
        Recorded so downstream consumers reuse that ONE decision instead of
        re-deriving it from the scores. ``evalstats.api._run_alignment_ppi``
        does exactly that when routing PPI's own ``method="auto"``: it
        previously re-derived the kind with a binary/bounded_01/unbounded
        test of its own, which had no ``"likert"`` branch and consulted
        neither ``score_range`` nor ``eval_type``, so Likert data on e.g. a
        1-5 scale fell through to ``"unbounded"`` and silently took
        ``ppi_t_interval`` -- leaving ``PPI_AUTO_METHOD_TABLE``'s ``likert``
        row (``ppi_logit_t``) unreachable. ``None`` for non-``auto`` methods
        and the LMM paths, where no such resolution happens.
    """

    benchmark: BenchmarkResult
    shape: BenchmarkShape
    pairwise: PairwiseMatrix
    robustness: RobustnessResult
    rank_dist: RankDistribution
    seed_variance: Optional[SeedVarianceResult] = None
    lmm_info: Optional["LMMInfo"] = None
    factorial_lmm_info: Optional["FactorialLMMInfo"] = None
    resolved_method: Optional[str] = None
    resolved_ci_method: Optional[str] = None
    resolved_score_range: Optional[tuple[float, float]] = None
    resolved_data_kind: Optional[str] = None
    p_value_method: Optional[str] = None
    ppi_applied: bool = False
    alignment_result: Optional["AlignmentResult"] = None

    @property
    def labels(self) -> list[str]:
        """Canonical entity labels for this bundle.

        The single source of truth is ``benchmark.template_labels`` -- the
        same list ``core.router`` feeds to every downstream construction.
        Read this rather than ``rank_dist.labels``: the rank distribution is
        opt-in work (see ``core.ranking.LazyRankDistribution``), so treating
        it as the label registry both inverts the dependency and can force a
        bootstrap nobody asked for.
        """
        return list(self.benchmark.template_labels)

    def summary(self, **kwargs) -> None:
        """Print the console summary for this bundle.

        Thin wrapper around :func:`evalstats.core.summary.print_analysis_summary`
        so that ``analyze()`` results can be printed the same way as
        :meth:`~evalstats.api.ComparisonResult.summary`. See that function
        for accepted keyword arguments.
        """
        from .summary import print_analysis_summary
        print_analysis_summary(self, **kwargs)


@dataclass
class MultiModelBundle:
    """Consolidated results from a multi-model benchmark analysis run.

    Contains three complementary views of the data:

    * **per_model** — one AnalysisBundle per model, answering "which
      prompt works best *within* each model?"
    * **model_level** — models compared on their mean score across all
      prompts, answering "which model is overall best?"
    * **template_level** — templates compared on their mean score across
      all models, answering "which prompt is best/worst overall?"
    * **cross_model** — all (model, template) pairs ranked together,
      answering "what is the single best model-prompt combination?"

    Attributes
    ----------
    benchmark : MultiModelBenchmark
        The underlying benchmark data.
    shape : BenchmarkShape
        Detected structural properties used for routing.
    per_model : dict[str, AnalysisBundle]
        One full analysis bundle per model, keyed by model label.
    model_level : AnalysisBundle
        Analysis where each 'template' is a model, scored by its mean
        performance across all prompts.
    template_level : AnalysisBundle
        Analysis where each 'template' is a prompt template, scored by
        its mean performance across all models.
    cross_model : AnalysisBundle
        Analysis of all N_models * N_templates (model, template) pairs
        treated as a flat list of 'templates'.
    best_pair : tuple[str, str]
        The (model_label, template_label) pair with the highest
        probability of ranking first in the cross_model analysis.
    """

    benchmark: MultiModelBenchmark
    shape: BenchmarkShape
    per_model: Dict[str, AnalysisBundle]
    model_level: AnalysisBundle
    template_level: AnalysisBundle
    cross_model: AnalysisBundle
    best_pair: tuple[str, str]

    def summary(self, **kwargs) -> None:
        """Print the console summary for this bundle (see :meth:`AnalysisBundle.summary`)."""
        from .summary import print_analysis_summary
        print_analysis_summary(self, **kwargs)


# ---------------------------------------------------------------------------
# Type aliases for analyze() return type
# ---------------------------------------------------------------------------

PerEvaluatorSingleModel = Dict[str, AnalysisBundle]
PerEvaluatorMultiModel = Dict[str, MultiModelBundle]
AnalysisResult = Union[
    AnalysisBundle,
    PerEvaluatorSingleModel,
    MultiModelBundle,
    PerEvaluatorMultiModel,
]
