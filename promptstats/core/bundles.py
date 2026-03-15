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
from .ranking import RankDistribution, PointAdvantageResult
from .variance import RobustnessResult, SeedVarianceResult
from .tokens import TokenAnalysisResult

if TYPE_CHECKING:
    from .mixed_effects import LMMInfo, FactorialLMMInfo


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
    point_advantage : PointAdvantageResult
        Mean advantage of each template over a reference, with
        epistemic CI and intrinsic spread bands.
    robustness : RobustnessResult
        Per-template robustness and variance metrics (on cell means).
    rank_dist : RankDistribution
        Bootstrap distribution over template rankings.
    seed_variance : SeedVarianceResult or None
        Seed-variance decomposition (instability scores).  Present only
        when the benchmark carries R >= 3 repeated runs.
    token_analysis : TokenAnalysisResult or None
        Token cost analysis with Pareto frontier.  Present only when
        token_usage was passed to analyze().
    lmm_info : LMMInfo or None
        Variance components and ICC from a standard one-factor LMM.
        Present only when method='lmm' was used.
    factorial_lmm_info : FactorialLMMInfo or None
        Factor tests and marginal means from a factorial LMM.  Present
        only when analyze_factorial() was used.
    """

    benchmark: BenchmarkResult
    shape: BenchmarkShape
    pairwise: PairwiseMatrix
    point_advantage: PointAdvantageResult
    robustness: RobustnessResult
    rank_dist: RankDistribution
    seed_variance: Optional[SeedVarianceResult] = None
    token_analysis: Optional[TokenAnalysisResult] = None
    lmm_info: Optional["LMMInfo"] = None
    factorial_lmm_info: Optional["FactorialLMMInfo"] = None
    resolved_method: Optional[str] = None


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
