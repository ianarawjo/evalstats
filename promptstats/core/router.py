"""Central router for selecting the appropriate analysis pipeline.

Inspects the 'shape' of a BenchmarkResult — number of models, prompt
templates, input variables, and evaluators — and dispatches to the
correct analysis functions. Raises informative errors for shapes that
are not yet supported.

Currently supported shape:
    models=1, prompts>1, input_vars=1, evaluators>=1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Mapping, Optional, Union

import numpy as np

from .types import BenchmarkResult
from .paired import PairwiseMatrix, all_pairwise
from .ranking import RankDistribution, MeanAdvantageResult, bootstrap_ranks, bootstrap_mean_advantage
from .variance import RobustnessResult, robustness_metrics


@dataclass
class BenchmarkShape:
    """Detected structural properties of a BenchmarkResult.

    Attributes
    ----------
    n_models : int
        Number of distinct LLM models. Always 1 for the current
        BenchmarkResult format; multi-model support is planned.
    n_prompts : int
        Number of prompt templates (axis 0 of the score matrix).
    n_input_vars : int
        Number of independent input variables. 1 when each benchmark
        input is a single value; >1 when input_labels are tuples
        representing a cross-product of variables.
    n_evaluators : int
        Number of evaluators/scorers. 1 when scores are 2D (already
        aggregated); >1 when scores are 3D (axis 2 = evaluators).
    """

    n_models: int
    n_prompts: int
    n_input_vars: int
    n_evaluators: int

    def __repr__(self) -> str:
        return (
            f"BenchmarkShape(models={self.n_models}, prompts={self.n_prompts}, "
            f"input_vars={self.n_input_vars}, evaluators={self.n_evaluators})"
        )


@dataclass
class AnalysisBundle:
    """Consolidated results from a full benchmark analysis run.

    Attributes
    ----------
    benchmark : BenchmarkResult
        The underlying benchmark data.
    shape : BenchmarkShape
        Detected structural properties used for routing.
    pairwise : PairwiseMatrix
        All pairwise statistical comparisons between templates.
    mean_advantage : MeanAdvantageResult
        Mean advantage of each template over a reference, with
        epistemic CI and intrinsic spread bands.
    robustness : RobustnessResult
        Per-template robustness and variance metrics.
    rank_dist : RankDistribution
        Bootstrap distribution over template rankings.
    """

    benchmark: BenchmarkResult
    shape: BenchmarkShape
    pairwise: PairwiseMatrix
    mean_advantage: MeanAdvantageResult
    robustness: RobustnessResult
    rank_dist: RankDistribution


AnalysisResult = Union[AnalysisBundle, Dict[str, AnalysisBundle]]


def analyze(
    result: BenchmarkResult,
    *,
    evaluator_mode: Literal["aggregate", "per_evaluator"] = "aggregate",
    reference: str = "grand_mean",
    method: Literal["bootstrap", "bca", "auto"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    spread_percentiles: tuple[float, float] = (10, 90),
    failure_threshold: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> AnalysisResult:
    """Run all standard analyses for a benchmark result.

    Inspects the shape of the benchmark and dispatches to the
    appropriate analysis pipeline. All four analyses (pairwise
    comparisons, mean advantage, robustness metrics, and rank
    distributions) share the same random state for reproducibility.

    Currently supported shape: models=1, prompts>1, input_vars=1,
    evaluators>=1 (single-variable inputs, one model, multiple prompts).

    Parameters
    ----------
    result : BenchmarkResult
        The benchmark data to analyze.
    evaluator_mode : str
        'aggregate' (default) analyzes the evaluator-aggregated 2D
        score matrix, matching prior behavior. 'per_evaluator' runs
        analyses separately for each evaluator and returns a dict
        keyed by evaluator label.
    reference : str
        Reference for mean advantage: 'grand_mean' (default) or a
        template label to compare all others against.
    method : str
        Statistical method for CIs and p-values: 'auto' (default),
        'bootstrap', or 'bca'. 'auto' selects BCa for 15–200 inputs.
    ci : float
        Confidence level for bootstrap intervals (default 0.95).
    n_bootstrap : int
        Number of bootstrap resamples (default 10,000).
    correction : str
        Multiple comparisons correction for pairwise tests:
        'holm' (default), 'bonferroni', 'fdr_bh', or 'none'.
    spread_percentiles : tuple[float, float]
        Percentiles for the intrinsic variance band in the mean
        advantage result (default (10, 90)).
    failure_threshold : float, optional
        If provided, computes the fraction of inputs scoring below
        this value in robustness metrics.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. Shared across
        all analysis calls.

    Returns
    -------
    AnalysisResult
        An AnalysisBundle when evaluator_mode='aggregate', otherwise
        dict[str, AnalysisBundle] keyed by evaluator name.

    Raises
    ------
    ValueError
        If the benchmark has fewer than 2 prompt templates.
    NotImplementedError
        If the benchmark shape requires a pipeline not yet implemented
        (multi-model or cross-product inputs).
    """
    shape = _detect_shape(result)
    _validate_supported(shape)

    if rng is None:
        rng = np.random.default_rng()

    if evaluator_mode not in {"aggregate", "per_evaluator"}:
        raise ValueError(
            f"Unknown evaluator_mode '{evaluator_mode}'. "
            "Expected 'aggregate' or 'per_evaluator'."
        )

    if evaluator_mode == "aggregate":
        return _analyze_single(
            result=result,
            shape=shape,
            reference=reference,
            method=method,
            ci=ci,
            n_bootstrap=n_bootstrap,
            correction=correction,
            spread_percentiles=spread_percentiles,
            failure_threshold=failure_threshold,
            rng=rng,
        )

    evaluator_names = result.evaluator_names if not result.is_aggregated else ["score"]
    outputs: Dict[str, AnalysisBundle] = {}

    if result.is_aggregated:
        outputs["score"] = _analyze_single(
            result=result,
            shape=shape,
            reference=reference,
            method=method,
            ci=ci,
            n_bootstrap=n_bootstrap,
            correction=correction,
            spread_percentiles=spread_percentiles,
            failure_threshold=failure_threshold,
            rng=rng,
        )
        return outputs

    for evaluator_idx, evaluator_name in enumerate(evaluator_names):
        evaluator_result = BenchmarkResult(
            scores=result.scores[:, :, evaluator_idx],
            template_labels=result.template_labels,
            input_labels=result.input_labels,
            input_metadata=result.input_metadata,
            baseline_template=result.baseline_template,
        )
        outputs[evaluator_name] = _analyze_single(
            result=evaluator_result,
            shape=shape,
            reference=reference,
            method=method,
            ci=ci,
            n_bootstrap=n_bootstrap,
            correction=correction,
            spread_percentiles=spread_percentiles,
            failure_threshold=failure_threshold,
            rng=rng,
        )

    return outputs


def _analyze_single(
    result: BenchmarkResult,
    shape: BenchmarkShape,
    *,
    reference: str,
    method: Literal["bootstrap", "bca", "auto"],
    ci: float,
    n_bootstrap: int,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"],
    spread_percentiles: tuple[float, float],
    failure_threshold: Optional[float],
    rng: np.random.Generator,
) -> AnalysisBundle:

    scores_2d = result.get_2d_scores()
    labels = result.template_labels

    pairwise = all_pairwise(
        scores_2d, labels,
        method=method, ci=ci, n_bootstrap=n_bootstrap,
        correction=correction, rng=rng,
    )
    mean_adv = bootstrap_mean_advantage(
        scores_2d, labels,
        reference=reference,
        method=method, ci=ci, n_bootstrap=n_bootstrap,
        spread_percentiles=spread_percentiles, rng=rng,
    )
    robustness = robustness_metrics(
        scores_2d, labels,
        failure_threshold=failure_threshold,
    )
    rank_dist = bootstrap_ranks(
        scores_2d, labels,
        n_bootstrap=n_bootstrap, rng=rng,
    )

    return AnalysisBundle(
        benchmark=result,
        shape=shape,
        pairwise=pairwise,
        mean_advantage=mean_adv,
        robustness=robustness,
        rank_dist=rank_dist,
    )


def print_analysis_summary(
    analysis: Union[AnalysisBundle, Mapping[str, AnalysisBundle]],
    *,
    top_pairwise: int = 5,
    line_width: int = 41,
) -> None:
    """Print a concise console summary of analyze() results.

    Parameters
    ----------
    analysis : AnalysisBundle or Mapping[str, AnalysisBundle]
        Output from analyze(). Either one aggregate bundle, or a dict
        keyed by evaluator label when evaluator_mode='per_evaluator'.
    top_pairwise : int
        Number of pairwise comparisons (sorted by p-value) to print.
    line_width : int
        Width of ASCII interval plots for mean advantage and pairwise
        comparisons.
    """
    if isinstance(analysis, AnalysisBundle):
        _print_bundle_summary(
            analysis,
            top_pairwise=top_pairwise,
            line_width=line_width,
        )
        return

    for evaluator_name, bundle in analysis.items():
        print(f"=== Evaluator: {evaluator_name} ===")
        _print_bundle_summary(
            bundle,
            top_pairwise=top_pairwise,
            line_width=line_width,
        )
        print()


def _print_bundle_summary(
    bundle: AnalysisBundle,
    *,
    top_pairwise: int,
    line_width: int,
) -> None:
    template_col_width = 24
    pair_col_width = 32

    print("=== Analysis Summary ===")
    print(f"Shape: {bundle.shape}")
    print(
        f"Templates: {bundle.benchmark.n_templates} | "
        f"Inputs: {bundle.benchmark.n_inputs}"
    )
    print()

    print("--- Robustness ---")
    print(bundle.robustness.summary_table().to_string())
    print()

    print("--- Rank Probabilities ---")
    print(f"  {'Template':<24s} {'P(Best)':>9s} {'E[Rank]':>9s}")
    for i, label in enumerate(bundle.rank_dist.labels):
        print(
            f"  {label:<24s} "
            f"{bundle.rank_dist.p_best[i]:>8.1%} "
            f"{bundle.rank_dist.expected_ranks[i]:>8.2f}"
        )
    print()

    print(f"--- Mean Advantage (reference={bundle.mean_advantage.reference}) ---")
    low_p, high_p = bundle.mean_advantage.spread_percentiles
    ma = bundle.mean_advantage
    ma_max_abs = max(
        1e-12,
        float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            ma.mean_advantages,
                            ma.bootstrap_ci_low,
                            ma.bootstrap_ci_high,
                            ma.spread_low,
                            ma.spread_high,
                        ]
                    )
                )
            )
        ),
    )
    ma_low = -ma_max_abs
    ma_high = ma_max_abs
    print(f"  axis: [{ma_low:+.3f}, {ma_high:+.3f}]  (· spread, ─ CI, ● mean, │ zero)")
    print(
        f"  {'Template':<{template_col_width}s} {'Interval Plot':<{line_width}s} {'Mean':>8s} "
        f"{'CI Low':>9s} {'CI High':>9s} {'Spread Lo':>10s} {'Spread Hi':>10s}"
    )
    for i, label in enumerate(bundle.mean_advantage.labels):
        template_label = _truncate_label(label, template_col_width)
        line = _ascii_interval_line(
            mean=float(bundle.mean_advantage.mean_advantages[i]),
            ci_low=float(bundle.mean_advantage.bootstrap_ci_low[i]),
            ci_high=float(bundle.mean_advantage.bootstrap_ci_high[i]),
            spread_low=float(bundle.mean_advantage.spread_low[i]),
            spread_high=float(bundle.mean_advantage.spread_high[i]),
            axis_low=ma_low,
            axis_high=ma_high,
            width=line_width,
        )
        print(
            f"  {template_label:<{template_col_width}s} "
            f"{line:<{line_width}s} "
            f"{bundle.mean_advantage.mean_advantages[i]:>+7.3f} "
            f"{bundle.mean_advantage.bootstrap_ci_low[i]:>+8.3f} "
            f"{bundle.mean_advantage.bootstrap_ci_high[i]:>+8.3f} "
            f"{bundle.mean_advantage.spread_low[i]:>+9.3f} "
            f"{bundle.mean_advantage.spread_high[i]:>+9.3f}"
        )
    print(f"  spread percentiles = ({low_p:g}, {high_p:g})")
    print()

    print("--- Pairwise Comparisons (lowest p-value first) ---")
    pair_results = sorted(
        bundle.pairwise.results.values(),
        key=lambda r: (r.p_value, -abs(r.mean_diff)),
    )
    max_pairs = max(0, min(top_pairwise, len(pair_results)))
    if max_pairs > 0:
        pair_max_abs = max(
            1e-12,
            max(
                max(
                    abs(float(result.mean_diff)),
                    abs(float(result.ci_low)),
                    abs(float(result.ci_high)),
                    abs(float(result.mean_diff - result.std_diff)),
                    abs(float(result.mean_diff + result.std_diff)),
                )
                for result in pair_results[:max_pairs]
            ),
        )
        pair_low = -pair_max_abs
        pair_high = pair_max_abs
        print(
            f"  axis: [{pair_low:+.3f}, {pair_high:+.3f}]  "
            "(· ±1σ, ─ CI, ● mean, │ zero)"
        )
        print(
            f"  {'Pair':<{pair_col_width}s} {'Interval Plot':<{line_width}s} {'Mean':>8s} "
            f"{'CI Low':>9s} {'CI High':>9s} {'σ':>8s} {'p':>9s} {'sig':>5s}"
        )

    for result in pair_results[:max_pairs]:
        line = _ascii_interval_line(
            mean=float(result.mean_diff),
            ci_low=float(result.ci_low),
            ci_high=float(result.ci_high),
            spread_low=float(result.mean_diff - result.std_diff),
            spread_high=float(result.mean_diff + result.std_diff),
            axis_low=pair_low,
            axis_high=pair_high,
            width=line_width,
        )
        pair_label = _truncate_label(
            f"{result.template_a} vs {result.template_b}",
            pair_col_width,
        )
        print(
            f"  {pair_label:<{pair_col_width}s} "
            f"{line:<{line_width}s} "
            f"{result.mean_diff:+.4f} "
            f"{result.ci_low:+.4f} "
            f"{result.ci_high:+.4f} "
            f"{result.std_diff:>7.4f} "
            f"{result.p_value:>9.4g} "
            f"{str(result.significant):>5s}"
        )

    if max_pairs == 0:
        print("  (no pairwise comparisons)")


def _ascii_interval_line(
    *,
    mean: float,
    ci_low: float,
    ci_high: float,
    spread_low: float,
    spread_high: float,
    axis_low: float,
    axis_high: float,
    width: int,
) -> str:
    """Render a one-line ASCII interval plot with zero marker."""
    width = max(9, int(width))
    axis_low = float(axis_low)
    axis_high = float(axis_high)
    if axis_high <= axis_low:
        axis_low -= 1.0
        axis_high += 1.0

    def to_idx(x: float) -> int:
        x_clamped = min(max(float(x), axis_low), axis_high)
        pos = (x_clamped - axis_low) / (axis_high - axis_low)
        return int(round(pos * (width - 1)))

    lo_spread_idx = min(to_idx(spread_low), to_idx(spread_high))
    hi_spread_idx = max(to_idx(spread_low), to_idx(spread_high))
    lo_ci_idx = min(to_idx(ci_low), to_idx(ci_high))
    hi_ci_idx = max(to_idx(ci_low), to_idx(ci_high))
    mean_idx = to_idx(mean)

    chars = [" "] * width
    for idx in range(lo_spread_idx, hi_spread_idx + 1):
        chars[idx] = "·"
    for idx in range(lo_ci_idx, hi_ci_idx + 1):
        chars[idx] = "─"

    zero_idx = to_idx(0.0)
    chars[zero_idx] = "│"
    chars[mean_idx] = "●"

    return "".join(chars)


def _truncate_label(text: str, width: int) -> str:
    """Fit text into a fixed-width column with ellipsis when needed."""
    width = max(1, int(width))
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 1] + "…"


def _detect_shape(result: BenchmarkResult) -> BenchmarkShape:
    """Infer the structural shape of a BenchmarkResult.

    n_models is always 1 because BenchmarkResult does not yet encode
    multiple models. n_input_vars is inferred from input_labels: tuple
    labels indicate a cross-product of multiple variables.
    """
    n_models = 1  # BenchmarkResult does not yet encode multiple models

    n_prompts = result.n_templates

    n_evaluators = 1 if result.is_aggregated else result.scores.shape[2]

    # Tuple input_labels signal a cross-product of multiple variables.
    if result.input_labels and isinstance(result.input_labels[0], tuple):
        n_input_vars = len(result.input_labels[0])
    else:
        n_input_vars = 1

    return BenchmarkShape(
        n_models=n_models,
        n_prompts=n_prompts,
        n_input_vars=n_input_vars,
        n_evaluators=n_evaluators,
    )


def _validate_supported(shape: BenchmarkShape) -> None:
    """Raise if the shape is outside the currently supported pipeline."""
    if shape.n_prompts < 2:
        raise ValueError(
            f"analyze() requires at least 2 prompt templates; got {shape.n_prompts}. "
            "Add more templates to enable comparative analysis."
        )

    if shape.n_models > 1:
        raise NotImplementedError(
            f"Multi-model analysis (n_models={shape.n_models}) is not yet supported. "
            "Run analyze() separately for each model, or collapse scores to a "
            "single model before calling analyze()."
        )

    if shape.n_input_vars > 1:
        raise NotImplementedError(
            f"Cross-product input analysis (n_input_vars={shape.n_input_vars}) is "
            "not yet supported. Flatten the input space to a single variable "
            "(e.g., by joining variable values into one label) before calling "
            "analyze()."
        )
