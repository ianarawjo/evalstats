"""High-level comparison API for prompt templates.

Provides ``compare_prompts()``, a simple entry point for the most common
use case: you have score arrays for two or more prompt variants and want
to know whether any difference is statistically meaningful.

Internally, ``compare_prompts()`` builds a ``BenchmarkResult`` from the
provided dict, runs the full ``analyze()`` pipeline, and wraps the results
in a ``ComparePromptsReport`` that surfaces the most useful numbers without
requiring knowledge of the underlying data structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .core.types import BenchmarkResult
from .core.router import analyze, AnalysisBundle, print_analysis_summary
from .core.paired import PairwiseMatrix, PairedDiffResult
from .core.resampling import bootstrap_means_1d, bca_interval_1d, resolve_resampling_method


# ---------------------------------------------------------------------------
# Per-template stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class PromptStats:
    """Descriptive and inferential statistics for a single prompt.

    Attributes
    ----------
    mean : float
        Mean score across all inputs (and runs, when multi-run data is given).
    median : float
        Median score across inputs.
    std : float
        Standard deviation of scores across inputs (between-input variance on
        cell means).
    ci_low : float
        Lower bound of the bootstrapped confidence interval on the configured
        statistic.
    ci_high : float
        Upper bound of the bootstrapped confidence interval on the configured
        statistic.
    """

    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ComparePromptsReport:
    """Results from :func:`compare_prompts`.

    Attributes
    ----------
    labels : list[str]
        Prompt labels in the order they were given.
    prompt_stats : dict[str, PromptStats]
        Per-prompt descriptive statistics and bootstrapped confidence
        intervals on the requested statistic.
    pairwise_p_values : dict[tuple[str, str], dict[str, float or None]]
        Pairwise p-values keyed by prompt-label tuples, for example
        ``("baseline", "v2")`` -> ``{"p_boot": ..., "p_wilcoxon": ...}``.
        Keys are stored in the same orientation as ``pairwise.results``.
        Use :meth:`get_pairwise_p_values` for direction-agnostic access.
    winners : list[str] or None
        Top prompt(s) under pairwise significance at level ``alpha``.
        A prompt is included when no other prompt is significantly better
        than it. If all prompts are tied (no significant pairwise
        differences), this is ``None``.
    p_best : float
        The minimum correction-adjusted p-value for the numerically best
        prompt vs any other prompt. When ``winners`` is ``None`` this is
        still the minimum p-value among all pairwise comparisons, useful
        for understanding how close the result was.
    pairwise : PairwiseMatrix
        Full pairwise comparison matrix with correction-adjusted p-values and
        bootstrapped confidence intervals.  Access individual comparisons
        via ``pairwise.get("A", "B")``.
    full_analysis : AnalysisBundle
        The complete analysis bundle returned by ``analyze()``.  Useful
        for accessing rank distributions, point-advantage plots,
        robustness metrics, and seed-variance decompositions.
    alpha : float
        Significance level used to determine winners (default 0.05).
    statistic : str
        Statistic used for comparisons and winner selection (``'mean'`` or
        ``'median'``).
    correction : str
        Multiple-comparison correction used for pairwise p-values.
    """

    labels: list[str]
    prompt_stats: dict[str, PromptStats]
    pairwise_p_values: dict[tuple[str, str], dict[str, Optional[float]]]
    winners: Optional[list[str]]
    p_best: float
    pairwise: PairwiseMatrix
    full_analysis: AnalysisBundle
    alpha: float = 0.05
    statistic: Literal["mean", "median"] = "mean"
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm"

    # ------------------------------------------------------------------
    # Convenience accessors (backward compat with .means usage)
    # ------------------------------------------------------------------

    @property
    def means(self) -> dict[str, float]:
        """Mean score per prompt (shorthand for ``{l: prompt_stats[l].mean}``)."""
        return {l: self.prompt_stats[l].mean for l in self.labels}

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def significant(self) -> bool:
        """True if any prompt is significantly better than another."""
        return self.winners is not None

    @property
    def winner(self) -> Optional[str]:
        """Backward-compatible singular winner.

        Returns the label only when exactly one winner exists; otherwise
        returns ``None``.
        """
        if self.winners is None or len(self.winners) != 1:
            return None
        return self.winners[0]

    def quick_summary(self) -> str:
        """Human-readable one-line summary of the comparison result."""
        best_label = self._best_label()
        pair = self._best_pair()
        diff = pair.point_diff
        ci_lo, ci_hi = pair.ci_low, pair.ci_high
        p = pair.p_value
        n = len(self.labels)
        stat_name = self.statistic
        best_stat = getattr(self.prompt_stats[best_label], stat_name)
        delta_name = f"Δ{stat_name}"
        correction_text = (
            "uncorrected"
            if self.correction == "none"
            else f"{self.correction}-corrected"
        )

        if n == 2:
            other = [l for l in self.labels if l != best_label][0]
            if self.winners is not None:
                return (
                    f"'{best_label}' is significantly better than '{other}' "
                    f"({delta_name}={diff:+.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], "
                    f"p={p:.4g}, {correction_text})"
                )
            else:
                return (
                    f"No significant difference between '{best_label}' and '{other}' "
                    f"({delta_name}={diff:+.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], "
                    f"p={p:.4g}, {correction_text})"
                )
        else:
            if self.winners is not None:
                if len(self.winners) == 1:
                    winner_text = f"winner: '{self.winners[0]}'"
                else:
                    winner_text = "winners: " + ", ".join(f"'{w}'" for w in self.winners)
                return (
                    f"Top prompt set ({winner_text})"
                )
            else:
                return (
                    f"All prompts are tied under pairwise tests; '{best_label}' leads "
                    f"numerically ({stat_name}={best_stat:.3f}) "
                    f"(min p={p:.4g}, {correction_text})"
                )

    def summary(self) -> None:
        """Print the full analysis summary to stdout."""
        print_analysis_summary(self.full_analysis)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print(self) -> None:
        """Backward-compatible alias for :meth:`summary`."""
        self.summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _best_label(self) -> str:
        return max(self.labels, key=lambda l: getattr(self.prompt_stats[l], self.statistic))

    def get_pairwise_p_values(self, a: str, b: str) -> dict[str, Optional[float]]:
        """Return pairwise p-values for templates ``a`` and ``b``.

        The lookup is direction-agnostic: ``(a, b)`` and ``(b, a)`` return
        the same dictionary.
        """
        if (a, b) in self.pairwise_p_values:
            return self.pairwise_p_values[(a, b)]
        if (b, a) in self.pairwise_p_values:
            return self.pairwise_p_values[(b, a)]
        raise KeyError(f"No pairwise p-values found for ({a}, {b}).")

    def _best_pair(self) -> PairedDiffResult:
        """Pairwise result for the best prompt vs its most distinguishable
        competitor (lowest Holm-corrected p-value)."""
        best = self._best_label()
        others = [l for l in self.labels if l != best]
        return min(
            (self.pairwise.get(best, other) for other in others),
            key=lambda r: r.p_value,
        )


def _compute_winners(
    labels: list[str],
    pairwise: PairwiseMatrix,
    alpha: float,
) -> Optional[list[str]]:
    """Compute top-tier winners from directed significant-better relations.

    A directed edge i→j exists when i is significantly better than j
    (correction-adjusted p < alpha and positive point difference).
    Winners are labels with zero incoming edges. If there are no edges,
    all prompts are tied and ``None`` is returned.
    """
    incoming = {label: 0 for label in labels}
    edge_count = 0

    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            result = pairwise.get(a, b)
            if result.p_value < alpha:
                if result.point_diff > 0:
                    incoming[b] += 1
                    edge_count += 1
                elif result.point_diff < 0:
                    incoming[a] += 1
                    edge_count += 1

    if edge_count == 0:
        return None

    winners = [label for label in labels if incoming[label] == 0]
    return winners if winners else None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare_prompts(
    scores: dict,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    method: Literal["bootstrap", "bca", "auto"] = "auto",
    statistic: Literal["mean", "median"] = "mean",
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> ComparePromptsReport:
    """Compare prompt templates with bootstrapped statistical tests.

    A convenience wrapper around :func:`analyze` for the common case where
    you have one score array per prompt variant and want a quick answer to
    "is any prompt significantly better than the others?"

    Parameters
    ----------
    scores : dict[str, array-like]
        Mapping from prompt label to score array.  Each value can be:

        * **1-D** ``(M,)`` — one score per benchmark input (single run).
        * **2-D** ``(M, R)`` — R repeated runs per input.  R ≥ 3 activates
          the nested two-level bootstrap so that run-to-run stochasticity
          is propagated into confidence intervals.

        All arrays must share the same M (and R when 2-D).

    alpha : float
        Significance threshold for declaring a winner (default 0.05).
        A prompt is named winner only if it beats at least one other prompt
        with a correction-adjusted p-value < alpha.
    n_bootstrap : int
        Bootstrap resamples (default 10,000).
    correction : str
        Multiple-comparisons correction: ``'holm'`` (default),
        ``'bonferroni'``, ``'fdr_bh'``, or ``'none'``.
    method : str
        Bootstrap variant: ``'auto'`` (default, picks BCa for 15 ≤ M ≤ 200),
        ``'bootstrap'`` (percentile), or ``'bca'``.
    statistic : str
        Central-tendency statistic: ``'mean'`` (default) or ``'median'``.
    ci : float
        Confidence level for intervals (default 0.95).
    rng : np.random.Generator, optional
        Random-number generator for reproducibility.

    Returns
    -------
    ComparePromptsReport

    Examples
    --------
    Binary pass/fail, two prompts:

    >>> import promptstats as ps
    >>> report = ps.compare_prompts({
    ...     "baseline": [1, 1, 0, 1, 0],
    ...     "v2":       [1, 1, 1, 1, 0],
    ... })
    >>> print(report.quick_summary())
    >>> print(report.prompt_stats["baseline"].ci_low,
    ...       report.prompt_stats["baseline"].ci_high)

    Three-way comparison with continuous scores:

    >>> report = ps.compare_prompts({
    ...     "zero-shot":        [0.80, 0.90, 0.70, 0.85],
    ...     "few-shot":         [0.75, 0.88, 0.65, 0.80],
    ...     "chain-of-thought": [0.82, 0.91, 0.73, 0.87],
    ... })
    >>> report.summary()

    Multi-run (nested bootstrap activated when R ≥ 3):

    >>> report = ps.compare_prompts({
    ...     "baseline": [[0.80, 0.82, 0.79], [0.90, 0.88, 0.91]],
    ...     "v2":       [[0.85, 0.87, 0.84], [0.92, 0.90, 0.93]],
    ... })
    """
    if not isinstance(scores, dict):
        raise TypeError(
            "scores must be a dict mapping prompt labels to score arrays. "
            "Example: {'baseline': [0.8, 0.9, 0.7], 'v2': [0.85, 0.92, 0.71]}"
        )
    if len(scores) < 2:
        raise ValueError(
            f"compare_prompts requires at least 2 prompts; got {len(scores)}."
        )

    if rng is None:
        rng = np.random.default_rng()

    labels = list(scores.keys())
    arrays = []
    for label, arr in scores.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim not in (1, 2):
            raise ValueError(
                f"Score array for '{label}' has {a.ndim} dimensions. "
                "Expected 1-D (one score per input) or "
                "2-D of shape (M inputs, R runs)."
            )
        arrays.append(a)

    ndims = {a.ndim for a in arrays}
    if len(ndims) > 1:
        raise ValueError(
            "All score arrays must have the same number of dimensions. "
            "Got a mix of 1-D and 2-D arrays. "
            "Use 2-D arrays for all prompts when providing multiple runs."
        )

    ndim = next(iter(ndims))
    ms = [a.shape[0] for a in arrays]
    if len(set(ms)) > 1:
        raise ValueError(
            "All score arrays must have the same number of inputs (M). "
            f"Got: {dict(zip(labels, ms))}"
        )

    if ndim == 2:
        rs = [a.shape[1] for a in arrays]
        if len(set(rs)) > 1:
            raise ValueError(
                "All 2-D score arrays must have the same number of runs (R). "
                f"Got: {dict(zip(labels, rs))}"
            )

    M = ms[0]
    scores_arr = np.stack(arrays, axis=0)  # (N, M) or (N, M, R)
    input_labels = [f"input_{i}" for i in range(M)]

    benchmark = BenchmarkResult(
        scores=scores_arr,
        template_labels=labels,
        input_labels=input_labels,
    )

    full_analysis: AnalysisBundle = analyze(  # type: ignore[assignment]
        benchmark,
        method=method,
        n_bootstrap=n_bootstrap,
        correction=correction,
        statistic=statistic,
        ci=ci,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Per-template descriptive stats and bootstrapped CIs on the configured
    # statistic.
    # Cell means (averaged over runs) are used as the per-input observations
    # for a single-level bootstrap — appropriate for estimating uncertainty
    # in each template's absolute location independently.
    # ------------------------------------------------------------------
    scores_2d = benchmark.get_2d_scores()  # (N, M)
    alpha_ci = 1.0 - ci
    resolved_method = resolve_resampling_method(method, M)

    rob = full_analysis.robustness  # RobustnessResult indexed parallel to labels

    pairwise_p_values: dict[tuple[str, str], dict[str, Optional[float]]] = {
        (a, b): {
            "p_boot": float(result.p_value),
            "p_wilcoxon": (
                float(result.wilcoxon_p)
                if result.wilcoxon_p is not None
                else None
            ),
        }
        for (a, b), result in full_analysis.pairwise.results.items()
    }

    prompt_stats: dict[str, PromptStats] = {}
    for i, label in enumerate(labels):
        row = scores_2d[i]  # (M,) cell means
        point_est = float(np.nanmean(row)) if statistic == "mean" else float(np.nanmedian(row))

        boot_stats = bootstrap_means_1d(row, n_bootstrap, rng, statistic=statistic)

        if resolved_method == "bca":
            ci_low, ci_high = bca_interval_1d(row, point_est, boot_stats, alpha_ci, statistic=statistic)
        else:
            ci_low = float(np.percentile(boot_stats, 100 * alpha_ci / 2))
            ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha_ci / 2)))

        prompt_stats[label] = PromptStats(
            mean=float(rob.mean[i]),
            median=float(rob.median[i]),
            std=float(rob.std[i]),
            ci_low=ci_low,
            ci_high=ci_high,
        )

    # ------------------------------------------------------------------
    # Winners: top tier under pairwise significance.
    # ------------------------------------------------------------------
    best_label = max(labels, key=lambda l: getattr(prompt_stats[l], statistic))
    other_labels = [l for l in labels if l != best_label]
    best_pairs = [full_analysis.pairwise.get(best_label, other) for other in other_labels]
    p_best = float(min(r.p_value for r in best_pairs))
    winners = _compute_winners(labels, full_analysis.pairwise, alpha)

    return ComparePromptsReport(
        labels=labels,
        prompt_stats=prompt_stats,
        pairwise_p_values=pairwise_p_values,
        winners=winners,
        p_best=p_best,
        pairwise=full_analysis.pairwise,
        full_analysis=full_analysis,
        alpha=alpha,
        statistic=statistic,
        correction=correction,
    )
