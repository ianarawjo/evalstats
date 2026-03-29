"""High-level comparison API for prompt templates.

Provides ``compare_prompts()`` and ``compare_models()``, simple entry
points for the most common use cases: compare prompt variants within a
single model, or compare models while accounting for prompt sensitivity.

Internally, these helpers build ``BenchmarkResult`` or
``MultiModelBenchmark`` objects from dictionaries, run the full
``analyze()`` pipeline, and return lightweight report objects that surface
the most useful numbers without requiring knowledge of the underlying data
structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .core.types import BenchmarkResult, MultiModelBenchmark
from .core.types import CompareMethod
from .core.router import analyze, AnalysisBundle, MultiModelBundle
from .core.summary import print_analysis_summary, print_compare_summary
from .core.paired import PairwiseMatrix, PairedDiffResult
from .core.resampling import bayes_bootstrap_means_1d, smooth_bootstrap_means_1d, bootstrap_means_1d, bca_interval_1d, resolve_resampling_method, bayes_binary_ci_1d, wilson_ci_1d
from .config import get_alpha_ci


# ---------------------------------------------------------------------------
# Shared stats/report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EntityStats:
    """Descriptive and inferential statistics for one compared entity."""

    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float


@dataclass
class CompareReport:
    """Unified comparison report for prompts or models.

    Attributes
    ----------
    labels : list[str]
        Names of the compared entities in input order.
    entity_stats : dict[str, EntityStats]
        Per-entity descriptive statistics and bootstrapped absolute CIs.
        These CIs are single-sample (marginal) intervals, not pairwise
        difference intervals — use ``pairwise`` for the latter.
    unbeaten : list[str] or None
        Labels of entities that no other entity has been proven to beat
        (correction-adjusted p < alpha).  These are the candidates you
        cannot rule out with the current data — not necessarily the
        strongest performers.  ``None`` when no significant pairwise
        differences were found at all.
    pairwise : PairwiseMatrix
        All pairwise statistical comparison results, including effect sizes,
        CIs, and corrected p-values. Access via ``pairwise.get(a, b)``.
    full_analysis : AnalysisBundle or MultiModelBundle
        The full internal analysis object. Use ``full_summary()`` to print
        the complete analysis, or access fields directly for advanced use.
    alpha : float
        Significance threshold used for CIs and for significance testing, if any,
        which determines the ``unbeaten`` set and is used in the ``quick_summary()`` text.
    statistic : str
        Central-tendency statistic used (``'mean'`` or ``'median'``).
    method : str
        Resolved bootstrap/test method used.
    correction : str
        Multiple-comparisons correction applied to p-values.
    entity_name_singular / entity_name_plural : str
        Human-readable entity type labels (``'prompt'``/``'prompts'`` or
        ``'model'``/``'models'``).
    """

    labels: list[str]
    entity_stats: dict[str, EntityStats]
    unbeaten: Optional[list[str]]
    pairwise: PairwiseMatrix
    full_analysis: AnalysisBundle | MultiModelBundle
    alpha: float = 0.01
    statistic: Literal["mean", "median"] = "mean"
    method: str = "smooth_bootstrap"
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "fdr_bh"
    entity_name_singular: str = "prompt"
    entity_name_plural: str = "prompts"
    simultaneous_ci: bool = False

    @property
    def means(self) -> dict[str, float]:
        return {label: self.entity_stats[label].mean for label in self.labels}

    @property
    def prompt_stats(self) -> dict[str, EntityStats]:
        return self.entity_stats

    @property
    def model_stats(self) -> dict[str, EntityStats]:
        return self.entity_stats

    @property
    def significant(self) -> bool:
        """True when at least one significant pairwise difference was found."""
        return self.unbeaten is not None

    @property
    def winner(self) -> Optional[str]:
        """The sole unbeaten entity, or None when multiple are unbeaten or none are."""
        if self.unbeaten and len(self.unbeaten) == 1:
            return self.unbeaten[0]
        return None

    def quick_summary(self) -> str:
        """One-line summary suitable for logging or a paper methods section.

        Always includes the best entity's absolute mean and 95% CI so the
        number can be cited directly.
        """
        best_label = self._best_label()
        best_stats = self.entity_stats[best_label]
        stat_name = self.statistic
        best_stat = getattr(best_stats, stat_name)
        delta_name = f"Δ{stat_name}"
        correction_text = "uncorrected" if self.correction == "none" else f"{self.correction}-corrected"
        method_text = self.method
        n = len(self.labels)

        if n == 2:
            other = [label for label in self.labels if label != best_label][0]
            pair = self._best_pair()
            diff, ci_lo, ci_hi, p = pair.point_diff, pair.ci_low, pair.ci_high, pair.p_value
            abs_str = f"{stat_name}={best_stat:.3f}, 95% CI [{best_stats.ci_low:.3f}, {best_stats.ci_high:.3f}]"
            diff_str = f"{delta_name}={diff:+.3f}, CI [{ci_lo:.3f}, {ci_hi:.3f}], p={p:.4g}"
            if self.unbeaten is not None:
                return (
                    f"'{best_label}' is significantly better than '{other}' "
                    f"({abs_str}; {diff_str}, {correction_text}, {method_text})"
                )
            return (
                f"No significant difference between '{best_label}' and '{other}' "
                f"({diff_str}, {correction_text}, {method_text}; "
                f"'{best_label}' {abs_str})"
            )

        # n > 2
        abs_str = f"{stat_name}={best_stat:.3f}, 95% CI [{best_stats.ci_low:.3f}, {best_stats.ci_high:.3f}]"
        if self.unbeaten is not None:
            # Show comparison vs. runner-up (2nd-highest by mean).
            sorted_labels = sorted(
                self.labels,
                key=lambda l: getattr(self.entity_stats[l], stat_name),
                reverse=True,
            )
            runner_up = sorted_labels[1]
            pair = self.pairwise.get(best_label, runner_up)
            diff_str = (
                f"{delta_name} vs '{runner_up}'={pair.point_diff:+.3f}, "
                f"CI [{pair.ci_low:.3f}, {pair.ci_high:.3f}], p={pair.p_value:.4g}"
            )
            if len(self.unbeaten) == 1:
                return (
                    f"'{best_label}' is best {self.entity_name_singular} "
                    f"({abs_str}; {diff_str}, {correction_text}, {method_text})"
                )
            unbeaten_str = ", ".join(f"'{w}'" for w in self.unbeaten)
            return (
                f"Unbeaten {self.entity_name_singular}s ({unbeaten_str}); '{best_label}' leads "
                f"({abs_str}; {diff_str}, {correction_text}, {method_text})"
            )

        min_p = self._best_pair().p_value
        return (
            f"No significant differences between {self.entity_name_plural}; '{best_label}' leads "
            f"({abs_str}; min p={min_p:.4g}, {correction_text}, {method_text})"
        )

    def print_ci_table(self, *, sort_by: str = "input_order") -> None:
        """Print a compact table of mean, 95% CI, and contention status per entity.

        Gives a quick, readable overview of where each entity stands before
        diving into pairwise tests.  Use ``summary()`` for the full statistical
        breakdown including pairwise p-values and the leaderboard.

        Parameters
        ----------
        sort_by : str
            Row ordering: ``'input_order'`` (default, preserves the order
            the entities were passed in), ``'mean'`` (descending), or
            ``'label'`` (alphabetical).
        """
        n = len(self.labels)
        if sort_by == "mean":
            ordered = sorted(
                self.labels,
                key=lambda l: self.entity_stats[l].mean,
                reverse=True,
            )
        elif sort_by == "label":
            ordered = sorted(self.labels)
        elif sort_by == "input_order":
            ordered = list(self.labels)
        else:
            raise ValueError(
                f"Unknown sort_by: {sort_by!r}. "
                "Expected 'input_order', 'mean', or 'label'."
            )

        unbeaten_set = set(self.unbeaten) if self.unbeaten else None
        no_sig = self.unbeaten is None

        label_w = max(max(len(l) for l in self.labels),
                      len(self.entity_name_singular.capitalize()))
        entity_header = self.entity_name_singular.capitalize()
        ci_col = 22

        print(f"  {entity_header:<{label_w}}  {'Mean':>8}  {'95% CI':<{ci_col}}  Status")
        print("  " + "-" * (label_w + 8 + ci_col + 16))
        for label in ordered:
            s = self.entity_stats[label]
            ci_str = f"[{s.ci_low:.1%}, {s.ci_high:.1%}]"
            if no_sig:
                status = "—"
            elif label in unbeaten_set:
                status = "in contention"
            else:
                status = "outperformed"
            print(f"  {label:<{label_w}}  {s.mean:>7.1%}  {ci_str:<{ci_col}}  {status}")

    def print_pair(self, a: str, b: str) -> None:
        """Print the pairwise comparison summary between two entities.

        Convenience wrapper for ``report.pairwise.get(a, b).summary()``.

        Parameters
        ----------
        a, b : str
            Labels of the two entities to compare.  Order determines the
            sign convention of the mean gap (a − b).
        """
        self.pairwise.get(a, b).summary()

    def summary(self) -> None:
        """Print a focused summary scoped to the entity comparison level.

        Shows pairwise comparisons and the executive leaderboard only.
        For the full internal analysis use ``full_summary()`` instead.
        """
        print_compare_summary(self)

    def full_summary(self) -> None:
        """Print the complete internal analysis (full AnalysisBundle / MultiModelBundle output)."""
        print_analysis_summary(self.full_analysis)

    def print(self) -> None:
        """Alias for ``summary()``."""
        self.summary()

    def _best_label(self) -> str:
        return max(self.labels, key=lambda label: getattr(self.entity_stats[label], self.statistic))

    def _best_pair(self) -> PairedDiffResult:
        best = self._best_label()
        others = [label for label in self.labels if label != best]
        return min(
            (self.pairwise.get(best, other) for other in others),
            key=lambda result: result.p_value,
        )


def _compute_unbeaten(
    labels: list[str],
    pairwise: PairwiseMatrix,
    alpha: float,
) -> Optional[list[str]]:
    """Compute the set of unbeaten entities from directed significant-better relations.

    A directed edge i→j exists when i is significantly better than j
    (correction-adjusted p < alpha and positive point difference).
    Unbeaten entities are those with zero incoming edges (nothing beats them).
    Returns ``None`` when no significant edges exist at all.
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

    unbeaten = [label for label in labels if incoming[label] == 0]
    return unbeaten if unbeaten else None


def _normalize_compare_models_scores(
    scores: dict,
    template_labels: Optional[list[str]],
) -> tuple[list[str], list[np.ndarray], Optional[list[str]]]:
    labels = list(scores.keys())
    values = list(scores.values())

    if not values:
        return labels, [], template_labels

    nested_templates = all(isinstance(value, dict) for value in values)
    if any(isinstance(value, dict) for value in values) and not nested_templates:
        raise TypeError(
            "scores values must be consistently array-like or nested dicts of template scores; "
            "do not mix both forms."
        )

    if nested_templates:
        first_templates = list(values[0].keys())
        if template_labels is None:
            resolved_template_labels = first_templates
        else:
            if len(template_labels) != len(first_templates):
                raise ValueError(
                    "template_labels length must match the number of templates (N). "
                    f"Got {len(template_labels)} labels for N={len(first_templates)}."
                )
            if set(template_labels) != set(first_templates):
                raise ValueError(
                    "For nested dict input, template_labels must match the inner template keys. "
                    f"Expected {sorted(first_templates)}, got {sorted(template_labels)}."
                )
            resolved_template_labels = list(template_labels)

        expected_template_set = set(first_templates)
        arrays: list[np.ndarray] = []

        for model_label, model_templates in scores.items():
            model_template_set = set(model_templates.keys())
            if model_template_set != expected_template_set:
                raise ValueError(
                    "All nested model dicts must contain the same template keys. "
                    f"Expected {sorted(expected_template_set)}, got {sorted(model_template_set)} "
                    f"for model '{model_label}'."
                )

            per_template_arrays: list[np.ndarray] = []
            for template_label in resolved_template_labels:
                a = np.asarray(model_templates[template_label], dtype=np.float64)
                if a.ndim not in (1, 2):
                    raise ValueError(
                        f"Score array for model '{model_label}', template '{template_label}' "
                        f"has {a.ndim} dimensions. Expected 1-D (M inputs) or "
                        "2-D (M inputs, R runs)."
                    )
                per_template_arrays.append(a)

            arrays.append(np.stack(per_template_arrays, axis=0))

        return labels, arrays, resolved_template_labels

    arrays = []
    for label, arr in scores.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim not in (1, 2):
            raise ValueError(
                f"Score array for '{label}' has {a.ndim} dimensions. "
                "Expected 1-D (M inputs) or 2-D (M inputs, R runs). "
                "For multiple templates, use nested dict form: "
                "{model: {template: array}}."
            )

        if a.ndim == 1:
            a = a[np.newaxis, :]
        else:
            a = a[np.newaxis, :, :]

        arrays.append(a)

    return labels, arrays, template_labels


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare_prompts(
    scores: dict,
    *,
    alpha: float | None = None,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "fdr_bh",
    method: CompareMethod = "auto",
    statistic: Literal["mean", "median"] = "mean",
    rng: Optional[np.random.Generator] = None,
    simultaneous_ci: bool = False,
) -> CompareReport:
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
        Significance threshold for CIs and for declaring a winner (default 0.01).
        A prompt is named winner only if it beats at least one other prompt
        with a correction-adjusted p-value < alpha.
    n_bootstrap : int
        Bootstrap resamples (default 10,000).
    correction : str
        Multiple-comparisons correction: ``'fdr_bh'`` (default),
        ``'holm'``, ``'bonferroni'``, or ``'none'``.
    method : str
        Bootstrap variant: ``'auto'`` (default, selects ``'smooth_bootstrap'``),
        ``'bootstrap'`` (percentile), ``'bca'``, ``'bayes_bootstrap'``,
        ``'smooth_bootstrap'``, ``'bayes_binary'``, ``'wilson'``,
        ``'newcombe'``, ``'fisher_exact'``, or ``'sign_test'``.
    statistic : str
        Central-tendency statistic: ``'mean'`` (default) or ``'median'``.
    rng : np.random.Generator, optional
        Random-number generator for reproducibility.
    simultaneous_ci : bool
        When ``True``, pairwise CIs are simultaneous (family-wise) rather
        than marginal, using the studentized bootstrap max-T method.  Less
        conservative than Bonferroni because it exploits the positive
        correlation between comparisons sharing the same benchmark inputs.
        Only supported for bootstrap-based methods; silently ignored for
        ``'newcombe'``, ``'fisher_exact'``, and ``'bayes_binary'``.

    Returns
    -------
    CompareReport

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
    if alpha is None:
        alpha = get_alpha_ci()
    
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
        ci=1.0-alpha,
        rng=rng,
        simultaneous_ci=simultaneous_ci,
    )

    # ------------------------------------------------------------------
    # Per-template descriptive stats and bootstrapped CIs on the configured
    # statistic.
    # Cell means (averaged over runs) are used as the per-input observations
    # for a single-level bootstrap — appropriate for estimating uncertainty
    # in each template's absolute location independently.
    # ------------------------------------------------------------------
    scores_2d = benchmark.get_2d_scores()  # (N, M)
    # Use the resolved method from the analysis bundle (may be 'bayes_binary',
    # 'newcombe', or a bootstrap variant), falling back to re-resolution.
    resolved_method = full_analysis.resolved_method or resolve_resampling_method(method, M)
    # Newcombe/fisher_exact/sign_test are pairwise methods; use bayes_binary or wilson for single-sample CI.
    # Pairwise-only methods need a single-sample fallback for entity stats CIs.
    # newcombe/fisher_exact/sign_test → smooth_bootstrap; bayes_binary → wilson.
    if resolved_method in {"newcombe", "fisher_exact", "sign_test"}:
        resolved_method = "smooth_bootstrap"
    elif resolved_method == "bayes_binary":
        resolved_method = "wilson"

    rob = full_analysis.robustness  # RobustnessResult indexed parallel to labels

    entity_stats: dict[str, EntityStats] = {}
    for i, label in enumerate(labels):
        row = scores_2d[i]  # (M,) cell means
        point_est = float(np.nanmean(row)) if statistic == "mean" else float(np.nanmedian(row))

        if resolved_method == "wilson":
            ci_low, ci_high = wilson_ci_1d(row, alpha)
        elif resolved_method == "bayes_bootstrap":
            boot_stats = bayes_bootstrap_means_1d(row, n_bootstrap, rng, statistic=statistic)
            ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
            ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))
        elif resolved_method == "smooth_bootstrap":
            boot_stats = smooth_bootstrap_means_1d(row, n_bootstrap, rng, statistic=statistic)
            ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
            ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))
        else:
            boot_stats = bootstrap_means_1d(row, n_bootstrap, rng, statistic=statistic)
            if resolved_method == "bca":
                ci_low, ci_high = bca_interval_1d(row, point_est, boot_stats, alpha, statistic=statistic)
            else:
                ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
                ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))

        entity_stats[label] = EntityStats(
            mean=float(rob.mean[i]),
            median=float(rob.median[i]),
            std=float(rob.std[i]),
            ci_low=ci_low,
            ci_high=ci_high,
        )

    top_tier = _compute_unbeaten(labels, full_analysis.pairwise, alpha)

    return CompareReport(
        labels=labels,
        entity_stats=entity_stats,
        unbeaten=top_tier,
        pairwise=full_analysis.pairwise,
        full_analysis=full_analysis,
        alpha=alpha,
        statistic=statistic,
        method=full_analysis.resolved_method or resolved_method,
        correction=correction,
        entity_name_singular="prompt",
        entity_name_plural="prompts",
        simultaneous_ci=full_analysis.pairwise.simultaneous_ci,
    )


def compare_models(
    scores: dict,
    *,
    alpha: float | None = None,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "fdr_bh",
    method: CompareMethod = "auto",
    statistic: Literal["mean", "median"] = "mean",
    template_model_collapse: Literal["mean", "as_runs", "auto"] = "auto",
    template_labels: Optional[list[str]] = None,
    rng: Optional[np.random.Generator] = None,
    simultaneous_ci: bool = False,
) -> CompareReport:
    """Compare models while accounting for prompt-template sensitivity.

    Parameters
    ----------
    scores : dict
        Mapping from model label to scores in one of these forms:

        * ``{"model": array}`` where each array is:
            - **1-D** ``(M,)`` for a single implicit template, or
            - **2-D** ``(M, R)`` for a single implicit template with R runs.
        * ``{"model": {"template": array}}`` where each inner array is:
            - **1-D** ``(M,)``, or
            - **2-D** ``(M, R)`` with R runs.

        For the nested-dict form, all models must provide the same template
        keys. If ``template_labels`` is omitted, inner-key order from the first
        model is used.
    alpha : float
        Significance threshold for CIs and for significance testing, if any.
    n_bootstrap : int
        Bootstrap resamples.
    correction : str
        Multiple-comparisons correction.
    method : str
        Bootstrap variant.
    statistic : str
        Central-tendency statistic: ``'mean'`` or ``'median'``.
    template_model_collapse : {"mean", "as_runs", "auto"}
        How to combine model scores in the template-level analysis
        ("which template is best overall across models?").

        Recommended: leave this as ``"auto"``.

        * ``"auto"`` (default): picks the least surprising behavior.
            - Single-template inputs: uses ``"mean"`` to avoid creating a
                synthetic run axis from the model count.
            - Multi-template inputs: uses ``"as_runs"`` to preserve
                cross-model variability in template-level uncertainty.
            - Explicit binary-only methods (``"wilson"``, ``"newcombe"``,
                ``"fisher_exact"``, ``"bayes_binary"``): keeps ``"as_runs"``
                so binary structure is preserved.
        * ``"mean"``: average across models first.
            Use this when you want a simple pooled template summary.
        * ``"as_runs"``: treat models as run-like replicates.
            Use this when you want template-level intervals to include
            between-model variability.
    template_labels : list[str], optional
        Prompt-template labels. If omitted, defaults to
        ``template_0 ... template_{N-1}``.
    rng : np.random.Generator, optional
        Random-number generator for reproducibility.
    """
    if not isinstance(scores, dict):
        raise TypeError(
            "scores must be a dict mapping model labels to score arrays. "
            "Example: {'gpt-4.1': [[...], [...]], 'llama-3.3': [[...], [...]]}"
        )
    if len(scores) < 2:
        raise ValueError(
            f"compare_models requires at least 2 models; got {len(scores)}."
        )

    if rng is None:
        rng = np.random.default_rng()

    if alpha is None:
        alpha = get_alpha_ci()
    labels, arrays, normalized_template_labels = _normalize_compare_models_scores(
        scores,
        template_labels,
    )

    ndims = {a.ndim for a in arrays}
    if len(ndims) > 1:
        raise ValueError(
            "All score arrays must have the same number of dimensions. "
            "Got a mix of 2-D and 3-D arrays."
        )

    ndim = next(iter(ndims))
    ns = [a.shape[0] for a in arrays]
    ms = [a.shape[1] for a in arrays]
    if len(set(ns)) > 1:
        raise ValueError(
            "All score arrays must have the same number of templates (N). "
            f"Got: {dict(zip(labels, ns))}"
        )
    if len(set(ms)) > 1:
        raise ValueError(
            "All score arrays must have the same number of inputs (M). "
            f"Got: {dict(zip(labels, ms))}"
        )
    if ndim == 3:
        rs = [a.shape[2] for a in arrays]
        if len(set(rs)) > 1:
            raise ValueError(
                "All 3-D score arrays must have the same number of runs (R). "
                f"Got: {dict(zip(labels, rs))}"
            )

    n_templates = ns[0]
    n_inputs = ms[0]
    resolved_template_labels = (
        normalized_template_labels
        if normalized_template_labels is not None
        else [f"template_{i}" for i in range(n_templates)]
    )
    if len(resolved_template_labels) != n_templates:
        raise ValueError(
            "template_labels length must match the number of templates (N). "
            f"Got {len(resolved_template_labels)} labels for N={n_templates}."
        )

    if template_model_collapse == "auto":
        binary_only_methods = {"wilson", "newcombe", "fisher_exact", "bayes_binary"}
        resolved_template_model_collapse: Literal["mean", "as_runs"] = (
            "mean"
            if (n_templates == 1 and method not in binary_only_methods)
            else "as_runs"
        )
    elif template_model_collapse in {"mean", "as_runs"}:
        resolved_template_model_collapse = template_model_collapse
    else:
        raise ValueError(
            f"Unknown template_model_collapse '{template_model_collapse}'. "
            "Expected 'auto', 'mean', or 'as_runs'."
        )

    scores_arr = np.stack(arrays, axis=0)
    input_labels = [f"input_{i}" for i in range(n_inputs)]
    benchmark = MultiModelBenchmark(
        scores=scores_arr,
        model_labels=labels,
        template_labels=resolved_template_labels,
        input_labels=input_labels,
    )

    full_analysis = analyze(
        benchmark,
        method=method,
        n_bootstrap=n_bootstrap,
        correction=correction,
        statistic=statistic,
        ci=1.0-alpha,
        rng=rng,
        template_model_collapse=resolved_template_model_collapse,
        simultaneous_ci=simultaneous_ci,
    )
    if not isinstance(full_analysis, MultiModelBundle):
        raise RuntimeError("Expected multi-model analysis bundle from analyze().")

    model_analysis = full_analysis.model_level
    scores_2d = benchmark.get_model_mean_result().get_2d_scores()  # (P, M)
    resolved_method = model_analysis.resolved_method or resolve_resampling_method(method, n_inputs)
    # Pairwise-only methods need a single-sample fallback for entity stats CIs.
    # newcombe/fisher_exact/sign_test → smooth_bootstrap; bayes_binary → wilson.
    if resolved_method in {"newcombe", "fisher_exact", "sign_test"}:
        resolved_method = "smooth_bootstrap"
    elif resolved_method == "bayes_binary":
        resolved_method = "wilson"
    rob = model_analysis.robustness

    entity_stats: dict[str, EntityStats] = {}
    for i, label in enumerate(labels):
        row = scores_2d[i]
        point_est = float(np.nanmean(row)) if statistic == "mean" else float(np.nanmedian(row))

        if resolved_method == "wilson":
            ci_low, ci_high = wilson_ci_1d(row, alpha)
        elif resolved_method == "bayes_bootstrap":
            boot_stats = bayes_bootstrap_means_1d(row, n_bootstrap, rng, statistic=statistic)
            ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
            ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))
        elif resolved_method == "smooth_bootstrap":
            boot_stats = smooth_bootstrap_means_1d(row, n_bootstrap, rng, statistic=statistic)
            ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
            ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))
        else:
            boot_stats = bootstrap_means_1d(row, n_bootstrap, rng, statistic=statistic)
            if resolved_method == "bca":
                ci_low, ci_high = bca_interval_1d(row, point_est, boot_stats, alpha, statistic=statistic)
            else:
                ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
                ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))

        entity_stats[label] = EntityStats(
            mean=float(rob.mean[i]),
            median=float(rob.median[i]),
            std=float(rob.std[i]),
            ci_low=ci_low,
            ci_high=ci_high,
        )

    top_tier = _compute_unbeaten(labels, model_analysis.pairwise, alpha)

    return CompareReport(
        labels=labels,
        entity_stats=entity_stats,
        unbeaten=top_tier,
        pairwise=model_analysis.pairwise,
        full_analysis=full_analysis,
        alpha=alpha,
        statistic=statistic,
        method=full_analysis.model_level.resolved_method or resolved_method,
        correction=correction,
        entity_name_singular="model",
        entity_name_plural="models",
        simultaneous_ci=model_analysis.pairwise.simultaneous_ci,
    )
