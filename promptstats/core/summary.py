"""Console summary formatters for analyze() results.

All terminal output produced by print_analysis_summary() lives here,
keeping the analysis router (router.py) free of display concerns.
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Union

import numpy as np

from .bundles import AnalysisBundle, MultiModelBundle
from .paired import PairedDiffResult, PairwiseMatrix
from .variance import SeedVarianceResult
from ..config import get_alpha_ci

if TYPE_CHECKING:
    from ..compare import CompareReport


# ---------------------------------------------------------------------------
# ANSI color helpers (disabled when stdout is not a TTY)
# ---------------------------------------------------------------------------

_ANSI = sys.stdout.isatty()

_RESET         = "\033[0m"  if _ANSI else ""
_BOLD          = "\033[1m"  if _ANSI else ""
_DIM           = "\033[2m"  if _ANSI else ""
_GREEN         = "\033[32m" if _ANSI else ""
_YELLOW        = "\033[33m" if _ANSI else ""
_CYAN          = "\033[36m" if _ANSI else ""
_BRIGHT_GREEN  = "\033[92m" if _ANSI else ""
_BRIGHT_YELLOW = "\033[93m" if _ANSI else ""
_BRIGHT_CYAN   = "\033[96m" if _ANSI else ""
_BRIGHT_RED    = "\033[91m" if _ANSI else ""


def _p_best_color(p: float) -> str:
    """Return an opening ANSI code sequence for a P(Best) value.

    > 50%  → bold green (likely winner)
    < 5%   → dim (unlikely)
    else   → no color
    """
    if not _ANSI:
        return ""
    if p > 0.50:
        return _BOLD + _BRIGHT_GREEN
    if p < 0.05:
        return _DIM
    return ""


def _rank_method_label(bundle: "AnalysisBundle") -> str:
    """Return a short parenthetical note describing how ranks were computed.

    Mirrors the method-mapping in ``bootstrap_rank_distribution`` so the label
    reflects what was actually used, not the pairwise CI method.
    """
    method = (bundle.resolved_method or "bootstrap").lower()
    # Map pairwise methods that don't drive ranking to their bootstrap equivalent.
    if method in {"lmm", "permutation", "newcombe", "fisher", "sign", "bayes_binary"}:
        rank_method = "bootstrap"
    elif method == "bca":
        rank_method = "BCA bootstrap"
    elif method == "bayes_bootstrap":
        rank_method = "Bayes bootstrap"
    elif method == "smooth_bootstrap":
        rank_method = "smooth bootstrap"
    else:
        rank_method = "bootstrap"

    n = bundle.rank_dist.n_bootstrap
    statistic = bundle.point_advantage.statistic.lower()
    return f"{rank_method}, n={n}, ranked by {statistic}"


def _pairwise_p_value_label(test_method: str) -> str:
    """Return a human-readable p-value method label for pairwise summaries."""
    method = test_method.lower()
    if "newcombe" in method:
        return "McNemar exact"
    if "fisher exact" in method:
        return "Fisher exact"
    if "sign test" in method:
        return "paired sign test"
    if "wilcoxon" in method:
        return "Wilcoxon signed-rank"
    if "bootstrap" in method:
        return "bootstrap"
    return test_method


def _pairwise_display_pvalue(pair: PairedDiffResult) -> tuple[float, str]:
    """Choose the p-value shown in single-pair summaries.

    Default behavior is to display the Wilcoxon signed-rank p-value when
    available, while preserving exact-test paths (McNemar/Fisher/sign test)
    where ``pair.p_value`` is the canonical inferential p-value.
    """
    method = pair.test_method.lower()
    is_exact_path = (
        "newcombe" in method
        or "mcnemar" in method
        or "fisher exact" in method
        or "sign test" in method
    )
    if not is_exact_path and pair.wilcoxon_p is not None:
        return float(pair.wilcoxon_p), "Wilcoxon signed-rank"
    return float(pair.p_value), _pairwise_p_value_label(pair.test_method)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def print_analysis_summary(
    analysis: Union[
        AnalysisBundle,
        MultiModelBundle,
        Mapping[str, AnalysisBundle],
        Mapping[str, MultiModelBundle],
    ],
    *,
    top_pairwise: int = None,
    line_width: int = 41,
    pairwise_sort: Literal["grouped", "significance"] = "grouped",
) -> None:
    """Print a concise console summary of analyze() results."""
    if isinstance(analysis, MultiModelBundle):
        _print_multi_model_summary(
            analysis,
            top_pairwise=top_pairwise,
            line_width=line_width,
            pairwise_sort=pairwise_sort,
        )
        return

    if isinstance(analysis, AnalysisBundle):
        _print_bundle_summary(
            analysis,
            top_pairwise=top_pairwise,
            line_width=line_width,
            pairwise_sort=pairwise_sort,
        )
        return

    for evaluator_name, bundle in analysis.items():
        _print_loud_section(f"Evaluator: {evaluator_name}")
        if isinstance(bundle, MultiModelBundle):
            _print_multi_model_summary(
                bundle,
                top_pairwise=top_pairwise,
                line_width=line_width,
                pairwise_sort=pairwise_sort,
            )
        else:
            _print_bundle_summary(
                bundle,
                top_pairwise=top_pairwise,
                line_width=line_width,
                pairwise_sort=pairwise_sort,
            )
        print()


def print_brief_summary(
    analysis: Union[
        AnalysisBundle,
        MultiModelBundle,
        Mapping[str, AnalysisBundle],
        Mapping[str, MultiModelBundle],
    ],
) -> None:
    """Print a compact leaderboard-only summary of analyze() results.

    Shows just the executive leaderboard — entity names, significance groups,
    mean scores, CIs, and verdicts — without the full statistical breakdown
    (no ASCII advantage plots, no pairwise tables, no robustness section).
    Use ``print_analysis_summary()`` for the complete output.
    """
    if isinstance(analysis, MultiModelBundle):
        _print_brief_multi_model(analysis)
        return

    if isinstance(analysis, AnalysisBundle):
        _print_brief_bundle(analysis)
        return

    # Per-evaluator dict.
    for evaluator_name, bundle in analysis.items():
        _print_loud_section(f"Evaluator: {evaluator_name}")
        if isinstance(bundle, MultiModelBundle):
            _print_brief_multi_model(bundle)
        else:
            _print_brief_bundle(bundle)
        print()


def _print_brief_bundle(
    bundle: AnalysisBundle,
    *,
    item_singular: str = "prompt",
    item_plural: str = "prompts",
) -> None:
    """Brief output for a single-model AnalysisBundle."""
    n = bundle.benchmark.n_templates
    m = bundle.benchmark.n_inputs
    n_runs = bundle.benchmark.n_runs
    method = bundle.resolved_method or "auto"
    alpha = get_alpha_ci()
    ci_pct = int(round((1 - alpha) * 100))
    runs_str = f" × {n_runs} runs" if n_runs > 1 else ""
    print(
        f"{n} {item_plural} | {m} inputs{runs_str} | "
        f"method={method} | {ci_pct}% CI"
    )
    print()
    _print_executive_summary(bundle, item_singular=item_singular)


def _print_brief_multi_model(bundle: MultiModelBundle) -> None:
    """Brief output for a MultiModelBundle."""
    n_models = bundle.benchmark.n_models
    n_templates = bundle.benchmark.n_templates
    m = bundle.benchmark.n_inputs
    n_runs = bundle.benchmark.n_runs
    method = bundle.model_level.resolved_method or "auto"
    alpha = get_alpha_ci()
    ci_pct = int(round((1 - alpha) * 100))
    runs_str = f" × {n_runs} runs" if n_runs > 1 else ""
    print(
        f"{n_models} models × {n_templates} prompts | {m} inputs{runs_str} | "
        f"method={method} | {ci_pct}% CI"
    )
    best_model, best_template = bundle.best_pair
    print(f"Best pair by mean: model='{best_model}'  prompt='{best_template}'")
    print()
    _print_executive_summary(bundle.model_level, item_singular="model")
    if n_templates > 1:
        print()
        _print_executive_summary(bundle.template_level, item_singular="prompt")


def print_pairwise_summary(
    pair: PairedDiffResult,
    *,
    alpha: Optional[float] = None,
    correction: str = "",
    line_width: int = 50,
) -> None:
    """Print a focused, human-readable summary for a single pairwise comparison.

    Displays the gap estimate, an ASCII interval plot of the confidence
    interval, and a plain-language verdict so you can immediately see whether
    the difference is statistically distinguishable from zero.

    Parameters
    ----------
    pair : PairedDiffResult
        A single pairwise comparison result, e.g. from
        ``report.pairwise.get("Model A", "Model B")``.
    alpha : float
        Significance threshold (default 0.01).
    correction : str
        Name of the multiple-comparisons correction applied, e.g. ``'fdr_bh'``.
        Shown in the header when provided.
    line_width : int
        Width of the ASCII interval plot (default 50 characters).

    Examples
    --------
    >>> pair = report.pairwise.get("Model A", "Model B")
    >>> from promptstats.core.summary import print_pairwise_summary
    >>> print_pairwise_summary(pair)

    Or use the convenience method directly on the pair or the matrix:

    >>> pair.summary()
    >>> report.pairwise.summary("Model A", "Model B")
    """
    if alpha is None:
        alpha = get_alpha_ci()
    a, b = pair.template_a, pair.template_b
    stat_label = pair.statistic.capitalize()
    ci_pct = int(round((1.0 - alpha) * 100))
    display_p_value, p_method_label = _pairwise_display_pvalue(pair)

    _print_loud_section(f"Pairwise: {a} vs. {b}")

    corr_str = f"  |  correction: {correction}" if correction else ""
    print(f"  method: {pair.test_method}{corr_str}  |  N={pair.n_inputs} inputs")
    print()

    # --- Gap and CI ---
    # Detect percentage-scale values for nicer formatting.
    all_vals = [pair.point_diff, pair.ci_low, pair.ci_high]
    looks_pct = all(abs(v) <= 1.5 for v in all_vals)
    if looks_pct:
        def _fmt(v: float) -> str:
            return f"{v:+.1%}"
    else:
        def _fmt(v: float) -> str:
            return f"{v:+.4f}"

    print(
        f"  {stat_label} gap ({a} − {b}):  "
        f"{_BOLD}{_fmt(pair.point_diff)}{_RESET}"
    )
    print(f"  {ci_pct}% CI:  [{_fmt(pair.ci_low)}, {_fmt(pair.ci_high)}]")
    print()

    # --- ASCII interval plot ---
    max_abs = max(
        1e-12,
        abs(pair.point_diff),
        abs(pair.ci_low),
        abs(pair.ci_high),
        abs(pair.point_diff - pair.std_diff),
        abs(pair.point_diff + pair.std_diff),
    )
    axis_low, axis_high = -max_abs, max_abs
    line = _ascii_interval_line(
        mean=pair.point_diff,
        ci_low=pair.ci_low,
        ci_high=pair.ci_high,
        spread_low=pair.point_diff - pair.std_diff,
        spread_high=pair.point_diff + pair.std_diff,
        axis_low=axis_low,
        axis_high=axis_high,
        width=line_width,
    )
    print(
        f"  axis: [{axis_low:+.3f}, {axis_high:+.3f}]  "
        f"(· ±1σ spread, ─ {ci_pct}% CI, ● {pair.statistic}, │ zero)"
    )
    print(f"  {b} (<0) {line} (>0) {a}")
    print()

    # --- Effect size and p-value ---
    d = pair.rank_biserial
    p_str = _format_p_value(display_p_value)
    sig = display_p_value < alpha
    sig_color = _BRIGHT_GREEN if (sig and _ANSI) else (_YELLOW if _ANSI else "")
    sig_reset = _RESET if _ANSI else ""
    sig_label = "significant" if sig else "not significant"
    print(
        f"  Effect size (rank-biserial r):  {d:+.3f}   "
        f"p ({p_method_label}) = {sig_color}{p_str}{sig_reset}  ({sig_label})"
    )
    print()


def print_compare_summary(
    report: "CompareReport",
    *,
    top_pairwise: int = None,
    line_width: int = 41,
    p_value_method: Optional[str] = None,
    pairwise_sort: Literal["grouped", "significance"] = "grouped",
) -> None:
    """Print a focused summary for compare_prompts / compare_models results.

    Shows only the pairwise comparisons and the executive leaderboard —
    scoped to the entity level (prompts or models) that was compared.
    For the full internal analysis use ``report.full_summary()`` instead.

    Parameters
    ----------
    p_value_method : str or None
        Which p-value to show in pairwise comparisons.  ``'auto'`` (default)
        picks the method commensurate with the CI (bootstrap p for bootstrap
        paths, Wilcoxon for others).  Options: ``'boot'``, ``'wsr'``,
        ``'nem'``, or ``None`` to suppress p-values.
    pairwise_sort : {"grouped", "significance"}
        Row order for the pairwise table. ``"grouped"`` groups by the left
        item (stable, scan-friendly), while ``"significance"`` orders by
        p-value then absolute effect size.
    """
    n = len(report.labels)
    # Get the AnalysisBundle appropriate to the entity-level comparison.
    if isinstance(report.full_analysis, MultiModelBundle):
        bundle = report.full_analysis.model_level
    else:
        bundle = report.full_analysis  # type: ignore[assignment]

    n_inputs = bundle.benchmark.n_inputs
    ci_pct = int(round((1 - report.alpha) * 100))

    _print_loud_section(f"{report.entity_name_plural.capitalize()} Comparison")
    print(
        f"{n} {report.entity_name_plural} | "
        f"{n_inputs} inputs | "
        f"method={report.method} | "
        f"{ci_pct}% confidence intervals (CI)"
    )
    print()

    _print_mean_advantage(
        bundle,
        item_singular=report.entity_name_singular,
        line_width=line_width,
    )
    print()
    _print_pairwise_section(
        bundle,
        top_pairwise=top_pairwise,
        line_width=line_width,
        p_value_method=p_value_method,
        pairwise_sort=pairwise_sort,
    )
    print()
    _print_executive_summary(bundle, item_singular=report.entity_name_singular)


# ---------------------------------------------------------------------------
# Section headers
# ---------------------------------------------------------------------------

def _print_loud_section(title: str) -> None:
    heading = f" {title.upper()} "
    border = "=" * len(heading)
    print(f"{_BOLD}{_BRIGHT_CYAN}{border}{_RESET}")
    print(f"{_BOLD}{_BRIGHT_CYAN}{heading}{_RESET}")
    print(f"{_BOLD}{_BRIGHT_CYAN}{border}{_RESET}")


def _print_subsection(title: str) -> None:
    """Print a secondary `--- Title ---` header in bold cyan."""
    print(f"{_BOLD}{_CYAN}{title}{_RESET}")


# ---------------------------------------------------------------------------
# Instability helpers
# ---------------------------------------------------------------------------

def _instability_label(instability: float) -> str:
    """Map an instability score (mean per-cell seed std) to a plain-language description.

    Thresholds are calibrated for scores normalised to roughly [0, 1].
    ``instability`` is the mean over inputs of the within-cell seed std,
    so a value of 0.10 means scores typically shift by ±0.10 across runs.
    """
    if np.isnan(instability):
        return "—"
    if instability >= 0.35:
        return "near-random across runs"
    if instability >= 0.20:
        return "highly noisy across runs"
    if instability >= 0.10:
        return "moderately noisy across runs"
    if instability >= 0.05:
        return "mostly stable across runs"
    if instability >= 0.01:
        return "very stable across runs"
    return "effectively deterministic across runs"


def _instability_color(instability: float) -> str:
    """Return an ANSI color for an instability score (empty string when colors off)."""
    if not _ANSI or np.isnan(instability):
        return ""
    if instability >= 0.20:
        return _BRIGHT_RED
    if instability >= 0.10:
        return _YELLOW
    return ""  # neutral — no color applied


def _stability_emoji_label(instability: float) -> str:
    """Return an emoji + label string for a stability column in the executive summary."""
    if np.isnan(instability):
        return "—"
    if instability >= 0.35:
        return "💀 Near-random"
    if instability >= 0.20:
        return "Noisy"
    if instability >= 0.10:
        return "Variable"
    if instability >= 0.05:
        return "Mostly Stable"
    return "Stable"


# ---------------------------------------------------------------------------
# Multi-model summary
# ---------------------------------------------------------------------------

def _print_multi_model_summary(
    bundle: MultiModelBundle,
    *,
    top_pairwise: int = None,
    line_width: int,
    pairwise_sort: Literal["grouped", "significance"] = "grouped",
) -> None:
    _print_loud_section("Multi-Model Analysis Summary")
    print(f"Shape: {bundle.shape}")
    print(
        f"Models: {bundle.benchmark.n_models} | "
        f"Templates: {bundle.benchmark.n_templates} | "
        f"Inputs: {bundle.benchmark.n_inputs}"
        + (f" | Runs: {bundle.benchmark.n_runs}" if bundle.benchmark.n_runs > 1 else "")
    )
    model_str = ", ".join(bundle.benchmark.model_labels)
    print(f"Models: {model_str}")
    best_model, best_template = bundle.best_pair
    print(f"{_BOLD}Best pair by mean:{_RESET} model='{_BRIGHT_GREEN}{best_model}{_RESET}'  template='{_BRIGHT_GREEN}{best_template}{_RESET}'")
    print()

    _print_loud_section("Model-level comparison (across all prompts):")
    _print_bundle_summary(
        bundle.model_level,
        top_pairwise=top_pairwise,
        line_width=line_width,
        item_singular="model",
        item_plural="models",
        pairwise_sort=pairwise_sort,
    )

    print()
    _print_loud_section("Cross-model per-template comparison (models collapsed):")
    _print_bundle_summary(
        bundle.template_level,
        top_pairwise=top_pairwise,
        line_width=line_width,
        item_singular="template",
        item_plural="templates",
        pairwise_sort=pairwise_sort,
    )
    best_idx = int(np.argmax(bundle.template_level.robustness.mean))
    best_template = bundle.template_level.benchmark.template_labels[best_idx]
    
    # Instability across runs across models
    instability_rows = _collect_cross_model_seed_instability_rows(bundle)
    if instability_rows:
        _print_cross_model_seed_instability(bundle, rows=instability_rows)
        most_stable_model, instability, *_ = instability_rows[0]
        print(
            f"  {_BOLD}{_BRIGHT_GREEN}-> Most stable model across runs:{_RESET} "
            f"'{most_stable_model}' "
            f"(instability={instability:.4f}, {_instability_label(instability)})"
        )

    for model_label, model_bundle in bundle.per_model.items():
        print()
        _print_loud_section(f"Per-Model Summary: {model_label}")
        _print_bundle_summary(
            model_bundle,
            top_pairwise=top_pairwise,
            line_width=line_width,
            pairwise_sort=pairwise_sort,
        )

    print()
    _print_loud_section("Cross-Model Ranking (all model/template pairs)")
    _print_model_template_matrix(bundle)
    p_best = bundle.cross_model.rank_dist.p_best
    expected_ranks = bundle.cross_model.rank_dist.expected_ranks
    rank_labels = bundle.cross_model.rank_dist.labels
    rank_pairs = [_split_model_template_label(label) for label in rank_labels]
    rank_bar_width = 14
    n_ranked_items = len(rank_labels)
    model_col_width = min(24, max(len(model) for model, _ in rank_pairs) + 2)
    template_col_width = min(24, max(len(template) for _, template in rank_pairs) + 2)
    top_indices = np.argsort(-p_best)
    n_show = len(top_indices)
    _print_subsection(f"--- Rank Probabilities: All {n_show} by P(Best) ({_rank_method_label(bundle.cross_model)}) ---")
    print(
        f"  {'Model':<{model_col_width}s} "
        f"{'Template':<{template_col_width}s} "
        f"{'P(Best)':>9s} {'':<{rank_bar_width}s} "
        f"{'E[Rank]':>9s} {'':<{rank_bar_width}s}"
    )
    for idx in top_indices[:n_show]:
        model_label, template_label = rank_pairs[idx]
        model_label = _truncate_label(model_label, model_col_width)
        template_label = _truncate_label(template_label, template_col_width)
        p_best_i = float(p_best[idx])
        expected_rank_i = float(expected_ranks[idx])
        p_color = _p_best_color(p_best_i)
        p_reset = _RESET if p_color else ""
        p_str = f"{p_best_i:>8.1%} {_ratio_bar(p_best_i, width=rank_bar_width)}"
        print(
            f"  {model_label:<{model_col_width}s} "
            f"{template_label:<{template_col_width}s} "
            f"{p_color}{p_str}{p_reset} "
            f"{expected_rank_i:>8.2f} "
            f"{_rank_hump_lane(expected_rank_i, n_ranked_items, width=rank_bar_width)}"
        )

    ma = bundle.cross_model.point_advantage
    cross_rob = bundle.cross_model.robustness
    stat_label = ma.statistic.capitalize()
    low_p, high_p = ma.spread_percentiles

    # Reference value (absolute scale) for the │ marker.
    if ma.reference == "grand_mean":
        ref_val = float(np.mean(cross_rob.mean))
    else:
        try:
            ref_idx = ma.labels.index(ma.reference)
            ref_val = float(cross_rob.mean[ref_idx])
        except ValueError:
            ref_val = float(np.mean(cross_rob.mean))

    # Absolute spread bands.
    abs_spread_lows = ma.spread_low + ref_val
    abs_spread_highs = ma.spread_high + ref_val

    # Axis bounds cover means, marginal CIs, and spread bands.
    cross_means = cross_rob.mean
    cross_ci_lows = cross_rob.ci_low
    cross_ci_highs = cross_rob.ci_high
    all_vals = np.concatenate([cross_means, cross_ci_lows, cross_ci_highs,
                               abs_spread_lows, abs_spread_highs])
    val_range = float(np.max(all_vals) - np.min(all_vals))
    pad = max(val_range * 0.05, 1e-4)
    ma_low = float(np.min(all_vals)) - pad
    ma_high = float(np.max(all_vals)) + pad

    ref_label_str = ma.reference if ma.reference != "grand_mean" else "grand mean"
    print()
    _print_subsection(f"--- {stat_label} Performance: All {n_show} (marginal CIs) ---")
    print(
        f"  axis: [{ma_low:.3f}, {ma_high:.3f}]  "
        f"(· spread, ─ CI, ● {stat_label.lower()}, │ {ref_label_str})  "
        f"spread percentiles = ({low_p:g}, {high_p:g})"
    )
    print(
        f"  {'Model':<{model_col_width}s} "
        f"{'Template':<{template_col_width}s} "
        f"{'Interval Plot':<{line_width}s} "
        f"{stat_label:>8s} {'CI Low':>9s} {'CI High':>9s} {'Spread Lo':>10s} {'Spread Hi':>10s}"
    )

    # Build a label→index map for cross_rob (labels may differ from ma.labels order).
    cross_labels = list(cross_rob.labels)

    for idx in top_indices[:n_show]:
        pair_label = ma.labels[idx]
        model_label, template_label = _split_model_template_label(pair_label)
        model_label = _truncate_label(model_label, model_col_width)
        template_label = _truncate_label(template_label, template_col_width)
        try:
            rob_idx = cross_labels.index(pair_label)
        except ValueError:
            rob_idx = idx
        abs_mean = float(cross_means[rob_idx])
        abs_ci_low = float(cross_ci_lows[rob_idx])
        abs_ci_high = float(cross_ci_highs[rob_idx])
        abs_spread_low = float(abs_spread_lows[idx])
        abs_spread_high = float(abs_spread_highs[idx])
        line = _ascii_interval_line(
            mean=abs_mean,
            ci_low=abs_ci_low,
            ci_high=abs_ci_high,
            spread_low=abs_spread_low,
            spread_high=abs_spread_high,
            axis_low=ma_low,
            axis_high=ma_high,
            width=line_width,
            reference=ref_val,
        )
        print(
            f"  {model_label:<{model_col_width}s} "
            f"{template_label:<{template_col_width}s} "
            f"{line:<{line_width}s} "
            f"{abs_mean:>7.3f} "
            f"{abs_ci_low:>8.3f} "
            f"{abs_ci_high:>8.3f} "
            f"{abs_spread_low:>9.3f} "
            f"{abs_spread_high:>9.3f}"
        )

    print()
    _print_cross_model_executive_summary(bundle)
    print()


def _print_model_template_matrix(bundle: MultiModelBundle) -> None:
    """Print a model × template score matrix (mean ±std, heat encoding)."""
    model_labels = bundle.benchmark.model_labels
    template_labels = bundle.benchmark.template_labels

    # Build (model, template) -> mean from the flat cross_model bundle.
    # Labels are formatted as "model / template" by get_flat_result().
    cell_mean: dict[tuple[str, str], float] = {}
    for label, m in zip(
        bundle.cross_model.rank_dist.labels,
        bundle.cross_model.robustness.mean,
    ):
        parts = label.split(" / ", 1)
        if len(parts) == 2:
            cell_mean[(parts[0], parts[1])] = float(m)

    all_means = list(cell_mean.values())
    mn, mx = min(all_means), max(all_means)
    best_mean = mx
    heat_chars = "·░▒▓█"

    def _heat(v: float) -> str:
        if mx == mn:
            return heat_chars[-1]
        idx = min(int((v - mn) / (mx - mn) * len(heat_chars)), len(heat_chars) - 1)
        return heat_chars[idx]

    # Cell width: at least enough for "0.800 ▓*" (8 chars), but expand
    # when template labels are longer so header/data columns stay aligned.
    CELL_W = max(8, max(len(t) for t in template_labels))
    model_col_w = max(len(m) for m in model_labels)

    def _fmt_cell(mdl: str, t: str) -> str:
        if (mdl, t) not in cell_mean:
            return f"{'N/A':^{CELL_W}}"
        m = cell_mean[(mdl, t)]
        h = _heat(m)
        marker = "*" if m == best_mean else " "
        plain = f"{m:.3f} {h}{marker}".rjust(CELL_W)
        if m == best_mean:
            return f"{_BOLD}{_BRIGHT_GREEN}{plain}{_RESET}"
        return plain

    # Header
    header = f"  {'':>{model_col_w}}"
    for t in template_labels:
        header += f"  {t:^{CELL_W}}"
    print(header)

    # Data rows
    div = "  " + "─" * max(1, len(header) - 2)
    print(div)
    for mdl in model_labels:
        row = f"  {mdl:>{model_col_w}}"
        for t in template_labels:
            row += f"  {_fmt_cell(mdl, t)}"
        print(row)

    # Footer
    print(div)
    print(f"  * = best pair by mean  |  heat: · (low) → █ (high), range [{mn:.3f}, {mx:.3f}]")
    print()


def _print_cross_model_executive_summary(bundle: MultiModelBundle) -> None:
    """Print executive leaderboard for cross-model (model/template) pairs."""
    cross = bundle.cross_model
    labels = list(cross.rank_dist.labels)
    n = len(labels)
    if n < 2:
        return

    means = cross.robustness.mean
    sort_idx = list(np.argsort(-means))
    labels_sorted = [labels[i] for i in sort_idx]
    label_to_group = _assign_significance_groups(cross.pairwise, labels_sorted)

    split_pairs = [_split_model_template_label(label) for label in labels]
    model_w = min(28, max(10, max(len(m) for m, _ in split_pairs) + 2))
    template_w = min(28, max(12, max(len(t) for _, t in split_pairs) + 2))
    grp_w = 4
    mean_w = 6
    ci_w = 15

    _print_subsection("--- Executive Summary (Cross-model pair leaderboard) ---")
    _cross_ci_header = (
        "Wilson CI" if cross.point_advantage.n_bootstrap == 0 else "CI"
    )
    header = (
        f"  {'Model':<{model_w}s}"
        f"  {'Template':<{template_w}s}"
        f"  {'Grp':^{grp_w}s}"
        f"  {'Mean':>{mean_w}s}"
        f"  {_cross_ci_header:<{ci_w}s}"
        "  Verdict"
    )
    sep = "  " + "─" * (len(header) - 2)
    print(header)
    print(sep)

    for label in labels_sorted:
        orig_idx = labels.index(label)
        mean_val = float(means[orig_idx])
        model_label, template_label = _split_model_template_label(label)

        ci_lo = float(cross.robustness.ci_low[orig_idx])
        ci_hi = float(cross.robustness.ci_high[orig_idx])
        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"

        group = label_to_group.get(label, "?")
        verdict = _exec_verdict(label, label_to_group, labels_sorted)

        plain_model = f"{_truncate_label(model_label, model_w):<{model_w}s}"
        plain_template = f"{_truncate_label(template_label, template_w):<{template_w}s}"
        plain_grp = f"{group:^{grp_w}s}"
        if group == "#1" and _ANSI:
            model_str = f"{_BOLD}{_BRIGHT_GREEN}{plain_model}{_RESET}"
            template_str = f"{_BOLD}{_BRIGHT_GREEN}{plain_template}{_RESET}"
            grp_str = f"{_BOLD}{_BRIGHT_GREEN}{plain_grp}{_RESET}"
            verdict_str = f"{_BRIGHT_GREEN}{verdict}{_RESET}"
        else:
            model_str = plain_model
            template_str = plain_template
            grp_str = plain_grp
            verdict_str = verdict

        print(
            f"  {model_str}"
            f"  {template_str}"
            f"  {grp_str}"
            f"  {mean_val:>{mean_w}.3f}"
            f"  {ci_str:<{ci_w}s}"
            f"  {verdict_str}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Single-model bundle summary
# ---------------------------------------------------------------------------

def _print_pairwise_section(
    bundle: "AnalysisBundle",
    *,
    top_pairwise: int = None,
    line_width: int,
    sort: bool = True,
    p_value_method: Optional[str] = None,
    pairwise_sort: Literal["grouped", "significance"] = "grouped",
) -> None:
    """Print the pairwise comparisons block for an AnalysisBundle.

    Extracted so it can be reused by both the full bundle summary and the
    focused CompareReport summary without duplicating code.

    Parameters
    ----------
    p_value_method : str or None
        Which p-value column to show.  ``'auto'`` (default) picks the method
        most commensurate with the CI: bootstrap p-value for bootstrap CI
        paths, exact-test p-value for newcombe/fisher/sign paths, and
        Wilcoxon signed-rank for LMM/other paths.  Explicit choices:
        ``'boot'`` (result.p_value), ``'wsr'`` (Wilcoxon signed-rank),
        ``'nem'`` (Nemenyi post-hoc).  Pass ``None`` to suppress p-values.
    pairwise_sort : {"grouped", "significance"}
        Sorting strategy for pairwise rows. ``"grouped"`` keeps a stable
        left-item grouping, while ``"significance"`` sorts by p-value then
        absolute effect size.
    """
    pair_item_col_width = 24
    pair_stat_col_width = 8
    pair_ci_col_width = 9
    pair_sigma_col_width = 8

    # Determine statistic label from the first result (all share the same statistic).
    first_result = next(iter(bundle.pairwise.results.values()), None)
    pair_stat_label = first_result.statistic.capitalize() if first_result else "Mean"

    # Detect CI method family for auto p-value selection.
    is_newcombe_pairwise = (
        first_result is not None
        and "newcombe" in first_result.test_method.lower()
        and "fisher" not in first_result.test_method.lower()
    )
    is_fisher_pairwise = (
        first_result is not None
        and "fisher exact" in first_result.test_method.lower()
    )
    is_sign_pairwise = (
        first_result is not None
        and "sign test" in first_result.test_method.lower()
    )
    is_bootstrap_path = (
        first_result is not None
        and "bootstrap" in first_result.test_method.lower()
    )

    # Whether simultaneous max-T CIs were used (affects p-value source for bootstrap paths).
    using_max_t = bundle.pairwise.simultaneous_ci_method == "max_t"

    # Resolve the effective p-value source and column header.
    if p_value_method == "auto":
        if is_newcombe_pairwise:
            eff_p_source, p_col_header = "boot", "p (McNemar)"
        elif is_fisher_pairwise:
            eff_p_source, p_col_header = "boot", "p (Fisher)"
        elif is_sign_pairwise:
            eff_p_source, p_col_header = "boot", "p (sign)"
        elif is_bootstrap_path:
            eff_p_source = "max_t" if using_max_t else "boot"
            p_col_header = "p (boot)"
        else:
            eff_p_source, p_col_header = "wsr", "p (wsr)"
    elif p_value_method == "boot":
        eff_p_source = "max_t" if (using_max_t and is_bootstrap_path) else "boot"
        p_col_header = "p (boot)"
    elif p_value_method == "wsr":
        eff_p_source, p_col_header = "wsr", "p (wsr)"
    elif p_value_method == "nem":
        eff_p_source, p_col_header = "nem", "p (nem)"
    else:  # None
        eff_p_source, p_col_header = None, None

    pair_p_col_width = max(10, len(p_col_header)) if p_col_header else 0

    _pairwise_header_method = first_result.test_method
    corr = bundle.pairwise.correction_method
    if eff_p_source is not None and corr and corr != "none":
        _pairwise_header_method += f" ({corr}-corrected p-values)"
    sim_ci_method = bundle.pairwise.simultaneous_ci_method
    if sim_ci_method == "max_t":
        _pairwise_header_method += " (simultaneous CIs computed with max-T)"
    elif sim_ci_method == "bonferroni":
        _pairwise_header_method += " (simultaneous CIs computed with Bonferroni)"
    _print_subsection(f"--- Pairwise Comparisons ({_pairwise_header_method}) ---")
    pair_results = list(bundle.pairwise.results.values())

    # Canonical left/right ordering based on expected-rank order keeps rows
    # readable by preventing arbitrary A/B flips between adjacent rows.
    rank_order = {
        label: idx
        for idx, (_, label) in enumerate(
            sorted(
                zip(bundle.rank_dist.expected_ranks, bundle.rank_dist.labels),
                key=lambda item: (float(item[0]), item[1]),
            )
        )
    }

    if pair_results:
        max_label_len = max(
            max(len(r.template_a), len(r.template_b)) for r in pair_results
        )
        pair_item_col_width = min(30, max(12, max_label_len + 2))

    normalized_rows = []
    for result in pair_results:
        a = result.template_a
        b = result.template_b
        pos_a = rank_order.get(a, len(rank_order))
        pos_b = rank_order.get(b, len(rank_order))
        swap = (pos_a > pos_b) or (pos_a == pos_b and a > b)

        if swap:
            left_item = b
            right_item = a
            point_diff = -float(result.point_diff)
            ci_low = -float(result.ci_high)
            ci_high = -float(result.ci_low)
            rank_biserial = -float(result.rank_biserial)
            left_pos = pos_b
            right_pos = pos_a
        else:
            left_item = a
            right_item = b
            point_diff = float(result.point_diff)
            ci_low = float(result.ci_low)
            ci_high = float(result.ci_high)
            rank_biserial = float(result.rank_biserial)
            left_pos = pos_a
            right_pos = pos_b

        normalized_rows.append(
            {
                "left": left_item,
                "right": right_item,
                "left_pos": left_pos,
                "right_pos": right_pos,
                "point_diff": point_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "std_diff": float(result.std_diff),
                "rank_biserial": rank_biserial,
                "p_value": result.p_value,
                "wilcoxon_p": result.wilcoxon_p,
            }
        )

    if pairwise_sort not in {"grouped", "significance"}:
        raise ValueError("pairwise_sort must be 'grouped' or 'significance'.")

    if sort:
        if pairwise_sort == "grouped":
            normalized_rows = sorted(
                normalized_rows,
                key=lambda row: (
                    row["left_pos"],
                    row["right_pos"],
                    row["p_value"],
                    -abs(row["point_diff"]),
                ),
            )
        else:
            normalized_rows = sorted(
                normalized_rows,
                key=lambda row: (
                    row["p_value"],
                    -abs(row["point_diff"]),
                    row["left_pos"],
                    row["right_pos"],
                ),
            )
    # By default, print all pairs unless top_pairwise is set.
    if top_pairwise is None:
        max_pairs = len(normalized_rows)
    else:
        max_pairs = max(0, min(top_pairwise, len(normalized_rows)))

    # Friedman omnibus line (printed before the interval plot when pairs exist).
    if max_pairs > 0 and bundle.pairwise.friedman is not None:
        fr = bundle.pairwise.friedman
        fr_p_str = _format_p_value(fr.p_value)
        fr_p_color = _BRIGHT_GREEN if fr.p_value <= 0.05 else _YELLOW
        print(f"  Friedman omnibus: χ²({fr.df}) = {fr.statistic:.3f}, p = {fr_p_color}{fr_p_str}{_RESET}")
        if fr.p_value > 0.05:
            print(f"  {_YELLOW}[!] Friedman p > 0.05: no significant omnibus effect — treat pairwise results with caution.{_RESET}")

    if max_pairs > 0:
        pair_max_abs = max(
            1e-12,
            max(
                max(
                    abs(float(row["point_diff"])),
                    abs(float(row["ci_low"])),
                    abs(float(row["ci_high"])),
                    abs(float(row["point_diff"] - row["std_diff"])),
                    abs(float(row["point_diff"] + row["std_diff"])),
                )
                for row in normalized_rows[:max_pairs]
            ),
        )
        pair_low = -pair_max_abs
        pair_high = pair_max_abs
        print(
            f"  legend: (· ±1σ, ─ CI, ● {pair_stat_label.lower()}, │ zero)    "
            f"axis: [{pair_low:+.3f}, {pair_high:+.3f}]    "
            "effect: Left - Right"
        )
        header = (
            f"  {'Left':<{pair_item_col_width}s} {'Right':<{pair_item_col_width}s} "
            f"{'Interval Plot':<{line_width}s} "
            f"{pair_stat_label:>{pair_stat_col_width}s} "
            f"{'CI Low':>{pair_ci_col_width}s} {'CI High':>{pair_ci_col_width}s} "
            f"{'ES':>{pair_sigma_col_width}s}"
        )
        if p_col_header:
            header += f" {p_col_header:>{pair_p_col_width}s}"
        print(header)

    for row_data in normalized_rows[:max_pairs]:
        line = _ascii_interval_line(
            mean=float(row_data["point_diff"]),
            ci_low=float(row_data["ci_low"]),
            ci_high=float(row_data["ci_high"]),
            spread_low=float(row_data["point_diff"] - row_data["std_diff"]),
            spread_high=float(row_data["point_diff"] + row_data["std_diff"]),
            axis_low=pair_low,
            axis_high=pair_high,
            width=line_width,
        )
        left_label = _truncate_label(str(row_data["left"]), pair_item_col_width)
        right_label = _truncate_label(str(row_data["right"]), pair_item_col_width)
        d_val = float(row_data["rank_biserial"])
        d_str = f"{d_val:>{pair_sigma_col_width}.3f}"
        row = (
            f"  {left_label:<{pair_item_col_width}s} "
            f"{right_label:<{pair_item_col_width}s} "
            f"{line:<{line_width}s} "
            f"{float(row_data['point_diff']):+{pair_stat_col_width}.4f} "
            f"{float(row_data['ci_low']):+{pair_ci_col_width}.4f} "
            f"{float(row_data['ci_high']):+{pair_ci_col_width}.4f} "
            f"{d_str}"
        )
        if eff_p_source in {"max_t", "boot"}:
            p_val = row_data["p_value"]
        elif eff_p_source == "wsr":
            p_val = row_data["wilcoxon_p"]
        elif eff_p_source == "nem":
            p_val = (
                bundle.pairwise.friedman.get_nemenyi_p(str(row_data["left"]), str(row_data["right"]))
                if bundle.pairwise.friedman is not None else None
            )
        else:
            p_val = None
        if eff_p_source is not None:
            row += f" {_format_p_value(p_val):>{pair_p_col_width}s}"
        print(row)

    if max_pairs == 0:
        print("  (no pairwise comparisons)")
    elif max_pairs > 0:
        print(f"{_DIM}  ES = Effect Size (r_rb) = rank biserial correlation (small≈0.1, medium≈0.3, large≈0.5){_RESET}")
        if eff_p_source in {"max_t", "boot"}:
            if is_newcombe_pairwise:
                print(f"  {p_col_header} = McNemar exact test (two-sided, uncorrected)")
            elif is_fisher_pairwise:
                print(f"  {p_col_header} = Fisher's exact test (two-sided, uncorrected)")
            elif is_sign_pairwise:
                print(f"  {p_col_header} = paired sign test (two-sided exact, ties dropped, uncorrected)")
            elif eff_p_source == "max_t":
                print(f"  {p_col_header} = max-T bootstrap p-value (FWER-controlled, commensurate with simultaneous CIs)")
            else:
                print(f"  {p_col_header} = bootstrap p-value ({bundle.pairwise.correction_method}-corrected)")
        elif eff_p_source == "wsr":
            print(f"  {p_col_header} = Wilcoxon signed-rank ({bundle.pairwise.correction_method}-corrected)")
        elif eff_p_source == "nem":
            print(f"  {p_col_header} = Nemenyi post-hoc (Friedman-based, FWER-controlled)")
        if eff_p_source is not None:
            print("  stars: * p<0.01, ** p<0.001, *** p<0.0001")
        print()
        labels_sorted = [
            label
            for _, label in sorted(
                zip(bundle.rank_dist.expected_ranks, bundle.rank_dist.labels),
                key=lambda item: (float(item[0]), item[1]),
            )
        ]
        _print_critical_difference_groups(
            bundle.pairwise,
            labels_sorted=labels_sorted,
            p_source="bootstrap",
        )


def _print_mean_advantage(
    bundle: "AnalysisBundle",
    *,
    item_singular: str = "template",
    line_width: int,
    template_col_width: int = 24,
) -> None:
    """Print the Mean/Median performance interval-plot table for an AnalysisBundle.

    Shows each entity's absolute mean with marginal bootstrap CIs (single-sample,
    independent per entity) and intrinsic spread bands.  A reference line marks
    the grand mean (or the specified reference entity) for visual comparison.
    """
    item_singular_title = item_singular.capitalize()
    ma = bundle.point_advantage
    stat_label = ma.statistic.capitalize()
    rob = bundle.robustness
    low_p, high_p = ma.spread_percentiles

    # Resolve the reference value (absolute scale) for the │ marker.
    if ma.reference == "grand_mean":
        ref_val = float(np.mean(rob.mean))
    else:
        try:
            ref_idx = ma.labels.index(ma.reference)
            ref_val = float(rob.mean[ref_idx])
        except ValueError:
            ref_val = float(np.mean(rob.mean))

    # Per-entity absolute values.
    abs_means = np.array([float(rob.mean[i]) for i in range(len(ma.labels))])

    abs_ci_lows = rob.ci_low
    abs_ci_highs = rob.ci_high

    # Absolute spread bands (percentiles of per-input advantages, shifted).
    abs_spread_lows = ma.spread_low + ref_val
    abs_spread_highs = ma.spread_high + ref_val

    # Axis bounds: cover means, CIs, and spread bands.
    all_vals = np.concatenate([abs_means, abs_ci_lows, abs_ci_highs,
                               abs_spread_lows, abs_spread_highs])
    val_range = float(np.max(all_vals) - np.min(all_vals))
    pad = max(val_range * 0.05, 1e-4)
    ma_low = float(np.min(all_vals)) - pad
    ma_high = float(np.max(all_vals)) + pad

    ci_note = "Wilson CIs" if ma.n_bootstrap == 0 else "marginal bootstrap CIs"
    _print_subsection(f"--- {stat_label} Performance ({ci_note}) ---")
    ref_label = ma.reference if ma.reference != "grand_mean" else "grand mean"
    print(
        f"  axis: [{ma_low:.3f}, {ma_high:.3f}]"
        f"  (· spread, ─ CI, ● {stat_label.lower()}, │ {ref_label})"
        f"  spread percentiles = ({low_p:g}, {high_p:g})"
    )
    print(
        f"  {item_singular_title:<{template_col_width}s} {'Interval Plot':<{line_width}s} {stat_label:>8s} "
        f"{'CI Low':>9s} {'CI High':>9s} {'Spread Lo':>10s} {'Spread Hi':>10s}"
    )
    for i, label in enumerate(ma.labels):
        template_label = _truncate_label(label, template_col_width)
        line = _ascii_interval_line(
            mean=abs_means[i],
            ci_low=float(abs_ci_lows[i]),
            ci_high=float(abs_ci_highs[i]),
            spread_low=float(abs_spread_lows[i]),
            spread_high=float(abs_spread_highs[i]),
            axis_low=ma_low,
            axis_high=ma_high,
            width=line_width,
            reference=ref_val,
        )
        print(
            f"  {template_label:<{template_col_width}s} "
            f"{line:<{line_width}s} "
            f"{abs_means[i]:>7.3f} "
            f"{float(abs_ci_lows[i]):>8.3f} "
            f"{float(abs_ci_highs[i]):>8.3f} "
            f"{float(abs_spread_lows[i]):>9.3f} "
            f"{float(abs_spread_highs[i]):>9.3f}"
        )


def _print_bundle_summary(
    bundle: AnalysisBundle,
    *,
    top_pairwise: int = None,
    line_width: int,
    item_singular: str = "template",
    item_plural: str = "templates",
    p_value_method: Optional[str] = None,
    pairwise_sort: Literal["grouped", "significance"] = "grouped",
) -> None:
    template_col_width = 24

    print(f"Shape: {bundle.shape}")
    n_runs = bundle.benchmark.n_runs
    item_singular_title = item_singular.capitalize()
    item_plural_title = item_plural.capitalize()
    print(
        f"{item_plural_title}: {bundle.benchmark.n_templates} | "
        f"Inputs: {bundle.benchmark.n_inputs}"
        + (f" | Runs: {n_runs}" if n_runs > 1 else "")
    )
    print()

    _print_subsection("--- Robustness ---")
    print(bundle.robustness.summary_table().to_string())
    print()

    _print_subsection(f"--- Rank Probabilities ({_rank_method_label(bundle)}) ---")
    max_rank_label_len = max((len(label) for label in bundle.rank_dist.labels), default=0)
    rank_label_col_width = min(40, max(len(item_singular_title) + 1, max_rank_label_len + 2))
    rank_bar_width = 14
    n_ranked_items = len(bundle.rank_dist.labels)
    print(
        f"  {item_singular_title:<{rank_label_col_width}s} "
        f"{'P(Best)':>9s} {'':<{rank_bar_width}s} "
        f"{'E[Rank]':>9s} {'':<{rank_bar_width}s}"
    )
    for i, label in enumerate(bundle.rank_dist.labels):
        rank_label = _truncate_label(label, rank_label_col_width)
        p_best = float(bundle.rank_dist.p_best[i])
        expected_rank = float(bundle.rank_dist.expected_ranks[i])
        p_color = _p_best_color(p_best)
        p_reset = _RESET if p_color else ""
        p_str = f"{p_best:>8.1%} {_ratio_bar(p_best, width=rank_bar_width)}"
        print(
            f"  {rank_label:<{rank_label_col_width}s} "
            f"{p_color}{p_str}{p_reset} "
            f"{expected_rank:>8.2f} {_rank_hump_lane(expected_rank, n_ranked_items, width=rank_bar_width)}"
        )
    print("  E[Rank] lane: left is better (#1); peak is sharper near integer ranks, softer near half-ranks")
    print()

    _print_mean_advantage(
        bundle,
        item_singular=item_singular,
        line_width=line_width,
        template_col_width=template_col_width,
    )
    print()

    _print_pairwise_section(
        bundle,
        top_pairwise=top_pairwise,
        line_width=line_width,
        p_value_method=p_value_method,
        pairwise_sort=pairwise_sort,
    )

    # Seed variance section (only when seeded data is present).
    if bundle.seed_variance is not None:
        print()
        _print_seed_variance(
            bundle.seed_variance,
            template_col_width=template_col_width,
            item_singular=item_singular,
        )

    # LMM diagnostics (standard one-factor LMM).
    if bundle.lmm_info is not None:
        print()
        _print_lmm_summary(bundle)

    # Factorial LMM diagnostics (factor tests + marginal means).
    if bundle.factorial_lmm_info is not None:
        print()
        _print_factorial_lmm_summary(bundle)

    # Executive summary leaderboard (always last — immediately visible in terminal).
    print()
    _print_executive_summary(bundle, item_singular=item_singular)


# ---------------------------------------------------------------------------
# Seed variance section
# ---------------------------------------------------------------------------

_BLOCK_CHARS = "▁▂▃▄▅▆▇█"


def _seed_noise_strip(
    per_cell_values: np.ndarray,
    scale_max: float,
    max_width: int = 40,
) -> str:
    """One Unicode block char per input, scaled against ``scale_max``.

    If there are more inputs than ``max_width``, inputs are averaged into
    bins first so the strip always fits within the column.
    """
    m = len(per_cell_values)
    if m == 0:
        return ""
    if scale_max <= 0:
        return _BLOCK_CHARS[0] * min(m, max_width)
    if m > max_width:
        bins = np.array_split(per_cell_values, max_width)
        values = np.array([b.mean() for b in bins])
    else:
        values = per_cell_values
    chars = []
    for v in values:
        idx = int(round(float(v) / scale_max * (len(_BLOCK_CHARS) - 1)))
        chars.append(_BLOCK_CHARS[max(0, min(idx, len(_BLOCK_CHARS) - 1))])
    return "".join(chars)


def _print_seed_variance(
    sv: SeedVarianceResult,
    template_col_width: int = 24,
    strip_width: int = 24,
    item_singular: str = "template",
) -> None:
    """Print seed variance decomposition with per-input heat strip."""
    _print_subsection(f"--- Per-input Variance Across Runs (R={sv.n_runs} runs) ---")
    global_cell_max = float(sv.per_cell_seed_std.max())
    print(
        f"  key: ▁–█ = per-input noise   "
        f"(globally scaled; █ = {global_cell_max:.4f})"
    )
    num_w = 10
    print(
        f"  {item_singular.capitalize():<{template_col_width}s}  "
        f"{'Per-input noise':<{strip_width}s}  "
        f"{'seed_std':>{num_w}s}  "
        f"{'input_std':>{num_w}s}  "
        f"{'total_std':>{num_w}s}  "
        f"{'instability':>{num_w}s}  "
        f"Verdict"
    )
    for i, label in enumerate(sv.labels):
        strip = _seed_noise_strip(
            sv.per_cell_seed_std[i], global_cell_max, max_width=strip_width
        )
        instability = float(sv.instability[i])
        verdict = _instability_label(instability)
        verdict_color = _instability_color(instability)
        print(
            f"  {_truncate_label(label, template_col_width):<{template_col_width}s}  "
            f"{strip:<{strip_width}s}  "
            f"{np.sqrt(sv.seed_var[i]):>{num_w}.4f}  "
            f"{np.sqrt(sv.input_var[i]):>{num_w}.4f}  "
            f"{np.sqrt(sv.total_var[i]):>{num_w}.4f}  "
            f"{instability:>{num_w}.4f}  "
            f"{verdict_color}{verdict}{_RESET}"
        )
    print()


def _collect_cross_model_seed_instability_rows(
    bundle: MultiModelBundle,
) -> list[tuple[str, float, float, float, str, float]]:
    """Collect sorted per-model instability rows for summary tables."""
    rows: list[tuple[str, float, float, float, str, float]] = []
    for model_label, model_bundle in bundle.per_model.items():
        sv = model_bundle.seed_variance
        if sv is None:
            continue

        overall_instability = float(np.mean(sv.per_cell_seed_std))
        template_instability_mean = float(np.mean(sv.instability))
        template_instability_std = float(np.std(sv.instability, ddof=0))

        noisiest_idx = int(np.argmax(sv.instability))
        noisiest_template = sv.labels[noisiest_idx]
        noisiest_value = float(sv.instability[noisiest_idx])

        rows.append((
            model_label,
            overall_instability,
            template_instability_mean,
            template_instability_std,
            noisiest_template,
            noisiest_value,
        ))

    rows.sort(key=lambda row: row[1])
    return rows


def _print_cross_model_seed_instability(
    bundle: MultiModelBundle,
    *,
    rows: Optional[list[tuple[str, float, float, float, str, float]]] = None,
) -> None:
    """Print cross-model instability comparison when seed variance is available."""
    if rows is None:
        rows = _collect_cross_model_seed_instability_rows(bundle)

    if len(rows) == 0:
        return

    print()
    _print_subsection("--- Cross-Model Instability (across templates & inputs) ---")
    print(
        "  lower is better (more stable): "
        "instability = mean within-cell run std"
    )
    model_w = max(16, min(34, max(len(row[0]) for row in rows)))
    print(
        f"  {'Model':<{model_w}s} "
        f"{'instability':>12s} "
        f"{'tpl_mean':>10s} "
        f"{'tpl_std':>9s} "
        f"{'Noisiest template':<24s} "
        "Verdict"
    )

    for (
        model_label,
        overall_instability,
        template_instability_mean,
        template_instability_std,
        noisiest_template,
        noisiest_value,
    ) in rows:
        noisiest_desc = f"{_truncate_label(noisiest_template, 16)} ({noisiest_value:.4f})"
        verdict = _instability_label(overall_instability)
        verdict_color = _instability_color(overall_instability)
        print(
            f"  {_truncate_label(model_label, model_w):<{model_w}s} "
            f"{overall_instability:>12.4f} "
            f"{template_instability_mean:>10.4f} "
            f"{template_instability_std:>9.4f} "
            f"{noisiest_desc:<24s} "
            f"{verdict_color}{verdict}{_RESET}"
        )
    print()


# ---------------------------------------------------------------------------
# LMM diagnostics
# ---------------------------------------------------------------------------

def _print_lmm_summary(bundle: AnalysisBundle) -> None:
    """Print LMM variance-component diagnostics for a standard (one-factor) LMM."""
    info = bundle.lmm_info
    if info is None:
        return
    _print_subsection("--- LMM Diagnostics ---")
    print(f"  Formula : {info.formula}")
    print(
        f"  ICC={info.icc:.3f}  σ_input={info.sigma_input:.4f}  "
        f"σ_resid={info.sigma_resid:.4f}  n_obs={info.n_obs}"
        + ("" if info.converged else f"  {_YELLOW}[convergence warning]{_RESET}")
    )


def _print_factorial_lmm_summary(bundle: AnalysisBundle) -> None:
    """Print factorial LMM diagnostics: variance components, factor tests, marginal means."""
    info = bundle.factorial_lmm_info
    if info is None:
        return

    _print_subsection("--- Factorial LMM Diagnostics ---")
    print(f"  Formula : {info.formula}")
    print(
        f"  ICC={info.icc:.3f}  σ_input={info.sigma_input:.4f}  "
        f"σ_resid={info.sigma_resid:.4f}  n_obs={info.n_obs}"
        + ("" if info.converged else f"  {_YELLOW}[convergence warning]{_RESET}")
    )
    print()

    # Factor / interaction Wald tests
    _print_subsection("--- Factor Tests (Wald χ²) ---")
    ft = info.factor_tests
    if ft is not None and len(ft) > 0:
        ft_sorted = ft.sort_values(["p_value", "statistic"], ascending=[True, False])
        term_w = min(42, max(len("Term"), max(len(str(t)) for t in ft_sorted["term"]) + 2))
        bar_w = 12
        print(
            f"  {'Term':<{term_w}s}  {'χ²':>10s}  {'df':>4s}  {'p-value':>12s}  {'Evidence':<{bar_w}s}"
        )
        print(f"  {'-' * term_w}  {'-' * 10}  {'-' * 4}  {'-' * 12}  {'-' * bar_w}")
        for _, row in ft_sorted.iterrows():
            pval = float(row["p_value"])
            evidence = 1.0 - float(np.clip(pval, 0.0, 1.0)) if not np.isnan(pval) else np.nan
            p_str = _format_p_value(pval)
            print(
                f"  {_truncate_label(str(row['term']), term_w):<{term_w}s}  "
                f"{float(row['statistic']):>10.3f}  "
                f"{float(row['df']):>4.0f}  "
                f"{p_str:>12s}  "
                f"{_ratio_bar(evidence, width=bar_w)}"
            )
        n_sig = int(np.sum(ft_sorted["p_value"].to_numpy(dtype=float) < 0.05))
        if n_sig == 0:
            print(
                f"  {_YELLOW}[!] No factor/interaction terms pass p < 0.05; "
                "interpret level differences cautiously." + f"{_RESET}"
            )
        else:
            print(f"  Significant terms (p < 0.05): {n_sig}/{len(ft_sorted)}")
    else:
        print("  (no factor tests available)")

    _print_factorial_interaction_plot(bundle, factor_tests=ft)

    # Estimated marginal means per factor
    mm = info.marginal_means
    if mm:
        line_width = 41
        for factor_name, mm_df in mm.items():
            print()
            _print_subsection(f"--- Marginal Means: {factor_name} ---")
            if len(mm_df) == 0:
                print("  (no marginal means available)")
                continue

            mm_sorted = mm_df.sort_values(["mean", "level"], ascending=[False, True]).reset_index(drop=True)
            means = mm_sorted["mean"].to_numpy(dtype=float)
            ci_low = mm_sorted["ci_low"].to_numpy(dtype=float)
            ci_high = mm_sorted["ci_high"].to_numpy(dtype=float)

            factor_center = float(np.mean(means))
            centered_mean = means - factor_center
            centered_low = ci_low - factor_center
            centered_high = ci_high - factor_center

            axis_max = max(
                1e-12,
                float(
                    np.max(
                        np.abs(
                            np.concatenate([centered_mean, centered_low, centered_high])
                        )
                    )
                ),
            )
            axis_low = -axis_max
            axis_high = axis_max
            level_w = min(28, max(len("Level"), max(len(str(v)) for v in mm_sorted["level"]) + 2))

            print(
                f"  axis: [{axis_low:+.3f}, {axis_high:+.3f}]  "
                "(─ CI, ● mean, │ factor mean)"
            )
            print(
                f"  {'Level':<{level_w}s} {'Interval Plot':<{line_width}s} "
                f"{'Mean':>8s} {'SE':>8s} {'CI Low':>9s} {'CI High':>9s} {'Δ vs avg':>10s}"
            )
            for i, row in mm_sorted.iterrows():
                interval_line = _ascii_interval_line(
                    mean=float(centered_mean[i]),
                    ci_low=float(centered_low[i]),
                    ci_high=float(centered_high[i]),
                    spread_low=float(centered_mean[i]),
                    spread_high=float(centered_mean[i]),
                    axis_low=axis_low,
                    axis_high=axis_high,
                    width=line_width,
                )
                print(
                    f"  {_truncate_label(str(row['level']), level_w):<{level_w}s} "
                    f"{interval_line:<{line_width}s} "
                    f"{float(row['mean']):>8.4f}  "
                    f"{float(row['se']):>8.4f}  "
                    f"{float(row['ci_low']):>9.4f}  "
                    f"{float(row['ci_high']):>9.4f}  "
                    f"{float(centered_mean[i]):>+10.4f}"
                )
            best_idx = int(np.argmax(means))
            best_level = str(mm_sorted.iloc[best_idx]["level"])
            best_mean = float(means[best_idx])
            print(
                f"  {_BRIGHT_GREEN}-> Highest marginal mean:{_RESET} "
                f"'{_BOLD}{_BRIGHT_GREEN}{best_level}{_RESET}' (mean={best_mean:.4f}, Δ={centered_mean[best_idx]:+.4f} vs factor average)"
            )


def _factor_names_from_term(term: str) -> list[str]:
    """Extract factor names from a model-term string such as ``C(a):C(b)``."""
    names = re.findall(r"C\(([^)]+)\)", str(term))
    if names:
        return names
    return [p.strip() for p in str(term).split(":") if p.strip()]


def _print_factorial_interaction_plot(
    bundle: AnalysisBundle,
    *,
    factor_tests,
    alpha: Optional[float] = None,
) -> None:
    """Render an optional terminal interaction plot via plotext when interaction is significant."""
    if alpha is None:
        alpha = get_alpha_ci()
    info = bundle.factorial_lmm_info
    if info is None or factor_tests is None or len(factor_tests) == 0:
        return

    is_interaction = factor_tests["term"].astype(str).str.contains(":", regex=False)
    if not bool(np.any(is_interaction)):
        return

    sig_interactions = factor_tests.loc[
        is_interaction & (factor_tests["p_value"].to_numpy(dtype=float) < alpha)
    ]
    if len(sig_interactions) == 0:
        return

    tf = bundle.benchmark.template_factors
    if tf is None or len(tf) != bundle.benchmark.n_templates:
        return

    best_row = sig_interactions.sort_values(["p_value", "statistic"], ascending=[True, False]).iloc[0]
    term = str(best_row["term"])
    factors = _factor_names_from_term(term)
    if len(factors) < 2:
        return

    x_factor, line_factor = factors[0], factors[1]
    if x_factor not in tf.columns or line_factor not in tf.columns:
        return

    tf_plot = tf.copy()
    tf_plot["_score"] = bundle.robustness.mean.astype(float)

    group_cols = [x_factor, line_factor]
    grouped = (
        tf_plot.groupby(group_cols, observed=True, dropna=False)["_score"]
        .mean()
        .reset_index()
    )
    if len(grouped) == 0:
        return

    x_levels = [str(v) for v in tf_plot[x_factor].drop_duplicates().tolist()]
    line_levels = [str(v) for v in tf_plot[line_factor].drop_duplicates().tolist()]

    x_map = {str(v): i for i, v in enumerate(x_levels)}

    grouped["_x_label"] = grouped[x_factor].astype(str)
    grouped["_line_label"] = grouped[line_factor].astype(str)
    grouped["_x_ord"] = grouped["_x_label"].map(x_map)
    grouped = grouped.sort_values(["_line_label", "_x_ord"]).reset_index(drop=True)

    print()
    _print_subsection("--- Interaction Plot (significant interaction) ---")
    print(
        f"  term='{term}'  (p={_format_p_value(float(best_row['p_value']))}); "
        f"x='{x_factor}', lines='{line_factor}'"
    )

    if len(factors) > 2:
        held = ", ".join(factors[2:])
        print(
            f"  {_YELLOW}[!] Higher-order interaction detected; plot shows only first two factors "
            f"and averages over: {held}.{_RESET}"
        )

    try:
        import plotext as plt  # type: ignore[import-not-found]
    except Exception:
        print(
            f"  {_YELLOW}[!] plotext not installed; skipping terminal interaction plot. "
            "Install with `pip install plotext` to enable this view."
            f"{_RESET}"
        )
        return

    try:
        plt.clear_figure()
        plt.canvas_color("default")
        plt.axes_color("default")
        plt.ticks_color("white")
        plt.plotsize(92, 22)
        plt.title(f"Interaction: {x_factor} × {line_factor}")
        plt.xlabel(x_factor)
        plt.ylabel("mean score")

        x_tick_vals = list(range(len(x_levels)))
        plt.xticks(x_tick_vals, x_levels)

        for line_level in line_levels:
            part = grouped[grouped["_line_label"] == line_level]
            if len(part) == 0:
                continue
            x_vals = part["_x_ord"].to_numpy(dtype=float).tolist()
            y_vals = part["_score"].to_numpy(dtype=float).tolist()
            plt.plot(x_vals, y_vals, marker="dot", label=line_level)

        grid_fn = getattr(plt, "grid", None)
        if callable(grid_fn):
            try:
                grid_fn(True, True)
            except TypeError:
                try:
                    grid_fn(True)
                except Exception:
                    pass

        legend_fn = getattr(plt, "legend", None)
        if callable(legend_fn):
            try:
                legend_fn(True)
            except TypeError:
                try:
                    legend_fn()
                except Exception:
                    pass

        plt.show()
    except Exception as exc:
        print(
            f"  {_YELLOW}[!] plotext rendering failed ({type(exc).__name__}: {exc}); "
            "continuing without plot."
            f"{_RESET}"
        )


# ---------------------------------------------------------------------------
# ASCII rendering primitives
# ---------------------------------------------------------------------------

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
    reference: float = 0.0,
) -> str:
    """Render a one-line ASCII interval plot with a reference marker.

    Parameters
    ----------
    reference : float
        Position of the ``│`` reference marker on the axis (default ``0.0``).
        Pass the grand mean when using absolute-scale plots so the reference
        line marks the average rather than zero.
    """
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

    ref_idx = to_idx(reference)
    chars[ref_idx] = "│"
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


def _split_model_template_label(label: str) -> tuple[str, str]:
    """Split labels of the form 'model / template' into separate columns."""
    parts = label.split(" / ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return label, ""


def _ratio_bar(value: float, width: int = 12) -> str:
    """Render a fixed-width progress bar for values in [0, 1]."""
    width = max(1, int(width))
    if np.isnan(value):
        return "░" * width
    clamped = float(np.clip(value, 0.0, 1.0))
    filled = int(round(clamped * width))
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def _rank_hump_lane(expected_rank: float, n_items: int, width: int = 14) -> str:
    """Render rank position as a horizontal lane with an adaptive hump.

    Left corresponds to rank #1 (best). The hump is sharper when
    ``expected_rank`` is near an integer and softer when it is near the
    midpoint between integers.
    """
    width = max(3, int(width))
    if n_items <= 1 or np.isnan(expected_rank):
        center = width // 2
        lane = ["─"] * width
        lane[center] = "█"
        return "".join(lane)

    clamped_rank = float(np.clip(expected_rank, 1.0, float(n_items)))
    pos = (clamped_rank - 1.0) / (float(n_items) - 1.0)
    center = int(round(pos * (width - 1)))

    frac_to_int = abs(clamped_rank - round(clamped_rank))
    sharpness = 1.0 - min(frac_to_int, 0.5) / 0.5

    if sharpness >= 0.67:
        profile = {0: "█", 1: "▆", 2: "▃"}
    elif sharpness >= 0.33:
        profile = {0: "▇", 1: "▅", 2: "▂"}
    else:
        profile = {0: "▆", 1: "▄", 2: "▁"}

    lane = ["─"] * width
    for offset, char in profile.items():
        left = center - offset
        right = center + offset
        if 0 <= left < width:
            lane[left] = char
        if 0 <= right < width:
            lane[right] = char

    return "".join(lane)


# ---------------------------------------------------------------------------
# p-value formatting
# ---------------------------------------------------------------------------

def _p_value_stars(p_value: Optional[float]) -> str:
    """Return significance stars for p-value thresholds (*, **, ***)."""
    if p_value is None:
        return ""
    if p_value < 0.0001:
        return "***"
    if p_value < 0.001:
        return "**"
    if p_value < 0.01:
        return "*"
    return ""


def _format_p_value(p_value: Optional[float]) -> str:
    """Format p-value with significance stars; return N/A for missing values."""
    if p_value is None:
        return "N/A"
    return f"{p_value:.4g}{_p_value_stars(p_value)}"


# ---------------------------------------------------------------------------
# Critical-difference group detection
# ---------------------------------------------------------------------------

def _pairwise_rank_band_p(
    pairwise: PairwiseMatrix,
    label_a: str,
    label_b: str,
    *,
    p_source: Literal["bootstrap", "wilcoxon"],
) -> Optional[float]:
    """Return the pairwise p-value used to decide rank-band indistinguishability."""
    try:
        result = pairwise.get(label_a, label_b)
    except KeyError:
        return None

    if p_source == "bootstrap":
        return float(result.p_value)
    if p_source == "wilcoxon":
        return None if result.wilcoxon_p is None else float(result.wilcoxon_p)

    p_values = [float(result.p_value)]
    if result.wilcoxon_p is not None:
        p_values.append(float(result.wilcoxon_p))
    return min(p_values) if p_values else None


def _critical_difference_groups(
    pairwise: PairwiseMatrix,
    *,
    labels_sorted: list[str],
    alpha: Optional[float] = None,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
) -> list[list[str]]:
    """Return contiguous, maximal non-significant rank bands.

    When simultaneous CIs have been computed, significance is determined by
    whether the pairwise CI excludes zero (consistent with the displayed CIs).
    Otherwise, correction-adjusted p-values compared to *alpha* are used.
    """
    if alpha is None:
        alpha = get_alpha_ci()
    if len(labels_sorted) < 2:
        return []

    n_labels = len(labels_sorted)
    use_ci = pairwise.simultaneous_ci_method is not None

    def _all_pairs_nonsignificant(group_labels: list[str]) -> bool:
        for i in range(len(group_labels)):
            for j in range(i + 1, len(group_labels)):
                if use_ci:
                    try:
                        result = pairwise.get(group_labels[i], group_labels[j])
                    except KeyError:
                        return False
                    if result.ci_low > 0 or result.ci_high < 0:
                        return False
                else:
                    p_value = _pairwise_rank_band_p(
                        pairwise, group_labels[i], group_labels[j], p_source=p_source,
                    )
                    if p_value is None or p_value < alpha:
                        return False
        return True

    candidate_groups: list[list[str]] = []
    for start_idx in range(n_labels - 1):
        best_group: Optional[list[str]] = None
        for end_idx in range(start_idx + 1, n_labels):
            group = labels_sorted[start_idx : end_idx + 1]
            if _all_pairs_nonsignificant(group):
                best_group = group
            else:
                break
        if best_group is not None:
            candidate_groups.append(best_group)

    def _is_contiguous_subsequence(smaller: list[str], larger: list[str]) -> bool:
        if len(smaller) >= len(larger):
            return False
        max_start = len(larger) - len(smaller)
        for start in range(max_start + 1):
            if larger[start : start + len(smaller)] == smaller:
                return True
        return False

    maximal_groups: list[list[str]] = []
    for group in candidate_groups:
        if any(
            _is_contiguous_subsequence(group, other)
            for other in candidate_groups
            if other is not group
        ):
            continue
        maximal_groups.append(group)

    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in maximal_groups:
        key = tuple(group)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(group)
    return deduped


def _single_clear_winner_label(
    pairwise: PairwiseMatrix,
    *,
    labels_sorted: list[str],
    alpha: Optional[float] = None,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
) -> Optional[str]:
    """Return the unique label that significantly beats every other label."""
    if alpha is None:
        alpha = get_alpha_ci()
    if len(labels_sorted) < 2:
        return None

    use_ci = pairwise.simultaneous_ci_method is not None
    winners: list[str] = []
    for candidate in labels_sorted:
        candidate_beats_all = True
        for other in labels_sorted:
            if other == candidate:
                continue

            try:
                result = pairwise.get(candidate, other)
            except KeyError:
                candidate_beats_all = False
                break

            if use_ci:
                sig = result.ci_low > 0 or result.ci_high < 0
                beats = float(result.point_diff) > 0
            else:
                p_value = _pairwise_rank_band_p(
                    pairwise, candidate, other, p_source=p_source,
                )
                sig = p_value is not None and p_value < alpha
                beats = float(result.point_diff) > 0

            if not sig or not beats:
                candidate_beats_all = False
                break

        if candidate_beats_all:
            winners.append(candidate)
            if len(winners) > 1:
                return None

    return winners[0] if len(winners) == 1 else None


def _print_critical_difference_groups(
    pairwise: PairwiseMatrix,
    *,
    labels_sorted: list[str],
    alpha: Optional[float] = None,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
) -> None:
    """Print a short CD-style summary of statistically indistinguishable groups."""
    if alpha is None:
        alpha = get_alpha_ci()
    if len(labels_sorted) < 2:
        return

    rank_pos = {label: idx + 1 for idx, label in enumerate(labels_sorted)}

    if pairwise.simultaneous_ci_method is not None:
        source_label = f"{(1-alpha)*100:.0f}% CI, {pairwise.simultaneous_ci_method}-adjusted"
    else:
        source_label = {
            "bootstrap": "p (boot)",
            "wilcoxon": "p (wsr)",
        }[p_source]

    groups = _critical_difference_groups(
        pairwise,
        labels_sorted=labels_sorted,
        alpha=alpha,
        p_source=p_source,
    )
    if not groups:
        print(
            f"  Statistically indistinguishable rank bands "
            f"({source_label}): none"
        )
        return

    print(
        f"  Statistically indistinguishable rank bands "
        f"{_DIM}(similar to critical difference diagrams) computed from {source_label}:{_RESET}"
    )
    for group in groups:
        start_rank = rank_pos[group[0]]
        end_rank = rank_pos[group[-1]]
        rank_span = f"#{start_rank}" if start_rank == end_rank else f"#{start_rank}–#{end_rank}"
        print(f"    {rank_span}: [{' ─ '.join(group)}]")

    clear_winner = _single_clear_winner_label(
        pairwise,
        labels_sorted=labels_sorted,
        alpha=alpha,
        p_source=p_source,
    )
    if clear_winner is not None:
        print()
        print(
            f"  {_BRIGHT_GREEN}-> Evidence suggests a clear best option:{_RESET} "
            f"'{_BOLD}{_BRIGHT_GREEN}{clear_winner}{_RESET}'"
        )


# ---------------------------------------------------------------------------
# Executive summary helpers
# ---------------------------------------------------------------------------

def _assign_significance_groups(
    pairwise: PairwiseMatrix,
    labels_sorted: list[str],
    alpha: Optional[float] = None,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
) -> dict[str, str]:
    """Assign numeric group IDs (#1, #2, #3…) to templates via CD-group analysis.

    Templates in the same maximal non-significant rank band share an ID.
    Group #1 is the band that contains the rank-1 template. Any template not
    found in any CD group receives a unique ID (it is distinctly ranked).
    """
    if alpha is None:
        alpha = get_alpha_ci()
    groups = _critical_difference_groups(
        pairwise, labels_sorted=labels_sorted, alpha=alpha, p_source=p_source,
    )
    rank_map = {label: i for i, label in enumerate(labels_sorted)}
    groups_sorted = sorted(groups, key=lambda g: min(rank_map.get(l, 999) for l in g))
    top_label = labels_sorted[0]

    label_to_group: dict[str, str] = {}

    # Group #1 is reserved for the top-ranked item and any non-significant ties
    # that share its maximal contiguous rank band.
    label_to_group[top_label] = "#1"
    for group in groups_sorted:
        if top_label in group:
            for label in group:
                label_to_group[label] = "#1"

    group_idx = 1
    for group in groups_sorted:
        if top_label in group:
            continue
        new_members = [l for l in group if l not in label_to_group]
        if new_members:
            group_id = f"#{group_idx + 1}"
            group_idx += 1
            for label in new_members:
                label_to_group[label] = group_id

    # Templates not in any CD group each get their own unique letter.
    for label in labels_sorted:
        if label not in label_to_group:
            group_id = f"#{group_idx + 1}"
            label_to_group[label] = group_id
            group_idx += 1

    return label_to_group


def _exec_verdict(
    label: str,
    label_to_group: dict[str, str],
    labels_sorted: list[str],
) -> str:
    """Human-readable verdict for a template in the executive summary."""
    my_group = label_to_group.get(label, "?")
    if my_group != "#1":
        return "Significant drop-off"
    group_1 = [l for l in labels_sorted if label_to_group.get(l) == "#1"]
    others = [l for l in group_1 if l != label]
    if not others:
        return "Likely best"
    if len(others) == 1:
        return f"Tied with {_truncate_label(others[0], 20)} as best"
    return f"Tied with {len(others)} others as best"


def _print_executive_summary(
    bundle: AnalysisBundle,
    *,
    item_singular: str = "template",
) -> None:
    """Print a concise executive leaderboard after the stats-heavy blocks.

    Shows each template's significance group, mean score, bootstrap CI,
    optional stability (when seed data is present), and a plain-language
    verdict so the user can assess results at a glance without scrolling up.
    """
    labels = list(bundle.rank_dist.labels)
    n = len(labels)
    if n < 2:
        return

    # Sort by mean score descending (best first).
    means = bundle.robustness.mean
    sort_idx = list(np.argsort(-means))
    labels_sorted = [labels[i] for i in sort_idx]

    # Significance group letters via CD groups.
    label_to_group = _assign_significance_groups(bundle.pairwise, labels_sorted)

    # Seed variance for stability column (optional).
    sv = bundle.seed_variance
    has_stability = sv is not None

    item_title = item_singular.capitalize()
    _print_subsection(f"--- Executive Summary ({item_title} leaderboard) ---")

    # Column widths.
    tpl_w = min(28, max(16, max(len(l) for l in labels)))
    grp_w = 4
    mean_w = 6
    ci_w = 15  # e.g. "[0.950, 0.990]" = 14 chars + 1 padding
    stab_w = 16

    # CI column header: Wilson CI when no bootstrap was used (binary data path).
    ci_col_header = "Wilson CI" if bundle.point_advantage.n_bootstrap == 0 else "CI"

    # Header row (no ANSI codes so widths match exactly).
    header_parts = [
        f"  {item_title:<{tpl_w}s}",
        f"  {'Grp':^{grp_w}s}",
        f"  {'Mean':>{mean_w}s}",
        f"  {ci_col_header:<{ci_w}s}",
    ]
    if has_stability:
        header_parts.append(f"  {'Stability':<{stab_w}s}")
    header_parts.append("  Verdict")
    header = "".join(header_parts)
    sep = "  " + "─" * (len(header) - 2)
    print(header)
    print(sep)

    for label in labels_sorted:
        orig_idx = labels.index(label)
        mean_val = float(means[orig_idx])

        ci_lo = float(bundle.robustness.ci_low[orig_idx])
        ci_hi = float(bundle.robustness.ci_high[orig_idx])
        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"

        group = label_to_group.get(label, "?")
        verdict = _exec_verdict(label, label_to_group, labels_sorted)

        # Pre-format fixed-width parts, then optionally wrap with ANSI.
        plain_label = f"{_truncate_label(label, tpl_w):<{tpl_w}s}"
        plain_grp = f"{group:^{grp_w}s}"
        if group == "#1" and _ANSI:
            label_str = f"{_BOLD}{_BRIGHT_GREEN}{plain_label}{_RESET}"
            grp_str = f"{_BOLD}{_BRIGHT_GREEN}{plain_grp}{_RESET}"
            verdict_str = f"{_BRIGHT_GREEN}{verdict}{_RESET}"
        else:
            label_str = plain_label
            grp_str = plain_grp
            verdict_str = verdict

        row = (
            f"  {label_str}"
            f"  {grp_str}"
            f"  {mean_val:>{mean_w}.3f}"
            f"  {ci_str:<{ci_w}s}"
        )

        if has_stability:
            sv_labels = list(sv.labels)
            if label in sv_labels:
                sv_idx = sv_labels.index(label)
                stab_str = _stability_emoji_label(float(sv.instability[sv_idx]))
            else:
                stab_str = "—"
            row += f"  {stab_str:<{stab_w}s}"

        row += f"  {verdict_str}"
        print(row)

    print(sep)
    print()
