"""Console summary printer for GroupComparisonResult (between-subjects
compare() results). Deliberately narrower than paired.py's summary: per-
group means with gradient CIs, a pairwise comparison table, and the
omnibus test at k>=3 -- no executive summary, critical-difference rank
bands, or forest-plot brackets. See PLAN_between_subjects_extension.md
§1/§3.6.

Reuses the paired path's low-level line-rendering primitives directly
(``_choose_interval_line`` et al.) rather than reimplementing them --
those already work on plain floats, no bundle/PairwiseMatrix required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evalstats.config import get_alpha_ci
from evalstats.core.summary import (
    _ANSI, _RESET, _BOLD, _DIM, _BRIGHT_MAGENTA, _BRIGHT_GREEN, _BRIGHT_RED,
    _choose_interval_line, _legend_ci_label, _mean_marker_legend,
    _print_subsection,
)

if TYPE_CHECKING:
    from evalstats.core.unpaired import GroupComparisonResult

_FAMILY_DISPLAY = {
    "binary_proportion": "proportion difference (Δp)",
    "rank_based": "stochastic dominance (θ = P(a>b))",
}
_ESTIMAND_LABEL = {"mean_diff": "Δ", "dominance": "θ"}


def _truncate(label: str, width: int) -> str:
    if len(label) <= width:
        return label
    return label[: max(1, width - 1)] + "…"


def print_group_comparison_summary(result: "GroupComparisonResult", *, style: str = "gradient") -> None:
    alpha = result.alpha
    ci_pct = int(round((1 - alpha) * 100))

    print(f"{_BOLD}Between-subjects comparison{_RESET}  "
          f"(design=unpaired; factor={result.factor_col!r}, metric={result.metric_col!r})")
    item_note = " (synthetic -- no item column found; each row is its own item)" if result.item_col_synthetic else ""
    print(f"Item column: {result.item_col!r}{item_note}")
    print(f"Groups: {len(result.groups)}  |  Score type: {result.score_type}  |  "
          f"Family: {_FAMILY_DISPLAY[result.family]}")
    print()

    if result.ppi_applied:
        banner = "═" * 58
        print(f"{_BOLD}{_BRIGHT_MAGENTA}{banner}{_RESET}")
        print(
            f"{_BOLD}{_BRIGHT_MAGENTA}PPI-CORRECTED — every estimate below relies on the "
            f"alignment report printed here first.{_RESET}"
        )
        print(f"{_BOLD}{_BRIGHT_MAGENTA}{banner}{_RESET}")
        if result.alignment_result is not None:
            result.alignment_result.summary()
        else:
            print(
                f"{_BOLD}{_BRIGHT_MAGENTA}(Alignment report unavailable -- run "
                f"judge_alignment(...).summary() directly.){_RESET}"
            )
        print()

    # ── Per-group means ──────────────────────────────────────────────────────
    _print_subsection("--- Group Means (marginal CIs) ---")
    all_vals = []
    for g in result.groups:
        all_vals.extend([g.mean - g.std, g.mean + g.std, g.ci_low, g.ci_high])
        if g.multi_ci:
            for lo, hi in g.multi_ci.values():
                all_vals.extend([lo, hi])
    all_vals = [v for v in all_vals if np.isfinite(v)]
    if all_vals:
        val_range = max(all_vals) - min(all_vals)
        pad = max(val_range * 0.05, 1e-4)
        axis_low, axis_high = min(all_vals) - pad, max(all_vals) + pad
    else:
        axis_low, axis_high = 0.0, 1.0
    grand_mean = float(np.mean([g.mean for g in result.groups]))

    label_width = min(24, max(8, max(len(g.label) for g in result.groups)))
    line_width = 44
    ci_legend = _legend_ci_label(style, ci_pct, any(g.multi_ci for g in result.groups))
    mean_marker = _mean_marker_legend(style, "grand mean")
    print(f"  axis: [{axis_low:.3f}, {axis_high:.3f}]  (· ±1σ, {ci_legend}{mean_marker}, │ grand mean)")
    print(f"  {'Group':<{label_width}s} {'Interval Plot':<{line_width}s} {'Mean':>8s} {'CI Low':>9s} {'CI High':>9s}")
    for g in result.groups:
        line = _choose_interval_line(
            mean=g.mean, ci_low=g.ci_low, ci_high=g.ci_high,
            spread_low=g.mean - g.std, spread_high=g.mean + g.std,
            axis_low=axis_low, axis_high=axis_high, width=line_width,
            reference=grand_mean, style=style, multi_ci=g.multi_ci,
        )
        print(f"  {_truncate(g.label, label_width):<{label_width}s} {line:<{line_width}s} "
              f"{g.mean:>8.3f} {g.ci_low:>9.3f} {g.ci_high:>9.3f}")
    print()

    # ── Omnibus test ─────────────────────────────────────────────────────────
    if result.omnibus_test_name is not None:
        _print_subsection(f"--- Omnibus Test: {result.omnibus_test_name} ---")
        p_str = f"{result.omnibus_p_value:.4f}" if result.omnibus_p_value >= 0.0001 else f"{result.omnibus_p_value:.2e}"
        print(f"  statistic = {result.omnibus_statistic:.4f}   p = {p_str}"
              f"{'  (uncorrected)' if result.ppi_applied else ''}")
        if result.omnibus_corrected_p_value is not None:
            cp = result.omnibus_corrected_p_value
            cp_str = f"{cp:.4f}" if cp >= 0.0001 else f"{cp:.2e}"
            print(f"  PPI-corrected p = {cp_str}")
        print()

    # ── Pairwise table ───────────────────────────────────────────────────────
    n_pairs = len(result.pairwise)
    correction_note = (
        f"  |  CI: {result.ci_correction}, p-value: {result.pvalue_correction} "
        f"(family of {n_pairs})" if n_pairs > 1 else ""
    )
    _print_subsection(f"--- Pairwise Comparisons ({_FAMILY_DISPLAY[result.family]}) ---")
    print(f"  α={alpha:g}{correction_note}")
    est_label = _ESTIMAND_LABEL[result.pairwise[0].estimand] if result.pairwise else "Δ"
    print(f"  {'Left':<{label_width}s} {'Right':<{label_width}s} {est_label:>8s} "
          f"{'CI Low':>9s} {'CI High':>9s} {'p':>9s}   Verdict")
    for p in result.pairwise:
        p_str = f"{p.p_value:.4f}" if p.p_value >= 0.0001 else f"{p.p_value:.2e}"
        if p.significant:
            direction = ">" if p.point_estimate > p.null_value else "<"
            verdict = f"{_BRIGHT_GREEN}significant ({p.label_a} {direction} {p.label_b}){_RESET}" if _ANSI else \
                f"significant ({p.label_a} {direction} {p.label_b})"
        else:
            verdict = f"{_DIM}not significant{_RESET}" if _ANSI else "not significant"
        print(f"  {_truncate(p.label_a, label_width):<{label_width}s} "
              f"{_truncate(p.label_b, label_width):<{label_width}s} "
              f"{p.point_estimate:>8.3f} {p.ci_low:>9.3f} {p.ci_high:>9.3f} {p_str:>9s}   {verdict}")
    if n_pairs > 1:
        print(f"  {_DIM}Verdict reflects the {result.ci_correction}-corrected CI; p is "
              f"independently {result.pvalue_correction}-corrected -- the two can rarely "
              f"disagree right at the boundary, since they're different (both valid) "
              f"FWER corrections.{_RESET}")
    print()
