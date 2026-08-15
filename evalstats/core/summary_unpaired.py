"""Console summary printer for GroupComparisonResult (between-subjects
compare() results). Deliberately narrower than paired.py's summary: per-
group means with gradient CIs, a pairwise comparison table, and the
omnibus test at k>=3 -- no executive summary, critical-difference rank
bands, or forest-plot brackets. See PLAN_between_subjects_extension.md
§1/§3.6.

Reuses the paired path's rendering primitives directly -- the PPI banner
(``_print_ppi_banner``), the per-entity means table
(``_print_mean_advantage``), the pairwise comparison table
(``_print_pairwise_section``), and the Pareto-front section
(``_print_pareto_section``, when ``secondary_metric=`` was passed) are the
*same* functions the paired path calls, not reimplementations, so a change
to any of them renders identically for both paths. What's genuinely
unpaired-specific (this engine's fixed Bonferroni-CI/Holm-p FWER scheme, vs.
the paired path's six CI/p-value method families plus Friedman/Nemenyi;
its own per-group joint bootstrap for the Pareto front, since there's no
shared item pool across disjoint groups) is resolved by
``_prepare_unpaired_pairwise_rows``/``pareto_bootstrap_unpaired`` into the
same shapes the shared renderers consume either way. The Behavioral
Agreement (McNemar-style) subsection is paired-only and never called here
-- ``agreement_mcc``/``binary_confusion`` need the same item scored by both
entities, which has no between-subjects equivalent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evalstats.core.summary import (
    _RESET, _BOLD,
    _print_subsection, _print_ppi_banner, _print_mean_advantage, _print_pairwise_section,
    _print_pareto_section,
    _FAMILY_DISPLAY_UNPAIRED as _FAMILY_DISPLAY,
)

if TYPE_CHECKING:
    from evalstats.core.unpaired import GroupComparisonResult


def print_group_comparison_summary(result: "GroupComparisonResult", *, style: str = "gradient") -> None:
    print(f"{_BOLD}Between-subjects comparison{_RESET}  "
          f"(design=unpaired; factor={result.factor_col!r}, metric={result.metric_col!r})")
    item_note = " (synthetic -- no item column found; each row is its own item)" if result.item_col_synthetic else ""
    print(f"Item column: {result.item_col!r}{item_note}")
    print(f"Groups: {len(result.groups)}  |  Score type: {result.score_type}  |  "
          f"Family: {_FAMILY_DISPLAY[result.family]}")
    print()

    if result.ppi_applied:
        _print_ppi_banner(result.alignment_result)

    # ── Per-group means ──────────────────────────────────────────────────────
    # Shared with the paired path's own per-entity means table
    # (_print_mean_advantage in core/summary.py) -- see that function's
    # docstring for why one function renders both.
    label_width = min(24, max(8, max(len(g.label) for g in result.groups)))
    line_width = 44
    _print_mean_advantage(
        labels=[g.label for g in result.groups],
        mean=np.array([g.mean for g in result.groups]),
        std=np.array([g.std for g in result.groups]),
        ci_low=np.array([g.ci_low for g in result.groups]),
        ci_high=np.array([g.ci_high for g in result.groups]),
        multi_ci_per_entity=[g.multi_ci for g in result.groups],
        resolved_ci_method=result.groups[0].method,
        item_singular="group",
        line_width=line_width,
        template_col_width=label_width,
        style=style,
    )
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
    # Shared with the paired path's own pairwise table (_print_pairwise_section
    # in core/summary.py) -- see that function's docstring for why one
    # function renders both, and _prepare_unpaired_pairwise_rows for how this
    # engine's fixed Bonferroni-CI/Holm-p scheme maps onto the shared row
    # shape (no Wilcoxon/Romano-Wolf/Nemenyi/etc. method-family detection,
    # no ranking-bootstrap-based canonical ordering, no ES/agreement columns).
    _print_pairwise_section(result, line_width=line_width, style=style)

    # ── Pareto front (secondary_metric=) ────────────────────────────────────
    # Shared with the paired path's own Pareto section (_print_pareto_section
    # in core/summary.py) -- see that function's docstring; the ASCII
    # scatterplot and Trade-off table render unmodified via
    # _GroupStatsAsRobustness, a minimal adapter so this engine's
    # list[GroupStat] duck-types as the RobustnessResult that function reads.
    if result.pareto is not None:
        _print_pareto_section(result.pareto, metric=result.metric_col, show_rank_probabilities=False)
