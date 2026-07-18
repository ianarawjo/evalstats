"""Recompute --mode simultaneous_ci publication tables directly from a saved
``*_simultaneous_ci_results.csv`` file, without re-running the simulation.

``latex_simultaneous_ci_overall_summary`` (cases/pvalues.py) -- the table a
live ``--latex`` run writes, and what ``\\label{tab:pvalues_simultaneous_ci_overall}``
in the paper is built from -- collapses across *k* entirely (its only
breakdown columns are per-*n*). That hides a real k-dependent failure: boot's
(and max_t's) joint-bootstrap widening undercovers specifically for binary
data at extreme proportions (p near 0 or 1) combined with low *k* and small
*n* -- e.g. coverage as low as ~0.50 for k=3 at n=15, vs. sidak/bonferroni's
~1.00 in the same cells. Averaged into one "Cov(null)" number across every
eval type and k, that failure is invisible; the per-n columns alone can't
surface it either, since it's a k-and-eval-type interaction, not purely an
n effect.

This script reproduces the existing by-n overall table (for parity with what
a live run already produces) and adds three new breakdowns:

- **by-k**: same row/column shape as the by-n table, but columned by k
  instead of n (collapsed across n and eval type). Shows the aggregate
  k-dependence directly.
- **by-(eval type, k)**: one row per (eval type, CI method), collapsed
  across n, with one Cov(null)/Cov(alt) column pair per k value swept.
  Isolates exactly which (eval type, k) cells break down, rather than
  averaging them away.
- **combined**: the single-table version of the above, meant for the paper --
  every column from the by-n overall table (Cov/Width/Score, null and alt)
  PLUS one Cov(null) column per k, with eval type folded into the CI method
  column itself (``"boot (bin)"``/``"boot (num)"``, continuous+likert summed
  into "num" via latex_tables.eval_type_group) instead of a separate column,
  and a ``\\midrule`` dividing the binary block from the numeric block.

Each table is piped through ``revise_latex_tables.revise_table()``, matching
what a live run's ``--latex`` output looks like after revision (dropped MC
band column, trimmed headers) -- the same treatment applied to
``tab:pvalues_simultaneous_ci_overall`` in the paper.

Usage:
  python -m simulations.harness.csv_to_simultaneous_ci_summary path/to/..._simultaneous_ci_results.csv
  python -m simulations.harness.csv_to_simultaneous_ci_summary path/to/... \\
      --alpha 0.05 --tables combined
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .latex_tables import booktabs_table, escape_latex, eval_type_group, eval_type_label
from .methods import CANONICAL_SIMULTANEOUS_CI_METHODS, SIMULTANEOUS_CI_METHODS
from .revise_latex_tables import revise_table

ALL_SIMULTANEOUS_CI_METHODS = SIMULTANEOUS_CI_METHODS + CANONICAL_SIMULTANEOUS_CI_METHODS

REQUIRED_COLUMNS = {
    "eval_type", "n", "k", "ci_method", "condition", "n_reps", "all_covered",
    "avg_width", "avg_score",
}


def _mc_proportion_stats(successes: float, total: float, z: float = 1.96) -> tuple[float, float, float, float]:
    """Same formula as cases/pvalues.py's _mc_proportion_stats."""
    if total <= 0:
        return (float("nan"),) * 4
    p_hat = successes / total
    mcse = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    return float(p_hat), mcse, max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {sorted(missing)} -- expected the "
            "--mode simultaneous_ci *_simultaneous_ci_results.csv schema "
            "(see save_results_artifacts_simultaneous_ci in cases/pvalues.py)."
        )
    df = df.copy()
    # avg_width/avg_score are per-rep AVERAGES already (avg_width = total_width
    # / n_reps at write time) -- reconstruct the totals so cells can be summed
    # across n/k/eval_type before re-dividing, matching how the live
    # SimultaneousCIResult-based aggregation works (weighted, not a mean of
    # means).
    df["total_width"] = df["avg_width"] * df["n_reps"]
    df["total_score"] = df["avg_score"] * df["n_reps"]
    return df


def _agg_row(rows: pd.DataFrame, condition: str) -> tuple[float, float, float, int, int]:
    """(coverage, width, score, covered_count, total_reps) for one condition."""
    sub = rows[rows["condition"] == condition] if condition == "alt" else rows[rows["condition"].isna()]
    t = int(sub["n_reps"].sum())
    if t == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    c = int(sub["all_covered"].sum())
    w = float(sub["total_width"].sum()) / t
    s = float(sub["total_score"].sum()) / t
    return c / t, w, s, c, t


def _method_order(df: pd.DataFrame) -> list[str]:
    present = set(df["ci_method"].unique())
    ordered = [m.name for m in ALL_SIMULTANEOUS_CI_METHODS if m.name in present]
    # Fall back to first-seen order for any method not in the canonical list
    # (e.g. a future addition this script hasn't been updated for).
    ordered += [m for m in dict.fromkeys(df["ci_method"]) if m not in ordered]
    return ordered


def build_overall_table(df: pd.DataFrame, *, alpha: float, label_suffix: str = "") -> str:
    """Reproduces latex_simultaneous_ci_overall_summary's by-n table."""
    target = 1.0 - alpha
    eval_types_present = set(df["eval_type"].unique())
    sizes_present = sorted(df["n"].unique())

    rows = []
    for cm in _method_order(df):
        c_rows = df[df["ci_method"] == cm]
        covered = set(c_rows["eval_type"].unique())
        cov_null, w_null, s_null, c_null, t_null = _agg_row(c_rows, "null")
        cov_alt, w_alt, s_alt, _, _ = _agg_row(c_rows, "alt")
        _, _, lo, hi = _mc_proportion_stats(c_null, t_null)
        row = [
            escape_latex(cm),
            f"{cov_null:.3f}" if np.isfinite(cov_null) else "-",
            f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
            f"{w_null:.4f}" if np.isfinite(w_null) else "-",
            f"{s_null:.4f}" if np.isfinite(s_null) else "-",
            f"{cov_alt:.3f}" if np.isfinite(cov_alt) else "-",
            f"{w_alt:.4f}" if np.isfinite(w_alt) else "-",
            f"{s_alt:.4f}" if np.isfinite(s_alt) else "-",
            eval_type_label(covered, eval_types_present),
        ]
        for n in sizes_present:
            cov_n, _, _, _, _ = _agg_row(c_rows[c_rows["n"] == n], "null")
            row.append(f"{cov_n:.3f}" if np.isfinite(cov_n) else "-")
        rows.append(row)

    return booktabs_table(
        caption=f"Simultaneous CI methods' family-wise coverage, average per-comparison width, "
                f"and average per-comparison interval score, by sample size (nominal coverage={target:.0%}).",
        label=f"tab:pvalues_simultaneous_ci_overall{label_suffix}",
        columns=["CI method", "Cov(null)", "95\\% MC band", "Width(null)", "Score(null)",
                 "Cov(alt)", "Width(alt)", "Score(alt)", "Eval types"]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
    )


def build_by_k_table(df: pd.DataFrame, *, alpha: float, label_suffix: str = "") -> str:
    """Same shape as build_overall_table, but columned by k instead of n
    (collapsed across n and eval type) -- surfaces the aggregate k-dependence
    build_overall_table's n-only columns can't show."""
    target = 1.0 - alpha
    eval_types_present = set(df["eval_type"].unique())
    ks_present = sorted(df["k"].unique())

    rows = []
    for cm in _method_order(df):
        c_rows = df[df["ci_method"] == cm]
        covered = set(c_rows["eval_type"].unique())
        cov_null, w_null, s_null, c_null, t_null = _agg_row(c_rows, "null")
        cov_alt, w_alt, s_alt, _, _ = _agg_row(c_rows, "alt")
        _, _, lo, hi = _mc_proportion_stats(c_null, t_null)
        row = [
            escape_latex(cm),
            f"{cov_null:.3f}" if np.isfinite(cov_null) else "-",
            f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
            f"{w_null:.4f}" if np.isfinite(w_null) else "-",
            f"{s_null:.4f}" if np.isfinite(s_null) else "-",
            f"{cov_alt:.3f}" if np.isfinite(cov_alt) else "-",
            f"{w_alt:.4f}" if np.isfinite(w_alt) else "-",
            f"{s_alt:.4f}" if np.isfinite(s_alt) else "-",
            eval_type_label(covered, eval_types_present),
        ]
        for k in ks_present:
            cov_k, _, _, _, _ = _agg_row(c_rows[c_rows["k"] == k], "null")
            row.append(f"{cov_k:.3f}" if np.isfinite(cov_k) else "-")
        rows.append(row)

    return booktabs_table(
        caption=f"Simultaneous CI methods' family-wise coverage, average per-comparison width, "
                f"and average per-comparison interval score, by number of arms compared $k$ "
                f"(nominal coverage={target:.0%}).",
        label=f"tab:pvalues_simultaneous_ci_by_k{label_suffix}",
        columns=["CI method", "Cov(null)", "95\\% MC band", "Width(null)", "Score(null)",
                 "Cov(alt)", "Width(alt)", "Score(alt)", "Eval types"]
                + [f"k={k}" for k in ks_present],
        rows=rows,
    )


def build_by_eval_type_and_k_table(df: pd.DataFrame, *, alpha: float, label_suffix: str = "") -> str:
    """One row per (eval type, CI method), collapsed across n, with one
    Cov(null) column per k value swept -- isolates exactly which (eval type,
    k) cells break down instead of averaging across both dimensions."""
    target = 1.0 - alpha
    eval_types_present = [et for et in ["binary", "continuous", "likert", "grades"] if et in set(df["eval_type"].unique())]
    ks_present = sorted(df["k"].unique())
    method_order = _method_order(df)

    rows = []
    for et in eval_types_present:
        et_df = df[df["eval_type"] == et]
        et_methods = [cm for cm in method_order if cm in set(et_df["ci_method"].unique())]
        for cm in et_methods:
            c_rows = et_df[et_df["ci_method"] == cm]
            cov_null, _, _, c_null, t_null = _agg_row(c_rows, "null")
            cov_alt, _, _, _, _ = _agg_row(c_rows, "alt")
            _, _, lo, hi = _mc_proportion_stats(c_null, t_null)
            row = [
                escape_latex(et), escape_latex(cm),
                f"{cov_null:.3f}" if np.isfinite(cov_null) else "-",
                f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
                f"{cov_alt:.3f}" if np.isfinite(cov_alt) else "-",
            ]
            for k in ks_present:
                cov_k, _, _, _, _ = _agg_row(c_rows[c_rows["k"] == k], "null")
                row.append(f"{cov_k:.3f}" if np.isfinite(cov_k) else "-")
            rows.append(row)

    return booktabs_table(
        caption=f"Simultaneous CI methods' family-wise coverage faceted by eval type, with one "
                f"Cov(null) column per number of arms compared $k$ (collapsed across sample size; "
                f"nominal coverage={target:.0%}). Isolates eval-type-by-$k$ interactions -- e.g. "
                f"boot/max\\_t's joint-bootstrap widening undercovering for binary data at low $k$ "
                f"-- that a table collapsed across both dimensions cannot show.",
        label=f"tab:pvalues_simultaneous_ci_by_eval_type_k{label_suffix}",
        columns=["Eval type", "CI method", "Cov(null)", "95\\% MC band", "Cov(alt)"]
                + [f"k={k}" for k in ks_present],
        rows=rows,
    )


def build_combined_type_k_table(
    df: pd.DataFrame, *, alpha: float, label_suffix: str = "", include_n_cols: bool = True,
) -> str:
    """Single-table version for the paper: every column from
    build_overall_table (Cov/Width/Score, null and alt) plus one Cov(null)
    column per k (collapsed across n) AND, when include_n_cols, one more
    Cov(null) column per n (collapsed across k) -- two independent one-way
    breakdowns side by side, not a full n-by-k cross-tab (which would need
    len(n)*len(k) columns and get unreadably wide). Eval type is folded into
    the CI method column itself ("boot (bin)"/"boot (num)", continuous+
    likert summed into "num" via eval_type_group -- matching the bin/numeric
    split latex_tables/revise_latex_tables already use elsewhere) rather
    than a separate column, with a \\midrule dividing the binary block from
    the numeric block. Replaces needing build_by_k_table +
    build_by_eval_type_and_k_table side by side to see the interaction."""
    target = 1.0 - alpha
    ks_present = sorted(df["k"].unique())
    sizes_present = sorted(df["n"].unique()) if include_n_cols else []
    method_order = _method_order(df)
    df = df.copy()
    df["group"] = df["eval_type"].map(eval_type_group)

    rows: list[list[str]] = []
    rule_before: set[int] = set()
    for group, suffix in [("binary", "bin"), ("numeric", "num")]:
        g_df = df[df["group"] == group]
        if g_df.empty:
            continue
        if rows:
            rule_before.add(len(rows))
        g_methods = [cm for cm in method_order if cm in set(g_df["ci_method"].unique())]
        for cm in g_methods:
            c_rows = g_df[g_df["ci_method"] == cm]
            cov_null, w_null, s_null, c_null, t_null = _agg_row(c_rows, "null")
            cov_alt, w_alt, s_alt, _, _ = _agg_row(c_rows, "alt")
            _, _, lo, hi = _mc_proportion_stats(c_null, t_null)
            row = [
                f"{escape_latex(cm)} ({suffix})",
                f"{cov_null:.3f}" if np.isfinite(cov_null) else "-",
                f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
                f"{w_null:.4f}" if np.isfinite(w_null) else "-",
                f"{s_null:.4f}" if np.isfinite(s_null) else "-",
                f"{cov_alt:.3f}" if np.isfinite(cov_alt) else "-",
                f"{w_alt:.4f}" if np.isfinite(w_alt) else "-",
                f"{s_alt:.4f}" if np.isfinite(s_alt) else "-",
            ]
            for k in ks_present:
                cov_k, _, _, _, _ = _agg_row(c_rows[c_rows["k"] == k], "null")
                row.append(f"{cov_k:.3f}" if np.isfinite(cov_k) else "-")
            for n in sizes_present:
                cov_n, _, _, _, _ = _agg_row(c_rows[c_rows["n"] == n], "null")
                row.append(f"{cov_n:.3f}" if np.isfinite(cov_n) else "-")
            rows.append(row)

    n_col_note = " and one Cov(null) column per sample size $n$ (collapsed across $k$)" if include_n_cols else ""
    return booktabs_table(
        caption=f"Simultaneous CI methods' family-wise coverage, average per-comparison width, and "
                f"average per-comparison interval score (nominal coverage={target:.0%}), with one "
                f"Cov(null) column per number of arms compared $k$ (collapsed across sample size)"
                f"{n_col_note}. Binary and numeric (continuous+likert) eval types are reported "
                f"separately -- each method appears as two rows, (bin) and (num) -- since boot/max\\_t's "
                f"joint-bootstrap widening undercovers specifically for binary data at low $k$ and "
                f"small $n$, an interaction a table averaged across eval types (or across $n$ or $k$) "
                f"cannot show.",
        label=f"tab:pvalues_simultaneous_ci_combined{label_suffix}",
        columns=["CI method", "Cov(null)", "95\\% MC band", "Width(null)", "Score(null)",
                 "Cov(alt)", "Width(alt)", "Score(alt)"] + [f"k={k}" for k in ks_present]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
        rule_before=rule_before,
    )


TABLE_BUILDERS = {
    "overall": build_overall_table,
    "by_k": build_by_k_table,
    "by_eval_type_k": build_by_eval_type_and_k_table,
    "combined": build_combined_type_k_table,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", help="Path to a *_simultaneous_ci_results.csv file.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level used for the run (default 0.05).")
    parser.add_argument("--tables", nargs="+", choices=list(TABLE_BUILDERS), default=list(TABLE_BUILDERS),
                         metavar="TABLE", help=f"Which tables to emit (default: all of {list(TABLE_BUILDERS)}).")
    parser.add_argument("--label-suffix", default="", help="Appended to every table's \\label{} (e.g. '_real' to avoid colliding with a synthetic-data run's tables in the same document).")
    parser.add_argument("--n-min", type=int, default=None, metavar="N", help="Restrict to rows with n >= N before building tables (e.g. --n-min 30 for a high-N-only cut, mirroring latex_simultaneous_ci_full_report's low-N/high-N split).")
    parser.add_argument("--n-max", type=int, default=None, metavar="N", help="Restrict to rows with n <= N before building tables (e.g. --n-max 30 for a low-N-only cut -- this is where boot/max_t's binary-extreme-proportion undercoverage is sharpest; see module docstring).")
    parser.add_argument("--no-n-cols", action="store_true", help="combined table only: omit the per-n Cov(null) columns (keep just per-k), if the full table is too wide for the page.")
    args = parser.parse_args(argv)

    try:
        df = _load(args.csv_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.n_min is not None:
        df = df[df["n"] >= args.n_min]
    if args.n_max is not None:
        df = df[df["n"] <= args.n_max]
    if df.empty:
        print("error: no rows left after applying --n-min/--n-max", file=sys.stderr)
        return 1

    blocks = []
    for name in args.tables:
        kwargs = {"alpha": args.alpha, "label_suffix": args.label_suffix}
        if name == "combined":
            kwargs["include_n_cols"] = not args.no_n_cols
        blocks.append(TABLE_BUILDERS[name](df, **kwargs))
    print("\n\n".join(revise_table(block) for block in blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
