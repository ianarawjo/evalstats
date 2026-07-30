"""Summarize a ``*_ppi_label_efficiency_results.csv`` (see
``cases/pvalues.py``'s ``run_ppi_label_efficiency_check``/``LabelEfficiencyPoint``)
into a compact LaTeX booktabs table of the label-efficiency multiplier
(``equiv_n_lab / n_lab``), suitable for the main body of a paper -- the full
N_lab x alignment-target factorial belongs in an appendix figure/table; this
is the "headline numbers" cut of it.

One row-group per eval_type present in the CSV (binary/continuous/likert, in
that fixed reporting order), one row per requested alignment target within a
group, one column per requested N_lab checkpoint. Cells show the multiplier
as "X.XXx"; a cell whose (eval_type, alignment_target, n_lab) combination
hit ``saturated=True`` (the classical human-only reference power curve was
already at its ceiling -- see LabelEfficiencyPoint.saturated's docstring) is
rendered as a "$\\geq$"-prefixed lower bound with a dagger, per this paper's
own \\S\\ref{sec:label-efficiency} caveat that saturated multipliers must be
flagged rather than presented as exact.

Rows are labeled by the REALIZED alignment_value (rounded), not the
alignment_target requested when the sweep was designed, matching
LabelEfficiencyPoint.alignment_value's own docstring ("plots/tables should
label by this, not the target, to avoid implying more precision than the
calibration has") -- the two will usually print identically at 2dp, but this
keeps the table honest about which number it's actually reporting.

Usage:
  python -m simulations.harness.label_efficiency_table path/to/..._results.csv
  python -m simulations.harness.label_efficiency_table path/to/..._results.csv \\
      --log path/to/..._summary.log \\
      --n-lab-cols 30,90,200 --alignment-targets 0.8,0.5,0.3 \\
      --caption "..." --label "tab:label_efficiency" --table-star
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

from .latex_tables import booktabs_table

REQUIRED_COLUMNS = {
    "eval_type", "alignment_metric", "alignment_target", "alignment_value",
    "n_lab", "n_reps", "equiv_n_lab", "multiplier", "saturated",
}

EVAL_TYPE_ORDER = ["binary", "continuous", "likert"]
EVAL_TYPE_LABEL = {"binary": "Binary", "continuous": "Continuous", "likert": "Likert"}
ALIGNMENT_METRIC_LABEL = {
    "kappa": r"Cohen's $\kappa$",
    "pearson_r": r"Pearson $r$",
    "weighted_kappa": r"weighted $\kappa$",
}

DEFAULT_ALIGNMENT_TARGETS = [0.8, 0.5, 0.3]
DEFAULT_N_LAB_COLS = [30, 90, 200]

_LOG_N_TOTAL_RE = re.compile(r"N=(\d+) total items")


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {sorted(missing)}")
    df["saturated"] = df["saturated"].astype(str).str.strip().str.lower() == "true"
    df["alignment_target"] = df["alignment_target"].astype(float).round(2)
    df["n_lab"] = df["n_lab"].astype(int)
    return df


def n_total_from_log(log_path: str) -> int | None:
    """Best-effort extraction of the fixed total-item count N from a
    ``run_ppi_label_efficiency_check`` summary log's own header line (e.g.
    "N=1000 total items throughout; only N_lab ... varies"). Returns None
    (rather than raising) if the log doesn't have this line, since the
    caption falls back to omitting the total-N mention."""
    try:
        with open(log_path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    m = _LOG_N_TOTAL_RE.search(text)
    return int(m.group(1)) if m else None


def format_multiplier(multiplier: float, saturated: bool) -> str:
    s = f"{multiplier:.2f}$\\times$"
    return f"$\\geq${s}$^\\dagger$" if saturated else s


def build_table(
    df: pd.DataFrame,
    *,
    alignment_targets: list[float],
    n_lab_cols: list[int],
    caption: str,
    label: str,
    table_star: bool,
) -> str:
    eval_types = [et for et in EVAL_TYPE_ORDER if et in set(df["eval_type"])]
    if not eval_types:
        raise ValueError("No known eval_type (binary/continuous/likert) found in CSV.")

    columns = ["Alignment"] + [rf"$N_\mathrm{{lab}}={n}$" for n in n_lab_cols]
    rows: list[list[str]] = []
    rule_before: set[int] = set()
    any_saturated = False
    min_n_reps: int | None = None

    for gi, et in enumerate(eval_types):
        et_df = df[df["eval_type"] == et]
        metric = et_df["alignment_metric"].iloc[0]
        metric_label = ALIGNMENT_METRIC_LABEL.get(metric, metric)
        if gi > 0:
            rule_before.add(len(rows))
        rows.append([rf"\multicolumn{{{len(columns)}}}{{l}}{{\textbf{{{EVAL_TYPE_LABEL[et]}}} (target {metric_label})}}"])

        for target in alignment_targets:
            target_df = et_df[(et_df["alignment_target"] - target).abs() < 1e-6]
            if target_df.empty:
                print(f"warning: no rows for eval_type={et!r}, alignment_target={target} -- skipping row", file=sys.stderr)
                continue
            realized = float(target_df["alignment_value"].iloc[0])
            row = [f"{realized:.2f}"]
            for n_lab in n_lab_cols:
                cell_df = target_df[target_df["n_lab"] == n_lab]
                if cell_df.empty:
                    print(f"warning: no row for eval_type={et!r}, target={target}, n_lab={n_lab} -- blank cell", file=sys.stderr)
                    row.append("--")
                    continue
                r = cell_df.iloc[0]
                row.append(format_multiplier(float(r["multiplier"]), bool(r["saturated"])))
                any_saturated = any_saturated or bool(r["saturated"])
                n_reps = int(r["n_reps"])
                min_n_reps = n_reps if min_n_reps is None else min(min_n_reps, n_reps)
            rows.append(row)

    full_caption = caption
    if min_n_reps is not None:
        full_caption += rf" Each cell from $\ge${min_n_reps} Monte Carlo replicates."
    if any_saturated:
        full_caption += (
            r" $^\dagger$: the human-only reference power curve was already saturated "
            r"at this $N_\mathrm{lab}$; the reported value is a lower bound on the true multiplier, not an exact estimate."
        )

    table = booktabs_table(
        caption=full_caption,
        label=label,
        columns=columns,
        rows=rows,
        col_align="l" + "r" * len(n_lab_cols),
        rule_before=rule_before,
    )
    if table_star:
        table = table.replace(r"\begin{table}", r"\begin{table*}").replace(r"\end{table}", r"\end{table*}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to a *_ppi_label_efficiency_results.csv")
    parser.add_argument("--log", default=None, help="Optional matching *_summary.log, used only to read the fixed total-N for the caption")
    parser.add_argument("--alignment-targets", default=",".join(str(x) for x in DEFAULT_ALIGNMENT_TARGETS),
                         help=f"Comma-separated alignment targets (rows), default {DEFAULT_ALIGNMENT_TARGETS}")
    parser.add_argument("--n-lab-cols", default=",".join(str(x) for x in DEFAULT_N_LAB_COLS),
                         help=f"Comma-separated N_lab checkpoints (columns), default {DEFAULT_N_LAB_COLS}")
    parser.add_argument("--caption", default=None, help="Table caption (a sensible default is generated if omitted)")
    parser.add_argument("--label", default="tab:label_efficiency", help="LaTeX \\label{} key")
    parser.add_argument("--table-star", action="store_true", help="Emit a table* (full-width, two-column) float instead of table")
    args = parser.parse_args()

    df = load_results(args.csv_path)
    alignment_targets = _parse_float_list(args.alignment_targets)
    n_lab_cols = _parse_int_list(args.n_lab_cols)

    caption = args.caption
    if caption is None:
        n_total = n_total_from_log(args.log) if args.log else None
        of_n = rf" out of $N={n_total}$ judge-scored items" if n_total else ""
        caption = (
            r"Label-efficiency multiplier ($N_\mathrm{lab}'/N_\mathrm{lab}$): the factor by which PPI-corrected "
            rf"testing with $N_\mathrm{{lab}}$ human labels{of_n} behaves like a larger human-only sample, "
            r"at several judge-alignment levels."
        )

    table = build_table(
        df, alignment_targets=alignment_targets, n_lab_cols=n_lab_cols,
        caption=caption, label=args.label, table_star=args.table_star,
    )
    print(table)


if __name__ == "__main__":
    main()
