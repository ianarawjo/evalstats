"""Shared LaTeX booktabs-table formatting for --latex CLI output.

Each case's `save_results_artifacts*` appends one of these tables to the same
`*_summary.log` file it already writes (not a separate artifact), so the
official-test manifest doesn't need new paths. Cases pass in numbers they've
already aggregated for the plain-text report -- this module only formats.
"""

from __future__ import annotations

NUMERIC_EVAL_TYPES = {"continuous", "likert", "grades"}


def escape_latex(s: str) -> str:
    return str(s).replace("_", r"\_")


def eval_type_label(covered: set[str], all_present: set[str]) -> str:
    """Summarize which eval types a row's data actually covers.

    'all' if it covers everything present in this run; 'numeric' if it covers
    exactly the numeric (non-binary) eval types present; 'binary' if it's
    binary-only; otherwise an explicit comma-separated list.
    """
    if not covered:
        return "-"
    if covered >= all_present:
        return "all"
    numeric_present = NUMERIC_EVAL_TYPES & all_present
    if covered == numeric_present and covered:
        return "numeric"
    if covered == {"binary"}:
        return "binary"
    return ", ".join(sorted(covered))


def booktabs_table(*, caption: str, label: str, columns: list[str], rows: list[list[str]], col_align: str | None = None) -> str:
    """Assemble a booktabs-style LaTeX table from pre-formatted string cells."""
    if col_align is None:
        col_align = "l" + "r" * (len(columns) - 1)
    header = " & ".join(columns) + r" \\"
    body_lines = [" & ".join(row) + r" \\" for row in rows]
    body = "\n".join(body_lines)
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{col_align}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
