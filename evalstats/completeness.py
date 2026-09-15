"""Keep only the items scored in every cell a paired comparison needs."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import pandas as pd

from evalstats.core.report import json_safe
from evalstats.loader import EvalLoadError, EvalResults


@dataclass
class CompletenessReport:
    """Which items :func:`complete_items` kept, and the cells the others lack."""

    axes: list[str]
    n_items: int
    n_excluded: int
    excluded_items: list[str]
    missing_cells: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return json_safe({
            "axes": self.axes,
            "n_items": self.n_items,
            "n_excluded": self.n_excluded,
            "excluded_items": self.excluded_items,
            "missing_cells": self.missing_cells,
        })


def complete_items(
    evaldata: EvalResults,
    factors: Union[str, list[str]],
    *,
    metric: Optional[str] = None,
) -> tuple[EvalResults, CompletenessReport]:
    """Drop items that lack a score for any cell of the paired design.

    A cell is one combination of the ``factors`` levels, the levels of any
    model or prompt column that varies (``compare()`` treats those as an axis
    too), and the run. The returned data is still paired; only whole items
    are removed.

    Parameters
    ----------
    evaldata : EvalResults
        Data from :func:`~evalstats.load_from`.
    factors : str or list[str]
        The factors you will pass to ``compare()``.
    metric : str, optional
        Metric column to check. Defaults to the first detected metric.

    Returns
    -------
    (EvalResults, CompletenessReport)
    """
    df = evaldata._df
    col = evaldata._col
    metric_col = metric or evaldata._metric_cols[0]
    if metric_col not in df.columns:
        raise EvalLoadError(f"metric column '{metric_col}' not found in data.")
    item_col = col["item"]

    def _column(f: str) -> Optional[str]:
        if f == "model":
            return col.get("model")
        if f in ("prompt", "template"):
            return col.get("prompt")
        return f

    axes: list[str] = []
    for f in [factors] if isinstance(factors, str) else list(factors):
        c = _column(f)
        if not c or c not in df.columns:
            raise EvalLoadError(
                f"Factor column for {f!r} not found in data. Available columns: {list(df.columns)}"
            )
        if c not in axes:
            axes.append(c)
    for role in ("model", "prompt"):
        c = col.get(role)
        if c and c in df.columns and c not in axes and df[c].nunique() >= 2:
            axes.append(c)
    run_col = col.get("run")
    cell_cols = axes + ([run_col] if run_col and run_col in df.columns else [])

    expected = list(itertools.product(*(pd.unique(df[c]) for c in cell_cols)))
    scored = df[df[metric_col].notna()]
    present: dict[Any, set] = {}
    for key in zip(scored[item_col], *(scored[c] for c in cell_cols)):
        present.setdefault(key[0], set()).add(key[1:])

    excluded: list = []
    missing_cells: dict[str, list[dict[str, Any]]] = {}
    for item in pd.unique(df[item_col]):
        have = present.get(item, set())
        missing = [combo for combo in expected if combo not in have]
        if missing:
            excluded.append(item)
            missing_cells[str(item)] = [dict(zip(cell_cols, combo)) for combo in missing]

    kept = df[~df[item_col].isin(excluded)].reset_index(drop=True)
    filtered = EvalResults(
        kept,
        score_types=dict(evaldata._score_types),
        metric_cols=list(evaldata._metric_cols),
        col=dict(col),
        factor_cols=list(evaldata._factor_cols),
        declared_factors=list(evaldata._declared_factors),
        declared_score_types=dict(evaldata._declared_score_types),
    )
    report = CompletenessReport(
        axes=cell_cols,
        n_items=int(kept[item_col].nunique()),
        n_excluded=len(excluded),
        excluded_items=[str(i) for i in excluded],
        missing_cells=missing_cells,
    )
    return filtered, report
