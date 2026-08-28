"""Paired-vs-unpaired experimental design detection.

Kept as a standalone leaf module (zero dependencies beyond pandas/numpy) so
both ``evalstats.labeling`` (the CLI sampling helper) and ``evalstats.api``
(``compare()``'s design auto-detection) can share exactly one implementation
without either importing the other -- same rationale as ``core/bundles.py``'s
own split, just for this one function instead of a family of dataclasses.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def detect_paired(df: pd.DataFrame, factor_col: Optional[str], item_col: str) -> bool:
    """True when item ids are (largely) shared across every factor level --
    the structure evalstats' loader assumes when building an item-aligned
    score matrix, and the common case for prompt/model comparisons. False
    when item pools are substantially disjoint per level (e.g. a genuinely
    between-subjects design with no shared item id).

    Uses a 90% overlap-with-the-full-item-universe threshold per level
    rather than requiring exact equality, since a few missing/dropped rows
    per condition shouldn't flip the detected design.
    """
    if factor_col is None or factor_col not in df.columns:
        return True
    levels = df[factor_col].dropna().unique()
    if len(levels) <= 1:
        return True
    item_sets = [set(df.loc[df[factor_col] == lvl, item_col].dropna()) for lvl in levels]
    universe = set.union(*item_sets) if item_sets else set()
    if not universe:
        return True
    overlaps = [len(s) / len(universe) for s in item_sets]
    return min(overlaps) >= 0.9
