"""Shared statistical helper utilities.

Centralizes small, reusable pieces of statistical logic so analysis modules
can share behavior and avoid copy/paste drift.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def correct_pvalues(
    p_values: np.ndarray,
    method: Literal["holm", "bonferroni", "fdr_bh"],
) -> np.ndarray:
    """Apply multiple-comparisons correction to p-values."""
    n = len(p_values)
    if n <= 1:
        return p_values.copy()

    if method == "bonferroni":
        return np.minimum(p_values * n, 1.0)

    if method == "holm":
        order = np.argsort(p_values)
        adjusted = np.empty(n)
        cummax = 0.0
        for rank, idx in enumerate(order):
            corrected = p_values[idx] * (n - rank)
            cummax = max(cummax, corrected)
            adjusted[idx] = min(cummax, 1.0)
        return adjusted

    if method == "fdr_bh":
        order = np.argsort(p_values)
        adjusted = np.empty(n)
        cummin = 1.0
        for rank in range(n - 1, -1, -1):
            idx = order[rank]
            corrected = p_values[idx] * n / (rank + 1)
            cummin = min(cummin, corrected)
            adjusted[idx] = min(cummin, 1.0)
        return adjusted

    raise ValueError(f"Unknown correction method: {method}")
