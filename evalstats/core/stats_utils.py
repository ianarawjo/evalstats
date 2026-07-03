"""Shared statistical helper utilities.

Centralizes small, reusable pieces of statistical logic so analysis modules
can share behavior and avoid copy/paste drift.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np


def rescaled_ci(
    fn: Callable[..., tuple[float, float]],
    values: np.ndarray,
    alpha: float,
    lo: float,
    hi: float,
    **kwargs,
) -> tuple[float, float]:
    """Apply a CI function whose domain/prior assumes a [0, 1] scale to data
    on an arbitrary [lo, hi] scale.

    Linearly maps ``values`` onto [0, 1], calls ``fn(scaled_values, alpha,
    **kwargs)``, then maps the returned interval back onto [lo, hi]. Useful
    for methods whose defaults are only valid on [0, 1] -- e.g.
    ``nig_ci_1d``'s default prior assumes the mean is near the scale's
    midpoint, and ``logit_t_ci_1d`` requires values strictly in [0, 1] -- when
    applied to data on a different natural scale (a 1-5 Likert scale, a 0-100
    grade scale, or a signed paired difference of two [0, 1] scores, for
    which ``lo=-1, hi=1`` correctly re-centers the prior at 0 instead of the
    raw [0, 1] default of 0.5).

    Parameters
    ----------
    fn : Callable[[np.ndarray, float, ...], tuple[float, float]]
        A CI function with signature ``(values, alpha, **kwargs) -> (low, high)``.
    values : np.ndarray
        1-D array of observed values, assumed to lie within ``[lo, hi]``.
    alpha : float
        Significance level, passed through to ``fn``.
    lo, hi : float
        The scale's known bounds.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        The interval, mapped back onto ``[lo, hi]``.
    """
    span = hi - lo
    scaled = (np.asarray(values, dtype=float) - lo) / span
    ci_low, ci_high = fn(scaled, alpha, **kwargs)
    return ci_low * span + lo, ci_high * span + lo


def interval_score(low: float, high: float, true_value: float, alpha: float) -> float:
    """Winkler/interval score: a single number combining CI width and coverage.

    ``score = (high - low) + (2/alpha) * miss_distance``, where ``miss_distance``
    is 0 if ``true_value`` falls inside ``[low, high]`` and otherwise the
    distance from ``true_value`` to the nearer bound. Lower is better.

    This is a strictly proper scoring rule (Gneiting & Raftery 2007): its
    expectation is uniquely minimized when ``[low, high]`` is the true nominal
    ``1 - alpha`` interval, so a method can't improve its expected score by
    reporting an interval narrower or wider than its genuine uncertainty. The
    penalty is asymmetric in practice: reporting too wide only ever costs the
    plain width term (a flat, linear tax), while reporting too narrow risks
    the ``2/alpha`` miss penalty (40x the miss distance at a 95% level) every
    time it fails to cover -- so it punishes overconfidence more severely than
    excess caution of the same magnitude.

    Parameters
    ----------
    low, high : float
        CI bounds (``low <= high``).
    true_value : float
        The estimand the interval is meant to cover.
    alpha : float
        Significance level the interval was built for (e.g. 0.05 for a 95% CI).

    Returns
    -------
    float
    """
    width = high - low
    if true_value < low:
        return width + (2.0 / alpha) * (low - true_value)
    if true_value > high:
        return width + (2.0 / alpha) * (true_value - high)
    return width


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
