"""Robustness and variance metrics for prompt templates.

Quantifies how consistent each template's performance is across inputs.
A template can be "best on average" but highly volatile — these metrics
make that tradeoff explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RobustnessResult:
    """Per-template robustness metrics.

    Attributes
    ----------
    labels : list[str]
        Template labels.
    mean : np.ndarray
        Mean score per template.
    std : np.ndarray
        Standard deviation per template.
    cv : np.ndarray
        Coefficient of variation (std / mean). NaN if mean is zero.
    iqr : np.ndarray
        Interquartile range per template.
    cvar_10 : np.ndarray
        Conditional Value at Risk: mean of the worst 10% of scores.
    percentiles : dict[int, np.ndarray]
        Score percentiles (10, 25, 50, 75, 90) per template.
    failure_rate : Optional[np.ndarray]
        Fraction of inputs below threshold, if threshold was specified.
    failure_threshold : Optional[float]
        The threshold used for failure_rate.
    """

    labels: list[str]
    mean: np.ndarray
    std: np.ndarray
    cv: np.ndarray
    iqr: np.ndarray
    cvar_10: np.ndarray
    percentiles: dict[int, np.ndarray]
    failure_rate: Optional[np.ndarray]
    failure_threshold: Optional[float]

    def summary_table(self):
        """Return a pandas DataFrame summarizing all metrics."""
        import pandas as pd

        data = {
            "template": self.labels,
            "mean": self.mean,
            "std": self.std,
            "cv": self.cv,
            "iqr": self.iqr,
            "cvar_10": self.cvar_10,
            "p10": self.percentiles[10],
            "p25": self.percentiles[25],
            "p50": self.percentiles[50],
            "p75": self.percentiles[75],
            "p90": self.percentiles[90],
        }
        if self.failure_rate is not None:
            data[f"failure_rate (<{self.failure_threshold})"] = self.failure_rate

        return pd.DataFrame(data).set_index("template")


def robustness_metrics(
    scores: np.ndarray,
    labels: list[str],
    failure_threshold: Optional[float] = None,
) -> RobustnessResult:
    """Compute robustness metrics for each template.

    Parameters
    ----------
    scores : np.ndarray
        2D score matrix of shape (N_templates, M_inputs).
    labels : list[str]
        Template labels.
    failure_threshold : float, optional
        If provided, computes the fraction of inputs scoring below this value.

    Returns
    -------
    RobustnessResult
    """
    n_templates, m_inputs = scores.shape

    mean = scores.mean(axis=1)
    std = scores.std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean != 0, std / np.abs(mean), np.nan)

    p10 = np.percentile(scores, 10, axis=1)
    p25 = np.percentile(scores, 25, axis=1)
    p50 = np.percentile(scores, 50, axis=1)
    p75 = np.percentile(scores, 75, axis=1)
    p90 = np.percentile(scores, 90, axis=1)
    iqr = p75 - p25

    # CVaR (Expected Shortfall): mean of the worst 10% of scores
    cvar_10 = np.empty(n_templates)
    k = max(1, int(np.floor(m_inputs * 0.10)))
    for i in range(n_templates):
        sorted_scores = np.sort(scores[i])
        cvar_10[i] = sorted_scores[:k].mean()

    failure_rate = None
    if failure_threshold is not None:
        failure_rate = (scores < failure_threshold).mean(axis=1)

    return RobustnessResult(
        labels=labels,
        mean=mean,
        std=std,
        cv=cv,
        iqr=iqr,
        cvar_10=cvar_10,
        percentiles={10: p10, 25: p25, 50: p50, 75: p75, 90: p90},
        failure_rate=failure_rate,
        failure_threshold=failure_threshold,
    )
