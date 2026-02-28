"""Bootstrap ranking analysis for prompt templates.

Provides rank distributions and mean advantage calculations that respect
the paired structure of benchmark data (same inputs across all templates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RankDistribution:
    """Bootstrap distribution over template rankings.

    Attributes
    ----------
    labels : list[str]
        Template labels.
    rank_probs : np.ndarray
        Shape (N_templates, N_templates). Entry [i, r] is the probability
        that template i achieves rank r (0-indexed, 0 = best).
    expected_ranks : np.ndarray
        Shape (N_templates,). Expected rank for each template (1-indexed).
    p_best : np.ndarray
        Shape (N_templates,). Probability each template is ranked first.
    n_bootstrap : int
        Number of bootstrap iterations used.
    """

    labels: list[str]
    rank_probs: np.ndarray
    expected_ranks: np.ndarray
    p_best: np.ndarray
    n_bootstrap: int


@dataclass
class MeanAdvantageResult:
    """Mean advantage of each template over a reference, with uncertainty.

    This is the core data structure for the mean advantage plot. It separates
    two kinds of uncertainty:

    - **Epistemic (CI on the mean)**: Would shrink with more benchmark inputs.
      Captured by bootstrap_ci_low/high.
    - **Intrinsic (score spread)**: A property of the template, won't shrink
      with more data. Captured by spread_low/high (percentiles of per-input
      advantages).

    Attributes
    ----------
    labels : list[str]
        Template labels.
    mean_advantages : np.ndarray
        Shape (N,). Mean advantage over reference for each template.
    bootstrap_ci_low : np.ndarray
        Shape (N,). Lower bound of bootstrap CI on the mean advantage.
    bootstrap_ci_high : np.ndarray
        Shape (N,). Upper bound of bootstrap CI on the mean advantage.
    spread_low : np.ndarray
        Shape (N,). Lower percentile of per-input advantage distribution.
    spread_high : np.ndarray
        Shape (N,). Upper percentile of per-input advantage distribution.
    reference : str
        Description of the reference used (e.g., 'grand_mean' or a template label).
    per_input_advantages : np.ndarray
        Shape (N, M). Raw per-input advantages for each template.
    n_bootstrap : int
        Number of bootstrap iterations used.
    spread_percentiles : tuple[float, float]
        The percentiles used for spread_low/high (e.g., (10, 90)).
    """

    labels: list[str]
    mean_advantages: np.ndarray
    bootstrap_ci_low: np.ndarray
    bootstrap_ci_high: np.ndarray
    spread_low: np.ndarray
    spread_high: np.ndarray
    reference: str
    per_input_advantages: np.ndarray
    n_bootstrap: int
    spread_percentiles: tuple[float, float]


def bootstrap_ranks(
    scores: np.ndarray,
    labels: list[str],
    n_bootstrap: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> RankDistribution:
    """Compute bootstrap distribution over template rankings.

    Resamples inputs with replacement, computes mean score per template
    on the resampled set, and records the full ranking. This captures
    uncertainty about rankings due to the specific choice of benchmark inputs.

    Parameters
    ----------
    scores : np.ndarray
        2D score matrix of shape (N_templates, M_inputs).
    labels : list[str]
        Template labels.
    n_bootstrap : int
        Number of bootstrap iterations.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    RankDistribution
    """
    if rng is None:
        rng = np.random.default_rng()

    n_templates, m_inputs = scores.shape
    rank_counts = np.zeros((n_templates, n_templates), dtype=np.int64)

    for _ in range(n_bootstrap):
        idx = rng.choice(m_inputs, size=m_inputs, replace=True)
        means = scores[:, idx].mean(axis=1)
        # argsort descending: best (highest mean) gets rank 0
        order = np.argsort(-means)
        for rank, template_idx in enumerate(order):
            rank_counts[template_idx, rank] += 1

    rank_probs = rank_counts / n_bootstrap
    # Expected rank (1-indexed for human readability)
    expected_ranks = (rank_probs * np.arange(1, n_templates + 1)).sum(axis=1)
    p_best = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_bootstrap,
    )


def bootstrap_mean_advantage(
    scores: np.ndarray,
    labels: list[str],
    reference: str = "grand_mean",
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    spread_percentiles: tuple[float, float] = (10, 90),
    rng: Optional[np.random.Generator] = None,
) -> MeanAdvantageResult:
    """Compute mean advantage over a reference with dual uncertainty bands.

    For each template, computes its per-input advantage over a reference
    (either the grand mean across all templates, or a specific baseline
    template). Then estimates:

    1. Bootstrap CI on the mean advantage (epistemic uncertainty).
    2. Percentile spread of per-input advantages (intrinsic variance).

    Parameters
    ----------
    scores : np.ndarray
        2D score matrix of shape (N_templates, M_inputs).
    labels : list[str]
        Template labels.
    reference : str
        Either 'grand_mean' (default) to compare against the per-input
        average across all templates, or a template label to compare
        against that specific template.
    n_bootstrap : int
        Number of bootstrap iterations for CI estimation.
    ci : float
        Confidence level for the bootstrap CI (default 0.95).
    spread_percentiles : tuple[float, float]
        Percentiles for the intrinsic variance band (default (10, 90)).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    MeanAdvantageResult
    """
    if rng is None:
        rng = np.random.default_rng()

    n_templates, m_inputs = scores.shape
    alpha = 1 - ci

    # Compute per-input advantages
    if reference == "grand_mean":
        ref_scores = scores.mean(axis=0)  # shape (M,)
        ref_label = "grand_mean"
    else:
        ref_idx = labels.index(reference)
        ref_scores = scores[ref_idx]  # shape (M,)
        ref_label = reference

    # advantages[i, m] = how much better template i is than the reference on input m
    advantages = scores - ref_scores[np.newaxis, :]  # shape (N, M)

    # Point estimates
    mean_adv = advantages.mean(axis=1)  # shape (N,)

    # Intrinsic spread: percentiles of per-input advantages
    spread_low = np.percentile(advantages, spread_percentiles[0], axis=1)
    spread_high = np.percentile(advantages, spread_percentiles[1], axis=1)

    # Bootstrap CI on the mean advantage
    boot_means = np.empty((n_templates, n_bootstrap))
    for b in range(n_bootstrap):
        idx = rng.choice(m_inputs, size=m_inputs, replace=True)
        boot_means[:, b] = advantages[:, idx].mean(axis=1)

    ci_low = np.percentile(boot_means, 100 * alpha / 2, axis=1)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=1)

    return MeanAdvantageResult(
        labels=labels,
        mean_advantages=mean_adv,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        spread_low=spread_low,
        spread_high=spread_high,
        reference=ref_label,
        per_input_advantages=advantages,
        n_bootstrap=n_bootstrap,
        spread_percentiles=spread_percentiles,
    )
