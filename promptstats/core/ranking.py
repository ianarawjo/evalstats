"""Bootstrap ranking analysis for prompt templates.

Provides rank distributions and mean advantage calculations that respect
the paired structure of benchmark data (same inputs across all templates).

When the score array includes a runs axis (R >= 3), all bootstrap functions
use a two-level (nested) resample: inputs are resampled in the outer level,
and runs within each selected input are resampled in the inner level.  This
correctly propagates seed variance into rank and CI estimates instead of
treating per-run cell means as fixed observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .resampling import (
    _weighted_median,
    bayes_bootstrap_resample_cell_means_once,
    smooth_bootstrap_resample_cell_means_once,
    nested_resample_cell_means_once,
    resolve_resampling_method,
)


def _accumulate_tie_aware_rank_mass(rank_counts: np.ndarray, agg: np.ndarray) -> None:
    """Accumulate one bootstrap draw of rank mass with fair tie handling.

    For each tie block of size ``t`` occupying ranks ``[r, r+t-1]``, each tied
    template receives ``1/t`` mass at each occupied rank. This removes the
    deterministic first-index tie bias introduced by ``np.argsort``.
    """
    order = np.argsort(-agg, kind="mergesort")
    sorted_scores = agg[order]

    start = 0
    n_templates = len(order)
    while start < n_templates:
        end = start + 1
        while end < n_templates and sorted_scores[end] == sorted_scores[start]:
            end += 1

        tie_indices = order[start:end]
        tie_size = end - start
        share = 1.0 / tie_size
        rank_counts[tie_indices, start:end] += share
        start = end


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


def bootstrap_ranks(
    scores: np.ndarray,
    labels: list[str],
    n_bootstrap: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto", "bayes_binary", "permutation"] = "auto",
    statistic: Literal["mean", "median"] = "mean",
) -> RankDistribution:
    """Compute bootstrap distribution over template rankings.

    Parameters
    ----------
    scores : np.ndarray
        Score array of shape ``(N, M)`` or ``(N, M, R)``.
        When ``R >= 3`` a two-level nested bootstrap is used.
        When ``R < 3`` (or 2-D input) the standard single-level resample
        is used.
    labels : list[str]
        Template labels.
    n_bootstrap : int
        Number of bootstrap iterations.
    method : str
        Resampling method for API consistency: ``'bootstrap'``, ``'bca'``,
        ``'bayes_bootstrap'``, ``'smooth_bootstrap'``, or ``'auto'``.  Rank
        distributions use multinomial (``'bootstrap'``/``'bca'``),
        Dirichlet (``'bayes_bootstrap'``), or smoothed KDE
        (``'smooth_bootstrap'``) outer weights. ``'auto'`` resolves to
        ``'smooth_bootstrap'``.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    statistic : str
        Statistic used to aggregate scores across inputs when determining
        template rankings per bootstrap resample: ``'median'`` (default)
        or ``'mean'``.

    Returns
    -------
    RankDistribution
    """
    if rng is None:
        rng = np.random.default_rng()

    if method not in {"bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto", "bayes_binary", "permutation"}:
        raise ValueError(f"Unknown method: {method}")

    # Rank distribution does not use a special Bayesian binary model;
    # treat bayes_binary as smooth_bootstrap for ranking purposes.
    # Permutation is a p-value method for pairwise tests; rank distributions
    # still use bootstrap-style resampling.
    if method == "bayes_binary":
        effective_method = "smooth_bootstrap"
    elif method == "permutation":
        effective_method = "bootstrap"
    else:
        effective_method = method
    m_inputs = scores.shape[1]
    resolved_method = resolve_resampling_method(effective_method, m_inputs)

    # ------------------------------------------------------------------ #
    # Seeded path (R >= 3)                                                #
    # ------------------------------------------------------------------ #
    if scores.ndim == 3 and scores.shape[2] >= 3:
        if resolved_method == "bayes_bootstrap":
            return _bayes_bootstrap_ranks_seeded(scores, labels, n_bootstrap, rng, statistic=statistic)
        if resolved_method == "smooth_bootstrap":
            return _smooth_bootstrap_ranks_seeded(scores, labels, n_bootstrap, rng, statistic=statistic)
        return _bootstrap_ranks_seeded(scores, labels, n_bootstrap, rng, statistic=statistic)

    # ------------------------------------------------------------------ #
    # Standard path (2-D or R < 3)                                        #
    # ------------------------------------------------------------------ #
    if scores.ndim == 3:
        scores = scores.mean(axis=2)  # collapse small run axis

    n_templates, m_inputs = scores.shape
    rank_counts = np.zeros((n_templates, n_templates), dtype=float)

    if resolved_method == "bayes_bootstrap":
        # Dirichlet-weighted aggregation per template instead of
        # multinomial input resampling.
        exp_mat = rng.exponential(1.0, size=(n_bootstrap, m_inputs))   # (B, M)
        weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)         # (B, M)
        if statistic == "median":
            for b in range(n_bootstrap):
                agg = np.array([_weighted_median(scores[t], weights[b]) for t in range(n_templates)])
                _accumulate_tie_aware_rank_mass(rank_counts, agg)
        else:
            for b in range(n_bootstrap):
                agg = scores @ weights[b]                               # (N,)
                _accumulate_tie_aware_rank_mass(rank_counts, agg)
    elif resolved_method == "smooth_bootstrap":
        # KDE-smoothed resample: resample inputs with replacement + add Gaussian noise.
        from scipy.stats import gaussian_kde as _kde
        # Compute per-template bandwidths from the M cell means.
        bws = np.zeros(n_templates)
        for t in range(n_templates):
            std_t = float(np.std(scores[t], ddof=1)) if m_inputs > 1 else 0.0
            if std_t > 0.0 and m_inputs >= 2:
                bws[t] = float(_kde(scores[t]).factor * std_t)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, m_inputs, size=m_inputs)
            samples = scores[:, idx].copy()                            # (N, M)
            for t in range(n_templates):
                if bws[t] > 0.0:
                    samples[t] += rng.normal(0.0, bws[t], size=m_inputs)
            if statistic == "median":
                agg = np.median(samples, axis=1)
            else:
                agg = samples.mean(axis=1)
            _accumulate_tie_aware_rank_mass(rank_counts, agg)
    elif statistic == "median":
        for _ in range(n_bootstrap):
            idx = rng.choice(m_inputs, size=m_inputs, replace=True)
            agg = np.median(scores[:, idx], axis=1)
            _accumulate_tie_aware_rank_mass(rank_counts, agg)
    else:
        for _ in range(n_bootstrap):
            idx = rng.choice(m_inputs, size=m_inputs, replace=True)
            agg = scores[:, idx].mean(axis=1)
            _accumulate_tie_aware_rank_mass(rank_counts, agg)

    rank_probs = rank_counts / n_bootstrap
    expected_ranks = (rank_probs * np.arange(1, n_templates + 1)).sum(axis=1)
    p_best = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_bootstrap,
    )


def _bootstrap_ranks_seeded(
    scores: np.ndarray,
    labels: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> RankDistribution:
    """Rank distribution via nested bootstrap for ``scores`` of shape ``(N, M, R)``."""
    N, _, _ = scores.shape
    rank_counts = np.zeros((N, N), dtype=float)

    for _ in range(n_bootstrap):
        boot_cell_means = nested_resample_cell_means_once(scores, rng)  # (N, M)
        if statistic == "median":
            agg = np.median(boot_cell_means, axis=1)                   # (N,)
        else:
            agg = boot_cell_means.mean(axis=1)                         # (N,)
        _accumulate_tie_aware_rank_mass(rank_counts, agg)

    rank_probs = rank_counts / n_bootstrap
    expected_ranks = (rank_probs * np.arange(1, N + 1)).sum(axis=1)
    p_best = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_bootstrap,
    )


def _bayes_bootstrap_ranks_seeded(
    scores: np.ndarray,
    labels: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> RankDistribution:
    """Bayesian bootstrap rank distribution via nested bootstrap for ``scores`` of shape ``(N, M, R)``.

    Inner level resamples R runs uniformly; outer level uses Dirichlet(1,...,1_M)
    weights instead of multinomial input resampling.
    """
    N, M, _ = scores.shape
    rank_counts = np.zeros((N, N), dtype=float)

    for _ in range(n_bootstrap):
        cell_means, w = bayes_bootstrap_resample_cell_means_once(scores, rng)  # (N, M), (M,)
        if statistic == "median":
            agg = np.array([_weighted_median(cell_means[t], w) for t in range(N)])
        else:
            agg = cell_means @ w                                        # (N,)
        _accumulate_tie_aware_rank_mass(rank_counts, agg)

    rank_probs = rank_counts / n_bootstrap
    expected_ranks = (rank_probs * np.arange(1, N + 1)).sum(axis=1)
    p_best = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_bootstrap,
    )


def _smooth_bootstrap_ranks_seeded(
    scores: np.ndarray,
    labels: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> RankDistribution:
    """Smoothed bootstrap rank distribution for ``scores`` of shape ``(N, M, R)``.

    Inner level resamples R runs uniformly; outer level resamples M inputs
    with replacement; Gaussian KDE noise is added to each resampled cell mean.
    """
    N, M, _ = scores.shape
    cell_means = scores.mean(axis=2)   # (N, M) — original cell means for bandwidth estimation

    from scipy.stats import gaussian_kde as _kde
    bws = np.zeros(N)
    for t in range(N):
        std_t = float(np.std(cell_means[t], ddof=1)) if M > 1 else 0.0
        if std_t > 0.0 and M >= 2:
            bws[t] = float(_kde(cell_means[t]).factor * std_t)

    rank_counts = np.zeros((N, N), dtype=float)
    for _ in range(n_bootstrap):
        boot_cell_means = smooth_bootstrap_resample_cell_means_once(scores, bws, rng)  # (N, M)
        if statistic == "median":
            agg = np.median(boot_cell_means, axis=1)
        else:
            agg = boot_cell_means.mean(axis=1)
        _accumulate_tie_aware_rank_mass(rank_counts, agg)

    rank_probs = rank_counts / n_bootstrap
    expected_ranks = (rank_probs * np.arange(1, N + 1)).sum(axis=1)
    p_best = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_bootstrap,
    )
