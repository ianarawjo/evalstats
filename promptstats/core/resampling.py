"""Shared bootstrap and BCa resampling utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats


def _stat(values: np.ndarray, statistic: Literal["mean", "median"]) -> float:
    """Apply *statistic* to a 1-D array and return a Python float."""
    if statistic == "median":
        return float(np.median(values))
    return float(np.mean(values))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of *values* given *weights* (must sum to 1)."""
    sorted_idx = np.argsort(values)
    cumsum = np.cumsum(weights[sorted_idx])
    idx = int(np.searchsorted(cumsum, 0.5))
    return float(values[sorted_idx[min(idx, len(values) - 1)]])


def resolve_resampling_method(
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto"],
    sample_size: int,
    *,
    bca_min_n: int = 15,
    bca_max_n: int = 200,
) -> Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap"]:
    """Resolve ``method='auto'`` to a concrete bootstrap method.

    Uses BCa for moderate sample sizes where acceleration/bias correction is
    typically beneficial, and percentile bootstrap otherwise.
    ``'bayes_bootstrap'`` and ``'smooth_bootstrap'`` are passed through unchanged.
    """
    if method == "auto":
        return "bca" if bca_min_n <= sample_size <= bca_max_n else "bootstrap"
    return method  # type: ignore[return-value]


def bootstrap_means_1d(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Generate bootstrap replicates of the sample statistic for 1-D values.

    Parameters
    ----------
    statistic : str
        ``'mean'`` (default) or ``'median'``.
    """
    m = len(values)
    boot_stats = np.empty(n_bootstrap)
    if statistic == "median":
        for b in range(n_bootstrap):
            idx = rng.choice(m, size=m, replace=True)
            boot_stats[b] = np.median(values[idx])
    else:
        for b in range(n_bootstrap):
            idx = rng.choice(m, size=m, replace=True)
            boot_stats[b] = np.mean(values[idx])
    return boot_stats


def bayes_bootstrap_means_1d(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Bayesian bootstrap replicates for 1-D values.

    Implements the Bayesian bootstrap (Rubin 1981) as used by Banks (1988)
    "Histospline smoothing the Bayesian bootstrap."  Rather than drawing
    integer-valued multinomial counts (as in the standard bootstrap), each
    replicate draws continuous Dirichlet(1,...,1) weights via normalised
    Exp(1) variates.  This gives smoother coverage—especially at small
    sample sizes—because it explores the full simplex of weight assignments
    rather than just the lattice of integer multiples of 1/n.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed values.
    n_bootstrap : int
        Number of Bayesian bootstrap replicates to draw.
    rng : np.random.Generator
        Random number generator.
    statistic : str
        ``'mean'`` (default) or ``'median'``.  For ``'mean'``, replicates are
        Dirichlet-weighted means; for ``'median'``, weighted medians.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    n = len(values)
    # Draw (n_bootstrap, n) Exp(1) variates; normalise rows → Dirichlet(1,...,1).
    exp_mat = rng.exponential(1.0, size=(n_bootstrap, n))          # (B, n)
    weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)         # (B, n)

    if statistic == "mean":
        return weights @ values                                     # (B,)

    # Weighted median: sort once by value, then walk the cumulative-weight CDF.
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_w = weights[:, sorted_idx]                              # (B, n)
    cum_w = np.cumsum(sorted_w, axis=1)                            # (B, n)
    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = int(np.searchsorted(cum_w[b], 0.5))
        boot_stats[b] = sorted_vals[min(idx, n - 1)]
    return boot_stats


def bayes_bootstrap_diffs_nested(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Bayesian nested bootstrap replicates of paired cell-mean differences.

    Outer level: Dirichlet(1,...,1_M) weights over the M inputs.
    Inner level: standard uniform resample of R runs within each input.

    Using Dirichlet outer weights (rather than multinomial resampling) gives
    smoother bootstrap distributions for small M—the primary motivation for
    Bayesian bootstrap over the standard nested bootstrap.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Per-cell score arrays of shape ``(M, R)``.
    n_bootstrap : int
        Number of bootstrap replicates.
    rng : np.random.Generator
        Random number generator.
    statistic : str
        ``'mean'`` (default) or ``'median'``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    M, R = scores_a.shape

    # Inner resample: which R runs for each (bootstrap, input) pair.
    run_idx = rng.integers(0, R, size=(n_bootstrap, M, R))         # (B, M, R)
    m_range = np.arange(M)[np.newaxis, :, np.newaxis]              # (1, M, 1)

    # Gather inner-resampled runs from all M original inputs.
    resampled_a = scores_a[m_range, run_idx]                       # (B, M, R)
    resampled_b = scores_b[m_range, run_idx]                       # (B, M, R)

    # Cell means (always mean over R inner-resampled runs).
    diffs = resampled_a.mean(axis=2) - resampled_b.mean(axis=2)   # (B, M)

    # Outer Dirichlet weights for the M inputs.
    exp_mat = rng.exponential(1.0, size=(n_bootstrap, M))          # (B, M)
    outer_weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)   # (B, M)

    if statistic == "mean":
        return (outer_weights * diffs).sum(axis=1)                 # (B,)

    # Weighted median per bootstrap sample.
    sorted_idx = np.argsort(diffs, axis=1)                         # (B, M)
    sorted_diffs = np.take_along_axis(diffs, sorted_idx, axis=1)
    sorted_w = np.take_along_axis(outer_weights, sorted_idx, axis=1)
    cumsum_w = np.cumsum(sorted_w, axis=1)                         # (B, M)
    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = int(np.searchsorted(cumsum_w[b], 0.5))
        boot_stats[b] = sorted_diffs[b, min(idx, M - 1)]
    return boot_stats


def bayes_bootstrap_resample_cell_means_once(
    scores: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One Bayesian bootstrap nested resample of per-input cell means.

    Inner level resamples R runs uniformly; outer level returns Dirichlet
    weights for the M inputs (rather than resampling them with replacement).

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, M, R)``.
    rng : np.random.Generator

    Returns
    -------
    cell_means : np.ndarray
        Shape ``(N, M)`` — inner-resampled cell means for all M inputs.
    outer_weights : np.ndarray
        Shape ``(M,)`` — Dirichlet(1,...,1) weights summing to 1.
    """
    N, M, R = scores.shape
    run_idx = rng.integers(0, R, size=(M, R))                      # (M, R)
    m_range = np.arange(M)[:, np.newaxis]                          # (M, 1)
    resampled = scores[:, m_range, run_idx]                        # (N, M, R)
    cell_means = resampled.mean(axis=2)                            # (N, M)

    exp_samp = rng.exponential(1.0, size=M)
    outer_weights = exp_samp / exp_samp.sum()                      # (M,)
    return cell_means, outer_weights


def smooth_bootstrap_means_1d(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Smoothed bootstrap replicates for 1-D values using Gaussian KDE.

    Each replicate resamples n observations with replacement from *values*
    and adds i.i.d. Gaussian noise with standard deviation equal to the KDE
    bandwidth (Scott's rule via ``scipy.stats.gaussian_kde``).  This smooths
    the discrete empirical distribution, which can improve coverage for
    continuous data—especially at small sample sizes.

    Falls back to the plain percentile bootstrap if ``std(values) == 0``
    or ``n < 2`` (KDE is degenerate).

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed values.
    n_bootstrap : int
        Number of smoothed bootstrap replicates.
    rng : np.random.Generator
        Random number generator.
    statistic : str
        ``'mean'`` (default) or ``'median'``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    from scipy.stats import gaussian_kde

    n = len(values)
    std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if std_val == 0.0 or n < 2:
        return bootstrap_means_1d(values, n_bootstrap, rng, statistic=statistic)

    h = float(gaussian_kde(values).factor * std_val)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    noise = rng.normal(0.0, h, size=(n_bootstrap, n))
    samples = values[idx] + noise          # (B, n)
    if statistic == "median":
        return np.median(samples, axis=1)  # (B,)
    return samples.mean(axis=1)            # (B,)


def smooth_bootstrap_diffs_nested(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Smoothed nested bootstrap replicates of paired cell-mean differences.

    KDE bandwidth is estimated from the M per-input cell-mean differences.
    The outer level resamples M inputs with replacement; the inner level
    resamples R runs; Gaussian noise with std = KDE bandwidth is then added
    to each resampled cell-mean difference.

    Falls back to ``bootstrap_diffs_nested`` if ``std(cell_diffs) == 0``
    or ``M < 2``.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Per-cell score arrays of shape ``(M, R)``.
    n_bootstrap : int
        Number of bootstrap replicates.
    rng : np.random.Generator
        Random number generator.
    statistic : str
        ``'mean'`` (default) or ``'median'``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.
    """
    from scipy.stats import gaussian_kde

    M, R = scores_a.shape
    cell_diffs = scores_a.mean(axis=1) - scores_b.mean(axis=1)   # (M,)
    std_val = float(np.std(cell_diffs, ddof=1)) if M > 1 else 0.0
    if std_val == 0.0 or M < 2:
        return bootstrap_diffs_nested(scores_a, scores_b, n_bootstrap, rng, statistic=statistic)

    h = float(gaussian_kde(cell_diffs).factor * std_val)

    input_idx = rng.integers(0, M, size=(n_bootstrap, M))         # (B, M)
    run_idx = rng.integers(0, R, size=(n_bootstrap, M, R))        # (B, M, R)
    sel_a = scores_a[input_idx]                                    # (B, M, R)
    sel_b = scores_b[input_idx]                                    # (B, M, R)
    b_range = np.arange(n_bootstrap)[:, np.newaxis, np.newaxis]   # (B, 1, 1)
    m_range = np.arange(M)[np.newaxis, :, np.newaxis]             # (1, M, 1)
    resampled_a = sel_a[b_range, m_range, run_idx]                 # (B, M, R)
    resampled_b = sel_b[b_range, m_range, run_idx]                 # (B, M, R)

    diffs = resampled_a.mean(axis=2) - resampled_b.mean(axis=2)   # (B, M)
    diffs += rng.normal(0.0, h, size=(n_bootstrap, M))
    if statistic == "median":
        return np.median(diffs, axis=1)
    return diffs.mean(axis=1)


def smooth_bootstrap_resample_cell_means_once(
    scores: np.ndarray,
    bandwidths: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One smoothed nested resample of per-input cell means.

    Inner level resamples R runs uniformly; outer level resamples M inputs
    with replacement. Gaussian noise with std = ``bandwidths[i]`` is then
    added to each resampled cell mean for template *i*.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, M, R)``.
    bandwidths : np.ndarray
        Shape ``(N,)`` — per-template KDE bandwidths.  Zero entries skip
        smoothing for that template (degenerate case).
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Shape ``(N, M)`` — smoothed resampled cell means.
    """
    N, M, R = scores.shape
    input_idx = rng.integers(0, M, size=M)      # (M,)
    run_idx = rng.integers(0, R, size=(M, R))   # (M, R)

    sel = scores[:, input_idx, :]               # (N, M, R)
    m_range = np.arange(M)[:, np.newaxis]       # (M, 1)
    resampled = sel[:, m_range, run_idx]        # (N, M, R)
    cell_means = resampled.mean(axis=2)         # (N, M)

    for i in range(N):
        if bandwidths[i] > 0.0:
            cell_means[i] += rng.normal(0.0, bandwidths[i], size=M)
    return cell_means


def bootstrap_ci_1d(
    values: np.ndarray,
    observed_stat: float,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap"],
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> tuple[float, float]:
    """Bootstrap, BCa, Bayesian bootstrap, or smoothed bootstrap CI for a 1-D array.

    Parameters
    ----------
    statistic : str
        ``'mean'`` (default) or ``'median'``.
    """
    if method == "bayes_bootstrap":
        boot_stats = bayes_bootstrap_means_1d(values, n_bootstrap, rng, statistic=statistic)
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )
    if method == "smooth_bootstrap":
        boot_stats = smooth_bootstrap_means_1d(values, n_bootstrap, rng, statistic=statistic)
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )
    boot_stats = bootstrap_means_1d(values, n_bootstrap, rng, statistic=statistic)
    if method == "bca":
        return bca_interval_1d(values, observed_stat, boot_stats, alpha, statistic=statistic)
    return (
        float(np.percentile(boot_stats, 100 * alpha / 2)),
        float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
    )


def bca_interval_1d(
    values: np.ndarray,
    observed_stat: float,
    boot_stats: np.ndarray,
    alpha: float,
    statistic: Literal["mean", "median"] = "mean",
) -> tuple[float, float]:
    """Compute BCa confidence interval for a statistic of 1-D values.

    The jackknife acceleration estimate uses *statistic* for the
    leave-one-out estimates, matching the bootstrap statistic being corrected.

    Parameters
    ----------
    statistic : str
        ``'mean'`` (default) or ``'median'``.
    """
    b = len(boot_stats)
    less_count = np.sum(boot_stats < observed_stat)
    prop_less = (less_count + 0.5) / (b + 1)
    z0 = stats.norm.ppf(prop_less)

    m = len(values)
    jackknife_stats = np.empty(m)
    for i in range(m):
        jackknife_stats[i] = _stat(np.delete(values, i), statistic)
    # The acceleration uses the mean of jackknife estimates (standard BCa formula).
    jack_mean = np.mean(jackknife_stats)
    d = jack_mean - jackknife_stats
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    accel = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0

    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)

    def adjusted_prob(z_alpha: float) -> float:
        denom_term = 1 - accel * (z0 + z_alpha)
        if denom_term == 0:
            return 0.5
        z_adj = z0 + (z0 + z_alpha) / denom_term
        p = stats.norm.cdf(z_adj)
        return float(np.clip(p, 0.0, 1.0))

    p_low = adjusted_prob(z_alpha_low)
    p_high = adjusted_prob(z_alpha_high)

    ci_low = float(np.percentile(boot_stats, 100 * p_low))
    ci_high = float(np.percentile(boot_stats, 100 * p_high))
    return ci_low, ci_high


def bootstrap_diffs_nested(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Bootstrap replicates of ``statistic(cell_mean_a − cell_mean_b)`` via
    two-level (nested) resampling over inputs then runs.

    Both inputs must share the same shape ``(M, R)`` where M is the number
    of benchmark inputs and R is the number of repeated runs per input.

    On each bootstrap iteration the outer level resamples M inputs with
    replacement; the inner level independently resamples R runs for each
    selected input.  This propagates both input-sampling uncertainty and
    within-cell seed variance into the resulting distribution.

    The cell-level aggregation over R runs always uses the mean (collapsing
    repeated runs to a stable cell estimate).  The *statistic* parameter
    controls the across-inputs aggregation: ``'mean'`` or ``'median'``.

    The implementation is fully vectorised across bootstrap iterations.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Per-cell score arrays of shape ``(M, R)`` for the two templates.
    n_bootstrap : int
        Number of bootstrap replicates to generate.
    rng : np.random.Generator
        Random number generator.
    statistic : str
        Across-inputs aggregator: ``'mean'`` (default) or ``'median'``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap,)``.  Each entry is the statistic of paired
        cell-mean differences for one bootstrap resample.
    """
    M, R = scores_a.shape

    # Outer resample: which M inputs to use for each bootstrap iteration.
    # Shape (n_bootstrap, M).
    input_idx = rng.integers(0, M, size=(n_bootstrap, M))

    # Inner resample: which R runs to use for each (bootstrap, input) pair.
    # Shape (n_bootstrap, M, R).
    run_idx = rng.integers(0, R, size=(n_bootstrap, M, R))

    # Gather inputs: scores_a[input_idx] broadcasts over the (B, M) index
    # into axis 0 of scores_a (shape M, R), giving shape (B, M, R).
    sel_a = scores_a[input_idx]   # (B, M, R)
    sel_b = scores_b[input_idx]   # (B, M, R)

    # Gather runs: for each (b, k), pick the R run indices in run_idx[b, k].
    # sel_a[b, k, run_idx[b, k, r]] for all b, k, r.
    b_range = np.arange(n_bootstrap)[:, np.newaxis, np.newaxis]  # (B, 1, 1)
    m_range = np.arange(M)[np.newaxis, :, np.newaxis]            # (1, M, 1)
    resampled_a = sel_a[b_range, m_range, run_idx]               # (B, M, R)
    resampled_b = sel_b[b_range, m_range, run_idx]               # (B, M, R)

    # Cell means (always mean over R runs) and per-input paired differences.
    diffs = resampled_a.mean(axis=2) - resampled_b.mean(axis=2)  # (B, M)
    if statistic == "median":
        return np.median(diffs, axis=1)                          # (B,)
    return diffs.mean(axis=1)                                    # (B,)


def nested_resample_cell_means_once(
    scores: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One nested resample of per-input cell means for ``scores`` of shape ``(N, M, R)``.

    Outer level resamples inputs; inner level resamples runs within each
    selected input. Returns resampled cell means of shape ``(N, M)``.
    """
    N, M, R = scores.shape
    input_idx = rng.integers(0, M, size=M)      # (M,)
    run_idx = rng.integers(0, R, size=(M, R))   # (M, R)

    sel = scores[:, input_idx, :]               # (N, M, R)
    m_range = np.arange(M)[:, np.newaxis]       # (M, 1)
    resampled = sel[:, m_range, run_idx]        # (N, M, R)
    return resampled.mean(axis=2)               # (N, M)
