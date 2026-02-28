"""Shared bootstrap and BCa resampling utilities."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_means_1d(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate bootstrap replicates of the sample mean for 1D values."""
    m = len(values)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(m, size=m, replace=True)
        boot_means[b] = np.mean(values[idx])
    return boot_means


def bca_interval_1d(
    values: np.ndarray,
    observed_mean: float,
    boot_means: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Compute BCa confidence interval for the mean of 1D values."""
    b = len(boot_means)
    less_count = np.sum(boot_means < observed_mean)
    prop_less = (less_count + 0.5) / (b + 1)
    z0 = stats.norm.ppf(prop_less)

    m = len(values)
    jackknife_means = np.empty(m)
    for i in range(m):
        jackknife_means[i] = np.mean(np.delete(values, i))
    jack_mean = np.mean(jackknife_means)
    d = jack_mean - jackknife_means
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

    ci_low = float(np.percentile(boot_means, 100 * p_low))
    ci_high = float(np.percentile(boot_means, 100 * p_high))
    return ci_low, ci_high
