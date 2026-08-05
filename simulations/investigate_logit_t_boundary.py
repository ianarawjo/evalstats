"""One-off investigation (2026-08-04): does logit_t_ci_1d undercover on
boundary-hugging, right-skewed [0, 1] data, and can a cheap closed-form
refinement recover calibration without giving up its width advantage on
well-behaved data?

Motivated by a real finding on 2026-08-03: a `ci_paired --data-source
openeval --by-n-violin-plot` run showed logit_t badly undercovering (87%
coverage at n=10 vs. the 95% nominal target, and still only 94.8% at
n=100) on the Llama-2-13b-hf vs. gemma-1.1-7b-it truthfulqa/bleu_max
comparison -- a metric whose scores are heavily right-skewed with a mean
near 0.05-0.08. The synthetic scenario suite doesn't generate continuous
distributions this close to a boundary, so this failure mode wasn't
visible in the harness's usual calibration sweeps (see
simulations/harness/scenarios/synthetic.py's build_pair_sources).

This script isolates the mechanism with single-sample (not paired-diff)
Beta-distributed synthetic data at several skew/boundary-proximity levels
matching the real corpus, and tests two candidate refinements against the
shipped logit_t_ci_1d (order=1 and order=2), nig_ci_1d, a plain t-interval,
and a BCa bootstrap reference (BCa's acceleration term explicitly corrects
for skewness, so it's a good upper-bound sanity check on how much
correcting for skewness alone can buy here).

Candidates
----------
logit_t_cf_raw   : logit_t's existing delta-method interval, but replacing
                    the symmetric t-quantile with a first-order
                    Cornish-Fisher skewness correction, w(z) = z +
                    (gamma1/6)(z^2 - 1), using the RAW sample's skewness
                    gamma1 (this targets the actual source of the problem:
                    x-bar's own sampling distribution is skewed at small n
                    when the underlying data is skewed, which is a
                    different mechanism from the transform-curvature bias
                    order=2 already corrects for).
logit_t_cf_logit : same Cornish-Fisher correction, but using the skewness
                    of the logit-transformed sample instead of the raw
                    sample, to check which scale the correction should be
                    estimated on.

Not part of the harness / --official-tests: standalone Monte Carlo script.
Run directly:

    .venv/bin/python simulations/investigate_logit_t_boundary.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from evalstats.core.resampling import (
    bootstrap_ci_1d,
    logit_t_ci_1d,
    nig_ci_1d,
    t_interval_ci_1d,
)
from evalstats.core.stats_utils import interval_score

ALPHA = 0.05
N_VALUES = [10, 15, 20, 30, 50, 75, 100]
N_REPS = 2000
N_BOOTSTRAP = 1000
SEED = 20260804


# --------------------------------------------------------------------------
# Extreme distribution regimes -- calibrated to mimic the real truthfulqa/
# bleu_max corpus (mean ~= 0.08, right-skewed, mass concentrated near 0)
# --------------------------------------------------------------------------

DISTRIBUTIONS = {
    # (alpha, beta) for scipy.stats.beta; mean = a/(a+b)
    "beta_extreme_low": (0.2, 6.0),    # mean 0.032, most extreme near-0 case
    "beta_low": (0.3, 4.0),            # mean 0.070, matches real truthfulqa/bleu_max
    "beta_moderate_low": (0.5, 3.0),   # mean 0.143, milder skew
    "beta_extreme_high": (6.0, 0.2),   # mean 0.968, mirror case near upper boundary
    "beta_symmetric": (2.0, 2.0),      # mean 0.5, sanity check -- should be easy for everyone
}


def true_mean(a: float, b: float) -> float:
    return a / (a + b)


def true_skew(a: float, b: float) -> float:
    return float(stats.beta.stats(a, b, moments="s"))


# --------------------------------------------------------------------------
# Candidate refinements
# --------------------------------------------------------------------------

def _logit_t_core(vals: np.ndarray, alpha: float):
    """Shared setup replicating logit_t_ci_1d's internals (see
    evalstats/core/resampling.py:557), returning the pieces a quantile-level
    refinement needs instead of the finished interval."""
    n = len(vals)
    x_bar = float(np.mean(vals))
    se = float(np.std(vals, ddof=1)) / np.sqrt(n)
    logit_mean = float(np.log(x_bar / (1.0 - x_bar)))
    se_logit = se / (x_bar * (1.0 - x_bar))
    return x_bar, se, logit_mean, se_logit


def logit_t_cf_ci_1d(values: np.ndarray, alpha: float, skew_scale: str = "raw") -> tuple[float, float]:
    """logit_t_ci_1d with a first-order Cornish-Fisher skewness correction
    applied to the t-quantile instead of a plain symmetric t-interval.

    w(z) = z + (gamma1/6)(z^2 - 1), evaluated at z = -t_crit and z = +t_crit
    (the base t-distribution's own alpha/2 and 1-alpha/2 quantiles), where
    gamma1 is the sample skewness -- of the raw values if skew_scale='raw',
    or of the logit-transformed values if skew_scale='logit'. Falls back to
    plain logit_t_ci_1d (order=1) if x_bar/se are degenerate.
    """
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n <= 1:
        mean = float(np.mean(vals)) if n == 1 else 0.0
        return (mean, mean)
    x_bar, se, logit_mean, se_logit = _logit_t_core(vals, alpha)
    if se <= 0.0 or not np.isfinite(se) or x_bar <= 0.0 or x_bar >= 1.0:
        return (x_bar, x_bar)

    if skew_scale == "raw":
        gamma1 = float(stats.skew(vals, bias=False))
    elif skew_scale == "logit":
        logit_vals = np.log(vals / (1.0 - vals))
        gamma1 = float(stats.skew(logit_vals, bias=False))
    else:
        raise ValueError(f"unknown skew_scale {skew_scale!r}")
    if not np.isfinite(gamma1):
        gamma1 = 0.0

    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    z_lo, z_hi = -t_crit, t_crit
    w_lo = z_lo + (gamma1 / 6.0) * (z_lo**2 - 1.0)
    w_hi = z_hi + (gamma1 / 6.0) * (z_hi**2 - 1.0)

    lo = float(1.0 / (1.0 + np.exp(-(logit_mean + w_lo * se_logit))))
    hi = float(1.0 / (1.0 + np.exp(-(logit_mean + w_hi * se_logit))))
    return (max(0.0, min(lo, hi)), min(1.0, max(lo, hi)))


def logit_t_cf_raw_ci_1d(values, alpha):
    return logit_t_cf_ci_1d(values, alpha, skew_scale="raw")


def logit_t_cf_logit_ci_1d(values, alpha):
    return logit_t_cf_ci_1d(values, alpha, skew_scale="logit")


def logit_t_order3_ci_1d(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """logit_t_ci_1d(order=2) plus a further 3rd-order Taylor bias
    correction to the point estimate, using g'''(x) = 2/x^3 + 2/(1-x)^3 and
    the sample's raw 3rd central moment m3 = mean((x_i - x_bar)^3) as a
    plug-in estimate of kappa3(X_bar) = m3/n^2 (the 3rd cumulant of the
    sample mean, to leading order in n). Re-derivation of the "3rd-order
    term" the shipped logit_t_ci_1d's docstring says was tried and found not
    to help -- but that test was on data later diagnosed as NOT genuinely
    skewed (a data-hygiene artifact), so this re-tests it on the genuinely
    skewed regime here.
    """
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n <= 1:
        mean = float(np.mean(vals)) if n == 1 else 0.0
        return (mean, mean)
    x_bar, se, logit_mean, se_logit = _logit_t_core(vals, alpha)
    if se <= 0.0 or not np.isfinite(se) or x_bar <= 0.0 or x_bar >= 1.0:
        return (x_bar, x_bar)

    g2 = -1.0 / x_bar**2 + 1.0 / (1.0 - x_bar) ** 2
    logit_mean -= 0.5 * g2 * se**2

    g3 = 2.0 / x_bar**3 + 2.0 / (1.0 - x_bar) ** 3
    m3 = float(np.mean((vals - x_bar) ** 3))
    kappa3_xbar = m3 / n**2
    logit_mean -= (1.0 / 6.0) * g3 * kappa3_xbar

    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    lo = float(1.0 / (1.0 + np.exp(-(logit_mean - t_crit * se_logit))))
    hi = float(1.0 / (1.0 + np.exp(-(logit_mean + t_crit * se_logit))))
    return (max(0.0, lo), min(1.0, hi))


METHODS = {
    "t_interval": lambda vals, alpha, rng: t_interval_ci_1d(vals, alpha),
    "logit_t (order=1, current)": lambda vals, alpha, rng: logit_t_ci_1d(vals, alpha, order=1),
    "logit_t (order=2)": lambda vals, alpha, rng: logit_t_ci_1d(vals, alpha, order=2),
    "logit_t (order=3, candidate)": lambda vals, alpha, rng: logit_t_order3_ci_1d(vals, alpha),
    "logit_t_cf_raw (candidate)": lambda vals, alpha, rng: logit_t_cf_raw_ci_1d(vals, alpha),
    "logit_t_cf_logit (candidate)": lambda vals, alpha, rng: logit_t_cf_logit_ci_1d(vals, alpha),
    "nig": lambda vals, alpha, rng: nig_ci_1d(vals, alpha),
    "bca_bootstrap (reference)": lambda vals, alpha, rng: bootstrap_ci_1d(
        vals, float(np.mean(vals)), "bca", N_BOOTSTRAP, alpha, rng,
    ),
    "bootstrap_t (reference)": lambda vals, alpha, rng: bootstrap_ci_1d(
        vals, float(np.mean(vals)), "bootstrap_t", N_BOOTSTRAP, alpha, rng,
    ),
}


def run() -> pd.DataFrame:
    rows = []
    rng_master = np.random.default_rng(SEED)
    t0 = time.time()
    for dist_name, (a, b) in DISTRIBUTIONS.items():
        mu = true_mean(a, b)
        skew = true_skew(a, b)
        for n in N_VALUES:
            covered = {m: 0 for m in METHODS}
            score_sum = {m: 0.0 for m in METHODS}
            width_sum = {m: 0.0 for m in METHODS}
            for _rep in range(N_REPS):
                rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
                vals = rng.beta(a, b, size=n)
                for method_name, fn in METHODS.items():
                    try:
                        lo, hi = fn(vals, ALPHA, rng)
                    except Exception:
                        lo = hi = float(np.mean(vals))
                    if lo <= mu <= hi:
                        covered[method_name] += 1
                    score_sum[method_name] += interval_score(lo, hi, mu, ALPHA)
                    width_sum[method_name] += hi - lo
            for method_name in METHODS:
                rows.append({
                    "distribution": dist_name, "true_mean": mu, "true_skew": skew,
                    "n": n, "method": method_name,
                    "coverage": covered[method_name] / N_REPS,
                    "mean_score": score_sum[method_name] / N_REPS,
                    "mean_width": width_sum[method_name] / N_REPS,
                })
        print(f"  done: {dist_name} (mean={mu:.3f}, skew={skew:.2f})  [{time.time()-t0:.0f}s elapsed]")
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame) -> None:
    target = 1.0 - ALPHA
    for dist_name in DISTRIBUTIONS:
        sub = df[df["distribution"] == dist_name]
        mu = sub["true_mean"].iloc[0]
        skew = sub["true_skew"].iloc[0]
        print(f"\n=== {dist_name}  (true_mean={mu:.3f}, true_skew={skew:.2f}) ===")
        cov = sub.pivot(index="n", columns="method", values="coverage")
        score = sub.pivot(index="n", columns="method", values="mean_score")
        cov_str = cov.round(3).to_string()
        print("Coverage (target = 0.950):")
        print(cov_str)
        print("\nMean interval score (lower is better):")
        print(score.round(4).to_string())


if __name__ == "__main__":
    print(f"Running {len(DISTRIBUTIONS)} distributions x {len(N_VALUES)} sizes x "
          f"{N_REPS} reps x {len(METHODS)} methods...")
    results_df = run()
    out_path = "simulations/out/investigate_logit_t_boundary_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved raw results to {out_path}")
    print_report(results_df)
