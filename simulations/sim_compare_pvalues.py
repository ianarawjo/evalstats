#!/usr/bin/env python3
"""
sim_compare_pvalues.py — P-value method comparison for LLM-eval style decisions.

This script benchmarks p-value procedures under synthetic paired-evaluation data,
with two complementary phases:

1) Pairwise method benchmark
   - Estimates Type-I error (null rejection rate) and power (alternative rejection)
     for pairwise A-vs-B comparisons across score types and sample sizes.
   - Tests two alternative-effect sizes per scenario (weak = 0.5×, full = 1.0×)
     to show a power curve rather than a single operating point.

2) Multi-arm multiplicity benchmark
   - Estimates family-wise false-positive rate under a global null
   - Estimates probability of correctly selecting a true best arm under an
     alternative, across p-value correction strategies (including Friedman+Nemenyi).
   - Bootstrap p-values are computed ONCE per simulation replicate; corrections
     are applied to the same raw p-values, so runtime scales with bootstrap draws
     rather than (draws × n_corrections).

The implementation intentionally reuses promptstats internals where possible so
results map directly to behavior in the library:
  - promptstats.core.paired.pairwise_differences
  - promptstats.core.paired.all_pairwise
  - promptstats.core.paired.friedman_nemenyi

Usage examples:
  python simulations/sim_compare_pvalues.py
  python simulations/sim_compare_pvalues.py --reps 1000 --bootstrap-n 2000
  python simulations/sim_compare_pvalues.py --sizes 10 20 50 100 --runs 3
  python simulations/sim_compare_pvalues.py --pairwise-only
  python simulations/sim_compare_pvalues.py --multiarm-only
  python simulations/sim_compare_pvalues.py --out-dir simulations/out --save-results save
"""

from __future__ import annotations

import argparse
import csv
import io
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

with np.errstate(all="ignore"):
    from promptstats.core.paired import (
        pairwise_differences,
        all_pairwise,
        friedman_nemenyi,
        _mcnemar_p,
    )
    from promptstats.core.stats_utils import correct_pvalues


PAIRWISE_METHODS = [
    "mcnemar",          # exact McNemar test (binary only)
    "fisher_exact",     # Fisher's exact test on paired 2x2 (binary only)
    "bootstrap",
    "bca",
    "bayes_bootstrap",
    "smooth_bootstrap",
    "permutation",
    "sign_test",        # exact paired sign-test p-value + bootstrap CI
    "newcombe",         # Newcombe CI + McNemar p-value (binary only)
    "bayes_binary",     # Bayesian binary model (binary only)
    "wilcoxon",
    "paired_t",
]

# "friedman_nemenyi" uses the Friedman omnibus + Nemenyi post-hoc instead of
# pairwise bootstrap; it is a separate test procedure rather than a p-value
# correction, but lives in the same comparison axis for the multi-arm benchmark.
MULTIARM_CORRECTIONS = ["none", "holm", "bonferroni", "fdr_bh", "friedman_nemenyi"]
EVAL_TYPES = ["binary", "continuous", "likert", "grades"]
PROGRESS_MODES = ["bar", "cell", "off"]
RESULTS_MODES = ["save", "off"]
PLOT_MODES = ["save", "off"]


@dataclass
class PairScenario:
    label: str
    eval_type: str
    generate_pair: Callable[[np.random.Generator, int, int, float], tuple[np.ndarray, np.ndarray]]
    # alt_conditions: list of (condition_label, delta) for the alternative hypothesis.
    # Include at least two deltas (weak and full) to show a power curve.
    alt_conditions: list[tuple[str, float]] = field(default_factory=list)
    # Empirical Cohen's |d| for each alt condition, computed once at build time via
    # _calibrate_cohens_d.  Parallel to alt_conditions (index i ↔ alt_conditions[i]).
    alt_cohens_d: list[float] = field(default_factory=list)


@dataclass
class PairwiseResult:
    eval_type: str
    scenario: str
    n: int
    runs: int
    method: str
    condition: str
    n_reps: int
    rejects: int
    p_sum: float
    cohens_d: float = 0.0  # 0.0 for null; empirical |d| for alt conditions


@dataclass
class MultiArmScenario:
    label: str
    eval_type: str
    generate_scores: Callable[[np.random.Generator, int, int, int, float], np.ndarray]
    alt_delta: float


@dataclass
class MultiArmResult:
    eval_type: str
    scenario: str
    n: int
    runs: int
    k: int
    correction: str
    condition: str
    n_reps: int
    any_reject: int
    best_selected: int


def _min_reps_for_mcse(target_mcse: float) -> int:
    """Worst-case reps needed for Bernoulli Monte Carlo MCSE target.

    Uses p(1-p) <= 0.25, so MCSE <= sqrt(0.25 / reps).
    """
    if target_mcse <= 0:
        return 0
    return int(np.ceil(0.25 / (target_mcse ** 2)))


class _ProgressReporter:
    def __init__(self, total: int, *, mode: str = "bar", label: str = "") -> None:
        self.total = max(int(total), 1)
        self.mode = mode
        self.label = label
        self.start = time.time()
        self.last_print = 0.0

    def update(self, step: int, detail: str = "") -> None:
        if self.mode == "off":
            return

        now = time.time()
        is_final = step >= self.total
        if not is_final and (now - self.last_print) < 0.2:
            return
        self.last_print = now

        if self.mode == "cell":
            pct = 100.0 * min(step, self.total) / self.total
            print(
                f"\r  [{step:>7d}/{self.total:<7d}] {pct:6.2f}%  {detail:<55s}",
                end="",
                flush=True,
            )
            if is_final:
                print()
            return

        frac = min(step, self.total) / self.total
        filled = int(28 * frac)
        bar = "█" * filled + "·" * (28 - filled)
        elapsed = max(now - self.start, 1e-9)
        rate = step / elapsed
        rem = max(self.total - step, 0)
        eta_sec = rem / max(rate, 1e-12)
        eta_m, eta_s = divmod(int(round(eta_sec)), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
        prefix = f"{self.label}: " if self.label else ""
        print(
            f"\r  {prefix}[{bar}] {100.0*frac:6.2f}%  {step:>7d}/{self.total:<7d}  ETA {eta_str}  {detail[:40]:<40s}",
            end="",
            flush=True,
        )
        if is_final:
            print()


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _gen_binary_pair(
    rng: np.random.Generator,
    n: int,
    runs: int,
    delta: float,
    base_p: float = 0.5,
    concentration: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = base_p * concentration
    beta = (1.0 - base_p) * concentration
    p_a = rng.beta(alpha, beta, size=(n, 1))
    p_b = np.clip(p_a + delta, 0.0, 1.0)
    a = rng.binomial(1, p_a, size=(n, runs)).astype(float)
    b = rng.binomial(1, p_b, size=(n, runs)).astype(float)
    return a, b


def _gen_continuous_pair(
    rng: np.random.Generator,
    n: int,
    runs: int,
    delta: float,
    a_shape: float = 2.0,
    b_shape: float = 5.0,
    sigma: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    base = rng.beta(a_shape, b_shape, size=(n, 1))
    shared = rng.normal(0.0, sigma, size=(n, runs))
    indiv_a = rng.normal(0.0, sigma * 0.5, size=(n, runs))
    indiv_b = rng.normal(0.0, sigma * 0.5, size=(n, runs))
    a_vals = np.clip(base + shared + indiv_a, 0.0, 1.0)
    b_vals = np.clip(base + delta + shared + indiv_b, 0.0, 1.0)
    return a_vals, b_vals


def _gen_continuous_pair_heavy_tail(
    rng: np.random.Generator,
    n: int,
    runs: int,
    delta: float,
    base_loc: float = 0.5,
    base_scale: float = 0.14,
    t_df: float = 2.5,
    tail_scale: float = 0.11,
    outlier_prob: float = 0.03,
    outlier_scale: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.clip(rng.normal(base_loc, base_scale, size=(n, 1)), 0.0, 1.0)
    shared = stats.t.rvs(df=t_df, size=(n, runs), random_state=rng) * tail_scale
    indiv_a = stats.t.rvs(df=t_df, size=(n, runs), random_state=rng) * (tail_scale * 0.7)
    indiv_b = stats.t.rvs(df=t_df, size=(n, runs), random_state=rng) * (tail_scale * 0.7)
    outlier_a = rng.normal(0.0, outlier_scale, size=(n, runs)) * (rng.random((n, runs)) < outlier_prob)
    outlier_b = rng.normal(0.0, outlier_scale, size=(n, runs)) * (rng.random((n, runs)) < outlier_prob)
    a_vals = np.clip(base + shared + indiv_a + outlier_a, 0.0, 1.0)
    b_vals = np.clip(base + delta + shared + indiv_b + outlier_b, 0.0, 1.0)
    return a_vals, b_vals


def _gen_likert_pair(
    rng: np.random.Generator,
    n: int,
    runs: int,
    delta: float,
    mu: float = 3.0,
    sigma: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    base = rng.normal(mu, sigma, size=(n, 1))
    shared = rng.normal(0.0, 0.35, size=(n, runs))
    indiv_a = rng.normal(0.0, 0.25, size=(n, runs))
    indiv_b = rng.normal(0.0, 0.25, size=(n, runs))
    a_vals = np.rint(np.clip(base + shared + indiv_a, 1.0, 5.0))
    b_vals = np.rint(np.clip(base + delta + shared + indiv_b, 1.0, 5.0))
    return a_vals, b_vals


def _gen_grades_pair(
    rng: np.random.Generator,
    n: int,
    runs: int,
    delta: float,
    mu: float = 58.0,
    sigma: float = 17.0,
) -> tuple[np.ndarray, np.ndarray]:
    base = rng.normal(mu, sigma, size=(n, 1))
    shared = rng.normal(0.0, sigma * 0.18, size=(n, runs))
    indiv_a = rng.normal(0.0, sigma * 0.12, size=(n, runs))
    indiv_b = rng.normal(0.0, sigma * 0.12, size=(n, runs))
    a_vals = np.clip(base + shared + indiv_a, 0.0, 100.0)
    b_vals = np.clip(base + delta + shared + indiv_b, 0.0, 100.0)
    return a_vals, b_vals


def _gen_grades_pair_heavy_tail(
    rng: np.random.Generator,
    n: int,
    runs: int,
    delta: float,
    mu: float = 55.0,
    sigma: float = 20.0,
    t_df: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.clip(mu + stats.t.rvs(df=t_df, size=(n, 1), random_state=rng) * sigma, 0.0, 100.0)
    shared = stats.t.rvs(df=t_df, size=(n, runs), random_state=rng) * (sigma * 0.22)
    indiv_a = stats.t.rvs(df=t_df, size=(n, runs), random_state=rng) * (sigma * 0.16)
    indiv_b = stats.t.rvs(df=t_df, size=(n, runs), random_state=rng) * (sigma * 0.16)
    a_vals = np.clip(base + shared + indiv_a, 0.0, 100.0)
    b_vals = np.clip(base + delta + shared + indiv_b, 0.0, 100.0)
    return a_vals, b_vals


def _calibrate_cohens_d(
    generate_pair: Callable,
    delta: float,
    n_ref: int = 2000,
) -> float:
    """Estimate empirical Cohen's |d| for a (generate_pair, delta) pair.

    Uses a large reference sample so the estimate is stable.  The signed d is
    mean(b - a) / std(b - a), which is positive when delta > 0 (B is better).
    Returns |d| so the value is always non-negative regardless of convention.
    """
    rng = np.random.default_rng(0)
    a, b = generate_pair(rng, n_ref, 1, delta)
    diffs = b[:, 0] - a[:, 0]
    sd = float(np.std(diffs, ddof=1))
    return abs(float(np.mean(diffs)) / sd) if sd > 1e-12 else 0.0


def build_pair_scenarios(*, include_extreme: bool = False) -> list[PairScenario]:
    scenarios: list[PairScenario] = []

    # Binary: balanced (p≈0.50)
    d = 0.06
    scenarios.append(PairScenario(
        label="binary-balanced",
        eval_type="binary",
        generate_pair=partial(_gen_binary_pair, base_p=0.50, concentration=12.0),
        alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
    ))

    # Binary: low success rate (p≈0.20)
    d = 0.05
    scenarios.append(PairScenario(
        label="binary-low",
        eval_type="binary",
        generate_pair=partial(_gen_binary_pair, base_p=0.20, concentration=10.0),
        alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
    ))

    # Binary: sparse (p≈0.05) — tests behavior when most examples fail completely
    d = 0.04
    scenarios.append(PairScenario(
        label="binary-sparse",
        eval_type="binary",
        generate_pair=partial(_gen_binary_pair, base_p=0.05, concentration=8.0),
        alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
    ))

    # Continuous: right-skewed
    d = 0.04
    scenarios.append(PairScenario(
        label="continuous-right-skew",
        eval_type="continuous",
        generate_pair=partial(_gen_continuous_pair, a_shape=2.0, b_shape=8.0, sigma=0.09),
        alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
    ))

    # Continuous: near centre
    d = 0.03
    scenarios.append(PairScenario(
        label="continuous-near-center",
        eval_type="continuous",
        generate_pair=partial(_gen_continuous_pair, a_shape=5.0, b_shape=5.0, sigma=0.10),
        alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
    ))

    # Likert: mid-range
    d = 0.28
    scenarios.append(PairScenario(
        label="likert-mid",
        eval_type="likert",
        generate_pair=partial(_gen_likert_pair, mu=3.0, sigma=0.90),
        alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
    ))

    # Likert: near floor
    d = 0.22
    scenarios.append(PairScenario(
        label="likert-near-floor",
        eval_type="likert",
        generate_pair=partial(_gen_likert_pair, mu=2.0, sigma=0.80),
        alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
    ))

    # Grades: mid-range
    d = 3.5
    scenarios.append(PairScenario(
        label="grades-mid",
        eval_type="grades",
        generate_pair=partial(_gen_grades_pair, mu=55.0, sigma=16.0),
        alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
    ))

    # Grades: high variance
    d = 4.0
    scenarios.append(PairScenario(
        label="grades-high-variance",
        eval_type="grades",
        generate_pair=partial(_gen_grades_pair, mu=52.0, sigma=24.0),
        alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
    ))

    if include_extreme:
        # Binary: very sparse event regime (near-zero success probability)
        d = 0.02
        scenarios.append(PairScenario(
            label="binary-ultra-sparse",
            eval_type="binary",
            generate_pair=partial(_gen_binary_pair, base_p=0.01, concentration=6.0),
            alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
        ))

        # Binary: near-ceiling regime (limited headroom)
        d = 0.018
        scenarios.append(PairScenario(
            label="binary-near-ceiling",
            eval_type="binary",
            generate_pair=partial(_gen_binary_pair, base_p=0.95, concentration=10.0),
            alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
        ))

        # Continuous: heavy-tailed + outlier contamination
        d = 0.025
        scenarios.append(PairScenario(
            label="continuous-heavy-tail",
            eval_type="continuous",
            generate_pair=partial(_gen_continuous_pair_heavy_tail, base_loc=0.50, base_scale=0.14, t_df=2.5),
            alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
        ))

        # Likert: near-ceiling compressed regime
        d = 0.16
        scenarios.append(PairScenario(
            label="likert-near-ceiling",
            eval_type="likert",
            generate_pair=partial(_gen_likert_pair, mu=4.45, sigma=0.55),
            alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
        ))

        # Grades: heavy-tailed broad-score regime
        d = 3.0
        scenarios.append(PairScenario(
            label="grades-heavy-tail",
            eval_type="grades",
            generate_pair=partial(_gen_grades_pair_heavy_tail, mu=56.0, sigma=22.0, t_df=3.0),
            alt_conditions=[("alt_half", d * 0.5), ("alt", d)],
        ))

    # Compute empirical Cohen's |d| for each scenario × alt condition once at
    # build time (n_ref=2000) so the metric is available in reports and CSV.
    for sc in scenarios:
        sc.alt_cohens_d = [
            _calibrate_cohens_d(sc.generate_pair, delta)
            for _, delta in sc.alt_conditions
        ]

    return scenarios


def _gen_multiarm_binary(
    rng: np.random.Generator,
    n: int,
    runs: int,
    k: int,
    delta: float,
    base_p: float = 0.5,
) -> np.ndarray:
    concentration = 10.0
    alpha = base_p * concentration
    beta = (1.0 - base_p) * concentration
    base = rng.beta(alpha, beta, size=(n, 1))
    effects = np.zeros(k)
    effects[0] = delta
    out = np.empty((k, n, runs), dtype=float)
    for j in range(k):
        p = np.clip(base + effects[j], 0.0, 1.0)
        out[j] = rng.binomial(1, p, size=(n, runs)).astype(float)
    return out


def _gen_multiarm_continuous(
    rng: np.random.Generator,
    n: int,
    runs: int,
    k: int,
    delta: float,
) -> np.ndarray:
    base = rng.beta(2.0, 5.0, size=(n, 1))
    shared = rng.normal(0.0, 0.10, size=(n, runs))
    out = np.empty((k, n, runs), dtype=float)
    effects = np.zeros(k)
    effects[0] = delta
    for j in range(k):
        indiv = rng.normal(0.0, 0.06, size=(n, runs))
        out[j] = np.clip(base + effects[j] + shared + indiv, 0.0, 1.0)
    return out


def _gen_multiarm_continuous_heavy_tail(
    rng: np.random.Generator,
    n: int,
    runs: int,
    k: int,
    delta: float,
) -> np.ndarray:
    base = np.clip(rng.normal(0.5, 0.15, size=(n, 1)), 0.0, 1.0)
    shared = stats.t.rvs(df=2.5, size=(n, runs), random_state=rng) * 0.11
    out = np.empty((k, n, runs), dtype=float)
    effects = np.zeros(k)
    effects[0] = delta
    for j in range(k):
        indiv = stats.t.rvs(df=2.5, size=(n, runs), random_state=rng) * 0.08
        spikes = rng.normal(0.0, 0.30, size=(n, runs)) * (rng.random((n, runs)) < 0.03)
        out[j] = np.clip(base + effects[j] + shared + indiv + spikes, 0.0, 1.0)
    return out


def _gen_multiarm_likert(
    rng: np.random.Generator,
    n: int,
    runs: int,
    k: int,
    delta: float,
) -> np.ndarray:
    base = rng.normal(3.0, 0.95, size=(n, 1))
    shared = rng.normal(0.0, 0.35, size=(n, runs))
    out = np.empty((k, n, runs), dtype=float)
    effects = np.zeros(k)
    effects[0] = delta
    for j in range(k):
        indiv = rng.normal(0.0, 0.25, size=(n, runs))
        out[j] = np.rint(np.clip(base + effects[j] + shared + indiv, 1.0, 5.0))
    return out


def _gen_multiarm_likert_near_ceiling(
    rng: np.random.Generator,
    n: int,
    runs: int,
    k: int,
    delta: float,
) -> np.ndarray:
    base = rng.normal(4.45, 0.55, size=(n, 1))
    shared = rng.normal(0.0, 0.28, size=(n, runs))
    out = np.empty((k, n, runs), dtype=float)
    effects = np.zeros(k)
    effects[0] = delta
    for j in range(k):
        indiv = rng.normal(0.0, 0.20, size=(n, runs))
        out[j] = np.rint(np.clip(base + effects[j] + shared + indiv, 1.0, 5.0))
    return out


def _gen_multiarm_grades(
    rng: np.random.Generator,
    n: int,
    runs: int,
    k: int,
    delta: float,
) -> np.ndarray:
    base = rng.normal(58.0, 18.0, size=(n, 1))
    shared = rng.normal(0.0, 3.5, size=(n, runs))
    out = np.empty((k, n, runs), dtype=float)
    effects = np.zeros(k)
    effects[0] = delta
    for j in range(k):
        indiv = rng.normal(0.0, 2.6, size=(n, runs))
        out[j] = np.clip(base + effects[j] + shared + indiv, 0.0, 100.0)
    return out


def _gen_multiarm_grades_heavy_tail(
    rng: np.random.Generator,
    n: int,
    runs: int,
    k: int,
    delta: float,
) -> np.ndarray:
    base = np.clip(56.0 + stats.t.rvs(df=3.0, size=(n, 1), random_state=rng) * 20.0, 0.0, 100.0)
    shared = stats.t.rvs(df=3.0, size=(n, runs), random_state=rng) * 4.2
    out = np.empty((k, n, runs), dtype=float)
    effects = np.zeros(k)
    effects[0] = delta
    for j in range(k):
        indiv = stats.t.rvs(df=3.0, size=(n, runs), random_state=rng) * 3.0
        out[j] = np.clip(base + effects[j] + shared + indiv, 0.0, 100.0)
    return out


def build_multiarm_scenarios(*, include_extreme: bool = False) -> list[MultiArmScenario]:
    scenarios = [
        MultiArmScenario(
            label="binary",
            eval_type="binary",
            generate_scores=partial(_gen_multiarm_binary, base_p=0.5),
            alt_delta=0.05,
        ),
        MultiArmScenario(
            label="continuous",
            eval_type="continuous",
            generate_scores=_gen_multiarm_continuous,
            alt_delta=0.03,
        ),
        MultiArmScenario(
            label="likert",
            eval_type="likert",
            generate_scores=_gen_multiarm_likert,
            alt_delta=0.24,
        ),
        MultiArmScenario(
            label="grades",
            eval_type="grades",
            generate_scores=_gen_multiarm_grades,
            alt_delta=3.0,
        ),
    ]

    if include_extreme:
        scenarios.extend([
            MultiArmScenario(
                label="binary-ultra-sparse",
                eval_type="binary",
                generate_scores=partial(_gen_multiarm_binary, base_p=0.01),
                alt_delta=0.02,
            ),
            MultiArmScenario(
                label="continuous-heavy-tail",
                eval_type="continuous",
                generate_scores=_gen_multiarm_continuous_heavy_tail,
                alt_delta=0.025,
            ),
            MultiArmScenario(
                label="likert-near-ceiling",
                eval_type="likert",
                generate_scores=_gen_multiarm_likert_near_ceiling,
                alt_delta=0.14,
            ),
            MultiArmScenario(
                label="grades-heavy-tail",
                eval_type="grades",
                generate_scores=_gen_multiarm_grades_heavy_tail,
                alt_delta=2.8,
            ),
        ])

    return scenarios


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _safe_wilcoxon_p(diffs: np.ndarray) -> float:
    if int(np.sum(diffs != 0)) < 1:
        return 1.0
    try:
        with np.errstate(all="ignore"):
            w = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        p = float(w.pvalue)
        if not np.isfinite(p):
            return 1.0
        return min(max(p, 0.0), 1.0)
    except Exception:
        return 1.0


def _safe_paired_t_p(diffs: np.ndarray) -> float:
    if len(diffs) <= 1:
        return 1.0
    try:
        with np.errstate(all="ignore"):
            t = stats.ttest_1samp(diffs, popmean=0.0, nan_policy="omit")
        p = float(t.pvalue)
        if not np.isfinite(p):
            return 1.0
        return min(max(p, 0.0), 1.0)
    except Exception:
        return 1.0


def _pairwise_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    method: str,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: str,
) -> float:
    # NOTE on runs>1: wilcoxon and paired_t operate on per-input cell-mean
    # differences (shape (n,) after collapsing the run axis). This is
    # intentional: these are classical paired tests that treat each input as
    # one observation. Bootstrap-based methods receive the full (n, runs) arrays
    # and use a nested bootstrap to propagate within-input run variance.
    diffs = a.mean(axis=1) - b.mean(axis=1)

    if method == "wilcoxon":
        return _safe_wilcoxon_p(diffs)
    if method == "paired_t":
        return _safe_paired_t_p(diffs)

    if method in {"newcombe", "bayes_binary", "mcnemar", "fisher_exact"}:
        # These methods require binary (0/1) input. When runs > 1, collapse
        # multiple runs to a single binary value via majority vote (mean >= 0.5).
        aa = (a.mean(axis=1) >= 0.5).astype(float)
        bb = (b.mean(axis=1) >= 0.5).astype(float)
        if method == "mcnemar":
            # Standalone exact McNemar test (no CI computation overhead).
            return _mcnemar_p(aa, bb)
        scores = np.stack([aa, bb], axis=0)
    else:
        if a.shape[1] == 1:
            scores = np.stack([a[:, 0], b[:, 0]], axis=0)
        else:
            scores = np.stack([a, b], axis=0)

    result = pairwise_differences(
        scores=scores,
        idx_a=0,
        idx_b=1,
        label_a="A",
        label_b="B",
        method=method,
        ci=0.95,
        n_bootstrap=n_bootstrap,
        rng=rng,
        statistic=statistic,
    )
    p = float(result.p_value)
    if not np.isfinite(p):
        return 1.0
    return min(max(p, 0.0), 1.0)


def _method_allowed(eval_type: str, method: str, runs: int = 1) -> bool:
    binary_only = {"newcombe", "bayes_binary", "mcnemar", "fisher_exact"}
    # These methods require binary inputs; they are excluded for non-binary eval types.
    # When runs > 1, _pairwise_pvalue collapses runs via majority vote before calling them.
    if method in binary_only and eval_type != "binary":
        return False
    return True


# ---------------------------------------------------------------------------
# Multi-arm helpers
# ---------------------------------------------------------------------------


def _compute_multiarm_metrics(
    *,
    scores: np.ndarray,
    labels: list[str],
    method: str,
    corrections: list[str],
    n_bootstrap: int,
    alpha: float,
    statistic: str,
    rng: np.random.Generator,
) -> dict[str, tuple[bool, bool]]:
    """Compute (any_reject, best_selected) for every correction strategy.

    Bootstrap/permutation p-values are computed ONCE with correction='none',
    then each non-Friedman correction is applied to the same raw p-values as
    a cheap array operation.  This avoids repeating expensive bootstrap
    computations for every correction variant.

    The 'friedman_nemenyi' strategy uses a separate rank-based code path
    that does not require bootstrap sampling.
    """
    results: dict[str, tuple[bool, bool]] = {}
    k = len(labels)
    # Upper-triangle pair list in the same order all_pairwise stores them.
    pairs = [(labels[i], labels[j]) for i in range(k) for j in range(i + 1, k)]
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    # --- Bootstrap / permutation path (compute once for all non-Friedman strategies) ---
    non_friedman = [c for c in corrections if c != "friedman_nemenyi"]
    if non_friedman:
        matrix_none = all_pairwise(
            scores=scores,
            labels=labels,
            method=method,
            ci=1.0 - alpha,
            n_bootstrap=n_bootstrap,
            correction="none",
            rng=rng,
            statistic=statistic,
        )
        raw_p = np.array([matrix_none.get(a, b).p_value for (a, b) in pairs])
        # Precompute O(1) pair lookup to avoid repeated O(k²) list scans below.
        pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}

        for correction in non_friedman:
            if correction == "none":
                adj_p = raw_p
            else:
                adj_p = correct_pvalues(raw_p, correction)

            has_any = bool(np.any(adj_p < alpha))

            # best_selected: arm_0 is the true best under the alternative.
            # It is "selected" if it beats every other arm at significance alpha.
            best = labels[0]
            best_selected = True
            for other in labels[1:]:
                pair_key = (best, other)
                pair_idx = pair_to_idx.get(pair_key)
                if pair_idx is None:
                    best_selected = False
                    break
                if not (adj_p[pair_idx] < alpha and matrix_none.get(best, other).point_diff > 0.0):
                    best_selected = False
                    break

            results[correction] = (has_any, best_selected)

    # --- Friedman + Nemenyi path (rank-based, no bootstrap) ---
    if "friedman_nemenyi" in corrections:
        try:
            fr = friedman_nemenyi(scores, labels)
            has_any = any(
                (fr.get_nemenyi_p(a, b) or 1.0) < alpha for (a, b) in pairs
            )

            best = labels[0]
            best_selected = True
            for other in labels[1:]:
                nem_p = fr.get_nemenyi_p(best, other)
                if nem_p is None:
                    best_selected = False
                    break
                # Direction check: arm_0 should have strictly lower avg rank
                # (lower rank = better under the descending-rank convention used
                # by friedman_nemenyi).
                rank_best = fr.avg_ranks[best]
                rank_other = fr.avg_ranks[other]
                if not (nem_p < alpha and rank_best < rank_other):
                    best_selected = False
                    break

            results["friedman_nemenyi"] = (has_any, best_selected)
        except Exception:
            results["friedman_nemenyi"] = (False, False)

    return results


# ---------------------------------------------------------------------------
# Simulation engines
# ---------------------------------------------------------------------------


def _pairwise_cell_worker(args: tuple) -> list[PairwiseResult]:
    """Process one (scenario, n) cell. Top-level so it is picklable for ProcessPoolExecutor."""
    scenario, n, runs, all_conditions, methods, n_reps, n_bootstrap, alpha, statistic, cell_seed = args

    # Split the cell seed into independent streams via SeedSequence:
    #   - data_rng: shared across all methods per rep (Common Random Numbers — each
    #     method sees the same generated data, reducing method-comparison variance).
    #   - per-method bootstrap rngs: independent of each other and of method order
    #     in PAIRWISE_METHODS, so adding/reordering methods doesn't shift results.
    ss = np.random.SeedSequence(cell_seed)
    data_rng = np.random.default_rng(ss.spawn(1)[0])
    method_rngs = {m: np.random.default_rng(s) for m, s in zip(methods, ss.spawn(len(methods)))}

    # Build a lookup: condition label → Cohen's |d| (0.0 for null).
    cohens_d_map: dict[str, float] = {"null": 0.0}
    for (cond, _), cd in zip(scenario.alt_conditions, scenario.alt_cohens_d):
        cohens_d_map[cond] = cd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reject_counts: dict[tuple[str, str], int] = {
            (m, c): 0 for m in methods for (c, _) in all_conditions
        }
        p_sums: dict[tuple[str, str], float] = {
            (m, c): 0.0 for m in methods for (c, _) in all_conditions
        }
        for _ in range(n_reps):
            for condition, delta in all_conditions:
                a, b = scenario.generate_pair(data_rng, n, runs, delta)
                for method in methods:
                    p = _pairwise_pvalue(
                        a, b, method=method, n_bootstrap=n_bootstrap,
                        rng=method_rngs[method], statistic=statistic,
                    )
                    p_sums[(method, condition)] += p
                    if p <= alpha:
                        reject_counts[(method, condition)] += 1

    results: list[PairwiseResult] = []
    for method in methods:
        for condition, _ in all_conditions:
            results.append(PairwiseResult(
                eval_type=scenario.eval_type,
                scenario=scenario.label,
                n=n,
                runs=runs,
                method=method,
                condition=condition,
                n_reps=n_reps,
                rejects=reject_counts[(method, condition)],
                p_sum=p_sums[(method, condition)],
                cohens_d=cohens_d_map.get(condition, 0.0),
            ))
    return results


def _multiarm_cell_worker(args: tuple) -> list[MultiArmResult]:
    """Process one (scenario, n) cell for the multi-arm phase."""
    (scenario, n, runs, k_arms, labels, corrections,
     n_reps, n_bootstrap, alpha, multiarm_method, statistic, cell_seed) = args
    rng = np.random.default_rng(cell_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        agg_any: dict[tuple[str, str], int] = {
            (c, cond): 0 for c in corrections for cond in ("null", "alt")
        }
        agg_best: dict[tuple[str, str], int] = {
            (c, cond): 0 for c in corrections for cond in ("null", "alt")
        }
        for _ in range(n_reps):
            for condition, delta in (("null", 0.0), ("alt", scenario.alt_delta)):
                scores = scenario.generate_scores(rng, n, runs, k_arms, delta)
                # Compute bootstrap p-values ONCE; apply all corrections cheaply.
                metrics = _compute_multiarm_metrics(
                    scores=scores,
                    labels=labels,
                    method=multiarm_method,
                    corrections=corrections,
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                    statistic=statistic,
                    rng=rng,
                )
                for correction in corrections:
                    any_reject, best_selected = metrics.get(correction, (False, False))
                    if any_reject:
                        agg_any[(correction, condition)] += 1
                    if best_selected:
                        agg_best[(correction, condition)] += 1

    results: list[MultiArmResult] = []
    for correction in corrections:
        for condition in ("null", "alt"):
            results.append(MultiArmResult(
                eval_type=scenario.eval_type,
                scenario=scenario.label,
                n=n,
                runs=runs,
                k=k_arms,
                correction=correction,
                condition=condition,
                n_reps=n_reps,
                any_reject=agg_any[(correction, condition)],
                best_selected=agg_best[(correction, condition)],
            ))
    return results


def run_pairwise_simulation(
    *,
    scenarios: list[PairScenario],
    sample_sizes: list[int],
    runs: int,
    n_reps: int,
    n_bootstrap: int,
    alpha: float,
    statistic: str,
    progress_mode: str,
    seed: int,
    n_jobs: int = 1,
) -> list[PairwiseResult]:
    # Build one arg-tuple per (scenario, n) cell; each cell gets a deterministic seed
    # derived from the base seed so results are reproducible regardless of job count.
    cells: list[tuple] = []
    for s_idx, scenario in enumerate(scenarios):
        methods = [m for m in PAIRWISE_METHODS if _method_allowed(scenario.eval_type, m, runs)]
        all_conditions = [("null", 0.0)] + scenario.alt_conditions
        for n_idx, n in enumerate(sample_sizes):
            cell_seed = seed + s_idx * 10_000 + n_idx * 1_000
            cells.append((scenario, n, runs, all_conditions, methods,
                          n_reps, n_bootstrap, alpha, statistic, cell_seed))

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pairwise")
    results: list[PairwiseResult] = []

    if n_jobs == 1:
        with warnings.catch_warnings():
            if progress_mode != "off":
                warnings.simplefilter("ignore", UserWarning)
            for step, cell in enumerate(cells, 1):
                sc, n = cell[0], cell[1]
                reporter.update(step, detail=f"{sc.eval_type} {sc.label} n={n}")
                results.extend(_pairwise_cell_worker(cell))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_cell = {executor.submit(_pairwise_cell_worker, cell): cell for cell in cells}
            completed = 0
            for future in as_completed(future_to_cell):
                completed += 1
                sc, n = future_to_cell[future][0], future_to_cell[future][1]
                reporter.update(completed, detail=f"{sc.eval_type} {sc.label} n={n}")
                results.extend(future.result())

    reporter.update(len(cells), detail="done")
    return results


def run_multiarm_simulation(
    *,
    scenarios: list[MultiArmScenario],
    sample_sizes: list[int],
    runs: int,
    k_arms: int,
    n_reps: int,
    n_bootstrap: int,
    alpha: float,
    multiarm_method: str,
    statistic: str,
    progress_mode: str,
    seed: int,
    n_jobs: int = 1,
) -> list[MultiArmResult]:
    labels = [f"arm_{i}" for i in range(k_arms)]
    cells: list[tuple] = []
    for s_idx, scenario in enumerate(scenarios):
        for n_idx, n in enumerate(sample_sizes):
            cell_seed = seed + s_idx * 10_000 + n_idx * 1_000
            cells.append((scenario, n, runs, k_arms, labels, MULTIARM_CORRECTIONS,
                          n_reps, n_bootstrap, alpha, multiarm_method, statistic, cell_seed))

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="multiarm")
    results: list[MultiArmResult] = []

    if n_jobs == 1:
        with warnings.catch_warnings():
            if progress_mode != "off":
                warnings.simplefilter("ignore", UserWarning)
            for step, cell in enumerate(cells, 1):
                sc, n = cell[0], cell[1]
                reporter.update(step, detail=f"{sc.eval_type} n={n}")
                results.extend(_multiarm_cell_worker(cell))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_cell = {executor.submit(_multiarm_cell_worker, cell): cell for cell in cells}
            completed = 0
            for future in as_completed(future_to_cell):
                completed += 1
                sc, n = future_to_cell[future][0], future_to_cell[future][1]
                reporter.update(completed, detail=f"{sc.eval_type} n={n}")
                results.extend(future.result())

    reporter.update(len(cells), detail="done")
    return results


# ---------------------------------------------------------------------------
# Reporting and persistence
# ---------------------------------------------------------------------------


def _mc_stats(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    if total <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    phat = successes / total
    mcse = float(np.sqrt(max(phat * (1.0 - phat), 0.0) / total))
    lo = max(0.0, phat - z * mcse)
    hi = min(1.0, phat + z * mcse)
    return float(phat), mcse, float(lo), float(hi)


def print_pairwise_report(results: list[PairwiseResult], alpha: float) -> None:
    print("\n" + "=" * 90)
    print("  P-VALUE METHOD COMPARISON — PAIRWISE A VS B")
    print(f"  Type-I target: {alpha:.3f}   |   Power target: as high as possible")
    print("=" * 90)

    # Discover the ordered set of alternative conditions from the data.
    # Sort so weaker effects (alt_half) precede stronger (alt), matching the
    # scenario definition order rather than pure alphabetical order.
    all_conds = sorted(
        {r.condition for r in results if r.condition != "null"},
        key=lambda c: (c.count("_"), c),
    )

    # Print effect-size calibration table so power figures can be interpreted
    # across eval types.  Cohen's |d| is computed at n=2000 per scenario.
    calib: dict[tuple[str, str], float] = {}
    for r in results:
        if r.condition != "null" and (r.scenario, r.condition) not in calib:
            calib[(r.scenario, r.condition)] = r.cohens_d
    if calib:
        print("\n  Effect-size calibration (empirical Cohen's |d|, n_ref=2000):")
        scenarios_seen = sorted({r.scenario for r in results})
        header = f"  {'scenario':<26}" + "".join(f" {c:>10}" for c in all_conds)
        print(header)
        print("  " + "-" * (26 + 11 * len(all_conds)))
        for sc in scenarios_seen:
            row = f"  {sc:<26}" + "".join(
                f" {calib.get((sc, c), float('nan')):>10.3f}" for c in all_conds
            )
            print(row)

    key = lambda r: (r.eval_type, r.method, r.n, r.condition)
    grouped: dict[tuple[str, str, int, str], list[PairwiseResult]] = defaultdict(list)
    for r in results:
        grouped[key(r)].append(r)

    methods = sorted({r.method for r in results})
    sizes = sorted({r.n for r in results})

    # Build header dynamically based on available alternative conditions.
    print()
    alt_headers = "".join(f" {'pwr_' + c:>8} {'band95':>15}" for c in all_conds)
    print(
        f"  {'method':<18} {'n':>5} {'typeI':>8} {'mcse':>8} {'band95':>15}"
        + alt_headers
        + f" {'mean_p0':>9}"
    )
    sep = f"  {'-'*18} {'-'*5} {'-'*8} {'-'*8} {'-'*15}" + "".join(f" {'-'*8} {'-'*15}" for _ in all_conds) + f" {'-'*9}"
    print(sep)

    for et in EVAL_TYPES:
        if not any(r.eval_type == et for r in results):
            continue
        print(f"\n  {'—'*6} {et.upper()} {'—'*6}")

        for method in methods:
            for n in sizes:
                null_rows = grouped.get((et, method, n, "null"), [])
                if not null_rows:
                    continue

                null_rej = int(sum(r.rejects for r in null_rows))
                null_tot = int(sum(r.n_reps for r in null_rows))
                type1, n_mcse, n_lo, n_hi = _mc_stats(null_rej, null_tot)
                mean_p0 = float(sum(r.p_sum for r in null_rows) / max(null_tot, 1))

                alt_cols = ""
                for cond in all_conds:
                    alt_rows = grouped.get((et, method, n, cond), [])
                    if alt_rows:
                        alt_rej = int(sum(r.rejects for r in alt_rows))
                        alt_tot = int(sum(r.n_reps for r in alt_rows))
                        power, _, a_lo, a_hi = _mc_stats(alt_rej, alt_tot)
                        alt_cols += f" {power:>8.3f} {f'{a_lo:.3f}-{a_hi:.3f}':>15}"
                    else:
                        alt_cols += f" {'n/a':>8} {'':>15}"

                print(
                    f"  {method:<18} {n:>5d} {type1:>8.3f} {n_mcse:>8.3f} {f'{n_lo:.3f}-{n_hi:.3f}':>15}"
                    + alt_cols
                    + f" {mean_p0:>9.3f}"
                )


def print_multiarm_report(results: list[MultiArmResult]) -> None:
    print("\n" + "=" * 78)
    print("  MULTI-ARM MULTIPLICITY COMPARISON")
    print("  Metrics: FWER under global null, and true-best selection under alternative")
    print("=" * 78)

    key = lambda r: (r.eval_type, r.correction, r.n, r.condition)
    grouped: dict[tuple[str, str, int, str], list[MultiArmResult]] = defaultdict(list)
    for r in results:
        grouped[key(r)].append(r)

    sizes = sorted({r.n for r in results})

    for et in EVAL_TYPES:
        if not any(r.eval_type == et for r in results):
            continue
        print(f"\n{'-' * 78}")
        print(f"  {et.upper()}")
        print(f"{'-' * 78}")
        print(
            f"  {'correction':<16} {'n':>5} {'fwer':>8} {'mcse_fw':>8} {'fwer95':>15} {'best_power':>11} {'mcse_best':>10} {'best95':>15}"
        )
        print(f"  {'-' * 16} {'-' * 5} {'-' * 8} {'-' * 8} {'-' * 15} {'-' * 11} {'-' * 10} {'-' * 15}")

        for correction in MULTIARM_CORRECTIONS:
            for n in sizes:
                null_rows = grouped.get((et, correction, n, "null"), [])
                alt_rows = grouped.get((et, correction, n, "alt"), [])
                if not null_rows or not alt_rows:
                    continue

                n_any = int(sum(r.any_reject for r in null_rows))
                n_tot = int(sum(r.n_reps for r in null_rows))
                a_best = int(sum(r.best_selected for r in alt_rows))
                a_tot = int(sum(r.n_reps for r in alt_rows))

                fwer, fwer_mcse, n_lo, n_hi = _mc_stats(n_any, n_tot)
                best, best_mcse, a_lo, a_hi = _mc_stats(a_best, a_tot)

                print(
                    f"  {correction:<16} {n:>5d} {fwer:>8.3f} {fwer_mcse:>8.3f} {f'{n_lo:.3f}-{n_hi:.3f}':>15} "
                    f"{best:>11.3f} {best_mcse:>10.3f} {f'{a_lo:.3f}-{a_hi:.3f}':>15}"
                )


def save_pairwise_metric_plot(
    *,
    results: list[PairwiseResult],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
    out_path: str,
    sample_size_filter: int | None = None,
) -> None:
    plot_results = results
    if sample_size_filter is not None:
        plot_results = [r for r in results if r.n == sample_size_filter]

    if not plot_results:
        print(f"Skipped pairwise plot (no matching data): {out_path}")
        return

    # Discover alternative conditions; sort weaker effects before stronger ones.
    alt_conds = sorted(
        {r.condition for r in plot_results if r.condition != "null"},
        key=lambda c: (c.count("_"), c),
    )
    # Columns: type-I | one per alt condition | mean null p-value (calibration)
    n_data_cols = 1 + len(alt_conds)
    n_cols = n_data_cols + 1
    methods = sorted({r.method for r in plot_results})

    fig_width = 7.8 * n_cols
    fig, axes = plt.subplots(
        nrows=len(EVAL_TYPES),
        ncols=n_cols,
        figsize=(fig_width, 11.5),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.34, "hspace": 0.30},
    )
    if len(EVAL_TYPES) == 1:
        axes = np.array([axes])
    # Ensure axes is always 2-D.
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    col_titles = (
        [f"Type-I Error (target={alpha:.2f}; red=MC95)"]
        + [f"Power [{c}] (red=MC95)" for c in alt_conds]
        + ["Mean null p-value (target≈0.50; red=mean±SE)"]
    )

    box_kwargs: dict[str, Any] = {
        "vert": False,
        "showmeans": True,
        "meanline": False,
        "patch_artist": False,
        "whiskerprops": {"linewidth": 1.2, "color": "black"},
        "capprops": {"linewidth": 1.2, "color": "black"},
        "medianprops": {"linewidth": 1.8, "color": "black"},
        "boxprops": {"linewidth": 1.4, "color": "black"},
        "meanprops": {"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4.5},
        "flierprops": {"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 2.8, "alpha": 0.55},
    }

    for r_idx, et in enumerate(EVAL_TYPES):
        et_results = [res for res in plot_results if res.eval_type == et]
        et_methods = [m for m in methods if any(res.method == m for res in et_results)]

        # Gather reject-rate series and uncertainty for the data columns.
        all_series: list[list[np.ndarray]] = [[] for _ in range(n_data_cols)]
        all_uncert: list[list[tuple[float, float, float]]] = [[] for _ in range(n_data_cols)]
        all_conditions = ["null"] + alt_conds
        # Separate series for the mean null p-value column.
        mean_p_series: list[np.ndarray] = []
        mean_p_uncert: list[tuple[float, float, float]] = []

        for method in et_methods:
            for c_idx, cond in enumerate(all_conditions):
                rows = [res for res in et_results if res.method == method and res.condition == cond]
                vals = np.array([res.rejects / res.n_reps for res in rows], dtype=float)
                if vals.size == 0:
                    all_series[c_idx].append(np.array([float("nan")]))
                    all_uncert[c_idx].append((float("nan"), float("nan"), float("nan")))
                    continue
                tot = int(sum(res.n_reps for res in rows))
                rej = int(sum(res.rejects for res in rows))
                p_hat, _, lo, hi = _mc_stats(rej, tot)
                all_series[c_idx].append(vals)
                all_uncert[c_idx].append((p_hat, lo, hi))

            # Mean null p-value series: p_sum / n_reps for each (scenario, n) cell.
            null_rows = [res for res in et_results if res.method == method and res.condition == "null"]
            if null_rows:
                mp_vals = np.array([res.p_sum / res.n_reps for res in null_rows], dtype=float)
                mean_p_series.append(mp_vals)
                mp_mean = float(np.mean(mp_vals))
                mp_se = float(np.std(mp_vals, ddof=1) / np.sqrt(len(mp_vals))) if len(mp_vals) > 1 else 0.0
                mean_p_uncert.append((mp_mean, max(0.0, mp_mean - 1.96 * mp_se), min(1.0, mp_mean + 1.96 * mp_se)))
            else:
                mean_p_series.append(np.array([float("nan")]))
                mean_p_uncert.append((float("nan"), float("nan"), float("nan")))

        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]

            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=10)

            is_mean_p_col = (c_idx == n_cols - 1)
            series = mean_p_series if is_mean_p_col else all_series[c_idx]
            uncert = mean_p_uncert if is_mean_p_col else all_uncert[c_idx]
            valid = [s for s in series if not (s.size == 1 and np.isnan(s[0]))]

            if not valid:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                ax.set_yticks([])
                continue

            bp = ax.boxplot(
                [s for s in series],
                tick_labels=et_methods[: len(series)],
                **box_kwargs,
            )
            ax.grid(axis="x", linestyle="--", linewidth=0.65, alpha=0.50)
            if c_idx == 0:
                xlabel = "Type-I error across scenarios × n"
            elif is_mean_p_col:
                xlabel = "Mean null p-value across scenarios × n"
            else:
                xlabel = f"Power [{all_conditions[c_idx]}] across scenarios × n"
            ax.set_xlabel(xlabel, fontsize=9)
            ax.tick_params(axis="y", labelsize=9, pad=2)
            ax.tick_params(axis="x", labelsize=8.5)
            ax.invert_yaxis()
            ax.set_xlim(0.0, 1.0)

            if c_idx == 0:
                low_ok = max(0.0, alpha - 0.02)
                hi_ok = min(1.0, alpha + 0.02)
                ax.axvspan(low_ok, hi_ok, color="#DDDDDD", alpha=0.35, zorder=0)
                ax.axvline(alpha, color="black", linestyle="-", linewidth=1.2)
            elif is_mean_p_col:
                # Well-calibrated tests should have mean null p ≈ 0.5.
                ax.axvline(0.5, color="black", linestyle="-", linewidth=1.2)
                ax.axvspan(0.40, 0.60, color="#DDDDDD", alpha=0.35, zorder=0)

            for y_pos, (p_hat, lo, hi) in enumerate(uncert, start=1):
                if np.isnan(lo) or np.isnan(hi):
                    continue
                ax.hlines(y=y_pos, xmin=lo, xmax=hi, color="tab:red", linewidth=2.1, zorder=5)
                ax.vlines([lo, hi], y_pos - 0.15, y_pos + 0.15, color="tab:red", linewidth=1.5, zorder=5)
                if not np.isnan(p_hat):
                    ax.plot(p_hat, y_pos, marker="|", color="tab:red", markersize=10, markeredgewidth=1.8, zorder=6)

            if c_idx == 0:
                ax.set_ylabel(et.upper(), fontsize=10.5)

            if bp and "means" in bp:
                for mean_artist in bp["means"]:
                    mean_artist.set_zorder(4)

    if sample_size_filter is not None:
        size_text = str(sample_size_filter)
    elif sample_sizes:
        size_text = ", ".join(str(n) for n in sample_sizes)
    else:
        unique_sizes = sorted({r.n for r in plot_results})
        size_text = ", ".join(str(n) for n in unique_sizes) if unique_sizes else "n/a"

    fig.suptitle(
        f"Pairwise P-value Method Comparison by Eval Type\n"
        f"reps={n_reps} | alpha={alpha} | n={size_text}",
        fontsize=13.5,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout(rect=[0.03, 0.02, 0.995, 0.95], w_pad=2.6)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out}")


def save_multiarm_metric_plot(
    *,
    results: list[MultiArmResult],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
    out_path: str,
    sample_size_filter: int | None = None,
) -> None:
    plot_results = results
    if sample_size_filter is not None:
        plot_results = [r for r in results if r.n == sample_size_filter]

    if not plot_results:
        print(f"Skipped multi-arm plot (no matching data): {out_path}")
        return

    corrections = [c for c in MULTIARM_CORRECTIONS if any(r.correction == c for r in plot_results)]

    fig, axes = plt.subplots(
        nrows=len(EVAL_TYPES),
        ncols=2,
        figsize=(14.8, 11.5),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.34, "hspace": 0.30},
    )
    if len(EVAL_TYPES) == 1:
        axes = np.array([axes])

    col_titles = [
        f"FWER under null (target≤{alpha:.2f}; red interval = MC95)",
        "True-best selection power (red interval = MC95)",
    ]

    box_kwargs: dict[str, Any] = {
        "vert": False,
        "showmeans": True,
        "meanline": False,
        "patch_artist": False,
        "whiskerprops": {"linewidth": 1.2, "color": "black"},
        "capprops": {"linewidth": 1.2, "color": "black"},
        "medianprops": {"linewidth": 1.8, "color": "black"},
        "boxprops": {"linewidth": 1.4, "color": "black"},
        "meanprops": {"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4.5},
        "flierprops": {"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 2.8, "alpha": 0.55},
    }

    for r_idx, et in enumerate(EVAL_TYPES):
        et_results = [res for res in plot_results if res.eval_type == et]
        et_corr = [c for c in corrections if any(res.correction == c for res in et_results)]

        fwer_series: list[np.ndarray] = []
        best_series: list[np.ndarray] = []
        fwer_uncertainty: list[tuple[float, float, float]] = []
        best_uncertainty: list[tuple[float, float, float]] = []

        for corr in et_corr:
            null_rows = [res for res in et_results if res.correction == corr and res.condition == "null"]
            alt_rows = [res for res in et_results if res.correction == corr and res.condition == "alt"]
            fwer_vals = np.array([res.any_reject / res.n_reps for res in null_rows], dtype=float)
            best_vals = np.array([res.best_selected / res.n_reps for res in alt_rows], dtype=float)
            if fwer_vals.size == 0 or best_vals.size == 0:
                continue

            fwer_series.append(fwer_vals)
            best_series.append(best_vals)

            fwer_tot = int(sum(res.n_reps for res in null_rows))
            fwer_hits = int(sum(res.any_reject for res in null_rows))
            best_tot = int(sum(res.n_reps for res in alt_rows))
            best_hits = int(sum(res.best_selected for res in alt_rows))
            p_fw, _, lo_fw, hi_fw = _mc_stats(fwer_hits, fwer_tot)
            p_bs, _, lo_bs, hi_bs = _mc_stats(best_hits, best_tot)
            fwer_uncertainty.append((p_fw, lo_fw, hi_fw))
            best_uncertainty.append((p_bs, lo_bs, hi_bs))

        metric_series = [fwer_series, best_series]
        metric_uncertainty = [fwer_uncertainty, best_uncertainty]
        metric_xlabels = [
            "FWER across scenarios × n",
            "Best-selection power across scenarios × n",
        ]

        for c_idx, (ax, series, uncert, xlabel) in enumerate(
            zip(axes[r_idx], metric_series, metric_uncertainty, metric_xlabels)
        ):
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=11)

            if not series:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                ax.set_yticks([])
                continue

            bp = ax.boxplot(series, tick_labels=et_corr[: len(series)], **box_kwargs)
            ax.grid(axis="x", linestyle="--", linewidth=0.65, alpha=0.50)
            ax.set_xlabel(xlabel, fontsize=9.5)
            ax.tick_params(axis="y", labelsize=9.5, pad=2)
            ax.tick_params(axis="x", labelsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0.0, 1.0)

            if c_idx == 0:
                ax.axvline(alpha, color="black", linestyle="-", linewidth=1.2)
                ax.axvspan(0.0, alpha, color="#DDDDDD", alpha=0.25, zorder=0)

            for y_pos, (p_hat, lo, hi) in enumerate(uncert, start=1):
                if np.isnan(lo) or np.isnan(hi):
                    continue
                ax.hlines(y=y_pos, xmin=lo, xmax=hi, color="tab:red", linewidth=2.1, zorder=5)
                ax.vlines([lo, hi], y_pos - 0.15, y_pos + 0.15, color="tab:red", linewidth=1.5, zorder=5)
                if not np.isnan(p_hat):
                    ax.plot(p_hat, y_pos, marker="|", color="tab:red", markersize=10, markeredgewidth=1.8, zorder=6)

            if c_idx == 0:
                ax.set_ylabel(et.upper(), fontsize=10.5)

            if bp and "means" in bp:
                for mean_artist in bp["means"]:
                    mean_artist.set_zorder(4)

    if sample_size_filter is not None:
        size_text = str(sample_size_filter)
    elif sample_sizes:
        size_text = ", ".join(str(n) for n in sample_sizes)
    else:
        unique_sizes = sorted({r.n for r in plot_results})
        size_text = ", ".join(str(n) for n in unique_sizes) if unique_sizes else "n/a"

    fig.suptitle(
        f"Multi-arm Multiplicity Comparison by Eval Type\n"
        f"reps={n_reps} | alpha={alpha} | n={size_text}",
        fontsize=13.5,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout(rect=[0.03, 0.02, 0.995, 0.95], w_pad=2.6)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out}")


def save_results_artifacts(
    *,
    pairwise_results: list[PairwiseResult],
    multiarm_results: list[MultiArmResult],
    out_dir: str,
    run_stem: str,
    alpha: float,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if pairwise_results:
        pair_csv = out / f"{run_stem}_pairwise.csv"
        with pair_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "eval_type",
                    "scenario",
                    "n",
                    "runs",
                    "method",
                    "condition",
                    "cohens_d",
                    "n_reps",
                    "rejects",
                    "rate",
                    "p_mean",
                    "mcse",
                    "band95_low",
                    "band95_high",
                ]
            )
            for r in pairwise_results:
                rate, mcse, lo, hi = _mc_stats(r.rejects, r.n_reps)
                p_mean = r.p_sum / max(r.n_reps, 1)
                writer.writerow(
                    [
                        r.eval_type,
                        r.scenario,
                        r.n,
                        r.runs,
                        r.method,
                        r.condition,
                        f"{r.cohens_d:.6f}",
                        r.n_reps,
                        r.rejects,
                        f"{rate:.8f}",
                        f"{p_mean:.8f}",
                        f"{mcse:.8f}",
                        f"{lo:.8f}",
                        f"{hi:.8f}",
                    ]
                )
        print(f"Saved pairwise CSV: {pair_csv}")

    if multiarm_results:
        multi_csv = out / f"{run_stem}_multiarm.csv"
        with multi_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "eval_type",
                    "scenario",
                    "n",
                    "runs",
                    "k",
                    "correction",
                    "condition",
                    "n_reps",
                    "any_reject",
                    "best_selected",
                    "any_reject_rate",
                    "any_reject_mcse",
                    "best_selected_rate",
                    "best_selected_mcse",
                ]
            )
            for r in multiarm_results:
                any_rate = r.any_reject / max(r.n_reps, 1)
                best_rate = r.best_selected / max(r.n_reps, 1)
                any_mcse = float(np.sqrt(max(any_rate * (1.0 - any_rate), 0.0) / max(r.n_reps, 1)))
                best_mcse = float(np.sqrt(max(best_rate * (1.0 - best_rate), 0.0) / max(r.n_reps, 1)))
                writer.writerow(
                    [
                        r.eval_type,
                        r.scenario,
                        r.n,
                        r.runs,
                        r.k,
                        r.correction,
                        r.condition,
                        r.n_reps,
                        r.any_reject,
                        r.best_selected,
                        f"{any_rate:.8f}",
                        f"{any_mcse:.8f}",
                        f"{best_rate:.8f}",
                        f"{best_mcse:.8f}",
                    ]
                )
        print(f"Saved multi-arm CSV: {multi_csv}")

    summary_log = out / f"{run_stem}_summary.log"
    text = io.StringIO()
    with redirect_stdout(text):
        if pairwise_results:
            print_pairwise_report(pairwise_results, alpha=alpha)
        if multiarm_results:
            print_multiarm_report(multiarm_results)
    summary_log.write_text(text.getvalue(), encoding="utf-8")
    print(f"Saved summary log: {summary_log}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50, 100], help="Sample sizes")
    parser.add_argument("--runs", type=int, default=1, help="Runs per input (R>=3 activates nested compare paths; binary-only methods use majority voting when R>1)")
    parser.add_argument("--reps", type=int, default=500, help="Monte Carlo reps per cell")
    parser.add_argument("--bootstrap-n", type=int, default=2000, help="Bootstrap/permutation samples per method")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--statistic", choices=["mean", "median"], default="mean", help="Statistic for paired effects")
    parser.add_argument("--k-arms", type=int, default=4, help="Number of templates/models in multi-arm phase")
    parser.add_argument(
        "--multiarm-method",
        choices=["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "permutation"],
        default="smooth_bootstrap",
        help="Pairwise method used inside the multi-arm phase (bootstrap p-values computed once; corrections applied to those same p-values)",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes (default: 1)")
    parser.add_argument("--pairwise-only", action="store_true", help="Run only pairwise phase")
    parser.add_argument("--multiarm-only", action="store_true", help="Run only multi-arm phase")
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar", help="Progress display mode")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save", help="Post-run plotting mode")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save", help="Save CSV + summary log")
    parser.add_argument("--out-dir", default="simulations/out", help="Output directory")
    parser.add_argument("--plots-dir", default=None, help="Directory for saved plots when --plots save (default: <out-dir>/plots)")
    parser.add_argument(
        "--include-extreme",
        action="store_true",
        help="Include additional extreme/distribution-shift stress scenarios in pairwise and multi-arm phases",
    )
    parser.add_argument(
        "--mcse-target",
        type=float,
        default=0.005,
        help="Target Monte Carlo standard error for key Bernoulli rates; script reports required reps and warns if --reps is lower (set <=0 to disable)",
    )
    args = parser.parse_args()
    plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")

    if args.pairwise_only and args.multiarm_only:
        parser.error("Choose at most one of --pairwise-only or --multiarm-only")

    run_pairwise = not args.multiarm_only
    run_multiarm = not args.pairwise_only

    print("\nP-value method simulation")
    print(f"  Sample sizes     : {args.sizes}")
    print(f"  Runs per input   : {args.runs}")
    print(f"  Reps per cell    : {args.reps}")
    print(f"  Bootstrap draws  : {args.bootstrap_n}")
    print(f"  Alpha            : {args.alpha}")
    print(f"  Statistic        : {args.statistic}")
    print(f"  Multi-arm K      : {args.k_arms}")
    print(f"  Multi-arm method : {args.multiarm_method}")
    print(f"  Seed             : {args.seed}")
    print(f"  Jobs             : {args.jobs}")
    print(f"  Progress         : {args.progress}")
    print(f"  Plots            : {args.plots}")
    print(f"  Save results     : {args.save_results}")
    print(f"  Out dir          : {args.out_dir}")
    print(f"  Extreme sweeps   : {'on' if args.include_extreme else 'off'}")
    print(f"  MCSE target      : {args.mcse_target}")
    if args.plots == "save":
        print(f"  Plots dir        : {plots_dir}")

    required_reps = _min_reps_for_mcse(args.mcse_target)
    if required_reps > 0:
        achieved_mcse = np.sqrt(0.25 / max(args.reps, 1))
        print(f"  MCSE reps req    : {required_reps} (worst-case Bernoulli)")
        print(f"  MCSE at --reps   : {achieved_mcse:.5f}")
        if args.reps < required_reps:
            print(
                f"  WARNING          : --reps={args.reps} is below MCSE target requirement; "
                f"consider --reps >= {required_reps}"
            )

    pairwise_results: list[PairwiseResult] = []
    multiarm_results: list[MultiArmResult] = []

    if run_pairwise:
        scenarios = build_pair_scenarios(include_extreme=args.include_extreme)
        print(f"\nRunning pairwise phase: {len(scenarios)} scenarios")
        pairwise_results = run_pairwise_simulation(
            scenarios=scenarios,
            sample_sizes=args.sizes,
            runs=args.runs,
            n_reps=args.reps,
            n_bootstrap=args.bootstrap_n,
            alpha=args.alpha,
            statistic=args.statistic,
            progress_mode=args.progress,
            seed=args.seed,
            n_jobs=args.jobs,
        )
        print_pairwise_report(pairwise_results, alpha=args.alpha)
        if args.plots == "save":
            stamp = time.strftime("%Y%m%d_%H%M%S")
            overall_name = f"sim_compare_pvalues_pairwise_runs{args.runs}_reps{args.reps}_overall_{stamp}.png"
            save_pairwise_metric_plot(
                results=pairwise_results,
                sample_sizes=args.sizes,
                alpha=args.alpha,
                n_reps=args.reps,
                out_path=str(Path(plots_dir) / overall_name),
            )
            for n in args.sizes:
                per_n_name = f"sim_compare_pvalues_pairwise_runs{args.runs}_reps{args.reps}_n{n}_{stamp}.png"
                save_pairwise_metric_plot(
                    results=pairwise_results,
                    sample_sizes=[n],
                    alpha=args.alpha,
                    n_reps=args.reps,
                    out_path=str(Path(plots_dir) / per_n_name),
                    sample_size_filter=n,
                )

    if run_multiarm:
        scenarios = build_multiarm_scenarios(include_extreme=args.include_extreme)
        print(f"\nRunning multi-arm phase: {len(scenarios)} scenarios")
        multiarm_results = run_multiarm_simulation(
            scenarios=scenarios,
            sample_sizes=args.sizes,
            runs=args.runs,
            k_arms=args.k_arms,
            n_reps=args.reps,
            n_bootstrap=args.bootstrap_n,
            alpha=args.alpha,
            multiarm_method=args.multiarm_method,
            statistic=args.statistic,
            progress_mode=args.progress,
            seed=args.seed + 100,
            n_jobs=args.jobs,
        )
        print_multiarm_report(multiarm_results)
        if args.plots == "save":
            stamp = time.strftime("%Y%m%d_%H%M%S")
            overall_name = f"sim_compare_pvalues_multiarm_runs{args.runs}_reps{args.reps}_k{args.k_arms}_overall_{stamp}.png"
            save_multiarm_metric_plot(
                results=multiarm_results,
                sample_sizes=args.sizes,
                alpha=args.alpha,
                n_reps=args.reps,
                out_path=str(Path(plots_dir) / overall_name),
            )
            for n in args.sizes:
                per_n_name = f"sim_compare_pvalues_multiarm_runs{args.runs}_reps{args.reps}_k{args.k_arms}_n{n}_{stamp}.png"
                save_multiarm_metric_plot(
                    results=multiarm_results,
                    sample_sizes=[n],
                    alpha=args.alpha,
                    n_reps=args.reps,
                    out_path=str(Path(plots_dir) / per_n_name),
                    sample_size_filter=n,
                )

    if args.save_results == "save":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stem = (
            f"sim_compare_pvalues_runs{args.runs}_reps{args.reps}_"
            f"boot{args.bootstrap_n}_k{args.k_arms}_{stamp}"
        )
        save_results_artifacts(
            pairwise_results=pairwise_results,
            multiarm_results=multiarm_results,
            out_dir=args.out_dir,
            run_stem=stem,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
