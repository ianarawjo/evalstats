#!/usr/bin/env python3
"""
sim_compare_boot.py — Bootstrap CI method comparison across eval score types.

Compares four bootstrap confidence-interval methods by running Monte Carlo
simulations where the true population mean is known exactly.

Methods:
  bootstrap        Percentile bootstrap
  bca              Bias-corrected and accelerated (BCa) bootstrap
  bayes_bootstrap  Bayesian (Dirichlet-weighted) bootstrap
  smooth_bootstrap Smoothed (KDE-perturbed) bootstrap

Eval output types:
  binary      Bernoulli 0/1 (pass/fail judgements)
  continuous  Continuous floats in [0, 1] via Beta distributions
  likert      Integer scores 1–5 (Likert-scale rubrics)
  grades      Scores 0–100 (test-like, truncated normal)

Metrics:
  coverage    Fraction of (1-alpha)*100% CIs that contain the true mean
              Target: equals the nominal level, e.g. 0.95 for alpha=0.05
  mean_width  Average CI width — a measure of precision (smaller is better
              when coverage is adequate)

Usage:
  python simulations/sim_compare_boot.py
  python simulations/sim_compare_boot.py --reps 500 --bootstrap-n 1000
  python simulations/sim_compare_boot.py --sizes 5 10 20 50 100
    python simulations/sim_compare_boot.py --estimand pairwise --runs 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from promptstats.core.resampling import (
        bootstrap_ci_1d,
        bca_interval_1d,
        bayes_bootstrap_means_1d,
        smooth_bootstrap_means_1d,
        bootstrap_means_1d,
        bootstrap_diffs_nested,
        bayes_bootstrap_diffs_nested,
        smooth_bootstrap_diffs_nested,
        resolve_resampling_method,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap"]
WILSON_METHOD = "wilson"
REPORT_METHODS = METHODS + [WILSON_METHOD]
EVAL_TYPES = ["binary", "continuous", "likert", "grades"]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    label: str
    eval_type: str
    generate: Callable[[np.random.Generator, int], np.ndarray]
    true_mean: float


@dataclass
class PairScenario:
    label: str
    eval_type: str
    generate_pair: Callable[[np.random.Generator, int, int], tuple[np.ndarray, np.ndarray]]
    true_diff: float


def _true_mean_clipped_normal(
    mu: float, sigma: float, lo: float = 0.0, hi: float = 100.0
) -> float:
    """Population mean of Normal(mu, sigma) clipped to [lo, hi] via large sample."""
    rng = np.random.default_rng(0)
    return float(np.clip(rng.normal(mu, sigma, size=2_000_000), lo, hi).mean())


def build_scenarios() -> list[Scenario]:
    """Return all simulation scenarios across the four eval types."""
    scenarios: list[Scenario] = []

    # ------------------------------------------------------------------
    # Binary: Bernoulli(p)  →  0/1 pass-fail judgements
    # ------------------------------------------------------------------
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        p_ = p
        scenarios.append(
            Scenario(
                label=f"p={p_}",
                eval_type="binary",
                generate=lambda rng, n, _p=p_: rng.binomial(1, _p, n).astype(float),
                true_mean=p_,
            )
        )

    # ------------------------------------------------------------------
    # Continuous [0, 1]: Beta distributions of varying shape
    # ------------------------------------------------------------------
    for label, a, b in [
        ("Uniform",        1.0, 1.0),   # flat
        ("U-shaped",       0.5, 0.5),   # bimodal-ish extremes
        ("right-skewed",   2.0, 8.0),   # mass near 0
        ("left-skewed",    8.0, 2.0),   # mass near 1
        ("moderate-skew",  2.0, 5.0),   # asymmetric centre
    ]:
        a_, b_ = a, b
        scenarios.append(
            Scenario(
                label=f"{label} Beta({a_},{b_})",
                eval_type="continuous",
                generate=lambda rng, n, _a=a_, _b=b_: rng.beta(_a, _b, n),
                true_mean=a_ / (a_ + b_),
            )
        )

    # ------------------------------------------------------------------
    # Likert 1–5: discrete integer scores
    # ------------------------------------------------------------------
    likert_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for label, probs in [
        ("uniform",       [0.20, 0.20, 0.20, 0.20, 0.20]),
        ("skewed-low",    [0.40, 0.30, 0.15, 0.10, 0.05]),
        ("skewed-high",   [0.05, 0.10, 0.15, 0.30, 0.40]),
        ("bimodal",       [0.35, 0.10, 0.10, 0.10, 0.35]),
        ("center-peaked", [0.05, 0.15, 0.60, 0.15, 0.05]),
    ]:
        probs_ = np.array(probs, dtype=float)
        true_mean_ = float(np.dot(likert_vals, probs_))
        scenarios.append(
            Scenario(
                label=label,
                eval_type="likert",
                generate=lambda rng, n, _p=probs_: rng.choice(
                    np.array([1.0, 2.0, 3.0, 4.0, 5.0]), size=n, p=_p
                ),
                true_mean=true_mean_,
            )
        )

    # ------------------------------------------------------------------
    # Grades 0–100: truncated normals of varying centre and spread
    # ------------------------------------------------------------------
    for label, mu, sigma in [
        ("symmetric",     50, 20),   # centred, moderate spread
        ("high-scoring",  75, 15),   # near ceiling
        ("low-scoring",   35, 20),   # near floor
        ("ceiling-heavy", 88, 10),   # mass near 100 — heavy clipping
        ("floor-heavy",   12, 10),   # mass near 0   — heavy clipping
    ]:
        mu_, sigma_ = mu, sigma
        true_mean_ = _true_mean_clipped_normal(mu_, sigma_)
        scenarios.append(
            Scenario(
                label=f"{label} N({mu_},{sigma_})",
                eval_type="grades",
                generate=lambda rng, n, _m=mu_, _s=sigma_: np.clip(
                    rng.normal(_m, _s, n), 0.0, 100.0
                ),
                true_mean=true_mean_,
            )
        )

    return scenarios


def _estimate_true_pair_diff(
    generate_pair: Callable[[np.random.Generator, int, int], tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = 0,
    n_mc: int = 300_000,
) -> float:
    """Estimate E[cell_mean(A) - cell_mean(B)] via a large synthetic sample."""
    rng = np.random.default_rng(seed)
    a, b = generate_pair(rng, n_mc, 1)
    return float(np.mean(a[:, 0] - b[:, 0]))


def build_pair_scenarios() -> list[PairScenario]:
    """Return paired-difference scenarios mirroring prompt/model comparisons."""
    scenarios: list[PairScenario] = []

    # Binary 0/1 with paired input difficulty and a fixed uplift for template B.
    for label, base_p, delta in [
        ("binary-balanced", 0.5, 0.08),
        ("binary-high", 0.8, 0.05),
        ("binary-low", 0.2, 0.05),
    ]:
        base_p_, delta_ = base_p, delta

        def _gen_binary(
            rng: np.random.Generator,
            n: int,
            runs: int,
            _bp: float = base_p_,
            _d: float = delta_,
        ) -> tuple[np.ndarray, np.ndarray]:
            conc = 12.0
            alpha = _bp * conc
            beta = (1.0 - _bp) * conc
            p_a = rng.beta(alpha, beta, size=(n, 1))
            p_b = np.clip(p_a + _d, 0.0, 1.0)
            a = rng.binomial(1, p_a, size=(n, runs)).astype(float)
            b = rng.binomial(1, p_b, size=(n, runs)).astype(float)
            return a, b

        scenarios.append(
            PairScenario(
                label=label,
                eval_type="binary",
                generate_pair=_gen_binary,
                true_diff=_estimate_true_pair_diff(_gen_binary),
            )
        )

    # Continuous [0,1] with paired latent difficulty + bounded noise.
    for label, a, b, delta, sigma in [
        ("continuous-uniform", 1.0, 1.0, 0.06, 0.10),
        ("continuous-right-skew", 2.0, 8.0, 0.04, 0.08),
        ("continuous-left-skew", 8.0, 2.0, 0.04, 0.08),
    ]:
        a_, b_, delta_, sigma_ = a, b, delta, sigma

        def _gen_continuous(
            rng: np.random.Generator,
            n: int,
            runs: int,
            _a: float = a_,
            _b: float = b_,
            _d: float = delta_,
            _s: float = sigma_,
        ) -> tuple[np.ndarray, np.ndarray]:
            base = rng.beta(_a, _b, size=(n, 1))
            shared = rng.normal(0.0, _s, size=(n, runs))
            indiv_a = rng.normal(0.0, _s * 0.5, size=(n, runs))
            indiv_b = rng.normal(0.0, _s * 0.5, size=(n, runs))
            a_vals = np.clip(base + shared + indiv_a, 0.0, 1.0)
            b_vals = np.clip(base + _d + shared + indiv_b, 0.0, 1.0)
            return a_vals, b_vals

        scenarios.append(
            PairScenario(
                label=label,
                eval_type="continuous",
                generate_pair=_gen_continuous,
                true_diff=_estimate_true_pair_diff(_gen_continuous),
            )
        )

    # Likert 1-5 via rounded latent scores.
    for label, mu, sigma, delta in [
        ("likert-mid", 3.0, 0.9, 0.35),
        ("likert-low", 2.2, 0.8, 0.30),
        ("likert-high", 3.8, 0.8, 0.30),
    ]:
        mu_, sigma_, delta_ = mu, sigma, delta

        def _gen_likert(
            rng: np.random.Generator,
            n: int,
            runs: int,
            _m: float = mu_,
            _s: float = sigma_,
            _d: float = delta_,
        ) -> tuple[np.ndarray, np.ndarray]:
            base = rng.normal(_m, _s, size=(n, 1))
            shared = rng.normal(0.0, 0.35, size=(n, runs))
            indiv_a = rng.normal(0.0, 0.25, size=(n, runs))
            indiv_b = rng.normal(0.0, 0.25, size=(n, runs))
            a_vals = np.rint(np.clip(base + shared + indiv_a, 1.0, 5.0))
            b_vals = np.rint(np.clip(base + _d + shared + indiv_b, 1.0, 5.0))
            return a_vals, b_vals

        scenarios.append(
            PairScenario(
                label=label,
                eval_type="likert",
                generate_pair=_gen_likert,
                true_diff=_estimate_true_pair_diff(_gen_likert),
            )
        )

    # Grades 0-100 with shared latent ability and bounded noise.
    for label, mu, sigma, delta in [
        ("grades-mid", 55.0, 18.0, 4.5),
        ("grades-low", 35.0, 16.0, 4.0),
        ("grades-high", 78.0, 14.0, 3.5),
    ]:
        mu_, sigma_, delta_ = mu, sigma, delta

        def _gen_grades(
            rng: np.random.Generator,
            n: int,
            runs: int,
            _m: float = mu_,
            _s: float = sigma_,
            _d: float = delta_,
        ) -> tuple[np.ndarray, np.ndarray]:
            base = rng.normal(_m, _s, size=(n, 1))
            shared = rng.normal(0.0, _s * 0.18, size=(n, runs))
            indiv_a = rng.normal(0.0, _s * 0.12, size=(n, runs))
            indiv_b = rng.normal(0.0, _s * 0.12, size=(n, runs))
            a_vals = np.clip(base + shared + indiv_a, 0.0, 100.0)
            b_vals = np.clip(base + _d + shared + indiv_b, 0.0, 100.0)
            return a_vals, b_vals

        scenarios.append(
            PairScenario(
                label=label,
                eval_type="grades",
                generate_pair=_gen_grades,
                true_diff=_estimate_true_pair_diff(_gen_grades),
            )
        )

    return scenarios


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    eval_type: str
    scenario: str
    n: int
    method: str
    n_reps: int
    covered: int      # number of reps where CI contained the true estimand
    total_width: float  # sum of CI widths across reps


def _stat(values: np.ndarray, statistic: str = "mean") -> float:
    return float(np.median(values)) if statistic == "median" else float(np.mean(values))


def _wilson_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    low = max(0.0, float(center - radius))
    high = min(1.0, float(center + radius))
    return low, high


def _pairwise_ci(
    a: np.ndarray,
    b: np.ndarray,
    method: str,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
    *,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Compute CI for paired mean/median difference A-B using promptstats logic."""
    n_inputs, runs = a.shape
    resolved_method = resolve_resampling_method(method, n_inputs)

    if runs >= 3:
        cell_diffs = a.mean(axis=1) - b.mean(axis=1)
        observed = _stat(cell_diffs, statistic=statistic)

        if resolved_method == "bayes_bootstrap":
            boot_stats = bayes_bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
            return (
                float(np.percentile(boot_stats, 100 * alpha / 2)),
                float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
            )
        if resolved_method == "smooth_bootstrap":
            boot_stats = smooth_bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
            return (
                float(np.percentile(boot_stats, 100 * alpha / 2)),
                float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
            )

        boot_stats = bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
        if resolved_method == "bca":
            return bca_interval_1d(cell_diffs, observed, boot_stats, alpha, statistic=statistic)
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )

    diffs = a.mean(axis=1) - b.mean(axis=1)
    observed = _stat(diffs, statistic=statistic)

    if resolved_method == "bayes_bootstrap":
        boot_stats = bayes_bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )
    if resolved_method == "smooth_bootstrap":
        boot_stats = smooth_bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )

    boot_stats = bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
    if resolved_method == "bca":
        return bca_interval_1d(diffs, observed, boot_stats, alpha, statistic=statistic)
    return (
        float(np.percentile(boot_stats, 100 * alpha / 2)),
        float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
    )


def run_simulation(
    scenarios: list[Scenario],
    sample_sizes: list[int],
    n_reps: int,
    n_bootstrap: int,
    alpha: float,
    seed: int = 42,
) -> list[SimResult]:
    rng = np.random.default_rng(seed)
    results: list[SimResult] = []

    total = len(scenarios) * len(sample_sizes)
    done = 0

    for scenario in scenarios:
        for n in sample_sizes:
            done += 1
            print(
                f"\r  [{done:3d}/{total}] "
                f"{scenario.eval_type:<12s}  "
                f"{scenario.label:<30s}  n={n:<3d}   ",
                end="",
                flush=True,
            )

            active_methods = METHODS.copy()
            if scenario.eval_type == "binary":
                active_methods.append(WILSON_METHOD)

            covered: dict[str, int] = {m: 0 for m in active_methods}
            total_w: dict[str, float] = {m: 0.0 for m in active_methods}

            for _ in range(n_reps):
                values = scenario.generate(rng, n)
                obs_mean = float(np.mean(values))

                for method in METHODS:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            ci_low, ci_high = bootstrap_ci_1d(
                                values,
                                obs_mean,
                                method=method,
                                n_bootstrap=n_bootstrap,
                                alpha=alpha,
                                rng=rng,
                            )
                    except Exception:
                        # Degenerate case: point CI at observed mean
                        ci_low = ci_high = obs_mean

                    if ci_low <= scenario.true_mean <= ci_high:
                        covered[method] += 1
                    total_w[method] += ci_high - ci_low

                if scenario.eval_type == "binary":
                    successes = int(np.sum(values))
                    ci_low, ci_high = _wilson_ci(successes, n, alpha)
                    if ci_low <= scenario.true_mean <= ci_high:
                        covered[WILSON_METHOD] += 1
                    total_w[WILSON_METHOD] += ci_high - ci_low

            for method in active_methods:
                results.append(
                    SimResult(
                        eval_type=scenario.eval_type,
                        scenario=scenario.label,
                        n=n,
                        method=method,
                        n_reps=n_reps,
                        covered=covered[method],
                        total_width=total_w[method],
                    )
                )

    print()  # end progress line
    return results


def run_pairwise_simulation(
    scenarios: list[PairScenario],
    sample_sizes: list[int],
    n_reps: int,
    n_bootstrap: int,
    alpha: float,
    runs: int,
    statistic: str,
    seed: int = 42,
) -> list[SimResult]:
    rng = np.random.default_rng(seed)
    results: list[SimResult] = []

    total = len(scenarios) * len(sample_sizes)
    done = 0

    for scenario in scenarios:
        for n in sample_sizes:
            done += 1
            print(
                f"\r  [{done:3d}/{total}] "
                f"{scenario.eval_type:<12s}  "
                f"{scenario.label:<30s}  n={n:<3d}   ",
                end="",
                flush=True,
            )

            covered: dict[str, int] = {m: 0 for m in METHODS}
            total_w: dict[str, float] = {m: 0.0 for m in METHODS}

            for _ in range(n_reps):
                a, b = scenario.generate_pair(rng, n, runs)

                for method in METHODS:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            ci_low, ci_high = _pairwise_ci(
                                a,
                                b,
                                method=method,
                                n_bootstrap=n_bootstrap,
                                alpha=alpha,
                                rng=rng,
                                statistic=statistic,
                            )
                    except Exception:
                        obs = _stat(a.mean(axis=1) - b.mean(axis=1), statistic=statistic)
                        ci_low = ci_high = obs

                    if ci_low <= scenario.true_diff <= ci_high:
                        covered[method] += 1
                    total_w[method] += ci_high - ci_low

            for method in METHODS:
                results.append(
                    SimResult(
                        eval_type=scenario.eval_type,
                        scenario=scenario.label,
                        n=n,
                        method=method,
                        n_reps=n_reps,
                        covered=covered[method],
                        total_width=total_w[method],
                    )
                )

    print()
    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _cov_marker(cov: float, target: float, tol: float = 0.04) -> str:
    """Single-character flag indicating coverage quality."""
    if cov < target - tol:
        return "▼"   # systematically under-covered
    if cov > target + tol:
        return "▲"   # over-conservative
    return " "


def _rule(width: int, char: str = "─") -> str:
    return char * width


def _print_grid(
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    cells: dict[tuple[str, str], str],
    row_w: int = 20,
    col_w: int = 9,
) -> None:
    total_w = row_w + 2 + (col_w + 2) * len(col_labels)
    print(f"\n  {title}")
    print(f"  {_rule(total_w)}")
    header = f"  {'':<{row_w}}" + "".join(f"  {c:>{col_w}}" for c in col_labels)
    print(header)
    print(f"  {_rule(total_w)}")
    for row in row_labels:
        line = f"  {row:<{row_w}}"
        for col in col_labels:
            val = cells.get((row, col), "─" * col_w)
            line += f"  {val:>{col_w}}"
        print(line)
    print(f"  {_rule(total_w)}")


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------


def print_report(
    results: list[SimResult],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
    estimand_label: str,
) -> None:
    target = 1.0 - alpha
    n_labels = [f"n={n}" for n in sample_sizes]
    present_methods = {r.method for r in results}
    method_labels = [m for m in REPORT_METHODS if m in present_methods]

    # ----------------------------------------------------------------
    # Aggregate: mean coverage and mean width across scenarios within
    # each (eval_type, method, n).  Also keep per-scenario for worst-case.
    # ----------------------------------------------------------------
    # agg[(eval_type, method, n)] accumulates per-scenario (cov, width) pairs
    agg: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    per_sc: dict[tuple, tuple[float, float]] = {}  # (et, sc, method, n)

    for r in results:
        cov = r.covered / r.n_reps
        width = r.total_width / r.n_reps
        agg[(r.eval_type, r.method, r.n)].append((cov, width))
        per_sc[(r.eval_type, r.scenario, r.method, r.n)] = (cov, width)

    def mean_cov(et: str, m: str, n: int) -> float:
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[0] for v in vals])) if vals else float("nan")

    def mean_wid(et: str, m: str, n: int) -> float:
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[1] for v in vals])) if vals else float("nan")

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    sep = "=" * 72
    print(f"\n{sep}")
    print("  BOOTSTRAP CI COMPARISON  —  SIMULATION RESULTS")
    print(f"  Estimand: {estimand_label}")
    print(f"  Nominal coverage: {target:.0%}   |   reps/scenario: {n_reps}")
    print(f"  ▼ = under-covered (<{target - 0.04:.0%})   ▲ = over-conservative (>{target + 0.04:.0%})")
    print(sep)

    # ----------------------------------------------------------------
    # Per-eval-type tables
    # ----------------------------------------------------------------
    for et in EVAL_TYPES:
        print(f"\n{'─'*72}")
        print(f"  {et.upper()}")
        print(f"{'─'*72}")

        # Coverage grid
        cov_cells: dict[tuple[str, str], str] = {}
        for m in method_labels:
            for i, n in enumerate(sample_sizes):
                cov = mean_cov(et, m, n)
                if np.isnan(cov):
                    cov_cells[(m, n_labels[i])] = "─"
                else:
                    marker = _cov_marker(cov, target)
                    cov_cells[(m, n_labels[i])] = f"{cov:.3f}{marker}"

        _print_grid(
            f"Coverage (target {target:.2f})",
            row_labels=method_labels,
            col_labels=n_labels,
            cells=cov_cells,
        )

        # Width grid
        wid_cells: dict[tuple[str, str], str] = {}
        for m in method_labels:
            for i, n in enumerate(sample_sizes):
                w = mean_wid(et, m, n)
                wid_cells[(m, n_labels[i])] = "─" if np.isnan(w) else f"{w:.4f}"

        _print_grid(
            "Mean CI Width (lower = more precise)",
            row_labels=method_labels,
            col_labels=n_labels,
            cells=wid_cells,
        )

    # ----------------------------------------------------------------
    # Worst-coverage scenarios (averaged across methods)
    # ----------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("  WORST COVERAGE CASES  (averaged across all four methods)")
    print(f"{'─'*72}")

    # Average coverage across methods for each (eval_type, scenario, n)
    sc_cov: dict[tuple, list[float]] = defaultdict(list)
    for (et, sc, m, n), (cov, _) in per_sc.items():
        sc_cov[(et, sc, n)].append(cov)

    worst = sorted(
        [(float(np.mean(v)), k) for k, v in sc_cov.items()],
        key=lambda x: x[0],
    )[:12]

    print(f"\n  {'Eval Type':<12}  {'Scenario':<32}  {'n':>4}  {'Avg Coverage':>13}")
    print(f"  {'─'*12}  {'─'*32}  {'─'*4}  {'─'*13}")
    for cov, (et, sc, n) in worst:
        mark = _cov_marker(cov, target)
        print(f"  {et:<12}  {sc:<32}  {n:>4}  {cov:>12.3f}{mark}")

    # ----------------------------------------------------------------
    # Per-method breakdown: coverage deficit/surplus at each n
    # ----------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("  COVERAGE DEVIATION FROM NOMINAL  (mean_coverage − target)")
    print(f"  Computed across all scenarios within each eval type.")
    print(f"{'─'*72}")

    for et in EVAL_TYPES:
        print(f"\n  {et}")
        hdr = f"    {'Method':<20}" + "".join(f"  {nl:>9}" for nl in n_labels)
        print(hdr)
        print(f"    {'─'*20}" + "─" * (11 * len(n_labels)))
        for m in method_labels:
            row = f"    {m:<20}"
            for n in sample_sizes:
                dev = mean_cov(et, m, n) - target
                row += "  {:>9}".format("─") if np.isnan(dev) else f"  {dev:>+9.3f}"
            print(row)

    # ----------------------------------------------------------------
    # Overall summary across everything
    # ----------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("  OVERALL SUMMARY  (averaged across all eval types, scenarios, n)")
    print(f"{'─'*72}")

    all_cov: dict[str, list[float]] = defaultdict(list)
    all_wid: dict[str, list[float]] = defaultdict(list)
    for (et, m, n), vals in agg.items():
        all_cov[m].extend(v[0] for v in vals)
        all_wid[m].extend(v[1] for v in vals)

    print(f"\n  {'Method':<20}  {'Mean Coverage':>14}  {'Mean Width':>11}  {'Coverage−Target':>16}")
    print(f"  {'─'*20}  {'─'*14}  {'─'*11}  {'─'*16}")
    for m in method_labels:
        mc = float(np.mean(all_cov[m]))
        mw = float(np.mean(all_wid[m]))
        dev = mc - target
        mark = _cov_marker(mc, target)
        print(
            f"  {m:<20}  {mc:>13.3f}{mark}  {mw:>11.4f}  {dev:>+16.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--estimand",
        choices=["mean", "pairwise"],
        default="mean",
        help=(
            "Target estimand: 'mean' (single-sample means, default) or "
            "'pairwise' (paired template difference A-B, matching compare paths)."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="R",
        help=(
            "Runs per input for --estimand pairwise. R>=3 activates nested "
            "bootstrap logic (default: 1)."
        ),
    )
    parser.add_argument(
        "--statistic",
        choices=["mean", "median"],
        default="mean",
        help="Statistic for --estimand pairwise (default: mean)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=200,
        metavar="N",
        help="Monte Carlo repetitions per (scenario, n) cell (default: 200)",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=500,
        metavar="N",
        help="Bootstrap replicates per CI estimate (default: 500)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level; CIs are (1-alpha)*100%% (default: 0.05)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        metavar="N",
        help="Sample sizes to evaluate (default: 5 10 20 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed (default: 42)",
    )
    args = parser.parse_args()

    print(f"\nBootstrap CI Simulation")
    print(f"  Estimand        : {args.estimand}")
    if args.estimand == "pairwise":
        print(f"  Runs per input  : {args.runs}")
        print(f"  Statistic       : {args.statistic}")
    print(f"  Reps per cell   : {args.reps}")
    print(f"  Bootstrap draws : {args.bootstrap_n}")
    print(f"  Alpha / CI level: {args.alpha} / {(1 - args.alpha):.0%}")
    print(f"  Sample sizes    : {args.sizes}")
    print(f"  Seed            : {args.seed}")

    print("\nBuilding scenarios …", end="", flush=True)
    if args.estimand == "mean":
        scenarios = build_scenarios()
    else:
        scenarios = build_pair_scenarios()
    n_by_type = {et: sum(1 for s in scenarios if s.eval_type == et) for et in EVAL_TYPES}
    print(
        f"  {len(scenarios)} total  "
        + "  ".join(f"{et}: {n_by_type[et]}" for et in EVAL_TYPES)
    )

    cells = len(scenarios) * len(args.sizes)
    bootstrap_calls = cells * args.reps * len(METHODS)
    if args.estimand == "mean":
        binary_cells = n_by_type["binary"] * len(args.sizes)
        wilson_calls = binary_cells * args.reps
        print(
            f"\nRunning {cells} cells × {args.reps} reps × {len(METHODS)} bootstrap methods "
            f"= {bootstrap_calls:,} CI calls, plus {wilson_calls:,} Wilson calls (binary only) …"
        )
    else:
        print(f"\nRunning {cells} cells × {args.reps} reps × {len(METHODS)} methods = {bootstrap_calls:,} CI calls …")

    if args.estimand == "mean":
        results = run_simulation(
            scenarios=scenarios,
            sample_sizes=args.sizes,
            n_reps=args.reps,
            n_bootstrap=args.bootstrap_n,
            alpha=args.alpha,
            seed=args.seed,
        )
        estimand_label = "template mean"
    else:
        results = run_pairwise_simulation(
            scenarios=scenarios,
            sample_sizes=args.sizes,
            n_reps=args.reps,
            n_bootstrap=args.bootstrap_n,
            alpha=args.alpha,
            runs=args.runs,
            statistic=args.statistic,
            seed=args.seed,
        )
        estimand_label = f"paired template difference ({args.statistic}, runs={args.runs})"

    print_report(
        results,
        sample_sizes=args.sizes,
        alpha=args.alpha,
        n_reps=args.reps,
        estimand_label=estimand_label,
    )


if __name__ == "__main__":
    main()
