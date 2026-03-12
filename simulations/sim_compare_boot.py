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
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from promptstats.core.resampling import bootstrap_ci_1d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap"]
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
    covered: int      # number of reps where CI contained the true mean
    total_width: float  # sum of CI widths across reps


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

            covered: dict[str, int] = {m: 0 for m in METHODS}
            total_w: dict[str, float] = {m: 0.0 for m in METHODS}

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

    print()  # end progress line
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
) -> None:
    target = 1.0 - alpha
    n_labels = [f"n={n}" for n in sample_sizes]

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
        for m in METHODS:
            for i, n in enumerate(sample_sizes):
                cov = mean_cov(et, m, n)
                marker = _cov_marker(cov, target)
                cov_cells[(m, n_labels[i])] = f"{cov:.3f}{marker}"

        _print_grid(
            f"Coverage (target {target:.2f})",
            row_labels=METHODS,
            col_labels=n_labels,
            cells=cov_cells,
        )

        # Width grid
        wid_cells: dict[tuple[str, str], str] = {}
        for m in METHODS:
            for i, n in enumerate(sample_sizes):
                w = mean_wid(et, m, n)
                wid_cells[(m, n_labels[i])] = f"{w:.4f}"

        _print_grid(
            "Mean CI Width (lower = more precise)",
            row_labels=METHODS,
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
        for m in METHODS:
            row = f"    {m:<20}"
            for n in sample_sizes:
                dev = mean_cov(et, m, n) - target
                row += f"  {dev:>+9.3f}"
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
    for m in METHODS:
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
    print(f"  Reps per cell   : {args.reps}")
    print(f"  Bootstrap draws : {args.bootstrap_n}")
    print(f"  Alpha / CI level: {args.alpha} / {(1 - args.alpha):.0%}")
    print(f"  Sample sizes    : {args.sizes}")
    print(f"  Seed            : {args.seed}")

    print("\nBuilding scenarios …", end="", flush=True)
    scenarios = build_scenarios()
    n_by_type = {et: sum(1 for s in scenarios if s.eval_type == et) for et in EVAL_TYPES}
    print(
        f"  {len(scenarios)} total  "
        + "  ".join(f"{et}: {n_by_type[et]}" for et in EVAL_TYPES)
    )

    cells = len(scenarios) * len(args.sizes)
    calls = cells * args.reps * len(METHODS)
    print(f"\nRunning {cells} cells × {args.reps} reps × {len(METHODS)} methods = {calls:,} CI calls …")

    results = run_simulation(
        scenarios=scenarios,
        sample_sizes=args.sizes,
        n_reps=args.reps,
        n_bootstrap=args.bootstrap_n,
        alpha=args.alpha,
        seed=args.seed,
    )

    print_report(results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps)


if __name__ == "__main__":
    main()
