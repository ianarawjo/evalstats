#!/usr/bin/env python3
"""
sim_compare_bayes.py — Bootstrap/score CIs vs Bayesian credible intervals.

Compares frequentist methods (bootstrap variants, Wilson/Newcombe score CIs)
against the Bayesian methods from bayes_evals.py (Bowyer 2025, ICML position
paper: "Don't use the CLT in LLM evals with fewer than a few hundred
datapoints") using Monte Carlo simulations where the true parameter is known.

Focus: binary (0/1 pass-fail) eval data, where the Bayesian conjugate methods
in bayes_evals.py are applicable. Non-binary data is excluded because bayes_evals
targets the binary case.

Single-model methods (estimating p, the true pass rate):
  bootstrap          Percentile bootstrap
  bca                BCa bootstrap
  bayes_bootstrap    Dirichlet-weighted bootstrap
  smooth_bootstrap   KDE-perturbed smooth bootstrap
  wilson             Wilson score interval
  bayes_indep        Beta(1,1) posterior credible interval (Bowyer 2025)

Pairwise methods (estimating p_A − p_B, two models on same questions):
  bootstrap / bca / bayes_bootstrap / smooth_bootstrap  (CI on per-item diffs)
  newcombe_score     Newcombe paired score interval
  bayes_indep_comp   Percentile CI from independent Beta posteriors
  bayes_paired_comp  Percentile CI from bivariate-normal paired model (Bowyer 2025)

Two pairwise scenarios are tested:
  pair-indep   A and B drawn from independent Bernoullis (no question correlation)
  pair-corr    Shared per-item difficulty (realistic: same questions for both models)

Metrics:
  coverage    Fraction of CIs containing the true parameter (nominal: 1−alpha)
  mean_width  Average CI width (narrower is more precise, given adequate coverage)

Usage:
  python simulations/sim_compare_bayes.py
  python simulations/sim_compare_bayes.py --estimand pairwise
  python simulations/sim_compare_bayes.py --reps 300 --sizes 5 10 20 50 100
  python simulations/sim_compare_bayes.py --bayes-n 5000 --bootstrap-n 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.stats as stats

# ── Import promptstats bootstrap utilities ─────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from promptstats.core.resampling import bootstrap_ci_1d

# ── Import bivariate normal CDF from bayes_evals.py ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from bayes_evals import binorm_cdf  # noqa: E402  (local sibling module)


# ---------------------------------------------------------------------------
# Method name constants
# ---------------------------------------------------------------------------

BOOT_METHODS = ["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap"]

SINGLE_METHODS = BOOT_METHODS + ["wilson", "bayes_indep"]
PAIR_METHODS   = BOOT_METHODS + ["newcombe_score", "bayes_indep_comp", "bayes_paired_comp"]


# ---------------------------------------------------------------------------
# Scenario dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    label: str
    kind: str   # "single", "pair-indep", or "pair-corr"
    generate: Callable
    true_param: float  # true mean (single) or true p_A − p_B (pairwise)


# ---------------------------------------------------------------------------
# Scenario builders  (binary data only)
# ---------------------------------------------------------------------------

def build_single_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]:
        p_ = p
        scenarios.append(Scenario(
            label=f"p={p_:.1f}",
            kind="single",
            generate=lambda rng, n, _p=p_: rng.binomial(1, _p, n).astype(float),
            true_param=p_,
        ))
    return scenarios


def _estimate_pair_diff(
    gen_fn: Callable,
    *,
    seed: int = 0,
    n_mc: int = 500_000,
) -> float:
    rng = np.random.default_rng(seed)
    a, b = gen_fn(rng, n_mc)
    return float(np.mean(a) - np.mean(b))


def build_pair_scenarios() -> list[Scenario]:
    """Return pairwise scenarios: independent pairs and correlated pairs."""
    scenarios: list[Scenario] = []

    # ------------------------------------------------------------------
    # Independent pairs — A and B drawn from separate Bernoullis
    # The Bayesian *independent* model is perfectly specified here.
    # ------------------------------------------------------------------
    for label, p_a, p_b in [
        ("indep (0.50 vs 0.58)", 0.50, 0.58),
        ("indep (0.70 vs 0.78)", 0.70, 0.78),
        ("indep (0.20 vs 0.28)", 0.20, 0.28),
        ("indep (0.40 vs 0.60)", 0.40, 0.60),
    ]:
        pa_, pb_ = p_a, p_b
        scenarios.append(Scenario(
            label=label,
            kind="pair-indep",
            generate=lambda rng, n, _pa=pa_, _pb=pb_: (
                rng.binomial(1, _pa, n).astype(float),
                rng.binomial(1, _pb, n).astype(float),
            ),
            true_param=p_a - p_b,
        ))

    # ------------------------------------------------------------------
    # Correlated pairs — shared per-item difficulty (realistic LLM eval)
    # The Bayesian *paired* model is better specified here.
    # ------------------------------------------------------------------
    for label, base_p, delta in [
        ("corr (base=0.50, Δ=0.08)", 0.50, 0.08),
        ("corr (base=0.70, Δ=0.05)", 0.70, 0.05),
        ("corr (base=0.30, Δ=0.05)", 0.30, 0.05),
        ("corr (base=0.50, Δ=0.15)", 0.50, 0.15),
    ]:
        bp_, d_ = base_p, delta

        def _gen_corr(
            rng: np.random.Generator,
            n: int,
            _bp: float = bp_,
            _d: float = d_,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Shared item difficulty with a fixed accuracy uplift for model B."""
            conc = 10.0
            al = _bp * conc
            be = (1.0 - _bp) * conc
            p_a = rng.beta(al, be, size=n)
            p_b = np.clip(p_a + _d, 0.0, 1.0)
            a = rng.binomial(1, p_a).astype(float)
            b = rng.binomial(1, p_b).astype(float)
            return a, b

        true_diff = _estimate_pair_diff(_gen_corr)
        scenarios.append(Scenario(
            label=label,
            kind="pair-corr",
            generate=_gen_corr,
            true_param=true_diff,
        ))

    return scenarios


# ---------------------------------------------------------------------------
# CI helper functions
# ---------------------------------------------------------------------------

def _wilson_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score CI for a Bernoulli proportion."""
    if n <= 0:
        return 0.0, 0.0
    p_hat = successes / n
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _newcombe_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Newcombe score CI for paired binary difference p(A=1) − p(B=1)."""
    n = len(a)
    if n <= 0:
        return 0.0, 0.0
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    m = n10 + n01
    if m == 0:
        return 0.0, 0.0
    lo_theta, hi_theta = _wilson_ci(successes=n10, n=m, alpha=alpha)
    scale = m / n
    return float(scale * (2.0 * lo_theta - 1.0)), float(scale * (2.0 * hi_theta - 1.0))


def _bayes_indep_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Beta(1,1) posterior credible interval for a Bernoulli proportion."""
    n = len(values)
    s = int(values.sum())
    lo, hi = stats.beta(s + 1, n - s + 1).interval(1.0 - alpha)
    return float(lo), float(hi)


def _bayes_indep_comp_ci(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    num_samples: int,
) -> tuple[float, float]:
    """
    Percentile CI for (p_A − p_B) from independent Beta posteriors.
    Draws samples from Beta(s_A+1, n-s_A+1) and Beta(s_B+1, n-s_B+1)
    independently, then takes percentiles of the sampled differences.
    """
    post_a = stats.beta(a.sum() + 1, len(a) - a.sum() + 1)
    post_b = stats.beta(b.sum() + 1, len(b) - b.sum() + 1)
    diff = post_a.rvs(size=num_samples) - post_b.rvs(size=num_samples)
    return (
        float(np.percentile(diff, 100.0 * alpha / 2.0)),
        float(np.percentile(diff, 100.0 * (1.0 - alpha / 2.0))),
    )


def _bayes_paired_comp_ci(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    num_samples: int,
) -> tuple[float, float]:
    """
    Percentile CI for (p_A − p_B) from the bivariate-normal paired model
    (Bowyer 2025, bayes_evals paired_comparisons).

    Uses importance sampling with a bivariate Gaussian latent model to account
    for per-question correlation between models A and B.

    NOTE: Uses the global NumPy RNG (numpy.random.*) to match bayes_evals.py
    internals — reproducibility is via averaging over many reps, not per-call.
    """
    S = float(np.sum(a * b))            # both correct
    T = float(np.sum(a * (1 - b)))      # A correct, B wrong
    U = float(np.sum((1 - a) * b))      # A wrong,  B correct
    V = float(np.sum((1 - a) * (1 - b)))  # both wrong

    # Proposal samples from the prior
    theta_As = np.random.beta(1, 1, size=num_samples)
    theta_Bs = np.random.beta(1, 1, size=num_samples)
    rhos = np.clip(2 * np.random.beta(4, 2, size=num_samples) - 1, -1 + 1e-20, 1 - 1e-20)
    diff = theta_As - theta_Bs

    mu_A = stats.norm.ppf(theta_As)
    mu_B = stats.norm.ppf(theta_Bs)

    th_V = binorm_cdf(0, 0, mu_A, mu_B, 1, 1, rhos)
    th_S = theta_As + theta_Bs + th_V - 1
    th_T = 1 - theta_Bs - th_V
    th_U = 1 - theta_As - th_V

    with np.errstate(divide="ignore", invalid="ignore"):
        log_w = (
            S * np.log(th_S)
            + T * np.log(th_T)
            + U * np.log(th_U)
            + V * np.log(th_V)
        )

    log_w -= np.nanmax(log_w)
    w = np.exp(log_w)
    w[np.isnan(w)] = 0.0
    w_sum = w.sum()

    if w_sum == 0.0:
        d_hat = float(np.mean(a) - np.mean(b))
        return d_hat, d_hat

    w /= w_sum
    diff_post = diff[np.random.choice(num_samples, size=num_samples, replace=True, p=w)]

    return (
        float(np.percentile(diff_post, 100.0 * alpha / 2.0)),
        float(np.percentile(diff_post, 100.0 * (1.0 - alpha / 2.0))),
    )


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    scenario: str
    kind: str
    n: int
    method: str
    n_reps: int
    covered: int
    total_width: float


def run_single_simulation(
    scenarios: list[Scenario],
    sample_sizes: list[int],
    n_reps: int,
    n_bootstrap: int,
    bayes_n: int,
    alpha: float,
    seed: int = 42,
) -> list[SimResult]:
    rng = np.random.default_rng(seed)
    results: list[SimResult] = []

    total = len(scenarios) * len(sample_sizes)
    done = 0

    for sc in scenarios:
        for n in sample_sizes:
            done += 1
            print(
                f"\r  [{done:3d}/{total}] {sc.label:<20s}  n={n:<4d}",
                end="",
                flush=True,
            )

            covered: dict[str, int]   = {m: 0   for m in SINGLE_METHODS}
            total_w: dict[str, float] = {m: 0.0 for m in SINGLE_METHODS}

            for _ in range(n_reps):
                values = sc.generate(rng, n)
                obs = float(np.mean(values))

                # Bootstrap methods
                for method in BOOT_METHODS:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            lo, hi = bootstrap_ci_1d(
                                values, obs,
                                method=method,
                                n_bootstrap=n_bootstrap,
                                alpha=alpha,
                                rng=rng,
                            )
                    except Exception:
                        lo = hi = obs
                    if lo <= sc.true_param <= hi:
                        covered[method] += 1
                    total_w[method] += hi - lo

                # Wilson
                lo, hi = _wilson_ci(int(values.sum()), n, alpha)
                if lo <= sc.true_param <= hi:
                    covered["wilson"] += 1
                total_w["wilson"] += hi - lo

                # Bayesian independent (Beta conjugate)
                try:
                    lo, hi = _bayes_indep_ci(values, alpha)
                except Exception:
                    lo = hi = obs
                if lo <= sc.true_param <= hi:
                    covered["bayes_indep"] += 1
                total_w["bayes_indep"] += hi - lo

            for m in SINGLE_METHODS:
                results.append(SimResult(
                    scenario=sc.label,
                    kind=sc.kind,
                    n=n,
                    method=m,
                    n_reps=n_reps,
                    covered=covered[m],
                    total_width=total_w[m],
                ))

    print()
    return results


def run_pair_simulation(
    scenarios: list[Scenario],
    sample_sizes: list[int],
    n_reps: int,
    n_bootstrap: int,
    bayes_n: int,
    alpha: float,
    seed: int = 42,
) -> list[SimResult]:
    rng = np.random.default_rng(seed)
    results: list[SimResult] = []

    total = len(scenarios) * len(sample_sizes)
    done = 0

    for sc in scenarios:
        for n in sample_sizes:
            done += 1
            print(
                f"\r  [{done:3d}/{total}] {sc.label:<36s}  n={n:<4d}",
                end="",
                flush=True,
            )

            covered: dict[str, int]   = {m: 0   for m in PAIR_METHODS}
            total_w: dict[str, float] = {m: 0.0 for m in PAIR_METHODS}

            for _ in range(n_reps):
                a, b = sc.generate(rng, n)
                diffs = a - b                        # per-item {-1, 0, +1}
                obs = float(np.mean(diffs))          # unbiased estimate of p_A − p_B

                # Bootstrap methods on per-item differences
                for method in BOOT_METHODS:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            lo, hi = bootstrap_ci_1d(
                                diffs, obs,
                                method=method,
                                n_bootstrap=n_bootstrap,
                                alpha=alpha,
                                rng=rng,
                            )
                    except Exception:
                        lo = hi = obs
                    if lo <= sc.true_param <= hi:
                        covered[method] += 1
                    total_w[method] += hi - lo

                # Newcombe paired score
                try:
                    lo, hi = _newcombe_ci(a, b, alpha)
                except Exception:
                    lo = hi = obs
                if lo <= sc.true_param <= hi:
                    covered["newcombe_score"] += 1
                total_w["newcombe_score"] += hi - lo

                # Bayesian independent comparison (ignores item correlation)
                try:
                    lo, hi = _bayes_indep_comp_ci(a, b, alpha, bayes_n)
                except Exception:
                    lo = hi = obs
                if lo <= sc.true_param <= hi:
                    covered["bayes_indep_comp"] += 1
                total_w["bayes_indep_comp"] += hi - lo

                # Bayesian paired comparison (models item correlation)
                try:
                    lo, hi = _bayes_paired_comp_ci(a, b, alpha, bayes_n)
                except Exception:
                    lo = hi = obs
                if lo <= sc.true_param <= hi:
                    covered["bayes_paired_comp"] += 1
                total_w["bayes_paired_comp"] += hi - lo

            for m in PAIR_METHODS:
                results.append(SimResult(
                    scenario=sc.label,
                    kind=sc.kind,
                    n=n,
                    method=m,
                    n_reps=n_reps,
                    covered=covered[m],
                    total_width=total_w[m],
                ))

    print()
    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _cov_marker(cov: float, target: float, tol: float = 0.04) -> str:
    if cov < target - tol:
        return "▼"
    if cov > target + tol:
        return "▲"
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
    method_order: list[str],
) -> None:
    target = 1.0 - alpha
    n_labels = [f"n={n}" for n in sample_sizes]
    present = {r.method for r in results}
    method_labels = [m for m in method_order if m in present]

    # Aggregate: (kind, method, n) → [(cov, width), ...]
    agg: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    per_sc: dict[tuple, tuple[float, float]] = {}

    for r in results:
        cov = r.covered / r.n_reps
        width = r.total_width / r.n_reps
        agg[(r.kind, r.method, r.n)].append((cov, width))
        per_sc[(r.kind, r.scenario, r.method, r.n)] = (cov, width)

    # Ordered unique kinds preserving insertion order
    kinds: list[str] = list(dict.fromkeys(r.kind for r in results))

    sep = "=" * 72
    print(f"\n{sep}")
    print("  BOOTSTRAP vs BAYESIAN CI COMPARISON  —  SIMULATION RESULTS")
    print(f"  Estimand : {estimand_label}")
    print(f"  Nominal coverage: {target:.0%}   |   reps/scenario: {n_reps}")
    print(f"  ▼ = under-covered (<{target - 0.04:.0%})   ▲ = over-conservative (>{target + 0.04:.0%})")
    print(sep)

    # ----------------------------------------------------------------
    # Per-kind tables
    # ----------------------------------------------------------------
    for kind in kinds:
        kind_label = {
            "single":    "SINGLE-MODEL (estimating p)",
            "pair-indep": "PAIRWISE — INDEPENDENT (no question correlation)",
            "pair-corr":  "PAIRWISE — CORRELATED (shared per-item difficulty)",
        }.get(kind, kind.upper())

        print(f"\n{'─'*72}")
        print(f"  {kind_label}")
        print(f"{'─'*72}")

        def _mean_cov(m: str, n: int) -> float:
            vals = agg.get((kind, m, n), [])
            return float(np.mean([v[0] for v in vals])) if vals else float("nan")

        def _mean_wid(m: str, n: int) -> float:
            vals = agg.get((kind, m, n), [])
            return float(np.mean([v[1] for v in vals])) if vals else float("nan")

        cov_cells: dict[tuple[str, str], str] = {}
        wid_cells: dict[tuple[str, str], str] = {}

        for m in method_labels:
            for i, n in enumerate(sample_sizes):
                cov = _mean_cov(m, n)
                if np.isnan(cov):
                    cov_cells[(m, n_labels[i])] = "─"
                else:
                    marker = _cov_marker(cov, target)
                    cov_cells[(m, n_labels[i])] = f"{cov:.3f}{marker}"

                w = _mean_wid(m, n)
                wid_cells[(m, n_labels[i])] = "─" if np.isnan(w) else f"{w:.4f}"

        _print_grid(
            f"Coverage (target {target:.2f})",
            row_labels=method_labels,
            col_labels=n_labels,
            cells=cov_cells,
        )
        _print_grid(
            "Mean CI Width (narrower = more precise)",
            row_labels=method_labels,
            col_labels=n_labels,
            cells=wid_cells,
        )

    # ----------------------------------------------------------------
    # Coverage deviation summary per kind
    # ----------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("  COVERAGE DEVIATION FROM NOMINAL  (mean_coverage − target)")
    print(f"  Computed across all scenarios within each group.")
    print(f"{'─'*72}")

    for kind in kinds:
        kind_short = {
            "single":    "single",
            "pair-indep": "pair-indep",
            "pair-corr":  "pair-corr",
        }.get(kind, kind)
        print(f"\n  {kind_short}")
        hdr = f"    {'Method':<20}" + "".join(f"  {nl:>9}" for nl in n_labels)
        print(hdr)
        print(f"    {'─'*20}" + "─" * (11 * len(n_labels)))
        for m in method_labels:
            row = f"    {m:<20}"
            for n in sample_sizes:
                vals = agg.get((kind, m, n), [])
                if not vals:
                    row += "  {:>9}".format("─")
                else:
                    dev = float(np.mean([v[0] for v in vals])) - target
                    row += f"  {dev:>+9.3f}"
            print(row)

    # ----------------------------------------------------------------
    # Overall summary across all kinds, scenarios, n
    # ----------------------------------------------------------------
    all_cov: dict[str, list[float]] = defaultdict(list)
    all_wid: dict[str, list[float]] = defaultdict(list)
    for (k, m, n), vals in agg.items():
        all_cov[m].extend(v[0] for v in vals)
        all_wid[m].extend(v[1] for v in vals)

    print(f"\n{'─'*72}")
    print("  OVERALL SUMMARY  (averaged across all groups, scenarios, n)")
    print(f"{'─'*72}")
    print(f"\n  {'Method':<20}  {'Mean Coverage':>14}  {'Mean Width':>11}  {'Cov−Target':>12}")
    print(f"  {'─'*20}  {'─'*14}  {'─'*11}  {'─'*12}")
    for m in method_labels:
        if m not in all_cov:
            continue
        mc = float(np.mean(all_cov[m]))
        mw = float(np.mean(all_wid[m]))
        dev = mc - target
        mark = _cov_marker(mc, target)
        print(f"  {m:<20}  {mc:>13.3f}{mark}  {mw:>11.4f}  {dev:>+12.3f}")
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
            "'mean': single-model pass-rate CIs (default). "
            "'pairwise': two-model difference p_A − p_B."
        ),
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
        help="Bootstrap replicates per CI call (default: 500)",
    )
    parser.add_argument(
        "--bayes-n",
        type=int,
        default=2000,
        metavar="N",
        help="Posterior samples per Bayesian CI call (default: 2000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level; CIs are (1−alpha)×100%% (default: 0.05)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100],
        metavar="N",
        help="Sample sizes to evaluate (default: 5 10 20 50 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed for bootstrap (default: 42). "
             "Note: Bayesian paired CI uses numpy global RNG.",
    )
    args = parser.parse_args()

    print(f"\nBootstrap vs Bayesian CI Simulation")
    print(f"  Estimand        : {args.estimand}")
    print(f"  Reps per cell   : {args.reps}")
    print(f"  Bootstrap draws : {args.bootstrap_n}")
    print(f"  Bayes samples   : {args.bayes_n}")
    print(f"  Alpha / CI level: {args.alpha} / {(1 - args.alpha):.0%}")
    print(f"  Sample sizes    : {args.sizes}")
    print(f"  Seed            : {args.seed}")

    if args.estimand == "mean":
        print("\nBuilding single-model scenarios …", end="", flush=True)
        scenarios = build_single_scenarios()
        print(f"  {len(scenarios)} binary scenarios")

        n_cells = len(scenarios) * len(args.sizes)
        n_boot_methods = len(BOOT_METHODS)
        print(
            f"\nRunning {n_cells} cells × {args.reps} reps × "
            f"{n_boot_methods} bootstrap methods + Wilson + Bayesian …"
        )

        results = run_single_simulation(
            scenarios=scenarios,
            sample_sizes=args.sizes,
            n_reps=args.reps,
            n_bootstrap=args.bootstrap_n,
            bayes_n=args.bayes_n,
            alpha=args.alpha,
            seed=args.seed,
        )
        estimand_label = "single-model pass rate p"
        method_order = SINGLE_METHODS

    else:
        print("\nBuilding pairwise scenarios …", end="", flush=True)
        scenarios = build_pair_scenarios()
        n_indep = sum(1 for s in scenarios if s.kind == "pair-indep")
        n_corr  = sum(1 for s in scenarios if s.kind == "pair-corr")
        print(f"  {len(scenarios)} total  ({n_indep} independent, {n_corr} correlated)")

        n_cells = len(scenarios) * len(args.sizes)
        print(
            f"\nRunning {n_cells} cells × {args.reps} reps × "
            f"{len(PAIR_METHODS)} methods "
            f"(note: bayes_paired_comp uses importance sampling, may be slow) …"
        )

        results = run_pair_simulation(
            scenarios=scenarios,
            sample_sizes=args.sizes,
            n_reps=args.reps,
            n_bootstrap=args.bootstrap_n,
            bayes_n=args.bayes_n,
            alpha=args.alpha,
            seed=args.seed,
        )
        estimand_label = "paired difference p_A − p_B"
        method_order = PAIR_METHODS

    print_report(
        results,
        sample_sizes=args.sizes,
        alpha=args.alpha,
        n_reps=args.reps,
        estimand_label=estimand_label,
        method_order=method_order,
    )


if __name__ == "__main__":
    main()
