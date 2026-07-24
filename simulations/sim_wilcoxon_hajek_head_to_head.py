"""Head-to-head prototype benchmark for Wilcoxon PPI variants.

Compares:
  - method="current" (median + mean-rectifier)
  - method="hajek_experimental" (linearized signed-rank score mean)

Across two simple scenarios:
  1) Type I stress: no true paired shift, but differential LLM bias.
  2) Power: real paired shift with modest judge noise.

Usage:
  .venv/bin/python -m simulations.sim_wilcoxon_hajek_head_to_head --reps 200 --n-boot 300
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import io

import numpy as np

from evalstats.tests import wilcoxon


@dataclass
class Scenario:
    name: str
    mu_a: float
    mu_b: float
    sigma: float
    bias_a: float
    bias_b: float
    llm_noise: float


def _paired_data(
    rng: np.random.Generator,
    *,
    n: int,
    n_lab: int,
    mu_a: float,
    mu_b: float,
    sigma: float,
    bias_a: float,
    bias_b: float,
    llm_noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired truth + LLM with shared subject effects."""
    subject_fx = rng.normal(0.0, 0.5, n)
    truth_a = mu_a + subject_fx + rng.normal(0.0, sigma, n)
    truth_b = mu_b + subject_fx + rng.normal(0.0, sigma, n)

    a = truth_a + bias_a + rng.normal(0.0, llm_noise, n)
    b = truth_b + bias_b + rng.normal(0.0, llm_noise, n)

    idx = rng.choice(n, n_lab, replace=False)
    a_lab = np.full(n, np.nan)
    b_lab = np.full(n, np.nan)
    a_lab[idx] = truth_a[idx]
    b_lab[idx] = truth_b[idx]
    return a, b, a_lab, b_lab


def _run_one(
    scenario: Scenario,
    *,
    n: int,
    n_lab: int,
    n_boot: int,
    alpha: float,
    reps: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    current_reject = 0
    hajek_reject = 0
    uncorrected_reject = 0

    current_abs_est = []
    hajek_abs_est = []

    for rep in range(reps):
        a, b, a_lab, b_lab = _paired_data(
            rng,
            n=n,
            n_lab=n_lab,
            mu_a=scenario.mu_a,
            mu_b=scenario.mu_b,
            sigma=scenario.sigma,
            bias_a=scenario.bias_a,
            bias_b=scenario.bias_b,
            llm_noise=scenario.llm_noise,
        )

        run_seed = seed * 100_000 + rep

        with contextlib.redirect_stdout(io.StringIO()):
            r_current = wilcoxon(
                a,
                b,
                x_lab=a_lab,
                y_lab=b_lab,
                method="current",
                n_boot=n_boot,
                rng=run_seed,
                print_result=False,
            )
            r_hajek = wilcoxon(
                a,
                b,
                x_lab=a_lab,
                y_lab=b_lab,
                method="hajek_experimental",
                n_boot=n_boot,
                rng=run_seed,
                print_result=False,
            )

        uncorrected_reject += int(r_current.p_value < alpha)
        current_reject += int((r_current.corrected_p_value or 1.0) < alpha)
        hajek_reject += int((r_hajek.corrected_p_value or 1.0) < alpha)

        current_abs_est.append(abs(float(r_current.corrected_estimate)))
        hajek_abs_est.append(abs(float(r_hajek.corrected_estimate)))

    return {
        "uncorrected_reject_rate": uncorrected_reject / reps,
        "current_reject_rate": current_reject / reps,
        "hajek_reject_rate": hajek_reject / reps,
        "current_mean_abs_estimate": float(np.mean(current_abs_est)),
        "hajek_mean_abs_estimate": float(np.mean(hajek_abs_est)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head Wilcoxon PPI benchmark")
    parser.add_argument("--reps", type=int, default=200, help="Monte Carlo replicates per scenario")
    parser.add_argument("--n", type=int, default=260, help="Paired sample size")
    parser.add_argument("--n-lab", type=int, default=70, help="Number of labeled pairs")
    parser.add_argument("--n-boot", type=int, default=300, help="Bootstrap draws per test call")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--seed", type=int, default=1234, help="Master RNG seed")
    args = parser.parse_args()

    scenarios = [
        Scenario(
            name="type1_bias_stress",
            mu_a=3.0,
            mu_b=3.0,
            sigma=1.0,
            bias_a=2.0,
            bias_b=0.0,
            llm_noise=0.15,
        ),
        Scenario(
            name="power_true_shift",
            mu_a=4.0,
            mu_b=3.0,
            sigma=1.0,
            bias_a=0.5,
            bias_b=0.0,
            llm_noise=0.15,
        ),
    ]

    print("=" * 88)
    print("Wilcoxon PPI head-to-head: current vs hajek_experimental")
    print(
        f"reps={args.reps}, n={args.n}, n_lab={args.n_lab}, "
        f"n_boot={args.n_boot}, alpha={args.alpha}, seed={args.seed}"
    )
    print("=" * 88)

    for i, sc in enumerate(scenarios):
        res = _run_one(
            sc,
            n=args.n,
            n_lab=args.n_lab,
            n_boot=args.n_boot,
            alpha=args.alpha,
            reps=args.reps,
            seed=args.seed + i,
        )

        print(f"\n[{sc.name}]")
        print(f"  uncorrected reject rate:      {res['uncorrected_reject_rate']:.3f}")
        print(f"  current corrected reject:     {res['current_reject_rate']:.3f}")
        print(f"  hajek corrected reject:       {res['hajek_reject_rate']:.3f}")
        print(f"  current mean |estimate|:      {res['current_mean_abs_estimate']:.3f}")
        print(f"  hajek mean |estimate|:        {res['hajek_mean_abs_estimate']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
