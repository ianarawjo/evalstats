"""One-off head-to-head: evalstats.tests._ppi_paired_tango's fixed-lambda=1
construction vs. its new power_tune=True (PPI++ closed-form lambda*) path.

Both are fully closed-form (no bootstrap), so this can run at high rep
counts cheaply. Compares Type-I error, power, and CI width across a
label_frac (n_lab) x llm_noise (judge quality) grid on binary judge-bias
data, using scenarios.synthetic's existing binary scenario machinery.

Usage: python -m simulations.investigate_tango_ppi_plus_plus
"""
from __future__ import annotations

import time

import numpy as np

from evalstats.tests import _ppi_paired_tango
from simulations.harness.scenarios import JudgeBiasSource
from simulations.harness.scenarios.synthetic import (
    PPI_BINARY_BIAS_MAGNITUDES,
    generate_judge_bias_cell,
)

ALPHA = 0.05
N = 200
N_LAB_TARGETS = (15, 30, 60, 100)
NOISE_LEVELS = {"good": 0.025, "baseline": 0.10, "poor": 0.40}
N_REPS = 4000
SEED = 20260810


def run_cell(n_lab: int, noise_label: str, noise: float, effect_size: float, n_reps: int, seed: int):
    sc = JudgeBiasSource(
        name=f"tango_pp.n_lab={n_lab}.noise={noise_label}.es={effect_size}",
        tag="tango_ppi_plus_plus",
        eval_type="binary",
        icc=0.20,
        n=N,
        label_frac=n_lab / N,
        llm_noise=noise,
        bias_type="differential",
        bias_delta=PPI_BINARY_BIAS_MAGNITUDES["severe"],
        effect_size=effect_size,
    )
    rng = np.random.default_rng(seed)

    rej_old = rej_new = 0
    width_old = np.empty(n_reps)
    width_new = np.empty(n_reps)
    lam_new = np.empty(n_reps)
    est_old = np.empty(n_reps)
    est_new = np.empty(n_reps)

    for i in range(n_reps):
        cell = generate_judge_bias_cell(sc, rng)
        r_old = _ppi_paired_tango(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, ALPHA, power_tune=False)
        r_new = _ppi_paired_tango(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, ALPHA, power_tune=True)
        rej_old += int(r_old.p_value < ALPHA)
        rej_new += int(r_new.p_value < ALPHA)
        width_old[i] = r_old.ci_high - r_old.ci_low
        width_new[i] = r_new.ci_high - r_new.ci_low
        lam_new[i] = r_new.lam if r_new.lam is not None else 1.0
        est_old[i] = r_old.estimate
        est_new[i] = r_new.estimate

    return {
        "n_lab": n_lab, "noise_label": noise_label, "effect_size": effect_size,
        "rate_old": rej_old / n_reps, "rate_new": rej_new / n_reps,
        "width_old": float(width_old.mean()), "width_new": float(width_new.mean()),
        "width_ratio": float(width_new.mean() / width_old.mean()) if width_old.mean() > 0 else float("nan"),
        "lam_mean": float(lam_new.mean()),
        "est_old_mean": float(est_old.mean()), "est_new_mean": float(est_new.mean()),
    }


def main():
    t0 = time.time()
    print(f"{'n_lab':>6} {'noise':>9} {'es':>5}  {'TypeI/pow_old':>13} {'TypeI/pow_new':>13} "
          f"{'width_old':>10} {'width_new':>10} {'width_ratio':>11} {'mean_lam':>9}")
    print("-" * 100)

    results = []
    seed = SEED
    # Type-I sweep (effect_size=0): every (n_lab, noise) combination.
    for n_lab in N_LAB_TARGETS:
        for noise_label, noise in NOISE_LEVELS.items():
            r = run_cell(n_lab, noise_label, noise, 0.0, N_REPS, seed)
            seed += 1
            results.append(r)
            print(f"{r['n_lab']:>6} {r['noise_label']:>9} {r['effect_size']:>5.2f}  "
                  f"{r['rate_old']:>13.4f} {r['rate_new']:>13.4f} "
                  f"{r['width_old']:>10.4f} {r['width_new']:>10.4f} {r['width_ratio']:>11.4f} "
                  f"{r['lam_mean']:>9.3f}")

    print("-" * 100)
    # Power sweep (real effect present): same grid, moderate effect size.
    for n_lab in N_LAB_TARGETS:
        for noise_label, noise in NOISE_LEVELS.items():
            r = run_cell(n_lab, noise_label, noise, 0.15, N_REPS, seed)
            seed += 1
            results.append(r)
            print(f"{r['n_lab']:>6} {r['noise_label']:>9} {r['effect_size']:>5.2f}  "
                  f"{r['rate_old']:>13.4f} {r['rate_new']:>13.4f} "
                  f"{r['width_old']:>10.4f} {r['width_new']:>10.4f} {r['width_ratio']:>11.4f} "
                  f"{r['lam_mean']:>9.3f}")

    print("-" * 100)
    null_rows = [r for r in results if r["effect_size"] == 0.0]
    power_rows = [r for r in results if r["effect_size"] != 0.0]
    print(f"Type-I: old max={max(r['rate_old'] for r in null_rows):.4f}, "
          f"new max={max(r['rate_new'] for r in null_rows):.4f} (nominal {ALPHA})")
    print(f"Power:  old mean={np.mean([r['rate_old'] for r in power_rows]):.4f}, "
          f"new mean={np.mean([r['rate_new'] for r in power_rows]):.4f}")
    print(f"Mean CI width ratio (new/old), power rows: "
          f"{np.mean([r['width_ratio'] for r in power_rows]):.4f}")
    print(f"Mean CI width ratio (new/old), null rows:  "
          f"{np.mean([r['width_ratio'] for r in null_rows]):.4f}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
