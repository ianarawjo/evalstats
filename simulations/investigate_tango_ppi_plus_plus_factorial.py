"""Scoped-down binary factorial calibration check for evalstats.tests.
_ppi_paired_tango's new power_tune=True path, following up on
investigate_tango_ppi_plus_plus.py's one-off head-to-head (which found
large power gains and no Type-I inflation across an n_lab x noise grid).

Crosses bias_magnitude x label_mechanism x n -- specifically targeting MNAR
labeling, the known failure mode for other PPI rectifiers in this codebase
(kruskal_mnar_experimental, friedman all show real
MCAR-cost or MNAR-residual tradeoffs). Deliberately small (18 cells) and
closed-form (no bootstrap), so this runs in well under a minute even at a
few thousand reps/cell -- NOT the full --factorial-check-binary sweep.

Usage: python -m simulations.investigate_tango_ppi_plus_plus_factorial
"""
from __future__ import annotations

import time

import numpy as np

from evalstats.tests import _ppi_paired_tango
from simulations.harness.scenarios import JudgeBiasSource
from simulations.harness.scenarios.synthetic import (
    PPI_BINARY_BIAS_MAGNITUDES,
    PPI_BINARY_NOISE_BASELINE,
    PPI_FACTORIAL_LABEL_MECHANISMS,
    generate_judge_bias_cell,
)

ALPHA = 0.05
N_VALUES = (60, 200)
BIAS_LABELS = ("none", "moderate", "severe")
LABEL_MECHANISMS = ("mcar", "mnar_mild", "mnar_strong")
N_REPS = 2000
SEED = 20260811


def run_cell(n: int, bias_label: str, lm_label: str, n_reps: int, seed: int):
    sc = JudgeBiasSource(
        name=f"tango_pp_factorial.n={n}.bm={bias_label}.lm={lm_label}",
        tag="tango_ppi_plus_plus_factorial",
        eval_type="binary",
        icc=0.20,
        n=n,
        label_frac=0.20,
        llm_noise=PPI_BINARY_NOISE_BASELINE,
        bias_type="differential" if bias_label != "none" else "none",
        bias_delta=PPI_BINARY_BIAS_MAGNITUDES[bias_label],
        effect_size=0.0,
        **PPI_FACTORIAL_LABEL_MECHANISMS[lm_label],
    )
    rng = np.random.default_rng(seed)

    rej_old = rej_new = 0
    width_old = np.empty(n_reps)
    width_new = np.empty(n_reps)
    lam_new = np.empty(n_reps)

    for i in range(n_reps):
        cell = generate_judge_bias_cell(sc, rng)
        r_old = _ppi_paired_tango(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, ALPHA, power_tune=False)
        r_new = _ppi_paired_tango(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, ALPHA, power_tune=True)
        rej_old += int(r_old.p_value < ALPHA)
        rej_new += int(r_new.p_value < ALPHA)
        width_old[i] = r_old.ci_high - r_old.ci_low
        width_new[i] = r_new.ci_high - r_new.ci_low
        lam_new[i] = r_new.lam if r_new.lam is not None else 1.0

    return {
        "n": n, "bias_label": bias_label, "lm_label": lm_label,
        "rate_old": rej_old / n_reps, "rate_new": rej_new / n_reps,
        "width_old": float(width_old.mean()), "width_new": float(width_new.mean()),
        "lam_mean": float(lam_new.mean()),
    }


def main():
    t0 = time.time()
    print(f"{'n':>4} {'bias':>9} {'label_mech':>11}  {'TypeI_old':>10} {'TypeI_new':>10} "
          f"{'width_old':>10} {'width_new':>10} {'mean_lam':>9}")
    print("-" * 90)

    results = []
    seed = SEED
    for n in N_VALUES:
        for bias_label in BIAS_LABELS:
            for lm_label in LABEL_MECHANISMS:
                r = run_cell(n, bias_label, lm_label, N_REPS, seed)
                seed += 1
                results.append(r)
                flag = "  <-- new worse" if r["rate_new"] > r["rate_old"] + 0.02 else ""
                print(f"{r['n']:>4} {r['bias_label']:>9} {r['lm_label']:>11}  "
                      f"{r['rate_old']:>10.4f} {r['rate_new']:>10.4f} "
                      f"{r['width_old']:>10.4f} {r['width_new']:>10.4f} {r['lam_mean']:>9.3f}{flag}")

    print("-" * 90)
    worst_old = max(results, key=lambda r: r["rate_old"])
    worst_new = max(results, key=lambda r: r["rate_new"])
    print(f"Worst Type-I, old: {worst_old['rate_old']:.4f} at "
          f"n={worst_old['n']}, bm={worst_old['bias_label']}, lm={worst_old['lm_label']}")
    print(f"Worst Type-I, new: {worst_new['rate_new']:.4f} at "
          f"n={worst_new['n']}, bm={worst_new['bias_label']}, lm={worst_new['lm_label']}")
    mnar_rows = [r for r in results if r["lm_label"] != "mcar"]
    print(f"MNAR-only mean Type-I: old={np.mean([r['rate_old'] for r in mnar_rows]):.4f}, "
          f"new={np.mean([r['rate_new'] for r in mnar_rows]):.4f}")
    mcar_rows = [r for r in results if r["lm_label"] == "mcar"]
    print(f"MCAR-only mean Type-I: old={np.mean([r['rate_old'] for r in mcar_rows]):.4f}, "
          f"new={np.mean([r['rate_new'] for r in mcar_rows]):.4f}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
