"""Extended validation grid for evalstats.api._ppi_bootstrap_t_joint_stats's
power_tune parameter, covering the realistic N >> N_lab regime the first
grid (investigate_joint_bootstrap_power_tune_grid.py) didn't: sparser
labeling (10% frac) and larger item pools (N up to 400). Fixed k=4 arms to
isolate the N/label_frac effect rather than crossing every factor.

Usage: python -m simulations.investigate_joint_bootstrap_power_tune_grid2
"""
from __future__ import annotations

import sys
import time
import warnings

sys.path.insert(0, "tests")

import evalstats as es
from evalstats.alignment import judge_alignment
from test_compound_ppi_fwer import _make_multiarm_binary, _rng

from simulations.investigate_joint_bootstrap_power_tune import _PowerTuneOverride
from simulations.investigate_joint_bootstrap_power_tune_grid import _n_labeled

ALPHA = 0.05
EFFECT_SIZE = 0.30
N_REPS_NULL = 150
N_REPS_POWER = 100

K_FIXED = 4
CONDITIONS = [
    (K_FIXED, 100, 0.10),
    (K_FIXED, 200, 0.10),
    (K_FIXED, 400, 0.10),
    (K_FIXED, 400, 0.20),
    (K_FIXED, 400, 0.40),
    (3, 400, 0.10),
    (5, 400, 0.10),
]


def measure_condition(n_entities: int, n_items: int, label_frac: float, power_tune: bool, seed_base: int):
    n_labeled = _n_labeled(n_items, label_frac)
    n_reject = 0
    n_detect = 0
    with _PowerTuneOverride(power_tune):
        for i in range(N_REPS_NULL):
            evaldata = _make_multiarm_binary(
                n_entities=n_entities, n_items=n_items, n_labeled=n_labeled,
                p_base=0.55, effect_size=0.0, agreement_rate=0.85, seed=seed_base + i,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
                result = es.compare(
                    evaldata, factors="model", metric="llm_score",
                    alignment={"llm_score": ar}, n_mc=30,
                    simultaneous_ci=False, correction="romano_wolf",
                    rng=_rng(seed_base + i),
                )
            bundle = result._primary_bundle()
            if any(pr.p_value < ALPHA for pr in bundle.pairwise.results.values()):
                n_reject += 1

        last_pair = (f"M0", f"M{n_entities - 1}")
        for i in range(N_REPS_POWER):
            evaldata = _make_multiarm_binary(
                n_entities=n_entities, n_items=n_items, n_labeled=n_labeled,
                p_base=0.55, effect_size=EFFECT_SIZE, agreement_rate=0.85, seed=seed_base + 30000 + i,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
                result = es.compare(
                    evaldata, factors="model", metric="llm_score",
                    alignment={"llm_score": ar}, n_mc=30,
                    simultaneous_ci=False, correction="romano_wolf",
                    rng=_rng(seed_base + 30000 + i),
                )
            bundle = result._primary_bundle()
            p = bundle.pairwise.get(*last_pair).p_value
            if p is not None and p < ALPHA:
                n_detect += 1
    return n_reject / N_REPS_NULL, n_detect / N_REPS_POWER


if __name__ == "__main__":
    t0 = time.time()
    print(f"{'k':>3} {'N':>5} {'lab%':>5} {'n_lab':>6}  {'FWER(F)':>8} {'FWER(T)':>8}  {'pow(F)':>7} {'pow(T)':>7}")
    print("-" * 65)
    rows = []
    seed = 11000
    for (k, n, lf) in CONDITIONS:
        fwer_f, pow_f = measure_condition(k, n, lf, False, seed)
        seed += 200000
        fwer_t, pow_t = measure_condition(k, n, lf, True, seed)
        seed += 200000
        n_lab = _n_labeled(n, lf)
        rows.append((k, n, lf, n_lab, fwer_f, fwer_t, pow_f, pow_t))
        print(f"{k:>3} {n:>5} {lf:>5.2f} {n_lab:>6}  {fwer_f:>8.4f} {fwer_t:>8.4f}  {pow_f:>7.4f} {pow_t:>7.4f}")

    print("\n=== Summary ===")
    worst_fwer_t = max(r[5] for r in rows)
    n_power_regressions = sum(1 for r in rows if r[7] < r[6] - 0.05)
    print(f"worst power_tune=True FWER = {worst_fwer_t:.4f} (nominal 0.05); "
          f"power regressions (>0.05 drop): {n_power_regressions}/{len(rows)}")

    print(f"\nDone in {time.time() - t0:.1f}s")
