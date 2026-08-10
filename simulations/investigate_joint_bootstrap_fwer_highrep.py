"""High-rep, FWER-only recheck of evalstats.api._ppi_bootstrap_t_joint_stats's
power_tune=True path, at the conditions from investigate_joint_bootstrap_
power_tune_grid.py that showed the most FWER movement, to distinguish real
inflation from Monte Carlo noise at the grid's original 80-rep precision.

Usage: python -m simulations.investigate_joint_bootstrap_fwer_highrep
"""
from __future__ import annotations

import sys
import time
import warnings

sys.path.insert(0, "tests")

import evalstats as es
from evalstats.alignment import validate_alignment
from test_compound_ppi_fwer import _make_multiarm_binary, _rng

from simulations.investigate_joint_bootstrap_power_tune import _PowerTuneOverride
from simulations.investigate_joint_bootstrap_power_tune_grid import _n_labeled

ALPHA = 0.05
N_REPS = 600

# Conditions to recheck: the two hottest RW FWER(true) readings (7.5% each)
# plus the one condition whose FWER(true) fell furthest from FWER(false)
# in the DOWNWARD direction (the k=5/N=50 outlier), all from the first grid.
CONDITIONS = [
    (3, 50, 0.40),
    (4, 150, 0.40),
    (5, 50, 0.40),
]


def measure_null_fwer_only(n_entities: int, n_items: int, label_frac: float, power_tune: bool, seed_base: int, n_reps: int) -> float:
    n_labeled = _n_labeled(n_items, label_frac)
    n_reject = 0
    with _PowerTuneOverride(power_tune):
        for i in range(n_reps):
            evaldata = _make_multiarm_binary(
                n_entities=n_entities, n_items=n_items, n_labeled=n_labeled,
                p_base=0.55, effect_size=0.0, agreement_rate=0.85, seed=seed_base + i,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = validate_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
                result = es.compare(
                    evaldata, factors="model", metric="llm_score",
                    alignment={"llm_score": ar}, n_mc=30,
                    simultaneous_ci=False, correction="romano_wolf",
                    rng=_rng(seed_base + i),
                )
            bundle = result._primary_bundle()
            if any(pr.p_value < ALPHA for pr in bundle.pairwise.results.values()):
                n_reject += 1
    return n_reject / n_reps


if __name__ == "__main__":
    import math
    t0 = time.time()
    se = math.sqrt(0.05 * 0.95 / N_REPS)
    print(f"N_REPS={N_REPS}, binomial SE at nominal 5% = {se:.4f}\n")
    print(f"{'k':>3} {'N':>5} {'lab%':>5}  {'FWER(F)':>8} {'FWER(T)':>8}  {'F z':>6} {'T z':>6}")
    print("-" * 55)
    seed = 9000
    for (k, n, lf) in CONDITIONS:
        fwer_f = measure_null_fwer_only(k, n, lf, False, seed, N_REPS)
        seed += 200000
        fwer_t = measure_null_fwer_only(k, n, lf, True, seed, N_REPS)
        seed += 200000
        z_f = (fwer_f - 0.05) / se
        z_t = (fwer_t - 0.05) / se
        print(f"{k:>3} {n:>5} {lf:>5.2f}  {fwer_f:>8.4f} {fwer_t:>8.4f}  {z_f:>6.2f} {z_t:>6.2f}")

    print(f"\nDone in {time.time() - t0:.1f}s")
