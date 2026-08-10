"""Fuller validation grid for evalstats.api._ppi_bootstrap_t_joint_stats's
power_tune parameter, following up on investigate_joint_bootstrap_power_tune.py's
single-condition head-to-head. Mirrors the original PPI+FWER PR's own
validation shape (k=3-5 arms, N=50-200, label fractions 20-40%, binary
data) with a curated (not full-factorial) set of conditions and reduced
rep counts to keep runtime bounded, matching this codebase's "screening
tier" convention (see simulations/harness/README.md).

Usage: python -m simulations.investigate_joint_bootstrap_power_tune_grid
"""
from __future__ import annotations

import sys
import time
import warnings

sys.path.insert(0, "tests")

import evalstats as es
from evalstats.alignment import validate_alignment
from test_compound_ppi_fwer import _make_multiarm_binary, _rng

from simulations.investigate_joint_bootstrap_power_tune import _PowerTuneOverride, _compare_with_max_t

ALPHA = 0.05

# (k_entities, n_items, label_frac) -- curated corners of the design space,
# not a full k x N x label_frac cross (would be 3x4x2=24 conditions).
RW_CONDITIONS = [
    (3, 50, 0.40),
    (3, 200, 0.20),
    (4, 100, 0.30),
    (4, 150, 0.40),   # matches investigate_joint_bootstrap_power_tune.py's single condition
    (5, 50, 0.40),
    (5, 200, 0.20),
]
MAX_T_CONDITIONS = [
    (3, 50, 0.40),
    (5, 200, 0.20),
]

N_REPS_NULL_RW = 80
N_REPS_POWER_RW = 60
N_REPS_NULL_MT = 50
N_REPS_POWER_MT = 40
EFFECT_SIZE = 0.30


def _n_labeled(n_items: int, label_frac: float) -> int:
    return max(15, round(n_items * label_frac))


def measure_romano_wolf_condition(n_entities: int, n_items: int, label_frac: float, power_tune: bool, seed_base: int):
    n_labeled = _n_labeled(n_items, label_frac)
    n_reject = 0
    n_detect = 0
    with _PowerTuneOverride(power_tune):
        for i in range(N_REPS_NULL_RW):
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

        last_pair = None
        for i in range(N_REPS_POWER_RW):
            evaldata = _make_multiarm_binary(
                n_entities=n_entities, n_items=n_items, n_labeled=n_labeled,
                p_base=0.55, effect_size=EFFECT_SIZE, agreement_rate=0.85, seed=seed_base + 20000 + i,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = validate_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
                result = es.compare(
                    evaldata, factors="model", metric="llm_score",
                    alignment={"llm_score": ar}, n_mc=30,
                    simultaneous_ci=False, correction="romano_wolf",
                    rng=_rng(seed_base + 20000 + i),
                )
            bundle = result._primary_bundle()
            last_pair = last_pair or (f"M0", f"M{n_entities - 1}")
            p = bundle.pairwise.get(*last_pair).p_value
            if p is not None and p < ALPHA:
                n_detect += 1
    return n_reject / N_REPS_NULL_RW, n_detect / N_REPS_POWER_RW


def measure_max_t_condition(n_entities: int, n_items: int, label_frac: float, power_tune: bool, seed_base: int):
    n_labeled = _n_labeled(n_items, label_frac)
    n_reject = 0
    n_detect = 0
    with _PowerTuneOverride(power_tune):
        for i in range(N_REPS_NULL_MT):
            evaldata = _make_multiarm_binary(
                n_entities=n_entities, n_items=n_items, n_labeled=n_labeled,
                p_base=0.55, effect_size=0.0, agreement_rate=0.85, seed=seed_base + i,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = validate_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
                result = _compare_with_max_t(evaldata, ar, seed_base + i)
            bundle = result._primary_bundle()
            assert bundle.pairwise.simultaneous_ci_method == "max_t"
            if any(pr.p_value < ALPHA for pr in bundle.pairwise.results.values()):
                n_reject += 1

        last_pair = None
        for i in range(N_REPS_POWER_MT):
            evaldata = _make_multiarm_binary(
                n_entities=n_entities, n_items=n_items, n_labeled=n_labeled,
                p_base=0.55, effect_size=EFFECT_SIZE, agreement_rate=0.85, seed=seed_base + 20000 + i,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = validate_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
                result = _compare_with_max_t(evaldata, ar, seed_base + 20000 + i)
            bundle = result._primary_bundle()
            last_pair = last_pair or (f"M0", f"M{n_entities - 1}")
            p = bundle.pairwise.get(*last_pair).p_value
            if p is not None and p < ALPHA:
                n_detect += 1
    return n_reject / N_REPS_NULL_MT, n_detect / N_REPS_POWER_MT


if __name__ == "__main__":
    t0 = time.time()

    print("=== Romano-Wolf grid ===")
    print(f"{'k':>3} {'N':>5} {'lab%':>5}  {'FWER(F)':>8} {'FWER(T)':>8}  {'pow(F)':>7} {'pow(T)':>7}")
    print("-" * 60)
    rw_rows = []
    seed = 6000
    for (k, n, lf) in RW_CONDITIONS:
        fwer_f, pow_f = measure_romano_wolf_condition(k, n, lf, False, seed)
        seed += 100000
        fwer_t, pow_t = measure_romano_wolf_condition(k, n, lf, True, seed)
        seed += 100000
        rw_rows.append((k, n, lf, fwer_f, fwer_t, pow_f, pow_t))
        print(f"{k:>3} {n:>5} {lf:>5.2f}  {fwer_f:>8.4f} {fwer_t:>8.4f}  {pow_f:>7.4f} {pow_t:>7.4f}")

    print("\n=== max-T grid ===")
    print(f"{'k':>3} {'N':>5} {'lab%':>5}  {'FWER(F)':>8} {'FWER(T)':>8}  {'pow(F)':>7} {'pow(T)':>7}")
    print("-" * 60)
    mt_rows = []
    for (k, n, lf) in MAX_T_CONDITIONS:
        fwer_f, pow_f = measure_max_t_condition(k, n, lf, False, seed)
        seed += 100000
        fwer_t, pow_t = measure_max_t_condition(k, n, lf, True, seed)
        seed += 100000
        mt_rows.append((k, n, lf, fwer_f, fwer_t, pow_f, pow_t))
        print(f"{k:>3} {n:>5} {lf:>5.2f}  {fwer_f:>8.4f} {fwer_t:>8.4f}  {pow_f:>7.4f} {pow_t:>7.4f}")

    print("\n=== Summary ===")
    worst_fwer_t_rw = max(r[4] for r in rw_rows)
    worst_fwer_t_mt = max(r[4] for r in mt_rows)
    n_power_regressions_rw = sum(1 for r in rw_rows if r[6] < r[5] - 0.05)
    n_power_regressions_mt = sum(1 for r in mt_rows if r[6] < r[5] - 0.05)
    print(f"Romano-Wolf: worst power_tune=True FWER = {worst_fwer_t_rw:.4f} (nominal 0.05); "
          f"power regressions (>0.05 drop): {n_power_regressions_rw}/{len(rw_rows)}")
    print(f"max-T:       worst power_tune=True FWER = {worst_fwer_t_mt:.4f} (nominal 0.05); "
          f"power regressions (>0.05 drop): {n_power_regressions_mt}/{len(mt_rows)}")

    print(f"\nDone in {time.time() - t0:.1f}s")
