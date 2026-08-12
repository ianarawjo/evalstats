"""Standalone measurement mirroring tests/test_compound_ppi_fwer.py's
TestCompoundCalibration.test_compound_correction_power_cost_is_bounded,
run before/after flipping evalstats.tests._ppi_paired_tango's power_tune
default, to quantify whether PPI++ power-tuning improves the compound
PPI+FWER path's detection power. Reuses the actual test module's data
generator directly rather than reimplementing it.

Usage: python -m simulations.investigate_compound_ppi_fwer_power
"""
from __future__ import annotations

import sys
import warnings

sys.path.insert(0, "tests")

import evalstats as es
from evalstats.alignment import judge_alignment
from test_compound_ppi_fwer import _make_multiarm_binary, _rng

N_REPS_POWER = 150
N_REPS_NULL = 300


def measure_power(seed_base: int = 2000) -> float:
    alpha = 0.05
    n_detected = 0
    for i in range(N_REPS_POWER):
        evaldata = _make_multiarm_binary(
            n_entities=4, n_items=150, n_labeled=60,
            p_base=0.55, effect_size=0.30, agreement_rate=0.85, seed=2000 + i,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=True, correction="shaffer",
                rng=_rng(2000 + i),
            )
        bundle = result._primary_bundle()
        pw = bundle.pairwise
        p = pw.get("M0", "M3").p_value
        if p is not None and p < alpha:
            n_detected += 1

    return n_detected / N_REPS_POWER


def measure_null_fwer(seed_base: int = 1000, n_reps: int = N_REPS_NULL) -> float:
    alpha = 0.05
    n_reject = 0
    for i in range(n_reps):
        evaldata = _make_multiarm_binary(
            n_entities=4, n_items=150, n_labeled=60,
            p_base=0.55, effect_size=0.0, agreement_rate=0.85, seed=1000 + i,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(
                evaldata, factors="model", metric="llm_score",
                alignment={"llm_score": ar}, n_mc=30,
                simultaneous_ci=True, correction="shaffer",
                rng=_rng(1000 + i),
            )
        bundle = result._primary_bundle()
        if any(pr.p_value < alpha for pr in bundle.pairwise.results.values()):
            n_reject += 1
    return n_reject / n_reps


if __name__ == "__main__":
    import time
    t0 = time.time()
    power_hat = measure_power()
    print(f"power_hat (compound PPI+FWER, true effect=0.30): {power_hat:.4f} over {N_REPS_POWER} reps")
    fwer_hat = measure_null_fwer()
    print(f"fwer_hat  (compound PPI+FWER, null):              {fwer_hat:.4f} over {N_REPS_NULL} reps")
    print(f"Done in {time.time() - t0:.1f}s")
