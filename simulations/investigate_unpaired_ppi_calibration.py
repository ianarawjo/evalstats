"""Numerical (not just visual) stress test of PPI correction on the
between-subjects (unpaired) path: does compare(design="unpaired",
alignment=...) actually control Type-I error under a biased judge, and
retain reasonable power when there's a real difference?

Motivated by a real bug (fixed alongside this script, see
core/unpaired.py's _compute_group_stats): the marginal per-group mean was
silently NEVER PPI-corrected, only the pairwise Delta-theta/Delta-p was.
Earlier battle-testing (investigate_final_stress_test.py) only checked that
PPI output didn't crash and looked visually sane -- it never compared
corrected vs. uncorrected numbers, which is exactly the check that would
have caught this. This script is that check, done properly: Monte Carlo
empirical Type-I error and power, across data types and both a null (no
true difference) and a biased-judge condition.

Core design, per condition:
  - Two groups, SAME true population distribution (Type-I conditions) or a
    known true difference (power conditions).
  - Group A's judge carries a deliberate, systematic bias (mean-shifting,
    like the paper's meme-misread scenario); Group B's judge is well-
    calibrated (noise only).
  - n_lab items/group are human-labeled (MCAR/random).
  - Run BOTH raw (no alignment=) and PPI-corrected (alignment=) pairwise
    comparisons; record whether each flags "significant" at alpha=0.05.
  - Under the null: empirical false-positive rate should be ~5% for the
    corrected comparison and can be far higher for the raw one (that's the
    whole point of PPI correction). Under a real difference: corrected
    power should be non-trivial (not destroyed by the correction).

Run:
    .venv/bin/python -m simulations.investigate_unpaired_ppi_calibration
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import evalstats as es
from evalstats.alignment import judge_alignment

warnings.filterwarnings("ignore")

ALPHA = 0.05
N_BOOT = 500  # modest, per "speed up diagnostic scripts" -- exploratory, not final numbers
N_PER_GROUP = 50
N_LAB = 15


def make_continuous(seed: int, *, true_diff: float, biased: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_PER_GROUP):
        true = float(np.clip(rng.normal(0.5, 0.15), 0, 1))
        judge = float(np.clip(true - 0.25, 0, 1)) if biased else float(np.clip(true + rng.normal(0, 0.04), 0, 1))
        rows.append({"model": "A", "item": f"A_{i}", "llm_score": judge,
                      "human_score": true if i < N_LAB else np.nan})
    for i in range(N_PER_GROUP):
        true = float(np.clip(rng.normal(0.5 + true_diff, 0.15), 0, 1))
        judge = float(np.clip(true + rng.normal(0, 0.04), 0, 1))
        rows.append({"model": "B", "item": f"B_{i}", "llm_score": judge,
                      "human_score": true if i < N_LAB else np.nan})
    return pd.DataFrame(rows)


def make_likert(seed: int, *, true_diff: float, biased: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_PER_GROUP):
        true = int(np.clip(round(rng.normal(3.0, 0.9)), 1, 5))
        judge = int(np.clip(true - 2, 1, 5)) if biased else int(np.clip(round(true + rng.normal(0, 0.3)), 1, 5))
        rows.append({"model": "A", "item": f"A_{i}", "llm_score": judge,
                      "human_score": true if i < N_LAB else np.nan})
    for i in range(N_PER_GROUP):
        true = int(np.clip(round(rng.normal(3.0 + true_diff, 0.9)), 1, 5))
        judge = int(np.clip(round(true + rng.normal(0, 0.3)), 1, 5))
        rows.append({"model": "B", "item": f"B_{i}", "llm_score": judge,
                      "human_score": true if i < N_LAB else np.nan})
    return pd.DataFrame(rows)


def make_binary(seed: int, *, true_diff: float, biased: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_PER_GROUP):
        true = float(rng.binomial(1, 0.5))
        judge = 1.0 - true if (biased and rng.random() < 0.6) else (true if rng.random() > 0.1 else 1 - true)
        rows.append({"model": "A", "item": f"A_{i}", "llm_score": judge,
                      "human_score": true if i < N_LAB else np.nan})
    for i in range(N_PER_GROUP):
        p_b = min(max(0.5 + true_diff, 0.01), 0.99)
        true = float(rng.binomial(1, p_b))
        judge = true if rng.random() > 0.1 else 1 - true
        rows.append({"model": "B", "item": f"B_{i}", "llm_score": judge,
                      "human_score": true if i < N_LAB else np.nan})
    return pd.DataFrame(rows)


def run_condition(name: str, maker, *, true_diff: float, biased: bool, n_reps: int, base_seed: int):
    n_raw_sig = 0
    n_ppi_sig = 0
    n_errors = 0
    for rep in range(n_reps):
        seed = base_seed + rep
        df = maker(seed, true_diff=true_diff, biased=biased)
        try:
            evaldata = es.load_from(df)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score", selection="random")
            r_raw = es.compare(evaldata, factors="model", metric="llm_score", design="unpaired",
                                rng=np.random.default_rng(seed + 100000), n_bootstrap=N_BOOT)
            r_ppi = es.compare(evaldata, factors="model", metric="llm_score", design="unpaired",
                                alignment={"llm_score": ar}, rng=np.random.default_rng(seed + 100000), n_bootstrap=N_BOOT)
            if r_raw.pairwise[0].significant:
                n_raw_sig += 1
            if r_ppi.pairwise[0].significant:
                n_ppi_sig += 1
        except Exception:
            n_errors += 1

    n_valid = n_reps - n_errors
    raw_rate = n_raw_sig / n_valid if n_valid else float("nan")
    ppi_rate = n_ppi_sig / n_valid if n_valid else float("nan")
    kind = "power" if true_diff != 0 else "Type-I error"
    print(f"{name:<55s} n={n_valid:<4d} errors={n_errors:<3d} "
          f"raw {kind}={raw_rate:.3f}  PPI-corrected {kind}={ppi_rate:.3f}")
    return raw_rate, ppi_rate, n_errors


def main():
    N_REPS = 300
    print("=" * 100)
    print(f"Type-I error under a biased judge (true diff=0, nominal alpha={ALPHA}, N_REPS={N_REPS})")
    print("=" * 100)
    results = {}
    results["continuous_biased_null"] = run_condition(
        "continuous, group A judge biased, true diff=0", make_continuous,
        true_diff=0.0, biased=True, n_reps=N_REPS, base_seed=1000,
    )
    results["likert_biased_null"] = run_condition(
        "likert 1-5, group A judge biased, true diff=0", make_likert,
        true_diff=0.0, biased=True, n_reps=N_REPS, base_seed=2000,
    )
    results["binary_biased_null"] = run_condition(
        "binary, group A judge biased, true diff=0", make_binary,
        true_diff=0.0, biased=True, n_reps=N_REPS, base_seed=3000,
    )

    print()
    print("=" * 100)
    print("Sanity check: Type-I error with NO judge bias at all (PPI shouldn't hurt when unneeded)")
    print("=" * 100)
    results["continuous_unbiased_null"] = run_condition(
        "continuous, no bias, true diff=0", make_continuous,
        true_diff=0.0, biased=False, n_reps=N_REPS, base_seed=4000,
    )
    results["likert_unbiased_null"] = run_condition(
        "likert 1-5, no bias, true diff=0", make_likert,
        true_diff=0.0, biased=False, n_reps=N_REPS, base_seed=5000,
    )

    print()
    print("=" * 100)
    print(f"Power under a biased judge (true diff != 0, N_REPS={N_REPS})")
    print("=" * 100)
    results["continuous_biased_power"] = run_condition(
        "continuous, group A judge biased, true diff=+0.25", make_continuous,
        true_diff=0.25, biased=True, n_reps=N_REPS, base_seed=6000,
    )
    results["likert_biased_power"] = run_condition(
        "likert 1-5, group A judge biased, true diff=+1.2", make_likert,
        true_diff=1.2, biased=True, n_reps=N_REPS, base_seed=7000,
    )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for name, (raw_rate, ppi_rate, n_err) in results.items():
        flag = ""
        if "null" in name and ppi_rate > 0.10:
            flag = "  <-- PPI Type-I error looks inflated (>10%)"
        if "power" in name and ppi_rate < 0.3:
            flag = "  <-- PPI power looks low (<30%)"
        print(f"  {name:<30s} raw={raw_rate:.3f}  ppi={ppi_rate:.3f}  errors={n_err}{flag}")


if __name__ == "__main__":
    main()
