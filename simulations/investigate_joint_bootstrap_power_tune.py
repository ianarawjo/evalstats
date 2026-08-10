"""One-off head-to-head: evalstats.api._ppi_bootstrap_t_joint_stats's fixed
lambda=1 construction vs. its new power_tune=True path, for the three
downstream constructions it backs (Romano-Wolf step-down p-values, max-T
simultaneous CIs, "boot" CI-widening via alpha_eff).

Monkeypatches the module-level function to force power_tune, then runs the
REAL es.compare() pipeline end-to-end -- this measures exactly what
flipping the default would change in production, with no risk of
hand-reconstructing the surrounding pair_keys/entity_idx/alpha scaffolding
incorrectly.

Usage: python -m simulations.investigate_joint_bootstrap_power_tune
"""
from __future__ import annotations

import functools
import sys
import time
import warnings

sys.path.insert(0, "tests")

import evalstats as es
import evalstats.api as api_mod
from evalstats.alignment import validate_alignment
from test_compound_ppi_fwer import _make_multiarm_binary, _rng

N_REPS_NULL = 200
N_REPS_POWER = 150


class _PowerTuneOverride:
    """Context manager: forces power_tune on every
    _ppi_bootstrap_t_joint_stats call for the duration of the block."""

    def __init__(self, power_tune: bool):
        self.power_tune = power_tune
        self._orig = api_mod._ppi_bootstrap_t_joint_stats

    def __enter__(self):
        orig = self._orig
        pt = self.power_tune

        @functools.wraps(orig)
        def _wrapped(*args, **kwargs):
            kwargs["power_tune"] = pt
            return orig(*args, **kwargs)

        api_mod._ppi_bootstrap_t_joint_stats = _wrapped
        return self

    def __exit__(self, *exc):
        api_mod._ppi_bootstrap_t_joint_stats = self._orig


def measure_romano_wolf(power_tune: bool, seed_base: int = 4000, effect_size: float = 0.30) -> float:
    alpha = 0.05
    n_detect = 0
    with _PowerTuneOverride(power_tune):
        for i in range(N_REPS_POWER):
            evaldata = _make_multiarm_binary(
                n_entities=4, n_items=150, n_labeled=60,
                p_base=0.55, effect_size=effect_size, agreement_rate=0.85, seed=seed_base + i,
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
            p = result._primary_bundle().pairwise.get("M0", "M3").p_value
            if p is not None and p < alpha:
                n_detect += 1
    return n_detect / N_REPS_POWER


def measure_romano_wolf_null_fwer(power_tune: bool, seed_base: int = 3000, n_reps: int = N_REPS_NULL) -> float:
    alpha = 0.05
    n_reject = 0
    with _PowerTuneOverride(power_tune):
        for i in range(n_reps):
            evaldata = _make_multiarm_binary(
                n_entities=4, n_items=150, n_labeled=60,
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
            if any(pr.p_value < alpha for pr in bundle.pairwise.results.values()):
                n_reject += 1
    return n_reject / n_reps


def _compare_with_max_t(evaldata, ar, seed: int):
    """max-T is never auto-selected (matching the non-PPI decision tree, which
    has no max-T node) -- only reachable via _run_alignment_ppi's explicit
    prefer="max_t", same as tests/test_compound_ppi_fwer.py's own pattern."""
    from evalstats.api import _run_alignment_ppi

    result = es.compare(
        evaldata, factors="model", metric="llm_score",
        method="bootstrap_t", simultaneous_ci=True, correction="none",
    )
    df = evaldata._df.copy()
    _run_alignment_ppi(
        result, df=df, metric_col="llm_score", factor_col="model", item_col="item",
        alignment_result=ar, alpha=0.05, n_boot=1000, correction="none",
        method="bootstrap_t", rng=_rng(seed), prefer="max_t",
    )
    return result


def measure_max_t(power_tune: bool, seed_base: int = 5000, effect_size: float = 0.30) -> tuple[float, float]:
    """Returns (fwer_null, power) for max_t simultaneous CIs, forcing prefer='max_t'."""
    alpha = 0.05
    n_reject_null = 0
    n_detect_power = 0
    with _PowerTuneOverride(power_tune):
        for i in range(N_REPS_NULL):
            evaldata = _make_multiarm_binary(
                n_entities=4, n_items=150, n_labeled=60,
                p_base=0.55, effect_size=0.0, agreement_rate=0.85, seed=seed_base + i,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = validate_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
                result = _compare_with_max_t(evaldata, ar, seed_base + i)
            bundle = result._primary_bundle()
            assert bundle.pairwise.simultaneous_ci_method == "max_t"
            if any(pr.p_value < alpha for pr in bundle.pairwise.results.values()):
                n_reject_null += 1
        for i in range(N_REPS_POWER):
            evaldata = _make_multiarm_binary(
                n_entities=4, n_items=150, n_labeled=60,
                p_base=0.55, effect_size=effect_size, agreement_rate=0.85, seed=seed_base + 10000 + i,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = validate_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
                result = _compare_with_max_t(evaldata, ar, seed_base + 10000 + i)
            p = result._primary_bundle().pairwise.get("M0", "M3").p_value
            if p is not None and p < alpha:
                n_detect_power += 1
    return n_reject_null / N_REPS_NULL, n_detect_power / N_REPS_POWER


if __name__ == "__main__":
    t0 = time.time()

    print("=== Romano-Wolf ===")
    rw_fwer_false = measure_romano_wolf_null_fwer(False)
    rw_fwer_true = measure_romano_wolf_null_fwer(True)
    print(f"null FWER: power_tune=False {rw_fwer_false:.4f}  |  power_tune=True {rw_fwer_true:.4f}")
    rw_power_false = measure_romano_wolf(False)
    rw_power_true = measure_romano_wolf(True)
    print(f"power:     power_tune=False {rw_power_false:.4f}  |  power_tune=True {rw_power_true:.4f}")

    print("\n=== max-T ===")
    mt_fwer_false, mt_power_false = measure_max_t(False)
    mt_fwer_true, mt_power_true = measure_max_t(True)
    print(f"null FWER: power_tune=False {mt_fwer_false:.4f}  |  power_tune=True {mt_fwer_true:.4f}")
    print(f"power:     power_tune=False {mt_power_false:.4f}  |  power_tune=True {mt_power_true:.4f}")

    print(f"\nDone in {time.time() - t0:.1f}s")
