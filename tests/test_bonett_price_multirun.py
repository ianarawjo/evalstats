"""Tests for the multi-run Bonett-Price paired-binary intervals.

Covers, in evalstats/core/resampling.py:
  - bonett_price_paired_ci_flat / _mean          (multi-run baselines)
  - bonett_price_paired_ci_multirun_cluster      (the derivation, no floor)
  - bonett_price_paired_ci_multirun_moments      (floor at w~/R)
  - bonett_price_paired_ci_multirun_effective    (floor at w~/R_eff)

The derivation these all rest on is spelled out in the comment block above
``_bp_item_moments`` in resampling.py. Two of its steps are *claims about
algebra*, and the tests that check them
(``test_bonett_price_is_wald_on_the_augmented_item_sample`` and
``test_kish_design_effect_reduces_to_the_item_level_variance``) are the load
-bearing ones here: if either ever fails, the docstrings are wrong, not just
the code.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from evalstats.core.resampling import (
    bonett_price_paired_ci,
    bonett_price_paired_ci_flat,
    bonett_price_paired_ci_mean,
    bonett_price_paired_ci_multirun_cluster,
    bonett_price_paired_ci_multirun_effective,
    bonett_price_paired_ci_multirun_moments,
)

MULTIRUN_VARIANTS = [
    bonett_price_paired_ci_multirun_cluster,
    bonett_price_paired_ci_multirun_moments,
    bonett_price_paired_ci_multirun_effective,
]
VARIANT_IDS = ["cluster", "moments", "effective"]


def _pairs_from_cells(n11: int, n10: int, n01: int, n00: int):
    """Single-run (a, b) arrays realising a given 2x2 table."""
    a = np.array([1] * n11 + [1] * n10 + [0] * n01 + [0] * n00, dtype=float)
    b = np.array([1] * n11 + [0] * n10 + [1] * n01 + [0] * n00, dtype=float)
    return a, b


def _multirun_corpus(rng, n_cells=40):
    """A varied set of (label, a, b) multi-run matrices."""
    out = []
    for _ in range(n_cells):
        n, runs = int(rng.integers(1, 80)), int(rng.integers(2, 12))
        kind = rng.integers(0, 4)
        if kind == 0:                                    # homogeneous items
            pa, pb = rng.uniform(0, 1), rng.uniform(0, 1)
            a = (rng.random((n, runs)) < pa).astype(float)
            b = (rng.random((n, runs)) < pb).astype(float)
        elif kind == 1:                                  # item heterogeneity
            p = rng.beta(1.5, 1.5, n)
            q = np.clip(p + rng.uniform(-0.3, 0.3), 0.0, 1.0)
            a = (rng.random((n, runs)) < p[:, None]).astype(float)
            b = (rng.random((n, runs)) < q[:, None]).astype(float)
        elif kind == 2:                                  # near-total agreement
            a = (rng.random((n, runs)) < 0.97).astype(float)
            b = a.copy()
            flip = rng.random((n, runs)) < 0.02
            b[flip] = 1.0 - b[flip]
        else:                                            # a corner
            a, b = np.ones((n, runs)), np.zeros((n, runs))
        out.append((f"kind={kind} n={n} runs={runs}", a, b))
    return out


# ---------------------------------------------------------------------------
# The derivation itself
# ---------------------------------------------------------------------------

def test_bonett_price_is_wald_on_the_augmented_item_sample():
    """BP == plain Wald on D_i, over the sample augmented with D = +1 and -1.

    This identity is what makes the multi-run generalisation well defined:
    the Laplace "+1 / +2" is two extra ITEMS, not a reweighting of the 2x2
    table, so extending it to (n_items, n_runs) data means adding two extra
    items -- never scaling the pseudo-counts by the number of runs.
    """
    rng = np.random.default_rng(11)
    for _ in range(300):
        n = int(rng.integers(1, 120))
        a = (rng.random(n) < rng.uniform(0, 1)).astype(float)
        b = (rng.random(n) < rng.uniform(0, 1)).astype(float)
        for alpha in (0.01, 0.05, 0.20):
            d = np.concatenate([a - b, [1.0, -1.0]])     # the two pseudo-items
            n_aug = len(d)                               # == n + 2
            centre = d.mean()                            # pseudo-items cancel
            var = (d * d).mean() - centre * centre       # ddof=0 plug-in
            z = float(stats.norm.ppf(1.0 - alpha / 2.0))
            se = np.sqrt(var / n_aug)
            expected = (
                float(np.clip(centre - z * se, -1.0, 1.0)),
                float(np.clip(centre + z * se, -1.0, 1.0)),
            )
            np.testing.assert_allclose(
                bonett_price_paired_ci(a, b, alpha), expected, atol=1e-14
            )


def test_kish_design_effect_reduces_to_the_item_level_variance():
    """Kish's R_eff, applied correctly, IS the item-level (cluster) variance.

    Pool all n*runs observations for a run-level variance ``B``, estimate the
    run-level ICC from the UNBIASED within-item variance, and inflate by the
    design effect ``1 + (R-1)*rho``: the result is the augmented item-level
    variance ``V`` identically, not just in expectation. That is why
    :func:`bonett_price_paired_ci_multirun_cluster` needs no correlation
    term, and why the ``effective`` variant's own rho has to be a different
    (heuristic) quantity to do anything at all.
    """
    rng = np.random.default_rng(12)
    for label, a, b in _multirun_corpus(rng, n_cells=120):
        n, runs = a.shape
        d = (a >= 0.5).astype(np.int8) - (b >= 0.5).astype(np.int8)
        delta_i = np.mean(d, axis=1, dtype=float)
        u_i = np.mean(np.abs(d), axis=1, dtype=float)

        n_aug = n + 2.0
        centre = float(np.sum(delta_i)) / n_aug
        v_item = (float(np.sum(delta_i**2)) + 2.0) / n_aug - centre**2   # V
        v_run = (float(np.sum(u_i)) + 2.0) / n_aug - centre**2           # B
        within = v_run - v_item                                          # w~
        if v_run <= 0.0:
            continue
        sigma_w_sq = runs / (runs - 1.0) * within        # unbiased within
        rho = 1.0 - sigma_w_sq / v_run                   # run-level ICC
        deff = 1.0 + (runs - 1.0) * rho                  # Kish
        assert v_run * deff / runs == pytest.approx(v_item, abs=1e-13), label


# ---------------------------------------------------------------------------
# Exact reduction at runs == 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", MULTIRUN_VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_multirun_reduces_exactly_to_single_run_over_an_exhaustive_grid(fn, alpha):
    """Every variant must BE bonett_price_paired_ci when runs == 1.

    Exhaustive over every 2x2 table with n <= 10. This holds by construction
    rather than by a special case: at runs == 1 each delta_i is in {-1, 0, 1},
    so sum(delta_i) = n10 - n01 and sum(delta_i^2) = n10 + n01, and every
    within-item variance w_i is 0 so no floor can engage.
    """
    for n in range(1, 11):
        for n10 in range(n + 1):
            for n01 in range(n - n10 + 1):
                for n11 in range(n - n10 - n01 + 1):
                    a, b = _pairs_from_cells(n11, n10, n01, n - n10 - n01 - n11)
                    np.testing.assert_allclose(
                        fn(a[:, None], b[:, None], alpha),
                        bonett_price_paired_ci(a, b, alpha),
                        atol=1e-14,
                        err_msg=f"n={n} n11={n11} n10={n10} n01={n01}",
                    )


@pytest.mark.parametrize("fn", MULTIRUN_VARIANTS, ids=VARIANT_IDS)
def test_multirun_reduces_to_single_run_at_larger_n(fn):
    rng = np.random.default_rng(13)
    for _ in range(200):
        n = int(rng.integers(20, 400))
        a = (rng.random(n) < rng.uniform(0, 1)).astype(float)
        b = (rng.random(n) < rng.uniform(0, 1)).astype(float)
        alpha = float(rng.choice([0.001, 0.01, 0.05, 0.10, 0.20]))
        np.testing.assert_allclose(
            fn(a[:, None], b[:, None], alpha),
            bonett_price_paired_ci(a, b, alpha),
            atol=1e-14,
        )


def test_flat_variant_is_the_first_run_only():
    rng = np.random.default_rng(14)
    a = (rng.random((50, 7)) < 0.7).astype(float)
    b = (rng.random((50, 7)) < 0.55).astype(float)
    np.testing.assert_allclose(
        bonett_price_paired_ci_flat(a, b, 0.05),
        bonett_price_paired_ci(a[:, 0], b[:, 0], 0.05),
        atol=1e-14,
    )
    # 1-D input is forwarded unchanged
    np.testing.assert_allclose(
        bonett_price_paired_ci_flat(a[:, 0], b[:, 0], 0.05),
        bonett_price_paired_ci(a[:, 0], b[:, 0], 0.05),
        atol=1e-14,
    )


def test_mean_variant_thresholds_the_run_means():
    rng = np.random.default_rng(15)
    a = (rng.random((40, 5)) < 0.7).astype(float)
    b = (rng.random((40, 5)) < 0.55).astype(float)
    np.testing.assert_allclose(
        bonett_price_paired_ci_mean(a, b, 0.05),
        bonett_price_paired_ci(a.mean(axis=1), b.mean(axis=1), 0.05),
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", MULTIRUN_VARIANTS, ids=VARIANT_IDS)
def test_limits_stay_in_range_and_contain_the_point_estimate(fn):
    """Limits are finite, inside [-1, 1], ordered, and bracket the estimate.

    The estimate an interval has to contain is its OWN -- Bonett-Price is
    deliberately biased toward zero by the Laplace shrinkage factor
    n/(n+2), so the relevant centre is that shrunk one.
    """
    rng = np.random.default_rng(16)
    for label, a, b in _multirun_corpus(rng, n_cells=150):
        n = a.shape[0]
        raw = float(np.mean(a.mean(axis=1) - b.mean(axis=1)))
        shrunk = raw * n / (n + 2.0)
        for alpha in (0.01, 0.05, 0.10):
            lo, hi = fn(a, b, alpha)
            assert np.isfinite(lo) and np.isfinite(hi), label
            assert -1.0 <= lo <= hi <= 1.0, (label, alpha, lo, hi)
            assert lo <= shrunk <= hi, (label, alpha, lo, hi, shrunk)


@pytest.mark.parametrize("fn", MULTIRUN_VARIANTS + [
    bonett_price_paired_ci_flat, bonett_price_paired_ci_mean,
], ids=VARIANT_IDS + ["flat", "mean"])
def test_swapping_a_and_b_reflects_the_interval(fn):
    """CI(A, B) == -reverse(CI(B, A)) exactly, for every variant."""
    rng = np.random.default_rng(17)
    for label, a, b in _multirun_corpus(rng, n_cells=120):
        for alpha in (0.01, 0.05, 0.10):
            lo, hi = fn(a, b, alpha)
            lo_rev, hi_rev = fn(b, a, alpha)
            assert lo == pytest.approx(-hi_rev, abs=1e-13), label
            assert hi == pytest.approx(-lo_rev, abs=1e-13), label


@pytest.mark.parametrize("fn", MULTIRUN_VARIANTS, ids=VARIANT_IDS)
def test_duplicated_runs_buy_no_information(fn):
    """R identical copies of one run must reproduce the single-run interval.

    The sharpest available check that the variants read the item, not the
    observation, as the unit of analysis: repeating a run R times adds no
    information, so the interval must not narrow by so much as a float.
    """
    rng = np.random.default_rng(18)
    for _ in range(120):
        n, runs = int(rng.integers(1, 150)), int(rng.integers(2, 12))
        a1 = (rng.random(n) < rng.uniform(0, 1)).astype(float)
        b1 = (rng.random(n) < rng.uniform(0, 1)).astype(float)
        a = np.tile(a1[:, None], (1, runs))
        b = np.tile(b1[:, None], (1, runs))
        for alpha in (0.01, 0.05):
            np.testing.assert_allclose(
                fn(a, b, alpha), bonett_price_paired_ci(a1, b1, alpha), atol=1e-14
            )


def test_widths_order_cluster_then_moments_then_effective():
    """The three variants differ only by the floor they put on V~."""
    rng = np.random.default_rng(19)
    for label, a, b in _multirun_corpus(rng, n_cells=150):
        wc, wm, we = (
            float(np.diff(fn(a, b, 0.05))[0]) for fn in MULTIRUN_VARIANTS
        )
        assert wc <= wm + 1e-12, label
        assert wm <= we + 1e-12, label


def test_cluster_narrows_monotonically_with_more_runs():
    """More runs per item genuinely buy precision, when runs carry noise."""
    rng = np.random.default_rng(20)
    p = rng.beta(2.0, 2.0, 120)
    widths = []
    for runs in (1, 2, 4, 8, 16):
        a = (rng.random((120, runs)) < p[:, None]).astype(float)
        b = (rng.random((120, runs)) < np.clip(p + 0.05, 0, 1)[:, None]).astype(float)
        widths.append(float(np.diff(bonett_price_paired_ci_multirun_cluster(a, b, 0.05))[0]))
    assert all(x > y for x, y in zip(widths, widths[1:])), widths


def test_never_degenerates_on_total_agreement():
    """Zero observed discordance at any R still gives a real interval.

    The single-run guarantee (:func:`bonett_price_paired_ci` has no
    zero-width case) has to survive into multi-run, and its width must not
    depend on R at all: N identical items tell you nothing more about the
    items you never sampled just because you re-ran each of them.
    """
    for fn in MULTIRUN_VARIANTS:
        widths = set()
        for runs in (1, 2, 8, 64):
            a = np.ones((30, runs))
            b = np.ones((30, runs))
            lo, hi = fn(a, b, 0.05)
            assert hi > lo
            assert lo <= 0.0 <= hi
            widths.add(round(hi - lo, 12))
        assert len(widths) == 1, (fn.__name__, widths)


@pytest.mark.parametrize("fn", MULTIRUN_VARIANTS, ids=VARIANT_IDS)
def test_rejects_bad_shapes(fn):
    with pytest.raises(ValueError):
        fn(np.ones((5, 3)), np.ones((5, 4)), 0.05)
    with pytest.raises(ValueError):
        fn(np.ones(5), np.ones(5), 0.05)
    with pytest.raises(ValueError):          # zero runs, not a ZeroDivisionError
        fn(np.ones((5, 0)), np.ones((5, 0)), 0.05)


# ---------------------------------------------------------------------------
# The derivation that FAILED, kept as a regression guard
# ---------------------------------------------------------------------------

def _bp_per_run_laplace(a, b, alpha=0.05):
    """REJECTED variant: Laplace pseudo-items at +-1/R instead of +-1.

    The tempting reading of "scale the pseudo-counts to the amount of data":
    with R runs per item, place the two pseudo-observations at one discordant
    RUN each (delta = +-1/R) rather than one discordant ITEM each. It still
    reduces to Bonett-Price at R == 1, which is exactly what makes it
    plausible enough to need a test.
    """
    d = (a >= 0.5).astype(np.int8) - (b >= 0.5).astype(np.int8)
    delta_i = np.mean(d, axis=1, dtype=float)
    n, runs = a.shape
    n_aug = n + 2.0
    centre = float(np.sum(delta_i)) / n_aug
    m2 = (float(np.sum(delta_i**2)) + 2.0 / (runs * runs)) / n_aug
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se = np.sqrt(max(m2 - centre * centre, 0.0) / n_aug)
    return float(np.clip(centre - z * se, -1, 1)), float(np.clip(centre + z * se, -1, 1))


def test_per_run_laplace_scaling_degenerates():
    """Why the pseudo-counts stay on the item scale.

    Scaling them by R makes the regularisation vanish as R grows: at zero
    observed discordance the interval collapses toward zero width, which is
    the exact degeneracy the Laplace adjustment exists to prevent. Item-level
    heterogeneity is bounded by N, not N*R -- re-running the same 30 items
    64 times says nothing about the items that were never sampled.
    """
    a, b = np.ones((30, 1)), np.ones((30, 1))
    assert _bp_per_run_laplace(a, b) == pytest.approx(
        bonett_price_paired_ci(a[:, 0], b[:, 0], 0.05), abs=1e-14
    )  # ... it does reduce correctly at R == 1, which is the trap

    widths = []
    for runs in (1, 4, 16, 64):
        a, b = np.ones((30, runs)), np.ones((30, runs))
        widths.append(float(np.diff(_bp_per_run_laplace(a, b))[0]))
        # the shipped variant is flat in R here (see the test above)
        assert float(np.diff(bonett_price_paired_ci_multirun_cluster(a, b, 0.05))[0]) == \
            pytest.approx(widths[0], abs=1e-12)
    assert widths[-1] < widths[0] / 30.0, widths       # 1/R collapse
