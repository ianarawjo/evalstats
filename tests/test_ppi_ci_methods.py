"""Tests for the PPI-corrected t-interval and logit-t CI methods.

evalstats.ppi._analytic_mean_correct/_analytic_logit_t_correct and their
evalstats.tests thin wrappers (_ppi_single_t_interval/_ppi_paired_t_interval/
_ppi_single_logit_t/_ppi_paired_logit_t) -- added 2026-08-05 to close a
previously-documented gap (evalstats/config.py's PPI_AUTO_METHOD_TABLE used
to silently fall back to bootstrap_t for bounded_01/unbounded numeric data
because no PPI-corrected logit_t/t-interval existed).

No pre-existing pytest coverage exists for _ppi_single_wilson/
_ppi_paired_mj_floor/_ppi_paired_bootstrap_t as standalone functions (confirmed
by repo-wide grep before writing this file), so there's no established bar
to match -- these tests are written from scratch, following this codebase's
own stated testing principles (tests/test_ppi_corrections.py's module
docstring): quantitative checks over multiple seeds, not single-draw
assertions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from evalstats.ppi import _analytic_mean_correct, _analytic_logit_t_correct, _analytic_mean_point_se
from evalstats.tests import (
    _ppi_single_t_interval,
    _ppi_paired_t_interval,
    _ppi_single_logit_t,
    _ppi_paired_logit_t,
)
from evalstats.config import resolve_ppi_auto_methods
from evalstats.api import _ppi_pairwise_dispatch, _ppi_robustness_dispatch


def _split_labels(rng, truth, llm, n_lab):
    n = len(truth)
    lab = np.full(n, np.nan)
    idx = rng.choice(n, size=n_lab, replace=False)
    lab[idx] = truth[idx]
    return lab


class TestAnalyticMeanPointSeRefactorSafety:
    """_analytic_mean_correct was refactored to delegate its point-estimate/
    variance derivation to _analytic_mean_point_se -- this is meant to be a
    pure extraction with unchanged arithmetic. Assert that directly rather
    than trusting it by inspection."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 99])
    @pytest.mark.parametrize("power_tune", [False, True])
    def test_point_se_matches_full_correct(self, seed, power_tune):
        rng = np.random.default_rng(seed)
        n_lab, n_all = 25, 150
        truth_lab = rng.uniform(0.2, 0.8, n_lab)
        llm_lab = truth_lab + rng.normal(0, 0.1, n_lab)
        llm_unlab = rng.uniform(0.2, 0.8, n_all) + rng.normal(0, 0.1, n_all)

        full = _analytic_mean_correct(truth_lab, llm_lab, llm_unlab, alpha=0.05, power_tune=power_tune)
        estimate, se, f_unlab, f_lab, rectifier, lam, df = _analytic_mean_point_se(
            truth_lab, llm_lab, llm_unlab, power_tune=power_tune,
        )

        assert estimate == pytest.approx(full.estimate, abs=1e-12)
        assert f_unlab == pytest.approx(full.llm_estimate, abs=1e-12)
        assert f_lab == pytest.approx(full.human_estimate, abs=1e-12)
        assert rectifier == pytest.approx(full.rectifier, abs=1e-12)
        if power_tune:
            assert lam == pytest.approx(full.lam, abs=1e-12)
        else:
            assert full.lam is None


class TestWrapperEquivalence:
    """_ppi_single_t_interval/_ppi_paired_t_interval are thin wrappers around
    _analytic_mean_correct -- their output must be bit-identical to calling
    it directly on the same split arrays."""

    def test_single_t_interval_matches_analytic_mean_correct(self):
        rng = np.random.default_rng(11)
        n = 120
        truth = rng.uniform(-5, 5, n)
        llm = truth + rng.normal(0, 1.0, n)
        a_lab = _split_labels(rng, truth, llm, n_lab=30)
        mask = ~np.isnan(a_lab)

        r = _ppi_single_t_interval(llm, a_lab, alpha=0.05, power_tune=False)
        expected = _analytic_mean_correct(
            np.asarray(a_lab, dtype=float)[mask], llm[mask], llm[~mask], alpha=0.05, power_tune=False,
        )
        assert r.estimate == pytest.approx(expected.estimate, abs=1e-12)
        assert r.ci_low == pytest.approx(expected.ci_low, abs=1e-12)
        assert r.ci_high == pytest.approx(expected.ci_high, abs=1e-12)
        assert r.p_value == pytest.approx(expected.p_value, abs=1e-12)

    def test_paired_t_interval_matches_analytic_mean_correct(self):
        rng = np.random.default_rng(12)
        n = 120
        truth_a = rng.uniform(-5, 5, n)
        truth_b = truth_a + rng.normal(0, 0.5, n)
        llm_a = truth_a + rng.normal(0, 1.0, n)
        llm_b = truth_b + rng.normal(0, 1.0, n)
        a_lab = _split_labels(rng, truth_a, llm_a, n_lab=40)
        b_lab = a_lab.copy()
        b_lab[~np.isnan(a_lab)] = truth_b[~np.isnan(a_lab)]
        mask = ~np.isnan(a_lab) & ~np.isnan(b_lab)

        r = _ppi_paired_t_interval(llm_a, llm_b, a_lab, b_lab, alpha=0.05, power_tune=False)
        diffs = llm_a - llm_b
        expected = _analytic_mean_correct(
            (a_lab - b_lab)[mask], diffs[mask], diffs[~mask], alpha=0.05, power_tune=False,
        )
        assert r.estimate == pytest.approx(expected.estimate, abs=1e-12)
        assert r.ci_low == pytest.approx(expected.ci_low, abs=1e-12)
        assert r.ci_high == pytest.approx(expected.ci_high, abs=1e-12)


class TestLogitTMatchesTIntervalOnEstimand:
    """Logit-t and t-interval share the identical closed-form point
    estimate/variance/p-value (evalstats.ppi._analytic_mean_point_se) --
    only the CI's shape differs. On identical inputs, estimate/p_value/
    rectifier must match exactly between the two wrapper families."""

    @pytest.mark.parametrize("seed", [0, 5, 21, 42])
    def test_single(self, seed):
        rng = np.random.default_rng(seed)
        n = 100
        truth = rng.uniform(0.1, 0.9, n)
        llm = np.clip(truth + rng.normal(0, 0.08, n), 0.0, 1.0)
        a_lab = _split_labels(rng, truth, llm, n_lab=25)

        r_t = _ppi_single_t_interval(llm, a_lab, alpha=0.05)
        r_l = _ppi_single_logit_t(llm, a_lab, alpha=0.05, lo=0.0, hi=1.0)

        assert r_t.estimate == pytest.approx(r_l.estimate, abs=1e-9)
        assert r_t.p_value == pytest.approx(r_l.p_value, abs=1e-9)
        assert r_t.rectifier == pytest.approx(r_l.rectifier, abs=1e-9)
        assert 0.0 <= r_l.ci_low <= r_l.ci_high <= 1.0

    @pytest.mark.parametrize("seed", [0, 5, 21, 42])
    def test_paired(self, seed):
        rng = np.random.default_rng(seed)
        n = 100
        truth_a = rng.uniform(0.2, 0.8, n)
        truth_b = np.clip(truth_a + 0.05, 0.0, 1.0)
        llm_a = np.clip(truth_a + rng.normal(0, 0.06, n), 0.0, 1.0)
        llm_b = np.clip(truth_b + rng.normal(0, 0.06, n), 0.0, 1.0)
        a_lab = _split_labels(rng, truth_a, llm_a, n_lab=30)
        b_lab = a_lab.copy()
        b_lab[~np.isnan(a_lab)] = truth_b[~np.isnan(a_lab)]

        r_t = _ppi_paired_t_interval(llm_a, llm_b, a_lab, b_lab, alpha=0.05)
        r_l = _ppi_paired_logit_t(llm_a, llm_b, a_lab, b_lab, alpha=0.05, lo=0.0, hi=1.0)

        assert r_t.estimate == pytest.approx(r_l.estimate, abs=1e-9)
        assert r_t.p_value == pytest.approx(r_l.p_value, abs=1e-9)
        assert -1.0 <= r_l.ci_low <= r_l.ci_high <= 1.0


class TestLogitTBoundaryHandling:
    """_analytic_logit_t_correct's degenerate/out-of-range handling."""

    def test_zero_variance_collapses_to_point_interval(self):
        truth_lab = np.full(15, 0.5)
        llm_lab = np.full(15, 0.5)
        llm_unlab = np.full(30, 0.5)
        r = _analytic_logit_t_correct(truth_lab, llm_lab, llm_unlab, alpha=0.05, power_tune=False)
        assert r.ci_low == r.ci_high == pytest.approx(r.estimate)

    def test_out_of_range_estimate_warns_and_clips_ci(self):
        rng = np.random.default_rng(3)
        truth_lab = np.full(20, 0.98) + rng.normal(0, 0.01, 20)
        llm_lab = np.full(20, 0.05) + rng.normal(0, 0.01, 20)
        llm_unlab = np.full(50, 0.90) + rng.normal(0, 0.01, 50)

        with pytest.warns(UserWarning, match="outside"):
            r = _analytic_logit_t_correct(truth_lab, llm_lab, llm_unlab, alpha=0.05, power_tune=False)

        assert r.estimate > 1.0  # reported point estimate stays UN-clipped
        assert 0.0 <= r.ci_low <= r.ci_high <= 1.0  # but the CI is clamped

    def test_no_spurious_warning_when_estimate_in_range(self):
        rng = np.random.default_rng(4)
        n = 60
        truth = rng.uniform(0.3, 0.7, n)
        llm = truth + rng.normal(0, 0.05, n)
        lab = _split_labels(rng, truth, llm, n_lab=20)
        mask = ~np.isnan(lab)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _analytic_logit_t_correct(
                np.asarray(lab, dtype=float)[mask], llm[mask], llm[~mask], alpha=0.05, power_tune=False,
            )
            assert not any(issubclass(x.category, UserWarning) and "outside" in str(x.message) for x in w)


class TestMonteCarloCoverage:
    """Concrete test of the confirmed design decision: analytic-always (no
    bootstrap threshold) must hold up across a range of N_lab, not just
    asymptotically. Mirrors the N_lab range evalstats.ppi.
    _ANALYTIC_ALWAYS_PREFERRED's own Wilcoxon precedent was validated at
    (30-200)."""

    @pytest.mark.parametrize("n_lab", [15, 30, 60, 130])
    def test_t_interval_coverage_near_nominal_unbounded(self, n_lab):
        rng = np.random.default_rng(1000 + n_lab)
        n_reps = 400
        n = 300
        alpha = 0.05
        covered = 0
        for _ in range(n_reps):
            truth = rng.normal(0.0, 2.0, n)
            llm = truth + rng.normal(0, 1.5, n)
            lab = np.full(n, np.nan)
            idx = rng.choice(n, size=n_lab, replace=False)
            lab[idx] = truth[idx]
            r = _ppi_single_t_interval(llm, lab, alpha=alpha)
            if r.ci_low <= 0.0 <= r.ci_high:  # true population mean is 0
                covered += 1
        rate = covered / n_reps
        # binomial MC tolerance around nominal 95% at n_reps=400: SE ~= 1.1pp
        assert 0.90 <= rate <= 1.0, f"n_lab={n_lab}: coverage {rate:.3f} outside tolerance"

    @pytest.mark.parametrize("n_lab", [15, 30, 60, 130])
    def test_logit_t_coverage_near_nominal_bounded(self, n_lab):
        rng = np.random.default_rng(2000 + n_lab)
        n_reps = 400
        n = 300
        alpha = 0.05
        true_mean = 0.5
        covered = 0
        for _ in range(n_reps):
            truth = np.clip(rng.normal(true_mean, 0.15, n), 0.0, 1.0)
            llm = np.clip(truth + rng.normal(0, 0.1, n), 0.0, 1.0)
            lab = np.full(n, np.nan)
            idx = rng.choice(n, size=n_lab, replace=False)
            lab[idx] = truth[idx]
            r = _ppi_single_logit_t(llm, lab, alpha=alpha, lo=0.0, hi=1.0)
            if r.ci_low <= true_mean <= r.ci_high:
                covered += 1
        rate = covered / n_reps
        assert 0.90 <= rate <= 1.0, f"n_lab={n_lab}: coverage {rate:.3f} outside tolerance"


class TestConfigRouting:
    def test_bounded_01_routes_to_ppi_logit_t(self):
        assert resolve_ppi_auto_methods("bounded_01") == ("ppi_logit_t", "ppi_logit_t")

    def test_unbounded_routes_to_ppi_t_interval(self):
        assert resolve_ppi_auto_methods("unbounded") == ("ppi_t_interval", "ppi_t_interval")

    def test_binary_routes_to_bonett_price(self):
        """Paired binary PPI routes to bonett_price, whose Laplace adjustment
        keeps the interval from collapsing toward zero width when the labeled
        subset carries little discordance information. mj_floor remains
        implemented (evalstats.tests._ppi_paired_mj_floor) and directly
        callable, but is no longer the auto-routed default."""
        assert resolve_ppi_auto_methods("binary") == ("bonett_price", "wilson")


class TestApiDispatch:
    """evalstats/api.py's public dispatch functions correctly route the new
    method strings, and do NOT collide with the pre-existing bare
    "t_interval" string (already mapped to a different, generic PPI-mean-
    diff bootstrap routine)."""

    def test_pairwise_dispatch_ppi_t_interval_and_ppi_logit_t(self):
        rng = np.random.default_rng(5)
        n = 80
        truth_a = rng.uniform(0.2, 0.8, n)
        truth_b = truth_a + 0.05
        llm_a = truth_a + rng.normal(0, 0.05, n)
        llm_b = truth_b + rng.normal(0, 0.05, n)
        a_lab = _split_labels(rng, truth_a, llm_a, n_lab=20)
        b_lab = a_lab.copy()
        b_lab[~np.isnan(a_lab)] = truth_b[~np.isnan(a_lab)]

        r1 = _ppi_pairwise_dispatch("ppi_t_interval", llm_a, llm_b, a_lab, b_lab, 0.05, 200, np.random.default_rng(0))
        r2 = _ppi_pairwise_dispatch("ppi_logit_t", llm_a, llm_b, a_lab, b_lab, 0.05, 200, np.random.default_rng(0))
        assert r1.estimate == pytest.approx(r2.estimate, abs=1e-9)

    def test_pairwise_dispatch_bare_t_interval_unaffected(self):
        """The existing "t_interval" string must still map to the generic
        bootstrap PPI-mean-diff routine, not the new closed-form method --
        no collision introduced by adding "ppi_t_interval"."""
        rng = np.random.default_rng(6)
        n = 80
        truth_a = rng.uniform(0.2, 0.8, n)
        truth_b = truth_a + 0.05
        llm_a = truth_a + rng.normal(0, 0.05, n)
        llm_b = truth_b + rng.normal(0, 0.05, n)
        a_lab = _split_labels(rng, truth_a, llm_a, n_lab=20)
        b_lab = a_lab.copy()
        b_lab[~np.isnan(a_lab)] = truth_b[~np.isnan(a_lab)]

        # Should not raise, and should still be a valid PPIResult (exact
        # values will differ from ppi_t_interval since it's bootstrap-based).
        r = _ppi_pairwise_dispatch("t_interval", llm_a, llm_b, a_lab, b_lab, 0.05, 200, np.random.default_rng(0))
        assert r.ci_low <= r.estimate <= r.ci_high

    def test_robustness_dispatch_ppi_t_interval_and_ppi_logit_t(self):
        rng = np.random.default_rng(7)
        n = 80
        truth = rng.uniform(0.2, 0.8, n)
        llm = truth + rng.normal(0, 0.05, n)
        a_lab = _split_labels(rng, truth, llm, n_lab=20)

        r1 = _ppi_robustness_dispatch("ppi_t_interval", llm, a_lab, 0.05, 200, np.random.default_rng(0))
        r2 = _ppi_robustness_dispatch("ppi_logit_t", llm, a_lab, 0.05, 200, np.random.default_rng(0))
        assert r1.estimate == pytest.approx(r2.estimate, abs=1e-9)
