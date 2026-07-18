"""Tests for evalstats.tests.{ttest, mannwhitney, wilcoxon} — PPI-corrected wrappers.

Design principles for rigor:
  - Bias correction tests are parametrized over MULTIPLE seeds so a single
    lucky draw cannot produce a false pass.
  - Bias correction tests are parametrized over ALL FOUR function variants
    (independent ttest, mannwhitney, paired ttest, wilcoxon) in one block so
    missing a function is structurally impossible.
  - Core property checks are QUANTITATIVE: the corrected estimate must be
    closer to the true parameter than the raw LLM estimate, not just "CI
    contains null."
  - Rectifier is tested for correct magnitude (scales linearly with bias),
    not just sign.
  - CI width is tested to be monotonically narrower as more items are labeled.
  - Non-Gaussian distributions (binary 0/1, Likert 1–5) are tested to ensure
    the correction is not tied to normality.

Scenarios:
  - Baseline: no labels → results match SciPy exactly, all PPI fields None
  - PPI field structure: populated, CI ordered, n_labeled/n_total, bounds
  - Differential bias (false positive): LLM inflates one group → all 4 fns,
    5 seeds each; CI contains null AND corrected estimate closer to truth
  - Symmetric bias: equal inflation → rectifier ≈ 0 for all 4 fns
  - True effect preserved: large real effect still detected after PPI
  - Rectifier magnitude: tracks linear bias across 5 bias values
  - CI width: monotonically narrows with more labels (ttest)
  - Non-Gaussian: binary and Likert false-positive tests
  - Paired tests: positional pairing, overlap requirement, length validation
  - Mann-Whitney: P(X>Y) semantics, stochastic dominance
  - Input handling: lists, partial labels, fully labeled, validation errors
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats as scipy_stats

from evalstats.tests import (
    TestResult,
    ttest,
    mannwhitney,
    wilcoxon,
    anova_oneway,
    friedman,
    kruskalwallis,
    _hedges_g_star,
    _ppi_anova_independent_p_value,
    _ppi_anova_repeated_p_value,
    _ppi_friedman_p_value,
    _friedman_rank_variance,
    _ppi_kruskal_wallis_pairwise,
    _kw_mean_sq_deviation_from_labeled,
)


# ─── Data generators ─────────────────────────────────────────────────────────

def _two_sample(rng, n=200, mu_a=3.5, mu_b=3.0, sigma=1.0,
                bias_a=0.0, bias_b=0.0, n_lab=50, llm_noise=0.10):
    """Independent two-sample data.  LLM = truth + bias + iid noise."""
    truth_a = rng.normal(mu_a, sigma, n)
    truth_b = rng.normal(mu_b, sigma, n)
    a = truth_a + bias_a + rng.normal(0, llm_noise, n)
    b = truth_b + bias_b + rng.normal(0, llm_noise, n)
    idx_a = rng.choice(n, n_lab, replace=False)
    idx_b = rng.choice(n, n_lab, replace=False)
    a_lab = np.full(n, np.nan); a_lab[idx_a] = truth_a[idx_a]
    b_lab = np.full(n, np.nan); b_lab[idx_b] = truth_b[idx_b]
    return a, b, a_lab, b_lab


def _paired(rng, n=200, mu_a=3.5, mu_b=3.0, sigma=1.0,
            bias_a=0.0, bias_b=0.0, n_lab=50, llm_noise=0.10):
    """Paired within-subject data.  Same positions labeled in both groups."""
    subject_fx = rng.normal(0, 0.5, n)
    truth_a = mu_a + subject_fx + rng.normal(0, sigma, n)
    truth_b = mu_b + subject_fx + rng.normal(0, sigma, n)
    a = truth_a + bias_a + rng.normal(0, llm_noise, n)
    b = truth_b + bias_b + rng.normal(0, llm_noise, n)
    idx = rng.choice(n, n_lab, replace=False)
    a_lab = np.full(n, np.nan); b_lab = np.full(n, np.nan)
    a_lab[idx] = truth_a[idx]; b_lab[idx] = truth_b[idx]
    return a, b, a_lab, b_lab


def _two_sample_binary(rng, n=300, p_a=0.5, p_b=0.5,
                       llm_p_a=0.5, llm_p_b=0.5, n_lab=80):
    """Binary (0/1) two-sample data.  LLM and human drawn from separate Bernoullis."""
    truth_a = rng.binomial(1, p_a, n).astype(float)
    truth_b = rng.binomial(1, p_b, n).astype(float)
    a = rng.binomial(1, llm_p_a, n).astype(float)
    b = rng.binomial(1, llm_p_b, n).astype(float)
    idx_a = rng.choice(n, n_lab, replace=False)
    idx_b = rng.choice(n, n_lab, replace=False)
    al = np.full(n, np.nan); al[idx_a] = truth_a[idx_a]
    bl = np.full(n, np.nan); bl[idx_b] = truth_b[idx_b]
    return a, b, al, bl


def _two_sample_likert(rng, n=300, mu_a=3.0, mu_b=3.0,
                       bias_a=0.0, bias_b=0.0, n_lab=80):
    """Likert 1–5 two-sample data.  Rounded normal, LLM offset and clamped."""
    truth_a = np.clip(np.round(rng.normal(mu_a, 0.8, n)), 1, 5)
    truth_b = np.clip(np.round(rng.normal(mu_b, 0.8, n)), 1, 5)
    a = np.clip(np.round(truth_a + bias_a + rng.normal(0, 0.25, n)), 1, 5)
    b = np.clip(np.round(truth_b + bias_b + rng.normal(0, 0.25, n)), 1, 5)
    idx_a = rng.choice(n, n_lab, replace=False)
    idx_b = rng.choice(n, n_lab, replace=False)
    al = np.full(n, np.nan); al[idx_a] = truth_a[idx_a]
    bl = np.full(n, np.nan); bl[idx_b] = truth_b[idx_b]
    return a, b, al, bl


def _two_sample_unbalanced(rng, n_a=400, n_b=120, mu_a=3.0, mu_b=3.0, sigma=1.0,
                           bias_a=0.0, bias_b=0.0, n_lab_a=80, n_lab_b=25,
                           llm_noise=0.10):
    """Independent two-sample data with unequal group sizes."""
    truth_a = rng.normal(mu_a, sigma, n_a)
    truth_b = rng.normal(mu_b, sigma, n_b)
    a = truth_a + bias_a + rng.normal(0, llm_noise, n_a)
    b = truth_b + bias_b + rng.normal(0, llm_noise, n_b)
    idx_a = rng.choice(n_a, n_lab_a, replace=False)
    idx_b = rng.choice(n_b, n_lab_b, replace=False)
    a_lab = np.full(n_a, np.nan); a_lab[idx_a] = truth_a[idx_a]
    b_lab = np.full(n_b, np.nan); b_lab[idx_b] = truth_b[idx_b]
    return a, b, a_lab, b_lab


def _llm_estimand(call_fn, null, a, b):
    """Raw LLM estimate of the same estimand as the corrected output."""
    if null == 0.5:
        return float(np.mean(a[:, None] > b[None, :]))
    if call_fn is _fn_wilcoxon:
        return float(np.median(a - b))
    return float(a.mean() - b.mean())


# ─── Uniform callables for multi-function parametrize ────────────────────────
# Each wrapper has signature (a, b, al, bl, **kw) → TestResult.
# null_val: the value the corrected CI should contain when there's no true effect
# (mean-diff functions: 0; mannwhitney P(X>Y): 0.5).

def _fn_ttest(a, b, al, bl, **kw):
    return ttest(a, b, a_lab=al, b_lab=bl, **kw)

def _fn_mw(a, b, al, bl, **kw):
    return mannwhitney(a, b, x_lab=al, y_lab=bl, **kw)

def _fn_ttest_paired(a, b, al, bl, **kw):
    return ttest(a, b, a_lab=al, b_lab=bl, paired=True, **kw)

def _fn_wilcoxon(a, b, al, bl, **kw):
    return wilcoxon(a, b, x_lab=al, y_lab=bl, **kw)


# (call_fn, null_val, data_gen, label)
INDEP_CASES = [
    (_fn_ttest,  0.0, _two_sample, "ttest"),
    (_fn_mw,     0.5, _two_sample, "mannwhitney"),
]
PAIRED_CASES = [
    (_fn_ttest_paired, 0.0, _paired, "ttest_paired"),
    (_fn_wilcoxon,     0.0, _paired, "wilcoxon"),
]
ALL_CASES = INDEP_CASES + PAIRED_CASES

_INDEP_PARAMS  = [pytest.param(*c[:3], id=c[3]) for c in INDEP_CASES]
_PAIRED_PARAMS = [pytest.param(*c[:3], id=c[3]) for c in PAIRED_CASES]
_ALL_PARAMS    = [pytest.param(*c[:3], id=c[3]) for c in ALL_CASES]

# Five seeds chosen to be spread across the integer space.
_SEEDS = [101, 202, 303, 404, 505]


# ─── Baseline: no labels → matches SciPy exactly ─────────────────────────────

class TestNoLabelBaseline:

    def test_hedges_g_star_helper_matches_exact_formula(self):
        rng = np.random.default_rng(77)
        a = rng.normal(0.4, 1.5, 33)
        b = rng.normal(-0.2, 0.8, 21)

        n1, n2 = len(a), len(b)
        m1, m2 = float(np.mean(a)), float(np.mean(b))
        s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))

        got = _hedges_g_star(m1, m2, s1, s2, n1, n2)

        d_star = (m1 - m2) / np.sqrt((s1**2 + s2**2) / 2.0)
        nu = (
            (n1 - 1) * (n2 - 1) * (s1**2 + s2**2) ** 2
            / ((n2 - 1) * s1**4 + (n1 - 1) * s2**4)
        )
        j = math.exp(
            math.lgamma(nu / 2.0)
            - 0.5 * math.log(nu / 2.0)
            - math.lgamma((nu - 1.0) / 2.0)
        )

        assert got == pytest.approx(d_star * j, rel=1e-12, abs=1e-12)

    def test_ttest_welch_effect_size_matches_hedges_g_star_s_exact(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0.8, 1.7, 40)
        b = rng.normal(-0.1, 0.9, 27)

        r = ttest(a, b, equal_var=False, print_result=False)

        n1, n2 = len(a), len(b)
        m1, m2 = float(np.mean(a)), float(np.mean(b))
        s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
        d_star = (m1 - m2) / np.sqrt((s1**2 + s2**2) / 2.0)

        nu = (
            (n1 - 1) * (n2 - 1) * (s1**2 + s2**2) ** 2
            / ((n2 - 1) * s1**4 + (n1 - 1) * s2**4)
        )
        j = math.exp(
            math.lgamma(nu / 2.0)
            - 0.5 * math.log(nu / 2.0)
            - math.lgamma((nu - 1.0) / 2.0)
        )

        assert r.extra["effect_size_name"] == "Hedges’ g_s*"
        assert r.extra["effect_size"] == pytest.approx(d_star * j, rel=1e-12, abs=1e-12)

    def test_ttest_independent_matches_scipy(self):
        rng = np.random.default_rng(0)
        a, b, *_ = _two_sample(rng, n=100)
        # Default is Welch's (equal_var=False)
        r = ttest(a, b)
        ref = scipy_stats.ttest_ind(a, b, equal_var=False)
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.p_value == pytest.approx(ref.pvalue)
        # Student's still available via equal_var=True
        r_student = ttest(a, b, equal_var=True)
        ref_student = scipy_stats.ttest_ind(a, b, equal_var=True)
        assert r_student.statistic == pytest.approx(ref_student.statistic)
        assert r_student.p_value == pytest.approx(ref_student.pvalue)

    def test_ttest_paired_matches_scipy(self):
        rng = np.random.default_rng(1)
        a, b, *_ = _paired(rng, n=100)
        r = ttest(a, b, paired=True)
        ref = scipy_stats.ttest_rel(a, b)
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.p_value == pytest.approx(ref.pvalue)

    def test_mannwhitney_matches_scipy(self):
        rng = np.random.default_rng(2)
        a, b, *_ = _two_sample(rng, n=100)
        r = mannwhitney(a, b)
        ref = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.p_value == pytest.approx(ref.pvalue)

    def test_wilcoxon_matches_scipy(self):
        rng = np.random.default_rng(3)
        a, b, *_ = _paired(rng, n=100)
        r = wilcoxon(a, b)
        ref = scipy_stats.wilcoxon(a, b, alternative="two-sided")
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.p_value == pytest.approx(ref.pvalue)

    @pytest.mark.parametrize("fn,kw", [
        (ttest, {}),
        (ttest, {"paired": True}),
        (mannwhitney, {}),
        (wilcoxon, {}),
    ])
    def test_all_ppi_fields_are_none_without_labels(self, fn, kw):
        rng = np.random.default_rng(4)
        a = rng.normal(0, 1, 80)
        b = rng.normal(0, 1, 80)
        r = fn(a, b, **kw)
        assert isinstance(r, TestResult)
        assert r.corrected_estimate is None
        assert r.corrected_ci is None
        assert r.corrected_p_value is None
        assert r.corrected_statistic is None
        assert r.rectifier is None
        assert r.n_labeled is None
        assert r.n_total is None

    def test_test_name_strings(self):
        rng = np.random.default_rng(5)
        a, b, *_ = _two_sample(rng, n=50)
        assert "t-test"       in ttest(a, b).test_name.lower()
        assert "paired"       in ttest(a, b, paired=True).test_name.lower()
        assert "mann-whitney" in mannwhitney(a, b).test_name.lower()
        assert "wilcoxon"     in wilcoxon(a, b).test_name.lower()


# ─── PPI field structure ──────────────────────────────────────────────────────

class TestPPIFieldStructure:

    @pytest.mark.parametrize("call_fn,null,gen", _ALL_PARAMS)
    def test_fields_populated_all_functions(self, call_fn, null, gen):
        rng = np.random.default_rng(10)
        a, b, al, bl = gen(rng)
        r = call_fn(a, b, al, bl, n_boot=200, rng=10)
        assert r.corrected_estimate is not None
        assert r.corrected_ci is not None
        assert r.corrected_p_value is not None
        assert r.rectifier is not None
        assert r.n_labeled is not None
        assert r.n_total is not None

    @pytest.mark.parametrize("call_fn,null,gen", _ALL_PARAMS)
    def test_ci_is_ordered_all_functions(self, call_fn, null, gen):
        rng = np.random.default_rng(11)
        a, b, al, bl = gen(rng)
        r = call_fn(a, b, al, bl, n_boot=200, rng=11)
        lo, hi = r.corrected_ci
        assert lo <= hi

    @pytest.mark.parametrize("call_fn,null,gen", _ALL_PARAMS)
    def test_corrected_p_value_in_0_1_all_functions(self, call_fn, null, gen):
        rng = np.random.default_rng(12)
        a, b, al, bl = gen(rng)
        r = call_fn(a, b, al, bl, n_boot=300, rng=12)
        assert 0.0 <= r.corrected_p_value <= 1.0

    def test_alpha_propagated(self):
        rng = np.random.default_rng(13)
        a, b, al, bl = _two_sample(rng)
        r = ttest(a, b, a_lab=al, b_lab=bl, alpha=0.10, n_boot=200, rng=13)
        assert r.alpha == 0.10

    def test_n_labeled_and_n_total_match_input_sizes(self):
        # n=200 per group, n_lab=50 → concat gives 100 labeled / 400 total
        rng = np.random.default_rng(14)
        a, b, al, bl = _two_sample(rng, n=200, n_lab=50)
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=200, rng=14)
        assert r.n_labeled == 100
        assert r.n_total == 400

    def test_reproducibility_with_rng_seed(self):
        rng = np.random.default_rng(15)
        a, b, al, bl = _two_sample(rng)
        kw = dict(a_lab=al, b_lab=bl, n_boot=500, rng=99)
        r1 = ttest(a, b, **kw)
        r2 = ttest(a, b, **kw)
        assert r1.corrected_estimate == r2.corrected_estimate
        assert r1.corrected_ci == r2.corrected_ci

    def test_rectifier_matches_ppi_formula_exactly(self):
        """θ̂_PPI = f(Ŷ_unlab) + rectifier — verified numerically.

        Y_hat_unlab must be DISJOINT from the labeled positions (see
        evalstats.ppi.correct's docstring), so the reference llm_diff here
        is computed on the unlabeled-only complement, not the full a/b.
        """
        rng = np.random.default_rng(16)
        a, b, al, bl = _two_sample(rng, n=200, mu_a=4.0, mu_b=3.0,
                                   bias_a=0.5, bias_b=0.0, n_lab=60)
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=200, rng=16)
        mask_a, mask_b = ~np.isnan(al), ~np.isnan(bl)
        llm_diff = float(a[~mask_a].mean() - b[~mask_b].mean())
        assert r.corrected_estimate == pytest.approx(llm_diff + r.rectifier, abs=1e-10)


# ─── Differential bias: false positive suppressed across seeds and functions ──

class TestDifferentialBiasFalsePositive:
    """LLM inflates group A by 2σ; groups are truly equal.

    Each test below runs for ALL FOUR function variants × FIVE seeds.
    The CI must contain the null value (qualitative) and the corrected
    estimate must be closer to the true value than the raw LLM estimate
    (quantitative).  5 seeds prevent a single lucky draw from passing.
    """

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("call_fn,null,gen", _ALL_PARAMS)
    def test_corrected_ci_contains_null(self, call_fn, null, gen, seed):
        """Corrected CI contains the null value even though the uncorrected test
        is highly significant (large false positive from LLM differential bias)."""
        rng = np.random.default_rng(seed)
        a, b, al, bl = gen(rng, n=300, mu_a=3.0, mu_b=3.0,
                           bias_a=2.0, bias_b=0.0, n_lab=80)
        r = call_fn(a, b, al, bl, n_boot=500, rng=seed + 10000)
        assert r.p_value < 0.001, (
            f"seed={seed}: uncorrected test should detect false positive, "
            f"got p={r.p_value:.4f}"
        )
        lo, hi = r.corrected_ci
        assert lo < null < hi, (
            f"seed={seed}: corrected CI ({lo:.3f}, {hi:.3f}) "
            f"should contain null={null}"
        )

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("call_fn,null,gen", _ALL_PARAMS)
    def test_corrected_estimate_closer_to_truth_than_llm(self, call_fn, null, gen, seed):
        """Quantitative check: the PPI correction must reduce estimation error.

        The true estimand equals null (no true effect).  The raw LLM estimate
        is far from null due to differential bias.  After PPI, the corrected
        estimate must be strictly closer to null than the raw LLM estimate.
        """
        rng = np.random.default_rng(seed)
        a, b, al, bl = gen(rng, n=300, mu_a=3.0, mu_b=3.0,
                           bias_a=2.0, bias_b=0.0, n_lab=80)
        r = call_fn(a, b, al, bl, n_boot=500, rng=seed + 10000)

        # Raw LLM estimate of the same estimand (before any correction)
        if null == 0.5:  # mannwhitney: P(X>Y) from raw LLM scores
            llm_est = float(np.mean(a[:, None] > b[None, :]))
        else:            # mean/median difference
            llm_est = float(np.mean(a - b)) if call_fn is _fn_wilcoxon else float(a.mean() - b.mean())
            if call_fn is _fn_wilcoxon:
                llm_est = float(np.median(a - b))

        error_corrected = abs(r.corrected_estimate - null)
        error_llm       = abs(llm_est - null)
        assert error_corrected < error_llm, (
            f"seed={seed}: corrected error={error_corrected:.3f} should be < "
            f"LLM error={error_llm:.3f} (corrected={r.corrected_estimate:.3f}, "
            f"llm={llm_est:.3f}, null={null})"
        )


# ─── Symmetric bias leaves estimate and rectifier unchanged ──────────────────

class TestSymmetricBias:
    """Equal LLM inflation on both groups → rectifier ≈ 0; estimate unchanged."""

    @pytest.mark.parametrize("call_fn,null,gen", _ALL_PARAMS)
    def test_rectifier_near_zero_for_symmetric_bias(self, call_fn, null, gen):
        rng = np.random.default_rng(50)
        a, b, al, bl = gen(rng, n=300, mu_a=4.0, mu_b=3.0,
                           bias_a=1.5, bias_b=1.5, n_lab=80)
        r = call_fn(a, b, al, bl, n_boot=300, rng=50)
        assert abs(r.rectifier) < 0.25, (
            f"{call_fn.__name__}: symmetric bias should give |rectifier| < 0.25, "
            f"got {r.rectifier:.3f}"
        )

    @pytest.mark.parametrize("call_fn,null,gen", _INDEP_PARAMS)
    def test_corrected_estimate_close_to_llm_for_symmetric_bias(self, call_fn, null, gen):
        """Corrected estimate should not move away from the raw LLM estimate when
        bias is symmetric (there is nothing to correct)."""
        rng = np.random.default_rng(51)
        a, b, al, bl = gen(rng, n=300, mu_a=4.0, mu_b=3.0,
                           bias_a=1.5, bias_b=1.5, n_lab=80)
        r_raw = call_fn(a, b, None, None, n_boot=200, rng=51) if False else None

        # Compute raw LLM estimate directly
        if null == 0.5:
            llm_est = float(np.mean(a[:, None] > b[None, :]))
        else:
            llm_est = float(a.mean() - b.mean())

        r = call_fn(a, b, al, bl, n_boot=300, rng=51)
        assert abs(r.corrected_estimate - llm_est) < 0.25, (
            f"{call_fn.__name__}: corrected should stay near LLM estimate under "
            f"symmetric bias; corrected={r.corrected_estimate:.3f}, llm={llm_est:.3f}"
        )


# ─── True effect is preserved (not over-corrected) ───────────────────────────

class TestTrueEffectPreserved:

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("call_fn,null,gen", _ALL_PARAMS)
    def test_large_true_effect_CI_excludes_null(self, call_fn, null, gen, seed):
        """A large genuine effect (Δ=2σ) remains detectable after PPI."""
        rng = np.random.default_rng(seed)
        a, b, al, bl = gen(rng, n=200, mu_a=5.0, mu_b=3.0, sigma=0.8,
                           bias_a=0.0, bias_b=0.0, n_lab=50)
        r = call_fn(a, b, al, bl, n_boot=300, rng=seed + 10000)
        lo, hi = r.corrected_ci
        # For all functions: lo > null means effect is entirely above the null
        assert lo > null, (
            f"seed={seed}: corrected CI ({lo:.3f}, {hi:.3f}) should be entirely "
            f"above null={null} for a 2σ true effect"
        )

    def test_ttest_ppi_corrects_underestimated_effect(self):
        """LLM inflates B → underestimates A−B; PPI shifts estimate toward true 0.5."""
        rng = np.random.default_rng(61)
        a, b, al, bl = _two_sample(rng, n=300, mu_a=3.5, mu_b=3.0,
                                   bias_a=0.0, bias_b=1.5, n_lab=80)
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=500, rng=61)
        llm_diff = float(a.mean() - b.mean())  # ≈ 3.5 − (3.0+1.5) = −1.0 (wrong sign)
        true_diff = 0.5
        assert abs(r.corrected_estimate - true_diff) < abs(llm_diff - true_diff), (
            f"corrected={r.corrected_estimate:.3f} should be closer to truth={true_diff} "
            f"than llm={llm_diff:.3f}"
        )


# ─── Rectifier magnitude scales with bias ────────────────────────────────────

class TestRectifierMagnitude:
    """Rectifier δ = f(Y_lab) − f(Ŷ_lab) scales linearly with bias.

    For ttest: rectifier ≈ −bias_a when bias_b=0 and mu_a=mu_b.
    Tested across 5 distinct bias magnitudes with a fixed data seed.
    """

    @pytest.mark.parametrize("bias_a", [0.5, 1.0, 1.5, 2.0, 2.5])
    def test_rectifier_approximately_equals_negative_bias(self, bias_a):
        """|rectifier − (−bias_a)| < 0.25 for each bias level (n=400, n_lab=100)."""
        rng = np.random.default_rng(12345)
        a, b, al, bl = _two_sample(rng, n=400, mu_a=3.0, mu_b=3.0,
                                   bias_a=bias_a, bias_b=0.0, n_lab=100)
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=300, rng=0)
        expected = -bias_a
        assert abs(r.rectifier - expected) < 0.25, (
            f"bias_a={bias_a}: expected rectifier ≈ {expected:.2f}, "
            f"got {r.rectifier:.3f}"
        )

    def test_rectifier_magnitude_is_monotone_with_bias(self):
        """Larger bias → larger |rectifier|.  Checked across 5 bias levels."""
        bias_values = [0.25, 0.75, 1.25, 1.75, 2.25]
        abs_rectifiers = []
        for bias_a in bias_values:
            rng = np.random.default_rng(99)   # same seed → only bias varies
            a, b, al, bl = _two_sample(rng, n=400, mu_a=3.0, mu_b=3.0,
                                       bias_a=bias_a, bias_b=0.0, n_lab=100)
            r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=300, rng=0)
            abs_rectifiers.append(abs(r.rectifier))
        for i in range(len(abs_rectifiers) - 1):
            assert abs_rectifiers[i] < abs_rectifiers[i + 1], (
                f"bias={bias_values[i]:.2f} gave |rectifier|={abs_rectifiers[i]:.3f} "
                f"but bias={bias_values[i+1]:.2f} gave {abs_rectifiers[i+1]:.3f} — "
                "expected strictly increasing"
            )

    def test_rectifier_near_zero_for_unbiased_llm(self):
        """No bias → |rectifier| < 0.12 (very low LLM noise)."""
        rng = np.random.default_rng(72)
        a, b, al, bl = _two_sample(rng, n=400, mu_a=3.5, mu_b=3.0,
                                   bias_a=0.0, bias_b=0.0, llm_noise=0.03,
                                   n_lab=100)
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=300, rng=72)
        assert abs(r.rectifier) < 0.12, (
            f"Expected |rectifier| < 0.12 for unbiased LLM, got {r.rectifier:.3f}"
        )

    def test_rectifier_sign_flips_with_bias_direction(self):
        """Bias on A: rectifier < 0.  Bias on B: rectifier > 0."""
        rng_a = np.random.default_rng(70)
        a, b, al, bl = _two_sample(rng_a, n=300, mu_a=3.0, mu_b=3.0,
                                   bias_a=2.0, bias_b=0.0, n_lab=80)
        r_a = ttest(a, b, a_lab=al, b_lab=bl, n_boot=200, rng=70)

        rng_b = np.random.default_rng(71)
        a, b, al, bl = _two_sample(rng_b, n=300, mu_a=3.0, mu_b=3.0,
                                   bias_a=0.0, bias_b=2.0, n_lab=80)
        r_b = ttest(a, b, a_lab=al, b_lab=bl, n_boot=200, rng=71)

        assert r_a.rectifier < -1.5, f"Expected rectifier<-1.5, got {r_a.rectifier:.3f}"
        assert r_b.rectifier > +1.5, f"Expected rectifier>+1.5, got {r_b.rectifier:.3f}"


# ─── CI width narrows with more labels ───────────────────────────────────────

class TestCIWidthMonotonicity:
    """More human labels → narrower PPI confidence interval.

    The labeled set size controls variance of the rectifier.  With the same
    a and b arrays, CI width should decrease as n_lab grows.
    """

    def test_ttest_ci_narrows_as_n_lab_increases(self):
        """CI width with 15 labeled items must be wider than with 60.

        Y_hat_unlab must be DISJOINT from the labeled positions (see
        evalstats.ppi.correct's docstring), so growing n_lab necessarily
        SHRINKS the primary (unlabeled) term's own sample size -- CI width
        is therefore NOT monotonically decreasing in n_lab across the full
        range (there's a real variance-optimal n_lab, and over-labeling
        past it makes the CI wider again, confirmed empirically: at
        llm_noise=0.10, going from n_lab=15 to n_lab=200 out of n=400
        WIDENS the CI, not narrows it -- the previous version of this test
        asserted the opposite because the pre-fix implementation always
        used the full N for the primary term regardless of n_lab, silently
        masking this tradeoff). This test instead stays within the small
        n_lab/N regime where the rectifier's variance reduction dominates
        the primary term's shrinkage, and uses a noisier judge
        (llm_noise=0.5) so that regime is wide enough to test robustly at a
        single seed (a small-noise judge makes the early-labeling benefit
        too close to bootstrap MC noise to assert deterministically).
        """
        rng_data = np.random.default_rng(999)
        n = 400
        truth_a = rng_data.normal(3.5, 1.0, n)
        truth_b = rng_data.normal(3.0, 1.0, n)
        a = truth_a + rng_data.normal(0, 0.5, n)
        b = truth_b + rng_data.normal(0, 0.5, n)

        def _ci_width(n_lab):
            rng_lab = np.random.default_rng(n_lab * 17)
            idx_a = rng_lab.choice(n, n_lab, replace=False)
            idx_b = rng_lab.choice(n, n_lab, replace=False)
            al = np.full(n, np.nan); al[idx_a] = truth_a[idx_a]
            bl = np.full(n, np.nan); bl[idx_b] = truth_b[idx_b]
            r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=2000, rng=42)
            lo, hi = r.corrected_ci
            return hi - lo

        w_small = _ci_width(15)
        w_large = _ci_width(60)
        assert w_small > w_large, (
            f"CI with 15 labels ({w_small:.4f}) should be wider than "
            f"with 60 labels ({w_large:.4f})"
        )

    def test_mannwhitney_ci_narrows_as_n_lab_increases(self):
        """Same endpoint check for Mann-Whitney P(X>Y) CI.

        Same small-n_lab/noisier-judge regime as
        test_ttest_ci_narrows_as_n_lab_increases -- see its docstring for
        why n_lab=15 vs. 200 (out of n=400) is not a valid comparison under
        the disjoint-sample requirement.
        """
        rng_data = np.random.default_rng(998)
        n = 400
        truth_a = rng_data.normal(3.5, 1.0, n)
        truth_b = rng_data.normal(3.0, 1.0, n)
        a = truth_a + rng_data.normal(0, 0.5, n)
        b = truth_b + rng_data.normal(0, 0.5, n)

        def _ci_width(n_lab):
            rng_lab = np.random.default_rng(n_lab * 19)
            idx_a = rng_lab.choice(n, n_lab, replace=False)
            idx_b = rng_lab.choice(n, n_lab, replace=False)
            al = np.full(n, np.nan); al[idx_a] = truth_a[idx_a]
            bl = np.full(n, np.nan); bl[idx_b] = truth_b[idx_b]
            r = mannwhitney(a, b, x_lab=al, y_lab=bl, n_boot=2000, rng=42)
            lo, hi = r.corrected_ci
            return hi - lo

        w_small = _ci_width(15)
        w_large = _ci_width(60)
        assert w_small > w_large, (
            f"CI with 15 labels ({w_small:.4f}) should be wider than "
            f"with 60 labels ({w_large:.4f})"
        )


# ─── Non-Gaussian distributions ──────────────────────────────────────────────

class TestNonGaussianDistributions:
    """PPI correction is distribution-agnostic; these tests confirm it works
    for binary (0/1) and Likert (1–5) data — the most common LLM eval formats."""

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_binary_differential_bias_corrects_false_positive_ttest(self, seed):
        """Binary scores: LLM systematically over-reports 1 for group A.
        True proportion: p_A = p_B = 0.5.  LLM: p_A = 0.85, p_B = 0.50.
        n=500 and n_lab=120 give stable results across seeds for 0/1 data."""
        rng = np.random.default_rng(seed)
        a, b, al, bl = _two_sample_binary(
            rng, n=500, p_a=0.5, p_b=0.5,
            llm_p_a=0.85, llm_p_b=0.50, n_lab=120,
        )
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=500, rng=seed + 10000)
        assert r.p_value < 0.001, f"seed={seed}: should detect false positive"
        lo, hi = r.corrected_ci
        assert lo < 0 < hi, (
            f"seed={seed}: binary corrected CI ({lo:.3f},{hi:.3f}) "
            f"should contain 0"
        )

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_binary_differential_bias_corrected_estimate_closer_to_truth(self, seed):
        rng = np.random.default_rng(seed)
        a, b, al, bl = _two_sample_binary(
            rng, n=500, p_a=0.5, p_b=0.5,
            llm_p_a=0.85, llm_p_b=0.50, n_lab=120,
        )
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=500, rng=seed + 10000)
        true_diff = 0.0
        llm_diff = float(a.mean() - b.mean())
        assert abs(r.corrected_estimate - true_diff) < abs(llm_diff - true_diff), (
            f"seed={seed}: binary corrected={r.corrected_estimate:.3f} should be "
            f"closer to 0 than llm={llm_diff:.3f}"
        )

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_likert_differential_bias_corrects_false_positive_ttest(self, seed):
        """Likert 1–5: LLM rates group A 0.8 points higher than truth (no true diff)."""
        rng = np.random.default_rng(seed)
        a, b, al, bl = _two_sample_likert(
            rng, n=350, mu_a=3.0, mu_b=3.0,
            bias_a=0.8, bias_b=0.0, n_lab=90,
        )
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=300, rng=seed + 10000)
        assert r.p_value < 0.05, f"seed={seed}: should detect false positive"
        lo, hi = r.corrected_ci
        assert lo < 0 < hi, (
            f"seed={seed}: Likert corrected CI ({lo:.3f},{hi:.3f}) "
            f"should contain 0"
        )

    def test_binary_unbiased_llm_rectifier_near_zero(self):
        """Binary data, unbiased LLM (same Bernoulli params for truth and LLM)
        → rectifier is sampling noise centred at 0.  Threshold of 0.20 covers
        ±3 SE for binary data with n_lab=100 per group (SE ≈ 0.07)."""
        rng = np.random.default_rng(200)
        a, b, al, bl = _two_sample_binary(
            rng, n=400, p_a=0.65, p_b=0.40,
            llm_p_a=0.65, llm_p_b=0.40, n_lab=100,
        )
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=300, rng=200)
        assert abs(r.rectifier) < 0.20


# ─── Paired test specifics ────────────────────────────────────────────────────

class TestPairedTests:

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_paired_ttest_false_positive_suppressed(self, seed):
        """Paired: LLM inflates a by 2σ, no true diff → CI contains 0."""
        rng = np.random.default_rng(seed)
        a, b, al, bl = _paired(rng, n=250, mu_a=3.0, mu_b=3.0,
                               bias_a=2.0, bias_b=0.0, n_lab=70)
        r = ttest(a, b, a_lab=al, b_lab=bl, paired=True, n_boot=300, rng=seed + 10000)
        assert r.p_value < 0.001, f"seed={seed}: uncorrected should be significant"
        lo, hi = r.corrected_ci
        assert lo < 0 < hi, f"seed={seed}: paired corrected CI ({lo:.3f},{hi:.3f}) should contain 0"

    def test_paired_ttest_large_effect_detected(self):
        rng = np.random.default_rng(80)
        a, b, al, bl = _paired(rng, n=200, mu_a=4.0, mu_b=3.0, n_lab=50)
        r = ttest(a, b, a_lab=al, b_lab=bl, paired=True, n_boot=500, rng=80)
        lo, hi = r.corrected_ci
        assert lo > 0, "Paired ttest corrected CI should be above 0 for Δ=1.0"

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_wilcoxon_false_positive_suppressed(self, seed):
        """Wilcoxon: LLM inflates a by 2σ, no true diff → PPI corrects toward 0.

        The corrected estimate should be near 0 (well under 0.5 vs raw signal ~2.0).
        We check the point estimate rather than CI containment because the
        Wilcoxon PPI uses a median estimand, giving a calibrated ~4% Type I rate;
        checking CI containment across 5 seeds would fail ~20% of test runs.
        """
        rng = np.random.default_rng(seed)
        a, b, al, bl = _paired(rng, n=250, mu_a=3.0, mu_b=3.0,
                               bias_a=2.0, bias_b=0.0, n_lab=70)
        r = wilcoxon(a, b, x_lab=al, y_lab=bl, n_boot=300, rng=seed + 10000)
        assert r.p_value < 0.001, f"seed={seed}: uncorrected should be significant"
        assert abs(r.corrected_estimate) < 0.5, (
            f"seed={seed}: wilcoxon corrected estimate {r.corrected_estimate:.3f} "
            f"should be near 0 (LLM signal was ~2.0)"
        )

    def test_wilcoxon_large_shift_detected(self):
        rng = np.random.default_rng(82)
        a, b, al, bl = _paired(rng, n=200, mu_a=4.5, mu_b=3.0, n_lab=50)
        r = wilcoxon(a, b, x_lab=al, y_lab=bl, n_boot=500, rng=82)
        lo, hi = r.corrected_ci
        assert lo > 0, "Wilcoxon corrected CI should exclude 0 for large true shift"
        assert abs(r.corrected_estimate - 1.5) < 0.5

    def test_wilcoxon_raises_when_no_overlap_in_labeled_positions(self):
        """y_lab all NaN → no position has both x and y labeled → ValueError."""
        rng = np.random.default_rng(83)
        a, b, al, _ = _paired(rng, n=100, n_lab=30)
        with pytest.raises(ValueError, match="At least 15 overlapping human-labeled positions"):
            wilcoxon(a, b, x_lab=al, y_lab=np.full(len(b), np.nan), n_boot=100)

    def test_ttest_paired_unequal_lengths_raises(self):
        rng = np.random.default_rng(84)
        a = rng.normal(0, 1, 80)
        b = rng.normal(0, 1, 90)
        with pytest.raises(ValueError, match="paired"):
            ttest(a, b, paired=True)

    def test_wilcoxon_unequal_lengths_raises(self):
        rng = np.random.default_rng(85)
        a = rng.normal(0, 1, 80)
        b = rng.normal(0, 1, 90)
        with pytest.raises(ValueError):
            wilcoxon(a, b)


# ─── Mann-Whitney specifics ───────────────────────────────────────────────────

class TestMannWhitneySpecific:

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_corrected_p_xy_near_half_when_groups_equal(self, seed):
        """Unbiased LLM, equal distributions → corrected P(X>Y) near 0.5."""
        rng = np.random.default_rng(seed)
        a, b, al, bl = _two_sample(rng, n=300, mu_a=3.0, mu_b=3.0,
                                   bias_a=0.0, bias_b=0.0, n_lab=80)
        r = mannwhitney(a, b, x_lab=al, y_lab=bl, n_boot=400, rng=seed + 10000)
        assert abs(r.corrected_estimate - 0.5) < 0.12, (
            f"seed={seed}: P(X>Y)={r.corrected_estimate:.3f} should be near 0.5"
        )

    def test_corrected_p_xy_high_when_x_dominates(self):
        """X ≫ Y → corrected P(X>Y) > 0.85."""
        rng = np.random.default_rng(91)
        a, b, al, bl = _two_sample(rng, n=200, mu_a=5.0, mu_b=3.0, sigma=0.8, n_lab=50)
        r = mannwhitney(a, b, x_lab=al, y_lab=bl, n_boot=500, rng=91)
        assert r.corrected_estimate > 0.85

    def test_estimand_field_present(self):
        rng = np.random.default_rng(92)
        a, b, al, bl = _two_sample(rng, n=100)
        r = mannwhitney(a, b, x_lab=al, y_lab=bl, n_boot=100, rng=92)
        assert "estimand" in r.extra
        assert "P(X > Y)" in r.extra["estimand"]


# ─── Input handling ───────────────────────────────────────────────────────────

class TestInputHandling:

    def test_list_inputs_accepted(self):
        rng = np.random.default_rng(100)
        a = rng.normal(0, 1, 60).tolist()
        b = rng.normal(0, 1, 60).tolist()
        al = [np.nan if i % 3 else a[i] for i in range(60)]
        bl = [np.nan if i % 3 else b[i] for i in range(60)]
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=100, rng=100)
        assert r.corrected_estimate is not None

    def test_one_lab_none_defaults_to_all_nan(self):
        """Only a_lab set; b_lab=None treated as all unlabeled for group B."""
        rng = np.random.default_rng(101)
        a, b, al, _ = _two_sample(rng, n=100, n_lab=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = ttest(a, b, a_lab=al, b_lab=None, n_boot=200, rng=101)
        assert r.corrected_estimate is not None

    def test_all_items_labeled_raises_informatively(self):
        """All items labeled -> PPI has no unlabeled pool to extrapolate the
        correction to (Y_hat_unlab must be DISJOINT from the labeled
        positions -- see evalstats.ppi.correct's docstring), so this now
        raises a clear, actionable error instead of silently reusing the
        labeled data as its own "unlabeled" set."""
        rng = np.random.default_rng(102)
        n = 80
        truth_a = rng.normal(3.5, 1.0, n)
        truth_b = rng.normal(3.0, 1.0, n)
        a = truth_a + rng.normal(0, 0.1, n)
        b = truth_b + rng.normal(0, 0.1, n)
        with pytest.raises(ValueError, match="unlabeled pool"):
            ttest(a, b, a_lab=truth_a, b_lab=truth_b, n_boot=200, rng=102)

    def test_both_labs_none_gives_uncorrected_result(self):
        rng = np.random.default_rng(103)
        a = rng.normal(0, 1, 60)
        b = rng.normal(0, 1, 60)
        r = ttest(a, b, a_lab=None, b_lab=None)
        assert r.corrected_estimate is None

    def test_both_labs_all_nan_raises(self):
        """All NaN in both lab arrays → ValueError before PPI (alignment step)."""
        rng = np.random.default_rng(104)
        a = rng.normal(0, 1, 60)
        b = rng.normal(0, 1, 60)
        with pytest.raises(ValueError, match="At least 15 human labels"):
            ttest(a, b, a_lab=np.full(60, np.nan), b_lab=np.full(60, np.nan),
                  n_boot=100)

    def test_raises_when_fewer_than_15_total_labels_independent(self):
        rng = np.random.default_rng(105)
        a, b, al, bl = _two_sample(rng, n=100, n_lab=7)
        # Keep only 6+6 labels total = 12 (<15)
        al[np.where(~np.isnan(al))[0][6:]] = np.nan
        bl[np.where(~np.isnan(bl))[0][6:]] = np.nan
        with pytest.raises(ValueError, match="At least 15 human labels"):
            ttest(a, b, a_lab=al, b_lab=bl, n_boot=100, rng=105)

    def test_warns_when_fewer_than_30_total_labels_independent(self):
        rng = np.random.default_rng(106)
        a, b, al, bl = _two_sample(rng, n=100, n_lab=12)
        # Keep 12+12 labels total = 24 (>=15 and <30)
        al[np.where(~np.isnan(al))[0][12:]] = np.nan
        bl[np.where(~np.isnan(bl))[0][12:]] = np.nan
        with pytest.warns(UserWarning, match="undercover below 30 labels"):
            ttest(a, b, a_lab=al, b_lab=bl, n_boot=100, rng=106)


# ─── Extra rigor checks (calibration, coherence, edge stress) ───────────────

class TestPPIExtraRigor:

    def test_ttest_corrected_type1_error_not_anti_conservative(self):
        """Across many null datasets, corrected false-positive rate stays bounded."""
        alpha = 0.05
        pvals = []
        for seed in range(300, 360):
            rng = np.random.default_rng(seed)
            a, b, al, bl = _two_sample(
                rng,
                n=180,
                mu_a=3.0,
                mu_b=3.0,
                bias_a=0.0,
                bias_b=0.0,
                n_lab=45,
            )
            r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=300, rng=seed)
            pvals.append(r.corrected_p_value)

        fpr = float(np.mean(np.asarray(pvals) < alpha))
        assert fpr <= 0.15, f"Corrected Type-I rate too high: {fpr:.3f}"

    def test_ttest_corrected_ci_coverage_under_null(self):
        """95% corrected CIs under null should cover 0 most of the time."""
        covered = 0
        n_trials = 40
        for seed in range(400, 400 + n_trials):
            rng = np.random.default_rng(seed)
            a, b, al, bl = _two_sample(
                rng,
                n=200,
                mu_a=3.0,
                mu_b=3.0,
                bias_a=0.0,
                bias_b=0.0,
                n_lab=50,
            )
            r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=300, rng=seed)
            lo, hi = r.corrected_ci
            covered += int(lo <= 0.0 <= hi)

        coverage = covered / n_trials
        assert coverage >= 0.80, f"Null CI coverage too low: {coverage:.3f}"

    @pytest.mark.parametrize("call_fn,null,gen", _ALL_PARAMS)
    def test_corrected_p_value_consistent_with_corrected_ci(self, call_fn, null, gen):
        """Two-sided corrected p-value and corrected CI should agree on rejection."""
        alpha = 0.05
        rng = np.random.default_rng(222)
        a, b, al, bl = gen(
            rng,
            n=240,
            mu_a=4.0,
            mu_b=3.2,
            sigma=0.9,
            bias_a=0.8,
            bias_b=0.0,
            n_lab=70,
        )
        r = call_fn(a, b, al, bl, alpha=alpha, n_boot=500, rng=222)
        lo, hi = r.corrected_ci

        ci_rejects = (hi < null) or (lo > null)
        p_rejects = r.corrected_p_value < alpha
        assert p_rejects == ci_rejects, (
            f"{call_fn.__name__}: corrected p/CI disagreement "
            f"(p={r.corrected_p_value:.4f}, ci=({lo:.4f}, {hi:.4f}), null={null})"
        )

    def test_rectifier_stable_across_label_masks(self):
        """Rectifier should not depend strongly on which items were labeled."""
        rng = np.random.default_rng(230)
        n = 380
        truth_a = rng.normal(3.0, 1.0, n)
        truth_b = rng.normal(3.0, 1.0, n)
        a = truth_a + 1.6 + rng.normal(0, 0.1, n)
        b = truth_b + 0.2 + rng.normal(0, 0.1, n)

        rectifiers = []
        for mask_seed in range(10):
            rng_mask = np.random.default_rng(1000 + mask_seed)
            idx_a = rng_mask.choice(n, 70, replace=False)
            idx_b = rng_mask.choice(n, 70, replace=False)
            al = np.full(n, np.nan); al[idx_a] = truth_a[idx_a]
            bl = np.full(n, np.nan); bl[idx_b] = truth_b[idx_b]
            r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=250, rng=mask_seed)
            rectifiers.append(r.rectifier)

        assert float(np.std(rectifiers)) < 0.22, (
            f"Rectifier varies too much across label masks: std={np.std(rectifiers):.3f}"
        )

    @pytest.mark.parametrize("call_fn,null", [(_fn_ttest, 0.0), (_fn_mw, 0.5)])
    def test_unbalanced_group_sizes_false_positive_corrected(self, call_fn, null):
        """Differential bias suppression still works with strongly unequal group sizes."""
        rng = np.random.default_rng(240)
        a, b, al, bl = _two_sample_unbalanced(
            rng,
            n_a=500,
            n_b=120,
            mu_a=3.0,
            mu_b=3.0,
            sigma=1.0,
            bias_a=2.0,
            bias_b=0.0,
            n_lab_a=90,
            n_lab_b=25,
        )
        r = call_fn(a, b, al, bl, n_boot=500, rng=240)
        lo, hi = r.corrected_ci
        assert r.p_value < 0.001
        assert lo < null < hi

    @pytest.mark.parametrize("call_fn,null,gen", _ALL_PARAMS)
    def test_aggregate_mse_reduction_across_seeds(self, call_fn, null, gen):
        """Across seeds, corrected estimates should reduce MSE vs raw LLM estimates."""
        llm_sq_errors = []
        corrected_sq_errors = []

        for seed in _SEEDS:
            rng = np.random.default_rng(seed)
            a, b, al, bl = gen(
                rng,
                n=300,
                mu_a=3.0,
                mu_b=3.0,
                bias_a=2.0,
                bias_b=0.0,
                n_lab=80,
            )
            r = call_fn(a, b, al, bl, n_boot=500, rng=seed + 10000)

            llm_est = _llm_estimand(call_fn, null, a, b)
            llm_sq_errors.append((llm_est - null) ** 2)
            corrected_sq_errors.append((r.corrected_estimate - null) ** 2)

        assert np.mean(corrected_sq_errors) < np.mean(llm_sq_errors), (
            f"{call_fn.__name__}: corrected MSE {np.mean(corrected_sq_errors):.4f} "
            f"should be < LLM MSE {np.mean(llm_sq_errors):.4f}"
        )

    def test_ttest_oracle_near_fully_labeled_close_to_human_estimate(self):
        """With most (not literally all) items labeled, corrected estimate
        should land reasonably close to the human-data estimand.

        Y_hat_unlab must be DISJOINT from the labeled positions (see
        evalstats.ppi.correct's docstring) -- 100% labeling leaves no
        unlabeled pool at all (see test_all_items_labeled_raises_
        informatively), so this uses n_lab = n - 40 instead: a large
        labeled majority with a small but genuine unlabeled remainder. The
        two estimates are no longer expected to match to tight tolerance --
        they're independent unbiased estimates of the same population
        quantity, drawn from different-sized slices of the data -- so the
        tolerance here is wide enough to comfortably hold given the primary
        term's now-smaller (40-item) unlabeled sample.
        """
        rng = np.random.default_rng(250)
        n = 220
        n_lab = n - 40
        truth_a = rng.normal(3.8, 0.9, n)
        truth_b = rng.normal(3.1, 0.9, n)
        a = truth_a + 0.7 + rng.normal(0, 0.08, n)
        b = truth_b - 0.4 + rng.normal(0, 0.08, n)
        al = np.full(n, np.nan)
        idx_a = rng.choice(n, n_lab, replace=False)
        al[idx_a] = truth_a[idx_a]
        bl = np.full(n, np.nan)
        idx_b = rng.choice(n, n_lab, replace=False)
        bl[idx_b] = truth_b[idx_b]

        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=500, rng=250)
        human_diff = float(truth_a.mean() - truth_b.mean())
        assert r.corrected_estimate == pytest.approx(human_diff, abs=0.25)


class TestBootstrapPValueStability:
    """Increasing bootstrap draws should not make corrected p-values materially smaller."""

    @staticmethod
    def _assert_not_meaningfully_smaller(p_1000, p_5000, p_10000, max_drop=0.05):
        """Allow small Monte Carlo jitter but reject materially smaller p-values."""
        assert p_5000 >= p_1000 - max_drop, (
            f"n_boot=5000 produced much smaller p ({p_5000:.4f}) than n_boot=1000 ({p_1000:.4f})"
        )
        assert p_10000 >= p_5000 - max_drop, (
            f"n_boot=10000 produced much smaller p ({p_10000:.4f}) than n_boot=5000 ({p_5000:.4f})"
        )
        assert p_10000 >= p_1000 - max_drop, (
            f"n_boot=10000 produced much smaller p ({p_10000:.4f}) than n_boot=1000 ({p_1000:.4f})"
        )

    def test_ttest_p_value_stable_as_n_boot_increases(self):
        rng = np.random.default_rng(261)
        a, b, al, bl = _two_sample(
            rng,
            n=300,
            mu_a=3.0,
            mu_b=3.0,
            bias_a=1.8,
            bias_b=0.0,
            n_lab=90,
        )
        p_1000 = ttest(a, b, a_lab=al, b_lab=bl, n_boot=1000, rng=1261).corrected_p_value
        p_5000 = ttest(a, b, a_lab=al, b_lab=bl, n_boot=5000, rng=1261).corrected_p_value
        p_10000 = ttest(a, b, a_lab=al, b_lab=bl, n_boot=10000, rng=1261).corrected_p_value

        self._assert_not_meaningfully_smaller(p_1000, p_5000, p_10000)

    def test_mannwhitney_p_value_stable_as_n_boot_increases(self):
        rng = np.random.default_rng(262)
        a, b, al, bl = _two_sample(
            rng,
            n=280,
            mu_a=3.0,
            mu_b=3.0,
            bias_a=1.6,
            bias_b=0.0,
            n_lab=85,
        )
        p_1000 = mannwhitney(a, b, x_lab=al, y_lab=bl, n_boot=1000, rng=1262).corrected_p_value
        p_5000 = mannwhitney(a, b, x_lab=al, y_lab=bl, n_boot=5000, rng=1262).corrected_p_value
        p_10000 = mannwhitney(a, b, x_lab=al, y_lab=bl, n_boot=10000, rng=1262).corrected_p_value

        self._assert_not_meaningfully_smaller(p_1000, p_5000, p_10000)

    def test_wilcoxon_p_value_stable_as_n_boot_increases(self):
        rng = np.random.default_rng(263)
        a, b, al, bl = _paired(
            rng,
            n=260,
            mu_a=3.0,
            mu_b=3.0,
            bias_a=1.6,
            bias_b=0.0,
            n_lab=80,
        )
        p_1000 = wilcoxon(a, b, x_lab=al, y_lab=bl, n_boot=1000, rng=1263).corrected_p_value
        p_5000 = wilcoxon(a, b, x_lab=al, y_lab=bl, n_boot=5000, rng=1263).corrected_p_value
        p_10000 = wilcoxon(a, b, x_lab=al, y_lab=bl, n_boot=10000, rng=1263).corrected_p_value

        self._assert_not_meaningfully_smaller(p_1000, p_5000, p_10000)

    def test_ttest_ci_narrows_across_multiple_label_levels(self):
        """CI width should shrink across increasing label budgets (allow tiny jitter).

        Same small-n_lab/noisier-judge regime as
        test_ttest_ci_narrows_as_n_lab_increases -- see its docstring. The
        original [20, 50, 100, 160, 230] range (out of n=450, up to 51%
        labeled) went well past the variance-optimal n_lab under the
        disjoint-sample requirement, so this uses [15, 25, 40, 60] instead
        (all comfortably in the small-n_lab/N regime).
        """
        rng_data = np.random.default_rng(260)
        n = 450
        truth_a = rng_data.normal(3.4, 1.0, n)
        truth_b = rng_data.normal(3.0, 1.0, n)
        a = truth_a + rng_data.normal(0, 0.5, n)
        b = truth_b + rng_data.normal(0, 0.5, n)

        widths = []
        for n_lab in [15, 25, 40, 60]:
            rng_lab = np.random.default_rng(500 + n_lab)
            idx_a = rng_lab.choice(n, n_lab, replace=False)
            idx_b = rng_lab.choice(n, n_lab, replace=False)
            al = np.full(n, np.nan); al[idx_a] = truth_a[idx_a]
            bl = np.full(n, np.nan); bl[idx_b] = truth_b[idx_b]
            r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=1500, rng=260 + n_lab)
            lo, hi = r.corrected_ci
            widths.append(hi - lo)

        for i in range(len(widths) - 1):
            assert widths[i + 1] <= widths[i] + 0.02, (
                f"Expected non-increasing widths, got {widths}"
            )

    @pytest.mark.parametrize("call_fn,null", [(_fn_ttest, 0.0), (_fn_mw, 0.5), (_fn_ttest_paired, 0.0), (_fn_wilcoxon, 0.0)])
    def test_tiny_labeled_set_raises_value_error(self, call_fn, null):
        """Policy check: fewer than 15 labels must fail fast."""
        rng = np.random.default_rng(270)

        if call_fn in (_fn_ttest_paired, _fn_wilcoxon):
            a, b, al, bl = _paired(rng, n=120, mu_a=3.0, mu_b=3.0, bias_a=1.0, bias_b=0.0, n_lab=2)
        else:
            a, b, al, bl = _two_sample(rng, n=120, mu_a=3.0, mu_b=3.0, bias_a=1.0, bias_b=0.0, n_lab=1)

        with pytest.raises(ValueError, match="At least 15"):
            call_fn(a, b, al, bl, n_boot=120, rng=270)

    @pytest.mark.parametrize("call_fn,null", [(_fn_ttest, 0.0), (_fn_mw, 0.5), (_fn_ttest_paired, 0.0), (_fn_wilcoxon, 0.0)])
    def test_15_to_29_labels_warns_not_errors(self, call_fn, null):
        """Policy check: 15..29 effective labels warn but still run."""
        rng = np.random.default_rng(271)

        if call_fn in (_fn_ttest_paired, _fn_wilcoxon):
            # paired effective labels are overlaps; here n_lab=20 overlaps
            a, b, al, bl = _paired(rng, n=120, mu_a=3.0, mu_b=3.0, bias_a=1.0, bias_b=0.0, n_lab=20)
        else:
            # independent effective labels are total labels across both groups; n_lab=10 gives 20 total
            a, b, al, bl = _two_sample(rng, n=120, mu_a=3.0, mu_b=3.0, bias_a=1.0, bias_b=0.0, n_lab=10)

        with pytest.warns(UserWarning, match="undercover below 30 labels"):
            r = call_fn(a, b, al, bl, n_boot=120, rng=271)
        assert r.corrected_estimate is not None


# ─── anova_oneway ─────────────────────────────────────────────────────────────

_ANOVA_SEEDS = [110, 220, 330, 440, 550]


def _multigroup(rng, k=3, n=200, mus=None, sigma=1.0, biases=None, n_lab=50, llm_noise=0.10):
    """Independent k-group data. LLM = truth + bias + iid noise."""
    if mus is None:
        mus = [3.0] * k
    if biases is None:
        biases = [0.0] * k
    truths = [rng.normal(mu, sigma, n) for mu in mus]
    groups = [truths[i] + biases[i] + rng.normal(0, llm_noise, n) for i in range(k)]
    groups_lab = [np.full(n, np.nan) for _ in range(k)]
    for i in range(k):
        idx = rng.choice(n, n_lab, replace=False)
        groups_lab[i][idx] = truths[i][idx]
    return groups, groups_lab


def _multigroup_repeated(rng, k=3, n=200, mus=None, sigma=1.0, biases=None, n_lab=50, llm_noise=0.10):
    """Repeated-measures k-condition data. Same subjects labeled in all conditions."""
    if mus is None:
        mus = [3.0] * k
    if biases is None:
        biases = [0.0] * k
    subject_fx = rng.normal(0, 0.5, n)
    truths = [mus[i] + subject_fx + rng.normal(0, sigma, n) for i in range(k)]
    groups = [truths[i] + biases[i] + rng.normal(0, llm_noise, n) for i in range(k)]
    idx = rng.choice(n, n_lab, replace=False)
    groups_lab = [np.full(n, np.nan) for _ in range(k)]
    for i in range(k):
        groups_lab[i][idx] = truths[i][idx]
    return groups, groups_lab


class TestAnovaOnewayBaseline:
    """No-label tests: statistics match scipy/statsmodels; PPI fields all None."""

    def test_independent_matches_scipy(self):
        rng = np.random.default_rng(500)
        groups, _ = _multigroup(rng, k=3, n=120)
        r = anova_oneway(*groups)
        ref = scipy_stats.f_oneway(*groups)
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.p_value == pytest.approx(ref.pvalue)

    def test_repeated_matches_statsmodels(self):
        import pandas as pd
        pytest.importorskip("statsmodels")
        from statsmodels.stats.anova import AnovaRM
        rng = np.random.default_rng(501)
        groups, _ = _multigroup_repeated(rng, k=3, n=80)
        r = anova_oneway(*groups, repeated=True)
        n_subjects, k = len(groups[0]), len(groups)
        stacked = np.column_stack(groups)
        df_long = pd.DataFrame({
            "subject": np.repeat(np.arange(n_subjects), k),
            "condition": np.tile(np.arange(k), n_subjects),
            "score": stacked.reshape(-1),
        })
        rm = AnovaRM(df_long, depvar="score", subject="subject", within=["condition"]).fit()
        row = rm.anova_table.iloc[0]
        assert r.statistic == pytest.approx(float(row["F Value"]), rel=1e-5)
        assert r.p_value == pytest.approx(float(row["Pr > F"]), rel=1e-5)

    def test_independent_invariant_to_group_permutation(self):
        rng = np.random.default_rng(5011)
        groups, _ = _multigroup(rng, k=4, n=100)
        r1 = anova_oneway(*groups)
        r2 = anova_oneway(*groups[::-1])
        assert r1.statistic == pytest.approx(r2.statistic)
        assert r1.p_value == pytest.approx(r2.p_value)

    def test_independent_invariant_to_shift_and_positive_scale(self):
        rng = np.random.default_rng(5012)
        groups, _ = _multigroup(rng, k=3, n=120)
        r = anova_oneway(*groups)
        groups_affine = [2.75 * g + 11.0 for g in groups]
        r_affine = anova_oneway(*groups_affine)
        assert r.statistic == pytest.approx(r_affine.statistic)
        assert r.p_value == pytest.approx(r_affine.p_value)

    def test_independent_unbalanced_matches_scipy(self):
        rng = np.random.default_rng(5013)
        g1 = rng.normal(0.0, 1.0, 60)
        g2 = rng.normal(0.3, 1.0, 140)
        g3 = rng.normal(-0.2, 1.0, 350)
        r = anova_oneway(g1, g2, g3)
        ref = scipy_stats.f_oneway(g1, g2, g3)
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.p_value == pytest.approx(ref.pvalue)

    def test_ppi_fields_none_without_labels_independent(self):
        rng = np.random.default_rng(502)
        groups, _ = _multigroup(rng, k=3, n=80)
        r = anova_oneway(*groups)
        assert isinstance(r, TestResult)
        assert r.corrected_estimate is None
        assert r.corrected_ci is None
        assert r.corrected_p_value is None
        assert r.rectifier is None
        assert r.n_labeled is None
        assert r.n_total is None

    def test_ppi_fields_none_without_labels_repeated(self):
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(503)
        groups, _ = _multigroup_repeated(rng, k=3, n=80)
        r = anova_oneway(*groups, repeated=True)
        assert r.corrected_estimate is None
        assert r.corrected_ci is None
        assert r.corrected_p_value is None
        assert r.rectifier is None
        assert r.n_labeled is None
        assert r.n_total is None

    def test_test_name_contains_anova(self):
        rng = np.random.default_rng(504)
        groups, _ = _multigroup(rng, k=3, n=60)
        assert "anova" in anova_oneway(*groups).test_name.lower()

    def test_requires_at_least_two_groups(self):
        rng = np.random.default_rng(505)
        g = rng.normal(0, 1, 50)
        with pytest.raises(ValueError, match="at least two groups"):
            anova_oneway(g)

    def test_repeated_raises_for_unequal_lengths(self):
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(506)
        g1 = rng.normal(0, 1, 80)
        g2 = rng.normal(0, 1, 90)
        with pytest.raises(ValueError, match="equal-length"):
            anova_oneway(g1, g2, repeated=True)

    def test_extra_dict_keys_independent(self):
        rng = np.random.default_rng(507)
        groups, _ = _multigroup(rng, k=4, n=80)
        r = anova_oneway(*groups)
        for key in ("_test", "k_groups", "n_total_obs", "group_sizes", "group_means", "df1", "df2", "eta_sq"):
            assert key in r.extra, f"Missing extra key: {key!r}"
        assert r.extra["k_groups"] == 4
        assert r.extra["df1"] == pytest.approx(3.0)   # k-1
        assert 0.0 <= r.extra["eta_sq"] <= 1.0

    def test_extra_dict_keys_repeated(self):
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(508)
        groups, _ = _multigroup_repeated(rng, k=3, n=60)
        r = anova_oneway(*groups, repeated=True)
        for key in ("_test", "k_groups", "n_subjects", "n_total_obs", "group_means", "df1", "df2", "eta_sq"):
            assert key in r.extra, f"Missing extra key: {key!r}"
        assert r.extra["repeated"] is True
        assert r.extra["k_groups"] == 3
        assert r.extra["n_subjects"] == 60
        assert 0.0 <= r.extra["eta_sq"] <= 1.0


class TestAnovaOnewayPPIFieldStructure:
    """PPI-corrected anova_oneway: field structure, ordering, and metadata."""

    def test_fields_populated_independent(self):
        rng = np.random.default_rng(510)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = anova_oneway(*groups, groups_lab=groups_lab, n_boot=200, rng=510)
        assert r.corrected_estimate is not None
        assert r.corrected_ci is not None
        assert r.corrected_p_value is not None
        assert r.rectifier is not None
        assert r.n_labeled is not None
        assert r.n_total is not None

    def test_fields_populated_repeated(self):
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(511)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=200, n_lab=50)
        r = anova_oneway(*groups, repeated=True, groups_lab=groups_lab, n_boot=200, rng=511)
        assert r.corrected_estimate is not None
        assert r.corrected_ci is not None
        assert r.corrected_p_value is not None
        assert r.n_labeled is not None
        assert r.n_total is not None

    def test_ci_is_ordered_independent(self):
        rng = np.random.default_rng(512)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = anova_oneway(*groups, groups_lab=groups_lab, n_boot=200, rng=512)
        lo, hi = r.corrected_ci
        assert lo <= hi

    def test_corrected_p_value_in_0_1(self):
        rng = np.random.default_rng(513)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = anova_oneway(*groups, groups_lab=groups_lab, n_boot=300, rng=513)
        assert 0.0 <= r.corrected_p_value <= 1.0

    def test_n_labeled_and_n_total_match_input_independent(self):
        """3 groups × 200 items × 50 labeled each → n_labeled=150, n_total=600."""
        rng = np.random.default_rng(514)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = anova_oneway(*groups, groups_lab=groups_lab, n_boot=200, rng=514)
        assert r.n_labeled == 150
        assert r.n_total == 600

    def test_n_labeled_and_n_total_match_input_repeated(self):
        """3 conditions × 200 subjects × 50 labeled subjects → n_labeled=150, n_total=600."""
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(5141)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=200, n_lab=50)
        r = anova_oneway(*groups, repeated=True, groups_lab=groups_lab, n_boot=200, rng=5141)
        assert r.n_labeled == 150
        assert r.n_total == 600

    def test_reproducibility_with_rng_seed(self):
        rng = np.random.default_rng(515)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        kw = dict(groups_lab=groups_lab, n_boot=500, rng=42)
        r1 = anova_oneway(*groups, **kw)
        r2 = anova_oneway(*groups, **kw)
        assert r1.corrected_estimate == r2.corrected_estimate
        assert r1.corrected_ci == r2.corrected_ci

    def test_alpha_propagated(self):
        rng = np.random.default_rng(516)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = anova_oneway(*groups, groups_lab=groups_lab, alpha=0.10, n_boot=200, rng=516)
        assert r.alpha == 0.10

    def test_corrected_estimate_equals_llm_plus_rectifier_independent(self):
        """Y_hat_unlab must be DISJOINT from the labeled positions (see
        evalstats.ppi.correct's docstring), so llm_between is computed on
        each group's unlabeled-only complement, not the full group."""
        rng = np.random.default_rng(517)
        groups, groups_lab = _multigroup(
            rng,
            k=3,
            n=220,
            mus=[3.0, 3.0, 3.0],
            biases=[1.5, 0.0, 0.0],
            n_lab=60,
        )
        r = anova_oneway(*groups, groups_lab=groups_lab, n_boot=200, rng=517)

        masks = [~np.isnan(g_lab) for g_lab in groups_lab]
        groups_unlab = [g[~m] for g, m in zip(groups, masks)]
        ns = np.array([len(g) for g in groups_unlab], dtype=float)
        means = np.array([float(np.mean(g)) for g in groups_unlab], dtype=float)
        grand = float(np.sum(ns * means) / np.sum(ns))
        llm_between = float(np.sum(ns * (means - grand) ** 2) / np.sum(ns))

        assert r.corrected_estimate == pytest.approx(llm_between + r.rectifier, abs=1e-10)

    def test_corrected_estimate_equals_llm_plus_rectifier_repeated(self):
        """Y_hat_unlab must be DISJOINT from the labeled subjects (see
        evalstats.ppi.correct's docstring), so llm_between is computed on
        the unlabeled-subject rows only, not the full matrix."""
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(518)
        groups, groups_lab = _multigroup_repeated(
            rng,
            k=3,
            n=220,
            mus=[3.0, 3.0, 3.0],
            biases=[1.5, 0.0, 0.0],
            n_lab=60,
        )
        r = anova_oneway(*groups, repeated=True, groups_lab=groups_lab, n_boot=200, rng=518)

        labels_mat = np.column_stack(groups_lab)
        overlap = np.all(~np.isnan(labels_mat), axis=1)
        llm_mat = np.column_stack(groups)[~overlap]
        centered = llm_mat - llm_mat.mean(axis=1, keepdims=True)
        cond_means = centered.mean(axis=0)
        llm_between = float(np.mean((cond_means - cond_means.mean()) ** 2))

        assert r.corrected_estimate == pytest.approx(llm_between + r.rectifier, abs=1e-10)

    def test_corrected_p_value_agrees_with_corrected_ci_independent(self):
        alpha = 0.05
        rng = np.random.default_rng(519)
        groups, groups_lab = _multigroup(
            rng,
            k=3,
            n=230,
            mus=[3.0, 4.1, 3.4],
            sigma=0.9,
            biases=[0.8, 0.0, 0.0],
            n_lab=70,
        )
        r = anova_oneway(*groups, groups_lab=groups_lab, alpha=alpha, n_boot=400, rng=519)
        lo, hi = r.corrected_ci
        ci_rejects = (hi < 0.0) or (lo > 0.0)
        p_rejects = r.corrected_p_value < alpha
        assert p_rejects == ci_rejects

    def test_corrected_p_value_agrees_with_corrected_ci_repeated(self):
        pytest.importorskip("statsmodels")
        alpha = 0.05
        rng = np.random.default_rng(520)
        groups, groups_lab = _multigroup_repeated(
            rng,
            k=3,
            n=230,
            mus=[3.0, 4.1, 3.4],
            sigma=0.9,
            biases=[0.8, 0.0, 0.0],
            n_lab=70,
        )
        r = anova_oneway(*groups, repeated=True, groups_lab=groups_lab, alpha=alpha, n_boot=400, rng=520)
        lo, hi = r.corrected_ci
        ci_rejects = (hi < 0.0) or (lo > 0.0)
        p_rejects = r.corrected_p_value < alpha
        assert p_rejects == ci_rejects

    def test_fully_labeled_raises_informatively_independent(self):
        """All items labeled -> no unlabeled pool left to extrapolate the
        correction to (Y_hat_unlab must be DISJOINT from the labeled
        positions -- see evalstats.ppi.correct's docstring), so this now
        raises a clear, actionable error."""
        rng = np.random.default_rng(521)
        groups, groups_lab = _multigroup(
            rng,
            k=3,
            n=180,
            mus=[3.1, 3.8, 2.9],
            sigma=0.8,
            biases=[0.9, -0.3, 0.5],
            n_lab=180,
        )
        with pytest.raises(ValueError, match="unlabeled pool"):
            anova_oneway(*groups, groups_lab=groups_lab, n_boot=250, rng=521)

    def test_fully_labeled_raises_informatively_repeated(self):
        """Same as test_fully_labeled_raises_informatively_independent, for
        the repeated-measures path."""
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(522)
        groups, groups_lab = _multigroup_repeated(
            rng,
            k=3,
            n=180,
            mus=[3.1, 3.8, 2.9],
            sigma=0.8,
            biases=[0.9, -0.3, 0.5],
            n_lab=180,
        )
        with pytest.raises(ValueError, match="unlabeled pool"):
            anova_oneway(*groups, repeated=True, groups_lab=groups_lab, n_boot=250, rng=522)

    def test_raises_when_fewer_than_15_labels_independent(self):
        rng = np.random.default_rng(523)
        groups, groups_lab = _multigroup(rng, k=3, n=120, n_lab=4)
        with pytest.raises(ValueError, match="At least 15 human labels"):
            anova_oneway(*groups, groups_lab=groups_lab, n_boot=120, rng=523)

    def test_warns_when_15_to_29_labels_independent(self):
        rng = np.random.default_rng(524)
        groups, groups_lab = _multigroup(rng, k=3, n=120, n_lab=6)
        with pytest.warns(UserWarning, match="undercover below 30 labels"):
            r = anova_oneway(*groups, groups_lab=groups_lab, n_boot=120, rng=524)
        assert r.corrected_estimate is not None

    def test_raises_when_fewer_than_15_overlapping_subjects_repeated(self):
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(525)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=120, n_lab=10)
        with pytest.raises(ValueError, match="At least 15 subjects"):
            anova_oneway(*groups, repeated=True, groups_lab=groups_lab, n_boot=120, rng=525)

    def test_warns_when_15_to_29_overlapping_subjects_repeated(self):
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(526)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=120, n_lab=20)
        with pytest.warns(UserWarning, match="undercover below 30 labels"):
            r = anova_oneway(*groups, repeated=True, groups_lab=groups_lab, n_boot=120, rng=526)
        assert r.corrected_estimate is not None


class TestAnovaOnewayFalsePositive:
    """LLM inflates one group but true groups are equal → corrected CI covers 0."""

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_differential_bias_corrected_CI_contains_zero_independent(self, seed):
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup(
            rng, k=3, n=300, mus=[3.0, 3.0, 3.0], biases=[2.0, 0.0, 0.0], n_lab=80
        )
        r = anova_oneway(*groups, groups_lab=groups_lab, n_boot=500, rng=seed + 20000)
        assert r.p_value < 0.001, f"seed={seed}: uncorrected should detect false positive"
        lo, hi = r.corrected_ci
        assert lo <= 0 <= hi, (
            f"seed={seed}: corrected CI ({lo:.4f}, {hi:.4f}) should contain 0 "
            f"(no true between-group variance)"
        )

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_differential_bias_corrected_estimate_closer_to_zero(self, seed):
        """Quantitative: corrected variance estimate must be closer to 0 than raw LLM estimate."""
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup(
            rng, k=3, n=300, mus=[3.0, 3.0, 3.0], biases=[2.0, 0.0, 0.0], n_lab=80
        )
        r = anova_oneway(*groups, groups_lab=groups_lab, n_boot=500, rng=seed + 20000)
        # Raw LLM between-group variance
        ns = np.array([len(g) for g in groups], dtype=float)
        means = np.array([float(np.mean(g)) for g in groups], dtype=float)
        grand = float(np.sum(ns * means) / np.sum(ns))
        llm_est = float(np.sum(ns * (means - grand) ** 2) / np.sum(ns))
        assert abs(r.corrected_estimate) < abs(llm_est), (
            f"seed={seed}: corrected={r.corrected_estimate:.4f} should be closer to 0 "
            f"than llm_est={llm_est:.4f}"
        )

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_differential_bias_corrected_CI_contains_zero_repeated(self, seed):
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup_repeated(
            rng, k=3, n=250, mus=[3.0, 3.0, 3.0], biases=[2.0, 0.0, 0.0], n_lab=70
        )
        r = anova_oneway(*groups, repeated=True, groups_lab=groups_lab, n_boot=500, rng=seed + 20000)
        assert r.p_value < 0.001, f"seed={seed}: uncorrected should detect false positive"
        lo, hi = r.corrected_ci
        assert lo <= 0 <= hi, (
            f"seed={seed}: repeated corrected CI ({lo:.4f}, {hi:.4f}) should contain 0"
        )

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_differential_bias_corrected_estimate_closer_to_zero_repeated(self, seed):
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup_repeated(
            rng, k=3, n=250, mus=[3.0, 3.0, 3.0], biases=[2.0, 0.0, 0.0], n_lab=70
        )
        r = anova_oneway(*groups, repeated=True, groups_lab=groups_lab, n_boot=500, rng=seed + 20000)

        llm_mat = np.column_stack(groups)
        centered = llm_mat - llm_mat.mean(axis=1, keepdims=True)
        cond_means = centered.mean(axis=0)
        llm_est = float(np.mean((cond_means - cond_means.mean()) ** 2))

        assert abs(r.corrected_estimate) < abs(llm_est), (
            f"seed={seed}: repeated corrected={r.corrected_estimate:.4f} should be closer to 0 "
            f"than llm_est={llm_est:.4f}"
        )


class TestAnovaOnewayTrueEffect:
    """Large true between-group effect survives PPI correction."""

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_large_true_effect_CI_excludes_zero_independent(self, seed):
        """True group means differ by ~2σ across conditions; corrected CI should exclude 0."""
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup(
            rng, k=3, n=200, mus=[3.0, 5.0, 4.0], sigma=0.8, biases=[0.0, 0.0, 0.0], n_lab=50
        )
        r = anova_oneway(*groups, groups_lab=groups_lab, n_boot=300, rng=seed + 30000)
        lo, hi = r.corrected_ci
        assert lo > 0, (
            f"seed={seed}: corrected CI ({lo:.3f}, {hi:.3f}) should be entirely "
            f"above 0 for a large true between-group effect"
        )

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_large_true_effect_CI_excludes_zero_repeated(self, seed):
        pytest.importorskip("statsmodels")
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup_repeated(
            rng, k=3, n=200, mus=[3.0, 5.0, 4.0], sigma=0.8, biases=[0.0, 0.0, 0.0], n_lab=50
        )
        r = anova_oneway(*groups, repeated=True, groups_lab=groups_lab, n_boot=300, rng=seed + 30000)
        lo, hi = r.corrected_ci
        assert lo > 0, (
            f"seed={seed}: repeated corrected CI ({lo:.3f}, {hi:.3f}) should be entirely "
            f"above 0 for a large true between-group effect"
        )


class TestAnovaOnewayCIWidthLabelBudget:
    """More human labels → narrower corrected CI for both ANOVA variants.

    We hold the synthetic LLM data fixed and increase n_lab, measuring the
    corrected CI width averaged over multiple label-mask draws at each budget.
    CI width is the right property to test here: more labels always reduces
    rectifier variance and therefore narrows the bootstrap CI.

    Note: the corrected *p-value* is NOT tested here.  For the ANOVA estimand
    (between-group variance, bounded ≥ 0), the p-value can dip at intermediate
    label counts as CI narrowing and estimate convergence interact non-linearly.
    CI width has no such non-monotone behaviour — it is the property the
    bootstrap directly controls.
    """

    def test_ci_width_narrows_with_n_lab_independent(self):
        """Independent one-way ANOVA: endpoint CI width drops materially with more labels."""
        seed = 610
        rng = np.random.default_rng(seed)
        k, n = 3, 320
        truths = [rng.normal(3.0, 1.0, n) for _ in range(k)]
        groups = [truths[i] + [1.3, 0.0, 0.0][i] + rng.normal(0.0, 0.10, n) for i in range(k)]

        n_labs = [15, 30, 60, 100, 160]
        widths = []

        for n_lab in n_labs:
            rep_w = []
            for rep in range(6):
                label_rng = np.random.default_rng(seed * 1000 + n_lab * 10 + rep)
                groups_lab = []
                for i in range(k):
                    lab = np.full(n, np.nan)
                    idx = label_rng.choice(n, n_lab, replace=False)
                    lab[idx] = truths[i][idx]
                    groups_lab.append(lab)
                r = anova_oneway(*groups, groups_lab=groups_lab, n_boot=2_000,
                                 rng=seed + 1000 + n_lab * 10 + rep)
                lo, hi = r.corrected_ci
                rep_w.append(hi - lo)
            widths.append(float(np.mean(rep_w)))

        # Endpoints: largest label budget must give materially narrower CI.
        assert widths[-1] < widths[0], (
            f"Expected CI to narrow from n_lab=15 to 160; got widths={widths}"
        )
        # Local trend: allow small upward jitter but no large reversals.
        for i in range(len(widths) - 1):
            assert widths[i + 1] <= widths[i] + 0.03, (
                f"Large upward flip between n_lab={n_labs[i]} and {n_labs[i+1]}: widths={widths}"
            )

    def test_ci_width_narrows_with_n_lab_repeated(self):
        """Repeated-measures one-way ANOVA: same endpoint CI width check."""
        pytest.importorskip("statsmodels")
        seed = 710
        rng = np.random.default_rng(seed)
        k, n = 3, 260
        subject_fx = rng.normal(0.0, 0.5, n)
        truths = [3.0 + subject_fx + rng.normal(0.0, 0.9, n) for _ in range(k)]
        groups = [truths[i] + [1.1, 0.0, 0.0][i] + rng.normal(0.0, 0.10, n) for i in range(k)]

        n_labs = [15, 30, 50, 80, 120]
        widths = []

        for n_lab in n_labs:
            rep_w = []
            for rep in range(6):
                label_rng = np.random.default_rng(seed * 1000 + n_lab * 10 + rep)
                idx = label_rng.choice(n, n_lab, replace=False)
                groups_lab = [np.full(n, np.nan) for _ in range(k)]
                for i in range(k):
                    groups_lab[i][idx] = truths[i][idx]
                r = anova_oneway(*groups, repeated=True, groups_lab=groups_lab,
                                 n_boot=2_000, rng=seed + 2000 + n_lab * 10 + rep)
                lo, hi = r.corrected_ci
                rep_w.append(hi - lo)
            widths.append(float(np.mean(rep_w)))

        assert widths[-1] < widths[0], (
            f"Expected CI to narrow from n_lab=15 to 120; got widths={widths}"
        )
        for i in range(len(widths) - 1):
            assert widths[i + 1] <= widths[i] + 0.03, (
                f"Large upward flip between n_lab={n_labs[i]} and {n_labs[i+1]}: widths={widths}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Type I error calibration for the corrected ANOVA p-value
#
# Under H₀ (equal group means) the corrected p-value should produce a Type I
# error rate ≤ 0.12 when α = 0.05.  The threshold 0.12 is the 3σ upper bound
# for a Binomial(100, 0.05) count of rejections, giving < 0.1% false-alarm
# probability for the assertion itself.
#
# Each test method draws 100 fixed seeds and calls the helper directly
# (bypassing the full anova_oneway pipeline for speed).
# ─────────────────────────────────────────────────────────────────────────────

class TestAnovaOnewayCorrectedPValueCalibration:
    """Type I error rate of _ppi_anova_independent_p_value and _ppi_anova_repeated_p_value."""

    SEEDS = list(range(5000, 5100))   # 100 reproducible seeds
    ALPHA = 0.05
    MAX_REJECTION_RATE = 0.12         # 3σ upper bound for Binomial(100, 0.05)

    # ── data helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _independent_p(
        seed: int,
        biases: list,
        mus: list,
        n: int = 200,
        n_lab: int = 40,
        sigma: float = 1.0,
        llm_noise: float = 0.12,
    ):
        """One draw of the independent-groups corrected p-value under H₀."""
        rng = np.random.default_rng(seed)
        k = len(mus)
        truths = [rng.normal(mu, sigma, n) for mu in mus]
        groups = [t + bias + rng.normal(0, llm_noise, n) for t, bias in zip(truths, biases)]
        idx = rng.choice(n, n_lab, replace=False)
        groups_lab = [np.full(n, np.nan) for _ in range(k)]
        for i in range(k):
            groups_lab[i][idx] = truths[i][idx]
        return _ppi_anova_independent_p_value(groups, groups_lab, k)

    @staticmethod
    def _repeated_p(
        seed: int,
        biases: list,
        mus: list,
        n_subjects: int = 150,
        n_lab: int = 35,
        sigma_subject: float = 0.6,
        sigma_residual: float = 0.8,
        llm_noise: float = 0.12,
    ):
        """One draw of the repeated-measures corrected p-value under H₀."""
        rng = np.random.default_rng(seed)
        k = len(mus)
        subject_fx = rng.normal(0, sigma_subject, n_subjects)
        truths = [
            mu + subject_fx + rng.normal(0, sigma_residual, n_subjects)
            for mu in mus
        ]
        groups = [
            t + bias + rng.normal(0, llm_noise, n_subjects)
            for t, bias in zip(truths, biases)
        ]
        idx = rng.choice(n_subjects, n_lab, replace=False)
        groups_lab = [np.full(n_subjects, np.nan) for _ in range(k)]
        for i in range(k):
            groups_lab[i][idx] = truths[i][idx]
        return _ppi_anova_repeated_p_value(groups, groups_lab, k)

    # ── independent-groups tests ───────────────────────────────────────────────

    def test_type_i_independent_null_unbiased(self):
        """No LLM bias: Type I rate should be ~0.05."""
        rejections = sum(
            1 for s in self.SEEDS
            if (p := self._independent_p(s, biases=[0.0, 0.0, 0.0], mus=[3.0, 3.0, 3.0]))
            is not None and p < self.ALPHA
        )
        rate = rejections / len(self.SEEDS)
        assert rate <= self.MAX_REJECTION_RATE, (
            f"Independent unbiased null: Type I rate {rate:.3f} > {self.MAX_REJECTION_RATE}"
        )

    def test_type_i_independent_null_biased(self):
        """Differential LLM bias (one group inflated): Type I rate should still be ~0.05."""
        rejections = sum(
            1 for s in self.SEEDS
            if (p := self._independent_p(s, biases=[1.5, 0.0, 0.0], mus=[3.0, 3.0, 3.0]))
            is not None and p < self.ALPHA
        )
        rate = rejections / len(self.SEEDS)
        assert rate <= self.MAX_REJECTION_RATE, (
            f"Independent biased null: Type I rate {rate:.3f} > {self.MAX_REJECTION_RATE}"
        )

    # ── repeated-measures tests ────────────────────────────────────────────────

    def test_type_i_repeated_null_unbiased(self):
        """No LLM bias: repeated-measures Type I rate should be ~0.05."""
        rejections = sum(
            1 for s in self.SEEDS
            if (p := self._repeated_p(s, biases=[0.0, 0.0, 0.0], mus=[3.0, 3.0, 3.0]))
            is not None and p < self.ALPHA
        )
        rate = rejections / len(self.SEEDS)
        assert rate <= self.MAX_REJECTION_RATE, (
            f"Repeated unbiased null: Type I rate {rate:.3f} > {self.MAX_REJECTION_RATE}"
        )

    def test_type_i_repeated_null_biased(self):
        """Differential LLM bias across conditions: repeated-measures Type I rate should be ~0.05."""
        rejections = sum(
            1 for s in self.SEEDS
            if (p := self._repeated_p(s, biases=[0.0, 0.8, 0.4], mus=[3.0, 3.0, 3.0]))
            is not None and p < self.ALPHA
        )
        rate = rejections / len(self.SEEDS)
        assert rate <= self.MAX_REJECTION_RATE, (
            f"Repeated biased null: Type I rate {rate:.3f} > {self.MAX_REJECTION_RATE}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# friedman — the rank-based analog of repeated-measures one-way ANOVA.
#
# Reuses _multigroup_repeated() since the data shape (k equal-length
# conditions, same subjects labeled across all conditions) is identical to
# the repeated-measures ANOVA case.
# ─────────────────────────────────────────────────────────────────────────────

class TestFriedmanRankVarianceFormula:
    """_friedman_rank_variance ties directly to the classical χ²_F statistic.

    χ²_F = 12·n·V/(k+1), where V = _friedman_rank_variance(matrix). Checked
    against scipy's own (tie-free, since data is continuous) statistic.
    """

    def test_matches_chi2_relation_no_ties(self):
        rng = np.random.default_rng(900)
        k, n = 4, 150
        groups = [rng.normal(0, 1, n) for _ in range(k)]
        matrix = np.column_stack(groups)
        V = _friedman_rank_variance(matrix)
        chi2_from_V = 12.0 * n * V / (k + 1)
        ref = scipy_stats.friedmanchisquare(*groups)
        assert chi2_from_V == pytest.approx(ref.statistic, rel=1e-8)


class TestFriedmanBaseline:
    """No-label tests: statistics match scipy; PPI fields all None."""

    def test_matches_scipy(self):
        rng = np.random.default_rng(901)
        groups, _ = _multigroup_repeated(rng, k=4, n=120)
        r = friedman(*groups)
        ref = scipy_stats.friedmanchisquare(*groups)
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.p_value == pytest.approx(ref.pvalue)

    def test_ppi_fields_none_without_labels(self):
        rng = np.random.default_rng(902)
        groups, _ = _multigroup_repeated(rng, k=3, n=80)
        r = friedman(*groups)
        assert isinstance(r, TestResult)
        assert r.corrected_estimate is None
        assert r.corrected_ci is None
        assert r.corrected_p_value is None
        assert r.rectifier is None
        assert r.n_labeled is None
        assert r.n_total is None
        assert "corrected_effect_size" not in r.extra

    def test_test_name_contains_friedman(self):
        rng = np.random.default_rng(903)
        groups, _ = _multigroup_repeated(rng, k=3, n=60)
        assert "friedman" in friedman(*groups).test_name.lower()

    def test_requires_at_least_three_conditions(self):
        rng = np.random.default_rng(904)
        g1 = rng.normal(0, 1, 50)
        g2 = rng.normal(0, 1, 50)
        with pytest.raises(ValueError, match="at least three conditions"):
            friedman(g1, g2)

    def test_raises_for_unequal_lengths(self):
        rng = np.random.default_rng(905)
        g1 = rng.normal(0, 1, 80)
        g2 = rng.normal(0, 1, 80)
        g3 = rng.normal(0, 1, 90)
        with pytest.raises(ValueError, match="equal-length"):
            friedman(g1, g2, g3)

    def test_extra_dict_keys(self):
        rng = np.random.default_rng(906)
        groups, _ = _multigroup_repeated(rng, k=4, n=80)
        r = friedman(*groups)
        for key in ("_test", "k_groups", "n_subjects", "group_means", "df", "effect_size", "effect_size_name"):
            assert key in r.extra, f"Missing extra key: {key!r}"
        assert r.extra["k_groups"] == 4
        assert r.extra["n_subjects"] == 80
        assert r.extra["df"] == pytest.approx(3.0)  # k-1
        assert 0.0 <= r.extra["effect_size"] <= 1.0  # Kendall's W is bounded [0, 1]

    def test_invariant_to_condition_permutation(self):
        rng = np.random.default_rng(907)
        groups, _ = _multigroup_repeated(rng, k=4, n=100, mus=[3.0, 4.0, 3.5, 5.0])
        r1 = friedman(*groups)
        r2 = friedman(*groups[::-1])
        assert r1.statistic == pytest.approx(r2.statistic)
        assert r1.p_value == pytest.approx(r2.p_value)
        assert r1.extra["effect_size"] == pytest.approx(r2.extra["effect_size"])


class TestFriedmanPPIFieldStructure:
    """PPI-corrected friedman: field structure, ordering, and metadata."""

    def test_fields_populated(self):
        rng = np.random.default_rng(910)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=200, n_lab=50)
        r = friedman(*groups, groups_lab=groups_lab, n_boot=200, rng=910)
        assert r.corrected_estimate is not None
        assert r.corrected_ci is not None
        assert r.corrected_p_value is not None
        assert r.rectifier is not None
        assert r.n_labeled is not None
        assert r.n_total is not None
        assert "corrected_effect_size" in r.extra
        assert "corrected_effect_size_name" in r.extra

    def test_ci_is_ordered(self):
        rng = np.random.default_rng(911)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=200, n_lab=50)
        r = friedman(*groups, groups_lab=groups_lab, n_boot=200, rng=911)
        lo, hi = r.corrected_ci
        assert lo <= hi

    def test_corrected_p_value_in_0_1(self):
        rng = np.random.default_rng(912)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=200, n_lab=50)
        r = friedman(*groups, groups_lab=groups_lab, n_boot=300, rng=912)
        assert 0.0 <= r.corrected_p_value <= 1.0

    def test_n_labeled_and_n_total_match_input(self):
        """3 conditions × 200 subjects × 50 labeled subjects → n_labeled=150, n_total=600."""
        rng = np.random.default_rng(913)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=200, n_lab=50)
        r = friedman(*groups, groups_lab=groups_lab, n_boot=200, rng=913)
        assert r.n_labeled == 150
        assert r.n_total == 600

    def test_reproducibility_with_rng_seed(self):
        rng = np.random.default_rng(914)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=200, n_lab=50)
        kw = dict(groups_lab=groups_lab, n_boot=500, rng=42)
        r1 = friedman(*groups, **kw)
        r2 = friedman(*groups, **kw)
        assert r1.corrected_estimate == r2.corrected_estimate
        assert r1.corrected_ci == r2.corrected_ci

    def test_alpha_propagated(self):
        rng = np.random.default_rng(915)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=200, n_lab=50)
        r = friedman(*groups, groups_lab=groups_lab, alpha=0.10, n_boot=200, rng=915)
        assert r.alpha == 0.10

    def test_corrected_estimate_equals_llm_plus_rectifier(self):
        """Y_hat_unlab must be DISJOINT from the labeled subjects (see
        evalstats.ppi.correct's docstring), so llm_est is computed on the
        unlabeled-subject rows only, not the full matrix."""
        rng = np.random.default_rng(916)
        groups, groups_lab = _multigroup_repeated(
            rng, k=3, n=220, mus=[3.0, 3.0, 3.0], biases=[1.5, 0.0, 0.0], n_lab=60,
        )
        r = friedman(*groups, groups_lab=groups_lab, n_boot=200, rng=916)

        labels_mat = np.column_stack(groups_lab)
        overlap = np.all(~np.isnan(labels_mat), axis=1)
        llm_est = _friedman_rank_variance(np.column_stack(groups)[~overlap])
        assert r.corrected_estimate == pytest.approx(llm_est + r.rectifier, abs=1e-10)

    def test_fully_labeled_raises_informatively(self):
        """All items labeled -> no unlabeled pool left to extrapolate the
        correction to (Y_hat_unlab must be DISJOINT from the labeled
        subjects -- see evalstats.ppi.correct's docstring), so this now
        raises a clear, actionable error."""
        rng = np.random.default_rng(917)
        groups, groups_lab = _multigroup_repeated(
            rng, k=3, n=180, mus=[3.1, 3.8, 2.9], sigma=0.8,
            biases=[0.9, -0.3, 0.5], n_lab=180,
        )
        with pytest.raises(ValueError, match="unlabeled pool"):
            friedman(*groups, groups_lab=groups_lab, n_boot=250, rng=917)

    def test_corrected_effect_size_matches_kendalls_w_rescaling(self):
        rng = np.random.default_rng(918)
        groups, groups_lab = _multigroup_repeated(rng, k=4, n=200, n_lab=60)
        r = friedman(*groups, groups_lab=groups_lab, n_boot=200, rng=918)

        k = 4
        expected_w = 12.0 * r.corrected_estimate / (k**2 - 1)
        assert r.extra["corrected_effect_size"] == pytest.approx(expected_w, abs=1e-10)

    def test_corrected_p_value_agrees_with_corrected_ci(self):
        alpha = 0.05
        rng = np.random.default_rng(919)
        groups, groups_lab = _multigroup_repeated(
            rng, k=3, n=230, mus=[3.0, 4.1, 3.4], sigma=0.9,
            biases=[0.8, 0.0, 0.0], n_lab=70,
        )
        r = friedman(*groups, groups_lab=groups_lab, alpha=alpha, n_boot=400, rng=919)
        lo, hi = r.corrected_ci
        ci_rejects = (hi < 0.0) or (lo > 0.0)
        p_rejects = r.corrected_p_value < alpha
        assert p_rejects == ci_rejects

    def test_raises_when_fewer_than_15_overlapping_subjects(self):
        rng = np.random.default_rng(920)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=120, n_lab=10)
        with pytest.raises(ValueError, match="At least 15 subjects"):
            friedman(*groups, groups_lab=groups_lab, n_boot=120, rng=920)

    def test_warns_when_15_to_29_overlapping_subjects(self):
        rng = np.random.default_rng(921)
        groups, groups_lab = _multigroup_repeated(rng, k=3, n=120, n_lab=20)
        with pytest.warns(UserWarning, match="undercover below 30 labels"):
            r = friedman(*groups, groups_lab=groups_lab, n_boot=120, rng=921)
        assert r.corrected_estimate is not None


class TestFriedmanFalsePositive:
    """LLM inflates one condition but true conditions are equal → corrected CI covers 0."""

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_differential_bias_corrected_CI_contains_zero(self, seed):
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup_repeated(
            rng, k=3, n=250, mus=[3.0, 3.0, 3.0], biases=[2.0, 0.0, 0.0], n_lab=70,
        )
        r = friedman(*groups, groups_lab=groups_lab, n_boot=500, rng=seed + 20000)
        assert r.p_value < 0.001, f"seed={seed}: uncorrected should detect false positive"
        lo, hi = r.corrected_ci
        assert lo <= 0 <= hi, (
            f"seed={seed}: corrected CI ({lo:.4f}, {hi:.4f}) should contain 0 "
            f"(no true between-condition rank variance)"
        )

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_differential_bias_corrected_estimate_closer_to_zero(self, seed):
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup_repeated(
            rng, k=3, n=250, mus=[3.0, 3.0, 3.0], biases=[2.0, 0.0, 0.0], n_lab=70,
        )
        r = friedman(*groups, groups_lab=groups_lab, n_boot=500, rng=seed + 20000)
        llm_est = _friedman_rank_variance(np.column_stack(groups))
        assert abs(r.corrected_estimate) < abs(llm_est), (
            f"seed={seed}: corrected={r.corrected_estimate:.4f} should be closer to 0 "
            f"than llm_est={llm_est:.4f}"
        )


class TestFriedmanTrueEffect:
    """Large true between-condition effect survives PPI correction."""

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_large_true_effect_CI_excludes_zero(self, seed):
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup_repeated(
            rng, k=3, n=200, mus=[3.0, 5.0, 4.0], sigma=0.8, biases=[0.0, 0.0, 0.0], n_lab=50,
        )
        r = friedman(*groups, groups_lab=groups_lab, n_boot=300, rng=seed + 30000)
        lo, hi = r.corrected_ci
        assert lo > 0, (
            f"seed={seed}: corrected CI ({lo:.3f}, {hi:.3f}) should be entirely "
            f"above 0 for a large true between-condition effect"
        )


class TestFriedmanCIWidthLabelBudget:
    """More human labels → narrower corrected CI (mirrors the ANOVA budget test)."""

    def test_ci_width_narrows_with_n_lab(self):
        seed = 930
        rng = np.random.default_rng(seed)
        k, n = 3, 260
        subject_fx = rng.normal(0.0, 0.5, n)
        truths = [3.0 + subject_fx + rng.normal(0.0, 0.9, n) for _ in range(k)]
        groups = [truths[i] + [1.1, 0.0, 0.0][i] + rng.normal(0.0, 0.10, n) for i in range(k)]

        n_labs = [15, 30, 50, 80, 120]
        widths = []

        for n_lab in n_labs:
            rep_w = []
            for rep in range(6):
                label_rng = np.random.default_rng(seed * 1000 + n_lab * 10 + rep)
                idx = label_rng.choice(n, n_lab, replace=False)
                groups_lab = [np.full(n, np.nan) for _ in range(k)]
                for i in range(k):
                    groups_lab[i][idx] = truths[i][idx]
                r = friedman(*groups, groups_lab=groups_lab,
                              n_boot=2_000, rng=seed + 2000 + n_lab * 10 + rep)
                lo, hi = r.corrected_ci
                rep_w.append(hi - lo)
            widths.append(float(np.mean(rep_w)))

        assert widths[-1] < widths[0], (
            f"Expected CI to narrow from n_lab=15 to 120; got widths={widths}"
        )
        for i in range(len(widths) - 1):
            assert widths[i + 1] <= widths[i] + 0.03, (
                f"Large upward flip between n_lab={n_labs[i]} and {n_labs[i+1]}: widths={widths}"
            )


class TestFriedmanCorrectedPValueCalibration:
    """Type I error rate of _ppi_friedman_p_value (mirrors the ANOVA calibration test)."""

    SEEDS = list(range(6000, 6100))   # 100 reproducible seeds
    ALPHA = 0.05
    MAX_REJECTION_RATE = 0.12          # 3σ upper bound for Binomial(100, 0.05)

    @staticmethod
    def _friedman_p(
        seed: int,
        biases: list,
        mus: list,
        n_subjects: int = 150,
        n_lab: int = 35,
        sigma_subject: float = 0.6,
        sigma_residual: float = 0.8,
        llm_noise: float = 0.12,
    ):
        """One draw of the Friedman corrected p-value under H₀."""
        rng = np.random.default_rng(seed)
        k = len(mus)
        subject_fx = rng.normal(0, sigma_subject, n_subjects)
        truths = [
            mu + subject_fx + rng.normal(0, sigma_residual, n_subjects)
            for mu in mus
        ]
        groups = [
            t + bias + rng.normal(0, llm_noise, n_subjects)
            for t, bias in zip(truths, biases)
        ]
        idx = rng.choice(n_subjects, n_lab, replace=False)
        groups_lab = [np.full(n_subjects, np.nan) for _ in range(k)]
        for i in range(k):
            groups_lab[i][idx] = truths[i][idx]
        return _ppi_friedman_p_value(groups, groups_lab, k)

    def test_type_i_null_unbiased(self):
        """No LLM bias: Type I rate should be ~0.05."""
        rejections = sum(
            1 for s in self.SEEDS
            if (p := self._friedman_p(s, biases=[0.0, 0.0, 0.0], mus=[3.0, 3.0, 3.0]))
            is not None and p < self.ALPHA
        )
        rate = rejections / len(self.SEEDS)
        assert rate <= self.MAX_REJECTION_RATE, (
            f"Friedman unbiased null: Type I rate {rate:.3f} > {self.MAX_REJECTION_RATE}"
        )

    def test_type_i_null_biased(self):
        """Differential LLM bias across conditions: Type I rate should still be ~0.05."""
        rejections = sum(
            1 for s in self.SEEDS
            if (p := self._friedman_p(s, biases=[0.0, 0.8, 0.4], mus=[3.0, 3.0, 3.0]))
            is not None and p < self.ALPHA
        )
        rate = rejections / len(self.SEEDS)
        assert rate <= self.MAX_REJECTION_RATE, (
            f"Friedman biased null: Type I rate {rate:.3f} > {self.MAX_REJECTION_RATE}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# kruskalwallis — the rank-based analog of independent-groups one-way ANOVA.
#
# Reuses _multigroup() since the data shape (k independent groups, labels
# scattered independently per group) is identical to the independent-groups
# ANOVA case. Unlike Friedman, Kruskal-Wallis's PPI estimand is built from
# pairwise stochastic-dominance probabilities (see _kw_pairwise_thetas),
# corrected via a Wald test with bootstrap covariance rather than a
# closed-form null variance.
# ─────────────────────────────────────────────────────────────────────────────

class TestKruskalBaseline:
    """No-label tests: statistics match scipy; PPI fields all None."""

    def test_matches_scipy(self):
        rng = np.random.default_rng(940)
        groups, _ = _multigroup(rng, k=4, n=120)
        r = kruskalwallis(*groups)
        ref = scipy_stats.kruskal(*groups)
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.p_value == pytest.approx(ref.pvalue)

    def test_ppi_fields_none_without_labels(self):
        rng = np.random.default_rng(941)
        groups, _ = _multigroup(rng, k=3, n=80)
        r = kruskalwallis(*groups)
        assert isinstance(r, TestResult)
        assert r.corrected_estimate is None
        assert r.corrected_ci is None
        assert r.corrected_p_value is None
        assert r.rectifier is None
        assert r.n_labeled is None
        assert r.n_total is None
        assert "corrected_effect_size" not in r.extra
        assert "pairwise_effects" not in r.extra

    def test_test_name_contains_kruskal(self):
        rng = np.random.default_rng(942)
        groups, _ = _multigroup(rng, k=3, n=60)
        assert "kruskal" in kruskalwallis(*groups).test_name.lower()

    def test_requires_at_least_three_groups(self):
        rng = np.random.default_rng(943)
        g1 = rng.normal(0, 1, 50)
        g2 = rng.normal(0, 1, 50)
        with pytest.raises(ValueError, match="at least three groups"):
            kruskalwallis(g1, g2)

    def test_allows_unequal_group_sizes(self):
        """Unlike friedman, kruskalwallis groups are independent — no length constraint."""
        rng = np.random.default_rng(944)
        g1 = rng.normal(0, 1, 60)
        g2 = rng.normal(0, 1, 90)
        g3 = rng.normal(0, 1, 120)
        r = kruskalwallis(g1, g2, g3)
        ref = scipy_stats.kruskal(g1, g2, g3)
        assert r.statistic == pytest.approx(ref.statistic)

    def test_extra_dict_keys(self):
        rng = np.random.default_rng(945)
        groups, _ = _multigroup(rng, k=4, n=80)
        r = kruskalwallis(*groups)
        for key in ("_test", "k_groups", "n_total_obs", "group_sizes", "group_medians",
                    "df", "effect_size", "effect_size_name"):
            assert key in r.extra, f"Missing extra key: {key!r}"
        assert r.extra["k_groups"] == 4
        assert r.extra["df"] == pytest.approx(3.0)  # k-1
        assert 0.0 <= r.extra["effect_size"] <= 1.0  # bounded [0, 1] by construction

    def test_invariant_to_group_permutation(self):
        rng = np.random.default_rng(946)
        groups, _ = _multigroup(rng, k=4, n=100, mus=[3.0, 4.0, 3.5, 5.0])
        r1 = kruskalwallis(*groups)
        r2 = kruskalwallis(*groups[::-1])
        assert r1.statistic == pytest.approx(r2.statistic)
        assert r1.p_value == pytest.approx(r2.p_value)
        assert r1.extra["effect_size"] == pytest.approx(r2.extra["effect_size"])


class TestKruskalPPIFieldStructure:
    """PPI-corrected kruskalwallis: field structure, ordering, and metadata."""

    def test_fields_populated(self):
        rng = np.random.default_rng(950)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=200, rng=950)
        assert r.corrected_estimate is not None
        assert r.corrected_ci is not None
        assert r.corrected_p_value is not None
        assert r.rectifier is not None
        assert r.n_labeled is not None
        assert r.n_total is not None
        assert "corrected_effect_size" in r.extra
        assert "pairwise_effects" in r.extra
        assert len(r.extra["pairwise_effects"]) == 3  # C(3,2) pairs

    def test_ci_is_ordered(self):
        rng = np.random.default_rng(951)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=200, rng=951)
        lo, hi = r.corrected_ci
        assert lo <= hi
        for info in r.extra["pairwise_effects"].values():
            plo, phi = info["ci"]
            assert plo <= phi
            assert 0.0 <= info["theta"] <= 1.0

    def test_corrected_p_value_in_0_1(self):
        rng = np.random.default_rng(952)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=300, rng=952)
        assert 0.0 <= r.corrected_p_value <= 1.0

    def test_wald_stat_is_nonnegative(self):
        """Quadratic form with a pseudo-inverse of a PSD covariance must be >= 0."""
        rng = np.random.default_rng(953)
        groups, groups_lab = _multigroup(rng, k=4, n=150, n_lab=40)
        r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=200, rng=953)
        assert r.extra["wald_stat"] >= 0.0

    def test_n_labeled_and_n_total_match_input(self):
        """3 groups x 200 items x 50 labeled each -> n_labeled=150, n_total=600."""
        rng = np.random.default_rng(954)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=200, rng=954)
        assert r.n_labeled == 150
        assert r.n_total == 600

    def test_reproducibility_with_rng_seed(self):
        rng = np.random.default_rng(955)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        kw = dict(groups_lab=groups_lab, n_boot=300, rng=42)
        r1 = kruskalwallis(*groups, **kw)
        r2 = kruskalwallis(*groups, **kw)
        assert r1.corrected_estimate == r2.corrected_estimate
        assert r1.corrected_ci == r2.corrected_ci
        assert r1.corrected_p_value == r2.corrected_p_value

    def test_alpha_propagated(self):
        rng = np.random.default_rng(956)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = kruskalwallis(*groups, groups_lab=groups_lab, alpha=0.10, n_boot=200, rng=956)
        assert r.alpha == 0.10

    def test_corrected_estimate_equals_llm_plus_rectifier(self):
        """Y_hat_unlab must be DISJOINT from the labeled positions (see
        evalstats.ppi.correct's docstring), so llm_est is computed on each
        group's unlabeled-only complement, not the full group."""
        rng = np.random.default_rng(957)
        groups, groups_lab = _multigroup(
            rng, k=3, n=220, mus=[3.0, 3.0, 3.0], biases=[1.5, 0.0, 0.0], n_lab=60,
        )
        r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=300, rng=957)

        masks = [~np.isnan(g_lab) for g_lab in groups_lab]
        groups_unlab = [g[~m] for g, m in zip(groups, masks)]
        Y = np.concatenate(groups_unlab)
        X = np.concatenate([np.full(len(g), gid, dtype=int) for gid, g in enumerate(groups_unlab)])
        llm_est = _kw_mean_sq_deviation_from_labeled(Y, X, k=3)
        assert r.corrected_estimate == pytest.approx(llm_est + r.rectifier, abs=1e-10)

    def test_fully_labeled_raises_informatively(self):
        """All items labeled -> no unlabeled pool left to extrapolate the
        correction to (Y_hat_unlab must be DISJOINT from the labeled
        positions -- see evalstats.ppi.correct's docstring), so this now
        raises a clear, actionable error."""
        rng = np.random.default_rng(958)
        groups, groups_lab = _multigroup(
            rng, k=3, n=180, mus=[3.1, 3.8, 2.9], sigma=0.8,
            biases=[0.9, -0.3, 0.5], n_lab=180,
        )
        with pytest.raises(ValueError, match="unlabeled pool"):
            kruskalwallis(*groups, groups_lab=groups_lab, n_boot=300, rng=958)

    def test_corrected_effect_size_equals_corrected_estimate(self):
        """By construction the headline effect size IS the corrected estimand."""
        rng = np.random.default_rng(959)
        groups, groups_lab = _multigroup(rng, k=3, n=200, n_lab=50)
        r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=200, rng=959)
        assert r.extra["corrected_effect_size"] == r.corrected_estimate

    def test_raises_when_fewer_than_15_labels(self):
        rng = np.random.default_rng(960)
        groups, groups_lab = _multigroup(rng, k=3, n=120, n_lab=4)
        with pytest.raises(ValueError, match="At least 15 human labels"):
            kruskalwallis(*groups, groups_lab=groups_lab, n_boot=120, rng=960)

    def test_warns_when_15_to_29_labels(self):
        rng = np.random.default_rng(961)
        groups, groups_lab = _multigroup(rng, k=3, n=120, n_lab=6)
        with pytest.warns(UserWarning, match="undercover below 30 labels"):
            r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=120, rng=961)
        assert r.corrected_estimate is not None


class TestKruskalFalsePositive:
    """LLM inflates one group but true groups are equal -> corrected CI covers 0."""

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_differential_bias_corrected_CI_contains_zero(self, seed):
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup(
            rng, k=3, n=300, mus=[3.0, 3.0, 3.0], biases=[2.0, 0.0, 0.0], n_lab=80,
        )
        r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=500, rng=seed + 20000)
        assert r.p_value < 0.001, f"seed={seed}: uncorrected should detect false positive"
        lo, hi = r.corrected_ci
        assert lo <= 0 <= hi, (
            f"seed={seed}: corrected CI ({lo:.4f}, {hi:.4f}) should contain 0 "
            f"(no true between-group dominance)"
        )

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_differential_bias_corrected_estimate_closer_to_zero(self, seed):
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup(
            rng, k=3, n=300, mus=[3.0, 3.0, 3.0], biases=[2.0, 0.0, 0.0], n_lab=80,
        )
        r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=500, rng=seed + 20000)
        Y = np.concatenate(groups)
        X = np.concatenate([np.full(len(g), gid, dtype=int) for gid, g in enumerate(groups)])
        llm_est = _kw_mean_sq_deviation_from_labeled(Y, X, k=3)
        assert abs(r.corrected_estimate) < abs(llm_est), (
            f"seed={seed}: corrected={r.corrected_estimate:.4f} should be closer to 0 "
            f"than llm_est={llm_est:.4f}"
        )


class TestKruskalTrueEffect:
    """Large true between-group effect survives PPI correction."""

    @pytest.mark.parametrize("seed", _ANOVA_SEEDS)
    def test_large_true_effect_CI_excludes_zero(self, seed):
        rng = np.random.default_rng(seed)
        groups, groups_lab = _multigroup(
            rng, k=3, n=200, mus=[3.0, 5.0, 4.0], sigma=0.8, biases=[0.0, 0.0, 0.0], n_lab=50,
        )
        r = kruskalwallis(*groups, groups_lab=groups_lab, n_boot=300, rng=seed + 30000)
        lo, hi = r.corrected_ci
        assert lo > 0, (
            f"seed={seed}: corrected CI ({lo:.3f}, {hi:.3f}) should be entirely "
            f"above 0 for a large true between-group effect"
        )


class TestKruskalCIWidthLabelBudget:
    """More human labels -> narrower corrected CI (mirrors the ANOVA/Friedman budget test)."""

    def test_ci_width_narrows_with_n_lab(self):
        seed = 970
        rng = np.random.default_rng(seed)
        k, n = 3, 260
        truths = [rng.normal(3.0, 1.0, n) for _ in range(k)]
        groups = [truths[i] + [1.1, 0.0, 0.0][i] + rng.normal(0.0, 0.10, n) for i in range(k)]

        n_labs = [15, 30, 50, 80, 120]
        widths = []

        for n_lab in n_labs:
            rep_w = []
            for rep in range(6):
                label_rng = np.random.default_rng(seed * 1000 + n_lab * 10 + rep)
                groups_lab = [np.full(n, np.nan) for _ in range(k)]
                for i in range(k):
                    idx = label_rng.choice(n, n_lab, replace=False)
                    groups_lab[i][idx] = truths[i][idx]
                r = kruskalwallis(*groups, groups_lab=groups_lab,
                                   n_boot=1_000, rng=seed + 2000 + n_lab * 10 + rep)
                lo, hi = r.corrected_ci
                rep_w.append(hi - lo)
            widths.append(float(np.mean(rep_w)))

        assert widths[-1] < widths[0], (
            f"Expected CI to narrow from n_lab=15 to 120; got widths={widths}"
        )
        for i in range(len(widths) - 1):
            assert widths[i + 1] <= widths[i] + 0.03, (
                f"Large upward flip between n_lab={n_labs[i]} and {n_labs[i+1]}: widths={widths}"
            )


class TestKruskalCorrectedPValueCalibration:
    """Type I error rate of the Wald-test p-value from _ppi_kruskal_wallis_pairwise.

    Lighter-weight than the simulation script's sweep (fewer seeds, smaller
    n_boot) — this is a fast in-suite sanity check, not the authoritative
    calibration validation (that's simulations/sim_type_i_calibration.py).
    """

    SEEDS = list(range(7000, 7060))   # 60 reproducible seeds
    ALPHA = 0.05
    MAX_REJECTION_RATE = 0.15         # loose bound given fewer seeds/smaller n_boot

    @staticmethod
    def _kw_p(
        seed: int,
        biases: list,
        mus: list,
        n: int = 150,
        n_lab: int = 40,
        sigma: float = 1.0,
        llm_noise: float = 0.12,
        n_boot: int = 150,
    ):
        rng = np.random.default_rng(seed)
        k = len(mus)
        truths = [rng.normal(mu, sigma, n) for mu in mus]
        groups = [t + bias + rng.normal(0, llm_noise, n) for t, bias in zip(truths, biases)]
        groups_lab = [np.full(n, np.nan) for _ in range(k)]
        for i in range(k):
            idx = rng.choice(n, n_lab, replace=False)
            groups_lab[i][idx] = truths[i][idx]
        pw = _ppi_kruskal_wallis_pairwise(
            groups, groups_lab, alpha=0.05, n_boot=n_boot, rng=seed
        )
        return pw["wald_p"]

    def test_type_i_null_unbiased(self):
        rejections = sum(
            1 for s in self.SEEDS
            if self._kw_p(s, biases=[0.0, 0.0, 0.0], mus=[3.0, 3.0, 3.0]) < self.ALPHA
        )
        rate = rejections / len(self.SEEDS)
        assert rate <= self.MAX_REJECTION_RATE, (
            f"Kruskal-Wallis unbiased null: Type I rate {rate:.3f} > {self.MAX_REJECTION_RATE}"
        )

    def test_type_i_null_biased(self):
        rejections = sum(
            1 for s in self.SEEDS
            if self._kw_p(s, biases=[0.0, 0.8, 0.4], mus=[3.0, 3.0, 3.0]) < self.ALPHA
        )
        rate = rejections / len(self.SEEDS)
        assert rate <= self.MAX_REJECTION_RATE, (
            f"Kruskal-Wallis biased null: Type I rate {rate:.3f} > {self.MAX_REJECTION_RATE}"
        )
