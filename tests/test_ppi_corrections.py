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

import warnings

import numpy as np
import pytest
from scipy import stats as scipy_stats

from evalstats.tests import TestResult, ttest, mannwhitney, wilcoxon


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

    def test_ttest_independent_matches_scipy(self):
        rng = np.random.default_rng(0)
        a, b, *_ = _two_sample(rng, n=100)
        r = ttest(a, b)
        ref = scipy_stats.ttest_ind(a, b)
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.p_value == pytest.approx(ref.pvalue)

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
        """θ̂_PPI = f(Ŷ_unlab) + rectifier — verified numerically."""
        rng = np.random.default_rng(16)
        a, b, al, bl = _two_sample(rng, n=200, mu_a=4.0, mu_b=3.0,
                                   bias_a=0.5, bias_b=0.0, n_lab=60)
        r = ttest(a, b, a_lab=al, b_lab=bl, n_boot=200, rng=16)
        llm_diff = float(a.mean() - b.mean())
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
        """CI width with 15 labeled items must be wider than with 200.

        We test the endpoints rather than every consecutive pair because
        intermediate steps can show non-monotone fluctuations due to bootstrap
        noise; the trend from 15 → 200 labels is always unambiguous.
        """
        rng_data = np.random.default_rng(999)
        n = 400
        truth_a = rng_data.normal(3.5, 1.0, n)
        truth_b = rng_data.normal(3.0, 1.0, n)
        a = truth_a + rng_data.normal(0, 0.10, n)
        b = truth_b + rng_data.normal(0, 0.10, n)

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
        w_large = _ci_width(200)
        assert w_small > w_large, (
            f"CI with 15 labels ({w_small:.4f}) should be wider than "
            f"with 200 labels ({w_large:.4f})"
        )

    def test_mannwhitney_ci_narrows_as_n_lab_increases(self):
        """Same endpoint check for Mann-Whitney P(X>Y) CI."""
        rng_data = np.random.default_rng(998)
        n = 400
        truth_a = rng_data.normal(3.5, 1.0, n)
        truth_b = rng_data.normal(3.0, 1.0, n)
        a = truth_a + rng_data.normal(0, 0.10, n)
        b = truth_b + rng_data.normal(0, 0.10, n)

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
        w_large = _ci_width(200)
        assert w_small > w_large, (
            f"CI with 15 labels ({w_small:.4f}) should be wider than "
            f"with 200 labels ({w_large:.4f})"
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
        """Wilcoxon: LLM inflates a by 2σ, no true diff → CI contains 0."""
        rng = np.random.default_rng(seed)
        a, b, al, bl = _paired(rng, n=250, mu_a=3.0, mu_b=3.0,
                               bias_a=2.0, bias_b=0.0, n_lab=70)
        r = wilcoxon(a, b, x_lab=al, y_lab=bl, n_boot=300, rng=seed + 10000)
        assert r.p_value < 0.001, f"seed={seed}: uncorrected should be significant"
        lo, hi = r.corrected_ci
        assert lo < 0 < hi, f"seed={seed}: wilcoxon corrected CI ({lo:.3f},{hi:.3f}) should contain 0"

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

    def test_all_items_labeled(self):
        """All items labeled → PPI still valid; n_labeled == n_total."""
        rng = np.random.default_rng(102)
        n = 80
        truth_a = rng.normal(3.5, 1.0, n)
        truth_b = rng.normal(3.0, 1.0, n)
        a = truth_a + rng.normal(0, 0.1, n)
        b = truth_b + rng.normal(0, 0.1, n)
        r = ttest(a, b, a_lab=truth_a, b_lab=truth_b, n_boot=200, rng=102)
        assert r.corrected_estimate is not None
        assert r.n_labeled == r.n_total

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

    def test_ttest_oracle_fully_labeled_matches_human_estimate(self):
        """With full labels, corrected estimate should match the human-data estimand."""
        rng = np.random.default_rng(250)
        n = 220
        truth_a = rng.normal(3.8, 0.9, n)
        truth_b = rng.normal(3.1, 0.9, n)
        a = truth_a + 0.7 + rng.normal(0, 0.08, n)
        b = truth_b - 0.4 + rng.normal(0, 0.08, n)

        r = ttest(a, b, a_lab=truth_a, b_lab=truth_b, n_boot=500, rng=250)
        human_diff = float(truth_a.mean() - truth_b.mean())
        assert r.corrected_estimate == pytest.approx(human_diff, abs=0.10)

    def test_ttest_ci_narrows_across_multiple_label_levels(self):
        """CI width should shrink across increasing label budgets (allow tiny jitter)."""
        rng_data = np.random.default_rng(260)
        n = 450
        truth_a = rng_data.normal(3.4, 1.0, n)
        truth_b = rng_data.normal(3.0, 1.0, n)
        a = truth_a + rng_data.normal(0, 0.10, n)
        b = truth_b + rng_data.normal(0, 0.10, n)

        widths = []
        for n_lab in [20, 50, 100, 160, 230]:
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
