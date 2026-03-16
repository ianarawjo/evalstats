"""Tests for bayes_binary routing — verifying the right method is invoked at the right moment.

Covers:
  - Public API exports: bayes_binary_ci_1d, bayes_paired_diff_ci, bayes_evals module
  - bayes_binary_ci_1d: basic correctness
  - bayes_paired_diff_ci: CI ordering, prob_a_greater bounds, symmetry
  - pairwise_differences with method='bayes_binary': test_method label, ValueError for
    non-binary data, seeded (R>=3) fallback to smooth_bootstrap
  - all_pairwise and vs_baseline with method='bayes_binary'
  - analyze() auto routing:
      · binary N < 100  → bayes_binary pairwise, Wilson advantage (n_bootstrap=0)
      · binary N >= 100 → newcombe pairwise,    Wilson advantage (n_bootstrap=0)
      · non-binary      → smooth_bootstrap, n_bootstrap > 0
  - analyze() explicit method='bayes_binary':
      · binary data   → bayes_binary pairwise, Wilson advantage
      · non-binary    → ValueError
  - Boundary: N=99 → bayes_binary, N=100 → newcombe
  - resolved_method on AnalysisBundle reflects pairwise method
  - compare_prompts routing for binary data (auto and explicit bayes_binary)
  - compare_models routing for binary data
  - Entity stats CIs match Wilson analytically for binary data
"""

from __future__ import annotations

import numpy as np
import pytest

import promptstats as ps
from promptstats.core.resampling import (
    bayes_binary_ci_1d,
    bayes_paired_diff_ci,
    is_binary_scores,
    wilson_ci_1d,
)
from promptstats.core.paired import pairwise_differences, all_pairwise, vs_baseline
from promptstats.core.router import analyze
from promptstats.core.types import BenchmarkResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _binary_scores(n_templates: int, m_inputs: int, probs: list[float] | None = None,
                   seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = probs or [0.5 + 0.1 * i for i in range(n_templates)]
    scores = np.stack([rng.binomial(1, p, m_inputs).astype(float) for p in probs])
    return scores


def _benchmark(scores: np.ndarray, labels: list[str]) -> BenchmarkResult:
    m = scores.shape[1]
    return BenchmarkResult(
        scores=scores,
        template_labels=labels,
        input_labels=[f"q{i}" for i in range(m)],
    )


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

def test_bayes_binary_ci_1d_is_exported():
    assert callable(ps.bayes_binary_ci_1d)


def test_bayes_paired_diff_ci_is_exported():
    assert callable(ps.bayes_paired_diff_ci)


def test_bayes_evals_module_is_exported():
    import promptstats
    assert hasattr(promptstats, "bayes_evals")
    mod = promptstats.bayes_evals
    for fn in ("get_bayes_posterior", "paired_comparisons", "binorm_cdf"):
        assert hasattr(mod, fn), f"bayes_evals missing {fn}"


# ---------------------------------------------------------------------------
# bayes_binary_ci_1d
# ---------------------------------------------------------------------------

def test_bayes_binary_ci_1d_ordering():
    values = np.array([1., 0., 1., 1., 0., 1., 0., 0.])
    lo, hi = bayes_binary_ci_1d(values, alpha=0.05)
    assert lo < hi
    assert 0.0 <= lo
    assert hi <= 1.0


def test_bayes_binary_ci_1d_brackets_sample_mean():
    values = np.array([1., 0., 1., 0., 1., 1., 0., 1., 1., 0.])
    lo, hi = bayes_binary_ci_1d(values, alpha=0.05)
    mean = values.mean()
    assert lo <= mean <= hi


def test_bayes_binary_ci_1d_all_ones():
    values = np.ones(10)
    lo, hi = bayes_binary_ci_1d(values, alpha=0.05)
    assert lo > 0.5
    assert hi <= 1.0


def test_bayes_binary_ci_1d_all_zeros():
    values = np.zeros(10)
    lo, hi = bayes_binary_ci_1d(values, alpha=0.05)
    assert lo >= 0.0
    assert hi < 0.5


def test_bayes_binary_ci_1d_narrows_with_more_data():
    small = np.array([1., 0., 1., 0., 1., 0.])
    large = np.tile(small, 20)
    lo_s, hi_s = bayes_binary_ci_1d(small, alpha=0.05)
    lo_l, hi_l = bayes_binary_ci_1d(large, alpha=0.05)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bayes_binary_ci_1d_90pct_narrower_than_95pct():
    values = np.array([1., 0., 1., 1., 0.])
    lo_90, hi_90 = bayes_binary_ci_1d(values, alpha=0.10)
    lo_95, hi_95 = bayes_binary_ci_1d(values, alpha=0.05)
    assert (hi_90 - lo_90) < (hi_95 - lo_95)


# ---------------------------------------------------------------------------
# bayes_paired_diff_ci
# ---------------------------------------------------------------------------

def test_bayes_paired_diff_ci_ordering():
    rng = _rng(1)
    a = rng.binomial(1, 0.7, 30).astype(float)
    b = rng.binomial(1, 0.5, 30).astype(float)
    lo, hi, prob = bayes_paired_diff_ci(a, b, alpha=0.05, rng=_rng(1))
    assert lo <= hi
    assert 0.0 <= prob <= 1.0


def test_bayes_paired_diff_ci_prob_a_greater_in_unit_interval():
    rng = _rng(2)
    a = rng.binomial(1, 0.6, 50).astype(float)
    b = rng.binomial(1, 0.4, 50).astype(float)
    _, _, prob = bayes_paired_diff_ci(a, b, alpha=0.05, rng=_rng(2))
    assert 0.0 <= prob <= 1.0


def test_bayes_paired_diff_ci_identical_arrays_straddles_zero():
    # When A == B the difference should be zero and CI should straddle 0
    a = np.array([1., 0., 1., 0., 1., 0., 1., 0.])
    b = a.copy()
    lo, hi, prob = bayes_paired_diff_ci(a, b, alpha=0.05, rng=_rng(5))
    assert lo <= 0.0 <= hi


def test_bayes_paired_diff_ci_a_dominates_gives_positive_ci():
    # A is always 1, B is always 0 → diff = 1.0, CI should be positive
    a = np.ones(20)
    b = np.zeros(20)
    lo, hi, prob = bayes_paired_diff_ci(a, b, alpha=0.05, rng=_rng(6))
    assert lo > 0.0
    assert hi <= 1.0
    assert prob > 0.9


def test_bayes_paired_diff_ci_antisymmetric():
    # P(A>B) and P(B>A) should be complementary (up to importance-sampling noise)
    rng = _rng(7)
    a = rng.binomial(1, 0.7, 40).astype(float)
    b = rng.binomial(1, 0.4, 40).astype(float)
    lo_ab, hi_ab, prob_ab = bayes_paired_diff_ci(a, b, alpha=0.05, rng=_rng(7))
    lo_ba, hi_ba, prob_ba = bayes_paired_diff_ci(b, a, alpha=0.05, rng=_rng(7))
    # Probability should roughly be complementary (stochastic — allow 5% slack)
    assert prob_ab + prob_ba == pytest.approx(1.0, abs=0.05)
    # CIs should be approximately sign-reversed (stochastic — allow 5% slack)
    np.testing.assert_allclose(lo_ab, -hi_ba, atol=0.05)
    np.testing.assert_allclose(hi_ab, -lo_ba, atol=0.05)


# ---------------------------------------------------------------------------
# pairwise_differences with method='bayes_binary'
# ---------------------------------------------------------------------------

def test_pairwise_differences_bayes_binary_test_method_label():
    scores = _binary_scores(2, 30, [0.7, 0.4], seed=10)
    result = pairwise_differences(scores, 0, 1, "A", "B",
                                  method="bayes_binary", ci=0.95, rng=_rng(10))
    assert "bayes binary" in result.test_method.lower()


def test_pairwise_differences_bayes_binary_ci_contains_point_diff():
    scores = _binary_scores(2, 40, [0.65, 0.45], seed=11)
    result = pairwise_differences(scores, 0, 1, "A", "B",
                                  method="bayes_binary", ci=0.95, rng=_rng(11))
    assert result.ci_low <= result.point_diff <= result.ci_high


def test_pairwise_differences_bayes_binary_p_value_in_bounds():
    scores = _binary_scores(2, 30, [0.7, 0.3], seed=12)
    result = pairwise_differences(scores, 0, 1, "A", "B",
                                  method="bayes_binary", ci=0.95, rng=_rng(12))
    assert 0.0 < result.p_value <= 1.0


def test_pairwise_differences_bayes_binary_raises_for_non_binary():
    rng = np.random.default_rng(13)
    scores = rng.uniform(0, 1, size=(2, 30))
    with pytest.raises(ValueError, match="binary"):
        pairwise_differences(scores, 0, 1, "A", "B",
                             method="bayes_binary", ci=0.95)


def test_pairwise_differences_bayes_binary_seeded_fallback_to_smooth():
    # R >= 3 runs → values are no longer binary → falls back to smooth_bootstrap
    rng = np.random.default_rng(14)
    scores = rng.binomial(1, 0.7, size=(2, 20, 5)).astype(float)
    result = pairwise_differences(scores, 0, 1, "A", "B",
                                  method="bayes_binary", ci=0.95, rng=_rng(14))
    assert "smooth" in result.test_method.lower()
    assert "bayes binary" not in result.test_method.lower()


def test_pairwise_differences_bayes_binary_identical_straddles_zero():
    a = np.array([1., 0., 1., 0., 1., 1., 0., 0., 1., 0.])
    b = a.copy()
    scores = np.stack([a, b])
    result = pairwise_differences(scores, 0, 1, "A", "B",
                                  method="bayes_binary", ci=0.95, rng=_rng(15))
    assert result.ci_low <= 0.0 <= result.ci_high


# ---------------------------------------------------------------------------
# all_pairwise with method='bayes_binary'
# ---------------------------------------------------------------------------

def test_all_pairwise_bayes_binary_smoke():
    scores = _binary_scores(3, 30, [0.7, 0.5, 0.3], seed=20)
    matrix = all_pairwise(scores, ["A", "B", "C"],
                          method="bayes_binary", ci=0.95, rng=_rng(20))
    for name_a, name_b in [("A", "B"), ("A", "C"), ("B", "C")]:
        r = matrix.get(name_a, name_b)
        assert "bayes binary" in r.test_method.lower()
        assert r.ci_low <= r.point_diff <= r.ci_high


def test_all_pairwise_bayes_binary_raises_for_non_binary():
    rng = np.random.default_rng(21)
    scores = rng.uniform(0, 1, size=(3, 25))
    with pytest.raises(ValueError, match="binary"):
        all_pairwise(scores, ["A", "B", "C"], method="bayes_binary", ci=0.95)


# ---------------------------------------------------------------------------
# vs_baseline with method='bayes_binary'
# ---------------------------------------------------------------------------

def test_vs_baseline_bayes_binary_smoke():
    scores = _binary_scores(3, 30, [0.5, 0.65, 0.35], seed=25)
    results = vs_baseline(scores, ["baseline", "A", "B"],
                          baseline="baseline",
                          method="bayes_binary", ci=0.95, rng=_rng(25))
    assert len(results) == 2  # A and B vs baseline
    for r in results:
        assert "bayes binary" in r.test_method.lower()
        assert r.ci_low <= r.point_diff <= r.ci_high


def test_vs_baseline_bayes_binary_raises_for_non_binary():
    rng = np.random.default_rng(26)
    scores = rng.uniform(0, 1, size=(3, 20))
    with pytest.raises(ValueError, match="binary"):
        vs_baseline(scores, ["baseline", "A", "B"],
                    baseline="baseline",
                    method="bayes_binary", ci=0.95)


# ---------------------------------------------------------------------------
# analyze() — auto routing for binary data
# ---------------------------------------------------------------------------

def test_analyze_auto_binary_small_n_pairwise_uses_bayes_binary():
    """N=50 < 100 → pairwise should use bayes_binary."""
    scores = _binary_scores(2, 50, [0.7, 0.5], seed=30)
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="auto", rng=_rng(30))
    pair = bundle.pairwise.get("A", "B")
    assert "bayes binary" in pair.test_method.lower()


def test_analyze_auto_binary_small_n_advantage_uses_wilson():
    """N=50 < 100 → advantage CI should use Wilson (n_bootstrap=0)."""
    scores = _binary_scores(2, 50, [0.7, 0.5], seed=31)
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="auto", rng=_rng(31))
    assert bundle.point_advantage.n_bootstrap == 0


def test_analyze_auto_binary_large_n_pairwise_uses_newcombe():
    """N=120 >= 100 → pairwise should use Newcombe."""
    scores = _binary_scores(2, 120, [0.7, 0.5], seed=32)
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="auto", rng=_rng(32))
    pair = bundle.pairwise.get("A", "B")
    assert "newcombe" in pair.test_method.lower()


def test_analyze_auto_binary_large_n_advantage_uses_wilson():
    """N=120 >= 100 → advantage CI should use Wilson (n_bootstrap=0)."""
    scores = _binary_scores(2, 120, [0.7, 0.5], seed=33)
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="auto", rng=_rng(33))
    assert bundle.point_advantage.n_bootstrap == 0


def test_analyze_auto_non_binary_uses_smooth_bootstrap():
    """Non-binary data → smooth_bootstrap pairwise and n_bootstrap > 0."""
    rng = np.random.default_rng(34)
    scores = rng.uniform(0, 1, size=(2, 40))
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="auto", rng=_rng(34))
    pair = bundle.pairwise.get("A", "B")
    assert "smooth" in pair.test_method.lower()
    assert bundle.point_advantage.n_bootstrap > 0


def test_analyze_auto_binary_resolved_method_is_bayes_binary():
    """resolved_method on the bundle should be 'bayes_binary' for small N."""
    scores = _binary_scores(2, 50, [0.7, 0.5], seed=35)
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="auto", rng=_rng(35))
    assert bundle.resolved_method == "bayes_binary"


def test_analyze_auto_binary_resolved_method_is_newcombe_for_large_n():
    """resolved_method on the bundle should be 'newcombe' for N >= 100."""
    scores = _binary_scores(2, 100, [0.7, 0.5], seed=36)
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="auto", rng=_rng(36))
    assert bundle.resolved_method == "newcombe"


# ---------------------------------------------------------------------------
# analyze() — boundary at N=99 vs N=100
# ---------------------------------------------------------------------------

def test_analyze_auto_boundary_99_uses_bayes_binary():
    scores = _binary_scores(2, 99, [0.6, 0.4], seed=40)
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="auto", rng=_rng(40))
    pair = bundle.pairwise.get("A", "B")
    assert "bayes binary" in pair.test_method.lower()
    assert bundle.resolved_method == "bayes_binary"


def test_analyze_auto_boundary_100_uses_newcombe():
    scores = _binary_scores(2, 100, [0.6, 0.4], seed=41)
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="auto", rng=_rng(41))
    pair = bundle.pairwise.get("A", "B")
    assert "newcombe" in pair.test_method.lower()
    assert bundle.resolved_method == "newcombe"


# ---------------------------------------------------------------------------
# analyze() — explicit method='bayes_binary'
# ---------------------------------------------------------------------------

def test_analyze_explicit_bayes_binary_with_binary_data():
    scores = _binary_scores(2, 40, [0.7, 0.4], seed=50)
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="bayes_binary", rng=_rng(50))
    pair = bundle.pairwise.get("A", "B")
    assert "bayes binary" in pair.test_method.lower()


def test_analyze_explicit_bayes_binary_advantage_uses_wilson():
    """Explicit bayes_binary → advantage CIs should still use Wilson (n_bootstrap=0)."""
    scores = _binary_scores(3, 40, [0.7, 0.5, 0.3], seed=51)
    bundle = analyze(_benchmark(scores, ["A", "B", "C"]),
                     method="bayes_binary", rng=_rng(51))
    assert bundle.point_advantage.n_bootstrap == 0


def test_analyze_explicit_bayes_binary_raises_for_non_binary():
    rng = np.random.default_rng(52)
    scores = rng.uniform(0, 1, size=(2, 30))
    with pytest.raises(ValueError, match="binary"):
        analyze(_benchmark(scores, ["A", "B"]),
                method="bayes_binary", rng=_rng(52))


def test_analyze_explicit_bayes_binary_resolved_method():
    scores = _binary_scores(2, 40, [0.7, 0.4], seed=53)
    bundle = analyze(_benchmark(scores, ["A", "B"]),
                     method="bayes_binary", rng=_rng(53))
    assert bundle.resolved_method == "bayes_binary"


def test_analyze_explicit_bayes_binary_three_way_all_pairs():
    """Three-way comparison with explicit bayes_binary should cover all pairs."""
    scores = _binary_scores(3, 40, [0.7, 0.5, 0.3], seed=54)
    bundle = analyze(_benchmark(scores, ["A", "B", "C"]),
                     method="bayes_binary", rng=_rng(54))
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
        r = bundle.pairwise.get(a, b)
        assert "bayes binary" in r.test_method.lower()
        assert 0.0 < r.p_value <= 1.0


# ---------------------------------------------------------------------------
# compare_prompts routing
# ---------------------------------------------------------------------------

def test_compare_prompts_auto_binary_small_n_pairwise_bayes_binary():
    """compare_prompts auto with binary N<100 → pairwise uses bayes_binary."""
    rng = np.random.default_rng(60)
    scores = {
        "A": rng.binomial(1, 0.7, 50).astype(float).tolist(),
        "B": rng.binomial(1, 0.4, 50).astype(float).tolist(),
    }
    report = ps.compare_prompts(scores, method="auto", rng=_rng(60))
    pair = report.pairwise.get("A", "B")
    assert "bayes binary" in pair.test_method.lower()


def test_compare_prompts_auto_binary_small_n_advantage_is_wilson():
    """compare_prompts auto with binary N<100 → advantage uses Wilson (n_bootstrap=0)."""
    rng = np.random.default_rng(61)
    scores = {
        "A": rng.binomial(1, 0.7, 50).astype(float).tolist(),
        "B": rng.binomial(1, 0.4, 50).astype(float).tolist(),
    }
    report = ps.compare_prompts(scores, method="auto", rng=_rng(61))
    assert report.full_analysis.point_advantage.n_bootstrap == 0


def test_compare_prompts_auto_binary_small_n_entity_stats_match_wilson():
    """Entity CIs for binary data should match Wilson CI analytically."""
    a_scores = np.array([1., 0., 1., 1., 0., 1., 0., 0., 1., 0.,
                         1., 1., 0., 1., 0., 1., 0., 1., 0., 1.])  # 11/20
    b_scores = np.array([0., 1., 1., 0., 0., 1., 0., 1., 0., 0.,
                         1., 0., 1., 0., 0., 1., 0., 0., 1., 0.])  # 8/20
    report = ps.compare_prompts(
        {"A": a_scores.tolist(), "B": b_scores.tolist()},
        method="auto", rng=_rng(62), n_bootstrap=500,
    )
    # Wilson CI for A: 11/20
    expected_lo_a, expected_hi_a = wilson_ci_1d(a_scores, alpha=0.05)
    np.testing.assert_allclose(
        report.prompt_stats["A"].ci_low, expected_lo_a, atol=1e-10,
    )
    np.testing.assert_allclose(
        report.prompt_stats["A"].ci_high, expected_hi_a, atol=1e-10,
    )


def test_compare_prompts_auto_binary_large_n_pairwise_newcombe():
    """compare_prompts auto with binary N>=100 → pairwise uses Newcombe."""
    rng = np.random.default_rng(63)
    scores = {
        "A": rng.binomial(1, 0.7, 110).astype(float).tolist(),
        "B": rng.binomial(1, 0.4, 110).astype(float).tolist(),
    }
    report = ps.compare_prompts(scores, method="auto", rng=_rng(63))
    pair = report.pairwise.get("A", "B")
    assert "newcombe" in pair.test_method.lower()


def test_compare_prompts_auto_binary_large_n_advantage_is_wilson():
    """compare_prompts auto with binary N>=100 → advantage uses Wilson (n_bootstrap=0)."""
    rng = np.random.default_rng(64)
    scores = {
        "A": rng.binomial(1, 0.7, 110).astype(float).tolist(),
        "B": rng.binomial(1, 0.4, 110).astype(float).tolist(),
    }
    report = ps.compare_prompts(scores, method="auto", rng=_rng(64))
    assert report.full_analysis.point_advantage.n_bootstrap == 0


def test_compare_prompts_explicit_bayes_binary_binary_data():
    rng = np.random.default_rng(65)
    scores = {
        "A": rng.binomial(1, 0.65, 40).astype(float).tolist(),
        "B": rng.binomial(1, 0.45, 40).astype(float).tolist(),
    }
    report = ps.compare_prompts(scores, method="bayes_binary", rng=_rng(65))
    pair = report.pairwise.get("A", "B")
    assert "bayes binary" in pair.test_method.lower()
    assert report.full_analysis.point_advantage.n_bootstrap == 0


def test_compare_prompts_explicit_bayes_binary_raises_for_non_binary():
    rng = np.random.default_rng(66)
    scores = {
        "A": rng.uniform(0, 1, 30).tolist(),
        "B": rng.uniform(0, 1, 30).tolist(),
    }
    with pytest.raises(ValueError, match="binary"):
        ps.compare_prompts(scores, method="bayes_binary", rng=_rng(66))


def test_compare_prompts_auto_non_binary_uses_smooth_bootstrap():
    rng = np.random.default_rng(67)
    scores = {
        "A": rng.uniform(0, 1, 30).tolist(),
        "B": rng.uniform(0, 1, 30).tolist(),
    }
    report = ps.compare_prompts(scores, method="auto", n_bootstrap=300, rng=_rng(67))
    pair = report.pairwise.get("A", "B")
    assert "smooth" in pair.test_method.lower()
    assert report.full_analysis.point_advantage.n_bootstrap > 0


# ---------------------------------------------------------------------------
# compare_models routing
# ---------------------------------------------------------------------------

def test_compare_models_auto_binary_small_n_pairwise_bayes_binary():
    """compare_models auto with binary N<100 → pairwise uses bayes_binary."""
    rng = np.random.default_rng(70)
    scores = {
        "model_a": rng.binomial(1, 0.7, 50).astype(float).tolist(),
        "model_b": rng.binomial(1, 0.4, 50).astype(float).tolist(),
    }
    report = ps.compare_models(scores, method="auto", rng=_rng(70))
    pair = report.pairwise.get("model_a", "model_b")
    assert "bayes binary" in pair.test_method.lower()


def test_compare_models_auto_binary_small_n_advantage_is_wilson():
    """compare_models auto with binary N<100 → advantage uses Wilson (n_bootstrap=0)."""
    rng = np.random.default_rng(71)
    scores = {
        "model_a": rng.binomial(1, 0.7, 50).astype(float).tolist(),
        "model_b": rng.binomial(1, 0.4, 50).astype(float).tolist(),
    }
    report = ps.compare_models(scores, method="auto", rng=_rng(71))
    assert report.full_analysis.model_level.point_advantage.n_bootstrap == 0


def test_compare_models_auto_binary_large_n_pairwise_newcombe():
    """compare_models auto with binary N>=100 → pairwise uses Newcombe."""
    rng = np.random.default_rng(72)
    scores = {
        "model_a": rng.binomial(1, 0.7, 110).astype(float).tolist(),
        "model_b": rng.binomial(1, 0.4, 110).astype(float).tolist(),
    }
    report = ps.compare_models(scores, method="auto", rng=_rng(72))
    pair = report.pairwise.get("model_a", "model_b")
    assert "newcombe" in pair.test_method.lower()


def test_compare_models_auto_binary_entity_stats_match_wilson():
    """Entity CIs for binary data should match Wilson CI analytically."""
    a_scores = np.array([1., 0., 1., 1., 0., 1., 0., 0., 1., 0.,
                         1., 1., 0., 1., 0., 1., 0., 1., 0., 1.])  # 11/20
    b_scores = np.array([0., 1., 1., 0., 0., 1., 0., 1., 0., 0.,
                         1., 0., 1., 0., 0., 1., 0., 0., 1., 0.])  # 8/20
    report = ps.compare_models(
        {"model_a": a_scores.tolist(), "model_b": b_scores.tolist()},
        method="auto", rng=_rng(73), n_bootstrap=500,
    )
    expected_lo, expected_hi = wilson_ci_1d(a_scores, alpha=0.05)
    np.testing.assert_allclose(
        report.model_stats["model_a"].ci_low, expected_lo, atol=1e-10,
    )
    np.testing.assert_allclose(
        report.model_stats["model_a"].ci_high, expected_hi, atol=1e-10,
    )


def test_compare_models_explicit_bayes_binary_flat_raises():
    """Explicit bayes_binary with flat 1D per-model arrays raises ValueError.

    Flat per-model arrays are stacked as the 'runs' dimension in the
    template-level analysis.  With N_models=2 < 3, inputs are pre-averaged
    across models (binary → 0/0.5/1), producing non-binary cell means.
    bayes_binary validation then correctly rejects the data.
    """
    rng = np.random.default_rng(74)
    scores = {
        "model_a": rng.binomial(1, 0.65, 40).astype(float).tolist(),
        "model_b": rng.binomial(1, 0.45, 40).astype(float).tolist(),
    }
    with pytest.raises(ValueError, match="binary"):
        ps.compare_models(scores, method="bayes_binary", rng=_rng(74))


def test_compare_models_explicit_bayes_binary_raises_for_non_binary():
    rng = np.random.default_rng(75)
    scores = {
        "model_a": rng.uniform(0, 1, 30).tolist(),
        "model_b": rng.uniform(0, 1, 30).tolist(),
    }
    with pytest.raises(ValueError, match="binary"):
        ps.compare_models(scores, method="bayes_binary", rng=_rng(75))
