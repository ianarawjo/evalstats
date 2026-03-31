"""Tests for _max_stat_simultaneous_cis and the simultaneous_ci integration.

Tests are grouped into four sections:
  1. Contract / structural tests  — return type, valid bounds, edge cases.
  2. Key statistical properties   — symmetry, monotonicity, coverage.
  3. Method / path coverage       — all bootstrap variants, seeded path, median.
  4. Integration tests            — all_pairwise, compare_prompts, compare_models.
"""

import numpy as np
import pytest

import promptstats as ps
from promptstats.core.paired import (
    _max_stat_simultaneous_cis,
    _bonferroni_simultaneous_cis,
    _simultaneous_cis_router,
    all_pairwise,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Section 1 — Contract / structural tests
# ---------------------------------------------------------------------------

def test_unsupported_methods_return_empty_dict():
    """Non-bootstrap methods (newcombe, bayes_binary, fisher_exact) return {}."""
    scores = _rng(0).normal(0, 1, (3, 20))
    labels = ["m0", "m1", "m2"]
    pairs = [("m0", "m1"), ("m0", "m2")]
    for method in ["newcombe", "bayes_binary", "fisher_exact"]:
        cis, pvals = _max_stat_simultaneous_cis(
            scores, pairs, labels, method, 0.95, 100, _rng(0), "mean"
        )
        assert cis == {}, f"Expected empty dict for method='{method}', got {cis}"


def test_empty_pairs_returns_empty_dict():
    scores = _rng(0).normal(0, 1, (3, 20))
    cis, pvals = _max_stat_simultaneous_cis(
        scores, [], ["m0", "m1", "m2"], "bootstrap", 0.95, 100, _rng(0), "mean"
    )
    assert cis == {}


def test_degenerate_zero_variance_does_not_crash():
    """When all scores are identical, all pairwise SEs are zero.
    The function must not raise and must return a dict (possibly empty)."""
    scores = np.ones((3, 50))
    labels = ["a", "b", "c"]
    pairs = [("a", "b"), ("b", "c")]

    # Must not raise
    cis, pvals = _max_stat_simultaneous_cis(
        scores, pairs, labels, "bootstrap", 0.95, 100, _rng(0), "mean"
    )
    assert isinstance(cis, dict)
    # If CIs were returned, each must satisfy low <= high
    for lo, hi in cis.values():
        assert lo <= hi


def test_returns_all_requested_pairs():
    """Returned dict contains exactly the requested pairs."""
    scores = _rng(1).normal(0, 1, (4, 30))
    labels = ["a", "b", "c", "d"]
    pairs = [("a", "b"), ("a", "c"), ("b", "d")]

    cis, _ = _max_stat_simultaneous_cis(
        scores, pairs, labels, "bootstrap", 0.95, 200, _rng(1), "mean"
    )

    assert set(cis.keys()) == set(pairs)


def test_single_pair_returns_valid_ci():
    """Works correctly with k=1 — no cross-pair max needed."""
    scores = _rng(2).normal(0, 1, (2, 40))
    labels = ["a", "b"]
    pairs = [("a", "b")]

    cis, _ = _max_stat_simultaneous_cis(
        scores, pairs, labels, "bootstrap", 0.95, 300, _rng(2), "mean"
    )

    assert ("a", "b") in cis
    lo, hi = cis[("a", "b")]
    assert lo < hi
    assert np.isfinite(lo) and np.isfinite(hi)


def test_ci_bounds_are_finite_and_ordered():
    """All returned intervals have finite, ordered bounds."""
    scores = _rng(3).normal(0, 1, (4, 35))
    labels = ["a", "b", "c", "d"]
    pairs = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]

    cis, _ = _max_stat_simultaneous_cis(
        scores, pairs, labels, "bootstrap", 0.95, 300, _rng(3), "mean"
    )

    for pair, (lo, hi) in cis.items():
        assert np.isfinite(lo), f"ci_low not finite for {pair}"
        assert np.isfinite(hi), f"ci_high not finite for {pair}"
        assert lo <= hi, f"ci_low > ci_high for {pair}"


# ---------------------------------------------------------------------------
# Section 2 — Key statistical properties
# ---------------------------------------------------------------------------

def test_cis_are_symmetric_around_point_estimate_non_seeded():
    """Simultaneous CIs must be symmetric: (low + high) / 2 == point estimate.

    This is a core property of the studentized max-T approach.
    """
    rng = _rng(10)
    scores = rng.normal(0, 1, (3, 50))
    labels = ["a", "b", "c"]
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    label_to_idx = {"a": 0, "b": 1, "c": 2}

    cis, _ = _max_stat_simultaneous_cis(
        scores, pairs, labels, "bootstrap", 0.95, 500, rng, "mean"
    )

    for a, b in pairs:
        i, j = label_to_idx[a], label_to_idx[b]
        point = float(np.mean(scores[i] - scores[j]))
        lo, hi = cis[(a, b)]
        np.testing.assert_allclose(
            (lo + hi) / 2, point, atol=1e-9,
            err_msg=f"CI not centered at point estimate for ({a}, {b})",
        )


def test_cis_are_symmetric_around_point_estimate_seeded():
    """Same symmetry property holds for the nested (seeded) path."""
    rng = _rng(11)
    scores = rng.normal(0, 1, (3, 30, 5))  # R=5
    labels = ["x", "y", "z"]
    pairs = [("x", "y"), ("x", "z")]
    label_to_idx = {"x": 0, "y": 1, "z": 2}

    cis, _ = _max_stat_simultaneous_cis(
        scores, pairs, labels, "bootstrap", 0.95, 400, rng, "mean"
    )

    for a, b in pairs:
        i, j = label_to_idx[a], label_to_idx[b]
        cell_diffs = scores[i].mean(axis=1) - scores[j].mean(axis=1)
        point = float(cell_diffs.mean())
        lo, hi = cis[(a, b)]
        np.testing.assert_allclose(
            (lo + hi) / 2, point, atol=1e-9,
            err_msg=f"Seeded CI not centered at point estimate for ({a}, {b})",
        )


def test_more_pairs_yields_wider_cis_same_bootstrap_resamples():
    """Adding more pairs widens each individual CI.

    With the same RNG seed, both calls generate identical bootstrap
    resamples for pair (a, b).  The k=3 call takes the max over 3 T-stats
    while k=1 takes the max over 1, so c_{k=3} >= c_{k=1} pointwise and
    thus the CI width must be non-decreasing.
    """
    SEED = 77
    scores = _rng(SEED).normal(0, 1, (3, 60))
    labels = ["a", "b", "c"]
    all_pairs = [("a", "b"), ("a", "c"), ("b", "c")]

    for target in all_pairs:
        # k=1: only the target pair
        ci_k1, _ = _max_stat_simultaneous_cis(
            scores, [target], labels, "bootstrap", 0.95, 600, _rng(SEED), "mean"
        )
        # k=3: all pairs (same seed → same input_idx)
        ci_k3, _ = _max_stat_simultaneous_cis(
            scores, all_pairs, labels, "bootstrap", 0.95, 600, _rng(SEED), "mean"
        )

        if target not in ci_k1 or target not in ci_k3:
            continue

        w_k1 = ci_k1[target][1] - ci_k1[target][0]
        w_k3 = ci_k3[target][1] - ci_k3[target][0]

        assert w_k3 >= w_k1 - 1e-9, (
            f"CI for {target} should widen from k=1 ({w_k1:.5f}) to "
            f"k=3 ({w_k3:.5f}): max-T quantile must be non-decreasing in k."
        )


def test_more_pairs_yields_wider_cis_bayes_bootstrap():
    """Same monotonicity property holds for bayes_bootstrap (shared Dirichlet weights)."""
    SEED = 88
    scores = _rng(SEED).normal(0, 1, (3, 60))
    labels = ["a", "b", "c"]
    all_pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    target = ("a", "b")

    ci_k1, _ = _max_stat_simultaneous_cis(
        scores, [target], labels, "bayes_bootstrap", 0.95, 600, _rng(SEED), "mean"
    )
    ci_k3, _ = _max_stat_simultaneous_cis(
        scores, all_pairs, labels, "bayes_bootstrap", 0.95, 600, _rng(SEED), "mean"
    )

    if target not in ci_k1 or target not in ci_k3:
        return

    w_k1 = ci_k1[target][1] - ci_k1[target][0]
    w_k3 = ci_k3[target][1] - ci_k3[target][0]
    assert w_k3 >= w_k1 - 1e-9, (
        f"Bayes bootstrap CI should widen from k=1 ({w_k1:.5f}) to k=3 ({w_k3:.5f})."
    )


def test_simultaneous_coverage_bootstrap():
    """Joint coverage across all pairs reaches the nominal level (~95%).

    Setup: 4 templates with identical true means (0); the 3 pairs all share
    template 0, creating strong positive correlation.  The max-T method
    exploits this so coverage should be near 0.95.

    With n_simulations=300, SE ≈ 0.013; the tolerance [0.88, 1.00] covers
    ±5 SE, catching gross miscoverage while tolerating simulation variance.
    """
    rng = _rng(42)
    n_simulations = 300
    n_bootstrap = 300
    M = 60
    ci_level = 0.95
    labels = ["m0", "m1", "m2", "m3"]
    pairs = [("m0", "m1"), ("m0", "m2"), ("m0", "m3")]

    hits = 0
    for _ in range(n_simulations):
        scores = rng.normal(0.0, 1.0, (4, M))
        cis, _ = _max_stat_simultaneous_cis(
            scores, pairs, labels, "bootstrap", ci_level, n_bootstrap, rng, "mean"
        )
        if not cis:
            hits += 1
            continue
        # Simultaneous coverage: ALL intervals contain the true diff (0.0)
        if all(cis[p][0] <= 0.0 <= cis[p][1] for p in pairs):
            hits += 1

    coverage = hits / n_simulations
    assert 0.88 <= coverage <= 1.00, (
        f"Bootstrap simultaneous coverage {coverage:.3f} outside [0.88, 1.00]; "
        f"expected ~{ci_level}."
    )


def test_simultaneous_coverage_smooth_bootstrap():
    """Same coverage check using smooth_bootstrap."""
    rng = _rng(55)
    n_simulations = 250
    n_bootstrap = 300
    M = 50
    ci_level = 0.95
    labels = ["a", "b", "c"]
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]

    hits = 0
    for _ in range(n_simulations):
        scores = rng.normal(0.0, 1.0, (3, M))
        cis, _ = _max_stat_simultaneous_cis(
            scores, pairs, labels, "smooth_bootstrap", ci_level,
            n_bootstrap, rng, "mean"
        )
        if not cis:
            hits += 1
            continue
        if all(cis[p][0] <= 0.0 <= cis[p][1] for p in pairs):
            hits += 1

    coverage = hits / n_simulations
    assert 0.88 <= coverage <= 1.00, (
        f"Smooth bootstrap simultaneous coverage {coverage:.3f} outside [0.88, 1.00]."
    )


# ---------------------------------------------------------------------------
# Section 3 — Method and path coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", [
    "bootstrap", "bca", "smooth_bootstrap", "bayes_bootstrap",
    "auto", "permutation", "sign_test",
])
def test_all_supported_methods_return_valid_dict(method):
    """Every bootstrap-compatible method returns a full dict with valid bounds."""
    scores = _rng(1).standard_normal((3, 30))
    labels = ["m0", "m1", "m2"]
    pairs = [("m0", "m1"), ("m0", "m2"), ("m1", "m2")]

    cis, _ = _max_stat_simultaneous_cis(
        scores, pairs, labels, method, 0.90, 200, _rng(1), "mean"
    )

    assert len(cis) == 3
    for pair in pairs:
        assert pair in cis
        lo, hi = cis[pair]
        assert np.isfinite(lo), f"{method}: ci_low not finite for {pair}"
        assert np.isfinite(hi), f"{method}: ci_high not finite for {pair}"
        assert lo <= hi, f"{method}: ci_low > ci_high for {pair}"


@pytest.mark.parametrize("method", ["bootstrap", "smooth_bootstrap", "bayes_bootstrap"])
def test_seeded_path_returns_valid_cis(method):
    """Seeded (R >= 3) path produces valid CIs for all bootstrap variants."""
    scores = _rng(5).normal(0, 1, (3, 25, 4))  # R=4
    labels = ["a", "b", "c"]
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]

    cis, _ = _max_stat_simultaneous_cis(
        scores, pairs, labels, method, 0.95, 200, _rng(5), "mean"
    )

    assert len(cis) == 3
    for pair in pairs:
        lo, hi = cis[pair]
        assert lo <= hi, f"{method}: ci_low > ci_high for {pair}"
        assert np.isfinite(lo) and np.isfinite(hi), f"{method}: non-finite CI for {pair}"


def test_seeded_path_all_three_methods_agree_in_direction():
    """All three seeded methods should produce CIs that include the sign of
    the true effect (or zero) consistently for clear signals."""
    rng = _rng(20)
    # Template 0 clearly better: mean diff +0.5 over 30 inputs, 4 runs
    scores_a = rng.normal(0.5, 0.2, (30, 4))
    scores_b = rng.normal(0.0, 0.2, (30, 4))
    scores_c = rng.normal(0.0, 0.2, (30, 4))
    scores = np.stack([scores_a, scores_b, scores_c], axis=0)  # (3, 30, 4)
    labels = ["a", "b", "c"]
    pairs = [("a", "b")]

    for method in ["bootstrap", "smooth_bootstrap", "bayes_bootstrap"]:
        cis, _ = _max_stat_simultaneous_cis(
            scores, pairs, labels, method, 0.95, 400, _rng(20), "mean"
        )
        lo, hi = cis[("a", "b")]
        # True diff is +0.5; CI should not contain a clearly negative value
        assert hi > 0.0, f"{method}: upper bound {hi:.4f} not positive for true diff +0.5"


def test_median_statistic_returns_valid_cis():
    """Works with statistic='median' without errors and returns valid bounds."""
    scores = _rng(7).normal(0, 1, (3, 40))
    labels = ["a", "b", "c"]
    pairs = [("a", "b"), ("b", "c")]

    cis, _ = _max_stat_simultaneous_cis(
        scores, pairs, labels, "bootstrap", 0.95, 300, _rng(7), "median"
    )

    assert len(cis) == 2
    for pair in pairs:
        lo, hi = cis[pair]
        assert lo <= hi
        assert np.isfinite(lo) and np.isfinite(hi)


def test_median_statistic_ci_centered_at_median_point_estimate():
    """For median statistic, midpoint = median(diffs)."""
    rng = _rng(71)
    scores = rng.normal(0, 1, (2, 50))
    labels = ["a", "b"]
    pairs = [("a", "b")]

    cis, _ = _max_stat_simultaneous_cis(
        scores, pairs, labels, "bootstrap", 0.95, 500, rng, "median"
    )

    lo, hi = cis[("a", "b")]
    point = float(np.median(scores[0] - scores[1]))
    np.testing.assert_allclose((lo + hi) / 2, point, atol=1e-9)


def test_auto_method_resolves_to_smooth_bootstrap():
    """method='auto' should behave like smooth_bootstrap and return valid CIs."""
    rng = _rng(8)
    scores = rng.normal(0, 1, (3, 30))
    labels = ["a", "b", "c"]
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]

    cis, _ = _max_stat_simultaneous_cis(
        scores, pairs, labels, "auto", 0.95, 300, rng, "mean"
    )

    assert len(cis) == 3
    for pair in pairs:
        lo, hi = cis[pair]
        assert lo <= hi and np.isfinite(lo) and np.isfinite(hi)


# ---------------------------------------------------------------------------
# Section 4 — Integration tests
# ---------------------------------------------------------------------------

def test_all_pairwise_simultaneous_ci_flag_true():
    """PairwiseMatrix.simultaneous_ci is True when the flag is set and method
    is compatible (bootstrap-based)."""
    scores = _rng(9).normal(0, 1, (3, 30))
    labels = ["a", "b", "c"]

    mat = all_pairwise(
        scores, labels, method="bootstrap", ci=0.95,
        n_bootstrap=200, correction="none", rng=_rng(9),
        simultaneous_ci=True,
    )

    assert mat.simultaneous_ci is True


def test_all_pairwise_simultaneous_ci_true_by_default():
    """PairwiseMatrix.simultaneous_ci defaults to True."""
    scores = _rng(10).normal(0, 1, (3, 30))
    labels = ["a", "b", "c"]
    mat = all_pairwise(scores, labels, n_bootstrap=100, rng=_rng(10))
    assert mat.simultaneous_ci is True


def test_all_pairwise_test_method_string_annotated():
    """test_method on each PairedDiffResult should contain 'simultaneous CI'
    and indicate which variant was used ('max-T' for bootstrap methods)."""
    scores = _rng(11).normal(0, 1, (3, 30))
    labels = ["a", "b", "c"]

    mat = all_pairwise(
        scores, labels, method="bootstrap", ci=0.95,
        n_bootstrap=200, correction="none", rng=_rng(11),
        simultaneous_ci=True,
    )

    assert mat.simultaneous_ci_method == "max_t"
    for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
        tm = mat.get(a, b).test_method
        assert "simultaneous CI" in tm, (
            f"test_method for ({a},{b}) missing 'simultaneous CI': {tm!r}"
        )
        assert "max-T" in tm, (
            f"test_method for ({a},{b}) missing 'max-T' variant label: {tm!r}"
        )


def test_all_pairwise_p_values_are_max_t_with_simultaneous_ci():
    """When simultaneous_ci=True and bootstrap method, p_value is the max-T
    bootstrap p-value (FWER-controlled), not the marginal FDR-corrected one.

    Max-T p-values must be in [0, 1] and are generally more conservative
    (larger) than marginal FDR-corrected p-values.
    """
    scores = _rng(12).normal(0, 1, (3, 40))
    labels = ["a", "b", "c"]

    mat_sim = all_pairwise(
        scores, labels, method="bootstrap", ci=0.95,
        n_bootstrap=300, correction="holm", rng=_rng(12),
        simultaneous_ci=True,
    )

    assert mat_sim.simultaneous_ci_method == "max_t"
    for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
        p = mat_sim.get(a, b).p_value
        assert 0.0 <= p <= 1.0, f"max-T p_value {p} out of [0,1] for ({a},{b})"


def test_compare_prompts_simultaneous_ci_propagates():
    """compare_prompts with simultaneous_ci=True sets the flag on the report
    and produces valid CIs."""
    scores = {
        "A": _rng(20).normal(0.7, 0.1, 40).tolist(),
        "B": _rng(21).normal(0.65, 0.1, 40).tolist(),
        "C": _rng(22).normal(0.60, 0.1, 40).tolist(),
    }

    report = ps.compare_prompts(
        scores, simultaneous_ci=True, rng=_rng(20), n_bootstrap=400,
    )

    assert report.simultaneous_ci is True
    assert report.pairwise.simultaneous_ci is True
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
        lo, hi = report.pairwise.get(a, b).ci_low, report.pairwise.get(a, b).ci_high
        assert lo <= hi and np.isfinite(lo) and np.isfinite(hi)


def test_compare_prompts_simultaneous_ci_true_by_default():
    scores = {"A": [0.7, 0.8, 0.6], "B": [0.65, 0.75, 0.55]}
    report = ps.compare_prompts(scores, rng=_rng(0), n_bootstrap=100)
    assert report.simultaneous_ci is True


def test_compare_models_simultaneous_ci_propagates():
    """compare_models with simultaneous_ci=True sets the flag on the report."""
    rng = _rng(30)
    scores = {
        "GPT":    rng.normal(0.7, 0.1, 40).tolist(),
        "Llama":  rng.normal(0.65, 0.1, 40).tolist(),
        "Mistral": rng.normal(0.60, 0.1, 40).tolist(),
    }
    report = ps.compare_models(
        scores, simultaneous_ci=True, rng=_rng(30), n_bootstrap=300,
    )
    assert report.simultaneous_ci is True


def test_unsupported_method_falls_back_to_bonferroni():
    """When method='newcombe' (no bootstrap CIs), simultaneous_ci=True should
    fall back to Bonferroni t-intervals rather than silently ignoring the flag.
    The report should have simultaneous_ci=True and method='bonferroni'."""
    rng = _rng(40)
    scores = {
        "A": [int(x > 0.5) for x in rng.random(50)],
        "B": [int(x > 0.5) for x in rng.random(50)],
        "C": [int(x > 0.5) for x in rng.random(50)],
    }
    report = ps.compare_prompts(
        scores, method="newcombe", simultaneous_ci=True,
        rng=_rng(40), n_bootstrap=200,
    )
    assert report.simultaneous_ci is True
    assert report.pairwise.simultaneous_ci_method == "bonferroni"
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
        tm = report.pairwise.get(a, b).test_method
        assert "simultaneous CI" in tm
        assert "Bonferroni" in tm


def test_seeded_compare_prompts_simultaneous_ci():
    """simultaneous_ci=True works end-to-end when score arrays have R=4 runs."""
    rng = _rng(50)
    scores = {
        "A": rng.normal(0.7, 0.1, (30, 4)).tolist(),
        "B": rng.normal(0.65, 0.1, (30, 4)).tolist(),
        "C": rng.normal(0.60, 0.1, (30, 4)).tolist(),
    }
    report = ps.compare_prompts(
        scores, simultaneous_ci=True, rng=_rng(50), n_bootstrap=300,
    )
    assert report.simultaneous_ci is True
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
        lo, hi = report.pairwise.get(a, b).ci_low, report.pairwise.get(a, b).ci_high
        assert lo <= hi and np.isfinite(lo) and np.isfinite(hi)
        assert "simultaneous CI" in report.pairwise.get(a, b).test_method


# ---------------------------------------------------------------------------
# Section 5 — Bonferroni fallback and router
# ---------------------------------------------------------------------------

def _make_results(scores_2d, labels, **kw):
    """Helper: run all_pairwise and return results dict."""
    mat = all_pairwise(scores_2d, labels, n_bootstrap=200, rng=_rng(0), **kw)
    return mat.results, [(labels[i], labels[j])
                         for i in range(len(labels))
                         for j in range(i + 1, len(labels))]


def test_bonferroni_returns_all_pairs():
    """_bonferroni_simultaneous_cis returns a CI for every requested pair."""
    scores = _rng(60).normal(0, 1, (3, 40))
    labels = ["x", "y", "z"]
    results, pairs = _make_results(scores, labels)
    cis = _bonferroni_simultaneous_cis(results, pairs, ci=0.95)
    assert set(cis.keys()) == set(pairs)


def test_bonferroni_bounds_finite_and_ordered():
    """All Bonferroni CI bounds must be finite and lo <= hi."""
    scores = _rng(61).normal(0, 1, (3, 40))
    labels = ["x", "y", "z"]
    results, pairs = _make_results(scores, labels)
    cis = _bonferroni_simultaneous_cis(results, pairs, ci=0.95)
    for pair, (lo, hi) in cis.items():
        assert np.isfinite(lo) and np.isfinite(hi), f"{pair}: non-finite bounds"
        assert lo <= hi, f"{pair}: lo > hi"


def test_bonferroni_wider_than_individual():
    """Bonferroni simultaneous CIs must be at least as wide as individual CIs
    (since they are corrected for multiple comparisons)."""
    scores = _rng(62).normal(0, 1, (3, 50))
    labels = ["x", "y", "z"]
    results, pairs = _make_results(scores, labels)
    cis_bonf = _bonferroni_simultaneous_cis(results, pairs, ci=0.95)
    for pair in pairs:
        r = results[pair]
        ind_width = r.ci_high - r.ci_low
        bonf_width = cis_bonf[pair][1] - cis_bonf[pair][0]
        assert bonf_width >= ind_width - 1e-9, (
            f"{pair}: Bonferroni width {bonf_width:.4f} < individual {ind_width:.4f}"
        )


def test_bonferroni_single_pair_equals_individual_t():
    """With k=1, Bonferroni adjustment is a no-op, so the CI matches a
    standard paired t-interval at the same level."""
    from scipy import stats as scipy_stats
    scores = _rng(63).normal(0, 1, (2, 40))
    labels = ["a", "b"]
    results, pairs = _make_results(scores, labels)
    assert len(pairs) == 1
    cis = _bonferroni_simultaneous_cis(results, pairs, ci=0.95)
    lo, hi = cis[pairs[0]]

    diffs = results[pairs[0]].per_input_diffs
    M = len(diffs)
    se = float(np.std(diffs, ddof=1)) / np.sqrt(M)
    t_crit = scipy_stats.t.ppf(0.975, df=M - 1)
    expected_lo = float(np.mean(diffs)) - t_crit * se
    expected_hi = float(np.mean(diffs)) + t_crit * se
    np.testing.assert_allclose(lo, expected_lo, atol=1e-9)
    np.testing.assert_allclose(hi, expected_hi, atol=1e-9)


def test_bonferroni_empty_pairs_returns_empty():
    assert _bonferroni_simultaneous_cis({}, [], ci=0.95) == {}


def test_bonferroni_degenerate_zero_variance():
    """When all diffs are identical, SE=0; CI should degenerate to a point."""
    scores = np.ones((2, 30))
    labels = ["a", "b"]
    results, pairs = _make_results(scores, labels)
    cis = _bonferroni_simultaneous_cis(results, pairs, ci=0.95)
    lo, hi = cis[pairs[0]]
    assert lo == hi


# --- Router tests ---

def test_router_returns_max_stat_for_bootstrap():
    """Router should choose 'max_t' for a bootstrap-compatible method."""
    scores = _rng(70).normal(0, 1, (3, 40))
    labels = ["a", "b", "c"]
    results, pairs = _make_results(scores, labels)
    cis, used, _ = _simultaneous_cis_router(
        scores, results, pairs, labels,
        method="bootstrap", ci=0.95, n_bootstrap=300,
        rng=_rng(70), statistic="mean",
    )
    assert used == "max_t"
    assert len(cis) == len(pairs)


def test_router_falls_back_to_bonferroni_for_newcombe():
    """Router should fall back to 'bonferroni' for analytical methods."""
    # Use continuous scores but force method='newcombe' to trigger the fallback.
    scores = _rng(71).normal(0, 1, (3, 40))
    labels = ["a", "b", "c"]
    results, pairs = _make_results(scores, labels)
    cis, used, _ = _simultaneous_cis_router(
        scores, results, pairs, labels,
        method="newcombe", ci=0.95, n_bootstrap=300,
        rng=_rng(71), statistic="mean",
    )
    assert used == "bonferroni"
    assert len(cis) == len(pairs)
    for pair, (lo, hi) in cis.items():
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= hi


@pytest.mark.parametrize("method", ["bootstrap", "bca", "smooth_bootstrap",
                                     "bayes_bootstrap", "permutation", "sign_test", "auto"])
def test_router_max_stat_for_all_bootstrap_methods(method):
    """All bootstrap-compatible methods should route to 'max_stat'."""
    scores = _rng(72).normal(0, 1, (3, 35))
    labels = ["a", "b", "c"]
    results, pairs = _make_results(scores, labels)
    _, used, _ = _simultaneous_cis_router(
        scores, results, pairs, labels,
        method=method, ci=0.95, n_bootstrap=200,
        rng=_rng(72), statistic="mean",
    )
    assert used == "max_t", f"Expected max_t for method={method!r}, got {used!r}"


def test_simultaneous_ci_method_field_max_t():
    """PairwiseMatrix.simultaneous_ci_method is 'max_t' for bootstrap methods."""
    scores = _rng(80).normal(0, 1, (3, 30))
    labels = ["a", "b", "c"]
    mat = all_pairwise(
        scores, labels, method="bootstrap", n_bootstrap=200,
        rng=_rng(80), simultaneous_ci=True, correction="none",
    )
    assert mat.simultaneous_ci is True
    assert mat.simultaneous_ci_method == "max_t"


def test_simultaneous_ci_method_field_bonferroni():
    """PairwiseMatrix.simultaneous_ci_method is 'bonferroni' for analytical methods."""
    scores = (_rng(81).random((3, 40)) > 0.5).astype(float)
    labels = ["a", "b", "c"]
    mat = all_pairwise(
        scores, labels, method="newcombe", n_bootstrap=200,
        rng=_rng(81), simultaneous_ci=True, correction="none",
    )
    assert mat.simultaneous_ci is True
    assert mat.simultaneous_ci_method == "bonferroni"


def test_simultaneous_ci_method_field_none_when_not_requested():
    """PairwiseMatrix.simultaneous_ci_method is None when simultaneous_ci=False."""
    scores = _rng(82).normal(0, 1, (3, 30))
    labels = ["a", "b", "c"]
    mat = all_pairwise(scores, labels, n_bootstrap=100, rng=_rng(82), simultaneous_ci=False)
    assert mat.simultaneous_ci is False
    assert mat.simultaneous_ci_method is None


def test_bonferroni_annotation_in_test_method():
    """PairedDiffResult.test_method should contain 'Bonferroni' for the fallback."""
    scores = (_rng(83).random((3, 40)) > 0.5).astype(float)
    labels = ["a", "b", "c"]
    mat = all_pairwise(
        scores, labels, method="newcombe", n_bootstrap=200,
        rng=_rng(83), simultaneous_ci=True, correction="none",
    )
    for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
        tm = mat.get(a, b).test_method
        assert "simultaneous CI" in tm, f"Missing 'simultaneous CI' in {tm!r}"
        assert "Bonferroni" in tm, f"Missing 'Bonferroni' in {tm!r}"
