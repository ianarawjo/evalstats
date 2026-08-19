"""Tests for _max_stat_simultaneous_cis and the simultaneous_ci integration.

Tests are grouped into four sections:
  1. Contract / structural tests  — return type, valid bounds, edge cases.
  2. Key statistical properties   — symmetry, monotonicity, coverage.
  3. Method / path coverage       — all bootstrap variants, seeded path, median.
  4. Integration tests            — all_pairwise, compare_prompts, compare_models.
"""

import numpy as np
import pytest

import evalstats as es
from evalstats.core.paired import (
    _max_stat_simultaneous_cis,
    _bonferroni_simultaneous_cis,
    _simultaneous_cis_router,
    _sidak_simultaneous_cis,
    _joint_bootstrap_scaled_simultaneous_cis,
    _joint_bootstrap_critical_value,
    all_pairwise,
)
from evalstats.core.resampling import (
    degenerate_sample_ci,
    tango_paired_ci,
    tango_paired_ci_from_diffs,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Section 1 — Contract / structural tests
# ---------------------------------------------------------------------------

def test_unsupported_methods_return_empty_dict():
    """Non-bootstrap methods (newcombe, bayes_binary) return {}."""
    scores = _rng(0).normal(0, 1, (3, 20))
    labels = ["m0", "m1", "m2"]
    pairs = [("m0", "m1"), ("m0", "m2")]
    for method in ["newcombe", "bayes_binary"]:
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
    """PairwiseMatrix.simultaneous_ci_method should record the variant used;
    the CI annotation is no longer baked into each PairedDiffResult.test_method.
    Explicitly requests prefer="bonferroni" -- fig:fwer-decision-tree's
    auto default no longer picks Bonferroni (see test_router_returns_boot_
    by_default_for_bootstrap for that)."""
    scores = _rng(11).normal(0, 1, (3, 30))
    labels = ["a", "b", "c"]

    mat = all_pairwise(
        scores, labels, method="bootstrap", ci=0.95,
        n_bootstrap=200, correction="none", rng=_rng(11),
        simultaneous_ci=True, prefer="bonferroni",
    )

    assert mat.simultaneous_ci_method == "bonferroni"


def test_all_pairwise_p_values_valid_with_simultaneous_ci_bonferroni():
    """When simultaneous_ci=True with prefer="bonferroni" explicitly requested,
    p_value stays the original marginal p-value — Bonferroni here only adjusts
    the CI bounds, not the p-values — and must still be a valid probability.
    """
    scores = _rng(12).normal(0, 1, (3, 40))
    labels = ["a", "b", "c"]

    mat_sim = all_pairwise(
        scores, labels, method="bootstrap", ci=0.95,
        n_bootstrap=300, correction="holm", rng=_rng(12),
        simultaneous_ci=True, prefer="bonferroni",
    )

    assert mat_sim.simultaneous_ci_method == "bonferroni"
    for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
        p = mat_sim.get(a, b).p_value
        assert 0.0 <= p <= 1.0, f"p_value {p} out of [0,1] for ({a},{b})"


def test_compare_prompts_simultaneous_ci_propagates():
    """compare_prompts with simultaneous_ci=True sets the flag on the report
    and produces valid CIs."""
    scores = {
        "A": _rng(20).normal(0.7, 0.1, 40).tolist(),
        "B": _rng(21).normal(0.65, 0.1, 40).tolist(),
        "C": _rng(22).normal(0.60, 0.1, 40).tolist(),
    }

    report = es.compare_prompts(
        scores, simultaneous_ci=True, rng=_rng(20), n_bootstrap=400,
    )

    assert report.simultaneous_ci is True
    assert report.pairwise.simultaneous_ci is True
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
        lo, hi = report.pairwise.get(a, b).ci_low, report.pairwise.get(a, b).ci_high
        assert lo <= hi and np.isfinite(lo) and np.isfinite(hi)


def test_compare_prompts_simultaneous_ci_true_by_default():
    scores = {"A": [0.7, 0.8, 0.6], "B": [0.65, 0.75, 0.55]}
    report = es.compare_prompts(scores, rng=_rng(0), n_bootstrap=100)
    assert report.simultaneous_ci is True


def test_compare_models_simultaneous_ci_propagates():
    """compare_models with simultaneous_ci=True sets the flag on the report."""
    rng = _rng(30)
    scores = {
        "GPT":    rng.normal(0.7, 0.1, 40).tolist(),
        "Llama":  rng.normal(0.65, 0.1, 40).tolist(),
        "Mistral": rng.normal(0.60, 0.1, 40).tolist(),
    }
    report = es.compare_models(
        scores, simultaneous_ci=True, rng=_rng(30), n_bootstrap=300,
    )
    assert report.simultaneous_ci is True


def test_newcombe_uses_auto_simultaneous_ci_default():
    """method='newcombe' has no bootstrap CIs, but Sidak/joint-bootstrap
    simultaneous CIs are ci_func-based, not tied to whether the point-
    estimate method is bootstrap-compatible -- so the "auto" default
    (fig:fwer-decision-tree) applies normally here rather than falling back
    to Bonferroni. That max-T-specific fallback for non-bootstrap methods is
    still real and covered by test_router_falls_back_to_bonferroni_for_newcombe
    (prefer="max_t")."""
    rng = _rng(40)
    scores = {
        "A": [int(x > 0.5) for x in rng.random(50)],
        "B": [int(x > 0.5) for x in rng.random(50)],
        "C": [int(x > 0.5) for x in rng.random(50)],
    }
    report = es.compare_prompts(
        scores, method="newcombe", simultaneous_ci=True,
        rng=_rng(40), n_bootstrap=200,
    )
    assert report.simultaneous_ci is True
    # binary, N=50 -> "boot" row of AUTO_SIMULTANEOUS_CI_METHOD_TABLE
    assert report.pairwise.simultaneous_ci_method == "boot"


def test_seeded_compare_prompts_simultaneous_ci():
    """simultaneous_ci=True works end-to-end when score arrays have R=4 runs."""
    rng = _rng(50)
    scores = {
        "A": rng.normal(0.7, 0.1, (30, 4)).tolist(),
        "B": rng.normal(0.65, 0.1, (30, 4)).tolist(),
        "C": rng.normal(0.60, 0.1, (30, 4)).tolist(),
    }
    report = es.compare_prompts(
        scores, simultaneous_ci=True, rng=_rng(50), n_bootstrap=300,
    )
    assert report.simultaneous_ci is True
    # Any construction is acceptable here -- this test is about seeded (R>=3)
    # data not crashing the simultaneous-CI pipeline, not about which
    # specific auto-routed method fires for this N/data-kind.
    assert report.pairwise.simultaneous_ci_method in {"sidak", "boot", "max_t", "bonferroni"}
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
        lo, hi = report.pairwise.get(a, b).ci_low, report.pairwise.get(a, b).ci_high
        assert lo <= hi and np.isfinite(lo) and np.isfinite(hi)


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


def test_bonferroni_degenerate_zero_variance_unbounded_is_infinite():
    """When all diffs are identical, SE=0 and no variance-driven interval is
    computable. With no bounds on the data there is nothing left to fall back
    on, so the CI is (-inf, +inf) -- explicitly NOT the zero-width point
    interval this used to return, which claimed certainty from a sample that
    contains no spread at all."""
    scores = np.ones((2, 30))
    labels = ["a", "b"]
    results, pairs = _make_results(scores, labels)
    with pytest.warns(UserWarning, match="zero variance"):
        cis = _bonferroni_simultaneous_cis(results, pairs, ci=0.95)
    lo, hi = cis[pairs[0]]
    assert lo == -np.inf and hi == np.inf


def test_bonferroni_degenerate_zero_variance_bounded_is_finite_and_wide():
    """Given the diff bounds, the same zero-variance pair gets the conservative
    Clopper-Pearson-based bound instead of an infinite (or zero-width) one."""
    scores = np.ones((2, 30))
    labels = ["a", "b"]
    results, pairs = _make_results(scores, labels)
    cis = _bonferroni_simultaneous_cis(results, pairs, ci=0.95, diff_bounds=(-1.0, 1.0))
    lo, hi = cis[pairs[0]]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi, "zero-variance pair must not get a zero-width interval"
    assert -1.0 <= lo <= 0.0 <= hi <= 1.0
    # Matches resampling.degenerate_sample_ci at the Bonferroni-adjusted alpha
    # (k=1 here, so alpha_adj == alpha).
    expected = degenerate_sample_ci(0.0, 30, 0.05, -1.0, 1.0)
    np.testing.assert_allclose((lo, hi), expected, atol=1e-12)


def test_bonferroni_degenerate_constant_nonzero_offset_covers_zero():
    """The reported failure: two arms with a constant offset (A = 0.9, B = 0.8
    on every item). The pair is the only one in the family, so it skips
    Sidak/boot entirely and lands on the Bonferroni fallback. It must not come
    back as (0.1, 0.1)."""
    scores = np.vstack([np.full(30, 0.9), np.full(30, 0.8)])
    labels = ["a", "b"]
    results, pairs = _make_results(scores, labels, method="logit_t", score_range=(0.0, 1.0))
    cis = _bonferroni_simultaneous_cis(results, pairs, ci=0.95, diff_bounds=(-1.0, 1.0))
    lo, hi = cis[pairs[0]]
    assert lo < 0.1 < hi, f"expected an interval around 0.1, got ({lo}, {hi})"
    assert hi - lo > 0.05


# --- Router tests ---

def test_router_returns_boot_by_default_for_unbounded_n_ge_30():
    """Router's "auto" default (fig:fwer-decision-tree) picks the joint
    bootstrap ("boot") for unbounded numeric data at N>=30, regardless of
    whether the point-estimate method is bootstrap-compatible."""
    scores = _rng(70).normal(0, 1, (3, 40))
    labels = ["a", "b", "c"]
    results, pairs = _make_results(scores, labels)
    cis, used, _ = _simultaneous_cis_router(
        scores, results, pairs, labels,
        method="bootstrap", ci=0.95, n_bootstrap=300,
        rng=_rng(70), statistic="mean",
    )
    assert used == "boot"
    assert len(cis) == len(pairs)


def test_router_returns_bonferroni_when_explicitly_preferred():
    """prefer="bonferroni" still forces the closed-form fallback directly."""
    scores = _rng(70).normal(0, 1, (3, 40))
    labels = ["a", "b", "c"]
    results, pairs = _make_results(scores, labels)
    cis, used, _ = _simultaneous_cis_router(
        scores, results, pairs, labels,
        method="bootstrap", ci=0.95, n_bootstrap=300,
        rng=_rng(70), statistic="mean",
        prefer="bonferroni",
    )
    assert used == "bonferroni"
    assert len(cis) == len(pairs)


def test_router_returns_max_stat_for_bootstrap_when_preferred():
    """Router should choose 'max_t' for a bootstrap-compatible method when
    explicitly requested via prefer='max_t'."""
    scores = _rng(70).normal(0, 1, (3, 40))
    labels = ["a", "b", "c"]
    results, pairs = _make_results(scores, labels)
    cis, used, _ = _simultaneous_cis_router(
        scores, results, pairs, labels,
        method="bootstrap", ci=0.95, n_bootstrap=300,
        rng=_rng(70), statistic="mean",
        prefer="max_t",
    )
    assert used == "max_t"
    assert len(cis) == len(pairs)


def test_router_falls_back_to_bonferroni_for_newcombe():
    """Router should fall back to 'bonferroni' for analytical methods when
    prefer="max_t" is requested (max_t needs a bootstrap distribution;
    Sidak/boot -- the auto default -- are ci_func-based and don't need this
    fallback at all, see test_router_returns_boot_by_default_for_unbounded_n_ge_30)."""
    # Use continuous scores but force method='newcombe' to trigger the fallback.
    scores = _rng(71).normal(0, 1, (3, 40))
    labels = ["a", "b", "c"]
    results, pairs = _make_results(scores, labels)
    cis, used, _ = _simultaneous_cis_router(
        scores, results, pairs, labels,
        method="newcombe", ci=0.95, n_bootstrap=300,
        rng=_rng(71), statistic="mean",
        prefer="max_t",
    )
    assert used == "bonferroni"
    assert len(cis) == len(pairs)
    for pair, (lo, hi) in cis.items():
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= hi


@pytest.mark.parametrize("method", ["bootstrap", "bca", "smooth_bootstrap",
                                     "bayes_bootstrap", "permutation", "sign_test", "auto"])
def test_router_max_stat_for_all_bootstrap_methods_when_preferred(method):
    """All bootstrap-compatible methods should route to 'max_t' when
    explicitly requested via prefer='max_t'."""
    scores = _rng(72).normal(0, 1, (3, 35))
    labels = ["a", "b", "c"]
    results, pairs = _make_results(scores, labels)
    _, used, _ = _simultaneous_cis_router(
        scores, results, pairs, labels,
        method=method, ci=0.95, n_bootstrap=200,
        rng=_rng(72), statistic="mean",
        prefer="max_t",
    )
    assert used == "max_t", f"Expected max_t for method={method!r}, got {used!r}"


def test_simultaneous_ci_method_field_boot_for_bootstrap():
    """PairwiseMatrix.simultaneous_ci_method follows the "auto" table by
    default (fig:fwer-decision-tree) -- unbounded numeric, N=30 -> "boot".
    Pass prefer="bonferroni"/"max_t" to all_pairwise() to force a specific
    construction instead."""
    scores = _rng(80).normal(0, 1, (3, 30))
    labels = ["a", "b", "c"]
    mat = all_pairwise(
        scores, labels, method="bootstrap", n_bootstrap=200,
        rng=_rng(80), simultaneous_ci=True, correction="none",
    )
    assert mat.simultaneous_ci is True
    assert mat.simultaneous_ci_method == "boot"


def test_simultaneous_ci_method_field_sidak():
    """PairwiseMatrix.simultaneous_ci_method is 'sidak' for binary data
    below the N=50 auto-routing threshold, regardless of the point-estimate
    method (Sidak is ci_func-based, not tied to bootstrap-compatibility)."""
    scores = (_rng(81).random((3, 40)) > 0.5).astype(float)
    labels = ["a", "b", "c"]
    mat = all_pairwise(
        scores, labels, method="newcombe", n_bootstrap=200,
        rng=_rng(81), simultaneous_ci=True, correction="none",
    )
    assert mat.simultaneous_ci is True
    assert mat.simultaneous_ci_method == "sidak"


def test_simultaneous_ci_method_field_none_when_not_requested():
    """PairwiseMatrix.simultaneous_ci_method is None when simultaneous_ci=False."""
    scores = _rng(82).normal(0, 1, (3, 30))
    labels = ["a", "b", "c"]
    mat = all_pairwise(scores, labels, n_bootstrap=100, rng=_rng(82), simultaneous_ci=False)
    assert mat.simultaneous_ci is False
    assert mat.simultaneous_ci_method is None


def test_bonferroni_annotation_in_test_method():
    """PairwiseMatrix.simultaneous_ci_method reflects the explicitly
    requested construction when prefer="bonferroni" is passed."""
    scores = (_rng(83).random((3, 40)) > 0.5).astype(float)
    labels = ["a", "b", "c"]
    mat = all_pairwise(
        scores, labels, method="newcombe", n_bootstrap=200,
        rng=_rng(83), simultaneous_ci=True, correction="none",
        prefer="bonferroni",
    )
    assert mat.simultaneous_ci_method == "bonferroni"


# ---------------------------------------------------------------------------
# Section 6 — Generic Sidak / joint-bootstrap-scaled simultaneous CIs.
# These take an arbitrary alpha-parameterized `ci_func` -- Tango's
# tango_paired_ci_from_diffs is exercised here as one concrete instantiation
# (binary paired data), but _sidak_simultaneous_cis /
# _joint_bootstrap_scaled_simultaneous_cis / _joint_bootstrap_critical_value
# themselves have no Tango-specific logic; test_..._is_method_agnostic below
# confirms a second, unrelated ci_func works identically.
# ---------------------------------------------------------------------------

def _binary_paired_scores(rng, k, M, p=0.5):
    """(k, M) 0/1 matrix, one row per arm, independently Bernoulli(p)."""
    return (rng.random((k, M)) < p).astype(float)


def _make_binary_results(scores_2d, labels, **kw):
    mat = all_pairwise(scores_2d, labels, method="tango", n_bootstrap=200, rng=_rng(0), **kw)
    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    return mat.results, pairs


def test_tango_paired_ci_from_diffs_matches_tango_paired_ci():
    """tango_paired_ci_from_diffs(a_bin - b_bin, alpha) must reproduce
    tango_paired_ci(a, b, alpha) exactly -- it's a refactor of the same
    closed-form math, not an independent implementation."""
    rng = _rng(200)
    a = (rng.random(80) < 0.6).astype(float)
    b = (rng.random(80) < 0.4).astype(float)
    for alpha in (0.01, 0.05, 0.2):
        expected = tango_paired_ci(a, b, alpha)
        actual = tango_paired_ci_from_diffs(a - b, alpha)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_tango_paired_ci_from_diffs_empty():
    assert tango_paired_ci_from_diffs(np.array([]), 0.05) == (0.0, 0.0)


def _t_interval_ci_func(diffs: np.ndarray, alpha: float) -> tuple[float, float]:
    """A second, unrelated alpha-parameterized CI formula (paired t-interval)
    used only to prove _sidak_simultaneous_cis /
    _joint_bootstrap_scaled_simultaneous_cis don't hardcode Tango anywhere."""
    from scipy import stats as scipy_stats
    diffs = np.asarray(diffs)
    M = len(diffs)
    if M < 2:
        return (float(diffs.mean()), float(diffs.mean()))
    se = float(np.std(diffs, ddof=1)) / np.sqrt(M)
    t_crit = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, df=M - 1))
    mean = float(diffs.mean())
    return (mean - t_crit * se, mean + t_crit * se)


def test_sidak_returns_all_pairs():
    scores = _binary_paired_scores(_rng(90), 3, 60)
    labels = ["a", "b", "c"]
    results, pairs = _make_binary_results(scores, labels)
    cis = _sidak_simultaneous_cis(results, pairs, ci=0.95, ci_func=tango_paired_ci_from_diffs)
    assert set(cis.keys()) == set(pairs)


def test_sidak_bounds_finite_and_ordered():
    scores = _binary_paired_scores(_rng(91), 4, 50)
    labels = ["a", "b", "c", "d"]
    results, pairs = _make_binary_results(scores, labels)
    cis = _sidak_simultaneous_cis(results, pairs, ci=0.95, ci_func=tango_paired_ci_from_diffs)
    for pair, (lo, hi) in cis.items():
        assert np.isfinite(lo) and np.isfinite(hi), f"{pair}: non-finite bounds"
        assert lo <= hi, f"{pair}: lo > hi"
        assert -1.0 <= lo and hi <= 1.0


def test_sidak_wider_than_naive_ci_func():
    """Sidak-adjusted intervals must be at least as wide as the naive
    per-pair CI from the same ci_func (k > 1 always widens the
    per-comparison alpha)."""
    scores = _binary_paired_scores(_rng(92), 4, 80)
    labels = ["a", "b", "c", "d"]
    results, pairs = _make_binary_results(scores, labels)
    cis_sidak = _sidak_simultaneous_cis(results, pairs, ci=0.95, ci_func=tango_paired_ci_from_diffs)
    for pair in pairs:
        r = results[pair]
        naive_lo, naive_hi = tango_paired_ci_from_diffs(r.per_input_diffs, 0.05)
        sidak_lo, sidak_hi = cis_sidak[pair]
        assert (sidak_hi - sidak_lo) >= (naive_hi - naive_lo) - 1e-9, (
            f"{pair}: Sidak width should be >= naive width"
        )


def test_sidak_single_pair_equals_naive_ci_func():
    """With k=1, Sidak's adjustment (1 - (1-alpha)**(1/1) == alpha) is a
    no-op, so the CI matches ci_func called directly at the same level."""
    scores = _binary_paired_scores(_rng(93), 2, 60)
    labels = ["a", "b"]
    results, pairs = _make_binary_results(scores, labels)
    assert len(pairs) == 1
    cis = _sidak_simultaneous_cis(results, pairs, ci=0.95, ci_func=tango_paired_ci_from_diffs)
    expected = tango_paired_ci_from_diffs(results[pairs[0]].per_input_diffs, 0.05)
    np.testing.assert_allclose(cis[pairs[0]], expected, atol=1e-9)


def test_sidak_empty_pairs_returns_empty():
    assert _sidak_simultaneous_cis({}, [], ci=0.95, ci_func=tango_paired_ci_from_diffs) == {}


def test_sidak_is_ci_func_agnostic():
    """Swapping ci_func for an unrelated paired-t formula must change which
    formula is called (not silently keep using Tango's), while the
    Sidak-adjustment machinery itself stays identical."""
    scores = _rng(103).normal(0, 1, (3, 50))  # continuous, NOT binary
    labels = ["a", "b", "c"]
    mat = all_pairwise(scores, labels, method="bootstrap", n_bootstrap=100, rng=_rng(0), correction="none")
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    cis = _sidak_simultaneous_cis(mat.results, pairs, ci=0.95, ci_func=_t_interval_ci_func)
    k = len(pairs)
    alpha_adj = 1.0 - (1.0 - 0.05) ** (1.0 / k)
    for pair in pairs:
        expected = _t_interval_ci_func(mat.results[pair].per_input_diffs, alpha_adj)
        np.testing.assert_allclose(cis[pair], expected, atol=1e-9)


def test_joint_bootstrap_critical_value_returns_finite_positive():
    scores = _binary_paired_scores(_rng(94), 4, 80)
    labels = ["a", "b", "c", "d"]
    pairs = [(labels[i], labels[j]) for i in range(4) for j in range(i + 1, 4)]
    c = _joint_bootstrap_critical_value(
        scores=scores, pairs=pairs, labels=labels, ci=0.95, n_bootstrap=500, rng=_rng(94),
    )
    assert c is not None
    assert np.isfinite(c) and c > 0.0


def test_joint_bootstrap_critical_value_empty_pairs():
    scores = _binary_paired_scores(_rng(95), 2, 40)
    labels = ["a", "b"]
    c = _joint_bootstrap_critical_value(
        scores=scores, pairs=[], labels=labels, ci=0.95, n_bootstrap=100, rng=_rng(95),
    )
    assert c is None


def test_joint_bootstrap_scaled_returns_all_pairs_or_empty():
    scores = _binary_paired_scores(_rng(96), 3, 70)
    labels = ["a", "b", "c"]
    results, pairs = _make_binary_results(scores, labels)
    cis = _joint_bootstrap_scaled_simultaneous_cis(
        scores=scores, results=results, pairs=pairs, labels=labels,
        ci=0.95, n_bootstrap=500, rng=_rng(96), ci_func=tango_paired_ci_from_diffs,
    )
    # Degenerate (all-same-value) draws can legitimately return {}; a
    # non-degenerate binary draw should return every requested pair.
    assert cis == {} or set(cis.keys()) == set(pairs)


def test_joint_bootstrap_scaled_bounds_finite_and_ordered():
    scores = _binary_paired_scores(_rng(97), 4, 80)
    labels = ["a", "b", "c", "d"]
    results, pairs = _make_binary_results(scores, labels)
    cis = _joint_bootstrap_scaled_simultaneous_cis(
        scores=scores, results=results, pairs=pairs, labels=labels,
        ci=0.95, n_bootstrap=500, rng=_rng(97), ci_func=tango_paired_ci_from_diffs,
    )
    for pair, (lo, hi) in cis.items():
        assert np.isfinite(lo) and np.isfinite(hi), f"{pair}: non-finite bounds"
        assert lo <= hi, f"{pair}: lo > hi"
        assert -1.0 <= lo and hi <= 1.0


def test_joint_bootstrap_scaled_empty_pairs_returns_empty():
    scores = _binary_paired_scores(_rng(98), 2, 40)
    labels = ["a", "b"]
    assert _joint_bootstrap_scaled_simultaneous_cis(
        scores=scores, results={}, pairs=[], labels=labels, ci=0.95, n_bootstrap=100, rng=_rng(98),
        ci_func=tango_paired_ci_from_diffs,
    ) == {}


def test_joint_bootstrap_scaled_degenerate_all_identical_returns_empty():
    """When every arm is constant (zero bootstrap variance for every pair),
    the joint critical value is undefined and the function must return {}
    rather than raising or dividing by zero."""
    scores = np.ones((3, 30))
    labels = ["a", "b", "c"]
    results, pairs = _make_binary_results(scores, labels)
    cis = _joint_bootstrap_scaled_simultaneous_cis(
        scores=scores, results=results, pairs=pairs, labels=labels,
        ci=0.95, n_bootstrap=200, rng=_rng(99), ci_func=tango_paired_ci_from_diffs,
    )
    assert cis == {}


def test_joint_bootstrap_scaled_is_ci_func_agnostic():
    """Same ci_func-swap check as test_sidak_is_ci_func_agnostic, for the
    joint-bootstrap-scaled construction."""
    scores = _rng(104).normal(0, 1, (3, 50))
    labels = ["a", "b", "c"]
    mat = all_pairwise(scores, labels, method="bootstrap", n_bootstrap=100, rng=_rng(0), correction="none")
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]

    from scipy import stats as scipy_stats

    c = _joint_bootstrap_critical_value(
        scores=scores, pairs=pairs, labels=labels, ci=0.95, n_bootstrap=300, rng=_rng(50),
    )
    assert c is not None
    alpha_eff = float(2.0 * (1.0 - scipy_stats.norm.cdf(c)))

    cis = _joint_bootstrap_scaled_simultaneous_cis(
        scores=scores, results=mat.results, pairs=pairs, labels=labels,
        ci=0.95, n_bootstrap=300, rng=_rng(50), ci_func=_t_interval_ci_func,
    )
    for pair in pairs:
        expected = _t_interval_ci_func(mat.results[pair].per_input_diffs, alpha_eff)
        np.testing.assert_allclose(cis[pair], expected, atol=1e-9)


def test_sidak_simultaneous_coverage_near_nominal():
    """Family-wise coverage (ALL pairs simultaneously cover a true diff of
    0) should be at or above the nominal level -- Sidak assumes
    independence between comparisons, so on real (positively correlated,
    shared-reference-arm) data it should be conservative, not under-cover.
    Exercised with ci_func=tango_paired_ci_from_diffs.

    n_simulations=200, ci_level=0.95: SE ~= 0.015 under the null, so the
    tolerance [0.85, 1.00] catches gross under-coverage while tolerating
    simulation variance in a fast test."""
    rng = _rng(101)
    n_simulations = 200
    M = 60
    ci_level = 0.95
    labels = ["m0", "m1", "m2", "m3"]
    pairs = [("m0", "m1"), ("m0", "m2"), ("m0", "m3")]

    hits = 0
    for _ in range(n_simulations):
        scores = _binary_paired_scores(rng, 4, M, p=0.5)  # all arms share p=0.5 -> true diff 0
        results, _ = _make_binary_results(scores, labels)
        cis = _sidak_simultaneous_cis(results, pairs, ci=ci_level, ci_func=tango_paired_ci_from_diffs)
        if all(cis[p][0] <= 0.0 <= cis[p][1] for p in pairs):
            hits += 1

    coverage = hits / n_simulations
    assert 0.85 <= coverage <= 1.00, (
        f"Sidak(tango) simultaneous coverage {coverage:.3f} outside [0.85, 1.00]; "
        f"expected >= {ci_level}."
    )


def test_router_two_arm_constant_offset_does_not_override_with_zero_width():
    """End-to-end through all_pairwise: exactly two arms with a constant offset
    (the k=1 case, which skips Sidak/boot by construction and always lands on
    the Bonferroni fallback). The simultaneous CI must not replace the
    method's own interval with a zero-width one at the point estimate."""
    scores = np.vstack([np.full(30, 0.9), np.full(30, 0.8)])
    mat = all_pairwise(
        scores, ["a", "b"], method="logit_t", score_range=(0.0, 1.0),
        multi_ci=True, rng=_rng(0),
    )
    assert mat.simultaneous_ci_method == "bonferroni"
    r = mat.results[("a", "b")]
    assert r.ci_low < r.ci_high, "zero-width simultaneous CI on a k=1 comparison"
    assert r.ci_low < r.point_diff < r.ci_high
    # k=1 makes Bonferroni's adjustment an exact no-op, so the simultaneous CI
    # should land on the method's own interval at the same alpha rather than
    # overriding it.
    np.testing.assert_allclose((r.ci_low, r.ci_high), r.multi_ci[0.05], atol=1e-12)


def test_router_k3_constant_offset_wider_than_k1():
    """The same degenerate pair inside a 3-arm family gets a *wider* interval
    (alpha/3 rather than alpha), not a narrower or zero-width one."""
    k1 = all_pairwise(
        np.vstack([np.full(30, 0.9), np.full(30, 0.8)]), ["a", "b"],
        method="logit_t", score_range=(0.0, 1.0), rng=_rng(0),
    ).results[("a", "b")]
    k3 = all_pairwise(
        np.vstack([np.full(30, 0.9), np.full(30, 0.8), np.full(30, 0.7)]), ["a", "b", "c"],
        method="logit_t", score_range=(0.0, 1.0), rng=_rng(0),
    ).results[("a", "b")]
    assert k3.ci_low < k3.ci_high
    assert (k3.ci_high - k3.ci_low) > (k1.ci_high - k1.ci_low)


def test_router_mixed_family_degenerate_pair_stays_finite():
    """One degenerate pair alongside two ordinary ones: the ordinary pairs keep
    their normal intervals and the degenerate one is not zero-width."""
    rng = _rng(3)
    scores = np.vstack([np.full(30, 0.9), np.full(30, 0.8), rng.uniform(0.2, 0.6, 30)])
    mat = all_pairwise(
        scores, ["a", "b", "c"], method="logit_t", score_range=(0.0, 1.0),
        n_bootstrap=400, rng=_rng(0),
    )
    for pair, r in mat.results.items():
        assert np.isfinite(r.ci_low) and np.isfinite(r.ci_high), pair
        assert r.ci_low < r.ci_high, f"{pair}: zero-width simultaneous CI"


def test_router_binary_degenerate_pair_uses_binary_diff_bounds():
    """All-1 vs all-0 binary arms: diffs are a constant +1, the extreme of the
    [-1, 1] diff support, so the interval runs up to (but not past) 1."""
    scores = np.vstack([np.ones(30), np.zeros(30)])
    mat = all_pairwise(scores, ["a", "b"], method="tango", rng=_rng(0))
    r = mat.results[("a", "b")]
    assert r.ci_low < 1.0 and r.ci_high == pytest.approx(1.0)
    assert r.ci_low > 0.0


def test_router_unbounded_degenerate_pair_not_zero_width_on_boot_route():
    """Unbounded data, k=3, one degenerate pair among two ordinary ones: the
    joint bootstrap succeeds (the other pairs carry variance), so the family
    does NOT reach the Bonferroni fallback. The degenerate pair must still not
    come back zero-width -- t_interval_ci_1d, the bounds-agnostic ci_func,
    keeps its own (mean, mean) contract, so the router wraps it."""
    rng = _rng(3)
    scores = np.vstack([np.full(30, 9.0), np.full(30, 8.0), rng.normal(5.0, 2.0, 30)])
    with pytest.warns(UserWarning, match="zero variance"):
        mat = all_pairwise(
            scores, ["a", "b", "c"], method="t_interval",
            n_bootstrap=400, rng=_rng(0),
        )
    assert mat.simultaneous_ci_method == "boot"
    deg = mat.results[("a", "b")]
    assert (deg.ci_low, deg.ci_high) == (-np.inf, np.inf)
    for pair in (("a", "c"), ("b", "c")):
        r = mat.results[pair]
        assert np.isfinite(r.ci_low) and np.isfinite(r.ci_high)
        assert r.ci_low < r.point_diff < r.ci_high


def test_router_unbounded_degenerate_pair_sidak_route_matches_boot():
    """Same, forced onto the Sidak route -- both k>=3 constructions share the
    wrapped ci_func, so neither can emit a zero-width interval."""
    rng = _rng(3)
    scores = np.vstack([np.full(20, 9.0), np.full(20, 8.0), rng.normal(5.0, 2.0, 20)])
    with pytest.warns(UserWarning, match="zero variance"):
        cis, used, _ = _simultaneous_cis_router(
            scores=scores,
            results=all_pairwise(scores, ["a", "b", "c"], method="t_interval",
                                 simultaneous_ci=False, rng=_rng(0)).results,
            pairs=[("a", "b"), ("a", "c"), ("b", "c")],
            labels=["a", "b", "c"], method="t_interval", ci=0.95,
            n_bootstrap=400, rng=_rng(0), statistic="mean", prefer="sidak",
        )
    assert used == "sidak"
    assert cis[("a", "b")] == (-np.inf, np.inf)
    assert np.isfinite(cis[("a", "c")][0])
