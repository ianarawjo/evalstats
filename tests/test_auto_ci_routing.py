"""Tests for method='auto' marginal-CI routing introduced in the real-sim branch.

Routing rules under test
------------------------
* Binary (0/1) data           → resolved_ci_method == "wilson"
* Continuous [0,1], any N     → resolved_ci_method == "nig"
* Unbounded numeric, N >= 60  → resolved_ci_method == "t_interval"
* Unbounded numeric, N < 60   → resolved_ci_method == "bootstrap_t"

Each test also verifies that the returned CIs are finite, ordered (lo < hi),
and bracket the sample mean at a reasonable confidence level.
"""

import numpy as np
import pytest

import evalstats as es
from evalstats.core.variance import robustness_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(scores: np.ndarray, n_templates: int = 2) -> es.BenchmarkResult:
    n_inputs = scores.shape[1]
    return es.BenchmarkResult(
        scores=scores,
        template_labels=[f"T{i}" for i in range(n_templates)],
        input_labels=[f"q{j}" for j in range(n_inputs)],
    )


def _ci_valid(bundle: es.AnalysisBundle) -> None:
    """Assert all marginal CIs are finite and properly ordered."""
    ci_lo = bundle.robustness.ci_low
    ci_hi = bundle.robustness.ci_high
    assert ci_lo is not None and ci_hi is not None, "CIs were not computed"
    assert np.all(np.isfinite(ci_lo)), "ci_low contains non-finite values"
    assert np.all(np.isfinite(ci_hi)), "ci_high contains non-finite values"
    assert np.all(ci_lo <= ci_hi), "ci_low > ci_high for at least one template"


def _ci_brackets_mean(bundle: es.AnalysisBundle) -> None:
    """Assert the CI contains the sample mean for every template."""
    mean = bundle.robustness.mean
    assert np.all(bundle.robustness.ci_low <= mean + 1e-9)
    assert np.all(bundle.robustness.ci_high >= mean - 1e-9)


# ---------------------------------------------------------------------------
# Routing: method selection
# ---------------------------------------------------------------------------

class TestAutoRouting:
    RNG = np.random.default_rng(42)

    def _analyze(self, scores, n_bootstrap=500):
        result = _make_result(scores, n_templates=scores.shape[0])
        return es.analyze(result, n_bootstrap=n_bootstrap, rng=np.random.default_rng(0))

    def test_binary_routes_to_wilson(self):
        scores = self.RNG.choice([0, 1], size=(2, 80)).astype(float)
        bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "wilson"

    def test_continuous_01_routes_to_nig(self):
        # Scores sampled from Beta(2,5) — strictly in (0,1), not binary.
        scores = np.random.default_rng(1).beta(2, 5, size=(2, 80))
        bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "nig"

    def test_continuous_01_boundary_values_route_to_nig(self):
        # Data that touches exactly 0 and 1 but has interior values.
        rng = np.random.default_rng(2)
        scores = rng.beta(0.5, 0.5, size=(2, 80))
        bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "nig"

    def test_unbounded_large_n_routes_to_t_interval(self):
        # Scores can exceed 1 — triggers the "beyond [0,1]" path.
        scores = np.random.default_rng(3).normal(5.0, 1.5, size=(2, 80))
        bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "t_interval"

    def test_unbounded_small_n_routes_to_bootstrap_t(self):
        scores = np.random.default_rng(4).normal(5.0, 1.5, size=(2, 30))
        bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "bootstrap_t"

    def test_boundary_n60_routes_to_t_interval(self):
        # Exactly N=60 should use t_interval, not bootstrap_t.
        scores = np.random.default_rng(5).normal(3.0, 1.0, size=(2, 60))
        bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "t_interval"

    def test_boundary_n59_routes_to_bootstrap_t(self):
        scores = np.random.default_rng(6).normal(3.0, 1.0, size=(2, 59))
        bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "bootstrap_t"


# ---------------------------------------------------------------------------
# CI quality: analytical methods don't need n_bootstrap
# ---------------------------------------------------------------------------

class TestAnalyticalMethodsNoBootstrap:
    """NIG, t_interval, and wilson are analytical — robustness_metrics should
    produce CIs even when n_bootstrap=None.  We test robustness_metrics
    directly here because analyze() also runs pairwise comparisons that
    require bootstrap; this isolates the gate-lifting change in variance.py."""

    def _rob(self, scores, method):
        labels = [f"T{i}" for i in range(scores.shape[0])]
        return robustness_metrics(scores, labels, n_bootstrap=None, marginal_method=method)

    def test_nig_ci_without_bootstrap(self):
        scores = np.random.default_rng(10).beta(2, 5, size=(2, 50))
        r = self._rob(scores, "nig")
        assert r.ci_low is not None and r.ci_high is not None
        assert np.all(np.isfinite(r.ci_low)) and np.all(np.isfinite(r.ci_high))
        assert np.all(r.ci_low <= r.ci_high)

    def test_t_interval_ci_without_bootstrap(self):
        scores = np.random.default_rng(11).normal(5.0, 1.0, size=(2, 80))
        r = self._rob(scores, "t_interval")
        assert r.ci_low is not None and r.ci_high is not None
        assert np.all(np.isfinite(r.ci_low)) and np.all(np.isfinite(r.ci_high))
        assert np.all(r.ci_low <= r.ci_high)

    def test_wilson_ci_without_bootstrap(self):
        scores = np.random.default_rng(12).choice([0, 1], size=(2, 80)).astype(float)
        r = self._rob(scores, "wilson")
        assert r.ci_low is not None and r.ci_high is not None
        assert np.all(np.isfinite(r.ci_low)) and np.all(np.isfinite(r.ci_high))
        assert np.all(r.ci_low <= r.ci_high)


# ---------------------------------------------------------------------------
# CI quality: values are finite, ordered, and bracket the mean
# ---------------------------------------------------------------------------

class TestCIQuality:
    """Smoke-test that each routed method produces reasonable CIs."""

    def _bundle(self, scores, n_bootstrap=800):
        result = _make_result(scores, n_templates=scores.shape[0])
        return es.analyze(result, n_bootstrap=n_bootstrap, rng=np.random.default_rng(99))

    def test_wilson_ci_quality(self):
        scores = np.random.default_rng(20).choice([0, 1], size=(3, 100)).astype(float)
        bundle = self._bundle(scores)
        _ci_valid(bundle)
        _ci_brackets_mean(bundle)

    def test_nig_ci_quality(self):
        scores = np.random.default_rng(21).beta(3, 3, size=(3, 60))
        bundle = self._bundle(scores)
        _ci_valid(bundle)
        _ci_brackets_mean(bundle)

    def test_t_interval_ci_quality(self):
        scores = np.random.default_rng(22).normal(7.0, 2.0, size=(3, 80))
        bundle = self._bundle(scores)
        _ci_valid(bundle)
        _ci_brackets_mean(bundle)

    def test_bootstrap_t_ci_quality(self):
        scores = np.random.default_rng(23).normal(7.0, 2.0, size=(3, 30))
        bundle = self._bundle(scores, n_bootstrap=400)
        _ci_valid(bundle)
        _ci_brackets_mean(bundle)

    def test_ci_width_shrinks_with_n_for_nig(self):
        """Larger N → narrower CI (sanity check for NIG)."""
        rng = np.random.default_rng(30)
        scores_small = rng.beta(2, 5, size=(2, 20))
        scores_large = rng.beta(2, 5, size=(2, 200))

        def width(scores):
            labels = [f"T{i}" for i in range(scores.shape[0])]
            r = robustness_metrics(scores, labels, n_bootstrap=None, marginal_method="nig")
            return float(np.mean(r.ci_high - r.ci_low))

        assert width(scores_small) > width(scores_large)

    def test_ci_width_shrinks_with_n_for_t_interval(self):
        """Larger N → narrower CI (sanity check for t_interval)."""
        rng = np.random.default_rng(31)
        scores_small = rng.normal(5.0, 1.0, size=(2, 30))
        scores_large = rng.normal(5.0, 1.0, size=(2, 200))

        def width(scores):
            labels = [f"T{i}" for i in range(scores.shape[0])]
            r = robustness_metrics(scores, labels, n_bootstrap=None, marginal_method="t_interval")
            return float(np.mean(r.ci_high - r.ci_low))

        assert width(scores_small) > width(scores_large)
