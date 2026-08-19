"""Tests for method='auto' marginal-CI routing.

Routing rules under test
------------------------
* Binary (0/1) data, single-run     → resolved_ci_method == "wilson"
* Binary (0/1) data, multi-run      → resolved_ci_method == "wilson" ("Wilson flat")
* Binary (0/1) data, but an explicit
  score_range wider than [0,1]      → resolved_ci_method == "logit_t", with a
                                       UserWarning -- the declaration beats the
                                       inference, since a sample of only 0s and
                                       1s doesn't prove the metric is Bernoulli
* Numeric data already in [0,1]     → resolved_ci_method == "logit_t"
                                       (exact bounds, but still warns since
                                       it's an inference, not a declaration)
* Numeric data outside [0,1],
  score_range given                 → resolved_ci_method == "logit_t", no warning
* Numeric data outside [0,1],
  score_range NOT given             → resolved_ci_method == "t_interval"
                                       (evalstats refuses to guess a [lo, hi]
                                       range from the sample's own min/max),
                                       with a UserWarning recommending
                                       score_range be passed explicitly

Each test also verifies that the returned CIs are finite, ordered (lo < hi),
and bracket the sample mean at a reasonable confidence level.
"""

import warnings

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

    def test_continuous_01_routes_to_logit_t(self):
        # Scores sampled from Beta(2,5) — strictly in (0,1), not binary.
        # Still warns: evalstats inferred the [0,1] range, wasn't told it.
        scores = np.random.default_rng(1).beta(2, 5, size=(2, 80))
        with pytest.warns(UserWarning, match="auto-detected"):
            bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "logit_t"
        assert bundle.resolved_score_range == (0.0, 1.0)

    def test_continuous_01_boundary_values_route_to_logit_t(self):
        # Data that touches exactly 0 and 1 but has interior values.
        rng = np.random.default_rng(2)
        scores = rng.beta(0.5, 0.5, size=(2, 80))
        with pytest.warns(UserWarning, match="auto-detected"):
            bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "logit_t"

    def test_unbounded_large_n_falls_back_to_t_interval_with_warning(self):
        # Scores exceed [0, 1] and no score_range is declared -- evalstats
        # refuses to guess a range from the sample's own min/max, so this
        # falls back to t_interval with a UserWarning recommending
        # score_range be passed explicitly.
        scores = np.random.default_rng(3).normal(5.0, 1.5, size=(2, 80))
        with pytest.warns(UserWarning, match="score_range"):
            bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "t_interval"
        assert bundle.resolved_score_range is None

    def test_unbounded_small_n_falls_back_to_t_interval_with_warning(self):
        scores = np.random.default_rng(4).normal(5.0, 1.5, size=(2, 30))
        with pytest.warns(UserWarning, match="score_range"):
            bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "t_interval"

    def test_unbounded_with_explicit_score_range_routes_to_logit_t_no_warning(self):
        # Declaring score_range routes to logit_t instead, with no warning.
        scores = np.random.default_rng(5).normal(3.0, 1.0, size=(2, 60))
        result = _make_result(scores, n_templates=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fail the test if anything warns
            bundle = es.analyze(
                result, n_bootstrap=500, rng=np.random.default_rng(0),
                score_range=(-5.0, 11.0),
            )
        assert bundle.resolved_ci_method == "logit_t"
        assert bundle.resolved_score_range == (-5.0, 11.0)

    def test_degenerate_constant_data_falls_back_to_t_interval(self):
        # Constant data isn't in [0,1] and no score_range is given, so this
        # takes the same "outside [0,1], no score_range" path as any other
        # unbounded data -- just confirming it doesn't crash on zero variance.
        scores = np.full((2, 30), 3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # zero-variance + score_range warnings, unrelated
            bundle = self._analyze(scores)
        assert bundle.resolved_ci_method == "t_interval"


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

    def test_bounded_01_auto_ci_quality(self):
        # Auto-detected [0, 1] data routes to logit_t (with a warning, since
        # the range was inferred rather than declared).
        scores = np.random.default_rng(21).beta(3, 3, size=(3, 60))
        with pytest.warns(UserWarning, match="auto-detected"):
            bundle = self._bundle(scores)
        assert bundle.resolved_ci_method == "logit_t"
        _ci_valid(bundle)
        _ci_brackets_mean(bundle)

    def test_unbounded_numeric_auto_ci_quality(self):
        # Unbounded numeric data with no score_range auto-routes to
        # t_interval (see TestAutoRouting) -- still produces valid,
        # mean-bracketing CIs.
        scores = np.random.default_rng(22).normal(7.0, 2.0, size=(3, 80))
        with pytest.warns(UserWarning, match="score_range"):
            bundle = self._bundle(scores)
        assert bundle.resolved_ci_method == "t_interval"
        _ci_valid(bundle)
        _ci_brackets_mean(bundle)

    def test_unbounded_numeric_small_n_auto_ci_quality(self):
        scores = np.random.default_rng(23).normal(7.0, 2.0, size=(3, 30))
        with pytest.warns(UserWarning, match="score_range"):
            bundle = self._bundle(scores, n_bootstrap=400)
        assert bundle.resolved_ci_method == "t_interval"
        _ci_valid(bundle)
        _ci_brackets_mean(bundle)

    def test_explicit_t_interval_ci_quality(self):
        # method='t_interval' is still available as an explicit choice,
        # bypassing the score_range/logit_t auto-routing entirely.
        scores = np.random.default_rng(24).normal(7.0, 2.0, size=(3, 80))
        result = _make_result(scores, n_templates=3)
        bundle = es.analyze(
            result, method="t_interval", n_bootstrap=800, rng=np.random.default_rng(99),
        )
        assert bundle.resolved_ci_method == "t_interval"
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


# ---------------------------------------------------------------------------
# score_range: explicit bounds for logit_t on non-[0,1] numeric scales
# ---------------------------------------------------------------------------

class TestScoreRange:
    def _bundle(self, scores, n_bootstrap=500, **kwargs):
        result = _make_result(scores, n_templates=scores.shape[0])
        return es.analyze(result, n_bootstrap=n_bootstrap, rng=np.random.default_rng(0), **kwargs)

    def test_likert_scale_with_score_range_no_warning(self):
        # Integer 1-5 data triggers evalstats' quantization auto-detection
        # (detect_quantization_step) when eval_type isn't given -- that's
        # intentional (see config.AUTO_ANALYZE_METHOD_TABLE's "likert" row),
        # and it emits a UserWarning explaining the switch. Passing
        # eval_type="likert" explicitly is the documented way to silence it.
        rng = np.random.default_rng(40)
        scores = rng.integers(1, 6, size=(2, 40)).astype(float)  # 1-5 Likert
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bundle = self._bundle(scores, score_range=(1, 5), eval_type="likert")
        assert bundle.resolved_ci_method == "logit_t"  # robustness/marginal CI, unaffected by likert routing
        assert bundle.resolved_score_range == (1.0, 5.0)
        _ci_valid(bundle)
        _ci_brackets_mean(bundle)

    def test_percentage_grade_with_score_range_no_warning(self):
        rng = np.random.default_rng(41)
        scores = rng.uniform(0, 100, size=(2, 40))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bundle = self._bundle(scores, score_range=(0, 100))
        assert bundle.resolved_ci_method == "logit_t"
        assert bundle.resolved_score_range == (0.0, 100.0)
        _ci_valid(bundle)
        _ci_brackets_mean(bundle)

    def test_pairwise_ci_uses_logit_t_with_score_range(self):
        # Genuinely continuous (non-quantized) data within score_range --
        # a 1-5 integer draw here would trigger the likert quantization
        # auto-detection and route pairwise to NIG instead (see
        # test_pairwise_ci_uses_nig_for_auto_detected_likert below).
        rng = np.random.default_rng(42)
        scores = rng.uniform(1, 5, size=(2, 40))
        bundle = self._bundle(scores, score_range=(1, 5))
        pair = bundle.pairwise.get("T0", "T1")
        assert "logit-t" in pair.test_method.lower()
        assert pair.ci_low <= pair.point_diff <= pair.ci_high

    def test_pairwise_ci_uses_nig_for_auto_detected_likert(self):
        rng = np.random.default_rng(42)
        scores = rng.integers(1, 6, size=(2, 40)).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = self._bundle(scores, score_range=(1, 5))
        pair = bundle.pairwise.get("T0", "T1")
        assert "nig" in pair.test_method.lower()
        assert pair.ci_low <= pair.point_diff <= pair.ci_high

    def test_score_range_violated_by_data_raises(self):
        rng = np.random.default_rng(43)
        scores = rng.uniform(0, 10, size=(2, 30))
        with pytest.raises(ValueError, match="score_range"):
            self._bundle(scores, score_range=(0, 1))

    def test_score_range_lo_ge_hi_raises(self):
        rng = np.random.default_rng(44)
        scores = rng.uniform(1, 5, size=(2, 30))
        with pytest.raises(ValueError, match="lo < hi"):
            self._bundle(scores, score_range=(5, 1))

    def test_explicit_wider_score_range_overrides_binary_detection(self):
        # A sample containing only 0s and 1s does not establish that the
        # metric is Bernoulli when the caller has declared it ranges wider
        # (here a 0-100 grade that happened to score only 0 or 1). The
        # explicit declaration wins over the inference from observed values,
        # and says so. See resampling.binary_routing_applies.
        rng = np.random.default_rng(45)
        scores = rng.choice([0, 1], size=(2, 40)).astype(float)
        with pytest.warns(UserWarning, match="would normally auto-detect as binary"):
            bundle = self._bundle(scores, score_range=(0, 100))
        assert bundle.resolved_ci_method == "logit_t"
        assert bundle.resolved_score_range == (0.0, 100.0)
        _ci_valid(bundle)
        _ci_brackets_mean(bundle)

    def test_binary_data_without_score_range_still_routes_to_wilson(self):
        # The common case is untouched: say nothing, and auto-detection stays
        # fully in charge.
        rng = np.random.default_rng(45)
        scores = rng.choice([0, 1], size=(2, 40)).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bundle = self._bundle(scores)
        assert bundle.resolved_ci_method == "wilson"
        assert bundle.resolved_score_range is None

    def test_binary_data_with_explicit_01_range_agrees_and_stays_binary(self):
        # score_range=(0, 1) agrees with the detection, so there is nothing to
        # override and nothing to warn about.
        rng = np.random.default_rng(45)
        scores = rng.choice([0, 1], size=(2, 40)).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bundle = self._bundle(scores, score_range=(0, 1))
        assert bundle.resolved_ci_method == "wilson"

    def test_binary_data_outside_declared_range_raises(self):
        # 0 is not a valid response on a 1-5 scale. This used to be swallowed
        # silently by the binary path; now the contradiction surfaces.
        rng = np.random.default_rng(45)
        scores = rng.choice([0, 1], size=(2, 40)).astype(float)
        with pytest.warns(UserWarning, match="would normally auto-detect as binary"):
            with pytest.raises(ValueError, match="falls outside it"):
                self._bundle(scores, score_range=(1, 5))
