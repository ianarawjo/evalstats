import numpy as np
import pytest

from evalstats.core.resampling import (
    bayes_bootstrap_means_1d,
    bca_interval_1d,
    bootstrap_ci_1d,
    bootstrap_diffs_nested,
    bootstrap_means_1d,
    beta_ci_1d,
    bootstrap_t_ci_nested,
    degenerate_sample_ci,
    logit_t_ci_1d,
    nested_resample_cell_means_once,
    resolve_resampling_method,
    smooth_bootstrap_diffs_nested,
    smooth_bootstrap_means_1d,
)
from evalstats.core.stats_utils import rescaled_ci


def test_resolve_resampling_method_auto_and_passthrough():
    assert resolve_resampling_method("auto", 14) == "bootstrap_t"
    assert resolve_resampling_method("auto", 199) == "bootstrap_t"
    assert resolve_resampling_method("auto", 200) == "bootstrap"
    assert resolve_resampling_method("auto", 201) == "bootstrap"
    assert resolve_resampling_method("bootstrap", 50) == "bootstrap"
    assert resolve_resampling_method("bca", 50) == "bca"


def test_bootstrap_means_1d_matches_reference_draws():
    values = np.array([0.5, 1.5, 2.5, 4.0])
    n_bootstrap = 12

    rng = np.random.default_rng(123)
    actual = bootstrap_means_1d(values, n_bootstrap, rng)

    rng_ref = np.random.default_rng(123)
    expected = np.empty(n_bootstrap)
    m = len(values)
    for b in range(n_bootstrap):
        idx = rng_ref.choice(m, size=m, replace=True)
        expected[b] = np.mean(values[idx])

    np.testing.assert_allclose(actual, expected)


def test_bootstrap_ci_1d_percentile_matches_quantiles():
    values = np.array([1.0, 2.0, 3.5, 4.0, 7.0])
    observed_mean = float(values.mean())
    alpha = 0.1
    n_bootstrap = 200

    rng = np.random.default_rng(7)
    ci_low, ci_high = bootstrap_ci_1d(
        values,
        observed_mean,
        method="bootstrap",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=rng,
    )

    rng_ref = np.random.default_rng(7)
    boot_means = bootstrap_means_1d(values, n_bootstrap, rng_ref)
    expected_low = float(np.percentile(boot_means, 100 * alpha / 2))
    expected_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    assert ci_low <= ci_high
    np.testing.assert_allclose([ci_low, ci_high], [expected_low, expected_high])


def test_bca_interval_1d_degenerate_constant_sample():
    values = np.full(8, 3.25)
    observed_mean = float(values.mean())
    boot_means = np.full(300, 3.25)

    ci_low, ci_high = bca_interval_1d(values, observed_mean, boot_means, alpha=0.05)

    assert ci_low == observed_mean
    assert ci_high == observed_mean


def test_bootstrap_diffs_nested_matches_reference_indices():
    scores_a = np.array(
        [
            [0.2, 0.7, 1.4],
            [1.0, 0.8, 1.2],
            [2.1, 2.4, 1.9],
            [0.5, 0.6, 0.9],
        ]
    )
    scores_b = np.array(
        [
            [0.1, 0.4, 1.0],
            [0.9, 0.5, 1.0],
            [1.9, 2.2, 1.6],
            [0.4, 0.5, 0.7],
        ]
    )
    n_bootstrap = 40

    rng = np.random.default_rng(202)
    actual = bootstrap_diffs_nested(scores_a, scores_b, n_bootstrap, rng)

    rng_ref = np.random.default_rng(202)
    m, r = scores_a.shape
    input_idx = rng_ref.integers(0, m, size=(n_bootstrap, m))
    run_idx = rng_ref.integers(0, r, size=(n_bootstrap, m, r))

    expected = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        diffs = np.empty(m)
        for k in range(m):
            src = input_idx[b, k]
            runs = run_idx[b, k]
            diffs[k] = np.mean(scores_a[src, runs]) - np.mean(scores_b[src, runs])
        expected[b] = diffs.mean()

    np.testing.assert_allclose(actual, expected)


def test_bootstrap_diffs_nested_constant_offset_is_invariant():
    rng = np.random.default_rng(99)
    scores_b = rng.normal(size=(6, 5))
    scores_a = scores_b + 1.75

    boot = bootstrap_diffs_nested(
        scores_a,
        scores_b,
        n_bootstrap=120,
        rng=np.random.default_rng(1001),
    )

    np.testing.assert_allclose(boot, np.full(120, 1.75))


def test_nested_resample_cell_means_once_matches_reference_indices():
    scores = np.array(
        [
            [
                [1.0, 1.2, 0.8],
                [1.5, 1.1, 1.4],
                [0.9, 1.0, 0.7],
                [2.0, 1.8, 2.2],
            ],
            [
                [3.0, 3.1, 2.9],
                [2.5, 2.6, 2.4],
                [2.8, 2.7, 2.9],
                [3.2, 3.3, 3.1],
            ],
        ]
    )

    rng = np.random.default_rng(555)
    actual = nested_resample_cell_means_once(scores, rng)

    rng_ref = np.random.default_rng(555)
    n, m, r = scores.shape
    input_idx = rng_ref.integers(0, m, size=m)
    run_idx = rng_ref.integers(0, r, size=(m, r))

    expected = np.empty((n, m))
    for i in range(n):
        for k in range(m):
            src = input_idx[k]
            runs = run_idx[k]
            expected[i, k] = np.mean(scores[i, src, runs])

    np.testing.assert_allclose(actual, expected)


def test_bootstrap_ci_1d_bca_skewed_differs_from_percentile_bootstrap():
    values = np.array([0.1, 0.2, 0.3, 0.4, 5.0])
    observed_mean = float(values.mean())
    alpha = 0.1
    n_bootstrap = 800

    # Use separate RNGs with the same seed so both methods use identical
    # bootstrap replicate streams.
    perc_low, perc_high = bootstrap_ci_1d(
        values,
        observed_mean,
        method="bootstrap",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(123),
    )
    bca_low, bca_high = bootstrap_ci_1d(
        values,
        observed_mean,
        method="bca",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(123),
    )

    # BCa should differ from percentile bootstrap CI for skewed distribution.
    assert not np.allclose(
        [bca_low, bca_high],
        [perc_low, perc_high],
        rtol=1e-4,
        atol=1e-4,
    )

    # Both intervals should still be ordered.
    assert perc_low <= perc_high
    assert bca_low <= bca_high


def test_smooth_bootstrap_means_1d_constant_values_matches_bootstrap_fallback():
    values = np.full(12, 4.2)
    n_bootstrap = 300

    with pytest.warns(UserWarning, match="smooth_bootstrap_means_1d falling back"):
        smooth = smooth_bootstrap_means_1d(
            values,
            n_bootstrap=n_bootstrap,
            rng=np.random.default_rng(77),
        )
    standard = bootstrap_means_1d(
        values,
        n_bootstrap=n_bootstrap,
        rng=np.random.default_rng(77),
    )

    np.testing.assert_allclose(smooth, standard)
    np.testing.assert_allclose(smooth, np.full(n_bootstrap, 4.2))


def test_bayes_bootstrap_means_1d_constant_values_are_degenerate():
    values = np.full(9, -1.75)

    boot_mean = bayes_bootstrap_means_1d(
        values,
        n_bootstrap=500,
        rng=np.random.default_rng(11),
        statistic="mean",
    )
    boot_median = bayes_bootstrap_means_1d(
        values,
        n_bootstrap=500,
        rng=np.random.default_rng(12),
        statistic="median",
    )

    np.testing.assert_allclose(boot_mean, np.full(500, -1.75))
    np.testing.assert_allclose(boot_median, np.full(500, -1.75))


def test_smooth_bootstrap_diffs_nested_constant_diffs_warns_and_falls_back():
    # Force deterministic fallback via M < 2.
    rng = np.random.default_rng(901)
    scores_b = rng.normal(size=(1, 5))
    scores_a = scores_b + 2.0
    n_bootstrap = 250

    with pytest.warns(UserWarning, match="smooth_bootstrap_diffs_nested falling back"):
        smooth = smooth_bootstrap_diffs_nested(
            scores_a,
            scores_b,
            n_bootstrap=n_bootstrap,
            rng=np.random.default_rng(333),
        )

    standard = bootstrap_diffs_nested(
        scores_a,
        scores_b,
        n_bootstrap=n_bootstrap,
        rng=np.random.default_rng(333),
    )
    np.testing.assert_allclose(smooth, standard)


def test_bayes_and_smooth_bootstrap_ci_similar_to_standard_for_large_n():
    rng_data = np.random.default_rng(2026)
    values = rng_data.normal(loc=1.0, scale=0.75, size=200)
    observed_mean = float(values.mean())

    alpha = 0.05
    n_bootstrap = 4000

    boot_ci = bootstrap_ci_1d(
        values,
        observed_mean,
        method="bootstrap",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(400),
    )
    bayes_ci = bootstrap_ci_1d(
        values,
        observed_mean,
        method="bayes_bootstrap",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(401),
    )
    smooth_ci = bootstrap_ci_1d(
        values,
        observed_mean,
        method="smooth_bootstrap",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(402),
    )

    for ci in (boot_ci, bayes_ci, smooth_ci):
        assert np.isfinite(ci[0])
        assert np.isfinite(ci[1])
        assert ci[0] <= ci[1]

    boot_width = boot_ci[1] - boot_ci[0]
    bayes_width = bayes_ci[1] - bayes_ci[0]
    smooth_width = smooth_ci[1] - smooth_ci[0]
    assert boot_width > 0.0
    assert bayes_width > 0.0
    assert smooth_width > 0.0

    # With clear signal and N=200, alternative bootstrap variants should be
    # close to the classic percentile-bootstrap interval.
    np.testing.assert_allclose(bayes_ci, boot_ci, atol=0.06, rtol=0.0)
    np.testing.assert_allclose(smooth_ci, boot_ci, atol=0.08, rtol=0.0)


def test_bootstrap_t_median_warns_and_matches_bootstrap_with_same_rng_stream():
    values = np.array([0.1, 0.2, 0.3, 0.4, 5.0])
    observed_median = float(np.median(values))
    alpha = 0.1
    n_bootstrap = 600

    with pytest.warns(UserWarning, match="falling back to percentile bootstrap for 'median'"):
        ci_t = bootstrap_ci_1d(
            values,
            observed_median,
            method="bootstrap_t",
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            rng=np.random.default_rng(1234),
            statistic="median",
        )

    ci_boot = bootstrap_ci_1d(
        values,
        observed_median,
        method="bootstrap",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(1234),
        statistic="median",
    )

    np.testing.assert_allclose(ci_t, ci_boot)


def test_bootstrap_t_mean_degenerate_boot_ses_falls_back_to_percentile():
    # Mostly constant values can yield many bootstrap replicates with SE*=0.
    values = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    observed_mean = float(np.mean(values))
    alpha = 0.1
    n_bootstrap = 800

    ci = bootstrap_ci_1d(
        values,
        observed_mean,
        method="bootstrap_t",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(2026),
        statistic="mean",
    )

    assert np.isfinite(ci[0])
    assert np.isfinite(ci[1])
    assert ci[0] <= ci[1]


def test_bootstrap_t_mean_tiny_se_instability_falls_back_to_percentile():
    # Discrete tied values can create many tiny nonzero SE* values, which can
    # destabilize studentized pivots if not guarded.
    values = np.array([1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 2/3, 1/3, 1.0], dtype=float)
    observed_mean = float(np.mean(values))
    alpha = 0.05
    n_bootstrap = 800

    ci_t = bootstrap_ci_1d(
        values,
        observed_mean,
        method="bootstrap_t",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(314159),
        statistic="mean",
    )
    ci_boot = bootstrap_ci_1d(
        values,
        observed_mean,
        method="bootstrap",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(314159),
        statistic="mean",
    )

    np.testing.assert_allclose(ci_t, ci_boot)


def test_bootstrap_t_nested_tiny_se_instability_falls_back_to_percentile():
    # Nearly discrete cell means (many ties) can generate tiny SE* in nested
    # studentization; ensure robust fallback to percentile nested bootstrap.
    scores = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    observed_mean = float(np.mean(scores.mean(axis=1)))
    alpha = 0.05
    n_bootstrap = 800

    ci_t = bootstrap_t_ci_nested(
        scores,
        observed_mean,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=np.random.default_rng(271828),
    )
    assert np.isfinite(ci_t[0])
    assert np.isfinite(ci_t[1])
    assert ci_t[0] <= ci_t[1]
    # Regression guard: nested bootstrap-t should remain numerically stable.
    assert (ci_t[1] - ci_t[0]) < 2.0

class TestDegenerateSampleCI:
    """The zero-variance fallback shared by logit_t_ci_1d and beta_ci_1d.

    A constant sample used to produce a zero-width interval from every
    variance-driven method, which covers the truth with probability 0 for any
    population that isn't literally a point mass. See degenerate_sample_ci.
    """

    def test_all_ones_matches_clopper_pearson(self):
        # n successes out of n: the fallback reduces exactly to the two-sided
        # Clopper-Pearson interval, so the bounded-continuous path agrees with
        # the binary path at the boundary instead of contradicting it.
        n, alpha = 20, 0.05
        lo, hi = logit_t_ci_1d(np.ones(n), alpha)
        assert hi == pytest.approx(1.0)
        assert lo == pytest.approx((alpha / 2) ** (1 / n))

    def test_all_zeros_is_the_mirror_image(self):
        n, alpha = 20, 0.05
        lo, hi = logit_t_ci_1d(np.zeros(n), alpha)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0 - (alpha / 2) ** (1 / n))

    def test_constant_interior_sample_brackets_the_value(self):
        lo, hi = logit_t_ci_1d(np.full(20, 0.7), 0.05)
        assert lo < 0.7 < hi
        assert 0.0 <= lo and hi <= 1.0

    def test_width_shrinks_with_n(self):
        widths = [np.diff(logit_t_ci_1d(np.full(n, 0.8), 0.05))[0] for n in (10, 20, 50, 100)]
        assert widths == sorted(widths, reverse=True)

    def test_inert_on_non_degenerate_data(self):
        rng = np.random.default_rng(0)
        vals = rng.beta(4, 2, size=40)
        # Same answer as the plain delta method: the fallback only fires on a
        # genuinely constant sample.
        x_bar = float(np.mean(vals))
        lo, hi = logit_t_ci_1d(vals, 0.05)
        assert lo < x_bar < hi
        assert np.diff([lo, hi])[0] < 0.2

    def test_beta_ci_uses_the_same_fallback(self):
        assert beta_ci_1d(np.ones(20), 0.05) == pytest.approx(logit_t_ci_1d(np.ones(20), 0.05))

    def test_rescaled_likert_floor_stays_in_range_and_points_upward(self):
        # All responses at the floor of a 1-5 scale: the interval must live
        # inside [1, 5] and open upward, not straddle the scale's edge.
        lo, hi = rescaled_ci(logit_t_ci_1d, np.ones(30), 0.05, 1.0, 5.0)
        assert lo == pytest.approx(1.0)
        assert 1.0 < hi < 5.0

    def test_rescaled_paired_zero_diffs_straddle_zero(self):
        # Every paired difference exactly 0 is not proof of a zero effect.
        lo, hi = rescaled_ci(logit_t_ci_1d, np.zeros(30), 0.05, -1.0, 1.0)
        assert lo < 0.0 < hi

    def test_helper_respects_arbitrary_bounds(self):
        lo, hi = degenerate_sample_ci(3.0, 25, 0.05, lo=1.0, hi=5.0)
        assert 1.0 <= lo < 3.0 < hi <= 5.0
