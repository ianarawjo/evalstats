"""Direct tests for evalstats.ppi.correct guardrails and core behavior."""

from __future__ import annotations

import numpy as np
import pytest

from evalstats.ppi import correct


def _mean_estimator(Y, X=None):
    if X is None:
        return float(np.mean(Y))
    return float(np.mean(Y))


class TestPPICoreValidation:

    @pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.2, np.nan])
    def test_rejects_invalid_alpha(self, alpha):
        y_lab = np.array([1.0, 2.0, 3.0])
        y_hat_lab = np.array([1.0, 2.0, 3.0])
        y_hat_unlab = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="alpha"):
            correct(
                _mean_estimator,
                Y_lab=y_lab,
                Y_hat_lab=y_hat_lab,
                Y_hat_unlab=y_hat_unlab,
                alpha=alpha,
                n_boot=10,
                rng=0,
            )

    @pytest.mark.parametrize("n_boot", [0, -1, 1.5, True])
    def test_rejects_invalid_n_boot(self, n_boot):
        y_lab = np.array([1.0, 2.0, 3.0])
        y_hat_lab = np.array([1.0, 2.0, 3.0])
        y_hat_unlab = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="n_boot"):
            correct(
                _mean_estimator,
                Y_lab=y_lab,
                Y_hat_lab=y_hat_lab,
                Y_hat_unlab=y_hat_unlab,
                n_boot=n_boot,
                rng=0,
            )

    def test_rejects_empty_labeled_arrays(self):
        with pytest.raises(ValueError, match="non-empty"):
            correct(
                _mean_estimator,
                Y_lab=np.array([]),
                Y_hat_lab=np.array([]),
                Y_hat_unlab=np.array([1.0, 2.0]),
                n_boot=10,
                rng=0,
            )

    def test_rejects_empty_unlabeled_array(self):
        with pytest.raises(ValueError, match="non-empty"):
            correct(
                _mean_estimator,
                Y_lab=np.array([1.0, 2.0]),
                Y_hat_lab=np.array([1.0, 2.0]),
                Y_hat_unlab=np.array([]),
                n_boot=10,
                rng=0,
            )

    def test_rejects_non_finite_inputs(self):
        with pytest.raises(ValueError, match="Y_lab contains non-finite"):
            correct(
                _mean_estimator,
                Y_lab=np.array([1.0, np.nan]),
                Y_hat_lab=np.array([1.0, 2.0]),
                Y_hat_unlab=np.array([1.0, 2.0, 3.0]),
                n_boot=10,
                rng=0,
            )

    def test_rejects_mismatched_ndim_inputs(self):
        with pytest.raises(ValueError, match="same ndim"):
            correct(
                _mean_estimator,
                Y_lab=np.array([1.0, 2.0]),
                Y_hat_lab=np.array([1.0, 2.0]),
                Y_hat_unlab=np.array([[1.0], [2.0], [3.0]]),
                n_boot=10,
                rng=0,
            )


class TestPPICoreEstimatorSafety:

    def test_non_finite_point_estimate_propagates(self):
        def bad_estimator(Y, X=None):
            return np.nan

        r = correct(
            bad_estimator,
            Y_lab=np.array([1.0, 2.0, 3.0]),
            Y_hat_lab=np.array([1.0, 2.0, 3.0]),
            Y_hat_unlab=np.array([1.0, 2.0, 3.0, 4.0]),
            n_boot=10,
            rng=0,
        )
        assert np.isnan(r.estimate)

    def test_non_finite_bootstrap_estimate_propagates(self):
        state = {"i": 0}

        def flaky_estimator(Y, X=None):
            state["i"] += 1
            if state["i"] > 3:
                return np.inf
            return float(np.mean(Y))

        r = correct(
            flaky_estimator,
            Y_lab=np.array([1.0, 2.0, 3.0]),
            Y_hat_lab=np.array([1.0, 2.0, 3.0]),
            Y_hat_unlab=np.array([1.0, 2.0, 3.0, 4.0]),
            n_boot=10,
            rng=0,
        )
        assert np.isnan(r.ci_low) or np.isinf(r.ci_low) or np.isnan(r.ci_high) or np.isinf(r.ci_high)


class TestPPICoreHappyPath:

    def test_returns_finite_outputs_and_bounded_pvalue(self):
        y_lab = np.array([1.0, 2.0, 3.0, 4.0])
        y_hat_lab = np.array([1.1, 2.1, 3.1, 4.1])
        y_hat_unlab = np.array([0.9, 1.9, 2.9, 3.9, 4.9, 5.9])

        r = correct(
            _mean_estimator,
            Y_lab=y_lab,
            Y_hat_lab=y_hat_lab,
            Y_hat_unlab=y_hat_unlab,
            n_boot=50,
            rng=123,
            compute_pvalue=True,
        )

        assert np.isfinite(r.estimate)
        assert np.isfinite(r.ci_low)
        assert np.isfinite(r.ci_high)
        assert 0.0 <= r.p_value <= 1.0
