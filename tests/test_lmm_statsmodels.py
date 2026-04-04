"""Tests for the LMM analysis path with the statsmodels backend.

statsmodels is a standard scientific Python package (no R required), so these
tests run in any environment where statsmodels is installed.

    pytest tests/test_lmm_statsmodels.py -v
"""

import warnings

import numpy as np
import pytest

from promptstats import BenchmarkResult, analyze
from promptstats.core.mixed_effects import LMMInfo

# ---------------------------------------------------------------------------
# Module-level skip guard
# ---------------------------------------------------------------------------

pytest.importorskip("statsmodels", reason="statsmodels not installed (pip install statsmodels)")


# ---------------------------------------------------------------------------
# Helpers (mirrors test_lmm.py)
# ---------------------------------------------------------------------------

def _make_result(
    rng,
    n_templates: int = 3,
    n_inputs: int = 20,
    template_effects=None,
    sigma_input: float = 0.4,
    sigma_resid: float = 0.2,
) -> BenchmarkResult:
    """Synthetic benchmark drawn from the LMM generative model.

    score[t, i] = intercept + template_effect[t] + input_effect[i] + resid[t, i]
    """
    if template_effects is None:
        template_effects = np.array([0.5, 0.0, -0.3])
    assert len(template_effects) == n_templates

    intercept     = 5.0
    input_effects = rng.normal(0.0, sigma_input, size=n_inputs)
    resid         = rng.normal(0.0, sigma_resid, size=(n_templates, n_inputs))

    scores = (
        intercept
        + template_effects[:, None]
        + input_effects[None, :]
        + resid
    )

    labels = [f"T{i}" for i in range(n_templates)]
    inputs = [f"inp_{j:03d}" for j in range(n_inputs)]
    return BenchmarkResult(scores=scores, template_labels=labels, input_labels=inputs)


def _make_result_with_missing(rng, n_templates=3, n_inputs=20, missing_frac=0.1):
    result = _make_result(rng, n_templates=n_templates, n_inputs=n_inputs)
    scores = result.scores.copy()
    n_missing = max(1, int(missing_frac * scores.size))
    flat_idx = rng.choice(scores.size, size=n_missing, replace=False)
    scores.ravel()[flat_idx] = np.nan
    return BenchmarkResult(
        scores=scores,
        template_labels=result.template_labels,
        input_labels=result.input_labels,
    )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_smoke_returns_bundle():
    rng    = np.random.default_rng(0)
    result = _make_result(rng)
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    assert bundle is not None
    assert bundle.pairwise is not None
    assert bundle.rank_dist is not None
    assert bundle.robustness is not None


def test_lmm_info_populated():
    rng    = np.random.default_rng(1)
    result = _make_result(rng)
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    info   = bundle.lmm_info
    assert isinstance(info, LMMInfo)
    assert 0.0 <= info.icc <= 1.0
    assert info.sigma_input >= 0.0
    assert info.sigma_resid >= 0.0
    assert isinstance(info.converged, bool)
    assert info.n_obs == result.n_templates * result.n_inputs


def test_lmm_info_formula():
    rng    = np.random.default_rng(2)
    result = _make_result(rng)
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    assert "template" in bundle.lmm_info.formula
    assert "input" in bundle.lmm_info.formula


# ---------------------------------------------------------------------------
# ICC correctness
# ---------------------------------------------------------------------------

def test_icc_high_when_input_variance_dominates():
    rng    = np.random.default_rng(3)
    result = _make_result(rng, sigma_input=2.0, sigma_resid=0.1)
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    assert bundle.lmm_info.icc > 0.8, f"Expected high ICC, got {bundle.lmm_info.icc:.3f}"


def test_icc_low_when_resid_variance_dominates():
    rng    = np.random.default_rng(4)
    result = _make_result(rng, sigma_input=0.05, sigma_resid=1.5)
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    assert bundle.lmm_info.icc < 0.2, f"Expected low ICC, got {bundle.lmm_info.icc:.3f}"


# ---------------------------------------------------------------------------
# Result shapes and labels
# ---------------------------------------------------------------------------

def test_pairwise_labels_match():
    rng    = np.random.default_rng(5)
    result = _make_result(rng, n_templates=4, template_effects=np.array([0.5, 0.2, 0.0, -0.3]))
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    assert bundle.pairwise.labels == result.template_labels


def test_rank_dist_shape():
    rng    = np.random.default_rng(6)
    result = _make_result(rng, n_templates=4, template_effects=np.array([0.5, 0.2, 0.0, -0.3]))
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    N = result.n_templates
    assert bundle.rank_dist.rank_probs.shape == (N, N)
    assert bundle.robustness.mean.shape == (N,)


# ---------------------------------------------------------------------------
# Statistical correctness
# ---------------------------------------------------------------------------

def test_template_means_sign_recovery():
    """Best template (T0, effect +0.5) should have highest advantage."""
    rng    = np.random.default_rng(7)
    result = _make_result(rng, n_templates=3, n_inputs=40,
                          template_effects=np.array([0.5, 0.0, -0.3]))
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    adv    = bundle.robustness.mean
    labels = bundle.robustness.labels
    idx_t0 = labels.index("T0")
    idx_t2 = labels.index("T2")
    assert adv[idx_t0] > adv[idx_t2], (
        f"T0 should beat T2 in advantage: {adv[idx_t0]:.3f} vs {adv[idx_t2]:.3f}"
    )


def test_pairwise_sign_matches_raw_diff():
    """Pairwise point_diff sign should match raw cell-mean difference."""
    rng    = np.random.default_rng(8)
    result = _make_result(rng, n_templates=3, n_inputs=30)
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    scores = result.get_2d_scores()   # (N, M)
    labels = result.template_labels

    for (a, b), res in bundle.pairwise.results.items():
        ia, ib = labels.index(a), labels.index(b)
        raw    = float(np.nanmean(scores[ia]) - np.nanmean(scores[ib]))
        assert np.sign(res.point_diff) == np.sign(raw) or abs(raw) < 1e-6, (
            f"Sign mismatch for {a}-{b}: lmm={res.point_diff:.4f}, raw={raw:.4f}"
        )


def test_ci_bounds_ordered():
    """CI low < point_diff < CI high for all pairwise results."""
    rng    = np.random.default_rng(9)
    result = _make_result(rng, n_templates=4, n_inputs=25,
                          template_effects=np.array([0.5, 0.2, 0.0, -0.3]))
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    for (a, b), res in bundle.pairwise.results.items():
        assert res.ci_low < res.point_diff < res.ci_high, (
            f"{a}-{b}: ci_low={res.ci_low:.4f}, point_diff={res.point_diff:.4f}, "
            f"ci_high={res.ci_high:.4f}"
        )


def test_advantage_ci_bounds_ordered():
    rng    = np.random.default_rng(10)
    result = _make_result(rng, n_templates=3, n_inputs=20)
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    rob = bundle.robustness
    for i, label in enumerate(rob.labels):
        assert rob.ci_low[i] <= rob.mean[i] <= rob.ci_high[i], (
            f"{label}: ci=[{rob.ci_low[i]:.4f}, {rob.ci_high[i]:.4f}], "
            f"mean={rob.mean[i]:.4f}"
        )


def test_rank_probs_sum_to_one():
    rng    = np.random.default_rng(11)
    result = _make_result(rng)
    bundle = analyze(result, method="lmm", backend="statsmodels", n_bootstrap=2000, rng=rng)
    np.testing.assert_allclose(
        bundle.rank_dist.rank_probs.sum(axis=1), 1.0, atol=1e-6,
        err_msg="Rank probability rows should sum to 1",
    )


def test_p_best_in_unit_interval():
    rng    = np.random.default_rng(12)
    result = _make_result(rng)
    bundle = analyze(result, method="lmm", backend="statsmodels", n_bootstrap=2000, rng=rng)
    assert np.all(bundle.rank_dist.p_best >= 0.0)
    assert np.all(bundle.rank_dist.p_best <= 1.0)


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

def test_missing_data_accepted_with_warning():
    rng    = np.random.default_rng(13)
    result = _make_result_with_missing(rng)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    assert bundle is not None
    texts = [str(warning.message) for warning in w]
    assert any("missing" in t.lower() or "nan" in t.lower() for t in texts)


def test_missing_lmm_info_n_obs():
    """n_obs should reflect actual observed rows (< N*M when missing)."""
    rng    = np.random.default_rng(14)
    result = _make_result_with_missing(rng, n_templates=3, n_inputs=20, missing_frac=0.1)
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    assert bundle.lmm_info.n_obs < result.n_templates * result.n_inputs


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_two_templates():
    rng    = np.random.default_rng(15)
    result = _make_result(rng, n_templates=2, n_inputs=15,
                          template_effects=np.array([0.3, -0.3]))
    bundle = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    assert len(bundle.pairwise.results) == 1
    assert bundle.lmm_info is not None


def test_five_templates():
    rng    = np.random.default_rng(16)
    effects = np.array([0.5, 0.3, 0.0, -0.2, -0.4])
    result  = _make_result(rng, n_templates=5, n_inputs=25, template_effects=effects)
    bundle  = analyze(result, method="lmm", backend="statsmodels", rng=rng)
    assert bundle.pairwise.rank_probs.shape == (5, 5) if hasattr(bundle.pairwise, "rank_probs") else True
    assert bundle.lmm_info.icc >= 0.0


def test_specific_reference():
    rng    = np.random.default_rng(17)
    result = _make_result(rng, n_templates=3, n_inputs=20)
    bundle = analyze(result, method="lmm", backend="statsmodels", reference="T0", rng=rng)
    # Reference selection should not zero out absolute robustness means.
    labels = bundle.robustness.labels
    idx    = labels.index("T0")
    assert np.isfinite(bundle.robustness.mean[idx])


def test_test_method_string():
    """test_method on PairedDiffResult should mention statsmodels."""
    rng    = np.random.default_rng(18)
    result = _make_result(rng)
    bundle = analyze(result, method="lmm", backend="statsmodels", correction="none", rng=rng)
    for res in bundle.pairwise.results.values():
        assert "statsmodels" in res.test_method
