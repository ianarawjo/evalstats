"""Parity tests: statsmodels backend vs. pymer4/lme4 backend.

Both backends fit the same REML model on the same data.  REML point
estimates (fixed effects, variance components) must therefore agree very
closely — differences arise only from numerical solver precision and minor
API-level details.  Wald CIs differ slightly because pymer4 uses per-contrast
Satterthwaite degrees of freedom while statsmodels uses a single residual DF;
for the dataset sizes used here (M=30, N=3) those DFs are almost identical.

Skip conditions
---------------
* All tests are skipped if statsmodels is not installed.
* All tests are additionally skipped if pymer4 / R / lme4 is not reachable.

Run (when R is available):
    pytest tests/test_lmm_backend_parity.py -v
"""

import numpy as np
import pytest

from promptstats import BenchmarkResult, analyze
from promptstats.core.mixed_effects import LMMInfo

# ---------------------------------------------------------------------------
# Skip guards — both backends must be present
# ---------------------------------------------------------------------------

pytest.importorskip("statsmodels", reason="statsmodels not installed")

pymer4 = pytest.importorskip(
    "pymer4",
    reason="pymer4 not installed (pip install pymer4; needs R + lme4 + emmeans)",
)

try:
    from promptstats.core.mixed_effects import _require_pymer4
    _require_pymer4()
except Exception:
    pytest.skip("pymer4 installed but R/lme4 not reachable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_result(
    rng,
    n_templates: int = 3,
    n_inputs: int = 30,
    template_effects=None,
    sigma_input: float = 0.4,
    sigma_resid: float = 0.2,
) -> BenchmarkResult:
    """Synthetic data from the exact LMM generative model."""
    if template_effects is None:
        template_effects = np.array([0.5, 0.0, -0.3])
    assert len(template_effects) == n_templates

    intercept     = 5.0
    input_effects = rng.normal(0.0, sigma_input, size=n_inputs)
    resid         = rng.normal(0.0, sigma_resid, size=(n_templates, n_inputs))
    scores = intercept + template_effects[:, None] + input_effects[None, :] + resid

    labels = [f"T{i}" for i in range(n_templates)]
    inputs = [f"inp_{j:03d}" for j in range(n_inputs)]
    return BenchmarkResult(scores=scores, template_labels=labels, input_labels=inputs)


def _run_both(result, rng_seed=99, **kw):
    """Return (sm_bundle, py_bundle) for the same result."""
    sm = analyze(result, method="lmm", backend="statsmodels",
                 rng=np.random.default_rng(rng_seed), **kw)
    py = analyze(result, method="lmm", backend="pymer4",
                 rng=np.random.default_rng(rng_seed), **kw)
    return sm, py


def _make_result_with_missing(
    rng,
    n_templates: int = 4,
    n_inputs: int = 50,
    missing_fraction: float = 0.20,
    template_effects=None,
    sigma_input: float = 0.45,
    sigma_resid: float = 0.25,
) -> BenchmarkResult:
    """Synthetic LMM data with MCAR missing cells."""
    result = _make_result(
        rng,
        n_templates=n_templates,
        n_inputs=n_inputs,
        template_effects=template_effects,
        sigma_input=sigma_input,
        sigma_resid=sigma_resid,
    )

    scores = result.scores.copy()
    total = scores.size
    n_missing = max(1, int(round(missing_fraction * total)))
    missing_idx = rng.choice(total, size=n_missing, replace=False)
    scores.ravel()[missing_idx] = np.nan

    return BenchmarkResult(
        scores=scores,
        template_labels=result.template_labels,
        input_labels=result.input_labels,
    )


def _pairwise_pvals(bundle):
    """Return deterministic pairwise p-values as ``{(a, b): p}``."""
    return {
        pair: bundle.pairwise.results[pair].p_value
        for pair in sorted(bundle.pairwise.results.keys())
    }


def _assert_pvalue_close(sm_p: float, py_p: float, *, pair_label: str):
    """Assert p-values are close on a scale suitable for their magnitude.

    Why piecewise tolerance?
    - For moderate/large p, absolute differences are most interpretable.
    - For small p, relative/log-scale agreement is more stable.
    """
    sm_p = float(np.clip(sm_p, 1e-300, 1.0))
    py_p = float(np.clip(py_p, 1e-300, 1.0))

    # Below this floor, differences are usually numerical (both are decisively tiny).
    tiny_floor = 1e-12
    sm_cmp = max(sm_p, tiny_floor)
    py_cmp = max(py_p, tiny_floor)

    # Very small p-values: compare orders of magnitude.
    if max(sm_p, py_p) < 1e-3:
        log_gap = abs(np.log10(sm_cmp) - np.log10(py_cmp))
        assert log_gap < 0.60, (
            f"{pair_label}: tiny p-values diverge too much on log scale "
            f"(sm={sm_p:.3e}, py={py_p:.3e}, |Δlog10|={log_gap:.3f})"
        )
        return

    # Mid/small p-values: enforce relative closeness.
    if max(sm_p, py_p) < 0.20:
        rel = abs(sm_p - py_p) / max(py_p, 1e-12)
        assert rel < 0.30, (
            f"{pair_label}: p-values not relatively close "
            f"(sm={sm_p:.5g}, py={py_p:.5g}, rel={rel:.1%})"
        )
        return

    # Large p-values: absolute closeness is the right scale.
    abs_gap = abs(sm_p - py_p)
    assert abs_gap < 0.08, (
        f"{pair_label}: large p-values not absolutely close "
        f"(sm={sm_p:.5g}, py={py_p:.5g}, abs gap={abs_gap:.4f})"
    )


# ---------------------------------------------------------------------------
# Fixed-effect point estimates
# ---------------------------------------------------------------------------

def test_pairwise_point_diff_agreement():
    """Pairwise point estimates must agree to within 0.01 (REML is exact)."""
    rng    = np.random.default_rng(0)
    result = _make_result(rng)
    sm, py = _run_both(result, correction="none")

    labels = result.template_labels
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            sm_r = sm.pairwise.get(a, b)
            py_r = py.pairwise.get(a, b)
            assert abs(sm_r.point_diff - py_r.point_diff) < 0.01, (
                f"{a}-{b}: sm={sm_r.point_diff:.5f}, py={py_r.point_diff:.5f}"
            )


def test_point_advantage_agreement():
    """Per-template mean advantages must agree to within 0.01."""
    rng    = np.random.default_rng(1)
    result = _make_result(rng)
    sm, py = _run_both(result)

    np.testing.assert_allclose(
        sm.point_advantage.point_advantages,
        py.point_advantage.point_advantages,
        atol=0.01,
        err_msg="point_advantages disagree between backends",
    )


def test_pairwise_sign_agreement():
    """point_diff must have the same sign for every pair."""
    rng    = np.random.default_rng(2)
    result = _make_result(rng, n_templates=4,
                          template_effects=np.array([0.5, 0.2, 0.0, -0.3]))
    sm, py = _run_both(result, correction="none")

    for (a, b) in sm.pairwise.results:
        sm_r = sm.pairwise.get(a, b)
        py_r = py.pairwise.get(a, b)
        assert np.sign(sm_r.point_diff) == np.sign(py_r.point_diff), (
            f"{a}-{b}: sm={sm_r.point_diff:.4f}, py={py_r.point_diff:.4f}"
        )


# ---------------------------------------------------------------------------
# Variance components / ICC
# ---------------------------------------------------------------------------

def test_icc_agreement():
    """ICC must agree to within 0.05 (same REML objective, near-identical estimates)."""
    rng    = np.random.default_rng(3)
    result = _make_result(rng, n_inputs=40)
    sm, py = _run_both(result)

    assert abs(sm.lmm_info.icc - py.lmm_info.icc) < 0.05, (
        f"ICC: sm={sm.lmm_info.icc:.4f}, py={py.lmm_info.icc:.4f}"
    )


def test_sigma_input_agreement():
    """Between-input SD must agree to within 0.05."""
    rng    = np.random.default_rng(4)
    result = _make_result(rng, n_inputs=40)
    sm, py = _run_both(result)

    assert abs(sm.lmm_info.sigma_input - py.lmm_info.sigma_input) < 0.05, (
        f"sigma_input: sm={sm.lmm_info.sigma_input:.4f}, py={py.lmm_info.sigma_input:.4f}"
    )


def test_sigma_resid_agreement():
    """Residual SD must agree to within 0.05."""
    rng    = np.random.default_rng(5)
    result = _make_result(rng, n_inputs=40)
    sm, py = _run_both(result)

    assert abs(sm.lmm_info.sigma_resid - py.lmm_info.sigma_resid) < 0.05, (
        f"sigma_resid: sm={sm.lmm_info.sigma_resid:.4f}, py={py.lmm_info.sigma_resid:.4f}"
    )


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def test_pairwise_ci_width_agreement():
    """CI widths (ci_high - ci_low) must agree within 10%.

    Satterthwaite DFs (pymer4) vs. residual DF (statsmodels) differ
    slightly; for M=30 they are essentially the same, so widths are close.
    """
    rng    = np.random.default_rng(6)
    result = _make_result(rng, n_inputs=30)
    sm, py = _run_both(result, correction="none")

    labels = result.template_labels
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b   = labels[i], labels[j]
            sm_r   = sm.pairwise.get(a, b)
            py_r   = py.pairwise.get(a, b)
            sm_w   = sm_r.ci_high - sm_r.ci_low
            py_w   = py_r.ci_high - py_r.ci_low
            # Relative difference should be < 10 %
            rel = abs(sm_w - py_w) / max(abs(py_w), 1e-8)
            assert rel < 0.10, (
                f"{a}-{b}: CI widths sm={sm_w:.4f}, py={py_w:.4f} "
                f"(relative diff={rel:.1%})"
            )


def test_advantage_ci_width_agreement():
    """Advantage CI widths must agree within 10%."""
    rng    = np.random.default_rng(7)
    result = _make_result(rng, n_inputs=30)
    sm, py = _run_both(result)

    sm_widths = sm.point_advantage.bootstrap_ci_high - sm.point_advantage.bootstrap_ci_low
    py_widths = py.point_advantage.bootstrap_ci_high - py.point_advantage.bootstrap_ci_low
    rel = np.abs(sm_widths - py_widths) / np.maximum(np.abs(py_widths), 1e-8)
    assert np.all(rel < 0.10), (
        f"Advantage CI width relative differences: {rel.tolist()}"
    )


# ---------------------------------------------------------------------------
# p-value ordering
# ---------------------------------------------------------------------------

def test_pairwise_p_value_ordering():
    """Both backends must rank pairs the same by p-value direction.

    If sm says pair A-B has a smaller p-value than A-C, py must agree.
    """
    rng    = np.random.default_rng(8)
    result = _make_result(rng, n_templates=4,
                          template_effects=np.array([0.8, 0.2, 0.0, -0.4]),
                          n_inputs=40)
    sm, py = _run_both(result, correction="none")

    pairs = list(sm.pairwise.results.keys())
    sm_pvals = [sm.pairwise.results[p].p_value for p in pairs]
    py_pvals = [py.pairwise.results[p].p_value for p in pairs]

    sm_order = np.argsort(sm_pvals)
    py_order = np.argsort(py_pvals)

    # Rank-order correlation of p-values: Spearman should be > 0.9
    from scipy.stats import spearmanr
    rho, _ = spearmanr(sm_pvals, py_pvals)
    assert rho > 0.90, f"p-value rank correlation between backends: rho={rho:.3f}"


def test_pairwise_p_value_closeness_many_templates_small_effects():
    """Hard case: many pairs + tightly spaced effects should still yield close p-values."""
    rng = np.random.default_rng(812)
    result = _make_result(
        rng,
        n_templates=6,
        n_inputs=60,
        template_effects=np.array([0.12, 0.09, 0.05, 0.00, -0.04, -0.08]),
        sigma_input=0.55,
        sigma_resid=0.35,
    )
    sm, py = _run_both(result, correction="none")

    sm_p = _pairwise_pvals(sm)
    py_p = _pairwise_pvals(py)
    assert sm_p.keys() == py_p.keys()

    for pair in sm_p:
        _assert_pvalue_close(sm_p[pair], py_p[pair], pair_label=f"{pair[0]}-{pair[1]}")


def test_pairwise_p_value_closeness_extreme_signal_log_scale():
    """Hard case: very strong effects create tiny p-values; compare on log scale."""
    rng = np.random.default_rng(913)
    result = _make_result(
        rng,
        n_templates=4,
        n_inputs=45,
        template_effects=np.array([1.20, 0.40, 0.00, -0.80]),
        sigma_input=0.35,
        sigma_resid=0.15,
    )
    sm, py = _run_both(result, correction="none")

    sm_p = _pairwise_pvals(sm)
    py_p = _pairwise_pvals(py)

    # Aggregate check: typical difference should be well below one order of magnitude.
    log_gaps = []
    for pair in sm_p:
        s = max(float(np.clip(sm_p[pair], 1e-300, 1.0)), 1e-12)
        p = max(float(np.clip(py_p[pair], 1e-300, 1.0)), 1e-12)
        log_gaps.append(abs(np.log10(s) - np.log10(p)))
    assert float(np.median(log_gaps)) < 0.50, f"Median |Δlog10(p)| too large: {log_gaps}"

    for pair in sm_p:
        _assert_pvalue_close(sm_p[pair], py_p[pair], pair_label=f"{pair[0]}-{pair[1]}")


@pytest.mark.parametrize("correction", ["bonferroni", "holm", "fdr_bh"])
def test_corrected_pairwise_p_value_closeness(correction):
    """Adjusted p-values should remain closely aligned under common corrections."""
    rng = np.random.default_rng(1017)
    result = _make_result(
        rng,
        n_templates=5,
        n_inputs=50,
        template_effects=np.array([0.45, 0.20, 0.05, -0.10, -0.35]),
        sigma_input=0.45,
        sigma_resid=0.25,
    )
    sm, py = _run_both(result, correction=correction)

    sm_p = _pairwise_pvals(sm)
    py_p = _pairwise_pvals(py)
    assert sm_p.keys() == py_p.keys()

    for pair in sm_p:
        _assert_pvalue_close(
            sm_p[pair],
            py_p[pair],
            pair_label=f"{pair[0]}-{pair[1]} ({correction})",
        )


@pytest.mark.parametrize("seed", [1201, 1202, 1203])
def test_pairwise_p_value_closeness_across_random_draws(seed):
    """Hard case: repeat near-threshold setup across seeds; p-value drift stays small."""
    rng = np.random.default_rng(seed)
    result = _make_result(
        rng,
        n_templates=4,
        n_inputs=36,
        template_effects=np.array([0.22, 0.10, 0.00, -0.10]),
        sigma_input=0.50,
        sigma_resid=0.30,
    )
    sm, py = _run_both(result, correction="none")

    sm_p = _pairwise_pvals(sm)
    py_p = _pairwise_pvals(py)
    deltas = np.array([abs(sm_p[pair] - py_p[pair]) for pair in sm_p], dtype=float)

    # Per-dataset aggregate guards keep the test strict but stable.
    assert float(np.mean(deltas)) < 0.04, f"mean |Δp| too large: {deltas.tolist()}"
    assert float(np.max(deltas)) < 0.12, f"max |Δp| too large: {deltas.tolist()}"


@pytest.mark.parametrize("correction", ["none", "holm", "fdr_bh"])
@pytest.mark.parametrize("missing_fraction", [0.10, 0.25])
def test_pairwise_p_value_closeness_with_mcar_missingness(correction, missing_fraction):
    """Hard case: p-values remain close with MCAR missing cells and corrections."""
    rng = np.random.default_rng(1307)
    result = _make_result_with_missing(
        rng,
        n_templates=5,
        n_inputs=70,
        missing_fraction=missing_fraction,
        template_effects=np.array([0.35, 0.18, 0.04, -0.08, -0.28]),
        sigma_input=0.50,
        sigma_resid=0.30,
    )

    sm, py = _run_both(result, correction=correction)
    sm_p = _pairwise_pvals(sm)
    py_p = _pairwise_pvals(py)

    assert sm_p.keys() == py_p.keys()
    assert sm.lmm_info.n_obs == py.lmm_info.n_obs

    for pair in sm_p:
        _assert_pvalue_close(
            sm_p[pair],
            py_p[pair],
            pair_label=f"{pair[0]}-{pair[1]} (miss={missing_fraction:.0%}, {correction})",
        )

    deltas = np.array([abs(sm_p[pair] - py_p[pair]) for pair in sm_p], dtype=float)
    assert float(np.mean(deltas)) < 0.05, f"mean |Δp| too large: {deltas.tolist()}"
    assert float(np.max(deltas)) < 0.15, f"max |Δp| too large: {deltas.tolist()}"


def test_significance_agreement():
    """Pairs called significant (p<0.05) by one backend should mostly agree.

    For the large-effect dataset (template_effects[0] - effects[-1] = 1.2,
    M=40), the strongly separated pairs should be significant in both.
    """
    rng    = np.random.default_rng(9)
    result = _make_result(rng, n_templates=3,
                          template_effects=np.array([0.6, 0.0, -0.6]),
                          n_inputs=40)
    sm, py = _run_both(result, correction="none")

    labels = result.template_labels
    # T0 vs T2 (effect = 1.2) must be significant in both
    sm_r = sm.pairwise.get("T0", "T2")
    py_r = py.pairwise.get("T0", "T2")
    assert sm_r.p_value < 0.05, f"sm p={sm_r.p_value:.4g} not < 0.05"
    assert py_r.p_value < 0.05, f"py p={py_r.p_value:.4g} not < 0.05"


# ---------------------------------------------------------------------------
# Rank distribution
# ---------------------------------------------------------------------------

def test_rank_probs_agreement():
    """Rank probability matrices must agree within 0.10 (Monte Carlo noise included)."""
    rng    = np.random.default_rng(10)
    result = _make_result(rng, n_inputs=30)
    sm, py = _run_both(result, n_bootstrap=20_000)

    np.testing.assert_allclose(
        sm.rank_dist.rank_probs,
        py.rank_dist.rank_probs,
        atol=0.10,
        err_msg="rank_probs disagree between backends beyond Monte Carlo tolerance",
    )


def test_p_best_ordering_agreement():
    """P(best) ranking must be identical between backends."""
    rng    = np.random.default_rng(11)
    result = _make_result(rng, n_templates=3,
                          template_effects=np.array([0.5, 0.0, -0.3]),
                          n_inputs=40)
    sm, py = _run_both(result, n_bootstrap=10_000)

    sm_best = int(np.argmax(sm.rank_dist.p_best))
    py_best = int(np.argmax(py.rank_dist.p_best))
    assert sm_best == py_best, (
        f"Best-ranked template differs: sm={sm.rank_dist.labels[sm_best]}, "
        f"py={py.rank_dist.labels[py_best]}"
    )
