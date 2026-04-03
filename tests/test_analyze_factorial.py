"""Tests for analyze_factorial with and without run_col.

    pytest tests/test_analyze_factorial.py -v
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import promptstats as ps
from promptstats.core.mixed_effects import FactorialLMMInfo

pytest.importorskip("statsmodels", reason="statsmodels not installed (pip install statsmodels)")


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def _make_factorial_df(
    rng,
    n_inputs: int = 20,
    chunkers: tuple = ("fixed", "semantic"),
    retrievals: tuple = ("bm25", "dense"),
    sigma_input: float = 0.4,
    sigma_resid: float = 0.2,
    n_runs: int = 0,
    chunker_effects: dict | None = None,
    retrieval_effects: dict | None = None,
) -> pd.DataFrame:
    """Synthetic long-form factorial DataFrame.

    Generative model::

        score = 5.0 + input_effect + chunker_effect + retrieval_effect + noise

    When *n_runs* > 0 a ``'seed'`` column is added with integer run IDs.
    """
    if chunker_effects is None:
        chunker_effects = {c: i * 0.2 for i, c in enumerate(chunkers)}
    if retrieval_effects is None:
        retrieval_effects = {r: i * 0.3 for i, r in enumerate(retrievals)}

    input_effects = rng.normal(0.0, sigma_input, size=n_inputs)
    rows = []
    run_range = range(n_runs) if n_runs > 0 else [None]

    for seed in run_range:
        for i in range(n_inputs):
            for c in chunkers:
                for ret in retrievals:
                    score = (
                        5.0
                        + input_effects[i]
                        + chunker_effects[c]
                        + retrieval_effects[ret]
                        + rng.normal(0.0, sigma_resid)
                    )
                    row = {
                        "input_id": f"q{i:03d}",
                        "chunker": c,
                        "retrieval": ret,
                        "score": score,
                    }
                    if n_runs > 0:
                        row["seed"] = seed
                    rows.append(row)

    return pd.DataFrame(rows)


def _assert_factorial_bundles_close(
    bundle_a,
    bundle_b,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> None:
    """Assert that two factorial analysis bundles are numerically aligned."""
    labels = bundle_a.benchmark.template_labels
    assert labels == bundle_b.benchmark.template_labels

    # Pairwise outputs
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1:]:
            left = bundle_a.pairwise.get(label_a, label_b)
            right = bundle_b.pairwise.get(label_a, label_b)
            assert left.template_a == right.template_a
            assert left.template_b == right.template_b
            assert left.test_method == right.test_method
            assert left.n_inputs == right.n_inputs
            assert left.n_runs == right.n_runs
            assert left.statistic == right.statistic
            assert_allclose(left.point_diff, right.point_diff, atol=atol, rtol=rtol)
            assert_allclose(left.std_diff, right.std_diff, atol=atol, rtol=rtol)
            assert_allclose(left.ci_low, right.ci_low, atol=atol, rtol=rtol)
            assert_allclose(left.ci_high, right.ci_high, atol=atol, rtol=rtol)
            assert_allclose(left.p_value, right.p_value, atol=atol, rtol=rtol)
            assert_allclose(
                left.per_input_diffs,
                right.per_input_diffs,
                atol=atol,
                rtol=rtol,
            )

    assert_allclose(bundle_a.robustness.ci_low, bundle_b.robustness.ci_low, atol=atol, rtol=rtol)
    assert_allclose(bundle_a.robustness.ci_high, bundle_b.robustness.ci_high, atol=atol, rtol=rtol)

    # Rank distribution
    rd_a = bundle_a.rank_dist
    rd_b = bundle_b.rank_dist
    assert rd_a.labels == rd_b.labels
    assert rd_a.n_bootstrap == rd_b.n_bootstrap
    assert_allclose(rd_a.rank_probs, rd_b.rank_probs, atol=atol, rtol=rtol)
    assert_allclose(rd_a.expected_ranks, rd_b.expected_ranks, atol=atol, rtol=rtol)
    assert_allclose(rd_a.p_best, rd_b.p_best, atol=atol, rtol=rtol)

    # Robustness metrics
    rb_a = bundle_a.robustness
    rb_b = bundle_b.robustness
    assert rb_a.labels == rb_b.labels
    assert rb_a.failure_threshold == rb_b.failure_threshold
    assert_allclose(rb_a.mean, rb_b.mean, atol=atol, rtol=rtol)
    assert_allclose(rb_a.median, rb_b.median, atol=atol, rtol=rtol)
    assert_allclose(rb_a.std, rb_b.std, atol=atol, rtol=rtol)
    assert_allclose(rb_a.cv, rb_b.cv, atol=atol, rtol=rtol, equal_nan=True)
    assert_allclose(rb_a.iqr, rb_b.iqr, atol=atol, rtol=rtol)
    assert_allclose(rb_a.cvar_10, rb_b.cvar_10, atol=atol, rtol=rtol)
    for p in (10, 25, 50, 75, 90):
        assert_allclose(
            rb_a.percentiles[p],
            rb_b.percentiles[p],
            atol=atol,
            rtol=rtol,
        )

    # LMM diagnostics
    info_a = bundle_a.factorial_lmm_info
    info_b = bundle_b.factorial_lmm_info
    assert isinstance(info_a, FactorialLMMInfo)
    assert isinstance(info_b, FactorialLMMInfo)
    assert info_a.factor_names == info_b.factor_names
    assert info_a.formula == info_b.formula
    assert info_a.n_obs == info_b.n_obs
    assert info_a.converged == info_b.converged
    assert_allclose(info_a.icc, info_b.icc, atol=atol, rtol=rtol)
    assert_allclose(info_a.sigma_input, info_b.sigma_input, atol=atol, rtol=rtol)
    assert_allclose(info_a.sigma_resid, info_b.sigma_resid, atol=atol, rtol=rtol)

    tests_a = info_a.factor_tests.sort_values("term").reset_index(drop=True)
    tests_b = info_b.factor_tests.sort_values("term").reset_index(drop=True)
    assert tests_a["term"].tolist() == tests_b["term"].tolist()
    assert_allclose(tests_a["statistic"].to_numpy(), tests_b["statistic"].to_numpy(), atol=atol, rtol=rtol)
    assert_allclose(tests_a["df"].to_numpy(), tests_b["df"].to_numpy(), atol=atol, rtol=rtol)
    assert_allclose(tests_a["p_value"].to_numpy(), tests_b["p_value"].to_numpy(), atol=atol, rtol=rtol)

    for factor in info_a.factor_names:
        mm_a = info_a.marginal_means[factor].sort_values("level").reset_index(drop=True)
        mm_b = info_b.marginal_means[factor].sort_values("level").reset_index(drop=True)
        assert mm_a["level"].tolist() == mm_b["level"].tolist()
        for col in ("mean", "se", "ci_low", "ci_high"):
            assert_allclose(mm_a[col].to_numpy(), mm_b[col].to_numpy(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_smoke_no_runs():
    rng = np.random.default_rng(0)
    data = _make_factorial_df(rng)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"])
    assert bundle is not None
    assert bundle.pairwise is not None
    assert bundle.rank_dist is not None
    assert bundle.robustness is not None


def test_smoke_with_runs():
    rng = np.random.default_rng(1)
    data = _make_factorial_df(rng, n_runs=5)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"], run_col="seed")
    assert bundle is not None
    assert bundle.pairwise is not None
    assert bundle.rank_dist is not None
    assert bundle.robustness is not None


# ---------------------------------------------------------------------------
# factorial_lmm_info contents
# ---------------------------------------------------------------------------

def test_factorial_lmm_info_without_runs():
    rng = np.random.default_rng(2)
    data = _make_factorial_df(rng)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"])
    info = bundle.factorial_lmm_info
    assert isinstance(info, FactorialLMMInfo)
    assert set(info.factor_names) == {"chunker", "retrieval"}
    assert 0.0 <= info.icc <= 1.0
    assert info.sigma_input >= 0.0
    assert info.sigma_resid >= 0.0
    assert isinstance(info.converged, bool)
    # factor_tests should reference both main effects
    terms = info.factor_tests["term"].tolist()
    assert any("chunker" in t for t in terms)
    assert any("retrieval" in t for t in terms)


def test_factorial_lmm_info_with_runs():
    rng = np.random.default_rng(3)
    data = _make_factorial_df(rng, n_runs=5)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"], run_col="seed")
    info = bundle.factorial_lmm_info
    assert isinstance(info, FactorialLMMInfo)
    assert set(info.factor_names) == {"chunker", "retrieval"}
    assert 0.0 <= info.icc <= 1.0


# ---------------------------------------------------------------------------
# n_obs: reflects cell-means vs per-run expansion
# ---------------------------------------------------------------------------

def test_n_obs_without_runs():
    rng = np.random.default_rng(4)
    n_inputs, chunkers, retrievals = 20, ("fixed", "semantic"), ("bm25", "dense")
    data = _make_factorial_df(rng, n_inputs=n_inputs, chunkers=chunkers, retrievals=retrievals)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"])
    n_templates = len(chunkers) * len(retrievals)
    assert bundle.factorial_lmm_info.n_obs == n_templates * n_inputs


def test_n_obs_with_runs():
    rng = np.random.default_rng(5)
    n_inputs, n_runs = 20, 5
    chunkers, retrievals = ("fixed", "semantic"), ("bm25", "dense")
    data = _make_factorial_df(rng, n_inputs=n_inputs, n_runs=n_runs, chunkers=chunkers, retrievals=retrievals)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"], run_col="seed")
    n_templates = len(chunkers) * len(retrievals)
    # LMM fitted on individual observations: N_templates × M_inputs × R_runs
    assert bundle.factorial_lmm_info.n_obs == n_templates * n_inputs * n_runs


# ---------------------------------------------------------------------------
# seed_variance
# ---------------------------------------------------------------------------

def test_seed_variance_none_without_runs():
    rng = np.random.default_rng(6)
    data = _make_factorial_df(rng)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"])
    assert bundle.seed_variance is None


def test_seed_variance_populated_with_runs():
    rng = np.random.default_rng(7)
    data = _make_factorial_df(rng, n_runs=5)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"], run_col="seed")
    assert bundle.seed_variance is not None


# ---------------------------------------------------------------------------
# CIs widen when seed variance is present
# ---------------------------------------------------------------------------

def test_n_obs_larger_with_runs():
    """n_obs must be R times larger when run_col is provided."""
    n_inputs, n_runs = 20, 5
    chunkers, retrievals = ("fixed", "semantic"), ("bm25", "dense")

    rng_no = np.random.default_rng(8)
    data_no = _make_factorial_df(rng_no, n_inputs=n_inputs, chunkers=chunkers, retrievals=retrievals)
    bundle_no = ps.analyze_factorial(data_no, factors=["chunker", "retrieval"])

    rng_with = np.random.default_rng(8)
    data_with = _make_factorial_df(rng_with, n_inputs=n_inputs, n_runs=n_runs, chunkers=chunkers, retrievals=retrievals)
    bundle_with = ps.analyze_factorial(data_with, factors=["chunker", "retrieval"], run_col="seed")

    assert bundle_with.factorial_lmm_info.n_obs == bundle_no.factorial_lmm_info.n_obs * n_runs


# ---------------------------------------------------------------------------
# BenchmarkResult scores shape
# ---------------------------------------------------------------------------

def test_scores_shape_without_runs():
    rng = np.random.default_rng(10)
    data = _make_factorial_df(rng, n_inputs=20)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"])
    scores = bundle.benchmark.scores
    assert scores.ndim == 2
    assert scores.shape[0] == 4  # 2 chunkers × 2 retrievals
    assert scores.shape[1] == 20


def test_scores_shape_with_runs():
    rng = np.random.default_rng(11)
    n_runs = 5
    data = _make_factorial_df(rng, n_inputs=20, n_runs=n_runs)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"], run_col="seed")
    scores = bundle.benchmark.scores
    assert scores.ndim == 3
    assert scores.shape[0] == 4   # templates
    assert scores.shape[1] == 20  # inputs
    assert scores.shape[2] == n_runs


# ---------------------------------------------------------------------------
# Statistical reasonableness checks
# ---------------------------------------------------------------------------

def test_effect_sizes_and_ranking_are_reasonable_no_runs():
    """Recovered effects should align with the synthetic generating process."""
    rng = np.random.default_rng(15)
    data = _make_factorial_df(
        rng,
        n_inputs=80,
        sigma_input=0.2,
        sigma_resid=0.1,
    )
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"], n_sim=3000)

    means = dict(zip(bundle.benchmark.template_labels, bundle.robustness.mean))

    # True means under defaults are:
    # fixed|bm25=5.0, fixed|dense=5.3, semantic|bm25=5.2, semantic|dense=5.5.
    assert means["semantic|dense"] > means["fixed|dense"] > means["fixed|bm25"]
    assert means["semantic|dense"] > means["semantic|bm25"] > means["fixed|bm25"]

    retrieval_delta_fixed = means["fixed|dense"] - means["fixed|bm25"]
    retrieval_delta_sem = means["semantic|dense"] - means["semantic|bm25"]
    chunker_delta_bm25 = means["semantic|bm25"] - means["fixed|bm25"]
    chunker_delta_dense = means["semantic|dense"] - means["fixed|dense"]

    assert retrieval_delta_fixed == pytest.approx(0.3, abs=0.09)
    assert retrieval_delta_sem == pytest.approx(0.3, abs=0.09)
    assert chunker_delta_bm25 == pytest.approx(0.2, abs=0.09)
    assert chunker_delta_dense == pytest.approx(0.2, abs=0.09)

    pair = bundle.pairwise.get("semantic|dense", "fixed|bm25")
    assert pair.point_diff == pytest.approx(0.5, abs=0.1)
    assert pair.ci_low > 0.0
    assert pair.p_value < 0.01

    best_idx = int(np.argmax(bundle.rank_dist.p_best))
    assert bundle.rank_dist.labels[best_idx] == "semantic|dense"
    assert bundle.rank_dist.p_best[best_idx] > 0.7


def test_effect_sizes_and_ranking_are_reasonable_with_runs():
    """Signal recovery should remain sensible when repeated runs are present."""
    rng = np.random.default_rng(16)
    data = _make_factorial_df(
        rng,
        n_inputs=60,
        n_runs=5,
        sigma_input=0.2,
        sigma_resid=0.1,
    )
    bundle = ps.analyze_factorial(
        data,
        factors=["chunker", "retrieval"],
        run_col="seed",
        n_sim=3000,
    )

    means = dict(zip(bundle.benchmark.template_labels, bundle.robustness.mean))
    assert means["semantic|dense"] > means["fixed|bm25"]

    pair = bundle.pairwise.get("semantic|dense", "fixed|bm25")
    assert pair.point_diff == pytest.approx(0.5, abs=0.12)
    assert pair.ci_low > 0.0
    assert pair.p_value < 0.01

    best_idx = int(np.argmax(bundle.rank_dist.p_best))
    assert bundle.rank_dist.labels[best_idx] == "semantic|dense"
    assert bundle.rank_dist.p_best[best_idx] > 0.65


# ---------------------------------------------------------------------------
# Backend parity helpers and tests
# ---------------------------------------------------------------------------

def _assert_factorial_bundles_compatible(bundle_a, bundle_b, *, atol: float = 0.05) -> None:
    """Assert that two factorial bundles from different backends are broadly compatible.

    Unlike ``_assert_factorial_bundles_close``, this function:
    - Skips ``test_method`` and ``formula`` string comparisons (differ across backends).
    - Uses a loose numerical tolerance for LMM-derived quantities (CIs, p-values).
    - Checks factor test *term names* and p-value *direction* (sig at α=0.05) rather
      than exact statistic values (chi2 vs F scales differ between backends).
    - Requires marginal-mean point estimates to be close but not identical.
    """
    labels = bundle_a.benchmark.template_labels
    assert labels == bundle_b.benchmark.template_labels

    # Pairwise: point estimates and CI direction should agree; p-values agree on α=0.05
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1:]:
            left = bundle_a.pairwise.get(label_a, label_b)
            right = bundle_b.pairwise.get(label_a, label_b)
            assert left.template_a == right.template_a
            assert left.template_b == right.template_b
            assert left.n_inputs == right.n_inputs
            assert left.n_runs == right.n_runs
            assert_allclose(left.point_diff, right.point_diff, atol=atol)
            # CI direction must agree
            assert (left.ci_low > 0) == (right.ci_low > 0), (
                f"CI direction disagrees for {label_a} vs {label_b}: "
                f"sm={left.ci_low:.4f} py={right.ci_low:.4f}"
            )
            # Significance at α=0.05 must agree
            assert (left.p_value < 0.05) == (right.p_value < 0.05), (
                f"p-value significance disagrees for {label_a} vs {label_b}: "
                f"sm={left.p_value:.4f} py={right.p_value:.4f}"
            )

    # Robustness means should be close (derived from raw data, not backend-specific)
    rb_a = bundle_a.robustness
    rb_b = bundle_b.robustness
    assert rb_a.labels == rb_b.labels
    assert_allclose(rb_a.mean, rb_b.mean, atol=atol)
    assert np.sign(rb_a.mean).tolist() == np.sign(rb_b.mean).tolist()

    # LMM diagnostics
    info_a = bundle_a.factorial_lmm_info
    info_b = bundle_b.factorial_lmm_info
    assert isinstance(info_a, FactorialLMMInfo)
    assert isinstance(info_b, FactorialLMMInfo)
    assert set(info_a.factor_names) == set(info_b.factor_names)
    assert info_a.n_obs == info_b.n_obs
    # Both should report convergence
    assert info_a.converged
    assert info_b.converged
    # ICC and variance components should be in the same ballpark
    assert_allclose(info_a.icc, info_b.icc, atol=0.1)
    assert_allclose(info_a.sigma_input, info_b.sigma_input, atol=0.15)
    assert_allclose(info_a.sigma_resid, info_b.sigma_resid, atol=0.15)

    # Factor tests: term names and p-value significance must agree.
    # Normalize away C(...) wrapping from statsmodels so names are comparable.
    import re as _re
    def _norm_term(t: str) -> str:
        return ":".join(
            _re.sub(r"^C\((.+)\)$", r"\1", part)
            for part in t.split(":")
        )

    tests_a = info_a.factor_tests.copy()
    tests_b = info_b.factor_tests.copy()
    tests_a["_norm"] = tests_a["term"].map(_norm_term)
    tests_b["_norm"] = tests_b["term"].map(_norm_term)
    tests_a = tests_a.sort_values("_norm").reset_index(drop=True)
    tests_b = tests_b.sort_values("_norm").reset_index(drop=True)
    assert tests_a["_norm"].tolist() == tests_b["_norm"].tolist(), (
        f"Factor test term mismatch: {tests_a['term'].tolist()} vs {tests_b['term'].tolist()}"
    )
    for (_, row_a), (_, row_b) in zip(tests_a.iterrows(), tests_b.iterrows()):
        assert (row_a["p_value"] < 0.05) == (row_b["p_value"] < 0.05), (
            f"Factor test significance disagrees for '{row_a['_norm']}': "
            f"sm={row_a['p_value']:.4f} py={row_b['p_value']:.4f}"
        )

    # Marginal means: point estimates should be close
    for factor in info_a.factor_names:
        mm_a = info_a.marginal_means[factor].sort_values("level").reset_index(drop=True)
        mm_b = info_b.marginal_means[factor].sort_values("level").reset_index(drop=True)
        assert mm_a["level"].tolist() == mm_b["level"].tolist()
        assert_allclose(mm_a["mean"].to_numpy(), mm_b["mean"].to_numpy(), atol=atol)


def _has_r_car() -> bool:
    """Return True if the R 'car' package is available via rpy2."""
    try:
        from rpy2.robjects.packages import importr, PackageNotInstalledError
        importr("car")
        return True
    except Exception:
        return False


@pytest.mark.parametrize("n_runs", [1, 5])
def test_backend_parity_statsmodels_vs_pymer4_request(n_runs: int):
    """statsmodels and pymer4 backends should produce broadly compatible outputs."""
    pytest.importorskip("pymer4", reason="pymer4 not installed")
    if not _has_r_car():
        pytest.skip("R package 'car' not installed (needed for pymer4 factor tests)")

    rng_data = np.random.default_rng(17 + n_runs)
    data = _make_factorial_df(rng_data, n_inputs=40, n_runs=n_runs)
    run_col = "seed"

    bundle_sm = ps.analyze_factorial(
        data,
        factors=["chunker", "retrieval"],
        run_col=run_col,
        backend="statsmodels",
        n_sim=2500,
        rng=np.random.default_rng(4242),
    )

    bundle_py = ps.analyze_factorial(
        data,
        factors=["chunker", "retrieval"],
        run_col=run_col,
        backend="pymer4",
        n_sim=2500,
        rng=np.random.default_rng(4242),
    )

    # Backends use different estimation engines; require approximate rather than exact agreement
    _assert_factorial_bundles_compatible(bundle_sm, bundle_py, atol=0.05)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_missing_run_col_raises():
    rng = np.random.default_rng(12)
    data = _make_factorial_df(rng)
    with pytest.raises(ValueError, match="Columns not found"):
        ps.analyze_factorial(data, factors=["chunker", "retrieval"], run_col="nonexistent")


def test_missing_factor_col_raises():
    rng = np.random.default_rng(13)
    data = _make_factorial_df(rng)
    with pytest.raises(ValueError, match="Columns not found"):
        ps.analyze_factorial(data, factors=["chunker", "nonexistent"])


def test_r2_runs_treated_as_no_runs():
    """R=2 is below is_seeded threshold; LMM should use cell means (n_obs same as no-runs case)."""
    n_inputs = 20
    chunkers, retrievals = ("fixed", "semantic"), ("bm25", "dense")
    n_templates = len(chunkers) * len(retrievals)

    rng = np.random.default_rng(14)
    data = _make_factorial_df(rng, n_inputs=n_inputs, n_runs=2, chunkers=chunkers, retrievals=retrievals)
    bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"], run_col="seed")
    # R=2 → is_seeded=False → LMM uses cell means → n_obs = N_templates * M_inputs
    assert bundle.factorial_lmm_info.n_obs == n_templates * n_inputs
