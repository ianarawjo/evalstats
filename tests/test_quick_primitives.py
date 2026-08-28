"""Tests for evalstats.quick (mean_ci, summarize, stability, judge_debias_mean_ci)
and the array-based judge_alignment() path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import evalstats as es
from evalstats.quick import MeanCI, GroupSummary, StabilityResult, DebiasedMeanCI


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# mean_ci
# ---------------------------------------------------------------------------

def test_mean_ci_basic_continuous():
    rng = _rng(1)
    scores = np.clip(rng.normal(0.75, 0.1, 60), 0, 1)
    result = es.mean_ci(scores)
    assert isinstance(result, MeanCI)
    assert result.ci_low < result.mean < result.ci_high
    assert result.n == 60
    assert abs(result.mean - float(np.mean(scores))) < 1e-9


def test_mean_ci_unpacks_positionally():
    rng = _rng(2)
    scores = np.clip(rng.normal(0.6, 0.1, 40), 0, 1)
    mean, lo, hi, n, method = es.mean_ci(scores)
    result = es.mean_ci(scores)
    assert mean == result.mean
    assert lo == result.ci_low
    assert hi == result.ci_high
    assert n == result.n
    assert method == result.method


def test_mean_ci_to_dict():
    rng = _rng(3)
    scores = np.clip(rng.normal(0.5, 0.1, 30), 0, 1)
    d = es.mean_ci(scores).to_dict()
    assert set(d.keys()) == {"mean", "ci_low", "ci_high", "n", "method"}


def test_mean_ci_binary_uses_wilson():
    rng = _rng(4)
    scores = (rng.random(100) < 0.7).astype(float)
    result = es.mean_ci(scores)
    assert result.method == "wilson"
    assert 0 <= result.ci_low <= result.mean <= result.ci_high <= 1


def test_mean_ci_rejects_2d_array():
    with pytest.raises(ValueError, match="1-D"):
        es.mean_ci(np.zeros((3, 3)))


def test_mean_ci_rejects_empty_array():
    with pytest.raises(ValueError, match="empty"):
        es.mean_ci(np.array([]))


def test_mean_ci_matches_compare_for_same_data():
    """mean_ci() should compute the identical number compare() would show
    for the same entity's marginal CI -- same underlying calibration path.
    (compare() needs >= 2 entities to run at all, so a second dummy model
    -- same [0, 1] range, so auto-detection resolves identically -- is
    added purely to satisfy that; only the "target" row is checked.)"""
    rng = _rng(5)
    scores = np.clip(rng.normal(0.7, 0.08, 50), 0, 1)
    dummy_scores = np.clip(rng.normal(0.4, 0.08, 50), 0, 1)
    result = es.mean_ci(scores, rng=_rng(99))

    rows = [
        {"model": "target", "item": f"q{i}", "score": s} for i, s in enumerate(scores)
    ] + [
        {"model": "dummy", "item": f"q{i}", "score": s} for i, s in enumerate(dummy_scores)
    ]
    df = pd.DataFrame(rows)
    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    cmp = es.compare(evaldata, factors="model", metric="score", rng=_rng(99))
    rob = cmp._analysis.robustness
    idx = list(rob.labels).index("target")
    assert abs(result.mean - float(rob.mean[idx])) < 1e-9
    assert result.method == cmp._analysis.resolved_ci_method


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_single_array_returns_flat_dict():
    rng = _rng(10)
    scores = np.clip(rng.normal(0.8, 0.1, 30), 0, 1)
    result = es.summarize(scores)
    assert isinstance(result, GroupSummary)
    assert result.labels == ["value"]
    d = result.to_dict()
    assert "mean" in d  # flat, not nested under "value"
    assert set(d.keys()) >= {"mean", "median", "std", "ci_low", "ci_high", "n", "method"}


def test_summarize_dict_of_arrays():
    rng = _rng(11)
    scores = {
        "a": np.clip(rng.normal(0.8, 0.1, 30), 0, 1),
        "b": np.clip(rng.normal(0.6, 0.1, 25), 0, 1),  # different N -- fine
    }
    result = es.summarize(scores)
    assert result.labels == ["a", "b"]
    d = result.to_dict()
    assert set(d.keys()) == {"a", "b"}
    assert d["a"]["n"] == 30
    assert d["b"]["n"] == 25
    frame = result.to_frame()
    assert list(frame.index) == ["a", "b"]
    assert "mean" in frame.columns and "ci_low" in frame.columns


def test_summarize_dataframe_factor_metric():
    rng = _rng(12)
    rows = []
    for m, mu in [("x", 0.7), ("y", 0.5)]:
        for i in range(20):
            rows.append({"model": m, "score": float(np.clip(rng.normal(mu, 0.1), 0, 1))})
    df = pd.DataFrame(rows)
    result = es.summarize(df, factor="model", metric="score")
    assert set(result.labels) == {"x", "y"}
    frame = result.to_frame()
    assert frame.loc["x", "mean"] > frame.loc["y", "mean"]


def test_summarize_dataframe_requires_factor_and_metric():
    df = pd.DataFrame({"model": ["a", "b"], "score": [0.5, 0.6]})
    with pytest.raises(ValueError, match="factor"):
        es.summarize(df)


def test_summarize_dataframe_bad_column_name():
    df = pd.DataFrame({"model": ["a", "b"], "score": [0.5, 0.6]})
    with pytest.raises(ValueError, match="not found"):
        es.summarize(df, factor="nope", metric="score")


def test_summarize_empty_dict_raises():
    with pytest.raises(ValueError, match="empty"):
        es.summarize({})


def test_summarize_empty_group_raises():
    with pytest.raises(ValueError, match="no scores"):
        es.summarize({"a": np.array([1.0, 2.0]), "b": np.array([])})


def test_summarize_matches_mean_ci_for_same_single_array():
    rng = _rng(13)
    scores = np.clip(rng.normal(0.65, 0.09, 45), 0, 1)
    m = es.mean_ci(scores, rng=_rng(7))
    s = es.summarize(scores, rng=_rng(7))
    assert abs(m.mean - s.mean[0]) < 1e-9
    assert abs(m.ci_low - s.ci_low[0]) < 1e-9
    assert abs(m.ci_high - s.ci_high[0]) < 1e-9


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------

def test_stability_single_config():
    rng = _rng(20)
    M, K = 80, 5
    base = rng.normal(0.7, 0.1, M)
    runs = np.array([np.clip(base + rng.normal(0, 0.02, M), 0, 1) for _ in range(K)])
    result = es.stability(runs)
    assert isinstance(result, StabilityResult)
    assert result.labels == ["value"]
    assert result.n_runs[0] == K
    assert result.instability[0] < 0.05  # tight noise -> should read as stable
    d = result.to_dict()
    assert "instability" in d and "icc" in d and "interpretation" in d


def test_stability_dict_ragged_run_counts():
    rng = _rng(21)
    M = 60
    base = rng.normal(0.6, 0.1, M)
    stable_runs = np.array([np.clip(base + rng.normal(0, 0.01, M), 0, 1) for _ in range(6)])
    noisy_runs = np.array([np.clip(rng.normal(0.5, 0.2, M), 0, 1) for _ in range(3)])
    result = es.stability({"stable": stable_runs, "noisy": noisy_runs})
    assert list(result.n_runs) == [6, 3]
    assert result.instability[0] < result.instability[1]  # stable really is more stable
    frame = result.to_frame()
    assert frame.loc["stable", "n_runs"] == 6
    assert frame.loc["noisy", "n_runs"] == 3


def test_stability_summary_prints_noise_strip(capsys):
    rng = _rng(21)
    M = 60
    base = rng.normal(0.6, 0.1, M)
    stable_runs = np.array([np.clip(base + rng.normal(0, 0.01, M), 0, 1) for _ in range(6)])
    noisy_runs = np.array([np.clip(rng.normal(0.5, 0.2, M), 0, 1) for _ in range(3)])
    result = es.stability({"stable": stable_runs, "noisy": noisy_runs})

    result.summary()
    out = capsys.readouterr().out
    assert "Per-input Variance Across Runs" in out
    assert "Per-input noise" in out
    assert "stable" in out and "noisy" in out


def test_stability_requires_at_least_3_runs():
    rng = _rng(22)
    runs = rng.normal(0.5, 0.1, (2, 30))
    with pytest.raises(ValueError, match=">= 3 runs"):
        es.stability(runs)


def test_stability_requires_matching_item_count():
    rng = _rng(23)
    a = rng.normal(0.5, 0.1, (4, 30))
    b = rng.normal(0.5, 0.1, (4, 25))
    with pytest.raises(ValueError, match="same number of items"):
        es.stability({"a": a, "b": b})


def test_stability_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        es.stability(np.zeros(10))


# ---------------------------------------------------------------------------
# judge_debias_mean_ci -- (judge_scores, human_scores) sparse form:
# same length, human_scores is NaN outside the labeled subset.
# ---------------------------------------------------------------------------

def _sparse_debias_pair(rng, n_total, n_labeled, *, true_mean=0.55, bias=0.2):
    human_all = np.clip(rng.normal(true_mean, 0.15, n_total), 0, 1)
    judge_all = np.clip(human_all + bias + rng.normal(0, 0.05, n_total), 0, 1)
    idx = rng.choice(n_total, n_labeled, replace=False)
    human_sparse = np.full(n_total, np.nan)
    human_sparse[idx] = human_all[idx]
    return judge_all, human_sparse


def test_judge_debias_mean_ci_recovers_true_mean_better_than_raw_judge():
    rng = _rng(30)
    n_total, n_labeled, true_mean = 400, 40, 0.55
    judge, human = _sparse_debias_pair(rng, n_total, n_labeled, true_mean=true_mean)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = es.judge_debias_mean_ci(judge, human, rng=_rng(31))
    assert isinstance(result, DebiasedMeanCI)
    # Corrected mean must be closer to the true mean than the raw judge mean.
    assert abs(result.mean - true_mean) < abs(result.judge_mean - true_mean)
    assert result.ci_low < result.mean < result.ci_high
    assert result.n_labeled == n_labeled
    assert result.n_unlabeled == n_total - n_labeled
    assert result.p_value is None  # compute_pvalue defaults to False


def test_judge_debias_mean_ci_to_dict():
    rng = _rng(32)
    judge, human = _sparse_debias_pair(rng, n_total=150, n_labeled=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = es.judge_debias_mean_ci(judge, human, rng=_rng(1)).to_dict()
    assert set(d.keys()) == {
        "mean", "ci_low", "ci_high", "judge_mean", "human_mean",
        "rectifier", "p_value", "n_labeled", "n_unlabeled",
    }


def test_judge_debias_mean_ci_rejects_mismatched_shapes():
    rng = _rng(33)
    with pytest.raises(ValueError, match="same length"):
        es.judge_debias_mean_ci(rng.normal(0.5, 0.1, 60), rng.normal(0.5, 0.1, 59))


def test_judge_debias_mean_ci_rejects_below_15_labeled():
    rng = _rng(34)
    judge, human = _sparse_debias_pair(rng, n_total=100, n_labeled=10)
    with pytest.raises(ValueError, match="at least 15 human-labeled"):
        es.judge_debias_mean_ci(judge, human)


def test_judge_debias_mean_ci_rejects_below_50_total():
    rng = _rng(35)
    judge, human = _sparse_debias_pair(rng, n_total=40, n_labeled=20)
    with pytest.raises(ValueError, match="at least 50 items total"):
        es.judge_debias_mean_ci(judge, human)


def test_judge_debias_mean_ci_rejects_all_items_labeled():
    rng = _rng(36)
    n = 60
    human_all = np.clip(rng.normal(0.5, 0.1, n), 0, 1)
    judge_all = np.clip(human_all + rng.normal(0, 0.05, n), 0, 1)
    with pytest.raises(ValueError, match="no unlabeled portion"):
        es.judge_debias_mean_ci(judge_all, human_all)  # no NaN at all


def test_judge_debias_mean_ci_warns_below_30_labeled_and_100_total():
    rng = _rng(37)
    judge, human = _sparse_debias_pair(rng, n_total=80, n_labeled=20)
    with pytest.warns(UserWarning, match="only 20 human-labeled"):
        es.judge_debias_mean_ci(judge, human)
    with pytest.warns(UserWarning, match="only 80 total items"):
        es.judge_debias_mean_ci(judge, human)


def test_judge_debias_mean_ci_always_warns_about_random_sampling():
    """Even with plenty of labels, the random-sampling assumption reminder
    should always fire -- it's a modeling assumption, not a sample-size
    issue, so it shouldn't be silenced just because n is large."""
    rng = _rng(38)
    judge, human = _sparse_debias_pair(rng, n_total=300, n_labeled=60)
    with pytest.warns(UserWarning, match="unbiased sample.*uniformly at random"):
        es.judge_debias_mean_ci(judge, human)


def test_judge_debias_mean_ci_compute_pvalue_opt_in():
    rng = _rng(39)
    judge, human = _sparse_debias_pair(rng, n_total=150, n_labeled=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = es.judge_debias_mean_ci(judge, human, compute_pvalue=True)
    assert result.p_value is not None


# ---------------------------------------------------------------------------
# judge_alignment -- array-based path (judge_scores, human_scores),
# human_scores sparse (NaN for unlabeled) or fully dense (no NaN at all).
# ---------------------------------------------------------------------------

def _sparse_pair(rng, n_total, n_labeled):
    """Build a (judge_scores, human_scores) pair in the sparse convention:
    same length, human_scores is NaN outside the labeled subset."""
    judge = np.clip(rng.integers(1, 6, n_total).astype(float) + rng.normal(0, 0.5, n_total), 1, 5)
    human = np.full(n_total, np.nan)
    idx = rng.choice(n_total, n_labeled, replace=False)
    human[idx] = rng.integers(1, 6, n_labeled).astype(float)
    return judge, human


def test_judge_alignment_array_form_sparse_basic():
    rng = _rng(40)
    judge, human = _sparse_pair(rng, n_total=300, n_labeled=40)
    result = es.judge_alignment(judge, human)
    assert result.n_labeled == 40
    assert result.n_total == 300  # derived automatically from judge_scores
    # Some items are unlabeled -> representativeness check runs for free.
    assert "score_distribution" in result.representativeness


def test_judge_alignment_array_form_dense_no_nan():
    """When human_scores has no NaN at all (ambiguous: could be '100%
    labeled' or 'caller already extracted just the labeled subset'), stay
    conservative and skip the representativeness check rather than
    silently comparing a set against itself."""
    rng = _rng(41)
    n = 40
    human = rng.integers(1, 6, n).astype(float)
    judge = np.clip(human + rng.normal(0, 0.5, n), 1, 5)
    result = es.judge_alignment(judge, human)
    assert result.n_labeled == n
    assert result.n_total == n
    assert result.representativeness == {}


def test_judge_alignment_array_form_explicit_all_judge_scores_overrides():
    rng = _rng(42)
    n_lab, n_total = 35, 250
    human = rng.integers(1, 6, n_lab).astype(float)
    judge = np.clip(human + rng.normal(0, 0.5, n_lab), 1, 5)
    all_judge = np.clip(rng.integers(1, 6, n_total).astype(float) + rng.normal(0, 0.3, n_total), 1, 5)
    result = es.judge_alignment(judge, human, all_judge_scores=all_judge)
    assert result.n_total == n_total
    assert "score_distribution" in result.representativeness
    # No slice-column checks in the array form (no DataFrame).
    assert not any(k.startswith("slice_") for k in result.representativeness)


def test_judge_alignment_array_form_display_names():
    rng = _rng(43)
    human = rng.integers(1, 6, 35).astype(float)
    judge = np.clip(human + rng.normal(0, 0.5, 35), 1, 5)
    result = es.judge_alignment(judge, human, llm_metric="my_judge", human_groundtruth="my_human")
    assert result.llm_metric == "my_judge"
    assert result.human_col == "my_human"


def test_judge_alignment_array_form_defaults_display_names():
    rng = _rng(44)
    human = rng.integers(1, 6, 35).astype(float)
    judge = np.clip(human + rng.normal(0, 0.5, 35), 1, 5)
    result = es.judge_alignment(judge, human)
    assert result.llm_metric == "judge"
    assert result.human_col == "human"


def test_judge_alignment_array_form_requires_human_scores():
    with pytest.raises(TypeError, match="requires both arrays"):
        es.judge_alignment(np.array([1.0, 2.0, 3.0]))


def test_judge_alignment_array_form_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same length"):
        es.judge_alignment(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_judge_alignment_array_form_rejects_all_nan_human_scores():
    with pytest.raises(ValueError, match="No labeled items"):
        es.judge_alignment(np.array([1.0, 2.0, 3.0]), np.array([np.nan, np.nan, np.nan]))


def test_judge_alignment_array_form_warns_below_30():
    rng = _rng(45)
    judge, human = _sparse_pair(rng, n_total=100, n_labeled=10)
    with pytest.warns(UserWarning, match="fewer than ~30"):
        es.judge_alignment(judge, human)


def test_judge_alignment_evaldata_form_still_requires_kwargs():
    """The dispatcher must still enforce llm_metric/human_groundtruth for
    the EvalResults form -- this is a straight rename + new sibling form,
    not a behavior change to the original signature's requiredness."""
    rows = [
        {"model": "a", "item": f"q{i}", "score": 0.5, "human": (0.5 if i < 10 else None)}
        for i in range(40)
    ]
    df = pd.DataFrame(rows)
    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    with pytest.raises(TypeError, match="requires llm_metric"):
        es.judge_alignment(evaldata)


def test_judge_alignment_evaldata_form_rejects_second_positional():
    rows = [
        {"model": "a", "item": f"q{i}", "score": 0.5, "human": (0.5 if i < 10 else None)}
        for i in range(40)
    ]
    df = pd.DataFrame(rows)
    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    with pytest.raises(TypeError, match="doesn't take a second positional"):
        es.judge_alignment(evaldata, np.array([1.0, 2.0]))
