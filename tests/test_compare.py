"""Tests for the new compare() / load_from() / EvalResults / ComparisonResult API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import evalstats as es
from evalstats.config import set_alpha_ci, get_alpha_ci
from evalstats.loader import EvalResults, EvalLoadError
from evalstats.api import ComparisonResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_alpha():
    original = get_alpha_ci()
    yield
    set_alpha_ci(original)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _binary_dict(n: int = 30, seed: int = 0) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    return {
        "A": rng.binomial(1, 0.7, n).astype(float).tolist(),
        "B": rng.binomial(1, 0.4, n).astype(float).tolist(),
    }


def _continuous_dict(n: int = 30, seed: int = 0) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    return {
        "X": rng.uniform(0.6, 0.9, n).tolist(),
        "Y": rng.uniform(0.3, 0.7, n).tolist(),
    }


# ---------------------------------------------------------------------------
# load_from
# ---------------------------------------------------------------------------

def test_load_from_dataframe_basic():
    df = pd.DataFrame({
        "model": ["A", "A", "B", "B"],
        "item": ["q1", "q2", "q1", "q2"],
        "score": [1.0, 0.0, 0.0, 1.0],
    })
    evaldata = es.load_from(df)
    assert isinstance(evaldata, EvalResults)
    assert evaldata.score_type == "binary"


def test_load_from_list_of_dicts():
    records = [
        {"model": "A", "item": "q1", "score": 1},
        {"model": "A", "item": "q2", "score": 0},
        {"model": "B", "item": "q1", "score": 0},
        {"model": "B", "item": "q2", "score": 1},
    ]
    evaldata = es.load_from(records)
    assert isinstance(evaldata, EvalResults)


def test_load_from_col_map():
    df = pd.DataFrame({
        "llm": ["A", "B"],
        "q_id": ["q1", "q1"],
        "result": [1.0, 0.0],
    })
    evaldata = es.load_from(df, col_map={"llm": "model", "q_id": "item", "result": "score"})
    assert isinstance(evaldata, EvalResults)


def test_load_from_raises_on_empty():
    with pytest.raises(EvalLoadError):
        es.load_from(pd.DataFrame())


# ---------------------------------------------------------------------------
# EvalResults.from_scores
# ---------------------------------------------------------------------------

def test_from_scores_flat_dict():
    scores = _binary_dict(20)
    evaldata = EvalResults.from_scores(scores, factors="model")
    assert isinstance(evaldata, EvalResults)
    assert evaldata.score_type == "binary"


def test_from_scores_2d_array_multirun():
    rng = np.random.default_rng(0)
    scores = {
        "A": rng.binomial(1, 0.7, (10, 3)).astype(float),  # 10 items × 3 runs
        "B": rng.binomial(1, 0.4, (10, 3)).astype(float),
    }
    evaldata = EvalResults.from_scores(scores, factors="model")
    assert isinstance(evaldata, EvalResults)


def test_from_scores_nested_dict():
    scores = {
        "gpt4": {"template_a": [1, 0, 1, 1, 0], "template_b": [0, 1, 1, 0, 1]},
        "claude": {"template_a": [1, 1, 1, 0, 0], "template_b": [0, 0, 1, 1, 0]},
    }
    evaldata = EvalResults.from_scores(scores)
    assert isinstance(evaldata, EvalResults)


# ---------------------------------------------------------------------------
# compare() — input coercion
# ---------------------------------------------------------------------------

def test_compare_accepts_dict_input():
    scores = _binary_dict(30)
    result = es.compare(scores, factors="model", rng=_rng())
    assert isinstance(result, ComparisonResult)


def test_compare_accepts_list_of_dicts():
    records = [
        {"model": "A", "item": str(i), "score": float(i % 2)}
        for i in range(20)
    ] + [
        {"model": "B", "item": str(i), "score": float((i + 1) % 2)}
        for i in range(20)
    ]
    result = es.compare(records, factors="model", rng=_rng())
    assert isinstance(result, ComparisonResult)


def test_compare_accepts_evalresults():
    df = pd.DataFrame({
        "model": ["A"] * 20 + ["B"] * 20,
        "item": [str(i % 20) for i in range(40)],
        "score": [1.0] * 12 + [0.0] * 8 + [0.0] * 14 + [1.0] * 6,
    })
    evaldata = es.load_from(df)
    result = es.compare(evaldata, factors="model", rng=_rng())
    assert isinstance(result, ComparisonResult)


def test_compare_raises_on_bad_input_type():
    with pytest.raises(TypeError, match="compare\\(\\) expects"):
        es.compare(42, factors="model")


# ---------------------------------------------------------------------------
# compare() — factor routing
# ---------------------------------------------------------------------------

def test_compare_model_factor_sets_entity_labels():
    scores = {"model_a": [1, 0, 1, 1, 0] * 6, "model_b": [0, 1, 0, 0, 1] * 6}
    result = es.compare(scores, factors="model", rng=_rng())
    assert set(result.labels) == {"model_a", "model_b"}


def test_compare_prompts_wrapper_returns_comparison_result():
    scores = _binary_dict(30)
    result = es.compare_prompts(scores, rng=_rng())
    assert isinstance(result, ComparisonResult)


def test_compare_models_wrapper_returns_comparison_result():
    scores = _binary_dict(30)
    result = es.compare_models(scores, rng=_rng())
    assert isinstance(result, ComparisonResult)


# ---------------------------------------------------------------------------
# ComparisonResult properties
# ---------------------------------------------------------------------------

def test_comparison_result_alpha_property():
    result = es.compare(_binary_dict(20), factors="model", alpha=0.10, rng=_rng())
    assert result.alpha == pytest.approx(0.10)


def test_comparison_result_alpha_from_global():
    set_alpha_ci(0.01)
    result = es.compare(_binary_dict(20), factors="model", rng=_rng())
    assert result.alpha == pytest.approx(0.01)


def test_comparison_result_labels():
    scores = {"X": [1, 0, 1] * 10, "Y": [0, 1, 0] * 10}
    result = es.compare(scores, factors="model", rng=_rng())
    assert set(result.labels) == {"X", "Y"}


def test_comparison_result_entity_stats_has_mean_and_ci():
    result = es.compare(_binary_dict(30), factors="model", rng=_rng())
    for lbl in result.labels:
        stats = result.entity_stats[lbl]
        assert hasattr(stats, "mean")
        assert hasattr(stats, "ci_low")
        assert hasattr(stats, "ci_high")
        assert 0.0 <= stats.ci_low <= stats.mean <= stats.ci_high <= 1.0


def test_comparison_result_pairwise_has_p_value():
    result = es.compare(_binary_dict(40), factors="model", rng=_rng())
    pair = result.pairwise.get("A", "B")
    assert pair is not None
    assert pair.p_value is not None
    assert 0.0 < pair.p_value <= 1.0


def test_comparison_result_unbeaten_returns_none_when_no_sig_diff():
    # Identical scores → no significant difference → unbeaten should be None
    scores = {"A": [1, 0, 1, 0, 1] * 4, "B": [1, 0, 1, 0, 1] * 4}
    result = es.compare(scores, factors="model", alpha=0.05, rng=_rng())
    assert result.unbeaten is None


def test_comparison_result_full_analysis_is_analysis_bundle():
    from evalstats.core.router import AnalysisBundle
    result = es.compare(_binary_dict(30), factors="model", rng=_rng())
    assert isinstance(result.full_analysis, AnalysisBundle)


# ---------------------------------------------------------------------------
# ComparisonResult.to_dict / to_frame
# ---------------------------------------------------------------------------

def test_to_dict_structure():
    result = es.compare(_binary_dict(20), factors="model", rng=_rng())
    d = result.to_dict()
    assert "entities" in d
    assert "pairwise" in d
    assert "alpha" in d
    assert isinstance(d["entities"], dict)
    assert isinstance(d["pairwise"], list)
    for lbl in result.labels:
        assert lbl in d["entities"]
        assert "mean" in d["entities"][lbl]
        assert "ci_low" in d["entities"][lbl]


def test_to_frame_has_entities_and_pairwise():
    result = es.compare(_binary_dict(20), factors="model", rng=_rng())
    frames = result.to_frame()
    assert "entities" in frames
    assert "pairwise" in frames
    assert "raw" in frames
    assert isinstance(frames["entities"], pd.DataFrame)
    assert "entity" in frames["entities"].columns
    assert "mean" in frames["entities"].columns


# ---------------------------------------------------------------------------
# summary / repr
# ---------------------------------------------------------------------------

def test_comparison_result_summary_runs(capsys):
    result = es.compare(_binary_dict(20), factors="model", rng=_rng())
    result.summary()
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_comparison_result_repr():
    result = es.compare(_binary_dict(20), factors="model", rng=_rng())
    r = repr(result)
    assert "ComparisonResult" in r
    assert "model" in r


# ---------------------------------------------------------------------------
# Alpha integration
# ---------------------------------------------------------------------------

def test_explicit_alpha_overrides_global():
    set_alpha_ci(0.01)
    result = es.compare(_binary_dict(20), factors="model", alpha=0.20, rng=_rng())
    assert result.alpha == pytest.approx(0.20)


def test_none_alpha_uses_global():
    set_alpha_ci(0.03)
    result = es.compare(_binary_dict(20), factors="model", rng=_rng())
    assert result.alpha == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# compare() — custom factor column names (non-"model" / non-"prompt")
# ---------------------------------------------------------------------------

def _chunker_scores(n: int = 40, seed: int = 0) -> dict:
    """Flat scores dict keyed by chunker strategy level."""
    rng = np.random.default_rng(seed)
    return {
        "fixed":    rng.binomial(1, 0.50, n).astype(float).tolist(),
        "sliding":  rng.binomial(1, 0.62, n).astype(float).tolist(),
        "semantic": rng.binomial(1, 0.74, n).astype(float).tolist(),
    }


def _retriever_scores(n: int = 40, seed: int = 0) -> dict:
    """Flat scores dict keyed by retriever strategy level."""
    rng = np.random.default_rng(seed)
    return {
        "bm25":  rng.binomial(1, 0.52, n).astype(float).tolist(),
        "dense": rng.binomial(1, 0.64, n).astype(float).tolist(),
        "hybrid":rng.binomial(1, 0.72, n).astype(float).tolist(),
    }


def _chunker_retriever_scores(n: int = 30, seed: int = 0) -> dict:
    """Nested scores dict: outer=chunker, inner=retriever."""
    rng = np.random.default_rng(seed)
    chunkers  = ["fixed", "sliding", "semantic"]
    retrievers = ["bm25", "dense", "hybrid"]
    effects_c = {"fixed": 0.00, "sliding": 0.05, "semantic": 0.10}
    effects_r = {"bm25": 0.00, "dense": 0.07, "hybrid": 0.12}
    return {
        c: {
            r: rng.binomial(1, min(0.55 + effects_c[c] + effects_r[r], 0.95), n)
                   .astype(float).tolist()
            for r in retrievers
        }
        for c in chunkers
    }


class TestCompareCustomFactors:
    """compare() with user-defined factor column names (not 'model' or 'prompt')."""

    def test_single_factor_chunker_labels(self):
        result = es.compare(_chunker_scores(), factors="chunker", rng=_rng())
        assert isinstance(result, ComparisonResult)
        assert set(result.labels) == {"fixed", "sliding", "semantic"}

    def test_single_factor_retriever_labels(self):
        result = es.compare(_retriever_scores(), factors="retriever", rng=_rng())
        assert isinstance(result, ComparisonResult)
        assert set(result.labels) == {"bm25", "dense", "hybrid"}

    def test_single_factor_pairwise_count(self):
        result = es.compare(_chunker_scores(), factors="chunker", rng=_rng())
        # C(3, 2) = 3 pairwise comparisons
        assert len(result.pairwise.results) == 3

    def test_single_factor_to_dict(self):
        result = es.compare(_retriever_scores(), factors="retriever", rng=_rng())
        d = result.to_dict()
        assert set(d["entities"].keys()) == {"bm25", "dense", "hybrid"}
        assert len(d["pairwise"]) == 3

    def test_single_factor_entity_stats_has_ci(self):
        result = es.compare(_chunker_scores(n=50), factors="chunker", rng=_rng())
        for lbl in result.labels:
            stats = result.entity_stats[lbl]
            assert 0.0 <= stats.ci_low <= stats.mean <= stats.ci_high <= 1.0

    def test_single_factor_summary_runs(self, capsys):
        result = es.compare(_chunker_scores(), factors="chunker", rng=_rng())
        result.summary()
        out = capsys.readouterr().out
        assert "fixed" in out
        assert "sliding" in out
        assert "semantic" in out

    def test_single_factor_user_factors_preserved(self):
        # The original factor name should be preserved in the result metadata.
        result = es.compare(_chunker_scores(), factors="chunker", rng=_rng())
        assert result._factors == "chunker"

    def test_factorial_two_custom_factors_returns_result(self):
        result = es.compare(
            _chunker_retriever_scores(), factors=["chunker", "retriever"], rng=_rng()
        )
        assert isinstance(result, ComparisonResult)
        assert result.full_analysis is not None

    def test_factorial_two_custom_factors_label_count(self):
        result = es.compare(
            _chunker_retriever_scores(), factors=["chunker", "retriever"], rng=_rng()
        )
        # 3 chunkers × 3 retrievers = 9 treatment cells
        assert len(result.labels) == 9

    def test_factorial_two_custom_factors_labels_contain_levels(self):
        result = es.compare(
            _chunker_retriever_scores(), factors=["chunker", "retriever"], rng=_rng()
        )
        label_text = " ".join(result.labels)
        for level in ["fixed", "sliding", "semantic", "bm25", "dense", "hybrid"]:
            assert level in label_text

    def test_factorial_summary_runs(self, capsys):
        result = es.compare(
            _chunker_retriever_scores(), factors=["chunker", "retriever"], rng=_rng()
        )
        result.summary()
        assert len(capsys.readouterr().out) > 0

    def test_factorial_user_factors_preserved(self):
        result = es.compare(
            _chunker_retriever_scores(), factors=["chunker", "retriever"], rng=_rng()
        )
        assert result._factors == ["chunker", "retriever"]
