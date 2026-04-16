import numpy as np
import pandas as pd
import pytest

import evalstats as es


def test_from_dataframe_returns_result_and_report_with_coercions():
    df = pd.DataFrame(
        {
            "prompt": ["A", "A", "B", "B", "A"],
            "input": ["i1", "i2", "i1", "i2", "i1"],
            "score": ["1.0", "bad", 0.7, 0.6, 0.9],
        }
    )

    result, report = es.from_dataframe(
        df,
        format="long",
        strict_complete_design=False,
        return_report=True,
    )

    assert isinstance(result, es.BenchmarkResult)
    assert report.format_detected == "long"
    assert report.score_non_numeric_coerced == 1
    assert report.duplicate_groups_collapsed == 1
    assert result.has_missing


def test_from_dataframe_can_allow_incomplete_design_when_not_strict():
    df = pd.DataFrame(
        {
            "prompt": ["A", "A", "B"],
            "input": ["i1", "i2", "i1"],
            "score": [0.9, 0.8, 0.7],
        }
    )

    result = es.from_dataframe(df, format="long", strict_complete_design=False)

    assert isinstance(result, es.BenchmarkResult)
    assert result.has_missing


def test_from_dataframe_multimodel_with_runs_and_evaluators():
    rows = []
    for model in ["m1", "m2"]:
        for prompt in ["A", "B"]:
            for inp in ["i1", "i2"]:
                for run in [0, 1, 2]:
                    for evaluator in ["acc", "fmt"]:
                        rows.append(
                            {
                                "model": model,
                                "prompt": prompt,
                                "input": inp,
                                "run": run,
                                "evaluator": evaluator,
                                "score": 0.5 + 0.1 * (prompt == "A"),
                            }
                        )
    df = pd.DataFrame(rows)

    result = es.from_dataframe(df, format="long")

    assert isinstance(result, es.MultiModelBenchmark)
    assert result.scores.shape == (2, 2, 2, 3, 2)


def test_from_dataframe_rejects_unknown_format():
    df = pd.DataFrame({"input": ["i1"], "A": [0.9], "B": [0.8]})

    with pytest.raises(ValueError, match="format must be one of"):
        es.from_dataframe(df, format="csv")


def test_from_dataframe_auto_detects_wide_and_long():
    df_wide = pd.DataFrame(
        {
            "input": ["i1", "i2"],
            "Prompt A": [0.9, 0.8],
            "Prompt B": [0.7, 0.6],
        }
    )
    _, report_wide = es.from_dataframe(df_wide, format="auto", return_report=True)
    assert report_wide.format_detected == "wide"

    df_long = pd.DataFrame(
        {
            "prompt": ["A", "A", "B", "B"],
            "input": ["i1", "i2", "i1", "i2"],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    )
    _, report_long = es.from_dataframe(df_long, format="auto", return_report=True)
    assert report_long.format_detected == "long"

    df_long_model_only = pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "input": ["i1", "i2", "i1", "i2"],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    )
    _, report_long_model_only = es.from_dataframe(
        df_long_model_only,
        format="auto",
        return_report=True,
    )
    assert report_long_model_only.format_detected == "long"


def test_from_dataframe_wide_strict_incomplete_raises():
    df = pd.DataFrame(
        {
            "input": ["i1", "i2"],
            "Prompt A": [0.9, "oops"],
            "Prompt B": [0.7, 0.6],
        }
    )

    with pytest.raises(ValueError, match="Incomplete design"):
        es.from_dataframe(df, format="wide", strict_complete_design=True)


def test_from_dataframe_wide_non_strict_tracks_coercion_note():
    df = pd.DataFrame(
        {
            "input": ["i1", "i2"],
            "Prompt A": [0.9, "oops"],
            "Prompt B": [0.7, 0.6],
        }
    )

    result, report = es.from_dataframe(
        df,
        format="wide",
        strict_complete_design=False,
        return_report=True,
    )

    assert isinstance(result, es.BenchmarkResult)
    assert result.has_missing
    assert report.score_non_numeric_coerced == 1
    assert any("non-numeric score value" in note for note in report.notes)


def test_from_dataframe_long_missing_required_columns_raises():
    df = pd.DataFrame(
        {
            "prompt": ["A", "B"],
            "input": ["i1", "i1"],
            "not_score": [0.9, 0.8],
        }
    )

    with pytest.raises(ValueError, match="requires prompt/template"):
        es.from_dataframe(df, format="long")


def test_from_dataframe_long_model_input_score_injects_implicit_template():
    df = pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "input": ["i1", "i2", "i1", "i2"],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    )

    result, report = es.from_dataframe(df, format="long", return_report=True)

    assert isinstance(result, es.MultiModelBenchmark)
    assert result.template_labels == ["default_prompt"]
    assert result.scores.shape == (2, 1, 2)
    assert any("injected implicit template label" in note for note in report.notes)


def test_from_dataframe_repair_true_fills_partial_run_slots_and_counts():
    df = pd.DataFrame(
        [
            {"prompt": "A", "input": "i1", "run": 0, "score": 1.0},
            {"prompt": "A", "input": "i1", "run": 1, "score": 0.8},
            {"prompt": "A", "input": "i2", "run": 0, "score": 0.9},
            {"prompt": "A", "input": "i2", "run": 1, "score": 0.7},
            {"prompt": "B", "input": "i1", "run": 0, "score": 0.6},
            {"prompt": "B", "input": "i1", "run": 1, "score": 0.4},
            {"prompt": "B", "input": "i2", "run": 0, "score": 0.3},
        ]
    )

    result, report = es.from_dataframe(
        df,
        format="long",
        repair=True,
        strict_complete_design=True,
        return_report=True,
    )

    assert isinstance(result, es.BenchmarkResult)
    assert report.run_nan_values_filled == 1
    b_idx = result.template_labels.index("B")
    i2_idx = result.input_labels.index("i2")
    # Missing run=1 is imputed from the available run value (0.3)
    assert result.scores[b_idx, i2_idx, 1] == pytest.approx(0.3)


def test_from_dataframe_repair_false_keeps_partial_run_missing_slots():
    df = pd.DataFrame(
        [
            {"prompt": "A", "input": "i1", "run": 0, "score": 1.0},
            {"prompt": "A", "input": "i1", "run": 1, "score": 0.8},
            {"prompt": "A", "input": "i2", "run": 0, "score": 0.9},
            {"prompt": "A", "input": "i2", "run": 1, "score": 0.7},
            {"prompt": "B", "input": "i1", "run": 0, "score": 0.6},
            {"prompt": "B", "input": "i1", "run": 1, "score": 0.4},
            {"prompt": "B", "input": "i2", "run": 0, "score": 0.3},
        ]
    )

    result, report = es.from_dataframe(
        df,
        format="long",
        repair=False,
        strict_complete_design=True,
        return_report=True,
    )

    assert isinstance(result, es.BenchmarkResult)
    assert report.run_nan_values_filled == 0
    b_idx = result.template_labels.index("B")
    i2_idx = result.input_labels.index("i2")
    assert np.isnan(result.scores[b_idx, i2_idx, 1])


def test_from_dataframe_alias_columns_are_supported_for_multimodel_runs_evals():
    rows = []
    for model in ["m1", "m2"]:
        for prompt in ["P1", "P2"]:
            for inp in ["x", "y"]:
                for repeat in [0, 1, 2]:
                    for metric_name in ["acc", "fmt"]:
                        rows.append(
                            {
                                "model_name": model,
                                "prompt_template": prompt,
                                "input_label": inp,
                                "repeat": repeat,
                                "metric_name": metric_name,
                                "value": 0.2 + 0.1 * (prompt == "P1") + 0.02 * repeat,
                            }
                        )
    df = pd.DataFrame(rows)

    result, report = es.from_dataframe(df, format="auto", return_report=True)

    assert isinstance(result, es.MultiModelBenchmark)
    assert result.scores.shape == (2, 2, 2, 3, 2)
    assert report.format_detected == "long"
    assert report.score_non_numeric_coerced == 0


def test_from_dataframe_duplicate_groups_are_collapsed_with_mean():
    df = pd.DataFrame(
        [
            {"prompt": "A", "input": "i1", "score": 0.2},
            {"prompt": "A", "input": "i1", "score": 0.8},
            {"prompt": "A", "input": "i2", "score": 0.9},
            {"prompt": "B", "input": "i1", "score": 0.4},
            {"prompt": "B", "input": "i2", "score": 0.5},
        ]
    )

    result, report = es.from_dataframe(df, format="long", return_report=True)

    assert report.duplicate_groups_collapsed == 1
    a_idx = result.template_labels.index("A")
    i1_idx = result.input_labels.index("i1")
    assert result.scores[a_idx, i1_idx] == pytest.approx(0.5)


def test_from_dataframe_multimodel_missing_in_one_model_mentions_model_in_error():
    df = pd.DataFrame(
        [
            {"model": "m1", "prompt": "A", "input": "i1", "score": 0.9},
            {"model": "m1", "prompt": "A", "input": "i2", "score": 0.8},
            {"model": "m1", "prompt": "B", "input": "i1", "score": 0.7},
            {"model": "m1", "prompt": "B", "input": "i2", "score": 0.6},
            {"model": "m2", "prompt": "A", "input": "i1", "score": 0.5},
            {"model": "m2", "prompt": "A", "input": "i2", "score": 0.4},
            {"model": "m2", "prompt": "B", "input": "i1", "score": 0.3},
        ]
    )

    with pytest.raises(ValueError, match="model 'm2'"):
        es.from_dataframe(df, format="long", strict_complete_design=True)


def test_data_load_report_to_lines_contains_key_fields():
    df = pd.DataFrame(
        {
            "prompt": ["A", "A", "B", "B"],
            "input": ["i1", "i2", "i1", "i2"],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    )

    _, report = es.from_dataframe(df, format="long", return_report=True)
    lines = report.to_lines()

    assert any("requested=long" in line for line in lines)
    assert any("score values coerced to NaN" in line for line in lines)
    assert any("duplicate groups collapsed by mean" in line for line in lines)
    assert any("missing run slots imputed from cell mean" in line for line in lines)
