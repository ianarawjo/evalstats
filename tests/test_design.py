"""Tests for evalstats.core.design.detect_paired -- the paired-vs-unpaired
design-detection heuristic shared by the labeling CLI and compare()'s
design="auto" routing.
"""
import pandas as pd
import pytest

from evalstats.core.design import detect_paired
from evalstats.labeling import _detect_paired as detect_paired_via_labeling


def test_detect_paired_reexport_identity():
    # labeling.py must share the exact same implementation, not a copy.
    assert detect_paired_via_labeling is detect_paired


def test_paired_when_items_fully_shared():
    df = pd.concat([
        pd.DataFrame({"model": m, "item": range(50)}) for m in ["A", "B", "C"]
    ], ignore_index=True)
    assert detect_paired(df, "model", "item") is True


def test_unpaired_when_items_fully_disjoint():
    rows = []
    for m in ["A", "B", "C"]:
        for i in range(50):
            rows.append({"model": m, "item": f"{m}_{i}"})
    df = pd.DataFrame(rows)
    assert detect_paired(df, "model", "item") is False


def test_paired_tolerates_a_few_missing_rows():
    df = pd.concat([
        pd.DataFrame({"model": m, "item": range(50)}) for m in ["A", "B", "C"]
    ], ignore_index=True)
    # Drop 3 of B's 50 rows (94% overlap remains) -- should still read as paired.
    df = df.drop(df[(df["model"] == "B")].index[:3]).reset_index(drop=True)
    assert detect_paired(df, "model", "item") is True


def test_unpaired_below_90pct_overlap_threshold():
    df = pd.concat([
        pd.DataFrame({"model": m, "item": range(50)}) for m in ["A", "B", "C"]
    ], ignore_index=True)
    # Drop 10 of B's 50 rows (80% overlap) -- should flip to unpaired.
    df = df.drop(df[(df["model"] == "B")].index[:10]).reset_index(drop=True)
    assert detect_paired(df, "model", "item") is False


def test_no_factor_column_is_paired_trivially():
    df = pd.DataFrame({"item": range(50), "score": range(50)})
    assert detect_paired(df, None, "item") is True
    assert detect_paired(df, "nonexistent_col", "item") is True


def test_single_factor_level_is_paired_trivially():
    df = pd.DataFrame({"model": ["A"] * 50, "item": range(50)})
    assert detect_paired(df, "model", "item") is True
