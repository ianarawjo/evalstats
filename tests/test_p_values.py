"""Tests for the p_values / pairwise_test feature.

Covers:
  - _resolve_p_value_method() resolution logic (unit)
  - analyze() stores the correct p_value_method in bundles
  - compare_prompts() / compare_models() propagate p_value_method
  - CompareReport.summary() shows/suppresses p-values based on stored method
  - print_analysis_summary() shows p-values based on bundle p_value_method
  - CLI --p-values and --pairwise-test flags pass through correctly
  - Setting pairwise_test explicitly enables p-values without --p-values flag
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

import evalstats as es
from evalstats import cli
from evalstats.core.router import _resolve_p_value_method
from evalstats.core.bundles import AnalysisBundle
from evalstats.core.types import BenchmarkResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


import re as _re

# The p-value column header in the pairwise table is always immediately preceded
# by the "ES" (effect size) column.  We match on this two-column pattern rather
# than bare "p (" to avoid false positives on prose like
# "indistinguishable rank bands (p (boot))" in the executive summary.
_P_COL_RE = _re.compile(r"ES\s+p \(")


def _has_p_column(text: str) -> bool:
    """Return True when the pairwise table contains a p-value column."""
    return bool(_P_COL_RE.search(text))


def _scores_2prompt(n: int = 40, seed: int = 0) -> dict:
    """Two-prompt dict with a clear winner for fast bootstrap tests."""
    rng = np.random.default_rng(seed)
    return {
        "A": rng.normal(0.75, 0.10, n).clip(0, 1).tolist(),
        "B": rng.normal(0.60, 0.10, n).clip(0, 1).tolist(),
    }


def _scores_3prompt(n: int = 40, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "A": rng.normal(0.80, 0.10, n).clip(0, 1).tolist(),
        "B": rng.normal(0.65, 0.10, n).clip(0, 1).tolist(),
        "C": rng.normal(0.50, 0.10, n).clip(0, 1).tolist(),
    }


def _make_benchmark(scores_dict: dict) -> BenchmarkResult:
    labels = list(scores_dict.keys())
    arrays = [np.array(v) for v in scores_dict.values()]
    m = arrays[0].shape[0]
    return BenchmarkResult(
        scores=np.stack(arrays, axis=0),
        template_labels=labels,
        input_labels=[f"i{i}" for i in range(m)],
    )


# ---------------------------------------------------------------------------
# Unit tests: _resolve_p_value_method
# ---------------------------------------------------------------------------

class TestResolvePValueMethod:
    def test_default_suppresses(self):
        assert _resolve_p_value_method(False, "auto", False) is None

    def test_p_values_true_no_omnibus_returns_auto(self):
        assert _resolve_p_value_method(True, "auto", False) == "auto"

    def test_p_values_true_omnibus_returns_wsr(self):
        # Wilcoxon is the standard Friedman post-hoc.
        assert _resolve_p_value_method(True, "auto", True) == "wsr"

    def test_explicit_bootstrap_enables_without_flag(self):
        assert _resolve_p_value_method(False, "bootstrap", False) == "boot"

    def test_explicit_wilcoxon_enables_without_flag(self):
        assert _resolve_p_value_method(False, "wilcoxon", False) == "wsr"

    def test_explicit_nemenyi_enables_without_flag(self):
        assert _resolve_p_value_method(False, "nemenyi", False) == "nem"

    def test_explicit_wilcoxon_overrides_bootstrap_path(self):
        # Explicitly choosing wilcoxon should work even when using bootstrap CIs.
        assert _resolve_p_value_method(True, "wilcoxon", False) == "wsr"

    def test_explicit_bootstrap_overrides_omnibus(self):
        # Explicit pairwise_test wins over omnibus-driven auto selection.
        assert _resolve_p_value_method(True, "bootstrap", True) == "boot"

    @pytest.mark.parametrize("test,expected", [
        ("bootstrap", "boot"),
        ("wilcoxon", "wsr"),
        ("nemenyi", "nem"),
    ])
    def test_explicit_tests_map_correctly(self, test, expected):
        assert _resolve_p_value_method(True, test, False) == expected


# ---------------------------------------------------------------------------
# analyze() stores p_value_method on the bundle
# ---------------------------------------------------------------------------

class TestAnalyzeStoresPValueMethod:
    def _bundle(self, **kw) -> AnalysisBundle:
        bench = _make_benchmark(_scores_2prompt())
        result = es.analyze(bench, n_bootstrap=200, rng=_rng(), **kw)
        assert isinstance(result, AnalysisBundle)
        return result

    def test_default_is_none(self):
        bundle = self._bundle()
        assert bundle.p_value_method is None

    def test_p_values_true_stores_auto(self):
        bundle = self._bundle(p_values=True)
        assert bundle.p_value_method == "auto"

    def test_p_values_true_omnibus_stores_wsr(self):
        bundle = self._bundle(p_values=True, omnibus=True)
        assert bundle.p_value_method == "wsr"

    @pytest.mark.parametrize("test,expected", [
        ("bootstrap", "boot"),
        ("wilcoxon", "wsr"),
        ("nemenyi", "nem"),
    ])
    def test_explicit_pairwise_test_stored(self, test, expected):
        bundle = self._bundle(pairwise_test=test)
        assert bundle.p_value_method == expected

    def test_explicit_pairwise_test_enables_without_p_values_flag(self):
        # Setting pairwise_test alone (without p_values=True) should enable p-values.
        bundle = self._bundle(pairwise_test="wilcoxon")
        assert bundle.p_value_method == "wsr"

    def test_multimodel_stores_p_value_method(self):
        bench = _make_benchmark(_scores_2prompt())
        from evalstats.core.types import MultiModelBenchmark
        scores = np.stack([bench.scores, bench.scores * 0.9], axis=0)
        mmb = MultiModelBenchmark(
            scores=scores,
            model_labels=["M1", "M2"],
            template_labels=bench.template_labels,
            input_labels=bench.input_labels,
        )
        result = es.analyze(mmb, n_bootstrap=200, rng=_rng(), p_values=True)
        from evalstats.core.bundles import MultiModelBundle
        assert isinstance(result, MultiModelBundle)
        assert result.model_level.p_value_method == "auto"
        assert result.template_level.p_value_method == "auto"


# ---------------------------------------------------------------------------
# compare_prompts() propagates p_value_method to CompareReport
# ---------------------------------------------------------------------------

class TestComparePropagatesPValueMethod:
    def test_default_report_p_value_method_is_none(self):
        report = es.compare_prompts(_scores_2prompt(), n_bootstrap=200, rng=_rng())
        assert report.p_value_method is None

    def test_p_values_true_report_stores_auto(self):
        report = es.compare_prompts(
            _scores_2prompt(), n_bootstrap=200, rng=_rng(), p_values=True
        )
        assert report.p_value_method == "auto"

    @pytest.mark.parametrize("test,expected", [
        ("bootstrap", "boot"),
        ("wilcoxon", "wsr"),
        ("nemenyi", "nem"),
    ])
    def test_explicit_pairwise_test_in_report(self, test, expected):
        report = es.compare_prompts(
            _scores_2prompt(), n_bootstrap=200, rng=_rng(), pairwise_test=test
        )
        assert report.p_value_method == expected

    def test_compare_models_propagates(self):
        scores = {
            "M1": _scores_2prompt(n=40, seed=0)["A"],
            "M2": _scores_2prompt(n=40, seed=1)["B"],
        }
        report = es.compare_models(scores, n_bootstrap=200, rng=_rng(), p_values=True)
        assert report.p_value_method == "auto"


# ---------------------------------------------------------------------------
# print_analysis_summary() output: p-values shown/hidden
# ---------------------------------------------------------------------------

class TestPrintAnalysisSummaryOutput:
    """Check that the p-value column appears or is absent in printed output."""

    def _run_analyze(self, **kw) -> str:
        import io
        from contextlib import redirect_stdout
        bench = _make_benchmark(_scores_3prompt())
        bundle = es.analyze(bench, n_bootstrap=300, rng=_rng(), **kw)
        buf = io.StringIO()
        with redirect_stdout(buf):
            es.print_analysis_summary(bundle)
        return buf.getvalue()

    def test_default_no_p_values_in_output(self):
        out = self._run_analyze()
        assert not _has_p_column(out)

    def test_p_values_true_shows_p_column(self):
        out = self._run_analyze(p_values=True)
        assert _has_p_column(out)

    def test_pairwise_test_wilcoxon_shows_wsr_column(self):
        out = self._run_analyze(pairwise_test="wilcoxon")
        assert "p (wsr)" in out

    def test_pairwise_test_bootstrap_shows_boot_column(self):
        out = self._run_analyze(pairwise_test="bootstrap")
        assert "p (boot)" in out

    def test_pairwise_test_nemenyi_shows_nem_column(self):
        out = self._run_analyze(pairwise_test="nemenyi")
        assert "p (nem)" in out

    def test_p_values_true_omnibus_shows_wsr(self):
        # With omnibus=True, auto resolves to wsr.
        out = self._run_analyze(p_values=True, omnibus=True)
        assert "p (wsr)" in out


# ---------------------------------------------------------------------------
# CompareReport.summary() output: p-values shown/hidden
# ---------------------------------------------------------------------------

class TestCompareReportSummaryOutput:
    """Check that CompareReport.summary() respects the stored p_value_method."""

    def _summary(self, report, **kw) -> str:
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            report.summary(**kw)
        return buf.getvalue()

    def test_default_no_p_values(self):
        report = es.compare_prompts(_scores_2prompt(), n_bootstrap=200, rng=_rng())
        out = self._summary(report)
        assert not _has_p_column(out)

    def test_p_values_true_shows_column(self):
        report = es.compare_prompts(
            _scores_2prompt(), n_bootstrap=200, rng=_rng(), p_values=True
        )
        out = self._summary(report)
        assert _has_p_column(out)

    def test_pairwise_test_wilcoxon_shows_wsr(self):
        report = es.compare_prompts(
            _scores_2prompt(), n_bootstrap=200, rng=_rng(), pairwise_test="wilcoxon"
        )
        out = self._summary(report)
        assert "p (wsr)" in out

    def test_explicit_override_suppresses(self):
        # Even if the report has p_value_method='auto', passing None should suppress.
        report = es.compare_prompts(
            _scores_2prompt(), n_bootstrap=200, rng=_rng(), p_values=True
        )
        assert report.p_value_method is not None
        out = self._summary(report, p_value_method=None)
        assert not _has_p_column(out)

    def test_explicit_override_enables(self):
        # Even if the report has p_value_method=None (default), passing 'wsr' should show.
        report = es.compare_prompts(_scores_2prompt(), n_bootstrap=200, rng=_rng())
        assert report.p_value_method is None
        out = self._summary(report, p_value_method="wsr")
        assert "p (wsr)" in out


# ---------------------------------------------------------------------------
# CLI: parser and forwarding
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_parser_p_values_default_false(self):
        parser = cli._build_parser()
        args = parser.parse_args(["analyze", "data.csv"])
        assert args.p_values is False

    def test_parser_pairwise_test_default_auto(self):
        parser = cli._build_parser()
        args = parser.parse_args(["analyze", "data.csv"])
        assert args.pairwise_test == "auto"

    def test_parser_accepts_p_values_flag(self):
        parser = cli._build_parser()
        args = parser.parse_args(["analyze", "data.csv", "--p-values"])
        assert args.p_values is True

    @pytest.mark.parametrize("test", ["auto", "bootstrap", "wilcoxon", "nemenyi"])
    def test_parser_accepts_pairwise_test(self, test):
        parser = cli._build_parser()
        args = parser.parse_args(["analyze", "data.csv", "--pairwise-test", test])
        assert args.pairwise_test == test

    def test_parser_rejects_invalid_pairwise_test(self):
        parser = cli._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["analyze", "data.csv", "--pairwise-test", "invalid"])


class TestCLIForwarding:
    """Check that CLI flags are forwarded to analyze() correctly."""

    def _run_cmd(self, tmp_path, monkeypatch, extra_args: argparse.Namespace) -> dict:
        """Run _cmd_analyze with fake analyze() and return captured kwargs."""
        source_df = __import__("pandas").DataFrame(
            {"prompt": ["A", "A", "B", "B"], "input": ["i1", "i2", "i1", "i2"], "score": [0.9, 0.8, 0.7, 0.6]}
        )
        file_path = tmp_path / "data.csv"
        source_df.to_csv(file_path, index=False)

        captured = {}

        def fake_analyze(benchmark, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

        monkeypatch.setattr("evalstats.io.from_dataframe", lambda df, **kw: (
            _make_benchmark(_scores_2prompt()),
            type("R", (), {"format_detected": "long"})(),
        ))
        monkeypatch.setattr("evalstats.core.router.analyze", fake_analyze)
        monkeypatch.setattr("evalstats.core.summary.print_analysis_summary", lambda *a, **k: None)

        args = argparse.Namespace(
            file=file_path,
            format="long",
            sheet="0",
            evaluator_mode="aggregate",
            ci=None,
            method="auto",
            backend="statsmodels",
            n_bootstrap=100,
            correction="fdr_bh",
            spread_percentiles=(10.0, 90.0),
            reference="grand_mean",
            failure_threshold=None,
            statistic="mean",
            template_model_collapse="as_runs",
            simultaneous_ci=True,
            omnibus=False,
            top_pairwise=5,
            out=None,
            **vars(extra_args),
        )
        cli._cmd_analyze(args)
        return captured

    def test_default_forwards_false_and_auto(self, tmp_path, monkeypatch):
        captured = self._run_cmd(
            tmp_path, monkeypatch,
            argparse.Namespace(p_values=False, pairwise_test="auto"),
        )
        assert captured["p_values"] is False
        assert captured["pairwise_test"] == "auto"

    def test_p_values_flag_forwarded(self, tmp_path, monkeypatch):
        captured = self._run_cmd(
            tmp_path, monkeypatch,
            argparse.Namespace(p_values=True, pairwise_test="auto"),
        )
        assert captured["p_values"] is True
        assert captured["pairwise_test"] == "auto"

    def test_pairwise_test_forwarded(self, tmp_path, monkeypatch):
        captured = self._run_cmd(
            tmp_path, monkeypatch,
            argparse.Namespace(p_values=False, pairwise_test="wilcoxon"),
        )
        assert captured["pairwise_test"] == "wilcoxon"

    @pytest.mark.parametrize("test", ["bootstrap", "wilcoxon", "nemenyi"])
    def test_all_pairwise_tests_forwarded(self, tmp_path, monkeypatch, test):
        captured = self._run_cmd(
            tmp_path, monkeypatch,
            argparse.Namespace(p_values=True, pairwise_test=test),
        )
        assert captured["pairwise_test"] == test


class TestCLIOutputShowsPValues:
    """End-to-end: run _cmd_analyze on real data and check printed output."""

    def _make_csv(self, tmp_path: Path) -> Path:
        import pandas as pd
        rng = np.random.default_rng(7)
        rows = []
        for prompt, mean in [("A", 0.80), ("B", 0.60), ("C", 0.50)]:
            for i in range(30):
                rows.append({"prompt": prompt, "input": f"i{i}", "score": float(np.clip(rng.normal(mean, 0.10), 0, 1))})
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def _run(self, tmp_path, capsys, extra: dict) -> str:
        csv_path = self._make_csv(tmp_path)
        base = dict(
            file=csv_path,
            format="long",
            sheet="0",
            evaluator_mode="aggregate",
            ci=None,
            method="smooth_bootstrap",
            backend="statsmodels",
            n_bootstrap=300,
            correction="fdr_bh",
            spread_percentiles=(10.0, 90.0),
            reference="grand_mean",
            failure_threshold=None,
            statistic="mean",
            template_model_collapse="as_runs",
            simultaneous_ci=False,
            omnibus=False,
            top_pairwise=5,
            out=None,
            p_values=False,
            pairwise_test="auto",
            brief=False,
        )
        base.update(extra)
        cli._cmd_analyze(argparse.Namespace(**base))
        return capsys.readouterr().out

    def test_default_no_p_values_in_output(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, {})
        assert not _has_p_column(out)

    def test_p_values_flag_adds_column(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, {"p_values": True})
        assert _has_p_column(out)

    def test_pairwise_test_wilcoxon_shows_wsr(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, {"pairwise_test": "wilcoxon"})
        assert "p (wsr)" in out

    def test_pairwise_test_bootstrap_shows_boot(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, {"pairwise_test": "bootstrap"})
        assert "p (boot)" in out

    def test_pairwise_test_nemenyi_shows_nem(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, {"pairwise_test": "nemenyi"})
        assert "p (nem)" in out

    def test_p_values_false_pairwise_test_auto_no_column(self, tmp_path, capsys):
        # Explicit defaults: both false/auto → no p-value column.
        out = self._run(tmp_path, capsys, {"p_values": False, "pairwise_test": "auto"})
        assert not _has_p_column(out)
