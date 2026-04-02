import types

import numpy as np
import pandas as pd

from promptstats.core.paired import FriedmanResult, PairedDiffResult, PairwiseMatrix
from promptstats.compare import CompareReport, EntityStats
from promptstats.vis.critical_difference import plot_critical_difference


def _make_pairwise_for_overlap_bands() -> PairwiseMatrix:
    # A-B and B-C are non-significant by CI overlap with zero; A-C is significant.
    results = {
        ("A", "B"): PairedDiffResult(
            template_a="A",
            template_b="B",
            point_diff=0.02,
            std_diff=0.10,
            ci_low=-0.05,
            ci_high=0.08,
            p_value=0.4,
            test_method="bootstrap",
            n_inputs=50,
            per_input_diffs=np.array([0.02, 0.01, -0.01]),
            statistic="mean",
        ),
        ("A", "C"): PairedDiffResult(
            template_a="A",
            template_b="C",
            point_diff=0.20,
            std_diff=0.10,
            ci_low=0.10,
            ci_high=0.30,
            p_value=0.001,
            test_method="bootstrap",
            n_inputs=50,
            per_input_diffs=np.array([0.2, 0.15, 0.25]),
            statistic="mean",
        ),
        ("B", "C"): PairedDiffResult(
            template_a="B",
            template_b="C",
            point_diff=0.03,
            std_diff=0.09,
            ci_low=-0.04,
            ci_high=0.09,
            p_value=0.3,
            test_method="bootstrap",
            n_inputs=50,
            per_input_diffs=np.array([0.03, 0.02, -0.01]),
            statistic="mean",
        ),
    }
    return PairwiseMatrix(
        labels=["A", "B", "C"],
        results=results,
        correction_method="holm",
        simultaneous_ci=True,
        simultaneous_ci_method="max_t",
    )


def test_plot_critical_difference_accepts_pairwise_and_uses_rank_bands(monkeypatch):
    captured: dict[str, object] = {}

    def fake_cd_diagram(ranks, sig_mat, ax=None, alpha=0.05):
        captured["ranks"] = dict(ranks)
        captured["sig_mat"] = sig_mat.copy()
        captured["alpha"] = alpha

    fake_module = types.ModuleType("scikit_posthocs")
    fake_module.critical_difference_diagram = fake_cd_diagram
    monkeypatch.setitem(__import__("sys").modules, "scikit_posthocs", fake_module)

    pairwise = _make_pairwise_for_overlap_bands()
    fig = plot_critical_difference(
        pairwise,
        labels_sorted=["A", "B", "C"],
        ranks={"A": 1.1, "B": 1.8, "C": 2.7},
        alpha=0.05,
    )

    assert fig is not None
    assert captured["ranks"] == {"A": 1.1, "B": 1.8, "C": 2.7}

    sig_mat = captured["sig_mat"]
    assert isinstance(sig_mat, pd.DataFrame)
    assert float(sig_mat.loc["A", "A"]) == 1.0
    assert float(sig_mat.loc["A", "B"]) == 1.0
    assert float(sig_mat.loc["B", "C"]) == 1.0
    assert float(sig_mat.loc["A", "C"]) == 0.0

    assert "Pairwise rank bands" in fig.axes[0].get_title(loc="left")


def test_plot_critical_difference_friedman_mode_still_works(monkeypatch):
    captured: dict[str, object] = {}

    def fake_cd_diagram(ranks, sig_mat, ax=None, alpha=0.05):
        captured["ranks"] = dict(ranks)
        captured["sig_mat"] = sig_mat.copy()

    fake_module = types.ModuleType("scikit_posthocs")
    fake_module.critical_difference_diagram = fake_cd_diagram
    monkeypatch.setitem(__import__("sys").modules, "scikit_posthocs", fake_module)

    friedman = FriedmanResult(
        statistic=8.5,
        df=2,
        p_value=0.014,
        nemenyi_p={("A", "B"): 0.03, ("A", "C"): 0.002, ("B", "C"): 0.21},
        avg_ranks={"A": 1.2, "B": 2.0, "C": 2.8},
        n_inputs=40,
        n_templates=3,
    )

    fig = plot_critical_difference(friedman, alpha=0.05)

    assert fig is not None
    assert captured["ranks"] == friedman.avg_ranks

    sig_mat = captured["sig_mat"]
    assert isinstance(sig_mat, pd.DataFrame)
    assert float(sig_mat.loc["A", "B"]) == 0.03
    assert float(sig_mat.loc["A", "C"]) == 0.002
    assert float(sig_mat.loc["B", "C"]) == 0.21

    assert "Friedman" in fig.axes[0].get_title(loc="left")


def test_plot_critical_difference_accepts_compare_report_and_infers_ranks(monkeypatch):
    captured: dict[str, object] = {}

    def fake_cd_diagram(ranks, sig_mat, ax=None, alpha=0.05):
        captured["ranks"] = dict(ranks)
        captured["sig_mat"] = sig_mat.copy()

    fake_module = types.ModuleType("scikit_posthocs")
    fake_module.critical_difference_diagram = fake_cd_diagram
    monkeypatch.setitem(__import__("sys").modules, "scikit_posthocs", fake_module)

    pairwise = _make_pairwise_for_overlap_bands()
    report = CompareReport(
        labels=["A", "B", "C"],
        entity_stats={
            "A": EntityStats(mean=0.70, median=0.70, std=0.05, ci_low=0.65, ci_high=0.75),
            "B": EntityStats(mean=0.60, median=0.60, std=0.05, ci_low=0.55, ci_high=0.65),
            "C": EntityStats(mean=0.50, median=0.50, std=0.05, ci_low=0.45, ci_high=0.55),
        },
        unbeaten=["A"],
        pairwise=pairwise,
        full_analysis=None,
    )

    fig = plot_critical_difference(report)

    assert fig is not None
    assert captured["ranks"] == {"A": 1.0, "B": 2.0, "C": 3.0}
    sig_mat = captured["sig_mat"]
    assert float(sig_mat.loc["A", "B"]) == 1.0
    assert float(sig_mat.loc["A", "C"]) == 0.0
