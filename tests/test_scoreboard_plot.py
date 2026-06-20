import pytest
from types import SimpleNamespace
from matplotlib.container import ErrorbarContainer

from evalstats.vis.scoreboard import plot_accuracy_bar


def _mock_report(labels, stats_dict, unbeaten=None):
    """Create a duck-type result object for scoreboard testing."""
    return SimpleNamespace(
        labels=labels,
        entity_stats={lbl: SimpleNamespace(**kw) for lbl, kw in stats_dict.items()},
        entity_name_singular="prompt",
        unbeaten=unbeaten,
    )


def test_plot_accuracy_bar_adds_errorbars_from_explicit_cis():
    fig = plot_accuracy_bar(
        {"A": 0.60, "B": 0.75},
        cis={"A": (0.50, 0.70), "B": (0.65, 0.85)},
        as_percent=False,
    )

    assert any(isinstance(c, ErrorbarContainer) for c in fig.axes[0].containers)


def test_plot_accuracy_bar_infers_errorbars_from_comparison_result():
    report = _mock_report(
        ["A", "B"],
        {
            "A": dict(mean=0.60, median=0.60, std=0.05, ci_low=0.52, ci_high=0.68),
            "B": dict(mean=0.75, median=0.75, std=0.05, ci_low=0.69, ci_high=0.81),
        },
        unbeaten=["B"],
    )

    fig = plot_accuracy_bar(
        report,
        cis={"A": (0.52, 0.68), "B": (0.69, 0.81)},
        as_percent=False,
    )

    assert any(isinstance(c, ErrorbarContainer) for c in fig.axes[0].containers)


def test_plot_accuracy_bar_raises_for_missing_ci_label():
    with pytest.raises(ValueError, match="cis is missing labels"):
        plot_accuracy_bar(
            {"A": 0.60, "B": 0.75},
            cis={"A": (0.50, 0.70)},
            as_percent=False,
        )


def test_plot_accuracy_bar_without_cis_has_no_errorbars_for_comparison_result():
    report = _mock_report(
        ["A", "B"],
        {
            "A": dict(mean=0.60, median=0.60, std=0.05, ci_low=0.52, ci_high=0.68),
            "B": dict(mean=0.75, median=0.75, std=0.05, ci_low=0.69, ci_high=0.81),
        },
        unbeaten=["B"],
    )

    fig = plot_accuracy_bar(report, as_percent=False)

    assert not any(isinstance(c, ErrorbarContainer) for c in fig.axes[0].containers)
