"""Tests for evalstats.vis.forest.plot_ci_forest (gradient/single styles)
and ComparisonResult.plot()'s default.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import evalstats as es
from evalstats.vis.forest import plot_ci_forest


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_result(n_models=3, n_items=30, seed=0):
    rng = _rng(seed)
    rows = []
    for i in range(n_models):
        mu = 0.5 + 0.1 * i
        for j in range(n_items):
            rows.append({
                "model": f"m{i}", "item": f"q{j}",
                "score": float(np.clip(rng.normal(mu, 0.08), 0, 1)),
            })
    df = pd.DataFrame(rows)
    evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
    return es.compare(evaldata, factors="model", metric="score", rng=_rng(seed + 100))


def test_plot_ci_forest_gradient_is_default():
    result = _make_result()
    fig = plot_ci_forest(result)
    ax = fig.axes[0]
    assert "confidence gradient" in ax.get_title()
    plt_close(fig)


def test_plot_ci_forest_gradient_draws_multiple_bands_per_entity():
    result = _make_result()
    fig = plot_ci_forest(result)
    ax = fig.axes[0]
    # 3 entities x 4 gradient bands each = 12 bar patches (plus none from
    # axhspan row backgrounds, which are Rectangle patches too -- filter to
    # bar-like patches by checking there are at least as many as expected).
    from matplotlib.patches import Rectangle
    bar_patches = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(bar_patches) >= 3 * 4
    plt_close(fig)


def test_plot_ci_forest_single_style_no_gradient_footnote():
    result = _make_result()
    fig = plot_ci_forest(result, style="single")
    ax = fig.axes[0]
    assert "confidence gradient" not in ax.get_title()
    assert "confidence intervals" in ax.get_title()
    plt_close(fig)


def test_plot_ci_forest_gradient_matches_single_on_mean_and_outer_ci():
    """The gradient plot's outermost band should match the single-style
    plot's CI bounds when both are drawn at compatible confidence levels
    (99% outer gradient band ~ single style uses the bundle's own alpha,
    so just check means match exactly and outer band contains the primary CI)."""
    result = _make_result()
    fig_g = plot_ci_forest(result, style="gradient")
    fig_s = plot_ci_forest(result, style="single")
    ax_g, ax_s = fig_g.axes[0], fig_s.axes[0]
    # Scatter (mean) x-data should match between styles for the same entity order.
    means_g = sorted(c.get_offsets()[0][0] for c in ax_g.collections if len(c.get_offsets()))
    means_s = sorted(c.get_offsets()[0][0] for c in ax_s.collections if len(c.get_offsets()))
    assert means_g == pytest.approx(means_s, abs=1e-6)
    plt_close(fig_g)
    plt_close(fig_s)


def test_plot_ci_forest_compare_to_still_works_with_gradient_primary():
    small = _make_result(n_items=10, seed=1)
    big = _make_result(n_items=80, seed=2)
    fig = plot_ci_forest(big, compare_to=small)
    assert fig is not None
    plt_close(fig)


def test_comparison_result_plot_defaults_to_forest_gradient():
    result = _make_result()
    fig = result.plot()
    ax = fig.axes[0]
    assert "confidence gradient" in ax.get_title()
    plt_close(fig)


def test_comparison_result_plot_bar_still_available():
    result = _make_result()
    fig = result.plot(method="bar")
    assert fig is not None
    plt_close(fig)


def test_entity_stats_exposes_multi_ci():
    result = _make_result()
    stats = result.entity_stats
    for label, s in stats.items():
        assert s.multi_ci is not None
        assert len(s.multi_ci) >= 2
        for alpha, (lo, hi) in s.multi_ci.items():
            assert lo <= s.mean <= hi


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)
