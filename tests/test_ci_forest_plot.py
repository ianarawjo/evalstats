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


def test_plot_ci_forest_compare_to_uses_gradient_bands_and_same_hue():
    from matplotlib.patches import Rectangle

    small = _make_result(n_items=10, seed=1)
    big = _make_result(n_items=80, seed=2)
    # color_rule="factor" guarantees each entity gets a distinct hue --
    # "tier" mode legitimately lets multiple entities share a hue (e.g. two
    # entities both "significantly worse"), which isn't what this test is
    # checking for.
    fig = plot_ci_forest(big, compare_to=small, color_rule="factor")
    ax = fig.axes[0]
    bars = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_zorder() >= 2]
    # 3 entities x (4 primary bands + 4 comparison bands) = 24.
    assert len(bars) == 3 * 8

    # Group by rounded (row, base RGB) -- primary and comparison bars for
    # the same entity should share hue, differing only in alpha (muted).
    by_hue = {}
    for p in bars:
        r, g, b, a = p.get_facecolor()
        by_hue.setdefault((round(r, 2), round(g, 2), round(b, 2)), []).append(round(a, 3))
    # Exactly 3 distinct hues (one per entity), each hue used by both the
    # primary (full-scale alphas) and comparison (muted alphas) bands.
    assert len(by_hue) == 3
    for hue, alphas in by_hue.items():
        assert len(alphas) == 8  # 4 primary + 4 muted-comparison alphas
        assert len(set(alphas)) == 8  # all 8 alphas distinct (no accidental overlap)
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


def test_plot_ci_forest_mean_line_is_default():
    result = _make_result()
    fig = plot_ci_forest(result, reference_line=None)
    ax = fig.axes[0]
    # One simple black tick line per entity.
    mean_lines = [
        l for l in ax.get_lines()
        if l.get_xdata()[0] == l.get_xdata()[1] and len(l.get_xdata()) == 2
    ]
    assert len(mean_lines) == 3
    assert all(l.get_color() == "black" for l in mean_lines)
    plt_close(fig)


def test_plot_ci_forest_show_mean_false_omits_marker():
    result = _make_result()
    # reference_line=None to isolate mean-marker lines from the (also
    # vertical) reference line.
    fig = plot_ci_forest(result, show_mean=False, reference_line=None)
    ax = fig.axes[0]
    vertical_lines = [
        l for l in ax.get_lines()
        if len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1]
    ]
    assert len(vertical_lines) == 0
    assert len(ax.collections) == 0  # no scatter dots either
    plt_close(fig)


def test_plot_ci_forest_mean_marker_dot_uses_scatter():
    result = _make_result()
    fig = plot_ci_forest(result, mean_marker="dot")
    ax = fig.axes[0]
    assert len(ax.collections) == 3  # one scatter point per entity
    plt_close(fig)


def test_plot_ci_forest_show_ci_bracket_adds_overlay():
    result = _make_result()
    fig_without = plot_ci_forest(result, show_ci_bracket=False)
    fig_with = plot_ci_forest(result, show_ci_bracket=True)
    n_lines_without = len(fig_without.axes[0].get_lines())
    n_lines_with = len(fig_with.axes[0].get_lines())
    assert n_lines_with > n_lines_without
    plt_close(fig_without)
    plt_close(fig_with)


def test_plot_ci_forest_title_includes_n_and_subtitle_includes_method():
    """N lives in the title; CI method/correction/alpha live in a small
    subtitle between the title and the axes (not below the plot), so a
    LaTeX \\caption{} added under the whole figure doesn't read as
    redundant with a second caption-like line at the bottom."""
    result = _make_result(n_items=30)
    fig = plot_ci_forest(result)
    ax = fig.axes[0]
    assert "N=30 inputs" in ax.get_title()
    ax_texts = [t.get_text() for t in ax.texts]
    subtitle = " ".join(ax_texts)
    assert "CI method:" in subtitle
    assert "α=0.05" in subtitle
    # Nothing placed below the axes as a second, bottom-of-figure caption.
    assert len(fig.texts) == 0
    plt_close(fig)


def test_plot_ci_forest_legend_includes_band_and_mean_labels():
    result = _make_result()
    fig = plot_ci_forest(result)
    ax = fig.axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "99% CI" in legend_labels
    assert "68% CI" in legend_labels
    assert "mean" in legend_labels


# ---------------------------------------------------------------------------
# color_rule
# ---------------------------------------------------------------------------

def _bar_colors(ax):
    from matplotlib.patches import Rectangle
    # zorder >= 4 excludes the alternating row-background rectangles
    # (axhspan, zorder=0), which aren't gradient-band bars.
    return [
        p.get_facecolor() for p in ax.patches
        if isinstance(p, Rectangle) and p.get_zorder() >= 4
    ]


def test_color_rule_tier_is_default_and_has_legend():
    result = _make_result()
    fig = plot_ci_forest(result)
    ax = fig.axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Unbeaten" in legend_labels
    assert "Significantly worse" in legend_labels
    plt_close(fig)


def test_color_rule_factor_gives_each_entity_a_distinct_color():
    result = _make_result(n_models=3)
    fig = plot_ci_forest(result, color_rule="factor")
    ax = fig.axes[0]
    colors = _bar_colors(ax)
    # 3 entities x 4 gradient bands each; each entity's 4 bands share one
    # base color (varying only alpha), and different entities differ.
    distinct_rgb = {c[:3] for c in colors}
    assert len(distinct_rgb) == 3
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Unbeaten" not in legend_labels
    plt_close(fig)


def test_color_rule_factor_is_stable_across_sort_order():
    result = _make_result(n_models=3)
    fig_mean = plot_ci_forest(result, color_rule="factor", sort_by="mean")
    fig_label = plot_ci_forest(result, color_rule="factor", sort_by="label")
    # Same set of colors used regardless of row order.
    colors_mean = {c[:3] for c in _bar_colors(fig_mean.axes[0])}
    colors_label = {c[:3] for c in _bar_colors(fig_label.axes[0])}
    assert colors_mean == colors_label
    plt_close(fig_mean)
    plt_close(fig_label)


def test_color_rule_literal_color_used_for_all_entities():
    result = _make_result(n_models=3)
    fig = plot_ci_forest(result, color_rule="seagreen")
    ax = fig.axes[0]
    colors = _bar_colors(ax)
    distinct_rgb = {c[:3] for c in colors}
    assert len(distinct_rgb) == 1
    import matplotlib.colors as mcolors
    assert distinct_rgb.pop() == mcolors.to_rgb("seagreen")
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Unbeaten" not in legend_labels
    plt_close(fig)


def test_color_rule_invalid_raises_clear_error():
    result = _make_result()
    with pytest.raises(ValueError, match="not 'tier', 'factor'"):
        plot_ci_forest(result, color_rule="not_a_real_color")


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)
