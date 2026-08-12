"""Per-entity CI forest plot for CompareReport objects.

Draws one horizontal CI bar per entity (prompt or model), coloured by
statistical tier, with the best-performing entity at the top.  An optional
second report can be overlaid for direct before/after comparison (e.g.,
to show how CI widths change when you double the eval set or add more runs).

Two styles (mirroring the same split ``print_analysis_summary``'s
``style=`` uses for the terminal's ASCII plots):

* ``"gradient"`` (default) -- nested CI bands at 68/90/95/99% (the same
  ``multi_ci`` data the terminal's ``░▒▓█`` gradient rendering uses),
  drawn as increasingly-opaque bars toward the mean, so the reader sees
  the confidence *gradient* rather than a single somewhat-arbitrary cutoff.
* ``"single"`` -- one CI band per entity, at whatever confidence level the
  report was computed with. Always used as the fallback when ``multi_ci``
  data isn't available (e.g. an LMM/Wald-type report with only one CI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..config import GRADIENT_CI_ALPHAS

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_PALETTE = {
    "unbeaten":      "#4a90d9",  # medium blue  — in-contention CIs
    "lower_tier":    "#e07b7b",  # muted red    — lower-tier CIs
    "no_sig":        "#8a9bb5",  # gray-blue    — no significant differences
    "ref_line":      "#cccccc",  # light gray   — reference line
    "grid":          "#EEF1F4",  # very light   — x grid
    "row_alt":       "#FAFBFC",  # off-white    — alternating rows
    "text":          "#2D333B",  # dark slate   — axis labels
    "text_secondary":"#6B7280",  # muted gray   — secondary text
}

# Per-band opacity for the gradient style, outermost (widest CI, 99%) to
# innermost (narrowest, 68%) -- same ordering convention as the terminal's
# _gradient_interval_line (sorted ascending by alpha = descending by CI
# width), just alpha-blended bars instead of block-character replacement.
_GRADIENT_BAND_ALPHAS = (0.22, 0.38, 0.58, 0.85)
_GRADIENT_BAND_HEIGHT = 0.5


def _lighten(color, amount: float) -> tuple:
    """Blend *color* toward white by *amount* (0 = unchanged, 1 = white).

    A genuine lighter tint of the same hue -- distinct from just lowering
    alpha, which fades toward whatever sits underneath (the page/slide
    background, not necessarily white) and reads more like a rendering
    artifact than an intentional "this is the secondary series" design.
    """
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def plot_ci_forest(
    report,
    compare_to=None,
    report_label: Optional[str] = None,
    compare_label: Optional[str] = None,
    reference_line: Optional[float] = 0.5,
    sort_by: str = "mean",
    as_percent: bool = True,
    style: Literal["gradient", "single"] = "gradient",
    color_rule: str = "tier",
    show_mean: bool = True,
    mean_marker: Literal["line", "dot"] = "line",
    show_ci_bracket: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> "Figure":
    """Plot per-entity confidence intervals as a horizontal forest plot.

    Parameters
    ----------
    report : CompareReport
        Primary report — the CIs and tier colouring are drawn from this.
        Returned by :func:`evalstats.compare_prompts` or
        :func:`evalstats.compare_models`.
    compare_to : CompareReport, optional
        A second report to overlay for comparison (e.g. a smaller or
        single-run eval).  Drawn offset above each row, using the *same*
        colour as that row (muted, via lower alpha) rather than a fixed
        unrelated colour -- so the two bands read as "same entity, two
        evals". Renders as gradient bands too when *style* is
        ``"gradient"`` and *compare_to* has ``multi_ci`` data, for the same
        consistency reason. Both reports must contain the same entity
        labels.
    report_label : str, optional
        Legend label for the primary report when *compare_to* is supplied.
        Defaults to ``"primary"``.
    compare_label : str, optional
        Legend label for *compare_to*.  Defaults to ``"comparison"``.
    reference_line : float, optional
        Draw a vertical dashed reference line at this value.  Set to
        ``None`` to suppress.  Defaults to ``0.5`` (50% accuracy).
    sort_by : str
        Row ordering:

        * ``"mean"`` (default) — descending by mean; best entity at top.
        * ``"label"`` — alphabetical.
        * ``"input_order"`` — preserves ``report.labels`` order.
    as_percent : bool
        When ``True`` (default), multiply CI values by 100 and format the
        x-axis as percentages.  Set to ``False`` for raw (0–1) scores.
    style : {"gradient", "single"}
        ``"gradient"`` (default) draws nested CI bands at 68/90/95/99%,
        increasingly opaque toward the mean -- the same ``multi_ci`` data
        the terminal's ``░▒▓█`` gradient plot uses, just rendered as
        matplotlib bars. Falls back to ``"single"`` automatically per
        entity when that entity has no ``multi_ci`` data (e.g. a Wald-type
        CI with only one level computed). ``"single"`` always draws one CI
        band at the report's own confidence level.
    color_rule : str
        How bars are coloured:

        * ``"tier"`` (default) -- by significance tier: "Unbeaten" vs.
          "Significantly worse" (or a single neutral colour when nothing
          is significantly different). This is the only mode with a
          colour-meaning legend, since it's the only one where colour
          carries information beyond "which entity is this" (the y-axis
          labels already say that).
        * ``"factor"`` -- each entity gets its own distinct colour from a
          qualitative palette (cycling past 10 entities). Useful when
          entities are a categorical factor in their own right (e.g.
          different models) and you want colour to track identity rather
          than significance.
        * any matplotlib colour spec (e.g. ``"#4a90d9"``, ``"steelblue"``)
          -- every bar uses that one colour.
    show_mean : bool
        Draw a marker at the point estimate (default ``True``). Set
        ``False`` to let the CI band(s) speak for themselves.
    mean_marker : {"line", "dot"}
        ``"line"`` (default) draws a short vertical tick crossing the band
        at the mean -- reads clearly against any band colour or opacity.
        ``"dot"`` draws the previous circle marker instead.
    show_ci_bracket : bool
        When ``True``, overlay a traditional bracket-style CI at the
        report's own (single) confidence level on top of the gradient
        bands -- for readers who want the familiar landmark in addition to
        the richer gradient. Default ``False``. Ignored when *style* is
        already ``"single"`` (there'd be nothing to add on top of).
    figsize : tuple[float, float], optional
        Figure size.  Defaults to ``(7.5, 0.45 * N + 1.8)``.
    title : str, optional
        Plot title.  A descriptive default is generated when omitted.
    ax : Axes, optional
        Existing axes to draw into.  A new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = report.labels
    n = len(labels)

    # ---- sort order -------------------------------------------------------
    means = np.array([report.entity_stats[l].mean for l in labels])
    if sort_by == "mean":
        order = list(np.argsort(-means))
    elif sort_by == "label":
        order = sorted(range(n), key=lambda i: labels[i])
    elif sort_by == "input_order":
        order = list(range(n))
    else:
        raise ValueError(
            f"Unknown sort_by: {sort_by!r}. "
            "Expected 'mean', 'label', or 'input_order'."
        )

    ordered_labels = [labels[i] for i in order]
    unbeaten = set(report.unbeaten) if report.unbeaten else set()

    # ---- colour rule --------------------------------------------------------
    if color_rule not in ("tier", "factor") and not mcolors.is_color_like(color_rule):
        raise ValueError(
            f"color_rule={color_rule!r} is not 'tier', 'factor', or a "
            "valid matplotlib colour spec (e.g. '#4a90d9', 'steelblue')."
        )
    factor_colors: dict = {}
    if color_rule == "factor":
        palette = plt.get_cmap("tab10").colors
        # Keyed by original label order (not sort-dependent ordered_labels)
        # so an entity's colour stays stable across different sort_by calls.
        factor_colors = {lbl: palette[i % len(palette)] for i, lbl in enumerate(labels)}

    def _entity_color(label: str) -> str:
        if color_rule == "tier":
            if not unbeaten:
                return _PALETTE["no_sig"]
            return _PALETTE["unbeaten"] if label in unbeaten else _PALETTE["lower_tier"]
        if color_rule == "factor":
            return factor_colors[label]
        return color_rule  # a literal colour spec, same for every entity

    scale = 100.0 if as_percent else 1.0

    def _ci(rep, label: str) -> tuple[float, float, float]:
        s = rep.entity_stats[label]
        return s.mean * scale, s.ci_low * scale, s.ci_high * scale

    def _multi_ci(rep, label: str) -> Optional[dict[float, tuple[float, float]]]:
        s = rep.entity_stats[label]
        raw = getattr(s, "multi_ci", None)
        if raw is None or len(raw) < 2:
            return None
        return {a: (lo * scale, hi * scale) for a, (lo, hi) in raw.items()}

    # ---- validate compare_to ----------------------------------------------
    if compare_to is not None:
        missing = set(labels) - set(compare_to.labels)
        if missing:
            raise ValueError(
                f"compare_to is missing labels present in report: {sorted(missing)}"
            )

    # ---- figure setup -----------------------------------------------------
    own_fig = ax is None
    if own_fig:
        if figsize is None:
            figsize = (7.5, max(3.0, 0.45 * n + 1.8))
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    else:
        fig = ax.get_figure()

    ax.set_facecolor("white")

    y_positions = np.arange(n)
    # More vertical room per row when stacking two bands (primary +
    # comparison) so thick gradient bars don't heavily overlap; a plain
    # single line needs much less. The comparison band is noticeably
    # thinner than the primary one -- a secondary, not a co-equal, series.
    has_gradient_rows = style == "gradient"
    offset = (0.28 if has_gradient_rows else 0.18) if compare_to is not None else 0.0
    primary_band_height = _GRADIENT_BAND_HEIGHT * (0.9 if compare_to is not None else 1.0)
    compare_band_height = _GRADIENT_BAND_HEIGHT * 0.45
    # Comparison bands/lines use the SAME hue as their row, lightened
    # (a real tint toward white, not just lower alpha -- see _lighten) so
    # the two read as "same entity, two evals" rather than an unrelated
    # series, while still visibly standing apart from the primary band.
    _LIGHTEN_AMOUNT = 0.55

    # ---- alternating row backgrounds --------------------------------------
    for i in range(n):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color=_PALETTE["row_alt"], zorder=0)

    # ---- reference line ---------------------------------------------------
    if reference_line is not None:
        ax.axvline(
            reference_line * scale,
            color=_PALETTE["ref_line"],
            lw=1.0,
            ls="--",
            zorder=1,
        )

    # ---- draw CIs ---------------------------------------------------------
    lw = 2.8
    ms = 55  # scatter marker size
    any_gradient_used = False

    def _draw_ci_row(
        y_row: float, lo: float, hi: float, mean_val: float,
        multi_ci_row: Optional[dict], row_color, band_alphas: tuple,
        band_height: float, zorder_base: int, draw_mean: bool,
        mean_tick_color: str, line_width: float,
    ) -> tuple[bool, int]:
        """Draw one CI row (gradient bands, falling back to a single line
        when multi_ci_row is None) plus an optional mean marker. Returns
        (used_gradient, next_free_zorder)."""
        used_gradient = False
        if multi_ci_row is not None:
            used_gradient = True
            # Widest CI (99%, smallest alpha) drawn first/lowest zorder,
            # narrowest (68%, largest alpha) drawn last/highest zorder --
            # same "inner band wins" convention as the terminal's
            # _gradient_interval_line, via z-order layering instead of
            # character replacement.
            sorted_alphas = sorted(multi_ci_row.keys())
            for band_i, a in enumerate(sorted_alphas):
                lo_a, hi_a = multi_ci_row[a]
                band_alpha = band_alphas[min(band_i, len(band_alphas) - 1)]
                ax.barh(
                    y_row, width=hi_a - lo_a, left=lo_a,
                    height=band_height,
                    color=row_color, alpha=band_alpha,
                    edgecolor="none", zorder=zorder_base + band_i,
                )
            next_z = zorder_base + len(band_alphas)
        else:
            ax.plot(
                [lo, hi], [y_row, y_row],
                color=row_color, lw=line_width,
                solid_capstyle="round", zorder=zorder_base,
            )
            next_z = zorder_base + 1
        if draw_mean:
            if mean_marker == "line":
                tick_h = band_height * 0.7
                ax.plot(
                    [mean_val, mean_val], [y_row - tick_h, y_row + tick_h],
                    color=mean_tick_color, lw=1.5, zorder=next_z + 1,
                )
            else:
                ax.scatter(
                    [mean_val], [y_row],
                    color=row_color, s=ms, zorder=next_z + 1,
                    edgecolor="white", linewidth=0.6,
                )
            next_z += 1
        return used_gradient, next_z

    for i, label in enumerate(ordered_labels):
        y = float(y_positions[i])
        mean5, lo5, hi5 = _ci(report, label)

        color = _entity_color(label)

        if compare_to is not None:
            # Comparison report -- same hue as the primary row, lightened
            # (a real tint, not just alpha) and thinner, so the two bands
            # read as "same entity, two evals" while still standing apart.
            mean0, lo0, hi0 = _ci(compare_to, label)
            multi_ci_cmp = _multi_ci(compare_to, label) if has_gradient_rows else None
            light_color = _lighten(color, _LIGHTEN_AMOUNT)
            used_grad_cmp, _ = _draw_ci_row(
                y + offset, lo0, hi0, mean0, multi_ci_cmp, light_color,
                _GRADIENT_BAND_ALPHAS, compare_band_height, 2, show_mean,
                _PALETTE["text_secondary"], lw * 0.6,
            )
            any_gradient_used = any_gradient_used or used_grad_cmp

        # Primary CI — full colour, offset below when compare_to given.
        # Gradient style falls back to single-band per-entity when this
        # entity has no multi_ci data (e.g. a Wald-type CI).
        multi_ci = _multi_ci(report, label) if style == "gradient" else None
        y_row = y - offset
        used_grad, top_zorder = _draw_ci_row(
            y_row, lo5, hi5, mean5, multi_ci, color,
            _GRADIENT_BAND_ALPHAS, primary_band_height, 4, show_mean,
            "black", lw,
        )
        any_gradient_used = any_gradient_used or used_grad

        # Optional traditional bracket-style CI overlaid on top of the
        # gradient bands, at the report's own single confidence level --
        # for readers who want that familiar landmark alongside the gradient.
        if show_ci_bracket and multi_ci is not None:
            ax.plot(
                [lo5, hi5], [y_row, y_row],
                color=_PALETTE["text"], lw=1.3, zorder=top_zorder,
                solid_capstyle="butt",
            )
            cap_h = primary_band_height * 0.22
            for x_cap in (lo5, hi5):
                ax.plot(
                    [x_cap, x_cap], [y_row - cap_h, y_row + cap_h],
                    color=_PALETTE["text"], lw=1.3, zorder=top_zorder,
                )

    # ---- axes styling -----------------------------------------------------
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_labels, fontsize=9, color=_PALETTE["text"])
    ax.invert_yaxis()  # best at top

    if as_percent:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel(
        f"{'Accuracy (%)' if as_percent else 'Score'}",
        fontsize=10,
        color=_PALETTE["text"],
        labelpad=8,
    )

    ax.xaxis.grid(True, color=_PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")

    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", colors=_PALETTE["text_secondary"], labelsize=9)

    # ---- gather methods metadata (for title + caption) --------------------
    bundle = getattr(report, "full_analysis", None)
    n_inputs = getattr(getattr(bundle, "benchmark", None), "n_inputs", None)
    alpha = getattr(report, "alpha", 0.05)
    ci_pct = int(round((1 - alpha) * 100))
    ci_method = getattr(bundle, "resolved_ci_method", None)
    correction = getattr(getattr(bundle, "pairwise", None), "correction_method", None)

    def _pretty(s: Optional[str]) -> Optional[str]:
        return s.replace("_", " ") if s else None

    # ---- title + methods subtitle ------------------------------------------
    if title is None:
        n_str = f"  |  N={n_inputs} inputs" if n_inputs else ""
        if any_gradient_used:
            ci_label = "68-99% confidence gradient"
        else:
            ci_label = f"{ci_pct}% confidence intervals"
        title = f"{ci_label} per {report.entity_name_singular}{n_str}"

    # Self-contained methods subtitle -- this figure is meant to stand on
    # its own once copied out of evalstats (into a paper, a slide, a post),
    # so the CI method/correction it was computed with travels with it
    # rather than only living in the surrounding terminal report. Placed
    # between the title and the axes (not below the plot) so it doesn't
    # read as a second, redundant caption once a LaTeX \caption{} is added
    # underneath the whole figure.
    caption_parts = []
    pretty_ci_method = _pretty(ci_method)
    if pretty_ci_method:
        caption_parts.append(f"CI method: {pretty_ci_method}")
    pretty_correction = _pretty(correction)
    if pretty_correction and pretty_correction != "none":
        caption_parts.append(f"FWER correction: {pretty_correction}")
    caption_parts.append(f"α={alpha:g}")
    if any_gradient_used:
        caption_parts.append("darker band = higher confidence")
    caption = "  |  ".join(caption_parts)

    ax.set_title(
        title,
        fontsize=10,
        color=_PALETTE["text"],
        pad=24 if caption else 10,
        loc="center",
    )
    if caption:
        ax.text(
            0.5, 1.02, caption,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.5, color=_PALETTE["text_secondary"],
        )

    # ---- legend -------------------------------------------------------------
    # One combined legend: entity-tier colours, plus (in gradient mode) a
    # neutral-colour swatch per confidence band -- so a reader encountering
    # this figure with no surrounding context (pasted into a paper, a slide,
    # a social post) can still read it unaided.
    legend_handles: list = []
    if compare_to is not None:
        # Primary vs. comparison is conveyed by thickness + tint (each row
        # keeps its own hue for both) -- a neutral swatch pair mirroring
        # that exact treatment (thin/light vs. thick/full) represents the
        # distinction regardless of color_rule.
        r_label = report_label or "primary"
        c_label = compare_label or "comparison"
        legend_handles += [
            Line2D([0], [0], color=_lighten(_PALETTE["text_secondary"], _LIGHTEN_AMOUNT),
                   lw=lw * 0.6, solid_capstyle="round", label=c_label),
            Line2D([0], [0], color=_PALETTE["text_secondary"], lw=lw,
                   solid_capstyle="round", label=r_label),
        ]
    elif color_rule == "tier" and unbeaten:
        # Only "tier" mode has a colour-meaning legend -- "factor" and a
        # literal colour spec both make colour track entity identity (or
        # nothing at all), which the y-axis labels already convey.
        legend_handles += [
            Line2D([0], [0], color=_PALETTE["unbeaten"], lw=lw,
                   solid_capstyle="round", label="Unbeaten"),
            Line2D([0], [0], color=_PALETTE["lower_tier"], lw=lw,
                   solid_capstyle="round", label="Significantly worse"),
        ]
    if any_gradient_used:
        neutral = _PALETTE["text_secondary"]
        # Same drawing order as the bands themselves: widest/lightest (99%)
        # first, narrowest/darkest (68%) last.
        band_labels = ["99% CI", "95% CI", "90% CI", "68% CI"]
        legend_handles += [
            Patch(facecolor=neutral, alpha=a, edgecolor="none", label=lbl)
            for a, lbl in zip(_GRADIENT_BAND_ALPHAS, band_labels)
        ]
    if show_mean and mean_marker == "line":
        legend_handles.append(
            Line2D([0], [0], color="black", lw=1.5, label="mean")
        )
    if show_ci_bracket and any_gradient_used:
        legend_handles.append(
            Line2D([0], [0], color=_PALETTE["text"], lw=1.3, label=f"{ci_pct}% CI (bracket)")
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5),
            frameon=True, facecolor="white",
            edgecolor=_PALETTE["grid"], framealpha=0.95,
            ncol=1,
        )

    if own_fig:
        fig.tight_layout()
        if legend_handles:
            # Legend sits outside the axes (bbox_to_anchor to the right) so
            # it never overlaps the bars -- reserve room for it with a
            # generous initial guess, then trim the canvas to hug the
            # legend's actual rendered width instead of leaving whatever
            # blank margin that guess didn't use (matters for pasting
            # straight into a paper without manual cropping).
            fig.subplots_adjust(right=0.78)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend = ax.get_legend()
            legend_px = legend.get_window_extent(renderer=renderer)
            fig_px_width = fig.get_window_extent(renderer=renderer).width
            pad_px = 8
            excess_px = fig_px_width - (legend_px.x1 + pad_px)
            if excess_px > 1:
                dpi = fig.dpi
                old_width_in, height_in = fig.get_size_inches()
                new_width_in = old_width_in - excess_px / dpi
                if new_width_in > 0:
                    # Rescale horizontal subplot fractions so the axes and
                    # legend keep their exact pixel position/size on the
                    # narrower canvas -- only the wasted margin is trimmed.
                    sp = fig.subplotpars
                    scale = old_width_in / new_width_in
                    fig.set_size_inches(new_width_in, height_in)
                    fig.subplots_adjust(
                        left=min(0.99, sp.left * scale),
                        right=min(1.0, sp.right * scale),
                    )

    return fig
