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
    "compare":       "#c0d8f0",  # light blue   — background / comparison report
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


def plot_ci_forest(
    report,
    compare_to=None,
    report_label: Optional[str] = None,
    compare_label: Optional[str] = None,
    reference_line: Optional[float] = 0.5,
    sort_by: str = "mean",
    as_percent: bool = True,
    style: Literal["gradient", "single"] = "gradient",
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
        single-run eval).  Its CIs are drawn in a lighter colour offset
        above each row so both intervals are visible simultaneously.
        Both reports must contain the same entity labels. Always drawn in
        the single-band style regardless of *style*, to keep the overlay
        legible.
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
    offset = 0.18 if compare_to is not None else 0.0

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

    for i, label in enumerate(ordered_labels):
        y = float(y_positions[i])
        mean5, lo5, hi5 = _ci(report, label)

        # Primary CI tier colour
        if not unbeaten:
            # No significant differences — use neutral colour
            color = _PALETTE["no_sig"]
        elif label in unbeaten:
            color = _PALETTE["unbeaten"]
        else:
            color = _PALETTE["lower_tier"]

        if compare_to is not None:
            # Comparison report — lighter, offset above
            mean0, lo0, hi0 = _ci(compare_to, label)
            ax.plot(
                [lo0, hi0], [y + offset, y + offset],
                color=_PALETTE["compare"], lw=lw,
                solid_capstyle="round", zorder=2,
            )
            ax.scatter(
                [mean0], [y + offset],
                color=_PALETTE["compare"], s=ms, zorder=3,
            )

        # Primary CI — full colour, offset below when compare_to given.
        # Gradient style falls back to single-band per-entity when this
        # entity has no multi_ci data (e.g. a Wald-type CI).
        multi_ci = _multi_ci(report, label) if style == "gradient" else None
        y_row = y - offset
        if multi_ci is not None:
            any_gradient_used = True
            # Widest CI (99%, smallest alpha) drawn first/lowest zorder,
            # narrowest (68%, largest alpha) drawn last/highest zorder --
            # same "inner band wins" convention as the terminal's
            # _gradient_interval_line, via z-order layering instead of
            # character replacement.
            sorted_alphas = sorted(multi_ci.keys())
            for band_i, a in enumerate(sorted_alphas):
                lo_a, hi_a = multi_ci[a]
                band_alpha = _GRADIENT_BAND_ALPHAS[
                    min(band_i, len(_GRADIENT_BAND_ALPHAS) - 1)
                ]
                ax.barh(
                    y_row, width=hi_a - lo_a, left=lo_a,
                    height=_GRADIENT_BAND_HEIGHT,
                    color=color, alpha=band_alpha,
                    edgecolor="none", zorder=4 + band_i,
                )
        else:
            ax.plot(
                [lo5, hi5], [y_row, y_row],
                color=color, lw=lw,
                solid_capstyle="round", zorder=4,
            )

        top_zorder = 4 + len(_GRADIENT_BAND_ALPHAS) + 1

        # Optional traditional bracket-style CI overlaid on top of the
        # gradient bands, at the report's own single confidence level --
        # for readers who want that familiar landmark alongside the gradient.
        if show_ci_bracket and multi_ci is not None:
            ax.plot(
                [lo5, hi5], [y_row, y_row],
                color=_PALETTE["text"], lw=1.3, zorder=top_zorder,
                solid_capstyle="butt",
            )
            cap_h = _GRADIENT_BAND_HEIGHT * 0.22
            for x_cap in (lo5, hi5):
                ax.plot(
                    [x_cap, x_cap], [y_row - cap_h, y_row + cap_h],
                    color=_PALETTE["text"], lw=1.3, zorder=top_zorder,
                )
            top_zorder += 1

        if show_mean:
            if mean_marker == "line":
                tick_h = _GRADIENT_BAND_HEIGHT * 0.7
                ax.plot(
                    [mean5, mean5], [y_row - tick_h, y_row + tick_h],
                    color="black", lw=1.5, zorder=top_zorder + 1,
                )
            else:
                ax.scatter(
                    [mean5], [y_row],
                    color=color, s=ms, zorder=top_zorder + 1,
                    edgecolor="white", linewidth=0.6,
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
        r_label = report_label or "primary"
        c_label = compare_label or "comparison"
        legend_handles += [
            Line2D([0], [0], color=_PALETTE["compare"], lw=lw,
                   solid_capstyle="round", label=c_label),
            Line2D([0], [0], color=_PALETTE["unbeaten"], lw=lw,
                   solid_capstyle="round", label=r_label),
        ]
    elif unbeaten:
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
            # it never overlaps the bars -- reserve room for it.
            fig.subplots_adjust(right=0.76)

    return fig
