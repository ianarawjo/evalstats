"""Per-entity CI forest plot for CompareReport objects.

Draws one horizontal CI bar per entity (prompt or model), coloured by
statistical tier, with the best-performing entity at the top.  An optional
second report can be overlaid for direct before/after comparison (e.g.,
to show how CI widths change when you double the eval set or add more runs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from evalstats.compare import CompareReport


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


def plot_ci_forest(
    report: CompareReport,
    compare_to: Optional[CompareReport] = None,
    report_label: Optional[str] = None,
    compare_label: Optional[str] = None,
    reference_line: Optional[float] = 0.5,
    sort_by: str = "mean",
    as_percent: bool = True,
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
        Both reports must contain the same entity labels.
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

    def _ci(rep: CompareReport, label: str) -> tuple[float, float, float]:
        s = rep.entity_stats[label]
        return s.mean * scale, s.ci_low * scale, s.ci_high * scale

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

        # Primary CI — full colour, offset below when compare_to given
        ax.plot(
            [lo5, hi5], [y - offset, y - offset],
            color=color, lw=lw,
            solid_capstyle="round", zorder=4,
        )
        ax.scatter(
            [mean5], [y - offset],
            color=color, s=ms, zorder=5,
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

    # ---- title ------------------------------------------------------------
    if title is None:
        n_inputs = report.full_analysis.n_inputs if hasattr(report.full_analysis, "n_inputs") else ""
        n_str = f"  |  N={n_inputs} inputs" if n_inputs else ""
        ci_pct = int(getattr(report, "ci", 0.95) * 100)
        title = f"95% confidence intervals per {report.entity_name_singular}{n_str}"

    ax.set_title(
        title,
        fontsize=10,
        color=_PALETTE["text"],
        pad=10,
        loc="center",
    )

    # ---- legend -----------------------------------------------------------
    if compare_to is not None:
        r_label = report_label or "primary"
        c_label = compare_label or "comparison"
        legend_handles = [
            Line2D([0], [0], color=_PALETTE["compare"], lw=lw,
                   solid_capstyle="round", label=c_label),
            Line2D([0], [0], color=_PALETTE["unbeaten"], lw=lw,
                   solid_capstyle="round", label=r_label),
        ]
        ax.legend(
            handles=legend_handles,
            fontsize=8, loc="lower right",
            frameon=True, facecolor="white",
            edgecolor=_PALETTE["grid"], framealpha=0.95,
        )
    elif unbeaten:
        legend_handles = [
            Line2D([0], [0], color=_PALETTE["unbeaten"], lw=lw,
                   solid_capstyle="round", label="In contention"),
            Line2D([0], [0], color=_PALETTE["lower_tier"], lw=lw,
                   solid_capstyle="round", label="Outperformed"),
        ]
        ax.legend(
            handles=legend_handles,
            fontsize=8, loc="lower right",
            frameon=True, facecolor="white",
            edgecolor=_PALETTE["grid"], framealpha=0.95,
        )

    if own_fig:
        fig.tight_layout()

    return fig
