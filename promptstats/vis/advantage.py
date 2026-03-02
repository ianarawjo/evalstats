"""Mean advantage plot with dual uncertainty bands.

The signature promptstats visualization. For each template, shows:

- A point for the mean advantage over a reference
- A thin inner band for the bootstrap CI on the mean (epistemic uncertainty:
  "how sure are we about the mean?")
- A wider outer band for the score spread (intrinsic variance: "how
  consistent is this template?")

This separates two fundamentally different concerns:
- Epistemic uncertainty shrinks with more benchmark inputs.
- Intrinsic variance is a property of the template and won't shrink.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from promptstats.core.types import BenchmarkResult
from promptstats.core.ranking import bootstrap_mean_advantage, MeanAdvantageResult


# -- Color palette --
_PALETTE = {
    "spread_band": "#D6E4F0",       # light blue — intrinsic spread
    "spread_edge": "#A3C1DA",       # medium blue — spread border
    "ci_band": "#2B6E99",           # dark blue — CI band
    "ci_edge": "#1A4A6B",           # darker blue — CI border
    "point_pos": "#1A4A6B",         # dark blue — point when positive
    "point_neg": "#8B4D5C",         # muted red — point when negative
    "point_zero": "#666666",        # gray — point when CI spans zero
    "zero_line": "#CCCCCC",         # light gray — zero reference
    "sig_marker": "#2B8C3E",        # green — significance marker
    "grid": "#F0F0F0",             # very light gray — grid
    "text": "#333333",             # dark gray — labels
    "text_secondary": "#777777",    # medium gray — secondary text
}


def plot_mean_advantage(
    result: BenchmarkResult | MeanAdvantageResult,
    reference: str = "grand_mean",
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    spread_percentiles: tuple[float, float] = (10, 90),
    sort_by: str = "mean",
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> Figure:
    """Plot mean advantage with dual uncertainty bands.

    Parameters
    ----------
    result : BenchmarkResult or MeanAdvantageResult
        Either raw benchmark data (will compute advantage internally) or
        a pre-computed MeanAdvantageResult.
    reference : str
        Reference for advantage computation. Either 'grand_mean' or a
        template label. Ignored if result is already a MeanAdvantageResult.
    n_bootstrap : int
        Bootstrap iterations. Ignored if result is a MeanAdvantageResult.
    ci : float
        Confidence level. Ignored if result is a MeanAdvantageResult.
    spread_percentiles : tuple[float, float]
        Percentiles for the spread band. Ignored if result is a
        MeanAdvantageResult.
    sort_by : str
        Sort order: 'mean' (descending by mean advantage), 'label'
        (alphabetical), or 'spread' (ascending by spread width).
    figsize : tuple[float, float], optional
        Figure size. Defaults to (10, 0.5 * N_templates + 1.5).
    title : str, optional
        Plot title. Defaults to a descriptive title.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Compute advantage if given raw BenchmarkResult
    if isinstance(result, BenchmarkResult):
        scores = result.get_2d_scores()
        adv = bootstrap_mean_advantage(
            scores=scores,
            labels=result.template_labels,
            reference=reference,
            n_bootstrap=n_bootstrap,
            ci=ci,
            spread_percentiles=spread_percentiles,
            rng=rng,
        )
    else:
        adv = result

    n = len(adv.labels)

    # Sort
    if sort_by == "mean":
        order = np.argsort(-adv.mean_advantages)
    elif sort_by == "label":
        order = np.argsort(adv.labels)
    elif sort_by == "spread":
        spread_widths = adv.spread_high - adv.spread_low
        order = np.argsort(spread_widths)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    labels = [adv.labels[i] for i in order]
    means = adv.mean_advantages[order]
    ci_lo = adv.bootstrap_ci_low[order]
    ci_hi = adv.bootstrap_ci_high[order]
    sp_lo = adv.spread_low[order]
    sp_hi = adv.spread_high[order]

    # Figure setup
    if figsize is None:
        figsize = (10, max(3, 0.5 * n + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = np.arange(n)

    # Spread band height and CI band height
    spread_height = 0.32
    ci_height = 0.14

    # Draw zero reference line
    ax.axvline(x=0, color=_PALETTE["zero_line"], linewidth=1.2, zorder=1)

    for i, y in enumerate(y_positions):
        # Determine point color based on significance
        if ci_lo[i] > 0:
            point_color = _PALETTE["point_pos"]
        elif ci_hi[i] < 0:
            point_color = _PALETTE["point_neg"]
        else:
            point_color = _PALETTE["point_zero"]

        # Outer band: intrinsic spread (10th-90th percentile)
        spread_rect = mpatches.FancyBboxPatch(
            (sp_lo[i], y - spread_height / 2),
            sp_hi[i] - sp_lo[i],
            spread_height,
            boxstyle="round,pad=0.02",
            facecolor=_PALETTE["spread_band"],
            edgecolor=_PALETTE["spread_edge"],
            linewidth=0.8,
            zorder=2,
        )
        ax.add_patch(spread_rect)

        # Inner band: bootstrap CI on the mean
        ci_rect = mpatches.FancyBboxPatch(
            (ci_lo[i], y - ci_height / 2),
            ci_hi[i] - ci_lo[i],
            ci_height,
            boxstyle="round,pad=0.01",
            facecolor=_PALETTE["ci_band"],
            edgecolor=_PALETTE["ci_edge"],
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(ci_rect)

        # Center point: mean advantage
        ax.plot(
            means[i], y,
            "o",
            color="white",
            markeredgecolor=point_color,
            markeredgewidth=1.5,
            markersize=7,
            zorder=4,
        )

        # Significance indicator
        if ci_lo[i] > 0 or ci_hi[i] < 0:
            ax.plot(
                means[i], y,
                "o",
                color=point_color,
                markersize=3.5,
                zorder=5,
            )

    # Axis configuration
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10, color=_PALETTE["text"])
    ax.invert_yaxis()  # best template at top

    ax.set_xlabel(
        f"Advantage over {adv.reference}",
        fontsize=10,
        color=_PALETTE["text"],
        labelpad=8,
    )

    # Grid
    ax.xaxis.grid(True, color=_PALETTE["grid"], linewidth=0.5, zorder=0)
    ax.yaxis.grid(False)

    # Remove spines
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_PALETTE["zero_line"])

    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", colors=_PALETTE["text_secondary"], labelsize=9)

    # Title
    if title is None:
        ref_desc = (
            "grand mean" if adv.reference == "grand_mean"
            else f"'{adv.reference}'"
        )
        title = f"Mean advantage over {ref_desc}"

    ax.set_title(
        title,
        fontsize=12,
        color=_PALETTE["text"],
        pad=12,
        loc="left",
        fontweight="semibold",
    )

    # Legend
    sp_lo_pct, sp_hi_pct = adv.spread_percentiles
    ci_pct = int(round((1 - 2 * (1 - (adv.bootstrap_ci_high[0] is not None))) * 100))
    # Infer CI from the fact that we used the ci parameter
    legend_handles = [
        mpatches.Patch(
            facecolor=_PALETTE["spread_band"],
            edgecolor=_PALETTE["spread_edge"],
            linewidth=0.8,
            label=f"Score spread ({sp_lo_pct}th–{sp_hi_pct}th pctl)",
        ),
        mpatches.Patch(
            facecolor=_PALETTE["ci_band"],
            edgecolor=_PALETTE["ci_edge"],
            linewidth=1.0,
            label=f"{int(ci * 100)}% CI on mean (bootstrap, n={adv.n_bootstrap:,})",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor=_PALETTE["grid"],
        framealpha=0.9,
    )

    # Annotation explaining the two bands
    ax.annotate(
        "wider light band = inconsistent template · "
        "wider dark band = uncertain estimate",
        xy=(0.5, -0.02),
        xycoords="axes fraction",
        ha="center",
        fontsize=7.5,
        color=_PALETTE["text_secondary"],
        style="italic",
    )

    fig.tight_layout()
    return fig
