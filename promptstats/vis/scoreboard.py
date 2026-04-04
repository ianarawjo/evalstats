"""Raw accuracy bar chart ("scoreboard") for prompt or model comparisons.

A simple first-look visualization before statistical testing.  Shows mean
accuracy per entity as a bar, with an optional dashed baseline reference
line to make relative gains immediately visible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Color palette (consistent with the rest of the vis module)
# ---------------------------------------------------------------------------

_PALETTE = {
    "bar":           "#94b8e0",  # soft blue   — bars
    "bar_highlight": "#4a90d9",  # medium blue — highlighted bar
    "baseline_line": "#777777",  # mid gray    — baseline reference
    "grid":          "#EEF1F4",  # very light  — y grid
    "text":          "#2D333B",  # dark slate  — axis labels
    "text_secondary":"#6B7280",  # muted gray  — tick labels
    "errorbar":      "#2D333B",  # dark slate  — CI whiskers/caps
}


def plot_accuracy_bar(
    scores: Union[dict, "CompareReport"],  # noqa: F821
    baseline: Optional[str] = None,
    cis: Optional[Mapping[str, Sequence[float]]] = None,
    sort_by: str = "input_order",
    as_percent: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    ax: Optional["Axes"] = None,
) -> "Figure":
    """Bar chart of mean accuracy per prompt or model (raw scoreboard).

    A quick, un-corrected view intended as a starting point before
    statistical analysis.  To visualise confidence intervals and
    significance tiers, use :func:`plot_ci_forest` instead.

    Parameters
    ----------
    scores : dict or CompareReport
        One of:

        * ``dict[str, float]`` — pre-computed mean per entity.
        * ``dict[str, array-like]`` — raw score arrays; means are computed
          internally.
        * :class:`~promptstats.compare.CompareReport` — uses the
          ``entity_stats`` means from the report.
    baseline : str, optional
        Label of a baseline entity.  A dashed reference line is drawn at
        its mean accuracy so gains and losses over the baseline are
        immediately visible.
    cis : mapping[str, sequence[float]], optional
        Optional confidence intervals per entity label, used to draw
        vertical error bars on each bar.

        Expected format: ``{label: (ci_low, ci_high)}`` in raw score units
        (0-1), regardless of ``as_percent``.
    sort_by : str
        Bar ordering:

        * ``"input_order"`` (default) — preserves the dict / label order.
        * ``"mean"`` — descending by mean; best entity leftmost.
        * ``"label"`` — alphabetical.
    as_percent : bool
        When ``True`` (default), display values as percentages (0–100).
        Set to ``False`` to keep raw (0–1) scores.
    figsize : tuple[float, float], optional
        Figure size.  Defaults to ``(max(5, 0.9 * N + 1.5), 3.8)``.
    title : str, optional
        Plot title.  A descriptive default is generated when omitted.
    ax : Axes, optional
        Existing axes to draw into.  A new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ---- normalise input --------------------------------------------------
    from promptstats.compare import CompareReport

    entity_name = "prompt"

    if isinstance(scores, CompareReport):
        entity_name = scores.entity_name_singular
        labels = scores.labels
        means_raw = {l: scores.entity_stats[l].mean for l in labels}
    elif isinstance(scores, dict):
        labels = list(scores.keys())
        means_raw = {}
        for label, val in scores.items():
            arr = np.asarray(val, dtype=np.float64)
            means_raw[label] = float(np.nanmean(arr)) if arr.ndim > 0 else float(arr)
    else:
        raise TypeError(
            "scores must be a dict or a CompareReport. "
            f"Got {type(scores).__name__!r}."
        )

    n = len(labels)
    if n == 0:
        raise ValueError("scores is empty.")

    # ---- sort order -------------------------------------------------------
    means_arr = np.array([means_raw[l] for l in labels])
    if sort_by == "input_order":
        order = list(range(n))
    elif sort_by == "mean":
        order = list(np.argsort(-means_arr))
    elif sort_by == "label":
        order = sorted(range(n), key=lambda i: labels[i])
    else:
        raise ValueError(
            f"Unknown sort_by: {sort_by!r}. "
            "Expected 'input_order', 'mean', or 'label'."
        )

    ordered_labels = [labels[i] for i in order]
    scale = 100.0 if as_percent else 1.0
    ordered_means = [means_raw[l] * scale for l in ordered_labels]

    yerr = None
    if cis is not None:
        missing = [label for label in ordered_labels if label not in cis]
        if missing:
            raise ValueError(
                "cis is missing labels present in scores: "
                f"{missing}."
            )

        yerr_low = []
        yerr_high = []
        for label in ordered_labels:
            ci_bounds = cis[label]
            if len(ci_bounds) != 2:
                raise ValueError(
                    f"cis[{label!r}] must be a pair (ci_low, ci_high)."
                )

            ci_low = float(ci_bounds[0])
            ci_high = float(ci_bounds[1])
            mean = means_raw[label]

            if ci_low > ci_high:
                raise ValueError(
                    f"cis[{label!r}] has ci_low > ci_high: "
                    f"({ci_low}, {ci_high})."
                )

            if not (ci_low <= mean <= ci_high):
                raise ValueError(
                    f"Mean for {label!r} ({mean:.6g}) is outside the provided "
                    f"CI ({ci_low:.6g}, {ci_high:.6g})."
                )

            yerr_low.append((mean - ci_low) * scale)
            yerr_high.append((ci_high - mean) * scale)

        yerr = np.vstack([yerr_low, yerr_high])

    # ---- figure setup -----------------------------------------------------
    own_fig = ax is None
    if own_fig:
        if figsize is None:
            figsize = (max(5.0, 0.9 * n + 1.5), 3.8)
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    else:
        fig = ax.get_figure()

    ax.set_facecolor("white")

    # ---- bars -------------------------------------------------------------
    x = np.arange(n)
    ax.bar(
        x,
        ordered_means,
        color=_PALETTE["bar"],
        width=0.6,
        yerr=yerr,
        ecolor=_PALETTE["errorbar"],
        capsize=3,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "zorder": 5},
        zorder=3,
    )

    # ---- baseline reference line ------------------------------------------
    if baseline is not None:
        if baseline not in means_raw:
            raise ValueError(
                f"baseline label {baseline!r} not found in scores. "
                f"Available labels: {list(means_raw.keys())}"
            )
        baseline_val = means_raw[baseline] * scale
        ax.axhline(
            baseline_val,
            color=_PALETTE["baseline_line"],
            lw=1.2, ls="--",
            label=f"Baseline ({baseline})",
            zorder=4,
        )
        ax.legend(fontsize=8)

    # ---- axes styling -----------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(
        ordered_labels, rotation=35, ha="right",
        fontsize=9, color=_PALETTE["text"],
    )
    ax.set_ylabel(
        "Accuracy (%)" if as_percent else "Score",
        fontsize=10, color=_PALETTE["text"],
    )
    ax.set_ylim(0, 100 if as_percent else 1.0)

    if as_percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    ax.grid(axis="y", color=_PALETTE["grid"], alpha=0.9, zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")

    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", colors=_PALETTE["text_secondary"], labelsize=9)

    # ---- title ------------------------------------------------------------
    if title is None:
        title = f"Mean accuracy per {entity_name}  (raw scoreboard, no correction)"

    ax.set_title(
        title,
        fontsize=10, color=_PALETTE["text"],
        pad=10, loc="center",
    )

    if own_fig:
        fig.tight_layout()

    return fig
