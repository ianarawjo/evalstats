"""Uncertainty-aware Pareto trade-off scatter (matplotlib).

Backs ``compare(..., secondary=...)``'s and :func:`~evalstats.tradeoff`'s
``.plot()``. Each entity is drawn as its point estimate plus a light cloud
of its own joint bootstrap replicates -- the same replicates
:func:`~evalstats.core.pareto.pareto_bootstrap` already draws to compute the
calibrated Pareto status and ``P(Pareto-optimal)``, not a separate,
independently-resampled cloud. Unlike independent per-axis error bars, the
cloud's shape is honest about *correlation* between the two metrics (e.g.
harder items being both slower and less accurate shows up as a tilted
cloud), and needs no distributional assumption (no covariance-ellipse
fitting) to be read correctly.

The frontier itself is drawn the same uncertainty-aware way: instead of one
crisp line through the point-estimate frontier's members (implying more
confidence than the data has about both *which* entities belong on it and
the line's exact shape), :func:`_bootstrap_frontier_ensemble` recomputes the
non-dominated set per bootstrap replicate and overlays many faint candidate
frontiers -- solid where replicates agree, frayed wherever a point is only
sometimes on the frontier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np

from evalstats.vis.forest import _PALETTE

if TYPE_CHECKING:
    from matplotlib.figure import Figure


_STATUS_COLOR = {
    "frontier": _PALETTE["unbeaten"],
    "dominated": _PALETTE["lower_tier"],
    "ambiguous": "#c98a1f",  # amber -- distinct from both blue and red
}
_STATUS_MARKER = {"frontier": "*", "dominated": "x", "ambiguous": "o"}
_STATUS_LABEL = {
    "frontier": "Best trade-off",
    "dominated": "Dominated",
    "ambiguous": "Ambiguous (unconfirmed)",
}


def _assign_label_offsets(xs, ys):
    """Greedy collision avoidance for point labels: cluster points that sit
    close together (in axis-range-normalized distance, so both axes count
    equally regardless of their units) and fan each cluster's labels out to
    different offset positions instead of stacking every label at the same
    fixed (dx, dy). Returns a list of (dx, dy, ha) per point, in points, for
    ``ax.annotate(..., textcoords="offset points", ha=ha)``.
    """
    n = len(xs)
    x_range = max(np.ptp(xs), 1e-9)
    y_range = max(np.ptp(ys), 1e-9)
    nx = (np.asarray(xs) - np.min(xs)) / x_range
    ny = (np.asarray(ys) - np.min(ys)) / y_range

    threshold = 0.06
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            d = ((nx[i] - nx[j]) ** 2 + (ny[i] - ny[j]) ** 2) ** 0.5
            if d < threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    # Candidate (dx, dy, ha) offsets, in points, roughly spiraling outward
    # so a cluster of several points still gets distinct, readable spots.
    candidates = [
        (7, 6, "left"), (7, -15, "left"), (-7, 6, "right"), (-7, -15, "right"),
        (7, 20, "left"), (7, -29, "left"), (-7, 20, "right"), (-7, -29, "right"),
        (14, 34, "left"), (14, -43, "left"),
    ]
    offsets: list[tuple] = [None] * n
    for members in clusters.values():
        members_sorted = sorted(members, key=lambda i: -ys[i])  # stable top-to-bottom stacking
        for k, i in enumerate(members_sorted):
            offsets[i] = candidates[k % len(candidates)]
    return offsets


def _resolve_label_overlaps(fig, anns, max_iter=40, step=10.0):
    """Push apart any pair of annotation text boxes that still overlap
    after the cluster-based initial placement, using real rendered bounding
    boxes -- accounts for actual label text width/height, which the
    point-distance heuristic in :func:`_assign_label_offsets` can't (a long
    label can collide with a neighbor that isn't "close" by position alone).
    """
    if len(anns) < 2:
        return
    margin = 4.0  # px -- stop once there's real whitespace, not just zero overlap
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(max_iter):
        boxes = [a.get_window_extent(renderer=renderer).padded(margin) for a in anns]
        moved = False
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if boxes[i].overlaps(boxes[j]):
                    dx_i, dy_i = anns[i].xyann
                    dx_j, dy_j = anns[j].xyann
                    if boxes[i].y0 >= boxes[j].y0:
                        anns[i].xyann = (dx_i, dy_i + step)
                        anns[j].xyann = (dx_j, dy_j - step)
                    else:
                        anns[j].xyann = (dx_j, dy_j + step)
                        anns[i].xyann = (dx_i, dy_i - step)
                    moved = True
        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def _bootstrap_frontier_ensemble(replicate_primary, replicate_secondary, direction, n_lines, rng):
    """Per-replicate Pareto frontiers, for drawing as a translucent band
    instead of one crisp line through the point-estimate frontier.

    A single line connecting the *mean* frontier's members implies more
    certainty than the data has -- both about which entities belong on it
    and about the line's exact shape between neighbors. Recomputing the
    same non-dominated test :func:`~evalstats.core.pareto.pareto_bootstrap`
    already applies internally, but per-replicate instead of on the
    aggregate tallies, gives ``n_lines`` candidate frontiers; overlaying
    them faintly shows where replicates agree (the band is solid) and
    where they don't (a point that's only sometimes on the frontier makes
    the band fray right around it).

    Parameters
    ----------
    replicate_primary, replicate_secondary : np.ndarray
        Shape ``(N, n_bootstrap)``, already oriented so higher = better on
        both axes (``ParetoBootstrapResult.replicate_primary`` /
        ``replicate_secondary``).
    direction : {"min", "max"}
        Secondary metric's real-world direction, to un-orient x back to
        display units.
    n_lines : int
        Number of replicates to draw (subsampled from the full bootstrap).
    rng : np.random.Generator

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        One ``(xs, ys)`` pair per drawn replicate, x-sorted, ready for
        ``ax.plot``.
    """
    n_entities, n_bootstrap = replicate_primary.shape
    n_lines = min(n_lines, n_bootstrap)
    cols = rng.choice(n_bootstrap, size=n_lines, replace=False)
    eye = np.eye(n_entities, dtype=bool)

    lines = []
    for j in cols:
        p1 = replicate_primary[:, j]
        p2 = replicate_secondary[:, j]
        ge_both = (p1[None, :] >= p1[:, None]) & (p2[None, :] >= p2[:, None])
        gt_either = (p1[None, :] > p1[:, None]) | (p2[None, :] > p2[:, None])
        dominates = ge_both & gt_either
        dominates &= ~eye
        is_dominated = dominates.any(axis=1)
        frontier = np.where(~is_dominated)[0]
        if len(frontier) < 2:
            continue
        x_display = p2[frontier] if direction == "max" else -p2[frontier]
        y_display = p1[frontier]
        order = np.argsort(x_display)
        lines.append((x_display[order], y_display[order]))
    return lines


def plot_pareto_tradeoff(
    pareto: dict,
    *,
    metric: Optional[str] = None,
    title: Optional[str] = None,
    n_cloud_points: int = 300,
    n_frontier_lines: int = 60,
    figsize: Optional[tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> "Figure":
    """Scatter plot of the primary vs. secondary metric, one bootstrap point
    cloud per entity, colored/marked by Pareto status.

    Requires *pareto* to carry joint bootstrap replicates -- i.e. it must
    come from ``compare(secondary=...)`` or :func:`~evalstats.tradeoff`,
    both of which request them via ``pareto_bootstrap(...,
    return_replicates=True)``.

    Parameters
    ----------
    pareto : dict
        The internal Pareto-analysis dict (``ComparisonResult._pareto`` /
        ``TradeoffResult._pareto``) -- same object
        :func:`~evalstats.core.summary._print_pareto_section` prints from.
    metric : str, optional
        Display name for the primary metric (e.g. "accuracy"). Falls back
        to "primary metric" when omitted.
    title : str, optional
        Plot title. Defaults to ``"{metric} vs. {secondary_metric}
        Trade-off"``.
    n_cloud_points : int
        Bootstrap replicates shown per entity (subsampled from the full
        joint bootstrap for a legible, not over-dense, cloud).
    n_frontier_lines : int
        Per-replicate Pareto frontiers drawn as a translucent band, in
        place of a single line through the point-estimate frontier (see
        :func:`_bootstrap_frontier_ensemble`) -- a single line implies more
        certainty about frontier membership and shape than the data has;
        overlaying many faint per-replicate frontiers instead shows where
        replicates agree (solid band) and where a point is only sometimes
        on the frontier (the band frays right around it).
    figsize : tuple[float, float], optional
        Figure size. Defaults to ``(7, 5.2)``.
    rng : np.random.Generator, optional
        Controls which replicates are subsampled for display. Reproducible
        by default (a fixed internal seed) when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    result = pareto["result"]
    statuses = pareto["statuses"]
    secondary_col = pareto["secondary_metric"]
    direction = pareto["direction"]
    if result.replicate_primary is None or result.replicate_secondary is None:
        raise ValueError(
            "pareto['result'] has no bootstrap replicates -- "
            "plot_pareto_tradeoff() requires pareto_bootstrap(..., "
            "return_replicates=True), which compare(secondary=...) and "
            "tradeoff() both request automatically."
        )

    labels = list(result.labels)
    metric_label = metric or "primary metric"

    # Point estimates come straight from the bootstrap result (already
    # oriented so higher = better on both axes); undo the secondary's
    # orientation flip for display so the axis reads in real units.
    order = {l: i for i, l in enumerate(labels)}
    ys = np.array([float(result.point_primary[order[l]]) for l in labels])
    xs_oriented = np.array([float(result.point_secondary[order[l]]) for l in labels])
    xs = xs_oriented if direction == "max" else -xs_oriented

    statuses_list = [statuses[l].status for l in labels]
    label_offsets = _assign_label_offsets(xs, ys)

    if figsize is None:
        figsize = (7, 5.2)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    rng = np.random.default_rng(rng) if rng is not None else np.random.default_rng(0)

    ensemble = _bootstrap_frontier_ensemble(
        result.replicate_primary, result.replicate_secondary, direction, n_frontier_lines, rng,
    )
    for line_x, line_y in ensemble:
        ax.plot(line_x, line_y, color=_PALETTE["unbeaten"], lw=1.0, alpha=0.05, zorder=1)

    ann_list = []
    for i, lbl in enumerate(labels):
        color = _STATUS_COLOR[statuses_list[i]]
        rx = result.replicate_secondary[i] if direction == "max" else -result.replicate_secondary[i]
        ry = result.replicate_primary[i]
        n_show = min(n_cloud_points, len(rx))
        sub = rng.choice(len(rx), size=n_show, replace=False)
        ax.scatter(rx[sub], ry[sub], s=6, color=color, alpha=0.12, zorder=2, linewidths=0)
        ax.scatter(
            [xs[i]], [ys[i]], marker=_STATUS_MARKER[statuses_list[i]],
            s=170 if statuses_list[i] == "frontier" else 90,
            facecolor=color, edgecolor="white", linewidth=0.8, zorder=3,
        )
        dx, dy, ha = label_offsets[i]
        ann_list.append(ax.annotate(
            lbl, (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
            ha=ha, fontsize=9, color=_PALETTE["text"],
        ))

    # The cluster-based initial offsets keep obviously-close points apart,
    # but can't account for actual label text width -- follow up with a
    # real bounding-box collision pass on the rendered text.
    _resolve_label_overlaps(fig, ann_list)

    dir_arrow = "← better" if direction == "min" else "→ better"
    ax.set_xlabel(f"{secondary_col} ({dir_arrow})", fontsize=10, color=_PALETTE["text"])
    # matplotlib rotates the y-label 90 degrees counterclockwise, so a
    # "→" in the (unrotated) label text ends up pointing up on the page.
    ax.set_ylabel(f"{metric_label} (→ better)", fontsize=10, color=_PALETTE["text"])
    if title is None:
        title = f"{metric_label} vs. {secondary_col} Trade-off"
    ax.set_title(title, fontsize=12, color=_PALETTE["text"], pad=14)
    ax.grid(True, color=_PALETTE["grid"], alpha=0.9, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=_PALETTE["text_secondary"], labelsize=9)

    handles = [
        plt.Line2D([0], [0], marker=_STATUS_MARKER[s], linestyle="none",
                   markerfacecolor=_STATUS_COLOR[s], markeredgecolor=_STATUS_COLOR[s],
                   markeredgewidth=1.5 if _STATUS_MARKER[s] == "x" else 1.0,
                   markersize=10, label=_STATUS_LABEL[s])
        for s in ["frontier", "ambiguous", "dominated"]
        if s in statuses_list
    ]
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=8.5, frameon=False)

    fig.tight_layout()
    return fig
