"""Critical difference diagram for Nemenyi post-hoc comparisons.

Delegates rendering to ``scikit_posthocs.critical_difference_diagram``,
which implements the Demšar (2006) layout.  This module is responsible for
translating a :class:`~promptstats.core.paired.FriedmanResult` into the
inputs that function expects (a rank dict and a symmetric p-value DataFrame)
and for wrapping the result in a properly sized, titled Figure.

References
----------
Demšar, J. (2006). Statistical comparisons of classifiers over multiple
data sets. Journal of Machine Learning Research, 7, 1–30.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from promptstats.compare import CompareReport
    from promptstats.core.bundles import AnalysisBundle

from promptstats.core.paired import FriedmanResult, PairwiseMatrix
from promptstats.config import get_alpha_ci


def _sig_matrix(friedman: FriedmanResult) -> pd.DataFrame:
    """Build a symmetric p-value DataFrame from the upper-triangle nemenyi_p.

    Diagonal entries are set to 1.0 (a treatment is not significantly
    different from itself).  ``scikit_posthocs.critical_difference_diagram``
    treats values *below* ``alpha`` as significant.
    """
    labels = list(friedman.avg_ranks.keys())
    mat = pd.DataFrame(1.0, index=labels, columns=labels)
    for (a, b), p in friedman.nemenyi_p.items():
        mat.loc[a, b] = p
        mat.loc[b, a] = p
    return mat


def _sig_matrix_from_rank_bands(
    pairwise: PairwiseMatrix,
    *,
    labels_sorted: list[str],
    alpha: float,
    p_source: Literal["bootstrap", "wilcoxon"],
) -> pd.DataFrame:
    """Build a symmetric significance matrix from CD-style rank bands.

    The matrix is encoded as pseudo p-values for compatibility with
    ``scikit_posthocs.critical_difference_diagram``:
    - 1.0 means non-significant (connected)
    - 0.0 means significant (not connected)
    """
    from promptstats.core.summary import _critical_difference_groups

    groups = _critical_difference_groups(
        pairwise,
        labels_sorted=labels_sorted,
        alpha=alpha,
        p_source=p_source,
    )
    labels = list(labels_sorted)
    mat = pd.DataFrame(0.0, index=labels, columns=labels)
    for label in labels:
        mat.loc[label, label] = 1.0

    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]
                mat.loc[a, b] = 1.0
                mat.loc[b, a] = 1.0

    return mat


def plot_critical_difference(
    result: Union[FriedmanResult, PairwiseMatrix, "CompareReport", "AnalysisBundle"],
    *,
    ranks: Optional[Mapping[str, float]] = None,
    labels_sorted: Optional[list[str]] = None,
    p_source: Literal["bootstrap", "wilcoxon"] = "bootstrap",
    alpha: Optional[float] = None,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    ax: Optional["Axes"] = None,
) -> "Figure":
    """Plot a critical difference diagram from Friedman or pairwise results.

    Uses :func:`scikit_posthocs.critical_difference_diagram` for rendering
    (Demšar 2006 layout: rank axis, left/right labels, crossbars for
    non-significant groups, CD bracket).

    Parameters
    ----------
    result : FriedmanResult | PairwiseMatrix | CompareReport | AnalysisBundle
        Either output of :func:`promptstats.friedman_nemenyi` (Friedman mode)
        or a pairwise results matrix (pairwise mode). You can also pass a
        ``compare_prompts``/``compare_models`` report object; in that case
        ``report.pairwise`` is used automatically.
        In pairwise mode,
        non-significant groups are
        derived from the same logic used in text summaries
        (``_critical_difference_groups``), preferring simultaneous CIs when
        available.
    ranks : mapping[str, float], optional
        Rank values to place items on the CD axis (1 is best). Recommended
        when using ``pairwise`` so the axis reflects expected-rank estimates.
        When omitted, ranks default to sequential values based on
        ``labels_sorted`` (or ``pairwise.labels``).
    labels_sorted : list[str], optional
        Rank order (best to worst) used to compute contiguous rank bands in
        ``pairwise`` mode.
    p_source : {"bootstrap", "wilcoxon"}
        P-value source for rank-band grouping when simultaneous CIs are not
        present in ``pairwise``.
    alpha : float
        Significance threshold for crossbar grouping (default 0.05).
    figsize : tuple[float, float], optional
        Figure size.  Defaults to ``(max(7, k + 3), 3.5)``.
    title : str, optional
        Plot title.  A descriptive default is generated when omitted.
    ax : Axes, optional
        Existing axes to draw into.  A new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if alpha is None:
        alpha = get_alpha_ci()
    # Accept CompareReport/AnalysisBundle by extracting .pairwise.
    report_like = None
    if not isinstance(result, (FriedmanResult, PairwiseMatrix)):
        pairwise_attr = getattr(result, "pairwise", None)
        if isinstance(pairwise_attr, PairwiseMatrix):
            report_like = result
            result = pairwise_attr
        else:
            raise TypeError(
                "result must be a FriedmanResult, PairwiseMatrix, or report with .pairwise."
            )

    try:
        from scikit_posthocs import critical_difference_diagram
    except ImportError as exc:
        raise ImportError(
            "scikit-posthocs is required for plot_critical_difference. "
            "Install it with: pip install scikit-posthocs"
        ) from exc

    if isinstance(result, FriedmanResult):
        friedman = result
        k = friedman.n_templates
        ranks_dict = dict(friedman.avg_ranks)
        sig_mat = _sig_matrix(friedman)
    else:
        pairwise = result

        # If the caller passed a high-level report and omitted ranks, derive
        # a stable best->worst ordering from absolute means.
        if report_like is not None and ranks is None:
            report_labels = getattr(report_like, "labels", None)
            report_stats = getattr(report_like, "entity_stats", None)
            if isinstance(report_labels, list) and isinstance(report_stats, Mapping):
                mean_items: list[tuple[str, float]] = []
                for label in report_labels:
                    stats = report_stats.get(label)
                    mean_val = getattr(stats, "mean", None)
                    if mean_val is None:
                        mean_items = []
                        break
                    mean_items.append((str(label), float(mean_val)))

                if mean_items:
                    mean_items.sort(key=lambda item: (-float(item[1]), str(item[0])))
                    ranks = {
                        label: float(idx + 1)
                        for idx, (label, _) in enumerate(mean_items)
                    }
                    if labels_sorted is None:
                        labels_sorted = [label for label, _ in mean_items]

        if ranks is not None:
            ranks_dict = {str(label): float(rank) for label, rank in ranks.items()}
            rank_sorted_labels = [
                label
                for label, _ in sorted(
                    ranks_dict.items(),
                    key=lambda item: (float(item[1]), str(item[0])),
                )
            ]
        else:
            if labels_sorted is None:
                rank_sorted_labels = list(pairwise.labels)
            else:
                rank_sorted_labels = list(labels_sorted)
            ranks_dict = {label: float(i + 1) for i, label in enumerate(rank_sorted_labels)}

        if labels_sorted is None:
            labels_for_groups = rank_sorted_labels
        else:
            labels_for_groups = list(labels_sorted)

        missing_labels = [label for label in labels_for_groups if label not in ranks_dict]
        if missing_labels:
            raise ValueError(
                "All labels_sorted entries must have rank values in ranks. "
                f"Missing: {missing_labels}"
            )

        k = len(ranks_dict)
        sig_mat = _sig_matrix_from_rank_bands(
            pairwise,
            labels_sorted=labels_for_groups,
            alpha=alpha,
            p_source=p_source,
        )

    # Ensure matrix index/columns match rank labels and order.
    rank_labels = list(ranks_dict.keys())
    sig_mat = sig_mat.reindex(index=rank_labels, columns=rank_labels, fill_value=0.0)

    own_fig = ax is None
    if own_fig:
        if figsize is None:
            figsize = (max(7.0, k * 0.9 + 3.0), 4.0)
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    else:
        fig = ax.get_figure()

    critical_difference_diagram(ranks_dict, sig_mat, ax=ax, alpha=alpha)

    if title is None:
        if isinstance(result, FriedmanResult):
            friedman = result
            title = (
                f"Critical Difference Diagram  ·  "
                f"Friedman χ²({friedman.df}) = {friedman.statistic:.2f},  "
                f"p = {friedman.p_value:.3g}"
            )
        else:
            pairwise = result
            if pairwise.simultaneous_ci_method is not None:
                source = f"CI ({pairwise.simultaneous_ci_method})"
            else:
                source = "p (boot)" if p_source == "bootstrap" else "p (wsr)"
            title = f"Critical Difference Diagram  ·  Pairwise rank bands from {source}"
    ax.set_title(
        title,
        fontsize=10, pad=10, loc="left", fontweight="semibold",
    )

    if own_fig:
        fig.tight_layout()

    return fig
