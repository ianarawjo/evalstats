"""Uncertainty-aware Pareto-front analysis for a primary metric + secondary metric(s).

Backs ``compare(..., secondary=...)``. Unlike a naive Pareto front on point
estimates -- which lets a marginally-lower cost or marginally-higher accuracy
count as "dominates" even when the underlying data can't distinguish the two
entities -- this module resamples both metrics jointly (same shared per-item
bootstrap index per replicate, the same idiom :func:`~evalstats.core.paired.all_pairwise`
uses for its own joint pairwise resampling) so that dominance calls only fire
when the data actually supports them, and so that correlation between the two
metrics (e.g. harder items being both slower and less accurate) is preserved
rather than silently dropped by two independent marginal bootstraps.

The joint replicates are the single source of truth for two different
summaries of the same underlying uncertainty:

* :func:`pareto_bootstrap` -- computes, per bootstrap replicate, which
  entities are on the empirical (point-in-that-replicate) Pareto frontier,
  and which entity dominates which. Tallying across replicates gives both a
  continuous ``P(entity is Pareto-optimal)`` and a bootstrap dominance
  probability for every ordered pair.
* :func:`classify_pareto_status` -- reduces those same tallies to a default,
  calibrated three-state label per entity ("frontier" / "dominated" /
  "ambiguous"), FWER-aware across the N-1 possible dominators of each entity,
  mirroring how :attr:`~evalstats.api.ComparisonResult.unbeaten` reports a
  set rather than a single "winner" for the single-metric case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


@dataclass
class ParetoBootstrapResult:
    """Joint bootstrap dominance tallies for a primary + secondary metric.

    Both metrics are assumed already oriented so that **higher is better**
    (the caller negates a "minimize" metric, e.g. latency, before calling --
    see :func:`orient_higher_is_better`). All comparisons here are therefore
    a single, direction-agnostic ">= on both axes, > on at least one"
    dominance rule.

    Attributes
    ----------
    labels : list[str]
        Entity labels, in the order all other arrays are indexed.
    point_primary : np.ndarray
        Shape ``(N,)``. Point estimate (mean or median) of the primary
        metric per entity, on the original per-item data (not resampled).
    point_secondary : np.ndarray
        Shape ``(N,)``. Point estimate of the secondary metric per entity.
    p_frontier : np.ndarray
        Shape ``(N,)``. Fraction of bootstrap replicates in which entity i
        was on the empirical (that replicate's own point values) Pareto
        frontier -- i.e. not dominated by any other entity in that replicate.
        This *is* ``P(entity is Pareto-optimal)`` -- the opt-in continuous
        output; no covariance-matrix fitting involved, since it's a direct
        tally over the empirical joint bootstrap distribution.
    p_dominated_by : np.ndarray
        Shape ``(N, N)``. ``p_dominated_by[i, j]`` = fraction of replicates
        in which entity j dominates entity i (both metrics at least as
        good, strictly better on at least one). Diagonal is always 0.
    n_bootstrap : int
        Number of bootstrap replicates the tallies above were computed from.
    """

    labels: list[str]
    point_primary: np.ndarray
    point_secondary: np.ndarray
    p_frontier: np.ndarray
    p_dominated_by: np.ndarray
    n_bootstrap: int


def orient_higher_is_better(scores: np.ndarray, direction: Literal["max", "min"]) -> np.ndarray:
    """Return *scores* (or its negation) so that higher is always better.

    :func:`pareto_bootstrap` and :func:`classify_pareto_status` only ever
    reason about ">= on both axes, > on at least one" -- callers with a
    "minimize" metric (e.g. latency, cost) negate it through this function
    first rather than the engine carrying direction-handling logic itself.
    """
    if direction == "max":
        return scores
    if direction == "min":
        return -scores
    raise ValueError(f"direction must be 'max' or 'min', got {direction!r}")


def pareto_bootstrap(
    scores_primary: np.ndarray,
    scores_secondary: np.ndarray,
    labels: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
    *,
    statistic: Literal["mean", "median"] = "mean",
    batch_size: int = 256,
) -> ParetoBootstrapResult:
    """Joint bootstrap Pareto-dominance tallies for two per-item-aligned metrics.

    Draws one shared per-item resample index per replicate (applied to every
    entity and both metrics at once -- the same "shared draw" idiom
    :func:`~evalstats.core.paired._joint_bootstrap_critical_value` uses),
    computes each entity's per-replicate (primary, secondary) point value,
    and tallies (a) which entities are non-dominated in that replicate and
    (b) which entity dominates which. Both metrics must already be oriented
    so higher is better (see :func:`orient_higher_is_better`) and measured on
    the same items (paired by column position), so that the shared resample
    preserves whatever correlation exists between them.

    Parameters
    ----------
    scores_primary, scores_secondary : np.ndarray
        Shape ``(N, M)`` -- one row per entity, one column per item, same
        item alignment for both arrays (column j is the same item in both).
    labels : list[str]
        Entity labels, length N.
    n_bootstrap : int
        Number of bootstrap replicates.
    rng : np.random.Generator
    statistic : {"mean", "median"}
        Per-entity, per-replicate central-tendency statistic.
    batch_size : int
        Replicates processed per chunk, bounding peak memory for the
        ``(N, N, batch)`` pairwise-dominance array (mirrors the chunking
        pattern used elsewhere in this codebase, e.g.
        :func:`~evalstats.core.paired.romano_wolf_stepdown_pvalues`).

    Returns
    -------
    ParetoBootstrapResult
    """
    N, M = scores_primary.shape
    if scores_secondary.shape != scores_primary.shape:
        raise ValueError(
            f"scores_primary and scores_secondary must have the same shape "
            f"(same entities, same items); got {scores_primary.shape} vs "
            f"{scores_secondary.shape}."
        )
    if len(labels) != N:
        raise ValueError(f"labels must have length {N}, got {len(labels)}.")

    def _stat(a: np.ndarray, axis: int) -> np.ndarray:
        return a.mean(axis=axis) if statistic == "mean" else np.median(a, axis=axis)

    point_primary = _stat(scores_primary, axis=1)
    point_secondary = _stat(scores_secondary, axis=1)

    frontier_count = np.zeros(N)
    dominated_count = np.zeros((N, N))  # [i, j] += 1 when j dominates i, this replicate
    eye = np.eye(N, dtype=bool)

    start = 0
    while start < n_bootstrap:
        stop = min(start + batch_size, n_bootstrap)
        b = stop - start
        idx = rng.integers(0, M, size=(b, M))  # shared across every entity and both metrics

        r1 = scores_primary[:, idx]    # (N, b, M)
        r2 = scores_secondary[:, idx]  # (N, b, M)
        bm1 = _stat(r1, axis=2)  # (N, b)
        bm2 = _stat(r2, axis=2)  # (N, b)

        m1_i, m1_j = bm1[:, None, :], bm1[None, :, :]  # (N,1,b), (1,N,b)
        m2_i, m2_j = bm2[:, None, :], bm2[None, :, :]
        ge_both = (m1_j >= m1_i) & (m2_j >= m2_i)
        gt_either = (m1_j > m1_i) | (m2_j > m2_i)
        j_dominates_i = ge_both & gt_either  # (N, N, b)
        j_dominates_i &= ~eye[:, :, None]    # an entity cannot dominate itself

        dominated_count += j_dominates_i.sum(axis=2)
        i_is_dominated = j_dominates_i.any(axis=1)  # (N, b)
        frontier_count += (~i_is_dominated).sum(axis=1)

        start = stop

    return ParetoBootstrapResult(
        labels=list(labels),
        point_primary=point_primary,
        point_secondary=point_secondary,
        p_frontier=frontier_count / n_bootstrap,
        p_dominated_by=dominated_count / n_bootstrap,
        n_bootstrap=n_bootstrap,
    )


@dataclass
class ParetoStatus:
    """Default three-state Pareto classification for one entity.

    Attributes
    ----------
    label : str
        Entity label.
    status : {"frontier", "dominated", "ambiguous"}
        - ``"frontier"``: not confidently dominated by anyone at *alpha*
          (Bonferroni-adjusted across the N-1 possible dominators), *and*
          the entity's own point estimate is itself on the raw
          point-estimate Pareto frontier -- the calibrated, "safe" verdict.
        - ``"dominated"``: confidently dominated by at least one specific
          entity at *alpha* -- ``dominated_by`` lists them.
        - ``"ambiguous"``: not confidently dominated by anyone, but the
          entity's own point estimate is beaten on both axes by some other
          entity's point estimate too -- there just isn't enough evidence to
          confirm that dominance statistically. This is the "can't rule it
          out either way" case a naive point-estimate-only Pareto front
          would silently misreport as a clean frontier member.
          ``ambiguous_vs`` lists the entities responsible.
    dominated_by : list[str]
        Entities that confidently dominate this one (empty unless
        ``status == "dominated"``).
    ambiguous_vs : list[str]
        Entities whose point estimate dominates this one's, without enough
        evidence to confirm it (empty unless ``status == "ambiguous"``).
    """

    label: str
    status: Literal["frontier", "dominated", "ambiguous"]
    dominated_by: list[str] = field(default_factory=list)
    ambiguous_vs: list[str] = field(default_factory=list)


def classify_pareto_status(
    result: ParetoBootstrapResult,
    *,
    alpha: float = 0.05,
) -> dict[str, ParetoStatus]:
    """Reduce :class:`ParetoBootstrapResult` to a default three-state label per entity.

    Dominance calls are Bonferroni-adjusted across the N-1 possible
    dominators of each entity (mirrors the FWER-control philosophy already
    used for pairwise significance elsewhere in evalstats): entity i is
    "dominated" only when some j clears ``p_dominated_by[i, j] >= 1 - alpha
    / (N - 1)``, not just the raw ``1 - alpha`` -- checking N-1 possible
    dominators per entity is itself a multiple-comparisons problem.

    Parameters
    ----------
    result : ParetoBootstrapResult
        Output of :func:`pareto_bootstrap`.
    alpha : float
        Significance level (e.g. 0.05 -> 95% confidence a claimed
        domination is real).

    Returns
    -------
    dict[str, ParetoStatus]
        Keyed by entity label.
    """
    labels = result.labels
    N = len(labels)
    threshold = 1.0 - (alpha / max(N - 1, 1))

    # Raw point-estimate Pareto frontier (no uncertainty) -- used only to
    # distinguish "frontier" from "ambiguous" once an entity isn't
    # confidently dominated by anyone.
    p1, p2 = result.point_primary, result.point_secondary
    point_dominated_by: dict[int, list[int]] = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dominates = (p1[j] >= p1[i]) and (p2[j] >= p2[i]) and (p1[j] > p1[i] or p2[j] > p2[i])
            if dominates:
                point_dominated_by[i].append(j)

    statuses: dict[str, ParetoStatus] = {}
    for i, label in enumerate(labels):
        confident_dominators = [
            labels[j] for j in range(N) if j != i and result.p_dominated_by[i, j] >= threshold
        ]
        if confident_dominators:
            statuses[label] = ParetoStatus(label=label, status="dominated", dominated_by=confident_dominators)
        elif point_dominated_by[i]:
            statuses[label] = ParetoStatus(
                label=label, status="ambiguous",
                ambiguous_vs=[labels[j] for j in point_dominated_by[i]],
            )
        else:
            statuses[label] = ParetoStatus(label=label, status="frontier")
    return statuses
