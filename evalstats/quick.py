"""Quick primitives for users who don't need compare()'s full comparative
report -- just a trustworthy point estimate (or a few other common building
blocks) as plain data, ready to hand off to their own plotting library or
downstream code.

These reuse the exact same auto method-selection and calibration machinery
compare() uses internally (see core.router.resolve_auto_robustness_method),
so a number returned here and the equivalent number inside a compare()
report are computed identically -- there is no separate, potentially-
drifting "lite" calibration path.

Every result type here follows the same output convention: attribute
access for the common fields, ``.to_dict()`` for a JSON-friendly plain
dict, and (for the batch-capable results) ``.to_frame()`` for a pandas
DataFrame -- mirroring how ``ComparisonResult`` already offers ``.to_dict()``
/``.to_frame()`` rather than committing to one output shape.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Optional, Union

import numpy as np
import pandas as pd

from .config import get_alpha_ci
from .core.router import resolve_auto_robustness_method
from .core.variance import robustness_metrics, seed_variance_decomposition, SeedVarianceResult


def _clean_1d(arr: np.ndarray, *, label: str, stacklevel: int) -> np.ndarray:
    """Drop NaN entries from a 1-D score array, warning if any were found,
    and raising a clear, attributed error if nothing valid remains.

    Real-world data has gaps -- a failed API call, a skipped item -- and
    ``compare()`` handles that by rejecting NaN outright with a hard error.
    That's the right call for a full comparative report, but it would
    defeat the purpose of a quick primitive: forcing every caller to
    hand-filter NaN themselves before getting a single number back. Instead,
    drop it and say so (not silently -- an unflagged NaN CI, which is what
    happens without this filtering, is worse than either option).
    """
    is_nan = np.isnan(arr)
    n_missing = int(np.sum(is_nan))
    if n_missing > 0:
        arr = arr[~is_nan]
        warnings.warn(
            f"{label}: dropped {n_missing} NaN (missing) value(s) out of "
            f"{n_missing + arr.size}; computed from the remaining {arr.size}.",
            UserWarning,
            stacklevel=stacklevel,
        )
    if arr.size == 0:
        raise ValueError(f"{label} has no valid (non-NaN) scores.")
    return arr


# ---------------------------------------------------------------------------
# mean_ci
# ---------------------------------------------------------------------------

class MeanCI(NamedTuple):
    """Calibrated mean + confidence interval for a single array of scores.

    A plain :class:`NamedTuple` so it works both ways: unpack positionally
    (``mean, ci_low, ci_high, n, method = es.mean_ci(scores)``) or use
    attribute access (``result.mean``). Call :meth:`to_dict` for a plain
    dict.

    Attributes
    ----------
    mean : float
        Point estimate (sample mean).
    ci_low, ci_high : float
        Bounds of the calibrated confidence interval.
    n : int
        Number of (non-NaN) scores the estimate is based on.
    method : str
        The CI method evalstats auto-selected (e.g. ``"logit_t"``,
        ``"wilson"``, ``"smooth_bootstrap"``) -- see
        :func:`~evalstats.core.router.resolve_auto_robustness_method` for
        the full routing table.
    """

    mean: float
    ci_low: float
    ci_high: float
    n: int
    method: str

    def to_dict(self) -> dict:
        """Return a plain, JSON-friendly dict."""
        return self._asdict()


def mean_ci(
    scores,
    *,
    alpha: Optional[float] = None,
    n_bootstrap: int = 10_000,
    score_range: Optional[tuple[float, float]] = None,
    rng=None,
) -> MeanCI:
    """Calibrated mean + confidence interval for a single array of scores.

    Auto-detects the data kind (binary / bounded [0, 1] / unbounded) and the
    sample size, then picks the same CI method ``compare()`` would use for
    this data -- Wilson for binary, logit-t for bounded continuous/Likert
    data, a bootstrap-t fallback for small unbounded samples, etc. No
    ``load_from()``, no factors, no comparison -- just the one number a lot
    of users actually want.

    Parameters
    ----------
    scores : array-like
        A 1-D array (or anything ``np.asarray`` accepts) of per-item scores
        for a single entity.
    alpha : float, optional
        Significance level. Defaults to :func:`evalstats.get_alpha_ci`'s
        current value (0.05 unless changed via :func:`evalstats.set_alpha_ci`).
    n_bootstrap : int
        Bootstrap resamples for the CI, when the auto-selected method is
        bootstrap-based (default 10,000, matching ``analyze()``'s default).
    score_range : (float, float), optional
        Explicit ``(min, max)`` bounds for the metric (e.g. ``(1, 5)`` for a
        Likert scale). Only used when the auto-selected method needs
        bounds-aware rescaling (``logit_t``); inferred from the data when
        not given and possible -- see
        :func:`~evalstats.core.router.resolve_auto_robustness_method`.
    rng : int, np.random.Generator, or None
        Seed or generator for reproducibility.

    Returns
    -------
    MeanCI

    Examples
    --------
    >>> import evalstats as es
    >>> result = es.mean_ci(accuracy_scores)
    >>> result.mean, result.ci_low, result.ci_high
    >>> mean, lo, hi, n, method = result  # positional unpack also works
    """
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"scores must be a 1-D array; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("scores must not be empty.")
    arr = _clean_1d(arr, label="scores", stacklevel=3)

    if alpha is None:
        alpha = get_alpha_ci()
    rng = np.random.default_rng(rng)

    scores_2d = arr.reshape(1, -1)
    _, robustness_method, resolved_score_range, _ = resolve_auto_robustness_method(
        scores_2d, score_range=score_range, stacklevel=3,
    )
    rob = robustness_metrics(
        scores_2d, ["_"],
        n_bootstrap=n_bootstrap, rng=rng, alpha=alpha,
        statistic="mean", marginal_method=robustness_method,
        multi_ci=False, score_range=resolved_score_range,
    )
    return MeanCI(
        mean=float(rob.mean[0]),
        ci_low=float(rob.ci_low[0]) if rob.ci_low is not None else float("nan"),
        ci_high=float(rob.ci_high[0]) if rob.ci_high is not None else float("nan"),
        n=int(arr.size),
        method=robustness_method,
    )


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

_SUMMARY_ROW_FIELDS = (
    "mean", "median", "std", "cv", "iqr", "cvar_10",
    "p10", "p25", "p50", "p75", "p90", "ci_low", "ci_high", "n", "method",
)


@dataclass
class GroupSummary:
    """Descriptive statistics + calibrated CI, one row per group.

    Returned by :func:`summarize`. Same field set as the "--- Robustness
    ---" table ``compare()`` prints, plus the calibrated ``ci_low``/
    ``ci_high`` compare()'s Mean Performance section shows separately --
    bundled into one table here since there's no other section to split it
    across.

    Each group's row is computed independently (its own auto-detected data
    kind, N, and CI method) rather than pooled, so groups of different
    sizes or types can be summarized in the same call.
    """

    labels: list[str]
    mean: np.ndarray
    median: np.ndarray
    std: np.ndarray
    cv: np.ndarray
    iqr: np.ndarray
    cvar_10: np.ndarray
    p10: np.ndarray
    p25: np.ndarray
    p50: np.ndarray
    p75: np.ndarray
    p90: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n: np.ndarray
    method: list[str]
    _single_ungrouped: bool = False

    def _row_dict(self, i: int) -> dict:
        return {
            "mean": float(self.mean[i]),
            "median": float(self.median[i]),
            "std": float(self.std[i]),
            "cv": float(self.cv[i]),
            "iqr": float(self.iqr[i]),
            "cvar_10": float(self.cvar_10[i]),
            "p10": float(self.p10[i]),
            "p25": float(self.p25[i]),
            "p50": float(self.p50[i]),
            "p75": float(self.p75[i]),
            "p90": float(self.p90[i]),
            "ci_low": float(self.ci_low[i]),
            "ci_high": float(self.ci_high[i]),
            "n": int(self.n[i]),
            "method": self.method[i],
        }

    def to_dict(self) -> dict:
        """Plain, JSON-friendly dict.

        A flat ``{"mean": ..., "ci_low": ..., ...}`` dict when
        :func:`summarize` was called on a single bare array; a
        ``{label: {...}}`` nested dict, one entry per group, otherwise.
        """
        if self._single_ungrouped:
            return self._row_dict(0)
        return {label: self._row_dict(i) for i, label in enumerate(self.labels)}

    def to_frame(self) -> pd.DataFrame:
        """Return one row per group as a pandas DataFrame, indexed by label."""
        data = {
            field: getattr(self, field)
            for field in _SUMMARY_ROW_FIELDS
        }
        return pd.DataFrame(data, index=pd.Index(self.labels, name="group"))


def summarize(
    scores: Union[np.ndarray, dict, pd.DataFrame],
    *,
    group_col: Optional[str] = None,
    value_col: Optional[str] = None,
    statistic: Literal["mean", "median"] = "mean",
    alpha: Optional[float] = None,
    n_bootstrap: int = 10_000,
    score_range: Optional[tuple[float, float]] = None,
    rng=None,
) -> GroupSummary:
    """Descriptive statistics + calibrated CI for one or more groups.

    Accepts whatever shape of data you already have:

    * A single 1-D array -- one group.
    * A ``{label: array}`` dict -- one row per key. Arrays don't need to be
      the same length.
    * A long-format DataFrame plus ``group_col``/``value_col`` -- one row
      per distinct value of ``group_col``.

    Each group is auto-calibrated independently (own data-kind/N detection,
    own CI method -- see :func:`mean_ci`), so this does *not* require a
    ``compare()``-style rectangular design; groups of different sizes or
    even different data kinds (e.g. one binary, one continuous) are fine.
    This is a descriptive summary only -- no significance testing or
    ranking between groups; use ``compare()`` for that.

    Parameters
    ----------
    scores : array-like, dict, or DataFrame
        See above.
    group_col, value_col : str, optional
        Required (and only used) when ``scores`` is a DataFrame.
    statistic : {"mean", "median"}
        Central-tendency statistic the CI is built around (default "mean").
        The ``mean``/``median`` columns of the result are always both
        reported regardless of this choice; it only affects ``ci_low``/
        ``ci_high``.
    alpha, n_bootstrap, score_range, rng
        See :func:`mean_ci`.

    Returns
    -------
    GroupSummary

    Examples
    --------
    >>> import evalstats as es
    >>> es.summarize({"gpt-4o": acc_gpt4o, "claude": acc_claude}).to_frame()
    >>> es.summarize(df, group_col="model", value_col="accuracy").to_frame()
    """
    single_ungrouped = False
    if isinstance(scores, pd.DataFrame):
        if group_col is None or value_col is None:
            raise ValueError(
                "summarize() on a DataFrame requires group_col and "
                "value_col, e.g. summarize(df, group_col='model', "
                "value_col='accuracy')."
            )
        if group_col not in scores.columns:
            raise ValueError(
                f"group_col '{group_col}' not found in DataFrame columns: "
                f"{list(scores.columns)}"
            )
        if value_col not in scores.columns:
            raise ValueError(
                f"value_col '{value_col}' not found in DataFrame columns: "
                f"{list(scores.columns)}"
            )
        groups = scores.groupby(group_col, sort=False)[value_col]
        labels = [str(k) for k in groups.groups.keys()]
        arrays = [
            groups.get_group(k).to_numpy(dtype=float)
            for k in groups.groups.keys()
        ]
    elif isinstance(scores, dict):
        if len(scores) == 0:
            raise ValueError("scores dict must not be empty.")
        labels = [str(k) for k in scores.keys()]
        arrays = [np.asarray(v, dtype=float) for v in scores.values()]
    else:
        arr = np.asarray(scores, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                "scores must be a 1-D array, a {label: array} dict, or a "
                f"DataFrame (with group_col/value_col); got shape {arr.shape}."
            )
        labels = ["value"]
        arrays = [arr]
        single_ungrouped = True

    for lbl, a in zip(labels, arrays):
        if a.size == 0:
            raise ValueError(f"Group '{lbl}' has no scores.")
    arrays = [
        _clean_1d(a, label=f"group '{lbl}'" if not single_ungrouped else "scores", stacklevel=3)
        for lbl, a in zip(labels, arrays)
    ]

    if alpha is None:
        alpha = get_alpha_ci()
    rng = np.random.default_rng(rng)

    mean = np.empty(len(labels))
    median = np.empty(len(labels))
    std = np.empty(len(labels))
    cv = np.empty(len(labels))
    iqr = np.empty(len(labels))
    cvar_10 = np.empty(len(labels))
    p10 = np.empty(len(labels))
    p25 = np.empty(len(labels))
    p50 = np.empty(len(labels))
    p75 = np.empty(len(labels))
    p90 = np.empty(len(labels))
    ci_low = np.empty(len(labels))
    ci_high = np.empty(len(labels))
    n = np.empty(len(labels), dtype=int)
    method: list[str] = []

    # Each group gets its own auto-detected method (own data kind, own N)
    # rather than pooling into one call -- see GroupSummary's docstring for
    # why groups don't need to share a rectangular design here.
    for i, a in enumerate(arrays):
        a_2d = a.reshape(1, -1)
        _, robustness_method, resolved_score_range, _ = resolve_auto_robustness_method(
            a_2d, score_range=score_range, stacklevel=3,
        )
        rob = robustness_metrics(
            a_2d, ["_"],
            n_bootstrap=n_bootstrap, rng=rng, alpha=alpha,
            statistic=statistic, marginal_method=robustness_method,
            multi_ci=False, score_range=resolved_score_range,
        )
        mean[i] = rob.mean[0]
        median[i] = rob.median[0]
        std[i] = rob.std[0]
        cv[i] = rob.cv[0]
        iqr[i] = rob.iqr[0]
        cvar_10[i] = rob.cvar_10[0]
        p10[i] = rob.percentiles[10][0]
        p25[i] = rob.percentiles[25][0]
        p50[i] = rob.percentiles[50][0]
        p75[i] = rob.percentiles[75][0]
        p90[i] = rob.percentiles[90][0]
        ci_low[i] = rob.ci_low[0] if rob.ci_low is not None else np.nan
        ci_high[i] = rob.ci_high[0] if rob.ci_high is not None else np.nan
        n[i] = a.size  # a is already NaN-cleaned above
        method.append(robustness_method)

    return GroupSummary(
        labels=labels, mean=mean, median=median, std=std, cv=cv, iqr=iqr,
        cvar_10=cvar_10, p10=p10, p25=p25, p50=p50, p75=p75, p90=p90,
        ci_low=ci_low, ci_high=ci_high, n=n, method=method,
        _single_ungrouped=single_ungrouped,
    )


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------

@dataclass
class StabilityResult:
    """Multi-run reliability metrics for one or more configs.

    Returned by :func:`stability`. See
    :class:`~evalstats.core.variance.SeedVarianceResult` for the underlying
    ``instability``/``icc`` decomposition this wraps.

    Attributes
    ----------
    labels : list[str]
        Config labels.
    instability : np.ndarray
        Mean within-item run-to-run standard deviation, in score-scale
        units -- "on average, how many points does the score move between
        runs for the same item?". Lower is more stable.
    icc : np.ndarray
        Intraclass correlation: of the variation across items, the fraction
        that's genuine item-level signal rather than run-to-run noise
        (bounded [0, 1], higher is more reliable).
    n_runs : np.ndarray
        Number of (non-padded) runs each config was actually evaluated
        over. Per-config, not a single shared count -- configs are allowed
        to have different run counts (see :func:`stability`).
    label_text : list[str]
        Plain-language interpretation of ``instability`` per config (e.g.
        "mostly stable across runs"), matching the wording ``compare()``'s
        printed summary uses for the same metric.
    """

    labels: list[str]
    instability: np.ndarray
    icc: np.ndarray
    n_runs: np.ndarray
    label_text: list[str]
    _seed_variance: Optional[SeedVarianceResult] = None  # underlying decomposition; powers summary()'s noise strip

    def summary(self, item_singular: str = "config") -> None:
        """Print the same reliability breakdown ``compare()`` shows in the
        terminal for multi-run data: a per-input noise strip alongside
        seed/input/total std, instability, ICC, and a plain-language
        verdict -- for when reliability is all you want to check, without
        a full multi-model comparison.
        """
        from .core.summary import _print_seed_variance
        _print_seed_variance(self._seed_variance, item_singular=item_singular)

    def _row_dict(self, i: int) -> dict:
        return {
            "instability": float(self.instability[i]),
            "icc": float(self.icc[i]) if not np.isnan(self.icc[i]) else None,
            "n_runs": int(self.n_runs[i]),
            "interpretation": self.label_text[i],
        }

    def to_dict(self) -> dict:
        """Plain dict: flat for a single config, ``{label: {...}}`` for several."""
        if len(self.labels) == 1:
            return self._row_dict(0)
        return {label: self._row_dict(i) for i, label in enumerate(self.labels)}

    def to_frame(self) -> pd.DataFrame:
        """One row per config as a pandas DataFrame, indexed by label."""
        return pd.DataFrame(
            {
                "instability": self.instability,
                "icc": self.icc,
                "n_runs": self.n_runs,
                "interpretation": self.label_text,
            },
            index=pd.Index(self.labels, name="config"),
        )


def _stability_core(labels: list[str], arrays: list[np.ndarray], *, warn_orientation: bool) -> StabilityResult:
    """Shared core behind both stability() input forms: validates shapes,
    pads a ragged run axis with NaN, and runs the seed-variance
    decomposition. ``warn_orientation`` is only True for the raw-array/dict
    form -- the DataFrame form has no orientation to get wrong (each row
    names its own run and item explicitly), so it's skipped there.
    """
    for lbl, a in zip(labels, arrays):
        if a.ndim != 2:
            raise ValueError(f"runs['{lbl}'] must be 2-D (K runs x M items); got shape {a.shape}.")

    m_values = {a.shape[1] for a in arrays}
    if len(m_values) != 1:
        raise ValueError(
            "All configs must be evaluated on the same number of items (M); "
            f"got M values {sorted(m_values)} across configs "
            f"{dict(zip(labels, (a.shape for a in arrays)))}."
        )
    m_items = m_values.pop()

    if any(a.shape[0] < 3 for a in arrays):
        offender = labels[[a.shape[0] for a in arrays].index(min(a.shape[0] for a in arrays))]
        raise ValueError(
            f"Seed-variance decomposition requires >= 3 runs per config; "
            f"config '{offender}' has {min(a.shape[0] for a in arrays)}."
        )

    if warn_orientation:
        # A (K runs, M items) array with K > M is unusual for real eval data
        # (far more items than repeated runs, typically) -- a much more
        # common mistake is passing an (M, K) array straight out of
        # df.pivot(index='item', columns='run') without transposing, which
        # silently swaps which axis is "runs" and which is "items" with no
        # error, producing a plausible-looking but wrong instability/icc
        # (confirmed: an (items, runs) mixup can flip "very stable" into
        # "near-random" on the same data). This heuristic won't catch every
        # case (K can legitimately exceed M for a heavily-repeated small
        # item set) -- prefer the DataFrame form below, which has no
        # orientation ambiguity to get wrong in the first place.
        for lbl, a in zip(labels, arrays):
            if a.shape[0] > a.shape[1]:
                warnings.warn(
                    f"runs['{lbl}'] has more rows ({a.shape[0]}) than columns "
                    f"({a.shape[1]}) -- stability() expects (K runs, M items), "
                    "and most real eval data has far more items than repeated "
                    "runs. If this came from a pivot table shaped (items, "
                    "runs), you likely need to transpose it (.T) before "
                    "calling stability(), or use the DataFrame form "
                    "(stability(df, factor_col=..., run_col=..., item_col=..., "
                    "metric_col=...)) instead, which has no orientation to get "
                    "wrong.",
                    UserWarning,
                    stacklevel=4,
                )

    # Different configs may have different K (run count); pad the run axis
    # with NaN -- seed_variance_decomposition's internal nanmean/nanvar
    # handle that safely (it's a closed-form ANOVA-style computation, not a
    # resampling procedure, so NaN-tolerant reductions are exact here).
    max_k = max(a.shape[0] for a in arrays)
    scores_3d = np.full((len(arrays), m_items, max_k), np.nan)
    for i, a in enumerate(arrays):
        scores_3d[i, :, : a.shape[0]] = a.T  # (K, M) -> (M, K)

    from .core.summary import _instability_label

    sv = seed_variance_decomposition(scores_3d, labels)
    actual_n_runs = np.array([a.shape[0] for a in arrays])
    return StabilityResult(
        labels=labels,
        instability=sv.instability,
        icc=sv.icc,
        n_runs=actual_n_runs,
        label_text=[_instability_label(float(v)) for v in sv.instability],
        _seed_variance=sv,
    )


def stability(
    runs: Union[np.ndarray, dict, pd.DataFrame],
    *,
    labels: Optional[list[str]] = None,
    factor_col: Optional[str] = None,
    run_col: Optional[str] = None,
    item_col: Optional[str] = None,
    metric_col: Optional[str] = None,
) -> StabilityResult:
    """Multi-run reliability: how much does a config's score move across
    repeated runs on the same items?

    Standalone version of the seed-instability decomposition ``compare()``
    shows for multi-run (seeded) benchmarks -- for deciding "is this
    configuration reliable enough to ship" without needing a full
    multi-model comparison.

    Three input forms:

    * A long-format DataFrame plus ``factor_col``/``run_col``/``item_col``/
      ``metric_col`` -- **recommended**, since each row names its own run
      and item explicitly, there's no axis-orientation to get wrong.
      Naming mirrors ``compare()``'s ``factors=``/``metric=``.
    * A single config's repeated-run scores as a 2-D array of shape
      ``(K, M)`` (K runs, M items -- note this is *not* what
      ``df.pivot(index='item', columns='run')`` gives you; that needs a
      ``.T`` first, or use the DataFrame form directly).
    * A ``{label: (K, M) array}`` dict of several configs.

    Configs must share the same M (same item set); K (number of runs,
    >= 3) can differ per config.

    Parameters
    ----------
    runs : array-like, dict, or DataFrame
        See above.
    labels : list[str], optional
        Override labels when ``runs`` is a single array (default
        ``["value"]``). Ignored for the dict/DataFrame forms.
    factor_col, run_col, item_col, metric_col : str, optional
        Required (and only used) when ``runs`` is a DataFrame.

    Returns
    -------
    StabilityResult

    Examples
    --------
    >>> import evalstats as es
    >>> es.stability(df, factor_col="config", run_col="run",
    ...              item_col="item", metric_col="score")
    >>> es.stability(rag_config_a_runs)  # shape (5, 200): 5 runs, 200 items
    >>> es.stability({"config_a": runs_a, "config_b": runs_b}).to_frame()
    """
    if isinstance(runs, pd.DataFrame):
        missing = [
            name for name, col in [
                ("factor_col", factor_col), ("run_col", run_col),
                ("item_col", item_col), ("metric_col", metric_col),
            ] if col is None
        ]
        if missing:
            raise ValueError(
                "stability() on a DataFrame requires factor_col, run_col, "
                "item_col, and metric_col; missing: " + ", ".join(missing)
            )
        for name, col in [
            ("factor_col", factor_col), ("run_col", run_col),
            ("item_col", item_col), ("metric_col", metric_col),
        ]:
            if col not in runs.columns:
                raise ValueError(f"{name} '{col}' not found in DataFrame columns: {list(runs.columns)}")

        input_labels: list[str] = []
        arrays: list[np.ndarray] = []
        item_order = None
        for config, group in runs.groupby(factor_col, sort=False):
            pivot = group.pivot(index=run_col, columns=item_col, values=metric_col)
            if item_order is None:
                item_order = list(pivot.columns)
            elif set(pivot.columns) != set(item_order):
                raise ValueError(
                    f"config '{config}' was scored on a different set of items "
                    "than the others -- stability() requires every config to "
                    "share the same item set."
                )
            input_labels.append(str(config))
            arrays.append(pivot.reindex(columns=item_order).to_numpy(dtype=float))
        return _stability_core(input_labels, arrays, warn_orientation=False)

    if isinstance(runs, dict):
        if len(runs) == 0:
            raise ValueError("runs dict must not be empty.")
        input_labels = [str(k) for k in runs.keys()]
        arrays = [np.asarray(v, dtype=float) for v in runs.values()]
    else:
        arr = np.asarray(runs, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                "runs must be a 2-D array (K runs x M items), a "
                "{label: array} dict of such arrays, or a DataFrame (with "
                f"factor_col/run_col/item_col/metric_col); got shape {arr.shape}."
            )
        input_labels = list(labels) if labels is not None else ["value"]
        if len(input_labels) != 1:
            raise ValueError(
                f"A single runs array takes at most one label; got {len(input_labels)}."
            )
        arrays = [arr]

    return _stability_core(input_labels, arrays, warn_orientation=True)


# ---------------------------------------------------------------------------
# tradeoff
# ---------------------------------------------------------------------------

@dataclass
class TradeoffResult:
    """Uncertainty-aware Pareto trade-off between a primary and a secondary metric.

    Returned by :func:`tradeoff`. Wraps the same joint-bootstrap dominance
    engine ``compare(secondary_metric=...)`` uses internally (see
    :mod:`evalstats.core.pareto`) -- for when the trade-off itself is all
    you want to check, without a full ``compare()`` comparison.

    Attributes
    ----------
    labels : list[str]
        Config labels.
    primary_metric, secondary_metric : str
        Column names of the two metrics being traded off. The primary
        metric is always assumed "higher is better"; ``direction`` says
        which way the secondary metric goes.
    direction : {"min", "max"}
        Whether a lower or higher secondary metric value is better.
    status : dict[str, str]
        Per-config Pareto classification, one of "frontier" (calibrated
        best-trade-off set), "dominated" (confidently beaten on both axes
        by some other config), or "ambiguous" (point estimate looks
        dominated, but there isn't enough evidence to confirm it) -- see
        :class:`~evalstats.core.pareto.ParetoStatus`.
    frontier_probability : dict[str, float]
        Per-config ``P(Pareto-optimal)``: fraction of joint bootstrap
        replicates in which the config wasn't dominated on both axes.
    """

    labels: list[str]
    primary_metric: str
    secondary_metric: str
    direction: Literal["min", "max"]
    status: dict[str, str]
    frontier_probability: dict[str, float]
    _pareto: dict = field(default_factory=dict)  # powers summary()/plot()/to_dict()/to_frame()

    def summary(self, *, show_rank_probabilities: bool = False) -> None:
        """Print the same Pareto Front breakdown ``compare()``'s ``summary()``
        shows for ``secondary_metric=`` -- the ASCII scatter, each entity's status
        and calibrated mean + CI on both metrics, and (optionally) the
        bootstrap ``P(Pareto-optimal)`` bar chart.
        """
        from .core.summary import _print_pareto_section
        _print_pareto_section(
            self._pareto, metric=self.primary_metric,
            show_rank_probabilities=show_rank_probabilities,
        )

    def plot(self, **kwargs):
        """Uncertainty-aware Pareto-front scatter (matplotlib).

        See :func:`~evalstats.vis.pareto.plot_pareto_tradeoff` for accepted
        keyword arguments.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .vis.pareto import plot_pareto_tradeoff
        return plot_pareto_tradeoff(self._pareto, metric=self.primary_metric, **kwargs)

    def to_dict(self) -> dict:
        """Plain, JSON-friendly dict: ``{label: {status, dominated_by,
        ambiguous_vs, p_pareto_optimal, primary: {...}, secondary: {...}}}``.
        """
        primary_rob = self._pareto["primary_robustness"]
        secondary_rob = self._pareto["secondary_robustness"]
        statuses = self._pareto["statuses"]
        p_idx = {l: i for i, l in enumerate(primary_rob.labels)}
        s_idx = {l: i for i, l in enumerate(secondary_rob.labels)}
        out: dict[str, dict] = {}
        for label in self.labels:
            st = statuses[label]
            pi, si = p_idx[label], s_idx[label]
            out[label] = {
                "status": st.status,
                "dominated_by": list(st.dominated_by),
                "ambiguous_vs": list(st.ambiguous_vs),
                "p_pareto_optimal": float(self.frontier_probability[label]),
                "primary": {
                    "mean": float(primary_rob.mean[pi]),
                    "ci_low": float(primary_rob.ci_low[pi]) if primary_rob.ci_low is not None else None,
                    "ci_high": float(primary_rob.ci_high[pi]) if primary_rob.ci_high is not None else None,
                },
                "secondary": {
                    "mean": float(secondary_rob.mean[si]),
                    "ci_low": float(secondary_rob.ci_low[si]) if secondary_rob.ci_low is not None else None,
                    "ci_high": float(secondary_rob.ci_high[si]) if secondary_rob.ci_high is not None else None,
                },
            }
        return out

    def to_frame(self) -> pd.DataFrame:
        """One row per config as a pandas DataFrame, indexed by label."""
        d = self.to_dict()
        rows = []
        for label in self.labels:
            e = d[label]
            rows.append({
                "status": e["status"],
                "p_pareto_optimal": e["p_pareto_optimal"],
                f"{self.primary_metric}_mean": e["primary"]["mean"],
                f"{self.primary_metric}_ci_low": e["primary"]["ci_low"],
                f"{self.primary_metric}_ci_high": e["primary"]["ci_high"],
                f"{self.secondary_metric}_mean": e["secondary"]["mean"],
                f"{self.secondary_metric}_ci_low": e["secondary"]["ci_low"],
                f"{self.secondary_metric}_ci_high": e["secondary"]["ci_high"],
                "dominated_by": ", ".join(e["dominated_by"]),
                "ambiguous_vs": ", ".join(e["ambiguous_vs"]),
            })
        return pd.DataFrame(rows, index=pd.Index(self.labels, name="config"))


def tradeoff(
    df: pd.DataFrame,
    *,
    factor_col: str,
    item_col: str,
    primary_metric: str,
    secondary_metric: dict[str, Literal["min", "max"]],
    alpha: Optional[float] = None,
    n_bootstrap: int = 10_000,
    rng=None,
) -> TradeoffResult:
    """Uncertainty-aware Pareto trade-off between a primary and a secondary metric.

    Standalone version of the joint-bootstrap Pareto-front analysis
    ``compare(..., secondary_metric=...)`` runs internally -- for when the
    trade-off itself (e.g. "which prompt gives the best accuracy-for-cost")
    is all you want to check, without a full comparative report. Unlike a
    naive Pareto front on point estimates alone -- which calls a config
    "dominated" any time another's mean beats it on both axes, even when
    the underlying data can't actually support that claim -- this jointly
    resamples both metrics together (a shared per-item bootstrap draw, so
    correlation between the metrics is preserved) and only calls a config
    "dominated" when the data backs it up; a merely-point-estimate-losing
    config is reported "ambiguous" instead.

    Requires a complete design: every config scored on every item, for
    both metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with one row per (config, item).
    factor_col, item_col : str
        Column names identifying the entity being compared (e.g. a prompt
        template or model) and the benchmark item. ``factor_col`` mirrors
        ``compare()``'s ``factors=``.
    primary_metric : str
        Column name of the primary metric (e.g. accuracy). Always assumed
        "higher is better". Mirrors ``compare()``'s ``metric=``.
    secondary_metric : dict[str, {"min", "max"}]
        Exactly one secondary metric column mapped to its direction, e.g.
        ``{"latency_s": "min"}`` or ``{"quality_score": "max"}``.
    alpha : float, optional
        Significance level for both metrics' marginal CIs and for the
        FWER-adjusted dominance calls (default: ``get_alpha_ci()``, 0.05).
    n_bootstrap : int
        Number of joint bootstrap replicates.
    rng : optional
        Seed or ``np.random.Generator``.

    Returns
    -------
    TradeoffResult

    Examples
    --------
    >>> import evalstats as es
    >>> result = es.tradeoff(
    ...     df, factor_col="prompt", item_col="item",
    ...     primary_metric="accuracy", secondary_metric={"cost_usd": "min"},
    ... )
    >>> result.status
    >>> result.plot()
    """
    if not isinstance(secondary_metric, dict) or len(secondary_metric) != 1:
        raise ValueError(
            "secondary_metric must be a single-entry dict mapping a metric column "
            "name to 'min' or 'max', e.g. secondary_metric={'latency_s': 'min'}."
        )
    (secondary_col, direction), = secondary_metric.items()
    if direction not in ("min", "max"):
        raise ValueError(f"secondary_metric={{'{secondary_col}': {direction!r}}} -- direction must be 'min' or 'max'.")
    for name, col in [
        ("factor_col", factor_col), ("item_col", item_col),
        ("primary_metric", primary_metric),
    ]:
        if col not in df.columns:
            raise ValueError(f"{name} '{col}' not found in DataFrame columns: {list(df.columns)}")
    if secondary_col not in df.columns:
        raise ValueError(f"secondary metric column '{secondary_col}' not found in DataFrame columns: {list(df.columns)}")

    # A thin wrapper over compare(secondary_metric=...) -- reuses its exact
    # Pareto-bootstrap + calibrated-marginal-CI machinery (core.pareto,
    # _run_pareto_if_needed) rather than re-deriving it here, so tradeoff()
    # and compare(secondary_metric=...) can never drift out of calibration sync.
    # load_from() needs canonical 'model'/'item' column names to build its
    # duplicate-checking key -- an arbitrary factor_col isn't enough on its
    # own, even though compare(factors=...) itself accepts any column name.
    from .loader import load_from
    from .api import compare

    rng_gen = np.random.default_rng(rng)
    evaldata = load_from(
        df, metric_cols=[primary_metric, secondary_col],
        col_map={factor_col: "model", item_col: "item"},
    )
    cr = compare(
        evaldata, factors="model", metric=primary_metric, block="item",
        secondary_metric=secondary_metric, alpha=alpha, n_bootstrap=n_bootstrap, rng=rng_gen,
    )
    if cr._pareto is None:
        raise ValueError(
            "Pareto-front analysis did not run -- check that factor_col/"
            "item_col/primary_metric/secondary_metric are correct and that every "
            "config was scored on every item for both metrics."
        )
    pareto = cr._pareto
    labels = list(pareto["result"].labels)
    statuses = pareto["statuses"]
    return TradeoffResult(
        labels=labels,
        primary_metric=primary_metric,
        secondary_metric=secondary_col,
        direction=direction,
        status={l: statuses[l].status for l in labels},
        frontier_probability=dict(zip(labels, pareto["result"].p_frontier.tolist())),
        _pareto=pareto,
    )


# ---------------------------------------------------------------------------
# judge_debias_mean_ci
# ---------------------------------------------------------------------------

class DebiasedMeanCI(NamedTuple):
    """PPI-corrected mean + CI for judge scores, debiased against a small
    human-labeled subset.

    Returned by :func:`judge_debias_mean_ci`. Shares ``mean``/``ci_low``/
    ``ci_high`` field names with :class:`MeanCI` so code consuming either
    doesn't need to branch on which one it got; the extra fields are
    diagnostic context specific to the correction.

    Attributes
    ----------
    mean : float
        PPI-corrected point estimate.
    ci_low, ci_high : float
        Bootstrap confidence interval on the corrected estimate.
    judge_mean : float
        Uncorrected mean of the judge-only scores (what you'd get without
        this correction).
    human_mean : float
        Mean of the human scores on the labeled subset alone.
    rectifier : float
        Signed correction term (``human_mean`` minus the judge's mean on
        that same labeled subset). Positive means the judge underrates on
        average; negative means it overrates.
    p_value : float or None
        Two-sided bootstrap p-value for H0: corrected mean == 0. ``None``
        unless requested (see ``compute_pvalue`` below) -- rarely
        interesting for a raw mean rather than a difference.
    n_labeled, n_unlabeled : int
        Number of items in the labeled and unlabeled sets.
    """

    mean: float
    ci_low: float
    ci_high: float
    judge_mean: float
    human_mean: float
    rectifier: float
    p_value: Optional[float]
    n_labeled: int
    n_unlabeled: int

    def to_dict(self) -> dict:
        """Return a plain, JSON-friendly dict."""
        return self._asdict()


def judge_debias_mean_ci(
    judge_scores,
    human_scores,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    compute_pvalue: bool = False,
    rng=None,
) -> DebiasedMeanCI:
    """PPI-corrected mean + CI for LLM judge scores, using a small
    human-labeled subset to debias them.

    For the common setup: a judge scored every item, but only a small
    subset also has human labels. Prediction-Powered Inference (PPI) uses
    the disagreement between judge and human on that labeled subset (the
    "rectifier") to correct the judge-only mean over the full dataset,
    without needing every item human-labeled.

    Only corrects a **mean** -- there is no meaningful per-item "debiased
    score"; PPI is a correction to an aggregate estimate, not a per-item
    imputation. For downstream comparisons across several entities/conditions
    (not just a single mean), use ``compare(..., alignment=...)`` together
    with :func:`judge_alignment` instead, which applies the same idea
    per-comparison with the full Friedman/Wilcoxon machinery.

    **The labeled subset must be an unbiased sample of the full dataset --
    ideally chosen uniformly at random.** This function always warns about
    this (see below), because it's easy to violate without realizing it:
    if which items get labeled is itself influenced by their score (e.g.
    "always double-check the highest-scoring ones"), the correction can
    stay biased by a large, non-vanishing amount regardless of how many
    labels you have. See :func:`~evalstats.ppi.correct`'s docstring for the
    full detail (this wraps it directly).

    Parameters
    ----------
    judge_scores : array-like
        Judge scores for every item (the full dataset).
    human_scores : array-like
        Same length as ``judge_scores``, with ``NaN`` for items that don't
        have a human label. (Every item must NOT be labeled -- see the
        "all items labeled" error below; if that's genuinely your
        situation, use :func:`mean_ci` on ``human_scores`` directly
        instead of this function.)
    alpha : float
        Significance level (default 0.05, i.e. 95% CI).
    n_bootstrap : int
        Bootstrap resamples (default 1000).
    compute_pvalue : bool
        Compute a two-sided p-value for H0: corrected mean == 0 (default
        False -- usually not the interesting question for a raw mean).
    rng : int, np.random.Generator, or None
        Seed or generator for reproducibility.

    Returns
    -------
    DebiasedMeanCI

    Examples
    --------
    >>> import evalstats as es
    >>> result = es.judge_debias_mean_ci(judge_scores, human_scores)
    >>> result.mean, result.ci_low, result.ci_high
    """
    from .ppi import correct as _ppi_correct

    judge_full = np.asarray(judge_scores, dtype=float)
    human_full = np.asarray(human_scores, dtype=float)
    if judge_full.shape != human_full.shape:
        raise ValueError(
            "judge_scores and human_scores must be the same length -- one "
            "judge score + one (possibly NaN) human score per item; got "
            f"shapes {judge_full.shape} and {human_full.shape}."
        )
    if judge_full.ndim != 1:
        raise ValueError(f"judge_scores/human_scores must be 1-D; got shape {judge_full.shape}.")

    labeled_mask = ~np.isnan(human_full)
    n_labeled = int(labeled_mask.sum())
    n_total = int(judge_full.size)
    n_unlabeled = n_total - n_labeled

    # Mirrors the exact thresholds api.py's _run_alignment_ppi already
    # enforces for compare(alignment=...)'s PPI path -- same underlying
    # method, same minimum sample sizes for the same reasons.
    if n_labeled < 15:
        raise ValueError(
            f"judge_debias_mean_ci requires at least 15 human-labeled "
            f"items; got {n_labeled}. Expand the labeled subset."
        )
    if n_total < 50:
        raise ValueError(
            f"judge_debias_mean_ci requires at least 50 items total; got "
            f"{n_total}. PPI correction is only beneficial at scale -- for "
            "small datasets, human-label everything and use mean_ci() on "
            "the human labels directly."
        )
    if n_unlabeled == 0:
        raise ValueError(
            "Every item is labeled (human_scores has no NaN) -- there's no "
            "unlabeled portion for PPI to correct. Use mean_ci() on "
            "human_scores directly instead."
        )
    if n_labeled < 30:
        warnings.warn(
            f"judge_debias_mean_ci: only {n_labeled} human-labeled items "
            "(recommend >= 30). The correction may under-cover at this "
            "sample size.",
            UserWarning,
            stacklevel=2,
        )
    if n_total < 100:
        warnings.warn(
            f"judge_debias_mean_ci: only {n_total} total items (recommend "
            ">= 100). The correction may under-cover at this sample size.",
            UserWarning,
            stacklevel=2,
        )
    warnings.warn(
        "judge_debias_mean_ci assumes the labeled subset (non-NaN entries "
        "of human_scores) is an unbiased sample of the full dataset -- "
        "ideally chosen uniformly at random. If which items got labeled "
        "was itself influenced by their score (e.g. always double-checking "
        "the highest-scoring ones), this correction can stay biased "
        "regardless of how many labels you have.",
        UserWarning,
        stacklevel=2,
    )

    y_hat_unlab = judge_full[~labeled_mask]
    y_lab = human_full[labeled_mask]
    y_hat_lab = judge_full[labeled_mask]

    result = _ppi_correct(
        np.mean,
        Y_lab=y_lab, Y_hat_lab=y_hat_lab, Y_hat_unlab=y_hat_unlab,
        alpha=alpha, n_boot=n_bootstrap, rng=rng, compute_pvalue=compute_pvalue,
    )
    return DebiasedMeanCI(
        mean=result.estimate,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        judge_mean=result.llm_estimate,
        human_mean=result.human_estimate,
        rectifier=result.rectifier,
        p_value=result.p_value,
        n_labeled=n_labeled,
        n_unlabeled=n_unlabeled,
    )
