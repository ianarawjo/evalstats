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
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Union

import numpy as np
import pandas as pd

from .config import get_alpha_ci
from .core.router import resolve_auto_robustness_method
from .core.variance import robustness_metrics, seed_variance_decomposition


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

    if alpha is None:
        alpha = get_alpha_ci()
    rng = np.random.default_rng(rng)

    scores_2d = arr.reshape(1, -1)
    _, robustness_method, resolved_score_range = resolve_auto_robustness_method(
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
        n=int(np.sum(~np.isnan(arr))),
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
        _, robustness_method, resolved_score_range = resolve_auto_robustness_method(
            a_2d, score_range=score_range, stacklevel=4,
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
        n[i] = int(np.sum(~np.isnan(a)))
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


def stability(runs: Union[np.ndarray, dict], *, labels: Optional[list[str]] = None) -> StabilityResult:
    """Multi-run reliability: how much does a config's score move across
    repeated runs on the same items?

    Standalone version of the seed-instability decomposition ``compare()``
    shows for multi-run (seeded) benchmarks -- for deciding "is this
    configuration reliable enough to ship" without needing a full
    multi-model comparison.

    Parameters
    ----------
    runs : array-like or dict
        A single config's repeated-run scores as a 2-D array of shape
        ``(K, M)`` (K runs, M items, same M items each run) -- one config;
        or a ``{label: (K, M) array}`` dict of several configs. Configs
        must share the same M (same item set); K (number of runs, >= 3) can
        differ per config.
    labels : list[str], optional
        Override labels when ``runs`` is a single array (default
        ``["value"]``). Ignored when ``runs`` is a dict (its keys are used).

    Returns
    -------
    StabilityResult

    Examples
    --------
    >>> import evalstats as es
    >>> es.stability(rag_config_a_runs)  # shape (5, 200): 5 runs, 200 items
    >>> es.stability({"config_a": runs_a, "config_b": runs_b}).to_frame()
    """
    if isinstance(runs, dict):
        if len(runs) == 0:
            raise ValueError("runs dict must not be empty.")
        input_labels = [str(k) for k in runs.keys()]
        arrays = [np.asarray(v, dtype=float) for v in runs.values()]
    else:
        arr = np.asarray(runs, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                "runs must be a 2-D array (K runs x M items) or a "
                f"{{label: array}} dict of such arrays; got shape {arr.shape}."
            )
        input_labels = list(labels) if labels is not None else ["value"]
        if len(input_labels) != 1:
            raise ValueError(
                f"A single runs array takes at most one label; got {len(input_labels)}."
            )
        arrays = [arr]

    for lbl, a in zip(input_labels, arrays):
        if a.ndim != 2:
            raise ValueError(f"runs['{lbl}'] must be 2-D (K runs x M items); got shape {a.shape}.")

    m_values = {a.shape[1] for a in arrays}
    if len(m_values) != 1:
        raise ValueError(
            "All configs must be evaluated on the same number of items (M); "
            f"got M values {sorted(m_values)} across configs "
            f"{dict(zip(input_labels, (a.shape for a in arrays)))}."
        )
    m_items = m_values.pop()

    # Different configs may have different K (run count); pad the run axis
    # with NaN -- seed_variance_decomposition's internal nanmean/nanvar
    # handle that safely (it's a closed-form ANOVA-style computation, not a
    # resampling procedure, so NaN-tolerant reductions are exact here).
    max_k = max(a.shape[0] for a in arrays)
    if any(a.shape[0] < 3 for a in arrays):
        offender = input_labels[[a.shape[0] for a in arrays].index(min(a.shape[0] for a in arrays))]
        raise ValueError(
            f"Seed-variance decomposition requires >= 3 runs per config; "
            f"config '{offender}' has {min(a.shape[0] for a in arrays)}."
        )
    scores_3d = np.full((len(arrays), m_items, max_k), np.nan)
    for i, a in enumerate(arrays):
        scores_3d[i, :, : a.shape[0]] = a.T  # (K, M) -> (M, K)

    from .core.summary import _instability_label

    sv = seed_variance_decomposition(scores_3d, input_labels)
    actual_n_runs = np.array([a.shape[0] for a in arrays])
    return StabilityResult(
        labels=input_labels,
        instability=sv.instability,
        icc=sv.icc,
        n_runs=actual_n_runs,
        label_text=[_instability_label(float(v)) for v in sv.instability],
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
    unlabeled_judge_scores,
    labeled_human_scores,
    labeled_judge_scores,
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

    Parameters
    ----------
    unlabeled_judge_scores : array-like
        Judge scores for every item that does NOT also have a human label.
        Must be disjoint from the labeled items below -- do not pass every
        item's judge score here if some of them are also in
        ``labeled_judge_scores``.
    labeled_human_scores : array-like
        Human scores for the labeled subset.
    labeled_judge_scores : array-like
        Judge scores for that SAME labeled subset, paired (same order,
        same length) with ``labeled_human_scores``.
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
    >>> result = es.judge_debias_mean_ci(
    ...     unlabeled_judge_scores=judge_scores[~has_human_label],
    ...     labeled_human_scores=human_scores[has_human_label],
    ...     labeled_judge_scores=judge_scores[has_human_label],
    ... )
    >>> result.mean, result.ci_low, result.ci_high
    """
    from .ppi import correct as _ppi_correct

    y_hat_unlab = np.asarray(unlabeled_judge_scores, dtype=float)
    y_lab = np.asarray(labeled_human_scores, dtype=float)
    y_hat_lab = np.asarray(labeled_judge_scores, dtype=float)

    if y_lab.shape != y_hat_lab.shape:
        raise ValueError(
            "labeled_human_scores and labeled_judge_scores must be paired, "
            f"same-shape arrays (one human + one judge score per labeled "
            f"item); got shapes {y_lab.shape} and {y_hat_lab.shape}."
        )
    if y_lab.size < 15:
        warnings.warn(
            f"Only {y_lab.size} labeled items -- PPI correction will be "
            "imprecise with fewer than ~15 labeled items. Consider "
            "expanding the labeled subset.",
            UserWarning,
            stacklevel=2,
        )

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
        n_labeled=int(y_lab.size),
        n_unlabeled=int(y_hat_unlab.size),
    )
