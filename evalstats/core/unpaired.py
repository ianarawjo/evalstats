"""Between-subjects (unpaired) comparison engine.

Sibling to ``core/paired.py``, not a branch inside it: the paired path's
entire machinery (``all_pairwise``, ``PairwiseMatrix``, ``PairedDiffResult``)
is built around item-matched differences (``per_input_diffs``), which has no
meaning for genuinely disjoint groups (e.g. different, unrelated reviewers
per app). This module implements the corresponding between-subjects
statistics as its own self-contained path, dispatched to from
``evalstats.api.compare()`` when ``design="unpaired"`` (or auto-detected),
and reuses the existing PPI-corrected test machinery in ``evalstats.tests``
as its statistical engine rather than reimplementing anything.

Two test families (see ``config.AUTO_UNPAIRED_METHOD_TABLE`` for the
decision and full rationale):

* **binary** data -- ``anova_oneway`` (omnibus, k>=3 only) + pairwise
  Welch's ``ttest``-equivalent CIs (mean/proportion difference).
* **continuous / likert / grade** data -- ``kruskalwallis`` (omnibus,
  k>=3 only) + pairwise Mann-Whitney-equivalent CIs (stochastic-dominance
  probability θ=P(a>b)).

At k=2 there is only one possible comparison, so there is no separate
omnibus test and no multiple-comparison correction to apply (Bonferroni/
Holm are no-ops at a family size of 1) -- the single pairwise result *is*
the answer. At k>=3, pairwise CIs get a Bonferroni correction and pairwise
p-values get a Holm correction, two independent axes mirroring how the
paired path separates its own simultaneous-CI and p-value-correction
machinery (see ``core/paired.py``'s ``_simultaneous_cis_router`` and
``correct_pvalues``).
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from evalstats.config import resolve_auto_unpaired_methods, get_alpha_ci
from evalstats.core.stats_utils import correct_pvalues
from evalstats.loader import _CANONICAL_ALIASES, _find_col, _detect_score_type

if TYPE_CHECKING:
    from evalstats.alignment import AlignmentResult

# Same literal value as evalstats.labeling.SYNTHETIC_ITEM_COL -- kept as an
# independent local constant rather than imported, since core/ modules
# shouldn't depend on the CLI-facing labeling module (wrong layering
# direction); it's a small, deliberate duplication of one string constant.
SYNTHETIC_ITEM_COL = "_row_item"


# ─────────────────────────────────────────────────────────────────────────────
# Result objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroupStat:
    """Descriptive stats + calibrated marginal CI for one group.

    Computed independently per group (own auto-detected data kind, own N,
    own CI method) via the same machinery ``evalstats.quick.summarize``
    uses -- there is no rectangular-design requirement here, so unbalanced
    group sizes are handled naturally.
    """
    label: str
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    method: str
    multi_ci: Optional[dict] = None  # {alpha: (lo, hi)} gradient CI bands


@dataclass
class GroupDiffResult:
    """One pairwise between-subjects comparison.

    Not ``PairedDiffResult`` -- there is no ``per_input_diffs`` here, since
    the two groups' items have no correspondence to difference.
    """
    label_a: str
    label_b: str
    estimand: str          # "mean_diff" (binary family) or "dominance" (rank-based family)
    null_value: float       # 0.0 for mean_diff, 0.5 for dominance
    point_estimate: float
    ci_low: float
    ci_high: float          # Bonferroni-corrected at k>=3; nominal alpha at k=2
    p_value: float          # Holm-corrected at k>=3; raw at k=2
    raw_p_value: float      # always uncorrected, for transparency
    n_a: int
    n_b: int

    @property
    def significant(self) -> bool:
        return not (self.ci_low <= self.null_value <= self.ci_high)


@dataclass
class GroupComparisonResult:
    """Result of a between-subjects ``compare(design="unpaired")`` call.

    Deliberately a narrower reporting surface than ``ComparisonResult``
    (no executive summary, critical-difference rank bands, or forest-plot
    brackets) -- per-group means with gradient CIs, a pairwise comparison
    table, and the omnibus test when k>=3. See PLAN_between_subjects_
    extension.md §1/§3.6 for the scope discussion.
    """
    factor_col: str
    metric_col: str
    item_col: str
    item_col_synthetic: bool
    score_type: str          # "binary" | "continuous" | "likert" | "grade"
    family: str              # "binary_proportion" | "rank_based"
    groups: list[GroupStat]
    pairwise: list[GroupDiffResult]
    omnibus_test_name: Optional[str]     # None at k=2 -- no separate omnibus test
    omnibus_statistic: Optional[float]
    omnibus_p_value: Optional[float]              # uncorrected
    omnibus_corrected_p_value: Optional[float]     # PPI-corrected, when alignment given
    alpha: float
    n_pairs: int
    ci_correction: str        # "bonferroni" or "none" (k=2, single comparison)
    pvalue_correction: str    # "holm" or "none" (k=2, single comparison)
    ppi_applied: bool
    alignment_result: Optional["AlignmentResult"] = None

    # ── convenience accessors ──────────────────────────────────────────────

    @property
    def labels(self) -> list[str]:
        return [g.label for g in self.groups]

    def _group(self, label: str) -> GroupStat:
        for g in self.groups:
            if g.label == label:
                return g
        raise KeyError(f"no group {label!r}; available: {self.labels}")

    def _pair(self, label_a: str, label_b: str) -> GroupDiffResult:
        for p in self.pairwise:
            if {p.label_a, p.label_b} == {label_a, label_b}:
                return p
        raise KeyError(f"no pairwise result for ({label_a!r}, {label_b!r})")

    # ── reporting ───────────────────────────────────────────────────────────

    def summary(self) -> None:
        from evalstats.core.summary_unpaired import print_group_comparison_summary
        print_group_comparison_summary(self)

    def plot(self, **kwargs):
        raise NotImplementedError(
            "GroupComparisonResult.plot() is not implemented yet -- between-"
            "subjects plotting is scoped for a later phase. Use .summary() "
            "or .to_frame() in the meantime."
        )

    def to_dict(self) -> dict:
        return {
            "design": "unpaired",
            "factor_col": self.factor_col,
            "metric_col": self.metric_col,
            "item_col": self.item_col,
            "item_col_synthetic": self.item_col_synthetic,
            "score_type": self.score_type,
            "family": self.family,
            "alpha": self.alpha,
            "ppi_applied": self.ppi_applied,
            "groups": {
                g.label: {
                    "n": g.n, "mean": g.mean, "ci_low": g.ci_low,
                    "ci_high": g.ci_high, "method": g.method,
                }
                for g in self.groups
            },
            "omnibus": (
                None if self.omnibus_test_name is None else {
                    "test_name": self.omnibus_test_name,
                    "statistic": self.omnibus_statistic,
                    "p_value": self.omnibus_p_value,
                    "corrected_p_value": self.omnibus_corrected_p_value,
                }
            ),
            "pairwise": [
                {
                    "a": p.label_a, "b": p.label_b, "estimand": p.estimand,
                    "point_estimate": p.point_estimate,
                    "ci_low": p.ci_low, "ci_high": p.ci_high,
                    "p_value": p.p_value, "raw_p_value": p.raw_p_value,
                    "significant": p.significant,
                    "n_a": p.n_a, "n_b": p.n_b,
                }
                for p in self.pairwise
            ],
            "ci_correction": self.ci_correction,
            "pvalue_correction": self.pvalue_correction,
        }

    def to_frame(self) -> pd.DataFrame:
        """One row per pairwise comparison."""
        rows = [
            {
                "a": p.label_a, "b": p.label_b, "estimand": p.estimand,
                "point_estimate": p.point_estimate,
                "ci_low": p.ci_low, "ci_high": p.ci_high,
                "p_value": p.p_value, "raw_p_value": p.raw_p_value,
                "significant": p.significant,
                "n_a": p.n_a, "n_b": p.n_b,
            }
            for p in self.pairwise
        ]
        return pd.DataFrame(rows)

    def groups_to_frame(self) -> pd.DataFrame:
        """One row per group (descriptive stats)."""
        rows = [
            {"label": g.label, "n": g.n, "mean": g.mean,
             "ci_low": g.ci_low, "ci_high": g.ci_high, "method": g.method}
            for g in self.groups
        ]
        return pd.DataFrame(rows).set_index("label")


# ─────────────────────────────────────────────────────────────────────────────
# FWER helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bonferroni_alpha(alpha: float, n_pairs: int) -> float:
    """Bonferroni-adjusted alpha for a family of n_pairs comparisons.

    Deliberately Bonferroni, not Šidák: Šidák's exactness needs (near-)
    independence between the pairwise statistics, which is unverified for
    this bootstrap's dependence structure (pairs sharing a group are
    correlated). Bonferroni's union bound holds regardless. See
    PLAN_between_subjects_extension.md §3.4.
    """
    return alpha if n_pairs <= 1 else alpha / n_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise engines -- one PPI + one non-PPI function per family. Both take a
# list of group arrays (any k>=2 -- Bonferroni/Holm no-op at n_pairs=1, so
# the k=2 case doesn't need separate code) and return a dict with "pairs",
# a point-estimate array, "ci_lo"/"ci_hi", and "pair_p" (uncorrected).
# ─────────────────────────────────────────────────────────────────────────────

def _rank_based_pairwise_ppi(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], alpha: float, n_boot: int, rng,
) -> dict:
    """θ_ab = P_mid(a>b) for every pair, PPI-corrected. Thin wrapper around
    the private kruskalwallis machinery -- valid at any k>=2 (kruskalwallis's
    own docstring: "For k=2 groups Kruskal-Wallis reduces to Mann-Whitney;
    this pairwise-θ framework is the direct generalization to k>2"), so this
    is reused uniformly rather than special-casing k=2 through mannwhitney().
    """
    from evalstats.tests import _ppi_kruskal_wallis_pairwise
    out = _ppi_kruskal_wallis_pairwise(groups, groups_lab, alpha, n_boot, rng)
    return {
        "pairs": out["pairs"], "point": out["theta_hat"],
        "ci_lo": out["ci_lo"], "ci_hi": out["ci_hi"], "pair_p": out["pair_p"],
    }


def _rank_based_pairwise_uncorrected(
    groups: list[np.ndarray], alpha: float, n_boot: int, rng,
) -> dict:
    """Non-PPI analog of :func:`_rank_based_pairwise_ppi` -- a stripped-down
    copy of ``_ppi_kruskal_wallis_pairwise`` with the rectifier terms
    removed (plain bootstrap of the same θ_ab estimator, no human labels).
    """
    from evalstats.tests import _kw_pairwise_thetas
    rng = np.random.default_rng(rng)
    k = len(groups)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    n_per_group = [len(g) for g in groups]
    theta_hat = _kw_pairwise_thetas(groups, pairs)

    boots = np.empty((n_boot, len(pairs)))
    for bi in range(n_boot):
        resampled = [groups[j][rng.integers(0, n_per_group[j], n_per_group[j])] for j in range(k)]
        boots[bi] = _kw_pairwise_thetas(resampled, pairs)

    ci_lo = np.percentile(boots, 100 * alpha / 2, axis=0)
    ci_hi = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)
    pair_p = 2.0 * np.minimum((boots <= 0.5).mean(axis=0), (boots >= 0.5).mean(axis=0))
    pair_p = np.minimum(pair_p, 1.0)
    return {"pairs": pairs, "point": theta_hat, "ci_lo": ci_lo, "ci_hi": ci_hi, "pair_p": pair_p}


def _binary_pairwise_ppi(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], alpha: float, power_tune: bool = True,
) -> dict:
    """Δp = mean(a) - mean(b) for every pair, PPI-corrected via the exact
    closed-form construction ``ttest()`` itself uses for the independent,
    labeled case (``_ppi_two_sample_t_interval``) -- called directly
    (bypassing the public wrapper) the same way the rank-based family calls
    ``_ppi_kruskal_wallis_pairwise`` directly, so both families are handled
    consistently and neither changes ``evalstats.tests``' public contract.
    """
    from evalstats.tests import _ppi_two_sample_t_interval
    k = len(groups)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    point = np.empty(len(pairs))
    ci_lo = np.empty(len(pairs))
    ci_hi = np.empty(len(pairs))
    pair_p = np.empty(len(pairs))
    for idx, (a, b) in enumerate(pairs):
        res = _ppi_two_sample_t_interval(
            groups[a], groups[b], groups_lab[a], groups_lab[b], alpha, power_tune=power_tune,
        )
        point[idx] = res.estimate
        ci_lo[idx] = res.ci_low
        ci_hi[idx] = res.ci_high
        pair_p[idx] = res.p_value
    return {"pairs": pairs, "point": point, "ci_lo": ci_lo, "ci_hi": ci_hi, "pair_p": pair_p}


def _binary_pairwise_uncorrected(groups: list[np.ndarray], alpha: float) -> dict:
    """Non-PPI analog: ordinary Welch's t-interval per pair (closed-form,
    via scipy -- no bootstrap needed since this has no PPI rectifier).
    """
    from scipy.stats import ttest_ind
    k = len(groups)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    point = np.empty(len(pairs))
    ci_lo = np.empty(len(pairs))
    ci_hi = np.empty(len(pairs))
    pair_p = np.empty(len(pairs))
    for idx, (a, b) in enumerate(pairs):
        r = ttest_ind(groups[a], groups[b], equal_var=False)
        ci = r.confidence_interval(confidence_level=1.0 - alpha)
        point[idx] = float(np.mean(groups[a]) - np.mean(groups[b]))
        ci_lo[idx] = float(ci.low)
        ci_hi[idx] = float(ci.high)
        pair_p[idx] = float(r.pvalue)
    return {"pairs": pairs, "point": point, "ci_lo": ci_lo, "ci_hi": ci_hi, "pair_p": pair_p}


# ─────────────────────────────────────────────────────────────────────────────
# Per-group descriptive stats
# ─────────────────────────────────────────────────────────────────────────────

def _compute_group_stats(
    labels: list[str], arrays: list[np.ndarray], *, alpha: float, n_bootstrap: int, rng,
) -> list[GroupStat]:
    """Per-group mean + calibrated marginal CI (with gradient multi_ci
    bands), computed independently per group -- the exact same building
    block ``evalstats.quick.summarize`` uses internally, called directly
    here (with multi_ci=True, which summarize()'s own public signature
    doesn't expose) rather than through that quick-primitive wrapper.
    """
    from evalstats.core.router import resolve_auto_robustness_method
    from evalstats.core.variance import robustness_metrics

    out = []
    for label, arr in zip(labels, arrays):
        a2d = arr.reshape(1, -1)
        _, robustness_method, resolved_score_range, _ = resolve_auto_robustness_method(
            a2d, stacklevel=4,
        )
        rob = robustness_metrics(
            a2d, ["_"],
            n_bootstrap=n_bootstrap, rng=rng, alpha=alpha,
            statistic="mean", marginal_method=robustness_method,
            multi_ci=True, score_range=resolved_score_range,
        )
        multi_ci = (
            {a: (float(lo[0]), float(hi[0])) for a, (lo, hi) in rob.multi_ci.items()}
            if rob.multi_ci is not None else None
        )
        out.append(GroupStat(
            label=label, n=int(arr.size), mean=float(rob.mean[0]), std=float(rob.std[0]),
            ci_low=float(rob.ci_low[0]) if rob.ci_low is not None else float("nan"),
            ci_high=float(rob.ci_high[0]) if rob.ci_high is not None else float("nan"),
            method=robustness_method, multi_ci=multi_ci,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def compare_unpaired(
    df: pd.DataFrame,
    *,
    factor_col: str,
    metric_col: str,
    item_col: Optional[str] = None,
    alignment: Optional[dict] = None,
    alpha: Optional[float] = None,
    n_boot: int = 2000,
    rng=None,
) -> GroupComparisonResult:
    """Between-subjects comparison engine -- see module docstring.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with at least ``factor_col`` and ``metric_col``.
    factor_col : str
        Column identifying which group each row belongs to.
    metric_col : str
        Numeric score column to compare.
    item_col : str, optional
        Row/item identifier column. When not given, auto-detected via the
        same canonical aliases ``load_from()`` uses; when none of those
        match either, a synthetic positional id is used (each row is its
        own item) -- between-subjects data commonly has no natural item id
        at all (e.g. just group + rating, no reviewer id).
    alignment : dict, optional
        ``{metric_col: AlignmentResult}``, matching ``compare()``'s own
        ``alignment=`` convention exactly -- the caller has already run
        ``judge_alignment()``; this just consumes the result (splitting its
        human-label column by group) and displays it inline.
    alpha : float, optional
        Significance level. Defaults to :func:`evalstats.get_alpha_ci`.
    n_boot : int
        Bootstrap resamples for the rank-based family's pairwise CIs
        (unused by the binary family, which is closed-form).
    rng : optional
        Seed or ``np.random.Generator``.

    Returns
    -------
    GroupComparisonResult
    """
    if factor_col not in df.columns:
        raise ValueError(f"factor_col {factor_col!r} not found in data.")
    if metric_col not in df.columns:
        raise ValueError(f"metric_col {metric_col!r} not found in data.")

    if alpha is None:
        alpha = get_alpha_ci()
    rng = np.random.default_rng(rng)

    resolved_item = item_col or _find_col(df, _CANONICAL_ALIASES["item"])
    item_synthetic = resolved_item is None
    if item_synthetic:
        resolved_item = SYNTHETIC_ITEM_COL
    elif resolved_item not in df.columns:
        raise ValueError(f"item_col {resolved_item!r} not found in data.")

    groups_series = df.groupby(factor_col, sort=False)[metric_col]
    labels = [str(k) for k in groups_series.groups.keys()]
    if len(labels) < 2:
        raise ValueError(
            f"factor_col {factor_col!r} has only {len(labels)} distinct value(s) -- "
            "need at least 2 groups to compare."
        )
    group_arrays = [
        groups_series.get_group(k).to_numpy(dtype=float) for k in groups_series.groups.keys()
    ]
    for lbl, arr in zip(labels, group_arrays):
        if arr.size == 0:
            raise ValueError(f"Group {lbl!r} has no scores.")

    score_type = _detect_score_type(df[metric_col].dropna())
    _, pairwise_method = resolve_auto_unpaired_methods(score_type)
    family = "binary_proportion" if pairwise_method == "ttest" else "rank_based"

    ppi_applied = alignment is not None and metric_col in alignment
    alignment_result = alignment[metric_col] if ppi_applied else None
    group_lab_arrays: Optional[list[np.ndarray]] = None
    if ppi_applied:
        human_col = alignment_result.human_col
        if human_col not in df.columns:
            raise ValueError(
                f"alignment result's human_groundtruth column {human_col!r} "
                "not found in data."
            )
        lab_series = df.groupby(factor_col, sort=False)[human_col]
        lab_groups = dict(lab_series.groups)
        group_lab_arrays = [
            df.loc[lab_groups[k], human_col].to_numpy(dtype=float)
            for k in groups_series.groups.keys()
        ]

    group_stats = _compute_group_stats(
        labels, group_arrays, alpha=alpha, n_bootstrap=n_boot, rng=rng,
    )

    k = len(labels)
    n_pairs = k * (k - 1) // 2
    ci_alpha = _bonferroni_alpha(alpha, n_pairs)

    # ── Omnibus test (k>=3 only -- at k=2 there's nothing to protect against,
    # the single pairwise comparison already answers the whole question) ────
    #
    # Suppressed stdout: every evalstats.tests function with labels given
    # unconditionally prints its own alignment report via _run_alignment_
    # report -- NOT gated by print_result (that only gates the TestResult's
    # own .summary()). That internal report re-runs judge_alignment() from
    # scratch with no selection= (always "unknown"), which would print a
    # second, worse, differently-labeled alignment report right in the
    # middle of ours -- we already print the caller's real, correctly-
    # disclosed AlignmentResult once via the PPI banner above. Left as-is
    # in evalstats.tests itself (existing, validated, widely-used behavior
    # for direct callers of that module -- not something to change here);
    # suppressed only at this call site.
    omnibus_test_name = omnibus_statistic = omnibus_p_value = omnibus_corrected_p_value = None
    if k >= 3:
        with contextlib.redirect_stdout(io.StringIO()) if ppi_applied else contextlib.nullcontext():
            if family == "binary_proportion":
                from evalstats.tests import anova_oneway
                om = anova_oneway(
                    *group_arrays, groups_lab=group_lab_arrays if ppi_applied else None,
                    alpha=alpha, n_boot=n_boot, rng=rng, print_result=False,
                )
                omnibus_test_name = "One-way ANOVA (independent)"
            else:
                from evalstats.tests import kruskalwallis
                om = kruskalwallis(
                    *group_arrays, groups_lab=group_lab_arrays if ppi_applied else None,
                    alpha=alpha, n_boot=n_boot, rng=rng, print_result=False,
                )
                omnibus_test_name = "Kruskal-Wallis test"
        omnibus_statistic = float(om.statistic)
        omnibus_p_value = float(om.p_value)
        omnibus_corrected_p_value = (
            float(om.corrected_p_value) if om.corrected_p_value is not None else None
        )

    # ── Pairwise table (all k>=2 -- Bonferroni/Holm no-op at n_pairs=1) ─────
    if family == "binary_proportion":
        pw = (
            _binary_pairwise_ppi(group_arrays, group_lab_arrays, ci_alpha)
            if ppi_applied else _binary_pairwise_uncorrected(group_arrays, ci_alpha)
        )
        estimand, null_value = "mean_diff", 0.0
    else:
        pw = (
            _rank_based_pairwise_ppi(group_arrays, group_lab_arrays, ci_alpha, n_boot, rng)
            if ppi_applied else _rank_based_pairwise_uncorrected(group_arrays, ci_alpha, n_boot, rng)
        )
        estimand, null_value = "dominance", 0.5

    raw_p = np.asarray(pw["pair_p"], dtype=float)
    corrected_p = correct_pvalues(raw_p, method="holm") if n_pairs > 1 else raw_p.copy()

    pairwise = [
        GroupDiffResult(
            label_a=labels[i], label_b=labels[j], estimand=estimand, null_value=null_value,
            point_estimate=float(pw["point"][idx]),
            ci_low=float(pw["ci_lo"][idx]), ci_high=float(pw["ci_hi"][idx]),
            p_value=float(corrected_p[idx]), raw_p_value=float(raw_p[idx]),
            n_a=int(group_arrays[i].size), n_b=int(group_arrays[j].size),
        )
        for idx, (i, j) in enumerate(pw["pairs"])
    ]

    return GroupComparisonResult(
        factor_col=factor_col, metric_col=metric_col,
        item_col=resolved_item, item_col_synthetic=item_synthetic,
        score_type=score_type, family=family,
        groups=group_stats, pairwise=pairwise,
        omnibus_test_name=omnibus_test_name, omnibus_statistic=omnibus_statistic,
        omnibus_p_value=omnibus_p_value, omnibus_corrected_p_value=omnibus_corrected_p_value,
        alpha=alpha, n_pairs=n_pairs,
        ci_correction="bonferroni" if n_pairs > 1 else "none",
        pvalue_correction="holm" if n_pairs > 1 else "none",
        ppi_applied=ppi_applied, alignment_result=alignment_result,
    )
