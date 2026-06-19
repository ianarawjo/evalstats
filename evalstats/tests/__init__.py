"""evalstats.tests — PPI-corrected wrappers for common statistical tests.

Each function mirrors the corresponding SciPy test but accepts optional
``*_lab`` arguments carrying sparse human labels for PPI correction of
LLM judge measurement error.

Calling convention
------------------
Pass score arrays exactly as you would to SciPy, plus ``a_lab`` / ``b_lab``
(or ``x_lab`` / ``y_lab``) arrays of the **same length** with ``NaN`` for
items that were not human-labeled::

    # Uncorrected — identical to scipy
    result = es.tests.ttest(a, b)

    # PPI-corrected — add sparse human labels
    result = es.tests.ttest(a, b, a_lab=human_a, b_lab=human_b)

Human-label arrays
------------------
``a_lab`` / ``b_lab`` (etc.) must be the **same length** as the
corresponding score array.  Set elements to ``NaN`` (or ``None``) for
items that have no human label.  At least one labeled item per group is
required; ~20–30+ labeled items per group is recommended for stable PPI.

Alignment report
----------------
When human labels are supplied, ``validate_alignment()`` is called
internally and its report is printed before the test result so alignment
quality is always visible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# TestResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Result returned by all ``evalstats.tests`` functions.

    Attributes
    ----------
    test_name : str
    statistic : float
        Test statistic from the uncorrected SciPy test.
    p_value : float
        p-value from the uncorrected SciPy test.
    corrected_estimate : float or None
        PPI-corrected point estimate.  ``None`` when no human labels supplied.
    corrected_ci : tuple[float, float] or None
        Bootstrap CI for the corrected estimate.
    corrected_p_value : float or None
        Bootstrap p-value for H₀: θ = 0 (or H₀: P(A>B) = 0.5 for
        Mann-Whitney).  Computed as
        ``2 * min(P(θ̂* ≤ 0), P(θ̂* ≥ 0))`` over PPI bootstrap resamples.
    corrected_statistic : float or None
        Re-computed test statistic from PPI-corrected inputs (omnibus tests).
    rectifier : float or None
        Bias correction term δ = human_estimate − llm_estimate on the
        labeled subset.  Positive means the LLM over-estimated.
    n_labeled : int or None
    n_total : int or None
    alpha : float
    extra : dict
        Test-specific supplementary values.
    """

    test_name: str
    statistic: float
    p_value: float
    corrected_estimate: Optional[float] = None
    corrected_ci: Optional[tuple] = None
    corrected_p_value: Optional[float] = None
    corrected_statistic: Optional[float] = None
    rectifier: Optional[float] = None
    n_labeled: Optional[int] = None
    n_total: Optional[int] = None
    alpha: float = 0.05
    extra: dict = field(default_factory=dict)

    def summary(self) -> None:
        """Print uncorrected and PPI-corrected results side by side."""
        ci_pct = int(round((1 - self.alpha) * 100))
        w = 30
        print(f"\n{self.test_name}")
        print("─" * 58)
        print(f"  {'Uncorrected':<{w}}  {'PPI-corrected'}")
        print(f"  {'─'*w}  {'─'*20}")
        print(f"  {'Statistic':<{w}}: {self.statistic:.4f}", end="")
        if self.corrected_statistic is not None:
            print(f"  →  {self.corrected_statistic:.4f}")
        else:
            print()
        print(f"  {'p-value':<{w}}: {self.p_value:.4f}", end="")
        if self.corrected_p_value is not None:
            print(f"  →  {self.corrected_p_value:.4f}")
        else:
            print()
        if self.corrected_estimate is not None:
            print(f"  {'Corrected estimate':<{w}}:        {self.corrected_estimate:.4f}")
        if self.corrected_ci is not None:
            lo, hi = self.corrected_ci
            print(f"  {f'{ci_pct}% CI':<{w}}:        [{lo:.4f}, {hi:.4f}]")
        if self.rectifier is not None:
            print(f"  {'Rectifier (δ)':<{w}}:        {self.rectifier:+.4f}")
        if self.n_labeled is not None:
            print(f"  {'Alignment set':<{w}}:        {self.n_labeled} / {self.n_total} items labeled")
        for k, v in self.extra.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk:<{w-4}}: {vv}")
            else:
                print(f"  {k:<{w}}: {v}")
        print()

    def __repr__(self) -> str:
        p_corr = f", corrected_p={self.corrected_p_value:.4f}" if self.corrected_p_value is not None else ""
        return f"TestResult({self.test_name}, p={self.p_value:.4f}{p_corr})"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce(arr) -> np.ndarray:
    """Coerce to float64 numpy array, converting None → NaN."""
    return np.where(
        pd.isna(arr), np.nan, np.asarray(arr, dtype=float)
    )


def _run_alignment_report(llm_all: np.ndarray, human_sparse: np.ndarray):
    """Fit and print an alignment report from raw arrays.

    Parameters
    ----------
    llm_all : 1-D array
        LLM scores for all items (no missing values).
    human_sparse : 1-D array
        Human labels for the same items, with NaN where unlabeled.

    Returns
    -------
    AlignmentResult
    """
    from evalstats.alignment import validate_alignment
    from evalstats.loader import _detect_score_type

    _LLM = "__llm__"
    _HUM = "__human__"
    df = pd.DataFrame({_LLM: llm_all, _HUM: human_sparse})

    class _EvalStub:
        def __init__(self):
            self._df = df
            self._score_types = {_LLM: _detect_score_type(pd.Series(llm_all))}

    ar = validate_alignment(_EvalStub(), llm_metric=_LLM, human_groundtruth=_HUM)
    ar.summary()
    return ar


def _sanitize_ppi_labels(
    a: np.ndarray,
    b: np.ndarray,
    a_lab,
    b_lab,
    *,
    paired: bool,
    label_names: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce sparse human labels for PPI wrappers.

    Policy:
      - fewer than 15 effective labels: raise ValueError
      - fewer than 30 effective labels: raise UserWarning

    Effective labels are counted as:
      - independent tests: non-NaN labels across both groups
      - paired tests: positions where both labels are non-NaN
    """
    a_name, b_name = label_names

    a_lab = _coerce(a_lab) if a_lab is not None else np.full(len(a), np.nan)
    b_lab = _coerce(b_lab) if b_lab is not None else np.full(len(b), np.nan)

    if len(a_lab) != len(a):
        raise ValueError(f"{a_name} must have same length as first score array ({len(a)}).")
    if len(b_lab) != len(b):
        raise ValueError(f"{b_name} must have same length as second score array ({len(b)}).")

    if paired:
        n_effective = int(np.sum(~np.isnan(a_lab) & ~np.isnan(b_lab)))
        min_msg = (
            "At least 15 overlapping human-labeled positions are required for PPI "
            f"(found {n_effective})."
        )
        warn_msg = (
            "Only {n} overlapping human-labeled positions were supplied. "
            "PPI bootstrap can undercover below 30 labels; consider labeling more items."
        )
    else:
        n_effective = int(np.sum(~np.isnan(a_lab)) + np.sum(~np.isnan(b_lab)))
        min_msg = (
            "At least 15 human labels are required for PPI "
            f"(found {n_effective} across {a_name}/{b_name})."
        )
        warn_msg = (
            "Only {n} human labels were supplied across both groups. "
            "PPI bootstrap can undercover below 30 labels; consider labeling more items."
        )

    if n_effective < 15:
        raise ValueError(min_msg)
    if n_effective < 30:
        warnings.warn(warn_msg.format(n=n_effective), UserWarning, stacklevel=3)

    return a_lab, b_lab


def _ppi_two_sample(
    a: np.ndarray,
    b: np.ndarray,
    a_lab: np.ndarray,
    b_lab: np.ndarray,
    estimator_func: Callable,
    alpha: float,
    n_boot: int,
    rng,
):
    """PPI correction for a two-sample scalar estimand.

    ``estimator_func(ya, yb) → float`` receives the group-A and group-B
    sub-arrays and returns a scalar.

    Internally uses integer group labels (0 = A, 1 = B) so group identity
    survives bootstrap resampling without string comparison.
    """
    from evalstats.ppi import correct

    mask_a = ~np.isnan(a_lab)
    mask_b = ~np.isnan(b_lab)

    if mask_a.sum() == 0 and mask_b.sum() == 0:
        raise ValueError("No labeled items found in a_lab or b_lab.")

    Y_hat_unlab = np.concatenate([a, b])
    X_unlab     = np.array([0] * len(a) + [1] * len(b), dtype=int)
    Y_lab       = np.concatenate([a_lab[mask_a], b_lab[mask_b]])
    Y_hat_lab   = np.concatenate([a[mask_a],     b[mask_b]])
    X_lab       = np.array([0] * int(mask_a.sum()) + [1] * int(mask_b.sum()), dtype=int)

    def _est(Y, X):
        return estimator_func(Y[X == 0], Y[X == 1])

    return correct(
        _est,
        Y_lab=Y_lab,
        Y_hat_lab=Y_hat_lab,
        Y_hat_unlab=Y_hat_unlab,
        X_lab=X_lab,
        X_unlab=X_unlab,
        alpha=alpha,
        n_boot=n_boot,
        rng=rng,
        compute_pvalue=True,
    )


def _ppi_paired_arrays(
    a: np.ndarray,
    b: np.ndarray,
    a_lab: np.ndarray,
    b_lab: np.ndarray,
    statistic: Callable,
    alpha: float,
    n_boot: int,
    rng,
):
    """PPI correction for a paired estimand ``statistic(a_i − b_i)``.

    Pairing is by array position — ``a[i]`` and ``b[i]`` are the same subject,
    exactly as in ``scipy.stats.ttest_rel`` and ``scipy.stats.wilcoxon``.
    A position is included in the labeled set only when *both*
    ``a_lab[i]`` and ``b_lab[i]`` are non-NaN.
    """
    from evalstats.ppi import correct

    mask = ~np.isnan(a_lab) & ~np.isnan(b_lab)
    if mask.sum() == 0:
        raise ValueError(
            "No positions have human labels for both groups in a_lab and b_lab."
        )

    Y_hat_unlab = a - b
    Y_hat_lab   = (a - b)[mask]
    Y_lab       = (a_lab - b_lab)[mask]

    return correct(
        statistic,
        Y_lab=Y_lab,
        Y_hat_lab=Y_hat_lab,
        Y_hat_unlab=Y_hat_unlab,
        X_lab=None,
        X_unlab=None,
        alpha=alpha,
        n_boot=n_boot,
        rng=rng,
        compute_pvalue=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ttest
# ─────────────────────────────────────────────────────────────────────────────

def ttest(
    a,
    b,
    *,
    a_lab=None,
    b_lab=None,
    paired: bool = False,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng=None,
) -> TestResult:
    """Independent-samples or paired t-test with optional PPI correction.

    Uncorrected: ``scipy.stats.ttest_ind`` or ``scipy.stats.ttest_rel``.

    PPI estimand:
      - Independent: ``mean(a) − mean(b)``.
      - Paired: ``mean(a_i − b_i)`` (pairing by array position, same as
        ``scipy.stats.ttest_rel``).

    The corrected p-value tests H₀: θ = 0 via
    ``2 * min(P(θ̂* ≤ 0), P(θ̂* ≥ 0))`` over PPI bootstrap resamples.
    No corrected t-statistic is reported; the bootstrap CI and p-value
    are the primary outputs.

    Parameters
    ----------
    a, b : array-like
        LLM scores for the two groups (or two conditions when ``paired=True``).
    a_lab, b_lab : array-like, optional
        Human labels for the same items as *a* and *b*, same length, with
        ``NaN`` for unlabeled items.  When supplied, PPI correction is applied
        and an alignment report is printed.
    paired : bool
        Use the paired t-test (default False).  When ``True``, ``a[i]`` and
        ``b[i]`` are treated as matched observations (same subject / item).

    Examples
    --------
    >>> # Uncorrected
    >>> result = es.tests.ttest(llm_a, llm_b)
    >>> # PPI-corrected
    >>> result = es.tests.ttest(llm_a, llm_b, a_lab=human_a, b_lab=human_b)
    >>> result.summary()
    """
    a = _coerce(a)
    b = _coerce(b)

    if paired:
        if len(a) != len(b):
            raise ValueError(
                f"paired=True requires equal-length arrays; got {len(a)} vs {len(b)}."
            )
        t_stat, p_val = _scipy_stats.ttest_rel(a, b)
        test_name = "Paired t-test"
    else:
        t_stat, p_val = _scipy_stats.ttest_ind(a, b)
        test_name = "Independent-samples t-test"

    corrected_estimate = corrected_ci = corrected_p = rectifier = None
    n_labeled = n_total = None

    if a_lab is not None or b_lab is not None:
        a_lab, b_lab = _sanitize_ppi_labels(
            a,
            b,
            a_lab,
            b_lab,
            paired=paired,
            label_names=("a_lab", "b_lab"),
        )

        ar = _run_alignment_report(
            np.concatenate([a, b]),
            np.concatenate([a_lab, b_lab]),
        )

        if paired:
            ppi = _ppi_paired_arrays(a, b, a_lab, b_lab, np.mean, alpha, n_boot, rng)
        else:
            def _indep(ya, yb):
                return float(ya.mean() - yb.mean())
            ppi = _ppi_two_sample(a, b, a_lab, b_lab, _indep, alpha, n_boot, rng)

        corrected_estimate = ppi.estimate
        corrected_ci       = (ppi.ci_low, ppi.ci_high)
        corrected_p        = ppi.p_value
        rectifier          = ppi.rectifier
        n_labeled          = ar.n_labeled
        n_total            = ar.n_total

    return TestResult(
        test_name=test_name,
        statistic=float(t_stat),
        p_value=float(p_val),
        corrected_estimate=corrected_estimate,
        corrected_ci=corrected_ci,
        corrected_p_value=corrected_p,
        rectifier=rectifier,
        n_labeled=n_labeled,
        n_total=n_total,
        alpha=alpha,
    )


# ─────────────────────────────────────────────────────────────────────────────
# mannwhitney
# ─────────────────────────────────────────────────────────────────────────────

def mannwhitney(
    x,
    y,
    *,
    x_lab=None,
    y_lab=None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng=None,
) -> TestResult:
    """Mann-Whitney U test with optional PPI correction.

    Uncorrected: ``scipy.stats.mannwhitneyu`` (two-sided).

    PPI estimand: ``P(X > Y) − 0.5``, shifted so the null H₀: P(X > Y) = 0.5
    maps to θ = 0.  The reported ``corrected_estimate`` is P(X > Y) itself
    (shifted back) for interpretability.

    Note: the O(n²) pairwise comparison ``mean(x_i > y_j)`` is recomputed on
    every bootstrap draw.  This is fast at typical PPI scales (N ≲ 2000).

    Parameters
    ----------
    x, y : array-like
        LLM scores for the two groups.
    x_lab, y_lab : array-like, optional
        Human labels for the same items, same length as *x* and *y*,
        with ``NaN`` for unlabeled items.

    Examples
    --------
    >>> result = es.tests.mannwhitney(llm_x, llm_y, x_lab=human_x, y_lab=human_y)
    >>> result.summary()
    """
    x = _coerce(x)
    y = _coerce(y)

    res = _scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
    u_stat, p_val = float(res.statistic), float(res.pvalue)

    corrected_estimate = corrected_ci = corrected_p = rectifier = None
    n_labeled = n_total = None

    if x_lab is not None or y_lab is not None:
        x_lab, y_lab = _sanitize_ppi_labels(
            x,
            y,
            x_lab,
            y_lab,
            paired=False,
            label_names=("x_lab", "y_lab"),
        )

        ar = _run_alignment_report(
            np.concatenate([x, y]),
            np.concatenate([x_lab, y_lab]),
        )

        # Shift by 0.5: null is P(X>Y)=0.5, so estimand θ=P(X>Y)-0.5 has null θ=0
        def _auc_shifted(xa, ya):
            if len(xa) == 0 or len(ya) == 0:
                return 0.0
            return float(np.mean(xa[:, None] > ya[None, :])) - 0.5

        ppi = _ppi_two_sample(x, y, x_lab, y_lab, _auc_shifted, alpha, n_boot, rng)

        corrected_estimate = ppi.estimate + 0.5       # report as P(X>Y)
        corrected_ci       = (ppi.ci_low + 0.5, ppi.ci_high + 0.5)
        corrected_p        = ppi.p_value
        rectifier          = ppi.rectifier
        n_labeled          = ar.n_labeled
        n_total            = ar.n_total

    return TestResult(
        test_name="Mann-Whitney U test",
        statistic=u_stat,
        p_value=p_val,
        corrected_estimate=corrected_estimate,
        corrected_ci=corrected_ci,
        corrected_p_value=corrected_p,
        rectifier=rectifier,
        n_labeled=n_labeled,
        n_total=n_total,
        alpha=alpha,
        extra={"estimand": "P(X > Y)"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# wilcoxon
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon(
    x,
    y,
    *,
    x_lab=None,
    y_lab=None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng=None,
) -> TestResult:
    """Wilcoxon signed-rank test with optional PPI correction.

    Uncorrected: ``scipy.stats.wilcoxon(x, y)`` (two-sided, paired by position).

    PPI estimand: Hodges-Lehmann location estimator — ``median(x_i − y_i)``.
    H₀ is location shift = 0, so the bootstrap p-value
    ``2 * min(P(θ̂* ≤ 0), P(θ̂* ≥ 0))`` is correctly centered at the null.

    Pairing is by array position (``x[i]`` paired with ``y[i]``), exactly as
    in ``scipy.stats.wilcoxon``.  A position enters the labeled set only when
    *both* ``x_lab[i]`` and ``y_lab[i]`` are non-NaN.

    Parameters
    ----------
    x, y : array-like
        Paired LLM scores (must be equal length).
    x_lab, y_lab : array-like, optional
        Human labels for the same items, same length as *x* and *y*,
        with ``NaN`` for unlabeled items.

    Examples
    --------
    >>> result = es.tests.wilcoxon(llm_x, llm_y, x_lab=human_x, y_lab=human_y)
    >>> result.summary()
    """
    x = _coerce(x)
    y = _coerce(y)

    if len(x) != len(y):
        raise ValueError(
            f"Wilcoxon requires equal-length arrays (paired by position); "
            f"got {len(x)} vs {len(y)}."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _scipy_stats.wilcoxon(x, y, alternative="two-sided")
    w_stat, p_val = float(res.statistic), float(res.pvalue)

    corrected_estimate = corrected_ci = corrected_p = rectifier = None
    n_labeled = n_total = None

    if x_lab is not None or y_lab is not None:
        x_lab, y_lab = _sanitize_ppi_labels(
            x,
            y,
            x_lab,
            y_lab,
            paired=True,
            label_names=("x_lab", "y_lab"),
        )

        ar = _run_alignment_report(
            np.concatenate([x, y]),
            np.concatenate([x_lab, y_lab]),
        )

        ppi = _ppi_paired_arrays(x, y, x_lab, y_lab, np.median, alpha, n_boot, rng)

        corrected_estimate = ppi.estimate
        corrected_ci       = (ppi.ci_low, ppi.ci_high)
        corrected_p        = ppi.p_value
        rectifier          = ppi.rectifier
        n_labeled          = ar.n_labeled
        n_total            = ar.n_total

    return TestResult(
        test_name="Wilcoxon signed-rank test",
        statistic=w_stat,
        p_value=p_val,
        corrected_estimate=corrected_estimate,
        corrected_ci=corrected_ci,
        corrected_p_value=corrected_p,
        rectifier=rectifier,
        n_labeled=n_labeled,
        n_total=n_total,
        alpha=alpha,
        extra={"estimand": "Hodges-Lehmann (median of paired diffs)"},
    )
