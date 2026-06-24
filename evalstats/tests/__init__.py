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
from scipy.special import gammaln


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
        labeled subset.  Positive means the LLM under-estimated (human > LLM);
        negative means the LLM over-estimated (LLM > human).
    n_labeled : int or None
    n_total : int or None
    alpha : float
    extra : dict
        Test-specific supplementary values.  For ``ttest``, when PPI
        correction is active, includes ``corrected_effect_size`` /
        ``corrected_effect_size_name`` — the standardized effect size
        (Cohen's d / d_z / Hedges' g_s*) computed with the PPI-corrected
        mean difference as the numerator instead of the raw LLM-judged
        means, so the reported effect size is consistent with
        ``corrected_estimate``.
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

    @staticmethod
    def _fmt_p(p: float, n_boot: int = None) -> str:
        """Return a string of the form '= X' or '< X' for use after 'p'."""
        if p == 0.0 and n_boot is not None:
            return f"< {1 / n_boot:.4f}"
        if p < 0.0001:
            return f"= {p:.2e}"
        return f"= {p:.4f}"

    @staticmethod
    def _bold(text: str) -> str:
        import sys
        if sys.stdout.isatty():
            return f"\033[1m{text}\033[0m"
        return text

    @staticmethod
    def _dim(text: str) -> str:
        import sys
        if sys.stdout.isatty():
            return f"\033[2m{text}\033[0m"
        return text

    @staticmethod
    def _effect_magnitude(d: float) -> str:
        """Cohen's-d-style magnitude bucket for a standardized effect size."""
        d = abs(d)
        return ("trivial" if d < 0.2 else
                "small"   if d < 0.5 else
                "medium"  if d < 0.8 else "large")

    def _ppi_estimand(self) -> str:
        """Plain-language description of the corrected estimand for each test."""
        ex = self.extra
        test = ex.get("_test", "")
        if test == "ttest":
            return ("mean of paired differences (A − B)"
                    if ex.get("paired") else "mean difference (A − B)")
        elif test == "mannwhitney":
            return "P(X > Y)"
        elif test == "wilcoxon":
            return "median of paired differences (X − Y)"
        elif test == "anova":
            return (
                "between-condition variance after removing subject means"
                if ex.get("repeated")
                else "between-group variance"
            )
        return "estimand"

    def _stat_line(self) -> str:
        """Format the primary statistic + p-value for the test (uncorrected)."""
        ex = self.extra
        p_str = self._fmt_p(self.p_value)
        test = ex.get("_test", "")
        if test == "ttest":
            df = ex.get("df")
            df_str = f"({df:.1f})" if df is not None else ""
            return f"t{df_str} = {self.statistic:.3f},  p {p_str}"
        elif test == "mannwhitney":
            return f"U = {self.statistic:.1f},  p {p_str}"
        elif test == "wilcoxon":
            return f"W = {self.statistic:.1f},  p {p_str}"
        elif test == "anova":
            df1 = ex.get("df1")
            df2 = ex.get("df2")
            if df1 is not None and df2 is not None:
                return f"F({df1:.1f}, {df2:.1f}) = {self.statistic:.3f},  p {p_str}"
            return f"F = {self.statistic:.3f},  p {p_str}"
        else:
            return f"stat = {self.statistic:.4f},  p {p_str}"

    def summary(self) -> None:
        """Print a human-readable result suitable for use in a report."""
        ex = self.extra
        ci_pct = int(round((1 - self.alpha) * 100))
        bar = "═" * 60
        B = self._bold

        print(f"\n{self.test_name}")
        print(bar)

        # ── Sample info and effect size ──────────────────────────────
        test = ex.get("_test", "")
        if test == "ttest":
            if ex.get("paired"):
                print(f"  Pairs:    n = {ex['n_a']}")
            else:
                print(f"  Samples:  A (n = {ex['n_a']}),  B (n = {ex['n_b']})")
                print(f"  Means:    A = {ex['mean_a']:.3f} (SD {ex['std_a']:.3f}),  "
                      f"B = {ex['mean_b']:.3f} (SD {ex['std_b']:.3f})")
            d = ex.get("effect_size")
            d_name = ex.get("effect_size_name", "Effect size")
            if d is not None:
                print(f"  Effect:   {d_name} = {d:.3f}  ({self._effect_magnitude(d)})")

        elif test == "mannwhitney":
            print(f"  Samples:  X (n = {ex['n_x']}),  Y (n = {ex['n_y']})")
            p_xgy = ex.get("p_x_gt_y")
            if p_xgy is not None:
                if abs(p_xgy - 0.5) < 0.02:
                    trend = "groups score similarly"
                elif p_xgy > 0.5:
                    trend = "X tends to score higher"
                else:
                    trend = "Y tends to score higher"
                print(f"  Effect:   P(X > Y) = {p_xgy:.3f}  ({trend})")

        elif test == "wilcoxon":
            print(f"  Pairs:    n = {ex['n_pairs']}")
            md = ex.get("median_diff")
            if md is not None:
                direction = ("X tends higher" if md > 0 else
                             "Y tends higher" if md < 0 else "no consistent shift")
                print(f"  Effect:   Median difference (X − Y) = {md:.4f}  ({direction})")

        elif test == "anova":
            if ex.get("repeated"):
                print(
                    f"  Design:   Repeated-measures one-way ANOVA "
                    f"(subjects = {ex['n_subjects']}, conditions = {ex['k_groups']})"
                )
            else:
                print(
                    f"  Design:   One-way ANOVA "
                    f"(groups = {ex['k_groups']}, total n = {ex['n_total_obs']})"
                )
                sizes = ex.get("group_sizes")
                if sizes is not None:
                    print(f"  Group n:  {', '.join(str(int(n)) for n in sizes)}")

            means = ex.get("group_means")
            if means is not None:
                mean_str = ", ".join(f"{m:.3f}" for m in means)
                print(f"  Means:    {mean_str}")

            eta_sq = ex.get("eta_sq")
            if eta_sq is not None:
                mag = ("trivial" if eta_sq < 0.01 else
                       "small" if eta_sq < 0.06 else
                       "medium" if eta_sq < 0.14 else "large")
                print(f"  Effect:   η² = {eta_sq:.3f}  ({mag})")

        print()

        if self.corrected_p_value is not None:
            # ── Uncorrected (first) ──────────────────────────────────
            print(f"  Uncorrected:  {self._stat_line()}  (α = {self.alpha})")
            if self.rectifier is not None and abs(self.rectifier) > 1e-9:
                print(f"  Estimated prediction bias:  δ = {self.rectifier:+.4f}")
            # ── PPI-corrected (last, with leading blank line) ────────
            if self.corrected_estimate is not None:
                lo, hi = self.corrected_ci
                n_boot = ex.get("n_boot")
                cp_str = self._fmt_p(self.corrected_p_value, n_boot=n_boot)
                boot_str = f",  bootstrap samples = {n_boot:,}" if n_boot is not None else ""
                print(f"\n  {B(f'PPI-corrected:  estimate = {self.corrected_estimate:.4f},  {ci_pct}% CI [{lo:.4f}, {hi:.4f}],  p {cp_str}  (α = {self.alpha}{boot_str})')}")
                d_corr = ex.get("corrected_effect_size")
                d_corr_name = ex.get("corrected_effect_size_name", "Effect size")
                if d_corr is not None:
                    print(f"  Effect (PPI-corrected):  {d_corr_name} = {d_corr:.3f}  ({self._effect_magnitude(d_corr)})")
                lab_str = (f"{self.n_labeled} of {self.n_total}"
                           if self.n_labeled is not None else "?")
                note = (f"Correction applied using Prediction-Powered Inference on "
                        f"{self._ppi_estimand()}; bias estimated from {lab_str} human "
                        f"labels and removed via PPI bootstrap procedure (Angelopoulos et al., 2023; "
                        f"Zrnic, 2024).")
                print(f"  {self._dim(note)}")
        else:
            # ── No PPI: single prominent result line ─────────────────
            print(f"  {B(self._stat_line())}  (α = {self.alpha})")

        print(bar)
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


def _hedges_bias_correction(df: float) -> float:
    """Small-sample bias correction J(df) for standardized mean differences.

    Uses the exact gamma-ratio form when df > 1 and a common finite-sample
    approximation as a fallback for very small positive df.
    """
    if not np.isfinite(df) or df <= 0.0:
        return 1.0
    if df > 1.0:
        log_j = gammaln(df / 2.0) - 0.5 * np.log(df / 2.0) - gammaln((df - 1.0) / 2.0)
        return float(np.exp(log_j))
    if df > 0.25:
        return float(1.0 - 3.0 / (4.0 * df - 1.0))
    return 1.0


def _hedges_g_star_from_diff(
    diff: float,
    sd1: float,
    sd2: float,
    n1: int,
    n2: int,
) -> float:
    """Compute Hedges g*_s for unequal-variance two-sample comparisons from
    a precomputed mean difference (``mean1 − mean2``).

    Uses d_av with exact small-sample bias correction J(nu).  Factored out
    of :func:`_hedges_g_star` so the same denominator/bias-correction logic
    can be reused with a PPI-corrected difference in place of the raw one.
    """
    sd_av = float(np.sqrt((sd1**2 + sd2**2) / 2.0))
    if sd_av <= 0.0:
        return 0.0

    d_star = float(diff / sd_av)

    nu_num = (n1 - 1) * (n2 - 1) * (sd1**2 + sd2**2) ** 2
    nu_den = (n2 - 1) * sd1**4 + (n1 - 1) * sd2**4
    nu = float(nu_num / nu_den) if nu_den > 0 else np.nan
    return float(_hedges_bias_correction(nu) * d_star)


def _hedges_g_star(
    mean1: float,
    mean2: float,
    sd1: float,
    sd2: float,
    n1: int,
    n2: int,
) -> float:
    """Compute Hedges g*_s for unequal-variance two-sample comparisons.

    Uses d_av with exact small-sample bias correction J(nu).
    """
    return _hedges_g_star_from_diff(mean1 - mean2, sd1, sd2, n1, n2)


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


def _p_x_gt_y_strict(x: np.ndarray, y: np.ndarray) -> float:
    """Compute P(X > Y) exactly, counting ties as 0.

    Uses sorting + binary search for O((n_x + n_y) log n_y) time and
    O(n_y) extra memory, avoiding O(n_x * n_y) pairwise matrices.
    """
    if len(x) == 0 or len(y) == 0:
        return 0.0

    y_sorted = np.sort(y)
    n_lt = np.searchsorted(y_sorted, x, side="left")
    return float(n_lt.sum() / (len(x) * len(y)))


def _p_x_gt_y_midrank(x: np.ndarray, y: np.ndarray) -> float:
    """Compute P(X > Y) + 0.5·P(X = Y) — the mid-rank (Wilcoxon-Mann-Whitney) convention.

    Equals 0.5 under H₀ (equal distributions) for ANY distribution, including
    discrete/ordinal (Likert, integer-valued).  This is equivalent to scipy's
    U-statistic divided by n_x·n_y, ensuring the PPI estimand θ = P_mid(X>Y) − 0.5
    is centered at 0 under H₀ regardless of tie frequency.
    """
    if len(x) == 0 or len(y) == 0:
        return 0.5

    y_sorted = np.sort(y)
    n_lt = np.searchsorted(y_sorted, x, side="left")   # count Y_j < X_i (strict)
    n_le = np.searchsorted(y_sorted, x, side="right")  # count Y_j ≤ X_i
    # mid-rank: count X_i > Y_j as 1, X_i == Y_j as 0.5
    return float((n_lt.sum() + 0.5 * (n_le - n_lt).sum()) / (len(x) * len(y)))


def _ppi_paired_arrays(
    a: np.ndarray,
    b: np.ndarray,
    a_lab: np.ndarray,
    b_lab: np.ndarray,
    statistic: Callable,
    alpha: float,
    n_boot: int,
    rng,
    rectifier_func=None,
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
        rectifier_func=rectifier_func,
    )


def _sanitize_multigroup_ppi_labels(
    groups: list[np.ndarray],
    groups_lab,
    *,
    repeated: bool,
) -> list[np.ndarray]:
    """Validate sparse labels for multi-group ANOVA wrappers.

    For repeated-measures ANOVA, an effective labeled unit is a subject with
    non-NaN labels in all conditions. For independent one-way ANOVA, effective
    labels are counted across all groups.
    """
    if groups_lab is None:
        raise ValueError("groups_lab must be provided for PPI correction.")
    if len(groups_lab) != len(groups):
        raise ValueError(
            f"groups_lab must have one label array per group "
            f"(got {len(groups_lab)} for {len(groups)} groups)."
        )

    labs = []
    for i, (g, g_lab) in enumerate(zip(groups, groups_lab), start=1):
        lab_arr = _coerce(g_lab)
        if len(lab_arr) != len(g):
            raise ValueError(
                f"groups_lab[{i - 1}] must match group {i} length "
                f"({len(lab_arr)} vs {len(g)})."
            )
        labs.append(lab_arr)

    if repeated:
        overlap = np.ones(len(groups[0]), dtype=bool)
        for lab_arr in labs:
            overlap &= ~np.isnan(lab_arr)
        n_effective = int(overlap.sum())
        min_msg = (
            "At least 15 subjects with labels in all conditions are required "
            f"for repeated-measures PPI ANOVA (found {n_effective})."
        )
        warn_msg = (
            "Only {n} subjects have labels in all conditions. "
            "PPI bootstrap can undercover below 30 labels; consider labeling more subjects."
        )
    else:
        n_effective = int(sum(np.sum(~np.isnan(lab_arr)) for lab_arr in labs))
        min_msg = (
            "At least 15 human labels are required for PPI one-way ANOVA "
            f"(found {n_effective} across all groups)."
        )
        warn_msg = (
            "Only {n} human labels were supplied across all groups. "
            "PPI bootstrap can undercover below 30 labels; consider labeling more items."
        )

    if n_effective < 15:
        raise ValueError(min_msg)
    if n_effective < 30:
        warnings.warn(warn_msg.format(n=n_effective), UserWarning, stacklevel=3)

    return labs


def _anova_between_variance_from_groups(groups: list[np.ndarray]) -> float:
    """Weighted between-group variance; zero iff all group means are equal."""
    nonempty = [g for g in groups if len(g) > 0]
    if len(nonempty) < 2:
        return 0.0

    ns = np.array([len(g) for g in nonempty], dtype=float)
    means = np.array([float(np.mean(g)) for g in nonempty], dtype=float)
    grand = float(np.sum(ns * means) / np.sum(ns))
    return float(np.sum(ns * (means - grand) ** 2) / np.sum(ns))


def _anova_between_variance_from_labeled(Y: np.ndarray, X: np.ndarray, n_groups: int) -> float:
    """Between-group variance from flattened arrays with integer group IDs."""
    groups = [Y[X == gid] for gid in range(n_groups)]
    return _anova_between_variance_from_groups(groups)


def _repeated_condition_variance(matrix: np.ndarray) -> float:
    """Condition variance after removing subject means (row-centering)."""
    if matrix.size == 0:
        return 0.0
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    cond_means = centered.mean(axis=0)
    return float(np.mean((cond_means - cond_means.mean()) ** 2))


def _ppi_anova_independent(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    alpha: float,
    n_boot: int,
    rng,
):
    """PPI correction for independent-groups one-way ANOVA effect estimand."""
    from evalstats.ppi import correct

    k = len(groups)
    masks = [~np.isnan(lab_arr) for lab_arr in groups_lab]

    Y_hat_unlab = np.concatenate(groups)
    X_unlab = np.concatenate([
        np.full(len(g), gid, dtype=int) for gid, g in enumerate(groups)
    ])

    Y_lab = np.concatenate([
        lab_arr[mask] for lab_arr, mask in zip(groups_lab, masks)
    ])
    Y_hat_lab = np.concatenate([
        g[mask] for g, mask in zip(groups, masks)
    ])
    X_lab = np.concatenate([
        np.full(int(mask.sum()), gid, dtype=int) for gid, mask in enumerate(masks)
    ])

    if len(Y_lab) == 0:
        raise ValueError("No labeled items found in groups_lab.")

    def _estimand(Y, X):
        return _anova_between_variance_from_labeled(Y, X, n_groups=k)

    return correct(
        _estimand,
        Y_lab=Y_lab,
        Y_hat_lab=Y_hat_lab,
        Y_hat_unlab=Y_hat_unlab,
        X_lab=X_lab,
        X_unlab=X_unlab,
        alpha=alpha,
        n_boot=n_boot,
        rng=rng,
        compute_pvalue=False,
    )


def _ppi_anova_repeated(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    alpha: float,
    n_boot: int,
    rng,
):
    """PPI correction for repeated-measures one-way ANOVA effect estimand."""
    from evalstats.ppi import correct

    labels_mat = np.column_stack(groups_lab)
    overlap = np.all(~np.isnan(labels_mat), axis=1)
    if overlap.sum() == 0:
        raise ValueError(
            "No subjects have labels in all conditions in groups_lab."
        )

    Y_hat_unlab = np.column_stack(groups)
    Y_hat_lab = Y_hat_unlab[overlap]
    Y_lab = labels_mat[overlap]

    return correct(
        _repeated_condition_variance,
        Y_lab=Y_lab,
        Y_hat_lab=Y_hat_lab,
        Y_hat_unlab=Y_hat_unlab,
        X_lab=None,
        X_unlab=None,
        alpha=alpha,
        n_boot=n_boot,
        rng=rng,
        compute_pvalue=False,
    )


def _ppi_anova_independent_p_value(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    k: int,
) -> Optional[float]:
    """Corrected p-value for independent ANOVA via per-group PPI mean corrections.

    The standard PPI bootstrap p-value is anti-conservative for the between-group
    variance estimand because it's bounded below by zero.  Instead, we apply PPI
    corrections at the level of group means (which ARE a valid, symmetric PPI
    estimand), compute a corrected F-statistic, and use the F-distribution.

    Var[μ̂ᵢ_PPI] = σ²/nᵢ + σ_llm² × (1/n_lab_i − 1/nᵢ)
    where σ² is the true within-group variance and σ_llm² is LLM noise variance.
    Both are estimated from the labeled residuals (llm − human within each group).
    The inflation factor Var[μ̂ᵢ_PPI] / Var[μ̂ᵢ_LLM] scales the F denominator so
    the test is calibrated regardless of how noisy the LLM judge is.
    """
    masks = [~np.isnan(g_lab) for g_lab in groups_lab]
    n_lab_arr = np.array([int(m.sum()) for m in masks], dtype=float)

    if np.any(n_lab_arr == 0):
        return None

    ns = np.array([len(g) for g in groups], dtype=float)
    N = int(ns.sum())

    # PPI-corrected group means: μ̂ᵢ_PPI = mean(llm_all_i) + (mean(human_lab_i) - mean(llm_lab_i))
    corr_means = np.array([
        g.mean() + (g_lab[mask].mean() - g[mask].mean())
        for g, g_lab, mask in zip(groups, groups_lab, masks)
    ])
    grand = (ns * corr_means).sum() / N
    ss_between = float(np.sum(ns * (corr_means - grand) ** 2))

    # Within-group MS from full LLM data  (≈ σ² + σ_llm²)
    ss_within = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    ms_within = ss_within / (N - k)

    # Per-group LLM noise variance σ_llm_i² from labeled residuals (llm − human)
    sigma_llm_sq_per_group = np.zeros(k)
    for i, (g, g_lab, mask) in enumerate(zip(groups, groups_lab, masks)):
        if mask.sum() > 1:
            sigma_llm_sq_per_group[i] = float(np.var(g[mask] - g_lab[mask], ddof=1))

    # n_i-weighted average noise (matches how ms_within pools group residuals)
    sigma_llm_sq_weighted = float(np.dot(ns, sigma_llm_sq_per_group) / N)
    sigma_sq = max(ms_within - sigma_llm_sq_weighted, 0.0)

    # Per-group inflation: Var[μ̂ᵢ_PPI] / Var[μ̂ᵢ_LLM]
    # = (σ² + σ_llm_i² × (nᵢ/n_lab_i − 1)) / ms_within
    # Weights: (N−nᵢ)/(N(k−1)) derived from E[SS_between_corr] = Σᵢ nᵢ(N−nᵢ)/N · Var[μ̂ᵢ]
    # For balanced groups these equal nᵢ/N (same as old formula).
    inflation_per_group = (sigma_sq + sigma_llm_sq_per_group * (ns / n_lab_arr - 1.0)) / ms_within
    w = (N - ns) / (N * (k - 1))
    inflation = float(np.dot(w, inflation_per_group))
    inflation = max(inflation, 1e-6)

    f_corr = (ss_between / (k - 1)) / (ms_within * inflation)
    if f_corr <= 0.0:
        return 1.0
    return float(_scipy_stats.f.sf(f_corr, dfn=k - 1, dfd=N - k))


def _ppi_anova_repeated_p_value(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    k: int,
) -> Optional[float]:
    """Corrected p-value for repeated-measures ANOVA via per-condition PPI corrections.

    Mirror of _ppi_anova_independent_p_value for the within-subjects design.
    Corrections are applied to condition means after removing each subject's
    mean (matching the repeated-measures F-test).

        The inflation factor accounts for rectifier uncertainty.

        To remain calibrated when LLM noise differs across conditions (heteroskedastic)
        or has cross-condition covariance, we estimate the effective noise variance in
        condition-contrast space from the labeled residual covariance:

            Σ̂_η = Cov(LLM − human) across conditions on labeled subjects,
            P = I − (1/k)11ᵀ,
            σ̂²_eff = tr(P Σ̂_η P) / (k−1).

        This is the exact variance scale that enters repeated-measures condition MS.
        The inflation then uses:

            inflation = [σ̂²_true + σ̂²_eff × (n_subjects/n_lab − 1)] / ms_residual,
            where σ̂²_true = max(ms_residual − σ̂²_eff, 0).
    """
    n_subjects = len(groups[0])
    labels_mat = np.column_stack(groups_lab)
    overlap = np.all(~np.isnan(labels_mat), axis=1)
    n_lab = int(overlap.sum())

    if n_lab == 0:
        return None

    llm_mat = np.column_stack(groups)  # (n_subjects, k)

    # Center by subject mean to isolate condition effects
    centered_llm_all = llm_mat - llm_mat.mean(axis=1, keepdims=True)
    cond_means_llm = centered_llm_all.mean(axis=0)  # (k,)

    # Per-condition PPI corrections estimated from labeled subjects
    llm_lab = llm_mat[overlap]
    human_lab = labels_mat[overlap]
    centered_llm_lab = llm_lab - llm_lab.mean(axis=1, keepdims=True)
    centered_human_lab = human_lab - human_lab.mean(axis=1, keepdims=True)
    delta = centered_human_lab.mean(axis=0) - centered_llm_lab.mean(axis=0)

    cond_means_ppi = cond_means_llm + delta
    grand_ppi = cond_means_ppi.mean()
    ss_condition_corr = float(n_subjects * np.sum((cond_means_ppi - grand_ppi) ** 2))

    # MS_residual (subject × condition interaction) from full LLM data
    grand_llm = llm_mat.mean()
    ss_total = float(np.sum((llm_mat - grand_llm) ** 2))
    ss_subjects = float(k * np.sum((llm_mat.mean(axis=1) - grand_llm) ** 2))
    ss_condition_llm = float(n_subjects * np.sum((cond_means_llm - cond_means_llm.mean()) ** 2))
    ss_residual = ss_total - ss_subjects - ss_condition_llm
    df_residual = (n_subjects - 1) * (k - 1)
    ms_residual = max(ss_residual / df_residual, 1e-12)

    # Estimate effective LLM noise variance in contrast space from labeled
    # condition-residual covariance. This handles heteroskedastic and correlated
    # condition noise more robustly than a homoskedastic scalar proxy.
    noise_lab = llm_lab - human_lab  # (n_lab, k)
    if n_lab > 1:
        cov_eta = np.cov(noise_lab, rowvar=False, ddof=1)
        cov_eta = np.atleast_2d(cov_eta)
        P = np.eye(k) - np.ones((k, k), dtype=float) / float(k)
        sigma_llm_sq = float(np.trace(P @ cov_eta @ P) / float(k - 1))
        sigma_llm_sq = max(sigma_llm_sq, 0.0)
    else:
        sigma_llm_sq = 0.0

    sigma_res_true_sq = max(ms_residual - sigma_llm_sq, 0.0)

    # inflation = [σ_res_true² + σ_η² × (n_subjects/n_lab − 1)] / ms_residual
    # Under H₀: E[ms_cond_ppi] = σ_ε² + σ_η²(n/n_lab−1) = ms_residual × inflation
    if ms_residual > 0:
        inflation = (sigma_res_true_sq + sigma_llm_sq * (n_subjects / n_lab - 1.0)) / ms_residual
    else:
        inflation = 1.0
    inflation = max(inflation, 1e-6)

    f_corr = (ss_condition_corr / (k - 1)) / (ms_residual * inflation)
    if f_corr <= 0.0:
        return 1.0

    # Finite labeled overlap adds estimation uncertainty to the rectifier.
    # Use a conservative effective denominator df capped by labeled contrasts.
    # This improves Type I calibration in sparse-label repeated settings.
    df_residual_eff = min(df_residual, max((n_lab - 1) * (k - 1), 1))
    return float(_scipy_stats.f.sf(f_corr, dfn=k - 1, dfd=df_residual_eff))


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
    equal_var: bool = False,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng=None,
    print_result: bool = True,
) -> TestResult:
    """Independent-samples or paired t-test with optional PPI correction.

    Uncorrected: ``scipy.stats.ttest_ind`` (with ``equal_var``) or ``scipy.stats.ttest_rel``.

    PPI estimand:
      - Independent: ``mean(a) − mean(b)``.
      - Paired: ``mean(a_i − b_i)`` (pairing by array position, same as
        ``scipy.stats.ttest_rel``).

    The corrected p-value tests H₀: θ = 0 via
    ``2 * min(P(θ̂* ≤ 0), P(θ̂* ≥ 0))`` over PPI bootstrap resamples.
    No corrected t-statistic is reported; the bootstrap CI and p-value
    are the primary outputs.

    When human labels are supplied, the reported effect size (printed as
    "Effect (PPI-corrected)") is also recomputed with the PPI-corrected
    mean difference as its numerator, rather than the raw, uncorrected
    LLM-judged means — keeping the effect size consistent with the
    corrected estimate. The standard deviations in the denominator are
    left uncorrected, since PPI only targets the mean estimand here.

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
    equal_var : bool
        Assume equal variances (Student's t-test) when ``True``; use the
        Welch–Satterthwaite approximation (Welch's t-test) when ``False``
        (default).  Welch's is recommended for most HCI comparisons because
        group variances are rarely equal.  Ignored when ``paired=True``.
    print_result : bool
        Print a summary table to stdout (default True).  Pass ``False`` to
        suppress output when calling from automated pipelines.

    Examples
    --------
    >>> result = es.tests.ttest(llm_a, llm_b)                          # prints
    >>> result = es.tests.ttest(llm_a, llm_b, print_result=False)      # silent
    >>> result = es.tests.ttest(llm_a, llm_b, a_lab=human_a, b_lab=human_b)
    """
    a = _coerce(a)
    b = _coerce(b)

    if paired:
        if len(a) != len(b):
            raise ValueError(
                f"paired=True requires equal-length arrays; got {len(a)} vs {len(b)}."
            )
        _res = _scipy_stats.ttest_rel(a, b)
        test_name = "Paired t-test"
    else:
        _res = _scipy_stats.ttest_ind(a, b, equal_var=equal_var)
        test_name = "Independent-samples t-test" if equal_var else "Welch's t-test"

    t_stat, p_val = float(_res.statistic), float(_res.pvalue)
    df = float(getattr(_res, "df", np.nan))

    # Effect size and descriptive stats
    if paired:
        diffs = a - b
        sd_diffs = float(np.std(diffs, ddof=1))
        # Cohen’s d_z: mean difference / SD of differences (standard for paired designs)
        effect_size_val = float(np.mean(diffs) / sd_diffs) if sd_diffs > 0 else 0.0
        extra = {
            "_test": "ttest",
            "paired": True,
            "n_a": len(a),
            "df": df if np.isfinite(df) else None,
            "effect_size": effect_size_val,
            "effect_size_name": "Cohen’s d_z",
            "n_boot": n_boot,
        }
    else:
        n_a, n_b = len(a), len(b)
        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        std_a, std_b = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
        if equal_var:
            # Student’s: Cohen’s d using pooled SD weighted by df
            denom_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
            effect_size_val = float((mean_a - mean_b) / denom_std) if denom_std > 0 else 0.0
            effect_size_name = "Cohen’s d"
        else:
            # Welch’s: Use Hedges’ g*_s instead (recommended by Delacre et al. 2021, backed up by simulation studies).
            effect_size_val = _hedges_g_star(
                mean_a, mean_b, std_a, std_b, n_a, n_b
            )
            effect_size_name = "Hedges’ g_s*"
        extra = {
            "_test": "ttest",
            "paired": False,
            "equal_var": equal_var,
            "n_a": n_a,
            "n_b": n_b,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": std_a,
            "std_b": std_b,
            "effect_size": effect_size_val,
            "effect_size_name": effect_size_name,
            "df": df if np.isfinite(df) else None,
            "n_boot": n_boot,
        }

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

        if paired:
            # Align on item-level differences: counts overlapping pairs, not individual labels
            _pair_mask = ~np.isnan(a_lab) & ~np.isnan(b_lab)
            ar = _run_alignment_report(
                a - b,
                np.where(_pair_mask, a_lab - b_lab, np.nan),
            )
            ppi = _ppi_paired_arrays(a, b, a_lab, b_lab, np.mean, alpha, n_boot, rng)
        else:
            ar = _run_alignment_report(
                np.concatenate([a, b]),
                np.concatenate([a_lab, b_lab]),
            )
            def _indep(ya, yb):
                return float(ya.mean() - yb.mean())
            ppi = _ppi_two_sample(a, b, a_lab, b_lab, _indep, alpha, n_boot, rng)

        corrected_estimate = ppi.estimate
        corrected_ci       = (ppi.ci_low, ppi.ci_high)
        corrected_p        = ppi.p_value
        rectifier          = ppi.rectifier
        n_labeled          = ar.n_labeled
        n_total            = ar.n_total

        # PPI-corrected effect size: re-use the same denominator (sample SDs
        # are unaffected by the mean-bias correction) but with the
        # PPI-corrected mean difference as the numerator, so the reported
        # effect size matches the corrected estimate rather than the raw,
        # uncorrected LLM-judged means.
        if paired:
            extra["corrected_effect_size"] = (
                float(corrected_estimate / sd_diffs) if sd_diffs > 0 else 0.0
            )
            extra["corrected_effect_size_name"] = "Cohen’s d_z"
        elif equal_var:
            extra["corrected_effect_size"] = (
                float(corrected_estimate / denom_std) if denom_std > 0 else 0.0
            )
            extra["corrected_effect_size_name"] = "Cohen’s d"
        else:
            extra["corrected_effect_size"] = _hedges_g_star_from_diff(
                corrected_estimate, std_a, std_b, n_a, n_b
            )
            extra["corrected_effect_size_name"] = "Hedges’ g_s*"

    result = TestResult(
        test_name=test_name,
        statistic=t_stat,
        p_value=p_val,
        corrected_estimate=corrected_estimate,
        corrected_ci=corrected_ci,
        corrected_p_value=corrected_p,
        rectifier=rectifier,
        n_labeled=n_labeled,
        n_total=n_total,
        alpha=alpha,
        extra=extra,
    )
    if print_result:
        result.summary()
    return result


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
    print_result: bool = True,
) -> TestResult:
    """Mann-Whitney U test with optional PPI correction.

    Uncorrected: ``scipy.stats.mannwhitneyu`` (two-sided).

    PPI estimand: ``P_mid(X > Y) − 0.5``, where ``P_mid(X > Y) = P(X > Y) + 0.5·P(X = Y)``
    is the mid-rank (Wilcoxon-Mann-Whitney) convention.  This equals 0.5 under H₀ for
    any distribution — including discrete/ordinal (Likert, integer-valued) — so the PPI
    null θ = 0 is always correct regardless of tie frequency.  The reported
    ``corrected_estimate`` is ``P_mid(X > Y)`` (shifted back) for interpretability, and
    is consistent with scipy's U-statistic / (n_x·n_y).

    Note: computed with sorting + binary search, O((n_x + n_y) log n_y).

    Parameters
    ----------
    x, y : array-like
        LLM scores for the two groups.
    x_lab, y_lab : array-like, optional
        Human labels for the same items, same length as *x* and *y*,
        with ``NaN`` for unlabeled items.
    print_result : bool
        Print a summary table to stdout (default True).  Pass ``False`` to
        suppress output when calling from automated pipelines.

    Examples
    --------
    >>> result = es.tests.mannwhitney(llm_x, llm_y)                    # prints
    >>> result = es.tests.mannwhitney(llm_x, llm_y, print_result=False) # silent
    >>> result = es.tests.mannwhitney(llm_x, llm_y, x_lab=human_x, y_lab=human_y)
    """
    x = _coerce(x)
    y = _coerce(y)

    res = _scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
    u_stat, p_val = float(res.statistic), float(res.pvalue)

    n_x, n_y = len(x), len(y)
    p_x_gt_y = u_stat / (n_x * n_y)  # common language effect size
    extra = {
        "_test": "mannwhitney",
        "n_x": n_x,
        "n_y": n_y,
        "p_x_gt_y": p_x_gt_y,
        "estimand": "P(X > Y)",
        "n_boot": n_boot,
    }

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

        # Mid-rank convention: P_mid(X>Y) = P(X>Y) + 0.5·P(X=Y) = 0.5 under H₀ for
        # any distribution (including discrete/Likert). Estimand θ = P_mid - 0.5 → 0.
        def _auc_shifted(xa, ya):
            return _p_x_gt_y_midrank(xa, ya) - 0.5

        ppi = _ppi_two_sample(x, y, x_lab, y_lab, _auc_shifted, alpha, n_boot, rng)

        corrected_estimate = ppi.estimate + 0.5       # report as P(X>Y)
        corrected_ci       = (ppi.ci_low + 0.5, ppi.ci_high + 0.5)
        corrected_p        = ppi.p_value
        rectifier          = ppi.rectifier
        n_labeled          = ar.n_labeled
        n_total            = ar.n_total

    result = TestResult(
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
        extra=extra,
    )
    if print_result:
        result.summary()
    return result


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
    print_result: bool = True,
) -> TestResult:
    """Wilcoxon signed-rank test with optional PPI correction.

    Uncorrected: ``scipy.stats.wilcoxon(x, y)`` (two-sided, paired by position).

    PPI estimand: ``median(x_i − y_i)`` for the main term (LLM diffs),
    with ``mean(x_lab_i − x_hat_i) − mean(y_lab_i − y_hat_i)`` as the
    rectifier.  Using the mean for the rectifier gives a well-calibrated
    bootstrap (the bootstrap of a mean is exact); using the median would
    overestimate the rectifier SE by ~2×, causing conservative Type I errors.
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
    print_result : bool
        Print a summary table to stdout (default True).  Pass ``False`` to
        suppress output when calling from automated pipelines.

    Examples
    --------
    >>> result = es.tests.wilcoxon(llm_x, llm_y)                      # prints
    >>> result = es.tests.wilcoxon(llm_x, llm_y, print_result=False)  # silent
    >>> result = es.tests.wilcoxon(llm_x, llm_y, x_lab=human_x, y_lab=human_y)
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

    diffs = x - y
    median_diff = float(np.median(diffs[diffs != 0])) if np.any(diffs != 0) else 0.0
    extra = {
        "_test": "wilcoxon",
        "n_pairs": len(x),
        "median_diff": median_diff,
        "n_boot": n_boot,
    }

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

        # Align on item-level differences: counts overlapping pairs, not individual labels
        _pair_mask = ~np.isnan(x_lab) & ~np.isnan(y_lab)
        ar = _run_alignment_report(
            x - y,
            np.where(_pair_mask, x_lab - y_lab, np.nan),
        )

        # Use np.mean for the rectifier (instead of np.median) to get a
        # better-calibrated bootstrap: the bootstrap of mean(D_lab) is exact,
        # while the bootstrap of median(Y_lab) − median(Y_hat_lab) overestimates
        # the variance by ~2× for typical n_lab, causing conservative Type I errors.
        # Both estimands converge to 0 under H₀; the hybrid is asymptotically valid.
        ppi = _ppi_paired_arrays(x, y, x_lab, y_lab, np.median, alpha, n_boot, rng,
                                 rectifier_func=np.mean)

        corrected_estimate = ppi.estimate
        corrected_ci       = (ppi.ci_low, ppi.ci_high)
        corrected_p        = ppi.p_value
        rectifier          = ppi.rectifier
        n_labeled          = ar.n_labeled
        n_total            = ar.n_total

    result = TestResult(
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
        extra=extra,
    )
    if print_result:
        result.summary()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# anova_oneway
# ─────────────────────────────────────────────────────────────────────────────

def anova_oneway(
    *groups,
    groups_lab=None,
    repeated: bool = False,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng=None,
    print_result: bool = True,
) -> TestResult:
    """One-way ANOVA (independent or repeated-measures) with optional PPI.

    Uncorrected:
      - ``repeated=False``: ``scipy.stats.f_oneway(*groups)``
      - ``repeated=True``: repeated-measures one-way ANOVA via
        ``statsmodels.stats.anova.AnovaRM``

    PPI estimand:
      - Independent one-way: weighted between-group variance.
      - Repeated one-way: between-condition variance after removing each
        subject's mean (row-centering).

    Parameters
    ----------
    *groups : array-like
        Two or more score arrays, one per group/condition.
    groups_lab : sequence[array-like], optional
        Sparse human labels aligned with each group. Must have one array per
        group, same lengths, with NaN for unlabeled items.
    repeated : bool
        If True, run repeated-measures one-way ANOVA with pairing by index.
    print_result : bool
        Print a summary table to stdout (default True).  Pass ``False`` to
        suppress output when calling from automated pipelines.

    Examples
    --------
    >>> result = es.tests.anova_oneway(g1, g2, g3)                             # prints
    >>> result = es.tests.anova_oneway(g1, g2, g3, print_result=False)         # silent
    >>> result = es.tests.anova_oneway(g1, g2, groups_lab=[lab1, lab2, lab3])  # PPI
    """
    if len(groups) < 2:
        raise ValueError("anova_oneway requires at least two groups.")

    groups = [_coerce(g) for g in groups]
    k = len(groups)

    if repeated:
        n_subjects = len(groups[0])
        for i, g in enumerate(groups[1:], start=2):
            if len(g) != n_subjects:
                raise ValueError(
                    "repeated=True requires equal-length arrays across all "
                    f"conditions; condition 1 has {n_subjects}, condition {i} has {len(g)}."
                )

        try:
            from statsmodels.stats.anova import AnovaRM
        except Exception as e:
            raise ImportError(
                "Repeated-measures ANOVA requires statsmodels. "
                "Install statsmodels to use anova_oneway(..., repeated=True)."
            ) from e

        stacked = np.column_stack(groups)
        df_long = pd.DataFrame(
            {
                "subject": np.repeat(np.arange(n_subjects), k),
                "condition": np.tile(np.arange(k), n_subjects),
                "score": stacked.reshape(-1),
            }
        )
        rm = AnovaRM(df_long, depvar="score", subject="subject", within=["condition"]).fit()
        row = rm.anova_table.iloc[0]

        f_stat = float(row["F Value"])
        p_val = float(row["Pr > F"])
        df1 = float(row["Num DF"])
        df2 = float(row["Den DF"])
        eta_sq = float((f_stat * df1) / (f_stat * df1 + df2)) if (f_stat * df1 + df2) > 0 else 0.0

        extra = {
            "_test": "anova",
            "repeated": True,
            "k_groups": k,
            "n_subjects": n_subjects,
            "n_total_obs": n_subjects * k,
            "group_means": [float(np.mean(g)) for g in groups],
            "df1": df1,
            "df2": df2,
            "eta_sq": eta_sq,
            "n_boot": n_boot,
        }
        test_name = "Repeated-measures one-way ANOVA"
    else:
        res = _scipy_stats.f_oneway(*groups)
        f_stat, p_val = float(res.statistic), float(res.pvalue)

        group_sizes = [len(g) for g in groups]
        n_obs = int(sum(group_sizes))
        df1 = float(k - 1)
        df2 = float(n_obs - k)

        all_scores = np.concatenate(groups)
        grand_mean = np.mean(all_scores)
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2
            for g in groups
        )
        ss_total = np.sum((all_scores - grand_mean) ** 2)

        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

        extra = {
            "_test": "anova",
            "repeated": False,
            "k_groups": k,
            "n_total_obs": n_obs,
            "group_sizes": group_sizes,
            "group_means": [float(np.mean(g)) for g in groups],
            "df1": df1,
            "df2": df2,
            "eta_sq": eta_sq,
            "n_boot": n_boot,
        }
        test_name = "One-way ANOVA"

    corrected_estimate = corrected_ci = corrected_p = rectifier = None
    n_labeled = n_total = None

    if groups_lab is not None:
        groups_lab = _sanitize_multigroup_ppi_labels(
            groups,
            groups_lab,
            repeated=repeated,
        )

        llm_all = np.concatenate(groups)
        human_sparse = np.concatenate(groups_lab)
        ar = _run_alignment_report(llm_all, human_sparse)

        if repeated:
            ppi = _ppi_anova_repeated(groups, groups_lab, alpha, n_boot, rng)
            corrected_p = _ppi_anova_repeated_p_value(groups, groups_lab, k)
        else:
            ppi = _ppi_anova_independent(groups, groups_lab, alpha, n_boot, rng)
            corrected_p = _ppi_anova_independent_p_value(groups, groups_lab, k)

        corrected_estimate = ppi.estimate
        corrected_ci = (ppi.ci_low, ppi.ci_high)
        rectifier = ppi.rectifier
        n_labeled = ar.n_labeled
        n_total = ar.n_total

    result = TestResult(
        test_name=test_name,
        statistic=f_stat,
        p_value=p_val,
        corrected_estimate=corrected_estimate,
        corrected_ci=corrected_ci,
        corrected_p_value=corrected_p,
        rectifier=rectifier,
        n_labeled=n_labeled,
        n_total=n_total,
        alpha=alpha,
        extra=extra,
    )
    if print_result:
        result.summary()
    return result
