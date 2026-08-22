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

**Which items get a human label must be chosen uniformly at random.** This
is a hard requirement, not a tip: PPI's rectifier assumes the labeled subset
is representative of the full population it's correcting. If labeling
targets specific items instead — e.g. "double-check the borderline or
highest-scoring responses," a common real-world review practice — that is
missing-not-at-random (MNAR) selection on the outcome, and every PPI
correction in this module can stay badly miscalibrated (30–65% false-
positive rate at nominal 5%, in simulation) **no matter how many items you
label**. Unlike ordinary small-sample noise, this does not shrink as
n_lab grows — it was confirmed to persist from n_lab=15 up through n_lab=300
out of N=400 (see ``simulations/harness/cases/pvalues.py --mode ppi``'s
MNAR-labeling sweep). A stratified rectifier was prototyped as a mitigation
and found to trade away calibration on the common (correctly random-sampled)
case for a partial improvement here, so it was not adopted — random sampling
of the labeled subset is the only fix. See :func:`evalstats.ppi.correct`'s
docstring for the full analysis.

Alignment report
----------------
When human labels are supplied, ``judge_alignment()`` is called
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

from evalstats.ppi import paired_walsh_midrank_theta


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
    lam : float or None
        The PPI++ power-tuning weight λ actually used, when the test was
        called with ``power_tune=True`` (see :func:`evalstats.ppi.correct`'s
        ``power_tune`` parameter). ``None`` when ``power_tune=False`` (the
        default -- λ is then implicitly 1.0) or when no human labels were
        supplied at all.
    n_labeled : int or None
    n_total : int or None
    alpha : float
    extra : dict
        Test-specific supplementary values.  For ``ttest`` and ``friedman``,
        when PPI correction is active, includes ``corrected_effect_size`` /
        ``corrected_effect_size_name`` — the standardized effect size
        (Cohen's d / d_z / Hedges' g_s* for ``ttest``; Kendall's W for
        ``friedman``) computed with the PPI-corrected estimate as the
        numerator instead of the raw LLM-judged values, so the reported
        effect size is consistent with ``corrected_estimate``.
    """

    test_name: str
    statistic: float
    p_value: float
    corrected_estimate: Optional[float] = None
    corrected_ci: Optional[tuple] = None
    corrected_p_value: Optional[float] = None
    corrected_statistic: Optional[float] = None
    rectifier: Optional[float] = None
    lam: Optional[float] = None
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
        from ..config import supports_ansi_color
        if supports_ansi_color():
            return f"\033[1m{text}\033[0m"
        return text

    @staticmethod
    def _dim(text: str) -> str:
        from ..config import supports_ansi_color
        if supports_ansi_color():
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
            return "Walsh-average midrank-sign statistic (Hodges-Lehmann-style, centered at 0 under H0)"
        elif test == "anova":
            return (
                "between-condition variance after removing subject means"
                if ex.get("repeated")
                else "between-group variance"
            )
        elif test == "friedman":
            return "variance of per-condition mean within-subject ranks"
        elif test == "kruskal":
            return "pairwise stochastic-dominance effects P_mid(group_a > group_b) across all group pairs"
        elif test == "lmm":
            return "variance of template fixed-effect means (score ~ template + (1|input))"
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
        elif test == "friedman":
            df = ex.get("df")
            df_str = f"({df:.0f})" if df is not None else ""
            return f"χ²{df_str} = {self.statistic:.3f},  p {p_str}"
        elif test == "kruskal":
            df = ex.get("df")
            df_str = f"({df:.0f})" if df is not None else ""
            return f"H{df_str} = {self.statistic:.3f},  p {p_str}"
        elif test == "lmm":
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

        elif test == "friedman":
            print(
                f"  Design:   Friedman test "
                f"(subjects = {ex['n_subjects']}, conditions = {ex['k_groups']})"
            )
            means = ex.get("group_means")
            if means is not None:
                mean_str = ", ".join(f"{m:.3f}" for m in means)
                print(f"  Means:    {mean_str}")

            w = ex.get("effect_size")
            w_name = ex.get("effect_size_name", "Effect size")
            if w is not None:
                print(f"  Effect:   {w_name} = {w:.3f}  ({self._effect_magnitude(w)})")

        elif test == "kruskal":
            print(
                f"  Design:   Kruskal-Wallis test "
                f"(groups = {ex['k_groups']}, total n = {ex['n_total_obs']})"
            )
            sizes = ex.get("group_sizes")
            if sizes is not None:
                print(f"  Group n:  {', '.join(str(int(n)) for n in sizes)}")
            medians = ex.get("group_medians")
            if medians is not None:
                median_str = ", ".join(f"{m:.3f}" for m in medians)
                print(f"  Medians:  {median_str}")

            d = ex.get("effect_size")
            d_name = ex.get("effect_size_name", "Effect size")
            if d is not None:
                print(f"  Effect:   {d_name} = {d:.3f}  ({self._effect_magnitude(d)})")

        elif test == "lmm":
            print(f"  Design:   {ex['k_groups']} groups, {ex['n_inputs']} inputs")
            means = ex.get("group_means")
            if means is not None:
                mean_str = ", ".join(f"{m:.3f}" for m in means)
                print(f"  Means:    {mean_str}")
            icc = ex.get("icc")
            if icc is not None:
                print(
                    f"  Variance: ICC = {icc:.3f}  "
                    f"(σ_input = {ex['sigma_input']:.3f}, σ_resid = {ex['sigma_resid']:.3f})"
                )
            d = ex.get("effect_size")
            d_name = ex.get("effect_size_name", "Effect size")
            if d is not None:
                print(f"  Effect:   {d_name} = {d:.5f}")

        print()

        if self.corrected_p_value is not None:
            # ── Uncorrected (first) ──────────────────────────────────
            print(f"  Uncorrected:  {self._stat_line()}  (α = {self.alpha})")
            if self.rectifier is not None and abs(self.rectifier) > 1e-9:
                print(f"  Estimated prediction bias:  δ = {self.rectifier:+.4f}")
            if self.lam is not None:
                print(f"  PPI++ power-tuning weight:  λ = {self.lam:.3f}")
            # ── PPI-corrected (last, with leading blank line) ────────
            if self.corrected_estimate is not None:
                n_boot = ex.get("n_boot")
                cp_str = self._fmt_p(self.corrected_p_value, n_boot=n_boot)
                if self.corrected_ci is not None:
                    lo, hi = self.corrected_ci
                    ci_str = f",  {ci_pct}% CI [{lo:.4f}, {hi:.4f}]"
                else:
                    ci_str = "  (no CI: quadratic-form estimand, not valid under a symmetric Wald CI)"
                boot_str = f",  bootstrap samples = {n_boot:,}" if n_boot is not None else ""
                print(f"\n  {B(f'PPI-corrected:  estimate = {self.corrected_estimate:.4f}{ci_str},  p {cp_str}  (α = {self.alpha}{boot_str})')}")
                d_corr = ex.get("corrected_effect_size")
                d_corr_name = ex.get("corrected_effect_size_name", "Effect size")
                if d_corr is not None and self.corrected_ci is not None:
                    print(f"  Effect (PPI-corrected):  {d_corr_name} = {d_corr:.3f}  ({self._effect_magnitude(d_corr)})")
                pairwise = ex.get("pairwise_effects")
                if pairwise:
                    print(f"\n  Pairwise PPI-corrected effects (P_mid(a > b), {ci_pct}% CI):")
                    for (a, b), info in pairwise.items():
                        plo, phi = info["ci"]
                        print(f"    group {a} vs group {b}:  θ = {info['theta']:.3f}  [{plo:.3f}, {phi:.3f}]")
                fixed_effects = ex.get("fixed_effects")
                if fixed_effects and any("corrected_estimate" in info for info in fixed_effects.values()):
                    print(f"\n  PPI-corrected fixed effects ({ci_pct}% CI):")
                    for name, info in fixed_effects.items():
                        if "corrected_estimate" not in info:
                            continue
                        clo, chi = info["corrected_ci"]
                        print(f"    {name}:  β = {info['corrected_estimate']:.4f}  [{clo:.4f}, {chi:.4f}]")
                lab_str = (f"{self.n_labeled} human labels of {self.n_total} total items"
                           if self.n_labeled is not None else "?")
                correction_method = (
                    "a closed-form rectified-GLS solve with cluster-robust sandwich SEs"
                    if test == "lmm" else "the PPI bootstrap procedure"
                )
                note = (f"Correction applied using Prediction-Powered Inference on "
                        f"{self._ppi_estimand()}; bias estimated from {lab_str} "
                        f"and removed via {correction_method} (Angelopoulos et al., 2023; "
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
    from evalstats.alignment import judge_alignment
    from evalstats.loader import EvalResults, _detect_score_type

    _LLM = "__llm__"
    _HUM = "__human__"
    df = pd.DataFrame({_LLM: llm_all, _HUM: human_sparse})

    class _EvalStub(EvalResults):
        # judge_alignment() dispatches on isinstance(x, EvalResults) -- a
        # real (if minimal) subclass rather than a duck-typed lookalike, so
        # it doesn't need EvalResults' full constructor args (metric_cols/
        # factor_cols). _judge_alignment_from_evaldata reads ._df,
        # ._score_types, and ._col (role -> actual col name, used to exclude
        # structural model/item/run columns from covariate checks); this stub
        # has no structural columns at all, so ._col is simply empty -- every
        # role lookup correctly resolves to None via .get().
        def __init__(self):
            self._df = df
            self._score_types = {_LLM: _detect_score_type(pd.Series(llm_all))}
            self._col = {}

    ar = judge_alignment(_EvalStub(), llm_metric=_LLM, human_groundtruth=_HUM)
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

    This only checks COUNT, not how the labeled subset was chosen. Count
    alone cannot catch the more serious failure mode: labels that were
    selected non-randomly (e.g. always double-checking the borderline/
    highest-scoring items) can leave PPI badly miscalibrated regardless of
    n_lab -- see the module docstring's "Which items get a human label"
    section and :func:`evalstats.ppi.correct`'s docstring for why this is a
    hard requirement, not a count to satisfy.
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
            "PPI bootstrap can undercover below 30 labels; consider labeling more items. "
            "This assumes those labels were chosen uniformly at random -- if labeling "
            "targeted specific items (e.g. borderline/highest-scoring), more labels will "
            "NOT fix miscalibration; see this module's docstring."
        )
    else:
        n_effective = int(np.sum(~np.isnan(a_lab)) + np.sum(~np.isnan(b_lab)))
        min_msg = (
            "At least 15 human labels are required for PPI "
            f"(found {n_effective} across {a_name}/{b_name})."
        )
        warn_msg = (
            "Only {n} human labels were supplied across both groups. "
            "PPI bootstrap can undercover below 30 labels; consider labeling more items. "
            "This assumes those labels were chosen uniformly at random -- if labeling "
            "targeted specific items (e.g. borderline/highest-scoring), more labels will "
            "NOT fix miscalibration; see this module's docstring."
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
    power_tune: bool = True,
):
    """PPI correction for a two-sample scalar estimand.

    ``estimator_func(ya, yb) → float`` receives the group-A and group-B
    sub-arrays and returns a scalar.

    Internally uses integer group labels (0 = A, 1 = B) so group identity
    survives bootstrap resampling without string comparison.

    Delegates to :func:`evalstats.ppi.correct`'s single global rectifier --
    see that docstring's random-sampling requirement. This function does NOT
    guard against non-random label selection (MNAR); it assumes the caller's
    a_lab/b_lab were chosen uniformly at random from a/b.

    ``power_tune`` is forwarded to :func:`evalstats.ppi.correct` as-is (see
    its docstring) -- defaults to True (PPI++ power-tuning); pass False to
    reproduce the original (2023) fixed-λ=1 PPI estimator exactly.
    """
    from evalstats.ppi import correct

    mask_a = ~np.isnan(a_lab)
    mask_b = ~np.isnan(b_lab)

    if mask_a.sum() == 0 and mask_b.sum() == 0:
        raise ValueError("No labeled items found in a_lab or b_lab.")

    # Y_hat_unlab/X_unlab must be DISJOINT from the labeled positions -- a/b
    # are the full per-group arrays (a[mask_a] are the SAME items as
    # a_lab[mask_a]), so exclude the labeled positions here. See
    # evalstats.ppi.correct's docstring for why disjointness matters.
    Y_hat_unlab = np.concatenate([a[~mask_a], b[~mask_b]])
    X_unlab     = np.array([0] * int((~mask_a).sum()) + [1] * int((~mask_b).sum()), dtype=int)
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
        power_tune=power_tune,
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


def _midrank_theta(x: np.ndarray, y: np.ndarray) -> float:
    """``_p_x_gt_y_midrank(x, y) - 0.5``, defined as 0.0 (rather than
    raising) when one side is empty -- a bootstrap resample of a small
    per-stratum labeled slice can legitimately draw zero items from one
    group, and 0.0 ("no evidence either way") is the natural identity value
    for this shifted, additive estimand."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    return _p_x_gt_y_midrank(x, y) - 0.5


# PPI's estimand for the Wilcoxon signed-rank family in _ppi_paired_arrays
# is evalstats.ppi.paired_walsh_midrank_theta (imported at module level
# above), not np.median or a plain per-item sign proportion -- median is
# the wrong estimand under heavy ties (locked at exactly 0 even under a
# large real effect); see that function's docstring in evalstats/ppi.py
# for the full rationale and its known small-n_lab extreme-tie residual.


# ─────────────────────────────────────────────────────────────────────────────
# Non-PPI paired/binary p-value helpers
#
# These back evalstats.core.paired's binary and permutation pairwise-comparison
# methods (newcombe, tango, bayes_binary, sign_test,
# permutation), none of which have a PPI-corrected estimand designed yet.
# They live here — rather than being reimplemented in evalstats.core.paired —
# so every scipy-backed p-value in evalstats has exactly one implementation.
# ─────────────────────────────────────────────────────────────────────────────

def _mcnemar_p(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Exact two-sided McNemar p-value for paired binary data.

    Under H0 (no difference), n10 ~ Binomial(m, 0.5) where m = n10 + n01 is
    the number of discordant pairs.  The two-sided p-value is
    ``2 * P(X ≤ min(n10, n01))`` clamped to [0, 1].

    Returns 1.0 when m == 0 (perfect agreement, no discordant pairs).
    """
    from scipy.stats import binom

    a_bin = (values_a >= 0.5).astype(int)
    b_bin = (values_b >= 0.5).astype(int)
    n10 = int(np.sum((a_bin == 1) & (b_bin == 0)))
    n01 = int(np.sum((a_bin == 0) & (b_bin == 1)))
    m = n10 + n01
    if m == 0:
        return 1.0
    k = min(n10, n01)
    p = float(2.0 * binom.cdf(k, m, 0.5))
    return min(p, 1.0)


def _paired_sign_test_p(diffs: np.ndarray) -> float:
    """Exact two-sided paired sign-test p-value on non-zero differences.

    Drops ties (zero differences), then tests whether positive/negative signs
    are equally likely under H0 via Binomial(n_nonzero, 0.5).

    Returns 1.0 when all paired differences are zero.
    """
    from scipy.stats import binomtest

    nonzero = np.asarray(diffs)[np.asarray(diffs) != 0]
    n_nonzero = int(len(nonzero))
    if n_nonzero == 0:
        return 1.0

    n_positive = int(np.sum(nonzero > 0))
    p = float(binomtest(n_positive, n_nonzero, p=0.5, alternative="two-sided").pvalue)
    if not np.isfinite(p):
        return 1.0
    return min(max(p, 0.0), 1.0)


def _paired_signflip_pvalue(
    diffs: np.ndarray,
    *,
    statistic: str,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided paired randomization p-value via sign-flipping.

    The null hypothesis is that paired differences are symmetric around zero,
    so each per-input difference can be multiplied by +1 or -1 with equal
    probability. This is the standard paired permutation/randomization test.
    """
    observed = abs(float(np.median(diffs)) if statistic == "median" else float(np.mean(diffs)))
    m = len(diffs)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_samples, m), replace=True)
    signed = signs * diffs[np.newaxis, :]
    if statistic == "median":
        null_stats = np.median(signed, axis=1)
    else:
        null_stats = np.mean(signed, axis=1)
    extreme_count = int(np.sum(np.abs(null_stats) >= observed))
    return float((extreme_count + 1) / (n_samples + 1))


def _kw_pairwise_thetas(
    group_arrays: list[np.ndarray],
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """θ_ab = P_mid(group_a > group_b) for every (a, b) in *pairs*.

    Kruskal-Wallis's classical statistic is built from GLOBAL pooled ranks,
    which (unlike Friedman's within-subject ranks) do NOT survive subsetting
    to just the labeled items — a single item's pooled rank depends on every
    other item in the dataset. Pairwise dominance probabilities avoid this:
    θ_ab is a U-statistic (an average over pairwise comparisons), exactly
    the same quantity already used by ``mannwhitney`` for two groups, so it
    is well-defined identically on the full data or any subsample.
    """
    return np.array([
        _p_x_gt_y_midrank(group_arrays[a], group_arrays[b]) for a, b in pairs
    ])


def _kw_mean_sq_deviation_from_labeled(Y: np.ndarray, X: np.ndarray, k: int) -> float:
    """4·mean((θ_ab − 0.5)²) over all pairs a<b — bounded in [0, 1], 0 under
    H₀ (every pair stochastically equal), larger as groups separate.

    This is the scalar omnibus estimand fed to ``correct()`` for the
    point estimate/CI (mirrors ``_anova_between_variance_from_labeled``).
    The actual hypothesis test uses the full pairwise vector and its joint
    bootstrap covariance instead (see ``_ppi_kruskal_wallis_pairwise``) —
    this scalar is for the headline ``corrected_estimate``/effect size only.
    """
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    group_arrays = [Y[X == j] for j in range(k)]
    thetas = _kw_pairwise_thetas(group_arrays, pairs)
    return float(4.0 * np.mean((thetas - 0.5) ** 2))


def _ppi_kruskal_wallis(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    alpha: float,
    n_boot: int,
    rng,
):
    """PPI correction for the Kruskal-Wallis mean-squared pairwise-dominance
    estimand, feeding ``kruskalwallis()``'s ``corrected_estimate``/
    ``corrected_ci``. Structurally the independent-groups ANOVA
    construction with a dominance estimator swapped in.
    """
    from evalstats.ppi import correct

    k = len(groups)
    masks = [~np.isnan(lab_arr) for lab_arr in groups_lab]

    # Y_hat_unlab/X_unlab must be DISJOINT from the labeled positions -- see
    # _ppi_two_sample and evalstats.ppi.correct's docstring.
    Y_hat_unlab = np.concatenate([g[~mask] for g, mask in zip(groups, masks)])
    X_unlab = np.concatenate([
        np.full(int((~mask).sum()), gid, dtype=int) for gid, mask in enumerate(masks)
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
        return _kw_mean_sq_deviation_from_labeled(Y, X, k)

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
        # Deliberately hardcoded, NOT inherited from correct()'s own
        # default. KEEP IT FALSE -- but NOT for the reason this comment
        # used to give.
        #
        # The original rationale was that kruskalwallis()'s
        # corrected_p_value came from _ppi_kruskal_wallis_pairwise's
        # joint bootstrap, which was "not power-tuned", and said to
        # "revisit once _ppi_kruskal_wallis_pairwise also gets
        # power-tuning". It got it (power_tune defaults True there), and
        # this pin was never revisited -- so the stated justification has
        # been false since, and kruskalwallis() does ship a power-tuned
        # p-value beside a lambda=1 estimate/CI.
        #
        # That desync turns out to be the GOOD configuration, so the flag
        # stays put and only the reasoning changes. Flipping this to True
        # collapses the corrected CI's null coverage, measured over 400
        # replicates at k=3, n=300/group, n_lab=80, true H0 (all group
        # means equal):
        #
        #     judge bias [2,0,0]   lambda=1: 0.938 covered, width 0.200
        #                          tuned:    0.460 covered, width 0.049
        #     judge bias [0,0,0]   lambda=1: 0.993 covered, width 0.043
        #                          tuned:    0.950 covered, width 0.037
        #
        # i.e. it fails precisely under the differential judge bias PPI
        # exists to correct. The point estimate is equally biased either
        # way (+0.0064 vs +0.0062 against a true 0); power-tuning simply
        # shrinks the interval ~4x around that upward bias. The root cause
        # is this path's naive percentile-bootstrap CI, which is
        # anti-conservative for a variance-like estimand bounded below by
        # zero -- exactly what _noncentral_f_ci_lambda was introduced to
        # fix for the ANOVA/Friedman family, and which this estimand never
        # migrated to. lambda=1's excess width was masking that, so this
        # pin is load-bearing by accident.
        #
        # Revisit only after giving this estimand a boundary-aware CI (the
        # noncentral-F test-inversion treatment, or a bias-corrected
        # bootstrap); re-run the coverage table above before flipping.
        power_tune=False,
    )


def _ppi_kruskal_wallis_pairwise(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    alpha: float,
    n_boot: int,
    rng,
    power_tune: bool = True,
) -> dict:
    """Joint PPI bootstrap of every pairwise dominance effect θ_ab, used for
    the omnibus Wald test (H₀: every θ_ab = 0.5) and for per-pair reporting.

    Why a dedicated bootstrap instead of calling ``correct()`` once per pair:
    the C(k,2) pairwise estimates are correlated (θ_12 and θ_13 both depend
    on group 1's data), and that covariance has to come from resampling all
    pairs from the SAME draw — independent per-pair bootstraps would treat
    them as uncorrelated and give the wrong covariance for the Wald test.
    Each draw resamples every group's unlabeled pool and labeled subset
    independently (groups are independent samples) but computes the full
    pairwise vector from that one draw, so within-draw correlations between
    pairs sharing a group come through correctly.

    The Wald test deliberately uses this *unrestricted* (signal-inclusive)
    bootstrap covariance rather than a null-restricted one — by construction,
    there's no "should I subtract the LLM-noise term from a null baseline"
    step the way the ANOVA/Friedman closed-form p-values needed, which is
    exactly the kind of derivation that produced a Type-I bug for Friedman.
    A Moore-Penrose pseudo-inverse handles the covariance's potential rank
    deficiency; its degrees of freedom come from the pseudo-inverse's own
    rank (at the same rcond), not a hardcoded k−1. The C(k,2) pairwise
    contrasts have only k−1 independent degrees of freedom for pairwise
    MEAN differences (linear combinations of k group means), but that does
    not hold in general for pairwise dominance probabilities
    (θ_ab = P_mid(a>b) is not a linear combination of per-group "effects"),
    so the covariance is generically full rank C(k,2), not k−1 -- deriving
    df from the actual rank handles both cases correctly, including
    whatever corner the covariance is genuinely near-singular. The
    reference distribution is an F (Hotelling's T²-style finite-sample
    correction, ν = total labeled observations across groups) rather than a
    plain chi-square, since chi-square is only the large-sample limit and
    is mildly anti-conservative whenever the labeled set is small.

    ``power_tune=True``: EXPERIMENTAL. The fixed-lambda=1 construction
    above, ``theta_unlab + (theta_lab_human - theta_lab_llm)``, has
    theta_unlab (the JUDGE-based, biased term) as the always-full-weight
    anchor and the labeled rectifier as what gets scaled -- backwards from
    the canonical PPI form, and (verified via simulation) only unbiased at
    lambda=1 for the identical reason the ANOVA/Friedman constructions
    were. Fixed by swapping the roles into the canonical form
    ``theta_lab_human + lambda*(theta_unlab - theta_lab_llm)`` (unbiased at
    ANY lambda, since the rectifier now has expectation exactly 0 -- both
    terms share the same judge bias, cancelling regardless of lambda).
    lambda is a single scalar shared across all pairs (one judge), chosen
    to minimize trace(Var[theta_hat(lambda)]) -- the same trace-minimizing
    principle used for the ANOVA/Friedman vector estimands, here applied to
    the bootstrap covariance directly rather than a closed-form matrix
    (this estimand was already bootstrap-based, so no new closed-form
    derivation was needed). Estimated from a FIRST bootstrap draw and held
    fixed for a SECOND, independent draw that builds the actual Wald
    covariance -- the same double-draw discipline
    ``evalstats.ppi.correct``'s bootstrap path uses to avoid double-dipping
    (estimating lambda and its own uncertainty from the same draw
    measurably undercovers). The reported covariance also includes the
    delta-method lambda-uncertainty term (matrix generalization of
    ``evalstats.ppi._lambda_var_inflation``), added directly to the
    bootstrap covariance after the fact -- NOT woven into the bootstrap
    loop itself, learned from a real bug in an earlier fix attempt at the
    structurally similar Romano-Wolf joint bootstrap-t (see
    ``evalstats.api._ppi_bootstrap_t_joint_stats`` and
    simulations/out/results_why_ppi_shrink_1_over_0.md Addendum 20/21):
    using each bootstrap replicate's own resampled rectifier there (instead
    of the fixed observed one) made the injected variance data-dependent
    within the bootstrap itself and caused a real FWER regression.
    """
    rng = np.random.default_rng(rng)
    k = len(groups)
    masks = [~np.isnan(lab_arr) for lab_arr in groups_lab]

    # Groups_unlab must be DISJOINT from the labeled positions -- boot_unlab
    # below is resampled independently of the labeled subset, which is only
    # valid for disjoint samples. See _ppi_two_sample and
    # evalstats.ppi.correct's docstring.
    groups_unlab = [g[~m] for g, m in zip(groups, masks)]
    Y_lab_groups = [lab_arr[m] for lab_arr, m in zip(groups_lab, masks)]
    Yhat_lab_groups = [g[m] for g, m in zip(groups, masks)]

    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    n_pairs = len(pairs)

    theta_unlab = _kw_pairwise_thetas(groups_unlab, pairs)
    theta_lab_human = _kw_pairwise_thetas(Y_lab_groups, pairs)
    theta_lab_llm = _kw_pairwise_thetas(Yhat_lab_groups, pairs)

    n_unlab_per_group = [len(g) for g in groups_unlab]
    n_lab_per_group = [len(y) for y in Y_lab_groups]
    n_lab_total = sum(n_lab_per_group)

    def _draw_components(n_draws):
        """One bootstrap draw's worth of (theta_unlab, theta_lab_human,
        theta_lab_llm) replicates -- factored out so power_tune=True can
        call it twice (lambda estimation, then the Wald covariance) without
        duplicating the resampling logic."""
        b_unlab_arr = np.empty((n_draws, n_pairs))
        b_lab_h_arr = np.empty((n_draws, n_pairs))
        b_lab_l_arr = np.empty((n_draws, n_pairs))
        for bi in range(n_draws):
            boot_unlab = [
                groups_unlab[j][rng.integers(0, n_unlab_per_group[j], n_unlab_per_group[j])]
                for j in range(k)
            ]
            # One shared index per group for the labeled resample: the human
            # and LLM values at a given index are the SAME item's two
            # measurements, so they must move together across bootstrap
            # draws (highly correlated in practice) — drawing independent
            # indices for each would destroy that correlation and grossly
            # inflate the rectifier's bootstrap variance.
            lab_idxs = [
                rng.integers(0, n_lab_per_group[j], n_lab_per_group[j]) if n_lab_per_group[j] > 0
                else np.arange(n_lab_per_group[j])
                for j in range(k)
            ]
            boot_lab_human = [Y_lab_groups[j][lab_idxs[j]] for j in range(k)]
            boot_lab_llm = [Yhat_lab_groups[j][lab_idxs[j]] for j in range(k)]
            b_unlab_arr[bi] = _kw_pairwise_thetas(boot_unlab, pairs)
            b_lab_h_arr[bi] = _kw_pairwise_thetas(boot_lab_human, pairs)
            b_lab_l_arr[bi] = _kw_pairwise_thetas(boot_lab_llm, pairs)
        return b_unlab_arr, b_lab_h_arr, b_lab_l_arr

    if power_tune:
        b1_unlab, b1_lab_h, b1_lab_l = _draw_components(n_boot)
        # lambda* minimizing trace(Var[theta_hat(lambda)]) -- disjointness
        # (theta_unlab independent of the labeled quantities) makes this the
        # same trace(Cov)/trace(Var) ratio used for the ANOVA/Friedman
        # vector estimands, just estimated from bootstrap replicates instead
        # of a closed-form covariance (this estimand has no closed form).
        cov_cross = np.cov(b1_lab_h, b1_lab_l, rowvar=False)[:n_pairs, n_pairs:]
        var_unlab = np.atleast_2d(np.cov(b1_unlab, rowvar=False))
        var_lab_l = np.atleast_2d(np.cov(b1_lab_l, rowvar=False))
        den = np.trace(var_unlab + var_lab_l)
        lam_raw = np.trace(cov_cross) / den if den > 1e-12 else 1.0
        lam_raw = min(max(lam_raw, 0.0), 1.0)

        # Raw-lambda replicates: split the b1 draw into batches and
        # recompute the same ratio within each (same idea as
        # evalstats.ppi._bootstrap_batch_lambda_replicates -- a single
        # bootstrap draw can't yield a ratio on its own, so pool a few per
        # batch).
        n_batches = max(5, min(30, n_boot // 50))
        batch_size = max(1, n_boot // n_batches)
        lam_replicates = np.empty(n_batches)
        for bi in range(n_batches):
            sl = slice(bi * batch_size, (bi + 1) * batch_size)
            cc = np.cov(b1_lab_h[sl], b1_lab_l[sl], rowvar=False)[:n_pairs, n_pairs:]
            vu = np.atleast_2d(np.cov(b1_unlab[sl], rowvar=False))
            vl = np.atleast_2d(np.cov(b1_lab_l[sl], rowvar=False))
            d = np.trace(vu + vl)
            lam_replicates[bi] = min(max(np.trace(cc) / d if d > 1e-12 else 1.0, 0.0), 1.0)

        from evalstats.ppi import _adaptive_shrink_lambda
        lam = _adaptive_shrink_lambda(lam_raw, lam_replicates, n_lab_total)

        # Canonical PPI form: theta_lab_human anchors (always full weight,
        # always unbiased), the judge-based rectifier is lambda-scaled --
        # see this function's power_tune=True docstring for why this ordering
        # (not the fixed-lambda=1 construction's) is unbiased at any lambda.
        theta_hat = theta_lab_human + lam * (theta_unlab - theta_lab_llm)
        r_term = theta_unlab - theta_lab_llm

        b2_unlab, b2_lab_h, b2_lab_l = _draw_components(n_boot)
        boots = b2_lab_h + lam * (b2_unlab - b2_lab_l)
        ci_lo = np.percentile(boots, 100 * alpha / 2, axis=0)
        ci_hi = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)
        cov = np.atleast_2d(np.cov(boots, rowvar=False, ddof=1))
        # Lambda's own estimation uncertainty, added directly to the
        # bootstrap covariance (NOT woven into the bootstrap loop) -- see
        # this function's power_tune=True docstring for why (a real bug in
        # an earlier attempt at the structurally similar Romano-Wolf fix).
        var_lam_hat = float(np.var(lam_replicates, ddof=1))
        cov = cov + np.outer(r_term, r_term) * var_lam_hat
    else:
        theta_hat = theta_unlab + (theta_lab_human - theta_lab_llm)
        b_unlab, b_lab_h, b_lab_l = _draw_components(n_boot)
        boots = b_unlab + (b_lab_h - b_lab_l)
        ci_lo = np.percentile(boots, 100 * alpha / 2, axis=0)
        ci_hi = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)
        cov = np.atleast_2d(np.cov(boots, rowvar=False, ddof=1))

    diff = theta_hat - 0.5
    rcond = 1e-8
    cov_pinv = np.linalg.pinv(cov, rcond=rcond)
    wald_stat = float(diff @ cov_pinv @ diff)
    # df = the pseudo-inverse's own rank (same rcond truncation), not a
    # hardcoded k-1. k-1 is the correct contrast-space dimension for
    # pairwise MEAN differences (k group means have exactly k-1 independent
    # linear contrasts among them), which is where the "rank-deficient, use
    # k-1" convention comes from -- but pairwise DOMINANCE probabilities
    # theta_ab = P_mid(a>b) are not linear combinations of k per-group
    # "effects" the way mean differences are, so there is no general exact
    # linear dependency forcing theta_23 to be determined by
    # theta_12/theta_13: the covariance is generically full rank C(k,2), not
    # k-1. Testing a Wald statistic built from a higher-rank covariance
    # against a chi-square(df=k-1) reference systematically over-rejects.
    # Deriving df from cov's own rank (at the same rcond used for the
    # pseudo-inverse) fixes this while still correctly falling back to a
    # smaller df in whatever corner the covariance genuinely is
    # near-singular, rather than assuming it always is.
    eigvals = np.linalg.eigvalsh(cov)
    max_eigval = float(eigvals.max()) if eigvals.size else 0.0

    if max_eigval <= 1e-12:
        # Every pairwise dominance estimate has (numerically) EXACTLY zero
        # bootstrap variance -- confirmed to happen on real data whenever
        # the labeled subsample preserves a strict, deterministic group
        # ordering (e.g. an exact rank-split positive control: the labeled
        # human-side theta is 1.0/0.0 on every possible resample, since
        # resampling can't undo a strict ordering), combined with a small
        # enough power-tuning lambda that the judge-side variance
        # contribution also rounds to zero. np.linalg.pinv on an all-zero
        # covariance returns an all-zero pseudo-inverse (it can't represent
        # "infinite precision"), which silently collapses wald_stat to 0
        # -- indistinguishable, from the Wald statistic alone, from "no
        # information", even though zero uncertainty around a nonzero
        # effect is the most CONFIDENT result a test can report, not an
        # absent one. df then also collapses to 0, which crashed this
        # function outright (ZeroDivisionError in the nu>df branch below)
        # rather than reporting the correct near-certain rejection --
        # silently swallowed by cases/ppi_real.py's per-rep try/except and
        # counted as "failed to detect", which is what collapsed real-data
        # kruskal power from ~0.83 (uncorrected) to ~0.20 (corrected) on
        # exactly these strong-effect scenarios. Bypasses wald_stat/pinv
        # entirely here and falls back to the same "check the point
        # estimate directly" idiom every other closed-form PPI backend
        # uses for an se<=0 degenerate case (e.g.
        # evalstats.ppi._analytic_walsh_theta_correct).
        wald_stat = 0.0
        df = 0
        wald_p = 0.0 if bool(np.any(np.abs(diff) > 1e-9)) else 1.0
    else:
        cov_pinv = np.linalg.pinv(cov, rcond=rcond)
        wald_stat = float(diff @ cov_pinv @ diff)
        df = int(np.linalg.matrix_rank(cov, tol=rcond * max_eigval))

        # Finite-sample (Hotelling's T²-style) correction: a chi-square
        # reference is only the n→∞ limit of the Wald statistic's
        # distribution, and is known to be mildly anti-conservative (too
        # many rejections) whenever the covariance is estimated from a
        # small effective sample — exactly the regime of a sparse labeled
        # set. ν is the total labeled observations across groups (the
        # classical Hotelling "n" feeding the covariance estimate); as ν →
        # ∞ this F-reference converges back to the chi-square one, so it
        # costs nothing in the well-labeled regime.
        nu = sum(n_lab_per_group)
        if nu > df:
            f_stat = wald_stat * (nu - df + 1) / (nu * df)
            wald_p = float(_scipy_stats.f.sf(f_stat, dfn=df, dfd=nu - df + 1)) if f_stat > 0 else 1.0
        else:
            wald_p = float(_scipy_stats.chi2.sf(wald_stat, df=df)) if wald_stat > 0 else 1.0

    # Per-pair two-sided bootstrap p-values, same convention as
    # TestResult.corrected_p_value's own definition (2*min(P(boot<=0.5),
    # P(boot>=0.5))) -- exposed alongside `boots` itself (additive, not a
    # contract change: existing callers only read the keys below already
    # present) so a caller building its own multi-pair FWER correction
    # (e.g. Holm across a family of pairs) has real per-pair p-values to
    # correct, not just the single omnibus wald_p. See
    # evalstats/core/unpaired.py.
    pair_p = 2.0 * np.minimum((boots <= 0.5).mean(axis=0), (boots >= 0.5).mean(axis=0))
    pair_p = np.minimum(pair_p, 1.0)

    return {
        "pairs": pairs,
        "theta_hat": theta_hat,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wald_stat": wald_stat,
        "wald_p": wald_p,
        "boots": boots,
        "pair_p": pair_p,
    }


def _ppi_kruskal_wallis_pairwise_mnar_experimental(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    alpha: float,
    n_boot: int,
    rng,
    n_strata: int = 2,
    min_lab_per_bin: int = 5,
) -> dict:
    """Corrected counterpart to :func:`_ppi_kruskal_wallis_pairwise`, using a
    per-group local (score-binned) rectifier instead of that function's
    single global one, generalized from 2 groups to k groups / all C(k,2)
    pairs jointly.

    Rationale for the local rectifier: a single global rectifier is exactly
    correct for a mean, but a rank/dominance estimand is not a mean of a
    fixed per-item quantity -- an item's contribution depends on the rest of
    the sample -- so under labeling that is non-uniform with respect to score
    (MNAR, e.g. "double-check the highest-scoring items") a global rectifier
    is miscalibrated. Correcting within score bins restores calibration by
    matching each unlabeled item to labeled items of comparable score.
    Labeled items are binned by their TRUE (human) value rather than by their
    own noisy LLM score: binning by the LLM score conditions on a variable
    that both the truth and the judge error feed into, which is a collider,
    and induces a spurious in-bin correlation that biases the per-bin
    discrepancy.

    WARNING -- the two-group sibling of this rectifier was REMOVED on
    2026-08-21 for being unvalidated and badly broken on binary data, and
    this function is the same construction one level up. Measured on binary
    (0/1) outcomes under plain MCAR at a real effect, the two-group local
    rectifiers returned point estimates 57-76% too large in magnitude
    (coverage 0.00-0.06 against a nominal 0.95). Cause: binning by score is
    degenerate when the judge score takes ~2 distinct values and nearly
    everything ties, so each bin's discrepancy is estimated from a
    near-empty, highly selected set. Continuous and Likert data were clean
    in the same test. This function inherits that failure mode whenever
    ``groups`` is coarse/binary -- ``n_strata`` bins over a 2-valued score
    cannot work. It survived removal only because it was scoped to the
    deliberate MNAR-robustness question; do not use it on binary data.

    For each group separately (not per pair -- a group's correction must be
    the same regardless of which other group it's being compared against,
    so the joint pairwise vector stays internally consistent): bin that
    group's own items (labeled + unlabeled) into ``n_strata`` quantile bins
    of that group's own LLM score, and correct each unlabeled item by the
    local (per-bin) mean human-minus-LLM discrepancy among that group's own
    labeled items in the same bin -- falling back to that group's global
    discrepancy when a bin has fewer than ``min_lab_per_bin`` labeled items.
    Every pairwise theta_ab is computed from these per-group corrected
    (labeled + corrected-unlabeled) arrays, never from a per-pair rectifier,
    so all C(k,2) pairs sharing a group stay consistent with each other. The
    joint bootstrap covariance the Wald test needs comes from resampling
    every group once per draw and recomputing the full pairwise vector,
    exactly as :func:`_ppi_kruskal_wallis_pairwise` already does. Bin edges
    are fixed once per group from that group's full sample and held fixed
    across the bootstrap.

    Not a complete fix -- a residual under MNAR labeling remains, and this
    rectifier costs real calibration under ordinary MCAR labeling relative
    to the plain global rectifier (:func:`_ppi_kruskal_wallis_pairwise`) at
    small labeled-sample counts combined with high judge noise. For that
    reason :func:`evalstats.tests.kruskalwallis` defaults to
    :func:`_ppi_kruskal_wallis_pairwise` (``method="global"``); this
    function remains available via ``method="mnar_experimental"`` for
    anyone deliberately studying the MNAR-robustness question. See
    ``simulations/harness/cases/pvalues.py --mode ppi``'s factorial sweep
    for the calibration studies behind this.
    """
    rng = np.random.default_rng(rng)
    k = len(groups)
    masks = [~np.isnan(lab_arr) for lab_arr in groups_lab]

    groups_unlab = [g[~m] for g, m in zip(groups, masks)]
    Y_lab_groups = [lab_arr[m] for lab_arr, m in zip(groups_lab, masks)]
    Yhat_lab_groups = [g[m] for g, m in zip(groups, masks)]

    def _bin_edges(llm_all: np.ndarray) -> np.ndarray:
        if n_strata <= 1:
            return np.array([])
        qs = np.linspace(0, 100, n_strata + 1)[1:-1]
        return np.percentile(llm_all, qs)

    edges = [_bin_edges(g) for g in groups]
    bin_unlab = [np.searchsorted(edges[j], groups_unlab[j], side="right") for j in range(k)]
    # Labeled items are binned by their TRUE (human) value, not their own
    # noisy LLM score -- see this function's docstring, "Collider fix" below,
    # for why binning them by LLM score instead is itself a source of
    # Type-I inflation under MNAR labeling.
    bin_lab = [np.searchsorted(edges[j], Y_lab_groups[j], side="right") for j in range(k)]

    def _correct_unlab(llm_unlab, bin_u, llm_lab, bin_l, truth_lab) -> np.ndarray:
        global_rect = float(np.mean(truth_lab) - np.mean(llm_lab)) if len(truth_lab) > 0 else 0.0
        rects = np.full(max(n_strata, 1), global_rect)
        for s in range(max(n_strata, 1)):
            m = bin_l == s
            if m.sum() >= min_lab_per_bin:
                rects[s] = float(np.mean(truth_lab[m]) - np.mean(llm_lab[m]))
        return llm_unlab + rects[bin_u]

    def _corrected_groups(gu, bu, gl_llm, bl, gl_human) -> list[np.ndarray]:
        return [
            np.concatenate([gl_human[j], _correct_unlab(gu[j], bu[j], gl_llm[j], bl[j], gl_human[j])])
            for j in range(k)
        ]

    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    n_pairs = len(pairs)

    corrected = _corrected_groups(groups_unlab, bin_unlab, Yhat_lab_groups, bin_lab, Y_lab_groups)
    theta_hat = _kw_pairwise_thetas(corrected, pairs)

    n_unlab_per_group = [len(g) for g in groups_unlab]
    n_lab_per_group = [len(y) for y in Y_lab_groups]

    boots = np.empty((n_boot, n_pairs))
    for bi in range(n_boot):
        idx_u = [
            rng.integers(0, n_unlab_per_group[j], n_unlab_per_group[j]) if n_unlab_per_group[j]
            else np.empty(0, dtype=int)
            for j in range(k)
        ]
        # One shared index per group for the labeled resample -- the human
        # and LLM values at a given index are the same item's two
        # measurements, so they must move together (see
        # _ppi_kruskal_wallis_pairwise's identical choice).
        idx_l = [
            rng.integers(0, n_lab_per_group[j], n_lab_per_group[j]) if n_lab_per_group[j]
            else np.empty(0, dtype=int)
            for j in range(k)
        ]
        b_gu = [groups_unlab[j][idx_u[j]] for j in range(k)]
        b_bu = [bin_unlab[j][idx_u[j]] for j in range(k)]
        b_lab_llm = [Yhat_lab_groups[j][idx_l[j]] for j in range(k)]
        b_bin_lab = [bin_lab[j][idx_l[j]] for j in range(k)]
        b_lab_human = [Y_lab_groups[j][idx_l[j]] for j in range(k)]
        b_corrected = _corrected_groups(b_gu, b_bu, b_lab_llm, b_bin_lab, b_lab_human)
        boots[bi] = _kw_pairwise_thetas(b_corrected, pairs)

    ci_lo = np.percentile(boots, 100 * alpha / 2, axis=0)
    ci_hi = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)

    cov = np.atleast_2d(np.cov(boots, rowvar=False, ddof=1))
    diff = theta_hat - 0.5
    rcond = 1e-8
    # df = the pseudo-inverse's OWN rank, not a hardcoded k-1 -- same fix,
    # same reasoning as _ppi_kruskal_wallis_pairwise's df (see that
    # function's docstring): pairwise DOMINANCE probabilities aren't linear
    # combinations of k group effects the way mean differences are, so
    # their covariance is generically full rank C(k,2), not k-1.
    eigvals = np.linalg.eigvalsh(cov)
    max_eigval = float(eigvals.max()) if eigvals.size else 0.0

    if max_eigval <= 1e-12:
        # See _ppi_kruskal_wallis_pairwise's identical degenerate-cov guard
        # for the full mechanism (a real, exact-ordering positive-control
        # effect can drive bootstrap variance to exactly 0, which pinv
        # treats as "no information" rather than "perfect certainty" --
        # crashing on a division by df=0 instead of reporting a confident
        # rejection).
        wald_stat = 0.0
        df = 0
        wald_p = 0.0 if bool(np.any(np.abs(diff) > 1e-9)) else 1.0
    else:
        cov_pinv = np.linalg.pinv(cov, rcond=rcond)
        wald_stat = float(diff @ cov_pinv @ diff)
        df = int(np.linalg.matrix_rank(cov, tol=rcond * max_eigval))

        nu = sum(n_lab_per_group)
        if nu > df:
            f_stat = wald_stat * (nu - df + 1) / (nu * df)
            wald_p = float(_scipy_stats.f.sf(f_stat, dfn=df, dfd=nu - df + 1)) if f_stat > 0 else 1.0
        else:
            wald_p = float(_scipy_stats.chi2.sf(wald_stat, df=df)) if wald_stat > 0 else 1.0

    return {
        "pairs": pairs,
        "theta_hat": theta_hat,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wald_stat": wald_stat,
        "wald_p": wald_p,
    }


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
    power_tune: bool = True,
):
    """PPI correction for a paired estimand ``statistic(a_i − b_i)``.

    Pairing is by array position — ``a[i]`` and ``b[i]`` are the same subject,
    exactly as in ``scipy.stats.ttest_rel`` and ``scipy.stats.wilcoxon``.
    A position is included in the labeled set only when *both*
    ``a_lab[i]`` and ``b_lab[i]`` are non-NaN.

    ``power_tune`` is forwarded to :func:`evalstats.ppi.correct` as-is (see
    its docstring) -- defaults to True (PPI++ power-tuning); pass False to
    reproduce the original (2023) fixed-λ=1 PPI estimator exactly.
    """
    from evalstats.ppi import correct

    mask = ~np.isnan(a_lab) & ~np.isnan(b_lab)
    if mask.sum() == 0:
        raise ValueError(
            "No positions have human labels for both groups in a_lab and b_lab."
        )

    # Y_hat_unlab must be DISJOINT from the labeled positions -- see
    # _ppi_two_sample and evalstats.ppi.correct's docstring.
    Y_hat_unlab = (a - b)[~mask]
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
        power_tune=power_tune,
    )


def _ppi_paired_bayes_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    a_lab: np.ndarray,
    b_lab: np.ndarray,
    alpha: float,
    n_boot: int,
    rng,
    power_tune: bool = True,
):
    """PPI correction for a paired mean-difference estimand
    ``mean(a_i - b_i)``, using Bayesian bootstrap (Dirichlet-weighted)
    resampling for both the full-sample and rectifier terms, instead of
    :func:`evalstats.ppi.correct`'s classical (multinomial) resampling.

    ``evalstats.core.paired``'s ``'bayes_bootstrap'`` method uses Dirichlet
    weighting (see :func:`evalstats.core.resampling.bayes_bootstrap_means_1d`)
    because it gives smoother, better-calibrated coverage than a classical
    bootstrap or an exact discrete test on sparse, heavily-tied data --
    e.g. paired binary diffs, which only take values in {-1, 0, 1}. Routing
    a PPI correction through ``evalstats.ppi.correct`` instead would silently
    drop that advantage (its bootstrap loop is hardcoded to classical
    resampling), producing a result numerically identical to a plain
    mean-based PPI correction (e.g. a paired t-test) despite the different
    label. This function exists to carry the Dirichlet-weighting property
    through to the corrected estimate too.

    Pairing is by array position, matching :func:`_ppi_paired_arrays`; a
    position is included in the labeled set only when *both* ``a_lab[i]``
    and ``b_lab[i]`` are non-NaN. ``diffs_lab_true``/``diffs_lab_llm`` are
    bootstrapped with the SAME per-replicate Dirichlet weight vector
    (reimplemented locally rather than via bayes_bootstrap_means_1d, which
    only returns one array's replicates and draws its own weights
    internally -- calling it twice would give two INDEPENDENT weight draws,
    destroying the pairing) so a given labeled item's human and LLM values
    move together across replicates, exactly as ``correct()`` does by
    reusing one integer index draw for both ``Y_lab[idx_lab]`` and
    ``Y_hat_lab[idx_lab]``. This is what makes ``power_tune`` possible here
    at all -- lambda's derivation needs Cov(b_lab, b_hat_lab), which
    requires the two separately, not just their pre-combined residual.

    ``power_tune`` mirrors :func:`evalstats.ppi.correct`'s parameter of the
    same name (see its docstring for the full lambda derivation and the
    n_lab-dependent shrinkage) -- reimplemented here rather than delegating
    to ``correct()``, since the whole point of this function is the
    Dirichlet-weighted resampling ``correct()`` doesn't support.
    """
    from evalstats.ppi import (
        PPIResult, _adaptive_shrink_lambda, _analytic_mean_correct,
        _bootstrap_batch_lambda_replicates, _lambda_var_inflation,
        _MIN_LAB_RECOMMENDED,
    )

    rng = np.random.default_rng(rng)

    mask = ~np.isnan(a_lab) & ~np.isnan(b_lab)
    if mask.sum() == 0:
        raise ValueError(
            "No positions have human labels for both groups in a_lab and b_lab."
        )

    # diffs_unlab must be DISJOINT from the labeled positions -- see
    # _ppi_two_sample and evalstats.ppi.correct's docstring.
    all_diffs = a - b
    diffs_unlab = all_diffs[~mask]
    diffs_lab_llm = all_diffs[mask]
    diffs_lab_true = (a_lab - b_lab)[mask]

    # Below _MIN_LAB_RECOMMENDED, Dirichlet-weighted resampling has the same
    # small-n_lab undercoverage as classical/bootstrap-t resampling -- the
    # limitation is "not enough real information in n_lab points", not which
    # resampling scheme reads that information. This is a mean estimand
    # (paired diffs), so correct()'s closed-form analytic path applies
    # directly.
    if len(diffs_lab_true) < _MIN_LAB_RECOMMENDED:
        return _analytic_mean_correct(diffs_lab_true, diffs_lab_llm, diffs_unlab, alpha, power_tune=power_tune)

    f_unlab = float(np.mean(diffs_unlab))
    f_lab = float(np.mean(diffs_lab_true))
    f_hat_lab = float(np.mean(diffs_lab_llm))
    rectifier = f_lab - f_hat_lab
    n_lab = len(diffs_lab_true)
    n_unlab = len(diffs_unlab)

    def _draw(n_draw: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        exp_unlab = rng.exponential(1.0, size=(n_draw, n_unlab))
        w_unlab = exp_unlab / exp_unlab.sum(axis=1, keepdims=True)
        b_unlab = w_unlab @ diffs_unlab
        exp_lab = rng.exponential(1.0, size=(n_draw, n_lab))
        w_lab = exp_lab / exp_lab.sum(axis=1, keepdims=True)
        b_lab = w_lab @ diffs_lab_true
        b_hat_lab = w_lab @ diffs_lab_llm
        return b_unlab, b_lab, b_hat_lab

    lam: Optional[float] = None
    if power_tune:
        b1_unlab, b1_lab, b1_hat_lab = _draw(n_boot)
        denom = float(np.var(b1_unlab - b1_hat_lab, ddof=1))
        if denom > 1e-12:
            lam_raw = float(np.cov(b1_lab, b1_hat_lab, ddof=1)[0, 1] / denom)
            lam_raw = min(max(lam_raw, 0.0), 1.0)
        else:
            lam_raw = 1.0
        # Adaptive shrinkage (see evalstats.ppi._adaptive_shrink_lambda's
        # docstring for the shared rationale) -- was previously fixed
        # toward a target of 1 regardless of what the data supported, like
        # every other power_tune site in this codebase before this fix.
        # Falls back to target=1 when diffs_lab_true is near-degenerate,
        # same guard evalstats.ppi.correct's bootstrap path uses.
        raw_var_lab_true = float(np.var(diffs_lab_true, ddof=1)) if n_lab > 1 else 0.0
        raw_var_lab_llm = float(np.var(diffs_lab_llm, ddof=1)) if n_lab > 1 else 0.0
        if n_lab <= 1 or raw_var_lab_true < raw_var_lab_llm * 1e-6:
            lam_replicates = None
        else:
            lam_replicates = _bootstrap_batch_lambda_replicates(b1_lab, b1_hat_lab, b1_unlab)
        lam = _adaptive_shrink_lambda(lam_raw, lam_replicates, n_lab)
        b2_unlab, b2_lab, b2_hat_lab = _draw(n_boot)
        estimate = f_lab + lam * (f_unlab - f_hat_lab)
        boots = b2_lab + lam * (b2_unlab - b2_hat_lab)
        # See evalstats.ppi._lambda_var_inflation's docstring and
        # evalstats.ppi.correct's bootstrap path (identical fix, mirrored
        # here): `lam` is fixed across every b2 replicate, so `boots`
        # understates variance by missing lambda's own estimation
        # uncertainty. Convolve it back in as independent noise, rather
        # than re-deriving lambda per replicate.
        extra_var = _lambda_var_inflation(f_unlab - f_hat_lab, lam_replicates)
        if extra_var > 0.0:
            boots = boots + rng.normal(0.0, np.sqrt(extra_var), size=boots.shape)
    else:
        b_unlab, b_lab, b_hat_lab = _draw(n_boot)
        estimate = f_unlab + rectifier
        boots = b_unlab + (b_lab - b_hat_lab)

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    p_value = float(2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0)))
    p_value = min(max(p_value, 0.0), 1.0)

    return PPIResult(
        estimate=float(estimate), ci_low=lo, ci_high=hi, alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=float(rectifier),
        p_value=p_value, lam=lam,
    )


def _ppi_require_unlabeled(n_all: int) -> None:
    """Guard against a zero-length disjoint-unlabeled sample.

    The two-term additive variance these ``_ppi_paired_*``/``_ppi_single_*``
    functions use (``Var(unlabeled)/n_all + Var(rectifier)/n_lab``) divides
    by ``n_all`` directly; without this check, an alignment set covering
    every item (n_all=0) raises a bare ``ZeroDivisionError`` instead of an
    actionable message. PPI's benefit comes from a large unlabeled pool plus
    a small labeled subset -- if every item is labeled, there is no
    LLM-only term left to correct and the human labels should be used
    directly instead.
    """
    if n_all == 0:
        raise ValueError(
            "PPI correction requires at least one unlabeled item, got n_all=0 "
            "(every item in this comparison is human-labeled, leaving no "
            "unlabeled residual for the LLM-only term). If every item is "
            "labeled, use the human labels directly instead of PPI correction."
        )


def _ppi_paired_bootstrap_t(
    a: np.ndarray,
    b: np.ndarray,
    a_lab: np.ndarray,
    b_lab: np.ndarray,
    alpha: float,
    n_boot: int,
    rng,
):
    """PPI correction for a paired mean-difference estimand
    ``mean(a_i - b_i)``, using a studentized bootstrap (bootstrap-t) pivot
    instead of :func:`evalstats.ppi.correct`'s plain percentile bootstrap --
    the paired-mean analogue of
    :func:`evalstats.core.resampling.bootstrap_t_ci_1d`, generalized to
    PPI's two-term variance decomposition.

    ``bootstrap_t_ci_1d`` studentizes a single sample using
    ``SE = std(sample, ddof=1) / sqrt(n)`` per replicate. A PPI estimator is
    a SUM of two independent terms -- the full-sample (unlabeled) mean and
    the labeled-subset rectifier -- so its variance is the sum of their two
    variances: ``Var(unlab)/N + Var(rectifier)/n_lab`` (the same
    independent-terms structure the PPI point estimate itself already
    assumes). Each bootstrap replicate resamples both terms (classically,
    matching :func:`evalstats.ppi.correct`'s resampling) and is studentized
    by that same two-term SE, computed from the replicate's own resampled
    arrays -- directly mirroring ``bootstrap_t_ci_1d``'s per-replicate SE,
    just with two variance components instead of one.

    Falls back to a plain percentile-bootstrap CI/p-value (mirroring
    ``bootstrap_t_ci_1d``'s own fallback) when the observed SE is zero or
    too many bootstrap replicates have a degenerate (near-zero) SE.

    Pairing is by array position, matching :func:`_ppi_paired_arrays`; a
    position is included in the labeled set only when *both* ``a_lab[i]``
    and ``b_lab[i]`` are non-NaN.
    """
    from evalstats.ppi import PPIResult, _analytic_mean_correct, _MIN_LAB_RECOMMENDED

    rng = np.random.default_rng(rng)

    mask = ~np.isnan(a_lab) & ~np.isnan(b_lab)
    if mask.sum() == 0:
        raise ValueError(
            "No positions have human labels for both groups in a_lab and b_lab."
        )

    # diffs_unlab must be DISJOINT from the labeled positions -- the
    # two-term variance decomposition below assumes independence, which
    # only holds for disjoint samples. See _ppi_two_sample and
    # evalstats.ppi.correct's docstring.
    all_diffs = a - b
    diffs_unlab = all_diffs[~mask]
    diffs_lab_llm = all_diffs[mask]
    diffs_lab_true = (a_lab - b_lab)[mask]
    rect_items = diffs_lab_true - diffs_lab_llm

    n_all = len(diffs_unlab)
    n_lab = len(rect_items)
    _ppi_require_unlabeled(n_all)

    # Below _MIN_LAB_RECOMMENDED, this function's own bootstrap-t is subject
    # to the same small-n_lab undercoverage as evalstats.ppi.correct's
    # percentile bootstrap -- the limitation is "not enough real information
    # in n_lab points", not which resampling scheme reads that information.
    # This function is a mean estimand (paired diffs), so the same
    # closed-form analytic path correct() uses applies directly -- delegate
    # instead of duplicating it.
    if n_lab < _MIN_LAB_RECOMMENDED:
        return _analytic_mean_correct(diffs_lab_true, diffs_lab_llm, diffs_unlab, alpha, power_tune=False)

    f_unlab = float(np.mean(diffs_unlab))
    f_lab = float(np.mean(diffs_lab_true))
    f_hat_lab = float(np.mean(diffs_lab_llm))
    rectifier = f_lab - f_hat_lab
    estimate = float(f_unlab + rectifier)

    def _var(x: np.ndarray) -> float:
        return float(np.var(x, ddof=1)) if len(x) > 1 else 0.0

    se_obs = float(np.sqrt(_var(diffs_unlab) / n_all + _var(rect_items) / n_lab))

    # ── Bootstrap: chunked, vectorized, mirroring bootstrap_t_ci_1d ────────
    chunk_size = max(1, min(n_boot, 4096, max(1, int(1_000_000 // max(max(n_all, n_lab), 1)))))
    boot_theta = np.empty(n_boot, dtype=float)
    boot_se = np.empty(n_boot, dtype=float)

    start = 0
    while start < n_boot:
        stop = min(start + chunk_size, n_boot)
        m = stop - start
        idx_all = rng.integers(0, n_all, size=(m, n_all))
        idx_lab = rng.integers(0, n_lab, size=(m, n_lab))
        unlab_samples = diffs_unlab[idx_all]  # (m, n_all)
        rect_samples = rect_items[idx_lab]    # (m, n_lab)
        boot_theta[start:stop] = unlab_samples.mean(axis=1) + rect_samples.mean(axis=1)
        boot_se[start:stop] = np.sqrt(
            np.var(unlab_samples, ddof=1, axis=1) / n_all
            + np.var(rect_samples, ddof=1, axis=1) / n_lab
        )
        start = stop

    def _percentile_result() -> "PPIResult":
        lo = float(np.percentile(boot_theta, 100 * alpha / 2))
        hi = float(np.percentile(boot_theta, 100 * (1 - alpha / 2)))
        p = float(2.0 * min(np.mean(boot_theta <= 0.0), np.mean(boot_theta >= 0.0)))
        p = min(max(p, 0.0), 1.0)
        return PPIResult(
            estimate=estimate, ci_low=lo, ci_high=hi, alpha=alpha,
            llm_estimate=f_unlab, human_estimate=f_lab, rectifier=float(rectifier),
            p_value=p,
        )

    if se_obs <= 0.0 or not np.isfinite(se_obs):
        return _percentile_result()

    valid = np.isfinite(boot_se) & (boot_se > 0.0)
    if not np.any(valid):
        return _percentile_result()
    se_floor = max(np.finfo(float).eps, 1e-8 * se_obs)
    tiny_frac = float(np.mean(valid & (boot_se < se_floor)))
    if tiny_frac > 0.05:
        return _percentile_result()
    valid = valid & (boot_se >= se_floor)
    if not np.any(valid):
        return _percentile_result()

    t_stats = (boot_theta[valid] - estimate) / boot_se[valid]
    t_lo = float(np.percentile(t_stats, 100.0 * alpha / 2))
    t_hi = float(np.percentile(t_stats, 100.0 * (1.0 - alpha / 2)))
    ci_low = float(estimate - t_hi * se_obs)
    ci_high = float(estimate - t_lo * se_obs)

    t_obs = estimate / se_obs
    p_value = float(2.0 * min(np.mean(t_stats <= t_obs), np.mean(t_stats >= t_obs)))
    p_value = min(max(p_value, 0.0), 1.0)

    return PPIResult(
        estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=float(rectifier),
        p_value=p_value,
    )


def _ppi_paired_tango(
    a: np.ndarray,
    b: np.ndarray,
    a_lab: np.ndarray,
    b_lab: np.ndarray,
    alpha: float,
    *,
    power_tune: bool = True,
):
    """PPI correction for the paired binary difference estimand
    ``evalstats.core.resampling.tango_paired_ci`` targets: ``mean(a_i - b_i)``,
    equivalently ``(n10 - n01) / n`` (the discordant-pair-rate difference).

    Tango's score interval has the Wilson-style form::

        center = d_hat / (1 + z^2/n)
        radius = z / (1 + z^2/n) * sqrt(V_hat + z^2/(4*n^2))

    where ``V_hat = (n10+n01)/n^2 - (n10-n01)^2/n^3``. That's exactly
    ``Var(a_i - b_i, ddof=0) / n`` -- the same naive per-item variance a
    Wald interval would use, divided by the sample size, just wrapped in a
    Wilson-style shrinkage/continuity correction for better small-sample
    coverage. Because it's already expressed as "some reference variance,
    divided by n," it generalizes to any PPI point-estimate/variance pair by
    substituting an effective n that reproduces the same "variance =
    (reference variance) / n" relationship the original formula assumes:
    ``n_eff = Var(unlabeled diffs) / V_hat_PPI``, using the disjoint
    unlabeled sample's own per-item variance as the reference throughout --
    the most stable per-item variance estimate available (largest sample),
    regardless of ``power_tune``. This reduces exactly to the original
    (uncorrected) Tango formula in the degenerate case where the "labeled"
    subset is the full sample with no judge error (rectifier -> 0,
    n_eff -> n).

    ``power_tune`` : when *True* (the default), the point estimate and
    ``V_hat_PPI`` come from :func:`evalstats.ppi._analytic_mean_point_se`'s
    closed-form variance-minimizing lambda* -- the same derivation
    :func:`evalstats.ppi._analytic_mean_correct`/``_analytic_logit_t_correct``
    use for the unbounded/bounded numeric mean estimand (``ppi_t_interval``/
    ``ppi_logit_t``) -- rather than the fixed lambda=1 rectifier. This
    estimand is exactly ``mean(a_i - b_i)``, identical to those two, so the
    same closed-form point-estimate/variance derivation applies unchanged;
    only the CI's shape differs (Wilson-style effective-n shrinkage here,
    vs. a plain or logit-transformed t-interval there). Validated against
    the full binary judge-bias scenario catalog (harness ``--mode ppi
    --eval-types binary``, 65 scenarios): zero Holm-confirmed miscalibrated
    cells at either setting, mean CI width ~13% narrower at *True* with no
    coverage cost. Pass *False* to reproduce the original fixed-lambda=1
    construction exactly (bit-for-bit -- the lambda=1 case of the general
    variance formula collapses back to ``Var(unlabeled diffs)/N +
    Var(rectifier residuals)/n_lab`` precisely); this is what
    ``simulations.harness.methods.TANGO_FIXED_LAMBDA`` runs.

    Uses sample variance (ddof=1) and a t(df=n_lab-1) critical value --
    not the classical Tango formula's own population variance (ddof=0) and
    normal-z critical value -- matching ``_analytic_mean_point_se``'s
    convention, for the same reason a t-interval uses n-1 degrees of
    freedom instead of a normal quantile when its own variance is estimated
    from finite data. ``n_lab`` (not ``n_all``) sets df since the labeled
    subset -- driving the rectifier term -- is the smaller, more uncertain
    sample.

    Unlike the other ``_ppi_paired_*`` functions here, this is fully
    closed-form -- no bootstrap resampling is used (or would be
    meaningful to add): Tango's coverage guarantee comes from the
    score-interval derivation itself, not from resampling a plug-in
    statistic, so there is no ``n_boot``/``rng`` parameter.

    Pairing is by array position, matching :func:`_ppi_paired_arrays`; a
    position is included in the labeled set only when *both* ``a_lab[i]``
    and ``b_lab[i]`` are non-NaN.
    """
    from evalstats.ppi import PPIResult, _analytic_mean_point_se
    from scipy.stats import t as _t_dist

    mask = ~np.isnan(a_lab) & ~np.isnan(b_lab)
    if mask.sum() == 0:
        raise ValueError(
            "No positions have human labels for both groups in a_lab and b_lab."
        )

    # diffs_unlab must be DISJOINT from the labeled positions -- the
    # additive two-term variance formula below assumes independence, which
    # only holds for disjoint samples. See _ppi_two_sample and
    # evalstats.ppi.correct's docstring.
    all_diffs = a - b
    diffs_unlab = all_diffs[~mask]
    diffs_lab_llm = all_diffs[mask]
    diffs_lab_true = (a_lab - b_lab)[mask]

    n_all = len(diffs_unlab)
    n_lab = len(diffs_lab_true)
    _ppi_require_unlabeled(n_all)

    estimate, se, f_unlab, f_lab, rectifier, lam, df = _analytic_mean_point_se(
        diffs_lab_true, diffs_lab_llm, diffs_unlab, power_tune,
    )
    v_hat = se * se

    # Reference per-item variance for the effective-n substitution -- the
    # disjoint unlabeled sample's own variance (ddof=1), same choice
    # regardless of power_tune (see docstring).
    sigma2_f = float(np.var(diffs_unlab, ddof=1)) if n_all > 1 else 0.0

    df = max(df, 1)
    t_crit = float(_t_dist.ppf(1.0 - alpha / 2.0, df))

    if v_hat <= 0.0 or not np.isfinite(v_hat) or sigma2_f <= 0.0:
        # Degenerate (e.g. every diff identical): no meaningful interval to derive.
        return PPIResult(
            estimate=estimate, ci_low=estimate, ci_high=estimate, alpha=alpha,
            llm_estimate=f_unlab, human_estimate=f_lab, rectifier=float(rectifier),
            p_value=1.0, lam=lam,
        )

    n_eff = sigma2_f / v_hat
    shrink = 1.0 / (1.0 + t_crit ** 2 / n_eff)
    center = estimate * shrink
    radius = float(t_crit * shrink * np.sqrt(v_hat + t_crit ** 2 / (4.0 * n_eff ** 2)))
    ci_low = float(np.clip(center - radius, -1.0, 1.0))
    ci_high = float(np.clip(center + radius, -1.0, 1.0))

    t_obs = estimate / np.sqrt(v_hat)
    p_value = float(2.0 * (1.0 - _t_dist.cdf(abs(t_obs), df)))
    p_value = min(max(p_value, 0.0), 1.0)

    return PPIResult(
        estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=float(rectifier),
        p_value=p_value, lam=lam,
    )


def _ppi_single_bootstrap_t(a: np.ndarray, a_lab: np.ndarray, alpha: float, n_boot: int, rng):
    """PPI correction for a single-sample mean estimand ``mean(a)``, using a
    studentized bootstrap (bootstrap-t) pivot -- the single-sample sibling
    of :func:`_ppi_paired_bootstrap_t` (see its docstring for the underlying
    two-term-variance derivation), with no pairing step: just one array and
    its own sparse human labels, instead of a paired difference.

    A position is included in the labeled set only when ``a_lab[i]`` is
    non-NaN.
    """
    from evalstats.ppi import PPIResult

    rng = np.random.default_rng(rng)

    mask = ~np.isnan(a_lab)
    if mask.sum() == 0:
        raise ValueError("No positions have human labels in a_lab.")

    # values_unlab must be DISJOINT from the labeled positions -- the
    # variance formula below assumes the full-sample and rectifier terms
    # are independent, which is only true for disjoint samples. See
    # _ppi_two_sample and evalstats.ppi.correct's docstring.
    all_values = np.asarray(a, dtype=float)
    values_unlab = all_values[~mask]
    values_lab_llm = all_values[mask]
    values_lab_true = np.asarray(a_lab, dtype=float)[mask]
    rect_items = values_lab_true - values_lab_llm

    n_all = len(values_unlab)
    n_lab = len(rect_items)
    _ppi_require_unlabeled(n_all)

    f_unlab = float(np.mean(values_unlab))
    f_lab = float(np.mean(values_lab_true))
    f_hat_lab = float(np.mean(values_lab_llm))
    rectifier = f_lab - f_hat_lab
    estimate = float(f_unlab + rectifier)

    def _var(x: np.ndarray) -> float:
        return float(np.var(x, ddof=1)) if len(x) > 1 else 0.0

    se_obs = float(np.sqrt(_var(values_unlab) / n_all + _var(rect_items) / n_lab))

    chunk_size = max(1, min(n_boot, 4096, max(1, int(1_000_000 // max(max(n_all, n_lab), 1)))))
    boot_theta = np.empty(n_boot, dtype=float)
    boot_se = np.empty(n_boot, dtype=float)

    start = 0
    while start < n_boot:
        stop = min(start + chunk_size, n_boot)
        m = stop - start
        idx_all = rng.integers(0, n_all, size=(m, n_all))
        idx_lab = rng.integers(0, n_lab, size=(m, n_lab))
        unlab_samples = values_unlab[idx_all]
        rect_samples = rect_items[idx_lab]
        boot_theta[start:stop] = unlab_samples.mean(axis=1) + rect_samples.mean(axis=1)
        boot_se[start:stop] = np.sqrt(
            np.var(unlab_samples, ddof=1, axis=1) / n_all
            + np.var(rect_samples, ddof=1, axis=1) / n_lab
        )
        start = stop

    def _percentile_result() -> "PPIResult":
        lo = float(np.percentile(boot_theta, 100 * alpha / 2))
        hi = float(np.percentile(boot_theta, 100 * (1 - alpha / 2)))
        return PPIResult(
            estimate=estimate, ci_low=lo, ci_high=hi, alpha=alpha,
            llm_estimate=f_unlab, human_estimate=f_lab, rectifier=float(rectifier),
            p_value=None,
        )

    if se_obs <= 0.0 or not np.isfinite(se_obs):
        return _percentile_result()

    valid = np.isfinite(boot_se) & (boot_se > 0.0)
    if not np.any(valid):
        return _percentile_result()
    se_floor = max(np.finfo(float).eps, 1e-8 * se_obs)
    tiny_frac = float(np.mean(valid & (boot_se < se_floor)))
    if tiny_frac > 0.05:
        return _percentile_result()
    valid = valid & (boot_se >= se_floor)
    if not np.any(valid):
        return _percentile_result()

    t_stats = (boot_theta[valid] - estimate) / boot_se[valid]
    t_lo = float(np.percentile(t_stats, 100.0 * alpha / 2))
    t_hi = float(np.percentile(t_stats, 100.0 * (1.0 - alpha / 2)))
    ci_low = float(estimate - t_hi * se_obs)
    ci_high = float(estimate - t_lo * se_obs)

    return PPIResult(
        estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=float(rectifier),
        p_value=None,
    )


def _ppi_single_wilson(a: np.ndarray, a_lab: np.ndarray, alpha: float):
    """PPI correction for a single-sample binary proportion ``mean(a)``:
    sample variance (ddof=1) + a t(df=n_lab-1) critical value (matching
    ``_analytic_mean_point_se``, the t-interval/logit-t backend), built as
    a plain symmetric interval around the corrected estimate and clamped
    to ``[0, 1]``. Fully closed-form -- no bootstrap resampling is used.

    Deliberately not a classical Wilson score interval, despite operating
    on a proportion. Wilson's shrinkage-toward-0.5 term --
    ``center = (p_hat + t^2/(2n_eff)) / (1 + t^2/n_eff)`` -- is derived by
    solving a score-test equation that evaluates a binomial's variance
    p(1-p)/n at each hypothesized true value, which is what makes Wilson
    well-calibrated for a genuine binomial proportion (variance genuinely
    grows toward p=0.5 and shrinks toward the boundaries). The
    PPI-corrected estimator's ``v_hat`` is a plug-in variance that does not
    vary with the hypothesized value that way, so borrowing Wilson's
    shrinkage term would introduce a spurious bias toward 0.5 with no
    compensating change in width. A plain symmetric t-interval on
    (estimate, v_hat, df) avoids that; clamping to [0, 1] (needed since a
    raw Wald-type interval can extend outside a proportion's valid range,
    especially near a boundary) can only improve coverage of a true value
    already known to lie in [0, 1] -- it never removes truth-containing
    mass.

    A position is included in the labeled set only when ``a_lab[i]`` is
    non-NaN.
    """
    from evalstats.ppi import PPIResult
    from scipy.stats import t as _t_dist

    mask = ~np.isnan(a_lab)
    if mask.sum() == 0:
        raise ValueError("No positions have human labels in a_lab.")

    # values_unlab must be DISJOINT from the labeled positions -- the
    # variance formula below assumes the full-sample and rectifier terms
    # are independent, which is only true for disjoint samples. See
    # _ppi_two_sample and evalstats.ppi.correct's docstring.
    all_values = np.asarray(a, dtype=float)
    values_unlab = all_values[~mask]
    values_lab_llm = all_values[mask]
    values_lab_true = np.asarray(a_lab, dtype=float)[mask]
    rect_items = values_lab_true - values_lab_llm

    n_all = len(values_unlab)
    n_lab = len(rect_items)
    _ppi_require_unlabeled(n_all)

    f_unlab = float(np.mean(values_unlab))
    f_lab = float(np.mean(values_lab_true))
    f_hat_lab = float(np.mean(values_lab_llm))
    rectifier = f_lab - f_hat_lab
    estimate = float(f_unlab + rectifier)

    def _svar(x: np.ndarray) -> float:
        # Sample variance (ddof=1) -- see the docstring note above.
        return float(np.var(x, ddof=1)) if len(x) > 1 else 0.0

    sigma2_f = _svar(values_unlab)      # raw (uncorrected) judge proportion's variance
    sigma2_rect = _svar(rect_items)
    v_hat = sigma2_f / n_all + sigma2_rect / n_lab

    df = max(n_lab - 1, 1)
    t_crit = float(_t_dist.ppf(1.0 - alpha / 2.0, df))

    if v_hat <= 0.0 or not np.isfinite(v_hat) or sigma2_f <= 0.0:
        return PPIResult(
            estimate=estimate, ci_low=estimate, ci_high=estimate, alpha=alpha,
            llm_estimate=f_unlab, human_estimate=f_lab, rectifier=float(rectifier),
            p_value=1.0,
        )

    se = float(np.sqrt(v_hat))
    ci_low = max(0.0, estimate - t_crit * se)
    ci_high = min(1.0, estimate + t_crit * se)

    t_obs = estimate / se
    p_value = float(2.0 * (1.0 - _t_dist.cdf(abs(t_obs), df)))
    p_value = min(max(p_value, 0.0), 1.0)

    return PPIResult(
        estimate=estimate, ci_low=float(ci_low), ci_high=float(ci_high), alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=float(rectifier),
        p_value=p_value,
    )


def _ppi_single_t_interval(a: np.ndarray, a_lab: np.ndarray, alpha: float, power_tune: bool = True):
    """PPI correction for a single-sample mean estimand ``mean(a)`` on an
    unbounded numeric scale, via the closed-form (no-bootstrap) analytic
    construction -- evalstats.ppi._analytic_mean_correct -- applied at
    EVERY n_lab (not just below _MIN_LAB_RECOMMENDED, unlike
    _ppi_paired_bootstrap_t's own small-n_lab-only use of the same
    function; see evalstats.ppi._ANALYTIC_ALWAYS_PREFERRED's precedent
    for always preferring the closed form over a bootstrap for a mean-type
    estimand). The single-sample sibling of _ppi_paired_t_interval and the
    closed-form analogue of _ppi_single_bootstrap_t (identical estimand,
    no bootstrap resampling at all).

    A position is included in the labeled set only when ``a_lab[i]`` is
    non-NaN.

    ``power_tune`` mirrors :func:`evalstats.ppi.correct`'s parameter of the
    same name (see its docstring's "PPI++ power-tuning" section) -- default
    True, matching every other paired/single PPI wrapper in this module.

    Passes ``label_shift_robust=True`` to :func:`evalstats.ppi.
    _analytic_mean_correct` -- unlike every paired/two-group PPI wrapper in
    this module, a single-arm mean estimand has no second group for a
    label-selection MNAR mechanism's point-estimate bias to cancel against
    (see :func:`evalstats.ppi._analytic_mean_point_se`'s
    ``label_shift_robust`` docstring for the full mechanism and
    simulations/out/results_why_ppi_shrink_1_over_0.md's Addendum 34).
    """
    from evalstats.ppi import _analytic_mean_correct

    mask = ~np.isnan(a_lab)
    if mask.sum() == 0:
        raise ValueError("No positions have human labels in a_lab.")

    all_values = np.asarray(a, dtype=float)
    values_unlab = all_values[~mask]
    values_lab_llm = all_values[mask]
    values_lab_true = np.asarray(a_lab, dtype=float)[mask]

    return _analytic_mean_correct(
        values_lab_true, values_lab_llm, values_unlab, alpha, power_tune=power_tune,
        label_shift_robust=True,
    )


def _ppi_paired_t_interval(
    a: np.ndarray, b: np.ndarray, a_lab: np.ndarray, b_lab: np.ndarray, alpha: float, power_tune: bool = True,
):
    """PPI correction for a paired mean-difference estimand ``mean(a_i -
    b_i)`` on an unbounded numeric scale, via the closed-form (no-
    bootstrap) analytic construction -- the closed-form analogue of
    _ppi_paired_bootstrap_t (identical estimand; that function already
    delegates to this same evalstats.ppi._analytic_mean_correct below
    _MIN_LAB_RECOMMENDED -- this function uses it unconditionally, at
    every n_lab).

    Pairing is by array position, matching _ppi_paired_bootstrap_t; a
    position is included in the labeled set only when *both* ``a_lab[i]``
    and ``b_lab[i]`` are non-NaN.

    ``power_tune`` mirrors :func:`evalstats.ppi.correct`'s parameter of the
    same name -- default True, matching every other paired/single PPI
    wrapper in this module.
    """
    from evalstats.ppi import _analytic_mean_correct

    mask = ~np.isnan(a_lab) & ~np.isnan(b_lab)
    if mask.sum() == 0:
        raise ValueError(
            "No positions have human labels for both groups in a_lab and b_lab."
        )

    all_diffs = a - b
    diffs_unlab = all_diffs[~mask]
    diffs_lab_llm = all_diffs[mask]
    diffs_lab_true = (a_lab - b_lab)[mask]

    return _analytic_mean_correct(diffs_lab_true, diffs_lab_llm, diffs_unlab, alpha, power_tune=power_tune)


def _ppi_two_sample_t_interval(
    a: np.ndarray, b: np.ndarray, a_lab: np.ndarray, b_lab: np.ndarray, alpha: float, power_tune: bool = True,
):
    """PPI correction for an INDEPENDENT two-sample mean-difference
    estimand ``mean(a) - mean(b)``, via a closed-form (no-bootstrap)
    construction -- the independent-groups analogue of
    :func:`_ppi_paired_t_interval`. Not registered with :func:`evalstats.
    ppi.correct`'s analytic-backend dispatch (which requires no
    covariates -- see that function's ``X_lab``/``X_unlab`` handling);
    this is a standalone function callers reach directly instead.

    Each group's own PPI-corrected mean/variance moments come from
    :func:`evalstats.ppi._analytic_mean_point_se` (``power_tune=False``)
    or, when ``power_tune=True``, from a SHARED lambda estimated once
    from both groups' POOLED labeled+unlabeled data
    (:func:`evalstats.ppi._pooled_two_group_lambda`) rather than each
    group independently estimating its own -- per-group lambda is fine
    under MCAR, but under MNAR (label selection correlated with an
    item's own value) it can distort the two groups' labeled subsamples
    asymmetrically, and a per-group lambda has no data to average that
    distortion out over (worst observed case: 0.260 rejection rate under
    the null on a binary MNAR scenario, vs. 0.038 after pooling -- see
    simulations/out/results_why_ppi_shrink_1_over_0.md's ttest-binary
    addendum). Pooling also, as a side effect, further reduces the
    original MCAR near-boundary inflation this construction was built to
    fix (roughly doubling the effective sample the lambda ratio is
    estimated from). Groups A and B are independent samples, so
    ``Var(A - B) = Var(A) + Var(B))`` plus, when lambda is shared, a joint
    lambda-uncertainty term (see ``_pooled_two_group_lambda``'s
    docstring); their two (generally unequal) degrees of freedom are
    combined via :func:`evalstats.ppi._cross_fit_satterthwaite_df` --
    despite the name, a generic two-independent-component Satterthwaite
    combiner (introduced for wilcoxon's cross-fitted CI, reused here as-is;
    nothing about it is specific to cross-fitting), computed from each
    group's own pre-joint-inflation variance (independent by construction),
    matching how :func:`_analytic_mean_point_se` already leaves df
    unadjusted for its own single-estimator lambda inflation term.

    Built specifically to fix ttest's real, validated small-sample
    weakness on binary/discrete proportion data: ``correct()``'s general
    percentile bootstrap (what ``ttest()``/``_ppi_two_sample`` used
    exclusively before this, since covariate-based estimators can never
    reach an analytic backend through ``correct()``'s own dispatch)
    undercovers on near-boundary discrete proportions -- the same broad
    "percentile bootstrap + discreteness" failure family already
    documented for the median-under-ties case (``_tie_jitter_scale``),
    just triggered here by boundary proximity rather than ties. Unlike
    ``PPI_WILSON`` (:func:`_ppi_single_wilson`), this does NOT clamp the
    resulting interval to a proportion's valid range -- kept general
    since this same construction also serves ``ttest()``'s continuous
    case, where a [-1, 1]-style clamp would be actively wrong. See
    simulations/out/results_why_ppi_shrink_1_over_0.md's ttest-binary
    addendum for the full diagnosis and validation.

    A position is included in a group's labeled set only when that
    group's own label is non-NaN; each group's masking is independent
    (unlike :func:`_ppi_paired_t_interval`, there is no shared pairing).
    """
    from evalstats.ppi import (
        _analytic_mean_point_se, _analytic_mean_point_se_given_lambda,
        _pooled_two_group_lambda, _cross_fit_satterthwaite_df,
    )
    from scipy.stats import t as _t_dist_local

    mask_a = ~np.isnan(a_lab)
    mask_b = ~np.isnan(b_lab)
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        raise ValueError(
            "Both groups need at least one labeled item in a_lab and b_lab."
        )

    if power_tune:
        lam, var_lam = _pooled_two_group_lambda(
            a_lab[mask_a], a[mask_a], a[~mask_a],
            b_lab[mask_b], b[mask_b], b[~mask_b],
        )
        est_a, var_a, f_unlab_a, f_lab_a, rect_a, r_a, df_a = _analytic_mean_point_se_given_lambda(
            a_lab[mask_a], a[mask_a], a[~mask_a], lam,
        )
        est_b, var_b, f_unlab_b, f_lab_b, rect_b, r_b, df_b = _analytic_mean_point_se_given_lambda(
            b_lab[mask_b], b[mask_b], b[~mask_b], lam,
        )
        estimate = est_a - est_b
        var_estimate = var_a + var_b + ((r_a - r_b) ** 2) * var_lam
        lam_a = lam_b = lam
    else:
        est_a, se_a, f_unlab_a, f_lab_a, rect_a, lam_a, df_a = _analytic_mean_point_se(
            a_lab[mask_a], a[mask_a], a[~mask_a], power_tune=False,
        )
        est_b, se_b, f_unlab_b, f_lab_b, rect_b, lam_b, df_b = _analytic_mean_point_se(
            b_lab[mask_b], b[mask_b], b[~mask_b], power_tune=False,
        )
        estimate = est_a - est_b
        var_a, var_b = se_a * se_a, se_b * se_b
        var_estimate = var_a + var_b

    se = float(np.sqrt(var_estimate))
    df = (
        _cross_fit_satterthwaite_df(var_a, float(df_a), var_b, float(df_b))
        if var_a > 0.0 and var_b > 0.0 else max(float(df_a), float(df_b), 1.0)
    )

    if se <= 0.0:
        ci_low = ci_high = estimate
        p_value = 1.0 if abs(estimate) < 1e-12 else 0.0
    else:
        t_crit = float(_t_dist_local.ppf(1.0 - alpha / 2.0, df))
        ci_low, ci_high = estimate - t_crit * se, estimate + t_crit * se
        p_value = min(max(float(2.0 * (1.0 - _t_dist_local.cdf(abs(estimate) / se, df))), 0.0), 1.0)

    # Combined lambda for reporting. When power_tune=True this is just
    # `lam` (both groups share the same pooled value, lam_a == lam_b);
    # kept as a labeled-sample-size-weighted average for generality and
    # to mirror wilcoxon's cross-fit combined-lambda convention.
    n_lab_a, n_lab_b = int(mask_a.sum()), int(mask_b.sum())
    if lam_a is not None and lam_b is not None:
        lam_combined = (n_lab_a * lam_a + n_lab_b * lam_b) / (n_lab_a + n_lab_b)
    else:
        lam_combined = None

    from evalstats.ppi import PPIResult
    return PPIResult(
        estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
        llm_estimate=float(f_unlab_a - f_unlab_b), human_estimate=float(f_lab_a - f_lab_b),
        rectifier=float(rect_a - rect_b), p_value=p_value, lam=lam_combined,
    )


def _ppi_single_logit_t(
    a: np.ndarray, a_lab: np.ndarray, alpha: float, lo: float = 0.0, hi: float = 1.0, power_tune: bool = True,
):
    """PPI correction for a single-sample mean estimand on a [lo, hi]-
    bounded numeric scale (continuous/likert/grades), via the closed-form
    logit-t construction -- evalstats.ppi._analytic_logit_t_correct.
    Single-sample sibling of _ppi_paired_logit_t; [lo,hi]-bounded analogue
    of _ppi_single_t_interval (identical closed-form point estimate/
    variance -- only the CI's shape differs).

    ``lo, hi`` default to (0.0, 1.0) -- the only range evalstats.api's PPI
    alignment path currently supports (its data_kind="bounded_01" means
    raw scores are LITERALLY in [0, 1] -- see
    evalstats.api._run_alignment_ppi's is_bounded_01_scores check, which
    has no score_range concept). The simulation harness (whose likert/
    grades scenarios are NOT natively [0, 1]) passes its own
    EVAL_TYPE_SCALE_BOUNDS[eval_type] explicitly instead.

    A position is included in the labeled set only when ``a_lab[i]`` is
    non-NaN.

    ``power_tune`` mirrors :func:`evalstats.ppi.correct`'s parameter of the
    same name -- default True, matching every other paired/single PPI
    wrapper in this module.

    Passes ``label_shift_robust=True`` to :func:`evalstats.ppi.
    _analytic_logit_t_correct` -- see :func:`_ppi_single_t_interval`'s
    docstring for why (identical mechanism, this estimand's [lo,hi]-bounded
    analogue).
    """
    from evalstats.ppi import _analytic_logit_t_correct

    mask = ~np.isnan(a_lab)
    if mask.sum() == 0:
        raise ValueError("No positions have human labels in a_lab.")

    all_values = np.asarray(a, dtype=float)
    values_unlab = all_values[~mask]
    values_lab_llm = all_values[mask]
    values_lab_true = np.asarray(a_lab, dtype=float)[mask]

    return _analytic_logit_t_correct(
        values_lab_true, values_lab_llm, values_unlab, alpha, power_tune=power_tune, lo=lo, hi=hi,
        label_shift_robust=True,
    )


def _ppi_paired_logit_t(
    a: np.ndarray, b: np.ndarray, a_lab: np.ndarray, b_lab: np.ndarray, alpha: float,
    lo: float = 0.0, hi: float = 1.0, power_tune: bool = True,
):
    """PPI correction for a paired mean-difference estimand on a [lo, hi]-
    bounded numeric scale, via the closed-form logit-t construction. A
    signed difference of two [lo, hi] scores spans [-(hi-lo), hi-lo], not
    [lo, hi] itself -- rescaled the same way evalstats.core.paired's own
    (non-PPI) "logit_t" pairwise-CI branch handles a paired logit-t CI:
    diff bounds are derived from the SCORE's own [lo, hi] span, not
    passed directly.

    Pairing is by array position, matching _ppi_paired_t_interval; a
    position is included in the labeled set only when *both* ``a_lab[i]``
    and ``b_lab[i]`` are non-NaN.

    ``power_tune`` mirrors :func:`evalstats.ppi.correct`'s parameter of the
    same name -- default True, matching every other paired/single PPI
    wrapper in this module.
    """
    from evalstats.ppi import _analytic_logit_t_correct

    mask = ~np.isnan(a_lab) & ~np.isnan(b_lab)
    if mask.sum() == 0:
        raise ValueError(
            "No positions have human labels for both groups in a_lab and b_lab."
        )

    all_diffs = a - b
    diffs_unlab = all_diffs[~mask]
    diffs_lab_llm = all_diffs[mask]
    diffs_lab_true = (a_lab - b_lab)[mask]

    diff_span = hi - lo
    diff_lo, diff_hi = -diff_span, diff_span

    return _analytic_logit_t_correct(
        diffs_lab_true, diffs_lab_llm, diffs_unlab, alpha, power_tune=power_tune, lo=diff_lo, hi=diff_hi,
    )


def _sanitize_multigroup_ppi_labels(
    groups: list[np.ndarray],
    groups_lab,
    *,
    repeated: bool,
    test_label: str = "one-way ANOVA",
) -> list[np.ndarray]:
    """Validate sparse labels for multi-group ANOVA/Friedman wrappers.

    For repeated-measures designs (ANOVA or Friedman), an effective labeled
    unit is a subject with non-NaN labels in all conditions. For independent
    one-way ANOVA, effective labels are counted across all groups.
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
            f"for repeated-measures PPI {test_label} (found {n_effective})."
        )
        warn_msg = (
            "Only {n} subjects have labels in all conditions. "
            "PPI bootstrap can undercover below 30 labels; consider labeling more subjects."
        )
    else:
        n_effective = int(sum(np.sum(~np.isnan(lab_arr)) for lab_arr in labs))
        min_msg = (
            f"At least 15 human labels are required for PPI {test_label} "
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


def _friedman_rank_variance(matrix: np.ndarray) -> float:
    """Condition variance of within-subject ranks — the Friedman analog of
    :func:`_repeated_condition_variance`.

    Each row (subject) is rank-transformed across conditions (average ranks
    for ties) before taking the variance of the per-condition mean ranks.
    Row means of a rank-transformed row are always the constant ``(k+1)/2``
    regardless of ties, so row-centering inside ``_repeated_condition_variance``
    is a harmless no-op here — this is exactly "repeated-measures ANOVA run on
    ranks instead of raw scores," the classical Iman–Davenport equivalence
    that underlies the Friedman test.
    """
    if matrix.size == 0:
        return 0.0
    ranks = _scipy_stats.rankdata(matrix, axis=1, method="average")
    return _repeated_condition_variance(ranks)


def _noncentral_f_ci_lambda(f_obs: float, dfn: float, dfd: float, alpha: float) -> tuple[float, float]:
    """Equal-tailed confidence interval for the noncentrality parameter λ of
    a noncentral-F distribution, via test inversion (Steiger & Fouladi,
    1997 -- the standard construction for ANOVA effect-size CIs, also used
    by e.g. R's ``MBESS::conf.limits.ncf``).

    λ_L solves ``P(F ≥ f_obs | ncf(dfn, dfd, λ_L)) = α/2`` (f_obs sits at
    the UPPER tail of ncf(λ_L) -- λ_L is the smallest noncentrality still
    barely consistent with an observation this large). λ_U solves
    ``P(F ≥ f_obs | ncf(dfn, dfd, λ_U)) = 1 − α/2`` (f_obs sits at the LOWER
    tail of ncf(λ_U) -- λ_U is the largest noncentrality still barely
    consistent with an observation this small). Both floored at 0 (a
    noncentrality parameter can't be negative) when even λ=0 doesn't reach
    the target tail probability -- this is the correct, not a degenerate,
    outcome: it means the data are so weak (or so strong) that the interval
    collapses against the boundary on that side.

    Exists because the naive percentile-bootstrap CI these three ANOVA-
    family PPI corrections' p-values already replaced (see e.g.
    :func:`_ppi_anova_independent_p_value`'s docstring) is anti-conservative
    for a variance-like, bounded-below-by-zero estimand -- the p-value was
    fixed via this same noncentral-F approach; the CI wasn't, until this
    function existed. Confirmed via ``tests/test_ppi_corrections.py``: the
    old CI and the corrected p-value could disagree outright on the
    reject/don't-reject decision (e.g. p=1.9e-10 while the old CI still
    contained the null) -- a test-inversion CI on the SAME F-statistic the
    p-value already uses is guaranteed consistent with it by construction."""
    from scipy import optimize
    from scipy.stats import ncf as _ncf, f as _f

    def sf(lam: float) -> float:
        if lam <= 1e-10:
            return float(_f.sf(f_obs, dfn, dfd))
        return float(_ncf.sf(f_obs, dfn, dfd, lam))

    def _solve(target: float) -> float:
        if sf(0.0) >= target:
            return 0.0
        hi = 1.0
        while sf(hi) < target and hi < 1e9:
            hi *= 2.0
        if sf(hi) < target:
            return hi  # didn't converge within range; return the (huge) bound rather than raise
        return float(optimize.brentq(lambda lam: sf(lam) - target, 0.0, hi))

    lam_L = _solve(alpha / 2.0)
    lam_U = _solve(1.0 - alpha / 2.0)
    return lam_L, lam_U


def _ppi_anova_independent_f_stat(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], k: int,
    power_tune: bool = True,
) -> Optional[dict]:
    """Shared F-statistic computation for independent-groups ANOVA's PPI
    correction -- factored out so the p-value and the (test-inversion) CI
    are guaranteed to agree, since both are derived from this exact same
    ``f_corr``/``dfn``/``dfd``/``denom`` rather than two separately-computed
    pipelines. Returns None under the same degenerate condition the old
    ``_ppi_anova_independent_p_value`` returned None for (a group with zero
    labels).

    ``power_tune`` (default *True*, matching every existing caller): see
    the docstring inside the ``if power_tune:`` branch below for why this
    needed a different per-group construction than the ``power_tune=False``
    branch's fixed-lambda=1 one, not just a lambda<1 plug-in, and for the
    pooled-lambda fix (2026-08-15) to a real-data Type-I inflation found
    with the original per-group-lambda version of this construction; see
    ``simulations/out/results_why_ppi_shrink_1_over_0.md``'s ANOVA
    power-tuning addenda for the full investigation."""
    masks = [~np.isnan(g_lab) for g_lab in groups_lab]
    n_lab_arr = np.array([int(m.sum()) for m in masks], dtype=float)

    if np.any(n_lab_arr == 0):
        return None

    ns = np.array([len(g) for g in groups], dtype=float)
    N = int(ns.sum())

    if power_tune:
        # A prior attempt at power-tuning this test (see correct()'s
        # power_tune docstring / simulations/harness/README.md's "PPI++
        # power-tuning" section) inflated Type-I to ~19%. That attempt (per
        # its own description) plugged a variance-minimizing lambda<1 into
        # THIS function's existing g.mean() + lambda*rectifier construction
        # below -- but that construction is only unbiased at lambda=1: `g`
        # is the FULL group's judge scores (labeled+unlabeled combined,
        # weight 1, fixed), and only the rectifier gets weighted by lambda,
        # so E[corr_means_i] = true_mean_i + (1-lambda)*bias_i -- genuinely
        # biased whenever lambda<1 and the judge has a real bias. Squared
        # into ss_between, that's exactly the reintroduced-bias failure
        # mode the README describes -- but it's an artifact of THIS
        # function's shortcut construction, not an inherent property of a
        # quadratic estimand.
        #
        # The standard PPI construction (`f_lab + lambda*(f_unlab -
        # f_hat_lab)`, full weight on the HUMAN term, disjoint unlabeled
        # sample) doesn't have this problem: its rectifier has expectation
        # ~=0 at ANY lambda (same population, disjoint samples), which is
        # exactly the property that makes power-tuning safe for a plain
        # mean. Reusing that exact construction here (once per group, via
        # _analytic_mean_point_se -- already gets the adaptive-target
        # shrinkage for free) keeps E[corr_means_i] = true_mean_i
        # regardless of lambda, so ss_between's null-expectation identity
        # should hold for ANY lambda, not just 1 -- AS LONG AS `denom`
        # below is also built from this same per-group variance, not the
        # power_tune=False branch's lambda=1-specific inflation formula.
        #
        # Lambda is estimated ONCE, POOLED across all k groups' labeled+
        # unlabeled data (evalstats.ppi._pooled_k_group_lambda), not
        # independently per group -- per-group lambda estimation was found
        # (2026-08-15, real-data validation) to systematically UNDERSTATE
        # `denom` (mean(ss_between)/(k-1) exceeded mean(denom) by ~14%
        # under a real MCAR null): each group's lambda is chosen
        # specifically to minimize THAT group's own reported variance
        # using that same finite sample's noisy moments, an
        # "argmin-then-evaluate-at-the-argmin" optimism bias distinct from
        # lambda's own sampling uncertainty (which
        # evalstats.ppi._lambda_var_inflation already separately corrects
        # for). Pooling increases the effective sample lambda is estimated
        # from, shrinking that optimism gap -- see _pooled_k_group_lambda's
        # docstring and simulations/out/results_why_ppi_shrink_1_over_0.md's
        # real-data ANOVA addendum for the full ground-truth validation
        # (real + synthetic, null + power, no regression found anywhere).
        from evalstats.ppi import (
            _analytic_mean_point_se_given_lambda, _pooled_k_group_lambda,
        )

        fully_labeled = [len(g[~m]) == 0 for g, m in zip(groups, masks)]
        pool_idx = [i for i, fl in enumerate(fully_labeled) if not fl]
        if pool_idx:
            lam, var_lam = _pooled_k_group_lambda(
                [groups_lab[i][masks[i]] for i in pool_idx],
                [groups[i][masks[i]] for i in pool_idx],
                [groups[i][~masks[i]] for i in pool_idx],
            )
        else:
            lam, var_lam = 1.0, 0.0

        corr_means = np.empty(k)
        var_ppi_per_group = np.empty(k)
        r_terms = np.zeros(k)
        for i, (g, g_lab, mask) in enumerate(zip(groups, groups_lab, masks)):
            Y_lab_i = g_lab[mask]
            Y_hat_lab_i = g[mask]
            Y_hat_unlab_i = g[~mask]
            if fully_labeled[i]:
                # This group is fully labeled -- lambda/the judge-side
                # rectifier plays no role for it at all, so it's excluded
                # from the pooled lambda estimate above too (see
                # pool_idx). Fall back to the human-labeled mean directly:
                # the power_tune=False branch's fixed-lambda=1 construction
                # (g.mean() + (g_lab[mask].mean() - g[mask].mean()))
                # reduces to exactly this when mask is all True, so this
                # stays consistent with that branch at 100% labeling.
                est_i = float(Y_lab_i.mean())
                n_lab_i = len(Y_lab_i)
                var_i = float(np.var(Y_lab_i, ddof=1) / n_lab_i) if n_lab_i > 1 else 0.0
            else:
                est_i, var_i, _, _, _, r_term_i, _ = _analytic_mean_point_se_given_lambda(
                    Y_lab_i, Y_hat_lab_i, Y_hat_unlab_i, lam,
                )
                r_terms[i] = r_term_i
            corr_means[i] = est_i
            var_ppi_per_group[i] = var_i
    else:
        # PPI-corrected group means: μ̂ᵢ_PPI = mean(llm_all_i) + (mean(human_lab_i) - mean(llm_lab_i))
        corr_means = np.array([
            g.mean() + (g_lab[mask].mean() - g[mask].mean())
            for g, g_lab, mask in zip(groups, groups_lab, masks)
        ])

    grand = (ns * corr_means).sum() / N
    ss_between = float(np.sum(ns * (corr_means - grand) ** 2))

    # Within-group MS from full LLM data  (≈ σ² + σ_llm²... but see the
    # Cov(true, noise) correction below: that's only exact if the judge's
    # error is uncorrelated with the true score.)
    ss_within = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    ms_within = ss_within / (N - k)

    if power_tune:
        # Per-group inflation: Var[μ̂ᵢ_PPI(lambda_i)] / Var[μ̂ᵢ_LLM], matching
        # the power_tune=False branch's own convention (see its docstring
        # below: "Var[μ̂ᵢ_PPI] = σ²/nᵢ + σ_llm² × (1/n_lab_i − 1/nᵢ)", a
        # MEAN-scale quantity, divided by Var[μ̂ᵢ_LLM] = ms_within/nᵢ --
        # the nᵢ's cancel to leave (σ² + σ_llm²×(nᵢ/n_lab_i−1))/ms_within,
        # which IS that branch's actual formula). var_ppi_per_group above
        # is _analytic_mean_point_se's se², already the correct MEAN-scale
        # Var[μ̂ᵢ_PPI(lambda_i)] -- so it needs the same "* nᵢ" to divide
        # by Var[μ̂ᵢ_LLM] rather than ms_within directly; leaving that out
        # was an initial bug caught by validate_anova_power_tune.py (Type-I
        # went to 1.0 -- denom came out ~nᵢ times too small).
        #
        # var_ppi_per_group deliberately excludes lambda's own estimation
        # uncertainty (_analytic_mean_point_se_given_lambda's docstring --
        # a pooled/shared lambda needs that added jointly, not once per
        # group). Added here as r_term_i^2 * var_lam per group, same
        # MEAN-scale-to-Var[μ̂ᵢ_LLM] weighting as the base term; a
        # first-order approximation (treats the shared-lambda perturbation
        # as an independent per-group addition rather than deriving
        # SS_between's full induced covariance structure under a
        # perfectly-correlated-across-groups lambda) that the ground-truth
        # Monte Carlo validation (see this branch's docstring above) found
        # sufficient in practice -- no residual miscalibration or power
        # cost detected on any tested cell.
        inflation_per_group = (var_ppi_per_group + (r_terms ** 2) * var_lam) * ns / ms_within
    else:
        # Per-group LLM noise variance σ_llm_i² from labeled residuals (llm − human),
        # AND per-group Cov(true, noise) from the same labeled subset.
        #
        # Var(judge_score) = σ² + σ_llm² + 2·Cov(true, noise) -- the "+2·Cov" term
        # is only zero if the judge's error is independent of the true score,
        # which a real LLM judge routinely violates (typically a negative
        # correlation: "regression to the mean"/compression toward the middle of
        # a bounded scale). Omitting it makes ms_within (computed from the
        # judge's own, compressed scores) come out below what "σ² + σ_llm²,
        # independent" would predict, so subtracting only σ_llm_sq_weighted
        # systematically under-estimates σ² by ~2·Cov(true, noise) -- a fixed
        # absolute shortfall that becomes a larger fraction of denom (and thus a
        # larger Type-I inflation) as label_frac grows, since the other term in
        # inflation_per_group shrinks with n_lab while this one doesn't. Not
        # visible on i.i.d.-Gaussian-noise synthetic data (Cov(true, noise) = 0
        # by construction) -- specific to judges whose error genuinely
        # correlates with the true value.
        sigma_llm_sq_per_group = np.zeros(k)
        cov_true_noise_per_group = np.zeros(k)
        for i, (g, g_lab, mask) in enumerate(zip(groups, groups_lab, masks)):
            if mask.sum() > 1:
                noise = g[mask] - g_lab[mask]
                sigma_llm_sq_per_group[i] = float(np.var(noise, ddof=1))
                cov_true_noise_per_group[i] = float(np.cov(g_lab[mask], noise, ddof=1)[0, 1])

        # n_i-weighted average (matches how ms_within pools group residuals)
        sigma_llm_sq_weighted = float(np.dot(ns, sigma_llm_sq_per_group) / N)
        cov_true_noise_weighted = float(np.dot(ns, cov_true_noise_per_group) / N)
        sigma_sq = max(ms_within - sigma_llm_sq_weighted - 2.0 * cov_true_noise_weighted, 0.0)

        # Per-group inflation: Var[μ̂ᵢ_PPI] / Var[μ̂ᵢ_LLM]
        # = (σ² + σ_llm_i² × (nᵢ/n_lab_i − 1)) / ms_within
        inflation_per_group = (sigma_sq + sigma_llm_sq_per_group * (ns / n_lab_arr - 1.0)) / ms_within

    # Weights: (N−nᵢ)/(N(k−1)) derived from E[SS_between_corr] = Σᵢ nᵢ(N−nᵢ)/N · Var[μ̂ᵢ]
    # -- a general identity for independent (across groups) unbiased
    # per-group estimates with their own variance, so it applies unchanged
    # to either branch's Var[μ̂ᵢ] above. For balanced groups these equal
    # nᵢ/N (same as old formula).
    w = (N - ns) / (N * (k - 1))
    inflation = float(np.dot(w, inflation_per_group))
    inflation = max(inflation, 1e-6)

    denom = ms_within * inflation
    f_corr = (ss_between / (k - 1)) / denom
    # theta (between-group variance, _anova_between_variance_from_labeled's
    # own scale) relates to the F-statistic's noncentrality lambda via
    # E[ss_between/(k-1)] = denom*(1 + lambda/(k-1)) and, by the classical
    # ANOVA identity, E[ss_between] = (k-1)*denom + N*theta -- combining:
    # lambda = N*theta/denom, i.e. theta = lambda*denom/N.
    return {"f_corr": f_corr, "dfn": k - 1, "dfd": N - k, "denom": denom, "scale": float(N)}


def _ppi_anova_independent_p_value(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    k: int,
    power_tune: bool = True,
) -> Optional[float]:
    """Corrected p-value for independent ANOVA via per-group PPI mean corrections.

    The standard PPI bootstrap p-value is anti-conservative for the between-group
    variance estimand because it's bounded below by zero.  Instead, we apply PPI
    corrections at the level of group means (which ARE a valid, symmetric PPI
    estimand), compute a corrected F-statistic, and use the F-distribution.

    Var[μ̂ᵢ_PPI] = σ²/nᵢ + σ_llm² × (1/n_lab_i − 1/nᵢ)
    where σ² is the true within-group variance and σ_llm² is LLM noise variance.
    Both are estimated from the labeled residuals (llm − human within each group)
    -- σ² specifically as ms_within − σ_llm² − 2·Cov(true, noise), NOT just
    ms_within − σ_llm² (see _ppi_anova_independent_f_stat's inline comment for
    why the Cov term matters: a real LLM judge's error is often correlated with
    the true score -- e.g. "regression to the mean" toward a bounded scale's
    middle -- which the simpler independent-noise formula misses entirely).
    The inflation factor Var[μ̂ᵢ_PPI] / Var[μ̂ᵢ_LLM] scales the F denominator so
    the test is calibrated regardless of how noisy the LLM judge is.

    See :func:`_ppi_anova_independent_ci` for the CI derived from this same
    F-statistic (guaranteed consistent with this p-value by construction)."""
    stat = _ppi_anova_independent_f_stat(groups, groups_lab, k, power_tune=power_tune)
    if stat is None:
        return None
    if stat["f_corr"] <= 0.0:
        return 1.0
    return float(_scipy_stats.f.sf(stat["f_corr"], dfn=stat["dfn"], dfd=stat["dfd"]))


def _ppi_anova_independent_ci(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], k: int, alpha: float,
    power_tune: bool = True,
) -> Optional[tuple[float, float, float]]:
    """(estimate, ci_low, ci_high) for independent ANOVA's between-group
    variance, via test-inversion on the SAME F-statistic
    :func:`_ppi_anova_independent_p_value` uses -- see
    :func:`_noncentral_f_ci_lambda`. Returns None under the same degenerate
    condition the p-value function does."""
    stat = _ppi_anova_independent_f_stat(groups, groups_lab, k, power_tune=power_tune)
    if stat is None:
        return None
    f_corr, dfn, dfd, denom, scale = stat["f_corr"], stat["dfn"], stat["dfd"], stat["denom"], stat["scale"]
    if f_corr <= 0.0 or not np.isfinite(f_corr):
        return 0.0, 0.0, 0.0
    # theta_hat = num_ms*(k-1)/N = f_corr*denom*dfn/scale (see
    # _ppi_anova_independent_f_stat's comment for the lambda<->theta relation)
    # is a sum-of-squares-type quantity, inherently >= 0 BEFORE debiasing, so
    # it carries a persistent positive bias under repeated sampling even at
    # the true null (theta=0) -- not just noise around zero. The classical
    # ANOVA identity in the comment above (E[ss_between] = (k-1)*denom +
    # N*theta) gives E[raw estimate] = dfn*denom/scale > 0 whenever theta=0.
    # Subtracting the expected-under-H0 term (dfn*denom/scale) is the same
    # debiasing eta-squared -> omega-squared/epsilon-squared does
    # classically.
    #
    # Deliberately NOT clamped at 0 afterward: a quantity that's genuinely
    # centered at 0 under the null still has roughly half its sampling
    # distribution below 0, and clamping away only that half strictly
    # increases the mean of what's left, reintroducing the same upward bias
    # one step removed. The CI construction below (a proper noncentral-F
    # test-inversion) doesn't depend on the point estimate's sign, so
    # nothing downstream requires non-negativity either. We therefore report
    # the unclamped, possibly-negative debiased estimate directly, the same
    # convention omega-squared/epsilon-squared use: a negative value is
    # evidence of no effect, not an invalid measurement.
    #
    # f_corr/dfn/dfd (what the p-value function uses) and the CI bounds
    # below (a proper noncentral-F test-inversion via _noncentral_f_ci_lambda)
    # are both unaffected by this change -- neither depends on `estimate`.
    estimate = (f_corr - 1.0) * dfn * denom / scale
    lam_L, lam_U = _noncentral_f_ci_lambda(f_corr, dfn, dfd, alpha)
    return estimate, lam_L * denom / scale, lam_U * denom / scale


def _repeated_anova_lambda_raw(human_lab, llm_lab, llm_unlab, P, k):
    """Raw (unshrunk) scalar lambda for repeated-measures ANOVA's vector
    estimand, minimizing trace(P @ Var[cond_means_ppi(lambda)] @ P) over
    lambda -- the natural vector generalization of the classical PPI++
    lambda* = Cov(true,llm)/[Var(unlab)+Var(hat_lab)], projected through
    the same centering matrix P the F-statistic itself uses. Returns
    (lam_raw, var_unlab, var_lab_llm, cov_cross) so callers can reuse the
    covariance pieces for the variance formula without recomputing them."""
    n_lab_, n_unlab_ = len(human_lab), len(llm_unlab)
    cov_cross = np.cov(human_lab, llm_lab, rowvar=False)[:k, k:] / n_lab_
    var_lab_llm = np.atleast_2d(np.cov(llm_lab, rowvar=False)) / n_lab_
    var_unlab = np.atleast_2d(np.cov(llm_unlab, rowvar=False)) / n_unlab_
    B = var_unlab + var_lab_llm
    num = np.trace(P @ cov_cross @ P)
    den = np.trace(P @ B @ P)
    lam_raw = num / den if den > 1e-12 else 1.0
    lam_raw = min(max(lam_raw, 0.0), 1.0)
    return lam_raw, var_unlab, var_lab_llm, cov_cross


def _repeated_anova_lambda_replicates(human_lab, llm_lab, llm_unlab, P, k, n_lab, n_boot=500):
    """Raw-lambda replicates for :func:`evalstats.ppi._adaptive_shrink_lambda`,
    for repeated-measures ANOVA's vector estimand -- a micro-bootstrap of
    just the labeled subjects (paired human/llm rows resampled together,
    same idea as :func:`evalstats.ppi._analytic_mean_lambda_replicates`),
    holding the large unlabeled sample's covariance fixed (stable, no need
    to resample it)."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_lab, size=(n_boot, n_lab))
    lam_reps = np.empty(n_boot)
    for b in range(n_boot):
        lam_reps[b], _, _, _ = _repeated_anova_lambda_raw(human_lab[idx[b]], llm_lab[idx[b]], llm_unlab, P, k)
    return lam_reps


def _ppi_anova_repeated_f_stat(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], k: int,
    power_tune: bool = True,
) -> Optional[dict]:
    """Shared F-statistic computation for repeated-measures ANOVA's PPI
    correction -- see :func:`_ppi_anova_independent_f_stat`'s docstring for
    why this is factored out (p-value/CI consistency by construction).

    ``power_tune=False`` (default): fixed lambda=1 rectifier, unchanged
    from the original implementation --

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

    ``power_tune=True``: EXPERIMENTAL. The fixed-lambda=1 construction above
    uses the FULL sample (labeled + unlabeled) for ``cond_means_llm``, which
    only cancels judge bias at lambda=1 -- the identical construction bug
    :func:`_ppi_anova_independent_f_stat` had before its own power_tune fix
    (verified via simulation: residual bias scales as (1-lambda)*bias, zero
    only at lambda=1). Fixed the same way: rebuilt around the standard
    disjoint PPI construction ``f_lab + lambda*(f_unlab - f_hat_lab))``,
    generalized to the k-dimensional condition-contrast vector, with a
    single SHARED scalar lambda (not per-condition -- the k conditions
    share one judge, so one lambda) chosen to minimize
    trace(P @ Var[cond_means_ppi(lambda)] @ P) -- the natural vector
    generalization of the classical PPI++ lambda* derivation, projected
    through the same centering matrix P the F-statistic itself uses.
    lambda is shrunk toward an adaptive target exactly as every other
    power_tune site in this codebase (see evalstats.ppi._adaptive_shrink_lambda),
    and the reported variance includes the delta-method lambda-uncertainty
    term (evalstats.ppi._lambda_var_inflation's rationale, generalized from
    a scalar r_term**2*Var(lambda_hat) to the matrix outer product
    np.outer(r_term, r_term)*Var(lambda_hat), since r_term is now a
    k-vector here).

    Validated via simulation (see
    simulations/out/results_why_ppi_shrink_1_over_0.md's repeated-ANOVA
    addendum): Type-I stays controlled (elevated at n_lab~15-20, converging
    to the fixed-lambda baseline by n_lab~40-60, the same small-sample
    pattern documented for every other adaptively-shrunk site in this
    codebase), with large power gains for poor/uninformative judges.

    Shares :func:`_ppi_friedman_f_stat`'s ``_repeated_anova_lambda_raw``/
    ``_repeated_anova_lambda_replicates`` machinery and, as of Addendum
    30, its closed-form variance-inflation fix
    (:func:`evalstats.ppi._shrunk_lambda_variance`) -- see that
    function's docstring for the mild residual power_tune=True inflation
    this partially (not fully) addresses, and why an n_lab-adaptive
    default switch was investigated and found not warranted."""
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

    unlab_idx = np.where(~overlap)[0]
    n_unlab = len(unlab_idx)

    # n_unlab<2 (includes the fully-labeled n_unlab==0 case) can't support
    # the adaptive lambda estimate below (its replicate-resampling needs at
    # least 2 unlabeled subjects) -- fall through to the fixed-lambda=1
    # computation below instead of erroring, same fallback used for
    # n_lab<2. At 100% labeling that fixed-lambda=1 construction already
    # reduces cleanly to the human-labeled result (no unlabeled
    # extrapolation term), matching _ppi_anova_independent_f_stat's
    # per-group fallback for the same edge case.
    if power_tune and n_unlab >= 2 and n_lab >= 2:
        llm_unlab = centered_llm_all[unlab_idx]
        f_unlab = llm_unlab.mean(axis=0)
        f_lab_llm = centered_llm_lab.mean(axis=0)
        f_lab_human = centered_human_lab.mean(axis=0)
        r_term = f_unlab - f_lab_llm  # (k,) -- disjoint unlabeled minus labeled llm

        P = np.eye(k) - np.ones((k, k), dtype=float) / float(k)
        lam_raw, var_unlab, var_lab_llm, cov_cross = _repeated_anova_lambda_raw(
            centered_human_lab, centered_llm_lab, llm_unlab, P, k,
        )
        # Degenerate guard, same rationale as every other power_tune site
        # (evalstats.ppi._analytic_mean_point_se's docstring): a near-
        # constant labeled sample can't reveal covariance no matter how
        # it's resampled.
        raw_var_human = np.var(centered_human_lab, axis=0, ddof=1).sum()
        raw_var_llm_lab = np.var(centered_llm_lab, axis=0, ddof=1).sum()
        if n_lab <= 1 or raw_var_human < raw_var_llm_lab * 1e-6:
            lam_replicates = None
        else:
            lam_replicates = _repeated_anova_lambda_replicates(
                centered_human_lab, centered_llm_lab, llm_unlab, P, k, n_lab,
            )
        from evalstats.ppi import _adaptive_shrink_lambda, _shrunk_lambda_variance, _POWER_TUNE_SHRINKAGE_C
        lam = _adaptive_shrink_lambda(lam_raw, lam_replicates, n_lab)

        cond_means_ppi = f_lab_human + lam * r_term
        grand_ppi = cond_means_ppi.mean()
        ss_condition_corr = float(n_subjects * np.sum((cond_means_ppi - grand_ppi) ** 2))

        var_human = np.atleast_2d(np.cov(centered_human_lab, rowvar=False)) / n_lab
        var_matrix = var_human + lam * lam * (var_unlab + var_lab_llm) - 2.0 * lam * cov_cross
        if lam_replicates is not None and len(lam_replicates) > 1:
            # Closed-form Var(shrunk lambda), accounting for the shrinkage
            # target's own sampling uncertainty -- see
            # evalstats.ppi._shrunk_lambda_variance's docstring for the
            # derivation and simulations/out/results_why_ppi_shrink_1_over_0.md's
            # friedman power_tune=True addendum for the validation (a plain
            # Var(lam_raw) plug-in here, the prior construction, implicitly
            # assumes w=1/no shrinkage, understating this term).
            var_lam_raw = float(np.var(lam_replicates, ddof=1))
            w = n_lab / (n_lab + _POWER_TUNE_SHRINKAGE_C)
            var_lam_shrunk = _shrunk_lambda_variance(lam_raw, var_lam_raw, w)
            var_matrix = var_matrix + np.outer(r_term, r_term) * var_lam_shrunk
        var_matrix = (var_matrix + var_matrix.T) / 2.0  # symmetrize away float noise

        # E[ss_condition_corr] = n_subjects * trace(P @ Var[cond_means_ppi] @ P)
        # for a mean-zero (under H0) k-vector -- see this function's
        # power_tune=True docstring section; denom is that expectation
        # divided by (k-1) to match f_corr's convention.
        denom = float(n_subjects * np.trace(P @ var_matrix @ P) / (k - 1))
        denom = max(denom, 1e-8)
        f_corr = (ss_condition_corr / (k - 1)) / denom
        df_residual_eff = max((n_lab - 1) * (k - 1), 1)
        return {
            "f_corr": f_corr, "dfn": k - 1, "dfd": df_residual_eff,
            "denom": denom, "scale": float(n_subjects * k),
        }

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

    denom = ms_residual * inflation
    f_corr = (ss_condition_corr / (k - 1)) / denom

    # Finite labeled overlap adds estimation uncertainty to the rectifier.
    # Use a conservative effective denominator df capped by labeled contrasts.
    # This improves Type I calibration in sparse-label repeated settings.
    df_residual_eff = min(df_residual, max((n_lab - 1) * (k - 1), 1))
    # theta (_repeated_condition_variance's own scale, mean over k of
    # squared condition-mean deviations) relates to lambda the same way
    # friedman's does -- see _ppi_friedman_f_stat's comment: theta =
    # lambda*denom/(n_subjects*k).
    return {"f_corr": f_corr, "dfn": k - 1, "dfd": df_residual_eff, "denom": denom, "scale": float(n_subjects * k)}


def _ppi_anova_repeated_p_value(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    k: int,
    power_tune: bool = True,
) -> Optional[float]:
    """Corrected p-value for repeated-measures ANOVA via per-condition PPI corrections.

    Mirror of _ppi_anova_independent_p_value for the within-subjects design.
    Corrections are applied to condition means after removing each subject's
    mean (matching the repeated-measures F-test). See
    :func:`_ppi_anova_repeated_f_stat` for the full derivation and
    :func:`_ppi_anova_repeated_ci` for the CI derived from this same
    F-statistic (guaranteed consistent with this p-value by construction)."""
    stat = _ppi_anova_repeated_f_stat(groups, groups_lab, k, power_tune=power_tune)
    if stat is None:
        return None
    if stat["f_corr"] <= 0.0:
        return 1.0
    return float(_scipy_stats.f.sf(stat["f_corr"], dfn=stat["dfn"], dfd=stat["dfd"]))


def _ppi_anova_repeated_ci(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], k: int, alpha: float,
    power_tune: bool = True,
) -> Optional[tuple[float, float, float]]:
    """(estimate, ci_low, ci_high) for repeated-measures ANOVA's condition
    variance, via test-inversion on the SAME F-statistic
    :func:`_ppi_anova_repeated_p_value` uses -- see
    :func:`_noncentral_f_ci_lambda`."""
    stat = _ppi_anova_repeated_f_stat(groups, groups_lab, k, power_tune=power_tune)
    if stat is None:
        return None
    f_corr, dfn, dfd, denom, scale = stat["f_corr"], stat["dfn"], stat["dfd"], stat["denom"], stat["scale"]
    if f_corr <= 0.0 or not np.isfinite(f_corr):
        return 0.0, 0.0, 0.0
    # Debiased the same way _ppi_anova_independent_ci's estimate is -- see
    # that function's comment for the full derivation/root-cause (this had
    # the identical max(.,0) floor, and the identical fix: report the
    # unclamped, possibly-negative debiased estimate directly, the same
    # convention omega-squared/epsilon-squared use). f_corr/dfn/dfd (the
    # p-value's inputs) and the CI bounds below are untouched by this.
    estimate = (f_corr - 1.0) * dfn * denom / scale
    lam_L, lam_U = _noncentral_f_ci_lambda(f_corr, dfn, dfd, alpha)
    return estimate, lam_L * denom / scale, lam_U * denom / scale


def _ppi_friedman_f_stat(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], k: int,
    power_tune: bool = True,
) -> Optional[dict]:
    """Shared F-statistic computation for Friedman's PPI correction -- see
    :func:`_ppi_anova_independent_f_stat`'s docstring for why this is
    factored out (p-value/CI consistency by construction).

    ``power_tune=True``: EXPERIMENTAL, reuses the SAME construction
    validated for :func:`_ppi_anova_repeated_f_stat` (disjoint unlabeled
    sample + a single shared scalar lambda minimizing
    trace(P @ Var[cond_means_ppi(lambda)] @ P), all from plain empirical
    covariances of rank-mean vectors -- see
    :func:`_repeated_anova_lambda_raw`/:func:`_repeated_anova_lambda_replicates`,
    reused here unchanged, just fed rank-transformed inputs instead of raw
    scores), applied to ranks instead of raw scores. This construction
    deliberately does NOT use an SS-decomposition residual anywhere (the
    thing this function's power_tune=False docstring documents as broken
    for ranks -- point 1 below): it only ever computes plain sample
    covariances of MEAN vectors (labeled human ranks, labeled/unlabeled
    llm ranks), which aren't subject to the same "anti-correlated with the
    tested effect" problem, since they're not residuals left over after
    removing an estimated condition effect. Validated via simulation (see
    simulations/out/results_why_ppi_shrink_1_over_0.md's Friedman
    addendum) -- Type-I stays controlled with real (if more modest than
    the continuous-score ANOVA cases, expected given ranks carry less
    information) power gains.

    Not a literal mirror of ``_ppi_anova_repeated_f_stat``, despite the
    similar structure. That function estimates its null/residual variance
    via the ANOVA sum-of-squares decomposition ``SS_residual = SS_total −
    SS_subjects − SS_condition``, then subtracts off the LLM-noise
    variance to recover the "true" residual variance -- valid because, for
    raw scores, ``SS_residual`` is a genuine estimate of the LLM-observed
    variance (``Σ_Y = Σ_true + Σ_llm_noise``). Neither step transfers to
    ranks:

    1. A rank matrix's ``SS_total`` (about the grand mean) is almost a
       fixed constant per row, independent of any condition effect -- under
       exchangeability each row is just some permutation of
       ``{1, ..., k}``, and its sum of squares about ``(k+1)/2`` depends
       only on the row's tie pattern, never on any true/biased condition
       effect. An SS-decomposition residual computed this way is
       anti-correlated with the very effect being tested (more signal
       leaves less "residual" by construction), which deflates the
       F-statistic's denominator and inflates Type I error.
    2. The quantity that does transfer is the classical Friedman result:
       under H₀, within-subject ranks of the human/ground-truth data are an
       exactly known, exchangeability-derived constant
       (``(k³−k)/12`` per row, with no ties) -- call it ``Σ_H``. Because it
       is already the human-side null baseline (not an LLM-side estimate
       contaminated by LLM bias), it must not have the LLM-noise variance
       subtracted back out before adding the rectifier's finite-sample noise
       term; ``ms_null`` already is ``Σ_H``.

       ``Σ_H`` is estimated (tie-aware, not the hardcoded no-tie constant)
       as a shrinkage blend of two per-row estimates, not from the full LLM
       rank matrix alone -- see ``ms_null``'s inline comment below for the
       mechanism: the full-matrix estimate assumes the LLM's tie structure
       resembles the human side's, which fails whenever the labeled data is
       exactly tied (true of every genuine repeated-measures Type-I null,
       since the same shared ground truth is revealed for all k
       conditions). The labeled human data's own row sum-of-squares is the
       asymptotically correct estimate as n_lab grows but too noisy alone
       at small n_lab; shrinking between the two closes that gap.

    Ranking is row-local (each subject's rank vector depends only on that
    subject's own scores), so ranking commutes with row selection: ranking
    the full matrix once and then slicing the labeled-subject rows gives the
    same result as ranking the labeled subset directly.

    Known, open limitation under MNAR labeling: Type-I error can run well
    above nominal, growing with population size N at fixed n_lab (unlike
    wilcoxon()'s small-n_lab residual, this is not a small-sample
    degeneracy -- friedman's correction is closed-form/analytic and never
    touches a bootstrap). Root cause: the labeled-subject sample used to
    estimate the LLM-noise variance is itself MNAR-selected, and under
    typical MNAR labeling (preferentially labeling subjects with a high
    mean truth level, whose within-subject scores tend to be less tied and
    less prone to rank-flipping from LLM noise) that sample systematically
    underestimates the population's true rank-contrast noise variance,
    which feeds a term that grows linearly with N. An inverse-propensity-
    weighting fix was prototyped and did not reliably close the gap (a
    regression on some cells), so this remains a documented, open
    limitation rather than an ``*_mnar_experimental`` opt-in -- see
    ``simulations/investigate_friedman_mnar.py`` for the investigation.
    MNAR is a documented, out-of-scope limitation for this package
    generally (see :func:`evalstats.ppi.correct`'s docstring); this is one
    more instance of that.

    Known, open (partially mitigated) limitation under ``power_tune=True``
    (MCAR labeling): a mild, broad-based Type-I inflation at small-to-
    moderate ``n_lab``, root-caused to the same same-sample lambda/point-
    estimate coupling that caused wilcoxon()'s much larger inflation
    (Addendum 28) -- but here the failure mode is a mild mean-level shift
    in ``f_corr``'s expectation, NOT a heavy tail like wilcoxon's, so
    wilcoxon's cross-fitting fix (and a joint-bootstrap variant, tried as
    a second candidate) do not transfer -- both were validated (ground-
    truth Monte Carlo variance checks plus rejection-rate sweeps) to make
    calibration WORSE, not better, and were rejected. What DOES help: the
    variance-inflation term now uses :func:`evalstats.ppi.
    _shrunk_lambda_variance`'s closed-form correction for the adaptive
    shrinkage TARGET's own sampling uncertainty (previously unaccounted
    for -- the prior term implicitly assumed no shrinkage, i.e. ``w=1``),
    which closes roughly 20-30% of the mean Type-I gap above nominal alpha
    (139-scenario sweep: mean 0.0542->0.0530, max 0.0767->0.0733) with no
    meaningful power cost -- real but partial, not a full fix. An n_lab-
    adaptive default switch (using ``power_tune=False`` below some n_lab
    threshold) was also investigated as a follow-up and found NOT
    warranted: a controlled, deconfounded sweep across n_lab~15-120 (two
    independent scenario families, 2500 reps/point) found no reliable
    n_lab regime where ``power_tune=True`` becomes calibration-superior to
    ``power_tune=False`` -- the one apparent crossover point did not
    replicate at the same n_lab in the second family, consistent with
    Monte Carlo noise rather than a real effect -- so switching would
    effectively mean "always use ``power_tune=False``," forfeiting
    ``power_tune``'s substantial documented power advantage for no
    reliable calibration gain. See simulations/out/
    results_why_ppi_shrink_1_over_0.md's Addendum 30 for the full
    investigation (four candidate fixes tried, one adopted).
    """
    n_subjects = len(groups[0])
    labels_mat = np.column_stack(groups_lab)
    overlap = np.all(~np.isnan(labels_mat), axis=1)
    n_lab = int(overlap.sum())

    if n_lab == 0:
        return None

    llm_mat = _scipy_stats.rankdata(np.column_stack(groups), axis=1, method="average")  # (n_subjects, k)

    cond_means_llm = llm_mat.mean(axis=0)  # (k,)

    # Per-condition PPI corrections estimated from labeled subjects' own ranks
    llm_lab = llm_mat[overlap]
    human_lab = _scipy_stats.rankdata(labels_mat[overlap], axis=1, method="average")
    delta = human_lab.mean(axis=0) - llm_lab.mean(axis=0)

    unlab_idx = np.where(~overlap)[0]
    n_unlab = len(unlab_idx)

    # Same n_unlab<2 fallback as _ppi_anova_repeated_f_stat -- see its
    # comment for the rationale (includes the fully-labeled n_unlab==0
    # case).
    if power_tune and n_unlab >= 2 and n_lab >= 2:
        llm_unlab = llm_mat[unlab_idx]
        f_unlab = llm_unlab.mean(axis=0)
        f_lab_llm = llm_lab.mean(axis=0)
        f_lab_human = human_lab.mean(axis=0)
        r_term = f_unlab - f_lab_llm

        P = np.eye(k) - np.ones((k, k), dtype=float) / float(k)
        lam_raw, var_unlab, var_lab_llm, cov_cross = _repeated_anova_lambda_raw(
            human_lab, llm_lab, llm_unlab, P, k,
        )
        raw_var_human = np.var(human_lab, axis=0, ddof=1).sum()
        raw_var_llm_lab = np.var(llm_lab, axis=0, ddof=1).sum()
        if n_lab <= 1 or raw_var_human < raw_var_llm_lab * 1e-6:
            lam_replicates = None
        else:
            lam_replicates = _repeated_anova_lambda_replicates(human_lab, llm_lab, llm_unlab, P, k, n_lab)
        from evalstats.ppi import _adaptive_shrink_lambda, _shrunk_lambda_variance, _POWER_TUNE_SHRINKAGE_C
        lam = _adaptive_shrink_lambda(lam_raw, lam_replicates, n_lab)

        cond_means_ppi = f_lab_human + lam * r_term
        grand_ppi = cond_means_ppi.mean()
        ss_condition_corr = float(n_subjects * np.sum((cond_means_ppi - grand_ppi) ** 2))

        var_human = np.atleast_2d(np.cov(human_lab, rowvar=False)) / n_lab
        var_matrix = var_human + lam * lam * (var_unlab + var_lab_llm) - 2.0 * lam * cov_cross
        if lam_replicates is not None and len(lam_replicates) > 1:
            # Closed-form Var(shrunk lambda) -- see
            # evalstats.ppi._shrunk_lambda_variance's docstring and
            # simulations/out/results_why_ppi_shrink_1_over_0.md's
            # friedman power_tune=True addendum.
            var_lam_raw = float(np.var(lam_replicates, ddof=1))
            w = n_lab / (n_lab + _POWER_TUNE_SHRINKAGE_C)
            var_lam_shrunk = _shrunk_lambda_variance(lam_raw, var_lam_raw, w)
            var_matrix = var_matrix + np.outer(r_term, r_term) * var_lam_shrunk
        var_matrix = (var_matrix + var_matrix.T) / 2.0

        denom = float(n_subjects * np.trace(P @ var_matrix @ P) / (k - 1))
        denom = max(denom, 1e-8)
        f_corr = (ss_condition_corr / (k - 1)) / denom
        df_residual_eff = max((n_lab - 1) * (k - 1), 1)
        return {
            "f_corr": f_corr, "dfn": k - 1, "dfd": df_residual_eff,
            "denom": denom, "scale": float(n_subjects * k),
        }

    cond_means_ppi = cond_means_llm + delta
    grand_ppi = cond_means_ppi.mean()
    ss_condition_corr = float(n_subjects * np.sum((cond_means_ppi - grand_ppi) ** 2))

    # Known (tie-adjusted) null variance scale for within-subject ranks: see
    # docstring. Estimated empirically (averaged per-row, tie-aware) from the
    # full LLM rank matrix rather than via SS-decomposition residual -- but
    # only as one half of a shrinkage blend with the labeled human data's own
    # row sum-of-squares (see below); using the LLM matrix alone borrows a
    # tie structure that assumes it resembles the human side's, which fails
    # whenever the labeled data is exactly tied (every genuine
    # repeated-measures Type-I null reveals the same shared ground truth for
    # all k conditions, so every labeled subject's row is a perfect k-way
    # tie, giving ms_null_human ~ 0, not the LLM-borrowed constant). Since
    # that borrowed constant does not shrink with n_lab while the
    # correctly-scaled noise term below does, it goes from a minor
    # contributor at small n_lab to the dominant one at large n_lab,
    # crushing f_corr toward 0.
    row_ss_llm = np.sum((llm_mat - (k + 1) / 2.0) ** 2, axis=1)
    ms_null_llm = max(float(np.mean(row_ss_llm)) / (k - 1), 1e-12)
    row_ss_human = np.sum((human_lab - (k + 1) / 2.0) ** 2, axis=1)
    ms_null_human = max(float(np.mean(row_ss_human)) / (k - 1), 1e-12)
    # Shrink toward the human-based (asymptotically correct for this
    # estimand) estimate as n_lab grows, the same _POWER_TUNE_SHRINKAGE_C
    # -style blend evalstats.ppi.correct's power-tuning uses for an
    # analogous small-n_lab-noisy-estimate problem: ms_null_human alone
    # fixes the large-label_frac collapse but leaves a mild, roughly uniform
    # inflation at small label_frac (it's itself noisy there, being
    # estimated from only n_lab rows); blending closes both gaps.
    from evalstats.ppi import _POWER_TUNE_SHRINKAGE_C
    w = n_lab / (n_lab + _POWER_TUNE_SHRINKAGE_C)
    ms_null = (1.0 - w) * ms_null_llm + w * ms_null_human

    # Estimate effective LLM noise variance in (rank) contrast space from the
    # labeled residual covariance, same as the repeated-ANOVA case.
    noise_lab = llm_lab - human_lab  # (n_lab, k)
    if n_lab > 1:
        cov_eta = np.cov(noise_lab, rowvar=False, ddof=1)
        cov_eta = np.atleast_2d(cov_eta)
        P = np.eye(k) - np.ones((k, k), dtype=float) / float(k)
        sigma_llm_sq = float(np.trace(P @ cov_eta @ P) / float(k - 1))
        sigma_llm_sq = max(sigma_llm_sq, 0.0)
    else:
        sigma_llm_sq = 0.0

    # n_subjects × Var(cond_means_ppi) = Σ_H + σ_η² × (n_subjects/n_lab − 1)
    # (see docstring point 2: Σ_H = ms_null, used directly with no
    # subtraction of σ_η²).
    denom = ms_null + sigma_llm_sq * (n_subjects / n_lab - 1.0)
    denom = max(denom, 1e-6)

    f_corr = (ss_condition_corr / (k - 1)) / denom

    # ms_null carries (almost) no estimation uncertainty of its own (it's a
    # known combinatorial constant, tie-adjusted from the full sample), so
    # the reference distribution's denominator df is driven entirely by how
    # well sigma_llm_sq is estimated from the n_lab labeled subjects.
    df_residual_eff = max((n_lab - 1) * (k - 1), 1)
    # theta (_friedman_rank_variance's own scale, mean over k of squared
    # condition-mean-RANK deviations) relates to lambda the same way
    # repeated ANOVA's does: E[ss_condition_corr] = (k-1)*denom +
    # n_subjects*k*theta (since theta = Σ(true effect)²/k here too) ->
    # theta = lambda*denom/(n_subjects*k).
    return {"f_corr": f_corr, "dfn": k - 1, "dfd": df_residual_eff, "denom": denom, "scale": float(n_subjects * k)}


def _ppi_friedman_p_value(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    k: int,
    power_tune: bool = True,
) -> Optional[float]:
    """Corrected p-value for the Friedman test via per-condition PPI
    corrections applied to within-subject ranks -- see
    :func:`_ppi_friedman_f_stat` for the full derivation and
    :func:`_ppi_friedman_ci` for the CI derived from this same F-statistic
    (guaranteed consistent with this p-value by construction)."""
    stat = _ppi_friedman_f_stat(groups, groups_lab, k, power_tune=power_tune)
    if stat is None:
        return None
    if stat["f_corr"] <= 0.0:
        return 1.0
    return float(_scipy_stats.f.sf(stat["f_corr"], dfn=stat["dfn"], dfd=stat["dfd"]))


def _ppi_friedman_ci(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], k: int, alpha: float,
    power_tune: bool = True,
) -> Optional[tuple[float, float, float]]:
    """(estimate, ci_low, ci_high) for Friedman's within-subject-rank
    condition variance, via test-inversion on the SAME F-statistic
    :func:`_ppi_friedman_p_value` uses -- see :func:`_noncentral_f_ci_lambda`."""
    stat = _ppi_friedman_f_stat(groups, groups_lab, k, power_tune=power_tune)
    if stat is None:
        return None
    f_corr, dfn, dfd, denom, scale = stat["f_corr"], stat["dfn"], stat["dfd"], stat["denom"], stat["scale"]
    if f_corr <= 0.0 or not np.isfinite(f_corr):
        return 0.0, 0.0, 0.0
    # Debiased the same way _ppi_anova_independent_ci's estimate is -- see
    # that function's comment for the full derivation/root-cause (this had
    # the identical max(.,0) floor, and the identical fix: report the
    # unclamped, possibly-negative debiased estimate directly, the same
    # convention omega-squared/epsilon-squared use). f_corr/dfn/dfd (the
    # p-value's inputs) and the CI bounds below are untouched by this.
    estimate = (f_corr - 1.0) * dfn * denom / scale
    lam_L, lam_U = _noncentral_f_ci_lambda(f_corr, dfn, dfd, alpha)
    return estimate, lam_L * denom / scale, lam_U * denom / scale


# ─────────────────────────────────────────────────────────────────────────────
# ttest
# ─────────────────────────────────────────────────────────────────────────────

def ttest(
    a,
    b,
    a_lab=None,
    b_lab=None,
    *,
    paired: bool = False,
    equal_var: bool = False,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng=None,
    print_result: bool = True,
    power_tune: bool = True,
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
    power_tune : bool
        Use PPI++'s variance-minimizing power-tuning weight λ instead of
        the original (2023) PPI estimator's fixed λ=1 (default True; see
        :func:`evalstats.ppi.correct`'s ``power_tune`` parameter for the
        full derivation, including validation status). λ never makes the
        PPI-corrected estimator less efficient than the classical
        (``a_lab``/``b_lab``-only) one; fixing λ=1 has no such guarantee.
        Pass ``power_tune=False`` to reproduce the original PPI estimator
        exactly. The value actually used is on the returned
        ``TestResult.lam``.

    Examples
    --------
    >>> result = es.tests.ttest(llm_a, llm_b)                          # prints
    >>> result = es.tests.ttest(llm_a, llm_b, print_result=False)      # silent
    >>> result = es.tests.ttest(llm_a, llm_b, human_a, human_b)     # positional
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

    corrected_estimate = corrected_ci = corrected_p = rectifier = lam = None
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
            ppi = _ppi_paired_arrays(a, b, a_lab, b_lab, np.mean, alpha, n_boot, rng, power_tune=power_tune)
        else:
            ar = _run_alignment_report(
                np.concatenate([a, b]),
                np.concatenate([a_lab, b_lab]),
            )
            ppi = _ppi_two_sample_t_interval(a, b, a_lab, b_lab, alpha, power_tune=power_tune)

        corrected_estimate = ppi.estimate
        corrected_ci       = (ppi.ci_low, ppi.ci_high)
        corrected_p        = ppi.p_value
        rectifier          = ppi.rectifier
        lam                = ppi.lam
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
        lam=lam,
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
    x_lab=None,
    y_lab=None,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng=None,
    print_result: bool = True,
    power_tune: bool = True,
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
    alpha : float
        Two-sided significance level for the PPI confidence interval
        (default 0.05).
    n_boot : int
        Bootstrap replicates for the PPI interval (default 2000).
    rng : int | np.random.Generator | None
        Seed or generator for the bootstrap.
    print_result : bool
        Print a summary table to stdout (default True).  Pass ``False`` to
        suppress output when calling from automated pipelines.
    power_tune : bool
        Use PPI++'s variance-minimizing power-tuning weight λ instead of
        fixed λ=1 (default True -- see :func:`evalstats.ppi.correct`'s
        ``power_tune`` parameter). Pass ``power_tune=False`` to reproduce
        the original PPI estimator exactly. The value actually used is on
        the returned ``TestResult.lam``.

    Examples
    --------
    >>> result = es.tests.mannwhitney(llm_x, llm_y)                    # prints
    >>> result = es.tests.mannwhitney(llm_x, llm_y, print_result=False) # silent
    >>> result = es.tests.mannwhitney(llm_x, llm_y, human_x, human_y)  # positional
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
        "ppi_method": "global",
    }

    corrected_estimate = corrected_ci = corrected_p = rectifier = lam = None
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

        ppi = _ppi_two_sample(x, y, x_lab, y_lab, _auc_shifted, alpha, n_boot, rng, power_tune=power_tune)

        corrected_estimate = ppi.estimate + 0.5       # report as P(X>Y)
        corrected_ci       = (ppi.ci_low + 0.5, ppi.ci_high + 0.5)
        corrected_p        = ppi.p_value
        rectifier          = ppi.rectifier
        lam                = getattr(ppi, "lam", None)
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
        lam=lam,
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
    x_lab=None,
    y_lab=None,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng=None,
    print_result: bool = True,
    power_tune: bool = True,
) -> TestResult:
    """Wilcoxon signed-rank test with optional PPI correction.

    Uncorrected: ``scipy.stats.wilcoxon(x, y)`` (two-sided, paired by position).

    PPI estimand:
    ``theta = P_mid(Walsh_ij > 0) − 0.5``, where ``Walsh_ij = (d_i + d_j) / 2``
    for every pair ``i <= j`` of paired LLM differences ``d = x − y``
    (including self-pairs) -- the exact sign-statistic construction behind
    the Hodges-Lehmann one-sample location estimator, computed by
    :func:`evalstats.ppi.paired_walsh_midrank_theta` and used as BOTH the
    main statistic and the rectifier (matched). ``theta`` is 0 under H₀
    (``D`` symmetric about 0) and asymptotically equivalent to the
    classical Wilcoxon signed-rank statistic, but bounded to ``[-0.5, 0.5]``
    -- unlike ``median(x − y)``, this is NOT a location shift on the
    original score scale. Unlike a plain per-item median or sign
    proportion, this estimand doesn't degenerate under heavy ties (e.g.
    Likert scores), and using it as both the statistic and the rectifier
    keeps them matched. H₀ is theta = 0, so the bootstrap p-value
    ``2 * min(P(θ̂* ≤ 0), P(θ̂* ≥ 0))`` is correctly centered at the null.
    This is what ``simulations/harness/cases/pvalues.py``/``ppi_real.py``
    validate as "wilcoxon" -- this function is kept one-to-one with that.

    Known limitation: on real, extremely tied, small-n_lab data (e.g. 88%
    exact ties, n_lab~15), Type-I error can still run well above nominal
    -- see :func:`evalstats.ppi.paired_walsh_midrank_theta`'s docstring.
    Not an issue at n_lab >= 30, where PPI correction for this estimand
    uses a closed-form analytic backend
    (:func:`evalstats.ppi._analytic_walsh_theta_correct`) that also gives
    better power than the percentile bootstrap across the full n_lab
    range with no calibration cost.

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
    power_tune : bool
        Use PPI++'s variance-minimizing power-tuning weight λ instead of
        fixed λ=1 (default True -- see :func:`evalstats.ppi.correct`'s
        ``power_tune`` parameter for validation status). Pass
        ``power_tune=False`` to reproduce the original PPI estimator
        exactly. The value actually used is on the returned
        ``TestResult.lam``.

    Examples
    --------
    >>> result = es.tests.wilcoxon(llm_x, llm_y)                      # prints
    >>> result = es.tests.wilcoxon(llm_x, llm_y, print_result=False)  # silent
    >>> result = es.tests.wilcoxon(llm_x, llm_y, human_x, human_y)    # positional
    >>> result = es.tests.wilcoxon(llm_x, llm_y, x_lab=human_x, y_lab=human_y)
    """
    x = _coerce(x)
    y = _coerce(y)

    if len(x) != len(y):
        raise ValueError(
            f"Wilcoxon requires equal-length arrays (paired by position); "
            f"got {len(x)} vs {len(y)}."
        )

    diffs = x - y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Deliberately left at scipy's default method="auto": "auto" does
        # genuinely different (not just slower) work for small tied/discrete
        # samples, running exhaustive permutation enumeration for a
        # rigorously tie-corrected exact p-value rather than an
        # approximation. That's legitimately expensive at small n, but it's
        # the correct p-value, and this function's result is user-facing
        # (es.tests.wilcoxon / PairedDiffResult.wilcoxon_p), so correctness
        # takes priority over speed here (see
        # simulations/harness/cases/pvalues.py's _safe_wilcoxon_p docstring
        # for the full performance characterization).
        res = _scipy_stats.wilcoxon(x, y, alternative="two-sided")
    w_stat, p_val = float(res.statistic), float(res.pvalue)
    median_diff = float(np.median(diffs[diffs != 0])) if np.any(diffs != 0) else 0.0
    extra = {
        "_test": "wilcoxon",
        "n_pairs": len(x),
        "median_diff": median_diff,
        "n_boot": n_boot,
        "ppi_method": "current",
    }

    corrected_estimate = corrected_ci = corrected_p = rectifier = lam = None
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

        # paired_walsh_midrank_theta (evalstats.ppi), matched as BOTH
        # statistic and rectifier -- see this function's docstring for
        # why this replaced the earlier mean-rectifier (biased) and
        # matched-median (power-collapses under ties) attempts, and why
        # it's kept one-to-one with what the simulation harness
        # validates as "wilcoxon".
        ppi = _ppi_paired_arrays(x, y, x_lab, y_lab, paired_walsh_midrank_theta, alpha, n_boot, rng,
                                 rectifier_func=paired_walsh_midrank_theta, power_tune=power_tune)

        corrected_estimate = ppi.estimate
        corrected_ci       = (ppi.ci_low, ppi.ci_high)
        corrected_p        = ppi.p_value
        rectifier          = ppi.rectifier
        lam                = ppi.lam
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
        lam=lam,
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

        # corrected_estimate/corrected_ci/corrected_p all come from the SAME
        # closed-form F-statistic pipeline (_ppi_anova_repeated_ci/
        # _ppi_anova_independent_ci compute the p-value's own noncentral-F
        # test-inversion CI, not a separately-bootstrapped one) -- see
        # _noncentral_f_ci_lambda's docstring for why the two used to be
        # able to disagree outright before this fix.
        if repeated:
            ci_result = _ppi_anova_repeated_ci(groups, groups_lab, k, alpha)
            corrected_p = _ppi_anova_repeated_p_value(groups, groups_lab, k)
        else:
            ci_result = _ppi_anova_independent_ci(groups, groups_lab, k, alpha)
            corrected_p = _ppi_anova_independent_p_value(groups, groups_lab, k)

        if ci_result is not None:
            corrected_estimate, ci_low, ci_high = ci_result
            corrected_ci = (ci_low, ci_high)
            # Disjoint unlabeled-only complement, matching evalstats.ppi.
            # correct's convention (Y_hat_unlab excludes labeled positions)
            # that every other "llm-only" baseline in this module uses.
            if repeated:
                labels_mat = np.column_stack(groups_lab)
                overlap = np.all(~np.isnan(labels_mat), axis=1)
                llm_unlab_matrix = np.column_stack(groups)[~overlap]
                uncorrected_variance = _repeated_condition_variance(llm_unlab_matrix)
            else:
                masks = [~np.isnan(g_lab) for g_lab in groups_lab]
                groups_unlab = [g[~m] for g, m in zip(groups, masks)]
                uncorrected_variance = _anova_between_variance_from_groups(groups_unlab)
            rectifier = corrected_estimate - uncorrected_variance
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


# ─────────────────────────────────────────────────────────────────────────────
# friedman
# ─────────────────────────────────────────────────────────────────────────────

def friedman(
    *groups,
    groups_lab=None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng=None,
    print_result: bool = True,
) -> TestResult:
    """Friedman test (repeated-measures, rank-based one-way) with optional PPI.

    Uncorrected: ``scipy.stats.friedmanchisquare(*groups)``.

    Friedman is the rank-based analog of repeated-measures one-way ANOVA —
    each subject's scores across the *k* conditions are ranked (average
    ranks for ties), and the test asks whether the mean rank differs across
    conditions. It applies only to within-subject (repeated) designs; for
    the independent-groups rank analog, use a Kruskal-Wallis test instead.

    PPI estimand: variance of the per-condition mean within-subject ranks
    (the Friedman analog of ``anova_oneway(..., repeated=True)``'s
    between-condition variance after row-centering). Because ranking needs a
    subject's full row, a subject is included in the labeled set only when
    it has human labels in *all* conditions — the same requirement as
    repeated-measures PPI ANOVA.

    The corrected p-value uses the Iman–Davenport F-on-ranks approximation
    with PPI-rectified condition mean ranks (mirroring how the repeated
    ANOVA correction works on raw condition means), since the plain
    ``2 * min(P(θ̂* ≤ 0), P(θ̂* ≥ 0))`` bootstrap trick is invalid for a
    variance-like estimand that's bounded below by zero.

    Known, open limitation under MNAR labeling: Type-I error can inflate
    well above nominal when the labeled subset is a non-representative
    (MNAR) sample of subjects and N (population size) is large -- the
    failure grows with N at fixed n_lab, unlike wilcoxon()'s own documented
    small-n_lab residual. See :func:`_ppi_friedman_f_stat`'s docstring for
    the mechanism and why a prototyped fix wasn't shipped.

    Parameters
    ----------
    *groups : array-like
        Three or more score arrays, one per condition, equal length
        (paired by array position — the same subject contributes one score
        per condition).
    groups_lab : sequence[array-like], optional
        Sparse human labels aligned with each group/condition. Must have one
        array per group, same lengths, with NaN for unlabeled items.
    print_result : bool
        Print a summary table to stdout (default True).  Pass ``False`` to
        suppress output when calling from automated pipelines.

    Examples
    --------
    >>> result = es.tests.friedman(g1, g2, g3)                             # prints
    >>> result = es.tests.friedman(g1, g2, g3, print_result=False)         # silent
    >>> result = es.tests.friedman(g1, g2, g3, groups_lab=[l1, l2, l3])    # PPI
    """
    if len(groups) < 3:
        raise ValueError("friedman requires at least three conditions (k >= 3).")

    groups = [_coerce(g) for g in groups]
    k = len(groups)
    n_subjects = len(groups[0])
    for i, g in enumerate(groups[1:], start=2):
        if len(g) != n_subjects:
            raise ValueError(
                "friedman requires equal-length arrays across all "
                f"conditions; condition 1 has {n_subjects}, condition {i} has {len(g)}."
            )

    res = _scipy_stats.friedmanchisquare(*groups)
    chi2_stat, p_val = float(res.statistic), float(res.pvalue)
    df = float(k - 1)

    # Kendall's W = χ²/(n(k-1)), computed from the same rank-variance
    # estimand used for the PPI correction so the uncorrected and
    # PPI-corrected effect sizes are directly comparable (see
    # _friedman_rank_variance docstring for the derivation
    # χ² = 12·n·V/(k+1) ⟹ W = 12·V/(k²-1)).
    rank_var = _friedman_rank_variance(np.column_stack(groups))
    kendalls_w = float(12.0 * rank_var / (k**2 - 1)) if k > 1 else 0.0

    extra = {
        "_test": "friedman",
        "k_groups": k,
        "n_subjects": n_subjects,
        "group_means": [float(np.mean(g)) for g in groups],
        "df": df,
        "effect_size": kendalls_w,
        "effect_size_name": "Kendall’s W",
        "n_boot": n_boot,
    }
    test_name = "Friedman test"

    corrected_estimate = corrected_ci = corrected_p = rectifier = None
    n_labeled = n_total = None

    if groups_lab is not None:
        groups_lab = _sanitize_multigroup_ppi_labels(
            groups,
            groups_lab,
            repeated=True,
            test_label="Friedman",
        )

        llm_all = np.concatenate(groups)
        human_sparse = np.concatenate(groups_lab)
        ar = _run_alignment_report(llm_all, human_sparse)

        # corrected_estimate/corrected_ci/corrected_p all come from the SAME
        # closed-form F-statistic pipeline -- see anova_oneway's matching
        # comment and _noncentral_f_ci_lambda's docstring.
        ci_result = _ppi_friedman_ci(groups, groups_lab, k, alpha)
        corrected_p = _ppi_friedman_p_value(groups, groups_lab, k)

        if ci_result is not None:
            corrected_estimate, ci_low, ci_high = ci_result
            corrected_ci = (ci_low, ci_high)
            # Disjoint unlabeled-only complement -- see anova_oneway's
            # matching comment for why.
            labels_mat = np.column_stack(groups_lab)
            overlap = np.all(~np.isnan(labels_mat), axis=1)
            llm_unlab_matrix = np.column_stack(groups)[~overlap]
            rectifier = corrected_estimate - _friedman_rank_variance(llm_unlab_matrix)
        n_labeled = ar.n_labeled
        n_total = ar.n_total

        # PPI-corrected Kendall's W: same rescaling, with the PPI-corrected
        # rank-variance estimate as the numerator instead of the raw one.
        extra["corrected_effect_size"] = (
            float(12.0 * corrected_estimate / (k**2 - 1)) if k > 1 else 0.0
        )
        extra["corrected_effect_size_name"] = "Kendall’s W"

    result = TestResult(
        test_name=test_name,
        statistic=chi2_stat,
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
# kruskalwallis
# ─────────────────────────────────────────────────────────────────────────────

def kruskalwallis(
    *groups,
    groups_lab=None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng=None,
    print_result: bool = True,
    method: str = "global",
) -> TestResult:
    """Kruskal-Wallis test (independent-groups, rank-based one-way) with
    optional PPI correction.

    Uncorrected: ``scipy.stats.kruskal(*groups)``.

    Kruskal-Wallis ranks ALL observations together in one pooled sample —
    unlike Friedman's within-subject ranks, this pooled rank does NOT
    survive subsetting to just the labeled items (a given item's global
    rank depends on every other item in the dataset). So instead of
    correcting pooled ranks directly, the PPI estimand here is built from
    pairwise stochastic-dominance probabilities
    ``θ_ab = P_mid(group_a > group_b)`` for every pair of groups — the same
    mid-rank U-statistic ``mannwhitney`` already uses for two groups, which
    *is* well-defined identically on the full data or any subsample. For
    k=2 groups Kruskal-Wallis reduces to Mann-Whitney; this pairwise-θ
    framework is the direct generalization to k>2.

    PPI correction is applied to each pairwise θ_ab via the standard
    rectifier formula, and the omnibus null H₀: "every θ_ab = 0.5" (no
    stochastic dominance between any pair of groups) is tested via a Wald
    test using the joint bootstrap covariance of the corrected θ̂ vector
    (Moore-Penrose pseudo-inverse, since the C(k,2) pairwise contrasts have
    only k−1 independent degrees of freedom; chi-square reference with
    df = k−1). This deliberately avoids the closed-form null-variance
    derivation that ``anova_oneway``/``friedman``'s p-values need — a Wald
    test uses the *unrestricted* bootstrap covariance by design, so there's
    no "subtract the LLM-noise term from a null baseline" step to get wrong.

    The headline ``corrected_estimate``/effect size is a single scalar
    summary, ``4·mean((θ_ab − 0.5)²)`` over all pairs (bounded [0, 1], 0
    under H₀) — corrected via the ordinary PPI bootstrap CI, same pattern as
    ``anova_oneway``/``friedman``. The full per-pair breakdown (corrected
    θ̂_ab + CI for every group pair) is available in ``extra["pairwise_effects"]``.

    Parameters
    ----------
    *groups : array-like
        Three or more independent-group score arrays. For exactly two
        groups, use ``mannwhitney`` instead.
    groups_lab : sequence[array-like], optional
        Sparse human labels aligned with each group. Must have one array per
        group, with ``NaN`` for unlabeled items.
    method : {"global", "mnar_experimental"}
        Which PPI correction to use for the pairwise-dominance Wald test
        (default ``"global"``). Same tradeoff as ``mannwhitney``'s
        equivalent option, one level up (k independent groups instead of 2).

        ``"global"`` (:func:`_ppi_kruskal_wallis_pairwise`) applies a
        single rectifier per pair -- simple, but can be miscalibrated
        under labeling that's non-uniform with respect to score combined
        with real judge bias and coarse/discrete scales.

        ``"mnar_experimental"`` (:func:`_ppi_kruskal_wallis_pairwise_mnar_experimental`)
        fixes that with a per-group, per-score-bin local rectifier instead
        (see that function's docstring), at the cost of real calibration
        under ordinary, correctly-random (MCAR) labeling -- the same
        MCAR-vs-MNAR tradeoff ``mannwhitney``'s equivalent fix has. Since
        this package's PPI correction generally assumes labels are sampled
        uniformly at random (see :func:`evalstats.ppi.correct`'s
        docstring) and treats MNAR labeling as an out-of-scope, documented
        limitation, paying the MCAR cost for MNAR robustness is the wrong
        trade for the default. ``"mnar_experimental"`` remains available
        for anyone deliberately studying the MNAR-robustness question --
        see ``simulations/harness/cases/pvalues.py --mode ppi`` (methods
        ``kruskal`` vs ``kruskal_mnar_experimental``) for the calibration
        study. Only affects ``corrected_p`` (the Wald test) --
        ``corrected_estimate`` (the scalar effect size from
        :func:`_ppi_kruskal_wallis`) always uses the single-global-
        rectifier estimator regardless of ``method``.
    print_result : bool
        Print a summary table to stdout (default True). Pass ``False`` to
        suppress output when calling from automated pipelines.

    Examples
    --------
    >>> result = es.tests.kruskalwallis(g1, g2, g3)                              # prints
    >>> result = es.tests.kruskalwallis(g1, g2, g3, print_result=False)          # silent
    >>> result = es.tests.kruskalwallis(g1, g2, g3, groups_lab=[l1, l2, l3])     # PPI
    """
    if method not in ("global", "mnar_experimental"):
        raise ValueError(f'method must be "global" or "mnar_experimental"; got {method!r}.')
    if len(groups) < 3:
        raise ValueError(
            "kruskalwallis requires at least three groups (k >= 3); "
            "for two groups, use mannwhitney()."
        )

    groups = [_coerce(g) for g in groups]
    k = len(groups)
    n_total_obs = int(sum(len(g) for g in groups))

    res = _scipy_stats.kruskal(*groups)
    h_stat, p_val = float(res.statistic), float(res.pvalue)
    df = float(k - 1)

    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    raw_thetas = _kw_pairwise_thetas(groups, pairs)
    effect_size_val = float(4.0 * np.mean((raw_thetas - 0.5) ** 2))

    extra = {
        "_test": "kruskal",
        "k_groups": k,
        "n_total_obs": n_total_obs,
        "group_sizes": [len(g) for g in groups],
        "group_medians": [float(np.median(g)) for g in groups],
        "df": df,
        "effect_size": effect_size_val,
        "effect_size_name": "Mean (P(a>b)−0.5)²",
        "n_boot": n_boot,
        "ppi_method": method,
    }
    test_name = "Kruskal-Wallis test"

    corrected_estimate = corrected_ci = corrected_p = rectifier = None
    n_labeled = n_total = None

    if groups_lab is not None:
        groups_lab = _sanitize_multigroup_ppi_labels(
            groups,
            groups_lab,
            repeated=False,
            test_label="Kruskal-Wallis",
        )

        llm_all = np.concatenate(groups)
        human_sparse = np.concatenate(groups_lab)
        ar = _run_alignment_report(llm_all, human_sparse)

        # NOTE: these two deliberately run in DIFFERENT lambda regimes --
        # the estimate/CI at lambda=1, the p-value power-tuned. That is not
        # an oversight; see _ppi_kruskal_wallis's comment for the coverage
        # measurements that keep it that way.
        ppi = _ppi_kruskal_wallis(groups, groups_lab, alpha, n_boot, rng)
        if method == "mnar_experimental":
            pw = _ppi_kruskal_wallis_pairwise_mnar_experimental(groups, groups_lab, alpha, n_boot, rng)
        else:
            pw = _ppi_kruskal_wallis_pairwise(groups, groups_lab, alpha, n_boot, rng)

        corrected_estimate = ppi.estimate
        corrected_ci = (ppi.ci_low, ppi.ci_high)
        corrected_p = pw["wald_p"]
        rectifier = ppi.rectifier
        n_labeled = ar.n_labeled
        n_total = ar.n_total

        extra["corrected_effect_size"] = corrected_estimate
        extra["corrected_effect_size_name"] = "Mean (P(a>b)−0.5)²"
        extra["wald_stat"] = pw["wald_stat"]
        extra["pairwise_effects"] = {
            pw["pairs"][i]: {
                "theta": float(pw["theta_hat"][i]),
                "ci": (float(pw["ci_lo"][i]), float(pw["ci_hi"][i])),
            }
            for i in range(len(pw["pairs"]))
        }

    result = TestResult(
        test_name=test_name,
        statistic=h_stat,
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


def _sanitize_lmm_ppi_labels(groups: list[np.ndarray], groups_lab) -> list[np.ndarray]:
    """Validate sparse labels for the PPI-corrected LMM.

    Effective unit: an input with at least one labeled template cell — not
    all of them, unlike the strict full-row requirement other
    repeated-measures PPI tests in this module use. The LMM rectifier only
    needs whichever labeled cells a given input actually has.
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

    labels_mat = np.column_stack(labs)
    n_lab_inputs = int(np.sum(np.any(~np.isnan(labels_mat), axis=1)))

    if n_lab_inputs < 15:
        raise ValueError(
            "At least 15 inputs with a labeled template cell are required "
            f"for PPI-corrected LMM (found {n_lab_inputs})."
        )
    if n_lab_inputs < 30:
        warnings.warn(
            f"Only {n_lab_inputs} inputs have a labeled template cell. "
            "PPI-corrected LMM standard errors can be unstable below 30 "
            "labeled inputs; consider labeling more.",
            UserWarning,
            stacklevel=3,
        )

    return labs


def _ppi_lmm_p_value(
    groups: list[np.ndarray],
    groups_lab: list[np.ndarray],
    k: int,
    factors=None,
    *,
    precomputed_fit: Optional[tuple] = None,
) -> Optional[float]:
    """Closed-form Wald-to-F omnibus p-value for the PPI-corrected LMM template effect.

    H0: every fixed effect (vs. the reference level) is zero. Uses the
    cluster-robust sandwich covariance from
    ``evalstats.core.mixed_effects._ppi_lmm_fixed_effects`` with a
    finite-sample Wald-to-F correction (Hotelling's T²-style, df = n_lab
    labeled inputs) rather than the asymptotic chi-square reference —
    mirrors ``_ppi_kruskal_wallis_pairwise``'s correction, the other test in
    this module whose corrected covariance comes from a sandwich/bootstrap
    rather than a closed-form null-variance derivation. Standalone (not used
    by ``lmm()`` itself, which already has the fitted ``PPILMMResult`` in
    scope) so the calibration simulation can call it directly without the
    array validation / alignment-report overhead of the public API, mirroring
    ``_ppi_anova_repeated_p_value`` / ``_ppi_friedman_p_value``. ``factors``
    is passed straight through to ``_ppi_lmm_fixed_effects`` so this same
    helper covers the single-factor, multi-factor, and nested-run lmm
    calibration variants alike. ``precomputed_fit`` is likewise forwarded
    straight through -- see ``_ppi_lmm_fixed_effects``'s docstring -- for
    callers that already have the LLM-only fit (e.g. from computing an
    uncorrected Wald F-test on the same ``groups``) and want to avoid
    fitting the identical MixedLM model twice.
    """
    from evalstats.core.mixed_effects import _ppi_lmm_fixed_effects

    template_labels = [f"T{i}" for i in range(k)]
    ppi = _ppi_lmm_fixed_effects(
        groups, groups_lab, template_labels, factors=factors, precomputed_fit=precomputed_fit,
    )

    df_q = k - 1
    if df_q <= 0 or ppi.n_lab <= df_q:
        return None

    beta_t = ppi.beta_ppi[1:]
    cov_t = ppi.cov_ppi[1:, 1:]
    cov_t_pinv = np.linalg.pinv(cov_t, rcond=1e-8)
    wald = float(beta_t @ cov_t_pinv @ beta_t)
    nu = ppi.n_lab
    f_corr = wald * (nu - df_q + 1) / (nu * df_q)
    if f_corr <= 0.0:
        return 1.0
    return float(_scipy_stats.f.sf(f_corr, df_q, nu - df_q + 1))


def lmm(
    *groups,
    groups_lab=None,
    factors=None,
    alpha: float = 0.05,
    print_result: bool = True,
) -> TestResult:
    """Linear mixed model (``score ~ <fixed factors> + (1|input)``) with optional PPI.

    Uncorrected: ``statsmodels`` ``MixedLM`` (REML) — the same model
    ``lmm_analyze`` fits, here as a direct array-based wrapper. An omnibus
    Wald F-test on the fixed effects (excluding the intercept) answers
    "does template/factor identity affect score at all," analogous to
    ``anova_oneway(repeated=True)`` but accounting for between-input
    variance via an explicit random intercept rather than row-centering.

    By default each ``groups[t]`` is one "template" (one fixed-factor
    level). Two extensions, composable with each other:

    - **Multiple crossed fixed factors**: pass ``factors``, a DataFrame with
      one row per group (same order as ``*groups``) and one column per fixed
      factor (e.g. columns ``["model", "prompt"]``). Builds
      ``score ~ C(model) * C(prompt) + (1|input)`` instead of
      ``score ~ C(template) + (1|input)``, with no other change to the
      correction — the rectifier, sandwich, and cross-covariance term are
      all generic in the design matrix already.
    - **Nested run replication**: pass each ``groups[t]`` as ``(n_inputs, R)``
      instead of ``(n_inputs,)`` for R repeated LLM samples per
      (template, input) cell with no shared identity across inputs (e.g.
      resampling at temperature > 0) — *not* a crossed random effect with
      its own variance component. ``groups_lab[t]`` stays ``(n_inputs,)``
      regardless of R: one human label per item, compared against the
      LLM's run-average for that item. Internally fit on the run-averaged
      data directly rather than the raw per-run rows, since a cell's R runs
      generally share a fixed residual component (real item-level
      heterogeneity) on top of independent per-run LLM noise — see
      ``evalstats.core.mixed_effects._fit_lmm_general``.

    PPI correction here is structurally different from every other test in
    this module. LMM fixed effects are estimated by (restricted) maximum
    likelihood — a nonlinear M-estimator — so the "plug-in" recipe used by
    ``ttest``/``friedman``/etc. (independently refit on full-LLM,
    labeled-human, and labeled-LLM data, then add the three results) is not
    a valid bias correction here, and re-estimating variance components on
    a small labeled subset is often unstable. Instead the correction solves
    the *combined* rectified estimating equation in one closed-form GLS
    step (see ``evalstats.core.mixed_effects._ppi_lmm_fixed_effects`` for
    the full derivation):

      1. σ²_input/σ²_resid are estimated once from the full LLM-scored fit
         and held fixed as nuisance parameters — only the fixed effects β
         are PPI-corrected.
      2. ``β̂_PPI = β̂_unlab + (n_inputs/n_lab) · M⁻¹ r``, where ``M`` is the
         full fit's GLS information matrix and ``r`` is the GLS rectifier
         summed over labeled inputs.
      3. Standard errors come from a cluster-robust sandwich over labeled
         *inputs* (the natural unit of exchangeability, mirroring the
         per-subject overlap requirement other repeated-measures PPI tests
         use) rather than a bootstrap — refitting variance components on a
         small labeled subset at every bootstrap draw would be unstable.

    The headline ``corrected_estimate`` is the variance of the
    PPI-corrected template means (the LMM analog of
    ``anova_oneway(repeated=True)``/``friedman``'s between-condition
    variance estimand), tested via a finite-sample Wald-to-F omnibus test
    (``corrected_p_value``). Because that headline is a quadratic form in
    β (variance-like, bounded below by zero), ``corrected_ci`` is left
    ``None`` — a symmetric Wald CI on a quadratic form isn't valid without
    resampling, which this closed-form approach deliberately avoids. Valid
    per-contrast Wald CIs (each template vs. the first group, which serves
    as the reference level) are reported in ``extra["fixed_effects"]``.

    A labeled input needs at least one (not necessarily all) labeled
    template cell.

    ``extra["fixed_effects"]`` is keyed by the actual fitted parameter name
    (e.g. ``"C(template)[T.T1]"``, or ``"C(model)[T.b]:C(prompt)[T.v2]"`` for
    an interaction term with ``factors``) rather than a group index, so it
    generalizes to any number of fixed factors without change.

    Parameters
    ----------
    *groups : array-like
        Two or more score arrays, one per template/group, paired by array
        position (the same input contributes one score per group). Each
        array is ``(n_inputs,)``, or ``(n_inputs, R)`` for R nested runs.
    groups_lab : sequence[array-like], optional
        Sparse human labels aligned with each group, always ``(n_inputs,)``
        regardless of R. NaN for unlabeled items.
    factors : pd.DataFrame, optional
        One row per group (same order as ``*groups``), one column per fixed
        factor. Defaults to a single implicit ``"template"`` factor.
    print_result : bool
        Print a summary table to stdout (default True).

    Examples
    --------
    >>> result = es.tests.lmm(g1, g2, g3)
    >>> result = es.tests.lmm(g1, g2, g3, groups_lab=[l1, l2, l3])
    >>> # Two crossed fixed factors (model x prompt), 4 groups = 2x2 combinations
    >>> factors = pd.DataFrame({"model": ["a", "a", "b", "b"], "prompt": ["v1", "v2", "v1", "v2"]})
    >>> result = es.tests.lmm(g_av1, g_av2, g_bv1, g_bv2, groups_lab=[...], factors=factors)
    >>> # Nested runs: 3 repeated LLM samples per (template, input) cell
    >>> result = es.tests.lmm(g1_runs, g2_runs, groups_lab=[l1, l2])  # g*_runs shape (n_inputs, 3)
    """
    from evalstats.core.mixed_effects import (
        _extract_variance_components_sm,
        _fit_lmm_general,
        _get_fe_vcov_sm,
        _compute_icc,
        _ppi_lmm_fixed_effects,
    )

    if len(groups) < 2:
        raise ValueError("lmm requires at least two templates/conditions.")

    groups = [_coerce(g) for g in groups]
    k = len(groups)
    n_inputs = groups[0].shape[0]
    for i, g in enumerate(groups[1:], start=2):
        if g.shape[0] != n_inputs:
            raise ValueError(
                "lmm requires equal-length arrays across all conditions "
                f"(same inputs); condition 1 has {n_inputs}, condition {i} has {g.shape[0]}."
            )
    if factors is not None and len(factors) != k:
        raise ValueError(
            f"factors must have one row per group (got {len(factors)} rows for {k} groups)."
        )

    template_labels = [f"T{i}" for i in range(k)]
    sm_result, df_full, X_row, R = _fit_lmm_general(groups, template_labels, factors)

    beta = sm_result.fe_params.to_numpy()
    cov = _get_fe_vcov_sm(sm_result)
    sigma_input, sigma_resid = _extract_variance_components_sm(sm_result)
    icc = _compute_icc(sigma_input, sigma_resid)
    param_names = sm_result.fe_params.index.tolist()

    # X_row[i] @ beta == predicted mean for group i (the design row used to
    # fit it), for any formula/factor combination -- generalizes the old
    # hand-rolled treatment-coding contrast matrix.
    mu_unlab = X_row @ beta

    df1 = k - 1
    df2 = float(sm_result.df_resid)
    var_means_unlab = float(np.var(mu_unlab))
    if df1 > 0:
        beta_t, cov_t = beta[1:], cov[1:, 1:]
        wald = float(beta_t @ np.linalg.solve(cov_t, beta_t))
        f_stat = wald / df1
        p_val = float(_scipy_stats.f.sf(f_stat, df1, df2)) if f_stat > 0 else 1.0
    else:
        f_stat, p_val = 0.0, 1.0

    extra = {
        "_test": "lmm",
        "k_groups": k,
        "n_inputs": n_inputs,
        "group_means": [float(np.mean(g)) for g in groups],
        "icc": icc,
        "sigma_input": sigma_input,
        "sigma_resid": sigma_resid,
        "df1": df1,
        "df2": df2,
        "effect_size": var_means_unlab,
        "effect_size_name": "Var(template means)",
        "fixed_effects": {
            name: {"estimate": float(beta[i]), "se": float(np.sqrt(max(cov[i, i], 0.0)))}
            for i, name in enumerate(param_names) if i > 0
        },
    }
    factor_desc = "template" if factors is None else " * ".join(factors.columns)
    run_desc = f", {R} nested runs/cell" if R > 1 else ""
    test_name = f"Linear mixed model (score ~ {factor_desc} + (1|input){run_desc})"

    corrected_estimate = corrected_ci = corrected_p = rectifier = None
    n_labeled = n_total = None

    if groups_lab is not None:
        groups_lab = _sanitize_lmm_ppi_labels(groups, groups_lab)

        llm_all = np.concatenate([g if g.ndim == 1 else g.mean(axis=1) for g in groups])
        human_sparse = np.concatenate(groups_lab)
        ar = _run_alignment_report(llm_all, human_sparse)

        ppi = _ppi_lmm_fixed_effects(groups, groups_lab, template_labels, factors=factors)
        n_lab, n_in = ppi.n_lab, ppi.n_inputs

        mu_ppi = X_row @ ppi.beta_ppi
        corrected_estimate = float(np.var(mu_ppi))
        rectifier = corrected_estimate - var_means_unlab

        df_q = k - 1
        if df_q > 0 and n_lab > df_q:
            beta_t_ppi = ppi.beta_ppi[1:]
            cov_t_ppi = ppi.cov_ppi[1:, 1:]
            cov_t_pinv = np.linalg.pinv(cov_t_ppi, rcond=1e-8)
            wald_corr = float(beta_t_ppi @ cov_t_pinv @ beta_t_ppi)
            nu = n_lab
            f_corr = wald_corr * (nu - df_q + 1) / (nu * df_q)
            corrected_p = float(_scipy_stats.f.sf(f_corr, df_q, nu - df_q + 1)) if f_corr > 0 else 1.0
            df_resid_eff = nu - df_q + 1
        else:
            wald_corr = None
            corrected_p = None
            df_resid_eff = max(n_lab - 1, 1)

        t_crit = float(_scipy_stats.t.ppf(1 - alpha / 2, df=df_resid_eff))
        fixed_effects = {}
        for i, name in enumerate(ppi.param_names):
            if i == 0:
                continue
            se_i = float(np.sqrt(max(ppi.cov_ppi[i, i], 0.0)))
            fixed_effects[name] = {
                "estimate": float(ppi.beta_unlab[i]),
                "se": float(np.sqrt(max(ppi.cov_unlab[i, i], 0.0))),
                "corrected_estimate": float(ppi.beta_ppi[i]),
                "corrected_se": se_i,
                "corrected_ci": (
                    float(ppi.beta_ppi[i] - t_crit * se_i),
                    float(ppi.beta_ppi[i] + t_crit * se_i),
                ),
            }

        n_labeled = n_lab
        n_total = n_in

        extra["fixed_effects"] = fixed_effects
        extra["wald_stat"] = wald_corr
        extra["corrected_effect_size"] = corrected_estimate
        extra["corrected_effect_size_name"] = "Var(template means)"

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
