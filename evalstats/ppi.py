"""evalstats.ppi — Prediction-Powered Inference (PPI) for arbitrary estimators.

Access via ``import evalstats as es; es.ppi.correct(...)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.special import expit as _sigmoid
from scipy.stats import t as _t_dist

from .core.resampling import _LOGIT_T_BOUNDARY_EPS

_MIN_LAB_RECOMMENDED = 30
"""Below this many labeled items, the percentile bootstrap is known to
undercover -- see :func:`correct`'s ``backend`` parameter. Shared with the
PPI-alignment warning threshold in ``evalstats/api.py`` and
``evalstats/alignment.py``."""

_POWER_TUNE_SHRINKAGE_C = 20.0
"""Pseudo-count for shrinking :func:`correct`'s power-tuning weight lambda
back toward 1 (vanilla PPI) as ``n_lab`` shrinks -- see the ``power_tune``
parameter docstring."""


@dataclass
class PPIResult:
    """Result returned by :func:`correct`.

    Attributes
    ----------
    estimate : float
        PPI-corrected point estimate.
    ci_low, ci_high : float
        Lower and upper bounds of the bootstrap confidence interval.
    alpha : float
        Significance level used (e.g. 0.05 for a 95 % CI).
    llm_estimate : float
        Uncorrected LLM-only estimate ``f(Ŷ_unlab, X_unlab)``.
    human_estimate : float
        Human-only estimate on the labeled subset ``f(Y_lab, X_lab)``.
    rectifier : float
        Bias-correction term ``human_estimate − f(Ŷ_lab, X_lab)``.
        Positive values mean the LLM overestimates; negative means it underestimates.
    p_value : float or None
        Two-sided bootstrap p-value for H₀: θ = 0.
        None when *compute_pvalue=False* was passed to :func:`correct`.
    """

    estimate: float
    ci_low: float
    ci_high: float
    alpha: float
    llm_estimate: float
    human_estimate: float
    rectifier: float
    p_value: Optional[float]
    lam: Optional[float] = None
    """The power-tuning weight actually used (see :func:`correct`'s
    ``power_tune`` parameter) -- ``None`` when ``power_tune=False`` (the
    default), in which case ``lam`` is implicitly 1.0 (today's fixed "full
    rectifier" PPI estimator). When ``power_tune=True``, this is the
    bootstrap-estimated PPI++ weight, clipped to ``[0, 1]``: 0 means the
    correction contributed nothing (falls back to the classical
    labels-only estimate, ``human_estimate``); 1 means the full rectifier
    was worth applying (identical to ``power_tune=False``'s estimate)."""

    def __repr__(self) -> str:
        ci_pct = int(round((1 - self.alpha) * 100))
        p_str = f", p={self.p_value:.4f}" if self.p_value is not None else ""
        return (
            f"PPIResult(estimate={self.estimate:.4f}, "
            f"{ci_pct}%CI=[{self.ci_low:.4f}, {self.ci_high:.4f}]"
            f"{p_str})"
        )

    def summary(self) -> None:
        """Print a human-readable summary of the PPI result."""
        ci_pct = int(round((1 - self.alpha) * 100))
        w = 26
        print(f"{'PPI corrected estimate':<{w}}: {self.estimate:.4f}")
        print(f"{f'{ci_pct}% CI':<{w}}: [{self.ci_low:.4f}, {self.ci_high:.4f}]")
        if self.p_value is not None:
            print(f"{'p-value (H₀: θ=0)':<{w}}: {self.p_value:.4f}")
        print(f"{'LLM-only estimate':<{w}}: {self.llm_estimate:.4f}")
        print(f"{'Human-only estimate':<{w}}: {self.human_estimate:.4f}")
        print(f"{'Rectifier (δ)':<{w}}: {self.rectifier:+.4f}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _call(func: Callable, Y: np.ndarray, X: Optional[np.ndarray]) -> float:
    """Invoke estimator with (Y,) or (Y, X) depending on whether X is supplied."""
    if X is None:
        return float(func(Y))
    return float(func(Y, X))


_MEDIAN_TIE_JITTER_DIVISOR = 20.0
"""``correct()``'s smoothed-bootstrap jitter std is (min positive gap
between distinct values in the data) / this divisor -- small enough to only
break exact ties, large enough to stop resamples from repeatedly landing on
the identical median. See :func:`_tie_jitter_scale`."""


def _tie_jitter_scale(arr: np.ndarray) -> float:
    """Std-dev of the Gaussian jitter ``correct()`` adds before each
    bootstrap resample's median, when ``estimator_func``/``rectifier_func``
    is ``np.median``.

    Percentile-bootstrapping a median on data with substantial exact ties
    is a known-bad combination (the "smoothed bootstrap" literature, e.g.
    Efron; Hall & DiCiccio-Romano): with enough repeated values, most
    resamples' median lands on the same repeated value, collapsing the
    bootstrap distribution toward a near-constant and making the CI/p-value
    severely too conservative. Adding noise far below the data's real
    resolution before each resample's median restores a non-degenerate
    bootstrap distribution without perturbing genuine order structure.

    Returns 0.0 (no jitter) when the array has fewer than 2 distinct
    values, or when every gap between consecutive distinct values is
    effectively zero (already degenerate; jittering wouldn't help).
    """
    uniq = np.unique(arr)
    if len(uniq) < 2:
        return 0.0
    gaps = np.diff(uniq)
    positive_gaps = gaps[gaps > 1e-15]
    if positive_gaps.size == 0:
        return 0.0
    return float(np.min(positive_gaps)) / _MEDIAN_TIE_JITTER_DIVISOR


def _walsh_theta_row(d: np.ndarray) -> float:
    """Exact O(n log n) computation of the one-sample midrank-sign
    statistic ``P_mid(Walsh_ij > 0)`` for a single array of paired
    differences ``d``, where ``Walsh_ij = (d_i + d_j) / 2`` for ``i <= j``
    (``n*(n+1)/2`` pairs total, including self-pairs ``i == j``). This is
    the Hodges-Lehmann one-sample location estimator's own construction
    (its point estimate is the median of the Walsh averages; this is the
    analogous midrank-tie-corrected sign statistic of those same averages).

    Computed via a sort + vectorized ``searchsorted`` rather than the naive
    O(n^2) pairwise enumeration, which is too slow to call thousands of
    times per :func:`correct` invocation at realistic corpus sizes."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return 0.5
    ds = np.sort(d)
    neg = -ds
    idx_right = np.searchsorted(ds, neg, side="right")
    idx_left = np.searchsorted(ds, neg, side="left")
    i_arr = np.arange(n)
    hi = np.maximum(idx_right, i_arr)
    lo = np.maximum(idx_left, i_arr)
    count_gt = n - hi
    count_eq = np.maximum(0, idx_right - lo)
    total_pairs = n * (n + 1) // 2
    return float((count_gt.sum() + 0.5 * count_eq.sum()) / total_pairs)


def paired_walsh_midrank_theta(d: np.ndarray) -> float:
    """``_walsh_theta_row(d) - 0.5``, shifted to be 0 under H0 (D symmetric
    about 0). This is :func:`correct`'s default estimand for the Wilcoxon
    signed-rank family (``wilcoxon()``, via
    ``evalstats.tests._ppi_paired_arrays``).

    Counting signs of all pairwise Walsh averages ``(d_i + d_j) / 2``,
    rather than a per-item sign proportion or the paired median, avoids the
    power collapse a median-based estimand suffers under heavy ties (the
    population median of a paired difference can stay locked at exactly 0
    even under a real, classical-Wilcoxon-detectable shift). It is
    asymptotically equivalent to the Wilcoxon signed-rank statistic itself.

    Known limitation: at small ``n_lab`` (~15) on data with an extreme tie
    rate, Type-I error can still run well above nominal even with
    :func:`_analytic_walsh_theta_correct`'s closed-form backend (which
    substantially reduces, but does not eliminate, the inflation vs. the
    plain percentile bootstrap). Not an issue at ``n_lab >= 30``. See
    ``simulations/harness/cases/pvalues.py``/``ppi_real.py``'s WILCOXON
    blocks for the calibration checks behind this.

    Registered in :func:`correct`'s ``_fast_batch`` dispatch (via
    :func:`_walsh_theta_batch`) so bootstrap replicates go through the same
    vectorized-resampling fast path ``np.mean``/``np.median`` use, instead
    of the slow per-replicate Python loop arbitrary ``estimator_func``
    callers fall into -- required for this to be practical at realistic
    corpus sizes and ``n_boot``."""
    d = np.asarray(d)
    if len(d) == 0:
        return 0.0
    return _walsh_theta_row(d) - 0.5


def _walsh_theta_batch(arr: np.ndarray) -> np.ndarray:
    """Row-wise :func:`paired_walsh_midrank_theta` over a ``(m, n)``
    bootstrap-replicate array, for :func:`correct`'s ``_fast_batch``
    dispatch. Still a Python-level loop over the ``m`` replicates (no known
    way to vectorize ``_walsh_theta_row``'s per-row sort + searchsorted
    across rows without materializing a worse O(m * n^2) intermediate), but
    each row's O(n log n) call is fast enough that looping ``m`` ~2000-4000
    times is practical.

    Returns the shifted (-0.5) value, matching
    ``paired_walsh_midrank_theta`` exactly -- not the raw
    ``_walsh_theta_row`` value. ``correct()`` looks this batch function up
    by the shifted function's own ``id()``, so an unshifted return here
    would offset every bootstrap replicate by +0.5 relative to the point
    estimate, pushing the whole bootstrap distribution away from 0."""
    return np.array([_walsh_theta_row(row) - 0.5 for row in arr])


def _walsh_theta_h1_components(d: np.ndarray) -> np.ndarray:
    """Per-item empirical Hajek-projection ("structural component") values
    for :func:`paired_walsh_midrank_theta`'s underlying pairwise kernel
    ``h(d_i,d_j) = 1{d_i+d_j>0} + 0.5*1{d_i+d_j=0}`` -- the same
    "structural components" construction DeLong et al. (1988) use to get
    the variance (and, for two correlated samples, covariance) of an AUC
    estimator, applied here to the analogous one-sample Walsh-average
    U-statistic instead of DeLong's two-sample Mann-Whitney U-statistic.

    ``h1_hat(d_i) = (1/n) * sum_j [1{d_i+d_j>0} + 0.5*1{d_i+d_j=0}]``, with
    ``j`` ranging over all ``n`` items (including ``j=i``) -- the plug-in
    estimate of the first-order Hajek projection ``h1(d) = E_D2[h(d, D2)]``,
    evaluated at each observed ``d_i``. This is a "leave-in" (not
    leave-one-out) projection, which does not matter for a variance
    estimate (only the point estimate needs the sharper leave-one-out
    correction, and this function is never used for the point estimate --
    :func:`paired_walsh_midrank_theta`, via :func:`_walsh_theta_row`,
    handles that separately).

    Used by :func:`_walsh_theta_analytic_variance` (single-sample variance,
    ``Var(U) ~= (4/n)*Var(h1_hat)``, the standard degree-2 U-statistic
    asymptotic variance) and directly by
    :func:`_analytic_walsh_theta_correct` (paired covariance: for two
    structural-component arrays computed on the same n items,
    ``Var(A-B) ~= (4/n)*Var(h1_hat_A - h1_hat_B)``, folding the
    variance-of-A, variance-of-B, and covariance(A,B) terms into one pass).

    Same O(n log n) sort + vectorized ``searchsorted`` construction as
    :func:`_walsh_theta_row`, just without that function's ``i<=j``
    upper-triangular restriction (every item is compared against all n
    others here, not just itself-and-later)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return np.empty(0, dtype=float)
    ds = np.sort(d)
    neg = -d
    idx_right = np.searchsorted(ds, neg, side="right")
    idx_left = np.searchsorted(ds, neg, side="left")
    count_gt = n - idx_right
    count_eq = idx_right - idx_left
    return (count_gt + 0.5 * count_eq) / n


def _walsh_theta_analytic_variance(d: np.ndarray) -> float:
    """Analytic (no-bootstrap) variance estimate of
    ``paired_walsh_midrank_theta(d)`` for a single sample ``d``, via the
    standard degree-2 U-statistic asymptotic variance
    ``Var(U_n) ~= (m^2/n)*zeta_1`` (``m=2`` for a pairwise kernel), with
    ``zeta_1 = Var(h1(D))`` estimated by the empirical plug-in structural
    components from :func:`_walsh_theta_h1_components`. See that
    function's docstring for the numerical validation against a Monte
    Carlo oracle."""
    n = len(d)
    if n < 2:
        return 0.0
    h1 = _walsh_theta_h1_components(d)
    return 4.0 * float(np.var(h1, ddof=1)) / n


def _analytic_walsh_theta_correct(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, Y_hat_unlab: np.ndarray,
    alpha: float, power_tune: bool,
) -> "PPIResult":
    """Closed-form (no-bootstrap) PPI correction for
    ``estimator_func=paired_walsh_midrank_theta`` -- the Walsh-average
    midrank-sign analogue of :func:`_analytic_mean_correct`'s role for
    ``np.mean``. Same two-term (rectifier + unlabeled-sample) variance
    decomposition and power-tuning shrinkage as that function; only the
    per-term variance/covariance estimator differs (Hajek-projection
    structural components -- :func:`_walsh_theta_h1_components` /
    :func:`_walsh_theta_analytic_variance` -- instead of the mean's plain
    sample variance/covariance).

    Used at every ``n_lab`` under ``backend="auto"`` (see
    :data:`_ANALYTIC_ALWAYS_PREFERRED`), not just below the usual n_lab=30
    cutoff: it dominates the percentile bootstrap on power for this
    estimand across the full n_lab range, with statistically
    indistinguishable Type-I calibration, so there is no bootstrap regime
    worth falling back to.

    Degrees of freedom for the Student-t interval: ``n_lab - 1``, matching
    :func:`_analytic_mean_correct`'s choice (the labeled term is the
    variance bottleneck there too, since n_all is typically much larger
    than n_lab)."""
    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)

    f_unlab = paired_walsh_midrank_theta(Y_hat_unlab)
    f_lab = paired_walsh_midrank_theta(Y_lab)
    f_hat_lab = paired_walsh_midrank_theta(Y_hat_lab)
    rectifier = f_lab - f_hat_lab

    var_unlab = _walsh_theta_analytic_variance(Y_hat_unlab) if n_all > 1 else 0.0
    var_lab = _walsh_theta_analytic_variance(Y_lab) if n_lab > 1 else 0.0
    var_hat_lab = _walsh_theta_analytic_variance(Y_hat_lab) if n_lab > 1 else 0.0

    if n_lab > 1:
        h1_lab = _walsh_theta_h1_components(Y_lab)
        h1_hat_lab = _walsh_theta_h1_components(Y_hat_lab)
        cov_lab_hatlab = 4.0 * float(np.cov(h1_lab, h1_hat_lab, ddof=1)[0, 1]) / n_lab
    else:
        cov_lab_hatlab = 0.0

    lam = 1.0
    if power_tune:
        denom = var_unlab + var_hat_lab
        if denom > 1e-12:
            lam = min(max(cov_lab_hatlab / denom, 0.0), 1.0)
        # else: degenerate variance -- fall back to lam=1, matching
        # _analytic_mean_correct and the bootstrap path.
        lam = 1.0 - (1.0 - lam) * n_lab / (n_lab + _POWER_TUNE_SHRINKAGE_C)

    estimate = f_lab + lam * (f_unlab - f_hat_lab)
    var_estimate = max(var_lab + lam * lam * (var_unlab + var_hat_lab) - 2.0 * lam * cov_lab_hatlab, 0.0)
    se = float(np.sqrt(var_estimate))
    df = max(n_lab - 1, 1)

    if se <= 0.0:
        ci_low = ci_high = estimate
        p_value = 1.0 if abs(estimate) < 1e-12 else 0.0
    else:
        t_crit = float(_t_dist.ppf(1.0 - alpha / 2.0, df))
        ci_low, ci_high = estimate - t_crit * se, estimate + t_crit * se
        p_value = min(max(float(2.0 * (1.0 - _t_dist.cdf(abs(estimate) / se, df))), 0.0), 1.0)

    return PPIResult(
        estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=rectifier,
        p_value=p_value, lam=(lam if power_tune else None),
    )


def _analytic_mean_point_se(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, Y_hat_unlab: np.ndarray, power_tune: bool,
) -> tuple[float, float, float, float, float, Optional[float], int]:
    """Shared closed-form point-estimate/SE/df computation for a PPI mean
    correction -- factored out of :func:`_analytic_mean_correct` so
    :func:`_analytic_logit_t_correct` can reuse the identical point-
    estimate/variance derivation (only the CI construction differs: a
    plain t-interval on the raw scale in ``_analytic_mean_correct`` vs. a
    delta-method logit-scale transform in ``_analytic_logit_t_correct``).
    See ``_analytic_mean_correct``'s docstring for the closed-form
    lambda*/variance derivation this implements.

    Returns (estimate, se, f_unlab, f_lab, rectifier, lam_or_None, df).
    """
    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)
    if n_all == 0:
        raise ValueError(
            "PPI correction requires at least one unlabeled item, got n_all=0 "
            "(every item in this comparison is human-labeled, leaving no "
            "unlabeled residual for the LLM-only term). If every item is "
            "labeled, use the human labels directly instead of PPI correction."
        )

    f_unlab = float(np.mean(Y_hat_unlab))
    f_lab = float(np.mean(Y_lab))
    f_hat_lab = float(np.mean(Y_hat_lab))
    rectifier = f_lab - f_hat_lab

    var_unlab = float(np.var(Y_hat_unlab, ddof=1)) / n_all if n_all > 1 else 0.0
    var_lab = float(np.var(Y_lab, ddof=1)) / n_lab if n_lab > 1 else 0.0
    var_hat_lab = float(np.var(Y_hat_lab, ddof=1)) / n_lab if n_lab > 1 else 0.0
    cov_lab_hatlab = float(np.cov(Y_lab, Y_hat_lab, ddof=1)[0, 1]) / n_lab if n_lab > 1 else 0.0

    lam = 1.0
    if power_tune:
        denom = var_unlab + var_hat_lab
        if denom > 1e-12:
            lam = min(max(cov_lab_hatlab / denom, 0.0), 1.0)
        # else: degenerate variance -- fall back to lam=1, matching the bootstrap path.
        # Same n_lab-dependent shrinkage toward 1 as correct()'s bootstrap path
        # (_POWER_TUNE_SHRINKAGE_C): a raw lambda* computed from only n_lab
        # points is itself a noisy estimate of the true population lambda.
        # Without this, a raw lambda*=0 (e.g. when Y_lab happens to have ~0
        # sample variance at small n_lab) collapses the estimate to f_lab with
        # se=0 exactly.
        lam = 1.0 - (1.0 - lam) * n_lab / (n_lab + _POWER_TUNE_SHRINKAGE_C)

    estimate = f_lab + lam * (f_unlab - f_hat_lab)
    var_estimate = max(var_lab + lam * lam * (var_unlab + var_hat_lab) - 2.0 * lam * cov_lab_hatlab, 0.0)
    se = float(np.sqrt(var_estimate))
    df = max(n_lab - 1, 1)

    return estimate, se, f_unlab, f_lab, rectifier, (lam if power_tune else None), df


def _analytic_mean_correct(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, Y_hat_unlab: np.ndarray,
    alpha: float, power_tune: bool,
) -> "PPIResult":
    """Closed-form (delta-method) PPI correction for
    ``estimator_func=np.mean`` -- no bootstrap resampling at all. See
    ``correct()``'s ``backend`` parameter for when this replaces the
    percentile bootstrap.

    The percentile bootstrap needs roughly n_lab >= 50 on noisy/discrete
    real data before Type-I error settles near nominal alpha; this
    closed-form path reaches the same target by n_lab ~= 25-30, since it
    plugs sample variances directly into a known distributional form
    (Student's t, df = n_lab - 1, since the labeled-subset term is the
    variance bottleneck) instead of approximating a sampling distribution
    from a small empirical resample.

    Point estimate is identical to the bootstrap path's (same f_unlab/
    f_lab/f_hat_lab/rectifier definitions); only the variance/CI/p-value
    construction differs. The shared point-estimate/variance/df
    computation lives in :func:`_analytic_mean_point_se` (this function is
    a thin wrapper around it, building a plain t-interval).

    power_tune's lambda* has a closed form here too (the original PPI++
    derivation for a mean/OLS-type estimand, Angelopoulos/Duchi/Zrnic
    2023): minimizing Var(F_lab + lambda*(F_unlab - F_hat_lab)) over
    lambda, where F_unlab is independent of (F_lab, F_hat_lab) by the
    disjointness requirement, gives

        lambda* = Cov(F_lab, F_hat_lab) / [Var(F_unlab) + Var(F_hat_lab)]

    with F_lab/F_hat_lab/F_unlab the sample means (not raw items) -- i.e.
    Var(F_hat_lab) = var(Y_hat_lab, ddof=1)/n_lab, Cov(F_lab, F_hat_lab) =
    cov(Y_lab, Y_hat_lab, ddof=1)/n_lab (paired at the item level,
    cross-item covariance is 0 under i.i.d. sampling), Var(F_unlab) =
    var(Y_hat_unlab, ddof=1)/n_all. ``correct()``'s bootstrap path
    estimates this same quantity by resampling, as a general-purpose
    stand-in that works for arbitrary ``estimator_func``; for the mean
    specifically this closed form is exact and needs no extra bootstrap
    pass. No small-n_lab shrinkage is applied (unlike the bootstrap path's
    ``_POWER_TUNE_SHRINKAGE_C``) -- that shrinkage exists specifically to
    compensate for the bootstrap's own small-sample weakness, which this
    path doesn't have.
    """
    estimate, se, f_unlab, f_lab, rectifier, lam, df = _analytic_mean_point_se(
        Y_lab, Y_hat_lab, Y_hat_unlab, power_tune,
    )

    if se <= 0.0:
        ci_low = ci_high = estimate
        p_value = 1.0 if abs(estimate) < 1e-12 else 0.0
    else:
        t_crit = float(_t_dist.ppf(1.0 - alpha / 2.0, df))
        ci_low, ci_high = estimate - t_crit * se, estimate + t_crit * se
        p_value = min(max(float(2.0 * (1.0 - _t_dist.cdf(abs(estimate) / se, df))), 0.0), 1.0)

    return PPIResult(
        estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=rectifier,
        p_value=p_value, lam=lam,
    )


def _analytic_logit_t_correct(
    Y_lab: np.ndarray, Y_hat_lab: np.ndarray, Y_hat_unlab: np.ndarray,
    alpha: float, power_tune: bool, lo: float = 0.0, hi: float = 1.0,
) -> "PPIResult":
    """Closed-form PPI correction for a [lo, hi]-bounded mean estimand, CI
    constructed on the logit scale -- the PPI analogue of
    ``evalstats.core.resampling.logit_t_ci_1d``. Built the same way
    ``_analytic_mean_correct`` is: identical point-estimate/variance
    derivation (delegated to ``_analytic_mean_point_se``, shared with
    ``_analytic_mean_correct`` -- the two differ only in how the CI is
    built from (estimate, se, df)), then a delta-method logit-scale
    t-interval instead of a plain one:

      1. Rescale (estimate, se) linearly onto [0, 1]: scaled = (x - lo) /
         (hi - lo). Valid because a linear rescale commutes with taking a
         sample mean/SE, so this does NOT require recomputing the point
         estimate/variance on rescaled arrays from scratch.
      2. logit(scaled_estimate), SE_logit = scaled_se / (scaled*(1-scaled))
         -- literally logit_t_ci_1d's own delta-method step, applied to
         PPI's corrected estimate/SE instead of a plain sample mean/SE.
      3. t-interval on the logit scale, df = max(n_lab-1, 1) -- the SAME
         df convention _analytic_mean_correct's plain t-interval uses
         (the labeled-subset term is this estimator's variance
         bottleneck either way).
      4. Back-transform via sigmoid, rescale to [lo, hi].

    Unlike logit_t_ci_1d's raw per-item values (guaranteed within [0, 1]
    once each item itself is, by that function's own stricter raw-value
    range check), PPI's corrected ESTIMATE is f_lab + lam*(f_unlab -
    f_hat_lab) -- a signed combination NOT itself constrained to [lo, hi]
    -- and CAN legitimately land outside it for a small/noisy labeled
    subset (the same phenomenon _ppi_single_wilson's docstring notes for
    its own p_hat_for_wilson clip). So instead of logit_t_ci_1d's
    raise-on-out-of-range (a real raw-data-hygiene bug there), the
    rescaled estimate here is clipped into [_LOGIT_T_BOUNDARY_EPS, 1 -
    _LOGIT_T_BOUNDARY_EPS] before the logit transform when it lands at or
    outside [0, 1] (with a UserWarning) -- reusing _ppi_single_wilson's
    established "clip before feeding a bounded-domain formula" precedent
    rather than inventing new logic. The returned ``estimate`` is left
    UN-clipped (same convention as _ppi_single_wilson: the reported point
    estimate is always the true PPI estimate; only the value fed
    internally to the bounded-domain formula is clipped). The final CI is
    clamped to [lo, hi] (matching logit_t_ci_1d's own guarantee).

    p_value uses estimate/se on the raw scale -- numerically identical to
    what ``_analytic_mean_correct`` would report on the same inputs. This
    matches ``evalstats/core/paired.py``'s classical (non-PPI)
    ``method="logit_t"`` branch, which also returns the plain paired-t-test
    p-value regardless of the logit CI construction -- only the CI's shape
    differs, never the significance test itself.
    """
    estimate, se, f_unlab, f_lab, rectifier, lam, df = _analytic_mean_point_se(
        Y_lab, Y_hat_lab, Y_hat_unlab, power_tune,
    )

    if se <= 0.0 or not np.isfinite(se):
        ci_low = ci_high = float(np.clip(estimate, lo, hi))
        p_value = 1.0 if abs(estimate) < 1e-12 else 0.0
        return PPIResult(
            estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
            llm_estimate=f_unlab, human_estimate=f_lab, rectifier=rectifier,
            p_value=p_value, lam=lam,
        )

    span = hi - lo
    scaled_est = (estimate - lo) / span
    scaled_se = se / span

    if scaled_est <= 0.0 or scaled_est >= 1.0:
        warnings.warn(
            f"_analytic_logit_t_correct: PPI-corrected estimate {estimate:.6g} is outside "
            f"[{lo:g}, {hi:g}] -- clipping toward the boundary for the logit transform. "
            "Expected occasionally for a small/noisy labeled subset (the correction term "
            "isn't itself constrained to [lo, hi] even though every individual observation "
            "is); see this function's docstring.",
            UserWarning, stacklevel=2,
        )
    clipped = float(np.clip(scaled_est, _LOGIT_T_BOUNDARY_EPS, 1.0 - _LOGIT_T_BOUNDARY_EPS))

    logit_mean = float(np.log(clipped / (1.0 - clipped)))
    se_logit = scaled_se / (clipped * (1.0 - clipped))
    t_crit = float(_t_dist.ppf(1.0 - alpha / 2.0, df))
    # scipy.special.expit (not a raw 1/(1+exp(-x))) -- numerically stable
    # for large |logit_mean +/- t_crit*se_logit|, which a clipped-but-still-
    # near-boundary estimate can produce (se_logit blows up as clipped
    # approaches 0 or 1); a raw exp() here can overflow before the ratio
    # collapses to the correct 0.0/1.0 limit.
    lo_s = float(_sigmoid(logit_mean - t_crit * se_logit))
    hi_s = float(_sigmoid(logit_mean + t_crit * se_logit))
    ci_low = float(np.clip(lo_s * span + lo, lo, hi))
    ci_high = float(np.clip(hi_s * span + lo, lo, hi))

    t_obs = estimate / se
    p_value = min(max(float(2.0 * (1.0 - _t_dist.cdf(abs(t_obs), df))), 0.0), 1.0)

    return PPIResult(
        estimate=estimate, ci_low=ci_low, ci_high=ci_high, alpha=alpha,
        llm_estimate=f_unlab, human_estimate=f_lab, rectifier=rectifier,
        p_value=p_value, lam=lam,
    )


_ANALYTIC_BACKENDS = {
    id(np.mean): _analytic_mean_correct,
    id(paired_walsh_midrank_theta): _analytic_walsh_theta_correct,
}
"""Maps ``estimator_func``'s identity (``id()``) to its closed-form
analytic corrector, for :func:`correct`'s ``backend`` dispatch. Extending
this to a new estimand requires both a corrector function matching
``_analytic_mean_correct``'s signature (``Y_lab, Y_hat_lab, Y_hat_unlab,
alpha, power_tune -> PPIResult``) and an entry here -- ``estimator_func``
and ``rectifier_func`` must resolve to the same entry, since the two-term
variance decomposition assumes the rectifier and the main estimand share
one variance/covariance model."""

_ANALYTIC_ALWAYS_PREFERRED = {id(paired_walsh_midrank_theta)}
"""``estimator_func`` identities that :func:`correct`'s ``backend="auto"``
routes to the analytic backend at every ``n_lab`` -- not only below
``_MIN_LAB_RECOMMENDED``, the threshold ``np.mean``'s entry in
:data:`_ANALYTIC_BACKENDS` still uses. ``paired_walsh_midrank_theta``
belongs here because its analytic backend beats the percentile bootstrap
on power across the full n_lab range with no calibration cost -- see
:func:`_analytic_walsh_theta_correct`'s docstring. ``np.mean`` is
deliberately not added here: its own n_lab<30 gate is independently
validated (see :func:`_analytic_mean_correct`) and this set exists so one
estimand's preference doesn't force a review of the shared
``_MIN_LAB_RECOMMENDED`` threshold."""


def resolve_arrays(
    df,
    *,
    metric_col: str,
    group_col: str,
    alignment_result,
):
    """Extract PPI arrays from a DataFrame and an AlignmentResult.

    Parameters
    ----------
    df : pd.DataFrame
    metric_col : str
        Column of LLM scores (present for all rows).
    group_col : str
        Column of group labels (factor / condition).
    alignment_result : AlignmentResult
        From :func:`~evalstats.alignment.validate_alignment`.
        Its ``human_col`` attribute identifies the sparse human-label column.

    Returns
    -------
    tuple
        ``(Y_hat_unlab, X_unlab, Y_lab, Y_hat_lab, X_lab)`` as numpy arrays,
        ready to pass directly to :func:`correct`. ``Y_hat_unlab``/
        ``X_unlab`` EXCLUDE the labeled rows (disjoint from ``Y_lab``/
        ``Y_hat_lab``/``X_lab``) -- see :func:`correct`'s docstring for why
        that disjointness is required for its bootstrap to be valid.
    """
    human_col = alignment_result.human_col
    labeled_mask = df[human_col].notna()
    unlabeled = df.loc[~labeled_mask]
    Y_hat_unlab = unlabeled[metric_col].to_numpy(dtype=float)
    X_unlab     = unlabeled[group_col].to_numpy()
    Y_lab       = df.loc[labeled_mask, human_col].to_numpy(dtype=float)
    Y_hat_lab   = df.loc[labeled_mask, metric_col].to_numpy(dtype=float)
    X_lab       = df.loc[labeled_mask, group_col].to_numpy()
    return Y_hat_unlab, X_unlab, Y_lab, Y_hat_lab, X_lab


# ── Public API ────────────────────────────────────────────────────────────────

def correct(
    estimator_func: Callable,
    *,
    Y_lab,
    Y_hat_lab,
    Y_hat_unlab,
    X_lab=None,
    X_unlab=None,
    alpha: float = 0.05,
    n_boot: int = 1000,
    rng=None,
    compute_pvalue: bool = True,
    rectifier_func: Optional[Callable] = None,
    power_tune: bool = True,
    backend: str = "auto",
) -> PPIResult:
    """Correct any scalar estimator for LLM judge measurement error using PPI.

    Given a large LLM-scored dataset and a small human-annotated subset,
    this function returns a bias-corrected estimate and bootstrap CI.

    The PPI corrected estimator is:

    .. code-block:: text

        θ̂_PPI = f(Ŷ_unlab, X_unlab)    [LLM on full unlabeled set]
               + f(Y_lab,   X_lab)       [human on labeled subset]
               − f(Ŷ_lab,  X_lab)       [LLM on labeled subset]

    The last two terms form the *rectifier*: the signed difference between
    what the human and the LLM said about the same items.  When the LLM is
    unbiased the rectifier is near zero; when it is biased the rectifier
    shifts the estimate toward the truth.

    This is the ORIGINAL prediction-powered inference estimator (Angelopoulos
    et al. 2023) -- equivalently, a *power-tuning* weight λ fixed at 1 in the
    generalized family θ̂_λ = f(Y_lab, X_lab) + λ·[f(Ŷ_unlab, X_unlab) −
    f(Ŷ_lab, X_lab)] (λ=0 recovers the classical, labels-only estimator;
    λ=1 recovers the formula above). See *power_tune* below for the
    variance-minimizing choice of λ (PPI++, Angelopoulos/Duchi/Zrnic 2023),
    which is never less efficient than the classical estimator by
    construction -- fixing λ=1 unconditionally (this function's behavior
    when *power_tune=False*) has no such guarantee: a sufficiently
    uninformative or noisy LLM can make the λ=1 estimator strictly WORSE
    (wider CI, lower power) than just running the classical test on
    *Y_lab* alone, since the rectifier's own bootstrap variance is paid in
    full regardless of whether the LLM earns it back.

    A percentile bootstrap CI is computed by independently resampling the
    unlabeled set (size N) and the labeled set (size n_lab) on each draw
    and recomputing the PPI estimator. This independent resampling is only
    valid because *Y_hat_unlab* and the labeled set are assumed to be
    genuinely DISJOINT samples (the original prediction-powered inference
    setup, Angelopoulos et al. 2023) — disjoint samples are independent by
    construction, so bootstrapping them separately is exact.

    **Callers must exclude the labeled items from *Y_hat_unlab*/*X_unlab*.**
    A common mistake: if your data is one score column plus a sparse human-
    label overlay (label human-reviewed a subset of everything you already
    LLM-scored), *Y_hat_unlab* is NOT "every item's LLM score" — it's only
    the LLM scores for the items that do NOT also appear in *Y_lab*. Passing
    the full (overlapping) array instead silently breaks the independence
    this function's bootstrap relies on: the two terms then share items, so
    resampling them separately ignores their true covariance and produces a
    CI/p-value that drifts from nominal coverage as a function of n_lab
    (confirmed via simulation — see ``simulations/harness/cases/pvalues.py
    --mode ppi``'s N x N_lab calibration grid). There is no parameter here
    to opt into a "shared" mode; correctness depends entirely on the caller
    constructing a genuinely disjoint *Y_hat_unlab* up front.

    **The labeled subset (*Y_lab*) must be selected independently of its own
    outcome value — ideally a uniform random sample of the full dataset.**
    The rectifier's validity relies on the labeled items being representative
    of the population it's correcting; if which items get human-labeled is
    itself influenced by the (true or LLM-judged) score — e.g. "always
    double-check the borderline/highest-scoring responses" — this is
    literally MNAR (missing-not-at-random) selection on the outcome, and the
    correction can stay miscalibrated by a large, non-vanishing amount
    regardless of how large *n_lab* is. This is not a small-sample artifact
    that more labels fixes: confirmed via simulation to persist (30–65%
    false-positive rate against a 5% nominal target) from n_lab=15 up through
    n_lab=300 out of N=400, non-monotonically, under a labeling process that
    preferentially selects high-scoring items (see
    ``simulations/harness/cases/pvalues.py --mode ppi``'s MNAR-labeling
    sweep). A stratified/local-rectifier mitigation was prototyped and does
    reduce the worst cells substantially at large n_lab (≳80), but only at
    the cost of measurably worse calibration on the common, correctly-random-
    sampled case — not worth the trade given the fix that actually works for
    free: sample which items to label uniformly at random. There is
    currently no supported way to opt into a stratified/propensity-adjusted
    rectifier for non-random labeling; random sampling of the labeled subset
    is a hard requirement for this function's calibration guarantee, not a
    recommendation to weigh against convenience.

    Parameters
    ----------
    estimator_func : callable
        ``f(Y) → float`` or ``f(Y, X) → float``.  Receives numpy arrays;
        must return a scalar.  X is forwarded only when *X_lab* / *X_unlab*
        are supplied.
    Y_lab : array-like, shape (n_lab,)
        Human-annotated scores for the labeled subset.
    Y_hat_lab : array-like, shape (n_lab,)
        LLM scores for the same items as *Y_lab* (paired, same order).
    Y_hat_unlab : array-like, shape (N,)
        LLM scores for the UNLABELED dataset only — i.e. every item that
        does NOT also appear in *Y_lab*/*Y_hat_lab*. Must be disjoint from
        the labeled set; see the warning above.
    X_lab : array-like, shape (n_lab, ...), optional
        Covariates / condition labels for the labeled subset.
        When provided, passed as the second argument to *estimator_func*,
        indexed consistently with Y.  Requires *X_unlab* to also be given.
    X_unlab : array-like, shape (N, ...), optional
        Covariates / condition labels for the full dataset.
        Required when *X_lab* is provided.
    alpha : float
        Significance level; ``1 − alpha`` gives the CI width (default 0.05).
    n_boot : int
        Bootstrap resamples (default 1000).
    rng : int or numpy.random.Generator, optional
        Seed or Generator for reproducibility.
    compute_pvalue : bool
        Compute a two-sided p-value for H₀: θ = 0 (default True).
    rectifier_func : callable, optional
        Alternative estimator used for the rectifier terms ``f(Y_lab)`` and
        ``f(Ŷ_lab)`` only.  When *None* (default), *estimator_func* is used
        for all three terms.  Providing a different function (e.g. ``np.mean``
        when *estimator_func* is ``np.median``) can improve bootstrap
        calibration for non-smooth estimands like the median.
    power_tune : bool
        When *True* (the default), use PPI++'s variance-minimizing power-
        tuning weight λ instead of fixing λ=1 (the original, 2023 PPI
        estimator -- pass *False* to reproduce it exactly). λ̂ is estimated
        from a bootstrap replicate draw -- λ̂ = Ĉov(b_lab, b_hat_lab) /
        V̂ar(b_unlab − b_hat_lab), where ``b_unlab``/``b_lab``/``b_hat_lab``
        are per-replicate ``f(Ŷ_unlab)``/``f(Y_lab)``/``f(Ŷ_hat_lab)`` draws
        -- rather than a closed-form gradient derivation specific to one
        model family (the PPI++ paper derives these for OLS/logistic/
        quantile regression; this bootstrap-plug-in version is a
        general-purpose stand-in that works for *any* ``estimator_func``,
        generalizing the same variance-minimization argument). A second,
        independent bootstrap draw then builds the percentile CI at that
        fixed λ̂ -- estimating λ̂ from the same draw used for the CI
        measurably undercovers, since λ̂ ends up partly optimized against
        noise specific to that one draw ("double dipping"); the split
        removes that circularity at the cost of one extra bootstrap pass.
        λ̂ is clipped to ``[0, 1]``, then shrunk back toward 1 by an
        n_lab-dependent amount (see ``_POWER_TUNE_SHRINKAGE_C``) -- a
        small n_lab makes the raw λ̂ estimate itself unreliable and exposes
        a separate, pre-existing small-sample weakness of the percentile
        bootstrap that this shrinkage compensates for; it is an empirical
        patch, not part of the published PPI++ derivation. Un-shrunk λ̂=0
        falls back to the classical labels-only estimate
        (``human_estimate``) when the LLM adds no value; λ=1 reproduces
        ``power_tune=False``'s estimate when the LLM is fully informative
        or n_lab is small. Falls back to λ=1 (unchanged behavior) if the
        bootstrap variance in the denominator is degenerate (≈0).
        ``PPIResult.lam`` reports the (shrunk) value actually used.

        ``kruskal``/``anova``/``friedman``/``bootstrap_t``/``tango_score``/
        ``lmm*`` and the MNAR-experimental rectifiers do not go through
        this function (bespoke bootstrap/closed-form code of their own)
        and are unaffected by ``power_tune``. For kruskal/anova/friedman
        this is deliberate: power-tuning does not transfer to their
        variance-like, quadratic-form estimand, whose λ=0 endpoint is the
        raw, judge-biased estimate rather than a safe classical fallback
        the way a scalar mean's is -- see ``simulations/harness/README.md``'s
        "PPI++ power-tuning" section.
    backend : {"auto", "bootstrap", "analytic"}
        How to build the CI/p-value. "bootstrap" is the percentile-
        resampling method described above, unconditionally. "analytic" is
        a closed-form (delta-method / Hajek-projection, depending on the
        estimand) alternative for ``estimator_func`` registered in
        :data:`_ANALYTIC_BACKENDS` (``np.mean`` -- see
        :func:`_analytic_mean_correct` -- and
        :func:`paired_walsh_midrank_theta` -- see
        :func:`_analytic_walsh_theta_correct`) with no covariates
        (``X_lab``/``X_unlab`` both None) and ``rectifier_func`` matching
        ``estimator_func`` (or ``None``); raises ``ValueError`` if
        requested for anything else. "auto" (the default) uses "analytic"
        when it's applicable and either ``n_lab < 30`` (below which the
        percentile bootstrap is known to undercover) or ``estimator_func``
        is in :data:`_ANALYTIC_ALWAYS_PREFERRED` (currently just
        ``paired_walsh_midrank_theta``, whose analytic backend beats the
        bootstrap on power at every n_lab -- see that constant's
        docstring), otherwise falls back to "bootstrap" and, if
        ``n_lab < 30`` there too, emits a ``UserWarning`` instead of
        silently returning an under-covering interval.

        On noisy/discrete real data, the percentile bootstrap needs
        n_lab >~ 50 before Type-I error settles near nominal alpha, while
        an applicable analytic path reaches the same target by
        n_lab ~= 25-30, since it plugs sample variances directly into a
        known (Student's t) distributional form instead of approximating a
        sampling distribution from a small empirical resample.

    Returns
    -------
    PPIResult

    Raises
    ------
    ValueError
        If inputs are malformed (invalid ``alpha``/``n_boot``, inconsistent
        lengths, empty arrays, non-finite values), or if exactly one of
        *X_lab* / *X_unlab* is supplied.

    Examples
    --------
    >>> import numpy as np
    >>> import evalstats as es
    >>>
    >>> def mean_diff(Y, X):
    ...     \"\"\"Mean score under condition A minus condition B.\"\"\"
    ...     return float(Y[X == "A"].mean() - Y[X == "B"].mean())
    >>>
    >>> result = es.ppi.correct(
    ...     estimator_func=mean_diff,
    ...     Y_lab=gold_df["human_score"].values,
    ...     Y_hat_lab=gold_df["llm_score"].values,
    ...     Y_hat_unlab=large_df["llm_score"].values,
    ...     X_lab=gold_df["condition"].values,
    ...     X_unlab=large_df["condition"].values,
    ...     alpha=0.05,
    ... )
    >>> result.summary()
    """
    rng = np.random.default_rng(rng)

    # Validate scalar control parameters early for clearer errors.
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as e:
        raise ValueError(f"alpha must be a finite float in (0, 1); got {alpha!r}.") from e
    if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")

    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)):
        raise ValueError(f"n_boot must be a positive integer; got {n_boot!r}.")
    if int(n_boot) <= 0:
        raise ValueError(f"n_boot must be a positive integer; got {n_boot!r}.")
    n_boot = int(n_boot)

    # ── Coerce inputs ─────────────────────────────────────────────────────────
    Y_lab       = np.asarray(Y_lab,       dtype=float)
    Y_hat_lab   = np.asarray(Y_hat_lab,   dtype=float)
    Y_hat_unlab = np.asarray(Y_hat_unlab, dtype=float)

    if Y_lab.ndim == 0 or Y_hat_lab.ndim == 0 or Y_hat_unlab.ndim == 0:
        raise ValueError(
            "Y_lab, Y_hat_lab, and Y_hat_unlab must be at least 1-D arrays."
        )
    if Y_lab.ndim != Y_hat_lab.ndim:
        raise ValueError(
            f"Y_lab and Y_hat_lab must have the same ndim "
            f"(got {Y_lab.ndim} vs {Y_hat_lab.ndim})."
        )
    if Y_hat_unlab.ndim != Y_hat_lab.ndim:
        raise ValueError(
            f"Y_hat_unlab and Y_hat_lab must have the same ndim "
            f"(got {Y_hat_unlab.ndim} vs {Y_hat_lab.ndim})."
        )
    if Y_lab.shape[1:] != Y_hat_lab.shape[1:]:
        raise ValueError(
            f"Y_lab and Y_hat_lab must have matching trailing shape "
            f"(got {Y_lab.shape[1:]} vs {Y_hat_lab.shape[1:]})."
        )
    if Y_hat_unlab.shape[1:] != Y_hat_lab.shape[1:]:
        raise ValueError(
            f"Y_hat_unlab and Y_hat_lab must have matching trailing shape "
            f"(got {Y_hat_unlab.shape[1:]} vs {Y_hat_lab.shape[1:]})."
        )

    if X_lab is not None:
        X_lab = np.asarray(X_lab)
    if X_unlab is not None:
        X_unlab = np.asarray(X_unlab)

    # ── Validate ──────────────────────────────────────────────────────────────
    if len(Y_lab) != len(Y_hat_lab):
        raise ValueError(
            f"Y_lab and Y_hat_lab must have the same length "
            f"(got {len(Y_lab)} vs {len(Y_hat_lab)})"
        )
    if (X_lab is None) != (X_unlab is None):
        raise ValueError(
            "Provide both X_lab and X_unlab, or neither. "
            "Exactly one was supplied."
        )
    if X_lab is not None and len(X_lab) != len(Y_lab):
        raise ValueError(
            f"X_lab must have the same length as Y_lab "
            f"(got {len(X_lab)} vs {len(Y_lab)})"
        )
    if X_unlab is not None and len(X_unlab) != len(Y_hat_unlab):
        raise ValueError(
            f"X_unlab must have the same length as Y_hat_unlab "
            f"(got {len(X_unlab)} vs {len(Y_hat_unlab)})"
        )

    if len(Y_lab) == 0:
        raise ValueError("Y_lab and Y_hat_lab must be non-empty.")
    if len(Y_hat_unlab) == 0:
        raise ValueError(
            "Y_hat_unlab must be non-empty. This usually means every item is "
            "already labeled -- PPI has no unlabeled pool left to extrapolate "
            "the correction to. With 100% human labels, just run a classical "
            "test directly on Y_lab instead of PPI."
        )

    if not np.all(np.isfinite(Y_lab)):
        raise ValueError("Y_lab contains non-finite values (NaN/inf).")
    if not np.all(np.isfinite(Y_hat_lab)):
        raise ValueError("Y_hat_lab contains non-finite values (NaN/inf).")
    if not np.all(np.isfinite(Y_hat_unlab)):
        raise ValueError("Y_hat_unlab contains non-finite values (NaN/inf).")

    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)

    _rect_fn = rectifier_func if rectifier_func is not None else estimator_func

    # ── backend dispatch (auto/bootstrap/analytic) ────────────────────────────
    if backend not in ("auto", "bootstrap", "analytic"):
        raise ValueError(f"backend must be 'auto', 'bootstrap', or 'analytic'; got {backend!r}.")
    _analytic_fn = _ANALYTIC_BACKENDS.get(id(estimator_func))
    _analytic_available = (
        _analytic_fn is not None and id(_rect_fn) == id(estimator_func)
        and X_lab is None and X_unlab is None
    )
    if backend == "analytic" and not _analytic_available:
        raise ValueError(
            "backend='analytic' requires estimator_func to be one of the registered analytic "
            "estimands (np.mean, or evalstats.ppi.paired_walsh_midrank_theta -- see "
            "_ANALYTIC_BACKENDS), with rectifier_func matching estimator_func (or None) and no "
            f"covariates; got estimator_func={estimator_func!r}, rectifier_func={rectifier_func!r}, "
            f"X_lab={'given' if X_lab is not None else None}."
        )
    use_analytic = backend == "analytic" or (
        backend == "auto" and _analytic_available and (
            n_lab < _MIN_LAB_RECOMMENDED or id(estimator_func) in _ANALYTIC_ALWAYS_PREFERRED
        )
    )
    if not use_analytic and n_lab < _MIN_LAB_RECOMMENDED:
        warnings.warn(
            f"PPI bootstrap CI/p-value with only {n_lab} labeled items (recommend >= "
            f"{_MIN_LAB_RECOMMENDED}) is known to undercover -- Type-I error above nominal "
            "alpha should be expected."
            + ("" if _analytic_available else " No closed-form 'analytic' backend is available "
               "for this estimator_func; consider collecting more labels."),
            UserWarning, stacklevel=2,
        )
    if use_analytic:
        return _analytic_fn(Y_lab, Y_hat_lab, Y_hat_unlab, alpha, power_tune)

    # ── Point estimate (lambda=1 terms; combined into `estimate` below) ──────
    f_unlab   = _call(estimator_func, Y_hat_unlab, X_unlab)
    f_lab     = _call(_rect_fn,       Y_lab,       X_lab)
    f_hat_lab = _call(_rect_fn,       Y_hat_lab,   X_lab)
    rectifier = f_lab - f_hat_lab

    # ── Bootstrap replicates ──────────────────────────────────────────────────
    # Fast path: when there are no covariates and both functions are one of
    # the built-ins actually used by this codebase's internal PPI dispatch
    # (np.mean / np.median / paired_walsh_midrank_theta), the whole
    # bootstrap batches over an added replicate axis instead of a Python
    # loop with n_boot scalar calls each -- this matters most for
    # np.median (which re-sorts on every call) and paired_walsh_midrank_
    # theta (an O(n log n) sort+searchsorted per call -- see that
    # function's docstring for why it NEEDS this fast path to be practical
    # at all, unlike the O(n^2) construction it replaced). Falls back to
    # the general per-replicate loop for arbitrary user-supplied estimator
    # functions or when X_lab/X_unlab are provided.
    #
    # Factored into a helper (drawing ONE full set of n_boot replicates per
    # call) because power_tune needs TWO independent draws -- see below.
    _fast_batch = {
        id(np.mean): lambda a: a.mean(axis=1),
        id(np.median): lambda a: np.median(a, axis=1),
        id(paired_walsh_midrank_theta): _walsh_theta_batch,
    }
    fast_est = _fast_batch.get(id(estimator_func)) if X_unlab is None else None
    fast_rect = _fast_batch.get(id(_rect_fn)) if X_lab is None else None

    # Smoothed-bootstrap jitter (see _tie_jitter_scale) -- only when the
    # estimator/rectifier is np.median specifically (mean's bootstrap
    # distribution doesn't degenerate under ties; jittering it would just
    # add pointless noise). 0.0 disables jitter (np.random.normal(0, 0, ...)
    # is exactly a no-op, so this is safe to add unconditionally below).
    _jitter_unlab = _tie_jitter_scale(Y_hat_unlab) if fast_est is not None and estimator_func is np.median else 0.0
    _jitter_labpair = (
        _tie_jitter_scale(np.concatenate([Y_lab, Y_hat_lab]))
        if fast_rect is not None and _rect_fn is np.median else 0.0
    )

    def _draw_replicates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b_unlab_arr = np.empty(n_boot)
        b_lab_arr = np.empty(n_boot)
        b_hat_lab_arr = np.empty(n_boot)
        if fast_est is not None and fast_rect is not None:
            chunk_size = max(1, min(n_boot, 4096, max(1, int(2_000_000 // max(n_all, n_lab, 1)))))
            start = 0
            while start < n_boot:
                stop = min(start + chunk_size, n_boot)
                m = stop - start
                idx_all = rng.integers(0, n_all, size=(m, n_all))
                idx_lab = rng.integers(0, n_lab, size=(m, n_lab))
                resampled_unlab = Y_hat_unlab[idx_all]
                resampled_lab = Y_lab[idx_lab]
                resampled_hat_lab = Y_hat_lab[idx_lab]
                if _jitter_unlab > 0.0:
                    resampled_unlab = resampled_unlab + rng.normal(0.0, _jitter_unlab, size=resampled_unlab.shape)
                if _jitter_labpair > 0.0:
                    resampled_lab = resampled_lab + rng.normal(0.0, _jitter_labpair, size=resampled_lab.shape)
                    resampled_hat_lab = resampled_hat_lab + rng.normal(0.0, _jitter_labpair, size=resampled_hat_lab.shape)
                b_unlab_arr[start:stop]   = fast_est(resampled_unlab)
                b_lab_arr[start:stop]     = fast_rect(resampled_lab)
                b_hat_lab_arr[start:stop] = fast_rect(resampled_hat_lab)
                start = stop
        else:
            for b in range(n_boot):
                idx_all = rng.integers(0, n_all, n_all)
                idx_lab = rng.integers(0, n_lab, n_lab)
                Xa_b = X_unlab[idx_all] if X_unlab is not None else None
                Xl_b = X_lab[idx_lab]   if X_lab   is not None else None
                Yl = Y_hat_unlab[idx_all]
                Ya = Y_lab[idx_lab]
                Yb = Y_hat_lab[idx_lab]
                if _jitter_unlab > 0.0:
                    Yl = Yl + rng.normal(0.0, _jitter_unlab, size=Yl.shape)
                if _jitter_labpair > 0.0:
                    Ya = Ya + rng.normal(0.0, _jitter_labpair, size=Ya.shape)
                    Yb = Yb + rng.normal(0.0, _jitter_labpair, size=Yb.shape)
                b_unlab_arr[b]   = _call(estimator_func, Yl, Xa_b)
                b_lab_arr[b]     = _call(_rect_fn,       Ya, Xl_b)
                b_hat_lab_arr[b] = _call(_rect_fn,       Yb, Xl_b)
        return b_unlab_arr, b_lab_arr, b_hat_lab_arr

    # ── Power tuning (PPI++, Angelopoulos/Duchi/Zrnic 2023) ──────────────────
    # lambda* minimizes Var(f_lab + lambda*(f_unlab - f_hat_lab)); since the
    # unlabeled replicate is drawn from an independent resample of the
    # (disjoint) unlabeled set, Cov(b_lab, b_unlab) = 0 in the bootstrap
    # distribution, which reduces the minimizer to a single covariance/
    # variance ratio -- see correct()'s power_tune parameter docstring for
    # the full derivation.
    #
    # Two independent bootstrap draws are used when power_tune=True: one
    # (b1_*) only to estimate lambda, a second, fresh one (b2_*) only to
    # build the percentile CI at that now-fixed lambda. Estimating lambda
    # and building its CI from the same replicates measurably undercovers
    # nominal coverage, since lambda ends up partly optimized away noise
    # specific to that one bootstrap draw ("double dipping"); splitting the
    # two draws removes that circularity at the cost of one extra bootstrap
    # pass (still cheap via the fast-batch path above). power_tune=False
    # needs only one draw.
    #
    # lambda is then shrunk back toward 1 by an n_lab-dependent amount
    # (lam_reg = 1 - (1-lam)*n_lab/(n_lab+_POWER_TUNE_SHRINKAGE_C)) before
    # being applied. Without this, power_tune=True's Type-I error runs
    # measurably worse than power_tune=False's baseline at small n_lab: the
    # percentile bootstrap CI of a small sample is itself mildly
    # anti-conservative, and vanilla PPI's fixed λ=1 masks that by always
    # blending in the large, well-behaved unlabeled-sample bootstrap.
    # Power-tuning, by correctly identifying an uninformative judge and
    # shrinking λ toward 0, leans more on that same small-sample
    # bootstrap -- which unmasks its pre-existing weakness rather than
    # introducing a new one. Shrinking lambda itself back toward 1 as
    # n_lab shrinks defers to vanilla PPI's already-tolerable baseline in
    # that regime. This is not part of the published PPI++ derivation --
    # it's an empirical patch for a bootstrap-construction limitation this
    # codebase already had, layered on top.
    lam: Optional[float] = None
    if power_tune:
        b1_unlab, b1_lab, b1_hat_lab = _draw_replicates()
        denom = float(np.var(b1_unlab - b1_hat_lab, ddof=1))
        if denom > 1e-12:
            lam = float(np.cov(b1_lab, b1_hat_lab, ddof=1)[0, 1] / denom)
            lam = min(max(lam, 0.0), 1.0)
        else:
            lam = 1.0  # degenerate bootstrap variance -- fall back, don't divide by ~0.
        lam = 1.0 - (1.0 - lam) * n_lab / (n_lab + _POWER_TUNE_SHRINKAGE_C)
        b2_unlab, b2_lab, b2_hat_lab = _draw_replicates()
        estimate = f_lab + lam * (f_unlab - f_hat_lab)
        boots = b2_lab + lam * (b2_unlab - b2_hat_lab)
    else:
        b_unlab_arr, b_lab_arr, b_hat_lab_arr = _draw_replicates()
        estimate = f_unlab + (f_lab - f_hat_lab)
        boots = b_unlab_arr + (b_lab_arr - b_hat_lab_arr)

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))

    # ── p-value ───────────────────────────────────────────────────────────────
    p_value: Optional[float] = None
    if compute_pvalue:
        # Proportion of bootstrap draws on each side of 0; two-sided.
        p_value = float(2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0)))
        p_value = min(max(p_value, 0.0), 1.0)

    return PPIResult(
        estimate=float(estimate),
        ci_low=lo,
        ci_high=hi,
        alpha=alpha,
        llm_estimate=float(f_unlab),
        human_estimate=float(f_lab),
        rectifier=float(rectifier),
        p_value=p_value,
        lam=lam,
    )
