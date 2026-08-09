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
"""Below this many labeled items, the percentile bootstrap this module
otherwise always used is known to undercover (see correct()'s docstring
and the "backend" parameter below) -- this is the SAME threshold
evalstats/api.py's PPI-alignment warning and evalstats/alignment.py
already use, not a new number invented for this module."""

_POWER_TUNE_SHRINKAGE_C = 20.0
"""Pseudo-count for shrinking correct()'s power-tuning weight lambda back
toward 1 (vanilla PPI) as n_lab shrinks -- see the power_tune shrinkage
comment inside correct() for the Monte Carlo finding this fixes. 20 matches
this codebase's existing "recommend >=20-30 labeled items" convention
(see evalstats.tests' module docstring) rather than introducing a new,
unvetted number; confirmed via simulation to close most of power_tune's
extra small-n_lab Type-I gap at n_lab=20 without fully collapsing back to
lambda=1 (i.e. without giving up all of the label-efficiency benefit)."""


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
"""correct()'s smoothed-bootstrap jitter std is (min positive gap between
distinct values in the data) / this divisor. 20 keeps the jitter well
below the data's real tie resolution (so it only breaks EXACT ties,
without perturbing genuine order structure at any coarser scale) while
still being large enough to stop bootstrap resamples from repeatedly
landing on the identical median -- see _tie_jitter_scale's docstring for
the failure mode this fixes."""


def _tie_jitter_scale(arr: np.ndarray) -> float:
    """Std-dev of the Gaussian jitter correct() adds before each bootstrap
    resample's median, when estimator_func/rectifier_func is np.median.

    Percentile-bootstrapping a MEDIAN on data with substantial exact ties
    is a known-bad combination (the "smoothed bootstrap" literature, e.g.
    Efron; Hall & DiCiccio-Romano on bootstrapping non-smooth statistics):
    with enough repeated values, most resamples' median lands on the SAME
    repeated value, so the bootstrap distribution collapses toward a
    near-constant, and the resulting CI/p-value is severely (not just
    mildly) too conservative -- confirmed directly on real wmt_da judge-
    score-diff data (2026-07-23): 92.6% of bootstrap replicate differences
    collapsed to exactly 0, driving a paired Wilcoxon PPI check's Type-I
    error to ~0 (vs. nominal 0.05) and NOT improving with n/n_lab (flat up
    to n_lab=400), since tie density is a property of the score scale, not
    sample size. Adding noise far below the data's real resolution before
    each resample's median restores a non-degenerate bootstrap distribution
    without perturbing genuine order structure.

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
    (``n*(n+1)/2`` pairs total, INCLUDING self-pairs ``i == j``). This is
    the Hodges-Lehmann one-sample location estimator's own construction
    (its point estimate is the MEDIAN of the Walsh averages; this is the
    analogous midrank-tie-corrected SIGN statistic of those same averages)
    -- see :func:`paired_walsh_midrank_theta`'s docstring for why this
    replaces the simpler per-item sign proportion PPI previously used for
    Wilcoxon.

    Computed via a sort + vectorized ``searchsorted`` rather than the
    naive O(n^2) pairwise enumeration (validated against an O(n^2) brute-
    force reference over 2000 random trials, including heavily-tied
    integer data, with zero discrepancy) -- the O(n^2) version is too slow
    to call thousands of times per :func:`correct` invocation at realistic
    corpus sizes (confirmed impractical at n~300, let alone ~1000+)."""
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
    about 0) -- :func:`correct`'s estimand for the Wilcoxon signed-rank
    family in ``evalstats.tests._ppi_paired_arrays``, replacing an earlier
    per-item sign proportion (``P(D>0) - 0.5``, briefly used 2026-07-25) as
    of 2026-07-26.

    Why introduced: the per-item sign proportion fixed Wilcoxon's median-
    based power collapse under heavy ties (population median of a paired
    difference can stay locked at exactly 0 even under a large, real,
    classical-Wilcoxon-detectable shift -- a wrong-estimand problem, not a
    bootstrap-degeneracy one). It was ALSO found to have severely inflated
    Type-I error (up to 28% vs. nominal 5%) when bootstrapped at small
    n_lab (~15) against real judge-pair data with an extreme tie rate (88%
    of items scored identically by two real judges on real appstore data)
    -- confirmed NOT caused by power-tuning (near-identical Type-I with
    power_tune on/off) nor fixable by a small-sample continuity correction
    on the point estimate (barely moved the needle, 28% -> 24% at
    strongest). This function -- counting signs of ALL PAIRWISE Walsh
    averages ``(d_i + d_j) / 2`` instead of individual item signs, the
    exact construction behind the Hodges-Lehmann one-sample location
    estimator and asymptotically equivalent to the Wilcoxon signed-rank
    statistic itself (unlike the sign-proportion it replaces, which only
    approximates it) -- was hypothesized to fix that inflation the same
    way ``evalstats.tests._p_x_gt_y_midrank`` (MWU's own two-sample
    U-statistic midrank correction) avoids it.

    That hypothesis did NOT hold on the specific extreme-tie appstore
    cell: measured Type-I was 26.4%, statistically indistinguishable from
    the sign proportion's own 26.4% on the identical data (2026-07-26).
    Root-caused further: on that specific population (88% exact zeros,
    heavily concentrated at one point), this function's raw value is a
    near-EXACT constant rescaling of the sign proportion's raw value
    (measured ratio ~1.88, stable to within 1% across 30 independent
    draws) -- since a percentile-bootstrap p-value is invariant to
    multiplying an estimator by a fixed positive constant (both the point
    estimate and every bootstrap replicate scale together), the two
    estimators give MATHEMATICALLY EQUIVALENT inference whenever the
    underlying population is this degenerate, regardless of which one is
    used. The earlier "U-statistic vs. raw proportion" explanation for
    MWU's better small-n_lab behavior therefore appears insufficient (or
    wrong) as a general theory -- MWU's twogroup null check uses two
    INDEPENDENT random subsamples, not this check's deterministically-
    identical-labeled-term construction (see
    scenarios/real_judge_bias.py's generate_real_paired_null_cell), which
    may be the more load-bearing structural difference. Kept as Wilcoxon's
    estimator anyway because it's still a genuine, real improvement over
    the ORIGINAL (median-based) estimator's power collapse, is validated
    safe (no regression) on continuous and non-extreme-tie data, and is
    the theoretically correct signed-rank construction -- but this
    specific extreme-tie, small-n_lab Type-I inflation remains OPEN and is
    NOT fixed by this function. See cases/pvalues.py's/ppi_real.py's
    WILCOXON blocks for where this is used and further discussion.

    UPDATE (2026-08-01): this extreme-tie residual was found to be
    substantially (not fully) reduced by :func:`correct`'s new
    ``paired_walsh_midrank_theta`` analytic backend (see
    :func:`_analytic_walsh_theta_correct`), which now activates
    automatically (``backend="auto"``, the default) below ``n_lab=30`` --
    the SAME n_lab range this extreme-tie inflation was measured in. On
    the SAME real appstore data at n_lab=15
    (``simulations/harness/scenarios/real_judge_bias.py``'s
    ``generate_real_paired_null_cell``, judges anthropic/claude-haiku-4.5
    x google/gemma-4-26b-a4b-it and google/gemma-4-26b-a4b-it x
    thinkingmachines/inkling -- a different, currently-collected judge
    roster than the original 88%-tie pair this docstring's numbers above
    were measured on, since judge rosters have changed since 2026-07-26;
    a like-for-like re-run against the ORIGINAL 88%-tie pair was not
    repeated), measured Type-I with the OLD (percentile-bootstrap-only)
    construction was 23.4-25.2% (500 reps, n_boot=1500) -- reproducing
    the same OPEN residual this docstring describes -- while the NEW
    analytic path measured 12.0-17.2% on the IDENTICAL data/seeds:
    roughly HALVES the inflation, but does NOT eliminate it (still well
    above nominal 5%). This is a genuine improvement, not a full fix --
    the residual remains open (still true as of 2026-08-02 -- re-checked
    on the SAME real appstore judge pairs at n_lab=15, where this
    estimand's analytic path already applied before AND after the
    2026-08-02 update below, so it's unaffected: 14.7%/12.7% analytic vs.
    15.7%/17.3% bootstrap, both well above nominal). Root cause of the
    original improvement: the analytic path replaces the percentile
    bootstrap (which needs n_lab >> 30 for this estimand to reliably
    converge) with a closed-form Hajek-projection variance / Student-t
    interval that doesn't need to approximate a sampling distribution from
    a small empirical resample. At n_lab=30/60 -- newly on the analytic
    path as of the 2026-08-02 update below, previously always bootstrap --
    the same real appstore pairs show Type-I close to nominal for BOTH
    backends with no systematic difference (5.0-8.0%, 300 reps each): the
    extreme-tie residual above is specific to small n_lab and does not
    reappear at n_lab=30/60 under the new dispatch.

    UPDATE (2026-08-02): the SEPARATE power-lift gap this docstring
    previously described as not closing under "this (or any other
    attempted) CI-construction change" WAS, in fact, closed by exactly
    that -- just not the specific change tried as of 2026-08-01,
    which only compared analytic-vs-bootstrap BELOW n_lab=30 (where
    analytic already ran). A follow-up head-to-head EXTENDING the analytic
    construction to n_lab=30/60/90/130 (same estimand held fixed, same
    drawn data, only the CI construction differing) found it beats the
    percentile bootstrap on power at every point checked, by a WIDENING
    margin as n_lab grows -- see :func:`_analytic_walsh_theta_correct`'s
    docstring for the full numbers. ``correct()``'s ``backend="auto"``
    dispatch (see :data:`_ANALYTIC_ALWAYS_PREFERRED`) now uses the
    analytic backend for this estimand at EVERY n_lab, not just below 30
    -- this is the estimand ``wilcoxon()`` uses by default, so this change
    applies to it directly. The gap has NOT necessarily vanished entirely
    (bootstrap and analytic weren't compared past n_lab=200, and the
    remaining gap to mean-based estimands like paired_t was not
    independently re-measured after this change), but it is substantially
    smaller than previously documented, and the mechanism previously
    called "not a fixable... inefficiency" turned out to be exactly that,
    once the fix was applied to the RIGHT n_lab range.

    Registered in :func:`correct`'s ``_fast_batch`` dispatch (via
    ``_walsh_theta_batch``) so this goes through the SAME vectorized-
    resampling fast path ``np.mean``/``np.median`` use, not the slow
    per-replicate Python loop arbitrary ``estimator_func`` callers
    otherwise fall into -- required for this to be practical at all at
    n_boot~2000 and realistic corpus sizes (arena/wmt_da up to ~1000-1300
    items): the naive O(n^2)-per-call version this replaces was
    empirically too slow to finish even a single :func:`correct` call in
    reasonable time at those sizes (confirmed: an early un-batched
    attempt didn't finish 500 reps in 5+ minutes at n~285)."""
    d = np.asarray(d)
    if len(d) == 0:
        return 0.0
    return _walsh_theta_row(d) - 0.5


def _walsh_theta_batch(arr: np.ndarray) -> np.ndarray:
    """Row-wise :func:`paired_walsh_midrank_theta` over a ``(m, n)``
    bootstrap-replicate array, for :func:`correct`'s ``_fast_batch``
    dispatch -- see that dict and ``paired_walsh_midrank_theta``'s
    docstring. Still a Python-level loop over the ``m`` replicates (no
    known way to vectorize ``_walsh_theta_row``'s per-row sort +
    searchsorted across rows without materializing an O(m * n^2)
    intermediate, which would be worse), but each row's O(n log n) call is
    fast enough that looping ``m`` ~2000-4000 times is practical -- unlike
    routing the O(n^2)-per-call version through the SLOW general
    per-replicate loop (which also re-does rng draws and array slicing in
    Python on every iteration, not just the estimator call).

    Returns the SHIFTED (-0.5) value, matching paired_walsh_midrank_theta
    exactly -- NOT the raw _walsh_theta_row value. correct() calls
    estimator_func/rectifier_func directly (already shifted) for the point
    estimate, but routes through this batch function (looked up by the
    SHIFTED function's own id()) for bootstrap replicates -- an unshifted
    return here would offset every bootstrap replicate by +0.5 relative to
    the point estimate, pushing the whole bootstrap distribution away from
    0 and driving Type-I error to ~100% (caught in testing 2026-07-26 --
    this off-by-one-half bug, not the underlying statistic, was the actual
    cause of that failure)."""
    return np.array([_walsh_theta_row(row) - 0.5 for row in arr])


def _walsh_theta_h1_components(d: np.ndarray) -> np.ndarray:
    """Per-item empirical Hajek-projection ("structural component") values
    for :func:`paired_walsh_midrank_theta`'s underlying pairwise kernel
    ``h(d_i,d_j) = 1{d_i+d_j>0} + 0.5*1{d_i+d_j=0}`` -- the SAME
    "structural components" construction DeLong et al. (1988) use to get
    the variance (and, for two correlated samples, covariance) of an AUC
    estimator, applied here to the analogous one-sample Walsh-average
    U-statistic instead of DeLong's two-sample Mann-Whitney U-statistic.

    ``h1_hat(d_i) = (1/n) * sum_j [1{d_i+d_j>0} + 0.5*1{d_i+d_j=0}]``, with
    ``j`` ranging over ALL ``n`` items (including ``j=i``) -- the plug-in
    estimate of the first-order Hajek projection ``h1(d) = E_D2[h(d, D2)]``,
    evaluated at each observed ``d_i``. This is a "leave-in" (not
    leave-one-out) projection: it differs from the textbook leave-one-out
    jackknife-style version by one O(1/n) term, which does not matter for a
    VARIANCE estimate (only the point estimate needs the sharper
    leave-one-out correction, and this function is never used for the
    point estimate -- :func:`paired_walsh_midrank_theta` itself, via
    :func:`_walsh_theta_row`, already handles that separately).

    Used by :func:`_walsh_theta_analytic_variance` (single-sample variance,
    ``Var(U) ~= (4/n)*Var(h1_hat)``, the standard degree-2 U-statistic
    asymptotic variance) and directly by
    :func:`_analytic_walsh_theta_correct` (paired covariance: for two
    structural-component arrays computed on the SAME n items,
    ``Var(A-B) ~= (4/n)*Var(h1_hat_A - h1_hat_B)``, which folds the
    variance-of-A, variance-of-B, and covariance(A,B) terms into one pass).

    Numerically validated (2026-08-01) against a many-fresh-draws Monte
    Carlo "oracle" sampling variance of ``paired_walsh_midrank_theta``
    itself, on the label-efficiency check's continuous good-judge scenario
    (``simulations/harness/scenarios/synthetic.py``'s
    ``build_ppi_label_efficiency_sources``, noise=0.0909 i.e. pearson
    r~0.80): the single-sample variance formula matched the oracle to
    within 7% at n_lab=15/30 and within 1% at n_lab=60; the paired
    (rectifier) covariance version matched the oracle rectifier variance
    to within 2-17% across the same n_lab grid -- close enough to trust as
    :func:`correct`'s analytic backend for this estimand, the role
    :func:`_analytic_mean_correct` already plays for ``np.mean``. Same
    O(n log n) sort + vectorized ``searchsorted`` construction as
    :func:`_walsh_theta_row`, just without that function's ``i<=j``
    upper-triangular restriction (every item is compared against ALL n
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

    Why this exists: ``paired_walsh_midrank_theta`` is not ``np.mean``, so
    it never qualified for :func:`correct`'s pre-2026-08-01 analytic
    backend, leaving ``wilcoxon()`` stuck on the percentile bootstrap at
    every n_lab -- unlike ``paired_t`` (``evalstats.tests.ttest``'s
    ``paired=True`` path), which gets the closed-form path automatically
    below ``n_lab=30``. This was root-caused (2026-08-01) as part of the
    label-efficiency check's finding that wilcoxon's PPI correction gives
    much less power lift than paired_t/ttest_welch/mwu at matched n_lab
    (see ``simulations/harness/cases/pvalues.py``'s
    ``run_ppi_label_efficiency_check`` and this module's own
    ``paired_walsh_midrank_theta`` docstring for the full numbers) --
    via a matched-random-draw comparison (the SAME
    ``generate_judge_bias_cell()`` draw fed to both estimators, so the
    comparison isn't confounded by different random samples): wilcoxon's
    percentile-bootstrap CI was ~2.6-2.8x wider than paired_t's at matched
    n_lab=30/60 (continuous, good judge), stable across n_boot=300-6000
    (ruling out Monte Carlo noise). Two follow-up checks then ruled out
    the bootstrap CONSTRUCTION as the cause: (a) the percentile bootstrap's
    OWN variance estimate matched a many-fresh-draws Monte Carlo "oracle"
    sampling variance closely for BOTH estimators (ratio 0.91-1.05 for
    wilcoxon's two PPI terms, 0.77-1.02 for paired_t's, at n_lab=30) --
    i.e. the bootstrap is not itself inflating variance beyond the
    estimator's true sampling variability; (b) a normal approximation
    built from the SAME bootstrap replicates gave essentially the same CI
    width as the percentile interval for both estimators (ratio
    0.995-1.028, bootstrap-distribution skew ~0) -- i.e. no meaningful
    skew-driven percentile-construction penalty either. So the wider CI
    reflects the Walsh-average rectifier's genuinely higher finite-sample
    variance on its own [-0.5, 0.5] scale, not a fixable resampling
    inefficiency -- this analytic replacement does NOT close that gap (see
    ``wilcoxon()``'s docstring for the measured before/after power
    numbers). Its value is narrower in scope: it (a) unlocks the SAME
    small-n_lab (<30) fast, better-calibrated path ``np.mean`` already
    has -- :func:`correct`'s percentile bootstrap is documented to need
    n_lab >~ 50 for good Type-I calibration on real judge-pair data, while
    an analytic path reaches that around n_lab~25-30 (see
    :func:`_analytic_mean_correct`'s docstring) -- and (b) removes the
    percentile bootstrap's own residual Monte Carlo noise, which was
    already shown to be small for this estimand (CI width stable across
    n_boot=300-6000, see above) but is not literally zero at finite
    n_boot.

    Degrees of freedom for the Student-t interval: ``n_lab - 1``, matching
    :func:`_analytic_mean_correct`'s choice (the labeled term is the
    variance bottleneck there too, for the same reason: n_all is typically
    much larger than n_lab, so its analytic variance contributes a much
    smaller share of the total variance).

    UPDATE (2026-08-02): the "does NOT close that gap" claim two
    paragraphs up was accurate for what was actually tested at the time
    (analytic vs. bootstrap ONLY below n_lab=30, where this backend
    already activated) but incomplete as a claim about CI construction in
    general -- it never tested what happens if the analytic construction
    is extended PAST n_lab=30, because :func:`correct`'s dispatch didn't
    allow that combination to run under "auto" then. A follow-up
    investigation did exactly that: called this function directly (via
    ``backend="analytic"``, bypassing the n_lab<30 gate) at n_lab=30/60/
    90/130, on the SAME estimand (``paired_walsh_midrank_theta``) and the
    SAME per-replicate drawn data as the percentile-bootstrap path, so
    only the CI construction differs -- an apples-to-apples isolation the
    2026-08-01 investigation didn't have.

    Power (continuous/likert good-judge synthetic data, 200 reps,
    n_boot=1500 for the bootstrap arm; analytic has no bootstrap and is
    exact given its inputs):

    ==========  ========  =========  ========  =======
    eval_type   n_lab     bootstrap  analytic  delta
    ==========  ========  =========  ========  =======
    continuous  30        0.390      0.495     +0.105
    continuous  60        0.505      0.780     +0.275
    continuous  90        0.415      0.955     +0.540
    continuous  130       0.150      0.980     +0.830
    likert      30        0.370      0.420     +0.050
    likert      60        0.525      0.770     +0.245
    likert      90        0.465      0.910     +0.445
    likert      130       0.245      0.975     +0.730
    ==========  ========  =========  ========  =======

    Analytic wins at every point, by a WIDENING margin -- the opposite of
    what "the gap doesn't close" implied. More strikingly, bootstrap's OWN
    power is non-monotonic: it peaks around n_lab=60 and falls back off by
    n_lab=130, while analytic's power climbs smoothly and monotonically to
    near-saturating. Type-I calibration was checked at the same 16 cells
    (effect_size=0) and was statistically indistinguishable between the
    two backends everywhere (0.010-0.065 vs. nominal 0.05, both backends,
    ordinary Monte Carlo noise at 200 reps) -- no calibration cost for the
    power gain. The oracle-variance validation
    :func:`_walsh_theta_h1_components` documents (originally only checked
    at n_lab=15/30/60) was separately extended to n_lab=90/130/200 (3000-
    draw Monte Carlo oracle, continuous/likert good-judge): single-sample
    variance ratio (analytic estimate / oracle) 0.936-1.002, paired-
    rectifier covariance-based variance ratio 0.965-1.033 -- TIGHTER than
    the original 7%/1% (single-sample) and 2-17% (paired) figures at
    n_lab=15-30/60, as expected since the Hajek-projection approximation
    is asymptotic and improves with n. Real appstore extreme-tie data
    (n_lab=30/60, two judge pairs, 300 reps each) showed no systematic
    Type-I difference between backends either (5.0-8.0%, both close to
    nominal) -- the OPEN small-n_lab (~15) extreme-tie residual documented
    above is unaffected by this change, since that residual was already on
    the analytic path before it.

    Given this, :data:`_ANALYTIC_ALWAYS_PREFERRED` now routes
    ``paired_walsh_midrank_theta`` through this function at EVERY n_lab
    under ``backend="auto"``, not just below 30 -- see that constant's
    docstring. ``np.mean``'s own n_lab<30 gate is UNCHANGED; nothing here
    revisits that estimator's separately-validated threshold.

    Working hypothesis (UNCONFIRMED) for WHY bootstrap's power specifically
    degrades at larger n_lab for this estimand: :func:`correct`'s
    ``power_tune`` shrinkage (``_POWER_TUNE_SHRINKAGE_C``) pulls the
    bootstrap-estimated lambda back toward 1 by an amount that WEAKENS as
    n_lab grows (``1 - (1-lam)*n_lab/(n_lab+20)``) -- by n_lab=90-130, the
    raw bootstrap lambda* (itself estimated from an EXTRA layer of
    resampling, per correct()'s power_tune docstring) is applied nearly
    unshrunk. If that raw lambda* is a noisier estimate for this estimand
    specifically than for e.g. a mean (plausible given the Walsh-average
    rectifier's own higher finite-sample variance, documented above), its
    noise could be actively hurting power once shrinkage stops masking it
    -- while this function's own lambda* is closed-form (no resampling
    noise in the lambda estimate itself), so it wouldn't inherit the same
    problem. This is a plausible story consistent with the shape of the
    numbers above (bootstrap peaks near n_lab=60, where shrinkage is still
    partially protective, and falls off past it) but was NOT directly
    verified (e.g. by comparing bootstrap power_tune=True vs. power_tune=
    False at these n_lab, or by inspecting the raw lambda* estimates'
    own variance) -- flagged here as a lead for future investigation, not
    an established mechanism."""
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
    :func:`_analytic_logit_t_correct` can reuse the IDENTICAL point-
    estimate/variance derivation (only the CI construction differs: a
    plain t-interval on the raw scale in ``_analytic_mean_correct`` vs. a
    delta-method logit-scale transform in ``_analytic_logit_t_correct``)
    without duplicating or re-deriving the variance algebra. See
    ``_analytic_mean_correct``'s docstring for the closed-form lambda*/
    variance derivation this implements, unchanged from before this was
    split out (verified via a fixed-seed before/after comparison).

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
        # (_POWER_TUNE_SHRINKAGE_C) -- NOT just to compensate for bootstrap's
        # own small-sample weakness (this path doesn't have that one), but
        # because a raw lambda* computed from only n_lab points is ITSELF a
        # noisy estimate of the true population lambda, regardless of whether
        # it's computed via resampling or directly from sample statistics.
        # Without this, a raw lambda*=0 (e.g. when Y_lab happens to have ~0
        # sample variance at small n_lab) collapses the estimate to f_lab with
        # se=0 exactly -- a real bug hit during testing (2026-07-23): a
        # same-inputs comparison against backend="bootstrap" showed the
        # bootstrap path did NOT degenerate on the identical data, entirely
        # because of this shrinkage step, which this path had omitted.
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
    """Closed-form (delta-method) PPI correction for estimator_func=np.mean --
    no bootstrap resampling at all. See correct()'s ``backend`` parameter
    for when this replaces the percentile bootstrap.

    Root-caused via real judge-pair data (2026-07-23): the percentile
    bootstrap needs roughly n_lab >= 50 on noisy/discrete real data before
    Type-I error settles near nominal alpha; this closed-form path reaches
    the same target by n_lab ~= 25-30, since it doesn't need to
    approximate a sampling distribution from a small empirical resample --
    it plugs sample variances directly into a known distributional form
    (Student's t, df = n_lab - 1, since the labeled-subset term is the
    bottleneck).

    Point estimate is identical to the bootstrap path's (same f_unlab/
    f_lab/f_hat_lab/rectifier definitions); only the variance/CI/p-value
    construction differs. That shared point-estimate/variance/df
    computation now lives in :func:`_analytic_mean_point_se` (this
    function is a thin wrapper around it, building a plain t-interval);
    see that function's own docstring for why it was split out.

    power_tune's lambda* has a closed form here too (the ORIGINAL PPI++
    derivation for a mean/OLS-type estimand -- Angelopoulos/Duchi/Zrnic
    2023's own result, not an approximation of it): minimizing
    Var(F_lab + lambda*(F_unlab - F_hat_lab)) over lambda, where F_unlab is
    independent of (F_lab, F_hat_lab) by the disjointness requirement,
    gives

        lambda* = Cov(F_lab, F_hat_lab) / [Var(F_unlab) + Var(F_hat_lab)]

    with F_lab/F_hat_lab/F_unlab the SAMPLE MEANS (not raw items) -- i.e.
    Var(F_hat_lab) = var(Y_hat_lab, ddof=1)/n_lab, Cov(F_lab, F_hat_lab) =
    cov(Y_lab, Y_hat_lab, ddof=1)/n_lab (paired at the item level, cross-
    item covariance is 0 under i.i.d. sampling), Var(F_unlab) =
    var(Y_hat_unlab, ddof=1)/n_all. correct()'s bootstrap path estimates
    this SAME quantity by resampling, as a general-purpose stand-in that
    works for arbitrary estimator_func; for the mean specifically this
    closed form is exact, not an approximation, and needs no extra
    bootstrap pass to get it. No small-n_lab shrinkage is applied (unlike
    the bootstrap path's _POWER_TUNE_SHRINKAGE_C) -- that shrinkage exists
    specifically to compensate for the bootstrap's own small-sample
    weakness, which this path doesn't have.
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
    evalstats.core.resampling.logit_t_ci_1d. Built the same way
    _analytic_mean_correct is: IDENTICAL point-estimate/variance
    derivation (delegated to _analytic_mean_point_se, shared with
    _analytic_mean_correct -- the two differ ONLY in how the CI is built
    from (estimate, se, df)), then a delta-method logit-scale t-interval
    instead of a plain one:

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

    p_value uses estimate/se on the RAW scale -- numerically identical to
    what _analytic_mean_correct would report on the same inputs. This
    matches an established project convention, confirmed by direct read
    of evalstats/core/paired.py's classical (non-PPI) method="logit_t"
    branch (lines ~764-799 as of 2026-08-05): it ALSO returns the plain
    paired-t-test p-value regardless of the logit CI construction -- only
    the CI's shape differs, never the significance test itself.
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
analytic corrector, for :func:`correct`'s ``backend`` dispatch. Added
2026-08-01 -- previously :func:`correct` special-cased ``estimator_func is
np.mean`` directly; generalized to a registry so
:func:`_analytic_walsh_theta_correct` (the Walsh-average midrank-sign
analogue for ``paired_walsh_midrank_theta``, i.e. ``wilcoxon()``'s
default estimand) could be added WITHOUT wilcoxon needing its own
parallel dispatch path -- both now go through the exact same
``backend="auto"`` logic np.mean already used. Extending this to a new
estimand requires BOTH a corrector function matching
``_analytic_mean_correct``'s signature (``Y_lab, Y_hat_lab, Y_hat_unlab,
alpha, power_tune -> PPIResult``) AND an entry here -- ``estimator_func``
and ``rectifier_func`` must resolve to the SAME entry (matched, exactly
like the pre-existing ``np.mean`` requirement), since the two-term
variance decomposition assumes the rectifier and the main estimand share
one variance/covariance model."""

_ANALYTIC_ALWAYS_PREFERRED = {id(paired_walsh_midrank_theta)}
"""``estimator_func`` identities that :func:`correct`'s ``backend="auto"``
should route to the analytic backend at EVERY ``n_lab`` -- not only below
``_MIN_LAB_RECOMMENDED``, the threshold ``np.mean``'s entry in
:data:`_ANALYTIC_BACKENDS` still uses unchanged. Added 2026-08-02 for
``paired_walsh_midrank_theta`` specifically, after a head-to-head
investigation (same estimand, same drawn data, only the CI construction
differing) found the analytic path beats the percentile bootstrap on
POWER at every ``n_lab`` checked from 30 up through 200 -- not just
matching it, the gap WIDENS as ``n_lab`` grows (e.g. continuous good-judge
power lift over the classical labels-only baseline: bootstrap peaked
around n_lab=60 then fell back off, while analytic climbed monotonically
to near-saturating power by n_lab=130-200) -- while Type-I calibration
stayed statistically indistinguishable between the two backends at every
n_lab tested (200-rep precision, both hovering near nominal 5% with
ordinary Monte Carlo noise, no systematic degradation either direction).
The oracle-variance validation :func:`_walsh_theta_h1_components` already
documents (originally only checked at n_lab=15/30/60) was extended to
n_lab=90/130/200 as part of the same pass and, if anything, got TIGHTER
(within ~1-3% of a 3000-draw Monte Carlo oracle at n_lab>=90, vs. the
original 7%/1% figures at n_lab=15-30/60) -- expected, since the
underlying Hajek-projection approximation is asymptotic and only
improves with n. See :func:`_analytic_walsh_theta_correct`'s docstring
for the full numbers and the (unconfirmed) working hypothesis for WHY the
bootstrap's own power degrades at larger n_lab specifically for this
estimand. ``np.mean`` is deliberately NOT added here: its own n_lab<30
gate was validated independently on its own terms (see
:func:`_analytic_mean_correct`'s docstring) and nothing in this
investigation touched that evidence -- this set exists specifically so
one estimand's preference doesn't have to be smuggled into, or force a
review of, the shared ``_MIN_LAB_RECOMMENDED`` threshold."""


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
        generalizing the SAME variance-minimization argument rather than a
        different one). A SECOND, independent bootstrap draw then builds
        the percentile CI at that fixed λ̂ -- estimating λ̂ from the SAME
        draw used for the CI was tried first and measurably undercovered
        (confirmed via Monte Carlo: nominal 95% landing at ~91-94% across a
        range of judge-quality regimes), since λ̂ ends up partly optimized
        against noise specific to that one draw ("double dipping"); the
        split removes that circularity at the cost of one extra bootstrap
        pass. λ̂ is clipped to ``[0, 1]``, then SHRUNK back toward 1 by an
        n_lab-dependent amount (see ``_POWER_TUNE_SHRINKAGE_C`` and the
        inline comment where it's applied) -- a small n_lab makes the raw
        λ̂ estimate itself unreliable AND exposes a separate, pre-existing
        small-sample weakness of the percentile bootstrap this shrinkage
        compensates for; this is an empirical patch, not part of the
        published PPI++ derivation. Un-shrunk λ̂=0 falls back to the
        classical labels-only estimate (``human_estimate``) when the LLM
        adds no value; λ=1 reproduces ``power_tune=False``'s estimate when
        the LLM is fully informative or n_lab is small. Falls back to λ=1
        (unchanged behavior) if the bootstrap variance in the denominator
        is degenerate (≈0). ``PPIResult.lam`` reports the (shrunk) value
        actually used. Became the default on 2026-07-23 after validating
        against the harness's full judge-bias-scenario catalog (139
        scenarios x ttest_welch/mwu/wilcoxon/paired_t): Type-I error with
        power_tune=True matches power_tune=False's own baseline calibration
        (within ~0.1-0.4 percentage points, both close to nominal), while
        CI width is consistently narrower -- see ``simulations/harness`` on
        the ``explore-ppi-plus-plus`` branch for the full validation.
        kruskal/anova/friedman/bootstrap_t/tango_score/lmm* and the MNAR-
        experimental rectifiers do NOT go through this function (bespoke
        bootstrap/closed-form code of their own) and are UNAFFECTED by this
        default -- they remain on their original, non-power-tuned
        estimators. For kruskal/anova/friedman specifically this is
        deliberate, not just "not yet done": a real attempt found power-
        tuning fundamentally does not transfer to their variance-like,
        quadratic-form estimand (their weight=0 endpoint is the RAW,
        judge-biased estimate, not a safe classical fallback the way a
        scalar mean's is -- see ``simulations/harness/README.md``'s "PPI++
        power-tuning" bullet for the full finding and the ~19% vs ~4%
        Type-I inflation this produced when tried).
    backend : {"auto", "bootstrap", "analytic"}
        How to build the CI/p-value. "bootstrap" is the percentile-
        resampling method described above, unconditionally. "analytic" is
        a closed-form (delta-method / Hajek-projection, depending on the
        estimand) alternative for ``estimator_func`` registered in
        :data:`_ANALYTIC_BACKENDS` (``np.mean`` -- see
        :func:`_analytic_mean_correct` -- and, as of 2026-08-01,
        :func:`paired_walsh_midrank_theta` -- see
        :func:`_analytic_walsh_theta_correct`) with no covariates
        (``X_lab``/``X_unlab`` both None) and ``rectifier_func`` matching
        ``estimator_func`` (or ``None``); raises ``ValueError`` if
        requested for anything else. "auto" (the default) uses "analytic"
        when it's applicable AND EITHER ``n_lab < 30`` (below which the
        percentile bootstrap is known to undercover -- see the
        power_tune docstring above) OR ``estimator_func`` is in
        :data:`_ANALYTIC_ALWAYS_PREFERRED` (currently just
        ``paired_walsh_midrank_theta`` -- see below), otherwise falls back
        to "bootstrap" and, if ``n_lab < 30`` there too (i.e. analytic
        wasn't applicable for this estimator_func), emits a
        ``UserWarning`` instead of silently returning an under-covering
        interval. Root-caused via real judge-pair data (2026-07-23): on
        noisy/discrete real data, the bootstrap needed n_lab >~ 50 before
        Type-I error settled near nominal alpha, while the analytic path
        reached the same target by n_lab ~= 25-30 -- it doesn't need to
        approximate a sampling distribution from a small empirical
        resample, since it plugs sample variances directly into a known
        (Student's t) distributional form instead. ``np.mean`` keeps
        exactly this n_lab<30 gate, unchanged -- its own threshold was
        validated on its own terms and this investigation didn't revisit
        that evidence.

        ``paired_walsh_midrank_theta`` is different: as of 2026-08-02 it
        uses "analytic" at EVERY n_lab, not just below 30. Earlier
        (2026-08-01) this docstring stated that "analytic" narrowed
        wilcoxon's CI ONLY at small n_lab and did NOT close a ~2.6-2.8x
        wider-CI gap vs. ``np.mean`` at n_lab>=30 -- that finding compared
        analytic against bootstrap ONLY below n_lab=30 (where analytic
        already activated) and never checked whether extending analytic
        PAST 30 would behave differently; it hadn't been tested. It has
        now: a head-to-head at n_lab=30/60/90/130 (same estimand, same
        drawn data, only the CI construction differing -- continuous/
        likert good-judge synthetic data, 200 reps, n_boot=1500) found
        analytic beats bootstrap on POWER at every point, by a WIDENING
        margin (e.g. continuous: lift over bootstrap +0.105/+0.275/+0.540/
        +0.830 at n_lab=30/60/90/130; likert: +0.050/+0.245/+0.445/+0.730)
        -- bootstrap's own power is in fact non-monotonic, peaking around
        n_lab=60 and falling back off by n_lab=130 (continuous:
        0.390->0.505->0.415->0.150; likert: 0.370->0.525->0.465->0.245),
        while analytic's power climbs smoothly and monotonically
        (continuous: 0.495->0.780->0.955->0.980). Type-I calibration was
        statistically indistinguishable between the two backends at every
        point checked (16 cells, all within ordinary Monte Carlo noise of
        nominal 5%, no systematic degradation either direction). The
        earlier "not a fixable bootstrap-construction inefficiency"
        framing is therefore SUPERSEDED for the power question
        specifically: it was accurate as a description of "does switching
        CONSTRUCTION at the SAME n_lab<30 threshold change n_lab>=30
        behavior" (no, because analytic never ran there before), but
        wrong as a claim that no CI-construction change could help --
        extending WHERE the analytic construction applies does help,
        substantially. See :func:`_analytic_walsh_theta_correct`'s
        docstring for the full numbers, the oracle-variance re-validation
        extended to n_lab=90/130/200, and an unconfirmed working
        hypothesis for WHY bootstrap's power specifically degrades at
        larger n_lab for this estimand.

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
    # unlabeled replicate is drawn from an INDEPENDENT resample of the
    # (disjoint) unlabeled set, Cov(b_lab, b_unlab) = 0 in the bootstrap
    # distribution, which reduces the minimizer to a single covariance/
    # variance ratio -- see correct()'s power_tune parameter docstring for
    # the full derivation.
    #
    # TWO INDEPENDENT bootstrap draws are used when power_tune=True: one
    # (b1_*) ONLY to estimate lambda, a second, FRESH one (b2_*) ONLY to
    # build the percentile CI at that now-fixed lambda. Estimating lambda
    # and building its CI from the SAME replicates (one draw, used twice)
    # was tried first and measurably undercovered relative to nominal (95%
    # target landing at ~91-94% across a range of judge-quality regimes,
    # confirmed via Monte Carlo, worse than power_tune=False's own ~92-96%
    # baseline at the same small-n_lab settings) -- lambda was implicitly
    # optimizing away noise specific to that one bootstrap draw ("double
    # dipping"), producing a CI narrower than the estimator's true sampling
    # variability. Splitting the two draws removes that circularity at the
    # cost of one extra bootstrap pass (still cheap via the fast-batch path
    # above) and measurably closed most of the gap in the same check.
    # power_tune=False needs only ONE draw (unchanged from before this
    # function supported power_tune at all).
    #
    # lambda is then SHRUNK back toward 1 by an n_lab-dependent amount
    # (lam_reg = 1 - (1-lam)*n_lab/(n_lab+_POWER_TUNE_SHRINKAGE_C)) before
    # being applied. Without this, a Monte Carlo sweep across the harness's
    # full judge-bias-scenario catalog (139 scenarios, see cases/pvalues.py
    # on the explore-ppi-plus-plus branch) found power_tune=True Type-I
    # error MEASURABLY WORSE than power_tune=False's own baseline -- worst
    # for paired estimands (paired_t/wilcoxon), ~7-8% vs ~5-6.5% at nominal
    # 5%. Root cause (confirmed via matched-seed comparison against a plain
    # classical parametric test on the SAME labeled-only data): the
    # percentile bootstrap CI of a SMALL sample is itself mildly anti-
    # conservative (a well-known limitation -- see this module's "PPI
    # bootstrap can undercover below 30 labels" warning elsewhere in this
    # codebase), and vanilla PPI's fixed λ=1 masks that by always blending
    # in the large, well-behaved unlabeled-sample bootstrap. Power-tuning,
    # by correctly identifying an uninformative judge and shrinking λ
    # toward 0, leans MORE on that same small-sample bootstrap -- which
    # unmasks its pre-existing weakness rather than introducing a new one.
    # Shrinking lambda itself back toward 1 as n_lab shrinks (a small-
    # n_lab, low-confidence lambda estimate defers to vanilla PPI's
    # already-tolerable baseline) closed most of this gap in the same
    # Monte Carlo check (confirmed at n_lab=20: raw ~8.9% -> regularized
    # ~7.0%, matching vanilla's own ~7.1% baseline at that n_lab). This is
    # NOT part of the published PPI++ derivation -- it's an empirical
    # patch for a bootstrap-construction limitation this codebase already
    # had, layered on top.
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
