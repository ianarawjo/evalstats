"""evalstats.ppi — Prediction-Powered Inference (PPI) for arbitrary estimators.

Access via ``import evalstats as es; es.ppi.correct(...)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.stats import t as _t_dist

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
    construction differs.

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
    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)

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
        a closed-form (delta-method) alternative for ``estimator_func is
        np.mean`` with no covariates (``X_lab``/``X_unlab`` both None) --
        see :func:`_analytic_mean_correct`; raises ``ValueError`` if
        requested for anything else. "auto" (the default) uses "analytic"
        when it's applicable AND ``n_lab < 30`` (below which the
        percentile bootstrap is known to undercover -- see the
        power_tune docstring above), otherwise falls back to "bootstrap"
        and, if ``n_lab < 30`` there too (i.e. analytic wasn't
        applicable for this estimator_func), emits a ``UserWarning``
        instead of silently returning an under-covering interval.
        Root-caused via real judge-pair data (2026-07-23): on noisy/
        discrete real data, the bootstrap needed n_lab >~ 50 before
        Type-I error settled near nominal alpha, while the analytic path
        reached the same target by n_lab ~= 25-30 -- it doesn't need to
        approximate a sampling distribution from a small empirical
        resample, since it plugs sample variances directly into a known
        (Student's t) distributional form instead.

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
    _analytic_available = (
        estimator_func is np.mean and _rect_fn is np.mean and X_lab is None and X_unlab is None
    )
    if backend == "analytic" and not _analytic_available:
        raise ValueError(
            "backend='analytic' requires estimator_func=np.mean (and rectifier_func=np.mean or "
            f"None) with no covariates; got estimator_func={estimator_func!r}, "
            f"rectifier_func={rectifier_func!r}, X_lab={'given' if X_lab is not None else None}."
        )
    use_analytic = backend == "analytic" or (
        backend == "auto" and _analytic_available and n_lab < _MIN_LAB_RECOMMENDED
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
        return _analytic_mean_correct(Y_lab, Y_hat_lab, Y_hat_unlab, alpha, power_tune)

    # ── Point estimate (lambda=1 terms; combined into `estimate` below) ──────
    f_unlab   = _call(estimator_func, Y_hat_unlab, X_unlab)
    f_lab     = _call(_rect_fn,       Y_lab,       X_lab)
    f_hat_lab = _call(_rect_fn,       Y_hat_lab,   X_lab)
    rectifier = f_lab - f_hat_lab

    # ── Bootstrap replicates ──────────────────────────────────────────────────
    # Fast path: when there are no covariates and both functions are one of
    # the built-ins actually used by this codebase's internal PPI dispatch
    # (np.mean / np.median), the whole bootstrap batches over an added
    # replicate axis instead of a Python loop with n_boot scalar calls each —
    # this matters most for np.median, which re-sorts on every call and
    # otherwise dominates runtime (see _ppi_paired_arrays's wilcoxon/median
    # callers). Falls back to the general per-replicate loop for arbitrary
    # user-supplied estimator functions or when X_lab/X_unlab are provided.
    #
    # Factored into a helper (drawing ONE full set of n_boot replicates per
    # call) because power_tune needs TWO independent draws -- see below.
    _fast_batch = {id(np.mean): lambda a: a.mean(axis=1), id(np.median): lambda a: np.median(a, axis=1)}
    fast_est = _fast_batch.get(id(estimator_func)) if X_unlab is None else None
    fast_rect = _fast_batch.get(id(_rect_fn)) if X_lab is None else None

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
                b_unlab_arr[start:stop]   = fast_est(Y_hat_unlab[idx_all])
                b_lab_arr[start:stop]     = fast_rect(Y_lab[idx_lab])
                b_hat_lab_arr[start:stop] = fast_rect(Y_hat_lab[idx_lab])
                start = stop
        else:
            for b in range(n_boot):
                idx_all = rng.integers(0, n_all, n_all)
                idx_lab = rng.integers(0, n_lab, n_lab)
                Xa_b = X_unlab[idx_all] if X_unlab is not None else None
                Xl_b = X_lab[idx_lab]   if X_lab   is not None else None
                b_unlab_arr[b]   = _call(estimator_func, Y_hat_unlab[idx_all], Xa_b)
                b_lab_arr[b]     = _call(_rect_fn,       Y_lab[idx_lab],       Xl_b)
                b_hat_lab_arr[b] = _call(_rect_fn,       Y_hat_lab[idx_lab],   Xl_b)
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
