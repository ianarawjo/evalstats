"""evalstats.ppi — Prediction-Powered Inference (PPI) for arbitrary estimators.

Access via ``import evalstats as es; es.ppi.correct(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


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

    # ── Point estimate ────────────────────────────────────────────────────────
    f_unlab   = _call(estimator_func, Y_hat_unlab, X_unlab)
    f_lab     = _call(_rect_fn,       Y_lab,       X_lab)
    f_hat_lab = _call(_rect_fn,       Y_hat_lab,   X_lab)

    estimate  = f_unlab + (f_lab - f_hat_lab)
    rectifier = f_lab - f_hat_lab

    # ── Bootstrap CI ──────────────────────────────────────────────────────────
    # Fast path: when there are no covariates and both functions are one of
    # the built-ins actually used by this codebase's internal PPI dispatch
    # (np.mean / np.median), the whole bootstrap batches over an added
    # replicate axis instead of a Python loop with n_boot scalar calls each —
    # this matters most for np.median, which re-sorts on every call and
    # otherwise dominates runtime (see _ppi_paired_arrays's wilcoxon/median
    # callers). Falls back to the general per-replicate loop for arbitrary
    # user-supplied estimator functions or when X_lab/X_unlab are provided.
    _fast_batch = {id(np.mean): lambda a: a.mean(axis=1), id(np.median): lambda a: np.median(a, axis=1)}
    fast_est = _fast_batch.get(id(estimator_func)) if X_unlab is None else None
    fast_rect = _fast_batch.get(id(_rect_fn)) if X_lab is None else None

    if fast_est is not None and fast_rect is not None:
        boots = np.empty(n_boot)
        chunk_size = max(1, min(n_boot, 4096, max(1, int(2_000_000 // max(n_all, n_lab, 1)))))
        start = 0
        while start < n_boot:
            stop = min(start + chunk_size, n_boot)
            m = stop - start
            idx_all = rng.integers(0, n_all, size=(m, n_all))
            idx_lab = rng.integers(0, n_lab, size=(m, n_lab))
            b_unlab   = fast_est(Y_hat_unlab[idx_all])
            b_lab     = fast_rect(Y_lab[idx_lab])
            b_hat_lab = fast_rect(Y_hat_lab[idx_lab])
            boots[start:stop] = b_unlab + (b_lab - b_hat_lab)
            start = stop
    else:
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx_all = rng.integers(0, n_all, n_all)
            idx_lab = rng.integers(0, n_lab, n_lab)

            Xa_b = X_unlab[idx_all] if X_unlab is not None else None
            Xl_b = X_lab[idx_lab]   if X_lab   is not None else None

            b_unlab   = _call(estimator_func, Y_hat_unlab[idx_all], Xa_b)
            b_lab     = _call(_rect_fn,       Y_lab[idx_lab],       Xl_b)
            b_hat_lab = _call(_rect_fn,       Y_hat_lab[idx_lab],   Xl_b)

            boots[b] = b_unlab + (b_lab - b_hat_lab)

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
    )
