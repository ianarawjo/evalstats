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
    and recomputing the PPI estimator.

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
        LLM scores for the full (unlabeled) dataset.
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

    Returns
    -------
    PPIResult

    Raises
    ------
    ValueError
        If array lengths are inconsistent, or if exactly one of *X_lab* /
        *X_unlab* is supplied.

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

    # ── Coerce inputs ─────────────────────────────────────────────────────────
    Y_lab       = np.asarray(Y_lab,       dtype=float)
    Y_hat_lab   = np.asarray(Y_hat_lab,   dtype=float)
    Y_hat_unlab = np.asarray(Y_hat_unlab, dtype=float)

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

    n_lab = len(Y_lab)
    n_all = len(Y_hat_unlab)

    # ── Point estimate ────────────────────────────────────────────────────────
    f_unlab   = _call(estimator_func, Y_hat_unlab, X_unlab)
    f_lab     = _call(estimator_func, Y_lab,       X_lab)
    f_hat_lab = _call(estimator_func, Y_hat_lab,   X_lab)

    estimate  = f_unlab + (f_lab - f_hat_lab)
    rectifier = f_lab - f_hat_lab

    # ── Bootstrap CI ──────────────────────────────────────────────────────────
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx_all = rng.integers(0, n_all, n_all)
        idx_lab = rng.integers(0, n_lab, n_lab)

        Xa_b = X_unlab[idx_all] if X_unlab is not None else None
        Xl_b = X_lab[idx_lab]   if X_lab   is not None else None

        b_unlab   = _call(estimator_func, Y_hat_unlab[idx_all], Xa_b)
        b_lab     = _call(estimator_func, Y_lab[idx_lab],       Xl_b)
        b_hat_lab = _call(estimator_func, Y_hat_lab[idx_lab],   Xl_b)

        boots[b] = b_unlab + (b_lab - b_hat_lab)

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))

    # ── p-value ───────────────────────────────────────────────────────────────
    p_value: Optional[float] = None
    if compute_pvalue:
        # Proportion of bootstrap draws on each side of 0; two-sided.
        p_value = float(2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0)))

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
