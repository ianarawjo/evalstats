"""Judge alignment validation and MC-based uncertainty propagation.

Provides :func:`validate_alignment` and :class:`AlignmentResult` for
characterising how well an LLM judge aligns with human graders, and for
propagating that uncertainty into downstream comparisons via Monte-Carlo
imputation of latent human labels.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, pearsonr, spearmanr


# ─────────────────────────────────────────────────────────────────────────────
# AlignmentResult
# ─────────────────────────────────────────────────────────────────────────────

class AlignmentResult:
    """Carries a fitted calibration model and alignment diagnostics.

    Created by :func:`validate_alignment`.  Pass it to
    ``compare(alignment={metric_col: result})`` to widen confidence intervals
    to account for LLM-judge measurement uncertainty via Monte-Carlo imputation.

    Attributes
    ----------
    llm_metric : str
        Column name of the LLM judge scores.
    human_col : str
        Column name of the human-label scores.
    score_type : str
        Detected score type: ``"binary"``, ``"likert"``, ``"continuous"``,
        or ``"grade"``.
    n_labeled : int
        Number of items with human labels (alignment set size).
    n_total : int
        Total number of items in the dataset.
    alignment_metrics : dict
        Point estimates and bootstrap CIs for each alignment metric.
    representativeness : dict
        Representativeness check results (distribution and slice columns).
    """

    def __init__(
        self,
        *,
        llm_metric: str,
        human_col: str,
        score_type: str,
        n_labeled: int,
        n_total: int,
        calibration: dict,
        alignment_metrics: dict,
        representativeness: dict,
    ) -> None:
        self.llm_metric = llm_metric
        self.human_col = human_col
        self.score_type = score_type
        self.n_labeled = n_labeled
        self.n_total = n_total
        self._calibration = calibration
        self.alignment_metrics = alignment_metrics
        self.representativeness = representativeness

    # ── sampling ─────────────────────────────────────────────────────────────

    def _sample_imputed_scores(
        self,
        llm_scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample one realisation of latent human labels from the calibration posterior.

        Draws calibration parameters from their Bayesian posterior, then
        samples a human label for each item conditioned on its LLM score.
        Both sources of uncertainty (parameter uncertainty and item-level
        noise) are propagated.

        Parameters
        ----------
        llm_scores : np.ndarray
            1-D array of LLM judge scores for all items, in dataset row order.
        rng : np.random.Generator
            Random number generator (advanced each call).

        Returns
        -------
        np.ndarray
            Imputed human scores, same shape as ``llm_scores``.
        """
        cal = self._calibration
        n = len(llm_scores)
        imputed = np.empty(n, dtype=float)

        if cal["type"] == "binary":
            # Sample Bernoulli probability for each LLM class from Beta posterior
            p = {l: rng.beta(cal["alpha"][l], cal["beta"][l]) for l in cal["classes"]}
            fallback_p = float(np.mean(list(p.values())))
            for i, s in enumerate(llm_scores):
                prob = p.get(int(round(s)), fallback_p)
                imputed[i] = float(rng.binomial(1, prob))

        elif cal["type"] == "likert":
            # Sample category probabilities for each LLM class from Dirichlet posterior
            probs = {l: rng.dirichlet(cal["concentration"][l]) for l in cal["llm_cats"]}
            human_cats = np.array(cal["human_cats"])
            n_cats = len(human_cats)
            fallback = np.ones(n_cats) / n_cats
            for i, s in enumerate(llm_scores):
                cat_probs = probs.get(s, fallback)
                idx = rng.choice(n_cats, p=cat_probs)
                imputed[i] = float(human_cats[idx])

        else:  # "continuous" or "grade"
            # Sample (intercept, slope, σ²) from Normal-Inverse-Gamma posterior
            # σ² ~ InvGamma(an, bn); coefs | σ² ~ N(mun, σ² * Vn)
            sigma2 = 1.0 / rng.gamma(shape=cal["an"], scale=1.0 / cal["bn"])
            sigma2 = max(float(sigma2), 1e-10)
            cov = sigma2 * cal["Vn"] + np.eye(2) * 1e-12
            L = np.linalg.cholesky(cov)
            coefs = cal["mun"] + L @ rng.standard_normal(2)
            mu_pred = coefs[0] + coefs[1] * llm_scores
            imputed = mu_pred + np.sqrt(sigma2) * rng.standard_normal(n)

        return imputed

    # ── display ──────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a plain-language alignment and representativeness report."""
        width = 58
        print("Judge alignment report")
        print("─" * width)
        pct = 100.0 * self.n_labeled / self.n_total if self.n_total > 0 else 0.0
        print(
            f"Alignment set  : {self.n_labeled} of {self.n_total} items "
            f"have human labels ({pct:.1f}%)"
        )
        print()

        # Representativeness
        rep = self.representativeness
        print("Representativeness check:")
        dist = rep.get("score_distribution")
        if dist:
            icon = "✓" if dist["passed"] else "⚠ "
            print(f"  Score distribution  : {icon}  {dist['message']}")
        for key, val in rep.items():
            if key == "score_distribution":
                continue
            if key.startswith("slice_"):
                col = key[len("slice_"):]
                icon = "✓" if val["passed"] else "⚠ "
                print(f"  {col!r:<20}: {icon}  {val['message']}")
        print()

        # Alignment metrics
        print(f"Alignment metrics (score type: {self.score_type}):")
        for entry in self.alignment_metrics.values():
            label = entry.get("label", "")
            est = entry["estimate"]
            lo = entry["ci_low"]
            hi = entry["ci_high"]
            print(f"  {label:<24}: {est:.3f}  [{lo:.3f}, {hi:.3f}]")
        print("─" * width)

    def __repr__(self) -> str:
        return (
            f"AlignmentResult(metric={self.llm_metric!r}, "
            f"score_type={self.score_type!r}, "
            f"n_labeled={self.n_labeled}/{self.n_total})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Calibration model fitting
# ─────────────────────────────────────────────────────────────────────────────

def _fit_calibration(
    llm_labels: np.ndarray,
    human_labels: np.ndarray,
    score_type: str,
) -> dict:
    if score_type == "binary":
        return _fit_binary(llm_labels, human_labels)
    elif score_type == "likert":
        return _fit_likert(llm_labels, human_labels)
    else:
        return _fit_continuous(llm_labels, human_labels, score_type)


def _fit_binary(llm: np.ndarray, human: np.ndarray) -> dict:
    prior = 1.0  # Beta(1,1) uniform prior
    alpha_post: dict = {}
    beta_post: dict = {}
    for l in [0, 1]:
        mask = (llm == l)
        n_l = int(mask.sum())
        n_pos = int((human[mask] == 1).sum()) if n_l > 0 else 0
        alpha_post[l] = float(n_pos + prior)
        beta_post[l] = float((n_l - n_pos) + prior)
    return {"type": "binary", "classes": [0, 1], "alpha": alpha_post, "beta": beta_post}


def _fit_likert(llm: np.ndarray, human: np.ndarray) -> dict:
    llm_cats = sorted(set(llm.tolist()))
    human_cats = sorted(set(llm.tolist()) | set(human.tolist()))
    n_human = len(human_cats)
    prior_conc = 1.0 / n_human  # near-uniform Dirichlet
    concentration: dict = {}
    for l in llm_cats:
        mask = (llm == l)
        counts = np.array(
            [(human[mask] == k).sum() for k in human_cats], dtype=float
        )
        concentration[l] = counts + prior_conc
    return {
        "type": "likert",
        "llm_cats": llm_cats,
        "human_cats": human_cats,
        "concentration": concentration,
    }


def _fit_continuous(llm: np.ndarray, human: np.ndarray, score_type: str) -> dict:
    n = len(llm)
    X = np.column_stack([np.ones(n), llm])
    y = human.astype(float)

    # Uninformative Normal-Inverse-Gamma prior
    V0_inv = np.eye(2) * 1e-6
    mu0 = np.zeros(2)
    a0, b0 = 1.0, 1.0

    XtX = X.T @ X
    Xty = X.T @ y
    Vn_inv = V0_inv + XtX
    Vn = np.linalg.inv(Vn_inv)
    mun = Vn @ (V0_inv @ mu0 + Xty)
    an = a0 + n / 2.0
    bn = float(b0 + 0.5 * (y @ y + mu0 @ V0_inv @ mu0 - mun @ Vn_inv @ mun))
    bn = max(bn, 1e-10)
    return {"type": score_type, "Vn": Vn, "mun": mun, "an": an, "bn": bn}


# ─────────────────────────────────────────────────────────────────────────────
# Alignment metrics
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci_2(
    fn,
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n = len(a)
    obs = float(fn(a, b))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = fn(a[idx], b[idx])
    lo = float(np.percentile(boot, 100.0 * alpha / 2))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2)))
    return obs, lo, hi


def _compute_alignment_metrics(
    llm: np.ndarray,
    human: np.ndarray,
    score_type: str,
    *,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> dict:
    metrics: dict = {}

    if score_type == "binary":
        def agree(a, b):
            return float(np.mean(a == b))

        def kappa(a, b):
            p_o = float(np.mean(a == b))
            p_e = float(
                np.mean(a == 1) * np.mean(b == 1)
                + np.mean(a == 0) * np.mean(b == 0)
            )
            return (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0

        est, lo, hi = _bootstrap_ci_2(agree, llm, human, alpha=alpha, rng=rng)
        metrics["percent_agreement"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Percent agreement",
        }
        est, lo, hi = _bootstrap_ci_2(kappa, llm, human, alpha=alpha, rng=rng)
        metrics["cohens_kappa"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Cohen's κ",
        }

    elif score_type == "likert":
        cats = sorted(set(llm.tolist()) | set(human.tolist()))
        k = len(cats)
        cat_idx = {c: i for i, c in enumerate(cats)}
        ii = np.arange(k, dtype=float)
        wm = 1.0 - (ii[:, None] - ii[None, :])**2 / max((k - 1)**2, 1)

        def wk(a, b):
            n = len(a)
            p_o = sum(
                1.0 - (cat_idx[ai] - cat_idx[bi])**2 / max((k - 1)**2, 1)
                for ai, bi in zip(a, b)
            ) / n
            p_a = np.array([(a == c).mean() for c in cats])
            p_b = np.array([(b == c).mean() for c in cats])
            p_e = float((p_a[:, None] * p_b[None, :] * wm).sum())
            return (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0

        def sp(a, b):
            r, _ = spearmanr(a, b)
            return float(r)

        if k >= 2:
            est, lo, hi = _bootstrap_ci_2(wk, llm, human, alpha=alpha, rng=rng)
            metrics["weighted_kappa"] = {
                "estimate": est, "ci_low": lo, "ci_high": hi,
                "label": "Weighted Cohen's κ",
            }
        est, lo, hi = _bootstrap_ci_2(sp, llm, human, alpha=alpha, rng=rng)
        metrics["spearman_r"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Spearman r",
        }

    else:  # continuous / grade
        def pe(a, b):
            r, _ = pearsonr(a, b)
            return float(r)

        def sp(a, b):
            r, _ = spearmanr(a, b)
            return float(r)

        est, lo, hi = _bootstrap_ci_2(pe, llm, human, alpha=alpha, rng=rng)
        metrics["pearson_r"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Pearson r",
        }
        est, lo, hi = _bootstrap_ci_2(sp, llm, human, alpha=alpha, rng=rng)
        metrics["spearman_r"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Spearman r",
        }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Representativeness checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_score_distribution(
    all_scores: np.ndarray,
    labeled_scores: np.ndarray,
    score_type: str,
) -> dict:
    if score_type == "binary":
        labeled_0 = int((labeled_scores == 0).sum())
        labeled_1 = int((labeled_scores == 1).sum())
        all_0 = int((all_scores == 0).sum())
        all_1 = int((all_scores == 1).sum())
        unlabeled_0 = all_0 - labeled_0
        unlabeled_1 = all_1 - labeled_1
        if unlabeled_0 + unlabeled_1 == 0:
            return {"passed": True, "message": "all items are labeled", "p_value": None}
        contingency = [[labeled_0, labeled_1], [unlabeled_0, unlabeled_1]]
        try:
            _, p, _, _ = chi2_contingency(contingency)
            p = float(p)
        except ValueError:
            p = 1.0
        passed = p >= 0.05
        msg = f"χ² p={p:.3f}"
        if not passed:
            msg += " — labeled 0/1 distribution differs from unlabeled pool"
    else:
        if len(np.unique(labeled_scores)) < 2:
            return {"passed": True, "message": "insufficient labeled variation to test", "p_value": None}
        _, p = ks_2samp(labeled_scores, all_scores)
        p = float(p)
        passed = p >= 0.05
        msg = f"KS p={p:.3f}"
        if not passed:
            msg += " — labeled subset appears non-representative of full score range"
    return {"passed": passed, "message": msg, "p_value": p}


def _check_slice_column(
    df: pd.DataFrame,
    labeled_mask: pd.Series,
    col: str,
) -> dict:
    labeled = df.loc[labeled_mask, col].dropna()
    unlabeled = df.loc[~labeled_mask, col].dropna()
    if len(unlabeled) == 0:
        return {"passed": True, "message": "no unlabeled items", "p_value": None}
    cats = sorted(df[col].dropna().unique())
    lab_counts = [(labeled == c).sum() for c in cats]
    unlab_counts = [(unlabeled == c).sum() for c in cats]
    contingency = [lab_counts, unlab_counts]
    try:
        _, p, _, _ = chi2_contingency(contingency)
        p = float(p)
    except ValueError:
        p = 1.0
    passed = p >= 0.05
    msg = f"χ² p={p:.3f}"
    if not passed:
        msg += " — labeled subset is over/under-represented in some categories"
    return {"passed": passed, "message": msg, "p_value": p}


# ─────────────────────────────────────────────────────────────────────────────
# validate_alignment
# ─────────────────────────────────────────────────────────────────────────────

def validate_alignment(
    evaldata,
    *,
    llm_metric: str,
    human_groundtruth: str,
    alpha: float = 0.05,
) -> AlignmentResult:
    """Validate how well an LLM judge aligns with human graders.

    Designed for the common case where LLM judge scores exist for all items
    but human labels are available for only a subset (the alignment set).
    Fits a Bayesian calibration model that can later be used to propagate
    judge uncertainty into downstream comparisons via
    ``compare(alignment={metric: result})``.

    Parameters
    ----------
    evaldata : EvalResults
        Evaluation data from :func:`load_from`.  Must contain both
        ``llm_metric`` and ``human_groundtruth`` as columns.
    llm_metric : str
        Column name of the LLM judge scores.  Must be present for all rows.
    human_groundtruth : str
        Column name of the human rater scores.  Expected to be sparsely
        populated: non-null for the alignment subset, ``NaN`` elsewhere.
    alpha : float
        Significance level for alignment metric CIs.  Default ``0.05``.

    Returns
    -------
    AlignmentResult
    """
    df = evaldata._df

    if llm_metric not in df.columns:
        raise ValueError(
            f"llm_metric column '{llm_metric}' not found in evaldata. "
            f"Available columns: {list(df.columns)}"
        )
    if human_groundtruth not in df.columns:
        raise ValueError(
            f"human_groundtruth column '{human_groundtruth}' not found in evaldata. "
            f"Available columns: {list(df.columns)}"
        )

    labeled_mask = df[human_groundtruth].notna()
    n_labeled = int(labeled_mask.sum())
    n_total = len(df)

    if n_labeled == 0:
        raise ValueError(
            f"No rows have human labels in '{human_groundtruth}'. "
            "Ensure it is NaN for unlabeled items and non-NaN for the alignment subset."
        )

    if n_labeled < 30:
        warnings.warn(
            f"Only {n_labeled} items have human labels. "
            "Alignment estimates will be imprecise with fewer than ~30 labeled items; "
            "consider expanding the alignment set for reliable uncertainty propagation.",
            UserWarning,
            stacklevel=2,
        )

    # Resolve score type
    score_type = evaldata._score_types.get(llm_metric)
    if score_type is None:
        from evalstats.loader import _detect_score_type
        score_type = _detect_score_type(df[llm_metric].dropna())

    llm_aligned = df.loc[labeled_mask, llm_metric].to_numpy(dtype=float)
    human_aligned = df.loc[labeled_mask, human_groundtruth].to_numpy(dtype=float)
    all_llm = df[llm_metric].to_numpy(dtype=float)

    # Fit Bayesian calibration model
    calibration = _fit_calibration(llm_aligned, human_aligned, score_type)

    # Compute alignment metrics with bootstrap CIs
    rng = np.random.default_rng(42)
    alignment_metrics = _compute_alignment_metrics(
        llm_aligned, human_aligned, score_type, alpha=alpha, rng=rng
    )

    # Representativeness: score distribution
    rep: dict = {}
    dist_result = _check_score_distribution(all_llm, llm_aligned, score_type)
    rep["score_distribution"] = dist_result
    if not dist_result["passed"]:
        warnings.warn(
            f"Representativeness warning: the {n_labeled} labeled items appear to have "
            f"a different {llm_metric} distribution than the full item pool "
            f"({dist_result['message']}). "
            "Alignment uncertainty estimates may not generalise to all items. "
            "Consider sampling human labels more broadly across the score range.",
            UserWarning,
            stacklevel=2,
        )

    # Representativeness: categorical slice columns
    slice_cols = [
        c for c in df.columns
        if c not in {llm_metric, human_groundtruth}
        and pd.api.types.is_string_dtype(df[c])
        and 1 < df[c].nunique() <= 20
    ]
    for col in slice_cols:
        col_result = _check_slice_column(df, labeled_mask, col)
        rep[f"slice_{col}"] = col_result
        if not col_result["passed"]:
            warnings.warn(
                f"Representativeness warning for column '{col}': the labeled subset "
                f"appears unevenly distributed across categories "
                f"({col_result['message']}). "
                "Consider stratified sampling of human labels.",
                UserWarning,
                stacklevel=2,
            )

    return AlignmentResult(
        llm_metric=llm_metric,
        human_col=human_groundtruth,
        score_type=score_type,
        n_labeled=n_labeled,
        n_total=n_total,
        calibration=calibration,
        alignment_metrics=alignment_metrics,
        representativeness=rep,
    )
