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
    bias_check : dict or None
        For likert/continuous/grade score types, compares the correlation-type
        metric (weighted κ or Pearson r) against ICC(2,1) to flag whether the
        judge is systematically biased in absolute scale despite tracking
        human relative ordering.  ``None`` for binary score types, where ICC
        isn't computed.
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
        bias_check: Optional[dict] = None,
    ) -> None:
        self.llm_metric = llm_metric
        self.human_col = human_col
        self.score_type = score_type
        self.n_labeled = n_labeled
        self.n_total = n_total
        self._calibration = calibration
        self.alignment_metrics = alignment_metrics
        self.representativeness = representativeness
        self.bias_check = bias_check

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

    def summary(self, verbose: bool = False) -> None:
        """Print an alignment and representativeness report.

        Parameters
        ----------
        verbose : bool
            If ``False`` (default), print a short, plain-language report:
            one line per check/metric, with an explanation only where
            something looks off. Aimed at readers who don't need the
            statistical background spelled out every time.
            If ``True``, print the full report — every metric's definition,
            why it was chosen, how to interpret it, and citation-ready
            wording for a paper.
        """
        if verbose:
            self._summary_verbose()
        else:
            self._summary_simple()

    def _header(self) -> None:
        pct = 100.0 * self.n_labeled / self.n_total if self.n_total > 0 else 0.0
        print("Judge alignment report")
        print("─" * 58)
        print(
            f"Alignment set  : {self.n_labeled} of {self.n_total} items "
            f"have human labels ({pct:.1f}%)"
        )
        print()

    def _summary_simple(self) -> None:
        self._header()

        if self.bias_check is not None:
            bc = self.bias_check
            if not bc["passed"]:
                print(
                    f"⚠ Possible judge bias: {bc['corr_label']} = "
                    f"{bc['corr_estimate']:.2f} but ICC(2,1) = {bc['icc_estimate']:.2f} "
                    "— the judge ranks items like humans do, but its raw scores "
                    "look shifted or compressed relative to human scores."
                )
                print(
                    "  Treat raw judge scores with caution; consider "
                    "recalibrating (compare(alignment=...)) before using them "
                    "directly. Run .summary(verbose=True) for the full check."
                )
            else:
                print(
                    f"✓ No sign of judge bias: {bc['corr_label']} and ICC(2,1) "
                    "roughly agree."
                )
            print()

        rep = self.representativeness
        rep_failed = [
            (k, v) for k, v in rep.items() if not v["passed"]
        ]
        if rep_failed:
            print("⚠ Representativeness: the labeled sample may not be representative")
            for key, val in rep_failed:
                name = "score distribution" if key == "score_distribution" else key[len("slice_"):]
                print(f"    - {name}: {val['message']}")
        else:
            print("✓ Representativeness: labeled items look like the full item pool")
        print()

        score_type_note = _SCORE_TYPE_NOTES.get(
            self.score_type, f"score type detected as {self.score_type!r}"
        )
        print(f"Alignment metrics ({self.score_type} scores — {score_type_note}):")
        for entry in self.alignment_metrics.values():
            label = entry.get("label", "")
            est = entry["estimate"]
            lo = entry["ci_low"]
            hi = entry["ci_high"]
            band = entry.get("band")
            tail = f"  {band}" if band else ""
            print(f"  {label:<20} {est:6.2f}  [{lo:5.2f}, {hi:5.2f}]{tail}")
        print()
        print("Run .summary(verbose=True) for definitions, rationale, and")
        print("citation-ready wording for each check above.")
        print("─" * 58)

    def _summary_verbose(self) -> None:
        self._header()

        if self.bias_check is not None and not self.bias_check["passed"]:
            print(f"⚠ Judge bias flag: {self.bias_check['message']}")
            print("   (see 'Bias diagnostics' below for details)")
            print()

        # Representativeness
        rep = self.representativeness

        def _print_check(title: str, val: dict) -> None:
            icon = "✓" if val["passed"] else "⚠ "
            print(f"  {title}: {icon}  {val['message']}")
            what = val.get("what")
            why = val.get("why")
            interpretation = val.get("interpretation")
            if what:
                print(f"      -> What this checks: {what}")
            if why:
                print(f"      -> Why it was computed in this case: {why}")
            if interpretation:
                print(f"      -> How to interpret this result: {interpretation}")
            print()

        print("Representativeness diagnostics:")
        dist = rep.get("score_distribution")
        if dist:
            _print_check("Score distribution", dist)
        for key, val in rep.items():
            if key == "score_distribution":
                continue
            if key.startswith("slice_"):
                col = key[len("slice_"):]
                _print_check(f"{col!r}", val)

        # Alignment metrics
        score_type_note = _SCORE_TYPE_NOTES.get(
            self.score_type, f"score type detected as {self.score_type!r}"
        )
        print(f"Alignment metrics (score type: {self.score_type}):")
        print(f"  ({score_type_note})")
        print()
        for entry in self.alignment_metrics.values():
            label = entry.get("label", "")
            est = entry["estimate"]
            lo = entry["ci_low"]
            hi = entry["ci_high"]
            print(f"  {label:<24}: {est:.3f}  [{lo:.3f}, {hi:.3f}]")
            what = entry.get("what")
            why = entry.get("why")
            interpretation = entry.get("interpretation")
            example = entry.get("example")
            if what:
                print(f"      -> What this metric is: {what}")
            if why:
                print(f"      -> Why it was computed in this case: {why}")
            if interpretation:
                print(f"      -> How to interpret this result: {interpretation}")
            if example:
                print(f"      -> Example paper reporting: {example}")
            print()

        if self.bias_check is not None:
            print("Bias diagnostics:")
            _print_check("Judge scale bias (correlation vs. ICC)", self.bias_check)

        print("─" * 58)

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
    a0, b0 = 1.0, 1e-6

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

# Short justification for *why* a given score type gets the metric set it does —
# printed in AlignmentResult.summary() so users can cite/justify the choice.
_SCORE_TYPE_NOTES = {
    "binary": (
        "labels are 0/1, so metrics designed for nominal categorical data are used"
    ),
    "likert": (
        "labels are ordered categories, so metrics designed for ordinal data are used"
    ),
    "continuous": (
        "labels are on a continuous scale, so correlation metrics are used"
    ),
    "grade": (
        "labels are numeric grades, so correlation metrics are used"
    ),
}


def _interpret_kappa(est: float, lo: float, hi: float, n: int, label: str) -> tuple[str, str, str]:
    """Landis & Koch (1977) benchmarks for kappa-type statistics."""
    if est < 0:
        band = "poor"
    elif est <= 0.20:
        band = "slight"
    elif est <= 0.40:
        band = "fair"
    elif est <= 0.60:
        band = "moderate"
    elif est <= 0.80:
        band = "substantial"
    else:
        band = "almost perfect"
    band_phrase = f"{band} agreement"
    interpretation = f"{band} agreement (Landis & Koch, 1977 benchmarks)"
    example = (
        f'"{label} = {est:.2f}, 95% CI [{lo:.2f}, {hi:.2f}] (n={n}), indicating '
        f'{band} agreement between the LLM judge and human raters, per the Landis '
        f'& Koch (1977) benchmarks."'
    )
    return band_phrase, interpretation, example


def _interpret_corr(est: float, lo: float, hi: float, n: int, label: str) -> tuple[str, str, str]:
    """Cohen (1988) conventions for correlation-coefficient magnitude."""
    a = abs(est)
    if a < 0.10:
        band = "negligible"
    elif a < 0.30:
        band = "small"
    elif a < 0.50:
        band = "medium"
    else:
        band = "large"
    direction = "positive" if est >= 0 else "negative"
    band_phrase = f"{band} {direction} correlation"
    interpretation = f"{band_phrase} (Cohen, 1988 conventions)"
    example = (
        f'"{label} = {est:.2f}, 95% CI [{lo:.2f}, {hi:.2f}] (n={n}), a {band} '
        f'{direction} correlation between the LLM judge and human scores (Cohen, '
        f'1988 conventions)."'
    )
    return band_phrase, interpretation, example


def _interpret_icc(est: float, lo: float, hi: float, n: int, label: str) -> tuple[str, str, str]:
    """Koo & Li (2016) benchmarks for ICC magnitude."""
    if est < 0.50:
        band = "poor"
    elif est < 0.75:
        band = "moderate"
    elif est < 0.90:
        band = "good"
    else:
        band = "excellent"
    band_phrase = f"{band} absolute agreement"
    interpretation = f"{band_phrase} (Koo & Li, 2016 benchmarks)"
    example = (
        f'"{label} = {est:.2f}, 95% CI [{lo:.2f}, {hi:.2f}] (n={n}), indicating '
        f'{band} absolute agreement between the LLM judge and human raters, per '
        f'Koo & Li (2016) benchmarks."'
    )
    return band_phrase, interpretation, example


def _interpret_pct_agreement(est: float, lo: float, hi: float, n: int, label: str) -> tuple[Optional[str], str, str]:
    interpretation = (
        "no universally-agreed threshold exists for raw percent agreement — read it "
        "alongside Cohen's κ, since it does not correct for chance and can look high "
        "purely from imbalanced label classes"
    )
    example = (
        f'"the LLM judge matched human labels on {est * 100:.1f}% of items, 95% CI '
        f'[{lo * 100:.1f}%, {hi * 100:.1f}%] (n={n})."'
    )
    return None, interpretation, example


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


def _icc_21(a: np.ndarray, b: np.ndarray) -> float:
    """Shrout & Fleiss (1979) ICC(2,1): two-way random effects, single rater,
    absolute agreement, for exactly two raters (``a``, ``b``).

    Unlike Pearson/Spearman r or weighted kappa's category-index distance,
    this is sensitive to a systematic offset or scale mismatch between the
    two raters — it measures whether they land on the same absolute values,
    not just whether they move together.
    """
    n = len(a)
    data = np.column_stack([a, b]).astype(float)
    k = 2
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    df_row = max(n - 1, 1)
    SSR = k * np.sum((row_means - grand_mean) ** 2)
    SSC = n * np.sum((col_means - grand_mean) ** 2)  # (k-1) == 1
    SST = np.sum((data - grand_mean) ** 2)
    SSE = SST - SSR - SSC

    MSR = SSR / df_row
    MSC = SSC
    MSE = SSE / df_row  # (n-1)(k-1) == n-1

    denom = MSR + MSE + 2.0 * (MSC - MSE) / n
    if denom <= 1e-12:
        return 1.0
    return float((MSR - MSE) / denom)


def _bootstrap_ci_gap(
    fn_corr,
    fn_icc,
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for (correlation-type metric − ICC(2,1)).

    Resamples items once per draw and evaluates both statistics on the same
    resample, so the CI reflects the sampling distribution of the *gap*
    itself, not the (looser, more conservative) union of two independently
    bootstrapped CIs.
    """
    n = len(a)
    obs = float(fn_corr(a, b) - fn_icc(a, b))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = fn_corr(a[idx], b[idx]) - fn_icc(a[idx], b[idx])
    lo = float(np.percentile(boot, 100.0 * alpha / 2))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2)))
    return obs, lo, hi


def _build_bias_check(
    corr_label: str,
    corr_est: float,
    icc_est: float,
    gap_est: float,
    gap_lo: float,
    gap_hi: float,
) -> dict:
    """Package the correlation-vs-ICC(2,1) discrepancy check into a dict with
    the same shape as the representativeness checks, so it can be rendered
    with the same ``_print_check`` helper in ``AlignmentResult.summary()``.
    """
    flagged = gap_lo > 0.0
    if flagged:
        message = (
            f"possible judge bias: {corr_label} = {corr_est:.3f} but "
            f"ICC(2,1) = {icc_est:.3f} (gap = {gap_est:.3f}, 95% CI "
            f"[{gap_lo:.3f}, {gap_hi:.3f}], excludes 0)"
        )
    else:
        message = (
            f"no evidence of scale bias: {corr_label} = {corr_est:.3f} and "
            f"ICC(2,1) = {icc_est:.3f} are consistent (gap 95% CI "
            f"[{gap_lo:.3f}, {gap_hi:.3f}] includes 0)"
        )
    what = (
        f"Compares {corr_label}, which is insensitive to a systematic offset "
        "or scale difference between judge and human scores, against "
        "ICC(2,1), which penalizes exactly that. A paired bootstrap CI is "
        "used so the comparison reflects the sampling distribution of the "
        "gap itself rather than the union of two separate CIs."
    )
    why = (
        "A correlation-type metric can look strong even when a judge is "
        "systematically shifted or compressed relative to human scores, "
        "since it only requires the two to move together. This check exists "
        "to catch that failure mode before it's mistaken for genuine "
        "agreement."
    )
    if flagged:
        interpretation = (
            "the judge tracks human relative ordering but disagrees on "
            "absolute scale — treat raw judge scores as biased; consider "
            "using the Bayesian calibration model fit by validate_alignment "
            "(e.g. via compare(alignment=...)) to correct for it before "
            "drawing conclusions from raw judge scores"
        )
    else:
        interpretation = (
            "the correlation and absolute-agreement metrics tell a "
            "consistent story — no sign that the judge's ranking ability is "
            "masking a scale or offset problem"
        )
    return {
        "passed": not flagged,
        "message": message,
        "corr_label": corr_label,
        "corr_estimate": corr_est,
        "icc_estimate": icc_est,
        "gap": gap_est,
        "gap_ci_low": gap_lo,
        "gap_ci_high": gap_hi,
        "what": what,
        "why": why,
        "interpretation": interpretation,
    }


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
        band, interp, example = _interpret_pct_agreement(est, lo, hi, len(llm), "Percent agreement")
        metrics["percent_agreement"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Percent agreement",
            "band": band,
            "what": (
                "The fraction of items where the LLM judge's label exactly matches "
                "the human label."
            ),
            "why": (
                "Included as an intuitive baseline agreement measure alongside "
                "Cohen's κ, since your judge produces binary (0/1) labels."
            ),
            "interpretation": interp,
            "example": example,
        }
        est, lo, hi = _bootstrap_ci_2(kappa, llm, human, alpha=alpha, rng=rng)
        band, interp, example = _interpret_kappa(est, lo, hi, len(llm), "Cohen's κ")
        metrics["cohens_kappa"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Cohen's κ",
            "band": band,
            "what": (
                "Percent agreement adjusted for the rate of agreement expected from "
                "two raters guessing at random, given the observed marginal label "
                "rates (Cohen, 1960)."
            ),
            "why": (
                "Your judge produces binary labels, so this nominal-data reliability "
                "statistic is the standard choice for reporting judge-human "
                "agreement in a paper."
            ),
            "interpretation": interp,
            "example": example,
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
            band, interp, example = _interpret_kappa(est, lo, hi, len(llm), "Weighted Cohen's κ")
            metrics["weighted_kappa"] = {
                "estimate": est, "ci_low": lo, "ci_high": hi,
                "label": "Weighted Cohen's κ",
                "band": band,
                "what": (
                    "Cohen's κ extended so that disagreements receive larger penalties "
                    "as ratings become farther apart on the ordinal scale (Cohen, 1968)."
                ),
                "why": (
                    "Your judge produces ordered categorical (Likert) labels, so an "
                    "ordinal-aware kappa is used instead of the unweighted version, "
                    "which would penalize a near-miss (e.g. judge=4 vs human=5) as "
                    "harshly as a large disagreement."
                ),
                "interpretation": interp,
                "example": example,
            }
        est, lo, hi = _bootstrap_ci_2(sp, llm, human, alpha=alpha, rng=rng)
        band, interp, example = _interpret_corr(est, lo, hi, len(llm), "Spearman r")
        metrics["spearman_r"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Spearman r",
            "band": band,
            "what": (
                "Rank correlation between judge and human scores — checks whether "
                "higher judge scores correspond to higher human scores, without "
                "assuming the categories are equally spaced."
            ),
            "why": (
                "Reported alongside weighted κ to show whether the judge preserves "
                "relative ordering, which matters if judge scores are mainly used "
                "to rank or compare outputs."
            ),
            "interpretation": interp,
            "example": example,
        }

        if k >= 2:
            icc_est, icc_lo, icc_hi = _bootstrap_ci_2(_icc_21, llm, human, alpha=alpha, rng=rng)
            band, interp, example = _interpret_icc(icc_est, icc_lo, icc_hi, len(llm), "ICC(2,1)")
            metrics["icc_21"] = {
                "estimate": icc_est, "ci_low": icc_lo, "ci_high": icc_hi,
                "label": "ICC(2,1)",
                "band": band,
                "what": (
                    "Two-way random-effects intraclass correlation for absolute "
                    "agreement (Shrout & Fleiss, 1979): unlike weighted κ's "
                    "category-index distance or Spearman r's rank comparison, it "
                    "is sensitive to a systematic offset between the judge and "
                    "human scale, not just whether they move together."
                ),
                "why": (
                    "Computed alongside weighted κ to check for absolute-scale "
                    "bias: a judge that ranks items correctly but is shifted or "
                    "compressed relative to human scores can still get a high "
                    "weighted κ / Spearman r while scoring poorly here."
                ),
                "interpretation": interp,
                "example": example,
            }

            gap_est, gap_lo, gap_hi = _bootstrap_ci_gap(wk, _icc_21, llm, human, alpha=alpha, rng=rng)
            metrics["_bias_check"] = _build_bias_check(
                "Weighted Cohen's κ", metrics["weighted_kappa"]["estimate"],
                icc_est, gap_est, gap_lo, gap_hi,
            )

    else:  # continuous / grade
        def pe(a, b):
            r, _ = pearsonr(a, b)
            return float(r)

        def sp(a, b):
            r, _ = spearmanr(a, b)
            return float(r)

        est, lo, hi = _bootstrap_ci_2(pe, llm, human, alpha=alpha, rng=rng)
        band, interp, example = _interpret_corr(est, lo, hi, len(llm), "Pearson r")
        metrics["pearson_r"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Pearson r",
            "band": band,
            "what": "Linear correlation coefficient between judge and human scores.",
            "why": (
                "Your judge produces continuous/numeric scores, so a correlation "
                "coefficient is the standard way to summarize agreement."
            ),
            "interpretation": interp,
            "example": example,
        }
        est, lo, hi = _bootstrap_ci_2(sp, llm, human, alpha=alpha, rng=rng)
        band, interp, example = _interpret_corr(est, lo, hi, len(llm), "Spearman r")
        metrics["spearman_r"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Spearman r",
            "band": band,
            "what": "Rank correlation between judge and human scores.",
            "why": (
                "Reported alongside Pearson r to check whether agreement holds even "
                "if the judge-human relationship is monotonic but non-linear (e.g. "
                "the judge saturates at high scores)."
            ),
            "interpretation": interp,
            "example": example,
        }

        icc_est, icc_lo, icc_hi = _bootstrap_ci_2(_icc_21, llm, human, alpha=alpha, rng=rng)
        band, interp, example = _interpret_icc(icc_est, icc_lo, icc_hi, len(llm), "ICC(2,1)")
        metrics["icc_21"] = {
            "estimate": icc_est, "ci_low": icc_lo, "ci_high": icc_hi,
            "label": "ICC(2,1)",
            "band": band,
            "what": (
                "Two-way random-effects intraclass correlation for absolute "
                "agreement (Shrout & Fleiss, 1979): unlike Pearson/Spearman r, "
                "which are invariant to any linear rescaling of one variable, "
                "this is sensitive to a systematic offset or scale mismatch "
                "between the judge and human scale."
            ),
            "why": (
                "Computed alongside Pearson r to check for absolute-scale bias: "
                "a judge that is consistently shifted or compressed relative to "
                "human scores can still score a perfect Pearson r while "
                "disagreeing badly here."
            ),
            "interpretation": interp,
            "example": example,
        }

        gap_est, gap_lo, gap_hi = _bootstrap_ci_gap(pe, _icc_21, llm, human, alpha=alpha, rng=rng)
        metrics["_bias_check"] = _build_bias_check(
            "Pearson r", metrics["pearson_r"]["estimate"],
            icc_est, gap_est, gap_lo, gap_hi,
        )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Representativeness checks
# ─────────────────────────────────────────────────────────────────────────────

# Why representativeness is checked at all — shared across the score-distribution
# and slice-column checks, since both exist to answer the same question.
_REPRESENTATIVENESS_WHY = (
    "The calibration model and alignment metrics above are fit only on the "
    "labeled subset; if that subset isn't representative of the full item pool, "
    "statistical inference may not generalize to "
    "unlabeled items."
)


def _interpret_representativeness(passed: bool, subject: str) -> str:
    if passed:
        return (
            f"no evidence (p ≥ 0.05) that {subject} differs between the labeled "
            "subset and the full pool — alignment estimates should generalize "
            "reasonably well"
        )
    return (
        f"{subject} differs between the labeled subset and the full pool "
        "(p < 0.05) — treat alignment estimates as potentially biased for "
        "unlabeled items; consider expanding or re-sampling the alignment set"
    )


def _check_score_distribution(
    all_scores: np.ndarray,
    labeled_scores: np.ndarray,
    score_type: str,
) -> dict:
    if score_type == "binary":
        what = (
            "Chi-square test comparing the labeled subset's 0/1 score distribution "
            "to the unlabeled pool's."
        )
        labeled_0 = int((labeled_scores == 0).sum())
        labeled_1 = int((labeled_scores == 1).sum())
        all_0 = int((all_scores == 0).sum())
        all_1 = int((all_scores == 1).sum())
        unlabeled_0 = all_0 - labeled_0
        unlabeled_1 = all_1 - labeled_1
        if unlabeled_0 + unlabeled_1 == 0:
            return {
                "passed": True, "message": "all items are labeled", "p_value": None,
                "what": what,
                "why": _REPRESENTATIVENESS_WHY,
                "interpretation": (
                    "not applicable — every item already has a human label, so "
                    "there is no unlabeled pool to generalize to"
                ),
            }
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
        what = (
            "Kolmogorov–Smirnov test comparing the labeled subset's score "
            "distribution to the full item pool's."
        )
        if len(np.unique(labeled_scores)) < 2:
            return {
                "passed": True, "message": "insufficient labeled variation to test", "p_value": None,
                "what": what,
                "why": _REPRESENTATIVENESS_WHY,
                "interpretation": (
                    "not applicable — the labeled scores don't vary enough to run "
                    "this test"
                ),
            }
        _, p = ks_2samp(labeled_scores, all_scores)
        p = float(p)
        passed = p >= 0.05
        msg = f"KS p={p:.3f}"
        if not passed:
            msg += " — labeled subset appears non-representative of full score range"
    return {
        "passed": passed, "message": msg, "p_value": p,
        "what": what,
        "why": _REPRESENTATIVENESS_WHY,
        "interpretation": _interpret_representativeness(passed, "the score distribution"),
    }


def _check_slice_column(
    df: pd.DataFrame,
    labeled_mask: pd.Series,
    col: str,
) -> dict:
    what = (
        f"Chi-square test comparing the distribution of {col!r} between labeled "
        "and unlabeled items."
    )
    why = (
        "Checks whether the alignment set is representative across this "
        "categorical variable — important if judge accuracy might vary by "
        "subgroup (e.g. domain, difficulty, model)."
    )
    labeled = df.loc[labeled_mask, col].dropna()
    unlabeled = df.loc[~labeled_mask, col].dropna()
    if len(unlabeled) == 0:
        return {
            "passed": True, "message": "no unlabeled items", "p_value": None,
            "what": what, "why": why,
            "interpretation": (
                "not applicable — there are no unlabeled items to compare against"
            ),
        }
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
    return {
        "passed": passed, "message": msg, "p_value": p,
        "what": what, "why": why,
        "interpretation": _interpret_representativeness(passed, f"{col!r}"),
    }


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
    bias_check = alignment_metrics.pop("_bias_check", None)

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
        bias_check=bias_check,
    )
