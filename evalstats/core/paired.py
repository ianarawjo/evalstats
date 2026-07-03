"""Paired statistical comparisons between templates.

All comparisons are paired by input, since every template is evaluated on the
same benchmark set. This eliminates input-level variance and dramatically
increases statistical power compared to unpaired tests.

When the score array includes a run axis (R >= 3), pairwise comparisons use
a two-level (nested) bootstrap that resamples both inputs and within-cell
runs, so that seed variance is correctly propagated into confidence intervals
rather than being silently discarded.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.stats import rankdata, studentized_range

from ..tests import (
    ttest as _es_ttest,
    wilcoxon as _es_wilcoxon,
    friedman as _es_friedman,
    _mcnemar_p,
    _fisher_exact_p,
    _paired_sign_test_p,
    _paired_signflip_pvalue,
)
from .resampling import (
    bca_interval_1d,
    bayes_bootstrap_means_1d,
    bayes_bootstrap_diffs_nested,
    smooth_bootstrap_means_1d,
    smooth_bootstrap_diffs_nested,
    bootstrap_diffs_nested,
    bootstrap_means_1d,
    bootstrap_t_ci_1d,
    bootstrap_t_ci_nested,
    resolve_resampling_method,
    newcombe_paired_ci,
    tango_paired_ci,
    tango_paired_ci_multirun_moments,
    t_interval_ci_1d,
    bayes_paired_diff_ci,
    is_binary_scores,
    _stat,
    _nested_cell_mean_diffs,
    _reduce_rows,
    _weighted_medians_rows,
)
from .stats_utils import correct_pvalues
from ..config import get_alpha_ci, GRADIENT_CI_ALPHAS


BAYES_BINARY_LARGE_N_THRESHOLD = 200


def _warn_bayes_binary_large_n(n_inputs: int, *, stacklevel: int = 4) -> None:
    """Warn when bayes_binary pairwise CI is used beyond its calibrated range."""
    if n_inputs < BAYES_BINARY_LARGE_N_THRESHOLD:
        return

    warnings.warn(
        "method='bayes_binary' was requested for pairwise binary comparison "
        f"with N={n_inputs} inputs. Simulations indicate this importance-"
        "sampling-based CI becomes dangerously overconfident at larger N "
        "(roughly ~10% at N=500 and ~20% at N=1000). "
        "Use method='newcombe' (or method='auto') for calibrated pairwise "
        "intervals at this sample size.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _rank_biserial(diffs: np.ndarray) -> float:
    """Rank biserial correlation for paired differences.

    Computed from the signed-rank decomposition of ``diffs``: rank the absolute
    values of non-zero differences, then return (R+ - R-) / (R+ + R-), where
    R+ and R- are the sums of ranks for positive and negative differences
    respectively.  Returns 0.0 when all differences are zero.

    Interpretation guidelines (Kerby, 2014): small ≈ 0.1, medium ≈ 0.3,
    large ≈ 0.5.  Range is [-1, 1].
    """
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    r_plus = float(np.sum(ranks[nonzero > 0]))
    r_minus = float(np.sum(ranks[nonzero < 0]))
    total = r_plus + r_minus
    return (r_plus - r_minus) / total if total > 0 else 0.0


def _compute_agreement_mcc(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> tuple[float, tuple[int, int, int, int]]:
    """Compute pairwise agreement MCC and confusion counts for two binary arrays.

    Treats ``values_a`` and ``values_b`` as binary vectors (thresholded at 0.5)
    and computes the Matthews Correlation Coefficient measuring how correlated
    their pass/fail patterns are — independent of which model is "better."

    MCC is symmetric: MCC(a, b) == MCC(b, a).  Range is [-1, 1]:
      +1 = identical pass/fail patterns
       0 = uncorrelated (independent errors)
      -1 = perfectly opposite patterns

    Returns
    -------
    (mcc, (n11, n10, n01, n00))
        n11 = both pass, n10 = A passes B fails, n01 = A fails B passes,
        n00 = both fail.
    """
    a = (np.asarray(values_a) >= 0.5).astype(int)
    b = (np.asarray(values_b) >= 0.5).astype(int)
    n11 = int(np.sum((a == 1) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    # MCC: TP=n11, TN=n00, FP=n01, FN=n10 (treating a as reference, b as prediction).
    # Symmetric: swapping a↔b swaps FP↔FN, giving the same value.
    tp, tn, fp, fn = n11, n00, n01, n10
    denom_sq = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom_sq == 0.0:
        mcc = 0.0
    else:
        mcc = float((tp * tn - fp * fn) / (denom_sq ** 0.5))
    return mcc, (n11, n10, n01, n00)


@dataclass
class PairedDiffResult:
    """Result of a paired comparison between two templates."""

    template_a: str
    template_b: str
    point_diff: float       # point estimate under the chosen statistic
    std_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    test_method: str
    n_inputs: int
    per_input_diffs: np.ndarray  # shape (M,) — per-input cell-mean differences
    n_runs: int = 1              # R used; 1 means no seed dimension
    statistic: str = "mean"      # 'mean' or 'median'
    wilcoxon_p: Optional[float] = None  # Wilcoxon signed-rank p-value (two-sided, on per_input_diffs)
    agreement_mcc: Optional[float] = None  # pass/fail pattern correlation (binary data only)
    binary_confusion: Optional[tuple[int, int, int, int]] = None  # (n11, n10, n01, n00)
    multi_ci: Optional[dict[float, tuple[float, float]]] = None  # {alpha: (lo, hi)} gradient bands

    @property
    def rank_biserial(self) -> float:
        """Rank biserial correlation for paired differences.

        Computed from ``per_input_diffs`` via the signed-rank decomposition:
        rank absolute non-zero differences, then return (R+ − R−) / (R+ + R−).
        Range is [−1, 1].  Interpretation guidelines (Kerby, 2014):
        small ≈ 0.1, medium ≈ 0.3, large ≈ 0.5.
        """
        return _rank_biserial(self.per_input_diffs)

    @property
    def effect_size(self) -> float:
        """Alias for ``rank_biserial``."""
        return self.rank_biserial

    def summary(self, *, alpha: Optional[float] = None, correction: str = "") -> None:
        """Print a focused summary for this pairwise comparison.

        Displays the gap, an ASCII interval plot of the confidence interval,
        and a plain-language verdict.

        Parameters
        ----------
        alpha : float
            Significance threshold (default 0.01).
        correction : str
            Name of the multiple-comparisons correction applied, shown in the
            header when provided.

        Examples
        --------
        >>> pair = report.pairwise.get("Model A", "Model B")
        >>> pair.summary()
        """
        if alpha is None:
            alpha = get_alpha_ci()
        from .summary import print_pairwise_summary
        print_pairwise_summary(self, alpha=alpha, correction=correction)


@dataclass
class FriedmanResult:
    """Friedman omnibus test + Nemenyi pairwise post-hoc.

    The Friedman test is a non-parametric alternative to repeated-measures
    ANOVA.  It ranks treatments within each block (input) and tests whether
    any treatment's average rank differs from the others.

    The Nemenyi post-hoc uses the Studentized range distribution to compare
    all pairs of average ranks simultaneously (FWER-controlled at the family
    level — no additional correction needed).
    """

    statistic: float                          # Friedman χ² statistic
    df: int                                   # degrees of freedom = k - 1
    p_value: float                            # omnibus p-value
    nemenyi_p: dict[tuple[str, str], float]  # upper-triangle pairwise p-values
    avg_ranks: dict[str, float]              # mean rank per template (1 = best)
    n_inputs: int                             # N blocks
    n_templates: int                          # k treatments

    def get_nemenyi_p(self, a: str, b: str) -> Optional[float]:
        """Return Nemenyi p for a pair regardless of storage order."""
        if (a, b) in self.nemenyi_p:
            return self.nemenyi_p[(a, b)]
        if (b, a) in self.nemenyi_p:
            return self.nemenyi_p[(b, a)]
        return None


def friedman_nemenyi(scores: np.ndarray, labels: list[str]) -> FriedmanResult:
    """Friedman omnibus test + Nemenyi pairwise post-hoc (scipy only).
    NOTE: This function is verified to match R's friedman.test and 
    PMCMRplus::frdAllPairsNemenyiTest on a reference matrix in the tests/.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(k, N)`` — k templates × N inputs.  If 3-D ``(k, N, R)``,
        cell means are taken over runs before ranking.
    labels : list[str]
        Template labels, length k.

    Returns
    -------
    FriedmanResult
    """
    scores = np.asarray(scores)
    if scores.ndim not in (2, 3):
        raise ValueError("scores must have shape (k, N) or (k, N, R)")

    if scores.ndim == 3:
        scores = scores.mean(axis=2)  # (k, N) cell means

    k, N = scores.shape

    if len(labels) != k:
        raise ValueError(f"labels length ({len(labels)}) must match number of templates ({k})")
    if k < 3:
        raise ValueError("Friedman test requires at least 3 templates (k >= 3)")
    if N < 1:
        raise ValueError("scores must include at least one input (N >= 1)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values")

    # Friedman omnibus test — delegates to evalstats.tests.friedman (uncorrected
    # path) so the scipy call has a single implementation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _res = _es_friedman(*[scores[i] for i in range(k)], print_result=False)
    stat, p_val = _res.statistic, _res.p_value
    if not (np.isfinite(stat) and np.isfinite(p_val)):
        # Degenerate case (e.g., all treatments tied for every input).
        stat, p_val = 0.0, 1.0

    # Average ranks: rank across k treatments within each input, then average.
    # rank_matrix[i, j] = rank of template i for input j.
    rank_matrix = np.apply_along_axis(rankdata, 0, -scores)  # (k, N)
    avg_ranks = rank_matrix.mean(axis=1)  # (k,)

    # Nemenyi post-hoc: compare pairs via the Studentized range distribution.
    # Standard error of average-rank differences under H0.
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    nemenyi_p: dict[tuple[str, str], float] = {}
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(avg_ranks[i] - avg_ranks[j]) / se
            # Convert to Studentized range statistic (factor sqrt(2) per Demšar 2006).
            p = float(studentized_range.sf(q * np.sqrt(2), k, np.inf))
            nemenyi_p[(labels[i], labels[j])] = p

    avg_ranks_dict = {labels[i]: float(avg_ranks[i]) for i in range(k)}

    return FriedmanResult(
        statistic=float(stat),
        df=k - 1,
        p_value=float(p_val),
        nemenyi_p=nemenyi_p,
        avg_ranks=avg_ranks_dict,
        n_inputs=N,
        n_templates=k,
    )


@dataclass
class PairwiseMatrix:
    """Results of all pairwise comparisons."""

    labels: list[str]
    results: dict[tuple[str, str], PairedDiffResult]
    correction_method: str
    friedman: Optional[FriedmanResult] = None
    simultaneous_ci: bool = True
    simultaneous_ci_method: Optional[str] = None  # 'max_t' or 'bonferroni'; None if not applied

    def get(self, a: str, b: str) -> PairedDiffResult:
        """Get the comparison result for templates a vs b."""
        if (a, b) in self.results:
            return self.results[(a, b)]
        if (b, a) in self.results:
            r = self.results[(b, a)]
            # Flip confusion counts: swap n10 ↔ n01 (A and B are exchanged).
            # agreement_mcc is symmetric so it stays the same.
            flipped_conf: Optional[tuple[int, int, int, int]] = None
            if r.binary_confusion is not None:
                n11, n10, n01, n00 = r.binary_confusion
                flipped_conf = (n11, n01, n10, n00)
            # Flip the result
            return PairedDiffResult(
                template_a=a,
                template_b=b,
                point_diff=-r.point_diff,
                std_diff=r.std_diff,
                ci_low=-r.ci_high,
                ci_high=-r.ci_low,
                p_value=r.p_value,
                test_method=r.test_method,
                n_inputs=r.n_inputs,
                per_input_diffs=-r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
                wilcoxon_p=r.wilcoxon_p,  # two-sided, so p is the same when flipping direction
                agreement_mcc=r.agreement_mcc,
                binary_confusion=flipped_conf,
            )
        raise KeyError(f"No comparison found for ({a}, {b})")

    def summary(self, a: str, b: str, *, alpha: Optional[float] = None) -> None:
        """Print a focused summary for the comparison between `a` and `b`.

        Retrieves the pairwise result via ``get(a, b)``, then delegates to
        ``PairedDiffResult.summary()``, automatically passing the correction
        method stored on this matrix.

        Parameters
        ----------
        a, b : str
            Entity labels.  The direction is always ``a − b``.
        alpha : float
            Significance threshold (default 0.01).

        Examples
        --------
        >>> report.pairwise.summary("Model A", "Model B")
        """
        if alpha is None:
            alpha = get_alpha_ci()
        pair = self.get(a, b)
        pair.summary(alpha=alpha, correction=self.correction_method)

    def point_diff_matrix(self) -> np.ndarray:
        """Return NxN matrix of point-estimate differences (mean or median)."""
        n = len(self.labels)
        mat = np.zeros((n, n))
        for i, a in enumerate(self.labels):
            for j, b in enumerate(self.labels):
                if i != j:
                    mat[i, j] = self.get(a, b).point_diff
        return mat


def pairwise_differences(
    scores: np.ndarray,
    idx_a: int,
    idx_b: int,
    label_a: str = "A",
    label_b: str = "B",
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto", "newcombe", "tango", "bayes_binary", "permutation", "fisher_exact", "sign_test", "t_interval"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "mean",
    multi_ci: bool = False,
) -> PairedDiffResult:
    """Compute paired differences between two templates.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(N, M)`` or ``(N, M, R)``.
        When ``R >= 3`` a two-level nested bootstrap is used so that seed
        variance contributes to the confidence interval.  ``R = 1`` or
        ``R = 2`` fall back to the standard (non-seeded) path.
    idx_a, idx_b : int
        Indices of the two templates to compare.
    label_a, label_b : str
        Human-readable labels for the templates.
    method : str
        Statistical method: ``'auto'`` (default), ``'bootstrap'``, ``'bca'``,
        ``'bayes_bootstrap'`` (Bayesian bootstrap), ``'smooth_bootstrap'``
        (smoothed bootstrap via Gaussian KDE), ``'bootstrap_t'``
        (studentized bootstrap-t CI), ``'newcombe'`` for paired
        binary (0/1) data using Newcombe CI + exact McNemar p-value,
        ``'tango'`` for paired binary (0/1) data using Tango score CI +
        exact McNemar p-value, or
        ``'fisher_exact'`` for paired binary (0/1) data using Newcombe CI
        + two-sided Fisher's exact p-value on the 2×2 contingency table, or
        ``'bayes_binary'`` for paired binary (0/1) data using the
        Dirichlet-multinomial Bayesian model (Bowyer et al. 2025).
        Requires binary data; raises ValueError otherwise.
        ``'permutation'`` computes a paired sign-flip randomization p-value
        and reports a percentile-bootstrap CI for the paired effect size.
        ``'sign_test'`` computes an exact two-sided paired sign-test p-value
        (ties dropped) and reports a percentile-bootstrap CI for the paired
        effect size.
        ``'auto'`` selects ``'smooth_bootstrap'`` for non-binary data.
    ci : float
        Confidence level for the interval (default 0.95).
    n_bootstrap : int
        Number of bootstrap resamples.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    statistic : str
        Point-estimate and bootstrap statistic: ``'mean'`` (default) or
        ``'median'``.

    Returns
    -------
    PairedDiffResult
    """
    if rng is None:
        rng = np.random.default_rng()

    def _seeded_fallback(seed_method: str) -> PairedDiffResult:
        return _pairwise_diffs_seeded(
            scores, idx_a, idx_b, label_a, label_b,
            method=seed_method, ci=ci, n_bootstrap=n_bootstrap,
            rng=rng, statistic=statistic, multi_ci=multi_ci,
        )

    def _paired_stats(values_a: np.ndarray, values_b: np.ndarray) -> tuple[np.ndarray, int, float, float]:
        diffs = values_a - values_b
        m = len(diffs)
        point_d = _stat(diffs, statistic)
        std_d = float(np.std(diffs, ddof=1))
        return diffs, m, point_d, std_d

    def _percentile_ci(boot_stats: np.ndarray, alpha_val: float) -> tuple[float, float]:
        ci_low = float(np.percentile(boot_stats, 100 * alpha_val / 2))
        ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha_val / 2)))
        return ci_low, ci_high

    def _bootstrap_tail_pvalue(boot_centered_stats: np.ndarray, point: float) -> float:
        extreme_count = np.sum(np.abs(boot_centered_stats) >= abs(point))
        return float((extreme_count + 1) / (n_bootstrap + 1))

    def _bootstrap_t_tail_pvalue_1d(values: np.ndarray, observed_stat: float) -> float:
        """Two-sided bootstrap-t p-value for 1-D paired differences.

        Uses studentized pivots ``t* = (theta* - theta_hat) / se*`` and compares
        against ``|t_obs| = |theta_hat| / se_obs`` for the null ``theta = 0``.
        Falls back to centered-bootstrap tail p-value when studentization is
        unstable or undefined.
        """
        n = len(values)
        centered_values = values - observed_stat

        def _fallback_centered_tail_pvalue() -> float:
            centered_boot = bootstrap_means_1d(
                centered_values, n_bootstrap=n_bootstrap, rng=rng, statistic="mean",
            )
            return _bootstrap_tail_pvalue(centered_boot, observed_stat)

        if n < 2:
            # Degenerate case: no variance estimate is available for studentization.
            return 1.0

        idx = rng.integers(0, n, size=(n_bootstrap, n))
        samples = values[idx]                                # (B, n)
        boot_stats = samples.mean(axis=1)                    # (B,)
        boot_ses = np.std(samples, ddof=1, axis=1) / np.sqrt(n)

        se_obs = float(np.std(values, ddof=1)) / np.sqrt(n)
        if se_obs <= 0.0 or not np.isfinite(se_obs):
            return _fallback_centered_tail_pvalue()

        valid = np.isfinite(boot_ses) & (boot_ses > 0.0)
        if not np.any(valid):
            return _fallback_centered_tail_pvalue()
        se_floor = max(np.finfo(float).eps, 1e-8 * se_obs)
        tiny_frac = float(np.mean(valid & (boot_ses < se_floor)))
        if tiny_frac > 0.05:
            return _fallback_centered_tail_pvalue()
        valid = valid & (boot_ses >= se_floor)
        if not np.any(valid):
            return _fallback_centered_tail_pvalue()

        t_stats = (boot_stats[valid] - observed_stat) / boot_ses[valid]
        t_obs = abs(observed_stat) / se_obs
        extreme_count = int(np.sum(np.abs(t_stats) >= t_obs))
        return float((extreme_count + 1) / (len(t_stats) + 1))

    def _build_result(
        *,
        diffs: np.ndarray,
        point_d: float,
        std_d: float,
        ci_low: float,
        ci_high: float,
        p_value: float,
        test_name: str,
        values_a: Optional[np.ndarray] = None,
        values_b: Optional[np.ndarray] = None,
        multi_ci_dict: Optional[dict[float, tuple[float, float]]] = None,
    ) -> PairedDiffResult:
        agr_mcc: Optional[float] = None
        bin_conf: Optional[tuple[int, int, int, int]] = None
        if (
            values_a is not None
            and values_b is not None
            and is_binary_scores(np.stack([values_a, values_b]))
        ):
            agr_mcc, bin_conf = _compute_agreement_mcc(values_a, values_b)

        # Two-sided Wilcoxon signed-rank p-value, reported alongside whatever
        # primary method was chosen. Calls evalstats.tests.wilcoxon directly
        # (uncorrected path) rather than a local reimplementation, so the
        # scipy call has a single home. That function raises when all paired
        # differences are zero (matching plain scipy); caught here since a
        # supplementary stat should degrade to None rather than crash the
        # whole comparison.
        wa = values_a if values_a is not None else diffs
        wb = values_b if values_b is not None else np.zeros_like(diffs)
        wilcoxon_p: Optional[float] = None
        if int(np.sum((wa - wb) != 0)) >= 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wilcoxon_p = float(_es_wilcoxon(wa, wb, print_result=False).p_value)
            except ValueError:
                wilcoxon_p = None

        return PairedDiffResult(
            template_a=label_a,
            template_b=label_b,
            point_diff=point_d,
            std_diff=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_method=test_name,
            n_inputs=len(diffs),
            per_input_diffs=diffs,
            n_runs=1,
            statistic=statistic,
            wilcoxon_p=wilcoxon_p,
            agreement_mcc=agr_mcc,
            binary_confusion=bin_conf,
            multi_ci=multi_ci_dict,
        )

    # ------------------------------------------------------------------ #
    # Bayesian binary path (Dirichlet-multinomial paired model)           #
    # ------------------------------------------------------------------ #
    if method == "bayes_binary":
        # When R >= 3 the per-run cell means are not binary values;
        # fall back to smooth bootstrap for the seeded nested path.
        if scores.ndim == 3 and scores.shape[2] >= 3:
            return _seeded_fallback("smooth_bootstrap")
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores
        values_a = flat[idx_a]
        values_b = flat[idx_b]
        if not is_binary_scores(flat):
            raise ValueError(
                "method='bayes_binary' requires binary (0/1) data, but "
                "non-binary values were found in the score array. "
                "Use is_binary_scores() to check before calling."
            )
        diffs, m, point_d, std_d = _paired_stats(values_a, values_b)
        _warn_bayes_binary_large_n(m)
        alpha_val = 1.0 - ci
        ci_low, ci_high, prob_a_greater = bayes_paired_diff_ci(
            values_a, values_b, alpha_val, num_samples=n_bootstrap, rng=rng,
        )
        # Two-sided Bayesian p-value: posterior mass on the wrong side × 2
        p_value = float(2.0 * min(prob_a_greater, 1.0 - prob_a_greater))
        p_value = max(1.0 / (n_bootstrap + 1), p_value)
        mci: Optional[dict[float, tuple[float, float]]] = None
        if multi_ci:
            mci = {}
            for _a in GRADIENT_CI_ALPHAS:
                _lo, _hi, _ = bayes_paired_diff_ci(values_a, values_b, _a, num_samples=n_bootstrap, rng=rng)
                mci[_a] = (_lo, _hi)
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name=f"bayes binary (n={n_bootstrap})",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Newcombe path for paired binary (0/1) data                         #
    # ------------------------------------------------------------------ #
    if method == "newcombe":
        # When R >= 3 the cell means are proportions, not binary values.
        # Fall back to smooth bootstrap for the seeded nested path.
        if scores.ndim == 3 and scores.shape[2] >= 3:
            return _seeded_fallback("smooth_bootstrap")
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores
        values_a = flat[idx_a]
        values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci
        ci_low, ci_high = newcombe_paired_ci(values_a, values_b, alpha_val)
        p_value = _mcnemar_p(values_a, values_b)
        mci = {_a: newcombe_paired_ci(values_a, values_b, _a) for _a in GRADIENT_CI_ALPHAS} if multi_ci else None
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="newcombe (mcnemar p-value)",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    if method == "tango":
        multirun = scores.ndim == 3 and scores.shape[2] > 1
        if multirun:
            # Multi-run: use the moment-decomposition variant (tango_multirun_mmnt),
            # which accounts for within-item run variance and reduces exactly to the
            # standard Tango CI when n_runs == 1. Better calibrated than the flat
            # baseline (tango_paired_ci_flat) in simulation.
            values_a_full = scores[idx_a]   # (M, R)
            values_b_full = scores[idx_b]   # (M, R)
            values_a = values_a_full[:, 0]  # for _paired_stats / mcnemar (single-run view)
            values_b = values_b_full[:, 0]
        else:
            flat = scores.mean(axis=2) if scores.ndim == 3 else scores
            values_a = flat[idx_a]
            values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci
        if multirun:
            ci_low, ci_high = tango_paired_ci_multirun_moments(values_a_full, values_b_full, alpha_val)
            if multi_ci:
                mci = {_a: tango_paired_ci_multirun_moments(values_a_full, values_b_full, _a) for _a in GRADIENT_CI_ALPHAS}
            else:
                mci = None
        else:
            ci_low, ci_high = tango_paired_ci(values_a, values_b, alpha_val)
            mci = {_a: tango_paired_ci(values_a, values_b, _a) for _a in GRADIENT_CI_ALPHAS} if multi_ci else None
        p_value = _mcnemar_p(values_a, values_b)
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="tango",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    if method == "fisher_exact":
        if scores.ndim == 3:
            if scores.shape[2] > 1:
                warnings.warn(
                    "method='fisher_exact' uses binary outcomes and therefore "
                    "uses run index 0 when repeated runs are present.",
                    UserWarning,
                    stacklevel=3,
                )
            flat = scores[:, :, 0]
        else:
            flat = scores

        if not is_binary_scores(flat):
            raise ValueError(
                "method='fisher_exact' requires binary (0/1) data, but "
                "non-binary values were found in the score array. "
                "Use is_binary_scores() to check before calling."
            )

        values_a = flat[idx_a]
        values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci
        ci_low, ci_high = newcombe_paired_ci(values_a, values_b, alpha_val)
        p_value = _fisher_exact_p(values_a, values_b)
        mci = {_a: newcombe_paired_ci(values_a, values_b, _a) for _a in GRADIENT_CI_ALPHAS} if multi_ci else None
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="fisher exact (newcombe ci)",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Paired sign test path                                               #
    # ------------------------------------------------------------------ #
    if method in {"sign_test", "permutation"}:
        if scores.ndim == 3 and scores.shape[2] >= 3:
            return _seeded_fallback(method)
        if scores.ndim == 3:
            scores = scores.mean(axis=2)

        _va_st = scores[idx_a]
        _vb_st = scores[idx_b]
        diffs, _, point_d, std_d = _paired_stats(_va_st, _vb_st)
        alpha = 1.0 - ci

        boot_stats = bootstrap_means_1d(
            diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
        )
        ci_low, ci_high = _percentile_ci(boot_stats, alpha)
        mci = {_a: _percentile_ci(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS} if multi_ci else None

        if method == "sign_test":
            p_value = _paired_sign_test_p(diffs)
            test_name = f"paired sign test + bootstrap ci (n={n_bootstrap})"
        else:
            p_value = _paired_signflip_pvalue(
                diffs, statistic=statistic, n_samples=n_bootstrap, rng=rng,
            )
            test_name = f"paired permutation + bootstrap ci (n={n_bootstrap})"

        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name=test_name,
            values_a=_va_st,
            values_b=_vb_st,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Paired t-interval path                                              #
    # ------------------------------------------------------------------ #
    if method == "t_interval":
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores
        values_a = flat[idx_a]
        values_b = flat[idx_b]
        diffs, _, point_d, std_d = _paired_stats(values_a, values_b)
        alpha_val = 1.0 - ci
        ci_low, ci_high = t_interval_ci_1d(diffs, alpha_val)
        # Delegates to evalstats.tests.ttest (uncorrected paired path) so the
        # scipy call has a single implementation.
        t_result = _es_ttest(values_a, values_b, paired=True, print_result=False)
        p_value = float(t_result.p_value) if np.isfinite(t_result.p_value) else 1.0
        mci = {_a: t_interval_ci_1d(diffs, _a) for _a in GRADIENT_CI_ALPHAS} if multi_ci else None
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="paired t-interval",
            values_a=values_a,
            values_b=values_b,
            multi_ci_dict=mci,
        )

    # ------------------------------------------------------------------ #
    # Route: seeded (R >= 3) vs. standard (2-D or R < 3)                 #
    # ------------------------------------------------------------------ #
    if scores.ndim == 3:
        R = scores.shape[2]
        if R >= 3:
            return _seeded_fallback(method)
        # R == 1 or R == 2: collapse to 2-D (warning already issued during validation)
        scores = scores.mean(axis=2)

    # ------------------------------------------------------------------ #
    # Standard (non-seeded) path                                          #
    # ------------------------------------------------------------------ #
    _va_std = scores[idx_a]
    _vb_std = scores[idx_b]
    diffs = _va_std - _vb_std
    m = len(diffs)
    point_d = _stat(diffs, statistic)
    std_d = float(np.std(diffs, ddof=1))
    alpha = 1 - ci

    resolved_method = resolve_resampling_method(method, m)

    mci: Optional[dict[float, tuple[float, float]]] = None

    if resolved_method == "bootstrap":
        centered_diffs = diffs - point_d
        boot_centered_stats = np.empty(n_bootstrap)
        if statistic == "median":
            for b in range(n_bootstrap):
                idx = rng.choice(m, size=m, replace=True)
                boot_centered_stats[b] = np.median(centered_diffs[idx])
        else:
            for b in range(n_bootstrap):
                idx = rng.choice(m, size=m, replace=True)
                boot_centered_stats[b] = np.mean(centered_diffs[idx])
        boot_stats = boot_centered_stats + point_d
        ci_low, ci_high = _percentile_ci(boot_stats, alpha)
        p_value = _bootstrap_tail_pvalue(boot_centered_stats, point_d)
        test_name = f"bootstrap (n={n_bootstrap})"
        if multi_ci:
            mci = {_a: _percentile_ci(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}

    elif resolved_method in {"bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t"}:
        samplers = {
            "bca": bootstrap_means_1d,
            "bayes_bootstrap": bayes_bootstrap_means_1d,
            "smooth_bootstrap": smooth_bootstrap_means_1d,
            "bootstrap_t": bootstrap_means_1d,
        }
        sampler = samplers[resolved_method]

        if resolved_method == "bootstrap_t":
            ci_low, ci_high = bootstrap_t_ci_1d(
                diffs,
                point_d,
                n_bootstrap,
                alpha,
                rng,
                statistic=statistic,
            )
            if multi_ci:
                mci = {
                    _a: bootstrap_t_ci_1d(
                        diffs,
                        point_d,
                        n_bootstrap,
                        _a,
                        rng,
                        statistic=statistic,
                    )
                    for _a in GRADIENT_CI_ALPHAS
                }
        else:
            boot_stats = sampler(
                diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
            )

        if resolved_method == "bca":
            ci_low, ci_high = bca_interval_1d(
                diffs, point_d, boot_stats, alpha, statistic=statistic,
            )
            if multi_ci:
                mci = {_a: bca_interval_1d(diffs, point_d, boot_stats, _a, statistic=statistic) for _a in GRADIENT_CI_ALPHAS}
        elif resolved_method != "bootstrap_t":
            ci_low, ci_high = _percentile_ci(boot_stats, alpha)
            if multi_ci:
                mci = {_a: _percentile_ci(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}

        if resolved_method == "bootstrap_t" and statistic == "mean":
            p_value = _bootstrap_t_tail_pvalue_1d(diffs, point_d)
        else:
            centered_diffs = diffs - point_d
            boot_centered_stats = sampler(
                centered_diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
            )
            p_value = _bootstrap_tail_pvalue(boot_centered_stats, point_d)

        test_labels = {
            "bca": "bca bootstrap",
            "bayes_bootstrap": "bayesian bootstrap",
            "smooth_bootstrap": "smooth bootstrap",
            "bootstrap_t": "bootstrap-t",
        }
        test_name = f"{test_labels[resolved_method]} (n={n_bootstrap})"

    else:
        raise ValueError(f"Unknown method: {method}")

    if method == "auto":
        test_name = f"auto→{test_name}"

    return _build_result(
        diffs=diffs,
        point_d=point_d,
        std_d=std_d,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        test_name=test_name,
        values_a=_va_std,
        values_b=_vb_std,
        multi_ci_dict=mci,
    )


def _pairwise_diffs_seeded(
    scores: np.ndarray,
    idx_a: int,
    idx_b: int,
    label_a: str,
    label_b: str,
    *,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto", "permutation", "sign_test"],
    ci: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
    multi_ci: bool = False,
) -> PairedDiffResult:
    """Seeded paired comparison using a two-level nested bootstrap.

    ``scores`` has shape ``(N, M, R)`` with R >= 3.

    Point estimates are computed from per-input cell means (averaged over
    runs).  The bootstrap resamples both inputs and within-cell runs so that
    seed variance is propagated into the CI.  For BCa, the jackknife
    acceleration is estimated at the input level (leaving one input out at a
    time), which is the correct primary sampling unit.
    """
    M, R = scores.shape[1], scores.shape[2]
    scores_a = scores[idx_a]   # (M, R)
    scores_b = scores[idx_b]   # (M, R)

    # Point estimates from cell means (within-cell aggregation always uses mean).
    cell_means_a = scores_a.mean(axis=1)    # (M,)
    cell_means_b = scores_b.mean(axis=1)    # (M,)
    cell_diffs = cell_means_a - cell_means_b  # (M,)

    point_d = _stat(cell_diffs, statistic)
    std_d = float(cell_diffs.std(ddof=1))
    alpha = 1 - ci

    resolved_method = resolve_resampling_method(method, M)

    def _percentile_ci(boot_stats: np.ndarray) -> tuple[float, float]:
        ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        return ci_low, ci_high

    def _percentile_ci_alpha(boot_stats: np.ndarray, a: float) -> tuple[float, float]:
        return (
            float(np.percentile(boot_stats, 100 * a / 2)),
            float(np.percentile(boot_stats, 100 * (1 - a / 2))),
        )

    def _bootstrap_tail_pvalue(boot_stats: np.ndarray) -> float:
        boot_centered = boot_stats - point_d
        extreme_count = np.sum(np.abs(boot_centered) >= abs(point_d))
        return float((extreme_count + 1) / (n_bootstrap + 1))

    def _bootstrap_t_tail_pvalue_nested(diff_scores: np.ndarray) -> float:
        """Two-sided bootstrap-t p-value for seeded paired differences.

        ``diff_scores`` has shape ``(M, R)``. Studentization is performed using
        bootstrap replicate SE over resampled input-level cell means.
        """
        m_inputs, n_runs = diff_scores.shape
        cell_means_obs = diff_scores.mean(axis=1)
        se_obs = float(np.std(cell_means_obs, ddof=1)) / np.sqrt(m_inputs)

        if se_obs <= 0.0 or not np.isfinite(se_obs):
            boot_stats_fallback = bootstrap_diffs_nested(
                scores_a, scores_b, n_bootstrap, rng, statistic="mean",
            )
            return _bootstrap_tail_pvalue(boot_stats_fallback)

        input_idx = rng.integers(0, m_inputs, size=(n_bootstrap, m_inputs))
        run_idx = rng.integers(0, n_runs, size=(n_bootstrap, m_inputs, n_runs))

        selected = diff_scores[input_idx]  # (B, M, R)
        b_rng = np.arange(n_bootstrap)[:, np.newaxis, np.newaxis]
        m_rng = np.arange(m_inputs)[np.newaxis, :, np.newaxis]
        resampled = selected[b_rng, m_rng, run_idx]  # (B, M, R)
        cell_means_boot = resampled.mean(axis=2)  # (B, M)

        boot_stats = cell_means_boot.mean(axis=1)  # (B,)
        boot_ses = np.std(cell_means_boot, ddof=1, axis=1) / np.sqrt(m_inputs)

        valid = np.isfinite(boot_ses) & (boot_ses > 0.0)
        if not np.any(valid):
            return _bootstrap_tail_pvalue(boot_stats)
        se_floor = max(np.finfo(float).eps, 1e-8 * se_obs)
        tiny_frac = float(np.mean(valid & (boot_ses < se_floor)))
        if tiny_frac > 0.05:
            return _bootstrap_tail_pvalue(boot_stats)
        valid = valid & (boot_ses >= se_floor)
        if not np.any(valid):
            return _bootstrap_tail_pvalue(boot_stats)

        t_stats = (boot_stats[valid] - point_d) / boot_ses[valid]
        t_obs = abs(point_d) / se_obs
        extreme_count = int(np.sum(np.abs(t_stats) >= t_obs))
        return float((extreme_count + 1) / (len(t_stats) + 1))

    mci_seeded: Optional[dict[float, tuple[float, float]]] = None

    if method == "permutation":
        boot_stats = bootstrap_diffs_nested(
            scores_a, scores_b, n_bootstrap, rng, statistic=statistic,
        )
        ci_low, ci_high = _percentile_ci(boot_stats)
        if multi_ci:
            mci_seeded = {_a: _percentile_ci_alpha(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}
        p_value = _paired_signflip_pvalue(
            cell_diffs, statistic=statistic, n_samples=n_bootstrap, rng=rng,
        )
        test_name = f"nested paired permutation + bootstrap ci (n={n_bootstrap}, R={R})"

    elif method == "sign_test":
        boot_stats = bootstrap_diffs_nested(
            scores_a, scores_b, n_bootstrap, rng, statistic=statistic,
        )
        ci_low, ci_high = _percentile_ci(boot_stats)
        if multi_ci:
            mci_seeded = {_a: _percentile_ci_alpha(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}
        p_value = _paired_sign_test_p(cell_diffs)
        test_name = f"nested paired sign test + bootstrap ci (n={n_bootstrap}, R={R})"

    elif resolved_method in {"bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t"}:
        samplers = {
            "bootstrap": bootstrap_diffs_nested,
            "bca": bootstrap_diffs_nested,
            "bayes_bootstrap": bayes_bootstrap_diffs_nested,
            "smooth_bootstrap": smooth_bootstrap_diffs_nested,
            "bootstrap_t": bootstrap_diffs_nested,
        }

        if resolved_method == "bootstrap_t" and statistic == "mean":
            # bootstrap_t_ci_nested/_bootstrap_t_tail_pvalue_nested draw their
            # own studentized resamples; the plain sampler is not needed here.
            diff_scores = scores_a - scores_b  # (M, R)
            ci_low, ci_high = bootstrap_t_ci_nested(
                diff_scores,
                point_d,
                n_bootstrap,
                alpha,
                rng,
            )
            if multi_ci:
                mci_seeded = {
                    _a: bootstrap_t_ci_nested(
                        diff_scores,
                        point_d,
                        n_bootstrap,
                        _a,
                        rng,
                    )
                    for _a in GRADIENT_CI_ALPHAS
                }
            p_value = _bootstrap_t_tail_pvalue_nested(diff_scores)
        else:
            boot_stats = samplers[resolved_method](
                scores_a, scores_b, n_bootstrap, rng, statistic=statistic,
            )

            if resolved_method == "bootstrap_t":
                # statistic == "median": studentization isn't implemented for
                # median, so fall back to plain percentile bootstrap.
                warnings.warn(
                    "nested bootstrap-t studentization is implemented for "
                    "'mean'; falling back to percentile bootstrap for "
                    "'median'.",
                    UserWarning,
                    stacklevel=3,
                )
                ci_low, ci_high = _percentile_ci(boot_stats)
                if multi_ci:
                    mci_seeded = {_a: _percentile_ci_alpha(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}
            elif resolved_method == "bca":
                # BCa: jackknife over inputs (the outer sampling unit) using cell_diffs.
                ci_low, ci_high = bca_interval_1d(
                    cell_diffs, point_d, boot_stats, alpha, statistic=statistic,
                )
                if multi_ci:
                    mci_seeded = {_a: bca_interval_1d(cell_diffs, point_d, boot_stats, _a, statistic=statistic) for _a in GRADIENT_CI_ALPHAS}
            else:
                ci_low, ci_high = _percentile_ci(boot_stats)
                if multi_ci:
                    mci_seeded = {_a: _percentile_ci_alpha(boot_stats, _a) for _a in GRADIENT_CI_ALPHAS}

            p_value = _bootstrap_tail_pvalue(boot_stats)

        test_labels = {
            "bootstrap": "nested bootstrap",
            "bca": "nested bca bootstrap",
            "bayes_bootstrap": "nested bayesian bootstrap",
            "smooth_bootstrap": "nested smooth bootstrap",
            "bootstrap_t": "nested bootstrap-t",
        }
        test_name = f"{test_labels[resolved_method]} (n={n_bootstrap}, R={R})"

    else:
        raise ValueError(f"Unknown method: {method}")

    if method == "auto":
        test_name = f"auto→{test_name}"

    # Two-sided Wilcoxon signed-rank p-value on cell means, reported alongside
    # whatever primary (nested) method was chosen. See the non-seeded path in
    # pairwise_differences for why the ValueError guard is needed here.
    wilcoxon_p: Optional[float] = None
    if int(np.sum(cell_diffs != 0)) >= 1:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wilcoxon_p = float(_es_wilcoxon(cell_means_a, cell_means_b, print_result=False).p_value)
        except ValueError:
            wilcoxon_p = None

    # Agreement MCC for seeded binary data: use per-input majority vote.
    agr_mcc: Optional[float] = None
    bin_conf: Optional[tuple[int, int, int, int]] = None
    if is_binary_scores(scores_a) and is_binary_scores(scores_b):
        majority_a = (cell_means_a >= 0.5).astype(float)
        majority_b = (cell_means_b >= 0.5).astype(float)
        agr_mcc, bin_conf = _compute_agreement_mcc(majority_a, majority_b)

    return PairedDiffResult(
        template_a=label_a,
        template_b=label_b,
        point_diff=point_d,
        std_diff=std_d,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        test_method=test_name,
        n_inputs=M,
        per_input_diffs=cell_diffs,
        n_runs=R,
        statistic=statistic,
        wilcoxon_p=wilcoxon_p,
        agreement_mcc=agr_mcc,
        binary_confusion=bin_conf,
        multi_ci=mci_seeded,
    )


def _apply_max_t_cis(
    boot_stats: np.ndarray,
    point_ests: np.ndarray,
    pairs: list,
    ci: float,
) -> tuple[dict, dict]:
    """Apply the studentized max-T critical value to a pre-built bootstrap matrix.

    This is the shared computation used by both the standard resampling path
    and the pre-computed bootstrap path (e.g. PPI) in
    :func:`_max_stat_simultaneous_cis`.

    Parameters
    ----------
    boot_stats : np.ndarray, shape (B, k)
        Bootstrap distribution of pairwise diffs, one column per pair.
    point_ests : np.ndarray, shape (k,)
        Observed pairwise point estimates.
    pairs : list[tuple[str, str]]
        Pair labels in the same order as columns of *boot_stats*.
    ci : float
        Simultaneous confidence level (e.g. 0.95).

    Returns
    -------
    tuple[dict, dict]
        ``(sim_cis, max_t_pvalues)``.
    """
    se = np.std(boot_stats, axis=0, ddof=1)  # (k,)
    valid = se > 1e-12

    if not np.any(valid):
        return {}, {}

    se_safe = np.where(valid, se, 1.0)
    T = (boot_stats - point_ests[np.newaxis, :]) / se_safe[np.newaxis, :]  # (B, k)
    M_b = np.max(np.abs(T[:, valid]), axis=1)  # (B,)
    c = float(np.quantile(M_b, ci))
    B_total = len(M_b)

    sim_cis: dict = {}
    max_t_pvalues: dict = {}
    for p_idx, pair in enumerate(pairs):
        if valid[p_idx]:
            half = c * se[p_idx]
            sim_cis[pair] = (
                float(point_ests[p_idx] - half),
                float(point_ests[p_idx] + half),
            )
            t_obs = abs(float(point_ests[p_idx])) / float(se[p_idx])
            extreme = int(np.sum(M_b >= t_obs))
            max_t_pvalues[pair] = float((extreme + 1) / (B_total + 1))
        else:
            sim_cis[pair] = (float(point_ests[p_idx]), float(point_ests[p_idx]))
            max_t_pvalues[pair] = 1.0

    return sim_cis, max_t_pvalues


def _max_stat_simultaneous_cis(
    scores: np.ndarray,
    pairs: list[tuple[str, str]],
    labels: list[str],
    method: str,
    ci: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
    *,
    precomputed_boot_stats: Optional[np.ndarray] = None,
    precomputed_point_ests: Optional[np.ndarray] = None,
) -> tuple[dict, dict]:
    """Compute simultaneous CIs via the studentized bootstrap max-T method.

    Uses shared resamples across all pairs so that the joint distribution of
    the max standardized statistic naturally accounts for correlations between
    comparisons (unlike Bonferroni, which assumes independence).

    For each bootstrap replicate *b* and each pair *(i, j)*, the standardized
    statistic is::

        T_ij^b = (θ̂_ij^b − θ̂_ij) / SE_ij

    where SE_ij = std({θ̂_ij^b}) over all B replicates.  The simultaneous
    critical value *c* is the (1−α) quantile of::

        M^b = max_{(i,j)} |T_ij^b|

    and each simultaneous CI is [θ̂_ij − c·SE_ij, θ̂_ij + c·SE_ij].

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, M)`` or ``(N, M, R)``.  When ``R >= 3`` the seeded
        nested bootstrap is used; otherwise scores are collapsed to 2-D.
        Ignored when *precomputed_boot_stats* is supplied.
    pairs : list[tuple[str, str]]
        All pairs for which simultaneous CIs should be computed, in the
        canonical (label_a, label_b) storage order.
    labels : list[str]
        Template labels — used to map names to row indices in *scores*.
        Ignored when *precomputed_boot_stats* is supplied.
    method : str
        Bootstrap variant.  Supported: ``'bootstrap'``, ``'bca'``,
        ``'bayes_bootstrap'``, ``'smooth_bootstrap'``, ``'bootstrap_t'``, ``'auto'``
        (treated as ``'smooth_bootstrap'``), ``'permutation'``,
        ``'sign_test'``.  Methods that do not use bootstrap resampling
        for CIs (``'newcombe'``, ``'tango'``, ``'fisher_exact'``, ``'bayes_binary'``,
        ``'lmm'``) are not supported; an empty dict is returned for these.
        Ignored when *precomputed_boot_stats* is supplied.
    ci : float
        Desired simultaneous confidence level (e.g. 0.95).
    n_bootstrap : int
        Number of bootstrap replicates.  Ignored when *precomputed_boot_stats*
        is supplied.
    rng : np.random.Generator
        Ignored when *precomputed_boot_stats* is supplied.
    statistic : str
        ``'mean'`` or ``'median'``.  Ignored when *precomputed_boot_stats*
        is supplied.
    precomputed_boot_stats : np.ndarray, shape (B, k), optional
        Pre-computed bootstrap distribution of pairwise diffs, one column
        per pair in *pairs* order.  When provided the resampling block is
        skipped entirely and the max-T statistic is derived directly from
        this matrix.  Requires *precomputed_point_ests*.
    precomputed_point_ests : np.ndarray, shape (k,), optional
        Observed pairwise point estimates corresponding to each column of
        *precomputed_boot_stats*.  Required when *precomputed_boot_stats*
        is supplied.

    Returns
    -------
    tuple[dict[tuple[str, str], tuple[float, float]], dict[tuple[str, str], float]]
        ``(sim_cis, max_t_pvalues)`` where *sim_cis* maps each pair to its
        ``(ci_low, ci_high)`` simultaneous CI.  Returns ``({}, {})`` for
        unsupported methods or degenerate inputs.
    """
    if len(pairs) == 0:
        return {}, {}

    # ── Pre-computed bootstrap path (e.g. PPI correction) ────────────────────
    # When the caller already has a joint bootstrap distribution (one draw per
    # row, one pair per column), skip all resampling and run the shared max-T
    # computation directly.
    if precomputed_boot_stats is not None:
        if precomputed_point_ests is None:
            raise ValueError(
                "precomputed_point_ests must be provided together with "
                "precomputed_boot_stats"
            )
        return _apply_max_t_cis(
            np.asarray(precomputed_boot_stats, dtype=float),
            np.asarray(precomputed_point_ests, dtype=float),
            pairs,
            ci,
        )

    # ── Standard path: resample from raw scores ───────────────────────────────
    _BOOTSTRAP_COMPATIBLE = {
        "bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t",
        "permutation", "sign_test", "auto",
    }
    # Resolve 'auto' to its concrete method
    if method == "auto":
        method = "smooth_bootstrap"

    if method not in _BOOTSTRAP_COMPATIBLE:
        return {}, {}

    k = len(pairs)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    pair_indices = [(label_to_idx[a], label_to_idx[b]) for (a, b) in pairs]

    seeded = scores.ndim == 3 and scores.shape[2] >= 3

    # ------------------------------------------------------------------
    # Seeded path  (N, M, R) with R >= 3
    # ------------------------------------------------------------------
    if seeded:
        M, R = scores.shape[1], scores.shape[2]

        # Point estimates: statistic of per-input cell-mean differences.
        point_ests = np.array([
            _stat(scores[i].mean(axis=1) - scores[j].mean(axis=1), statistic)
            for (i, j) in pair_indices
        ])

        boot_stats_cols: list[np.ndarray] = []

        if method == "bayes_bootstrap":
            # Shared inner run-resample indices and shared Dirichlet weights.
            run_idx = rng.integers(0, R, size=(n_bootstrap, M, R))  # (B, M, R)
            exp_mat = rng.exponential(1.0, size=(n_bootstrap, M))
            outer_weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)  # (B, M)
            for (i, j) in pair_indices:
                diffs = _nested_cell_mean_diffs(
                    scores[i], scores[j], run_idx,
                )  # (B, M) — no outer resampling; Dirichlet weights applied below
                if statistic == "mean":
                    boot_stats_cols.append(
                        (outer_weights * diffs).sum(axis=1)
                    )
                else:
                    boot_stats_cols.append(
                        _weighted_medians_rows(diffs, outer_weights)
                    )
        else:
            # Shared outer input indices and inner run indices.
            input_idx = rng.integers(0, M, size=(n_bootstrap, M))  # (B, M)
            run_idx = rng.integers(0, R, size=(n_bootstrap, M, R))  # (B, M, R)
            for (i, j) in pair_indices:
                if method == "smooth_bootstrap":
                    from scipy.stats import gaussian_kde
                    cell_diffs = scores[i].mean(axis=1) - scores[j].mean(axis=1)
                    std_val = float(np.std(cell_diffs, ddof=1)) if M > 1 else 0.0
                    h = 0.0
                    if M >= 2 and np.isfinite(std_val) and std_val > 0:
                        try:
                            h = float(gaussian_kde(cell_diffs).factor * std_val)
                        except np.linalg.LinAlgError:
                            pass
                    diffs = _nested_cell_mean_diffs(
                        scores[i], scores[j], run_idx, input_idx,
                    )  # (B, M)
                    if h > 0.0:
                        diffs = diffs + rng.normal(0.0, h, size=(n_bootstrap, M))
                else:
                    # bootstrap, bca, permutation, sign_test
                    diffs = _nested_cell_mean_diffs(
                        scores[i], scores[j], run_idx, input_idx,
                    )  # (B, M)
                boot_stats_cols.append(_reduce_rows(diffs, statistic))  # (B,)

        boot_stats = np.column_stack(boot_stats_cols)  # (B, k)

    # ------------------------------------------------------------------
    # Non-seeded path  (N, M) or (N, M, R) with R < 3 collapsed to 2-D
    # ------------------------------------------------------------------
    else:
        def _batch_resample(
            diffs_mat: np.ndarray,
            input_idx: np.ndarray,
            statistic: str,
            batch_size: int = 128,
            bandwidths: Optional[np.ndarray] = None,
            noise_rng: Optional[np.random.Generator] = None,
        ) -> np.ndarray:
            """Memory-efficient joint resampling for Max-T statistics.

            Processes bootstrap resamples in batches so that only a slice of
            shape (batch, M, k) is live at once rather than the full (B, M, k).
            When ``bandwidths`` and ``noise_rng`` are supplied, KDE noise is
            added per-batch before aggregation (smooth bootstrap path).
            """
            M_mat = diffs_mat.T  # (M, k) — transposed for cache-friendly row access
            B, M = input_idx.shape
            k = diffs_mat.shape[0]
            out = np.empty((B, k), dtype=diffs_mat.dtype)

            for start in range(0, B, batch_size):
                end = min(start + batch_size, B)
                batch = end - start
                # (batch, M, k)
                chunk = M_mat[input_idx[start:end]]
                if bandwidths is not None and noise_rng is not None:
                    chunk = chunk + (
                        noise_rng.normal(0.0, 1.0, size=(batch, M, k))
                        * bandwidths[np.newaxis, np.newaxis, :]
                    )
                if statistic == "mean":
                    out[start:end] = chunk.mean(axis=1)
                else:
                    out[start:end] = np.median(chunk, axis=1)

            return out

        scores_2d = scores.mean(axis=2) if scores.ndim == 3 else scores  # (N, M)
        M = scores_2d.shape[1]

        # Per-pair diffs stacked: (k, M).
        # diffs_mat[:, input_idx] uses numpy fancy indexing to produce
        # shape (k, B, M), then .mean(axis=2).T → (B, k).
        diffs_mat = np.stack(
            [scores_2d[i] - scores_2d[j] for (i, j) in pair_indices],
            axis=0,
        )  # (k, M)

        if statistic == "mean":
            point_ests = diffs_mat.mean(axis=1)  # (k,)
        else:
            point_ests = np.median(diffs_mat, axis=1)  # (k,)

        if method == "bayes_bootstrap":
            # Shared Dirichlet weights over the M inputs.
            exp_mat = rng.exponential(1.0, size=(n_bootstrap, M))
            weights = exp_mat / exp_mat.sum(axis=1, keepdims=True)  # (B, M)
            if statistic == "mean":
                # (B, M) @ (M, k) → (B, k)
                boot_stats = weights @ diffs_mat.T
            else:
                boot_stats = np.empty((n_bootstrap, k))
                for p_idx in range(k):
                    vals = np.broadcast_to(diffs_mat[p_idx], (n_bootstrap, M))
                    boot_stats[:, p_idx] = _weighted_medians_rows(
                        np.ascontiguousarray(vals), weights,
                    )

        elif method == "smooth_bootstrap":
            from scipy.stats import gaussian_kde
            # Per-pair KDE bandwidth; shared input indices.
            bandwidths = np.zeros(k)
            for p_idx in range(k):
                d = diffs_mat[p_idx]
                std_val = float(np.std(d, ddof=1)) if M > 1 else 0.0
                if M >= 2 and np.isfinite(std_val) and std_val > 0:
                    try:
                        bandwidths[p_idx] = float(gaussian_kde(d).factor * std_val)
                    except np.linalg.LinAlgError:
                        pass

            input_idx = rng.integers(0, M, size=(n_bootstrap, M))
            boot_stats = _batch_resample(
                diffs_mat, input_idx, statistic,
                bandwidths=bandwidths, noise_rng=rng,
            )  # (B, k)

        elif method == "bootstrap_t" and statistic == "mean":
            # Studentized max-T: per-bootstrap-sample SE eliminates the
            # anti-conservative bias of plain pivots, which underestimate
            # SE by sqrt((n-1)/n).  The studentized pivot T_b = (d_b - d_obs)/se_b
            # and observed t_obs = |d_obs|/se_obs both follow approximately
            # t_{M-1}, so the Romano-Wolf guarantee holds.
            input_idx = rng.integers(0, M, size=(n_bootstrap, M))
            obs_se = np.std(diffs_mat, axis=1, ddof=1) / np.sqrt(M)  # (k,)

            M_mat_T = diffs_mat.T  # (M, k) — transposed for cache-friendly access
            batch_sz = 128
            bmeans_rows: list[np.ndarray] = []
            bses_rows: list[np.ndarray] = []
            for _s in range(0, n_bootstrap, batch_sz):
                _e = min(_s + batch_sz, n_bootstrap)
                chunk = M_mat_T[input_idx[_s:_e]]  # (batch, M, k)
                bmeans_rows.append(chunk.mean(axis=1))
                bses_rows.append(chunk.std(axis=1, ddof=1) / np.sqrt(M))
            boot_means_b = np.concatenate(bmeans_rows, axis=0)  # (B, k)
            boot_ses_b = np.concatenate(bses_rows, axis=0)  # (B, k)

            se_b_safe = np.where(boot_ses_b > 1e-12, boot_ses_b, 1.0)
            T_stud = (boot_means_b - point_ests) / se_b_safe  # (B, k)

            obs_se_safe = np.where(obs_se > 1e-12, obs_se, 1.0)
            t_obs_stud = np.abs(point_ests) / obs_se_safe  # (k,)
            se_valid_b = obs_se > 1e-12
            if not np.any(se_valid_b):
                return {}, {}

            M_b_stud = np.max(np.abs(T_stud[:, se_valid_b]), axis=1)  # (B,)
            c_stud = float(np.quantile(M_b_stud, ci))
            B_total_stud = len(M_b_stud)

            sim_cis_stud: dict = {}
            max_t_pvalues_stud: dict = {}
            for p_idx, pair in enumerate(pairs):
                if se_valid_b[p_idx]:
                    half = c_stud * float(obs_se[p_idx])
                    sim_cis_stud[pair] = (
                        float(point_ests[p_idx] - half),
                        float(point_ests[p_idx] + half),
                    )
                    t_val = float(t_obs_stud[p_idx])
                    extreme = int(np.sum(M_b_stud >= t_val))
                    max_t_pvalues_stud[pair] = float((extreme + 1) / (B_total_stud + 1))
                else:
                    sim_cis_stud[pair] = (float(point_ests[p_idx]), float(point_ests[p_idx]))
                    max_t_pvalues_stud[pair] = 1.0
            return sim_cis_stud, max_t_pvalues_stud

        else:
            # bootstrap, bca, permutation, sign_test, bootstrap_t+median —
            # shared integer indices, plain (non-studentized) pivots.
            input_idx = rng.integers(0, M, size=(n_bootstrap, M))
            # _batch_resample already computes the per-pair statistic: (B, k)
            boot_stats = _batch_resample(diffs_mat, input_idx, statistic)  # (B, k)

    return _apply_max_t_cis(boot_stats, point_ests, pairs, ci)


def _bonferroni_simultaneous_cis(
    results: dict[tuple[str, str], "PairedDiffResult"],
    pairs: list[tuple[str, str]],
    ci: float,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Bonferroni-corrected simultaneous CIs via per-pair paired t-intervals.

    Each CI is recomputed at the Bonferroni-adjusted confidence level
    ``1 − (1−ci)/k`` (where *k* = number of pairs) using the
    ``per_input_diffs`` already stored in each :class:`PairedDiffResult`.
    This makes the result independent of the original CI method, so it
    works as a universal fallback for non-bootstrap methods such as
    ``'newcombe'``, ``'tango'``, ``'fisher_exact'``, and ``'bayes_binary'``.

    Returns
    -------
    dict[tuple[str, str], tuple[float, float]]
        Maps each pair to its ``(ci_low, ci_high)`` simultaneous CI.
        Returns an empty dict when *pairs* is empty.
    """
    from scipy import stats as _scipy_stats

    k = len(pairs)
    if k == 0:
        return {}

    alpha_adj = (1.0 - ci) / k  # per-comparison alpha after Bonferroni

    sim_cis: dict[tuple[str, str], tuple[float, float]] = {}
    for pair in pairs:
        r = results[pair]
        diffs = r.per_input_diffs
        M = len(diffs)
        if M < 2:
            sim_cis[pair] = (float(r.point_diff), float(r.point_diff))
            continue
        se = float(np.std(diffs, ddof=1)) / np.sqrt(M)
        if se < 1e-12:
            sim_cis[pair] = (float(r.point_diff), float(r.point_diff))
            continue
        t_crit = float(_scipy_stats.t.ppf(1.0 - alpha_adj / 2.0, df=M - 1))
        half = t_crit * se
        sim_cis[pair] = (float(r.point_diff - half), float(r.point_diff + half))

    return sim_cis


# Methods for which _max_stat_simultaneous_cis can produce bootstrap CIs.
_SIMULTANEOUS_CI_BOOTSTRAP_METHODS = {
    "bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t",
    "permutation", "sign_test", "auto",
}


def _simultaneous_cis_router(
    scores: np.ndarray,
    results: dict[tuple[str, str], "PairedDiffResult"],
    pairs: list[tuple[str, str]],
    labels: list[str],
    method: str,
    ci: float,
    n_bootstrap: int,
    rng: "np.random.Generator",
    statistic: str,
) -> tuple[dict[tuple[str, str], tuple[float, float]], str]:
    """Route simultaneous CI computation to the best available method.

    Prefers the studentized bootstrap max-T method
    (:func:`_max_stat_simultaneous_cis`) when the chosen test *method* is
    bootstrap-compatible.  Falls back to Bonferroni t-intervals
    (:func:`_bonferroni_simultaneous_cis`) for analytical methods such as
    ``'newcombe'``, ``'tango'``, ``'fisher_exact'``, and ``'bayes_binary'``, and also
    as a safety net if the bootstrap path returns an empty result.

    Returns
    -------
    tuple[dict, str, dict]
        ``(cis, method_used, max_t_pvalues)`` where *method_used* is
        ``'max_t'`` or ``'bonferroni'``.  *max_t_pvalues* maps each pair to
        its max-T p-value when *method_used* is ``'max_t'``; empty dict
        otherwise.
    """
    if method in _SIMULTANEOUS_CI_BOOTSTRAP_METHODS:
        cis, max_t_pvalues = _max_stat_simultaneous_cis(
            scores=scores,
            pairs=pairs,
            labels=labels,
            method=method,
            ci=ci,
            n_bootstrap=n_bootstrap,
            rng=rng,
            statistic=statistic,
        )
        if cis:
            return cis, "max_t", max_t_pvalues

    # Fallback: Bonferroni t-intervals work for any method.
    cis = _bonferroni_simultaneous_cis(results=results, pairs=pairs, ci=ci)
    return cis, "bonferroni", {}


def all_pairwise(
    scores: np.ndarray,
    labels: list[str],
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto", "newcombe", "tango", "bayes_binary", "permutation", "fisher_exact", "sign_test", "t_interval"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "fdr_bh",
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "mean",
    simultaneous_ci: bool = True,
    omnibus: bool = False,
    multi_ci: bool = False,
) -> PairwiseMatrix:
    """Compute all pairwise comparisons with multiple comparisons correction.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(N, M)`` or ``(N, M, R)``.
        When ``R >= 3`` each comparison uses the nested bootstrap.
    labels : list[str]
        Template labels.
    method : str
        Statistical test method.
    ci : float
        Confidence level.
    n_bootstrap : int
        Number of bootstrap resamples.
    correction : str
        Multiple comparisons correction: ``'fdr_bh'`` (default),
        ``'holm'``, ``'bonferroni'``, or ``'none'``.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    statistic : str
        Point-estimate and bootstrap statistic: ``'median'`` (default) or
        ``'mean'``.
    simultaneous_ci : bool
        When ``True``, replace individual pairwise CIs with simultaneous
        (family-wise) CIs.  The method is chosen automatically:

        * **Bootstrap-compatible methods** (``'bootstrap'``, ``'bca'``,
                    ``'bayes_bootstrap'``, ``'smooth_bootstrap'``, ``'bootstrap_t'``, ``'permutation'``,
          ``'sign_test'``, ``'auto'``): studentized bootstrap max-T
          (Romano–Wolf).  All pairs share the same bootstrap resamples so
          the joint distribution of ``max_{(i,j)} |T_ij^b|`` accounts for
          the correlation between comparisons.  Less conservative than
          Bonferroni and widely used in genomics for situations with many correlated tests. 
            * **Analytical methods** (``'newcombe'``, ``'tango'``, ``'fisher_exact'``,
          ``'bayes_binary'``): Bonferroni t-intervals at the
          ``1 − (1−α)/k`` level, computed from ``per_input_diffs``.

        The method actually used is recorded in
        :attr:`PairwiseMatrix.simultaneous_ci_method` (``'max_stat'`` or
    omnibus : bool
        When ``True``, run the Friedman omnibus test (with Nemenyi post-hoc)
        alongside the pairwise comparisons.  Requires k ≥ 3.  Defaults to
        ``False`` — the Friedman test is a NHST procedure that may not be
        desirable in estimation-focused workflows.  The result is stored in
        :attr:`PairwiseMatrix.friedman`.
        ``'bonferroni'``) and annotated in each result's ``test_method``
        string.

    Returns
    -------
    PairwiseMatrix
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(labels)
    results = {}
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            result = pairwise_differences(
                scores, i, j, labels[i], labels[j],
                method=method, ci=ci, n_bootstrap=n_bootstrap, rng=rng,
                statistic=statistic, multi_ci=multi_ci,
            )
            results[(labels[i], labels[j])] = result
            pairs.append((labels[i], labels[j]))

    # Apply multiple comparisons correction to bootstrap p-values (and Wilcoxon if available).
    if correction != "none" and len(pairs) > 1:
        p_values = np.array([results[p].p_value for p in pairs])
        adjusted = correct_pvalues(p_values, correction)

        # Correct Wilcoxon p-values independently (only for pairs where the test ran).
        wsr_pairs = [p for p in pairs if results[p].wilcoxon_p is not None]
        if len(wsr_pairs) > 1:
            wsr_pvals = np.array([results[p].wilcoxon_p for p in wsr_pairs], dtype=float)
            wsr_adj_map = dict(zip(wsr_pairs, correct_pvalues(wsr_pvals, correction)))
        else:
            wsr_adj_map = {p: results[p].wilcoxon_p for p in wsr_pairs}

        for pair, adj_p in zip(pairs, adjusted):
            r = results[pair]
            adj_wsr = wsr_adj_map.get(pair, r.wilcoxon_p)
            results[pair] = PairedDiffResult(
                template_a=r.template_a,
                template_b=r.template_b,
                point_diff=r.point_diff,
                std_diff=r.std_diff,
                ci_low=r.ci_low,
                ci_high=r.ci_high,
                p_value=float(adj_p),
                test_method=r.test_method,
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
                wilcoxon_p=float(adj_wsr) if adj_wsr is not None else None,
                agreement_mcc=r.agreement_mcc,
                binary_confusion=r.binary_confusion,
                multi_ci=r.multi_ci,
            )

    # Simultaneous CIs: bootstrap max-T when possible, Bonferroni otherwise.
    applied_simultaneous_ci = False
    applied_simultaneous_ci_method: Optional[str] = None
    if simultaneous_ci and len(pairs) > 0:
        sim_cis, sim_method, sim_pvalues = _simultaneous_cis_router(
            scores=scores,
            results=results,
            pairs=pairs,
            labels=labels,
            method=method,
            ci=ci,
            n_bootstrap=n_bootstrap,
            rng=rng,
            statistic=statistic,
        )
        if sim_cis:
            applied_simultaneous_ci = True
            applied_simultaneous_ci_method = sim_method
            ci_label = (
                "simultaneous CIs computed with max-T"
                if sim_method == "max_t"
                else "simultaneous CIs computed with Bonferroni"
            )
            for pair, (ci_low, ci_high) in sim_cis.items():
                r = results[pair]
                results[pair] = PairedDiffResult(
                    template_a=r.template_a,
                    template_b=r.template_b,
                    point_diff=r.point_diff,
                    std_diff=r.std_diff,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    p_value=(
                        sim_pvalues.get(pair, r.p_value)
                        if sim_method == "max_t" and method in {
                            "bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto",
                        }
                        else r.p_value
                    ),
                    test_method=r.test_method,
                    n_inputs=r.n_inputs,
                    per_input_diffs=r.per_input_diffs,
                    n_runs=r.n_runs,
                    statistic=r.statistic,
                    wilcoxon_p=r.wilcoxon_p,
                    agreement_mcc=r.agreement_mcc,
                    binary_confusion=r.binary_confusion,
                    multi_ci=r.multi_ci,
                )

    # Friedman omnibus + Nemenyi post-hoc (only when explicitly requested).
    friedman: Optional[FriedmanResult] = None
    if omnibus and len(labels) >= 3:
        try:
            friedman = friedman_nemenyi(scores, labels)
        except Exception:
            pass

    return PairwiseMatrix(
        labels=labels,
        results=results,
        correction_method=correction,
        friedman=friedman,
        simultaneous_ci=applied_simultaneous_ci,
        simultaneous_ci_method=applied_simultaneous_ci_method,
    )


def vs_baseline(
    scores: np.ndarray,
    labels: list[str],
    baseline: str,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t", "auto", "newcombe", "tango", "bayes_binary", "permutation", "fisher_exact", "sign_test", "t_interval"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "fdr_bh",
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "mean",
) -> list[PairedDiffResult]:
    """Compare all templates against a designated baseline.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(N, M)`` or ``(N, M, R)``.
    labels : list[str]
        Template labels.
    baseline : str
        Label of the baseline template.
    method, ci, n_bootstrap, correction, rng :
        Same as ``all_pairwise``.
    statistic : str
        Point-estimate and bootstrap statistic: ``'median'`` (default) or
        ``'mean'``.

    Returns
    -------
    list[PairedDiffResult]
        One result per non-baseline template.
    """
    if rng is None:
        rng = np.random.default_rng()

    baseline_idx = labels.index(baseline)
    results = []

    for i, label in enumerate(labels):
        if i == baseline_idx:
            continue
        result = pairwise_differences(
            scores, i, baseline_idx, label, baseline,
            method=method, ci=ci, n_bootstrap=n_bootstrap, rng=rng,
            statistic=statistic,
        )
        results.append(result)

    # Apply correction to bootstrap p-values (and Wilcoxon if available).
    if correction != "none" and len(results) > 1:
        p_values = np.array([r.p_value for r in results])
        adjusted = correct_pvalues(p_values, correction)

        # Correct Wilcoxon p-values independently.
        wsr_results = [r for r in results if r.wilcoxon_p is not None]
        if len(wsr_results) > 1:
            wsr_pvals = np.array([r.wilcoxon_p for r in wsr_results], dtype=float)
            wsr_adj_vals = correct_pvalues(wsr_pvals, correction)
            wsr_adj_map = {
                (r.template_a, r.template_b): float(v)
                for r, v in zip(wsr_results, wsr_adj_vals)
            }
        else:
            wsr_adj_map = {
                (r.template_a, r.template_b): r.wilcoxon_p for r in wsr_results
            }

        results = [
            PairedDiffResult(
                template_a=r.template_a,
                template_b=r.template_b,
                point_diff=r.point_diff,
                std_diff=r.std_diff,
                ci_low=r.ci_low,
                ci_high=r.ci_high,
                p_value=float(adj_p),
                test_method=f"{r.test_method} ({correction}-corrected)",
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
                wilcoxon_p=wsr_adj_map.get((r.template_a, r.template_b), r.wilcoxon_p),
            )
            for r, adj_p in zip(results, adjusted)
        ]

    return results

