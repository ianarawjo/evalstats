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
from scipy.stats import rankdata, friedmanchisquare, studentized_range

from .resampling import (
    bca_interval_1d,
    bayes_bootstrap_means_1d,
    bayes_bootstrap_diffs_nested,
    smooth_bootstrap_means_1d,
    smooth_bootstrap_diffs_nested,
    bootstrap_diffs_nested,
    bootstrap_means_1d,
    resolve_resampling_method,
    newcombe_paired_ci,
    bayes_paired_diff_ci,
    is_binary_scores,
    _stat,
    _nested_cell_mean_diffs,
    _reduce_rows,
    _weighted_medians_rows,
)
from .stats_utils import correct_pvalues
from ..config import get_alpha_ci


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


def _wilcoxon_signed_rank_p(diffs: np.ndarray) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p-value for per-input paired differences.

    Uses ``zero_method='wilcox'`` (discards zero differences before ranking),
    which is the most common convention.  Returns ``None`` if the test cannot
    be computed (all differences are zero, or fewer than one non-zero pair).
    """
    from scipy.stats import wilcoxon  # scipy is a core dep; import here to keep top-level clean

    if int(np.sum(diffs != 0)) < 1:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(result.pvalue)
    except ValueError:
        return None


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


def _fisher_exact_p(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Two-sided Fisher's exact p-value on the paired 2×2 contingency table.

    Uses table layout::

        [[n11, n10],
         [n01, n00]]

    where n10 is ``A=1, B=0`` and n01 is ``A=0, B=1``.

    Note that Fisher's exact test treats margins as fixed and does not exploit
    pairing in the same way as McNemar; it is provided as an optional
    conservative exact alternative for binary comparisons.
    """
    from scipy.stats import fisher_exact

    a_bin = (values_a >= 0.5).astype(int)
    b_bin = (values_b >= 0.5).astype(int)

    n11 = int(np.sum((a_bin == 1) & (b_bin == 1)))
    n10 = int(np.sum((a_bin == 1) & (b_bin == 0)))
    n01 = int(np.sum((a_bin == 0) & (b_bin == 1)))
    n00 = int(np.sum((a_bin == 0) & (b_bin == 0)))

    table = np.array([[n11, n10], [n01, n00]], dtype=int)
    _, p = fisher_exact(table, alternative="two-sided")
    p = float(p)
    if not np.isfinite(p):
        return 1.0
    return min(max(p, 0.0), 1.0)


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
    statistic: Literal["mean", "median"],
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided paired randomization p-value via sign-flipping.

    The null hypothesis is that paired differences are symmetric around zero,
    so each per-input difference can be multiplied by +1 or -1 with equal
    probability. This is the standard paired permutation/randomization test.
    """
    observed = abs(_stat(diffs, statistic))
    m = len(diffs)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_samples, m), replace=True)
    signed = signs * diffs[np.newaxis, :]
    if statistic == "median":
        null_stats = np.median(signed, axis=1)
    else:
        null_stats = np.mean(signed, axis=1)
    extreme_count = int(np.sum(np.abs(null_stats) >= observed))
    return float((extreme_count + 1) / (n_samples + 1))


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

    def summary(self, *, alpha: float | None = None, correction: str = "") -> None:
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

    # Friedman omnibus test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stat, p_val = friedmanchisquare(*[scores[i] for i in range(k)])
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
    simultaneous_ci: bool = False
    simultaneous_ci_method: Optional[str] = None  # 'max_t' or 'bonferroni'; None if not applied

    def get(self, a: str, b: str) -> PairedDiffResult:
        """Get the comparison result for templates a vs b."""
        if (a, b) in self.results:
            return self.results[(a, b)]
        if (b, a) in self.results:
            r = self.results[(b, a)]
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
            )
        raise KeyError(f"No comparison found for ({a}, {b})")

    def summary(self, a: str, b: str, *, alpha: float | None = None) -> None:
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
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto", "newcombe", "bayes_binary", "permutation", "fisher_exact", "sign_test"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "mean",
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
        (smoothed bootstrap via Gaussian KDE), ``'newcombe'`` for paired
        binary (0/1) data using Newcombe CI + exact McNemar p-value, or
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
            rng=rng, statistic=statistic,
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

    def _build_result(
        *,
        diffs: np.ndarray,
        point_d: float,
        std_d: float,
        ci_low: float,
        ci_high: float,
        p_value: float,
        test_name: str,
    ) -> PairedDiffResult:
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
            wilcoxon_p=_wilcoxon_signed_rank_p(diffs),
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
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name=f"bayes binary (n={n_bootstrap})",
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
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="newcombe (mcnemar p-value)",
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
        return _build_result(
            diffs=diffs,
            point_d=point_d,
            std_d=std_d,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            test_name="fisher exact (newcombe ci)",
        )

    # ------------------------------------------------------------------ #
    # Paired sign test path                                               #
    # ------------------------------------------------------------------ #
    if method in {"sign_test", "permutation"}:
        if scores.ndim == 3 and scores.shape[2] >= 3:
            return _seeded_fallback(method)
        if scores.ndim == 3:
            scores = scores.mean(axis=2)

        diffs, _, point_d, std_d = _paired_stats(scores[idx_a], scores[idx_b])
        alpha = 1.0 - ci

        boot_stats = bootstrap_means_1d(
            diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
        )
        ci_low, ci_high = _percentile_ci(boot_stats, alpha)

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
    diffs = scores[idx_a] - scores[idx_b]
    m = len(diffs)
    point_d = _stat(diffs, statistic)
    std_d = float(np.std(diffs, ddof=1))
    alpha = 1 - ci

    resolved_method = resolve_resampling_method(method, m)

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

    elif resolved_method in {"bca", "bayes_bootstrap", "smooth_bootstrap"}:
        samplers = {
            "bca": bootstrap_means_1d,
            "bayes_bootstrap": bayes_bootstrap_means_1d,
            "smooth_bootstrap": smooth_bootstrap_means_1d,
        }
        sampler = samplers[resolved_method]

        boot_stats = sampler(
            diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
        )
        if resolved_method == "bca":
            ci_low, ci_high = bca_interval_1d(
                diffs, point_d, boot_stats, alpha, statistic=statistic,
            )
        else:
            ci_low, ci_high = _percentile_ci(boot_stats, alpha)

        centered_diffs = diffs - point_d
        boot_centered_stats = sampler(
            centered_diffs, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
        )
        p_value = _bootstrap_tail_pvalue(boot_centered_stats, point_d)

        test_labels = {
            "bca": "bca bootstrap",
            "bayes_bootstrap": "bayesian bootstrap",
            "smooth_bootstrap": "smooth bootstrap",
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
    )


def _pairwise_diffs_seeded(
    scores: np.ndarray,
    idx_a: int,
    idx_b: int,
    label_a: str,
    label_b: str,
    *,
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto", "permutation", "sign_test"],
    ci: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
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

    def _bootstrap_tail_pvalue(boot_stats: np.ndarray) -> float:
        boot_centered = boot_stats - point_d
        extreme_count = np.sum(np.abs(boot_centered) >= abs(point_d))
        return float((extreme_count + 1) / (n_bootstrap + 1))

    if method == "permutation":
        boot_stats = bootstrap_diffs_nested(
            scores_a, scores_b, n_bootstrap, rng, statistic=statistic,
        )
        ci_low, ci_high = _percentile_ci(boot_stats)
        p_value = _paired_signflip_pvalue(
            cell_diffs, statistic=statistic, n_samples=n_bootstrap, rng=rng,
        )
        test_name = f"nested paired permutation + bootstrap ci (n={n_bootstrap}, R={R})"

    elif method == "sign_test":
        boot_stats = bootstrap_diffs_nested(
            scores_a, scores_b, n_bootstrap, rng, statistic=statistic,
        )
        ci_low, ci_high = _percentile_ci(boot_stats)
        p_value = _paired_sign_test_p(cell_diffs)
        test_name = f"nested paired sign test + bootstrap ci (n={n_bootstrap}, R={R})"

    elif resolved_method in {"bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap"}:
        samplers = {
            "bootstrap": bootstrap_diffs_nested,
            "bca": bootstrap_diffs_nested,
            "bayes_bootstrap": bayes_bootstrap_diffs_nested,
            "smooth_bootstrap": smooth_bootstrap_diffs_nested,
        }
        boot_stats = samplers[resolved_method](
            scores_a, scores_b, n_bootstrap, rng, statistic=statistic,
        )

        if resolved_method == "bca":
            # BCa: jackknife over inputs (the outer sampling unit) using cell_diffs.
            ci_low, ci_high = bca_interval_1d(
                cell_diffs, point_d, boot_stats, alpha, statistic=statistic,
            )
        else:
            ci_low, ci_high = _percentile_ci(boot_stats)

        p_value = _bootstrap_tail_pvalue(boot_stats)

        test_labels = {
            "bootstrap": "nested bootstrap",
            "bca": "nested bca bootstrap",
            "bayes_bootstrap": "nested bayesian bootstrap",
            "smooth_bootstrap": "nested smooth bootstrap",
        }
        test_name = f"{test_labels[resolved_method]} (n={n_bootstrap}, R={R})"

    else:
        raise ValueError(f"Unknown method: {method}")

    if method == "auto":
        test_name = f"auto→{test_name}"

    wilcoxon_p = _wilcoxon_signed_rank_p(cell_diffs)

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
    )


def _max_stat_simultaneous_cis(
    scores: np.ndarray,
    pairs: list[tuple[str, str]],
    labels: list[str],
    method: str,
    ci: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    statistic: Literal["mean", "median"],
) -> dict[tuple[str, str], tuple[float, float]]:
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
    pairs : list[tuple[str, str]]
        All pairs for which simultaneous CIs should be computed, in the
        canonical (label_a, label_b) storage order.
    labels : list[str]
        Template labels — used to map names to row indices in *scores*.
    method : str
        Bootstrap variant.  Supported: ``'bootstrap'``, ``'bca'``,
        ``'bayes_bootstrap'``, ``'smooth_bootstrap'``, ``'auto'``
        (treated as ``'smooth_bootstrap'``), ``'permutation'``,
        ``'sign_test'``.  Methods that do not use bootstrap resampling
        for CIs (``'newcombe'``, ``'fisher_exact'``, ``'bayes_binary'``,
        ``'lmm'``) are not supported; an empty dict is returned for these.
    ci : float
        Desired simultaneous confidence level (e.g. 0.95).
    n_bootstrap : int
        Number of bootstrap replicates.
    rng : np.random.Generator
    statistic : str
        ``'mean'`` or ``'median'``.

    Returns
    -------
    dict[tuple[str, str], tuple[float, float]]
        Maps each pair to its ``(ci_low, ci_high)`` simultaneous CI.
        Returns an empty dict for unsupported methods.
    """
    _BOOTSTRAP_COMPATIBLE = {
        "bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap",
        "permutation", "sign_test", "auto",
    }
    # Resolve 'auto' to its concrete method
    if method == "auto":
        method = "smooth_bootstrap"

    if method not in _BOOTSTRAP_COMPATIBLE or len(pairs) == 0:
        return {}

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

        else:
            # bootstrap, bca, permutation, sign_test — shared integer indices.
            input_idx = rng.integers(0, M, size=(n_bootstrap, M))
            # _batch_resample already computes the per-pair statistic: (B, k)
            boot_stats = _batch_resample(diffs_mat, input_idx, statistic)  # (B, k)

    # ------------------------------------------------------------------
    # Studentized max-T critical value and simultaneous CIs
    # ------------------------------------------------------------------
    se = np.std(boot_stats, axis=0, ddof=1)  # (k,)
    valid = se > 1e-12

    if not np.any(valid):
        # All SEs degenerate; simultaneous CI cannot be computed.
        return {}

    se_safe = np.where(valid, se, 1.0)
    T = (boot_stats - point_ests[np.newaxis, :]) / se_safe[np.newaxis, :]  # (B, k)

    # Max over valid pairs only; quantile gives the (1−α) simultaneous critical value.
    M_b = np.max(np.abs(T[:, valid]), axis=1)  # (B,)
    c = float(np.quantile(M_b, ci))

    sim_cis: dict[tuple[str, str], tuple[float, float]] = {}
    for p_idx, pair in enumerate(pairs):
        if valid[p_idx]:
            half = c * se[p_idx]
            sim_cis[pair] = (
                float(point_ests[p_idx] - half),
                float(point_ests[p_idx] + half),
            )
        else:
            # SE is zero (constant differences); CI degenerates to a point.
            sim_cis[pair] = (float(point_ests[p_idx]), float(point_ests[p_idx]))

    return sim_cis


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
    ``'newcombe'``, ``'fisher_exact'``, and ``'bayes_binary'``.

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
    "bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap",
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
    ``'newcombe'``, ``'fisher_exact'``, and ``'bayes_binary'``, and also
    as a safety net if the bootstrap path returns an empty result.

    Returns
    -------
    tuple[dict, str]
        ``(cis, method_used)`` where *method_used* is ``'max_t'`` or
        ``'bonferroni'``.
    """
    if method in _SIMULTANEOUS_CI_BOOTSTRAP_METHODS:
        cis = _max_stat_simultaneous_cis(
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
            return cis, "max_t"

    # Fallback: Bonferroni t-intervals work for any method.
    cis = _bonferroni_simultaneous_cis(results=results, pairs=pairs, ci=ci)
    return cis, "bonferroni"


def all_pairwise(
    scores: np.ndarray,
    labels: list[str],
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto", "newcombe", "bayes_binary", "permutation", "fisher_exact", "sign_test"] = "auto",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "fdr_bh",
    rng: Optional[np.random.Generator] = None,
    statistic: Literal["mean", "median"] = "mean",
    simultaneous_ci: bool = False,
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
          ``'bayes_bootstrap'``, ``'smooth_bootstrap'``, ``'permutation'``,
          ``'sign_test'``, ``'auto'``): studentized bootstrap max-T
          (Romano–Wolf).  All pairs share the same bootstrap resamples so
          the joint distribution of ``max_{(i,j)} |T_ij^b|`` accounts for
          the correlation between comparisons.  Less conservative than
          Bonferroni.
        * **Analytical methods** (``'newcombe'``, ``'fisher_exact'``,
          ``'bayes_binary'``): Bonferroni t-intervals at the
          ``1 − (1−α)/k`` level, computed from ``per_input_diffs``.

        The method actually used is recorded in
        :attr:`PairwiseMatrix.simultaneous_ci_method` (``'max_stat'`` or
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
                statistic=statistic,
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
                test_method=f"{r.test_method} ({correction}-corrected)",
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
                n_runs=r.n_runs,
                statistic=r.statistic,
                wilcoxon_p=float(adj_wsr) if adj_wsr is not None else None,
            )

    # Simultaneous CIs: bootstrap max-T when possible, Bonferroni otherwise.
    applied_simultaneous_ci = False
    applied_simultaneous_ci_method: Optional[str] = None
    if simultaneous_ci and len(pairs) > 0:
        sim_cis, sim_method = _simultaneous_cis_router(
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
                "simultaneous CI (max-T)"
                if sim_method == "max_t"
                else "simultaneous CI (Bonferroni)"
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
                    p_value=r.p_value,
                    test_method=f"{r.test_method} ({ci_label})",
                    n_inputs=r.n_inputs,
                    per_input_diffs=r.per_input_diffs,
                    n_runs=r.n_runs,
                    statistic=r.statistic,
                    wilcoxon_p=r.wilcoxon_p,
                )

    # Friedman omnibus + Nemenyi post-hoc (only meaningful for k >= 2).
    friedman: Optional[FriedmanResult] = None
    if len(labels) >= 2:
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
    method: Literal["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "auto", "newcombe", "bayes_binary", "permutation", "fisher_exact", "sign_test"] = "auto",
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

