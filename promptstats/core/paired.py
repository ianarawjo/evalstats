"""Paired statistical comparisons between templates.

All comparisons are paired by input, since every template is evaluated on the
same benchmark set. This eliminates input-level variance and dramatically
increases statistical power compared to unpaired tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy import stats


@dataclass
class PairedDiffResult:
    """Result of a paired comparison between two templates."""

    template_a: str
    template_b: str
    mean_diff: float
    std_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    test_method: str
    n_inputs: int
    per_input_diffs: np.ndarray  # shape (M,)

    @property
    def significant(self) -> bool:
        """Whether the difference is significant (CI excludes zero)."""
        return self.ci_low > 0 or self.ci_high < 0

    @property
    def effect_size(self) -> float:
        """Cohen's d (paired): mean_diff / std_diff."""
        if self.std_diff == 0:
            return float("inf") if self.mean_diff != 0 else 0.0
        return self.mean_diff / self.std_diff


@dataclass
class PairwiseMatrix:
    """Results of all pairwise comparisons."""

    labels: list[str]
    results: dict[tuple[str, str], PairedDiffResult]
    correction_method: str

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
                mean_diff=-r.mean_diff,
                std_diff=r.std_diff,
                ci_low=-r.ci_high,
                ci_high=-r.ci_low,
                p_value=r.p_value,
                test_method=r.test_method,
                n_inputs=r.n_inputs,
                per_input_diffs=-r.per_input_diffs,
            )
        raise KeyError(f"No comparison found for ({a}, {b})")

    def mean_diff_matrix(self) -> np.ndarray:
        """Return NxN matrix of mean differences."""
        n = len(self.labels)
        mat = np.zeros((n, n))
        for i, a in enumerate(self.labels):
            for j, b in enumerate(self.labels):
                if i != j:
                    mat[i, j] = self.get(a, b).mean_diff
        return mat


def pairwise_differences(
    scores: np.ndarray,
    idx_a: int,
    idx_b: int,
    label_a: str = "A",
    label_b: str = "B",
    method: Literal["bootstrap", "paired_t", "wilcoxon"] = "bootstrap",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> PairedDiffResult:
    """Compute paired differences between two templates.

    Parameters
    ----------
    scores : np.ndarray
        2D score matrix of shape (N_templates, M_inputs).
    idx_a, idx_b : int
        Indices of the two templates to compare.
    label_a, label_b : str
        Human-readable labels for the templates.
    method : str
        Statistical test: 'bootstrap' (default, recommended), 'paired_t',
        or 'wilcoxon'.
    ci : float
        Confidence level for the interval (default 0.95).
    n_bootstrap : int
        Number of bootstrap resamples (only used for 'bootstrap' method).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    PairedDiffResult
    """
    if rng is None:
        rng = np.random.default_rng()

    diffs = scores[idx_a] - scores[idx_b]
    m = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    alpha = 1 - ci

    if method == "bootstrap":
        centered_diffs = diffs - mean_d
        boot_centered_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.choice(m, size=m, replace=True)
            boot_centered_means[b] = np.mean(centered_diffs[idx])
        # Shift null-centered bootstrap means back by observed mean for CI.
        boot_means = boot_centered_means + mean_d
        ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        # Bootstrap p-value (two-sided): estimate tail area under H0 by
        # centering diffs to enforce a zero-mean null distribution.
        extreme_count = np.sum(np.abs(boot_centered_means) >= abs(mean_d))
        p_value = float((extreme_count + 1) / (n_bootstrap + 1))
        test_name = f"bootstrap (n={n_bootstrap})"

    elif method == "paired_t":
        t_stat, p_value = stats.ttest_rel(scores[idx_a], scores[idx_b])
        se = std_d / np.sqrt(m)
        t_crit = stats.t.ppf(1 - alpha / 2, df=m - 1)
        ci_low = mean_d - t_crit * se
        ci_high = mean_d + t_crit * se
        p_value = float(p_value)
        test_name = "paired t-test"

    elif method == "wilcoxon":
        if np.all(diffs == 0):
            p_value = 1.0
        else:
            _, p_value = stats.wilcoxon(diffs, alternative="two-sided")
            p_value = float(p_value)
        # For Wilcoxon, use bootstrap CI anyway (Wilcoxon doesn't give one natively)
        boot_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.choice(m, size=m, replace=True)
            boot_means[b] = np.mean(diffs[idx])
        ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        test_name = "wilcoxon signed-rank"

    else:
        raise ValueError(f"Unknown method: {method}")

    return PairedDiffResult(
        template_a=label_a,
        template_b=label_b,
        mean_diff=mean_d,
        std_diff=std_d,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        test_method=test_name,
        n_inputs=m,
        per_input_diffs=diffs,
    )


def all_pairwise(
    scores: np.ndarray,
    labels: list[str],
    method: Literal["bootstrap", "paired_t", "wilcoxon"] = "bootstrap",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    rng: Optional[np.random.Generator] = None,
) -> PairwiseMatrix:
    """Compute all pairwise comparisons with multiple comparisons correction.

    Parameters
    ----------
    scores : np.ndarray
        2D score matrix of shape (N_templates, M_inputs).
    labels : list[str]
        Template labels.
    method : str
        Statistical test method.
    ci : float
        Confidence level.
    n_bootstrap : int
        Number of bootstrap resamples.
    correction : str
        Multiple comparisons correction: 'holm' (default), 'bonferroni',
        'fdr_bh', or 'none'.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

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
            )
            results[(labels[i], labels[j])] = result
            pairs.append((labels[i], labels[j]))

    # Apply multiple comparisons correction to p-values
    if correction != "none" and len(pairs) > 1:
        p_values = np.array([results[p].p_value for p in pairs])
        adjusted = _correct_pvalues(p_values, correction)
        for pair, adj_p in zip(pairs, adjusted):
            r = results[pair]
            results[pair] = PairedDiffResult(
                template_a=r.template_a,
                template_b=r.template_b,
                mean_diff=r.mean_diff,
                std_diff=r.std_diff,
                ci_low=r.ci_low,
                ci_high=r.ci_high,
                p_value=float(adj_p),
                test_method=f"{r.test_method} ({correction}-corrected)",
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
            )

    return PairwiseMatrix(labels=labels, results=results, correction_method=correction)


def vs_baseline(
    scores: np.ndarray,
    labels: list[str],
    baseline: str,
    method: Literal["bootstrap", "paired_t", "wilcoxon"] = "bootstrap",
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    rng: Optional[np.random.Generator] = None,
) -> list[PairedDiffResult]:
    """Compare all templates against a designated baseline.

    Parameters
    ----------
    scores : np.ndarray
        2D score matrix of shape (N_templates, M_inputs).
    labels : list[str]
        Template labels.
    baseline : str
        Label of the baseline template.
    method, ci, n_bootstrap, correction, rng :
        Same as all_pairwise.

    Returns
    -------
    list[PairedDiffResult]
        One result per non-baseline template, comparing it to the baseline.
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
        )
        results.append(result)

    # Apply correction
    if correction != "none" and len(results) > 1:
        p_values = np.array([r.p_value for r in results])
        adjusted = _correct_pvalues(p_values, correction)
        results = [
            PairedDiffResult(
                template_a=r.template_a,
                template_b=r.template_b,
                mean_diff=r.mean_diff,
                std_diff=r.std_diff,
                ci_low=r.ci_low,
                ci_high=r.ci_high,
                p_value=float(adj_p),
                test_method=f"{r.test_method} ({correction}-corrected)",
                n_inputs=r.n_inputs,
                per_input_diffs=r.per_input_diffs,
            )
            for r, adj_p in zip(results, adjusted)
        ]

    return results


def _correct_pvalues(
    p_values: np.ndarray,
    method: str,
) -> np.ndarray:
    """Apply multiple comparisons correction to p-values."""
    n = len(p_values)
    if n <= 1:
        return p_values.copy()

    if method == "bonferroni":
        return np.minimum(p_values * n, 1.0)

    elif method == "holm":
        order = np.argsort(p_values)
        adjusted = np.empty(n)
        cummax = 0.0
        for rank, idx in enumerate(order):
            corrected = p_values[idx] * (n - rank)
            cummax = max(cummax, corrected)
            adjusted[idx] = min(cummax, 1.0)
        return adjusted

    elif method == "fdr_bh":
        order = np.argsort(p_values)
        adjusted = np.empty(n)
        cummin = 1.0
        for rank in range(n - 1, -1, -1):
            idx = order[rank]
            corrected = p_values[idx] * n / (rank + 1)
            cummin = min(cummin, corrected)
            adjusted[idx] = min(cummin, 1.0)
        return adjusted

    else:
        raise ValueError(f"Unknown correction method: {method}")
