"""Linear Mixed Model (LMM) analysis using pymer4 / lme4.

Fits the one-way mixed model::

    score ~ template + (1 | input)

``template`` is a fixed effect (the quantity we care about); ``input``
identity is a random intercept that absorbs between-input variance.  This
is the correct model for a paired benchmark design where every template is
evaluated on every input (complete block design).

When ``R >= 3`` runs are available, scores are first collapsed to per-input
cell means before fitting (averaging over runs).  The between-run (seed)
variance decomposition is still reported separately via the existing
``SeedVarianceResult``.

Outputs are mapped to the same result types as the bootstrap path
(``PairwiseMatrix``, ``MeanAdvantageResult``, ``RankDistribution``), so
consumers of ``AnalysisBundle`` do not need to know which method was used.
The one addition is ``LMMInfo``, stored on ``AnalysisBundle.lmm_info``,
which exposes the ICC and variance components from the fitted model.

Requirements
------------
* ``pymer4`` (``pip install pymer4``)
* R with the ``lme4`` and ``emmeans`` packages installed

When to prefer LMM over bootstrap
-----------------------------------
* M inputs < ~15  — bootstrap CIs are unstable; LMM borrows strength
  from the model structure and gives better-calibrated CIs.
* You want a clean ICC decomposition of between-input vs. residual variance.
* Score distributions are sufficiently well-behaved (roughly Gaussian
  conditional on the random effect).

Limitations (Phase 1)
----------------------
* Template labels must not contain the substring `` - `` (space-dash-space),
  as this is used to parse emmeans contrast strings.
* Multi-model analysis (``MultiModelBenchmark``) is supported: LMM is run
  independently per model, exactly like the bootstrap path.
* The ``method='lmm'`` option is not compatible with ``method='bca'`` or
  ``method='auto'``; it must be specified explicitly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats

from .paired import PairedDiffResult, PairwiseMatrix
from .ranking import MeanAdvantageResult, RankDistribution
from .variance import RobustnessResult, SeedVarianceResult, robustness_metrics, seed_variance_decomposition
from .stats_utils import correct_pvalues


# ---------------------------------------------------------------------------
# LMM diagnostics data class
# ---------------------------------------------------------------------------

@dataclass
class LMMInfo:
    """Variance components and fit diagnostics from the fitted LMM.

    Attributes
    ----------
    icc : float
        Intraclass correlation coefficient: σ²_input / (σ²_input + σ²_resid).
        Fraction of total score variance explained by between-input differences.
        High ICC (> 0.5) means inputs are very heterogeneous relative to
        within-cell noise; the paired design is especially valuable here.
    sigma_input : float
        Estimated standard deviation of the input random effect (between-input SD).
    sigma_resid : float
        Estimated residual standard deviation (within-cell SD).
    n_obs : int
        Number of observations used to fit the model (N_templates × M_inputs).
    formula : str
        The model formula used.
    converged : bool
        Whether lme4 reported a successful convergence.
    """

    icc: float
    sigma_input: float
    sigma_resid: float
    n_obs: int
    formula: str
    converged: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_pymer4():
    """Import and return ``pymer4.models.Lmer``, or raise a helpful ImportError."""
    try:
        from pymer4.models import Lmer  # type: ignore[import]
        return Lmer
    except ImportError:
        raise ImportError(
            "pymer4 is required for method='lmm'. Install it with:\n"
            "    pip install pymer4\n\n"
            "pymer4 also requires R with the lme4 and emmeans packages:\n"
            "    install.packages(c('lme4', 'emmeans'))\n\n"
            "See https://eshinjolly.com/pymer4/ for full setup instructions."
        ) from None


def _col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name from *candidates* that exists in *df*.

    Raises ``KeyError`` with a helpful message if none are found.
    This handles minor API differences across pymer4 versions.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find any of {candidates} in DataFrame columns {list(df.columns)}. "
        "This may indicate a pymer4 version incompatibility. "
        "Please open an issue with the output of `model.coefs` / `model.ranef_var`."
    )


def _scores_to_long_df(
    scores_2d: np.ndarray,
    template_labels: list[str],
    input_labels: list[str],
) -> pd.DataFrame:
    """Convert an ``(N, M)`` cell-mean score matrix to a long-form DataFrame.

    Returns a DataFrame with columns ``'template'``, ``'input'``, ``'score'``.
    The template column is stored as a pandas ``Categorical`` with categories
    in the order given by *template_labels*, so lme4's treatment coding uses
    ``template_labels[0]`` as the reference level.
    """
    N, M = scores_2d.shape
    templates = np.repeat(template_labels, M)
    inputs = np.tile(input_labels, N)
    scores = scores_2d.ravel()
    df = pd.DataFrame({"template": templates, "input": inputs, "score": scores})
    # Explicit category order → first label is the reference in treatment coding.
    df["template"] = pd.Categorical(df["template"], categories=template_labels, ordered=False)
    return df


def _fit_lmm(df: pd.DataFrame, Lmer, alpha: float):
    """Fit ``score ~ template + (1|input)`` with Wald confidence intervals.

    Uses REML estimation (better for variance components) and Wald CIs
    (fast; appropriate given the balanced complete-block structure).
    """
    model = Lmer("score ~ template + (1|input)", data=df)
    model.fit(REML=True, conf_int="Wald", alpha=alpha, verbose=False)
    return model


def _extract_template_means(model, labels: list[str]) -> np.ndarray:
    """Compute fitted marginal means for each template from treatment-coded LMM.

    With R's default treatment coding the first category is the reference:

    * μ₀  = intercept
    * μᵢ  = intercept + β_i   for i > 0

    Returns shape ``(N,)``.
    """
    coefs = model.coefs
    est_col = _col(coefs, ["Estimate", "estimate", "Coef", "coef"])
    betas = coefs[est_col].values  # (N,): [intercept, β₁, …, β_{N-1}]

    N = len(labels)
    means = np.empty(N)
    means[0] = betas[0]
    means[1:] = betas[0] + betas[1:]
    return means


def _t_crit_from_coefs(coefs: pd.DataFrame, alpha: float) -> float:
    """Return the conservative t critical value from the fixed-effects DFs."""
    df_col = next(
        (c for c in ["DF", "df", "Df", "Ddf"] if c in coefs.columns), None
    )
    if df_col is not None:
        min_df = float(coefs[df_col].min())
        return float(scipy.stats.t.ppf(1 - alpha / 2, df=min_df))
    # Fall back to standard normal (conservative for large N)
    return float(scipy.stats.norm.ppf(1 - alpha / 2))


# ---------------------------------------------------------------------------
# Pairwise comparisons
# ---------------------------------------------------------------------------

def _lmm_to_pairwise(
    model,
    labels: list[str],
    cell_means_2d: np.ndarray,
    ci: float,
    correction: str,
) -> PairwiseMatrix:
    """Build a ``PairwiseMatrix`` from LMM emmeans pairwise contrasts.

    Uses ``model.post_hoc("template", p_adjust="none")`` which calls R's
    ``emmeans::contrast()`` under the hood, giving Wald CIs and Satterthwaite
    degrees of freedom for each pairwise contrast.

    Multiple-comparisons correction is applied afterwards using the same
    ``correct_pvalues()`` function used by the bootstrap path.
    """
    alpha = 1 - ci
    M = cell_means_2d.shape[1]

    _, contrasts_df = model.post_hoc(marginal_vars="template", p_adjust="none")

    # Defensive column name lookup — column names vary across pymer4 versions.
    contrast_col = _col(contrasts_df, ["Contrast", "contrast"])
    est_col      = _col(contrasts_df, ["Estimate", "estimate"])
    se_col       = _col(contrasts_df, ["SE", "se", "Std. Error", "std.error"])
    df_col       = _col(contrasts_df, ["DF", "df", "Df"])
    pval_col     = _col(contrasts_df, ["P-val", "p.value", "p value", "Pr(>|t|)", "P.Value"])

    results: dict[tuple[str, str], PairedDiffResult] = {}
    pairs: list[tuple[str, str]] = []

    for _, row in contrasts_df.iterrows():
        contrast_str = str(row[contrast_col])

        # emmeans may prefix each level with the variable name ("templateA - templateB")
        # or just use the level names ("A - B").  Strip known prefixes.
        for label in labels:
            contrast_str = contrast_str.replace(f"template{label}", label)
            contrast_str = contrast_str.replace(f"template {label}", label)

        if " - " not in contrast_str:
            warnings.warn(
                f"Unexpected contrast string from pymer4/emmeans: '{row[contrast_col]}'. "
                "Skipping. Check that template labels do not contain ' - '.",
                UserWarning,
                stacklevel=4,
            )
            continue

        label_a, label_b = (s.strip() for s in contrast_str.split(" - ", 1))

        if label_a not in labels or label_b not in labels:
            warnings.warn(
                f"Contrast '{contrast_str}' could not be matched to known labels "
                f"{labels}. Skipping.",
                UserWarning,
                stacklevel=4,
            )
            continue

        estimate = float(row[est_col])
        se       = float(row[se_col])
        df_val   = float(row[df_col])
        p_val    = float(row[pval_col])

        t_crit = float(scipy.stats.t.ppf(1 - alpha / 2, df=df_val))
        ci_low  = estimate - t_crit * se
        ci_high = estimate + t_crit * se

        idx_a = labels.index(label_a)
        idx_b = labels.index(label_b)
        per_input_diffs = cell_means_2d[idx_a] - cell_means_2d[idx_b]

        res = PairedDiffResult(
            template_a=label_a,
            template_b=label_b,
            mean_diff=estimate,
            std_diff=float(np.std(per_input_diffs, ddof=1)),
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_val,
            test_method=f"lmm wald (pymer4, df={df_val:.0f})",
            n_inputs=M,
            per_input_diffs=per_input_diffs,
            n_runs=1,  # cell means are already run-averaged
        )
        results[(label_a, label_b)] = res
        pairs.append((label_a, label_b))

    if not results:
        raise RuntimeError(
            "pymer4 post_hoc returned no usable contrasts. "
            "Check that template labels are simple strings without ' - '."
        )

    # Apply multiple-comparisons correction
    if correction != "none" and len(pairs) > 1:
        p_values = np.array([results[p].p_value for p in pairs])
        adjusted = correct_pvalues(p_values, correction)
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
                n_runs=r.n_runs,
            )

    return PairwiseMatrix(labels=labels, results=results, correction_method=correction)


# ---------------------------------------------------------------------------
# Mean advantage
# ---------------------------------------------------------------------------

def _build_advantage_contrast_matrix(N: int, ref_idx: Optional[int]) -> np.ndarray:
    """Build the (N × N) contrast matrix L for template advantages.

    Each row L[i] is a vector of coefficients such that::

        advantage[i] = L[i] @ beta

    where ``beta = [intercept, β₁, …, β_{N-1}]`` uses treatment coding
    with ``template_labels[0]`` as the reference level.

    Parameters
    ----------
    N : int
        Number of templates.
    ref_idx : int or None
        Index of the reference template, or ``None`` for grand-mean reference.
    """
    L = np.zeros((N, N))

    if ref_idx is None:
        # Grand-mean reference: advantage_i = μ_i - (1/N) * Σ_k μ_k
        #
        # μ₀ = β₀,   μⱼ = β₀ + βⱼ  (j > 0)
        # grand_mean = β₀ + (1/N) * Σ_{j>0} βⱼ
        #
        # advantage_0 = -(1/N) * Σ_{j>0} βⱼ  →  L[0, 1:] = -1/N
        # advantage_i = βᵢ - (1/N) * Σ_{j>0} βⱼ
        #             →  L[i, i] = 1 - 1/N,  L[i, j≠i, j>0] = -1/N
        L[0, 1:] = -1.0 / N
        for i in range(1, N):
            L[i, 1:] = -1.0 / N
            L[i, i]  =  1.0 - 1.0 / N
    else:
        # Specific-template reference: advantage_i = μ_i - μ_{ref}
        #
        # Cases (treatment coding, ref level = template 0):
        #   i == ref_idx               → advantage = 0   (L[i] = 0)
        #   i == 0, ref_idx > 0        → -β_{ref}
        #   i > 0,  ref_idx == 0       → β_i
        #   i > 0,  ref_idx > 0, i≠ref → β_i - β_{ref}
        for i in range(N):
            if i == ref_idx:
                continue  # all zeros
            if i == 0:
                L[i, ref_idx] = -1.0
            elif ref_idx == 0:
                L[i, i] = 1.0
            else:
                L[i, i]       =  1.0
                L[i, ref_idx] = -1.0

    return L


def _lmm_to_mean_advantage(
    model,
    labels: list[str],
    cell_means_2d: np.ndarray,
    ci: float,
    spread_percentiles: tuple[float, float],
    reference: str,
) -> MeanAdvantageResult:
    """Compute mean advantages from LMM fixed effects using the delta method.

    Point estimates are the fitted LMM marginal means (equal to raw cell-mean
    averages in the balanced design).  Confidence intervals use Wald intervals
    derived from the fixed-effects variance–covariance matrix via the delta
    method, so they correctly account for the correlated estimation of
    treatment effects.

    The intrinsic spread bands (``spread_low`` / ``spread_high``) are still
    computed from raw cell-mean advantages, exactly as in the bootstrap path.

    Notes
    -----
    ``n_bootstrap=0`` on the returned ``MeanAdvantageResult`` signals that
    these are parametric Wald intervals, not bootstrap intervals.
    """
    N = len(labels)
    alpha = 1 - ci

    coefs = model.coefs
    est_col = _col(coefs, ["Estimate", "estimate", "Coef", "coef"])
    betas = coefs[est_col].values  # (N,)

    # Fitted template means
    template_means = np.empty(N)
    template_means[0] = betas[0]
    template_means[1:] = betas[0] + betas[1:]

    # Reference
    if reference == "grand_mean":
        ref_value = template_means.mean()
        ref_idx   = None
        ref_label = "grand_mean"
    else:
        ref_idx   = labels.index(reference)
        ref_value = template_means[ref_idx]
        ref_label = reference

    mean_advantages = template_means - ref_value  # (N,)

    # --- Wald CIs via delta method -----------------------------------------
    # vcov: (N × N) variance-covariance matrix of the fixed effects.
    # pymer4 exposes this as model.vcov (pandas DataFrame or numpy array).
    vcov = np.asarray(model.vcov)   # (N, N)

    L = _build_advantage_contrast_matrix(N, ref_idx)   # (N, N)

    # Variance of each advantage: var(L[i] @ beta) = L[i] @ vcov @ L[i].T
    cov_adv  = L @ vcov @ L.T          # (N, N)
    var_adv  = np.diag(cov_adv)        # (N,)
    se_adv   = np.sqrt(np.maximum(var_adv, 0.0))

    t_crit   = _t_crit_from_coefs(coefs, alpha)
    ci_low   = mean_advantages - t_crit * se_adv
    ci_high  = mean_advantages + t_crit * se_adv

    # --- Raw cell-mean advantages for spread bands -------------------------
    if reference == "grand_mean":
        cell_ref = cell_means_2d.mean(axis=0)     # (M,)
    else:
        cell_ref = cell_means_2d[ref_idx]          # (M,)

    raw_advantages = cell_means_2d - cell_ref[np.newaxis, :]  # (N, M)
    spread_low  = np.percentile(raw_advantages, spread_percentiles[0], axis=1)
    spread_high = np.percentile(raw_advantages, spread_percentiles[1], axis=1)

    return MeanAdvantageResult(
        labels=labels,
        mean_advantages=mean_advantages,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        spread_low=spread_low,
        spread_high=spread_high,
        reference=ref_label,
        per_input_advantages=raw_advantages,
        n_bootstrap=0,           # 0 = parametric Wald, not bootstrap
        spread_percentiles=spread_percentiles,
    )


# ---------------------------------------------------------------------------
# Rank distribution
# ---------------------------------------------------------------------------

def _extract_variance_components(model) -> tuple[float, float]:
    """Return (sigma_input, sigma_resid) from the fitted LMM.

    Handles minor differences in how pymer4 labels the rows of
    ``model.ranef_var`` across versions.
    """
    ranef_var = model.ranef_var
    var_col  = _col(ranef_var, ["Var", "variance", "Variance", "var"])
    name_col = _col(ranef_var, ["Name", "name", "Groups", "groups", "grp"])

    sigma_input = 0.0
    sigma_resid = 0.0

    for _, row in ranef_var.iterrows():
        name    = str(row[name_col]).lower().strip()
        var_val = float(row[var_col])
        if "residual" in name or "error" in name or name in {"", "resid"}:
            sigma_resid = np.sqrt(max(var_val, 0.0))
        else:
            # All non-residual rows are random-effect groups; we expect only
            # one (input), but accumulate if multiple are present (future use).
            sigma_input = np.sqrt(max(var_val, 0.0))

    return sigma_input, sigma_resid


def _lmm_to_rank_dist(
    model,
    labels: list[str],
    cell_means_2d: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
) -> RankDistribution:
    """Parametric rank distribution via simulation from the fitted LMM.

    At each iteration:

    1. Draw M new input random effects ~ N(0, σ²_input).
    2. Draw residuals ~ N(0, σ²_resid) for each (template, input) cell.
    3. Rank templates by their mean over the M simulated inputs.

    This propagates both the estimation uncertainty (via the fixed-effect
    means) and the structural variance (σ²_input, σ²_resid) into the rank
    distribution, making it more informative than a bootstrap on cell means
    when M is small.
    """
    N = len(labels)
    M = cell_means_2d.shape[1]

    template_means           = _extract_template_means(model, labels)
    sigma_input, sigma_resid = _extract_variance_components(model)

    rank_counts = np.zeros((N, N), dtype=np.int64)

    for _ in range(n_sim):
        input_effects = (
            rng.normal(0.0, sigma_input, size=M)
            if sigma_input > 0 else np.zeros(M)
        )
        resid = (
            rng.normal(0.0, sigma_resid, size=(N, M))
            if sigma_resid > 0 else np.zeros((N, M))
        )
        sim_scores = template_means[:, None] + input_effects[None, :] + resid
        order = np.argsort(-sim_scores.mean(axis=1))
        for rank, tidx in enumerate(order):
            rank_counts[tidx, rank] += 1

    rank_probs    = rank_counts / n_sim
    expected_ranks = (rank_probs * np.arange(1, N + 1)).sum(axis=1)
    p_best        = rank_probs[:, 0]

    return RankDistribution(
        labels=labels,
        rank_probs=rank_probs,
        expected_ranks=expected_ranks,
        p_best=p_best,
        n_bootstrap=n_sim,
    )


# ---------------------------------------------------------------------------
# LMM diagnostics
# ---------------------------------------------------------------------------

def _build_lmm_info(model, N: int, M: int) -> LMMInfo:
    """Extract ``LMMInfo`` from a fitted pymer4 ``Lmer`` model."""
    sigma_input, sigma_resid = _extract_variance_components(model)

    var_input = sigma_input ** 2
    var_resid = sigma_resid ** 2
    total_var = var_input + var_resid
    icc = var_input / total_var if total_var > 0 else 0.0

    # pymer4 stores convergence warnings in model.warnings (list of strings)
    converged = True
    if hasattr(model, "warnings") and model.warnings:
        converged = not any(
            "convergence" in str(w).lower() or "singular" in str(w).lower()
            for w in model.warnings
        )

    return LMMInfo(
        icc=icc,
        sigma_input=sigma_input,
        sigma_resid=sigma_resid,
        n_obs=N * M,
        formula="score ~ template + (1|input)",
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def lmm_analyze(
    result,
    *,
    reference: str = "grand_mean",
    ci: float = 0.95,
    correction: str = "holm",
    spread_percentiles: tuple[float, float] = (10, 90),
    failure_threshold: Optional[float] = None,
    n_sim: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[PairwiseMatrix, MeanAdvantageResult, RankDistribution,
           RobustnessResult, Optional[SeedVarianceResult], LMMInfo]:
    """Run the full LMM analysis pipeline on a ``BenchmarkResult``.

    Fits ``score ~ template + (1|input)`` on the cell-mean scores and maps
    the model output to the same result types as the bootstrap path.

    Parameters
    ----------
    result : BenchmarkResult
        The benchmark data to analyse.
    reference : str
        Reference for mean advantage: ``'grand_mean'`` or a template label.
    ci : float
        Confidence level for Wald intervals (default 0.95).
    correction : str
        Multiple-comparisons correction: ``'holm'`` (default),
        ``'bonferroni'``, ``'fdr_bh'``, or ``'none'``.
    spread_percentiles : tuple[float, float]
        Percentiles for the intrinsic variance band (default ``(10, 90)``).
    failure_threshold : float, optional
        Threshold for failure-rate computation in robustness metrics.
    n_sim : int
        Number of parametric simulations for the rank distribution
        (default 10,000).  Analogous to ``n_bootstrap`` in the bootstrap path.
    rng : np.random.Generator, optional
        Random number generator for the rank simulation.

    Returns
    -------
    tuple
        ``(pairwise, mean_adv, rank_dist, robustness, seed_var, lmm_info)``
        where types match those returned by the bootstrap analysis path.

    Raises
    ------
    ImportError
        If pymer4 is not installed.
    RuntimeError
        If the model fails to converge or returns unusable contrasts.
    """
    Lmer = _require_pymer4()

    if rng is None:
        rng = np.random.default_rng()

    N = result.n_templates
    M = result.n_inputs

    if M < 5:
        warnings.warn(
            f"LMM analysis with only M={M} inputs may be unreliable. "
            "Consider using the default bootstrap method (method='auto') "
            "or collecting more benchmark inputs.",
            UserWarning,
            stacklevel=3,
        )

    # Use cell means for model fitting; keep run scores for seed_var.
    cell_means_2d = result.get_2d_scores()   # (N, M)
    run_scores    = result.get_run_scores()  # (N, M, R)
    labels        = result.template_labels
    inputs        = result.input_labels

    # Fit the LMM
    df    = _scores_to_long_df(cell_means_2d, labels, inputs)
    alpha = 1.0 - ci
    model = _fit_lmm(df, Lmer, alpha)

    if not model.fitted:
        raise RuntimeError(
            "LMM failed to fit. Check that scores have sufficient variance "
            "across inputs and that template labels are well-formed."
        )

    # Build all analysis components
    pairwise  = _lmm_to_pairwise(model, labels, cell_means_2d, ci, correction)
    mean_adv  = _lmm_to_mean_advantage(
        model, labels, cell_means_2d, ci, spread_percentiles, reference
    )
    rank_dist = _lmm_to_rank_dist(model, labels, cell_means_2d, n_sim, rng)
    robustness = robustness_metrics(run_scores, labels, failure_threshold=failure_threshold)

    seed_var: Optional[SeedVarianceResult] = None
    if result.is_seeded:
        seed_var = seed_variance_decomposition(run_scores, labels)

    lmm_info = _build_lmm_info(model, N, M)

    if not lmm_info.converged:
        warnings.warn(
            "The LMM optimizer reported a convergence warning or singular fit. "
            "Results may be unreliable. Consider using the bootstrap method "
            "(method='auto') or simplifying the model.",
            UserWarning,
            stacklevel=3,
        )

    return pairwise, mean_adv, rank_dist, robustness, seed_var, lmm_info
