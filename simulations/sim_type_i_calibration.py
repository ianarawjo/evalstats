#!/usr/bin/env python3
"""
sim_type_i_calibration.py — Type I error calibration study for PPI-corrected tests.

Tests eight PPI-corrected hypothesis tests across a broad parameter sweep to check
whether the corrected p-value stays near α = 0.05 under H₀:

  ttest       independent mean difference (two groups)
  mannwhitney independent P(X > Y) − 0.5 (two groups)
  wilcoxon    paired median difference (same subjects, two conditions)
  anova_ind   one-way ANOVA, between-subjects (three groups)
  anova_rep   one-way ANOVA, within-subjects / repeated-measures (three conditions)
  friedman    Friedman test, within-subjects rank variance (three conditions) —
              reuses the same repeated-measures data as anova_rep, since
              Friedman is the rank-based analog of repeated-measures ANOVA.
  kruskal     Kruskal-Wallis test, independent-groups pairwise stochastic
              dominance (three groups) — reuses the same independent-groups
              data as anova_ind, since Kruskal-Wallis is the rank-based
              analog of independent-groups one-way ANOVA. Unlike the other
              omnibus tests, its corrected p-value is a Wald test over the
              joint bootstrap covariance of pairwise effects, not a
              closed-form null-variance approximation.
  lmm         Linear mixed model, score ~ template + (1|input) (three
              conditions) — reuses the same repeated-measures data as
              anova_rep/friedman, since that data is already generated from
              exactly this model (a per-input random intercept plus
              residual plus a template fixed effect). Unlike anova_rep/
              friedman, the correction is a single closed-form rectified-GLS
              solve with a cluster-robust sandwich covariance (see
              evalstats.core.mixed_effects._ppi_lmm_fixed_effects) rather
              than a bootstrap-then-closed-form-p-value combination — LMM
              fixed effects are an M-estimator, so the "plug-in" recipe used
              by every other test here isn't a valid correction for them.
              Type I calibration only: this test is intentionally excluded
              from the bias/CI-coverage effect-size check (its headline
              "variance of template means" estimand is a quadratic form in
              the fixed effects and deliberately has no CI; see
              es.tests.lmm()'s docstring). Single fixed factor + a single
              (1|input) random effect only — no run-level random effect, no
              factorial (two-factor) designs yet.

Factors swept (one at a time from a fixed baseline):
  distribution   normal, binary (0/1), likert (1–5), skewed (log-normal)
  sample size    n = 30, 60, 200, 400
  balance        group size ratio 1:1, 2:1, 4:1  (independent tests only)
  label frac     3%, 8%, 20%, 40% of items have human labels
  LLM noise      additive iid noise σ = 0.0, 0.10, 0.35, 0.70
  bias type      none / constant (equal for all groups) / differential (one group)
  heteroskedastic  different LLM noise SD per group
  stress         extreme combinations (small+sparse, large+noisy, etc.)

For each scenario × test × rep we draw independent data under H₀ and record
whether the corrected p-value falls below α.  Type I rate = fraction rejected.
Under H₀ the ideal rate is exactly α; we flag with ● when rate > 2σ above α.

A second, complementary check (printed after the Type I table) tracks BIAS and
CI COVERAGE of the corrected point estimate itself — i.e. not "does p stay
calibrated" but "is the estimate centered at the truth, and does its CI cover
that truth at the nominal rate." For ttest/ttest_welch/mw/wilcoxon/kruskal this
is free: those already bootstrap in the main sweep, so we just read estimate/CI
off the result object that's already computed. anova_ind/anova_rep/friedman use
fast closed-form p-value-only functions in the main sweep instead (to avoid
slowing it down), so their bias/coverage check runs as a separate, smaller pass
(--effect-reps, default 200) that calls their bootstrap-based scalar estimate
function directly — the same one anova_oneway()/friedman() already call for
corrected_estimate in normal use. The "truth" each estimate is compared against
is not always 0: anova_ind/anova_rep/friedman's "between-condition variance"
estimand is a variance OF GROUP MEANS, which has a small positive expectation
under H0 purely from finite-sample noise in the means (the textbook "E[MS_between]
= σ² under H0" fact) — so the truth is estimated by Monte Carlo on pure-truth
data at each scenario's exact sample sizes (_gold_null_values) rather than
assumed to be 0. Use --no-effect-check to skip this second check entirely.

A third, separate mode (--effect-size) checks STATISTICAL POWER instead of
Type I error: every scenario in the SAME grid normally draws every group/
condition from an identical true distribution (only the LLM-judge artifacts —
bias_type, llm_noise — ever differ between groups), so nothing in the default
sweep ever tests whether the corrected test can detect a real effect, only
whether it avoids false positives when there isn't one. --effect-size N
injects a real, monotonic mean shift of size N into the truth itself (group/
condition 0 unshifted, 1 gets +N, 2 gets +2N) across the exact same 31-scenario
grid, and the printed table becomes "STATISTICAL POWER" — rejection rate is now
the GOOD outcome (low power is flagged, not high). This automatically disables
the bias/coverage/width effect-size check (its gold-reference null assumes H0,
which no longer holds); --plot switches to a dedicated power heatmap/scatter
(_plot_power_results) instead of the Type I one.

The internal PPI functions (_ppi_two_sample, _ppi_paired_arrays, etc.) are called
directly to skip the validate_alignment overhead (~360ms/call) — we are testing
statistical calibration, not the pipeline UX.

Usage:
    python simulations/sim_type_i_calibration.py
    python simulations/sim_type_i_calibration.py --reps 200 --n-boot 100   # quick
    python simulations/sim_type_i_calibration.py --reps 1000 --jobs 8      # thorough
    python simulations/sim_type_i_calibration.py --plot                    # save + show figures
    python simulations/sim_type_i_calibration.py --out-csv simulations/out/type_i.csv
    python simulations/sim_type_i_calibration.py --tests friedman            # isolate one test
    python simulations/sim_type_i_calibration.py --tests friedman anova_rep  # isolate a subset
    python simulations/sim_type_i_calibration.py --effect-reps 500          # more effect-size reps
    python simulations/sim_type_i_calibration.py --no-effect-check          # Type I only, fastest
    python simulations/sim_type_i_calibration.py --effect-size 0.5          # power sweep instead

--tests runs the exact same scenario sweep ("the official test") but skips the
PPI computation for any test not listed, so isolating a single test is both a
display filter and a real speedup (the other tests' bootstrap/statsmodels work
is never run).
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.tests import (
        _ppi_two_sample,
        _ppi_paired_arrays,
        _p_x_gt_y_midrank,
        _ppi_anova_independent_p_value,
        _ppi_anova_repeated_p_value,
        _ppi_friedman_p_value,
        _ppi_kruskal_wallis_pairwise,
        _ppi_anova_independent,
        _ppi_anova_repeated,
        _ppi_friedman,
        _ppi_lmm_p_value,
        _anova_between_variance_from_groups,
        _repeated_condition_variance,
        _friedman_rank_variance,
        _kw_pairwise_thetas,
    )
    from evalstats.core.mixed_effects import _fit_lmm_general, _get_fe_vcov_sm


# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA = 0.05
TEST_NAMES = [
    "ttest", "ttest_welch", "mw", "wilcoxon", "anova_ind", "anova_rep", "friedman", "kruskal",
    "lmm", "lmm_factorial", "lmm_runs",
]
# Short column headers for the printed table, keyed by canonical test name so
# a --tests subset (in any order) still renders correctly.
TEST_SHORT_NAMES = {
    "ttest": "ttest", "ttest_welch": "ttest_w", "mw": "mw", "wilcoxon": "wilcoxon",
    "anova_ind": "anova_i", "anova_rep": "anova_r", "friedman": "fried", "kruskal": "kw",
    "lmm": "lmm", "lmm_factorial": "lmm_fact", "lmm_runs": "lmm_runs",
}
# lmm and its variants are Type-I-only: their headline estimand has no valid
# CI by design (a quadratic form in the fixed effects — see
# es.tests.lmm()'s docstring), so they're excluded from the bias/coverage
# effect-size check (_gold_null_values, effect_results, _print_effect_table)
# entirely, not just given a 0.0 floor.
_LMM_VARIANTS = ("lmm", "lmm_factorial", "lmm_runs")
_EFFECT_CHECK_TESTS = [t for t in TEST_NAMES if t not in _LMM_VARIANTS]
# Number of independent nested LLM runs per (factor-combo, input) cell for
# lmm_runs, testing the GLS-sufficiency run-collapsing path in
# _ppi_lmm_fixed_effects (see evalstats/core/mixed_effects.py).
_LMM_RUNS_R = 3
# 2x2 crossed fixed-factor design for lmm_factorial: factor combos in the
# same order as the 4 groups built for it in _run_one.
_LMM_FACTORIAL_FACTORS = pd.DataFrame({
    "f1": ["a", "a", "b", "b"],
    "f2": ["x", "y", "x", "y"],
})
_SIGMA_TRUTH = 1.0   # within-group truth SD (normal / likert / skewed)
_SIGMA_SUB   = 0.7   # between-subject SD for paired/repeated designs
_SIGMA_COND  = 0.6   # within-subject residual SD (paired/repeated)
_MU_CONT     = 3.0   # null mean for continuous distributions
_MU_BIN      = 0.5   # null probability for binary (Bernoulli)
_MIN_LAB     = 15    # minimum labeled items per group / condition


# ── Scenario definition ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Scenario:
    name: str
    tag: str                         # factor group for output grouping
    dist: str = "normal"             # 'normal', 'binary', 'likert', 'skewed'
    n: int = 100                     # group-A / n_subjects size
    n2: Optional[int] = None         # group-B size (None → same as n)
    n3: Optional[int] = None         # group-C size (None → same as n, anova only)
    label_frac: float = 0.20         # fraction of items with human labels
    llm_noise: float = 0.20          # LLM noise SD for group A / all groups
    llm_noise2: Optional[float] = None  # group-B LLM noise SD
    llm_noise3: Optional[float] = None  # group-C LLM noise SD
    bias_type: str = "differential"  # 'none', 'constant', 'differential'
    bias_delta: float = 0.30         # group-A inflation for 'differential'
    bias_const: float = 0.40         # shared inflation for 'constant'
    effect_size: float = 0.0         # TRUE mean shift injected into truth itself (not an LLM
                                      # artifact like bias_*) — 0.0 reproduces the Type I sweep
                                      # exactly; >0 turns the same grid into a power sweep (see
                                      # --effect-size). Monotonic across groups/conditions:
                                      # group/condition 0 unshifted, 1 gets +effect_size, 2 (if
                                      # present) gets +2*effect_size.


def _build_scenarios() -> list[Scenario]:
    B: dict = dict(
        dist="normal", n=100, n2=None, n3=None,
        label_frac=0.20, llm_noise=0.20, llm_noise2=None, llm_noise3=None,
        bias_type="differential", bias_delta=0.30, bias_const=0.40,
    )

    S: list[Scenario] = []

    def sc(name, tag, **kw):
        return Scenario(name=name, tag=tag, **{**B, **kw})

    # ── Distribution ──────────────────────────────────────────────────────────
    for dist in ["normal", "likert", "skewed"]:  # removed "binary" since current stats tests (MWU, ttest, etc) aren't designed for it
        S.append(sc(f"dist·{dist}", "distribution", dist=dist))

    # ── Sample size ───────────────────────────────────────────────────────────
    for n in [60, 100, 200, 400]:
        S.append(sc(f"n={n}", "sample_size", n=n))

    # ── Balance (total n ≈ 200; independent tests only) ───────────────────────
    S.append(sc("balance·1:1", "balance", n=100, n2=100, n3=100))
    S.append(sc("balance·2:1", "balance", n=67,  n2=133, n3=100))
    S.append(sc("balance·4:1", "balance", n=40,  n2=160, n3=100))

    # ── Label fraction ────────────────────────────────────────────────────────
    for lab in [0.05, 0.10, 0.20, 0.40]:
        S.append(sc(f"lab·{lab:.0%}", "label_frac", label_frac=lab))

    # ── LLM noise ────────────────────────────────────────────────────────────
    for noise in [0.0, 0.10, 0.35, 0.70]:
        S.append(sc(f"noise·{noise}", "llm_noise", llm_noise=noise))

    # ── Bias type ─────────────────────────────────────────────────────────────
    for bt in ["none", "constant", "differential"]:
        S.append(sc(f"bias·{bt}", "bias_type", bias_type=bt))

    # ── Heteroskedastic LLM noise ─────────────────────────────────────────────
    S.append(sc("hetero·mild",    "heteroskedastic",
               llm_noise=0.05, llm_noise2=0.50, llm_noise3=0.25))
    S.append(sc("hetero·extreme", "heteroskedastic",
               llm_noise=0.02, llm_noise2=0.80, llm_noise3=0.40))

    # ── Stress tests ──────────────────────────────────────────────────────────
    S.append(sc("stress·small+sparse",   "stress", n=30,  label_frac=0.07))
    S.append(sc("stress·large+noisy",    "stress", n=300, llm_noise=0.70))
    S.append(sc("stress·unbal+diff",     "stress", n=40,  n2=200, label_frac=0.08))
    S.append(sc("stress·tiny_lab",       "stress", n=200, label_frac=0.02))

    # ── Interaction stress tests (multi-factor, not one-at-a-time) ──────────
    S.append(sc(
        "interact·small+unbal+hetero+diff",
        "interaction",
        n=30,
        n2=120,
        n3=80,
        label_frac=0.05,
        llm_noise=0.05,
        llm_noise2=0.70,
        llm_noise3=0.35,
        bias_type="differential",
    ))
    S.append(sc(
        "interact·small+sparse+noisy+const",
        "interaction",
        n=40,
        label_frac=0.05,
        llm_noise=0.80,
        bias_type="constant",
    ))
    S.append(sc(
        "interact·large+sparse+hetero+diff",
        "interaction",
        n=250,
        n2=400,
        n3=300,
        label_frac=0.03,
        llm_noise=0.02,
        llm_noise2=0.50,
        llm_noise3=0.90,
        bias_type="differential",
    ))
    S.append(sc(
        "interact·mid+extreme-hetero+none",
        "interaction",
        n=100,
        n2=150,
        n3=60,
        label_frac=0.08,
        llm_noise=0.01,
        llm_noise2=0.90,
        llm_noise3=0.40,
        bias_type="none",
    ))

    return S


SCENARIOS: list[Scenario] = _build_scenarios()


def _validate_scenarios(scenarios: list[Scenario]) -> None:
    """Fail fast when a scenario cannot satisfy the n_lab >= _MIN_LAB invariant."""
    for sc in scenarios:
        n2 = sc.n2 if sc.n2 is not None else sc.n
        n3 = sc.n3 if sc.n3 is not None else sc.n
        smallest_n = min(sc.n, n2, n3)
        if smallest_n < _MIN_LAB:
            raise ValueError(
                f"Scenario {sc.name!r} has group size {smallest_n}, "
                f"which violates n_lab >= {_MIN_LAB}."
            )


_validate_scenarios(SCENARIOS)


# ── Data generation ───────────────────────────────────────────────────────────

def _mu_null(dist: str) -> float:
    return _MU_BIN if dist == "binary" else _MU_CONT


def _sample_truth(dist: str, n: int, rng: np.random.Generator) -> np.ndarray:
    mu = _mu_null(dist)
    if dist == "normal":
        return rng.normal(mu, _SIGMA_TRUTH, n)
    if dist == "binary":
        return rng.binomial(1, mu, n).astype(float)
    if dist == "likert":
        return np.clip(np.round(rng.normal(mu, _SIGMA_TRUTH, n)), 1.0, 5.0)
    if dist == "skewed":
        sigma_log = 0.70
        mu_log = np.log(mu) - sigma_log ** 2 / 2
        return rng.lognormal(mu_log, sigma_log, n)
    raise ValueError(f"Unknown dist: {dist!r}")


def _biases(sc: Scenario) -> tuple[float, float, float]:
    if sc.bias_type == "none":
        return 0.0, 0.0, 0.0
    if sc.bias_type == "constant":
        return sc.bias_const, sc.bias_const, sc.bias_const
    if sc.bias_type == "differential":
        return sc.bias_delta, 0.0, 0.0
    raise ValueError(f"Unknown bias_type: {sc.bias_type!r}")


def _biases_4(sc: Scenario) -> tuple[float, float, float, float]:
    """Same bias_type semantics as ``_biases`` but for lmm_factorial's 4
    groups (2x2 crossed factors) instead of 3."""
    if sc.bias_type == "none":
        return 0.0, 0.0, 0.0, 0.0
    if sc.bias_type == "constant":
        return (sc.bias_const,) * 4
    if sc.bias_type == "differential":
        return sc.bias_delta, 0.0, 0.0, 0.0
    raise ValueError(f"Unknown bias_type: {sc.bias_type!r}")


def _llm(truth: np.ndarray, bias: float, noise_sd: float, rng: np.random.Generator) -> np.ndarray:
    if noise_sd == 0.0:
        return truth + bias
    return truth + bias + rng.normal(0.0, noise_sd, len(truth))


def _labels_independent(truth: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Independent random label positions (for ttest/mw/anova_ind groups)."""
    n = len(truth)
    if n < _MIN_LAB:
        raise ValueError(f"Need n >= {_MIN_LAB} to enforce n_lab >= {_MIN_LAB}; got n={n}")
    n_lab = min(n, max(_MIN_LAB, int(round(n * frac))))
    lab = np.full(n, np.nan)
    idx = rng.choice(n, n_lab, replace=False)
    lab[idx] = truth[idx]
    return lab


def _labels_shared(truths: list[np.ndarray], frac: float,
                   rng: np.random.Generator) -> list[np.ndarray]:
    """Same subject positions labeled across all conditions (paired/repeated)."""
    n = len(truths[0])
    if n < _MIN_LAB:
        raise ValueError(f"Need n >= {_MIN_LAB} to enforce n_lab >= {_MIN_LAB}; got n={n}")
    n_lab = min(n, max(_MIN_LAB, int(round(n * frac))))
    idx = rng.choice(n, n_lab, replace=False)
    labs = []
    for truth in truths:
        lab = np.full(n, np.nan)
        lab[idx] = truth[idx]
        labs.append(lab)
    return labs


def _uncorrected_anova_independent_p_value(groups: list[np.ndarray]) -> float:
    return float(_scipy_stats.f_oneway(*groups).pvalue)


def _uncorrected_anova_repeated_p_value(groups: list[np.ndarray]) -> float:
    try:
        import pandas as pd
        from statsmodels.stats.anova import AnovaRM
    except Exception as exc:
        raise ImportError(
            "Repeated-measures ANOVA requires pandas and statsmodels."
        ) from exc

    k = len(groups)
    n_subjects = len(groups[0])
    stacked = np.column_stack(groups)
    df_long = pd.DataFrame(
        {
            "subject": np.repeat(np.arange(n_subjects), k),
            "condition": np.tile(np.arange(k), n_subjects),
            "score": stacked.reshape(-1),
        }
    )
    rm = AnovaRM(df_long, depvar="score", subject="subject", within=["condition"]).fit()
    return float(rm.anova_table.iloc[0]["Pr > F"])


def _uncorrected_friedman_p_value(groups: list[np.ndarray]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(_scipy_stats.friedmanchisquare(*groups).pvalue)


def _uncorrected_kruskal_p_value(groups: list[np.ndarray]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(_scipy_stats.kruskal(*groups).pvalue)


def _uncorrected_lmm_p_value(groups: list[np.ndarray], factors=None) -> float:
    """Uncorrected (LLM-only) Wald F-test for the fixed effects of
    ``score ~ <fixed factors> + (1|input)``, fit via statsmodels MixedLM (REML).

    Mirrors ``es.tests.lmm()``'s uncorrected branch directly (rather than
    calling the public API) to skip its array validation / extra-dict
    bookkeeping, the same way ``_uncorrected_anova_repeated_p_value`` above
    re-fits ``AnovaRM`` directly instead of calling ``anova_oneway()``. Uses
    ``_fit_lmm_general`` (single helper backing single-factor, multi-factor,
    and nested-run groups alike) so this one function covers the lmm,
    lmm_factorial, and lmm_runs calibration variants.
    """
    k = len(groups)
    template_labels = [f"T{i}" for i in range(k)]
    sm_result, _df_full, _X_row, _R = _fit_lmm_general(groups, template_labels, factors)

    beta = sm_result.fe_params.to_numpy()
    cov = _get_fe_vcov_sm(sm_result)
    df1 = k - 1
    df2 = float(sm_result.df_resid)
    beta_t, cov_t = beta[1:], cov[1:, 1:]
    wald = float(beta_t @ np.linalg.solve(cov_t, beta_t))
    f_stat = wald / df1
    return float(_scipy_stats.f.sf(f_stat, df1, df2)) if f_stat > 0 else 1.0


def _gold_null_values(sc: Scenario, n_mc: int = 3000, seed: int = 0) -> dict[str, float]:
    """Monte Carlo expectation of each test's RAW estimand on pure-truth
    (no LLM bias, no LLM noise) data at this scenario's exact sample sizes —
    used as the effect-size check's "true null value" instead of a hardcoded 0.

    Most estimands here (mean difference, P(X>Y)−0.5, paired median
    difference) ARE exactly 0 in expectation regardless of sample size, so
    this just confirms that empirically. The "between-condition variance"
    estimands anova_ind/anova_rep/friedman report as corrected_estimate are
    different in kind: they are sample variances OF GROUP MEANS, and even
    when every group is drawn from an identical distribution, that quantity
    has a small POSITIVE expectation purely from finite-sample noise in the
    means themselves — the textbook "E[MS_between] = σ² under H0" fact. That
    floor shrinks like O(1/N) but is not 0 at finite N, so comparing against
    a fixed 0 would misreport a real, expected statistical property as bias.
    Estimating the floor by direct Monte Carlo (rather than a closed-form
    derivation per distribution/design) is what keeps this robust across the
    normal/likert/skewed/binary scenarios and repeated-vs-independent designs.
    """
    rng = np.random.default_rng(seed)
    n1 = sc.n
    n2 = sc.n2 if sc.n2 is not None else sc.n
    n3 = sc.n3 if sc.n3 is not None else sc.n

    diffs2 = np.empty(n_mc)
    thetas2 = np.empty(n_mc)
    for i in range(n_mc):
        a = _sample_truth(sc.dist, n1, rng)
        b = _sample_truth(sc.dist, n2, rng)
        diffs2[i] = a.mean() - b.mean()
        thetas2[i] = _p_x_gt_y_midrank(a, b) - 0.5

    meds = np.empty(n_mc)
    for i in range(n_mc):
        base = rng.normal(_mu_null(sc.dist), _SIGMA_SUB, n1)
        if sc.dist == "binary":
            p_sub = rng.uniform(0.2, 0.8, n1)
            x = rng.binomial(1, p_sub, n1).astype(float)
            y = rng.binomial(1, p_sub, n1).astype(float)
        else:
            x = base + rng.normal(0.0, _SIGMA_COND, n1)
            y = base + rng.normal(0.0, _SIGMA_COND, n1)
        meds[i] = float(np.median(x - y))

    bv = np.empty(n_mc)
    for i in range(n_mc):
        a3 = _sample_truth(sc.dist, n1, rng)
        b3 = _sample_truth(sc.dist, n2, rng)
        c3 = _sample_truth(sc.dist, n3, rng)
        bv[i] = _anova_between_variance_from_groups([a3, b3, c3])

    rv = np.empty(n_mc)
    frv = np.empty(n_mc)
    for i in range(n_mc):
        if sc.dist == "binary":
            p_sub3 = rng.uniform(0.2, 0.8, n1)
            A = rng.binomial(1, p_sub3, n1).astype(float)
            B = rng.binomial(1, p_sub3, n1).astype(float)
            C = rng.binomial(1, p_sub3, n1).astype(float)
        else:
            base3 = rng.normal(_mu_null(sc.dist), _SIGMA_SUB, n1)
            A = base3 + rng.normal(0.0, _SIGMA_COND, n1)
            B = base3 + rng.normal(0.0, _SIGMA_COND, n1)
            C = base3 + rng.normal(0.0, _SIGMA_COND, n1)
        mat = np.column_stack([A, B, C])
        rv[i] = _repeated_condition_variance(mat)
        frv[i] = _friedman_rank_variance(mat)

    return {
        "ttest": float(diffs2.mean()),
        "ttest_welch": float(diffs2.mean()),
        "mw": float(thetas2.mean()),
        "wilcoxon": float(meds.mean()),
        "anova_ind": float(bv.mean()),
        "anova_rep": float(rv.mean()),
        "friedman": float(frv.mean()),
        "kruskal": 0.5,  # exact by symmetry (P(X>Y)=0.5 under H0 at any n) — no floor effect
    }


# ── Worker ────────────────────────────────────────────────────────────────────

def _run_one(args: tuple) -> tuple[
    int, dict[str, bool | None], dict[str, bool | None], dict[str, tuple[float, float, float, float]]
]:
    """Process one (scenario_idx, seed, n_boot, active_tests, effect_size_override) tuple.

    ``effect_size_override``: None uses the scenario's own ``effect_size``
    field (0.0 for every scenario in the Type I sweep); a float overrides it
    for every scenario uniformly — this is how --effect-size turns the exact
    same grid into a power sweep without needing a second scenario list.

    Returns (sc_idx, corrected_results, uncorrected_results, effect_results).
    Only tests in ``active_tests`` are computed — others are simply absent
    from the result dicts (not None; None is reserved for "computed but
    failed/unavailable"). None means the p-value could not be computed
    (e.g. zero labeled items).

    ``effect_results`` piggybacks on bootstrap calls this function ALREADY
    makes for ttest/ttest_welch/mw/wilcoxon/kruskal (all five call a
    ``correct()``-based PPI function whose result object already carries
    ``estimate``/``ci_low``/``ci_high`` — we just weren't reading those
    fields before). It does NOT cover anova_ind/anova_rep/friedman, which
    deliberately use closed-form p-value-only functions here to stay fast;
    their effect-size check runs as a separate, smaller pass — see
    ``_run_one_effect_anova``. Values are returned RAW (not re-centered):
    each test's true null value isn't always 0 at finite sample size (the
    "between-condition variance" estimands have a small positive floor from
    sampling noise in group means, and kruskal's raw θ is centered on 0.5,
    not 0) — see ``_gold_null_values``, which the caller uses to re-center
    before computing bias/coverage.
    """
    sc_idx, seed, n_boot, active_tests, effect_size_override = args
    sc: Scenario = SCENARIOS[sc_idx]
    rng = np.random.default_rng(seed)

    n1 = sc.n
    n2 = sc.n2 if sc.n2 is not None else sc.n
    n3 = sc.n3 if sc.n3 is not None else sc.n
    noise1 = sc.llm_noise
    noise2 = sc.llm_noise2 if sc.llm_noise2 is not None else sc.llm_noise
    noise3 = sc.llm_noise3 if sc.llm_noise3 is not None else sc.llm_noise
    bias_a, bias_b, bias_c = _biases(sc)

    # effect_size injects a REAL mean shift into the truth itself (monotonic:
    # group/condition 0 unshifted, 1 gets +effect_size, 2 gets +2*effect_size)
    # — unlike bias_a/b/c, which are LLM-judge artifacts only. 0.0 (the Type I
    # sweep's default) makes every line below a no-op; >0 turns this into a
    # power sweep (see --effect-size). Skipped for dist="binary" (defensive
    # branches below, not currently in SCENARIOS) since an additive shift
    # doesn't make sense on a 0/1 scale.
    es = effect_size_override if effect_size_override is not None else sc.effect_size

    # ── Independent two-group data (ttest, mannwhitney) ────────────────────
    truth_a2 = _sample_truth(sc.dist, n1, rng)
    truth_b2 = _sample_truth(sc.dist, n2, rng) + es
    llm_a2 = _llm(truth_a2, bias_a, noise1, rng)
    llm_b2 = _llm(truth_b2, bias_b, noise2, rng)
    lab_a2 = _labels_independent(truth_a2, sc.label_frac, rng)
    lab_b2 = _labels_independent(truth_b2, sc.label_frac, rng)

    # ── Paired data (wilcoxon) ─────────────────────────────────────────────
    # n_subjects with shared within-subject baseline; H₀: no condition effect
    base_2 = rng.normal(_mu_null(sc.dist), _SIGMA_SUB, n1)
    if sc.dist == "binary":
        p_sub = rng.uniform(0.2, 0.8, n1)
        truth_x = rng.binomial(1, p_sub, n1).astype(float)
        truth_y = rng.binomial(1, p_sub, n1).astype(float)
    else:
        truth_x = base_2 + rng.normal(0.0, _SIGMA_COND, n1)
        truth_y = base_2 + rng.normal(0.0, _SIGMA_COND, n1) + es
    llm_x = _llm(truth_x, bias_a, noise1, rng)
    llm_y = _llm(truth_y, bias_b, noise2, rng)
    lab_x, lab_y = _labels_shared([truth_x, truth_y], sc.label_frac, rng)

    # ── Independent three-group data (anova_ind) ───────────────────────────
    truth_a3 = _sample_truth(sc.dist, n1, rng)
    truth_b3 = _sample_truth(sc.dist, n2, rng) + es
    truth_c3 = _sample_truth(sc.dist, n3, rng) + 2 * es
    llm_a3 = _llm(truth_a3, bias_a, noise1, rng)
    llm_b3 = _llm(truth_b3, bias_b, noise2, rng)
    llm_c3 = _llm(truth_c3, bias_c, noise3, rng)
    lab_a3 = _labels_independent(truth_a3, sc.label_frac, rng)
    lab_b3 = _labels_independent(truth_b3, sc.label_frac, rng)
    lab_c3 = _labels_independent(truth_c3, sc.label_frac, rng)

    # ── Repeated-measures three-group data (anova_rep) ─────────────────────
    if sc.dist == "binary":
        p_sub3 = rng.uniform(0.2, 0.8, n1)
        truth_A = rng.binomial(1, p_sub3, n1).astype(float)
        truth_B = rng.binomial(1, p_sub3, n1).astype(float)
        truth_C = rng.binomial(1, p_sub3, n1).astype(float)
    else:
        base_3 = rng.normal(_mu_null(sc.dist), _SIGMA_SUB, n1)
        truth_A = base_3 + rng.normal(0.0, _SIGMA_COND, n1)
        truth_B = base_3 + rng.normal(0.0, _SIGMA_COND, n1) + es
        truth_C = base_3 + rng.normal(0.0, _SIGMA_COND, n1) + 2 * es
    llm_A = _llm(truth_A, bias_a, noise1, rng)
    llm_B = _llm(truth_B, bias_b, noise2, rng)
    llm_C = _llm(truth_C, bias_c, noise3, rng)
    lab_A, lab_B, lab_C = _labels_shared([truth_A, truth_B, truth_C], sc.label_frac, rng)

    # ── 2x2 crossed fixed-factor data (lmm_factorial) ──────────────────────
    # Own 4-group truth/llm/lab arrays (own random intercept), since the
    # 2x2 design (f1 x f2) has 4 conditions, not 3 — same monotonic
    # effect_size convention as truth_A/B/C above (group 0 unshifted, each
    # subsequent group +es), generalized to 4 groups.
    bias_w, bias_x, bias_y, bias_z = _biases_4(sc)
    if sc.dist == "binary":
        p_sub4 = rng.uniform(0.2, 0.8, n1)
        truth_W = rng.binomial(1, p_sub4, n1).astype(float)
        truth_X = rng.binomial(1, p_sub4, n1).astype(float)
        truth_Y = rng.binomial(1, p_sub4, n1).astype(float)
        truth_Z = rng.binomial(1, p_sub4, n1).astype(float)
    else:
        base_4 = rng.normal(_mu_null(sc.dist), _SIGMA_SUB, n1)
        truth_W = base_4 + rng.normal(0.0, _SIGMA_COND, n1)
        truth_X = base_4 + rng.normal(0.0, _SIGMA_COND, n1) + es
        truth_Y = base_4 + rng.normal(0.0, _SIGMA_COND, n1) + 2 * es
        truth_Z = base_4 + rng.normal(0.0, _SIGMA_COND, n1) + 3 * es
    llm_W = _llm(truth_W, bias_w, noise1, rng)
    llm_X = _llm(truth_X, bias_x, noise2, rng)
    llm_Y = _llm(truth_Y, bias_y, noise3, rng)
    llm_Z = _llm(truth_Z, bias_z, noise1, rng)
    lab_W, lab_X, lab_Y, lab_Z = _labels_shared(
        [truth_W, truth_X, truth_Y, truth_Z], sc.label_frac, rng
    )

    # ── Nested run replication (lmm_runs) ───────────────────────────────────
    # Reuses (truth_A, truth_B, truth_C) / (lab_A, lab_B, lab_C) above — same
    # per-input random intercept, just R independent LLM-noise draws per
    # cell instead of one, exercising the GLS-sufficiency run-collapsing
    # path in _ppi_lmm_fixed_effects. Human labels stay single-valued per
    # item regardless of R (no run dimension on the labeled side).
    llm_A_runs = np.column_stack([_llm(truth_A, bias_a, noise1, rng) for _ in range(_LMM_RUNS_R)])
    llm_B_runs = np.column_stack([_llm(truth_B, bias_b, noise2, rng) for _ in range(_LMM_RUNS_R)])
    llm_C_runs = np.column_stack([_llm(truth_C, bias_c, noise3, rng) for _ in range(_LMM_RUNS_R)])

    corrected_results: dict[str, bool | None] = {}
    uncorrected_results: dict[str, bool | None] = {}
    effect_results: dict[str, tuple[float, float, float, float]] = {}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ttest (Student's, equal_var=True): estimand = E[A] − E[B]
        if "ttest" in active_tests:
            try:
                p_uncorrected = float(_scipy_stats.ttest_ind(llm_a2, llm_b2, equal_var=True).pvalue)
                uncorrected_results["ttest"] = p_uncorrected < ALPHA
                r = _ppi_two_sample(
                    llm_a2, llm_b2, lab_a2, lab_b2,
                    lambda ya, yb: float(ya.mean() - yb.mean()),
                    ALPHA, n_boot, _rng_seed(),
                )
                corrected_results["ttest"] = r.p_value < ALPHA
                effect_results["ttest"] = (r.estimate, r.ci_low, r.ci_high, r.llm_estimate)
            except Exception:
                corrected_results["ttest"] = None
                uncorrected_results["ttest"] = None

        # ttest_welch (Welch's, equal_var=False): same PPI estimand, Welch df for uncorrected
        if "ttest_welch" in active_tests:
            try:
                p_uncorrected = float(_scipy_stats.ttest_ind(llm_a2, llm_b2, equal_var=False).pvalue)
                uncorrected_results["ttest_welch"] = p_uncorrected < ALPHA
                r = _ppi_two_sample(
                    llm_a2, llm_b2, lab_a2, lab_b2,
                    lambda ya, yb: float(ya.mean() - yb.mean()),
                    ALPHA, n_boot, _rng_seed(),
                )
                corrected_results["ttest_welch"] = r.p_value < ALPHA
                effect_results["ttest_welch"] = (r.estimate, r.ci_low, r.ci_high, r.llm_estimate)
            except Exception:
                corrected_results["ttest_welch"] = None
                uncorrected_results["ttest_welch"] = None

        # mannwhitney: mid-rank estimand = P_mid(X>Y) - 0.5; 0 under H₀ for any dist.
        if "mw" in active_tests:
            try:
                p_uncorrected = float(
                    _scipy_stats.mannwhitneyu(llm_a2, llm_b2, alternative="two-sided").pvalue
                )
                uncorrected_results["mw"] = p_uncorrected < ALPHA
                r = _ppi_two_sample(
                    llm_a2, llm_b2, lab_a2, lab_b2,
                    lambda xa, ya: _p_x_gt_y_midrank(xa, ya) - 0.5,
                    ALPHA, n_boot, _rng_seed(),
                )
                corrected_results["mw"] = r.p_value < ALPHA
                effect_results["mw"] = (r.estimate, r.ci_low, r.ci_high, r.llm_estimate)
            except Exception:
                corrected_results["mw"] = None
                uncorrected_results["mw"] = None

        # wilcoxon: estimand = median(X − Y), paired
        if "wilcoxon" in active_tests:
            try:
                p_uncorrected = float(_scipy_stats.wilcoxon(llm_x, llm_y, alternative="two-sided").pvalue)
                uncorrected_results["wilcoxon"] = p_uncorrected < ALPHA
                r = _ppi_paired_arrays(
                    llm_x, llm_y, lab_x, lab_y,
                    np.median, ALPHA, n_boot, _rng_seed(),
                    rectifier_func=np.mean,
                )
                corrected_results["wilcoxon"] = r.p_value < ALPHA
                effect_results["wilcoxon"] = (r.estimate, r.ci_low, r.ci_high, r.llm_estimate)
            except Exception:
                corrected_results["wilcoxon"] = None
                uncorrected_results["wilcoxon"] = None

        # anova_ind: direct closed-form F-test
        if "anova_ind" in active_tests:
            try:
                groups_ind = [llm_a3, llm_b3, llm_c3]
                groups_ind_lab = [lab_a3, lab_b3, lab_c3]
                p_uncorrected = _uncorrected_anova_independent_p_value(groups_ind)
                uncorrected_results["anova_ind"] = p_uncorrected < ALPHA
                p = _ppi_anova_independent_p_value(
                    groups_ind,
                    groups_ind_lab,
                    k=len(groups_ind),
                )
                corrected_results["anova_ind"] = (p is not None) and (p < ALPHA)
            except Exception:
                corrected_results["anova_ind"] = None
                uncorrected_results["anova_ind"] = None

        # anova_rep: direct closed-form F-test
        if "anova_rep" in active_tests:
            try:
                groups_rep = [llm_A, llm_B, llm_C]
                groups_rep_lab = [lab_A, lab_B, lab_C]
                p_uncorrected = _uncorrected_anova_repeated_p_value(groups_rep)
                uncorrected_results["anova_rep"] = p_uncorrected < ALPHA
                p = _ppi_anova_repeated_p_value(
                    groups_rep,
                    groups_rep_lab,
                    k=len(groups_rep),
                )
                corrected_results["anova_rep"] = (p is not None) and (p < ALPHA)
            except Exception:
                corrected_results["anova_rep"] = None
                uncorrected_results["anova_rep"] = None

        # friedman: rank-based repeated-measures analog of anova_rep — reuses
        # the same (llm_A, llm_B, llm_C) / (lab_A, lab_B, lab_C) data, since
        # the Friedman estimand is just anova_rep's variance estimand applied
        # to within-subject ranks instead of raw scores.
        if "friedman" in active_tests:
            try:
                groups_fr = [llm_A, llm_B, llm_C]
                groups_fr_lab = [lab_A, lab_B, lab_C]
                p_uncorrected = _uncorrected_friedman_p_value(groups_fr)
                uncorrected_results["friedman"] = p_uncorrected < ALPHA
                p = _ppi_friedman_p_value(
                    groups_fr,
                    groups_fr_lab,
                    k=len(groups_fr),
                )
                corrected_results["friedman"] = (p is not None) and (p < ALPHA)
            except Exception:
                corrected_results["friedman"] = None
                uncorrected_results["friedman"] = None

        # kruskal: rank-based independent-groups analog of anova_ind — reuses
        # the same (llm_a3, llm_b3, llm_c3) / (lab_a3, lab_b3, lab_c3) data.
        # Unlike the other omnibus tests, this is a Wald test over the joint
        # bootstrap covariance of pairwise dominance effects, not a
        # closed-form approximation, so it needs its own n_boot/rng draw.
        if "kruskal" in active_tests:
            try:
                groups_kw = [llm_a3, llm_b3, llm_c3]
                groups_kw_lab = [lab_a3, lab_b3, lab_c3]
                p_uncorrected = _uncorrected_kruskal_p_value(groups_kw)
                uncorrected_results["kruskal"] = p_uncorrected < ALPHA
                pw = _ppi_kruskal_wallis_pairwise(
                    groups_kw, groups_kw_lab, alpha=ALPHA, n_boot=n_boot, rng=_rng_seed(),
                )
                corrected_results["kruskal"] = pw["wald_p"] < ALPHA
                # Raw (un-centered) pairwise θ, averaged across pairs — kept
                # in the same raw scale as every other test's effect_results
                # entry; _print_effect_table subtracts each test's own null
                # value (0.5 here, exact by symmetry — see _gold_null_values).
                # llm_theta is the same quantity before PPI rectification —
                # kruskal's analog of the other tests' "free" llm_estimate.
                llm_theta = _kw_pairwise_thetas(groups_kw, pw["pairs"])
                effect_results["kruskal"] = (
                    float(np.mean(pw["theta_hat"])),
                    float(np.mean(pw["ci_lo"])),
                    float(np.mean(pw["ci_hi"])),
                    float(np.mean(llm_theta)),
                )
            except Exception:
                corrected_results["kruskal"] = None
                uncorrected_results["kruskal"] = None

        # lmm: parametric repeated-measures analog of anova_rep, with an
        # explicit between-input random-intercept variance component instead
        # of row-centering — reuses the same (llm_A, llm_B, llm_C) /
        # (lab_A, lab_B, lab_C) data, since that data is already generated
        # from exactly this model. Unlike anova_rep/friedman, the correction
        # is a single closed-form rectified-GLS solve (no bootstrap), so it's
        # cheap enough to run directly in the main Type I sweep rather than
        # needing a separate slow pass. No effect_results entry: its headline
        # estimand has no valid CI by design (see es.tests.lmm()'s docstring),
        # so it's intentionally excluded from the bias/CI-coverage check.
        if "lmm" in active_tests:
            try:
                groups_lmm = [llm_A, llm_B, llm_C]
                groups_lmm_lab = [lab_A, lab_B, lab_C]
                p_uncorrected = _uncorrected_lmm_p_value(groups_lmm)
                uncorrected_results["lmm"] = p_uncorrected < ALPHA
                p = _ppi_lmm_p_value(
                    groups_lmm,
                    groups_lmm_lab,
                    k=len(groups_lmm),
                )
                corrected_results["lmm"] = (p is not None) and (p < ALPHA)
            except Exception:
                corrected_results["lmm"] = None
                uncorrected_results["lmm"] = None

        # lmm_factorial: generalizes lmm to two crossed fixed factors
        # (score ~ C(f1) * C(f2) + (1|input)) via es.tests.lmm()'s
        # ``factors=`` argument — reuses the (llm_W..Z) / (lab_W..Z) 2x2
        # data above. No effect_results entry, same reason as lmm.
        if "lmm_factorial" in active_tests:
            try:
                groups_lf = [llm_W, llm_X, llm_Y, llm_Z]
                groups_lf_lab = [lab_W, lab_X, lab_Y, lab_Z]
                p_uncorrected = _uncorrected_lmm_p_value(groups_lf, factors=_LMM_FACTORIAL_FACTORS)
                uncorrected_results["lmm_factorial"] = p_uncorrected < ALPHA
                p = _ppi_lmm_p_value(
                    groups_lf,
                    groups_lf_lab,
                    k=len(groups_lf),
                    factors=_LMM_FACTORIAL_FACTORS,
                )
                corrected_results["lmm_factorial"] = (p is not None) and (p < ALPHA)
            except Exception:
                corrected_results["lmm_factorial"] = None
                uncorrected_results["lmm_factorial"] = None

        # lmm_runs: generalizes lmm to nested run replication (R independent
        # LLM samples per cell) — reuses (llm_A_runs, llm_B_runs, llm_C_runs)
        # / (lab_A, lab_B, lab_C) above. No effect_results entry, same
        # reason as lmm.
        if "lmm_runs" in active_tests:
            try:
                groups_runs = [llm_A_runs, llm_B_runs, llm_C_runs]
                groups_runs_lab = [lab_A, lab_B, lab_C]
                p_uncorrected = _uncorrected_lmm_p_value(groups_runs)
                uncorrected_results["lmm_runs"] = p_uncorrected < ALPHA
                p = _ppi_lmm_p_value(
                    groups_runs,
                    groups_runs_lab,
                    k=len(groups_runs),
                )
                corrected_results["lmm_runs"] = (p is not None) and (p < ALPHA)
            except Exception:
                corrected_results["lmm_runs"] = None
                uncorrected_results["lmm_runs"] = None

    return sc_idx, corrected_results, uncorrected_results, effect_results


_EFFECT_ANOVA_TESTS = ("anova_ind", "anova_rep", "friedman")


def _run_one_effect_anova(args: tuple) -> tuple[int, dict[str, tuple[float, float, float, float]]]:
    """Effect-size bias/coverage pass for anova_ind/anova_rep/friedman only.

    These three deliberately use closed-form, bootstrap-free p-value
    functions in ``_run_one`` to keep the (high-rep) Type I sweep fast —
    there is no "free" bootstrap result to read an estimate/CI off of, the
    way there is for ttest/mw/wilcoxon/kruskal. Getting their corrected
    point estimate at all means calling their bootstrap-based scalar
    function (``_ppi_anova_independent`` etc. — the same one the public
    ``anova_oneway()``/``friedman()`` API already calls for `corrected_estimate`
    in normal use, so this isn't introducing a new kind of computation, just
    a new place that runs it). To avoid multiplying that cost by the Type I
    sweep's (potentially large) --reps, this runs as a separate pass with
    its own, typically much smaller, --effect-reps count.
    """
    sc_idx, seed, n_boot, active_tests = args
    sc: Scenario = SCENARIOS[sc_idx]
    rng = np.random.default_rng(seed)

    n1 = sc.n
    n2 = sc.n2 if sc.n2 is not None else sc.n
    n3 = sc.n3 if sc.n3 is not None else sc.n
    noise1 = sc.llm_noise
    noise2 = sc.llm_noise2 if sc.llm_noise2 is not None else sc.llm_noise
    noise3 = sc.llm_noise3 if sc.llm_noise3 is not None else sc.llm_noise
    bias_a, bias_b, bias_c = _biases(sc)

    effect_results: dict[str, tuple[float, float, float, float]] = {}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if "anova_ind" in active_tests:
            try:
                truth_a3 = _sample_truth(sc.dist, n1, rng)
                truth_b3 = _sample_truth(sc.dist, n2, rng)
                truth_c3 = _sample_truth(sc.dist, n3, rng)
                llm_a3 = _llm(truth_a3, bias_a, noise1, rng)
                llm_b3 = _llm(truth_b3, bias_b, noise2, rng)
                llm_c3 = _llm(truth_c3, bias_c, noise3, rng)
                lab_a3 = _labels_independent(truth_a3, sc.label_frac, rng)
                lab_b3 = _labels_independent(truth_b3, sc.label_frac, rng)
                lab_c3 = _labels_independent(truth_c3, sc.label_frac, rng)
                ppi = _ppi_anova_independent(
                    [llm_a3, llm_b3, llm_c3], [lab_a3, lab_b3, lab_c3],
                    ALPHA, n_boot, _rng_seed(),
                )
                effect_results["anova_ind"] = (ppi.estimate, ppi.ci_low, ppi.ci_high, ppi.llm_estimate)
            except Exception:
                pass

        if "anova_rep" in active_tests or "friedman" in active_tests:
            try:
                if sc.dist == "binary":
                    p_sub3 = rng.uniform(0.2, 0.8, n1)
                    truth_A = rng.binomial(1, p_sub3, n1).astype(float)
                    truth_B = rng.binomial(1, p_sub3, n1).astype(float)
                    truth_C = rng.binomial(1, p_sub3, n1).astype(float)
                else:
                    base_3 = rng.normal(_mu_null(sc.dist), _SIGMA_SUB, n1)
                    truth_A = base_3 + rng.normal(0.0, _SIGMA_COND, n1)
                    truth_B = base_3 + rng.normal(0.0, _SIGMA_COND, n1)
                    truth_C = base_3 + rng.normal(0.0, _SIGMA_COND, n1)
                llm_A = _llm(truth_A, bias_a, noise1, rng)
                llm_B = _llm(truth_B, bias_b, noise2, rng)
                llm_C = _llm(truth_C, bias_c, noise3, rng)
                lab_A, lab_B, lab_C = _labels_shared([truth_A, truth_B, truth_C], sc.label_frac, rng)

                if "anova_rep" in active_tests:
                    try:
                        ppi = _ppi_anova_repeated(
                            [llm_A, llm_B, llm_C], [lab_A, lab_B, lab_C],
                            ALPHA, n_boot, _rng_seed(),
                        )
                        effect_results["anova_rep"] = (ppi.estimate, ppi.ci_low, ppi.ci_high, ppi.llm_estimate)
                    except Exception:
                        pass

                if "friedman" in active_tests:
                    try:
                        ppi = _ppi_friedman(
                            [llm_A, llm_B, llm_C], [lab_A, lab_B, lab_C],
                            ALPHA, n_boot, _rng_seed(),
                        )
                        effect_results["friedman"] = (ppi.estimate, ppi.ci_low, ppi.ci_high, ppi.llm_estimate)
                    except Exception:
                        pass
            except Exception:
                pass

    return sc_idx, effect_results


# ── Output helpers ────────────────────────────────────────────────────────────

_COL_W = 7   # width per rate cell


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli rate."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    radius = (z / denom) * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
    lo = max(0.0, center - radius)
    hi = min(1.0, center + radius)
    return lo, hi


def _binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    log_c = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    log_p = k * math.log(p) + (n - k) * math.log1p(-p)
    return float(math.exp(log_c + log_p))


def _binom_test_two_sided(k: int, n: int, p0: float) -> float:
    """Exact two-sided binomial p-value (probability ordering definition)."""
    if n <= 0:
        return float("nan")
    p_obs = _binom_pmf(k, n, p0)
    total = 0.0
    eps = 1e-15
    for i in range(n + 1):
        pi = _binom_pmf(i, n, p0)
        if pi <= p_obs + eps:
            total += pi
    return min(1.0, max(0.0, total))


def _holm_rejections(pvals: list[tuple[tuple[int, str], float]], alpha: float = 0.05) -> set[tuple[int, str]]:
    """Return rejected cells under Holm-Bonferroni family-wise error control."""
    ordered = sorted(pvals, key=lambda x: x[1])
    m = len(ordered)
    rejected: set[tuple[int, str]] = set()
    for i, (cell, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        if p <= thresh:
            rejected.add(cell)
        else:
            break
    return rejected

def _fmt_rate(rate: float | None, flag2: float, flag3: float) -> str:
    if rate is None:
        return "  n/a  "
    s = f"{rate:.3f}"
    if rate > flag3:
        s += "●●"
    elif rate > flag2:
        s += "● "
    else:
        s += "  "
    return s


def _rate_matrix(counts: dict, totals: dict, test_names: list[str] = TEST_NAMES) -> np.ndarray:
    mat = np.full((len(SCENARIOS), len(test_names)), np.nan, dtype=float)
    for i in range(len(SCENARIOS)):
        for j, t in enumerate(test_names):
            n = totals[i][t]
            if n > 0:
                mat[i, j] = counts[i][t] / n
    return mat


def _print_table(
    counts: dict,
    totals: dict,
    uncorrected_counts: dict,
    uncorrected_totals: dict,
    n_reps: int,
    n_boot: int,
    test_names: list[str] = TEST_NAMES,
) -> None:
    sigma = (ALPHA * (1 - ALPHA) / n_reps) ** 0.5
    flag2 = ALPHA + 2 * sigma
    flag3 = ALPHA + 3 * sigma

    width = 80
    bar = "─" * width
    dbar = "═" * width

    def rates(sc_idx: int) -> dict[str, float | None]:
        return {
            t: counts[sc_idx][t] / totals[sc_idx][t]
            if totals[sc_idx][t] > 0 else None
            for t in test_names
        }

    def uncorrected_rates(sc_idx: int) -> dict[str, float | None]:
        return {
            t: uncorrected_counts[sc_idx][t] / uncorrected_totals[sc_idx][t]
            if uncorrected_totals[sc_idx][t] > 0 else None
            for t in test_names
        }

    def alpha_outside_ci(sc_idx: int, test_name: str) -> bool:
        n = totals[sc_idx][test_name]
        if n <= 0:
            return False
        lo, hi = _wilson_interval(counts[sc_idx][test_name], n)
        return (ALPHA < lo) or (ALPHA > hi)

    print()
    print(dbar)
    print(f"  TYPE I ERROR CALIBRATION — PPI-corrected hypothesis tests")
    print(f"  n_reps={n_reps}  n_boot={n_boot}  α={ALPHA}")
    print(f"  hard constraint: n_lab >= {_MIN_LAB}")
    print(f"  2σ flag (●): rate > {flag2:.3f}    3σ flag (●●): rate > {flag3:.3f}")
    print(f"  Wilson flag (†): 95% CI for rejection rate excludes α")
    print(f"  Holm flag (‡): exact binomial miscalibration survives family-wise correction")
    print(dbar)

    col_names = [TEST_SHORT_NAMES[t] for t in test_names]
    header_lbl = f"{'Scenario':<32}" + "".join(f"{c:^9}" for c in col_names)
    print()
    print(header_lbl)
    print(bar)

    seen_tags: list[str] = []
    tag_order: list[str] = []
    for sc in SCENARIOS:
        if sc.tag not in tag_order:
            tag_order.append(sc.tag)

    all_rates: list[dict[str, float | None]] = []
    all_uncorrected_rates: list[dict[str, float | None]] = []
    for sc_idx, sc in enumerate(SCENARIOS):
        all_rates.append(rates(sc_idx))
        all_uncorrected_rates.append(uncorrected_rates(sc_idx))

    pvals: list[tuple[tuple[int, str], float]] = []
    for sc_idx in range(len(SCENARIOS)):
        for t in test_names:
            n = totals[sc_idx][t]
            if n > 0:
                pvals.append(((sc_idx, t), _binom_test_two_sided(counts[sc_idx][t], n, ALPHA)))
    holm_bad = _holm_rejections(pvals, alpha=0.05)

    for tag in tag_order:
        indices = [i for i, sc in enumerate(SCENARIOS) if sc.tag == tag]
        tag_label = f"[{tag}]"
        print(f"\n{tag_label}")
        for i in indices:
            sc = SCENARIOS[i]
            r = all_rates[i]
            row = f"  {sc.name:<30}"
            for t in test_names:
                row += f" {_fmt_rate(r.get(t), flag2, flag3)}"
            if any(alpha_outside_ci(i, t) for t in test_names):
                row += "  †"
            if any((i, t) in holm_bad for t in test_names):
                row += "‡"
            print(row)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print(bar)
    print("SUMMARY")
    print()

    n_conditions = len(SCENARIOS) * len(test_names)

    flags2 = sum(
        1 for r in all_rates for t in test_names
        if r.get(t) is not None and r[t] > flag2
    )
    flags3 = sum(
        1 for r in all_rates for t in test_names
        if r.get(t) is not None and r[t] > flag3
    )
    wilson_miscal = sum(
        1
        for sc_idx in range(len(SCENARIOS))
        for t in test_names
        if alpha_outside_ci(sc_idx, t)
    )
    nominal_miscal = sum(1 for _, p in pvals if p < 0.05)
    uncorrected_flags2 = sum(
        1 for r in all_uncorrected_rates for t in test_names
        if r.get(t) is not None and r[t] > flag2
    )
    uncorrected_flags3 = sum(
        1 for r in all_uncorrected_rates for t in test_names
        if r.get(t) is not None and r[t] > flag3
    )

    print(f"  Scenarios: {len(SCENARIOS)}  |  Tests: {len(test_names)}  |  Conditions: {n_conditions}")
    print(f"  Inflated at 2σ (rate > {flag2:.3f}):  {flags2}/{n_conditions}")
    print(f"  Inflated at 3σ (rate > {flag3:.3f}):  {flags3}/{n_conditions}")
    print(f"  Wilson miscalibrated (α outside 95% CI): {wilson_miscal}/{n_conditions}")
    print(f"  Exact-binomial p<0.05 (corrected rates, unadjusted): {nominal_miscal}/{n_conditions}")
    print(f"  Holm-confirmed miscalibrated cells: {len(holm_bad)}/{n_conditions}")
    print()
    print("  Uncorrected aggregate")
    print(f"  Inflated at 2σ (rate > {flag2:.3f}):  {uncorrected_flags2}/{n_conditions}")
    print(f"  Inflated at 3σ (rate > {flag3:.3f}):  {uncorrected_flags3}/{n_conditions}")
    print()
    print(f"  {'Test':<12}  {'corr max':>9}  {'corr mean':>9}  {'corr med':>9}  {'unc max':>9}  {'unc mean':>9}  {'unc med':>9}")
    for t in test_names:
        col_rates = [r[t] for r in all_rates if r.get(t) is not None]
        col_uncorrected = [r[t] for r in all_uncorrected_rates if r.get(t) is not None]
        if col_rates or col_uncorrected:
            corr_max = max(col_rates) if col_rates else float("nan")
            corr_mean = float(np.mean(col_rates)) if col_rates else float("nan")
            corr_median = float(np.median(col_rates)) if col_rates else float("nan")
            unc_max = max(col_uncorrected) if col_uncorrected else float("nan")
            unc_mean = float(np.mean(col_uncorrected)) if col_uncorrected else float("nan")
            unc_median = float(np.median(col_uncorrected)) if col_uncorrected else float("nan")
            print(
                f"  {t:<12}  {corr_max:>9.3f}  {corr_mean:>9.3f}  {corr_median:>9.3f}  "
                f"{unc_max:>9.3f}  {unc_mean:>9.3f}  {unc_median:>9.3f}"
            )


def _print_power_table(
    counts: dict,
    totals: dict,
    uncorrected_counts: dict,
    uncorrected_totals: dict,
    n_reps: int,
    n_boot: int,
    effect_size: float,
    test_names: list[str] = TEST_NAMES,
    power_flag: float = 0.80,
) -> None:
    """Statistical power (rejection rate under a REAL injected effect),
    same scenario grid and same rejection-decision plumbing as the Type I
    table — counts/totals here come from the same _run_one, just fed data
    where effect_size > 0 (see Scenario.effect_size). Low corrected power is
    flagged (●); want HIGH rate here, the opposite of the Type I table.

    The uncorrected column is NOT a fair "power" comparison: every scenario
    here still carries its own LLM bias_type artifact on top of the real
    effect, so an uncorrected rejection can come from either the genuine
    effect or the bias alone. It's shown for context, not as a baseline to
    beat.
    """
    width = 80
    bar = "─" * width
    dbar = "═" * width

    mat = _rate_matrix(counts, totals, test_names)
    unc_mat = _rate_matrix(uncorrected_counts, uncorrected_totals, test_names)

    print()
    print(dbar)
    print("  STATISTICAL POWER — PPI-corrected hypothesis tests under a real injected effect")
    print(f"  n_reps={n_reps}  n_boot={n_boot}  effect_size={effect_size}  α={ALPHA}")
    print(f"  Low-power flag (●): corrected rejection rate < {power_flag:.2f}")
    print("  'uncorrected' column is NOT a fair baseline here — see docstring")
    print(dbar)

    col_names = [TEST_SHORT_NAMES[t] for t in test_names]
    header_lbl = f"{'Scenario':<32}" + "".join(f"{c:^9}" for c in col_names)
    print()
    print(header_lbl)
    print(bar)

    tag_order: list[str] = []
    for sc in SCENARIOS:
        if sc.tag not in tag_order:
            tag_order.append(sc.tag)

    def fmt_power(rate: float | None) -> str:
        if rate is None:
            return "  n/a  "
        s = f"{rate:.3f}"
        s += "●  " if rate < power_flag else "   "
        return s

    for tag in tag_order:
        indices = [i for i, sc in enumerate(SCENARIOS) if sc.tag == tag]
        print(f"\n[{tag}]")
        for i in indices:
            sc = SCENARIOS[i]
            row = f"  {sc.name:<30}"
            for j, t in enumerate(test_names):
                row += f" {fmt_power(mat[i, j] if np.isfinite(mat[i, j]) else None)}"
            print(row)

    print()
    print(bar)
    print("SUMMARY")
    print()
    n_conditions = len(SCENARIOS) * len(test_names)
    finite_mat = mat[np.isfinite(mat)]
    low_power = int(np.sum(finite_mat < power_flag))
    print(f"  Scenarios: {len(SCENARIOS)}  |  Tests: {len(test_names)}  |  Conditions: {n_conditions}")
    print(f"  Low power (rate < {power_flag:.2f}): {low_power}/{finite_mat.size}")
    print()
    print(f"  {'Test':<12}  {'corr min':>9}  {'corr mean':>9}  {'corr med':>9}  {'unc min':>9}  {'unc mean':>9}  {'unc med':>9}")
    for j, t in enumerate(test_names):
        col = mat[:, j][np.isfinite(mat[:, j])]
        col_unc = unc_mat[:, j][np.isfinite(unc_mat[:, j])]
        if col.size or col_unc.size:
            corr_min = float(np.min(col)) if col.size else float("nan")
            corr_mean = float(np.mean(col)) if col.size else float("nan")
            corr_med = float(np.median(col)) if col.size else float("nan")
            unc_min = float(np.min(col_unc)) if col_unc.size else float("nan")
            unc_mean = float(np.mean(col_unc)) if col_unc.size else float("nan")
            unc_med = float(np.median(col_unc)) if col_unc.size else float("nan")
            print(
                f"  {t:<12}  {corr_min:>9.3f}  {corr_mean:>9.3f}  {corr_med:>9.3f}  "
                f"{unc_min:>9.3f}  {unc_mean:>9.3f}  {unc_med:>9.3f}"
            )


def _effect_cell_stats(
    samples: list[tuple[float, float, float, float]], null_val: float
) -> tuple[float, float, float, float, int]:
    """(mean_bias, z, coverage_rate, mean_ci_width, n) for one (scenario,
    test) cell, all relative to ``null_val`` — shared by the printed table
    and the plot so both report exactly the same numbers.
    """
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    estimates = np.array([s[0] for s in samples]) - null_val
    contains = np.array([(s[1] <= null_val <= s[2]) for s in samples])
    ci_widths = np.array([s[2] - s[1] for s in samples])
    bias_mean = float(estimates.mean())
    bias_se = float(estimates.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = bias_mean / bias_se if bias_se and bias_se > 0 else float("nan")
    coverage = float(contains.mean())
    mean_ci_width = float(ci_widths.mean())
    return bias_mean, z, coverage, mean_ci_width, n


def _uncorrected_bias_z(
    samples: list[tuple[float, float, float, float]], null_val: float
) -> tuple[float, float]:
    """(mean_bias, z) for the RAW (pre-PPI-correction) LLM-only estimate —
    the 4th element of each sample tuple — relative to ``null_val``. There's
    no per-replicate CI for the uncorrected estimate in this script, so this
    only has a bias/z reading, not a coverage one (see ``main()``'s coverage
    proxy via the existing uncorrected-rejection-rate matrix instead).
    """
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan")
    llm_estimates = np.array([s[3] for s in samples]) - null_val
    bias_mean = float(llm_estimates.mean())
    bias_se = float(llm_estimates.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = bias_mean / bias_se if bias_se and bias_se > 0 else float("nan")
    return bias_mean, z


def _effect_matrices(
    effect_samples: dict[int, dict[str, list[tuple[float, float, float, float]]]],
    gold_null: dict[int, dict[str, float]],
    test_names: list[str] = TEST_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(bias_mat, z_mat, coverage_mat, unc_z_mat, width_mat, rel_width_mat),
    each shape (n_scenarios, n_tests) — used by the heatmap/scatter plots.
    ``unc_z_mat`` is the RAW (uncorrected) LLM-only estimate's bias z-score,
    for the gray "uncorrected" overlay in the scatter plot. ``width_mat`` is
    the mean PPI bootstrap CI width in each test's own raw units (mean
    differences, probability deviations, variance-like estimands, etc. all
    have different natural scales, so raw widths aren't comparable across
    test columns); ``rel_width_mat`` divides each test's column by its own
    median width, making "wider/narrower than this test's typical case" the
    comparable quantity across tests on a single shared color scale.
    """
    n_scenarios = len(SCENARIOS)
    bias_mat = np.full((n_scenarios, len(test_names)), np.nan)
    z_mat = np.full((n_scenarios, len(test_names)), np.nan)
    cov_mat = np.full((n_scenarios, len(test_names)), np.nan)
    unc_z_mat = np.full((n_scenarios, len(test_names)), np.nan)
    width_mat = np.full((n_scenarios, len(test_names)), np.nan)
    for sc_idx in range(n_scenarios):
        for j, t in enumerate(test_names):
            samples = effect_samples.get(sc_idx, {}).get(t, [])
            null_val = gold_null.get(sc_idx, {}).get(t, 0.0)
            bias_mean, z, coverage, mean_ci_width, n = _effect_cell_stats(samples, null_val)
            if n > 0:
                bias_mat[sc_idx, j] = bias_mean
                z_mat[sc_idx, j] = z
                cov_mat[sc_idx, j] = coverage
                width_mat[sc_idx, j] = mean_ci_width
                _, unc_z = _uncorrected_bias_z(samples, null_val)
                unc_z_mat[sc_idx, j] = unc_z

    rel_width_mat = np.full_like(width_mat, np.nan)
    for j in range(len(test_names)):
        col = width_mat[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            median = float(np.median(finite))
            if median > 0:
                rel_width_mat[:, j] = col / median

    return bias_mat, z_mat, cov_mat, unc_z_mat, width_mat, rel_width_mat


def _print_effect_table(
    effect_samples: dict[int, dict[str, list[tuple[float, float, float, float]]]],
    gold_null: dict[int, dict[str, float]],
    test_names: list[str] = TEST_NAMES,
) -> None:
    """Bias, CI coverage, and CI width of the corrected point estimate,
    complementing the Type I table: Type I checks whether the p-value stays
    calibrated; this checks whether the estimate itself is centered at the
    truth, whether its CI covers that truth at the nominal rate, and how
    wide that CI typically is (and where it's narrowest/widest — CIs should
    shrink as labels/sample size grow and widen as noise grows; a flat or
    inverted trend would itself be a red flag, not just the raw number).
    ``gold_null`` supplies each (scenario, test) pair's true value — NOT
    always 0; see ``_gold_null_values`` for why anova_ind/anova_rep/friedman
    have a small positive finite-sample floor instead, and kruskal's raw θ
    is centered on 0.5.
    """
    width = 80
    bar = "─" * width
    dbar = "═" * width
    target_cov = 1.0 - ALPHA
    cov_flag_margin = 0.02  # flag only if meaningfully (not just noisily) under target

    print()
    print(dbar)
    print("  EFFECT SIZE CALIBRATION — bias & CI coverage of the corrected estimate")
    print("  (vs. Monte Carlo gold-reference null value per scenario/test — see _gold_null_values)")
    print(dbar)
    print()
    header = (
        f"  {'Test':<12} {'n':>6} {'mean bias':>10} {'worst |z|':>10} "
        f"{'worst scen (bias)':<24} {'coverage':>9} {'cov min':>8} {'worst scen (cov)':<22} "
        f"{'mean width':>11} {'width range':<24}"
    )
    print(header)
    print(bar)

    flagged: list[str] = []

    for t in test_names:
        all_estimates: list[np.ndarray] = []
        all_contains0: list[np.ndarray] = []
        all_widths: list[np.ndarray] = []
        bias_per_scenario: list[tuple[str, float, float]] = []   # name, |z|, mean bias
        cov_per_scenario: list[tuple[str, float, int]] = []       # name, rate, n
        width_per_scenario: list[tuple[str, float]] = []          # name, mean width

        for sc_idx, sc in enumerate(SCENARIOS):
            samples = effect_samples.get(sc_idx, {}).get(t, [])
            n = len(samples)
            if n == 0:
                continue
            null_val = gold_null.get(sc_idx, {}).get(t, 0.0)
            bias_mean, z, cov_rate, mean_ci_width, _ = _effect_cell_stats(samples, null_val)
            estimates = np.array([s[0] for s in samples]) - null_val
            contains0 = np.array([(s[1] <= null_val <= s[2]) for s in samples])
            widths = np.array([s[2] - s[1] for s in samples])
            all_estimates.append(estimates)
            all_contains0.append(contains0)
            all_widths.append(widths)

            bias_per_scenario.append((sc.name, abs(z) if np.isfinite(z) else 0.0, bias_mean))
            cov_per_scenario.append((sc.name, cov_rate, n))
            width_per_scenario.append((sc.name, mean_ci_width))

            if np.isfinite(z) and abs(z) > 3.0:
                flagged.append(
                    f"    bias    {sc.name:<28} {t:<10} mean={bias_mean:+.4f}  z={z:+.2f}  (n={n})"
                )
            lo_cov, hi_cov = _wilson_interval(int(contains0.sum()), n)
            if hi_cov < target_cov - cov_flag_margin:
                flagged.append(
                    f"    cover   {sc.name:<28} {t:<10} coverage={cov_rate:.3f}  "
                    f"Wilson=[{lo_cov:.3f},{hi_cov:.3f}]  (n={n})"
                )

        if not all_estimates:
            continue

        cat_estimates = np.concatenate(all_estimates)
        cat_contains0 = np.concatenate(all_contains0)
        cat_widths = np.concatenate(all_widths)
        n_total = len(cat_estimates)
        mean_bias = float(cat_estimates.mean())
        coverage = float(cat_contains0.mean())
        mean_width_all = float(cat_widths.mean())

        worst_bias_name, worst_bias_absz, _ = max(bias_per_scenario, key=lambda x: x[1])
        worst_cov_name, worst_cov_rate, _ = min(cov_per_scenario, key=lambda x: x[1])
        narrowest_name, narrowest_w = min(width_per_scenario, key=lambda x: x[1])
        widest_name, widest_w = max(width_per_scenario, key=lambda x: x[1])
        width_range_str = f"{narrowest_w:.3f} ({narrowest_name}) – {widest_w:.3f} ({widest_name})"

        print(
            f"  {t:<12} {n_total:>6} {mean_bias:>+10.4f} {worst_bias_absz:>10.2f} "
            f"{worst_bias_name:<24} {coverage:>9.3f} {worst_cov_rate:>8.3f} {worst_cov_name:<22} "
            f"{mean_width_all:>11.4f} {width_range_str:<24}"
        )

    print()
    if flagged:
        print(
            f"  Flagged cells (|bias z| > 3, or coverage Wilson upper bound "
            f"< {target_cov - cov_flag_margin:.2f}):"
        )
        for line in flagged:
            print(line)
    else:
        print("  No scenario×test cells flagged for bias or under-coverage.")


def _save_csv(
    counts: dict,
    totals: dict,
    uncorrected_counts: dict,
    uncorrected_totals: dict,
    path: str,
    test_names: list[str] = TEST_NAMES,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scenario", "tag"]
            + [f"{t}_corrected" for t in test_names]
            + [f"{t}_uncorrected" for t in test_names]
            + [f"{t}_corrected_n" for t in test_names]
            + [f"{t}_uncorrected_n" for t in test_names]
        )
        for sc_idx, sc in enumerate(SCENARIOS):
            row = [sc.name, sc.tag]
            for t in test_names:
                tot = totals[sc_idx][t]
                row.append(f"{counts[sc_idx][t] / tot:.4f}" if tot else "")
            for t in test_names:
                tot = uncorrected_totals[sc_idx][t]
                row.append(f"{uncorrected_counts[sc_idx][t] / tot:.4f}" if tot else "")
            for t in test_names:
                row.append(totals[sc_idx][t])
            for t in test_names:
                row.append(uncorrected_totals[sc_idx][t])
            writer.writerow(row)
    print(f"Results saved to {path}")


def _plot_results(
    counts: dict,
    totals: dict,
    uncorrected_counts: dict,
    uncorrected_totals: dict,
    n_reps: int,
    n_boot: int,
    out_path: str,
    test_names: list[str] = TEST_NAMES,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n_scenarios = len(SCENARIOS)
    mat = _rate_matrix(counts, totals, test_names)
    uncorrected_mat = _rate_matrix(uncorrected_counts, uncorrected_totals, test_names)

    # Center the color scale at alpha so well-calibrated cells are neutral.
    finite = mat[np.isfinite(mat)]
    vmax = max(ALPHA * 1.8, float(np.nanmax(finite)) if finite.size else ALPHA)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=ALPHA, vmax=vmax)

    fig_h = max(7.0, 0.34 * n_scenarios)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", norm=norm)

    ax.set_xticks(np.arange(len(test_names)))
    ax.set_xticklabels(test_names, rotation=0)
    ax.set_yticks(np.arange(n_scenarios))
    ax.set_yticklabels([sc.name for sc in SCENARIOS], fontsize=8)
    ax.set_xlabel("Test")
    ax.set_ylabel("Scenario")
    ax.set_title(
        f"Type I calibration heatmap (alpha={ALPHA}, reps={n_reps}, n_boot={n_boot})"
    )

    # Horizontal separators between factor blocks.
    for i in range(1, n_scenarios):
        if SCENARIOS[i].tag != SCENARIOS[i - 1].tag:
            ax.axhline(i - 0.5, color="white", lw=1.2, alpha=0.9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Observed rejection rate")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    scatter_out_path = str(Path(out_path).with_name(Path(out_path).stem + "_scatter.png"))
    fig2, ax2 = plt.subplots(figsize=(10.0, 5.8))
    rng = np.random.default_rng(0)
    colors = [plt.cm.tab10(i) for i in range(len(test_names))]
    unc_label_added = False

    for j, t in enumerate(test_names):
        vals_unc = uncorrected_mat[:, j]
        keep_unc = np.isfinite(vals_unc)
        y_unc = vals_unc[keep_unc]
        x_unc = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_unc)))
        ax2.scatter(
            x_unc,
            y_unc,
            s=18,
            alpha=0.35,
            color="#808080",
            label="uncorrected" if not unc_label_added else None,
            zorder=1,
        )
        unc_label_added = True

        vals = mat[:, j]
        keep = np.isfinite(vals)
        y = vals[keep]
        x = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep)))
        ax2.scatter(x, y, s=20, alpha=0.65, color=colors[j], label=t, zorder=2)

    ax2.axhline(ALPHA, color="black", ls="--", lw=1.1, label=f"alpha={ALPHA}")
    ax2.set_xlim(-0.5, len(test_names) - 0.5)
    scatter_max = np.nanmax(np.concatenate([mat.ravel(), uncorrected_mat.ravel()]))
    if not np.isfinite(scatter_max):
        scatter_max = 0.2
    ax2.set_ylim(0.0, max(0.2, float(scatter_max) * 1.05))
    ax2.set_xticks(np.arange(len(test_names)))
    ax2.set_xticklabels(test_names)
    ax2.set_ylabel("Observed rejection rate")
    ax2.set_xlabel("Test")
    ax2.set_title("Type I calibration scatter (all scenario x test table cells)")
    ax2.grid(axis="y", alpha=0.25, lw=0.8)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    fig2.tight_layout()
    fig2.savefig(scatter_out_path, dpi=180, bbox_inches="tight")
    print(f"Plot saved to {scatter_out_path}")

    backend = plt.get_backend().lower()
    if backend not in {"agg", "pdf", "svg", "ps", "cairo", "pgf", "template"}:
        plt.show()


def _plot_effect_results(
    effect_samples: dict[int, dict[str, list[tuple[float, float, float, float]]]],
    gold_null: dict[int, dict[str, float]],
    uncorrected_counts: dict,
    uncorrected_totals: dict,
    out_path: str,
    test_names: list[str] = TEST_NAMES,
) -> None:
    """Heatmaps of bias z-score and CI coverage, scenario × test — the
    effect-size analog of ``_plot_results``'s Type I heatmap. Bias is shown
    as a z-score (not raw bias) because the raw scale differs by test (a
    mean-difference bias and a probability-deviation bias aren't comparable
    numbers); z lets every test sit on the same "how many SEs from the truth"
    axis.

    The scatter plot additionally overlays the UNCORRECTED (raw LLM-only)
    counterpart as gray dots, mirroring ``_plot_results``'s scatter:
      - bias panel: the raw LLM-only estimate's own bias z-score (no PPI
        correction at all).
      - coverage panel: a proxy via test/CI duality — "uncorrected test
        fails to reject the gold-null" is equivalent to "the uncorrected
        (1-α) CI would cover it," so 1 − (uncorrected rejection rate against
        0) approximates uncorrected coverage of the gold-null. This is exact
        for ttest/ttest_welch/mw/wilcoxon/kruskal (gold-null ≈ 0/0.5 there)
        and a good approximation for anova_ind/anova_rep/friedman, whose
        gold-null floor is small relative to the much larger LLM bias this
        panel is dominated by.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n_scenarios = len(SCENARIOS)
    target_cov = 1.0 - ALPHA
    _, z_mat, cov_mat, unc_z_mat, _, rel_width_mat = _effect_matrices(
        effect_samples, gold_null, test_names
    )
    unc_rate_mat = _rate_matrix(uncorrected_counts, uncorrected_totals, test_names)
    unc_cov_mat = 1.0 - unc_rate_mat

    z_finite = z_mat[np.isfinite(z_mat)]
    z_abs_max = max(3.5, float(np.nanmax(np.abs(z_finite))) if z_finite.size else 3.5)
    z_norm = TwoSlopeNorm(vmin=-z_abs_max, vcenter=0.0, vmax=z_abs_max)

    cov_finite = cov_mat[np.isfinite(cov_mat)]
    cov_vmin = min(0.85, float(np.nanmin(cov_finite)) if cov_finite.size else 0.85)
    cov_norm = TwoSlopeNorm(vmin=cov_vmin, vcenter=target_cov, vmax=1.0)

    rw_finite = rel_width_mat[np.isfinite(rel_width_mat)]
    rw_max = max(2.0, float(np.nanmax(rw_finite)) if rw_finite.size else 2.0)
    rw_min = min(0.5, float(np.nanmin(rw_finite)) if rw_finite.size else 0.5)
    rw_norm = TwoSlopeNorm(vmin=rw_min, vcenter=1.0, vmax=rw_max)

    fig_h = max(7.0, 0.34 * n_scenarios)
    fig, (ax1, ax2, ax5) = plt.subplots(1, 3, figsize=(22.0, fig_h))

    im1 = ax1.imshow(z_mat, aspect="auto", cmap="RdBu_r", norm=z_norm)
    ax1.set_xticks(np.arange(len(test_names)))
    ax1.set_xticklabels(test_names, rotation=45, ha="right")
    ax1.set_yticks(np.arange(n_scenarios))
    ax1.set_yticklabels([sc.name for sc in SCENARIOS], fontsize=8)
    ax1.set_title("Bias z-score\n(estimate − gold-reference null, in SEs)")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    cbar1.set_label("z")

    im2 = ax2.imshow(cov_mat, aspect="auto", cmap="RdYlGn", norm=cov_norm)
    ax2.set_xticks(np.arange(len(test_names)))
    ax2.set_xticklabels(test_names, rotation=45, ha="right")
    ax2.set_yticks(np.arange(n_scenarios))
    ax2.set_yticklabels([])
    ax2.set_title(f"CI coverage of truth\n(target = {target_cov:.2f})")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cbar2.set_label("coverage")

    im5 = ax5.imshow(rel_width_mat, aspect="auto", cmap="PuOr_r", norm=rw_norm)
    ax5.set_xticks(np.arange(len(test_names)))
    ax5.set_xticklabels(test_names, rotation=45, ha="right")
    ax5.set_yticks(np.arange(n_scenarios))
    ax5.set_yticklabels([])
    ax5.set_title("CI width relative to\neach test's own median")
    cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.04, pad=0.02)
    cbar5.set_label("width / median width")

    for ax in (ax1, ax2, ax5):
        for i in range(1, n_scenarios):
            if SCENARIOS[i].tag != SCENARIOS[i - 1].tag:
                ax.axhline(i - 0.5, color="white", lw=1.2, alpha=0.9)

    fig.suptitle("Effect-size calibration: bias, CI coverage & CI width of the corrected estimate")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    scatter_out_path = str(Path(out_path).with_name(Path(out_path).stem + "_scatter.png"))
    fig2, (ax3, ax4, ax6) = plt.subplots(1, 3, figsize=(19.0, 5.8))
    rng = np.random.default_rng(0)
    colors = [plt.cm.tab10(i) for i in range(len(test_names))]
    unc_label_added = False

    for j, t in enumerate(test_names):
        unc_zvals = unc_z_mat[:, j]
        keep_unc = np.isfinite(unc_zvals)
        x_unc = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_unc)))
        ax3.scatter(
            x_unc, unc_zvals[keep_unc], s=18, alpha=0.35, color="#808080",
            label="uncorrected" if not unc_label_added else None, zorder=1,
        )

        unc_covvals = unc_cov_mat[:, j]
        keep_unc_c = np.isfinite(unc_covvals)
        x_unc_c = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_unc_c)))
        ax4.scatter(
            x_unc_c, unc_covvals[keep_unc_c], s=18, alpha=0.35, color="#808080",
            label="uncorrected" if not unc_label_added else None, zorder=1,
        )
        unc_label_added = True

        zvals = z_mat[:, j]
        keep = np.isfinite(zvals)
        x = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep)))
        ax3.scatter(x, zvals[keep], s=20, alpha=0.65, color=colors[j], zorder=2)

        covvals = cov_mat[:, j]
        keep_c = np.isfinite(covvals)
        x_c = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_c)))
        ax4.scatter(x_c, covvals[keep_c], s=20, alpha=0.65, color=colors[j], zorder=2)

        rwvals = rel_width_mat[:, j]
        keep_rw = np.isfinite(rwvals)
        x_rw = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_rw)))
        ax6.scatter(x_rw, rwvals[keep_rw], s=20, alpha=0.65, color=colors[j], zorder=2)

    ax3.axhline(0.0, color="black", ls="--", lw=1.1)
    ax3.axhline(3.0, color="gray", ls=":", lw=0.9)
    ax3.axhline(-3.0, color="gray", ls=":", lw=0.9)
    ax3.set_xticks(np.arange(len(test_names)))
    ax3.set_xticklabels(test_names, rotation=45, ha="right")
    ax3.set_ylabel("bias z-score")
    ax3.set_title("Bias z-score (all scenario × test cells)")
    ax3.grid(axis="y", alpha=0.25, lw=0.8)
    ax3.legend(loc="upper right", fontsize=8)
    # Uncorrected z's can be an order of magnitude larger than corrected ones
    # (a persistent LLM bias keeps growing in z as reps shrink its SE) — symlog
    # keeps the near-zero corrected dots legible while still showing the
    # uncorrected scale, instead of one or the other being squashed flat.
    ax3.set_yscale("symlog", linthresh=5.0)

    ax4.axhline(target_cov, color="black", ls="--", lw=1.1, label=f"target={target_cov:.2f}")
    ax4.set_xticks(np.arange(len(test_names)))
    ax4.set_xticklabels(test_names, rotation=45, ha="right")
    ax4.set_ylim(0.0, 1.02)
    ax4.set_ylabel("CI coverage")
    ax4.set_title("CI coverage (all scenario × test cells)")
    ax4.grid(axis="y", alpha=0.25, lw=0.8)
    ax4.legend(loc="lower right", fontsize=8)

    ax6.axhline(1.0, color="black", ls="--", lw=1.1, label="this test's median width")
    ax6.set_xticks(np.arange(len(test_names)))
    ax6.set_xticklabels(test_names, rotation=45, ha="right")
    ax6.set_ylabel("CI width / test's median width")
    ax6.set_title("Relative CI width (all scenario × test cells)\n(no uncorrected analog — PPI bootstrap CI only)")
    ax6.set_yscale("log")
    ax6.grid(axis="y", alpha=0.25, lw=0.8)
    ax6.legend(loc="upper right", fontsize=8)

    fig2.tight_layout()
    fig2.savefig(scatter_out_path, dpi=180, bbox_inches="tight")
    print(f"Plot saved to {scatter_out_path}")

    backend = plt.get_backend().lower()
    if backend not in {"agg", "pdf", "svg", "ps", "cairo", "pgf", "template"}:
        plt.show()


def _plot_power_results(
    counts: dict,
    totals: dict,
    uncorrected_counts: dict,
    uncorrected_totals: dict,
    n_reps: int,
    n_boot: int,
    effect_size: float,
    out_path: str,
    test_names: list[str] = TEST_NAMES,
    power_flag: float = 0.80,
) -> None:
    """Heatmap + scatter of statistical power, scenario × test — the power
    analog of ``_plot_results``'s Type I heatmap. Color/flag direction is
    inverted from the Type I plot: here HIGH rejection rate is good (real
    effect correctly detected), so the diverging scale is centered at
    ``power_flag`` (0.80) rather than α, and red marks LOW power, not high.

    The scatter's gray "uncorrected" dots are shown for context only, not as
    a baseline to beat — every scenario here still carries its own LLM
    bias_type artifact on top of the real injected effect, so an uncorrected
    rejection can come from either the genuine effect or the bias alone.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n_scenarios = len(SCENARIOS)
    mat = _rate_matrix(counts, totals, test_names)
    unc_mat = _rate_matrix(uncorrected_counts, uncorrected_totals, test_names)

    norm = TwoSlopeNorm(vmin=0.0, vcenter=power_flag, vmax=1.0)

    fig_h = max(7.0, 0.34 * n_scenarios)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", norm=norm)

    ax.set_xticks(np.arange(len(test_names)))
    ax.set_xticklabels(test_names, rotation=0)
    ax.set_yticks(np.arange(n_scenarios))
    ax.set_yticklabels([sc.name for sc in SCENARIOS], fontsize=8)
    ax.set_xlabel("Test")
    ax.set_ylabel("Scenario")
    ax.set_title(
        f"Power heatmap (effect_size={effect_size}, reps={n_reps}, n_boot={n_boot})"
    )

    for i in range(1, n_scenarios):
        if SCENARIOS[i].tag != SCENARIOS[i - 1].tag:
            ax.axhline(i - 0.5, color="white", lw=1.2, alpha=0.9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Power (rejection rate under real effect)")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    scatter_out_path = str(Path(out_path).with_name(Path(out_path).stem + "_scatter.png"))
    fig2, ax2 = plt.subplots(figsize=(10.0, 5.8))
    rng = np.random.default_rng(0)
    colors = [plt.cm.tab10(i) for i in range(len(test_names))]
    unc_label_added = False

    for j, t in enumerate(test_names):
        vals_unc = unc_mat[:, j]
        keep_unc = np.isfinite(vals_unc)
        y_unc = vals_unc[keep_unc]
        x_unc = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_unc)))
        ax2.scatter(
            x_unc, y_unc, s=18, alpha=0.35, color="#808080",
            label="uncorrected (not a fair baseline — see title)" if not unc_label_added else None,
            zorder=1,
        )
        unc_label_added = True

        vals = mat[:, j]
        keep = np.isfinite(vals)
        y = vals[keep]
        x = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep)))
        ax2.scatter(x, y, s=20, alpha=0.65, color=colors[j], label=t, zorder=2)

    ax2.axhline(power_flag, color="black", ls="--", lw=1.1, label=f"power_flag={power_flag:.2f}")
    ax2.set_xlim(-0.5, len(test_names) - 0.5)
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xticks(np.arange(len(test_names)))
    ax2.set_xticklabels(test_names)
    ax2.set_ylabel("Power (rejection rate)")
    ax2.set_xlabel("Test")
    ax2.set_title(
        f"Power scatter, all scenario × test cells (effect_size={effect_size})\n"
        "gray = uncorrected rate, includes LLM bias_type artifacts — not a fair power baseline"
    )
    ax2.grid(axis="y", alpha=0.25, lw=0.8)
    ax2.legend(loc="lower right", fontsize=8, ncol=2)

    fig2.tight_layout()
    fig2.savefig(scatter_out_path, dpi=180, bbox_inches="tight")
    print(f"Plot saved to {scatter_out_path}")

    backend = plt.get_backend().lower()
    if backend not in {"agg", "pdf", "svg", "ps", "cairo", "pgf", "template"}:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Type I error calibration study for PPI-corrected hypothesis tests."
    )
    parser.add_argument("--reps",   type=int, default=500,
                        help="Bootstrap replications per scenario (default 500)")
    parser.add_argument("--n-boot", type=int, default=300,
                        help="Bootstrap draws per test call (default 300)")
    parser.add_argument("--jobs",   type=int, default=max(1, mp.cpu_count() - 1),
                        help="Parallel workers (default: cpu_count−1)")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Add to all seeds (use to run independent replicates)")
    parser.add_argument("--out-csv", type=str, default=None,
                        help="Path to save results as CSV")
    parser.add_argument("--plot", action="store_true",
                        help="Generate calibration heatmap + scatterplot and save to simulations/out/plots/ with unique filenames")
    parser.add_argument("--tests", type=str, nargs="+", choices=TEST_NAMES, default=None,
                        metavar="TEST",
                        help=f"Restrict to a subset of tests (choices: {', '.join(TEST_NAMES)}). "
                             "Runs the exact same scenario sweep but skips computation for any "
                             "test not listed — e.g. --tests friedman to isolate and speed up "
                             "just the Friedman calibration check. Default: run all.")
    parser.add_argument("--effect-reps", type=int, default=200,
                        help="Replications for the anova_ind/anova_rep/friedman effect-size "
                             "bias/coverage check (default 200). This is a separate, smaller pass "
                             "specifically for those three tests — they normally use closed-form "
                             "p-value-only functions in the main sweep to stay fast, so checking "
                             "their corrected estimate's bias/coverage means a second, dedicated "
                             "call to their bootstrap-based estimate function, decoupled from "
                             "--reps so it doesn't multiply the main sweep's cost. ttest/ttest_welch/"
                             "mw/wilcoxon/kruskal already bootstrap in the main sweep, so their "
                             "effect-size check is free and always uses --reps.")
    parser.add_argument("--no-effect-check", action="store_true",
                        help="Skip the effect-size bias/coverage check entirely (Type I only).")
    parser.add_argument("--effect-size", type=float, default=None,
                        help="Inject a REAL mean shift of this size into every scenario's truth "
                             "(monotonic across groups/conditions — see Scenario.effect_size), "
                             "turning the Type I sweep into a POWER sweep over the exact same "
                             "scenario grid: rejection rate becomes power instead of Type I error. "
                             "Automatically disables the bias/coverage/width effect-size check, "
                             "since its gold-reference null assumes H0. Default: None (Type I mode, "
                             "effect_size=0 everywhere).")
    args = parser.parse_args()

    N_REPS  = args.reps
    N_BOOT  = args.n_boot
    JOBS    = args.jobs
    OFFSET  = args.seed_offset
    EFFECT_REPS = args.effect_reps
    ACTIVE_TESTS = [t for t in TEST_NAMES if t in set(args.tests)] if args.tests else list(TEST_NAMES)
    EFFECT_CHECK_TESTS = [t for t in ACTIVE_TESTS if t in _EFFECT_CHECK_TESTS]
    POWER_MODE = args.effect_size is not None and args.effect_size != 0.0
    EFFECT_SIZE_OVERRIDE = args.effect_size

    n_scenarios = len(SCENARIOS)
    work = [
        (sc_idx, OFFSET + seed, N_BOOT, ACTIVE_TESTS, EFFECT_SIZE_OVERRIDE)
        for sc_idx in range(n_scenarios)
        for seed in range(1000, 1000 + N_REPS)
    ]
    total_jobs = len(work)

    counts = {i: {t: 0 for t in TEST_NAMES} for i in range(n_scenarios)}
    totals = {i: {t: 0 for t in TEST_NAMES} for i in range(n_scenarios)}
    uncorrected_counts = {i: {t: 0 for t in TEST_NAMES} for i in range(n_scenarios)}
    uncorrected_totals = {i: {t: 0 for t in TEST_NAMES} for i in range(n_scenarios)}
    effect_samples: dict[int, dict[str, list[tuple[float, float, float, float]]]] = {
        i: {t: [] for t in TEST_NAMES} for i in range(n_scenarios)
    }

    run_effect_check = (not args.no_effect_check) and (not POWER_MODE)
    if POWER_MODE and not args.no_effect_check:
        print("\nPower mode: skipping the bias/coverage/width effect-size check "
              "(its gold-reference null assumes H0, which doesn't hold once a real "
              "effect is injected).")
    if run_effect_check:
        print(f"\nComputing gold-reference null values for {n_scenarios} scenarios (Monte Carlo, no bootstrap)...")
        t_gold = time.time()
        gold_null = {i: _gold_null_values(SCENARIOS[i], seed=i) for i in range(n_scenarios)}
        print(f"  done in {time.time()-t_gold:.1f}s")

    if POWER_MODE:
        print(
            f"\nPower sweep: {n_scenarios} scenarios × {N_REPS} reps × "
            f"{len(ACTIVE_TESTS)} tests = {total_jobs} jobs  "
            f"({JOBS} workers, n_boot={N_BOOT}, effect_size={EFFECT_SIZE_OVERRIDE})"
        )
    else:
        print(
            f"\nType I Calibration: {n_scenarios} scenarios × {N_REPS} reps × "
            f"{len(ACTIVE_TESTS)} tests = {total_jobs} jobs  "
            f"({JOBS} workers, n_boot={N_BOOT})"
        )
    if ACTIVE_TESTS != TEST_NAMES:
        print(f"  Active tests: {', '.join(ACTIVE_TESTS)}")

    t0 = time.time()
    done = 0
    report_every = max(1, total_jobs // 20)

    with mp.Pool(processes=JOBS) as pool:
        for sc_idx, corrected_result, uncorrected_result, effect_result in pool.imap_unordered(
            _run_one, work, chunksize=20
        ):
            for t, rejected in corrected_result.items():
                totals[sc_idx][t] += 1
                if rejected:
                    counts[sc_idx][t] += 1
            for t, rejected in uncorrected_result.items():
                uncorrected_totals[sc_idx][t] += 1
                if rejected:
                    uncorrected_counts[sc_idx][t] += 1
            for t, triple in effect_result.items():
                effect_samples[sc_idx][t].append(triple)
            done += 1
            if done % report_every == 0 or done == total_jobs:
                elapsed = time.time() - t0
                eta = elapsed / done * (total_jobs - done)
                print(
                    f"  {done:>6}/{total_jobs}  "
                    f"({100*done/total_jobs:.0f}%)  "
                    f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                    flush=True,
                )

    print(f"\nFinished in {time.time()-t0:.1f}s")
    if POWER_MODE:
        _print_power_table(
            counts, totals, uncorrected_counts, uncorrected_totals,
            N_REPS, N_BOOT, EFFECT_SIZE_OVERRIDE, ACTIVE_TESTS,
        )
    else:
        _print_table(counts, totals, uncorrected_counts, uncorrected_totals, N_REPS, N_BOOT, ACTIVE_TESTS)

    if run_effect_check:
        anova_tests_active = [t for t in _EFFECT_ANOVA_TESTS if t in ACTIVE_TESTS]
        if anova_tests_active and EFFECT_REPS > 0:
            effect_work = [
                (sc_idx, OFFSET + 500_000 + seed, N_BOOT, anova_tests_active)
                for sc_idx in range(n_scenarios)
                for seed in range(EFFECT_REPS)
            ]
            print(
                f"\nEffect-size check (anova_ind/anova_rep/friedman only): "
                f"{n_scenarios} scenarios × {EFFECT_REPS} reps × {len(anova_tests_active)} tests "
                f"= {len(effect_work)} jobs  ({JOBS} workers, n_boot={N_BOOT})"
            )
            t1 = time.time()
            with mp.Pool(processes=JOBS) as pool:
                for sc_idx, effect_result in pool.imap_unordered(
                    _run_one_effect_anova, effect_work, chunksize=20
                ):
                    for t, triple in effect_result.items():
                        effect_samples[sc_idx][t].append(triple)
            print(f"Finished effect-size pass in {time.time()-t1:.1f}s")

        _print_effect_table(effect_samples, gold_null, EFFECT_CHECK_TESTS)

    if args.out_csv:
        _save_csv(counts, totals, uncorrected_counts, uncorrected_totals, args.out_csv, ACTIVE_TESTS)

    if args.plot:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        nonce = str(time.time_ns() % 1_000_000_000).zfill(9)

        if POWER_MODE:
            power_fname = f"power_{stamp}_{nonce}.png"
            default_power_plot_path = os.path.join(_HERE, "out", "plots", power_fname)
            _plot_power_results(
                counts, totals, uncorrected_counts, uncorrected_totals,
                N_REPS, N_BOOT, EFFECT_SIZE_OVERRIDE, default_power_plot_path, ACTIVE_TESTS,
            )
        else:
            fname = f"type_i_calibration_{stamp}_{nonce}.png"
            default_plot_path = os.path.join(_HERE, "out", "plots", fname)
            _plot_results(
                counts,
                totals,
                uncorrected_counts,
                uncorrected_totals,
                N_REPS,
                N_BOOT,
                default_plot_path,
                ACTIVE_TESTS,
            )

            if run_effect_check:
                effect_fname = f"effect_calibration_{stamp}_{nonce}.png"
                default_effect_plot_path = os.path.join(_HERE, "out", "plots", effect_fname)
                _plot_effect_results(
                    effect_samples, gold_null, uncorrected_counts, uncorrected_totals,
                    default_effect_plot_path, EFFECT_CHECK_TESTS,
                )


if __name__ == "__main__":
    main()
