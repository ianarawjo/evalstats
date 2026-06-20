#!/usr/bin/env python3
"""
sim_type_i_calibration.py — Type I error calibration study for PPI-corrected tests.

Tests five PPI-corrected hypothesis tests across a broad parameter sweep to check
whether the corrected p-value stays near α = 0.05 under H₀:

  ttest       independent mean difference (two groups)
  mannwhitney independent P(X > Y) − 0.5 (two groups)
  wilcoxon    paired median difference (same subjects, two conditions)
  anova_ind   one-way ANOVA, between-subjects (three groups)
  anova_rep   one-way ANOVA, within-subjects / repeated-measures (three conditions)

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

The internal PPI functions (_ppi_two_sample, _ppi_paired_arrays, etc.) are called
directly to skip the validate_alignment overhead (~360ms/call) — we are testing
statistical calibration, not the pipeline UX.

Usage:
    python simulations/sim_type_i_calibration.py
    python simulations/sim_type_i_calibration.py --reps 200 --n-boot 100   # quick
    python simulations/sim_type_i_calibration.py --reps 1000 --jobs 8      # thorough
    python simulations/sim_type_i_calibration.py --plot                    # save + show figures
    python simulations/sim_type_i_calibration.py --out-csv simulations/out/type_i.csv
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
    )


# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA = 0.05
TEST_NAMES = ["ttest", "mw", "wilcoxon", "anova_ind", "anova_rep"]
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


# ── Worker ────────────────────────────────────────────────────────────────────

def _run_one(args: tuple) -> tuple[int, dict[str, bool | None], dict[str, bool | None]]:
    """Process one (scenario_idx, seed, n_boot) triplet.

    Returns (sc_idx, corrected_results, uncorrected_results).
    None means the p-value could not be computed (e.g. zero labeled items).
    """
    sc_idx, seed, n_boot = args
    sc: Scenario = SCENARIOS[sc_idx]
    rng = np.random.default_rng(seed)

    n1 = sc.n
    n2 = sc.n2 if sc.n2 is not None else sc.n
    n3 = sc.n3 if sc.n3 is not None else sc.n
    noise1 = sc.llm_noise
    noise2 = sc.llm_noise2 if sc.llm_noise2 is not None else sc.llm_noise
    noise3 = sc.llm_noise3 if sc.llm_noise3 is not None else sc.llm_noise
    bias_a, bias_b, bias_c = _biases(sc)

    # ── Independent two-group data (ttest, mannwhitney) ────────────────────
    truth_a2 = _sample_truth(sc.dist, n1, rng)
    truth_b2 = _sample_truth(sc.dist, n2, rng)
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
        truth_y = base_2 + rng.normal(0.0, _SIGMA_COND, n1)
    llm_x = _llm(truth_x, bias_a, noise1, rng)
    llm_y = _llm(truth_y, bias_b, noise2, rng)
    lab_x, lab_y = _labels_shared([truth_x, truth_y], sc.label_frac, rng)

    # ── Independent three-group data (anova_ind) ───────────────────────────
    truth_a3 = _sample_truth(sc.dist, n1, rng)
    truth_b3 = _sample_truth(sc.dist, n2, rng)
    truth_c3 = _sample_truth(sc.dist, n3, rng)
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
        truth_B = base_3 + rng.normal(0.0, _SIGMA_COND, n1)
        truth_C = base_3 + rng.normal(0.0, _SIGMA_COND, n1)
    llm_A = _llm(truth_A, bias_a, noise1, rng)
    llm_B = _llm(truth_B, bias_b, noise2, rng)
    llm_C = _llm(truth_C, bias_c, noise3, rng)
    lab_A, lab_B, lab_C = _labels_shared([truth_A, truth_B, truth_C], sc.label_frac, rng)

    corrected_results: dict[str, bool | None] = {}
    uncorrected_results: dict[str, bool | None] = {}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ttest: estimand = E[A] − E[B]
        try:
            p_uncorrected = float(_scipy_stats.ttest_ind(llm_a2, llm_b2).pvalue)
            uncorrected_results["ttest"] = p_uncorrected < ALPHA
            r = _ppi_two_sample(
                llm_a2, llm_b2, lab_a2, lab_b2,
                lambda ya, yb: float(ya.mean() - yb.mean()),
                ALPHA, n_boot, _rng_seed(),
            )
            corrected_results["ttest"] = r.p_value < ALPHA
        except Exception:
            corrected_results["ttest"] = None
            uncorrected_results["ttest"] = None

        # mannwhitney: mid-rank estimand = P_mid(X>Y) - 0.5; 0 under H₀ for any dist.
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
        except Exception:
            corrected_results["mw"] = None
            uncorrected_results["mw"] = None

        # wilcoxon: estimand = median(X − Y), paired
        try:
            p_uncorrected = float(_scipy_stats.wilcoxon(llm_x, llm_y, alternative="two-sided").pvalue)
            uncorrected_results["wilcoxon"] = p_uncorrected < ALPHA
            r = _ppi_paired_arrays(
                llm_x, llm_y, lab_x, lab_y,
                np.median, ALPHA, n_boot, _rng_seed(),
                rectifier_func=np.mean,
            )
            corrected_results["wilcoxon"] = r.p_value < ALPHA
        except Exception:
            corrected_results["wilcoxon"] = None
            uncorrected_results["wilcoxon"] = None

        # anova_ind: direct closed-form F-test
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

    return sc_idx, corrected_results, uncorrected_results


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


def _rate_matrix(counts: dict, totals: dict) -> np.ndarray:
    mat = np.full((len(SCENARIOS), len(TEST_NAMES)), np.nan, dtype=float)
    for i in range(len(SCENARIOS)):
        for j, t in enumerate(TEST_NAMES):
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
            for t in TEST_NAMES
        }

    def uncorrected_rates(sc_idx: int) -> dict[str, float | None]:
        return {
            t: uncorrected_counts[sc_idx][t] / uncorrected_totals[sc_idx][t]
            if uncorrected_totals[sc_idx][t] > 0 else None
            for t in TEST_NAMES
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

    col_names = ["ttest", "mw", "wilcoxon", "anova_i", "anova_r"]
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
        for t in TEST_NAMES:
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
            for t in TEST_NAMES:
                row += f" {_fmt_rate(r.get(t), flag2, flag3)}"
            if any(alpha_outside_ci(i, t) for t in TEST_NAMES):
                row += "  †"
            if any((i, t) in holm_bad for t in TEST_NAMES):
                row += "‡"
            print(row)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print(bar)
    print("SUMMARY")
    print()

    n_conditions = len(SCENARIOS) * len(TEST_NAMES)

    flags2 = sum(
        1 for r in all_rates for t in TEST_NAMES
        if r.get(t) is not None and r[t] > flag2
    )
    flags3 = sum(
        1 for r in all_rates for t in TEST_NAMES
        if r.get(t) is not None and r[t] > flag3
    )
    wilson_miscal = sum(
        1
        for sc_idx in range(len(SCENARIOS))
        for t in TEST_NAMES
        if alpha_outside_ci(sc_idx, t)
    )
    nominal_miscal = sum(1 for _, p in pvals if p < 0.05)
    uncorrected_flags2 = sum(
        1 for r in all_uncorrected_rates for t in TEST_NAMES
        if r.get(t) is not None and r[t] > flag2
    )
    uncorrected_flags3 = sum(
        1 for r in all_uncorrected_rates for t in TEST_NAMES
        if r.get(t) is not None and r[t] > flag3
    )

    print(f"  Scenarios: {len(SCENARIOS)}  |  Tests: {len(TEST_NAMES)}  |  Conditions: {n_conditions}")
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
    for t in TEST_NAMES:
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


def _save_csv(
    counts: dict,
    totals: dict,
    uncorrected_counts: dict,
    uncorrected_totals: dict,
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scenario", "tag"]
            + [f"{t}_corrected" for t in TEST_NAMES]
            + [f"{t}_uncorrected" for t in TEST_NAMES]
            + [f"{t}_corrected_n" for t in TEST_NAMES]
            + [f"{t}_uncorrected_n" for t in TEST_NAMES]
        )
        for sc_idx, sc in enumerate(SCENARIOS):
            row = [sc.name, sc.tag]
            for t in TEST_NAMES:
                tot = totals[sc_idx][t]
                row.append(f"{counts[sc_idx][t] / tot:.4f}" if tot else "")
            for t in TEST_NAMES:
                tot = uncorrected_totals[sc_idx][t]
                row.append(f"{uncorrected_counts[sc_idx][t] / tot:.4f}" if tot else "")
            for t in TEST_NAMES:
                row.append(totals[sc_idx][t])
            for t in TEST_NAMES:
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
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n_scenarios = len(SCENARIOS)
    mat = _rate_matrix(counts, totals)
    uncorrected_mat = _rate_matrix(uncorrected_counts, uncorrected_totals)

    # Center the color scale at alpha so well-calibrated cells are neutral.
    finite = mat[np.isfinite(mat)]
    vmax = max(ALPHA * 1.8, float(np.nanmax(finite)) if finite.size else ALPHA)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=ALPHA, vmax=vmax)

    fig_h = max(7.0, 0.34 * n_scenarios)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", norm=norm)

    ax.set_xticks(np.arange(len(TEST_NAMES)))
    ax.set_xticklabels(["ttest", "mw", "wilcoxon", "anova_ind", "anova_rep"], rotation=0)
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
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    unc_label_added = False

    for j, t in enumerate(TEST_NAMES):
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
    ax2.set_xlim(-0.5, len(TEST_NAMES) - 0.5)
    scatter_max = np.nanmax(np.concatenate([mat.ravel(), uncorrected_mat.ravel()]))
    if not np.isfinite(scatter_max):
        scatter_max = 0.2
    ax2.set_ylim(0.0, max(0.2, float(scatter_max) * 1.05))
    ax2.set_xticks(np.arange(len(TEST_NAMES)))
    ax2.set_xticklabels(["ttest", "mw", "wilcoxon", "anova_ind", "anova_rep"])
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
    args = parser.parse_args()

    N_REPS  = args.reps
    N_BOOT  = args.n_boot
    JOBS    = args.jobs
    OFFSET  = args.seed_offset

    n_scenarios = len(SCENARIOS)
    work = [
        (sc_idx, OFFSET + seed, N_BOOT)
        for sc_idx in range(n_scenarios)
        for seed in range(1000, 1000 + N_REPS)
    ]
    total_jobs = len(work)

    counts = {i: {t: 0 for t in TEST_NAMES} for i in range(n_scenarios)}
    totals = {i: {t: 0 for t in TEST_NAMES} for i in range(n_scenarios)}
    uncorrected_counts = {i: {t: 0 for t in TEST_NAMES} for i in range(n_scenarios)}
    uncorrected_totals = {i: {t: 0 for t in TEST_NAMES} for i in range(n_scenarios)}

    print(
        f"\nType I Calibration: {n_scenarios} scenarios × {N_REPS} reps × "
        f"{len(TEST_NAMES)} tests = {total_jobs} jobs  "
        f"({JOBS} workers, n_boot={N_BOOT})"
    )

    t0 = time.time()
    done = 0
    report_every = max(1, total_jobs // 20)

    with mp.Pool(processes=JOBS) as pool:
        for sc_idx, corrected_result, uncorrected_result in pool.imap_unordered(_run_one, work, chunksize=20):
            for t, rejected in corrected_result.items():
                totals[sc_idx][t] += 1
                if rejected:
                    counts[sc_idx][t] += 1
            for t, rejected in uncorrected_result.items():
                uncorrected_totals[sc_idx][t] += 1
                if rejected:
                    uncorrected_counts[sc_idx][t] += 1
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
    _print_table(counts, totals, uncorrected_counts, uncorrected_totals, N_REPS, N_BOOT)

    if args.out_csv:
        _save_csv(counts, totals, uncorrected_counts, uncorrected_totals, args.out_csv)

    if args.plot:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        nonce = str(time.time_ns() % 1_000_000_000).zfill(9)
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
        )


if __name__ == "__main__":
    main()
