#!/usr/bin/env python3
"""
sim_compare_boot.py — Bootstrap CI method comparison across eval score types.

Compares five bootstrap confidence-interval methods by running Monte Carlo
simulations where the true population mean is known exactly.

Methods:
  bootstrap        Percentile bootstrap
  bca              Bias-corrected and accelerated (BCa) bootstrap
  bayes_bootstrap  Bayesian (Dirichlet-weighted) bootstrap
  smooth_bootstrap Smoothed (KDE-perturbed) bootstrap
  bootstrap_t      Studentized (bootstrap-t) bootstrap
    wilson           Wilson score CI for single-sample binary means
    jeffreys        Jeffreys interval for single-sample binary means
    newcombe_score   Newcombe score CI for paired binary differences
    tango_score      Tango score CI for paired binary differences
    bayes_indep      Beta-conjugate Bayesian interval for binary means
    bayes_indep_comp Independent Beta-posteriors interval for paired binaries
    bayes_paired_comp Paired Bayesian interval using bayes_evals latent model

Eval output types:
  binary      Bernoulli 0/1 (pass/fail judgements)
  continuous  Continuous floats in [0, 1] via Beta distributions
  likert      Integer scores 1–5 (Likert-scale rubrics)
  grades      Scores 0–100 (test-like, truncated normal)

Metrics:
  coverage    Fraction of (1-alpha)*100% CIs that contain the true mean
              Target: equals the nominal level, e.g. 0.95 for alpha=0.05
  mean_width  Average CI width — a measure of precision (smaller is better
              when coverage is adequate)
    mc_uncert   Monte Carlo uncertainty for coverage with MCSE + 95% bands

Usage:
  python simulations/sim_compare_boot.py
    python simulations/sim_compare_boot.py --scenario-suite expanded
    python simulations/sim_compare_boot.py --progress bar
        python simulations/sim_compare_boot.py --official-test-pairwise-only
  python simulations/sim_compare_boot.py --reps 500 --bootstrap-n 1000
        python simulations/sim_compare_boot.py --bayes-n 2000
        python simulations/sim_compare_boot.py --out-dir simulations/out --save-results save
        python simulations/sim_compare_boot.py --plots save --plots-dir simulations/out/plots
  python simulations/sim_compare_boot.py --sizes 5 10 20 50 100
    python simulations/sim_compare_boot.py --estimand pairwise --runs 3
    python simulations/sim_compare_boot.py --official-test
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# Ensure both simulations/ (for bayes_evals) and the project root (for evalstats
# source) are on the path before any installed copies in site-packages.
for _p in [_HERE, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from bayes_evals import binorm_cdf  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.resampling import (
        bootstrap_ci_1d,
        bootstrap_t_ci_1d,
        bca_interval_1d,
        bayes_bootstrap_means_1d,
        smooth_bootstrap_means_1d,
        bootstrap_means_1d,
        bootstrap_diffs_nested,
        bayes_bootstrap_diffs_nested,
        smooth_bootstrap_diffs_nested,
        resolve_resampling_method,
        wald_ci_1d,
        clopper_pearson_ci_1d,
        jeffreys_ci_1d,
        t_interval_ci_1d,
        beta_ci_1d,
        logit_t_ci_1d,
        nig_ci_1d,
        el_ci_1d,
        tango_paired_ci,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t"]
WILSON_METHOD = "wilson"
JEFFREYS_METHOD = "jeffreys"
NEWCOMBE_METHOD = "newcombe_score"
TANGO_METHOD = "tango_score"
BAYES_SINGLE_METHOD = "bayes_indep"
BAYES_PAIR_INDEP_METHOD = "bayes_indep_comp"
BAYES_PAIR_PAIRED_METHOD = "bayes_paired_comp"
WALD_METHOD = "wald"
CP_METHOD = "clopper_pearson"
T_INTERVAL_METHOD = "t_interval"
BETA_METHOD = "beta"
LOGIT_T_METHOD = "logit_t"
NIG_METHOD = "nig"
EL_METHOD = "el"
# Methods only applied to continuous (non-binary) eval types
CONTINUOUS_EXTRA_METHODS = [BETA_METHOD, LOGIT_T_METHOD, NIG_METHOD, EL_METHOD]
PAIRWISE_EXTRA_METHODS = [T_INTERVAL_METHOD, NIG_METHOD, EL_METHOD]
REPORT_METHODS = METHODS + [
    T_INTERVAL_METHOD,
    WILSON_METHOD,
    JEFFREYS_METHOD,
    NEWCOMBE_METHOD,
    TANGO_METHOD,
    WALD_METHOD,
    CP_METHOD,
    BAYES_SINGLE_METHOD,
    BAYES_PAIR_INDEP_METHOD,
    BAYES_PAIR_PAIRED_METHOD,
] + CONTINUOUS_EXTRA_METHODS
EVAL_TYPES = ["binary", "continuous", "likert", "grades"]
SCENARIO_SUITES = ["standard", "expanded", "extreme"]
PROGRESS_MODES = ["bar", "cell", "off"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    label: str
    eval_type: str
    generate: Callable[[np.random.Generator, int], np.ndarray]
    true_mean: float


@dataclass
class PairScenario:
    label: str
    eval_type: str
    generate_pair: Callable[[np.random.Generator, int, int], tuple[np.ndarray, np.ndarray]]
    true_diff: float
    icc: float = 0.0       # intraclass correlation of per-input scores
    cohens_d: float = 0.0  # standardized effect size (delta / total_std)
    is_null: bool = False  # True when delta=0 (used for Type I error measurement)


class _ProgressReporter:
    """Lightweight terminal progress reporter with optional ETA."""

    def __init__(self, total: int, *, mode: str = "bar", label: str = "") -> None:
        self.total = max(int(total), 1)
        self.mode = mode
        self.label = label
        self.start = time.time()
        self.last_print = 0.0

    def update(self, step: int, detail: str = "") -> None:
        if self.mode == "off":
            return

        now = time.time()
        is_final = step >= self.total
        if not is_final and (now - self.last_print) < 0.2:
            return
        self.last_print = now

        if self.mode == "cell":
            pct = 100.0 * min(step, self.total) / self.total
            print(
                f"\r  [{step:>7d}/{self.total:<7d}] {pct:6.2f}%  {detail:<55s}",
                end="",
                flush=True,
            )
            if is_final:
                print()
            return

        # bar mode
        frac = min(step, self.total) / self.total
        filled = int(28 * frac)
        bar = "█" * filled + "·" * (28 - filled)
        elapsed = max(now - self.start, 1e-9)
        rate = step / elapsed
        rem = max(self.total - step, 0)
        eta_sec = rem / max(rate, 1e-12)
        eta_m, eta_s = divmod(int(round(eta_sec)), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
        prefix = f"{self.label}: " if self.label else ""
        print(
            f"\r  {prefix}[{bar}] {100.0*frac:6.2f}%  {step:>7d}/{self.total:<7d}  ETA {eta_str}  {detail[:40]:<40s}",
            end="",
            flush=True,
        )
        if is_final:
            print()


def _true_mean_clipped_normal(
    mu: float, sigma: float, lo: float = 0.0, hi: float = 100.0
) -> float:
    """Population mean of Normal(mu, sigma) clipped to [lo, hi] via large sample."""
    rng = np.random.default_rng(0)
    return float(np.clip(rng.normal(mu, sigma, size=2_000_000), lo, hi).mean())


def _estimate_true_mean_mc(
    generate: Callable[[np.random.Generator, int], np.ndarray],
    *,
    seed: int = 0,
    n_mc: int = 500_000,
) -> float:
    """Estimate population mean via large Monte Carlo draw for complex generators.

    Uses n_mc=500,000 samples. For all scenarios used here the resulting MC
    standard error is < 0.001 (well below the CI widths under study), so
    estimand error is negligible relative to method-comparison noise.
    """
    rng = np.random.default_rng(seed)
    return float(np.mean(generate(rng, n_mc)))


def build_scenarios(suite: str = "standard") -> list[Scenario]:
    """Return simulation scenarios across the four eval types."""
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")

    scenarios: list[Scenario] = []

    # ------------------------------------------------------------------
    # Binary: Bernoulli(p)  →  0/1 pass-fail judgements
    # ------------------------------------------------------------------
    binary_ps = [0.1, 0.3, 0.5, 0.7, 0.9]
    if suite == "expanded":
        binary_ps += [0.02, 0.05, 0.95, 0.98]

    for p in binary_ps:
        p_ = p
        scenarios.append(
            Scenario(
                label=f"p={p_}",
                eval_type="binary",
                generate=lambda rng, n, _p=p_: rng.binomial(1, _p, n).astype(float),
                true_mean=p_,
            )
        )

    # ------------------------------------------------------------------
    # Continuous [0, 1]: Beta distributions of varying shape
    # ------------------------------------------------------------------
    continuous_specs = [
        ("Uniform",        1.0, 1.0),   # flat
        ("U-shaped",       0.5, 0.5),   # bimodal-ish extremes
        ("right-skewed",   2.0, 8.0),   # mass near 0
        ("left-skewed",    8.0, 2.0),   # mass near 1
        ("moderate-skew",  2.0, 5.0),   # asymmetric centre
    ]
    if suite == "expanded":
        continuous_specs.extend(
            [
                ("extreme-right", 0.35, 6.0),
                ("extreme-left", 6.0, 0.35),
                ("near-boundaries", 0.3, 0.3),
                ("near-center", 6.0, 6.0),
            ]
        )

    for label, a, b in continuous_specs:
        a_, b_ = a, b
        scenarios.append(
            Scenario(
                label=f"{label} Beta({a_},{b_})",
                eval_type="continuous",
                generate=lambda rng, n, _a=a_, _b=b_: rng.beta(_a, _b, n),
                true_mean=a_ / (a_ + b_),
            )
        )

    # logit-normal is a principled model for bounded continuous scores
    # (common in LLM rubric outputs); always included in standard suite.
    def _gen_logit_normal(rng: np.random.Generator, n: int) -> np.ndarray:
        logits = rng.normal(-0.35, 1.35, size=n)
        return 1.0 / (1.0 + np.exp(-logits))

    scenarios.append(
        Scenario(
            label="logit-normal(mu=-0.35,s=1.35)",
            eval_type="continuous",
            generate=_gen_logit_normal,
            true_mean=_estimate_true_mean_mc(_gen_logit_normal),
        )
    )

    # Zero-inflated and one-inflated — point-mass spike at the boundary mixed
    # with a Beta component.  Models extreme low/high LLM performance: a model
    # that almost always fails produces a spike at 0; a near-ceiling model
    # produces a spike at 1.  True means are computed analytically.
    def _gen_zero_inflated(rng: np.random.Generator, n: int) -> np.ndarray:
        spike = rng.random(n) < 0.70
        return np.where(spike, 0.0, rng.beta(2.0, 4.0, n))

    scenarios.append(
        Scenario(
            label="zero-inflated(π=0.70,Beta(2,4))",
            eval_type="continuous",
            generate=_gen_zero_inflated,
            # E[X] = 0.70*0 + 0.30*Beta_mean = 0.30 * 2/(2+4)
            true_mean=0.30 * (2.0 / 6.0),
        )
    )

    def _gen_one_inflated(rng: np.random.Generator, n: int) -> np.ndarray:
        spike = rng.random(n) < 0.70
        return np.where(spike, 1.0, rng.beta(4.0, 2.0, n))

    scenarios.append(
        Scenario(
            label="one-inflated(π=0.70,Beta(4,2))",
            eval_type="continuous",
            generate=_gen_one_inflated,
            # E[X] = 0.70*1 + 0.30*Beta_mean = 0.70 + 0.30 * 4/(4+2)
            true_mean=0.70 + 0.30 * (4.0 / 6.0),
        )
    )

    if suite == "expanded":
        def _gen_mix_continuous(rng: np.random.Generator, n: int) -> np.ndarray:
            selector = rng.binomial(1, 0.55, size=n).astype(bool)
            vals = np.empty(n, dtype=float)
            vals[selector] = rng.beta(0.5, 4.0, size=int(np.sum(selector)))
            vals[~selector] = rng.beta(5.5, 1.2, size=int(np.sum(~selector)))
            return vals

        scenarios.append(
            Scenario(
                label="mixture Beta(0.5,4.0)/(5.5,1.2)",
                eval_type="continuous",
                generate=_gen_mix_continuous,
                true_mean=_estimate_true_mean_mc(_gen_mix_continuous),
            )
        )

    # ------------------------------------------------------------------
    # Likert 1–5: latent-normal model (consistent with build_pair_scenarios)
    # ------------------------------------------------------------------
    # Each generator draws from N(mu_lat, sigma_lat) on a latent scale, then
    # rounds to integers in {1,...,5} via rint+clip.  Using the same generative
    # family as build_pair_scenarios ensures that single-sample and pairwise
    # results are directly comparable.  True means are estimated via MC
    # (n_mc=500,000; SE < 0.001).

    def _make_likert_normal(mu: float, sigma: float) -> Callable[[np.random.Generator, int], np.ndarray]:
        def _gen(rng: np.random.Generator, n: int, _m: float = mu, _s: float = sigma) -> np.ndarray:
            return np.clip(np.rint(rng.normal(_m, _s, n)), 1.0, 5.0)
        return _gen

    def _make_likert_bimodal(mu1: float, mu2: float, sigma: float) -> Callable[[np.random.Generator, int], np.ndarray]:
        def _gen(rng: np.random.Generator, n: int, _m1: float = mu1, _m2: float = mu2, _s: float = sigma) -> np.ndarray:
            sel = rng.random(n) < 0.5
            latents = np.where(sel, rng.normal(_m1, _s, n), rng.normal(_m2, _s, n))
            return np.clip(np.rint(latents), 1.0, 5.0)
        return _gen

    # Standard suite: five shapes covering the main families
    _likert_standard = [
        ("uniform",       _make_likert_normal(3.0, 2.0)),    # high variance → flat
        ("skewed-low",    _make_likert_normal(2.0, 1.1)),    # mass at 1–2
        ("skewed-high",   _make_likert_normal(4.0, 1.1)),    # mass at 4–5
        ("bimodal",       _make_likert_bimodal(1.5, 4.5, 0.65)),  # peaks at extremes
        ("center-peaked", _make_likert_normal(3.0, 0.55)),   # sharp peak at 3
    ]
    for label, gen in _likert_standard:
        scenarios.append(
            Scenario(
                label=label,
                eval_type="likert",
                generate=gen,
                true_mean=_estimate_true_mean_mc(gen),
            )
        )

    if suite == "expanded":
        _likert_expanded = [
            ("near-floor",   _make_likert_normal(1.5, 0.65)),       # mostly 1s
            ("near-ceiling", _make_likert_normal(4.5, 0.65)),       # mostly 5s
            ("polarized",    _make_likert_bimodal(1.3, 4.7, 0.50)), # extreme bimodal
            ("flat-middle",  _make_likert_normal(3.0, 1.4)),        # moderate spread
        ]
        for label, gen in _likert_expanded:
            scenarios.append(
                Scenario(
                    label=label,
                    eval_type="likert",
                    generate=gen,
                    true_mean=_estimate_true_mean_mc(gen),
                )
            )

    # ------------------------------------------------------------------
    # Grades 0–100: truncated normals of varying centre and spread
    # ------------------------------------------------------------------
    grade_specs = [
        ("symmetric",     50, 20),   # centred, moderate spread
        ("high-scoring",  75, 15),   # near ceiling
        ("low-scoring",   35, 20),   # near floor
        ("ceiling-heavy", 88, 10),   # mass near 100 — heavy clipping
        ("floor-heavy",   12, 10),   # mass near 0   — heavy clipping
    ]
    if suite == "expanded":
        grade_specs.extend(
            [
                ("very-high", 92, 7),
                ("very-low", 8, 7),
                ("high-variance", 50, 34),
            ]
        )

    for label, mu, sigma in grade_specs:
        mu_, sigma_ = mu, sigma
        true_mean_ = _true_mean_clipped_normal(mu_, sigma_)
        scenarios.append(
            Scenario(
                label=f"{label} N({mu_},{sigma_})",
                eval_type="grades",
                generate=lambda rng, n, _m=mu_, _s=sigma_: np.clip(
                    rng.normal(_m, _s, n), 0.0, 100.0
                ),
                true_mean=true_mean_,
            )
        )

    # Mixture of 3 normal components — captures bimodal/trimodal grade
    # distributions common in LLM evals (fail / partial / full credit).
    def _gen_grade_mixture(rng: np.random.Generator, n: int) -> np.ndarray:
        flags = rng.choice(3, size=n, p=[0.20, 0.50, 0.30])
        vals = np.empty(n, dtype=float)
        for bucket, mu, sigma in [(0, 22.0, 11.0), (1, 58.0, 14.0), (2, 88.0, 8.0)]:
            mask = flags == bucket
            vals[mask] = rng.normal(mu, sigma, size=int(np.sum(mask)))
        return np.clip(vals, 0.0, 100.0)

    scenarios.append(
        Scenario(
            label="mixture-truncnorm(3 components)",
            eval_type="grades",
            generate=_gen_grade_mixture,
            true_mean=_estimate_true_mean_mc(_gen_grade_mixture),
        )
    )

    # Heavy-tailed t(df=3) — models grade distributions with occasional
    # outlier runs (model crashes, OOD inputs, etc.).
    def _gen_grade_heavy_tail(rng: np.random.Generator, n: int) -> np.ndarray:
        vals = 52.0 + 16.0 * rng.standard_t(df=3.0, size=n)
        return np.clip(vals, 0.0, 100.0)

    scenarios.append(
        Scenario(
            label="heavy-tail t(df=3)",
            eval_type="grades",
            generate=_gen_grade_heavy_tail,
            true_mean=_estimate_true_mean_mc(_gen_grade_heavy_tail),
        )
    )

    if suite == "expanded":
        # Zero-spiked and hundred-spiked grades — point-mass spike at floor/ceiling
        # mixed with a truncated-normal body.  Models complete failure (spike at 0)
        # and near-perfect performance (spike at 100) common in LLM coding/math evals.
        def _gen_grade_zero_spiked(rng: np.random.Generator, n: int) -> np.ndarray:
            spike = rng.random(n) < 0.40
            body = np.clip(rng.normal(45.0, 20.0, n), 0.0, 100.0)
            return np.where(spike, 0.0, body)

        scenarios.append(
            Scenario(
                label="zero-spiked(π=0.40,N(45,20))",
                eval_type="grades",
                generate=_gen_grade_zero_spiked,
                true_mean=_estimate_true_mean_mc(_gen_grade_zero_spiked),
            )
        )

        def _gen_grade_hundred_spiked(rng: np.random.Generator, n: int) -> np.ndarray:
            spike = rng.random(n) < 0.40
            body = np.clip(rng.normal(65.0, 18.0, n), 0.0, 100.0)
            return np.where(spike, 100.0, body)

        scenarios.append(
            Scenario(
                label="hundred-spiked(π=0.40,N(65,18))",
                eval_type="grades",
                generate=_gen_grade_hundred_spiked,
                true_mean=_estimate_true_mean_mc(_gen_grade_hundred_spiked),
            )
        )

    return scenarios


def _binary_conc_from_icc(icc: float) -> float:
    """Beta concentration for Bernoulli input probabilities giving target ICC.

    For a Bernoulli-Beta hierarchical model where per-input success probabilities
    are drawn from Beta(conc*p0, conc*(1-p0)), the intraclass correlation of
    observed scores is ICC = 1/(conc + 1).  Inverted: conc = 1/ICC - 1.
    """
    return max(1.0 / max(icc, 1e-9) - 1.0, 0.1)


def _beta_var(a: float, b: float) -> float:
    """Variance of Beta(a, b)."""
    return a * b / ((a + b) ** 2 * (a + b + 1))


def _estimate_true_pair_diff(
    generate_pair: Callable[[np.random.Generator, int, int], tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = 0,
    n_mc: int = 300_000,
) -> float:
    """Estimate E[cell_mean(A) - cell_mean(B)] via a large synthetic sample.

    Uses n_mc=300,000 items (single run). For all pairwise scenarios used here
    the resulting MC standard error on the true diff is < 0.002, negligible
    relative to the CI widths under study.
    """
    rng = np.random.default_rng(seed)
    a, b = generate_pair(rng, n_mc, 1)
    return float(np.mean(a[:, 0] - b[:, 0]))


def build_pair_scenarios(
    suite: str = "standard",
    icc_values: list[float] | tuple[float, ...] = (0.10, 0.25, 0.40),
    cohens_d_values: list[float] | tuple[float, ...] = (0.3,),
    include_null: bool = False,
) -> list[PairScenario]:
    """Return paired-difference scenarios parameterised by ICC and Cohen's d.

    Data-generating processes are reparameterised so that the intraclass
    correlation (ICC = between-input variance / total variance) and the
    standardised effect size (Cohen's d = delta / total_std) are explicit
    inputs.  This makes the sweep principled and reviewer-defensible.

    ICC definitions per eval type
    ──────────────────────────────
    Binary   : Bernoulli-Beta hierarchical model.  Per-input success probs
               p_i ~ Beta(conc·p0, conc·(1-p0)), giving ICC = 1/(conc+1).
    Continuous: Beta base distribution with ICC-derived Gaussian noise.
               ICC = Var(base) / (Var(base) + 2·noise_std²).
    Likert   : Latent-normal model.  ICC = base_std² / total_var,
               total_std fixed at ``_LIKERT_TOTAL_STD`` on the latent scale.
    Grades   : Same latent-normal structure; total_std = ``_GRADES_TOTAL_STD``.

    Parameters
    ----------
    suite : str
        ``'standard'`` or ``'expanded'``.
    icc_values : sequence of float
        ICC values to sweep.  Each value generates a separate batch of
        scenarios.  Default: (0.10, 0.25, 0.40).
    cohens_d_values : sequence of float
        Non-null standardised effect sizes to evaluate.  Default: (0.3,).
        A null (d=0) variant is automatically prepended when
        ``include_null=True``.
    include_null : bool
        If True, prepend d=0 scenarios for every (eval_type, shape, icc)
        combination.  These are flagged ``is_null=True`` in ``PairScenario``
        and are used to measure Type I error rates.
    """
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")

    scenarios: list[PairScenario] = []

    icc_list = list(icc_values)
    d_list = list(cohens_d_values)
    if include_null:
        d_list = [0.0] + [d for d in d_list if d > 0.0]

    # ── Latent-scale standard deviations for Likert / Grades ───────────────
    # These fix the total marginal std for each score type so that Cohen's d
    # has a concrete, consistent meaning across ICC levels.
    _LIKERT_TOTAL_STD = 1.2   # latent scale, maps to {1,...,5} after rounding
    _GRADES_TOTAL_STD = 20.0  # [0, 100] scale

    # ── Binary ─────────────────────────────────────────────────────────────
    # ICC = 1/(conc+1)  ←→  conc = 1/ICC - 1
    # total_std ≈ sqrt(p0·(1-p0))  (marginal Bernoulli std at base rate p0)
    # delta = d · total_std  (probability-scale uplift for template B)
    binary_shapes: list[tuple[str, float]] = [
        ("binary-balanced", 0.5),
        ("binary-high", 0.8),
        ("binary-low", 0.2),
        ("binary-near-ceil", 0.92),
    ]
    if suite == "expanded":
        binary_shapes += [
            ("binary-rare", 0.05),
            ("binary-near-ceil-hi", 0.95),
        ]

    for icc in icc_list:
        conc = _binary_conc_from_icc(icc)
        for shape_label, base_p in binary_shapes:
            total_std = float(np.sqrt(base_p * (1.0 - base_p)))
            for d in d_list:
                delta = d * total_std
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|icc={icc:.2f}|{effect_tag}"
                bp_, delta_, conc_ = base_p, delta, conc

                def _gen_binary(
                    rng: np.random.Generator, n: int, runs: int,
                    _bp: float = bp_, _d: float = delta_, _c: float = conc_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    p_a = rng.beta(_bp * _c, (1.0 - _bp) * _c, size=(n, 1))
                    p_b = np.clip(p_a + _d, 0.0, 1.0)
                    a = rng.binomial(1, p_a, size=(n, runs)).astype(float)
                    b = rng.binomial(1, p_b, size=(n, runs)).astype(float)
                    return a, b

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_binary)
                scenarios.append(PairScenario(
                    label=label, eval_type="binary",
                    generate_pair=_gen_binary, true_diff=true_diff,
                    icc=icc, cohens_d=d, is_null=is_null,
                ))

    # Explicit stress-test binary regimes with highly one-sided discordance.
    # These target small-n behavior where one discordant cell is near zero
    # (for example p10 << p01), which can be challenging for some CIs.
    if suite in ("expanded", "extreme"):
        # Joint probabilities over paired outcomes:
        #   p11 = P(A=1, B=1), p10 = P(A=1, B=0),
        #   p01 = P(A=0, B=1), p00 = 1 - p11 - p10 - p01.
        asym_binary_specs: list[tuple[str, float, float, float]] = [
            # Matches the empirically challenging geometry seen in real data.
            ("binary-onesided-neg-extreme", 0.001, 0.384, 0.000),
            ("binary-onesided-pos-extreme", 0.384, 0.001, 0.000),
            # Less extreme but still strongly asymmetric discordance.
            ("binary-onesided-neg-strong", 0.020, 0.300, 0.050),
            ("binary-onesided-pos-strong", 0.300, 0.020, 0.050),
            # Very severe asymmetry with large effect and dense discordance.
            ("binary-onesided-neg-ultra", 0.000, 0.520, 0.000),
            ("binary-onesided-pos-ultra", 0.520, 0.000, 0.000),
            # Sparse-discordance variants (few discordant pairs in small n).
            ("binary-onesided-neg-sparse", 0.001, 0.090, 0.030),
            ("binary-onesided-pos-sparse", 0.090, 0.001, 0.030),
            # Near-boundary marginals with one-sided discordance.
            ("binary-onesided-neg-near-ceil", 0.000, 0.080, 0.900),
            ("binary-onesided-pos-near-floor", 0.080, 0.000, 0.020),
            # Moderate but asymmetric controls (helps map transition region).
            ("binary-onesided-neg-moderate", 0.050, 0.220, 0.150),
            ("binary-onesided-pos-moderate", 0.220, 0.050, 0.150),
        ]

        for shape_label, p10, p01, p11 in asym_binary_specs:
            p00 = 1.0 - (p11 + p10 + p01)
            if p00 <= 0.0:
                raise ValueError(
                    f"Invalid asymmetric binary scenario {shape_label}: "
                    f"probabilities sum to >= 1.0"
                )

            probs = np.array([p11, p10, p01, p00], dtype=float)
            true_diff = float(p10 - p01)
            label = (
                f"{shape_label}|p10={p10:.3f}|p01={p01:.3f}|"
                f"p11={p11:.3f}|p00={p00:.3f}"
            )

            def _gen_binary_asym(
                rng: np.random.Generator,
                n: int,
                runs: int,
                _probs: np.ndarray = probs,
            ) -> tuple[np.ndarray, np.ndarray]:
                # State coding: 0->11, 1->10, 2->01, 3->00.
                z = rng.choice(4, size=(n, runs), p=_probs)
                a = np.isin(z, (0, 1)).astype(float)
                b = np.isin(z, (0, 2)).astype(float)
                return a, b

            scenarios.append(PairScenario(
                label=label,
                eval_type="binary",
                generate_pair=_gen_binary_asym,
                true_diff=true_diff,
                icc=0.0,
                cohens_d=0.0,
                is_null=False,
            ))

    # ── Continuous [0, 1] ──────────────────────────────────────────────────
    # Base distribution shape is Beta(a, b).  Given ICC and Var(base),
    # per-component Gaussian noise std is derived as:
    #   ICC = Var(base) / (Var(base) + 2·noise_std²)
    #   →  noise_std = sqrt(Var(base) · (1/ICC - 1) / 2)
    # Shared and individual noises both use noise_std (equal split).
    # total_std ≈ sqrt(Var(base)/ICC)  (pre-clipping approximation).
    # delta = d · total_std.
    continuous_shapes: list[tuple[str, float, float]] = [
        ("cont-uniform", 1.0, 1.0),
        ("cont-right-skew", 2.0, 8.0),
        ("cont-left-skew", 8.0, 2.0),
    ]
    if suite == "expanded":
        continuous_shapes += [
            ("cont-moderate-skew", 2.0, 5.0),
            ("cont-boundary", 0.6, 0.6),
        ]

    for icc in icc_list:
        for shape_label, a_beta, b_beta in continuous_shapes:
            var_base = _beta_var(a_beta, b_beta)
            noise_std = float(np.sqrt(max(var_base * (1.0 / max(icc, 1e-9) - 1.0) / 2.0, 0.0)))
            total_std = float(np.sqrt(var_base / max(icc, 1e-9)))
            for d in d_list:
                delta = d * total_std
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|icc={icc:.2f}|{effect_tag}"
                a_, b_, ns_, delta_ = a_beta, b_beta, noise_std, delta

                def _gen_continuous(
                    rng: np.random.Generator, n: int, runs: int,
                    _a: float = a_, _b: float = b_,
                    _ns: float = ns_, _d: float = delta_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    base = rng.beta(_a, _b, size=(n, 1))
                    shared = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_a = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_b = rng.normal(0.0, _ns, size=(n, runs))
                    a_vals = np.clip(base + shared + indiv_a, 0.0, 1.0)
                    b_vals = np.clip(base + _d + shared + indiv_b, 0.0, 1.0)
                    return a_vals, b_vals

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_continuous)
                scenarios.append(PairScenario(
                    label=label, eval_type="continuous",
                    generate_pair=_gen_continuous, true_diff=true_diff,
                    icc=icc, cohens_d=d, is_null=is_null,
                ))

    # ── Likert 1–5 ─────────────────────────────────────────────────────────
    # Latent-normal model rounded to {1,...,5}.
    # ICC = base_std² / total_var  with total_var = _LIKERT_TOTAL_STD².
    # base_std = sqrt(ICC) · _LIKERT_TOTAL_STD
    # noise_std = sqrt((1-ICC)/2) · _LIKERT_TOTAL_STD  (shared + indiv equal)
    # delta_latent = d · _LIKERT_TOTAL_STD  (on the latent scale before rounding)
    likert_shapes: list[tuple[str, float]] = [
        ("likert-mid", 3.0),
        ("likert-low", 2.2),
        ("likert-high", 3.8),
    ]
    if suite == "expanded":
        likert_shapes += [
            ("likert-polarized", 3.0),
            ("likert-floor", 1.8),
        ]

    for icc in icc_list:
        base_std_l = float(np.sqrt(icc)) * _LIKERT_TOTAL_STD
        noise_std_l = float(np.sqrt(max((1.0 - icc) / 2.0, 0.0))) * _LIKERT_TOTAL_STD
        for shape_label, mu_lat in likert_shapes:
            for d in d_list:
                delta = d * _LIKERT_TOTAL_STD
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|icc={icc:.2f}|{effect_tag}"
                m_, bs_, ns_, delta_ = mu_lat, base_std_l, noise_std_l, delta

                def _gen_likert(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _bs: float = bs_,
                    _ns: float = ns_, _d: float = delta_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    base = rng.normal(_m, _bs, size=(n, 1))
                    shared = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_a = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_b = rng.normal(0.0, _ns, size=(n, runs))
                    a_vals = np.rint(np.clip(base + shared + indiv_a, 1.0, 5.0))
                    b_vals = np.rint(np.clip(base + _d + shared + indiv_b, 1.0, 5.0))
                    return a_vals, b_vals

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_likert)
                scenarios.append(PairScenario(
                    label=label, eval_type="likert",
                    generate_pair=_gen_likert, true_diff=true_diff,
                    icc=icc, cohens_d=d, is_null=is_null,
                ))

    # ── Grades 0–100 ───────────────────────────────────────────────────────
    # Same latent-normal structure as Likert; total_std = _GRADES_TOTAL_STD.
    # ICC = base_std² / total_var  with total_var = _GRADES_TOTAL_STD².
    # delta_grades = d · _GRADES_TOTAL_STD  (on the [0,100] scale).
    grades_shapes: list[tuple[str, float]] = [
        ("grades-mid", 55.0),
        ("grades-low", 35.0),
        ("grades-high", 78.0),
    ]
    if suite == "expanded":
        grades_shapes += [
            ("grades-ceiling", 86.0),
            ("grades-floor", 20.0),
        ]

    for icc in icc_list:
        base_std_g = float(np.sqrt(icc)) * _GRADES_TOTAL_STD
        noise_std_g = float(np.sqrt(max((1.0 - icc) / 2.0, 0.0))) * _GRADES_TOTAL_STD
        for shape_label, mu_g in grades_shapes:
            for d in d_list:
                delta = d * _GRADES_TOTAL_STD
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|icc={icc:.2f}|{effect_tag}"
                m_, bs_, ns_, delta_ = mu_g, base_std_g, noise_std_g, delta

                def _gen_grades(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _bs: float = bs_,
                    _ns: float = ns_, _d: float = delta_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    base = rng.normal(_m, _bs, size=(n, 1))
                    shared = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_a = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_b = rng.normal(0.0, _ns, size=(n, runs))
                    a_vals = np.clip(base + shared + indiv_a, 0.0, 100.0)
                    b_vals = np.clip(base + _d + shared + indiv_b, 0.0, 100.0)
                    return a_vals, b_vals

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_grades)
                scenarios.append(PairScenario(
                    label=label, eval_type="grades",
                    generate_pair=_gen_grades, true_diff=true_diff,
                    icc=icc, cohens_d=d, is_null=is_null,
                ))

    return scenarios


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    eval_type: str
    scenario: str
    n: int
    method: str
    n_reps: int
    covered: int        # number of reps where CI contained the true estimand
    total_width: float  # sum of CI widths across reps
    total_time: float = 0.0     # sum of per-rep CI wall-clock times (seconds)
    total_time_sq: float = 0.0  # sum of squared per-rep times (for SE via Var=E[x²]−E[x]²)
    is_null: bool = False       # True when the scenario has delta=0 (Type I error measurement)


def _stat(values: np.ndarray, statistic: str = "mean") -> float:
    return float(np.median(values)) if statistic == "median" else float(np.mean(values))


def _wilson_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    low = max(0.0, float(center - radius))
    high = min(1.0, float(center + radius))
    return low, high


def _newcombe_paired_score_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """
    Newcombe score CI for paired binary difference p(A=1) - p(B=1).

    Uses the discordant-pairs formulation:
      d = (n10 - n01) / n = (m / n) * (2*theta - 1),
    where m = n10 + n01 and theta = n10 / m.
    A Wilson score interval is computed for theta and then transformed back
    to the difference scale.
    """
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Newcombe paired score CI expects two 1D arrays with equal shape.")

    n = int(a.shape[0])
    if n <= 0:
        return (0.0, 0.0)

    a_bin = (a >= 0.5).astype(int)
    b_bin = (b >= 0.5).astype(int)

    n10 = int(np.sum((a_bin == 1) & (b_bin == 0)))
    n01 = int(np.sum((a_bin == 0) & (b_bin == 1)))
    m = n10 + n01

    if m == 0:
        return (0.0, 0.0)

    theta_low, theta_high = _wilson_ci(successes=n10, n=m, alpha=alpha)
    scale = m / n
    low = scale * (2.0 * theta_low - 1.0)
    high = scale * (2.0 * theta_high - 1.0)
    return float(low), float(high)


def _bayes_indep_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Beta(1,1) posterior credible interval for a Bernoulli proportion."""
    n = int(values.shape[0])
    s = int(np.sum(values >= 0.5))
    lo, hi = stats.beta(s + 1, n - s + 1).interval(1.0 - alpha)
    return float(lo), float(hi)


def _bayes_indep_comp_ci(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Independent Beta-posteriors CI for paired binary difference p(A=1)-p(B=1)."""
    a_bin = (a >= 0.5).astype(float)
    b_bin = (b >= 0.5).astype(float)

    post_a = rng.beta(float(np.sum(a_bin)) + 1.0, float(a_bin.shape[0] - np.sum(a_bin)) + 1.0, size=num_samples)
    post_b = rng.beta(float(np.sum(b_bin)) + 1.0, float(b_bin.shape[0] - np.sum(b_bin)) + 1.0, size=num_samples)
    diff = post_a - post_b
    return (
        float(np.percentile(diff, 100.0 * alpha / 2.0)),
        float(np.percentile(diff, 100.0 * (1.0 - alpha / 2.0))),
    )


def _bayes_paired_comp_ci(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Paired Bayesian CI for p(A=1)-p(B=1) using the bivariate-normal latent model
    from bayes_evals.py.
    """
    a_bin = (a >= 0.5).astype(float)
    b_bin = (b >= 0.5).astype(float)

    s = float(np.sum(a_bin * b_bin))
    t = float(np.sum(a_bin * (1.0 - b_bin)))
    u = float(np.sum((1.0 - a_bin) * b_bin))
    v = float(np.sum((1.0 - a_bin) * (1.0 - b_bin)))

    theta_as = rng.beta(1.0, 1.0, size=num_samples)
    theta_bs = rng.beta(1.0, 1.0, size=num_samples)
    rhos = np.clip(2.0 * rng.beta(4.0, 2.0, size=num_samples) - 1.0, -1 + 1e-20, 1 - 1e-20)
    diff = theta_as - theta_bs

    mu_a = stats.norm.ppf(theta_as)
    mu_b = stats.norm.ppf(theta_bs)

    th_v = binorm_cdf(0, 0, mu_a, mu_b, 1, 1, rhos)
    th_s = theta_as + theta_bs + th_v - 1.0
    th_t = 1.0 - theta_bs - th_v
    th_u = 1.0 - theta_as - th_v

    with np.errstate(divide="ignore", invalid="ignore"):
        log_w = s * np.log(th_s) + t * np.log(th_t) + u * np.log(th_u) + v * np.log(th_v)

    log_w -= np.nanmax(log_w)
    w = np.exp(log_w)
    w[np.isnan(w)] = 0.0
    w_sum = float(np.sum(w))

    if w_sum <= 0.0:
        d_hat = float(np.mean(a_bin) - np.mean(b_bin))
        return d_hat, d_hat

    w /= w_sum
    diff_post = diff[rng.choice(num_samples, size=num_samples, replace=True, p=w)]

    return (
        float(np.percentile(diff_post, 100.0 * alpha / 2.0)),
        float(np.percentile(diff_post, 100.0 * (1.0 - alpha / 2.0))),
    )


def _pairwise_ci(
    a: np.ndarray,
    b: np.ndarray,
    method: str,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
    *,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Compute CI for paired mean/median difference A-B using evalstats logic."""
    n_inputs, runs = a.shape
    resolved_method = resolve_resampling_method(method, n_inputs)

    if runs >= 3:
        cell_diffs = a.mean(axis=1) - b.mean(axis=1)
        observed = _stat(cell_diffs, statistic=statistic)

        if resolved_method == "bayes_bootstrap":
            boot_stats = bayes_bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
            return (
                float(np.percentile(boot_stats, 100 * alpha / 2)),
                float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
            )
        if resolved_method == "smooth_bootstrap":
            boot_stats = smooth_bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
            return (
                float(np.percentile(boot_stats, 100 * alpha / 2)),
                float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
            )

        if resolved_method == "bootstrap_t":
            return bootstrap_t_ci_1d(cell_diffs, observed, n_bootstrap, alpha, rng, statistic=statistic)
        boot_stats = bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
        if resolved_method == "bca":
            return bca_interval_1d(cell_diffs, observed, boot_stats, alpha, statistic=statistic)
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )

    diffs = a.mean(axis=1) - b.mean(axis=1)
    observed = _stat(diffs, statistic=statistic)

    if resolved_method == "bayes_bootstrap":
        boot_stats = bayes_bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )
    if resolved_method == "smooth_bootstrap":
        boot_stats = smooth_bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        )
    if resolved_method == "bootstrap_t":
        return bootstrap_t_ci_1d(diffs, observed, n_bootstrap, alpha, rng, statistic=statistic)

    boot_stats = bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
    if resolved_method == "bca":
        return bca_interval_1d(diffs, observed, boot_stats, alpha, statistic=statistic)
    return (
        float(np.percentile(boot_stats, 100 * alpha / 2)),
        float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# Per-cell worker functions
# Scenarios contain lambda/closure objects that are not picklable, so we store
# them in a module-level variable that fork-inherited workers can access by
# index.  Only plain ints/floats/seeds travel through the task queue.
# ---------------------------------------------------------------------------

_WORKER_SCENARIOS: list = []  # set by run_simulation / run_pairwise_simulation before forking


def _run_mean_cell(args: tuple) -> list[SimResult]:
    """Run all reps for one (scenario, n) cell — mean estimand."""
    sc_idx, n, n_reps, n_bootstrap, bayes_n, alpha, seed = args
    scenario = _WORKER_SCENARIOS[sc_idx]
    rng = np.random.default_rng(seed)

    active_methods = METHODS + [T_INTERVAL_METHOD]
    if scenario.eval_type == "binary":
        active_methods += [WILSON_METHOD, JEFFREYS_METHOD, WALD_METHOD, CP_METHOD, BAYES_SINGLE_METHOD]
    elif scenario.eval_type == "continuous":
        active_methods += CONTINUOUS_EXTRA_METHODS

    covered: dict[str, int] = {m: 0 for m in active_methods}
    total_w: dict[str, float] = {m: 0.0 for m in active_methods}
    total_t: dict[str, float] = {m: 0.0 for m in active_methods}
    total_t_sq: dict[str, float] = {m: 0.0 for m in active_methods}

    for _rep in range(n_reps):
        values = scenario.generate(rng, n)
        obs_mean = float(np.mean(values))

        for method in METHODS:
            _t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ci_low, ci_high = bootstrap_ci_1d(
                        values, obs_mean, method=method,
                        n_bootstrap=n_bootstrap, alpha=alpha, rng=rng,
                    )
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[method] += _el
            total_t_sq[method] += _el * _el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[method] += 1
            total_w[method] += ci_high - ci_low

        _t0 = time.perf_counter()
        try:
            ci_low, ci_high = t_interval_ci_1d(values, alpha)
        except Exception:
            ci_low = ci_high = obs_mean
        _el = time.perf_counter() - _t0
        total_t[T_INTERVAL_METHOD] += _el
        total_t_sq[T_INTERVAL_METHOD] += _el * _el
        if ci_low <= scenario.true_mean <= ci_high:
            covered[T_INTERVAL_METHOD] += 1
        total_w[T_INTERVAL_METHOD] += ci_high - ci_low

        if scenario.eval_type == "continuous":
            _continuous_method_fns = {
                BETA_METHOD: beta_ci_1d,
                LOGIT_T_METHOD: logit_t_ci_1d,
                NIG_METHOD: nig_ci_1d,
                EL_METHOD: el_ci_1d,
            }
            for _method, _fn in _continuous_method_fns.items():
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = _fn(values, alpha)
                except Exception:
                    ci_low = ci_high = obs_mean
                _el = time.perf_counter() - _t0
                total_t[_method] += _el
                total_t_sq[_method] += _el * _el
                if ci_low <= scenario.true_mean <= ci_high:
                    covered[_method] += 1
                total_w[_method] += ci_high - ci_low

        if scenario.eval_type == "binary":
            successes = int(np.sum(values))

            _t0 = time.perf_counter()
            ci_low, ci_high = _wilson_ci(successes, n, alpha)
            _el = time.perf_counter() - _t0
            total_t[WILSON_METHOD] += _el
            total_t_sq[WILSON_METHOD] += _el * _el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[WILSON_METHOD] += 1
            total_w[WILSON_METHOD] += ci_high - ci_low

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = jeffreys_ci_1d(values, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[JEFFREYS_METHOD] += _el
            total_t_sq[JEFFREYS_METHOD] += _el * _el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[JEFFREYS_METHOD] += 1
            total_w[JEFFREYS_METHOD] += ci_high - ci_low

            _t0 = time.perf_counter()
            ci_low, ci_high = wald_ci_1d(values, alpha)
            _el = time.perf_counter() - _t0
            total_t[WALD_METHOD] += _el
            total_t_sq[WALD_METHOD] += _el * _el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[WALD_METHOD] += 1
            total_w[WALD_METHOD] += ci_high - ci_low

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = clopper_pearson_ci_1d(values, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[CP_METHOD] += _el
            total_t_sq[CP_METHOD] += _el * _el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[CP_METHOD] += 1
            total_w[CP_METHOD] += ci_high - ci_low

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_indep_ci(values, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[BAYES_SINGLE_METHOD] += _el
            total_t_sq[BAYES_SINGLE_METHOD] += _el * _el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[BAYES_SINGLE_METHOD] += 1
            total_w[BAYES_SINGLE_METHOD] += ci_high - ci_low

    return [
        SimResult(
            eval_type=scenario.eval_type,
            scenario=scenario.label,
            n=n,
            method=method,
            n_reps=n_reps,
            covered=covered[method],
            total_width=total_w[method],
            total_time=total_t[method],
            total_time_sq=total_t_sq[method],
        )
        for method in active_methods
    ]


def _run_pairwise_cell(args: tuple) -> list[SimResult]:
    """Run all reps for one (scenario, n) cell — pairwise estimand."""
    sc_idx, n, n_reps, n_bootstrap, bayes_n, alpha, runs, statistic, seed = args
    scenario = _WORKER_SCENARIOS[sc_idx]
    rng = np.random.default_rng(seed)

    add_newcombe = scenario.eval_type == "binary" and statistic == "mean"
    add_tango = scenario.eval_type == "binary" and statistic == "mean"
    add_bayes_binary = scenario.eval_type == "binary" and statistic == "mean"
    add_pairwise_extras = statistic == "mean" and scenario.eval_type != "binary"

    active_methods = METHODS.copy()
    if add_pairwise_extras:
        active_methods.extend(PAIRWISE_EXTRA_METHODS)
    if add_newcombe:
        active_methods.append(NEWCOMBE_METHOD)
    if add_tango:
        active_methods.append(TANGO_METHOD)
    if add_bayes_binary:
        active_methods.extend([BAYES_PAIR_INDEP_METHOD, BAYES_PAIR_PAIRED_METHOD])

    covered: dict[str, int] = {m: 0 for m in active_methods}
    total_w: dict[str, float] = {m: 0.0 for m in active_methods}
    total_t: dict[str, float] = {m: 0.0 for m in active_methods}
    total_t_sq: dict[str, float] = {m: 0.0 for m in active_methods}

    for _rep in range(n_reps):
        a, b = scenario.generate_pair(rng, n, runs)

        for method in METHODS:
            _t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ci_low, ci_high = _pairwise_ci(
                        a, b, method=method,
                        n_bootstrap=n_bootstrap, alpha=alpha, rng=rng, statistic=statistic,
                    )
            except Exception:
                obs = _stat(a.mean(axis=1) - b.mean(axis=1), statistic=statistic)
                ci_low = ci_high = obs
            _el = time.perf_counter() - _t0
            total_t[method] += _el
            total_t_sq[method] += _el * _el
            if ci_low <= scenario.true_diff <= ci_high:
                covered[method] += 1
            total_w[method] += ci_high - ci_low

        if add_pairwise_extras:
            pair_diffs = a.mean(axis=1) - b.mean(axis=1)
            obs = float(np.mean(pair_diffs))
            for method in PAIRWISE_EXTRA_METHODS:
                _t0 = time.perf_counter()
                try:
                    if method == T_INTERVAL_METHOD:
                        ci_low, ci_high = t_interval_ci_1d(pair_diffs, alpha)
                    elif method == NIG_METHOD:
                        ci_low, ci_high = nig_ci_1d(pair_diffs, alpha)
                    else:
                        ci_low, ci_high = el_ci_1d(pair_diffs, alpha)
                except Exception:
                    ci_low = ci_high = obs
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                if ci_low <= scenario.true_diff <= ci_high:
                    covered[method] += 1
                total_w[method] += ci_high - ci_low

        if add_newcombe:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _newcombe_paired_score_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                obs = float(np.mean(a[:, 0] - b[:, 0]))
                ci_low = ci_high = obs
            _el = time.perf_counter() - _t0
            total_t[NEWCOMBE_METHOD] += _el
            total_t_sq[NEWCOMBE_METHOD] += _el * _el
            if ci_low <= scenario.true_diff <= ci_high:
                covered[NEWCOMBE_METHOD] += 1
            total_w[NEWCOMBE_METHOD] += ci_high - ci_low

        if add_tango:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = tango_paired_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                obs = float(np.mean(a[:, 0] - b[:, 0]))
                ci_low = ci_high = obs
            _el = time.perf_counter() - _t0
            total_t[TANGO_METHOD] += _el
            total_t_sq[TANGO_METHOD] += _el * _el
            if ci_low <= scenario.true_diff <= ci_high:
                covered[TANGO_METHOD] += 1
            total_w[TANGO_METHOD] += ci_high - ci_low

        if add_bayes_binary:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_indep_comp_ci(a[:, 0], b[:, 0], alpha, bayes_n, rng)
            except Exception:
                obs = float(np.mean(a[:, 0] - b[:, 0]))
                ci_low = ci_high = obs
            _el = time.perf_counter() - _t0
            total_t[BAYES_PAIR_INDEP_METHOD] += _el
            total_t_sq[BAYES_PAIR_INDEP_METHOD] += _el * _el
            if ci_low <= scenario.true_diff <= ci_high:
                covered[BAYES_PAIR_INDEP_METHOD] += 1
            total_w[BAYES_PAIR_INDEP_METHOD] += ci_high - ci_low

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_paired_comp_ci(a[:, 0], b[:, 0], alpha, bayes_n, rng)
            except Exception:
                obs = float(np.mean(a[:, 0] - b[:, 0]))
                ci_low = ci_high = obs
            _el = time.perf_counter() - _t0
            total_t[BAYES_PAIR_PAIRED_METHOD] += _el
            total_t_sq[BAYES_PAIR_PAIRED_METHOD] += _el * _el
            if ci_low <= scenario.true_diff <= ci_high:
                covered[BAYES_PAIR_PAIRED_METHOD] += 1
            total_w[BAYES_PAIR_PAIRED_METHOD] += ci_high - ci_low

    return [
        SimResult(
            eval_type=scenario.eval_type,
            scenario=scenario.label,
            n=n,
            method=method,
            n_reps=n_reps,
            covered=covered[method],
            total_width=total_w[method],
            total_time=total_t[method],
            total_time_sq=total_t_sq[method],
            is_null=scenario.is_null,
        )
        for method in active_methods
    ]


def run_simulation(
    scenarios: list[Scenario],
    sample_sizes: list[int],
    n_reps: int,
    n_bootstrap: int,
    bayes_n: int,
    alpha: float,
    progress_mode: str = "bar",
    seed: int = 42,
    n_workers: int = 1,
) -> list[SimResult]:
    global _WORKER_SCENARIOS
    _WORKER_SCENARIOS = scenarios  # inherited by forked workers

    ss = np.random.SeedSequence(seed)
    idx_size_pairs = list(itertools.product(range(len(scenarios)), sample_sizes))
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(idx_size_pairs))]
    args_list = [
        (sc_idx, n, n_reps, n_bootstrap, bayes_n, alpha, child_seeds[i])
        for i, (sc_idx, n) in enumerate(idx_size_pairs)
    ]
    total_cells = len(args_list)
    reporter = _ProgressReporter(total_cells, mode=progress_mode, label="mean")
    results: list[SimResult] = []

    if n_workers == 1:
        for i, args in enumerate(args_list):
            results.extend(_run_mean_cell(args))
            sc = scenarios[args[0]]
            reporter.update(i + 1, detail=f"{sc.eval_type} {sc.label} n={args[1]}")
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_mean_cell, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1, detail=f"cells done: {i + 1}/{total_cells}")

    reporter.update(total_cells, detail="done")
    return results


def run_pairwise_simulation(
    scenarios: list[PairScenario],
    sample_sizes: list[int],
    n_reps: int,
    n_bootstrap: int,
    bayes_n: int,
    alpha: float,
    runs: int,
    statistic: str,
    progress_mode: str = "bar",
    seed: int = 42,
    n_workers: int = 1,
) -> list[SimResult]:
    global _WORKER_SCENARIOS
    _WORKER_SCENARIOS = scenarios  # inherited by forked workers

    if runs > 1 and statistic == "mean" and any(sc.eval_type == "binary" for sc in scenarios):
        print(
            "\nNote: binary pairwise-only methods "
            "(newcombe_score, bayes_indep_comp, bayes_paired_comp) "
            "use run index 0 when runs>1."
        )

    ss = np.random.SeedSequence(seed)
    idx_size_pairs = list(itertools.product(range(len(scenarios)), sample_sizes))
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(idx_size_pairs))]
    args_list = [
        (sc_idx, n, n_reps, n_bootstrap, bayes_n, alpha, runs, statistic, child_seeds[i])
        for i, (sc_idx, n) in enumerate(idx_size_pairs)
    ]
    total_cells = len(args_list)
    reporter = _ProgressReporter(total_cells, mode=progress_mode, label="pairwise")
    results: list[SimResult] = []

    if n_workers == 1:
        for i, args in enumerate(args_list):
            results.extend(_run_pairwise_cell(args))
            sc = scenarios[args[0]]
            reporter.update(i + 1, detail=f"{sc.eval_type} {sc.label} n={args[1]}")
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_pairwise_cell, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1, detail=f"cells done: {i + 1}/{total_cells}")

    reporter.update(total_cells, detail="done")
    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _cov_marker(cov: float, target: float, tol: float = 0.04) -> str:
    """Single-character flag indicating coverage quality."""
    if cov < target - tol:
        return "▼"   # systematically under-covered
    if cov > target + tol:
        return "▲"   # over-conservative
    return " "


def _mc_proportion_stats(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    """Return (p_hat, mcse, lo, hi) for a Monte Carlo proportion estimate."""
    if total <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    p_hat = successes / total
    mcse = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    lo = max(0.0, p_hat - z * mcse)
    hi = min(1.0, p_hat + z * mcse)
    return float(p_hat), mcse, float(lo), float(hi)


def _rule(width: int, char: str = "─") -> str:
    return char * width


def _time_stats(subset: list[SimResult]) -> tuple[float, float]:
    """Return (avg_ms, se_ms) for CI wall-clock time, pooled over all reps in subset."""
    total_reps = sum(r.n_reps for r in subset)
    if total_reps <= 0:
        return float("nan"), float("nan")
    sum_t = sum(r.total_time for r in subset)
    sum_t2 = sum(r.total_time_sq for r in subset)
    avg = sum_t / total_reps
    var = max(0.0, sum_t2 / total_reps - avg * avg)
    se = float(np.sqrt(var / total_reps))
    return avg * 1000.0, se * 1000.0


def _print_grid(
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    cells: dict[tuple[str, str], str],
    row_w: int = 20,
    col_w: int = 9,
) -> None:
    total_w = row_w + 2 + (col_w + 2) * len(col_labels)
    print(f"\n  {title}")
    print(f"  {_rule(total_w)}")
    header = f"  {'':<{row_w}}" + "".join(f"  {c:>{col_w}}" for c in col_labels)
    print(header)
    print(f"  {_rule(total_w)}")
    for row in row_labels:
        line = f"  {row:<{row_w}}"
        for col in col_labels:
            val = cells.get((row, col), "─" * col_w)
            line += f"  {val:>{col_w}}"
        print(line)
    print(f"  {_rule(total_w)}")


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------


def print_report(
    results: list[SimResult],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
    estimand_label: str,
) -> None:
    target = 1.0 - alpha
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    n_labels = [f"n={n}" for n in sample_sizes]
    present_methods = {r.method for r in results}
    method_labels = [m for m in REPORT_METHODS if m in present_methods]

    # ----------------------------------------------------------------
    # Aggregate: mean coverage and mean width across scenarios within
    # each (eval_type, method, n).  Also keep per-scenario for worst-case.
    # ----------------------------------------------------------------
    # agg[(eval_type, method, n)] accumulates per-scenario (cov, width) pairs
    agg: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    agg_counts: dict[tuple, tuple[int, int]] = defaultdict(lambda: (0, 0))  # covered, total
    per_sc: dict[tuple, tuple[float, float]] = {}  # (et, sc, method, n)

    for r in results:
        if r.is_null:
            continue  # null scenarios reported separately in Type I error section
        cov = r.covered / r.n_reps
        width = r.total_width / r.n_reps
        agg[(r.eval_type, r.method, r.n)].append((cov, width))
        c_prev, t_prev = agg_counts[(r.eval_type, r.method, r.n)]
        agg_counts[(r.eval_type, r.method, r.n)] = (c_prev + r.covered, t_prev + r.n_reps)
        per_sc[(r.eval_type, r.scenario, r.method, r.n)] = (cov, width)

    def mean_cov(et: str, m: str, n: int) -> float:
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[0] for v in vals])) if vals else float("nan")

    def mean_wid(et: str, m: str, n: int) -> float:
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[1] for v in vals])) if vals else float("nan")

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    sep = "=" * 72
    print(f"\n{sep}")
    print("  BOOTSTRAP CI COMPARISON  —  SIMULATION RESULTS")
    print(f"  Estimand: {estimand_label}")
    print(f"  Nominal coverage: {target:.0%}   |   reps/scenario: {n_reps}")
    print(f"  ▼ = under-covered (<{target - 0.04:.0%})   ▲ = over-conservative (>{target + 0.04:.0%})")
    print("  MC uncertainty reported as normal-approximation 95% bands on coverage.")
    print(sep)

    # ----------------------------------------------------------------
    # Per-eval-type tables
    # ----------------------------------------------------------------
    for et in eval_types_present:
        print(f"\n{'─'*72}")
        print(f"  {et.upper()}")
        print(f"{'─'*72}")

        # Coverage grid
        cov_cells: dict[tuple[str, str], str] = {}
        for m in method_labels:
            for i, n in enumerate(sample_sizes):
                cov = mean_cov(et, m, n)
                if np.isnan(cov):
                    cov_cells[(m, n_labels[i])] = "─"
                else:
                    marker = _cov_marker(cov, target)
                    cov_cells[(m, n_labels[i])] = f"{cov:.3f}{marker}"

        _print_grid(
            f"Coverage (target {target:.2f})",
            row_labels=method_labels,
            col_labels=n_labels,
            cells=cov_cells,
        )

        band_cells: dict[tuple[str, str], str] = {}
        for m in method_labels:
            for i, n in enumerate(sample_sizes):
                covered, total = agg_counts.get((et, m, n), (0, 0))
                if total <= 0:
                    band_cells[(m, n_labels[i])] = "─"
                    continue
                _, _, lo, hi = _mc_proportion_stats(covered, total)
                band_cells[(m, n_labels[i])] = f"{lo:.3f}-{hi:.3f}"

        _print_grid(
            "Coverage MC 95% Band",
            row_labels=method_labels,
            col_labels=n_labels,
            cells=band_cells,
            col_w=13,
        )

        # Width grid
        wid_cells: dict[tuple[str, str], str] = {}
        for m in method_labels:
            for i, n in enumerate(sample_sizes):
                w = mean_wid(et, m, n)
                wid_cells[(m, n_labels[i])] = "─" if np.isnan(w) else f"{w:.4f}"

        _print_grid(
            "Mean CI Width (lower = more precise)",
            row_labels=method_labels,
            col_labels=n_labels,
            cells=wid_cells,
        )

        # Cost grid
        time_cells: dict[tuple[str, str], str] = {}
        for m in method_labels:
            for i, n in enumerate(sample_sizes):
                subset = [r for r in results if r.eval_type == et and r.method == m and r.n == n]
                avg_ms, se_ms = _time_stats(subset)
                if np.isfinite(avg_ms):
                    time_cells[(m, n_labels[i])] = f"{avg_ms:.3f}±{se_ms:.3f}"
        _print_grid(
            "Mean CI Time (ms) ± SE",
            row_labels=method_labels,
            col_labels=n_labels,
            cells=time_cells,
            col_w=13,
        )

    # ----------------------------------------------------------------
    # Worst-coverage scenarios (averaged across methods)
    # ----------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("  WORST COVERAGE CASES  (averaged across included methods)")
    print(f"{'─'*72}")

    # Average coverage across methods for each (eval_type, scenario, n)
    sc_cov: dict[tuple, list[float]] = defaultdict(list)
    for (et, sc, m, n), (cov, _) in per_sc.items():
        sc_cov[(et, sc, n)].append(cov)

    worst = sorted(
        [(float(np.mean(v)), k) for k, v in sc_cov.items()],
        key=lambda x: x[0],
    )[:12]

    print(f"\n  {'Eval Type':<12}  {'Scenario':<32}  {'n':>4}  {'Avg Coverage':>13}")
    print(f"  {'─'*12}  {'─'*32}  {'─'*4}  {'─'*13}")
    for cov, (et, sc, n) in worst:
        mark = _cov_marker(cov, target)
        print(f"  {et:<12}  {sc:<32}  {n:>4}  {cov:>12.3f}{mark}")

    # ----------------------------------------------------------------
    # Per-method breakdown: coverage deficit/surplus at each n
    # ----------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("  COVERAGE DEVIATION FROM NOMINAL  (mean_coverage − target)")
    print(f"  Computed across all scenarios within each eval type.")
    print(f"{'─'*72}")

    for et in eval_types_present:
        print(f"\n  {et}")
        hdr = f"    {'Method':<20}" + "".join(f"  {nl:>9}" for nl in n_labels)
        print(hdr)
        print(f"    {'─'*20}" + "─" * (11 * len(n_labels)))
        for m in method_labels:
            row = f"    {m:<20}"
            for n in sample_sizes:
                dev = mean_cov(et, m, n) - target
                row += "  {:>9}".format("─") if np.isnan(dev) else f"  {dev:>+9.3f}"
            print(row)

    # ----------------------------------------------------------------
    # Type I error (null scenarios only, delta = 0)
    # ----------------------------------------------------------------
    null_results = [r for r in results if r.is_null]
    if null_results:
        print(f"\n{'─'*72}")
        print("  TYPE I ERROR RATE  (null scenarios: delta = 0)")
        print(f"  Empirical P(CI excludes 0) — target = alpha = {alpha:.2f}")
        print(f"  Values near {alpha:.2f} indicate correct calibration under the null.")
        print(f"{'─'*72}")

        null_agg: dict[tuple, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for r in null_results:
            c_prev, t_prev = null_agg[(r.eval_type, r.method, r.n)]
            null_agg[(r.eval_type, r.method, r.n)] = (c_prev + r.covered, t_prev + r.n_reps)

        present_null_methods = {r.method for r in null_results}
        null_method_labels = [m for m in method_labels if m in present_null_methods]

        for et in eval_types_present:
            et_null = [(et, m, n) for m in null_method_labels for n in sample_sizes
                       if null_agg.get((et, m, n), (0, 0))[1] > 0]
            if not et_null:
                continue
            print(f"\n  {et}  (type I error = 1 − null coverage; target ≈ {alpha:.2f})")
            hdr = f"    {'Method':<20}" + "".join(f"  {nl:>9}" for nl in n_labels)
            print(hdr)
            print(f"    {'─'*20}" + "─" * (11 * len(n_labels)))
            for m in null_method_labels:
                row = f"    {m:<20}"
                for n in sample_sizes:
                    c_tot, t_tot = null_agg.get((et, m, n), (0, 0))
                    if t_tot <= 0:
                        row += f"  {'─':>9}"
                    else:
                        t1e = 1.0 - c_tot / t_tot
                        marker = "▼" if t1e > alpha + 0.04 else ("▲" if t1e < alpha - 0.04 else " ")
                        row += f"  {t1e:>8.3f}{marker}"
                print(row)

    # ----------------------------------------------------------------
    # Overall summary across everything
    # ----------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("  OVERALL SUMMARY  (averaged across all eval types, scenarios, n)")
    print(f"{'─'*72}")

    all_cov: dict[str, list[float]] = defaultdict(list)
    all_wid: dict[str, list[float]] = defaultdict(list)
    all_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for (et, m, n), vals in agg.items():
        all_cov[m].extend(v[0] for v in vals)
        all_wid[m].extend(v[1] for v in vals)
        c, t = agg_counts[(et, m, n)]
        c_prev, t_prev = all_counts[m]
        all_counts[m] = (c_prev + c, t_prev + t)

    print(f"\n  {'Method':<20}  {'Cov':>6}  {'MCSE':>7}  {'Band95':>13}  {'Width':>8}  {'Dev':>8}  {'Time(ms)':>14}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*7}  {'─'*13}  {'─'*8}  {'─'*8}  {'─'*14}")
    for m in method_labels:
        mc = float(np.mean(all_cov[m]))
        mw = float(np.mean(all_wid[m]))
        dev = mc - target
        mark = _cov_marker(mc, target)
        c_tot, t_tot = all_counts[m]
        _, mcse, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        avg_ms, se_ms = _time_stats([r for r in results if r.method == m])
        time_str = f"{avg_ms:.3f}±{se_ms:.3f}" if np.isfinite(avg_ms) else "─"
        print(
            f"  {m:<20}  {mc:>5.3f}{mark}  {mcse:>7.4f}  {f'{lo:.3f}-{hi:.3f}':>13}  {mw:>8.4f}  {dev:>+8.3f}  {time_str:>14}"
        )
    print()

    # ── Cost × coverage transfer analysis ────────────────────────────────
    print(f"\n{'─'*72}")
    print("  COST × COVERAGE TRANSFER ANALYSIS")
    print("  Methods ranked by mean CI time (cheapest first) within each eval type.")
    print("  ★ = cheapest adequate method (coverage ≥ target−0.04) at that N.")
    print(f"{'─'*72}")

    for et in eval_types_present:
        et_results = [r for r in results if r.eval_type == et]
        if not et_results:
            continue
        et_methods = [m for m in method_labels if any(r.method == m for r in et_results)]

        method_avg_time: dict[str, float] = {}
        for m in et_methods:
            avg_ms, _ = _time_stats([r for r in et_results if r.method == m])
            method_avg_time[m] = avg_ms if np.isfinite(avg_ms) else float("inf")

        sorted_methods = sorted(et_methods, key=lambda m: method_avg_time[m])

        # For each N, find cheapest adequate method
        cheapest_adequate: dict[str, str] = {}
        for i, n in enumerate(sample_sizes):
            nl = n_labels[i]
            for m in sorted_methods:
                cov = mean_cov(et, m, n)
                if not np.isnan(cov) and cov >= target - 0.04:
                    cheapest_adequate[nl] = m
                    break

        print(f"\n  [{et}]")
        col_w_t = 16
        col_w_c = 9
        hdr = (f"  {'Method':<20}  {'Time(ms)±SE':>{col_w_t}}"
               + "".join(f"  {nl:>{col_w_c}}" for nl in n_labels))
        print(hdr)
        print(f"  {'─'*20}  {'─'*col_w_t}" + "".join(f"  {'─'*col_w_c}" for _ in n_labels))

        for m in sorted_methods:
            avg_ms, se_ms = _time_stats([r for r in et_results if r.method == m])
            time_str = f"{avg_ms:.3f}±{se_ms:.3f}" if np.isfinite(avg_ms) else "─"
            row = f"  {m:<20}  {time_str:>{col_w_t}}"
            for i, n in enumerate(sample_sizes):
                nl = n_labels[i]
                cov = mean_cov(et, m, n)
                if np.isnan(cov):
                    row += f"  {'─':>{col_w_c}}"
                else:
                    star = "★" if cheapest_adequate.get(nl) == m else " "
                    cell = f"{cov:.3f}{_cov_marker(cov, target)}{star}"
                    row += f"  {cell:>{col_w_c + 1}}"
            print(row)

        slowest_ms = max(
            (method_avg_time[m] for m in sorted_methods if np.isfinite(method_avg_time[m])),
            default=float("nan"),
        )
        print(f"\n  Transfer summary:")
        for i, n in enumerate(sample_sizes):
            nl = n_labels[i]
            ca = cheapest_adequate.get(nl)
            if ca is None:
                print(f"    {nl}: no method achieved adequate coverage")
            else:
                ca_ms = method_avg_time[ca]
                speedup = slowest_ms / ca_ms if ca_ms > 0 else float("inf")
                print(f"    {nl}: {ca:<22} {ca_ms:.4f} ms  ({speedup:.0f}× faster than slowest bootstrap)")
    print()


def save_metric_plot(
    *,
    results: list[SimResult],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
    estimand_label: str,
    out_path: str,
    sample_size_filter: int | None = None,
) -> None:
    """Save a multi-panel interval/box plot of method performance by eval type."""
    plot_results = results
    if sample_size_filter is not None:
        plot_results = [r for r in results if r.n == sample_size_filter]

    if not plot_results:
        print(f"Skipped plot (no matching data): {out_path}")
        return

    target = 1.0 - alpha
    present_methods = {r.method for r in plot_results}
    method_labels = [m for m in REPORT_METHODS if m in present_methods]

    fig, axes = plt.subplots(
        nrows=len(EVAL_TYPES),
        ncols=2,
        figsize=(14.8, 11.5),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.34, "hspace": 0.30},
    )
    if len(EVAL_TYPES) == 1:
        axes = np.array([axes])

    col_titles = [f"Coverage (target={target:.2f}; red interval = MC95)", "Mean CI Width"]

    box_kwargs: dict[str, Any] = {
        "vert": False,
        "showmeans": True,
        "meanline": False,
        "patch_artist": False,
        "whiskerprops": {"linewidth": 1.2, "color": "black"},
        "capprops": {"linewidth": 1.2, "color": "black"},
        "medianprops": {"linewidth": 1.8, "color": "black"},
        "boxprops": {"linewidth": 1.4, "color": "black"},
        "meanprops": {"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4.5},
        "flierprops": {"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 2.8, "alpha": 0.55},
    }

    for r_idx, et in enumerate(EVAL_TYPES):
        et_results = [res for res in plot_results if res.eval_type == et]
        et_methods = [m for m in method_labels if any(res.method == m for res in et_results)]

        cov_series: list[np.ndarray] = []
        wid_series: list[np.ndarray] = []
        cov_uncertainty: list[tuple[float, float, float]] = []

        for method in et_methods:
            subset = [res for res in et_results if res.method == method]
            cov_vals = np.array([res.covered / res.n_reps for res in subset], dtype=float)
            wid_vals = np.array([res.total_width / res.n_reps for res in subset], dtype=float)
            if cov_vals.size > 0:
                cov_series.append(cov_vals)
                wid_series.append(wid_vals)

                covered_tot = int(sum(res.covered for res in subset))
                total_tot = int(sum(res.n_reps for res in subset))
                p_hat, _, lo, hi = _mc_proportion_stats(covered_tot, total_tot)
                cov_uncertainty.append((p_hat, lo, hi))

        metric_series = [cov_series, wid_series]
        metric_xlabels = [
            "Coverage across scenarios × n",
            "CI width across scenarios × n",
        ]

        for c_idx, (ax, series, xlabel) in enumerate(zip(axes[r_idx], metric_series, metric_xlabels)):
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=11)

            if not series:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                ax.set_yticks([])
                continue

            bp = ax.boxplot(series, tick_labels=et_methods, **box_kwargs)

            ax.grid(axis="x", linestyle="--", linewidth=0.65, alpha=0.50)
            ax.set_xlabel(xlabel, fontsize=9.5)
            ax.tick_params(axis="y", labelsize=9.5, pad=2)
            ax.tick_params(axis="x", labelsize=9)
            ax.invert_yaxis()

            if c_idx == 0:
                low_ok = max(0.0, target - 0.04)
                hi_ok = min(1.0, target + 0.04)
                ax.axvspan(low_ok, hi_ok, color="#DDDDDD", alpha=0.35, zorder=0)
                ax.axvline(target, color="black", linestyle="-", linewidth=1.2)
                ax.set_xlim(0.0, 1.0)

                for y_pos, (p_hat, lo, hi) in enumerate(cov_uncertainty, start=1):
                    if np.isnan(lo) or np.isnan(hi):
                        continue
                    ax.hlines(y=y_pos, xmin=lo, xmax=hi, color="tab:red", linewidth=2.1, zorder=5)
                    ax.vlines([lo, hi], y_pos - 0.15, y_pos + 0.15, color="tab:red", linewidth=1.5, zorder=5)
                    if not np.isnan(p_hat):
                        ax.plot(p_hat, y_pos, marker="|", color="tab:red", markersize=10, markeredgewidth=1.8, zorder=6)
            else:
                x_max = max(float(np.max(vals)) for vals in series if vals.size > 0)
                ax.set_xlim(0.0, x_max * 1.08 if x_max > 0 else 1.0)

            if c_idx == 0:
                ax.set_ylabel(et.upper(), fontsize=10.5)

            if bp and "means" in bp:
                for mean_artist in bp["means"]:
                    mean_artist.set_zorder(4)

    if sample_size_filter is not None:
        size_text = str(sample_size_filter)
    elif sample_sizes:
        size_text = ", ".join(str(n) for n in sample_sizes)
    else:
        unique_sizes = sorted({r.n for r in plot_results})
        size_text = ", ".join(str(n) for n in unique_sizes) if unique_sizes else "n/a"

    fig.suptitle(
        f"Method Comparison by Eval Type (Interval / Box Plots)\n"
        f"Estimand: {estimand_label} | reps={n_reps} | alpha={alpha} | n={size_text}",
        fontsize=13.5,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout(rect=[0.03, 0.02, 0.995, 0.95], w_pad=2.6)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out}")


_METHOD_COLORS: dict[str, str] = {
    "bootstrap":          "#1f77b4",
    "bca":                "#2ca02c",
    "bayes_bootstrap":    "#ff7f0e",
    "smooth_bootstrap":   "#9467bd",
    "bootstrap_t":        "#d62728",
    "t_interval":         "#8c564b",
    "wilson":             "#e377c2",
    "jeffreys":           "#e9cd14",
    "wald":               "#7f7f7f",
    "clopper_pearson":    "#bcbd22",
    "bayes_indep":        "#17becf",
    "newcombe_score":     "#aec7e8",
    "bayes_indep_comp":   "#ffbb78",
    "bayes_paired_comp":  "#98df8a",
    "beta":               "#f0027f",
    "logit_t":            "#a6761d",
    "nig":                "#888888",
    "el":                 "#00441b",
}
_N_MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*"]


def save_cost_plot(
    *,
    results: list[SimResult],
    alpha: float,
    n_reps: int,
    estimand_label: str,
    out_path: str,
) -> None:
    """Scatter plot: x = mean CI time (log ms), y = coverage; one subplot per eval type."""
    if not results:
        print(f"Skipped cost plot (no data): {out_path}")
        return

    target = 1.0 - alpha
    present_methods = {r.method for r in results}
    method_labels = [m for m in REPORT_METHODS if m in present_methods]
    sample_sizes = sorted({r.n for r in results})

    nrows = len(EVAL_TYPES)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1,
        figsize=(11.0, 4.5 * nrows),
        squeeze=False,
        gridspec_kw={"hspace": 0.45},
    )

    # Indices of sample sizes to annotate with "n=X" labels (first, middle, last)
    def _label_indices(ns: list[int]) -> set[int]:
        if len(ns) <= 2:
            return set(range(len(ns)))
        return {0, len(ns) // 2, len(ns) - 1}

    for row_idx, et in enumerate(EVAL_TYPES):
        ax = axes[row_idx][0]
        et_results = [r for r in results if r.eval_type == et]

        ax.axhspan(max(0.0, target - 0.04), min(1.0, target + 0.04),
                   color="#DDDDDD", alpha=0.40, zorder=0)
        ax.axhline(target, color="black", linewidth=1.1, linestyle="--", zorder=1)

        legend_method_handles = []

        for m in method_labels:
            color = _METHOD_COLORS.get(m, "#333333")
            m_results = [r for r in et_results if r.method == m]
            if not m_results:
                continue

            # Collect valid (n, avg_ms, cov, xerr) points, sorted by n
            points: list[tuple[int, float, float, float]] = []
            for n in sample_sizes:
                subset = [r for r in m_results if r.n == n]
                if not subset:
                    continue
                avg_ms, se_ms = _time_stats(subset)
                if not np.isfinite(avg_ms) or avg_ms <= 0:
                    continue
                cov = float(np.mean([r.covered / r.n_reps for r in subset]))
                points.append((n, avg_ms, cov, 1.96 * se_ms))

            if not points:
                continue

            xs = [p[1] for p in points]
            ys = [p[2] for p in points]

            # Draw connecting line, then error-bar markers on top
            ax.plot(xs, ys, color=color, linewidth=1.1, alpha=0.55, zorder=2)
            ax.errorbar(
                xs, ys,
                xerr=[p[3] for p in points],
                fmt="o", color=color,
                markersize=6, markeredgewidth=0.7, markeredgecolor="white",
                elinewidth=0.9, capsize=2.5, capthick=0.9,
                alpha=0.90, zorder=3,
            )

            # Sparse n= labels: first, middle, last point only
            label_idxs = _label_indices(points)
            for i, (n, x, y, _) in enumerate(points):
                if i in label_idxs:
                    ax.annotate(
                        f"n={n}",
                        xy=(x, y),
                        xytext=(0, 4),
                        textcoords="offset points",
                        fontsize=6.5,
                        ha="center",
                        va="bottom",
                        color=color,
                        alpha=0.85,
                    )

            legend_method_handles.append(
                plt.Line2D([0], [0], marker="o", color=color,
                           markerfacecolor=color, markersize=7, label=m,
                           linewidth=1.5)
            )

        ax.set_xscale("log")
        ax.set_xlabel("Mean CI time (ms) — log scale  [error bars: ±1.96 SE]", fontsize=9.5)
        ax.set_ylabel("Coverage rate", fontsize=9.5)
        ax.set_title(f"eval type: {et}", fontsize=10.5)
        ax.set_ylim(max(0.0, target - 0.20), min(1.01, target + 0.12))
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
        ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.35)
        ax.tick_params(labelsize=8.5)

        if not et_results:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

        if legend_method_handles:
            ax.legend(
                handles=legend_method_handles,
                title="Method",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                fontsize=7.5,
                title_fontsize=8,
                framealpha=0.85,
                ncol=1,
            )

    fig.suptitle(
        f"Cost × Coverage Trade-off\n"
        f"Estimand: {estimand_label}  |  x = mean CI compute time  |  y = empirical coverage  |"
        f"  target = {target:.0%}  |  reps={n_reps}",
        fontsize=10.5,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=[0.02, 0.02, 0.80, 0.93])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cost plot: {out}")


def save_coverage_vs_n_plot(
    *,
    results: list[SimResult],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
    estimand_label: str,
    out_path: str,
) -> None:
    """Coverage vs. sample size line plots — one subplot per eval type, all methods overlaid."""
    if not results:
        print(f"Skipped coverage-vs-n plot (no data): {out_path}")
        return

    target = 1.0 - alpha
    present_methods = {r.method for r in results}
    method_labels = [m for m in REPORT_METHODS if m in present_methods]

    rows = [
        {
            "eval_type": r.eval_type,
            "scenario": r.scenario,
            "method": r.method,
            "n": r.n,
            "coverage": r.covered / r.n_reps,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df = df[df["method"].isin(method_labels)]

    # Average across reps within each (eval_type, scenario, method, n) cell,
    # then compute mean ± std across scenarios.
    scenario_level = (
        df.groupby(["eval_type", "scenario", "method", "n"], as_index=False)
          .agg(coverage=("coverage", "mean"))
    )
    agg = (
        scenario_level.groupby(["eval_type", "method", "n"], as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            coverage_count=("coverage", "count"),
        )
    )

    palette = {m: _METHOD_COLORS.get(m, "#333333") for m in method_labels}

    fig, axes = plt.subplots(
        1, len(EVAL_TYPES),
        figsize=(5.5 * len(EVAL_TYPES), 5),
        squeeze=False,
    )

    for col_idx, et in enumerate(EVAL_TYPES):
        ax = axes[0][col_idx]
        et_agg = agg[agg["eval_type"] == et].copy()
        et_methods = [m for m in method_labels if m in et_agg["method"].values]

        if et_agg.empty:
            ax.axhline(target, linestyle="--", color="tab:cyan", linewidth=1.2)
            ax.set_xticks(sample_sizes)
            ax.set_xticklabels([str(n) for n in sample_sizes])
            ax.set_xlabel("Sample size (n)")
            ax.set_ylabel("Empirical coverage" if col_idx == 0 else "")
            ax.set_title(et.upper())
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        sns.lineplot(
            data=et_agg,
            x="n",
            y="coverage_mean",
            hue="method",
            hue_order=et_methods,
            palette=palette,
            marker=None,
            linewidth=1.0,
            alpha=0.70,
            ax=ax,
        )

        for method, sub in et_agg.groupby("method"):
            if sub["coverage_std"].isna().all():
                continue
            sub = sub.sort_values("n")
            color = _METHOD_COLORS.get(str(method), "#333333")
            se = sub["coverage_std"] / np.sqrt(sub["coverage_count"])
            ax.errorbar(
                sub["n"],
                sub["coverage_mean"],
                yerr=se,
                fmt="none",
                color=color,
                elinewidth=0.8,
                capsize=2,
                alpha=0.45,
            )
            ax.scatter(
                sub["n"],
                sub["coverage_mean"],
                s=28,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                alpha=0.85,
                zorder=3,
            )

        ns = sorted(et_agg["n"].unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.axhline(target, linestyle="--", color="tab:cyan", linewidth=1.2)
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Empirical coverage" if col_idx == 0 else "")
        ax.set_title(et.upper())
        if et_methods:
            ax.legend(title="Method", fontsize=7.5, title_fontsize=8)

    fig.suptitle(
        f"Coverage vs. Sample Size\n"
        f"Estimand: {estimand_label} | reps={n_reps} | alpha={alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved coverage-vs-n plot: {out}")


def save_width_vs_n_plot(
    *,
    results: list[SimResult],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
    estimand_label: str,
    out_path: str,
) -> None:
    """Mean CI width vs. sample size line plots — one subplot per eval type, all methods overlaid."""
    if not results:
        print(f"Skipped width-vs-n plot (no data): {out_path}")
        return

    present_methods = {r.method for r in results}
    method_labels = [m for m in REPORT_METHODS if m in present_methods]

    rows = [
        {
            "eval_type": r.eval_type,
            "scenario": r.scenario,
            "method": r.method,
            "n": r.n,
            "width": r.total_width / r.n_reps,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df = df[df["method"].isin(method_labels)]

    scenario_level = (
        df.groupby(["eval_type", "scenario", "method", "n"], as_index=False)
          .agg(width=("width", "mean"))
    )
    agg = (
        scenario_level.groupby(["eval_type", "method", "n"], as_index=False)
        .agg(
            width_mean=("width", "mean"),
            width_std=("width", "std"),
            width_count=("width", "count"),
        )
    )

    palette = {m: _METHOD_COLORS.get(m, "#333333") for m in method_labels}

    fig, axes = plt.subplots(
        1, len(EVAL_TYPES),
        figsize=(5.5 * len(EVAL_TYPES), 5),
        squeeze=False,
    )

    for col_idx, et in enumerate(EVAL_TYPES):
        ax = axes[0][col_idx]
        et_agg = agg[agg["eval_type"] == et].copy()
        et_methods = [m for m in method_labels if m in et_agg["method"].values]

        if et_agg.empty:
            ax.set_xticks(sample_sizes)
            ax.set_xticklabels([str(n) for n in sample_sizes])
            ax.set_xlabel("Sample size (n)")
            ax.set_ylabel("Mean CI width" if col_idx == 0 else "")
            ax.set_title(et.upper())
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        sns.lineplot(
            data=et_agg,
            x="n",
            y="width_mean",
            hue="method",
            hue_order=et_methods,
            palette=palette,
            marker=None,
            linewidth=1.0,
            alpha=0.70,
            ax=ax,
        )

        for method, sub in et_agg.groupby("method"):
            if sub["width_std"].isna().all():
                continue
            sub = sub.sort_values("n")
            color = _METHOD_COLORS.get(str(method), "#333333")
            se = sub["width_std"] / np.sqrt(sub["width_count"])
            ax.errorbar(
                sub["n"],
                sub["width_mean"],
                yerr=se,
                fmt="none",
                color=color,
                elinewidth=0.8,
                capsize=2,
                alpha=0.45,
            )
            ax.scatter(
                sub["n"],
                sub["width_mean"],
                s=28,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                alpha=0.85,
                zorder=3,
            )

        ns = sorted(et_agg["n"].unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Mean CI width" if col_idx == 0 else "")
        ax.set_title(et.upper())
        if et_methods:
            ax.legend(title="Method", fontsize=7.5, title_fontsize=8)

    fig.suptitle(
        f"CI Width vs. Sample Size\n"
        f"Estimand: {estimand_label} | reps={n_reps} | alpha={alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved width-vs-n plot: {out}")


def save_results_artifacts(
    *,
    results: list[SimResult],
    alpha: float,
    sample_sizes: list[int],
    n_reps: int,
    estimand_label: str,
    out_dir: str,
    run_stem: str,
) -> None:
    """Save per-cell results CSV and full text report log for a simulation run."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    csv_path = out_base / f"{run_stem}_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "eval_type",
                "scenario",
                "n",
                "method",
                "n_reps",
                "covered",
                "total_width",
                "coverage",
                "mean_width",
                "mcse",
                "band95_low",
                "band95_high",
                "avg_time_ms",
                "se_time_ms",
            ]
        )
        for r in results:
            coverage = r.covered / r.n_reps
            mean_width = r.total_width / r.n_reps
            _, mcse, lo, hi = _mc_proportion_stats(r.covered, r.n_reps)
            avg_ms, se_ms = _time_stats([r])
            writer.writerow(
                [
                    r.eval_type,
                    r.scenario,
                    r.n,
                    r.method,
                    r.n_reps,
                    r.covered,
                    f"{r.total_width:.8f}",
                    f"{coverage:.8f}",
                    f"{mean_width:.8f}",
                    f"{mcse:.8f}",
                    f"{lo:.8f}",
                    f"{hi:.8f}",
                    f"{avg_ms:.6f}" if np.isfinite(avg_ms) else "",
                    f"{se_ms:.6f}" if np.isfinite(se_ms) else "",
                ]
            )

    summary_path = out_base / f"{run_stem}_summary.log"
    report_buf = io.StringIO()
    with redirect_stdout(report_buf):
        print_report(
            results=results,
            sample_sizes=sample_sizes,
            alpha=alpha,
            n_reps=n_reps,
            estimand_label=estimand_label,
        )

    summary_path.write_text(report_buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _run_benchmark(
    *,
    estimand: str,
    runs: int,
    statistic: str,
    reps: int,
    bootstrap_n: int,
    bayes_n: int,
    alpha: float,
    sizes: list[int],
    seed: int,
    scenario_suite: str,
    progress_mode: str,
    plot_mode: str,
    save_results: str,
    out_dir: str,
    plots_dir: str,
    eval_types: list[str] | None = None,
    scenario_label_contains: list[str] | None = None,
    icc_values: list[float] = (0.10, 0.25, 0.40),
    cohens_d_values: list[float] = (0.3,),
    include_null: bool = False,
    label: str | None = None,
    n_workers: int = 1,
) -> None:
    if label:
        print(f"\n{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}")

    print(f"\nBootstrap CI Simulation")
    print(f"  Estimand        : {estimand}")
    print(f"  Scenario suite  : {scenario_suite}")
    if eval_types:
        print(f"  Eval types      : {eval_types}")
    else:
        print(f"  Eval types      : all ({EVAL_TYPES})")
    if scenario_label_contains:
        print(f"  Scenario filter : contains all {scenario_label_contains}")
    if estimand == "pairwise":
        print(f"  Runs per input  : {runs}")
        print(f"  Statistic       : {statistic}")
        print(f"  ICC values      : {list(icc_values)}")
        print(f"  Cohen's d       : {list(cohens_d_values)}")
        print(f"  Include null    : {include_null}")
    print(f"  Reps per cell   : {reps}")
    print(f"  Bootstrap draws : {bootstrap_n}")
    print(f"  Bayes samples   : {bayes_n}")
    print(f"  Alpha / CI level: {alpha} / {(1 - alpha):.0%}")
    print(f"  Sample sizes    : {sizes}")
    print(f"  Seed            : {seed}")
    print(f"  Workers         : {n_workers}")
    print(f"  Progress mode   : {progress_mode}")
    print(f"  Plots           : {plot_mode}")
    print(f"  Save results    : {save_results}")
    print(f"  Out dir         : {out_dir}")
    if plot_mode == "save":
        print(f"  Plots dir       : {plots_dir}")

    print("\nBuilding scenarios …", end="", flush=True)
    if estimand == "mean":
        scenarios = build_scenarios(suite=scenario_suite)
    else:
        scenarios = build_pair_scenarios(
            suite=scenario_suite,
            icc_values=list(icc_values),
            cohens_d_values=list(cohens_d_values),
            include_null=include_null,
        )

    if eval_types:
        requested_eval_types = set(eval_types)
        scenarios = [s for s in scenarios if s.eval_type in requested_eval_types]
        if not scenarios:
            raise ValueError(
                f"No scenarios left after filtering eval types {sorted(requested_eval_types)}."
            )

    if scenario_label_contains:
        needles = [tok.lower() for tok in scenario_label_contains]
        scenarios = [
            s for s in scenarios
            if all(tok in s.label.lower() for tok in needles)
        ]
        if not scenarios:
            raise ValueError(
                "No scenarios left after filtering by scenario labels containing "
                f"{scenario_label_contains}."
            )

    n_by_type = {et: sum(1 for s in scenarios if s.eval_type == et) for et in EVAL_TYPES}
    print(
        f"  {len(scenarios)} total  "
        + "  ".join(f"{et}: {n_by_type[et]}" for et in EVAL_TYPES)
    )

    cells = len(scenarios) * len(sizes)
    bootstrap_calls = cells * reps * len(METHODS)
    if estimand == "mean":
        binary_cells = n_by_type["binary"] * len(sizes)
        wilson_calls = binary_cells * reps
        bayes_calls = binary_cells * reps
        print(
            f"\nRunning {cells} cells × {reps} reps × {len(METHODS)} bootstrap methods "
            f"= {bootstrap_calls:,} CI calls, plus {wilson_calls:,} Wilson calls and "
            f"{bayes_calls:,} Bayesian-independent calls (binary only) …"
        )
        results = run_simulation(
            scenarios=scenarios,
            sample_sizes=sizes,
            n_reps=reps,
            n_bootstrap=bootstrap_n,
            bayes_n=bayes_n,
            alpha=alpha,
            progress_mode=progress_mode,
            seed=seed,
            n_workers=n_workers,
        )
        estimand_label = "template mean"
    else:
        binary_cells = n_by_type["binary"] * len(sizes)
        newcombe_calls = binary_cells * reps if statistic == "mean" else 0
        bayes_pair_calls = 2 * binary_cells * reps if statistic == "mean" else 0
        pairwise_method_count = len(METHODS) + (len(PAIRWISE_EXTRA_METHODS) if statistic == "mean" else 0)
        extra = (
            f", plus {newcombe_calls:,} Newcombe calls and {bayes_pair_calls:,} Bayesian pairwise calls "
            f"(binary, statistic=mean)"
            if newcombe_calls
            else ""
        )
        print(
            f"\nRunning {cells} cells × {reps} reps × {pairwise_method_count} methods "
            f"= {bootstrap_calls:,} CI calls{extra} …"
        )
        results = run_pairwise_simulation(
            scenarios=scenarios,
            sample_sizes=sizes,
            n_reps=reps,
            n_bootstrap=bootstrap_n,
            bayes_n=bayes_n,
            alpha=alpha,
            runs=runs,
            statistic=statistic,
            progress_mode=progress_mode,
            seed=seed,
            n_workers=n_workers,
        )
        estimand_label = f"paired template difference ({statistic}, runs={runs})"

    print_report(
        results,
        sample_sizes=sizes,
        alpha=alpha,
        n_reps=reps,
        estimand_label=estimand_label,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_stem = (
        f"sim_compare_boot_{estimand}_{scenario_suite}_"
        f"runs{runs}_stat{statistic}_reps{reps}_{stamp}"
    )

    if save_results == "save":
        save_results_artifacts(
            results=results,
            alpha=alpha,
            sample_sizes=sizes,
            n_reps=reps,
            estimand_label=estimand_label,
            out_dir=out_dir,
            run_stem=run_stem,
        )

    if plot_mode == "save":
        overall_file_name = (
            f"sim_compare_boot_{estimand}_{scenario_suite}_"
            f"runs{runs}_stat{statistic}_reps{reps}_overall_{stamp}.png"
        )
        save_metric_plot(
            results=results,
            sample_sizes=sizes,
            alpha=alpha,
            n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(Path(plots_dir) / overall_file_name),
        )

        for n in sizes:
            per_n_file_name = (
                f"sim_compare_boot_{estimand}_{scenario_suite}_"
                f"runs{runs}_stat{statistic}_reps{reps}_n{n}_{stamp}.png"
            )
            save_metric_plot(
                results=results,
                sample_sizes=[n],
                alpha=alpha,
                n_reps=reps,
                estimand_label=estimand_label,
                out_path=str(Path(plots_dir) / per_n_file_name),
                sample_size_filter=n,
            )

        cost_file_name = (
            f"sim_compare_boot_{estimand}_{scenario_suite}_"
            f"runs{runs}_stat{statistic}_reps{reps}_cost_coverage_{stamp}.png"
        )
        save_cost_plot(
            results=results,
            alpha=alpha,
            n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(Path(plots_dir) / cost_file_name),
        )

        cov_vs_n_file_name = (
            f"sim_compare_boot_{estimand}_{scenario_suite}_"
            f"runs{runs}_stat{statistic}_reps{reps}_coverage_vs_n_{stamp}.png"
        )
        save_coverage_vs_n_plot(
            results=results,
            sample_sizes=sizes,
            alpha=alpha,
            n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(Path(plots_dir) / cov_vs_n_file_name),
        )

        wid_vs_n_file_name = (
            f"sim_compare_boot_{estimand}_{scenario_suite}_"
            f"runs{runs}_stat{statistic}_reps{reps}_width_vs_n_{stamp}.png"
        )
        save_width_vs_n_plot(
            results=results,
            sample_sizes=sizes,
            alpha=alpha,
            n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(Path(plots_dir) / wid_vs_n_file_name),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario-suite",
        choices=SCENARIO_SUITES,
        default="expanded",
        help="Scenario breadth to run (default: expanded)",
    )
    parser.add_argument(
        "--eval-types",
        nargs="+",
        choices=EVAL_TYPES,
        default=None,
        metavar="TYPE",
        help=(
            "Restrict simulation to specific eval types. "
            f"Choices: {' '.join(EVAL_TYPES)}. "
            "Default: all types."
        ),
    )
    parser.add_argument(
        "--scenario-label-contains",
        nargs="+",
        default=None,
        metavar="TOKEN",
        help=(
            "Keep only scenarios whose label contains all provided substrings "
            "(case-insensitive). Useful for targeted stress tests, e.g. "
            "--scenario-label-contains binary onesided extreme"
        ),
    )
    parser.add_argument(
        "--official-test",
        action="store_true",
        help=(
            "Run the intensive official benchmark battery with robust preset "
            "options (overrides routine settings)."
        ),
    )
    parser.add_argument(
        "--official-test-pairwise-only",
        action="store_true",
        help=(
            "Run only the pairwise phase of the intensive official benchmark "
            "battery (overrides routine settings)."
        ),
    )
    parser.add_argument(
        "--progress",
        choices=PROGRESS_MODES,
        default="bar",
        help="Progress display mode: bar (ETA), cell, or off (default: bar)",
    )
    parser.add_argument(
        "--plots",
        choices=PLOT_MODES,
        default="save",
        help="Post-run plotting mode: save or off (default: save)",
    )
    parser.add_argument(
        "--save-results",
        choices=RESULTS_MODES,
        default="save",
        help="Write run results CSV and summary log: save or off (default: save)",
    )
    parser.add_argument(
        "--out-dir",
        default="simulations/out",
        help="Base output directory for non-plot artifacts (default: simulations/out)",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory for saved plots when --plots save (default: <out-dir>/plots)",
    )
    parser.add_argument(
        "--estimand",
        choices=["mean", "pairwise"],
        default="mean",
        help=(
            "Target estimand: 'mean' (single-sample means, default) or "
            "'pairwise' (paired template difference A-B, matching compare paths)."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="R",
        help=(
            "Runs per input for --estimand pairwise. R>=3 activates nested "
            "bootstrap logic (default: 1)."
        ),
    )
    parser.add_argument(
        "--statistic",
        choices=["mean", "median"],
        default="mean",
        help="Statistic for --estimand pairwise (default: mean)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=200,
        metavar="N",
        help="Monte Carlo repetitions per (scenario, n) cell (default: 200)",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=500,
        metavar="N",
        help="Bootstrap replicates per CI estimate (default: 500)",
    )
    parser.add_argument(
        "--bayes-n",
        type=int,
        default=2000,
        metavar="N",
        help="Posterior samples per Bayesian CI estimate (default: 2000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level; CIs are (1-alpha)*100%% (default: 0.05)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        metavar="N",
        help="Sample sizes to evaluate (default: 5 10 20 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed (default: 42)",
    )
    parser.add_argument(
        "--icc-values",
        type=float,
        nargs="+",
        default=[0.05, 0.20, 0.40, 0.60, 0.80],
        metavar="ICC",
        help=(
            "Intraclass correlation values for pairwise scenarios. "
            "ICC = between-input variance / total variance. "
            "Each value generates a separate scenario batch. "
            "(default: 0.05 0.20 0.40 0.60 0.80)"
        ),
    )
    parser.add_argument(
        "--cohens-d-values",
        type=float,
        nargs="+",
        default=[0.3],
        metavar="D",
        help=(
            "Standardised effect sizes (Cohen's d = delta/total_std) for "
            "pairwise scenarios. Null (d=0) is added automatically when "
            "--include-null is set. (default: 0.3)"
        ),
    )
    parser.add_argument(
        "--include-null",
        action="store_true",
        default=False,
        help=(
            "Add delta=0 (null) pairwise scenarios for Type I error measurement. "
            "Null results are shown in a separate section of the report."
        ),
    )
    parser.add_argument(
        "--alpha-sweep",
        type=float,
        nargs="+",
        default=None,
        metavar="A",
        help=(
            "Run the full benchmark at each of these alpha levels in sequence. "
            "Overrides --alpha for the official test and non-interactive runs. "
            "E.g.: --alpha-sweep 0.05 0.10"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        metavar="N",
        help=(
            "Number of parallel worker processes for the simulation. "
            "Each (scenario, n) cell runs independently in its own process. "
            f"Default: all logical CPUs ({os.cpu_count()}). "
            "Use --workers 1 for sequential execution."
        ),
    )
    args = parser.parse_args()
    plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")

    if args.official_test and args.official_test_pairwise_only:
        parser.error("Choose only one of --official-test or --official-test-pairwise-only.")

    if args.official_test or args.official_test_pairwise_only:
        print("\nRunning OFFICIAL TEST battery with robust, intensive presets.")
        print("This mode intentionally prioritizes rigor over runtime.")

        if args.official_test_pairwise_only:
            print("Pairwise-only mode enabled: skipping single-sample phase.")

        # Sizes start from 10: decisions based on n=5 are rarely meaningful
        # in practice, and some methods are numerically unstable at n=5.
        official_sizes = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        official_icc   = [0.05, 0.20, 0.40, 0.60, 0.80]
        official_d     = [0.2, 0.4]
        official_alphas = [0.01, 0.05]

        if args.official_test:
            for phase_alpha in official_alphas:
                _run_benchmark(
                    estimand="mean",
                    runs=1,
                    statistic="mean",
                    reps=2000,
                    bootstrap_n=10000,
                    bayes_n=10000,
                    alpha=phase_alpha,
                    sizes=official_sizes,
                    seed=args.seed,
                    scenario_suite="expanded",
                    progress_mode=args.progress,
                    plot_mode=args.plots,
                    save_results=args.save_results,
                    out_dir=args.out_dir,
                    plots_dir=plots_dir,
                    eval_types=args.eval_types,
                    scenario_label_contains=args.scenario_label_contains,
                    n_workers=args.workers,
                    label=(
                        f"OFFICIAL TEST · Single-sample mean estimand · "
                        f"alpha={phase_alpha}"
                    ),
                )

        for phase_idx, phase_alpha in enumerate(official_alphas):
            _run_benchmark(
                estimand="pairwise",
                runs=1,
                statistic="mean",
                reps=2000,
                bootstrap_n=10000,
                bayes_n=10000,
                alpha=phase_alpha,
                sizes=official_sizes,
                seed=args.seed + 1 + phase_idx,
                scenario_suite="expanded",
                icc_values=official_icc,
                cohens_d_values=official_d,
                include_null=True,
                progress_mode=args.progress,
                plot_mode=args.plots,
                save_results=args.save_results,
                out_dir=args.out_dir,
                plots_dir=plots_dir,
                eval_types=args.eval_types,
                scenario_label_contains=args.scenario_label_contains,
                n_workers=args.workers,
                label=(
                    f"OFFICIAL TEST · Pairwise estimand · "
                    f"ICC sweep {official_icc} · d sweep {official_d} · "
                    f"alpha={phase_alpha}"
                ),
            )
        return

    # Non-official: respect --alpha-sweep if provided, otherwise use --alpha
    alpha_list = args.alpha_sweep if args.alpha_sweep else [args.alpha]
    for run_alpha in alpha_list:
        _run_benchmark(
            estimand=args.estimand,
            runs=args.runs,
            statistic=args.statistic,
            reps=args.reps,
            bootstrap_n=args.bootstrap_n,
            bayes_n=args.bayes_n,
            alpha=run_alpha,
            sizes=args.sizes,
            seed=args.seed,
            scenario_suite=args.scenario_suite,
            icc_values=args.icc_values,
            cohens_d_values=args.cohens_d_values,
            include_null=args.include_null,
            progress_mode=args.progress,
            plot_mode=args.plots,
            save_results=args.save_results,
            out_dir=args.out_dir,
            plots_dir=plots_dir,
            eval_types=args.eval_types,
            scenario_label_contains=args.scenario_label_contains,
            n_workers=args.workers,
        )


if __name__ == "__main__":
    main()
