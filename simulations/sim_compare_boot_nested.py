#!/usr/bin/env python3
"""
sim_compare_boot_nested.py — CI method comparison for multi-run (nested) single-sample estimation.

Extension of sim_compare_boot.py to the case where each benchmark input is evaluated
R times, producing a score matrix of shape (N inputs, R runs).  The estimand is the
grand population mean  μ = E[score_{i,r}].

For each scenario, a "run noise fraction"
    f_run = σ²_run / (σ²_input + σ²_run)
controls how much of the total score variance is attributable to within-input
stochastic run noise versus stable between-input differences.  The simulation
sweeps f_run from near-zero (near-deterministic outputs per input, high ICC) to
near-one (run-dominated noise, low ICC).

CI families under comparison:

  cell-means  Reduce each input to a per-input cell mean (mean of R runs), then
              apply existing single-sample CI methods.  The sample variance of
              cell means captures both sources of variance (σ²_input + σ²_run/R),
              so coverage should be nominal regardless of f_run.

  nested      Two-level nested bootstrap over the full (N, R) matrix: outer
              resample of inputs, inner resample of runs within each selected
              input.  Theoretically optimal for multi-run data; reduces to
              plain bootstrap when R = 1.

  flat        Apply each method to the flattened N×R array treating all
              observations as iid.  Under-covers when f_run is small (ICC > 0)
              because the iid assumption ignores input-level clustering; converges
              to nominal coverage as f_run → 1.

Methods:
  bootstrap               Percentile bootstrap  (cell means)
  bca                     BCa bootstrap  (cell means)
  bayes_bootstrap         Bayesian bootstrap  (cell means)
  smooth_bootstrap        Smoothed bootstrap  (cell means)
  bootstrap_t             Studentized bootstrap  (cell means)
  t_interval              Student's t  (cell means)
  bootstrap_nested        Two-level nested bootstrap  (N×R matrix)
  bayes_bootstrap_nested  Bayesian nested bootstrap  (N×R matrix)
  smooth_bootstrap_nested Smoothed two-level nested bootstrap  (N×R matrix)
  bca_nested              BCa interval using nested bootstrap replicates  (N×R matrix)
  bootstrap_t_nested      Studentized bootstrap-t using nested resampling  (N×R matrix)
  t_interval_flat         Student's t  (all N×R obs, iid — baseline)
  bootstrap_flat          Percentile bootstrap  (all N×R obs, iid — baseline)
  Binary extra (flat iid): wilson_flat, wald_flat, clopper_pearson_flat, bayes_indep_flat
  Binary extra (nested):   wilson_de, wilson_od, beta_binomial
  Continuous extra (cell means): beta, logit_t, nig, el

Eval output types:
  binary      Bernoulli 0/1 via Bernoulli-Beta hierarchical model
  continuous  Continuous [0, 1] via Beta base + Gaussian run noise
  likert      Integer 1–5 via latent-normal model
  grades      Scores 0–100 via truncated-normal model

Usage:
  python simulations/sim_compare_boot_nested.py
  python simulations/sim_compare_boot_nested.py --runs 3
  python simulations/sim_compare_boot_nested.py --runs-sweep 1 2 3 5 10
  python simulations/sim_compare_boot_nested.py --run-noise-fracs 0.01 0.1 0.3 0.5 0.7 0.9
  python simulations/sim_compare_boot_nested.py --reps 500 --bootstrap-n 1000
  python simulations/sim_compare_boot_nested.py --sizes 10 20 50
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
from dataclasses import dataclass, field
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
for _p in [_HERE, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.resampling import (
        bootstrap_ci_1d,
        bootstrap_t_ci_1d,
        bca_interval_1d,
        bayes_bootstrap_means_1d,
        smooth_bootstrap_means_1d,
        bootstrap_means_1d,
        resolve_resampling_method,
        wald_ci_1d,
        clopper_pearson_ci_1d,
        t_interval_ci_1d,
        beta_ci_1d,
        logit_t_ci_1d,
        nig_ci_1d,
        nig_ci_nested,
        el_ci_1d,
        bootstrap_means_nested,
        bayes_bootstrap_means_nested,
        smooth_bootstrap_means_nested,
        bootstrap_t_ci_nested,
        wilson_nested_de,
        wilson_nested_od,
        wilson_nested_bb,
        bootstrap_diffs_nested,
        bayes_bootstrap_diffs_nested,
        smooth_bootstrap_diffs_nested,
        newcombe_paired_ci,
        tango_paired_ci,
        tango_paired_ci_multirun_discordance,
        tango_paired_ci_multirun_moments,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t"]

# Nested methods: applied directly to the (N, R) score matrix
BOOTSTRAP_NESTED_METHOD   = "bootstrap_nested"
BAYES_NESTED_METHOD       = "bayes_bootstrap_nested"
SMOOTH_NESTED_METHOD      = "smooth_bootstrap_nested"
BCA_NESTED_METHOD         = "bca_nested"
BOOTSTRAP_T_NESTED_METHOD = "bootstrap_t_nested"
NESTED_METHODS            = [BOOTSTRAP_NESTED_METHOD, BAYES_NESTED_METHOD, SMOOTH_NESTED_METHOD, BCA_NESTED_METHOD, BOOTSTRAP_T_NESTED_METHOD]

# t-interval on cell means (cluster-robust, always valid)
T_INTERVAL_METHOD         = "t_interval"

# Flat methods: applied to flattened N×R array (iid assumption — baseline)
T_INTERVAL_FLAT_METHOD    = "t_interval_flat"
BOOTSTRAP_FLAT_METHOD     = "bootstrap_flat"
FLAT_METHODS              = [T_INTERVAL_FLAT_METHOD, BOOTSTRAP_FLAT_METHOD]

# Binary-specific flat methods
WILSON_FLAT_METHOD        = "wilson_flat"
WALD_FLAT_METHOD          = "wald_flat"
CP_FLAT_METHOD            = "clopper_pearson_flat"
BAYES_INDEP_FLAT_METHOD   = "bayes_indep_flat"
BINARY_FLAT_METHODS       = [WILSON_FLAT_METHOD, WALD_FLAT_METHOD, CP_FLAT_METHOD, BAYES_INDEP_FLAT_METHOD]

# Binary-specific nested methods (multi-run clustered Wilson variants)
WILSON_DE_METHOD          = "wilson_de"
WILSON_OD_METHOD          = "wilson_od"
BETA_BINOMIAL_METHOD      = "beta_binomial"
BINARY_NESTED_METHODS     = [WILSON_DE_METHOD, WILSON_OD_METHOD, BETA_BINOMIAL_METHOD]

# Continuous-specific nested methods (operate on full N×R matrix)
NIG_NESTED_METHOD         = "nig_nested"
CONTINUOUS_NESTED_METHODS = [NIG_NESTED_METHOD]

# Pairwise-diff nested methods: applied directly to (N, R) pair matrices
BOOTSTRAP_DIFF_NESTED_METHOD = "bootstrap_diff_nested"
BAYES_DIFF_NESTED_METHOD     = "bayes_diff_nested"
SMOOTH_DIFF_NESTED_METHOD    = "smooth_diff_nested"
PAIR_DIFF_NESTED_METHODS     = [BOOTSTRAP_DIFF_NESTED_METHOD, BAYES_DIFF_NESTED_METHOD, SMOOTH_DIFF_NESTED_METHOD]

# Pairwise binary flat methods (first-run-only iid baseline)
TANGO_FLAT_METHOD         = "tango_flat"
NEWCOMBE_FLAT_METHOD      = "newcombe_flat"
BINARY_PAIR_FLAT_METHODS  = [TANGO_FLAT_METHOD, NEWCOMBE_FLAT_METHOD]

# Pairwise binary nested (full N×R matrix)
TANGO_MULTIRUN_METHOD      = "tango_multirun_disc"
TANGO_MULTIRUN_MOMENTS_METHOD = "tango_multirun_mmnt"
BINARY_PAIR_NESTED_METHODS = [TANGO_MULTIRUN_METHOD, TANGO_MULTIRUN_MOMENTS_METHOD]

# Continuous-only methods on cell means
BETA_METHOD    = "beta"
LOGIT_T_METHOD = "logit_t"
NIG_METHOD     = "nig"
EL_METHOD      = "el"
CONTINUOUS_EXTRA_METHODS = [BETA_METHOD, LOGIT_T_METHOD, NIG_METHOD, EL_METHOD]

# Ordered list for single-sample report tables
REPORT_METHODS = (
    METHODS
    + [T_INTERVAL_METHOD]
    + NESTED_METHODS
    + FLAT_METHODS
    + BINARY_FLAT_METHODS
    + BINARY_NESTED_METHODS
    + CONTINUOUS_NESTED_METHODS
    + CONTINUOUS_EXTRA_METHODS
)

# Ordered list for pairwise report tables
REPORT_PAIRWISE_METHODS = (
    METHODS
    + [T_INTERVAL_METHOD]
    + PAIR_DIFF_NESTED_METHODS
    + BINARY_PAIR_FLAT_METHODS
    + BINARY_PAIR_NESTED_METHODS
)

# Unified method order used in plots when mixed result sets are present.
REPORT_ALL_METHODS = list(dict.fromkeys(REPORT_METHODS + REPORT_PAIRWISE_METHODS))

EVAL_TYPES       = ["binary", "continuous", "likert", "grades"]
SCENARIO_SUITES  = ["standard", "expanded"]
PROGRESS_MODES   = ["bar", "cell", "off"]
PLOT_MODES       = ["save", "off"]
RESULTS_MODES    = ["save", "off"]

RUN_NOISE_FRACS_DEFAULT = [0.01, 0.10, 0.30, 0.50, 0.70, 0.90]

OFFICIAL_RUNS_SWEEP = [3, 5, 8, 12]
OFFICIAL_RUN_NOISE_FRACS = [0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.95]
OFFICIAL_ICC_VALUES = [0.05, 0.15, 0.30, 0.50]
OFFICIAL_SIZES = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass
class MultiRunScenario:
    label: str
    eval_type: str
    generate: Callable[[np.random.Generator, int, int], np.ndarray]
    """generate(rng, n_inputs, n_runs) -> ndarray of shape (n_inputs, n_runs)"""
    true_mean: float
    run_noise_frac: float


@dataclass
class PairMultiRunScenario:
    label: str
    eval_type: str
    generate_pair: Callable[[np.random.Generator, int, int], tuple[np.ndarray, np.ndarray]]
    """generate_pair(rng, n_inputs, n_runs) -> (A, B) each of shape (n_inputs, n_runs)"""
    true_diff: float
    run_noise_frac: float
    icc: float = 0.0
    is_null: bool = False


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
                end="", flush=True,
            )
            if is_final:
                print()
            return

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
            end="", flush=True,
        )
        if is_final:
            print()


def _beta_var(a: float, b: float) -> float:
    return a * b / ((a + b) ** 2 * (a + b + 1))


def _true_mean_clipped_normal(mu: float, sigma: float, lo: float = 0.0, hi: float = 100.0) -> float:
    rng = np.random.default_rng(0)
    return float(np.clip(rng.normal(mu, sigma, size=2_000_000), lo, hi).mean())


def _estimate_true_mean_mc(
    generate: Callable[[np.random.Generator, int, int], np.ndarray],
    *,
    seed: int = 0,
    n_mc: int = 500_000,
) -> float:
    """Estimate population mean via large MC draw: E[x_{i,r}] ≈ grand mean of large sample."""
    rng = np.random.default_rng(seed)
    scores = generate(rng, n_mc, 1)  # (n_mc, 1)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def build_multirun_scenarios(
    run_noise_fracs: list[float],
    suite: str = "standard",
    heteroscedastic: bool = False,
) -> list[MultiRunScenario]:
    """Build single-sample multi-run scenarios parameterised by run_noise_frac.

    For each (eval_type shape, run_noise_frac) pair one MultiRunScenario is
    created.  The generator returns an (n, runs) array.

    run_noise_frac = σ²_run / (σ²_input + σ²_run):
      ≈ 0   near-deterministic outputs per input (high ICC)
      = 0.5  equal input and run variance
      ≈ 1   run noise dominates (low ICC)

    DGPs:
      binary:     Bernoulli-Beta hierarchical.  p_i ~ Beta(conc·p0, conc·(1-p0)),
                  ICC = 1/(conc+1),  f_run = conc/(conc+1).
      continuous: base_i ~ Beta(a,b), x_{i,r} = clip(base_i + N(0,σ_run), 0,1).
                  σ²_run = Var(base) · f/(1-f).
      likert:     latent_i ~ N(mu, σ_input), x_{i,r} = round(clip(latent_i + N(0,σ_run),1,5)).
                  σ_input = √(1-f) · s0,  σ_run = √f · s0,  where s0 is the shape sigma.
      grades:     same as likert, clipped to [0, 100].
    """
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")

    scenarios: list[MultiRunScenario] = []

    # ── Binary ──────────────────────────────────────────────────────────────
    # p_i ~ Beta(conc·p0, conc·(1-p0)),  conc = f/(1-f)
    # true_mean = p0 exactly (E[p_i] = p0 regardless of conc)
    binary_ps = [0.1, 0.3, 0.5, 0.7, 0.9]
    if suite == "expanded":
        binary_ps += [0.02, 0.05, 0.95, 0.98]

    for p0 in binary_ps:
        for f in run_noise_fracs:
            conc = float(max(f, 1e-6)) / float(max(1.0 - f, 1e-6))
            p_, c_ = p0, conc

            def _gen_bin(
                rng: np.random.Generator, n: int, runs: int,
                _p: float = p_, _c: float = c_,
            ) -> np.ndarray:
                p_i = rng.beta(_c * _p, _c * (1.0 - _p), size=(n, 1))
                return rng.binomial(1, p_i, size=(n, runs)).astype(float)

            scenarios.append(MultiRunScenario(
                label=f"p={p0}|f={f:.2f}",
                eval_type="binary",
                generate=_gen_bin,
                true_mean=p0,
                run_noise_frac=f,
            ))

    # ── Continuous [0, 1] ───────────────────────────────────────────────────
    # base_i ~ Beta(a, b),  σ_run = √(Var_base · f/(1-f))
    # x_{i,r} = clip(base_i + N(0, σ_run), 0, 1)
    continuous_specs: list[tuple[str, float, float]] = [
        ("Uniform",       1.0, 1.0),
        ("U-shaped",      0.5, 0.5),
        ("right-skewed",  2.0, 8.0),
        ("left-skewed",   8.0, 2.0),
        ("moderate-skew", 2.0, 5.0),
    ]
    if suite == "expanded":
        continuous_specs.extend([
            ("extreme-right",   0.35, 6.0),
            ("extreme-left",    6.0, 0.35),
            ("near-boundaries", 0.3, 0.3),
            ("near-center",     6.0, 6.0),
        ])

    for shape_label, a_b, b_b in continuous_specs:
        var_base = _beta_var(a_b, b_b)
        for f in run_noise_fracs:
            sigma_run = float(np.sqrt(var_base * f / max(1.0 - f, 1e-9)))
            a_, b_, sr_ = a_b, b_b, sigma_run

            def _gen_cont(
                rng: np.random.Generator, n: int, runs: int,
                _a: float = a_, _b: float = b_, _sr: float = sr_,
                _hetero: bool = heteroscedastic,
            ) -> np.ndarray:
                base  = rng.beta(_a, _b, size=(n, 1))
                if _sr > 0.0:
                    if _hetero:
                        sigma_i = _sr * 2.0 * np.sqrt(base * (1.0 - base))
                        noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                    else:
                        noise = rng.normal(0.0, _sr, size=(n, runs))
                else:
                    noise = np.zeros((n, runs))
                return np.clip(base + noise, 0.0, 1.0)

            true_mean = _estimate_true_mean_mc(_gen_cont)
            scenarios.append(MultiRunScenario(
                label=f"{shape_label}|f={f:.2f}",
                eval_type="continuous",
                generate=_gen_cont,
                true_mean=true_mean,
                run_noise_frac=f,
            ))

    # logit-normal base (always included)
    for f in run_noise_fracs:
        rng_tmp = np.random.default_rng(1)
        base_logit = rng_tmp.normal(-0.35, 1.35, 200_000)
        base_vals  = 1.0 / (1.0 + np.exp(-base_logit))
        var_logit  = float(np.var(base_vals))
        sigma_run  = float(np.sqrt(var_logit * f / max(1.0 - f, 1e-9)))
        f_, sr_ = f, sigma_run

        def _gen_logit(
            rng: np.random.Generator, n: int, runs: int,
            _f: float = f_, _sr: float = sr_,
            _hetero: bool = heteroscedastic,
        ) -> np.ndarray:
            logits = rng.normal(-0.35, 1.35, size=(n, 1))
            base   = 1.0 / (1.0 + np.exp(-logits))
            if _sr > 0.0:
                if _hetero:
                    sigma_i = _sr * 2.0 * np.sqrt(base * (1.0 - base))
                    noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                else:
                    noise = rng.normal(0.0, _sr, size=(n, runs))
            else:
                noise = np.zeros((n, runs))
            return np.clip(base + noise, 0.0, 1.0)

        true_mean = _estimate_true_mean_mc(_gen_logit)
        scenarios.append(MultiRunScenario(
            label=f"logit-normal|f={f:.2f}",
            eval_type="continuous",
            generate=_gen_logit,
            true_mean=true_mean,
            run_noise_frac=f,
        ))

    # zero-inflated and one-inflated
    for shape_name, spike_val, beta_a, beta_b, spike_prob in [
        ("zero-inflated",   0.0, 2.0, 4.0, 0.70),
        ("one-inflated",    1.0, 4.0, 2.0, 0.70),
    ]:
        # estimate base variance via MC on single-run
        rng_tmp = np.random.default_rng(2)
        spike_mask = rng_tmp.random(200_000) < spike_prob
        body = rng_tmp.beta(beta_a, beta_b, 200_000)
        base_vals_zi = np.where(spike_mask, spike_val, body)
        var_zi = float(np.var(base_vals_zi))

        for f in run_noise_fracs:
            sigma_run = float(np.sqrt(var_zi * f / max(1.0 - f, 1e-9)))
            sv_, ba_, bb_, sp_, sr_ = spike_val, beta_a, beta_b, spike_prob, sigma_run

            def _gen_infl(
                rng: np.random.Generator, n: int, runs: int,
                _sv: float = sv_, _ba: float = ba_, _bb: float = bb_,
                _sp: float = sp_, _sr: float = sr_,
                _hetero: bool = heteroscedastic,
            ) -> np.ndarray:
                spike_i = rng.random((n, 1)) < _sp   # per-input spike, persistent across runs
                base    = np.where(spike_i, _sv, rng.beta(_ba, _bb, size=(n, 1)))
                if _sr > 0.0:
                    if _hetero:
                        sigma_i = _sr * 2.0 * np.sqrt(base * (1.0 - base))
                        noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                    else:
                        noise = rng.normal(0.0, _sr, size=(n, runs))
                else:
                    noise = np.zeros((n, runs))
                return np.clip(base + noise, 0.0, 1.0)

            true_mean = _estimate_true_mean_mc(_gen_infl)
            scenarios.append(MultiRunScenario(
                label=f"{shape_name}|f={f:.2f}",
                eval_type="continuous",
                generate=_gen_infl,
                true_mean=true_mean,
                run_noise_frac=f,
            ))

    if suite == "expanded":
        # mixture Beta
        rng_tmp = np.random.default_rng(3)
        sel = rng_tmp.binomial(1, 0.55, size=200_000).astype(bool)
        v   = np.empty(200_000, dtype=float)
        v[sel]  = rng_tmp.beta(0.5, 4.0, size=int(np.sum(sel)))
        v[~sel] = rng_tmp.beta(5.5, 1.2, size=int(np.sum(~sel)))
        var_mix = float(np.var(v))

        for f in run_noise_fracs:
            sigma_run = float(np.sqrt(var_mix * f / max(1.0 - f, 1e-9)))
            sr_ = sigma_run

            def _gen_mix(
                rng: np.random.Generator, n: int, runs: int, _sr: float = sr_,
                _hetero: bool = heteroscedastic,
            ) -> np.ndarray:
                selector = rng.binomial(1, 0.55, size=(n, 1)).astype(bool)
                base = np.where(
                    selector,
                    rng.beta(0.5, 4.0, size=(n, 1)),
                    rng.beta(5.5, 1.2, size=(n, 1)),
                )
                if _sr > 0.0:
                    if _hetero:
                        sigma_i = _sr * 2.0 * np.sqrt(base * (1.0 - base))
                        noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                    else:
                        noise = rng.normal(0.0, _sr, size=(n, runs))
                else:
                    noise = np.zeros((n, runs))
                return np.clip(base + noise, 0.0, 1.0)

            true_mean = _estimate_true_mean_mc(_gen_mix)
            scenarios.append(MultiRunScenario(
                label=f"mix-Beta|f={f:.2f}",
                eval_type="continuous",
                generate=_gen_mix,
                true_mean=true_mean,
                run_noise_frac=f,
            ))

    # ── Likert 1–5 ──────────────────────────────────────────────────────────
    # latent_i ~ N(mu, σ_input),  σ_input = √(1-f) · s0,  σ_run = √f · s0
    # x_{i,r} = round(clip(latent_i + N(0, σ_run), 1, 5))
    # Both latent sigma components are derived from the original shape sigma s0
    # so that total latent variance = s0² regardless of f_run.
    _likert_standard: list[tuple[str, float, float, bool]] = [
        ("uniform",       3.0, 2.0,  False),
        ("skewed-low",    2.0, 1.1,  False),
        ("skewed-high",   4.0, 1.1,  False),
        ("bimodal",       3.0, 0.65, True),   # sigma is the within-mode std
        ("center-peaked", 3.0, 0.55, False),
    ]
    if suite == "expanded":
        _likert_standard += [
            ("near-floor",   1.5, 0.65, False),
            ("near-ceiling", 4.5, 0.65, False),
            ("flat-middle",  3.0, 1.4,  False),
        ]

    for shape_label, mu_lat, s0, is_bimodal in _likert_standard:
        for f in run_noise_fracs:
            sigma_input_l = float(np.sqrt(max(1.0 - f, 0.0))) * s0
            sigma_run_l   = float(np.sqrt(f)) * s0
            m_, si_, sr_, bim_ = mu_lat, sigma_input_l, sigma_run_l, is_bimodal

            if is_bimodal:
                # Two latent modes at mu±1.5; within-mode spread scaled by sigma_input
                def _gen_likert_bim(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _si: float = si_, _sr: float = sr_,
                    _hetero: bool = heteroscedastic,
                ) -> np.ndarray:
                    mode      = rng.integers(0, 2, size=(n, 1))
                    mu_mode   = np.where(mode == 0, _m - 1.5, _m + 1.5)
                    latent_i  = mu_mode + rng.normal(0.0, _si, size=(n, 1))
                    if _sr > 0.0:
                        if _hetero:
                            p_i = np.clip((latent_i - 1.0) / 4.0, 0.0, 1.0)
                            sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                            noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                        else:
                            noise = rng.normal(0.0, _sr, size=(n, runs))
                    else:
                        noise = np.zeros((n, runs))
                    return np.rint(np.clip(latent_i + noise, 1.0, 5.0))

                gen_fn = _gen_likert_bim
            else:
                def _gen_likert_norm(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _si: float = si_, _sr: float = sr_,
                    _hetero: bool = heteroscedastic,
                ) -> np.ndarray:
                    latent_i = rng.normal(_m, _si, size=(n, 1)) if _si > 0.0 else np.full((n, 1), _m)
                    if _sr > 0.0:
                        if _hetero:
                            p_i = np.clip((latent_i - 1.0) / 4.0, 0.0, 1.0)
                            sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                            noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                        else:
                            noise = rng.normal(0.0, _sr, size=(n, runs))
                    else:
                        noise = np.zeros((n, runs))
                    return np.rint(np.clip(latent_i + noise, 1.0, 5.0))

                gen_fn = _gen_likert_norm

            true_mean = _estimate_true_mean_mc(gen_fn)
            scenarios.append(MultiRunScenario(
                label=f"{shape_label}|f={f:.2f}",
                eval_type="likert",
                generate=gen_fn,
                true_mean=true_mean,
                run_noise_frac=f,
            ))

    # ── Grades 0–100 ────────────────────────────────────────────────────────
    # Same structure as Likert but clipped to [0, 100]; s0 is the original sigma.
    _grades_standard: list[tuple[str, float, float]] = [
        ("symmetric",     50.0, 20.0),
        ("high-scoring",  75.0, 15.0),
        ("low-scoring",   35.0, 20.0),
        ("ceiling-heavy", 88.0, 10.0),
        ("floor-heavy",   12.0, 10.0),
    ]
    if suite == "expanded":
        _grades_standard += [
            ("very-high",    92.0,  7.0),
            ("very-low",      8.0,  7.0),
            ("high-variance", 50.0, 34.0),
        ]

    for shape_label, mu_g, s0_g in _grades_standard:
        for f in run_noise_fracs:
            sigma_input_g = float(np.sqrt(max(1.0 - f, 0.0))) * s0_g
            sigma_run_g   = float(np.sqrt(f)) * s0_g
            m_, si_, sr_ = mu_g, sigma_input_g, sigma_run_g

            def _gen_grades(
                rng: np.random.Generator, n: int, runs: int,
                _m: float = m_, _si: float = si_, _sr: float = sr_,
                _hetero: bool = heteroscedastic,
            ) -> np.ndarray:
                latent_i = rng.normal(_m, _si, size=(n, 1)) if _si > 0.0 else np.full((n, 1), _m)
                if _sr > 0.0:
                    if _hetero:
                        p_i = np.clip(latent_i / 100.0, 0.0, 1.0)
                        sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                        noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                    else:
                        noise = rng.normal(0.0, _sr, size=(n, runs))
                else:
                    noise = np.zeros((n, runs))
                return np.clip(latent_i + noise, 0.0, 100.0)

            true_mean = _estimate_true_mean_mc(_gen_grades)
            scenarios.append(MultiRunScenario(
                label=f"{shape_label}|f={f:.2f}",
                eval_type="grades",
                generate=_gen_grades,
                true_mean=true_mean,
                run_noise_frac=f,
            ))

    # Grades mixture (always included)
    for f in run_noise_fracs:
        s0_mix = 22.0  # typical component sigma
        sigma_input_m = float(np.sqrt(max(1.0 - f, 0.0))) * s0_mix
        sigma_run_m   = float(np.sqrt(f)) * s0_mix
        si_, sr_ = sigma_input_m, sigma_run_m

        def _gen_grade_mix(
            rng: np.random.Generator, n: int, runs: int,
            _si: float = si_, _sr: float = sr_,
            _hetero: bool = heteroscedastic,
        ) -> np.ndarray:
            flags = rng.choice(3, size=(n, 1), p=[0.20, 0.50, 0.30])
            mu_i = np.where(flags == 0, 22.0, np.where(flags == 1, 58.0, 88.0))
            latent_i = mu_i + rng.normal(0.0, _si, size=(n, 1)) if _si > 0.0 else mu_i.astype(float)
            if _sr > 0.0:
                if _hetero:
                    p_i = np.clip(latent_i / 100.0, 0.0, 1.0)
                    sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                    noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                else:
                    noise = rng.normal(0.0, _sr, size=(n, runs))
            else:
                noise = np.zeros((n, runs))
            return np.clip(latent_i + noise, 0.0, 100.0)

        true_mean = _estimate_true_mean_mc(_gen_grade_mix)
        scenarios.append(MultiRunScenario(
            label=f"mixture-3comp|f={f:.2f}",
            eval_type="grades",
            generate=_gen_grade_mix,
            true_mean=true_mean,
            run_noise_frac=f,
        ))

    return scenarios


# ---------------------------------------------------------------------------
# Pairwise multi-run scenarios
# ---------------------------------------------------------------------------


def _estimate_true_pair_diff_mc(
    generate_pair: Callable[[np.random.Generator, int, int], tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = 0,
    n_mc: int = 300_000,
) -> float:
    """Estimate E[cell_mean(A) − cell_mean(B)] via a large synthetic sample."""
    rng = np.random.default_rng(seed)
    a, b = generate_pair(rng, n_mc, 1)
    return float(np.mean(a[:, 0] - b[:, 0]))


def build_pair_multirun_scenarios(
    run_noise_fracs: list[float],
    suite: str = "standard",
    cohens_d_values: list[float] | None = None,
    include_null: bool = False,
    heteroscedastic: bool = False,
) -> list[PairMultiRunScenario]:
    """Build pairwise multi-run paired-difference scenarios parameterised by run_noise_frac.

    Each scenario yields two (n, R) arrays (A and B) from the same hierarchical
    DGP used in :func:`build_multirun_scenarios`, with an additive delta on B's
    latent mean.  The ICC (between-input clustering) is directly determined by
    ``run_noise_frac``:  low ``f_run`` → high ICC → clustered runs.

    Parameters
    ----------
    run_noise_fracs : list[float]
        Values of f_run = σ²_run / (σ²_input + σ²_run) to sweep.
    suite : str
        ``'standard'`` or ``'expanded'``.
    cohens_d_values : list[float] or None
        Effect sizes (delta / total_std) for non-null scenarios.  Defaults to
        ``[0.3]``.
    include_null : bool
        If True, prepend delta=0 scenarios for each (shape, f_run) combination.
    """
    if cohens_d_values is None:
        cohens_d_values = [0.3]

    d_list: list[float] = []
    if include_null:
        d_list.append(0.0)
    d_list.extend(d for d in cohens_d_values if d > 0.0)

    scenarios: list[PairMultiRunScenario] = []

    # ── Binary ──────────────────────────────────────────────────────────────
    binary_shapes: list[tuple[str, float]] = [
        ("binary-balanced", 0.5),
        ("binary-high",     0.8),
        ("binary-low",      0.2),
    ]
    if suite == "expanded":
        binary_shapes += [("binary-rare", 0.05), ("binary-near-ceil", 0.93)]

    for shape_label, p0 in binary_shapes:
        total_std = float(np.sqrt(p0 * (1.0 - p0)))
        for f in run_noise_fracs:
            conc = float(max(f, 1e-6)) / float(max(1.0 - f, 1e-6))
            icc = 1.0 / (conc + 1.0)
            for d in d_list:
                delta = d * total_std
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|f={f:.2f}|{effect_tag}"
                p_, c_, delta_ = p0, conc, delta

                def _gen_bin_pair(
                    rng: np.random.Generator, n: int, runs: int,
                    _p: float = p_, _c: float = c_, _d: float = delta_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    p_a = rng.beta(_c * _p, _c * (1.0 - _p), size=(n, 1))
                    p_b = np.clip(p_a + _d, 0.0, 1.0)
                    a = rng.binomial(1, p_a, size=(n, runs)).astype(float)
                    b = rng.binomial(1, p_b, size=(n, runs)).astype(float)
                    return a, b

                true_diff = 0.0 if is_null else _estimate_true_pair_diff_mc(_gen_bin_pair)
                scenarios.append(PairMultiRunScenario(
                    label=label, eval_type="binary",
                    generate_pair=_gen_bin_pair,
                    true_diff=true_diff, run_noise_frac=f, icc=icc, is_null=is_null,
                ))

    # ── Continuous [0, 1] ───────────────────────────────────────────────────
    continuous_shapes: list[tuple[str, float, float]] = [
        ("cont-uniform",     1.0, 1.0),
        ("cont-right-skew",  2.0, 8.0),
        ("cont-left-skew",   8.0, 2.0),
    ]
    if suite == "expanded":
        continuous_shapes += [
            ("cont-moderate-skew", 2.0, 5.0),
            ("cont-boundary",      0.6, 0.6),
        ]

    for shape_label, a_b, b_b in continuous_shapes:
        var_base   = _beta_var(a_b, b_b)
        total_std  = float(np.sqrt(var_base))
        for f in run_noise_fracs:
            sigma_run = float(np.sqrt(var_base * f / max(1.0 - f, 1e-9)))
            icc = 1.0 / (1.0 + sigma_run ** 2 / max(var_base, 1e-12))
            for d in d_list:
                delta = d * total_std
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|f={f:.2f}|{effect_tag}"
                a_, b_, sr_, delta_ = a_b, b_b, sigma_run, delta

                def _gen_cont_pair(
                    rng: np.random.Generator, n: int, runs: int,
                    _a: float = a_, _b: float = b_,
                    _sr: float = sr_, _d: float = delta_,
                    _hetero: bool = heteroscedastic,
                ) -> tuple[np.ndarray, np.ndarray]:
                    base = rng.beta(_a, _b, size=(n, 1))
                    if _sr > 0.0:
                        if _hetero:
                            sigma_i = _sr * 2.0 * np.sqrt(base * (1.0 - base))
                            noise_a = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                            noise_b = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                        else:
                            noise_a = rng.normal(0.0, _sr, size=(n, runs))
                            noise_b = rng.normal(0.0, _sr, size=(n, runs))
                    else:
                        noise_a = np.zeros((n, runs))
                        noise_b = np.zeros((n, runs))
                    a_sc = np.clip(base + noise_a, 0.0, 1.0)
                    b_sc = np.clip(base + _d + noise_b, 0.0, 1.0)
                    return a_sc, b_sc

                true_diff = 0.0 if is_null else _estimate_true_pair_diff_mc(_gen_cont_pair)
                scenarios.append(PairMultiRunScenario(
                    label=label, eval_type="continuous",
                    generate_pair=_gen_cont_pair,
                    true_diff=true_diff, run_noise_frac=f, icc=icc, is_null=is_null,
                ))

    # ── Likert 1–5 ──────────────────────────────────────────────────────────
    _LIKERT_TOTAL_STD = 1.2
    likert_shapes: list[tuple[str, float]] = [
        ("likert-mid",  3.0),
        ("likert-low",  2.2),
        ("likert-high", 3.8),
    ]
    if suite == "expanded":
        likert_shapes += [("likert-floor", 1.8), ("likert-ceil", 4.2)]

    for shape_label, mu_lat in likert_shapes:
        for f in run_noise_fracs:
            sigma_input_l = float(np.sqrt(max(1.0 - f, 0.0))) * _LIKERT_TOTAL_STD
            sigma_run_l   = float(np.sqrt(f)) * _LIKERT_TOTAL_STD
            icc = (1.0 - f)
            for d in d_list:
                delta = d * _LIKERT_TOTAL_STD
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|f={f:.2f}|{effect_tag}"
                m_, si_, sr_, delta_ = mu_lat, sigma_input_l, sigma_run_l, delta

                def _gen_likert_pair(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _si: float = si_, _sr: float = sr_, _d: float = delta_,
                    _hetero: bool = heteroscedastic,
                ) -> tuple[np.ndarray, np.ndarray]:
                    base = rng.normal(_m, _si, size=(n, 1)) if _si > 0.0 else np.full((n, 1), _m)
                    if _sr > 0.0:
                        if _hetero:
                            p_i = np.clip((base - 1.0) / 4.0, 0.0, 1.0)
                            sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                            noise_a = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                            noise_b = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                        else:
                            noise_a = rng.normal(0.0, _sr, size=(n, runs))
                            noise_b = rng.normal(0.0, _sr, size=(n, runs))
                    else:
                        noise_a = np.zeros((n, runs))
                        noise_b = np.zeros((n, runs))
                    a_sc = np.rint(np.clip(base + noise_a, 1.0, 5.0))
                    b_sc = np.rint(np.clip(base + _d + noise_b, 1.0, 5.0))
                    return a_sc, b_sc

                true_diff = 0.0 if is_null else _estimate_true_pair_diff_mc(_gen_likert_pair)
                scenarios.append(PairMultiRunScenario(
                    label=label, eval_type="likert",
                    generate_pair=_gen_likert_pair,
                    true_diff=true_diff, run_noise_frac=f, icc=icc, is_null=is_null,
                ))

    # ── Grades 0–100 ────────────────────────────────────────────────────────
    _GRADES_TOTAL_STD = 20.0
    grades_shapes: list[tuple[str, float]] = [
        ("grades-mid",  55.0),
        ("grades-low",  35.0),
        ("grades-high", 78.0),
    ]
    if suite == "expanded":
        grades_shapes += [("grades-ceiling", 86.0), ("grades-floor", 20.0)]

    for shape_label, mu_g in grades_shapes:
        for f in run_noise_fracs:
            sigma_input_g = float(np.sqrt(max(1.0 - f, 0.0))) * _GRADES_TOTAL_STD
            sigma_run_g   = float(np.sqrt(f)) * _GRADES_TOTAL_STD
            icc = (1.0 - f)
            for d in d_list:
                delta = d * _GRADES_TOTAL_STD
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|f={f:.2f}|{effect_tag}"
                m_, si_, sr_, delta_ = mu_g, sigma_input_g, sigma_run_g, delta

                def _gen_grades_pair(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _si: float = si_, _sr: float = sr_, _d: float = delta_,
                    _hetero: bool = heteroscedastic,
                ) -> tuple[np.ndarray, np.ndarray]:
                    base = rng.normal(_m, _si, size=(n, 1)) if _si > 0.0 else np.full((n, 1), _m)
                    if _sr > 0.0:
                        if _hetero:
                            p_i = np.clip(base / 100.0, 0.0, 1.0)
                            sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                            noise_a = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                            noise_b = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                        else:
                            noise_a = rng.normal(0.0, _sr, size=(n, runs))
                            noise_b = rng.normal(0.0, _sr, size=(n, runs))
                    else:
                        noise_a = np.zeros((n, runs))
                        noise_b = np.zeros((n, runs))
                    a_sc = np.clip(base + noise_a, 0.0, 100.0)
                    b_sc = np.clip(base + _d + noise_b, 0.0, 100.0)
                    return a_sc, b_sc

                true_diff = 0.0 if is_null else _estimate_true_pair_diff_mc(_gen_grades_pair)
                scenarios.append(PairMultiRunScenario(
                    label=label, eval_type="grades",
                    generate_pair=_gen_grades_pair,
                    true_diff=true_diff, run_noise_frac=f, icc=icc, is_null=is_null,
                ))

    return scenarios


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    eval_type: str
    scenario: str
    n: int
    method: str
    n_reps: int
    covered: int
    total_width: float
    total_time: float = 0.0
    total_time_sq: float = 0.0
    run_noise_frac: float = 0.0
    runs: int = 1
    is_null: bool = False
    icc: float = 0.0


def _stat(values: np.ndarray, statistic: str = "mean") -> float:
    return float(np.median(values)) if statistic == "median" else float(np.mean(values))


def _wilson_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _bayes_indep_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    n = int(values.shape[0])
    s = int(np.sum(values >= 0.5))
    lo, hi = stats.beta(s + 1, n - s + 1).interval(1.0 - alpha)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-cell worker
# ---------------------------------------------------------------------------

_WORKER_SCENARIOS: list = []


def _run_multirun_cell(args: tuple) -> list[SimResult]:
    """Run all reps for one (scenario, n, runs) cell — multi-run mean estimand."""
    sc_idx, n, runs, n_reps, n_bootstrap, bayes_n, alpha, seed = args
    scenario = _WORKER_SCENARIOS[sc_idx]
    rng = np.random.default_rng(seed)

    active_methods = METHODS + [T_INTERVAL_METHOD] + NESTED_METHODS + FLAT_METHODS
    if scenario.eval_type == "binary":
        active_methods += BINARY_FLAT_METHODS + BINARY_NESTED_METHODS
    elif scenario.eval_type == "continuous":
        active_methods += CONTINUOUS_NESTED_METHODS
        active_methods += CONTINUOUS_EXTRA_METHODS

    covered:   dict[str, int]   = {m: 0   for m in active_methods}
    total_w:   dict[str, float] = {m: 0.0 for m in active_methods}
    total_t:   dict[str, float] = {m: 0.0 for m in active_methods}
    total_t_sq: dict[str, float] = {m: 0.0 for m in active_methods}

    _cont_fns = {
        BETA_METHOD:    beta_ci_1d,
        LOGIT_T_METHOD: logit_t_ci_1d,
        NIG_METHOD:     nig_ci_1d,
        EL_METHOD:      el_ci_1d,
    }

    for _rep in range(n_reps):
        scores     = scenario.generate(rng, n, runs)   # (n, runs)
        cell_means = scores.mean(axis=1)               # (n,)
        flat       = scores.ravel()                    # (n*runs,)
        obs_mean   = float(np.mean(cell_means))

        # ── Cell-means bootstrap family ──────────────────────────────────
        for method in METHODS:
            n_draws = bayes_n if method == "bayes_bootstrap" else n_bootstrap
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ci_low, ci_high = bootstrap_ci_1d(
                        cell_means, obs_mean, method=method,
                        n_bootstrap=n_draws, alpha=alpha, rng=rng,
                    )
            except Exception:
                ci_low = ci_high = obs_mean
            el = time.perf_counter() - t0
            total_t[method]    += el
            total_t_sq[method] += el * el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[method] += 1
            total_w[method] += ci_high - ci_low

        # ── t-interval on cell means ──────────────────────────────────────
        t0 = time.perf_counter()
        try:
            ci_low, ci_high = t_interval_ci_1d(cell_means, alpha)
        except Exception:
            ci_low = ci_high = obs_mean
        el = time.perf_counter() - t0
        total_t[T_INTERVAL_METHOD]    += el
        total_t_sq[T_INTERVAL_METHOD] += el * el
        if ci_low <= scenario.true_mean <= ci_high:
            covered[T_INTERVAL_METHOD] += 1
        total_w[T_INTERVAL_METHOD] += ci_high - ci_low

        # ── Nested bootstrap (full N×R matrix) ───────────────────────────
        for method, fn in [
            (BOOTSTRAP_NESTED_METHOD, bootstrap_means_nested),
            (BAYES_NESTED_METHOD,     bayes_bootstrap_means_nested),
            (SMOOTH_NESTED_METHOD,    smooth_bootstrap_means_nested),
        ]:
            n_draws = bayes_n if method == BAYES_NESTED_METHOD else n_bootstrap
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    # Smooth nested bootstrap can frequently fall back on
                    # degenerate cells (std=0); keep logs readable.
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*falling back to plain bootstrap; no KDE smoothing applied.*",
                        category=UserWarning,
                    )
                    boot_stats = fn(scores, n_draws, rng)
                ci_low  = float(np.percentile(boot_stats, 100 * alpha / 2))
                ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
            except Exception:
                ci_low = ci_high = obs_mean
            el = time.perf_counter() - t0
            total_t[method]    += el
            total_t_sq[method] += el * el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[method] += 1
            total_w[method] += ci_high - ci_low

        # ── BCa interval using nested bootstrap replicates ────────────────
        t0 = time.perf_counter()
        try:
            boot_stats = bootstrap_means_nested(scores, n_bootstrap, rng)
            ci_low, ci_high = bca_interval_1d(cell_means, obs_mean, boot_stats, alpha)
        except Exception:
            ci_low = ci_high = obs_mean
        el = time.perf_counter() - t0
        total_t[BCA_NESTED_METHOD]    += el
        total_t_sq[BCA_NESTED_METHOD] += el * el
        if ci_low <= scenario.true_mean <= ci_high:
            covered[BCA_NESTED_METHOD] += 1
        total_w[BCA_NESTED_METHOD] += ci_high - ci_low

        # ── Bootstrap-t via nested resampling ────────────────────────────
        t0 = time.perf_counter()
        try:
            ci_low, ci_high = bootstrap_t_ci_nested(scores, obs_mean, n_bootstrap, alpha, rng)
        except Exception:
            ci_low = ci_high = obs_mean
        el = time.perf_counter() - t0
        total_t[BOOTSTRAP_T_NESTED_METHOD]    += el
        total_t_sq[BOOTSTRAP_T_NESTED_METHOD] += el * el
        if ci_low <= scenario.true_mean <= ci_high:
            covered[BOOTSTRAP_T_NESTED_METHOD] += 1
        total_w[BOOTSTRAP_T_NESTED_METHOD] += ci_high - ci_low

        # ── Flat methods (all N×R obs, iid assumption) ───────────────────
        t0 = time.perf_counter()
        try:
            ci_low, ci_high = t_interval_ci_1d(flat, alpha)
        except Exception:
            ci_low = ci_high = float(np.mean(flat))
        el = time.perf_counter() - t0
        total_t[T_INTERVAL_FLAT_METHOD]    += el
        total_t_sq[T_INTERVAL_FLAT_METHOD] += el * el
        if ci_low <= scenario.true_mean <= ci_high:
            covered[T_INTERVAL_FLAT_METHOD] += 1
        total_w[T_INTERVAL_FLAT_METHOD] += ci_high - ci_low

        t0 = time.perf_counter()
        try:
            boot_flat = bootstrap_means_1d(flat, n_bootstrap, rng)
            ci_low  = float(np.percentile(boot_flat, 100 * alpha / 2))
            ci_high = float(np.percentile(boot_flat, 100 * (1 - alpha / 2)))
        except Exception:
            ci_low = ci_high = float(np.mean(flat))
        el = time.perf_counter() - t0
        total_t[BOOTSTRAP_FLAT_METHOD]    += el
        total_t_sq[BOOTSTRAP_FLAT_METHOD] += el * el
        if ci_low <= scenario.true_mean <= ci_high:
            covered[BOOTSTRAP_FLAT_METHOD] += 1
        total_w[BOOTSTRAP_FLAT_METHOD] += ci_high - ci_low

        # ── Binary flat (integer-count methods on flattened N×R data) ────
        if scenario.eval_type == "binary":
            succ_flat = int(np.sum(flat))
            n_flat    = len(flat)
            obs_flat  = float(np.mean(flat))

            t0 = time.perf_counter()
            ci_low, ci_high = _wilson_ci(succ_flat, n_flat, alpha)
            el = time.perf_counter() - t0
            total_t[WILSON_FLAT_METHOD]    += el
            total_t_sq[WILSON_FLAT_METHOD] += el * el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[WILSON_FLAT_METHOD] += 1
            total_w[WILSON_FLAT_METHOD] += ci_high - ci_low

            t0 = time.perf_counter()
            ci_low, ci_high = wald_ci_1d(flat, alpha)
            el = time.perf_counter() - t0
            total_t[WALD_FLAT_METHOD]    += el
            total_t_sq[WALD_FLAT_METHOD] += el * el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[WALD_FLAT_METHOD] += 1
            total_w[WALD_FLAT_METHOD] += ci_high - ci_low

            t0 = time.perf_counter()
            try:
                ci_low, ci_high = clopper_pearson_ci_1d(flat, alpha)
            except Exception:
                ci_low = ci_high = obs_flat
            el = time.perf_counter() - t0
            total_t[CP_FLAT_METHOD]    += el
            total_t_sq[CP_FLAT_METHOD] += el * el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[CP_FLAT_METHOD] += 1
            total_w[CP_FLAT_METHOD] += ci_high - ci_low

            t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_indep_ci(flat, alpha)
            except Exception:
                ci_low = ci_high = obs_flat
            el = time.perf_counter() - t0
            total_t[BAYES_INDEP_FLAT_METHOD]    += el
            total_t_sq[BAYES_INDEP_FLAT_METHOD] += el * el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[BAYES_INDEP_FLAT_METHOD] += 1
            total_w[BAYES_INDEP_FLAT_METHOD] += ci_high - ci_low

            # ── Clustered Wilson variants (nested, multi-run) ─────────────
            for method, fn in [
                (WILSON_DE_METHOD,     wilson_nested_de),
                (WILSON_OD_METHOD,     wilson_nested_od),
                (BETA_BINOMIAL_METHOD, wilson_nested_bb),
            ]:
                t0 = time.perf_counter()
                try:
                    ci_low, ci_high = fn(scores, alpha)
                except Exception:
                    ci_low = ci_high = obs_flat
                el = time.perf_counter() - t0
                total_t[method]    += el
                total_t_sq[method] += el * el
                if ci_low <= scenario.true_mean <= ci_high:
                    covered[method] += 1
                total_w[method] += ci_high - ci_low

        # ── Continuous extra methods on cell means ────────────────────────
        if scenario.eval_type == "continuous":
            t0 = time.perf_counter()
            try:
                ci_low, ci_high = nig_ci_nested(scores, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            el = time.perf_counter() - t0
            total_t[NIG_NESTED_METHOD]    += el
            total_t_sq[NIG_NESTED_METHOD] += el * el
            if ci_low <= scenario.true_mean <= ci_high:
                covered[NIG_NESTED_METHOD] += 1
            total_w[NIG_NESTED_METHOD] += ci_high - ci_low

            for meth, fn in _cont_fns.items():
                t0 = time.perf_counter()
                try:
                    ci_low, ci_high = fn(cell_means, alpha)
                except Exception:
                    ci_low = ci_high = obs_mean
                el = time.perf_counter() - t0
                total_t[meth]    += el
                total_t_sq[meth] += el * el
                if ci_low <= scenario.true_mean <= ci_high:
                    covered[meth] += 1
                total_w[meth] += ci_high - ci_low

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
            run_noise_frac=scenario.run_noise_frac,
            runs=runs,
        )
        for method in active_methods
    ]


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pairwise multi-run worker
# ---------------------------------------------------------------------------

_WORKER_PAIR_SCENARIOS: list = []


def _run_pairwise_multirun_cell(args: tuple) -> list[SimResult]:
    """Run all reps for one (pair scenario, n, runs) cell — pairwise mean-diff estimand."""
    sc_idx, n, runs, n_reps, n_bootstrap, bayes_n, alpha, seed = args
    scenario = _WORKER_PAIR_SCENARIOS[sc_idx]
    rng = np.random.default_rng(seed)

    active_methods = METHODS + [T_INTERVAL_METHOD] + PAIR_DIFF_NESTED_METHODS
    if scenario.eval_type == "binary":
        active_methods += BINARY_PAIR_FLAT_METHODS + BINARY_PAIR_NESTED_METHODS

    covered:    dict[str, int]   = {m: 0   for m in active_methods}
    total_w:    dict[str, float] = {m: 0.0 for m in active_methods}
    total_t:    dict[str, float] = {m: 0.0 for m in active_methods}
    total_t_sq: dict[str, float] = {m: 0.0 for m in active_methods}

    for _rep in range(n_reps):
        a, b          = scenario.generate_pair(rng, n, runs)    # (n, runs) each
        cell_means_a  = a.mean(axis=1)
        cell_means_b  = b.mean(axis=1)
        cell_diffs    = cell_means_a - cell_means_b             # (n,)
        obs_diff      = float(np.mean(cell_diffs))

        # ── Cell-mean diff bootstrap family ──────────────────────────────
        for method in METHODS:
            n_draws = bayes_n if method == "bayes_bootstrap" else n_bootstrap
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ci_low, ci_high = bootstrap_ci_1d(
                        cell_diffs, obs_diff, method=method,
                        n_bootstrap=n_draws, alpha=alpha, rng=rng,
                    )
            except Exception:
                ci_low = ci_high = obs_diff
            el = time.perf_counter() - t0
            total_t[method]    += el
            total_t_sq[method] += el * el
            if ci_low <= scenario.true_diff <= ci_high:
                covered[method] += 1
            total_w[method] += ci_high - ci_low

        # ── t-interval on cell-mean diffs ─────────────────────────────────
        t0 = time.perf_counter()
        try:
            ci_low, ci_high = t_interval_ci_1d(cell_diffs, alpha)
        except Exception:
            ci_low = ci_high = obs_diff
        el = time.perf_counter() - t0
        total_t[T_INTERVAL_METHOD]    += el
        total_t_sq[T_INTERVAL_METHOD] += el * el
        if ci_low <= scenario.true_diff <= ci_high:
            covered[T_INTERVAL_METHOD] += 1
        total_w[T_INTERVAL_METHOD] += ci_high - ci_low

        # ── Nested pairwise diff methods (full N×R pair matrices) ─────────
        for method, fn in [
            (BOOTSTRAP_DIFF_NESTED_METHOD, bootstrap_diffs_nested),
            (BAYES_DIFF_NESTED_METHOD,     bayes_bootstrap_diffs_nested),
            (SMOOTH_DIFF_NESTED_METHOD,    smooth_bootstrap_diffs_nested),
        ]:
            n_draws = bayes_n if method == BAYES_DIFF_NESTED_METHOD else n_bootstrap
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*falling back to plain bootstrap; no KDE smoothing applied.*",
                        category=UserWarning,
                    )
                    boot_stats = fn(a, b, n_draws, rng)
                ci_low  = float(np.percentile(boot_stats, 100 * alpha / 2))
                ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
            except Exception:
                ci_low = ci_high = obs_diff
            el = time.perf_counter() - t0
            total_t[method]    += el
            total_t_sq[method] += el * el
            if ci_low <= scenario.true_diff <= ci_high:
                covered[method] += 1
            total_w[method] += ci_high - ci_low

        # ── Binary pairwise methods ───────────────────────────────────────
        if scenario.eval_type == "binary":
            a0, b0 = a[:, 0], b[:, 0]   # first run only (flat iid baseline)

            t0 = time.perf_counter()
            try:
                ci_low, ci_high = tango_paired_ci(a0, b0, alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a0 - b0))
            el = time.perf_counter() - t0
            total_t[TANGO_FLAT_METHOD]    += el
            total_t_sq[TANGO_FLAT_METHOD] += el * el
            if ci_low <= scenario.true_diff <= ci_high:
                covered[TANGO_FLAT_METHOD] += 1
            total_w[TANGO_FLAT_METHOD] += ci_high - ci_low

            t0 = time.perf_counter()
            try:
                ci_low, ci_high = newcombe_paired_ci(a0, b0, alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a0 - b0))
            el = time.perf_counter() - t0
            total_t[NEWCOMBE_FLAT_METHOD]    += el
            total_t_sq[NEWCOMBE_FLAT_METHOD] += el * el
            if ci_low <= scenario.true_diff <= ci_high:
                covered[NEWCOMBE_FLAT_METHOD] += 1
            total_w[NEWCOMBE_FLAT_METHOD] += ci_high - ci_low

            for method, fn in [
                (TANGO_MULTIRUN_METHOD, tango_paired_ci_multirun_discordance),
                (TANGO_MULTIRUN_MOMENTS_METHOD, tango_paired_ci_multirun_moments),
            ]:
                t0 = time.perf_counter()
                try:
                    ci_low, ci_high = fn(a, b, alpha)
                except Exception:
                    ci_low = ci_high = obs_diff
                el = time.perf_counter() - t0
                total_t[method]    += el
                total_t_sq[method] += el * el
                if ci_low <= scenario.true_diff <= ci_high:
                    covered[method] += 1
                total_w[method] += ci_high - ci_low

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
            run_noise_frac=scenario.run_noise_frac,
            runs=runs,
            is_null=scenario.is_null,
            icc=scenario.icc,
        )
        for method in active_methods
    ]


# ---------------------------------------------------------------------------
# Pairwise simulation runner


def run_multirun_simulation(
    scenarios: list[MultiRunScenario],
    sample_sizes: list[int],
    runs: int,
    n_reps: int,
    n_bootstrap: int,
    bayes_n: int,
    alpha: float,
    progress_mode: str = "bar",
    seed: int = 42,
    n_workers: int = 1,
) -> list[SimResult]:
    global _WORKER_SCENARIOS
    _WORKER_SCENARIOS = scenarios

    ss = np.random.SeedSequence(seed)
    idx_size_pairs = list(itertools.product(range(len(scenarios)), sample_sizes))
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(idx_size_pairs))]
    args_list = [
        (sc_idx, n, runs, n_reps, n_bootstrap, bayes_n, alpha, child_seeds[i])
        for i, (sc_idx, n) in enumerate(idx_size_pairs)
    ]
    total_cells = len(args_list)
    reporter = _ProgressReporter(total_cells, mode=progress_mode, label="multirun")
    results: list[SimResult] = []

    if n_workers == 1:
        for i, args in enumerate(args_list):
            results.extend(_run_multirun_cell(args))
            sc = scenarios[args[0]]
            reporter.update(i + 1, detail=f"{sc.eval_type} {sc.label} n={args[1]}")
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_multirun_cell, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1, detail=f"cells done: {i + 1}/{total_cells}")

    reporter.update(total_cells, detail="done")
    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def run_pairwise_multirun_simulation(
    scenarios: list[PairMultiRunScenario],
    sample_sizes: list[int],
    runs: int,
    n_reps: int,
    n_bootstrap: int,
    bayes_n: int,
    alpha: float,
    progress_mode: str = "bar",
    seed: int = 42,
    n_workers: int = 1,
) -> list[SimResult]:
    global _WORKER_PAIR_SCENARIOS
    _WORKER_PAIR_SCENARIOS = scenarios

    ss = np.random.SeedSequence(seed)
    idx_size_pairs = list(itertools.product(range(len(scenarios)), sample_sizes))
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(idx_size_pairs))]
    args_list = [
        (sc_idx, n, runs, n_reps, n_bootstrap, bayes_n, alpha, child_seeds[i])
        for i, (sc_idx, n) in enumerate(idx_size_pairs)
    ]
    total_cells = len(args_list)
    reporter = _ProgressReporter(total_cells, mode=progress_mode, label="pairwise-multirun")
    results: list[SimResult] = []

    if n_workers == 1:
        for i, args in enumerate(args_list):
            results.extend(_run_pairwise_multirun_cell(args))
            sc = scenarios[args[0]]
            reporter.update(i + 1, detail=f"{sc.eval_type} {sc.label} n={args[1]}")
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(
                pool.imap_unordered(_run_pairwise_multirun_cell, args_list)
            ):
                results.extend(cell_results)
                reporter.update(i + 1, detail=f"cells done: {i + 1}/{total_cells}")

    reporter.update(total_cells, detail="done")
    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _cov_marker(cov: float, target: float, tol: float = 0.04) -> str:
    if cov < target - tol:
        return "▼"
    if cov > target + tol:
        return "▲"
    return " "


def _mc_proportion_stats(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    if total <= 0:
        return (float("nan"),) * 4
    p_hat = successes / total
    mcse  = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    lo, hi = max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)
    return float(p_hat), mcse, float(lo), float(hi)


def _rule(width: int, char: str = "─") -> str:
    return char * width


def _time_stats(subset: list[SimResult]) -> tuple[float, float]:
    total_reps = sum(r.n_reps for r in subset)
    if total_reps <= 0:
        return float("nan"), float("nan")
    sum_t  = sum(r.total_time    for r in subset)
    sum_t2 = sum(r.total_time_sq for r in subset)
    avg = sum_t / total_reps
    var = max(0.0, sum_t2 / total_reps - avg * avg)
    se  = float(np.sqrt(var / total_reps))
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
    method_order: list[str] | None = None,
    report_title: str = "MULTI-RUN CI COMPARISON  —  SIMULATION RESULTS",
) -> None:
    target = 1.0 - alpha
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    n_labels = [f"n={n}" for n in sample_sizes]
    present_methods = {r.method for r in results}
    order = method_order if method_order is not None else REPORT_METHODS
    method_labels   = [m for m in order if m in present_methods]

    agg:        dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    agg_counts: dict[tuple, tuple[int, int]]           = defaultdict(lambda: (0, 0))
    per_sc:     dict[tuple, tuple[float, float]]       = {}

    for r in results:
        cov   = r.covered / r.n_reps
        width = r.total_width / r.n_reps
        agg[(r.eval_type, r.method, r.n)].append((cov, width))
        c_p, t_p = agg_counts[(r.eval_type, r.method, r.n)]
        agg_counts[(r.eval_type, r.method, r.n)] = (c_p + r.covered, t_p + r.n_reps)
        per_sc[(r.eval_type, r.scenario, r.method, r.n)] = (cov, width)

    def mean_cov(et: str, m: str, n: int) -> float:
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[0] for v in vals])) if vals else float("nan")

    def mean_wid(et: str, m: str, n: int) -> float:
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[1] for v in vals])) if vals else float("nan")

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {report_title}")
    print(f"  Estimand: {estimand_label}")
    print(f"  Nominal coverage: {target:.0%}   |   reps/scenario: {n_reps}")
    print(f"  ▼ = under-covered (<{target - 0.04:.0%})   ▲ = over-conservative (>{target + 0.04:.0%})")
    print("  Coverage averaged across all scenarios including run_noise_frac sweep.")
    print(sep)

    for et in eval_types_present:
        print(f"\n{'─'*72}")
        print(f"  {et.upper()}")
        print(f"{'─'*72}")

        cov_cells: dict[tuple[str, str], str] = {}
        for m in method_labels:
            for i, n in enumerate(sample_sizes):
                cov = mean_cov(et, m, n)
                if np.isnan(cov):
                    cov_cells[(m, n_labels[i])] = "─"
                else:
                    cov_cells[(m, n_labels[i])] = f"{cov:.3f}{_cov_marker(cov, target)}"

        _print_grid(
            f"Coverage (target {target:.2f}) — averaged across f_run and shapes",
            row_labels=method_labels, col_labels=n_labels, cells=cov_cells,
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
            row_labels=method_labels, col_labels=n_labels,
            cells=band_cells, col_w=13,
        )

        wid_cells: dict[tuple[str, str], str] = {}
        for m in method_labels:
            for i, n in enumerate(sample_sizes):
                w = mean_wid(et, m, n)
                wid_cells[(m, n_labels[i])] = "─" if np.isnan(w) else f"{w:.4f}"

        _print_grid(
            "Mean CI Width", row_labels=method_labels,
            col_labels=n_labels, cells=wid_cells,
        )

    # ── Coverage by run_noise_frac (new section) ──────────────────────────
    run_noise_fracs = sorted({r.run_noise_frac for r in results})
    if len(run_noise_fracs) > 1:
        print(f"\n{'─'*72}")
        print("  COVERAGE BY RUN NOISE FRACTION  (averaged across shapes and N)")
        print(f"  Rows = methods.  Columns = f_run = σ²_run / σ²_total.")
        print(f"  Flat methods expected to under-cover at low f_run (high ICC).")
        print(f"{'─'*72}")

        for et in eval_types_present:
            print(f"\n  {et}")
            f_labels = [f"f={f:.2f}" for f in run_noise_fracs]
            hdr = f"    {'Method':<25}" + "".join(f"  {fl:>9}" for fl in f_labels)
            print(hdr)
            print(f"    {'─'*25}" + "─" * (11 * len(run_noise_fracs)))

            for m in method_labels:
                row = f"    {m:<25}"
                for f in run_noise_fracs:
                    subset = [r for r in results if r.eval_type == et and r.method == m and r.run_noise_frac == f]
                    if not subset:
                        row += f"  {'─':>9}"
                        continue
                    c_tot = sum(r.covered for r in subset)
                    t_tot = sum(r.n_reps  for r in subset)
                    cov   = c_tot / t_tot if t_tot > 0 else float("nan")
                    cell  = "─" if np.isnan(cov) else f"{cov:.3f}{_cov_marker(cov, target)}"
                    row  += f"  {cell:>9}"
                print(row)

    # ── Worst-coverage scenarios ───────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  WORST COVERAGE CASES  (averaged across included methods)")
    print(f"{'─'*72}")

    sc_cov: dict[tuple, list[float]] = defaultdict(list)
    for (et, sc, m, n), (cov, _) in per_sc.items():
        sc_cov[(et, sc, n)].append(cov)

    worst = sorted(
        [(float(np.mean(v)), k) for k, v in sc_cov.items()],
        key=lambda x: x[0],
    )[:12]

    print(f"\n  {'Eval Type':<12}  {'Scenario':<36}  {'n':>4}  {'Avg Coverage':>13}")
    print(f"  {'─'*12}  {'─'*36}  {'─'*4}  {'─'*13}")
    for cov, (et, sc, n) in worst:
        mark = _cov_marker(cov, target)
        print(f"  {et:<12}  {sc:<36}  {n:>4}  {cov:>12.3f}{mark}")

    # ── Coverage deviation from nominal ───────────────────────────────────
    print(f"\n{'─'*72}")
    print("  COVERAGE DEVIATION FROM NOMINAL  (mean_coverage − target)")
    print(f"{'─'*72}")

    for et in eval_types_present:
        print(f"\n  {et}")
        hdr = f"    {'Method':<25}" + "".join(f"  {nl:>9}" for nl in n_labels)
        print(hdr)
        print(f"    {'─'*25}" + "─" * (11 * len(n_labels)))
        for m in method_labels:
            row = f"    {m:<25}"
            for n in sample_sizes:
                dev = mean_cov(et, m, n) - target
                row += "  {:>9}".format("─") if np.isnan(dev) else f"  {dev:>+9.3f}"
            print(row)

    # ── Overall summary ────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  OVERALL SUMMARY  (averaged across all eval types, scenarios, n, f_run)")
    print(f"{'─'*72}")

    all_cov:    dict[str, list[float]] = defaultdict(list)
    all_wid:    dict[str, list[float]] = defaultdict(list)
    all_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for (et, m, n), vals in agg.items():
        all_cov[m].extend(v[0] for v in vals)
        all_wid[m].extend(v[1] for v in vals)
        c, t = agg_counts[(et, m, n)]
        c_p, t_p = all_counts[m]
        all_counts[m] = (c_p + c, t_p + t)

    print(f"\n  {'Method':<25}  {'Cov':>6}  {'MCSE':>7}  {'Band95':>13}  {'Width':>8}  {'Dev':>8}  {'Time(ms)':>14}")
    print(f"  {'─'*25}  {'─'*6}  {'─'*7}  {'─'*13}  {'─'*8}  {'─'*8}  {'─'*14}")
    for m in method_labels:
        mc   = float(np.mean(all_cov[m]))
        mw   = float(np.mean(all_wid[m]))
        dev  = mc - target
        mark = _cov_marker(mc, target)
        c_tot, t_tot = all_counts[m]
        _, mcse, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        avg_ms, se_ms = _time_stats([r for r in results if r.method == m])
        time_str = f"{avg_ms:.3f}±{se_ms:.3f}" if np.isfinite(avg_ms) else "─"
        print(
            f"  {m:<25}  {mc:>5.3f}{mark}  {mcse:>7.4f}  {f'{lo:.3f}-{hi:.3f}':>13}  {mw:>8.4f}  {dev:>+8.3f}  {time_str:>14}"
        )
    print()


def print_pairwise_report(
    results: list[SimResult],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
    estimand_label: str,
) -> None:
    print_report(
        results=results,
        sample_sizes=sample_sizes,
        alpha=alpha,
        n_reps=n_reps,
        estimand_label=estimand_label,
        method_order=REPORT_PAIRWISE_METHODS,
        report_title="MULTI-RUN PAIRWISE CI COMPARISON  —  SIMULATION RESULTS",
    )


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

_METHOD_COLORS: dict[str, str] = {
    "bootstrap":                "#1f77b4",
    "bca":                      "#2ca02c",
    "bayes_bootstrap":          "#ff7f0e",
    "smooth_bootstrap":         "#9467bd",
    "bootstrap_t":              "#d62728",
    "t_interval":               "#8c564b",
    "bootstrap_nested":         "#17becf",
    "bayes_bootstrap_nested":   "#bcbd22",
    "smooth_bootstrap_nested":  "#e377c2",
    "bca_nested":               "#ff9896",
    "bootstrap_t_nested":       "#c49c94",
    "wilson_de":                "#dbdb8d",
    "wilson_od":                "#9edae5",
    "beta_binomial":            "#f7b6d2",
    "t_interval_flat":          "#aec7e8",
    "bootstrap_flat":           "#ffbb78",
    "wilson_flat":              "#98df8a",
    "wald_flat":                "#7f7f7f",
    "clopper_pearson_flat":     "#c5b0d5",
    "bayes_indep_flat":         "#f7b6d2",
    "beta":                     "#f0027f",
    "logit_t":                  "#a6761d",
    "nig":                      "#888888",
    "nig_nested":               "#444444",
    "el":                       "#00441b",
    "bootstrap_diff_nested":    "#1b9e77",
    "bayes_diff_nested":        "#d95f02",
    "smooth_diff_nested":       "#7570b3",
    "tango_flat":               "#e7298a",
    "newcombe_flat":            "#66a61e",
    "tango_multirun_disc":      "#e6ab02",
    "tango_multirun_mmnt":      "#1b9e77",
}


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
    """Box plots of coverage and CI width per method per eval type."""
    plot_results = results
    if sample_size_filter is not None:
        plot_results = [r for r in results if r.n == sample_size_filter]
    if not plot_results:
        print(f"Skipped plot (no matching data): {out_path}")
        return

    target         = 1.0 - alpha
    present_methods = {r.method for r in plot_results}
    method_labels   = [m for m in REPORT_ALL_METHODS if m in present_methods]

    fig, axes = plt.subplots(
        nrows=len(EVAL_TYPES), ncols=2,
        figsize=(14.8, 3.0 * len(EVAL_TYPES)),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.34, "hspace": 0.40},
    )
    if len(EVAL_TYPES) == 1:
        axes = np.array([axes])

    box_kwargs: dict[str, Any] = {
        "vert": False, "showmeans": True, "meanline": False,
        "patch_artist": False,
        "whiskerprops": {"linewidth": 1.2, "color": "black"},
        "capprops":     {"linewidth": 1.2, "color": "black"},
        "medianprops":  {"linewidth": 1.8, "color": "black"},
        "boxprops":     {"linewidth": 1.4, "color": "black"},
        "meanprops":    {"marker": "D", "markerfacecolor": "white",
                         "markeredgecolor": "black", "markersize": 4.5},
        "flierprops":   {"marker": "o", "markerfacecolor": "black",
                         "markeredgecolor": "black", "markersize": 2.8, "alpha": 0.55},
    }

    for r_idx, et in enumerate(EVAL_TYPES):
        et_results = [res for res in plot_results if res.eval_type == et]
        et_methods = [m for m in method_labels if any(res.method == m for res in et_results)]

        cov_series: list[np.ndarray] = []
        wid_series: list[np.ndarray] = []
        cov_uncertainty: list[tuple[float, float, float]] = []

        for method in et_methods:
            subset   = [res for res in et_results if res.method == method]
            cov_vals = np.array([res.covered / res.n_reps for res in subset], dtype=float)
            wid_vals = np.array([res.total_width / res.n_reps for res in subset], dtype=float)
            if cov_vals.size > 0:
                cov_series.append(cov_vals)
                wid_series.append(wid_vals)
                c_tot = int(sum(res.covered for res in subset))
                t_tot = int(sum(res.n_reps  for res in subset))
                p_hat, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
                cov_uncertainty.append((p_hat, lo, hi))

        for c_idx, (ax, series) in enumerate(zip(axes[r_idx], [cov_series, wid_series])):
            if r_idx == 0:
                ax.set_title(
                    ["Coverage (target; red = MC95)", "Mean CI Width"][c_idx], fontsize=10,
                )
            if not series:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                ax.set_yticks([])
                continue

            bp = ax.boxplot(series, tick_labels=et_methods, **box_kwargs)
            ax.grid(axis="x", linestyle="--", linewidth=0.65, alpha=0.50)
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=8.5, pad=2)
            ax.tick_params(axis="x", labelsize=8.5)

            if c_idx == 0:
                ax.axvspan(max(0.0, target - 0.04), min(1.0, target + 0.04),
                           color="#DDDDDD", alpha=0.35, zorder=0)
                ax.axvline(target, color="black", linestyle="-", linewidth=1.2)
                ax.set_xlim(0.0, 1.0)
                for y_pos, (p_hat, lo, hi) in enumerate(cov_uncertainty, start=1):
                    if np.isnan(lo) or np.isnan(hi):
                        continue
                    ax.hlines(y=y_pos, xmin=lo, xmax=hi, color="tab:red", linewidth=2.1, zorder=5)
                    ax.vlines([lo, hi], y_pos - 0.15, y_pos + 0.15, color="tab:red", linewidth=1.5, zorder=5)
                ax.set_ylabel(et.upper(), fontsize=10)
            else:
                x_max = max((float(np.max(v)) for v in series if v.size > 0), default=1.0)
                ax.set_xlim(0.0, x_max * 1.08 if x_max > 0 else 1.0)

    if sample_size_filter is not None:
        size_text = str(sample_size_filter)
    elif sample_sizes:
        size_text = ", ".join(str(n) for n in sample_sizes)
    else:
        unique_sizes = sorted({r.n for r in plot_results})
        size_text = ", ".join(str(n) for n in unique_sizes) if unique_sizes else "n/a"

    fig.suptitle(
        f"Multi-run CI Comparison (Box Plots)\n"
        f"{estimand_label} | alpha={alpha} | n={size_text}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=[0.03, 0.02, 0.995, 0.95], w_pad=2.6)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out}")


def save_coverage_vs_run_noise_plot(
    *,
    results: list[SimResult],
    alpha: float,
    n_reps: int,
    estimand_label: str,
    out_path: str,
) -> None:
    """Coverage vs. run_noise_frac — key plot for understanding when nested/flat matters."""
    if not results:
        print(f"Skipped coverage-vs-run-noise plot (no data): {out_path}")
        return

    target          = 1.0 - alpha
    present_methods = {r.method for r in results}
    method_labels   = [m for m in REPORT_ALL_METHODS if m in present_methods]
    run_noise_fracs = sorted({r.run_noise_frac for r in results})
    if len(run_noise_fracs) < 2:
        print(f"Skipped coverage-vs-run-noise plot (only one f_run value): {out_path}")
        return

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    nrows = len(eval_types_present)
    fig, axes = plt.subplots(nrows, 1, figsize=(10.0, 4.0 * nrows),
                             squeeze=False,
                             gridspec_kw={"hspace": 0.45})

    for row_idx, et in enumerate(eval_types_present):
        ax = axes[row_idx][0]
        ax.axhspan(max(0.0, target - 0.04), min(1.0, target + 0.04),
                   color="#DDDDDD", alpha=0.40, zorder=0)
        ax.axhline(target, color="black", linewidth=1.2, linestyle="--", zorder=1)

        for method in method_labels:
            color = _METHOD_COLORS.get(method, "#333333")
            covs  = []
            for f in run_noise_fracs:
                subset = [r for r in results
                          if r.eval_type == et and r.method == method and r.run_noise_frac == f]
                if not subset:
                    covs.append(float("nan"))
                    continue
                c_tot = sum(r.covered for r in subset)
                t_tot = sum(r.n_reps  for r in subset)
                covs.append(c_tot / t_tot if t_tot > 0 else float("nan"))

            xs = [f for f, c in zip(run_noise_fracs, covs) if not np.isnan(c)]
            ys = [c for c in covs if not np.isnan(c)]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", color=color, linewidth=1.4,
                    label=method, markersize=5, alpha=0.85)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(max(0.0, target - 0.25), min(1.01, target + 0.12))
        ax.set_xlabel("Run noise fraction  f_run = σ²_run / (σ²_input + σ²_run)", fontsize=9.5)
        ax.set_ylabel("Empirical coverage", fontsize=9.5)
        ax.set_title(f"eval type: {et}", fontsize=10.5)
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
        ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.35)
        ax.legend(
            fontsize=7.5, ncol=1,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            framealpha=0.85,
        )

    runs_val = results[0].runs if results else "?"
    fig.suptitle(
        f"Coverage vs. Run Noise Fraction\n"
        f"{estimand_label}  |  runs={runs_val}  |  alpha={alpha}  |  reps={n_reps}\n"
        f"Averaged across all shapes and sample sizes",
        fontsize=11,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=[0.02, 0.02, 0.80, 0.93])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved coverage-vs-run-noise plot: {out}")


def save_coverage_vs_n_plot(
    *,
    results: list[SimResult],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
    estimand_label: str,
    out_path: str,
) -> None:
    """Coverage vs. sample size — averaged across f_run and shapes."""
    if not results:
        print(f"Skipped coverage-vs-n plot (no data): {out_path}")
        return

    target         = 1.0 - alpha
    present_methods = {r.method for r in results}
    method_labels   = [m for m in REPORT_ALL_METHODS if m in present_methods]

    rows = [
        {"eval_type": r.eval_type, "scenario": r.scenario,
         "method": r.method, "n": r.n, "coverage": r.covered / r.n_reps}
        for r in results
    ]
    df = pd.DataFrame(rows)
    df = df[df["method"].isin(method_labels)]

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
    eval_types_present = [et for et in EVAL_TYPES if et in df["eval_type"].values]
    fig, axes = plt.subplots(
        1, len(eval_types_present),
        figsize=(5.5 * len(eval_types_present), 5),
        squeeze=False,
    )

    for col_idx, et in enumerate(eval_types_present):
        ax     = axes[0][col_idx]
        et_agg = agg[agg["eval_type"] == et].copy()
        et_methods = [m for m in method_labels if m in et_agg["method"].values]

        if et_agg.empty:
            ax.set_title(et.upper())
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        sns.lineplot(
            data=et_agg, x="n", y="coverage_mean", hue="method",
            hue_order=et_methods, palette=palette,
            marker=None, linewidth=1.0, alpha=0.70, ax=ax,
        )
        for method, sub in et_agg.groupby("method"):
            if sub["coverage_std"].isna().all():
                continue
            sub = sub.sort_values("n")
            color = _METHOD_COLORS.get(str(method), "#333333")
            se = sub["coverage_std"] / np.sqrt(sub["coverage_count"])
            ax.errorbar(sub["n"], sub["coverage_mean"], yerr=se, fmt="none",
                        color=color, elinewidth=0.8, capsize=2, alpha=0.45)
            ax.scatter(sub["n"], sub["coverage_mean"], s=28, color=color,
                       edgecolors="white", linewidths=0.6, alpha=0.85, zorder=3)

        ns = sorted(et_agg["n"].unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.axhline(target, linestyle="--", color="tab:cyan", linewidth=1.2)
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Empirical coverage" if col_idx == 0 else "")
        ax.set_title(et.upper())
        if et_methods:
            ax.legend(title="Method", fontsize=7, title_fontsize=8)

    fig.suptitle(
        f"Coverage vs. Sample Size  (averaged over f_run and shapes)\n"
        f"{estimand_label} | alpha={alpha}",
        fontsize=11,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved coverage-vs-n plot: {out}")


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
    method_labels = [m for m in REPORT_ALL_METHODS if m in present_methods]
    sample_sizes = sorted({r.n for r in results})
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    if not eval_types_present:
        print(f"Skipped cost plot (no eval types present): {out_path}")
        return

    fig, axes = plt.subplots(
        nrows=len(eval_types_present),
        ncols=1,
        figsize=(11.0, 4.5 * len(eval_types_present)),
        squeeze=False,
        gridspec_kw={"hspace": 0.45},
    )

    def _label_indices(ns: list[int]) -> set[int]:
        if len(ns) <= 2:
            return set(range(len(ns)))
        return {0, len(ns) // 2, len(ns) - 1}

    for row_idx, et in enumerate(eval_types_present):
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

            points: list[tuple[int, float, float, float]] = []
            for n in sample_sizes:
                subset = [r for r in m_results if r.n == n]
                if not subset:
                    continue
                avg_ms, se_ms = _time_stats(subset)
                if not np.isfinite(avg_ms) or avg_ms < 0:
                    continue
                avg_ms = max(avg_ms, 1e-4)  # floor for log scale (very fast analytical methods)
                cov = float(np.mean([r.covered / r.n_reps for r in subset]))
                points.append((n, avg_ms, cov, 1.96 * se_ms))

            if not points:
                continue

            xs = [p[1] for p in points]
            ys = [p[2] for p in points]

            ax.plot(xs, ys, color=color, linewidth=1.1, alpha=0.55, zorder=2)
            ax.errorbar(
                xs, ys,
                xerr=[p[3] for p in points],
                fmt="o", color=color,
                markersize=6, markeredgewidth=0.7, markeredgecolor="white",
                elinewidth=0.9, capsize=2.5, capthick=0.9,
                alpha=0.90, zorder=3,
            )

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


def save_results_artifacts(
    *,
    results: list[SimResult],
    alpha: float,
    sample_sizes: list[int],
    n_reps: int,
    estimand_label: str,
    out_dir: str,
    run_stem: str,
    report_fn: Callable[[list[SimResult], list[int], float, int, str], None] = print_report,
) -> None:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    csv_path = out_base / f"{run_stem}_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "eval_type", "scenario", "n", "method", "n_reps", "covered",
            "total_width", "coverage", "mean_width", "mcse", "band95_low", "band95_high",
            "avg_time_ms", "se_time_ms", "run_noise_frac", "runs",
        ])
        for r in results:
            coverage   = r.covered / r.n_reps
            mean_width = r.total_width / r.n_reps
            _, mcse, lo, hi = _mc_proportion_stats(r.covered, r.n_reps)
            avg_ms, se_ms   = _time_stats([r])
            writer.writerow([
                r.eval_type, r.scenario, r.n, r.method, r.n_reps, r.covered,
                f"{r.total_width:.8f}", f"{coverage:.8f}", f"{mean_width:.8f}",
                f"{mcse:.8f}", f"{lo:.8f}", f"{hi:.8f}",
                f"{avg_ms:.6f}" if np.isfinite(avg_ms) else "",
                f"{se_ms:.6f}"  if np.isfinite(se_ms)  else "",
                f"{r.run_noise_frac:.4f}", str(r.runs),
            ])

    summary_path = out_base / f"{run_stem}_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        report_fn(
            results=results, sample_sizes=sample_sizes,
            alpha=alpha, n_reps=n_reps, estimand_label=estimand_label,
        )
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log:     {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _run_benchmark(
    *,
    runs: int,
    reps: int,
    bootstrap_n: int,
    bayes_n: int,
    alpha: float,
    sizes: list[int],
    seed: int,
    scenario_suite: str,
    run_noise_fracs: list[float],
    progress_mode: str,
    plot_mode: str,
    save_results: str,
    out_dir: str,
    plots_dir: str,
    eval_types: list[str] | None = None,
    label: str | None = None,
    n_workers: int = 1,
    heteroscedastic: bool = False,
) -> list[SimResult]:
    if label:
        print(f"\n{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}")

    print(f"\nMulti-run CI Simulation")
    print(f"  Scenario suite   : {scenario_suite}")
    if eval_types:
        print(f"  Eval types       : {eval_types}")
    else:
        print(f"  Eval types       : all ({EVAL_TYPES})")
    print(f"  Runs per input   : {runs}")
    print(f"  Run noise fracs  : {run_noise_fracs}")
    print(f"  Reps per cell    : {reps}")
    print(f"  Bootstrap draws  : {bootstrap_n}")
    print(f"  Bayes draws      : {bayes_n}")
    print(f"  Alpha / CI level : {alpha} / {(1 - alpha):.0%}")
    print(f"  Sample sizes     : {sizes}")
    print(f"  Seed             : {seed}")
    print(f"  Workers          : {n_workers}")
    print(f"  Progress mode    : {progress_mode}")
    print(f"  Plots            : {plot_mode}")
    print(f"  Save results     : {save_results}")
    print(f"  Out dir          : {out_dir}")

    print("\nBuilding scenarios …", end="", flush=True)
    scenarios = build_multirun_scenarios(run_noise_fracs, suite=scenario_suite, heteroscedastic=heteroscedastic)

    if eval_types:
        requested = set(eval_types)
        scenarios = [s for s in scenarios if s.eval_type in requested]
        if not scenarios:
            raise ValueError(f"No scenarios left after filtering eval types {sorted(requested)}.")

    n_by_type = {et: sum(1 for s in scenarios if s.eval_type == et) for et in EVAL_TYPES}
    print(
        f"  {len(scenarios)} total  "
        + "  ".join(f"{et}: {n_by_type[et]}" for et in EVAL_TYPES)
    )

    cells = len(scenarios) * len(sizes)
    print(f"\nRunning {cells} cells × {reps} reps × {len(METHODS)+len(NESTED_METHODS)+len(FLAT_METHODS)} core methods …")

    results = run_multirun_simulation(
        scenarios=scenarios,
        sample_sizes=sizes,
        runs=runs,
        n_reps=reps,
        n_bootstrap=bootstrap_n,
        bayes_n=bayes_n,
        alpha=alpha,
        progress_mode=progress_mode,
        seed=seed,
        n_workers=n_workers,
    )

    estimand_label = f"single-sample grand mean (runs={runs}){' [hetero]' if heteroscedastic else ''}"

    print_report(
        results,
        sample_sizes=sizes,
        alpha=alpha,
        n_reps=reps,
        estimand_label=estimand_label,
    )

    stamp    = time.strftime("%Y%m%d_%H%M%S")
    run_stem = (
        f"sim_nested_runs{runs}_{scenario_suite}_"
        f"alpha{alpha:.3f}_reps{reps}_{stamp}"
    )

    if save_results == "save":
        save_results_artifacts(
            results=results, alpha=alpha, sample_sizes=sizes,
            n_reps=reps, estimand_label=estimand_label,
            out_dir=out_dir, run_stem=run_stem,
        )

    if plot_mode == "save":
        pdir = Path(plots_dir)

        save_coverage_vs_run_noise_plot(
            results=results, alpha=alpha, n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(pdir / f"{run_stem}_coverage_vs_run_noise.png"),
        )

        save_metric_plot(
            results=results, sample_sizes=sizes, alpha=alpha, n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(pdir / f"{run_stem}_overall.png"),
        )

        for n in sizes:
            save_metric_plot(
                results=results, sample_sizes=[n], alpha=alpha, n_reps=reps,
                estimand_label=estimand_label,
                out_path=str(pdir / f"{run_stem}_n{n}.png"),
                sample_size_filter=n,
            )

        save_coverage_vs_n_plot(
            results=results, sample_sizes=sizes, alpha=alpha, n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(pdir / f"{run_stem}_coverage_vs_n.png"),
        )

        save_cost_plot(
            results=results,
            alpha=alpha,
            n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(pdir / f"{run_stem}_cost_coverage.png"),
        )

    return results


def _run_benchmark_pairwise(
    *,
    runs: int,
    reps: int,
    bootstrap_n: int,
    bayes_n: int,
    alpha: float,
    sizes: list[int],
    seed: int,
    scenario_suite: str,
    run_noise_fracs: list[float],
    cohens_d_values: list[float],
    include_null: bool,
    progress_mode: str,
    plot_mode: str,
    save_results: str,
    out_dir: str,
    plots_dir: str,
    eval_types: list[str] | None = None,
    label: str | None = None,
    n_workers: int = 1,
    heteroscedastic: bool = False,
) -> list[SimResult]:
    if label:
        print(f"\n{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}")

    print(f"\nPairwise Multi-run CI Simulation")
    print(f"  Scenario suite   : {scenario_suite}")
    if eval_types:
        print(f"  Eval types       : {eval_types}")
    else:
        print(f"  Eval types       : all ({EVAL_TYPES})")
    print(f"  Runs per input   : {runs}")
    print(f"  Run noise fracs  : {run_noise_fracs}")
    print(f"  Cohen's d values : {cohens_d_values}")
    print(f"  Include null     : {include_null}")
    print(f"  Reps per cell    : {reps}")
    print(f"  Bootstrap draws  : {bootstrap_n}")
    print(f"  Bayes draws      : {bayes_n}")
    print(f"  Alpha / CI level : {alpha} / {(1 - alpha):.0%}")
    print(f"  Sample sizes     : {sizes}")
    print(f"  Seed             : {seed}")
    print(f"  Workers          : {n_workers}")
    print(f"  Progress mode    : {progress_mode}")
    print(f"  Plots            : {plot_mode}")
    print(f"  Save results     : {save_results}")
    print(f"  Out dir          : {out_dir}")

    print("\nBuilding pairwise scenarios …", end="", flush=True)
    scenarios = build_pair_multirun_scenarios(
        run_noise_fracs,
        suite=scenario_suite,
        cohens_d_values=cohens_d_values,
        include_null=include_null,
        heteroscedastic=heteroscedastic,
    )

    if eval_types:
        requested = set(eval_types)
        scenarios = [s for s in scenarios if s.eval_type in requested]
        if not scenarios:
            raise ValueError(f"No pairwise scenarios left after filtering eval types {sorted(requested)}.")

    n_by_type = {et: sum(1 for s in scenarios if s.eval_type == et) for et in EVAL_TYPES}
    print(
        f"  {len(scenarios)} total  "
        + "  ".join(f"{et}: {n_by_type[et]}" for et in EVAL_TYPES)
    )

    binary_extra = len(BINARY_PAIR_FLAT_METHODS) + len(BINARY_PAIR_NESTED_METHODS)
    print(
        f"\nRunning {len(scenarios) * len(sizes)} cells × {reps} reps × "
        f"{len(METHODS) + 1 + len(PAIR_DIFF_NESTED_METHODS)}+binary({binary_extra}) methods …"
    )

    results = run_pairwise_multirun_simulation(
        scenarios=scenarios,
        sample_sizes=sizes,
        runs=runs,
        n_reps=reps,
        n_bootstrap=bootstrap_n,
        bayes_n=bayes_n,
        alpha=alpha,
        progress_mode=progress_mode,
        seed=seed,
        n_workers=n_workers,
    )

    estimand_label = f"pairwise difference E[cell_mean(A) - cell_mean(B)] (runs={runs}){' [hetero]' if heteroscedastic else ''}"

    print_pairwise_report(
        results,
        sample_sizes=sizes,
        alpha=alpha,
        n_reps=reps,
        estimand_label=estimand_label,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_stem = (
        f"sim_nested_pairwise_runs{runs}_{scenario_suite}_"
        f"alpha{alpha:.3f}_reps{reps}_{stamp}"
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
            report_fn=print_pairwise_report,
        )

    if plot_mode == "save":
        pdir = Path(plots_dir)

        save_coverage_vs_run_noise_plot(
            results=results,
            alpha=alpha,
            n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(pdir / f"{run_stem}_coverage_vs_run_noise.png"),
        )

        save_metric_plot(
            results=results,
            sample_sizes=sizes,
            alpha=alpha,
            n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(pdir / f"{run_stem}_overall.png"),
        )

        for n in sizes:
            save_metric_plot(
                results=results,
                sample_sizes=[n],
                alpha=alpha,
                n_reps=reps,
                estimand_label=estimand_label,
                out_path=str(pdir / f"{run_stem}_n{n}.png"),
                sample_size_filter=n,
            )

        save_coverage_vs_n_plot(
            results=results,
            sample_sizes=sizes,
            alpha=alpha,
            n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(pdir / f"{run_stem}_coverage_vs_n.png"),
        )

        save_cost_plot(
            results=results,
            alpha=alpha,
            n_reps=reps,
            estimand_label=estimand_label,
            out_path=str(pdir / f"{run_stem}_cost_coverage.png"),
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario-suite", choices=SCENARIO_SUITES, default="standard",
                        help="Scenario breadth (default: standard)")
    parser.add_argument("--eval-types", nargs="+", choices=EVAL_TYPES, default=None,
                        metavar="TYPE",
                        help=f"Restrict to specific eval types. Choices: {' '.join(EVAL_TYPES)}")
    parser.add_argument("--runs", type=int, default=3, metavar="R",
                        help="Runs per input when not sweeping (default: 3)")
    parser.add_argument("--runs-sweep", type=int, nargs="+", default=None, metavar="R",
                        help=(
                            "Sweep over multiple R values, e.g. --runs-sweep 1 2 3 5 10. "
                            "Overrides --runs; all results are combined and labelled by R."
                        ))
    parser.add_argument("--run-noise-fracs", type=float, nargs="+",
                        default=RUN_NOISE_FRACS_DEFAULT, metavar="F",
                        help=(
                            "Run-noise fractions to sweep: f = σ²_run / (σ²_input + σ²_run). "
                            f"Default: {RUN_NOISE_FRACS_DEFAULT}"
                        ))
    parser.add_argument("--reps", type=int, default=200, metavar="N",
                        help="MC repetitions per (scenario, n) cell (default: 200)")
    parser.add_argument("--bootstrap-n", type=int, default=500, metavar="N",
                        help="Bootstrap replicates per CI estimate (default: 500)")
    parser.add_argument("--bayes-n", type=int, default=None, metavar="N",
                        help="Bayesian bootstrap replicates for bayes methods (default: --bootstrap-n)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 50], metavar="N",
                        help="Sample sizes N (default: 5 10 20 50)")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed (default: 42)")
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar",
                        help="Progress display mode (default: bar)")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save",
                        help="Post-run plotting (default: save)")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save",
                        help="Write CSV and summary log (default: save)")
    parser.add_argument("--out-dir", default="simulations/out",
                        help="Output directory for artifacts (default: simulations/out)")
    parser.add_argument("--plots-dir", default=None,
                        help="Directory for plots (default: <out-dir>/plots)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, metavar="N",
                        help=(
                            "Parallel worker processes (default: all CPUs "
                            f"= {os.cpu_count()})."
                        ))
    parser.add_argument("--estimand", choices=["mean", "pairwise", "both"], default="mean",
                        help="Which estimand simulation(s) to run (default: mean)")
    parser.add_argument("--cohens-d-values", type=float, nargs="+", default=[0.3], metavar="D",
                        help="Pairwise effect sizes (Cohen's d-style scaling, default: 0.3)")
    parser.add_argument("--icc-values", type=float, nargs="+", default=None, metavar="ICC",
                        help="Optional ICC sweep; converted to run-noise via f_run = 1 - ICC")
    parser.add_argument("--include-null", action="store_true",
                        help="Include null (d=0) pairwise scenarios")
    parser.add_argument("--heteroscedastic", action="store_true",
                        help="Use heteroscedastic run noise (sigma_run_i ∝ 2√(p_i(1−p_i)))"
                             " so mid-range inputs have higher within-run variance than"
                             " floor/ceiling inputs (mimics real LLM eval variability).")
    parser.add_argument("--official-test", action="store_true",
                        help="Run official large benchmark preset (overrides key sweep args)")
    args = parser.parse_args()

    if args.official_test:
        args.reps = 2000
        args.bootstrap_n = 10000
        args.bayes_n = 10000
        args.alpha = 0.05
        args.sizes = OFFICIAL_SIZES.copy()
        args.seed = 44
        args.scenario_suite = "expanded"
        args.icc_values = OFFICIAL_ICC_VALUES.copy()
        args.cohens_d_values = [0.2, 0.4]
        args.include_null = True
        args.progress = "bar"
        args.workers = 12
        args.runs_sweep = OFFICIAL_RUNS_SWEEP.copy()
        args.run_noise_fracs = OFFICIAL_RUN_NOISE_FRACS.copy()
        args.estimand = "both"
        args.heteroscedastic = True

    if args.bayes_n is None:
        args.bayes_n = args.bootstrap_n

    if args.icc_values:
        icc_as_run_noise = [float(np.clip(1.0 - icc, 0.0, 1.0)) for icc in args.icc_values]
        args.run_noise_fracs = sorted(set(args.run_noise_fracs + icc_as_run_noise))

    plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")

    runs_list = args.runs_sweep if args.runs_sweep else [args.runs]

    all_results: list[SimResult] = []
    for r_val in runs_list:
        if args.estimand in ("mean", "both"):
            label = f"Multi-run nested CI sim · runs={r_val} · alpha={args.alpha}" if len(runs_list) > 1 else None
            r_results = _run_benchmark(
                runs=r_val,
                reps=args.reps,
                bootstrap_n=args.bootstrap_n,
                bayes_n=args.bayes_n,
                alpha=args.alpha,
                sizes=args.sizes,
                seed=args.seed,
                scenario_suite=args.scenario_suite,
                run_noise_fracs=args.run_noise_fracs,
                progress_mode=args.progress,
                plot_mode=args.plots,
                save_results=args.save_results,
                out_dir=args.out_dir,
                plots_dir=plots_dir,
                eval_types=args.eval_types,
                label=label,
                n_workers=args.workers,
                heteroscedastic=args.heteroscedastic,
            )
            all_results.extend(r_results)

        if args.estimand in ("pairwise", "both"):
            label = f"Pairwise multi-run nested CI sim · runs={r_val} · alpha={args.alpha}" if len(runs_list) > 1 else None
            r_results = _run_benchmark_pairwise(
                runs=r_val,
                reps=args.reps,
                bootstrap_n=args.bootstrap_n,
                bayes_n=args.bayes_n,
                alpha=args.alpha,
                sizes=args.sizes,
                seed=args.seed,
                scenario_suite=args.scenario_suite,
                run_noise_fracs=args.run_noise_fracs,
                cohens_d_values=args.cohens_d_values,
                include_null=args.include_null,
                progress_mode=args.progress,
                plot_mode=args.plots,
                save_results=args.save_results,
                out_dir=args.out_dir,
                plots_dir=plots_dir,
                eval_types=args.eval_types,
                label=label,
                n_workers=args.workers,
                heteroscedastic=args.heteroscedastic,
            )
            all_results.extend(r_results)

    # If runs sweep was requested, produce a combined coverage-vs-run-noise plot
    # across all R values (each R value as a separate line family) for reference.
    if args.runs_sweep and len(runs_list) > 1 and args.plots == "save":
        print("\n(Combined across all R values — see per-R plots above for detail.)")


if __name__ == "__main__":
    main()
