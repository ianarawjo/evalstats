#!/usr/bin/env python3
"""
sim_alignment_methods.py — MC+Rubin's vs PPIBoot: CI validity and efficiency
under LLM judge measurement uncertainty.

Compares five approaches for constructing confidence intervals on pairwise model
comparisons when LLM judge scores are an imperfect proxy for human judgement:

  llm_only    Percentile bootstrap on raw LLM scores
              (biased estimator for the true human quantity when alignment < 1)
  human_only  Percentile bootstrap on labeled-human-only data
              (valid, wide CIs due to small n_labeled)
  ppi         PPIBoot: debiased estimator μ̂_PPI = mean(Ŷ_all) + (mean(Y_lab) − mean(Ŷ_lab))
              with paired percentile bootstrap for variance
  mc_rubin    MC imputation + Rubin's (1987) combining rules
              (from the main evalstats package)
  oracle      Percentile bootstrap on full human labels for all N items
              (theoretical best; requires n_labeled = n_items, never available in practice)

True estimand: HUMAN-JUDGMENT accuracy (entity means μ_A, μ_B and pairwise diff δ = μ_A − μ_B).
Coverage is measured against these true human quantities, NOT the LLM-based quantities.

Key sweep dimensions:
  eval_type       binary | likert | continuous
  agreement_rate  0.5 – 1.0  (LLM-human agreement; 1.0 = perfect, 0.5 = random)
  n_items         items per model  (labeled subset is n_labeled // 2 per model)
  n_labeled       total labeled items across both models (split evenly)

True human parameters (fixed across all agreement_rate levels):
  binary:     p_A = 0.65, p_B = 0.50, δ = 0.15
  likert:     μ_A ~ round(N(4.0,1.0)) clipped [1,5], μ_B ~ round(N(3.0,1.0))
  continuous: Y_A ~ Beta(7,3)  μ_A = 0.70,  Y_B ~ Beta(4,6)  μ_B = 0.40

LLM noise models (parameterised by agreement_rate):
  binary:     flip with prob (1 − agreement_rate)
  likert:     exact match with prob agreement_rate, else add ±1 (clipped)
  continuous: additive Gaussian ε ~ N(0, σ²) clipped to [0,1];
              σ calibrated so Pearson r ≈ agreement_rate

Approximate run times (M2 Pro, single process unless noted):
  --quick                                    ~2s   (50 reps, binary, 4 agreement rates)
  --reps 500 --eval-types binary             ~5 min
  --reps 500 (all 3 eval types)              ~15 min
  --reps 1000 (all 3 eval types)             ~30 min
  --official-test --workers 0               ~35 min  (all scenarios; recommended)

Usage:
  python simulations/sim_alignment_methods.py --quick
  python simulations/sim_alignment_methods.py --reps 1000 --eval-types binary
  python simulations/sim_alignment_methods.py --scenarios standard null biased_llm
  python simulations/sim_alignment_methods.py --official-test --workers 0
  python simulations/sim_alignment_methods.py --out-dir simulations/out --plots save
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as _norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_HERE, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.alignment import _fit_calibration, AlignmentResult


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

METHODS   = ["llm_only", "human_only", "ppi", "ppi_linear", "calibrated_ppi", "mc_rubin", "oracle"]
SCENARIOS = ["standard", "null", "biased_llm"]

# True human parameters — fixed, independent of LLM alignment quality.
# p_A, p_B are human accuracy rates; delta = p_A - p_B is the true pairwise diff.
_TRUE_BINARY   = {"mu_a": 0.65, "mu_b": 0.50, "delta": 0.15}
_TRUE_CONTINUOUS = {"a_a": 7.0, "b_a": 3.0,   # Beta(7,3) -> mu = 0.70
                    "a_b": 4.0, "b_b": 6.0,   # Beta(4,6) -> mu = 0.40
                    "mu_a": 7.0 / 10.0, "mu_b": 4.0 / 10.0, "delta": 0.30}
# Null effect: both models identical distributions (delta = 0 -> tests Type I error)
_TRUE_NULL_BINARY     = {"mu_a": 0.60, "mu_b": 0.60, "delta": 0.0}
_TRUE_NULL_CONTINUOUS = {"a_a": 5.0, "b_a": 5.0, "a_b": 5.0, "b_b": 5.0,
                          "mu_a": 0.50, "mu_b": 0.50, "delta": 0.0}

# Likert true means estimated via MC at module load (fast — 500K samples)
_LIKERT_MU_A = _LIKERT_MU_B = _LIKERT_DELTA = None  # filled in _init_likert_true_means()
_NULL_LIKERT_MU: float | None = None                 # filled in _init_null_likert_true_means()

# Systematic upward bias applied in biased_llm continuous scenario
_CONT_BIAS = 0.15


def _init_likert_true_means() -> None:
    global _LIKERT_MU_A, _LIKERT_MU_B, _LIKERT_DELTA
    rng = np.random.default_rng(0)
    n = 500_000
    a = np.clip(np.rint(rng.normal(4.0, 1.0, n)), 1.0, 5.0)
    b = np.clip(np.rint(rng.normal(3.0, 1.0, n)), 1.0, 5.0)
    _LIKERT_MU_A = float(np.mean(a))
    _LIKERT_MU_B = float(np.mean(b))
    _LIKERT_DELTA = _LIKERT_MU_A - _LIKERT_MU_B


def _init_null_likert_true_means() -> None:
    global _NULL_LIKERT_MU
    rng = np.random.default_rng(1)
    n = 500_000
    a = np.clip(np.rint(rng.normal(3.5, 1.0, n)), 1.0, 5.0)
    _NULL_LIKERT_MU = float(np.mean(a))


_init_likert_true_means()
_init_null_likert_true_means()


def true_params(eval_type: str, scenario: str = "standard") -> tuple[float, float, float]:
    """Return (mu_a, mu_b, delta) for the given eval type and scenario."""
    if scenario == "null":
        if eval_type == "binary":
            return _TRUE_NULL_BINARY["mu_a"], _TRUE_NULL_BINARY["mu_b"], _TRUE_NULL_BINARY["delta"]
        if eval_type == "continuous":
            return (_TRUE_NULL_CONTINUOUS["mu_a"], _TRUE_NULL_CONTINUOUS["mu_b"],
                    _TRUE_NULL_CONTINUOUS["delta"])
        return _NULL_LIKERT_MU, _NULL_LIKERT_MU, 0.0
    # standard and biased_llm share the same true human parameters
    if eval_type == "binary":
        return _TRUE_BINARY["mu_a"], _TRUE_BINARY["mu_b"], _TRUE_BINARY["delta"]
    if eval_type == "continuous":
        return _TRUE_CONTINUOUS["mu_a"], _TRUE_CONTINUOUS["mu_b"], _TRUE_CONTINUOUS["delta"]
    return _LIKERT_MU_A, _LIKERT_MU_B, _LIKERT_DELTA


# Noise σ² for continuous scores: solve Corr(Y, clip(Y+ε,0,1)) ≈ agreement_rate.
# Pre-compute for a range of values using Beta(7,3) variance as reference.
_CONT_VAR_Y = 7.0 * 3.0 / (10.0 ** 2 * 11.0)  # Var(Beta(7,3)) ≈ 0.019


def _continuous_noise_sigma(agreement_rate: float) -> float:
    """Gaussian noise σ calibrated so Pearson r ≈ agreement_rate (pre-clip approx)."""
    if agreement_rate >= 1.0:
        return 0.0
    r2 = max(float(agreement_rate) ** 2, 1e-6)
    return float(np.sqrt(_CONT_VAR_Y * (1.0 / r2 - 1.0)))


# ──────────────────────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────────────────────

def gen_true_human(n: int, eval_type: str, model: str,
                   rng: np.random.Generator,
                   scenario: str = "standard") -> np.ndarray:
    """Generate n ground-truth human scores for a given model, eval type, and scenario."""
    if scenario == "null":
        # Both models draw from the same distribution (delta = 0)
        if eval_type == "binary":
            return rng.binomial(1, _TRUE_NULL_BINARY["mu_a"], n).astype(float)
        if eval_type == "continuous":
            return rng.beta(_TRUE_NULL_CONTINUOUS["a_a"], _TRUE_NULL_CONTINUOUS["b_a"], n)
        return np.clip(np.rint(rng.normal(3.5, 1.0, n)), 1.0, 5.0)
    # standard and biased_llm use the same true human distributions
    if eval_type == "binary":
        p = _TRUE_BINARY["mu_a"] if model == "A" else _TRUE_BINARY["mu_b"]
        return rng.binomial(1, p, n).astype(float)
    if eval_type == "continuous":
        if model == "A":
            return rng.beta(_TRUE_CONTINUOUS["a_a"], _TRUE_CONTINUOUS["b_a"], n)
        return rng.beta(_TRUE_CONTINUOUS["a_b"], _TRUE_CONTINUOUS["b_b"], n)
    # likert
    mu = 4.0 if model == "A" else 3.0
    return np.clip(np.rint(rng.normal(mu, 1.0, n)), 1.0, 5.0)


def apply_llm_noise(human: np.ndarray, agreement_rate: float,
                    eval_type: str, rng: np.random.Generator,
                    scenario: str = "standard") -> np.ndarray:
    """Apply LLM judge noise to true human labels.

    standard / null
        symmetric random noise:
          binary:     flip with prob (1 - agreement_rate)
          likert:     exact match with prob agreement_rate, else +/-1 clipped to [1,5]
          continuous: additive N(0, sigma^2) clipped to [0,1]

    biased_llm
        directional noise — when wrong, always predict high (lenient judge):
          binary:     wrong -> always predict 1
          likert:     wrong -> +1 (never -1), clipped to [1,5]
          continuous: N(+_CONT_BIAS, sigma^2) clipped to [0,1]
        This tests whether PPI's debiasing property corrects systematic overestimation.
    """
    n = len(human)
    if scenario == "biased_llm" and agreement_rate < 1.0:
        if eval_type == "binary":
            correct = rng.random(n) < agreement_rate
            return np.where(correct, human, 1.0)   # wrong -> always 1 (lenient)
        if eval_type == "likert":
            correct = rng.random(n) < agreement_rate
            return np.clip(np.where(correct, human, human + 1.0), 1.0, 5.0)
        # continuous: Gaussian noise with positive mean shift
        sigma = _continuous_noise_sigma(agreement_rate)
        return np.clip(human + rng.normal(_CONT_BIAS, sigma if sigma > 0.0 else 1e-9, n), 0.0, 1.0)
    # standard / null / biased_llm at perfect alignment: symmetric random noise
    if eval_type == "binary":
        flip = rng.random(n) >= agreement_rate
        return np.where(flip, 1.0 - human, human)
    if eval_type == "likert":
        noisy = rng.random(n) >= agreement_rate
        direction = rng.choice([-1, 1], size=n)
        return np.clip(human + np.where(noisy, direction, 0), 1.0, 5.0)
    # continuous
    sigma = _continuous_noise_sigma(agreement_rate)
    if sigma == 0.0:
        return human.copy()
    return np.clip(human + rng.normal(0.0, sigma, n), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Fast vectorised bootstrap helpers
# ──────────────────────────────────────────────────────────────────────────────

def _boot_mean_ci(data: np.ndarray, alpha: float,
                  n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of data. Returns (mean, lo, hi)."""
    n = len(data)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = data[idx].mean(axis=1)
    return (float(data.mean()),
            float(np.percentile(boot, 50.0 * alpha)),
            float(np.percentile(boot, 100.0 - 50.0 * alpha)))


def _boot_diff_ci(da: np.ndarray, db: np.ndarray, alpha: float,
                  n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Paired percentile bootstrap CI for mean(da) − mean(db). Returns (diff, lo, hi)."""
    n = len(da)
    idx = rng.integers(0, n, size=(n_boot, n))  # paired resample
    boot = da[idx].mean(axis=1) - db[idx].mean(axis=1)
    obs = float(da.mean() - db.mean())
    return (obs,
            float(np.percentile(boot, 50.0 * alpha)),
            float(np.percentile(boot, 100.0 - 50.0 * alpha)))


# ──────────────────────────────────────────────────────────────────────────────
# Calibrated-PPI: calibration models and estimator
# ──────────────────────────────────────────────────────────────────────────────

class _IdentityCalibration:
    def fit(self, x, y): return self
    def predict(self, x): return np.asarray(x)


class _LinearCalibration:
    def fit(self, x, y):
        x = np.asarray(x)
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
        self.a_, self.b_ = beta[0], beta[1]
        return self

    def predict(self, x):
        return self.a_ + self.b_ * np.asarray(x)


class _RidgeCalibration:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, x, y):
        self.model_ = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model_.fit(np.asarray(x).reshape(-1, 1), y)
        return self

    def predict(self, x):
        return self.model_.predict(np.asarray(x).reshape(-1, 1))


class _SigmoidCalibration:
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, x, y):
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            # Single class in training data — fall back to constant predictor
            self._const = float(y.mean())
            self.model_ = None
        else:
            self._const = None
            self.model_ = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000)
            self.model_.fit(np.asarray(x).reshape(-1, 1), y)
        return self

    def predict(self, x):
        if self.model_ is None:
            return np.full(len(np.asarray(x)), self._const)
        return self.model_.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]


class _IsotonicCalibration:
    def fit(self, x, y):
        self.model_ = IsotonicRegression(out_of_bounds="clip")
        self.model_.fit(x, y)
        return self

    def predict(self, x):
        return self.model_.predict(np.asarray(x))




# ──────────────────────────────────────────────────────────────────────────────
# CI methods
# ──────────────────────────────────────────────────────────────────────────────

def _ci_llm_only(
    llm_a: np.ndarray, llm_b: np.ndarray,
    alpha: float, n_boot: int, rng: np.random.Generator,
) -> dict:
    """Standard bootstrap CI on raw LLM scores. Biased when LLM ≠ human."""
    mu_a, lo_a, hi_a = _boot_mean_ci(llm_a, alpha, n_boot, rng)
    mu_b, lo_b, hi_b = _boot_mean_ci(llm_b, alpha, n_boot, rng)
    diff, lo_d, hi_d = _boot_diff_ci(llm_a, llm_b, alpha, n_boot, rng)
    return dict(mu_a=mu_a, lo_a=lo_a, hi_a=hi_a,
                mu_b=mu_b, lo_b=lo_b, hi_b=hi_b,
                diff=diff, lo_d=lo_d, hi_d=hi_d)


def _ci_human_only(
    human_a_lab: np.ndarray, human_b_lab: np.ndarray,
    alpha: float, n_boot: int, rng: np.random.Generator,
) -> dict:
    """Bootstrap CI on labeled-human-only data. Valid but wide (small n_labeled)."""
    mu_a, lo_a, hi_a = _boot_mean_ci(human_a_lab, alpha, n_boot, rng)
    mu_b, lo_b, hi_b = _boot_mean_ci(human_b_lab, alpha, n_boot, rng)
    diff, lo_d, hi_d = _boot_diff_ci(human_a_lab, human_b_lab, alpha, n_boot, rng)
    return dict(mu_a=mu_a, lo_a=lo_a, hi_a=hi_a,
                mu_b=mu_b, lo_b=lo_b, hi_b=hi_b,
                diff=diff, lo_d=lo_d, hi_d=hi_d)


def _ci_oracle(
    human_a_all: np.ndarray, human_b_all: np.ndarray,
    alpha: float, n_boot: int, rng: np.random.Generator,
) -> dict:
    """Bootstrap CI on all true human labels (only achievable with infinite budget)."""
    mu_a, lo_a, hi_a = _boot_mean_ci(human_a_all, alpha, n_boot, rng)
    mu_b, lo_b, hi_b = _boot_mean_ci(human_b_all, alpha, n_boot, rng)
    diff, lo_d, hi_d = _boot_diff_ci(human_a_all, human_b_all, alpha, n_boot, rng)
    return dict(mu_a=mu_a, lo_a=lo_a, hi_a=hi_a,
                mu_b=mu_b, lo_b=lo_b, hi_b=hi_b,
                diff=diff, lo_d=lo_d, hi_d=hi_d)


def _ci_ppi(
    llm_a_all: np.ndarray, llm_b_all: np.ndarray,
    human_a_lab: np.ndarray, llm_a_lab: np.ndarray,
    human_b_lab: np.ndarray, llm_b_lab: np.ndarray,
    alpha: float, n_boot: int, rng: np.random.Generator,
) -> dict:
    """PPIBoot: prediction-powered inference with paired percentile bootstrap.

    PPI point estimate (per entity k):
        μ̂_k = mean(Ŷ_k_all) + (mean(Y_k_lab) − mean(Ŷ_k_lab))

    Bootstrap: for each draw, resample the full LLM array AND the labeled array
    independently.  Variance is driven by both the full-set sampling uncertainty
    and the labeled-correction variance.

    Reference: Kiyani et al. (2024) "Statistical Inference with Imperfect Models"
    """
    n_a = len(llm_a_all)
    n_b = len(llm_b_all)
    n_lab = len(human_a_lab)  # same for both (equal split)

    # Point estimates
    delta_a = float(human_a_lab.mean() - llm_a_lab.mean())
    delta_b = float(human_b_lab.mean() - llm_b_lab.mean())
    mu_a = float(llm_a_all.mean()) + delta_a
    mu_b = float(llm_b_all.mean()) + delta_b
    diff = mu_a - mu_b

    # Bootstrap: resample full sets (paired across A and B) + labeled sets (paired)
    idx_all = rng.integers(0, n_a, size=(n_boot, n_a))       # paired full resample
    idx_lab = rng.integers(0, n_lab, size=(n_boot, n_lab))   # paired labeled resample

    mu_a_b = llm_a_all[idx_all].mean(axis=1) + (
        human_a_lab[idx_lab].mean(axis=1) - llm_a_lab[idx_lab].mean(axis=1)
    )
    mu_b_b = llm_b_all[idx_all].mean(axis=1) + (
        human_b_lab[idx_lab].mean(axis=1) - llm_b_lab[idx_lab].mean(axis=1)
    )
    diff_b = mu_a_b - mu_b_b

    lo_a  = float(np.percentile(mu_a_b, 50.0 * alpha))
    hi_a  = float(np.percentile(mu_a_b, 100.0 - 50.0 * alpha))
    lo_b  = float(np.percentile(mu_b_b, 50.0 * alpha))
    hi_b  = float(np.percentile(mu_b_b, 100.0 - 50.0 * alpha))
    lo_d  = float(np.percentile(diff_b, 50.0 * alpha))
    hi_d  = float(np.percentile(diff_b, 100.0 - 50.0 * alpha))

    return dict(mu_a=mu_a, lo_a=lo_a, hi_a=hi_a,
                mu_b=mu_b, lo_b=lo_b, hi_b=hi_b,
                diff=diff, lo_d=lo_d, hi_d=hi_d)


def _ci_ppi_linear(
    llm_a_all: np.ndarray, llm_b_all: np.ndarray,
    human_a_lab: np.ndarray, llm_a_lab: np.ndarray,
    human_b_lab: np.ndarray, llm_b_lab: np.ndarray,
    alpha: float, n_boot: int, rng: np.random.Generator,
    ridge_alpha: float = 0.1,
) -> dict:
    """Calibrated PPIBoot with vectorised ridge-OLS calibration — PPI++ style.

    Fits β = Cov(Ŷ_lab, Y_lab) / (Var(Ŷ_lab) + ridge_alpha) per bootstrap draw.
    ridge_alpha > 0 prevents β from blowing up in bootstrap resamples that
    happen to have near-zero spread in Ŷ_lab (common with few labels and
    continuous scores).  ridge_alpha=0 recovers vanilla OLS.

    Point estimate:
        μ_k = mean(Y_k_lab) + β * (mean(Ŷ_k_all) − mean(Ŷ_k_lab))
    """
    n_all = len(llm_a_all)
    n_lab = len(human_a_lab)

    def _point(y, x_all, x_lab):
        xc = x_lab - x_lab.mean()
        denom = float(np.dot(xc, xc)) + ridge_alpha
        b = float(np.dot(xc, y - y.mean())) / denom
        return y.mean() + b * (x_all.mean() - x_lab.mean())

    mu_a = _point(human_a_lab, llm_a_all, llm_a_lab)
    mu_b = _point(human_b_lab, llm_b_all, llm_b_lab)
    diff = mu_a - mu_b

    idx_all = rng.integers(0, n_all, (n_boot, n_all))
    idx_lab = rng.integers(0, n_lab, (n_boot, n_lab))

    def _boot(y, x_all, x_lab):
        y_b  = y[idx_lab]       # (n_boot, n_lab)
        xl_b = x_lab[idx_lab]   # (n_boot, n_lab)
        xa_b = x_all[idx_all]   # (n_boot, n_all)
        xm_b = xl_b.mean(axis=1, keepdims=True)
        ym_b = y_b.mean(axis=1, keepdims=True)
        xc   = xl_b - xm_b
        denom = (xc ** 2).sum(axis=1) + ridge_alpha
        b = (xc * (y_b - ym_b)).sum(axis=1) / denom
        return ym_b.squeeze() + b * (xa_b.mean(axis=1) - xm_b.squeeze())

    boot_a    = _boot(human_a_lab, llm_a_all, llm_a_lab)
    boot_b    = _boot(human_b_lab, llm_b_all, llm_b_lab)
    boot_diff = boot_a - boot_b

    def _pct(arr, q): return float(np.percentile(arr, q))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return dict(
        mu_a=float(mu_a), lo_a=_pct(boot_a, lo), hi_a=_pct(boot_a, hi),
        mu_b=float(mu_b), lo_b=_pct(boot_b, lo), hi_b=_pct(boot_b, hi),
        diff=float(diff),  lo_d=_pct(boot_diff, lo), hi_d=_pct(boot_diff, hi),
    )


def _pick_calibrator(eval_type: str):
    """Choose calibrator by eval_type.

    binary:              sigmoid (logistic + L2) — 2-param, won't memorise few labels
    continuous / likert: ridge — smooth, regularised, scale-invariant
    """
    if eval_type == "binary":
        return _SigmoidCalibration
    return _RidgeCalibration


def _ci_calibrated_ppi(
    llm_a_all: np.ndarray, llm_b_all: np.ndarray,
    human_a_lab: np.ndarray, llm_a_lab: np.ndarray,
    human_b_lab: np.ndarray, llm_b_lab: np.ndarray,
    alpha: float, n_boot_cal_ppi: int, rng: np.random.Generator,
    eval_type: str = "continuous",
) -> dict:
    """Calibrated PPIBoot: fit g once, then vectorised bootstrap on precomputed arrays.

    Matches the approach of Boyeau et al. (2024, arXiv:2411.12665): the calibration
    model is fitted once on the labeled data, not refitted per bootstrap draw.

    Calibrator is chosen per eval_type to avoid overfitting with few labels:
      binary     → sigmoid (logistic + L2)
      continuous → ridge regression
      likert     → ridge regression

    Point estimate:
        mu_k = mean(g(Yhat_k_all)) + mean(Y_k_lab - g(Yhat_k_lab))

    Bootstrap: g is fixed; resample g(Yhat_k_all) and residuals independently.
    Fully vectorised — same cost as standard PPI.
    """
    calibrator_factory = _pick_calibrator(eval_type)
    n_all = len(llm_a_all)
    n_lab = len(human_a_lab)

    # Fit one calibrator per entity on the full labeled set
    cal_a = calibrator_factory().fit(llm_a_lab, human_a_lab)
    cal_b = calibrator_factory().fit(llm_b_lab, human_b_lab)

    # Precompute calibrated predictions and residuals (g is now fixed)
    g_a_all = cal_a.predict(llm_a_all)
    g_b_all = cal_b.predict(llm_b_all)
    r_a = human_a_lab - cal_a.predict(llm_a_lab)
    r_b = human_b_lab - cal_b.predict(llm_b_lab)

    mu_a = float(g_a_all.mean() + r_a.mean())
    mu_b = float(g_b_all.mean() + r_b.mean())
    diff = mu_a - mu_b

    # Vectorised bootstrap: just index into precomputed arrays
    idx_all = rng.integers(0, n_all, (n_boot_cal_ppi, n_all))
    idx_lab = rng.integers(0, n_lab, (n_boot_cal_ppi, n_lab))

    boot_a    = g_a_all[idx_all].mean(axis=1) + r_a[idx_lab].mean(axis=1)
    boot_b    = g_b_all[idx_all].mean(axis=1) + r_b[idx_lab].mean(axis=1)
    boot_diff = boot_a - boot_b

    def _pct(arr, q): return float(np.percentile(arr, q))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return dict(
        mu_a=mu_a, lo_a=_pct(boot_a, lo), hi_a=_pct(boot_a, hi),
        mu_b=mu_b, lo_b=_pct(boot_b, lo), hi_b=_pct(boot_b, hi),
        diff=diff,  lo_d=_pct(boot_diff, lo), hi_d=_pct(boot_diff, hi),
    )


def _ci_mc_rubin(
    llm_a_all: np.ndarray, llm_b_all: np.ndarray,
    human_a_lab: np.ndarray, llm_a_lab: np.ndarray,
    human_b_lab: np.ndarray, llm_b_lab: np.ndarray,
    eval_type: str,
    alpha: float, n_mc: int, n_boot_inner: int,
    rng: np.random.Generator,
) -> dict:
    """MC imputation + Rubin's (1987) combining rules.

    1. Fit a separate Bayesian calibration model per model entity (A and B).
       Pooling is WRONG for continuous scores: it conflates between-model
       variation (A scores higher than B on both LLM and human) with
       within-model calibration quality, inflating beta and attenuating the
       estimated delta (ecological confounding / Simpson's paradox).
       Binary and likert class-conditional models are immune, but using
       per-model calibration is correct and consistent for all score types.
    2. For m = 1..n_mc:
       a. Sample parameters from each model's calibration posterior.
       b. Impute full human labels for A using cal_a, B using cal_b.
       c. Run vectorised bootstrap on imputed data.
    3. Pool entity means and CI half-widths via Rubin's rules:
       T = W̄ + (1 + 1/M)·B,  CI = Q̄ ± z·√T
    """
    # Fit one calibration model per entity — avoids between-model confounding
    cal_a = _fit_calibration(llm_a_lab, human_a_lab, eval_type)
    cal_b = _fit_calibration(llm_b_lab, human_b_lab, eval_type)
    n_total = len(llm_a_all) + len(llm_b_all)
    ar_a = AlignmentResult(
        llm_metric="score", human_col="human",
        score_type=eval_type, n_labeled=len(human_a_lab),
        n_total=n_total, calibration=cal_a,
        alignment_metrics={}, representativeness={},
    )
    ar_b = AlignmentResult(
        llm_metric="score", human_col="human",
        score_type=eval_type, n_labeled=len(human_b_lab),
        n_total=n_total, calibration=cal_b,
        alignment_metrics={}, representativeness={},
    )

    n_a = len(llm_a_all)
    n_b = len(llm_b_all)

    means_a   = np.empty(n_mc)
    means_b   = np.empty(n_mc)
    diffs_mc  = np.empty(n_mc)
    ci_lo_a   = np.empty(n_mc)
    ci_hi_a   = np.empty(n_mc)
    ci_lo_b   = np.empty(n_mc)
    ci_hi_b   = np.empty(n_mc)
    ci_lo_d   = np.empty(n_mc)
    ci_hi_d   = np.empty(n_mc)

    for m in range(n_mc):
        # Impute each model's full dataset from its own calibration posterior
        imp_a = ar_a._sample_imputed_scores(llm_a_all, rng)
        imp_b = ar_b._sample_imputed_scores(llm_b_all, rng)

        means_a[m], ci_lo_a[m], ci_hi_a[m] = _boot_mean_ci(imp_a, alpha, n_boot_inner, rng)
        means_b[m], ci_lo_b[m], ci_hi_b[m] = _boot_mean_ci(imp_b, alpha, n_boot_inner, rng)
        d, lo_d, hi_d                       = _boot_diff_ci(imp_a, imp_b, alpha, n_boot_inner, rng)
        diffs_mc[m] = d
        ci_lo_d[m]  = lo_d
        ci_hi_d[m]  = hi_d

    # Rubin's combining rules
    z = float(_norm.ppf(1.0 - alpha / 2.0))

    def _rubin(means_m, ci_lo_m, ci_hi_m):
        Q_bar = float(means_m.mean())
        se_w  = (ci_hi_m - ci_lo_m) / (2.0 * z)
        W_bar = float(np.mean(se_w ** 2))
        B     = float(np.var(means_m, ddof=1))
        T     = W_bar + (1.0 + 1.0 / n_mc) * B
        se_t  = float(np.sqrt(max(T, 0.0)))
        return Q_bar, Q_bar - z * se_t, Q_bar + z * se_t

    mu_a, lo_a, hi_a = _rubin(means_a, ci_lo_a, ci_hi_a)
    mu_b, lo_b, hi_b = _rubin(means_b, ci_lo_b, ci_hi_b)
    diff, lo_d, hi_d = _rubin(diffs_mc, ci_lo_d, ci_hi_d)

    return dict(mu_a=mu_a, lo_a=lo_a, hi_a=hi_a,
                mu_b=mu_b, lo_b=lo_b, hi_b=hi_b,
                diff=diff, lo_d=lo_d, hi_d=hi_d)


# ──────────────────────────────────────────────────────────────────────────────
# Per-cell simulation result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CellResult:
    eval_type: str
    agreement_rate: float
    n_items: int
    n_labeled: int
    method: str
    scenario: str          # "standard" | "null" | "biased_llm"
    n_reps: int
    # Coverage counts (CI contains true human quantity)
    cov_a: int
    cov_b: int
    cov_diff: int
    # CI width sums
    sum_w_a: float
    sum_w_b: float
    sum_w_diff: float
    # Point-estimate bias sums |point − true|
    sum_bias_a: float
    sum_bias_b: float
    sum_bias_diff: float

    @property
    def coverage_a(self) -> float:
        return self.cov_a / self.n_reps

    @property
    def coverage_b(self) -> float:
        return self.cov_b / self.n_reps

    @property
    def coverage_diff(self) -> float:
        return self.cov_diff / self.n_reps

    @property
    def mean_width_diff(self) -> float:
        return self.sum_w_diff / self.n_reps

    @property
    def mean_bias_diff(self) -> float:
        return self.sum_bias_diff / self.n_reps


# ──────────────────────────────────────────────────────────────────────────────
# Per-cell worker
# ──────────────────────────────────────────────────────────────────────────────

def run_cell(
    eval_type: str,
    agreement_rate: float,
    n_items: int,
    n_labeled: int,
    n_reps: int,
    alpha: float,
    n_boot: int,
    n_mc: int,
    n_boot_inner: int,
    seed: int,
    methods: Sequence[str],
    scenario: str = "standard",
    n_boot_cal_ppi: int = 200,
) -> list[CellResult]:
    """Run n_reps Monte Carlo reps for one cell. Returns one CellResult per method."""
    rng = np.random.default_rng(seed)
    true_mu_a, true_mu_b, true_delta = true_params(eval_type, scenario)

    n_lab_a = n_labeled // 2
    n_lab_b = n_labeled - n_lab_a

    counts  = {m: {"cov_a": 0, "cov_b": 0, "cov_d": 0} for m in methods}
    sums_w  = {m: {"a": 0.0, "b": 0.0, "d": 0.0} for m in methods}
    sums_b  = {m: {"a": 0.0, "b": 0.0, "d": 0.0} for m in methods}

    for _ in range(n_reps):
        # Ground-truth human scores
        human_a = gen_true_human(n_items, eval_type, "A", rng, scenario)
        human_b = gen_true_human(n_items, eval_type, "B", rng, scenario)

        # LLM judge scores (noisy proxy — model depends on scenario)
        llm_a = apply_llm_noise(human_a, agreement_rate, eval_type, rng, scenario)
        llm_b = apply_llm_noise(human_b, agreement_rate, eval_type, rng, scenario)

        # Labeled subset (random split, equal per model)
        idx_a_lab = rng.choice(n_items, size=n_lab_a, replace=False)
        idx_b_lab = rng.choice(n_items, size=n_lab_b, replace=False)
        human_a_lab = human_a[idx_a_lab]
        llm_a_lab   = llm_a[idx_a_lab]
        human_b_lab = human_b[idx_b_lab]
        llm_b_lab   = llm_b[idx_b_lab]

        # Run each method
        method_cis: dict[str, dict] = {}
        for m in methods:
            if m == "llm_only":
                method_cis[m] = _ci_llm_only(llm_a, llm_b, alpha, n_boot, rng)
            elif m == "human_only":
                method_cis[m] = _ci_human_only(human_a_lab, human_b_lab, alpha, n_boot, rng)
            elif m == "oracle":
                method_cis[m] = _ci_oracle(human_a, human_b, alpha, n_boot, rng)
            elif m == "ppi":
                method_cis[m] = _ci_ppi(
                    llm_a, llm_b,
                    human_a_lab, llm_a_lab, human_b_lab, llm_b_lab,
                    alpha, n_boot, rng,
                )
            elif m == "ppi_linear":
                method_cis[m] = _ci_ppi_linear(
                    llm_a, llm_b,
                    human_a_lab, llm_a_lab, human_b_lab, llm_b_lab,
                    alpha, n_boot, rng,
                )
            elif m == "calibrated_ppi":
                method_cis[m] = _ci_calibrated_ppi(
                    llm_a, llm_b,
                    human_a_lab, llm_a_lab, human_b_lab, llm_b_lab,
                    alpha, n_boot_cal_ppi, rng,
                    eval_type=eval_type,
                )
            elif m == "mc_rubin":
                method_cis[m] = _ci_mc_rubin(
                    llm_a, llm_b,
                    human_a_lab, llm_a_lab, human_b_lab, llm_b_lab,
                    eval_type, alpha, n_mc, n_boot_inner, rng,
                )

        # Accumulate
        for m, ci in method_cis.items():
            if ci["lo_a"] <= true_mu_a <= ci["hi_a"]:
                counts[m]["cov_a"] += 1
            if ci["lo_b"] <= true_mu_b <= ci["hi_b"]:
                counts[m]["cov_b"] += 1
            if ci["lo_d"] <= true_delta <= ci["hi_d"]:
                counts[m]["cov_d"] += 1
            sums_w[m]["a"] += ci["hi_a"] - ci["lo_a"]
            sums_w[m]["b"] += ci["hi_b"] - ci["lo_b"]
            sums_w[m]["d"] += ci["hi_d"] - ci["lo_d"]
            sums_b[m]["a"] += abs(ci["mu_a"] - true_mu_a)
            sums_b[m]["b"] += abs(ci["mu_b"] - true_mu_b)
            sums_b[m]["d"] += abs(ci["diff"] - true_delta)

    return [
        CellResult(
            eval_type=eval_type, agreement_rate=agreement_rate,
            n_items=n_items, n_labeled=n_labeled,
            method=m, scenario=scenario, n_reps=n_reps,
            cov_a=counts[m]["cov_a"], cov_b=counts[m]["cov_b"],
            cov_diff=counts[m]["cov_d"],
            sum_w_a=sums_w[m]["a"], sum_w_b=sums_w[m]["b"],
            sum_w_diff=sums_w[m]["d"],
            sum_bias_a=sums_b[m]["a"], sum_bias_b=sums_b[m]["b"],
            sum_bias_diff=sums_b[m]["d"],
        )
        for m in methods
    ]


# Module-level worker for multiprocessing.Pool (must be importable at top level).
# Args tuple order matches run_cell keyword parameters — all plain Python types.
def _run_cell_worker(args: tuple) -> list[CellResult]:
    return run_cell(
        eval_type=args[0], agreement_rate=args[1], n_items=args[2],
        n_labeled=args[3], n_reps=args[4], alpha=args[5], n_boot=args[6],
        n_mc=args[7], n_boot_inner=args[8], seed=args[9], methods=args[10],
        scenario=args[11], n_boot_cal_ppi=args[12],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(
    eval_types: list[str],
    agreement_rates: list[float],
    n_items_list: list[int],
    n_labeled_list: list[int],
    methods: list[str],
    n_reps: int,
    alpha: float,
    n_boot: int,
    n_mc: int,
    n_boot_inner: int,
    progress: bool = True,
    scenarios: list[str] | None = None,
    n_workers: int = 1,
    n_boot_cal_ppi: int = 200,
) -> list[CellResult]:
    """Run the full simulation sweep. Returns all CellResult objects.

    scenarios   list of scenario names to sweep (default: ["standard"])
    n_workers   parallel worker processes; 1 = sequential (default)
    """
    if scenarios is None:
        scenarios = ["standard"]

    cells = [
        (sc, et, ar, ni, nl)
        for sc in scenarios
        for et in eval_types
        for ar in agreement_rates
        for ni in n_items_list
        for nl in n_labeled_list
        if nl <= ni
    ]
    n_cells = len(cells)
    results: list[CellResult] = []

    # Seed offsets per scenario so results are independent across scenarios
    _SC_SEED = {"standard": 0, "null": 1_000_000_000, "biased_llm": 2_000_000_000}

    task_args = [
        (et, ar, ni, nl, n_reps, alpha, n_boot, n_mc, n_boot_inner,
         _SC_SEED.get(sc, 0) + cell_idx * 1_000_000 + ni * 1000 + nl,
         list(methods), sc, n_boot_cal_ppi)
        for cell_idx, (sc, et, ar, ni, nl) in enumerate(cells)
    ]

    t0 = time.time()

    def _print_progress(done: int) -> None:
        elapsed = time.time() - t0
        eta = elapsed / done * (n_cells - done) if done > 0 else 0
        eta_m, eta_s = divmod(int(round(eta)), 60)
        bar_w = 30
        filled = int(bar_w * done / n_cells)
        bar = "█" * filled + "·" * (bar_w - filled)
        print(
            f"\r  [{bar}] {100*done/n_cells:5.1f}%"
            f"  {done}/{n_cells}  ETA {eta_m:02d}:{eta_s:02d}",
            end="", flush=True,
        )

    if n_workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(
                pool.imap_unordered(_run_cell_worker, task_args)
            ):
                results.extend(cell_results)
                if progress:
                    _print_progress(i + 1)
    else:
        for cell_idx, (args, (sc, et, ar, ni, nl)) in enumerate(zip(task_args, cells)):
            results.extend(_run_cell_worker(args))
            if progress:
                elapsed = time.time() - t0
                done = cell_idx + 1
                eta = elapsed / done * (n_cells - done) if done > 0 else 0
                eta_m, eta_s = divmod(int(round(eta)), 60)
                bar_w = 30
                filled = int(bar_w * done / n_cells)
                bar = "█" * filled + "·" * (bar_w - filled)
                detail = f"{sc}|{et}|r={ar:.1f}|n={ni}|lab={nl}"
                print(
                    f"\r  [{bar}] {100*done/n_cells:5.1f}%"
                    f"  {done}/{n_cells}  ETA {eta_m:02d}:{eta_s:02d}"
                    f"  {detail:<38s}",
                    end="", flush=True,
                )

    if progress:
        print()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

_COV_GOOD = " "    # within ±0.03 of nominal
_COV_LOW  = "↓"   # undercovering by > 0.03
_COV_HIGH = "↑"   # overcovering  by > 0.03


def _cov_mark(cov: float, target: float, tol: float = 0.03) -> str:
    if cov < target - tol:
        return _COV_LOW
    if cov > target + tol:
        return _COV_HIGH
    return _COV_GOOD


_METHOD_DISPLAY = {
    "llm_only":      "llm_only  ",
    "human_only":    "human_only",
    "ppi":           "ppi       ",
    "ppi_linear":    "ppi_linear",
    "calibrated_ppi": "cal_ppi   ",
    "mc_rubin":      "mc_rubin  ",
    "oracle":        "oracle    ",
}


_SCENARIO_LABELS = {
    "standard":   "STANDARD  (delta != 0, symmetric random noise)",
    "null":       "NULL EFFECT  (delta = 0 — Cov(delta) = P(CI contains 0); Type I error = 1 - Cov(delta))",
    "biased_llm": "BIASED LLM  (directional noise: when wrong, always predict high — tests debiasing)",
}


def print_report(results: list[CellResult], alpha: float = 0.05) -> None:
    """Print an ASCII table comparing all methods across all conditions, grouped by scenario."""
    target = 1.0 - alpha
    scenarios    = sorted({r.scenario for r in results},
                          key=lambda s: SCENARIOS.index(s) if s in SCENARIOS else 99)
    eval_types   = sorted({r.eval_type for r in results})
    n_items_list = sorted({r.n_items for r in results})
    n_lab_list   = sorted({r.n_labeled for r in results})
    ar_list      = sorted({r.agreement_rate for r in results})
    methods      = [m for m in METHODS if any(r.method == m for r in results)]

    # Index results for fast lookup
    idx: dict[tuple, CellResult] = {}
    for r in results:
        idx[(r.scenario, r.eval_type, r.agreement_rate, r.n_items, r.n_labeled, r.method)] = r

    for scenario in scenarios:
        print()
        print("=" * 92)
        print(f"  SCENARIO: {_SCENARIO_LABELS.get(scenario, scenario.upper())}")
        print(f"  Nominal coverage: {target:.0%}   Markers: {_COV_LOW}=undercovers  {_COV_HIGH}=overcovers")
        print("=" * 92)

        for et in eval_types:
            true_mu_a, true_mu_b, true_delta = true_params(et, scenario)
            print(f"\n  -- {et.upper()}  (true delta = {true_delta:.3f}) --")

            for ni in n_items_list:
                for nl in n_lab_list:
                    if nl > ni:
                        continue
                    print(f"\n  n_items={ni}  n_labeled={nl}  (n_lab per model ~= {nl//2})")
                    hdr = (f"  {'AgreementRate':>13}  {'Method':<10}  |"
                           f"  {'Cov(d)':>6}  {'Cov(A)':>6}  {'Cov(B)':>6}  |"
                           f"  {'Width(d)':>8}  {'Bias(d)':>7}")
                    print(hdr)
                    print("  " + "-" * (len(hdr) - 2))

                    for ar in ar_list:
                        first_in_group = True
                        for m in methods:
                            key = (scenario, et, ar, ni, nl, m)
                            if key not in idx:
                                continue
                            r = idx[key]
                            cov_d = r.coverage_diff
                            cov_a = r.coverage_a
                            cov_b = r.coverage_b
                            wd    = r.mean_width_diff
                            bd    = r.mean_bias_diff

                            ar_str = f"{ar:.2f}" if first_in_group else "     "
                            mark_d = _cov_mark(cov_d, target)
                            mark_a = _cov_mark(cov_a, target)
                            mark_b = _cov_mark(cov_b, target)
                            print(
                                f"  {ar_str:>13}  {_METHOD_DISPLAY[m]}  |"
                                f"  {cov_d:.3f}{mark_d}  {cov_a:.3f}{mark_a}"
                                f"  {cov_b:.3f}{mark_b}  |"
                                f"  {wd:>8.4f}  {bd:>7.4f}"
                            )
                            first_in_group = False

    print()
    print("=" * 92)
    print()


def print_summary_table(results: list[CellResult], alpha: float = 0.05) -> None:
    """Print a compact summary for each (scenario, eval_type, method), averaged across
    agreement_rates, n_items, n_labeled."""
    target = 1.0 - alpha
    scenarios  = sorted({r.scenario for r in results},
                        key=lambda s: SCENARIOS.index(s) if s in SCENARIOS else 99)
    eval_types = sorted({r.eval_type for r in results})
    methods    = [m for m in METHODS if any(r.method == m for r in results)]

    for sc in scenarios:
        sc_results = [r for r in results if r.scenario == sc]
        if not sc_results:
            continue
        sc_label = _SCENARIO_LABELS.get(sc, sc.upper())
        print(f"\n  SUMMARY [{sc_label[:40]}]:")
        hdr = f"  {'Eval':>12}  {'Method':<10}  {'Cov(d)':>7}  {'Cov(A)':>7}  {'Cov(B)':>7}  {'Width(d)':>9}  {'Bias(d)':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for et in eval_types:
            for m in methods:
                subset = [r for r in sc_results if r.eval_type == et and r.method == m]
                if not subset:
                    continue
                cov_d = float(np.mean([r.coverage_diff for r in subset]))
                cov_a = float(np.mean([r.coverage_a    for r in subset]))
                cov_b = float(np.mean([r.coverage_b    for r in subset]))
                wd    = float(np.mean([r.mean_width_diff for r in subset]))
                bd    = float(np.mean([r.mean_bias_diff  for r in subset]))
                mark  = _cov_mark(cov_d, target)
                print(
                    f"  {et:>12}  {_METHOD_DISPLAY[m]}"
                    f"  {cov_d:.3f}{mark}  {cov_a:.3f}   {cov_b:.3f}"
                    f"  {wd:>9.4f}  {bd:>8.4f}"
                )
            print()


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

_METHOD_COLORS = {
    "llm_only":      "#d62728",
    "human_only":    "#8c564b",
    "ppi":           "#2ca02c",
    "ppi_linear":    "#9467bd",
    "calibrated_ppi": "#ff7f0e",
    "mc_rubin":      "#1f77b4",
    "oracle":        "#7f7f7f",
}
_METHOD_MARKERS = {
    "llm_only":      "X",
    "human_only":    "s",
    "ppi":           "D",
    "ppi_linear":    "P",
    "calibrated_ppi": "v",
    "mc_rubin":      "o",
    "oracle":        "^",
}
_METHOD_LS = {
    "llm_only":      "--",
    "human_only":    ":",
    "ppi":           "-",
    "ppi_linear":    "-",
    "calibrated_ppi": "-",
    "mc_rubin":      "-",
    "oracle":        "-.",
}


def make_plots(
    results: list[CellResult],
    alpha: float = 0.05,
    out_path: str | None = None,
    scenario: str = "standard",
) -> None:
    """Plot coverage and CI width vs agreement_rate for one scenario."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    filtered = [r for r in results if r.scenario == scenario]
    if not filtered:
        print(f"  No results for scenario '{scenario}', skipping plot.")
        return

    target = 1.0 - alpha
    eval_types   = sorted({r.eval_type for r in filtered})
    n_items_list = sorted({r.n_items for r in filtered})
    n_lab_list   = sorted({r.n_labeled for r in filtered})
    ar_list      = sorted({r.agreement_rate for r in filtered})
    methods      = [m for m in METHODS if any(r.method == m for r in filtered)]

    # Layout: rows = eval_type, cols per row = 2 x n_sizes x n_labs (coverage + width)
    n_size_combos = sum(1 for ni in n_items_list for nl in n_lab_list if nl <= ni)
    n_cols = n_size_combos * 2

    fig, axes = plt.subplots(
        nrows=len(eval_types), ncols=n_cols,
        figsize=(4.0 * n_cols, 4.0 * len(eval_types)),
        squeeze=False,
    )

    idx_results: dict[tuple, CellResult] = {
        (r.eval_type, r.agreement_rate, r.n_items, r.n_labeled, r.method): r
        for r in filtered
    }

    size_combos = [(ni, nl) for ni in n_items_list for nl in n_lab_list if nl <= ni]

    for r_idx, et in enumerate(eval_types):
        for c_base, (ni, nl) in enumerate(size_combos):
            ax_cov = axes[r_idx][c_base * 2]
            ax_wid = axes[r_idx][c_base * 2 + 1]

            for m in methods:
                cov_pts = []
                wid_pts = []
                for ar in ar_list:
                    key = (et, ar, ni, nl, m)
                    if key not in idx_results:
                        continue
                    r = idx_results[key]
                    cov_pts.append((ar, r.coverage_diff))
                    wid_pts.append((ar, r.mean_width_diff))

                if not cov_pts:
                    continue

                xs = [p[0] for p in cov_pts]
                ys_cov = [p[1] for p in cov_pts]
                ys_wid = [p[1] for p in wid_pts]
                color = _METHOD_COLORS.get(m, "#333333")
                marker = _METHOD_MARKERS.get(m, "o")
                ls = _METHOD_LS.get(m, "-")

                ax_cov.plot(xs, ys_cov, color=color, linestyle=ls,
                            marker=marker, markersize=5, linewidth=1.4, label=m)
                ax_wid.plot(xs, ys_wid, color=color, linestyle=ls,
                            marker=marker, markersize=5, linewidth=1.4, label=m)

            # Coverage reference band
            ax_cov.axhline(target, color="black", linewidth=1.0, linestyle="--", zorder=0)
            ax_cov.axhspan(target - 0.03, target + 0.03,
                           color="#DDDDDD", alpha=0.4, zorder=0)
            ax_cov.set_ylim(max(0.0, target - 0.25), 1.01)
            ax_cov.set_xlabel("Agreement rate", fontsize=8)
            ax_cov.set_ylabel("Coverage (δ)", fontsize=8)
            ax_cov.set_title(f"{et} | n={ni} lab={nl}\nCoverage", fontsize=8)
            ax_cov.tick_params(labelsize=7)

            ax_wid.set_xlabel("Agreement rate", fontsize=8)
            ax_wid.set_ylabel("Mean CI width (δ)", fontsize=8)
            ax_wid.set_title(f"{et} | n={ni} lab={nl}\nWidth", fontsize=8)
            ax_wid.tick_params(labelsize=7)

    # Collect handles/labels from every subplot, deduplicate by label, preserve method order
    seen: dict[str, object] = {}
    for row in axes:
        for ax in row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen[l] = h
    ordered = [(seen[m], m) for m in methods if m in seen]
    if ordered:
        leg_handles, leg_labels = zip(*ordered)
        fig.legend(leg_handles, leg_labels, loc="lower center", ncol=len(leg_labels),
                   fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"MC+Rubin's vs PPIBoot: Coverage and Width on True Human Quantity\n"
        f"Scenario: {scenario}  |  alpha={alpha}  target coverage={target:.0%}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot: {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CSV output
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(results: list[CellResult], path: str) -> None:
    import csv
    fields = [
        "scenario", "eval_type", "agreement_rate", "n_items", "n_labeled", "method",
        "n_reps", "coverage_diff", "coverage_a", "coverage_b",
        "mean_width_diff", "mean_width_a", "mean_width_b",
        "mean_bias_diff", "mean_bias_a", "mean_bias_b",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "scenario": r.scenario,
                "eval_type": r.eval_type, "agreement_rate": r.agreement_rate,
                "n_items": r.n_items, "n_labeled": r.n_labeled, "method": r.method,
                "n_reps": r.n_reps,
                "coverage_diff": f"{r.coverage_diff:.4f}",
                "coverage_a": f"{r.coverage_a:.4f}",
                "coverage_b": f"{r.coverage_b:.4f}",
                "mean_width_diff": f"{r.mean_width_diff:.5f}",
                "mean_width_a":    f"{r.sum_w_a/r.n_reps:.5f}",
                "mean_width_b":    f"{r.sum_w_b/r.n_reps:.5f}",
                "mean_bias_diff":  f"{r.mean_bias_diff:.5f}",
                "mean_bias_a":     f"{r.sum_bias_a/r.n_reps:.5f}",
                "mean_bias_b":     f"{r.sum_bias_b/r.n_reps:.5f}",
            })
    print(f"  Saved CSV: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Official test battery
# ──────────────────────────────────────────────────────────────────────────────

def _run_official_test(n_workers: int, args: object) -> None:
    """Run the full reproducibility benchmark with fixed rigorous parameters.

    Designed to be stable, interpretable, and comparable across runs.

    Parameter rationale:
      reps=1000      MCSE ~ 0.007 at 95% coverage; stable enough to see ±2% deviations.
      n_mc=50        More MC draws stabilise Rubin's combining rules pooling.
      n_boot_inner=400  Tighter inner bootstrap CIs reduce variance in T (Rubin total var).
      n_boot=1000    Standard outer bootstrap; 1000 draws is already adequate.
      All 3 scenarios  standard/null/biased_llm cover random noise, Type I error,
                        and directional bias — the three regimes relevant for this comparison.
    """
    import io
    import contextlib

    OFFICIAL_REPS          = 1000
    OFFICIAL_N_ITEMS       = [50, 100, 200]
    OFFICIAL_N_LABELED     = [6, 15, 30, 60] #, 120]
    OFFICIAL_AR            = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    OFFICIAL_EVAL_TYPES    = ["binary", "likert", "continuous"]
    OFFICIAL_SCENARIOS     = ["standard", "null", "biased_llm"]
    OFFICIAL_N_BOOT        = 1000
    OFFICIAL_N_BOOT_CAL_PPI = getattr(args, "n_boot_cal_ppi", 200)
    OFFICIAL_N_MC          = 50
    OFFICIAL_N_BOOT_INNER  = 400
    OFFICIAL_ALPHA         = 0.05

    out_dir = Path(getattr(args, "out_dir", None) or "simulations/out")
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"sim_alignment_OFFICIAL_reps{OFFICIAL_REPS}_{ts}"

    OFFICIAL_METHODS = [m for m in METHODS if m != "mc_rubin"] 
    
    n_valid_cells = sum(
        1 for _ in OFFICIAL_SCENARIOS for _ in OFFICIAL_EVAL_TYPES
        for _ in OFFICIAL_AR
        for ni in OFFICIAL_N_ITEMS for nl in OFFICIAL_N_LABELED
        if nl <= ni
    )

    print()
    print("  sim_alignment_methods.py  ──  OFFICIAL TEST")
    print(f"  {'─'*60}")
    print(f"  Reps per cell  : {OFFICIAL_REPS}  (MCSE ~ {(0.95*0.05/OFFICIAL_REPS)**0.5:.3f})")
    print(f"  alpha          : {OFFICIAL_ALPHA}  (target coverage {1-OFFICIAL_ALPHA:.0%})")
    print(f"  n_boot (outer) : {OFFICIAL_N_BOOT}")
    print(f"  n_boot_cal_ppi : {OFFICIAL_N_BOOT_CAL_PPI}")
    print(f"  methods        : {OFFICIAL_METHODS}")
    print(f"  eval_types     : {OFFICIAL_EVAL_TYPES}")
    print(f"  n_items        : {OFFICIAL_N_ITEMS}")
    print(f"  n_labeled      : {OFFICIAL_N_LABELED}")
    print(f"  agreement rates: {OFFICIAL_AR}")
    print(f"  scenarios      : {OFFICIAL_SCENARIOS}")
    print(f"  valid cells    : {n_valid_cells}")
    print(f"  workers        : {n_workers}")
    print(f"  output dir     : {out_dir}")
    print(f"  {'─'*60}")
    print()

    t0 = time.time()
    results = run_simulation(
        eval_types=OFFICIAL_EVAL_TYPES,
        agreement_rates=OFFICIAL_AR,
        n_items_list=OFFICIAL_N_ITEMS,
        n_labeled_list=OFFICIAL_N_LABELED,
        methods=OFFICIAL_METHODS,
        n_reps=OFFICIAL_REPS,
        alpha=OFFICIAL_ALPHA,
        n_boot=OFFICIAL_N_BOOT,
        n_mc=OFFICIAL_N_MC,
        n_boot_inner=OFFICIAL_N_BOOT_INNER,
        progress=True,
        scenarios=OFFICIAL_SCENARIOS,
        n_workers=n_workers,
        n_boot_cal_ppi=OFFICIAL_N_BOOT_CAL_PPI,
    )
    elapsed = time.time() - t0
    print(f"  Simulation complete: {elapsed:.1f}s  ({elapsed/60:.1f} min)\n")

    print_report(results, alpha=OFFICIAL_ALPHA)
    print_summary_table(results, alpha=OFFICIAL_ALPHA)

    # Always save outputs for reproducibility
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(out_dir / f"{run_tag}_results.csv")
    log_path = str(out_dir / f"{run_tag}_summary.log")
    save_csv(results, csv_path)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(results, alpha=OFFICIAL_ALPHA)
        print_summary_table(results, alpha=OFFICIAL_ALPHA)
    with open(log_path, "w") as f:
        f.write(buf.getvalue())
    print(f"  Saved log: {log_path}")

    plots_mode = getattr(args, "plots", "save")
    if plots_mode in ("save", "show"):
        plots_dir = Path(getattr(args, "plots_dir", None) or str(out_dir / "plots"))
        for sc in OFFICIAL_SCENARIOS:
            plot_path = (
                str(plots_dir / f"{run_tag}_{sc}_plot.png")
                if plots_mode == "save" else None
            )
            make_plots(results, alpha=OFFICIAL_ALPHA, out_path=plot_path, scenario=sc)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate MC+Rubin's vs PPIBoot for LLM judge alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reps",         type=int,   default=500,
                        help="Monte Carlo repetitions per cell")
    parser.add_argument("--alpha",        type=float, default=0.05,
                        help="Significance level (1−alpha = CI nominal coverage)")
    parser.add_argument("--n-boot",       type=int,   default=1000,
                        help="Bootstrap samples for llm_only/human_only/oracle/ppi")
    parser.add_argument("--n-mc",         type=int,   default=30,
                        help="MC imputation draws for mc_rubin")
    parser.add_argument("--n-boot-inner", type=int,   default=200,
                        help="Inner bootstrap samples per MC draw for mc_rubin")
    parser.add_argument("--n-boot-cal-ppi", type=int, default=200,
                        help="Bootstrap samples for calibrated_ppi (sklearn loop — slower than vectorised methods)")
    parser.add_argument("--n-items",      type=int,   nargs="+", default=[50, 100, 200],
                        help="Items per model")
    parser.add_argument("--n-labeled",    type=int,   nargs="+", default=[15, 30, 60],
                        help="Total labeled items (split evenly across both models)")
    parser.add_argument("--agreement-rates", type=float, nargs="+",
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="LLM-human agreement rates to sweep")
    parser.add_argument("--eval-types",   type=str,   nargs="+",
                        default=["binary", "likert", "continuous"],
                        choices=["binary", "likert", "continuous"],
                        help="Score types to simulate")
    parser.add_argument("--methods",      type=str,   nargs="+", default=list(METHODS),
                        choices=METHODS,
                        help="Methods to compare (subset of: " + " ".join(METHODS) + ")")
    parser.add_argument("--scenarios",    type=str,   nargs="+", default=["standard"],
                        choices=SCENARIOS,
                        help=(
                            "Scenario variants to run. "
                            "'standard': random symmetric noise, delta!=0. "
                            "'null': delta=0, measures Type I error. "
                            "'biased_llm': directional LLM noise (lenient judge), tests debiasing. "
                            "Example: --scenarios standard null biased_llm"
                        ))
    parser.add_argument("--workers",      type=int,   default=1,
                        help="Parallel worker processes (default 1 = sequential). "
                             "Use --workers 0 to set to os.cpu_count().")
    parser.add_argument("--official-test", action="store_true",
                        help=(
                            "Run the full reproducibility benchmark (overrides all sweep "
                            "settings). Sweeps all 3 scenarios, all eval types, "
                            "n_items=[50,100,200], n_labeled=[15,30,60], 6 agreement rates, "
                            "1000 reps, n_mc=50, n_boot_inner=400. "
                            "Always saves to simulations/out (or --out-dir). "
                            "Recommended: --workers 0 (all cores). "
                            "Approx: ~35 min with 8 workers on an M2 Pro."
                        ))
    parser.add_argument("--quick",        action="store_true",
                        help="Quick smoke-test: --reps 50 --n-items 100 --n-labeled 30 "
                             "--eval-types binary --n-mc 20 --n-boot 200 --n-boot-inner 100 "
                             "--agreement-rates 0.5 0.7 0.9 1.0 --scenarios standard")
    parser.add_argument("--out-dir",      type=str,   default=None,
                        help="Directory to save outputs (CSV and log). Default: no save.")
    parser.add_argument("--plots",        type=str,   default="off", choices=["save", "show", "off"],
                        help="Whether to save/show plots")
    parser.add_argument("--plots-dir",    type=str,   default=None,
                        help="Directory for plots (defaults to --out-dir/plots)")
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    if args.official_test:
        _run_official_test(n_workers=n_workers, args=args)
        return

    if args.quick:
        args.reps            = 50
        args.n_items         = [100]
        args.n_labeled       = [30]
        args.eval_types      = ["binary"]
        args.n_mc            = 20
        args.n_boot          = 200
        args.n_boot_inner    = 100
        args.agreement_rates = [0.5, 0.7, 0.9, 1.0]
        args.scenarios       = ["standard"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    et_tag = "_".join(args.eval_types)
    sc_tag = "_".join(args.scenarios)
    run_tag = f"sim_alignment_{et_tag}_{sc_tag}_reps{args.reps}_{ts}"

    print()
    print("  sim_alignment_methods.py")
    print(f"  {'-'*60}")
    print(f"  Reps           : {args.reps}")
    print(f"  alpha          : {args.alpha}  (target coverage {1-args.alpha:.0%})")
    print(f"  n_boot (outer) : {args.n_boot}")
    print(f"  n_boot_cal_ppi : {args.n_boot_cal_ppi}")
    print(f"  n_mc           : {args.n_mc}")
    print(f"  n_boot_inner   : {args.n_boot_inner}")
    print(f"  eval_types     : {args.eval_types}")
    print(f"  n_items        : {args.n_items}")
    print(f"  n_labeled      : {args.n_labeled}")
    print(f"  agreement rates: {args.agreement_rates}")
    print(f"  methods        : {args.methods}")
    print(f"  scenarios      : {args.scenarios}")
    print(f"  workers        : {n_workers}")
    print(f"  {'-'*60}")
    print()

    t0 = time.time()
    results = run_simulation(
        eval_types=args.eval_types,
        agreement_rates=args.agreement_rates,
        n_items_list=args.n_items,
        n_labeled_list=args.n_labeled,
        methods=args.methods,
        n_reps=args.reps,
        alpha=args.alpha,
        n_boot=args.n_boot,
        n_mc=args.n_mc,
        n_boot_inner=args.n_boot_inner,
        progress=True,
        scenarios=args.scenarios,
        n_workers=n_workers,
        n_boot_cal_ppi=args.n_boot_cal_ppi,
    )
    elapsed = time.time() - t0
    print(f"  Simulation complete: {elapsed:.1f}s  ({elapsed/60:.1f} min)\n")

    print_report(results, alpha=args.alpha)
    print_summary_table(results, alpha=args.alpha)

    # Save outputs
    if args.out_dir:
        out_dir = Path(args.out_dir)
        csv_path = str(out_dir / f"{run_tag}_results.csv")
        log_path = str(out_dir / f"{run_tag}_summary.log")
        save_csv(results, csv_path)

        # Capture printed report to log
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(results, alpha=args.alpha)
            print_summary_table(results, alpha=args.alpha)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(buf.getvalue())
        print(f"  Saved log:  {log_path}")

    if args.plots in ("save", "show"):
        plots_dir = args.plots_dir or (str(Path(args.out_dir) / "plots") if args.out_dir else None)
        for sc in args.scenarios:
            if args.plots == "save" and plots_dir:
                plot_path = str(Path(plots_dir) / f"{run_tag}_{sc}_plot.png")
            else:
                plot_path = None
            make_plots(results, alpha=args.alpha, out_path=plot_path, scenario=sc)


if __name__ == "__main__":
    main()
