#!/usr/bin/env python3
"""
sim_tango_real.py — Tango CI method comparison on real binary benchmark data.

Validates tango-family paired-binary CI methods against real LLM eval data:

  1. Load binary benchmark corpora for multiple models from DOVE_Lite or OpenEval.
  2. For each (model_A, model_B, benchmark) triple: align scores on shared items
     to build a CorpusPair — the 'population'.
  3. True estimand = mean(scores_A) − mean(scores_B) over all shared items.
  4. Repeatedly subsample n paired items WITHOUT replacement and compute CIs.
  5. Check coverage against the corpus-level true_diff.

Methods compared
    Default (single-run)
        tango                 Tango score CI on single-run paired data
        newcombe              Newcombe-Wilson CI on single-run paired data
        bootstrap             Percentile bootstrap on paired diffs
        t_interval            Student's t on paired diffs
        plus bootstrap variants and Bayesian paired baselines

    Optional (--multi-run-methods)
        tango_multirun_cluster   Cluster-robust Tango
        tango_multirun_effective Effective-N Tango
        tango_multirun_mmnt      Moments-based Tango
        *_nested diff bootstrap methods

For R=1 (single-run real data), all multirun variants should match tango.
Divergence at R=1 would indicate a bug in a multirun extension.

──────────────────────────────────────────────────────────────────────────────
DOVE_Lite  (--source dove, default)
  https://huggingface.co/datasets/nlphuji/DOVE_Lite
  Binary benchmarks: hellaswag, arc_challenge, gsm8k, squad, quality
  Multiple models per benchmark → pairs formed from all combinations.

OpenEval  (--source openeval)
  https://huggingface.co/datasets/human-centered-eval/OpenEval
  Specify --models and --benchmarks; pairs formed from all model combinations
  per benchmark.  Models must share the same benchmark items (same item_ids).

──────────────────────────────────────────────────────────────────────────────
Usage
  # DOVE (default)
  python simulations/sim_tango_real.py
  python simulations/sim_tango_real.py --benchmarks hellaswag arc_challenge
  python simulations/sim_tango_real.py --models Llama-3.2-1B-Instruct Mistral-7B-Instruct-v0.3

  # OpenEval (default pairs are pre-confirmed; uses all default model lists)
  python simulations/sim_tango_real.py --source openeval
  python simulations/sim_tango_real.py --source openeval --benchmarks bbq mmlu-pro
  python simulations/sim_tango_real.py --source openeval --benchmarks bbq \\
      --models falcon-40b llama-7b

  # Shared options
  python simulations/sim_tango_real.py --reps 500
  python simulations/sim_tango_real.py --sizes 10 20 30 40 50 75 100
    python simulations/sim_tango_real.py --workers 8
  python simulations/sim_tango_real.py --out-dir simulations/out --plots save
  python simulations/sim_tango_real.py --alpha 0.10
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_HERE, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.resampling import (
        bootstrap_ci_1d,
        t_interval_ci_1d,
        newcombe_paired_ci,
        tango_paired_ci_flat,
        tango_paired_ci_multirun_cluster,
        tango_paired_ci_multirun_effective,
        tango_paired_ci_multirun_moments,
        bootstrap_diffs_nested,
        bayes_bootstrap_diffs_nested,
        smooth_bootstrap_diffs_nested,
    )
    from evalstats.core.bayes_evals import binorm_cdf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TANGO_FLAT_METHOD      = "tango"
NEWCOMBE_FLAT_METHOD   = "newcombe"
TANGO_CLUSTER_METHOD   = "tango_multirun_cluster"
TANGO_EFFECTIVE_METHOD = "tango_multirun_effective"
TANGO_MOMENTS_METHOD   = "tango_multirun_mmnt"
BOOTSTRAP_METHOD       = "bootstrap"
BCA_METHOD             = "bca"
BAYES_BOOTSTRAP_METHOD = "bayes_bootstrap"
SMOOTH_BOOTSTRAP_METHOD = "smooth_bootstrap"
BOOTSTRAP_T_METHOD     = "bootstrap_t"
T_INTERVAL_METHOD      = "t_interval"
BOOTSTRAP_DIFF_NESTED_METHOD = "bootstrap_diff_nested"
BAYES_DIFF_NESTED_METHOD     = "bayes_diff_nested"
SMOOTH_DIFF_NESTED_METHOD    = "smooth_diff_nested"
BAYES_PAIR_INDEP_METHOD      = "bayes_indep_comp"
BAYES_PAIR_PAIRED_METHOD     = "bayes_paired_comp"

ALL_METHODS = [
    TANGO_FLAT_METHOD,
    NEWCOMBE_FLAT_METHOD,
    TANGO_CLUSTER_METHOD,
    TANGO_EFFECTIVE_METHOD,
    TANGO_MOMENTS_METHOD,
    BOOTSTRAP_METHOD,
    BCA_METHOD,
    BAYES_BOOTSTRAP_METHOD,
    SMOOTH_BOOTSTRAP_METHOD,
    BOOTSTRAP_T_METHOD,
    T_INTERVAL_METHOD,
    BOOTSTRAP_DIFF_NESTED_METHOD,
    BAYES_DIFF_NESTED_METHOD,
    SMOOTH_DIFF_NESTED_METHOD,
    BAYES_PAIR_INDEP_METHOD,
    BAYES_PAIR_PAIRED_METHOD,
]

SINGLE_RUN_METHODS = [
    TANGO_FLAT_METHOD,
    NEWCOMBE_FLAT_METHOD,
    BOOTSTRAP_METHOD,
    BCA_METHOD,
    BAYES_BOOTSTRAP_METHOD,
    SMOOTH_BOOTSTRAP_METHOD,
    BOOTSTRAP_T_METHOD,
    T_INTERVAL_METHOD,
    BAYES_PAIR_INDEP_METHOD,
    BAYES_PAIR_PAIRED_METHOD,
]

MULTI_RUN_ONLY_METHODS = [
    TANGO_CLUSTER_METHOD,
    TANGO_EFFECTIVE_METHOD,
    TANGO_MOMENTS_METHOD,
    BOOTSTRAP_DIFF_NESTED_METHOD,
    BAYES_DIFF_NESTED_METHOD,
    SMOOTH_DIFF_NESTED_METHOD,
]

_METHOD_COLORS: dict[str, str] = {
    TANGO_FLAT_METHOD:           "#e7298a",
    NEWCOMBE_FLAT_METHOD:        "#66a61e",
    TANGO_CLUSTER_METHOD:        "#e6ab02",
    TANGO_EFFECTIVE_METHOD:      "#a6761d",
    TANGO_MOMENTS_METHOD:        "#1b9e77",
    BOOTSTRAP_METHOD:            "#1f77b4",
    BCA_METHOD:                  "#2ca02c",
    BAYES_BOOTSTRAP_METHOD:      "#ff7f0e",
    SMOOTH_BOOTSTRAP_METHOD:     "#9467bd",
    BOOTSTRAP_T_METHOD:          "#d62728",
    T_INTERVAL_METHOD:           "#8c564b",
    BOOTSTRAP_DIFF_NESTED_METHOD:"#1b9e77",
    BAYES_DIFF_NESTED_METHOD:    "#d95f02",
    SMOOTH_DIFF_NESTED_METHOD:   "#7570b3",
    BAYES_PAIR_INDEP_METHOD:     "#17becf",
    BAYES_PAIR_PAIRED_METHOD:    "#bcbd22",
}

_METHOD_LABELS: dict[str, str] = {
    TANGO_FLAT_METHOD:           "tango",
    NEWCOMBE_FLAT_METHOD:        "newcombe",
    TANGO_CLUSTER_METHOD:        "tango_cluster",
    TANGO_EFFECTIVE_METHOD:      "tango_effective",
    TANGO_MOMENTS_METHOD:        "tango_moments",
    BOOTSTRAP_METHOD:            "bootstrap",
    BCA_METHOD:                  "bca",
    BAYES_BOOTSTRAP_METHOD:      "bayes_bootstrap",
    SMOOTH_BOOTSTRAP_METHOD:     "smooth_bootstrap",
    BOOTSTRAP_T_METHOD:          "bootstrap_t",
    T_INTERVAL_METHOD:           "t_interval",
    BOOTSTRAP_DIFF_NESTED_METHOD:"bootstrap_diff_nested",
    BAYES_DIFF_NESTED_METHOD:    "bayes_diff_nested",
    SMOOTH_DIFF_NESTED_METHOD:   "smooth_diff_nested",
    BAYES_PAIR_INDEP_METHOD:     "bayes_indep_comp",
    BAYES_PAIR_PAIRED_METHOD:    "bayes_paired_comp",
}

SOURCES = ["dove", "openeval"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]
PROGRESS_MODES = ["bar", "cell", "off"]

DEFAULT_SIZES = [10, 20, 30, 40, 50, 75, 100]
DEFAULT_REPS = 1000
DEFAULT_BOOTSTRAP_N = 2000
DEFAULT_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Pairwise Bayesian helper functions
# ---------------------------------------------------------------------------


def _bayes_indep_comp_ci(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Independent Beta-posteriors CI for paired binary difference p(A=1)-p(B=1)."""
    post_a = rng.beta(float(np.sum(a)) + 1.0, float(len(a) - np.sum(a)) + 1.0, size=num_samples)
    post_b = rng.beta(float(np.sum(b)) + 1.0, float(len(b) - np.sum(b)) + 1.0, size=num_samples)
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
    """Paired Bayesian CI for p(A=1)-p(B=1) using the bivariate-normal latent model."""
    s = float(np.sum(a * b))
    t = float(np.sum(a * (1.0 - b)))
    u = float(np.sum((1.0 - a) * b))
    v = float(np.sum((1.0 - a) * (1.0 - b)))

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
        d_hat = float(np.mean(a) - np.mean(b))
        return d_hat, d_hat

    w /= w_sum
    diff_post = diff[rng.choice(num_samples, size=num_samples, replace=True, p=w)]
    return (
        float(np.percentile(diff_post, 100.0 * alpha / 2.0)),
        float(np.percentile(diff_post, 100.0 * (1.0 - alpha / 2.0))),
    )


# ---------------------------------------------------------------------------
# Corpus descriptive statistics helpers
# ---------------------------------------------------------------------------


def _estimate_icc_binary_pairs(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Estimate ICC(1,1) for binary paired data via one-way random-effects ANOVA.

    Treats each item as a "group" with two observations (A and B score).  The
    formula for k=2 observations per group is::

        ICC = (MS_between - MS_within) / (MS_between + MS_within)

    where::

        item_mean_i  = (a_i + b_i) / 2
        grand_mean   = mean of all scores
        SS_between   = 2 * sum((item_mean_i - grand_mean)^2)
        SS_within    = sum((a_i - b_i)^2) / 2
        MS_between   = SS_between / (n - 1)
        MS_within    = SS_within / n

    Returns a value in [-1, 1]; negative values are clipped to 0 for display.
    """
    n = len(scores_a)
    if n < 2:
        return float("nan")
    grand_mean = float(np.mean(np.concatenate([scores_a, scores_b])))
    item_means = (scores_a + scores_b) / 2.0
    ss_between = 2.0 * float(np.sum((item_means - grand_mean) ** 2))
    ss_within  = float(np.sum((scores_a - scores_b) ** 2)) / 2.0
    ms_between = ss_between / (n - 1)
    ms_within  = ss_within  / n
    denom = ms_between + ms_within
    if denom <= 0.0:
        return float("nan")
    return float((ms_between - ms_within) / denom)


def print_corpus_pair_stats(corpus_pairs: list) -> None:
    """Print descriptive statistics for each corpus pair including ICC estimate."""
    if not corpus_pairs:
        return

    print(f"\n  {'─'*76}")
    print(f"  Corpus pair descriptive statistics")
    print(f"  {'─'*76}")
    header = (
        f"  {'Pair (A vs B)':<44}  {'N':>6}  {'p_A':>5}  {'p_B':>5}  "
        f"{'diff':>6}  {'disc%':>5}  {'ICC':>5}"
    )
    print(header)
    print(f"  {'─'*76}")

    for cp in corpus_pairs:
        a = cp.scores_a
        b = cp.scores_b
        n = cp.corpus_size
        p_a = float(np.mean(a))
        p_b = float(np.mean(b))
        n10 = int(np.sum((a == 1) & (b == 0)))
        n01 = int(np.sum((a == 0) & (b == 1)))
        disc_pct = 100.0 * (n10 + n01) / n
        icc = _estimate_icc_binary_pairs(a, b)
        icc_str = f"{icc:.3f}" if np.isfinite(icc) else "  n/a"
        pair_label = f"{cp.model_a[:20]} vs {cp.model_b[:20]}"
        print(
            f"  {pair_label:<44}  {n:>6,}  {p_a:>5.3f}  {p_b:>5.3f}  "
            f"{cp.true_diff:>+6.3f}  {disc_pct:>5.1f}  {icc_str:>5}"
        )
        # Expected discordant pairs at small N (illustrative)
        exp_disc_20 = 20 * (n10 + n01) / n
        exp_disc_40 = 40 * (n10 + n01) / n
        print(
            f"    → disc pairs: n10={n10}, n01={n01}; "
            f"E[disc|n=20]={exp_disc_20:.1f}, E[disc|n=40]={exp_disc_40:.1f}"
        )

    print(f"  {'─'*76}\n")


# ---------------------------------------------------------------------------
# DOVE constants (binary benchmarks only)
# ---------------------------------------------------------------------------

DOVE_REPO = "nlphuji/DOVE_Lite"

DOVE_BINARY_BENCHMARKS = ["hellaswag", "arc_challenge", "gsm8k", "squad", "quality"]

# File specs for binary DOVE benchmarks.
_DOVE_FILE: dict[str, str] = {
    "hellaswag":   "hellaswag.parquet",
    "arc_challenge": "ai2_arc.arc_challenge.parquet",
    "gsm8k":       "gsm8k.parquet",
    "squad":       "squad.parquet",
    "quality":     "quality.parquet",
}

# Default (model, benchmark) pairs to load.  Multiple models per benchmark
# → pairs are formed from all combinations.
DOVE_DEFAULT_MODEL_BENCH: list[tuple[str, str]] = [
    ("Llama-3.2-1B-Instruct",     "hellaswag"),
    ("OLMoE-1B-7B-0924-Instruct", "hellaswag"),
    ("Mistral-7B-Instruct-v0.3",  "hellaswag"),
    ("Meta-Llama-3-8B-Instruct",  "hellaswag"),
    ("Llama-3.2-1B-Instruct",     "arc_challenge"),
    ("OLMoE-1B-7B-0924-Instruct", "arc_challenge"),
    ("Mistral-7B-Instruct-v0.3",  "arc_challenge"),
    ("Meta-Llama-3-8B-Instruct",  "arc_challenge"),
]


# ---------------------------------------------------------------------------
# OpenEval constants
# ---------------------------------------------------------------------------

OPENEVAL_REPO = "human-centered-eval/OpenEval"

# Binary benchmark specs: eval_type must be "binary".
OPENEVAL_BINARY_SPECS: dict[str, dict] = {
    "mmlu-pro":      {"metric_name": None, "score_scale": 1.0},
    "gpqa":          {"metric_name": None, "score_scale": 1.0},
    "boolq":         {"metric_name": None, "score_scale": 1.0},
    "imdb":          {"metric_name": None, "score_scale": 1.0},
    "culturalbench": {"metric_name": None, "score_scale": 1.0},
    "opentom":       {"metric_name": None, "score_scale": 1.0},
    "bbq":           {"metric_name": None, "score_scale": 1.0},
    "hi-tom":        {"metric_name": None, "score_scale": 1.0},
    "omni-math":     {"metric_name": None, "score_scale": 1.0},
    "emobench":      {"metric_name": None, "score_scale": 1.0},
    "salad-bench":   {"metric_name": None, "score_scale": 1.0},
}

# Default (model, benchmark) pairs confirmed to have data in OpenEval.
# Coverage is sparse — do NOT use the full models × benchmarks cross-product.
# Each benchmark lists the specific models known to have responses for it.
OPENEVAL_DEFAULT_MODEL_BENCH: list[tuple[str, str]] = [
    # bbq — social-bias QA
    ("falcon-40b",                "bbq"),
    ("llama-7b",                  "bbq"),
    ("qwen-2.5-32b-instruct",     "bbq"),
    # mmlu-pro — knowledge/reasoning
    ("gemma-2-27b-it",            "mmlu-pro"),
    ("gemma-3-4b-it",             "mmlu-pro"),
    ("qwen-3-30b-instruct",       "mmlu-pro"),
    ("llama-2-70b-hf",            "mmlu-pro"),
    # culturalbench — cultural knowledge
    ("DeepSeek-R1",               "culturalbench"),
    ("gpt-4o",                    "culturalbench"),
    ("gpt-4.1-mini",              "culturalbench"),
    ("o4-mini",                   "culturalbench"),
    ("grok-4",                    "culturalbench"),
    # hi-tom — higher-order theory-of-mind
    ("gpt-4o-mini",               "hi-tom"),
    ("grok-4",                    "hi-tom"),
    ("phi-4",                     "hi-tom"),
    ("DeepSeek-R1",               "hi-tom"),
    # opentom — theory-of-mind
    ("DeepSeek-R1",               "opentom"),
    ("gpt-4.1",                   "opentom"),
    ("o1",                        "opentom"),
    # salad-bench — safety/alignment
    ("DeepSeek-R1",               "salad-bench"),
    ("grok-4",                    "salad-bench"),
    ("kimi-k2",                   "salad-bench"),
    ("phi-4",                     "salad-bench"),
    # omni-math — mathematical reasoning
    ("gemma-3-27b-it",           "omni-math"),
    ("qwen-2.5-14b-instruct",    "omni-math"),
    ("qwen-2.5-72b-instruct",    "omni-math"),
    ("llama-2-13b-hf",           "omni-math"),
]

_OE_RESP_ID_RE = re.compile(r"^(.+?)_(\d{8}T\d{6}Z)_(\d+)")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CorpusPair:
    """Aligned paired scores for two models on the same benchmark."""
    model_a: str
    model_b: str
    benchmark_id: str
    source: str          # "dove" | "openeval"
    scores_a: np.ndarray  # shape (N_shared,)
    scores_b: np.ndarray  # shape (N_shared,)
    true_diff: float      # mean(scores_a − scores_b) — population ground truth
    corpus_size: int      # N_shared = number of shared items


@dataclass
class PairSimResult:
    model_a: str
    model_b: str
    benchmark_id: str
    source: str
    corpus_size: int
    true_diff: float
    n: int
    method: str
    n_reps: int
    covered: int
    total_width: float
    total_time: float = 0.0
    total_time_sq: float = 0.0


_WORKER_CORPUS_PAIRS: list[CorpusPair] = []
_WORKER_METHODS: list[str] = []


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------


class _ProgressReporter:
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
            f"\r  {prefix}[{bar}] {100.0*frac:6.2f}%  {step:>7d}/{self.total:<7d}  "
            f"ETA {eta_str}  {detail[:40]:<40s}",
            end="", flush=True,
        )
        if is_final:
            print()


# ---------------------------------------------------------------------------
# DOVE loading helpers
# ---------------------------------------------------------------------------


def _dove_load_item_scores(
    model: str,
    benchmark_id: str,
    *,
    dove_repo: str = DOVE_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
) -> dict[int, float] | None:
    """Load DOVE_Lite for one (model, benchmark) and return {sample_index: score}.

    Returns None if the file is not found or score extraction fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    fname = _DOVE_FILE.get(benchmark_id)
    if fname is None:
        print(f"  Skip  {model}/{benchmark_id}: not a supported binary benchmark")
        return None

    # Try shot-count variants
    path_candidates: list[str] = []
    for shots in (0, 5, 2, 3):
        path_candidates.append(f"{model}/en/{shots}_shot/{fname}")
    path_candidates.append(f"{model}/en/5_shots/{fname}")

    dataset = None
    for fp in path_candidates:
        try:
            dataset = load_dataset(
                dove_repo, data_files=fp, split="train",
                token=hf_token, cache_dir=cache_dir,
            )
            break
        except Exception:
            continue

    if dataset is None:
        print(f"  Skip  {model}/{benchmark_id}: no DOVE file found")
        return None

    # Extract scores
    col_names = dataset.column_names
    raw_scores: list[float] = []
    if "evaluation" in col_names:
        eval_col = dataset["evaluation"]
        if isinstance(eval_col, list) and eval_col and isinstance(eval_col[0], dict):
            raw_scores = [float(e.get("score", float("nan"))) for e in eval_col]
        else:
            try:
                raw_scores = [float(v) for v in dataset["evaluation"]["score"]]
            except Exception:
                pass
    if not raw_scores and "score" in col_names:
        raw_scores = [float(v) for v in dataset["score"]]
    if not raw_scores:
        print(f"  Skip  {model}/{benchmark_id}: score extraction failed")
        return None

    # Extract sample indices
    indices: list[int] | None = None
    if "sample_index" in col_names:
        try:
            indices = [int(v) for v in dataset["sample_index"]]
        except Exception:
            pass
    if indices is None and "instance" in col_names:
        try:
            inst_col = dataset["instance"]
            if isinstance(inst_col, list) and inst_col and isinstance(inst_col[0], dict):
                indices = [
                    int(inst["sample_identifier"]["hf_index"])
                    for inst in inst_col
                ]
        except Exception:
            pass

    if indices is None:
        # No index info — use position as index (all items unique)
        return {i: s for i, s in enumerate(raw_scores) if np.isfinite(s)}

    # Keep first occurrence per sample_index (deduplicate prompt perturbations)
    result: dict[int, float] = {}
    for idx, score in zip(indices, raw_scores):
        if idx not in result and np.isfinite(score):
            result[idx] = score
    return result


def build_dove_corpus_pairs(
    model_bench_pairs: list[tuple[str, str]],
    *,
    dove_repo: str = DOVE_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_pair_size: int = 50,
) -> list[CorpusPair]:
    """Build CorpusPairs from DOVE_Lite by loading and aligning on sample_index.

    Groups (model, benchmark) pairs by benchmark, loads each model's
    item-score mapping, then forms all model-pair combinations per benchmark
    by intersecting shared sample indices.
    """
    # Group by benchmark
    bench_models: dict[str, list[str]] = defaultdict(list)
    for model, bench in model_bench_pairs:
        if bench in DOVE_BINARY_BENCHMARKS:
            bench_models[bench].append(model)
        else:
            print(f"  Warning: {bench} is not a supported binary DOVE benchmark. Skipping.")

    corpus_pairs: list[CorpusPair] = []

    for bench, models in bench_models.items():
        print(f"\nLoading DOVE_Lite: benchmark={bench}, models={models} …")
        # Load item-score maps for all models on this benchmark
        item_maps: dict[str, dict[int, float]] = {}
        for model in models:
            m = _dove_load_item_scores(
                model, bench,
                dove_repo=dove_repo, hf_token=hf_token, cache_dir=cache_dir,
            )
            if m is not None:
                print(f"  OK    {model}/{bench}: {len(m)} items")
                item_maps[model] = m
            # else: already printed a skip message

        if len(item_maps) < 2:
            print(f"  Skip  {bench}: fewer than 2 models loaded successfully.")
            continue

        # Form all model-pair combinations
        loaded_models = list(item_maps.keys())
        for model_a, model_b in combinations(loaded_models, 2):
            shared_keys = sorted(item_maps[model_a].keys() & item_maps[model_b].keys())
            if len(shared_keys) < min_pair_size:
                print(
                    f"  Skip  ({model_a}, {model_b}) on {bench}: "
                    f"only {len(shared_keys)} shared items < {min_pair_size}"
                )
                continue
            scores_a = np.array([item_maps[model_a][k] for k in shared_keys])
            scores_b = np.array([item_maps[model_b][k] for k in shared_keys])
            true_diff = float(np.mean(scores_a - scores_b))
            print(
                f"  Pair  ({model_a} vs {model_b}) on {bench}: "
                f"N={len(shared_keys)}, mean_A={np.mean(scores_a):.4f}, "
                f"mean_B={np.mean(scores_b):.4f}, true_diff={true_diff:+.4f}"
            )
            corpus_pairs.append(CorpusPair(
                model_a=model_a, model_b=model_b, benchmark_id=bench, source="dove",
                scores_a=scores_a, scores_b=scores_b,
                true_diff=true_diff, corpus_size=len(shared_keys),
            ))

    print(f"\n  {len(corpus_pairs)} corpus pairs built from DOVE_Lite.\n")
    return corpus_pairs


# ---------------------------------------------------------------------------
# OpenEval loading helpers
# ---------------------------------------------------------------------------


def _oe_parse(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return val
    return val


def _oe_get_model_name(model_val: Any) -> str | None:
    obj = _oe_parse(model_val)
    if isinstance(obj, dict):
        return obj.get("name")
    return None


def _oe_parse_response_id(response_id: str) -> tuple[str, str] | tuple[None, None]:
    m = _OE_RESP_ID_RE.match(response_id)
    if m is None:
        return None, None
    source = m.group(1)
    item_id = f"{source}_{m.group(2)}_{m.group(3)}"
    return source, item_id


def _oe_extract_score(scores_val: Any, metric_name: str | None) -> float | None:
    data = _oe_parse(scores_val)
    if isinstance(data, list):
        if not data:
            return None
        if metric_name is None:
            entry = data[0]
        else:
            def _name(e: Any) -> str | None:
                if not isinstance(e, dict):
                    return None
                m = _oe_parse(e.get("metric"))
                return m.get("name") if isinstance(m, dict) else None
            entry = next((e for e in data if _name(e) == metric_name), None)
            if entry is None:
                return None
        if not isinstance(entry, dict):
            return None
        val = entry.get("value")
    elif isinstance(data, dict):
        metrics_raw = _oe_parse(data.get("metric"))
        value_raw = data.get("value")
        if isinstance(metrics_raw, list) and isinstance(value_raw, list):
            if not value_raw:
                return None
            if metric_name is None:
                val = value_raw[0]
            else:
                val = None
                for m_obj, v in zip(metrics_raw, value_raw):
                    m_obj = _oe_parse(m_obj)
                    if isinstance(m_obj, dict) and m_obj.get("name") == metric_name:
                        val = v
                        break
        else:
            val = value_raw
    else:
        return None
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def build_openeval_corpus_pairs(
    model_bench_pairs: list[tuple[str, str]],
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_pair_size: int = 50,
) -> list[CorpusPair]:
    """Build CorpusPairs from OpenEval by aligning on item_id.

    Loads the response table once, filters to the given (model, benchmark)
    pairs, accumulates item→score maps per (model, benchmark), then forms all
    model-pair combinations per benchmark by intersecting shared item_ids.

    Coverage in OpenEval is sparse — pass only confirmed (model, benchmark)
    pairs; do NOT expand a cross-product of all models × all benchmarks.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    unknown_benches = [b for _, b in model_bench_pairs if b not in OPENEVAL_BINARY_SPECS]
    if unknown_benches:
        print(
            f"Warning: unsupported OpenEval binary benchmark IDs: {sorted(set(unknown_benches))}.\n"
            f"  Supported: {list(OPENEVAL_BINARY_SPECS)}"
        )
        model_bench_pairs = [(m, b) for m, b in model_bench_pairs if b in OPENEVAL_BINARY_SPECS]
    if not model_bench_pairs:
        return []

    pairs_set  = set(model_bench_pairs)
    bench_set  = {b for _, b in model_bench_pairs}
    models     = list(dict.fromkeys(m for m, _ in model_bench_pairs))

    print("Loading OpenEval response table (~1.4 GB; cached after first download) …")
    response_ds = load_dataset(
        openeval_repo, "response", split="train",
        token=hf_token, cache_dir=cache_dir,
    )

    def _keep_row(batch: dict) -> list[bool]:
        keep = []
        for rid, model_val in zip(batch["response_id"], batch["model"]):
            source, _ = _oe_parse_response_id(rid)
            if source not in bench_set:
                keep.append(False)
                continue
            mname = _oe_get_model_name(model_val)
            keep.append((mname, source) in pairs_set)
        return keep

    response_ds = response_ds.filter(_keep_row, batched=True, batch_size=5_000)
    print(f"  {len(response_ds):,} responses after filtering.")

    # Accumulate item_id → score per (model, benchmark)
    item_maps: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    n_dedup = 0
    for row in response_ds:
        rid = row.get("response_id", "")
        source, item_id = _oe_parse_response_id(rid)
        if source is None:
            continue
        mname = _oe_get_model_name(row.get("model"))
        if mname is None or (mname, source) not in pairs_set:
            continue
        key = (mname, source)
        if item_id in item_maps[key]:
            n_dedup += 1
            continue
        spec = OPENEVAL_BINARY_SPECS[source]
        score = _oe_extract_score(row.get("scores"), spec["metric_name"])
        if score is not None and np.isfinite(score):
            item_maps[key][item_id] = float(score) * spec["score_scale"]

    if n_dedup > 0:
        print(f"  {n_dedup:,} duplicate rows removed (kept first per item × model).")

    # ── Binary-score normalization per (model, benchmark) ─────────────────────
    for (model, bench), scores_map in list(item_maps.items()):
        keys = list(scores_map.keys())
        vals = np.array([scores_map[k] for k in keys], dtype=float)
        non_binary_mask = ~np.isin(vals, [0.0, 1.0])
        if np.any(non_binary_mask):
            # OpenEval binary tasks occasionally contain near-binary float values.
            # Round to nearest integer then clip to enforce {0,1}.
            rounded_vals = np.clip(np.rint(vals), 0.0, 1.0)
            unique_bad = np.unique(vals[non_binary_mask])[:5]
            print(
                f"  Warning: {model}/{bench} has {int(np.sum(non_binary_mask)):,} non-binary scores "
                f"(e.g. {unique_bad}). Rounded to {{0,1}}."
            )
            item_maps[(model, bench)] = {k: float(v) for k, v in zip(keys, rounded_vals)}

    corpus_pairs: list[CorpusPair] = []
    for bench in sorted(bench_set):
        # Only models requested for this specific benchmark (preserves input order).
        requested = list(dict.fromkeys(m for m, b in model_bench_pairs if b == bench))
        bench_models = [m for m in requested if (m, bench) in item_maps and item_maps[(m, bench)]]
        if len(bench_models) < 2:
            print(f"  Skip  {bench}: fewer than 2 models with data.")
            continue
        print(f"\n  Benchmark: {bench}")
        for model in bench_models:
            n = len(item_maps[(model, bench)])
            print(f"    {model}: {n:,} items")
        for model_a, model_b in combinations(bench_models, 2):
            map_a = item_maps[(model_a, bench)]
            map_b = item_maps[(model_b, bench)]
            shared_ids = sorted(map_a.keys() & map_b.keys())
            if len(shared_ids) < min_pair_size:
                print(
                    f"  Skip  ({model_a}, {model_b}) on {bench}: "
                    f"{len(shared_ids)} shared items < {min_pair_size}"
                )
                continue
            scores_a = np.array([map_a[k] for k in shared_ids])
            scores_b = np.array([map_b[k] for k in shared_ids])
            # Enforce binary values on the aligned pair (hard guard).
            bad_a = scores_a[~np.isin(scores_a, [0.0, 1.0])]
            bad_b = scores_b[~np.isin(scores_b, [0.0, 1.0])]
            if len(bad_a) > 0 or len(bad_b) > 0:
                print(
                    f"  Skip  ({model_a} vs {model_b}) on {bench}: "
                    f"non-binary scores found after alignment "
                    f"({len(bad_a)} in A, {len(bad_b)} in B). Skipping pair."
                )
                continue
            true_diff = float(np.mean(scores_a - scores_b))
            print(
                f"  Pair  ({model_a} vs {model_b}): N={len(shared_ids)}, "
                f"mean_A={np.mean(scores_a):.4f}, mean_B={np.mean(scores_b):.4f}, "
                f"true_diff={true_diff:+.4f}"
            )
            corpus_pairs.append(CorpusPair(
                model_a=model_a, model_b=model_b, benchmark_id=bench, source="openeval",
                scores_a=scores_a, scores_b=scores_b,
                true_diff=true_diff, corpus_size=len(shared_ids),
            ))

    print(f"\n  {len(corpus_pairs)} corpus pairs built from OpenEval.\n")
    return corpus_pairs


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def _run_pairwise_real_cell(
    args: tuple[int, int, int, int, float, list[int]],
) -> list[PairSimResult]:
    cp_idx, n, n_reps, n_bootstrap, alpha, seed_state = args
    cp = _WORKER_CORPUS_PAIRS[cp_idx]

    rng = np.random.default_rng(np.random.SeedSequence(seed_state))

    N = cp.corpus_size
    true_diff = cp.true_diff
    methods = _WORKER_METHODS or ALL_METHODS
    covered: dict[str, int] = {m: 0 for m in methods}
    total_w: dict[str, float] = {m: 0.0 for m in methods}
    total_t: dict[str, float] = {m: 0.0 for m in methods}
    total_t_sq: dict[str, float] = {m: 0.0 for m in methods}

    for _ in range(n_reps):
        idxs = rng.choice(N, size=n, replace=False)
        # Shape (n, 1): treat as single-run matrices for multirun variants.
        a = cp.scores_a[idxs].reshape(n, 1)
        b = cp.scores_b[idxs].reshape(n, 1)
        diffs = a[:, 0] - b[:, 0]
        obs_diff = float(np.mean(diffs))

        # ── tango_flat ──────────────────────────────────────────────
        if TANGO_FLAT_METHOD in methods:
            _t = time.perf_counter()
            try:
                ci_lo, ci_hi = tango_paired_ci_flat(a, b, alpha)
            except Exception:
                ci_lo = ci_hi = obs_diff
            _el = time.perf_counter() - _t
            total_t[TANGO_FLAT_METHOD] += _el
            total_t_sq[TANGO_FLAT_METHOD] += _el * _el
            if ci_lo <= true_diff <= ci_hi:
                covered[TANGO_FLAT_METHOD] += 1
            total_w[TANGO_FLAT_METHOD] += ci_hi - ci_lo

        # ── newcombe_flat ────────────────────────────────────────────
        if NEWCOMBE_FLAT_METHOD in methods:
            _t = time.perf_counter()
            try:
                ci_lo, ci_hi = newcombe_paired_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                ci_lo = ci_hi = obs_diff
            _el = time.perf_counter() - _t
            total_t[NEWCOMBE_FLAT_METHOD] += _el
            total_t_sq[NEWCOMBE_FLAT_METHOD] += _el * _el
            if ci_lo <= true_diff <= ci_hi:
                covered[NEWCOMBE_FLAT_METHOD] += 1
            total_w[NEWCOMBE_FLAT_METHOD] += ci_hi - ci_lo

        # ── tango_multirun_cluster ───────────────────────────────────
        if TANGO_CLUSTER_METHOD in methods:
            _t = time.perf_counter()
            try:
                ci_lo, ci_hi = tango_paired_ci_multirun_cluster(a, b, alpha)
            except Exception:
                ci_lo = ci_hi = obs_diff
            _el = time.perf_counter() - _t
            total_t[TANGO_CLUSTER_METHOD] += _el
            total_t_sq[TANGO_CLUSTER_METHOD] += _el * _el
            if ci_lo <= true_diff <= ci_hi:
                covered[TANGO_CLUSTER_METHOD] += 1
            total_w[TANGO_CLUSTER_METHOD] += ci_hi - ci_lo

        # ── tango_multirun_effective ─────────────────────────────────
        if TANGO_EFFECTIVE_METHOD in methods:
            _t = time.perf_counter()
            try:
                ci_lo, ci_hi = tango_paired_ci_multirun_effective(a, b, alpha)
            except Exception:
                ci_lo = ci_hi = obs_diff
            _el = time.perf_counter() - _t
            total_t[TANGO_EFFECTIVE_METHOD] += _el
            total_t_sq[TANGO_EFFECTIVE_METHOD] += _el * _el
            if ci_lo <= true_diff <= ci_hi:
                covered[TANGO_EFFECTIVE_METHOD] += 1
            total_w[TANGO_EFFECTIVE_METHOD] += ci_hi - ci_lo

        # ── tango_multirun_mmnt ──────────────────────────────────────
        if TANGO_MOMENTS_METHOD in methods:
            _t = time.perf_counter()
            try:
                ci_lo, ci_hi = tango_paired_ci_multirun_moments(a, b, alpha)
            except Exception:
                ci_lo = ci_hi = obs_diff
            _el = time.perf_counter() - _t
            total_t[TANGO_MOMENTS_METHOD] += _el
            total_t_sq[TANGO_MOMENTS_METHOD] += _el * _el
            if ci_lo <= true_diff <= ci_hi:
                covered[TANGO_MOMENTS_METHOD] += 1
            total_w[TANGO_MOMENTS_METHOD] += ci_hi - ci_lo

        # ── bootstrap family on paired diffs (cell-mean diffs, R=1) ──
        for _method in [
            BOOTSTRAP_METHOD,
            BCA_METHOD,
            BAYES_BOOTSTRAP_METHOD,
            SMOOTH_BOOTSTRAP_METHOD,
            BOOTSTRAP_T_METHOD,
        ]:
            if _method not in methods:
                continue
            _t = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ci_lo, ci_hi = bootstrap_ci_1d(
                        diffs,
                        obs_diff,
                        method=_method,
                        n_bootstrap=n_bootstrap,
                        alpha=alpha,
                        rng=rng,
                    )
            except Exception:
                ci_lo = ci_hi = obs_diff
            _el = time.perf_counter() - _t
            total_t[_method] += _el
            total_t_sq[_method] += _el * _el
            if ci_lo <= true_diff <= ci_hi:
                covered[_method] += 1
            total_w[_method] += ci_hi - ci_lo

        # ── t_interval on paired diffs ───────────────────────────────
        if T_INTERVAL_METHOD in methods:
            _t = time.perf_counter()
            try:
                ci_lo, ci_hi = t_interval_ci_1d(diffs, alpha)
            except Exception:
                ci_lo = ci_hi = obs_diff
            _el = time.perf_counter() - _t
            total_t[T_INTERVAL_METHOD] += _el
            total_t_sq[T_INTERVAL_METHOD] += _el * _el
            if ci_lo <= true_diff <= ci_hi:
                covered[T_INTERVAL_METHOD] += 1
            total_w[T_INTERVAL_METHOD] += ci_hi - ci_lo

        # ── nested diff bootstraps on (n, 1) pair matrices ───────────
        for _method, _fn in [
            (BOOTSTRAP_DIFF_NESTED_METHOD, bootstrap_diffs_nested),
            (BAYES_DIFF_NESTED_METHOD, bayes_bootstrap_diffs_nested),
            (SMOOTH_DIFF_NESTED_METHOD, smooth_bootstrap_diffs_nested),
        ]:
            if _method not in methods:
                continue
            _t = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*falling back to plain bootstrap.*",
                        category=UserWarning,
                    )
                    _boot = _fn(a, b, n_bootstrap, rng)
                ci_lo = float(np.percentile(_boot, 100 * alpha / 2))
                ci_hi = float(np.percentile(_boot, 100 * (1 - alpha / 2)))
            except Exception:
                ci_lo = ci_hi = obs_diff
            _el = time.perf_counter() - _t
            total_t[_method] += _el
            total_t_sq[_method] += _el * _el
            if ci_lo <= true_diff <= ci_hi:
                covered[_method] += 1
            total_w[_method] += ci_hi - ci_lo

        # ── bayes_indep_comp ─────────────────────────────────────────
        if BAYES_PAIR_INDEP_METHOD in methods:
            _t = time.perf_counter()
            try:
                ci_lo, ci_hi = _bayes_indep_comp_ci(a[:, 0], b[:, 0], alpha, n_bootstrap, rng)
            except Exception:
                ci_lo = ci_hi = obs_diff
            _el = time.perf_counter() - _t
            total_t[BAYES_PAIR_INDEP_METHOD] += _el
            total_t_sq[BAYES_PAIR_INDEP_METHOD] += _el * _el
            if ci_lo <= true_diff <= ci_hi:
                covered[BAYES_PAIR_INDEP_METHOD] += 1
            total_w[BAYES_PAIR_INDEP_METHOD] += ci_hi - ci_lo

        # ── bayes_paired_comp ────────────────────────────────────────
        if BAYES_PAIR_PAIRED_METHOD in methods:
            _t = time.perf_counter()
            try:
                ci_lo, ci_hi = _bayes_paired_comp_ci(a[:, 0], b[:, 0], alpha, n_bootstrap, rng)
            except Exception:
                ci_lo = ci_hi = obs_diff
            _el = time.perf_counter() - _t
            total_t[BAYES_PAIR_PAIRED_METHOD] += _el
            total_t_sq[BAYES_PAIR_PAIRED_METHOD] += _el * _el
            if ci_lo <= true_diff <= ci_hi:
                covered[BAYES_PAIR_PAIRED_METHOD] += 1
            total_w[BAYES_PAIR_PAIRED_METHOD] += ci_hi - ci_lo

    return [
        PairSimResult(
            model_a=cp.model_a,
            model_b=cp.model_b,
            benchmark_id=cp.benchmark_id,
            source=cp.source,
            corpus_size=N,
            true_diff=true_diff,
            n=n,
            method=method,
            n_reps=n_reps,
            covered=covered[method],
            total_width=total_w[method],
            total_time=total_t[method],
            total_time_sq=total_t_sq[method],
        )
        for method in methods
    ]


def run_pairwise_simulation(
    corpus_pairs: list[CorpusPair],
    methods: list[str],
    sample_sizes: list[int],
    n_reps: int,
    n_bootstrap: int,
    alpha: float,
    progress_mode: str = "bar",
    seed: int = 42,
    n_workers: int = 1,
) -> list[PairSimResult]:
    """For each CorpusPair × valid sample size, draw n paired items WOR n_reps times.

    Computes all tango and baseline CI methods on the sampled paired binary
    data and checks coverage against the corpus-level true_diff.
    """
    global _WORKER_CORPUS_PAIRS
    global _WORKER_METHODS
    _WORKER_CORPUS_PAIRS = corpus_pairs
    _WORKER_METHODS = list(methods)

    idx_size_pairs: list[tuple[int, int]] = []
    for cp_idx, cp in enumerate(corpus_pairs):
        valid_sizes = [n for n in sample_sizes if n < cp.corpus_size]
        if not valid_sizes:
            print(
                f"  Warning: all sample sizes >= N={cp.corpus_size} for "
                f"({cp.model_a} vs {cp.model_b}) on {cp.benchmark_id}. Skipping."
            )
            continue
        idx_size_pairs.extend(itertools.product([cp_idx], valid_sizes))

    if not idx_size_pairs:
        return []

    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(idx_size_pairs))]
    args_list = [
        (cp_idx, n, n_reps, n_bootstrap, alpha, child_seeds[i])
        for i, (cp_idx, n) in enumerate(idx_size_pairs)
    ]

    reporter = _ProgressReporter(len(args_list), mode=progress_mode, label="sim")
    results: list[PairSimResult] = []

    if n_workers == 1:
        for i, args in enumerate(args_list):
            results.extend(_run_pairwise_real_cell(args))
            cp = corpus_pairs[args[0]]
            reporter.update(i + 1, detail=f"{cp.benchmark_id}/{cp.model_a[:16]} n={args[1]}")
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_pairwise_real_cell, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1, detail=f"cells done: {i + 1}/{len(args_list)}")

    reporter.update(len(args_list), detail="done")
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


def _mc_proportion_stats(
    successes: int, total: int, z: float = 1.96
) -> tuple[float, float, float, float]:
    if total <= 0:
        return (float("nan"),) * 4
    p_hat = successes / total
    mcse  = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    lo, hi = max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)
    return float(p_hat), mcse, float(lo), float(hi)


def _rule(width: int, char: str = "─") -> str:
    return char * width


def _time_stats(subset: list[PairSimResult]) -> tuple[float, float]:
    total_reps = sum(r.n_reps for r in subset)
    if total_reps <= 0:
        return float("nan"), float("nan")
    sum_t = sum(r.total_time for r in subset)
    sum_t2 = sum(r.total_time_sq for r in subset)
    avg = sum_t / total_reps
    var = max(0.0, sum_t2 / total_reps - avg * avg)
    se = float(np.sqrt(var / total_reps))
    return avg * 1000.0, se * 1000.0


def print_report(
    results: list[PairSimResult],
    corpus_pairs: list[CorpusPair],
    methods: list[str],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
) -> None:
    """Print coverage and CI-width tables aggregated across all pairs."""
    if not results:
        print("No results to report.")
        return

    target = 1.0 - alpha
    used_ns = sorted({r.n for r in results})

    print(f"\n{'═'*80}")
    print(f"  Tango Real-Data Simulation  (target coverage {target:.0%}, α={alpha})")
    print(f"  Pairs: {len(corpus_pairs)}  |  Reps/cell: {n_reps}  |  N tested: {used_ns}")
    print(f"{'═'*80}")

    # ── Per-benchmark, per-pair coverage table ────────────────────────────────
    benchmarks = sorted({r.benchmark_id for r in results})
    for bench in benchmarks:
        bench_results = [r for r in results if r.benchmark_id == bench]
        pairs_in_bench = sorted(
            {(r.model_a, r.model_b) for r in bench_results},
            key=lambda x: x[0],
        )
        print(f"\n  Benchmark: {bench}")
        for ma, mb in pairs_in_bench:
            pair_results = [r for r in bench_results if r.model_a == ma and r.model_b == mb]
            true_diff = pair_results[0].true_diff if pair_results else float("nan")
            N = pair_results[0].corpus_size if pair_results else 0
            print(
                f"\n  ({ma}  vs  {mb})\n"
                f"  N={N}, true_diff={true_diff:+.4f}"
            )
            col_w = 9
            ns_here = sorted({r.n for r in pair_results})
            header_cols = "  ".join(f"n={n:>3}" for n in ns_here)
            print(f"  {'Method':<25}  {header_cols}")
            print(f"  {_rule(25 + 2 + 8 * len(ns_here))}")
            for method in methods:
                m_results = [r for r in pair_results if r.method == method]
                if not m_results:
                    continue
                row = f"  {method:<25}"
                for n in ns_here:
                    subset = [r for r in m_results if r.n == n]
                    if not subset:
                        row += f"  {'─':>{col_w}}"
                        continue
                    cov = float(np.mean([r.covered / r.n_reps for r in subset]))
                    row += f"  {cov:.3f}{_cov_marker(cov, target)}"
                print(row)

    # ── Aggregated coverage table ─────────────────────────────────────────────
    print(f"\n  Aggregated coverage across all {len(corpus_pairs)} pairs")
    print(f"  {'Method':<25}  " + "  ".join(f"n={n:>3}" for n in used_ns))
    print(f"  {_rule(25 + 2 + 8 * len(used_ns))}")
    for method in methods:
        m_results = [r for r in results if r.method == method]
        if not m_results:
            continue
        row = f"  {method:<25}"
        for n in used_ns:
            subset = [r for r in m_results if r.n == n]
            if not subset:
                row += f"  {'─':>7}"
                continue
            cov = float(np.mean([r.covered / r.n_reps for r in subset]))
            row += f"  {cov:.3f}{_cov_marker(cov, target)}"
        print(row)

    # ── Aggregated mean CI width table ────────────────────────────────────────
    print(f"\n  Mean CI width (aggregated across all pairs)")
    print(f"  {'Method':<25}  " + "  ".join(f"n={n:>3}" for n in used_ns))
    print(f"  {_rule(25 + 2 + 8 * len(used_ns))}")
    for method in methods:
        m_results = [r for r in results if r.method == method]
        if not m_results:
            continue
        row = f"  {method:<25}"
        for n in used_ns:
            subset = [r for r in m_results if r.n == n]
            if not subset:
                row += f"  {'─':>7}"
                continue
            avg_w = float(np.mean([r.total_width / r.n_reps for r in subset]))
            row += f"  {avg_w:.4f}"
        print(row)

    # -- Overall summary ----------------------------------------------------
    print(f"\n{'─'*72}")
    print("  OVERALL SUMMARY  (averaged across all benchmarks, pairs, and n)")
    print(f"{'─'*72}")

    all_cov: dict[str, list[float]] = defaultdict(list)
    all_wid: dict[str, list[float]] = defaultdict(list)
    all_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in results:
        all_cov[r.method].append(r.covered / r.n_reps)
        all_wid[r.method].append(r.total_width / r.n_reps)
        c_prev, t_prev = all_counts[r.method]
        all_counts[r.method] = (c_prev + r.covered, t_prev + r.n_reps)

    print(f"\n  {'Method':<25}  {'Cov':>6}  {'MCSE':>7}  {'Band95':>13}  {'Width':>8}  {'Dev':>8}  {'Time(ms)':>14}")
    print(f"  {'─'*25}  {'─'*6}  {'─'*7}  {'─'*13}  {'─'*8}  {'─'*8}  {'─'*14}")
    for method in methods:
        if not all_cov[method]:
            continue
        mc = float(np.mean(all_cov[method]))
        mw = float(np.mean(all_wid[method]))
        dev = mc - target
        mark = _cov_marker(mc, target)
        c_tot, t_tot = all_counts[method]
        _, mcse, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        avg_ms, se_ms = _time_stats([r for r in results if r.method == method])
        time_str = f"{avg_ms:.3f}±{se_ms:.3f}" if np.isfinite(avg_ms) else "─"
        print(
            f"  {method:<25}  {mc:>5.3f}{mark}  {mcse:>7.4f}  {f'{lo:.3f}-{hi:.3f}':>13}  {mw:>8.4f}  {dev:>+8.3f}  {time_str:>14}"
        )
    print()


# ---------------------------------------------------------------------------
# CSV save
# ---------------------------------------------------------------------------


def save_results_csv(results: list[PairSimResult], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model_a", "model_b", "benchmark_id", "source",
            "corpus_size", "true_diff", "n", "method",
            "n_reps", "covered", "coverage",
            "mean_width", "total_time", "total_time_sq",
        ])
        for r in results:
            w.writerow([
                r.model_a, r.model_b, r.benchmark_id, r.source,
                r.corpus_size, f"{r.true_diff:.6f}", r.n, r.method,
                r.n_reps, r.covered,
                f"{r.covered / max(r.n_reps, 1):.6f}",
                f"{r.total_width / max(r.n_reps, 1):.6f}",
                f"{r.total_time:.6f}", f"{r.total_time_sq:.6f}",
            ])
    print(f"  Results saved to: {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _set_sparse_xticks(ax: Any, ns: list[int], *, max_labels: int = 8) -> None:
    if not ns:
        return
    ns_sorted = sorted(ns)
    ax.set_xticks(ns_sorted)
    if len(ns_sorted) <= max_labels:
        ax.set_xticklabels([str(n) for n in ns_sorted])
        return
    keep = set(np.linspace(0, len(ns_sorted) - 1, max_labels, dtype=int).tolist())
    keep.update({0, len(ns_sorted) - 1})
    ax.set_xticklabels([str(n) if i in keep else "" for i, n in enumerate(ns_sorted)])


def save_plots(
    results: list[PairSimResult],
    methods: list[str],
    alpha: float,
    out_dir: str,
    prefix: str = "tango_real",
) -> None:
    """Save coverage, width, and coverage×cost plots for all and per benchmark."""
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        print("No results — skipping plots.")
        return

    target = 1.0 - alpha
    used_ns = sorted({r.n for r in results})
    benchmarks = sorted({r.benchmark_id for r in results})
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sns.set_style("whitegrid")

    def _plot_coverage_and_width(
        res: list[PairSimResult],
        title: str,
        path_stem: str,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_cov, ax_wid = axes

        for method in methods:
            m_res = [r for r in res if r.method == method]
            if not m_res:
                continue
            cov_by_n  = []
            wid_by_n  = []
            for n in used_ns:
                subset = [r for r in m_res if r.n == n]
                if not subset:
                    cov_by_n.append(float("nan"))
                    wid_by_n.append(float("nan"))
                else:
                    cov_by_n.append(float(np.mean([r.covered / r.n_reps for r in subset])))
                    wid_by_n.append(float(np.mean([r.total_width / r.n_reps for r in subset])))

            color = _METHOD_COLORS.get(method, "#333333")
            label = _METHOD_LABELS.get(method, method)
            ax_cov.plot(used_ns, cov_by_n, marker="o", label=label, color=color)
            ax_wid.plot(used_ns, wid_by_n, marker="o", label=label, color=color)

        # Coverage plot
        ax_cov.axhline(target, color="black", linestyle="--", linewidth=1.0, label=f"nominal {target:.0%}")
        ax_cov.axhspan(target - 0.04, target + 0.04, color="black", alpha=0.07)
        ax_cov.set_xlabel("Sample size n")
        ax_cov.set_ylabel("Coverage")
        ax_cov.set_title(f"{title}\nCoverage (target {target:.0%})")
        ax_cov.set_ylim(max(0.0, target - 0.15), 1.02)
        _set_sparse_xticks(ax_cov, used_ns)
        ax_cov.legend(fontsize=8, ncol=1, loc="lower right")

        # Width plot
        ax_wid.set_xlabel("Sample size n")
        ax_wid.set_ylabel("Mean CI width")
        ax_wid.set_title(f"{title}\nMean CI Width")
        _set_sparse_xticks(ax_wid, used_ns)
        ax_wid.legend(fontsize=8, ncol=1)

        fig.tight_layout()
        fpath = os.path.join(out_dir, f"{path_stem}_{run_ts}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {fpath}")

    def _plot_width(
        res: list[PairSimResult],
        title: str,
        path_stem: str,
    ) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        for method in methods:
            m_res = [r for r in res if r.method == method]
            if not m_res:
                continue
            wid_by_n = []
            for n in used_ns:
                subset = [r for r in m_res if r.n == n]
                if not subset:
                    wid_by_n.append(float("nan"))
                else:
                    wid_by_n.append(float(np.mean([r.total_width / r.n_reps for r in subset])))

            color = _METHOD_COLORS.get(method, "#333333")
            label = _METHOD_LABELS.get(method, method)
            ax.plot(used_ns, wid_by_n, marker="o", label=label, color=color)

        ax.set_xlabel("Sample size n")
        ax.set_ylabel("Mean CI width")
        ax.set_title(f"{title}\nMean CI Width")
        _set_sparse_xticks(ax, used_ns)
        ax.legend(fontsize=8, ncol=1)

        fig.tight_layout()
        fpath = os.path.join(out_dir, f"{path_stem}_{run_ts}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {fpath}")

    def _plot_coverage_vs_cost(
        res: list[PairSimResult],
        title: str,
        path_stem: str,
    ) -> None:
        if not res:
            return

        fig, ax = plt.subplots(1, 1, figsize=(9, 5.2))
        ax.axhspan(max(0.0, target - 0.04), min(1.0, target + 0.04),
                   color="#DDDDDD", alpha=0.40, zorder=0)
        ax.axhline(target, color="black", linewidth=1.1, linestyle="--", zorder=1)

        for method in methods:
            m_res = [r for r in res if r.method == method]
            if not m_res:
                continue

            points: list[tuple[int, float, float, float]] = []
            for n in used_ns:
                subset = [r for r in m_res if r.n == n]
                if not subset:
                    continue
                avg_ms, se_ms = _time_stats(subset)
                if not np.isfinite(avg_ms) or avg_ms < 0:
                    continue
                avg_ms = max(avg_ms, 1e-4)
                cov = float(np.mean([r.covered / r.n_reps for r in subset]))
                points.append((n, avg_ms, cov, 1.96 * se_ms))

            if not points:
                continue

            color = _METHOD_COLORS.get(method, "#333333")
            label = _METHOD_LABELS.get(method, method)

            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            ax.plot(xs, ys, color=color, linewidth=1.1, alpha=0.55, zorder=2)
            ax.errorbar(
                xs,
                ys,
                xerr=[p[3] for p in points],
                fmt="o",
                color=color,
                label=label,
                markersize=6,
                markeredgewidth=0.7,
                markeredgecolor="white",
                elinewidth=0.9,
                capsize=2.5,
                capthick=0.9,
                alpha=0.90,
                zorder=3,
            )

            label_idxs = {0, len(points) // 2, len(points) - 1} if len(points) > 2 else set(range(len(points)))
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

        ax.set_xscale("log")
        ax.set_xlabel("Mean CI time (ms) — log scale  [error bars: ±1.96 SE]")
        ax.set_ylabel("Coverage rate")
        ax.set_title(f"{title}\nCoverage × Cost (target {target:.0%})")
        ax.set_ylim(max(0.0, target - 0.20), min(1.01, target + 0.12))
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
        ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.35)
        ax.tick_params(labelsize=8.5)
        ax.legend(fontsize=8, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.85)

        fig.tight_layout()
        fpath = os.path.join(out_dir, f"{path_stem}_{run_ts}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {fpath}")

    # Aggregated plot
    _plot_coverage_and_width(results, "All benchmarks (aggregated)", f"{prefix}_all")
    _plot_width(results, "All benchmarks (aggregated)", f"{prefix}_all_width")
    _plot_coverage_vs_cost(results, "All benchmarks (aggregated)", f"{prefix}_all_cov_x_cost")

    # Per-benchmark plots
    for bench in benchmarks:
        bench_res = [r for r in results if r.benchmark_id == bench]
        if bench_res:
            _plot_coverage_and_width(bench_res, f"Benchmark: {bench}", f"{prefix}_{bench}")
            _plot_width(bench_res, f"Benchmark: {bench}", f"{prefix}_{bench}_width")
            _plot_coverage_vs_cost(bench_res, f"Benchmark: {bench}", f"{prefix}_{bench}_cov_x_cost")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare tango CI methods on real binary benchmark data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=SOURCES, default="dove",
                        help="Data source")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Benchmark IDs (default: all defaults for source)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names (default: all defaults for source)")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="Simulation repetitions per cell")
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N,
                        help="Bootstrap resamples for the 'bootstrap' baseline method")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                        help="Sample sizes to sweep")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Significance level (CI level = 1 - alpha)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Parallel worker processes")
    parser.add_argument("--min-pair-size", type=int, default=50,
                        help="Minimum shared items required to form a pair")
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar",
                        help="Progress display mode")
    parser.add_argument(
        "--multi-run-methods",
        action="store_true",
        help="Enable multirun and *_nested methods (default: single-run methods only)",
    )
    parser.add_argument("--plots", choices=PLOT_MODES, default="save",
                        help="Post-run plot mode")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save",
                        help="Write CSV results")
    parser.add_argument("--out-dir", default="simulations/out",
                        help="Base output directory")
    parser.add_argument("--plots-dir", default=None,
                        help="Directory for plots (default: <out-dir>/plots)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token for private/gated datasets")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace cache directory")

    args = parser.parse_args()

    hf_token  = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir   = args.out_dir
    plots_dir = args.plots_dir or str(Path(out_dir) / "plots")

    # Resolve defaults
    if args.source == "dove":
        if args.models is None and args.benchmarks is None:
            model_bench = DOVE_DEFAULT_MODEL_BENCH
        else:
            bms = args.benchmarks or list(dict.fromkeys(b for _, b in DOVE_DEFAULT_MODEL_BENCH))
            mds = args.models     or list(dict.fromkeys(m for m, _ in DOVE_DEFAULT_MODEL_BENCH))
            model_bench = [(m, b) for m in mds for b in bms]
    else:
        # OpenEval: filter the confirmed default pairs by requested benchmarks/models.
        # Never cross-product all models × all benchmarks — coverage is sparse.
        if args.models is None and args.benchmarks is None:
            model_bench = OPENEVAL_DEFAULT_MODEL_BENCH
        else:
            bms = set(args.benchmarks) if args.benchmarks else None
            mds = set(args.models)     if args.models     else None
            model_bench = [
                (m, b) for m, b in OPENEVAL_DEFAULT_MODEL_BENCH
                if (bms is None or b in bms) and (mds is None or m in mds)
            ]
            if not model_bench:
                print(
                    "No matching (model, benchmark) pairs found in the OpenEval defaults.\n"
                    "  Available benchmarks: "
                    + str(sorted({b for _, b in OPENEVAL_DEFAULT_MODEL_BENCH}))
                    + "\n  Available models: "
                    + str(sorted({m for m, _ in OPENEVAL_DEFAULT_MODEL_BENCH}))
                )
                sys.exit(1)

    methods = list(SINGLE_RUN_METHODS)
    if args.multi_run_methods:
        methods.extend(MULTI_RUN_ONLY_METHODS)

    print(f"\nTango Real-Data Simulation")
    print(f"  Source        : {args.source}")
    print(f"  Reps/cell     : {args.reps}")
    print(f"  Bootstrap n   : {args.bootstrap_n}")
    print(f"  Alpha / level : {args.alpha} / {1 - args.alpha:.0%}")
    print(f"  Sample sizes  : {args.sizes}")
    print(f"  Seed          : {args.seed}")
    print(f"  Workers       : {max(1, args.workers)}")
    print(f"  Min pair size : {args.min_pair_size}")
    print(f"  Multi-run meth: {'on' if args.multi_run_methods else 'off'}")
    print(f"  Methods       : {methods}")
    print(f"  HF token      : {'set' if hf_token else 'not set (cached login)'}")

    # ── Load corpus pairs ─────────────────────────────────────────────────────
    if args.source == "dove":
        corpus_pairs = build_dove_corpus_pairs(
            model_bench,
            dove_repo=DOVE_REPO, hf_token=hf_token, cache_dir=args.cache_dir,
            min_pair_size=args.min_pair_size,
        )
    else:
        corpus_pairs = build_openeval_corpus_pairs(
            model_bench,
            openeval_repo=OPENEVAL_REPO, hf_token=hf_token, cache_dir=args.cache_dir,
            min_pair_size=args.min_pair_size,
        )

    if not corpus_pairs:
        print("No corpus pairs loaded — exiting.")
        sys.exit(1)

    print(f"  {len(corpus_pairs)} corpus pairs ready for simulation.\n")
    for cp in corpus_pairs:
        print(
            f"  ({cp.model_a}  vs  {cp.model_b}) [{cp.benchmark_id}]  "
            f"N={cp.corpus_size}, true_diff={cp.true_diff:+.4f}"
        )
    print_corpus_pair_stats(corpus_pairs)

    # ── Run simulation ────────────────────────────────────────────────────────
    t_start = time.time()
    results = run_pairwise_simulation(
        corpus_pairs=corpus_pairs,
        methods=methods,
        sample_sizes=args.sizes,
        n_reps=args.reps,
        n_bootstrap=args.bootstrap_n,
        alpha=args.alpha,
        progress_mode=args.progress,
        seed=args.seed,
        n_workers=max(1, args.workers),
    )
    elapsed = time.time() - t_start
    print(f"\n  Simulation complete in {elapsed:.1f}s  ({len(results):,} result records)")

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(results, corpus_pairs, methods, args.sizes, args.alpha, args.reps)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.save_results == "save":
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"tango_real_{args.source}.csv")
        save_results_csv(results, csv_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plots == "save":
        os.makedirs(plots_dir, exist_ok=True)
        save_plots(
            results,
            methods=methods,
            alpha=args.alpha,
            out_dir=plots_dir,
            prefix=f"tango_real_{args.source}",
        )


if __name__ == "__main__":
    main()
