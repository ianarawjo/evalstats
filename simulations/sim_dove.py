#!/usr/bin/env python3
"""
sim_dove.py — Bootstrap CI comparison on real benchmark data (DOVE_Lite or OpenEval).

Unlike sim_compare_boot.py (which draws from synthetic parametric distributions),
this simulation uses real instance-level model outputs:

  1. Load each (model, benchmark) corpus from a real eval dataset.
  2. Treat the full corpus as the 'population'; corpus mean = ground truth.
  3. Repeatedly subsample WITHOUT replacement (n items per rep, n << N).
  4. Compute CIs for each method and check coverage against the corpus mean.

This validates CI method performance on real eval-like score distributions,
complementing the synthetic-data results from sim_compare_boot.py.

Finite-population note
  The estimand is explicitly the corpus mean, not an infinite-population
  parameter.  For the sample sizes tested (n << N), the FPC sqrt((N-n)/N)
  is negligible (>0.995 for n≤200, N≥6000); a table is printed at runtime.

──────────────────────────────────────────────────────────────────────────────
DOVE_Lite  (--source dove, default)
  https://huggingface.co/datasets/nlphuji/DOVE_Lite
  Requires HuggingFace approval.  Log in first: huggingface-cli login
  Benchmarks: hellaswag, arc_challenge, humaneval  (binary only currently;
    see DOVE_BENCHMARK_SPECS stubs for how to add continuous benchmarks)
  File layout: {model}/en/{shots}_shot/{benchmark}.parquet

OpenEval  (--source openeval)
  https://huggingface.co/datasets/human-centered-eval/OpenEval
  No approval required (CC-BY-NC-4.0).
  Benchmarks:
    binary     mmlu-pro, gpqa, boolq, imdb
    continuous cnndm, xsum  (ROUGE/BERTScore ∈ [0,1])
    (likert/grades: stubs — add when suitable benchmarks are identified)
  Three-table schema: bench | item (56K rows) | response (584K rows)
  Joining: response.response_id starts with the corresponding item.item_id.

──────────────────────────────────────────────────────────────────────────────
Usage
  # DOVE (default)
  python simulations/sim_dove.py
  python simulations/sim_dove.py --benchmarks hellaswag arc_challenge
  python simulations/sim_dove.py --models Llama-3.2-1B-Instruct Meta-Llama-3-8B-Instruct
  python simulations/sim_dove.py --hf-token hf_xxxx

  # OpenEval
  python simulations/sim_dove.py --source openeval
  python simulations/sim_dove.py --source openeval --benchmarks mmlu-pro cnndm
  python simulations/sim_dove.py --source openeval --list-models
  python simulations/sim_dove.py --source openeval --models gpt-4o gpt-4o-mini

  # Shared options
  python simulations/sim_dove.py --reps 500 --bootstrap-n 1000
  python simulations/sim_dove.py --sizes 10 20 50 100
  python simulations/sim_dove.py --out-dir simulations/out --save-results save
  python simulations/sim_dove.py --plots save --plots-dir simulations/out/plots
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.resampling import (
        bootstrap_ci_1d,
        wilson_ci,
        wald_ci,
        clopper_pearson_ci,
        t_interval_ci_1d,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

SOURCES = ["dove", "openeval", "all"]

METHODS = ["bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t"]
WILSON_METHOD = "wilson"
BAYES_SINGLE_METHOD = "bayes_indep"
WALD_METHOD = "wald"
CP_METHOD = "clopper_pearson"
T_INTERVAL_METHOD = "t_interval"
REPORT_METHODS = METHODS + [T_INTERVAL_METHOD, WILSON_METHOD, WALD_METHOD, CP_METHOD, BAYES_SINGLE_METHOD]

PROGRESS_MODES = ["bar", "cell", "off"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]


# ─────────────────────────────────────────────────────────────────────────────
# DOVE_Lite — constants and benchmark specs
# ─────────────────────────────────────────────────────────────────────────────

DOVE_REPO = "nlphuji/DOVE_Lite"

# Curated default (model, benchmark_id) pairs for DOVE_Lite.
# Each pair is confirmed to have data; the cross-product of all models × all
# benchmarks does NOT work because coverage varies by model family (e.g.
# GPT-4o-mini/Llama-3.3-70B have humaneval/cnn_dailymail/wmt14 but not
# hellaswag/arc_challenge; OLMoE lacks quality).
# Run explore_dove.py to discover more valid combinations.
DOVE_DEFAULT_PAIRS: list[tuple[str, str]] = [
    # hellaswag — commonsense NLI (0/1 correct)
    ("Llama-3.2-1B-Instruct",     "hellaswag"),   # small — lower accuracy tier
    ("OLMoE-1B-7B-0924-Instruct", "hellaswag"),   # MoE 1B active — lower-mid tier
    ("Mistral-7B-Instruct-v0.3",  "hellaswag"),   # mid tier
    ("Meta-Llama-3-8B-Instruct",  "hellaswag"),   # mid tier
    # arc_challenge — science QA (0/1 correct)
    ("Llama-3.2-1B-Instruct",     "arc_challenge"),
    ("OLMoE-1B-7B-0924-Instruct", "arc_challenge"),
    ("Mistral-7B-Instruct-v0.3",  "arc_challenge"),
    ("Meta-Llama-3-8B-Instruct",  "arc_challenge"),
    # gsm8k — grade-school math (0/1); humaneval excluded (DOVE uses fractional test-case scores)
    ("GPT-4o-mini",               "gsm8k"),       # upper-mid tier
    ("Llama-3.3-70B-Instruct",    "gsm8k"),       # high accuracy tier
    # squad — reading comprehension (0/1)
    ("GPT-4o-mini",               "squad"),
    ("Llama-3.3-70B-Instruct",    "squad"),
    # quality — long-context reading comprehension (0/1); OLMoE lacks this file
    ("Llama-3.2-1B-Instruct",     "quality"),
    ("Mistral-7B-Instruct-v0.3",  "quality"),
    ("Meta-Llama-3-8B-Instruct",  "quality"),
    # cnn_dailymail — summarization quality score in [0,1]
    ("GPT-4o-mini",               "cnn_dailymail"),
    ("Llama-3.3-70B-Instruct",    "cnn_dailymail"),
    # wmt14.cs-en — translation quality score in [0,1]
    ("GPT-4o-mini",               "wmt14.cs-en"),
    ("Llama-3.3-70B-Instruct",    "wmt14.cs-en"),
]


@dataclass
class DoveBenchmarkSpec:
    benchmark_id: str
    file_name: str
    eval_type: str       # "binary" | "continuous" | "likert" | "grades"
    description: str
    language: str = "en"
    shots: int = 0


# Benchmarks currently implemented. Stubs show where to add other eval types
# once a suitable DOVE file (with continuous/Likert/grade scores) is identified.
DOVE_BENCHMARK_SPECS: dict[str, DoveBenchmarkSpec] = {
    "hellaswag": DoveBenchmarkSpec(
        benchmark_id="hellaswag",
        file_name="hellaswag.parquet",
        eval_type="binary",
        description="HellaSwag commonsense NLI (0/1 correct)",
    ),
    "arc_challenge": DoveBenchmarkSpec(
        benchmark_id="arc_challenge",
        file_name="ai2_arc.arc_challenge.parquet",
        eval_type="binary",
        description="ARC Challenge science QA (0/1 correct)",
    ),
    "gsm8k": DoveBenchmarkSpec(
        benchmark_id="gsm8k",
        file_name="gsm8k.parquet",
        eval_type="binary",
        description="GSM8K grade-school math (0/1 correct)",
    ),
    "squad": DoveBenchmarkSpec(
        benchmark_id="squad",
        file_name="squad.parquet",
        eval_type="binary",
        description="SQuAD reading comprehension (0/1 correct)",
    ),
    "quality": DoveBenchmarkSpec(
        benchmark_id="quality",
        file_name="quality.parquet",
        eval_type="binary",
        description="QuALITY long-context reading comprehension (0/1 correct)",
    ),
    "cnn_dailymail": DoveBenchmarkSpec(
        benchmark_id="cnn_dailymail",
        file_name="cnn_dailymail.parquet",
        eval_type="continuous",
        description="CNN/DailyMail summarization quality score in [0,1]",
    ),
    "wmt14.cs-en": DoveBenchmarkSpec(
        benchmark_id="wmt14.cs-en",
        file_name="wmt14.cs-en.parquet",
        eval_type="continuous",
        description="WMT14 Czech-to-English translation quality score in [0,1]",
    ),
}

DOVE_DEFAULT_BENCHMARKS: list[str] = list(dict.fromkeys(b for _, b in DOVE_DEFAULT_PAIRS))


# ─────────────────────────────────────────────────────────────────────────────
# OpenEval — constants and benchmark specs
# ─────────────────────────────────────────────────────────────────────────────

OPENEVAL_REPO = "human-centered-eval/OpenEval"

# response_id format: {source}_{YYYYMMDDTHHMMSSZ}_{index}_{model_name}_{run}
# item_id  is just  : {source}_{YYYYMMDDTHHMMSSZ}_{index}
# Extracting source + item_id from response_id avoids loading the item table entirely.
_OE_RESP_ID_RE = re.compile(r"^(.+?)_(\d{8}T\d{6}Z)_(\d+)")

# Curated default (model, benchmark_id) pairs for OpenEval.
# Each pair is confirmed to have data in the dataset; the cross-product of all
# models × all benchmarks does NOT work because coverage is sparse.
# Run --list-models --benchmarks <id> to discover more valid combinations.
OPENEVAL_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("falcon-7b-instruct",       "mmlu-pro"),      # binary  — knowledge/reasoning (metric: chain_of_thought_correctness; low mean reflects CoT format mismatch, not knowledge)
    ("gpt-4o",                   "culturalbench"), # binary  — cultural knowledge
    ("o4-mini",                  "opentom"),       # binary  — theory-of-mind
    ("llama-65b",                "bbq"),           # binary  — social-bias QA
    ("vicuna-13b-v1.3",          "cnndm"),         # continuous — summarization ROUGE
    ("DeepSeek-V3-0324",         "do-not-answer"), # binary  — safety refusal
    ("DeepSeek-R1",              "emobench"),      # binary  — emotional intelligence
    ("qwen-3-80b-instruct",      "gpqa"),          # binary  — graduate science reasoning
    ("gpt-4.1-mini",             "hi-tom"),        # binary  — higher-order ToM (o1-mini excluded: quasi_prefix_exact_match scores all zeros, likely response-format mismatch)
    ("gemma-3-27b-it",           "ifeval"),        # binary  — instruction following
    ("falcon-40b-instruct",      "imdb"),          # binary  — sentiment classification
    ("qwen-2.5-72b-instruct",    "omni-math"),     # binary  — math reasoning
    ("kimi-k2",                  "salad-bench"),   # binary  — safety alignment
    ("llama-2-70b",              "xsum"),          # continuous — summarization ROUGE
    # ("grok-4",                   "hi-tom"),
    ("grok-4",                    "truthfulqa"),    # continuous — BLEU-max ÷ 100 → [0,1]   ALT: phi-4
]


@dataclass
class OpenEvalBenchmarkSpec:
    benchmark_id: str    # must match the source prefix in response_id (before the timestamp)
    eval_type: str       # "binary" | "continuous" | "likert" | "grades"
    description: str
    metric_name: str | None = None  # None = use first score entry for each response
    score_scale: float = 1.0        # multiply extracted score by this factor (e.g. 0.01 to normalise 0–100 → 0–1)


# Benchmark IDs must match the source component in response_id
# (i.e. the part before _{YYYYMMDDTHHMMSSZ}_ in each response_id).
# Run --list-benchmarks to see the exact IDs present in the dataset.
OPENEVAL_BENCHMARK_SPECS: dict[str, OpenEvalBenchmarkSpec] = {
    # ── Binary (0/1 correct) ───────────────────────────────────────────────
    "mmlu-pro": OpenEvalBenchmarkSpec(
        benchmark_id="mmlu-pro",
        eval_type="binary",
        description="MMLU-Pro knowledge/reasoning (0/1 correct)",
    ),
    "gpqa": OpenEvalBenchmarkSpec(
        benchmark_id="gpqa",
        eval_type="binary",
        description="GPQA graduate-level science reasoning (0/1 correct)",
    ),
    "boolq": OpenEvalBenchmarkSpec(
        benchmark_id="boolq",
        eval_type="binary",
        description="BoolQ yes/no reading comprehension (0/1 correct)",
    ),
    "imdb": OpenEvalBenchmarkSpec(
        benchmark_id="imdb",
        eval_type="binary",
        description="IMDB sentiment classification (0/1 correct)",
    ),
    "truthfulqa": OpenEvalBenchmarkSpec(
        benchmark_id="truthfulqa",
        eval_type="continuous",
        description="TruthfulQA BLEU-max fluency/truthfulness score ∈ [0,1]",
        metric_name="bleu_max",
        score_scale=0.01,  # OpenEval stores bleu_max as 0–100; normalise to [0,1]
    ),
    "culturalbench": OpenEvalBenchmarkSpec(
        benchmark_id="culturalbench",
        eval_type="binary",
        description="CulturalBench cultural knowledge QA (0/1 correct)",
    ),
    "opentom": OpenEvalBenchmarkSpec(
        benchmark_id="opentom",
        eval_type="binary",
        description="OpenToM Theory-of-Mind reasoning QA (0/1 correct)",
    ),
    "bbq": OpenEvalBenchmarkSpec(
        benchmark_id="bbq",
        eval_type="binary",
        description="BBQ social-bias benchmark for QA (0/1 correct)",
    ),
    "bold": OpenEvalBenchmarkSpec(
        benchmark_id="bold",
        eval_type="binary",
        description="BOLD bias-sensitive generation benchmark",
    ),
    "do-not-answer": OpenEvalBenchmarkSpec(
        benchmark_id="do-not-answer",
        eval_type="continuous", # min 0 max 6, clustered at 0s 
        description="Do-Not-Answer safety refusal benchmark",
    ),
    "hi-tom": OpenEvalBenchmarkSpec(
        benchmark_id="hi-tom",
        eval_type="binary",
        description="Hi-ToM higher-order theory-of-mind reasoning benchmark",
    ),
    "ifeval": OpenEvalBenchmarkSpec(
        benchmark_id="ifeval",
        eval_type="continuous", # 0/1 correct but many near 0.5s → treat as continuous for CI purposes
        description="Instruction-following evaluation benchmark",
    ),
    "omni-math": OpenEvalBenchmarkSpec(
        benchmark_id="omni-math",
        eval_type="binary",
        description="Omni-Math mathematical reasoning benchmark",
    ),
    "salad-bench": OpenEvalBenchmarkSpec(
        benchmark_id="salad-bench",
        eval_type="binary",
        description="SALAD-Bench safety/alignment benchmark",
    ),
    # ── Continuous ∈ [0, 1] — generation quality metrics ──────────────────
    "cnndm": OpenEvalBenchmarkSpec(
        benchmark_id="cnndm",
        eval_type="continuous",
        description="CNN/DailyMail summarization ROUGE-L ∈ [0,1]",
        metric_name="rouge_l",   # parallel-list layout: rouge_1/rouge_2/rouge_l/summac
    ),
    "xsum": OpenEvalBenchmarkSpec(
        benchmark_id="xsum",
        eval_type="continuous",
        description="XSUM abstractive summarization ROUGE-L ∈ [0,1]",
        metric_name="rouge_l",   # same parallel-list layout as cnndm
    ),
    "emobench": OpenEvalBenchmarkSpec(
        benchmark_id="emobench",
        eval_type="binary",
        description="EmoBench emotional intelligence benchmark",
    ),
    # ── Stubs for Likert / grades ──────────────────────────────────────────
    # Uncomment once a suitable OpenEval benchmark with ordinal scores is confirmed:
    #
    # "emobench": OpenEvalBenchmarkSpec(
    #     benchmark_id="emobench",
    #     eval_type="likert",
    #     description="PLACEHOLDER: EmoBench emotional intelligence (check score range)",
    # ),
}

OPENEVAL_DEFAULT_BENCHMARKS: list[str] = list(dict.fromkeys(b for _, b in OPENEVAL_DEFAULT_PAIRS))


# ─────────────────────────────────────────────────────────────────────────────
# Shared data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Corpus:
    """Full benchmark corpus for one (model, benchmark) pair — the 'population'."""
    model: str
    benchmark_id: str
    eval_type: str
    source: str          # "dove" | "openeval"
    scores: np.ndarray   # shape (N,); all deduplicated, finite instance scores
    corpus_mean: float   # ground truth estimand = mean of all scores
    corpus_size: int     # N


@dataclass
class SimResult:
    model: str
    benchmark_id: str
    eval_type: str
    source: str
    corpus_size: int
    corpus_mean: float
    n: int
    method: str
    n_reps: int
    covered: int
    total_width: float
    total_time: float = 0.0     # sum of per-rep CI wall-clock times (seconds)
    total_time_sq: float = 0.0  # sum of squared per-rep times (for SE via Var=E[x²]−E[x]²)


# ─────────────────────────────────────────────────────────────────────────────
# DOVE_Lite — loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def _dove_extract_scores(dataset: Any) -> np.ndarray:
    """Extract per-row scores from a loaded DOVE HF Dataset.

    DOVE stores scores in a nested 'evaluation' struct with a 'score' field.
    Falls back to a flat 'score' column if the struct is absent.
    """
    col_names = dataset.column_names

    if "evaluation" in col_names:
        eval_col = dataset["evaluation"]
        # List-of-dicts (most HF/Arrow backends)
        if isinstance(eval_col, list) and len(eval_col) > 0:
            sample = eval_col[0]
            if isinstance(sample, dict) and "score" in sample:
                return np.array([e["score"] for e in eval_col], dtype=float)
        # PyArrow struct array exposed as a mapping
        try:
            return np.array(dataset["evaluation"]["score"], dtype=float)
        except (TypeError, KeyError):
            pass

    if "score" in col_names:
        return np.array(dataset["score"], dtype=float)

    raise ValueError(
        f"Cannot find score column. Dataset columns: {col_names}. "
        "Expected 'evaluation' struct with 'score' field, or flat 'score' column."
    )


def _dove_extract_instance_indices(dataset: Any) -> np.ndarray | None:
    """Return per-row item indices for deduplication, or None if unavailable.

    Tries two schemas in order:
    1. Flat schema (most DOVE_Lite files): top-level 'sample_index' int column.
    2. Nested schema: 'instance.sample_identifier.hf_index'.
    """
    col_names = dataset.column_names

    # Flat schema — present in all inspected DOVE_Lite parquet files
    if "sample_index" in col_names:
        try:
            return np.array(dataset["sample_index"], dtype=int)
        except Exception:
            pass

    # Nested schema fallback
    if "instance" not in col_names:
        return None
    instance_col = dataset["instance"]
    if not (isinstance(instance_col, list) and len(instance_col) > 0):
        return None
    try:
        sample = instance_col[0]
        if not isinstance(sample, dict):
            return None
        sample_id = sample.get("sample_identifier")
        if not isinstance(sample_id, dict) or "hf_index" not in sample_id:
            return None
        return np.array(
            [inst["sample_identifier"]["hf_index"] for inst in instance_col],
            dtype=int,
        )
    except Exception:
        return None


def _dove_deduplicate(
    scores: np.ndarray,
    indices: np.ndarray | None,
) -> np.ndarray:
    """Keep one score per unique item index (first occurrence).

    DOVE contains multiple prompt perturbations per item. Keeping the first
    occurrence per unique sample_index (or hf_index) ensures each element of
    the returned array corresponds to an independent benchmark item, satisfying
    the IID assumption required by the bootstrap CI methods.
    """
    if indices is None:
        return scores
    seen: dict[int, float] = {}
    for idx, score in zip(indices.tolist(), scores.tolist()):
        if idx not in seen:
            seen[idx] = float(score)
    return np.array(list(seen.values()), dtype=float)


def load_dove_corpus(
    model: str,
    spec: DoveBenchmarkSpec,
    *,
    dove_repo: str = DOVE_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_size: int = 50,
    verbose: bool = True,
) -> Corpus | None:
    """Load and deduplicate one (model, benchmark) corpus from DOVE_Lite."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the HuggingFace datasets library: pip install datasets"
        )

    shots_str = f"{spec.shots}_shot"
    file_path = f"{model}/{spec.language}/{shots_str}/{spec.file_name}"
    dataset = None
    tried = [file_path]

    try:
        dataset = load_dataset(
            dove_repo, data_files=file_path, split="train",
            token=hf_token, cache_dir=cache_dir,
        )
    except Exception:
        # Try alternate shot counts in priority order.
        # Some models only have 5_shot data (e.g. GPT-4o-mini, Llama-3.3-70B).
        fallbacks = []
        for n in (5, 2, 3):
            if n != spec.shots:
                fallbacks.append(f"{model}/{spec.language}/{n}_shot/{spec.file_name}")
        if spec.shots > 0:
            fallbacks.append(f"{model}/{spec.language}/{spec.shots}_shots/{spec.file_name}")
        for alt in fallbacks:
            tried.append(alt)
            try:
                dataset = load_dataset(
                    dove_repo, data_files=alt, split="train",
                    token=hf_token, cache_dir=cache_dir,
                )
                break
            except Exception:
                pass

    if dataset is None:
        if verbose:
            print(f"  Skip  {model}/{spec.benchmark_id}: file not found {tried}")
        return None

    try:
        raw_scores = _dove_extract_scores(dataset)
        indices = _dove_extract_instance_indices(dataset)
        scores = _dove_deduplicate(raw_scores, indices)
    except Exception as exc:
        if verbose:
            print(f"  Skip  {model}/{spec.benchmark_id}: score extraction failed — {exc}")
        return None

    scores = scores[np.isfinite(scores)]
    if len(scores) < min_size:
        if verbose:
            print(f"  Skip  {model}/{spec.benchmark_id}: corpus too small (N={len(scores)} < {min_size})")
        return None

    if verbose:
        n_removed = len(raw_scores) - len(scores) if indices is not None else 0
        dup_note = f", {n_removed} prompt-pert. rows removed" if n_removed > 0 else ""
        print(
            f"  OK    {model}/{spec.benchmark_id}: "
            f"N={len(scores)}, mean={np.mean(scores):.4f}{dup_note}\n"
            f"        {_score_dist_summary(scores, spec.eval_type)}"
        )

    return Corpus(
        model=model, benchmark_id=spec.benchmark_id, eval_type=spec.eval_type,
        source="dove", scores=scores, corpus_mean=float(np.mean(scores)),
        corpus_size=len(scores),
    )


def build_dove_corpora(
    pairs: list[tuple[str, str]],
    *,
    dove_repo: str = DOVE_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_corpus_size: int = 50,
) -> list[Corpus]:
    """Load (model, benchmark) corpora from DOVE_Lite for the given pairs."""
    unknown = [(m, b) for m, b in pairs if b not in DOVE_BENCHMARK_SPECS]
    if unknown:
        print(f"Warning: unknown DOVE benchmark IDs {[b for _, b in unknown]}. Available: {list(DOVE_BENCHMARK_SPECS)}")
        pairs = [(m, b) for m, b in pairs if b in DOVE_BENCHMARK_SPECS]

    print(f"\nLoading DOVE_Lite corpora ({len(pairs)} model × benchmark pairs) …")

    corpora: list[Corpus] = []
    for model, bid in pairs:
        c = load_dove_corpus(
            model, DOVE_BENCHMARK_SPECS[bid],
            dove_repo=dove_repo, hf_token=hf_token,
            cache_dir=cache_dir, min_size=min_corpus_size,
        )
        if c is not None:
            corpora.append(c)

    print(f"  {len(corpora)}/{len(pairs)} corpora loaded successfully.\n")
    return corpora


# ─────────────────────────────────────────────────────────────────────────────
# OpenEval — loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def _oe_parse(val: Any) -> Any:
    """Coerce a value that may be a JSON string or an already-parsed object."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return val
    return val


def _oe_get_model_name(model_val: Any) -> str | None:
    """Extract model name from the 'model' field (dict or JSON string)."""
    obj = _oe_parse(model_val)
    if isinstance(obj, dict):
        return obj.get("name")
    return None


def _oe_parse_response_id(response_id: str) -> tuple[str, str] | tuple[None, None]:
    """
    Extract (source_benchmark, item_id) from an OpenEval response_id.

    response_id format: {source}_{YYYYMMDDTHHMMSSZ}_{index}_{model}_{run}
    item_id      =    : {source}_{YYYYMMDDTHHMMSSZ}_{index}

    Returns (None, None) if the format does not match.
    This avoids needing the item table entirely.
    """
    m = _OE_RESP_ID_RE.match(response_id)
    if m is None:
        return None, None
    source = m.group(1)
    item_id = f"{source}_{m.group(2)}_{m.group(3)}"
    return source, item_id


def _oe_extract_score(scores_val: Any, metric_name: str | None) -> float | None:
    """
    Extract a numeric score from OpenEval's scores field.

    Three observed layouts — all handled:

    Layout A — list-of-dicts (schema-documented):
        [{"metric": {"name": "accuracy", ...}, "value": 1.0}, ...]

    Layout B — dict with scalar value:
        {"metric": {"name": "accuracy", ...}, "value": 1.0}

    Layout C — dict with parallel metric-list + value-list (observed in practice):
        {"metric": [{"name": "rouge_1", ...}, {"name": "rouge_l", ...}, ...],
         "value":  [0.42, 0.38, ...]}
        The i-th value corresponds to the i-th metric descriptor.

    If metric_name is None, returns the value from the first entry/metric.
    Returns None if no suitable entry or value is found.
    """
    data = _oe_parse(scores_val)

    # ── Layout A: list-of-dicts ──────────────────────────────────────────
    if isinstance(data, list):
        if not data:
            return None

        def _list_entry_metric_name(e: Any) -> str | None:
            if not isinstance(e, dict):
                return None
            metric = _oe_parse(e.get("metric"))
            if isinstance(metric, dict):
                return metric.get("name")
            return None

        if metric_name is None:
            entry = data[0]
        else:
            entry = next(
                (e for e in data if _list_entry_metric_name(e) == metric_name), None
            )
            if entry is None:
                return None

        if not isinstance(entry, dict):
            return None
        val = entry.get("value")

    elif isinstance(data, dict):
        metrics_raw = _oe_parse(data.get("metric"))
        value_raw = data.get("value")

        if isinstance(metrics_raw, list) and isinstance(value_raw, list):
            # ── Layout C: parallel metric-list + value-list ──────────────
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
            # ── Layout B: dict with scalar value ─────────────────────────
            val = value_raw

    else:
        return None

    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _oe_first_metric_name(scores_val: Any) -> str | None:
    """Return the name of the first metric in the scores field (for logging)."""
    data = _oe_parse(scores_val)
    if isinstance(data, list) and data:
        e0 = data[0]
        if isinstance(e0, dict):
            metric = _oe_parse(e0.get("metric"))
            if isinstance(metric, dict):
                return metric.get("name")
    elif isinstance(data, dict):
        metric = _oe_parse(data.get("metric"))
        if isinstance(metric, dict):
            return metric.get("name")
        if isinstance(metric, list) and metric:
            # Layout C: parallel list — first element of metric list
            m0 = _oe_parse(metric[0])
            if isinstance(m0, dict):
                return m0.get("name")
    return None


def list_openeval_models(
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    benchmark_filter: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Return {model_name: {benchmark_source: response_count}}.

    If *benchmark_filter* is given, only count responses for those benchmarks
    (models with zero matching responses are omitted from the result).

    Scans the response table once via response_id regex — no item table needed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    filter_set = set(benchmark_filter) if benchmark_filter else None
    msg = (
        f"Scanning OpenEval responses for benchmarks {sorted(filter_set)} …"
        if filter_set else
        "Scanning OpenEval response table for model names and benchmark coverage …"
    )
    print(msg)
    ds = load_dataset(
        openeval_repo, "response", split="train",
        token=hf_token, cache_dir=cache_dir,
    )
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in ds:
        name = _oe_get_model_name(row.get("model"))
        source, _ = _oe_parse_response_id(row.get("response_id", ""))
        if name and source:
            if filter_set is None or source in filter_set:
                counts[name][source] += 1
    # Convert inner defaultdicts and drop models with no matching responses
    return {
        model: dict(bench_counts)
        for model, bench_counts in counts.items()
        if bench_counts
    }


def list_openeval_benchmarks(
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, int]:
    """
    Return {benchmark_source: response_count} for all benchmarks in OpenEval.

    Extracted from response_id via regex — no item table needed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print("Scanning OpenEval response table for benchmark IDs …")
    ds = load_dataset(
        openeval_repo, "response", split="train",
        token=hf_token, cache_dir=cache_dir,
    )
    counts: dict[str, int] = defaultdict(int)
    for row in ds:
        source, _ = _oe_parse_response_id(row.get("response_id", ""))
        if source:
            counts[source] += 1
    return dict(sorted(counts.items()))


def build_openeval_corpora(
    pairs: list[tuple[str, str]],
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_corpus_size: int = 50,
) -> list[Corpus]:
    """
    Load OpenEval corpora for the given (model, benchmark_id) pairs.

    Strategy (no item table needed)
    --------------------------------
    The response_id encodes both the benchmark source and item_id:
        response_id = "{source}_{YYYYMMDDTHHMMSSZ}_{index}_{model}_{run}"
        item_id     = "{source}_{YYYYMMDDTHHMMSSZ}_{index}"

    Pairs must be confirmed to have data; the full cross-product of all models
    × all benchmarks does NOT work — OpenEval coverage is sparse. Use
    --list-models --benchmarks <id> to discover valid combinations.

    If a benchmark ID is not in OPENEVAL_BENCHMARK_SPECS, run --list-benchmarks.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    # Validate benchmark IDs
    unknown = [b for _, b in pairs if b not in OPENEVAL_BENCHMARK_SPECS]
    if unknown:
        print(
            f"Warning: unknown OpenEval benchmark IDs {sorted(set(unknown))}.\n"
            f"  Run --list-benchmarks to see exact IDs present in the dataset.\n"
            f"  Known IDs: {list(OPENEVAL_BENCHMARK_SPECS)}"
        )
        pairs = [(m, b) for m, b in pairs if b in OPENEVAL_BENCHMARK_SPECS]
    if not pairs:
        return []

    pairs_set: set[tuple[str, str]] = set(pairs)      # (model, benchmark)
    benchmark_ids_set = {b for _, b in pairs}
    models_set        = {m for m, _ in pairs}

    # ── Load + filter response table ──────────────────────────────────────
    print("Loading OpenEval response table (~6.4 GB; cached after first download) …")
    response_ds = load_dataset(
        openeval_repo, "response", split="train",
        token=hf_token, cache_dir=cache_dir,
    )

    print(f"  Filtering to {len(pairs)} (model, benchmark) pairs …")

    def _keep_row(batch: dict) -> list[bool]:
        keep = []
        for rid, model_val in zip(batch["response_id"], batch["model"]):
            source, _ = _oe_parse_response_id(rid)
            if source not in benchmark_ids_set:
                keep.append(False)
                continue
            mname = _oe_get_model_name(model_val)
            keep.append((mname, source) in pairs_set)
        return keep

    response_ds = response_ds.filter(_keep_row, batched=True, batch_size=5_000)
    n_filtered = len(response_ds)
    print(f"  {n_filtered:,} responses match requested pairs.")

    if n_filtered == 0:
        print(
            "  No responses found. Tips:\n"
            "    • Run --list-benchmarks to verify benchmark source IDs.\n"
            "    • Run --list-models --benchmarks <ids> to verify model names."
        )
        return []

    # Per-pair breakdown — lets the user spot missing combos before the accumulation pass.
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in response_ds:
        src, _ = _oe_parse_response_id(row.get("response_id", ""))
        mname = _oe_get_model_name(row.get("model"))
        if mname and src:
            pair_counts[(mname, src)] += 1

    print("  Response counts per pair:")
    any_zero = False
    for model_name, bench in pairs:
        n = pair_counts.get((model_name, bench), 0)
        flag = "  ← no data!" if n == 0 else ""
        print(f"    {model_name}/{bench}: {n:,}{flag}")
        if n == 0:
            any_zero = True
    if any_zero:
        print("  Tip: run --list-models --benchmarks <ids> to find models with data.")
    print()

    # ── Accumulate scores; deduplicate by item_id ─────────────────────────
    score_accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen_items: dict[tuple[str, str], set[str]] = defaultdict(set)
    metric_seen: dict[str, str] = {}
    n_dedup = 0
    n_no_score = 0
    _score_fail_samples: list[Any] = []   # raw scores fields that failed, for diagnostics

    for row in response_ds:
        response_id = row.get("response_id", "")
        source, item_id = _oe_parse_response_id(response_id)
        if source is None:
            continue

        model_name = _oe_get_model_name(row.get("model"))
        if model_name is None or (model_name, source) not in pairs_set:
            continue

        key = (model_name, source)

        # Deduplicate: keep only first response per (item_id, model, benchmark)
        if item_id in seen_items[key]:
            n_dedup += 1
            continue
        seen_items[key].add(item_id)

        spec = OPENEVAL_BENCHMARK_SPECS[source]
        scores_raw = row.get("scores")
        score = _oe_extract_score(scores_raw, spec.metric_name)
        if score is None or not np.isfinite(score):
            n_no_score += 1
            if len(_score_fail_samples) < 5:
                _score_fail_samples.append(scores_raw)
            continue

        if source not in metric_seen:
            mname = _oe_first_metric_name(scores_raw)
            if mname:
                metric_seen[source] = mname

        score_accum[key].append(float(score) * spec.score_scale)

    if n_dedup > 0:
        print(f"  {n_dedup:,} duplicate (item × model) rows removed.")
    if n_no_score > 0:
        print(f"  {n_no_score:,} rows skipped (score missing or non-finite).")
        if _score_fail_samples:
            print("  Sample 'scores' field values that failed extraction:")
            for i, raw in enumerate(_score_fail_samples):
                r = repr(raw)
                if len(r) > 300:
                    r = r[:300] + " …"
                print(f"    [{i}] type={type(raw).__name__}  value={r}")

    # ── Build Corpus objects (preserve pair order) ────────────────────────
    corpora: list[Corpus] = []
    seen_benches: set[str] = set()
    for model_name, bench in pairs:
        spec = OPENEVAL_BENCHMARK_SPECS[bench]
        if bench not in seen_benches:
            mname = metric_seen.get(bench, spec.metric_name or "first")
            print(f"\n  Benchmark: {bench}  [metric used: {mname}]")
            seen_benches.add(bench)
        scores_list = score_accum.get((model_name, bench), [])
        arr = np.array(scores_list, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_corpus_size:
            print(f"  Skip  {model_name}/{bench}: N={len(arr)} < {min_corpus_size}")
            continue
        print(
            f"  OK    {model_name}/{bench}: N={len(arr)}, mean={np.mean(arr):.4f}\n"
            f"        {_score_dist_summary(arr, spec.eval_type)}"
        )
        corpora.append(Corpus(
            model=model_name, benchmark_id=bench,
            eval_type=spec.eval_type, source="openeval",
            scores=arr, corpus_mean=float(np.mean(arr)),
            corpus_size=len(arr),
        ))

    print(f"\n  {len(corpora)} corpora loaded successfully.\n")
    return corpora


# ─────────────────────────────────────────────────────────────────────────────
# CI helpers (binary-specific; general bootstrap methods use bootstrap_ci_1d)
# ─────────────────────────────────────────────────────────────────────────────


def _score_dist_summary(scores: np.ndarray, eval_type: str) -> str:
    """Score distribution diagnostic for corpus load-time output.

    Returns a one-line string normally; appends a second line with 10
    representative sample values when a *** warning is triggered.
    """
    n = len(scores)
    if n == 0:
        return "[empty]"
    pct_zero = 100.0 * np.mean(np.isclose(scores, 0.0))
    pct_one  = 100.0 * np.mean(np.isclose(scores, 1.0))
    looks_binary = (pct_zero + pct_one) >= 99.0
    parts = [
        f"min={scores.min():.4f}",
        f"max={scores.max():.4f}",
        f"std={scores.std():.4f}",
        f"zeros={pct_zero:.1f}%",
        f"ones={pct_one:.1f}%",
    ]
    flag = ""
    if eval_type == "binary" and not looks_binary:
        flag = "  *** NOT cleanly binary — binary CI methods may be unreliable ***"
    elif eval_type != "binary" and looks_binary:
        flag = f"  *** looks binary but eval_type={eval_type!r} — consider changing to binary ***"

    line1 = "[" + "  ".join(parts) + "]" + flag

    if not flag:
        return line1

    # 10 evenly-spaced draws across the sorted distribution
    sorted_scores = np.sort(scores)
    idxs = np.linspace(0, n - 1, min(10, n), dtype=int)
    sample_str = "  ".join(f"{sorted_scores[i]:.4f}" for i in idxs)
    return line1 + f"\n        sample values: [{sample_str}]"


def _binary_successes(values: np.ndarray) -> int:
    """Return Bernoulli successes using a 0.5 threshold for float-safe binary data."""
    return int(np.sum(values >= 0.5))


def _bayes_indep_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Beta(1,1) posterior credible interval for a Bernoulli proportion."""
    n = int(values.shape[0])
    s = _binary_successes(values)
    lo, hi = stats.beta(s + 1, n - s + 1).interval(1.0 - alpha)
    return float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────────
# Progress reporter
# ─────────────────────────────────────────────────────────────────────────────


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
            f"\r  {prefix}[{bar}] {100.0 * frac:6.2f}%  {step:>7d}/{self.total:<7d}  "
            f"ETA {eta_str}  {detail[:40]:<40s}",
            end="", flush=True,
        )
        if is_final:
            print()


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────


def run_simulation(
    corpora: list[Corpus],
    sample_sizes: list[int],
    n_reps: int,
    n_bootstrap: int,
    bayes_n: int,
    alpha: float,
    progress_mode: str = "bar",
    seed: int = 42,
) -> list[SimResult]:
    """
    For each corpus × valid sample size, draw n items WOR from the corpus
    n_reps times, compute all CI methods, and check coverage against the
    corpus mean (= ground truth).

    Sample sizes >= corpus size are silently skipped (WOR requires n < N).
    """
    rng = np.random.default_rng(seed)
    results: list[SimResult] = []

    cells = sum(1 for c in corpora for n in sample_sizes if n < c.corpus_size)
    reporter = _ProgressReporter(cells * n_reps, mode=progress_mode, label="sim")
    step = 0

    for corpus in corpora:
        valid_sizes = [n for n in sample_sizes if n < corpus.corpus_size]
        if not valid_sizes:
            print(
                f"  Warning: all sample sizes >= N={corpus.corpus_size} for "
                f"{corpus.model}/{corpus.benchmark_id}. Skipping."
            )
            continue

        scores_arr = corpus.scores
        N = corpus.corpus_size
        true_mean = corpus.corpus_mean
        is_binary = corpus.eval_type == "binary"

        for n in valid_sizes:
            active_methods = METHODS + [T_INTERVAL_METHOD]
            if is_binary:
                active_methods += [WILSON_METHOD, WALD_METHOD, CP_METHOD, BAYES_SINGLE_METHOD]

            covered: dict[str, int] = {m: 0 for m in active_methods}
            total_w: dict[str, float] = {m: 0.0 for m in active_methods}
            total_t: dict[str, float] = {m: 0.0 for m in active_methods}
            total_t_sq: dict[str, float] = {m: 0.0 for m in active_methods}

            for _rep in range(n_reps):
                # Without-replacement subsample of the corpus
                idxs = rng.choice(N, size=n, replace=False)
                values = scores_arr[idxs]
                obs_mean = float(np.mean(values))

                step += 1
                reporter.update(
                    step,
                    detail=f"{corpus.benchmark_id}/{corpus.model[:20]} n={n}",
                )

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

                    if ci_low <= true_mean <= ci_high:
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
                if ci_low <= true_mean <= ci_high:
                    covered[T_INTERVAL_METHOD] += 1
                total_w[T_INTERVAL_METHOD] += ci_high - ci_low

                if is_binary:
                    successes = _binary_successes(values)
                    n_obs = int(values.shape[0])

                    _t0 = time.perf_counter()
                    ci_low, ci_high = wilson_ci(successes, n_obs, alpha)
                    _el = time.perf_counter() - _t0
                    total_t[WILSON_METHOD] += _el
                    total_t_sq[WILSON_METHOD] += _el * _el
                    if ci_low <= true_mean <= ci_high:
                        covered[WILSON_METHOD] += 1
                    total_w[WILSON_METHOD] += ci_high - ci_low

                    _t0 = time.perf_counter()
                    ci_low, ci_high = wald_ci(successes, n_obs, alpha)
                    _el = time.perf_counter() - _t0
                    total_t[WALD_METHOD] += _el
                    total_t_sq[WALD_METHOD] += _el * _el
                    if ci_low <= true_mean <= ci_high:
                        covered[WALD_METHOD] += 1
                    total_w[WALD_METHOD] += ci_high - ci_low

                    _t0 = time.perf_counter()
                    try:
                        ci_low, ci_high = clopper_pearson_ci(successes, n_obs, alpha)
                    except Exception:
                        ci_low = ci_high = obs_mean
                    _el = time.perf_counter() - _t0
                    total_t[CP_METHOD] += _el
                    total_t_sq[CP_METHOD] += _el * _el
                    if ci_low <= true_mean <= ci_high:
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
                    if ci_low <= true_mean <= ci_high:
                        covered[BAYES_SINGLE_METHOD] += 1
                    total_w[BAYES_SINGLE_METHOD] += ci_high - ci_low

            for method in active_methods:
                results.append(SimResult(
                    model=corpus.model, benchmark_id=corpus.benchmark_id,
                    eval_type=corpus.eval_type, source=corpus.source,
                    corpus_size=N, corpus_mean=true_mean, n=n, method=method,
                    n_reps=n_reps, covered=covered[method], total_width=total_w[method],
                    total_time=total_t[method], total_time_sq=total_t_sq[method],
                ))

    reporter.update(cells * n_reps, detail="done")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────


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
        return float("nan"), float("nan"), float("nan"), float("nan")
    p_hat = successes / total
    mcse = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    return float(p_hat), mcse, max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)


def _rule(width: int, char: str = "─") -> str:
    return char * width


def _print_grid(
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    cells: dict[tuple[str, str], str],
    row_w: int = 22,
    col_w: int = 9,
) -> None:
    total_w = row_w + 2 + (col_w + 2) * len(col_labels)
    print(f"\n  {title}")
    print(f"  {_rule(total_w)}")
    print(f"  {'':<{row_w}}" + "".join(f"  {c:>{col_w}}" for c in col_labels))
    print(f"  {_rule(total_w)}")
    for row in row_labels:
        line = f"  {row:<{row_w}}"
        for col in col_labels:
            val = cells.get((row, col), "─" * col_w)
            line += f"  {val:>{col_w}}"
        print(line)
    print(f"  {_rule(total_w)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main report
# ─────────────────────────────────────────────────────────────────────────────


def print_report(
    results: list[SimResult],
    corpora: list[Corpus],
    sample_sizes: list[int],
    alpha: float,
    n_reps: int,
    source: str,
) -> None:
    target = 1.0 - alpha
    present_methods = {r.method for r in results}
    method_labels = [m for m in REPORT_METHODS if m in present_methods]
    benchmarks = sorted({r.benchmark_id for r in results})
    used_ns = sorted({r.n for r in results})
    n_labels = [f"n={n}" for n in used_ns]

    source_label = {
        "dove": f"DOVE_Lite ({DOVE_REPO})",
        "openeval": f"OpenEval ({OPENEVAL_REPO})",
    }.get(source, source)

    sep = "=" * 76
    print(f"\n{sep}")
    print("  BOOTSTRAP CI COMPARISON — REAL BENCHMARK SUBSAMPLING")
    print(f"  Source  : {source_label}")
    print(f"  Nominal : {target:.0%}   |   reps/cell: {n_reps}")
    print(f"  Sampling: without replacement (WOR) from each corpus")
    print(f"  Estimand: corpus mean (full corpus = 'population')")
    print(f"  ▼ = under-covered (<{target - 0.04:.0%})   ▲ = over-conservative (>{target + 0.04:.0%})")
    print(sep)

    # FPC table
    corpus_sizes = sorted({c.corpus_size for c in corpora})
    if corpus_sizes and used_ns:
        print(f"\n  Finite-Population Correction  sqrt((N-n)/N)  [≈1.0 → negligible bias]")
        print(f"  {'N':<7}" + "".join(f"  {'n='+str(n):>8}" for n in used_ns))
        print(f"  {_rule(7 + 10 * len(used_ns))}")
        for N in corpus_sizes:
            row = f"  {N:<7}"
            for n in used_ns:
                fpc = np.sqrt((N - n) / N) if n < N else float("nan")
                row += f"  {fpc:>8.5f}" if np.isfinite(fpc) else f"  {'n/a':>8}"
            print(row)

    # ── Per-benchmark sections ────────────────────────────────────────────
    for bid in benchmarks:
        bid_results = [r for r in results if r.benchmark_id == bid]
        bid_corpora = [c for c in corpora if c.benchmark_id == bid]
        eval_type = bid_results[0].eval_type if bid_results else "?"

        spec_desc = (
            DOVE_BENCHMARK_SPECS.get(bid, DoveBenchmarkSpec("", "", "", bid)).description
            if source == "dove"
            else getattr(OPENEVAL_BENCHMARK_SPECS.get(bid), "description", bid)
        )

        bid_ns = sorted({r.n for r in bid_results})
        bid_nl = [f"n={n}" for n in bid_ns]

        print(f"\n{'─'*76}")
        print(f"  BENCHMARK: {bid}  [{eval_type}]")
        print(f"  {spec_desc}")
        print(f"{'─'*76}")

        # Corpus stats
        max_n = max(bid_ns) if bid_ns else 0
        print(f"\n  Corpus stats:")
        print(f"  {'Model':<42}  {'N':>6}  {'Corpus mean':>12}  {'FPC(n=max)':>11}")
        print(f"  {_rule(42)}  {_rule(6)}  {_rule(12)}  {_rule(11)}")
        for c in sorted(bid_corpora, key=lambda x: x.corpus_mean):
            fpc = np.sqrt((c.corpus_size - max_n) / c.corpus_size) if max_n < c.corpus_size else float("nan")
            fpc_str = f"{fpc:.5f}" if np.isfinite(fpc) else "n/a"
            print(f"  {c.model:<42}  {c.corpus_size:>6}  {c.corpus_mean:>12.4f}  {fpc_str:>11}")

        # Coverage / width grids averaged across models
        active_methods = [m for m in method_labels if any(r.method == m for r in bid_results)]
        cov_cells: dict[tuple[str, str], str] = {}
        wid_cells: dict[tuple[str, str], str] = {}
        band_cells: dict[tuple[str, str], str] = {}

        for m in active_methods:
            for n, nl in zip(bid_ns, bid_nl):
                subset = [r for r in bid_results if r.method == m and r.n == n]
                if not subset:
                    continue
                cov = float(np.mean([r.covered / r.n_reps for r in subset]))
                wid = float(np.mean([r.total_width / r.n_reps for r in subset]))
                cov_cells[(m, nl)] = f"{cov:.3f}{_cov_marker(cov, target)}"
                wid_cells[(m, nl)] = f"{wid:.4f}"
                tc = sum(r.covered for r in subset)
                tn = sum(r.n_reps for r in subset)
                _, _, lo, hi = _mc_proportion_stats(tc, tn)
                band_cells[(m, nl)] = f"{lo:.3f}-{hi:.3f}"

        row_labels_active = [m for m in active_methods if any((m, nl) in cov_cells for nl in bid_nl)]

        _print_grid(
            f"Coverage averaged across models (target {target:.2f})",
            row_labels=row_labels_active, col_labels=bid_nl, cells=cov_cells,
        )
        _print_grid(
            "Coverage MC 95% Band",
            row_labels=row_labels_active, col_labels=bid_nl, cells=band_cells, col_w=13,
        )
        _print_grid(
            "Mean CI Width — averaged across models",
            row_labels=row_labels_active, col_labels=bid_nl, cells=wid_cells,
        )

        time_cells: dict[tuple[str, str], str] = {}
        for m in row_labels_active:
            for n, nl in zip(bid_ns, bid_nl):
                subset = [r for r in bid_results if r.method == m and r.n == n]
                avg_ms, se_ms = _time_stats(subset)
                if np.isfinite(avg_ms):
                    time_cells[(m, nl)] = f"{avg_ms:.3f}±{se_ms:.3f}"
        _print_grid(
            "Mean CI Time (ms) ± SE — averaged across models",
            row_labels=row_labels_active, col_labels=bid_nl, cells=time_cells, col_w=13,
        )

        # Per-model breakdown
        print(f"\n  Per-model coverage breakdown:")
        for model in sorted({r.model for r in bid_results}):
            model_results = [r for r in bid_results if r.model == model]
            corpus = next((c for c in bid_corpora if c.model == model), None)
            if corpus is None:
                continue
            print(f"\n    {model}  (N={corpus.corpus_size}, mean={corpus.corpus_mean:.4f})")
            m_cov: dict[tuple[str, str], str] = {}
            for m in active_methods:
                for n, nl in zip(bid_ns, bid_nl):
                    hits = [r for r in model_results if r.method == m and r.n == n]
                    if hits:
                        r0 = hits[0]
                        cov = r0.covered / r0.n_reps
                        m_cov[(m, nl)] = f"{cov:.3f}{_cov_marker(cov, target)}"
            per_model_rows = [m for m in active_methods if any((m, nl) in m_cov for nl in bid_nl)]
            _print_grid("Coverage", row_labels=per_model_rows, col_labels=bid_nl, cells=m_cov)

    # ── Overall summary ───────────────────────────────────────────────────
    print(f"\n{'─'*76}")
    print("  OVERALL SUMMARY  (mean across all benchmarks, models, n)")
    print(f"{'─'*76}")

    all_cov: dict[str, list[float]] = defaultdict(list)
    all_wid: dict[str, list[float]] = defaultdict(list)
    all_counts: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for r in results:
        all_cov[r.method].append(r.covered / r.n_reps)
        all_wid[r.method].append(r.total_width / r.n_reps)
        all_counts[r.method].append((r.covered, r.n_reps))

    print(
        f"\n  {'Method':<22}  {'Cov':>6}  {'MCSE':>7}  {'Band95':>13}  {'Width':>8}  {'Dev':>8}  {'Time(ms)':>14}"
    )
    print(f"  {'─'*22}  {'─'*6}  {'─'*7}  {'─'*13}  {'─'*8}  {'─'*8}  {'─'*14}")
    for m in method_labels:
        if m not in all_cov or not all_cov[m]:
            continue
        mc = float(np.mean(all_cov[m]))
        mw = float(np.mean(all_wid[m]))
        dev = mc - target
        mark = _cov_marker(mc, target)
        c_tot = sum(c for c, _ in all_counts[m])
        t_tot = sum(t for _, t in all_counts[m])
        _, mcse, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        avg_ms, se_ms = _time_stats([r for r in results if r.method == m])
        time_str = f"{avg_ms:.3f}±{se_ms:.3f}" if np.isfinite(avg_ms) else "─"
        print(
            f"  {m:<22}  {mc:>5.3f}{mark}  {mcse:>7.4f}  "
            f"{f'{lo:.3f}-{hi:.3f}':>13}  {mw:>8.4f}  {dev:>+8.3f}  {time_str:>14}"
        )
    print()

    # ── Cost × coverage transfer analysis ────────────────────────────────
    print(f"\n{'─'*76}")
    print("  COST × COVERAGE TRANSFER ANALYSIS")
    print("  Methods ranked by mean CI time (cheapest first) within each eval type.")
    print("  ★ = cheapest adequate method (coverage ≥ target−0.04) at that N.")
    print(f"{'─'*76}")

    eval_types_present = list(dict.fromkeys(r.eval_type for r in results))
    for et in eval_types_present:
        et_results = [r for r in results if r.eval_type == et]
        et_methods = [m for m in method_labels if any(r.method == m for r in et_results)]
        et_ns = sorted({r.n for r in et_results})
        et_nl = [f"n={n}" for n in et_ns]

        # Compute avg_time and coverage per (method, n), averaged across benchmarks/models
        method_avg_time: dict[str, float] = {}
        for m in et_methods:
            avg_ms, _ = _time_stats([r for r in et_results if r.method == m])
            method_avg_time[m] = avg_ms if np.isfinite(avg_ms) else float("inf")

        sorted_methods = sorted(et_methods, key=lambda m: method_avg_time[m])

        # For each N, find the cheapest adequate method
        cheapest_adequate: dict[str, str] = {}  # n_label -> method
        for n, nl in zip(et_ns, et_nl):
            for m in sorted_methods:
                subset = [r for r in et_results if r.method == m and r.n == n]
                if not subset:
                    continue
                cov = float(np.mean([r.covered / r.n_reps for r in subset]))
                if cov >= target - 0.04:
                    cheapest_adequate[nl] = m
                    break

        print(f"\n  [{et}]")
        col_w_t = 16
        col_w_c = 9
        hdr = (f"  {'Method':<22}  {'Time(ms)±SE':>{col_w_t}}"
               + "".join(f"  {nl:>{col_w_c}}" for nl in et_nl))
        print(hdr)
        print(f"  {'─'*22}  {'─'*col_w_t}" + "".join(f"  {'─'*col_w_c}" for _ in et_nl))

        for m in sorted_methods:
            avg_ms, se_ms = _time_stats([r for r in et_results if r.method == m])
            time_str = f"{avg_ms:.3f}±{se_ms:.3f}" if np.isfinite(avg_ms) else "─"
            row = f"  {m:<22}  {time_str:>{col_w_t}}"
            for n, nl in zip(et_ns, et_nl):
                subset = [r for r in et_results if r.method == m and r.n == n]
                if not subset:
                    row += f"  {'─':>{col_w_c}}"
                    continue
                cov = float(np.mean([r.covered / r.n_reps for r in subset]))
                star = "★" if cheapest_adequate.get(nl) == m else " "
                row += f"  {cov:.3f}{_cov_marker(cov, target)}{star}".rjust(col_w_c + 2)
            print(row)

        # Speedup summary
        slowest_ms = max(
            (method_avg_time[m] for m in sorted_methods if np.isfinite(method_avg_time[m])),
            default=float("nan"),
        )
        print(f"\n  Transfer summary (★ = recommended cheap default at each N):")
        for nl in et_nl:
            ca = cheapest_adequate.get(nl)
            if ca is None:
                print(f"    {nl}: no method achieved adequate coverage")
            else:
                ca_ms = method_avg_time[ca]
                speedup = slowest_ms / ca_ms if ca_ms > 0 else float("inf")
                print(f"    {nl}: {ca:<22} {ca_ms:.4f} ms  ({speedup:.0f}× faster than slowest bootstrap)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Cost × coverage scatter plot
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_COLORS: dict[str, str] = {
    "bootstrap":          "#1f77b4",
    "bca":                "#2ca02c",
    "bayes_bootstrap":    "#ff7f0e",
    "smooth_bootstrap":   "#9467bd",
    "bootstrap_t":        "#d62728",
    "t_interval":         "#8c564b",
    "wilson":             "#e377c2",
    "wald":               "#7f7f7f",
    "clopper_pearson":    "#bcbd22",
    "bayes_indep":        "#17becf",
}
_N_MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*"]


def save_cost_plot(
    *,
    results: list[SimResult],
    alpha: float,
    n_reps: int,
    source: str,
    out_path: str,
) -> None:
    """Scatter plot: x = mean CI time (log ms), y = coverage; one subplot per eval type."""
    if not results:
        print(f"Skipped cost plot (no data): {out_path}")
        return

    target = 1.0 - alpha
    present_methods = {r.method for r in results}
    method_labels = [m for m in REPORT_METHODS if m in present_methods]
    eval_types = list(dict.fromkeys(r.eval_type for r in results))
    sample_sizes = sorted({r.n for r in results})

    nrows = len(eval_types)
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

    for row_idx, et in enumerate(eval_types):
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

    source_label = source.upper()
    fig.suptitle(
        f"{source_label} — Cost × Coverage Trade-off\n"
        f"x = mean CI compute time  |  y = empirical coverage  |  target = {target:.0%}  |  reps={n_reps}",
        fontsize=11,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=[0.02, 0.02, 0.82, 0.93])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cost plot: {out}")


def save_coverage_vs_n_plot(
    *,
    results: list[SimResult],
    alpha: float,
    n_reps: int,
    source: str,
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
            "corpus_key": f"{r.model}/{r.benchmark_id}",
            "method": r.method,
            "n": r.n,
            "coverage": r.covered / r.n_reps,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df = df[df["method"].isin(method_labels)]

    # Average across reps within each (eval_type, corpus, method, n) cell,
    # then compute mean ± std across corpora.
    corpus_level = (
        df.groupby(["eval_type", "corpus_key", "method", "n"], as_index=False)
          .agg(coverage=("coverage", "mean"))
    )
    agg = (
        corpus_level.groupby(["eval_type", "method", "n"], as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            coverage_count=("coverage", "count"),
        )
    )

    eval_types_present = sorted(agg["eval_type"].unique())
    if not eval_types_present:
        print(f"Skipped coverage-vs-n plot (no eval types): {out_path}")
        return

    palette = {m: _METHOD_COLORS.get(m, "#333333") for m in method_labels}

    fig, axes = plt.subplots(
        1, len(eval_types_present),
        figsize=(5.5 * len(eval_types_present), 5),
        squeeze=False,
    )

    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        et_agg = agg[agg["eval_type"] == et].copy()
        et_methods = [m for m in method_labels if m in et_agg["method"].values]

        sns.lineplot(
            data=et_agg,
            x="n",
            y="coverage_mean",
            hue="method",
            hue_order=et_methods,
            palette=palette,
            marker="o",
            ax=ax,
        )

        for method, sub in et_agg.groupby("method"):
            if sub["coverage_std"].isna().all():
                continue
            color = _METHOD_COLORS.get(str(method), "#333333")
            se = sub["coverage_std"] / np.sqrt(sub["coverage_count"])
            ax.errorbar(
                sub["n"],
                sub["coverage_mean"],
                yerr=se,
                fmt="o-",
                color=color,
                capsize=3,
                alpha=0.9,
            )

        ns = sorted(et_agg["n"].unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.axhline(target, linestyle="--", color="tab:cyan", linewidth=1.2)
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Empirical coverage" if col_idx == 0 else "")
        ax.set_title(et.upper())
        ax.legend(title="Method", fontsize=7.5, title_fontsize=8)

    source_label = source.upper()
    fig.suptitle(
        f"{source_label} — Coverage vs. Sample Size | reps={n_reps} | alpha={alpha}",
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
    alpha: float,
    n_reps: int,
    source: str,
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
            "corpus_key": f"{r.model}/{r.benchmark_id}",
            "method": r.method,
            "n": r.n,
            "width": r.total_width / r.n_reps,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df = df[df["method"].isin(method_labels)]

    corpus_level = (
        df.groupby(["eval_type", "corpus_key", "method", "n"], as_index=False)
          .agg(width=("width", "mean"))
    )
    agg = (
        corpus_level.groupby(["eval_type", "method", "n"], as_index=False)
        .agg(
            width_mean=("width", "mean"),
            width_std=("width", "std"),
            width_count=("width", "count"),
        )
    )

    eval_types_present = sorted(agg["eval_type"].unique())
    if not eval_types_present:
        print(f"Skipped width-vs-n plot (no eval types): {out_path}")
        return

    palette = {m: _METHOD_COLORS.get(m, "#333333") for m in method_labels}

    fig, axes = plt.subplots(
        1, len(eval_types_present),
        figsize=(5.5 * len(eval_types_present), 5),
        squeeze=False,
    )

    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        et_agg = agg[agg["eval_type"] == et].copy()
        et_methods = [m for m in method_labels if m in et_agg["method"].values]

        sns.lineplot(
            data=et_agg,
            x="n",
            y="width_mean",
            hue="method",
            hue_order=et_methods,
            palette=palette,
            marker="o",
            ax=ax,
        )

        for method, sub in et_agg.groupby("method"):
            if sub["width_std"].isna().all():
                continue
            color = _METHOD_COLORS.get(str(method), "#333333")
            se = sub["width_std"] / np.sqrt(sub["width_count"])
            ax.errorbar(
                sub["n"],
                sub["width_mean"],
                yerr=se,
                fmt="o-",
                color=color,
                capsize=3,
                alpha=0.9,
            )

        ns = sorted(et_agg["n"].unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Mean CI width" if col_idx == 0 else "")
        ax.set_title(et.upper())
        ax.legend(title="Method", fontsize=7.5, title_fontsize=8)

    source_label = source.upper()
    fig.suptitle(
        f"{source_label} — CI Width vs. Sample Size | reps={n_reps} | alpha={alpha}",
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


# ─────────────────────────────────────────────────────────────────────────────
# Save artifacts
# ─────────────────────────────────────────────────────────────────────────────


def save_artifacts(
    *,
    results: list[SimResult],
    corpora: list[Corpus],
    alpha: float,
    sample_sizes: list[int],
    n_reps: int,
    source: str,
    out_dir: str,
    run_stem: str,
) -> None:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    csv_path = out_base / f"{run_stem}_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "source", "benchmark_id", "eval_type", "model",
            "corpus_size", "corpus_mean", "n", "method", "n_reps",
            "covered", "total_width", "coverage", "mean_width",
            "mcse", "band95_low", "band95_high",
            "avg_time_ms", "se_time_ms",
        ])
        for r in results:
            coverage = r.covered / r.n_reps
            mean_width = r.total_width / r.n_reps
            _, mcse, lo, hi = _mc_proportion_stats(r.covered, r.n_reps)
            avg_ms, se_ms = _time_stats([r])
            writer.writerow([
                r.source, r.benchmark_id, r.eval_type, r.model,
                r.corpus_size, f"{r.corpus_mean:.8f}", r.n, r.method,
                r.n_reps, r.covered, f"{r.total_width:.8f}",
                f"{coverage:.8f}", f"{mean_width:.8f}",
                f"{mcse:.8f}", f"{lo:.8f}", f"{hi:.8f}",
                f"{avg_ms:.6f}" if np.isfinite(avg_ms) else "",
                f"{se_ms:.6f}" if np.isfinite(se_ms) else "",
            ])

    summary_path = out_base / f"{run_stem}_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(
            results=results, corpora=corpora, sample_sizes=sample_sizes,
            alpha=alpha, n_reps=n_reps, source=source,
        )
    summary_path.write_text(buf.getvalue(), encoding="utf-8")

    print(f"Saved results : {csv_path}")
    print(f"Saved log     : {summary_path}")


def save_plots(
    *,
    results: list[SimResult],
    corpora: list[Corpus],
    alpha: float,
    n_reps: int,
    source: str,
    out_path: str,
) -> None:
    """Save a per-benchmark coverage + width figure."""
    if not results:
        print(f"Skipped plot (no data): {out_path}")
        return

    target = 1.0 - alpha
    benchmarks = sorted({r.benchmark_id for r in results})
    present_methods = {r.method for r in results}
    method_labels = [m for m in REPORT_METHODS if m in present_methods]

    nrows = len(benchmarks)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=2, figsize=(13.5, 4.5 * nrows), squeeze=False,
        gridspec_kw={"wspace": 0.32, "hspace": 0.38},
    )
    col_titles = [
        f"Coverage (target={target:.2f}; red = MC95 band)",
        "Mean CI Width",
    ]
    box_kwargs: dict = dict(
        vert=False, showmeans=True, meanline=False, patch_artist=False,
        whiskerprops={"linewidth": 1.1, "color": "black"},
        capprops={"linewidth": 1.1, "color": "black"},
        medianprops={"linewidth": 1.6, "color": "black"},
        boxprops={"linewidth": 1.3, "color": "black"},
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 4},
        flierprops={"marker": "o", "markerfacecolor": "black",
                    "markeredgecolor": "black", "markersize": 2.5, "alpha": 0.5},
    )

    for row_idx, bid in enumerate(benchmarks):
        bid_results = [r for r in results if r.benchmark_id == bid]
        bid_methods = [m for m in method_labels if any(r.method == m for r in bid_results)]

        cov_series: list[np.ndarray] = []
        wid_series: list[np.ndarray] = []
        cov_uncertainty: list[tuple[float, float, float]] = []

        for m in bid_methods:
            subset = [r for r in bid_results if r.method == m]
            if not subset:
                continue
            cov_series.append(np.array([r.covered / r.n_reps for r in subset]))
            wid_series.append(np.array([r.total_width / r.n_reps for r in subset]))
            c_tot = sum(r.covered for r in subset)
            t_tot = sum(r.n_reps for r in subset)
            p_hat, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
            cov_uncertainty.append((p_hat, lo, hi))

        for col_idx, (ax, series) in enumerate(zip(axes[row_idx], [cov_series, wid_series])):
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11)
            if not series:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                ax.set_yticks([])
                continue

            ax.boxplot(series, tick_labels=bid_methods, **box_kwargs)
            ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
            ax.tick_params(axis="y", labelsize=9)
            ax.tick_params(axis="x", labelsize=9)
            ax.invert_yaxis()

            if col_idx == 0:
                ax.axvspan(max(0.0, target - 0.04), min(1.0, target + 0.04),
                           color="#DDDDDD", alpha=0.35, zorder=0)
                ax.axvline(target, color="black", linewidth=1.2)
                ax.set_xlim(0.0, 1.0)
                for y_pos, (p_hat, lo, hi) in enumerate(cov_uncertainty, start=1):
                    if np.isnan(lo):
                        continue
                    ax.hlines(y_pos, lo, hi, color="tab:red", linewidth=2.0, zorder=5)
                    ax.vlines([lo, hi], y_pos - 0.15, y_pos + 0.15,
                              color="tab:red", linewidth=1.4, zorder=5)
                    if not np.isnan(p_hat):
                        ax.plot(p_hat, y_pos, marker="|", color="tab:red",
                                markersize=9, markeredgewidth=1.6, zorder=6)
                ax.set_ylabel(bid, fontsize=10.5)
            else:
                x_max = max((float(np.max(s)) for s in series if s.size > 0), default=1.0)
                ax.set_xlim(0.0, x_max * 1.08 if x_max > 0 else 1.0)

            ax.set_xlabel(
                "Coverage across (model × n) cells" if col_idx == 0
                else "CI width across (model × n) cells",
                fontsize=9,
            )

    models_shown = sorted({c.model for c in corpora})
    source_label = source.upper()
    fig.suptitle(
        f"{source_label} — CI Method Comparison on Real Benchmark Data\n"
        f"reps={n_reps} | alpha={alpha} | models: {', '.join(models_shown[:4])}"
        + (" …" if len(models_shown) > 4 else ""),
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=[0.03, 0.02, 0.995, 0.95])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot    : {out}")


def save_eval_type_plot(
    *,
    results: list[SimResult],
    corpora: list[Corpus],
    alpha: float,
    n_reps: int,
    source: str,
    out_path: str,
) -> None:
    """Aggregate box-plot figure: one row per eval type, columns = coverage and mean CI width.

    Mirrors sim_compare_boot's save_metric_plot — each box aggregates across all
    (model × benchmark × n) cells of that eval type.
    """
    if not results:
        print(f"Skipped eval-type plot (no data): {out_path}")
        return

    target = 1.0 - alpha
    present_methods = {r.method for r in results}
    method_labels = [m for m in REPORT_METHODS if m in present_methods]

    eval_types_present = [et for et in ["binary", "continuous", "likert", "grades"]
                          if any(r.eval_type == et for r in results)]
    if not eval_types_present:
        print(f"Skipped eval-type plot (no eval types): {out_path}")
        return

    box_kwargs: dict = dict(
        vert=False, showmeans=True, meanline=False, patch_artist=False,
        whiskerprops={"linewidth": 1.2, "color": "black"},
        capprops={"linewidth": 1.2, "color": "black"},
        medianprops={"linewidth": 1.8, "color": "black"},
        boxprops={"linewidth": 1.4, "color": "black"},
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 4.5},
        flierprops={"marker": "o", "markerfacecolor": "black",
                    "markeredgecolor": "black", "markersize": 2.8, "alpha": 0.55},
    )

    col_titles = [
        f"Coverage (target={target:.2f}; red interval = MC95)",
        "Mean CI Width",
    ]

    fig, axes = plt.subplots(
        nrows=len(eval_types_present), ncols=2,
        figsize=(14.8, 4.0 * len(eval_types_present)),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.34, "hspace": 0.30},
    )
    if len(eval_types_present) == 1:
        axes = np.array([axes])

    for r_idx, et in enumerate(eval_types_present):
        et_results = [r for r in results if r.eval_type == et]
        et_methods = [m for m in method_labels if any(r.method == m for r in et_results)]

        cov_series: list[np.ndarray] = []
        wid_series: list[np.ndarray] = []
        cov_uncertainty: list[tuple[float, float, float]] = []

        for method in et_methods:
            subset = [r for r in et_results if r.method == method]
            if not subset:
                continue
            cov_series.append(np.array([r.covered / r.n_reps for r in subset]))
            wid_series.append(np.array([r.total_width / r.n_reps for r in subset]))
            c_tot = sum(r.covered for r in subset)
            t_tot = sum(r.n_reps for r in subset)
            p_hat, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
            cov_uncertainty.append((p_hat, lo, hi))

        for c_idx, (ax, series, xlabel) in enumerate(zip(
            axes[r_idx],
            [cov_series, wid_series],
            ["Coverage across (model × benchmark × n) cells",
             "CI width across (model × benchmark × n) cells"],
        )):
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
                ax.axvspan(max(0.0, target - 0.04), min(1.0, target + 0.04),
                           color="#DDDDDD", alpha=0.35, zorder=0)
                ax.axvline(target, color="black", linestyle="-", linewidth=1.2)
                ax.set_xlim(0.0, 1.0)
                for y_pos, (p_hat, lo, hi) in enumerate(cov_uncertainty, start=1):
                    if np.isnan(lo) or np.isnan(hi):
                        continue
                    ax.hlines(y=y_pos, xmin=lo, xmax=hi, color="tab:red", linewidth=2.1, zorder=5)
                    ax.vlines([lo, hi], y_pos - 0.15, y_pos + 0.15,
                              color="tab:red", linewidth=1.5, zorder=5)
                    if not np.isnan(p_hat):
                        ax.plot(p_hat, y_pos, marker="|", color="tab:red",
                                markersize=10, markeredgewidth=1.8, zorder=6)
                ax.set_ylabel(et.upper(), fontsize=10.5)
            else:
                x_max = max((float(np.max(s)) for s in series if s.size > 0), default=1.0)
                ax.set_xlim(0.0, x_max * 1.08 if x_max > 0 else 1.0)

            if bp and "means" in bp:
                for mean_artist in bp["means"]:
                    mean_artist.set_zorder(4)

    models_shown = sorted({c.model for c in corpora})
    source_label = source.upper()
    sample_sizes = sorted({r.n for r in results})
    size_text = ", ".join(str(n) for n in sample_sizes)

    fig.suptitle(
        f"{source_label} — CI Method Comparison by Eval Type (Interval / Box Plots)\n"
        f"reps={n_reps} | alpha={alpha} | n={size_text} | "
        f"models: {', '.join(models_shown[:4])}" + (" …" if len(models_shown) > 4 else ""),
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=[0.03, 0.02, 0.995, 0.95], w_pad=2.6)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved eval-type plot: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="dove",
        help="Benchmark dataset source: 'dove' (default) or 'openeval'",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help=(
            "Print all unique model names available in the chosen source and exit. "
            "Useful for discovering model names to pass to --models."
        ),
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help=(
            "Print all available benchmark IDs (with response counts) for the chosen "
            "source and exit.  Useful for discovering IDs to pass to --benchmarks."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,   # resolved per source below
        metavar="ID",
        help=(
            "Benchmark IDs to run. "
            f"DOVE defaults (from DOVE_DEFAULT_PAIRS): {DOVE_DEFAULT_BENCHMARKS}. "
            f"OpenEval defaults: {OPENEVAL_DEFAULT_BENCHMARKS}. "
            f"DOVE available: {list(DOVE_BENCHMARK_SPECS)}. "
            f"OpenEval available: {list(OPENEVAL_BENCHMARK_SPECS)}."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,   # resolved per source below
        metavar="NAME",
        help=(
            "Model names to include (exact match to dataset names). "
            "Defaults to a preset list per source. "
            "For OpenEval, run --list-models to discover available names."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace access token (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help="Local directory for caching downloaded HF dataset files.",
    )
    parser.add_argument(
        "--min-corpus-size",
        type=int,
        default=50,
        metavar="N",
        help="Skip (model, benchmark) pairs with fewer than N instances (default: 50)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=200,
        metavar="N",
        help="Monte Carlo reps per (corpus, n) cell (default: 200)",
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
        default=[10, 20, 50, 100],
        metavar="N",
        help="Sample sizes to evaluate (default: 10 20 50 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed (default: 42)",
    )
    parser.add_argument(
        "--progress",
        choices=PROGRESS_MODES,
        default="bar",
        help="Progress display mode (default: bar)",
    )
    parser.add_argument(
        "--plots",
        choices=PLOT_MODES,
        default="save",
        help="Post-run plot mode (default: save)",
    )
    parser.add_argument(
        "--save-results",
        choices=RESULTS_MODES,
        default="save",
        help="Write CSV + log (default: save)",
    )
    parser.add_argument(
        "--out-dir",
        default="simulations/out",
        help="Base output directory (default: simulations/out)",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory for saved plots (default: <out-dir>/plots)",
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")
    source = args.source

    # ── --list-models shortcut ────────────────────────────────────────────
    if args.list_models:
        if source == "dove":
            print("DOVE_Lite model names are the top-level directory names in the repo.")
            print(f"Browse: https://huggingface.co/datasets/{DOVE_REPO}/tree/main")
            print(f"\nPreset default pairs (model, benchmark):")
            for m, b in DOVE_DEFAULT_PAIRS:
                print(f"  {m:<38}  {b}")
        else:
            bench_filter = args.benchmarks  # None → show all benchmarks
            coverage = list_openeval_models(
                openeval_repo=OPENEVAL_REPO, hf_token=hf_token, cache_dir=args.cache_dir,
                benchmark_filter=bench_filter,
            )
            target_benches = sorted(bench_filter) if bench_filter else None
            if bench_filter:
                print(f"\nOpenEval models with data for benchmarks {bench_filter} "
                      f"({len(coverage)} models):")
            else:
                print(f"\nOpenEval model names ({len(coverage)} total):")

            if not coverage:
                print("  (none found — check benchmark IDs with --list-benchmarks)")
            elif target_benches:
                # Tabular view: one column per requested benchmark
                col_w = max(len(b) for b in target_benches)
                header = "  {:<35s}  {}".format(
                    "Model",
                    "  ".join(f"{b:>{col_w}}" for b in target_benches),
                )
                print(header)
                print("  " + "-" * (len(header) - 2))
                for model_name, bench_counts in sorted(coverage.items()):
                    row_cells = []
                    for b in target_benches:
                        n = bench_counts.get(b, 0)
                        row_cells.append(f"{n:>{col_w},}" if n else f"{'—':>{col_w}}")
                    print(f"  {model_name:<35s}  {'  '.join(row_cells)}")
            else:
                # No filter: show a full benchmark matrix for readability
                all_benches = sorted({b for counts in coverage.values() for b in counts})
                model_w = max(35, max(len(m) for m in coverage))
                col_ws = {
                    b: max(len(b), max(len(f"{counts.get(b, 0):,}") for counts in coverage.values()))
                    for b in all_benches
                }
                header = "  {:<{mw}s}  {}".format(
                    "Model",
                    "  ".join(f"{b:>{col_ws[b]}}" for b in all_benches),
                    mw=model_w,
                )
                print(header)
                print("  " + "-" * (len(header) - 2))
                for model_name, bench_counts in sorted(coverage.items()):
                    row_cells = []
                    for b in all_benches:
                        n = bench_counts.get(b, 0)
                        row_cells.append(f"{n:>{col_ws[b]},}" if n else f"{'—':>{col_ws[b]}}")
                    print(f"  {model_name:<{model_w}s}  {'  '.join(row_cells)}")
        return

    # ── --list-benchmarks shortcut ────────────────────────────────────────
    if args.list_benchmarks:
        if source == "dove":
            print("DOVE_Lite benchmark IDs (available in this simulation):")
            for bid, spec in DOVE_BENCHMARK_SPECS.items():
                print(f"  {bid:20s}  eval_type={spec.eval_type}  file={spec.file_name}")
        else:
            counts = list_openeval_benchmarks(
                openeval_repo=OPENEVAL_REPO, hf_token=hf_token, cache_dir=args.cache_dir,
            )
            print(f"\nOpenEval benchmark sources ({len(counts)} total, sorted by name):")
            for bench_id, count in counts.items():
                known = bench_id in OPENEVAL_BENCHMARK_SPECS
                tag = ""
                if known:
                    spec = OPENEVAL_BENCHMARK_SPECS[bench_id]
                    tag = f"  eval_type={spec.eval_type}  ← supported"
                print(f"  {bench_id:25s}  {count:6,} responses{tag}")
        return

    # ── Resolve defaults per source ───────────────────────────────────────
    if source in ("dove", "all"):
        if args.models is None and args.benchmarks is None:
            dove_pairs = DOVE_DEFAULT_PAIRS
        else:
            bms = args.benchmarks or list(dict.fromkeys(b for _, b in DOVE_DEFAULT_PAIRS))
            mds = args.models     or list(dict.fromkeys(m for m, _ in DOVE_DEFAULT_PAIRS))
            dove_pairs = [(m, b) for m in mds for b in bms]
    else:
        dove_pairs = []

    if source in ("openeval", "all"):
        # OpenEval: work with explicit (model, benchmark) pairs, not a cross-product,
        # because dataset coverage is sparse.
        if args.models is None and args.benchmarks is None:
            oe_pairs = OPENEVAL_DEFAULT_PAIRS
        else:
            bms = args.benchmarks or list(dict.fromkeys(b for _, b in OPENEVAL_DEFAULT_PAIRS))
            mds = args.models     or list(dict.fromkeys(m for m, _ in OPENEVAL_DEFAULT_PAIRS))
            oe_pairs = [(m, b) for m in mds for b in bms]
    else:
        oe_pairs = []

    benchmarks = list(dict.fromkeys(b for _, b in dove_pairs + oe_pairs))
    models     = list(dict.fromkeys(m for m, _ in dove_pairs + oe_pairs))

    # ── Print run config ──────────────────────────────────────────────────
    print(f"\nReal-Data Bootstrap CI Simulation")
    print(f"  Source          : {source}")
    if dove_pairs:
        print(f"  DOVE pairs      : {dove_pairs}")
    if oe_pairs:
        print(f"  OpenEval pairs  : {oe_pairs}")
    print(f"  Reps per cell   : {args.reps}")
    print(f"  Bootstrap draws : {args.bootstrap_n}")
    print(f"  Alpha / CI level: {args.alpha} / {(1 - args.alpha):.0%}")
    print(f"  Sample sizes    : {args.sizes}")
    print(f"  Seed            : {args.seed}")
    print(f"  HF token        : {'set' if hf_token else 'not set (using cached login)'}")
    print(f"  Cache dir       : {args.cache_dir or 'HF default'}")

    # ── Load corpora ──────────────────────────────────────────────────────
    corpora = []
    if dove_pairs:
        corpora += build_dove_corpora(
            pairs=dove_pairs,
            dove_repo=DOVE_REPO, hf_token=hf_token,
            cache_dir=args.cache_dir, min_corpus_size=args.min_corpus_size,
        )
    if oe_pairs:
        corpora += build_openeval_corpora(
            pairs=oe_pairs,
            openeval_repo=OPENEVAL_REPO, hf_token=hf_token,
            cache_dir=args.cache_dir, min_corpus_size=args.min_corpus_size,
        )

    if not corpora:
        hints = (
            "  • HuggingFace access approval for nlphuji/DOVE_Lite (DOVE)\n"
            "  • Login: huggingface-cli login  (or --hf-token TOKEN)\n"
            "  • Model/benchmark names must match exactly (see --list-models, --list-benchmarks)\n"
        )
        print(f"\nNo corpora loaded. Check:\n{hints}")
        sys.exit(1)

    # ── Simulate ──────────────────────────────────────────────────────────
    cells = sum(1 for c in corpora for n in args.sizes if n < c.corpus_size)
    print(
        f"\nRunning {cells} (corpus, n) cells × {args.reps} reps × {len(METHODS)} "
        f"bootstrap methods = {cells * args.reps * len(METHODS):,} CI calls …"
    )

    results = run_simulation(
        corpora=corpora, sample_sizes=args.sizes, n_reps=args.reps,
        n_bootstrap=args.bootstrap_n, bayes_n=args.bayes_n, alpha=args.alpha,
        progress_mode=args.progress, seed=args.seed,
    )

    print_report(
        results=results, corpora=corpora, sample_sizes=args.sizes,
        alpha=args.alpha, n_reps=args.reps, source=source,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_stem = f"sim_{source}_reps{args.reps}_{stamp}"

    if args.save_results == "save":
        save_artifacts(
            results=results, corpora=corpora, alpha=args.alpha,
            sample_sizes=args.sizes, n_reps=args.reps, source=source,
            out_dir=args.out_dir, run_stem=run_stem,
        )

    if args.plots == "save":
        save_plots(
            results=results, corpora=corpora, alpha=args.alpha,
            n_reps=args.reps, source=source,
            out_path=str(Path(plots_dir) / f"{run_stem}_overview.png"),
        )
        save_eval_type_plot(
            results=results, corpora=corpora, alpha=args.alpha,
            n_reps=args.reps, source=source,
            out_path=str(Path(plots_dir) / f"{run_stem}_by_eval_type.png"),
        )
        save_cost_plot(
            results=results, alpha=args.alpha, n_reps=args.reps,
            source=source,
            out_path=str(Path(plots_dir) / f"{run_stem}_cost_coverage.png"),
        )
        save_coverage_vs_n_plot(
            results=results, alpha=args.alpha, n_reps=args.reps,
            source=source,
            out_path=str(Path(plots_dir) / f"{run_stem}_coverage_vs_n.png"),
        )
        save_width_vs_n_plot(
            results=results, alpha=args.alpha, n_reps=args.reps,
            source=source,
            out_path=str(Path(plots_dir) / f"{run_stem}_width_vs_n.png"),
        )


if __name__ == "__main__":
    main()
