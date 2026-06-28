"""Real-data adapter: DOVE_Lite / OpenEval corpora exposed as CISources.

Ported from ``simulations/sim_dove.py``'s DOVE_Lite and OpenEval loading
logic. A real-data ``Corpus`` (one (model, benchmark) pair's full set of
finite, deduplicated instance scores) is exposed through the same
``CISource`` interface used by ``scenarios/synthetic.py``: ``generate(rng,
n)`` draws an i.i.d.-without-replacement subsample of size n from the corpus,
and ``true_mean`` is the corpus mean (the "ground truth" estimand). This lets
``cases/ci_single.py`` run one simulation loop over either synthetic or real
sources.

Sample sizes >= a corpus's size cannot be drawn without replacement and are
silently skipped by the caller (see ``cases/ci_single.py``).
"""

from __future__ import annotations

import csv as _csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from . import CISource, CIPairSource

SOURCES = ["dove", "openeval", "all"]
PAIR_SOURCES = ["dove", "openeval", "inspect", "all"]

# ─────────────────────────────────────────────────────────────────────────────
# DOVE_Lite -- constants and benchmark specs
# ─────────────────────────────────────────────────────────────────────────────

DOVE_REPO = "nlphuji/DOVE_Lite"

# Curated default (model, benchmark_id) pairs for DOVE_Lite.
# Each pair is confirmed to have data; the cross-product of all models x all
# benchmarks does NOT work because coverage varies by model family.
# Run simulations/explore_dove.py to discover more valid combinations.
DOVE_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("Llama-3.2-1B-Instruct", "hellaswag"),
    ("OLMoE-1B-7B-0924-Instruct", "hellaswag"),
    ("Mistral-7B-Instruct-v0.3", "hellaswag"),
    ("Meta-Llama-3-8B-Instruct", "hellaswag"),
    ("Llama-3.2-1B-Instruct", "arc_challenge"),
    ("OLMoE-1B-7B-0924-Instruct", "arc_challenge"),
    ("Mistral-7B-Instruct-v0.3", "arc_challenge"),
    ("Meta-Llama-3-8B-Instruct", "arc_challenge"),
    ("GPT-4o-mini", "gsm8k"),
    ("Llama-3.3-70B-Instruct", "gsm8k"),
    ("GPT-4o-mini", "squad"),
    ("Llama-3.3-70B-Instruct", "squad"),
    ("Llama-3.2-1B-Instruct", "quality"),
    ("Mistral-7B-Instruct-v0.3", "quality"),
    ("Meta-Llama-3-8B-Instruct", "quality"),
    ("GPT-4o-mini", "cnn_dailymail"),
    ("Llama-3.3-70B-Instruct", "cnn_dailymail"),
    ("GPT-4o-mini", "wmt14.cs-en"),
    ("Llama-3.3-70B-Instruct", "wmt14.cs-en"),
]


@dataclass
class DoveBenchmarkSpec:
    benchmark_id: str
    file_name: str
    eval_type: str  # "binary" | "continuous" | "likert" | "grades"
    description: str
    language: str = "en"
    shots: int = 0
    score_bounds: tuple[float, float] | None = None  # (lo, hi) -> rescale to [0,1]


DOVE_BENCHMARK_SPECS: dict[str, DoveBenchmarkSpec] = {
    "hellaswag": DoveBenchmarkSpec(
        benchmark_id="hellaswag", file_name="hellaswag.parquet",
        eval_type="binary", description="HellaSwag commonsense NLI (0/1 correct)",
    ),
    "arc_challenge": DoveBenchmarkSpec(
        benchmark_id="arc_challenge", file_name="ai2_arc.arc_challenge.parquet",
        eval_type="binary", description="ARC Challenge science QA (0/1 correct)",
    ),
    "gsm8k": DoveBenchmarkSpec(
        benchmark_id="gsm8k", file_name="gsm8k.parquet",
        eval_type="binary", description="GSM8K grade-school math (0/1 correct)",
    ),
    "squad": DoveBenchmarkSpec(
        benchmark_id="squad", file_name="squad.parquet",
        eval_type="binary", description="SQuAD reading comprehension (0/1 correct)",
    ),
    "quality": DoveBenchmarkSpec(
        benchmark_id="quality", file_name="quality.parquet",
        eval_type="binary", description="QuALITY long-context reading comprehension (0/1 correct)",
    ),
    "cnn_dailymail": DoveBenchmarkSpec(
        benchmark_id="cnn_dailymail", file_name="cnn_dailymail.parquet",
        eval_type="continuous", description="CNN/DailyMail summarization quality score in [0,1]",
    ),
    "wmt14.cs-en": DoveBenchmarkSpec(
        benchmark_id="wmt14.cs-en", file_name="wmt14.cs-en.parquet",
        eval_type="continuous", description="WMT14 Czech-to-English translation quality score in [0,1]",
    ),
}

DOVE_DEFAULT_BENCHMARKS: list[str] = list(dict.fromkeys(b for _, b in DOVE_DEFAULT_PAIRS))

# ─────────────────────────────────────────────────────────────────────────────
# OpenEval -- constants and benchmark specs
# ─────────────────────────────────────────────────────────────────────────────

OPENEVAL_REPO = "human-centered-eval/OpenEval"

# response_id format: {source}_{YYYYMMDDTHHMMSSZ}_{index}_{model_name}_{run}
# item_id  is just  : {source}_{YYYYMMDDTHHMMSSZ}_{index}
_OE_RESP_ID_RE = re.compile(r"^(.+?)_(\d{8}T\d{6}Z)_(\d+)")

OPENEVAL_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("falcon-7b-instruct", "mmlu-pro"),
    ("gpt-4o", "culturalbench"),
    ("o4-mini", "opentom"),
    ("llama-65b", "bbq"),
    ("vicuna-13b-v1.3", "cnndm"),
    ("DeepSeek-V3-0324", "do-not-answer"),
    ("DeepSeek-R1", "emobench"),
    ("qwen-3-80b-instruct", "gpqa"),
    ("gpt-4.1-mini", "hi-tom"),
    ("gemma-3-27b-it", "ifeval"),
    ("falcon-40b-instruct", "imdb"),
    ("qwen-2.5-72b-instruct", "omni-math"),
    ("kimi-k2", "salad-bench"),
    ("llama-2-70b", "xsum"),
    ("grok-4", "truthfulqa"),
]


@dataclass
class OpenEvalBenchmarkSpec:
    benchmark_id: str
    eval_type: str
    description: str
    metric_name: str | None = None
    score_scale: float = 1.0
    score_bounds: tuple[float, float] | None = None


OPENEVAL_BENCHMARK_SPECS: dict[str, OpenEvalBenchmarkSpec] = {
    "mmlu-pro": OpenEvalBenchmarkSpec(
        benchmark_id="mmlu-pro", eval_type="binary",
        description="MMLU-Pro knowledge/reasoning (0/1 correct)",
    ),
    "gpqa": OpenEvalBenchmarkSpec(
        benchmark_id="gpqa", eval_type="binary",
        description="GPQA graduate-level science reasoning (0/1 correct)",
    ),
    "boolq": OpenEvalBenchmarkSpec(
        benchmark_id="boolq", eval_type="binary",
        description="BoolQ yes/no reading comprehension (0/1 correct)",
    ),
    "imdb": OpenEvalBenchmarkSpec(
        benchmark_id="imdb", eval_type="binary",
        description="IMDB sentiment classification (0/1 correct)",
    ),
    "truthfulqa": OpenEvalBenchmarkSpec(
        benchmark_id="truthfulqa", eval_type="continuous",
        description="TruthfulQA BLEU-max fluency/truthfulness score in [0,1]",
        metric_name="bleu_max", score_scale=0.01,
    ),
    "culturalbench": OpenEvalBenchmarkSpec(
        benchmark_id="culturalbench", eval_type="binary",
        description="CulturalBench cultural knowledge QA (0/1 correct)",
    ),
    "opentom": OpenEvalBenchmarkSpec(
        benchmark_id="opentom", eval_type="binary",
        description="OpenToM Theory-of-Mind reasoning QA (0/1 correct)",
    ),
    "bbq": OpenEvalBenchmarkSpec(
        benchmark_id="bbq", eval_type="binary",
        description="BBQ social-bias benchmark for QA (0/1 correct)",
    ),
    "bold": OpenEvalBenchmarkSpec(
        benchmark_id="bold", eval_type="binary",
        description="BOLD bias-sensitive generation benchmark",
    ),
    "do-not-answer": OpenEvalBenchmarkSpec(
        benchmark_id="do-not-answer", eval_type="continuous",
        description="Do-Not-Answer safety refusal benchmark (scores 0-6, rescaled to [0,1])",
        score_bounds=(0.0, 6.0),
    ),
    "hi-tom": OpenEvalBenchmarkSpec(
        benchmark_id="hi-tom", eval_type="binary",
        description="Hi-ToM higher-order theory-of-mind reasoning benchmark",
    ),
    "ifeval": OpenEvalBenchmarkSpec(
        benchmark_id="ifeval", eval_type="continuous",
        description="Instruction-following evaluation benchmark",
    ),
    "omni-math": OpenEvalBenchmarkSpec(
        benchmark_id="omni-math", eval_type="binary",
        description="Omni-Math mathematical reasoning benchmark",
    ),
    "salad-bench": OpenEvalBenchmarkSpec(
        benchmark_id="salad-bench", eval_type="binary",
        description="SALAD-Bench safety/alignment benchmark",
    ),
    "cnndm": OpenEvalBenchmarkSpec(
        benchmark_id="cnndm", eval_type="continuous",
        description="CNN/DailyMail summarization ROUGE-L in [0,1]",
        metric_name="rouge_l",
    ),
    "xsum": OpenEvalBenchmarkSpec(
        benchmark_id="xsum", eval_type="continuous",
        description="XSUM abstractive summarization ROUGE-L in [0,1]",
        metric_name="rouge_l",
    ),
    "emobench": OpenEvalBenchmarkSpec(
        benchmark_id="emobench", eval_type="binary",
        description="EmoBench emotional intelligence benchmark",
    ),
}

OPENEVAL_DEFAULT_BENCHMARKS: list[str] = list(dict.fromkeys(b for _, b in OPENEVAL_DEFAULT_PAIRS))


# ─────────────────────────────────────────────────────────────────────────────
# Shared data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Corpus:
    """Full benchmark corpus for one (model, benchmark) pair -- the 'population'."""
    model: str
    benchmark_id: str
    eval_type: str
    source: str  # "dove" | "openeval"
    scores: np.ndarray  # shape (N,); all deduplicated, finite instance scores
    corpus_mean: float  # ground truth estimand = mean of all scores
    corpus_size: int  # N


def corpus_to_ci_source(corpus: Corpus) -> CISource:
    """Wrap a real-data Corpus as a CISource (WOR subsampling, bounded by corpus size)."""
    scores = corpus.scores
    n_total = corpus.corpus_size

    def _generate(rng: np.random.Generator, n: int, _scores: np.ndarray = scores, _n_total: int = n_total) -> np.ndarray:
        idxs = rng.choice(_n_total, size=n, replace=False)
        return _scores[idxs]

    return CISource(
        label=f"{corpus.model}/{corpus.benchmark_id}",
        eval_type=corpus.eval_type,
        true_mean=corpus.corpus_mean,
        generate=_generate,
        source=corpus.source,
        max_n=corpus.corpus_size,
        model=corpus.model,
        benchmark_id=corpus.benchmark_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DOVE_Lite -- loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def _dove_extract_scores(dataset: Any) -> np.ndarray:
    """Extract per-row scores from a loaded DOVE HF Dataset.

    DOVE stores scores in a nested 'evaluation' struct with a 'score' field.
    Falls back to a flat 'score' column if the struct is absent.
    """
    col_names = dataset.column_names

    if "evaluation" in col_names:
        eval_col = dataset["evaluation"]
        if isinstance(eval_col, list) and len(eval_col) > 0:
            sample = eval_col[0]
            if isinstance(sample, dict) and "score" in sample:
                return np.array([e["score"] for e in eval_col], dtype=float)
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
    """Return per-row item indices for deduplication, or None if unavailable."""
    col_names = dataset.column_names

    if "sample_index" in col_names:
        try:
            return np.array(dataset["sample_index"], dtype=int)
        except Exception:
            pass

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


def _dove_deduplicate(scores: np.ndarray, indices: np.ndarray | None) -> np.ndarray:
    """Keep one score per unique item index (first occurrence).

    DOVE contains multiple prompt perturbations per item. Keeping the first
    occurrence per unique sample_index (or hf_index) ensures each element of
    the returned array corresponds to an independent benchmark item,
    satisfying the IID assumption required by the bootstrap CI methods.
    """
    if indices is None:
        return scores
    seen: dict[int, float] = {}
    for idx, score in zip(indices.tolist(), scores.tolist()):
        if idx not in seen:
            seen[idx] = float(score)
    return np.array(list(seen.values()), dtype=float)


def _score_dist_summary(scores: np.ndarray, eval_type: str) -> str:
    """Score distribution diagnostic for corpus load-time output."""
    n = len(scores)
    if n == 0:
        return "[empty]"
    pct_zero = 100.0 * np.mean(np.isclose(scores, 0.0))
    pct_one = 100.0 * np.mean(np.isclose(scores, 1.0))
    looks_binary = (pct_zero + pct_one) >= 99.0
    parts = [
        f"min={scores.min():.4f}", f"max={scores.max():.4f}", f"std={scores.std():.4f}",
        f"zeros={pct_zero:.1f}%", f"ones={pct_one:.1f}%",
    ]
    flag = ""
    if eval_type == "binary" and not looks_binary:
        flag = "  *** NOT cleanly binary -- binary CI methods may be unreliable ***"
    elif eval_type != "binary" and looks_binary:
        flag = f"  *** looks binary but eval_type={eval_type!r} -- consider changing to binary ***"
    line1 = "[" + "  ".join(parts) + "]" + flag
    if not flag:
        return line1
    sorted_scores = np.sort(scores)
    idxs = np.linspace(0, n - 1, min(10, n), dtype=int)
    sample_str = "  ".join(f"{sorted_scores[i]:.4f}" for i in idxs)
    return line1 + f"\n        sample values: [{sample_str}]"


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
        raise ImportError("Install the HuggingFace datasets library: pip install datasets")

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
            print(f"  Skip  {model}/{spec.benchmark_id}: score extraction failed -- {exc}")
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

    if spec.score_bounds is not None:
        lo, hi = spec.score_bounds
        scores = (scores - lo) / (hi - lo)

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

    print(f"\nLoading DOVE_Lite corpora ({len(pairs)} model x benchmark pairs) ...")

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
# OpenEval -- loading helpers
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
    obj = _oe_parse(model_val)
    if isinstance(obj, dict):
        return obj.get("name")
    return None


def _oe_parse_response_id(response_id: str) -> tuple[str, str] | tuple[None, None]:
    """Extract (source_benchmark, item_id) from an OpenEval response_id."""
    m = _OE_RESP_ID_RE.match(response_id)
    if m is None:
        return None, None
    source = m.group(1)
    item_id = f"{source}_{m.group(2)}_{m.group(3)}"
    return source, item_id


def _oe_extract_score(scores_val: Any, metric_name: str | None) -> float | None:
    """Extract a numeric score from OpenEval's scores field (3 observed layouts)."""
    data = _oe_parse(scores_val)

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
            entry = next((e for e in data if _list_entry_metric_name(e) == metric_name), None)
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


def _oe_first_metric_name(scores_val: Any) -> str | None:
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
    """Return {model_name: {benchmark_source: response_count}}."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    filter_set = set(benchmark_filter) if benchmark_filter else None
    msg = (
        f"Scanning OpenEval responses for benchmarks {sorted(filter_set)} ..."
        if filter_set else
        "Scanning OpenEval response table for model names and benchmark coverage ..."
    )
    print(msg)
    ds = load_dataset(openeval_repo, "response", split="train", token=hf_token, cache_dir=cache_dir)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in ds:
        name = _oe_get_model_name(row.get("model"))
        source, _ = _oe_parse_response_id(row.get("response_id", ""))
        if name and source:
            if filter_set is None or source in filter_set:
                counts[name][source] += 1
    return {model: dict(bench_counts) for model, bench_counts in counts.items() if bench_counts}


def list_openeval_benchmarks(
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, int]:
    """Return {benchmark_source: response_count} for all benchmarks in OpenEval."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print("Scanning OpenEval response table for benchmark IDs ...")
    ds = load_dataset(openeval_repo, "response", split="train", token=hf_token, cache_dir=cache_dir)
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
    """Load OpenEval corpora for the given (model, benchmark_id) pairs.

    No item table needed -- response_id encodes both the benchmark source and
    item_id. Pairs must be confirmed to have data; run --list-models
    --benchmarks <id> to discover valid combinations.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

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

    pairs_set: set[tuple[str, str]] = set(pairs)
    benchmark_ids_set = {b for _, b in pairs}

    print("Loading OpenEval response table (~6.4 GB; cached after first download) ...")
    response_ds = load_dataset(openeval_repo, "response", split="train", token=hf_token, cache_dir=cache_dir)

    print(f"  Filtering to {len(pairs)} (model, benchmark) pairs ...")

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
            "    - Run --list-benchmarks to verify benchmark source IDs.\n"
            "    - Run --list-models --benchmarks <ids> to verify model names."
        )
        return []

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
        flag = "  <- no data!" if n == 0 else ""
        print(f"    {model_name}/{bench}: {n:,}{flag}")
        if n == 0:
            any_zero = True
    if any_zero:
        print("  Tip: run --list-models --benchmarks <ids> to find models with data.")
    print()

    score_accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen_items: dict[tuple[str, str], set[str]] = defaultdict(set)
    metric_seen: dict[str, str] = {}
    n_dedup = 0
    n_no_score = 0
    _score_fail_samples: list[Any] = []

    for row in response_ds:
        response_id = row.get("response_id", "")
        source, item_id = _oe_parse_response_id(response_id)
        if source is None:
            continue

        model_name = _oe_get_model_name(row.get("model"))
        if model_name is None or (model_name, source) not in pairs_set:
            continue

        key = (model_name, source)
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
        print(f"  {n_dedup:,} duplicate (item x model) rows removed.")
    if n_no_score > 0:
        print(f"  {n_no_score:,} rows skipped (score missing or non-finite).")
        if _score_fail_samples:
            print("  Sample 'scores' field values that failed extraction:")
            for i, raw in enumerate(_score_fail_samples):
                r = repr(raw)
                if len(r) > 300:
                    r = r[:300] + " ..."
                print(f"    [{i}] type={type(raw).__name__}  value={r}")

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
        if spec.score_bounds is not None:
            lo, hi = spec.score_bounds
            arr = (arr - lo) / (hi - lo)
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


def build_real_data_sources(
    source: str,
    *,
    benchmarks: list[str] | None = None,
    models: list[str] | None = None,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_corpus_size: int = 50,
) -> list[CISource]:
    """Resolve (model, benchmark) pairs for `source` and return them as CISources.

    `source` is one of "dove", "openeval", or "all". When `benchmarks` is
    given, only the default pairs whose benchmark_id is in that list are
    used; `models` similarly filters by model name. With no filters, each
    source's curated default pairs (DOVE_DEFAULT_PAIRS / OPENEVAL_DEFAULT_PAIRS)
    are used.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown real-data source: {source!r}. Choices: {SOURCES}")

    def _filter_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        out = pairs
        if benchmarks:
            out = [(m, b) for m, b in out if b in benchmarks]
        if models:
            out = [(m, b) for m, b in out if m in models]
        return out

    corpora: list[Corpus] = []
    if source in ("dove", "all"):
        corpora += build_dove_corpora(
            _filter_pairs(DOVE_DEFAULT_PAIRS),
            hf_token=hf_token, cache_dir=cache_dir, min_corpus_size=min_corpus_size,
        )
    if source in ("openeval", "all"):
        corpora += build_openeval_corpora(
            _filter_pairs(OPENEVAL_DEFAULT_PAIRS),
            hf_token=hf_token, cache_dir=cache_dir, min_corpus_size=min_corpus_size,
        )

    return [corpus_to_ci_source(c) for c in corpora]


# ─────────────────────────────────────────────────────────────────────────────
# Paired (shared-item) real data -- ported from simulations/sim_tango_real.py
#
# Restricted to known-binary benchmarks (same restriction as the legacy
# script: Newcombe/Tango/Bayes-paired CI formulas are binary-only) and to
# R=1 (single run per item) -- multi-run real pairs (Tango multirun variants,
# nested-diff bootstrap, lmm_diff) are a documented exception, deferred to a
# future ci_nested real-data extension. See harness README known exceptions.
# ─────────────────────────────────────────────────────────────────────────────

DOVE_PAIR_BINARY_BENCHMARKS = ["hellaswag", "arc_challenge", "gsm8k", "squad", "quality"]

DOVE_PAIR_FILE: dict[str, str] = {
    "hellaswag": "hellaswag.parquet",
    "arc_challenge": "ai2_arc.arc_challenge.parquet",
    "gsm8k": "gsm8k.parquet",
    "squad": "squad.parquet",
    "quality": "quality.parquet",
}

# Default (model, benchmark) pairs to load. Multiple models per benchmark ->
# pairs are formed from all combinations.
DOVE_PAIR_DEFAULT_MODEL_BENCH: list[tuple[str, str]] = [
    ("Llama-3.2-1B-Instruct", "hellaswag"),
    ("OLMoE-1B-7B-0924-Instruct", "hellaswag"),
    ("Mistral-7B-Instruct-v0.3", "hellaswag"),
    ("Meta-Llama-3-8B-Instruct", "hellaswag"),
    ("Llama-3.2-1B-Instruct", "arc_challenge"),
    ("OLMoE-1B-7B-0924-Instruct", "arc_challenge"),
    ("Mistral-7B-Instruct-v0.3", "arc_challenge"),
    ("Meta-Llama-3-8B-Instruct", "arc_challenge"),
]

OPENEVAL_PAIR_BINARY_SPECS: dict[str, dict] = {
    "mmlu-pro": {"metric_name": None, "score_scale": 1.0},
    "gpqa": {"metric_name": None, "score_scale": 1.0},
    "boolq": {"metric_name": None, "score_scale": 1.0},
    "imdb": {"metric_name": None, "score_scale": 1.0},
    "culturalbench": {"metric_name": None, "score_scale": 1.0},
    "opentom": {"metric_name": None, "score_scale": 1.0},
    "bbq": {"metric_name": None, "score_scale": 1.0},
    "hi-tom": {"metric_name": None, "score_scale": 1.0},
    "omni-math": {"metric_name": None, "score_scale": 1.0},
    "emobench": {"metric_name": None, "score_scale": 1.0},
    "salad-bench": {"metric_name": None, "score_scale": 1.0},
}

# Default (model, benchmark) pairs confirmed to have data in OpenEval.
OPENEVAL_PAIR_DEFAULT_MODEL_BENCH: list[tuple[str, str]] = [
    ("falcon-40b", "bbq"),
    ("llama-7b", "bbq"),
    ("qwen-2.5-32b-instruct", "bbq"),
    ("gemma-2-27b-it", "mmlu-pro"),
    ("gemma-3-4b-it", "mmlu-pro"),
    ("qwen-3-30b-instruct", "mmlu-pro"),
    ("llama-2-70b-hf", "mmlu-pro"),
    ("DeepSeek-R1", "culturalbench"),
    ("gpt-4o", "culturalbench"),
    ("gpt-4.1-mini", "culturalbench"),
    ("o4-mini", "culturalbench"),
    ("grok-4", "culturalbench"),
    ("gpt-4o-mini", "hi-tom"),
    ("grok-4", "hi-tom"),
    ("phi-4", "hi-tom"),
    ("DeepSeek-R1", "hi-tom"),
    ("DeepSeek-R1", "opentom"),
    ("gpt-4.1", "opentom"),
    ("o1", "opentom"),
    ("DeepSeek-R1", "salad-bench"),
    ("grok-4", "salad-bench"),
    ("kimi-k2", "salad-bench"),
    ("phi-4", "salad-bench"),
    ("gemma-3-27b-it", "omni-math"),
    ("qwen-2.5-14b-instruct", "omni-math"),
    ("qwen-2.5-72b-instruct", "omni-math"),
    ("llama-2-13b-hf", "omni-math"),
]


@dataclass
class CorpusPair:
    """Aligned paired scores for two models on the same benchmark (R=1 only)."""
    model_a: str
    model_b: str
    benchmark_id: str
    source: str  # "dove" | "openeval" | "inspect"
    scores_a: np.ndarray  # shape (N,)
    scores_b: np.ndarray  # shape (N,)
    true_diff: float  # mean(scores_a - scores_b) -- population ground truth
    corpus_size: int  # N = number of shared items


def corpus_pair_to_ci_pair_source(cp: CorpusPair) -> CIPairSource:
    """Wrap a real-data CorpusPair as a CIPairSource (WOR subsampling, R=1 only)."""
    scores_a, scores_b, n_total = cp.scores_a, cp.scores_b, cp.corpus_size

    def _generate_pair(
        rng: np.random.Generator, n: int, runs: int,
        _a: np.ndarray = scores_a, _b: np.ndarray = scores_b, _n_total: int = n_total,
    ) -> tuple[np.ndarray, np.ndarray]:
        if runs != 1:
            raise ValueError("Real-data pair sources only support runs=1 in this pass.")
        idxs = rng.choice(_n_total, size=n, replace=False)
        return _a[idxs].reshape(n, 1), _b[idxs].reshape(n, 1)

    return CIPairSource(
        label=f"{cp.model_a} vs {cp.model_b}/{cp.benchmark_id}",
        eval_type="binary",
        true_diff=cp.true_diff,
        generate_pair=_generate_pair,
        source=cp.source,
        max_n=cp.corpus_size,
        model_a=cp.model_a,
        model_b=cp.model_b,
        benchmark_id=cp.benchmark_id,
    )


def _dove_load_item_scores(
    model: str,
    benchmark_id: str,
    *,
    dove_repo: str = DOVE_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
) -> dict[int, float] | None:
    """Load DOVE_Lite for one (model, benchmark) and return {sample_index: score}."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    fname = DOVE_PAIR_FILE.get(benchmark_id)
    if fname is None:
        print(f"  Skip  {model}/{benchmark_id}: not a supported binary benchmark")
        return None

    path_candidates: list[str] = [f"{model}/en/{shots}_shot/{fname}" for shots in (0, 5, 2, 3)]
    path_candidates.append(f"{model}/en/5_shots/{fname}")

    dataset = None
    for fp in path_candidates:
        try:
            dataset = load_dataset(dove_repo, data_files=fp, split="train", token=hf_token, cache_dir=cache_dir)
            break
        except Exception:
            continue

    if dataset is None:
        print(f"  Skip  {model}/{benchmark_id}: no DOVE file found")
        return None

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
                indices = [int(inst["sample_identifier"]["hf_index"]) for inst in inst_col]
        except Exception:
            pass

    if indices is None:
        return {i: s for i, s in enumerate(raw_scores) if np.isfinite(s)}

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
    """Build CorpusPairs from DOVE_Lite by loading and aligning on sample_index."""
    bench_models: dict[str, list[str]] = defaultdict(list)
    for model, bench in model_bench_pairs:
        if bench in DOVE_PAIR_BINARY_BENCHMARKS:
            bench_models[bench].append(model)
        else:
            print(f"  Warning: {bench} is not a supported binary DOVE benchmark. Skipping.")

    corpus_pairs: list[CorpusPair] = []
    for bench, models in bench_models.items():
        print(f"\nLoading DOVE_Lite: benchmark={bench}, models={models} ...")
        item_maps: dict[str, dict[int, float]] = {}
        for model in models:
            m = _dove_load_item_scores(model, bench, dove_repo=dove_repo, hf_token=hf_token, cache_dir=cache_dir)
            if m is not None:
                print(f"  OK    {model}/{bench}: {len(m)} items")
                item_maps[model] = m

        if len(item_maps) < 2:
            print(f"  Skip  {bench}: fewer than 2 models loaded successfully.")
            continue

        loaded_models = list(item_maps.keys())
        for model_a, model_b in combinations(loaded_models, 2):
            shared_keys = sorted(item_maps[model_a].keys() & item_maps[model_b].keys())
            if len(shared_keys) < min_pair_size:
                print(f"  Skip  ({model_a}, {model_b}) on {bench}: only {len(shared_keys)} shared items < {min_pair_size}")
                continue
            scores_a = np.array([item_maps[model_a][k] for k in shared_keys])
            scores_b = np.array([item_maps[model_b][k] for k in shared_keys])
            true_diff = float(np.mean(scores_a - scores_b))
            print(
                f"  Pair  ({model_a} vs {model_b}) on {bench}: N={len(shared_keys)}, "
                f"mean_A={np.mean(scores_a):.4f}, mean_B={np.mean(scores_b):.4f}, true_diff={true_diff:+.4f}"
            )
            corpus_pairs.append(CorpusPair(
                model_a=model_a, model_b=model_b, benchmark_id=bench, source="dove",
                scores_a=scores_a, scores_b=scores_b, true_diff=true_diff, corpus_size=len(shared_keys),
            ))

    print(f"\n  {len(corpus_pairs)} corpus pairs built from DOVE_Lite.\n")
    return corpus_pairs


def build_openeval_corpus_pairs(
    model_bench_pairs: list[tuple[str, str]],
    *,
    openeval_repo: str = OPENEVAL_REPO,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_pair_size: int = 50,
) -> list[CorpusPair]:
    """Build CorpusPairs from OpenEval by aligning on item_id (binary benchmarks only)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    unknown_benches = [b for _, b in model_bench_pairs if b not in OPENEVAL_PAIR_BINARY_SPECS]
    if unknown_benches:
        print(
            f"Warning: unsupported OpenEval binary benchmark IDs: {sorted(set(unknown_benches))}.\n"
            f"  Supported: {list(OPENEVAL_PAIR_BINARY_SPECS)}"
        )
        model_bench_pairs = [(m, b) for m, b in model_bench_pairs if b in OPENEVAL_PAIR_BINARY_SPECS]
    if not model_bench_pairs:
        return []

    pairs_set = set(model_bench_pairs)
    bench_set = {b for _, b in model_bench_pairs}

    print("Loading OpenEval response table (~1.4 GB; cached after first download) ...")
    response_ds = load_dataset(openeval_repo, "response", split="train", token=hf_token, cache_dir=cache_dir)

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
        spec = OPENEVAL_PAIR_BINARY_SPECS[source]
        score = _oe_extract_score(row.get("scores"), spec["metric_name"])
        if score is not None and np.isfinite(score):
            item_maps[key][item_id] = float(score) * spec["score_scale"]

    if n_dedup > 0:
        print(f"  {n_dedup:,} duplicate rows removed (kept first per item x model).")

    for (model, bench), scores_map in list(item_maps.items()):
        keys = list(scores_map.keys())
        vals = np.array([scores_map[k] for k in keys], dtype=float)
        non_binary_mask = ~np.isin(vals, [0.0, 1.0])
        if np.any(non_binary_mask):
            rounded_vals = np.clip(np.rint(vals), 0.0, 1.0)
            unique_bad = np.unique(vals[non_binary_mask])[:5]
            print(f"  Warning: {model}/{bench} has {int(np.sum(non_binary_mask)):,} non-binary scores (e.g. {unique_bad}). Rounded to {{0,1}}.")
            item_maps[(model, bench)] = {k: float(v) for k, v in zip(keys, rounded_vals)}

    corpus_pairs: list[CorpusPair] = []
    for bench in sorted(bench_set):
        requested = list(dict.fromkeys(m for m, b in model_bench_pairs if b == bench))
        bench_models = [m for m in requested if (m, bench) in item_maps and item_maps[(m, bench)]]
        if len(bench_models) < 2:
            print(f"  Skip  {bench}: fewer than 2 models with data.")
            continue
        print(f"\n  Benchmark: {bench}")
        for model in bench_models:
            print(f"    {model}: {len(item_maps[(model, bench)]):,} items")
        for model_a, model_b in combinations(bench_models, 2):
            map_a = item_maps[(model_a, bench)]
            map_b = item_maps[(model_b, bench)]
            shared_ids = sorted(map_a.keys() & map_b.keys())
            if len(shared_ids) < min_pair_size:
                print(f"  Skip  ({model_a}, {model_b}) on {bench}: {len(shared_ids)} shared items < {min_pair_size}")
                continue
            scores_a = np.array([map_a[k] for k in shared_ids])
            scores_b = np.array([map_b[k] for k in shared_ids])
            bad_a = scores_a[~np.isin(scores_a, [0.0, 1.0])]
            bad_b = scores_b[~np.isin(scores_b, [0.0, 1.0])]
            if len(bad_a) > 0 or len(bad_b) > 0:
                print(f"  Skip  ({model_a} vs {model_b}) on {bench}: non-binary scores after alignment. Skipping pair.")
                continue
            true_diff = float(np.mean(scores_a - scores_b))
            print(
                f"  Pair  ({model_a} vs {model_b}): N={len(shared_ids)}, mean_A={np.mean(scores_a):.4f}, "
                f"mean_B={np.mean(scores_b):.4f}, true_diff={true_diff:+.4f}"
            )
            corpus_pairs.append(CorpusPair(
                model_a=model_a, model_b=model_b, benchmark_id=bench, source="openeval",
                scores_a=scores_a, scores_b=scores_b, true_diff=true_diff, corpus_size=len(shared_ids),
            ))

    print(f"\n  {len(corpus_pairs)} corpus pairs built from OpenEval.\n")
    return corpus_pairs


def build_inspect_corpus_pairs(
    csv_path: str,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    *,
    min_pair_size: int = 50,
) -> list[CorpusPair]:
    """Build CorpusPairs (R=1 only) from a CSV produced by collect_inspect_benchmarks.py.

    CSV columns: benchmark, model, item_id, run_idx, score. Any run_idx != 0
    is discarded (this pass is flat/R=1 only -- see module docstring).
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Inspect data file not found: {csv_path}\n  Run collect_inspect_benchmarks.py first to generate it."
        )

    item_maps: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    print(f"Loading Inspect AI data from: {csv_path}")
    n_rows = 0
    with p.open(newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            bench = row.get("benchmark", "").strip()
            model = row.get("model", "").strip()
            item_id = row.get("item_id", "").strip()
            try:
                run_idx = int(row.get("run_idx", 0))
                score = float(row.get("score", float("nan")))
            except (ValueError, TypeError):
                continue
            if run_idx != 0 or not bench or not model or not item_id or not np.isfinite(score):
                continue
            if benchmarks is not None and bench not in benchmarks:
                continue
            if models is not None and model not in models:
                continue
            item_maps[(model, bench)][item_id] = score
            n_rows += 1

    print(f"  {n_rows:,} rows loaded (run_idx=0 only).")
    if not item_maps:
        print("  No data found -- check --benchmarks / --models filters match the CSV.")
        return []

    all_benches = sorted({b for _, b in item_maps.keys()})
    corpus_pairs: list[CorpusPair] = []

    for bench in all_benches:
        bench_models = sorted(m for m, b in item_maps.keys() if b == bench)
        if len(bench_models) < 2:
            print(f"  Skip  {bench}: only {len(bench_models)} model(s) -- need >= 2 to form a pair")
            continue
        print(f"\n  Benchmark: {bench}")
        for model in bench_models:
            print(f"    {model}: {len(item_maps[(model, bench)]):,} items")

        for model_a, model_b in combinations(bench_models, 2):
            map_a = item_maps[(model_a, bench)]
            map_b = item_maps[(model_b, bench)]
            shared_ids = sorted(map_a.keys() & map_b.keys())
            if len(shared_ids) < min_pair_size:
                print(f"  Skip  ({model_a} vs {model_b}) on {bench}: {len(shared_ids)} shared items < {min_pair_size}")
                continue
            scores_a = np.array([map_a[k] for k in shared_ids])
            scores_b = np.array([map_b[k] for k in shared_ids])
            true_diff = float(np.mean(scores_a - scores_b))
            short_a = model_a.split("/")[-1] if "/" in model_a else model_a
            short_b = model_b.split("/")[-1] if "/" in model_b else model_b
            print(
                f"  Pair  ({short_a} vs {short_b}): N={len(shared_ids)}, mean_A={np.mean(scores_a):.4f}, "
                f"mean_B={np.mean(scores_b):.4f}, true_diff={true_diff:+.4f}"
            )
            corpus_pairs.append(CorpusPair(
                model_a=model_a, model_b=model_b, benchmark_id=bench, source="inspect",
                scores_a=scores_a, scores_b=scores_b, true_diff=true_diff, corpus_size=len(shared_ids),
            ))

    print(f"\n  {len(corpus_pairs)} corpus pairs built from Inspect AI data.\n")
    return corpus_pairs


def build_real_pair_sources(
    source: str,
    *,
    benchmarks: list[str] | None = None,
    models: list[str] | None = None,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    min_pair_size: int = 50,
    inspect_csv: str | None = None,
) -> list[CIPairSource]:
    """Resolve (model, benchmark) pairs for `source` and return them as CIPairSources.

    `source` is one of "dove", "openeval", "inspect", or "all" ("all" skips
    "inspect" since it requires an explicit --inspect-csv path). R=1 only --
    see module docstring.
    """
    if source not in PAIR_SOURCES:
        raise ValueError(f"Unknown real-data pair source: {source!r}. Choices: {PAIR_SOURCES}")

    def _filter_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        out = pairs
        if benchmarks:
            out = [(m, b) for m, b in out if b in benchmarks]
        if models:
            out = [(m, b) for m, b in out if m in models]
        return out

    corpus_pairs: list[CorpusPair] = []
    if source in ("dove", "all"):
        corpus_pairs += build_dove_corpus_pairs(
            _filter_pairs(DOVE_PAIR_DEFAULT_MODEL_BENCH),
            hf_token=hf_token, cache_dir=cache_dir, min_pair_size=min_pair_size,
        )
    if source in ("openeval", "all"):
        corpus_pairs += build_openeval_corpus_pairs(
            _filter_pairs(OPENEVAL_PAIR_DEFAULT_MODEL_BENCH),
            hf_token=hf_token, cache_dir=cache_dir, min_pair_size=min_pair_size,
        )
    if source == "inspect":
        if not inspect_csv:
            raise ValueError("--inspect-csv is required when --data-source inspect.")
        corpus_pairs += build_inspect_corpus_pairs(
            inspect_csv, models=models, benchmarks=benchmarks, min_pair_size=min_pair_size,
        )

    return [corpus_pair_to_ci_pair_source(cp) for cp in corpus_pairs]
