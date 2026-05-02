#!/usr/bin/env python3
"""
collect_inspect_benchmarks.py — Run Inspect AI knowledge benchmarks via
OpenRouter and save per-sample binary scores to a CSV file.

The CSV is the input for sim_tango_real.py --source inspect, which samples
from it to test paired CI methods under real LLM accuracy distributions.

Install dependencies first:
  pip install inspect-ai inspect-evals

Usage:
  python simulations/collect_inspect_benchmarks.py \\
      --models openrouter/meta-llama/llama-3.1-8b-instruct \\
               openrouter/mistralai/mistral-7b-instruct \\
      --benchmarks mmlu arc_challenge hellaswag \\
      --limit 1000 \\
      --output simulations/out/inspect_benchmarks.csv

  # Multi-run (R=3 runs per item per model, enables nested/multirun sim methods)
  python simulations/collect_inspect_benchmarks.py \\
      --models openrouter/meta-llama/llama-3.1-8b-instruct \\
               openrouter/mistralai/mistral-7b-instruct \\
      --runs 3 --limit 500

  # Resume after interruption (skips already-collected model/benchmark/run combos)
  python simulations/collect_inspect_benchmarks.py --resume

Environment variables:
  OPENROUTER_API_KEY — required for OpenRouter models
  OPENROUTER_BASE_URL — optional, defaults to https://openrouter.ai/api/v1

Output CSV columns:
  benchmark   short benchmark name (e.g. "mmlu")
  model       full model ID passed to Inspect AI (e.g. "openrouter/meta-llama/...")
  item_id     stable sample ID from the dataset (str)
  run_idx     0-indexed run number (0 for single-run)
  score       binary 0.0 or 1.0 (correct / incorrect)
    est_input_tokens   input tokens for this sample (actual usage when available; estimate fallback)
    est_output_tokens  output tokens for this sample (actual usage when available; estimate fallback)
    est_cost_usd       USD cost for this sample (actual usage when available; estimate fallback)
    cost_source        "actual" when provider usage is found, otherwise "estimate"
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

_BENCHMARK_FNS: dict[str, Any] = {}          # name → callable() → Task
_BENCHMARK_LOAD_ERRORS: dict[str, str] = {}  # name → error message


def _try_register(name: str, module: str, attr: str, aliases: list[str] | None = None) -> None:
    try:
        mod = __import__(module, fromlist=[attr])
        fn = getattr(mod, attr)
        if not callable(fn):
            raise TypeError(f"{module}.{attr} is not callable (got {type(fn).__name__})")
        _BENCHMARK_FNS[name] = fn
        for alias in (aliases or []):
            _BENCHMARK_FNS[alias] = fn
    except Exception as exc:
        msg = str(exc)
        _BENCHMARK_LOAD_ERRORS[name] = msg
        for alias in (aliases or []):
            _BENCHMARK_LOAD_ERRORS[alias] = msg


def _load_benchmark_registry() -> None:
    """Lazily populate the registry; import errors are silently recorded."""
    if _BENCHMARK_FNS or _BENCHMARK_LOAD_ERRORS:
        return
    _try_register("mmlu",         "inspect_evals.mmlu",       "mmlu_0_shot", aliases=["mmlu_0_shot"])
    # _try_register("mmlu_5_shot",  "inspect_evals.mmlu",       "mmlu_5_shot")
    _try_register("arc",          "inspect_evals.arc",        "arc_challenge", aliases=["arc_challenge"])
    # _try_register("arc_easy",     "inspect_evals.arc",        "arc_easy")
    _try_register("hellaswag",    "inspect_evals.hellaswag",  "hellaswag")
    _try_register("gsm8k",        "inspect_evals.gsm8k",      "gsm8k")
    _try_register("winogrande",   "inspect_evals.winogrande", "winogrande")
    _try_register("truthfulqa",   "inspect_evals.truthfulqa", "truthfulqa")
    _try_register("boolq",        "inspect_evals.boolq",      "boolq")
    _try_register("piqa",         "inspect_evals.piqa",       "piqa")
    _try_register("bbq",          "inspect_evals.bbq",        "bbq")


DEFAULT_BENCHMARKS = ["mmlu", "arc", "hellaswag", "gsm8k", "winogrande", "truthfulqa", "boolq", "piqa", "bbq"]

DEFAULT_LIMIT   = 1000
DEFAULT_RUNS    = 5
DEFAULT_OUTPUT  = "simulations/out/inspect_benchmarks.csv"
DEFAULT_LOG_DIR = "simulations/out/inspect_logs"
DEFAULT_EVAL_TIMEOUT_SECONDS = 120

# ---------------------------------------------------------------------------
# Cost estimation tables
# ---------------------------------------------------------------------------

# (input_$/1M_tokens, output_$/1M_tokens) — prices as of 2026-05
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "openrouter/meta-llama/llama-3.1-8b-instruct": (0.02,   0.05),
    "openrouter/mistralai/ministral-8b-2512":      (0.15,   0.15),
    "openrouter/google/gemma-3n-e4b-it":           (0.06,   0.12),
    "openrouter/ibm-granite/granite-4.1-8b":       (0.05,   0.10),
    "openrouter/qwen/qwen3.5-35b-a3b":             (0.1625, 1.30),
    "openrouter/openai/gpt-4o-mini":               (0.15,   0.60),
}
DEFAULT_MODELS = list(MODEL_COSTS.keys())

# Approximate avg tokens per sample for each benchmark.
# Input = system prompt + question + answer choices; output = answer letter (or short reasoning).
_BENCH_IN: dict[str, int] = {
    "mmlu":       600,
    # "mmlu_pro":   900,
    "arc":        300,
    "hellaswag":  200,
    "gsm8k":      400,
    "winogrande": 120,
    "truthfulqa": 400,
    "boolq":      500,
    "piqa":       200,
    "bbq":        500,
}
_BENCH_OUT: dict[str, int] = {
    "mmlu":       10,
    # "mmlu_pro":   10,
    "arc":        10,
    "hellaswag":  10,
    "gsm8k":      150,   # chain-of-thought solutions
    "winogrande": 10,
    "truthfulqa": 50,
    "boolq":      5,
    "piqa":       10,
    "bbq":        10,
}
_DEFAULT_IN  = 500
_DEFAULT_OUT = 20


# ---------------------------------------------------------------------------
# Score conversion
# ---------------------------------------------------------------------------


def _to_binary(value: Any) -> float | None:
    """Convert an Inspect AI Score.value to 0.0 / 1.0, or None if unparseable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        if v == 1.0:
            return 1.0
        if v == 0.0:
            return 0.0
        if 0.0 < v < 1.0:
            return round(v)   # threshold partial-credit to nearest binary
        return None
    if isinstance(value, str):
        up = value.strip().upper()
        if up in ("C", "CORRECT", "TRUE", "1", "YES", "PASS", "RIGHT"):
            return 1.0
        if up in ("I", "INCORRECT", "FALSE", "0", "NO", "FAIL", "WRONG"):
            return 0.0
    return None


def _extract_binary_score(scores: dict | None) -> float | None:
    """Return the first binary score found in a sample's scores dict."""
    if not scores:
        return None
    for _, score in scores.items():
        binary = _to_binary(score.value)
        if binary is not None:
            return binary
    return None


# ---------------------------------------------------------------------------
# Token/cost estimation helpers
# ---------------------------------------------------------------------------


def _get_model_prices(model_id: str) -> tuple[float, float]:
    """Return (input_price, output_price) in $/1M tokens for a model."""
    return MODEL_COSTS.get(model_id, (0.15, 0.15))


def _estimate_sample_tokens(benchmark: str) -> tuple[int, int]:
    """Return estimated (input_tokens, output_tokens) for one sample."""
    return _BENCH_IN.get(benchmark, _DEFAULT_IN), _BENCH_OUT.get(benchmark, _DEFAULT_OUT)


def _estimate_sample_cost_usd(benchmark: str, model_id: str) -> float:
    """Return estimated cost in USD for one sample."""
    avg_in, avg_out = _estimate_sample_tokens(benchmark)
    in_price, out_price = _get_model_prices(model_id)
    return (avg_in * in_price + avg_out * out_price) / 1_000_000


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _obj_dict(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        return obj
    for method in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        try:
            data = vars(obj)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None


def _iter_usage_candidates(obj: Any, max_depth: int = 5) -> list[dict[str, Any]]:
    """Collect possible usage dicts from nested sample/log objects."""
    out: list[dict[str, Any]] = []
    stack: list[tuple[Any, int]] = [(obj, 0)]
    seen: set[int] = set()

    while stack:
        cur, depth = stack.pop()
        if cur is None or depth > max_depth:
            continue
        cur_id = id(cur)
        if cur_id in seen:
            continue
        seen.add(cur_id)

        if isinstance(cur, dict):
            usage = cur.get("usage")
            if isinstance(usage, dict):
                out.append(usage)
            if any(k in cur for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost", "input_tokens", "output_tokens")):
                out.append(cur)
            for v in cur.values():
                stack.append((v, depth + 1))
            continue

        if isinstance(cur, (list, tuple)):
            for v in cur:
                stack.append((v, depth + 1))
            continue

        as_dict = _obj_dict(cur)
        if as_dict is not None:
            stack.append((as_dict, depth + 1))

    return out


def _extract_from_model_usage(sample: Any) -> tuple[int | None, int | None, float | None] | None:
    """Extract usage from sample.model_usage when Inspect provides it."""
    usage = getattr(sample, "model_usage", None)
    if usage is None:
        return None

    usage_dict = _obj_dict(usage)

    if usage_dict is not None:
        inp = _to_int(usage_dict.get("input_tokens"))
        out = _to_int(usage_dict.get("output_tokens"))
        cost = _to_float(usage_dict.get("total_cost"))
    else:
        inp = _to_int(getattr(usage, "input_tokens", None))
        out = _to_int(getattr(usage, "output_tokens", None))
        cost = _to_float(getattr(usage, "total_cost", None))

    if inp is None and out is None and cost is None:
        return None
    return inp, out, cost


def _extract_usage_or_estimate(sample: Any, benchmark: str, model_id: str) -> tuple[int, int, float, str]:
    """Return (input_tokens, output_tokens, cost_usd, source)."""
    est_in, est_out = _estimate_sample_tokens(benchmark)
    est_cost = _estimate_sample_cost_usd(benchmark, model_id)

    # Prefer Inspect's typed usage object; this maps directly to provider billing fields.
    from_model_usage = _extract_from_model_usage(sample)
    if from_model_usage is not None:
        inp, out, cost = from_model_usage
        resolved_in = est_in if inp is None else inp
        resolved_out = est_out if out is None else out

        if cost is not None and cost >= 0:
            return resolved_in, resolved_out, cost, "actual"

        in_price, out_price = _get_model_prices(model_id)
        derived_cost = (resolved_in * in_price + resolved_out * out_price) / 1_000_000
        return resolved_in, resolved_out, derived_cost, "estimate"

    best_with_cost: tuple[int | None, int | None, float] | None = None
    best_tokens_only: tuple[int | None, int | None] | None = None

    for cand in _iter_usage_candidates(sample):
        inp = _to_int(cand.get("prompt_tokens"))
        if inp is None:
            inp = _to_int(cand.get("input_tokens"))

        out = _to_int(cand.get("completion_tokens"))
        if out is None:
            out = _to_int(cand.get("output_tokens"))

        cost = _to_float(cand.get("cost"))
        if cost is None:
            cost = _to_float(cand.get("total_cost"))

        if cost is not None and cost >= 0:
            best_with_cost = (inp, out, cost)
            break

        if inp is not None or out is not None:
            best_tokens_only = (inp, out)

    if best_with_cost is not None:
        inp = est_in if best_with_cost[0] is None else best_with_cost[0]
        out = est_out if best_with_cost[1] is None else best_with_cost[1]
        return inp, out, best_with_cost[2], "actual"

    if best_tokens_only is not None:
        inp = est_in if best_tokens_only[0] is None else best_tokens_only[0]
        out = est_out if best_tokens_only[1] is None else best_tokens_only[1]
        in_price, out_price = _get_model_prices(model_id)
        derived_cost = (inp * in_price + out * out_price) / 1_000_000
        return inp, out, derived_cost, "estimate"

    return est_in, est_out, est_cost, "estimate"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(
    models: list[str],
    benchmarks: list[str],
    limit: int,
    runs: int,
) -> None:
    """Print a per-model cost breakdown and total, then exit."""
    col_m  = max(len(m) for m in models) + 2
    col_b  = max(len(b) for b in benchmarks) + 2

    print(f"\n{'─'*72}")
    print(f"  Cost estimate  "
          f"(limit={limit} samples/bench, runs={runs}, {len(models)} models × {len(benchmarks)} benchmarks)")
    print(f"{'─'*72}")
    print(f"  {'Model':<{col_m}}  {'Benchmark':<{col_b}}  "
          f"{'Calls':>7}  {'In tok':>8}  {'Out tok':>8}  {'Cost $':>9}")
    print(f"  {'─'*col_m}  {'─'*col_b}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*9}")

    grand_total = 0.0
    model_totals: dict[str, float] = {}

    for model in models:
        in_price, out_price = MODEL_COSTS.get(model, (0.15, 0.15))
        if model not in MODEL_COSTS:
            print(f"  Note: {model!r} not in MODEL_COSTS — using $0.15/1M for both in/out")
        short_m = model.split("/")[-1] if "/" in model else model
        model_total = 0.0
        for bench in benchmarks:
            avg_in  = _BENCH_IN.get(bench,  _DEFAULT_IN)
            avg_out = _BENCH_OUT.get(bench, _DEFAULT_OUT)
            n_calls = limit * runs
            cost = n_calls * (avg_in * in_price + avg_out * out_price) / 1_000_000
            model_total  += cost
            grand_total  += cost
            print(f"  {short_m:<{col_m}}  {bench:<{col_b}}  "
                  f"{n_calls:>7,}  {avg_in:>8,}  {avg_out:>8,}  ${cost:>8.4f}")
        model_totals[model] = model_total

    print(f"\n  {'Model subtotals':}")
    for model, total in model_totals.items():
        short_m = model.split("/")[-1] if "/" in model else model
        print(f"    {short_m:<{col_m}}  ${total:.4f}")

    print(f"\n  {'TOTAL ESTIMATED COST':}  ${grand_total:.4f}  (~${grand_total*1.2:.4f} with 20% buffer)")
    print(f"{'─'*72}\n")
    print("  Token counts are approximate averages — actual cost may vary ±50%.")
    print("  Output tokens are a small fraction of cost for most MCQ benchmarks.")
    print("  Run without --estimate-cost to start collection.\n")


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _load_resume_state(csv_path: str) -> tuple[set[tuple[str, str, int]], float, int]:
    """Return (existing_keys, estimated_budget_usd, row_count) from existing CSV."""
    existing: set[tuple[str, str, int]] = set()
    est_budget_usd = 0.0
    row_count = 0
    p = Path(csv_path)
    if not p.exists():
        return existing, est_budget_usd, row_count
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            try:
                existing.add((row["benchmark"], row["model"], int(row["run_idx"])))
            except (KeyError, ValueError):
                pass
            # Prefer stored estimate when available; fall back to benchmark/model estimate.
            raw = row.get("est_cost_usd")
            if raw is not None and raw.strip() != "":
                try:
                    est_budget_usd += float(raw)
                    continue
                except ValueError:
                    pass
            bench = row.get("benchmark")
            model = row.get("model")
            if bench and model:
                est_budget_usd += _estimate_sample_cost_usd(bench, model)
    return existing, est_budget_usd, row_count


# ---------------------------------------------------------------------------
# Core eval runner
# ---------------------------------------------------------------------------


def run_one_eval(
    benchmark: str,
    model_id: str,
    run_idx: int,
    limit: int,
    log_dir: str,
    temperature: float = 0.7,
    eval_timeout_seconds: int = DEFAULT_EVAL_TIMEOUT_SECONDS,
) -> tuple[list[tuple[str, str, str, int, float, int, int, float, str]], float]:
    """Run one (benchmark, model, run) eval (epochs=1) and return scored rows.

    Each call is a single epoch so results can be written to disk immediately
    after it returns.  If the process crashes, only this one run is lost.

    Returns (records, eval_est_cost_usd) where each record is:
    (benchmark, model, item_id, run_idx, score, est_input_tokens, est_output_tokens, est_cost_usd, cost_source).
    Unscored/missing samples are silently skipped.
    """
    try:
        from inspect_ai._eval.eval import eval as inspect_eval
    except ImportError:
        print("ERROR: inspect-ai not installed. Run: pip install inspect-ai inspect-evals")
        sys.exit(1)

    _load_benchmark_registry()
    task_fn = _BENCHMARK_FNS.get(benchmark)
    if task_fn is None:
        err = _BENCHMARK_LOAD_ERRORS.get(benchmark, "benchmark not found in registry")
        print(f"  Skip  {benchmark}: {err}")
        return [], 0.0

    print(
        f"  Running  {benchmark}  model={model_id}  run={run_idx}  "
        f"limit={limit}  working_limit={eval_timeout_seconds}s"
    )
    t0 = time.time()

    try:
        logs = inspect_eval(
            task_fn(),
            model=model_id,
            limit=limit,
            epochs=1,
            log_dir=log_dir,
            temperature=temperature,
            working_limit=eval_timeout_seconds if eval_timeout_seconds > 0 else None,
        )
    except Exception as exc:
        print(f"  ERROR  {benchmark}/{model_id} run={run_idx}: {exc}")
        return [], 0.0

    elapsed = time.time() - t0
    log = logs[0] if isinstance(logs, list) else logs

    if log.status not in ("success", "cancelled"):
        print(f"  WARN  {benchmark}/{model_id} run={run_idx}: status={log.status!r}")

    samples = log.samples or []
    records: list[tuple[str, str, str, int, float, int, int, float, str]] = []
    n_missing = 0
    eval_est_cost_usd = 0.0

    for sample in samples:
        sample_in, sample_out, sample_cost, source = _extract_usage_or_estimate(sample, benchmark, model_id)
        eval_est_cost_usd += sample_cost

        item_id = str(sample.id)
        score = _extract_binary_score(sample.scores)
        if score is None:
            n_missing += 1
            continue
        records.append(
            (benchmark, model_id, item_id, run_idx, score, sample_in, sample_out, sample_cost, source)
        )

    print(
        f"  OK     {benchmark}/{model_id} run={run_idx}: {len(records)} scored "
        f"({n_missing} unscored)  [{elapsed:.1f}s]"
    )
    return records, eval_est_cost_usd


# ---------------------------------------------------------------------------
# CSV writer helpers
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "benchmark",
    "model",
    "item_id",
    "run_idx",
    "score",
    "est_input_tokens",
    "est_output_tokens",
    "est_cost_usd",
    "cost_source",
]


def _upgrade_csv_schema_if_needed(csv_path: str) -> None:
    """Upgrade legacy CSVs to include estimated token/cost columns."""
    p = Path(csv_path)
    if not p.exists():
        return

    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames or []

    if existing_header == CSV_HEADER:
        return

    required = {"benchmark", "model", "item_id", "run_idx", "score"}
    if not required.issubset(set(existing_header)):
        print(
            f"WARNING: Existing CSV at {csv_path} has incompatible columns: {existing_header}. "
            "Not upgrading schema automatically."
        )
        return

    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with p.open(newline="") as src, tmp_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in reader:
            bench = row.get("benchmark", "")
            model = row.get("model", "")
            est_in, est_out = _estimate_sample_tokens(bench)
            est_cost = _estimate_sample_cost_usd(bench, model)
            writer.writerow(
                {
                    "benchmark": row.get("benchmark", ""),
                    "model": row.get("model", ""),
                    "item_id": row.get("item_id", ""),
                    "run_idx": row.get("run_idx", ""),
                    "score": row.get("score", ""),
                    "est_input_tokens": str(est_in),
                    "est_output_tokens": str(est_out),
                    "est_cost_usd": f"{est_cost:.8f}",
                    "cost_source": row.get("cost_source", "estimate"),
                }
            )

    os.replace(tmp_path, p)
    print(f"Upgraded CSV schema with estimated token/cost columns: {csv_path}")


def _append_records(
    csv_path: str,
    records: list[tuple[str, str, str, int, float, int, int, float, str]],
    write_header: bool,
) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if Path(csv_path).exists() and not write_header else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for rec in records:
            benchmark, model, item_id, run_idx, score, est_in, est_out, est_cost, source = rec
            writer.writerow(
                [
                    benchmark,
                    model,
                    item_id,
                    run_idx,
                    f"{score:.1f}",
                    est_in,
                    est_out,
                    f"{est_cost:.8f}",
                    source,
                ]
            )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Inspect AI knowledge benchmarks via OpenRouter and collect "
            "per-sample binary scores for use with sim_tango_real.py --source inspect."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="OpenRouter model IDs (e.g. openrouter/meta-llama/llama-3.1-8b-instruct)",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
        help=(
            "Benchmark short names. Available: mmlu, mmlu_0_shot, mmlu_5_shot, arc, "
            "arc_challenge, arc_easy, hellaswag, gsm8k, winogrande, truthfulqa, boolq, piqa, bbq"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help="Max samples per benchmark (passed to Inspect AI)",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=(
            "Number of evaluation runs (epochs) per item per model. "
            "Use >1 to enable multi-run / nested CI methods in sim_tango_real.py."
        ),
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--log-dir", default=DEFAULT_LOG_DIR,
        help="Directory for Inspect AI eval logs (one JSON per run)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7 for multi-run, 0.0 for single-run)",
    )
    parser.add_argument(
        "--eval-timeout-seconds", type=int, default=DEFAULT_EVAL_TIMEOUT_SECONDS,
        help="Inspect working_limit in seconds per eval call (<=0 disables)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", dest="no_resume",
        help="Ignore existing --output file and re-run all evals from scratch",
    )
    parser.add_argument(
        "--estimate-cost", action="store_true",
        help="Print a cost breakdown for the planned collection and exit (no evals run)",
    )

    args = parser.parse_args()

    if args.estimate_cost:
        estimate_cost(args.models, args.benchmarks, args.limit, args.runs)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "WARNING: OPENROUTER_API_KEY not set. Evals will fail unless you are "
            "using a provider that does not require authentication."
        )

    # Load benchmark registry to show import errors early
    _load_benchmark_registry()
    for bench in args.benchmarks:
        if bench not in _BENCHMARK_FNS:
            err = _BENCHMARK_LOAD_ERRORS.get(bench, "not in registry")
            print(f"WARNING: benchmark '{bench}' unavailable: {err}")

    existing_keys: set[tuple[str, str, int]] = set()
    estimated_budget_used = 0.0
    existing_rows = 0

    _upgrade_csv_schema_if_needed(args.output)

    if not args.no_resume and Path(args.output).exists():
        existing_keys, estimated_budget_used, existing_rows = _load_resume_state(args.output)
        print(f"Resume: {len(existing_keys)} (benchmark, model, run_idx) combos already collected.")
        print(f"Resume: estimated budget already used from existing CSV: ${estimated_budget_used:.4f}")

    total_evals = len(args.models) * len(args.benchmarks) * args.runs
    done = 0
    first_write = not Path(args.output).exists()

    print(f"\nCollecting Inspect AI benchmark scores")
    print(f"  Models      : {args.models}")
    print(f"  Benchmarks  : {args.benchmarks}")
    print(f"  Limit/bench : {args.limit}")
    print(f"  Runs/model  : {args.runs}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Working limit: {args.eval_timeout_seconds}s")
    print(f"  Output      : {args.output}")
    print(f"  Log dir     : {args.log_dir}")
    print(f"  Total evals : {len(args.models)} × {len(args.benchmarks)} × {args.runs} runs "
          f"= {total_evals}\n")

    all_collected = existing_rows if not args.no_resume else 0
    session_est_budget_used = 0.0
    session_model_spend: dict[str, float] = defaultdict(float)
    session_benchmark_spend: dict[str, float] = defaultdict(float)

    # Each (model, benchmark, run_idx) is a separate eval call so results are
    # written to disk immediately after each call — a crash loses at most one run.
    for model in args.models:
        for bench in args.benchmarks:
            for run_idx in range(args.runs):
                if (bench, model, run_idx) in existing_keys:
                    print(f"  Skip  {bench}/{model} run={run_idx}: already collected")
                    done += 1
                    continue

                records, eval_est_cost_usd = run_one_eval(
                    benchmark=bench,
                    model_id=model,
                    run_idx=run_idx,
                    limit=args.limit,
                    log_dir=args.log_dir,
                    temperature=args.temperature,
                    eval_timeout_seconds=args.eval_timeout_seconds,
                )
                estimated_budget_used += eval_est_cost_usd
                session_est_budget_used += eval_est_cost_usd
                session_model_spend[model] += eval_est_cost_usd
                session_benchmark_spend[bench] += eval_est_cost_usd

                # Write immediately — crash after this line loses nothing already saved
                if records:
                    _append_records(args.output, records, write_header=first_write)
                    first_write = False
                    all_collected += len(records)

                done += 1
                pct = 100.0 * done / total_evals
                print(f"  Progress: {done}/{total_evals} evals  ({pct:.0f}%)  "
                        f"total rows: {all_collected:,}  "
                        f"est. budget used: ${estimated_budget_used:.4f}\n")

    print(f"Done. {all_collected:,} scored rows written to: {args.output}")
    if session_est_budget_used > 0:
        print("\nEstimated spend for this run:")
        print(f"  Session total          : ${session_est_budget_used:.4f}")
        print(f"  Cumulative (incl resume): ${estimated_budget_used:.4f}")

        print("\n  By model:")
        for model, spend in sorted(session_model_spend.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {model}: ${spend:.4f}")

        print("\n  By benchmark:")
        for bench, spend in sorted(session_benchmark_spend.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {bench}: ${spend:.4f}")

    print(
        f"\nTo use in sim_tango_real.py:\n"
        f"  python simulations/sim_tango_real.py "
        f"--source inspect --inspect-data {args.output}"
    )


if __name__ == "__main__":
    main()
