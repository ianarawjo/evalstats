"""Judge-scoring companion to collect_appstore_reviews_only.py.

Scores the reviews collected by that script (simulations/out/appstore_
scenario_reviews.csv by default) with one or more LLM judges, writing to
its own separate output -- never simulations/out/judge_bias_appstore_
scores.csv, which the judge-bias simulation harness reads as shared ground
truth.

Reuses the actual judge-calling machinery (client construction, retry
logic, the App Store prompt/response format) from collect_judge_bias_
data.py directly -- not a re-implementation, so there's exactly one place
that knows how to talk to OpenRouter/Ollama and one prompt template to
keep in sync.

Setup:
    pip install openai   # if not already installed

    # OpenRouter models (e.g. thinkingmachines/inkling, anthropic/*, etc.):
    export OPENROUTER_API_KEY=...

    # Ollama models (free, local, no key) -- must already be pulled:
    ollama pull gemma3:4b

Usage:
    # Same judge the paper example's kappa figures come from, plus a
    # couple more for variety (mix OpenRouter and Ollama in one run --
    # backend is per-model via --model-backends, see below):
    python -m simulations.collect_appstore_judge_scores_only \
        --models thinkingmachines/inkling gemma3:4b \
        --model-backends thinkingmachines/inkling=openrouter gemma3:4b=ollama

    # All models on one backend:
    python -m simulations.collect_appstore_judge_scores_only \
        --backend openrouter --models thinkingmachines/inkling anthropic/claude-haiku-4.5

Re-running is safe and additive: already-scored (item, model, run) combos
are skipped, so you can add more models or resume an interrupted run
without re-paying for anything already collected.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulations.collect_judge_bias_data import (  # noqa: E402
    _make_client, _call_judge, _check_model_available,
    _build_appstore_prompt, _parse_appstore_response, _progress,
)

DEFAULT_ITEMS_PATH = "simulations/out/appstore_scenario_reviews.csv"
DEFAULT_SCORES_PATH = "simulations/out/appstore_scenario_judge_scores.csv"
DEFAULT_MERGED_PATH = "simulations/out/appstore_scenario_judged.csv"
SCORES_FIELDNAMES = ["item_id", "judge_model", "run_idx", "judge_score", "raw_response", "collected_at"]
MERGED_FIELDNAMES = ["item_id", "app_id", "human_label", "judge_model", "run_idx", "judge_score"]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def _judge_one(client, model: str, backend: str, max_retries: int, sleep_s: float, item: dict, run_idx: int):
    ji = {"title": item["title"], "text": item["text"]}
    messages = _build_appstore_prompt(ji)
    raw = _call_judge(client, model, messages, backend=backend, max_retries=max_retries, sleep_s=sleep_s)
    score = None if raw is None else _parse_appstore_response(raw)
    return item, run_idx, raw, score


def _parse_model_backends(pairs: list[str] | None) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--model-backends entries must be MODEL=BACKEND, got {p!r}")
        model, backend = p.split("=", 1)
        out[model] = backend
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default=DEFAULT_ITEMS_PATH)
    ap.add_argument("--scores-out", default=DEFAULT_SCORES_PATH)
    ap.add_argument("--merged-out", default=DEFAULT_MERGED_PATH)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--backend", choices=["openrouter", "ollama"], default="ollama",
                     help="Default backend for any model not listed in --model-backends.")
    ap.add_argument("--model-backends", nargs="+", default=None,
                     help="Per-model backend overrides as MODEL=BACKEND (e.g. "
                          "thinkingmachines/inkling=openrouter), for mixing backends in one run.")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="Cap total (item, run) combos per model, for a smoke test.")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0, help="Delay after each successful call.")
    args = ap.parse_args()

    items_path = Path(args.items)
    scores_path = Path(args.scores_out)
    merged_path = Path(args.merged_out)
    model_backends = _parse_model_backends(args.model_backends)

    items = _read_csv(items_path)
    if not items:
        raise SystemExit(f"No items in {items_path} -- run collect_appstore_reviews_only.py first.")
    print(f"Loaded {len(items)} reviews from {items_path}")
    print(f"Scores -> {scores_path} (separate from judge_bias_appstore_scores.csv)")
    print()

    existing_scores = _read_csv(scores_path)
    done_keys = {(r["item_id"], r["judge_model"], r["run_idx"]) for r in existing_scores}
    write_lock = threading.Lock()

    clients: dict[str, object] = {}
    for model_i, model in enumerate(args.models, start=1):
        backend = model_backends.get(model, args.backend)
        print(f"{'=' * 72}\n[{model_i}/{len(args.models)}] model={model!r} backend={backend!r}\n{'=' * 72}")

        if backend not in clients:
            try:
                clients[backend] = _make_client(backend)
            except SystemExit as e:
                print(f"  SKIPPING backend {backend!r}: {e}")
                continue
        client = clients[backend]

        err = _check_model_available(client, model, backend)
        if err is not None:
            print(f"  SKIPPING {model!r} -- not reachable: {err}")
            continue

        work_items: list[tuple[dict, int]] = []
        n_skipped = 0
        for item in items:
            for run_idx in range(args.runs):
                key = (item["item_id"], model, str(run_idx))
                if key in done_keys:
                    n_skipped += 1
                    continue
                work_items.append((item, run_idx))
        if args.limit is not None:
            work_items = work_items[:args.limit]

        print(f"{len(items)} items, runs={args.runs}, {n_skipped} combos already done, "
              f"{len(work_items)} to collect, concurrency={args.concurrency}.")

        n_new = n_failed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(_judge_one, client, model, backend, args.max_retries, args.sleep, item, run_idx)
                for item, run_idx in work_items
            ]
            pbar = _progress(as_completed(futures), total=len(futures), desc=model, unit="call")
            for fut in pbar:
                try:
                    item, run_idx, raw, score = fut.result()
                except Exception as e:  # noqa: BLE001 -- one bad item must not abort the run
                    n_failed += 1
                    continue
                if score is None:
                    n_failed += 1
                    msg = f"    could not parse response for {item['item_id']} run {run_idx}: {raw!r}"
                    pbar.write(msg) if hasattr(pbar, "write") else print(msg)
                    continue
                row = {
                    "item_id": item["item_id"], "judge_model": model, "run_idx": run_idx,
                    "judge_score": score, "raw_response": raw,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
                with write_lock:
                    _append_csv_row(scores_path, row, SCORES_FIELDNAMES)
                n_new += 1
            if hasattr(pbar, "close"):
                pbar.close()

        print(f"  -> {n_new} new scores written ({n_skipped} already done, {n_failed} unparseable)\n")

    # Merged convenience view: item + human_label + every judge's score.
    items_by_id = {r["item_id"]: r for r in items}
    all_scores = _read_csv(scores_path)
    merged = []
    for s in all_scores:
        it = items_by_id.get(s["item_id"])
        if it is None:
            continue
        merged.append({
            "item_id": s["item_id"], "app_id": it["app_id"], "human_label": it["human_label"],
            "judge_model": s["judge_model"], "run_idx": s["run_idx"], "judge_score": s["judge_score"],
        })
    _write_csv(merged_path, merged, MERGED_FIELDNAMES)
    print(f"Merged view -> {merged_path}: {len(merged)} (item, judge_model, run) rows.")


if __name__ == "__main__":
    main()
