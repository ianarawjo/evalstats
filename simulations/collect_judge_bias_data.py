#!/usr/bin/env python3
"""
collect_judge_bias_data.py -- one-off script to build small real-data
calibration sets for the PPI judge-bias simulations
(simulations/harness/cases/pvalues.py --mode ppi), one dataset per eval_type:

  binary      lmarena-ai/arena-human-preference-140k   human pairwise votes
              on live model battles (fresh: timestamps run into 2025).
  continuous  RicardoRei/wmt-da-human-evaluation        human 0-100 Direct
              Assessment scores of machine-translation quality.
  likert      Apple App Store customer-review RSS feed  human 1-5 star
              ratings -- fetched live, so always as fresh as "today".

This is deliberately NOT part of simulations/harness/ -- it is a one-off
data-collection tool, not simulation code the harness imports. It has two
subcommands:

  collect-data
      Fetch each dataset's items + real human labels (no LLM calls) and
      write them to simulations/out/judge_bias_<key>_items.csv. Safe to
      re-run: existing item_ids are left alone, only new ones are added.

  collect-judge-scores
      For each item collected above, prompt an LLM judge (OpenRouter or a
      local Ollama server) for its own score on the SAME scale as the human
      label -- withholding the human label from the prompt -- and append
      the result to simulations/out/judge_bias_<key>_scores.csv. Resumable:
      already-collected (item_id, judge_model, run_idx) triples are skipped.
      Also (re)writes a convenience merged view,
      simulations/out/judge_bias_<key>.csv, joining items + scores into one
      file with both a human_label and a judge_score column per row.

Why these three datasets: all three are either continuously updated
(arena, app-store reviews) or carry an explicit year/date field you can
filter on, so you can select a slice that postdates a judge model's
training cutoff -- the whole point being a "did the judge actually infer
this, or does it just remember the label" sanity check. The app-store feed
in particular returns literally today's reviews.

Install dependencies first (all present in the project's dev venv, .venv/):
  pip install datasets huggingface_hub requests openai tqdm  # tqdm optional, for progress bars

Usage:
  # Step 1: download/fetch the three real-data corpora (no LLM calls yet)
  python simulations/collect_judge_bias_data.py collect-data

  # Step 2a: score with a cheap OpenRouter model
  export OPENROUTER_API_KEY=...
  python simulations/collect_judge_bias_data.py collect-judge-scores \\
      --backend openrouter --models openai/gpt-4o-mini

  # Step 2b: score locally with Ollama instead (cheaper, no API key) --
  # --models must match tags you've already pulled (see `ollama list`)
  ollama pull llama3.1:8b
  python simulations/collect_judge_bias_data.py collect-judge-scores \\
      --backend ollama --models llama3.1:8b

  # Multiple local judges in one unattended run (e.g. overnight) -- each
  # runs to completion in sequence before the next starts; a bad/unpulled
  # model name is skipped (with a warning) rather than aborting the run
  python simulations/collect_judge_bias_data.py collect-judge-scores \\
      --backend ollama --models gemma3:1b gemma3:4b qwen3:4b gpt-oss:20b

  # Only one data type, small trial run
  python simulations/collect_judge_bias_data.py collect-judge-scores \\
      --types likert --backend ollama --models gemma3:4b --limit 20

Output CSV columns (simulations/out/judge_bias_<key>.csv, the merged view):
  item_id      stable per-item ID (str)
  eval_type    "binary" | "continuous" | "likert"
  dataset      "arena" | "wmt_da" | "appstore"
  human_label  ground-truth human label, NATIVE scale per eval_type:
               binary in {0.0, 1.0}; continuous (WMT DA) in [0, 100];
               likert (app-store stars) in {1, 2, 3, 4, 5}
  judge_model  judge model ID used for this row
  run_idx      0-indexed judge replicate for this (item, model)
  judge_score  judge's score, SAME native scale as human_label
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    key: str  # "arena" | "wmt_da" | "appstore"
    eval_type: str  # "binary" | "continuous" | "likert"
    native_bounds: tuple[float, float]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "binary": DatasetSpec(key="arena", eval_type="binary", native_bounds=(0.0, 1.0)),
    "continuous": DatasetSpec(key="wmt_da", eval_type="continuous", native_bounds=(0.0, 100.0)),
    "likert": DatasetSpec(key="appstore", eval_type="likert", native_bounds=(1.0, 5.0)),
}

DEFAULT_APPSTORE_APP_IDS = [
    284882215,   # Facebook
    389801252,   # Instagram
    324684580,   # Spotify
    310633997,   # WhatsApp Messenger
    835599320,   # TikTok
    447188370,   # Snapchat
    454638411,   # Messenger
    297606951,   # Amazon
    922103212,   # McDonald's
    6446901002,  # Threads
    1641486558,  # Temu
    283646709,   # PayPal
    585027354,   # Google Maps
    429047995,   # Pinterest
    546505307,   # Zoom Workplace
    331177714,   # Starbucks
    311548709,   # Wells Fargo
]
# NOTE: Apple's customer-review RSS feed is an unofficial, unreliable
# endpoint -- the same app_id can return 0 entries on one request and 50 on
# the next (observed directly while building this list), and a burst of
# many requests in a short window (e.g. testing many app_ids back to back)
# appears to trigger a temporary all-empty state that outlasts a few
# retries. fetch_appstore_items retries an empty first page a few times to
# absorb ordinary flakiness, but can't fully compensate for a rate-limited
# window. The list above is deliberately generous (17 apps, several
# categories) so total volume stays reasonable even when several apps come
# back empty on a given run -- and since collect-data is additive, simply
# re-running the command later accumulates more items rather than
# resetting.

ITEMS_FIELDNAMES = ["item_id", "eval_type", "dataset", "human_label", "judge_input"]
SCORES_FIELDNAMES = ["item_id", "judge_model", "run_idx", "judge_score", "raw_response", "collected_at"]
MERGED_FIELDNAMES = ["item_id", "eval_type", "dataset", "human_label", "judge_model", "run_idx", "judge_score"]


def _out_paths(out_dir: Path, spec: DatasetSpec) -> dict[str, Path]:
    return {
        "items": out_dir / f"judge_bias_{spec.key}_items.csv",
        "scores": out_dir / f"judge_bias_{spec.key}_scores.csv",
        "merged": out_dir / f"judge_bias_{spec.key}.csv",
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Reservoir sampling (avoid materializing the full upstream dataset)
# ---------------------------------------------------------------------------


def _reservoir_add(reservoir: list, item: Any, seen: int, k: int, rng: random.Random) -> None:
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        j = rng.randint(0, seen - 1)
        if j < k:
            reservoir[j] = item


# ---------------------------------------------------------------------------
# Progress bar helper -- tqdm is optional (not a hard dependency of this
# script); falls back to the plain iterable with a one-time hint if it isn't
# installed. Callers guard the tqdm-only methods (set_postfix/write/close)
# with hasattr so they work identically either way.
# ---------------------------------------------------------------------------


def _progress(iterable, **kwargs):
    try:
        from tqdm import tqdm
    except ImportError:
        print("  (pip install tqdm to see a progress bar here)")
        return iterable
    return tqdm(iterable, **kwargs)


def _warn_if_short(reservoir: list, n_items: int, hint: str) -> None:
    if len(reservoir) < n_items:
        print(f"  NOTE: only got {len(reservoir)}/{n_items} requested -- {hint}")


# ---------------------------------------------------------------------------
# collect-data: binary -- lmarena-ai/arena-human-preference-140k
# ---------------------------------------------------------------------------

ARENA_DATASET = "lmarena-ai/arena-human-preference-140k"


def _extract_text(content_blocks: list[dict] | None) -> str:
    parts = []
    for b in content_blocks or []:
        if b.get("type") == "text" and b.get("text"):
            parts.append(b["text"])
    return "\n".join(parts)


def _conversation_prompt_and_response(conv: list[dict]) -> tuple[str, str]:
    """Returns (all user turns joined, final assistant turn) -- keeps the
    judge prompt simple for the common single-turn case while still working
    for multi-turn battles."""
    user_parts: list[str] = []
    last_assistant = ""
    for turn in conv or []:
        role = (turn.get("role") or "").lower()
        text = _extract_text(turn.get("content"))
        if role == "user":
            user_parts.append(text)
        else:
            last_assistant = text
    return "\n".join(user_parts), last_assistant


def _arena_row_to_item(row: dict, max_chars: int = 3000) -> dict | None:
    prompt_a, resp_a = _conversation_prompt_and_response(row.get("conversation_a"))
    prompt_b, resp_b = _conversation_prompt_and_response(row.get("conversation_b"))
    prompt = prompt_a or prompt_b
    if not prompt or not resp_a or not resp_b:
        return None
    return {
        "item_id": row["id"],
        "human_label": 1.0 if row["winner"] == "model_a" else 0.0,
        "judge_input": {
            "prompt": prompt[:max_chars],
            "response_a": resp_a[:max_chars],
            "response_b": resp_b[:max_chars],
        },
    }


def fetch_arena_items(*, n_items: int, min_date: str | None, max_scan: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print(f"Streaming {ARENA_DATASET} (binary: human pairwise arena votes) ...")
    ds = load_dataset(ARENA_DATASET, split="train", streaming=True)
    # NOTE: filtered against an explicit calendar date, not "days before today"
    # -- this dataset is a fixed HF snapshot (its battles stop at whatever date
    # it was last published), not a live feed, so "N days before wall-clock
    # now" would silently filter out everything once the snapshot is more than
    # N days old. Pick --arena-min-date by comparing the observed timestamp
    # range printed below against your judge model's training cutoff.
    cutoff = datetime.fromisoformat(min_date) if min_date else None

    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    scanned = 0
    ts_min = ts_max = None
    pbar = _progress(ds, total=max_scan, desc="arena", unit="row")
    for row in pbar:
        scanned += 1
        if scanned > max_scan:
            break
        if row.get("winner") not in ("model_a", "model_b"):
            continue
        ts = row.get("timestamp")
        if ts is not None:
            ts_min = ts if ts_min is None or ts < ts_min else ts_min
            ts_max = ts if ts_max is None or ts > ts_max else ts_max
        if cutoff is not None and ts is not None and ts < cutoff:
            continue
        item = _arena_row_to_item(row)
        if item is None:
            continue
        seen += 1
        _reservoir_add(reservoir, item, seen, n_items, rng)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(eligible=seen, sampled=len(reservoir))
    if hasattr(pbar, "close"):
        pbar.close()

    print(f"  Scanned {scanned:,} rows, {seen:,} eligible (winner in model_a/model_b"
          f"{f', timestamp >= {min_date}' if min_date else ''}), sampled {len(reservoir)}.")
    if ts_min is not None:
        print(f"  Observed timestamp range in scanned rows: {ts_min} .. {ts_max} "
              f"-- set --arena-min-date within/near this range.")
    if min_date and seen == 0:
        print(f"  NOTE: 0 eligible rows -- --arena-min-date={min_date} is likely outside the "
              f"range above (or --arena-max-scan is too small to reach it).")
    _warn_if_short(reservoir, n_items, "raise --arena-max-scan to scan further into the stream.")
    return reservoir


# ---------------------------------------------------------------------------
# collect-data: continuous -- RicardoRei/wmt-da-human-evaluation
# ---------------------------------------------------------------------------

WMT_DA_DATASET = "RicardoRei/wmt-da-human-evaluation"


def fetch_wmt_da_items(*, n_items: int, min_year: int, max_scan: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print(f"Streaming {WMT_DA_DATASET} (continuous: human 0-100 MT Direct Assessment scores) ...")
    ds = load_dataset(WMT_DA_DATASET, split="train", streaming=True)

    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    scanned = 0
    pbar = _progress(ds, total=max_scan, desc="wmt_da", unit="row")
    for row in pbar:
        scanned += 1
        if scanned > max_scan:
            break
        if (row.get("year") or 0) < min_year:
            continue
        raw = row.get("raw")
        if raw is None or raw != raw:  # NaN check
            continue
        item = {
            "item_id": f"wmt_da_{scanned}",
            "human_label": float(raw),
            "judge_input": {
                "lp": row.get("lp"), "source": row.get("src"),
                "reference": row.get("ref"), "candidate": row.get("mt"),
            },
        }
        seen += 1
        _reservoir_add(reservoir, item, seen, n_items, rng)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(eligible=seen, sampled=len(reservoir))
    if hasattr(pbar, "close"):
        pbar.close()

    print(f"  Scanned {scanned:,} rows, {seen:,} eligible (year >= {min_year}, has 'raw' score), "
          f"sampled {len(reservoir)}.")
    if seen == 0:
        print(f"  NOTE: 0 eligible rows -- this HF mirror's ~1.29M rows are ordered ascending by "
              f"year (2017 through ~2022), so a low --wmt-max-scan can finish before reaching "
              f"{min_year}. Try a higher --wmt-max-scan (default already covers the whole "
              f"dataset) or a lower --wmt-min-year.")
    _warn_if_short(reservoir, n_items, "raise --wmt-max-scan (it may not have reached the end of the ascending-by-year stream yet).")
    return reservoir


# ---------------------------------------------------------------------------
# collect-data: likert -- Apple App Store customer-review RSS feed
# ---------------------------------------------------------------------------


def fetch_appstore_items(
    *, app_ids: list[int], country: str, n_items: int, days_back: int, max_pages: int, seed: int,
    empty_retries: int = 3, retry_sleep: float = 2.0,
) -> list[dict]:
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    print(f"Fetching Apple App Store customer-review RSS feed for {len(app_ids)} app(s) "
          f"(likert: human 1-5 star ratings, live/today's reviews) ...")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    items: list[dict] = []
    pbar = _progress(app_ids, desc="appstore", unit="app")
    for app_id in pbar:
        n_app = 0
        page = 1
        retries_left = empty_retries
        while page <= max_pages:
            url = (f"https://itunes.apple.com/{country}/rss/customerreviews/"
                   f"page={page}/id={app_id}/sortBy=mostRecent/json")
            try:
                resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            except requests.RequestException as e:
                msg = f"  {app_id} page {page}: request failed ({e}), stopping this app."
                pbar.write(msg) if hasattr(pbar, "write") else print(msg)
                break
            if resp.status_code != 200:
                break
            entries = resp.json().get("feed", {}).get("entry", [])
            if not entries:
                # Apple's feed is flaky per-call (observed: same app_id
                # returns 0 entries one request, 50 the next, seconds
                # apart) -- only retry the FIRST page, and only while this
                # app hasn't yielded anything yet, since a later empty page
                # legitimately just means end-of-feed.
                if n_app == 0 and retries_left > 0:
                    retries_left -= 1
                    time.sleep(retry_sleep)
                    continue
                break
            for e in entries:
                rating = e.get("im:rating", {}).get("label")
                updated = e.get("updated", {}).get("label")
                rid = e.get("id", {}).get("label")
                if rating is None or updated is None or rid is None:
                    continue
                try:
                    ts = datetime.fromisoformat(updated)
                except ValueError:
                    continue
                if ts < cutoff:
                    continue
                title = e.get("title", {}).get("label", "")
                text = e.get("content", {}).get("label", "")
                items.append({
                    "item_id": f"appstore_{app_id}_{rid}",
                    "human_label": float(int(rating)),
                    "judge_input": {"title": title, "text": text},
                })
                n_app += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(app=app_id, collected=len(items))
            page += 1
            time.sleep(0.3)
        msg = f"  app {app_id}: {n_app} reviews within {days_back}d."
        pbar.write(msg) if hasattr(pbar, "write") else print(msg)
    if hasattr(pbar, "close"):
        pbar.close()

    rng = random.Random(seed)
    rng.shuffle(items)
    print(f"  Total {len(items)} reviews collected, sampling {min(n_items, len(items))}.")
    _warn_if_short(items, n_items, "Apple's RSS feed is unofficial and unreliable per-app/per-call "
                   "-- collect-data is additive, so simply re-running this command later (when "
                   "more apps' feeds happen to respond) will accumulate more items rather than "
                   "starting over. You can also add more --appstore-app-ids or raise "
                   "--appstore-days-back / --appstore-max-pages.")
    return items[:n_items]


# ---------------------------------------------------------------------------
# collect-data: dispatch + merge-safe write
# ---------------------------------------------------------------------------

FETCHERS: dict[str, Callable[..., list[dict]]] = {
    "arena": fetch_arena_items,
    "wmt_da": fetch_wmt_da_items,
    "appstore": fetch_appstore_items,
}


def _item_to_row(item: dict, spec: DatasetSpec) -> dict:
    return {
        "item_id": item["item_id"],
        "eval_type": spec.eval_type,
        "dataset": spec.key,
        "human_label": item["human_label"],
        "judge_input": json.dumps(item["judge_input"], ensure_ascii=False),
    }


def run_collect_data(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    for eval_type in args.types:
        spec = DATASET_SPECS[eval_type]
        paths = _out_paths(out_dir, spec)
        existing_rows = _read_csv(paths["items"])
        existing_ids = {r["item_id"] for r in existing_rows}

        if spec.key == "arena":
            items = fetch_arena_items(
                n_items=args.n_items, min_date=args.arena_min_date,
                max_scan=args.arena_max_scan, seed=args.seed,
            )
        elif spec.key == "wmt_da":
            items = fetch_wmt_da_items(
                n_items=args.n_items, min_year=args.wmt_min_year,
                max_scan=args.wmt_max_scan, seed=args.seed,
            )
        elif spec.key == "appstore":
            items = fetch_appstore_items(
                app_ids=args.appstore_app_ids, country=args.appstore_country,
                n_items=args.n_items, days_back=args.appstore_days_back,
                max_pages=args.appstore_max_pages, seed=args.seed,
                empty_retries=args.appstore_empty_retries, retry_sleep=args.appstore_retry_sleep,
            )
        else:
            raise ValueError(spec.key)

        # Dedup against previously-saved item_ids AND within this batch itself --
        # arena/wmt_da can't produce internal duplicates (each source row is
        # visited at most once per run by construction), but appstore's
        # pagination combined with the feed's observed flakiness/reordering
        # means the same review could in principle turn up twice in one fetch.
        new_rows = []
        seen_this_run: set[str] = set()
        n_dupe = 0
        for it in items:
            iid = it["item_id"]
            if iid in existing_ids or iid in seen_this_run:
                n_dupe += 1
                continue
            seen_this_run.add(iid)
            new_rows.append(_item_to_row(it, spec))
        all_rows = existing_rows + new_rows
        _write_csv(paths["items"], all_rows, ITEMS_FIELDNAMES)
        dupe_note = f" ({n_dupe} duplicate item_id(s) skipped)" if n_dupe else ""
        print(f"  -> {paths['items']}: {len(existing_rows)} existing + {len(new_rows)} new "
              f"= {len(all_rows)} items.{dupe_note}\n")


# ---------------------------------------------------------------------------
# collect-judge-scores: prompt builders + response parsers, per dataset key
# ---------------------------------------------------------------------------


def _build_arena_prompt(ji: dict) -> list[dict]:
    return [
        {"role": "system", "content": (
            "You are comparing two AI assistant responses to the same user message. "
            "Judge which response is better overall (helpfulness, correctness, clarity). "
            "Respond with exactly one character: A or B. No explanation, no punctuation."
        )},
        {"role": "user", "content": (
            f"User message:\n{ji['prompt']}\n\n"
            f"Response A:\n{ji['response_a']}\n\n"
            f"Response B:\n{ji['response_b']}\n\n"
            "Which response is better, A or B?"
        )},
    ]


def _parse_arena_response(text: str) -> float | None:
    m = re.search(r"\b([AB])\b", text.strip(), re.IGNORECASE)
    return None if m is None else (1.0 if m.group(1).upper() == "A" else 0.0)


def _build_wmt_da_prompt(ji: dict) -> list[dict]:
    return [
        {"role": "system", "content": (
            "You are an expert machine-translation quality evaluator. Rate the candidate "
            "translation's quality on a scale from 0 (nonsense) to 100 (perfect), given the "
            "source text and a reference translation. Respond with ONLY the integer score."
        )},
        {"role": "user", "content": (
            f"Language pair: {ji['lp']}\n\n"
            f"Source:\n{ji['source']}\n\n"
            f"Reference translation:\n{ji['reference']}\n\n"
            f"Candidate translation:\n{ji['candidate']}\n\n"
            "Quality score (0-100):"
        )},
    ]


def _parse_wmt_da_response(text: str) -> float | None:
    m = re.search(r"-?\d+(\.\d+)?", text)
    return None if m is None else max(0.0, min(100.0, float(m.group(0))))


def _build_appstore_prompt(ji: dict) -> list[dict]:
    return [
        {"role": "system", "content": (
            "You predict the star rating (1 to 5) a user gave an app, based only on their "
            "review text. Respond with ONLY a single integer from 1 to 5."
        )},
        {"role": "user", "content": (
            f"Review title: {ji['title']}\nReview text: {ji['text']}\n\n"
            "Predicted star rating (1-5):"
        )},
    ]


def _parse_appstore_response(text: str) -> float | None:
    m = re.search(r"[1-5]", text)
    return None if m is None else float(m.group(0))


PROMPT_BUILDERS: dict[str, Callable[[dict], list[dict]]] = {
    "arena": _build_arena_prompt, "wmt_da": _build_wmt_da_prompt, "appstore": _build_appstore_prompt,
}
RESPONSE_PARSERS: dict[str, Callable[[str], float | None]] = {
    "arena": _parse_arena_response, "wmt_da": _parse_wmt_da_response, "appstore": _parse_appstore_response,
}


# ---------------------------------------------------------------------------
# Judge client (OpenAI-compatible chat-completions endpoint) -- both
# OpenRouter and a local Ollama server speak this same API, so one thin
# client covers both backends.
# ---------------------------------------------------------------------------


def _make_client(backend: str):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    if backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY must be set for --backend openrouter.")
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return OpenAI(base_url=base_url, api_key=api_key)
    if backend == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAI(base_url=base_url, api_key="ollama")
    raise ValueError(f"Unknown backend: {backend!r}")


def _extra_body_for(backend: str) -> dict:
    """Backend-specific extra request-body fields for reasoning models.

    OpenRouter normalizes reasoning-token control across providers via a
    top-level `reasoning` object -- {"exclude": True} lets the model still
    think internally (answer quality isn't degraded) but strips the
    reasoning trace out of the response, so message.content gets the final
    answer instead of an empty string starved by a small max_tokens budget.
    Ignored harmlessly by non-reasoning OpenRouter models. NOT sent for
    Ollama: its equivalent (`"think": False`, a differently-shaped, Ollama-
    specific field) was tried empirically against gpt-oss:20b and had no
    effect -- that path instead relies on the generous max_tokens=2048
    below, confirmed empirically to leave reasoning models enough room to
    finish thinking AND answer (a real arena item used 393 completion
    tokens end-to-end, finish_reason="stop"). This OpenRouter-side fix is
    UNVERIFIED against a live reasoning model (no OPENROUTER_API_KEY
    available in this environment to test with) -- worth a small --limit
    smoke test against whichever reasoning model you actually use there."""
    if backend == "openrouter":
        return {"reasoning": {"exclude": True}}
    return {}


def _check_model_available(client, model: str, backend: str) -> str | None:
    """One quick, no-retry call to fail fast on a bad/unpulled model name --
    returns None if the model responded, else a short error string. Without
    this, an overnight multi-model run would burn through max_retries'
    exponential backoff on EVERY single item for a model that was never
    going to work (e.g. a typo, or `ollama pull` never run for it)."""
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=1,
            extra_body=_extra_body_for(backend),
        )
        return None
    except Exception as e:  # noqa: BLE001 -- any failure here means "skip this model"
        return str(e)


def _call_judge(
    client, model: str, messages: list[dict], *, backend: str, max_retries: int, sleep_s: float,
) -> str | None:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                # 2048, not something terse like 16 -- reasoning models (e.g.
                # gpt-oss:20b) spend a chunk of the budget on a separate
                # "reasoning" channel (Ollama splits it into
                # message.reasoning, not message.content) BEFORE emitting the
                # actual answer; a small cap hits finish_reason="length"
                # mid-thought and returns empty content, which then fails to
                # parse downstream. Confirmed empirically: a real arena item
                # used 393 completion tokens end-to-end. Non-reasoning models
                # just stop early (finish_reason="stop") well under this cap,
                # so it costs them nothing. See _extra_body_for for the
                # OpenRouter-side reasoning.exclude complement to this.
                model=model, messages=messages, temperature=0.0, max_tokens=2048,
                extra_body=_extra_body_for(backend),
            )
            time.sleep(sleep_s)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 -- best-effort retry over any transport/API error
            last_err = e
            time.sleep(min(2 ** attempt, 10))
    print(f"    WARNING: judge call failed after {max_retries} attempts: {last_err}")
    return None


# ---------------------------------------------------------------------------
# collect-judge-scores: dispatch
# ---------------------------------------------------------------------------


def run_collect_judge_scores(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    client = _make_client(args.backend)

    n_models = len(args.models)
    for model_i, model in enumerate(args.models, start=1):
        print(f"\n{'=' * 72}\n[{model_i}/{n_models}] model={model!r}\n{'=' * 72}")

        err = _check_model_available(client, model, args.backend)
        if err is not None:
            print(f"  SKIPPING {model!r} -- not reachable: {err}")
            continue

        for eval_type in args.types:
            spec = DATASET_SPECS[eval_type]
            paths = _out_paths(out_dir, spec)
            items = _read_csv(paths["items"])
            if not items:
                print(f"[{eval_type}] no items in {paths['items']} -- run collect-data first. Skipping.")
                continue

            existing_scores = _read_csv(paths["scores"])
            done_keys = {(r["item_id"], r["judge_model"], r["run_idx"]) for r in existing_scores}

            build_prompt = PROMPT_BUILDERS[spec.key]
            parse_response = RESPONSE_PARSERS[spec.key]

            print(f"[{eval_type}] {len(items)} items, model={model!r}, runs={args.runs}, "
                  f"{len(done_keys)} (item,model,run) combos already collected.")

            n_new = 0
            n_skipped = 0
            n_failed = 0
            pbar = _progress(items, desc=f"{eval_type}-{model}", unit="item")
            for item in pbar:
                if args.limit is not None and n_new >= args.limit:
                    break
                ji = json.loads(item["judge_input"])
                for run_idx in range(args.runs):
                    key = (item["item_id"], model, str(run_idx))
                    if key in done_keys:
                        n_skipped += 1
                        continue
                    if args.limit is not None and n_new >= args.limit:
                        break
                    messages = build_prompt(ji)
                    raw = _call_judge(client, model, messages, backend=args.backend, max_retries=args.max_retries, sleep_s=args.sleep)
                    score = None if raw is None else parse_response(raw)
                    if score is None:
                        n_failed += 1
                        msg = f"    could not parse judge response for {item['item_id']} run {run_idx}: {raw!r}"
                        pbar.write(msg) if hasattr(pbar, "write") else print(msg)
                        continue
                    row = {
                        "item_id": item["item_id"], "judge_model": model, "run_idx": run_idx,
                        "judge_score": score, "raw_response": raw,
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    }
                    _append_csv_row(paths["scores"], row, SCORES_FIELDNAMES)
                    done_keys.add(key)
                    n_new += 1
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(new=n_new, skipped=n_skipped, failed=n_failed)
            if hasattr(pbar, "close"):
                pbar.close()

            print(f"  -> {n_new} new judge scores written to {paths['scores']}"
                  f" ({n_skipped} already done, {n_failed} unparseable)")
            _write_merged_view(paths, spec)

    print(f"\n{'=' * 72}\nDone -- {n_models} model(s) processed.\n{'=' * 72}")


def _write_merged_view(paths: dict[str, Path], spec: DatasetSpec) -> None:
    items = {r["item_id"]: r for r in _read_csv(paths["items"])}
    scores = _read_csv(paths["scores"])
    merged = []
    for s in scores:
        it = items.get(s["item_id"])
        if it is None:
            continue
        merged.append({
            "item_id": s["item_id"], "eval_type": it["eval_type"], "dataset": it["dataset"],
            "human_label": it["human_label"], "judge_model": s["judge_model"],
            "run_idx": s["run_idx"], "judge_score": s["judge_score"],
        })
    _write_csv(paths["merged"], merged, MERGED_FIELDNAMES)
    print(f"  -> {paths['merged']}: {len(merged)} merged (item, judge_model, run) rows.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect real-data (human_label, judge_score) calibration sets for pvalues.py's PPI mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    common_types = dict(
        choices=["binary", "continuous", "likert"], nargs="+",
        default=["binary", "continuous", "likert"],
        help="Which eval-type dataset(s) to process (default: all three).",
    )

    pd_ = sub.add_parser("collect-data", help="Fetch items + human labels (no LLM calls).")
    pd_.add_argument("--types", **common_types)
    pd_.add_argument("--out-dir", default="simulations/out")
    pd_.add_argument("--n-items", type=int, default=1000,
                      help="Items to sample per dataset (PPI regimes typically want >= ~1000).")
    pd_.add_argument("--seed", type=int, default=0)
    pd_.add_argument("--arena-max-scan", type=int, default=100_000,
                      help="Cap on upstream rows scanned for arena streaming reservoir sampling "
                           "(~72%% of rows are eligible, so this comfortably covers --n-items 1000+).")
    pd_.add_argument("--arena-min-date", default=None,
                      help="ISO date (YYYY-MM-DD); only keep arena battles at/after this date. "
                           "Default: no filter (this dataset is a fixed snapshot, not a live feed -- "
                           "the run prints its observed timestamp range so you can pick a value that "
                           "postdates your judge model's training cutoff).")
    pd_.add_argument("--wmt-min-year", type=int, default=2020,
                      help="Only keep WMT DA rows with year >= this.")
    pd_.add_argument("--wmt-max-scan", type=int, default=1_400_000,
                      help="Cap on upstream rows scanned for WMT DA streaming reservoir sampling. "
                           "This mirror's ~1.29M rows are ordered ASCENDING by year (2017 first) -- "
                           "the default covers the whole dataset so a --wmt-min-year filter can "
                           "actually be reached; a low cap here silently yields 0 eligible rows "
                           "if it stops scanning before the requested year range.")
    pd_.add_argument("--appstore-app-ids", type=int, nargs="+", default=DEFAULT_APPSTORE_APP_IDS)
    pd_.add_argument("--appstore-country", default="us")
    pd_.add_argument("--appstore-days-back", type=int, default=45,
                      help="Only keep app-store reviews updated within this many days.")
    pd_.add_argument("--appstore-max-pages", type=int, default=15,
                      help="RSS pages (~50 reviews each) to fetch per app.")
    pd_.add_argument("--appstore-empty-retries", type=int, default=3,
                      help="Apple's feed is flaky per-call (same app_id can return 0 entries one "
                           "request and 50 the next); retry an app's empty first page this many "
                           "times before concluding it truly has none in range.")
    pd_.add_argument("--appstore-retry-sleep", type=float, default=2.0,
                      help="Seconds to wait between empty-page retries.")
    pd_.set_defaults(func=run_collect_data)

    ps = sub.add_parser("collect-judge-scores", help="Prompt an LLM judge and cache its scores.")
    ps.add_argument("--types", **common_types)
    ps.add_argument("--out-dir", default="simulations/out")
    ps.add_argument("--backend", choices=["openrouter", "ollama"], required=True)
    ps.add_argument("--models", nargs="+", required=True,
                     help="One or more OpenRouter model ids (e.g. openai/gpt-4o-mini) or local Ollama "
                          "tags (e.g. llama3.1:8b qwen3:4b gpt-oss:20b), run in sequence -- e.g. for an "
                          "unattended overnight multi-judge collection. Each gets a quick reachability "
                          "check first (a bad/unpulled model name is skipped with a warning instead of "
                          "burning through every item's retries).")
    ps.add_argument("--runs", type=int, default=1, help="Independent judge replicates per item.")
    ps.add_argument("--limit", type=int, default=None, help="Cap on NEW judge calls per model, per type, this invocation.")
    ps.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between judge calls.")
    ps.add_argument("--max-retries", type=int, default=3)
    ps.set_defaults(func=run_collect_judge_scores)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
