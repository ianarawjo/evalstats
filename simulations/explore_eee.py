#!/usr/bin/env python3
"""
explore_eee.py — Explore the Every Eval Ever (EEE) datastore on HuggingFace.

Scans the EEE manifest to find which (benchmark, model) pairs have
instance-level JSONL files, then optionally fetches the first row of each
to preview the score format — without downloading full files.

Dataset: https://huggingface.co/datasets/evaleval/EEE_datastore

Usage
  python simulations/explore_eee.py
  python simulations/explore_eee.py --no-preview       # skip first-row fetch
  python simulations/explore_eee.py --benchmark mmlu   # filter by benchmark name
  python simulations/explore_eee.py --limit 10         # cap first-row fetches
  python simulations/explore_eee.py --hf-token hf_xxx  # private/gated repos
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Optional

import requests

HF_REPO = "evaleval/EEE_datastore"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
HF_API  = f"https://huggingface.co/api/datasets/{HF_REPO}"
MANIFEST_URL = f"{HF_BASE}/manifest.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def hf_headers(token: Optional[str]) -> dict:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def fetch_manifest(token: Optional[str]) -> dict:
    print("Fetching manifest.json …", flush=True)
    r = requests.get(MANIFEST_URL, headers=hf_headers(token), timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_first_line(path: str, token: Optional[str]) -> Optional[dict]:
    """Stream just enough bytes to get the first JSONL line."""
    url = f"{HF_BASE}/{path}"
    try:
        with requests.get(
            url,
            headers={**hf_headers(token), "Range": "bytes=0-4095"},
            stream=True,
            timeout=30,
        ) as r:
            if r.status_code not in (200, 206):
                return None
            chunk = b""
            for data in r.iter_content(chunk_size=4096):
                chunk += data
                if b"\n" in chunk:
                    break
            first_line = chunk.split(b"\n")[0].strip()
            if not first_line:
                return None
            return json.loads(first_line)
    except Exception as e:
        print(f"    [warn] could not fetch {path}: {e}", file=sys.stderr)
        return None


def parse_path(path: str) -> tuple[str, str, str, str]:
    """
    data/{benchmark}/{developer}/{model}/{uuid}.jsonl
    Returns (benchmark, developer, model, uuid).
    """
    parts = path.split("/")
    # parts: ['data', benchmark, developer, model, filename]
    if len(parts) < 5:
        return ("?", "?", "?", "?")
    benchmark  = parts[1]
    developer  = parts[2]
    model      = parts[3]
    uuid       = parts[4].rsplit(".", 1)[0]
    return benchmark, developer, model, uuid


# ── summary helpers ───────────────────────────────────────────────────────────

def score_summary(row: dict) -> str:
    """Pull the score value out of an instance-level row for display."""
    ev = row.get("evaluation", {})
    score  = ev.get("score")
    correct = ev.get("is_correct")
    parts = []
    if score is not None:
        parts.append(f"score={score}")
    if correct is not None:
        parts.append(f"is_correct={correct}")
    if not parts:
        # fallback: show all evaluation keys
        parts = [f"{k}={v}" for k, v in ev.items()]
    return ", ".join(parts) if parts else "(no evaluation field)"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-preview", action="store_true", help="skip fetching first rows")
    ap.add_argument("--benchmark",  default=None, help="filter: only show benchmarks matching this substring")
    ap.add_argument("--limit",      type=int, default=None, help="max number of JSONL files to preview")
    ap.add_argument("--hf-token",   default=None, help="HuggingFace token for private/gated repos")
    args = ap.parse_args()

    manifest = fetch_manifest(args.hf_token)
    files: dict = manifest.get("files", {})

    # ── partition into json (aggregate) vs jsonl (instance-level) ────────────
    aggregate_paths = []
    instance_paths  = []
    for path in files:
        if not path.startswith("data/"):
            continue
        if path.endswith(".jsonl"):
            instance_paths.append(path)
        elif path.endswith(".json"):
            aggregate_paths.append(path)

    print(f"\nManifest totals: {len(aggregate_paths)} aggregate (.json), {len(instance_paths)} instance-level (.jsonl)\n")

    # ── filter ────────────────────────────────────────────────────────────────
    filtered = instance_paths
    if args.benchmark:
        filtered = [p for p in filtered if args.benchmark.lower() in p.lower()]
        print(f"Filtered to {len(filtered)} JSONL files matching '{args.benchmark}'\n")

    if not filtered:
        print("No instance-level JSONL files found.")
        return

    # ── group by benchmark ────────────────────────────────────────────────────
    by_benchmark: dict[str, list[tuple]] = defaultdict(list)
    for path in sorted(filtered):
        bm, dev, model, uuid = parse_path(path)
        by_benchmark[bm].append((dev, model, uuid, path))

    # ── summary table ─────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"{'BENCHMARK':<28}  {'# JSONL files':>13}  {'MODELS (sample)'}")
    print("=" * 72)
    for bm in sorted(by_benchmark):
        entries = by_benchmark[bm]
        model_names = sorted({f"{dev}/{model}" for dev, model, _, _ in entries})
        sample = ", ".join(model_names[:3])
        if len(model_names) > 3:
            sample += f", … (+{len(model_names)-3} more)"
        print(f"  {bm:<26}  {len(entries):>13}  {sample}")
    print("=" * 72)
    print(f"\nTotal: {len(filtered)} JSONL files across {len(by_benchmark)} benchmarks\n")

    if args.no_preview:
        return

    # ── first-row previews ────────────────────────────────────────────────────
    print("Fetching first row of each JSONL file (up to --limit) …\n")
    to_preview = filtered
    if args.limit:
        to_preview = filtered[: args.limit]

    prev_bm = None
    for i, path in enumerate(to_preview, 1):
        bm, dev, model, uuid = parse_path(path)
        if bm != prev_bm:
            print(f"\n── {bm} ──────────────────────────────────────────────")
            prev_bm = bm

        label = f"{dev}/{model}"
        print(f"  [{i}/{len(to_preview)}] {label}")
        row = fetch_first_line(path, args.hf_token)
        if row is None:
            print("    (could not fetch row)")
            continue

        schema  = row.get("schema_version", "?")
        eval_id = row.get("evaluation_id", "?")
        model_id = row.get("model_id", "?")
        sample_id = row.get("sample_id", "?")
        interact = row.get("interaction_type", "?")
        score_str = score_summary(row)

        print(f"    schema:       {schema}")
        print(f"    eval_id:      {eval_id}")
        print(f"    model_id:     {model_id}")
        print(f"    sample_id:    {sample_id}")
        print(f"    interaction:  {interact}")
        print(f"    evaluation:   {score_str}")

        # show any extra top-level keys beyond the standard ones
        standard_keys = {"schema_version", "evaluation_id", "model_id", "sample_id",
                         "interaction_type", "input", "output", "evaluation"}
        extra = {k: v for k, v in row.items() if k not in standard_keys}
        if extra:
            print(f"    extra keys:   {list(extra.keys())}")

    print()


if __name__ == "__main__":
    main()
