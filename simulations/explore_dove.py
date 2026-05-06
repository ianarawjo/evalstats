#!/usr/bin/env python3
"""
explore_dove.py — Explore model × benchmark coverage in DOVE_Lite.

Lists every (model, benchmark) combination available in the dataset and
renders the result as a compact table without downloading any parquet data.

Dataset: https://huggingface.co/datasets/nlphuji/DOVE_Lite
Auth:    huggingface-cli login  OR  --hf-token hf_xxx
         (DOVE_Lite requires HuggingFace approval to access)

Usage
  python simulations/explore_dove.py                   # grouped matrix (default)
  python simulations/explore_dove.py --mode list       # one row per model
  python simulations/explore_dove.py --mode full       # full model × benchmark matrix
  python simulations/explore_dove.py --lang en         # filter language (default: en)
  python simulations/explore_dove.py --shots 0         # filter shot count
  python simulations/explore_dove.py --filter mmlu     # keep benchmarks matching substring
  python simulations/explore_dove.py --hf-token hf_xxx
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Optional

# ── benchmark family groupings ────────────────────────────────────────────────
# Maps a family label to a substring that matches the relevant parquet filenames.
BENCHMARK_FAMILIES: dict[str, str] = {
    "mmlu":        "mmlu.",           # mmlu.<subject>
    "mmlu_pro":    "mmlu_pro.",       # mmlu_pro.<subject>
    "arc":         "ai2_arc.arc_",    # ai2_arc.arc_challenge / arc_easy
    "hellaswag":   "hellaswag",
    "openbookqa":  "openbookqa",
    "race":        "race",
    "social_iqa":  "social_iqa",
    "other":       None,              # catch-all
}


def family_of(benchmark: str) -> str:
    for fam, substr in BENCHMARK_FAMILIES.items():
        if fam == "other":
            continue
        if substr and substr in benchmark:
            return fam
    return "other"


# ── file listing ──────────────────────────────────────────────────────────────

def list_dove_files(token: Optional[str]) -> list[str]:
    """Return all repo file paths using huggingface_hub (handles pagination)."""
    from huggingface_hub import list_repo_tree

    kwargs: dict = dict(repo_id="nlphuji/DOVE_Lite", repo_type="dataset", recursive=True)
    if token:
        kwargs["token"] = token

    print("Listing repository files (may take a moment) …", flush=True)
    paths = []
    for item in list_repo_tree(**kwargs):
        if hasattr(item, "path"):
            paths.append(item.path)
    return paths


def parse_parquet_paths(
    all_paths: list[str],
    lang_filter: Optional[str],
    shots_filter: Optional[int],
) -> dict[str, dict[str, set[int]]]:
    """
    Parse paths of the form  {model}/{lang}/{N}_shot/{benchmark}.parquet
    Returns  {model: {benchmark: {shot_counts}}}
    """
    result: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for path in all_paths:
        if not path.endswith(".parquet"):
            continue
        parts = path.split("/")
        if len(parts) != 4:
            continue
        model, lang, shot_dir, fname = parts
        if lang_filter and lang != lang_filter:
            continue
        if not shot_dir.endswith("_shot"):
            continue
        try:
            n_shots = int(shot_dir.replace("_shot", ""))
        except ValueError:
            continue
        if shots_filter is not None and n_shots != shots_filter:
            continue
        benchmark = fname[:-len(".parquet")]
        result[model][benchmark].add(n_shots)
    return result


# ── display helpers ────────────────────────────────────────────────────────────

def fmt_shots(shots: set[int]) -> str:
    return "/".join(str(s) for s in sorted(shots)) + "-shot"


def print_grouped_matrix(
    coverage: dict[str, dict[str, set[int]]],
    benchmark_filter: Optional[str],
) -> None:
    """
    Compact model × benchmark-family matrix.
    Columns = families; cell = count of benchmarks available (or blank).
    """
    all_benchmarks: set[str] = set()
    for bm_map in coverage.values():
        all_benchmarks.update(bm_map.keys())

    if benchmark_filter:
        all_benchmarks = {b for b in all_benchmarks if benchmark_filter.lower() in b.lower()}

    families = [f for f in BENCHMARK_FAMILIES if f != "other"]
    # check whether "other" family has any members
    if any(family_of(b) == "other" for b in all_benchmarks):
        families.append("other")

    models = sorted(coverage.keys())

    # count benchmarks per (model, family)
    def count(model: str, fam: str) -> int:
        return sum(1 for b in coverage[model] if family_of(b) == fam and b in all_benchmarks)

    # column widths
    model_w = max(len(m) for m in models)
    col_w = max(max(len(f) for f in families), 5)
    shots_w = 12

    header = f"  {'MODEL':<{model_w}}  {'SHOTS':<{shots_w}}" + "".join(f"  {f:^{col_w}}" for f in families)
    print()
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for model in models:
        bm_map = coverage[model]
        all_shots: set[int] = set()
        for shots in bm_map.values():
            all_shots.update(shots)
        shots_str = fmt_shots(all_shots) if all_shots else "?"
        row = f"  {model:<{model_w}}  {shots_str:<{shots_w}}"
        for fam in families:
            n = count(model, fam)
            cell = str(n) if n else "-"
            row += f"  {cell:^{col_w}}"
        print(row)
    print("=" * len(header))
    print(f"\nCell values = number of parquet files available for that benchmark family.\n")
    print(f"Models: {len(models)}   Unique benchmarks: {len(all_benchmarks)}")
    print(f"Families: {', '.join(families)}")


def print_list_mode(
    coverage: dict[str, dict[str, set[int]]],
    benchmark_filter: Optional[str],
) -> None:
    """One row per model; lists available benchmark families and counts."""
    models = sorted(coverage.keys())
    print()
    print(f"{'MODEL':<40}  {'SHOTS':<14}  BENCHMARKS")
    print("-" * 100)
    for model in models:
        bm_map = coverage[model]
        bms = set(bm_map.keys())
        if benchmark_filter:
            bms = {b for b in bms if benchmark_filter.lower() in b.lower()}
        all_shots: set[int] = set()
        for b in bms:
            all_shots.update(bm_map[b])
        shots_str = fmt_shots(all_shots) if all_shots else "?"

        fam_buckets: dict[str, list[str]] = defaultdict(list)
        for b in sorted(bms):
            fam_buckets[family_of(b)].append(b)
        parts = []
        for fam, members in sorted(fam_buckets.items()):
            if fam == "other":
                parts.append(f"other: {', '.join(members)}")
            else:
                parts.append(f"{fam}({len(members)})")
        fam_str = "  ".join(parts)
        print(f"  {model:<38}  {shots_str:<14}  {fam_str}")
    print()
    print(f"Total models: {len(models)}")


def print_full_matrix(
    coverage: dict[str, dict[str, set[int]]],
    benchmark_filter: Optional[str],
) -> None:
    """Full model × benchmark matrix with ✓ marks (may be very wide)."""
    all_benchmarks = sorted(
        b for bm_map in coverage.values() for b in bm_map
        if not benchmark_filter or benchmark_filter.lower() in b.lower()
    )
    if not all_benchmarks:
        print("No benchmarks match the filter.")
        return
    models = sorted(coverage.keys())

    model_w = max(len(m) for m in models)
    bm_w = max(len(b) for b in all_benchmarks)

    # header: model column then one column per benchmark
    # transpose layout: rows=benchmarks, cols=models (easier to read in terminal)
    col_w = max(len(m) for m in models)
    header = f"  {'BENCHMARK':<{bm_w}}" + "".join(f"  {m[:col_w]:<{col_w}}" for m in models)
    print()
    print(header[:200] + ("  …" if len(header) > 200 else ""))
    print("-" * min(len(header), 200))
    for bm in all_benchmarks:
        row = f"  {bm:<{bm_w}}"
        for model in models:
            mark = "✓" if bm in coverage[model] else " "
            row += f"  {mark:^{col_w}}"
        print(row[:200] + ("  …" if len(row) > 200 else ""))
    print()
    print(f"Models: {len(models)}   Benchmarks shown: {len(all_benchmarks)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode", choices=["grouped", "list", "full"], default="grouped",
        help="display mode: grouped matrix (default), per-model list, or full matrix",
    )
    ap.add_argument("--lang",    default="en",   help="language code to include (default: en)")
    ap.add_argument("--shots",   type=int, default=None, help="filter to a specific shot count")
    ap.add_argument("--filter",  default=None,   help="keep only benchmarks matching this substring")
    ap.add_argument("--hf-token", default=None,  help="HuggingFace token (or set HF_TOKEN env var)")
    args = ap.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")

    try:
        all_paths = list_dove_files(token)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower() or "access" in msg.lower():
            print(
                "\nAccess denied. DOVE_Lite requires HuggingFace approval.\n"
                "  1. Request access at https://huggingface.co/datasets/nlphuji/DOVE_Lite\n"
                "  2. Run: huggingface-cli login\n"
                "     Or pass: --hf-token hf_xxx\n",
                file=sys.stderr,
            )
        else:
            print(f"Error listing repository: {e}", file=sys.stderr)
        sys.exit(1)

    parquet_count = sum(1 for p in all_paths if p.endswith(".parquet"))
    print(f"Found {len(all_paths)} total files, {parquet_count} parquet files.\n")

    coverage = parse_parquet_paths(all_paths, lang_filter=args.lang, shots_filter=args.shots)

    if not coverage:
        print(f"No data found for lang={args.lang!r}" + (f", shots={args.shots}" if args.shots is not None else "") + ".")
        return

    if args.mode == "grouped":
        print_grouped_matrix(coverage, benchmark_filter=args.filter)
    elif args.mode == "list":
        print_list_mode(coverage, benchmark_filter=args.filter)
    elif args.mode == "full":
        print_full_matrix(coverage, benchmark_filter=args.filter)


if __name__ == "__main__":
    main()
