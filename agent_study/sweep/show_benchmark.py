"""Prints a human-readable preview of benchmark_v1.json to the terminal --
manifest-level stats, a couple of items in full detail (so the exact JSON
shape is visible), and compact summaries of a diverse slice of items
spanning each grid axis one at a time. Read-only; no API calls.

    .venv/bin/python -m agent_study.sweep.show_benchmark
    .venv/bin/python -m agent_study.sweep.show_benchmark --item-id models__k5__continuous__clean__clear_winner__n50__rep1
    .venv/bin/python -m agent_study.sweep.show_benchmark --full 4 --no-slices
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "benchmark_v1.json"

# Fixed "anchor" params held constant while a slice below varies exactly one
# axis, so each slice isolates what that one axis actually changes. task_type
# is deliberately absent: it's a balanced-random covariate assigned per cell
# (build_benchmark.py's assign_task_types), not a freely choosable axis, so
# an anchor cell only ever exists under whichever task_type it was actually
# given -- see cell_key/find_item below.
ANCHOR = dict(k=5, data_type="continuous", dgp_regime="clean",
               ground_truth_regime="clear_winner", n=50, rep=1)


def cell_key(k, data_type, dgp_regime, ground_truth_regime, n, rep) -> tuple:
    """Key ignoring task_type, since it's not a free choice per cell anymore
    -- looking an item up by its other 5 params plus rep is unambiguous
    (exactly one task_type was assigned to that cell)."""
    return (k, data_type, dgp_regime, ground_truth_regime, n, rep)


def build_slices(axes: dict) -> list[tuple[str, list[tuple]]]:
    """Each slice varies exactly one axis off ANCHOR; (label, [cell_keys])."""
    slices = []

    # DGP regime x data type (the axis this sweep exists to add).
    keys = []
    for data_type, regimes in axes["dgp_regime_by_data_type"].items():
        for dgp_regime in regimes:
            a = {**ANCHOR, "data_type": data_type, "dgp_regime": dgp_regime}
            keys.append(cell_key(**a))
    slices.append(("DGP regime x data type (rest held at anchor)", keys))

    # k (entities compared).
    keys = [cell_key(**{**ANCHOR, "k": k}) for k in axes["k"]]
    slices.append(("k / entities compared", keys))

    # Ground truth regime (effect size).
    keys = [cell_key(**{**ANCHOR, "ground_truth_regime": r}) for r in axes["ground_truth_regime"]]
    slices.append(("ground_truth_regime / effect size", keys))

    # Sample size.
    keys = [cell_key(**{**ANCHOR, "n": n}) for n in axes["n"]]
    slices.append(("n / sample size", keys))

    return slices


def print_task_type_balance(items: list[dict]) -> None:
    """task_type is a balanced random covariate now (not a crossed axis), so
    there's no single-cell slice that shows both values -- instead, confirm
    the balance holds manifest-wide."""
    counts = Counter(it["task_type"] for it in items)
    print(f"--- task_type balance (balanced random covariate, not a crossed axis) ---")
    print(f"  {dict(counts)}")
    print()


def summarize_data(data: dict) -> str:
    cols = data["columns"]
    rows = data["rows"]
    ei, si = cols.index("entity"), cols.index("score")
    by_entity: dict[str, list[float]] = {}
    for r in rows:
        by_entity.setdefault(r[ei], []).append(r[si])
    means = {k: round(statistics.mean(v), 3) for k, v in by_entity.items()}
    shown = dict(list(means.items())[:6])
    suffix = f", ... (+{len(means) - 6} more)" if len(means) > 6 else ""
    return f"{len(rows)} rows, {len(means)} entities. Sample means: {shown}{suffix}"


def print_full_item(it: dict) -> None:
    print("=" * 100)
    print(f"item_id: {it['item_id']}")
    print(f"task_type={it['task_type']}  k={it['k']}  data_type={it['data_type']}  "
          f"dgp_regime={it['dgp_regime']}  ground_truth_regime={it['ground_truth_regime']}  "
          f"n={it['n']}  rep={it['rep']}  seed={it['seed']}")
    print("-" * 100)
    print("PROMPT:")
    print(it["prompt"])
    print("-" * 100)
    cols = it["data"]["columns"]
    rows = it["data"]["rows"]
    print(f"DATA ({len(rows)} rows total, columns={cols}), first 8 rows:")
    for r in rows[:8]:
        print(f"  {r}")
    print(f"  ... ({summarize_data(it['data'])})")
    print("-" * 100)
    print("GROUND TRUTH (harness-only -- never shown to an agent):")
    print(json.dumps(it["ground_truth"], indent=2))
    print("=" * 100)
    print()


def print_summary_item(it: dict) -> None:
    gt = it["ground_truth"]
    print(f"  {it['item_id']}")
    print(f"    power={gt['power']:<7} true_best={str(gt['true_best']):<10} "
          f"correct_decisions={gt['correct_decisions']}")
    print(f"    {summarize_data(it['data'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--full", type=int, default=2, metavar="N", help="How many items to show in full detail.")
    parser.add_argument("--item-id", default=None, help="Show one specific item in full detail and exit.")
    parser.add_argument("--no-slices", action="store_true", help="Skip the per-axis summary slices.")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run `python -m agent_study.sweep.build_benchmark` first "
            "(it's gitignored, a local build artifact regenerated from seeds)."
        )
    manifest = json.loads(path.read_text())
    items_by_id = {it["item_id"]: it for it in manifest["items"]}
    items_by_cell_key = {
        cell_key(it["k"], it["data_type"], it["dgp_regime"], it["ground_truth_regime"], it["n"], it["rep"]): it
        for it in manifest["items"]
    }

    print(f"benchmark_version: {manifest['benchmark_version']}  generated_at: {manifest['generated_at']}")
    print(f"n_cells={manifest['n_cells']}  n_items={manifest['n_items']}  file={path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")
    print(f"axes: {json.dumps(manifest['axes'])}")
    print()

    if args.item_id:
        it = items_by_id.get(args.item_id)
        if it is None:
            raise SystemExit(f"item_id {args.item_id!r} not found in manifest.")
        print_full_item(it)
        return

    print(f"### {args.full} item(s) in full detail ###\n")
    for it in manifest["items"][: args.full]:
        print_full_item(it)

    if args.no_slices:
        return

    print("### Representative slices across the grid (one axis varies at a time; compact summaries) ###\n")
    print(f"Anchor (held fixed except where noted): {ANCHOR}\n")
    for label, keys in build_slices(manifest["axes"]):
        print(f"--- {label} ---")
        for key in keys:
            it = items_by_cell_key.get(key)
            if it is None:
                print(f"  {key} -> NOT FOUND")
                continue
            print_summary_item(it)
        print()

    print_task_type_balance(manifest["items"])


if __name__ == "__main__":
    main()
