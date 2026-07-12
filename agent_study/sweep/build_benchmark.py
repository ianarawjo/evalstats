"""Builds the full sweep into a single, portable JSON manifest.

    .venv/bin/python -m agent_study.sweep.build_benchmark --dry-run-cells 5
    .venv/bin/python -m agent_study.sweep.build_benchmark

Makes no network/API calls -- this is pure local data generation (see
generate.py), safe to (re)run freely. The grid: task_type(2) x k(4) x
ground_truth_regime(3) x n(4) x [data_type x dgp_regime restricted per data
type](9) = 864 cells x n_reps(5) = 4,320 items. Each item embeds its own
data (a synthetic draw), prompt text, and ground truth inline, so a
different harness can run this benchmark without importing any of our
Python simulation code -- see sweep/README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent_study.sweep.dgp import DGP_REGIMES_BY_DATA_TYPE
from agent_study.sweep.generate import compute_cell_ground_truth, generate_rep_data
from agent_study.sweep.prompts import build_prompt

TASK_TYPES = ["prompts", "models"]
K_VALUES = [2, 5, 10, 20]
GROUND_TRUTH_REGIMES = ["true_null", "borderline", "clear_winner"]
N_VALUES = [15, 30, 50, 200]
N_REPS = 5
SEED_BASE = 1000

OUT_PATH = Path(__file__).parent / "benchmark_v1.json"


def iter_cells():
    """Yields every (task_type, k, data_type, dgp_regime, ground_truth_regime, n)
    cell in the grid, in a fixed order so seeds are deterministic run to run."""
    for task_type in TASK_TYPES:
        for k in K_VALUES:
            for data_type, regimes in DGP_REGIMES_BY_DATA_TYPE.items():
                for dgp_regime in regimes:
                    for ground_truth_regime in GROUND_TRUTH_REGIMES:
                        for n in N_VALUES:
                            yield task_type, k, data_type, dgp_regime, ground_truth_regime, n


def build(out_path: Path, n_reps: int, limit_cells: int | None = None) -> dict:
    cells = list(iter_cells())
    if limit_cells is not None:
        cells = cells[:limit_cells]
    print(f"{len(cells)} cells x {n_reps} reps = {len(cells) * n_reps} items")

    items = []
    t0 = time.time()
    for cell_idx, (task_type, k, data_type, dgp_regime, ground_truth_regime, n) in enumerate(cells):
        gt = compute_cell_ground_truth(task_type, k, data_type, ground_truth_regime, n, dgp_regime)
        prompt = build_prompt(task_type, k, data_type, n)
        cell_id = f"{task_type}__k{k}__{data_type}__{dgp_regime}__{ground_truth_regime}__n{n}"

        for rep in range(1, n_reps + 1):
            seed = SEED_BASE + cell_idx * 100 + rep  # deterministic, unique per (cell, rep); reps < 100/cell
            df = generate_rep_data(task_type, k, data_type, n, dgp_regime, gt.cohens_d, seed=seed)
            items.append({
                "item_id": f"{cell_id}__rep{rep}",
                "task_type": task_type,
                "k": k,
                "data_type": data_type,
                "dgp_regime": dgp_regime,
                "ground_truth_regime": ground_truth_regime,
                "n": n,
                "rep": rep,
                "seed": seed,
                "prompt": prompt,
                "data": {"columns": list(df.columns), "rows": df.values.tolist()},
                "ground_truth": gt.as_dict(),
            })

        if (cell_idx + 1) % 25 == 0 or cell_idx == len(cells) - 1:
            elapsed = time.time() - t0
            rate = (cell_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(cells) - cell_idx - 1) / rate if rate > 0 else float("nan")
            print(f"  {cell_idx + 1}/{len(cells)} cells done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    manifest = {
        "benchmark_version": "sweep_v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "axes": {
            "task_type": TASK_TYPES,
            "k": K_VALUES,
            "data_type": list(DGP_REGIMES_BY_DATA_TYPE.keys()),
            "dgp_regime_by_data_type": DGP_REGIMES_BY_DATA_TYPE,
            "ground_truth_regime": GROUND_TRUTH_REGIMES,
            "n": N_VALUES,
            "n_reps": n_reps,
        },
        "n_cells": len(cells),
        "n_items": len(items),
        "items": items,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest))
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {len(items)} items ({len(cells)} cells) to {out_path} ({size_mb:.1f} MB)")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--reps", type=int, default=N_REPS)
    parser.add_argument(
        "--dry-run-cells", type=int, default=None, metavar="N",
        help="Only build the first N cells (for a quick sanity-check build before committing to the full grid).",
    )
    args = parser.parse_args()
    build(Path(args.out), n_reps=args.reps, limit_cells=args.dry_run_cells)
