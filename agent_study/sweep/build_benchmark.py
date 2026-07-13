"""Builds the full sweep into a single, portable JSON manifest.

    .venv/bin/python -m agent_study.sweep.build_benchmark --dry-run-cells 5
    .venv/bin/python -m agent_study.sweep.build_benchmark

Makes no network/API calls -- this is pure local data generation (see
generate.py), safe to (re)run freely. The grid: k(4) x ground_truth_regime(3)
x n(4) x [data_type x dgp_regime restricted per data type](9) = 432 cells x
n_reps(5) = 2,160 items. task_type (prompts/models) is NOT crossed into the
grid -- it's assigned per cell as a balanced random covariate instead (216
cells each, see assign_task_types below), since which word we use doesn't
change any analytical property of the design; fully crossing it would just
double every other cell for no added coverage. Each item embeds its own
data (a synthetic draw), prompt text, and ground truth inline, so a
different harness can run this benchmark without importing any of our
Python simulation code -- see sweep/README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from agent_study.sweep.dgp import DGP_REGIMES_BY_DATA_TYPE, get_dgp_config, sample_baseline_shift, sample_icc_per_arm
from agent_study.sweep.generate import DEFAULT_ICC, compute_cell_ground_truth, generate_rep_data
from agent_study.sweep.prompts import build_prompt

TASK_TYPES = ["prompts", "models"]
K_VALUES = [2, 5, 10, 20]
GROUND_TRUTH_REGIMES = ["true_null", "borderline", "clear_winner"]
N_VALUES = [15, 30, 50, 200]
N_REPS = 5
SEED_BASE = 1000
TASK_TYPE_ASSIGNMENT_SEED = SEED_BASE - 1  # separate from any cell's own seed block

OUT_PATH = Path(__file__).parent / "benchmark_v1.json"


def iter_cells():
    """Yields every (k, data_type, dgp_regime, ground_truth_regime, n) cell
    in the grid, in a fixed order so seeds are deterministic run to run.
    task_type is not part of this cross -- see assign_task_types."""
    for k in K_VALUES:
        for data_type, regimes in DGP_REGIMES_BY_DATA_TYPE.items():
            for dgp_regime in regimes:
                for ground_truth_regime in GROUND_TRUTH_REGIMES:
                    for n in N_VALUES:
                        yield k, data_type, dgp_regime, ground_truth_regime, n


def assign_task_types(n_cells: int) -> list[str]:
    """A fixed, deterministically-shuffled list with exactly n_cells//2 of
    each task_type (one extra "prompts" if n_cells is odd) -- guaranteed
    50/50 balance, unlike an independent per-cell coin flip which would only
    average out to 50/50 rather than hit it exactly. Which label a cell gets
    doesn't depend on that cell's own properties, so this is safe to shuffle
    independently of the cell-iteration order."""
    half = n_cells // 2
    labels = ["prompts"] * (n_cells - half) + ["models"] * half
    np.random.default_rng(TASK_TYPE_ASSIGNMENT_SEED).shuffle(labels)
    return labels


def build(out_path: Path, n_reps: int, limit_cells: int | None = None) -> dict:
    cells = list(iter_cells())
    if limit_cells is not None:
        cells = cells[:limit_cells]
    task_types = assign_task_types(len(cells))
    print(f"{len(cells)} cells x {n_reps} reps = {len(cells) * n_reps} items")

    items = []
    t0 = time.time()
    for cell_idx, (k, data_type, dgp_regime, ground_truth_regime, n) in enumerate(cells):
        task_type = task_types[cell_idx]
        # Per-cell RNG (offset 0 in this cell's seed block; reps use 1..n_reps
        # below, so there's no overlap) decides which arm wins, where the
        # whole comparison sits on the scale, and (for icc_spread regimes)
        # each arm's own reliability -- all shared by every rep of this cell,
        # since they're repeated draws of the same design, not different
        # designs. See generate.py's module docstring for why none of these
        # should be fixed the way they used to be.
        cell_rng = np.random.default_rng(SEED_BASE + cell_idx * 100)
        winner_idx = int(cell_rng.integers(0, k))
        baseline_shift = sample_baseline_shift(data_type, cell_rng)
        cfg = get_dgp_config(data_type, dgp_regime)
        icc = sample_icc_per_arm(k, cell_rng) if cfg.icc_spread else DEFAULT_ICC

        gt = compute_cell_ground_truth(
            task_type, k, data_type, ground_truth_regime, n, dgp_regime, winner_idx, baseline_shift, icc,
        )
        prompt = build_prompt(task_type, k, data_type, n)
        cell_id = f"{task_type}__k{k}__{data_type}__{dgp_regime}__{ground_truth_regime}__n{n}"

        for rep in range(1, n_reps + 1):
            seed = SEED_BASE + cell_idx * 100 + rep  # deterministic, unique per (cell, rep); reps < 100/cell
            df = generate_rep_data(
                task_type, k, data_type, n, dgp_regime, gt.cohens_d, winner_idx, baseline_shift, icc, seed=seed,
            )
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
            "task_type_assignment": "balanced_random_covariate",  # not fully crossed -- see assign_task_types
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
