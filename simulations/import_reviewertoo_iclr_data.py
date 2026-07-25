#!/usr/bin/env python3
"""import_reviewertoo_iclr_data.py -- one-off converter that turns the
demfier/reviewertoo-iclr2026-reviews HF dataset's `papers` table (a local
parquet download, since the repo is gated -- see this script's usage note)
into the judge_bias_<dataset>.csv / judge_bias_<dataset>_items.csv format
simulations/harness/scenarios/real_judge_bias.py's
load_real_judge_bias_corpus expects, same schema
simulations/collect_judge_bias_data.py writes for arena/wmt_da/appstore.

Registers as dataset key "iclr_metareview" (eval_type "binary", native
scale (0.0, 1.0)) -- a second, independent "binary" dataset alongside
arena (REAL_JUDGE_BIAS_DATASETS keys are free-standing strings, not
one-per-eval_type, unlike collect_judge_bias_data.py's fetcher-side
DATASET_SPECS registry).

human_label  = ground_truth_binary  (the real conference accept/reject
               decision) -> 1.0 if "accept", 0.0 if "reject".
judge_score  = metareviewer_decision_binary  (ReviewerToo's LLM-generated
               metareview decision, same accept/reject mapping) -- the
               dataset's own reported metareviewer accuracy is 55.5%
               across 18,322 evaluable papers, i.e. this judge is
               deliberately noisy/weakly-calibrated real data, not a
               strong judge.

Only ONE "judge" exists in this data (ReviewerToo's single metareviewer
pipeline, not several independently-collected judge models like arena/
wmt_da/appstore/privacy_judge) -- judge_model is hardcoded to
"reviewertoo_metareviewer". This means ppi_real's twogroup/paired/omnibus
checks (which need >=2 or >=3 judge models) produce ZERO cells for this
dataset -- all_judge_pairs/all_judge_triples naturally return [] for a
1-judge corpus, so this dataset only ever feeds the single-sample
bias/coverage check. That's an accepted, structural limitation of the
source data (not a bug to fix here), same as how wmt_da_paired is the
only dataset that feeds the within-item paired bias/coverage check.

A random 4000-paper sample (fixed seed, `--n-sample`) is drawn from the
18,322 rows with non-blank ground_truth_binary/metareviewer_decision_binary
(88 of 18,410 rows have neither and are dropped) -- per-user request, to
avoid checking in a dataset anywhere near the full 75MB parquet's size
into the repo (which also carries abstract/metareview free text this
converter never needs).

Usage:
  # 1. Request access to the gated HF dataset, log in, then download the
  #    papers table only (NOT persona_reviews, NOT the full_pipeline
  #    archive):
  #      huggingface-cli download demfier/reviewertoo-iclr2026-reviews \\
  #          papers.parquet --repo-type dataset --local-dir <dir>
  # 2. python simulations/import_reviewertoo_iclr_data.py \\
  #        --parquet-path <dir>/papers.parquet
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

ITEMS_FIELDNAMES = ["item_id", "eval_type", "dataset", "human_label", "judge_input"]
MERGED_FIELDNAMES = ["item_id", "eval_type", "dataset", "human_label", "judge_model", "run_idx", "judge_score"]

DATASET_KEY = "iclr_metareview"
EVAL_TYPE = "binary"
JUDGE_MODEL = "reviewertoo_metareviewer"
_BINARY_MAP = {"accept": 1.0, "reject": 0.0}


def _item_id(paper_id: str) -> str:
    return f"{DATASET_KEY}_{paper_id}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-path", default="simulations/out/ICML2026-ReviewerToo-judge-dataset-papers.parquet")
    ap.add_argument("--out-dir", default="simulations/out")
    ap.add_argument("--n-sample", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet_path, columns=[
        "paper_id", "ground_truth_binary", "metareviewer_decision_binary",
    ])
    df = df[df["ground_truth_binary"].isin(_BINARY_MAP) & df["metareviewer_decision_binary"].isin(_BINARY_MAP)]
    n_evaluable = len(df)
    if args.n_sample < n_evaluable:
        df = df.sample(n=args.n_sample, random_state=args.seed)
    df = df.sort_values("paper_id")  # deterministic row order regardless of sample() internals

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / f"judge_bias_{DATASET_KEY}_items.csv"
    merged_path = out_dir / f"judge_bias_{DATASET_KEY}.csv"

    with items_path.open("w", newline="", encoding="utf-8") as f_items, \
         merged_path.open("w", newline="", encoding="utf-8") as f_merged:
        w_items = csv.DictWriter(f_items, fieldnames=ITEMS_FIELDNAMES)
        w_merged = csv.DictWriter(f_merged, fieldnames=MERGED_FIELDNAMES)
        w_items.writeheader()
        w_merged.writeheader()
        for row in df.itertuples(index=False):
            item_id = _item_id(row.paper_id)
            human_label = _BINARY_MAP[row.ground_truth_binary]
            judge_score = _BINARY_MAP[row.metareviewer_decision_binary]
            w_items.writerow({
                "item_id": item_id, "eval_type": EVAL_TYPE, "dataset": DATASET_KEY,
                "human_label": human_label, "judge_input": "",
            })
            w_merged.writerow({
                "item_id": item_id, "eval_type": EVAL_TYPE, "dataset": DATASET_KEY,
                "human_label": human_label, "judge_model": JUDGE_MODEL, "run_idx": 0,
                "judge_score": judge_score,
            })

    print(f"Wrote {items_path} and {merged_path}: {len(df)} papers sampled from {n_evaluable} evaluable "
          f"(of {len(pd.read_parquet(args.parquet_path, columns=['paper_id']))} total).")
    print(f"  Single judge_model ({JUDGE_MODEL!r}) -- only the single-sample bias/coverage check applies; "
          f"twogroup/paired/omnibus checks need >=2/3 judge models and will produce zero cells for this dataset.")


if __name__ == "__main__":
    main()
