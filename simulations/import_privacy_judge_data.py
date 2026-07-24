#!/usr/bin/env python3
"""import_privacy_judge_data.py -- one-off converter that turns the
downloaded sjmeis/privacy-judge replication data (a local checkout, e.g.
~/Documents/privacy-judge/data/) into the judge_bias_<dataset>.csv /
judge_bias_<dataset>_items.csv format simulations/harness/scenarios/
real_judge_bias.py's load_real_judge_bias_corpus expects (the same schema
simulations/collect_judge_bias_data.py writes for arena/wmt_da/appstore --
see that file's module docstring for the exact column meanings).

Registers as dataset key "privacy_judge" (eval_type "continuous", native
scale (1.0, 5.0) -- treated as continuous rather than "likert" since
human_label here is the MEAN of 50-68 individual survey participants'
1-5 ratings per item, not a single discrete label).

Source files consumed (from --privacy-judge-dir):
  llm_improved_combined.csv  13 judge models x 250 items, 1-5 Likert
                              privacy-risk scores using the paper's
                              "improved" prompt (the "simple" prompt variant
                              is deliberately NOT imported -- see this
                              script's module docstring discussion; we
                              already have plenty of poor-calibration judge
                              data, so only one prompt variant is used to
                              avoid conflating "different model" with
                              "different prompt" in the judge_model
                              namespace).
  survey_results.csv         677 human participants x 250 items (sparse --
                              each item rated by ~50-68 of the 677 people),
                              plus demographic columns (continent, age,
                              gender, education) that are ignored here.
  texts.json                 item index -> raw text, used only for the
                              items CSV's judge_input column (not read by
                              load_real_judge_bias_corpus itself, but kept
                              for parity with collect_judge_bias_data.py's
                              schema and any future re-prompting use).

human_label per item = mean of every participant's non-missing rating for
that item (a noisy estimate, not a single ground truth -- unlike arena/
wmt_da/appstore, which already have one authoritative human label per
item). Averaging is an accepted simplification here since the raw
per-participant ratings remain available in survey_results.csv for anyone
who wants finer-grained analysis later.

A judge model's score for an item is skipped entirely (no row emitted) if
llm_improved_combined.csv has it blank (103/3250 cells, affecting 77/250
items) -- load_real_judge_bias_corpus's alignment already drops any item
not scored by every requested judge_model, so this naturally shrinks the
usable corpus to the ~173/250 items all 13 judges scored (or more, if a
caller later restricts --judge-models to a subset with fuller coverage).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ITEMS_FIELDNAMES = ["item_id", "eval_type", "dataset", "human_label", "judge_input"]
MERGED_FIELDNAMES = ["item_id", "eval_type", "dataset", "human_label", "judge_model", "run_idx", "judge_score"]

DATASET_KEY = "privacy_judge"
EVAL_TYPE = "continuous"
N_ITEMS = 250


def _item_id(idx: int) -> str:
    return f"{DATASET_KEY}_{idx}"


def _load_texts(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _load_human_labels(path: Path) -> dict[int, float]:
    """Mean of every participant's non-missing rating, per item index."""
    sums = [0.0] * N_ITEMS
    counts = [0] * N_ITEMS
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for idx in range(N_ITEMS):
                v = row.get(str(idx), "")
                if v is None or v.strip() == "":
                    continue
                sums[idx] += float(v)
                counts[idx] += 1
    missing = [idx for idx in range(N_ITEMS) if counts[idx] == 0]
    if missing:
        raise ValueError(f"No survey ratings at all for item indices {missing} -- cannot compute human_label.")
    return {idx: sums[idx] / counts[idx] for idx in range(N_ITEMS)}


def _load_judge_scores(path: Path) -> dict[str, dict[int, float]]:
    """judge_model -> {item_idx: score}, omitting blank cells entirely."""
    out: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header is a stray 0..250 index row, not needed (see module docstring)
        for row in reader:
            model = row[0]
            scores: dict[int, float] = {}
            for idx in range(N_ITEMS):
                v = row[idx + 1]
                if v.strip() == "":
                    continue
                scores[idx] = float(v)
            out[model] = scores
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--privacy-judge-dir", default=str(Path.home() / "Documents" / "privacy-judge" / "data"))
    ap.add_argument("--out-dir", default="simulations/out")
    ap.add_argument("--llm-file", default="llm_improved_combined.csv",
                     help="Which combined LLM-scores file to import (default: improved-prompt variant).")
    args = ap.parse_args()

    src = Path(args.privacy_judge_dir)
    texts = _load_texts(src / "texts.json")
    human_labels = _load_human_labels(src / "survey_results.csv")
    judge_scores = _load_judge_scores(src / args.llm_file)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / f"judge_bias_{DATASET_KEY}_items.csv"
    merged_path = out_dir / f"judge_bias_{DATASET_KEY}.csv"

    with items_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ITEMS_FIELDNAMES)
        w.writeheader()
        for idx in range(N_ITEMS):
            w.writerow({
                "item_id": _item_id(idx), "eval_type": EVAL_TYPE, "dataset": DATASET_KEY,
                "human_label": human_labels[idx], "judge_input": texts.get(idx, ""),
            })

    n_rows = 0
    with merged_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MERGED_FIELDNAMES)
        w.writeheader()
        for model, scores in judge_scores.items():
            for idx, score in scores.items():
                w.writerow({
                    "item_id": _item_id(idx), "eval_type": EVAL_TYPE, "dataset": DATASET_KEY,
                    "human_label": human_labels[idx], "judge_model": model, "run_idx": 0,
                    "judge_score": score,
                })
                n_rows += 1

    n_complete = sum(
        1 for idx in range(N_ITEMS) if all(idx in judge_scores[m] for m in judge_scores)
    )
    print(f"Wrote {items_path} ({N_ITEMS} items) and {merged_path} ({n_rows} rows, "
          f"{len(judge_scores)} judge models).")
    print(f"  {n_complete}/{N_ITEMS} items have scores from every judge model "
          f"(the rest will be dropped by alignment unless --judge-models restricts to a fuller-coverage subset).")


if __name__ == "__main__":
    main()
