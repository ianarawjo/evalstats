"""Generate a synthetic code-review-agent dataset for the judge alignment /
PPI correction demo (see code_review_judge_alignment_demo.py).

Story: 10 code-review agent models (named by family + parameter
size, e.g. "llama-2-70b") each review the same 120 pull-request diffs. An
LLM judge scores review quality 1-5. The judge has two systematic biases:

  1. Self-preference — the judge is the same model as one of the ten
      (`gemma-2-9b`), and rates its own outputs higher regardless of true
     quality.
  2. Length bias — the judge inherently rewards longer reviews. Three of
      the ten models (`llama-3.1-8b`, `yi-34b`, `gemini-1.5-pro`) happen to
     write much longer reviews than the rest, independent of whether those
     reviews are actually better; the bias is a property of the judge, not
     a special rule applied only to them.

The two genuinely best models (`falcon-40b` and `llama-2-70b`) are neither
the judge's self-model nor unusually verbose, so the judge under-rates them
relative to the (biased) leaders. `claude-3-opus`, the largest model, is
also strong and unbiased -- included so "biggest = best" isn't a safe
shortcut either.

A random 40 of the 120 items are chosen as the human-labeled "gold set",
and human labels are populated for ALL 10 models on those same 40 items
(equal count per model), matching evalstats' paired-by-item design.

Usage:
    python examples/generate_code_review_alignment_demo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).parent / "code_review_judge_alignment_demo.csv"

rng = np.random.default_rng(40)

N_ITEMS = 200
N_LABELED = 30

# name              size_b  true_quality   bias
#   true_quality: what a careful human reviewer would score this model's
#   reviews, on average, on a 1-5 scale. Loosely tracks parameter size but
#   isn't strictly monotonic in it -- training/tuning matters as much as
#   raw size, which is why the biggest model (claude-3-opus) isn't the
#   best, and a mid-size model (falcon-40b) is.
#   bias: "self" = this is the judge's own model (self-preference bonus);
#         "verbose" = writes much longer reviews (judge's length bias
#         rewards this regardless of quality); None = no bias.
MODELS = [
    ("gemma-2-9b",         9,   2.65, "self"),
    ("llama-3.1-8b",       8,   2.75, "verbose"),
    ("llama-2-13b",       13,   2.95, None),
    ("gpt-oss-20b",      20,   3.10, None),
    ("yi-34b",            34,   3.30, "verbose"),
    ("qwen2.5-72b",       72,   3.35, None),
    ("claude-3-opus",    110,   3.75, None),
    ("gemini-1.5-pro",    90,   3.55, "verbose"),
    ("llama-2-70b",       70,   4.00, None),   # true co-best (close rival)
    ("falcon-40b",        40,   3.95, None),   # true best
]

SELF_BONUS = 1.1           # judge's flat bonus for grading its own outputs
LENGTH_BIAS_COEF = 0.45    # judge score reward per +100 words
VERBOSE_LENGTH_MEAN = 300
NORMAL_LENGTH_MEAN = 145
LENGTH_SD = 25
ITEM_EFFECT_SD = 0.30      # shared per-item difficulty shock (paired design)
MODEL_NOISE_SD = 0.45      # per (model, item) idiosyncratic true-quality noise
HUMAN_NOISE_SD = 0.35      # human rater noise around true quality
JUDGE_NOISE_SD = 0.35      # judge noise around its (biased) raw score


def generate() -> pd.DataFrame:
    items = [f"item_{i + 1}" for i in range(N_ITEMS)]
    item_effect = {it: rng.normal(0, ITEM_EFFECT_SD) for it in items}

    rows = []
    for name, _size_b, true_q, bias in MODELS:
        is_self = bias == "self"
        is_verbose = bias == "verbose"
        length_mean = VERBOSE_LENGTH_MEAN if is_verbose else NORMAL_LENGTH_MEAN
        for it in items:
            true_score = true_q + item_effect[it] + rng.normal(0, MODEL_NOISE_SD)
            length = max(30, rng.normal(length_mean, LENGTH_SD))

            judge_raw = (
                true_score
                + (SELF_BONUS if is_self else 0.0)
                + LENGTH_BIAS_COEF * (length - NORMAL_LENGTH_MEAN) / 100.0
                + rng.normal(0, JUDGE_NOISE_SD)
            )
            judge_score = int(np.clip(round(judge_raw), 1, 5))

            rows.append({
                "model": name,
                "item": it,
                "review_score": judge_score,
                "response_length": int(round(length)),
                "_true_score": true_score,
            })

    df = pd.DataFrame(rows)

    # Human labels: the SAME 40 items are labeled across all 10 models, so
    # every model gets exactly N_LABELED labeled rows.
    labeled_items = set(rng.choice(items, size=N_LABELED, replace=False))
    df["human_score"] = np.nan
    mask = df["item"].isin(labeled_items)
    human_raw = df.loc[mask, "_true_score"] + rng.normal(0, HUMAN_NOISE_SD, size=mask.sum())
    df.loc[mask, "human_score"] = np.clip(np.round(human_raw), 1, 5)

    return df.drop(columns=["_true_score"])


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows ({df['human_score'].notna().sum()} labeled) to {OUT_PATH}")

    # Presenter-only ground truth -- not for the audience. Useful for
    # sanity-checking the reveal (or re-tuning the constants above) before
    # the workshop.
    print("\n=== Ground truth (presenter reference only) ===")
    truth = pd.DataFrame(MODELS, columns=["model", "size_b", "true_quality", "bias"])
    truth = truth.sort_values("true_quality", ascending=False)
    print(truth.to_string(index=False))
