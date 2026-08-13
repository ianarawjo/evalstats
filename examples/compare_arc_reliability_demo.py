"""Real-data demo: matched accuracy does not mean matched reliability.

Two real language models (evaluated via OpenRouter using Inspect AI) on ARC,
a benchmark of grade-school science exam questions, each run 5 independent
times on the same 50 questions -- a deliberately small sample, since
evalstats is built for exactly this small-N regime. Llama-3.1-8B and
Gemma-3n score close enough to be statistically tied (75.6% vs. 82.0%), so a
single-run comparison would call them interchangeable. But the repeat
structure reveals something a single run cannot: Gemma-3n's answers are
"effectively deterministic" run to run, while Llama-3.1-8B is only "mostly
stable" -- visibly noisier from one run to the next, even though its average
accuracy is in the same range. Consistent behavior on repeated calls to the
same input is itself a real dependability property, separate from raw
accuracy, and only visible once you look past a single run.

evalstats surfaces this two ways:

1. compare() includes it automatically whenever it detects a repeated `run`
   column -- no extra step, it's part of the same analysis used to compare
   models in the first place.
2. stability() is a standalone shortcut for when reliability is *all* you
   want to check -- e.g. "is this one model/config reliable enough to
   ship?" -- without running a full multi-model comparison.

Both report the same underlying numbers.

Source data: simulations/out/inspect_benchmarks.csv (committed in this
repo), collected via simulations/collect_inspect_benchmarks.py -- real
per-question scores for ARC (Clark et al., 2018) from 6 real models run
over OpenRouter using Inspect AI, 5 independent runs per model. This script
filters down to 2 of those models and a fixed 50-question random subsample
(seed=0, reproducible), writes the result to examples/arc_results.csv, and
runs the same comparisons evalstats' API would.

Usage:
    python examples/compare_arc_reliability_demo.py

    # Then, to see the full reliability breakdown (seed_std/input_std
    # decomposition, ICC, per-model instability verdict) as terminal
    # output -- e.g. for a screenshot:
    evalstats analyze examples/arc_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

import evalstats as es
from evalstats.core.summary import print_analysis_summary


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = REPO_ROOT / "simulations" / "out" / "inspect_benchmarks.csv"
OUT_PATH = REPO_ROOT / "examples" / "arc_results.csv"

# Real OpenRouter model IDs -> short display names for the demo.
MODEL_RENAME = {
    "openrouter/meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b",
    "openrouter/google/gemma-3n-e4b-it": "gemma-3n",
}
N_ITEMS = 50
SEED = 1

df = pd.read_csv(SOURCE_PATH)
df = df[
    (df["benchmark"] == "arc")
    & (df["model"].isin(MODEL_RENAME))
].copy()
df["model"] = df["model"].map(MODEL_RENAME)

# A complete design: only keep questions every model answered in every run,
# then take a fixed-seed random subsample so the demo is reproducible.
items_by_model = df.groupby("model")["item_id"].apply(set)
common_items = sorted(set.intersection(*items_by_model.values))
rng = np.random.default_rng(SEED)
sample_items = set(rng.choice(common_items, size=N_ITEMS, replace=False))
df = df[df["item_id"].isin(sample_items)]

long_df = df.rename(columns={"item_id": "item", "run_idx": "run"})[["model", "item", "run", "score"]]
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
long_df.to_csv(OUT_PATH, index=False)

print(
    f"Wrote {OUT_PATH.relative_to(REPO_ROOT)} "
    f"({len(long_df)} rows, {long_df['model'].nunique()} models, "
    f"{long_df['item'].nunique()} questions, {long_df['run'].nunique()} runs each)"
)
print()
print("For the full reliability breakdown as terminal output:")
print(f"  evalstats analyze {OUT_PATH.relative_to(REPO_ROOT)}")
print()

evaldata = es.load_from(long_df)
result = es.compare(
    evaldata, factors="model", score_range=(0, 1),
    rng=np.random.default_rng(SEED + 1),
)
print_analysis_summary(
    result.full_analysis, top_pairwise=5,
    item_singular="model", item_plural="models",
)

print()
print("=" * 70)
print("es.stability() -- the same reliability check, on its own")
print("=" * 70)
stability_result = es.stability(
    long_df, config_col="model", run_col="run", item_col="item", value_col="score",
)
print(stability_result.to_frame())
