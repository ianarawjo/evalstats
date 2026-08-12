"""Real-data demo: statistical ties are not transitive.

Three real language models (evaluated via OpenRouter using Inspect AI) on
BBQ, a benchmark of social-bias probes. Their raw accuracies look like a
smooth decline (78%, 71%, 69%), but the pairwise significance tests reveal
a subtler, genuinely counter-intuitive pattern: the top two models are
statistically tied despite a 7-point gap, the bottom two are *also* tied
despite only a 2-point gap -- yet the top model is significantly ahead of
the bottom one. "Tied with" is not transitive here, exactly the kind of
thing a naive leaderboard reading misses, and exactly what evalstats'
pairwise comparison table and significance rank-bands make visible.

Source data: simulations/out/inspect_benchmarks.csv (committed in this
repo), collected via simulations/collect_inspect_benchmarks.py -- real
per-question scores for BBQ (Parrish et al., 2022) from 6 real models run
over OpenRouter using Inspect AI. This script filters down to 3 of those
models and a fixed 100-question random subsample (seed=0, reproducible),
writes the result to examples/bbq_results.csv, and runs the same
comparison evalstats' API would.

Usage:
    python examples/compare_bbq_transitivity_demo.py

    # Then, to see the same analysis (gradient plots, pairwise p-values,
    # rank bands) as full terminal output -- e.g. for a screenshot:
    evalstats analyze examples/bbq_results.csv --p-values
"""

from pathlib import Path

import numpy as np
import pandas as pd

import evalstats as es
from evalstats.core.summary import print_analysis_summary


# Width (in characters) of the ASCII gradient bars in the terminal output --
# print_analysis_summary()'s own default (41) renders wide; shrink this if
# you need the output to fit a narrower terminal or a screenshot crop.
LINE_WIDTH = 41

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = REPO_ROOT / "simulations" / "out" / "inspect_benchmarks.csv"
OUT_PATH = REPO_ROOT / "examples" / "bbq_results.csv"

# Real OpenRouter model IDs -> short display names for the demo.
MODEL_RENAME = {
    "openrouter/openai/gpt-4o-mini": "gpt-4o-mini",
    "openrouter/ibm-granite/granite-4.1-8b": "granite-4.1-8b",
    "openrouter/google/gemma-3n-e4b-it": "gemma-3n-e4b-it",
}
N_ITEMS = 100
SEED = 0

df = pd.read_csv(SOURCE_PATH)
df = df[
    (df["benchmark"] == "bbq")
    & (df["run_idx"] == 0)
    & (df["model"].isin(MODEL_RENAME))
].copy()
df["model"] = df["model"].map(MODEL_RENAME)

# A complete design: only keep questions every one of the 3 models answered,
# then take a fixed-seed random subsample so the demo is reproducible.
items_by_model = df.groupby("model")["item_id"].apply(set)
common_items = sorted(set.intersection(*items_by_model.values))
rng = np.random.default_rng(SEED)
sample_items = set(rng.choice(common_items, size=N_ITEMS, replace=False))
df = df[df["item_id"].isin(sample_items)]

long_df = df.rename(columns={"item_id": "item"})[["model", "item", "score"]]
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
long_df.to_csv(OUT_PATH, index=False)

print(
    f"Wrote {OUT_PATH.relative_to(REPO_ROOT)} "
    f"({len(long_df)} rows, {long_df['model'].nunique()} models, "
    f"{long_df['item'].nunique()} questions)"
)
print()
print("For full terminal output (gradient plots, pairwise p-values, rank bands):")
print(f"  evalstats analyze {OUT_PATH.relative_to(REPO_ROOT)} --p-values")
print()

evaldata = es.load_from(long_df)
result = es.compare(
    evaldata, factors="model", score_range=(0, 1),
    # correction left at its default ("auto" -- resolves to "shaffer" or
    # "romano_wolf" depending on N and data shape; matches `evalstats
    # analyze`'s own default too).
    rng=np.random.default_rng(SEED + 1),
)
print_analysis_summary(
    result.full_analysis, top_pairwise=5, p_value_method="wsr",
    line_width=LINE_WIDTH, item_singular="model", item_plural="models",
)
