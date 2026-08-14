"""Real-data demo: picking the decision-making backend for a small agent.

Say a small autonomous agent needs to pick the right action at each step
from a set of options, using a language model as that step's
decision-making backend -- currently Llama-3.1-8B. Gemma-3n is a candidate
swap: architecturally designed to run far cheaper at inference time despite
having a similar parameter count on disk. But the agent calls this model
autonomously, unattended, many times a day -- so raw accuracy isn't the only
thing that matters. Does it make the same decision when it sees the same
situation twice?

Both models are tested on 50 decision scenarios (a deliberately small
sample -- evalstats is built for exactly this regime), each run 5
independent times to capture how much a model's own output varies from
call to call. Llama-3.1-8B and Gemma-3n score close enough to be
statistically tied on accuracy, so a single-run comparison would call them
interchangeable. But the repeat structure reveals something a single run
cannot: Gemma-3n's decisions are "effectively deterministic" run to run,
while Llama-3.1-8B is "moderately noisy" -- visibly less consistent from
one call to the next, even though its average accuracy is in the same
range. For an agent running unattended, that consistency is itself a real
dependability property, separate from accuracy, and only visible once you
look past a single run.

evalstats surfaces this two ways:

1. compare() includes it automatically whenever it detects a repeated `run`
   column -- no extra step, it's part of the same analysis used to compare
   models in the first place.
2. stability() is a standalone shortcut for when reliability is *all* you
   want to check -- e.g. "is this one model/config reliable enough to
   ship?" -- without running a full multi-model comparison.

Both report the same underlying numbers. A third view, plot_run_disagreement(),
turns this into a figure: one row per model, with a bar over every scenario
where the 5 runs didn't all agree -- taller means a more even split. Ink
never depends on whether the model was *right*, only on whether it was
*consistent*, so a model that's confidently wrong every time looks exactly
as quiet as one that's confidently right every time. Saved to
examples/arc_reliability_plot.png.

Real data note: each "decision scenario" here is actually a question from
ARC (Clark et al., 2018), a benchmark of grade-school science questions --
repositioned as a stand-in for an agent's decision points, not a literal
quiz. Source: simulations/out/inspect_benchmarks.csv (committed in this
repo), collected via simulations/collect_inspect_benchmarks.py -- real
per-question scores from 6 real models run over OpenRouter using Inspect
AI, 5 independent runs per model. This script filters down to 2 of those
models and a fixed 50-question random subsample (seed=1, reproducible),
writes the result to examples/arc_results.csv, and runs the same
comparisons evalstats' API would.

Usage:
    python examples/compare_arc_reliability_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import evalstats as es
from evalstats.core.summary import print_analysis_summary
from evalstats.vis.reliability import plot_run_disagreement


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
    f"{long_df['item'].nunique()} scenarios, {long_df['run'].nunique()} runs each)"
)
print()

print("=" * 70)
print("es.compare() -- full comparison, including the reliability breakdown")
print("=" * 70)
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
    long_df, factor="model", run_col="run", item_col="item", metric="score",
)
stability_result.summary(item_singular="model")

print()
print("=" * 70)
print("plot_run_disagreement() -- the same reliability check, as a figure")
print("=" * 70)
PLOT_PATH = REPO_ROOT / "examples" / "arc_reliability_plot.png"
fig = plot_run_disagreement(result.full_analysis, title="Run-to-Run Reliability on Agent Decision Scenarios")
fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"Wrote {PLOT_PATH.relative_to(REPO_ROOT)}")
