"""Uncertainty-aware trade-off demo: accuracy isn't the only axis that matters.

In interactive systems, a prompt that answers slightly more accurately
after a longer pause often loses to one that answers slightly less
accurately near-instantly -- response latency isn't just an infrastructure
cost, it's felt directly by the person waiting on the other end of the
conversation. A leaderboard sorted by accuracy alone hides that entirely.

A student is building a conversational assistant -- something users talk to
turn-by-turn, where every pause before a reply is a pause someone is
sitting through. They draft 8 prompt variants spanning a spectrum: terse
and zero-shot at one end (fast, less careful), verbose chain-of-thought
with self-consistency voting at the other (slower, more careful). Their
first pass is the obvious one: measure accuracy, sort, ship the top prompt
-- the most elaborate chain-of-thought variant.

Then they run a pilot test. Their advisor sits in, watches a real user wait
through a multi-second pause before each reply, and asks the obvious
question the accuracy-only leaderboard never raised: what does this look
like once you factor in latency, not just accuracy?

That's what compare(secondary_metric=...) / tradeoff() are for. Both jointly
bootstrap accuracy and latency together (a shared per-item resample, not
two independent marginal bootstraps) and report a calibrated verdict per
prompt -- "frontier" (best trade-off), "dominated" (confidently worse on
both axes), or "ambiguous" (looks worse by point estimate, but the data
can't actually confirm it). The Pareto plot doesn't pick an axis to weight
for the student -- that's a UX judgment call about how much latency their
users will tolerate -- it just makes the shape of the trade-off visible, so
"highest accuracy wins" stops being the unreflective default.

Data note: synthetic, with a documented ground-truth shape (see
ACCURACY/LATENCY_S below) -- there's no real per-prompt latency dataset in
this repo to draw from, so this mirrors the existing
compare_pareto_secondary_metric.py in being illustrative rather than
measured.

Usage:
    python examples/tradeoff_prompt_latency_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import evalstats as es


rng = np.random.default_rng(11)

N_ITEMS = 30

# A spectrum from terse/fast to verbose/careful. Accuracy climbs steadily;
# latency climbs faster, especially at the self-consistency end (multiple
# reasoning samples, voted) -- the shape that makes "just pick the top
# accuracy score" the wrong reflex for something a person is waiting on.
#
# zero-shot is the odd one out: with no instruction to be concise, the
# model tends to hedge and ramble before answering -- worse *and* slower
# than the deliberately terse prompt, a "worst of both worlds" case that
# tradeoff() should confidently call dominated, not just unclear.
# few-shot-3 is a second one: padding the prompt with three unexplained
# examples costs tokens (and thus latency) without teaching the model to
# actually reason, so it ends up worse *and* slower than one-line-cot's
# single explicit reasoning hint -- also confidently dominated.
# few-shot-5-cot sits close enough to cot-verbose (worse point estimate on
# both axes, but only slightly) to land "ambiguous" rather than confidently
# either way -- the calibration point: a naive point-estimate-only Pareto
# front would silently call this "dominated" too.
ACCURACY = {
    "zero-shot": 0.50,
    "terse": 0.645,
    "one-line-cot": 0.70,
    "few-shot-3": 0.60,
    "cot": 0.78,
    "cot-verbose": 0.80,
    "few-shot-5-cot": 0.785,
    "cot-selfconsistency": 0.845,
}
LATENCY_S = {  # lower is better
    "zero-shot": 0.62,
    "terse": 0.29,
    "one-line-cot": 0.62,
    "few-shot-3": 1.05,
    "cot": 1.55,
    "cot-verbose": 2.05,
    "few-shot-5-cot": 2.75,
    "cot-selfconsistency": 4.6,
}

rows = []
for prompt, acc in ACCURACY.items():
    lat = LATENCY_S[prompt]
    for i in range(N_ITEMS):
        rows.append({
            "prompt": prompt,
            "item": f"q{i:03d}",
            "accuracy": float(np.clip(rng.normal(acc, 0.09), 0, 1)),
            "latency_s": float(np.clip(rng.normal(lat, lat * 0.15), 0.05, None)),
        })
df = pd.DataFrame(rows)

print("=" * 70)
print("The student's first pass: sort by accuracy alone")
print("=" * 70)
leaderboard = df.groupby("prompt")["accuracy"].mean().sort_values(ascending=False)
print(leaderboard.to_string())
print(f"\n-> Ships '{leaderboard.index[0]}': the highest accuracy, full stop.")
print()

print("=" * 70)
print("The advisor's question, after watching a pilot user wait: what about latency?")
print("es.tradeoff() -- accuracy vs. latency, jointly")
print("=" * 70)
result = es.tradeoff(
    df, factor="prompt", item_col="item",
    primary_metric="accuracy", secondary_metric={"latency_s": "min"},
    rng=np.random.default_rng(12),
)
result.summary(show_rank_probabilities=True)

print()
print("=" * 70)
print("result.plot() -- the trade-off as a figure")
print("=" * 70)
REPO_ROOT = Path(__file__).resolve().parent.parent
PLOT_PATH = REPO_ROOT / "examples" / "tradeoff_prompt_latency_plot.png"
fig = result.plot()
fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"Wrote {PLOT_PATH.relative_to(REPO_ROOT)}")

# A smaller figsize for a paper figure -- font sizes are fixed in points,
# so shrinking the physical figure makes the text proportionally bigger
# relative to the plot without needing to touch fontsize anywhere.
PAPER_PLOT_PATH = REPO_ROOT / "examples" / "tradeoff_prompt_latency_plot_small.png"
fig_small = result.plot(figsize=(5.6, 3.7))
fig_small.savefig(PAPER_PLOT_PATH, dpi=200, bbox_inches="tight")
print(f"Wrote {PAPER_PLOT_PATH.relative_to(REPO_ROOT)}")
