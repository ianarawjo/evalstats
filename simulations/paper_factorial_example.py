"""Generates the paper's two-factor (model x prompt) motivating example --
Scenario 1 from S:motivation, used in the "Comparing two factors
simultaneously" subsection.

DISCLOSED SYNTHETIC DATA: unlike paper_flipflop_example.py, there is no
underlying real dataset here. Model and prompt names are fictional. This is
a deliberate illustration, not a benchmarking claim about any real model.

Story the numbers are built to tell: a developer grid-searches 8 candidate
models x 5 prompt strategies on N=30 held-out support tickets, where each
response is parsed into a structured record and serialized to a canonical
string (fixed field order and formatting). That string is compared to the
ground-truth record's canonical serialization via Levenshtein (edit)
distance, normalized by the longer of the two string lengths and
subtracted from 1: score = 1 - edit_distance(response, reference) /
max(len(response), len(reference)). This is a standard character-error-
-rate-style metric -- continuous by construction (a distance-over-length
ratio, not a handful of discrete field-match flags averaged together),
fully deterministic, and requires no ML model or subjective judgment of
any kind, LLM or human. Marginal effects alone would predict the flagship
model (Atlas-XL) paired with the on-average-best prompt (self-critique) as
the winner. The actual winner is a genuine interaction: Ember-1 (a
mid-tier model tuned for schema-constrained generation) paired with
structured-output prompting -- neither the best model nor the best prompt
on its own -- edges out the "obvious" combo, because it produces
serializations that land consistently closer (in edit distance) to the
canonical reference. That's only visible in the cross-factor table, which
is the point of the subsection: single-factor summaries would miss it.

Design is fully paired/repeated-measures: every (model, prompt) cell is
scored on the same 30 tickets, giving the high-covariance, many-comparison
regime (40 cells, C(40,2)=780 pairwise diffs) where evalstats' FWER
correction methods are meant to shine.

Run:
    .venv/bin/python -m simulations.paper_factorial_example
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUT_CSV = "simulations/out/paper_factorial_example.csv"

N_TICKETS = 30
# Score is a [0, 1] normalized edit-distance similarity between the
# response's canonically-serialized structured output and the ground-truth
# record's serialization (1 - Levenshtein distance / max string length),
# not a 0-100 grade. Continuous by construction, no rubric, no ML model.
# The CLI auto-detects the [0, 1] range and picks its better-calibrated
# logit-t robustness method silently, instead of falling back to the
# bounds-agnostic t_interval with a "pass score_range explicitly" warning
# (analyze's --score-range isn't exposed as a CLI flag yet).
BASELINE = 0.50
SIGMA_INPUT = 0.06   # ticket-to-ticket difficulty spread
SIGMA_RESID = 0.05   # scorer/sampling noise (model stochasticity across runs)
SEED = 7

MODEL_EFFECTS = {
    "Atlas-XL": 0.18,
    "Solstice-Pro": 0.14,
    "Ember-1": 0.12,
    "Atlas-M": 0.10,
    "Nimbus-70B": 0.08,
    "Solstice-Mini": 0.06,
    "Kestrel-2": 0.04,
    "Nimbus-7B": 0.00,
}
PROMPT_EFFECTS = {
    "self-critique": 0.08,
    "chain-of-thought": 0.06,
    "structured-output": 0.05,
    "few-shot": 0.04,
    "zero-shot": 0.00,
}
# The one non-obvious synergy: Ember-1 is schema-tuned, so structured-output
# prompting unlocks disproportionate gains for it specifically.
INTERACTIONS = {
    ("Ember-1", "structured-output"): 0.12,
}


def make_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = list(MODEL_EFFECTS)
    prompts = list(PROMPT_EFFECTS)
    input_effects = rng.normal(0.0, SIGMA_INPUT, size=N_TICKETS)

    rows = []
    for i in range(N_TICKETS):
        for m in models:
            for p in prompts:
                score = (
                    BASELINE
                    + input_effects[i]
                    + MODEL_EFFECTS[m]
                    + PROMPT_EFFECTS[p]
                    + INTERACTIONS.get((m, p), 0.0)
                    + rng.normal(0.0, SIGMA_RESID)
                )
                rows.append({
                    "input": f"ticket_{i:02d}",
                    "model": m,
                    "prompt": p,
                    "score": float(np.clip(score, 0.0, 1.0)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    df = make_data()
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows ({N_TICKETS} tickets x {len(MODEL_EFFECTS)} models x "
          f"{len(PROMPT_EFFECTS)} prompts) to {OUT_CSV}")

    # Sanity check: which (model, prompt) cell has the highest true mean?
    true_means = df.groupby(["model", "prompt"])["score"].mean().sort_values(ascending=False)
    print("\nTop 5 cells by empirical mean:")
    print(true_means.head(5).to_string())

    best_model_marginal = df.groupby("model")["score"].mean().idxmax()
    best_prompt_marginal = df.groupby("prompt")["score"].mean().idxmax()
    print(f"\nBest model on average: {best_model_marginal}")
    print(f"Best prompt on average: {best_prompt_marginal}")
    print(f"'Obvious' combo from marginals: ({best_model_marginal}, {best_prompt_marginal})")
    print(f"Actual best cell: {true_means.index[0]}")


if __name__ == "__main__":
    main()
