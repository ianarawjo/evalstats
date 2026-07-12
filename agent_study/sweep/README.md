# Agent study: expanded sweep (`sweep_v1`)

A larger, more systematic benchmark than the two hand-built families in
`agent_study/` (`prompt_ab`, `model_benchmark`). Where those tested one
scenario each, this is a full grid over the dimensions of `evalstats`'
feature surface, stored as a single portable JSON manifest rather than
generated on the fly by Python code -- **this directory only builds and
documents the benchmark dataset. It does not run any agents.** Execution
against this manifest is a separate, later step.

## The grid

| Axis | Levels | Count |
|---|---|---|
| `task_type` | prompts, models | 2 |
| `k` (entities compared) | 2, 5, 10, 20 | 4 |
| `ground_truth_regime` (effect size) | true_null (d=0), borderline (d=0.2), clear_winner (d=0.8) | 3 |
| `n` (sample size) | 15, 30, 50, 200 | 4 |
| `data_type` x `dgp_regime` | see below | 9 |

`dgp_regime` is restricted per `data_type` rather than fully crossed, since
not every regime is a meaningful, distinct concept for every data type:

| data_type | dgp_regime levels | why |
|---|---|---|
| continuous | clean, unequal_variance, skewed, heavy_tail, mixture | all 5 are real, distinguishable properties of a continuous distribution |
| likert | clean, skewed, mixture | bounded 1-5 scale; heavy-tail/outlier is a stretch here, and its own catalog already has skewed + bimodal shapes |
| binary | clean | a Bernoulli is fully described by its rate -- no separate concept of skew, heavy tails, or "unequal variance" beyond differing rates, which is already what a non-null effect *is* |

Total: 2 x 4 x 3 x 4 x 9 = **864 cells** x **5 reps/cell** = **4,320 items**.

**Held fixed / out of scope for this sweep** (see the design discussion that
produced this grid, in conversation history, for the reasoning):
- Judge-score-correction (PPI/`validate_alignment`) task type -- deferred to
  a separate, smaller sub-sweep with its own axes (judge reliability,
  human-label fraction), since those don't apply to `prompts`/`models`.
- Correlation structure: paired throughout (`base_corr=1.0` -- the same
  items scored across every entity), matching `evalstats`' own design
  assumption and how both earlier families already worked.
- Prompt style: implicit only.
- Model (Haiku/Sonnet/etc.) and `evalstats` on/off: these are properties of
  a *run*, not of a benchmark *item* -- the same item is meant to be run
  under both conditions and (optionally) multiple models later.

## Why these specific dgp_regime levels

The point of this axis is cases where a naive, normality-assumed analysis
(a plain t-test assuming iid Gaussian, equal variance) gives unreliable
results -- the regime `evalstats`'s bootstrap/robust methods exist for.
- **unequal_variance**: breaks the standard pooled-variance t-test;
  Welch's correction or bootstrap needed. (`sample_group_truth(...,
  heteroscedastic=True)`.)
- **skewed**: at small n the sampling distribution of the mean isn't yet
  Gaussian, so a normal-theory CI has systematically wrong coverage --
  the textbook justification for BCa-style bootstrap correction. Compounds
  directly with the `n` sweep (worst at n=15, should shrink by n=200).
- **heavy_tail**: a "contaminated normal" -- ~88% well-behaved data, ~12%
  slammed to an exact 0 or 1 (catastrophic failures / perfect scores mixed
  into otherwise ordinary data). A few outliers can inflate a naive
  variance estimate enough to mask a real effect or manufacture a false
  one; this is a new custom shape (`dgp.py`'s
  `_outlier_contaminated_sampler`), not yet in
  `simulations/harness/scenarios/synthetic.py`'s shared catalog.
- **mixture**: a qualitatively different failure mode from skew -- not "the
  CI has wrong coverage" but "the mean is a well-estimated number that
  doesn't describe anyone" (e.g. `cont-mixture`/`grades-mixture`'s multiple
  real sub-populations folded into one measured group).

`clean` in each case is the control: roughly normal-ish (or the
data-type's ordinary default shape), equal variance -- where a naive
analysis should actually do fine, giving something to compare the other
regimes against.

## Decision schema (unified across the whole sweep)

Both earlier families each had their own answer schema (`deploy`/
`do_not_deploy`/`inconclusive` for `prompt_ab`; `model_1..model_8`/
`inconclusive` for `model_benchmark`). Neither generalizes cleanly across a
k-sweep, so this benchmark uses one schema everywhere: entity labels
(`prompt_1..prompt_k` or `model_1..model_k`, arm 0 is always the one that
carries the effect shift when non-null) plus `"inconclusive"`, for both
`clear_best` and `decision`:

```json
{"reasoning": "...", "clear_best": "model_3", "certainty": "low", "decision": "inconclusive"}
```

See `agent_study/prompts.py`'s docstrings for the full rationale behind
splitting `clear_best` (honest point-estimate read) from `certainty`
(self-reported confidence) from `decision` (the actual recommendation,
allowed to diverge from `clear_best`) -- that schema is unchanged here,
just generalized to arbitrary k.

## Manifest schema (`benchmark_v1.json`)

```
{
  "benchmark_version": "sweep_v1",
  "generated_at": "...",
  "axes": { ... the grid above, machine-readable ... },
  "n_cells": 864,
  "n_items": 4320,
  "items": [
    {
      "item_id": "models__k5__continuous__heavy_tail__borderline__n30__rep3",
      "task_type": "models",
      "k": 5,
      "data_type": "continuous",
      "dgp_regime": "heavy_tail",
      "ground_truth_regime": "borderline",
      "n": 30,
      "rep": 3,
      "seed": 1247403,
      "prompt": "We're picking which of 5 candidate models ... <full prompt text>",
      "data": {
        "columns": ["item_id", "entity", "score"],
        "rows": [[0, "model_1", 0.71], [0, "model_2", 0.44], ...]
      },
      "ground_truth": {
        "entity_labels": ["model_1", "model_2", "model_3", "model_4", "model_5"],
        "cohens_d": 0.2,
        "is_null": false,
        "power": 0.31,
        "true_best": "model_1",
        "correct_decisions": ["model_1", "inconclusive"],
        "correct_clear_best": ["model_1", "inconclusive"],
        "valid_decisions": ["model_1", "model_2", "model_3", "model_4", "model_5", "inconclusive"],
        "valid_clear_best": ["model_1", "model_2", "model_3", "model_4", "model_5", "inconclusive"]
      }
    },
    ...
  ]
}
```

`ground_truth.power` is a Monte-Carlo estimate (see `generate.py`'s
`_estimate_power`) of the probability that a correctly-conducted,
Holm-corrected analysis would identify `true_best` as significantly ahead
of every other entity at this exact (k, n, effect size, DGP regime) design
-- it's a property of the *cell* (shared by all 5 reps), not of one
specific random draw. `correct_decisions`/`correct_clear_best` accept
`"inconclusive"` alongside the directional answer whenever `power < 0.8`,
so a well-calibrated "we can't tell" isn't scored as a mistake (see
`agent_study/scenarios.py`'s `POWER_THRESHOLD` for precedent).

**This is fully self-contained.** A different harness can run this
benchmark without importing any of our Python simulation code: write
`data.rows` out as a CSV using `data.columns` as the header, drop it plus
`prompt` into an agent's workspace, and score its `recommendation.json`
against `ground_truth` -- that's the entire integration surface.

## Regenerating

```
.venv/bin/python -m agent_study.sweep.build_benchmark --dry-run-cells 5   # quick sanity check, no API calls
.venv/bin/python -m agent_study.sweep.build_benchmark                     # full build, ~20-30 min, no API calls
```

Fully deterministic: every item's `seed` is derived from its position in
the fixed cell-iteration order (`build_benchmark.py`'s `iter_cells`), so
re-running produces byte-identical data. Nothing in this directory calls
an LLM API -- it's pure local computation (numpy/scipy/statsmodels), safe
to re-run freely.

## What's not here yet

- No execution harness for this manifest (that's `agent_study/run_agent.py`
  et al., built for the two smaller families -- adapting it to consume
  `benchmark_v1.json` items generically is a separate next step).
- Judge-score-correction task type and its own reliability/label-fraction
  sub-sweep.
- Explicit-prompt variant.
