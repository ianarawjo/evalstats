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

Total: 4 x 3 x 4 x 9 = **432 cells** x **5 reps/cell** = **2,160 items**.

`task_type` (`prompts` vs `models`) is **not** part of this cross. Which
word we use to describe the entities being compared doesn't change any
statistical property of the design -- it's cosmetic framing, not a real
factor -- so fully crossing it would just double every other cell for no
added analytical coverage. Instead it's assigned as a **balanced random
covariate**: each of the 432 cells gets exactly one of `prompts`/`models`
(216 cells each, via a fixed shuffle, so the split is exact rather than an
average over independent coin flips -- see `build_benchmark.py`'s
`assign_task_types`). This still gives full statistical power to check
whether `task_type` wording measurably affects agent behavior, without
presupposing in advance that it's important enough to fully cross.

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
  Welch's correction or bootstrap needed. Implemented via a different `icc`
  (reliability) drawn independently per arm (`dgp.py`'s
  `sample_icc_per_arm`) rather than `sample_group_truth`'s own
  `heteroscedastic=True` flag -- that flag scales noise by how close a
  value sits to the scale's boundary, so arms only end up with visibly
  different variance if they also land at very different absolute
  positions; verified empirically that with a shared baseline it left every
  arm at ~0.35 std regardless of the flag. Per-arm `icc` instead creates
  variance differences that are a property of the arm itself, independent
  of where the comparison sits on the scale -- what "unequal variance"
  should actually mean. `icc` is drawn independently of `winner_idx`, so the
  winning arm isn't systematically more or less reliable than the rest.
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

## Things randomized per cell, not fixed

An early version of this generator always put the effect on entity index 0
(so `true_best` was always `prompt_1`/`model_1`) and always centered every
comparison at whatever level its shape's default happened to be (~0.4-0.5
for the continuous/binary shapes here, ~3.0 for `likert-mid`). Both are
real benchmark flaws, not cosmetic: a fixed winner index is a positional
shortcut a sufficiently capable (or memory-having) agent could exploit
without doing any actual analysis, and always testing the same middle-of-
scale region can't tell you whether `evalstats` behaves the same near a
bounded scale's edges as it does in the middle -- variance mechanically
shrinks near 0/1 (or near 1/5), which changes CI shape/skew even under an
otherwise "clean" DGP regime.

All of the below are drawn once per cell (shared by that cell's 5 reps,
since power and the correct answer are properties of the design, not of one
draw) via a dedicated per-cell RNG in `build_benchmark.py`:

- **`winner_idx`**: which entity (0..k-1) carries the effect, uniform over
  `range(k)`. `ground_truth.true_best`/`correct_decisions` follow whichever
  entity actually won, not always entity 1.
- **`baseline_shift`**: a common additive shift applied to *every* arm
  before the winner's extra delta is added on top (`dgp.py`'s
  `sample_baseline_shift`, +/-40% of the data type's total range) -- moves
  where the whole comparison sits on the scale (e.g. a binary cell might
  sit around 0.15 pass rate or 0.9, not always ~0.5) while leaving the
  *difference* between arms exactly what `ground_truth_regime` specifies.
- **`icc`** (`unequal_variance` cells only): a different reliability drawn
  independently per arm (`dgp.py`'s `sample_icc_per_arm`, range 0.1-0.5)
  instead of a single shared value -- see "Why these specific dgp_regime
  levels" above for why this replaced `sample_group_truth`'s
  `heteroscedastic=True` flag. All other cells use a single shared
  `DEFAULT_ICC=0.25` (same value both earlier families used).

All three are stored in `ground_truth` (`winner_idx`, `baseline_shift`,
`icc`) for anyone later analyzing whether `evalstats`'s accuracy varies by
scale position or arm reliability, not just averaged across it.
- **`task_type`**: not a per-cell random draw in the same sense as the
  three above -- it's the balanced-random covariate described under "The
  grid," assigned once across all 432 cells (not resampled per cell) so the
  overall split is exactly 216/216 rather than an expectation.

## Decision schema (unified across the whole sweep)

Both earlier families each had their own answer schema (`deploy`/
`do_not_deploy`/`inconclusive` for `prompt_ab`; `model_1..model_8`/
`inconclusive` for `model_benchmark`). Neither generalizes cleanly across a
k-sweep, so this benchmark uses one schema everywhere: entity labels
(`prompt_1..prompt_k` or `model_1..model_k`, whichever one is
`ground_truth.winner_idx` carries the effect shift when non-null -- not
always entity 1, see "Things randomized per cell" above) plus
`"inconclusive"`, for both `clear_best` and `decision`:

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
  "n_cells": 432,
  "n_items": 2160,
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
        "true_best": "model_2",
        "winner_idx": 1,
        "baseline_shift": -0.1234,
        "icc": 0.25,
        "correct_decisions": ["model_2", "inconclusive"],
        "correct_clear_best": ["model_2", "inconclusive"],
        "valid_decisions": ["model_1", "model_2", "model_3", "model_4", "model_5", "inconclusive"],
        "valid_clear_best": ["model_1", "model_2", "model_3", "model_4", "model_5", "inconclusive"]
      }
    },
    ...
  ]
}
```

`ground_truth.icc` is a single float for every regime except
`unequal_variance`, where it's instead a list of `k` floats (one per arm,
`dgp.py`'s `sample_icc_per_arm`) -- a different reliability drawn
independently per entity, which is what actually creates cross-arm variance
heterogeneity (see "Why these specific dgp_regime levels" above).

`ground_truth.power` is a Monte-Carlo estimate (see `generate.py`'s
`_estimate_power`) of the probability that a correctly-conducted,
Holm-corrected analysis would identify `true_best` as significantly ahead
of every other entity at this exact (k, n, effect size, DGP regime) design
-- it's a property of the *cell* (shared by all 5 reps), not of one
specific random draw. `correct_decisions`/`correct_clear_best` accept
`"inconclusive"` alongside the directional answer whenever `power < 0.8`,
so a well-calibrated "we can't tell" isn't scored as a mistake (see
`agent_study/scenarios.py`'s `POWER_THRESHOLD` for precedent).

This power estimate uses a plain Welch/pooled t-test + Holm correction, not
literally `evalstats`' own default methods (BCa bootstrap CIs + `fdr_bh`) --
running the real bootstrap pipeline inside a 300-rep Monte Carlo loop, once
per cell, wasn't worth the extra compute for a ground-truth proxy. This
isn't "cheating" in the sense of stacking the deck in `evalstats`' favor:
`evalstats`' defaults were themselves chosen from simulations showing they
approximate the true sampling behavior well across these same regimes, so a
correctly-conducted t-test is a reasonable stand-in for "what a
well-calibrated method would conclude here," and the actual question this
benchmark asks is whether an agent, with or without `evalstats`, can
reproduce that judgment -- not whether `evalstats` matches its own ground
truth by construction.

**This is fully self-contained.** A different harness can run this
benchmark without importing any of our Python simulation code: write
`data.rows` out as a CSV using `data.columns` as the header, drop it plus
`prompt` into an agent's workspace, and score its `recommendation.json`
against `ground_truth` -- that's the entire integration surface.

## Regenerating

```
.venv/bin/python -m agent_study.sweep.build_benchmark --dry-run-cells 5   # quick sanity check, no API calls
.venv/bin/python -m agent_study.sweep.build_benchmark                     # full build, no API calls
```

Runtime is dominated by the per-cell 300-rep Monte Carlo power estimate,
not data generation itself. Expect somewhere in the 20-30 min range on an
unloaded machine for the current 432-cell grid, but this can run much
longer if the machine throttles or sleeps during a long unattended
background run -- that's a machine-scheduling artifact, not a sign of a
hung process.

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
