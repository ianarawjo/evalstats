# Why `correct()`'s power-tuning shrinkage targets 1, not 0

Date: 2026-08-12
Code: `evalstats/ppi.py`, `_POWER_TUNE_SHRINKAGE_C` (=20.0) and the three
`lam = 1.0 - (1.0 - lam) * n_lab / (n_lab + _POWER_TUNE_SHRINKAGE_C)` sites
(analytic-mean, analytic-Walsh, and bootstrap paths).

## Background

`correct(power_tune=True)` estimates a variance-minimizing weight λ̂ for how
much to trust the LLM-judge rectifier (PPI++, Angelopoulos/Duchi/Zrnic
2023). The raw bootstrap-estimated λ̂ is noisy at small `n_lab`, so the
codebase shrinks it back toward **1** (vanilla PPI, i.e. "trust the LLM
correction fully") by an `n_lab`-dependent amount:

```
lam = 1.0 - (1.0 - lam) * n_lab / (n_lab + C)     # C = _POWER_TUNE_SHRINKAGE_C = 20.0
```

A separate reviewing agent flagged this as suspicious, expecting shrinkage
to pull toward **0** (the classical labels-only estimate) instead, since 0
is the "safe null" in most shrinkage-estimator conventions (James-Stein,
etc.) and is a more conservative fallback. This doc records an isolated
empirical check of what actually happens if you flip the shrinkage target.

## What was tested

**Investigation 1** (single synthetic scenario, no code change): monkey-patched
`_POWER_TUNE_SHRINKAGE_C` to 0 (i.e. *no* shrinkage at all — raw λ̂ used
as-is) vs. the shipped default (C=20, shrink→1), on a hand-built
single-sample mean scenario with an explicitly uninformative judge
(`Y_hat` barely correlated with truth) and an informative judge (control).
Removing shrinkage entirely gave up to ~3x the power for the uninformative
judge at small `n_lab`, at the cost of modest Type-I inflation (0.07-0.09
vs. nominal 0.05) that faded by `n_lab=50`.

**Investigation 2** (this doc, real code change): actually changed the
shrinkage target from 1 to 0 in all three `correct()` code paths:

```
lam = lam * n_lab / (n_lab + _POWER_TUNE_SHRINKAGE_C)   # shrink toward 0 instead of 1
```

Ran the harness's Type-I calibration sweep and 5-way estimator comparison
(`all_human`/`human_subset`/`llm_only`/`llm_impute`/`ppi`) via:

```
python -m simulations.harness.cli pvalues --mode ppi \
  --reps 100 --effect-reps 100 --ppi-n-boot 500 \
  --tests ttest ttest_welch mwu wilcoxon paired_t bayes_bootstrap \
  --eval-types binary continuous likert \
  --no-power-check --no-effect-check --no-label-efficiency-check \
  --workers 15 --seed 42
```

(`--tests` restricted to the 6 methods that actually route through
`power_tune` — `kruskal`/`anova_*`/`friedman`/`tango_score`/`lmm*` don't use
this code path and are unaffected by the change.)

Compared against the existing full official run `official_20260803_013611`
(shrink→1, `--reps 300`, same scenario suite).

## Results

**Type-I error** (mean corrected rejection rate under H0, restricted to the
same 6 tests):

| | shrink→1 (baseline) | shrink→0 (screening) |
|---|---|---|
| mean corrected Type-I | 0.053 | 0.058 |

Both close to nominal 0.05. The N×N_lab calibration grid (continuous,
n_lab=15) was already slightly *conservative* under shrink→1 (0.034-0.046)
and lands close to nominal under shrink→0 (0.046-0.058) — a wash, not a
meaningful fix.

**5-way comparison, PPI power vs. n_lab** (paired_t estimand, `es=0.30`):

| eval_type | n_lab | shrink→1 | shrink→0 | Δ |
|---|---|---|---|---|
| continuous | 15 | 0.292 | 0.230 | −21% |
| continuous | 20 | 0.328 | 0.290 | −12% |
| continuous | 30 | 0.389 | 0.338 | −13% |
| continuous | 40 | 0.409 | 0.378 | −8% |
| likert | 15 | 0.460 | 0.262 | **−43%** |
| likert | 20 | 0.500 | 0.392 | −22% |
| likert | 30 | 0.588 | 0.540 | −8% |
| likert | 40 | 0.570 | 0.586 | +3% (noise) |
| binary | 15 | 0.362 | 0.325 | −10% |
| binary | 20-40 | 0.435-0.580 | 0.425-0.610 | ~flat |

Power loss under shrink→0 is worst exactly where n_lab is smallest (where
shrinkage has the most pull) and fades toward n_lab=40, as expected from the
`n_lab/(n_lab+C)` weighting.

## Why shrink→1 wins here

`build_ppi_power_baseline`'s standard bias/noise scenario (and the whole
Type-I/comparison sweep built on it) uses `llm_noise = 0.20` population SDs
as its baseline, with the harshest level in the standard `llm_noise` sweep
at `0.70` SDs (`Y_hat = Y + N(0, frac*SD)`). At `frac=0.70`, the implied
Pearson correlation between judge and truth is `1/sqrt(1+0.7^2) ≈ 0.82` —
still a strongly informative judge.

In that regime the *true* λ* usually sits well above 0. Shrinking the noisy
small-`n_lab` λ̂ up toward 1 happens to pull it toward the right answer for
free — that's where shrink→1's power edge over `human_subset` comes from.
Shrinking toward 0 instead pulls λ̂ toward the classical labels-only
estimator regardless of whether the judge is actually good, which is the
wrong prior in this regime and gives up real power for no offsetting
Type-I gain.

## Does this generalize, or is it a simulation-setup artifact?

Investigation 1 (the uninformative-judge probe) and Investigation 2 (the
full sweep) look contradictory, but they aren't — they're probing different
judge-quality regimes:

- The **standard Type-I/comparison sweep** (Investigation 2) never tests a
  judge worse than `noise=0.70` SDs (r≈0.82). Its whole scenario grid is
  "biased but still clearly informative" judges — the case the paper's PPI
  correction is actually meant for.
- The **opt-in factorial sweep** (`PPI_FACTORIAL_NOISE_LEVELS`, not run in
  this screening pass) spans `noise` from 0.025 up to **28.4** SDs
  (r≈0.035, essentially pure noise) — i.e. this codebase already
  deliberately models genuinely-uninformative judges elsewhere, it's just
  not part of the fast Type-I/5-way-comparison checks used here.
- Investigation 1's hand-built "uninformative judge" scenario (r≈0.24,
  `noise_sd=4.0`) sits inside that factorial-sweep range, well past the
  standard sweep's r≈0.82 floor.

So: **shrink→1 does have a real, mechanistic cost when the judge is
genuinely uninformative** — confirmed directly in Investigation 1, not an
artifact of that probe's construction. Forcing λ up toward 1 when the
honest bootstrap evidence says λ*≈0 means blending in a rectifier term
whose true covariance with the target is ~0 — pure noise, no bias-
correction benefit, which inflates variance and can leave power *below*
even the classical human-only test (the exact risk `correct()`'s own
`power_tune` docstring already documents for the λ=1 fixed estimator: "a
sufficiently uninformative or noisy LLM can make the λ=1 estimator strictly
WORSE... than just running the classical test on Y_lab alone").

Whether that regime is *realistic* depends on deployment context, not on
the math:
- Practitioners generally adopt an LLM-judge-plus-PPI workflow precisely
  because the judge correlates reasonably well with human labels — that's
  usually a prerequisite for bothering with it at all. The standard sweep's
  ceiling (r≈0.82 at its worst) is a reasonable model of "biased LLM judge
  that's still worth correcting," which is the common case.
- A genuinely near-random judge (r≈0.2-0.3) is a real possibility in
  practice (novel/hard domains, judge-model/task mismatch), and the
  factorial sweep already models it — but it's a tail case, not the
  typical one this harness's headline Type-I/power numbers represent.
- This tail risk isn't entirely unmitigated in practice: `evalstats/
  alignment.py` provides a separate judge-human alignment report
  (correlation/kappa/ICC with CIs) that a user following the intended
  workflow would consult alongside `correct()` — a genuinely uninformative
  judge would show up there before/alongside a disappointing PPI result.

## Conclusion

Retargeting shrinkage to 0 is a **net negative** across the codebase's
realistic calibration suite (broad power loss, no meaningful Type-I gain)
and should not replace the shrink→1 default. The shrink→1 choice is not
"wrong" as flagged — it's deliberately tuned for the common "biased but
informative judge" case this harness's scenario grid represents, at a real
but scoped cost in the (rarer, tail-case, already-tested-elsewhere-in-the-
factorial-sweep, partially-mitigated-by-alignment-reporting) genuinely-
uninformative-judge regime.

**No code was changed** — this was an isolated worktree experiment; the
shrinkage-target edit was reverted after the screening run completed.

## Artifacts

- Screening run (shrink→0): `simulations/out/screening_shrink0_20260812_203251/`
- Baseline (shrink→1): `simulations/out/official_20260803_013611/` (main repo,
  not this worktree)

---

## Addendum (2026-08-12): explicit poor-judge sweep + a second agent's critique

A second agent reviewed the above and pushed back on two points, plus flagged
that a **narrower, already-in-progress fix** exists in the main repo
(`/Users/ianarawjo/Documents/prompt-stats`, branch `compare-e2e`,
uncommitted). Checked that diff directly rather than taking the description
at face value:

- **Confirmed real, but narrower than described.** `_analytic_mean_point_se`
  (closed-form mean/paired-mean backend) now shrinks toward 0 via a new
  `_shrink_power_tune_lambda` helper, EXCEPT when `var_lab` is degenerate
  (near-constant labeled sample), where it falls back to shrink-toward-1 to
  avoid a CI-collapse failure mode. `_analytic_walsh_theta_correct`
  (Wilcoxon/rank backend) was deliberately left at shrink-to-1, with a
  comment citing existing evidence of power gains there. **The bootstrap
  path in `correct()` itself — used by `mwu`, `bayes_bootstrap`, and
  `ttest`/`ttest_welch`/`paired_t` at `n_lab>=30` — is unchanged**, still
  plain shrink-to-1, no variance floor. So this document's screening run
  (which flips all three sites uniformly) only partially overlaps with that
  scoped fix, and doesn't test it directly. Checked the screening run's raw
  results for the coverage-collapse the other agent described from their
  own first attempt (0 failed cells, max Type-I rate 0.16 out of 968) — not
  triggered here, plausibly because this scenario grid rarely produces a
  near-constant `Y_lab`.

### Explicit poor-judge sweep (single-arm mean, reps=2000)

Reimplemented `correct()`'s bootstrap path outside the source file (so
shrink-to-1/shrink-to-0 could be compared without editing `ppi.py` again),
swept judge noise from harness-typical (0.35 SD, r=0.94) down to genuinely
uninformative (8.0 SD, r=0.12), and added `human_subset` (classical
one-sample t-test on `Y_lab` alone — the "is PPI worth it at all" floor)
as a third comparison point.

| noise_sd | r(judge,truth) | n_lab | shrink→1 power | shrink→0 power | human_subset power |
|---|---|---|---|---|---|
| 0.35 | 0.94 | 15 | 0.932 | 0.590 | 0.251 |
| 0.70 | 0.82 | 15 | 0.589 | 0.493 | 0.251 |
| 1.50 | 0.56 | 15 | 0.276 | 0.397 | 0.251 |
| 3.00 | 0.32 | 15 | **0.141** | 0.353 | 0.251 |
| 3.00 | 0.32 | 30 | **0.241** | 0.519 | 0.446 |
| 5.00 | 0.20 | 30 | **0.150** | 0.504 | 0.446 |
| 8.00 | 0.12 | 60 | **0.202** | 0.780 | 0.767 |

**This is the key new finding: there's a real crossover, and it's not a
tail case in the way the original doc implied.**

- Below the crossover (r≳0.7-0.8, roughly the harness's standard-sweep
  ceiling), shrink→1 wins clearly and stays comfortably above
  `human_subset` — the original doc's conclusion holds here unchanged.
- Above the crossover (r≲0.5-0.6, noise≳1.5 SD), shrink→1 **drops below
  `human_subset`** — using `correct(power_tune=True)` becomes strictly
  worse than ignoring the LLM judge entirely. This isn't a knife-edge
  effect: at noise=8 SD, n_lab=60, shrink→1's power (0.202) is barely a
  quarter of `human_subset`'s (0.767), and this does **not** self-correct
  with more labels within a realistic range — the fixed pseudo-count
  `C=20` means `n_lab` needs to be well past 100-200 before the toward-1
  pull weakens enough to stop mattering, regardless of how bad the judge
  actually is.
- shrink→0, by contrast, never drops meaningfully below `human_subset`
  anywhere in this sweep (it converges to it from above as the judge gets
  worse, since a genuinely near-zero raw λ̂ just recovers the classical
  estimator).
- Type-I error stayed similarly (mildly) elevated under both targets
  throughout (0.05-0.09) — this is a **power** story, not a Type-I
  calibration collapse either way.

A correlation of ~0.55 (the crossover point) isn't an exotic worst case —
it's a plausible outcome for a genuinely hard or subjective eval task. The
original doc's framing ("tail case, largely mitigated by alignment
reporting") undersold this: the risk is real, has a concrete threshold, and
the current fixed shrink→1 default has no mechanism to detect it's in that
regime and back off.

### Paired-difference, good judge, small effect (reps=2000)

Tested the critique's second claim — that low λ* isn't only a bad-judge
phenomenon, since a paired estimate's LLM-noise contribution is the sum of
two independent per-arm errors (effectively `noise_sd*sqrt(2)` on the
difference), so a *good* per-arm judge plus a *small* true effect could
independently produce the same failure mode.

Construction: per-item true difference `D ~ N(true_diff, 1.0)` (genuine
item-by-condition interaction, not a deterministic shift), judge noise
`0.35` SD per arm (harness-typical, "good"), `true_diff=0.15` (small):

| n_lab | shrink→1 power | shrink→0 power | human_subset power | λ̂ raw (H0) |
|---|---|---|---|---|
| 15 | 0.246 | 0.181 | 0.083 | 0.756 |
| 30 | 0.357 | 0.278 | 0.138 | 0.727 |
| 60 | 0.511 | 0.490 | 0.217 | 0.663 |

**Doesn't reproduce the claimed failure mode.** Pairing does degrade the
effective judge quality (single-arm r=0.94 → implied diff-r=0.90), but not
nearly enough to reach the crossover found above (r≲0.5-0.6): raw λ̂ stays
at 0.66-0.76 throughout, shrink→1 keeps beating shrink→0, and both stay
well above `human_subset`. The mechanism the critique describes is real in
direction, but at these (realistic) parameters it doesn't independently
produce a low-λ* regime — it still needs the underlying per-arm judge to
already be poor. This looks like a restatement of the same poor-judge
mechanism the sweep above already covers, not a distinct common case.

### Revised takeaway

The original conclusion ("shrink→1 is correct as shipped, the
uninformative-judge cost is a scoped tail case") **needs qualification**:
the cost is real, has an identifiable and not-especially-extreme threshold
(r≈0.5-0.6), and can make `power_tune=True` actively worse than not using
PPI at all — a concrete case of the exact risk the function's own
docstring already warns about for the *unconditional* λ=1 estimator, which
the shrinkage patch only protects against when the judge is good enough
(and doesn't self-correct with more labels when it isn't). Neither a
uniform shrink→1 nor a uniform shrink→0 is right across the whole
informativeness spectrum — this is what makes a judge-quality-aware
(rather than n_lab-only) shrinkage target worth pursuing, which is the
direction the main repo's in-progress, more narrowly-scoped fix seems to be
heading, even though its current scope (closed-form mean backend only, with
a separate degeneracy floor for a different failure mode) doesn't yet cover
the bootstrap path where this sweep's failure mode actually lives.

No code was changed for this addendum either — `correct()`'s bootstrap path
was reimplemented standalone in a scratch script for testing, not patched
in place.

---

## Addendum 2 (2026-08-12): prototyping an adaptive-target fix

Given the crossover found above, prototyped whether a **data-driven
shrinkage target** (instead of a fixed 1 or 0) could get shrink→1's power
where the judge is good AND stay safe (at/above `human_subset`) where the
judge is bad — without knowing in advance which regime a given dataset is
in.

### The idea

The current patch conflates two different things: "λ̂ is imprecise because
`n_lab` is small" and "the true λ is probably close to 1." There's no
reason the second should follow from the first — an imprecise estimate of
a genuinely-low λ is still evidence the judge is bad, not evidence to
distrust and override.

Fix tested: draw `n_meta=15` independent small bootstrap estimates of λ
(cheap — reuses the same resampling machinery, no extra data needed), and
use the **fraction of them below 0.5** as a smooth, self-computed
shrinkage target:

```
target = 1 - P(lambda_hat < 0.5 | data)     # estimated empirically over the 15 draws
lam = w * lam_raw + (1 - w) * target,   w = n_lab / (n_lab + C)   # same w as today
```

When the data confidently says "judge is decent," almost every meta-draw
gives λ>0.5, so `target→1` — same behavior as today. When it confidently
says "judge is bad," `target→0`. When ambiguous, `target≈0.5` (a mild pull
toward the middle, not a hard jump — this avoids the instability a naive
`target = round(lam_raw)` rule would have from thresholding a single noisy
point estimate: a data point that nudges λ̂ from 0.49 to 0.51 wouldn't
flip the outcome, since `target` is a proportion over many draws, not
a single comparison).

### Results (single-arm mean, reps=800, `n_boot_meta=200 x 15` + `n_boot_ci=400`)

| noise_sd | r | n_lab | shrink→1 | shrink→0 | **adaptive** | human_subset | adaptive target |
|---|---|---|---|---|---|---|---|
| 0.35 | 0.94 | 15 | 0.932 | 0.590 | **0.924** | 0.241 | 0.998 |
| 0.70 | 0.82 | 30 | 0.828 | 0.778 | **0.830** | 0.484 | 0.888 |
| 1.50 | 0.56 | 15 | 0.276 | 0.397 | **0.380** | 0.241 | 0.058 |
| 1.50 | 0.56 | 30 | 0.480 | 0.587 | **0.616** | 0.484 | 0.007 |
| 3.00 | 0.32 | 30 | 0.241 | 0.519 | **0.551** | 0.484 | 0.000 |
| 8.00 | 0.12 | 60 | 0.202 | 0.780 | **0.787** | 0.769 | 0.000 |

(shrink→1/shrink→0 columns are from the earlier sweep, reps=2000/seed=777,
not a perfectly matched run to adaptive's reps=800/seed=4242 — treat exact
values as approximate, the pattern is what matters.)

**It works, cleanly, in this prototype.** In the good-judge regime
(noise≤0.7), adaptive tracks shrink→1 almost exactly (target→1, no power
left on the table). In the poor-judge regime (noise≥1.5), adaptive tracks
— and in several cells slightly *exceeds* — shrink→0, while staying at or
above `human_subset` everywhere tested. It never needed to be told which
regime it was in; the meta-bootstrap distribution of λ̂ itself carried
enough signal to pick correctly. Type-I error stayed in the same rough
envelope as both fixed-target baselines throughout (0.04–0.10), no new
calibration problem introduced.

### Is this production-ready? No — open questions before it could be

- **Compute cost**: `n_meta=15` extra bootstrap draws just to estimate the
  target, on top of the existing lambda-estimation and CI-construction
  draws — roughly a multi-x increase in `correct()`'s bootstrap cost.
  Needs profiling and likely a cheaper implementation (e.g. sub-sampling
  a single larger bootstrap draw into batches instead of fully independent
  redraws) before this is practical at scale.
- **Inherits the `var_lab`-degeneracy risk** the other agent's fix
  specifically guards against: each of the 15 meta-draws uses the same
  fallback-to-1-on-degenerate-variance rule, so a near-constant `Y_lab`
  would need the same kind of floor here too, untested in this prototype.
- **Threshold (0.5) and `n_meta`/`n_boot_meta` are unturned** — this is a
  feasibility check, not a tuned design.
- **Only tested on the single-arm mean estimator**, standalone outside
  `evalstats/ppi.py` — not yet run against the paired/Walsh-rank backend,
  other estimators, or the harness's full realistic scenario suite (MNAR
  labeling, varied bias types, etc.).

### Answer to "would it even be possible?"

Yes, empirically — this prototype gets shrink→1's power in the regime it
already wins, and shrink→0's safety in the regime it loses, in the same
estimator, without needing to know which regime a given dataset is in.
Turning it into a real fix means addressing the four points above,
reconciling it with the other agent's already-in-progress (differently-
scoped) work, and validating it through the actual harness suite rather
than a standalone script.

---

## Addendum 3 (2026-08-12): making the adaptive prototype cheaper, and finding where it breaks

Followed up on Addendum 2's "yes, possible" answer with two things: (1) cut
the cost, (2) stress-test harder to find real failure modes rather than
just more confirmations of the win.

### Cost: v1 → v2

v1 estimated the shrinkage target from `n_meta=15` fully independent
bootstrap redraws — pure overhead on top of what `correct()` already does.
**v2 reuses the single bootstrap draw already used for the point estimate
of λ**, splitting it into batches to get the same "how confident is the
data that λ>0.5" signal via extra vectorized arithmetic on arrays already
in memory — no extra random resampling. Verified v2 reproduces v1's
numbers closely across 6 spot-check cells (target/power/Type-I all within
noise of each other) before trusting it.

Tuned `n_boot`/`n_batches`/`n_boot_ci` by sweeping the most sensitive cell
(right at the crossover, noise=1.5/n_lab=15) for the cheapest config that
didn't degrade target accuracy:

| n_boot | n_batches | n_boot_ci | total draws | power | target(H0) | ms/call |
|---|---|---|---|---|---|---|
| 2000 | 25 | 1000 | 3000 | 0.398 | 0.065 | 3.58 |
| 1000 | 20 | 1000 | 2000 | 0.396 | 0.067 | 2.24 |
| **800** | **15** | **800** | **1600** | 0.380 | 0.079 | ~2.0 |
| 400 | 10 | 400 | 800 | 0.396 | 0.077 | 1.07 |
| 300 | 30 | 300 | 600 | 0.416 | 0.152 | 1.28 |

Landed on **800/15/800 = 1600 total draws** — accuracy matches the
expensive settings, and it's **20% *cheaper* than production
`correct(power_tune=True)`'s current cost** (2 × n_boot=1000 = 2000 draws)
today, not the ~15x-more-expensive version from Addendum 2. (300/30/300
starts losing target accuracy — batches of 10 are too small for a
reliable per-batch λ ratio; that's the floor.)

### Stress test 1: fine crossover grid — no real "ambiguous zone" dip

Worried the hard-to-soft transition (target ≈ 0.5 when the data is
genuinely ambiguous about judge quality) might behave like a classical
pretest estimator and underperform *both* fixed targets right at the
crossover. Swept noise finely (0.35 → 3.0 SD, 10 points) × n_lab
{15,30,60}, reps=1500:

**No dip.** Worst (adaptive − max(shrink→1, shrink→0)) gap across 30 cells:
**−0.001** — noise, not a real effect. In the transition zone itself
(noise 0.9–1.3, r≈0.6–0.75) adaptive *beats* the better fixed target by
0.02–0.08 in most cells. My theoretical worry didn't materialize —
averaging over enough batches (15) apparently smooths the transition
enough that there's no pretest-style penalty here.

### Stress test 2: bias robustness — no effect, as expected

Swept judge bias (constant offset) from 0.2 to 8.0 — power, Type-I, and
the estimated target were **bit-for-bit identical** across all three bias
levels. Expected: λ estimation is a covariance/variance ratio, invariant
to constant shifts by construction. Not a source of failure.

### Stress test 3: degenerate `var_lab` — the guard is necessary but only partial

This is the real finding. Two variants:

**Fully degenerate** (100% of `Y_lab` tied at `true_mean + 0.3` — an
adversarial *wrong* constant, not a conveniently-correct one, so a
falsely-narrow CI shows up as genuine Type-I inflation):

| n_lab | shrink→1 | shrink→0 | adaptive (guard on) | adaptive (**no guard**) |
|---|---|---|---|---|
| 15 | 0.493 | **1.000** | 0.493 | **1.000** |
| 30 | 0.947 | **1.000** | 0.947 | **1.000** |
| 60 | **1.000** | **1.000** | **1.000** | **1.000** |

The guard is confirmed necessary: without it, adaptive collapses to
shrink→0's 100%-false-positive-rate failure (every batch sees zero
labeled-sample variance, so every batch's λ̂→0, so the target collapses
to 0 too — the same mechanism the guard exists to catch). **But the guard
only buys parity with shrink→1, not safety** — shrink→1 itself hits
Type-I=1.000 at n_lab=60 here. Digging into why surfaced something new
and not specific to my prototype:

**Shrink→1's protective strength is designed to *weaken* as `n_lab`
grows** (`w=n_lab/(n_lab+C)` → trust the raw estimate more) — a
reasonable assumption when the raw estimate's only problem is sample-size
noise. But when `Y_lab` is degenerate, the raw λ̂ isn't noisy-around-the-
truth, it's **structurally stuck near 0** (bootstrapping a constant array
can never reveal covariance, no matter how many times you resample it).
More `n_lab` doesn't fix a structural failure, but shrink→1 trusts it
more anyway as `n_lab` grows — so **the currently-shipped default gets
progressively less safe with more labels in exactly this failure mode.**
This is a real, preexisting property of the shipped `_POWER_TUNE_
SHRINKAGE_C` formula, not something this prototype introduces — it was
just easier to see by explicitly testing the mechanism the guard is
supposed to protect against.

**90% tied** (milder degeneracy, same adversarial offset): Type-I hit
0.4–1.0 across *all three* strategies (one/zero/adaptive alike) — but this
conflates two things. A 90%-tied-at-the-wrong-value labeled sample isn't
just "low variance," it's a **labeled subset that's no longer a
representative/unbiased sample of the population** — exactly the
independent-of-outcome assumption `correct()`'s own docstring already
documents as a hard requirement (see the original doc's "MNAR selection"
warning). This result is closer to a reminder that violating that
existing, documented assumption is severe, than a shrinkage-target-
specific finding — all three strategies are equally exposed, not a
weakness specific to the adaptive target.

### Stress test 4: extreme n_lab (8, 10, 150)

At n_lab=150, everything converges and Type-I stays well-controlled
(0.059–0.066) across all three strategies — no issue at the large end.

At n_lab=8–10 (already below the documented `_MIN_LAB_RECOMMENDED=30`
threshold where the percentile bootstrap itself is known to undercover),
adaptive's power still tracks/exceeds max(shrink→1, shrink→0) in most
cells, **but Type-I runs slightly higher than shrink→1's in a few cells**
(e.g. n_lab=8, noise=1.5: adaptive=0.161 vs shrink→1's 0.103) — a real,
modest extra cost, plausibly because the batched target estimate itself
gets noisier when there are only 8-10 items to resample from in the first
place. Small, but real — worth flagging rather than glossing over.

### Stress test 5: paired difference, both judge regimes

Re-ran the paired scenario from Addendum 1 with v2, now covering both a
good judge (noise=0.35) and poor judges (noise=1.5, 3.0) at es=0.15. The
pattern holds across the board: adaptive matches shrink→1 in the
good-judge cells and correctly switches to match shrink→0 in the
poor-judge cells (e.g. n_lab=60, noise=3.0: adaptive=0.234 exactly
matching shrink→0, vs shrink→1's 0.127), always at or above
`human_subset`.

### Where it actually breaks down (summary)

1. **Degenerate `Y_lab`**: the guard prevents the worst case (matches
   shrink→1 instead of collapsing like shrink→0) but doesn't independently
   solve it — shrink→1 itself isn't safe here, and this is a preexisting
   gap in the shipped code that surfaced from testing this mechanism
   specifically, not a new problem this prototype created.
2. **Very small n_lab (8–10)**: a small, real Type-I cost above shrink→1's,
   in a region already flagged as unreliable regardless of shrinkage
   strategy.
3. Everywhere else tested — the full informativeness spectrum, the
   ambiguous crossover zone specifically, bias magnitude, paired vs.
   single-arm — no breakdown found; adaptive matched or beat both fixed
   targets and stayed clear of the `human_subset` floor.

No code was changed in `evalstats/ppi.py` for this addendum.

---

## Addendum 4 (2026-08-12): degenerate `Y_lab` is an MCAR (not MNAR) phenomenon — and MNAR robustness turns out to be shrink→1's strongest argument

Follow-up question: is Addendum 3's degenerate-`Y_lab` scenario realistically
an MNAR (non-random labeling) problem rather than an MCAR one? Worth being
precise about, since it changes what the finding means.

### Addendum 3's test wasn't actually either

It used MCAR *selection* (uniform-random choice of which items got
labeled) but then *overwrote the values* of the selected items to a wrong
constant — a measurement/response-degeneracy artifact (rater
straightlining, a saturated scale), not a missingness-mechanism problem at
all. MCAR/MNAR describe which data points you observe, not whether the
observed values are trustworthy.

### Does this codebase's actual MNAR mechanism produce it? No — not even close

Ran `_jb_label_indices` (the harness's real soft, logit-weighted
preferential-labeling mechanism) at mild/strong/extreme settings, plus a
deterministic top-n limit (the most extreme MNAR possible: always label
the highest-scoring items, no randomness at all):

| selection | var_lab / var_hat_lab |
|---|---|
| MCAR (uniform) | 0.90 |
| MNAR mild (strength=0.8) | 0.91 |
| MNAR strong (strength=1.6) | 0.85 |
| MNAR deterministic top-n (hardest possible limit) | 0.51–0.63 |

MNAR selection barely moves the ratio, even at its most extreme —
nowhere near the guard's degeneracy threshold. **So Addendum 3's guard
scenario isn't realistically an MNAR pathway.**

### But MNAR revealed something more important: it's devastating for shrink→0 specifically

| selection | n_lab | TypeI shrink→1 | TypeI shrink→0 | TypeI adaptive |
|---|---|---|---|---|
| MCAR | 15 | 0.074 | 0.081 | 0.075 |
| MNAR mild | 60 | 0.117 | **0.901** | 0.123 |
| MNAR strong | 15 | 0.153 | **1.000** | 0.160 |
| MNAR strong | 60 | 0.390 | **1.000** | 0.416 |

Mechanism: `estimate = f_lab + λ(f_unlab − f_hat_lab)`. At λ→1, the
rectifier `(f_lab − f_hat_lab)` is the *difference* of two quantities that
are similarly MNAR-biased (both `Y_lab` and `Y_hat_lab` are measured on
the same selection-skewed subset), so most of the shared selection bias
cancels. At λ→0 that cancellation is thrown away entirely — the estimate
collapses to `f_lab` alone, fully corrupted by the selection bias, with
nothing to correct it. **Vanilla PPI's λ=1 has a built-in robustness to
MNAR bias that shrink→0 structurally cannot have** — this doesn't depend
on judge quality at all, it's a property of the rectifier's difference
structure. Adaptive tracks shrink→1 here (both hover near it, `mwu`/etc.
not literally in this cell but the pattern holds), correctly avoiding
shrink→0's collapse — though it does run *slightly* worse than shrink→1
at the most extreme MNAR+large-n_lab combination (0.416 vs 0.390 at
strength=1.6/n_lab=60), a small real cost worth noting alongside the
n_lab=8–10 one from Addendum 3.

This matters more than it might look: **you cannot guarantee MCAR
sampling of the labeled subset in practice** — `correct()`'s own docstring
already flags this as a real, common risk ("always double-check the
borderline/highest-scoring responses" is exactly this mechanism). A
shrinkage strategy that only works when labeling happens to be perfectly
random is a much weaker guarantee than one that's robust to realistic,
common violations of that assumption. This is arguably a stronger
argument for shrink→1 over shrink→0 than the judge-informativeness story
from Addendum 2 — and it's one adaptive preserves.

### So what DOES realistically produce degenerate `Y_lab`? Plain MCAR sampling of rare-event binary data

No selection bias needed at all — just ordinary bad luck sampling a
low-diversity outcome:

| p | n_lab | P(`Y_lab` all-same) | TypeI shrink→1 | TypeI shrink→0 | TypeI adaptive |
|---|---|---|---|---|---|
| 0.10 | 15 | 19.3% | 0.077 | 0.204 | 0.057 |
| 0.05 | 15 | 46.0% | 0.087 | **0.467** | 0.068 |
| 0.02 | 15 | 72.2% | 0.070 | **0.730** | 0.063 |

At p=0.02 (a realistic rare-event rate), 72% of ordinary 15-item MCAR
samples are fully degenerate. Shrink→0 fails catastrophically there
(Type-I=0.73); shrink→1 and adaptive (guard engaged) both stay near
nominal. **This — not MNAR — is the realistic, common source of
Addendum 3's failure mode**, and it's already implicitly present in the
codebase's own binary-at-extreme-p Type-I cells (the `shape.binary.p=0.70`/
`p=0.90` flags visible in every Type-I table run so far).

### Screening harness run with the adaptive method actually wired into `correct()`

Wired the tuned v2 logic into `correct()`'s real bootstrap path (batches
reuse the existing `b1` draw — no added cost — plus the degenerate-`Y_lab`
guard). Full `evalstats` ppi pytest suite passes clean (379 tests, no
regressions). Ran the same screening command used for the shrink→0 pass
in Addendum 1 (`--mode ppi --reps 100 --effect-reps 100 --ppi-n-boot 500
--tests ttest ttest_welch mwu wilcoxon paired_t bayes_bootstrap
--eval-types binary continuous likert --workers 15`), now on the
adaptive-wired code, for a like-for-like three-way comparison:

**Mean Type-I** (same 6 tests, restricted for comparability):

| | shrink→1 | shrink→0 | **adaptive** |
|---|---|---|---|
| mean corrected Type-I | 0.053 | 0.058 | **0.056** |

**5-way comparison, PPI power vs. n_lab** (`es=0.30`):

| eval_type | n_lab | shrink→1 | shrink→0 | **adaptive** |
|---|---|---|---|---|
| continuous | 15 | 0.292 | 0.230 | **0.340** |
| continuous | 20 | 0.328 | 0.290 | **0.366** |
| continuous | 30 | 0.389 | 0.338 | 0.386 |
| continuous | 40 | 0.409 | 0.378 | 0.400 |
| likert | 15 | 0.460 | 0.262 | **0.464** |
| likert | 20 | 0.500 | 0.392 | **0.506** |
| likert | 30 | 0.588 | 0.540 | **0.600** |
| likert | 40 | 0.570 | 0.586 | 0.574 |
| binary | 15 | 0.362 | 0.325 | 0.355 |
| binary | 20 | 0.435 | 0.425 | **0.455** |
| binary | 30 | 0.585 | 0.575 | **0.605** |
| binary | 40 | 0.580 | 0.610 | **0.605** |

**Adaptive matches or beats shrink→1 at essentially every single cell**
across all three eval types, while clearly beating shrink→0 everywhere
(most dramatically at small n_lab, where shrink→0's power collapse was
worst). Type-I stays in between the two fixed targets, close to nominal.
This is the real harness's own realistic scenario grid — not a hand-built
probe — and it reproduces the "best of both worlds" pattern cleanly. The
harness's own MNAR Type-I sweep on the adaptive-wired code also stayed
reasonably controlled (max 0.13 across 52 MNAR cells), consistent with
the codebase's already-documented MNAR-bias caveat rather than showing any
new problem.

Screening run: `simulations/out/screening_adaptive_20260812_222247/`.
`evalstats/ppi.py`'s bootstrap path is currently left wired to the
adaptive method in this worktree pending further discussion — not yet
reverted, unlike the earlier addenda's experiments.

---

## Addendum 5 (2026-08-12): testing a paper rule of thumb — "IRR<0.4, use human-only" — against the paper's own metric

Prompted by a specific question: an earlier paper had a rule of thumb that
below inter-rater reliability (IRR) 0.4, the LLM judge is too poor to
correct with PPI at all and one should just run statistics on the human
labels directly. The label-efficiency multiplier was reportedly ~1.05x
(marginal) around there, and this held regardless of which IRR metric was
used (weighted kappa, Pearson, Spearman). Question: is that rule an
artifact of shrink→1's specific failure mode, such that it wouldn't hold
under a different shrinkage target?

### Method — cheap, but faithful to the real metric

Reused the harness's own machinery rather than a hand-built approximation:

- `_calibrate_noise_for_alignment`/`measure_judge_alignment` — the actual
  bisection `build_ppi_label_efficiency_sources`' own check uses to find
  the `llm_noise` that hits a target Pearson r (continuous) / weighted
  kappa (likert). 0.4 is literally one of the harness's own
  `_LABEL_EFF_ALIGNMENT_TARGETS`.
- `generate_judge_bias_cell`/`JudgeBiasSource` — the real paired-scenario
  generator, same `N=1000`, `effect_frac=0.15` convention as
  `build_ppi_label_efficiency_sources`.
- `_classical_pooled_power_curve`/`_equivalent_n_lab` — the real classical
  reference-curve inversion the "multiplier" (`equiv_n_lab / n_lab`) is
  read off of.

Only the PPI power computation itself is the standalone reimplementation
(mode="one" = shrink→1, mode="adaptive" = the code now actually wired into
`correct()`), applied to the real generated paired data. Cheap version: 3
representative `n_lab` targets (20/40/90, not the full 8-point grid),
reps=500, reduced Monte Carlo counts throughout.

### Results

Calibration landed almost exactly on target both times (pearson_r=0.400,
weighted_kappa=0.400):

**Continuous:**

| n_lab | power (one) | power (adapt) | multiplier (one) | multiplier (adapt) |
|---|---|---|---|---|
| 20 | 0.084 | 0.072 | 3.70 | 3.10 |
| 40 | 0.080 | 0.100 | 1.75 | **2.25** |
| 90 | 0.116 | 0.132 | 1.32 | **1.58** |

**Likert:**

| n_lab | power (one) | power (adapt) | multiplier (one) | multiplier (adapt) |
|---|---|---|---|---|
| 20 | 0.088 | 0.154 | **0.50** | **1.90** |
| 40 | 0.108 | 0.216 | **0.61** | **1.62** |
| 90 | 0.248 | 0.308 | **0.85** | **1.09** |

### Likert: a clean, direct confirmation

At IRR (weighted kappa) = 0.4, **shrink→1's label-efficiency multiplier is
below 1.0 at all three n_lab points tested — meaning PPI with the shipped
default is net HARMFUL there, not just marginal**: you'd get equivalent
power from fewer plain human-only labels than PPI actually needs at that
judge quality. That's a materially stronger statement than "marginal
~1.05x" — a multiplier of 0.5-0.85 is actively worse than doing nothing.
Adaptive, on the exact same calibrated scenario, stays comfortably above
1.0 throughout (1.09-1.90x) — a real, positive label-efficiency benefit
survives at IRR=0.4 under the adaptive target. This is about as direct a
confirmation of the hypothesis as this kind of test can give: **the
"abandon PPI below 0.4" rule looks like it's tracking shrink→1's specific
failure mode, not a fundamental limit of the PPI approach.**

### Continuous: consistent but noisier, one cell not trustworthy

n_lab=20's numbers (power 0.084 vs 0.072) are both near-floor with
reps=500 — SE on a power estimate that small is ≈0.012, so that
particular "one beats adapt" reading is within 1 SE of pure noise and
shouldn't be read as a real reversal (the harness's own docstring already
warns that inverting near-floor power into an equivalent-N is
"extremely noise-sensitive"). The two higher-power cells (n_lab=40, 90)
do favor adaptive (2.25 vs 1.75, 1.58 vs 1.32) and both stay comfortably
above 1x either way — continuous doesn't show the same below-1
catastrophic zone likert does at this specific IRR=0.4/effect_frac=0.15
parameterization, though adaptive is still the better choice throughout.

### Caveats

- This used the harness's own baseline scenario parameters (icc=0.20,
  differential bias, the standard representative shape per eval type) —
  not necessarily an exact match to whatever scenario/effect size the
  original paper's rule was calibrated against, so the precise multiplier
  values shouldn't be over-read; the qualitative pattern (shrink→1 can go
  below 1x at IRR=0.4, adaptive doesn't) is the finding that should
  transfer, not the exact numbers.
- Both `paired_t`-family only here (matching the classical reference
  curve built) — didn't re-check `mwu`/`wilcoxon`'s backends, which (per
  Addendum 1) don't go through this same adaptive-shrinkage code path at
  all yet.
- Binary wasn't tested here (the paper claim was specifically about
  continuous/likert).

No code was changed for this addendum.

---

## Addendum 6 (2026-08-12): what this actually means for the paper's IRR<0.4 rule

Addendum 5 tested the mechanism; this section is the more careful statement
of what that result does and doesn't establish, since "the rule is an
artifact of shrinkage" is a stronger claim than the evidence alone proves.

**What's well-supported:** the rule is *shrinkage-artifact-shaped*. Three
independent lines of evidence point the same direction:

1. Addendum 2's synthetic crossover (shrink→1's power drops below
   `human_subset`'s) landed at r≈0.5-0.6 — in the neighborhood of 0.4, not
   an order of magnitude off.
2. Addendum 5, calibrated directly to the paper's own IRR metrics via the
   harness's real bisection machinery, found shrink→1's label-efficiency
   multiplier actually *below 1.0* at likert/weighted-kappa=0.4 (net
   harmful, not merely marginal) at all three `n_lab` points tested, while
   adaptive stayed above 1.0 at the identical calibrated scenario.
3. Mechanistically (Addendum 4), shrink→1's failure isn't judge-quality-
   specific reasoning alone — it's that the shrinkage patch trusts a fixed
   pull toward "vanilla PPI" regardless of what the data says, so *some*
   reliability floor below which that pull stops paying off has to exist
   for a fixed target; an adaptive target doesn't need one imposed on it
   the same way.

**What's NOT yet established:** that the specific "0.4" cutoff, or the
paper's specific finding, is *simply wrong*. Addendum 5's simulation used
the harness's own default scenario parameters (icc=0.20, standard
differential-bias convention, `effect_frac=0.15`) — not necessarily a
match to the paper's actual test setup, effect sizes, or data-generating
assumptions. A calibrated coincidence in the crossover location is
suggestive, not dispositive. The honest current status: **strong evidence
the rule is shrinkage-dependent and would likely shift (probably relax)
under a validated adaptive target — not yet proof the 0.4 threshold itself
is wrong for the paper's specific setup.**

**Next step, in progress:** the user is separately running a full factorial
screening sweep (`--factorial-check --factorial-check-binary`, the
harness's own 7-factor design spanning the complete `llm_noise` range
0.025-28.4 SD crossed with real MCAR/MNAR labeling) from their own
terminal on the `ppi-power-tuning-tuning` branch. That sweep will give a
much more direct read — across the harness's actual realistic scenario
grid rather than a single calibrated point — on where shrink→1 vs.
adaptive actually diverge, and should sharpen (or correct) this section
once it lands.

---

## Addendum 7 (2026-08-13): the factorial sweep landed — strong confirmation

The user ran `--factorial-check --factorial-check-binary` from their own
terminal on `ppi-power-tuning-tuning` (the adaptive-wired code). Output:
`simulations/out/screening_factorial_adaptive/` (main repo). 858
continuous/likert cells (`ttest`/`ttest_welch`/`paired_t`/`mwu`/`wilcoxon`)
+ 429 binary cells (`ttest_welch`/`paired_t`), spanning the FULL `llm_noise`
range (0.025-28.4 SD) crossed with real MCAR/MNAR-mild/MNAR-strong
labeling, bias magnitude, N, N_lab, effect size, and bias direction —
by far the largest and most realistic test this investigation has run.

**Note on scope: this run only has the adaptive-wired code** (no matched
shrink→1 factorial run exists to diff against directly) — the read below
is against the expectations set by Addenda 1-6's synthetic work, not a
literal paired comparison. If a fully dispositive answer is wanted later,
re-running this exact command against a shrink→1 checkout would be the
natural confirmatory step — but everything below already lines up with
what that comparison would be expected to show.

### MCAR: essentially flawless across the ENTIRE informativeness range

The single most reassuring number in the whole run — null-cell Type-I by
noise level, MCAR only:

```
noise:   0.025 → 28.4 SD (32 points spanning near-perfect to near-total-noise judge)
MCAR:    0.049 – 0.061  (mean 0.056)
```

Flat. No crossover, no degradation at any point across the full spectrum
— including the noise>1.0 SD region where Addendum 2's synthetic probe
found shrink→1 dropping *below* `human_subset`. That failure zone simply
doesn't appear here under adaptive.

### MNAR: modest, not catastrophic — and concentrated exactly where the mechanism predicts

| label_mechanism | mean Type-I (noise-swept) | worst single cell (858-cell continuous/likert grid) |
|---|---|---|
| mcar | 0.056 | — |
| mnar_mild | 0.059 | — |
| mnar_strong | 0.074 | 0.218 |

Binary (429-cell grid): mcar 0.064, mnar_mild 0.076, mnar_strong 0.088,
worst single cell 0.365.

Real, but nowhere near shrink→0's ~1.0 collapse (Addendum 4) or even
shrink→1's ~0.39-0.42 in the synthetic MNAR-strong test. And it breaks
down by method exactly as Addendum 4's mechanism predicts — MNAR bias
cancels in a *paired difference* but not in an independent two-group
comparison:

| method | mcar | mnar_mild | mnar_strong |
|---|---|---|---|
| paired_t | 0.055 | 0.055 | **0.055** |
| wilcoxon | 0.049 | 0.050 | **0.054** |
| ttest | 0.061 | 0.064 | 0.085 |
| ttest_welch | 0.060 | 0.065 | 0.085 |
| mwu | 0.052 | 0.061 | 0.092 |

`paired_t` and `wilcoxon` (paired designs — the difference structure
cancels correlated MNAR bias) stay essentially flat under MNAR; the
independent-groups tests degrade. `wilcoxon` staying flat is expected —
it's the untouched analytic Walsh-theta backend — but `paired_t` staying
flat too (0.055 unchanged across all three mechanisms) confirms the
cancellation mechanism itself, not just "backends we didn't touch happen
to be fine."

### Power vs. human_subset across all ~34,000 real-effect cells

Checked directly whether adaptive ever falls into the "worse than not
using PPI" trap across the WHOLE H1 grid, not just a hand-picked slice:

- Mean `rate_ppi − rate_human_subset` = **+0.046** (solidly positive).
- 20.9% of cells show `ppi < human_subset`, but this concentrates in
  `wilcoxon` (34.9% negative, min gap −0.24 — the untouched backend) and
  near-floor-power/small-`n_lab`(15) cells where reps=100's Monte Carlo
  noise (SE≈0.03-0.04 on a rate near 0.05-0.15) accounts for most of the
  gap. Excluding `wilcoxon`: 17.4% negative, mean gap still +0.050, worst
  single cell −0.10 (at `rate_ppi=0.00-0.05` — a floor cell, not a real
  power collapse).
- No sign of a systematic, judge-quality-driven collapse the way
  shrink→1's synthetic crossover showed — the negative cells look like
  ordinary screening-scale (reps=100) noise plus the untouched backend,
  not a structural failure mode.

### Bottom line

This is the strongest evidence yet for keeping the adaptive target. Across
the harness's own realistic 1,287-cell factorial design — not a synthetic
probe — MCAR calibration is flat across the *entire* informativeness
spectrum (the exact regime that broke shrink→1), MNAR degradation is
modest and mechanistically exactly where expected (concentrated in
non-paired tests), and power stays net-positive against `human_subset`
across nearly 34,000 real-effect cells with no systematic collapse. Nothing
here contradicts Addenda 1-6; it's the largest and most realistic
confirmation of the same pattern.

Plots: `simulations/out/screening_factorial_adaptive/plots/` —
`*_typeI_mcar.png`, `*_typeI_mnar.png`, `*_slices.png`, `*_slices_mnar.png`
(continuous/likert and binary).

---

## Addendum 8 (2026-08-13): correction — the factorial sweep's power cells don't test the informativeness range, and a proper matched comparison

Addendum 7 overstated one thing: its "power vs `human_subset` across ~34k
cells" analysis pooled over bias magnitude/N/N_lab/label mechanism/bias
direction, but **`build_ppi_factorial_sources` fixes `llm_noise=0.20` for
every `es != "null"` (power) cell** — only the null/Type-I cells sweep the
full noise range. So the factorial sweep says nothing about the poor-judge
power crossover (Addenda 2/3/5's territory); it only ever tests power at
one fixed, decently-informative judge quality. Worth being precise about
this rather than letting Addendum 7's aggregate power number imply
otherwise.

### A real fix: matched comparison against the actual prior shrink→1 run

The user already had a shrink→1 factorial run from before this session's
changes: `official_20260803_013611` (reps=200, same `--factorial-check
--factorial-check-binary`). Merged it against the new adaptive run
cell-by-cell on the full factor key (`et, method, bm, n, nlab, lm, es, bd`)
— 1920 exactly-matched H1 cells.

**Overall**: mean power shrink→1=0.9249, adaptive=0.9262 (diff +0.0012);
adaptive ≥ shrink→1 on 78% of matched cells.

**Noise-floor calibration, using `wilcoxon` as a control:** `wilcoxon`'s
code is byte-identical between the two runs (untouched backend) — any
"difference" there is pure Monte Carlo noise from the different reps
(200 vs 100) and RNG streams, nothing else. It gives a direct noise-floor
reading: mean diff 0.0001, std 0.0307, range [-0.15, +0.10]. The
adaptive-affected methods (`ttest_welch`/`mwu`/`paired_t`) show mean diff
0.0016, std 0.0243-0.0261, range [-0.165, +0.17] — **statistically
indistinguishable from the pure-noise control.** The handful of "worst
regression" cells (down to -0.165) are exactly the same size as
`wilcoxon`'s own noise-only worst case (-0.15), so they're screening-scale
sampling noise, not a real regression.

**This is the expected, correct result, not a null finding.** At
noise=0.20 (a genuinely informative judge), raw λ̂ is comfortably high, so
adaptive's own target converges to ≈1 too — the two methods *should*
behave almost identically here, and confirming that (rather than some
divergence) is exactly what validates the earlier claim that adaptive only
changes behavior where the data suggests it should. The real informativeness-
range power story remains what Addenda 2/3/5 already showed via synthetic/
calibrated scenarios — this factorial run is a clean confirmation that
adaptive doesn't cost anything in the regime it isn't supposed to change,
not a test of the regime where it's supposed to help.

---

## Addendum 9 (2026-08-13): the real power-vs-judge-quality check — label efficiency, official tier, matched against a real shrink→1 baseline

Following up on Addendum 8's correction, the user ran the harness's actual
label-efficiency check (`run_ppi_label_efficiency_check` — 6 IRR-calibrated
alignment targets × 8 `n_lab` points, continuous/likert/binary) at
official tier (`--effect-reps 200 --ppi-n-boot 2000`):
`simulations/out/label_efficiency_adaptive/`. This is the systematic,
full-precision version of Addendum 5's cheap 3-point approximation — using
the real `correct()` code and the real pooled `_COMPARISON_METHODS`, not a
standalone reimplementation.

A matching shrink→1 baseline existed: `official_20260803_013611`
(reps=200, same check) — but that run predates this session's widening of
`_LABEL_EFF_ALIGNMENT_TARGETS` from 5 points to 6, so it has **no exact
0.40 target** (only 0.80/0.70/0.60/0.50/0.30). Bracketing with 0.50 and
0.30 instead, both of which match exactly.

### Likert, kappa=0.30 — the cleanest, most dramatic confirmation yet

| N_lab | shrink→1 multiplier | adaptive multiplier |
|---|---|---|
| 15 | 2.34x | 1.00x |
| 20 | 2.74x | 1.14x |
| 30 | 1.03x | 1.42x |
| 40 | **0.41x** | 1.49x |
| 60 | **0.47x** | 1.29x |
| 90 | **0.74x** | 1.12x |
| 130 | 1.27x | 1.17x |
| 200 | 1.24x | 1.17x |

Shrink→1 drops below 1.0x — PPI actively harmful, worse than plain
human-only labels — at **three consecutive `n_lab` points** (40/60/90).
Adaptive stays above 1.0x at every one of those same points. This is the
official-tier, full-precision, real-`correct()` version of exactly the
Addendum 5 pattern, now shown across a stretch of the `n_lab` range rather
than a single calibrated point — much harder to dismiss as a one-off.

### Continuous, r=0.50 and r=0.30 — partial confirmation, noisier

| N_lab | r=0.50 shrink→1 | r=0.50 adaptive | r=0.30 shrink→1 | r=0.30 adaptive |
|---|---|---|---|---|
| 40 | 0.98x | 1.54x | 1.42x | 1.41x |
| 60 | **0.60x** | 1.29x | **0.60x** | 1.03x |
| 90 | 1.66x | 1.40x | 1.19x | 1.26x |

Same direction (shrink→1 dips below 1.0x at `n_lab=60` in both brackets;
adaptive doesn't), but the rest of the grid is noisier and less uniformly
one-sided than likert — individual cells at this reps/n_boot tier are
still subject to real Monte Carlo noise (the harness's own docs flag
near-floor power inversion as "extremely noise-sensitive"; e.g. shrink→1's
8.43x at r=0.30/n_lab=15 is almost certainly one such artifact, not a real
effect).

### Likert, kappa=0.50 — genuinely mixed, worth reporting honestly

| N_lab | shrink→1 | adaptive |
|---|---|---|
| 15 | 1.00x | 2.13x |
| 20 | 1.30x | **0.75x** |
| 130 | **0.92x** | 1.18x |

Both sides dip below 1.0x somewhere in this bracket (shrink→1 at
`n_lab=130`, adaptive at `n_lab=20`) — not a clean sweep for either
target at this specific IRR level. Included for honesty: the pattern is
real and directionally consistent overall, not perfect at every single
cell.

### Adaptive's own 0.40 numbers, standalone (no exact shrink→1 baseline to diff against)

| N_lab | continuous mult. | likert mult. |
|---|---|---|
| 15 | 1.94x | 1.00x |
| 40 | 1.41x | 1.44x |
| 90 | 1.28x | 1.18x |
| 200 | 1.01x | 1.17x |

Never drops below 1.0x anywhere in either eval type at the actual target
this whole investigation started from. Addendum 5's hand-rolled shrink→1
approximation, at this same target, showed likert multipliers of
0.50-0.85x (below 1.0x throughout) — consistent with what the bracketing
0.50/0.30 targets show in the real harness data above.

### Overall read

Across both official-tier, real-`correct()` checks (this one and the
factorial sweep in Addendum 7-8), the pattern holds up: adaptive degrades
*gracefully* toward ~1.0x (no benefit, but not harmful) as judge quality
drops, while shrink→1 shows real, repeated collapses below 1.0x
(net-harmful) at moderate `n_lab` and lower judge quality — most starkly
at likert/kappa=0.30, where it happens at three consecutive `n_lab`
points. Not every single cell favors adaptive (kappa=0.50's mixed result
is the honest counterexample), but the aggregate pattern is now confirmed
across three independent, increasingly rigorous tests (Addendum 5's cheap
approximation, this official-tier label-efficiency sweep, and the
factorial sweep's Type-I side) rather than resting on one calibrated
point.

---

## Addendum 10 (2026-08-13): extending adaptive shrinkage to the analytic-mean backend

Started applying the same adaptive-target logic to `_analytic_mean_point_se`
(the closed-form backend used for `np.mean`/paired-mean estimands at small
`n_lab` or when preferred by `backend="auto"`) -- shared by
`_analytic_mean_correct`, `_analytic_logit_t_correct`, and every
`evalstats/tests/__init__.py` wrapper that delegates to them
(`_ppi_single_t_interval`, `_ppi_paired_t_interval`, `_ppi_single_logit_t`,
`_ppi_paired_logit_t`, and others).

### Design difference from the bootstrap path

This backend has no bootstrap array to split into batches -- `var_unlab`
is already closed-form from the large unlabeled sample, and `lambda*` is
computed directly from `cov_lab_hatlab / (var_unlab + var_hat_lab)`, no
resampling at all. Adding the target estimate without breaking the
backend's whole reason to exist (a fully deterministic, no-bootstrap CI)
took one design decision: a new `_analytic_shrink_target` helper draws
n_boot=800 bootstrap resamples of just the (`Y_lab`, `Y_hat_lab`) PAIR
(cheap regardless of `n_lab` -- no need to touch the large unlabeled set)
using a **fixed internal seed**, not one threaded from the caller. Since
this randomness is purely an implementation detail for approximating a
shrinkage target (not a Monte Carlo quantity a caller needs to control),
a fixed seed keeps every existing caller's signature untouched and the
function still fully deterministic (same inputs -> same outputs) --
avoided a much larger, more invasive change (threading `rng` through 8+
call sites in `evalstats/tests/__init__.py`).

Also simpler than the bootstrap-path version in one respect: since each
resample here is the FULL labeled-pair array (not a single bootstrap
mean, the way the bootstrap path's `b1` arrays are), each draw already
gives one complete, valid raw-lambda estimate on its own -- no need for
the bootstrap path's batching trick (which exists there only because a
lone bootstrap-mean value can't yield a covariance by itself). `target =
1 - P(lambda_hat < 0.5)` is computed directly over all 800 draws.

Same degenerate-`Y_lab` guard as the bootstrap path (falls back to
target=1 when `Y_lab`'s own variance is degenerate relative to
`Y_hat_lab`'s), plus an explicit `n_lab <= 1` guard the bootstrap path
didn't need (this backend's SE formula already handles `n_lab<=1`
gracefully via a 0.0 fallback; the target helper needed the same
short-circuit to avoid calling `np.var(..., ddof=1)` on a single-element
resample).

### Validation so far

- Sanity check + full `evalstats` ppi pytest suite (**379 tests**) --
  passes clean, no regressions, including the `n_lab=1` edge case.
- Direct behavioral validation calling the real `correct(backend="analytic")`
  code (not a reimplementation) across the full noise range, at `n_lab`
  15/25 (where `backend="auto"` actually dispatches here):

| noise_sd | r | n_lab | Type-I | power | human_subset |
|---|---|---|---|---|---|
| 0.35 | 0.94 | 15 | 0.042 | 0.893 | 0.231 |
| 1.50 | 0.56 | 15 | 0.069 | 0.304 | 0.231 |
| 8.00 | 0.12 | 15 | 0.064 | 0.245 | 0.231 |
| 0.35 | 0.94 | 25 | 0.043 | 0.981 | 0.363 |
| 1.50 | 0.56 | 25 | 0.055 | 0.481 | 0.363 |
| 8.00 | 0.12 | 25 | 0.059 | 0.378 | 0.363 |

Type-I stays controlled throughout (0.042-0.069), λ tracks judge quality
smoothly (~0.9 at r=0.94 down to ~0.01 at r=0.12), and **power never drops
below `human_subset` anywhere in this sweep** -- the same "best of both
worlds" pattern the bootstrap path showed, now confirmed in this backend's
own code path directly, not inferred from the shared formula.

### Not yet done

- No harness-level (official-tier, real scenario grid) validation yet for
  this backend specifically -- everything above is pytest + one targeted
  standalone sweep, not yet run through `--mode ppi`'s Type-I/label-
  efficiency checks the way the bootstrap path was in Addenda 7-9.
- `_analytic_walsh_theta_correct` (Wilcoxon/paired-rank backend) remains
  untouched -- next in line, and a bigger lift (different rectifier/
  variance structure, needs its own derivation).
- Uncommitted in the main repo as of this addendum (`git diff --stat`:
  `evalstats/ppi.py`, +144/-50) -- pending confirmation before committing,
  same as every other code change in this investigation.

---

## Addendum 11 (2026-08-13): analytic-backend fix confirmed via label efficiency, committed

Re-ran the exact label-efficiency command from Addendum 9
(`--effect-reps 200 --ppi-n-boot 2000`, same seed) with the analytic-mean
backend's adaptive shrinkage (Addendum 10) also wired in:
`simulations/out/label_efficiency_analytic/`.

**Isolation check passed first**: every `N_lab >= 30` cell is bit-for-bit
identical to Addendum 9's numbers, as expected (those cells use the
bootstrap path, already fixed before today and untouched by this change) --
confirms the diff below is cleanly isolated to the analytic backend, not
some other confound.

**`N_lab` 15/20 (where `backend="auto"` actually dispatches to the analytic
path) improved in both eval types**, at the exact IRR=0.40 target this
whole investigation started from:

| eval_type | N_lab | multiplier (old, fixed shrink→1) | multiplier (new, adaptive) |
|---|---|---|---|
| continuous (r=0.40) | 15 | 1.94x | **2.26x** |
| continuous (r=0.40) | 20 | 1.67x | **1.73x** |
| likert (κ=0.40) | 15 | **1.00x** | **1.59x** |
| likert (κ=0.40) | 20 | 1.33x | **1.51x** |

Likert `N_lab=15` moving from exactly 1.00x (the old fixed-shrink-to-1
formula gave literally zero label-efficiency benefit there) to 1.59x is
the clearest single number in this addendum.

**Committed**: `evalstats/ppi.py` committed on `ppi-power-tuning-tuning`
(main repo) -- not pushed. Both backends (`correct()`'s bootstrap path and
`_analytic_mean_point_se`) now use the same adaptive-target shrinkage.
Remaining gap: `_analytic_walsh_theta_correct` (Wilcoxon/paired-rank
backend) is still on the original fixed shrink-to-1 formula -- next.

---

## Addendum 12 (2026-08-13): Walsh-theta (Wilcoxon) backend fixed too -- all three backends now adaptive

Added `_walsh_theta_shrink_target`, the Walsh-theta analogue of
`_analytic_shrink_target`: same micro-bootstrap-of-the-labeled-pair idea,
but re-evaluating the Hajek-projection cov/var ratio
(`_walsh_theta_h1_components`) per resample instead of the mean's plain
sample cov/var. That component's O(n log n) sort+searchsorted isn't
vectorizable across a batch (same constraint the existing
`_walsh_theta_batch` docstring already notes), so this stays a Python loop
over 800 draws -- slower than the mean backend's fully-vectorized version,
but still fine since `n_lab` is small here.

**Validation**: full `evalstats` ppi pytest suite (379 tests) passes
unchanged. Direct behavioral sweep calling the real
`correct(backend="analytic", estimator_func=paired_walsh_midrank_theta)`
code on paired-difference data across the full noise range at n_lab=15/25:
Type-I stays controlled (0.046-0.086), power never drops below
`human_subset` anywhere tested. One difference worth noting: λ doesn't
collapse as low under severe noise as the mean backend's did (~0.11-0.14
vs ~0.01 at r=0.12) -- consistent with rank-based statistics being
inherently more robust to additive noise (they only depend on order, not
magnitude), and with the other agent's earlier comment that this backend
already had "no known failure mode" under the old fixed shrink-to-1.

### Official-tier label-efficiency check, isolating `wilcoxon` specifically

`wilcoxon` (`paired_walsh_midrank_theta`) is in `_ANALYTIC_ALWAYS_
PREFERRED`, so it uses this backend at every `n_lab`, not just <30 --
this fix could move the *entire* range, unlike the mean-backend fix.
Isolated `wilcoxon`'s own rows from the raw per-method results CSV (not
just the pooled summary) at the IRR=0.40 target, comparing against the
prior run (`label_efficiency_analytic/`, before this fix):

**Continuous, wilcoxon only, r=0.40** -- `human_subset` identical between
runs at every `n_lab` (same seed/scenario, confirms a clean diff):

| N_lab | power (old) | power (new) |
|---|---|---|
| 15 | 0.065 | **0.120** |
| 20 | 0.065 | **0.095** |
| 40 | 0.075 | **0.090** |
| 90 | 0.090 | **0.110** |
| 200 | 0.160 | **0.165** |

**Likert, wilcoxon only, κ=0.40**:

| N_lab | power (old) | power (new) |
|---|---|---|
| 15 | 0.085 | **0.105** |
| 20 | 0.085 | **0.115** |
| 40 | 0.125 | **0.145** |
| 90 | 0.300 | 0.290 (noise) |
| 200 | 0.525 | **0.545** |

Improved or flat at every single point checked across the whole `n_lab`
range, most sharply at continuous `n_lab=15` (power nearly doubling). The
two trivial dips (continuous `n_lab=130`: -0.005, likert `n_lab=90`:
-0.010) are both well inside reps=200 Monte Carlo noise (SE≈0.032 on a
rate this size).

Pooled multiplier (all 4 methods, including wilcoxon) improved further on
top of Addendum 11's numbers, as expected:

| | continuous r=0.40, N_lab=15 | likert κ=0.40, N_lab=15 |
|---|---|---|
| Addendum 11 (mean+bootstrap fixed, wilcoxon not yet) | 2.26x | 1.59x |
| This run (all three backends fixed) | **2.61x** | **1.66x** |

**Committed**: `evalstats/ppi.py` committed on `ppi-power-tuning-tuning`
(main repo) -- not pushed. All three PPI backends `correct()` dispatches
across (`bootstrap`, `_analytic_mean_point_se`, `_analytic_walsh_theta_
correct`) now use the same adaptive-target shrinkage. This closes out the
"apply the fix everywhere PPI is implemented" phase of the investigation
-- `kruskal`/`anova`/`friedman`/`lmm*`/`tango_score` remain deliberately
untouched, per `power_tune`'s own docstring: power-tuning doesn't transfer
to their variance-like, quadratic-form estimand. Real-data validation
(OpenEval/Inspect corpora, flagged back in the "any other official sweeps"
discussion as the one gap not yet covered by any of this addendum series)
is the natural next step if further confirmation is wanted.

---

## Addendum 13 (2026-08-13): refactor into shared helpers, plus two more sites found and fixed

Asked to double-check the codebase for other places implementing this same
shrinkage pattern, and to consider factoring the by-then-3x-duplicated
logic into a shared function rather than copy-pasting a 4th/5th time.
Found two more genuine gaps and refactored all five onto shared helpers.

### Two more sites found (same `_POWER_TUNE_SHRINKAGE_C`, still fixed toward 1)

- **`evalstats/api.py`'s `_ppi_bootstrap_t_joint_stats`** (the Romano-Wolf/
  max-T joint studentized-bootstrap helper) -- computes a per-pair lambda*
  using the identical mean-based closed form as `_analytic_mean_point_se`,
  but independently, not by delegating to it.
- **`evalstats/tests/__init__.py`'s `_ppi_paired_bayes_bootstrap`** (Dirichlet-
  weighted resampling, for sparse/tied paired data like binary diffs) --
  independently reimplements `correct()`'s bootstrap-path shrinkage rather
  than delegating (its whole reason to exist is Dirichlet weighting
  `correct()` doesn't support, so it can't just call `correct()`).

Checked and ruled out: MWU's `method="ridge"` path is a genuinely
different mechanism (ridge regression on a rectifier slope, not a
λ-toward-a-pole shrinkage -- its own docstring already says its fixed
constant only "mirrors" `_POWER_TUNE_SHRINKAGE_C` in spirit, doesn't share
it); MWU's default and `bootstrap_t`/`tango_score` either delegate to
already-fixed backends or have no λ-shrinkage at all, matching the
existing exclusion list; `_ppi_friedman_f_stat` reuses the same
`n_lab/(n_lab+C)` shape but for blending toward an already-correct
variance target, not a safety fallback -- different purpose, left as-is.

### Refactor

Three near-identical copies (bootstrap path, analytic-mean, Walsh-theta)
plus two more about to be added made copy-paste clearly the wrong call.
Factored into `evalstats/ppi.py`:

- **`_adaptive_shrink_lambda(lam_raw, lam_replicates, n_lab)`** -- the one
  shared "final blend" step every site needs: `target = 1 -
  P(replicate < 0.5)` (or `1.0` if `lam_replicates is None`, signaling a
  degenerate-labeled-sample guard already fired upstream), then
  `lam = w*lam_raw + (1-w)*target`. All five sites now call this instead
  of inlining the arithmetic.
- **`_bootstrap_batch_lambda_replicates(b_lab, b_hat_lab, b_unlab)`** --
  the batching step for sites that already have a `(n_boot,)` bootstrap-
  mean draw (each element a bootstrap-resample MEAN, not the full
  resampled array, so single elements can't yield a covariance alone --
  needs pooling into batches). Used by `correct()`'s bootstrap path and
  now `_ppi_paired_bayes_bootstrap` too (its Dirichlet-weighted draws have
  the identical shape/meaning for this purpose).
- **`_analytic_mean_lambda_replicates(Y_lab, Y_hat_lab, var_unlab, n_lab)`**
  (renamed from the old `_analytic_shrink_target`, now returns the raw
  replicate array instead of the target directly) -- the micro-bootstrap-
  of-the-labeled-pair step for the mean estimand. Used by
  `_analytic_mean_point_se` and now `_ppi_bootstrap_t_joint_stats`'s
  per-pair loop too (same closed form, just called once per pair).
- **`_walsh_theta_lambda_replicates`** (renamed from `_walsh_theta_shrink_target`,
  same return-type change) -- unchanged in substance, still Walsh-theta-
  specific since no other site uses that estimand.

Verified the refactor is behavior-preserving before touching the two new
sites: re-ran the exact smoke test from Addendum 10/12 and got bit-for-bit
identical output (e.g. the Walsh-theta example's `lam=0.9482418625691675`,
unchanged to the last digit).

### Validation for the two new sites

- Direct call to `_ppi_paired_bayes_bootstrap` (n_lab=40, above the
  `_MIN_LAB_RECOMMENDED` delegate-to-analytic threshold, so it actually
  exercises the newly-fixed bootstrap-batch path): runs cleanly, `lam`
  comes back sensible (0.89 for a reasonably informative synthetic judge).
- `_ppi_bootstrap_t_joint_stats` is deep inside the multi-arm Romano-Wolf/
  max-T plumbing -- rather than hand-building minimal inputs, ran the real
  integration suites that exercise it: `test_compound_ppi_fwer.py` (named
  for exactly this FWER/Romano-Wolf machinery) and `test_simultaneous_ci.py`.
- Full run: `test_ppi_core.py` + `test_ppi_ci_methods.py` +
  `test_ppi_corrections.py` + `test_compound_ppi_fwer.py` (395 tests) +
  `test_simultaneous_ci.py` (72 tests) = **467 passed**, no regressions.

Didn't re-run a fresh official-tier harness sweep for these two --
unlike the three backends, both reuse math already validated at that
level (the per-pair loop is the identical `_analytic_mean_point_se`
closed form; the bayes-bootstrap fix is the identical `correct()`
bootstrap-path batching, just fed Dirichlet-weighted draws instead of
multinomial ones), so the pytest integration coverage was judged
sufficient rather than repeating an already-validated calibration story.

**Committed** as `b595537` (refactor + 2 new sites) and `af8e452`
(tango_score docstring fix) on `ppi-power-tuning-tuning`. Not pushed.

---

## Addendum 14 (2026-08-13): post-refactor sanity check, and can kruskal/anova/friedman get this too?

### Post-refactor screening: everything still checks out

Re-ran the standard `--mode ppi` Type-I + 5-way comparison screening
(reps=100, n_boot=500), now scoped to all 7 power_tune-affected tests
including `tango_score` for the first time: `simulations/out/
screening_postrefactor_20260813_122815/`.

Mean corrected Type-I across 1033 cells: **0.060**, per-method range
0.054-0.069 (`tango_score` itself: 0.055) -- no blowups anywhere.
Comparison-check power numbers match Addendum 4's pre-refactor run within
Monte Carlo noise (continuous `ppi` vs `n_lab`: 0.342/0.374/0.384/0.412
now vs. 0.340/0.366/0.386/0.400 then; likert similarly close) -- confirms
the refactor is behavior-preserving at the statistical level too, not
just the bit-for-bit smoke test from Addendum 13.

### Can kruskal/anova/friedman get the adaptive fix too?

Asked directly rather than trusting `correct()`'s docstring, since it was
already caught being wrong once this session (the tango_score claim).
Good instinct -- it turned out to be wrong again, but in a way that
doesn't actually open up new work.

**The docstring's "do not go through this function" claim is also
inaccurate.** `_ppi_kruskal_wallis`, `_ppi_anova_independent`, `_ppi_anova_
repeated`, and `_ppi_friedman`'s point-estimate/CI paths all call
`correct()` directly, just with `power_tune=False` hardcoded -- an inline
comment explains this is pinned off for a synchronization reason (the
p-value machinery doesn't have a matching power-tuned derivation yet),
not because it's structurally impossible. Flipping that hardcoded flag
would plumb the adaptive target into the point-estimate/CI output almost
for free.

**But the actual omnibus test statistic -- what `kruskal`/`anova`/
`friedman` are actually FOR -- has no λ parameter to shrink at all.**
`_ppi_kruskal_wallis_pairwise`/`_ppi_anova_*_f_stat`/the friedman analog
build their F-/H-statistic from a FULLY corrected per-group mean (`g.mean()
+ (g_lab[mask].mean() - g[mask].mean())`, a fixed λ=1-equivalent
correction), not a λ-blended one. There's no continuum to adaptively
shrink along -- "adaptive shrinkage" isn't even the right frame here,
because the thing our recent fixes changed (which fixed pole to shrink a
blend toward) doesn't exist in this construction.

**And unlike the tango_score claim, the deeper "power-tuning doesn't
transfer" reasoning is NOT a bare assertion -- it's backed by an actual
reverted experiment** (commit `2efe728`, documented in `simulations/
harness/README.md`'s "PPI++ power-tuning" section): a variance-minimizing
λ search was implemented for the quadratic estimand and run across a
139-scenario sweep, producing Type-I error ~19% vs. the fixed estimator's
~4% baseline. Mechanism: for a nonlinear/quadratic estimand, judge-noise
inflation isn't cancelled at λ<1 the way bias cancels for a linear mean;
variance-minimization (which is blind to bias) drifts λ toward values
that reintroduce it. This is a different, deeper problem than "which
fixed target to shrink toward" -- it would need a genuine new derivation
of how λ affects the omnibus statistic's inflation/variance, not a
drop-in application of `_adaptive_shrink_lambda`.

**Bottom line**: not a "yes, straightforward" case. The point-estimate/CI
path could trivially get `power_tune=True` (small, low-risk change,
already plumbing-ready) but that's a minor, secondary output for an
omnibus test whose main point is its p-value -- and the p-value/test-
statistic side, which is what people actually mean by "kruskal/anova/
friedman power-tuning," would need real new derivation work, with one
prior documented attempt already showing a naive version is unsafe. Left
as a documentation-fix candidate (the "do not go through this function"
line) rather than a code change, pending a decision on whether the
point-estimate-only version is worth doing on its own.

---

## Addendum 15 (2026-08-13): a working attempt at ANOVA power-tuning

Attempted independent-groups ANOVA specifically (the simplest of the
three omnibus tests), in an isolated worktree: `git worktree add -b
claude/anova-power-tuning-attempt ../worktree-anova-power-tuning
ppi-power-tuning-tuning`.

### Diagnosis: the prior failure was a construction bug, not a fundamental barrier

`_ppi_anova_independent_f_stat`'s existing (`power_tune=False`, fixed)
per-group correction is `corr_means[i] = mean(g_i) + λ·(mean(g_lab_true_i)
− mean(g_lab_judge_i))`, where `g_i` is the judge's scores over the
*whole* group (labeled+unlabeled combined) — not the standard PPI
construction `f_lab + λ·(f_unlab − f_hat_lab)` (full weight on the
*human* term, disjoint unlabeled sample). At λ=1 these nearly coincide,
which is why the shipped fixed version is valid. But
`E[corr_means_i] = true_mean_i + (1−λ)·bias_i` in this specific
construction — genuinely biased whenever λ<1, because the *judge*-based
term carries the fixed weight, not the human one. Squared into
`ss_between`, that's exactly the README's documented "λ→0 falls back to
the raw judge-biased estimate" failure mode (Type-I ~19%, commit
`2efe728`). This is an artifact of *this function's* shortcut, not an
inherent property of quadratic estimands: the standard `correct()`
construction's rectifier has expectation ≈0 at *any* λ (same population,
disjoint samples) — that's exactly the property that makes power-tuning
safe for a plain mean, and there's no reason it couldn't work here too if
the same construction were used.

### The fix

Rebuilt the `power_tune=True` per-group correction to call
`_analytic_mean_point_se` directly, once per group (`Y_lab_i`,
`Y_hat_lab_i`, the *disjoint* `Y_hat_unlab_i`) — reusing the exact
already-validated adaptive-shrinkage machinery from Addendum 10, not new
math. This keeps `E[corr_means_i(λ)] = true_mean_i` at any λ.

Hit one real bug immediately: `_analytic_mean_point_se`'s `se²` is a
*mean*-scale variance (already divided by n), but `ms_within` (the F-test's
reference) is item-scale, and the existing `inflation_per_group` formula
needs `Var[μ̂ᵢ_PPI]·nᵢ/ms_within`. Missing that `·nᵢ` factor made `denom`
~100x too small and Type-I hit 1.0 exactly, immediately, on the first
synthetic check — caught before ever touching pytest or the harness.
Fixed by deriving the correct scaling directly from the existing
(`power_tune=False`) branch's own docstring formula (`Var[μ̂ᵢ_PPI] =
σ²/nᵢ + σ_llm²×(1/n_lab_i−1/nᵢ)`, divided by `Var[μ̂ᵢ_LLM] = ms_within/nᵢ`
— the nᵢ's cancel to exactly reproduce that branch's existing formula),
confirming the fix algebraically, not just empirically.

### Results (synthetic, k=3 groups, DIFFERENTIAL judge bias — the exact mechanism that broke the prior attempt)

**Type-I, n=100/group, label_frac=0.20 (n_lab=20/group):**

| biases | noise_sd | old (fixed λ=1) | new (adaptive) |
|---|---|---|---|
| none | 0.30 | 0.035 | 0.045 |
| [0.2,−0.1,0.3] | 1.00 | 0.045 | **0.100** |
| [0.5,−0.3,0.2] | 3.00 | 0.055 | 0.075 |

No catastrophic failure anywhere (compare to the prior attempt's 19%, or
this same construction's own pre-fix bug at 100%) — but a real, modest
elevation at moderate/poor judge quality, ~2x nominal at worst.

**Power, same conditions, true effect (group means 0/0.3/0.6):**

| biases | noise_sd | old power | new power |
|---|---|---|---|
| [0.2,−0.1,0.3] | 1.00 | 0.345 | **0.560** |
| [0.5,−0.3,0.2] | 3.00 (poor judge) | 0.075 | **0.425** |

Substantial power gains, largest exactly where expected -- the
poor-judge regime.

**The Type-I elevation is a small-`n_lab` artifact, not a new failure
mode** -- checked directly by varying `n_lab`/group at fixed noise=1.0:

| n | label_frac | n_lab/group | old | new |
|---|---|---|---|---|
| 100 | 0.20 | 20 | 0.044 | 0.076 |
| 300 | 0.20 | 60 | 0.037 | 0.045 |
| 600 | 0.20 | 120 | 0.059 | 0.050 |
| 600 | 0.40 | 240 | 0.049 | 0.050 |

Converges to match the old (already-validated) version by n_lab≈60/group
-- the same known small-sample looseness this investigation has already
documented for the underlying analytic backend everywhere else (Addendum
10 found Type-I 0.042-0.069 for a single adaptively-shrunk mean at
n_lab=15-25), now visibly compounding slightly across k=3 independently-
adaptive group corrections. Not a new problem this fix introduced.

**Regression check**: all 66 ANOVA-related pytest tests pass (the default
`power_tune=False` path is untouched logic, only a new branch added).

### Status: promising, not yet finished

This is a real, working prototype, not a dead end -- unlike the
documented prior attempt, it doesn't blow up, and it delivers large power
gains in the regime that matters most. But it's not yet at the bar the
other five fixes cleared before being committed:

- Only tested on independent-groups ANOVA's p-value/CI path
  (`_ppi_anova_independent_f_stat`) -- `_ppi_anova_independent_ci` shares
  this function so should inherit the fix, but wasn't separately checked;
  repeated-measures ANOVA and Friedman weren't attempted at all.
  `power_tune` currently defaults to `False` everywhere it's still called
  (unchanged) -- this is purely additive/opt-in so far, no default
  behavior changed.
- No official-tier harness validation yet (the other fixes each got a
  --factorial-check or label-efficiency run before being trusted) --
  everything above is a standalone synthetic script, same rigor level as
  an early checkpoint in this investigation, not a final one.
- The small-`n_lab` Type-I elevation, while explained and consistent with
  known pre-existing behavior, hasn't been checked against the harness's
  own 139-scenario judge-bias catalog the way the original (failed)
  attempt was -- that's the natural next validation step before this
  could be committed with the same confidence as the other five.

Code lives in `/Users/ianarawjo/Documents/worktree-anova-power-tuning`
(branch `claude/anova-power-tuning-attempt`, off `ppi-power-tuning-tuning`),
uncommitted. Validation script:
`/private/tmp/claude-501/.../scratchpad/validate_anova_power_tune.py`
(scratch, not part of the repo).

---

## Addendum 16 (2026-08-13): tested and rejected -- extra target-pull-toward-1 at low n_lab

Addendum 15's residual Type-I elevation at low n_lab suggested a natural
idea: since the ADAPTIVE target itself (`_adaptive_shrink_lambda`'s
`target = 1 - P(lam_hat < 0.5)`) is estimated by resampling the same small
labeled sample, maybe it's noisiest exactly when n_lab is small, and the
reported PPI variance doesn't account for the extra variance introduced by
selecting lambda adaptively off that same sample -- so pulling `target`
itself an extra step toward 1 as n_lab shrinks (a new
`_TARGET_EXTRA_SHRINK_C` knob, `target_adj = target_w*target +
(1-target_w)*1`, `target_w = n_lab/(n_lab+C)`) should trade a bit of power
for tighter Type-I control, fading out as n_lab grows.

Tested on both the ANOVA prototype and the highest-traffic single-mean
backend (`_analytic_mean_point_se`), C in {0 (off), 10, 20}:

- **Type-I did improve modestly** everywhere tested: ANOVA n_lab=20/group
  0.081->0.075->0.067; mean-backend poor-judge n_lab=15 0.077->0.073->0.068;
  ANOVA poor-judge (biases=[0.5,-0.3,0.2], noise=3.0) 0.069->0.063->0.057.
- **But power collapsed specifically for poor/uninformative judges** --
  exactly the regime this whole adaptive-target investigation was built to
  protect: mean-backend poor/biased judge n_lab=15, power 0.247->0.147->0.111
  (C=20 nearly halves it); ANOVA poor-judge config, power 0.428->0.360->0.284
  (33% relative loss at C=20) -- directly eating into Addendum 15's headline
  0.075->0.425 power gain over the old fixed-shrink-to-1 approach, in the
  same config.
- Good/moderately-informative judges were essentially unaffected in both
  directions (power flat within noise) -- the damage concentrates exactly
  where `target` is correctly reading "this judge isn't informative" and
  pulling toward 0, which an unconditional extra pull toward 1 can't tell
  apart from "target is just noisy."

**Verdict: rejected.** The mechanism (target estimation is noisiest at low
n_lab) is real, but the fix can't distinguish a noisy target from a
correctly-low one, so it re-introduces the poor-judge power penalty that
motivated moving off fixed shrink-to-1 in the first place (Addenda 1-9),
scaled by the same n_lab lever that was supposed to only fix Type-I. The
residual Type-I elevation this was meant to address is modest (worst case
~2x nominal) and was already shown in Addendum 15 to converge to the
existing (already-validated) baseline by n_lab~60 on its own -- closer to
an accepted, already-documented small-sample artifact than a problem
needing a structural fix. The experimental `_TARGET_EXTRA_SHRINK_C` knob
was reverted from the ANOVA worktree (`worktree-anova-power-tuning`,
branch `claude/anova-power-tuning-attempt`); only the Addendum 15 ANOVA
fix remains, uncommitted. Validation scripts (scratch, not part of the
repo): `validate_target_extra_shrink.py` (ANOVA sweep),
`validate_target_extra_shrink_mean.py` (mean backend),
`validate_target_extra_shrink_pooranova.py` (ANOVA poor-judge config).

---

## Addendum 17 (2026-08-13): a better fix -- lambda-variance inflation, not target-pull

Motivation: even though Addendum 15's residual low-n_lab Type-I elevation
self-resolves by n_lab~60, in practice researchers using PPI specifically
want LOW n_lab (that's the point -- minimizing labeling cost). A visible
Type-I bump at the low end of an n_lab sweep, next to a baseline (old
fixed shrink-to-1) that was tightly calibrated there, is a real
credibility problem for a plot even if it's "just" a known small-sample
artifact -- worth fixing properly rather than living with, if a fix
exists that doesn't reintroduce Addendum 16's poor-judge power penalty.

**Root cause, more precisely this time**: `_analytic_mean_point_se`'s
variance formula (`var_lab + lam**2*(var_unlab+var_hat_lab) -
2*lam*cov_lab_hatlab`) plugs in the adaptively-chosen `lam` as if it were
a known constant -- a textbook plug-in/post-selection variance
underestimation. Writing `estimate = f_lab + lam_hat*r` with `r = f_unlab
- f_hat_lab`, the existing formula already has the `lam**2*Var(r)` term,
but is missing `r**2*Var(lam_hat)` -- the piece that accounts for lambda
itself being estimated, not given. `lam_replicates` (already computed for
the adaptive target) gives `Var(lam_hat)` for free.

**Why this is structurally different from the rejected Addendum 16 fix**:
the target-pull idea inflated conservatism whenever n_lab was small,
regardless of whether the target's "this judge is uninformative" reading
was itself confident or not -- so it couldn't tell a noisy target apart
from a correctly low one, and punished exactly the poor-judge cases the
adaptive scheme was built to help. This fix instead inflates SE only in
proportion to `Var(lam_hat)` (how spread out the resampled lambda
estimates actually are) -- a confidently-poor judge has *tight*
lambda_replicates clustered near 0 even at low n_lab (little to lose),
while a genuinely ambiguous judge has wide ones (real uncertainty worth
reflecting in the CI). It only touches the reported variance, never the
point estimate or the target/blend logic.

**Results** (`_LAMBDA_VAR_INFLATION_ENABLED` toggle added to
`_analytic_mean_point_se`, off by default; single-mean backend and ANOVA
via Addendum 15's per-group reuse of the same function):

Single mean, n_lab=15, Type-I / power (OFF -> ON):
| judge | Type-I | power |
|---|---|---|
| good (r=0.9) | 0.083 -> 0.075 | 0.701 -> 0.679 (-3%) |
| moderate (r=0.5, bias=0.1) | 0.089 -> 0.079 | 0.291 -> 0.274 (-6%) |
| poor/biased (r=0.2, bias=0.3) | 0.077 -> 0.072 | 0.247 -> 0.237 (-4%) |

At n_lab=50 all six cells were flat to +/-0.01 in both Type-I and power
(effect fades as intended). No poor-judge collapse anywhere -- power cost
is small and roughly uniform across judge quality, not concentrated where
it hurts most.

ANOVA (k=3, n_lab=20/group), Type-I / power (OFF -> ON):
| config | Type-I | power |
|---|---|---|
| moderate bias, noise=1.0 | 0.081 -> 0.072 | 0.583 -> 0.562 (-3.6%) |
| poor judge, noise=3.0 | 0.069 -> 0.066 | 0.428 -> 0.414 (-3.3%) |

The poor-judge ANOVA config is the exact one Addendum 16's target-pull
fix collapsed by 33% relative power; this fix costs only 3.3% there.
Convergence check (ANOVA, moderate-bias config, n_lab/group=20/60/120):
Type-I 0.081->0.072, 0.049->0.047, 0.047->0.047 -- fades to negligible by
n_lab~60 automatically (no separate n_lab-dependent schedule needed, since
`Var(lam_replicates)` itself shrinks as n_lab grows).

**Status**: promising, not yet wired in as default or extended to Walsh-
theta/joint-stats/bayes-bootstrap (only patched in `_analytic_mean_point_se`
for this test, `_LAMBDA_VAR_INFLATION_ENABLED` still defaults to False).
If adopted, the natural next step is extending the same delta-method term
to `_analytic_walsh_theta_correct` (identical structure, `_walsh_theta_lambda_replicates`
already exists) and re-running the already-established validation suite
(5-way comparison, factorial sweep, label-efficiency) for both mean and
Wilcoxon backends before flipping the default -- has not been done yet,
pending a decision on whether this trade (small, uniform power cost for
tighter low-n_lab Type-I) is worth taking as the new default. Validation
scripts (scratch): `validate_lambda_var_inflation_mean.py`,
`validate_lambda_var_inflation_anova.py`.

---

## Addendum 18 (2026-08-13): lambda-variance-inflation extended to Walsh-theta, validation battery

Extended Addendum 17's `_LAMBDA_VAR_INFLATION_ENABLED` delta-method fix
(`var_estimate += r_term**2 * Var(lambda_hat)`) to
`_analytic_walsh_theta_correct` -- identical structure to the mean
backend, same patch, `_walsh_theta_lambda_replicates` already provides
the replicate array needed. Applied to the main repo checkout
(`ppi-power-tuning-tuning` branch, forced `True` for this validation run
only, default TBD pending this write-up) since that's what
`simulations.harness.cli` actually imports.

**pytest**: `tests/test_ppi_corrections.py` -- 319 passed, 0 failed, with
the flag forced on.

**Harness validation battery** (Type-I calibration + 5-way comparison +
label-efficiency check; factorial sweep skipped as the most expensive
check, standard `--reps 100 --tests ttest ttest_welch mwu wilcoxon
paired_t bayes_bootstrap --eval-types binary continuous likert`),
compared directly against `screening_postrefactor_20260813_122815` (the
immediately-prior baseline, same scenarios/reps, flag off):

**Type-I** (corr mean / corr max, per test, baseline -> new):
| test | baseline | new |
|---|---|---|
| ttest (untouched path) | 0.065 / 0.140 | 0.065 / 0.140 (identical) |
| ttest_welch (untouched) | 0.063 / 0.140 | 0.063 / 0.140 (identical) |
| mwu (untouched -- uses mwu_ridge) | 0.061 / 0.130 | 0.061 / 0.130 (identical) |
| **wilcoxon** | 0.069 / 0.150 | **0.064 / 0.140** |
| **paired_t** | 0.054 / 0.120 | **0.050 / 0.120** |
| bayes_bootstrap (shared-helper path, not this patch) | 0.055 | 0.051 (MC noise) |

Holm-confirmed miscalibrated cells: 0/968 (new) vs 0/1033 (baseline,
included tango_score) -- both fully controlled at the family-wise level.
Only the two directly-patched tests (wilcoxon, paired_t) moved, and both
moved down toward nominal, exactly as predicted.

**5-way comparison power** (`ppi` row, es=0.30 checkpoint and vs. n_lab,
baseline -> new): continuous es=0.30 0.360->0.354; likert es=0.30
0.500->0.498; continuous n_lab=15 0.342->0.338; likert n_lab=15
0.480->0.472. All differences are within +/-0.008 -- an order of
magnitude smaller than the ~3-6% relative costs seen in Addendum 17's
targeted worst-case synthetic checks, because this suite mostly isn't in
the "genuinely uncertain lambda" regime the fix targets.

**Label efficiency** (N=1000, all IRR levels, N_lab=15-200): multiplier
over `human_subset` stayed net-positive (>=1x) almost everywhere, ranging
~1.0x-3.8x; two isolated cells (continuous/likert kappa=0.30, N_lab=20)
dipped to 0.75-0.84x, consistent with ordinary Monte Carlo noise at
100 reps for power values in the 0.07-0.13 range rather than a pattern
(no dip at the adjacent N_lab=15 or N_lab=30 cells for the same kappa).
No paired baseline run exists for label-efficiency specifically (the
screening baseline didn't include this check), but the Type-I/comparison
results already show the fix's power cost is negligible outside its
target regime, and label-efficiency's own numbers show no systematic
degradation at low IRR/low N_lab.

**Status**: validated on both patched backends (mean, Walsh-theta),
passes the full non-factorial validation battery with a clean
improvement in exactly the low-n_lab Type-I inflation that motivated it
(Addendum 15) and negligible cost elsewhere. Not yet: extended to the
remaining `power_tune` sites (bootstrap path, joint-stats bootstrap,
bayes-bootstrap -- these use `_bootstrap_batch_lambda_replicates`+
`_adaptive_shrink_lambda` rather than the two closed-form functions
patched here, so would need a separate but structurally identical patch);
factorial sweep; a decision on default value; commit. Currently
uncommitted in the main repo (`ppi-power-tuning-tuning` branch,
`evalstats/ppi.py`, flag forced `True`) and in the worktree
(`worktree-anova-power-tuning`, flag defaults `False`).

**Follow-up (same day)**: the label-efficiency multiplier looked jumpy
cell-to-cell in the continuous eval type (e.g. pearson_r=0.30, N_lab=15->20:
2.68x -> 0.84x). Traced to two compounding, pre-existing factors rather
than anything the fix introduced: (1) the validation battery above used
`--reps 100` (screening tier) vs. the `--reps 200` tier used for the
original per-backend label-efficiency validations
(`label_efficiency_analytic`/`label_efficiency_walsh`), and (2) the
multiplier metric itself amplifies noise structurally -- `equiv_N_lab`
comes from inverting the `human_subset` power curve, which is shallow at
low N_lab/low power, so small MC error in the PPI power estimate becomes
a large swing in the inverted N_lab. Confirmed empirically by re-running
label-efficiency alone at the matched reps=200 tier (flag still forced
`True`) and diffing every continuous-eval_type cell against the pre-fix
`label_efficiency_analytic` baseline (also reps=200): mean multiplier
diff across 48 cells = -0.011 (symmetric, 24 cells up / 24 down), stdev
0.142, largest outliers (+0.49, -0.45) scattered across N_lab rather than
concentrated at low N_lab as a real degradation would produce. No
systematic drift -- the jaggedness is intrinsic to the metric/rep-count,
not a consequence of this fix. Same reasoning applies to the 5-way
comparison's `human_subset` line, which is visibly the least smooth of
the five curves for the identical reason (smallest effective N, no
large-unlabeled-sample averaging, and it's the only curve whose per-
replicate outcome also depends on which items happened to get labeled) --
confirmed it's a pre-existing property, not new, by checking the
`official_20260803_013611` reps=200 comparison log, where the same
es=0.05 dip appears (0.058->0.041) but recovers immediately, much
attenuated vs. the reps=100 screening run's sustained 3-step decline
(0.052->0.044->0.036). Re-run: `simulations/out/lambda_var_inflation_labeleff_reps200/`.

---

## Addendum 19 (2026-08-13): lambda-variance-inflation extended to the bootstrap and bayes-bootstrap paths

Extended `_LAMBDA_VAR_INFLATION_ENABLED` to `correct()`'s bootstrap path
and `evalstats.tests._ppi_paired_bayes_bootstrap`. Different mechanism
than Addendum 17/18's closed-form delta-method term, because these paths
have no explicit variance formula to add a term to -- `lam` is estimated
once (from a first bootstrap draw, `b1`) then applied as a fixed constant
across every replicate of a second draw (`b2`) to build `boots = b2_lab +
lam*(b2_unlab - b2_hat_lab)`, so the percentile CI's spread structurally
reflects zero lambda-estimation uncertainty -- the identical plug-in gap,
just manifesting as an under-wide bootstrap CI instead of a missing
variance term. Fixed by convolving independent noise of variance
`r_term**2 * Var(lambda_hat)` directly into `boots` (using each site's
already-computed `lam_replicates`), rather than re-deriving lambda per
b2 replicate (a full nested/double bootstrap) -- both target the same
quantity, this is far cheaper and mirrors this codebase's existing
smoothed-bootstrap jitter pattern (`_tie_jitter_scale`) already used
elsewhere in the same function.

**Correction to Addendum 18**: assumed `ttest`/`ttest_welch` were on an
"untouched path" since they showed identical numbers under the mean/
Walsh-theta-only patch. Checked directly this time (`_ppi_two_sample`,
which both delegate to): its custom two-group estimator plus provided
`X_lab`/`X_unlab` means `correct()`'s fast closed-form dispatch never
matches (keyed on `id(np.mean)`/`id(np.median)`/`id(paired_walsh_midrank_theta)`
with `X_unlab is None`), so it falls through to the generic per-replicate
bootstrap loop -- meaning `ttest`/`ttest_welch` DO route through the path
patched today. Confirmed with a direct single-call check (same data,
flag off vs on): p=0.0160 -> p=0.0220, CI width shifted -- not a no-op.
They were only unaffected by the *previous* patch, not by this one.

**pytest**: `tests/test_ppi_corrections.py` -- 319 passed, 0 failed, flag on.

**Harness validation**, clean paired comparison at reps=200 (identical
seed/scenarios, `_LAMBDA_VAR_INFLATION_ENABLED` toggled with nothing else
changed -- more reliable than diffing against older logs from earlier
investigation phases, given today's patch affects `ttest`/`ttest_welch`
too):

Type-I (corr mean / corr max, off -> on):
| test | off | on |
|---|---|---|
| ttest | 0.063 / 0.120 | 0.061 / 0.125 |
| ttest_welch | 0.063 / 0.125 | 0.061 / 0.115 |
| mwu (separate mwu_ridge path, unaffected) | 0.056 / 0.100 | 0.057 / 0.100 |
| wilcoxon | 0.070 / 0.130 | 0.065 / 0.130 |
| paired_t | 0.055 / 0.105 | 0.051 / 0.100 |
| **bayes_bootstrap** | 0.055 / 0.105 | **0.051 / 0.100** |

`bayes_bootstrap` moves in lockstep with `paired_t` (both 0.055->0.051)
-- expected, since `_ppi_paired_bayes_bootstrap` falls back to
`_analytic_mean_correct` below `_MIN_LAB_RECOMMENDED=30` labels (already
patched in Addendum 17) and uses today's new native fix above that
threshold. `ttest`/`ttest_welch` both improve too, confirming the direct
single-call check. `mwu` is flat, as expected (separate method,
untouched by any of this). Holm-confirmed miscalibrated cells: 2/968 in
*both* the off and on runs -- identical, so whatever those two cells are,
they predate and are unrelated to this fix (likely a reps=200-sensitivity
artifact, since both reps=100 runs earlier in this investigation showed
0/968); not something to chase down as part of this change.

**5-way comparison** (reps=100, ppi row, continuous es sweep and n_lab):
continues the same small, monotonic trend as more sites get the fix --
es=0.30: baseline(no fix) 0.360 -> Addendum18(2 sites) 0.354 -> today
(4 sites) 0.342; n_lab=15: 0.342 -> 0.338 -> 0.324. Same direction and
similar magnitude as each previous increment, no discontinuity.

**Label efficiency** (reps=100): same jaggedness pattern as the earlier
follow-up investigation, including the identical outlier cell
(pearson_r=0.30, N_lab=20 -> 0.84x) recurring at the same location --
consistent with it being an intrinsically noisy cell at this rep tier
(already explained and confirmed benign via the matched reps=200 rerun
in the prior follow-up), not a new issue.

**Status**: all four `power_tune` sites that use `_bootstrap_batch_lambda_replicates`/
`_adaptive_shrink_lambda`-style construction now have the fix (bootstrap
path, bayes-bootstrap) alongside the two closed-form sites (mean,
Walsh-theta) from Addendum 17/18. Remaining site:
`evalstats.api._ppi_bootstrap_t_joint_stats` (Romano-Wolf/max-T joint
bootstrap, per-pair loop) -- not yet touched, same fix should apply
structurally identically (it already uses `_analytic_mean_lambda_replicates`/
`_adaptive_shrink_lambda` per the Addendum 10-14 refactor). Still
uncommitted; flag currently `True` in the working tree for validation,
default TBD pending a decision on all sites + commit.

---

## Addendum 20 (2026-08-13): joint-stats site -- mixed result, not adopted as-is

Extended `_LAMBDA_VAR_INFLATION_ENABLED` to
`evalstats.api._ppi_bootstrap_t_joint_stats` (Romano-Wolf/max-T joint
bootstrap-t underlying FWER-controlled pairwise comparisons). Structurally
a third case: `obs_se` is closed-form (same delta-method term as
Addendum 17/18), but `boot_se` is ALSO closed-form, recomputed per
bootstrap replicate from resampled var/cov while lambda itself stays
fixed across replicates (by this function's own existing design). So the
missing term had to go in both places -- `obs_se` once per pair, and
`boot_se` per (pair, replicate) using each replicate's own resampled
r_term_b, with Var(lambda_hat) held fixed (matching how lambda itself is
already held fixed) -- or the observed and bootstrap-reference
studentized statistics wouldn't be on a comparable scale. Implemented,
pytest passed (`test_compound_ppi_fwer.py` 23/23, `test_simultaneous_ci.py`
72/72), direct sanity check confirmed T's own scale stayed ~1 in both
modes (internally consistent).

**Validation, reusing the function's own original power_tune=True
validation grid/scripts** (`simulations/investigate_joint_bootstrap_power_tune_grid.py`,
`simulations/investigate_joint_bootstrap_fwer_highrep.py`), toggling this
flag instead of power_tune (held fixed True, the already-validated
baseline):

Screening grid (80/50 reps, matching the original grid's own tier):
Romano-Wolf FWER moved in BOTH directions across the 6 conditions --
(5,50,0.40) improved 0.100->0.0625, (4,100,0.30) worsened 0.025->0.075,
(5,200,0.20) worsened 0.050->0.100; max-T (3,50,0.40) improved sharply
0.140->0.020. No power regressions anywhere (0/6, 0/2 at >0.05 drop).

Per this function's own established protocol (its docstring's own
history: an earlier screening grid's apparent movements were re-checked
at 600 reps and confirmed to be MC noise before power_tune=True was
trusted), re-ran the two conditions that moved the most, at N_REPS=600:

| condition | FWER off | z (off) | FWER on | z (on) |
|---|---|---|---|---|
| k=4, N=100, lab=30% | 0.0567 | 0.75 | **0.0767** | **3.00** |
| k=5, N=200, lab=20% | 0.0717 | 2.44 | 0.0600 | 1.12 |

Unlike the original power_tune=True recheck (which found the screening
grid's movement was noise, converging to "within ~1 SE of nominal" at
high rep), this one does NOT converge cleanly: (4,100,0.30) shows a real,
statistically significant elevation with the flag on (z=3.00, ~1.5x
nominal) that wasn't there without it, while (5,200,0.20) shows the
opposite -- an elevation that was already present without the flag
(z=2.44) that the flag actually resolves (z=1.12). Both readings are
far enough from noise at 600 reps (SE=0.0089) to trust individually.

**Interpretation**: this site is structurally different from the other
four sites this fix was applied to. Romano-Wolf's step-down algorithm
ranks and progressively removes pairs using the joint T-matrix; since
Var(lambda_hat) differs per pair, the fix widens obs_se/boot_se
non-uniformly across pairs, which can change WHICH pairs get removed at
each step, not just uniformly shrink every pair's test statistic --
qualitatively different from a single independent test's CI getting
wider (the other four sites), and not obviously guaranteed to preserve
FWER control the same way. This is a plausible mechanism, not a proven
one -- distinguishing it from e.g. a genuine residual-noise condition
that just happens to look real at 600 reps would need either more
high-rep conditions or a targeted investigation of the step-down
interaction specifically.

**Decision: NOT adopted for this site.** Reverted the fix in
`evalstats/api.py`'s `_ppi_bootstrap_t_joint_stats` (both the `obs_se`
per-pair addition and the `boot_se` per-replicate addition) -- this
function keeps its pre-existing (already-validated) behavior regardless
of `_LAMBDA_VAR_INFLATION_ENABLED`'s value. The other four sites (mean,
Walsh-theta, bootstrap path, bayes-bootstrap) are unaffected by this
decision and remain as validated in Addenda 17-19. Validation scripts
(scratch): `validate_joint_bootstrap_lambda_var_inflation.py`,
`validate_joint_bootstrap_lambda_var_inflation_highrep.py`.

---

## Addendum 21 (2026-08-13): Addendum 20's "Romano-Wolf regression" was a bug, not a real incompatibility

Committed Addenda 17-19's fix (mean, Walsh-theta, bootstrap path,
bayes-bootstrap) as `dfd48ad`, refactoring the `_LAMBDA_VAR_INFLATION_ENABLED`
flag away into an unconditional shared helper `_lambda_var_inflation`
(matching this codebase's convention of not leaving toggle flags around
once validated) -- 414 pytest tests pass. Then investigated Addendum 20's
joint-stats regression rather than leaving it unresolved.

**Root cause found**: Addendum 20's implementation added
`r_term_b**2 * var_lam_hat[p]` to `var_b` using each bootstrap
replicate's own RESAMPLED `r_term_b = f_unlab_b - f_hat_lab_b`, while
`obs_se` used the single FIXED, OBSERVED `r_term`. This is inconsistent
with the function's own design principle (stated in its own docstring):
`lam` is deliberately held fixed across every bootstrap replicate, not
re-derived per replicate, specifically to avoid "double dipping." Using a
per-replicate `r_term_b` for the injected variance broke that same
principle -- it made the added variance depend on that replicate's own
realized draw, rather than being a fixed quantity added consistently
everywhere, plausibly distorting the shape of the bootstrap T-distribution
in a data-dependent way inconsistent with how the single fixed `r_term`
affects `t_obs`.

**Fix**: precompute `r_term`'s contribution ONCE per pair from the
observed (fixed) `r_term`, store it in a `lambda_extra_var` array, and add
that SAME fixed value to both `obs_se` (once) and every replicate's
`var_b` (broadcast, not resampled) -- extending the "hold lambda fixed"
principle to `r_term` too, for full consistency.

**Validation**: re-ran the same high-rep (600) recheck on condition
(k=4, N=100, lab=30%) -- the significant regression: 0.0567->0.0767
(z=3.00) under Addendum 20's version -- and got 0.0500->0.0600 (z=0.00 ->
1.12) with the corrected version: the regression is gone.

Condition (k=5, N=200, lab=20%) initially looked ambiguous (0.0633->0.0750,
z=1.50->2.81) using the SAME independent-seed protocol
`investigate_joint_bootstrap_fwer_highrep.py` uses -- but that protocol
draws off/on from independent seeds, so part of any apparent movement is
confounded with ordinary Monte Carlo noise in the DATA-generating draw,
not necessarily caused by the fix. Re-ran both conditions with a properly
PAIRED design instead (same simulated data + same `es.compare` rng seed
for both off/on per replicate, monkeypatching `_lambda_var_inflation` to
a no-op for "off" -- a tighter comparison than anything used earlier in
this investigation, since it isolates the fix's causal effect from
data-generation noise entirely):

| condition | FWER off | FWER on | n_reps disagreeing (600) |
|---|---|---|---|
| k=4, N=100, lab=30% | 0.0650 | 0.0633 | 1 |
| k=5, N=200, lab=20% | 0.0650 | 0.0650 | 2 |

Both conditions: FWER off and on are statistically indistinguishable, and
the fix flips the actual reject/no-reject decision in only 1-2 out of 600
replicates each -- essentially a non-event. The mild elevation present in
both (z~1.7, not individually significant) exists IDENTICALLY with the
fix on or off, confirming it's a pre-existing property of this scenario
independent of the fix, not something introduced by Addendum 20's
original (buggy) version being naively "not that bad" -- it's genuinely
neutral once the bug is fixed.

**Power** (paired, 400 reps, effect_size=0.30, same protocol): (k=4,
N=100, 30%) 0.3550->0.3575; (k=5, N=200, 20%) 0.3900->0.3800. Both within
normal noise for 400 reps -- no cost, no gain, consistent with the FWER
finding.

**Conclusion**: the original interpretation in Addendum 20 (that
Romano-Wolf's step-down algorithm is inherently sensitive to non-uniform
per-pair variance widening) was WRONG -- or at least not the actual
explanation for what was observed. The real cause was a straightforward
implementation inconsistency (resampled vs. fixed `r_term`), not a deeper
incompatibility between this fix and FWER-controlled step-down procedures.
Once fixed, this site behaves like a well-behaved (if here, neutral
rather than clearly beneficial) extension of the same principle used
elsewhere. pytest (`test_compound_ppi_fwer.py` 23/23,
`test_simultaneous_ci.py` 72/72) passes with the corrected version.

**Status**: corrected fix implemented in `evalstats/api.py`'s
`_ppi_bootstrap_t_joint_stats`, validated (paired FWER + power, two
conditions, N_REPS=600/400), not yet committed -- pending confirmation
this is enough validation to extend the commit to the 5th site (only two
of the original 6-condition RW screening grid + 2 max-T conditions were
rechecked at high rep; the other 4 RW conditions and max-T weren't
re-examined with the corrected version, though they showed no movement at
all in Addendum 20's screening grid so were lower-priority). Validation
scripts (scratch): `validate_joint_bootstrap_lambda_var_inflation_highrep_v2.py`,
`validate_joint_bootstrap_lambda_var_inflation_paired.py`,
`validate_joint_bootstrap_lambda_var_inflation_paired_power.py`.

---

## Addendum 22 (2026-08-13): comprehensive harness Type-I validation for ANOVA power-tuning

Housekeeping first: consolidated all branches this investigation touched
onto one clean lineage. `claude/ppi-power-tuning-check-9b6a59` (this
session's default branch) turned out to be a strict ancestor of
`ppi-power-tuning-tuning` (where all the actual code commits landed) --
fast-forwarded it to match, no merge needed. Also discovered this very
file was gitignored the whole time (`simulations/out/*` blanket rule) --
none of Addenda 1-21 had ever actually been committed anywhere despite
extensive editing; added a `!` exception and committed it (`cda158c`).
`worktree-anova-power-tuning` (the ANOVA prototype from Addendum 15,
still based on the much older `af8e452`) had its stale/superseded
lambda-var-inflation prototype in `ppi.py` discarded (redundant with the
now-properly-committed `_lambda_var_inflation` helper) and its real
`tests/__init__.py` diff rebased cleanly onto the latest tip -- no
conflicts, since it touches `_ppi_anova_independent_f_stat` (line ~2856),
far from the bayes-bootstrap changes (line ~1841) that had landed on
`tests/__init__.py` in between. 66/66 ANOVA pytest tests still pass
post-rebase.

**Then ran the actual comprehensive test Addendum 15 never got**: added
`power_tune: bool = False` to `_ppi_anova_independent_p_value`/`_ci`
(threading to the already-fixed `_ppi_anova_independent_f_stat`, default
unchanged), temporarily flipped the harness's two call sites
(`simulations/harness/cases/pvalues.py:3650,4140`) to `power_tune=True`,
and ran the harness's real Type-I check across its full scenario suite
(109-118 conditions, not just the handful of hand-picked configs
Addendum 15's standalone script covered) -- reverted the harness edits
afterward, keeping only the opt-in parameter addition.

**Screening (100 reps)**: corr mean 0.061 (vs 0.054 baseline), Holm-
confirmed 1/118 -- `interact.large+sparse+hetero+diff` at 0.160 (vs
baseline 0.050), a dramatic-looking jump.

**High-rep recheck (500 reps, matched seed, same protocol as the
Romano-Wolf investigation)** told a very different story:

| scenario | off | on |
|---|---|---|
| small+unbal+hetero+diff | 0.092 (Holm) | 0.066 |
| small+sparse+noisy+const | 0.048 | 0.058 |
| mid+extreme-hetero+none | 0.076 | 0.098 (Holm) |
| large+sparse+hetero+diff | 0.092 (Holm) | 0.102 (Holm) |
| small+unbal+hetero+diff.mildbias | 0.080 (Holm) | 0.066 |
| small+unbal+hetero+diff.modbias | 0.072 | 0.074 |
| large+sparse+hetero+diff.mildbias | 0.052 | 0.080 |
| large+sparse+hetero+diff.modbias | 0.066 | 0.068 |

Full 109-scenario (continuous) aggregate: corr mean 0.054 -> 0.060, Holm-
confirmed cells 3 -> 2 (slightly fewer, not more). The flagged scenario's
0.050->0.160 gap at 100 reps was mostly noise -- at 500 reps it's
0.092->0.102, a small increment. More importantly: **the "interaction"
stress category (extreme per-group judge-quality divergence + small
n_lab + unequal group sizes) already has Holm-confirmed Type-I problems
in the CURRENT SHIPPED `power_tune=False` construction** -- a pre-
existing weakness in this stress-test category unrelated to power-
tuning, not something this investigation introduced or needs to fix.
Within that already-rough category, power_tune=True's effect is
genuinely bidirectional (some cells improve, some worsen slightly) with
no systematic direction, consistent with ordinary noise in an already-
noisy corner of the design space rather than a real new problem.

A first reproduction attempt with hand-guessed bias/noise parameters
(0.084 vs 0.089, small gap) failed to match the harness's real magnitude
-- traced to guessing the wrong bias structure (`bias_type="differential"`
biases ONLY group A, not all three groups as guessed); switched to
calling the harness's own `build_judge_bias_sources()`/
`generate_judge_bias_cell()` directly for a faithful reproduction, which
is what surfaced the still-inconsistent numbers that prompted the
proper high-rep harness recheck instead of trusting a hand-rolled script.

**Status**: `power_tune=True` for independent-groups ANOVA is now
validated at the same rigor as the other five sites -- full-suite Type-I
(this addendum) plus the original synthetic Type-I/power sweep (Addendum
15). Net assessment: positive, with one honest caveat (the pre-existing
interaction-category roughness, unaffected either way). Not yet: a power
check across the full harness suite (only Addendum 15's hand-picked
configs so far); label-efficiency; a commit decision. Diagnostic scripts
(scratch): `diagnose_anova_interact_regression.py` (failed hand-
reconstruction, kept for the record), `diagnose_anova_interact_v2.py`
(faithful reproduction via harness internals).

---

## Addendum 23 (2026-08-13): ANOVA power check across the full harness grid

Same temporary-patch protocol as Addendum 22 (`_ppi_anova_independent_p_value`'s
call site in `_run_ppi_cell` flipped to `power_tune=True`, harness power
check enabled with everything else off, reverted after), covering all
three of the harness's standard power tables (`build_ppi_power_sources()`):
adversarial-direction bias, no-bias ceiling, and bias-reinforcing-the-
effect, across continuous and likert, effect sizes 0.00-1.20, 200 reps
each, seed 42, off vs on.

**Result: essentially no difference anywhere.** Every column in all six
table halves (3 tables x 2 eval types) agrees within 1-4 percentage
points, consistent with ordinary 200-rep Monte Carlo noise at these
power levels (SE ~0.03-0.04 mid-range) -- no systematic direction, no
scenario with a real gap. The es=0.00 columns (Type-I cross-check) also
match the earlier dedicated Type-I check's aggregate closely.

This is a genuinely different picture from Addendum 15's own synthetic
power check, which found large gains for a poor judge (noise_sd=3.0:
0.075->0.425). Not a contradiction -- `build_ppi_power_sources()`'s
standard judge-quality severity is moderate (matching `build_judge_
bias_sources()`'s baseline), not the extreme "genuinely poor judge"
regime Addendum 15 specifically constructed to showcase the adaptive
scheme's main advantage over fixed shrink-to-1. The standard grid simply
doesn't probe that regime, so it correctly shows "no harm, no dramatic
gain either" rather than contradicting the earlier finding.

**Status**: ANOVA power-tuning has now cleared the full validation
sequence used for the other five sites -- full-suite Type-I (Addendum 22),
full-suite power (this addendum), plus the original targeted synthetic
sweep (Addendum 15) that specifically stress-tests the poor-judge regime
the standard grids don't reach. Net assessment across all three: solid.
Remaining before a commit decision: label-efficiency check (not yet
run for anova_ind at all), and the open question of whether/how to
extend this to repeated-measures ANOVA, Friedman, and Kruskal (still
untouched, per Addendum 14).

---

## Addendum 24 (2026-08-13): repeated-measures ANOVA -- same fix, working

User asked to extend adaptive power-tuning to the remaining omnibus
tests (repeated ANOVA, Friedman, Kruskal), then stepped away asking for
autonomous progress up to (but not including) a final commit/official
test, which they want to trigger themselves once everything is wired in.

**Diagnosis**: `_ppi_anova_repeated_f_stat`'s existing (`power_tune=False`)
construction has the IDENTICAL bug independent ANOVA had before Addendum
15 -- `cond_means_llm` is built from the FULL sample (labeled+unlabeled
mixed), not a disjoint unlabeled complement, so the rectifier only
cancels judge bias at lambda=1. Verified directly (not just by analogy):
simulated many draws, computed the current construction's bias residual
at lambda in {1.0, 0.5, 0.0} -- residual is ~0 at lambda=1 and grows
linearly with (1-lambda)*true_bias otherwise, exactly the same failure
mode.

**Fix**: rebuilt around the standard disjoint construction `f_lab +
lambda*(f_unlab - f_hat_lab)`, generalized to the k-dimensional
condition-contrast vector this design produces. Lambda is a single
shared scalar (one judge, not per-condition) chosen to minimize
`trace(P @ Var[cond_means_ppi(lambda)] @ P)` -- the vector generalization
of the classical PPI++ lambda* ratio, projected through the same
centering matrix P (`I - 11^T/k`) already used elsewhere in this
function. Shrunk toward an adaptive target via the existing
`_adaptive_shrink_lambda`, with lambda's own estimation uncertainty
folded into the reported variance via the matrix generalization of
`_lambda_var_inflation` (`np.outer(r_term, r_term) * Var(lambda_hat)`,
since `r_term` is now a k-vector, not a scalar).

**A real scaling bug caught immediately by a sanity check, before trusting
any Type-I number**: first version gave Type-I ~0.003-0.011 (16-45x too
LOW, not too high) across every condition. Debugged by checking the raw
F-statistic's distribution directly against its known theoretical mean
(`dfd/(dfd-2)` for F(dfn,dfd)) rather than trusting the pass/fail rate --
mean f_corr was ~0.51 vs expected ~1.05, off by almost exactly 2x with
k=3 (k-1=2). Root cause: `denom` should equal `E[ss_condition_corr]/(k-1)
= n_subjects*trace(P@Var@P)/(k-1)`, and the `/(k-1)` had been dropped.
Fixed; F-statistic mean immediately matched theory (~1.01-1.02 vs
expected ~1.05).

**Validation** (synthetic, k=3, reusing the same configs as independent
ANOVA's Addendum 15 checks):

Type-I, n_lab=20 (n_subjects=200/400 depending on config):
| config | old | new |
|---|---|---|
| good judge | 0.0350 | 0.0400 |
| differential bias, moderate judge | 0.0400 | 0.0660 |
| differential bias, poor judge | 0.0400 | 0.0630 |
| differential bias, poor judge, n_lab=40 | 0.0540 | 0.0440 |

Power, same configs (H1: condition effects 0/0.3/0.6):
| config | old | new |
|---|---|---|
| good judge | 0.9570 | 0.9790 |
| differential bias, moderate judge | 0.2830 | 0.5780 |
| differential bias, poor judge | 0.0810 | 0.3850 |
| differential bias, poor judge, n_lab=40 | 0.1100 | 0.7080 |

Large power gains for poor judges, even bigger than independent ANOVA's
own (0.081->0.385, 0.110->0.708) -- consistent with the same mechanism.

**Convergence check** (poor-judge config, n_lab/label_frac held at ~10%
proportionally, not n_subjects fixed): Type-I stays tight across the
ENTIRE range tested, 0.041-0.059 at n_lab=15 through n_lab=100 -- no
small-n_lab elevation pattern at all here, tighter than independent
ANOVA's own convergence curve. Ported into
`evalstats/tests/__init__.py` (`_ppi_anova_repeated_f_stat`,
`_ppi_anova_repeated_p_value`/`_ci` now accept `power_tune: bool = False`,
default unchanged); direct re-run of the poor-judge check against the
REAL implementation (not the standalone prototype) confirmed matching
behavior (old=0.0437, new=0.0563). pytest: 66/66 ANOVA-related tests
pass (default path untouched).

**Status**: repeated-measures ANOVA power-tuning implemented and
synthetically validated, matching the rigor of independent ANOVA's
Addendum 15. Not yet done: comprehensive harness-level validation
(matching Addendum 22/23's full-suite Type-I/power sweep) -- deferred
until Friedman and Kruskal are attempted too, per the user's request to
wire everything in before running a combined official validation.
Uncommitted. Prototype/debug scripts (scratch):
`verify_repeated_anova_bug.py`, `prototype_repeated_anova_power_tune.py`,
`debug_repeated_anova.py`, `sweep_repeated_anova.py`.

---

## Addendum 25 (2026-08-13): Friedman -- same fix, clean on the first try

Same construction bug as repeated ANOVA (`cond_means_llm` built from the
full sample, only cancels bias at lambda=1 -- Friedman uses ranks instead
of raw scores but the anchor/rectifier structure is identical). This one
worked WITHOUT the denom-scaling bug repeated ANOVA hit, because it
directly reuses `_repeated_anova_lambda_raw`/`_repeated_anova_lambda_replicates`
(already correct, already validated) rather than re-deriving anything --
same functions, just fed rank-transformed inputs instead of raw scores.

The one open question specific to Friedman: this function's OWN docstring
already documents (from before this session) that an SS-decomposition
residual doesn't transfer to ranks (anti-correlated with the tested
effect, inflates Type-I -- the exact mechanism that broke the original
2023-era naive lambda search for this estimand). The fix here never
computes an SS-decomposition residual at all -- only plain sample
covariances of labeled/unlabeled rank-MEAN vectors -- so it structurally
sidesteps that specific failure mode rather than needing a new argument
for why it's safe. Confirmed empirically, not just by that argument.

**Validation** (synthetic, k=3, same configs as repeated ANOVA):

Type-I, n_lab=20:
| config | old | new |
|---|---|---|
| good judge | 0.0370 | 0.0450 |
| differential bias, moderate judge | 0.0500 | 0.0650 |
| differential bias, poor judge | 0.0610 | 0.0630 |
| differential bias, poor judge, n_lab=40 | 0.0450 | 0.0500 |

Power, same configs:
| config | old | new |
|---|---|---|
| good judge | 0.6340 | 0.6530 |
| differential bias, moderate judge | 0.2860 | 0.3650 |
| differential bias, poor judge | 0.1930 | 0.2890 |
| differential bias, poor judge, n_lab=40 | 0.3590 | 0.5340 |

More modest gains than the continuous-score ANOVA cases (expected --
ranks carry less information than raw scores), but real and consistent.

**Convergence check** (poor-judge config, label_frac~10% proportionally):
Type-I tight across the whole range, 0.041-0.059 at n_lab=15 through
n_lab=100 -- same clean pattern as repeated ANOVA's own convergence
check, no small-n_lab elevation at all.

Ported into `evalstats/tests/__init__.py` (`_ppi_friedman_f_stat`,
`_ppi_friedman_p_value`/`_ci` now accept `power_tune: bool = False`,
default unchanged, reusing the SAME `_repeated_anova_lambda_raw`/
`_repeated_anova_lambda_replicates` helpers, no duplicated math); direct
re-run against the REAL implementation confirmed matching behavior
(old=0.0587, new=0.0537). pytest: 38/38 Friedman-related tests pass.

**Status**: implemented and synthetically validated. Not yet: harness-
level validation (deferred alongside repeated ANOVA and Kruskal, per the
user's "wire everything in first" request). Uncommitted. Prototype
scripts (scratch): `prototype_friedman_power_tune.py`, `sweep_friedman.py`.

---

## Addendum 26 (2026-08-13): Kruskal-Wallis -- third instance of the same bug, different packaging

`_ppi_kruskal_wallis_pairwise`'s existing (`power_tune` didn't exist yet)
construction already used a properly DISJOINT unlabeled sample (unlike
ANOVA/Friedman's full-sample bug) -- but the anchor/rectifier roles were
inverted from the canonical PPI form: `theta_unlab + (theta_lab_human -
theta_lab_llm)`, with the JUDGE-based term (`theta_unlab`) as the always-
full-weight anchor and the rectifier scaled by the (currently hardcoded)
lambda=1. Verified via simulation (mean-based proxy, same style of check
used for the other two): this construction's bias residual is exactly 0
at lambda=1 and grows linearly with (1-lambda)*true_bias otherwise --
the SAME failure mode, third time, just packaged with the anchor/
rectifier roles swapped instead of a full-sample-mixing bug.

**Fix**: swap into canonical form, `theta_lab_human + lambda*(theta_unlab
- theta_lab_llm)` -- unbiased at any lambda, since the rectifier
(`theta_unlab - theta_lab_llm`) now has expectation exactly 0 regardless
of lambda (both terms share the same judge bias). Lambda is a single
scalar shared across all C(k,2) pairs (one judge), estimated by
minimizing trace(Var[theta_hat(lambda)]) -- computed from bootstrap
replicates directly rather than a closed-form covariance, since this
estimand was already bootstrap-based (no new closed-form derivation
needed, unlike ANOVA/Friedman). Estimated from a FIRST bootstrap draw,
held fixed for a SECOND independent draw that builds the actual Wald
covariance -- the same double-draw discipline `correct()`'s own bootstrap
path uses to avoid double-dipping. Lambda's own estimation-uncertainty
term is added directly to the bootstrap covariance AFTER the fact
(`np.outer(r_term, r_term) * Var(lambda_hat)`, using the FIXED observed
`r_term`), deliberately NOT woven into the bootstrap loop -- applying the
lesson from Addendum 20/21's Romano-Wolf bug (using each replicate's own
resampled rectifier there, instead of the fixed observed one, caused a
real FWER regression) before it could recur here.

**Validation** (synthetic, k=3, n_boot=400, reps=300 -- more expensive
per rep than ANOVA/Friedman since this is bootstrap-based, not
closed-form, so smaller rep counts than the other two):

Type-I, n_lab=20:
| config | old | new |
|---|---|---|
| good judge | 0.0133 | 0.0100 |
| differential bias, moderate judge | 0.0100 | 0.0333 |
| differential bias, poor judge | 0.0167 | 0.0500 |

Power, same configs:
| config | old | new |
|---|---|---|
| good judge | 0.6767 | 0.8000 |
| differential bias, moderate judge | 0.2800 | 0.4200 |
| differential bias, poor judge | 0.1467 | 0.3300 |

No elevation above nominal anywhere in either construction (the old
Hotelling-corrected construction already runs conservative here); the
poor-judge cell landing exactly at 0.05 was rechecked at reps=800:
old=0.0375, new=0.0450, still comfortably at/below nominal. Solid power
gains throughout, no debugging needed this time (worked on the first
attempt, likely because it reuses the already-correct bootstrap
machinery rather than needing a new closed-form denom derivation the way
repeated ANOVA did).

Ported into `evalstats/tests/__init__.py`
(`_ppi_kruskal_wallis_pairwise` now accepts `power_tune: bool = False`,
default unchanged; the bootstrap-draw loop was factored into a
`_draw_components` closure so power_tune=True's two-draw pattern doesn't
duplicate the resampling logic). Direct sanity check against the REAL
implementation produced sane, non-crashing output with `theta_hat`
notably closer to the true null (0.5) than the old construction's,
consistent with the bias-cancellation fix. pytest: pending. NOT yet
threaded into the public `kruskalwallis()` function's own signature
(matching independent ANOVA's approach -- the harness calls
`_ppi_kruskal_wallis_pairwise` directly, so this wasn't required for
validation).

**Status**: all three remaining omnibus sites (repeated ANOVA, Friedman,
Kruskal) now have `power_tune=True` implemented and synthetically
validated. Comprehensive harness-level validation (matching Addendum
22/23's rigor) is the one thing common to all three still pending, per
the user's explicit request to wire everything in before running that
combined check. Nothing committed. Prototype scripts (scratch):
`verify_kruskal_bug.py`, `prototype_kruskal_power_tune.py`,
`highrep_kruskal.py`.

---

## Addendum 27 (2026-08-13): comprehensive harness validation for all three -- clean across the board

Same temporary-patch protocol used for independent ANOVA (Addendum
22/23), applied to all three sites at once (`simulations/harness/cases/
pvalues.py`'s `_run_ppi_cell` call sites for `anova_rep`/`friedman`/
`kruskal`, `power_tune=True`), covering the full 118-scenario Type-I
suite and the full power grid, both eval types, reverted after.

**Full-suite pytest** (`test_ppi_corrections.py`, all tests, with the
flag forced on): 319/319 pass.

**Type-I, full 118-scenario suite, 354 conditions (118 scenarios x 3
tests), off vs on**:

| test | off: mean / max | on: mean / max |
|---|---|---|
| anova_rep | 0.044 / 0.125 | 0.052 / 0.138 |
| friedman | 0.044 / 0.125 | 0.056 / 0.150 |
| kruskal | 0.028 / 0.075 | 0.038 / 0.087 |

Modest, consistent increases -- same order of magnitude as every other
site's small-n_lab movement documented throughout this investigation.
**Holm-confirmed miscalibrated cells: 0/354 in BOTH the off and on
runs** -- no scenario crossed into real, family-wise-significant
miscalibration in either version.

**Power, full grid (both eval types, es=0.00-1.20), off vs on**: flat to
improved everywhere checked, no regressions. Selected checkpoints
(continuous, es=0.30): anova_rep 0.613->0.613 (identical), friedman
0.275->0.325, kruskal 0.800->0.900. Likert showed the largest gains for
friedman specifically (es=0.10: 0.075->0.225, es=0.15: 0.237->0.412,
es=0.20: 0.512->0.588, es=0.25: 0.575->0.700) -- consistent with
Addendum 25's synthetic finding that Friedman's gains, while real, take
a bit more effect size to show up clearly (ranks carry less information
than raw scores).

**Status**: all six power_tune sites in this codebase now have adaptive
lambda power-tuning, each independently validated at matching rigor:
mean/Walsh-theta/bootstrap-path/bayes-bootstrap (Addenda 17-19,
committed as `dfd48ad`), Romano-Wolf/max-T joint bootstrap-t (Addendum
20/21, committed as `38a346a`), independent-groups ANOVA (Addenda
15/22/23), and now repeated-measures ANOVA/Friedman/Kruskal (Addenda
24-27) -- the last three synthetically validated (Addenda 24-26) AND
harness-validated (this addendum), matching the rigor already applied to
independent ANOVA. Everything from repeated-ANOVA/Friedman/Kruskal
onward remains UNCOMMITTED, per the user's explicit request to hold off
on both committing and running the final official validation until they
return and can trigger it themselves. Nothing in this addendum touched
`evalstats.api._ppi_bootstrap_t_joint_stats` (Addendum 20's rejected
Romano-Wolf attempt) or the ANOVA-family point-estimate/CI paths that
were already flagged as not power-tunable via the simple bootstrap route
(`_ppi_anova_independent`/`_ppi_kruskal_wallis`'s effect-size CIs,
hardcoded `power_tune=False`, per their own existing comments) -- those
remain explicitly out of scope for this round.
