# How label-efficiency multipliers are measured, and how the measurement fooled us

**Status:** implemented (`LabelEfficiencyPoint.inversion_ratio` /
`.inversion_clamped` / `.well_conditioned`, `_INVERSION_DEV_TOL` in
`simulations/harness/cases/pvalues.py`). Reproduce the validating instrument
with `python -m simulations.investigate_inversion_conditioning correlations`
and `... bound`.

Fourth note in the sequence. `WHY_WILCOXON_USES_SPEARMAN.md` established which
correlation each test's influence function implies;
`RANK_PPI_TAIL_SENSITIVITY.md` showed judge-error shape controls the gap;
`WHICH_RHO_FOR_WHICH_TEST.md` asked whether the rho^2 rule survives that. This
note is about something different and more embarrassing: **for a while, a
defect in how we measured the multiplier was being read as a defect in the
estimator.**

## The setup

The multiplier is measured by inverting a classical power curve: take the PPI
arm's rejection rate, ask what n a human-only classical test would need to
reach it, divide by `n_lab`. That inversion is only as good as its
conditioning. Where the reference curve is flat -- small effect size, small
`n_lab` -- `dn/dP` is large, so the binomial noise in a rejection rate maps to
a huge swing in equivalent n.

This produced a specific, plausible, wrong story. Continuous `wilcoxon`'s
measured/predicted ratio fell **0.84 -> 0.71** across the judge-quality tiers,
which reads as "rank-based PPI degrades as the judge improves" -- an
interesting finding, and a publishable-sounding one. It was an artifact.

## The instrument that settled it

The trick is that PPI's variance reduction can be measured **without any power
curve at all**. In the infinite-unlabeled-pool limit,

```
VRF = 1 - corr(theta_hat_lab, theta_hat_pred_lab)^2
```

so resampling labeled sets and correlating the two sample statistics measures
the governing `rho^2` directly: no lambda tuning, no inversion, no saturation,
no reference curve. Running the real estimator on the same design and
comparing its achieved variance to `1 - rho^2*(1 - n_lab/N)` then separates
*correlation error* from *estimator shortfall* from *measurement artifact*.

That is `simulations/investigate_inversion_conditioning.py`, and it is the
reason the conclusions below are trustworthy: every claim is cross-checked by
two instruments with no shared machinery.

### What it found

**The correlations were never the problem.** Across 24 cells the named
shortcuts the sweep uses land essentially on the identity:

| test | correlation used | slope | intercept | R^2 |
|---|---|---|---|---|
| `wilcoxon` | Spearman on differences `D` | 0.987 | +0.001 | 0.997 |
| `mwu` | Spearman on group scores | 0.977 | +0.022 | 0.9994 |

**The estimators mostly attain their bounds.** Actual variance vs the
finite-pool bound (1.00 = attains it; above 1.00 = leaves efficiency on the
table):

| | rho^2=0.2 | 0.4 | 0.7 |
|---|---|---|---|
| `mwu` continuous | 1.011 | 1.002 | 1.011 |
| `mwu` likert | 1.177 | 1.236 | 1.243 |
| `wilcoxon` continuous | 1.008 | 1.047 | 1.099 |
| `wilcoxon` likert | 0.968 | 0.950 | 0.898 |

Continuous MWU hits its bound exactly. Likert MWU falls short by a flat
~20% -- a **level, not a drift**, and a genuine discreteness cost. Nothing
here resembles the 0.84 -> 0.71 slide the power-scale measurement reported,
which is what proved the slide was ours.

## The fix: gate, don't divide

The human-subset arm is a classical test on exactly `n_lab` labeled items and
**uses no judge scores at all**, so feeding its rejection rate back through the
same curve must return `n_lab`. That makes it an independent probe of each
cell's conditioning -- it measures the curve, not the thing being estimated.
Cells whose probe lands more than `_INVERSION_DEV_TOL` (0.15) from 1.00 are
excluded, exactly as `saturated` already excludes the opposite end of the
curve.

**The first attempt was to divide by it, and that is wrong.** If the inversion
were *biased*, dividing each multiplier by the probe would cancel the bias.
It is not biased -- pooled, its median is 0.97-1.01 per eval_type x method.
The failure mode is **variance**: the same run spans 0.28 to 7.50 across
cells. Dividing therefore removes nothing and injects that spread into every
number. Measured, it pushed continuous `paired_t` from 0.029 to 0.083 mean
deviation and produced 5 cells *above* the control-variate bound, which is
impossible. Recording this because "correct for the bias you can measure" is
the obvious move and it makes things worse.

## The root cause, found later: curves built at the wrong effect size

Everything above is real and the gate is worth keeping, but it treated a
SYMPTOM. On 2026-08-18 the actual cause turned up:
`save_ppi_label_efficiency_plots_per_method` passed `r.effect_size` to
`_classical_pooled_power_curve`. `PPIComparisonResult.effect_size` is the
eval-type-RELATIVE FRACTION -- its own docstring says so, and says it is
metadata rather than what `generate_judge_bias_cell` reads -- not the absolute
magnitude the curve builder needs. The pooled path was always correct, using
`sources[0].effect_size`.

So every PER-METHOD reference curve was built at es = 0.15-0.35 (a fraction)
instead of the eval type's true magnitude, and the error went in DIFFERENT
DIRECTIONS per eval type, which is exactly why it never looked like one clean
offset:

| eval type | true es | curve built at | consequence |
|---|---|---|---|
| continuous | 0.018-0.042 | 0.15-0.35 | far too powerful -> every inversion clamped to the grid floor (97% clamped, 0% well-conditioned) |
| likert | 0.172-0.401 | 0.15-0.35 | too weak -> overshoot, median inversion 2.881 |
| binary | 0.131-0.306 | 0.15-0.35 | too weak -> overshoot, median inversion 1.621 |

against a target of 1.000. After the fix all three read median exactly 1.000
and per-method retention goes 3.1% -> 36.2% (pooled 62.7%) at 60 reps, and to
74.1% (pooled 87.5%) at 300.

At 300 reps the remaining losses split 13.8% ill-conditioned (shrinks as
1/sqrt(reps)), 11.4% clamped and 2.0% saturated. The latter two are structural
-- np.interp cannot express an answer outside the grid, and PPI's power cannot
be inverted once it passes the curve's ceiling -- so ~13% is a floor no rep
count reaches past.

**How it was caught, which is the reusable part:** the gate's retention did not
improve between a 10-rep and a 60-rep run (2.9% -> 3.1%). Monte Carlo noise has
to shrink with reps; a systematic error does not. That single comparison ruled
out the entire "needs more reps" hypothesis and pointed at the curve.

**What this means for the numbers below.** The retention figures and the
clamp-trap analysis stand -- they are properties of the inversion, not of the
effect size. But every per-method MULTIPLIER measured before the fix is
invalid, including the ones this note used to characterise the artifact. The
gate was doing real work, just not for the reason recorded here: it was
rejecting cells whose curves were simply wrong.

## The clamp trap

`_equivalent_n_lab` inverts with `np.interp`, which **clamps** at `n_grid`'s
endpoints rather than extrapolating. The human arm at the smallest `n_lab` has
power near alpha, at or below the curve's left edge, so its inversion pins to
`n_grid.min() == _JB_MIN_LAB ==` that same `n_lab` and returns a ratio of
exactly 1.000 however ill-conditioned the cell actually is.

Measured: **53% of `n_lab=15` cells returned exactly 1.000, and none returned
below it**, against a median of 0.91 at `n_lab=20`. A gate that passes the
worst-conditioned corner of the design because the arithmetic cannot express
failure there is worse than no gate. Clamped inversions are now explicitly
untrusted.

The tell that this was right: retention became monotone in both axes, which is
what conditioning predicts and what the pre-clamp-fix version violated.

| axis | retention |
|---|---|
| `n_lab` 15 -> 200 | 24% -> 85% |
| effect_frac 0.15 -> 0.35 | 43% -> 75% |

61% of cells survive overall.

## What it fixed

Continuous `wilcoxon`, measured/predicted by tier:

| | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| before | 0.84 | 0.80 | 0.82 | 0.77 | 0.90 | **0.71** |
| after | 0.98 | 0.91 | 0.84 | 0.98 | 0.93 | **0.95** |

The drift is gone. The evidence that this is a real fix rather than a filter
tuned to flatter results is that the gated power-scale numbers now
independently reproduce the variance-scale instrument:

| cell | power scale (gated) | variance scale | agree? |
|---|---|---|---|
| likert `wilcoxon` | 0.94 flat | ~0.94 | yes |
| likert `mwu` | 0.82 plateau | 0.80-0.85 | yes |

Note especially that **`mwu` is not pulled to 1.00**. The gate removes
measurement artifact and leaves the genuine discreteness shortfall standing.
A filter that made everything agree with theory would be evidence of
overfitting; one that removes a drift and preserves a level is doing its job.

## Caveats

- The bound `1 - rho^2*(1 - n_lab/N)` was derived and validated for the MEAN
  case and is applied here to U-statistics with the labeled set nested inside
  N. Continuous MWU hitting it at 1.011/1.002/1.011 is decent evidence it
  transfers, but likert `wilcoxon`'s sub-1.00 readings (0.968/0.950/0.898) are
  exactly where a mean-case approximation would show up. At 400 reps the SE on
  a variance ratio is ~7%, so those are 0.5-1.5 SE below 1.00 and cannot be
  cleanly separated from noise. Checked and ruled out: shrinkage (bias <=3%,
  and the human-only comparator carries the same mild bias).
- The before/after numbers come from **replaying the saved 300-rep raw results
  through the gated code path**, not from a fresh sweep. That CSV predates the
  effect-size rounding fix and stores the *frac* in its `effect_size` column,
  so the true effect sizes had to be recovered from the source builder. A
  native run is the real confirmation and has not been done.
- Binary's top tier is **unaffected and still wrong**: `paired_t` 1.17 and
  `ttest_welch` 1.38, both above the bound. Unrelated to inversion
  conditioning; still open.
- `_INVERSION_DEV_TOL = 0.15` is a judgement call, not a derived threshold.
  Tightening it monotonically improves the worst offender and costs cell count;
  see the constant's docstring for the sweep that chose it.

## The measurement that needs none of this

The whole apparatus above -- the gate, the clamp handling, the conditioning
analysis -- exists to rescue a multiplier obtained by inverting a power curve.
The control-variate factor is a VARIANCE RATIO by definition, so it can be
measured as one instead: run both arms, take Var(classical)/Var(PPI) across
replicates, and no curve is involved at any point.

That is `PPIComparisonResult.var_human_subset` / `var_ppi`, surfaced as
`variance_multiplier`. It reports in 100% of cells against the inverted
multiplier's ~49% at 60 reps, and where both are defined they agree to a
median ratio of 0.992.

It also settled the binary anomaly this note's gate could only flag: measured
over 3000 replicates the true variance ratio sits at 0.92-0.95 of the
control-variate bound at every binary tier, while the inverted multiplier
reported 1.24-1.37x of it at the top tier. PPI never beat its bound; the
inversion did.

The inverted multiplier is still worth reporting -- it is in the unit a
practitioner acts on, "this many labels" -- but it is the derived quantity,
and where the two disagree the variance ratio is the one to believe.

## The transferable lesson

A measurement pipeline with a conditioning problem does not fail loudly. It
produces smooth, monotone, interpretable trends that invite mechanistic
explanation -- here, a story about rank estimators degrading with judge
quality that survived several rounds of scrutiny because it was internally
consistent and directionally plausible. What killed it was building a second
instrument that shared no machinery with the first. Before explaining a
gradient, check that the ruler is straight.

## Binary's top-tier overshoot: diagnosed

Binary's strongest judge tier reported a multiplier 1.23x its own
control-variate bound -- impossible, and the single most flaggable number in
the figures. Resolved 2026-08-18.

**It is not the estimator.** Measured directly over 3000 replicates, the true
variance ratio Var(classical)/Var(PPI) sits at 0.92-0.95 of the bound at every
binary tier, top one included (0.948 at n_lab=60, 0.944 at n_lab=200). PPI
never beats its bound. See the variance-route section above.

**It is not high power per se.** Attainment does rise with ppi_power, but only
weakly (Spearman 0.345), and binary's other five tiers sit at 0.950-0.982
despite reaching comparable powers:

    tier   0.2    0.3    0.4    0.5    0.6    0.7
    att   0.982  0.950  0.979  0.959  0.971  1.227

**It is the reference curve running out of range.** Binary's classical power
curve saturates far earlier than the others', so most of the n_grid carries no
information:

    eval / method            n at P=0.95   n at P=0.99   % of grid still rising
    binary     paired_t              57            80            36%
    binary     ttest_welch          222           317            67%
    likert     paired_t             117           167            53%
    continuous paired_t             499           681            81%

Binary paired_t exhausts its curve by n~80 of a grid running to 1500. Past
that the curve is flat at ~1.0 -- 19 of its 35 adjacent grid points hold
literally tied raw power, against 2 for continuous -- and
_smooth_monotone_power_curve's tie-breaking (a 1e-9 ramp) is what an inversion
landing there actually reads. The strongest tier is precisely where PPI's
power is high enough to land in that flat region, which is why only that tier
shows it.

**What to do about it.** Nothing, in the inverted multiplier: this is the
inversion's known failure mode arriving where the design guarantees it will,
and the conditioning gate cannot catch it because the affected cells are
neither clamped (the answer is inside the grid) nor saturated (ppi_power sits
below the curve's max). Report `variance_multiplier` for these cells instead
-- it needs no curve and reads 0.94 of the bound, which is the sound number.

A grid extending past 1500 would NOT help; the problem is the curve reaching
1.0, not the grid ending.

**A fourth gate criterion was tried and does not work.** The obvious move,
given the diagnosis above, is to flag cells whose equiv_n_lab lands past the n
where the curve reaches P=0.99 -- i.e. in the region carrying no information.
Measured: it flags 4.6% of binary cells and moves the top tier from 1.227 to
**1.247**, slightly worse. The excess is spread across the tier rather than
concentrated in the cells that land furthest out, so the criterion removes
cells roughly at random with respect to the thing it is meant to catch. Do not
re-attempt without evidence that the affected cells are separable at all.

**The variance route does not visibly separate them at 60 reps either.** It
reads 1.116 on that tier against the inverted route's 1.290 -- better, but
still above the bound, and 27.7% of all its cells exceed 1.05 against the
inverted route's 31.3%. That is sampling noise, not a second anomaly: a
variance taken over 60 replicates carries ~18% standard error, so the ratio
carries ~26%. The clean 0.92-0.95 figures quoted above come from 3000
replicates at fixed design points, not from a 60-rep sweep. Expect the sweep's
variance_multiplier to become decisive around 300 reps (~11% SE) and not
before.

### A real Type I bug, which turns out NOT to be the cause

An external analysis proposed that binary's overshoot is a calibration leak
rather than an estimator or inversion problem: at a 2% flip rate the rectifier
residuals are nonzero only on discordant items, so with probability
~(1-p_disc)^n_lab a replicate sees NO discordance, the variance estimate
degenerates, and the test rejects almost always -- inflating power and Type I
alike, worst at small n_lab.

**The Type I half is correct.** Measured under a true null (800 reps,
alpha=0.05):

    tier   n_lab   Type I   P(no discordance)
    0.70      15    0.160        0.155
    0.70      30    0.079        0.039
    0.70      60    0.056        0.001
    0.70     200    0.049        0.000
    0.50     any    0.046-0.072  ~0
    0.20     any    0.052-0.068  ~0

Tier-specific, n_lab-dependent, and the excess (0.110) tracks P(no
discordance) (0.155). This is a genuine calibration bug in its own right and
should be fixed -- a variance floor, t critical values keyed to observed
discordances, or a bootstrap that cannot collapse. Note the realised
discordance rate is well above the nominal 2%, because
_ppi_power_baseline_binary also applies a bias term that pulls the two flip
probabilities apart.

**But it does not explain the multiplier overshoot.** The analysis proposed
its own validation -- bin the excess by n_lab and check it decays like
(1-p_disc)^n_lab. It does the opposite:

    n_lab      15     20     30     40     60     90    130    200
    tier 0.70   0.973  1.137  1.306  1.195  1.174  1.320  1.408  1.344
    Type I      0.160    --   0.079    --   0.056    --     --   0.049

Spearman(excess, n_lab) = +0.881. The overshoot GROWS with n_lab while the
Type I inflation shrinks; at n_lab=15, where the variance estimate degenerates
most often, the multiplier is fine (0.973), and at n_lab=200, where Type I is
exactly nominal, it is worst (1.344).

So the two phenomena are separate. Whatever drives the multiplier excess
scales with n_lab -- consistent with the classical-side story (higher n_lab ->
higher PPI power -> inversion lands further up the flattening curve) rather
than with variance degeneracy. That remains unresolved; note only that the
leading candidate has now been tested and eliminated, by the test its own
proposer suggested.
