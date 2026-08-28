# Is rank-based PPI's shortfall an artifact of our Gaussian DGP?

**Status:** investigation, no code change. Reproduce with
`python -m simulations.investigate_rank_ppi_tail_sensitivity`.
Companion to `WHY_WILCOXON_USES_SPEARMAN.md`, which established *which*
correlation each method uses; this note asks whether the residual shortfall
that survived that fix is a property of our simulation rather than of PPI.

## The question

Wilcoxon is known to be ~5% less efficient than the paired t-test under
normality (ARE = 3/pi ~= 0.955; McKean 2003) and *more* efficient under heavy
tails (ARE = 1.5 under Laplace, unbounded under Cauchy; Hodges-Lehmann bound
0.864 from below). Our sweeps report rank-based PPI falling short of its
control-variate bound, and our DGP is Gaussian throughout. So: are we just
re-measuring the textbook ARE, and would a different DGP reverse the finding?

## The commentary, as delivered

> Your intuition about the DGP is right, but it operates through a different
> channel than the ARE -- and the ARE itself can't be the explanation here.
>
> **The 3/pi loss cancels out of the number I reported.** The 0.92 and 0.81 are
> *within-method* ratios: the numerator is PPI-Wilcoxon, the denominator is the
> classical **Wilcoxon** reference curve. Both arms pay the rank tax, so it
> divides out. The measurement confirms it -- under a Gaussian DGP, paired-*t*
> attains 0.912 of its own bound and Wilcoxon attains 0.916 of its own.
> Identical shortfall. So McKean's ~5% is real but it isn't what my plot is
> showing; that shortfall is PPI failing to reach its control-variate bound,
> and it's not specific to ranks.
>
> **But the DGP does control the result -- through the correlation, not the
> efficiency.** What sets Wilcoxon's PPI bound is rho_S^2 (Spearman on the
> differences) versus paired-*t*'s rho_P^2. Whether ranks win depends on which
> correlation survives the noise better, and that's a property of the *judge's
> error distribution*:
>
> | judge errors | rho_P^2 | rho_S^2 | rank bonus | predicted bound |
> |---|---|---|---|---|
> | Gaussian | 0.499 | 0.473 | -0.026 | t 1.81x / **w 1.74x** |
> | Laplace | 0.499 | 0.517 | +0.018 | t 1.82x / **w 1.87x** |
> | t_3 | 0.509 | 0.583 | +0.074 | t 1.85x / **w 2.10x** |
> | contaminated (8% @ 5 sigma) | 0.501 | 0.612 | +0.111 | t 1.82x / **w 2.23x** |
>
> Pearson is pinned at ~0.50 by construction across every row -- only the shape
> of the judge's error changes. Gaussian errors are the one case where ranks
> lose.
>
> **The reversal survives to actual power.** Holding the signal distribution
> fixed and changing only the judge, the classical arms don't move (t
> 0.637->0.632, w 0.615->0.607 -- noise), but the PPI arms swap: PPI-*t* beats
> PPI-Wilcoxon under a Gaussian judge (McNemar p<0.0001), and PPI-Wilcoxon beats
> PPI-*t* under a contaminated one (p=0.0041). So this is a PPI phenomenon, not
> the classical ARE resurfacing.
>
> **And our simulation sits squarely in the losing corner.** Judge noise is
> `rng.normal(0.0, noise_sd, ...)` everywhere (`synthetic.py:1312`), with
> Gaussian or Gaussian-mixture truth. The toy's Gaussian-judge penalty of 0.026
> brackets our measured `rank_penalty` (+0.017 Likert, +0.047 continuous), so
> it's reproducing the same effect.
>
> Two things I'd hold back on. First, one direction of your hypothesis doesn't
> hold: putting the heavy tail in the *signal* (shared by human and judge) makes
> the rank penalty **worse**, not better -- shared outliers inflate Pearson's
> covariance. Only judge-side tails produce the bonus. Second, this doesn't
> resolve the continuous-Wilcoxon drift -- the toy shows both tests falling
> equally short (~0.88), which doesn't reproduce the sweep's asymmetry where
> paired-*t* hits 1.01-1.03 while Wilcoxon sits at 0.92. Different mechanism,
> still open.
>
> Practically, this reads as a limitations paragraph: our multipliers for rank
> tests are a *lower bound* on what a practitioner with an occasionally-erratic
> LLM judge would see.

Figure: `simulations/out/labeleff_rho2_full/rank_ppi_tail_sensitivity.png`.

## Supporting numbers

**exp1, classical ARE reproduces** (validates the setup -- if this had come out
flat, nothing else in the note would be trustworthy):

| signal DGP | classical t | classical w | w/t |
|---|---|---|---|
| normal | 0.760 | 0.745 | 0.98 |
| Laplace | 0.770 | 0.878 | 1.14 |
| t_3 | 0.793 | 0.933 | 1.18 |
| contaminated | 0.773 | 0.933 | 1.21 |
| lognormal | 0.895 | 0.495 | 0.55 |

The lognormal row is **not** an efficiency result: signed-rank tests symmetry
about zero, and lognormal differences are skewed, so the test is estimating a
different quantity. Do not cite it as evidence about ranks.

**exp3, power with McNemar** (1500 reps, n=600 pool, n_lab=60, delta=0.30):

| judge | classical t | classical w | PPI t | PPI w | w-only | t-only | McNemar p |
|---|---|---|---|---|---|---|---|
| Gaussian | 0.637 | 0.615 | 0.861 | **0.829** | 29 | 78 | <0.0001 |
| contaminated | 0.632 | 0.607 | 0.883 | **0.905** | 75 | 43 | 0.0041 |

The classical columns barely move between rows -- the signal is identical, only
the judge changed. That is what makes this a PPI effect rather than an ARE one.

## Practical judge-error modes (exp4)

1-5 Likert, applied at the item level then differenced. The columns to compare
are `judge_mean_shift` against `rho^2`:

| mode | judge mean shift | rho_P^2 | rho_S^2 | bonus | mult_t | mult_w |
|---|---|---|---|---|---|---|
| clean | +0.00 | 0.523 | 0.498 | -0.025 | 1.817 | 1.565 |
| offset (uniform leniency) | **+1.20** | 0.523 | 0.498 | -0.025 | 1.817 | 1.565 |
| differential (one arm) | **+0.45** | 0.523 | 0.498 | -0.025 | 1.817 | **1.331** |
| round (integer Likert) | +0.00 | 0.462 | 0.439 | -0.023 | 1.640 | 1.419 |
| clip (won't score below 3) | +0.32 | 0.373 | 0.364 | -0.009 | 1.527 | 1.428 |
| refuse (15% -> midpoint) | -0.04 | 0.439 | 0.402 | -0.036 | 1.524 | 1.367 |
| contaminated (8% misread) | +0.00 | 0.334 | **0.418** | **+0.084** | 1.285 | 1.277 |
| heteroskedastic (hard items) | +0.00 | 0.262 | 0.259 | -0.002 | 1.166 | 1.216 |

Read the `offset` row against `clean`: a judge running a full 1.2 points lenient
is **byte-identical** on every downstream quantity. Location shifts are free --
PPI's rectifier removes them and both correlations are location-invariant. This
is the same fact that ruled out ICC/CCC as the rule-of-thumb axis (see
`rho2-label-efficiency-axis` memory): agreement metrics penalise exactly what
PPI already fixes.

What costs you is anything that scrambles item-level *ordering* or adds
item-level *variance*: ties from integer rounding, information destroyed by
clipping, mass dumped at a default value by refusals, item-dependent noise.
Mean-shifting and correlation-destroying are near-independent axes, and only the
second one matters.

`contam` is the only mode that produces a rank bonus, and it is the mode that
best describes a judge that is usually right but occasionally reads an item
catastrophically wrong -- arguably common in practice.

## A mechanism the predicted bound cannot see (exp5)

Falling out of the `differential` row above: a constant offset on **one arm**
leaves Pearson and Spearman *exactly* unchanged (both location-invariant) and
leaves paired_t's attainment *exactly* unchanged -- but degrades Wilcoxon's,
because the Walsh estimand `P(D_i + D_j > 0)` is **not** location-invariant.

| bias (raw) | bias (SD of D) | rho_P^2 | rho_S^2 | r_t | r_w |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.543 | 0.518 | 0.875 | 0.906 |
| 0.25 | 0.30 | 0.543 | 0.518 | 0.875 | 0.911 |
| 0.50 | 0.60 | 0.543 | 0.518 | 0.875 | 0.896 |
| 1.00 | 1.19 | 0.543 | 0.518 | 0.875 | **0.758** |
| 1.50 | 1.79 | 0.543 | 0.518 | 0.875 | **0.599** |
| 2.00 | 2.39 | 0.543 | 0.518 | 0.875 | **0.546** |

`r_t` is constant to three decimals across the whole grid; `rho_S^2` never moves
at all, so the predicted bound is flat while the realized multiplier halves.

**This does not explain our drift.** Our label-efficiency sources inherit
`bias_type="differential"` at `_jb_bias_magnitude(et)` = 0.30 population SD with
`icc=0.20`, i.e. ~0.24 SD of the difference -- which lands in the flat part of
the grid, between the 0.00 and 0.25 rows where the effect is nil. Stated
explicitly because the coincidence is tempting and wrong.

It is still worth reporting as a **practical caution**: real judges can easily
carry differential bias above 1 SD of the difference, where Wilcoxon's realized
saving falls to ~0.55 of what rho_S^2 predicts. A user reading the rule of thumb
off a Spearman correlation would be over-promised.

## What this changes

- The paper's rank-test multipliers should be framed as a **lower bound**, valid
  for a well-behaved Gaussian-error judge, with the note that an erratic judge
  moves rank tests *up*, not down.
- The rule of thumb stated in rho^2 is unaffected -- it was chosen for
  bias-invariance and that survives everything here.
- `WHY_WILCOXON_USES_SPEARMAN.md`'s open item ("continuous's drift did not
  close") remains open. Differential bias is now excluded as the cause at our
  settings.

## Caveats

- Variance ratios under t_3 and lognormal are unreliable (infinite 4th moment
  means Var of the variance estimate does not exist). `exp1`/`exp2` report a
  robust MAD^2 ratio alongside; trust only where the two agree. They disagree
  most at t_3 (`m_w` 1.895 vs `mr_w` 1.593).
- The toy DGP is difference-level and does not go through
  `generate_judge_bias_cell`, so it validates mechanisms, not our sweep's exact
  numbers. Its uniform ~0.88 attainment for *both* tests does not reproduce the
  sweep's asymmetry (paired_t 1.01-1.03 vs wilcoxon 0.92) -- most likely
  lambda-estimation noise at n_lab=60, which the power-curve route handles
  differently. Not chased down.
- exp3's reversal is significant but modest (~0.02 in power). It is corroborated
  by the much more precisely estimated rho_S^2 - rho_P^2 gap, not standing alone.
