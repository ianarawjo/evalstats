# Rank vs parametric PPI: where power swaps, and why rho^2 still works

**Status:** investigation, no code change. Reproduce with
`python -m simulations.investigate_rho2_sufficiency`,
`python -m simulations.investigate_rank_parametric_crossover`, then
`python simulations/plot_rank_crossover_and_sufficiency.py`.

Fourth note in the sequence. It answers the objection that
`RANK_PPI_TAIL_SENSITIVITY.md` raised but could not settle: if our sweep only
uses Gaussian judge noise, and rank tests only lose under Gaussian judge noise,
then the sweep's "rank tests extract less from PPI" is not a finding about PPI --
it is a finding about our simulation.

That objection is correct, and the fix is a dedicated experiment rather than a
noise-shape axis on the main grid. Two results follow.

## 1. rho^2 IS sufficient -- for both families, each against its own rho

This is the load-bearing justification for the rule of thumb. The claim is not
"rho^2 correlates with savings" but "rho^2 is a *sufficient statistic* for
savings": two judges with the same rho^2 but differently shaped errors must give
the same multiplier, or the rule is under-specified.

25 cells -- 5 noise shapes (Gaussian, Laplace, t_3, contaminated 8% @ 5 sigma,
sign-flip 12%) x 5 Pearson levels (0.20-0.80) -- 1500 reps each, paired design.
Measured multiplier divided by `1/(1 - rho^2*(1 - n_lab/N))`:

| judge noise shape | paired_t vs rho_P^2 | wilcoxon vs rho_S^2 |
|---|---|---|
| Gaussian | 0.946 (sd 0.020) | 0.951 (sd 0.008) |
| Laplace | 0.991 (sd 0.022) | 0.972 (sd 0.011) |
| t_3 | 1.008 (sd 0.019) | 0.962 (sd 0.012) |
| contaminated | 0.995 (sd 0.008) | 0.919 (sd 0.010) |
| sign-flip | 0.996 (sd 0.005) | 0.964 (sd 0.013) |
| **all 25 cells** | **0.987 +/- 0.028** | **0.954 +/- 0.022** |

Both collapse onto one curve. Figure:
`simulations/out/labeleff_rho2_full/rho2_sufficiency.png`.

**The rule of thumb survives a non-Gaussian judge, provided the right rho is
used.** That proviso is not cosmetic -- see `WHICH_RHO_FOR_WHICH_TEST.md`. The
same 25 cells plotted against the *wrong* correlation do not collapse: at
rho_P^2 = 0.50, rho_S^2 ranges 0.474 (Gaussian) to 0.612 (contaminated).

Note that `wilcoxon` sits ~3% below `paired_t` (0.954 vs 0.987). Rank-based PPI
attains slightly less of its own bound. That is a real, reportable finding, and
it is *much* smaller than the raw multiplier gap the main sweep shows -- most of
that gap is the rank penalty rho_S^2 < rho_P^2, which is a judge property, not a
test property.

## 2. The power crossover is real and locatable

Contamination fraction `eps` moves rho_S^2 continuously while rho_P^2 is pinned
analytically (`kappa = sqrt(1/target - 1)` fixes Pearson exactly whatever the
shape, since Pearson sees only second moments). 1500 reps, McNemar on
within-replicate discordances. Figure:
`simulations/out/labeleff_rho2_full/rank_parametric_crossover.png`.

At `rho_P^2 = 0.50`:

| eps | rank bonus | PPI-t | PPI-w | gap | McNemar p |
|---|---|---|---|---|---|
| 0.00 | -0.025 | 0.871 | 0.821 | **-0.049** | <0.0001 |
| 0.02 | +0.045 | 0.878 | 0.857 | -0.021 | 0.007 |
| 0.05 | +0.094 | 0.881 | 0.889 | +0.008 | 0.335 |
| 0.08 | +0.113 | 0.881 | 0.903 | **+0.021** | 0.006 |
| 0.12 | +0.117 | 0.883 | 0.905 | +0.022 | 0.006 |

The classical columns are constant at 0.628 / 0.6133 across every row -- the
signal never changes, only the judge -- so the swap is entirely a PPI effect.
**A judge that goes badly wrong on ~5% of items is enough to flip which test to
prefer.**

## 3. The ARE alone does not predict where it crosses

PPI-wilcoxon should overtake when the rank bonus repays the classical ARE
deficit: `ARE/(1 - rho_S^2*(1-f)) > 1/(1 - rho_P^2*(1-f))`. It does not fit.

| tier | measured crossing | ARE only | ARE x attainment | residual |
|---|---|---|---|---|
| 0.30 | +0.1109 | +0.0366 | +0.0628 | 1.77x |
| 0.50 | +0.0800 | +0.0275 | +0.0473 | 1.69x |
| 0.70 | +0.0336 | +0.0185 | +0.0318 | 1.06x |

Folding in the measured attainment gap (`ARE_eff = 3/pi * 0.954/0.987 = 0.923`)
closes roughly half, and fully explains the 0.70 tier. **The rest is
unexplained.** Ranks need a larger bonus in the power domain than the variance
domain accounts for, and the discrepancy grows as the judge gets worse. Likely
related to `WHY_WILCOXON_USES_SPEARMAN.md`'s open continuous-wilcoxon drift;
not established.

## What to report

Under Gaussian judge noise, rank tests lose **three** ways, and only the first
is intrinsic:

1. the classical ARE (3/pi ~= 0.955) -- real, unavoidable, correctly reported;
2. the rank penalty rho_S^2 < rho_P^2 (~-0.02 under Gaussian) -- **a property of
   the judge's error shape, which reverses to +0.11 under a contaminated
   judge**;
3. an attainment deficit of ~3% (0.954 vs 0.987), plus an unexplained further
   deficit visible only in the power domain.

The main sweep reports their sum as though all three were intrinsic. They are
not, and (2) is the largest of them.

## Method: pair the human side, always

Both scripts here originally shipped an RNG-desync bug, in two different forms:
`_noise`/`_contam` consumed one variate for some shapes and two for others, so a
single shared stream silently gave each condition a *different* `D` and a
different labelled subset. In the crossover script it was visible as
classical_t = 0.6607 at eps=0 against 0.6280 elsewhere, when the classical arm
cannot depend on the judge at all. In the sufficiency script it inflated the
between-shape spread at tier 0.50 to att_t 0.884-1.001 -- a plausible-looking
"contaminated judges do better" finding that is entirely artefact.

Any between-condition comparison in this codebase must draw the human side
(`D`, labelling) once from its own rng stream and replay it across conditions.
This also makes the classical arm byte-identical across rows, which is the
largest variance reduction available. Both scripts now do it and say so.

## Open

- The residual 1.7x under-prediction of the crossing at the 0.30 and 0.50 tiers.
- Gaussian is the *worst* shape for `paired_t` attainment (0.946 vs ~0.99
  elsewhere), consistent in sign with
  `investigate_rho2_noise_shape_invariance.py`'s +3.4% (p=0.13) but larger here.
  Small, but it has now appeared in three separate measurements.
- `mwu` is in neither experiment. Its Spearman mapping is inferred, not derived
  (see `WHY_WILCOXON_USES_SPEARMAN.md`), so it cannot carry a crossover claim.
  It needs its own influence-function derivation before it belongs here.
