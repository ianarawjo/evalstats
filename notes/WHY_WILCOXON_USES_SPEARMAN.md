# Why Wilcoxon's PPI prediction does not use Pearson r²

**Status:** implemented in `simulations/harness/cases/pvalues.py`
(`_METHOD_CORR_KIND`, `_method_rho2`). Written up here because the reasoning is
easy to lose and the wrong choice looks plausible and fails quietly.

## The short version

PPI's label-efficiency saving is

```
saving = 1 / (1 - rho^2 * (1 - n_lab/N))
```

but **`rho` is not one fixed quantity across tests**. It is the correlation
between the *influence functions* of the labeled estimator and the judge-based
rectifier, and that changes with the test. Using the judge-vs-human score
correlation everywhere is wrong on two independent axes:

| axis | mean-type tests | rank-type tests |
|---|---|---|
| **estimand** | influence function is linear in the values -> **Pearson** | influence function is a function of ranks -> **Spearman** |
| **structure** | independent groups -> correlate the **scores** | paired -> correlate the **differences** `D = Y_x - Y_y` vs `Dhat = f_x - f_y` |

So the mapping we use is:

| method | correlation |
|---|---|
| `ttest`, `ttest_welch` | Pearson, group scores |
| `paired_t` | Pearson, paired differences |
| `wilcoxon` | **Spearman, paired differences** |
| `mwu` | Spearman, group scores *(derivation below)* |

## Why Spearman for the signed-rank test

The Wilcoxon signed-rank statistic is asymptotically the pseudomedian
U-statistic (van der Vaart Ch. 12; Serfling Ch. 5). Its Hajek projection is

```
g(d) = 1 - F_D(-d) - theta
```

with **no sign-indicator term**. The exact identity
`W+ = sum_{i<j} 1(D_i + D_j > 0) + sum_i 1(D_i > 0)` does carry a sign sum, but
after normalising by `C(n,2)` that term is `O_p(1/n) = o_p(n^{-1/2})`, so it
vanishes at the scale the projection lives on. (A tempting "fix" that folds
`1/2 * 1(d>0)` into the kernel is wrong: summed over pairs it weights the sign
terms by `(n-1)/2`, matching `W+` only at `n = 3`, and the resulting influence
function gives a null variance `13/(12n)` against the textbook `1/(3n)` -- off
by 13/4. If someone proposes that correction, this is the check that settles
it.)

Because `g` is a function of `F_D`, the governing correlation is the grade
correlation -- Spearman. The identification is **exact under H0 when `D` and
`Dhat` are each symmetric about 0** (then `F_D(-D) = 1 - F_D(D)` and the
reflection cancels), and first-order under the local alternatives that power
analysis lives in. Away from that regime it is a Spearman-like grade
correlation of the reflected transforms rather than Spearman exactly.

## Why Spearman for Mann-Whitney too, and why "placements" are not needed

MWU's mapping was originally a placeholder -- the derivation above is for the
*signed-rank* statistic, and MWU is a different object. It has since been
derived and checked, and the mapping stands.

For `theta = P(X < Y)` the Hoeffding decomposition gives **two** influence
functions, one per group:

```
g_A(x) = 1 - G(x) - theta        g_B(y) = F(y) - theta
```

Each observation's IF is the *opposite* group's CDF evaluated at it -- these
are **placements** (Orban-Wolfe), and with ties the mid-distribution form
`P(X<y) + 0.5*P(X=y)`. This is the exact analogue of the signed-rank caveat:
Wilcoxon's IF equals the own-rank transform only under symmetry about 0; MWU's
equals it only under `F = G`. Same law, and in each case the named-correlation
reduction is a null-regime identity. The correct general object is the
**placement correlation**, a weighted mixture of both groups' covariances.

**And in our regime it makes no difference.** Measured over 24 cells
(`investigate_inversion_conditioning correlations`), placement correlation and
Spearman-on-scores agree to ~1%, and both track the directly-measured
governing `rho^2`:

| predictor | slope | intercept | R^2 | max abs dev |
|---|---|---|---|---|
| Spearman on scores | 0.977 | +0.022 | 0.9994 | 8.7% |
| placement correlation | 0.989 | +0.019 | 0.9992 | 8.8% |

The reason is that the placement/Spearman split only opens up **far from the
null**, and a power sweep by construction lives at local alternatives. Measured
across our entire effect ladder, `theta = P_mid(X>Y)` sits at 0.393-0.461 --
between 0.04 and 0.11 from the null -- and `Var(F(Y))` is 0.78-0.93 x 1/12,
not the heavy compression a well-separated design would produce.

So: adopt placements if you want the assumption removed, but do not expect them
to fix anything. An external derivation predicted they would close a 20-30% gap
in likert `mwu`; they move the prediction by about 1%. **The gap was never in
the correlation** -- see `HOW_MULTIPLIERS_ARE_MEASURED.md`.

Note the contrast that matters for the paper: `wilcoxon` is paired, so its
Spearman must be computed on the differences `D` and the score-level tier label
is the *wrong* x-coordinate (0.70 on scores is 0.49 on differences). `mwu` is a
two-group test with no differencing, so score-level Spearman is already right
and the tier label means what it says.

## Why "differences, not scores" mattered more than Pearson-vs-Spearman

This was the larger of the two corrections and the less obvious one. Measured
on likert at the `rho^2 = 0.70` tier: score-level `rho^2` is **0.700** while
`Pearson(D, Dhat)^2` is **0.552**. Differencing two noisy measurements changes
the signal-to-noise ratio, and likert's discretisation compounds it.

The Pearson-minus-Spearman gap, by contrast, is small on our judge model
(`+0.047` continuous, `+0.017` likert, `+0.002` binary) because the judge is a
monotone near-linear transform of truth plus noise -- there is little
nonlinearity for rank correlation to see. **An early test that swapped Pearson
for Spearman at the score level found "no difference" and concluded Spearman
does not help. That conclusion was wrong**: it used the wrong Spearman. The
first-order fix is moving to differences at all; Spearman is the second-order
refinement on top.

## What it fixed

| method | score-level rho^2 | own correlation |
|---|---|---|
| continuous `paired_t` | **1.16** (exceeds the bound -- impossible) | 1.02 |
| likert `paired_t` | 0.82 | 1.01 |
| likert `wilcoxon` | 0.71, drifting 0.82 -> 0.65 across tiers | **0.97, flat** |

(measured / predicted; 1.00 = achieves the control-variate bound)

**CORRECTED 2026-08-18.** The right-hand column previously read 1.03 / 1.02 /
0.94, from per-method reference curves built at the wrong effect size -- see
`HOW_MULTIPLIERS_ARE_MEASURED.md`. Re-derived on fixed curves at 300 reps,
gaussian arm, median measured/predicted:

| eval type | `paired_t` | `wilcoxon` | `mwu` | `ttest` | `ttest_welch` |
|---|---|---|---|---|---|
| continuous | 1.017 | **0.901** | 0.997 | 0.966 | 0.943 |
| likert | 1.024 | **0.941** | 0.855 | 0.954 | 0.974 |
| binary | 1.010 | -- | -- | -- | 0.932 |

Pooled over both judge-noise families: `paired_t` 1.018-1.057, `wilcoxon`
0.934-0.949, `mwu` 0.921-0.927, `ttest` 0.962-0.969, `ttest_welch` 0.951-0.972.

**This retracts the central quantitative claim this note used to make.**
Wilcoxon attains 0.90-0.95 of its own bound against `paired_t`'s 1.01-1.02 --
a gap of roughly 5-10%, not the 8-19% the wrong-effect-size curves reported,
and not the near-parity a first 60-rep pass suggested. `mwu` on likert is the
weakest at 0.855, and `mwu` is the method whose Spearman mapping was inferred
rather than derived.

Two independent confirmations that the small gap is real and the large one was
not:

- `investigate_rho2_sufficiency.py` measures attainment with NO power curve
  and finds `wilcoxon` 0.954 +/- 0.022 against `paired_t` 0.987 +/- 0.028.
- The sweep's own curve-free `variance_multiplier` (see
  `PPIComparisonResult.var_human_subset`) agrees with the inverted multiplier
  to a median ratio of 0.992 wherever both are defined.



## The `rank_penalty` diagnostic

`rho^2_pearson - rho^2_spearman` is now a column in the per-method table. It is
how much of the judge's linear signal a rank-based analysis cannot use, and it
predicts the PPI-t-test vs PPI-Wilcoxon gap **from a calibration set, before
running any sweep**. It is small on our synthetic judges but should widen on
heavy-tailed data, where a regression-trained judge nails the extremes
(inflating Pearson) while scrambling the central ranks that Wilcoxon actually
depends on.

## Since resolved

- **continuous `wilcoxon`'s drift was ours, not the estimator's.** The
  0.84 -> 0.71 slide was a power-curve inversion artifact; with ill-conditioned
  cells gated out it reads 0.98/0.91/0.84/0.98/0.93/0.95 across the tiers, flat.
  See `HOW_MULTIPLIERS_ARE_MEASURED.md`.
- **`mwu`'s mapping is now derived, not inferred** (section above), and the
  likert `mwu` drift that made it look wrong was the same inversion artifact.
  What survives gating is a *level*, ~0.82, confirmed independently at the
  variance scale (0.80-0.85) -- a real discreteness cost in the estimator, not
  a correlation error.

## Open

- **binary at the top tier** (`rho^2 = 0.7`) remains anomalous (`paired_t` 1.17,
  `ttest_welch` 1.38) -- above the control-variate bound, which is impossible.
  Unrelated to this change and unaffected by the inversion gate.

## Provenance

The influence-function argument came from an external analysis reviewing two
other derivations; the numbers above are our own measurements on the 300-rep
sweep in `simulations/out/labeleff_rho2_full/`. The claim that rank-based PPI
falls short of the mean-based bound is, as far as we know, novel -- no prior
work applies PPI to rank statistics. **But see the correction above: on fixed
curves the shortfall is ~3% (curve-free instrument) rather than the 8-19% this
note originally reported, and is not distinguishable from zero in the sweep
itself.** Report the small gap, not a general rank deficit.

## See also: is this shortfall just our Gaussian DGP?

Wilcoxon is ~5% less efficient than the paired t-test under normality (ARE =
3/pi; McKean 2003) and *more* efficient under heavy tails, so the natural
suspicion is that the shortfall recorded above is the textbook ARE in disguise,
and that a different data-generating process would reverse it.

It is not the ARE -- that cancels out of a within-method ratio, and both tests
are measured falling equally short of their own bounds under a Gaussian DGP.
But the suspicion is half right: **Gaussian judge errors are the one case where
ranks lose.** Give the judge Laplace, t_3, or contaminated errors, holding
Pearson fixed, and the rank penalty becomes a rank bonus large enough to flip
which test wins on power. Our sims use `rng.normal` judge noise throughout, so
the multipliers reported here are a *lower bound* for rank tests.

Full write-up, supporting measurements, the practical judge-error-mode taxonomy,
and a differential-bias mechanism that `rho_S^2` is structurally blind to:
**`notes/RANK_PPI_TAIL_SENSITIVITY.md`** (reproduce with
`python -m simulations.investigate_rank_ppi_tail_sensitivity`).
