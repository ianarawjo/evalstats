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
| `mwu` | Spearman, group scores *(see caveat below)* |

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
| continuous `paired_t` | **1.16** (exceeds the bound -- impossible) | 1.03 |
| likert `paired_t` | 0.82 | 1.02 |
| likert `wilcoxon` | 0.71, drifting 0.82 -> 0.65 across tiers | **0.92, flat** |

(measured / predicted; 1.00 = achieves the control-variate bound)

The paired mean-tests now sit at 1.01-1.03, i.e. they hit the bound. Likert
Wilcoxon's *drift* is gone -- what remains is a level, not a degradation, which
is a materially different claim about rank-based PPI.

## The `rank_penalty` diagnostic

`rho^2_pearson - rho^2_spearman` is now a column in the per-method table. It is
how much of the judge's linear signal a rank-based analysis cannot use, and it
predicts the PPI-t-test vs PPI-Wilcoxon gap **from a calibration set, before
running any sweep**. It is small on our synthetic judges but should widen on
heavy-tailed data, where a regression-trained judge nails the extremes
(inflating Pearson) while scrambling the central ranks that Wilcoxon actually
depends on.

## Open, and honestly unresolved

- **continuous `wilcoxon` still drifts** (0.84 -> 0.71 across tiers, pooled
  0.81). Likert's drift closed; continuous's did not. Difference-Spearman is
  not the whole story there.
- **`mwu`'s mapping is inferred, not derived.** The derivation above is for the
  *signed-rank* statistic. MWU is an independent-groups rank test whose
  influence function involves the cross-sample CDF `F_X(Y)`, which is not
  obviously Spearman on marginal scores. Suggestively, likert `mwu` did not
  drift before this change and does now (0.88 -> 0.71) -- which is evidence the
  mapping may be wrong rather than evidence it is right. Treat it as a
  placeholder.
- **binary at the top tier** (`rho^2 = 0.7`) remains anomalous (`paired_t` 1.17,
  `ttest_welch` 1.32) for reasons unrelated to this change.

## Provenance

The influence-function argument came from an external analysis reviewing two
other derivations; the numbers above are our own measurements on the 300-rep
sweep in `simulations/out/labeleff_rho2_full/`. The claim that rank-based PPI
falls short of the mean-based bound is, as far as we know, novel -- no prior
work applies PPI to rank statistics -- so it should be reported as a finding
with the caveats above, not as a defect to be explained away.
