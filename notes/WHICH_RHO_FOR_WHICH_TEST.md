# Why the rho^2 rule of thumb holds up, and why rho must be chosen per test type

**Status:** investigation, no code change. The per-test-family mapping this note
argues for is already implemented (`_METHOD_CORR_KIND` in
`simulations/harness/cases/pvalues.py`); what is missing is that the *paper's*
rule-of-thumb section does not yet say it. Reproduce with
`python simulations/investigate_rho2_noise_shape_invariance.py`.

Third note in a sequence (a fourth, `HOW_MULTIPLIERS_ARE_MEASURED.md`, covers
how the multiplier itself is measured and the inversion artifact that briefly
masqueraded as an estimator defect). `WHY_WILCOXON_USES_SPEARMAN.md` established
which correlation each method's influence function implies.
`RANK_PPI_TAIL_SENSITIVITY.md` showed that judge-error *shape* controls the gap
between them. This note asks the question those two raise: **if the rule of
thumb is stated in rho^2, and rho^2 moves with judge noise shape, is the rule of
thumb itself under-specified?**

## The question that decides it

Not "is real LLM-judge noise Gaussian" (it is not -- see
`RANK_PPI_TAIL_SENSITIVITY.md`'s exp4). The question is whether the
**rho^2 -> multiplier map is invariant to judge noise shape**. If it is, the
rule of thumb is safe and the main sweep does not need a noise-shape axis. If it
is not, rho^2 is not a sufficient statistic and the sweep is under-specified.

Pearson can be pinned analytically: for `Dhat = D + kappa*E` with `E`
unit-variance and independent of `D`,

```
rho_P^2 = 1 / (1 + kappa^2)      exactly, whatever E's shape
```

because Pearson depends only on second moments. So `kappa = sqrt(1/rho^2 - 1)`
holds Pearson fixed at a tier while the shape of `E` varies freely.

## Answer: the map is invariant for mean tests

5000 reps, n=600 pool / n_lab=60, `rho_P^2` pinned at 0.50 by construction.
Paired design: the human side (`D`, labelled subset) is drawn once and replayed
for every shape, so the classical arm is byte-identical across rows
(`var = 0.017269`) and every difference below is attributable to the judge.
CIs are bootstrap over replicates.

| judge noise shape | mult_t | 95% CI | vs Gaussian | p |
|---|---|---|---|---|
| Gaussian | 1.7056 | [1.6428, 1.7722] | -- | -- |
| Laplace | 1.7635 | [1.6985, 1.8301] | +3.40% | 0.163 |
| contaminated (8% @ 5 sigma) | 1.7639 | [1.6990, 1.8298] | +3.42% | 0.133 |
| sign-flip (12%) | 1.7307 | [1.6668, 1.7969] | +1.47% | 0.535 |

Nothing separates the shapes. This is what theory demands: the mean's influence
function is linear in `D`, so the influence-function correlation *is* Pearson,
and the multiplier must be pinned by it alone.

**Honest residual:** all three non-Gaussian shapes land *above* the Gaussian
baseline (+3.40%, +3.42%, +1.47%). Three of three sharing a sign hints at a real
sub-5% effect that this many reps cannot resolve. It is far below anything that
matters for a rule of thumb, and it is not contamination-specific -- Laplace and
contaminated are indistinguishable from each other -- so it does not motivate a
sweep axis. Recorded rather than rounded away.

**Method note, recorded because it inverts the usual advice:** the kurtosis
column is ~0 for every shape, i.e. the PPI *estimator* is near-Gaussian even
when the judge noise is not (it is an average). So the plain Var-ratio is the
efficient estimator here and the robust MAD^2 ratio is the *noisier* one -- the
opposite of the situation in `RANK_PPI_TAIL_SENSITIVITY.md`'s exp1, where t_3
and lognormal genuinely lack the moments Var needs. Check the estimator's own
tail weight before reaching for a robust statistic; do not assume heavy-tailed
inputs mean a heavy-tailed estimator.

**A retracted over-read, and how far it had to be chased.** A first pass at 400
reps gave spreads of 20-26% and briefly looked like evidence that rho^2 was
insufficient even for mean tests. The apparent contaminated-judge excess decayed
at every increase in rigour:

| pass | design | contam excess |
|---|---|---|
| 400 reps | unpaired, Var-ratio | +9% |
| 800 reps | unpaired, Var-ratio | +9% |
| 5000 reps | unpaired, analytic SE | +5.24% (z=1.28) |
| 5000 reps | **paired + bootstrap CI** | **+3.42% (p=0.133)** |

Recorded because the wrong version was a plausible finding that would have
justified an expensive and unnecessary sweep axis.

**A second method note:** the analytic SE used in the third row assumed the
classical and PPI arms were independent (`2/n + 2/n`). They are strongly
positively correlated -- `PPI = classical + lambda*rectifier` -- so that SE is
*conservative* and could have hidden a real effect rather than manufacturing
one. Pairing the human side across shapes and bootstrapping over replicates is
the right design; it removes the between-shape nuisance variation entirely.

## But rho_S^2 moves a lot at matched rho_P^2

The same construction, reading Spearman instead:

| tier (rho_P^2 pinned) | rho_S^2 range across the four shapes |
|---|---|
| 0.30 | 0.276 - 0.433 |
| 0.50 | 0.471 - 0.611 |
| 0.70 | 0.675 - 0.756 |

At matched Pearson 0.50, Spearman ranges 0.47-0.61. Both are legitimate
measurements of the same judge; they simply answer different questions, and only
one of them is the right input for a given test.

## The consequence

**A practitioner who computes Pearson, reads the rule of thumb, and then runs
Wilcoxon gets a materially wrong number.** Under an erratic judge they are
under-promised (Spearman is higher than Pearson, so the real saving exceeds the
prediction); under other error shapes they are over-promised. The error is not
small: at tier 0.50 the implied multiplier spans 1.73x to 2.22x depending on
which correlation is used.

The fix is not more simulation. It is that the reported metric must be stated
**per test family**:

| test family | correlation to compute | on what |
|---|---|---|
| `ttest`, `ttest_welch` | Pearson | group scores |
| `paired_t` | Pearson | paired differences `D` |
| `mwu` | Spearman | group scores (inferred -- see the Spearman note's caveat) |
| `wilcoxon` | Spearman | paired differences `D` |

## Recommendation on sweeping more noise shapes

**Do not add a noise-shape axis to the main label-efficiency sweep.** In cost
order:

1. *Free* -- state the metric per test family in the rule-of-thumb text. This is
   the actual finding and it is a text change.
2. *Cheap* -- one supplementary robustness figure from
   `simulations/investigate_rank_ppi_tail_sensitivity.py`: mean-test invariance
   across noise shapes, plus the rank caveat.
3. *Skip* -- the noise-shape axis. It would multiply an already hours-long sweep
   by 4-5, invalidate every cached reference curve (a different DGP means
   different classical power, so `_classical_pooled_power_curve`'s entire cache
   is stale), and the thing it would test is the thing that came back invariant.

## Open

- The sub-5% positive offset for non-Gaussian judge noise (all three shapes,
  same sign) is unresolved and would need substantially more reps, or an
  analytic argument, to call. Stated tolerance for the invariance claim:
  **rho_P^2 pins the mean-test multiplier to within ~3.5%** across the judge
  noise shapes tested.
- `mwu`'s row in the table above is inferred, not derived. See
  `WHY_WILCOXON_USES_SPEARMAN.md`.
- Only `rho_P^2 = 0.50` was tested at high rep count. The 400-rep scan covered
  tiers 0.30 and 0.70 as well and showed nothing shape-dependent beyond noise,
  but neither was re-run under the paired design.
