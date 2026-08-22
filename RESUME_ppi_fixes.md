# PPI fixes + compare_e2e scenario rework — 2026-08-21/22

Verification is COMPLETE; this file is kept as the validation record.

Originally written as a handoff when a session closed mid-verification. All
outstanding items below have since been run and are recorded inline.

## What changed

**1. `evalstats/api.py` — relative floor on the joint bootstrap's SE**
New module constant `_JOINT_BOOT_SE_REL_FLOOR = 0.20` and one line in
`_ppi_bootstrap_t_joint_stats`:

    boot_se = np.maximum(boot_se, _JOINT_BOOT_SE_REL_FLOOR * obs_se[None, :])

Set the constant to `0.0` to get the old behaviour back exactly.

Fixes: a near-degenerate pair's bootstrap SE could collapse (the old guard
was absolute, `1e-12`, so it could not see a small-but-nonzero collapse),
sending that pair's `|T|` to 60–2000. Both consumers of this ONE joint
resample reduce it with a MAX over pairs — Romano-Wolf's step-down
suffix-max, and `_M_b_from_T` for max-T / "boot" CI widening — so one bad
pair poisons the whole family. Measured: a degenerate pair with |T|max=66
drove an UNRELATED pair's Romano-Wolf p from ~0 to 0.363 while that pair's
CI still excluded 0 by a wide margin.

**2. `evalstats/api.py` + `core/bundles.py` + `core/router.py` — likert PPI routing**
PPI's `method="auto"` re-derived `data_kind` locally with only
binary/bounded_01/unbounded branches, ignoring `score_range` and
`eval_type`. Likert (e.g. 1–5) fell through to `unbounded` and silently took
`ppi_t_interval`, making `PPI_AUTO_METHOD_TABLE`'s `likert -> ppi_logit_t`
row unreachable. Now the router's single decision is recorded as
`AnalysisBundle.resolved_data_kind` and reused; the old local test remains
as the fallback for non-`auto` callers. (`data_kind` also had to be
initialized to `None` in router.py — it was only bound inside the
`method == "auto"` branch.)

## Verification status

DONE, before touching api.py:
  - binding rates: 12–19% on degenerate cells, **0.0000%** on 8
    non-degenerate conditions at every c up to 0.50
  - FWER on the non-degenerate DGP: identical for c in {0,.1,.2,.3,.5}
  - power on degenerate cells: 0.665 -> 1.000, FWER unchanged
  - symptom: contradictory reps 7/20 -> 0/20; romano_wolf p 0.371 -> 0.001
  - `tests/test_auto_ci_routing.py`, `test_bayes_binary_routing.py`: 78 pass
    (routing fix only — run BEFORE the boot_se floor was added)

DONE after resuming — shipped-vs-prefix validation (validate3 / 400 reps,
`simulations/out/joint_bootstrap_se_floor/val3.log`). FWER **identical to 4
decimals in all 10 conditions**; power on the degenerate cells:
    continuous k=3 N=100   0.6425 -> 0.9975
    continuous k=3 N=200   0.6825 -> 1.0000
    continuous k=5 N=100   0.3850 -> 1.0000
    likert (all N)         unchanged, as predicted
    all 4 non-degenerate   unchanged to 4 decimals
So the floor recovers power exactly where the bootstrap had broken down and
changes literally nothing anywhere else.

DONE: regression tests — **167 passed** (test_compound_ppi_fwer.py incl.
TestRomanoWolfCalibration, test_ppi_ci_methods.py, test_simultaneous_ci.py,
test_auto_ci_routing.py), 23 min.

DONE: null-binding FWER test — the hole in the earlier evidence. Ordinary
nulls have uniq(d_true)==1 so the floor is inert and "FWER unchanged" was
trivial. Built a null where it DOES bind (near-identical arms: shared base,
each arm perturbs a small random subset by ±delta with mean-zero signs).
**With binding up to 12% under a genuine null, FWER is unchanged** (largest
move +0.0025 at 0.28% binding = 0.23 MC SE). See
`simulations/investigate_joint_bootstrap_se_floor_nullbind.py` and
`simulations/out/joint_bootstrap_se_floor/val6.log`.

REMAINING:
  3. End-to-end:
     `.venv/bin/python -m simulations.harness.cli compare_e2e --reps 60 --eval-types likert continuous --k-values 3 --sizes 50 100 200 --ppi-fracs none 0.20 0.40 --plots off --save-results off --progress off`
     Pass = continuous PPI power clears its subset-only `ref.pwr` floor
     (was 20 points BELOW it), and likert `fam.cov` falls from 99.5%.
  4. Not reviewed: the max-T CI path and other consumers of this same joint
     resample may want the same treatment.

## Caveat on likert

Its low-N conservatism is MOSTLY LEGITIMATE, not this bug. `alpha_eff` comes
from `M_b`'s p95 (3.07 -> 0.00214, matching the observed 0.002116), and a
max-|T| p95 of 3.07 at n_lab=20 is ordinary bootstrap-t small-sample
inflation; the degenerate tail sits in the p99/max. Likert binds only ~1% at
c=0.2 vs continuous's 18%. Expect a small move, not a large one.

UNVERIFIED residual suspect, do not act on without checking: the bootstrap-t
critical value is converted to an effective alpha through a NORMAL quantile
in `_ppi_alpha_eff_from_M_b`, then fed to a t-based CI formula — possibly
double-counting small-sample inflation.

## Also disproven this session

The hypothesis that likert's conservatism came from `logit_t` hitting its
Clopper-Pearson fallback (`degenerate_sample_ci`): instrumented every
module-level reference, **0 calls** in every likert and continuous cell at
N=20/50, PPI and non-PPI. Instrument verified against a forced constant
sample, which does trigger it.

## Files

Untracked helpers (safe to delete):
  simulations/investigate_joint_bootstrap_se_floor_*.py   (validation harness)
  simulations/out/joint_bootstrap_se_floor/               (pre-fix logs + api.py backup)
  RESUME_ppi_fixes.md                                     (this file)


---

## Final status (2026-08-22)

All verification complete:

- **167 regression tests** pass (test_compound_ppi_fwer incl.
  TestRomanoWolfCalibration, test_ppi_ci_methods, test_simultaneous_ci,
  test_auto_ci_routing), plus 115 routing/dispatch and 58 unpaired.
- **Shipped-vs-prefix, 400 reps**: FWER identical to 4dp in all 10 conditions;
  power on degenerate cells 0.6425 -> 0.9975 / 0.3850 -> 1.0000.
- **Null-binding FWER test**: with the floor binding up to 12% of replicates
  under a genuine null, FWER is unchanged (largest move +0.0025 = 0.23 MC SE).
- **compare_e2e end-to-end**: Type-I nominal on both paths; PPI power above
  the human-subset floor on all three eval types.

## Two findings that were NOT library bugs

1. **"PPI below the human-subset floor"** -- a k-mismatch in the PLOTTING
   code, not in evalstats. oracle/subset rates exist only at
   REFERENCE_ESTIMATOR_K=3 while PPI's power_rate exists at every k, so
   pooling PPI over k=2+3 against a k=3-only reference compared a 1-step
   effect against a 2-step one. At continuous N=250: PPI(k=2+3)=0.840,
   PPI(k=3)=0.962, subset(k=3)=0.905. PPI was winning throughout.
2. **Romano-Wolf "missing" on the non-PPI path** -- not missing. Both paths
   share AUTO_PVALUE_CORRECTION_METHOD_TABLE (Shaffer <30, Romano-Wolf >=30);
   they differ only because the subset arm legitimately sees fewer items.

## Romano-Wolf vs Shaffer under PPI (the open question)

Same cells, only the correction varied. Romano-Wolf matches or edges Shaffer
everywhere and is slightly LESS conservative (Type-I 0.017 vs 0.008 at
k=3/N=250), consistent with the existing simulation findings. Differences are
inside MC noise at 80 reps (SE ~0.028), so the defensible claim is
"indistinguishable, no evidence of a power cost" -- not "Romano-Wolf wins".

## Still open

- `nlab=30, N=250` Type-I: a small fixed label budget with a large unlabelled
  pool is where the rectifier is estimated from fewest labels. Every look so
  far has been at rep counts where any drift is inside noise. Worth a
  dedicated high-rep check.
- `_pooled_k_group_lambda` still carries the uncentered-pooling defect its
  two-group sibling had (fixed in 836f811), deliberately unfixed pending
  Type-I/coverage validation.
