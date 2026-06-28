# evalstats simulation harness

Centralized CLI for evalstats' calibration/comparison Monte Carlo simulations,
consolidating the scripts in `simulations/*.py` into one runner so results
can be reproduced for the paper's Supplementary Material with a single
command.

```
python -m simulations.harness.cli --list-cases
python -m simulations.harness.cli ci_single --reps 50 --sizes 10 20
python -m simulations.harness.cli ci_single --data-source dove
python -m simulations.harness.cli ci_single --nested-mode --runs-sweep 5 --reps 200
python -m simulations.harness.cli ci_paired --nested-mode --runs-sweep 5 --reps 200 --include-null
python -m simulations.harness.cli pvalues --mode pairwise --reps 200 --sizes 10 20 50
python -m simulations.harness.cli pvalues --mode multiarm --k-arms 4
python -m simulations.harness.cli pvalues --mode ppi --tests ttest wilcoxon anova_rep
python -m simulations.harness.cli --official-tests all
python -m simulations.harness.cli --official-tests ci_single
```

`--official-tests {all|case1,case2,...}` runs each case's canonical
"official test" preset and writes every artifact into one timestamped
`simulations/out/official_<timestamp>/` directory with a `manifest.json`
index (args used, output paths, key metrics, pass/fail, duration per case).

The legacy scripts in `simulations/*.py` are left untouched; the harness is
purely additive.

## Shared scenario library

- `scenarios/synthetic.py` -- canonical binary/continuous/likert/grades
  single-sample generators (ported from `sim_compare_boot.py`'s
  `build_scenarios()`). Cases should pull distributions from here rather
  than redefining their own, so the paper can describe these distributions
  once.
- `scenarios/real_data.py` -- DOVE_Lite / OpenEval corpus loaders (ported
  from `sim_dove.py`), exposed through the same `CISource` interface as the
  synthetic generators (`generate(rng, n) -> ndarray`, `true_mean`) so a
  single simulation loop can run over either. Also has shared-item DOVE /
  OpenEval / Inspect-AI *pairing* (ported from `sim_tango_real.py`),
  exposed as `CIPairSource` (`generate_pair(rng, n, runs) -> (a, b)`,
  `true_diff`) for `cases/ci_paired.py`.
- `scenarios/synthetic.py` also has `build_multirun_sources()` and
  `build_pair_multirun_sources()` (ported from
  `sim_compare_boot_nested.py`), parameterized by `run_noise_frac` rather
  than ICC, for `cases/ci_single.py`'s and `cases/ci_paired.py`'s
  `--nested-mode`. `CISource` gained an optional `generate_runs(rng, n,
  runs) -> (n, runs)` field for this (`None` for ordinary flat sources);
  `CIPairSource.generate_pair` was already `runs`-aware so no interface
  change was needed there.
- `scenarios/synthetic.py` also has `build_multiarm_sources()` (ported from
  `sim_compare_pvalues.py`'s `build_multiarm_scenarios()`), exposed as
  `MultiArmSource` (`generate_scores(rng, n, runs, k, delta) -> (k, n,
  runs)`), and `build_judge_bias_sources()` / `generate_judge_bias_cell()`
  (ported from `sim_type_i_calibration.py`'s `_build_scenarios()` /
  `_run_one()`'s data-generation half), exposed as `JudgeBiasSource` and
  `JudgeBiasCellData` -- all for `cases/pvalues.py`. `cases/pvalues.py`'s
  pairwise (non-PPI) mode reuses the existing `build_pair_sources()` rather
  than porting `sim_compare_pvalues.py`'s own near-duplicate pairwise
  generator -- see Known exceptions below.

## Shared method registry

`methods.py` defines a `Method(name, color)` dataclass; every CI method is
one named instance (`BOOTSTRAP`, `WILSON`, `BETA`, ...), ported from
`sim_compare_boot.py`'s `_METHOD_COLORS`. Cases work with the instances
directly -- `method.name` where a plain string is structurally required
(CSV fields, dict keys passed to `evalstats.core.resampling`, pandas
columns), `method.color` for plotting -- instead of threading a bare
string through a separate per-case color lookup table. Groupings like
`BOOTSTRAP_METHODS` and `CONTINUOUS_EXTRA_METHODS` are plain lists of
`Method` instances; `order_present_methods(present_names)` filters the
canonical ordering down to whatever a run actually computed, returning
`Method` objects in canonical order. `get_method_color(name)` remains as a
convenience for the few spots (e.g. pandas `groupby` results) where only a
plain string is available. Add new `Method` instances/groupings there as
new cases are ported -- don't predeclare methods for cases that don't
exist as code yet. `pvalues` added `PAIRWISE_PVALUE_METHODS`,
`MULTIARM_CORRECTION_METHODS`, and `PPI_TEST_METHODS` when
`cases/pvalues.py` was actually ported. Note some names overlap by concept
but not by computation across groupings and intentionally get distinct
`Method` instances (e.g. `newcombe` for `pairwise_differences(method=
"newcombe")` vs. `ci_paired`'s `newcombe_score` for the older
`_newcombe_paired_score_ci` helper); `WILCOXON` ("wilcoxon") is the one
exception that's genuinely shared (same paired-difference test) and is
reused as-is across the pairwise-pvalue and PPI-test-name groupings.

## Cases

| Case | Status | Ported from | Data sources |
|---|---|---|---|
| `ci_single` | implemented | `sim_compare_boot.py` (single-sample) + `sim_dove.py` + `sim_compare_boot_nested.py` (mean phase, via `--nested-mode`) | synthetic, dove, openeval, all (`--nested-mode`: synthetic only) |
| `ci_paired` | implemented | `sim_compare_boot.py` (pairwise) + `sim_tango_real.py` + `sim_compare_boot_nested.py` (pairwise phase, via `--nested-mode`) | synthetic, dove, openeval, inspect, all (`--nested-mode`: synthetic only) |
| `pvalues` | implemented | `sim_compare_pvalues.py` (`--mode pairwise`/`multiarm`) + `sim_type_i_calibration.py` (`--mode ppi`) | synthetic only |
| `alignment` | planned | `sim_alignment_methods.py` | synthetic only |

There is no separate `ci_nested` case: `sim_compare_boot_nested.py` was
folded into `ci_single`'s and `ci_paired`'s `--nested-mode` flag instead of
becoming a third file, so single- and paired-sample CI logic for a given
estimand stays in one place. `--nested-mode` sweeps multi-run scenarios
parameterized by `run_noise_frac` (`f_run = var_run / (var_input +
var_run)`, optionally folding in an `--icc-values` sweep via `f_run = 1 -
ICC`) and reports "flat" (cell-mean reduction) and "*_nested"/"*_flat"/
"*_multirun_*" CI methods side by side, so the cost of ignoring run
structure is visible. See `methods.py`'s `NESTED_METHODS` /
`BINARY_FLAT_METHODS` / `BINARY_NESTED_METHODS` /
`PAIR_DIFF_NESTED_METHODS` / `BINARY_PAIR_FLAT_METHODS` /
`BINARY_PAIR_NESTED_METHODS` groupings for the full method list.
`nested_official_args()` in each case file documents the canonical
large-scale nested preset (mirrors `sim_compare_boot_nested.py
--official-test`); like the real-data presets, it isn't wired into
`--official-tests` and must be invoked manually -- see each case's
docstring for the exact flags.

`bayes_evals.py` is a vendored dependency (Beta-Bernoulli / Dirichlet-
multinomial Bayesian CI methods), not a case -- it's imported by cases the
same way `evalstats.core.resampling` is.

## Known exceptions (documented, not silently dropped)

- `alignment` is synthetic-only for now -- a natural fit for OpenEval (which
  has paired human + LLM labels) once ported.
- `ci_paired`'s real-data path (dove/openeval/inspect) only supports `runs=1`
  (flat, single run per item) and is restricted to known-binary benchmarks,
  matching `sim_tango_real.py`'s `SINGLE_RUN_METHODS` scope. Multi-run real
  pairs (Tango multirun variants, nested-diff bootstrap, `lmm_diff`) are
  deferred to a future real-data extension.
- `--nested-mode` (both cases) is synthetic-only and, for `ci_paired`,
  `statistic=mean` only -- matching `sim_compare_boot_nested.py`, which
  never supported `--statistic median` for its multi-run paths.
- `official_args()` for both `ci_single` and `ci_paired` is synthetic-only --
  real-data sources need network/HF Hub access not assumed available in
  `--official-tests` runs. Run `--data-source dove/openeval/inspect` manually
  when that access is available.
- `scenarios/synthetic.py`'s `"extreme"` suite is currently identical to
  `"expanded"` -- reserved for future stress-test scenarios as more cases
  are ported.
- There is no separate `ppi_calibration` case: `sim_type_i_calibration.py`
  was folded into `cases/pvalues.py`'s `--mode ppi` instead of becoming a
  third file, since both halves of `pvalues` answer "is this statistical
  decision trustworthy" at different levels of the stack (raw CI/p-value
  procedures vs. high-level `evalstats.tests` wrappers with PPI), sharing
  report/plot/CLI scaffolding but not scenario generators.
- `cases/pvalues.py`'s `--mode pairwise` reuses `build_pair_sources()` (the
  same ICC x Cohen's-d grid `ci_paired` uses) rather than porting
  `sim_compare_pvalues.py`'s own near-duplicate pairwise generator. This
  means its synthetic scenarios do **not** numerically match the legacy
  script's pairwise phase (different distributions/parameterization) --
  verification for this mode is by sanity check (Type-I error ~ alpha,
  power increasing with effect size and n) rather than exact parity. The
  per-method p-value computation (`_pairwise_pvalue`) and `--mode multiarm`
  (scenarios, metrics, and cell-level orchestration) were ported faithfully
  and do match exactly -- see Verification below.
- `cases/pvalues.py`'s `--mode ppi` "dist" axis (normal/likert/skewed) is
  intentionally narrower than `EVAL_TYPES` (no binary -- current PPI tests
  like ttest/MWU aren't designed for it) and is a different axis entirely
  from `EVAL_TYPES` (judge-measurement-error parameters: bias, scale-
  calibration slope, label sparsity/MNAR-ness, repeated-measures error
  correlation -- not score-distribution shape).
- `judge_bias.py` (planned, for `alignment`) will keep
  `sim_alignment_methods.py`'s agreement-rate proxy-noise model as its own
  named generator, separate from `scenarios/synthetic.py`'s
  `build_judge_bias_sources()`/`generate_judge_bias_cell()` (ported from
  `sim_type_i_calibration.py` for `cases/pvalues.py`'s PPI mode) -- they
  model different things (agreement-rate noise vs. bias/slope/MNAR
  miscalibration).

## Verification

- `ci_single`'s synthetic path was checked against the legacy
  `sim_compare_boot.py --estimand mean` for exact numeric parity (same
  seed): all per-(eval_type, scenario, n, method) coverage and width values
  matched to 8 decimal places across binary and continuous eval types. The
  one expected exception is the `beta` CI method, which is itself an
  unseeded parametric bootstrap in `evalstats.core.resampling.beta_ci_1d`
  (no `rng` threaded through by either the legacy script or the harness),
  so its width has inherent run-to-run Monte Carlo noise independent of
  this port.
- `ci_paired`'s synthetic path was checked against the legacy
  `sim_compare_boot.py --estimand pairwise` for exact numeric parity (same
  seed) across binary and continuous eval types, including the
  `--include-null` Type I error path: 0 mismatches across 144+240 rows
  checked (coverage and mean width, all methods including
  `newcombe_score`/`tango_score`/`bayes_indep_comp`/`bayes_paired_comp`).
- `ci_single --nested-mode` was checked against the legacy
  `sim_compare_boot_nested.py --estimand mean` for exact numeric parity
  (same seed) across binary and continuous eval types: 0 mismatches across
  880 rows checked (every nested/flat method except `beta`, which has the
  same inherent unseeded-MC-noise exception noted above).
- `ci_paired --nested-mode` was checked against the legacy
  `sim_compare_boot_nested.py --estimand pairwise` for exact numeric parity
  (same seed) across binary and continuous eval types, including
  `--include-null`: 0 mismatches across 600 rows checked (every flat,
  nested, and Tango-multirun method).
- `pvalues --mode multiarm` was checked against the legacy
  `sim_compare_pvalues.py` multi-arm phase for exact numeric parity (same
  seed): `build_multiarm_sources()` vs. `build_multiarm_scenarios()`'s raw
  generator output matched across all 8 shapes (4 base + 4 extreme) and
  multiple seeds; `_compute_multiarm_metrics` matched across 15 trials x 5
  correction strategies; the full per-(scenario, n) cell worker
  (`_run_multiarm_cell` vs. `_multiarm_cell_worker`) matched with 0
  mismatches across 8 scenarios x 10 (correction, condition) rows = 80 rows.
- `pvalues --mode pairwise`'s per-method p-value computation
  (`_pairwise_pvalue`) matched the legacy function exactly across 8 trials x
  12 methods (binary and non-binary), including the `mcnemar`/`fisher_exact`
  /`newcombe`/`bayes_binary` binary-only special cases. The scenario data
  itself is a known, documented exception (reuses `build_pair_sources()`
  rather than porting the legacy script's own pairwise generator -- see
  Known exceptions), so this mode was sanity-checked instead of exact-
  matched: at reps=300, sizes=[30, 60], Type-I error sits at or near the
  nominal alpha=0.05 for all methods except `fisher_exact` (a known
  conservativeness-inversion artifact of exact tests on small/sparse binary
  tables also present in the legacy script) and `bayes_binary` (a known
  liberal bias from its prior), and power increases monotonically with both
  Cohen's d and n for every method.
- `pvalues --mode ppi` was checked against the legacy
  `sim_type_i_calibration.py`'s `_run_one` for exact reproduction (same
  seed): since `generate_judge_bias_cell` already reproduces `_run_one`'s
  entire data-generation sequence byte-for-byte (verified separately across
  6 scenarios including binary dist, MNAR, and `repeated_corr`), and the
  per-replicate test-computation code calls the identical sequence of
  `evalstats.tests` PPI functions in the identical order, single-replicate
  runs starting from the same seed are exactly reproducible end to end: 0
  mismatches across all 43 scenarios x 3 reps x 11 tests = 1419 corrected/
  uncorrected reject-decision pairs checked, including every bootstrap-based
  PPI correction (`_ppi_two_sample`, `_ppi_paired_arrays`,
  `_ppi_kruskal_wallis_pairwise`) and closed-form LMM Wald test
  (`lmm`/`lmm_factorial`/`lmm_runs`).
