# evalstats simulation harness

Centralized CLI for evalstats' calibration/comparison Monte Carlo simulations,
consolidating the scripts in `simulations/*.py` into one runner so results
can be reproduced for the paper's Supplementary Material with a single
command.

```
python -m simulations.harness.cli --list-cases
python -m simulations.harness.cli ci_single --reps 50 --sizes 10 20
python -m simulations.harness.cli ci_single --data-source real
python -m simulations.harness.cli ci_single --nested-mode --runs-sweep 5 --reps 200
python -m simulations.harness.cli ci_paired --nested-mode --runs-sweep 5 --reps 200 --include-null
python -m simulations.harness.cli pvalues --mode pairwise --reps 200 --sizes 10 20 50
python -m simulations.harness.cli pvalues --mode multiarm --k-arms 4
python -m simulations.harness.cli pvalues --mode simultaneous_ci --k-arms 4
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

## Platform notes

All parallel cases (`n_workers > 1`) fork worker pools via
`multiprocessing.get_context("fork")`. On macOS, numpy links against Apple's
Accelerate framework for BLAS/LAPACK, which parallelizes routines like SVD
internally via Grand Central Dispatch (GCD) worker threads; forking from a
process that already has GCD state (from an earlier numpy/scipy call) can
leave that state inconsistent in the child, which then segfaults
(`EXC_BAD_ACCESS`, `"crashed on child side of fork pre-exec"` in the crash
report) the next time it calls into Accelerate -- e.g. `statsmodels`'
`MixedLM` REML fit (`cases/pvalues.py`'s `lmm`/`lmm_factorial`/`lmm_runs` ppi
tests), which uses SVD internally. `multiprocessing.Pool` has no way to
detect a worker dying this way, so `imap_unordered` just hangs forever
waiting for a result that will never arrive -- indistinguishable from a
stalled/slow cell from the outside. `simulations/harness/__init__.py` sets
`VECLIB_MAXIMUM_THREADS=1` (forcing Accelerate to run single-threaded, no
GCD dispatch) before anything else in the package can import numpy, which
sidesteps the hazard entirely. This is automatic and needs no action from
callers; it's documented here so a future segfault/hang during a parallel
run isn't mistaken for a new bug.

## Shared scenario library

`scenarios/synthetic.py` is built around ONE unified per-eval-type truth/
noise model, so every case describes the same handful of distributions
instead of each case (or each *mode* within a case) defining its own:

- **Shape catalog**: `BINARY_SHAPES` / `CONTINUOUS_SHAPES` / `LIKERT_SHAPES`
  / `GRADES_SHAPES` (indexed by eval type via `SHAPES_BY_EVAL_TYPE`), each a
  list of `ShapeSpec(label, eval_type, kind, params, custom_sampler,
  suite_tier)`. `kind="param"` shapes are smooth parametric families
  (binary: Bernoulli-Beta hierarchical with base success prob `p0`;
  continuous: `Beta(a, b)`; likert/grades: latent-`Normal(mu, total_std)`
  rounded/clipped to range). `kind="custom"` shapes (mixtures, zero/one-
  inflated, heavy-tailed, bimodal) are bespoke single-population generators
  with no closed-form CDF/quantile function. Both kinds support **any**
  group count `k` via the generator below, but custom shapes only at
  `base_corr=1.0` (fully shared item across groups) -- partial agreement
  for custom shapes would need an empirical-quantile copula (no closed-form
  `.ppf()` to invert through) and isn't implemented yet. `suite_tier`
  ("standard" | "expanded") gates which shapes a given `suite` argument
  includes.
- **Generator**: `sample_group_truth(shape, n, runs, k, icc, rng, corr=0.5,
  effects=None, heteroscedastic=False, base_corr=1.0) -> ndarray (k, n,
  runs)`. `icc` = between-item variance / total variance (same meaning
  everywhere it's used; may be a scalar or a length-k per-group sequence);
  `corr` = correlation between any two of the `k` groups' within-item
  noise; `base_corr` = correlation between any two of the `k` groups'
  item-level/base values (a Gaussian-copula "different items/judges,
  correlated quality" model for `kind="param"` shapes; broadcast-only,
  `base_corr=1.0`-only for `kind="custom"`); `effects[j]` is added to group
  `j`'s truth before clipping/rounding. `k=1` is the single-sample case (no
  inter-group correlation concept -- the lone group gets the full noise
  variance directly); `k=2` at `corr=0.5`, `base_corr=1.0`, scalar `icc`
  reproduces what was previously `build_pair_sources`' fixed, un-tunable
  shared/individual noise split exactly; `k>2` generalizes the same model
  to multi-arm/repeated-measures designs.
- **Consumers**: `build_single_sample_sources()` (k=1, `CISource`, full
  catalog including custom shapes), `build_pair_sources()` (k=2,
  `CIPairSource`, ICC x Cohen's-d grid, full catalog including custom
  shapes -- except its `run_noise_fracs` multi-run path, which stays
  param-shapes-only since it needs `base_corr<1.0`), `build_multiarm_sources()`
  (k>=2, `MultiArmSource`, ICC x Cohen's-d x eval-type sweep over the full
  catalog including custom shapes), and `build_judge_bias_sources()` /
  `generate_judge_bias_cell()`
  (`JudgeBiasSource` / `JudgeBiasCellData`, for `cases/pvalues.py`'s PPI
  mode -- picks ONE representative shape per eval type via `_ppi_shape`,
  layering judge-bias/noise/MNAR-label sweeps on top of the same truth
  model; `JudgeBiasSource.shape_label` can override the representative
  shape to sweep a pathological "custom" one instead -- see the `shape.*`
  scenarios in `build_judge_bias_sources()`). All four pull from the
  identical shape catalog and generator, so a given `(icc, cohens_d)` means
  the same thing in every one of them.
- The LLM-judge MEASUREMENT-ERROR model (`_jb_llm`/`_jb_llm_repeated`,
  layered on top of truth by `generate_judge_bias_cell` -- a separate axis
  from the shape catalog above; see `JudgeBiasSource`'s docstring) defaults
  to symmetric Gaussian noise but supports `JudgeBiasSource.noise_family =
  "contaminated"`: a zero-mean two-component variance-mixture (mostly a
  small width, occasionally -- probability `contam_frac` -- a much larger
  one, `contam_scale` times wider), solved so its TOTAL variance always
  equals what `noise_family="gaussian"` would use at the same `llm_noise`
  -- modeling "judge is mostly right, occasionally catastrophically wrong"
  as an opt-in robustness check rather than a default-model replacement
  (changing the default would invalidate this harness's exact-parity
  verification against the legacy scripts). For repeated-measures
  structures, the "is this a catastrophic-error item" REGIME is correlated
  across groups/conditions via the same copula-style mechanism `base_corr`
  uses (`repeated_corr` controls it, same parameter that correlates
  Gaussian noise magnitude in the default family) -- see
  `build_judge_bias_sources()`'s `noise_family.*` scenarios.
- `scenarios/real_data.py` -- OpenEval (downloaded from the HuggingFace Hub
  on first use) / Inspect AI (a CSV of locally-run results, see
  `simulations/collect_inspect_benchmarks.py`) corpus loaders, exposed
  through the same `CISource` interface as the synthetic generators
  (`generate(rng, n) -> ndarray`, `true_mean`) so a single simulation loop
  can run over either. Also has shared-item OpenEval / Inspect AI *pairing*,
  exposed as `CIPairSource` (`generate_pair(rng, n, runs) -> (a, b)`,
  `true_diff`) for `cases/ci_paired.py` and `cases/pvalues.py`'s `--mode
  pairwise`. `--data-source real` combines openeval + inspect for maximum
  real-data diversity, skipping inspect with a warning (rather than
  failing) if its CSV isn't present locally. `corpus_to_shape_spec(corpus)`
  wraps a real `Corpus` as a "custom" `ShapeSpec` (bootstrap-resampled, i.e.
  with replacement, unlike the without-replacement subsampling
  `corpus_to_ci_source`/`corpus_pair_to_ci_pair_source` use) so real
  marginal distributions can be dropped into any of `synthetic.py`'s
  k-group/icc/base_corr machinery -- currently unused (`ppi` mode still has
  no real-data variant; see below).
  `MultiArmCorpus`/`multiarm_corpus_to_source`/`build_real_multiarm_sources`
  are a separate, more direct real-data path for `cases/pvalues.py`'s
  `--mode multiarm`: for each benchmark, ALL of its available real models
  (not just a curated pair) are aligned on shared `item_id`, ordered by
  descending real corpus mean (arm 0 = the empirically best-performing real
  model, matching `MultiArmSource`'s "arm 0 carries the alternative
  hypothesis" convention). The "alt" condition returns those real,
  unpermuted arms (a genuine best-arm signal, no synthetic delta injected);
  the "null" condition (`delta == 0.0`) independently permutes each sampled
  item's k real scores across arm slots -- the same permutation-null idea
  as `corpus_pair_to_null_ci_pair_source`, generalized to k arms, so every
  emitted value is still real and unmodified but the expected across-arm
  difference is zeroed out. `MultiArmSource.max_k` bounds `--k-arms` to
  however many real models a benchmark actually has, the same way `max_n`
  bounds `--sizes`; oversized k/n cells are skipped with a warning rather
  than failing the run.
- `cases/ci_single.py`'s and `cases/ci_paired.py`'s `--nested-mode`
  (multi-run, parameterized by `run_noise_frac` rather than ICC -- `f_run =
  1 - icc`) is now an optional axis on the SAME `build_single_sample_sources`
  / `build_pair_sources` functions, not a separate generator catalog:
  - `build_single_sample_sources(suite, run_noise_fracs=None,
    heteroscedastic=False)`: when `run_noise_fracs` is given, every shape
    (including "custom" ones, via an MC-estimated base variance) gets one
    `CISource` per `f_run` with `generate_runs(rng, n, runs) -> (n, runs)`
    set; `None` (the default) keeps the original single-run-only behavior.
  - `build_pair_sources(..., run_noise_fracs=None, heteroscedastic=False,
    pairwise_noise_grid=False, cross_item_rho=0.7)`: when `run_noise_fracs`
    is given, sweeps `(f_a, f_b)` pairs (independent A/B run-noise, or the
    full cross product under `pairwise_noise_grid`) through
    `sample_group_truth`'s `base_corr` (= `cross_item_rho`, a Gaussian-copula
    "different items/judges, correlated quality" model -- 1.0 = fully
    shared item, the non-multi-run default) and per-group `icc` (`[1 - f_a,
    1 - f_b]`); `None` keeps the original icc_values x Cohen's-d sweep with
    `base_corr=1.0` unchanged. The asym-binary stress-test block gets a
    matching `redraw_frac` run-noise mechanism in this mode.
  - `sample_group_truth` itself grew the `base_corr` and per-group-`icc`
    parameters (plus `heteroscedastic`) to support this -- see its
    docstring. The pre-unification implementations are kept as
    `_legacy_build_multirun_sources()` / `_legacy_build_pair_multirun_sources()`
    (unused, reference-only, same pattern as the `_legacy_build_*` functions
    above).
  - `CISource.generate_runs(rng, n, runs) -> (n, runs)` (`None` for ordinary
    flat sources) is the field this populates; `CIPairSource.generate_pair`
    was already `runs`-aware so no interface change was needed there.

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
| `ci_single` | implemented | `sim_compare_boot.py` (single-sample) + `sim_compare_boot_nested.py` (mean phase, via `--nested-mode`) | synthetic, openeval, inspect, real (`--nested-mode`: synthetic only) |
| `ci_paired` | implemented | `sim_compare_boot.py` (pairwise) + `sim_tango_real.py` + `sim_compare_boot_nested.py` (pairwise phase, via `--nested-mode`) | synthetic, openeval, inspect, real (`--nested-mode`: synthetic only) |
| `pvalues` | implemented | `sim_compare_pvalues.py` (`--mode pairwise`/`multiarm`) + `sim_type_i_calibration.py` (`--mode ppi`); `--mode simultaneous_ci` is new (no legacy script) | `--mode pairwise`/`multiarm`/`simultaneous_ci`: synthetic, openeval, inspect, real; `ppi`: synthetic only |
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
- `ci_paired`'s real-data path (openeval/inspect/real) only supports `runs=1`
  (flat, single run per item) and is restricted to known-binary benchmarks,
  matching `sim_tango_real.py`'s `SINGLE_RUN_METHODS` scope. Multi-run real
  pairs (Tango multirun variants, nested-diff bootstrap, `lmm_diff`) are
  deferred to a future real-data extension.
- `--nested-mode` (both cases) is synthetic-only and, for `ci_paired`,
  `statistic=mean` only -- matching `sim_compare_boot_nested.py`, which
  never supported `--statistic median` for its multi-run paths.
- `official_args()` for both `ci_single` and `ci_paired` is synthetic-only --
  real-data sources need network/HF Hub access not assumed available in
  `--official-tests` runs. Run `--data-source openeval/inspect/real` manually
  when that access is available.
- `scenarios/synthetic.py`'s `"extreme"` suite is currently identical to
  `"expanded"` -- reserved for future stress-test scenarios as more cases
  are ported.
- `--mode ppi`'s Type-I sweep (`build_judge_bias_sources`) never sets
  `effect_size` above 0, so it can only show whether PPI correction controls
  false positives, never whether it retains the power to detect a genuine
  difference under that same bias -- a method could trivially "pass" Type-I
  calibration by rejecting almost nothing, ever. Two checks close that gap,
  both on by default (`--no-power-check` / `--no-comparison-check` to skip):
  - **Power check** (`build_ppi_power_sources`, `print_ppi_power_report`/
    `save_ppi_power_plot`): the SAME per-eval-type judge-bias severity as
    `build_judge_bias_sources`' `eval_type.*` baseline, but swept across a
    real `effect_size` grid (`PPI_POWER_EFFECT_FRACS`, expressed as a
    fraction of each eval type's own scale span via `_jb_effect_magnitude`,
    the effect-size analogue of `_jb_bias_magnitude`) instead of held at 0.
    `es=0.00` doubles as a Type-I cross-check against the main sweep.
    Binary is excluded: `effect_size` is an additive shift applied directly
    to a 0/1 truth draw in `generate_judge_bias_cell`, which only stays
    valid at `es=0`.
  - **Estimator comparison** (`build_ppi_comparison_label_frac_sources` +
    the power sources' effect_size grid, `PPIComparisonResult`/
    `run_ppi_comparison_simulation`/`save_ppi_comparison_plot`): for ONE
    representative paired-mean estimand (`paired_t`), rejection rate for
    five ways of turning (sparse human labels + biased LLM-judge scores)
    into a test -- `all_human` (oracle: classical test on the full, dense
    ground truth), `human_subset` (classical test on ONLY the labeled
    subset's truth -- the "why not just collect more human labels instead"
    baseline), `llm_only` (uncorrected LLM judge scores), `llm_impute` (LLM
    scores with labeled positions overwritten by the true label -- a naive
    missing-data-imputation baseline with NO PPI rectifier, showing that
    "filling in what you know" isn't the same as correcting for what you
    don't), and `ppi` (PPI-corrected). Swept two ways: vs. `effect_size` at
    fixed `label_frac` (reusing `build_ppi_power_sources`, tag `"power"`),
    and vs. `label_frac` at a fixed moderate `effect_size`
    (`build_ppi_comparison_label_frac_sources`, tag `"compare_label_frac"`)
    -- the latter answers "how many human labels does PPI actually need to
    approach all_human's power" directly, which the effect_size axis alone
    cannot. `JudgeBiasCellData.truth_x`/`truth_y` (dense, unmasked ground
    truth for the paired structure only) were added to
    `generate_judge_bias_cell`'s return value to support the `all_human`/
    `human_subset` arms; every other test structure's dense truth is
    discarded as before. Sequential only (`run_ppi_comparison_simulation`)
    -- the comparison grid (24 scenarios total) is small enough that
    `run_ppi_simulation`'s parallel/progress-dict machinery would be pure
    overhead.
  - The power check's baseline (`build_ppi_power_sources`) only exercises
    ONE bias configuration: `bias_type="differential"` with the fixed judge
    bias always OPPOSING the injected real effect (see
    `build_ppi_power_reinforcing_sources`' docstring for the sign
    arithmetic) -- which produces the "cancellation dip" visible in the
    uncorrected power curve as effect_size approaches the bias magnitude.
    Two more variants close the resulting gaps, each with its own plot
    rather than folded into the main one:
    `build_ppi_power_reinforcing_sources` (bias_delta's sign flipped, so
    bias and effect stack instead of fighting -- arguably the more
    dangerous real case, since the uncorrected curve just climbs faster
    with no visible anomaly to flag it) and `build_ppi_power_nobias_sources`
    (`bias_type="none"` -- the "no free lunch" check: does PPI correction
    cost power when there's nothing to correct for?). All three share the
    same `"<prefix>.<eval_type>.es=<frac>"` name shape (see
    `_PPI_POWER_NAME_RE`), so `print_ppi_power_report`/`save_ppi_power_plot`
    are reused as-is for the reinforcing/no-bias variants (just with a
    `header`/`title_suffix` override); only the opposing-vs-reinforcing
    overlay (`save_ppi_power_direction_plot`) is a dedicated new function.
  - Every check above (power, direction, no-bias, and the 5-way comparison)
    fixes N (total items) at 100 and only varies N_lab through `label_frac`
    -- so none of them can answer whether calibration/power depends on the
    RATIO N_lab/N or the ABSOLUTE N_lab count. `PPI_COMPARISON_LABEL_FRACS`
    was also fixed here: at N=100, `_JB_MIN_LAB`'s floor made label_frac=0.05
    and 0.10 both resolve to the SAME N_lab=15, which the plot's x-axis
    (nominal label_frac) hid -- `PPIComparisonResult.n_lab` now reports the
    REALIZED count, and both the label_frac comparison row and its grid
    (`PPI_COMPARISON_LABEL_FRACS = (0.15, 0.20, 0.30, 0.40)`) were chosen to
    avoid collisions. `build_ppi_nlab_grid_sources(effect_frac)` crosses N
    (`PPI_NLAB_GRID_N_VALUES`, 30-400) and N_lab (`PPI_NLAB_GRID_NLAB_VALUES`,
    15-80) INDEPENDENTLY, back-solving `label_frac = n_lab / n` per cell so
    the floor never binds; `effect_frac=0.0` is the calibration question
    (Type-I ~ alpha across the whole plane), a nonzero `effect_frac` is the
    power companion at the same grid. One representative eval type
    (continuous) only, same "one estimand is the point" scoping as the 5-way
    comparison's paired_t choice. `save_ppi_nlab_grid_plot` renders both as
    heatmaps (diverging colormap centered on alpha for calibration,
    sequential for power) so a reader can scan a ROW (N's effect at fixed
    N_lab) vs. a COLUMN (N_lab's effect at fixed N) directly -- confirmed
    empirically that power is much more sensitive to N_lab than to N at
    small N_lab (e.g. N_lab=15 stays low across N=60..400), consistent with
    PPI's two-term variance (`Var(unlabeled)/N + Var(rectifier)/N_lab`) --
    the rectifier term dominates until N_lab is large enough, at which point
    adding more N (cheap LLM-judged items) barely moves power further.
- There is no separate `ppi_calibration` case: `sim_type_i_calibration.py`
  was folded into `cases/pvalues.py`'s `--mode ppi` instead of becoming a
  third file, since both halves of `pvalues` answer "is this statistical
  decision trustworthy" at different levels of the stack (raw CI/p-value
  procedures vs. high-level `evalstats.tests` wrappers with PPI), sharing
  report/plot/CLI scaffolding but not scenario generators.
- `cases/pvalues.py`'s `--mode pairwise`/`--mode multiarm`/`--mode ppi` all
  reuse the SAME shape catalog and `sample_group_truth` generator described
  above. None of the three numerically matches its legacy script anymore
  (`sim_compare_pvalues.py`'s own pairwise/multi-arm generators,
  `sim_type_i_calibration.py`'s "normal"/"likert"/"skewed" truth model) --
  this was a deliberate, agreed-upon trade: cross-mode consistency (one
  truth-generating process per eval type, describable once in the paper)
  over per-mode exact parity with each legacy script. Verification for all
  three is by sanity check (Type-I error ~ alpha, power increasing with
  effect size and n) -- see Verification below.
- `cases/pvalues.py`'s `--mode ppi` sweeps `eval_type` in
  `{continuous, likert, grades}` in full, picking one representative shape
  per eval type (`cont-right-skew` / `likert-mid` / `grades-mid`) rather
  than sweeping the full shape catalog the way `--mode multiarm` does --
  judge-bias/noise/MNAR-label parameters are PPI's actual axis of interest,
  not distribution shape. `binary` (representative shape `p=0.50`) is
  supported only for tests whose estimand is the paired-mean difference
  (a proportion is just the mean of a 0/1 variable, so PPI's rectifier
  applies unchanged) -- `ttest`/`ttest_welch` (two independent groups),
  `paired_t` (paired), `bayes_bootstrap` (paired, Dirichlet-weighted), and
  `tango_score` (paired, score interval) -- as a single baseline-settings
  scenario rather than being swept across every other factor.
  `bayes_bootstrap` PPI-corrects the identical paired-mean estimand
  as `paired_t`, but via Dirichlet-weighted (Bayesian) bootstrap resampling
  (`evalstats.tests._ppi_paired_bayes_bootstrap`) instead of
  `evalstats.ppi.correct`'s classical resampling -- kept as a validated
  alternative method, not a recommended default: real-data testing found
  it underperforms, so `paired_t` remains the reasonable default for
  binary p-values. `bootstrap_t` PPI-corrects the SAME paired-mean
  estimand via a studentized-bootstrap pivot
  (`evalstats.tests._ppi_paired_bootstrap_t`), generalizing
  `evalstats.core.resampling.bootstrap_t_ci_1d`'s per-replicate SE
  (`std/sqrt(n)`) to PPI's two-independent-term variance
  (`Var(unlabeled term)/N + Var(rectifier term)/n_lab`) -- numeric
  (continuous/likert/grades) ONLY, not extended to binary, since its
  value is specifically for resampling-based CI estimation on numeric
  data at N>=50 (`ci_paired.py`), not pairwise binary p-values.
  `tango_score` is the mirror image -- binary ONLY, not numeric --
  PPI-correcting `evalstats.core.resampling.tango_paired_ci`'s score
  interval (`evalstats.tests._ppi_paired_tango`): its variance term
  `(n10+n01)/n^2 - (n10-n01)^2/n^3` is exactly `Var(diffs, ddof=0) / n`,
  so it generalizes to PPI's two-term variance by substituting an
  effective n (`n_eff = Var(unlabeled diffs) / V_hat_PPI`) into the SAME
  Wilson-style shrinkage formula -- fully closed-form (no bootstrap), and
  reduces EXACTLY to the original uncorrected formula when the "labeled"
  subset is the full sample with no judge error (verified numerically).
  `mcnemar` is deliberately NOT PPI-corrected: its defining feature is an
  exact small-sample binomial test on discordant-pair counts, and a
  PPI-corrected numerator is generally non-integer, breaking that
  exactness -- left as future work. The rank-based family
  (`mw`/`wilcoxon`/`friedman`/`kruskal`) and ANOVA/LMM
  stay continuous/likert/grades-only: they assume a scale that breaks down
  under binary's massive ties, and the additive noise/bias/slope
  judge-miscalibration model (`scenarios.synthetic._jb_llm`/
  `_jb_llm_repeated`) has no binary equivalent for those structures -- see
  `_jb_llm_binary`/`_jb_llm_repeated_binary`, a confusion-matrix
  (flip-probability) model used only for the two-group-independent and
  paired structures.
- `judge_bias.py` (planned, for `alignment`) will keep
  `sim_alignment_methods.py`'s agreement-rate proxy-noise model as its own
  named generator, separate from `scenarios/synthetic.py`'s
  `build_judge_bias_sources()`/`generate_judge_bias_cell()` (ported from
  `sim_type_i_calibration.py` for `cases/pvalues.py`'s PPI mode) -- they
  model different things (agreement-rate noise vs. bias/slope/MNAR
  miscalibration).

## Verification

Two generations of verification exist here, reflecting the shape-catalog
unification described above:

**Pre-unification** (`ci_single`, `ci_paired`, and `pvalues`' first port),
checked against the original legacy scripts for exact numeric parity, byte
for byte, at fixed seeds:
- `ci_single`'s synthetic path vs. `sim_compare_boot.py --estimand mean`:
  all per-(eval_type, scenario, n, method) coverage/width matched to 8
  decimal places (binary, continuous), except `beta` (itself an unseeded
  parametric bootstrap, inherent MC noise on both sides).
- `ci_paired`'s synthetic path vs. `sim_compare_boot.py --estimand
  pairwise`, including `--include-null`: 0 mismatches / 384 rows.
- `ci_single --nested-mode` vs. `sim_compare_boot_nested.py --estimand
  mean`: 0 mismatches / 880 rows (except `beta`, same exception).
- `ci_paired --nested-mode` vs. `sim_compare_boot_nested.py --estimand
  pairwise`, including `--include-null`: 0 mismatches / 600 rows.
- `pvalues --mode multiarm` vs. `sim_compare_pvalues.py`'s multi-arm phase:
  generator output, `_compute_multiarm_metrics`, and the full cell worker
  all matched exactly (80+ rows, 5 correction strategies).
- `pvalues --mode ppi` vs. `sim_type_i_calibration.py`'s `_run_one`: 0
  mismatches across 43 scenarios x 3 reps x 11 tests = 1419 reject-decision
  pairs, including every bootstrap-based PPI correction and LMM Wald test.

**Post-unification** (current state): the shape catalog merge intentionally
broke exact parity with the legacy scripts for `build_pair_sources`,
`build_single_sample_sources`, `build_multiarm_sources`, and
`build_judge_bias_sources`/`generate_judge_bias_cell` -- this was an
explicit, agreed trade (cross-mode truth-distribution consistency over
per-mode legacy-script parity). Verification shifted to:
- **Algebraic + numeric equivalence of the refactor itself**: the new
  `sample_group_truth(k=2, corr=0.5, ...)` was derived to exactly reproduce
  the *pre-unification* `build_pair_sources`' fixed shared/individual noise
  split (verified algebraically: both reduce to the same `noise_std`).
  Confirmed numerically against `_legacy_build_pair_sources` (the kept,
  unused pre-merge implementation) for every (eval_type, shape, icc,
  cohens_d) combination present in both catalogs: 0 mismatches across 252
  continuous/likert/grades rows (matched by label) + 36 binary rows
  (matched by underlying `p0`, since binary shape labels were renamed from
  `binary-balanced`/etc. to `p=0.50`/etc. when the catalogs merged), each
  checked at 3 seeds.
- **Sanity checks** (Type-I error ~ alpha, power increasing with effect
  size and n, FWER controlled by correction strategy, PPI-corrected rates
  near nominal alpha vs. heavily inflated uncorrected rates) across all
  three `pvalues` modes and `ci_single`/`ci_paired`, post-unification, at
  moderate rep counts. No regressions found beyond pre-existing, shape-
  specific fragilities already present in the legacy catalogs (e.g.
  `logit_t`'s poor coverage on zero/one-inflated continuous shapes and on
  likert/grades scores generally -- confirmed present identically in
  `_legacy_build_single_sample_sources`, not introduced by the merge).
- **`--nested-mode`'s run_noise_fracs unification** (`build_single_sample_sources`
  /`build_pair_sources` absorbing `_legacy_build_multirun_sources`/
  `_legacy_build_pair_multirun_sources`): algebraically derived (and then
  numerically confirmed) that `sample_group_truth(k=1, icc=1-f_run, runs>1)`
  reproduces every "param" shape's run-noise-frac formula exactly (binary's
  Beta-Bernoulli concentration, continuous's fixed-base/growing-noise model,
  likert/grades' fixed-total-variance split) -- mean/within-item-variance
  matched the legacy generator to MC precision across binary, continuous,
  likert, grades, and every custom shape (zero/one-inflated, mixtures,
  heavy-tail), homoscedastic and heteroscedastic. For pairs,
  `sample_group_truth`'s new `base_corr` (Gaussian-copula item-level
  correlation, generalizing `cross_item_rho` to arbitrary k) and per-group
  `icc` reproduce `_legacy_build_pair_multirun_sources`' mean/variance/
  cross-correlation across all four eval types once `corr=0.0` is passed for
  the noise layer (the legacy pair-multirun model never correlated noise
  across A/B, only the base) -- the asym-binary stress-test's `redraw_frac`
  mechanism (item-vs-fresh-draw mixing) matched bit-for-bit (0 mismatches
  across all 12 specs, exact RNG-stream parity, since that block's generator
  was copied verbatim into the new `build_pair_sources` path).
- **Custom shapes at k>=2** (`build_pair_sources`'s and `build_multiarm_sources`'
  flat paths, plus three new `shape.*` PPI scenarios): mixtures, zero/one-
  inflated, and heavy-tailed shapes were previously k=1-only, exempting the
  catalog's most realistic-pathology shapes from the pairwise/multi-arm/PPI
  power and Type-I-error checks. Sanity-checked post-extension: Type-I error
  for every custom-shape pair sits within MC noise of nominal alpha=0.05
  across all bootstrap/permutation/parametric pairwise methods (reps=200);
  multi-arm FWER stays controlled under correction for every custom shape;
  PPI correction still pulls the corrected rejection rate back to ~alpha for
  all three new `shape.*` scenarios even when the uncorrected rate is
  severely inflated (e.g. `shape.cont-zero-inflated`'s uncorrected rate hits
  ~1.0 across every test -- a fixed additive judge bias is enormous relative
  to a distribution where 70% of mass sits at exactly 0 -- while the
  corrected rate stays near alpha for all of them, a legitimate and
  reportable robustness finding, not an artifact).
- **Custom shapes at base_corr<1.0** (`_custom_shape_inverse_cdf`'s
  empirical-quantile copula, unblocking custom shapes in
  `build_pair_sources`' `run_noise_fracs` multi-run path): marginal mean and
  point-mass fraction (e.g. `cont-zero-inflated`'s 70% spike at exactly 0)
  are preserved exactly regardless of base_corr, confirming the copula
  doesn't distort the shape's own distribution. Realized rank/Pearson
  correlation were checked against the Gaussian-copula's known identity
  (Spearman = (6/pi)*arcsin(base_corr/2) for continuous marginals) and
  matched exactly for a tie-free custom shape (`cont-mixture`); shapes with
  large point masses (`cont-zero-inflated`) realize a further-compressed
  correlation due to tied ranks, as expected -- documented as a caveat in
  `sample_group_truth`'s `base_corr` docstring rather than treated as a
  defect, since the SAME caveat applies to `base_corr`'s pre-existing
  binary/continuous "param"-shape copula path too (this was a latent
  inaccuracy in last session's docstring, corrected here: `base_corr` is
  the latent Gaussian-copula correlation, not the realized output
  correlation, for every shape except likert/grades, whose base is
  additively decomposed in Gaussian space directly and so realizes
  `base_corr` as its Pearson correlation exactly, modulo likert's rounding
  and any shape's boundary clipping). Type-I error for a custom-shape pair
  under the new `base_corr<1.0` multi-run path was cross-checked against
  the same shape's `base_corr=1.0` flat-path Type-I error and found
  consistent (~0.90 coverage for the unbiased `bootstrap` method on
  `cont-zero-inflated` either way) -- a pre-existing, shape-specific
  fragility, not introduced by this extension.
- **Contaminated judge noise** (`_contaminated_noise_stds`, `noise_family`
  on `_jb_llm`/`_jb_llm_repeated`/`JudgeBiasSource`): variance-preservation
  confirmed exactly (a `noise_sd=0.20` zero-mean two-component mixture at
  `contam_frac=0.10, contam_scale=4.0` realizes total variance 0.0400,
  matching `noise_sd**2` to 4 decimal places at n=500,000) and the resulting
  distribution is correctly heavy-tailed (excess kurtosis ~12.9 vs. the
  Gaussian baseline's ~3.0 at the same total variance). The correlated-
  regime mechanism was checked directly: at `repeated_corr=0`, the
  co-occurrence rate of "catastrophic" judge errors across two repeated
  conditions matches the independence baseline (~0.011 vs. predicted
  ~0.011); at `repeated_corr=0.9` it's ~3x higher (~0.030), confirming the
  regime-correlation knob works as intended. End-to-end: PPI correction
  still pulls the corrected rejection rate back to ~alpha across all three
  `noise_family.*` scenarios and every test, even though the uncorrected
  rate is the same maximal ~1.0 as the Gaussian-noise differential-bias
  baseline -- i.e. PPI's Type-I-inflation fix is not an artifact of
  assuming symmetric judge-noise.
