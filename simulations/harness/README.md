# evalstats simulation harness

Centralized CLI for evalstats' calibration/comparison Monte Carlo simulations,
consolidating the scripts in `simulations/*.py` into one runner so results
can be reproduced for the paper's Supplementary Material with a single
command.

```
python -m simulations.harness.cli --list-cases
python -m simulations.harness.cli ci_single --reps 50 --sizes 10 20
python -m simulations.harness.cli ci_single --data-source dove
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
exist as code yet (e.g. `pvalues`' `mcnemar`/`holm`/etc. should be added
when `cases/pvalues.py` is actually ported, not before).

## Cases

| Case | Status | Ported from | Data sources |
|---|---|---|---|
| `ci_single` | implemented | `sim_compare_boot.py` (single-sample) + `sim_dove.py` | synthetic, dove, openeval, all |
| `ci_paired` | implemented | `sim_compare_boot.py` (pairwise) + `sim_tango_real.py` | synthetic, dove, openeval, inspect, all (all = dove+openeval) |
| `ci_nested` | planned | `sim_compare_boot_nested.py` | synthetic only |
| `pvalues` | planned | `sim_compare_pvalues.py` | synthetic only |
| `alignment` | planned | `sim_alignment_methods.py` | synthetic only |
| `ppi_calibration` | planned | `sim_type_i_calibration.py` | synthetic only |

`bayes_evals.py` is a vendored dependency (Beta-Bernoulli / Dirichlet-
multinomial Bayesian CI methods), not a case -- it's imported by cases the
same way `evalstats.core.resampling` is.

## Known exceptions (documented, not silently dropped)

- `ci_nested`, `alignment`, and `ppi_calibration` are synthetic-only for now.
  Real-data extension is future work: `ci_nested` would need a real source
  with repeated per-item runs; `alignment` is a natural fit for OpenEval
  (which has paired human + LLM labels) once ported; `ppi_calibration` needs
  ground truth plus a judge-bias model that real datasets don't provide.
- `ci_paired`'s real-data path (dove/openeval/inspect) only supports `runs=1`
  (flat, single run per item) and is restricted to known-binary benchmarks,
  matching `sim_tango_real.py`'s `SINGLE_RUN_METHODS` scope. Multi-run real
  pairs (Tango multirun variants, nested-diff bootstrap, `lmm_diff`) are
  deferred to a future `ci_nested` real-data extension.
- `official_args()` for both `ci_single` and `ci_paired` is synthetic-only --
  real-data sources need network/HF Hub access not assumed available in
  `--official-tests` runs. Run `--data-source dove/openeval/inspect` manually
  when that access is available.
- `scenarios/synthetic.py`'s `"extreme"` suite is currently identical to
  `"expanded"` -- reserved for future stress-test scenarios as cases are
  ported (e.g. `sim_compare_pvalues.py`'s near-ceiling/ultra-sparse variants).
- `judge_bias.py` (planned, for `alignment` + `ppi_calibration`) will keep
  `sim_alignment_methods.py`'s agreement-rate proxy-noise model and
  `sim_type_i_calibration.py`'s bias/slope/MNAR judge-miscalibration model as
  separate, named generators rather than forcing them into one -- they model
  different things.

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
