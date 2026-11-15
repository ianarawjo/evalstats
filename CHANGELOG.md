# Changelog

## 0.3.4

`judge_alignment` now takes the same evaldata as `compare()` and reports everything
in one call: an ordinal agreement score for half-point scales, and rho^2 and `N_eff`
for the tests `compare()` will run.

### Added

- `judge_alignment(evaldata, ...)` on data with 2+ conditions reports rho^2 and
  `N_eff` for the tests `compare()` will run on it, in `AlignmentResult.comparison`
  and in `.summary()`. The design is read from the item ids (the same items in every
  condition is within-subjects; items that never repeat is between-subjects); pass
  `design=` when the ids don't settle it. `test=` is now honoured on this form to
  report one named test instead. `compare()` and `judge_alignment` read one table of
  which test runs for which design, so they report the same numbers.
- `judge_alignment(..., n_total=)`: the size of the full dataset, for when the items
  passed in are only the human-labeled validation set. `N_eff` is then computed for
  labeling that many items out of `n_total`. An int, or a `{condition: size}` dict
  for several conditions.
- `score_type=` on the evaldata form of `judge_alignment`, as on the array form.

### Changed

- Scores on any bounded quantization grid, such as half-point ratings, are now
  detected as `likert`, the same detector `compare()` uses, so they get
  quadratic-weighted kappa instead of being treated as continuous. Declaring
  `score_type="likert"` on scores that are not on a grid raises, rather than
  producing a meaningless kappa. Kappa now counts categories as positions on the
  grid, so a level nobody used still counts toward the distance between its
  neighbours; this can change kappa on whole-number data with an unused level.
- Test-specific alignment is reported as rho^2, with its confidence interval,
  rather than rho. The raw Pearson and Spearman lines are still r.
- With every item labeled and no `n_total=`, the report leaves out `N_eff`, which
  would only restate the sample size.
- The single-condition form prints rho^2 and `N_eff` when `test="mean_estimate"`.
- The label-position check is skipped when `n_total=` says the rows passed in are
  only a validation subset, where position carries no information.

### Removed

- `judge_alignment({condition: (judge, human)}, design=..., test=...)`. Pass the
  evaldata you give `compare()`; the conditions, design and tests are read from it.

## 0.3.3

Two fixes to the effect size reported beside a judge-corrected comparison.

### Fixed

- The judge-corrected rank-biserial could fall outside `[-1, 1]`. The corrected
  estimand is a sum -- a human term plus a weighted judge correction -- rather
  than a normalized rank count, so nothing bounded it. With few labeled items
  and a large true effect it left the range in roughly one run in ten. The
  reported effect size is now clipped, in both the paired and unpaired paths;
  the estimate, interval and p-value are unchanged.
- A paired comparison reported two different effect sizes: the printed table
  showed the corrected value while `pairwise.get(a, b).rank_biserial` returned
  the uncorrected one. Both now read the same field.

## 0.3.2

A public, JSON-safe result API for tools that show evalstats results outside the
terminal (first user: ChainForge's Vis Node). ChainForge can require this version
in place of `evalstats>=0.3.1` and drop its imports of private helpers.

### Added

- `ComparisonResult.rank_bands()`: entities in leaderboard order with `rank`,
  `band` (1 = top), `verdict` (`likely_best`, `tied_for_best`,
  `significant_drop_off`), `verdict_text`, `tied_with`, factor `levels`, mean and
  CI. The rows, order and wording of the executive summary.
- `ComparisonResult.methods()`: codes and display names for the design, data kind,
  mean CI, pairwise CI and simultaneous-CI method, the p-value test and its
  correction, the omnibus test, the rank-band criterion, PPI, bootstrap count,
  seed and evalstats version.
- `ComparisonResult.notes`: `evalstats.Note` records (`code`, `message`,
  `severity`, `entities`) for the warnings `compare()` raised. They are collected
  per call in a context variable, so concurrent calls need no
  `warnings.catch_warnings` or lock. The warnings are still emitted.
- `to_dict()` adds per entity `levels`, `rank`, `band`, `verdict`,
  `verdict_text` and `tied_with`; per pair `a_levels`, `b_levels` and
  `significant`; and top-level `order`, `notes` and `methods`. `to_frame()` adds
  `rank`, `band`, `verdict` and `significant`.
- Typed errors, each a `ValueError` subclass: `InsufficientItemsError`
  (`n_items`, `min_items`), `MissingCellsError` (`missing`, `n_missing`),
  `TooFewGroupsError` (`n_groups`, `factor`) and `AmbiguousLabelsError`
  (`labels`). `evalstats.MIN_ITEMS` is the 15-item floor.
- `evalstats.complete_items(evaldata, factors)` drops items that lack a score in
  any (factor levels x run) cell and returns a `CompletenessReport` of what it
  dropped. It never changes the design.
- `PairedDiffResult.p_test` and `PairwiseMatrix.p_value_test` name the test
  behind `p_value`; `evalstats.core.paired.P_TEST_NAMES` gives display names.

### Changed

- **One pairwise p-value.** `PairedDiffResult.p_value` is always the p-value the
  pairwise table prints: bootstrap-t with Romano-Wolf step-down correction when
  that correction resolves, the method's own test on binary data (McNemar mid-p,
  or inverted from the CI for multiple runs), and Wilcoxon signed-rank otherwise,
  corrected as the table footer states. Previously, for numeric data without Romano-Wolf (N < 30 with
  k >= 3, or k = 2), the table printed `wilcoxon_p` while `p_value` held the CI
  method's own p-value. Printed p-values, CIs and means are unchanged.
- `pair.summary()` reports that same p-value. It used to prefer Wilcoxon
  whenever one existed.
- The table footer names the correction for McNemar and sign-test p-values, which
  it had labelled "uncorrected" although they were corrected.
- With an explicit `method=` on binary data, the default p-value is the method's
  own test rather than Wilcoxon on 0/1 scores.
- `to_dict()` returns only JSON types; non-finite numbers become `None`.
- Rank bands, `significant` and `unbeaten` share one criterion: the simultaneous
  CI excludes 0, or corrected p < alpha when simultaneous CIs are off.
  `unbeaten` used p < alpha before.
- Leaderboard order is descending mean with ties in data order, in the executive
  summary and the pairwise table alike (the table broke ties by label). The
  executive summary uses the result's alpha, not the global one.
- Two-factor cell labels (`"model / prompt"`) that would collide raise
  `AmbiguousLabelsError`. Factor levels come from the data, not from splitting
  labels.
- Romano-Wolf and max-T p-values name their test as bootstrap-t, with Romano-Wolf
  or max-T as the correction, in `.summary()` and `methods()`. They used to name
  the correction as the test.
- `MissingCellsError` suggests `evalstats.complete_items()` first, and
  `TooFewGroupsError` from `compare()` names the factor with too few levels.
- `import evalstats` no longer imports matplotlib; plotting functions load it on
  first use.
- `pairwise_test="nemenyi"` computes Nemenyi p-values without `omnibus=True`
  (k >= 3).
- A Wilcoxon p-value for a pair whose differences are all zero is 1.0; it was
  missing before.
- `pairwise_test="nemenyi"` with two entities reports the default p-values and
  adds a `nemenyi_unavailable` note; `.summary()` used to show a column of N/A.
- `.summary()` labels intervals with the result's alpha everywhere. Two-factor
  summaries printed "95%" and "α=0.05" for other alphas, and the rank-band line
  and pairwise legend used the global alpha.

### Removed

- `PairedDiffResult.wilcoxon_p`. Request Wilcoxon with
  `compare(..., pairwise_test="wilcoxon")`; it is then `p_value`.
- `compute_wilcoxon=` on `pairwise_differences()` and `all_pairwise()`.
- `p_source=` on `plot_critical_difference()`.
- `summary(p_value_method=...)` no longer switches tests at print time. A value
  naming a different test than the stored one raises `ValueError`; `None` still
  hides the column.

### Simulations

- `simulations/harness/cases/compare_e2e.py` scores Type-I error and power from
  `p_value`. In the numeric N = 15, k > 2 cells that was the CI method's p-value
  rather than the printed Wilcoxon p-value; re-run it for the final numbers.
