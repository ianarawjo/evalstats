# Between-subjects (unpaired) support — executive summary

**Branch:** `worktree-between-subjects-support` (fresh worktree, per your instruction)
**Status:** Implementation complete, battle-tested, independently reviewed, all findings fixed, code shared with the paired path everywhere it genuinely could be, Pareto-front support added. **Nothing is committed yet** — per standing project guidance, I don't commit/merge without your go-ahead.

## Bottom line

`compare()` now supports genuinely between-subjects (unpaired) data via a new `design=` parameter, alongside the existing within-subjects (paired) path — verified **behaviorally unchanged** for every existing call pattern, both by tracing the code and by running the full existing test suite around it, plus byte-for-byte output diffs across representative scenarios.

```python
r = es.compare(evaldata, factors="model", metric="score")
# ValueError: Data for factor 'model' looks between-subjects (items are not
# shared across the compared groups)... Pass design="unpaired" to run the
# between-subjects comparison instead, or design="paired" to force it anyway.

r = es.compare(evaldata, factors="model", metric="score", design="unpaired")
r.summary()   # per-group means, Kruskal-Wallis/ANOVA omnibus, Bonferroni/Holm pairwise table
r.to_frame()  # one row per pairwise comparison

# Pareto-front analysis now works for between-subjects data too:
r = es.compare(evaldata, factors="model", metric="accuracy", design="unpaired",
                secondary_metric={"latency_ms": "min"})
r.pareto_status            # {"B": frontier, "A": dominated, ...}
r.pareto_frontier_probability
```

## What was built

1. **`evalstats/core/design.py`** — `detect_paired()`, moved out of `labeling.py` into a shared leaf module (zero behavior change).
2. **`evalstats/config.py`** — `AUTO_UNPAIRED_METHOD_TABLE`: binary → one-way ANOVA/Welch's t-test (Δp); continuous/likert/grade → Kruskal-Wallis/Mann-Whitney U (θ = P(a>b)), per your decision. The table's `family` field now genuinely drives dispatch (a review finding — see below).
3. **`evalstats/core/unpaired.py`** — the engine: `compare_unpaired()`, `GroupComparisonResult`/`GroupDiffResult`, Bonferroni-corrected CIs + Holm-corrected p-values, PPI support, synthetic-item-column fallback, and now `secondary_metric=` Pareto-front support (`pareto_bootstrap_unpaired` in `core/pareto.py` — a new function, the existing paired-path `pareto_bootstrap` is untouched).
4. **`evalstats/core/summary_unpaired.py`** + **`evalstats/core/summary.py`** — the console `.summary()` printer. This went further than originally planned: after you asked me to check whether shared machinery actually stayed shared, I found the PPI banner, per-entity means table, and pairwise comparison table were each either duplicated or hand-rolled separately from the paired path. All three are now **the same functions** the paired path calls — see "Code sharing" below.
5. **`design=` wiring in `evalstats/api.py`** — inserted as a self-contained block right before the existing paths A/B/C, that either returns/raises immediately or falls through completely unchanged.
6. **`tests/test_unpaired.py`** — 52 tests covering the engine, reporting surface, PPI, Pareto, and `design=` routing.
7. **`simulations/investigate_unpaired_battle_test.py`** — battle-test/calibration script, now with three parts: crash grid (192 combos), Type-I/power calibration, and a Pareto-front crash grid (48 combos).

## Code sharing with the paired path (your specific ask)

You asked directly: is the unpaired output actually sharing machinery with the paired path, or is there duplicate code that should be unified? The honest answer at the time was "partially" — some low-level primitives were shared, but the PPI banner, means table, and pairwise table were each separate implementations that merely looked similar. You asked me to unify them properly, including editing the paired path's own functions, with care and regression testing. Done:

- **PPI banner** — extracted into one shared `_print_ppi_banner()`, called by both paths. Small, low-risk.
- **Per-entity means table** — `_print_mean_advantage()` in `core/summary.py` now takes plain arrays (labels/mean/std/ci_low/ci_high/multi_ci) instead of a paired-specific `RobustnessResult`, so both paths call the literal same function. Verified byte-identical paired output before/after.
- **Pairwise comparison table** — the bigger piece. The paired path's `_print_pairwise_section` is ~420 lines handling six CI/p-value method families (Wilcoxon, bootstrap, Romano-Wolf, Newcombe, sign-test, max-T) plus Friedman/Nemenyi and critical-difference rank bands — none of which apply to unpaired data. I refactored it into `_prepare_paired_pairwise_rows()`/`_prepare_unpaired_pairwise_rows()` (each resolving their own method-specific logic into a common row+metadata shape) feeding one shared rendering core. Verified **byte-identical paired output** across three representative scenarios (default Wilcoxon/Romano-Wolf, binary/Newcombe, explicit bootstrap) plus a manual Nemenyi check, by diffing against snapshots taken before the refactor. As you flagged separately, the Behavioral Agreement (McNemar-style pass/fail) subsection was pulled into its own `_print_behavioral_agreement_section()`, paired-only, since it needs the same item scored by both entities — no between-subjects equivalent exists.
- **Pareto-front section** — turned out to already be dict-generic (`_print_pareto_section` only ever reads `.labels`/`.mean`/`.ci_low`/`.ci_high` off whatever's in the `pareto` dict). A tiny adapter (`_GroupStatsAsRobustness`) lets it render the unpaired case unmodified — including the ASCII scatterplot.

**Side effect worth knowing about:** the unpaired pairwise table's format changed as a result. It now shows an interval-plot bar per comparison (previously text-only) and numeric p-values with significance stars, matching the paired path exactly — replacing the old "Verdict: significant (A < B)" text column. The dominance family's θ is now shown as a signed deviation from its null (`Δθ`, e.g. `-0.40` instead of raw `θ=0.10`) so the shared axis math (which assumes a zero-centered quantity, same as the paired path's "Left − Right") applies uniformly; Δp for binary data is unaffected since its null was already 0. All genuinely new/better in my judgment, but flagging since it's a visible format change from what I showed you earlier in this thread.

## Pareto-front support for between-subjects data (your decision to build it)

`secondary_metric=` now works under `design="unpaired"`. The statistical approach differs from the paired path by necessity: the paired path draws one *shared* per-item bootstrap index applied to every entity (valid because every entity shares the same items); between-subjects groups have no shared item pool, so each group's own rows are resampled *independently*, while still preserving each row's own primary/secondary pairing (same reviewer, same response). New `pareto_bootstrap_unpaired()` in `core/pareto.py` implements this — the existing `pareto_bootstrap()` is completely untouched. `classify_pareto_status()` (frontier/dominated/ambiguous classification) is reused unchanged, since it only consumes the bootstrap's output shape, not how it was produced.

`GroupComparisonResult.pareto_status`/`.pareto_frontier_probability` mirror `ComparisonResult`'s own attributes exactly. Handles unbalanced groups, k=2 through k=6+, both `min`/`max` directions, PPI alongside Pareto, and row-level NaN in either metric (dropped jointly to preserve the pairing) — all confirmed via the new battle-test grid (48/48 passing) and 10 dedicated pytest tests.

## Bugs found and fixed (9, all from before the Pareto/sharing work — see prior summary detail if useful)

Three review passes beyond the initial implementation: my own integration review, a battle-test grid + calibration check, and an independent review agent with no visibility into my own work. 3 critical (multi-run N-inflation, a ZeroDivisionError in PPI Kruskal-Wallis at k=2, NaN silently poisoning CIs/crashing), 4 should-fix (silently-dropped `secondary_metric=`/`method=`/`score_range=`, a decorative routing-table field), 2 minor. All fixed with regression tests; final crash grid was 192/192 clean before the Pareto work started, still 192/192 (plus 48/48 Pareto-specific) now.

## Paired-path safety (your top priority, held throughout)

- Traced every line inserted into `compare()`: mutates only new local variables, or returns/raises immediately.
- `design="paired"` never evaluates the new branches — provably a no-op. `design="auto"` on paired data falls through unchanged.
- Verified via **byte-identical output diffs** (not just "tests still pass") for the pairwise-table refactor specifically, across Wilcoxon/Romano-Wolf, Newcombe/binary, and explicit-bootstrap scenarios.
- Full targeted regression suite passes throughout: `test_compare.py`, `test_alignment.py`, `test_design.py`, `test_p_values.py`, `test_ci_forest_plot.py`, `test_bayes_binary_routing.py`, `test_compound_ppi_fwer.py` — 319 tests, all green, run again after every major edit.
- 26 pre-existing failures in `test_pareto.py`/`test_quick_primitives.py` (confirmed via git-stash comparison to predate all of this session's work — a `secondary=` vs `secondary_metric=` naming drift) remain exactly 26, unchanged by any of this — not something I introduced, not something I fixed.

## Deliberate scope limits (unchanged from earlier discussion)

- Multi-run/seeded data raises a clear error under `design="unpaired"` rather than attempting nested-run resampling.
- The synthetic-item-column fallback needs a one-line workaround (`df["item"] = range(len(df))`) before `load_from()`, since that function unconditionally requires an item column and I didn't want to touch code shared by every existing call.
- `baseline=`, `pairwise_test=`, `show_rank_probabilities=` still have no effect under `design="unpaired"`. `p_values=`/`omnibus=` **are** honored now (a separate decision you made mid-session) — unpaired-specific defaults of `True`, not `compare()`'s own `False`.

## Where things stand

- 3 commits already on this branch (Phase 0, config table, core engine) from earlier in the session.
- Everything else — `api.py`, `core/summary.py` (substantial refactor), `core/summary_unpaired.py`, `core/unpaired.py`, `core/pareto.py`, `config.py`, `tests/__init__.py`, plus `tests/test_unpaired.py` and the battle-test script — is uncommitted, awaiting your review.
- **Nothing has been committed or merged.**

## Suggested next steps

1. Skim this summary; the pairwise-table format change (bars + Δθ framing) is the one thing worth a deliberate look, not just a rubber-stamp, since it's a visible behavior change to what you saw earlier.
2. If it looks good, say the word and I'll commit this as a logically-separated set of commits.
3. The `test_pareto.py`/`test_quick_primitives.py` pre-existing failures are still there if you want a follow-up investigation — happy to spin that off separately.
