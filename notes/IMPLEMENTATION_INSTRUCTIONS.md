# Implementation instructions — between-subjects support for compare()

Written 2026-08-15. This file exists because this is a long, multi-hour autonomous
task and context may get compacted along the way — if you're picking this up
after a compaction, read this file fully before doing anything else, then check
`TaskList`/git log/git status in this worktree to see how far things got.

## The task, verbatim from the user (paraphrased for clarity, not shortened in substance)

> Proceed with implementation of PLAN_between_subjects_extension.md. Do this in a
> fresh worktree so the main branch stays clear, since this is potentially a huge
> change. Implement everything according to plan, spinning off subagents as needed.
>
> Then, once implemented, double-check quality and coverage of the codebase --
> look for overlooked integration problems.
>
> Set up a small simulation setup, feeding a wide variety of different data to
> compare() to battle-test the between-subjects additions:
>   - with AND without LLM judge/alignment (PPI) data
>   - various group sizes, INCLUDING UNBALANCED groups
>   - various numbers of conditions (k=2, k=3, k=5+, etc.)
>   - secondary metrics (the Pareto-front feature) -- does/should this interact
>     with unpaired data at all? verify graceful behavior either way
>   - multi-run data and sensitivity/per-run noise checks
> Hone the API and fix any issues found.
>
> Once that's done, review a SECOND time from the perspective of a user of the
> API: do the calls make sense? Does it work in a variety of situations? Where
> does it fall short? Revise again based on this.
>
> Once satisfied, present an executive summary report for when the user gets
> back from a long break.
>
> CRITICAL CONSTRAINT: the existing 'paired' path must remain COMPLETELY
> untouched in behavior -- virtually identical to before. Be surgical and exact
> with additions/edits. Do not bloat complexity.
>
> The full pytest suite is extremely slow -- run it only sparingly (e.g. once
> near the very end, or not at all if targeted tests give enough confidence).
> Prefer targeted test runs scoped to just the files/modules being changed.
>
> Save these instructions somewhere (this file) since the task is long and
> context may not survive intact. The user is going offline for several hours
> and wants this driven to completion autonomously -- don't stop to ask
> clarifying questions unless genuinely blocked on something only the user can
> decide (in which case, make the most reasonable call, note it clearly in the
> final report, and keep going -- don't stall the whole task on it).

## Where to find the actual technical plan

`PLAN_between_subjects_extension.md` in this worktree's root (copied in from the
main checkout, since it's gitignored and doesn't travel with git branching). It
is the authoritative technical spec: architecture (§3), phased plan (§4),
decided-vs-open items (§5), non-goals (§6). All the binary-vs-other-types
routing, Bonferroni-CI/Holm-p-value correction, `design=` opt-in, etc. decisions
were made through extensive back-and-forth with the user and are FINAL -- do not
re-litigate them, just implement what's written there. The two still-open items
in §5 (grade-as-continuous routing, binary t-interval boundary-behavior check)
are explicitly flagged as needing verification during implementation, not
further design discussion.

## Process checklist (update as you go -- treat this file as the source of truth
## for "what phase am I in" if TaskList state is ever ambiguous after a compaction)

- [ ] Phase 0: relocate `_detect_paired` to `evalstats/core/design.py`
- [ ] Phase 1 core: `AUTO_UNPAIRED_METHOD_TABLE` in config.py
- [ ] Phase 1 core: `evalstats/core/unpaired.py` -- dispatcher, both test families,
      Bonferroni CI + Holm p-value correction, PPI and non-PPI variants,
      synthetic-item-column fallback
- [ ] Phase 1 core: `GroupComparisonResult` + `GroupDiffResult` dataclasses,
      `to_dict()`/`to_frame()`
- [ ] Phase 1 core: wire `compare()` with `design="auto"|"paired"|"unpaired"`,
      verify paired behavior is 100% unchanged when design="auto" and data pivots
      cleanly
- [ ] Phase 1 core: `.summary()` printer (reusing `_gradient_interval_line`) +
      inline alignment disclosure
- [ ] Phase 1 core: calibration check for Bonferroni CIs + binary boundary-
      behavior check (the specific open item from PLAN §5)
- [ ] Phase 1 core: `tests/test_unpaired.py`
- [ ] Rewrite the paper's App Store/FlipFlop scenario against the new API
      (this is example/demo code, not the paper text itself -- if a runnable
      example script exists from earlier in the session, update it; don't
      touch LaTeX)
- [ ] Integration review pass (look for overlooked interactions: secondary_metric/
      Pareto, multi-run/seeded data, factorial, existing paired tests still green)
- [ ] Battle-test simulation harness (see task brief above) -- build it, run it,
      fix what it finds
- [ ] User-perspective API review pass -- revise based on findings
- [ ] Confirm paired path untouched: targeted test run comparing before/after
      behavior, not just "tests pass"
- [ ] Write executive summary report for the user

## Constraints to keep re-reading, not just once

1. **Paired path behavior must be pixel-identical to before this change**, for
   any call that doesn't explicitly request the new path. This is the single
   most important constraint. When in doubt, check by diffing actual output
   (e.g. `.summary()` text) on a real example before/after, not just "no
   exceptions raised."
2. **Surgical, not bloated.** Don't add abstractions, config flags, or
   generality beyond what the plan calls for. Reuse existing machinery
   (`evalstats.tests.*`, `evalstats.quick.summarize`, `_gradient_interval_line`,
   `correct_pvalues`) rather than reimplementing.
3. **No full pytest suite runs except sparingly.** Use targeted invocations,
   e.g. `pytest tests/test_unpaired.py`, `pytest tests/test_paired.py -k
   something`, or direct Python smoke scripts. If you do run the full suite,
   treat it as a rare, deliberate checkpoint, not a routine step.
4. **This is a worktree** (`worktree-between-subjects-support` branch, at
   `.claude/worktrees/between-subjects-support`). Commit here as work lands
   (matching this session's established pattern elsewhere: commit after each
   verified chunk, don't batch everything into one giant commit). The user
   will merge into their main branch themselves when ready -- do not merge or
   push anywhere.
5. Don't ask the user clarifying questions during this run -- they're offline.
   Make the most reasonable call on anything ambiguous, log the decision (in
   commit messages and/or the final report), and keep moving.
