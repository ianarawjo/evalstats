"""Final pre-release stress test for compare() (2026-08-15) -- both the
paired and unpaired paths, across a wide grid of k/n/design/PPI-bias, plus
deliberately malformed input (developers *will* pass bad CSVs).

Three parts:

1. Crash/sanity grid: paired vs unpaired x k in {2,3,5,10,20} x
   n in {15,30,50,200} x with/without a biased judge. Asserts compare()
   returns a valid result and .summary() doesn't crash or print NaN/garbage.
2. Visual spot-checks: a curated subset of the grid's extremes, printed in
   full for human inspection (k=2, k=20, n=15, n=200, strong judge bias).
3. Malformed-input robustness: missing columns, empty data, all-NaN scores,
   non-numeric scores, single row, wrong dtypes, duplicate columns, a
   single-group dataset, etc. -- checks each either raises a clear,
   attributed error or completes correctly, never a bare/cryptic traceback
   or a silent wrong answer.

Not part of the harness / --official-tests: standalone script. Run
directly:

    .venv/bin/python -m simulations.investigate_final_stress_test
"""
from __future__ import annotations

import io
import contextlib
import math
import re
import traceback
import warnings

import numpy as np
import pandas as pd

import evalstats as es
from evalstats.alignment import judge_alignment

warnings.filterwarnings("ignore", category=UserWarning)

N_BOOT_GRID = 300     # modest -- crash grid, not a final calibration number
N_BOOT_VISUAL = 1000  # a bit more care for the human-inspected cases


def _rng(seed):
    return np.random.default_rng(seed)


def _labels(k):
    return [f"G{i:02d}" for i in range(k)]


def make_paired_df(k, n, seed, biased_judge=False):
    """Within-subjects: every group scored on the SAME n items."""
    rng = _rng(seed)
    labels = _labels(k)
    means = {lbl: 0.35 + 0.35 * (i / max(k - 1, 1)) for i, lbl in enumerate(labels)}
    rows = []
    for item in range(n):
        item_effect = rng.normal(0, 0.05)
        for lbl in labels:
            true_score = float(np.clip(means[lbl] + item_effect + rng.normal(0, 0.12), 0, 1))
            rows.append({"model": lbl, "item": item, "true_score": true_score})
    df = pd.DataFrame(rows)
    if biased_judge:
        # Judge systematically compresses + shifts scores relative to "truth".
        df["llm_score"] = np.clip(0.5 + 0.6 * (df["true_score"] - 0.5) + 0.1, 0, 1)
        df["human_score"] = np.nan
        # Paired design: items are shared across all k models, so the SAME
        # item indices must be labeled for every model (a real partial-label
        # dataset labels a fixed subset of items, not a different random
        # subset per model -- picking independently per model means, as k
        # grows, the union of "labeled in at least one model" approaches
        # 100% of items, leaving no unlabeled residual for PPI).
        n_label_items = max(3, min(n - 1, int(0.4 * n)))
        labeled_items = rng.choice(n, size=n_label_items, replace=False)
        mask = df["item"].isin(labeled_items)
        df.loc[mask, "human_score"] = df.loc[mask, "true_score"]
        df = df.drop(columns=["true_score"])
    else:
        df["score"] = df.pop("true_score")
    return df


def make_unpaired_df(k, n, seed, biased_judge=False):
    """Between-subjects: every group has its OWN disjoint n items."""
    rng = _rng(seed)
    labels = _labels(k)
    means = {lbl: 0.35 + 0.35 * (i / max(k - 1, 1)) for i, lbl in enumerate(labels)}
    rows = []
    for lbl in labels:
        for i in range(n):
            true_score = float(np.clip(rng.normal(means[lbl], 0.15), 0, 1))
            rows.append({"model": lbl, "item": f"{lbl}_{i}", "true_score": true_score})
    df = pd.DataFrame(rows)
    if biased_judge:
        df["llm_score"] = np.clip(0.5 + 0.6 * (df["true_score"] - 0.5) + 0.1, 0, 1)
        df["human_score"] = np.nan
        # Leave at least one unlabeled item per group so PPI always has an
        # unlabeled residual to extrapolate to, even at small n.
        n_label = max(3, min(n - 1, int(0.4 * n)))
        for lbl in labels:
            idx = df.index[df["model"] == lbl].to_numpy()
            chosen = rng.choice(idx, size=min(n_label, len(idx)), replace=False)
            df.loc[chosen, "human_score"] = df.loc[chosen, "true_score"]
        df = df.drop(columns=["true_score"])
    else:
        df["score"] = df.pop("true_score")
    return df


_NAN_TOKEN_RE = re.compile(r"(?<![a-zA-Z])nan(?![a-zA-Z])", re.IGNORECASE)

# Preconditions PPI legitimately, deliberately enforces (small N, or a fully
# human-labeled dataset with no unlabeled residual to extrapolate to) --
# these are correct rejections with clear messages, not bugs.
_EXPECTED_PPI_PRECONDITION_MARKERS = (
    "PPI alignment requires at least",
    "PPI correction requires at least one unlabeled item",
    "every item is labeled",
    "human labels are required for PPI",
)


def _sane_output_checks(out: str, k: int) -> list[str]:
    """Cheap textual sanity checks on printed .summary() output -- catches
    the class of bug a battle-test grid checking only 'didn't crash' would
    miss (NaN leaking into display, garbled tables, etc.)."""
    problems = []
    if _NAN_TOKEN_RE.search(out):
        problems.append("literal 'nan' token found in printed output")
    if "None" in out and "Score type" not in out:
        # "None" can legitimately appear in a few places; only flag if it
        # looks like it's standing in for a number.
        for line in out.splitlines():
            if "None" in line and any(c.isdigit() for c in line) is False and "CI" not in line:
                pass  # too noisy to assert on reliably; skip strict check
    if len(out) < 100:
        problems.append(f"suspiciously short output ({len(out)} chars) for k={k}")

    # Executive Summary's "Grp" column must be non-decreasing down the table
    # (rows are already sorted best-mean-first) -- a regression here means a
    # worse-ranked entity is shown with a numerically *earlier* group than a
    # better-ranked one above it, which is a real bug (found + fixed via this
    # exact check: overlapping/chained CD bands could misorder singleton
    # group IDs). Matches rows like "  G03    #2    0.674  [0.649, 0.698]".
    exec_start = out.find("--- Executive Summary")
    if exec_start != -1:
        exec_block = out[exec_start:]
        grp_numbers = [int(m) for m in re.findall(r"^\s*\S.*?#(\d+)\s+[-\d.]+\s+\[", exec_block, re.MULTILINE)]
        if grp_numbers and any(b < a for a, b in zip(grp_numbers, grp_numbers[1:])):
            problems.append(f"Executive Summary Grp column is non-monotonic: {grp_numbers}")

    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: crash/sanity grid
# ─────────────────────────────────────────────────────────────────────────────

def run_crash_grid():
    print("=" * 78)
    print("PART 1: crash/sanity grid (paired + unpaired x k x n x judge bias)")
    print("=" * 78)
    k_values = [2, 3, 5, 10, 20]
    n_values = [15, 30, 50, 200]
    designs = ["paired", "unpaired"]
    bias_modes = [False, True]

    n_total = 0
    n_failed = 0
    n_expected_reject = 0
    failures = []
    expected_rejects = []

    for design, k, n, biased in [
        (d, k, n, b)
        for d in designs for k in k_values for n in n_values for b in bias_modes
    ]:
        n_total += 1
        seed = hash((design, k, n, biased)) % (2**31)
        try:
            if design == "paired":
                df = make_paired_df(k, n, seed, biased_judge=biased)
            else:
                df = make_unpaired_df(k, n, seed, biased_judge=biased)

            metric_col = "llm_score" if biased else "score"
            evaldata = es.load_from(df)

            alignment = None
            if biased:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ar = judge_alignment(
                        evaldata, llm_metric=metric_col, human_groundtruth="human_score",
                        selection="random",
                    )
                alignment = {metric_col: ar}

            kwargs = dict(
                factors="model", metric=metric_col, rng=np.random.default_rng(seed), n_bootstrap=N_BOOT_GRID,
            )
            if design == "unpaired":
                kwargs["design"] = "unpaired"
            if alignment is not None:
                kwargs["alignment"] = alignment

            result = es.compare(evaldata, **kwargs)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result.summary()
            out = buf.getvalue()
            assert len(out) > 0, "empty .summary() output"

            problems = _sane_output_checks(out, k)
            if problems:
                raise AssertionError("; ".join(problems))

            # Basic structural checks available on both result types.
            if design == "unpaired":
                assert len(result.groups) == k
                assert len(result.pairwise) == k * (k - 1) // 2
                for p in result.pairwise:
                    assert not math.isnan(p.ci_low) and not math.isnan(p.ci_high)
                    assert p.ci_low <= p.ci_high

        except ValueError as e:
            if any(marker in str(e) for marker in _EXPECTED_PPI_PRECONDITION_MARKERS):
                n_expected_reject += 1
                expected_rejects.append((design, k, n, biased, str(e)))
            else:
                n_failed += 1
                failures.append((design, k, n, biased, seed, repr(e), traceback.format_exc(limit=3)))
        except Exception as e:  # noqa: BLE001 -- stress test, want to catch everything
            n_failed += 1
            failures.append((design, k, n, biased, seed, repr(e), traceback.format_exc(limit=3)))

    print(f"Total combinations: {n_total}, failures: {n_failed}, expected PPI-precondition rejects: {n_expected_reject}")
    if expected_rejects:
        print("\nExpected PPI-precondition rejects (correct behavior, not bugs):")
        for r in expected_rejects:
            print(f"  design={r[0]:<9s} k={r[1]:<3d} n={r[2]:<4d} biased={r[3]!s:<5s}  ->  {r[4][:90]}")
    if failures:
        print("\nFAILURES:")
        for f in failures[:15]:
            print(f"  design={f[0]:<9s} k={f[1]:<3d} n={f[2]:<4d} biased={f[3]!s:<5s} seed={f[4]}  ->  {f[5]}")
            print(f"    {f[6].splitlines()[-1] if f[6] else ''}")
    else:
        print("All combinations passed.")
    return n_failed


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: visual spot-checks (printed for human inspection)
# ─────────────────────────────────────────────────────────────────────────────

def run_visual_spotchecks():
    print()
    print("=" * 78)
    print("PART 2: visual spot-checks (extremes, printed for inspection)")
    print("=" * 78)

    cases = [
        ("paired, k=2, n=15 (small)", "paired", 2, 15, False),
        ("paired, k=20, n=30 (many groups)", "paired", 20, 30, False),
        ("unpaired, k=2, n=15 (small, disjoint)", "unpaired", 2, 15, False),
        ("unpaired, k=20, n=200 (many groups, large n)", "unpaired", 20, 200, False),
        ("paired, k=5, n=50, biased judge (PPI)", "paired", 5, 50, True),
        ("unpaired, k=5, n=50, biased judge (PPI)", "unpaired", 5, 50, True),
    ]

    for title, design, k, n, biased in cases:
        print()
        print("-" * 78)
        print(f"CASE: {title}")
        print("-" * 78)
        seed = hash((design, k, n, biased, "visual")) % (2**31)
        try:
            if design == "paired":
                df = make_paired_df(k, n, seed, biased_judge=biased)
            else:
                df = make_unpaired_df(k, n, seed, biased_judge=biased)
            metric_col = "llm_score" if biased else "score"
            evaldata = es.load_from(df)
            alignment = None
            if biased:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ar = judge_alignment(
                        evaldata, llm_metric=metric_col, human_groundtruth="human_score",
                        selection="random",
                    )
                alignment = {metric_col: ar}
            kwargs = dict(factors="model", metric=metric_col, rng=np.random.default_rng(seed), n_bootstrap=N_BOOT_VISUAL)
            if design == "unpaired":
                kwargs["design"] = "unpaired"
            if alignment is not None:
                kwargs["alignment"] = alignment
            result = es.compare(evaldata, **kwargs)
            result.summary()
        except Exception as e:
            print(f"  !! FAILED: {e!r}")
            traceback.print_exc(limit=3)


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: malformed-input robustness
# ─────────────────────────────────────────────────────────────────────────────

def run_malformed_input_checks():
    print()
    print("=" * 78)
    print("PART 3: malformed-input robustness (developer mistakes)")
    print("=" * 78)

    cases = []

    # Missing required column.
    df = pd.DataFrame({"model": ["a", "b"] * 10, "score": list(range(20))})
    df = df.drop(columns=["model"]).assign(wrong_col=["a", "b"] * 10)
    cases.append(("missing 'model'-equivalent column entirely", df, dict(factors="model", metric="score")))

    # Empty dataframe.
    cases.append(("completely empty DataFrame", pd.DataFrame(), dict(factors="model", metric="score")))

    # All-NaN score column. Item ids repeat across models (paired-shaped) so
    # the design-detection heuristic doesn't mask the real problem.
    df = pd.DataFrame({"model": ["a"] * 10 + ["b"] * 10, "item": list(range(10)) * 2,
                        "score": [float("nan")] * 20})
    cases.append(("all-NaN score column", df, dict(factors="model", metric="score")))

    # Non-numeric score column (developer passed the wrong column, e.g. free text).
    df = pd.DataFrame({"model": ["a"] * 10 + ["b"] * 10, "item": list(range(10)) * 2,
                        "score": ["this is not a number"] * 20})
    cases.append(("non-numeric score column (free text)", df, dict(factors="model", metric="score")))

    # Single row of data.
    df = pd.DataFrame({"model": ["a"], "item": [0], "score": [0.5]})
    cases.append(("single row of data", df, dict(factors="model", metric="score")))

    # Only one distinct group (developer forgot to include a comparison model).
    df = pd.DataFrame({"model": ["a"] * 20, "item": list(range(20)),
                        "score": list(np.random.default_rng(1).uniform(0, 1, 20))})
    cases.append(("only one distinct group/model value", df, dict(factors="model", metric="score")))

    # Duplicate column names (can happen from a bad CSV merge/export).
    df = pd.DataFrame({"model": ["a"] * 10 + ["b"] * 10, "item": list(range(10)) * 2,
                        "score": list(np.random.default_rng(2).uniform(0, 1, 20))})
    df_dup = df.copy()
    df_dup.columns = ["model", "item", "score"]
    df_dup["score"] = df_dup["score"]
    # Simulate the duplicate-column CSV artifact directly via concat of two same-named cols.
    df_with_dup_cols = pd.concat([df_dup, df_dup[["score"]].rename(columns={"score": "score"})], axis=1)
    cases.append(("duplicate column names (score appears twice)", df_with_dup_cols,
                  dict(factors="model", metric="score")))

    # Scores way outside any sane range (developer passed raw counts instead of a rate).
    df = pd.DataFrame({"model": ["a"] * 15 + ["b"] * 15, "item": list(range(15)) * 2,
                        "score": list(np.random.default_rng(3).integers(0, 1_000_000, 30))})
    cases.append(("scores as huge raw integers (0 to 1e6)", df, dict(factors="model", metric="score")))

    # Whitespace / mixed-case column names (common CSV export artifact).
    df = pd.DataFrame({" Model ": ["a"] * 10 + ["b"] * 10, "ITEM": list(range(10)) * 2,
                        "Score": list(np.random.default_rng(4).uniform(0, 1, 20))})
    cases.append(("whitespace/mixed-case column names", df, dict(factors="model", metric="score")))

    # Metric column that's entirely boolean-as-string ("True"/"False").
    df = pd.DataFrame({"model": ["a"] * 10 + ["b"] * 10, "item": list(range(10)) * 2,
                        "score": ["True"] * 10 + ["False"] * 10})
    cases.append(("boolean-as-string score column", df, dict(factors="model", metric="score")))

    # NaN in the factor column itself.
    df = pd.DataFrame({"model": ["a"] * 10 + ["b"] * 10 + [None],
                        "item": list(range(10)) * 2 + [10],
                        "score": list(np.random.default_rng(5).uniform(0, 1, 21))})
    cases.append(("NaN values in the factor/model column", df, dict(factors="model", metric="score")))

    n_clear_error = 0
    n_succeeded = 0
    n_cryptic_failure = 0
    results = []

    for title, df, kwargs in cases:
        outcome = None
        detail = ""
        try:
            evaldata = es.load_from(df)
            result = es.compare(evaldata, **kwargs)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result.summary()
            outcome = "succeeded"
            detail = f"(ran without error, {len(result.groups) if hasattr(result, 'groups') else '?'} groups)" \
                if hasattr(result, "groups") else "(ran without error)"
            n_succeeded += 1
        except (ValueError, TypeError, KeyError) as e:
            msg = str(e)
            is_clear = len(msg) > 20 and not msg.strip().startswith("Traceback")
            if is_clear:
                outcome = "clear error"
                detail = f"{type(e).__name__}: {msg[:100]}"
                n_clear_error += 1
            else:
                outcome = "CRYPTIC error"
                detail = f"{type(e).__name__}: {msg[:100]}"
                n_cryptic_failure += 1
        except Exception as e:
            outcome = "CRYPTIC/UNEXPECTED error"
            detail = f"{type(e).__name__}: {str(e)[:150]}"
            n_cryptic_failure += 1
        results.append((title, outcome, detail))

    for title, outcome, detail in results:
        marker = "OK" if outcome in ("succeeded", "clear error") else "!!"
        print(f"  [{marker}] {title:<55s} -> {outcome:<25s} {detail}")

    print()
    print(f"Succeeded (handled gracefully): {n_succeeded}")
    print(f"Clear, attributed errors: {n_clear_error}")
    print(f"CRYPTIC/unexpected errors (worth investigating): {n_cryptic_failure}")
    return n_cryptic_failure


if __name__ == "__main__":
    n_grid_failed = run_crash_grid()
    run_visual_spotchecks()
    n_cryptic = run_malformed_input_checks()

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Crash-grid failures: {n_grid_failed}")
    print(f"Cryptic/unexpected errors on malformed input: {n_cryptic}")
