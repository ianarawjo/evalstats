"""Standalone driver for the Type-I error check on the four PPI rank tests
under a null with unequal HUMAN-side truth spread across groups/conditions
-- see simulations/out/PLAN_hetero_null_check.md for the full brief.

This is a one-off diagnostic script, NOT part of the harness's official
sweep -- do not add these scenarios to
scenarios.synthetic.build_judge_bias_sources.

Usage (from the repo root, inside .venv):
    python -m simulations.hetero_null_check --n-reps 1000 --n-boot 500 --n-workers 8
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from simulations.harness.scenarios import JudgeBiasSource
from simulations.harness.scenarios.synthetic import generate_judge_bias_cell
from simulations.harness.cases.pvalues import run_ppi_simulation

ACTIVE_TESTS = ["mwu", "kruskal", "wilcoxon", "friedman", "ttest_welch", "paired_t"]
ALPHA = 0.05

# Copied from scenarios.synthetic.build_judge_bias_sources' own baseline `B`
# so every factor other than eval_type/judge/n/spread matches the official
# sweep exactly.
BASELINE = dict(
    n2=None, n3=None,
    label_frac=0.20, llm_noise=0.20, llm_noise2=None, llm_noise3=None,
    bias_const=0.40,
    bias_extra_a=0.0, bias_extra_b=0.0, bias_extra_c=0.0, bias_extra_d=0.0,
    slope_a=1.0, slope_b=1.0, slope_c=1.0, slope_d=1.0,
    label_mnar=False, mnar_strength=1.0, mnar_mode="high",
    repeated_corr=0.0, effect_size=0.0,
)

# Per-eval-type icc and shape overrides. Continuous needs shape_label=
# "cont-uniform" (a symmetric Beta(1,1)) instead of the sweep's default
# "cont-right-skew" -- the gate check (Step 1.3 of the plan) confirmed the
# skewed default does NOT preserve the weak null (P(A>B)+0.5P(A=B)=0.5)
# under a truth_scale_b stretch, while cont-uniform does (|theta-0.5| <
# 0.0001 at s=2/4). Likert's default representative shape ("likert-mid")
# already passes the gate (|theta-0.5| < 0.001) without an override.
EVAL_TYPE_CFG = {
    "continuous": dict(icc=0.20, shape_label="cont-uniform"),
    "likert": dict(icc=0.20, shape_label=None),
}

JUDGES = [
    ("judge=unbiased", dict(bias_type="none")),
    ("judge=diffbias", dict(bias_type="differential", bias_delta=0.30)),
]

NS = [100, 400]
SPREAD_RATIOS = [1.0, 2.0, 4.0]


def build_sources(eval_types: list[str]) -> list[JudgeBiasSource]:
    sources = []
    for eval_type in eval_types:
        cfg = EVAL_TYPE_CFG[eval_type]
        for judge_label, judge_kw in JUDGES:
            for n in NS:
                for s in SPREAD_RATIOS:
                    kw = {**BASELINE, **cfg, **judge_kw, "eval_type": eval_type, "n": n,
                          "truth_scale_b": s, "truth_scale_c": s}
                    if kw["shape_label"] is None:
                        del kw["shape_label"]
                    sources.append(JudgeBiasSource(
                        name=f"hetero_truth.{eval_type}.{judge_label}.n={n}.sB={s}",
                        tag="hetero_truth", **kw,
                    ))
    return sources


# ---------------------------------------------------------------------------
# Step 3: classical-on-all-human / classical-on-labeled-only baselines.
# Cheapest route (per the plan): call generate_judge_bias_cell ourselves in
# a loop and run scipy's textbook tests directly on the dense truth arrays
# (all_human) and on the sparse labeled subset (labeled_only), rather than
# routing through the harness's LLM-judge/PPI machinery at all.
# ---------------------------------------------------------------------------

def _labeled(truth: np.ndarray, lab: np.ndarray) -> np.ndarray:
    return truth[~np.isnan(lab)]


def _classical_reject_counts(sc: JudgeBiasSource, n_reps: int, seed: int) -> dict[str, dict[str, int]]:
    """Returns {test_name: {"allhuman": rejects, "labeled": rejects, "n_failed_allhuman": .., "n_failed_labeled": ..}}."""
    rng = np.random.default_rng(seed)
    tally = {t: {"allhuman": 0, "labeled": 0, "n_failed_allhuman": 0, "n_failed_labeled": 0} for t in ACTIVE_TESTS}

    for _ in range(n_reps):
        cell = generate_judge_bias_cell(sc, rng)

        # mwu / ttest_welch: independent two groups (a2, b2)
        for test, fn in [
            ("mwu", lambda x, y: scipy_stats.mannwhitneyu(x, y, alternative="two-sided").pvalue),
            ("ttest_welch", lambda x, y: scipy_stats.ttest_ind(x, y, equal_var=False).pvalue),
        ]:
            try:
                p = fn(cell.truth_a2, cell.truth_b2)
                tally[test]["allhuman"] += int(p < ALPHA)
            except Exception:
                tally[test]["n_failed_allhuman"] += 1
            try:
                p = fn(_labeled(cell.truth_a2, cell.lab_a2), _labeled(cell.truth_b2, cell.lab_b2))
                tally[test]["labeled"] += int(p < ALPHA)
            except Exception:
                tally[test]["n_failed_labeled"] += 1

        # kruskal: independent three groups (a3, b3, c3)
        try:
            p = scipy_stats.kruskal(cell.truth_a3, cell.truth_b3, cell.truth_c3).pvalue
            tally["kruskal"]["allhuman"] += int(p < ALPHA)
        except Exception:
            tally["kruskal"]["n_failed_allhuman"] += 1
        try:
            p = scipy_stats.kruskal(
                _labeled(cell.truth_a3, cell.lab_a3), _labeled(cell.truth_b3, cell.lab_b3),
                _labeled(cell.truth_c3, cell.lab_c3),
            ).pvalue
            tally["kruskal"]["labeled"] += int(p < ALPHA)
        except Exception:
            tally["kruskal"]["n_failed_labeled"] += 1

        # wilcoxon / paired_t: paired two conditions (x, y)
        for test, fn in [
            ("wilcoxon", lambda x, y: scipy_stats.wilcoxon(x, y).pvalue),
            ("paired_t", lambda x, y: scipy_stats.ttest_rel(x, y).pvalue),
        ]:
            try:
                p = fn(cell.truth_x, cell.truth_y)
                tally[test]["allhuman"] += int(p < ALPHA)
            except Exception:
                tally[test]["n_failed_allhuman"] += 1
            try:
                # lab_x/lab_y share the same labeled index set (_jb_labels_shared).
                p = fn(_labeled(cell.truth_x, cell.lab_x), _labeled(cell.truth_y, cell.lab_y))
                tally[test]["labeled"] += int(p < ALPHA)
            except Exception:
                tally[test]["n_failed_labeled"] += 1

        # friedman: three repeated conditions (A, B, C)
        try:
            p = scipy_stats.friedmanchisquare(cell.truth_A, cell.truth_B, cell.truth_C).pvalue
            tally["friedman"]["allhuman"] += int(p < ALPHA)
        except Exception:
            tally["friedman"]["n_failed_allhuman"] += 1
        try:
            p = scipy_stats.friedmanchisquare(
                _labeled(cell.truth_A, cell.lab_A), _labeled(cell.truth_B, cell.lab_B),
                _labeled(cell.truth_C, cell.lab_C),
            ).pvalue
            tally["friedman"]["labeled"] += int(p < ALPHA)
        except Exception:
            tally["friedman"]["n_failed_labeled"] += 1

    return tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=500, help="Reps per cell (plan default: 1000; halved here per the project's 'speed up diagnostic scripts' guidance for a first exploratory pass).")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--eval-types", nargs="+", default=["continuous", "likert"], choices=["continuous", "likert"])
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sources = build_sources(args.eval_types)
    print(f"Built {len(sources)} scenarios x {len(ACTIVE_TESTS)} tests, n_reps={args.n_reps}, n_boot={args.n_boot}, n_workers={args.n_workers}")

    t0 = time.time()
    ppi_results = run_ppi_simulation(
        sources, active_tests=ACTIVE_TESTS, n_reps=args.n_reps, n_boot=args.n_boot,
        seed=args.seed, n_workers=args.n_workers,
    )
    t1 = time.time()
    print(f"PPI+uncorrected sweep done in {t1 - t0:.1f}s")

    # Index PPI results by (name, test)
    ppi_by_key: dict[tuple[str, str], object] = {(r.name, r.test): r for r in ppi_results}

    # Classical baselines: independent, much cheaper loop.
    classical_by_name: dict[str, dict] = {}
    for i, sc in enumerate(sources):
        classical_by_name[sc.name] = _classical_reject_counts(sc, args.n_reps, seed=args.seed * 100_003 + i)
    t2 = time.time()
    print(f"Classical-baseline loop done in {t2 - t1:.1f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "out" / f"hetero_null_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"

    rows = []
    for sc in sources:
        eval_type = sc.eval_type
        judge = "unbiased" if sc.bias_type == "none" else "diffbias"
        for test in ACTIVE_TESTS:
            r = ppi_by_key.get((sc.name, test))
            cls = classical_by_name[sc.name][test]
            n_eff = args.n_reps - (r.n_failed if r is not None else 0)
            rows.append(dict(
                name=sc.name, test=test, eval_type=eval_type, judge=judge, n=sc.n, spread_ratio=sc.truth_scale_b,
                n_reps=args.n_reps,
                typeI_ppi=(r.corrected_rejects / n_eff) if (r is not None and n_eff > 0) else None,
                typeI_uncorrected=(r.uncorrected_rejects / n_eff) if (r is not None and n_eff > 0) else None,
                n_failed_ppi=r.n_failed if r is not None else None,
                typeI_classical_allhuman=cls["allhuman"] / max(args.n_reps - cls["n_failed_allhuman"], 1),
                typeI_classical_labeled_only=cls["labeled"] / max(args.n_reps - cls["n_failed_labeled"], 1),
                n_failed_classical_allhuman=cls["n_failed_allhuman"],
                n_failed_classical_labeled=cls["n_failed_labeled"],
            ))

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
