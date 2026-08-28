"""One-off investigation (2026-08-11): sanity-check counterpart to
investigate_likert_family_wise_smalln.py -- does CONTINUOUS data show the
same small-N family-wise coverage collapse likert does, under the same
logit_t/nig/smooth_bootstrap comparison?

Motivation: simulations/harness/cases/compare_e2e.py found likert's
family-wise coverage collapsing badly at small n_items (65% vs 95% nominal
at n=15, worsening to ~18% at k=10 for the worst shape), but explicitly
NOT present in continuous data at the same n_items using the SAME
underlying CI method (logit_t) -- see that harness case's own investigation
notes. investigate_likert_family_wise_smalln.py then confirmed (after
fixing a self-referential-coverage bug) that this is real and severe, and
that nig fixes it completely at a real score cost concentrated exactly
where the failure is worst.

This script re-runs the identical methodology on continuous data, as a
direct check that continuous truly doesn't share the failure (ruling out
"logit_t is broken at small N in general" in favor of "likert's
discreteness/quantization is the specific driver") -- not just trusting
the earlier compare_e2e finding, but re-verifying it with the same
Sidak-widened, swappable-ci_func, bug-fixed methodology used for likert.

Not part of the harness / --official-tests: standalone Monte Carlo script.
Run directly:

    .venv/bin/python simulations/investigate_continuous_family_wise_smalln.py
"""

from __future__ import annotations

import time
import warnings
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evalstats.core.paired import _sidak_simultaneous_cis
from evalstats.core.resampling import logit_t_ci_1d, nig_ci_1d, smooth_bootstrap_means_1d
from evalstats.core.stats_utils import interval_score, rescaled_ci
from simulations.harness.scenarios.synthetic import (
    CONTINUOUS_SHAPES, _jb_effect_magnitude, _tier_shapes, sample_group_truth,
)

ALPHA = 0.05
N_VALUES = [10, 15, 20, 30, 60]
K_VALUES = [3, 5, 10]
N_REPS = 300
N_BOOTSTRAP = 1000  # per-pair resample count for smooth_bootstrap's ci_func
SEED = 20260811
TRUE_MEAN_MC_N = 200_000  # matches compare_e2e's _TRUE_MEAN_MC_N convention

# Standard-tier only -- matches compare_e2e's own shape catalog exactly.
CONTINUOUS_SHAPES = _tier_shapes(CONTINUOUS_SHAPES, "standard")
CONTINUOUS_SCALE = (0.0, 1.0)
EFFECT_FRAC = 0.15  # matches compare_e2e's DEFAULT_EFFECT_FRAC

METHODS = ["logit_t", "nig", "smooth_bootstrap"]


def build_ci_func(method: str, rng: np.random.Generator):
    lo, hi = CONTINUOUS_SCALE
    span = hi - lo
    diff_lo, diff_hi = -span, span
    if method == "logit_t":
        return lambda diffs, alpha: rescaled_ci(logit_t_ci_1d, diffs, alpha, diff_lo, diff_hi)
    if method == "nig":
        return lambda diffs, alpha: rescaled_ci(nig_ci_1d, diffs, alpha, diff_lo, diff_hi)
    if method == "smooth_bootstrap":
        def _ci(diffs, alpha):
            boot = smooth_bootstrap_means_1d(diffs, N_BOOTSTRAP, rng, statistic="mean")
            return (float(np.percentile(boot, 100 * alpha / 2)),
                    float(np.percentile(boot, 100 * (1 - alpha / 2))))
        return _ci
    raise ValueError(method)


def family_wise_cis(diffs_by_arm: np.ndarray, labels: list[str], ci_func, alpha: float):
    """diffs_by_arm: (k, n) truth array, one row per arm, paired by column
    (item) index. Returns {(a,b): (lo,hi)} for every pair, Sidak-widened."""
    pairs = list(combinations(labels, 2))
    idx = {lbl: i for i, lbl in enumerate(labels)}
    results = {
        (a, b): SimpleNamespace(per_input_diffs=diffs_by_arm[idx[a]] - diffs_by_arm[idx[b]])
        for a, b in pairs
    }
    return _sidak_simultaneous_cis(results=results, pairs=pairs, ci=1.0 - alpha, ci_func=ci_func)


def run() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    total_cells = len(CONTINUOUS_SHAPES) * len(K_VALUES) * len(N_VALUES)
    cell_i = 0
    for shape in CONTINUOUS_SHAPES:
        for k in K_VALUES:
            labels = [f"M{i}" for i in range(k)]
            effect_step = _jb_effect_magnitude("continuous", EFFECT_FRAC)
            effects = np.arange(k, dtype=float) * effect_step
            # TRUE population means -- large separate MC draw, NOT the small
            # test sample's own mean (see investigate_likert_..._smalln.py's
            # docstring for why that was a real bug in the first version).
            true_means = sample_group_truth(
                shape, TRUE_MEAN_MC_N, 1, k, 1.0, rng, effects=effects,
            )[:, :, 0].mean(axis=1)
            for n_items in N_VALUES:
                cell_i += 1
                per_method = {m: dict(covered=0, total=0, width_sum=0.0, score_sum=0.0, n=0) for m in METHODS}
                ci_funcs = {m: build_ci_func(m, rng) for m in METHODS}
                for _rep in range(N_REPS):
                    truth = sample_group_truth(shape, n_items, 1, k, 1.0, rng, effects=effects)[:, :, 0]
                    for method in METHODS:
                        cis = family_wise_cis(truth, labels, ci_funcs[method], ALPHA)
                        all_covered = True
                        for (a, b), (lo, hi) in cis.items():
                            true_diff = true_means[labels.index(a)] - true_means[labels.index(b)]
                            covered = lo <= true_diff <= hi
                            if not covered:
                                all_covered = False
                            per_method[method]["width_sum"] += hi - lo
                            per_method[method]["score_sum"] += interval_score(lo, hi, true_diff, ALPHA)
                            per_method[method]["n"] += 1
                        per_method[method]["total"] += 1
                        if all_covered:
                            per_method[method]["covered"] += 1
                for method in METHODS:
                    d = per_method[method]
                    rows.append(dict(
                        shape=shape.label, k=k, n_items=n_items, method=method,
                        family_coverage=d["covered"] / d["total"],
                        mean_width=d["width_sum"] / d["n"],
                        mean_score=d["score_sum"] / d["n"],
                    ))
                elapsed = time.time() - t0
                print(f"\r  cell {cell_i}/{total_cells}  ({elapsed:.0f}s elapsed)", end="", flush=True)
    print()
    return pd.DataFrame(rows)


METHOD_COLORS = {"logit_t": "#a6761d", "nig": "#888888", "smooth_bootstrap": "#9467bd"}


def save_by_k_violin_plot(df: pd.DataFrame, out_dir: str, run_stem: str) -> list[str]:
    """Same structure as investigate_likert_family_wise_smalln.py's plot:
    one ROW per k, one violin per method at each n, each dot one shape."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - ALPHA
    ks = sorted(df["k"].unique())
    ns = sorted(df["n_items"].unique())
    n_order = [str(n) for n in ns]
    df = df.copy()
    df["n_label"] = df["n_items"].astype(str)

    out_paths: list[str] = []
    for metric, ylabel, fname_suffix in [
        ("family_coverage", "Family-wise coverage per shape\n(ALL C(k,2) pairs simultaneously covered)", "by_k_violin_coverage"),
        ("mean_score", "Mean interval score per shape\n(per pair, lower=better)", "by_k_violin_score"),
    ]:
        fig, axes = plt.subplots(len(ks), 1, figsize=(1.4 * len(ns) + 3.0, 4.2 * len(ks)), squeeze=False)
        for row_idx, k in enumerate(ks):
            ax = axes[row_idx][0]
            k_df = df[df["k"] == k]
            sns.violinplot(
                data=k_df, x="n_label", y=metric, order=n_order, hue="method", hue_order=METHODS,
                palette=METHOD_COLORS, cut=0, inner="quartile", linewidth=0.8, dodge=True, alpha=0.35, ax=ax,
            )
            sns.stripplot(
                data=k_df, x="n_label", y=metric, order=n_order, hue="method", hue_order=METHODS,
                palette=METHOD_COLORS, size=4, alpha=0.6, dodge=True, jitter=0.15,
                linewidth=0.4, edgecolor="white", legend=False, ax=ax,
            )
            if metric == "family_coverage":
                ax.axhline(target, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)

            handles, _ = ax.get_legend_handles_labels()
            ax.legend(
                handles=handles[:len(METHODS)], title="Method", fontsize=8, title_fontsize=9,
                loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
            )
            ax.set_xlabel("n_items (per arm)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"k={k}  ({k * (k - 1) // 2} pairs)")

        fig.suptitle(
            f"Continuous family-wise {'coverage' if metric == 'family_coverage' else 'interval score'} vs. n_items, by arm count (k)\n"
            f"{run_stem} | reps={N_REPS} | alpha={ALPHA}",
            fontsize=12,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = str(Path(out_dir) / f"{run_stem}_{fname_suffix}.png")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


if __name__ == "__main__":
    df = run()
    out_path = "simulations/out/investigate_continuous_family_wise_smalln_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}\n")

    run_stem = f"investigate_continuous_family_wise_smalln_reps{N_REPS}_{time.strftime('%Y%m%d_%H%M%S')}"
    plot_paths = save_by_k_violin_plot(df, "simulations/out/plots", run_stem)
    for p in plot_paths:
        print(f"Saved plot: {p}")

    print("=" * 100)
    print("Family-wise coverage, mean width, mean interval score -- pooled across all continuous shapes")
    print(f"(alpha={ALPHA}, nominal target={1 - ALPHA:.0%}, reps={N_REPS})")
    print("=" * 100)
    for k in K_VALUES:
        print(f"\n--- k={k} ---")
        g = df[df.k == k].groupby(["n_items", "method"]).agg(
            family_coverage=("family_coverage", "mean"),
            mean_width=("mean_width", "mean"),
            mean_score=("mean_score", "mean"),
        ).reset_index()
        piv_cov = g.pivot(index="n_items", columns="method", values="family_coverage")[METHODS]
        piv_width = g.pivot(index="n_items", columns="method", values="mean_width")[METHODS]
        piv_score = g.pivot(index="n_items", columns="method", values="mean_score")[METHODS]
        print("Coverage:")
        print(piv_cov.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Mean width:")
        print(piv_width.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Mean score (lower=better):")
        print(piv_score.to_string(float_format=lambda x: f"{x:.3f}"))
