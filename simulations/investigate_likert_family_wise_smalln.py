"""One-off investigation (2026-08-11): does swapping the per-pair CI method
under Sidak family-wise widening fix likert's small-N family-wise coverage
collapse found by simulations/harness/cases/compare_e2e.py?

Real finding that motivated this (compare_e2e, reps=50, full standard-tier
likert shape catalog): family-wise (k>2, ALL C(k,2) pairs simultaneously
covered) coverage for likert data at n_items=15 was 65.0% against a 95%
nominal target, worsening sharply with arm count (down to 18% at k=10 for
the worst shape, likert-bimodal). NOT present in continuous data using the
same underlying CI method (logit_t) or in binary data (Tango) -- see that
harness case's own investigation notes. A direct single-pair check
(`ci_paired --eval-types likert --methods logit_t nig smooth_bootstrap
--sizes 10 15 20 30 60 --reps 500`) shows the per-pair MinCov (worst single
scenario) is only 83-92% at n=15, not catastrophic -- so the family-wise
"ALL k*(k-1)/2 pairs must hold" requirement is amplifying an already-shaky
per-pair CI, not itself introducing a new bug. This script isolates whether
a per-pair method with a better worst-case (nig) or the current default
(logit_t) or a resampling alternative (smooth_bootstrap) actually fixes the
FAMILY-WISE number once Sidak widening (the auto-resolved method for
n_items<30 numeric data, confirmed via evalstats.config.
resolve_auto_simultaneous_ci_method) is layered on top -- and at what width/
score cost, since coverage alone isn't the goal (nominal-or-slightly-above,
not maximally conservative).

Reuses compare_e2e's own data generation (sample_group_truth + the same
population-SD-standardized effect size convention) so this reproduces the
exact regime the collapse was found in, rather than a different synthetic
setup that might not reproduce it.

Not part of the harness / --official-tests: standalone Monte Carlo script.
Run directly:

    .venv/bin/python simulations/investigate_likert_family_wise_smalln.py
"""

from __future__ import annotations

import time
import warnings
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evalstats.core.paired import (
    PairedDiffResult, _joint_bootstrap_scaled_simultaneous_cis, _sidak_simultaneous_cis,
)
from evalstats.core.resampling import logit_t_ci_1d, nig_ci_1d, smooth_bootstrap_means_1d
from evalstats.core.stats_utils import interval_score, rescaled_ci
from simulations.harness.scenarios.synthetic import (
    LIKERT_SHAPES, _jb_effect_magnitude, _tier_shapes, sample_group_truth,
)

ALPHA = 0.05
N_VALUES = [10, 15, 20, 30, 60]
K_VALUES = [3, 5, 10]
N_REPS = 300
N_BOOTSTRAP = 1000  # per-pair resample count for smooth_bootstrap's ci_func, and
                     # the joint-bootstrap widening's own resample count for logit_t_boot
SEED = 20260811
TRUE_MEAN_MC_N = 200_000  # matches compare_e2e's _TRUE_MEAN_MC_N convention

# Standard-tier only -- matches compare_e2e's own shape catalog exactly, so
# this reproduces the exact regime the collapse was found in (LIKERT_SHAPES
# alone also includes "expanded"-tier shapes compare_e2e never tested).
LIKERT_SHAPES = _tier_shapes(LIKERT_SHAPES, "standard")
LIKERT_SCALE = (1.0, 5.0)
EFFECT_FRAC = 0.15  # matches compare_e2e's DEFAULT_EFFECT_FRAC

# logit_t_boot (joint-bootstrap widening instead of Sidak, same logit_t
# per-pair CI) was tested and RULED OUT: coverage tracked plain logit_t
# almost exactly at every k/n (confirmed directly -- see this script's git
# history / the session that built it). Sidak's independence assumption
# isn't the culprit; the per-pair CI construction itself is. Replaced here
# with two DITHERING variants, which target the actual diagnosed mechanism
# directly: likert's rounding to an integer scale erases real underlying
# variability (a continuous latent value near a rounding boundary could
# have rounded either way), so at small N the sample of rounded diffs is
# often literally constant or near-constant (confirmed: smooth_bootstrap's
# own fallback warning fires on "sample std=0" repeatedly in this exact
# regime) -- both logit_t's normal-approximation variance estimate and
# smooth_bootstrap's KDE step then badly underestimate the true uncertainty.
# Dithering (adding U(-0.5, +0.5) jitter to each item's rounded value before
# differencing, then clipping back to the scale) is the standard technique
# for recovering a plausible pre-rounding continuous approximation -- zero-
# mean and symmetric, so it doesn't bias the estimate, but it un-collapses
# the degenerate sample distribution that's breaking both methods.
METHODS = ["logit_t", "nig", "smooth_bootstrap", "logit_t_dither", "smooth_bootstrap_dither"]
WIDENING = {m: "sidak" for m in METHODS}
DITHER_METHODS = {"logit_t_dither", "smooth_bootstrap_dither"}


def dither(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lo, hi = LIKERT_SCALE
    return np.clip(truth + rng.uniform(-0.5, 0.5, size=truth.shape), lo, hi)


def build_ci_func(method: str, rng: np.random.Generator):
    lo, hi = LIKERT_SCALE
    span = hi - lo
    diff_lo, diff_hi = -span, span
    if method in ("logit_t", "logit_t_dither"):
        return lambda diffs, alpha: rescaled_ci(logit_t_ci_1d, diffs, alpha, diff_lo, diff_hi)
    if method == "nig":
        return lambda diffs, alpha: rescaled_ci(nig_ci_1d, diffs, alpha, diff_lo, diff_hi)
    if method in ("smooth_bootstrap", "smooth_bootstrap_dither"):
        def _ci(diffs, alpha):
            boot = smooth_bootstrap_means_1d(diffs, N_BOOTSTRAP, rng, statistic="mean")
            return (float(np.percentile(boot, 100 * alpha / 2)),
                    float(np.percentile(boot, 100 * (1 - alpha / 2))))
        return _ci
    raise ValueError(method)


def family_wise_cis(
    diffs_by_arm: np.ndarray, labels: list[str], ci_func, alpha: float,
    widening: str, rng: np.random.Generator,
):
    """diffs_by_arm: (k, n) truth array, one row per arm, paired by column
    (item) index. Returns {(a,b): (lo,hi)} for every pair, widened by either
    Sidak (assumes independent pairs) or joint bootstrap (models the real
    cross-pair correlation from shared items)."""
    pairs = list(combinations(labels, 2))
    idx = {lbl: i for i, lbl in enumerate(labels)}
    if widening == "sidak":
        results = {
            (a, b): SimpleNamespace(per_input_diffs=diffs_by_arm[idx[a]] - diffs_by_arm[idx[b]])
            for a, b in pairs
        }
        return _sidak_simultaneous_cis(results=results, pairs=pairs, ci=1.0 - alpha, ci_func=ci_func)
    if widening == "boot":
        results = {}
        for a, b in pairs:
            diffs = diffs_by_arm[idx[a]] - diffs_by_arm[idx[b]]
            results[(a, b)] = PairedDiffResult(
                template_a=a, template_b=b,
                point_diff=float(diffs.mean()), std_diff=float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                ci_low=float("nan"), ci_high=float("nan"),
                p_value=float("nan"), test_method="logit_t",
                n_inputs=len(diffs), per_input_diffs=diffs,
            )
        return _joint_bootstrap_scaled_simultaneous_cis(
            scores=diffs_by_arm, results=results, pairs=pairs, labels=labels,
            ci=1.0 - alpha, n_bootstrap=N_BOOTSTRAP, rng=rng, ci_func=ci_func, statistic="mean",
        )
    raise ValueError(widening)


def run() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    total_cells = len(LIKERT_SHAPES) * len(K_VALUES) * len(N_VALUES)
    cell_i = 0
    for shape in LIKERT_SHAPES:
        for k in K_VALUES:
            labels = [f"M{i}" for i in range(k)]
            effect_step = _jb_effect_magnitude("likert", EFFECT_FRAC, scale_bounds=LIKERT_SCALE)
            effects = np.arange(k, dtype=float) * effect_step
            # TRUE population means for this (shape, k, effects) -- a large,
            # separate Monte Carlo draw, NOT the small test sample's own mean
            # (which a CI trivially contains almost by construction, since
            # it's built from and centered near that same sample -- this was
            # the bug in the first version of this script: it silently tested
            # self-consistency, not calibration against the actual truth).
            true_means = sample_group_truth(
                shape, TRUE_MEAN_MC_N, 1, k, 1.0, rng, effects=effects,
            )[:, :, 0].mean(axis=1)
            extreme_pair = (labels[0], labels[-1])
            for n_items in N_VALUES:
                cell_i += 1
                per_method = {
                    m: dict(covered=0, total=0, width_sum=0.0, score_sum=0.0, n=0, extreme_reject=0)
                    for m in METHODS
                }
                ci_funcs = {m: build_ci_func(m, rng) for m in METHODS}
                for _rep in range(N_REPS):
                    truth = sample_group_truth(shape, n_items, 1, k, 1.0, rng, effects=effects)[:, :, 0]
                    for method in METHODS:
                        method_truth = dither(truth, rng) if method in DITHER_METHODS else truth
                        cis = family_wise_cis(method_truth, labels, ci_funcs[method], ALPHA, WIDENING[method], rng)
                        all_covered = True
                        for (a, b), (lo, hi) in cis.items():
                            true_diff = true_means[labels.index(a)] - true_means[labels.index(b)]
                            covered = lo <= true_diff <= hi
                            if not covered:
                                all_covered = False
                            per_method[method]["width_sum"] += hi - lo
                            per_method[method]["score_sum"] += interval_score(lo, hi, true_diff, ALPHA)
                            per_method[method]["n"] += 1
                            # Power: does the (family-wise-widened) CI for the
                            # extreme pair (largest true gap, M0 vs M{k-1})
                            # exclude zero -- the CI-duality equivalent of a
                            # FWER-corrected p-value < alpha for that pair.
                            if (a, b) == extreme_pair and (lo > 0.0 or hi < 0.0):
                                per_method[method]["extreme_reject"] += 1
                        per_method[method]["total"] += 1
                        if all_covered:
                            per_method[method]["covered"] += 1
                for method in METHODS:
                    d = per_method[method]
                    if d["n"] == 0:
                        print(f"\n  WARNING: {shape.label} k={k} n_items={n_items} method={method} -- "
                              f"family_wise_cis returned 0 pairs across all {N_REPS} reps, skipping cell")
                        continue
                    rows.append(dict(
                        shape=shape.label, k=k, n_items=n_items, method=method,
                        family_coverage=d["covered"] / d["total"],
                        mean_width=d["width_sum"] / d["n"],
                        mean_score=d["score_sum"] / d["n"],
                        power=d["extreme_reject"] / d["total"],
                    ))
                elapsed = time.time() - t0
                print(f"\r  cell {cell_i}/{total_cells}  ({elapsed:.0f}s elapsed)", end="", flush=True)
    print()
    return pd.DataFrame(rows)


METHOD_COLORS = {
    "logit_t": "#a6761d", "nig": "#888888", "smooth_bootstrap": "#9467bd",
    "logit_t_dither": "#1f77b4", "smooth_bootstrap_dither": "#d62728",
}


def save_by_k_violin_plot(df: pd.DataFrame, out_dir: str, run_stem: str) -> list[str]:
    """Grouped violin plots of per-shape family-wise coverage and interval
    score vs. n_items -- one ROW per k (not one column per eval_type, like
    ci_paired.py's by-n-violin-plot, since eval_type is fixed at likert here
    and k is the axis that actually drives the story: the same per-shape
    weakness that's a minor per-pair issue at k=2/3 compounds into a
    catastrophic family-wise failure at k=10, via the "ALL C(k,2) pairs must
    hold" AND). One violin per method at each n (dodged); each dot is one
    likert shape's family_coverage/mean_score at that (k, n, method)."""
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
        ("power", "Power per shape\n(extreme pair CI excludes zero)", "by_k_violin_power"),
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

        metric_label = {"family_coverage": "coverage", "mean_score": "interval score", "power": "power"}[metric]
        fig.suptitle(
            f"Likert family-wise {metric_label} vs. n_items, by arm count (k)\n"
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
    out_path = "simulations/out/investigate_likert_family_wise_smalln_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}\n")

    run_stem = f"investigate_likert_family_wise_smalln_reps{N_REPS}_{time.strftime('%Y%m%d_%H%M%S')}"
    plot_paths = save_by_k_violin_plot(df, "simulations/out/plots", run_stem)
    for p in plot_paths:
        print(f"Saved plot: {p}")

    print("=" * 100)
    print("Family-wise coverage, mean width, mean interval score, power -- pooled across all likert shapes")
    print(f"(alpha={ALPHA}, nominal target={1 - ALPHA:.0%}, reps={N_REPS})")
    print("=" * 100)
    for k in K_VALUES:
        print(f"\n--- k={k} ---")
        g = df[df.k == k].groupby(["n_items", "method"]).agg(
            family_coverage=("family_coverage", "mean"),
            mean_width=("mean_width", "mean"),
            mean_score=("mean_score", "mean"),
            power=("power", "mean"),
        ).reset_index()
        piv_cov = g.pivot(index="n_items", columns="method", values="family_coverage")[METHODS]
        piv_width = g.pivot(index="n_items", columns="method", values="mean_width")[METHODS]
        piv_score = g.pivot(index="n_items", columns="method", values="mean_score")[METHODS]
        piv_power = g.pivot(index="n_items", columns="method", values="power")[METHODS]
        print("Coverage:")
        print(piv_cov.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Mean width:")
        print(piv_width.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Mean score (lower=better):")
        print(piv_score.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Power (extreme pair CI excludes zero):")
        print(piv_power.to_string(float_format=lambda x: f"{x:.3f}"))
