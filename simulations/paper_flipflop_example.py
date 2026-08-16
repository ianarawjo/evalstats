"""Reproduces the numbers behind the paper's FlipFlop/Wavelength motivating
example (a UX researcher comparing app-store reviews across four apps with
an LLM judge) from REAL collected App Store review + judge-score data -- no
synthetic data anywhere in this script.

Data: simulations/out/appstore_scenario_reviews.csv (collected by
collect_appstore_reviews_only.py) + simulations/out/appstore_scenario_
judge_scores.csv (collected by collect_appstore_judge_scores_only.py).
Both gitignored, so they exist only in whichever checkout actually ran the
collection -- see those two scripts' docstrings to regenerate.

Judge model: anthropic/claude-haiku-4.5 -- one of five judges scored
against this data; selected because it shows the widest per-app kappa
spread (0.646 on FlipFlop vs. 0.881 on Wavelength) while still having a
solid pooled headline (0.78, "substantial"). All five judges independently
showed their own worst calibration on FlipFlop, so this isn't specific to
one model's quirks -- see the analysis that picked this judge for the full
per-judge comparison.

Real per-app N=300 here (not 50 like the original single-judge dataset),
so unlike that first pass, this script doesn't need to caveat working at a
small scale -- the labeling budget (N_LAB=15/app) stays modest by design
(that's the point of the demo), but the underlying pool is now genuinely
large.

Fictionalized name mapping (app names in the paper are fictional, the
underlying reviews and ratings are real; the in-paper explanation for WHY
FlipFlop's judge struggles -- "meme-fluent, ironic register" -- is
deliberately fictionalized narrative color, not a claim about the actual
mechanism):
    TikTok -> FlipFlop        (the app under study)
    Google Maps -> Wavelength (its closest raw-judge "competitor")
    Instagram -> Snippet
    Facebook -> Bubblegum

Run:
    .venv/bin/python -m simulations.paper_flipflop_example
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy import stats

import evalstats as es
from evalstats.alignment import judge_alignment

DEFAULT_DATA_DIR = "simulations/out"
JUDGE = "anthropic/claude-haiku-4.5"
APP_ID_TO_REAL_NAME = {
    "284882215": "Facebook", "835599320": "TikTok",
    "585027354": "Google Maps", "389801252": "Instagram",
}
REAL_NAME_TO_FICTIONAL = {
    "TikTok": "FlipFlop", "Google Maps": "Wavelength",
    "Instagram": "Snippet", "Facebook": "Bubblegum",
}
N_LAB = 15  # matches the paper's `evalstats label ... --n-lab 15`


def load(data_dir: str) -> pd.DataFrame:
    items_path = f"{data_dir}/appstore_scenario_reviews.csv"
    scores_path = f"{data_dir}/appstore_scenario_judge_scores.csv"
    try:
        items = pd.read_csv(items_path)
        scores = pd.read_csv(scores_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc.filename} not found -- run collect_appstore_reviews_only.py and "
            "collect_appstore_judge_scores_only.py first (see their docstrings)."
        ) from exc
    scores = scores[scores["judge_model"] == JUDGE]
    df = items.merge(scores[["item_id", "judge_score"]], on="item_id", how="inner")
    df["app_id"] = df["app_id"].astype(str)
    df["app_real"] = df["app_id"].map(APP_ID_TO_REAL_NAME)
    df["app"] = df["app_real"].map(REAL_NAME_TO_FICTIONAL)
    if df["app"].isna().any():
        raise ValueError("Some app_ids aren't in APP_ID_TO_REAL_NAME -- update the mapping.")
    df["human_label"] = df["human_label"].astype(float)
    return df[["item_id", "app", "app_real", "human_label", "judge_score"]].reset_index(drop=True)


def print_kappa_table(df: pd.DataFrame) -> None:
    print("=== Weighted (quadratic) Cohen's kappa, real data ===")
    pooled_k = cohen_kappa_score(df["human_label"], df["judge_score"], weights="quadratic")
    print(f"Pooled (all {len(df)}):      kappa = {pooled_k:.2f}")
    by_mean = sorted(df["app"].unique(), key=lambda a: -df[df["app"] == a]["human_label"].mean())
    for app in by_mean:
        sub = df[df["app"] == app]
        k = cohen_kappa_score(sub["human_label"], sub["judge_score"], weights="quadratic")
        real = sub["app_real"].iloc[0]
        print(f"  {app:<12s} ({real:<11s}) kappa = {k:.2f}  n={len(sub):<4d} "
              f"human_mean={sub['human_label'].mean():.2f}  judge_mean={sub['judge_score'].mean():.2f}")


def _naive_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, tuple[float, float], float]:
    """A basic Welch two-sample comparison -- what a researcher who skipped
    evalstats entirely might run directly on raw scores."""
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return diff, ci, p


def _pairwise_row(result, label_a: str, label_b: str):
    return next(p for p in result.pairwise if {p.label_a, p.label_b} == {label_a, label_b})


def sample_labels(df: pd.DataFrame, n_lab: int, seed: int) -> pd.DataFrame:
    """Reveal n_lab real human labels per app (random, MCAR); the rest stay judge-only."""
    rng = np.random.default_rng(seed)
    out = df.rename(columns={"judge_score": "satisfaction_score", "item_id": "item"}).copy()
    out["human_score"] = np.nan
    for app in out["app"].unique():
        idx = out.index[out["app"] == app].to_numpy()
        chosen = rng.choice(idx, size=n_lab, replace=False)
        out.loc[chosen, "human_score"] = df.loc[chosen, "human_label"]
    return out.drop(columns=["human_label", "app_real"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    df = load(args.data_dir)
    print(f"Loaded {len(df)} real reviews across {df['app'].nunique()} apps, judge={JUDGE}")
    print()
    print_kappa_table(df)

    print()
    print(f"=== Ground truth: all {len(df) // df['app'].nunique()} real human labels/app, via es.compare(design='unpaired') ===")
    truth_df = df.rename(columns={"item_id": "item", "human_label": "satisfaction_score"}).drop(
        columns=["judge_score", "app_real"]
    )
    truth_result = es.compare(
        es.load_from(truth_df), factors="app", metric="satisfaction_score", design="unpaired",
        score_range=(1, 5), rng=np.random.default_rng(99), n_bootstrap=2000,
    )
    truth_row = _pairwise_row(truth_result, "FlipFlop", "Wavelength")
    truth_means = {g.label: g.mean for g in truth_result.groups}
    sign = 1 if truth_row.label_a == "FlipFlop" else -1
    null = 0.5
    dt = sign * (truth_row.point_estimate - null)
    lo, hi = sorted((sign * (truth_row.ci_low - null), sign * (truth_row.ci_high - null)))
    print(f"  FlipFlop mean={truth_means['FlipFlop']:.3f}, Wavelength mean={truth_means['Wavelength']:.3f}, "
          f"diff={truth_means['FlipFlop'] - truth_means['Wavelength']:+.3f}")
    print(f"  Delta-theta (FlipFlop - Wavelength): {dt:+.4f}, "
          f"95% CI=[{lo:+.4f}, {hi:+.4f}], p={truth_row.p_value:.4f}, significant={truth_row.significant}")
    diff_t, ci_t, p_t = _naive_two_sample(
        df[df["app"] == "FlipFlop"]["human_label"].to_numpy(),
        df[df["app"] == "Wavelength"]["human_label"].to_numpy(),
    )
    print(f"  (naive Welch t-test on the same full data, for reference: "
          f"diff={diff_t:+.3f}, 95% CI=[{ci_t[0]:+.3f}, {ci_t[1]:+.3f}], p={p_t:.4f})")

    print()
    print("=== Raw judge scores directly (no evalstats/PPI), FlipFlop vs Wavelength ===")
    diff_j, ci_j, p_j = _naive_two_sample(
        df[df["app"] == "FlipFlop"]["judge_score"].to_numpy(),
        df[df["app"] == "Wavelength"]["judge_score"].to_numpy(),
    )
    print(f"  Welch t-test: diff={diff_j:+.3f}, 95% CI=[{ci_j[0]:+.3f}, {ci_j[1]:+.3f}], p={p_j:.4f}")

    print()
    print(f"=== evalstats: PPI-corrected compare(design='unpaired'), n_lab={N_LAB}/app, real labels ===")
    lab_df = sample_labels(df, N_LAB, seed=42)
    evaldata = es.load_from(lab_df)
    ar = judge_alignment(
        evaldata, llm_metric="satisfaction_score", human_groundtruth="human_score",
        selection="random",
    )
    result = es.compare(
        evaldata, factors="app", metric="satisfaction_score", design="unpaired",
        alignment={"satisfaction_score": ar}, score_range=(1, 5),
        rng=np.random.default_rng(43), n_bootstrap=2000,
    )
    row = _pairwise_row(result, "FlipFlop", "Wavelength")
    means = {g.label: g.mean for g in result.groups}
    sign = 1 if row.label_a == "FlipFlop" else -1
    dt2 = sign * (row.point_estimate - null)
    lo2, hi2 = sorted((sign * (row.ci_low - null), sign * (row.ci_high - null)))
    print(f"  Delta-theta (FlipFlop - Wavelength): {dt2:+.4f}, "
          f"95% CI=[{lo2:+.4f}, {hi2:+.4f}], p={row.p_value:.4f}, significant={row.significant}")
    print(f"  PPI-corrected means: FlipFlop={means['FlipFlop']:.3f}, Wavelength={means['Wavelength']:.3f}, "
          f"diff={means['FlipFlop'] - means['Wavelength']:+.3f}")
    print()
    print("----- Full evalstats output (result.summary()) -----")
    result.summary()

    print()
    print(f"=== Robustness: 10 independent random {N_LAB}-per-app label draws (real data) ===")
    n_corrected_significant = 0
    for trial in range(10):
        tdf = sample_labels(df, N_LAB, seed=1000 + trial)
        ev = es.load_from(tdf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar_t = judge_alignment(
                ev, llm_metric="satisfaction_score", human_groundtruth="human_score", selection="random",
            )
        res_t = es.compare(
            ev, factors="app", metric="satisfaction_score", design="unpaired",
            alignment={"satisfaction_score": ar_t}, score_range=(1, 5),
            rng=np.random.default_rng(2000 + trial), n_bootstrap=1000,
        )
        if _pairwise_row(res_t, "FlipFlop", "Wavelength").significant:
            n_corrected_significant += 1
    raw_significant = not (ci_j[0] <= 0 <= ci_j[1])
    print(f"  Raw judge comparison significant (doesn't depend on the label draw): {raw_significant}")
    print(f"  PPI-corrected comparison significant in {n_corrected_significant}/10 real-data draws")


if __name__ == "__main__":
    main()
