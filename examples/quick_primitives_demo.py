"""Quick primitives: calibrated numbers for users who don't want compare()'s
full comparative report -- just a trustworthy point estimate (or a few other
common building blocks), returned as plain data to hand to your own plotting
library or downstream code.

Every result here reuses the exact same auto method-selection and
calibration machinery compare() uses internally, so a number computed here
and the equivalent number inside a compare() report are identical.

Usage:
    python examples/quick_primitives_demo.py
"""

import numpy as np

import evalstats as es


rng = np.random.default_rng(0)

print("=" * 70)
print("1. mean_ci() -- calibrated mean + CI for a single array")
print("=" * 70)

accuracy = np.clip(rng.normal(0.82, 0.09, 60), 0, 1)
result = es.mean_ci(accuracy)
print(f"mean={result.mean:.3f}  95% CI=[{result.ci_low:.3f}, {result.ci_high:.3f}]  "
      f"n={result.n}  method={result.method}")
# Unpacks positionally too, for the "just give me the numbers" case:
mean, lo, hi, n, method = result
print(f"unpacked: {mean:.3f}, [{lo:.3f}, {hi:.3f}]")
print()

print("=" * 70)
print("2. summarize() -- descriptive + CI table for several groups at once")
print("=" * 70)

scores_by_model = {
    "gpt-4o": np.clip(rng.normal(0.85, 0.08, 50), 0, 1),
    "claude-sonnet": np.clip(rng.normal(0.80, 0.09, 50), 0, 1),
    "llama-70b": np.clip(rng.normal(0.70, 0.10, 45), 0, 1),  # different N is fine
}
table = es.summarize(scores_by_model)
print(table.to_frame()[["mean", "ci_low", "ci_high", "n", "method"]])
print()
print("Same data as a plain dict, e.g. to write out as JSON:")
print(table.to_dict()["gpt-4o"])
print()

print("=" * 70)
print("3. stability() -- multi-run reliability, standalone")
print("=" * 70)

M, K = 150, 5  # 150 items, 5 repeated runs
base = rng.normal(0.78, 0.1, M)
stable_runs = np.array([np.clip(base + rng.normal(0, 0.02, M), 0, 1) for _ in range(K)])
flaky_runs = np.array([np.clip(rng.normal(0.6, 0.2, M), 0, 1) for _ in range(K)])

stab = es.stability({"rag_config_a": stable_runs, "rag_config_b": flaky_runs})
print(stab.to_frame())
print()

print("=" * 70)
print("4. judge_alignment() -- array-based form, no load_from() needed")
print("=" * 70)

# The natural shape most real data comes in: a judge score for every item,
# and a human score that's only filled in for a small labeled subset (NaN
# elsewhere). No manual masking needed -- pass both arrays as-is.
n_total = 300
n_labeled = 40
judge_all_likert = np.clip(
    np.round(rng.integers(1, 6, n_total).astype(float) + rng.normal(0.3, 0.6, n_total)), 1, 5
)
human_sparse = np.full(n_total, np.nan)
labeled_idx = rng.choice(n_total, n_labeled, replace=False)
human_sparse[labeled_idx] = np.clip(
    np.round(judge_all_likert[labeled_idx] + rng.normal(-0.3, 0.5, n_labeled)), 1, 5
)

alignment = es.judge_alignment(judge_all_likert, human_sparse)
print(f"score_type={alignment.score_type}  n_labeled={alignment.n_labeled}  n_total={alignment.n_total}")
print(f"representativeness check ran automatically: {'score_distribution' in alignment.representativeness}")
kappa = alignment.alignment_metrics.get("weighted_kappa") or alignment.alignment_metrics.get("cohens_kappa")
if kappa is not None:
    print(f"weighted kappa: {kappa['estimate']:.3f}  95% CI=[{kappa['ci_low']:.3f}, {kappa['ci_high']:.3f}]")
print()

print("=" * 70)
print("5. judge_debias_mean_ci() -- PPI-corrected mean, judge scores only")
print("=" * 70)

n_total = 400
n_labeled = 40
true_mean = 0.55
judge_bias = 0.18  # judge systematically overrates

human_all = np.clip(rng.normal(true_mean, 0.15, n_total), 0, 1)
judge_all = np.clip(human_all + judge_bias + rng.normal(0, 0.05, n_total), 0, 1)

# Same sparse convention as judge_alignment() above: one array per item,
# human scores NaN outside the labeled subset.
human_sparse2 = np.full(n_total, np.nan)
labeled_idx2 = rng.choice(n_total, n_labeled, replace=False)
human_sparse2[labeled_idx2] = human_all[labeled_idx2]

import warnings
with warnings.catch_warnings():
    # judge_debias_mean_ci always reminds you the labeled subset must be a
    # random sample -- expected here since we did sample uniformly at random.
    warnings.simplefilter("ignore", UserWarning)
    debiased = es.judge_debias_mean_ci(judge_all, human_sparse2)

print(f"judge-only mean (biased):   {debiased.judge_mean:.3f}")
print(f"PPI-corrected mean:         {debiased.mean:.3f}  "
      f"95% CI=[{debiased.ci_low:.3f}, {debiased.ci_high:.3f}]")
print(f"true mean (for comparison): {true_mean:.3f}")
print(f"rectifier: {debiased.rectifier:+.3f}  "
      f"(n_labeled={debiased.n_labeled}, n_unlabeled={debiased.n_unlabeled})")
