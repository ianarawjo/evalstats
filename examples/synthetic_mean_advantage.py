"""Test and demo for evalstats: generates synthetic data and produces a
robustness-first interval plot."""

import numpy as np

import evalstats as estats

# --- Synthetic benchmark data ---
# 6 templates, 100 inputs, designed to illustrate different behaviors
rng = np.random.default_rng(42)
M = 100  # inputs

scores = np.zeros((6, M))

# Template A: "The Reliable Winner" — high mean, low variance
scores[0] = rng.normal(loc=8.0, scale=0.8, size=M)

# Template B: "The Close Second" — slightly lower mean, also low variance
scores[1] = rng.normal(loc=7.5, scale=0.9, size=M)

# Template C: "The Volatile Genius" — high mean but huge variance
# Sometimes brilliant (10), sometimes terrible (3)
scores[2] = np.where(
    rng.random(M) > 0.3,
    rng.normal(loc=9.0, scale=0.5, size=M),
    rng.normal(loc=4.0, scale=1.0, size=M),
)

# Template D: "The Mediocre but Steady" — average score, very tight
scores[3] = rng.normal(loc=6.5, scale=0.4, size=M)

# Template E: "The Underperformer" — clearly below average
scores[4] = rng.normal(loc=5.0, scale=1.2, size=M)

# Template F: "Barely Different from Average" — right at the mean, moderate variance
scores[5] = rng.normal(loc=6.8, scale=1.0, size=M)

# Clip to reasonable range
scores = np.clip(scores, 0, 10)

labels = [
    "A: Reliable Winner",
    "B: Close Second",
    "C: Volatile Genius",
    "D: Steady Mediocre",
    "E: Underperformer",
    "F: Near Average",
]
input_labels = [f"input_{i:03d}" for i in range(M)]

# --- Create BenchmarkResult ---
result = estats.BenchmarkResult(
    scores=scores,
    template_labels=labels,
    input_labels=input_labels,
)

print(f"Created BenchmarkResult: {result.n_templates} templates × {result.n_inputs} inputs")
print()

# --- Robustness metrics ---
rob = estats.robustness_metrics(scores, labels, failure_threshold=4.0)
print("=== Robustness Summary ===")
print(rob.summary_table().to_string())
print()

# --- Pairwise differences (A vs C as an interesting pair) ---
diff_ac = estats.pairwise_differences(
    scores, 0, 2, labels[0], labels[2], method="auto", rng=rng,
)
print(f"=== Pairwise: {diff_ac.template_a} vs {diff_ac.template_b} ===")
print(f"  Mean diff: {diff_ac.point_diff:+.3f}")
print(f"  95% CI:    [{diff_ac.ci_low:+.3f}, {diff_ac.ci_high:+.3f}]")
print(f"  p-value:   {diff_ac.p_value:.4f}")
print(f"  Effect size (rank-biserial): {diff_ac.effect_size:.3f}")
print()

# --- Bootstrap ranking ---
ranks = estats.bootstrap_ranks(scores, labels, n_bootstrap=10_000, rng=rng)
print("=== Bootstrap Rank Probabilities ===")
template_col_width = min(40, max(len("Template") + 1, max(len(label) for label in labels) + 2))
print(f"{'Template':<{template_col_width}s} {'P(Best)':>8s} {'E[Rank]':>8s}")
for i, label in enumerate(labels):
    print(
        f"  {label:<{template_col_width}.{template_col_width}s} "
        f"{ranks.p_best[i]:>7.1%} {ranks.expected_ranks[i]:>7.2f}"
    )
print()

# --- Marginal CIs on absolute means ---
rob_ci = estats.robustness_metrics(
    scores,
    labels,
    failure_threshold=4.0,
    n_bootstrap=10_000,
    rng=np.random.default_rng(42),
    alpha=0.05,
    statistic="mean",
    marginal_method="smooth_bootstrap",
)
print("=== Absolute Mean Performance ===")
print(f"{'Template':<25s} {'Mean':>7s} {'CI Low':>8s} {'CI High':>8s}")
for i in range(len(labels)):
    print(
        f"  {labels[i]:<23s} "
        f"{rob_ci.mean[i]:>+6.3f} "
        f"{rob_ci.ci_low[i]:>+7.3f} "
        f"{rob_ci.ci_high[i]:>+7.3f}"
    )
print()

# --- Generate the plot ---
fig = estats.plot_point_estimates(result, rng=np.random.default_rng(42))
fig.savefig("robustness_interval_plot.png", dpi=150, bbox_inches="tight")
print("Saved: robustness_interval_plot.png")

# Also generate a version with a custom title
fig2 = estats.plot_point_estimates(
    result,
    title="Absolute Performance by Template",
    rng=np.random.default_rng(42),
)
fig2.savefig("robustness_interval_plot_titled.png", dpi=150, bbox_inches="tight")
print("Saved: robustness_interval_plot_titled.png")

print("\nDone!")
