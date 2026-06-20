"""Demonstration of evalstats.tests: ttest, mannwhitney, wilcoxon, anova_oneway.

Five scenarios using synthetic LLM judge scores, each with a different
research question typical of AI/HCI evaluation:

  1. Independent t-test        — do two models differ on a continuous quality scale?
  2. Mann-Whitney U test       — non-parametric comparison on ordinal Likert scores
  3. Wilcoxon signed-rank      — paired comparison (same items rated by two prompts)
  4. One-way ANOVA             — three AI writing assistants, between-subjects design
  5. Repeated-measures ANOVA   — three prompt strategies, within-subjects design

Usage:
    python examples/hypothesis_tests_demo.py
"""

import numpy as np
from evalstats.tests import ttest, mannwhitney, wilcoxon, anova_oneway


# Data generation uses a shared rng; bootstrap seeds are fixed integers so
# changing n_boot never affects the synthetic data draws.
rng = np.random.default_rng(2025)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _section(title):
    width = 64
    print()
    print("─" * width)
    print(f"  {title}")
    print("─" * width)


def _sparse_labels(truth, rng, n_labeled):
    """Return a sparse label array: n_labeled items revealed, rest NaN."""
    labels = np.full(len(truth), np.nan)
    idx = rng.choice(len(truth), n_labeled, replace=False)
    labels[idx] = truth[idx]
    return labels


def _sparse_labels_paired(truth_x, truth_y, rng, n_labeled):
    """Sparse labels for a paired test — same positions labeled in both arrays."""
    idx = rng.choice(len(truth_x), n_labeled, replace=False)
    x_lab = np.full(len(truth_x), np.nan); x_lab[idx] = truth_x[idx]
    y_lab = np.full(len(truth_y), np.nan); y_lab[idx] = truth_y[idx]
    return x_lab, y_lab


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Independent t-test
# Research question: "Does Model B score significantly higher than Model A on
# a continuous helpfulness scale (0–5)?"
#
# Ground truth: Model B is better (mean 3.8 vs 3.4).
# LLM judge: accurate but adds small noise.
# Human labels: 30 of 150 items per group are spot-checked.
# ─────────────────────────────────────────────────────────────────────────────

_section("Scenario 1 — Independent t-test (continuous scores, two conditions)")

n = 150
truth_a = rng.normal(3.4, 0.9, n)
truth_b = rng.normal(3.8, 0.9, n)

# LLM scores: unbiased but noisy
llm_a = truth_a + rng.normal(0, 0.15, n)
llm_b = truth_b + rng.normal(0, 0.15, n)

# Human spot-checks on 30 items per group
human_a = _sparse_labels(truth_a, rng, n_labeled=30)
human_b = _sparse_labels(truth_b, rng, n_labeled=30)

ttest(llm_a, llm_b, a_lab=human_a, b_lab=human_b, n_boot=10_000, rng=1)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Mann-Whitney U test
# Research question: "Does Prompt A produce higher quality responses than
# Prompt B on a 1–5 Likert scale?"
#
# Ground truth: NO difference — both prompts score identically.
# LLM judge: inflates Prompt A scores by +0.5 (differential bias).
# Without correction the bias makes it look significant; PPI removes it.
# Human labels: 60 of 200 items per group are spot-checked.
# ─────────────────────────────────────────────────────────────────────────────

_section("Scenario 2 — Mann-Whitney U test (false positive corrected by PPI)")

def _likert(mu, sigma, n, rng):
    return np.clip(np.round(rng.normal(mu, sigma, n)), 1, 5)

n = 200
truth_x = _likert(3.3, 1.1, n, rng)
truth_y = _likert(3.3, 1.1, n, rng)   # identical distribution — no true effect

# LLM inflates Prompt-A scores only (differential bias creates a false positive)
llm_x = np.clip(truth_x + 0.5 + rng.normal(0, 0.15, n), 1, 5)
llm_y = np.clip(truth_y        + rng.normal(0, 0.15, n), 1, 5)

human_x = _sparse_labels(truth_x, rng, n_labeled=60)
human_y = _sparse_labels(truth_y, rng, n_labeled=60)

mannwhitney(llm_x, llm_y, x_lab=human_x, y_lab=human_y, n_boot=10_000, rng=2)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Wilcoxon signed-rank test
# Research question: "Does Chain-of-Thought prompting improve scores compared
# to a standard prompt, measured on the same 80 test items?"
#
# Ground truth: CoT helps by +0.3 median points.
# LLM judge: near-perfect, tiny noise only.
# Human labels: 25 of 80 paired items are spot-checked.
# ─────────────────────────────────────────────────────────────────────────────

_section("Scenario 3 — Wilcoxon signed-rank test (paired items, CoT vs. standard)")

n = 80
base = rng.normal(3.2, 1.0, n)
truth_cot      = base + rng.normal(0.3, 0.3, n)   # CoT adds ~0.3
truth_standard = base + rng.normal(0.0, 0.3, n)

llm_cot      = truth_cot + rng.normal(0, 0.1, n)
llm_standard = truth_standard + rng.normal(0, 0.1, n)

human_cot, human_standard = _sparse_labels_paired(truth_cot, truth_standard, rng, n_labeled=25)

wilcoxon(llm_cot, llm_standard,
         x_lab=human_cot, y_lab=human_standard,
         n_boot=10_000, rng=3)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: One-way ANOVA (independent groups / between-subjects)
# Research question: "Do three AI writing assistants (A, B, C) produce
# different fluency scores on a held-out essay test?"
#
# Ground truth: Assistant B is genuinely better (mean 4.2 vs 3.5 for A and C).
# LLM judge: differentially biased — it over-scores A/C and under-scores B.
#   Uncorrected ANOVA underestimates the true gap; PPI uses human labels to
#   correct that bias.
# Human labels: 40 of 150 essays per assistant are annotated by a human rater.
# ─────────────────────────────────────────────────────────────────────────────

_section("Scenario 4 — One-way ANOVA (three AI writing assistants, between-subjects)")

n = 150
truth_a = rng.normal(3.5, 0.9, n)
truth_b = rng.normal(4.2, 0.9, n)   # Assistant B is genuinely better
truth_c = rng.normal(3.5, 0.9, n)

# LLM fluency scores: biased (A/C inflated, B deflated) + small noise
llm_a = truth_a + 0.25 + rng.normal(0, 0.12, n)
llm_b = truth_b - 0.35 + rng.normal(0, 0.14, n)
llm_c = truth_c + 0.15 + rng.normal(0, 0.16, n)

# Human spot-checks on 40 essays per assistant (independent samples)
human_a = _sparse_labels(truth_a, rng, n_labeled=40)
human_b = _sparse_labels(truth_b, rng, n_labeled=40)
human_c = _sparse_labels(truth_c, rng, n_labeled=40)

anova_oneway(llm_a, llm_b, llm_c,
             groups_lab=[human_a, human_b, human_c],
             n_boot=10_000, rng=4)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5: Repeated-measures one-way ANOVA (within-subjects)
# Research question: "Does prompting strategy (zero-shot, few-shot, CoT)
# affect response quality across the same 120 benchmark questions?"
#
# Ground truth: NO difference — all three strategies score equally.
# LLM judge: inflates few-shot scores by +0.7 (differential bias).
#   Without correction the bias makes few-shot look significantly better.
#   PPI uses human annotations to remove the false positive.
# Human labels: 40 of 120 questions annotated under all three strategies
#   (same questions labeled, as required for within-subjects analysis).
# ─────────────────────────────────────────────────────────────────────────────

_section("Scenario 5 — Repeated-measures ANOVA (three prompt strategies, within-subjects)")

n = 120
subject_fx = rng.normal(0, 0.6, n)           # per-question difficulty

truth_zeroshot = 3.4 + subject_fx + rng.normal(0, 0.8, n)
truth_fewshot  = 3.4 + subject_fx + rng.normal(0, 0.8, n)  # same true quality
truth_cot      = 3.4 + subject_fx + rng.normal(0, 0.8, n)

# LLM inflates few-shot scores only (differential bias → false positive)
llm_zeroshot = truth_zeroshot + rng.normal(0, 0.12, n)
llm_fewshot  = truth_fewshot  + 0.7 + rng.normal(0, 0.12, n)
llm_cot      = truth_cot      + rng.normal(0, 0.12, n)

# Same 40 questions annotated under all three strategies (within-subjects requirement)
idx_labeled = rng.choice(n, 40, replace=False)
human_zeroshot = np.full(n, np.nan); human_zeroshot[idx_labeled] = truth_zeroshot[idx_labeled]
human_fewshot  = np.full(n, np.nan); human_fewshot[idx_labeled]  = truth_fewshot[idx_labeled]
human_cot      = np.full(n, np.nan); human_cot[idx_labeled]      = truth_cot[idx_labeled]

anova_oneway(llm_zeroshot, llm_fewshot, llm_cot,
             repeated=True,
             groups_lab=[human_zeroshot, human_fewshot, human_cot],
             n_boot=10_000, rng=5)
