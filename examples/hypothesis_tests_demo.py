"""Demonstration of evalstats.tests: ttest, mannwhitney, and wilcoxon.

Three scenarios using synthetic LLM judge scores, each with a different
research question typical of AI/HCI evaluation:

  1. Independent t-test   — do two models differ on a continuous quality scale?
  2. Mann-Whitney U test  — non-parametric comparison on ordinal Likert scores
  3. Wilcoxon signed-rank — paired comparison (same items rated by two prompts)

Usage:
    python examples/hypothesis_tests_demo.py
"""

import numpy as np
from evalstats.tests import ttest, mannwhitney, wilcoxon


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

ttest(llm_a, llm_b, a_lab=human_a, b_lab=human_b, n_boot=1000, rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Mann-Whitney U test
# Research question: "Does Prompt A produce higher quality responses than
# Prompt B on a 1–5 Likert scale?"
#
# Ground truth: NO difference — both prompts score identically.
# LLM judge: inflates Prompt A scores by +0.5 (differential bias).
# Without correction the bias makes it look significant; PPI removes it.
# Human labels: 30 items per group are spot-checked.
# ─────────────────────────────────────────────────────────────────────────────

_section("Scenario 2 — Mann-Whitney U test (false positive corrected by PPI)")

def _likert(mu, sigma, n, rng):
    return np.clip(np.round(rng.normal(mu, sigma, n)), 1, 5)

n = 150
truth_x = _likert(3.3, 1.1, n, rng)
truth_y = _likert(3.3, 1.1, n, rng)   # identical distribution — no true effect

# LLM inflates Prompt-A scores only (differential bias creates a false positive)
llm_x = np.clip(truth_x + 0.5 + rng.normal(0, 0.15, n), 1, 5)
llm_y = np.clip(truth_y        + rng.normal(0, 0.15, n), 1, 5)

human_x = _sparse_labels(truth_x, rng, n_labeled=30)
human_y = _sparse_labels(truth_y, rng, n_labeled=30)

mannwhitney(llm_x, llm_y, x_lab=human_x, y_lab=human_y, n_boot=1000, rng=rng)


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
         n_boot=1000, rng=rng)
