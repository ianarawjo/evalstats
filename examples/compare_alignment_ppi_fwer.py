"""Compound scenario: PPI alignment correction layered on top of FWER
(family-wise error rate) control, for a k >= 3 model comparison.

Simulates a common real-world scenario:
  - 4 models are scored by an LLM judge on 150 items each (cheap, scalable).
  - A human annotator has labelled 60 of those items per model (expensive,
    sparse) -- judge_alignment()/compare(alignment=...) use these to
    debias the LLM-only estimates via Prediction-Powered Inference (PPI).
  - With 4 models there are C(4,2) = 6 pairwise comparisons, so a family-wise
    error rate (FWER) correction is also needed to avoid false positives
    from testing 6 hypotheses at once (compare(..., simultaneous_ci=True,
    correction=...)).

This demonstrates that PPI correction and FWER control now compose
correctly through the SAME compare() call: both the simultaneous-CI
construction ("sidak" for small N, "boot" -- joint bootstrap with an
effective alpha -- for larger N) AND the p-value correction ("shaffer" for
small N, "romano_wolf" step-down for larger N) resolve automatically on top
of the PPI-corrected estimates, matching evalstats' decision tree exactly --
not just on raw LLM scores, and not silently downgraded to a weaker
fallback.

Prints four side-by-side comparisons so you can see what each layer costs:
  1. Uncorrected      -- raw LLM scores, no PPI, no FWER control
  2. PPI-only         -- judge-bias corrected, but no FWER control
  3. FWER-only        -- raw LLM scores, FWER-controlled across all 6 pairs
  4. Compound         -- PPI-corrected AND FWER-controlled together

Usage:
    python examples/compare_alignment_ppi_fwer.py
"""

import numpy as np
import pandas as pd

import evalstats as es

# ── Simulation parameters ─────────────────────────────────────────────────────

rng = np.random.default_rng(7)

N_ITEMS   = 150   # items per model
N_LABELED = 60    # items per model with human annotation (40% labeled)

# True human pass rates (what an unbiased judge would measure)
TRUE_RATE = {
    "model_A": 0.70,
    "model_B": 0.55,
    "model_C": 0.48,
    "model_D": 0.40,
}

# LLM judge parameters (True/False Positive Rate). model_A's judge is
# noticeably biased (high false-positive rate) -- the other three have a
# more evenly-noisy judge.
LLM_TPR = {"model_A": 0.90, "model_B": 0.85, "model_C": 0.85, "model_D": 0.85}
LLM_FPR = {"model_A": 0.55, "model_B": 0.20, "model_C": 0.20, "model_D": 0.20}

HUMAN_ACCURACY = 0.95  # near-perfect human rater

# ── Build the full dataset ────────────────────────────────────────────────────
# Each (model, item) pair has a latent true quality (1 = pass, 0 = fail).
# The LLM and human both observe the same item, producing correlated scores.

rows = []
# The SAME items are human-labeled for every model (paired by input) -- this
# is what lets a simultaneous-CI construction that accounts for correlation
# between pairs (evalstats' "boot": joint bootstrap with an effective alpha)
# actually engage. Labeling a different random subset of items per model
# (as you might if annotators are assigned per-model rather than per-item)
# leaves too little item overlap between pairs for that joint construction,
# and evalstats falls back to the (still valid, just more conservative)
# Bonferroni construction instead -- with a UserWarning explaining why.
gold_indices = rng.choice(N_ITEMS, size=N_LABELED, replace=False)

for model in TRUE_RATE:
    true_rate = TRUE_RATE[model]
    tpr = LLM_TPR[model]
    fpr = LLM_FPR[model]

    quality = rng.binomial(1, true_rate, N_ITEMS)

    human_score = np.where(
        quality == 1,
        rng.binomial(1, HUMAN_ACCURACY, N_ITEMS),
        rng.binomial(1, 1 - HUMAN_ACCURACY, N_ITEMS),
    ).astype(float)

    llm_score = np.where(
        quality == 1,
        rng.binomial(1, tpr, N_ITEMS),
        rng.binomial(1, fpr, N_ITEMS),
    ).astype(float)

    for item_id in range(N_ITEMS):
        rows.append({
            "model": model,
            "item": item_id,
            "llm_score": llm_score[item_id],
            "human_score": float("nan"),  # filled in below for labeled items
        })

    for idx in gold_indices:
        row_loc = len(rows) - N_ITEMS + idx
        rows[row_loc]["human_score"] = human_score[idx]

df = pd.DataFrame(rows)
evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})

print("Simulation check — LLM win rates (expected vs actual):")
for model in TRUE_RATE:
    m = df[df["model"] == model]
    llm_mean = m["llm_score"].mean()
    human_mean = m["human_score"].dropna().mean()
    exp_llm = LLM_TPR[model] * TRUE_RATE[model] + LLM_FPR[model] * (1 - TRUE_RATE[model])
    print(f"  {model}: true={TRUE_RATE[model]:.2f}  "
          f"llm_all={llm_mean:.3f} (expected≈{exp_llm:.3f})  "
          f"human_labeled={human_mean:.3f}")
print()

# ── Validate alignment ────────────────────────────────────────────────────────

print("=" * 70)
print("STEP 1 — Validate LLM judge alignment")
print("=" * 70)

ar = es.judge_alignment(
    evaldata,
    llm_metric="llm_score",
    human_groundtruth="human_score",
    selection="random",  # gold_indices above was rng.choice(..., replace=False)
)
ar.summary()

# ── 1. Uncorrected: raw LLM scores, no PPI, no FWER control ──────────────────

print("=" * 70)
print("STEP 2 — Uncorrected  (raw LLM scores, no PPI, no FWER control)")
print("=" * 70)

result_raw = es.compare(
    evaldata,
    factors="model",
    metric="llm_score",
    simultaneous_ci=False,
    correction="none",
    rng=np.random.default_rng(1),
)
result_raw.summary()

# ── 2. PPI-only: judge-bias corrected, but no FWER control ───────────────────

print("=" * 70)
print("STEP 3 — PPI-only  (judge-bias corrected, no FWER control)")
print("=" * 70)

result_ppi_only = es.compare(
    evaldata,
    factors="model",
    metric="llm_score",
    alignment={"llm_score": ar},
    simultaneous_ci=False,
    correction="none",
    rng=np.random.default_rng(2),
)
result_ppi_only.summary()

# ── 3. FWER-only: raw LLM scores, FWER-controlled across all 6 pairs ────────

print("=" * 70)
print("STEP 4 — FWER-only  (raw LLM scores, FWER-controlled)")
print("=" * 70)

result_fwer_only = es.compare(
    evaldata,
    factors="model",
    metric="llm_score",
    simultaneous_ci=True,
    correction="auto",
    rng=np.random.default_rng(3),
)
result_fwer_only.summary()
bundle_fwer_only = result_fwer_only._primary_bundle()
print(f"  -> simultaneous_ci_method resolved to: "
      f"{bundle_fwer_only.pairwise.simultaneous_ci_method!r}")
print(f"  -> correction_method resolved to:      "
      f"{bundle_fwer_only.pairwise.correction_method!r}")
print()

# ── 4. Compound: PPI-corrected AND FWER-controlled together ─────────────────

print("=" * 70)
print("STEP 5 — Compound  (PPI-corrected AND FWER-controlled)")
print("=" * 70)
print(f"  ({N_LABELED} human labels per model, {len(TRUE_RATE)} models -> "
      f"{len(TRUE_RATE) * (len(TRUE_RATE) - 1) // 2} pairwise comparisons)")
print()

result_compound = es.compare(
    evaldata,
    factors="model",
    metric="llm_score",
    alignment={"llm_score": ar},
    simultaneous_ci=True,
    correction="auto",
    rng=np.random.default_rng(4),
)
result_compound.summary()
bundle_compound = result_compound._primary_bundle()
print(f"  -> resolved pairwise method:           {bundle_compound.resolved_method!r}")
print(f"  -> simultaneous_ci_method resolved to: "
      f"{bundle_compound.pairwise.simultaneous_ci_method!r}")
print(f"  -> correction_method resolved to:      "
      f"{bundle_compound.pairwise.correction_method!r}")
print("     (these now default to the same constructions (\"sidak\"/\"boot\" for")
print("      CIs, \"shaffer\"/\"romano_wolf\" for p-values) the non-PPI path uses,")
print("      matching evalstats' FWER decision tree -- neither gets silently")
print("      stuck at a weaker fallback (Bonferroni-only / Shaffer-only).)")
print()

# ── 5. Side-by-side summary of point estimates and pairwise verdicts ────────

print("=" * 70)
print("STEP 6 — Point estimates: raw vs PPI-only vs true rate")
print("=" * 70)

labels = list(bundle_compound.benchmark.template_labels)
bundle_raw = result_raw._primary_bundle()
bundle_ppi_only = result_ppi_only._primary_bundle()

print(f"  {'Model':<10}  {'Raw LLM':>9}  {'PPI-only':>9}  {'Compound':>9}  {'True rate':>9}")
print("  " + "-" * 58)
for i, lbl in enumerate(labels):
    raw_est = bundle_raw.robustness.mean[i]
    ppi_est = bundle_ppi_only.robustness.mean[i]
    compound_est = bundle_compound.robustness.mean[i]
    true_r = TRUE_RATE.get(lbl, float("nan"))
    print(f"  {lbl:<10}  {raw_est:>8.3f}   {ppi_est:>8.3f}   {compound_est:>8.3f}   {true_r:>8.3f}")
print()

print("=" * 70)
print("STEP 7 — A note on what to expect")
print("=" * 70)
print("""
  PPI correction and FWER control both cost some detection power on their
  own -- and applied together, that cost compounds. Each layer independently
  spends some of the marginal estimate's "headroom" below the significance
  threshold (PPI by adding judge-noise-correction variance; FWER control by
  widening CIs / inflating p-values to hold across every comparison). At
  realistic label fractions (PPI is designed for sparse, expensive human
  labels -- often 10-20% of items, not the 40% used here for a clearer demo)
  and moderate judge agreement, don't be surprised if the compound path
  detects a real difference far less often than either layer would in
  isolation. This isn't a bug in evalstats -- it's the honest cost of
  controlling two different error sources at once with a finite labeled
  sample.

  "romano_wolf" (the default p-value correction at N>=30) does recover some
  of that cost back versus the older "shaffer"-only default -- validated
  across a grid of arm counts/N/label fractions to be at least as powerful
  as Shaffer's in every condition tested, never worse -- but it's a real,
  bounded improvement, not a fix: it typically recovers a modest fraction
  of the gap, not all of it. The most reliable way to recover the rest is
  still more (or better-agreeing) human labels, not a different correction
  method.
""")
