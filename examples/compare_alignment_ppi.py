"""LLM judge alignment validation and PPI correction example.

Simulates a common real-world scenario:
  - An LLM judge has scored 200 model outputs per model (cheap, scalable).
  - A human annotator has labelled 40 of those outputs (expensive, sparse).
  - The LLM judge has a systematic bias: a high false-positive rate for
    model_A, causing it to dramatically overestimate model_A's win rate.

Demonstrates:
  1. validate_alignment() — quantify LLM-vs-human agreement on the gold set.
  2. compare(..., alignment=...) — PPI-corrected model comparison that uses
     the human labels to debias the LLM-only estimates.
  3. es.ppi.correct() — apply PPI to a custom estimator (win-rate advantage
     of model_A over model_B on the full unlabeled dataset).

Usage:
    python examples/compare_alignment_ppi.py
"""

import numpy as np
import pandas as pd

import evalstats as es

# ── Simulation parameters ─────────────────────────────────────────────────────

rng = np.random.default_rng(42)

N_ITEMS   = 200   # items per model
N_LABELED = 40    # items per model with human annotation (the gold set)

# True human win rates for each model (what an unbiased judge would measure)
TRUE_RATE = {"model_A": 0.68, "model_B": 0.54, "model_C": 0.40}

# LLM judge parameters:
#   True Positive Rate (LLM says "pass" when item truly passes)
#   False Positive Rate (LLM says "pass" when item truly fails)
# model_A has a high FPR — the LLM is biased toward calling it correct.
LLM_TPR = {"model_A": 0.92, "model_B": 0.87, "model_C": 0.87}
LLM_FPR = {"model_A": 0.62, "model_B": 0.14, "model_C": 0.10}

# Implied LLM win rates (for reference)
# model_A: 0.92*0.68 + 0.62*0.32 ≈ 0.825
# model_B: 0.87*0.54 + 0.14*0.46 ≈ 0.534
# model_C: 0.87*0.40 + 0.10*0.60 ≈ 0.408

# ── Build the full dataset ────────────────────────────────────────────────────
# Each (model, item) pair has a latent true quality.
# The LLM and human both observe the same item, producing correlated scores.

rows = []
for model in TRUE_RATE:
    true_rate = TRUE_RATE[model]
    tpr       = LLM_TPR[model]
    fpr       = LLM_FPR[model]

    # Latent item quality: 1 = truly good, 0 = truly bad
    quality = rng.binomial(1, true_rate, N_ITEMS)

    # Human label: near-perfect rater (95% accuracy)
    human_accuracy = 0.95
    human_score = np.where(
        quality == 1,
        rng.binomial(1, human_accuracy,     N_ITEMS),
        rng.binomial(1, 1 - human_accuracy, N_ITEMS),
    ).astype(float)

    # LLM score: biased rater (TPR/FPR model)
    llm_score = np.where(
        quality == 1,
        rng.binomial(1, tpr, N_ITEMS),
        rng.binomial(1, fpr, N_ITEMS),
    ).astype(float)

    for item_id in range(N_ITEMS):
        rows.append({
            "model":      model,
            "item":       item_id,
            "llm_score":  llm_score[item_id],
            "human_score": float("nan"),  # filled in below for gold items
        })

    # Select N_LABELED items to receive human annotation
    gold_indices = rng.choice(N_ITEMS, size=N_LABELED, replace=False)
    for idx in gold_indices:
        row_loc = len(rows) - N_ITEMS + idx
        rows[row_loc]["human_score"] = human_score[idx]

df = pd.DataFrame(rows)

# ── Quick check: print expected vs actual LLM win rates ──────────────────────

print("Simulation check — LLM win rates (expected vs actual):")
for model in TRUE_RATE:
    m = df[df["model"] == model]
    llm_mean   = m["llm_score"].mean()
    human_mean = m["human_score"].dropna().mean()
    exp_llm    = LLM_TPR[model] * TRUE_RATE[model] + LLM_FPR[model] * (1 - TRUE_RATE[model])
    print(f"  {model}: true={TRUE_RATE[model]:.2f}  "
          f"llm_all={llm_mean:.3f} (expected≈{exp_llm:.3f})  "
          f"human_labeled={human_mean:.3f}")
print()

# ── Load into evalstats ───────────────────────────────────────────────────────

evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})

# ── 1. Validate alignment ─────────────────────────────────────────────────────

print("=" * 62)
print("STEP 1 — Validate LLM judge alignment")
print("=" * 62)

ar = es.validate_alignment(
    evaldata,
    llm_metric="llm_score",
    human_groundtruth="human_score",
)
ar.summary()

# ── 2a. Uncorrected comparison (LLM scores only) ──────────────────────────────

print("=" * 62)
print("STEP 2a — Uncorrected model comparison  (LLM judge only)")
print("=" * 62)

result_llm = es.compare(
    evaldata,
    factors="model",
    metric="llm_score",
    n_bootstrap=2_000,
    rng=np.random.default_rng(1),
)
result_llm.summary()

# ── 2b. PPI-corrected comparison ──────────────────────────────────────────────

print("=" * 62)
print("STEP 2b — PPI-corrected model comparison")
print("=" * 62)
print(f"  ({N_LABELED} human labels per model used to debias the LLM estimates)")
print()

result_ppi = es.compare(
    evaldata,
    factors="model",
    metric="llm_score",
    alignment={"llm_score": ar},
    n_mc=1_000,
    rng=np.random.default_rng(2),
)
result_ppi.summary()

# ── 2c. Side-by-side: LLM-only vs PPI vs true rates ─────────────────────────

print("=" * 62)
print("STEP 2c — Point estimates: LLM-only vs PPI-corrected vs true")
print("=" * 62)

bundle_llm = result_llm._primary_bundle()
bundle_ppi = result_ppi._primary_bundle()
labels     = list(bundle_llm.benchmark.template_labels)

print(f"  {'Model':<10}  {'LLM-only':>9}  {'PPI-corr':>9}  {'True rate':>9}")
print("  " + "-" * 46)
for i, lbl in enumerate(labels):
    llm_est = bundle_llm.robustness.mean[i]
    ppi_est = bundle_ppi.robustness.mean[i]
    true_r  = TRUE_RATE.get(lbl, float("nan"))
    print(f"  {lbl:<10}  {llm_est:>8.3f}   {ppi_est:>8.3f}   {true_r:>8.3f}")

vc = result_ppi.to_dict()["variance_components"]
print()
print(f"  PPI rectifiers (human_mean − llm_mean on the {N_LABELED}-item gold set):")
for entity, info in vc["entities"].items():
    sign = "+" if info["rectifier"] >= 0 else ""
    print(f"    {entity}: {sign}{info['rectifier']:.3f}  "
          f"(n_labeled={info['n_labeled']}, "
          f"llm_mean={info['llm_mean']:.3f}, "
          f"ppi_mean={info['ppi_mean']:.3f})")
print()

# ── 3. ppi.correct() with a custom estimator ─────────────────────────────────

print("=" * 62)
print("STEP 3 — es.ppi.correct() with a custom estimator")
print("=" * 62)
print("  Estimate: win-rate advantage of model_A over model_B")
print("  on the full LLM-scored dataset, corrected with gold labels.")
print()

def win_rate_diff(scores, models):
    return float(scores[models == "model_A"].mean() - scores[models == "model_B"].mean())

ab_mask_all = df["model"].isin(["model_A", "model_B"])
ab_mask_lab = ab_mask_all & df["human_score"].notna()

ppi_result = es.ppi.correct(
    estimator_func=win_rate_diff,
    Y_hat_unlab=df.loc[ab_mask_all, "llm_score"].values,
    Y_hat_lab  =df.loc[ab_mask_lab, "llm_score"].values,
    Y_lab      =df.loc[ab_mask_lab, "human_score"].values,
    X_unlab    =df.loc[ab_mask_all, "model"].values,
    X_lab      =df.loc[ab_mask_lab, "model"].values,
    alpha=0.05,
    n_boot=2_000,
    rng=np.random.default_rng(3),
)

ppi_result.summary()
true_diff = TRUE_RATE["model_A"] - TRUE_RATE["model_B"]
print()
print(f"  True advantage (A − B):    {true_diff:+.3f}")
