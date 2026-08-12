"""LLM judge alignment with human subjects data: between- and within-subjects designs.

Demonstrates how compare(..., alignment=...) handles two common HCI study designs:

  BETWEEN-SUBJECTS
    20 shared essay topics; a *different* participant writes each topic under
    each condition.  The blocking unit is the topic (not the person).  Each
    participant appears only once, so the correlation across conditions comes
    from topic-level difficulty, not from person-level consistency.

  WITHIN-SUBJECTS
    60 participants each complete tasks under *all three* conditions
    (counterbalanced).  The blocking unit is the participant.  evalstats
    detects the repeated structure via the item column and resamples
    *participants* (not individual essays) in the PPI bootstrap, preserving
    the within-person correlation that tightens pairwise CIs.

Both designs produce complete, balanced score matrices so the standard
bootstrap path works in both cases.  The within-subjects design typically
yields narrower CIs because participant-level individual differences are
accounted for by the paired resampling.

Key point: the human alignment labels are collected at the essay level (a
random sample of individual essays, not a random sample of topics or people).
The PPI labeled-set bootstrap always resamples at the observation level to
match this sampling scheme.

Usage:
    python examples/compare_alignment_subjects.py
"""

import numpy as np
import pandas as pd
import evalstats as es

# ── Simulation parameters ────────────────────────────────────────────────────

rng = np.random.default_rng(42)

N_TOPICS         = 60    # essay topics shared across conditions (between-subjects)
N_WS_PARTICIPANTS = 60   # participants in the within-subjects arm
N_LABELED        = 40    # essays that also receive a human expert score
CONDITIONS       = ["A", "B", "C"]

# True underlying quality rates per condition
TRUE_QUALITY = {"A": 0.70, "B": 0.55, "C": 0.42}

# LLM judge: condition A has a high false-positive rate (systematic bias)
LLM_TPR = {"A": 0.92, "B": 0.87, "C": 0.85}
LLM_FPR = {"A": 0.58, "B": 0.13, "C": 0.11}

HUMAN_ACCURACY = 0.95   # human expert near-perfect rater


def _essay(condition: str) -> tuple[float, int]:
    """Return (llm_score, latent_quality) for a single essay."""
    quality = int(rng.binomial(1, TRUE_QUALITY[condition]))
    tpr, fpr = LLM_TPR[condition], LLM_FPR[condition]
    llm = int(rng.binomial(1, tpr if quality else fpr))
    return float(llm), quality


def _human(quality: int) -> float:
    p = HUMAN_ACCURACY if quality else (1 - HUMAN_ACCURACY)
    return float(rng.binomial(1, p))


# ── BETWEEN-SUBJECTS dataset ─────────────────────────────────────────────────
# 20 topics × 3 conditions = 60 essays; a *different* participant writes
# each (topic, condition) cell.  Blocking unit = topic.

print("=" * 62)
print("Building BETWEEN-SUBJECTS dataset")
print(f"  {N_TOPICS} topics × {len(CONDITIONS)} conditions = "
      f"{N_TOPICS * len(CONDITIONS)} essays; different participant per cell (same topic scale as WS)")
print("=" * 62)

rows_bs = []
for topic_id in range(N_TOPICS):
    for cond in CONDITIONS:
        llm, quality = _essay(cond)
        rows_bs.append({
            "topic":     f"T{topic_id:02d}",
            "condition": cond,
            "llm_score": llm,
            "_quality":  quality,
        })

df_bs = pd.DataFrame(rows_bs)

# Randomly label a subset of individual essays (not topics).
human_bs = np.full(len(df_bs), np.nan)
for i in rng.choice(len(df_bs), size=N_LABELED, replace=False):
    human_bs[i] = _human(int(df_bs.loc[i, "_quality"]))
df_bs["human_score"] = human_bs
df_bs = df_bs.drop(columns="_quality")

print(f"  LLM win rates:  { {c: round(df_bs[df_bs.condition==c].llm_score.mean(),3) for c in CONDITIONS} }")
print(f"  True win rates: {TRUE_QUALITY}")
print(f"  Human-labeled essays: {int(df_bs['human_score'].notna().sum())}")
print()

# "condition" → "model" so evalstats builds ["model", "item"] duplicate key.
# Each (condition, topic) pair is unique — no sparse/NaN cells.
evaldata_bs = es.load_from(
    df_bs,
    col_map={"condition": "model", "topic": "item"},
)


# ── WITHIN-SUBJECTS dataset ──────────────────────────────────────────────────
# 60 participants × 3 conditions = 180 essays; each participant writes under
# all conditions (counterbalanced).  Blocking unit = participant.

print("=" * 62)
print("Building WITHIN-SUBJECTS dataset")
print(f"  {N_WS_PARTICIPANTS} participants × {len(CONDITIONS)} conditions = "
      f"{N_WS_PARTICIPANTS * len(CONDITIONS)} essays; same participant per row")
print("=" * 62)

rows_ws = []
for pid in range(N_WS_PARTICIPANTS):
    for cond in CONDITIONS:
        llm, quality = _essay(cond)
        rows_ws.append({
            "participant": f"P{pid:02d}",
            "condition":   cond,
            "llm_score":   llm,
            "_quality":    quality,
        })

df_ws = pd.DataFrame(rows_ws)

human_ws = np.full(len(df_ws), np.nan)
for i in rng.choice(len(df_ws), size=N_LABELED, replace=False):
    human_ws[i] = _human(int(df_ws.loc[i, "_quality"]))
df_ws["human_score"] = human_ws
df_ws = df_ws.drop(columns="_quality")

print(f"  LLM win rates:  { {c: round(df_ws[df_ws.condition==c].llm_score.mean(),3) for c in CONDITIONS} }")
print(f"  True win rates: {TRUE_QUALITY}")
print(f"  Human-labeled essays: {int(df_ws['human_score'].notna().sum())}")
print()

# Participant IDs repeat across conditions → evalstats detects within-subjects
# structure and switches to participant-level resampling in the PPI bootstrap.
evaldata_ws = es.load_from(
    df_ws,
    col_map={"condition": "model", "participant": "item"},
)


# ── Step 1: validate LLM judge alignment ─────────────────────────────────────

print("=" * 62)
print("STEP 1 — LLM judge alignment (between-subjects data)")
print("=" * 62)
ar_bs = es.judge_alignment(
    evaldata_bs,
    llm_metric="llm_score",
    human_groundtruth="human_score",
)
ar_bs.summary()

print("=" * 62)
print("STEP 1 — LLM judge alignment (within-subjects data)")
print("=" * 62)
ar_ws = es.judge_alignment(
    evaldata_ws,
    llm_metric="llm_score",
    human_groundtruth="human_score",
)
ar_ws.summary()


# ── Step 2: uncorrected vs PPI-corrected comparisons ─────────────────────────

print("=" * 62)
print("STEP 2a — Uncorrected comparison (between-subjects, LLM only)")
print("=" * 62)
result_bs_llm = es.compare(
    evaldata_bs,
    factors="model",
    metric="llm_score",
    n_bootstrap=2_000,
    rng=np.random.default_rng(1),
)
result_bs_llm.summary()

print("=" * 62)
print("STEP 2b — PPI-corrected comparison (between-subjects)")
print(f"  block = topic; bootstrap resamples topics (not participants)")
print("=" * 62)
result_bs_ppi = es.compare(
    evaldata_bs,
    factors="model",
    metric="llm_score",
    alignment={"llm_score": ar_bs},
    n_mc=1_000,
    rng=np.random.default_rng(2),
)
result_bs_ppi.summary()

print("=" * 62)
print("STEP 2c — Uncorrected comparison (within-subjects, LLM only)")
print("=" * 62)
result_ws_llm = es.compare(
    evaldata_ws,
    factors="model",
    metric="llm_score",
    n_bootstrap=2_000,
    rng=np.random.default_rng(3),
)
result_ws_llm.summary()

print("=" * 62)
print("STEP 2d — PPI-corrected comparison (within-subjects)")
print(f"  block = participant; bootstrap resamples participants")
print("=" * 62)
result_ws_ppi = es.compare(
    evaldata_ws,
    factors="model",
    metric="llm_score",
    alignment={"llm_score": ar_ws},
    n_mc=1_000,
    rng=np.random.default_rng(4),
)
result_ws_ppi.summary()


# ── Step 3: side-by-side summary ────────────────────────────────────────────

print("=" * 62)
print("STEP 3 — Point estimates and pairwise CI widths")
print("=" * 62)

bundle_bs_llm = result_bs_llm._primary_bundle()
bundle_bs_ppi = result_bs_ppi._primary_bundle()
bundle_ws_llm = result_ws_llm._primary_bundle()
bundle_ws_ppi = result_ws_ppi._primary_bundle()

labels = list(bundle_bs_llm.benchmark.template_labels)
print(f"\n  Condition means (true / LLM-only / PPI-corrected):")
print(f"  {'Cond':<6}  {'true':>6}  {'BS-LLM':>8}  {'BS-PPI':>8}  {'WS-LLM':>8}  {'WS-PPI':>8}")
print("  " + "-" * 54)
for i, lbl in enumerate(labels):
    print(f"  {lbl:<6}  {TRUE_QUALITY[lbl]:>6.3f}  "
          f"{bundle_bs_llm.robustness.mean[i]:>8.3f}  "
          f"{bundle_bs_ppi.robustness.mean[i]:>8.3f}  "
          f"{bundle_ws_llm.robustness.mean[i]:>8.3f}  "
          f"{bundle_ws_ppi.robustness.mean[i]:>8.3f}")

pair_keys = list(bundle_bs_ppi.pairwise.results.keys())
print(f"\n  Pairwise simultaneous CI widths (narrower = more precise):")
print(f"  {'Pair':<8}  {'BS-LLM':>8}  {'BS-PPI':>8}  {'WS-LLM':>8}  {'WS-PPI':>8}")
print("  " + "-" * 48)
for key in pair_keys:
    def w(bundle):
        pr = bundle.pairwise.results[key]
        return pr.ci_high - pr.ci_low
    print(f"  {key[0]}−{key[1]:<6}  {w(bundle_bs_llm):>8.3f}  {w(bundle_bs_ppi):>8.3f}  "
          f"  {w(bundle_ws_llm):>8.3f}  {w(bundle_ws_ppi):>8.3f}")

vc_bs = result_bs_ppi.to_dict().get("variance_components", {})
vc_ws = result_ws_ppi.to_dict().get("variance_components", {})
print(f"\n  PPI diagnostics:")
print(f"  Between-subjects — design: {vc_bs.get('design')}, "
      f"n_all={vc_bs.get('n_all')} topics, n_lab={vc_bs.get('n_lab')} essays")
print(f"  Within-subjects  — design: {vc_ws.get('design')}, "
      f"n_all={vc_ws.get('n_all')} participants, n_lab={vc_ws.get('n_lab')} essays")
print()
print("Notes:")
print("  - Both designs are detected as 'within_subjects' because topics/participants")
print("    repeat across conditions — evalstats resamples items in both cases.")
print("  - WS-LLM CIs are slightly narrower than BS-LLM CIs (same direction as theory).")
print("  - PPI CIs are dominated by the labeled-set uncertainty (40 / 180 = 22% labeled);")
print("    the BS/WS gap is small relative to that noise.")
print("  - In real data with strong within-person ability effects, the WS advantage")
print("    would be more pronounced; this simulation has no within-unit random effect.")
