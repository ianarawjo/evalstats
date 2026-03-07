"""Factorial LMM demo: RAG pipeline with chunker × retrieval_method.

Simulates a benchmark where 30 questions are each evaluated with every
combination of two pipeline factors:

  * chunker   : fixed_512 | sliding_256 | semantic
  * retrieval : bm25 | dense | hybrid

This gives 3 × 3 = 9 treatment cells × 30 questions = 270 observations.
Each row in the DataFrame represents one pipeline output and records which
chunker and retrieval method produced it — the typical shape of a post-hoc
tagged RAG experiment.

Ground-truth additive effects used to generate the data:

  chunker effect   : fixed_512 = 0.00, sliding_256 = +0.03, semantic = +0.08
  retrieval effect : bm25 = 0.00, dense = +0.06, hybrid = +0.10

Usage::

    python examples/factorial_rag_demo.py
"""

import numpy as np
import pandas as pd

import promptstats as ps


# ---------------------------------------------------------------------------
# Simulate data
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)

CHUNKERS   = ["fixed_512", "sliding_256", "semantic"]
RETRIEVALS = ["bm25", "dense", "hybrid"]

CHUNKER_EFFECT   = {"fixed_512": 0.00, "sliding_256": 0.03, "semantic": 0.08}
RETRIEVAL_EFFECT = {"bm25": 0.00, "dense": 0.06, "hybrid": 0.10}

N_QUESTIONS = 30
BASE_SCORE  = 0.65
SIGMA_INPUT = 0.10   # between-question variance (drives ICC)
SIGMA_RESID = 0.04   # residual within-cell noise

question_intercepts = rng.normal(0, SIGMA_INPUT, N_QUESTIONS)

rows = []
for q_idx in range(N_QUESTIONS):
    q_id = f"q{q_idx + 1:03d}"
    for chunker in CHUNKERS:
        for retrieval in RETRIEVALS:
            score = (
                BASE_SCORE
                + question_intercepts[q_idx]
                + CHUNKER_EFFECT[chunker]
                + RETRIEVAL_EFFECT[retrieval]
                + rng.normal(0, SIGMA_RESID)
            )
            rows.append({
                "input_id":  q_id,
                "chunker":   chunker,
                "retrieval": retrieval,
                "score":     float(np.clip(score, 0.0, 1.0)),
            })

data = pd.DataFrame(rows)

print(
    f"Dataset: {len(data)} rows, "
    f"{data['input_id'].nunique()} questions, "
    f"{data['chunker'].nunique()} chunkers × "
    f"{data['retrieval'].nunique()} retrieval methods\n"
)

# ---------------------------------------------------------------------------
# Run the factorial analysis
# ---------------------------------------------------------------------------

bundle = ps.analyze_factorial(
    data,
    factors=["chunker", "retrieval"],
    random_effect="input_id",
    score_col="score",
    rng=np.random.default_rng(0),
)

fi = bundle.factorial_lmm_info

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

print("=" * 60)
print("Factorial LMM results")
print("=" * 60)
print(f"Formula : {fi.formula}")
print(f"ICC     : {fi.icc:.3f}  (fraction of variance due to question identity)")
print(f"σ_input : {fi.sigma_input:.4f}  (between-question SD)")
print(f"σ_resid : {fi.sigma_resid:.4f}  (within-cell residual SD)")
print(f"n_obs   : {fi.n_obs}")
print()

print("Factor / interaction Wald tests:")
print(fi.factor_tests.to_string(index=False))
print()

print("Estimated marginal means — chunker:")
print(fi.marginal_means["chunker"].to_string(index=False))
print()

print("Estimated marginal means — retrieval:")
print(fi.marginal_means["retrieval"].to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Pairwise comparisons (top 10 by p-value)
# ---------------------------------------------------------------------------

pw_rows = [
    {
        "cell_A":   a,
        "cell_B":   b,
        "diff":     f"{res.point_diff:+.4f}",
        "95% CI":   f"[{res.ci_low:.4f}, {res.ci_high:.4f}]",
        "p_adj":    f"{res.p_value:.4g}",
    }
    for (a, b), res in bundle.pairwise.results.items()
]
pw_df = (
    pd.DataFrame(pw_rows)
    .assign(_p=lambda d: d["p_adj"].astype(float))
    .sort_values("_p")
    .drop(columns="_p")
    .head(10)
)
print("Top pairwise comparisons (Holm-corrected, sorted by p-value):")
print(pw_df.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Robustness summary
# ---------------------------------------------------------------------------

rob = bundle.robustness
rob_df = pd.DataFrame({
    "cell":   bundle.benchmark.template_labels,
    "mean":   [f"{v:.4f}" for v in rob.mean],
    "std":    [f"{v:.4f}" for v in rob.std],
})
print("Per-cell robustness (mean ± std over questions):")
print(rob_df.to_string(index=False))
