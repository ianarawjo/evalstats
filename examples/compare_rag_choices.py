"""Compare RAG pipeline choices with evalstats.compare().

Shows three ways to slice a synthetic RAG benchmark using custom factor names:

  1. Compare chunking strategies only   (flat dict, factors="chunker")
  2. Compare retrieval methods only     (flat dict, factors="retriever")
  3. Compare all 9 combinations at once (nested dict, factors=["chunker", "retriever"])

The dataset simulates binary relevance scores (1 = retrieved doc was relevant)
for 80 questions run through each pipeline variant.  Ground-truth effects:

  chunker effect  : fixed = 0.00, sliding = +0.08, semantic = +0.16
  retriever effect: bm25  = 0.00, dense   = +0.09, hybrid   = +0.15

Usage:
    python examples/compare_rag_choices.py
"""

import numpy as np

import evalstats as es


rng = np.random.default_rng(42)

CHUNKERS   = ["fixed", "sliding", "semantic"]
RETRIEVERS = ["bm25",  "dense",   "hybrid"]

CHUNKER_EFFECT   = {"fixed": 0.00, "sliding": 0.08, "semantic": 0.16}
RETRIEVER_EFFECT = {"bm25":  0.00, "dense":   0.09, "hybrid":   0.15}

N = 80   # questions per pipeline cell
BASE_P = 0.52


def _scores(chunker: str, retriever: str) -> list:
    p = min(BASE_P + CHUNKER_EFFECT[chunker] + RETRIEVER_EFFECT[retriever], 0.97)
    return rng.binomial(1, p, N).astype(float).tolist()


# Pre-generate all 3×3 scores so every analysis below uses identical data.
all_scores = {c: {r: _scores(c, r) for r in RETRIEVERS} for c in CHUNKERS}


# ---------------------------------------------------------------------------
# 1. Compare chunking strategies (collapse across retriever)
# ---------------------------------------------------------------------------

chunker_scores = {
    c: [s for r in RETRIEVERS for s in all_scores[c][r]]
    for c in CHUNKERS
}

print("=" * 60)
print("1. Chunking strategy comparison")
print("=" * 60)

chunker_result = es.compare(
    chunker_scores,
    factors="chunker",
    statistic="mean",
    n_bootstrap=2_000,
    rng=np.random.default_rng(1),
)

chunker_result.summary()
print()


# ---------------------------------------------------------------------------
# 2. Compare retrieval methods (collapse across chunker)
# ---------------------------------------------------------------------------

retriever_scores = {
    r: [s for c in CHUNKERS for s in all_scores[c][r]]
    for r in RETRIEVERS
}

print("=" * 60)
print("2. Retrieval method comparison")
print("=" * 60)

retriever_result = es.compare(
    retriever_scores,
    factors="retriever",
    statistic="mean",
    n_bootstrap=2_000,
    rng=np.random.default_rng(2),
)

retriever_result.summary()
print()


# ---------------------------------------------------------------------------
# 3. Full factorial: all 9 chunker × retriever combinations
# ---------------------------------------------------------------------------

print("=" * 60)
print("3. Factorial: chunker × retriever (9 pipeline cells)")
print("=" * 60)

factorial_result = es.compare(
    all_scores,
    factors=["chunker", "retriever"],
    statistic="mean",
    n_bootstrap=2_000,
    rng=np.random.default_rng(3),
)

factorial_result.summary()
