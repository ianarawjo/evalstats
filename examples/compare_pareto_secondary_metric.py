"""Uncertainty-aware Pareto-front analysis: compare(secondary=...).

A developer is choosing among 5 models and cares about both accuracy and
latency. A naive Pareto front on point estimates alone would call a
"dominates" verdict any time one model's mean beats another's on both axes
-- even when the underlying per-item data can't actually support that claim.
compare(secondary=...) instead jointly bootstraps both metrics (a shared
per-item resample, not two independent marginal bootstraps) so dominance
calls are only made when the data backs them up, and reports a calibrated
three-state verdict per model: "frontier", "dominated", or "ambiguous"
(point estimate looks dominated, but there isn't enough evidence to confirm
it -- the case a naive point-estimate-only Pareto front would silently
misreport as a clean frontier member).

Ground-truth shape used to generate the data below:
  gpt-4o          high accuracy, moderate latency
  gpt-4o-mini     lower accuracy, very low latency      -- a real tradeoff
  claude-sonnet   worse accuracy AND worse latency than gpt-4o -- dominated
  claude-haiku    accuracy/latency both slightly behind gpt-4o-mini, but
                  noisy enough at this N that it isn't a confident call
                  -- "ambiguous", not "dominated"
  llama-70b       mid accuracy, mid latency              -- a real tradeoff

Usage:
    python examples/compare_pareto_secondary_metric.py
"""

import numpy as np
import pandas as pd

import evalstats as es


rng = np.random.default_rng(7)

N_ITEMS = 120

ACCURACY = {
    "gpt-4o": 0.85,
    "gpt-4o-mini": 0.72,
    "claude-sonnet": 0.80,
    "claude-haiku": 0.705,
    "llama-70b": 0.78,
}
LATENCY_S = {  # lower is better
    "gpt-4o": 2.0,
    "gpt-4o-mini": 0.4,
    "claude-sonnet": 2.6,
    "claude-haiku": 0.42,
    "llama-70b": 1.1,
}

rows = []
for model, acc in ACCURACY.items():
    lat = LATENCY_S[model]
    for i in range(N_ITEMS):
        rows.append({
            "model": model,
            "item": f"q{i:03d}",
            "accuracy": float(np.clip(rng.normal(acc, 0.09), 0, 1)),
            "latency_s": float(np.clip(rng.normal(lat, 0.35), 0.05, None)),
        })
df = pd.DataFrame(rows)

evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})

result = es.compare(
    evaldata,
    factors="model",
    metric="accuracy",
    secondary={"latency_s": "min"},
    rng=np.random.default_rng(42),
    show_rank_probabilities=True,  # also print the continuous P(Pareto-optimal)
)

print("=" * 70)
print("result.summary() -- full comparison, including the Pareto Front block")
print("=" * 70)
result.summary(top_pairwise=4)

print("=" * 70)
print("Programmatic access")
print("=" * 70)
print("result.pareto_status:")
for label, status in result.pareto_status.items():
    print(f"  {label:<15s} {status.status:<10s} dominated_by={status.dominated_by} ambiguous_vs={status.ambiguous_vs}")
print()
print("result.pareto_frontier_probability (P(Pareto-optimal) per model):")
for label, p in result.pareto_frontier_probability.items():
    print(f"  {label:<15s} {p:.1%}")
print()
print("result.to_dict()['pareto']:")
import json
print(json.dumps(result.to_dict(show_rank_probabilities=True)["pareto"], indent=2))
