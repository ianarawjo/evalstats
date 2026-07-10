"""Generates the "8-model benchmark" scenario family's three ground-truth
datasets -- the second family, alongside scenarios.py's paired-prompt
family. See the plan discussion in agent_study/README.md for why: a 2-arm
comparison may be too simple a regime for evalstats to visibly help in;
picking a winner among 8 models requires correcting for ~28 pairwise
comparisons' worth of multiple-testing inflation, which a baseline agent is
less likely to do unprompted.

Reuses simulations.harness.scenarios.synthetic.build_multiarm_sources (the
k-arm analogue of build_pair_sources used by scenarios.py) instead of a new
data-generating process.

Run directly to sanity-check the three datasets before spending any API
calls on agent runs:

    python -m agent_study.scenarios_model_benchmark
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from simulations.harness.scenarios.synthetic import (
    SHAPES_BY_EVAL_TYPE,
    build_multiarm_sources,
    group_total_std,
)

# Same shape/icc as scenarios.py's prompt_ab family, so the two families'
# data are visually/statistically comparable.
SHAPE_LABEL = "cont-logit-normal"
ICC = 0.25
K = 8
MODEL_LABELS = [f"model_{i + 1}" for i in range(K)]  # index 0 -> model_1, carries the shift

PHENOMENA = {
    "true_null": dict(cohens_d=0.0, n=60, seed=201),
    "clear_best": dict(cohens_d=0.8, n=60, seed=202),
    "borderline_best": dict(cohens_d=0.2, n=20, seed=203),
}

POWER_THRESHOLD = 0.8  # same convention as scenarios.py


@dataclass
class ModelBenchmarkInstance:
    name: str
    eval_type: str
    n: int
    k: int
    is_null: bool
    cohens_d: float
    power: float
    true_best_model: str | None  # None for true_null (no defensible winner)
    data: pd.DataFrame  # columns: item_id, model, score (long format)


def _find_source():
    sources = build_multiarm_sources(suite="standard", icc=ICC, eval_types=["continuous"])
    matches = [s for s in sources if SHAPE_LABEL in s.label]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one source for {SHAPE_LABEL!r}, got {len(matches)}")
    return matches[0]


def _delta_for(d: float) -> float:
    shape = next(s for s in SHAPES_BY_EVAL_TYPE["continuous"] if s.label == SHAPE_LABEL)
    return d * group_total_std(shape, ICC)


def _estimate_power(source, delta: float, n: int, *, seed: int = 999, n_mc: int = 500, alpha: float = 0.05) -> float:
    """Monte Carlo power: fraction of size-n, k=8 draws where model_1 has
    the highest sample mean AND is significantly greater than every other
    model via one-sided pairwise t-tests, Holm-corrected across the 7
    comparisons. For delta=0 this isn't the general family-wise error rate
    (probability ANY arm falsely looks like a confirmed winner) -- it's the
    (much smaller) probability that model_1 SPECIFICALLY both has the
    highest of 8 equal-mean draws by chance AND clears every corrected
    pairwise test, which is why it comes out near 0 rather than near alpha.
    That's fine here: it's only ever compared against POWER_THRESHOLD to
    decide whether "inconclusive" should be accepted alongside the
    directional answer, and a near-0 value clears that bar the same way a
    near-alpha value would. n_mc=500 (vs. the pair family's 1000) since each
    draw here costs 7 t-tests instead of 1."""
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_mc):
        scores = source.generate_scores(rng, n, 1, K, delta)[:, :, 0]  # (k, n)
        if scores[0].std(ddof=1) == 0:
            continue
        if not all(scores[0].mean() > scores[j].mean() for j in range(1, K)):
            continue
        p_values = [stats.ttest_ind(scores[0], scores[j], alternative="greater").pvalue for j in range(1, K)]
        _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method="holm")
        if all(p < alpha for p in p_corrected):
            wins += 1
    return wins / n_mc


def build_scenario(name: str) -> ModelBenchmarkInstance:
    spec = PHENOMENA[name]
    source = _find_source()
    delta = _delta_for(spec["cohens_d"])
    is_null = spec["cohens_d"] == 0.0

    rng = np.random.default_rng(spec["seed"])
    scores = source.generate_scores(rng, spec["n"], 1, K, delta)[:, :, 0]  # (k, n)
    power = _estimate_power(source, delta, spec["n"])

    data = pd.DataFrame({
        "item_id": np.tile(np.arange(spec["n"]), K),
        "model": np.repeat(MODEL_LABELS, spec["n"]),
        "score": scores.reshape(-1),
    })
    return ModelBenchmarkInstance(
        name=name, eval_type=source.eval_type, n=spec["n"], k=K, is_null=is_null,
        cohens_d=spec["cohens_d"], power=power,
        true_best_model=None if is_null else MODEL_LABELS[0], data=data,
    )


def write_scenario(instance: ModelBenchmarkInstance, out_dir: Path) -> None:
    """Writes data.csv (agent-visible) and ground_truth.json (harness-only,
    must never be copied into an agent's workspace directory)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    instance.data.to_csv(out_dir / "data.csv", index=False)

    if instance.is_null:
        # No incumbent in this family's framing -- under a true null there's
        # no defensible named winner, only "inconclusive".
        correct_decisions = ["inconclusive"]
    elif instance.power >= POWER_THRESHOLD:
        correct_decisions = [instance.true_best_model]
    else:
        correct_decisions = [instance.true_best_model, "inconclusive"]

    ground_truth = {
        "scenario": instance.name,
        "eval_type": instance.eval_type,
        "n": instance.n,
        "k": instance.k,
        "is_null": instance.is_null,
        "cohens_d": instance.cohens_d,
        "power": instance.power,
        "true_best_model": instance.true_best_model,
        "correct_decisions": correct_decisions,
        "valid_decisions": MODEL_LABELS + ["inconclusive"],
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))


if __name__ == "__main__":
    scratch = Path(__file__).parent / "runs" / "_scenario_dryrun_model_benchmark"
    for name in PHENOMENA:
        instance = build_scenario(name)
        write_scenario(instance, scratch / name)
        means = instance.data.groupby("model")["score"].mean().reindex(MODEL_LABELS)
        print(f"{name}: n={instance.n} k={instance.k} is_null={instance.is_null} "
              f"cohens_d={instance.cohens_d} power={instance.power:.3f} "
              f"true_best={instance.true_best_model}")
        print(f"  sample means: {dict(means.round(4))}")
    print(f"\nWrote dry-run datasets to {scratch}")
