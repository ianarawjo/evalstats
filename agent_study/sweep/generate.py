"""Generalized k-arm data + ground truth generator for the full sweep.

Calls simulations.harness.scenarios.synthetic.sample_group_truth directly
(bypassing build_pair_sources/build_multiarm_sources, each specialized to
one k) so a single code path covers k in {2, 5, 10, 20} uniformly across
every data type and DGP regime -- unifying what scenarios.py (k=2) and
scenarios_model_benchmark.py (k=8) each did separately for the two earlier,
smaller families.

Decision schema is unified too: rather than the old prompt_ab family's
deploy/do_not_deploy framing (which doesn't generalize past k=2), every item
in this sweep uses entity-name-or-"inconclusive" as its answer space
(prompt_1..prompt_k or model_1..model_k), matching what the model_benchmark
family already did -- k=2 just becomes the k=2 special case of one grid
instead of a differently-shaped problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from agent_study.sweep.dgp import get_dgp_config
from simulations.harness.scenarios.synthetic import group_total_std, sample_group_truth

ICC = 0.25  # reliability, same fixed value the two earlier families used
EFFECT_SIZES = {"true_null": 0.0, "borderline": 0.2, "clear_winner": 0.8}
POWER_THRESHOLD = 0.8  # same convention as scenarios.py / scenarios_model_benchmark.py
POWER_N_MC = 300  # Monte Carlo reps for the power estimate (per cell, not per rep)


def make_entity_labels(task_type: str, k: int) -> list[str]:
    prefix = "prompt" if task_type == "prompts" else "model"
    return [f"{prefix}_{i + 1}" for i in range(k)]


@dataclass
class CellGroundTruth:
    """Ground truth for one (task_type, k, data_type, ground_truth_regime, n,
    dgp_regime) cell -- shared by all 5 reps of that cell, since power/
    correct answers are a property of the design, not of one random draw."""

    entity_labels: list[str]
    cohens_d: float
    is_null: bool
    delta: float
    power: float
    true_best: str | None
    correct_decisions: list[str]
    valid_decisions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "entity_labels": self.entity_labels,
            "cohens_d": self.cohens_d,
            "is_null": self.is_null,
            "power": round(self.power, 4),
            "true_best": self.true_best,
            "correct_decisions": self.correct_decisions,
            "correct_clear_best": self.correct_decisions,  # unified answer space -- see module docstring
            "valid_decisions": self.valid_decisions,
            "valid_clear_best": self.valid_decisions,
        }


def _estimate_power(
    shape, heteroscedastic: bool, k: int, n: int, delta: float,
    *, seed: int = 999, n_mc: int = POWER_N_MC, alpha: float = 0.05,
) -> float:
    """Monte Carlo power: fraction of size-n, k-arm draws where entity_1 has
    the highest sample mean AND is significantly greater than every other
    entity via one-sided pairwise t-tests, Holm-corrected across the k-1
    comparisons (a no-op correction at k=2, so this is one function across
    the whole k sweep). For delta=0 this isn't the general family-wise error
    rate, it's the smaller probability that entity_1 SPECIFICALLY both wins
    by chance and clears every corrected test -- see
    scenarios_model_benchmark.py's _estimate_power for the same reasoning;
    it's fine here since power is only ever compared against
    POWER_THRESHOLD, and a near-0 value clears that bar the same way a
    near-alpha value would."""
    if delta == 0.0:
        return float(alpha)
    rng = np.random.default_rng(seed)
    effects = np.zeros(k)
    effects[0] = delta
    wins = 0
    for _ in range(n_mc):
        scores = sample_group_truth(shape, n, 1, k, ICC, rng, effects=effects, heteroscedastic=heteroscedastic)[:, :, 0]
        if scores[0].std(ddof=1) == 0:
            continue
        if not all(scores[0].mean() > scores[j].mean() for j in range(1, k)):
            continue
        p_values = [stats.ttest_ind(scores[0], scores[j], alternative="greater").pvalue for j in range(1, k)]
        _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method="holm")
        if all(p < alpha for p in p_corrected):
            wins += 1
    return wins / n_mc


def compute_cell_ground_truth(
    task_type: str, k: int, data_type: str, ground_truth_regime: str, n: int, dgp_regime: str,
) -> CellGroundTruth:
    cfg = get_dgp_config(data_type, dgp_regime)
    cohens_d = EFFECT_SIZES[ground_truth_regime]
    is_null = cohens_d == 0.0
    delta = cohens_d * group_total_std(cfg.shape, ICC)
    power = _estimate_power(cfg.shape, cfg.heteroscedastic, k, n, delta)
    labels = make_entity_labels(task_type, k)
    true_best = None if is_null else labels[0]

    if is_null:
        correct_decisions = ["inconclusive"]
    elif power >= POWER_THRESHOLD:
        correct_decisions = [true_best]
    else:
        correct_decisions = [true_best, "inconclusive"]

    return CellGroundTruth(
        entity_labels=labels, cohens_d=cohens_d, is_null=is_null, delta=delta, power=power,
        true_best=true_best, correct_decisions=correct_decisions, valid_decisions=labels + ["inconclusive"],
    )


def generate_rep_data(
    task_type: str, k: int, data_type: str, n: int, dgp_regime: str, cohens_d: float, seed: int,
) -> pd.DataFrame:
    """One actual random draw (one rep) for a cell -- long format, columns
    item_id/entity/score, consistent across every task_type/data_type/k so
    a single prompt template and a single scorer both work unchanged."""
    cfg = get_dgp_config(data_type, dgp_regime)
    delta = cohens_d * group_total_std(cfg.shape, ICC)
    labels = make_entity_labels(task_type, k)
    effects = np.zeros(k)
    effects[0] = delta
    rng = np.random.default_rng(seed)
    scores = sample_group_truth(cfg.shape, n, 1, k, ICC, rng, effects=effects, heteroscedastic=cfg.heteroscedastic)[:, :, 0]

    return pd.DataFrame({
        "item_id": np.tile(np.arange(n), k),
        "entity": np.repeat(labels, n),
        "score": scores.reshape(-1),
    })
