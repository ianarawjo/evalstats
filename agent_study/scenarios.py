"""Generates the v0 benchmark's three ground-truth-bearing datasets.

Reuses ``simulations.harness.scenarios.synthetic.build_pair_sources`` (the
same data-generating process the simulation harness uses to calibrate CI
methods) instead of writing a new synthetic-data generator. Each phenomenon
below is one ``CIPairSource`` instance drawn at a fixed seed and sample size.

Run directly to sanity-check the three datasets before spending any API
calls on agent runs:

    python -m agent_study.scenarios
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from simulations.harness.scenarios.synthetic import build_pair_sources

# One symmetric, unimodal continuous shape (scores squeezed into [0, 1] by a
# logit-normal transform) present in the "standard" suite tier -- reads
# naturally as an eval score (e.g. an LLM judge or graded-task score).
SHAPE_LABEL = "cont-logit-normal"
ICC = 0.25

PHENOMENA = {
    "true_null": dict(cohens_d=0.0, n=60, seed=101),
    "clear_improvement": dict(cohens_d=0.8, n=60, seed=102),
    "borderline_improvement": dict(cohens_d=0.2, n=20, seed=103),
}

# Below this Monte-Carlo-estimated power, a well-calibrated agent is expected
# to often (correctly) fail to reach significance -- so "inconclusive" is
# accepted as a correct decision alongside the direction-based one, instead
# of only accepting the single answer that matches the true effect's sign.
POWER_THRESHOLD = 0.8


@dataclass
class ScenarioInstance:
    name: str
    eval_type: str
    n: int
    true_diff: float
    is_null: bool
    cohens_d: float
    power: float
    data: pd.DataFrame  # columns: item_id, prompt_a_score, prompt_b_score


def _find_source(cohens_d: float):
    sources = build_pair_sources(
        suite="standard", icc_values=(ICC,), cohens_d_values=(0.2, 0.8), include_null=True,
    )
    matches = [
        s for s in sources
        if s.eval_type == "continuous" and SHAPE_LABEL in s.label and s.icc == ICC and s.cohens_d == cohens_d
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one source for cohens_d={cohens_d}, got {len(matches)}: "
            f"{[m.label for m in matches]}"
        )
    return matches[0]


def _estimate_power(source, n: int, *, seed: int = 999, n_mc: int = 1000, alpha: float = 0.05) -> float:
    """Monte Carlo power: fraction of size-n paired samples where a paired
    t-test on (a - b) rejects the null in the true effect's direction.
    Simulation-based (rather than a closed-form formula) so it works for
    every shape in the catalog, including the non-parametric/custom ones."""
    if source.is_null:
        return float(alpha)  # power isn't meaningful under a true null; report the false-positive rate instead.
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_mc):
        a, b = source.generate_pair(rng, n, 1)
        diffs = a[:, 0] - b[:, 0]
        if diffs.std(ddof=1) == 0:
            continue
        t_stat, p = stats.ttest_1samp(diffs, 0.0)
        if p < alpha and np.sign(diffs.mean()) == np.sign(source.true_diff):
            rejections += 1
    return rejections / n_mc


def build_scenario(name: str) -> ScenarioInstance:
    spec = PHENOMENA[name]
    source = _find_source(spec["cohens_d"])
    rng = np.random.default_rng(spec["seed"])
    a, b = source.generate_pair(rng, spec["n"], 1)  # each (n, 1)
    data = pd.DataFrame({
        "item_id": np.arange(spec["n"]),
        "prompt_a_score": a[:, 0],
        "prompt_b_score": b[:, 0],
    })
    power = _estimate_power(source, spec["n"])
    return ScenarioInstance(
        name=name, eval_type=source.eval_type, n=spec["n"], true_diff=source.true_diff,
        is_null=source.is_null, cohens_d=source.cohens_d, power=power, data=data,
    )


def write_scenario(instance: ScenarioInstance, out_dir: Path) -> None:
    """Writes data.csv (agent-visible) and ground_truth.json (harness-only,
    must never be copied into an agent's workspace directory)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    instance.data.to_csv(out_dir / "data.csv", index=False)
    # true_diff is mean(prompt_a) - mean(prompt_b) (see _estimate_true_pair_diff
    # in simulations/harness/scenarios/synthetic.py) -- negative means B is
    # better, so a negative true_diff is the "deploy candidate B" case.
    directional_decision = "deploy" if instance.true_diff < 0 else "do_not_deploy"
    if instance.is_null:
        # No true effect: declining to deploy and honestly saying the
        # evidence doesn't support a change are both correct.
        correct_decisions = ["do_not_deploy", "inconclusive"]
    elif instance.power >= POWER_THRESHOLD:
        # Well-powered: a properly analyzed sample should reach significance
        # in the true direction, so only the directional answer counts.
        correct_decisions = [directional_decision]
    else:
        # Underpowered: a well-calibrated agent will often correctly fail to
        # reach significance, so honestly flagging that is not a mistake --
        # only confidently claiming the wrong direction is.
        correct_decisions = [directional_decision, "inconclusive"]
    ground_truth = {
        "scenario": instance.name,
        "eval_type": instance.eval_type,
        "n": instance.n,
        "true_diff": instance.true_diff,
        "is_null": instance.is_null,
        "cohens_d": instance.cohens_d,
        "power": instance.power,
        "directional_decision": directional_decision,
        "correct_decisions": correct_decisions,
        "valid_decisions": ["deploy", "do_not_deploy", "inconclusive"],
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))


if __name__ == "__main__":
    scratch = Path(__file__).parent / "runs" / "_scenario_dryrun"
    for name in PHENOMENA:
        instance = build_scenario(name)
        write_scenario(instance, scratch / name)
        print(f"{name}: n={instance.n} true_diff={instance.true_diff:.4f} "
              f"is_null={instance.is_null} cohens_d={instance.cohens_d} power={instance.power:.2f} "
              f"sample_mean_diff={(instance.data.prompt_b_score - instance.data.prompt_a_score).mean():.4f}")
    print(f"\nWrote dry-run datasets to {scratch}")
