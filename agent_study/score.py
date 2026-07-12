"""Scores one run: does the agent's recommendation.json decision match one of
the analytically-known correct_decisions in ground_truth.json?

Family-agnostic: correct_decisions and valid_decisions both live in
ground_truth.json (written by each family's scenarios_*.py), rather than
being hardcoded here, so this same scorer works for prompt_ab's
deploy/do_not_deploy/inconclusive and model_benchmark's model_1..model_8/
inconclusive without modification.

correct_decisions is a list, not a single label: under low statistical power
(see each family's POWER_THRESHOLD), a well-calibrated "inconclusive" is
accepted alongside the direction/model-based answer, so honest uncertainty
isn't scored as a mistake.

v0 only scores decision correctness. Effect-size accuracy and confidence
calibration are deferred past v0.
"""

from __future__ import annotations

import json
from pathlib import Path


def score_run(workspace_dir: Path, ground_truth_path: Path) -> dict:
    ground_truth = json.loads(ground_truth_path.read_text())
    valid_decisions = ground_truth["valid_decisions"]

    rec_path = workspace_dir / "recommendation.json"
    if not rec_path.exists():
        return {
            **_base_row(ground_truth),
            "decision": None,
            "certainty": None,
            "correct": False,
            "failure": "recommendation.json missing",
        }

    try:
        rec = json.loads(rec_path.read_text())
        decision = rec.get("decision")
        certainty = rec.get("certainty")
    except (json.JSONDecodeError, AttributeError) as e:
        return {
            **_base_row(ground_truth),
            "decision": None,
            "certainty": None,
            "correct": False,
            "failure": f"recommendation.json unparseable: {e}",
        }

    if decision not in valid_decisions:
        return {
            **_base_row(ground_truth),
            "decision": decision,
            "certainty": certainty,
            "correct": False,
            "failure": f"decision not one of {valid_decisions}: {decision!r}",
        }

    return {
        **_base_row(ground_truth),
        "decision": decision,
        "certainty": certainty,
        "correct": decision in ground_truth["correct_decisions"],
        "failure": None,
    }


def _base_row(ground_truth: dict) -> dict:
    """Passes through every ground_truth field generically (both families
    share scenario/is_null/power/correct_decisions; family-specific fields
    like true_diff vs. true_best_model just pass through too), formatting
    any list-valued fields as '|'-joined strings for a flat CSV row."""
    row = {}
    for key, value in ground_truth.items():
        if key == "valid_decisions":
            continue  # not interesting per-row; same for every scenario in a family
        row[key] = "|".join(str(v) for v in value) if isinstance(value, list) else value
    return row
