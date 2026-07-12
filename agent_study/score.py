"""Scores one run: do the agent's recommendation.json fields match one of
the analytically-known correct labels in ground_truth.json?

Family-agnostic: correct_* and valid_* fields live in ground_truth.json
(written by each family's scenarios_*.py), rather than being hardcoded here,
so this same scorer works for prompt_ab and model_benchmark without
modification.

correct_decisions is a list, not a single label: under low statistical power
(see each family's POWER_THRESHOLD), a well-calibrated "inconclusive" is
accepted alongside the direction/model-based answer, so honest uncertainty
isn't scored as a mistake.

v0 scores decision correctness and clear_best correctness. Effect-size
accuracy and confidence calibration are deferred past v0.
"""

from __future__ import annotations

import json
from pathlib import Path


def score_run(workspace_dir: Path, ground_truth_path: Path) -> dict:
    ground_truth = json.loads(ground_truth_path.read_text())
    valid_decisions = ground_truth["valid_decisions"]
    valid_clear_best = ground_truth.get("valid_clear_best", _infer_valid_clear_best(ground_truth))
    correct_clear_best = ground_truth.get("correct_clear_best", _infer_correct_clear_best(ground_truth))

    rec_path = workspace_dir / "recommendation.json"
    if not rec_path.exists():
        return {
            **_base_row(ground_truth),
            "clear_best": None,
            "certainty": None,
            "decision": None,
            "correct": False,
            "failure": "recommendation.json missing",
        }

    try:
        rec = json.loads(rec_path.read_text())
        clear_best = rec.get("clear_best")
        certainty = rec.get("certainty")
        decision = rec.get("decision")
    except (json.JSONDecodeError, AttributeError) as e:
        return {
            **_base_row(ground_truth),
            "clear_best": None,
            "certainty": None,
            "decision": None,
            "correct": False,
            "failure": f"recommendation.json unparseable: {e}",
        }

    failure_parts = []
    if decision not in valid_decisions:
        failure_parts.append(f"decision not one of {valid_decisions}: {decision!r}")
    if clear_best not in valid_clear_best:
        failure_parts.append(f"clear_best not one of {valid_clear_best}: {clear_best!r}")

    return {
        **_base_row(ground_truth),
        "clear_best": clear_best,
        "certainty": certainty,
        "decision": decision,
        "clear_best_correct": clear_best in correct_clear_best if clear_best in valid_clear_best else False,
        "correct": decision in ground_truth["correct_decisions"] if decision in valid_decisions else False,
        "failure": "; ".join(failure_parts) if failure_parts else None,
    }


def _base_row(ground_truth: dict) -> dict:
    """Passes through every ground_truth field generically (both families
    share scenario/is_null/power/correct_decisions; family-specific fields
    like true_diff vs. true_best_model just pass through too), formatting
    any list-valued fields as '|'-joined strings for a flat CSV row."""
    row = {}
    for key, value in ground_truth.items():
        if key in {"valid_decisions", "valid_clear_best"}:
            continue  # not interesting per-row; same for every scenario in a family
        row[key] = "|".join(str(v) for v in value) if isinstance(value, list) else value
    return row


def _infer_valid_clear_best(ground_truth: dict) -> list[str]:
    if "directional_decision" in ground_truth:
        # prompt_ab family: clear_best tracks which prompt is best.
        return ["prompt_a", "prompt_b", "inconclusive"]
    # model_benchmark family: clear_best shares label space with decision.
    return list(ground_truth["valid_decisions"])


def _infer_correct_clear_best(ground_truth: dict) -> list[str]:
    explicit = ground_truth.get("correct_clear_best")
    if isinstance(explicit, list):
        return explicit

    if "directional_clear_best" in ground_truth:
        directional = ground_truth["directional_clear_best"]
        if ground_truth.get("is_null"):
            return ["inconclusive"]
        if isinstance(ground_truth.get("power"), (int, float)) and ground_truth["power"] < 0.8:
            return [directional, "inconclusive"]
        return [directional]

    if "directional_decision" in ground_truth:
        mapping = {
            "deploy": "prompt_b",
            "do_not_deploy": "prompt_a",
        }
        directional = mapping.get(ground_truth["directional_decision"])
        if directional is None:
            return ["inconclusive"]
        if ground_truth.get("is_null"):
            return ["inconclusive"]
        if isinstance(ground_truth.get("power"), (int, float)) and ground_truth["power"] < 0.8:
            return [directional, "inconclusive"]
        return [directional]

    return list(ground_truth["correct_decisions"])
