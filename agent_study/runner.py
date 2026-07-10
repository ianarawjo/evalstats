"""CLI orchestrator for the v0 agent study.

Must run under .agent-study-venv-runner (has claude-agent-sdk + numpy/scipy/
pandas installed):

    .agent-study-venv-runner/bin/python -m agent_study.runner --dry-run
    .agent-study-venv-runner/bin/python -m agent_study.runner

Generates each scenario, runs the agent once per (scenario, condition) pair,
scores the result, and writes results.csv + manifest.json into a timestamped
runs/<timestamp>/ directory (mirroring simulations/harness/cli.py's
--official-tests output convention).

The agent's actual workspace is NOT this directory -- see workspace.py /
isolation.py. It's a blindly-named temp dir outside the repo, copied back
here (under the real scenario__condition name) only after the run completes,
so nothing scenario/condition-identifying or repo-adjacent is ever visible
to the agent.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import agent_study.scenarios as prompt_ab_scenarios
import agent_study.scenarios_model_benchmark as model_benchmark_scenarios
from agent_study.run_agent import run_agent_sync
from agent_study.score import score_run
from agent_study.workspace import build_workspace, make_run_workspace_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ["full", "baseline"]
ARCHIVE_ARTIFACTS = [
    "data.csv", "PROMPT.md", "recommendation.json", "run_meta.json",
    "transcript.txt", "transcript.jsonl", "containment_violations.json",
]

# family -> its scenarios_*.py module (PHENOMENA dict, build_scenario, write_scenario).
FAMILIES = {
    "prompt_ab": prompt_ab_scenarios,
    "model_benchmark": model_benchmark_scenarios,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", default="prompt_ab", choices=list(FAMILIES), help="Scenario family to run.")
    parser.add_argument(
        "--scenarios", default=None,
        help="Comma-separated phenomenon names (default: all phenomena in --family).",
    )
    parser.add_argument("--conditions", default=",".join(CONDITIONS), help="Comma-separated: full,baseline.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build scenarios + workspaces and print the planned run matrix, but don't call the agent.",
    )
    parser.add_argument("--out-dir", default=None, help="Output dir (default: runs/<timestamp>/).")
    return parser


def _archive(agent_ws: Path, archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVE_ARTIFACTS:
        src = agent_ws / name
        if src.exists():
            shutil.copy2(src, archive_dir / name)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    family_module = FAMILIES[args.family]
    scenario_names = args.scenarios.split(",") if args.scenarios else list(family_module.PHENOMENA)
    conditions = args.conditions.split(",")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "agent_study" / "runs" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_dir = out_dir / "_ground_truth"  # harness-only, never copied into an agent workspace

    plan = [(s, c) for s in scenario_names for c in conditions]
    print(f"Family: {args.family}. Run matrix ({len(plan)} runs): {plan}")

    rows = []
    for scenario_name, condition in plan:
        instance = family_module.build_scenario(scenario_name)
        gt_dir = truth_dir / scenario_name
        family_module.write_scenario(instance, gt_dir)

        archive_dir = out_dir / f"{scenario_name}__{condition}"  # real name, harness-side only
        agent_ws = make_run_workspace_dir()  # opaque name, outside the repo, what the agent actually sees
        build_workspace(instance, condition, agent_ws, family=args.family)

        if args.dry_run:
            _archive(agent_ws, archive_dir)
            print(f"[dry-run] prepared {agent_ws} (archived to {archive_dir})")
            continue

        print(f"Running agent: scenario={scenario_name} condition={condition} (workspace hidden) ...")
        prompt_text = (agent_ws / "PROMPT.md").read_text()
        meta = run_agent_sync(agent_ws, condition, prompt_text, args.family)
        result = score_run(agent_ws, gt_dir / "ground_truth.json")
        result.update({
            "condition": condition, "archive_dir": str(archive_dir),
            **{f"meta_{k}": v for k, v in meta.items()},
        })
        rows.append(result)

        _archive(agent_ws, archive_dir)
        shutil.rmtree(agent_ws, ignore_errors=True)

        flag = " [!] CONTAINMENT VIOLATION ATTEMPTED" if meta.get("n_containment_violations") else ""
        print(f"  -> decision={result['decision']} correct={result['correct']} "
              f"cost=${meta.get('total_cost_usd')} turns={meta.get('num_turns')}{flag}")

    if args.dry_run:
        print(f"\nDry run complete. Archived workspaces under {out_dir}")
        return

    results_path = out_dir / "results.csv"
    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "generated_at": timestamp,
        "family": args.family,
        "scenarios": scenario_names,
        "conditions": conditions,
        "n_runs": len(rows),
        "results_path": str(results_path),
        "total_containment_violations": sum(r.get("meta_n_containment_violations", 0) for r in rows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(rows)} results to {results_path}")
    _print_summary(rows)


def _print_summary(rows: list[dict]) -> None:
    by_condition: dict[str, list[bool]] = {}
    for r in rows:
        by_condition.setdefault(r["condition"], []).append(bool(r["correct"]))
    print("\nDecision accuracy by condition:")
    for condition, results in by_condition.items():
        acc = sum(results) / len(results) if results else float("nan")
        print(f"  {condition}: {sum(results)}/{len(results)} correct ({acc:.0%})")

    total_violations = sum(r.get("meta_n_containment_violations", 0) for r in rows)
    if total_violations:
        print(f"\n[!] {total_violations} containment violation(s) attempted and blocked -- see "
              f"containment_violations.json in the affected run's archive dir.")


if __name__ == "__main__":
    main()
