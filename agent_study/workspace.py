"""Builds one agent-visible scratch workspace per (scenario, condition) run.

The workspace directory is exactly what the agent sees via its Read/Write/Bash
tools -- it must never contain ground_truth.json or anything else that would
leak the correct answer.
"""

from __future__ import annotations

import secrets
import tempfile
from pathlib import Path
from typing import Any

from agent_study.prompts import implicit_prompt, model_benchmark_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent

# Agent workspaces live outside the repo entirely, under an opaque random
# name -- see isolation.py's module docstring for why (a real run both
# leaked a nearby file and, per a hypothesis worth taking seriously, may
# have had its behavior shaped by scenario/condition names visible in the
# workspace path).
WORKSPACE_ROOT = Path(tempfile.gettempdir()) / "agent_study_workspaces"


def make_run_workspace_dir() -> Path:
    return WORKSPACE_ROOT / secrets.token_hex(8)

# Condition -> venv whose bin/ should be prepended to PATH for the agent's
# Bash tool, so `python`/`pip` inside the sandbox resolve to that
# environment. "full" has evalstats installed; "baseline" does not (and has
# no network access, so it can't `pip install` its way around that).
VENV_BY_CONDITION = {
    "full": REPO_ROOT / ".agent-study-venv-full",
    "baseline": REPO_ROOT / ".agent-study-venv-baseline",
}


def venv_bin(condition: str) -> Path:
    if condition not in VENV_BY_CONDITION:
        raise ValueError(f"Unknown condition {condition!r}, expected one of {list(VENV_BY_CONDITION)}")
    return VENV_BY_CONDITION[condition] / "bin"


def build_workspace(instance: Any, condition: str, run_dir: Path, family: str = "prompt_ab") -> Path:
    """Writes data.csv + PROMPT.md into run_dir and returns run_dir (the
    agent's cwd). Does NOT write ground_truth.json here -- callers should
    write that to a sibling location outside the workspace."""
    run_dir.mkdir(parents=True, exist_ok=True)
    instance.data.to_csv(run_dir / "data.csv", index=False)
    evalstats_available = condition == "full"
    if family == "prompt_ab":
        prompt_text = implicit_prompt(n=instance.n, evalstats_available=evalstats_available)
    elif family == "model_benchmark":
        prompt_text = model_benchmark_prompt(n=instance.n, k=instance.k, evalstats_available=evalstats_available)
    else:
        raise ValueError(f"Unknown family {family!r}")
    (run_dir / "PROMPT.md").write_text(prompt_text)
    return run_dir
