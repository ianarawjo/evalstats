"""Dispatches one agent episode to the selected backend.

    .agent-study-venv-runner/bin/python -m agent_study.run_agent <workspace_dir> <condition> [family] [backend]

Backends implement the same run_agent_sync(workspace_dir, condition, prompt,
family) -> dict contract:
    - "claude_agent_sdk" (run_agent_claude_agent_sdk.py) -- the original,
      fully validated backend. Default.
    - "openhands" (run_agent_openhands.py) -- Docker-sandboxed,
      model-agnostic. Under construction; see that module.

Shared, backend-agnostic helpers (the transcript-event schema every backend
normalizes into, and the tool-use summarizer built on top of it) live in
backend_common.py, not here, to avoid a circular import between this
dispatcher and the backend modules it imports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import agent_study.run_agent_claude_agent_sdk as claude_agent_sdk_backend
import agent_study.run_agent_openhands as openhands_backend

BACKENDS = {
    "claude_agent_sdk": claude_agent_sdk_backend.run_agent_sync,
    "openhands": openhands_backend.run_agent_sync,
}


def run_agent_sync(
    workspace_dir: Path, condition: str, prompt: str,
    family: str = "prompt_ab", backend: str = "claude_agent_sdk",
) -> dict:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}, expected one of {list(BACKENDS)}")
    return BACKENDS[backend](workspace_dir, condition, prompt, family)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4, 5):
        print("usage: run_agent.py <workspace_dir> <condition: full|baseline> "
              "[family: prompt_ab|model_benchmark] [backend: claude_agent_sdk|openhands]", file=sys.stderr)
        sys.exit(1)
    ws_dir = Path(sys.argv[1]).resolve()
    cond = sys.argv[2]
    fam = sys.argv[3] if len(sys.argv) >= 4 else "prompt_ab"
    backend_name = sys.argv[4] if len(sys.argv) == 5 else "claude_agent_sdk"
    prompt_text = (ws_dir / "PROMPT.md").read_text()
    meta = run_agent_sync(ws_dir, cond, prompt_text, fam, backend_name)
    print(json.dumps(meta, indent=2))
