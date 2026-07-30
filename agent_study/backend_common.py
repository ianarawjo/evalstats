"""Backend-agnostic helpers shared by every run_agent_<backend>.py module.

Each backend normalizes its own message/event format into the same
transcript-event shape (`{"type": "text"|"tool_use"|"tool_result", ...}`,
matching what run_agent_claude_agent_sdk.py already produces) before handing
off to `summarize_tool_use` below, so tool_summary/transcript.jsonl/archival
stay backend-agnostic. Split into its own module (rather than living in
run_agent.py, the dispatcher) to avoid a circular import: the dispatcher
needs to import each backend's `run_agent_sync`, and each backend needs
these helpers -- putting the helpers in a third, lower module breaks the cycle.
"""

from __future__ import annotations

import re

IMPORT_RE = re.compile(r"(?:^|\n)\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)")


def summarize_tool_use(transcript_events: list[dict], bash_commands: list[str], used_evalstats_tool: bool) -> str:
    """A short, harness-computed (not agent-self-reported) synopsis of what
    the agent actually did, for a quick scan of a results table -- tool call
    sequence and the Python libraries its Bash/terminal commands imported."""
    sequence = [e["name"] for e in transcript_events if e["type"] == "tool_use"]
    collapsed: list[list] = []
    for name in sequence:
        if collapsed and collapsed[-1][0] == name:
            collapsed[-1][1] += 1
        else:
            collapsed.append([name, 1])
    seq_str = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in collapsed)

    libs = sorted({m.group(1) for cmd in bash_commands for m in IMPORT_RE.finditer(cmd)})
    libs_str = ", ".join(libs) if libs else "none"

    evalstats_note = "used evalstats tool" if used_evalstats_tool else "did not use evalstats"
    return f"tools: [{seq_str}]; python libs: [{libs_str}]; {evalstats_note}"


def attempted_evalstats_import(bash_commands: list[str]) -> bool:
    """True iff any bash/terminal command contains an actual `import
    evalstats`/`from evalstats import ...` statement -- tighter than a bare
    substring check, which produced false positives in practice (a command
    that merely *mentions* "evalstats" in a comment or a failed guess at its
    CLI, without ever executing real evalstats code)."""
    return any(m.group(1) == "evalstats" for cmd in bash_commands for m in IMPORT_RE.finditer(cmd))
