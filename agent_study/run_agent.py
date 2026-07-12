"""Runs one agent episode against a prepared workspace.

Must run under .agent-study-venv-runner (has claude-agent-sdk installed):

    .agent-study-venv-runner/bin/python -m agent_study.run_agent <workspace_dir> <condition>

Requires ANTHROPIC_API_KEY -- see agent_study/.env.example.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from agent_study.env import load_env
from agent_study.evalstats_tools import build_evalstats_mcp_server, build_evalstats_mcp_server_model_benchmark
from agent_study.isolation import build_pre_tool_use_hook
from agent_study.workspace import VENV_BY_CONDITION, venv_bin

# Hard per-run ceiling: kills a run if it runs away, so a single misbehaving
# episode can't blow through the study's budget unattended.
MAX_TURNS = 40
MAX_BUDGET_USD = 1.0

# family -> (MCP server builder, tool name to allow). Each family's evalstats
# tool is registered only in the "full" condition -- see _run_agent below.
_EVALSTATS_TOOL_BY_FAMILY = {
    "prompt_ab": (build_evalstats_mcp_server, "mcp__evalstats__compare_prompts"),
    "model_benchmark": (build_evalstats_mcp_server_model_benchmark, "mcp__evalstats__compare_models"),
}


def _find_cli_path() -> str:
    exec_path = os.environ.get("CLAUDE_CODE_EXECPATH")
    if exec_path and Path(exec_path).exists():
        return exec_path
    which = shutil.which("claude")
    if which:
        return which
    raise RuntimeError(
        "Could not locate the `claude` CLI binary (checked $CLAUDE_CODE_EXECPATH "
        "and PATH). Install @anthropic-ai/claude-code or set cli_path explicitly."
    )


def run_agent_sync(workspace_dir: Path, condition: str, prompt: str, family: str = "prompt_ab") -> dict:
    return asyncio.run(_run_agent(workspace_dir, condition, prompt, family))


_IMPORT_RE = re.compile(r"(?:^|\n)\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)")


def _summarize_tool_use(transcript_events: list[dict], bash_commands: list[str], used_evalstats_tool: bool) -> str:
    """A short, harness-computed (not agent-self-reported) synopsis of what
    the agent actually did, for a quick scan of a results table -- tool call
    sequence and the Python libraries its Bash commands imported."""
    sequence = [e["name"] for e in transcript_events if e["type"] == "tool_use"]
    collapsed: list[list] = []
    for name in sequence:
        if collapsed and collapsed[-1][0] == name:
            collapsed[-1][1] += 1
        else:
            collapsed.append([name, 1])
    seq_str = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in collapsed)

    libs = sorted({m.group(1) for cmd in bash_commands for m in _IMPORT_RE.finditer(cmd)})
    libs_str = ", ".join(libs) if libs else "none"

    evalstats_note = "used evalstats tool" if used_evalstats_tool else "did not use evalstats"
    return f"tools: [{seq_str}]; python libs: [{libs_str}]; {evalstats_note}"


async def _run_agent(workspace_dir: Path, condition: str, prompt: str, family: str = "prompt_ab") -> dict:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Copy agent_study/.env.example to "
            "agent_study/.env and fill in a key."
        )
    if family not in _EVALSTATS_TOOL_BY_FAMILY:
        raise ValueError(f"Unknown family {family!r}, expected one of {list(_EVALSTATS_TOOL_BY_FAMILY)}")

    bin_dir = venv_bin(condition)
    allowed_tools = ["Read", "Write", "Bash"]
    mcp_servers = {}
    if condition == "full":
        # Give evalstats a self-documenting tool affordance (name +
        # description + schema) in addition to being pip-installed in the
        # Bash venv -- see evalstats_tools.py for why.
        build_mcp_server, tool_name = _EVALSTATS_TOOL_BY_FAMILY[family]
        mcp_servers["evalstats"] = build_mcp_server(workspace_dir)
        allowed_tools.append(tool_name)

    # PreToolUse hook fires unconditionally regardless of permission_mode
    # (unlike can_use_tool, which bypassPermissions skips entirely) -- see
    # isolation.py for what leaked before this was added and why the rule is
    # shaped the way it is.
    containment_violations: list[dict] = []
    isolation_hook = build_pre_tool_use_hook(
        workspace_dir=workspace_dir, venv_dir=VENV_BY_CONDITION[condition], violations=containment_violations,
    )

    options = ClaudeAgentOptions(
        cli_path=_find_cli_path(),
        model="haiku",
        cwd=str(workspace_dir),
        allowed_tools=allowed_tools,
        mcp_servers=mcp_servers,
        permission_mode="bypassPermissions",
        max_turns=MAX_TURNS,
        max_budget_usd=MAX_BUDGET_USD,
        # The SDK merges this on top of the inherited parent environment, so
        # only the PATH override needs to be listed here.
        env={"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"},
        # No network for the agent's Bash commands in either condition --
        # evalstats/numpy/etc. are already installed in the venv it's using,
        # and this prevents a "baseline" run from `pip install`-ing its way
        # around the condition.
        sandbox={"enabled": True, "network": {"allowedDomains": []}},
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[isolation_hook])]},
    )

    # transcript.txt: prose only, for a quick human read. transcript.jsonl:
    # every message including tool_use/tool_result, so later analysis can
    # verify what the agent actually ran (e.g. did a "full"-condition run
    # actually import evalstats, or just have it available and not use it).
    transcript_lines: list[str] = []
    transcript_events: list[dict] = []
    bash_commands: list[str] = []
    used_evalstats_tool = False
    result_meta: dict = {}

    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    transcript_lines.append(block.text)
                    transcript_events.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    transcript_events.append({"type": "tool_use", "name": block.name, "input": block.input})
                    if block.name == "Bash" and isinstance(block.input, dict):
                        cmd = block.input.get("command")
                        if isinstance(cmd, str):
                            bash_commands.append(cmd)
                    elif block.name.startswith("mcp__evalstats__"):
                        used_evalstats_tool = True
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    content = block.content
                    if isinstance(content, list):
                        content = "\n".join(
                            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                        )
                    transcript_events.append({
                        "type": "tool_result", "is_error": block.is_error,
                        "content": (content or "")[:4000],
                    })
        elif isinstance(msg, ResultMessage):
            result_meta = {
                "num_turns": msg.num_turns,
                "duration_ms": msg.duration_ms,
                "total_cost_usd": msg.total_cost_usd,
                "is_error": msg.is_error,
                "stop_reason": msg.stop_reason,
            }

    # Real runs showed the naive `"evalstats" in cmd` check produces false
    # positives: a Bash command can mention "evalstats" while merely failing
    # to invoke it (a comment, a guessed CLI call, an attempted subprocess
    # wrapper) without ever running real evalstats code. `_IMPORT_RE`-based
    # matching (already used for the tool_summary's python-libs list) is a
    # tighter signal: it only fires on an actual `import evalstats`/`from
    # evalstats import ...` statement, which is at least a genuine attempt
    # to execute evalstats code (still no guarantee it succeeded -- that
    # would need parsing the tool_result for a traceback).
    attempted_evalstats_import = any(
        m.group(1) == "evalstats" for cmd in bash_commands for m in _IMPORT_RE.finditer(cmd)
    )
    result_meta["used_evalstats"] = used_evalstats_tool or attempted_evalstats_import
    result_meta["used_evalstats_tool_call"] = used_evalstats_tool
    result_meta["attempted_evalstats_import_in_bash"] = attempted_evalstats_import
    result_meta["n_bash_commands"] = len(bash_commands)
    result_meta["n_containment_violations"] = len(containment_violations)
    result_meta["tool_summary"] = _summarize_tool_use(transcript_events, bash_commands, result_meta["used_evalstats"])
    if containment_violations:
        (workspace_dir / "containment_violations.json").write_text(json.dumps(containment_violations, indent=2))

    (workspace_dir / "transcript.txt").write_text("\n\n---\n\n".join(transcript_lines))
    with (workspace_dir / "transcript.jsonl").open("w") as f:
        for event in transcript_events:
            f.write(json.dumps(event) + "\n")
    (workspace_dir / "run_meta.json").write_text(json.dumps(result_meta, indent=2))
    return result_meta


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("usage: run_agent.py <workspace_dir> <condition: full|baseline> [family: prompt_ab|model_benchmark]",
              file=sys.stderr)
        sys.exit(1)
    ws_dir = Path(sys.argv[1]).resolve()
    cond = sys.argv[2]
    fam = sys.argv[3] if len(sys.argv) == 4 else "prompt_ab"
    prompt_text = (ws_dir / "PROMPT.md").read_text()
    meta = run_agent_sync(ws_dir, cond, prompt_text, fam)
    print(json.dumps(meta, indent=2))
