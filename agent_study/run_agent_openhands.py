"""Runs one agent episode against a prepared workspace, via the OpenHands
backend.

Uses openhands.sdk's own Python API (Agent/Conversation/DockerWorkspace)
directly, NOT the `openhands` CLI's plain `--headless` mode -- that path
constructs a bare `Workspace(working_dir=os.getcwd())` with no container at
all (confirmed by reading openhands_cli/setup.py), so it runs completely
unsandboxed on the host by default. DockerWorkspace is the only path in this
SDK that gives real container isolation, and it's only reachable via the
Python API, not a CLI flag.

Must run under .agent-study-venv-runner, which needs to be Python 3.12 (not
3.14) -- openhands-sdk's litellm dependency has a Rust extension (via PyO3)
that doesn't build on 3.14 yet.

Live-validated (2026-07-12): one full clear_improvement/full episode via
Claude Haiku, correct decision, evalstats MCP tool called successfully. That
first live run needed 4 real fixes past what source-reading alone predicted
-- all against openhands-sdk 1.35.0, whose own docstrings/examples were
stale relative to the installed version in ways only a live run surfaced:
  1. platform: DockerWorkspace defaults to "linux/amd64"; a locally-built
     image without an explicit --platform is native-arch-only (arm64 on
     this Mac) -- now auto-detected (DOCKER_PLATFORM below).
  2. mcp_config shape: dict[str, MCPServer] directly (server name as the
     top-level key), NOT wrapped in a "mcpServers" key the way the
     standard MCP client convention (and this SDK's own docstring example)
     shows.
  3. Tool registration name: the registry key is the tool class's own
     `.name` attribute ("terminal", "file_editor"), not the class name
     ("TerminalTool", "FileEditorTool") the Tool field's docstring example
     uses. Also needed *some* registration mechanism at all -- the
     client-side dynamic one (RemoteConversation sending
     tool_module_qualnames) is documented in the agent-server's own
     preload_modules() docstring as racy, and failed live; --import-modules
     baked into the image's entrypoint (entrypoint.sh) is what actually
     worked reliably.
  4. MCP server process location: Agent._initialize (where mcp_config
     servers connect) runs wherever the Conversation executes -- for a
     DockerWorkspace-backed RemoteConversation, that's server-side (inside
     the container), not our host orchestrator process the way the
     claude_agent_sdk backend's in-process tool does. A host path in
     mcp_config's "command" is meaningless there.
"""

from __future__ import annotations

import json
import os
import platform as platform_module
import sys
import time
from pathlib import Path

from openhands.sdk import LLM, Agent, Conversation, Event, Tool
from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.sdk.event.llm_convertible import ActionEvent, MessageEvent, ObservationEvent
from openhands.workspace.docker.workspace import DockerWorkspace
from pydantic import SecretStr

# Importing these (not just referencing "TerminalTool"/"FileEditorTool" by
# string in Tool(name=...)) is required, not decorative: tool classes
# self-register as a side effect of being imported (openhands/sdk/tool/registry.py),
# and RemoteConversation reads that *client-side* registry to compute
# tool_module_qualnames, which it sends to the container's agent-server so
# it can dynamically import the same modules there. Skipping this import
# left both registries empty and produced a live, confirmed
# KeyError: "ToolDefinition 'TerminalTool' is not registered" server-side.
from openhands.tools.file_editor import FileEditorTool  # noqa: F401
from openhands.tools.terminal import TerminalTool  # noqa: F401

from agent_study.backend_common import attempted_evalstats_import, summarize_tool_use
from agent_study.env import load_env
from agent_study.workspace import DOCKER_IMAGE_BY_CONDITION

# Mirrors run_agent_claude_agent_sdk.py's MAX_TURNS -- kills a run if it
# runs away, so a single misbehaving episode can't blow through budget
# unattended. openhands.sdk calls this max_iteration_per_run.
MAX_ITERATIONS = 40

# "Claude via OpenHands first" per discussion -- validate the harness itself
# before swapping to a cheap/open model. litellm-style model string;
# untested, may need adjustment once live (see module docstring).
DEFAULT_MODEL = os.environ.get("OPENHANDS_LLM_MODEL", "claude-haiku-4-5")

# DockerWorkspace defaults to platform="linux/amd64" (a typical x86 CI/cloud
# assumption), but our images are built for whatever this machine's native
# arch is -- on Apple Silicon that's arm64, and requesting amd64 against an
# arm64-only local image tag makes Docker try to *pull* it from a registry
# instead of finding it locally (confirmed: got exactly that failure before
# this fix). Auto-detect so this also works unchanged on an x86 host/cloud
# instance, overridable via env var for edge cases.
_MACHINE_TO_DOCKER_PLATFORM = {"arm64": "linux/arm64", "aarch64": "linux/arm64", "x86_64": "linux/amd64"}
DOCKER_PLATFORM = os.environ.get(
    "OPENHANDS_DOCKER_PLATFORM", _MACHINE_TO_DOCKER_PLATFORM.get(platform_module.machine(), "linux/amd64"),
)

# openhands-sdk doesn't compute cost for us the way claude_agent_sdk's
# ResultMessage.total_cost_usd does -- ConversationStats/Metrics tracks
# accumulated_cost when the LLM's pricing is known to litellm, which should
# cover Claude models out of the box; flagged here in case it reads as 0.0
# unexpectedly for a model litellm doesn't have pricing data for.


def run_agent_sync(workspace_dir: Path, condition: str, prompt: str, family: str = "prompt_ab") -> dict:
    load_env()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Copy agent_study/.env.example to "
            "agent_study/.env and fill in a key."
        )
    if condition not in DOCKER_IMAGE_BY_CONDITION:
        raise ValueError(f"Unknown condition {condition!r}, expected one of {list(DOCKER_IMAGE_BY_CONDITION)}")

    llm = LLM(model=DEFAULT_MODEL, api_key=SecretStr(api_key))
    # Only "full" gets the evalstats tool -- same rule and same rationale as
    # the claude_agent_sdk backend (evalstats_tools.py's module docstring).
    # Spawned via the shared standalone stdio MCP server (evalstats_mcp_server.py).
    # Unlike the claude_agent_sdk backend (where the equivalent tool runs
    # in-process in our own orchestrator), this one runs INSIDE the container:
    # confirmed live that Agent._initialize (where mcp_config-declared servers
    # connect) executes wherever the Conversation actually runs, which for a
    # DockerWorkspace-backed RemoteConversation is server-side, same as tool
    # resolution below. A host path here (e.g. sys.executable) is meaningless
    # inside the container -- produced a live "MCP Connection Failure" before
    # this fix. /opt/evalstats_mcp_server.py and container-relative /workspace
    # are baked into Dockerfile.full (see its comments).
    #
    # NOTE: Agent.mcp_config is typed dict[str, MCPServer] in the installed
    # SDK version (1.35.0) -- server name directly as the top-level key, NOT
    # wrapped in a "mcpServers" key the way the standard MCP client config
    # convention (and this SDK's own docstring example, apparently stale for
    # this version) shows. Confirmed by bisecting a real ValidationError live.
    mcp_config = {
        "evalstats": {
            "command": "python3",
            "args": ["/opt/evalstats_mcp_server.py"],
            "env": {"EVALSTATS_WORKSPACE_DIR": "/workspace"},
        }
    } if condition == "full" else {}
    # Registered names are the tool classes' own .name attribute ("terminal",
    # "file_editor"), not the class names -- confirmed live; the class-name
    # form is what Tool's own field docstring shows, which is stale for this
    # installed SDK version (1.35.0), same pattern as the mcp_config surprise
    # above.
    agent = Agent(
        llm=llm,
        tools=[Tool(name="terminal"), Tool(name="file_editor")],
        mcp_config=mcp_config,
    )

    # MCP-sourced tool names in this SDK are unprefixed -- confirmed via
    # `assert self.name == self.mcp_tool.name` in openhands/sdk/mcp/tool.py,
    # unlike Claude Code's `mcp__<server>__<tool>` convention -- so the
    # evalstats tools show up as bare "compare_prompts"/"compare_models".
    EVALSTATS_TOOL_NAMES = {"compare_prompts", "compare_models"}

    transcript_lines: list[str] = []
    transcript_events: list[dict] = []
    bash_commands: list[str] = []
    used_evalstats_tool_flag = [False]

    def _on_event(event: Event) -> None:
        # Normalize into the shared {"type": ...} schema from
        # backend_common.py so summarize_tool_use/archival work the same
        # regardless of backend -- mirrors what run_agent_claude_agent_sdk.py
        # does inline in its query() loop.
        if isinstance(event, MessageEvent) and event.source == "agent":
            text = "".join(c.text for c in event.llm_message.content if hasattr(c, "text"))
            if text:
                transcript_lines.append(text)
                transcript_events.append({"type": "text", "text": text})
        elif isinstance(event, ActionEvent):
            tool_input = event.action.model_dump() if event.action is not None else {}
            transcript_events.append({"type": "tool_use", "name": event.tool_name, "input": tool_input})
            if event.tool_name == "terminal":
                cmd = tool_input.get("command")
                if isinstance(cmd, str):
                    bash_commands.append(cmd)
            elif event.tool_name in EVALSTATS_TOOL_NAMES:
                used_evalstats_tool_flag[0] = True
        elif isinstance(event, ObservationEvent):
            content = str(event.observation)
            transcript_events.append({"type": "tool_result", "is_error": False, "content": content[:4000]})

    start = time.monotonic()
    is_error = False
    stop_reason = "end_turn"
    try:
        with DockerWorkspace(
            server_image=DOCKER_IMAGE_BY_CONDITION[condition],
            volumes=[f"{workspace_dir}:/workspace"],
            platform=DOCKER_PLATFORM,
        ) as workspace:
            conversation = Conversation(
                agent=agent, workspace=workspace, callbacks=[_on_event], max_iteration_per_run=MAX_ITERATIONS,
            )
            conversation.send_message(prompt)
            conversation.run()
            final_text = get_agent_final_response(conversation.state.events)
            if final_text and not transcript_lines:
                transcript_lines.append(final_text)

            cost = 0.0
            try:
                for metrics in conversation.conversation_stats.usage_to_metrics.values():
                    cost += metrics.accumulated_cost
            except Exception:
                cost = None  # unverified accessor -- see module docstring
    except Exception as e:
        is_error = True
        stop_reason = f"error: {e}"
        cost = None

    duration_ms = int((time.monotonic() - start) * 1000)
    num_turns = sum(1 for e in transcript_events if e["type"] == "tool_use")

    attempted_import = attempted_evalstats_import(bash_commands)
    used_evalstats_tool = used_evalstats_tool_flag[0]
    result_meta = {
        "num_turns": num_turns,
        "duration_ms": duration_ms,
        "total_cost_usd": cost,
        "is_error": is_error,
        "stop_reason": stop_reason,
        "used_evalstats": used_evalstats_tool or attempted_import,
        "used_evalstats_tool_call": used_evalstats_tool,
        "attempted_evalstats_import_in_bash": attempted_import,
        "n_bash_commands": len(bash_commands),
        # No containment-violations concept for this backend -- Docker's own
        # filesystem namespace is the isolation boundary, not a hook that
        # can deny/log an attempt. See the plan's Isolation section.
        "n_containment_violations": None,
        "tool_summary": summarize_tool_use(transcript_events, bash_commands, used_evalstats_tool or attempted_import),
    }

    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "transcript.txt").write_text("\n\n---\n\n".join(transcript_lines))
    with (workspace_dir / "transcript.jsonl").open("w") as f:
        for event in transcript_events:
            f.write(json.dumps(event) + "\n")
    (workspace_dir / "run_meta.json").write_text(json.dumps(result_meta, indent=2))
    return result_meta


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("usage: run_agent_openhands.py <workspace_dir> <condition: full|baseline> "
              "[family: prompt_ab|model_benchmark]", file=sys.stderr)
        sys.exit(1)
    ws_dir = Path(sys.argv[1]).resolve()
    cond = sys.argv[2]
    fam = sys.argv[3] if len(sys.argv) == 4 else "prompt_ab"
    prompt_text = (ws_dir / "PROMPT.md").read_text()
    meta = run_agent_sync(ws_dir, cond, prompt_text, fam)
    print(json.dumps(meta, indent=2))
