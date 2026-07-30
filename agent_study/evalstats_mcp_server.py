"""Standalone stdio MCP server exposing evalstats' compare_prompts() and
compare_models() as tools -- the shared implementation both backends can
point at (Claude Agent SDK via McpStdioServerConfig, OpenHands via
Agent.mcp_config's dict[str, MCPServer]), intended to eventually
replace the in-process, claude_agent_sdk-specific builders in
evalstats_tools.py. Per the plan: build this for the shared path, but
evalstats_tools.py stays in place and in active use for the claude_agent_sdk
backend until this one is validated side by side -- no risky big-bang swap.

Runs in the orchestrator's own process/venv (.agent-study-venv-runner, which
has evalstats+pandas+mcp installed), NOT inside a sandboxed condition
venv/Docker image -- same execution location as the existing in-process
tool. This works because a run's workspace directory is always a plain host
path the orchestrator already controls (for the openhands backend, it's the
same host directory bind-mounted into the container, so reading data.csv
from here and reading it from inside the container see the same file).

Which workspace's data.csv to read comes from the EVALSTATS_WORKSPACE_DIR
env var (set via the spawning backend's per-server `env` config), since a
standalone stdio process can't receive it as a plain Python argument the way
the in-process builders do.

    python -m agent_study.evalstats_mcp_server

NOT YET LIVE-VALIDATED -- see run_agent_openhands.py's module docstring.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

import evalstats as estats

mcp = FastMCP("evalstats")


def _workspace_dir() -> Path:
    value = os.environ.get("EVALSTATS_WORKSPACE_DIR")
    if not value:
        raise RuntimeError("EVALSTATS_WORKSPACE_DIR not set -- must be passed via the MCP server's env config.")
    return Path(value)


def _capture_summary(report) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report.summary(style="line")
    return f"{buf.getvalue()}\n\nStructured result:\n{report.to_dict()}"


@mcp.tool(
    description=(
        "Statistically compare two prompt/model variants using evalstats' compare_prompts(), "
        "a bootstrap-based comparison that reports a confidence interval, p-value, and effect "
        "size for the difference between variants -- more principled than eyeballing means. "
        "Reads data.csv in the current workspace (columns: item_id, prompt_a_score, "
        "prompt_b_score -- paired scores, same item scored under both variants, higher is "
        "better). Returns the human-readable summary plus a structured JSON breakdown."
    ),
)
def compare_prompts(statistic: str = "mean") -> str:
    df = pd.read_csv(_workspace_dir() / "data.csv")
    scores = {"prompt_a": df["prompt_a_score"].to_numpy(), "prompt_b": df["prompt_b_score"].to_numpy()}
    report = estats.compare_prompts(scores, statistic=statistic, n_bootstrap=2000)
    return _capture_summary(report)


@mcp.tool(
    description=(
        "Statistically compare several models' scores using evalstats' compare_models(), "
        "which reports confidence intervals, multiple-comparison-corrected p-values, and "
        "which models are NOT significantly beaten by any other (the 'unbeaten' set) -- more "
        "principled than picking whichever has the highest raw mean, especially with many "
        "models where naive pairwise testing inflates false positives. Reads data.csv in the "
        "current workspace (long format: item_id, model, score -- one row per model/item "
        "pair, same items scored by every model, higher is better). Returns the human-readable "
        "summary plus a structured JSON breakdown."
    ),
)
def compare_models(statistic: str = "mean") -> str:
    df = pd.read_csv(_workspace_dir() / "data.csv")
    scores = {model: group.sort_values("item_id")["score"].to_numpy() for model, group in df.groupby("model")}
    report = estats.compare_models(scores, statistic=statistic, correction="fdr_bh", n_bootstrap=2000)
    return _capture_summary(report)


if __name__ == "__main__":
    mcp.run(transport="stdio")
