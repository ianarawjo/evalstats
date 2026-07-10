"""In-process MCP server exposing evalstats.compare_prompts() as a callable
tool, used only in the "full" condition.

Why a tool and not just a pip-installed package: the smoke test showed the
agent never even checked what was installed -- it hand-rolled a bootstrap
with bare `random`, ignoring numpy/scipy/evalstats entirely. Well-known
libraries like scipy get used "for free" because the model already knows
their API from training; evalstats doesn't have that advantage, and there's
no time in a short, budget-capped episode for the agent to cold-read its
source to learn it. Exposing it as an MCP tool (name + description + schema)
gives it the same kind of self-documenting affordance scipy gets from prior
knowledge, without the prompt itself hinting which method to use.

Runs in-process in the orchestrator (.agent-study-venv-runner), not in the
agent's sandboxed venv -- see run_agent.py.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Annotated, Any

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server, tool


def build_evalstats_mcp_server(workspace_dir: Path) -> McpSdkServerConfig:
    @tool(
        "compare_prompts",
        (
            "Statistically compare two prompt/model variants using evalstats' compare_prompts(), "
            "a bootstrap-based comparison that reports a confidence interval, p-value, and effect "
            "size for the difference between variants -- more principled than eyeballing means. "
            "Reads data.csv in the current workspace (columns: item_id, prompt_a_score, "
            "prompt_b_score -- paired scores, same item scored under both variants, higher is "
            "better). Returns the human-readable summary plus a structured JSON breakdown."
        ),
        {"statistic": Annotated[str, "'mean' or 'median'. Defaults to 'mean'."]},
    )
    async def compare_prompts_tool(args: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd

        import evalstats as estats

        df = pd.read_csv(workspace_dir / "data.csv")
        scores = {
            "prompt_a": df["prompt_a_score"].to_numpy(),
            "prompt_b": df["prompt_b_score"].to_numpy(),
        }
        statistic = args.get("statistic") or "mean"
        report = estats.compare_prompts(scores, statistic=statistic, n_bootstrap=2000)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report.summary(style="line")
        text = f"{buf.getvalue()}\n\nStructured result:\n{report.to_dict()}"
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server(name="evalstats", tools=[compare_prompts_tool])


def build_evalstats_mcp_server_model_benchmark(workspace_dir: Path) -> McpSdkServerConfig:
    """model_benchmark family's analogue of build_evalstats_mcp_server above
    -- same rationale, wrapping compare_models() instead of compare_prompts()
    since this family has k=8 arms rather than 2. Correction matters a lot
    more here (28 pairwise comparisons among 8 models vs. 1 for a pair),
    which is the whole point of this scenario family -- so it's passed
    explicitly rather than left at whatever compare()'s default is."""

    @tool(
        "compare_models",
        (
            "Statistically compare several models' scores using evalstats' compare_models(), "
            "which reports confidence intervals, multiple-comparison-corrected p-values, and "
            "which models are NOT significantly beaten by any other (the 'unbeaten' set) -- more "
            "principled than picking whichever has the highest raw mean, especially with many "
            "models where naive pairwise testing inflates false positives. Reads data.csv in the "
            "current workspace (long format: item_id, model, score -- one row per model/item "
            "pair, same items scored by every model, higher is better). Returns the human-readable "
            "summary plus a structured JSON breakdown."
        ),
        {"statistic": Annotated[str, "'mean' or 'median'. Defaults to 'mean'."]},
    )
    async def compare_models_tool(args: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd

        import evalstats as estats

        df = pd.read_csv(workspace_dir / "data.csv")
        scores = {
            model: group.sort_values("item_id")["score"].to_numpy()
            for model, group in df.groupby("model")
        }
        statistic = args.get("statistic") or "mean"
        report = estats.compare_models(scores, statistic=statistic, correction="fdr_bh", n_bootstrap=2000)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report.summary(style="line")
        text = f"{buf.getvalue()}\n\nStructured result:\n{report.to_dict()}"
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server(name="evalstats", tools=[compare_models_tool])
