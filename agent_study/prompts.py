"""Prompt templates for the v0 agent study.

v0 only implements the "implicit" (developer-oriented) prompt style -- the
realistic case where a developer asks for a recommendation without asking for
statistical rigor. An "explicit" (statistics-oriented) variant is deferred
past v0 (see the plan).
"""

from __future__ import annotations

# Fixed regardless of prompt style: how the agent must report its final
# answer, so scoring doesn't depend on parsing free text. This is a harness
# convention about output format, not a hint about which statistical method
# to use.
_ANSWER_FOOTER = """
When you're done, write your final recommendation to a file named
`recommendation.json` in the current directory, in this exact format:

{"decision": "deploy", "reasoning": "<one paragraph>"}

`decision` must be exactly one of "deploy", "do_not_deploy", or
"inconclusive" (use "inconclusive" only if the evidence genuinely doesn't
support a confident call either way).
"""


# Only added in the "full" condition. This names the tool (evalstats is
# obscure enough that the model has no training-data familiarity with it the
# way it would with e.g. scipy, so it needs to at least be told the tool
# exists) without saying which method to use or that it must be used --
# same as a developer mentioning "we have an internal stats helper" without
# dictating the analysis.
_EVALSTATS_MENTION = (
    "\nYou have a `compare_prompts` tool (evalstats) available if it's useful "
    "for this kind of analysis.\n"
)


def implicit_prompt(n: int, candidate_label: str = "prompt_b", evalstats_available: bool = False) -> str:
    mention = _EVALSTATS_MENTION if evalstats_available else ""
    return f"""We're deciding whether to replace our current production prompt
("prompt_a") with a new candidate ("{candidate_label}"). I ran both prompts
against the same set of {n} test inputs and logged the scores in `data.csv`
in the current directory (columns: item_id, prompt_a_score, prompt_b_score --
higher is better).

Take a look at the results and tell me whether we should deploy
{candidate_label} to replace prompt_a.
{mention}{_ANSWER_FOOTER}"""


# --- model_benchmark family (8-model comparison, no incumbent) ---

_MODEL_BENCHMARK_ANSWER_FOOTER_TEMPLATE = """
When you're done, write your final recommendation to a file named
`recommendation.json` in the current directory, in this exact format:

{{"decision": "model_3", "reasoning": "<one paragraph>"}}

`decision` must be exactly one of {options} (use "inconclusive" only if the
evidence genuinely doesn't support confidently naming a single best model).
"""

_EVALSTATS_MENTION_MODEL_BENCHMARK = (
    "\nYou have a `compare_models` tool (evalstats) available if it's useful "
    "for this kind of analysis.\n"
)


def model_benchmark_prompt(n: int, k: int = 8, evalstats_available: bool = False) -> str:
    mention = _EVALSTATS_MENTION_MODEL_BENCHMARK if evalstats_available else ""
    model_labels = [f"model_{i + 1}" for i in range(k)]
    options = ", ".join(f'"{m}"' for m in model_labels) + ', or "inconclusive"'
    footer = _MODEL_BENCHMARK_ANSWER_FOOTER_TEMPLATE.format(options=options)
    return f"""We're picking which of {k} candidate models to use in production.
I ran all {k} models ({", ".join(model_labels)}) against the same set of {n}
test inputs and logged the scores in `data.csv` in the current directory
(columns: item_id, model, score -- higher is better; one row per
model/item pair).

Take a look at the results and tell me which model we should use.
{mention}{footer}"""
