"""One parametrized implicit-prompt template covering the whole sweep
(task_type x k x data_type x n), replacing the two earlier families'
separate hand-written templates (agent_study/prompts.py's implicit_prompt
and model_benchmark_prompt) now that the decision schema is unified to
entity-name-or-"inconclusive" across every k (see generate.py's docstring).

Deliberately says nothing about n's DGP regime or ground_truth_regime --
those are what the agent has to discover from the data itself. evalstats
availability is a run-time condition, not a property of the stored
benchmark item, so it isn't baked in here; a harness running this benchmark
appends its own tool-availability mention (or doesn't) when it launches an
agent against a given item.
"""

from __future__ import annotations

from agent_study.sweep.generate import make_entity_labels

_DATA_TYPE_DESCRIPTIONS = {
    "binary": "1 = pass, 0 = fail",
    "continuous": "a 0-1 score, higher is better",
    "likert": "a 1-5 rating, higher is better",
}

_ANSWER_FOOTER_TEMPLATE = """
When you're done, write your final recommendation to a file named
`recommendation.json` in the current directory, in this exact format:

{{"reasoning": "<one paragraph>", "clear_best": "...", "certainty": "<low|medium|high>", "decision": "..."}}

`clear_best` and `decision` must each be exactly one of {options}.
`clear_best` is which {noun} the evidence points to as the clear best, using
"inconclusive" if nothing clearly stands out. `certainty` is your own
certainty in that read. `decision` is your final recommendation -- it may
differ from `clear_best`.
"""


def build_prompt(task_type: str, k: int, data_type: str, n: int) -> str:
    labels = make_entity_labels(task_type, k)
    noun = "prompt" if task_type == "prompts" else "model"
    task_desc = (
        f"We're picking which of {k} candidate prompt variants to use in production."
        if task_type == "prompts"
        else f"We're picking which of {k} candidate models to use in production."
    )
    score_desc = _DATA_TYPE_DESCRIPTIONS[data_type]
    options = ", ".join(f'"{lbl}"' for lbl in labels) + ', or "inconclusive"'
    footer = _ANSWER_FOOTER_TEMPLATE.format(options=options, noun=noun)

    return f"""{task_desc}
I ran all {k} {noun}s ({", ".join(labels)}) against the same set of {n}
test inputs and logged the scores in `data.csv` in the current directory
(columns: item_id, entity, score -- score is {score_desc}; one row per
{noun}/item pair).

Take a look at the results and tell me which {noun} we should use.
{footer}"""
