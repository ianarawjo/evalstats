"""Factorial prompt-template demo with real Ollama model calls.

Compares prompt template factors:

    * few_shot : zero_shot | concise_examples | detailed_examples
    * persona  : neutral | domain_expert | strict_reviewer

Each prompt is built from exactly three inputs:

    1) few_shot block (search factor)
    2) persona style (search factor)
    3) case_input text (actual benchmark case)

For every case and factor combination, the script calls a local Ollama model,
extracts APPROVE/REJECT/ESCALATE, computes accuracy against ground truth, and
runs factorial LMM analysis.

Requirements:
        - Install and run Ollama: `ollama serve`
        - Pull a model first, e.g. `ollama pull gemma3:1b`

Usage:

        python examples/factorial_prompt_template_variations.py
"""

import json
import re
import time
from urllib import error, request

import numpy as np
import pandas as pd

from evalstats.core.router import analyze_factorial
from evalstats.core.summary import print_analysis_summary


MODEL = "gemma3:1b"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
VALID_LABELS = {"APPROVE", "REJECT", "ESCALATE"}


def build_prompt(few_shot: str, persona: str, case_input: str) -> str:
    """Build one prompt from the 3 required inputs.

    Parameters
    ----------
    few_shot : str
        Few-shot demonstration block.
    persona : str
        Persona / instruction style.
    case_input : str
        The actual case text to analyze.
    """
    return (
        f"Persona instructions:\n{persona}\n\n"
        f"Few-shot block:\n{few_shot}\n\n"
        "Task:\n"
        "Label the case as one of: APPROVE, REJECT, ESCALATE.\n\n"
        f"Case input:\n{case_input}\n"
    )


FEW_SHOT_VARIANTS = {
    "zero_shot": "No examples; reason directly from the case details.",
    "concise_examples": (
        "Example A: All required compliance checks passed -> APPROVE.\n"
        "Example B: Missing identity document -> ESCALATE."
    ),
    "detailed_examples": (
        "Example 1: Clean background checks and full documentation -> APPROVE.\n"
        "Example 2: Confirmed sanctions match and forged records -> REJECT.\n"
        "Example 3: No sanctions hit but key files missing -> ESCALATE."
    ),
}

PERSONA_VARIANTS = {
    "neutral": "You are a careful assistant. Follow the policy exactly and be concise.",
    "domain_expert": "You are a senior risk analyst. Weigh policy evidence rigorously.",
    "strict_reviewer": "You are a strict reviewer. If critical information is missing, prefer ESCALATE.",
}

CASES = [
    (
        "Applicant provides complete documentation, no sanctions hits, "
        "stable income records, and low-risk transaction history.",
        "APPROVE",
    ),
    (
        "Applicant has a direct sanctions match and conflicting identity records "
        "across submitted documents.",
        "REJECT",
    ),
    (
        "No sanctions match found, but proof of address is missing and employer "
        "verification is pending.",
        "ESCALATE",
    ),
    (
        "All mandatory checks pass, references are verified, and past account "
        "behavior is consistent.",
        "APPROVE",
    ),
    (
        "Evidence suggests falsified bank statements and prior confirmed fraud "
        "activity.",
        "REJECT",
    ),
    (
        "Identity appears valid, but transaction purpose is unclear and a key tax "
        "document is unreadable.",
        "ESCALATE",
    ),
    (
        "Documentation is complete, compliance checklist is satisfied, and risk "
        "score is below threshold.",
        "APPROVE",
    ),
    (
        "Applicant is linked to prohibited entities in two independent screening "
        "systems.",
        "REJECT",
    ),
    (
        "Most checks are clean, but beneficial ownership information is incomplete.",
        "ESCALATE",
    ),
    (
        "Verified identity, transparent source of funds, and no adverse findings.",
        "APPROVE",
    ),
    (
        "A confirmed watchlist match and fabricated supporting letter are present.",
        "REJECT",
    ),
    (
        "No direct disqualifiers, but onboarding questionnaire contains major gaps.",
        "ESCALATE",
    ),
]


def call_model(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 128,
        },
    }
    req = request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(
            "Failed to call Ollama at http://127.0.0.1:11434. "
            f"Is `ollama serve` running and is model `{MODEL}` pulled?"
        ) from exc

    data = json.loads(body)
    return data.get("message", {}).get("content", "").strip()


def extract_label(output: str) -> str | None:
    upper = output.upper()
    for label in VALID_LABELS:
        if re.search(rf"\b{label}\b", upper):
            return label
    return None


def score_accuracy(output: str, ground_truth: str) -> float:
    predicted = extract_label(output)
    if predicted is None:
        return 0.0
    return 1.0 if predicted == ground_truth else 0.0

rows = []
total_calls = len(CASES) * len(FEW_SHOT_VARIANTS) * len(PERSONA_VARIANTS)
done = 0

print(f"Model      : {MODEL}")
print(f"Cases      : {len(CASES)}")
print(
    f"Factor grid: {len(FEW_SHOT_VARIANTS)} few_shot × "
    f"{len(PERSONA_VARIANTS)} persona"
)
print(f"Total calls: {total_calls}\n")

t0 = time.time()

for case_idx, (case_text, ground_truth) in enumerate(CASES):
    input_id = f"case_{case_idx + 1:03d}"

    for few_shot_name, few_shot_block in FEW_SHOT_VARIANTS.items():
        for persona_name, persona_text in PERSONA_VARIANTS.items():
            prompt_text = build_prompt(
                few_shot=few_shot_block,
                persona=persona_text,
                case_input=case_text,
            )
            output_text = call_model(prompt_text)
            predicted_label = extract_label(output_text)
            score = score_accuracy(output_text, ground_truth)

            done += 1
            pred_text = predicted_label if predicted_label is not None else "NONE"
            ok_mark = "✓" if score == 1.0 else "✗"
            print(
                f"[{done:3d}/{total_calls}] "
                f"case={input_id} "
                f"few_shot={few_shot_name:<17s} "
                f"persona={persona_name:<15s} "
                f"truth={ground_truth:<8s} pred={pred_text:<8s} {ok_mark}"
            )

            rows.append(
                {
                    "input_id": input_id,
                    "few_shot": few_shot_name,
                    "persona": persona_name,
                    "case_input": case_text,
                    "prompt": prompt_text,
                    "ground_truth": ground_truth,
                    "output": output_text,
                    "predicted": predicted_label,
                    "score": score,
                }
            )

elapsed = time.time() - t0
print(f"\nCompleted in {elapsed:.1f}s\n")

data = pd.DataFrame(rows)

print(
    f"Dataset: {len(data)} rows, "
    f"{data['input_id'].nunique()} cases, "
    f"{data['few_shot'].nunique()} few_shot levels × "
    f"{data['persona'].nunique()} persona levels\n"
)

bundle = analyze_factorial(
    data,
    factors=["few_shot", "persona"],
    random_effect="input_id",
    score_col="score",
    rng=np.random.default_rng(0),
)

print_analysis_summary(bundle, top_pairwise=12)
