"""Generate eval data for the "Finding Your Best Prompt" tutorial notebook.

Task: support-ticket intent classification into one of four categories:
  BILLING | TECHNICAL | ACCOUNT | GENERAL

Eight prompt variants are tested against the same 40 tickets (10 per category)
using a local Ollama model.  The results are written to a flat CSV that the
notebook loads for statistical analysis.

With --runs 1 (default) the model runs at temperature=0 for a deterministic
single pass.  With --runs N > 1 the model runs at temperature=0.7 so each
repeat produces different outputs, capturing run-to-run stochasticity.

Requirements:
    - Install Ollama and start it: `ollama serve`
    - Pull the model once: `ollama pull gemma3:1b`
      (or set MODEL= to any other tag you have locally)

Usage:
    python examples/support_ticket_prompts.py
    python examples/support_ticket_prompts.py --model gemma3:1b
    python examples/support_ticket_prompts.py --runs 3
    python examples/support_ticket_prompts.py --out path/to/output.csv
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib import error, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemma3:1b"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_OUT = Path(__file__).parent.parent / "website" / "notebooks" / "best_prompt_eval.csv"

VALID_CATEGORIES = {"BILLING", "TECHNICAL", "ACCOUNT", "GENERAL"}

# ---------------------------------------------------------------------------
# Eval inputs: 40 support tickets, 10 per category
# (ticket_text, ground_truth_category)
# ---------------------------------------------------------------------------

INPUTS = [
    # --- BILLING ---
    ("I was charged twice for my subscription this month.",                          "BILLING"),
    ("My invoice shows the wrong amount — I'm on the basic plan but billed for premium.", "BILLING"),
    ("I canceled my account three weeks ago but I'm still being charged.",           "BILLING"),
    ("Can I get a refund for the past two months? I wasn't using the service.",      "BILLING"),
    ("My payment method expired but your system still charged me somehow.",          "BILLING"),
    ("I need copies of all my invoices from last year for tax purposes.",            "BILLING"),
    ("The price on your website doesn't match what I was charged.",                  "BILLING"),
    ("I upgraded my plan mid-month — how is proration calculated?",                  "BILLING"),
    ("My credit card was declined but I see a pending charge. Please clarify.",      "BILLING"),
    ("I applied a discount code at checkout but my bill wasn't reduced.",            "BILLING"),

    # --- TECHNICAL ---
    ("The app crashes every time I try to export a report.",                         "TECHNICAL"),
    ("I can't upload files larger than 10 MB even though my plan allows 100 MB.",   "TECHNICAL"),
    ("The dashboard takes over 30 seconds to load. Has something changed?",          "TECHNICAL"),
    ("I'm getting a 500 error when connecting to your API.",                         "TECHNICAL"),
    ("The iOS app keeps logging me out every few minutes.",                          "TECHNICAL"),
    ("My webhook notifications stopped working after last Tuesday's update.",        "TECHNICAL"),
    ("Search results are showing data from other users — this looks like a bug.",    "TECHNICAL"),
    ("The CSV export is missing three columns that were present last week.",         "TECHNICAL"),
    ("I set up SSO with Okta but users get a 403 error when logging in.",           "TECHNICAL"),
    ("The dark mode toggle in settings doesn't persist between sessions.",           "TECHNICAL"),

    # --- ACCOUNT ---
    ("I need to transfer my account to a different email address.",                  "ACCOUNT"),
    ("How do I add a team member to my organization?",                               "ACCOUNT"),
    ("I forgot my password and the reset email isn't arriving.",                     "ACCOUNT"),
    ("Can I merge my two accounts? I accidentally created a second one.",            "ACCOUNT"),
    ("I want to change the name and logo on my organization profile.",               "ACCOUNT"),
    ("How do I download all my data before I close my account?",                     "ACCOUNT"),
    ("My colleague left the company and I need to revoke their access immediately.", "ACCOUNT"),
    ("Can I change my username? I recently got married and use a new name.",         "ACCOUNT"),
    ("I'm locked out after too many failed login attempts. How do I regain access?", "ACCOUNT"),
    ("I need to add a secondary admin who can manage users on our team.",            "ACCOUNT"),

    # --- GENERAL ---
    ("What's the difference between your basic and premium plans?",                  "GENERAL"),
    ("Do you have a mobile app for Android?",                                        "GENERAL"),
    ("Can your platform integrate with Salesforce?",                                 "GENERAL"),
    ("I'm evaluating your product for my company — can someone schedule a demo?",   "GENERAL"),
    ("What are your data retention policies? How long do you keep my data?",         "GENERAL"),
    ("Is your service compliant with GDPR and CCPA?",                                "GENERAL"),
    ("Do you offer discounts for nonprofits?",                                       "GENERAL"),
    ("What's your uptime SLA?",                                                      "GENERAL"),
    ("Can I use your API with Python? Is there an official SDK?",                    "GENERAL"),
    ("We're a startup with 5 employees — do you have a startup pricing tier?",       "GENERAL"),
]

INPUT_IDS = [f"ticket_{i:02d}" for i in range(len(INPUTS))]

# ---------------------------------------------------------------------------
# Eight prompt variants
# ---------------------------------------------------------------------------

_CATEGORIES_LINE = "  BILLING   – questions about charges, invoices, refunds, or payments.\n" \
                   "  TECHNICAL  – bug reports, errors, performance issues, or integration problems.\n" \
                   "  ACCOUNT    – user management, login, profile settings, or data export.\n" \
                   "  GENERAL    – product questions, pricing inquiries, demos, or compliance."

PROMPTS = {
    "P0_baseline": (
        "Classify the following support ticket into exactly one category.\n"
        "Reply with a single word: BILLING, TECHNICAL, ACCOUNT, or GENERAL.\n\n"
        "Ticket: {ticket}\n\n"
        "Category:"
    ),

    "P1_role": (
        "You are a support triage specialist.\n"
        "Classify the following support ticket into exactly one category.\n"
        "Reply with a single word: BILLING, TECHNICAL, ACCOUNT, or GENERAL.\n\n"
        "Ticket: {ticket}\n\n"
        "Category:"
    ),

    "P2_definitions": (
        "You are a support triage specialist.\n"
        "Classify the following support ticket into exactly one of these categories:\n"
        f"{_CATEGORIES_LINE}\n\n"
        "Reply with a single word only.\n\n"
        "Ticket: {ticket}\n\n"
        "Category:"
    ),

    "P3_few_shot_1": (
        "You are a support triage specialist.\n"
        "Classify the following support ticket into exactly one of these categories:\n"
        f"{_CATEGORIES_LINE}\n\n"
        "Reply with a single word only.\n\n"
        "Examples:\n"
        "Ticket: I was billed twice last month.\n"
        "Category: BILLING\n\n"
        "Ticket: The API returns a 500 error on every request.\n"
        "Category: TECHNICAL\n\n"
        "Ticket: I need to add my colleague as an admin.\n"
        "Category: ACCOUNT\n\n"
        "Ticket: Do you support two-factor authentication?\n"
        "Category: GENERAL\n\n"
        "Now classify:\n"
        "Ticket: {ticket}\n\n"
        "Category:"
    ),

    "P4_few_shot_3": (
        "You are a support triage specialist.\n"
        "Classify the following support ticket into exactly one of these categories:\n"
        f"{_CATEGORIES_LINE}\n\n"
        "Reply with a single word only.\n\n"
        "Examples:\n"
        "Ticket: I was billed twice last month.\n"
        "Category: BILLING\n\n"
        "Ticket: My subscription renewed even though I canceled.\n"
        "Category: BILLING\n\n"
        "Ticket: Can I get a receipt for my last payment?\n"
        "Category: BILLING\n\n"
        "Ticket: The API returns a 500 error on every request.\n"
        "Category: TECHNICAL\n\n"
        "Ticket: The mobile app crashes when I open the settings page.\n"
        "Category: TECHNICAL\n\n"
        "Ticket: My webhook stopped receiving events after the last update.\n"
        "Category: TECHNICAL\n\n"
        "Ticket: I need to add my colleague as an admin.\n"
        "Category: ACCOUNT\n\n"
        "Ticket: How do I change the email on my account?\n"
        "Category: ACCOUNT\n\n"
        "Ticket: I'm locked out after too many wrong password attempts.\n"
        "Category: ACCOUNT\n\n"
        "Ticket: Do you support two-factor authentication?\n"
        "Category: GENERAL\n\n"
        "Ticket: What are your data residency options?\n"
        "Category: GENERAL\n\n"
        "Ticket: Do you offer an annual pricing discount?\n"
        "Category: GENERAL\n\n"
        "Now classify:\n"
        "Ticket: {ticket}\n\n"
        "Category:"
    ),

    "P5_chain_of_thought": (
        "You are a support triage specialist.\n"
        "Classify the following support ticket into exactly one of these categories:\n"
        f"{_CATEGORIES_LINE}\n\n"
        "First, briefly explain your reasoning in one sentence.\n"
        "Then on a new line write exactly: Category: <LABEL>\n\n"
        "Ticket: {ticket}"
    ),

    "P6_json_output": (
        "You are a support triage specialist.\n"
        "Classify the following support ticket into exactly one of these categories:\n"
        f"{_CATEGORIES_LINE}\n\n"
        'Respond with valid JSON only, in the form: {{"category": "<LABEL>"}}\n'
        "No other text.\n\n"
        "Ticket: {ticket}"
    ),

    "P7_negative_framing": (
        "You are a support triage specialist.\n"
        "Classify the following support ticket into exactly one of these categories:\n"
        f"{_CATEGORIES_LINE}\n\n"
        "Reply with a single word only.\n\n"
        "Important distinctions:\n"
        "- Do NOT use BILLING for account settings changes — use ACCOUNT.\n"
        "- Do NOT use ACCOUNT for payment or invoice questions — use BILLING.\n"
        "- Do NOT use TECHNICAL for questions about features or pricing — use GENERAL.\n"
        "- Do NOT use GENERAL for actual bugs or errors — use TECHNICAL.\n\n"
        "Ticket: {ticket}\n\n"
        "Category:"
    ),
}

PROMPT_IDS = list(PROMPTS.keys())

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str, temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 128},
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
            "Could not reach Ollama at http://127.0.0.1:11434. "
            "Is `ollama serve` running?"
        ) from exc
    data = json.loads(body)
    return data.get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def extract_category(output: str) -> str | None:
    """Return the first valid category label found in the output, or None."""
    # Check for JSON output first
    json_match = re.search(r'"category"\s*:\s*"([^"]+)"', output, re.IGNORECASE)
    if json_match:
        candidate = json_match.group(1).strip().upper()
        if candidate in VALID_CATEGORIES:
            return candidate

    # For chain-of-thought: look for "Category: LABEL"
    cot_match = re.search(r"Category:\s*([A-Z]+)", output, re.IGNORECASE)
    if cot_match:
        candidate = cot_match.group(1).strip().upper()
        if candidate in VALID_CATEGORIES:
            return candidate

    # Fallback: first valid label word anywhere in output
    for label in VALID_CATEGORIES:
        if re.search(rf"\b{label}\b", output.upper()):
            return label
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(model: str, out_path: Path, n_runs: int = 1) -> None:
    n_prompts = len(PROMPT_IDS)
    n_inputs = len(INPUTS)
    temperature = 0.0 if n_runs == 1 else 0.7
    total = n_prompts * n_inputs * n_runs

    print(f"Model       : {model}")
    print(f"Prompts     : {n_prompts}  ({', '.join(PROMPT_IDS)})")
    print(f"Tickets     : {n_inputs}  (10 per category)")
    print(f"Runs        : {n_runs}  (temperature={temperature})")
    print(f"Total calls : {total}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done = 0
    t0 = time.time()

    for run_idx in range(1, n_runs + 1):
        for p_id, template in PROMPTS.items():
            for i_idx, (ticket_text, ground_truth) in enumerate(INPUTS):
                prompt = template.format(ticket=ticket_text)
                output = call_ollama(prompt, model, temperature=temperature)
                predicted = extract_category(output)
                correct = 1 if predicted == ground_truth else 0

                rows.append({
                    "prompt_id":    p_id,
                    "run_idx":      run_idx,
                    "input_id":     INPUT_IDS[i_idx],
                    "ticket":       ticket_text,
                    "category":     ground_truth,
                    "output":       output,
                    "predicted":    predicted if predicted else "",
                    "correct":      correct,
                })

                done += 1
                status = "✓" if correct else "✗"
                print(
                    f"  [{done:3d}/{total}] run {run_idx}/{n_runs} | {p_id:<24s} | ticket {i_idx:02d} | "
                    f"truth={ground_truth:<9s} pred={str(predicted):<9s} {status} | "
                    f"'{output[:40]}'"
                )

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}")

    # Quick accuracy summary (averaged across runs)
    print("\nAccuracy by prompt (mean across all runs):")
    for p_id in PROMPT_IDS:
        p_rows = [r for r in rows if r["prompt_id"] == p_id]
        acc = sum(r["correct"] for r in p_rows) / len(p_rows)
        bar = "█" * round(acc * 20)
        print(f"  {p_id:<24s}  {acc:.1%}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per (prompt, ticket) pair (default: 1). "
                             "Runs > 1 use temperature=0.7 to capture stochasticity.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output CSV path (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    run(args.model, args.out, n_runs=args.runs)


if __name__ == "__main__":
    main()
