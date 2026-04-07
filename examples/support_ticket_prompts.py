"""Generate eval data for the "Finding Your Best Prompt" tutorial notebook.

Task: support-ticket intent classification into one of four categories:
  BILLING | TECHNICAL | ACCOUNT | GENERAL

Eight prompt variants are tested against the same 120 tickets (30 per category)
using a local Ollama model.  The results are written to a flat CSV that the
notebook loads for statistical analysis.

The ticket set includes 10 straightforward examples per category plus 20 harder
"boundary" cases per category — tickets that straddle two categories or use
surface framing that misleads simpler prompts (e.g. "charged for a feature that
never worked" → BILLING not TECHNICAL; "2FA codes not arriving" → TECHNICAL not
ACCOUNT; "what's the difference between Owner and Admin?" → GENERAL not ACCOUNT).
These harder cases create real accuracy separation between prompt variants.

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
    python examples/support_ticket_prompts.py --models gemma3:1b llama3.2:3b
    python examples/support_ticket_prompts.py --reasoning
    python examples/support_ticket_prompts.py --token-estimator tiktoken
    python examples/support_ticket_prompts.py --token-estimator tiktoken --tiktoken-encoding o200k_base
    python examples/support_ticket_prompts.py --n-inputs 15
    python examples/support_ticket_prompts.py --runs 3
    python examples/support_ticket_prompts.py --out path/to/output.csv
"""

import argparse
import csv
import hashlib
import importlib
import json
import random
import re
import time
from pathlib import Path
from urllib import error, request
from typing import Any

try:
    tiktoken = importlib.import_module("tiktoken")
except ImportError:  # Optional dependency.
    tiktoken = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemma3:1b"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_OUT = Path(__file__).parent.parent / "website" / "notebooks" / "best_prompt_eval.csv"
DEFAULT_CACHE = Path(__file__).parent / "support_ticket_prompts_cache.jsonl"

VALID_CATEGORIES = {"BILLING", "TECHNICAL", "ACCOUNT", "GENERAL"}

# ---------------------------------------------------------------------------
# Eval inputs: 120 support tickets, 30 per category
# First 10 per category: straightforward cases (clear keyword signals)
# Last  20 per category: harder boundary cases (cross-category framing)
# (ticket_text, ground_truth_category)
# ---------------------------------------------------------------------------

INPUTS = [
    # ── BILLING — straightforward ──────────────────────────────────────────
    ("I was charged twice for my subscription this month.",                                       "BILLING"),
    ("My invoice shows the wrong amount — I'm on the basic plan but billed for premium.",        "BILLING"),
    ("I canceled my account three weeks ago but I'm still being charged.",                        "BILLING"),
    ("Can I get a refund for the past two months? I wasn't using the service.",                   "BILLING"),
    ("My payment method expired but your system still charged me somehow.",                        "BILLING"),
    ("I need copies of all my invoices from last year for tax purposes.",                          "BILLING"),
    ("The price on your website doesn't match what I was charged.",                                "BILLING"),
    ("I upgraded my plan mid-month — how is proration calculated?",                               "BILLING"),
    ("My credit card was declined but I see a pending charge. Please clarify.",                   "BILLING"),
    ("I applied a discount code at checkout but my bill wasn't reduced.",                          "BILLING"),

    # ── BILLING — harder boundary cases ────────────────────────────────────
    # These look like they could be TECHNICAL or ACCOUNT but the root ask is financial
    ("The feature I paid to upgrade for last month has never worked — I'd like a credit.",        "BILLING"),
    ("Our workspace was suspended for non-payment even though I sent the wire transfer two weeks ago.", "BILLING"),
    ("I see a charge from your company on my bank statement but I never signed up.",              "BILLING"),
    ("Your annual plan is listed as '$X per user per month' but my invoice total doesn't add up.", "BILLING"),
    ("We're switching from monthly to annual — will we receive a prorated credit for the remainder?", "BILLING"),
    ("My company needs to pay by bank transfer rather than credit card. Is that supported?",      "BILLING"),
    ("I upgraded mid-cycle and the charge looks different from what your pricing page describes.", "BILLING"),
    ("I was on a grandfathered plan you discontinued — you moved me to a new plan without notice.", "BILLING"),
    ("I need an official VAT invoice with our company's registered address for our accountant.",  "BILLING"),
    ("Your pricing calculator on the website gives a different total than what I was charged.",   "BILLING"),
    ("We downgraded last week, but today's invoice still reflects the old higher tier.",          "BILLING"),
    ("Our purchase order cap is lower than the renewal amount, can we adjust this before the charge runs?", "BILLING"),
    ("The receipt says annual prepay but finance sees two separate monthly charges.",              "BILLING"),
    ("I removed three seats before renewal but was billed for the original seat count.",           "BILLING"),
    ("Can you reissue last quarter's invoices under our new legal entity name?",                  "BILLING"),
    ("We were tax-exempt in your system, but sales tax appeared on this month's bill.",           "BILLING"),
    ("Our card was charged after we switched to net-30 invoice terms.",                           "BILLING"),
    ("The renewal quote from sales and the amount auto-charged by self-serve don't match.",       "BILLING"),
    ("Please split one consolidated invoice into separate invoices per department cost center.",   "BILLING"),
    ("We prepaid annually, yet the account still shows as past due.",                              "BILLING"),

    # ── TECHNICAL — straightforward ────────────────────────────────────────
    ("The app crashes every time I try to export a report.",                                       "TECHNICAL"),
    ("I can't upload files larger than 10 MB even though my plan allows 100 MB.",                 "TECHNICAL"),
    ("The dashboard takes over 30 seconds to load. Has something changed?",                        "TECHNICAL"),
    ("I'm getting a 500 error when connecting to your API.",                                       "TECHNICAL"),
    ("The iOS app keeps logging me out every few minutes.",                                        "TECHNICAL"),
    ("My webhook notifications stopped working after last Tuesday's update.",                      "TECHNICAL"),
    ("Search results are showing data from other users — this looks like a bug.",                  "TECHNICAL"),
    ("The CSV export is missing three columns that were present last week.",                       "TECHNICAL"),
    ("I set up SSO with Okta but users get a 403 error when logging in.",                         "TECHNICAL"),
    ("The dark mode toggle in settings doesn't persist between sessions.",                         "TECHNICAL"),

    # ── TECHNICAL — harder boundary cases ──────────────────────────────────
    # Look like ACCOUNT or GENERAL but the root issue is a system/code malfunction
    ("The docs say my plan includes the /batch endpoint but I keep getting 403 Forbidden.",       "TECHNICAL"),
    ("After I rotated my API key in the settings page, all our production integrations broke.",   "TECHNICAL"),
    ("Two-factor authentication codes never arrive by SMS, no matter how many times I retry.",    "TECHNICAL"),
    ("Your status page shows all green but users in Asia-Pacific can't reach the service at all.", "TECHNICAL"),
    ("The bulk CSV import returns 'completed successfully' but none of the records appear.",       "TECHNICAL"),
    ("Notification emails from your platform land in our users' spam folders — is there a fix?",  "TECHNICAL"),
    ("The analytics chart shows last month's data even when I explicitly select 'last 7 days.'",  "TECHNICAL"),
    ("I followed your official Python SDK quickstart exactly but get 401 Unauthorized every call.", "TECHNICAL"),
    ("Since the UI update last week, keyboard shortcuts I depend on daily have stopped working.", "TECHNICAL"),
    ("The audit log is missing entries for file-deletion events — we need these for compliance.", "TECHNICAL"),
    ("Password reset links open a blank page in Safari but work fine in Chrome.",                 "TECHNICAL"),
    ("Our SCIM sync says success, but no users are actually provisioned in your app.",            "TECHNICAL"),
    ("Requests to your EU endpoint intermittently time out while US endpoint requests succeed.",  "TECHNICAL"),
    ("We enabled IP allowlisting, and now even approved office IPs get blocked randomly.",        "TECHNICAL"),
    ("The desktop app auto-updated overnight and now fails to launch on Windows 11.",             "TECHNICAL"),
    ("Webhook signatures no longer validate after your latest API version rollout.",              "TECHNICAL"),
    ("CSV imports with UTF-8 characters complete, but accented names are corrupted in records.",  "TECHNICAL"),
    ("The admin UI says role changes saved, but permissions don't change until hours later.",     "TECHNICAL"),
    ("Our API usage dashboard resets to zero every morning despite continuous traffic.",           "TECHNICAL"),
    ("Session timeout is configured for 8 hours, but users are forced to re-login every 15 minutes.", "TECHNICAL"),

    # ── ACCOUNT — straightforward ───────────────────────────────────────────
    ("I need to transfer my account to a different email address.",                                "ACCOUNT"),
    ("How do I add a team member to my organization?",                                             "ACCOUNT"),
    ("I forgot my password and the reset email isn't arriving.",                                   "ACCOUNT"),
    ("Can I merge my two accounts? I accidentally created a second one.",                          "ACCOUNT"),
    ("I want to change the name and logo on my organization profile.",                             "ACCOUNT"),
    ("How do I download all my data before I close my account?",                                   "ACCOUNT"),
    ("My colleague left the company and I need to revoke their access immediately.",               "ACCOUNT"),
    ("Can I change my username? I recently got married and use a new name.",                       "ACCOUNT"),
    ("I'm locked out after too many failed login attempts. How do I regain access?",               "ACCOUNT"),
    ("I need to add a secondary admin who can manage users on our team.",                          "ACCOUNT"),

    # ── ACCOUNT — harder boundary cases ────────────────────────────────────
    # Look like TECHNICAL or GENERAL but the root ask is account/user management
    ("We need to enable SAML SSO with our identity provider — how do we get that set up?",        "ACCOUNT"),
    ("Our primary admin left the company unexpectedly. How do we transfer ownership to someone else?", "ACCOUNT"),
    ("I accidentally deleted an entire workspace that had live data — can it be restored?",       "ACCOUNT"),
    ("We want to restrict login to specific IP address ranges for our whole organization.",        "ACCOUNT"),
    ("Can we have a separate contact for invoice alerts versus security alerts on the same org?",  "ACCOUNT"),
    ("We need to split our single workspace into two separate organizations for two business units.", "ACCOUNT"),
    ("I changed my email address but now two-factor codes are still going to the old address.",   "ACCOUNT"),
    ("My API key was accidentally pushed to a public GitHub repo — I need to invalidate it now.", "ACCOUNT"),
    ("Our external auditors need read-only access to our workspace without the ability to change anything.", "ACCOUNT"),
    ("A former contractor still appears in our org. How do I remove them without losing their work?", "ACCOUNT"),
    ("Can we require all users in our org to use SSO and disable password login entirely?",        "ACCOUNT"),
    ("I need to rotate every personal access token for former employees in one place.",           "ACCOUNT"),
    ("How can I transfer ownership of specific projects to another team lead?",                    "ACCOUNT"),
    ("We need separate workspaces under one contract with isolated user directories.",             "ACCOUNT"),
    ("A user changed their name; how do we update display name without creating a new profile?",  "ACCOUNT"),
    ("Can we enforce MFA for admins only while leaving members optional?",                         "ACCOUNT"),
    ("How do I temporarily suspend a user and restore them later with the same access?",          "ACCOUNT"),
    ("Our domain changed after rebranding — how do we migrate all user logins to the new domain?", "ACCOUNT"),
    ("Can I delegate user-management permissions without granting invoice access?",                "ACCOUNT"),
    ("What's the cleanest way to offboard a department while preserving shared assets they created?", "ACCOUNT"),

    # ── GENERAL — straightforward ───────────────────────────────────────────
    ("What's the difference between your basic and premium plans?",                                "GENERAL"),
    ("Do you have a mobile app for Android?",                                                      "GENERAL"),
    ("Can your platform integrate with Salesforce?",                                               "GENERAL"),
    ("I'm evaluating your product for my company — can someone schedule a demo?",                 "GENERAL"),
    ("What are your data retention policies? How long do you keep my data?",                       "GENERAL"),
    ("Is your service compliant with GDPR and CCPA?",                                              "GENERAL"),
    ("Do you offer discounts for nonprofits?",                                                     "GENERAL"),
    ("What's your uptime SLA?",                                                                    "GENERAL"),
    ("Can I use your API with Python? Is there an official SDK?",                                  "GENERAL"),
    ("We're a startup with 5 employees — do you have a startup pricing tier?",                     "GENERAL"),

    # ── GENERAL — harder boundary cases ────────────────────────────────────
    # Look like BILLING, ACCOUNT, or TECHNICAL but are product/policy questions
    ("What's the maximum API rate limit on the business plan, and can it be raised?",             "GENERAL"),
    ("Do you support custom domains for the client-facing portal?",                                "GENERAL"),
    ("How does your per-seat pricing work if our team size changes month to month?",               "GENERAL"),
    ("If I cancel, how long is my data retained and can I export it after cancellation?",          "GENERAL"),
    ("Is there a sandbox or staging environment for testing API calls without affecting live data?", "GENERAL"),
    ("Do you have a public product roadmap showing what features are coming?",                     "GENERAL"),
    ("What's the difference between the Owner, Admin, and Member roles in your permission model?", "GENERAL"),
    ("Can we sign a custom data processing agreement with specific terms for our enterprise deal?", "GENERAL"),
    ("What types of events does your audit log capture, and how long are they retained?",          "GENERAL"),
    ("We're comparing vendors — do you have a SOC 2 report or security whitepaper we can review?", "GENERAL"),
    ("Do you support customer-managed encryption keys, and on which plans is that available?",     "GENERAL"),
    ("Is there an on-prem or private cloud deployment option for regulated industries?",            "GENERAL"),
    ("What are your documented RTO and RPO targets for disaster recovery?",                        "GENERAL"),
    ("Do you publish deprecation timelines before removing API endpoints?",                         "GENERAL"),
    ("Can enterprise contracts include data residency restricted to Canada only?",                  "GENERAL"),
    ("Do you provide a standard BA agreement process and expected security review timeline?",       "GENERAL"),
    ("Which identity providers are officially supported for SSO and SCIM integrations?",           "GENERAL"),
    ("Is audit-log retention configurable per workspace, and what are the plan limits?",           "GENERAL"),
    ("Do you offer professional services for migration from a legacy ticketing platform?",         "GENERAL"),
    ("Are there contractual penalties or credits if monthly uptime falls below SLA commitments?",  "GENERAL"),

    # Adding even more cases, with broken grammar / English and typos
    # ── BILLING — noisy / shorthand / angry ───────────────────────────
    ("charged twice again this month fix this", "BILLING"),
    ("invoice wrong amount says premium im not on that plan", "BILLING"),
    ("cancelled weeks ago why still charging me???", "BILLING"),
    ("need refund wasnt even usind this", "BILLING"),
    ("card expired but still got charged how", "BILLING"),
    ("price on site not what i paid explain", "BILLING"),
    ("used discount code didnt apply wtf", "BILLING"),
    ("upgraded mid month math not adding up on bill", "BILLING"),
    ("why pending charge when payment failed", "BILLING"),
    ("need all invoice from last year asap taxes", "BILLING"),


    # ── TECHNICAL — noisy / shorthand / angry ─────────────────────────
    ("app keeps crashing when exporting fix pls", "TECHNICAL"),
    ("cant upload files over 10mb even tho plan allows it", "TECHNICAL"),
    ("dashboard slow like 30s load what changed", "TECHNICAL"),
    ("api just returns 500 now nothing works", "TECHNICAL"),
    ("ios app logs me out every few mins super annoying", "TECHNICAL"),
    ("webhooks stopped after last update ???", "TECHNICAL"),
    ("seeing other users data this is bad", "TECHNICAL"),
    ("csv export missing columns again", "TECHNICAL"),
    ("sso setup okta but users get 403", "TECHNICAL"),
    ("dark mode resets every time why", "TECHNICAL"),


    # ── ACCOUNT — noisy / shorthand / angry ───────────────────────────
    ("need change account email asap", "ACCOUNT"),
    ("how add new team member??", "ACCOUNT"),
    ("password reset email not coming", "ACCOUNT"),
    ("made 2 accounts by mistake can u merge", "ACCOUNT"),
    ("change org name and logo where", "ACCOUNT"),
    ("how download all data before closing acct", "ACCOUNT"),
    ("remove user they left company urgent", "ACCOUNT"),
    ("want change username got new name", "ACCOUNT"),
    ("locked out too many attempts help", "ACCOUNT"),
    ("want add another admin with permissions", "ACCOUNT"),


    # ── GENERAL — noisy / shorthand / angry ───────────────────────────
    ("difference basic vs premium??", "GENERAL"),
    ("android app exist or no", "GENERAL"),
    ("does this integrate with salesforce", "GENERAL"),
    ("yo i need a demo for company who to talk to", "GENERAL"),
    ("how long u keep data retention policy?", "GENERAL"),
    ("bro you gdpr compliant or what?", "GENERAL"),
    ("nonprofit discount available?", "GENERAL"),
    ("uptime sla details??", "GENERAL"),
    ("api python sdk official or not", "GENERAL"),
    ("startup pricing for small team??", "GENERAL"),
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

def estimate_token_count_heuristic(text: str) -> int:
    """Rough token estimate from text (fast heuristic, not model-exact)."""
    if not text:
        return 0
    # Split into words and punctuation-like chunks for a lightweight estimate.
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


_TIKTOKEN_ENCODER_CACHE: dict[tuple[str, str], Any] = {}


def _resolve_tiktoken_encoder(model: str, fallback_encoding: str = "cl100k_base") -> tuple[Any | None, str | None]:
    """Resolve a tiktoken encoder for model names that may include Ollama tags."""
    if tiktoken is None:
        return None, None

    cache_key = (model, fallback_encoding)
    if cache_key in _TIKTOKEN_ENCODER_CACHE:
        return _TIKTOKEN_ENCODER_CACHE[cache_key], "model"

    base_model = model.split(":", 1)[0]
    try:
        encoder = tiktoken.encoding_for_model(base_model)
        _TIKTOKEN_ENCODER_CACHE[cache_key] = encoder
        return encoder, "model"
    except KeyError:
        try:
            encoder = tiktoken.get_encoding(fallback_encoding)
            _TIKTOKEN_ENCODER_CACHE[cache_key] = encoder
            return encoder, f"fallback:{fallback_encoding}"
        except KeyError:
            return None, None


def estimate_token_count(
    text: str,
    model: str,
    estimator: str = "auto",
    tiktoken_encoding: str = "o200k_base",
) -> tuple[int, str]:
    """Return token estimate and method label for auditability in output CSV."""
    if not text:
        return 0, "empty"

    if estimator in {"auto", "tiktoken"}:
        encoder, source = _resolve_tiktoken_encoder(model, fallback_encoding=tiktoken_encoding)
        if encoder is not None:
            return len(encoder.encode(text)), f"tiktoken:{source}"
        if estimator == "tiktoken":
            return estimate_token_count_heuristic(text), "heuristic(no-tiktoken)"

    return estimate_token_count_heuristic(text), "heuristic"


def _cache_key_payload(
    model: str,
    ticket_text: str,
    prompt_id: str,
    prompt: str,
    run_idx: int,
    temperature: float,
    include_reasoning: bool,
    num_predict: int | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "ticket_text": ticket_text,
        "prompt_id": prompt_id,
        # Include prompt hash so cache automatically invalidates if prompt text changes.
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "run_idx": run_idx,
        "temperature": temperature,
        "include_reasoning": include_reasoning,
        "num_predict": num_predict,
    }


def _cache_key(cache_payload: dict[str, Any]) -> str:
    return json.dumps(cache_payload, sort_keys=True, separators=(",", ":"))


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load line-delimited JSON cache into memory for fast key lookup."""
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache

    with cache_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[warn] Skipping malformed cache row at line {line_no} in {cache_path}")
                continue

            key = row.get("key")
            response = row.get("response")
            if isinstance(key, str) and isinstance(response, dict):
                cache[key] = response

    return cache


def append_cache_entry(cache_path: Path, key: str, request_payload: dict[str, Any], response: dict[str, Any]) -> None:
    """Append one response to cache immediately so partial progress survives crashes."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "request": request_payload,
        "response": response,
        "cached_at_unix": time.time(),
    }
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def call_ollama(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    include_reasoning: bool = False,
    num_predict: int | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": temperature}
    if num_predict is not None:
        options["num_predict"] = num_predict

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    }
    reasoning_fallback = False

    def _post(body_payload: dict[str, Any]) -> dict[str, Any]:
        req = request.Request(
            OLLAMA_CHAT_URL,
            data=json.dumps(body_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.code}. "
                f"Response body: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama at http://127.0.0.1:11434. "
                "Is `ollama serve` running?"
            ) from exc

        return json.loads(body)

    if include_reasoning:
        try:
            data = _post({**payload, "think": True})
        except RuntimeError as exc:
            # Some Ollama/model combinations reject `think`; auto-fallback to normal output.
            if "HTTP 400" in str(exc):
                data = _post(payload)
                reasoning_fallback = True
            else:
                raise
    else:
        data = _post(payload)

    message = data.get("message", {})
    output_text = str(message.get("content", "")).strip()
    reasoning_text = str(
        message.get("thinking", message.get("reasoning", data.get("thinking", "")))
    ).strip()
    return {
        "output": output_text,
        "reasoning": reasoning_text,
        "reasoning_fallback": reasoning_fallback,
        "total_duration_ms": round((int(data.get("total_duration", 0) or 0)) / 1_000_000, 2),
        "prompt_eval_count": int(data.get("prompt_eval_count", 0) or 0),
        "eval_count": int(data.get("eval_count", 0) or 0),
    }


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


def select_inputs(n_inputs: int | None, seed: int | None = None) -> list[tuple[int, str, str]]:
    """Select up to n_inputs tickets, sampled as evenly as possible across classes.

    Returns rows as (original_index, ticket_text, ground_truth_category).
    """
    all_rows = [(i, ticket, category) for i, (ticket, category) in enumerate(INPUTS)]
    if n_inputs is None:
        return all_rows

    if n_inputs < 1 or n_inputs > len(INPUTS):
        raise ValueError(f"n_inputs must be between 1 and {len(INPUTS)}")

    rng = random.Random(seed)
    by_category: dict[str, list[tuple[int, str, str]]] = {c: [] for c in VALID_CATEGORIES}
    for row in all_rows:
        by_category[row[2]].append(row)

    for rows in by_category.values():
        rng.shuffle(rows)

    categories = sorted(VALID_CATEGORIES)
    n_categories = len(categories)
    base = n_inputs // n_categories
    remainder = n_inputs % n_categories

    per_category = {c: base for c in categories}
    extra_order = categories[:]
    rng.shuffle(extra_order)
    for c in extra_order[:remainder]:
        per_category[c] += 1

    selected: list[tuple[int, str, str]] = []
    for c in categories:
        need = per_category[c]
        available = len(by_category[c])
        if need > available:
            raise ValueError(
                f"Requested {need} examples for {c}, but only {available} are available"
            )
        selected.extend(by_category[c][:need])

    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    models: list[str],
    out_path: Path,
    n_runs: int = 1,
    n_inputs: int | None = None,
    seed: int | None = None,
    include_reasoning: bool = False,
    token_estimator: str = "auto",
    tiktoken_encoding: str = "o200k_base",
    num_predict: int | None = None,
    cache_path: Path | None = DEFAULT_CACHE,
) -> None:
    n_prompts = len(PROMPT_IDS)
    selected_inputs = select_inputs(n_inputs=n_inputs, seed=seed)
    n_selected_inputs = len(selected_inputs)
    temperature = 0.0 if n_runs == 1 else 0.7
    total = len(models) * n_prompts * n_selected_inputs * n_runs

    print(f"Models      : {', '.join(models)}")
    print(f"Prompts     : {n_prompts}  ({', '.join(PROMPT_IDS)})")
    if n_inputs is None:
        print(f"Tickets     : {n_selected_inputs}  (30 per category: 10 easy + 20 hard boundary cases)")
    else:
        print(f"Tickets     : {n_selected_inputs} sampled evenly across categories")
    print(f"Runs        : {n_runs}  (temperature={temperature})")
    print(f"num_predict : {num_predict if num_predict is not None else 'default (unset)'}")
    print(f"Reasoning   : {'enabled' if include_reasoning else 'disabled'}")
    print(f"Token est.  : {token_estimator} (tiktoken encoding fallback={tiktoken_encoding})")
    if cache_path is None:
        print("Cache       : disabled")
        cache: dict[str, dict[str, Any]] = {}
    else:
        cache = load_cache(cache_path)
        print(f"Cache       : {cache_path}  ({len(cache)} entries loaded)")
    print(f"Total calls : {total}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done = 0
    t0 = time.time()

    for model in models:
        warned_reasoning_fallback = False
        for run_idx in range(1, n_runs + 1):
            for p_id, template in PROMPTS.items():
                for orig_idx, ticket_text, ground_truth in selected_inputs:
                    prompt = template.format(ticket=ticket_text)
                    cache_payload = _cache_key_payload(
                        model=model,
                        ticket_text=ticket_text,
                        prompt_id=p_id,
                        prompt=prompt,
                        run_idx=run_idx,
                        temperature=temperature,
                        include_reasoning=include_reasoning,
                        num_predict=num_predict,
                    )
                    key = _cache_key(cache_payload)

                    from_cache = False
                    if cache_path is not None and key in cache:
                        response = cache[key]
                        from_cache = True
                    else:
                        response = call_ollama(
                            prompt,
                            model,
                            temperature=temperature,
                            include_reasoning=include_reasoning,
                            num_predict=num_predict,
                        )
                        if cache_path is not None:
                            append_cache_entry(cache_path, key=key, request_payload=cache_payload, response=response)
                            cache[key] = response

                    output = response["output"]
                    reasoning = response["reasoning"]

                    if include_reasoning and response.get("reasoning_fallback") and not warned_reasoning_fallback:
                        print(
                            f"[warn] Reasoning output is not supported by Ollama model {model}. "
                            "Auto-fallback is active (continuing without reasoning traces)."
                        )
                        warned_reasoning_fallback = True

                    predicted = extract_category(output)
                    correct = 1 if predicted == ground_truth else 0

                    # We don't have exact token counts for reasoning tokens from Ollama,
                    # so we need to estimate them, we do this using tiktoken...
                    reasoning_tokens_est, reasoning_token_method = estimate_token_count(
                        reasoning,
                        model,
                        estimator=token_estimator,
                        tiktoken_encoding=tiktoken_encoding,
                    )

                    # We do have exact eval counts from Ollama for the output tokens,
                    # but these are only for output, not reasoning. 
                    output_tokens_est = response["eval_count"]
                    output_token_method = "ollama_eval_count"
                    total_reasoning_output_tokens_est = reasoning_tokens_est + output_tokens_est

                    rows.append({
                        "model":       model,
                        "prompt_id":   p_id,
                        "run_idx":     run_idx,
                        "input_id":    INPUT_IDS[orig_idx],
                        "ticket":      ticket_text,
                        "category":    ground_truth,
                        "reasoning":   reasoning,
                        "output":      output,
                        "predicted":   predicted if predicted else "",
                        "correct":     correct,
                        "reasoning_tokens_est": reasoning_tokens_est,
                        "reasoning_token_method": reasoning_token_method,
                        "output_tokens_est": output_tokens_est,
                        "output_token_method": output_token_method,
                        "total_reasoning_output_tokens_est": total_reasoning_output_tokens_est,
                        "total_duration_ms": response["total_duration_ms"],
                        "prompt_eval_count": response["prompt_eval_count"],
                        "eval_count": response["eval_count"],
                    })

                    done += 1
                    status = "✓" if correct else "✗"
                    print(
                        f"  [{done:3d}/{total}] model={model:<16s} | run {run_idx}/{n_runs} | {p_id:<24s} | ticket {orig_idx:02d} | "
                        f"truth={ground_truth:<9s} pred={str(predicted):<9s} {status} | "
                        f"tok~{total_reasoning_output_tokens_est:<4d} | {'[cache] ' if from_cache else ''}'{output[:40]}'"
                    )

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}")

    # Quick accuracy summary (averaged across runs)
    print("\nAccuracy by model and prompt (mean across all runs):")
    for model in models:
        print(f"  {model}:")
        for p_id in PROMPT_IDS:
            p_rows = [r for r in rows if r["model"] == model and r["prompt_id"] == p_id]
            acc = sum(r["correct"] for r in p_rows) / len(p_rows)
            bar = "█" * round(acc * 20)
            print(f"    {p_id:<24s}  {acc:.1%}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Single Ollama model tag (default: {DEFAULT_MODEL}). Ignored when --models is provided.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="One or more Ollama model tags. Example: --models gemma3:1b llama3.2:3b")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per (prompt, ticket) pair (default: 1). "
                             "Runs > 1 use temperature=0.7 to capture stochasticity.")
    parser.add_argument("--n-inputs", type=int, default=None,
                        help="Optional number of tickets to evaluate. "
                             "Randomly sampled as evenly as possible across categories.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for --n-inputs sampling.")
    parser.add_argument("--reasoning", action="store_true",
                        help="Request reasoning traces from models that support thinking output. "
                             "Adds reasoning and token-estimate columns to output rows.")
    parser.add_argument("--num-predict", type=int, default=None,
                        help="Optional max generated tokens per call. Unset by default.")
    parser.add_argument("--token-estimator", choices=["auto", "tiktoken", "heuristic"], default="auto",
                        help="Token estimation method. 'auto' prefers tiktoken when available and falls back to heuristic.")
    parser.add_argument("--tiktoken-encoding", default="o200k_base",
                        help="Fallback tiktoken encoding when model name is unknown to tiktoken (default: o200k_base).")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE,
                        help=f"JSONL cache path for incremental per-call persistence (default: {DEFAULT_CACHE}).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable read/write cache behavior.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output CSV path (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    models = args.models if args.models else [args.model]
    cache_path = None if args.no_cache else args.cache_path
    run(
        models,
        args.out,
        n_runs=args.runs,
        n_inputs=args.n_inputs,
        seed=args.seed,
        include_reasoning=args.reasoning,
        token_estimator=args.token_estimator,
        tiktoken_encoding=args.tiktoken_encoding,
        num_predict=args.num_predict,
        cache_path=cache_path,
    )


if __name__ == "__main__":
    main()
