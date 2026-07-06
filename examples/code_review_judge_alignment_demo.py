"""Workshop demo: does the LLM judge actually measure code-review quality?

Scenario: 10 code-review agent models each reviewed the same 120 pull
request diffs. An LLM judge scored review quality on a 1-5 scale for every
(model, item) pair. A subset of 40 items also has human gold labels for
all 10 models.

Walkthrough:
  1. Trust the judge -- compare models on the raw LLM judge scores.
  2. Question the judge -- check whether judge scores track something
     they shouldn't (response length).
  3. Validate the judge against the human labels we do have.
  4. Re-run the SAME compare() call with alignment= to get PPI-corrected
     means, CIs, and p-values that account for the judge's bias.

Data: code_review_judge_alignment_demo.csv (generate with
generate_code_review_alignment_demo.py). The true model quality and bias
assignments used to generate this data are NOT used below -- this script
only sees what an evaluator would actually have in practice.

Usage:
    python examples/code_review_judge_alignment_demo.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

import evalstats as es

CSV_PATH = Path(__file__).parent / "code_review_judge_alignment_demo.csv"


def _banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    evaldata = es.load_from(df)

    # ── Step 1: trust the judge ──────────────────────────────────────────────
    _banner("STEP 1 — Compare models, trusting the LLM judge metric as-is")
    result = es.compare(evaldata, factors="model", metric="llm_judge_score")
    result.summary()

    # ── Step 2: question the judge ───────────────────────────────────────────
    _banner("STEP 2 — Sanity check: does the judge score track response length?")
    r, p = pearsonr(df["response_length"], df["llm_judge_score"])
    print(f"corr(response_length, llm_judge_score) = {r:.3f}  (p = {p:.2e})")
    print("\nMean response length by model (longer != better review):")
    by_model = (
        df.groupby("model")
        .agg(mean_judge_score=("llm_judge_score", "mean"),
             mean_response_length=("response_length", "mean"))
        .sort_values("mean_judge_score", ascending=False)
    )
    print(by_model.to_string())
    print(
        "\nLength shouldn't drive a code-review quality score on its own -- "
        "worth checking the judge against real human ratings before trusting it."
    )

    # ── Step 3: validate the judge against the human labels we have ─────────
    _banner("STEP 3 — Validate the judge against the 40-item human gold set")
    alignment = es.validate_alignment(
        evaldata,
        llm_metric="llm_judge_score",
        human_groundtruth="human_score",
    )
    alignment.summary()

    # ── Step 4: re-run the SAME compare() call with alignment= ──────────────
    _banner("STEP 4 — Re-run compare() with alignment=, correcting for judge bias")
    result_corrected = es.compare(
        evaldata,
        factors="model",
        metric="llm_judge_score",
        alignment={"llm_judge_score": alignment},
    )
    result_corrected.summary()

    # ── Bonus: which models did the judge over/under-rate? ───────────────────
    _banner("BONUS — PPI rectifier: which models did the judge get wrong?")
    entities = result_corrected.to_dict()["variance_components"]["entities"]
    rows = sorted(entities.items(), key=lambda kv: kv[1]["rectifier"])
    print(f"{'Model':<16} {'Judge-only mean':>16} {'Corrected mean':>16} {'Rectifier':>12}")
    for name, e in rows:
        print(
            f"{name:<16} {e['llm_mean']:>16.3f} {e['ppi_mean']:>16.3f} "
            f"{e['rectifier']:>+12.3f}"
        )
    print(
        "\nA negative rectifier means the judge overrated that model relative "
        "to its human-labeled items; positive means the judge underrated it."
    )


if __name__ == "__main__":
    main()
