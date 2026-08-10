"""Multi-run sensitivity analysis on real published data: AI4Math benchmark.

Loads the run-level results behind Alvarado Gonzalez et al., "Do Repetitions
Matter? Strengthening Reliability in LLM Evaluations" (arXiv:2509.24086),
which reruns 8 models x 2 prompt strategies (zero-shot / chain-of-thought)
across 105 AI4Math problems, 3 times each (a complete "seeded" R=3 design:
8 x 2 x 105 x 3 = 10,080 rows, binary correct/incorrect). The paper's whole
point is that a single run per (model, prompt) cell produces unstable
rankings -- they report that 10/12 slices invert at least one pairwise rank
relative to the 3-run majority vote.

This script re-analyzes their public run-level CSV with evalstats instead of
their custom R pipeline, to show the two things a "sensitivity analysis"
usage scenario needs:

  1. Run-to-run RELIABILITY per model: evalstats detects the repeated `run`
     column automatically (a "seeded" benchmark) and reports a per-item
     seed_std / input_std / total_std decomposition plus an "instability"
     verdict for each model -- a direct, CI-backed analogue of the paper's
     own rank-instability analysis.
  2. PROMPT SENSITIVITY: zero-shot vs. chain-of-thought, compared with paired
     CIs per model, to see whether the prompting strategy that "wins" is
     itself stable or a coin flip.

Data provenance & license note
-------------------------------
The source repository (github.com/malvarado-tech/AI4MATH_2_R_project) has no
LICENSE file, so this script does NOT vendor a copy of the CSV into the
evalstats repo. Instead it downloads the file at runtime from GitHub's raw
content host and caches it locally under examples/.cache/ (gitignored) so
repeat runs don't re-download. Delete that directory, or pass --refresh, to
force a fresh download.

Usage:
    python examples/compare_ai4math_multirun.py
    python examples/compare_ai4math_multirun.py --language EN
    python examples/compare_ai4math_multirun.py --refresh
    python examples/compare_ai4math_multirun.py --n-bootstrap 20000 --alpha 0.01
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

import evalstats as es


SOURCE_URL = (
    "https://raw.githubusercontent.com/malvarado-tech/AI4MATH_2_R_project/"
    "main/grok-4vso3-mini_cualitative_comparison/ai4math_runs.csv"
)
CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "ai4math_runs.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-run sensitivity analysis (reliability + prompt sensitivity) "
            "on the AI4Math run-level benchmark data."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=f"Path to a local copy of ai4math_runs.csv (default: download+cache at {CACHE_PATH}).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download even if a cached copy already exists.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["EN", "ES"],
        help="Restrict to one language subset (default: use both, pooled).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Global alpha for CI width (default: evalstats default, 0.05).",
    )
    parser.add_argument(
        "--top-pairwise",
        type=int,
        default=10,
        help="Number of pairwise comparisons to print per section (default: 10).",
    )
    return parser.parse_args()


def _download_csv(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SOURCE_URL}\n  -> {dest}")
    with urllib.request.urlopen(SOURCE_URL) as resp:
        dest.write_bytes(resp.read())


def _load_ai4math(csv_path: Path, refresh: bool) -> pd.DataFrame:
    if csv_path == CACHE_PATH and (refresh or not csv_path.exists()):
        _download_csv(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    required = {"problem_id", "domain", "language", "prompt", "model", "run", "correct"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    return raw_df.rename(columns={"problem_id": "item", "correct": "score"})


def main() -> None:
    args = _parse_args()

    if args.alpha is not None:
        if not (0.0 < args.alpha < 1.0):
            raise ValueError("--alpha must be strictly between 0 and 1.")
        es.set_alpha_ci(args.alpha)

    csv_path = args.csv.expanduser().resolve() if args.csv else CACHE_PATH
    df = _load_ai4math(csv_path, refresh=args.refresh)

    if args.language:
        df = df[df["language"] == args.language].copy()

    print(
        f"Rows: {len(df)} | Models: {df['model'].nunique()} | "
        f"Prompts: {df['prompt'].nunique()} | Problems: {df['item'].nunique()} | "
        f"Runs: {df['run'].nunique()} | Domains: {df['domain'].nunique()}"
        + (f" | Language: {args.language}" if args.language else " | Languages: EN+ES pooled")
    )
    print(f"Global alpha: {es.get_alpha_ci():.4f}\n")

    evaldata = es.load_from(df)
    evaldata.summary()

    # ── 1. Run-to-run reliability per model ─────────────────────────────────
    print("\n" + "=" * 78)
    print("1) RELIABILITY: how much does each model's score move between runs?")
    print(
        "   (Alvarado Gonzalez et al. 2025 found single-run rankings invert vs.\n"
        "   the 3-run majority for 10/12 slices -- the seed_std / instability\n"
        "   columns below are evalstats' CI-backed version of that same check.)"
    )
    print("=" * 78)
    reliability = es.compare(
        evaldata,
        factors="model",
        score_range=(0, 1),
        n_bootstrap=args.n_bootstrap,
    )
    reliability.summary(top_pairwise=args.top_pairwise)

    # ── 2. Prompt sensitivity: zero-shot vs. chain-of-thought ───────────────
    print("\n" + "=" * 78)
    print("2) PROMPT SENSITIVITY: zero-shot (ZS) vs. chain-of-thought (COT)")
    print("   per model, with paired CIs -- is the 'better' strategy stable")
    print("   or within the noise band?")
    print("=" * 78)
    prompt_sensitivity = es.compare(
        evaldata,
        factors="prompt",
        score_range=(0, 1),
        n_bootstrap=args.n_bootstrap,
    )
    prompt_sensitivity.summary(top_pairwise=args.top_pairwise)


if __name__ == "__main__":
    main()
