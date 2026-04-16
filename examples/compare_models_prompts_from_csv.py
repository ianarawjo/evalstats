"""Load a saved multi-model x prompt benchmark CSV and analyze it with evalstats.

This example compares models and prompts from a pre-collected dataset rather than
calling any model APIs live.

Dataset default:
    website/notebooks/ticket-multi-model-x-prompt.csv

Expected input columns in the CSV:
    model, prompt_id, run_idx, input_id, correct

Usage:
    python examples/compare_models_prompts_from_csv.py

Optional:
    python examples/compare_models_prompts_from_csv.py --csv path/to/file.csv
    python examples/compare_models_prompts_from_csv.py --n-bootstrap 3000
    python examples/compare_models_prompts_from_csv.py --method auto
    python examples/compare_models_prompts_from_csv.py --alpha 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import evalstats as estats


DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "website"
    / "notebooks"
    / "ticket-multi-model-x-prompt-3models.csv"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare models and prompts from a long-form benchmark CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to benchmark CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples (default: 3000)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=[
            "auto",
            "bootstrap",
            "bca",
            "bayes_bootstrap",
            "smooth_bootstrap",
            "permutation",
            "sign_test",
            "lmm",
            "wilson",
            "newcombe",
            "fisher_exact",
        ],
        help="Statistical method passed to evalstats.analyze(...).",
    )
    parser.add_argument(
        "--correction",
        type=str,
        default="holm",
        choices=["holm", "bonferroni", "fdr_bh", "none"],
        help="Multiple-comparisons correction (default: holm)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Global alpha for confidence level and significance threshold "
            "(for example 0.01 for 99%% CI). Uses evalstats default if omitted."
        ),
    )
    return parser.parse_args()


def _prepare_long_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "prompt_id", "run_idx", "input_id", "correct"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing)
            + "\nExpected at least: "
            + ", ".join(sorted(required))
        )

    long_df = raw_df.rename(
        columns={
            "prompt_id": "prompt",
            "run_idx": "run",
            "input_id": "input",
            "correct": "score",
        }
    )[["model", "prompt", "input", "run", "score"]].copy()

    # Ensure numeric score for robust parsing by evalstats.from_dataframe.
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
    return long_df


def main() -> None:
    args = _parse_args()

    if args.alpha is not None:
        if not (0.0 < args.alpha < 1.0):
            raise ValueError("--alpha must be strictly between 0 and 1.")
        estats.set_alpha_ci(args.alpha)

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    long_df = _prepare_long_df(raw_df)

    print(
        f"Rows: {len(long_df)} | Models: {long_df['model'].nunique()} | "
        f"Prompts: {long_df['prompt'].nunique()} | Inputs: {long_df['input'].nunique()} | "
        f"Runs: {long_df['run'].nunique()}"
    )
    print(f"Global alpha: {estats.get_alpha_ci():.4f}")

    benchmark, load_report = estats.from_dataframe(
        long_df,
        format="long",
        repair=True,
        strict_complete_design=True,
        return_report=True,
    )

    print("\n=== Load report ===")
    for line in load_report.to_lines():
        print(" ", line)

    print("\n=== analyze(...): models + prompts ===")
    analysis = estats.analyze(
        benchmark,
        evaluator_mode="aggregate",
        reference="grand_mean",
        method=args.method,
        n_bootstrap=args.n_bootstrap,
        correction=args.correction,
    )
    estats.print_analysis_summary(analysis, top_pairwise=10)


if __name__ == "__main__":
    main()