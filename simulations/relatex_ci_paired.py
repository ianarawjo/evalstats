"""Regenerate ci_paired's LaTeX overall-summary table from a finished results CSV.

`latex_overall_summary()` normally runs inside a sweep behind --latex, but it
reads only the fields the results CSV already carries, so the table can be
rebuilt from a completed run instead of re-simulating. Useful when the method
set changes but the underlying sweep has not, or when --latex was not passed.

    python simulations/relatex_ci_paired.py RESULTS.csv > table.tex
    python simulations/relatex_ci_paired.py RESULTS.csv --eval-types binary
"""
import argparse
import pandas as pd

from simulations.harness.cases.ci_paired import SimResult, latex_overall_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--eval-types", nargs="+", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["method"] = df["method"].astype(str)
    if args.methods:
        df = df[df.method.isin(args.methods)]
    if args.eval_types:
        df = df[df.eval_type.isin(args.eval_types)]
    if df.empty:
        raise SystemExit("no rows after filtering")

    results = [
        SimResult(
            source=str(r.source), label=str(r.label), eval_type=str(r.eval_type),
            n=int(r.n), method=str(r.method), n_reps=int(r.n_reps),
            covered=int(r.covered), total_width=float(r.total_width),
            total_score=float(r.total_score), is_null=bool(r.is_null),
            # the CSV carries these too; dropping them silently zeroes the
            # Penalty / Type-I / Power / Time columns in the rebuilt table
            total_pen_under=float(r.mean_pen_under) * int(r.n_reps),
            total_pen_over=float(r.mean_pen_over) * int(r.n_reps),
            rejects=int(r.rejects),
            total_time=float(r.total_time),
            total_time_sq=float(r.total_time_sq),
        )
        for r in df.itertuples()
    ]
    print(latex_overall_summary(results, args.alpha, int(df.n_reps.iloc[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
