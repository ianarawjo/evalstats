"""Regenerate the by-n coverage/score violin plots from an existing results CSV.

save_by_n_violin_plot() normally runs inside a sweep, but it only reads eight
fields per row, all of which the results CSV already carries. So the figure can
be rebuilt from a finished run instead of re-simulating -- useful when the
methods being contrasted change but the underlying sweep has not.

    python simulations/replot_ci_paired_violins.py RESULTS.csv \
        --methods bonett_price mj_floor bayes_paired_comp --eval-types binary
"""
import argparse
import pathlib

import pandas as pd

from simulations.harness.cases.ci_paired import SimResult, save_by_n_violin_plot
from simulations.harness.methods import METHODS_BY_NAME


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--eval-types", nargs="+", default=["binary"])
    ap.add_argument("--out-dir", default="simulations/out/plots")
    ap.add_argument("--stem", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df.method.isin(args.methods) & df.eval_type.isin(args.eval_types)]
    if df.empty:
        raise SystemExit(f"no rows for methods={args.methods} eval_types={args.eval_types}")
    missing = set(args.methods) - set(df.method.unique())
    if missing:
        raise SystemExit(f"methods absent from this CSV: {sorted(missing)}")

    results = [
        SimResult(
            source=str(r.source), label=str(r.label), eval_type=str(r.eval_type),
            n=int(r.n), method=str(r.method), n_reps=int(r.n_reps),
            covered=int(r.covered), total_width=float(r.total_width),
            total_score=float(r.total_score), is_null=bool(r.is_null),
        )
        for r in df.itertuples()
    ]
    stem = args.stem or pathlib.Path(args.csv).stem.replace("_results", "")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    paths = save_by_n_violin_plot(
        results=results, alpha=args.alpha,
        n_reps=int(df.n_reps.iloc[0]), out_dir=args.out_dir, run_stem=stem,
    )
    print(f"{len(results)} cells, n = {sorted(df.n.unique())}")
    for p in paths:
        print("  wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
