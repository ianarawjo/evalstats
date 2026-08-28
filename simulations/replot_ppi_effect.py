"""Regenerate the PPI effect-size bias/coverage/width plot from a results CSV.

`save_ppi_effect_plot()` normally runs inside the sweep, but PPIEffectResult's
fields are exactly the columns the effect results CSV already carries, so the
figure can be rebuilt without re-simulating -- useful when a method is renamed
or the curated method set changes.

    python simulations/replot_ppi_effect.py EFFECT_RESULTS.csv --ci-comparison
"""
import argparse
import pathlib
import pandas as pd

from simulations.harness.cases.pvalues import PPIEffectResult, save_ppi_effect_plot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ci-comparison", action="store_true",
                    help="plot the curated CI-construction set instead of the textbook tests")
    ap.add_argument("--nonstandard", action="store_true")
    ap.add_argument("--regime", default="")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    results = [
        PPIEffectResult(
            name=str(r.name_), tag=str(r.tag), test=str(r.test), n=int(r.n),
            n_samples=int(r.n_samples), null_value=float(r.null_value),
            mean_bias=float(r.mean_bias), bias_z=float(r.bias_z),
            coverage=float(r.coverage), mean_ci_width=float(r.mean_ci_width),
            uncorrected_bias_z=float(r.uncorrected_bias_z),
        )
        for r in df.rename(columns={"name": "name_"}).itertuples()
    ]
    out = args.out or str(pathlib.Path(args.csv).with_suffix("").as_posix()
                          .replace("_results", "") + "_replot.png")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    p = save_ppi_effect_plot(results=results, alpha=args.alpha, out_path=out,
                             ci_comparison=args.ci_comparison,
                             nonstandard=args.nonstandard, regime=args.regime)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
