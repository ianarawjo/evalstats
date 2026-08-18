"""Regenerate every label-efficiency figure from a finished run's CSVs.

A full sweep costs hours; the figures cost seconds. Whenever plotting changes
-- a fixed legend, a marker convention, a new panel layout -- this rebuilds
the complete figure set from artifacts already on disk, with no re-simulation.

It reconstructs the two in-memory shapes the plot functions expect:

  * LabelEfficiencyPoint list, from *_results.csv          -> pooled figures
  * PPIComparisonResult list + calib_rows, from            -> per-method figures,
    *_raw_results.csv and *_calibration.csv                   per-method table,
                                                              noise-family figure

The per-method path re-inverts power curves, so it needs the reference-curve
cache warm at the same ref_n_mc/seed the run used (defaults here match the
CLI's: ref_n_mc=3000, seed = --seed + 14 = 56). With a warm cache the whole
regeneration is a few seconds; cold it would rebuild curves, so pass the same
values the sweep used rather than letting them drift.

Usage:
    python -m simulations.replot_label_efficiency --run-dir simulations/out/labeleff_noisefamily
    python -m simulations.replot_label_efficiency --run-dir DIR --out-dir /tmp/figs
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from simulations.harness.cases.pvalues import (
    LabelEfficiencyPoint,
    PPIComparisonResult,
    _pool_label_eff_across_es,
    save_ppi_label_efficiency_noise_family_plot,
    save_ppi_label_efficiency_per_method_table,
    save_ppi_label_efficiency_plots,
    save_ppi_label_efficiency_plots_per_method,
)


def _one(run_dir: str, suffix: str) -> str:
    hits = sorted(glob.glob(f"{run_dir}/*{suffix}"))
    if not hits:
        raise SystemExit(f"no *{suffix} in {run_dir}")
    return hits[-1]  # newest by timestamped name


def _points(results_csv: str) -> list[LabelEfficiencyPoint]:
    df = pd.read_csv(results_csv)
    return [LabelEfficiencyPoint(
        eval_type=t.eval_type, judge_noise=t.judge_noise, alignment_metric=t.alignment_metric,
        alignment_target=t.alignment_target, alignment_value=t.alignment_value, n_lab=t.n_lab,
        ppi_power=t.ppi_power, equiv_n_lab=t.equiv_n_lab, n_reps=t.n_reps, saturated=t.saturated,
        effect_frac=t.effect_frac, mult_lo=t.multiplier_lo, mult_hi=t.multiplier_hi, rho2=t.rho2,
        predicted_mult=t.predicted_mult, inversion_ratio=t.inversion_ratio,
        inversion_clamped=t.inversion_clamped,
        noise_family=getattr(t, "noise_family", "gaussian"),
    ) for t in df.itertuples()]


def _raw_and_calib(raw_csv: str, calib_csv: str):
    raw = [PPIComparisonResult(
        name=t.name, tag=t.tag, eval_type=t.eval_type, n=t.n, n_reps=t.n_reps,
        effect_size=t.effect_size, label_frac=t.label_frac, n_lab=t.n_lab, method=t.method,
        # the CSV stores RATES; the dataclass wants counts
        rejects_all_human=int(round(t.rate_all_human * t.n_reps)),
        rejects_human_subset=int(round(t.rate_human_subset * t.n_reps)),
        rejects_llm_only=int(round(t.rate_llm_only * t.n_reps)),
        rejects_llm_impute=int(round(t.rate_llm_impute * t.n_reps)),
        rejects_ppi=int(round(t.rate_ppi * t.n_reps)), n_failed=t.n_failed,
    ) for t in pd.read_csv(raw_csv).itertuples()]

    cdf = pd.read_csv(calib_csv)
    fixed = {"eval_type", "noise_family", "judge_noise", "alignment_metric",
             "alignment_target", "alignment_achieved"}
    extra = [c for c in cdf.columns if c not in fixed]
    calib = []
    for t in cdf.itertuples():
        panel = {c: float(getattr(t, c)) for c in extra
                 if pd.notna(getattr(t, c, None))}
        calib.append((t.eval_type, t.judge_noise, t.alignment_metric, t.alignment_target,
                      t.alignment_achieved, panel, getattr(t, "noise_family", "gaussian")))
    return raw, calib


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/plots_replot")
    ap.add_argument("--ref-n-mc", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=56, help="the sweep's --seed + 14")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir or f"{args.run_dir}/plots_replot")
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / "labeleff.png")

    pts = _points(_one(args.run_dir, "_ppi_label_efficiency_results.csv"))
    raw, calib = _raw_and_calib(_one(args.run_dir, "_ppi_label_efficiency_raw_results.csv"),
                                _one(args.run_dir, "_ppi_label_efficiency_calibration.csv"))
    written: list[str] = []

    written += save_ppi_label_efficiency_plots(pts, out_path=stem)

    pm_paths, pm_points = save_ppi_label_efficiency_plots_per_method(
        raw, calib, out_path=stem, ref_n_mc=args.ref_n_mc, seed=args.seed)
    written += pm_paths
    if pm_points:
        written.append(save_ppi_label_efficiency_per_method_table(
            pm_points, out_dir=str(out), run_stem="labeleff"))
        try:
            written.append(save_ppi_label_efficiency_noise_family_plot(
                pm_points, str(out / "labeleff_plot_noisefamily.png")))
        except Exception as exc:
            print(f"  (noise-family figure skipped: {type(exc).__name__}: {exc})")

    # How many cells the figures omit, for the caption -- the plots no longer
    # draw a marker for them (see save_ppi_label_efficiency_plot), so the count
    # has to come from somewhere it can be explained.
    usable = sum(1 for p in pts if not p.saturated and p.well_conditioned)
    print(f"\nwrote {len(written)} artifacts to {out}")
    print(f"caption figure: {usable}/{len(pts)} pooled cells reported "
          f"({usable/len(pts)*100:.1f}%); the rest fail the inversion "
          f"conditioning check and are omitted rather than drawn.")


if __name__ == "__main__":
    main()
