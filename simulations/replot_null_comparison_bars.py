"""Regenerate null_comparison_bars.png (supplementary) from a run's CSVs.

Same gap as replot_nlab_grid.py: the harness only writes this figure inline
during a --plots save run, so a sweep executed with plots off produces the
numbers but no figure, and there was no way to get it back short of re-running.
This rebuilds PPIComparisonResult rows from the run's own CSVs, pools them
across methods exactly as the harness does, and calls the SAME plotting
function -- so the output comes from the shipped code path.

The figure pools _COMPARISON_METHODS (ttest_welch/paired_t/mwu/wilcoxon), so it
moves whenever any of those estimators changes.

Usage:
    python simulations/replot_null_comparison_bars.py <run_dir> <stem> <out.png>
"""
from __future__ import annotations
import os, sys, warnings
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import simulations.harness.cases.pvalues as PV


def _rows(path: str) -> list:
    d = pd.read_csv(path)
    out = []
    for _, r in d.iterrows():
        res = PV.PPIComparisonResult(
            name=r["name"], tag=r["tag"], eval_type=r["eval_type"], n=int(r["n"]),
            n_reps=int(r["n_reps"]), effect_size=float(r["effect_size"]),
            label_frac=float(r["label_frac"]) if "label_frac" in d.columns else float("nan"),
            n_lab=int(r["n_lab"]),
        )
        res.method = r["method"]
        for col, attr in (("rate_ppi", "rejects_ppi"), ("rate_all_human", "rejects_all_human"),
                          ("rate_human_subset", "rejects_human_subset"),
                          ("rate_llm_only", "rejects_llm_only"),
                          ("rate_llm_impute", "rejects_llm_impute")):
            if col in d.columns and hasattr(res, attr):
                setattr(res, attr, int(round(float(r[col]) * int(r["n_reps"]))))
        out.append(res)
    return out


def _pooled(run_dir: str, fname: str) -> list | None:
    p = os.path.join(run_dir, fname)
    if not os.path.exists(p):
        print(f"  (absent: {fname})")
        return None
    rows = _rows(p)
    pooled = PV.pool_ppi_comparison_across_methods(rows)
    print(f"  {fname[:64]:64s} {len(rows):4d} -> {len(pooled):3d} pooled")
    return pooled


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    run, stem, out = sys.argv[1:4]
    res = _pooled(run, f"pvalues_ppi_comparison_{stem}_ppi_comparison_results.csv")
    res_b = _pooled(run, f"pvalues_ppi_comparison_binary_{stem}_ppi_comparison_results.csv")
    cal = _pooled(run, f"pvalues_ppi_nlab_grid_{stem}_calibration_ppi_nlab_grid_results.csv")
    cal_b = _pooled(run, f"pvalues_ppi_nlab_grid_binary_{stem}_calibration_ppi_nlab_grid_results.csv")
    if not res:
        print("no comparison results found", file=sys.stderr)
        return 1
    print("wrote", PV.save_ppi_null_comparison_plot(
        results=res, alpha=0.05, out_path=out,
        nlab_cal_results=cal, results_binary=res_b, nlab_cal_results_binary=cal_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
