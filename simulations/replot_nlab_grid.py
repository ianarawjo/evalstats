"""Regenerate nlab_grid_heatmap.png (supptab N x N_lab figure) from a run's CSVs.

The harness only writes this figure inline, during a --plots save run
(save_ppi_nlab_grid_plot, called from the pvalues case). That means a run
executed with plots off -- as the MWU-correction reruns were -- produces the
numbers but no figure, and there was previously no way to get the figure back
without re-running the sweep. This script closes that gap: it rebuilds
PPIComparisonResult rows from the run's own CSVs, pools them across methods
exactly as the harness does (pool_ppi_comparison_across_methods), and calls
the SAME plotting function, so the output is the shipped code path rather than
a parallel reimplementation.

Usage:
    python simulations/replot_nlab_grid.py <run_dir> <run_stem> <out.png>

e.g.
    python simulations/replot_nlab_grid.py \
        simulations/out/mwu_comparison_20260905 \
        pvalues_ppi_nlab_grid_reps200_20260905_093856 \
        nlab_grid_heatmap.png
"""
from __future__ import annotations
import os, sys, warnings
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import simulations.harness.cases.pvalues as PV


def _load(run_dir: str, stem: str, kind: str) -> list:
    """One PPIComparisonResult per CSV row. Only the fields
    save_ppi_nlab_grid_plot reads (eval_type/n/n_lab/n_reps/rejects_ppi) and
    the pooler needs are populated; rate columns are converted back to the
    integer reject counts the dataclass carries."""
    path = os.path.join(run_dir, f"{stem}_{kind}_ppi_nlab_grid_results.csv")
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
    print(f"  {kind:12s} {len(out):4d} rows, methods={sorted({x.method for x in out})}")
    return out


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    run_dir, stem, out_path = sys.argv[1:4]
    cal = PV.pool_ppi_comparison_across_methods(_load(run_dir, stem, "calibration"))
    pw = PV.pool_ppi_comparison_across_methods(_load(run_dir, stem, "power"))
    print(f"  pooled: calibration={len(cal)}, power={len(pw)}")
    print("wrote", PV.save_ppi_nlab_grid_plot(
        calibration_results=cal or None, power_results=pw or None, alpha=0.05, out_path=out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
