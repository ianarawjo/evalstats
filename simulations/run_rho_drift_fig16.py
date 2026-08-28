"""Official-precision rho-drift run for the paper's effect-size-invariance figure.

Sources fig:le-esinv from the RHO-DRIFT check rather than the older
label-efficiency es-invariance sub-check. The latter only ever showed "flat
lines" against a single pooled reference; the drift check measures, per effect,
both the rho^2 the realized variance multiplier implies (rho2_implied) and the
structure-appropriate score-level rho^2 it should equal (rho2_score), which is
what makes the mean-vs-rank split visible at all.

n_reps=2000 is the official tier and is NOT optional here: the check reads a
variance ratio directly, so its precision goes as sqrt(2/reps) -- ~10% relative
at 200, ~3% at 2000 -- and the sub-5% deviations that separate "invariant"
(the mean-type methods) from "drifting" sit under the noise floor at 200.
See official_args_ppi_rho_drift and RhoDriftPoint.rho2_implied_se.

Omnibus methods are excluded (only_methods): they carry bootstraps the
two-group methods don't, dominate the runtime, and appear in no panel here.
"""
import argparse, datetime as _dt, sys
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
import simulations.harness.cases.pvalues as P

FIG16_METHODS = ("ttest", "ttest_welch", "paired_t", "mwu", "wilcoxon")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=58)   # official base 42 + 16
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--progress", choices=P.PROGRESS_MODES, default="cell")
    ap.add_argument("--shape", type=str, default=None,
                    help="truth-marginal shape label; 'cont-near-center' removes the "
                         "boundary-clipping confound that makes paired_t rise ~5%%.")
    a = ap.parse_args()

    print(f"rho-drift (fig:le-esinv source) -- reps={a.reps}, n_boot={a.n_boot}, "
          f"methods={list(FIG16_METHODS)}, workers={a.workers}, "
          f"shape={a.shape or 'default'}", flush=True)
    pts, calib = P.run_ppi_rho_drift_check(
        n_reps=a.reps, n_boot=a.n_boot, seed=a.seed,
        eval_types=("continuous",), n_workers=a.workers,
        progress_mode=a.progress, only_methods=FIG16_METHODS,
        shape_label=a.shape,
    )
    P.print_ppi_rho_drift_report(pts)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    P.save_results_artifacts_ppi_rho_drift(
        pts, "simulations/out",
        f"pvalues_ppi_rho_drift_fig16_reps{a.reps}"
        f"{('_' + a.shape) if a.shape else ''}_{stamp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
