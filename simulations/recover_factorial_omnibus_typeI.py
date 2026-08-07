"""Recovery script (2026-08-06): the official --mode ppi run
(official_args_ppi, NOT the standalone official_args_ppi_factorial preset)
never sets factorial_omnibus=True, so its 12-hour --factorial-check pass
only ever computed the 5 two-group/paired methods (_COMPARISON_METHODS:
ttest, ttest_welch, paired_t, mwu, wilcoxon) against the ~6798-scenario
factorial grid -- the 4 omnibus methods (_COMPARISON_METHODS_OMNIBUS:
anova_ind, anova_rep, friedman, kruskal) were simply never run, so
save_ppi_factorial_typeI_violin_plot's saved PNGs only ever show 5 of the
9 tests appendix_mockup.tex expects (see its typeI-factorial figure note).

Re-running the full official pass just to add the omnibus arm would repeat
all 5 already-computed methods (plus the unrelated effect/comparison/
factorial_binary/label_efficiency checks the 12 hours also paid for) for
nothing. Instead, this script:

  1. Rebuilds the EXACT SAME factorial_sources (build_ppi_factorial_sources
     with the same likert_max/noise_levels the official run used --
     confirmed via the existing CSV's filename, which has no "_lmax"/
     "_fastnoise" suffix, i.e. the defaults) and runs ONLY the 4 omnibus
     methods against them via run_ppi_comparison_simulation(methods=
     _COMPARISON_METHODS_OMNIBUS), at the same reps/n_boot precision tier
     (--reps/--n-boot default to the values official_args_ppi used:
     effect_reps=200, ppi_n_boot=2000 -- override with --reps/--n-boot if
     the original run used something else).
  2. Saves that omnibus-only raw data to its own CSV/log via the harness's
     own save_results_artifacts_ppi_factorial, exactly the way run()'s
     `if factorial_omnibus:` branch would have (see pvalues.py's run(),
     the factorial_check block, second save_results_artifacts_ppi_factorial
     call) -- so the new data is saved in the SAME schema/format as if the
     official run had done this from the start.
  3. Loads the ORIGINAL run's already-saved two-group CSV (7MB, ~34k rows)
     back into lightweight objects carrying just the 5 attributes
     save_ppi_factorial_typeI_violin_plot actually reads (.name, .method,
     .n_reps, .rejects_llm_only, .rejects_ppi) -- no need to reconstruct a
     full PPIComparisonResult, since that's all this specific plot touches
     (confirmed by reading its body).
  4. Calls save_ppi_factorial_typeI_violin_plot(two_group_results=<step 3>,
     omnibus_results=<step 1>, ...) for both lm_filter="mcar" and "mnar",
     producing the final unified 9-test violin plots the appendix needs,
     WITHOUT re-running any of the 5 already-paid-for two-group methods.

Usage:
  python -m simulations.recover_factorial_omnibus_typeI \\
      --csv simulations/out/official_20260805_222825/pvalues_ppi_factorial_reps200_20260805_222914_ppi_factorial_results.csv
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulations.harness.scenarios.synthetic import (
    build_ppi_factorial_sources, PPI_FACTORIAL_NOISE_LEVELS, PPI_FACTORIAL_NOISE_LEVELS_FAST,
)
from simulations.harness.cases.pvalues import (
    run_ppi_comparison_simulation, pool_ppi_comparison_across_methods,
    save_results_artifacts_ppi_factorial, save_ppi_factorial_typeI_violin_plot,
    print_ppi_factorial_report, _parse_ppi_factorial_name,
    _COMPARISON_METHODS, _COMPARISON_METHODS_OMNIBUS, _COMPARISON_METHODS_OMNIBUS_LABEL,
)


def load_twogroup_raw(csv_path: str) -> list[SimpleNamespace]:
    """Reconstruct just the 5 attributes save_ppi_factorial_typeI_violin_plot
    reads off each row of the original run's saved raw CSV (name, method,
    n_reps, rejects_llm_only, rejects_ppi) -- everything else that CSV
    stores (et/bm/n/nlab/lm/es/bd/noise, the other rate_* columns) is
    unused by this specific plot, so it's not worth reconstructing a full
    PPIComparisonResult (which would need dataclass fields -- tag,
    effect_size, label_frac -- this CSV never stored in the first place)."""
    out = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv_mod.DictReader(handle):
            if row["method"] not in _COMPARISON_METHODS:
                skipped += 1
                continue
            n_reps = int(row["n_reps"])
            out.append(SimpleNamespace(
                name=row["name"], method=row["method"], n_reps=n_reps,
                rejects_llm_only=round(float(row["rate_llm_only"]) * n_reps) if n_reps else 0,
                rejects_ppi=round(float(row["rate_ppi"]) * n_reps) if n_reps else 0,
            ))
    if skipped:
        print(f"  (skipped {skipped} rows whose method wasn't in _COMPARISON_METHODS -- unexpected, check the CSV)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Path to the original run's *_ppi_factorial_results.csv (two-group raw data)")
    parser.add_argument("--reps", type=int, default=200, help="Must match the original run's factorial_reps (default 200, official_args_ppi's effect_reps)")
    parser.add_argument("--n-boot", type=int, default=2000, help="Must match the original run's factorial_n_boot (default 2000, official_args_ppi's ppi_n_boot)")
    parser.add_argument("--likert-max", type=int, default=5, help="Must match the original run's --factorial-likert-max (default 5)")
    parser.add_argument("--fast-noise", action="store_true", default=False, help="Pass only if the original run used --factorial-fast-noise")
    parser.add_argument("--seed", type=int, default=50, help="Default 50 = official_args_ppi's base seed (42) + 8, the offset run() applies to factorial_check specifically")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-dir", default=None, help="Default: same directory as --csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Original run_stem is the CSV filename minus "_ppi_factorial_results.csv".
    orig_stem = csv_path.name.removesuffix("_ppi_factorial_results.csv")
    omnibus_stem = f"{orig_stem}_omnibusonly"

    noise_levels = PPI_FACTORIAL_NOISE_LEVELS_FAST if args.fast_noise else PPI_FACTORIAL_NOISE_LEVELS
    print(f"Rebuilding factorial_sources (likert_max={args.likert_max}, "
          f"noise_levels={'fast' if args.fast_noise else 'standard'})...")
    factorial_sources = build_ppi_factorial_sources(likert_max=args.likert_max, noise_levels=noise_levels)
    print(f"{len(factorial_sources)} scenarios x {len(_COMPARISON_METHODS_OMNIBUS)} omnibus methods "
          f"({_COMPARISON_METHODS_OMNIBUS}), reps={args.reps}, n_boot={args.n_boot}, "
          f"seed={args.seed}, workers={args.workers}")

    t0 = time.time()
    omnibus_raw = run_ppi_comparison_simulation(
        factorial_sources, n_reps=args.reps, n_boot=args.n_boot,
        progress_mode="bar", seed=args.seed, n_workers=args.workers,
        methods=_COMPARISON_METHODS_OMNIBUS, power_tune=True,
    )
    print(f"\nOmnibus-only factorial run done in {time.time()-t0:.1f}s, {len(omnibus_raw)} cells")

    omnibus_pooled = pool_ppi_comparison_across_methods(omnibus_raw)
    omnibus_baseline = [r for r in omnibus_pooled if _parse_ppi_factorial_name(r.name)["noise"] == 0.20]

    save_results_artifacts_ppi_factorial(
        results=omnibus_raw, pooled_results=omnibus_baseline, alpha=args.alpha,
        out_dir=str(out_dir), run_stem=omnibus_stem, write_csv=True,
        label=_COMPARISON_METHODS_OMNIBUS_LABEL,
        null_results_full=omnibus_pooled, raw_results_full=omnibus_raw,
    )
    print_ppi_factorial_report(
        omnibus_baseline, alpha=args.alpha, label=_COMPARISON_METHODS_OMNIBUS_LABEL,
        null_results_full=omnibus_pooled, raw_results_full=omnibus_raw,
    )

    print(f"\nLoading original two-group raw CSV: {csv_path}")
    twogroup_raw = load_twogroup_raw(str(csv_path))
    print(f"  {len(twogroup_raw)} rows loaded, methods present: "
          f"{sorted(set(r.method for r in twogroup_raw))}")

    for lm_filter, suffix in (("mcar", "typeI_mcar"), ("mnar", "typeI_mnar")):
        out_path = plots_dir / f"{orig_stem}_{suffix}.png"
        if out_path.exists():
            backup_path = out_path.with_name(out_path.stem + "_5test_only.png")
            out_path.replace(backup_path)
            print(f"  backed up existing 5-test plot -> {backup_path}")
        saved = save_ppi_factorial_typeI_violin_plot(
            two_group_results=twogroup_raw, omnibus_results=omnibus_raw,
            alpha=args.alpha, out_path=str(out_path), lm_filter=lm_filter,
        )
        print(f"Saved unified 9-test plot: {saved}")


if __name__ == "__main__":
    main()
