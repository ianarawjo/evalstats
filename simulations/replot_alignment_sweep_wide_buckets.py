"""One-off variant of save_ppi_alignment_sweep_plot (see
simulations/harness/cases/pvalues.py) for the paper's main-text figure
(fig:corr-likert-weighted-judge-bias), which is placed at full printed page
width via \\includegraphics[width=\\linewidth]{...} in a figure* environment.

The original plot buckets alignment into 10-point-wide (0.10) bands, which
for likert/weighted-kappa yields ~10 side-by-side panels -- fine on screen,
but each panel's title/tick/legend text becomes illegibly small once shrunk
to fit a printed page column. This script re-derives the SAME view straight
from an already-saved *_alignment_{mcar,mnar}_ppi_alignment_sweep_results.csv
(no simulation rerun) using:

  - 20-point-wide (0.20) alignment buckets instead of 10-point, so there are
    only ~5 panels instead of ~10 -- each one twice as wide.
  - Substantially larger fonts throughout (title/suptitle/tick/axis/legend),
    sized for the panel count/width this produces.

Bucket regime split (no_bias / bias_present, moderate-bias excluded) is
recomputed via the CURRENT _alignment_regime, not read from the CSV's own
"regime" column -- that column was written by an earlier version of the
sweep for some pre-2026-08-03 runs and may predate the moderate-bias-
exclusion fix (see _alignment_regime's docstring in pvalues.py).

Usage:
  .venv/bin/python -m simulations.replot_alignment_sweep_wide_buckets \\
      simulations/out/factorial_alignment_recheck_20260803_135105/pvalues_ppi_factorial_reps30_20260803_135106_alignment_mcar_ppi_alignment_sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from simulations.harness.cases.pvalues import (
    _alignment_regime,
    _kappa_band,
    _metric_pct,
    _ppi_wilson_interval,
)

BUCKET_WIDTH = 20  # 0.20 on the 0-1 kappa scale (vs. the original's 10 / 0.10)


def _bucket(pct: float, width: int = BUCKET_WIDTH) -> tuple[int, str]:
    lo = int(pct // width) * width
    lo = max(0, min(lo, 100 - width))
    return lo, f"{lo}-{lo + width}%"


def load_rows(csv_path: str, eval_type: str, metric: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for r in rows:
        if r["eval_type"] != eval_type or not r[metric]:
            continue
        regime = _alignment_regime(r["bias_label"])
        if regime == "excluded":
            continue
        out.append({
            "metric_val": float(r[metric]),
            "regime": regime,
            "n_reps": int(r["n_reps"]),
            "rejects_llm_only": round(float(r["rate_llm_only"]) * int(r["n_reps"])),
            "rejects_ppi": round(float(r["rate_ppi"]) * int(r["n_reps"])),
        })
    return out


def plot(rows: list[dict], *, eval_type: str, display_label: str, symbol: str, alpha: float, out_path: str) -> str:
    if not rows:
        raise ValueError("No rows to plot.")
    bar_width = 0.35
    group_gap = 0.5
    regimes = ("no_bias", "bias_present")
    arm_colors = {"llm_only": "#e7298a", "ppi": "#FFD400"}
    arm_edgecolors = {"llm_only": "none", "ppi": "#8a6d00"}
    arm_labels = {"llm_only": "uncorrected", "ppi": "PPI-corrected"}

    # A figure this wide (in matplotlib inches) gets shrunk hard when placed
    # at \linewidth in the paper -- what determines the PRINTED font size is
    # the fontsize/figure-width-in-inches ratio, not the raw point values
    # below. These are large enough, relative to PANEL_WIDTH_IN further down,
    # to still read at print size after that shrink (tuned against an actual
    # ACM sigconf two-column figure* render, not just the on-screen preview).
    FS_TICK = 19
    FS_TITLE = 23
    FS_SUPTITLE = 23
    FS_YLABEL = 23
    FS_LEGEND = 21
    FS_YTICKS = 19

    # Locally line-broken AND shortened (vs. _PPI_ALIGNMENT_REGIME_LABEL's
    # single-line "no judge bias (noise only)") -- at this figure's larger
    # tick font size and narrower (see PANEL_WIDTH_IN) panels, even the
    # two-line break alone is wide enough to collide with the adjacent
    # group's label; dropping "judge" keeps the longest line short enough.
    # Purely a presentation choice for THIS print-width plot, not a change
    # to the shared regime label text itself.
    regime_label_lines = {"no_bias": "no bias\n(noise only)", "bias_present": "bias present"}

    buckets = sorted({_bucket(_metric_pct(r["metric_val"])) for r in rows})

    # Narrower than the earlier 4.6in/panel: since \includegraphics[width=
    # \linewidth] normalizes the FINAL printed width regardless of this
    # figure's own inch size, a narrower source figure at the SAME font
    # points raises the printed font size (see FS_* comment above).
    PANEL_WIDTH_IN = 3.5
    fig, axes = plt.subplots(1, len(buckets), figsize=(PANEL_WIDTH_IN * len(buckets), 5.4), squeeze=False, sharey=True)
    for col, (lo, label) in enumerate(buckets):
        ax = axes[0][col]
        ax.axhline(
            alpha, color="black", ls="--", lw=1.6, alpha=0.6, zorder=1,
            label="nominal α" if col == 0 else None,
        )
        xticks, xticklabels = [], []
        for gi, regime in enumerate(regimes):  # ALWAYS both, even if empty
            cells = [r for r in rows if _bucket(_metric_pct(r["metric_val"])) == (lo, label) and r["regime"] == regime]
            n_reps_tot = sum(c["n_reps"] for c in cells)
            for ai, arm in enumerate(("llm_only", "ppi")):
                x = gi * (2 * bar_width + group_gap) + ai * bar_width
                if n_reps_tot == 0:
                    continue
                rejects_tot = sum(c[f"rejects_{arm}"] for c in cells)
                rate = rejects_tot / n_reps_tot
                lo_ci, hi_ci = _ppi_wilson_interval(rejects_tot, n_reps_tot)
                ax.bar(
                    x, rate, width=bar_width, color=arm_colors[arm], edgecolor=arm_edgecolors[arm],
                    linewidth=1.2, zorder=2,
                    label=arm_labels[arm] if (col == 0 and gi == 0) else None,
                )
                ax.errorbar(
                    x, rate, yerr=[[max(0.0, rate - lo_ci)], [max(0.0, hi_ci - rate)]],
                    fmt="none", ecolor="black", elinewidth=1.4, capsize=4, zorder=4,
                )
            mid = gi * (2 * bar_width + group_gap) + bar_width / 2
            xticks.append(mid)
            xticklabels.append(f"{regime_label_lines[regime]}\n(n={len(cells)})")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=FS_TICK)
        ax.set_xlim(-0.3, (2 * bar_width + group_gap) * len(regimes) - group_gap + 0.05)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="y", labelsize=FS_YTICKS)
        band = _kappa_band((lo + BUCKET_WIDTH / 2) / 100.0)
        ax.set_title(f"{symbol}={lo / 100:.1f}-{(lo + BUCKET_WIDTH) / 100:.1f}\n({band})", fontsize=FS_TITLE)
        if col == 0:
            ax.set_ylabel("False positive rate", fontsize=FS_YLABEL)
        ax.grid(axis="y", alpha=0.25, lw=1.0, zorder=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=FS_LEGEND, frameon=True)
    fig.suptitle(
        f"{eval_type.capitalize()}: False-Positive Rate by Judge-Human Alignment ({display_label})",
        fontsize=FS_SUPTITLE, y=0.99,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to a *_alignment_{mcar,mnar}_ppi_alignment_sweep_results.csv")
    parser.add_argument("--out", default=None, help="Output image path")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    rows = load_rows(args.csv_path, eval_type="likert", metric="weighted_kappa")
    csv_stem = Path(args.csv_path).stem.removesuffix("_ppi_alignment_sweep_results")
    out_path = args.out or str(
        Path(args.csv_path).parent / "plots" / f"{csv_stem}_likert_weighted_kappa_wide_buckets.png"
    )
    saved = plot(rows, eval_type="likert", display_label="weighted κ", symbol="κ", alpha=args.alpha, out_path=out_path)
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
