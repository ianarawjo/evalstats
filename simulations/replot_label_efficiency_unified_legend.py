"""One-off variant of save_ppi_label_efficiency_plot (see
simulations/harness/cases/pvalues.py) with a SINGLE shared legend to the
right of all three panels, instead of one legend per panel.

The shared legend labels each alignment tier "IRR~=<target>" (a unified,
eval-type-agnostic name for the calibrated alignment TARGET, which is
comparable across panels by construction -- each panel's judge noise is
independently calibrated via _calibrate_noise_for_alignment to hit the
SAME target) rather than repeating each panel's own metric symbol (kappa
for binary, weighted kappa for likert, Pearson r for continuous). Which
raw metric "IRR" cashes out to per eval type belongs in the figure
caption, not the legend -- see the note this script prints to stdout.

Because one shared legend needs far less horizontal room than three
per-panel legends, panels are packed much tighter (wspace) than the
original plot.

Usage:
  python -m simulations.replot_label_efficiency_unified_legend path/to/..._results.csv
  python -m simulations.replot_label_efficiency_unified_legend path/to/..._results.csv --out plot.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REQUIRED_COLUMNS = {"eval_type", "alignment_target", "n_lab", "equiv_n_lab", "saturated"}
EVAL_TYPE_ORDER = ("binary", "continuous", "likert")
EVAL_TYPE_LABEL = {"binary": "Binary", "continuous": "Continuous", "likert": "Likert"}
EVAL_TYPE_METRIC_NOTE = {
    "binary": r"Cohen's $\kappa$", "continuous": r"Pearson $r$", "likert": r"weighted $\kappa$",
}

# Kept identical to save_ppi_label_efficiency_plot's _LABEL_EFF_MARKER_SHAPES/
# _LABEL_EFF_MARKER_SIZE so a reader flipping between the two figures sees
# the same shape-to-tier mapping. "^" stays reserved for the saturated
# lower-bound overlay marker, never reused as a tier's own line marker.
MARKER_SHAPES = ("o", "s", "D", "P", "X", "*")
MARKER_SIZE = {"*": 9, "P": 6, "X": 6}


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {sorted(missing)}")
    if df["saturated"].dtype == object:
        df["saturated"] = df["saturated"].astype(str).str.strip().str.lower() == "true"
    else:
        df["saturated"] = df["saturated"].astype(bool)
    df["alignment_target"] = df["alignment_target"].astype(float).round(2)
    df["n_lab"] = df["n_lab"].astype(int)
    return df


def plot(df: pd.DataFrame, out_path: str) -> str:
    eval_types = [et for et in EVAL_TYPE_ORDER if et in set(df["eval_type"])]
    if not eval_types:
        raise ValueError("No known eval_type (binary/continuous/likert) found in CSV.")

    # Global target -> color/marker mapping, shared across ALL panels, so
    # the one legend means the same thing everywhere it's read against.
    targets = sorted(df["alignment_target"].unique(), reverse=True)
    cmap = plt.cm.viridis
    color_of = {t: cmap(0.15 + 0.7 * i / max(1, len(targets) - 1)) for i, t in enumerate(targets)}
    marker_of = {t: MARKER_SHAPES[i % len(MARKER_SHAPES)] for i, t in enumerate(targets)}
    baseline_target = min(targets, key=lambda t: abs(t - 0.7))
    any_saturated = bool(df["saturated"].any())

    fig, axes = plt.subplots(
        1, len(eval_types), figsize=(4.4 * len(eval_types), 4.2), squeeze=False,
        gridspec_kw={"wspace": 0.15},
    )
    axes = axes[0]

    for col, et in enumerate(eval_types):
        ax = axes[col]
        et_df = df[df["eval_type"] == et]

        unsaturated = et_df[~et_df["saturated"]]
        if len(unsaturated):
            max_val = max(unsaturated["n_lab"].max(), unsaturated["equiv_n_lab"].max()) * 1.15
        else:
            max_val = et_df["n_lab"].max() * 3.0

        ax.plot([0, max_val], [0, max_val], color="black", ls="--", lw=1.2, alpha=0.6, zorder=2)

        for t in targets:
            rows = et_df[et_df["alignment_target"] == t].sort_values("n_lab")
            if rows.empty:
                continue
            xs = rows["n_lab"].to_numpy()
            ys = np.where(rows["saturated"], np.minimum(rows["equiv_n_lab"], max_val * 0.97), rows["equiv_n_lab"])
            color = color_of[t]
            marker = marker_of[t]
            is_baseline = t == baseline_target
            ax.plot(
                xs, ys, color=color, marker=marker, markersize=MARKER_SIZE.get(marker, 5),
                linewidth=2.0 if is_baseline else 1.4, zorder=4,
            )
            sat = rows[rows["saturated"]]
            if len(sat):
                ax.plot(
                    sat["n_lab"], np.minimum(sat["equiv_n_lab"], max_val * 0.97),
                    color=color, marker="^", markersize=7, linestyle="none", zorder=5,
                )

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel("Num human labels used")
        ax.set_ylabel("Num human labels a classical test would need" if col == 0 else "")
        ax.set_title(EVAL_TYPE_LABEL[et])
        ax.set_aspect("equal", adjustable="box")

    # Legend built from proxy artists (not references to plotted lines) so
    # it's complete even if the first panel happens to be missing a target
    # another panel has.
    legend_handles = [Line2D([0], [0], color="black", ls="--", lw=1.2, alpha=0.6)]
    legend_labels = ["No benefit (y = x)"]
    for t in targets:
        color = color_of[t]
        marker = marker_of[t]
        legend_handles.append(Line2D(
            [0], [0], color=color, marker=marker, markersize=MARKER_SIZE.get(marker, 5),
            linewidth=2.0 if t == baseline_target else 1.4,
        ))
        legend_labels.append(f"IRR~={t:.1f}")
    if any_saturated:
        legend_handles.append(Line2D([0], [0], color="gray", marker="^", markersize=7, linestyle="none"))
        legend_labels.append("power saturated")

    # Anchored to the RIGHTMOST axes' own transAxes (not a hand-picked
    # figure-fraction number): a figure-fraction anchor has to be re-tuned
    # any time panel count/content changes tight_layout's actual axes
    # width, and over/under-shooting it either leaves a dead gap (anchor
    # too far right) or overlaps the last panel (anchor too far left, or
    # tight_layout not respecting the reserved rect exactly for equal-
    # aspect axes -- both observed while tuning this). Anchoring to the
    # last axes' own coordinate system gives a fixed, panel-relative gap
    # (5% of that axes' width) that's stable regardless of the overall
    # figure layout.
    axes[-1].legend(
        legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.05, 0.5),
        fontsize=8, frameon=True, borderaxespad=0.3,
    )

    fig.suptitle(
        "Label Efficiency: Human Labels a Classical Test Would Need to Match PPI's Power",
        fontsize=11,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to a *_ppi_label_efficiency_results.csv")
    parser.add_argument("--out", default=None, help="Output image path (default: alongside csv_path, suffixed _unified_legend.png)")
    args = parser.parse_args()

    df = load_results(args.csv_path)
    out_path = args.out or str(Path(args.csv_path).with_name(Path(args.csv_path).stem + "_unified_legend.png"))
    saved = plot(df, out_path)
    print(f"Saved: {saved}")

    eval_types = [et for et in EVAL_TYPE_ORDER if et in set(df["eval_type"])]
    metric_note = "; ".join(f"{EVAL_TYPE_LABEL[et]}={EVAL_TYPE_METRIC_NOTE[et]}" for et in eval_types)
    print(f"Caption note -- IRR is calibrated per panel via its own metric: {metric_note}.")


if __name__ == "__main__":
    main()
