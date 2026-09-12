r"""Replot the five-way PPI estimator comparison (fig:five-way-comparison-ppi-power)
at print size, from the comparison sweep's own results CSVs.

The original is 2484x1184 rendered at \linewidth in a figure*, which prints
~3.34in tall -- the largest cuttable float in the main text. Three things make
it that tall:

  1. the legend sits to the RIGHT of the panels, taking ~13% of the width and
     forcing the panels narrower (and so, at fixed aspect, taller);
  2. every one of the six panels repeats its own axis label plus a two-line
     "(label budget fixed at ...)" note, six times over for two distinct
     statements;
  3. y tick labels and "Rejection rate" are repeated on all three columns.

All three are fixed here by sharing axes, moving the legend to a bottom strip,
and demoting the fixed-parameter notes to the caption, where they are said once.

Rows are the sweep's two tags: `power` (rejection rate vs true effect, label
budget held fixed) and `compare_label_frac` (rejection rate vs N_lab, effect
held fixed). Columns are eval types.

Each point pools the sweep's methods for that cell by averaging their rejection
rates -- the same pooling the original does (its binary panel is labelled
"2 tests pooled"). Binary carries fewer applicable tests than continuous or
likert, which is a property of the sweep, not of this script.

CSVs may be passed more than once. Rows are indexed by eval_type and later
files win, so a run covering one eval type can be layered over an older, wider
one -- print provenance with --show-provenance before trusting a mixed figure,
because columns sourced from different runs were produced by different versions
of the library.

Usage:
  python simulations/replot_five_way.py A.csv [B.csv ...] --layout 2x3 -o out.png
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (column, label, colour, linestyle, marker). Order sets the legend order and
# the z-order: the two arms the figure is arguing about are drawn last.
ARMS = [
    ("rate_human_subset", "human subset only",            "#5A5A5A", ":",  "D"),
    ("rate_llm_only",     "LLM only (uncorrected)",       "#E7298A", "--", "^"),
    ("rate_llm_impute",   "LLM + label overwrite (no PPI)", "#7570B3", "--", "s"),
    ("rate_all_human",    "all human (oracle)",           "#1B9E77", "-",  "o"),
    ("rate_ppi",          "PPI-corrected",                "#D95F02", "-",  "o"),
]
ETS = [("binary", "Binary"), ("continuous", "Continuous"), ("likert", "Likert")]
ROWS = [("power", "effect_size", "true effect size"),
        ("compare_label_frac", "n_lab", r"$N_{lab}$ (labeled items)")]

# The binary sweep writes its own tag names (power_binary / complab_binary)
# rather than reusing the two-group ones. Normalise, or the binary column comes
# back silently empty -- which is exactly what it did the first time.
TAG_ALIAS = {"power": "power", "power_binary": "power",
             "compare_label_frac": "compare_label_frac",
             "complab_binary": "compare_label_frac"}


def _run_stem(fname):
    """The YYYYMMDD_HHMMSS stamp identifying the run that wrote this file."""
    m = re.search(r"(\d{8}_\d{6})", fname)
    return m.group(1) if m else None


def load(paths):
    """eval_type -> tag -> x -> arm -> [rates].  Later files win per eval_type."""
    by_et = {}
    prov = {}
    for p in paths:
        seen = set()
        rows = list(csv.DictReader(open(p)))
        for r in rows:
            seen.add(r.get("eval_type"))
        for et in seen:
            by_et[et] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            prov[et] = os.path.basename(p)
        for r in rows:
            et = r.get("eval_type")
            tag = TAG_ALIAS.get(r.get("tag", ""))
            if tag is None:
                continue
            xcol = "effect_size" if tag == "power" else "n_lab"
            try:
                x = float(r[xcol])
            except (KeyError, TypeError, ValueError):
                continue
            for col, *_ in ARMS:
                try:
                    v = float(r[col])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    by_et[et][tag][x][col].append(v)
    return by_et, prov


def panel(ax, cell, alpha, show_legend_handles):
    xs = sorted(cell)
    for col, lab, c, ls, mk in ARMS:
        ys = [float(np.mean(cell[x][col])) if cell[x].get(col) else np.nan for x in xs]
        if not any(np.isfinite(y) for y in ys):
            continue
        ax.plot(xs, ys, ls=ls, color=c, marker=mk, ms=2.0, lw=1.1,
                markeredgewidth=0, label=lab if show_legend_handles else None)
    ax.axhline(alpha, color="0.35", ls="--", lw=0.7, zorder=1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("-o", "--out", default="five_way_replot.png")
    ap.add_argument("--layout", choices=("2x3", "1x3"), default="2x3",
                    help="2x3 keeps both sweeps; 1x3 keeps only the effect-size "
                         "row and is roughly half the height.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--legend", choices=("bottom", "right"), default="bottom",
                    help="'right' puts the key in a side margin, which narrows "
                         "the panels; with a taller figure that trades wide flat "
                         "curves for squarer ones that read better.")
    ap.add_argument("--legend-frac", type=float, default=0.22,
                    help="fraction of the width reserved for a right-hand legend.")
    ap.add_argument("--preset", choices=("omnibus",), default=None,
                    help="Named layout preset. 'omnibus' reproduces "
                         "fig:five-way-comparison-omnibus exactly: legend on the "
                         "right, effect-size axis clipped at 0.8 (the curves are "
                         "flat past it), width 7.10, height 2.37 -- which lands "
                         "at the committed figure's 3.318 aspect. Explicitly "
                         "passed flags still win. NOT made the bare default: "
                         "this script also renders the main-text compact figure "
                         "(fig:five-way-comparison-ppi-power), whose x axis runs "
                         "to 1.2 and which passes no --effect-max, so a 0.8 "
                         "default would silently truncate it.")
    ap.add_argument("--effect-max", type=float, default=None, metavar="X",
                    help="Clip the effect-size row's x axis at X (e.g. 0.8). Every arm "
                         "saturates at 1.0 well before the sweep's largest effect, so the "
                         "tail is flat lines; cutting it gives the informative region more "
                         "width. Off by default -- the full sweep is still what ran.")
    ap.add_argument("--show-provenance", action="store_true")
    a = ap.parse_args()
    if a.preset == "omnibus":
        _explicit = set(sys.argv[1:])
        if not {"--legend"} & _explicit:                       a.legend = "right"
        if not {"--effect-max"} & _explicit:                   a.effect_max = 0.8
        if not {"--width"} & _explicit:                        a.width = 7.10
        if not {"--height"} & _explicit:                       a.height = 2.37

    by_et, prov = load(a.csv)
    if not by_et:
        print("no usable rows", file=sys.stderr)
        return 1
    if a.show_provenance:
        print("  provenance (eval_type -> file):")
        for et, f in sorted(prov.items()):
            print(f"    {et:12s} {f}")
        # Compare RUN STEMS, not filenames: one run legitimately writes binary
        # to its own *_binary_* file, so filename inequality is not evidence of
        # mixed provenance and warning on it cries wolf on every correct run.
        stems = {(_run_stem(v) or v) for v in prov.values()}
        if len(stems) > 1:
            print(f"  NOTE: columns come from {len(stems)} different runs "
                  f"({', '.join(sorted(stems))}) -- see module docstring.")
        else:
            print(f"  single run: {stems.pop()}")

    rows = ROWS if a.layout == "2x3" else ROWS[:1]
    # Columns are the eval types actually present, in ETS order. With all
    # three present (the main-text figure) this is identical to len(ETS);
    # with fewer -- the omnibus sweep has no binary arm, and the official
    # preset excludes "grades" -- it drops the empty column rather than
    # rendering a panel-wide gap.
    ets = [(et, disp) for et, disp in ETS if et in by_et] or ETS
    nrow, ncol = len(rows), len(ets)
    h = a.height or (2.35 if nrow == 2 else 1.45)

    rc = {"font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.5,
          "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
          "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(nrow, ncol, figsize=(a.width, h),
                                 squeeze=False, sharey=True)
        row_labelled = [False] * len(rows)
        for i, (tag, xcol, xlab) in enumerate(rows):
            for j, (et, disp) in enumerate(ets):
                ax = axes[i][j]
                cell = by_et.get(et, {}).get(tag)
                if not cell:
                    ax.set_visible(False)
                    continue
                panel(ax, cell, a.alpha, show_legend_handles=False)
                if i == 0:
                    ax.set_title(disp, pad=3)
                if not row_labelled[i]:
                    ax.set_ylabel("rejection rate")
                    ax.tick_params(labelleft=True)
                    row_labelled[i] = True
                # Once per ROW, under the middle column: all three columns
                # share an x meaning within a row, so three copies of the same
                # words is clutter. Rows differ, so it cannot be hoisted to one
                # label for the whole figure.
                if j == 1:
                    ax.set_xlabel(xlab)
                # Clip only the effect-size row: n_lab's x axis is a budget,
                # not a saturating quantity, so the same cut there would drop
                # real signal rather than a flat tail.
                if a.effect_max is not None and xcol == "effect_size":
                    ax.set_xlim(right=a.effect_max)
        # Built from ARMS rather than scraped off a panel: the first panel may
        # be hidden (an eval type absent from the CSVs), and scraping it then
        # yields a legend with only the alpha line in it.
        handles = [Line2D([0], [0], color=c, ls=ls, marker=mk, ms=2.4, lw=1.1,
                          markeredgewidth=0)
                   for _c, _l, c, ls, mk in ARMS]
        labels = [lab for _c, lab, *_ in ARMS]
        handles.append(Line2D([0], [0], ls="--", lw=0.7, color="0.35"))
        labels.append(rf"nominal $\alpha={a.alpha:g}$")
        if a.legend == "right":
            fig.tight_layout(rect=[0, 0, 1.0 - a.legend_frac, 1],
                             w_pad=0.6, h_pad=0.7)
            fig.legend(handles, labels, loc="center left", ncol=1, frameon=False,
                       handlelength=1.8, handletextpad=0.5, labelspacing=0.7,
                       borderaxespad=0.0,
                       bbox_to_anchor=(1.0 - a.legend_frac + 0.01, 0.5))
        else:
            fig.tight_layout(rect=[0, 0.10 if nrow == 2 else 0.16, 1, 1],
                             w_pad=0.6, h_pad=0.7)
            fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                       handlelength=1.8, columnspacing=1.4, handletextpad=0.5,
                       borderaxespad=0.1, bbox_to_anchor=(0.5, -0.01))
        fig.savefig(a.out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
