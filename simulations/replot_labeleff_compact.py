r"""Compact redraws of the label-efficiency figure (fig:label_efficiency).

The shipped version wastes most of its area. Every curve stops at
``n_lab = 200``, but the x axis runs to 600--700 so that the ``y = x`` "no
benefit" reference sits at 45 degrees. Holding that 1:1 aspect while the y
range reaches ~700 forces panels three times wider than the data, and at
\linewidth that width is what sets the height (2.37in for a plot whose ink
occupies the leftmost third).

Two ways out:

  equiv       Keep the "labels a classical test would need" framing, but let
              the axes fit the data. y = x stops being a 45-degree line, which
              costs some geometric intuition and saves most of the width.
  multiplier  Plot the multiplier itself (equiv_n_lab / n_lab), which the CSV
              already carries with confidence bounds. "No benefit" becomes a
              horizontal line at 1.0, the y range collapses from ~700 to ~4,
              and all three panels can share one axis.

Usage:
  python simulations/replot_labeleff_compact.py <label_efficiency_results.csv> \
      --design multiplier -o out.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ETS = [("binary", "Binary"), ("continuous", "Continuous"), ("likert", "Likert")]
TIERS = ["0.70", "0.60", "0.50", "0.40", "0.30", "0.20"]
MARKS = {"0.70": "o", "0.60": "s", "0.50": "D", "0.40": "P", "0.30": "X", "0.20": "*"}


def load(path, family="gaussian"):
    """(eval_type, tier) -> n_lab -> dict of pooled columns."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(open(path)):
        if family and r.get("noise_family") != family:
            continue
        # Saturated cells are clamped to the power grid's edge (ppi_power hits
        # 1.0, so "how many human labels would match this power" has no finite
        # answer). Including them is not a small distortion: one saturated arm
        # at binary/rho2=0.70/n_lab=40 carries mult=37.5 against its siblings'
        # 4.3-5.6, and averaging it in triples the cell. The shipped plotter
        # skips them for the same reason -- see
        # save_ppi_label_efficiency_plots' docstring.
        if str(r.get("saturated", "")).strip().lower() == "true":
            continue
        et, tier = r.get("eval_type"), r.get("alignment_target")
        if tier not in TIERS:
            continue
        try:
            nl = int(r["n_lab"])
        except (KeyError, ValueError):
            continue
        for col in ("equiv_n_lab", "multiplier", "multiplier_lo",
                    "multiplier_hi", "predicted_mult"):
            try:
                v = float(r[col])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(v):
                acc[(et, tier)][nl][col].append(v)
    out = {}
    for k, byn in acc.items():
        out[k] = {n: {c: float(np.mean(v)) for c, v in cols.items()}
                  for n, cols in byn.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="labeleff_compact.png")
    ap.add_argument("--design", choices=("equiv", "multiplier"), default="multiplier")
    ap.add_argument("--family", default="gaussian")
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=None)
    a = ap.parse_args()

    data = load(a.csv, a.family)
    if not data:
        print("no rows", file=sys.stderr)
        return 1
    cmap = plt.get_cmap("viridis")
    colors = {t: cmap(i / (len(TIERS) - 1)) for i, t in enumerate(TIERS)}

    rc = {"font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.5,
          "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
          "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6}
    mult = a.design == "multiplier"
    h = a.height or (1.50 if mult else 1.90)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(a.width, h), sharey=mult)
        for j, (et, disp) in enumerate(ETS):
            ax = axes[j]
            for t in TIERS:
                cell = data.get((et, t))
                if not cell:
                    continue
                xs = sorted(cell)
                if mult:
                    # Derived from the pooled equiv_n_lab rather than from the
                    # pooled multiplier column, so this figure and the shipped
                    # one are the same numbers on two axes.
                    ys = [cell[x].get("equiv_n_lab", np.nan) / x for x in xs]
                else:
                    ys = [cell[x].get("equiv_n_lab", np.nan) for x in xs]
                    # The caption promises these: "Dotted lines show the
                    # predicted efficiency 1/(1 - rho^2(1 - N_lab/N))". Drop
                    # them and the caption describes something absent.
                    pred = [cell[x].get("predicted_mult", np.nan) * x for x in xs]
                    if any(np.isfinite(v) for v in pred):
                        ax.plot(xs, pred, color=colors[t], ls=":", lw=1.0,
                                alpha=0.9)
                ax.plot(xs, ys, color=colors[t], marker=MARKS[t], ms=2.6, lw=1.1,
                        markeredgewidth=0)
            if mult:
                ax.axhline(1.0, color="0.35", ls="--", lw=0.8)
                ax.set_ylim(0.8, None)
            else:
                lim = 215
                ax.plot([0, lim], [0, lim], ls="--", color="0.45", lw=0.8)
                ax.set_xlim(0, lim)
                # Every 100, not matplotlib's default. Binary's range reaches
                # ~660 and autoscaling then picks a 200 step, so the three
                # panels end up on different tick intervals and a reader
                # comparing them has to re-read the axis each time.
                ax.yaxis.set_major_locator(MultipleLocator(100))
            ax.set_title(disp, pad=3)
            ax.set_xlabel("human labels used ($N_{lab}$)")
            ax.grid(alpha=0.18, lw=0.5)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("effective multiplier" if mult
                              else "labels a classical\ntest would need")
        handles = [Line2D([0], [0], color=colors[t], marker=MARKS[t], ms=2.8,
                          lw=1.1, markeredgewidth=0) for t in TIERS]
        labels = [rf"$\rho^2\!\approx\!{t}$" for t in TIERS]
        handles.append(Line2D([0], [0], ls="--", lw=0.8, color="0.35"))
        labels.append("no benefit")
        if not mult:
            handles.append(Line2D([0], [0], ls=":", lw=1.0, color="0.35"))
            labels.append(r"predicted from $\rho^2$")
        fig.tight_layout(rect=[0, 0.11, 1, 1], w_pad=0.7)
        fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False,
                   handlelength=1.5, columnspacing=1.1, handletextpad=0.4,
                   borderaxespad=0.1, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(a.out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
