r"""Rebuild the paper's effect-size-invariance figure (fig:le-esinv) from the
RHO-DRIFT check's CSV, as a 1x4 panel row.

The figure this replaces claimed invariance outright ("the multiplier is a
property of judge quality, not of the effect being measured"). That holds for
the mean-type estimands and fails for the rank ones, so the figure now shows
one method per panel and lets the reader see which is which.

Panels 1-2 (mean-based) -- the influence function of a mean is psi(y) = y - mu,
linear in the value, so rho is a plain Pearson correlation and a location shift
cannot move it. Invariance here is exact algebra, not an empirical regularity.

Panels 3-4 (rank-based) -- their influence functions involve the CDF, whose
shape changes as the groups separate, so the realized rho^2 falls away from the
effect-invariant Spearman recipe a planner would have used. The gap between the
solid line and the dashed one IS the planning error.

Every panel plots rho2_implied -- the rho^2 the MEASURED variance multiplier
implies, i.e. the quantity the label-efficiency formula actually needs --
against that method's own named recipe (dashed black). Error bars are +/-1
Monte-Carlo sigma (RhoDriftPoint.rho2_implied_se, a paired bootstrap over
replicates); without them a reader cannot separate a real drift from draw
noise, which is exactly the confusion a 200-rep run of this check produced.

The four panels SHARE a y-axis on purpose. Left to their own autoscale, the two
mean panels would magnify a flat line's last decimal into a visible wiggle and
read as drift -- the opposite of the figure's claim.

Drawn at its final printed width (7in), matching _fwer_panels_figure, so
nothing is downscaled by \includegraphics.

Usage:
    python simulations/plot_rho_drift_esinv.py <drift_results.csv> [-o out.png]
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

# (method, display, family). ttest_welch is deliberately absent: it targets an
# IDENTICAL estimand to ttest through an identical PPI call, so its panel would
# be a pixel-for-pixel duplicate.
PANELS = [
    ("ttest",     r"Two-sample $t$", "mean"),
    ("paired_t",  r"Paired $t$",     "mean"),
    ("mwu",       "Mann-Whitney",    "rank"),
    ("wilcoxon",  "Wilcoxon",        "rank"),
]

C_MEAN = "#1B3A5C"
C_RANK = "#C1553B"
C_REC = "#111111"
C_NAIVE = "#8A8F98"


def load(path: str) -> dict[str, dict[float, dict]]:
    by: dict[str, dict[float, dict]] = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r.get("eval_type") != "continuous":
            continue
        by[r["method"]][float(r["effect_frac"])] = r
    return by


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def draw_panel(ax, row, disp, family, show_ylabel, reference="recipe"):
    col = C_MEAN if family == "mean" else C_RANK
    xs = sorted(row)
    ys = [_f(row[x], "rho2_implied") for x in xs]
    es = [_f(row[x], "rho2_implied_se") for x in xs]
    es = [0.0 if not np.isfinite(e) else e for e in es]
    ref_last = float("nan")
    if reference == "compare":
        # Three lines: what PPI actually delivered, what the library's
        # influence-function linearization predicts, and what the naive
        # named-correlation recipe predicts. The recipe-to-solid gap is the
        # problem this figure documents; evalstats hugging the solid line is
        # the resolution, and both have to be visible for either to mean
        # anything.
        ev = [_f(row[x], "rho2_evalstats") for x in xs]
        if any(np.isfinite(v) for v in ev):
            ax.plot(xs, ev, ls="--", color=C_REC, lw=1.1, zorder=2)
        rec = next((_f(row[x], "rho2_recipe") for x in xs
                    if np.isfinite(_f(row[x], "rho2_recipe"))), float("nan"))
        if np.isfinite(rec):
            ax.plot([xs[0], xs[-1]], [rec, rec], ls=":", color=C_NAIVE, lw=1.3,
                    zorder=2)
            ref_last = rec  # annotate the NAIVE recipe's error: the finding
    elif reference == "score":
        # rho2_score is measured PER EFFECT, so this reference is a curve, not
        # a flat line. It is what a mean-type method's implied rho^2 must equal
        # (exact algebra), which is the comparison the harness control makes.
        ref = [_f(row[x], "rho2_score") for x in xs]
        if any(np.isfinite(v) for v in ref):
            ax.plot(xs, ref, ls="--", color=C_REC, lw=1.0, zorder=2)
            ref_last = ref[-1]
    else:
        # rho2_recipe is effect-invariant by construction -- a flat line, and
        # the number a planner would actually have used.
        rec = next((_f(row[x], "rho2_recipe") for x in xs
                    if np.isfinite(_f(row[x], "rho2_recipe"))), float("nan"))
        if np.isfinite(rec):
            ax.plot([xs[0], xs[-1]], [rec, rec], ls="--", color=C_REC, lw=1.0,
                    zorder=2)
            ref_last = rec
    ax.errorbar(xs, ys, yerr=es, color=col, lw=1.4, marker="o", ms=2.8,
                capsize=1.6, elinewidth=0.7, zorder=3)
    ax.set_title(disp, pad=4)
    ax.set_xlabel("effect size (d)")
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_xlim(-0.15, 2.15)
    ax.set_box_aspect(1)
    ax.grid(alpha=0.22, lw=0.5)
    if show_ylabel:
        ax.set_ylabel(r"$\rho^2$ implied by multiplier")
    return xs, ys, col, ref_last


def annotate_drift(ax, xs, ys, col, ref_last):
    """Label the GAP to the dashed reference at the largest effect.

    Not the d=0 -> d=max drift of the solid line, which was the first cut and
    is the wrong quantity under either reference: a paired_t whose score-level
    rho^2 genuinely rises is SUPPOSED to rise with it, so its own drift reads
    as failure when it is tracking correctly. The gap to the reference is what
    the figure is claiming about, and it is the same rule for both variants.

    Placed in whichever right-hand corner the data vacates -- a fixed
    bottom-right collided with the falling rank lines, which end low on the
    right, exactly where the label went. Uses the shared ylim, since sharey
    gives every panel the same one."""
    if not (np.isfinite(ys[-1]) and np.isfinite(ref_last) and ref_last > 0):
        return
    lo, hi = ax.get_ylim()
    top = ys[-1] < 0.5 * (lo + hi)
    ax.annotate(f"{ys[-1] / ref_last - 1:+.0%}",
                xy=(0.95, 0.93 if top else 0.06), xycoords="axes fraction",
                ha="right", va="top" if top else "bottom",
                fontsize=7.0, color=col, fontweight="bold")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="labeleff_es_invariance_pooled.png")
    ap.add_argument("--reference", choices=("recipe", "score", "compare"),
                    default="recipe",
                    help="dashed line: the flat named-correlation recipe a planner "
                         "would use (default), or the per-effect score-level rho^2 "
                         "the estimator must equal.")
    a = ap.parse_args()

    by = load(a.csv)
    missing = [m for m, _, _ in PANELS if m not in by]
    if missing:
        print(f"missing methods in {a.csv}: {missing}", file=sys.stderr)
        return 1

    with plt.rc_context({
        "font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }):
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3), sharey=True)
        drawn = []
        for i, (m, disp, fam) in enumerate(PANELS):
            drawn.append(draw_panel(axes[i], by[m], disp, fam,
                                    show_ylabel=(i == 0),
                                    reference=a.reference))
        for ax, (xs, ys, col, ref_last) in zip(axes, drawn):
            annotate_drift(ax, xs, ys, col, ref_last)
        handles = [
            Line2D([0], [0], color=C_MEAN, lw=1.4, marker="o", ms=2.8),
            Line2D([0], [0], color=C_RANK, lw=1.4, marker="o", ms=2.8),
            Line2D([0], [0], color=C_REC, lw=1.1, ls="--"),
        ]
        labels = ["mean-based (measured)", "rank-based (measured)",
                  (r"score-level $\rho^2$ (what it must equal)"
                   if a.reference == "score"
                   else r"\textsc{evalstats} $\rho^2$ (influence function)"
                   if a.reference == "compare"
                   else r"recipe prediction (Pearson / Spearman $\rho^2$)")]
        if a.reference == "compare":
            handles.append(Line2D([0], [0], color=C_NAIVE, lw=1.3, ls=":"))
            labels.append(r"naive Pearson / Spearman $\rho^2$")
            labels[2] = r"evalstats $\rho^2$ (influence function)"
        fig.tight_layout(rect=[0, 0.17, 1, 1], w_pad=0.7)
        fig.legend(handles, labels, loc="lower center",
                   ncol=4 if a.reference == "compare" else 3, frameon=False,
                   handlelength=1.5, columnspacing=1.4, handletextpad=0.4,
                   borderaxespad=0.1, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(a.out, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
