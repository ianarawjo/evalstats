r"""Replot the factorial Type-I violin+strip figure (fig:ppi-type-i-error) at
print size, from the sweep's own results CSV.

The original is 2686x734 rendered at \linewidth in a figure*, which prints
~1.9in tall with a legend that reproduces the x-axis tick labels one-for-one:
nine colours encoding exactly what the nine tick labels already say, in a
legend box eating ~15% of the width. This drops the legend, spends the width
on the data, and keeps the violin+strip language.

The whole claim is "corrected sits on nominal alpha, uncorrected does not", and
all of that lives in y < 0.15 while the axis runs to 1.0. --yscale controls how
much of the figure that bottom slice gets.

Layouts:
  vertical    tests along x, rate on y (the original's orientation)
  horizontal  tests as rows -- longest test names set horizontally, no rotation

Y scales:
  linear  0..1, faithful but spends 85% of the height on empty space
  sqrt    expands the bottom where the story is, keeps the full range visible
  split   broken axis: a tall 0..0.15 panel over a short 0.15..1.0 strip

Usage:
  python simulations/replot_typeI_violin.py <factorial_results.csv> \
      --layout vertical --yscale sqrt -o out.png
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
from matplotlib.legend_handler import HandlerTuple

# Order matches save_ppi_factorial_typeI_violin_plot: the five two-group /
# paired tests, then the four omnibus ones.
ORDER = ["ttest", "ttest_welch", "paired_t", "mwu", "wilcoxon",
         "anova_ind", "anova_rep", "friedman", "kruskal"]
SHORT = {"ttest": "$t$-test (indep)", "ttest_welch": "Welch's $t$",
         "paired_t": "paired $t$", "mwu": "Mann-Whitney",
         "wilcoxon": "Wilcoxon", "anova_ind": "ANOVA (indep)",
         "anova_rep": "RM-ANOVA", "friedman": "Friedman",
         "kruskal": "Kruskal-Wallis"}

# Per-test colours, taken from the harness's own get_method_color so this
# figure matches every other per-method plot in the paper rather than
# inventing a second palette for the same nine tests.
METHOD_COLOR = {
    "ttest": "#1f77b4", "ttest_welch": "#d62728", "paired_t": "#bd9e39",
    "mwu": "#2ca02c", "wilcoxon": "#8ca252", "anova_ind": "#e6550d",
    "anova_rep": "#fd8d3c", "friedman": "#756bb1", "kruskal": "#e377c2",
}

C_UNC = "#9AA0A6"   # uncorrected
C_COR = "#1B6CA8"   # legend stand-in only; real marks use METHOD_COLOR
C_A = "#111111"     # nominal alpha


def load(path, lm="mcar", es="null"):
    """Two schemas, auto-detected by column presence.

    factorial sweep : rate_llm_only / rate_ppi, keyed by `method`, with es and
                      lm columns to filter on.
    real-judge sweep: uncorrected_rate / corrected_rate, keyed by `test`, every
                      row already a null cell (no es column), pooled over the
                      real datasets in `tag`.

    Both are the same figure, so they share one renderer rather than drifting
    into two lookalike plots -- the real-data panel is explicitly captioned as
    the counterpart to the factorial one."""
    rows = list(csv.DictReader(open(path)))
    if rows and "corrected_rate" in rows[0]:
        return _load_real(rows)
    by = defaultdict(lambda: ([], []))
    for r in rows:
        if r.get("es") != es:
            continue
        lmv = r.get("lm", "")
        if lm == "mcar" and lmv != "mcar":
            continue
        if lm == "mnar" and not lmv.startswith("mnar"):
            continue
        try:
            u, c = float(r["rate_llm_only"]), float(r["rate_ppi"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(u) and np.isfinite(c):
            by[r["method"]][0].append(u)
            by[r["method"]][1].append(c)
    return {m: (np.array(v[0]), np.array(v[1])) for m, v in by.items()}


# CI methods that share the results file with the nine hypothesis tests; they
# are not tests and have no place on this axis.
_NOT_A_TEST = {"ppi_logit_t", "ppi_t_interval", "tango_score"}


def _load_real(rows):
    by = defaultdict(lambda: ([], []))
    for r in rows:
        m = r.get("test")
        if not m or m in _NOT_A_TEST:
            continue
        try:
            u, c = float(r["uncorrected_rate"]), float(r["corrected_rate"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(u) and np.isfinite(c):
            by[m][0].append(u)
            by[m][1].append(c)
    return {m: (np.array(v[0]), np.array(v[1])) for m, v in by.items()}


def _strip(ax, pos, vals, color, horiz, rng, width=0.30, s=1.4):
    """Violin body plus a jittered strip. Both, deliberately: the violin alone
    hides how many cells there are, and the strip alone hides where the mass
    sits once the points saturate."""
    if len(vals) < 2:
        return
    try:
        vp = ax.violinplot([vals], positions=[pos], widths=width,
                           showmedians=True, showextrema=False, vert=not horiz)
        b = vp["bodies"][0]
        b.set_facecolor(color); b.set_edgecolor(color); b.set_alpha(0.35)
        vp["cmedians"].set_color(color); vp["cmedians"].set_linewidth(1.0)
    except Exception:
        pass
    j = rng.uniform(-0.085, 0.085, size=len(vals))
    if horiz:
        ax.scatter(vals, np.full(len(vals), pos) + j, s=s, alpha=0.35,
                   color=color, edgecolors="none", zorder=3, rasterized=True)
    else:
        ax.scatter(np.full(len(vals), pos) + j, vals, s=s, alpha=0.35,
                   color=color, edgecolors="none", zorder=3, rasterized=True)


def draw(ax, data, alpha, horiz, off=0.17, band=None):
    rng = np.random.default_rng(0)
    if band is not None:
        lo, hi = band
        (ax.axhspan if not horiz else ax.axvspan)(lo, hi, color=C_A, alpha=0.07,
                                                  lw=0, zorder=0)
    meth = [m for m in ORDER if m in data]
    for i, m in enumerate(meth):
        u, c = data[m]
        _strip(ax, i - off, u, C_UNC, horiz, rng)
        _strip(ax, i + off, c, METHOD_COLOR.get(m, C_COR), horiz, rng)
    ticks, labels = range(len(meth)), [SHORT.get(m, m) for m in meth]
    if horiz:
        ax.set_yticks(list(ticks)); ax.set_yticklabels(labels)
        ax.axvline(alpha, color=C_A, ls="--", lw=0.9, zorder=4)
        ax.set_ylim(-0.6, len(meth) - 0.4)
        ax.invert_yaxis()
    else:
        ax.set_xticks(list(ticks)); ax.set_xticklabels(labels)
        ax.axhline(alpha, color=C_A, ls="--", lw=0.9, zorder=4)
        ax.set_xlim(-0.6, len(meth) - 0.4)
    return meth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="typeI_replot.png")
    ap.add_argument("--layout", choices=("vertical", "horizontal"), default="vertical")
    ap.add_argument("--yscale", choices=("linear", "sqrt", "split"), default="sqrt")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--lm", choices=("mcar", "mnar"), default="mcar")
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--band", action="store_true",
                    help="shade the binomial sampling interval a single null cell "
                         "should land in at --n-reps. Turns 'the corrected mass "
                         "looks calibrated' into something checkable: at 200 reps a "
                         "perfectly calibrated test still scatters over roughly "
                         "0.02-0.08, so mass inside the band IS the claim, and "
                         "K-W sitting low against it is a real finding rather "
                         "than an eyeballed impression.")
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--band-legend", action="store_true",
                    help="also give the band a legend entry. Off by default: the "
                         "band needs a sentence to mean anything ('the interval a "
                         "perfectly calibrated test still scatters over at this rep "
                         "count'), and a four-word legend label cannot carry that. "
                         "Define it in the caption instead.")
    ap.add_argument("--inline-legend", action="store_true",
                    help="put the legend inside the axes; saves ~0.15in of height.")
    a = ap.parse_args()

    data = load(a.csv, lm=a.lm)
    if not data:
        print(f"no null/{a.lm} rows in {a.csv}", file=sys.stderr)
        return 1
    horiz = a.layout == "horizontal"
    rate_lab = "Type-I rate (null cells)"

    rc = {"font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.5,
          "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
          "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6}

    band = None
    if a.band:
        # Normal approximation to Binomial(n_reps, alpha)/n_reps -- fine at
        # alpha=0.05, n=200 (n*alpha=10), and the point is the band's WIDTH,
        # not its exact tail behaviour.
        se = float(np.sqrt(a.alpha * (1 - a.alpha) / a.n_reps))
        band = (max(0.0, a.alpha - 1.96 * se), a.alpha + 1.96 * se)
        print(f"  cell-level 95% band at n_reps={a.n_reps}: "
              f"[{band[0]:.3f}, {band[1]:.3f}]")

    with plt.rc_context(rc):
        if a.yscale == "split" and not horiz:
            h = a.height or 2.15
            fig, (hi, lo) = plt.subplots(
                2, 1, figsize=(a.width, h), sharex=True,
                gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.08})
            for ax in (hi, lo):
                draw(ax, data, a.alpha, horiz, band=band)
            hi.set_ylim(0.15, 1.02); lo.set_ylim(0, 0.15)
            hi.spines["bottom"].set_visible(False); lo.spines["top"].set_visible(False)
            hi.tick_params(labelbottom=False, bottom=False)
            hi.set_yticks([0.5, 1.0]); lo.set_yticks([0, 0.05, 0.10, 0.15])
            # break marks
            for ax, y in ((hi, 0), (lo, 1)):
                ax.plot([0, 1], [y, y], transform=ax.transAxes, clip_on=False,
                        color="k", lw=0.7, ls=(0, (3, 3)))
            lo.set_ylabel(rate_lab); lo.yaxis.set_label_coords(-0.055, 0.75)
            axes = [hi, lo]; main_ax = lo
        else:
            h = a.height or (2.4 if horiz else 1.85)
            fig, ax = plt.subplots(figsize=(a.width, h))
            draw(ax, data, a.alpha, horiz, band=band)
            if a.yscale == "sqrt":
                (ax.set_xscale if horiz else ax.set_yscale)("function",
                    functions=(lambda x: np.sqrt(np.clip(x, 0, None)),
                               lambda x: np.power(x, 2)))
                t = [0, 0.05, 0.2, 0.5, 1.0]
                (ax.set_xticks if horiz else ax.set_yticks)(t)
            (ax.set_xlim if horiz else ax.set_ylim)(0, 1.02)
            (ax.set_xlabel if horiz else ax.set_ylabel)(rate_lab)
            axes = [ax]; main_ax = ax

        for ax in axes:
            ax.grid(axis="x" if horiz else "y", alpha=0.18, lw=0.5)
            ax.set_axisbelow(True)
        # The corrected swatch is three of the actual test colours, so the
        # legend says "colour encodes the test" without restating the nine
        # names the x axis already carries.
        swatch = tuple(Line2D([0], [0], marker="o", ls="", ms=3, color=METHOD_COLOR[k])
                       for k in ("ttest", "mwu", "friedman"))
        handles = [Line2D([0], [0], marker="o", ls="", ms=3, color=C_UNC),
                   swatch,
                   Line2D([0], [0], ls="--", lw=0.9, color=C_A)]
        labels = ["uncorrected (LLM judge only)", "PPI-corrected (colour = test)",
                  rf"nominal $\alpha={a.alpha:g}$"]
        if a.band and a.band_legend:
            handles.append(Line2D([0], [0], lw=5, color=C_A, alpha=0.18))
            labels.append(rf"95% band if exactly calibrated ($n$={a.n_reps})")
        if a.inline_legend:
            main_ax.legend(handles, labels, loc="upper left", ncol=2, frameon=False,
                           handlelength=1.6, columnspacing=1.2, handletextpad=0.5,
                           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
                           borderaxespad=0.3, fontsize=6.0)
            fig.tight_layout()
        else:
            fig.legend(handles, labels, loc="upper center",
                       ncol=4 if (a.band and a.band_legend) else 3, frameon=False,
                       handlelength=1.6, columnspacing=1.4, handletextpad=0.5,
                       handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
                       bbox_to_anchor=(0.5, 1.02))
            fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(a.out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
