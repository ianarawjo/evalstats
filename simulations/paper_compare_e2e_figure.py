"""Condensed compare_e2e figure for the paper -- the end-to-end integration
result, not the diagnostic grid.

save_calibration_plot draws 7 rows x 3 columns because it is a debugging
instrument: every metric, every ppi_config, every N. The paper needs the
opposite -- one small figure whose only job is "evalstats is calibrated
end-to-end, and PPI buys real power." This reads the same results CSVs and
draws that, in three alternative framings:

  --style scorecard  one panel: every calibration metric x eval type as a
                     deviation from its own target, with Monte-Carlo 95% CIs.
                     Coverage (95%) and Type-I (5%) have different targets, so
                     plotting the DEVIATION puts them on one honest axis
                     instead of a dual axis. Smallest option; reads as
                     "everything lands on the line".
  --style duo        two panels: family-wise coverage vs N, and PPI vs
                     subset-only power vs N. Keeps the N dependence, which is
                     where calibration actually gets hard.
  --style trio       three panels: coverage, Type-I, PPI power gain -- all vs N.

Usage:
    python -m simulations.paper_compare_e2e_figure A_results.csv B_results.csv \
        --style scorecard -o fig.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.replot_compare_e2e import load_results  # noqa: E402

# ColorBrewer Dark2, the order the paper's other figures already use.
ET_COLOR = {"binary": "#1B9E77", "continuous": "#D95F02", "likert": "#7570B3"}
ET_MARK = {"binary": "o", "continuous": "s", "likert": "^"}
ET_ORDER = ["binary", "continuous", "likert"]


def _rate(rows, num, den):
    n = sum(getattr(r, num) for r in rows)
    d = sum(getattr(r, den) for r in rows)
    if not d:
        return float("nan"), float("nan")
    p = n / d
    return p, float(np.sqrt(max(p * (1 - p), 0) / d))


def _nok(rows):
    return sum(r.n_reps - r.n_errors for r in rows)


def _reject_rate(rows, attr):
    d = _nok(rows)
    if not d:
        return float("nan"), float("nan")
    p = sum(getattr(r, attr) for r in rows) / d
    return p, float(np.sqrt(max(p * (1 - p), 0) / d))


def _ref_ks(rows):
    return {r.k for r in rows if r.oracle_n_ok > 0 or r.subset_n_ok > 0}


# metric label, target, selector, (numerator, denominator) or reject-attr
METRICS = [
    ("Per-arm CI", 0.95, lambda rs: rs, ("marginal_covered", "marginal_total")),
    ("Pairwise CI\n(uncorrected)", 0.95, lambda rs: [r for r in rs if r.k == 2],
     ("pairwise_covered", "pairwise_total")),
    ("Family-wise CI\n(FWER-corrected)", 0.95, lambda rs: [r for r in rs if r.k > 2],
     ("family_covered", "family_total")),
    ("Type-I error\n(FWER)", 0.05, lambda rs: [r for r in rs if r.is_null and r.k > 2], "any_reject"),
]


def scorecard(results, out, width, height):
    import matplotlib.pyplot as plt
    ys, labels, seps = [], [], []
    bands, band_safe = [], []
    pos = 0.0
    pts = []
    for mi, (mlabel, target, sel, spec) in enumerate(METRICS):
        _top = pos + 0.5
        for et in ET_ORDER:
            rows = sel([r for r in results if r.eval_type == et])
            if not rows:
                continue
            if isinstance(spec, tuple):
                p, se = _rate(rows, *spec)
            else:
                p, se = _reject_rate(rows, spec)
            pts.append((pos, (p - target) * 100, se * 100, et))
            pos -= 1.0
        labels.append((pos + len(ET_ORDER) / 2.0 - 0.5, mlabel))
        bands.append((pos + 0.5, _top))
        # +1 where over-target is the conservative direction (coverage),
        # -1 where under-target is (Type-I).
        band_safe.append(-1 if target < 0.5 else 1)
        pos -= 0.8
        if mi < len(METRICS) - 1:
            seps.append(pos + 0.4)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axvline(0, color="black", lw=1.0, zorder=2)
    # Shade the SAFE side of nominal, which flips between metrics: a coverage
    # metric is conservative ABOVE its target, a Type-I rate is conservative
    # BELOW it. Without this the Type-I row's negative deviations read as a
    # failure when they are the opposite.
    for (y0, y1), safe in zip(bands, band_safe):
        ax.axhspan(y0, y1, xmin=0.5 if safe > 0 else 0.0,
                   xmax=1.0 if safe > 0 else 0.5,
                   color="#1B9E77", alpha=0.055, lw=0, zorder=0)
    for y, dev, se, et in pts:
        ax.errorbar(dev, y, xerr=1.96 * se, fmt=ET_MARK[et], ms=3.6, lw=0,
                    elinewidth=1.0, capsize=1.6, color=ET_COLOR[et], zorder=3)
    for s in seps:
        ax.axhline(s, color="0.85", lw=0.6, zorder=1)
    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([l for _, l in labels])
    ax.set_xlabel("deviation from nominal (percentage points)")
    ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    h = [plt.Line2D([], [], color=ET_COLOR[e], marker=ET_MARK[e], ls="", ms=3.6) for e in ET_ORDER]
    # Outside the axes: at this size an inside legend lands on the Type-I row.
    ax.legend(h, ET_ORDER, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=3, frameon=False, handletextpad=0.3, borderpad=0.2,
              columnspacing=1.1)
    ax.margins(x=0.18)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out


def _by_n(rows, fn):
    ns = sorted({r.n_items for r in rows})
    xs, ys, es = [], [], []
    for n in ns:
        p, se = fn([r for r in rows if r.n_items == n])
        if p == p:
            xs.append(n); ys.append(p); es.append(se)
    return xs, ys, es


def _panel_cov(ax, results, target):
    for et in ET_ORDER:
        rows = [r for r in results if r.eval_type == et and r.k > 2]
        xs, ys, es = _by_n(rows, lambda rs: _rate(rs, "family_covered", "family_total"))
        ax.errorbar(xs, ys, yerr=[1.96 * e for e in es], marker=ET_MARK[et], ms=3.2,
                    lw=1.3, elinewidth=0.8, capsize=1.4, color=ET_COLOR[et], label=et)
    ax.axhline(target, color="black", ls="--", lw=0.9)
    ax.set_ylim(0.88, 1.0)
    ax.set_ylabel("family-wise CI coverage")


def _panel_type1(ax, results, alpha):
    for et in ET_ORDER:
        rows = [r for r in results if r.eval_type == et and r.k > 2 and r.is_null]
        xs, ys, es = _by_n(rows, lambda rs: _reject_rate(rs, "any_reject"))
        ax.errorbar(xs, ys, yerr=[1.96 * e for e in es], marker=ET_MARK[et], ms=3.2,
                    lw=1.3, elinewidth=0.8, capsize=1.4, color=ET_COLOR[et], label=et)
    ax.axhline(alpha, color="black", ls="--", lw=0.9)
    ax.set_ylim(0, 0.10)
    ax.set_ylabel("Type-I error (FWER)")


def _panel_ppi(ax, results, gain_only):
    for et in ET_ORDER:
        et_rows = [r for r in results if r.eval_type == et]
        ks = _ref_ks(et_rows)
        rows = [r for r in et_rows if not r.is_null and (not ks or r.k in ks)
                and r.ppi_config != "none"]
        ns = sorted({r.n_items for r in rows})
        xs, gp, gs = [], [], []
        for n in ns:
            sub = [r for r in rows if r.n_items == n]
            d = _nok(sub)
            sd = sum(r.subset_n_ok for r in sub)
            if not d or not sd:
                continue
            p = sum(r.extreme_reject for r in sub) / d
            s = sum(r.subset_extreme_reject for r in sub) / sd
            xs.append(n); gp.append(p); gs.append(s)
        if gain_only:
            ax.plot(xs, [(a - b) * 100 for a, b in zip(gp, gs)], marker=ET_MARK[et],
                    ms=3.2, lw=1.3, color=ET_COLOR[et], label=et)
        else:
            ax.plot(xs, gp, marker=ET_MARK[et], ms=3.2, lw=1.3, color=ET_COLOR[et], label=f"{et} (PPI)")
            ax.plot(xs, gs, marker=ET_MARK[et], ms=3.2, lw=1.1, ls="--", color=ET_COLOR[et],
                    alpha=.75, label=f"{et} (labels only)")
    if gain_only:
        ax.axhline(0, color="black", ls="--", lw=0.9)
        ax.set_ylabel("PPI power gain (pts)")
    else:
        ax.set_ylabel("power (extreme pair)")


def multi(results, out, style, width, height, alpha):
    import matplotlib.pyplot as plt
    n = 2 if style == "duo" else 3
    fig, axes = plt.subplots(1, n, figsize=(width, height))
    _panel_cov(axes[0], results, 1 - alpha)
    if style == "duo":
        _panel_ppi(axes[1], results, gain_only=False)
    else:
        _panel_type1(axes[1], results, alpha)
        _panel_ppi(axes[2], results, gain_only=True)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("items per arm (N)")
        ax.grid(alpha=.25); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    if style == "duo":
        # Two small legends instead of one 6-entry block that lands on the
        # data: colour carries eval type, linestyle carries the estimator.
        h1 = [plt.Line2D([], [], color=ET_COLOR[e], marker=ET_MARK[e], ls="-", ms=3.2, lw=1.3)
              for e in ET_ORDER]
        h2 = [plt.Line2D([], [], color="0.35", ls="-", lw=1.3),
              plt.Line2D([], [], color="0.35", ls="--", lw=1.1)]
        leg1 = axes[-1].legend(h1, ET_ORDER, loc="upper left", frameon=False,
                               handletextpad=0.4, borderpad=0.2, labelspacing=0.2)
        axes[-1].add_artist(leg1)
        axes[-1].legend(h2, ["PPI", "labels only"], loc="lower right", frameon=False,
                        handletextpad=0.4, borderpad=0.2, labelspacing=0.2)
    else:
        axes[-1].legend(frameon=False, handletextpad=0.4, borderpad=0.2, labelspacing=0.25)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--style", choices=("scorecard", "duo", "trio"), default="scorecard")
    ap.add_argument("-o", "--out", default="compare_e2e_paper.pdf")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--height", type=float, default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    results = load_results(a.csv)
    print(f"loaded {len(results)} cells; eval_types {sorted({r.eval_type for r in results})}")
    if a.style == "scorecard":
        out = scorecard(results, a.out, a.width or 3.3, a.height or 2.4)
    else:
        w = a.width or (3.4 if a.style == "duo" else 7.0)
        out = multi(results, a.out, a.style, w, a.height or (1.7 if a.style == "duo" else 1.9), a.alpha)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
