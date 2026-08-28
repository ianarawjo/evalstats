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
from simulations.harness.cases.compare_e2e import _format_log_size_axis  # noqa: E402

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


def _thin_log_ticks(sizes, min_decades: float = 0.11):
    """Drop tick labels that would collide on a log axis, keeping the endpoints.

    The harness's own plot is wide enough to label every swept N; at this
    figure's width 800 and 1000 sit 0.097 decades apart and overprint. Walks
    left to right keeping a tick only once it clears `min_decades` from the
    last kept one, then forces the final value back in (and drops its
    predecessor if that reintroduced a collision) so the axis still ends at
    the largest N actually run.
    """
    xs = sorted(set(int(n) for n in sizes))
    if len(xs) < 3:
        return xs
    kept = [xs[0]]
    for x in xs[1:]:
        if np.log10(x) - np.log10(kept[-1]) >= min_decades:
            kept.append(x)
    if kept[-1] != xs[-1]:
        if np.log10(xs[-1]) - np.log10(kept[-1]) < min_decades:
            kept.pop()
        kept.append(xs[-1])
    return kept


def _ppi_cells(results, et):
    """Per-cell (N, ppi_frac) PPI-vs-labels-only gain, keyed by absolute budget.

    n_lab -- the number of human labels per arm -- is what the gain actually
    tracks. Measured on this grid it explains ~0.88 of the variance in gain,
    against ~0.30 for N and ~0.40 for the labelled FRACTION: cells with the
    same n_lab agree closely even when they come from different N and
    different fractions (binary at n_lab=80: 21.0, 20.0, 15.1 points from
    10%/800, 20%/400, 40%/200). That is the PPI story stated correctly -- what
    you spend on labels sets the gain, not how big the eval set is.
    """
    ks = _ref_ks([r for r in results if r.eval_type == et])
    # Aggregate to the (N, fraction) CONDITION before differencing. A single
    # row here is one shape x icc cell at ~100 reps, far too noisy to read a
    # difference of two rates off; the condition pools its shapes the same way
    # the summary table does.
    groups: dict[tuple[int, float], list] = {}
    for r in results:
        if r.eval_type != et or r.is_null or r.ppi_config == "none":
            continue
        if ks and r.k not in ks:
            continue
        if not r.ppi_config.startswith("frac="):
            continue
        frac = float(r.ppi_config.split("=", 1)[1])
        groups.setdefault((r.n_items, frac), []).append(r)
    out = []
    for (n_items, frac), rs in groups.items():
        den = sum(x.n_reps - x.n_errors for x in rs)
        sden = sum(x.subset_n_ok for x in rs)
        if not den or not sden:
            continue
        gain = (sum(x.extreme_reject for x in rs) / den
                - sum(x.subset_extreme_reject for x in rs) / sden) * 100.0
        out.append((int(round(frac * n_items)), gain))
    return out


def _ppi_gain_se(results, et, n_lab):
    """CONSERVATIVE Monte-Carlo SE of the gain at one budget, in points.

    Independence-assuming, and therefore too WIDE: the labels-only reference
    is computed on the same draw and the same labelled items as the PPI arm
    (see _run_cell), so the two rates are positively correlated and the paired
    SE is smaller than sqrt(se_a^2 + se_b^2). Recovering the paired interval
    needs the joint outcome counts (both reject / PPI only / reference only /
    neither), which the results CSV does not carry -- it stores each arm's
    marginal count. Treat these bars as an upper bound, not an interval.
    """
    ks = _ref_ks([r for r in results if r.eval_type == et])
    rs = [r for r in results
          if r.eval_type == et and not r.is_null and r.ppi_config.startswith("frac=")
          and (not ks or r.k in ks) and r.subset_n_ok > 0
          and int(round(float(r.ppi_config.split("=", 1)[1]) * r.n_items)) == n_lab]
    den = sum(x.n_reps - x.n_errors for x in rs)
    sden = sum(x.subset_n_ok for x in rs)
    if not den or not sden:
        return float("nan")
    a = sum(x.extreme_reject for x in rs) / den
    b = sum(x.subset_extreme_reject for x in rs) / sden
    return float(np.sqrt(a * (1 - a) / den + b * (1 - b) / sden) * 100.0)


def _panel_ppi_budget(ax, results, show_err=False):
    """Gain against the absolute labelling budget -- see _ppi_cells."""
    for et in ET_ORDER:
        cells = _ppi_cells(results, et)
        if not cells:
            continue
        xs = sorted({c[0] for c in cells})
        # light scatter of the individual (N, frac) cells behind the mean, so
        # the collapse onto one curve is visible rather than asserted
        ax.scatter([c[0] for c in cells], [c[1] for c in cells], s=7,
                   color=ET_COLOR[et], alpha=0.45, lw=0, zorder=2)
        ys = [float(np.mean([c[1] for c in cells if c[0] == x])) for x in xs]
        if show_err:
            es = [1.96 * _ppi_gain_se(results, et, x) for x in xs]
            ax.errorbar(xs, ys, yerr=es, marker=ET_MARK[et], ms=3.2, lw=1.3,
                        elinewidth=0.8, capsize=1.4, color=ET_COLOR[et],
                        label=et, zorder=3)
        else:
            ax.plot(xs, ys, marker=ET_MARK[et], ms=3.2, lw=1.3,
                    color=ET_COLOR[et], label=et, zorder=3)
    ax.axhline(0, color="black", ls="--", lw=0.9)
    ax.set_ylabel("PPI power gain\n(percentage points)")
    ax.set_title("pooled over N and budget %", fontsize=6.5, pad=2)


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


def _panel_marginal(ax, results, target):
    for et in ET_ORDER:
        rows = [r for r in results if r.eval_type == et]
        xs, ys, es = _by_n(rows, lambda rs: _rate(rs, "marginal_covered", "marginal_total"))
        ax.errorbar(xs, ys, yerr=[1.96 * e for e in es], marker=ET_MARK[et], ms=3.2,
                    lw=1.3, elinewidth=0.8, capsize=1.4, color=ET_COLOR[et], label=et)
    ax.axhline(target, color="black", ls="--", lw=0.9)
    ax.set_ylim(0.88, 1.0)
    ax.set_ylabel("per-arm CI coverage")
    ax.set_title("marginal (no multiplicity)", fontsize=6.5, pad=2)


def _panel_pairwise_k2(ax, results, target):
    for et in ET_ORDER:
        rows = [r for r in results if r.eval_type == et and r.k == 2]
        xs, ys, es = _by_n(rows, lambda rs: _rate(rs, "pairwise_covered", "pairwise_total"))
        ax.errorbar(xs, ys, yerr=[1.96 * e for e in es], marker=ET_MARK[et], ms=3.2,
                    lw=1.3, elinewidth=0.8, capsize=1.4, color=ET_COLOR[et], label=et)
    ax.axhline(target, color="black", ls="--", lw=0.9)
    ax.set_ylim(0.88, 1.0)
    ax.set_ylabel("pairwise CI coverage")
    ax.set_title("k=2, uncorrected", fontsize=6.5, pad=2)


def _panel_cov_by_ppi(ax, results, target):
    """Family-wise coverage split by whether PPI correction is in play.

    The other coverage panels pool the PPI and non-PPI cells, which asserts
    rather than shows that the FWER machinery still holds once PPI is layered
    under it -- the one combination a reader is most likely to doubt, since
    each is a correction in its own right.
    """
    for et in ET_ORDER:
        for cfg_is_ppi, ls in ((False, "-"), (True, "--")):
            rows = [r for r in results if r.eval_type == et and r.k > 2
                    and ((r.ppi_config != "none") == cfg_is_ppi)]
            xs, ys, _ = _by_n(rows, lambda rs: _rate(rs, "family_covered", "family_total"))
            ax.plot(xs, ys, marker=ET_MARK[et], ms=3.0, lw=1.2, ls=ls, color=ET_COLOR[et],
                    alpha=1.0 if not cfg_is_ppi else 0.8)
    ax.axhline(target, color="black", ls="--", lw=0.9)
    ax.set_ylim(0.88, 1.0)
    ax.set_ylabel("family-wise CI coverage")
    ax.set_title("plain FWER vs PPI+FWER", fontsize=6.5, pad=2)


def _panel_type1(ax, results, alpha, by_ppi=False):
    """Type-I error, optionally split by whether PPI is in play.

    Layering PPI under the FWER correction costs a small but real amount of
    Type-I: +0.35pp on binary (z=+3.0), +0.31 likert, +0.21 continuous. It
    stays under alpha everywhere (worst cell 0.046), so this is a cost to
    state rather than a failure to hide -- and pooling the two arms, which is
    what the compact styles do, averages it away entirely.
    """
    for et in ET_ORDER:
        base = [r for r in results if r.eval_type == et and r.k > 2 and r.is_null]
        if not by_ppi:
            xs, ys, es = _by_n(base, lambda rs: _reject_rate(rs, "any_reject"))
            ax.errorbar(xs, ys, yerr=[1.96 * e for e in es], marker=ET_MARK[et], ms=3.2,
                        lw=1.3, elinewidth=0.8, capsize=1.4, color=ET_COLOR[et], label=et)
            continue
        for cfg_is_ppi, ls in ((False, "-"), (True, "--")):
            rows = [r for r in base if (r.ppi_config != "none") == cfg_is_ppi]
            xs, ys, _ = _by_n(rows, lambda rs: _reject_rate(rs, "any_reject"))
            ax.plot(xs, ys, marker=ET_MARK[et], ms=3.0, lw=1.2, ls=ls,
                    color=ET_COLOR[et], alpha=1.0 if not cfg_is_ppi else 0.8)
    ax.axhline(alpha, color="black", ls="--", lw=0.9)
    ax.set_ylim(0, 0.10)
    ax.set_ylabel("Type-I error (FWER)")


def _panel_ppi(ax, results, gain_only, ppi_frac=None):
    """PPI vs its labels-only floor.

    `ppi_frac` pins the labelling budget. Pooling the swept fractions instead
    is NOT a like-for-like curve in N: cells below evalstats' n_lab>=15 floor
    are skipped, so which fractions are eligible changes with N (at N=60 only
    frac=0.40 survives; all three appear from N=200), and the gain depends
    strongly on the fraction -- binary +18.8 points at 0.10 against +6.6 at
    0.40. Pooling therefore mixes a shifting composition into what reads as a
    pure sample-size effect, steepening the rising limb. Holding the budget
    fixed makes the N axis mean one thing.
    """
    for et in ET_ORDER:
        et_rows = [r for r in results if r.eval_type == et]
        ks = _ref_ks(et_rows)
        rows = [r for r in et_rows if not r.is_null and (not ks or r.k in ks)
                and r.ppi_config != "none"
                and (ppi_frac is None or r.ppi_config == f"frac={ppi_frac:.2f}")]
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
    _budget = ("labels: %d%% of N" % round(ppi_frac * 100)) if ppi_frac else "pooled budgets"
    if gain_only:
        ax.axhline(0, color="black", ls="--", lw=0.9)
        ax.set_ylabel("PPI power gain\n(percentage points)")
    else:
        ax.set_ylabel("power (extreme pair)")
    ax.set_title(_budget, fontsize=6.5, pad=2)


K_COLOR = {2: "#999999", 3: "#1B9E77", 5: "#D95F02", 10: "#7570B3"}


def _power(rows):
    ok = _nok(rows)
    return (sum(r.extreme_reject for r in rows) / ok) if ok else float("nan")


def power_views(results, out, width, height):
    """Power of the auto path, three ways. Diagnostic, not a calibration claim.

    compare_e2e runs everything at correction="auto", so this is ONE p-value
    path -- Friedman omnibus, Wilcoxon pairwise, Shaffer correction -- not a
    comparison between methods (that sweep is cases/pvalues.py). What it shows
    is that the calibration reported elsewhere is not bought by a test that
    simply never rejects: coverage and Type-I hold while power still rises
    with N, which is the pairing that makes either number worth having.

    The k panel reads as the multiplicity cost, and only because
    MAX_EFFECT_SPAN_STEPS holds the arm-0..arm-(k-1) separation constant above
    k=3: same true effect, more pairs, stricter correction. k=2 is EXCLUDED
    from that reading and drawn dashed -- the cap leaves it at a single step,
    so its extreme pair is half the separation of k>=3 and its lower curve is
    a smaller effect, not a property of testing two arms.
    """
    import matplotlib.pyplot as plt
    eff = [r for r in results if not r.is_null]
    fig, ax = plt.subplots(1, 3, figsize=(width, height))

    for et in ET_ORDER:
        g = [r for r in eff if r.eval_type == et]
        ns = sorted({r.n_items for r in g})
        ys, es = [], []
        for n in ns:
            rows = [r for r in g if r.n_items == n]
            p = _power(rows); ok = _nok(rows)
            ys.append(p); es.append(1.96 * np.sqrt(max(p * (1 - p), 0) / ok) if ok else 0)
        ax[0].errorbar(ns, ys, yerr=es, marker=ET_MARK[et], ms=3.2, lw=1.3,
                       elinewidth=0.8, capsize=1.4, color=ET_COLOR[et], label=et)
    ax[0].set_ylabel("power (extreme pair)")
    ax[0].set_title("by data type", pad=2)
    ax[0].legend(frameon=False, loc="lower right", handletextpad=0.4, labelspacing=0.2)

    for k in sorted({r.k for r in eff}):
        g = [r for r in eff if r.k == k]
        ns = sorted({r.n_items for r in g})
        ys = [_power([r for r in g if r.n_items == n]) for n in ns]
        ax[1].plot(ns, ys, marker="o", ms=3.0, lw=1.3, color=K_COLOR.get(k, "#444444"),
                   ls="--" if k == 2 else "-", label=f"k={k}" + (" *" if k == 2 else ""))
    ax[1].set_title("by number of arms $k$", pad=2)
    ax[1].legend(frameon=False, loc="lower right", handletextpad=0.4, labelspacing=0.2,
                 ncol=2, columnspacing=0.8)

    mid = [r for r in eff if r.n_items in (100, 200)]
    for i, et in enumerate(ET_ORDER):
        g = [r for r in mid if r.eval_type == et]
        shapes = sorted({r.shape_label for r in g})
        vals = sorted(_power([r for r in g if r.shape_label == sh]) for sh in shapes)
        vals = [v for v in vals if v == v]
        if not vals:
            continue
        xs = np.full(len(vals), i) + np.linspace(-0.16, 0.16, len(vals))
        ax[2].scatter(xs, vals, s=9, color=ET_COLOR[et], alpha=0.75, lw=0)
        ax[2].plot([i - 0.26, i + 0.26], [np.median(vals)] * 2, color=ET_COLOR[et], lw=1.6)
    ax[2].set_xticks(range(len(ET_ORDER)))
    ax[2].set_xticklabels(ET_ORDER, rotation=12)
    # matplotlib is not LaTeX: "--" renders literally, so use a real en dash.
    ax[2].set_title("per-shape spread ($N$=100\u2013200)", pad=2)
    ax[2].set_ylabel("power (extreme pair)")

    for a in ax[:2]:
        a.set_xscale("log")
        _format_log_size_axis(a, _thin_log_ticks(sorted({r.n_items for r in eff})))
        for lbl in a.get_xticklabels():
            lbl.set_fontsize(6.0)
        a.set_xlabel("items per arm (N)", labelpad=1.0)
    for a in ax:
        a.set_ylim(0, 1.02); a.grid(alpha=.25); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out


def multi(results, out, style, width, height, alpha, ppi_frac=None, ppi_x="n", ppi_err=False):
    import matplotlib.pyplot as plt
    if style == "full":
        fig, axgrid = plt.subplots(2, 3, figsize=(width, height))
        axes = list(axgrid.ravel())
    else:
        n = 2 if style == "duo" else 3
        fig, axes = plt.subplots(1, n, figsize=(width, height))
    cov_ns = sorted({r.n_items for r in results if r.k > 2})
    ppi_ns = sorted({r.n_items for r in results
                     if not r.is_null and r.ppi_config != "none" and r.subset_n_ok > 0})
    if style == "full":
        t = 1 - alpha
        _panel_marginal(axes[0], results, t)
        _panel_pairwise_k2(axes[1], results, t)
        _panel_cov(axes[2], results, t); axes[2].set_title("k>2, FWER-corrected", fontsize=6.5, pad=2)
        _panel_cov_by_ppi(axes[3], results, t)
        _panel_type1(axes[4], results, alpha, by_ppi=True)
        axes[4].set_title("plain FWER vs PPI+FWER", fontsize=6.5, pad=2)
        _panel_ppi_budget(axes[5], results, show_err=ppi_err)
        lab_ns = sorted({c[0] for et in ET_ORDER for c in _ppi_cells(results, et)})
        panel_sizes = [cov_ns, sorted({r.n_items for r in results if r.k == 2}),
                       cov_ns, cov_ns, cov_ns, lab_ns]
        for ax, sizes in zip(axes, panel_sizes):
            ax.set_xscale("log")
            _format_log_size_axis(ax, _thin_log_ticks(sizes))
            for lbl in ax.get_xticklabels():
                lbl.set_fontsize(6.0)
            # Small labelpad: the 45-degree tick labels already occupy the
            # gap under the axis, and at the default pad the top row's xlabel
            # drifted down into the bottom row's panel titles.
            ax.set_xlabel("human labels per arm" if ax is axes[5] else "items per arm (N)",
                          labelpad=1.0)
            ax.grid(alpha=.25); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        h1 = [plt.Line2D([], [], color=ET_COLOR[e], marker=ET_MARK[e], ls="-", ms=3.2, lw=1.3)
              for e in ET_ORDER]
        h2 = [plt.Line2D([], [], color="0.35", ls="-", lw=1.2),
              plt.Line2D([], [], color="0.35", ls="--", lw=1.2)]
        axes[0].legend(h1, ET_ORDER, loc="lower right", frameon=False,
                       handletextpad=0.4, borderpad=0.2, labelspacing=0.2)
        axes[3].legend(h2, ["no PPI", "PPI"], loc="lower right", frameon=False,
                       handletextpad=0.4, borderpad=0.2, labelspacing=0.2)
        # h_pad buys the extra clearance between the top row's xlabel and the
        # bottom row's titles that labelpad alone cannot.
        fig.tight_layout(pad=0.4, h_pad=1.4)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        return out

    if ppi_frac is not None:
        ppi_ns = sorted({r.n_items for r in results
                         if not r.is_null and r.ppi_config == f"frac={ppi_frac:.2f}"
                         and r.subset_n_ok > 0})
    _panel_cov(axes[0], results, 1 - alpha)
    if style == "duo":
        _panel_ppi(axes[1], results, gain_only=False, ppi_frac=ppi_frac)
        panel_sizes = [cov_ns, ppi_ns]
    else:
        _panel_type1(axes[1], results, alpha)
        if ppi_x == "budget":
            _panel_ppi_budget(axes[2], results, show_err=ppi_err)
            lab_ns = sorted({c[0] for et in ET_ORDER for c in _ppi_cells(results, et)})
            panel_sizes = [cov_ns, cov_ns, lab_ns]
        else:
            _panel_ppi(axes[2], results, gain_only=True, ppi_frac=ppi_frac)
            panel_sizes = [cov_ns, cov_ns, ppi_ns]
    for ax, sizes in zip(axes, panel_sizes):
        ax.set_xscale("log")
        # Plain integers at exactly the N values swept -- these are sample
        # sizes read off directly, not quantities spanning many orders of
        # magnitude where 10^2 notation would be the natural choice. Shared
        # with the harness's own plot so both label the axis the same way.
        _format_log_size_axis(ax, _thin_log_ticks(sizes))
        ax.tick_params(axis="x", labelsize=6.0)
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(6.0)
        ax.set_xlabel("human labels per arm" if (ppi_x == "budget" and ax is axes[-1])
                      else "items per arm (N)")
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
    ap.add_argument("--style", choices=("scorecard", "duo", "trio", "full", "power"),
                    default="scorecard")
    ap.add_argument("-o", "--out", default="compare_e2e_paper.pdf")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--ppi-err", action="store_true",
                    help="Draw Monte-Carlo error bars on the budget panel. They are "
                         "CONSERVATIVE (independence-assuming) because the paired joint "
                         "counts are not in the CSV -- see _ppi_gain_se.")
    ap.add_argument("--ppi-x", choices=("n", "budget"), default="n",
                    help="x-axis of the PPI panel: 'n' = items per arm at a fixed budget "
                         "fraction, 'budget' = absolute human labels per arm, pooled over "
                         "N and fraction (the axis the gain actually tracks).")
    ap.add_argument("--ppi-frac", default="0.20",
                    help="Labelling budget to show in the PPI panel, as a fraction of N "
                         "(0.10/0.20/0.40), or 'all' to pool them. Pooling is not "
                         "like-for-like across N -- see _panel_ppi. Default 0.20.")
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
    if a.style == "power":
        out = power_views(results, a.out, a.width or 7.6, a.height or 2.2)
    elif a.style == "scorecard":
        out = scorecard(results, a.out, a.width or 3.3, a.height or 2.4)
    else:
        w = a.width or (3.4 if a.style == "duo" else 7.0)
        if a.style == "full":
            a.height = a.height or 3.6
        pf = None if str(a.ppi_frac).lower() == "all" else float(a.ppi_frac)
        out = multi(results, a.out, a.style, w,
                    a.height or (1.7 if a.style == "duo" else 1.9), a.alpha, pf,
                    a.ppi_x, a.ppi_err)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
