"""Regenerate the Figure 3 evidence panels (and their appendix companions).

Figure 3's decision trees state recommendations; these panels are the evidence
for them, laid out so each tree leaf has a column beneath it. Three rows, the
same grammar on both halves:

    row 1  calibration against nominal   coverage (a) / error rate (b)
    row 2  efficiency                    interval score (a) / power (b)
    row 3  the trade-off frontier        worst case vs efficiency

Blue is the recommended method, orange the strongest rival (defined by rule,
see ``_rival``), gray every other method in that leaf's appendix table row --
nothing is curated away.

Outputs, into ``simulations/out/paper_fig3_<timestamp>/``:

    fig3_evidence.pdf    the main-text panel, full textwidth
    rw_stability.pdf     appendix, ONE COLUMN: why the tree splits at n=30
    rw_grid.pdf          appendix: Romano-Wolf vs Shaffer over the n x k grid
    MANIFEST.md          source runs, key numbers, and the checks below

Usage:
    python -m simulations.paper_fig3_evidence
    python -m simulations.paper_fig3_evidence --out-dir some/where --scale 0.94

Three things this script is deliberate about, because each was got wrong once
while the figure was being built:

  * **Provenance.** The runs are pinned by path below, not globbed. The
    "Canonical Simulations to Report in Paper" folder is STALE for ci_single /
    ci_paired (predates the mean_score columns) and for the p-value suites
    (has ``mcnemar`` but not ``mcnemar_midp``). ``--check`` reproduces a few
    published appendix numbers so a swapped run fails loudly.
  * **Worst case means per scenario.** Both halves take the 10th/90th
    percentile across scenarios WITHIN each n, then the worst n. Pooling n and
    scenario into one quantile mixes two sources of spread; pooling scenarios
    away entirely hides methods that are calibrated on average and not
    everywhere.
  * **The efficiency reference must be a valid method.** "Power below best" is
    measured against the best method that stays inside the Monte-Carlo null
    band. Normalising to the best method overall lets a procedure that
    over-rejects (or, in the family panel, an FDR procedure) set the bar for
    everyone else.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import pandas as pd
from matplotlib.ticker import (FixedFormatter, FixedLocator, NullFormatter,
                               NullLocator)

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "simulations" / "out"

# --------------------------------------------------------------------------
# Pinned source runs. See the module docstring before changing any of these.
# --------------------------------------------------------------------------
RUNS = {
    "ci_single": "out/official_20260819_153506/"
                 "ci_single_synthetic_reps2000_20260819_155650_results.csv",
    "ci_paired": "out/official_20260825_020035/"
                 "ci_paired_synthetic_statmean_reps300_20260825_031413_results.csv",
    "pairwise":  "out/official_20260831_001109/"
                 "pvalues_pairwise_synthetic_reps300_20260831_012043_pairwise_results.csv",
    "multiarm":  "out/official_20260823_025955_multiarm_synth/"
                 "pvalues_multiarm_reps500_20260823_025956_multiarm_results.csv",
    "multiarm_real": "out/official_20260831_130350/"
                     "pvalues_multiarm_reps500_20260831_130354_multiarm_results.csv",
    "simult":    "out/pvalues_simultaneous_ci_reps300_20260827_012434_"
                 "simultaneous_ci_results.csv",
}

# Published appendix values these runs must reproduce, so a swapped run is loud.
CHECKS = [
    ("ci_single", "tab:ci_single:sim wilson binary coverage by n",
     lambda d: _cov_by_n(d, "binary", ["wilson"]).loc["wilson"].round(3).tolist()[:5],
     [0.947, 0.956, 0.949, 0.959, 0.947]),
    ("ci_paired", "tab:ci_paired:single:synth bonett_price binary coverage at n=15/30/50/80/100",
     lambda d: _cov_by_n(d[~d.is_null], "binary", ["bonett_price"])
               .loc["bonett_price"].reindex([15, 30, 50, 80, 100]).round(3).tolist(),
     [0.992, 0.983, 0.975, 0.970, 0.967]),
    ("pairwise", "mcnemar_midp is present and distinct from mcnemar",
     lambda d: sorted({"mcnemar", "mcnemar_midp"} & set(d.method.unique())),
     ["mcnemar", "mcnemar_midp"]),
]

REC = "#1a5fb4"      # the recommended method
RIVAL = "#a03000"    # the strongest rival
ALT = "#c4c4c4"      # every other method tested
ALTD = "#8f8f8f"
NOM = "#000000"

NICE = {
    "logit_t": "Logit-$t$", "logit_t_dither": "Logit-$t$ dith.", "wilson": "Wilson",
    "nig": "NIG", "bonett_price": "Bonett–Price", "mj_floor": "MJ floor",
    "mj_unfloored": "MJ unfl.", "wald": "Wald", "bootstrap": "boot.", "bca": "BCa",
    "t_interval": "$t$-int.", "bayes_bootstrap": "Bayes boot.",
    "smooth_bootstrap": "smooth boot.", "smooth_bootstrap_dither": "sm.boot.dith.",
    "el": "EL", "beta": "beta", "jeffreys": "Jeffreys",
    "clopper_pearson": "Clopper–P.", "bootstrap_t": "boot-$t$",
    "bayes_indep": "Bayes ind.", "bayes_indep_comp": "Bayes ind.",
    "bayes_paired_comp": "Bayes paired", "tango_exact": "Tango exact",
    "tango_scc": "Tango SCC", "newcombe_mover": "Newcombe",
    "wald_indep": "Wald ind.", "mcnemar": "McNemar exact",
    "mcnemar_midp": "McNemar mid-$p$", "wilcoxon": "Wilcoxon",
    "sign_test": "sign test", "paired_t": "paired $t$",
    "permutation": "permutation", "newcombe": "Newcombe",
    "bayes_binary": "Bayes binary", "sidak": "Šidák", "max_t": "max-$T$",
    "boot": "joint boot.", "boot_cal": "cal. boot.", "shaffer": "Shaffer",
    "romano_wolf": "Romano–Wolf", "bonferroni": "Bonferroni", "fdr_bh": "BH (FDR)",
    "westfall_young": "Westfall–Young", "holm": "Holm", "hochberg": "Hochberg",
    "friedman_nemenyi": "Nemenyi",
}

_CACHE: dict[str, pd.DataFrame] = {}


def load(key: str) -> pd.DataFrame:
    if key not in _CACHE:
        p = ROOT / "simulations" / RUNS[key]
        if not p.exists():
            raise SystemExit(
                f"missing pinned run for {key!r}:\n  {p}\n"
                "See RUNS at the top of this file. Do not substitute a run from\n"
                "'Canonical Simulations to Report in Paper' -- those are stale.")
        _CACHE[key] = pd.read_csv(p)
    return _CACHE[key]


def _cov_by_n(d, eval_type, methods):
    d = d[(d.eval_type == eval_type) & (d.method.isin(methods))]
    return (d.groupby(["method", "n"])
             .apply(lambda x: x.covered.sum() / x.n_reps.sum(), include_groups=False)
             .unstack())


def run_checks(verbose=True) -> list[str]:
    """Reproduce published appendix numbers. Returns failure messages."""
    fails = []
    for key, what, fn, expect in CHECKS:
        try:
            got = fn(load(key))
        except Exception as e:                                # pragma: no cover
            fails.append(f"{key}: {what}: raised {e!r}")
            continue
        ok = got == expect
        if verbose:
            print(f"  [{'ok ' if ok else 'FAIL'}] {key}: {what}")
        if not ok:
            fails.append(f"{key}: {what}\n      expected {expect}\n      got      {got}")
    return fails


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
_BAND: dict = {}


def null_band(nominal, reps, cells, n_grid, upper=True, B=4000, seed=0):
    """What a PERFECTLY calibrated method would show on the worst-case
    statistic used here, given this panel's reps per cell, cells per n, and
    number of sample sizes.

    Simulated rather than closed-form because the statistic is a quantile of a
    quantile (the 90th percentile across scenarios within each n, then the
    worst n). Anything at or under this is indistinguishable from nominal;
    reading a difference inside the band as real is the mistake this exists to
    prevent.
    """
    key = (round(float(nominal), 4), int(reps), int(cells), int(n_grid), upper, B)
    if key not in _BAND:
        rng = np.random.default_rng(seed)
        draws = rng.binomial(int(reps), nominal,
                             size=(B, int(n_grid), int(cells))) / reps
        per_n = np.quantile(draws, .90 if upper else .10, axis=2)
        stat = per_n.max(axis=1) if upper else per_n.min(axis=1)
        _BAND[key] = float(np.quantile(stat, .90 if upper else .10))
    return _BAND[key]


CI_LEAVES = [
    dict(short="Binary",             kind="ci_single", et="binary",     rec="wilson"),
    dict(short="Numeric",            kind="ci_single", et="continuous", rec="logit_t"),
    dict(short="Binary $\\Delta$",   kind="ci_paired", et="binary",     rec="bonett_price"),
    dict(short="Cont. $\\Delta$",    kind="ci_paired", et="continuous", rec="logit_t"),
    dict(short="Likert $\\Delta$",   kind="ci_paired", et="likert",     rec="nig"),
]


def ci_stats(leaf, nmax=100):
    d = load(leaf["kind"])
    if "is_null" in d.columns:
        d = d[~d.is_null]                     # placebo rows are not coverage cells
    d = d[(d.eval_type == leaf["et"]) & (d.n >= 15) & (d.n <= nmax)]
    per = d.groupby(["method", "n", "label"]).apply(
        lambda x: x.covered.sum() / x.n_reps.sum(), include_groups=False)
    cov = per.groupby(["method", "n"]).mean().unstack()
    q10n = per.groupby(["method", "n"]).quantile(.10).unstack()
    wid = d.pivot_table(index="method", columns="n", values="mean_width", aggfunc="mean")
    sc = d.pivot_table(index="method", columns="n", values="mean_score", aggfunc="mean")
    keep = cov.dropna().index.intersection(sc.dropna().index)
    cov, q10n, wid, sc = cov.loc[keep], q10n.loc[keep], wid.loc[keep], sc.loc[keep]
    return dict(cov=cov, q10n=q10n, wid=wid, sc=sc,
                worst=q10n.min(axis=1),
                rels=(sc / sc.min(axis=0)).mean(axis=1),
                relsn=sc / sc.min(axis=0),
                reps=float(d.n_reps.median()),
                cells=int(per.groupby(["method", "n"]).size().max()),
                n_grid=int(cov.shape[1]), rec=leaf["rec"], nominal=.95)


def _worst_per_scenario(percell, upper=True):
    """q90 (or q10) across scenarios within each n, then the worst n."""
    q = percell.groupby(level=[0, 1]).quantile(.90 if upper else .10).unstack()
    return q.max(axis=1) if upper else q.min(axis=1)


def _valid_reference(eff, worst, nominal, reps, cells, n_grid, upper=True):
    """Efficiency relative to the best method that stays inside the null band."""
    lim = null_band(nominal, reps, cells, n_grid, upper=upper)
    ok = [m for m in eff.index
          if (worst[m] <= lim if upper else worst[m] >= lim)]
    ref = eff.loc[ok].max(axis=0) if ok else eff.max(axis=0)
    return (eff / ref).clip(upper=1.0).mean(axis=1)


def pairwise_stats(et, rec):
    d = load("pairwise")
    d = d[d.eval_type == et]
    nul, alt = d[d.condition.isna()], d[d.condition == "d=0.40"]
    rate = lambda g: g.rejects.sum() / g.n_reps.sum()
    cal = nul.groupby(["method", "n"]).apply(rate, include_groups=False).unstack()
    eff = alt.groupby(["method", "n"]).apply(rate, include_groups=False).unstack()
    keep = cal.dropna().index.intersection(eff.dropna().index)
    cal, eff = cal.loc[keep], eff.loc[keep]
    percell = nul.groupby(["method", "n", "label"]).apply(rate, include_groups=False)
    worst = _worst_per_scenario(percell).reindex(keep)
    reps = float(nul.n_reps.median())
    cells = int(percell.groupby(level=[0, 1]).size().max())
    return dict(cal=cal, eff=eff, worst=worst, nominal=.05, rec=rec,
                reps=reps, cells=cells, n_grid=int(cal.shape[1]),
                releff=_valid_reference(eff, worst, .05, reps, cells, cal.shape[1]))


def simult_stats(et, rec="sidak"):
    d = load("simult")
    d = d[d.condition.isna() & (d.eval_type == et) & (d.ci_method != "none")]
    cov = d.pivot_table(index="ci_method", columns="n", values="coverage_rate",
                        aggfunc="mean")
    wid = d.pivot_table(index="ci_method", columns="n", values="avg_width",
                        aggfunc="mean")
    # k is a cell dimension: a family of 20 is a harder cell than a family of 3
    percell = d.groupby(["ci_method", "n", "label", "k"]).coverage_rate.mean()
    q = percell.groupby(level=[0, 1]).quantile(.10).unstack()
    worst_cov = q.min(axis=1)
    # error rate and 0-1 efficiency, so this panel shares the (b) block's axes
    eff = wid.min(axis=0) / wid
    reps = float(d.n_reps.median())
    cells = int(percell.groupby(level=[0, 1]).size().max())
    return dict(cal=1 - cov, eff=eff, worst=1 - worst_cov, nominal=.05, rec=rec,
                reps=reps, cells=cells, n_grid=int(cov.shape[1]),
                releff=eff.mean(axis=1))


def family_stats(et, rec="romano_wolf"):
    d = load("multiarm")
    nul = d[d.condition.isna() & (d.eval_type == et)]
    alt = d[(d.condition == "alt") & (d.eval_type == et)]
    cal = nul.pivot_table(index="correction", columns="n", values="any_reject_rate",
                          aggfunc="mean")
    eff = alt.pivot_table(index="correction", columns="n",
                          values="best_selected_rate", aggfunc="mean")
    # BH controls the FDR, not the FWER: it does not belong in a FWER panel, and
    # as the highest-power method there it would silently set the reference.
    keep = [m for m in cal.index if m not in ("none", "fdr_bh")]
    cal, eff = cal.loc[keep], eff.loc[keep]
    percell = nul.groupby(["correction", "n", "k", "label"]).any_reject_rate.mean()
    q = percell.groupby(level=[0, 1]).quantile(.90).unstack()
    worst = q.max(axis=1).reindex(keep)
    reps = float(nul.n_reps.median())
    cells = int(percell.groupby(level=[0, 1]).size().max())
    return dict(cal=cal, eff=eff, worst=worst, nominal=.05, rec=rec,
                reps=reps, cells=cells, n_grid=int(cal.shape[1]),
                releff=_valid_reference(eff, worst, .05, reps, cells, cal.shape[1]))


PV_PANELS = [
    ("1 pair, bin.",        lambda: pairwise_stats("binary", "mcnemar_midp"), [10, 30, 100]),
    ("1 pair, num.",        lambda: pairwise_stats("continuous", "wilcoxon"), [10, 30, 100]),
    ("Simult. CIs",         lambda: simult_stats("continuous"),               [15, 50, 200]),
    ("Family, $k{\\geq}3$", lambda: family_stats("continuous"),               [15, 100, 1000]),
]


def _rival(S, ci=False):
    """The one method a skeptic would name.

    CI side: best interval score among the rest. p-value side: highest power
    among methods that stay inside the null band, ties broken toward the
    better-calibrated one.
    """
    o = [m for m in S["releff" if not ci else "rels"].index if m != S["rec"]]
    if ci:
        return S["rels"][o].idxmin()
    lim = null_band(S["nominal"], S["reps"], S["cells"], S["n_grid"])
    ok = [m for m in o if S["worst"][m] <= lim] or o
    best = S["releff"][ok].max()
    near = [m for m in ok if S["releff"][m] >= best - .005]
    return min(near, key=lambda m: S["worst"][m]) if len(near) > 1 else \
        S["releff"][ok].idxmax()


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
class Type:
    """Point sizes at figsize scale; the saved figure is tight-cropped, so at
    \\textwidth these render about 1.2x larger than the numbers here."""
    def __init__(self, f=.94):
        self.title, self.tick = 6.0 * f, 5.2 * f
        self.axlab, self.inline, self.small = 5.4 * f, 5.5 * f, 4.8 * f


def _tidy(ax, t):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(.5)
    ax.spines["bottom"].set_linewidth(.5)
    ax.tick_params(labelsize=t.tick, width=.5, length=2, pad=1.4)


def _xticks(ax, vals):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(vals))
    ax.xaxis.set_major_formatter(FixedFormatter(
        [("1k" if v >= 1000 else str(v)) for v in vals]))
    ax.xaxis.set_minor_locator(NullLocator())


LEG_W, LEG_H = .50, .27
CORNERS = [((.035, .97), "left"), ((.035, .32), "left")]


def _corner_counts(ax):
    pts = []
    for ln in ax.get_lines():
        xd, yd = np.atleast_1d(ln.get_xdata()), np.atleast_1d(ln.get_ydata())
        for xx, yy in zip(xd, yd):
            try:
                pts.append(ax.transLimits.transform(ax.transScale.transform((xx, yy))))
            except Exception:
                pass
    if not pts:
        return [0, 0]
    P = np.array(pts)
    P = P[(P[:, 0] >= -.05) & (P[:, 0] <= 1.05) & (P[:, 1] >= -.05) & (P[:, 1] <= 1.05)]
    return [int(np.sum((P[:, 0] >= xr[0]) & (P[:, 0] <= xr[1]) &
                       (P[:, 1] >= yr[0]) & (P[:, 1] <= yr[1])))
            for xr, yr in [((0, LEG_W), (1 - LEG_H, 1)), ((0, LEG_W), (0, LEG_H))]]


def _corner(ax, rows, t, xy, ha):
    """Two-line mini legend on a white ground, so a label can never land on a
    curve, a rule, or a neighbouring panel."""
    fs = t.inline * .90
    step = (fs * 1.55 / 72) / ax.figure.get_figheight() / ax.get_position().height
    for i, (txt, col, bold) in enumerate(rows):
        ax.text(xy[0], xy[1] - i * step, txt, transform=ax.transAxes, fontsize=fs,
                color=col, ha=ha, va="top", zorder=9,
                fontweight="bold" if bold else "normal",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.72, pad=.6))


FLOOR = .002
FR_MAJOR, FR_MAJOR_LAB = [FLOOR, 1.0], ["best", "+100%"]
FR_MINOR = [.01, .1]


def _frontier(ax, x, y):
    pts = sorted(zip(x, y))
    fx, fy, best = [], [], -np.inf
    for a, b in pts:
        if b > best:
            fx.append(a); fy.append(b); best = b
    sx, sy = [], []
    for i in range(len(fx)):
        sx += [fx[i]]; sy += [fy[i]]
        if i + 1 < len(fx):
            sx += [fx[i + 1]]; sy += [fy[i]]
    ax.plot(sx, sy, color=ALTD, lw=.5, ls=(0, (2, 1.5)), zorder=2)


def _scatter(ax, x, y, rec, riv, ylo, yhi, nominal, t, band, upper):
    """upper=True: low y is good (error rates). upper=False: high y is good."""
    good_lo = upper
    off = 0
    for m in x.index:
        if m in (rec, riv):
            continue
        v = y[m]
        if (v > yhi) if good_lo else (v < ylo):
            off += 1
            edge = yhi - (yhi - ylo) * .012 if good_lo else ylo + (yhi - ylo) * .012
            ax.scatter([x[m]], [edge], s=7, marker="v", facecolor="white",
                       edgecolor=ALTD, linewidths=.5, zorder=3, clip_on=False)
        else:
            ax.scatter([x[m]], [v], s=6, facecolor="white", edgecolor=ALTD,
                       linewidths=.5, zorder=3)
    inside = [m for m in x.index if ylo <= y[m] <= yhi]
    _frontier(ax, x[inside].values,
              (-y[inside] if good_lo else y[inside]).values)
    ax.axhspan(*((ylo, band) if good_lo else (band, yhi)),
               color="#000000", alpha=.045, lw=0, zorder=0)
    ax.scatter([x[riv]], [np.clip(y[riv], ylo, yhi)], s=13, color=RIVAL, zorder=5)
    ax.scatter([x[rec]], [np.clip(y[rec], ylo, yhi)], s=24, color=REC, zorder=6)
    ax.axhline(nominal, color=NOM, lw=.55, ls=(0, (3, 2)), zorder=2)
    ax.set_ylim(ylo, yhi)
    ax.set_xscale("log"); ax.set_xlim(FLOOR * .70, 1.7)
    ax.xaxis.set_major_locator(FixedLocator(FR_MAJOR))
    ax.xaxis.set_major_formatter(FixedFormatter(FR_MAJOR_LAB))
    ax.xaxis.set_minor_locator(FixedLocator(FR_MINOR))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=1.1, width=.4)
    _tidy(ax, t)
    if off:
        ax.text(.965, .05, f"{off} off-scale", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=t.small, color=ALTD)


def _ci_cov(ax, S, t):
    rec, riv = S["rec"], _rival(S, ci=True)
    ns = np.array(S["cov"].columns, float)
    for m in S["cov"].index:
        if m not in (rec, riv):
            ax.plot(ns, S["cov"].loc[m].values, color=ALT, lw=.6, zorder=2)
    ax.plot(ns, S["cov"].loc[riv].values, color=RIVAL, lw=1.0, zorder=4)
    ax.fill_between(ns, S["q10n"].loc[rec].values, S["cov"].loc[rec].values,
                    color=REC, alpha=.20, lw=0, zorder=5)
    ax.plot(ns, S["cov"].loc[rec].values, color=REC, lw=1.5, zorder=6)
    ax.axhline(.95, color=NOM, lw=.55, ls=(0, (3, 2)), zorder=3)
    ax.set_ylim(.855, 1.005); ax.set_yticks([.88, .92, .95, 1.0])
    ax.set_yticklabels([".88", ".92", ".95", "1.0"])
    _xticks(ax, [15, 30, 100]); ax.set_xlim(14, 112); _tidy(ax, t)
    return [(NICE.get(rec, rec), REC, True), (NICE.get(riv, riv), RIVAL, False)]


def _ci_score(ax, S, t):
    rec, riv = S["rec"], _rival(S, ci=True)
    ns = np.array(S["relsn"].columns, float)
    for m in S["relsn"].index:
        if m not in (rec, riv):
            ax.plot(ns, S["relsn"].loc[m].values, color=ALT, lw=.6)
    ax.plot(ns, S["relsn"].loc[riv].values, color=RIVAL, lw=.9)
    ax.plot(ns, S["relsn"].loc[rec].values, color=REC, lw=1.4)
    ax.axhline(1.0, color=NOM, lw=.5, ls=(0, (3, 2)))
    ax.set_ylim(.985, 1.34); ax.set_yticks([1.0, 1.2]); ax.set_yticklabels(["1", "1.2"])
    _xticks(ax, [15, 30, 100]); ax.set_xlim(14, 112); _tidy(ax, t)


def _ci_front(ax, S, t):
    rec, riv = S["rec"], _rival(S, ci=True)
    x = (S["rels"] - 1.0).clip(lower=FLOOR)
    band = null_band(.95, S["reps"], S["cells"], S["n_grid"], upper=False)
    _scatter(ax, x, S["worst"], rec, riv, .855, .985, .95, t, band, upper=False)
    ax.set_yticks([.88, .92, .95]); ax.set_yticklabels([".88", ".92", ".95"])


def _pv_cal(ax, S, xt, t):
    rec, riv = S["rec"], _rival(S)
    ns = np.array(S["cal"].columns, float)
    for m in S["cal"].index:
        if m not in (rec, riv):
            ax.plot(ns, S["cal"].loc[m].values, color=ALT, lw=.6, zorder=2)
    ax.plot(ns, S["cal"].loc[riv].values, color=RIVAL, lw=1.0, zorder=4)
    ax.plot(ns, S["cal"].loc[rec].values, color=REC, lw=1.5, zorder=6)
    ax.axhline(.05, color=NOM, lw=.55, ls=(0, (3, 2)), zorder=3)
    ax.set_ylim(0, .115); ax.set_yticks([0, .05, .10])
    ax.set_yticklabels(["0", ".05", ".10"])
    _xticks(ax, xt); ax.set_xlim(min(xt) * .90, max(xt) * 1.10); _tidy(ax, t)
    return [(NICE.get(rec, rec), REC, True), (NICE.get(riv, riv), RIVAL, False)]


def _pv_eff(ax, S, xt, t):
    rec, riv = S["rec"], _rival(S)
    ns = np.array(S["eff"].columns, float)
    for m in S["eff"].index:
        if m not in (rec, riv):
            ax.plot(ns, S["eff"].loc[m].values, color=ALT, lw=.6)
    ax.plot(ns, S["eff"].loc[riv].values, color=RIVAL, lw=.9)
    ax.plot(ns, S["eff"].loc[rec].values, color=REC, lw=1.4)
    ax.set_ylim(-.03, 1.06); ax.set_yticks([0, .5, 1.0])
    ax.set_yticklabels(["0", ".5", "1"])
    _xticks(ax, xt); ax.set_xlim(min(xt) * .90, max(xt) * 1.10); _tidy(ax, t)


def _pv_front(ax, S, t):
    rec, riv = S["rec"], _rival(S)
    x = (1 - S["releff"]).clip(lower=FLOOR)
    band = null_band(.05, S["reps"], S["cells"], S["n_grid"])
    _scatter(ax, x, S["worst"], rec, riv, 0, .165, .05, t, band, upper=True)
    ax.set_yticks([0, .05, .10, .15]); ax.set_yticklabels(["0", ".05", ".10", ".15"])


def evidence_panel(path, scale=.94, w=7.0, h=2.30):
    """The main-text panel: (a) five CI leaves, (b) four p-value leaves."""
    t = Type(scale)
    CI = [ci_stats(l) for l in CI_LEAVES]
    PV = [(name, fn(), xt) for name, fn, xt in PV_PANELS]
    fig = plt.figure(figsize=(w, h))
    outer = fig.add_gridspec(2, 2, height_ratios=[1.78, .95],
                             width_ratios=[5, 4.35], hspace=.42, wspace=.145)
    tL = outer[0, 0].subgridspec(1, 5, wspace=.24)
    tR = outer[0, 1].subgridspec(1, 4, wspace=.26)
    bL = outer[1, 0].subgridspec(1, 5, wspace=.38)
    bR = outer[1, 1].subgridspec(1, 4, wspace=.42)
    legA, legB = [], []

    for j, (leaf, S) in enumerate(zip(CI_LEAVES, CI)):
        sub = tL[0, j].subgridspec(2, 1, height_ratios=[1.30, .48], hspace=.14)
        a1 = fig.add_subplot(sub[0]); legA.append((a1, _ci_cov(a1, S, t)))
        a1.set_title(leaf["short"], fontsize=t.title, pad=2.0)
        a1.set_xticklabels([]); a1.tick_params(axis="x", length=1.2)
        a2 = fig.add_subplot(sub[1]); _ci_score(a2, S, t)
        a3 = fig.add_subplot(bL[0, j]); _ci_front(a3, S, t)
        if j == 2:
            a2.set_xlabel("eval-set size $n$", fontsize=t.axlab, labelpad=1)
            a3.set_xlabel("interval score above the best method",
                          fontsize=t.axlab, labelpad=2.5)
        for a in (a1, a2, a3):
            if j:
                a.set_yticklabels([])
        if not j:
            a1.set_ylabel("coverage", fontsize=t.axlab, labelpad=1.5)
            a2.set_ylabel("score", fontsize=t.axlab, labelpad=1.5)
            a3.set_ylabel("worst-case cov.", fontsize=t.axlab, labelpad=1.5)

    for j, (name, S, xt) in enumerate(PV):
        sub = tR[0, j].subgridspec(2, 1, height_ratios=[1.30, .48], hspace=.14)
        a1 = fig.add_subplot(sub[0]); legB.append((a1, _pv_cal(a1, S, xt, t)))
        a1.set_title(name, fontsize=t.title, pad=2.0)
        a1.set_xticklabels([]); a1.tick_params(axis="x", length=1.2)
        a2 = fig.add_subplot(sub[1]); _pv_eff(a2, S, xt, t)
        a3 = fig.add_subplot(bR[0, j]); _pv_front(a3, S, t)
        if j == 1:
            a2.set_xlabel("eval-set size $n$", fontsize=t.axlab, labelpad=1)
            a3.set_xlabel("power below the best method", fontsize=t.axlab, labelpad=2.5)
        for a in (a1, a2, a3):
            if j:
                a.set_yticklabels([])
        if not j:
            a1.set_ylabel("error rate", fontsize=t.axlab, labelpad=1.5)
            a2.set_ylabel("power / eff.", fontsize=t.axlab, labelpad=1.5)
            a3.set_ylabel("worst-case err.", fontsize=t.axlab, labelpad=1.5)

    for block, nom in ((legA, .95), (legB, .05)):
        votes = np.sum([_corner_counts(ax) for ax, _ in block], axis=0)
        pick = int(np.argmin(votes))
        xy, ha = CORNERS[pick]
        if pick == 0:                      # hang a top legend off the rule
            lo, hi = block[0][0].get_ylim()
            xy = (xy[0], min(.97, (nom - lo) / (hi - lo) + .04 + LEG_H))
        for ax, rows in block:
            _corner(ax, rows, t, xy, ha)

    fig.savefig(path, bbox_inches="tight", pad_inches=.01)
    plt.close(fig)
    return CI, PV


def stability_column(path, scale=.94, w=3.33, h=1.55):
    """Appendix, ONE COLUMN: why the tree recommends Romano-Wolf only at n>=30.

    It is a resampling procedure, and on real benchmark data its joint null is
    too coarse to estimate below n~20: at n=10 it runs 56% over nominal and its
    spread across benchmark families is three times what Monte-Carlo error
    explains. The synthetic suite cannot show this -- its grid starts at n=15
    and its smooth scenarios make Romano-Wolf conservative there.
    """
    t = Type(scale)
    d = load("multiarm_real")
    nul = d[d.condition.isna()]
    reps = int(nul.n_reps.median())
    sd = np.sqrt(.05 * .95 / reps)
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    for ax, (corr, lab, col) in zip(axes, [("romano_wolf", "Romano–Wolf", REC),
                                           ("shaffer", "Shaffer", RIVAL)]):
        g = (nul[nul.correction == corr]
             .groupby(["label", "n"]).any_reject_rate.mean().unstack())
        ns = np.array(g.columns, float)
        ax.axhspan(.05 - 1.96 * sd, .05 + 1.96 * sd, color="#000000", alpha=.06,
                   lw=0, zorder=0)
        ax.axvspan(ns.min() * .95, 30, color=RIVAL, alpha=.05, lw=0, zorder=0)
        for _, row in g.iterrows():
            ax.plot(ns, row.values, color=ALT, lw=.55, zorder=2)
        ax.plot(ns, g.mean().values, color=col, lw=1.7, zorder=5)
        ax.axhline(.05, color=NOM, lw=.55, ls=(0, (3, 2)), zorder=3)
        ax.set_ylim(0, .145); ax.set_yticks([0, .05, .10])
        ax.set_yticklabels(["0", ".05", ".10"])
        _xticks(ax, [10, 30, 100]); ax.set_xlim(ns.min() * .95, ns.max() * 1.1)
        _tidy(ax, t)
        ax.set_title(lab, fontsize=t.title, pad=2.2)
        ax.set_xlabel("eval-set size $n$", fontsize=t.axlab, labelpad=1)
    axes[0].set_ylabel("FWER, one line per\nbenchmark family", fontsize=t.axlab,
                       labelpad=1.5)
    axes[0].text(10.3, .138, "$n{<}30$", fontsize=t.small, color=RIVAL,
                 ha="left", va="top")
    fig.subplots_adjust(wspace=.10)
    fig.savefig(path, bbox_inches="tight", pad_inches=.01)
    plt.close(fig)


def rw_grid(path, scale=.94, w=7.0, h=2.9):
    """Appendix: Romano-Wolf minus Shaffer over the whole n x k grid, on BOTH
    power metrics -- they disagree, and the disagreement is the finding."""
    t = Type(scale)
    d = load("multiarm")
    alt = d[d.condition == "alt"]
    ets = ["binary", "continuous", "likert"]
    mets = [("any_reject", "detect any true difference"),
            ("best_selected", "identify the best arm")]
    fig, axes = plt.subplots(2, 3, figsize=(w, h))
    for r, (met, mlab) in enumerate(mets):
        for c, et in enumerate(ets):
            ax = axes[r, c]
            s = alt[(alt.eval_type == et)
                    & (alt.correction.isin(["romano_wolf", "shaffer"]))]
            g = s.groupby(["correction", "n", "k"]).agg(hit=(met, "sum"),
                                                        tot=("n_reps", "sum"))
            p = (g.hit / g.tot).unstack(0); tot = g.tot.unstack(0)
            D = (p["romano_wolf"] - p["shaffer"]).unstack("k")
            SE = np.sqrt(p["romano_wolf"] * (1 - p["romano_wolf"]) / tot["romano_wolf"]
                         + p["shaffer"] * (1 - p["shaffer"]) / tot["shaffer"]).unstack("k")
            im = ax.imshow(D.values, cmap="RdBu", vmin=-.09, vmax=.09, aspect="auto")
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    v = D.values[i, j]
                    ax.text(j, i, f"{v:+.2f}".replace("+0.", "+.").replace("-0.", "-."),
                            ha="center", va="center", fontsize=t.small * .95,
                            color="black" if abs(v) < .055 else "white",
                            fontweight="bold" if abs(v) > 1.96 * SE.values[i, j] else "normal")
            ax.set_xticks(range(D.shape[1])); ax.set_xticklabels(D.columns, fontsize=t.tick)
            ax.set_yticks(range(D.shape[0]))
            ax.set_yticklabels([("1k" if v >= 1000 else v) for v in D.index],
                               fontsize=t.tick)
            ax.tick_params(length=1.5, width=.4, pad=1)
            for sp in ax.spines.values():
                sp.set_linewidth(.4)
            if r == 0:
                ax.set_title(et, fontsize=t.title, pad=3)
            if r == 1:
                ax.set_xlabel("arms $k$", fontsize=t.axlab, labelpad=1)
            if c == 0:
                ax.set_ylabel(f"{mlab}\n\neval-set size $n$", fontsize=t.axlab,
                              labelpad=1)
    cb = fig.colorbar(im, ax=axes, fraction=.016, pad=.012)
    cb.set_label("Romano–Wolf power $-$ Shaffer power", fontsize=t.axlab)
    cb.ax.tick_params(labelsize=t.tick, length=1.5, width=.4)
    fig.savefig(path, bbox_inches="tight", pad_inches=.01)
    plt.close(fig)


# --------------------------------------------------------------------------
def write_manifest(path, out_dir, CI, PV, scale, fails):
    lines = [f"# Figure 3 evidence panels", "",
             f"Generated {_dt.datetime.now():%Y-%m-%d %H:%M} by "
             f"`python -m simulations.paper_fig3_evidence` (scale {scale}).", "",
             "## Source runs (pinned; see RUNS in the script)", ""]
    for k, v in RUNS.items():
        lines.append(f"- **{k}** — `simulations/{v}`")
    lines += ["", "## Provenance checks", ""]
    lines += ["All published appendix values reproduced." if not fails else
              "**FAILED** — the pinned runs no longer reproduce the paper:"]
    lines += [f"\n```\n{f}\n```" for f in fails]

    lines += ["", "## Key numbers behind each panel", "",
              "`worst` = 10th/90th percentile across scenarios within each *n*, then",
              "the worst *n*. `band` = what a perfectly calibrated method would show",
              "on that statistic at this run's reps and cell count; a difference",
              "inside the band is not readable as real.", ""]
    for leaf, S in zip(CI_LEAVES, CI):
        band = null_band(.95, S["reps"], S["cells"], S["n_grid"], upper=False)
        riv = _rival(S, ci=True)
        lines.append(f"### (a) {leaf['short']} — band {band:.3f}, "
                     f"{len(S['cov'])} methods, {int(S['reps'])} reps/cell")
        t = pd.DataFrame({"worst-case coverage": S["worst"].round(3),
                          "score (x best)": S["rels"].round(3)}).sort_values(
                              "worst-case coverage", ascending=False)
        t["role"] = ["**recommended**" if m == S["rec"] else
                     ("*rival*" if m == riv else "") for m in t.index]
        lines += ["", t.to_markdown(), ""]
    for (name, S, _) in PV:
        band = null_band(.05, S["reps"], S["cells"], S["n_grid"])
        riv = _rival(S)
        lines.append(f"### (b) {name} — band {band:.3f}, "
                     f"{len(S['cal'])} methods, {int(S['reps'])} reps/cell")
        t = pd.DataFrame({"worst-case error": S["worst"].round(3),
                          "efficiency (x best valid)": S["releff"].round(3)}
                         ).sort_values("worst-case error")
        t["role"] = ["**recommended**" if m == S["rec"] else
                     ("*rival*" if m == riv else "") for m in t.index]
        lines += ["", t.to_markdown(), ""]

    d = load("multiarm_real")
    nul, alt = d[d.condition.isna()], d[d.condition == "alt"]
    f = nul.groupby(["correction", "n"]).any_reject_rate.mean().unstack()
    p = alt.groupby(["correction", "n"]).any_reject_rate.mean().unstack()
    g = nul[nul.correction == "romano_wolf"].groupby(["n", "label"]).any_reject_rate.mean()
    lines += ["", "## Appendix: Romano-Wolf small-n instability (real data)", "",
              "Why the tree splits at n=30. FWER, spread across benchmark families,",
              "and the apparent power gain that vanishes with it.", "",
              pd.DataFrame({
                  "RW FWER": f.loc["romano_wolf"].round(3),
                  "sd across families": g.groupby("n").std().round(3),
                  "worst family": g.groupby("n").max().round(3),
                  "Shaffer FWER": f.loc["shaffer"].round(3),
                  "apparent power gain":
                      (p.loc["romano_wolf"] - p.loc["shaffer"]).round(3),
              }).to_markdown(), ""]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None,
                    help="default: simulations/out/paper_fig3_<timestamp>/")
    ap.add_argument("--scale", type=float, default=.94,
                    help="type scale; 0.94 is what the paper uses (default)")
    ap.add_argument("--check-only", action="store_true",
                    help="verify the pinned runs reproduce the paper, draw nothing")
    A = ap.parse_args(argv)

    print("Provenance checks:")
    fails = run_checks()
    if fails:
        print("\n".join("  " + f for f in fails), file=sys.stderr)
        if A.check_only:
            return 1
        print("  (continuing; the manifest will record the failure)", file=sys.stderr)
    if A.check_only:
        return 0

    out = Path(A.out_dir) if A.out_dir else \
        OUT_ROOT / f"paper_fig3_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting to {out}")
    CI, PV = evidence_panel(out / "fig3_evidence.pdf", scale=A.scale)
    print("  fig3_evidence.pdf   main text, full textwidth")
    stability_column(out / "rw_stability.pdf", scale=A.scale)
    print("  rw_stability.pdf    appendix, one column")
    rw_grid(out / "rw_grid.pdf", scale=A.scale)
    print("  rw_grid.pdf         appendix, full textwidth")
    write_manifest(out / "MANIFEST.md", out, CI, PV, A.scale, fails)
    print("  MANIFEST.md         source runs, per-panel numbers, checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
