r"""Generate the numerical tables for the PPI validation and end-to-end checks.

Every figure in the PPI study reports its result graphically only, so a reader
who wants to check a claim has nothing to read. These are the tables that make
them checkable. Each one is built from the SAME result file its companion
figure is drawn from, so table and figure cannot drift apart:

  appendix-c   per-test Type-I + power, next to fig:typeI-factorial
               <- official_20260820_205604 factorial (synthetic, MCAR + MNAR)
               <- "ppi_real - use this for paper - official_20260819_104123"
               <- pvalues_ppi_comparison_reps200_20260825_222459 (+ binary)
               <- official_20260805_222825 omnibus comparison
  appendix-e   per-data-type coverage / Type-I / PPI power gain, next to fig:e2e
               <- the two compare_e2e CSVs named in
                  REPRODUCE_compare_e2e_figures.sh

Provenance for each of these was established by regenerating the committed
figure and diffing it against the PNG in media/simulations, not by matching
numbers alone -- see the module-level notes in each loader.

Usage:
  python -m simulations.make_ppi_tables appendix-c
  python -m simulations.make_ppi_tables appendix-e
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd

from simulations.harness.latex_tables import error_rate_cell

OUT = pathlib.Path("simulations/out")


def _strip0(text: str) -> str:
    r"""0.046 -> .046, leaving any \cellcolor prefix intact.

    The appendix prints rates without the leading zero (compress_tables.py
    does the same transform on the already-rendered tables), so a table that
    kept it would be the only one on the page that did.

    Anchored on a zero that begins the number, so it cannot eat the zero out
    of a value like 10.5 or out of a colour spec like red!20.
    """
    return re.sub(r"(?<![\d.])0\.", ".", text)


def rate_cell(value: float, alpha: float) -> str:
    """Shaded, leading-zero-stripped Type-I cell."""
    return _strip0(error_rate_cell(value, alpha))


def plain_rate(value: float) -> str:
    """Unshaded rate. Used for the uncorrected and worst-cell columns, which
    the existing p-value tables also leave unshaded: shading a column whose
    every entry is far from nominal by construction (an uncorrected rate, or
    the max over thousands of cells) colours the whole column and stops
    carrying information."""
    if value is None or not np.isfinite(value):
        return "--"
    return _strip0(f"{round(value, 3):.3f}")

# The nine tests, in the order every other per-method figure in the paper
# uses (replot_typeI_violin.ORDER), so a reader scanning between the table
# and the violins reads the same sequence.
ORDER = ["ttest", "ttest_welch", "paired_t", "mwu", "wilcoxon",
         "anova_ind", "anova_rep", "friedman", "kruskal"]
SHORT = {"ttest": r"$t$-test (indep)", "ttest_welch": r"Welch's $t$",
         "paired_t": r"paired $t$", "mwu": "Mann-Whitney $U$",
         "wilcoxon": "Wilcoxon", "anova_ind": "ANOVA (indep)",
         "anova_rep": "RM-ANOVA", "friedman": "Friedman",
         "kruskal": "Kruskal-Wallis"}

# CI constructions that share the results files with the nine hypothesis
# tests. They are not tests and have no row here.
NOT_A_TEST = {"ppi_logit_t", "ppi_t_interval", "ppi_wilson", "ppi_bonett_price",
              "tango_score", "ppi_logit_t_single", "ppi_t_interval_single"}

FACTORIAL = (OUT / "official_20260820_205604"
             / "pvalues_ppi_factorial_reps200_20260820_205621_ppi_factorial_results.csv")
REAL = (OUT / "ppi_real - use this for paper - official_20260819_104123"
        / "ppi_real_hypothesis_reps200_20260819_104131_ppi_results.csv")
FIVEWAY = OUT / "pvalues_ppi_comparison_reps200_20260825_222459_ppi_comparison_results.csv"
FIVEWAY_BIN = OUT / "pvalues_ppi_comparison_binary_reps200_20260825_222459_ppi_comparison_results.csv"
OMNIBUS = (OUT / "official_20260805_222825"
           / "pvalues_ppi_comparison_omnibus_reps200_20260805_222914_ppi_comparison_results.csv")
E2E = [OUT / "compare_e2e_reps100_20260828_001348_results.csv",
       OUT / "compare_e2e_reps100_20260828_032126_results.csv"]

# The binary comparison sweep writes its own tag names rather than reusing
# the two-group ones; normalise or the binary rows vanish silently.
TAG_ALIAS = {"power_binary": "power", "complab_binary": "compare_label_frac"}


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def factorial_typeI(path: pathlib.Path, lm: str) -> pd.DataFrame:
    """Per-test corrected/uncorrected Type-I over the factorial's null cells.

    Filters exactly as replot_typeI_violin.load() does -- es == "null", and
    lm either "mcar" or any "mnar*" -- so the table's mean sits on the same
    cells the violin plots. `keep_default_na=False` matters: the sweep writes
    the literal string "null" in the es column, which pandas would otherwise
    read as NaN and drop every row.
    """
    d = pd.read_csv(path, keep_default_na=False)
    d = d[(d.es == "null") & (~d.method.isin(NOT_A_TEST))].copy()
    d = d[d.lm == "mcar"] if lm == "mcar" else d[d.lm.str.startswith("mnar")]
    d = _num(d, ["rate_ppi", "rate_llm_only"])
    d = d[d.rate_ppi.notna() & d.rate_llm_only.notna()]
    g = d.groupby("method").agg(ppi=("rate_ppi", "mean"),
                                unc=("rate_llm_only", "mean"),
                                worst=("rate_ppi", "max"),
                                cells=("rate_ppi", "size"))
    return g


def real_typeI(path: pathlib.Path) -> pd.DataFrame:
    """Same three statistics on the real judge corpora.

    Every row in this file is already a null cell (no es column), pooled over
    dataset, label fraction, item count and judge pair.
    """
    d = pd.read_csv(path)
    d = d[~d.test.isin(NOT_A_TEST)].copy()
    d = _num(d, ["corrected_rate", "uncorrected_rate"])
    d = d[d.corrected_rate.notna()]
    g = d.groupby("test").agg(ppi=("corrected_rate", "mean"),
                              unc=("uncorrected_rate", "mean"),
                              worst=("corrected_rate", "max"),
                              cells=("corrected_rate", "size"))
    return g


def power_at(effect: float = 0.3) -> pd.DataFrame:
    """PPI-corrected power against the human-only subset at one effect size.

    Pools the comparison sweep's eval types per test, which is what the
    five-way figure's panels show side by side. The pairwise sweep carries
    five tests (two on binary); the four omnibus tests come from the omnibus
    sweep, which is a separate run at the same grid.
    """
    frames = []
    for p in (FIVEWAY, FIVEWAY_BIN, OMNIBUS):
        if not p.exists():
            sys.exit(f"missing {p}")
        f = pd.read_csv(p)
        f["tag"] = f.tag.map(lambda t: TAG_ALIAS.get(t, t))
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    # "grades" is excluded from the paper's presets and from every figure.
    d = d[(d.tag == "power") & (d.eval_type != "grades")]
    d = d[np.isclose(d.effect_size, effect)]
    d = _num(d, ["rate_ppi", "rate_human_subset"])
    g = d.groupby("method").agg(ppi=("rate_ppi", "mean"),
                                subset=("rate_human_subset", "mean"),
                                cells=("rate_ppi", "size"))
    return g


def appendix_c(alpha: float = 0.05) -> str:
    mcar = factorial_typeI(FACTORIAL, "mcar")
    mnar = factorial_typeI(FACTORIAL, "mnar")
    real = real_typeI(REAL)
    pw = power_at(0.3)

    def cell(frame, test, col):
        if test not in frame.index:
            return "--"
        return rate_cell(float(frame.loc[test, col]), alpha)

    def plain(frame, test, col):
        if test not in frame.index:
            return "--"
        return plain_rate(float(frame.loc[test, col]))

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lrrrrrrrrr!{\hspace{4pt}\vrule\hspace{4pt}}rr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Synthetic, MCAR} & \multicolumn{3}{c}{Synthetic, MNAR}"
        r" & \multicolumn{3}{c}{Real judge data} & \multicolumn{2}{c}{Power at $d{=}0.3$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}",
        r"Test & PPI & Unc. & Max & PPI & Unc. & Max & PPI & Unc. & Max & PPI & Subset \\",
        r"\midrule",
    ]
    for t in ORDER:
        row = [SHORT[t].replace("_", r"\_")]
        for frame in (mcar, mnar, real):
            row += [cell(frame, t, "ppi"), plain(frame, t, "unc"),
                    plain(frame, t, "worst")]
        row += [plain(pw, t, "ppi"), plain(pw, t, "subset")]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    fmt = lambda v: f"{v:,}".replace(",", "{,}")
    n_mcar = int(mcar.cells.max()) if len(mcar) else 0
    n_mnar = int(mnar.cells.max()) if len(mnar) else 0
    n_real = int(real.cells.max()) if len(real) else 0
    lines.append(
        r"\caption{Type I error of the nine PPI-corrected tests at nominal "
        r"$\alpha{=}0.05$, the numbers behind Figure~\ref{fig:ppi-type-i-error} "
        r"(MCAR) and Figure~\ref{fig:typeI-factorial} (MNAR). \textbf{PPI} is the "
        r"mean corrected rejection rate across null cells and \textbf{Unc.} the "
        r"mean rate of the same statistic run directly on uncorrected judge "
        r"scores. \textbf{Max} is the worst single null cell's corrected rate, "
        r"the analogue of MinCov in the CI tables: it separates tests that are "
        r"calibrated everywhere from those that are calibrated only on average. "
        f"The synthetic columns pool {fmt(n_mcar)} MCAR and {fmt(n_mnar)} MNAR null cells per "
        r"test over continuous and Likert data, the same cells the two figures "
        f"plot; the real columns pool {fmt(n_real)} cells per test over the four "
        r"multi-judge corpora of Table~\ref{tab:real-judge-data}. Power is "
        r"measured at Cohen's $d{=}0.3$ against a human-only subset of the same "
        r"labeling budget, pooled over eval types. Red marks rates above nominal, "
        r"blue below; intensity scales with distance from $\alpha$, on the same "
        r"ramp as the CI tables. Per-factor breakdowns appear in "
        r"Supplementary~S3.}")
    lines += [r"\label{tab:ppi:typeI:pertest}", r"\end{table*}"]
    return "\n".join(lines)


def _gain_by_budget(d: pd.DataFrame, et: str | None) -> pd.Series:
    """PPI-vs-labels-only gain in points, keyed by absolute label budget.

    Replicates paper_compare_e2e_figure._ppi_cells: it differences
    `extreme_reject` against `subset_extreme_reject` after pooling each
    (N, fraction) condition, restricts to the k values that actually carry a
    reference arm, and keys the result on n_lab -- the absolute number of
    human labels per arm, which is what the gain tracks. Using power_rate and
    averaging per-cell instead gives a materially different (and much flatter)
    number, because it mixes budgets where the labels-only baseline is already
    at its ceiling into the same mean.
    """
    s = d if et is None else d[d.eval_type == et]
    ks = set(s[(s.oracle_n_ok > 0) | (s.subset_n_ok > 0)].k)
    p = s[(~s.is_null) & (s.ppi_config != "none")
          & (s.ppi_config.str.startswith("frac=")) & (s.k.isin(ks))].copy()
    p["n_lab"] = (p.ppi_config.str.split("=").str[1].astype(float)
                  * p.n_items).round().astype(int)
    return p.groupby("n_lab").apply(
        lambda r: (r.extreme_reject.sum() / (r.n_reps - r.n_errors).sum()
                   - r.subset_extreme_reject.sum() / r.subset_n_ok.sum()) * 100.0,
        include_groups=False)


def appendix_e(alpha: float = 0.05) -> str:
    """Per-data-type coverage, Type-I and PPI power gain for compare().

    Aggregation matches paper_compare_e2e_figure.py exactly: coverage is
    count-weighted sum(covered)/sum(total), NOT a mean of per-cell rates,
    and each level uses its own row filter (per-arm all rows, pairwise k==2,
    family k>2). Type-I is over null cells at k>2.
    """
    d = pd.concat([pd.read_csv(p) for p in E2E], ignore_index=True)

    def rate(sub, num, den):
        return sub[num].sum() / sub[den].sum()

    def rej(sub, attr):
        return sub[attr].sum() / (sub.n_reps - sub.n_errors).sum()

    def row(label, s):
        k2, kk = s[s.k == 2], s[s.k > 2]
        nulls = kk[kk.is_null]
        g = _gain_by_budget(d, None if label.startswith(r"\textit") else s.eval_type.iloc[0])
        peak_lab = int(g.idxmax())
        return " & ".join([
            label,
            f"{rate(s, 'marginal_covered', 'marginal_total'):.3f}",
            f"{rate(k2, 'pairwise_covered', 'pairwise_total'):.3f}",
            f"{rate(kk, 'family_covered', 'family_total'):.3f}",
            error_rate_cell(rej(nulls, "any_reject"), alpha),
            f"{g.max():+.1f}", f"{peak_lab}", f"{g.loc[g.index.max()]:+.1f}",
        ]) + r" \\"

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r" & \multicolumn{3}{c}{CI coverage (target $.95$)} & "
        r"& \multicolumn{3}{c}{PPI power gain over labels-only (points)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){6-8}",
        r"Data type & Per-arm & Pairwise & Family & Type-I & "
        r"Peak & at $N_{lab}$ & at $N_{lab}{=}400$ \\",
        r"\midrule",
    ]
    for et in ("binary", "continuous", "likert"):
        lines.append(row(et.capitalize(), d[d.eval_type == et]))
    lines += [r"\midrule", row(r"\textit{All}", d)]
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{End-to-end results of \texttt{compare()} by data type, the "
        r"numbers behind Figure~\ref{fig:e2e}. Coverage is reported at each "
        r"level of multiplicity: per-arm CIs over every cell, uncorrected "
        r"pairwise CIs at $k{=}2$ (where \v{S}id'ak's $\alpha_{adj}$ reduces to "
        r"plain $\alpha$, so this is each interval's own calibration), and "
        r"FWER-corrected simultaneous CIs at $k>2$. Type-I is the family-wise "
        r"rejection rate over null cells at $k>2$; it is controlled on every "
        r"data type. The gain columns report PPI-corrected power minus the "
        r"power of analyzing the labeled subset alone on the same draws, in "
        r"percentage points, as a function of the absolute label budget "
        r"$N_{lab}$ per arm: the peak gain, the budget at which it peaks, and "
        r"the gain remaining at the largest budget we ran. The gain tracks "
        r"$N_{lab}$ rather than the eval-set size $N$ or the labeled fraction, "
        r"which is why it is reported against it. On binary and continuous data "
        r"the gain decays to essentially nothing by the largest budget, which is "
        r"the regime where the labels-only baseline has enough data to approach "
        r"the power ceiling unaided and a practitioner no longer needs PPI. On "
        r"Likert data a smaller gain persists there. "
        r"Every rate is pooled over the cells contributing to it, weighting by "
        r"the number of intervals or replicates rather than averaging per-cell "
        r"rates.}",
        r"\label{tab:e2e:bydatatype}", r"\end{table*}",
    ]
    return "\n".join(lines)


def _factorial_null(lm: str) -> pd.DataFrame:
    d = pd.read_csv(FACTORIAL, keep_default_na=False)
    d = d[(d.es == "null") & (~d.method.isin(NOT_A_TEST))].copy()
    d = d[d.lm == "mcar"] if lm == "mcar" else d[d.lm.str.startswith("mnar")]
    d = _num(d, ["rate_ppi", "rate_llm_only", "noise"])
    d = d[d.rate_ppi.notna() & d.rate_llm_only.notna()]
    # Judge noise is on the data's own scale, so its range differs by an order
    # of magnitude between continuous and Likert. Bin within eval type, or a
    # single global cut would put every Likert cell in one bucket.
    d["noise_band"] = (d.groupby("et").noise
                       .transform(lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"])))
    return d


FACTOR_LABELS = [
    ("et", "Data type", None),
    ("bm", "Judge bias magnitude", ["none", "moderate", "severe"]),
    ("noise_band", "Judge noise", ["low", "mid", "high"]),
    ("n", r"Items $N$", None),
    ("nlab", r"Labeled items $N_{lab}$", None),
]


def supp_factorial() -> str:
    """Type-I by each factor of the factorial sweep, pooled over the nine tests.

    The appendix table asks whether each TEST holds up; this asks whether each
    FACTOR of the sweep moves the answer. Both are computed from the same null
    cells, so the two tables' overall means agree by construction.
    """
    mcar, mnar = _factorial_null("mcar"), _factorial_null("mnar")
    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r" & & \multicolumn{3}{c}{MCAR} & \multicolumn{3}{c}{MNAR} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Factor & Level & PPI & Unc. & Max & PPI & Unc. & Max \\",
        r"\midrule",
    ]
    for col, label, order in FACTOR_LABELS:
        levels = order or sorted(mcar[col].dropna().unique())
        for i, lv in enumerate(levels):
            cells = [label if i == 0 else "", str(lv)]
            for frame in (mcar, mnar):
                s = frame[frame[col].astype(str) == str(lv)]
                if not len(s):
                    cells += ["--", "--", "--"]
                    continue
                cells += [rate_cell(s.rate_ppi.mean(), 0.05),
                          plain_rate(s.rate_llm_only.mean()),
                          plain_rate(s.rate_ppi.max())]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines[-1] = r"\midrule"
    tot = []
    for frame in (mcar, mnar):
        tot += [rate_cell(frame.rate_ppi.mean(), 0.05),
                plain_rate(frame.rate_llm_only.mean()),
                plain_rate(frame.rate_ppi.max())]
    lines.append(" & ".join([r"\textit{All}", ""] + tot) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{PPI-corrected Type I error by each factor of the synthetic "
        r"factorial sweep, pooled over the nine tests, at nominal "
        r"$\alpha{=}0.05$. \textbf{PPI} is the mean corrected rejection rate "
        r"over that level's null cells, \textbf{Unc.} the mean rate of the same "
        r"statistic on uncorrected judge scores, and \textbf{Max} the worst "
        r"single cell. Judge noise is binned into within-eval-type terciles "
        r"because its range differs by an order of magnitude between continuous "
        r"and Likert data. The corrected rate is flat across every factor, "
        r"including the ones that move the uncorrected rate most; under MNAR "
        r"the worst cells grow, which is the instability the paper's appendix "
        r"attributes to the rank-based tests.}",
        r"\label{supptab:ppi:factorial:byfactor}", r"\end{table}",
    ]
    return "\n".join(lines)


def supp_alignment() -> str:
    """False-positive rate by IRR metric and alignment level.

    Reads the same MCAR alignment sweeps that fig:alignment-panels is drawn
    from, and bins them the same way replot_alignment_panels.binned does:
    where a metric takes few distinct values the design's own levels are used,
    otherwise quantile bins with even per-point sample size.
    """
    dirs = [OUT / "official_20260820_205604", OUT / "LIKERT_REFIX"]
    frames = []
    for dpath in dirs:
        got = [pd.read_csv(f) for f in
               sorted(dpath.glob("*alignment_mcar_ppi_alignment_sweep_results.csv"))]
        if got:
            frames.append(pd.concat(got, ignore_index=True))
    d = frames[0]
    for nxt in frames[1:]:
        d = pd.concat([d[~d.eval_type.isin(set(nxt.eval_type))], nxt], ignore_index=True)
    panels = [("binary", "kappa", r"Cohen's $\kappa$"),
              ("continuous", "pearson_r", r"Pearson $r$"),
              ("likert", "icc_21", "ICC(2,1)"),
              ("likert", "weighted_kappa", r"Weighted $\kappa$"),
              ("likert", "percent_agreement", r"\% agreement")]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Eval type & IRR metric & Level & Unc. (bias) & Unc. (no bias) & PPI \\",
        r"\midrule",
    ]
    for et, metric, label in panels:
        sub = d[(d.eval_type == et) & (d.regime != "excluded")]
        x = pd.to_numeric(sub[metric], errors="coerce")
        levels = np.unique(np.round(x.dropna(), 3))
        if len(levels) <= 12:
            groups = [(lv, np.isclose(x, lv, atol=1e-3)) for lv in levels]
        else:
            qs = np.unique(np.quantile(x.dropna(), np.linspace(0, 1, 9)))
            idx = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
            groups = [(f"{qs[b]:.2f}--{qs[b + 1]:.2f}", idx == b) for b in range(len(qs) - 1)]
        for i, (lv, mask) in enumerate(groups):
            g = sub[mask.values if hasattr(mask, "values") else mask]
            if not len(g):
                continue
            bias, nob = g[g.regime == "bias_present"], g[g.regime == "no_bias"]
            lines.append(" & ".join([
                et.capitalize() if i == 0 else "", label if i == 0 else "",
                lv if isinstance(lv, str) else f"{lv:.2f}",
                plain_rate(bias.rate_llm_only.mean()) if len(bias) else "--",
                plain_rate(nob.rate_llm_only.mean()) if len(nob) else "--",
                rate_cell(bias.rate_ppi.mean(), 0.05) if len(bias) else "--",
            ]) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines = lines[:-1]
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{False-positive rate by judge--human alignment, the numbers "
        r"behind the judge--human alignment figure in the paper's appendix. "
        r"\textbf{Unc.\ (bias)} is the uncorrected rejection rate when judge "
        r"bias is injected, \textbf{Unc.\ (no bias)} the noise-only control at "
        r"the same alignment, and \textbf{PPI} the corrected rate on the "
        r"bias-present cells. The two uncorrected columns together are what "
        r"attributes an elevated rate to bias rather than to noise. The "
        r"corrected column stays at nominal across every metric and level, "
        r"including the alignment range where the uncorrected rate peaks. "
        r"Levels are the design's own where a metric takes few distinct values, "
        r"and quantile bins otherwise, matching the figure. A dash marks a "
        r"bucket the corresponding regime never reached: the bias-present and "
        r"noise-only arms span different alignment ranges by construction, "
        r"since injected bias itself caps the attainable agreement.}",
        r"\label{supptab:ppi:alignment}", r"\end{table}",
    ]
    return "\n".join(lines)


def supp_power() -> str:
    """Five-way power comparison per test, eval type and label budget."""
    frames = []
    for p in (FIVEWAY, FIVEWAY_BIN, OMNIBUS):
        f = pd.read_csv(p)
        f["tag"] = f.tag.map(lambda t: TAG_ALIAS.get(t, t))
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.eval_type != "grades"]
    arms = [("rate_all_human", "Oracle"), ("rate_ppi", "PPI"),
            ("rate_llm_impute", "Overwrite"), ("rate_llm_only", "LLM only"),
            ("rate_human_subset", "Subset")]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r" & & \multicolumn{5}{c}{Rejection rate} \\", r"\cmidrule(lr){3-7}",
        "Eval type & Test & " + " & ".join(l for _, l in arms) + r" \\",
        r"\midrule",
    ]
    pw = d[(d.tag == "power") & np.isclose(d.effect_size, 0.3)]
    for et in ("binary", "continuous", "likert"):
        s = pw[pw.eval_type == et]
        tests = [t for t in ORDER if t in set(s.method)]
        for i, t in enumerate(tests):
            g = s[s.method == t]
            lines.append(" & ".join(
                [et.capitalize() if i == 0 else "", SHORT[t]]
                + [plain_rate(g[c].mean()) for c, _ in arms]) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines = lines[:-1]
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Power of the five estimators at Cohen's $d{=}0.3$ and a 20\% "
        r"label budget, the numbers behind the effect-size row of "
        r"the five-way pairwise and omnibus power figures in the paper's "
        r"appendix. "
        r"\textbf{Oracle} analyzes every item's human label, \textbf{Subset} "
        r"only the labeled items, and \textbf{PPI} the corrected combination of "
        r"labels and judge scores. \textbf{Overwrite} replaces the judge score "
        r"with the human label where one exists and applies no correction, and "
        r"\textbf{LLM only} runs the statistic on raw judge scores. PPI sits "
        r"between the oracle and the labeled subset for every test, which is "
        r"the ordering the figures show. The uncorrected arms are not "
        r"comparable on power, since their Type I error is not controlled "
        r"(see the per-test Type I table in the paper's appendix).}",
        r"\label{supptab:ppi:power}", r"\end{table}",
    ]
    return "\n".join(lines)


def supp_ci_methods() -> str:
    """PPI-corrected CI constructions: bias, coverage and width."""
    # The reps2000 sweep is the one behind the figure: it is the only
    # synthetic effect run carrying ppi_bonett_price, and its mean coverages
    # (.947 for bonett_price, .983 for Wilson) are the 94.7 and 98.3 the
    # figure's caption quotes. Its real-data counterpart is the 162228 run,
    # whose sibling 161333 has no bonett_price rows at all.
    synth = [OUT / "pvalues_ppi_effect_reps2000_20260826_152110_ppi_effect_results.csv"]
    real = OUT / "ppi_real_effect_reps200_20260826_162228_ppi_effect_results.csv"
    keep = ["ppi_bonett_price", "ppi_wilson", "ppi_logit_t", "ppi_t_interval"]
    disp = {"ppi_bonett_price": r"\texttt{bonett\_price}", "ppi_wilson": "Wilson",
            "ppi_logit_t": "logit-$t$", "ppi_t_interval": "$t$-interval"}
    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r" & \multicolumn{3}{c}{Synthetic} & \multicolumn{3}{c}{Real judge data} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"CI method & Bias & Cov. & Width & Bias & Cov. & Width \\",
        r"\midrule",
    ]
    frames = {}
    frames["synth"] = pd.concat([pd.read_csv(f) for f in synth if f.exists()],
                                ignore_index=True)
    frames["real"] = pd.read_csv(real)
    for m in keep:
        cells = [disp[m]]
        for k in ("synth", "real"):
            f = frames.get(k)
            g = f[f.test == m] if f is not None else None
            if g is None or not len(g):
                cells += ["--", "--", "--"]
                continue
            bias = g.mean_bias.mean()
            cells += [f"{bias:+.3f}" if abs(round(bias, 3)) > 0 else "0.000",
                      _strip0(f"{g.coverage.mean():.3f}"),
                      f"{g.mean_ci_width.mean():.3f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{The four PPI-corrected CI constructions \evalstats{} returns "
        r"for \texttt{method=\textquotedbl auto\textquotedbl}, the numbers behind "
        r"the CI-method figure in the paper's appendix. "
        r"\textbf{Bias} is the mean signed error of the point estimate, "
        r"\textbf{Cov.}\ the mean interval coverage against a $0.95$ target, and "
        r"\textbf{Width} the mean interval width. Coverage sits at or above "
        r"nominal throughout. \texttt{bonett\_price} runs conservative on real "
        r"data because the null setup gives both arms the same human labels, "
        r"leaving its rectifier no disagreements to learn from.}",
        r"\label{supptab:ppi:cimethods}", r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("appendix-c", "appendix-e", "supp-factorial",
                                      "supp-alignment", "supp-power", "supp-ci"))
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    fns = {"appendix-c": lambda: appendix_c(a.alpha),
           "appendix-e": lambda: appendix_e(a.alpha),
           "supp-factorial": supp_factorial,
           "supp-alignment": supp_alignment,
           "supp-power": supp_power,
           "supp-ci": supp_ci_methods}
    print(fns[a.which]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
