"""Build the two LaTeX tables for the label-efficiency appendix.

Reads a finished run's results and per-method CSVs and writes
paper/appendix_label_efficiency_tables.tex. Bootstrap CIs on every median, so
a reader can see which differences are resolvable.

    python -m simulations.make_appendix_tables

Table 1 is the multiplier by data type and judge quality -- the headline
numbers. Table 2 is the one for a skeptical reviewer: attainment against the
control-variate bound, measured by two instruments that share no machinery
(power-curve inversion, and a direct variance ratio needing no curve).

Table 2 is deliberately NOT rounded into agreement. Three per-method entries
sit 1-3% above the bound with CIs excluding 1.0, so "consistent within Monte
Carlo error" would be false. What is true is that no test exceeds the bound on
BOTH instruments -- where one reads high the other reads low -- and that is
what the caption claims.
"""
import pandas as pd, numpy as np, glob
D = 'simulations/out/labeleff_final'
r = pd.read_csv(glob.glob(D + '/*_ppi_label_efficiency_results.csv')[0])
pm = pd.read_csv(glob.glob(D + '/figs_paper/*per_method.csv')[0])
rng = np.random.default_rng(7)

def bci(v, B=4000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    b = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(B)]
    return np.median(v), np.percentile(b, 2.5), np.percentile(b, 97.5)

def cell(m, lo, hi, p=2, times=False):
    # \times on the multiplier table only -- Table 2 holds dimensionless
    # attainment ratios, where a multiplication sign would be wrong.
    x = r"$\times$" if times else ""
    return f"{m:.{p}f}{x}\\,{{\\tiny[{lo:.{p}f},{hi:.{p}f}]}}"

u = r[r.well_conditioned & ~r.saturated]
NLABS = [30, 90, 200]          # 15 is too sparse to report (likert has none)
TIERS = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
L = []
L.append(r"\begin{table}[t]\centering\small")
L.append(r"\caption{Label-efficiency multiplier by data type, judge quality and "
         r"labeling budget. Each cell is the median over the four effect-size arms with a "
         r"bootstrap 95\% CI; $2\times$ means PPI matched the classical test's power on half "
         r"the human labels. Rows are the judge--human agreement $\rho^2$ a practitioner "
         r"would measure on a pilot set (Pearson for mean-based tests, Spearman for "
         r"rank-based; see Figure~\ref{fig:le-lookup}). Cells failing the inversion "
         r"conditioning check are excluded, and $n_{lab}=15$ is omitted entirely for lack of "
         r"usable cells.}")
L.append(r"\label{tab:le-mult}")
L.append(r"\begin{tabular}{r|" + "c" * len(NLABS) + r"}\toprule")
L.append(r"$\rho^2$ & " + " & ".join(rf"$n_{{lab}}={n}$" for n in NLABS) + r" \\")
for et, lbl in (('binary', 'Binary'), ('continuous', 'Continuous'), ('likert', 'Likert')):
    L.append(r"\midrule \multicolumn{" + str(len(NLABS) + 1) +
             r"}{l}{\textbf{" + lbl + r"}} \\")
    for t in TIERS:
        cs = []
        for n in NLABS:
            g = u[(u.eval_type == et) & (u.alignment_target == t) & (u.n_lab == n)]
            cs.append("--" if len(g) < 3 else cell(*bci(g.multiplier), times=True))
        L.append(f"{t:.2f} & " + " & ".join(cs) + r" \\")
L.append(r"\bottomrule\end{tabular}\end{table}")
L.append("")

q = pm[pm.well_conditioned & ~pm.saturated].copy()
q['inv'] = q.multiplier / q.predicted_mult
q['var'] = q.variance_multiplier / q.predicted_mult
L.append(r"\begin{table}[t]\centering\small")
L.append(r"\caption{Attainment: measured efficiency divided by the control-variate bound "
         r"$1/(1-\rho^2(1-n_{lab}/N))$, which it should not exceed. Measured two ways that "
         r"share no machinery -- inverting a classical power curve, and a direct variance "
         r"ratio needing no curve. Bootstrap 95\% CIs. No test exceeds the bound on both "
         r"instruments.}")
L.append(r"\label{tab:le-attain}")
L.append(r"\begin{tabular}{lccr}\toprule")
L.append(r"test & power-curve inversion & variance ratio & cells \\ \midrule")
for m_, lbl in (('ttest', r"$t$-test"), ('ttest_welch', r"Welch $t$"), ('paired_t', r"paired $t$"),
                ('mwu', r"Mann--Whitney"), ('wilcoxon', r"Wilcoxon")):
    g = q[q.method == m_]
    L.append(f"{lbl} & {cell(*bci(g['inv']), p=3)} & {cell(*bci(g['var']), p=3)} & {len(g)} \\\\")
L.append(r"\midrule")
L.append(f"pooled & {cell(*bci(q['inv']), p=3)} & {cell(*bci(q['var']), p=3)} & {len(q)} \\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
open('paper/appendix_label_efficiency_tables.tex', 'w').write("\n".join(L) + "\n")
print("\n".join(L))
