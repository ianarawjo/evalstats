"""Fits and reports the label-efficiency "N-formula" rule of thumb from a
``*_ppi_nformula_results.csv`` (see ``simulations/harness/cases/pvalues.py``'s
``run_ppi_nformula_check``), for citing/defending in the paper's appendix.

MODEL. PPI++'s asymptotic variance for the corrected estimator decomposes
(informally) into a labeled-subset term and a judge-pool term:

    Var(theta_hat_PPI) ~= lambda*^2 * Var(f(X)) / N + (1 - rho^2) * Var(Y) / N_lab

where rho is the judge-truth correlation, N_lab the labeled count, N the
total (judge-scored) item count, and lambda* the PPI++ power-tuning
weight. Setting this equal to a classical human-only estimator's variance
Var(Y)/N_lab' (same variance => same power at a fixed alpha) and dividing
through by Var(Y)/N_lab gives:

    1/multiplier = N_lab/N_lab' = c0*(1 - rho^2) + c1*(N_lab/N)

where multiplier := N_lab'/N_lab (the label-efficiency multiplier this
harness already reports), c0 is theoretically 1, and c1 = lambda*^2 *
Var(f(X))/Var(Y) is an eval-type-specific variance-ratio constant. This
script fits (c0, c1) per eval_type via OLS THROUGH THE ORIGIN (no
intercept -- the derivation above has none) with HC3 heteroskedasticity-
robust standard errors, then runs two Wald tests against the SAME fitted
model:
  - c1 = 0: is the N/N_lab term a statistically real improvement over the
    original N-independent formula (1/multiplier = c0*(1-rho^2))?
  - c0 = 1: is the fitted calibration constant consistent with the raw,
    uncalibrated asymptotic prediction?

rho is the eval type's own realized alignment metric (Pearson r for
continuous; weighted kappa for likert; kappa for binary) from the CSV's
`alignment_value` column -- an approximation to a "true" correlation for
binary/likert (kappa isn't literally a Pearson r), carried over unchanged
from the exploratory fit this script formalizes; see the CSV's own
alignment_metric column for which metric backs each eval_type's rows.

Saturated cells (see LabelEfficiencyPoint.saturated's docstring in
cases/pvalues.py) are EXCLUDED throughout: `equiv_n_lab` -- and so
`multiplier` -- is a meaningless lower-bound artifact for those rows, not
a real observation of the quantity this model predicts.

Usage:
  python -m simulations.fit_nformula_rule_of_thumb path/to/..._ppi_nformula_results.csv
  python -m simulations.fit_nformula_rule_of_thumb path/to/..._ppi_nformula_results.csv \\
      --out-coefs coefs.csv --out-latex table.tex
"""

from __future__ import annotations

import argparse

import pandas as pd
import statsmodels.formula.api as smf

from simulations.harness.latex_tables import booktabs_table

EVAL_TYPE_ORDER = ["binary", "continuous", "likert"]
EVAL_TYPE_LABEL = {"binary": "Binary", "continuous": "Continuous", "likert": "Likert"}
REQUIRED_COLUMNS = {"eval_type", "n", "n_lab", "alignment_value", "multiplier", "saturated"}


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {sorted(missing)}")
    df["saturated"] = df["saturated"].astype(str).str.strip().str.lower() == "true" if df["saturated"].dtype == object else df["saturated"].astype(bool)
    return df


def fit_eval_type(df: pd.DataFrame, eval_type: str) -> dict:
    """Fits the full (c0, c1) model for one eval_type's unsaturated rows,
    plus the two Wald tests described in this module's docstring. Returns
    a flat dict of everything build_coef_table/build_latex_table need --
    coefficients, robust SEs, 95% CIs, R^2, n_obs/n_excluded, and both
    Wald tests' (F, p) pairs."""
    sub = df[(df["eval_type"] == eval_type) & (~df["saturated"])].copy()
    n_excluded = int((df["eval_type"] == eval_type).sum() - len(sub))
    sub["one_minus_rho2"] = 1.0 - sub["alignment_value"] ** 2
    sub["nlab_over_n"] = sub["n_lab"] / sub["n"]
    sub["y"] = 1.0 / sub["multiplier"]

    fit = smf.ols("y ~ 0 + one_minus_rho2 + nlab_over_n", data=sub).fit(cov_type="HC3")
    c0, c1 = fit.params["one_minus_rho2"], fit.params["nlab_over_n"]
    c0_se, c1_se = fit.bse["one_minus_rho2"], fit.bse["nlab_over_n"]
    ci = fit.conf_int(alpha=0.05)
    c0_ci = tuple(ci.loc["one_minus_rho2"])
    c1_ci = tuple(ci.loc["nlab_over_n"])

    wald_c1_zero = fit.wald_test("nlab_over_n = 0", use_f=True, scalar=True)
    wald_c0_one = fit.wald_test("one_minus_rho2 = 1", use_f=True, scalar=True)

    pred_multiplier = 1.0 / (c0 * sub["one_minus_rho2"] + c1 * sub["nlab_over_n"])
    rel_err = (sub["multiplier"] / pred_multiplier - 1.0).abs()

    return dict(
        eval_type=eval_type, n_obs=len(sub), n_excluded_saturated=n_excluded,
        c0=c0, c0_se=c0_se, c0_ci_lo=c0_ci[0], c0_ci_hi=c0_ci[1],
        c1=c1, c1_se=c1_se, c1_ci_lo=c1_ci[0], c1_ci_hi=c1_ci[1],
        r2_uncentered=fit.rsquared,
        f_c1zero=float(wald_c1_zero.statistic), p_c1zero=float(wald_c1_zero.pvalue),
        f_c0one=float(wald_c0_one.statistic), p_c0one=float(wald_c0_one.pvalue),
        median_abs_rel_err=float(rel_err.median()), q75_abs_rel_err=float(rel_err.quantile(0.75)),
    )


def print_report(fits: list[dict]) -> None:
    print(f"\n{'=' * 92}\n  LABEL-EFFICIENCY N-FORMULA FIT\n"
          f"  1/multiplier = c0*(1-rho^2) + c1*(N_lab/N), OLS through the origin, HC3 robust SE\n{'=' * 92}")
    for f in fits:
        print(f"\n  [{f['eval_type']}]  n_obs={f['n_obs']} (excluded {f['n_excluded_saturated']} saturated), "
              f"R^2(uncentered)={f['r2_uncentered']:.3f}")
        print(f"    c0 = {f['c0']:.3f}  (SE {f['c0_se']:.3f}, 95% CI [{f['c0_ci_lo']:.3f}, {f['c0_ci_hi']:.3f}])")
        print(f"    c1 = {f['c1']:.3f}  (SE {f['c1_se']:.3f}, 95% CI [{f['c1_ci_lo']:.3f}, {f['c1_ci_hi']:.3f}])")
        print(f"    Wald test c1=0 (is the N/N_lab term needed?):  F={f['f_c1zero']:.2f}, p={f['p_c1zero']:.2e}")
        print(f"    Wald test c0=1 (matches raw asymptotic theory?): F={f['f_c0one']:.2f}, p={f['p_c0one']:.3f}"
              + ("  (not rejected)" if f['p_c0one'] > 0.05 else "  (rejected at alpha=0.05)"))
        print(f"    median |predicted/actual - 1| = {f['median_abs_rel_err']:.1%}  "
              f"(75th pct: {f['q75_abs_rel_err']:.1%})")
    print()


def build_coef_csv(fits: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(fits)[[
        "eval_type", "n_obs", "n_excluded_saturated", "c0", "c0_se", "c0_ci_lo", "c0_ci_hi",
        "c1", "c1_se", "c1_ci_lo", "c1_ci_hi", "r2_uncentered",
        "f_c1zero", "p_c1zero", "f_c0one", "p_c0one", "median_abs_rel_err", "q75_abs_rel_err",
    ]]


def build_latex_table(fits: list[dict], *, caption: str | None, label: str) -> str:
    by_et = {f["eval_type"]: f for f in fits}
    columns = ["Data type", r"$c_0$", r"$c_1$", r"$R^2$", r"$n$", r"Median error"]
    rows = []
    for et in EVAL_TYPE_ORDER:
        if et not in by_et:
            continue
        f = by_et[et]
        rows.append([
            EVAL_TYPE_LABEL[et],
            rf"{f['c0']:.2f} $\pm$ {f['c0_se']:.2f}",
            rf"{f['c1']:.2f} $\pm$ {f['c1_se']:.2f}",
            f"{f['r2_uncentered']:.2f}",
            str(f["n_obs"]),
            f"{f['median_abs_rel_err'] * 100:.0f}\\%",  # literal "%" is a LaTeX comment char -- must be escaped
        ])
    if caption is None:
        caption = (
            r"Fitted coefficients for the label-efficiency rule of thumb "
            r"$N_\mathrm{lab}' \approx N_\mathrm{lab} / \left[c_0(1-\rho^2) + c_1 (N_\mathrm{lab}/N)\right]$, "
            r"fit by OLS through the origin (HC3-robust SEs shown as $\pm$) on the label-efficiency "
            r"N-formula sweep (run\_ppi\_nformula\_check), excluding saturated cells. $R^2$ is uncentered "
            r"(appropriate for a through-origin fit). Median error is the median absolute relative error of "
            r"the fitted multiplier against the observed multiplier."
        )
    return booktabs_table(
        caption=caption, label=label, columns=columns, rows=rows, col_align="lrrrrr",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to a *_ppi_nformula_results.csv")
    parser.add_argument("--out-coefs", default=None, help="Optional path to save fitted coefficients as CSV")
    parser.add_argument("--out-latex", default=None, help="Optional path to save the LaTeX coefficients table")
    parser.add_argument("--label", default="tab:nformula_fit", help="LaTeX \\label{} key")
    parser.add_argument("--caption", default=None, help="Table caption (a sensible default is generated if omitted)")
    args = parser.parse_args()

    df = load_results(args.csv_path)
    eval_types = [et for et in EVAL_TYPE_ORDER if et in set(df["eval_type"])]
    fits = [fit_eval_type(df, et) for et in eval_types]

    print_report(fits)

    if args.out_coefs:
        build_coef_csv(fits).to_csv(args.out_coefs, index=False)
        print(f"Saved coefficients: {args.out_coefs}")

    table = build_latex_table(fits, caption=args.caption, label=args.label)
    if args.out_latex:
        with open(args.out_latex, "w", encoding="utf-8") as fh:
            fh.write(table)
        print(f"Saved LaTeX table: {args.out_latex}")
    else:
        print(table)


if __name__ == "__main__":
    main()
