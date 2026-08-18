"""Figures for the two rho^2 robustness experiments.

Importable (cases/pvalues.py's save_ppi_rho2_robustness_plots calls
plot_sufficiency/plot_crossover directly on in-memory frames) and runnable
standalone, in which case it reads the CSVs the investigate_* scripts write.

See notes/WHICH_RHO_FOR_WHICH_TEST.md and
notes/RANK_VS_PARAMETRIC_CROSSOVER.md for what each figure establishes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = 60 / 600  # n_lab / N in both experiments
COL = {"normal": "#3b76af", "laplace": "#61a05f", "contam": "#c0392b",
       "flip": "#8e6bb0", "t3": "#d98c2b"}
LBL = {"normal": "Gaussian (our sims)", "laplace": "Laplace", "contam": "contaminated 8%",
       "flip": "sign-flip 12%", "t3": "$t_3$"}


def _theory(r2):
    return 1.0 / (1.0 - np.asarray(r2) * (1.0 - F))


def plot_sufficiency(df: pd.DataFrame, out_path: str) -> str:
    """Collapse check: every judge-noise shape must fall on ONE curve, with
    each test family plotted against ITS OWN correlation (Pearson for
    paired_t, Spearman for wilcoxon). Collapse = rho^2 is a sufficient
    statistic for the multiplier, which is what the rule of thumb claims."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    xs = np.linspace(0.12, 0.88, 200)
    for ax, (xcol, ycol, name, rho) in zip(
            (a1, a2),
            [("rP2", "mult_t", "paired $t$", r"$\rho_P^2$  (Pearson)"),
             ("rS2", "mult_w", "wilcoxon", r"$\rho_S^2$  (Spearman)")]):
        ax.plot(xs, _theory(xs), "-", color="#333", lw=1.8, zorder=1,
                label=r"$1/(1-\rho^2(1-n_{lab}/N))$")
        for sh, g in df.groupby("shape"):
            ax.plot(g[xcol], g[ycol], "o", ms=8, color=COL.get(sh, "#777"),
                    label=LBL.get(sh, sh), mec="white", mew=1.1, zorder=3)
        ax.set_xlabel(rho)
        ax.set_title(name, fontsize=11.5)
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)
    a1.set_ylabel("label-efficiency multiplier")
    a1.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Is $\\rho^2$ sufficient? Every judge-noise shape must fall on ONE curve\n"
                 "each family plotted against ITS OWN correlation", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_crossover(df: pd.DataFrame, out_path: str) -> str:
    """Where PPI power swaps between rank-based and parametric tests as the
    judge's error SHAPE moves rho_S^2 at pinned rho_P^2. Stars mark the
    measured crossing; dotted lines the crossing predicted from the classical
    ARE alone, which under-predicts it (see the note)."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    tc = {0.3: "#3b76af", 0.5: "#c0392b", 0.7: "#61a05f"}
    for t, g in df.groupby("tier"):
        g = g.sort_values("bonus")
        col = tc.get(round(float(t), 2), "#777")
        ax.plot(g.bonus, g.power_gap, "o-", color=col, lw=2.2, ms=6,
                label=fr"$\rho_P^2={t:.2f}$")
        gap, b = g.power_gap.values, g.bonus.values
        i = np.where(np.diff(np.sign(gap)))[0]
        if len(i):
            i = i[0]
            bc = b[i] + (-gap[i] / (gap[i + 1] - gap[i])) * (b[i + 1] - b[i])
            ax.plot([bc], [0], "*", ms=17, color=col, mec="white", mew=1.2, zorder=5)
        ax.axvline(g.crossover_bonus.iloc[0], color=col, ls=":", lw=1.6, alpha=.8)
    ax.axhline(0, color="#333", lw=1.2)
    ax.axvline(0, color="#999", lw=1, ls="--")
    ax.set_xlabel(r"rank bonus   $\rho_S^2-\rho_P^2$   (judge-error shape moves this)")
    ax.set_ylabel("PPI power:  wilcoxon $-$ paired $t$")
    ax.set_title("Where rank-based PPI overtakes parametric PPI\n"
                 "stars = measured crossing;  dotted = predicted from the classical ARE alone",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    d = "simulations/out/labeleff_rho2_full"
    print(plot_crossover(pd.read_csv(f"{d}/rank_parametric_crossover.csv"),
                         f"{d}/rank_parametric_crossover.png"))
    print(plot_sufficiency(pd.read_csv(f"{d}/rho2_sufficiency.csv"),
                           f"{d}/rho2_sufficiency.png"))


if __name__ == "__main__":
    main()
