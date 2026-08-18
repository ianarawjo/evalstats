import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
f = 60/600
theory = lambda r2: 1.0/(1.0 - r2*(1.0-f))
COL = {"normal":"#3b76af","laplace":"#61a05f","contam":"#c0392b","flip":"#8e6bb0","t3":"#d98c2b"}
LBL = {"normal":"Gaussian (our sims)","laplace":"Laplace","contam":"contaminated 8%",
       "flip":"sign-flip 12%","t3":"$t_3$"}

# ---------- Figure 1: crossover ----------
d = pd.read_csv("simulations/out/labeleff_rho2_full/rank_parametric_crossover.csv")
fig, ax = plt.subplots(figsize=(8.6, 5.4))
tc = {0.3:"#3b76af", 0.5:"#c0392b", 0.7:"#61a05f"}
for t, g in d.groupby("tier"):
    g = g.sort_values("bonus")
    ax.plot(g.bonus, g.power_gap, "o-", color=tc[t], lw=2.2, ms=6,
            label=fr"$\rho_P^2={t:.2f}$")
    gap, b = g.power_gap.values, g.bonus.values
    i = np.where(np.diff(np.sign(gap)))[0][0]
    w = -gap[i]/(gap[i+1]-gap[i]); bc = b[i]+w*(b[i+1]-b[i])
    ax.plot([bc], [0], "*", ms=17, color=tc[t], mec="white", mew=1.2, zorder=5)
    ax.axvline(g.crossover_bonus.iloc[0], color=tc[t], ls=":", lw=1.6, alpha=.8)
ax.axhline(0, color="#333", lw=1.2)
ax.axvline(0, color="#999", lw=1, ls="--")
ax.set_xlabel(r"rank bonus   $\rho_S^2-\rho_P^2$   (judge-error shape moves this)")
ax.set_ylabel("PPI power:  wilcoxon $-$ paired $t$")
ax.set_title("Where rank-based PPI overtakes parametric PPI\n"
             "stars = measured crossing;  dotted = predicted from the classical ARE alone",
             fontsize=11)
ax.annotate("Gaussian judge\n(ranks lose)", xy=(-0.022, -0.045), fontsize=8.5,
            ha="center", color="#555")
ax.annotate("contaminated judge\n(ranks win)", xy=(0.135, 0.026), fontsize=8.5,
            ha="center", color="#555")
ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=.25); ax.set_axisbelow(True)
fig.tight_layout()
o1 = "simulations/out/labeleff_rho2_full/rank_parametric_crossover.png"
fig.savefig(o1, dpi=150, bbox_inches="tight"); print("saved", o1)

# ---------- Figure 2: sufficiency collapse ----------
s = pd.read_csv("simulations/out/labeleff_rho2_full/rho2_sufficiency.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
xs = np.linspace(0.12, 0.88, 200)
for ax, (xcol, ycol, name, rho) in zip(
        (a1, a2),
        [("rP2","mult_t","paired $t$", r"$\rho_P^2$  (Pearson)"),
         ("rS2","mult_w","wilcoxon",   r"$\rho_S^2$  (Spearman)")]):
    ax.plot(xs, theory(xs), "-", color="#333", lw=1.8, zorder=1,
            label=r"$1/(1-\rho^2(1-n_{lab}/N))$")
    for sh, g in s.groupby("shape"):
        ax.plot(g[xcol], g[ycol], "o", ms=8, color=COL[sh], label=LBL[sh],
                mec="white", mew=1.1, zorder=3)
    ax.set_xlabel(rho); ax.set_title(name, fontsize=11.5)
    ax.grid(alpha=.25); ax.set_axisbelow(True)
a1.set_ylabel("label-efficiency multiplier")
a1.legend(fontsize=8.5, loc="upper left")
fig.suptitle("Is $\\rho^2$ sufficient? Every judge-noise shape must fall on ONE curve\n"
             "each family plotted against ITS OWN correlation", fontsize=11.5)
fig.tight_layout()
o2 = "simulations/out/labeleff_rho2_full/rho2_sufficiency.png"
fig.savefig(o2, dpi=150, bbox_inches="tight"); print("saved", o2)

# residuals about the theory curve = the quantitative collapse claim
print("\n=== residual from theory curve (measured/predicted), by shape ===")
for sh, g in s.groupby("shape"):
    at = (g.mult_t/theory(g.rP2)).values; aw = (g.mult_w/theory(g.rS2)).values
    print(f"  {sh:8s} paired_t {at.mean():.3f} (sd {at.std():.3f})   "
          f"wilcoxon {aw.mean():.3f} (sd {aw.std():.3f})")
at = (s.mult_t/theory(s.rP2)); aw = (s.mult_w/theory(s.rS2))
print(f"\n  ALL      paired_t {at.mean():.3f} +/- {at.std():.3f}   "
      f"wilcoxon {aw.mean():.3f} +/- {aw.std():.3f}   (n={len(s)} cells)")
