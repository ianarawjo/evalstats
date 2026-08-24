"""Weighted interval score (WIS) robustness check for the paired binary CIs.

Our ci_paired tables score methods with a single interval score at
alpha=0.05, because 95% CIs are what practitioners report. Bracher, Ray,
Gneiting & Reich (2021) instead use the WEIGHTED interval score, eq. (1):

    WIS = 1/(K + 1/2) * ( w0*|y - m| + sum_k w_k * IS_{alpha_k} )
    w_k = alpha_k / 2,  w0 = 1/2

averaged over K=11 levels (alpha = 0.02, 0.05, 0.1, 0.2, ..., 0.9) plus the
predictive median. With those weights WIS ~ CRPS.

The question this answers: does scoring at 95% alone hide anything? A method
could be well calibrated at 95% and badly calibrated at 50%. Reports
(a) whether WIS reorders the methods relative to IS(0.05), and
(b) per-alpha coverage, which is the direct diagnostic.

Diagnostic settings (few reps); not for final numbers.
"""
import numpy as np

from evalstats.core.resampling import (
    mj_floor_paired_ci,
    bonett_price_paired_ci,
    newcombe_mover_paired_ci,
    tango_scc_paired_ci,
)
from evalstats.core.stats_utils import interval_score

# Bracher et al. (2021), section 2.2: K = 11 levels.
ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K = len(ALPHAS)

METHODS = {
    "mj_floor": lambda a, b, al: mj_floor_paired_ci(a, b, al),
    "bonett_price": lambda a, b, al: bonett_price_paired_ci(a, b, al),
    "newcombe_mover": lambda a, b, al: newcombe_mover_paired_ci(a, b, al),
    "tango_exact": lambda a, b, al: tango_scc_paired_ci(a, b, al, c=0.0),
}


def cell_probs(p_a: float, s: float, delta: float):
    """(p11, p10, p01, p00) for marginal p_a, discordance s, true diff delta."""
    p10 = (s + delta) / 2.0
    p01 = (s - delta) / 2.0
    p11 = p_a - p10
    p00 = 1.0 - p11 - p10 - p01
    probs = np.array([p11, p10, p01, p00])
    return None if np.any(probs < 1e-9) else probs


def draw(rng, probs, n):
    cell = rng.choice(4, size=n, p=probs)
    a = np.isin(cell, [0, 1]).astype(float)   # A=1 in cells 11, 10
    b = np.isin(cell, [0, 2]).astype(float)   # B=1 in cells 11, 01
    return a, b


def main(reps=300, seed=11):
    configs = []
    for p_a in (0.3, 0.5, 0.8):
        for s in (0.10, 0.25, 0.50):
            for delta in (0.0, 0.05, 0.15):
                if delta > s:
                    continue
                pr = cell_probs(p_a, s, delta)
                if pr is not None:
                    configs.append((p_a, s, delta, pr))
    sizes = (15, 30, 50, 100)
    print(f"{len(configs)} configs x {len(sizes)} sizes x {reps} reps, "
          f"{len(METHODS)} methods x {K} alphas")

    is05 = {m: [] for m in METHODS}
    wis = {m: [] for m in METHODS}
    cov = {(m, al): [] for m in METHODS for al in ALPHAS}

    rng = np.random.default_rng(seed)
    for (p_a, s, delta, probs) in configs:
        for n in sizes:
            acc_is, acc_wis = {m: 0.0 for m in METHODS}, {m: 0.0 for m in METHODS}
            acc_cov = {(m, al): 0 for m in METHODS for al in ALPHAS}
            for _ in range(reps):
                a, b = draw(rng, probs, n)
                point = float(a.mean() - b.mean())
                for mname, fn in METHODS.items():
                    total = 0.5 * abs(delta - point)      # w0 * |y - m|
                    for al in ALPHAS:
                        lo, hi = fn(a, b, al)
                        sc = interval_score(lo, hi, delta, al)
                        total += (al / 2.0) * sc
                        if lo <= delta <= hi:
                            acc_cov[(mname, al)] += 1
                        if al == 0.05:
                            acc_is[mname] += sc
                    acc_wis[mname] += total / (K + 0.5)
            for m in METHODS:
                is05[m].append(acc_is[m] / reps)
                wis[m].append(acc_wis[m] / reps)
                for al in ALPHAS:
                    cov[(m, al)].append(acc_cov[(m, al)] / reps)

    print("\n=== headline: does WIS reorder the methods? ===")
    print(f"{'method':<18}{'IS(0.05)':>11}{'rank':>6}{'WIS':>11}{'rank':>6}")
    is_rank = {m: r for r, m in enumerate(sorted(METHODS, key=lambda m: np.mean(is05[m])), 1)}
    wis_rank = {m: r for r, m in enumerate(sorted(METHODS, key=lambda m: np.mean(wis[m])), 1)}
    for m in METHODS:
        print(f"{m:<18}{np.mean(is05[m]):>11.4f}{is_rank[m]:>6}"
              f"{np.mean(wis[m]):>11.4f}{wis_rank[m]:>6}")

    print("\n=== coverage by nominal level (the actual diagnostic) ===")
    hdr = "".join(f"{int(100*(1-al)):>7}%" for al in ALPHAS)
    print(f"{'method':<18}{hdr}")
    for m in METHODS:
        row = "".join(f"{np.mean(cov[(m, al)]):>8.3f}" for al in ALPHAS)
        print(f"{m:<18}{row}")
    print(f"{'NOMINAL':<18}" + "".join(f"{1-al:>8.3f}" for al in ALPHAS))

    print("\n=== signed deviation from nominal (positive = conservative) ===")
    print(f"{'method':<18}{hdr}{'  worst':>9}")
    for m in METHODS:
        devs = [np.mean(cov[(m, al)]) - (1 - al) for al in ALPHAS]
        row = "".join(f"{d:>+8.3f}" for d in devs)
        print(f"{m:<18}{row}{min(devs):>+9.3f}")


if __name__ == "__main__":
    main()
