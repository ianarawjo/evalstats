"""FWER validation on a NON-DEGENERATE DGP.

validate.py's FWER was identical at every c, which means the floor never
bound under its null -- because compare_e2e's generator (icc=1.0, shared
items, effects as a pure shift) makes the null paired difference EXACTLY
constant, so those pairs are dropped as degenerate before the floor can
matter. That is not a real test of whether the floor costs FWER control.

Here each arm gets its OWN per-item noise, so the paired difference has
genuine spread and every pair stays valid. Sweeps judge quality (rho) and
n_lab, including the regimes where boot_se is most variable (small n_lab,
excellent judge) -- exactly where an over-aggressive floor would show up as
FWER inflation.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
sys.path.insert(0, "/private/tmp/claude-501/-Users-ianarawjo-Documents-prompt-stats/f2b00edf-b3f0-44a7-88c1-f7e3cc56181a/scratchpad/rwfix")
warnings.filterwarnings("ignore")

import patched
from evalstats.api import (_ppi_romano_wolf_pvalues_from_joint_stats,
                           _M_b_from_T, _ppi_alpha_eff_from_M_b)
ALPHA = 0.05


def cell(rng, K, N, n_lab, rho, effect):
    """Each arm: base item quality + its OWN per-item noise -> non-degenerate
    paired differences. Judge sees truth + noise calibrated to correlation rho."""
    base = rng.normal(0, 1, N)
    truth = np.empty((K, N))
    for i in range(K):
        truth[i] = base + rng.normal(0, 1, N) + i * effect
    sig = np.sqrt(1.0 / rho**2 - 1.0) if rho < 1 else 0.0
    llm = truth + rng.normal(0, sig, truth.shape)
    lab_idx = rng.choice(N, size=n_lab, replace=False)
    lm = np.full_like(llm, np.nan); lm[:, lab_idx] = truth[:, lab_idx]
    return llm, lm


def run(fn, K, N, n_lab, rho, effect, reps, n_boot, seed):
    rng = np.random.default_rng(seed)
    ents = [f"M{i}" for i in range(K)]
    ei = {e: i for i, e in enumerate(ents)}
    pks = [(a, b) for i, a in enumerate(ents) for b in ents[i+1:]]
    anyrej, ext, aeffs, nvalid = [], [], [], []
    for _ in range(reps):
        llm, lm = cell(rng, K, N, n_lab, rho, effect)
        out = fn(llm, lm, pks, ei, n_boot,
                 np.random.default_rng(int(rng.integers(0, 2**31))), power_tune=True)
        if out is None:
            continue
        pe, ose, valid, T, tobs = out
        rw = _ppi_romano_wolf_pvalues_from_joint_stats(pe, ose, valid, T, tobs, pks)
        anyrej.append(any(p < ALPHA for p in rw.values()))
        ext.append(rw[("M0", f"M{K-1}")] < ALPHA)
        aeffs.append(_ppi_alpha_eff_from_M_b(_M_b_from_T(T, valid), 1 - ALPHA))
        nvalid.append(int(valid.sum()))
    return (float(np.mean(anyrej)), float(np.mean(ext)),
            float(np.median(aeffs)), float(np.mean(nvalid)))


if __name__ == "__main__":
    REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    NBOOT = 800
    CS = [0.0, 0.10, 0.20, 0.30, 0.50]
    fns = {c: patched.make(c) for c in CS}
    se = np.sqrt(ALPHA*(1-ALPHA)/REPS)
    print(f"NON-DEGENERATE DGP  reps={REPS} n_boot={NBOOT} alpha={ALPHA} "
          f"(MC SE={se:.4f}; flag if FWER > {ALPHA+3*se:.4f})")
    CONDS = [
        # K,  N,  n_lab, rho   -- stress: small n_lab and/or excellent judge
        (3, 200,  20, 0.99), (3, 200,  20, 0.90), (3, 200,  40, 0.99),
        (3, 200,  80, 0.95), (3, 100,  20, 0.99), (5, 200,  40, 0.95),
        (5, 200,  20, 0.99), (3, 400, 160, 0.80),
    ]
    for K, N, n_lab, rho in CONDS:
        sid = 1-(1-ALPHA)**(1.0/(K*(K-1)//2))
        print(f"\n--- k={K} N={N} n_lab={n_lab} rho={rho}  (Sidak={sid:.4f}) ---")
        print(f"{'c':>6s} {'FWER(null)':>11s} {'power(d=.3)':>12s} {'a_eff':>10s} {'valid pairs':>12s}")
        for c in CS:
            f, _, a, nv = run(fns[c], K, N, n_lab, rho, 0.0, REPS, NBOOT, 777)
            _, p, _, _  = run(fns[c], K, N, n_lab, rho, 0.30, REPS, NBOOT, 31337)
            flag = "  <-- FWER BREACH" if f > ALPHA + 3*se else ""
            print(f"{c:>6.2f} {f:>11.4f} {p:>12.4f} {a:>10.6f} {nv:>12.2f}{flag}")
