"""Blanket SCC over-covers (0.985-0.994); plain Tango under-covers only where
discordant counts are low AND one-sided. So route between them.

Candidate signals for "plain Tango is about to fail here":
  n_disc      total discordant items
  min_side    min(n_plus, n_minus) -- captures the ASYMMETRY that distinguishes
              p=0.90 (n+ >> n-, Tango fails) from p=0.50 (n+ ~ n-, Tango fine)
              even though both have ~2.4 total events
"""
import sys, warnings
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")
from scipy.stats import norm
from evalstats.core.resampling import mj_floor_paired_ci_from_diffs, tango_scc_paired_ci

ALPHA = 0.05; Z = norm.ppf(1 - ALPHA/2)

def combine(lo, hi, rect, se_u, est):
    hl = np.sqrt(max(rect-lo, 0)**2 + (Z*se_u)**2)
    hh = np.sqrt(max(hi-rect, 0)**2 + (Z*se_u)**2)
    return max(0.0, est-hl), min(1.0, est+hh)

RULES = ["tango", "scc", "route n<10", "route n<20", "route minside<5"]

def run(N, n_lab, p, flip, reps=2500, seed=7):
    rng = np.random.default_rng(seed)
    cov = {r: [] for r in RULES}; wid = {r: [] for r in RULES}
    for _ in range(reps):
        truth = (rng.random(N) < p).astype(float)
        llm = np.where(rng.random(N) < flip, 1.0-truth, truth)
        idx = rng.choice(N, n_lab, replace=False)
        m = np.zeros(N, bool); m[idx] = True
        tl, ll = truth[m], llm[m]; unlab = llm[~m]
        fu = float(np.mean(unlab)); s2f = float(np.var(unlab, ddof=1))
        se_u = float(np.sqrt(s2f/len(unlab)))
        d = tl - ll; rect = float(np.mean(d)); est = fu + rect
        npl = int(np.sum(d > 0)); nmi = int(np.sum(d < 0)); nd = npl + nmi
        try:
            t_lo, t_hi = mj_floor_paired_ci_from_diffs(d, ALPHA)
            s_lo, s_hi = tango_scc_paired_ci(tl, ll, ALPHA, c=0.125)
        except Exception:
            continue
        picks = {"tango": (t_lo, t_hi), "scc": (s_lo, s_hi),
                 "route n<10": (s_lo, s_hi) if nd < 10 else (t_lo, t_hi),
                 "route n<20": (s_lo, s_hi) if nd < 20 else (t_lo, t_hi),
                 "route minside<5": (s_lo, s_hi) if min(npl, nmi) < 5 else (t_lo, t_hi)}
        for r, (lo, hi) in picks.items():
            a, b = combine(lo, hi, rect, se_u, est)
            cov[r].append(a <= p <= b); wid[r].append(b-a)
    return ({r: np.mean(v) for r, v in cov.items()},
            {r: np.mean(v) for r, v in wid.items()})

print("coverage (width) -- nominal 0.95\n")
print(f"{'p':>5s} {'flip':>5s} {'n_lab':>6s} {'N':>6s} " + "".join(f"{r:>17s}" for r in RULES))
for p, flip, n_lab in ((0.50,0.08,30),(0.90,0.08,30),(0.95,0.08,30),
                       (0.50,0.30,120),(0.90,0.30,120),(0.70,0.15,60)):
    for N in (1000, 5000):
        c, w = run(N, n_lab, p, flip)
        print(f"{p:>5.2f} {flip:>5.2f} {n_lab:>6d} {N:>6d} "
              + "".join(f"{c[r]:>10.4f}({w[r]:.2f})" for r in RULES))
