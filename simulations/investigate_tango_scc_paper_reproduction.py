"""Reproduce Chang et al. (2024) Figure 1 and compare to what they report.

Their setup: N=30, pa=20%, panels pb in {0.3, 0.4, 0.5} (i.e. Delta = 0.1,
0.2, 0.3), rho on the x-axis, S=20000, nominal 95%.

What their Figure 1 shows (read off the plot):
  Wald   ~0.90-0.945 (below nominal, falling with Delta)
  Score  ~0.925-0.955, becoming ANTI-conservative as Delta grows
  SCC-S  ~0.955-0.978
  SCC-M/L slightly higher again

If our transcription is right we should land on those. If we come out far
higher (e.g. 0.99+) the printed equations and the implementation disagree
with the paper's own simulation, which is the thing to find out.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")
from evalstats.core.resampling import tango_scc_paired_ci, tango_paired_ci

ALPHA = 0.05

def paired_binary(rng, n, pa, pb, rho):
    """Joint cell probs for given marginals and Pearson correlation."""
    p11 = pa*pb + rho*np.sqrt(pa*(1-pa)*pb*(1-pb))
    p10 = pa - p11; p01 = pb - p11; p00 = 1 - p11 - p10 - p01
    probs = np.array([p11, p10, p01, p00])
    if np.any(probs < -1e-12): return None
    probs = np.clip(probs, 0, None); probs /= probs.sum()
    draw = rng.choice(4, size=n, p=probs)
    a = np.isin(draw, [0, 1]).astype(float)   # A=1 in cells 11,10
    b = np.isin(draw, [0, 2]).astype(float)   # B=1 in cells 11,01
    return a, b

def cover(N, pa, pb, rho, reps=4000, seed=3):
    rng = np.random.default_rng(seed)
    tgt = pa - pb                      # function's estimand: p(A=1)-p(B=1)
    out = {"score": [], "SCC-S": [], "SCC-M": [], "SCC-L": []}
    for _ in range(reps):
        r = paired_binary(rng, N, pa, pb, rho)
        if r is None: return None
        a, b = r
        lo, hi = tango_paired_ci(a, b, ALPHA); out["score"].append(lo <= tgt <= hi)
        for lbl, c in (("SCC-S",0.125), ("SCC-M",0.25), ("SCC-L",0.5)):
            lo, hi = tango_scc_paired_ci(a, b, ALPHA, c=c)
            out[lbl].append(lo <= tgt <= hi)
    return {k: float(np.mean(v)) for k, v in out.items()}

print("Chang et al. Fig.1 reproduction: N=30, pa=0.20, nominal 0.95")
print(f"{'pb':>5s} {'Delta':>6s} {'rho':>5s} {'score':>8s} {'SCC-S':>8s} {'SCC-M':>8s} {'SCC-L':>8s}")
for pb in (0.3, 0.4, 0.5):
    for rho in (0.1, 0.3, 0.5):
        r = cover(30, 0.20, pb, rho)
        if r is None:
            print(f"{pb:>5.2f} {pb-0.2:>6.2f} {rho:>5.2f}   (infeasible)"); continue
        print(f"{pb:>5.2f} {pb-0.2:>6.2f} {rho:>5.2f} {r['score']:>8.4f} "
              f"{r['SCC-S']:>8.4f} {r['SCC-M']:>8.4f} {r['SCC-L']:>8.4f}")
    print()
print("paper reports (read off Fig.1):  score ~0.925-0.955   SCC-S ~0.955-0.978")
