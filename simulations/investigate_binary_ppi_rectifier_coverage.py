"""Two questions before shipping a fix.

1. Is Tango-quadrature NEVER WORSE than the current plug-in? If it dominates,
   adopting it is free.
2. Where is the threshold below which NO construction is reliable? Coverage
   tracks the observed discordant count, so the warning should key on that.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")
from scipy.stats import norm, t as _t
from evalstats.core.resampling import mj_floor_paired_ci_from_diffs
ALPHA = 0.05

def one(rng, N, n_lab, p, flip):
    truth = (rng.random(N) < p).astype(float)
    llm = np.where(rng.random(N) < flip, 1.0 - truth, truth)
    idx = rng.choice(N, n_lab, replace=False)
    m = np.zeros(N, bool); m[idx] = True
    rect = truth[m] - llm[m]; unlab = llm[~m]
    fu = float(np.mean(unlab)); s2f = float(np.var(unlab, ddof=1))
    se_f = float(np.sqrt(s2f/len(unlab))); rhat = float(np.mean(rect))
    est = fu + rhat
    tc = _t.ppf(1-ALPHA/2, n_lab-1); zc = norm.ppf(1-ALPHA/2)
    v = s2f/len(unlab) + float(np.var(rect, ddof=1))/n_lab
    h = tc*np.sqrt(max(v,0.0))
    a_ok = max(0,est-h) <= p <= min(1,est+h)
    r_lo, r_hi = mj_floor_paired_ci_from_diffs(rect, ALPHA)
    hw_lo = np.sqrt(max(rhat-r_lo,0)**2 + (zc*se_f)**2)
    hw_hi = np.sqrt(max(r_hi-rhat,0)**2 + (zc*se_f)**2)
    t_ok = max(0.0,est-hw_lo) <= p <= min(1.0,est+hw_hi)
    return a_ok, t_ok, int(np.sum(rect != 0)), (min(1,est+h)-max(0,est-h)), \
           (min(1.0,est+hw_hi)-max(0.0,est-hw_lo))

print("1) Tango-quadrature vs plug-in across a wide grid (coverage)")
print(f"{'p':>5s} {'flip':>5s} {'n_lab':>6s} {'N':>6s} {'plugin':>8s} {'tango':>8s} {'delta':>8s} {'w ratio':>8s}")
worse = 0
for p in (0.3, 0.5, 0.7, 0.9, 0.95):
    for flip in (0.05, 0.15):
        for n_lab in (30, 100):
            for N in (200, 2000):
                rng = np.random.default_rng(11)
                A=T=0; wa=[]; wt=[]
                for _ in range(1500):
                    a,t,_,x,y = one(rng,N,n_lab,p,flip); A+=a; T+=t; wa.append(x); wt.append(y)
                a_c, t_c = A/1500, T/1500
                if t_c < a_c - 0.01: worse += 1
                print(f"{p:>5.2f} {flip:>5.2f} {n_lab:>6d} {N:>6d} {a_c:>8.4f} {t_c:>8.4f} "
                      f"{t_c-a_c:>+8.4f} {np.mean(wt)/np.mean(wa):>8.3f}")
print(f"\ncells where Tango is materially WORSE (>1pt): {worse}\n")

print("2) coverage vs OBSERVED discordant count (pooled over the same grid)")
rng = np.random.default_rng(3)
buckets = {}
for p in (0.3,0.5,0.7,0.9,0.95):
    for flip in (0.05,0.10,0.20,0.40):
        for n_lab in (20,30,60,120,250):
            for _ in range(400):
                a,t,nz,_,_ = one(rng,2000,n_lab,p,flip)
                buckets.setdefault(min(nz,40)//5*5, []).append((a,t))
print(f"{'#discordant':>12s} {'n':>7s} {'plugin cov':>11s} {'tango cov':>10s}")
for b in sorted(buckets):
    v = buckets[b]
    print(f"{str(b)+'-'+str(b+4):>12s} {len(v):>7d} "
          f"{np.mean([x[0] for x in v]):>11.4f} {np.mean([x[1] for x in v]):>10.4f}")
