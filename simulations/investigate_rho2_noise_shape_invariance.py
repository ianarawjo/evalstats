"""Is the rho^2 -> label-efficiency multiplier map invariant to JUDGE NOISE SHAPE?

This is the question that decides whether the main label-efficiency sweep needs
a noise-shape axis. The rule of thumb claims rho^2 is a sufficient statistic for
savings. Theory says it must be, for MEAN tests: the mean's influence function
is linear in D, so the influence-function correlation IS Pearson, and the
multiplier is pinned by Pearson alone whatever shape the judge's errors take.

Pearson is pinned analytically: for Dhat = D + kappa*E with E unit-variance and
independent of D, rho_P^2 = 1/(1+kappa^2) EXACTLY, whatever E's shape, because
Pearson depends only on second moments. kappa=1.0 gives rho_P^2 = 0.50.

Answer: yes, to within ~3.5%. See notes/WHICH_RHO_FOR_WHICH_TEST.md -- including
why the rule of thumb nonetheless has to name a DIFFERENT correlation for rank
tests, and the 400-rep over-read this replaced.

Two fixes over the first pass:
 1. SEPARATE RNG streams for the human side (D, labelling) and the judge noise,
    so every shape sees the IDENTICAL D and identical labelled subset. The
    classical arm is then byte-identical across shapes and the comparison is
    paired -- all remaining variation is the judge.
 2. Bootstrap CI over replicates instead of an analytic SE. The analytic
    2/n + 2/n assumed the classical and PPI arms were independent; they are
    strongly positively correlated (PPI = classical + lambda*rectifier), so
    that SE is conservative and could hide a real effect.
"""
import numpy as np, warnings, json
from scipy import stats
warnings.filterwarnings("ignore")
from evalstats.tests import _ppi_paired_arrays

N, NLAB, REPS, NBOOT = 600, 60, 5000, 4000
KAPPA = 1.0
SHAPES = ("normal", "laplace", "contam", "flip")

def noi(k, rng, n):
    if k == "normal":  return rng.standard_normal(n)
    if k == "laplace": return rng.laplace(0, 1, n) / np.sqrt(2)
    if k == "contam":
        s = rng.standard_normal(n) * np.where(rng.random(n) < .08, 5.0, 1.0)
        return s / np.sqrt(.92 + .08 * 25)
    if k == "flip":
        s = rng.standard_normal(n)
        return s * np.where(rng.random(n) < .12, -3.0, 1.0) / np.sqrt(.88 + .12 * 9)

# Human side: drawn ONCE, replayed for every shape.
hr = np.random.default_rng(2024)
Ds   = [0.35 + hr.standard_normal(N) for _ in range(REPS)]
labs = [hr.choice(N, NLAB, replace=False) for _ in range(REPS)]
classical = np.array([Ds[i][labs[i]].mean() for i in range(REPS)])

ppi = {}
for shape in SHAPES:
    jr = np.random.default_rng(909)          # judge stream, same seed each shape
    er = np.random.default_rng(555)          # estimator stream
    vals = np.empty(REPS)
    for i in range(REPS):
        D = Ds[i]
        Dhat = D + KAPPA * noi(shape, jr, N)
        lab = np.zeros(N, bool); lab[labs[i]] = True
        a, b = Dhat, np.zeros(N)
        al, bl = np.where(lab, D, np.nan), np.where(lab, 0., np.nan)
        vals[i] = _ppi_paired_arrays(a, b, al, bl, np.mean, .05, 30, er,
                                     rectifier_func=np.mean).estimate
    ppi[shape] = vals
    print(f"  {shape} done", flush=True)

def ratio(idx, shape):
    return np.var(classical[idx]) / np.var(ppi[shape][idx])

rng = np.random.default_rng(3)
boot_idx = [rng.integers(0, REPS, REPS) for _ in range(NBOOT)]
print(f"\nclassical arm identical across shapes: var={np.var(classical):.6f}")
print(f"\n{'shape':9s} {'mult_t':>8s}  {'95% CI':>18s}")
pt_est, pt_boot = {}, {}
for s in SHAPES:
    pt_est[s] = ratio(np.arange(REPS), s)
    pt_boot[s] = np.array([ratio(b, s) for b in boot_idx])
    lo, hi = np.percentile(pt_boot[s], [2.5, 97.5])
    print(f"{s:9s} {pt_est[s]:8.4f}  [{lo:7.4f}, {hi:7.4f}]")

print(f"\n=== paired difference vs the Gaussian-judge baseline ===")
for s in SHAPES[1:]:
    d = pt_boot[s] - pt_boot["normal"]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"  {s:9s} {pt_est[s]-pt_est['normal']:+7.4f} "
          f"({(pt_est[s]/pt_est['normal']-1)*100:+5.2f}%)  "
          f"95% CI [{lo:+7.4f}, {hi:+7.4f}]  p={p:.4f}")
json.dump({s: float(pt_est[s]) for s in SHAPES}, open("/tmp/dgp/contam_paired.json","w"), indent=2)
