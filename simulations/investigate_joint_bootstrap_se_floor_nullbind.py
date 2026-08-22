"""THE FWER TEST THAT ACTUALLY BINDS UNDER THE NULL.

Prior attempts were vacuous, for two different reasons:
  - compare_e2e's own null has uniq(d_true)==1 (an EXACTLY constant paired
    difference), so boot_se never collapses and the floor never engages.
  - an "identical arms" null makes d_lab_true exactly 0 -> lambda=0 -> var_b
    and obs_var both reduce to the same fixed lambda_extra_var, so
    boot_se == obs_se by construction.
Both give binding 0% under the null, so "FWER unchanged" was guaranteed by
inertness rather than by the clamp being harmless.

FWER is a null-only property, so inertness under the null IS a valid proof of
unchanged Type-I -- but only for nulls where the floor is inert. The case the
api.py comment explicitly claims this protects ("two nearly identical arms")
is a null with SMALL BUT NONZERO variance in the paired difference, which is
exactly where the floor should bind. That null is built here:

  every arm shares a common base; each arm then perturbs a small random
  SUBSET of items by +/- delta with mean zero. E[difference] = 0 (a true
  null), but d_true has several distinct values and tiny variance, and
  lambda > 0 -- so boot_se genuinely collapses on some resamples.

`sparsity` (fraction of items perturbed) and `delta` sweep the degeneracy.
Reports binding% next to pre/post FWER so they are read together: a row with
binding ~0% carries no information; only rows with real binding test the
clamp.
"""
import sys, warnings, inspect, textwrap
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")

import evalstats.api as api
from evalstats.api import _ppi_romano_wolf_pvalues_from_joint_stats
from simulations.investigate_joint_bootstrap_se_floor_binding import pair_boot_se

ALPHA = 0.05
SHIPPED = api._ppi_bootstrap_t_joint_stats
_src = inspect.getsource(SHIPPED).replace(
    "_JOINT_BOOT_SE_REL_FLOOR * obs_se", "0.0 * obs_se"
).replace("def _ppi_bootstrap_t_joint_stats(", "def _prefix(")
_ns = dict(api.__dict__); _ns["np"] = np
exec(compile(textwrap.dedent(_src), "<prefix>", "exec"), _ns)
PREFIX = _ns["_prefix"]


def null_cell(rng, K, N, n_lab, sparsity, delta, rho=0.95):
    """True null (E[arm difference] == 0) with a near-degenerate,
    multi-valued paired difference."""
    base = rng.normal(0, 1, N)
    truth = np.tile(base, (K, 1))
    n_pert = max(1, int(round(sparsity * N)))
    for i in range(K):
        idx = rng.choice(N, size=n_pert, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_pert)
        signs -= signs.mean()          # mean-zero perturbation -> stays null
        truth[i, idx] = truth[i, idx] + delta * signs
    sig = np.sqrt(1.0 / rho**2 - 1.0)
    llm = truth + rng.normal(0, sig, truth.shape)
    lab = rng.choice(N, size=n_lab, replace=False)
    lm = np.full_like(llm, np.nan); lm[:, lab] = truth[:, lab]
    return llm, lm


def fwer(fn, K, N, n_lab, sparsity, delta, reps, n_boot, seed):
    rng = np.random.default_rng(seed)
    ents = [f"M{i}" for i in range(K)]; ei = {e: i for i, e in enumerate(ents)}
    pks = [(a, b) for i, a in enumerate(ents) for b in ents[i+1:]]
    rej, nv = [], []
    for _ in range(reps):
        llm, lm = null_cell(rng, K, N, n_lab, sparsity, delta)
        out = fn(llm, lm, pks, ei, n_boot,
                 np.random.default_rng(int(rng.integers(0, 2**31))), power_tune=True)
        if out is None: continue
        pe, ose, valid, T, tobs = out
        rw = _ppi_romano_wolf_pvalues_from_joint_stats(pe, ose, valid, T, tobs, pks)
        rej.append(any(p < ALPHA for p in rw.values())); nv.append(int(valid.sum()))
    return (float(np.mean(rej)) if rej else float("nan"),
            float(np.mean(nv)) if nv else float("nan"))


def bind(K, N, n_lab, sparsity, delta, reps, seed, c=0.20):
    rng = np.random.default_rng(seed); r, u = [], []
    for _ in range(reps):
        llm, lm = null_cell(rng, K, N, n_lab, sparsity, delta)
        for ia, ib in [(a, b) for a in range(K) for b in range(a+1, K)]:
            bse, obs = pair_boot_se(llm, lm, ia, ib, n_boot=400)
            if obs <= 1e-12: continue
            r.append(float(np.mean(bse < c*obs)))
    return float(np.mean(r)) if r else float("nan")


if __name__ == "__main__":
    REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    NB = 800
    se = np.sqrt(ALPHA*(1-ALPHA)/REPS)
    print(f"NULL-BINDING FWER  reps={REPS} n_boot={NB} alpha={ALPHA} "
          f"(MC SE={se:.4f}; breach if > {ALPHA+3*se:.4f})")
    print("rows with bind%~0 are uninformative; only binding rows test the clamp\n")
    print(f"{'K':>2s} {'N':>4s} {'nlab':>5s} {'sparse':>7s} {'delta':>6s} {'bind%':>8s} "
          f"{'FWERpre':>8s} {'FWERpost':>9s} {'delta':>8s} {'valid':>6s}")
    for K, N, n_lab in ((3, 200, 40), (3, 200, 20), (5, 200, 40)):
        for sparsity, delta in ((0.02, 0.5), (0.05, 0.5), (0.10, 0.3),
                                (0.02, 2.0), (0.05, 2.0)):
            b = bind(K, N, n_lab, sparsity, delta, max(15, REPS//15), 555)
            f0, _ = fwer(PREFIX,  K, N, n_lab, sparsity, delta, REPS, NB, 909)
            f1, v = fwer(SHIPPED, K, N, n_lab, sparsity, delta, REPS, NB, 909)
            d = f1 - f0
            flag = "  <-- BREACH" if f1 > ALPHA + 3*se else ""
            if abs(d) > 3*se: flag += "  <-- MOVED"
            print(f"{K:>2d} {N:>4d} {n_lab:>5d} {sparsity:>7.2f} {delta:>6.1f} {b:>8.3%} "
                  f"{f0:>8.4f} {f1:>9.4f} {d:>+8.4f} {v:>6.2f}{flag}")
