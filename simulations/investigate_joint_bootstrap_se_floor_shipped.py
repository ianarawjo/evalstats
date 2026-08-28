"""Post-fix validation of the SHIPPED code path (no monkeypatching).

Calls evalstats.api._ppi_bootstrap_t_joint_stats directly, so this measures
exactly what users get. Compares against the pre-fix behaviour by importing
the saved pre-fix source as a baseline.

Reports FWER (must stay <= alpha) and power on BOTH DGPs:
  degenerate     -- compare_e2e's generator (where the bug lives)
  non-degenerate -- per-arm item noise (where the floor must stay inert)
"""
import sys, warnings, inspect, textwrap
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")

import evalstats.api as api
from evalstats.api import (_ppi_romano_wolf_pvalues_from_joint_stats,
                           _M_b_from_T, _ppi_alpha_eff_from_M_b)
from simulations.harness.cases.compare_e2e import (
    _apply_judge_noise, _effect_step_for, _effect_frac_for, _agreement_for)
from simulations.harness.scenarios.synthetic import (
    CONTINUOUS_SHAPES, LIKERT_SHAPES, sample_group_truth)

ALPHA = 0.05
SHIPPED = api._ppi_bootstrap_t_joint_stats

# pre-fix baseline: same source with the floor coefficient forced to 0
_src = inspect.getsource(SHIPPED).replace(
    "_JOINT_BOOT_SE_REL_FLOOR * obs_se", "0.0 * obs_se"
).replace("def _ppi_bootstrap_t_joint_stats(", "def _prefix(")
_ns = dict(api.__dict__); _ns["np"] = np
exec(compile(textwrap.dedent(_src), "<prefix>", "exec"), _ns)
PREFIX = _ns["_prefix"]

SHAPES = {"continuous": CONTINUOUS_SHAPES, "likert": LIKERT_SHAPES}

def degen(rng, et, K, N, frac, effect_frac):
    shape = SHAPES[et][0]
    step = _effect_step_for(et, _effect_frac_for(et)) * effect_frac
    eff = np.arange(K, dtype=float) * step
    t = sample_group_truth(shape, N, 1, K, 1.0, rng, effects=eff)[:, :, 0]
    l = _apply_judge_noise(t, et, rng, _agreement_for(eval_type))
    nl = max(1, round(N*frac)); lab = rng.choice(N, nl, replace=False)
    lm = np.full_like(l, np.nan); lm[:, lab] = t[:, lab]
    return l, lm

def nondegen(rng, K, N, n_lab, rho, effect):
    base = rng.normal(0,1,N); t = np.empty((K,N))
    for i in range(K): t[i] = base + rng.normal(0,1,N) + i*effect
    sig = np.sqrt(1/rho**2 - 1)
    l = t + rng.normal(0, sig, t.shape)
    lab = rng.choice(N, n_lab, replace=False)
    lm = np.full_like(l, np.nan); lm[:, lab] = t[:, lab]
    return l, lm

def measure(fn, gen, K, reps, n_boot, seed):
    rng = np.random.default_rng(seed)
    ents=[f"M{i}" for i in range(K)]; ei={e:i for i,e in enumerate(ents)}
    pks=[(a,b) for i,a in enumerate(ents) for b in ents[i+1:]]
    anyrej, ext, ae = [], [], []
    for _ in range(reps):
        l, lm = gen(rng)
        out = fn(l, lm, pks, ei, n_boot, np.random.default_rng(int(rng.integers(0,2**31))), power_tune=True)
        if out is None: continue
        pe,ose,valid,T,tobs = out
        rw=_ppi_romano_wolf_pvalues_from_joint_stats(pe,ose,valid,T,tobs,pks)
        anyrej.append(any(p<ALPHA for p in rw.values()))
        ext.append(rw[("M0",f"M{K-1}")]<ALPHA)
        ae.append(_ppi_alpha_eff_from_M_b(_M_b_from_T(T,valid),1-ALPHA))
    return (np.mean(anyrej) if anyrej else np.nan,
            np.mean(ext) if ext else np.nan,
            np.median(ae) if ae else np.nan)

if __name__ == "__main__":
    REPS=int(sys.argv[1]) if len(sys.argv)>1 else 400
    NB=800
    se=np.sqrt(ALPHA*(1-ALPHA)/REPS)
    print(f"SHIPPED vs PRE-FIX   reps={REPS} n_boot={NB} alpha={ALPHA} (MC SE={se:.4f})\n")
    print(f"{'DGP / condition':38s} {'ver':8s} {'FWER':>8s} {'power':>8s} {'a_eff':>10s}")
    rows=[]
    for et,K,N,frac in (("continuous",3,100,0.40),("continuous",3,200,0.40),
                        ("continuous",5,100,0.40),("likert",3,50,0.40),
                        ("likert",3,100,0.40),("likert",3,200,0.40)):
        for tag,fn in (("prefix",PREFIX),("SHIPPED",SHIPPED)):
            f,_,a = measure(fn, lambda r,et=et,K=K,N=N,frac=frac: degen(r,et,K,N,frac,0.0), K, REPS, NB, 4242)
            _,p,_ = measure(fn, lambda r,et=et,K=K,N=N,frac=frac: degen(r,et,K,N,frac,1.0), K, REPS, NB, 31337)
            flag="  <-- BREACH" if f>ALPHA+3*se else ""
            print(f"{'degen '+et+f' k={K} N={N}':38s} {tag:8s} {f:>8.4f} {p:>8.4f} {a:>10.6f}{flag}")
        print()
    for K,N,nl,rho in ((3,200,20,0.99),(3,200,40,0.95),(5,200,40,0.95),(3,100,20,0.99)):
        for tag,fn in (("prefix",PREFIX),("SHIPPED",SHIPPED)):
            f,_,a = measure(fn, lambda r,K=K,N=N,nl=nl,rho=rho: nondegen(r,K,N,nl,rho,0.0), K, REPS, NB, 777)
            _,p,_ = measure(fn, lambda r,K=K,N=N,nl=nl,rho=rho: nondegen(r,K,N,nl,rho,0.30), K, REPS, NB, 31337)
            flag="  <-- BREACH" if f>ALPHA+3*se else ""
            print(f"{f'nondegen k={K} N={N} nlab={nl} rho={rho}':38s} {tag:8s} {f:>8.4f} {p:>8.4f} {a:>10.6f}{flag}")
        print()
