"""SELF-CHECK: identical results across every c could mean "the floor is
inert on good data" (the claim) OR "my patch never applies the floor" (a
bug in my own harness). Distinguish by measuring the BINDING RATE directly:
what fraction of (replicate, pair) cells have boot_se < c*obs_se.

Must show: ~0% on the non-degenerate DGP, clearly >0% on the degenerate
compare_e2e continuous cells.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")

import evalstats.api as api
from simulations.harness.cases.compare_e2e import (
    _apply_judge_noise, _effect_step_for, _effect_frac_for, _agreement_for)
from simulations.harness.scenarios.synthetic import (
    CONTINUOUS_SHAPES, LIKERT_SHAPES, sample_group_truth)

# Recreate the internals just far enough to expose boot_se, by calling the
# shipped function and recovering boot_se from T:  T=(theta_b-point)/boot_se.
# We cannot recover it from T alone, so instead re-run the SAME resample
# math here for one pair -- reusing the shipped lambda via the shipped fn's
# obs_se is not possible either, so compute both locally and cross-check
# obs_se against the shipped value to prove the replication is faithful.
from evalstats.ppi import (_adaptive_shrink_lambda, _analytic_mean_lambda_replicates,
                           _lambda_var_inflation)

def pair_boot_se(llm, lm, ia, ib, n_boot=2000, seed=1):
    m = ~np.isnan(lm[ia]) & ~np.isnan(lm[ib])
    d_unlab = llm[ia][~m]-llm[ib][~m]
    d_lt = lm[ia][m]-lm[ib][m]
    d_ll = llm[ia][m]-llm[ib][m]
    n_lab = d_lt.size; n_unlab = d_unlab.size
    f_u=d_unlab.mean(); f_l=d_lt.mean(); f_h=d_ll.mean()
    vu=d_unlab.var(ddof=1)/n_unlab; vl=d_lt.var(ddof=1)/n_lab; vh=d_ll.var(ddof=1)/n_lab
    cv=np.cov(d_lt,d_ll,ddof=1)[0,1]/n_lab
    lr=min(max(cv/(vu+vh),0.0),1.0)
    rv_t=d_lt.var(ddof=1); rv_l=d_ll.var(ddof=1)
    reps=None if (n_lab<=1 or rv_t<rv_l*1e-6) else _analytic_mean_lambda_replicates(d_lt,d_ll,vu,n_lab)
    lam=_adaptive_shrink_lambda(lr,reps,n_lab)
    extra=_lambda_var_inflation(f_u-f_h,reps)
    obs=np.sqrt(max(vl+lam*lam*(vu+vh)-2*lam*cv,0.0)+extra)
    g=np.random.default_rng(seed)
    iu=g.integers(0,n_unlab,size=(n_boot,n_unlab)); il=g.integers(0,n_lab,size=(n_boot,n_lab))
    us=d_unlab[iu]; lt=d_lt[il]; ll=d_ll[il]
    fl=lt.mean(1); fh=ll.mean(1)
    vub=us.var(1,ddof=1)/n_unlab; vlb=lt.var(1,ddof=1)/n_lab; vhb=ll.var(1,ddof=1)/n_lab
    cvb=((lt-fl[:,None])*(ll-fh[:,None])).sum(1)/(n_lab-1)/n_lab
    vb=np.maximum(vlb+lam*lam*(vub+vhb)-2*lam*cvb,0.0)+extra
    return np.sqrt(vb), obs

def degenerate_cells(eval_type, shapes, N, reps=15):
    shape=shapes[0]; st=_effect_step_for(eval_type,_effect_frac_for(eval_type))
    eff=np.arange(3.)*st; rng=np.random.default_rng(7)
    for _ in range(reps):
        t=sample_group_truth(shape,N,1,3,1.0,rng,effects=eff)[:,:,0]
        l=_apply_judge_noise(t,eval_type,rng,_agreement_for(eval_type))
        nl=max(1,round(N*0.40)); lab=rng.choice(N,nl,replace=False)
        lm=np.full_like(l,np.nan); lm[:,lab]=t[:,lab]
        yield l,lm

def nondegenerate_cells(N, n_lab, rho, reps=15):
    rng=np.random.default_rng(777)
    for _ in range(reps):
        base=rng.normal(0,1,N); t=np.empty((3,N))
        for i in range(3): t[i]=base+rng.normal(0,1,N)
        sig=np.sqrt(1/rho**2-1)
        l=t+rng.normal(0,sig,t.shape)
        lab=rng.choice(N,n_lab,replace=False)
        lm=np.full_like(l,np.nan); lm[:,lab]=t[:,lab]
        yield l,lm

CS=[0.10,0.20,0.30,0.50]
def report(tag, gen):
    rates={c:[] for c in CS}
    for l,lm in gen:
        for (ia,ib) in ((0,1),(0,2),(1,2)):
            bse,obs = pair_boot_se(l,lm,ia,ib)
            if obs<=1e-12: continue
            for c in CS: rates[c].append(float(np.mean(bse < c*obs)))
    print(f"  {tag:44s} " + "  ".join(f"c={c}:{np.mean(rates[c]):7.4%}" for c in CS))

print("Binding rate = fraction of bootstrap replicates with boot_se < c*obs_se\n")
print("DEGENERATE (compare_e2e generator, icc=1.0 shared items):")
report("continuous N=100 frac=.40", degenerate_cells("continuous",CONTINUOUS_SHAPES,100))
report("continuous N=200 frac=.40", degenerate_cells("continuous",CONTINUOUS_SHAPES,200))
report("likert     N=100 frac=.40", degenerate_cells("likert",LIKERT_SHAPES,100))
print("\nNON-DEGENERATE (per-arm item noise):")
report("N=200 n_lab=20 rho=0.99", nondegenerate_cells(200,20,0.99))
report("N=200 n_lab=40 rho=0.95", nondegenerate_cells(200,40,0.95))
report("N=100 n_lab=20 rho=0.99", nondegenerate_cells(100,20,0.99))
