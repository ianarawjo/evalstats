"""Binary's null-case lambda shifts by -0.0386 under centring (vs -0.0073 on
continuous), and the original validation of the UNCENTRED version was
null-anchored. So: did centring move binary ANOVA's Type-I?

This is the check my continuous-only measurement could not make.
"""
import sys, warnings, inspect, textwrap
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")
from scipy import stats as _st
import evalstats.ppi as _ppi
import evalstats.tests as _t

ALPHA = 0.05
CENTRED = _ppi._pooled_k_group_lambda
_src = inspect.getsource(CENTRED)
_OLD = """    Y_lab = np.concatenate([_c(g) for g in Y_lab_groups])
    Y_hat_lab = np.concatenate([_c(g) for g in Y_hat_lab_groups])
    Y_hat_unlab = np.concatenate([_c(g) for g in Y_hat_unlab_groups])"""
_NEW = """    Y_lab = np.concatenate(Y_lab_groups)
    Y_hat_lab = np.concatenate(Y_hat_lab_groups)
    Y_hat_unlab = np.concatenate(Y_hat_unlab_groups)"""
_ns = dict(_ppi.__dict__); _ns["np"] = np
exec(compile(textwrap.dedent(_src.replace(_OLD,_NEW).replace(
    "def _pooled_k_group_lambda(", "def _old(")), "<o>", "exec"), _ns)
OLD = _ns["_old"]

def cell(rng, k, n, n_lab, p, flip, effect):
    """Binary k-group cell: base rate p, per-group effect step, judge flip."""
    gs, ls = [], []
    for i in range(k):
        pi = min(max(p + i*effect, 0.01), 0.99)
        t = (rng.random(n) < pi).astype(float)
        l = np.where(rng.random(n) < flip, 1.0-t, t)
        idx = rng.choice(n, n_lab, replace=False)
        lab = np.full(n, np.nan); lab[idx] = t[idx]
        gs.append(l); ls.append(lab)
    return gs, ls

def run(fn, k, n, n_lab, p, flip, effect, reps, seed):
    _ppi._pooled_k_group_lambda = fn
    rng = np.random.default_rng(seed); rej = []
    for _ in range(reps):
        gs, ls = cell(rng, k, n, n_lab, p, flip, effect)
        out = _t._ppi_anova_independent_f_stat(gs, ls, k, power_tune=True)
        if out is None: continue
        pv = float(_st.f.sf(out["f_corr"], out["dfn"], out["dfd"]))
        if np.isfinite(pv): rej.append(pv < ALPHA)
    _ppi._pooled_k_group_lambda = CENTRED
    return float(np.mean(rej)) if rej else np.nan

REPS = 600
se = np.sqrt(ALPHA*(1-ALPHA)/REPS)
print(f"BINARY ANOVA Type-I, reps={REPS}, MC SE={se:.4f} (flag if move > {3*se:.4f})\n")
print(f"{'p':>5s} {'flip':>5s} {'k':>3s} {'n':>5s} {'n_lab':>6s} "
      f"{'uncentred':>10s} {'centred':>9s} {'delta':>8s}")
for p, flip in ((0.50,0.10),(0.80,0.10),(0.90,0.05),(0.30,0.15)):
    for k, n, n_lab in ((3,200,60),(3,400,80),(5,200,60)):
        a = run(OLD,     k, n, n_lab, p, flip, 0.0, REPS, 909)
        b = run(CENTRED, k, n, n_lab, p, flip, 0.0, REPS, 909)
        flag = "  <-- MOVED" if abs(b-a) > 3*se else ""
        print(f"{p:>5.2f} {flip:>5.2f} {k:>3d} {n:>5d} {n_lab:>6d} "
              f"{a:>10.4f} {b:>9.4f} {b-a:>+8.4f}{flag}")
