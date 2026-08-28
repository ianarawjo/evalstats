"""Direct mechanism check: does centring actually change lambda, and does the
shipped (uncentred) lambda DRIFT with the effect size the way the two-group
sibling's did before 836f811?

The f-stat dict does not expose lambda, so the power sweep could not show
this. Call both lambda functions on identical data instead.
"""
import sys, warnings, inspect, textwrap
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")
import evalstats.ppi as _ppi

SHIPPED = _ppi._pooled_k_group_lambda
_src = inspect.getsource(SHIPPED)
_OLD = """    Y_lab = np.concatenate(Y_lab_groups)
    Y_hat_lab = np.concatenate(Y_hat_lab_groups)
    Y_hat_unlab = np.concatenate(Y_hat_unlab_groups)"""
_NEW = """    def _c(x):
        x = np.asarray(x, dtype=float)
        return x - x.mean() if x.size else x
    Y_lab = np.concatenate([_c(g) for g in Y_lab_groups])
    Y_hat_lab = np.concatenate([_c(g) for g in Y_hat_lab_groups])
    Y_hat_unlab = np.concatenate([_c(g) for g in Y_hat_unlab_groups])"""
_ns = dict(_ppi.__dict__); _ns["np"] = np
exec(compile(textwrap.dedent(_src.replace(_OLD, _NEW)
                             .replace("def _pooled_k_group_lambda(", "def _centred(")),
             "<c>", "exec"), _ns)
CENTRED = _ns["_centred"]

def sweep(k=3, n=300, n_lab=60, rho=0.80, reps=300):
    print(f"k={k} n={n} n_lab={n_lab} judge rho={rho}   (lambda, mean over {reps} reps)")
    print(f"{'effect':>7s} {'lam shipped':>12s} {'lam centred':>12s} {'diff':>8s}")
    base = None
    for eff in (0.0, 0.15, 0.35, 0.60, 1.00, 2.00):
        rng = np.random.default_rng(5)
        a, b = [], []
        for _ in range(reps):
            truth = [rng.normal(i * eff, 1.0, n) for i in range(k)]
            sig = np.sqrt(1 / rho**2 - 1)
            llm = [t + rng.normal(0, sig, n) for t in truth]
            YL, YHL, YHU = [], [], []
            for t, l in zip(truth, llm):
                idx = rng.choice(n, size=n_lab, replace=False)
                m = np.zeros(n, bool); m[idx] = True
                YL.append(t[m]); YHL.append(l[m]); YHU.append(l[~m])
            a.append(SHIPPED(YL, YHL, YHU)[0])
            b.append(CENTRED(YL, YHL, YHU)[0])
        ma, mb = np.mean(a), np.mean(b)
        if base is None: base = (ma, mb)
        print(f"{eff:>7.2f} {ma:>12.4f} {mb:>12.4f} {mb-ma:>+8.4f}")
    print(f"\n  shipped lambda drift from effect=0 to 2.0: {base[0]:.4f} -> (see last row)")
    print("  centred lambda should be FLAT if the mechanism is between-group spread.")

sweep()
