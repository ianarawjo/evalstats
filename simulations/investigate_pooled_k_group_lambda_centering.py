"""Should _pooled_k_group_lambda centre per group before pooling?

Its two-group sibling _pooled_two_group_lambda was fixed that way in 836f811:
concatenating groups UNCENTERED puts the between-group spread into the pooled
variance, inflating `denom` in lam_raw = cov/denom and so depressing lambda.
Lambda then drifts with the effect size, costing efficiency.

The k-group version was left alone with an explicit NOTE saying it had only
been checked at the estimator-variance level, never for Type-I/coverage.

The prediction that makes this testable: under the NULL every group has the
same mean, so centring is nearly a no-op and Type-I should be UNCHANGED --
which is exactly why the existing null-only validation passed and did not
catch it. The defect should bite under a REAL EFFECT, where between-group
spread is large, so power should IMPROVE. A fix that instead moves Type-I is
a fix that must not ship.

Measures both, on the same cells, patched-vs-shipped, without editing ppi.py.
"""
import sys, warnings, inspect, textwrap
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")

import evalstats.ppi as _ppi
import evalstats.tests as _t
from scipy import stats as _st

ALPHA = 0.05
SHIPPED = _ppi._pooled_k_group_lambda

_src = inspect.getsource(SHIPPED)
_OLD = """    Y_lab = np.concatenate(Y_lab_groups)
    Y_hat_lab = np.concatenate(Y_hat_lab_groups)
    Y_hat_unlab = np.concatenate(Y_hat_unlab_groups)"""
assert _src.count(_OLD) == 1, "anchor not found -- ppi.py changed"
_NEW = """    def _c(x):
        x = np.asarray(x, dtype=float)
        return x - x.mean() if x.size else x
    Y_lab = np.concatenate([_c(g) for g in Y_lab_groups])
    Y_hat_lab = np.concatenate([_c(g) for g in Y_hat_lab_groups])
    Y_hat_unlab = np.concatenate([_c(g) for g in Y_hat_unlab_groups])"""
_ns = dict(_ppi.__dict__); _ns["np"] = np
exec(compile(textwrap.dedent(_src.replace(_OLD, _NEW)
                             .replace("def _pooled_k_group_lambda(", "def _centred(")),
             "<centred>", "exec"), _ns)
CENTRED = _ns["_centred"]


def cell(rng, k, n, n_lab, effect, rho=0.80):
    """k independent groups; judge = truth + noise calibrated to corr rho."""
    truth = [rng.normal(i * effect, 1.0, n) for i in range(k)]
    sig = np.sqrt(1.0 / rho**2 - 1.0)
    llm = [t + rng.normal(0, sig, n) for t in truth]
    labs = []
    for t in truth:
        idx = rng.choice(n, size=n_lab, replace=False)
        a = np.full(n, np.nan); a[idx] = t[idx]
        labs.append(a)
    return llm, labs


def run(fn, k, n, n_lab, effect, reps, seed):
    _ppi._pooled_k_group_lambda = fn
    import importlib
    rng = np.random.default_rng(seed)
    rej, lams = [], []
    for _ in range(reps):
        llm, labs = cell(rng, k, n, n_lab, effect)
        out = _t._ppi_anova_independent_f_stat(llm, labs, k, power_tune=True)
        if out is None: continue
        p = float(_st.f.sf(out["f_corr"], out["dfn"], out["dfd"]))
        if np.isfinite(p): rej.append(p < ALPHA)
        if "lam" in out and np.isfinite(out.get("lam", np.nan)): lams.append(out["lam"])
    _ppi._pooled_k_group_lambda = SHIPPED
    return (float(np.mean(rej)) if rej else np.nan,
            float(np.mean(lams)) if lams else np.nan)


if __name__ == "__main__":
    REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    se = np.sqrt(ALPHA * (1 - ALPHA) / REPS)
    print(f"_pooled_k_group_lambda: shipped (uncentred) vs centred-before-pooling")
    print(f"reps={REPS}  alpha={ALPHA}  MC SE(null)={se:.4f}\n")
    for tag, effect in (("TYPE-I (null, effect=0)", 0.0),
                        ("POWER (effect=0.06)", 0.06),
                        ("POWER (effect=0.10)", 0.10),
                        ("POWER (effect=0.15)", 0.15)):
        print(f"════ {tag} ════")
        print(f"{'k':>2s} {'n':>5s} {'n_lab':>6s} {'shipped':>9s} {'centred':>9s} {'delta':>8s}"
              f" {'lam ship':>9s} {'lam cent':>9s}")
        for k, n, n_lab in ((3, 200, 40), (3, 400, 60), (5, 200, 40), (4, 300, 50)):
            a, la = run(SHIPPED, k, n, n_lab, effect, REPS, 909)
            b, lb = run(CENTRED, k, n, n_lab, effect, REPS, 909)
            flag = ""
            if effect == 0.0 and abs(b - a) > 3 * se: flag = "  <-- TYPE-I MOVED"
            print(f"{k:>2d} {n:>5d} {n_lab:>6d} {a:>9.4f} {b:>9.4f} {b-a:>+8.4f}"
                  f" {la:>9.4f} {lb:>9.4f}{flag}")
        print()
