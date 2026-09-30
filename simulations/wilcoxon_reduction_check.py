"""Does PPI-Wilcoxon reduce to classical Wilcoxon when the judge adds nothing?

Three checks:
  1. STATISTIC identity: theta_W is an exact affine function of the classical
     W+ under scipy's zsplit zero convention, ties included.
  2. VARIANCE identity: with lam=0 the joint sign-flip null variance is the
     classical randomization variance of the signed-rank statistic.
  3. P-VALUE agreement with scipy when the judge is pure noise (lam ~ 0).

Usage: python wilcoxon_reduction_check.py [--reps N] [--tag NAME]
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
from scipy.stats import wilcoxon                                    # noqa: E402
from evalstats import ppi as _ppi                                   # noqa: E402
from evalstats.ppi import (paired_walsh_midrank_theta,              # noqa: E402
                           _analytic_walsh_theta_correct,
                           _walsh_theta_joint_signflip_null_var)

OUT = Path(__file__).parent / "out" / "wilcoxon_guard"


def implied_Wplus(d):
    n = len(d)
    return (paired_walsh_midrank_theta(d) + 0.5) * n * (n + 1) / 2


def check_statistic(rng, T=500):
    out = {}
    for name, gen in (("continuous", lambda: rng.normal(0.4, 1.0, 15)),
                      ("likert_-2..2", lambda: rng.integers(-2, 3, 20).astype(float)),
                      ("heavy_ties", lambda: rng.integers(-1, 2, 25).astype(float))):
        hits = {m: 0 for m in ("wilcox", "pratt", "zsplit")}
        n = 0
        for _ in range(T):
            d = gen()
            if not np.any(d != 0):
                continue
            n += 1
            imp = implied_Wplus(d)
            for m in hits:
                W = wilcoxon(d, zero_method=m, alternative="greater").statistic
                hits[m] += abs(W - imp) < 1e-9
        out[name] = {m: f"{hits[m]}/{n}" for m in hits}
    return out


def check_variance(rng, T=300):
    """lam=0 joint flip variance vs the classical signed-rank null variance
    (2n+1)/(6n(n+1)), valid when there are no ties among |d|."""
    rel = []
    for _ in range(T):
        n = 20
        d = rng.normal(0.3, 1.0, n)                # continuous: no ties
        v = _walsh_theta_joint_signflip_null_var(d, np.zeros(n), 0.0)
        closed = (2 * n + 1) / (6 * n * (n + 1))
        rel.append(v / closed)
    return dict(mean_ratio=float(np.mean(rel)), sd_ratio=float(np.std(rel)))


def check_pvalues(rng, T=1500, n_lab=25, N=200):
    """Judge = pure noise, so lam collapses and the corrected test should
    behave like classical Wilcoxon on the labeled subset."""
    ours, theirs, lams = [], [], []
    for _ in range(T):
        d = rng.normal(0.25, 1.0, N)
        j = rng.normal(0.0, 1.0, N)                # uninformative judge
        idx = rng.permutation(N); lab, unl = idx[:n_lab], idx[n_lab:]
        r = _analytic_walsh_theta_correct(d[lab], j[lab], j[unl], 0.05, True)
        ours.append(r.p_value); lams.append(r.lam if r.lam is not None else np.nan)
        theirs.append(wilcoxon(d[lab], zero_method="zsplit").pvalue)
    o, t = np.array(ours), np.array(theirs)
    return dict(
        mean_lambda=float(np.nanmean(lams)),
        corr=float(np.corrcoef(o, t)[0, 1]),
        median_abs_diff=float(np.median(np.abs(o - t))),
        decision_agreement=float(np.mean((o < .05) == (t < .05))),
        our_reject=float(np.mean(o < .05)), scipy_reject=float(np.mean(t < .05)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="joint_signflip")
    ap.add_argument("--reps", type=int, default=1500)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)

    print(f"[{a.tag}] joint sign-flip active: {_ppi._WALSH_JOINT_SIGNFLIP}\n", flush=True)
    st = check_statistic(rng)
    print("1. STATISTIC == classical W+ (exact matches):", flush=True)
    for k, v in st.items():
        print(f"   {k:14s} " + "  ".join(f"{m}={c}" for m, c in v.items()), flush=True)

    va = check_variance(rng)
    print(f"\n2. VARIANCE at lam=0 vs closed-form (2n+1)/(6n(n+1)):"
          f" ratio {va['mean_ratio']:.4f} +/- {va['sd_ratio']:.4f}", flush=True)

    pv = check_pvalues(rng, T=a.reps)
    print(f"\n3. P-VALUES vs scipy (uninformative judge, mean lambda {pv['mean_lambda']:.3f}):", flush=True)
    print(f"   corr {pv['corr']:.4f} | median |diff| {pv['median_abs_diff']:.4f}"
          f" | decisions agree {pv['decision_agreement']:.1%}", flush=True)
    print(f"   reject rate ours {pv['our_reject']:.3f} vs scipy {pv['scipy_reject']:.3f}", flush=True)

    p = OUT / f"reduction_{a.tag}.json"
    p.write_text(json.dumps(dict(statistic=st, variance=va, pvalues=pv), indent=1))
    print(f"\nwrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
