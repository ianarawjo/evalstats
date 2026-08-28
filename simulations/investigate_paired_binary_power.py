"""Does Bonett-Price's conservative coverage cost real inferential power?

The ci_paired tables show bonett_price over-covering, especially at small n
(.996 at n=10 synthetic). Over-coverage means wider intervals, which means
failing to detect differences that are really there. This measures that
directly: for a known true difference, how often does each method's CI
exclude zero?

Reports, per (n, delta):
  - power  = P(CI excludes 0) when delta != 0
  - type1  = P(CI excludes 0) when delta == 0  (nominal 0.05)
  - width  = mean interval width

Diagnostic settings; not for final numbers.
"""
import numpy as np

from evalstats.core.resampling import (
    mj_floor_paired_ci, bonett_price_paired_ci, newcombe_mover_paired_ci,
    tango_scc_paired_ci, bayes_paired_diff_ci,
)

METHODS = {
    "mj_floor":       lambda a, b, al, rng: mj_floor_paired_ci(a, b, al),
    "bonett_price":   lambda a, b, al, rng: bonett_price_paired_ci(a, b, al),
    "newcombe_mover": lambda a, b, al, rng: newcombe_mover_paired_ci(a, b, al),
    "tango_scc":      lambda a, b, al, rng: tango_scc_paired_ci(a, b, al, c=0.0),
    # bayes_paired_diff_ci returns (lo, hi, p_direction) -- take the interval only
    "bayes_paired":   lambda a, b, al, rng: bayes_paired_diff_ci(a, b, al, num_samples=2000, rng=rng)[:2],
}


def probs(p_a, s, delta):
    p10, p01 = (s + delta) / 2.0, (s - delta) / 2.0
    p11 = p_a - p10
    p00 = 1.0 - p11 - p10 - p01
    v = np.array([p11, p10, p01, p00])
    return None if np.any(v < 1e-9) else v


def main(reps=1200, alpha=0.05, p_a=0.5, seed=17):
    rng = np.random.default_rng(seed)
    for s in (0.15, 0.35):
        print(f"\n{'='*74}\ndiscordance S={s:.2f}, p_A={p_a}, {reps} reps, nominal alpha={alpha}\n{'='*74}")
        for delta in (0.0, 0.05, 0.10, 0.15):
            pr = probs(p_a, s, delta)
            if pr is None:
                continue
            kind = "TYPE I" if delta == 0 else "POWER "
            print(f"\n{kind}  delta={delta:.2f}")
            print(f"  {'method':<16}" + "".join(f"{'n='+str(n):>9}" for n in (15, 30, 50, 100))
                  + "   " + "".join(f"{'w@'+str(n):>8}" for n in (15, 50)))
            res = {m: {} for m in METHODS}
            wid = {m: {} for m in METHODS}
            for n in (15, 30, 50, 100):
                hit = {m: 0 for m in METHODS}
                w = {m: 0.0 for m in METHODS}
                for _ in range(reps):
                    cell = rng.choice(4, size=n, p=pr)
                    a = np.isin(cell, [0, 1]).astype(float)
                    b = np.isin(cell, [0, 2]).astype(float)
                    for mname, fn in METHODS.items():
                        lo, hi = fn(a, b, alpha, rng)
                        w[mname] += hi - lo
                        if lo > 0.0 or hi < 0.0:
                            hit[mname] += 1
                for m in METHODS:
                    res[m][n] = hit[m] / reps
                    wid[m][n] = w[m] / reps
            for m in METHODS:
                print(f"  {m:<16}" + "".join(f"{res[m][n]:>9.3f}" for n in (15, 30, 50, 100))
                      + "   " + "".join(f"{wid[m][n]:>8.3f}" for n in (15, 50)))


if __name__ == "__main__":
    main()
