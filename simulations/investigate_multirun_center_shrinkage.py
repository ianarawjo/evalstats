"""Why no multi-run mj_floor variant fixes the coverage tail.

All three multi-run variants (effective / moments / cluster) place the
interval centre at

    center = d_hat / (1 + z^2 / n_items)

The denominator uses the ITEM count only -- never the run count R, and never
R_eff. So the score shrinkage toward zero is invariant to how many runs are
collected: k = n/(n+z^2) is 0.722 at n=10 and 0.929 at n=50, a 28% and 7%
pull toward the null respectively. On lopsided scenarios (|delta| large) that
bias is what breaks coverage, and buying more runs cannot touch it.

Bonett-Price shrinks by n/(n+2) instead -- 0.833 at n=10 -- roughly half as
hard, which is why it survives the same cells.

Consequence, reproduced below: on a lopsided cell with R=5 and high ICC,
Bonett-Price applied to a SINGLE run beats every multi-run variant using all
five. The multi-run family is worse than discarding 80% of the data.

Note mj_floor_paired_ci_multirun_cluster is exactly the plain "var_delta/n,
no R_eff" variant, and at high ICC it returns intervals identical to
_multirun_effective -- see investigate_mj_floor_er_reff_algebra.py.
"""
import numpy as np

from evalstats.core.resampling import (
    mj_floor_paired_ci_multirun_effective as ER,
    mj_floor_paired_ci_multirun_cluster as CL,
    bonett_price_paired_ci as BP,
    newcombe_mover_paired_ci as NM,
)

Z2 = 1.959963985 ** 2


def draw(rng, n, R, p10, p01, p11, flip=0.03):
    p00 = 1 - p10 - p01 - p11
    cell = rng.choice(4, size=n, p=[p11, p10, p01, p00])
    A = np.repeat(np.isin(cell, [0, 1]).astype(float)[:, None], R, axis=1)
    B = np.repeat(np.isin(cell, [0, 2]).astype(float)[:, None], R, axis=1)
    return (np.abs(A - (rng.uniform(size=(n, R)) < flip)),
            np.abs(B - (rng.uniform(size=(n, R)) < flip)))


def main(reps=3000, seed=11):
    print("centre shrinkage k (d_hat -> k * d_hat):")
    for n in (10, 20, 50, 100):
        print(f"  n={n:>4}  mj_floor family n/(n+z^2)={n/(n+Z2):.3f}   "
              f"bonett_price n/(n+2)={n/(n+2):.3f}")
    p10, p01, p11 = 0.30, 0.005, 0.35
    true = p10 - p01
    print(f"\nlopsided cell p10={p10}, p01={p01}, R=5, {reps} reps -- coverage")
    print(f"{'n':>5}{'mj_floor_er':>13}{'cluster':>10}{'bonett(1 run)':>15}{'newcombe(1 run)':>17}")
    rng = np.random.default_rng(seed)
    for n in (10, 20, 50):
        hit = dict(er=0, cl=0, bp=0, nm=0)
        for _ in range(reps):
            A, B = draw(rng, n, 5, p10, p01, p11)
            for k, fn in (("er", lambda: ER(A, B, 0.05)), ("cl", lambda: CL(A, B, 0.05)),
                          ("bp", lambda: BP(A[:, 0], B[:, 0], 0.05)),
                          ("nm", lambda: NM(A[:, 0], B[:, 0], 0.05))):
                lo, hi = fn()
                hit[k] += lo <= true <= hi
        print(f"{n:>5}{hit['er']/reps:>13.3f}{hit['cl']/reps:>10.3f}"
              f"{hit['bp']/reps:>15.3f}{hit['nm']/reps:>17.3f}")


if __name__ == "__main__":
    main()
