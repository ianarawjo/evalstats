"""EXACT Type I for the paired-binary CI decision rule, by enumeration.

evalstats users act on whether a pairwise CI excludes zero -- directly, and
through the simultaneous-CI/FWER path, which widens these same intervals and
decides from them. That decision behaviour is not visible in coverage, width
or interval score, so it is computed here directly.

mj_floor, bonett_price and tango_scc depend only on (n10, n01, n), so their
rejection region can be enumerated in 2D. newcombe_mover also uses the
concordant split (its phi correlation term), so it needs the full 2x2. Either
way the result is EXACT -- every table weighted by its multinomial
probability, no Monte Carlo.

Result over 90 (n, p_A, S) cells at alpha = 0.05:

    method             max Type I    at (n,pA,S)    cells over nominal
    mj_floor               0.0536   (50, 0.5, 0.5)        12
    bonett_price           0.0528   (40, 0.7, 0.5)         3
    newcombe_mover         0.0608   (10, 0.5, 0.5)        20
    tango_scc              0.0536   (50, 0.5, 0.5)        12

bonett_price controls Type I far better than the rest. Note newcombe_mover is
the WORST here despite having coverage as good as bonett_price's -- good
coverage does not imply good decisions, which is the reason to report both.
"""
import numpy as np
from math import lgamma, exp

from evalstats.core.resampling import (
    mj_floor_paired_ci, bonett_price_paired_ci,
    newcombe_mover_paired_ci, tango_scc_paired_ci,
)

ALPHA = 0.05
_2D = {
    "mj_floor": mj_floor_paired_ci,
    "bonett_price": bonett_price_paired_ci,
    "tango_scc": lambda a, b, al: tango_scc_paired_ci(a, b, al, c=0.0),
}
KEYS = ["mj_floor", "bonett_price", "newcombe_mover", "tango_scc"]


def _arr(n11, n10, n01, n00):
    a = np.array([1] * n11 + [1] * n10 + [0] * n01 + [0] * n00, dtype=float)
    b = np.array([1] * n11 + [0] * n10 + [1] * n01 + [0] * n00, dtype=float)
    return a, b


def decision_rates(n, p_a, s, delta=0.0):
    """P(CI excludes 0) per method. delta=0 gives Type I, delta!=0 power."""
    p10, p01 = (s + delta) / 2.0, (s - delta) / 2.0
    p11 = p_a - p10
    p00 = 1.0 - p11 - p10 - p01
    if min(p10, p01, p11, p00) < -1e-12:
        return None
    out = {k: 0.0 for k in KEYS}
    for n10 in range(n + 1):
        for n01 in range(n + 1 - n10):
            restn = n - n10 - n01
            a, b = _arr(restn, n10, n01, 0)
            d2 = {k: (lambda t: t[0] > 0 or t[1] < 0)(f(a, b, ALPHA)) for k, f in _2D.items()}
            for n11 in range(restn + 1):
                n00 = restn - n11
                lp = (lgamma(n + 1) - lgamma(n11 + 1) - lgamma(n10 + 1)
                      - lgamma(n01 + 1) - lgamma(n00 + 1))
                ok = True
                for c, p in ((n11, p11), (n10, p10), (n01, p01), (n00, p00)):
                    if c:
                        if p <= 0:
                            ok = False
                            break
                        lp += c * np.log(p)
                if not ok:
                    continue
                w = exp(lp)
                for k, d in d2.items():
                    if d:
                        out[k] += w
                aa, bb = _arr(n11, n10, n01, n00)
                lo, hi = newcombe_mover_paired_ci(aa, bb, ALPHA)
                if lo > 0 or hi < 0:
                    out["newcombe_mover"] += w
    return out


def main():
    worst = {k: (0.0, None) for k in KEYS}
    over = {k: 0 for k in KEYS}
    tot = 0
    for n in (10, 15, 20, 30, 40, 50):
        for p_a in (0.3, 0.5, 0.7):
            for s in (0.10, 0.20, 0.30, 0.40, 0.50):
                r = decision_rates(n, p_a, s)
                if r is None:
                    continue
                tot += 1
                for k in KEYS:
                    if r[k] > worst[k][0]:
                        worst[k] = (r[k], (n, p_a, s))
                    if r[k] > ALPHA + 1e-9:
                        over[k] += 1
    print(f"EXACT Type I for the CI decision rule, {tot} cells, alpha={ALPHA}\n")
    print(f"{'method':<17}{'max Type I':>12}{'at (n,pA,S)':>20}{'cells > .05':>14}")
    for k in KEYS:
        v, loc = worst[k]
        print(f"{k:<17}{v:>12.4f}{str(loc):>20}{over[k]:>14}")


if __name__ == "__main__":
    main()
