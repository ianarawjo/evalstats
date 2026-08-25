"""Is the effective-runs correction in mj_floor_er doing anything?

`mj_floor_paired_ci_multirun_effective` computes a Kish effective run count
R_eff = R/(1+(R-1)rho) and splits the variance into between/within terms:

    between = max(var_delta - within_bar/R_eff, 0) / n
    within  = within_bar / (n * R_eff)

When the max() does not clamp, those sum to EXACTLY var_delta/n -- the R_eff
terms cancel and the correction is a no-op. The base var_delta is already
Var(delta_i, ddof=1), the centred cluster-level variance, so between-run
correlation is fully absorbed by treating items as the independent unit;
there is nothing left for R_eff to correct.

When it does clamp, the variance becomes var_delta*(1-rho)(1+(R-1)rho)/n, an
INFLATION. Writing f(rho) = (1-rho)(1+(R-1)rho), the clamp fires exactly when
f(rho) > 1, i.e. 0 < rho < (R-2)/(R-1), and f peaks at 1 + (R-2)^2/(4(R-1)):

    R= 3  peak 1.125x   R= 5  peak 1.562x   R=10  peak 2.778x

So the correction is either inert or a conservative variance inflation -- not
the run-correlation adjustment the name and docstring describe. Verified here
against the shipped implementation by reconstructing its radicand.

Found by literature review (Eliasziw & Donner 1991 use exactly this design
effect for the ICC-adjusted McNemar TEST, where it is correct because the base
variance there is not already cluster-level).
"""
import numpy as np
from scipy.stats import norm

from evalstats.core.resampling import (
    mj_floor_paired_ci_multirun_effective as ER,
    _mj_discordance_floor,
)


def predicted_var_term(a, b):
    ab = (a >= 0.5).astype(int); bb = (b >= 0.5).astype(int)
    d10 = np.mean((ab == 1) & (bb == 0), axis=1)
    d01 = np.mean((ab == 0) & (bb == 1), axis=1)
    delta = d10 - d01; u = d10 + d01
    n, R = a.shape
    vd = float(np.var(delta, ddof=1))
    wb = float(np.mean(np.maximum(u - delta * delta, 0.0)))
    rho = max(0.0, min(1.0, 1.0 - (wb / (vd * R + 1e-12)))) if vd > 0 else 0.0
    r_eff = R / (1.0 + (R - 1.0) * rho)
    clamped = (vd - wb / r_eff) < 0
    return ((wb / r_eff) / n if clamped else vd / n), clamped


def main(trials=400, seed=4):
    rng = np.random.default_rng(seed)
    agree = clamp_hits = 0
    for _ in range(trials):
        n = int(rng.integers(15, 60)); R = int(rng.choice([3, 5, 10]))
        p = rng.uniform(0.2, 0.8); rt = rng.uniform(0, 0.9)
        lat = rng.uniform(0, 1, size=(n, 2))
        A = (rng.uniform(size=(n, R)) < (rt * (lat[:, 0:1] > 0.5) + (1 - rt) * p)).astype(float)
        B = (rng.uniform(size=(n, R)) < (rt * (lat[:, 1:2] > 0.5) + (1 - rt) * p)).astype(float)
        pv, clamped = predicted_var_term(A, B)
        clamp_hits += clamped
        lo, hi = ER(A, B, 0.05)
        z = norm.ppf(0.975); z2 = z * z; denom = 1 + z2 / n
        rad = ((hi - lo) / 2 * denom / z) ** 2
        ab = (A >= 0.5).astype(int); bb = (B >= 0.5).astype(int)
        u = np.mean((ab == 1) & (bb == 0), axis=1) + np.mean((ab == 0) & (bb == 1), axis=1)
        floorterm = z2 * _mj_discordance_floor(float(np.mean(u))) / (n * n)
        agree += abs((rad - floorterm) - pv) < 1e-9
    print(f"closed form matches shipped code: {agree}/{trials}")
    print(f"clamp fired in:                  {clamp_hits}/{trials}")
    for R in (3, 5, 10):
        print(f"  R={R:>2}: peak inflation {1 + (R-2)**2/(4*(R-1)):.3f}x, "
              f"clamp active for rho < {(R-2)/(R-1):.3f}")


if __name__ == "__main__":
    main()
