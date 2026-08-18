"""Is rho^2 a SUFFICIENT statistic for the PPI multiplier, across judge noise shapes?

This is what the paper's rule of thumb actually claims: quote one number about
your judge, read off your savings. That claim is only safe if the multiplier is
a function of rho^2 ALONE -- if two judges with the same rho^2 but differently
shaped errors give different savings, the rule is under-specified.

Rather than pinning rho^2 and comparing shapes (which only probes one point), we
sweep noise SCALE and SHAPE independently and ask whether all the resulting
(rho^2, multiplier) pairs collapse onto a single curve. Collapse = sufficiency.

Two curves are tested, because the correct rho differs by test family (see
notes/WHICH_RHO_FOR_WHICH_TEST.md):

    paired_t  vs  rho_P^2   (Pearson; mean influence function is linear in D)
    wilcoxon  vs  rho_S^2   (Spearman; rank influence function is a function of ranks)

The wilcoxon-vs-rho_S^2 collapse is the one nobody has checked -- rank-based PPI
is uncharted, and it is the curve the paper needs in order to state a rule of
thumb that survives a non-Gaussian judge.

DESIGN NOTE (this bit is load-bearing). The human side -- D and the labelled
subset -- is drawn ONCE and replayed for every (shape, tier) cell, from an rng
stream separate from the judge's. Without that, the shapes silently stop being
comparable: _noise consumes one variate for normal/laplace/t3 but two for
contam/flip, so a single shared stream desynchronises and each shape sees a
different D and a different labelled subset. The first run of this script had
exactly that bug, and it inflated the apparent between-shape spread at tier 0.50
to att_t 0.884-1.001 -- against +3.4% (p=0.13) for the same contrast measured
under a paired design in investigate_rho2_noise_shape_invariance.py. Replaying
the human side also makes the classical arm byte-identical across every row,
which is the single biggest variance reduction available here.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from evalstats.ppi import paired_walsh_midrank_theta as walsh
from evalstats.tests import _ppi_paired_arrays

N_POOL, N_LAB, DELTA = 600, 60, 0.35
SHAPES = ("normal", "laplace", "contam", "flip", "t3")
PEARSON_TARGETS = (0.20, 0.35, 0.50, 0.65, 0.80)


def _noise(kind: str, rng, n: int) -> np.ndarray:
    if kind == "normal":
        return rng.standard_normal(n)
    if kind == "laplace":
        return rng.laplace(0, 1, n) / np.sqrt(2)
    if kind == "t3":
        return rng.standard_t(3, n) / np.sqrt(3)
    if kind == "contam":
        s = rng.standard_normal(n) * np.where(rng.random(n) < .08, 5.0, 1.0)
        return s / np.sqrt(.92 + .08 * 25)
    if kind == "flip":
        s = rng.standard_normal(n)
        return s * np.where(rng.random(n) < .12, -3.0, 1.0) / np.sqrt(.88 + .12 * 9)
    raise ValueError(kind)


def _savings(rho2: float) -> float:
    return 1.0 / (1.0 - rho2 * (1.0 - N_LAB / N_POOL))


def run(reps: int, n_boot: int, seed: int = 61) -> pd.DataFrame:
    # Human side: drawn once, replayed for every cell. See the DESIGN NOTE.
    hr = np.random.default_rng(seed)
    Ds = [DELTA + hr.standard_normal(N_POOL) for _ in range(reps)]
    labs = [hr.choice(N_POOL, N_LAB, replace=False) for _ in range(reps)]
    cl_t = np.array([Ds[i][labs[i]].mean() for i in range(reps)])
    cl_w = np.array([walsh(Ds[i][labs[i]]) for i in range(reps)])
    var_t, var_w = float(np.var(cl_t)), float(np.var(cl_w))
    print(f"classical arm (identical across every row): var_t={var_t:.6g} "
          f"var_w={var_w:.6g}\n", flush=True)

    rows = []
    for shape in SHAPES:
        for tgt in PEARSON_TARGETS:
            kappa = np.sqrt(1.0 / tgt - 1.0)   # pins rho_P^2 = tgt exactly
            jr = np.random.default_rng(909)    # judge stream, same seed each cell
            er = np.random.default_rng(555)    # estimator stream
            t0 = time.time()
            rp, rs, pp_t, pp_w = ([] for _ in range(4))
            for i in range(reps):
                D = Ds[i]
                Dhat = D + kappa * _noise(shape, jr, N_POOL)
                rp.append(stats.pearsonr(D, Dhat)[0])
                rs.append(stats.spearmanr(D, Dhat)[0])
                lab = np.zeros(N_POOL, bool)
                lab[labs[i]] = True
                a, b = Dhat, np.zeros(N_POOL)
                al, bl = np.where(lab, D, np.nan), np.where(lab, 0.0, np.nan)
                pp_t.append(_ppi_paired_arrays(a, b, al, bl, np.mean, .05, n_boot, er,
                                               rectifier_func=np.mean).estimate)
                pp_w.append(_ppi_paired_arrays(a, b, al, bl, walsh, .05, n_boot, er,
                                               rectifier_func=walsh).estimate)
            rP2, rS2 = float(np.mean(rp)) ** 2, float(np.mean(rs)) ** 2
            m_t = float(var_t / np.var(pp_t))
            m_w = float(var_w / np.var(pp_w))
            rows.append(dict(shape=shape, pearson_target=tgt, rP2=rP2, rS2=rS2,
                             bonus=rS2 - rP2, mult_t=m_t, mult_w=m_w,
                             pred_t=_savings(rP2), pred_w=_savings(rS2),
                             att_t=m_t / _savings(rP2), att_w=m_w / _savings(rS2),
                             secs=round(time.time() - t0, 1)))
            print(f"  {shape:8s} rP2={rP2:.3f} rS2={rS2:.3f}  "
                  f"mult_t={m_t:.3f} mult_w={m_w:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=1500)
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--out", default="simulations/out/labeleff_rho2_full/rho2_sufficiency.csv")
    args = ap.parse_args()
    df = run(args.reps, args.n_boot)
    import pathlib
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print("\n" + df.drop(columns=["secs"]).round(4).to_string(index=False))
    print(f"\nwrote {args.out}")
    print("\n=== collapse check: spread of attainment across SHAPES at each tier ===")
    for tgt, g in df.groupby("pearson_target"):
        print(f"  target {tgt:.2f}:  att_t {g.att_t.min():.3f}-{g.att_t.max():.3f}"
              f"   att_w {g.att_w.min():.3f}-{g.att_w.max():.3f}")


if __name__ == "__main__":
    main()
