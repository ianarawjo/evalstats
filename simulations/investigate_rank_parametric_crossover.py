"""Where exactly does PPI power swap between rank-based and parametric tests?

Motivation: the main label-efficiency sweep uses Gaussian judge noise
throughout, and under Gaussian noise rank tests lose twice over -- once to the
classical ARE (3/pi ~= 0.955), once to the rank penalty rho_S^2 < rho_P^2. That
makes "rank tests extract less from PPI" look like a general fact when it is
conditional on an assumption no real LLM judge satisfies. This sweep locates the
crossover so the conditional can be stated precisely.

Design: contamination fraction eps moves rho_S^2 continuously while rho_P^2 is
pinned analytically. For Dhat = D + kappa*E with E unit-variance and independent
of D, rho_P^2 = 1/(1+kappa^2) EXACTLY regardless of E's shape -- Pearson sees
only second moments -- so kappa = sqrt(1/target - 1) holds Pearson fixed at a
tier while eps varies the SHAPE freely. Any movement in the rank/parametric gap
is then attributable to shape alone.

Predicted crossover: PPI-wilcoxon overtakes PPI-t when the rank bonus repays the
classical ARE deficit,

    ARE / (1 - rho_S^2*(1-f))  >  1 / (1 - rho_P^2*(1-f)),   f = n_lab/N

i.e. NOT at rho_S^2 = rho_P^2 -- ranks must win by enough to cover the ~4.5%
they start behind. Solving for the crossing rank bonus at each tier gives the
dashed line the measurements are checked against.

Usage:
    python -m simulations.investigate_rank_parametric_crossover
    python -m simulations.investigate_rank_parametric_crossover --reps 400
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

N_POOL, N_LAB = 600, 60
DELTA = 0.30
OUTLIER_SCALE = 5.0
ARE_NORMAL = 3.0 / np.pi          # Wilcoxon vs t under a normal signal
TIERS = (0.30, 0.50, 0.70)
EPS_GRID = (0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.22)


def _contam(rng, n: int, eps: float) -> np.ndarray:
    """Unit-variance contaminated normal: (1-eps) at scale 1, eps at OUTLIER_SCALE.

    Draws BOTH variates unconditionally, including at eps=0. An early return
    there would consume one rng call instead of two, desynchronising every
    subsequent draw in the replicate -- so the eps=0 row would see a different
    labelled subset than the eps>0 rows and stop being a valid baseline for
    them. That bug was live in the first run of this script and was visible in
    the classical_t/classical_w columns, which must be constant across eps
    (the signal does not depend on the judge) but read 0.6607/0.6353 at eps=0
    against 0.6280/0.6133 everywhere else."""
    z = rng.standard_normal(n)
    u = rng.random(n)
    if eps <= 0:
        return z
    s = z * np.where(u < eps, OUTLIER_SCALE, 1.0)
    return s / np.sqrt((1.0 - eps) + eps * OUTLIER_SCALE ** 2)


def _crossover_bonus(rho_p2: float, f: float, are: float = ARE_NORMAL) -> float:
    """Rank bonus rho_S^2 - rho_P^2 at which PPI-wilcoxon's power overtakes
    PPI-t's, from the inequality in the module docstring. Returns the bonus,
    not the absolute rho_S^2."""
    rho_s2 = (1.0 - are * (1.0 - rho_p2 * (1.0 - f))) / (1.0 - f)
    return rho_s2 - rho_p2


def run(reps: int, n_boot: int, seed: int = 17) -> pd.DataFrame:
    f = N_LAB / N_POOL
    rows = []
    for tier in TIERS:
        kappa = np.sqrt(1.0 / tier - 1.0)      # pins rho_P^2 = tier exactly
        for eps in EPS_GRID:
            rng = np.random.default_rng(seed)
            t0 = time.time()
            rp, rs = [], []
            ct = cw = pt = pw = 0
            w_only = t_only = 0
            for _ in range(reps):
                D = DELTA + rng.standard_normal(N_POOL)
                Dhat = D + kappa * _contam(rng, N_POOL, eps)
                rp.append(stats.pearsonr(D, Dhat)[0])
                rs.append(stats.spearmanr(D, Dhat)[0])
                lab = np.zeros(N_POOL, bool)
                lab[rng.choice(N_POOL, N_LAB, replace=False)] = True
                dl = D[lab]
                ct += stats.ttest_1samp(dl, 0).pvalue < .05
                cw += stats.wilcoxon(dl).pvalue < .05
                a, b = Dhat, np.zeros(N_POOL)
                al, bl = np.where(lab, D, np.nan), np.where(lab, 0.0, np.nan)
                hit_t = _ppi_paired_arrays(a, b, al, bl, np.mean, .05, n_boot, rng,
                                           rectifier_func=np.mean).p_value < .05
                hit_w = _ppi_paired_arrays(a, b, al, bl, walsh, .05, n_boot, rng,
                                           rectifier_func=walsh).p_value < .05
                pt += hit_t
                pw += hit_w
                w_only += (hit_w and not hit_t)
                t_only += (hit_t and not hit_w)
            disc = w_only + t_only
            rP2, rS2 = float(np.mean(rp)) ** 2, float(np.mean(rs)) ** 2
            rows.append(dict(
                tier=tier, eps=eps, rP2=rP2, rS2=rS2, bonus=rS2 - rP2,
                crossover_bonus=_crossover_bonus(tier, f),
                classical_t=ct / reps, classical_w=cw / reps,
                ppi_t=pt / reps, ppi_w=pw / reps, power_gap=(pw - pt) / reps,
                w_only=w_only, t_only=t_only,
                mcnemar_p=float(stats.binomtest(w_only, disc, 0.5).pvalue) if disc else 1.0,
                secs=round(time.time() - t0, 1),
            ))
            print(f"  tier={tier} eps={eps:.2f}  rS2-rP2={rS2-rP2:+.3f}  "
                  f"ppi_w-ppi_t={(pw-pt)/reps:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=1500)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default="simulations/out/labeleff_rho2_full/rank_parametric_crossover.csv")
    args = ap.parse_args()
    df = run(args.reps, args.n_boot)
    import pathlib
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print("\n" + df.drop(columns=["secs"]).round(4).to_string(index=False))
    print(f"\nwrote {args.out}")
    print("\n=== predicted crossover bonus per tier ===")
    f = N_LAB / N_POOL
    for t in TIERS:
        print(f"  rho_P^2={t:.2f}: wilcoxon overtakes at rank bonus "
              f"{_crossover_bonus(t, f):+.4f}")


if __name__ == "__main__":
    main()
