"""Type I of PPI-Wilcoxon across DGP shapes, to check the degenerate-set fix
did not disturb ordinary regimes. Compares against classical Wilcoxon run on
all N (oracle) and on the labeled subset only."""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
from scipy.stats import wilcoxon                                   # noqa: E402
from evalstats.ppi import _analytic_walsh_theta_correct            # noqa: E402

OUT = Path(__file__).parent / "out" / "wilcoxon_guard"
PM = 1.22395   # pseudomedian of exp(Z): recentre to put the weak null at 0


def draw(kind, n, rng):
    if kind == "symmetric":   return rng.normal(0, 1, n)
    if kind == "skewed_pm0":  return np.exp(rng.normal(0, 1, n)) - PM
    if kind == "likert_ties": return rng.choice([-2, -1, 0, 0, 0, 1, 2], n).astype(float)
    if kind == "heavy_ties":  return rng.choice([-1, 0, 0, 0, 0, 1], n).astype(float)
    raise ValueError(kind)


def run(kind, n_lab, N=200, T=1200, seed=0):
    rng = np.random.default_rng(seed)
    a = b = c = 0
    for _ in range(T):
        d = draw(kind, N, rng)
        j = d + rng.normal(0, 0.5, N)
        idx = rng.permutation(N); lab, unl = idx[:n_lab], idx[n_lab:]
        a += _analytic_walsh_theta_correct(d[lab], j[lab], j[unl], 0.05, True).p_value < 0.05
        if np.any(d != 0):
            b += wilcoxon(d, zero_method="zsplit").pvalue < 0.05
        if np.any(d[lab] != 0):
            c += wilcoxon(d[lab], zero_method="zsplit").pvalue < 0.05
    return a / T, b / T, c / T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="joint_plus_floor")
    ap.add_argument("--reps", type=int, default=1200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[{a.tag}] Type I by DGP shape\n")
    print(f"{'DGP':14s} {'nlab':>4s} | {'PPI-W':>7s} {'classical(N)':>12s} {'classical(lab)':>14s}", flush=True)
    for kind in ("symmetric", "skewed_pm0", "likert_ties", "heavy_ties"):
        for n_lab in (15, 30):
            x, y, z = run(kind, n_lab, T=a.reps)
            rows.append(dict(dgp=kind, n_lab=n_lab, ppi_w=x, classical_all=y, classical_lab=z))
            print(f"{kind:14s} {n_lab:4d} | {x:7.3f} {y:12.3f} {z:14.3f}", flush=True)
    (OUT / f"dgp_variety_{a.tag}.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT / f'dgp_variety_{a.tag}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
