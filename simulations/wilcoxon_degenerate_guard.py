"""Type I / power probe for the degenerate-labeled-set failure in PPI-Wilcoxon.

The failure: when the labeled human differences are all tied at zero AND the
labeled judge differences are unanimous, both Var(psi^H) and V^J_lab collapse
to zero, the standard error is carried entirely by the unlabeled judge term,
and a numerically tiny correction becomes significant. Classical Wilcoxon
returns p=1 on an all-zero labeled set and cannot reject at all.

Usage:  python wilcoxon_degenerate_guard.py [--reps N] [--tag NAME]
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
from evalstats.ppi import _analytic_walsh_theta_correct  # noqa: E402

OUT = Path(__file__).parent / "out" / "wilcoxon_guard"
ZERO_FRACS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def one_cell(zf, n_lab, bias, effect, N, T, seed):
    """effect=0 -> Type I; effect>0 -> power (shifts the non-tied items)."""
    rng = np.random.default_rng(seed)
    rej = 0; az_n = 0; az_rej = 0; near_n = 0; near_rej = 0
    for _ in range(T):
        tied = rng.random(N) < zf
        sign = rng.choice([-1.0, 1.0], N)
        if effect > 0:                      # tilt the non-tied items positive
            sign = np.where(rng.random(N) < 0.5 + effect, 1.0, -1.0)
        d = np.where(tied, 0.0, sign)
        j = d + bias + rng.normal(0, 0.3, N)
        idx = rng.permutation(N); lab, unl = idx[:n_lab], idx[n_lab:]
        r = _analytic_walsh_theta_correct(d[lab], j[lab], j[unl], 0.05, True)
        hit = bool(r.p_value < 0.05)
        rej += hit
        nz = int(np.count_nonzero(d[lab]))
        if nz == 0: az_n += 1; az_rej += hit
        if nz <= 1: near_n += 1; near_rej += hit
    return dict(zero_frac=zf, n_lab=n_lab, bias=bias, effect=effect, reps=T,
                reject=rej / T,
                all_zero_n=az_n, all_zero_reject=(az_rej / az_n if az_n else None),
                near_n=near_n, near_reject=(near_rej / near_n if near_n else None))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6000)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--n-lab", type=int, nargs="+", default=[15])
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"[{a.tag}] TYPE I  (truth null, judge biased +1.0)", flush=True)
    print(f"{'zf':>5s} {'nlab':>4s} | {'uncond':>7s} | {'P(allzero)':>10s} {'rej|allzero':>11s}"
          f" | {'rej|<=1nz':>9s}", flush=True)
    for n_lab in a.n_lab:
        for zf in ZERO_FRACS:
            r = one_cell(zf, n_lab, 1.0, 0.0, 200, a.reps, a.seed)
            rows.append(r)
            az = r["all_zero_reject"]; nr = r["near_reject"]
            print(f"{zf:5.2f} {n_lab:4d} | {r['reject']:7.3f} | {r['all_zero_n']/a.reps:10.1%}"
                  f" {(f'{az:.3f}' if az is not None else '   -   '):>11s}"
                  f" | {(f'{nr:.3f}' if nr is not None else '  -  '):>9s}", flush=True)

    print(f"\n[{a.tag}] POWER  (true positive tilt 0.15 on non-tied items)", flush=True)
    for n_lab in a.n_lab:
        for zf in (0.60, 0.75, 0.85):
            r = one_cell(zf, n_lab, 1.0, 0.15, 200, max(a.reps // 3, 800), a.seed + 7)
            rows.append(r)
            print(f"{zf:5.2f} {n_lab:4d} | power {r['reject']:.3f}", flush=True)

    p = OUT / f"{a.tag}.json"
    p.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
