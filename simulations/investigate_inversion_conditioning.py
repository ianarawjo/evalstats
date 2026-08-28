"""Measure PPI label efficiency at the VARIANCE scale, with no power curve.

The sweep's headline multiplier is obtained by inverting a classical power
curve. Where that curve is flat the inversion is ill-conditioned, and the
resulting distortion is easy to mistake for an estimator defect -- see
`notes/HOW_MULTIPLIERS_ARE_MEASURED.md`. This script is the independent
instrument used to tell those apart.

Two measurements, neither of which touches a power curve:

  `correlations`  In the infinite-unlabeled-pool limit the PPI++ variance
                  reduction is exactly VRF = 1 - corr(theta_lab, theta_pred_lab)^2.
                  Resampling labeled sets and correlating the two sample
                  statistics measures that governing rho^2 directly -- no
                  lambda tuning, no inversion, no saturation. Reported next to
                  the named-correlation shortcuts the sweep actually uses, so a
                  mismatch shows up as a ratio away from 1.00.

  `bound`         Runs the REAL evalstats estimator and compares its achieved
                  variance to the finite-pool bound 1 - rho^2*(1 - n_lab/N).
                  A ratio above 1.00 means the estimator leaves efficiency on
                  the table; below 1.00 is impossible for a control variate and
                  indicates noise, shrinkage, or a bound that does not apply.

Usage:
    python -m simulations.investigate_inversion_conditioning correlations
    python -m simulations.investigate_inversion_conditioning bound --reps 400
    python -m simulations.investigate_inversion_conditioning correlations --quick

Defaults are sized for a few minutes, not for publication precision. `bound`
at 400 reps carries roughly 7% SE on a variance ratio, wide enough that a
single cell 5% off 1.00 means nothing and a monotone trend across tiers means
a little. `correlations` needs MORE replicates than feels necessary, and MWU
needs more than Wilcoxon: MWU resamples two independent arms, so its sample
statistic carries roughly twice the sampling noise, and at 400 reps its
measured rho^2 reads 20-30% low across every tier. Do not read a shortfall off
a low-rep run -- at 4000 reps the same cells sit within 1-9%. Raise --reps for
a number worth quoting.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import warnings
from dataclasses import replace

import numpy as np
from scipy.stats import pearsonr, spearmanr

from evalstats.ppi import paired_walsh_midrank_theta
from evalstats.tests import mannwhitney as ev_mannwhitney, wilcoxon as ev_wilcoxon
from simulations.harness.cases.pvalues import _ppi_power_baseline, _ppi_power_baseline_binary
from simulations.harness.scenarios.synthetic import JudgeBiasSource, generate_judge_bias_cell

# Calibrated judge noise per rho^2 tier, from the sweep's own calibration csv.
NOISE = {
    "continuous": {0.2: 0.2435, 0.3: 0.1856, 0.4: 0.1487, 0.5: 0.1214, 0.6: 0.0990, 0.7: 0.0795},
    "likert":     {0.2: 2.1527, 0.3: 1.6530, 0.4: 1.3274, 0.5: 1.0820, 0.6: 0.8730, 0.7: 0.6874},
}
# The ef=0.35 arm -- the best-conditioned rung of the effect ladder.
ES = {"continuous": 0.04221, "likert": 0.40064}


def _cell(eval_type: str, noise: float, es: float, n: int, seed: int):
    base = _ppi_power_baseline_binary() if eval_type == "binary" else _ppi_power_baseline(eval_type)
    kw = dict(base)
    kw["llm_noise"] = noise
    sc = JudgeBiasSource(name="_probe", tag="_ref", effect_size=es, **kw)
    return generate_judge_bias_cell(replace(sc, n=n), np.random.default_rng(seed))


def _midplace(v: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """P(ref < v) + 0.5*P(ref == v) -- the mid-rank placement of v within ref.

    This is the two-sample influence function (the "placement" of
    Orban-Wolfe): each observation's IF is the OPPOSITE group's CDF evaluated
    at that observation, with the mid-rank convention for ties."""
    n = len(ref_sorted)
    lo = np.searchsorted(ref_sorted, v, side="left")
    hi = np.searchsorted(ref_sorted, v, side="right")
    return (lo + hi) / (2.0 * n)


def _hajek_paired(d: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """g(d) = 1 - F_mid(-d): the signed-rank Hajek projection, mid-rank ties."""
    n = len(ref_sorted)
    lo = np.searchsorted(ref_sorted, -d, side="left")
    hi = np.searchsorted(ref_sorted, -d, side="right")
    return ((n - hi) + 0.5 * (hi - lo)) / n


def correlations_mwu(eval_type, noise, es, n_lab, reps, pool, seed):
    """Governing rho^2 for MWU vs the placement correlation and score Spearman."""
    c = _cell(eval_type, noise, es, pool, seed)
    tA, pA = np.asarray(c.truth_a2, float), np.asarray(c.llm_a2, float)
    tB, pB = np.asarray(c.truth_b2, float), np.asarray(c.llm_b2, float)
    sA, sB, spA, spB = np.sort(tA), np.sort(tB), np.sort(pA), np.sort(pB)

    gA, gB = _midplace(tA, sB), 1.0 - _midplace(tB, sA)
    hA, hB = _midplace(pA, spB), 1.0 - _midplace(pB, spA)
    cA, cB = np.cov(gA, hA)[0, 1], np.cov(gB, hB)[0, 1]
    # m == n, so the 1/m and 1/n weights cancel out of the ratio.
    rho_place2 = (cA + cB) ** 2 / ((gA.var() + gB.var()) * (hA.var() + hB.var()))

    rng = np.random.default_rng(seed + 1)
    iA = rng.integers(0, pool, size=(reps, n_lab))
    iB = rng.integers(0, pool, size=(reps, n_lab))
    th_t = np.array([float(np.mean(_midplace(tA[iA[r]], np.sort(tB[iB[r]])))) for r in range(reps)])
    th_p = np.array([float(np.mean(_midplace(pA[iA[r]], np.sort(pB[iB[r]])))) for r in range(reps)])
    return (pearsonr(th_t, th_p).statistic ** 2, rho_place2,
            spearmanr(tA, pA).statistic ** 2)


def correlations_wilcoxon(eval_type, noise, es, n_lab, reps, pool, seed):
    """Governing rho^2 for Wilcoxon vs its empirical IF correlation and Spearman(D)."""
    c = _cell(eval_type, noise, es, pool, seed)
    D = np.asarray(c.truth_x, float) - np.asarray(c.truth_y, float)
    Dh = np.asarray(c.llm_x, float) - np.asarray(c.llm_y, float)
    g, gh = _hajek_paired(D, np.sort(D)), _hajek_paired(Dh, np.sort(Dh))
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, pool, size=(reps, n_lab))
    th_t = np.array([paired_walsh_midrank_theta(D[i]) for i in idx])
    th_p = np.array([paired_walsh_midrank_theta(Dh[i]) for i in idx])
    return (pearsonr(th_t, th_p).statistic ** 2,
            pearsonr(g, gh).statistic ** 2,
            spearmanr(D, Dh).statistic ** 2)


def bound_check(method, eval_type, tier, n, n_lab, reps, pool, seed):
    """Actual estimator variance vs the finite-pool control-variate bound."""
    noise, es = NOISE[eval_type][tier], ES[eval_type]
    c = _cell(eval_type, noise, es, pool, seed)
    if method == "wilcoxon":
        tx, ty = np.asarray(c.truth_x, float), np.asarray(c.truth_y, float)
        px, py = np.asarray(c.llm_x, float), np.asarray(c.llm_y, float)
        D, Dh = tx - ty, px - py
        r0 = np.random.default_rng(seed)
        idx = r0.integers(0, pool, size=(min(reps * 6, 2500), n_lab))
        rho2 = pearsonr(np.array([paired_walsh_midrank_theta(D[i]) for i in idx]),
                        np.array([paired_walsh_midrank_theta(Dh[i]) for i in idx])).statistic ** 2
    else:
        tA, pA = np.asarray(c.truth_a2, float), np.asarray(c.llm_a2, float)
        tB, pB = np.asarray(c.truth_b2, float), np.asarray(c.llm_b2, float)
        rho2, _, _ = correlations_mwu(eval_type, noise, es, n_lab, min(reps * 6, 2500), pool, seed)

    rng = np.random.default_rng(seed + 1)
    ppi, human = [], []
    with contextlib.redirect_stdout(io.StringIO()):
        for r in range(reps):
            if method == "wilcoxon":
                i = rng.integers(0, pool, n)
                XT, YT, XP, YP = tx[i], ty[i], px[i], py[i]
                lab = rng.permutation(n)[:n_lab]
                xl, yl = np.full(n, np.nan), np.full(n, np.nan)
                xl[lab], yl[lab] = XT[lab], YT[lab]
                try:
                    ppi.append(ev_wilcoxon(XP, YP, xl, yl, n_boot=20,
                                           rng=np.random.default_rng(r), print_result=False).corrected_estimate)
                except Exception:
                    ppi.append(np.nan)
                human.append(paired_walsh_midrank_theta((XT - YT)[lab]))
            else:
                ia, ib = rng.integers(0, pool, n), rng.integers(0, pool, n)
                xt, xp, yt, yp = tA[ia], pA[ia], tB[ib], pB[ib]
                la, lb = rng.permutation(n)[:n_lab], rng.permutation(n)[:n_lab]
                xl, yl = np.full(n, np.nan), np.full(n, np.nan)
                xl[la], yl[lb] = xt[la], yt[lb]
                try:
                    ppi.append(ev_mannwhitney(xp, yp, xl, yl, n_boot=20,
                                              rng=np.random.default_rng(r), print_result=False).corrected_estimate)
                except Exception:
                    ppi.append(np.nan)
                human.append(float(np.mean(_midplace(xt[la], np.sort(yt[lb])))))
    ppi, human = np.array(ppi, float), np.array(human, float)
    ok = ~np.isnan(ppi)
    vrf = float(np.nanvar(ppi) / np.var(human[ok]))
    exact = 1.0 - rho2 * (1.0 - n_lab / n)
    return rho2, vrf, exact


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["correlations", "bound"])
    ap.add_argument("--reps", type=int, default=0, help="MC replicates (default: 3000 / 400 by mode)")
    ap.add_argument("--pool", type=int, default=120_000, help="population size defining the CDFs")
    ap.add_argument("--n-lab", type=int, default=200)
    ap.add_argument("--n", type=int, default=1000, help="bound mode: full pool per arm")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--quick", action="store_true", help="3 tiers and fewer reps")
    args = ap.parse_args()

    tiers = (0.2, 0.4, 0.7) if args.quick else (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    if args.mode == "correlations":
        reps = args.reps or (2000 if args.quick else 4000)
        print(f"{'test':9s} {'type':11s} {'tier':>5s} | {'measured':>9s} {'IF/placement':>12s} "
              f"{'named shortcut':>14s} | {'IF/meas':>8s} {'named/meas':>10s}")
        print("-" * 92)
        for name, fn in (("mwu", correlations_mwu), ("wilcoxon", correlations_wilcoxon)):
            for et in ("continuous", "likert"):
                for t in tiers:
                    m, i, s = fn(et, NOISE[et][t], ES[et], args.n_lab, reps, args.pool, args.seed)
                    print(f"{name:9s} {et:11s} {t:5.1f} | {m:9.4f} {i:12.4f} {s:14.4f} | "
                          f"{i/m:8.3f} {s/m:10.3f}", flush=True)
            print()
    else:
        reps = args.reps or (150 if args.quick else 400)
        print(f"{'test':9s} {'type':11s} {'tier':>5s} | {'rho2':>7s} {'VRF actual':>10s} "
              f"{'exact bound':>11s} | {'act/bound':>9s}")
        print("-" * 78)
        for name in ("mwu", "wilcoxon"):
            for et in ("continuous", "likert"):
                for t in tiers:
                    r2, v, ex = bound_check(name, et, t, args.n, args.n_lab, reps, args.pool, args.seed)
                    print(f"{name:9s} {et:11s} {t:5.1f} | {r2:7.4f} {v:10.4f} {ex:11.4f} | "
                          f"{v/ex:9.3f}", flush=True)
            print()


if __name__ == "__main__":
    main()
