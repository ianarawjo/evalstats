"""Is rank-based PPI's shortfall an artifact of our Gaussian data-generating
process -- and would a different DGP reverse it?

Motivation: Wilcoxon is known to be ~5% less efficient than the paired t-test
under normality (ARE = 3/pi ~= 0.955; McKean 2003) and MORE efficient under
heavy tails (ARE = 1.5 under Laplace). Our sweeps report rank-based PPI falling
short of its control-variate bound, and our DGP is Gaussian throughout -- so the
obvious worry is that we are just re-measuring the textbook ARE.

We are not. See notes/RANK_PPI_TAIL_SENSITIVITY.md for the full write-up.

Three experiments:

  exp1  Heavy tail in the SIGNAL (shared by human and judge). Confirms the
        classical ARE reproduces (validating the setup), and shows that the
        within-method measured/predicted ratio does NOT distinguish the two
        tests -- the ARE cancels out of it.

  exp2  Heavy tail in the JUDGE's errors only, with the signal held Gaussian
        and Pearson rho^2 pinned at ~0.50 by construction. This is where the
        rank penalty flips to a rank bonus.

  exp3  Common-currency power check for exp2's reversal, with a paired
        (McNemar) test so the swap is not read off Monte Carlo noise.

  exp4  Realistic LLM-judge error modes on a 1-5 Likert scale, applied at the
        ITEM level then differenced, so rounding/clipping/refusal act where
        they really do. Reports what each mode does to the judge's MEAN
        separately from what it does to the correlation -- the two are
        near-independent, which is the whole reason rho^2 is the right axis.

  exp5  Differential-bias sweep. Falls out of exp4: a constant offset on ONE
        arm leaves Pearson AND Spearman exactly unchanged (both are
        location-invariant) and leaves paired_t's attainment exactly
        unchanged, but degrades wilcoxon's badly, because the Walsh
        estimand is NOT location-invariant. The predicted bound cannot see
        this at all.

Usage:
    python -m simulations.investigate_rank_ppi_tail_sensitivity
    python -m simulations.investigate_rank_ppi_tail_sensitivity --exp 2 --reps 400
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
MAD_C = 1.4826


def _noise(kind: str, rng, size: int) -> np.ndarray:
    """Mean-zero, unit-variance draws from each family.

    Every family is standardized to unit variance so that a given judge-noise
    multiplier means the same signal-to-noise ratio across families -- only the
    SHAPE of the tail differs. Without this the comparison would confound tail
    shape with noise magnitude.
    """
    if kind == "normal":
        return rng.standard_normal(size)
    if kind == "laplace":
        return rng.laplace(0.0, 1.0, size) / np.sqrt(2.0)
    if kind == "t3":
        return rng.standard_t(3, size) / np.sqrt(3.0)
    if kind == "contam":  # 8% wild judge calls at 5x scale
        s = rng.standard_normal(size) * np.where(rng.random(size) < 0.08, 5.0, 1.0)
        return s / np.sqrt(0.92 + 0.08 * 25.0)
    if kind == "flip":  # judge occasionally reverses sign entirely
        s = rng.standard_normal(size)
        return s * np.where(rng.random(size) < 0.12, -3.0, 1.0)
    if kind == "lognorm":
        s = rng.lognormal(0.0, 0.9, size)
        return (s - s.mean()) / s.std()
    raise ValueError(kind)


def _savings(rho2: float, n_lab: int = N_LAB, n: int = N_POOL) -> float:
    """PPI's control-variate bound -- the UNLABELED-fraction form. See
    `_ppi_predicted_savings` in simulations/harness/cases/pvalues.py."""
    return 1.0 / (1.0 - rho2 * (1.0 - n_lab / n))


def _robust_disp(x) -> float:
    """MAD^2, a stand-in for variance that stays finite when the estimator's
    4th moment does not (t3 and lognormal). A plain Var() ratio over finite
    reps is itself unstable there, so exp1 reports both and they are only
    trusted where they agree."""
    x = np.asarray(x)
    return float((MAD_C * np.median(np.abs(x - np.median(x)))) ** 2)


def _ppi_arms(D: np.ndarray, Dhat: np.ndarray, lab: np.ndarray, rng, n_boot: int):
    """Run both PPI arms on one replicate. `b`/`bl` are zeros so that
    `a - b == Dhat` and `a_lab - b_lab == D`: _ppi_paired_arrays only ever
    looks at the differences, so this drives the real estimator on our
    difference-level DGP without inventing a two-column scenario."""
    a, b = Dhat, np.zeros_like(Dhat)
    al = np.where(lab, D, np.nan)
    bl = np.where(lab, 0.0, np.nan)
    r_t = _ppi_paired_arrays(a, b, al, bl, np.mean, 0.05, n_boot, rng, rectifier_func=np.mean)
    r_w = _ppi_paired_arrays(a, b, al, bl, walsh, 0.05, n_boot, rng, rectifier_func=walsh)
    return r_t, r_w


def exp1(reps: int, delta: float = 0.35, seed: int = 7) -> pd.DataFrame:
    """Heavy tail in the SIGNAL, shared by human and judge."""
    rows = []
    for kind in ("normal", "laplace", "t3", "contam", "lognorm"):
        for kappa in (0.5, 1.0):
            rng = np.random.default_rng(seed)
            cl_t, cl_w, pp_t, pp_w, pt_p, pw_p, rp, rs = ([] for _ in range(8))
            for _ in range(reps):
                D = delta + _noise(kind, rng, N_POOL)
                Dhat = D + kappa * _noise(kind, rng, N_POOL)
                lab = np.zeros(N_POOL, bool)
                lab[rng.choice(N_POOL, N_LAB, replace=False)] = True
                rp.append(stats.pearsonr(D, Dhat)[0])
                rs.append(stats.spearmanr(D, Dhat)[0])
                dl = D[lab]
                cl_t.append(dl.mean())
                cl_w.append(walsh(dl))
                pt_p.append(stats.ttest_1samp(dl, 0).pvalue)
                pw_p.append(stats.wilcoxon(dl).pvalue)
                r_t, r_w = _ppi_arms(D, Dhat, lab, rng, n_boot=30)
                pp_t.append(r_t.estimate)
                pp_w.append(r_w.estimate)
            rP, rS = float(np.mean(rp)), float(np.mean(rs))
            m_t, m_w = np.var(cl_t) / np.var(pp_t), np.var(cl_w) / np.var(pp_w)
            rows.append(dict(
                dgp=kind, kappa=kappa, rP2=rP ** 2, rS2=rS ** 2, penalty=rP ** 2 - rS ** 2,
                pow_t=float(np.mean(np.array(pt_p) < .05)),
                pow_w=float(np.mean(np.array(pw_p) < .05)),
                m_t=m_t, mr_t=_robust_disp(cl_t) / _robust_disp(pp_t),
                p_t=_savings(rP ** 2), r_t=m_t / _savings(rP ** 2),
                m_w=m_w, mr_w=_robust_disp(cl_w) / _robust_disp(pp_w),
                p_w=_savings(rS ** 2), r_w=m_w / _savings(rS ** 2),
            ))
            print(f"  exp1 {kind:8s} kappa={kappa}", flush=True)
    return pd.DataFrame(rows)


def exp2(reps: int, delta: float = 0.35, seed: int = 11) -> pd.DataFrame:
    """Heavy tail in the JUDGE only; signal Gaussian, Pearson pinned at ~0.50."""
    rows = []
    for jk in ("normal", "laplace", "t3", "contam", "flip"):
        rng = np.random.default_rng(seed)
        cl_t, cl_w, pp_t, pp_w, rp, rs = ([] for _ in range(6))
        for _ in range(reps):
            D = delta + rng.standard_normal(N_POOL)          # signal: NORMAL
            Dhat = D + 1.0 * _noise(jk, rng, N_POOL)          # judge noise: varies
            lab = np.zeros(N_POOL, bool)
            lab[rng.choice(N_POOL, N_LAB, replace=False)] = True
            rp.append(stats.pearsonr(D, Dhat)[0])
            rs.append(stats.spearmanr(D, Dhat)[0])
            dl = D[lab]
            cl_t.append(dl.mean())
            cl_w.append(walsh(dl))
            r_t, r_w = _ppi_arms(D, Dhat, lab, rng, n_boot=30)
            pp_t.append(r_t.estimate)
            pp_w.append(r_w.estimate)
        rP, rS = float(np.mean(rp)), float(np.mean(rs))
        m_t, m_w = np.var(cl_t) / np.var(pp_t), np.var(cl_w) / np.var(pp_w)
        rows.append(dict(
            judge=jk, rP2=rP ** 2, rS2=rS ** 2, bonus=rS ** 2 - rP ** 2,
            p_t=_savings(rP ** 2), m_t=m_t, mr_t=_robust_disp(cl_t) / _robust_disp(pp_t),
            r_t=m_t / _savings(rP ** 2),
            p_w=_savings(rS ** 2), m_w=m_w, mr_w=_robust_disp(cl_w) / _robust_disp(pp_w),
            r_w=m_w / _savings(rS ** 2),
        ))
        print(f"  exp2 judge={jk:8s}", flush=True)
    return pd.DataFrame(rows)


def exp3(reps: int, delta: float = 0.30, n_boot: int = 500, seed: int = 23) -> pd.DataFrame:
    """Power on a common scale, with McNemar on the within-replicate discordances.

    Multipliers are within-method, so they cannot answer 'which test should I
    run'. Power can, and pairing the two arms within a replicate is what makes
    a ~0.02 swap readable against Monte Carlo noise."""
    rows = []
    for jk in ("normal", "contam"):
        rng = np.random.default_rng(seed)
        t0 = time.time()
        ct = cw = pt = pw = 0
        w_only = t_only = 0
        for _ in range(reps):
            D = delta + rng.standard_normal(N_POOL)
            Dhat = D + _noise(jk, rng, N_POOL)
            lab = np.zeros(N_POOL, bool)
            lab[rng.choice(N_POOL, N_LAB, replace=False)] = True
            dl = D[lab]
            ct += stats.ttest_1samp(dl, 0).pvalue < .05
            cw += stats.wilcoxon(dl).pvalue < .05
            r_t, r_w = _ppi_arms(D, Dhat, lab, rng, n_boot=n_boot)
            hit_t, hit_w = r_t.p_value < .05, r_w.p_value < .05
            pt += hit_t
            pw += hit_w
            w_only += (hit_w and not hit_t)
            t_only += (hit_t and not hit_w)
        disc = w_only + t_only
        mcn = stats.binomtest(w_only, disc, 0.5).pvalue if disc else 1.0
        rows.append(dict(judge=jk, classical_t=ct / reps, classical_w=cw / reps,
                         ppi_t=pt / reps, ppi_w=pw / reps,
                         w_only=w_only, t_only=t_only, mcnemar_p=mcn,
                         secs=round(time.time() - t0, 1)))
        print(f"  exp3 judge={jk:8s} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def _likert_pair(mode: str, rng, n: int = N_POOL):
    """One replicate of a 1-5 Likert paired design under judge-error `mode`.
    Returns (D, Dhat, judge_mean_shift)."""
    lo, hi = 1.0, 5.0
    subj = rng.normal(0, .8, n)
    Yx = np.clip(3.45 + subj + rng.normal(0, .6, n), lo, hi)
    Yy = np.clip(3.05 + subj + rng.normal(0, .6, n), lo, hi)
    jitter = lambda: rng.normal(0, .55, n)
    fx, fy = Yx + jitter(), Yy + jitter()
    if mode == "clean":
        pass
    elif mode == "offset":          # uniform leniency, both arms
        fx, fy = fx + 1.2, fy + 1.2
    elif mode == "differential":    # bias hits ONE arm only
        fx = fx + 0.9
    elif mode == "round":           # integer Likert -> heavy ties
        fx, fy = np.rint(fx), np.rint(fy)
    elif mode == "clip":            # judge refuses to score below 3
        fx, fy = np.clip(fx, 3, 5), np.clip(fy, 3, 5)
    elif mode == "refuse":          # 15% parse-fail -> default to midpoint
        m = rng.random(n) < .15
        fx, fy = np.where(m, 3.0, fx), np.where(m, 3.0, fy)
    elif mode == "contam":          # 8% wild misread
        m = rng.random(n) < .08
        fx = np.where(m, fx + rng.normal(0, 3.0, n), fx)
    elif mode == "hetero":          # noisy only on hard (mid-scale) items
        h = np.exp(-((Yx - 3.0) ** 2) / 1.5)
        fx = fx + rng.normal(0, 1.6, n) * h
    else:
        raise ValueError(mode)
    return Yx - Yy, fx - fy, float(np.mean((fx + fy) / 2 - (Yx + Yy) / 2))


def exp4(reps: int, seed: int = 5) -> pd.DataFrame:
    """Realistic judge-error modes: mean shift vs correlation vs realized gain."""
    rows = []
    for mode in ("clean", "offset", "differential", "round", "clip",
                 "refuse", "contam", "hetero"):
        rng = np.random.default_rng(seed)
        rp, rs, shift, cl_t, cl_w, pp_t, pp_w = ([] for _ in range(7))
        for _ in range(reps):
            D, Dhat, sh = _likert_pair(mode, rng)
            shift.append(sh)
            rp.append(stats.pearsonr(D, Dhat)[0])
            rs.append(stats.spearmanr(D, Dhat)[0])
            lab = np.zeros(N_POOL, bool)
            lab[rng.choice(N_POOL, N_LAB, replace=False)] = True
            dl = D[lab]
            cl_t.append(dl.mean())
            cl_w.append(walsh(dl))
            r_t, r_w = _ppi_arms(D, Dhat, lab, rng, n_boot=30)
            pp_t.append(r_t.estimate)
            pp_w.append(r_w.estimate)
        rP, rS = float(np.mean(rp)), float(np.mean(rs))
        rows.append(dict(mode=mode, judge_mean_shift=float(np.mean(shift)),
                         rP2=rP ** 2, rS2=rS ** 2, bonus=rS ** 2 - rP ** 2,
                         mult_t=np.var(cl_t) / np.var(pp_t),
                         mult_w=np.var(cl_w) / np.var(pp_w)))
        print(f"  exp4 {mode:13s}", flush=True)
    return pd.DataFrame(rows)


def exp5(reps: int, seed: int = 31) -> pd.DataFrame:
    """Differential-bias sweep: rho is blind to it, wilcoxon is not.

    Our own sweeps run bias_type='differential' at 0.30 population SD with
    icc=0.20, i.e. ~0.24 SD of the DIFFERENCE -- which lands in the flat part
    of this grid. So this is a real mechanism the predicted bound cannot see,
    but it is too small at our settings to explain the observed drift."""
    rows = []
    for bias in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        rng = np.random.default_rng(seed)
        rp, rs, cl_t, cl_w, pp_t, pp_w = ([] for _ in range(6))
        for _ in range(reps):
            subj = rng.normal(0, .8, N_POOL)
            Yx = 3.45 + subj + rng.normal(0, .6, N_POOL)
            Yy = 3.05 + subj + rng.normal(0, .6, N_POOL)
            fx = Yx + rng.normal(0, .55, N_POOL) + bias   # bias on ONE arm
            fy = Yy + rng.normal(0, .55, N_POOL)
            D, Dhat = Yx - Yy, fx - fy
            rp.append(stats.pearsonr(D, Dhat)[0])
            rs.append(stats.spearmanr(D, Dhat)[0])
            lab = np.zeros(N_POOL, bool)
            lab[rng.choice(N_POOL, N_LAB, replace=False)] = True
            dl = D[lab]
            cl_t.append(dl.mean())
            cl_w.append(walsh(dl))
            r_t, r_w = _ppi_arms(D, Dhat, lab, rng, n_boot=30)
            pp_t.append(r_t.estimate)
            pp_w.append(r_w.estimate)
        rP, rS = float(np.mean(rp)), float(np.mean(rs))
        m_t, m_w = np.var(cl_t) / np.var(pp_t), np.var(cl_w) / np.var(pp_w)
        rows.append(dict(bias_raw=bias, bias_in_sd_of_D=bias / float(np.std(D)),
                         rP2=rP ** 2, rS2=rS ** 2,
                         m_t=m_t, p_t=_savings(rP ** 2), r_t=m_t / _savings(rP ** 2),
                         m_w=m_w, p_w=_savings(rS ** 2), r_w=m_w / _savings(rS ** 2)))
        print(f"  exp5 bias={bias:.2f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", type=int, default=0, help="1-5, or 0 for all")
    ap.add_argument("--reps", type=int, default=0, help="override the per-experiment default")
    args = ap.parse_args()
    pd.set_option("display.width", 220)

    if args.exp in (0, 1):
        df = exp1(args.reps or 400)
        print("\n=== exp1  Q1: classical power on the labeled subset (ARE check) ===")
        print(df[df.kappa == 0.5][["dgp", "pow_t", "pow_w"]]
              .assign(ratio=lambda d: d.pow_w / d.pow_t).round(3).to_string(index=False))
        print("\n=== exp1  Q2: does each test attain its OWN bound? (ARE cancels) ===")
        print(df[["dgp", "kappa", "rP2", "rS2", "penalty", "m_t", "p_t", "r_t",
                  "m_w", "p_w", "r_w"]].round(3).to_string(index=False))

    if args.exp in (0, 2):
        df = exp2(args.reps or 800)
        print("\n=== exp2  rank penalty flips sign when the JUDGE is heavy-tailed ===")
        print(df[["judge", "rP2", "rS2", "bonus", "p_t", "p_w"]].round(3).to_string(index=False))
        print("\n=== exp2  Var-ratio (m_) vs robust MAD^2-ratio (mr_) ===")
        print(df[["judge", "m_t", "mr_t", "r_t", "m_w", "mr_w", "r_w"]].round(3).to_string(index=False))

    if args.exp in (0, 3):
        df = exp3(args.reps or 1500)
        print("\n=== exp3  power on a common scale, with McNemar ===")
        print(df.round(4).to_string(index=False))

    if args.exp in (0, 4):
        df = exp4(args.reps or 300)
        print("\n=== exp4  judge error mode -> mean shift, correlation, realized gain ===")
        print(df.round(3).to_string(index=False))

    if args.exp in (0, 5):
        df = exp5(args.reps or 500)
        print("\n=== exp5  differential bias: rho is blind, wilcoxon is not ===")
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
