"""Validate the relative boot_se floor on the PPI joint bootstrap.

A fix that only restores POWER is worthless if it breaks FWER, so both are
measured on the same draws, for each floor coefficient c:

  FWER  : all arms equal (null). P(any pair rejected) must stay <= alpha.
  power : effects ladder.        P(extreme pair rejected). Want it back up.
  a_eff : the effective per-pair alpha the "boot" CI widening derives from
          M_b. Should sit near the Sidak value (1-(1-.05)^(1/n_pairs)),
          NOT collapse toward 0 (which is what over-widens likert's CIs).

c=0.0 reproduces the shipped code exactly (verified bit-for-bit already).
"""
import sys, warnings
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
sys.path.insert(0, "/private/tmp/claude-501/-Users-ianarawjo-Documents-prompt-stats/f2b00edf-b3f0-44a7-88c1-f7e3cc56181a/scratchpad/rwfix")
warnings.filterwarnings("ignore")

import patched
from evalstats.api import (_ppi_romano_wolf_pvalues_from_joint_stats,
                           _M_b_from_T, _ppi_alpha_eff_from_M_b)
from simulations.harness.cases.compare_e2e import (
    _apply_judge_noise, _effect_step_for, _effect_frac_for, _agreement_for)
from simulations.harness.scenarios.synthetic import (
    CONTINUOUS_SHAPES, LIKERT_SHAPES, sample_group_truth)

ALPHA = 0.05
SHAPES = {"continuous": CONTINUOUS_SHAPES, "likert": LIKERT_SHAPES}


def one_rep(fn, eval_type, K, N, frac, is_null, rng, n_boot):
    shape = SHAPES[eval_type][0]
    step = 0.0 if is_null else _effect_step_for(eval_type, _effect_frac_for(eval_type))
    eff = np.arange(K, dtype=float) * step
    truth = sample_group_truth(shape, N, 1, K, 1.0, rng, effects=eff)[:, :, 0]
    llm = _apply_judge_noise(truth, eval_type, rng, _agreement_for(eval_type))
    n_lab = max(1, round(N * frac))
    lab = rng.choice(N, size=n_lab, replace=False)
    lm = np.full_like(llm, np.nan); lm[:, lab] = truth[:, lab]
    ents = [f"M{i}" for i in range(K)]
    ei = {e: i for i, e in enumerate(ents)}
    pks = [(a, b) for i, a in enumerate(ents) for b in ents[i+1:]]
    out = fn(llm, lm, pks, ei, n_boot, np.random.default_rng(int(rng.integers(0, 2**31))),
             power_tune=True)
    if out is None:
        return None
    pe, ose, valid, T, tobs = out
    rw = _ppi_romano_wolf_pvalues_from_joint_stats(pe, ose, valid, T, tobs, pks)
    a_eff = _ppi_alpha_eff_from_M_b(_M_b_from_T(T, valid), 1 - ALPHA)
    any_rej = any(p < ALPHA for p in rw.values())
    ext = rw[("M0", f"M{K-1}")]
    return any_rej, ext < ALPHA, a_eff


def run(fn, eval_type, K, N, frac, reps, n_boot, seed):
    rng = np.random.default_rng(seed)
    fw, pw, ae = [], [], []
    for _ in range(reps):
        r = one_rep(fn, eval_type, K, N, frac, True, rng, n_boot)
        if r: fw.append(r[0]); ae.append(r[2])
    rng = np.random.default_rng(seed + 10_000)
    for _ in range(reps):
        r = one_rep(fn, eval_type, K, N, frac, False, rng, n_boot)
        if r: pw.append(r[1])
    return (float(np.mean(fw)) if fw else float("nan"),
            float(np.mean(pw)) if pw else float("nan"),
            float(np.median(ae)) if ae else float("nan"))


if __name__ == "__main__":
    REPS, NBOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 200, 1000
    CS = [0.0, 0.10, 0.20, 0.30, 0.50]
    CONDS = [("continuous", 3, 100, 0.40), ("continuous", 3, 200, 0.40),
             ("continuous", 3, 100, 0.20), ("likert", 3, 50, 0.40),
             ("likert", 3, 100, 0.40), ("likert", 3, 200, 0.40),
             ("continuous", 5, 100, 0.40), ("likert", 5, 100, 0.40)]
    fns = {c: patched.make(c) for c in CS}
    print(f"reps={REPS} n_boot={NBOOT} alpha={ALPHA}  (c=0.0 == shipped)")
    for et, K, N, frac in CONDS:
        sidak = 1 - (1 - ALPHA) ** (1.0 / (K*(K-1)//2))
        print(f"\n--- {et} k={K} N={N} frac={frac}   (Sidak per-pair alpha={sidak:.4f}) ---")
        print(f"{'c':>6s} {'FWER(null)':>11s} {'power':>8s} {'median a_eff':>13s}")
        for c in CS:
            f, p, a = run(fns[c], et, K, N, frac, REPS, NBOOT, 4242)
            flag = "  <-- FWER BREACH" if f > ALPHA + 3*np.sqrt(ALPHA*(1-ALPHA)/max(REPS,1)) else ""
            print(f"{c:>6.2f} {f:>11.4f} {p:>8.4f} {a:>13.6f}{flag}")
