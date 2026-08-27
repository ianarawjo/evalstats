"""How much does the FlipFlop scenario depend on WHICH items got human labels?

`paper_flipflop_example.py` shows one draw of 30 labeled items per app (the
paper's PAPER_SEED). A reader is entitled to ask whether that draw was lucky,
so this script re-runs the whole corrected analysis over many independent
draws and reports the distribution. It is the source of the paper's robustness
footnote in the mixed human-AI judge scenario.

Two sources of randomness vary together here, on purpose: the labeling draw
AND the bootstrap seed. Holding the bootstrap fixed would understate the
spread a reader re-running the example actually sees. (When SELECTING a draw
rather than characterising the spread, hold the bootstrap fixed instead, or
you select on bootstrap noise.)

The footnote this replaced cited 8 draws, which was badly underpowered: it
reported that the corrected omnibus "never rejects" (it rejects in 2 of 150),
put the median p at 0.48 (0.71 over 150), and gave a range of 0.19-0.80 that
is roughly a fifth of the true 0.018-0.999. A min and a max from 8 draws will
always understate a range -- they are the statistics most sensitive to n.

Result at the paper's settings (150 draws, n_lab=30/app, n_bootstrap=2000):

    corrected omnibus p : median 0.710, range 0.018-0.999
                          rejects at alpha=.05 in 2 of 150
    FlipFlop-Wavelength : significant in 6 of 150
    theta               : median 0.458, range 0.358-0.555

    the shown draw (PAPER_SEED=8): theta 0.458 -- rank 1 of 150 closest to
    the median theta, off by 0.0004 -- while its omnibus p sits at the 23rd
    percentile, i.e. a slightly HARDER case for the correction than typical.

Runtime is ~6s per draw, so the default 150 takes ~15 minutes.

Run:
    .venv/bin/python -m simulations.paper_flipflop_draw_robustness
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

import evalstats as es
from evalstats.alignment import judge_alignment

from simulations.paper_flipflop_example import (
    DEFAULT_DATA_DIR,
    N_LAB,
    PAPER_SEED,
    load,
    sample_labels,
)


def run_draw(df, draw_seed: int, n_lab: int, n_bootstrap: int) -> dict:
    """One labeling draw, corrected end to end, as the scenario runs it."""
    lab = sample_labels(df, n_lab, seed=draw_seed)
    evaldata = es.load_from(lab)
    ar = judge_alignment(
        evaldata, llm_metric="satisfaction_score",
        human_groundtruth="human_score", selection="random",
    )
    result = es.compare(
        evaldata, factors="app", metric="satisfaction_score", design="unpaired",
        alignment={"satisfaction_score": ar}, score_range=(1, 5),
        # Bootstrap seed tracks the draw seed: see the module docstring on why
        # these vary together rather than one being pinned.
        rng=np.random.default_rng(1000 + draw_seed), n_bootstrap=n_bootstrap,
    )
    pair = next(c for c in result.pairwise
                if {c.label_a, c.label_b} == {"FlipFlop", "Wavelength"})
    means = {g.label: g.mean for g in result.groups}
    return dict(
        seed=draw_seed,
        p_omnibus=result.omnibus_corrected_p_value,
        theta=pair.point_estimate,
        p_pair=pair.p_value,
        significant=bool(pair.significant),
        gap=means["FlipFlop"] - means["Wavelength"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--n-draws", type=int, default=150)
    ap.add_argument("--n-lab", type=int, default=N_LAB,
                    help="human labels per condition (default: the paper's)")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--quiet", action="store_true", help="suppress per-draw lines")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    df = load(args.data_dir)
    print(f"Loaded {len(df)} real reviews across {df['app'].nunique()} apps; "
          f"{args.n_draws} draws of {args.n_lab} labels/app\n")

    rows = []
    for seed in range(args.n_draws):
        rows.append(run_draw(df, seed, args.n_lab, args.n_bootstrap))
        if not args.quiet:
            r = rows[-1]
            print(f"  seed {r['seed']:3d}  p_omnibus={r['p_omnibus']:.4f}  "
                  f"theta={r['theta']:.3f}  p_pair={r['p_pair']:.4f}  "
                  f"significant={r['significant']}", flush=True)

    p_om = np.array([r["p_omnibus"] for r in rows])
    theta = np.array([r["theta"] for r in rows])
    sig = np.array([r["significant"] for r in rows])
    n = len(rows)

    print(f"\n===== {n} independent labeling draws =====")
    print(f"corrected omnibus p : median {np.median(p_om):.3f}  "
          f"range {p_om.min():.3f}-{p_om.max():.3f}")
    print(f"                      rejects at alpha=.05 in {(p_om < 0.05).sum()} of {n}")
    print(f"FlipFlop-Wavelength : significant in {sig.sum()} of {n}")
    print(f"theta               : median {np.median(theta):.3f}  "
          f"range {theta.min():.3f}-{theta.max():.3f}")

    shown = next((r for r in rows if r["seed"] == PAPER_SEED), None)
    if shown is not None:
        d = np.abs(theta - np.median(theta))
        rank = int(np.argsort(d).tolist().index(rows.index(shown))) + 1
        print(f"\nthe draw the paper shows (PAPER_SEED={PAPER_SEED}):")
        print(f"  theta      {shown['theta']:.3f}  (rank {rank} of {n} closest to "
              f"the median theta)")
        print(f"  omnibus p  {shown['p_omnibus']:.4f}  "
              f"({(p_om < shown['p_omnibus']).mean():.0%} percentile -- lower is a "
              f"HARDER case for the correction)")
        print(f"  gap        {shown['gap']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
