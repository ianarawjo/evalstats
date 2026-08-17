"""Pre-build the classical reference power curves a label-efficiency sweep needs.

The curves are a pure, seeded function of (eval_type, effect size, methods,
n_grid, n_mc, seed), so cases/pvalues.py memoizes them to disk
(_POWER_CURVE_CACHE_DIR). Building them costs roughly 90 s each and a default
sweep needs 60 of them -- 12 pooled plus 48 per-method -- so a cold cache adds
about an hour and a half to the first run and nothing to any run after it.

Curves are independent, so this builds them across processes. On 16 cores the
full set drops from ~90 min to a few minutes.

Run it after anything that invalidates the cache: bumping
_POWER_CURVE_CACHE_VERSION, changing the data-generation path, or moving to a
different --seed or ref_n_mc. Re-running when the cache is already warm is
free -- every job is a cache hit -- so it is safe to run before an official
test as a matter of course.

Usage:
    python -m simulations.warm_power_curve_cache
    python -m simulations.warm_power_curve_cache --workers 8 --ref-n-mc 3000 --seed 56

The default seed is 56 because the CLI derives the label-efficiency seed as
args.seed + 14, so a default `--seed 42` sweep uses 56. Pass --seed to match a
non-default run; a mismatch is not an error, it just means the sweep misses
every entry written here and rebuilds them itself.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from simulations.harness.cases.pvalues import (
    _COMPARISON_METHODS,
    _COMPARISON_METHODS_BINARY,
    _JB_MIN_LAB,
    _POWER_CURVE_CACHE_DIR,
    _classical_pooled_power_curve,
)
from simulations.harness.scenarios import synthetic as S


def _build(job):
    """Build one curve. Returns (label, seconds, was_already_cached)."""
    eval_type, es, methods, label, n_mc, seed = job
    n_grid = np.geomspace(float(_JB_MIN_LAB), 1500.0, 36)
    t = time.time()
    _classical_pooled_power_curve(eval_type, es, methods, n_grid, n_mc, seed)
    return label, time.time() - t, None


def _jobs(n_mc: int, seed: int):
    src = {}
    for ef in S.PPI_LABEL_EFF_EFFECT_FRACS:
        for s in S.build_ppi_label_efficiency_sources(effect_frac=ef):
            src.setdefault((s.eval_type, ef), s.effect_size)
        for s in S.build_ppi_label_efficiency_sources_binary(effect_frac=ef):
            src.setdefault((s.eval_type, ef), s.effect_size)
    out = []
    for eval_type in ("binary", "continuous", "likert"):
        methods = _COMPARISON_METHODS_BINARY if eval_type == "binary" else _COMPARISON_METHODS
        for ef in S.PPI_LABEL_EFF_EFFECT_FRACS:
            es = src[(eval_type, ef)]
            # The pooled curve backs the headline figures; the per-method ones
            # back the per-method figures and table, which compare each test
            # against its OWN classical power rather than a pooled average.
            out.append((eval_type, es, tuple(methods), f"{eval_type} es={ef:.2f} pooled", n_mc, seed))
            for m in methods:
                out.append((eval_type, es, (m,), f"{eval_type} es={ef:.2f} {m}", n_mc, seed))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel processes (default: cpu_count-1)")
    ap.add_argument("--ref-n-mc", type=int, default=3000,
                    help="Monte Carlo draws per grid point; must match the sweep's ref_n_mc")
    ap.add_argument("--seed", type=int, default=56,
                    help="curve seed; the CLI uses --seed + 14, so a default run needs 56")
    args = ap.parse_args()

    import os
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    jobs = _jobs(args.ref_n_mc, args.seed)
    before = len(list(_POWER_CURVE_CACHE_DIR.glob("*.npy"))) if _POWER_CURVE_CACHE_DIR.exists() else 0
    print(f"{len(jobs)} curves at ref_n_mc={args.ref_n_mc}, seed={args.seed}, {workers} workers")
    print(f"cache dir: {_POWER_CURVE_CACHE_DIR}  ({before} files present)\n", flush=True)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build, j): j for j in jobs}
        for fut in as_completed(futures):
            label, secs, _ = fut.result()
            done += 1
            el = time.time() - t0
            print(f"  [{done:2d}/{len(jobs)}] {label:34s} {secs:6.1f}s   "
                  f"elapsed {el/60:5.1f}m  eta {el/done*(len(jobs)-done)/60:5.1f}m", flush=True)

    after = len(list(_POWER_CURVE_CACHE_DIR.glob("*.npy")))
    print(f"\ndone in {(time.time()-t0)/60:.1f} min; cache now holds {after} files "
          f"({after - before} new)")


if __name__ == "__main__":
    main()
