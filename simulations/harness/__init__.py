"""Centralized simulation harness for evalstats.

This package consolidates the calibration/comparison Monte Carlo simulations
in ``simulations/*.py`` into a single CLI entry point (``python -m
simulations.harness.cli``) built on a shared synthetic-scenario library
(``scenarios/synthetic.py``) and a shared real-data adapter
(``scenarios/real_data.py``), so distributions and data sources can be
described once for the paper's Supplementary Material instead of once per
script. See ``simulations/harness/README.md`` for the case list and known
exceptions. The legacy scripts in ``simulations/`` are left untouched.
"""

import os

# Must run before numpy/scipy/statsmodels are imported anywhere under this
# package (this __init__.py is guaranteed to execute first, since
# `simulations` itself has no __init__.py of its own -- a namespace
# package). On macOS, numpy links against Apple's Accelerate framework for
# BLAS/LAPACK, which parallelizes routines like SVD internally via Grand
# Central Dispatch (GCD) worker threads. Every case module here forks worker
# pools (multiprocessing.get_context("fork")) for Monte Carlo parallelism;
# fork() only duplicates the calling thread, so any GCD state already
# spun up in the parent (from an earlier numpy/scipy call) can be left
# inconsistent in the child. The child then segfaults (EXC_BAD_ACCESS,
# "crashed on child side of fork pre-exec" in the crash report) the next
# time it calls into Accelerate -- e.g. statsmodels' MixedLM REML fit,
# which uses SVD internally (cases/pvalues.py's lmm/lmm_factorial/lmm_runs
# ppi tests). multiprocessing.Pool has no way to detect a worker dying this
# way, so imap_unordered just hangs forever waiting for a result that will
# never arrive -- indistinguishable from a hang from the outside. Forcing
# Accelerate to run single-threaded (no GCD dispatch at all) sidesteps the
# hazard entirely. setdefault so an explicit user override still wins.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
