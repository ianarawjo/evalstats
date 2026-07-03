"""ci_paired case: paired-difference CI coverage, synthetic and/or real benchmark data.

One engine operating over ``scenarios.CIPairSource`` objects, so the same
CI-method dispatch and report format covers synthetic paired distributions
(``scenarios/synthetic.py``) and real shared-item OpenEval / Inspect AI
corpora (``scenarios/real_data.py``) alike.

Methods compared
-----------------
bootstrap, bca, bayes_bootstrap, smooth_bootstrap, bootstrap_t (all eval
types, statistic=mean or median); t_interval, nig, el (non-binary,
statistic=mean only); newcombe_score, tango_score, bayes_indep_comp,
bayes_paired_comp (binary, statistic=mean only).

Known exceptions (see simulations/harness/README.md):
- Real-data pairs (openeval/inspect/real) only support R=1 (flat, single
  run per item) and are restricted to known-binary benchmarks. Multi-run
  real pairs (Tango multirun variants, nested-diff bootstrap, lmm_diff) are
  deferred to a future real-data extension.

--nested-mode (ported from simulations/sim_compare_boot_nested.py, pairwise
phase) sweeps multi-run paired scenarios parameterised by run_noise_frac
(synthetic only), reporting flat (cell-mean reduction) and "*_nested"/
"*_flat"/"*_multirun_*" CI methods side by side. There is no separate
ci_nested case -- see cases/ci_single.py's --nested-mode for the
single-sample analogue.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import multiprocessing as _mp
import os
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.resampling import (
        bootstrap_ci_1d,
        bootstrap_t_ci_1d,
        bca_interval_1d,
        bayes_bootstrap_means_1d,
        smooth_bootstrap_means_1d,
        bootstrap_means_1d,
        bootstrap_diffs_nested,
        bayes_bootstrap_diffs_nested,
        smooth_bootstrap_diffs_nested,
        resolve_resampling_method,
        t_interval_ci_1d,
        nig_ci_1d,
        el_ci_1d,
        tango_paired_ci,
        newcombe_paired_ci,
        tango_paired_ci_flat,
        tango_paired_ci_multirun_cluster,
        tango_paired_ci_multirun_effective,
        tango_paired_ci_multirun_moments,
    )
    from evalstats.core.stats_utils import interval_score, rescaled_ci

from ...bayes_evals import binorm_cdf  # noqa: E402

from ..latex_tables import booktabs_table, escape_latex, eval_type_label
from ..scenarios import CIPairSource, EVAL_TYPES, EVAL_TYPE_SCALE_BOUNDS
from ..scenarios.synthetic import (
    SCENARIO_SUITES,
    RUN_NOISE_FRACS_DEFAULT,
    build_pair_sources,
)
from ..scenarios.real_data import DEFAULT_INSPECT_CSV, PAIR_SOURCES as REAL_PAIR_SOURCES, build_real_pair_sources
from ..methods import (
    BOOTSTRAP_METHODS as METHODS,
    BAYES_BOOTSTRAP,
    T_INTERVAL,
    NIG,
    EL,
    NEWCOMBE,
    TANGO,
    BAYES_PAIR_INDEP,
    BAYES_PAIR_PAIRED,
    PAIRWISE_EXTRA_METHODS,
    PAIR_DIFF_NESTED_METHODS,
    BOOTSTRAP_DIFF_NESTED,
    BAYES_DIFF_NESTED,
    SMOOTH_DIFF_NESTED,
    BINARY_PAIR_FLAT_METHODS,
    TANGO_FLAT,
    NEWCOMBE_FLAT,
    BINARY_PAIR_NESTED_METHODS,
    TANGO_MULTIRUN_CLUSTER,
    TANGO_MULTIRUN_EFFECTIVE,
    TANGO_MULTIRUN_MOMENTS,
    get_method_color,
    order_present_methods,
)
from . import CaseResult

CASE_NAME = "ci_paired"

DATA_SOURCES = ["synthetic"] + REAL_PAIR_SOURCES
PROGRESS_MODES = ["bar", "cell", "off"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]


@dataclass
class SimResult:
    source: str  # "synthetic" | "openeval" | "inspect" | "real"
    label: str
    eval_type: str
    n: int
    method: str
    n_reps: int
    covered: int
    total_width: float
    total_score: float = 0.0
    """Sum of interval_score() (see evalstats.core.stats_utils) across n_reps."""
    total_time: float = 0.0
    total_time_sq: float = 0.0
    is_null: bool = False
    model_a: str | None = None
    model_b: str | None = None
    benchmark_id: str | None = None
    corpus_size: int | None = None
    true_diff: float | None = None
    run_noise_frac: float = 0.0
    """f_run for --nested-mode rows (scenarios.synthetic.build_pair_sources' run_noise_fracs); 0.0 otherwise."""
    runs: int = 1
    """Runs per input for --nested-mode rows; 1 (flat) otherwise."""


def _stat(values: np.ndarray, statistic: str = "mean") -> float:
    return float(np.median(values)) if statistic == "median" else float(np.mean(values))


def _wilson_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    from scipy.stats import norm

    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _newcombe_paired_score_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Newcombe score CI for paired binary difference p(A=1) - p(B=1)."""
    n = int(a.shape[0])
    if n <= 0:
        return (0.0, 0.0)
    a_bin = (a >= 0.5).astype(int)
    b_bin = (b >= 0.5).astype(int)
    n10 = int(np.sum((a_bin == 1) & (b_bin == 0)))
    n01 = int(np.sum((a_bin == 0) & (b_bin == 1)))
    m = n10 + n01
    if m == 0:
        return (0.0, 0.0)
    theta_low, theta_high = _wilson_ci(successes=n10, n=m, alpha=alpha)
    scale = m / n
    return float(scale * (2.0 * theta_low - 1.0)), float(scale * (2.0 * theta_high - 1.0))


def _bayes_indep_comp_ci(a: np.ndarray, b: np.ndarray, alpha: float, num_samples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Independent Beta-posteriors CI for paired binary difference p(A=1)-p(B=1)."""
    a_bin = (a >= 0.5).astype(float)
    b_bin = (b >= 0.5).astype(float)
    post_a = rng.beta(float(np.sum(a_bin)) + 1.0, float(a_bin.shape[0] - np.sum(a_bin)) + 1.0, size=num_samples)
    post_b = rng.beta(float(np.sum(b_bin)) + 1.0, float(b_bin.shape[0] - np.sum(b_bin)) + 1.0, size=num_samples)
    diff = post_a - post_b
    return float(np.percentile(diff, 100.0 * alpha / 2.0)), float(np.percentile(diff, 100.0 * (1.0 - alpha / 2.0)))


def _bayes_paired_comp_ci(a: np.ndarray, b: np.ndarray, alpha: float, num_samples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Paired Bayesian CI for p(A=1)-p(B=1) using the bivariate-normal latent model from bayes_evals.py."""
    a_bin = (a >= 0.5).astype(float)
    b_bin = (b >= 0.5).astype(float)

    s = float(np.sum(a_bin * b_bin))
    t = float(np.sum(a_bin * (1.0 - b_bin)))
    u = float(np.sum((1.0 - a_bin) * b_bin))
    v = float(np.sum((1.0 - a_bin) * (1.0 - b_bin)))

    theta_as = rng.beta(1.0, 1.0, size=num_samples)
    theta_bs = rng.beta(1.0, 1.0, size=num_samples)
    rhos = np.clip(2.0 * rng.beta(4.0, 2.0, size=num_samples) - 1.0, -1 + 1e-20, 1 - 1e-20)
    diff = theta_as - theta_bs

    mu_a = stats.norm.ppf(theta_as)
    mu_b = stats.norm.ppf(theta_bs)

    th_v = binorm_cdf(0, 0, mu_a, mu_b, 1, 1, rhos)
    th_s = theta_as + theta_bs + th_v - 1.0
    th_t = 1.0 - theta_bs - th_v
    th_u = 1.0 - theta_as - th_v

    with np.errstate(divide="ignore", invalid="ignore"):
        log_w = s * np.log(th_s) + t * np.log(th_t) + u * np.log(th_u) + v * np.log(th_v)

    log_w -= np.nanmax(log_w)
    w = np.exp(log_w)
    w[np.isnan(w)] = 0.0
    w_sum = float(np.sum(w))

    if w_sum <= 0.0:
        d_hat = float(np.mean(a_bin) - np.mean(b_bin))
        return d_hat, d_hat

    w /= w_sum
    diff_post = diff[rng.choice(num_samples, size=num_samples, replace=True, p=w)]
    return float(np.percentile(diff_post, 100.0 * alpha / 2.0)), float(np.percentile(diff_post, 100.0 * (1.0 - alpha / 2.0)))


def _pairwise_ci(
    a: np.ndarray, b: np.ndarray, method: str, n_bootstrap: int, alpha: float,
    rng: np.random.Generator, *, statistic: str = "mean",
) -> tuple[float, float]:
    """Compute CI for paired mean/median difference A-B using evalstats logic."""
    n_inputs, runs = a.shape
    resolved_method = resolve_resampling_method(method, n_inputs)

    if runs >= 3:
        cell_diffs = a.mean(axis=1) - b.mean(axis=1)
        observed = _stat(cell_diffs, statistic=statistic)

        if resolved_method == "bayes_bootstrap":
            boot_stats = bayes_bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
            return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        if resolved_method == "smooth_bootstrap":
            boot_stats = smooth_bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
            return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        if resolved_method == "bootstrap_t":
            return bootstrap_t_ci_1d(cell_diffs, observed, n_bootstrap, alpha, rng, statistic=statistic)
        boot_stats = bootstrap_diffs_nested(a, b, n_bootstrap, rng, statistic=statistic)
        if resolved_method == "bca":
            return bca_interval_1d(cell_diffs, observed, boot_stats, alpha, statistic=statistic)
        return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    diffs = a.mean(axis=1) - b.mean(axis=1)
    observed = _stat(diffs, statistic=statistic)

    if resolved_method == "bayes_bootstrap":
        boot_stats = bayes_bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
        return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    if resolved_method == "smooth_bootstrap":
        boot_stats = smooth_bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
        return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    if resolved_method == "bootstrap_t":
        return bootstrap_t_ci_1d(diffs, observed, n_bootstrap, alpha, rng, statistic=statistic)

    boot_stats = bootstrap_means_1d(diffs, n_bootstrap, rng, statistic=statistic)
    if resolved_method == "bca":
        return bca_interval_1d(diffs, observed, boot_stats, alpha, statistic=statistic)
    return float(np.percentile(boot_stats, 100 * alpha / 2)), float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))


def _run_cell(
    source_obj: CIPairSource, n: int, n_reps: int, n_bootstrap: int, bayes_n: int,
    alpha: float, runs: int, statistic: str, seed,
) -> list[SimResult]:
    """Run all reps for one (source, n) cell -- pairwise estimand."""
    rng = np.random.default_rng(seed)

    add_newcombe = source_obj.eval_type == "binary" and statistic == "mean"
    add_tango = source_obj.eval_type == "binary" and statistic == "mean"
    add_bayes_binary = source_obj.eval_type == "binary" and statistic == "mean"
    add_pairwise_extras = statistic == "mean" and source_obj.eval_type != "binary"

    active_methods = list(METHODS)
    if add_pairwise_extras:
        active_methods += PAIRWISE_EXTRA_METHODS
    if add_newcombe:
        active_methods.append(NEWCOMBE)
    if add_tango:
        active_methods.append(TANGO)
    if add_bayes_binary:
        active_methods += [BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED]

    covered: dict = {m: 0 for m in active_methods}
    total_w: dict = {m: 0.0 for m in active_methods}
    total_score: dict = {m: 0.0 for m in active_methods}
    total_t: dict = {m: 0.0 for m in active_methods}
    total_t_sq: dict = {m: 0.0 for m in active_methods}
    true_diff = source_obj.true_diff

    def _record(method, ci_low: float, ci_high: float) -> None:
        if ci_low <= true_diff <= ci_high:
            covered[method] += 1
        total_w[method] += ci_high - ci_low
        total_score[method] += interval_score(ci_low, ci_high, true_diff, alpha)

    for _rep in range(n_reps):
        a, b = source_obj.generate_pair(rng, n, runs)

        for method in METHODS:
            _t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ci_low, ci_high = _pairwise_ci(
                        a, b, method=method.name, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng, statistic=statistic,
                    )
            except Exception:
                obs = _stat(a.mean(axis=1) - b.mean(axis=1), statistic=statistic)
                ci_low = ci_high = obs
            _el = time.perf_counter() - _t0
            total_t[method] += _el
            total_t_sq[method] += _el * _el
            _record(method, ci_low, ci_high)

        if add_pairwise_extras:
            pair_diffs = a.mean(axis=1) - b.mean(axis=1)
            obs = float(np.mean(pair_diffs))
            # A paired difference of two [scale_lo, scale_hi] values ranges
            # over [-span, span], naturally centred at 0 -- not [scale_lo,
            # scale_hi] itself. nig_ci_1d's default prior assumes its input
            # is centred at the scale's midpoint, so pass diff_lo/diff_hi
            # (not scale_lo/scale_hi) to correctly re-centre it at 0.
            _scale_lo, _scale_hi = EVAL_TYPE_SCALE_BOUNDS[source_obj.eval_type]
            diff_span = _scale_hi - _scale_lo
            diff_lo, diff_hi = -diff_span, diff_span
            for method, fn in zip(PAIRWISE_EXTRA_METHODS, (t_interval_ci_1d, nig_ci_1d, el_ci_1d)):
                _t0 = time.perf_counter()
                try:
                    if fn is nig_ci_1d:
                        ci_low, ci_high = rescaled_ci(fn, pair_diffs, alpha, diff_lo, diff_hi)
                    else:
                        ci_low, ci_high = fn(pair_diffs, alpha)
                except Exception:
                    ci_low = ci_high = obs
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                _record(method, ci_low, ci_high)

        if add_newcombe:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _newcombe_paired_score_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[NEWCOMBE] += _el
            total_t_sq[NEWCOMBE] += _el * _el
            _record(NEWCOMBE, ci_low, ci_high)

        if add_tango:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = tango_paired_ci(a[:, 0], b[:, 0], alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[TANGO] += _el
            total_t_sq[TANGO] += _el * _el
            _record(TANGO, ci_low, ci_high)

        if add_bayes_binary:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_indep_comp_ci(a[:, 0], b[:, 0], alpha, bayes_n, rng)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[BAYES_PAIR_INDEP] += _el
            total_t_sq[BAYES_PAIR_INDEP] += _el * _el
            _record(BAYES_PAIR_INDEP, ci_low, ci_high)

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_paired_comp_ci(a[:, 0], b[:, 0], alpha, bayes_n, rng)
            except Exception:
                ci_low = ci_high = float(np.mean(a[:, 0] - b[:, 0]))
            _el = time.perf_counter() - _t0
            total_t[BAYES_PAIR_PAIRED] += _el
            total_t_sq[BAYES_PAIR_PAIRED] += _el * _el
            _record(BAYES_PAIR_PAIRED, ci_low, ci_high)

    return [
        SimResult(
            source=source_obj.source, label=source_obj.label, eval_type=source_obj.eval_type,
            n=n, method=method.name, n_reps=n_reps, covered=covered[method],
            total_width=total_w[method], total_score=total_score[method],
            total_time=total_t[method], total_time_sq=total_t_sq[method],
            is_null=source_obj.is_null, model_a=source_obj.model_a, model_b=source_obj.model_b,
            benchmark_id=source_obj.benchmark_id, corpus_size=source_obj.max_n,
            true_diff=(source_obj.true_diff if source_obj.source != "synthetic" else None),
        )
        for method in active_methods
    ]


class _ProgressReporter:
    def __init__(self, total: int, *, mode: str = "bar", label: str = "") -> None:
        self.total = max(int(total), 1)
        self.mode = mode
        self.label = label
        self.start = time.time()
        self.last_print = 0.0

    def update(self, step: int, detail: str = "") -> None:
        if self.mode == "off":
            return
        now = time.time()
        is_final = step >= self.total
        if not is_final and (now - self.last_print) < 0.2:
            return
        self.last_print = now
        if self.mode == "cell":
            pct = 100.0 * min(step, self.total) / self.total
            print(f"\r  [{step:>7d}/{self.total:<7d}] {pct:6.2f}%  {detail:<55s}", end="", flush=True)
            if is_final:
                print()
            return
        frac = min(step, self.total) / self.total
        filled = int(28 * frac)
        bar = "█" * filled + "░" * (28 - filled)
        elapsed = max(now - self.start, 1e-9)
        rate = step / elapsed
        eta_sec = max(self.total - step, 0) / max(rate, 1e-12)
        eta_m, eta_s = divmod(int(round(eta_sec)), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        prefix = f"{self.label}: " if self.label else ""
        print(
            f"\r  {prefix}[{bar}] {100.0*frac:6.2f}%  {step:>7d}/{self.total:<7d}  "
            f"ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}  {detail[:40]:<40s}",
            end="", flush=True,
        )
        if is_final:
            print()


_CELL_SOURCES: list = []  # fork-inherited worker state for run_simulation
_NESTED_CELL_SOURCES: list = []  # fork-inherited worker state for run_nested_pairwise_simulation


def _run_cell_worker(args: tuple) -> list[SimResult]:
    sc_idx, n, n_reps, n_bootstrap, bayes_n, alpha, runs, statistic, seed = args
    return _run_cell(_CELL_SOURCES[sc_idx], n, n_reps, n_bootstrap, bayes_n, alpha, runs, statistic, seed)


def _run_nested_cell_worker(args: tuple) -> list[SimResult]:
    sc_idx, n, runs, n_reps, n_bootstrap, bayes_n, alpha, seed = args
    return _run_nested_pairwise_cell(_NESTED_CELL_SOURCES[sc_idx], n, runs, n_reps, n_bootstrap, bayes_n, alpha, seed)


def run_simulation(
    sources: list[CIPairSource], sample_sizes: list[int], n_reps: int, n_bootstrap: int,
    bayes_n: int, alpha: float, runs: int, statistic: str,
    progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
) -> list[SimResult]:
    global _CELL_SOURCES
    _CELL_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = [(i, n) for i, s in enumerate(sources) for n in sample_sizes if s.max_n is None or n < s.max_n]
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, n_reps, n_bootstrap, bayes_n, alpha, runs, statistic, seed)
                 for (sc_idx, n), seed in zip(cells, child_seeds)]

    skipped = [(s, n) for s in sources for n in sample_sizes if not (s.max_n is None or n < s.max_n)]
    for s, n in skipped:
        print(f"  Warning: sample size n={n} >= corpus size {s.max_n} for {s.label}. Skipping.")

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="ci_paired")
    results: list[SimResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_cell_worker(a))
            sc_idx, n = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} {sources[sc_idx].label} n={n}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


# ---------------------------------------------------------------------------
# Nested mode (--nested-mode): flat vs. nested pairwise CI methods on
# multi-run data, ported from sim_compare_boot_nested.py's
# _run_pairwise_multirun_cell. Mean-statistic only (matches the legacy
# script -- nested mode never supported --statistic median).
# ---------------------------------------------------------------------------


def _run_nested_pairwise_cell(
    source_obj: CIPairSource, n: int, runs: int, n_reps: int, n_bootstrap: int, bayes_n: int, alpha: float, seed,
) -> list[SimResult]:
    """Run all reps for one (source, n, runs) cell -- flat-vs-nested pairwise mean-diff estimand."""
    rng = np.random.default_rng(seed)
    is_binary = source_obj.eval_type == "binary"
    true_diff = source_obj.true_diff

    active_methods = list(METHODS) + [T_INTERVAL] + PAIR_DIFF_NESTED_METHODS
    if is_binary:
        active_methods += BINARY_PAIR_FLAT_METHODS + BINARY_PAIR_NESTED_METHODS

    covered: dict = {m: 0 for m in active_methods}
    total_w: dict = {m: 0.0 for m in active_methods}
    total_score: dict = {m: 0.0 for m in active_methods}
    total_t: dict = {m: 0.0 for m in active_methods}
    total_t_sq: dict = {m: 0.0 for m in active_methods}

    def _record(method, ci_low: float, ci_high: float) -> None:
        if ci_low <= true_diff <= ci_high:
            covered[method] += 1
        total_w[method] += ci_high - ci_low
        total_score[method] += interval_score(ci_low, ci_high, true_diff, alpha)

    for _rep in range(n_reps):
        a, b = source_obj.generate_pair(rng, n, runs)
        cell_diffs = a.mean(axis=1) - b.mean(axis=1)
        obs_diff = float(np.mean(cell_diffs))

        # -- Cell-mean diff bootstrap family (flat) --
        for method in METHODS:
            n_draws = bayes_n if method is BAYES_BOOTSTRAP else n_bootstrap
            _t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ci_low, ci_high = bootstrap_ci_1d(cell_diffs, obs_diff, method=method.name, n_bootstrap=n_draws, alpha=alpha, rng=rng)
            except Exception:
                ci_low = ci_high = obs_diff
            _el = time.perf_counter() - _t0
            total_t[method] += _el
            total_t_sq[method] += _el * _el
            _record(method, ci_low, ci_high)

        # -- t-interval on cell-mean diffs --
        _t0 = time.perf_counter()
        try:
            ci_low, ci_high = t_interval_ci_1d(cell_diffs, alpha)
        except Exception:
            ci_low = ci_high = obs_diff
        _el = time.perf_counter() - _t0
        total_t[T_INTERVAL] += _el
        total_t_sq[T_INTERVAL] += _el * _el
        _record(T_INTERVAL, ci_low, ci_high)

        # -- Nested pairwise diff methods (full N x R pair matrices) --
        for method, fn in [
            (BOOTSTRAP_DIFF_NESTED, bootstrap_diffs_nested),
            (BAYES_DIFF_NESTED, bayes_bootstrap_diffs_nested),
            (SMOOTH_DIFF_NESTED, smooth_bootstrap_diffs_nested),
        ]:
            n_draws = bayes_n if method is BAYES_DIFF_NESTED else n_bootstrap
            _t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*falling back to plain bootstrap; no KDE smoothing applied.*",
                        category=UserWarning,
                    )
                    boot_stats = fn(a, b, n_draws, rng)
                ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
                ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
            except Exception:
                ci_low = ci_high = obs_diff
            _el = time.perf_counter() - _t0
            total_t[method] += _el
            total_t_sq[method] += _el * _el
            _record(method, ci_low, ci_high)

        # -- Binary pairwise methods --
        if is_binary:
            a0, b0 = a[:, 0], b[:, 0]  # first run only (flat iid baseline)

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = tango_paired_ci_flat(a, b, alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a0 - b0))
            _el = time.perf_counter() - _t0
            total_t[TANGO_FLAT] += _el
            total_t_sq[TANGO_FLAT] += _el * _el
            _record(TANGO_FLAT, ci_low, ci_high)

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = newcombe_paired_ci(a0, b0, alpha)
            except Exception:
                ci_low = ci_high = float(np.mean(a0 - b0))
            _el = time.perf_counter() - _t0
            total_t[NEWCOMBE_FLAT] += _el
            total_t_sq[NEWCOMBE_FLAT] += _el * _el
            _record(NEWCOMBE_FLAT, ci_low, ci_high)

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_indep_comp_ci(a0, b0, alpha, n_bootstrap, rng)
            except Exception:
                ci_low = ci_high = float(np.mean(a0 - b0))
            _el = time.perf_counter() - _t0
            total_t[BAYES_PAIR_INDEP] += _el
            total_t_sq[BAYES_PAIR_INDEP] += _el * _el
            _record(BAYES_PAIR_INDEP, ci_low, ci_high)

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_paired_comp_ci(a0, b0, alpha, n_bootstrap, rng)
            except Exception:
                ci_low = ci_high = float(np.mean(a0 - b0))
            _el = time.perf_counter() - _t0
            total_t[BAYES_PAIR_PAIRED] += _el
            total_t_sq[BAYES_PAIR_PAIRED] += _el * _el
            _record(BAYES_PAIR_PAIRED, ci_low, ci_high)

            for method, fn in [
                (TANGO_MULTIRUN_CLUSTER, tango_paired_ci_multirun_cluster),
                (TANGO_MULTIRUN_EFFECTIVE, tango_paired_ci_multirun_effective),
                (TANGO_MULTIRUN_MOMENTS, tango_paired_ci_multirun_moments),
            ]:
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = fn(a, b, alpha)
                except Exception:
                    ci_low = ci_high = obs_diff
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                _record(method, ci_low, ci_high)

    return [
        SimResult(
            source="synthetic", label=source_obj.label, eval_type=source_obj.eval_type,
            n=n, method=method.name, n_reps=n_reps, covered=covered[method],
            total_width=total_w[method], total_score=total_score[method],
            total_time=total_t[method], total_time_sq=total_t_sq[method],
            is_null=source_obj.is_null, run_noise_frac=source_obj.run_noise_frac, runs=runs,
        )
        for method in active_methods
    ]


def run_nested_pairwise_simulation(
    sources: list[CIPairSource], sample_sizes: list[int], runs: int, n_reps: int, n_bootstrap: int,
    bayes_n: int, alpha: float, progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
) -> list[SimResult]:
    global _NESTED_CELL_SOURCES
    _NESTED_CELL_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = [(i, n) for i, s in enumerate(sources) for n in sample_sizes]
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, runs, n_reps, n_bootstrap, bayes_n, alpha, seed)
                 for (sc_idx, n), seed in zip(cells, child_seeds)]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label=f"ci_paired-nested[runs={runs}]")
    results: list[SimResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_nested_cell_worker(a))
            sc_idx, n = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} {sources[sc_idx].label} n={n}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_nested_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _cov_marker(cov: float, target: float, tol: float = 0.04) -> str:
    if cov < target - tol:
        return "v"
    if cov > target + tol:
        return "^"
    return " "


def _mc_proportion_stats(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    if total <= 0:
        return (float("nan"),) * 4
    p_hat = successes / total
    mcse = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    return float(p_hat), mcse, max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)


def _time_stats(subset: list[SimResult]) -> tuple[float, float]:
    total_reps = sum(r.n_reps for r in subset)
    if total_reps <= 0:
        return float("nan"), float("nan")
    sum_t = sum(r.total_time for r in subset)
    sum_t2 = sum(r.total_time_sq for r in subset)
    avg = sum_t / total_reps
    var = max(0.0, sum_t2 / total_reps - avg * avg)
    return avg * 1000.0, float(np.sqrt(var / total_reps)) * 1000.0


def _print_overall_summary_table(
    title: str,
    eval_types: list[str],
    results: list[SimResult],
    agg: dict[tuple, list[tuple[float, float, float]]],
    agg_counts: dict[tuple, tuple[int, int]],
    target: float,
    sizes_present: list[int],
) -> None:
    """Print one OVERALL SUMMARY table, aggregated only over `eval_types`.

    No-ops (prints nothing) if none of `eval_types` are present in `results`,
    so callers can unconditionally request e.g. a binary-only table even when
    a given run has no binary data. `results` should already be non-null
    (i.e. `is_null=False`) since `agg`/`agg_counts` are built from those.
    """
    present_methods = {r.method for r in results if r.eval_type in eval_types}
    if not present_methods:
        return
    method_labels = [m.name for m in order_present_methods(present_methods)]

    all_cov: dict[str, list[float]] = defaultdict(list)
    all_wid: dict[str, list[float]] = defaultdict(list)
    all_score: dict[str, list[float]] = defaultdict(list)
    all_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    per_n_counts: dict[tuple[str, int], tuple[int, int]] = defaultdict(lambda: (0, 0))
    for (et, m, n), vals in agg.items():
        if et not in eval_types:
            continue
        all_cov[m].extend(v[0] for v in vals)
        all_wid[m].extend(v[1] for v in vals)
        all_score[m].extend(v[2] for v in vals)
        c, t = agg_counts[(et, m, n)]
        c_prev, t_prev = all_counts[m]
        all_counts[m] = (c_prev + c, t_prev + t)
        c_prev_n, t_prev_n = per_n_counts[(m, n)]
        per_n_counts[(m, n)] = (c_prev_n + c, t_prev_n + t)

    n_cols_hdr = "".join(f"  {'n='+str(n):>7}" for n in sizes_present)
    print(f"\n{'-'*72}\n  {title}\n{'-'*72}")
    print(f"\n  {'Method':<20}  {'Cov':>6}  {'Band95':>13}  {'Width':>8}  {'Score':>8}  {'Time(ms)':>14}{n_cols_hdr}")
    for m in method_labels:
        mc = float(np.mean(all_cov[m])) if all_cov[m] else float("nan")
        mw = float(np.mean(all_wid[m])) if all_wid[m] else float("nan")
        ms = float(np.mean(all_score[m])) if all_score[m] else float("nan")
        c_tot, t_tot = all_counts[m]
        _, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        avg_ms, se_ms = _time_stats(
            [r for r in results if r.method == m and r.eval_type in eval_types]
        )
        time_str = f"{avg_ms:.3f}+-{se_ms:.3f}" if np.isfinite(avg_ms) else "-"
        n_cols_vals = ""
        for n in sizes_present:
            c_n, t_n = per_n_counts.get((m, n), (0, 0))
            cov_n = c_n / t_n if t_n > 0 else float("nan")
            n_cols_vals += f"  {cov_n:>5.3f}{_cov_marker(cov_n, target)} " if np.isfinite(cov_n) else f"  {'  -':>7}"
        print(f"  {m:<20}  {mc:>5.3f}{_cov_marker(mc, target)}  {f'{lo:.3f}-{hi:.3f}':>13}  {mw:>8.4f}  {ms:>8.4f}  {time_str:>14}{n_cols_vals}")


def print_report(results: list[SimResult], sample_sizes: list[int], alpha: float, n_reps: int, statistic: str) -> None:
    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    present_methods = {r.method for r in non_null}
    method_labels = [m.name for m in order_present_methods(present_methods)]

    agg: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    agg_counts: dict[tuple, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in non_null:
        cov = r.covered / r.n_reps
        width = r.total_width / r.n_reps
        score = r.total_score / r.n_reps
        agg[(r.eval_type, r.method, r.n)].append((cov, width, score))
        c_prev, t_prev = agg_counts[(r.eval_type, r.method, r.n)]
        agg_counts[(r.eval_type, r.method, r.n)] = (c_prev + r.covered, t_prev + r.n_reps)

    def mean_cov(et, m, n):
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[0] for v in vals])) if vals else float("nan")

    sep = "=" * 72
    print(f"\n{sep}\n  CI_PAIRED COVERAGE -- SIMULATION RESULTS\n"
          f"  Estimand: paired template difference ({statistic})\n"
          f"  Nominal coverage: {target:.0%}   |   reps/cell: {n_reps}\n"
          f"  v = under-covered   ^ = over-conservative\n"
          f"  Score = interval score (width + (2/alpha)*miss-distance; lower is better --\n"
          f"  see evalstats.core.stats_utils.interval_score)\n{sep}")

    for et in eval_types_present:
        print(f"\n  [{et}]")
        hdr = f"    {'Method':<20}" + "".join(f"  n={n:<6}" for n in sample_sizes)
        print(hdr)
        for m in method_labels:
            row = f"    {m:<20}"
            for n in sample_sizes:
                cov = mean_cov(et, m, n)
                row += "  " + (" " * 7 if np.isnan(cov) else f"{cov:.3f}{_cov_marker(cov, target)}".ljust(8))
            print(row)

    # Split into three OVERALL SUMMARY tables -- binary, continuous [0,1], and
    # numeric (likert + grades averaged together) -- since these data types
    # are answered by very different method families and a single pooled
    # table obscures which methods actually perform best for which type.
    sizes_present = sorted({r.n for r in non_null})
    _print_overall_summary_table(
        "OVERALL SUMMARY -- BINARY (averaged across sources)",
        ["binary"], non_null, agg, agg_counts, target, sizes_present,
    )
    _print_overall_summary_table(
        "OVERALL SUMMARY -- CONTINUOUS [0,1] (averaged across sources)",
        ["continuous"], non_null, agg, agg_counts, target, sizes_present,
    )
    _print_overall_summary_table(
        "OVERALL SUMMARY -- NUMERIC: LIKERT + GRADES (averaged across sources)",
        ["likert", "grades"], non_null, agg, agg_counts, target, sizes_present,
    )

    null_results = [r for r in results if r.is_null]
    if null_results:
        print(f"\n{'-'*72}\n  TYPE I ERROR RATE (null scenarios: delta = 0)\n"
              f"  Empirical P(CI excludes 0) -- target = alpha = {alpha:.2f}\n{'-'*72}")
        null_agg: dict[tuple, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for r in null_results:
            c_prev, t_prev = null_agg[(r.eval_type, r.method, r.n)]
            null_agg[(r.eval_type, r.method, r.n)] = (c_prev + r.covered, t_prev + r.n_reps)
        present_null_methods = {r.method for r in null_results}
        null_method_labels = [m for m in method_labels if m in present_null_methods]
        null_eval_types = [et for et in EVAL_TYPES if any(r.eval_type == et for r in null_results)]
        for et in null_eval_types:
            print(f"\n  {et}  (type I error = 1 - null coverage; target ~ {alpha:.2f})")
            hdr = f"    {'Method':<20}" + "".join(f"  n={n:<6}" for n in sample_sizes)
            print(hdr)
            for m in null_method_labels:
                row = f"    {m:<20}"
                for n in sample_sizes:
                    c_tot, t_tot = null_agg.get((et, m, n), (0, 0))
                    if t_tot <= 0:
                        row += "  " + " " * 7
                    else:
                        t1e = 1.0 - c_tot / t_tot
                        marker = "v" if t1e > alpha + 0.04 else ("^" if t1e < alpha - 0.04 else " ")
                        row += f"  {t1e:.3f}{marker}".ljust(9)
                print(row)
    print()


def latex_overall_summary(results: list[SimResult], alpha: float, n_reps: int) -> str:
    """LaTeX booktabs version of print_report's OVERALL SUMMARY block
    (non-null rows only), plus one coverage column per sample size actually
    swept, appended to the right -- the aggregate "Coverage" column
    collapses across n and can hide miscalibration that only shows up at
    small or large sample sizes."""
    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    eval_types_present = {et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)}
    present_methods = {r.method for r in non_null}
    method_labels = [m.name for m in order_present_methods(present_methods)]
    sizes_present = sorted({r.n for r in non_null})

    agg: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    agg_counts: dict[tuple, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in non_null:
        cov = r.covered / r.n_reps
        width = r.total_width / r.n_reps
        score = r.total_score / r.n_reps
        agg[(r.eval_type, r.method, r.n)].append((cov, width, score))
        c_prev, t_prev = agg_counts[(r.eval_type, r.method, r.n)]
        agg_counts[(r.eval_type, r.method, r.n)] = (c_prev + r.covered, t_prev + r.n_reps)

    all_cov: dict[str, list[float]] = defaultdict(list)
    all_wid: dict[str, list[float]] = defaultdict(list)
    all_score: dict[str, list[float]] = defaultdict(list)
    all_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    per_n_counts: dict[tuple[str, int], tuple[int, int]] = defaultdict(lambda: (0, 0))
    method_eval_types: dict[str, set[str]] = defaultdict(set)
    for (et, m, n), vals in agg.items():
        all_cov[m].extend(v[0] for v in vals)
        all_wid[m].extend(v[1] for v in vals)
        all_score[m].extend(v[2] for v in vals)
        method_eval_types[m].add(et)
        c, t = agg_counts[(et, m, n)]
        c_prev, t_prev = all_counts[m]
        all_counts[m] = (c_prev + c, t_prev + t)
        c_prev_n, t_prev_n = per_n_counts[(m, n)]
        per_n_counts[(m, n)] = (c_prev_n + c, t_prev_n + t)

    rows = []
    for m in method_labels:
        mc = float(np.mean(all_cov[m])) if all_cov[m] else float("nan")
        mw = float(np.mean(all_wid[m])) if all_wid[m] else float("nan")
        ms = float(np.mean(all_score[m])) if all_score[m] else float("nan")
        c_tot, t_tot = all_counts[m]
        _, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        avg_ms, se_ms = _time_stats([r for r in non_null if r.method == m])
        time_str = f"${avg_ms:.3f} \\pm {se_ms:.3f}$" if np.isfinite(avg_ms) else "-"
        et_label = eval_type_label(method_eval_types[m], eval_types_present)
        row = [
            escape_latex(m),
            f"{mc:.3f}" if np.isfinite(mc) else "-",
            f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
            f"{mw:.4f}" if np.isfinite(mw) else "-",
            f"{ms:.4f}" if np.isfinite(ms) else "-",
            time_str,
            et_label,
        ]
        for n in sizes_present:
            c_n, t_n = per_n_counts.get((m, n), (0, 0))
            cov_n = c_n / t_n if t_n > 0 else float("nan")
            row.append(f"{cov_n:.3f}" if np.isfinite(cov_n) else "-")
        rows.append(row)

    return booktabs_table(
        caption=(
            f"ci\\_paired: overall CI coverage summary (nominal {target:.0%}, reps/cell={n_reps}). "
            "Score is the interval score (width + $\\frac{2}{\\alpha}\\times$miss-distance; lower is better)."
        ),
        label="tab:ci_paired_overall",
        columns=["Method", "Coverage", "95\\% MC band", "Mean width", "Score", "Time (ms)", "Eval types"]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
    )


def save_results_artifacts(
    *, results: list[SimResult], alpha: float, sample_sizes: list[int], n_reps: int,
    statistic: str, out_dir: str, run_stem: str, latex: bool = False,
) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    csv_path = out_base / f"{run_stem}_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source", "model_a", "model_b", "benchmark_id", "label", "eval_type", "n", "method", "n_reps",
            "covered", "total_width", "coverage", "mean_width", "mcse", "band95_low", "band95_high",
            "avg_time_ms", "se_time_ms", "is_null", "corpus_size", "true_diff", "run_noise_frac", "runs",
        ])
        for r in results:
            coverage = r.covered / r.n_reps
            mean_width = r.total_width / r.n_reps
            _, mcse, lo, hi = _mc_proportion_stats(r.covered, r.n_reps)
            avg_ms, se_ms = _time_stats([r])
            writer.writerow([
                r.source, r.model_a or "", r.model_b or "", r.benchmark_id or "", r.label, r.eval_type, r.n,
                r.method, r.n_reps, r.covered, f"{r.total_width:.8f}", f"{coverage:.8f}", f"{mean_width:.8f}",
                f"{mcse:.8f}", f"{lo:.8f}", f"{hi:.8f}",
                f"{avg_ms:.6f}" if np.isfinite(avg_ms) else "",
                f"{se_ms:.6f}" if np.isfinite(se_ms) else "",
                r.is_null, r.corpus_size if r.corpus_size is not None else "",
                f"{r.true_diff:.8f}" if r.true_diff is not None else "",
                f"{r.run_noise_frac:.6f}", r.runs,
            ])

    summary_path = out_base / f"{run_stem}_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(results, sample_sizes=sample_sizes, alpha=alpha, n_reps=n_reps, statistic=statistic)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_overall_summary(results, alpha=alpha, n_reps=n_reps)
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_coverage_vs_n_plot(*, results: list[SimResult], sample_sizes: list[int], alpha: float, n_reps: int, out_path: str) -> str:
    """Coverage vs. sample size line plots -- one subplot per eval type, all methods overlaid."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    present_methods = {r.method for r in non_null}
    method_objs = order_present_methods(present_methods)
    method_names = [m.name for m in method_objs]

    df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "method": r.method, "n": r.n, "coverage": r.covered / r.n_reps}
        for r in non_null
    ])
    df = df[df["method"].isin(method_names)]
    label_level = df.groupby(["eval_type", "label", "method", "n"], as_index=False).agg(coverage=("coverage", "mean"))
    agg = label_level.groupby(["eval_type", "method", "n"], as_index=False).agg(
        coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"), coverage_count=("coverage", "count"),
    )
    palette = {m.name: m.color for m in method_objs}

    fig, axes = plt.subplots(1, max(len(eval_types_present), 1), figsize=(5.5 * max(len(eval_types_present), 1), 5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        et_agg = agg[agg["eval_type"] == et].copy()
        et_methods = [name for name in method_names if name in et_agg["method"].values]
        if et_agg.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        sns.lineplot(data=et_agg, x="n", y="coverage_mean", hue="method", hue_order=et_methods,
                     palette=palette, marker=None, linewidth=1.0, alpha=0.70, ax=ax)
        for method, sub in et_agg.groupby("method"):
            if sub["coverage_std"].isna().all():
                continue
            sub = sub.sort_values("n")
            color = get_method_color(str(method))
            se = sub["coverage_std"] / np.sqrt(sub["coverage_count"])
            ax.errorbar(sub["n"], sub["coverage_mean"], yerr=se, fmt="none", color=color, elinewidth=0.8, capsize=2, alpha=0.45)
            ax.scatter(sub["n"], sub["coverage_mean"], s=28, color=color, edgecolors="white", linewidths=0.6, alpha=0.85, zorder=3)

        ns = sorted(et_agg["n"].unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.axhline(target, linestyle="--", color="tab:cyan", linewidth=1.2)
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Empirical coverage" if col_idx == 0 else "")
        ax.set_title(et.upper())
        if et_methods:
            ax.legend(title="Method", fontsize=7.5, title_fontsize=8)

    fig.suptitle(f"Coverage vs. Sample Size\nci_paired | reps={n_reps} | alpha={alpha}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_width_vs_n_plot(*, results: list[SimResult], sample_sizes: list[int], alpha: float, n_reps: int, out_path: str) -> str:
    """Mean CI width vs. sample size line plots -- one subplot per eval type, all methods overlaid."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    non_null = [r for r in results if not r.is_null]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    present_methods = {r.method for r in non_null}
    method_objs = order_present_methods(present_methods)
    method_names = [m.name for m in method_objs]

    df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "method": r.method, "n": r.n, "width": r.total_width / r.n_reps}
        for r in non_null
    ])
    df = df[df["method"].isin(method_names)]
    label_level = df.groupby(["eval_type", "label", "method", "n"], as_index=False).agg(width=("width", "mean"))
    agg = label_level.groupby(["eval_type", "method", "n"], as_index=False).agg(
        width_mean=("width", "mean"), width_std=("width", "std"), width_count=("width", "count"),
    )
    palette = {m.name: m.color for m in method_objs}

    fig, axes = plt.subplots(1, max(len(eval_types_present), 1), figsize=(5.5 * max(len(eval_types_present), 1), 5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        et_agg = agg[agg["eval_type"] == et].copy()
        et_methods = [name for name in method_names if name in et_agg["method"].values]
        if et_agg.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        sns.lineplot(data=et_agg, x="n", y="width_mean", hue="method", hue_order=et_methods,
                     palette=palette, marker=None, linewidth=1.0, alpha=0.70, ax=ax)
        for method, sub in et_agg.groupby("method"):
            if sub["width_std"].isna().all():
                continue
            sub = sub.sort_values("n")
            color = get_method_color(str(method))
            se = sub["width_std"] / np.sqrt(sub["width_count"])
            ax.errorbar(sub["n"], sub["width_mean"], yerr=se, fmt="none", color=color, elinewidth=0.8, capsize=2, alpha=0.45)
            ax.scatter(sub["n"], sub["width_mean"], s=28, color=color, edgecolors="white", linewidths=0.6, alpha=0.85, zorder=3)

        ns = sorted(et_agg["n"].unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Mean CI width" if col_idx == 0 else "")
        ax.set_title(et.upper())
        if et_methods:
            ax.legend(title="Method", fontsize=7.5, title_fontsize=8)

    fig.suptitle(f"CI Width vs. Sample Size\nci_paired | reps={n_reps} | alpha={alpha}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_cost_plot(*, results: list[SimResult], alpha: float, n_reps: int, out_path: str) -> str:
    """Scatter plot: x = mean CI time (log ms), y = coverage; one subplot per eval type."""
    import matplotlib.pyplot as plt

    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    present_methods = {r.method for r in non_null}
    method_objs = order_present_methods(present_methods)
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]

    def _label_indices(ns: list) -> set:
        if len(ns) <= 2:
            return set(range(len(ns)))
        return {0, len(ns) // 2, len(ns) - 1}

    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(11.0, 4.5 * nrows), squeeze=False, gridspec_kw={"hspace": 0.45})

    for row_idx, et in enumerate(eval_types_present):
        ax = axes[row_idx][0]
        et_results = [r for r in non_null if r.eval_type == et]
        sample_sizes = sorted({r.n for r in et_results})

        ax.axhspan(max(0.0, target - 0.04), min(1.0, target + 0.04), color="#DDDDDD", alpha=0.40, zorder=0)
        ax.axhline(target, color="black", linewidth=1.1, linestyle="--", zorder=1)

        legend_handles = []
        all_xs: list[float] = []
        for m in method_objs:
            color = m.color
            m_results = [r for r in et_results if r.method == m.name]
            if not m_results:
                continue
            points = []
            for n in sample_sizes:
                subset = [r for r in m_results if r.n == n]
                if not subset:
                    continue
                avg_ms, se_ms = _time_stats(subset)
                if not np.isfinite(avg_ms) or avg_ms <= 0:
                    continue
                cov = float(np.mean([r.covered / r.n_reps for r in subset]))
                points.append((n, avg_ms, cov, 1.96 * se_ms))
            if not points:
                continue

            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            all_xs.extend(xs)
            ax.plot(xs, ys, color=color, linewidth=1.1, alpha=0.55, zorder=2)
            ax.errorbar(xs, ys, xerr=[p[3] for p in points], fmt="o", color=color,
                        markersize=6, markeredgewidth=0.7, markeredgecolor="white",
                        elinewidth=0.9, capsize=2.5, capthick=0.9, alpha=0.90, zorder=3)

            for i, (n, x, y, _) in enumerate(points):
                if i in _label_indices(points):
                    ax.annotate(f"n={n}", xy=(x, y), xytext=(0, 4), textcoords="offset points",
                                fontsize=6.5, ha="center", va="bottom", color=color, alpha=0.85)

            legend_handles.append(plt.Line2D([0], [0], marker="o", color=color, markerfacecolor=color, markersize=7, label=m.name, linewidth=1.5))

        ax.set_xscale("log")
        ax.set_xlabel("Mean CI time (ms) -- log scale", fontsize=9.5)
        ax.set_ylabel("Coverage rate", fontsize=9.5)
        ax.set_title(f"eval type: {et}", fontsize=10.5)
        ax.set_ylim(max(0.0, target - 0.20), min(1.01, target + 0.12))
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
        ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.35)
        ax.tick_params(labelsize=8.5)
        if all_xs:
            import matplotlib.ticker as _ticker
            lo_ms, hi_ms = min(all_xs), max(all_xs)
            n_ticks = max(4, min(8, len(all_xs) // 2))
            tick_locs = np.logspace(np.log10(max(lo_ms, 1e-9)), np.log10(max(hi_ms, 1e-8)), n_ticks)
            ax.xaxis.set_major_locator(_ticker.FixedLocator(tick_locs))
            ax.xaxis.set_major_formatter(_ticker.FuncFormatter(lambda x, _: f"{x:.3g}"))
            ax.xaxis.set_minor_locator(_ticker.NullLocator())

        if not et_results:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        if legend_handles:
            ax.legend(handles=legend_handles, title="Method", loc="center left", bbox_to_anchor=(1.02, 0.5),
                      borderaxespad=0.0, fontsize=7.5, title_fontsize=8, framealpha=0.85, ncol=1)

    fig.suptitle(
        f"Cost x Coverage Trade-off\n"
        f"ci_paired | x = mean CI compute time | y = empirical coverage | target = {target:.0%} | reps={n_reps}",
        fontsize=10.5,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=[0.02, 0.02, 0.80, 0.93])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_coverage_vs_run_noise_plot(*, results: list[SimResult], alpha: float, n_reps: int, out_path: str) -> str | None:
    """Coverage vs. run_noise_frac -- key --nested-mode plot for when nested/flat matters."""
    import matplotlib.pyplot as plt

    target = 1.0 - alpha
    non_null = [r for r in results if not r.is_null]
    present_methods = {r.method for r in non_null}
    method_objs = order_present_methods(present_methods)
    run_noise_fracs = sorted({r.run_noise_frac for r in non_null})
    if len(run_noise_fracs) < 2:
        print(f"Skipped coverage-vs-run-noise plot (only one f_run value): {out_path}")
        return None

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in non_null)]
    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows, 1, figsize=(10.0, 4.0 * nrows), squeeze=False, gridspec_kw={"hspace": 0.45})

    for row_idx, et in enumerate(eval_types_present):
        ax = axes[row_idx][0]
        ax.axhspan(max(0.0, target - 0.04), min(1.0, target + 0.04), color="#DDDDDD", alpha=0.40, zorder=0)
        ax.axhline(target, color="black", linewidth=1.2, linestyle="--", zorder=1)

        for m in method_objs:
            covs = []
            for f in run_noise_fracs:
                subset = [r for r in non_null if r.eval_type == et and r.method == m.name and r.run_noise_frac == f]
                if not subset:
                    covs.append(float("nan"))
                    continue
                c_tot = sum(r.covered for r in subset)
                t_tot = sum(r.n_reps for r in subset)
                covs.append(c_tot / t_tot if t_tot > 0 else float("nan"))

            xs = [f for f, c in zip(run_noise_fracs, covs) if not np.isnan(c)]
            ys = [c for c in covs if not np.isnan(c)]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", color=m.color, linewidth=1.4, label=m.name, markersize=5, alpha=0.85)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(max(0.0, target - 0.25), min(1.01, target + 0.12))
        ax.set_xlabel("Run noise fraction  f_run = var_run / (var_input + var_run)", fontsize=9.5)
        ax.set_ylabel("Empirical coverage", fontsize=9.5)
        ax.set_title(f"eval type: {et}", fontsize=10.5)
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
        ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.35)
        ax.legend(fontsize=7.5, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.85)

    runs_val = non_null[0].runs if non_null else "?"
    fig.suptitle(
        f"Coverage vs. Run Noise Fraction\n"
        f"ci_paired (nested mode) | runs={runs_val} | alpha={alpha} | reps={n_reps}\n"
        f"Averaged across all shapes and sample sizes",
        fontsize=11,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=[0.02, 0.02, 0.80, 0.93])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="synthetic",
                         help="'synthetic' (default), or a real-data source: " + ", ".join(REAL_PAIR_SOURCES))
    parser.add_argument("--scenario-suite", choices=SCENARIO_SUITES, default="expanded",
                         help="Synthetic scenario breadth (ignored for real data sources)")
    parser.add_argument("--eval-types", nargs="+", choices=EVAL_TYPES, default=None, metavar="TYPE")
    parser.add_argument("--benchmarks", nargs="+", default=None, metavar="ID", help="Real-data: benchmark IDs to filter to")
    parser.add_argument("--models", nargs="+", default=None, metavar="NAME", help="Real-data: model names to filter to")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--min-pair-size", type=int, default=50)
    parser.add_argument("--inspect-csv", default=None,
                         help=f"Path to CSV from collect_inspect_benchmarks.py "
                              f"(used by --data-source inspect/real; defaults to {DEFAULT_INSPECT_CSV!r})")
    parser.add_argument("--runs", type=int, default=1, metavar="R", help="Runs per input. R>=3 activates nested bootstrap logic (synthetic only; default: 1)")
    parser.add_argument("--statistic", choices=["mean", "median"], default="mean")
    parser.add_argument("--reps", type=int, default=200, metavar="N")
    parser.add_argument("--bootstrap-n", type=int, default=500, metavar="N")
    parser.add_argument("--bayes-n", type=int, default=2000, metavar="N")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 50], metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--icc-values", type=float, nargs="+", default=None, metavar="ICC",
                         help="Flat mode: ICC sweep for build_pair_sources (default: 0.05 0.20 0.40 0.60 0.80). "
                              "Nested mode: optional additional ICC sweep, converted via f_run = 1 - ICC.")
    parser.add_argument("--cohens-d-values", type=float, nargs="+", default=[0.3], metavar="D")
    parser.add_argument("--include-null", action="store_true", default=False)
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--latex", action="store_true", default=False,
                         help="Append a LaTeX booktabs overall-summary table to the saved summary .log file.")
    parser.add_argument("--nested-mode", action="store_true", default=False,
                         help="Multi-run flat-vs-nested pairwise CI comparison (ported from "
                              "sim_compare_boot_nested.py), synthetic only, statistic=mean only.")
    parser.add_argument("--runs-sweep", type=int, nargs="+", default=None, metavar="R",
                         help="Nested mode: sweep multiple R values, overrides --runs")
    parser.add_argument("--run-noise-fracs", type=float, nargs="+", default=RUN_NOISE_FRACS_DEFAULT, metavar="F",
                         help="Nested mode: f_run = var_run / (var_input + var_run) values to sweep "
                              "(--icc-values is also accepted in nested mode, converted via f_run = 1 - ICC)")
    parser.add_argument("--heteroscedastic", action="store_true", default=False,
                         help="Nested mode: run noise scales with input value (mimics real LLM eval variability)")
    parser.add_argument("--pairwise-noise-grid", action="store_true", default=False,
                         help="Nested mode: use full Cartesian grid of run-noise fractions across models "
                              "(all f_A, f_B combinations) instead of matched f_A=f_B pairs")
    parser.add_argument("--pairwise-noise-grid-max", type=int, default=None, metavar="K",
                         help="Nested mode: optional cap on number of (f_A, f_B) combinations")
    parser.add_argument("--pairwise-noise-grid-seed", type=int, default=42, metavar="N")
    parser.add_argument("--cross-item-rho", type=float, default=0.7, metavar="RHO",
                         help="Nested mode: Gaussian-copula correlation between A's and B's item-level latent scores")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N",
                         help="Parallel worker processes (default: cpu_count-1; 1=sequential).")


def official_args(base_seed: int = 42) -> argparse.Namespace:
    """Canonical official-test preset, mirroring sim_compare_boot.py --official-test
    (pairwise phase). Synthetic only -- real-data sources need network/HF access
    not assumed available; run --data-source openeval/inspect/real manually."""
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="expanded", eval_types=None,
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        runs=1, statistic="mean", reps=2000, bootstrap_n=10000, bayes_n=10000, alpha=0.05,
        sizes=[10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        seed=base_seed, icc_values=[0.05, 0.20, 0.40, 0.60, 0.80], cohens_d_values=[0.2, 0.4], include_null=True,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        nested_mode=False, runs_sweep=None, run_noise_fracs=RUN_NOISE_FRACS_DEFAULT, heteroscedastic=False,
        pairwise_noise_grid=False, pairwise_noise_grid_max=None, pairwise_noise_grid_seed=42, cross_item_rho=0.7,
        workers=max(1, (os.cpu_count() or 2) - 1),
    )


def quick_args(base_seed: int = 43, data_source: str = "synthetic") -> argparse.Namespace:
    """Fast sanity-check preset for --quick-test: same shape catalog as
    official_args() but with reps/sizes/bootstrap_n cut down for a quick pass
    that confirms the pipeline (incl. --latex output) still works.
    ``data_source="real"`` (or 'openeval'/'inspect') swaps in
    build_real_pair_sources() instead of the synthetic shape catalog --
    --quick-test calls this twice per case (synthetic, then real) so the
    real-data path doesn't go unexercised between --official-tests runs."""
    return argparse.Namespace(
        data_source=data_source, scenario_suite="standard", eval_types=None,
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        runs=1, statistic="mean", reps=3, bootstrap_n=200, bayes_n=200, alpha=0.05,
        sizes=[10, 30, 50],
        seed=base_seed, icc_values=[0.20], cohens_d_values=[0.3], include_null=True,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        nested_mode=False, runs_sweep=None, run_noise_fracs=RUN_NOISE_FRACS_DEFAULT, heteroscedastic=False,
        pairwise_noise_grid=False, pairwise_noise_grid_max=None, pairwise_noise_grid_seed=42, cross_item_rho=0.7,
        latex=True, workers=1,
    )


def nested_official_args(base_seed: int = 44) -> argparse.Namespace:
    """Canonical --nested-mode official preset, mirroring sim_compare_boot_nested.py's
    --official-test (pairwise-estimand phase). Not wired into --official-tests (the
    harness runs one preset per case); invoke manually:
    python -m simulations.harness.cli ci_paired --nested-mode --runs-sweep 5 --reps 200
      --bootstrap-n 10000 --bayes-n 10000 --scenario-suite expanded --icc-values 0.05 0.30 0.50
      --run-noise-fracs 0.01 0.1 0.3 0.5 --cohens-d-values 0.2 0.4 --include-null --heteroscedastic
      --sizes 10 20 30 50 75 100 --seed 44
    """
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="expanded", eval_types=None,
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        runs=5, statistic="mean", reps=200, bootstrap_n=10000, bayes_n=10000, alpha=0.05,
        sizes=[10, 20, 30, 50, 75, 100],
        seed=base_seed, icc_values=[0.05, 0.30, 0.50], cohens_d_values=[0.2, 0.4], include_null=True,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        nested_mode=True, runs_sweep=[5], run_noise_fracs=[0.01, 0.1, 0.3, 0.5], heteroscedastic=True,
        pairwise_noise_grid=False, pairwise_noise_grid_max=None, pairwise_noise_grid_seed=42, cross_item_rho=0.7,
        workers=max(1, (os.cpu_count() or 2) - 1),
    )


def run(args: argparse.Namespace) -> CaseResult:
    t0 = time.time()
    try:
        plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")
        nested_mode = getattr(args, "nested_mode", False)

        if nested_mode:
            if args.data_source != "synthetic":
                raise ValueError("--nested-mode only supports --data-source synthetic.")

            run_noise_fracs = list(args.run_noise_fracs)
            if args.icc_values:
                icc_as_run_noise = [float(np.clip(1.0 - icc, 0.0, 1.0)) for icc in args.icc_values]
                run_noise_fracs = sorted(set(run_noise_fracs + icc_as_run_noise))
            bayes_n = args.bayes_n if args.bayes_n is not None else args.bootstrap_n
            runs_list = args.runs_sweep if args.runs_sweep else [args.runs]

            print(f"\nci_paired simulation (nested mode) -- runs={runs_list}, run_noise_fracs={run_noise_fracs}")
            sources = build_pair_sources(
                suite=args.scenario_suite, cohens_d_values=args.cohens_d_values,
                include_null=args.include_null, run_noise_fracs=run_noise_fracs, heteroscedastic=args.heteroscedastic,
                pairwise_noise_grid=args.pairwise_noise_grid, pairwise_noise_grid_max=args.pairwise_noise_grid_max,
                pairwise_noise_grid_seed=args.pairwise_noise_grid_seed, cross_item_rho=args.cross_item_rho,
            )
            if args.eval_types:
                requested = set(args.eval_types)
                sources = [s for s in sources if s.eval_type in requested]
            if not sources:
                raise ValueError("No CIPairSources left after filtering.")
            print(f"  {len(sources)} sources, sizes={args.sizes}, reps={args.reps}, alpha={args.alpha}")

            results: list[SimResult] = []
            n_workers = getattr(args, "workers", 1)
            for r_val in runs_list:
                results.extend(run_nested_pairwise_simulation(
                    sources, sample_sizes=args.sizes, runs=r_val, n_reps=args.reps, n_bootstrap=args.bootstrap_n,
                    bayes_n=bayes_n, alpha=args.alpha, progress_mode=args.progress, seed=args.seed,
                    n_workers=n_workers,
                ))
            print_report(results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps, statistic="mean")

            stamp = time.strftime("%Y%m%d_%H%M%S")
            run_stem = f"ci_paired_nested_runs{'-'.join(str(r) for r in runs_list)}_reps{args.reps}_{stamp}"
            output_paths: list[str] = []

            if args.save_results == "save":
                output_paths += save_results_artifacts(
                    results=results, alpha=args.alpha, sample_sizes=args.sizes, n_reps=args.reps,
                    statistic="mean", out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False),
                )

            if args.plots == "save":
                cov_path = save_coverage_vs_n_plot(
                    results=results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps,
                    out_path=str(Path(plots_dir) / f"{run_stem}_coverage_vs_n.png"),
                )
                width_path = save_width_vs_n_plot(
                    results=results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps,
                    out_path=str(Path(plots_dir) / f"{run_stem}_width_vs_n.png"),
                )
                cost_path = save_cost_plot(
                    results=results, alpha=args.alpha, n_reps=args.reps,
                    out_path=str(Path(plots_dir) / f"{run_stem}_cost_coverage.png"),
                )
                output_paths += [cov_path, width_path, cost_path]
                run_noise_path = save_coverage_vs_run_noise_plot(
                    results=results, alpha=args.alpha, n_reps=args.reps,
                    out_path=str(Path(plots_dir) / f"{run_stem}_coverage_vs_run_noise.png"),
                )
                if run_noise_path:
                    output_paths.append(run_noise_path)
                print(f"Saved plots: {output_paths[-4:] if run_noise_path else output_paths[-3:]}")

            non_null = [r for r in results if not r.is_null]
            overall_cov = float(np.mean([r.covered / r.n_reps for r in non_null])) if non_null else float("nan")
            return CaseResult(
                case_name=CASE_NAME, status="ok", output_paths=output_paths,
                key_metrics={"n_results": len(results), "overall_mean_coverage": overall_cov},
                duration_s=time.time() - t0,
            )

        print(f"\nci_paired simulation -- data_source={args.data_source}, statistic={args.statistic}")
        if args.data_source == "synthetic":
            icc_values = args.icc_values if args.icc_values is not None else [0.05, 0.20, 0.40, 0.60, 0.80]
            sources = build_pair_sources(
                suite=args.scenario_suite, icc_values=icc_values,
                cohens_d_values=args.cohens_d_values, include_null=args.include_null,
            )
        else:
            runs = args.runs
            if runs != 1:
                print("  Warning: real-data sources only support --runs 1 in this pass; forcing runs=1.")
                runs = 1
            args = argparse.Namespace(**{**vars(args), "runs": runs})
            sources = build_real_pair_sources(
                args.data_source, benchmarks=args.benchmarks, models=args.models,
                hf_token=args.hf_token, cache_dir=args.cache_dir, min_pair_size=args.min_pair_size,
                inspect_csv=args.inspect_csv, include_null=args.include_null,
            )

        if args.eval_types:
            requested = set(args.eval_types)
            sources = [s for s in sources if s.eval_type in requested]
        if not sources:
            raise ValueError("No CIPairSources left after filtering.")

        print(f"  {len(sources)} sources, sizes={args.sizes}, reps={args.reps}, alpha={args.alpha}, runs={args.runs}")

        results = run_simulation(
            sources, sample_sizes=args.sizes, n_reps=args.reps, n_bootstrap=args.bootstrap_n,
            bayes_n=args.bayes_n, alpha=args.alpha, runs=args.runs, statistic=args.statistic,
            progress_mode=args.progress, seed=args.seed, n_workers=getattr(args, "workers", 1),
        )
        print_report(results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps, statistic=args.statistic)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_stem = f"ci_paired_{args.data_source}_stat{args.statistic}_reps{args.reps}_{stamp}"
        output_paths: list[str] = []

        if args.save_results == "save":
            output_paths += save_results_artifacts(
                results=results, alpha=args.alpha, sample_sizes=args.sizes, n_reps=args.reps,
                statistic=args.statistic, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False),
            )

        if args.plots == "save":
            cov_path = save_coverage_vs_n_plot(
                results=results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps,
                out_path=str(Path(plots_dir) / f"{run_stem}_coverage_vs_n.png"),
            )
            width_path = save_width_vs_n_plot(
                results=results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps,
                out_path=str(Path(plots_dir) / f"{run_stem}_width_vs_n.png"),
            )
            cost_path = save_cost_plot(
                results=results, alpha=args.alpha, n_reps=args.reps,
                out_path=str(Path(plots_dir) / f"{run_stem}_cost_coverage.png"),
            )
            output_paths += [cov_path, width_path, cost_path]
            print(f"Saved plots: {cov_path}, {width_path}, {cost_path}")

        non_null = [r for r in results if not r.is_null]
        overall_cov = float(np.mean([r.covered / r.n_reps for r in non_null])) if non_null else float("nan")
        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics={"n_results": len(results), "overall_mean_coverage": overall_cov},
            duration_s=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return CaseResult(case_name=CASE_NAME, status="error", error=str(exc), duration_s=time.time() - t0)
