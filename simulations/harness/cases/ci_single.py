"""ci_single case: single-sample CI coverage, synthetic and/or real benchmark data.

Consolidates ``simulations/sim_compare_boot.py``'s single-sample ("mean"
estimand) simulation loop and ``simulations/sim_dove.py``'s real-data loop
into one engine operating over ``scenarios.CISource`` objects, so the same
CI-method dispatch and report format covers synthetic distributions
(``scenarios/synthetic.py``) and real DOVE_Lite/OpenEval corpora
(``scenarios/real_data.py``) alike.

Methods compared
-----------------
bootstrap, bca, bayes_bootstrap, smooth_bootstrap, bootstrap_t, t_interval
(all eval types); wilson, jeffreys, wald, clopper_pearson, bayes_indep
(binary only); beta, logit_t, nig, el (continuous/likert/grades only).

Known exception: only this case (and ci_paired) has a real-data path.
alignment/ppi_calibration remain synthetic-only this pass -- see
simulations/harness/README.md.

--nested-mode (ported from simulations/sim_compare_boot_nested.py) sweeps
multi-run scenarios parameterised by run_noise_frac instead, reporting
"flat" (cell-mean reduction) and "*_nested" (full N-by-R matrix) CI methods
side by side so the cost of ignoring run structure is visible. There is no
separate ci_nested case -- single- and paired-sample nested logic lives
here and in cases/ci_paired.py respectively.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.resampling import (
        bootstrap_ci_1d,
        wald_ci_1d,
        clopper_pearson_ci_1d,
        jeffreys_ci_1d,
        t_interval_ci_1d,
        beta_ci_1d,
        logit_t_ci_1d,
        nig_ci_1d,
        el_ci_1d,
        bca_interval_1d,
        bootstrap_means_nested,
        bayes_bootstrap_means_nested,
        smooth_bootstrap_means_nested,
        bootstrap_t_ci_nested,
        wilson_nested_de,
        wilson_nested_od,
        wilson_nested_bb,
        nig_ci_nested,
    )

from ..scenarios import CISource, EVAL_TYPES
from ..scenarios.synthetic import (
    SCENARIO_SUITES,
    RUN_NOISE_FRACS_DEFAULT,
    build_single_sample_sources,
)
from ..scenarios.real_data import SOURCES as REAL_DATA_SOURCES, build_real_data_sources
from ..methods import (
    BOOTSTRAP_METHODS as METHODS,
    T_INTERVAL,
    WILSON,
    JEFFREYS,
    WALD,
    CLOPPER_PEARSON,
    BAYES_SINGLE,
    BETA,
    LOGIT_T,
    NIG,
    EL,
    BINARY_SINGLE_EXTRA_METHODS,
    CONTINUOUS_EXTRA_METHODS,
    BOOTSTRAP,
    BAYES_BOOTSTRAP,
    NESTED_METHODS,
    BOOTSTRAP_NESTED,
    BAYES_NESTED,
    SMOOTH_NESTED,
    BCA_NESTED,
    BOOTSTRAP_T_NESTED,
    BINARY_FLAT_METHODS,
    WILSON_FLAT,
    WALD_FLAT,
    CP_FLAT,
    BAYES_INDEP_FLAT,
    BINARY_NESTED_METHODS,
    WILSON_DE,
    WILSON_OD,
    BETA_BINOMIAL,
    CONTINUOUS_NESTED_METHODS,
    NIG_NESTED,
    get_method_color,
    order_present_methods,
)
from . import CaseResult

CASE_NAME = "ci_single"

DATA_SOURCES = ["synthetic"] + REAL_DATA_SOURCES
PROGRESS_MODES = ["bar", "cell", "off"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]


@dataclass
class SimResult:
    source: str  # "synthetic" | "dove" | "openeval"
    label: str  # scenario label (synthetic) or "model/benchmark_id" (real)
    eval_type: str
    n: int
    method: str
    n_reps: int
    covered: int
    total_width: float
    total_time: float = 0.0
    total_time_sq: float = 0.0
    model: str | None = None
    benchmark_id: str | None = None
    corpus_size: int | None = None
    corpus_mean: float | None = None
    run_noise_frac: float = 0.0
    """f_run for --nested-mode rows (scenarios.synthetic.build_single_sample_sources' run_noise_fracs); 0.0 otherwise."""
    runs: int = 1
    """Runs per input for --nested-mode rows; 1 (flat) otherwise."""


def _wilson_ci(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _bayes_indep_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Beta(1,1) posterior credible interval for a Bernoulli proportion."""
    n = int(values.shape[0])
    s = int(np.sum(values >= 0.5))
    lo, hi = stats.beta(s + 1, n - s + 1).interval(1.0 - alpha)
    return float(lo), float(hi)


def _run_cell(source_obj: CISource, n: int, n_reps: int, n_bootstrap: int, alpha: float, seed) -> list[SimResult]:
    """Run all reps for one (source, n) cell."""
    rng = np.random.default_rng(seed)
    is_binary = source_obj.eval_type == "binary"

    active_methods = METHODS + [T_INTERVAL]
    if is_binary:
        active_methods += BINARY_SINGLE_EXTRA_METHODS
    else:
        active_methods += CONTINUOUS_EXTRA_METHODS

    covered: dict = {m: 0 for m in active_methods}
    total_w: dict = {m: 0.0 for m in active_methods}
    total_t: dict = {m: 0.0 for m in active_methods}
    total_t_sq: dict = {m: 0.0 for m in active_methods}
    true_mean = source_obj.true_mean

    for _rep in range(n_reps):
        values = source_obj.generate(rng, n)
        obs_mean = float(np.mean(values))

        for method in METHODS:
            _t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ci_low, ci_high = bootstrap_ci_1d(
                        values, obs_mean, method=method.name, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng,
                    )
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[method] += _el
            total_t_sq[method] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[method] += 1
            total_w[method] += ci_high - ci_low

        _t0 = time.perf_counter()
        try:
            ci_low, ci_high = t_interval_ci_1d(values, alpha)
        except Exception:
            ci_low = ci_high = obs_mean
        _el = time.perf_counter() - _t0
        total_t[T_INTERVAL] += _el
        total_t_sq[T_INTERVAL] += _el * _el
        if ci_low <= true_mean <= ci_high:
            covered[T_INTERVAL] += 1
        total_w[T_INTERVAL] += ci_high - ci_low

        if not is_binary:
            for _method, _fn in zip(CONTINUOUS_EXTRA_METHODS, (beta_ci_1d, logit_t_ci_1d, nig_ci_1d, el_ci_1d)):
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = _fn(values, alpha)
                except Exception:
                    ci_low = ci_high = obs_mean
                _el = time.perf_counter() - _t0
                total_t[_method] += _el
                total_t_sq[_method] += _el * _el
                if ci_low <= true_mean <= ci_high:
                    covered[_method] += 1
                total_w[_method] += ci_high - ci_low

        if is_binary:
            successes = int(np.sum(values >= 0.5))

            _t0 = time.perf_counter()
            ci_low, ci_high = _wilson_ci(successes, n, alpha)
            _el = time.perf_counter() - _t0
            total_t[WILSON] += _el
            total_t_sq[WILSON] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[WILSON] += 1
            total_w[WILSON] += ci_high - ci_low

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = jeffreys_ci_1d(values, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[JEFFREYS] += _el
            total_t_sq[JEFFREYS] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[JEFFREYS] += 1
            total_w[JEFFREYS] += ci_high - ci_low

            _t0 = time.perf_counter()
            ci_low, ci_high = wald_ci_1d(values, alpha)
            _el = time.perf_counter() - _t0
            total_t[WALD] += _el
            total_t_sq[WALD] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[WALD] += 1
            total_w[WALD] += ci_high - ci_low

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = clopper_pearson_ci_1d(values, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[CLOPPER_PEARSON] += _el
            total_t_sq[CLOPPER_PEARSON] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[CLOPPER_PEARSON] += 1
            total_w[CLOPPER_PEARSON] += ci_high - ci_low

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_indep_ci(values, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[BAYES_SINGLE] += _el
            total_t_sq[BAYES_SINGLE] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[BAYES_SINGLE] += 1
            total_w[BAYES_SINGLE] += ci_high - ci_low

    return [
        SimResult(
            source=source_obj.source, label=source_obj.label, eval_type=source_obj.eval_type,
            n=n, method=method.name, n_reps=n_reps, covered=covered[method],
            total_width=total_w[method], total_time=total_t[method], total_time_sq=total_t_sq[method],
            model=source_obj.model, benchmark_id=source_obj.benchmark_id,
            corpus_size=source_obj.max_n, corpus_mean=(source_obj.true_mean if source_obj.source != "synthetic" else None),
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
        bar = "#" * filled + "." * (28 - filled)
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


def run_simulation(
    sources: list[CISource],
    sample_sizes: list[int],
    n_reps: int,
    n_bootstrap: int,
    alpha: float,
    progress_mode: str = "bar",
    seed: int = 42,
) -> list[SimResult]:
    ss = np.random.SeedSequence(seed)
    cells = [(s, n) for s in sources for n in sample_sizes if s.max_n is None or n < s.max_n]
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]

    skipped = [(s, n) for s in sources for n in sample_sizes if not (s.max_n is None or n < s.max_n)]
    for s, n in skipped:
        print(f"  Warning: sample size n={n} >= corpus size {s.max_n} for {s.label}. Skipping.")

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="ci_single")
    results: list[SimResult] = []
    for i, (source_obj, n) in enumerate(cells):
        results.extend(_run_cell(source_obj, n, n_reps, n_bootstrap, alpha, child_seeds[i]))
        reporter.update(i + 1, detail=f"{source_obj.eval_type} {source_obj.label} n={n}")
    reporter.update(len(cells), detail="done")
    return results


# ---------------------------------------------------------------------------
# Nested mode (--nested-mode): flat vs. nested CI methods on multi-run data,
# ported from sim_compare_boot_nested.py's _run_multirun_cell.
# ---------------------------------------------------------------------------

_CONT_NESTED_FNS = {BETA: beta_ci_1d, LOGIT_T: logit_t_ci_1d, NIG: nig_ci_1d, EL: el_ci_1d}


def _run_nested_cell(
    source_obj: CISource, n: int, runs: int, n_reps: int, n_bootstrap: int, bayes_n: int, alpha: float, seed,
) -> list[SimResult]:
    """Run all reps for one (source, n, runs) cell -- flat-vs-nested single-sample mean estimand."""
    rng = np.random.default_rng(seed)
    is_binary = source_obj.eval_type == "binary"
    is_continuous = source_obj.eval_type == "continuous"
    true_mean = source_obj.true_mean

    active_methods = list(METHODS) + [T_INTERVAL] + NESTED_METHODS
    if is_binary:
        active_methods += BINARY_FLAT_METHODS + BINARY_NESTED_METHODS + CONTINUOUS_NESTED_METHODS + [NIG]
    elif is_continuous:
        active_methods += CONTINUOUS_NESTED_METHODS + CONTINUOUS_EXTRA_METHODS

    covered: dict = {m: 0 for m in active_methods}
    total_w: dict = {m: 0.0 for m in active_methods}
    total_t: dict = {m: 0.0 for m in active_methods}
    total_t_sq: dict = {m: 0.0 for m in active_methods}

    for _rep in range(n_reps):
        scores = source_obj.generate_runs(rng, n, runs)  # (n, runs)
        cell_means = scores.mean(axis=1)
        obs_mean = float(np.mean(cell_means))

        # -- Cell-means bootstrap family (flat) --
        for method in METHODS:
            n_draws = bayes_n if method is BAYES_BOOTSTRAP else n_bootstrap
            _t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ci_low, ci_high = bootstrap_ci_1d(cell_means, obs_mean, method=method.name, n_bootstrap=n_draws, alpha=alpha, rng=rng)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[method] += _el
            total_t_sq[method] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[method] += 1
            total_w[method] += ci_high - ci_low

        # -- t-interval on cell means --
        _t0 = time.perf_counter()
        try:
            ci_low, ci_high = t_interval_ci_1d(cell_means, alpha)
        except Exception:
            ci_low = ci_high = obs_mean
        _el = time.perf_counter() - _t0
        total_t[T_INTERVAL] += _el
        total_t_sq[T_INTERVAL] += _el * _el
        if ci_low <= true_mean <= ci_high:
            covered[T_INTERVAL] += 1
        total_w[T_INTERVAL] += ci_high - ci_low

        # -- Nested bootstrap (full N x R matrix) --
        for method, fn in [
            (BOOTSTRAP_NESTED, bootstrap_means_nested),
            (BAYES_NESTED, bayes_bootstrap_means_nested),
            (SMOOTH_NESTED, smooth_bootstrap_means_nested),
        ]:
            n_draws = bayes_n if method is BAYES_NESTED else n_bootstrap
            _t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*falling back to plain bootstrap; no KDE smoothing applied.*",
                        category=UserWarning,
                    )
                    boot_stats = fn(scores, n_draws, rng)
                ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
                ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[method] += _el
            total_t_sq[method] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[method] += 1
            total_w[method] += ci_high - ci_low

        # -- BCa interval using nested bootstrap replicates --
        _t0 = time.perf_counter()
        try:
            boot_stats = bootstrap_means_nested(scores, n_bootstrap, rng)
            ci_low, ci_high = bca_interval_1d(cell_means, obs_mean, boot_stats, alpha)
        except Exception:
            ci_low = ci_high = obs_mean
        _el = time.perf_counter() - _t0
        total_t[BCA_NESTED] += _el
        total_t_sq[BCA_NESTED] += _el * _el
        if ci_low <= true_mean <= ci_high:
            covered[BCA_NESTED] += 1
        total_w[BCA_NESTED] += ci_high - ci_low

        # -- Bootstrap-t via nested resampling --
        _t0 = time.perf_counter()
        try:
            ci_low, ci_high = bootstrap_t_ci_nested(scores, obs_mean, n_bootstrap, alpha, rng)
        except Exception:
            ci_low = ci_high = obs_mean
        _el = time.perf_counter() - _t0
        total_t[BOOTSTRAP_T_NESTED] += _el
        total_t_sq[BOOTSTRAP_T_NESTED] += _el * _el
        if ci_low <= true_mean <= ci_high:
            covered[BOOTSTRAP_T_NESTED] += 1
        total_w[BOOTSTRAP_T_NESTED] += ci_high - ci_low

        # -- Binary flat methods on cell means --
        if is_binary:
            succ_flat = int(np.round(np.sum(cell_means)))
            n_flat = len(cell_means)

            _t0 = time.perf_counter()
            ci_low, ci_high = _wilson_ci(succ_flat, n_flat, alpha)
            _el = time.perf_counter() - _t0
            total_t[WILSON_FLAT] += _el
            total_t_sq[WILSON_FLAT] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[WILSON_FLAT] += 1
            total_w[WILSON_FLAT] += ci_high - ci_low

            _t0 = time.perf_counter()
            ci_low, ci_high = wald_ci_1d(cell_means, alpha)
            _el = time.perf_counter() - _t0
            total_t[WALD_FLAT] += _el
            total_t_sq[WALD_FLAT] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[WALD_FLAT] += 1
            total_w[WALD_FLAT] += ci_high - ci_low

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = clopper_pearson_ci_1d(cell_means, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[CP_FLAT] += _el
            total_t_sq[CP_FLAT] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[CP_FLAT] += 1
            total_w[CP_FLAT] += ci_high - ci_low

            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = _bayes_indep_ci(cell_means, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[BAYES_INDEP_FLAT] += _el
            total_t_sq[BAYES_INDEP_FLAT] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[BAYES_INDEP_FLAT] += 1
            total_w[BAYES_INDEP_FLAT] += ci_high - ci_low

            # -- Clustered Wilson variants (nested, multi-run) --
            for method, fn in [(WILSON_DE, wilson_nested_de), (WILSON_OD, wilson_nested_od), (BETA_BINOMIAL, wilson_nested_bb)]:
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = fn(scores, alpha)
                except Exception:
                    ci_low = ci_high = obs_mean
                _el = time.perf_counter() - _t0
                total_t[method] += _el
                total_t_sq[method] += _el * _el
                if ci_low <= true_mean <= ci_high:
                    covered[method] += 1
                total_w[method] += ci_high - ci_low

        # -- NIG nested on full matrix (continuous + binary approximation) --
        if is_binary or is_continuous:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = nig_ci_nested(scores, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[NIG_NESTED] += _el
            total_t_sq[NIG_NESTED] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[NIG_NESTED] += 1
            total_w[NIG_NESTED] += ci_high - ci_low

        # -- NIG on cell means (binary approximation) --
        if is_binary:
            _t0 = time.perf_counter()
            try:
                ci_low, ci_high = nig_ci_1d(cell_means, alpha)
            except Exception:
                ci_low = ci_high = obs_mean
            _el = time.perf_counter() - _t0
            total_t[NIG] += _el
            total_t_sq[NIG] += _el * _el
            if ci_low <= true_mean <= ci_high:
                covered[NIG] += 1
            total_w[NIG] += ci_high - ci_low

        # -- Continuous extra methods on cell means --
        if is_continuous:
            for _method, _fn in _CONT_NESTED_FNS.items():
                _t0 = time.perf_counter()
                try:
                    ci_low, ci_high = _fn(cell_means, alpha)
                except Exception:
                    ci_low = ci_high = obs_mean
                _el = time.perf_counter() - _t0
                total_t[_method] += _el
                total_t_sq[_method] += _el * _el
                if ci_low <= true_mean <= ci_high:
                    covered[_method] += 1
                total_w[_method] += ci_high - ci_low

    return [
        SimResult(
            source="synthetic", label=source_obj.label, eval_type=source_obj.eval_type,
            n=n, method=method.name, n_reps=n_reps, covered=covered[method],
            total_width=total_w[method], total_time=total_t[method], total_time_sq=total_t_sq[method],
            run_noise_frac=source_obj.run_noise_frac or 0.0, runs=runs,
        )
        for method in active_methods
    ]


def run_nested_simulation(
    sources: list[CISource], sample_sizes: list[int], runs: int, n_reps: int, n_bootstrap: int,
    bayes_n: int, alpha: float, progress_mode: str = "bar", seed: int = 42,
) -> list[SimResult]:
    ss = np.random.SeedSequence(seed)
    cells = [(s, n) for s in sources for n in sample_sizes]
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label=f"ci_single-nested[runs={runs}]")
    results: list[SimResult] = []
    for i, (source_obj, n) in enumerate(cells):
        results.extend(_run_nested_cell(source_obj, n, runs, n_reps, n_bootstrap, bayes_n, alpha, child_seeds[i]))
        reporter.update(i + 1, detail=f"{source_obj.eval_type} {source_obj.label} n={n}")
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


def print_report(results: list[SimResult], sample_sizes: list[int], alpha: float, n_reps: int) -> None:
    target = 1.0 - alpha
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    present_methods = {r.method for r in results}
    method_labels = [m.name for m in order_present_methods(present_methods)]

    agg: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    agg_counts: dict[tuple, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in results:
        cov = r.covered / r.n_reps
        width = r.total_width / r.n_reps
        agg[(r.eval_type, r.method, r.n)].append((cov, width))
        c_prev, t_prev = agg_counts[(r.eval_type, r.method, r.n)]
        agg_counts[(r.eval_type, r.method, r.n)] = (c_prev + r.covered, t_prev + r.n_reps)

    def mean_cov(et, m, n):
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[0] for v in vals])) if vals else float("nan")

    def mean_wid(et, m, n):
        vals = agg.get((et, m, n), [])
        return float(np.mean([v[1] for v in vals])) if vals else float("nan")

    sep = "=" * 72
    print(f"\n{sep}\n  CI_SINGLE COVERAGE -- SIMULATION RESULTS\n"
          f"  Nominal coverage: {target:.0%}   |   reps/cell: {n_reps}\n"
          f"  v = under-covered   ^ = over-conservative\n{sep}")

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

    print(f"\n{'-'*72}\n  OVERALL SUMMARY (averaged across eval types, sources, n)\n{'-'*72}")
    all_cov: dict[str, list[float]] = defaultdict(list)
    all_wid: dict[str, list[float]] = defaultdict(list)
    all_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for (et, m, n), vals in agg.items():
        all_cov[m].extend(v[0] for v in vals)
        all_wid[m].extend(v[1] for v in vals)
        c, t = agg_counts[(et, m, n)]
        c_prev, t_prev = all_counts[m]
        all_counts[m] = (c_prev + c, t_prev + t)

    print(f"\n  {'Method':<20}  {'Cov':>6}  {'Band95':>13}  {'Width':>8}  {'Time(ms)':>14}")
    for m in method_labels:
        mc = float(np.mean(all_cov[m])) if all_cov[m] else float("nan")
        mw = float(np.mean(all_wid[m])) if all_wid[m] else float("nan")
        c_tot, t_tot = all_counts[m]
        _, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        avg_ms, se_ms = _time_stats([r for r in results if r.method == m])
        time_str = f"{avg_ms:.3f}+-{se_ms:.3f}" if np.isfinite(avg_ms) else "-"
        print(f"  {m:<20}  {mc:>5.3f}{_cov_marker(mc, target)}  {f'{lo:.3f}-{hi:.3f}':>13}  {mw:>8.4f}  {time_str:>14}")
    print()


def save_results_artifacts(*, results: list[SimResult], alpha: float, sample_sizes: list[int], n_reps: int, out_dir: str, run_stem: str) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    csv_path = out_base / f"{run_stem}_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source", "model", "benchmark_id", "label", "eval_type", "n", "method", "n_reps",
            "covered", "total_width", "coverage", "mean_width", "mcse", "band95_low", "band95_high",
            "avg_time_ms", "se_time_ms", "corpus_size", "corpus_mean", "run_noise_frac", "runs",
        ])
        for r in results:
            coverage = r.covered / r.n_reps
            mean_width = r.total_width / r.n_reps
            _, mcse, lo, hi = _mc_proportion_stats(r.covered, r.n_reps)
            avg_ms, se_ms = _time_stats([r])
            writer.writerow([
                r.source, r.model or "", r.benchmark_id or "", r.label, r.eval_type, r.n, r.method, r.n_reps,
                r.covered, f"{r.total_width:.8f}", f"{coverage:.8f}", f"{mean_width:.8f}",
                f"{mcse:.8f}", f"{lo:.8f}", f"{hi:.8f}",
                f"{avg_ms:.6f}" if np.isfinite(avg_ms) else "",
                f"{se_ms:.6f}" if np.isfinite(se_ms) else "",
                r.corpus_size if r.corpus_size is not None else "",
                f"{r.corpus_mean:.8f}" if r.corpus_mean is not None else "",
                f"{r.run_noise_frac:.6f}", r.runs,
            ])

    summary_path = out_base / f"{run_stem}_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(results, sample_sizes=sample_sizes, alpha=alpha, n_reps=n_reps)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")

    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_coverage_vs_n_plot(*, results: list[SimResult], sample_sizes: list[int], alpha: float, n_reps: int, out_path: str) -> str:
    """Coverage vs. sample size line plots -- one subplot per eval type, all methods overlaid."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - alpha
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    present_methods = {r.method for r in results}
    method_objs = order_present_methods(present_methods)
    method_names = [m.name for m in method_objs]

    df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "method": r.method, "n": r.n, "coverage": r.covered / r.n_reps}
        for r in results
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

    fig.suptitle(f"Coverage vs. Sample Size\nci_single | reps={n_reps} | alpha={alpha}", fontsize=12)
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

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    present_methods = {r.method for r in results}
    method_objs = order_present_methods(present_methods)
    method_names = [m.name for m in method_objs]

    df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "method": r.method, "n": r.n, "width": r.total_width / r.n_reps}
        for r in results
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

    fig.suptitle(f"CI Width vs. Sample Size\nci_single | reps={n_reps} | alpha={alpha}", fontsize=12)
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
    present_methods = {r.method for r in results}
    method_objs = order_present_methods(present_methods)
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]

    def _label_indices(ns: list) -> set:
        if len(ns) <= 2:
            return set(range(len(ns)))
        return {0, len(ns) // 2, len(ns) - 1}

    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(11.0, 4.5 * nrows), squeeze=False, gridspec_kw={"hspace": 0.45})

    for row_idx, et in enumerate(eval_types_present):
        ax = axes[row_idx][0]
        et_results = [r for r in results if r.eval_type == et]
        sample_sizes = sorted({r.n for r in et_results})

        ax.axhspan(max(0.0, target - 0.04), min(1.0, target + 0.04), color="#DDDDDD", alpha=0.40, zorder=0)
        ax.axhline(target, color="black", linewidth=1.1, linestyle="--", zorder=1)

        legend_handles = []
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
        ax.set_xlabel("Mean CI time (ms) -- log scale  [error bars: +-1.96 SE]", fontsize=9.5)
        ax.set_ylabel("Coverage rate", fontsize=9.5)
        ax.set_title(f"eval type: {et}", fontsize=10.5)
        ax.set_ylim(max(0.0, target - 0.20), min(1.01, target + 0.12))
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
        ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.35)
        ax.tick_params(labelsize=8.5)

        if not et_results:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        if legend_handles:
            ax.legend(handles=legend_handles, title="Method", loc="center left", bbox_to_anchor=(1.02, 0.5),
                      borderaxespad=0.0, fontsize=7.5, title_fontsize=8, framealpha=0.85, ncol=1)

    fig.suptitle(
        f"Cost x Coverage Trade-off\n"
        f"ci_single | x = mean CI compute time | y = empirical coverage | target = {target:.0%} | reps={n_reps}",
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
    present_methods = {r.method for r in results}
    method_objs = order_present_methods(present_methods)
    run_noise_fracs = sorted({r.run_noise_frac for r in results})
    if len(run_noise_fracs) < 2:
        print(f"Skipped coverage-vs-run-noise plot (only one f_run value): {out_path}")
        return None

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows, 1, figsize=(10.0, 4.0 * nrows), squeeze=False, gridspec_kw={"hspace": 0.45})

    for row_idx, et in enumerate(eval_types_present):
        ax = axes[row_idx][0]
        ax.axhspan(max(0.0, target - 0.04), min(1.0, target + 0.04), color="#DDDDDD", alpha=0.40, zorder=0)
        ax.axhline(target, color="black", linewidth=1.2, linestyle="--", zorder=1)

        for m in method_objs:
            covs = []
            for f in run_noise_fracs:
                subset = [r for r in results if r.eval_type == et and r.method == m.name and r.run_noise_frac == f]
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

    runs_val = results[0].runs if results else "?"
    fig.suptitle(
        f"Coverage vs. Run Noise Fraction\n"
        f"ci_single (nested mode) | runs={runs_val} | alpha={alpha} | reps={n_reps}\n"
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
                         help="'synthetic' (default), or a real-data source: " + ", ".join(REAL_DATA_SOURCES))
    parser.add_argument("--scenario-suite", choices=SCENARIO_SUITES, default="expanded",
                         help="Synthetic scenario breadth (ignored for real data sources)")
    parser.add_argument("--eval-types", nargs="+", choices=EVAL_TYPES, default=None, metavar="TYPE")
    parser.add_argument("--benchmarks", nargs="+", default=None, metavar="ID", help="Real-data: benchmark IDs to filter to")
    parser.add_argument("--models", nargs="+", default=None, metavar="NAME", help="Real-data: model names to filter to")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--min-corpus-size", type=int, default=50)
    parser.add_argument("--reps", type=int, default=200, metavar="N")
    parser.add_argument("--bootstrap-n", type=int, default=500, metavar="N")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 50], metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--nested-mode", action="store_true", default=False,
                         help="Multi-run flat-vs-nested CI comparison (ported from "
                              "sim_compare_boot_nested.py), synthetic only.")
    parser.add_argument("--runs", type=int, default=3, metavar="R",
                         help="Nested mode: runs per input when not sweeping (default: 3)")
    parser.add_argument("--runs-sweep", type=int, nargs="+", default=None, metavar="R",
                         help="Nested mode: sweep multiple R values, overrides --runs")
    parser.add_argument("--run-noise-fracs", type=float, nargs="+", default=RUN_NOISE_FRACS_DEFAULT, metavar="F",
                         help="Nested mode: f_run = var_run / (var_input + var_run) values to sweep")
    parser.add_argument("--icc-values", type=float, nargs="+", default=None, metavar="ICC",
                         help="Nested mode: optional ICC sweep, converted to run-noise via f_run = 1 - ICC")
    parser.add_argument("--bayes-n", type=int, default=None, metavar="N",
                         help="Nested mode: Bayesian bootstrap draws (default: --bootstrap-n)")
    parser.add_argument("--heteroscedastic", action="store_true", default=False,
                         help="Nested mode: run noise scales with input value (mimics real LLM eval variability)")


def official_args(base_seed: int = 42) -> argparse.Namespace:
    """Canonical official-test preset, mirroring sim_compare_boot.py --official-test
    (single-sample phase). Synthetic only -- real-data sources need network/HF
    access not assumed available; run --data-source dove/openeval manually."""
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="expanded", eval_types=None,
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_corpus_size=50,
        reps=2000, bootstrap_n=10000, alpha=0.05,
        sizes=[10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        seed=base_seed, progress="bar", plots="save", save_results="save",
        out_dir="simulations/out", plots_dir=None,
        nested_mode=False, runs=3, runs_sweep=None, run_noise_fracs=RUN_NOISE_FRACS_DEFAULT,
        icc_values=None, bayes_n=None, heteroscedastic=False,
    )


def nested_official_args(base_seed: int = 44) -> argparse.Namespace:
    """Canonical --nested-mode official preset, mirroring sim_compare_boot_nested.py's
    --official-test (mean-estimand phase). Not wired into --official-tests (the harness
    runs one preset per case); invoke manually:
    python -m simulations.harness.cli ci_single --nested-mode --runs-sweep 5 --reps 200
      --bootstrap-n 10000 --bayes-n 10000 --scenario-suite expanded --icc-values 0.05 0.30 0.50
      --run-noise-fracs 0.01 0.1 0.3 0.5 --heteroscedastic --sizes 10 20 30 50 75 100 --seed 44
    """
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="expanded", eval_types=None,
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_corpus_size=50,
        reps=200, bootstrap_n=10000, alpha=0.05, sizes=[10, 20, 30, 50, 75, 100],
        seed=base_seed, progress="bar", plots="save", save_results="save",
        out_dir="simulations/out", plots_dir=None,
        nested_mode=True, runs=5, runs_sweep=[5], run_noise_fracs=[0.01, 0.1, 0.3, 0.5],
        icc_values=[0.05, 0.30, 0.50], bayes_n=10000, heteroscedastic=True,
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

            print(f"\nci_single simulation (nested mode) -- runs={runs_list}, run_noise_fracs={run_noise_fracs}")
            sources = build_single_sample_sources(
                suite=args.scenario_suite, run_noise_fracs=run_noise_fracs, heteroscedastic=args.heteroscedastic,
            )
            if args.eval_types:
                requested = set(args.eval_types)
                sources = [s for s in sources if s.eval_type in requested]
            if not sources:
                raise ValueError("No CISources left after filtering.")
            print(f"  {len(sources)} sources, sizes={args.sizes}, reps={args.reps}, alpha={args.alpha}")

            results: list[SimResult] = []
            for r_val in runs_list:
                results.extend(run_nested_simulation(
                    sources, sample_sizes=args.sizes, runs=r_val, n_reps=args.reps, n_bootstrap=args.bootstrap_n,
                    bayes_n=bayes_n, alpha=args.alpha, progress_mode=args.progress, seed=args.seed,
                ))
            print_report(results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps)

            stamp = time.strftime("%Y%m%d_%H%M%S")
            run_stem = f"ci_single_nested_runs{'-'.join(str(r) for r in runs_list)}_reps{args.reps}_{stamp}"
            output_paths: list[str] = []

            if args.save_results == "save":
                output_paths += save_results_artifacts(
                    results=results, alpha=args.alpha, sample_sizes=args.sizes, n_reps=args.reps,
                    out_dir=args.out_dir, run_stem=run_stem,
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

            overall_cov = float(np.mean([r.covered / r.n_reps for r in results])) if results else float("nan")
            return CaseResult(
                case_name=CASE_NAME, status="ok", output_paths=output_paths,
                key_metrics={"n_results": len(results), "overall_mean_coverage": overall_cov},
                duration_s=time.time() - t0,
            )

        plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")

        print(f"\nci_single simulation -- data_source={args.data_source}")
        if args.data_source == "synthetic":
            sources = build_single_sample_sources(suite=args.scenario_suite)
        else:
            sources = build_real_data_sources(
                args.data_source, benchmarks=args.benchmarks, models=args.models,
                hf_token=args.hf_token, cache_dir=args.cache_dir, min_corpus_size=args.min_corpus_size,
            )

        if args.eval_types:
            requested = set(args.eval_types)
            sources = [s for s in sources if s.eval_type in requested]
        if not sources:
            raise ValueError("No CISources left after filtering.")

        print(f"  {len(sources)} sources, sizes={args.sizes}, reps={args.reps}, alpha={args.alpha}")

        results = run_simulation(
            sources, sample_sizes=args.sizes, n_reps=args.reps, n_bootstrap=args.bootstrap_n,
            alpha=args.alpha, progress_mode=args.progress, seed=args.seed,
        )
        print_report(results, sample_sizes=args.sizes, alpha=args.alpha, n_reps=args.reps)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_stem = f"ci_single_{args.data_source}_reps{args.reps}_{stamp}"
        output_paths: list[str] = []

        if args.save_results == "save":
            output_paths += save_results_artifacts(
                results=results, alpha=args.alpha, sample_sizes=args.sizes, n_reps=args.reps,
                out_dir=args.out_dir, run_stem=run_stem,
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

        overall_cov = float(np.mean([r.covered / r.n_reps for r in results])) if results else float("nan")
        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics={"n_results": len(results), "overall_mean_coverage": overall_cov},
            duration_s=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return CaseResult(case_name=CASE_NAME, status="error", error=str(exc), duration_s=time.time() - t0)
