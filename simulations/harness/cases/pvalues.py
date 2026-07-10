"""pvalues case: p-value/rejection-decision calibration, non-PPI and PPI-corrected.

Consolidates two complementary benchmarks into one file with several modes
(``--mode {pairwise,multiarm,ppi,simultaneous_ci,pairwise_multiarm,all}``):

Non-PPI path (ported from ``simulations/sim_compare_pvalues.py``)
------------------------------------------------------------------
Given LLM-only scores (no human labels), which raw p-value/rejection
procedure is best calibrated?

- ``pairwise``: Type-I error and power for pairwise A-vs-B comparisons,
  via ``scenarios.synthetic.build_pair_sources``'s ICC x Cohen's-d grid
  (the same paired-difference scenario library ``cases/ci_paired.py`` uses)
  as the null/weak-alt/full-alt conditions.
- ``multiarm``: family-wise false-positive rate and best-arm selection power
  across p-value correction strategies (holm/bonferroni/fdr_bh/
  friedman_nemenyi), via ``scenarios.synthetic.build_multiarm_sources``,
  sweeping the SAME shape catalog ``build_pair_sources`` uses, generalized
  to k arms.
- ``simultaneous_ci``: family-wise CI coverage and average per-comparison
  width for the three simultaneous-CI constructions with a well-established
  dual to multiarm's p-value corrections -- ``none`` (naive per-pair CI, no
  simultaneous adjustment -- the "why do you need any correction?"
  baseline), Bonferroni t-intervals, and max-T (studentized bootstrap,
  Romano-Wolf, what ``evalstats.core.paired.all_pairwise`` uses by default)
  -- forced side by side on the SAME draw with the SAME point-estimate
  method held fixed (bypassing ``all_pairwise``'s own automatic
  method-based routing for max-T/Bonferroni), on the identical k-arm
  sources ``multiarm`` uses. multiarm's other three corrections
  (holm/fdr_bh/friedman_nemenyi) have no such CI dual -- holm/fdr_bh are
  p-value-only adjustments, and friedman_nemenyi operates on rank
  differences rather than the raw mean-difference scale a CI needs. This is
  the evidence for why max-T is the harness's default simultaneous-CI
  method: it should hit nominal coverage same as Bonferroni (unlike
  ``none``, which should visibly under-cover as k grows), so a narrower
  average width at matching coverage is what actually distinguishes it from
  Bonferroni.

PPI-corrected path (ported from ``simulations/sim_type_i_calibration.py``)
---------------------------------------------------------------------------
- ``ppi``: given LLM scores plus sparse (possibly MNAR) human labels, does
  PPI correction (``evalstats.tests``' internal machinery) fix the Type-I
  inflation that judge bias/miscalibration causes in the uncorrected
  (scipy-equivalent) version? Sweeps judge-bias parameters via
  ``scenarios.synthetic.build_judge_bias_sources`` /
  ``generate_judge_bias_cell``, one factor at a time from a fixed baseline,
  layered on top of ONE representative shape per eval type from the SAME
  catalog the other two modes use. Includes a ``noise_family.*`` factor
  (``JudgeBiasSource.noise_family="contaminated"``) checking the same
  question under "judge mostly right, occasionally catastrophically wrong"
  measurement error instead of the default symmetric Gaussian -- same total
  noise variance either way, just redistributed.

There is no separate ``ppi_calibration`` case: it was folded in here instead,
since both halves answer "is this statistical decision trustworthy" at
different levels of the stack (raw CI/p-value procedures vs. high-level
``evalstats.tests`` wrappers with PPI), sharing report/plot/CLI scaffolding
AND, as of the shape-catalog unification, the underlying truth-generating
process too (``scenarios.synthetic.sample_group_truth`` -- see that
module's docstring and the harness README's "Shared scenario library"
section).

Known exceptions (see simulations/harness/README.md):
- ``ppi`` mode's one-factor-at-a-time sweep covers ``eval_type`` in
  ``{continuous, likert, grades}`` in full (one representative shape per
  eval type rather than the full catalog ``multiarm`` sweeps -- judge-bias/
  noise/label-fraction/etc. parameters are PPI's actual axis of interest,
  not distribution shape); ``binary`` is supported only for the
  two-independent-groups/paired mean-based tests (``ttest``/``ttest_welch``/
  ``paired_t``/``bayes_bootstrap`` -- a proportion is just the mean of a
  0/1 variable, so PPI's rectifier applies unchanged), as a single
  baseline-settings scenario rather than swept across every other factor.
  ``bayes_bootstrap`` PPI-corrects the same paired-mean estimand as
  ``paired_t`` but via Dirichlet-weighted (Bayesian) bootstrap resampling
  instead of ``evalstats.ppi.correct``'s classical one (see
  ``evalstats.tests._ppi_paired_bayes_bootstrap``) -- kept as a validated
  alternative, not a recommended default (real-data testing found it
  underperforms; ``paired_t`` is the reasonable default for binary p-values).
  ``bootstrap_t`` PPI-corrects the SAME paired-mean estimand via a
  studentized-bootstrap pivot (see
  ``evalstats.tests._ppi_paired_bootstrap_t``), generalizing
  ``evalstats.core.resampling.bootstrap_t_ci_1d``'s per-replicate SE to
  PPI's two-term variance -- numeric (continuous/likert/grades) ONLY, not
  extended to binary, since its value is specifically for resampling-based
  CI estimation on numeric data at N>=50 (``ci_paired.py``), not pairwise
  binary p-values. ``tango_score`` is the mirror image -- binary ONLY, not
  numeric -- PPI-correcting ``evalstats.core.resampling.tango_paired_ci``'s
  score interval (see ``evalstats.tests._ppi_paired_tango``): its variance
  term ``(n10+n01)/n^2 - (n10-n01)^2/n^3`` is exactly
  ``Var(diffs, ddof=0) / n``, so it generalizes to PPI's two-term variance
  by substituting an effective n (``n_eff = Var(unlabeled diffs) /
  V_hat_PPI``) into the SAME Wilson-style shrinkage formula -- fully
  closed-form, no bootstrap needed, and reduces EXACTLY to the original
  (uncorrected) formula when the "labeled" subset is the full sample with
  no judge error. ``mcnemar`` is intentionally NOT PPI-corrected here: its
  distinguishing feature is an EXACT small-sample binomial test on
  discordant-pair counts, and a PPI-corrected numerator is generally
  non-integer, breaking that exactness -- left as future work pending a
  firmer statistical basis rather than shipping an ad-hoc adaptation. The
  rank-based family (``mw``/``wilcoxon``/``friedman``/``kruskal``) and
  ANOVA/LMM remain continuous/likert/grades-only: they assume a scale that
  doesn't hold up under binary's massive ties, and the judge-bias noise
  model used for those structures doesn't have a binary-compatible variant
  (yet) -- see ``scenarios.synthetic``'s
  ``_jb_llm_binary``/``_jb_llm_repeated_binary``.
- None of the three modes numerically matches its legacy script anymore
  (a deliberate trade: cross-mode truth-distribution consistency over
  per-mode legacy-script parity) -- verification is by sanity check
  (Type-I ~ alpha, power increasing with effect size/n) for all three.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import multiprocessing as _mp
import os
import threading
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.paired import pairwise_differences, all_pairwise, friedman_nemenyi, _bonferroni_simultaneous_cis, _simultaneous_cis_router
    from evalstats.core.stats_utils import correct_pvalues
    from evalstats.core.resampling import bayes_bootstrap_means_1d
    from evalstats.tests import (
        _ppi_two_sample,
        _ppi_paired_arrays,
        _ppi_paired_bayes_bootstrap,
        _ppi_paired_bootstrap_t,
        _ppi_paired_tango,
        _p_x_gt_y_midrank,
        _ppi_anova_independent_p_value,
        _ppi_anova_repeated_p_value,
        _ppi_friedman_p_value,
        _ppi_anova_independent,
        _ppi_anova_repeated,
        _ppi_friedman,
        _ppi_kruskal_wallis_pairwise,
        _ppi_lmm_p_value,
        _kw_pairwise_thetas,
        _mcnemar_p,
    )
    from evalstats.core.mixed_effects import _fit_lmm_general, _get_fe_vcov_sm

from ..latex_tables import booktabs_table, escape_latex, eval_type_label
from ..scenarios import CIPairSource, MultiArmSource, JudgeBiasSource, EVAL_TYPES
from ..scenarios.synthetic import (
    SCENARIO_SUITES,
    build_pair_sources,
    build_multiarm_sources,
    build_judge_bias_sources,
    generate_judge_bias_cell,
    estimate_judge_bias_gold_null_values,
    JUDGE_BIAS_LMM_FACTORIAL_FACTORS,
)
from ..scenarios.real_data import (
    DEFAULT_INSPECT_CSV, PAIR_SOURCES as REAL_PAIR_SOURCES, build_real_pair_sources, build_real_multiarm_sources,
)
from ..methods import (
    PAIRWISE_PVALUE_METHODS,
    MCNEMAR,
    BOOTSTRAP,
    BOOTSTRAP_T,
    BCA,
    BAYES_BOOTSTRAP,
    SMOOTH_BOOTSTRAP,
    PERMUTATION,
    SIGN_TEST,
    NEWCOMBE_PVAL,
    BAYES_BINARY,
    WILCOXON,
    PAIRED_T,
    TANGO,
    MULTIARM_CORRECTION_METHODS,
    SIMULTANEOUS_CI_METHODS,
    CORR_NONE,
    PPI_TEST_METHODS,
    TTEST,
    TTEST_WELCH,
    MW,
    ANOVA_IND,
    ANOVA_REP,
    FRIEDMAN,
    KRUSKAL,
    LMM,
    LMM_FACTORIAL,
    LMM_RUNS,
    get_method_color,
    order_present_methods,
)
from . import CaseResult

CASE_NAME = "pvalues"

MODES = ["pairwise", "multiarm", "ppi", "simultaneous_ci", "pairwise_multiarm", "all"]
DATA_SOURCES = ["synthetic"] + REAL_PAIR_SOURCES
PROGRESS_MODES = ["bar", "cell", "off"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]
ALPHA_DEFAULT = 0.05

_BINARY_ONLY_PVAL_METHODS = {NEWCOMBE_PVAL.name, BAYES_BINARY.name, MCNEMAR.name}

# `none` stays in _run_simultaneous_ci_cell's data collection and in
# print_simultaneous_ci_report's tables (it's the "why do you need any
# correction at all" baseline there), but is dropped from every
# simultaneous_ci *plot* -- it's so far below nominal coverage that it
# squashes the Bonferroni-vs-max-T comparison those plots exist to show.
SIMULTANEOUS_CI_PLOT_METHODS = [m for m in SIMULTANEOUS_CI_METHODS if m.name != CORR_NONE.name]


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


def _mc_proportion_stats(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    if total <= 0:
        return (float("nan"),) * 4
    p_hat = successes / total
    mcse = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    return float(p_hat), mcse, max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)


# ---------------------------------------------------------------------------
# Pairwise mode (non-PPI): Type-I error and power across raw p-value
# procedures, ported from sim_compare_pvalues.py's pairwise phase. Reuses
# scenarios.synthetic.build_pair_sources -- its is_null rows are the "null"
# condition, its cohens_d_values sweep is the alt-condition power curve.
# Real-data sources (build_real_pair_sources) have no synthetic Cohen's d;
# their non-null condition is labeled "real" instead (see _run_pairwise_cell).
# ---------------------------------------------------------------------------


@dataclass
class PairwiseResult:
    eval_type: str
    label: str
    n: int
    method: str
    condition: str  # "null" | f"d={cohens_d:.2f}" | "real" (real-data non-null)
    n_reps: int
    rejects: int
    p_sum: float
    cohens_d: float = 0.0


def _safe_wilcoxon_p(diffs: np.ndarray) -> float:
    if int(np.sum(diffs != 0)) < 1:
        return 1.0
    try:
        with np.errstate(all="ignore"):
            w = scipy_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        p = float(w.pvalue)
        return min(max(p, 0.0), 1.0) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _safe_paired_t_p(diffs: np.ndarray) -> float:
    if len(diffs) <= 1:
        return 1.0
    try:
        with np.errstate(all="ignore"):
            t = scipy_stats.ttest_1samp(diffs, popmean=0.0, nan_policy="omit")
        p = float(t.pvalue)
        return min(max(p, 0.0), 1.0) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _pairwise_pvalue(a: np.ndarray, b: np.ndarray, method: str, n_bootstrap: int, rng: np.random.Generator, statistic: str) -> float:
    """Compute one method's p-value for paired A-vs-B data of shape (n, runs)."""
    diffs = a.mean(axis=1) - b.mean(axis=1)

    if method == WILCOXON.name:
        return _safe_wilcoxon_p(diffs)
    if method == PAIRED_T.name:
        return _safe_paired_t_p(diffs)

    if method in _BINARY_ONLY_PVAL_METHODS:
        aa = (a.mean(axis=1) >= 0.5).astype(float)
        bb = (b.mean(axis=1) >= 0.5).astype(float)
        if method == MCNEMAR.name:
            return _mcnemar_p(aa, bb)
        scores = np.stack([aa, bb], axis=0)
    else:
        scores = np.stack([a[:, 0], b[:, 0]], axis=0) if a.shape[1] == 1 else np.stack([a, b], axis=0)

    result = pairwise_differences(
        scores=scores, idx_a=0, idx_b=1, label_a="A", label_b="B",
        method=method, ci=0.95, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
    )
    p = float(result.p_value)
    return min(max(p, 0.0), 1.0) if np.isfinite(p) else 1.0


def _pairwise_methods_allowed(eval_type: str) -> list:
    return [m for m in PAIRWISE_PVALUE_METHODS if m.name not in _BINARY_ONLY_PVAL_METHODS or eval_type == "binary"]


def _run_pairwise_cell(
    source: CIPairSource, n: int, runs: int, n_reps: int, n_bootstrap: int, alpha: float, statistic: str, seed,
) -> list[PairwiseResult]:
    methods = _pairwise_methods_allowed(source.eval_type)
    if source.is_null:
        condition = "null"
    elif source.source != "synthetic":
        # Real A-vs-B pairs have a genuine (usually nonzero) true_diff, but no
        # synthetic Cohen's d -- label the power column "real" rather than the
        # misleading "d=0.00" (CIPairSource.cohens_d's default for real data).
        condition = "real"
    else:
        condition = f"d={source.cohens_d:.2f}"

    ss = np.random.SeedSequence(seed)
    data_rng = np.random.default_rng(ss.spawn(1)[0])
    method_rngs = {m.name: np.random.default_rng(s) for m, s in zip(methods, ss.spawn(len(methods)))}

    rejects: dict[str, int] = {m.name: 0 for m in methods}
    p_sums: dict[str, float] = {m.name: 0.0 for m in methods}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Real paired binary data frequently has many items where both
        # models score identically (diffs == 0), which triggers scipy's
        # nan_policy="omit" wrapper (_axis_nan_policy.py) to warn about
        # catastrophic cancellation in its internal moment calculation --
        # benign here (the p-value is still valid), same as the
        # friedmanchisquare/kruskal RuntimeWarning suppression below.
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_reps):
            a, b = source.generate_pair(data_rng, n, runs)
            for m in methods:
                p = _pairwise_pvalue(a, b, method=m.name, n_bootstrap=n_bootstrap, rng=method_rngs[m.name], statistic=statistic)
                p_sums[m.name] += p
                if p <= alpha:
                    rejects[m.name] += 1

    return [
        PairwiseResult(
            eval_type=source.eval_type, label=source.label, n=n, method=m.name, condition=condition,
            n_reps=n_reps, rejects=rejects[m.name], p_sum=p_sums[m.name], cohens_d=source.cohens_d,
        )
        for m in methods
    ]


_PAIRWISE_SOURCES: list = []  # fork-inherited worker state for run_pairwise_simulation
_MULTIARM_SOURCES: list = []  # fork-inherited worker state for run_multiarm_simulation


def _run_pairwise_cell_worker(args: tuple) -> list[PairwiseResult]:
    sc_idx, n, runs, n_reps, n_bootstrap, alpha, statistic, seed = args
    return _run_pairwise_cell(_PAIRWISE_SOURCES[sc_idx], n, runs, n_reps, n_bootstrap, alpha, statistic, seed)


def _run_multiarm_cell_worker(args: tuple) -> list[MultiArmResult]:
    sc_idx, n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed = args
    return _run_multiarm_cell(_MULTIARM_SOURCES[sc_idx], n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed)


def _run_ppi_cell_worker(args: tuple) -> list:
    sc, active_tests, n_reps, n_boot, seed, progress_dict = args
    return _run_ppi_cell(sc, active_tests, n_reps, n_boot, seed, progress_dict=progress_dict, progress_key=sc.name)


def _run_ppi_effect_cell_worker(args: tuple) -> tuple:
    sc_idx, sc, active_tests, n_reps, n_boot, seed = args
    return (sc_idx, _run_ppi_effect_cell(sc, active_tests, n_reps, n_boot, seed))


def run_pairwise_simulation(
    sources: list[CIPairSource], sample_sizes: list[int], runs: int, n_reps: int, n_bootstrap: int,
    alpha: float, statistic: str, progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
) -> list[PairwiseResult]:
    global _PAIRWISE_SOURCES
    _PAIRWISE_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = [(i, n) for i, s in enumerate(sources) for n in sample_sizes]
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, runs, n_reps, n_bootstrap, alpha, statistic, seed)
                 for (sc_idx, n), seed in zip(cells, child_seeds)]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pvalues-pairwise")
    results: list[PairwiseResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_pairwise_cell_worker(a))
            sc_idx, n = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} {sources[sc_idx].label} n={n}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_pairwise_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


def print_pairwise_report(results: list[PairwiseResult], alpha: float) -> None:
    print(f"\n{'='*78}\n  PVALUES (PAIRWISE, NON-PPI) -- TYPE I ERROR + POWER\n  Nominal alpha: {alpha}\n{'='*78}")
    present_methods = {r.method for r in results}
    method_labels = [m.name for m in order_present_methods(present_methods)]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]

    for et in eval_types_present:
        et_rows = [r for r in results if r.eval_type == et]
        # Per-eval-type, not global: some eval types (e.g. "binary", via
        # build_pair_sources' hand-picked asymmetric scenarios) have a
        # non-null condition literally labeled "d=0.00" alongside the real
        # "null" rows. Computing this list globally would leak that column
        # into every other eval type's table too, where no such non-null
        # d=0.00 rows exist -- showing a spurious all-nan "power(d=0.00)"
        # column instead of just omitting it.
        et_conditions = sorted({r.condition for r in et_rows if r.condition != "null"})
        print(f"\n  [{et}]")
        hdr = f"    {'Method':<14} {'typeI':>8}" + "".join(f"  power({c})".rjust(14) for c in et_conditions)
        print(hdr)
        for m in method_labels:
            m_rows = [r for r in et_rows if r.method == m]
            if not m_rows:
                continue
            null_rows = [r for r in m_rows if r.condition == "null"]
            c_tot = sum(r.rejects for r in null_rows)
            t_tot = sum(r.n_reps for r in null_rows)
            type1 = c_tot / t_tot if t_tot > 0 else float("nan")
            row = f"    {m:<14} {type1:>8.3f}"
            for c in et_conditions:
                c_rows = [r for r in m_rows if r.condition == c]
                cr = sum(r.rejects for r in c_rows)
                ct = sum(r.n_reps for r in c_rows)
                pw = cr / ct if ct > 0 else float("nan")
                row += f"  {pw:>12.3f}"
            print(row)

    conditions = sorted({r.condition for r in results if r.condition != "null"})
    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    n_cols = "".join(f"  {'n='+str(n):>7}" for n in sizes_present)
    print(f"\n{'-'*72}\n  OVERALL SUMMARY (collapsed across eval types, sources, n)\n{'-'*72}")
    print(f"  MaxT1 = worst per-scenario Type-I error seen for that method (not an average) --\n"
          f"  flags methods whose good mean Type-I error hides an inflated scenario/n cell.")
    print(f"\n  {'Method':<20}  {'TypeI':>6}  {'MaxT1':>7}  {'Band95':>13}  {'MeanPow':>8}{n_cols}")
    for m in method_labels:
        m_rows = [r for r in results if r.method == m]
        if not m_rows:
            continue
        null_rows = [r for r in m_rows if r.condition == "null"]
        c_tot = sum(r.rejects for r in null_rows)
        t_tot = sum(r.n_reps for r in null_rows)
        type1 = c_tot / t_tot if t_tot > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        band = f"{lo:.3f}-{hi:.3f}" if np.isfinite(lo) else "-"
        power_cells = []
        for c in conditions:
            c_rows = [r for r in m_rows if r.condition == c]
            cr = sum(r.rejects for r in c_rows)
            ct = sum(r.n_reps for r in c_rows)
            power_cells.append(cr / ct if ct > 0 else float("nan"))
        mean_power = float(np.mean([p for p in power_cells if np.isfinite(p)])) if power_cells else float("nan")
        marker = "*" if np.isfinite(type1) and type1 > alpha + 0.02 else " "
        per_label_t1 = defaultdict(lambda: [0, 0])
        for r in null_rows:
            acc = per_label_t1[(r.eval_type, r.label)]
            acc[0] += r.rejects
            acc[1] += r.n_reps
        label_rates = [c / t for c, t in per_label_t1.values() if t > 0]
        worst_t1 = max(label_rates) if label_rates else float("nan")
        worst_str = f"{worst_t1:.3f}{'*' if np.isfinite(worst_t1) and worst_t1 > alpha + 0.02 else ' '}" if np.isfinite(worst_t1) else "-"
        n_type1 = ""
        for n in sizes_present:
            n_rows = [r for r in null_rows if r.n == n]
            c_n = sum(r.rejects for r in n_rows)
            t_n = sum(r.n_reps for r in n_rows)
            t1_n = c_n / t_n if t_n > 0 else float("nan")
            n_type1 += f"  {t1_n:>7.3f}" if np.isfinite(t1_n) else f"  {'  -':>7}"
        print(f"  {m:<20}  {type1:>5.3f}{marker}  {worst_str:>7}  {band:>13}  {mean_power:>8.3f}{n_type1}")
    print(f"  (* = TypeI > alpha + 0.02)")
    print()


def latex_pairwise_overall_summary(results: list[PairwiseResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-method Type-I error (with its 95%
    MC band) + mean power, collapsed across eval types, plus one Type-I
    column per sample size actually swept, appended to the right -- the
    aggregate Type-I column collapses across n and can hide miscalibration
    that only shows up at small or large sample sizes."""
    present_methods = {r.method for r in results}
    method_labels = [m.name for m in order_present_methods(present_methods)]
    eval_types_present = {et for et in EVAL_TYPES if any(r.eval_type == et for r in results)}
    conditions = sorted({r.condition for r in results if r.condition != "null"})
    sizes_present = sorted({r.n for r in results if r.condition == "null"})

    rows = []
    for m in method_labels:
        m_rows = [r for r in results if r.method == m]
        if not m_rows:
            continue
        covered = {r.eval_type for r in m_rows}
        null_rows = [r for r in m_rows if r.condition == "null"]
        c_tot = sum(r.rejects for r in null_rows)
        t_tot = sum(r.n_reps for r in null_rows)
        type1 = c_tot / t_tot if t_tot > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(c_tot, t_tot)
        power_cells = []
        for c in conditions:
            c_rows = [r for r in m_rows if r.condition == c]
            cr = sum(r.rejects for r in c_rows)
            ct = sum(r.n_reps for r in c_rows)
            power_cells.append(cr / ct if ct > 0 else float("nan"))
        mean_power = float(np.mean([p for p in power_cells if np.isfinite(p)])) if power_cells else float("nan")
        row = [
            escape_latex(m),
            f"{type1:.3f}" if np.isfinite(type1) else "-",
            f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
            f"{mean_power:.3f}" if np.isfinite(mean_power) else "-",
            eval_type_label(covered, eval_types_present),
        ]
        for n in sizes_present:
            n_rows = [r for r in null_rows if r.n == n]
            c_n = sum(r.rejects for r in n_rows)
            t_n = sum(r.n_reps for r in n_rows)
            type1_n = c_n / t_n if t_n > 0 else float("nan")
            row.append(f"{type1_n:.3f}" if np.isfinite(type1_n) else "-")
        rows.append(row)

    return booktabs_table(
        caption=f"pvalues (pairwise, non-PPI): Type-I error and mean power across conditions (nominal alpha={alpha}).",
        label="tab:pvalues_pairwise_overall",
        columns=["Method", "Type-I error", "95\\% MC band", "Mean power", "Eval types"]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
    )


def save_results_artifacts_pairwise(*, results: list[PairwiseResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_pairwise_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_type", "label", "n", "method", "condition", "n_reps", "rejects", "reject_rate", "mean_p", "cohens_d"])
        for r in results:
            writer.writerow([
                r.eval_type, r.label, r.n, r.method, r.condition, r.n_reps, r.rejects,
                f"{r.rejects / r.n_reps:.8f}", f"{r.p_sum / r.n_reps:.8f}", f"{r.cohens_d:.6f}",
            ])
    summary_path = out_base / f"{run_stem}_pairwise_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_pairwise_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_pairwise_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def _save_pairwise_typeI_power_plot_one(
    *, results: list[PairwiseResult], alpha: float, out_path: str, eval_type: str,
) -> str:
    """Save a Type-I error (left) + power (right) plot for a single eval_type."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as _ticker

    present_methods = {r.method for r in results}
    method_objs = order_present_methods(present_methods)
    et_rows = [r for r in results if r.eval_type == eval_type]
    sample_sizes = sorted({r.n for r in results})

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.0, 4.2), squeeze=False)
    ax_t1, ax_pw = axes[0][0], axes[0][1]
    ax_t1.axhline(alpha, color="black", linewidth=1.0, linestyle="--")
    ax_t1.axhspan(max(0.0, alpha - 0.02), alpha + 0.02, color="#DDDDDD", alpha=0.4, zorder=0)

    for m in method_objs:
        m_rows = [r for r in et_rows if r.method == m.name]
        if not m_rows:
            continue
        null_rows = [r for r in m_rows if r.condition == "null"]
        xs, ys = [], []
        for n in sample_sizes:
            subset = [r for r in null_rows if r.n == n]
            if not subset:
                continue
            c = sum(r.rejects for r in subset)
            t = sum(r.n_reps for r in subset)
            xs.append(n)
            ys.append(c / t if t > 0 else float("nan"))
        if xs:
            ax_t1.plot(xs, ys, marker="o", color=m.color, markersize=4, linewidth=1.2, label=m.name, alpha=0.85)

        alt_rows = [r for r in m_rows if r.condition != "null"]
        xs2, ys2 = [], []
        for n in sample_sizes:
            subset = [r for r in alt_rows if r.n == n]
            if not subset:
                continue
            c = sum(r.rejects for r in subset)
            t = sum(r.n_reps for r in subset)
            xs2.append(n)
            ys2.append(c / t if t > 0 else float("nan"))
        if xs2:
            ax_pw.plot(xs2, ys2, marker="o", color=m.color, markersize=4, linewidth=1.2, label=m.name, alpha=0.85)

    ax_t1.set_title(f"{eval_type}: Type-I error")
    ax_t1.set_xlabel("n")
    ax_t1.set_ylabel("Rejection rate (null)")
    ax_t1.set_xscale("log")
    ax_pw.set_title(f"{eval_type}: power (mean over alt conditions)")
    ax_pw.set_xlabel("n")
    ax_pw.set_ylabel("Rejection rate (alt)")
    ax_pw.set_xscale("log")
    ax_t1.legend(fontsize=6.5, loc="upper right")
    _loc = _ticker.FixedLocator(sample_sizes)
    _fmt = _ticker.FuncFormatter(lambda x, _: str(int(x)))
    _nul = _ticker.NullLocator()
    for _ax in (ax_t1, ax_pw):
        _ax.xaxis.set_major_locator(_loc)
        _ax.xaxis.set_major_formatter(_fmt)
        _ax.xaxis.set_minor_locator(_nul)
    # Ensure y=0 lines are visible even when all methods have zero Type-I error
    # (binary null under the shared-item model gives d_i=0 for all i, so all
    # tests correctly return p=1 -- FWER=0 is right, but visually looks blank).
    t1_lo, t1_hi = ax_t1.get_ylim()
    if t1_hi - t1_lo < 0.04:
        ax_t1.set_ylim(-0.005, max(t1_hi, alpha + 0.04))
    elif t1_lo > -0.003:
        ax_t1.set_ylim(-0.003, t1_hi)
    if eval_type == "binary":
        null_t1_vals = [
            r.rejects / r.n_reps
            for r in [r for r in et_rows if r.condition == "null"]
            if r.n_reps > 0
        ]
        if null_t1_vals and max(null_t1_vals) == 0.0:
            ax_t1.text(0.5, 0.25, "T1=0 (shared-item model:\nA≡B under null)", transform=ax_t1.transAxes,
                       ha="center", va="center", fontsize=7.5, color="#555555", style="italic")

    fig.suptitle(f"pvalues (pairwise, non-PPI): Type-I + Power [{eval_type}] | alpha={alpha}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_pairwise_typeI_power_plot(*, results: list[PairwiseResult], alpha: float, out_path: str) -> list[str]:
    """Save one Type-I error + power plot per eval_type present in results.

    ``out_path`` is used as the base path; ``_{eval_type}`` is inserted before
    the file extension for each saved file.  Returns all saved paths.
    """
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    if not eval_types_present:
        return []
    base = Path(out_path)
    stem, suffix = base.stem, base.suffix or ".png"
    saved: list[str] = []
    for et in eval_types_present:
        et_path = str(base.parent / f"{stem}_{et}{suffix}")
        _save_pairwise_typeI_power_plot_one(results=results, alpha=alpha, out_path=et_path, eval_type=et)
        saved.append(et_path)
    return saved


def save_pairwise_reliability_violin_plot(*, results: list[PairwiseResult], alpha: float, out_path: str) -> str:
    """Cross-scenario reliability: violin+strip of per-scenario Type-I error and
    power, one dot per (label, method) -- the pairwise-testing analogue of
    ci_single/ci_paired's reliability violin. Exposes the spread the OVERALL
    SUMMARY table's pooled Type-I error hides: a method with alpha-level Type-I
    error on average can still have scenario-specific inflation that pooling
    across labels masks."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    present_methods = {r.method for r in results}
    method_objs = order_present_methods(present_methods)
    method_names = [m.name for m in method_objs]
    palette = {m.name: m.color for m in method_objs}

    null_df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "method": r.method, "typeI": r.rejects / r.n_reps}
        for r in results if r.condition == "null" and r.n_reps > 0 and r.method in method_names
    ])
    alt_df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "method": r.method, "power": r.rejects / r.n_reps}
        for r in results if r.condition != "null" and r.n_reps > 0 and r.method in method_names
    ])
    null_scenario = (
        null_df.groupby(["eval_type", "label", "method"], as_index=False).agg(typeI=("typeI", "mean"))
        if not null_df.empty else null_df
    )
    alt_scenario = (
        alt_df.groupby(["eval_type", "label", "method"], as_index=False).agg(power=("power", "mean"))
        if not alt_df.empty else alt_df
    )

    n_cols = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        for row_idx, (scenario_df, metric, ylabel, ref_line) in enumerate([
            (null_scenario, "typeI", "Type-I error per scenario", alpha),
            (alt_scenario, "power", "Power per scenario", None),
        ]):
            ax = axes[row_idx][col_idx]
            et_df = scenario_df[scenario_df["eval_type"] == et] if not scenario_df.empty else scenario_df
            if et_df.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue
            et_methods = [name for name in method_names if name in et_df["method"].values]
            sns.violinplot(
                data=et_df, x="method", y=metric, order=et_methods, hue="method",
                hue_order=et_methods, palette=palette, cut=0, inner=None, linewidth=0.8,
                alpha=0.35, legend=False, ax=ax,
            )
            sns.stripplot(
                data=et_df, x="method", y=metric, order=et_methods, hue="method",
                hue_order=et_methods, palette=palette, size=4, alpha=0.7, jitter=0.25,
                linewidth=0.4, edgecolor="white", legend=False, ax=ax,
            )
            if ref_line is not None:
                ax.axhline(ref_line, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper() if row_idx == 0 else "")
            ax.tick_params(axis="x", rotation=45)
            for tick_label in ax.get_xticklabels():
                tick_label.set_ha("right")

    fig.suptitle(
        f"Cross-Scenario Reliability (one dot = one scenario)\npvalues pairwise | alpha={alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Multi-arm mode (non-PPI): family-wise error rate and best-arm selection
# power across p-value correction strategies, ported from
# sim_compare_pvalues.py's multi-arm phase.
# ---------------------------------------------------------------------------


@dataclass
class MultiArmResult:
    eval_type: str
    label: str
    n: int
    k: int
    correction: str
    condition: str  # "null" | "alt"
    n_reps: int
    any_reject: int
    best_selected: int
    total_time: float = 0.0  # total wall-clock seconds for all n_reps of this condition


def _compute_multiarm_metrics(
    *, scores: np.ndarray, labels: list[str], method: str, corrections: list[str],
    n_bootstrap: int, alpha: float, statistic: str, rng: np.random.Generator,
) -> dict[str, tuple[bool, bool]]:
    """Compute (any_reject, best_selected) for every correction strategy.

    Holm/Bonferroni/FDR use truly raw (marginal) bootstrap p-values from a
    call with simultaneous_ci=False.  The ``max_t`` entry reuses that SAME
    call's per-pair results, feeding them into all_pairwise's own
    simultaneous-CI router (_simultaneous_cis_router) directly rather than
    making a second all_pairwise(..., simultaneous_ci=True) call -- the
    latter would silently redo the entire k(k-1)/2-pair bootstrap loop from
    scratch just to layer the simultaneous-CI step on top, since
    all_pairwise always computes marginal results first regardless of
    simultaneous_ci. The router only reads `results` as a Bonferroni
    fallback (never as bootstrap input for max-T), so nothing here is
    short-circuited or approximated relative to two separate calls.  For
    bootstrap-compatible methods the resulting p-values are Romano-Wolf
    max-T FWER-controlled p-values -- each is the min p-value commensurate
    with the simultaneous CI that was reported to the user.  For
    non-bootstrap methods (permutation) max_t falls back to the raw
    (marginal) p-value, matching what all_pairwise itself does when its
    router falls back to Bonferroni (it only widens the CI, not `.p_value`).

    Also passes compute_wilcoxon=False to the raw all_pairwise call: the
    Wilcoxon signed-rank p-value it would otherwise compute for every pair
    is never read here (MULTIARM_CORRECTION_METHODS has no "wilcoxon"
    entry), so skipping it avoids a pure-overhead scipy call per pair.
    """
    results: dict[str, tuple[bool, bool]] = {}
    k = len(labels)
    pairs = [(labels[i], labels[j]) for i in range(k) for j in range(i + 1, k)]

    non_friedman_non_maxt = [c for c in corrections if c not in ("friedman_nemenyi", "max_t")]
    include_max_t = "max_t" in corrections

    if non_friedman_non_maxt or include_max_t:
        # Raw (marginal) p-values: no simultaneous-CI adjustment, no post-hoc correction.
        matrix_raw = all_pairwise(
            scores=scores, labels=labels, method=method, ci=1.0 - alpha,
            n_bootstrap=n_bootstrap, correction="none", rng=rng, statistic=statistic,
            simultaneous_ci=False, compute_wilcoxon=False,
        )
        raw_p = np.array([matrix_raw.get(a, b).p_value for (a, b) in pairs])
        pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}

        for correction in non_friedman_non_maxt:
            adj_p = raw_p if correction == "none" else correct_pvalues(raw_p, correction)
            has_any = bool(np.any(adj_p < alpha))
            best = labels[0]
            best_selected = True
            for other in labels[1:]:
                pair_idx = pair_to_idx.get((best, other))
                if pair_idx is None or not (adj_p[pair_idx] < alpha and matrix_raw.get(best, other).point_diff > 0.0):
                    best_selected = False
                    break
            results[correction] = (has_any, best_selected)

        if include_max_t:
            try:
                _sim_cis, sim_method, sim_pvalues = _simultaneous_cis_router(
                    scores=scores, results=matrix_raw.results, pairs=pairs, labels=labels,
                    method=method, ci=1.0 - alpha, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
                    prefer="max_t",
                )
                maxt_p = (
                    np.array([sim_pvalues[pair] for pair in pairs])
                    if sim_method == "max_t"
                    else raw_p
                )
                has_any = bool(np.any(maxt_p < alpha))
                best = labels[0]
                best_selected = True
                for other in labels[1:]:
                    pair_idx = pair_to_idx.get((best, other))
                    if pair_idx is None or not (maxt_p[pair_idx] < alpha and matrix_raw.get(best, other).point_diff > 0.0):
                        best_selected = False
                        break
                results["max_t"] = (has_any, best_selected)
            except Exception:
                results["max_t"] = (False, False)

    if "friedman_nemenyi" in corrections:
        try:
            fr = friedman_nemenyi(scores, labels)
            has_any = any(
                (p is not None and p < alpha) for (a, b) in pairs for p in [fr.get_nemenyi_p(a, b)]
            )
            best = labels[0]
            best_selected = True
            for other in labels[1:]:
                nem_p = fr.get_nemenyi_p(best, other)
                if nem_p is None:
                    best_selected = False
                    break
                if not (nem_p < alpha and fr.avg_ranks[best] < fr.avg_ranks[other]):
                    best_selected = False
                    break
            results["friedman_nemenyi"] = (has_any, best_selected)
        except Exception:
            results["friedman_nemenyi"] = (False, False)

    return results


def _run_multiarm_cell(
    source: MultiArmSource, n: int, runs: int, k_arms: int, n_reps: int, n_bootstrap: int,
    alpha: float, multiarm_method: str, statistic: str, seed,
) -> list[MultiArmResult]:
    labels = [f"arm_{i}" for i in range(k_arms)]
    corrections = [m.name for m in MULTIARM_CORRECTION_METHODS]
    rng = np.random.default_rng(seed)

    agg_any: dict[tuple[str, str], int] = {(c, cond): 0 for c in corrections for cond in ("null", "alt")}
    agg_best: dict[tuple[str, str], int] = {(c, cond): 0 for c in corrections for cond in ("null", "alt")}
    agg_time: dict[str, float] = {"null": 0.0, "alt": 0.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_reps):
            for condition, delta in (("null", 0.0), ("alt", source.alt_delta)):
                _t0 = time.perf_counter()
                scores = source.generate_scores(rng, n, runs, k_arms, delta)
                metrics = _compute_multiarm_metrics(
                    scores=scores, labels=labels, method=multiarm_method, corrections=corrections,
                    n_bootstrap=n_bootstrap, alpha=alpha, statistic=statistic, rng=rng,
                )
                agg_time[condition] += time.perf_counter() - _t0
                for correction in corrections:
                    any_reject, best_selected = metrics.get(correction, (False, False))
                    if any_reject:
                        agg_any[(correction, condition)] += 1
                    if best_selected:
                        agg_best[(correction, condition)] += 1

    return [
        MultiArmResult(
            eval_type=source.eval_type, label=source.label, n=n, k=k_arms, correction=correction,
            condition=condition, n_reps=n_reps, any_reject=agg_any[(correction, condition)],
            best_selected=agg_best[(correction, condition)],
            total_time=agg_time[condition],
        )
        for correction in corrections
        for condition in ("null", "alt")
    ]


def _multiarm_cell_feasible(s: MultiArmSource, n: int, k: int) -> bool:
    """Shared by run_multiarm_simulation and run_simultaneous_ci_simulation
    (both sweep the same MultiArmSource list over the same n x k grid)."""
    return (s.max_n is None or n < s.max_n) and (s.max_k is None or k <= s.max_k)


def _multiarm_style_cells(
    sources: list[MultiArmSource], sample_sizes: list[int], k_values: list[int],
) -> list[tuple[int, int, int]]:
    """(source_idx, n, k) cells feasible for every source, printing a skip
    warning (mirroring CISource.max_n's skip pattern) for infeasible ones."""
    cells = [(i, n, k) for i, s in enumerate(sources) for n in sample_sizes for k in k_values
             if _multiarm_cell_feasible(s, n, k)]
    skipped = [(s, n, k) for s in sources for n in sample_sizes for k in k_values
               if not _multiarm_cell_feasible(s, n, k)]
    for s, n, k in skipped:
        reason = f"n={n} >= corpus size {s.max_n}" if not (s.max_n is None or n < s.max_n) else f"k={k} > {s.max_k} real arms available"
        print(f"  Warning: {reason} for {s.label}. Skipping.")
    return cells


def run_multiarm_simulation(
    sources: list[MultiArmSource], sample_sizes: list[int], runs: int, k_values: list[int], n_reps: int,
    n_bootstrap: int, alpha: float, multiarm_method: str, statistic: str, progress_mode: str = "bar",
    seed: int = 42, n_workers: int = 1,
) -> list[MultiArmResult]:
    global _MULTIARM_SOURCES
    _MULTIARM_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = _multiarm_style_cells(sources, sample_sizes, k_values)

    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, runs, k, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed)
                 for (sc_idx, n, k), seed in zip(cells, child_seeds)]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pvalues-multiarm")
    results: list[MultiArmResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_multiarm_cell_worker(a))
            sc_idx, n, k = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} n={n} k={k}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_multiarm_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


def _time_stats_multiarm(results: list[MultiArmResult]) -> tuple[float, float]:
    """Average ± SE of wall-clock time per rep in milliseconds across cells."""
    valid = [r for r in results if r.total_time > 0 and r.n_reps > 0]
    if not valid:
        return float("nan"), float("nan")
    per_rep_ms = [r.total_time * 1000.0 / r.n_reps for r in valid]
    avg = float(np.mean(per_rep_ms))
    se = float(np.std(per_rep_ms, ddof=1) / np.sqrt(len(per_rep_ms))) if len(per_rep_ms) > 1 else 0.0
    return avg, se


def print_multiarm_report(results: list[MultiArmResult], alpha: float) -> None:
    print(f"\n{'='*78}\n  PVALUES (MULTI-ARM, NON-PPI) -- FWER + BEST-ARM POWER\n  Nominal alpha: {alpha}\n{'='*78}")
    corrections = [m.name for m in MULTIARM_CORRECTION_METHODS if m.name in {r.correction for r in results}]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    ks_present = sorted({r.k for r in results})
    for et in eval_types_present:
        for k in ks_present:
            subset = [r for r in results if r.eval_type == et and r.k == k]
            if not subset:
                continue
            print(f"\n  [{et}, k={k}]")
            print(f"    {'Correction':<20} {'FWER':>8} {'BestPower':>10}")
            for corr in corrections:
                c_rows = [r for r in subset if r.correction == corr]
                null_rows = [r for r in c_rows if r.condition == "null"]
                alt_rows = [r for r in c_rows if r.condition == "alt"]
                fwer_t = sum(r.n_reps for r in null_rows)
                fwer_c = sum(r.any_reject for r in null_rows)
                power_t = sum(r.n_reps for r in alt_rows)
                power_c = sum(r.best_selected for r in alt_rows)
                fwer = fwer_c / fwer_t if fwer_t > 0 else float("nan")
                power = power_c / power_t if power_t > 0 else float("nan")
                print(f"    {corr:<20} {fwer:>8.3f} {power:>10.3f}")

    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    print(f"\n{'-'*72}\n  OVERALL SUMMARY (collapsed across eval types, sources, n, k)\n{'-'*72}")
    print(f"  MaxFWER = worst per-scenario FWER seen for that correction (not an average) --\n"
          f"  flags corrections whose good mean FWER hides an inflated scenario/n/k cell.")
    n_cols = "".join(f"  {'n='+str(n):>7}" for n in sizes_present)
    k_cols = "".join(f"  {'k='+str(k):>6}" for k in ks_present)
    print(f"\n  {'Correction':<20}  {'FWER':>6}  {'MaxFWER':>8}  {'Band95':>13}  {'BestPow':>8}  {'Time(ms)':>14}{n_cols}{k_cols}")
    for corr in corrections:
        c_rows = [r for r in results if r.correction == corr]
        null_rows = [r for r in c_rows if r.condition == "null"]
        alt_rows = [r for r in c_rows if r.condition == "alt"]
        fwer_c = sum(r.any_reject for r in null_rows)
        fwer_t = sum(r.n_reps for r in null_rows)
        power_c = sum(r.best_selected for r in alt_rows)
        power_t = sum(r.n_reps for r in alt_rows)
        fwer = fwer_c / fwer_t if fwer_t > 0 else float("nan")
        power = power_c / power_t if power_t > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(fwer_c, fwer_t)
        avg_ms, se_ms = _time_stats_multiarm(null_rows)
        band = f"{lo:.3f}-{hi:.3f}" if np.isfinite(lo) else "-"
        time_str = f"{avg_ms:.1f}+-{se_ms:.1f}" if np.isfinite(avg_ms) else "-"
        marker = "*" if np.isfinite(fwer) and fwer > alpha + 0.02 else " "
        per_label_fwer = defaultdict(lambda: [0, 0])
        for r in null_rows:
            acc = per_label_fwer[(r.eval_type, r.label)]
            acc[0] += r.any_reject
            acc[1] += r.n_reps
        label_rates = [c / t for c, t in per_label_fwer.values() if t > 0]
        worst_fwer = max(label_rates) if label_rates else float("nan")
        worst_str = f"{worst_fwer:.3f}{'*' if np.isfinite(worst_fwer) and worst_fwer > alpha + 0.02 else ' '}" if np.isfinite(worst_fwer) else "-"
        n_fwer = ""
        for n in sizes_present:
            n_null = [r for r in null_rows if r.n == n]
            nc = sum(r.any_reject for r in n_null)
            nt = sum(r.n_reps for r in n_null)
            nf = nc / nt if nt > 0 else float("nan")
            n_fwer += f"  {nf:>7.3f}" if np.isfinite(nf) else f"  {'  -':>7}"
        k_fwer = ""
        for k in ks_present:
            k_null = [r for r in null_rows if r.k == k]
            kc = sum(r.any_reject for r in k_null)
            kt = sum(r.n_reps for r in k_null)
            kf = kc / kt if kt > 0 else float("nan")
            k_fwer += f"  {kf:>6.3f}" if np.isfinite(kf) else f"  {'  -':>6}"
        print(f"  {corr:<20}  {fwer:>5.3f}{marker}  {worst_str:>8}  {band:>13}  {power:>8.3f}  {time_str:>14}{n_fwer}{k_fwer}")
    print(f"  (* = FWER > alpha + 0.02)")


def latex_multiarm_overall_summary(results: list[MultiArmResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-correction FWER (with its 95% MC
    band) + best-arm power, collapsed across eval types, plus one FWER
    column per sample size and per k value actually swept."""
    corrections = [m.name for m in MULTIARM_CORRECTION_METHODS if m.name in {r.correction for r in results}]
    eval_types_present = {et for et in EVAL_TYPES if any(r.eval_type == et for r in results)}
    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    ks_present = sorted({r.k for r in results if r.condition == "null"})

    rows = []
    for corr in corrections:
        c_rows = [r for r in results if r.correction == corr]
        covered = {r.eval_type for r in c_rows}
        null_rows = [r for r in c_rows if r.condition == "null"]
        alt_rows = [r for r in c_rows if r.condition == "alt"]
        fwer_t = sum(r.n_reps for r in null_rows)
        fwer_c = sum(r.any_reject for r in null_rows)
        power_t = sum(r.n_reps for r in alt_rows)
        power_c = sum(r.best_selected for r in alt_rows)
        fwer = fwer_c / fwer_t if fwer_t > 0 else float("nan")
        power = power_c / power_t if power_t > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(fwer_c, fwer_t)
        avg_ms, se_ms = _time_stats_multiarm(null_rows)
        time_str = f"${avg_ms:.1f} \\pm {se_ms:.1f}$" if np.isfinite(avg_ms) else "-"
        row = [
            escape_latex(corr),
            f"{fwer:.3f}" if np.isfinite(fwer) else "-",
            f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
            f"{power:.3f}" if np.isfinite(power) else "-",
            time_str,
            eval_type_label(covered, eval_types_present),
        ]
        for n in sizes_present:
            n_rows = [r for r in null_rows if r.n == n]
            c_n = sum(r.any_reject for r in n_rows)
            t_n = sum(r.n_reps for r in n_rows)
            fwer_n = c_n / t_n if t_n > 0 else float("nan")
            row.append(f"{fwer_n:.3f}" if np.isfinite(fwer_n) else "-")
        for k in ks_present:
            k_rows = [r for r in null_rows if r.k == k]
            c_k = sum(r.any_reject for r in k_rows)
            t_k = sum(r.n_reps for r in k_rows)
            fwer_k = c_k / t_k if t_k > 0 else float("nan")
            row.append(f"{fwer_k:.3f}" if np.isfinite(fwer_k) else "-")
        rows.append(row)

    return booktabs_table(
        caption=f"pvalues (multi-arm, non-PPI): FWER and best-arm selection power (nominal alpha={alpha}). "
                f"Per-$n$ and per-$k$ FWER columns are collapsed across the other dimension and across eval types.",
        label="tab:pvalues_multiarm_overall",
        columns=["Correction", "FWER", "95\\% MC band", "Best-arm power", "Time (ms)", "Eval types"]
                + [f"n={n}" for n in sizes_present]
                + [f"k={k}" for k in ks_present],
        rows=rows,
    )


def save_results_artifacts_multiarm(*, results: list[MultiArmResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_multiarm_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_type", "label", "n", "k", "correction", "condition", "n_reps", "any_reject", "best_selected", "any_reject_rate", "best_selected_rate", "total_time_s", "time_ms_per_rep"])
        for r in results:
            time_ms = (r.total_time * 1000.0 / r.n_reps) if r.n_reps > 0 and r.total_time > 0 else float("nan")
            writer.writerow([
                r.eval_type, r.label, r.n, r.k, r.correction, r.condition, r.n_reps, r.any_reject, r.best_selected,
                f"{r.any_reject / r.n_reps:.8f}", f"{r.best_selected / r.n_reps:.8f}",
                f"{r.total_time:.6f}", f"{time_ms:.4f}" if not (time_ms != time_ms) else "",
            ])
    summary_path = out_base / f"{run_stem}_multiarm_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_multiarm_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_multiarm_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_multiarm_fwer_power_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """FWER vs. best-arm-selection power, one point per correction strategy per eval type."""
    import matplotlib.pyplot as plt

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows=1, ncols=nrows, figsize=(5.0 * nrows, 5.0), squeeze=False)

    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        et_rows = [r for r in results if r.eval_type == et]
        ax.axvline(alpha, color="black", linestyle="--", linewidth=1.0)
        for m in MULTIARM_CORRECTION_METHODS:
            c_rows = [r for r in et_rows if r.correction == m.name]
            null_rows = [r for r in c_rows if r.condition == "null"]
            alt_rows = [r for r in c_rows if r.condition == "alt"]
            t1 = sum(r.n_reps for r in null_rows)
            c1 = sum(r.any_reject for r in null_rows)
            t2 = sum(r.n_reps for r in alt_rows)
            c2 = sum(r.best_selected for r in alt_rows)
            if t1 == 0 or t2 == 0:
                continue
            fwer = c1 / t1
            power = c2 / t2
            ax.scatter([fwer], [power], color=m.color, s=60, label=m.name, edgecolors="white", linewidths=0.6)
        ax.set_xlabel("FWER (null)")
        ax.set_ylabel("Best-arm selection power (alt)")
        ax.set_title(f"eval type: {et}")
        ax.set_xlim(-0.02, max(0.3, alpha * 4))
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("pvalues (multi-arm, non-PPI): FWER vs. best-arm power", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_multiarm_fwer_vs_k_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """FWER and best-arm power as a function of k (number of arms), one curve per
    correction method, collapsed across eval types and sample sizes.  Only
    produced when more than one k value was swept; returns out_path unchanged
    (without writing) if all results share the same k."""
    import matplotlib.pyplot as plt

    ks_present = sorted({r.k for r in results})
    if len(ks_present) < 2:
        return out_path

    fig, (ax_fwer, ax_pow) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    ax_fwer.axhline(alpha, color="black", linewidth=1.0, linestyle="--", label=f"α={alpha}")
    ax_fwer.axhspan(max(0.0, alpha - 0.02), alpha + 0.02, color="#DDDDDD", alpha=0.4, zorder=0)

    for m in MULTIARM_CORRECTION_METHODS:
        c_rows = [r for r in results if r.correction == m.name]
        if not c_rows:
            continue
        xs, ys_fwer, ys_pow = [], [], []
        for k in ks_present:
            k_rows = [r for r in c_rows if r.k == k]
            null_rows = [r for r in k_rows if r.condition == "null"]
            alt_rows = [r for r in k_rows if r.condition == "alt"]
            fwer_t = sum(r.n_reps for r in null_rows)
            fwer_c = sum(r.any_reject for r in null_rows)
            power_t = sum(r.n_reps for r in alt_rows)
            power_c = sum(r.best_selected for r in alt_rows)
            if fwer_t == 0 or power_t == 0:
                continue
            xs.append(k)
            ys_fwer.append(fwer_c / fwer_t)
            ys_pow.append(power_c / power_t)
        if xs:
            ax_fwer.plot(xs, ys_fwer, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            ax_pow.plot(xs, ys_pow, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)

    ax_fwer.set_xlabel("k (number of arms)")
    ax_fwer.set_ylabel("FWER (null)")
    ax_fwer.set_title("FWER vs. number of arms")
    ax_fwer.set_ylim(bottom=0.0)
    ax_fwer.legend(fontsize=7)

    ax_pow.set_xlabel("k (number of arms)")
    ax_pow.set_ylabel("Best-arm selection power (alt)")
    ax_pow.set_title("Power vs. number of arms")
    ax_pow.set_ylim(0.0, 1.02)
    ax_pow.legend(fontsize=7)

    fig.suptitle("pvalues (multi-arm, non-PPI): FWER and power vs. k", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_multiarm_reliability_violin_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """Cross-scenario reliability: violin+strip of per-scenario FWER and
    best-arm power, one dot per (label, correction) -- the multi-arm analogue
    of the pairwise reliability violin. Exposes the spread the OVERALL SUMMARY
    table's pooled FWER hides: a correction with alpha-level FWER on average
    can still have scenario-specific inflation that pooling across labels
    masks, collapsed across n and k the same way the headline table is."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    corrections = [m.name for m in MULTIARM_CORRECTION_METHODS if m.name in {r.correction for r in results}]
    palette = {m.name: m.color for m in MULTIARM_CORRECTION_METHODS}

    null_df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "correction": r.correction, "fwer": r.any_reject / r.n_reps}
        for r in results if r.condition == "null" and r.n_reps > 0
    ])
    alt_df = pd.DataFrame([
        {"eval_type": r.eval_type, "label": r.label, "correction": r.correction, "power": r.best_selected / r.n_reps}
        for r in results if r.condition == "alt" and r.n_reps > 0
    ])
    null_scenario = (
        null_df.groupby(["eval_type", "label", "correction"], as_index=False).agg(fwer=("fwer", "mean"))
        if not null_df.empty else null_df
    )
    alt_scenario = (
        alt_df.groupby(["eval_type", "label", "correction"], as_index=False).agg(power=("power", "mean"))
        if not alt_df.empty else alt_df
    )

    n_cols = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        for row_idx, (scenario_df, metric, ylabel, ref_line) in enumerate([
            (null_scenario, "fwer", "FWER per scenario", alpha),
            (alt_scenario, "power", "Best-arm power per scenario", None),
        ]):
            ax = axes[row_idx][col_idx]
            et_df = scenario_df[scenario_df["eval_type"] == et] if not scenario_df.empty else scenario_df
            if et_df.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue
            et_corrections = [name for name in corrections if name in et_df["correction"].values]
            sns.violinplot(
                data=et_df, x="correction", y=metric, order=et_corrections, hue="correction",
                hue_order=et_corrections, palette=palette, cut=0, inner=None, linewidth=0.8,
                alpha=0.35, legend=False, ax=ax,
            )
            sns.stripplot(
                data=et_df, x="correction", y=metric, order=et_corrections, hue="correction",
                hue_order=et_corrections, palette=palette, size=4, alpha=0.7, jitter=0.25,
                linewidth=0.4, edgecolor="white", legend=False, ax=ax,
            )
            if ref_line is not None:
                ax.axhline(ref_line, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper() if row_idx == 0 else "")
            ax.tick_params(axis="x", rotation=45)
            for tick_label in ax.get_xticklabels():
                tick_label.set_ha("right")

    fig.suptitle(
        f"Cross-Scenario Reliability (one dot = one scenario)\npvalues multi-arm | alpha={alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Simultaneous-CI mode (non-PPI): calibration check for
# evalstats.core.paired.all_pairwise's simultaneous (family-wise) confidence
# intervals. Checks the three of multiarm's six p-value correction
# strategies that have an established simultaneous-CI dual: `none` (naive
# per-pair CI, no adjustment -- the uncorrected baseline), Bonferroni
# t-intervals, and max-T (studentized bootstrap, Romano-Wolf) -- the two
# non-naive constructions all_pairwise's own router (_simultaneous_cis_router)
# picks between automatically based on whether `method` is
# bootstrap-compatible. (holm/fdr_bh/friedman_nemenyi have no CI dual --
# holm/fdr_bh are p-value-only adjustments, friedman_nemenyi is on the rank
# scale -- so they're multiarm-only.) The router's auto-selection isn't
# something a caller can independently override, so this mode reaches past
# it (via the private _bonferroni_simultaneous_cis helper) to force all
# three constructions on the exact same draw with the exact same
# point-estimate `method` held fixed, which is the only way to get an
# apples-to-apples "does max-T actually beat Bonferroni (and beat doing
# nothing), all else equal" comparison instead of confounding the
# CI-construction choice with a method-family choice too.
#
# Reuses the SAME k-arm MultiArmSource scenarios (synthetic and real) that
# --mode multiarm sweeps, since both questions ("which p-value correction"
# and "which CI construction") share the identical underlying k-arm
# generative model -- just a different measurement per rep (coverage +
# width of the constructed simultaneous CI, instead of reject/best-arm).
# ---------------------------------------------------------------------------


@dataclass
class SimultaneousCIResult:
    eval_type: str
    label: str
    n: int
    k: int
    ci_method: str  # "max_t" | "bonferroni"
    condition: str  # "null" | "alt"
    n_reps: int
    all_covered: int
    """Count of reps where EVERY one of the k(k-1)/2 pairwise simultaneous
    CIs simultaneously contained its true difference (the family-wise
    coverage event -- the CI-construction analogue of multiarm's
    any_reject/FWER, just measuring the opposite: a miss on ANY pair, not a
    false rejection on any pair)."""
    total_width: float
    """Sum, across reps, of that rep's MEAN CI width across all k(k-1)/2
    pairs -- dividing by n_reps gives the average per-comparison width,
    comparable across different k and n."""
    total_score: float = 0.0
    """Sum, across reps, of that rep's FAMILY-WISE interval score: mean CI
    width across all k(k-1)/2 pairs, plus (2/alpha) * the WORST pair's miss
    distance (0 iff all_covered). Deliberately not the mean of each pair's own
    interval_score() (see evalstats.core.stats_utils) -- that marginal,
    per-comparison version rewards `none` even when its family-wise coverage
    collapses, since each of its individual intervals is close to nominally
    calibrated on its own. Using the worst pair's miss distance ties the
    penalty to the same "did ANY pair miss" event that all_covered measures."""
    total_time: float = 0.0  # total wall-clock seconds for all n_reps of this condition


def _run_simultaneous_ci_cell(
    source: MultiArmSource, n: int, runs: int, k_arms: int, n_reps: int, n_bootstrap: int,
    alpha: float, multiarm_method: str, statistic: str, seed,
) -> list[SimultaneousCIResult]:
    labels = [f"arm_{i}" for i in range(k_arms)]
    pairs = [(labels[i], labels[j]) for i in range(k_arms) for j in range(i + 1, k_arms)]
    ci = 1.0 - alpha
    rng = np.random.default_rng(seed)

    ci_methods = [m.name for m in SIMULTANEOUS_CI_METHODS]
    agg_covered: dict[tuple[str, str], int] = {(m, cond): 0 for m in ci_methods for cond in ("null", "alt")}
    agg_width: dict[tuple[str, str], float] = {(m, cond): 0.0 for m in ci_methods for cond in ("null", "alt")}
    agg_score: dict[tuple[str, str], float] = {(m, cond): 0.0 for m in ci_methods for cond in ("null", "alt")}
    agg_time: dict[str, float] = {"null": 0.0, "alt": 0.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_reps):
            for condition, delta in (("null", 0.0), ("alt", source.alt_delta)):
                _t0 = time.perf_counter()
                scores = source.generate_scores(rng, n, runs, k_arms, delta)
                true_means = source.true_means(k_arms, delta)

                # Raw (unadjusted) per-pair results -- gives per_input_diffs
                # for Bonferroni's t-interval formula, held for ALL THREE CI
                # constructions so only the CI math differs, not the draw.
                # The raw per-pair CI (matrix_raw.get(*pair).ci_low/.ci_high,
                # at plain level `ci`, no simultaneous widening at all) IS
                # the "none" construction -- multiarm's own "why do you need
                # any correction?" baseline, on the CI side instead of p-values.
                matrix_raw = all_pairwise(
                    scores=scores, labels=labels, method=multiarm_method, ci=ci,
                    n_bootstrap=n_bootstrap, correction="none", rng=rng, statistic=statistic,
                    simultaneous_ci=False,
                )
                none_cis = {pair: (matrix_raw.get(*pair).ci_low, matrix_raw.get(*pair).ci_high) for pair in pairs}
                bonf_cis = _bonferroni_simultaneous_cis(results=matrix_raw.results, pairs=pairs, ci=ci)

                # max-T: call _simultaneous_cis_router directly (the same
                # function all_pairwise(simultaneous_ci=True) would call
                # internally) instead of a second all_pairwise(...,
                # simultaneous_ci=True) call. That second call would
                # redundantly redo all k(k-1)/2 marginal pairwise_differences
                # computations -- for bootstrap_t, each an independent nested
                # double bootstrap -- purely to rebuild a `results` dict this
                # router only reads for its rare Bonferroni-fallback safety
                # net, when matrix_raw.results above already has exactly
                # that. Skipping the duplicate roughly halves this mode's
                # runtime with bootstrap_t (the expensive part is the k(k-1)/2
                # independent nested marginal bootstraps, not the one shared
                # max-T resample). Guard against that rare degenerate-data
                # fallback so a silent Bonferroni substitution never gets
                # miscounted as a max-T result.
                maxt_cis: dict = {}
                sim_cis, sim_method, _ = _simultaneous_cis_router(
                    scores=scores, results=matrix_raw.results, pairs=pairs, labels=labels,
                    method=multiarm_method, ci=ci, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
                    prefer="max_t",
                )
                if sim_method == "max_t":
                    maxt_cis = sim_cis

                agg_time[condition] += time.perf_counter() - _t0

                for method_name, cis in (("none", none_cis), ("bonferroni", bonf_cis), ("max_t", maxt_cis)):
                    if not cis:
                        continue
                    widths: list[float] = []
                    miss_distances: list[float] = []
                    covered_all = True
                    for (label_a, label_b) in pairs:
                        idx_a, idx_b = labels.index(label_a), labels.index(label_b)
                        true_diff = true_means[idx_a] - true_means[idx_b]
                        lo, hi = cis[(label_a, label_b)]
                        widths.append(hi - lo)
                        if true_diff < lo:
                            miss_distances.append(lo - true_diff)
                        elif true_diff > hi:
                            miss_distances.append(true_diff - hi)
                        else:
                            miss_distances.append(0.0)
                        if not (lo <= true_diff <= hi):
                            covered_all = False
                    # Family-wise interval score: mean width (still a legitimate
                    # per-comparison cost) + (2/alpha) * the WORST pair's miss
                    # distance, not the mean of each pair's own miss distance.
                    # interval_score() alone is a marginal, per-comparison proper
                    # score -- averaging it pair-by-pair rewards "none" (whose
                    # individual pairs are each ~nominally calibrated on their
                    # own) even when its family-wise coverage collapses, since
                    # the penalty never triggers on the same "did ANY pair miss"
                    # event that all_covered measures. Using max ties the miss
                    # penalty to that exact event: it's 0 iff all_covered, and
                    # positive iff at least one pair missed, so a method that
                    # buys family-wise coverage by widening every interval is no
                    # longer penalized as if it were miscalibrated per-pair.
                    family_score = float(np.mean(widths)) + (2.0 / alpha) * (max(miss_distances) if miss_distances else 0.0)
                    agg_width[(method_name, condition)] += float(np.mean(widths)) if widths else 0.0
                    agg_score[(method_name, condition)] += family_score
                    if covered_all:
                        agg_covered[(method_name, condition)] += 1

    return [
        SimultaneousCIResult(
            eval_type=source.eval_type, label=source.label, n=n, k=k_arms, ci_method=method_name,
            condition=condition, n_reps=n_reps, all_covered=agg_covered[(method_name, condition)],
            total_width=agg_width[(method_name, condition)], total_score=agg_score[(method_name, condition)],
            total_time=agg_time[condition],
        )
        for method_name in ci_methods
        for condition in ("null", "alt")
    ]


_SIMULTANEOUS_CI_SOURCES: list = []  # fork-inherited worker state for run_simultaneous_ci_simulation


def _run_simultaneous_ci_cell_worker(args: tuple) -> list[SimultaneousCIResult]:
    sc_idx, n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed = args
    return _run_simultaneous_ci_cell(
        _SIMULTANEOUS_CI_SOURCES[sc_idx], n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed,
    )


def run_simultaneous_ci_simulation(
    sources: list[MultiArmSource], sample_sizes: list[int], runs: int, k_values: list[int], n_reps: int,
    n_bootstrap: int, alpha: float, multiarm_method: str, statistic: str, progress_mode: str = "bar",
    seed: int = 42, n_workers: int = 1,
) -> list[SimultaneousCIResult]:
    global _SIMULTANEOUS_CI_SOURCES
    _SIMULTANEOUS_CI_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = _multiarm_style_cells(sources, sample_sizes, k_values)

    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, runs, k, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed)
                 for (sc_idx, n, k), seed in zip(cells, child_seeds)]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pvalues-simultaneous_ci")
    results: list[SimultaneousCIResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_simultaneous_ci_cell_worker(a))
            sc_idx, n, k = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} n={n} k={k}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_simultaneous_ci_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


def _time_stats_simultaneous_ci(results: list[SimultaneousCIResult]) -> tuple[float, float]:
    """Average ± SE of wall-clock time per rep in milliseconds across cells."""
    valid = [r for r in results if r.total_time > 0 and r.n_reps > 0]
    if not valid:
        return float("nan"), float("nan")
    per_rep_ms = [r.total_time * 1000.0 / r.n_reps for r in valid]
    avg = float(np.mean(per_rep_ms))
    se = float(np.std(per_rep_ms, ddof=1) / np.sqrt(len(per_rep_ms))) if len(per_rep_ms) > 1 else 0.0
    return avg, se


def print_simultaneous_ci_report(results: list[SimultaneousCIResult], alpha: float) -> None:
    target = 1.0 - alpha
    print(f"\n{'='*78}\n  PVALUES (SIMULTANEOUS CI) -- none vs. BONFERRONI vs. max-T\n"
          f"  Nominal family-wise coverage: {target:.0%}\n{'='*78}")
    ci_methods = [m.name for m in SIMULTANEOUS_CI_METHODS if m.name in {r.ci_method for r in results}]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    ks_present = sorted({r.k for r in results})

    for et in eval_types_present:
        for k in ks_present:
            subset = [r for r in results if r.eval_type == et and r.k == k]
            if not subset:
                continue
            print(f"\n  [{et}, k={k}]")
            print(f"    {'CI method':<12} {'Cov(null)':>10} {'Width(null)':>12} {'Score(null)':>12} {'Cov(alt)':>10} {'Width(alt)':>12} {'Score(alt)':>11}")
            for cm in ci_methods:
                c_rows = [r for r in subset if r.ci_method == cm]
                null_rows = [r for r in c_rows if r.condition == "null"]
                alt_rows = [r for r in c_rows if r.condition == "alt"]
                t_null = sum(r.n_reps for r in null_rows)
                c_null = sum(r.all_covered for r in null_rows)
                w_null = sum(r.total_width for r in null_rows) / t_null if t_null > 0 else float("nan")
                s_null = sum(r.total_score for r in null_rows) / t_null if t_null > 0 else float("nan")
                t_alt = sum(r.n_reps for r in alt_rows)
                c_alt = sum(r.all_covered for r in alt_rows)
                w_alt = sum(r.total_width for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
                s_alt = sum(r.total_score for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
                cov_null = c_null / t_null if t_null > 0 else float("nan")
                cov_alt = c_alt / t_alt if t_alt > 0 else float("nan")
                print(f"    {cm:<12} {cov_null:>10.3f} {w_null:>12.4f} {s_null:>12.4f} {cov_alt:>10.3f} {w_alt:>12.4f} {s_alt:>11.4f}")

    _print_simultaneous_overall_summary_table(
        "OVERALL SUMMARY (collapsed across eval types, sources, n, k)", results, ci_methods, target,
    )
    _print_simultaneous_overall_summary_table(
        "OVERALL SUMMARY -- LOW N (n <= 30)", [r for r in results if r.n <= 30], ci_methods, target,
    )
    _print_simultaneous_overall_summary_table(
        "OVERALL SUMMARY -- HIGH N (n >= 30)", [r for r in results if r.n >= 30], ci_methods, target,
    )


def _print_simultaneous_overall_summary_table(
    title: str, results: list[SimultaneousCIResult], ci_methods: list[str], target: float,
) -> None:
    """One OVERALL SUMMARY table for print_simultaneous_ci_report, over
    whatever subset of `results` the caller passes in (e.g. all of them, or
    just the low-N / high-N slice -- see that function's low-N vs. high-N
    split, which exists because max-T's bootstrap_t studentization is
    well-behaved at large N but develops a random-denominator instability at
    small N with many simultaneous pairs; pooling both regimes into one
    table hides exactly that crossover)."""
    ks_present = sorted({r.k for r in results})
    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    print(f"\n{'-'*72}\n  {title}\n{'-'*72}")
    print(f"  MinCov = worst per-scenario family-wise coverage seen for that CI method (not\n"
          f"  an average) -- flags methods whose good mean coverage hides an unreliable\n"
          f"  scenario/n/k cell.")
    n_cols = "".join(f"  {'n='+str(n):>9}" for n in sizes_present)
    k_cols = "".join(f"  {'k='+str(k):>8}" for k in ks_present)
    print(f"\n  {'CI method':<12}  {'Cov(null)':>9}  {'MinCov':>7}  {'Band95':>13}  {'Width(null)':>11}  {'Score(null)':>12}  "
          f"{'Cov(alt)':>8}  {'Width(alt)':>10}  {'Score(alt)':>11}  {'Time(ms)':>14}{n_cols}{k_cols}")
    for cm in ci_methods:
        c_rows = [r for r in results if r.ci_method == cm]
        null_rows = [r for r in c_rows if r.condition == "null"]
        alt_rows = [r for r in c_rows if r.condition == "alt"]
        t_null = sum(r.n_reps for r in null_rows)
        c_null = sum(r.all_covered for r in null_rows)
        if t_null == 0:
            continue
        w_null = sum(r.total_width for r in null_rows) / t_null if t_null > 0 else float("nan")
        s_null = sum(r.total_score for r in null_rows) / t_null if t_null > 0 else float("nan")
        t_alt = sum(r.n_reps for r in alt_rows)
        c_alt = sum(r.all_covered for r in alt_rows)
        w_alt = sum(r.total_width for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
        s_alt = sum(r.total_score for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
        cov_null = c_null / t_null if t_null > 0 else float("nan")
        cov_alt = c_alt / t_alt if t_alt > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(c_null, t_null)
        band = f"{lo:.3f}-{hi:.3f}" if np.isfinite(lo) else "-"
        avg_ms, se_ms = _time_stats_simultaneous_ci(null_rows)
        time_str = f"{avg_ms:.1f}+-{se_ms:.1f}" if np.isfinite(avg_ms) else "-"
        marker = "*" if np.isfinite(cov_null) and abs(cov_null - target) > 0.02 else " "
        per_label_cov = defaultdict(lambda: [0, 0])
        for r in null_rows:
            acc = per_label_cov[(r.eval_type, r.label)]
            acc[0] += r.all_covered
            acc[1] += r.n_reps
        label_rates = [c / t for c, t in per_label_cov.values() if t > 0]
        worst_cov = min(label_rates) if label_rates else float("nan")
        worst_str = f"{worst_cov:.3f}{'*' if np.isfinite(worst_cov) and abs(worst_cov - target) > 0.02 else ' '}" if np.isfinite(worst_cov) else "-"
        n_cells = ""
        for n in sizes_present:
            n_null = [r for r in null_rows if r.n == n]
            nc = sum(r.all_covered for r in n_null)
            nt = sum(r.n_reps for r in n_null)
            nf = nc / nt if nt > 0 else float("nan")
            n_cells += f"  {nf:>9.3f}" if np.isfinite(nf) else f"  {'  -':>9}"
        k_cells = ""
        for k in ks_present:
            k_null = [r for r in null_rows if r.k == k]
            kc = sum(r.all_covered for r in k_null)
            kt = sum(r.n_reps for r in k_null)
            kf = kc / kt if kt > 0 else float("nan")
            k_cells += f"  {kf:>8.3f}" if np.isfinite(kf) else f"  {'  -':>8}"
        print(f"  {cm:<12}  {cov_null:>8.3f}{marker}  {worst_str:>7}  {band:>13}  {w_null:>11.4f}  {s_null:>12.4f}  "
              f"{cov_alt:>8.3f}  {w_alt:>10.4f}  {s_alt:>11.4f}  {time_str:>14}{n_cells}{k_cells}")
    print(f"  (* = |coverage - nominal| > 0.02; narrower Width/Score at matching coverage is better)")
    print()


def latex_simultaneous_ci_overall_summary(results: list[SimultaneousCIResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-CI-method family-wise coverage
    (null, with its 95% MC band) + average width (null and alt), collapsed
    across eval types, plus one coverage column per sample size actually
    swept -- the headline evidence for max-T over the alternatives: `none`
    should visibly under-cover (no simultaneous adjustment at all), while
    Bonferroni and max-T should both hit nominal coverage, so the tie-breaker
    between those two is which one gets there with a narrower average CI."""
    target = 1.0 - alpha
    ci_methods = [m.name for m in SIMULTANEOUS_CI_METHODS if m.name in {r.ci_method for r in results}]
    eval_types_present = {et for et in EVAL_TYPES if any(r.eval_type == et for r in results)}
    sizes_present = sorted({r.n for r in results if r.condition == "null"})

    rows = []
    for cm in ci_methods:
        c_rows = [r for r in results if r.ci_method == cm]
        covered = {r.eval_type for r in c_rows}
        null_rows = [r for r in c_rows if r.condition == "null"]
        alt_rows = [r for r in c_rows if r.condition == "alt"]
        t_null = sum(r.n_reps for r in null_rows)
        c_null = sum(r.all_covered for r in null_rows)
        w_null = sum(r.total_width for r in null_rows) / t_null if t_null > 0 else float("nan")
        s_null = sum(r.total_score for r in null_rows) / t_null if t_null > 0 else float("nan")
        t_alt = sum(r.n_reps for r in alt_rows)
        c_alt = sum(r.all_covered for r in alt_rows)
        w_alt = sum(r.total_width for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
        s_alt = sum(r.total_score for r in alt_rows) / t_alt if t_alt > 0 else float("nan")
        cov_null = c_null / t_null if t_null > 0 else float("nan")
        cov_alt = c_alt / t_alt if t_alt > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(c_null, t_null)
        row = [
            escape_latex(cm),
            f"{cov_null:.3f}" if np.isfinite(cov_null) else "-",
            f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
            f"{w_null:.4f}" if np.isfinite(w_null) else "-",
            f"{s_null:.4f}" if np.isfinite(s_null) else "-",
            f"{cov_alt:.3f}" if np.isfinite(cov_alt) else "-",
            f"{w_alt:.4f}" if np.isfinite(w_alt) else "-",
            f"{s_alt:.4f}" if np.isfinite(s_alt) else "-",
            eval_type_label(covered, eval_types_present),
        ]
        for n in sizes_present:
            n_rows = [r for r in null_rows if r.n == n]
            c_n = sum(r.all_covered for r in n_rows)
            t_n = sum(r.n_reps for r in n_rows)
            cov_n = c_n / t_n if t_n > 0 else float("nan")
            row.append(f"{cov_n:.3f}" if np.isfinite(cov_n) else "-")
        rows.append(row)

    return booktabs_table(
        caption=f"pvalues (simultaneous CI): family-wise coverage, average per-comparison width, "
                f"and average per-comparison interval score, none vs. Bonferroni vs. max-T "
                f"(nominal coverage={target:.0%}).",
        label="tab:pvalues_simultaneous_ci_overall",
        columns=["CI method", "Cov(null)", "95\\% MC band", "Width(null)", "Score(null)",
                 "Cov(alt)", "Width(alt)", "Score(alt)", "Eval types"]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
    )


def save_results_artifacts_simultaneous_ci(
    *, results: list[SimultaneousCIResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False,
) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_simultaneous_ci_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_type", "label", "n", "k", "ci_method", "condition", "n_reps", "all_covered", "coverage_rate", "avg_width", "avg_score", "total_time_s", "time_ms_per_rep"])
        for r in results:
            time_ms = (r.total_time * 1000.0 / r.n_reps) if r.n_reps > 0 and r.total_time > 0 else float("nan")
            writer.writerow([
                r.eval_type, r.label, r.n, r.k, r.ci_method, r.condition, r.n_reps, r.all_covered,
                f"{r.all_covered / r.n_reps:.8f}", f"{r.total_width / r.n_reps:.8f}", f"{r.total_score / r.n_reps:.8f}",
                f"{r.total_time:.6f}", f"{time_ms:.4f}" if not (time_ms != time_ms) else "",
            ])
    summary_path = out_base / f"{run_stem}_simultaneous_ci_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_simultaneous_ci_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_simultaneous_ci_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_simultaneous_ci_coverage_width_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Coverage vs. width, one point per (scenario, CI method) per eval type
    (null condition) -- deliberately NOT one pooled dot per method, since a
    method's pooled-average width can look reasonable while individual
    scenario/n/k cells sit far from it (see max-T's random-denominator
    instability at small N + large k, evalstats.core.paired.
    _max_stat_simultaneous_cis's bootstrap_t branch): a single dot per
    method would average that away. Only plots Bonferroni and max-T
    (SIMULTANEOUS_CI_PLOT_METHODS excludes `none`, which sits so far below
    nominal coverage -- no simultaneous adjustment at all -- that it
    squashes the Bonferroni-vs-max-T comparison this plot exists to show;
    `none` is still in the printed/logged report tables and the CSV).
    Whichever cloud sits further left (narrower width) at matching
    coverage is the better default -- and stray points reveal exactly
    which scenarios don't follow that pattern."""
    import matplotlib.pyplot as plt

    target = 1.0 - alpha
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows=1, ncols=nrows, figsize=(5.0 * nrows, 5.0), squeeze=False)

    plot_method_names = {m.name for m in SIMULTANEOUS_CI_PLOT_METHODS}
    null_rows_all = [
        r for r in results
        if r.condition == "null" and r.n_reps > 0 and r.ci_method in plot_method_names
    ]
    df = pd.DataFrame([
        {
            "eval_type": r.eval_type, "label": r.label, "ci_method": r.ci_method,
            "coverage": r.all_covered / r.n_reps, "width": r.total_width / r.n_reps,
        }
        for r in null_rows_all
    ])
    scenario_level = (
        df.groupby(["eval_type", "label", "ci_method"], as_index=False).agg(
            coverage=("coverage", "mean"), width=("width", "mean"),
        )
        if not df.empty else df
    )

    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        ax.axhline(target, color="black", linestyle="--", linewidth=1.0)
        et_df = scenario_level[scenario_level["eval_type"] == et] if not scenario_level.empty else scenario_level
        for m in SIMULTANEOUS_CI_PLOT_METHODS:
            m_df = et_df[et_df["ci_method"] == m.name] if not et_df.empty else et_df
            if m_df.empty:
                continue
            ax.scatter(
                m_df["width"], m_df["coverage"], color=m.color, s=34, label=m.name,
                edgecolors="white", linewidths=0.5, alpha=0.75,
            )
        ax.set_xlabel("Average per-comparison CI width (null)")
        ax.set_ylabel("Family-wise coverage (null)")
        ax.set_title(f"eval type: {et}")
        # Zoom to the actual coverage spread (plus the nominal line) rather
        # than a fixed [0, 1] -- with `none` dropped from this plot, every
        # remaining point usually clusters near nominal, and a full [0, 1]
        # axis squashes that spread into an unreadable sliver at the top.
        if not et_df.empty:
            cov_vals = et_df["coverage"].tolist() + [target]
            lo, hi = min(cov_vals), max(cov_vals)
            pad = max(0.01, (hi - lo) * 0.15)
            ax.set_ylim(max(0.0, lo - pad), min(1.02, hi + pad))
        else:
            ax.set_ylim(0.0, 1.02)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "pvalues (simultaneous CI): coverage vs. width, Bonferroni vs. max-T\n"
        "(one dot = one scenario, averaged across n and k)",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_simultaneous_ci_coverage_width_vs_k_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Family-wise coverage and average width as a function of k (number of
    arms), one curve per CI method, collapsed across eval types and sample
    sizes -- mirrors save_multiarm_fwer_vs_k_plot. This is the direct
    picture of "pairwise comparisons grow as k(k-1)/2": Bonferroni's width
    should grow faster than max-T's, since Bonferroni's per-comparison
    budget (alpha/pairs) shrinks with the pair count while max-T's joint
    bootstrap doesn't pay that same tax. Only plots Bonferroni and max-T
    (SIMULTANEOUS_CI_PLOT_METHODS excludes `none`, whose coverage falling
    further below nominal as k grows is a different, already-obvious story
    that would squash this one on the same axes -- `none` is still in the
    printed/logged report tables and the CSV). Only produced when more
    than one k value was swept; returns out_path unchanged (without
    writing) if all results share the same k."""
    import matplotlib.pyplot as plt

    target = 1.0 - alpha
    ks_present = sorted({r.k for r in results})
    if len(ks_present) < 2:
        return out_path

    fig, (ax_cov, ax_width) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    ax_cov.axhline(target, color="black", linewidth=1.0, linestyle="--", label=f"nominal={target:.0%}")

    for m in SIMULTANEOUS_CI_PLOT_METHODS:
        c_rows = [r for r in results if r.ci_method == m.name]
        if not c_rows:
            continue
        xs, ys_cov, ys_width = [], [], []
        for k in ks_present:
            k_rows = [r for r in c_rows if r.k == k]
            null_rows = [r for r in k_rows if r.condition == "null"]
            t_null = sum(r.n_reps for r in null_rows)
            c_null = sum(r.all_covered for r in null_rows)
            w_null = sum(r.total_width for r in null_rows)
            if t_null == 0:
                continue
            xs.append(k)
            ys_cov.append(c_null / t_null)
            ys_width.append(w_null / t_null)
        if xs:
            ax_cov.plot(xs, ys_cov, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            ax_width.plot(xs, ys_width, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)

    ax_cov.set_xlabel("k (number of arms)")
    ax_cov.set_ylabel("Family-wise coverage (null)")
    ax_cov.set_title("Coverage vs. number of arms")
    ax_cov.set_ylim(0.0, 1.02)
    ax_cov.legend(fontsize=7)

    ax_width.set_xlabel("k (number of arms)")
    ax_width.set_ylabel("Average per-comparison CI width (null)")
    ax_width.set_title("Width vs. number of arms")
    ax_width.set_ylim(bottom=0.0)
    ax_width.legend(fontsize=7)

    fig.suptitle("pvalues (simultaneous CI): coverage and width vs. k", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_simultaneous_ci_reliability_violin_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Cross-scenario reliability: violin+strip of per-scenario family-wise
    coverage and average per-comparison interval score (null condition), one
    dot per (label, ci_method) -- the simultaneous-CI analogue of the
    pairwise/multi-arm reliability violins, and consistent with ci_single/
    ci_paired's reliability violin (coverage + interval score, not width).
    Exposes the spread the OVERALL SUMMARY table's pooled coverage hides: a
    method with nominal family-wise coverage on average can still miss badly
    on a specific scenario/k cell that pooling across labels masks. Only
    plots Bonferroni and max-T (`none` is dropped -- see
    SIMULTANEOUS_CI_PLOT_METHODS -- since it's so far below nominal
    coverage that it squashes the comparison this plot exists to show;
    it's still in the printed/logged report tables and the CSV)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - alpha
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    ci_methods = [m.name for m in SIMULTANEOUS_CI_PLOT_METHODS if m.name in {r.ci_method for r in results}]
    palette = {m.name: m.color for m in SIMULTANEOUS_CI_PLOT_METHODS}

    null_rows = [r for r in results if r.condition == "null" and r.n_reps > 0]
    df = pd.DataFrame([
        {
            "eval_type": r.eval_type, "label": r.label, "ci_method": r.ci_method,
            "coverage": r.all_covered / r.n_reps, "score": r.total_score / r.n_reps,
        }
        for r in null_rows
    ])
    scenario_level = (
        df.groupby(["eval_type", "label", "ci_method"], as_index=False).agg(
            coverage=("coverage", "mean"), score=("score", "mean"),
        )
        if not df.empty else df
    )

    n_cols = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
    for col_idx, et in enumerate(eval_types_present):
        et_df = scenario_level[scenario_level["eval_type"] == et] if not scenario_level.empty else scenario_level
        for row_idx, (metric, ylabel, ref_line) in enumerate([
            ("coverage", "Family-wise coverage per scenario", target),
            ("score", "Interval score per scenario", None),
        ]):
            ax = axes[row_idx][col_idx]
            if et_df.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue
            et_methods = [name for name in ci_methods if name in et_df["ci_method"].values]
            sns.violinplot(
                data=et_df, x="ci_method", y=metric, order=et_methods, hue="ci_method",
                hue_order=et_methods, palette=palette, cut=0, inner=None, linewidth=0.8,
                alpha=0.35, legend=False, ax=ax,
            )
            sns.stripplot(
                data=et_df, x="ci_method", y=metric, order=et_methods, hue="ci_method",
                hue_order=et_methods, palette=palette, size=4, alpha=0.7, jitter=0.25,
                linewidth=0.4, edgecolor="white", legend=False, ax=ax,
            )
            if ref_line is not None:
                ax.axhline(ref_line, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper() if row_idx == 0 else "")
            ax.tick_params(axis="x", rotation=45)
            for tick_label in ax.get_xticklabels():
                tick_label.set_ha("right")

    fig.suptitle(
        f"Cross-Scenario Reliability (one dot = one scenario)\npvalues simultaneous CI | alpha={alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_simultaneous_ci_violin_vs_n_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Grouped violin plots of family-wise coverage and interval score vs.
    sample size n (null condition), one violin per CI method at each n
    (dodged side by side), faceted by eval type -- the Bonferroni/max-T
    analogue of ci_paired.py's --violin-plot (tango_score vs. tango_scc vs.
    bayes_paired_comp vs. N).

    Each violin pools every (scenario, k) cell at that n rather than
    averaging k away: the small-N/large-k interaction is exactly what
    widens these violins and drags their tails at small n (max-T's
    random-denominator instability in the studentized-bootstrap-t branch of
    evalstats.core.paired._max_stat_simultaneous_cis -- resampling just n
    points to re-estimate a per-replicate SE gets noisy at small n, and
    taking a max over k(k-1)/2 simultaneous pairs multiplies the chances of
    hitting a near-zero denominator on any given replicate), so collapsing
    across k here would hide the very thing this plot exists to show.

    Only plots Bonferroni and max-T (`none` is dropped -- see
    SIMULTANEOUS_CI_PLOT_METHODS -- since it's so far below nominal
    coverage that it squashes the comparison this plot exists to show;
    it's still in the printed/logged report tables and the CSV).
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns

    target = 1.0 - alpha
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    ci_methods = [m.name for m in SIMULTANEOUS_CI_PLOT_METHODS if m.name in {r.ci_method for r in results}]
    palette = {m.name: m.color for m in SIMULTANEOUS_CI_PLOT_METHODS}

    null_rows = [r for r in results if r.condition == "null" and r.n_reps > 0]
    df = pd.DataFrame([
        {
            "eval_type": r.eval_type, "label": r.label, "k": r.k, "n": r.n, "ci_method": r.ci_method,
            "coverage": r.all_covered / r.n_reps, "score": r.total_score / r.n_reps,
        }
        for r in null_rows
    ])

    n_cols = max(len(eval_types_present), 1)
    if df.empty:
        fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5), squeeze=False)
        for ax_row in axes:
            for ax in ax_row:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    ns_present = sorted(df["n"].unique())
    n_order = [str(n) for n in ns_present]
    df["n_label"] = df["n"].astype(str)

    col_width = 1.3 * len(ns_present) + 2.5
    fig, axes = plt.subplots(2, n_cols, figsize=(col_width * n_cols, 9.0), squeeze=False)
    legend_handles = [mpatches.Patch(facecolor=palette[m], alpha=0.5, label=m) for m in ci_methods]

    for col_idx, et in enumerate(eval_types_present):
        et_df = df[df["eval_type"] == et]
        et_methods = [name for name in ci_methods if name in et_df["ci_method"].values]
        for row_idx, (metric, ylabel, ref_line) in enumerate([
            ("coverage", "Family-wise coverage", target),
            ("score", "Interval score", None),
        ]):
            ax = axes[row_idx][col_idx]
            if et_df.empty or not et_methods:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue
            sns.violinplot(
                data=et_df, x="n_label", y=metric, order=n_order, hue="ci_method", hue_order=et_methods,
                palette=palette, cut=0, inner="quartile", linewidth=0.7, dodge=True, alpha=0.35,
                legend=False, ax=ax,
            )
            sns.stripplot(
                data=et_df, x="n_label", y=metric, order=n_order, hue="ci_method", hue_order=et_methods,
                palette=palette, size=3, alpha=0.5, jitter=0.2, dodge=True, linewidth=0.3,
                edgecolor="white", legend=False, ax=ax,
            )
            if ref_line is not None:
                ax.axhline(ref_line, linestyle="--", color="tab:cyan", linewidth=1.2, zorder=0)
            ax.set_xlabel("n" if row_idx == 1 else "")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(et.upper() if row_idx == 0 else "")

    axes[0][-1].legend(
        handles=legend_handles, title="CI method", fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )

    fig.suptitle(
        "Coverage / Interval Score vs. Sample Size (null)\n"
        f"pvalues simultaneous CI | alpha={alpha} | each violin pools all k and scenarios at that n",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PPI mode: Type-I error calibration for evalstats.tests' PPI-corrected
# wrappers under judge bias/miscalibration, ported from
# sim_type_i_calibration.py's _run_one. Calls evalstats.tests' internal PPI
# functions directly (the same functions back the public es.tests.* API) to
# skip validate_alignment overhead, exactly as the legacy script does.
# ---------------------------------------------------------------------------


@dataclass
class PPIResult:
    name: str
    tag: str
    test: str
    n_reps: int
    corrected_rejects: int
    uncorrected_rejects: int
    n_failed: int = 0
    n: int = 0
    """Group/condition-A sample size for this scenario (JudgeBiasSource.n).
    Only the 'sample_size' tag actually sweeps this (n=60/100/200/400);
    every other scenario uses the fixed baseline -- see
    latex_ppi_overall_summary's per-n columns, which are sourced from that
    one tag rather than every scenario (most of which share n=100)."""


@dataclass
class PPIEffectResult:
    """Bias and CI-coverage summary for one (scenario, test) cell's
    PPI-corrected point estimate -- complements PPIResult's Type-I check
    (does the p-value stay calibrated) with: is the estimate itself centered
    at the truth, and does its CI cover that truth at the nominal rate?
    Ported from sim_type_i_calibration.py's effect_results/_gold_null_values
    check; see run_ppi_effect_check."""
    name: str
    tag: str
    test: str
    n: int
    n_samples: int
    """Number of successful (non-failed) bootstrap draws this cell's stats are based on."""
    null_value: float
    """Monte Carlo gold-reference null value this estimate is compared against
    (not always 0 -- see estimate_judge_bias_gold_null_values)."""
    mean_bias: float
    """mean(estimate - null_value) across draws."""
    bias_z: float
    """mean_bias / SE(mean_bias) -- a |z| > 3 flags a real (not just noisy) bias."""
    coverage: float
    """Fraction of draws whose CI contains null_value."""
    mean_ci_width: float
    uncorrected_bias_z: float
    """Same z-score, but for the RAW (pre-PPI) LLM-only estimate -- contrast
    for how much PPI correction actually reduced bias."""


def _uncorrected_anova_independent_p_value(groups: list[np.ndarray]) -> float:
    return float(scipy_stats.f_oneway(*groups).pvalue)


def _uncorrected_anova_repeated_p_value(groups: list[np.ndarray]) -> float:
    from statsmodels.stats.anova import AnovaRM

    k = len(groups)
    n_subjects = len(groups[0])
    stacked = np.column_stack(groups)
    df_long = pd.DataFrame({
        "subject": np.repeat(np.arange(n_subjects), k),
        "condition": np.tile(np.arange(k), n_subjects),
        "score": stacked.reshape(-1),
    })
    rm = AnovaRM(df_long, depvar="score", subject="subject", within=["condition"]).fit()
    return float(rm.anova_table.iloc[0]["Pr > F"])


def _uncorrected_friedman_p_value(groups: list[np.ndarray]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(scipy_stats.friedmanchisquare(*groups).pvalue)


def _uncorrected_kruskal_p_value(groups: list[np.ndarray]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(scipy_stats.kruskal(*groups).pvalue)


def _uncorrected_bayes_bootstrap_paired_p_value(diffs: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    """LLM-only (uncorrected) Bayesian-bootstrap two-sided p-value for
    H0: mean(diffs) = 0 -- the same Dirichlet-weighted resampling
    evalstats.core.paired's 'bayes_bootstrap' method uses, applied directly
    (no PPI correction) as the baseline _ppi_paired_bayes_bootstrap's
    corrected version is compared against."""
    boots = bayes_bootstrap_means_1d(diffs, n_boot, rng, statistic="mean")
    p = float(2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0)))
    return min(max(p, 0.0), 1.0)


def _uncorrected_bootstrap_t_paired_p_value(diffs: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    """LLM-only (uncorrected) studentized-bootstrap two-sided p-value for
    H0: mean(diffs) = 0 -- same pivot construction as
    evalstats.core.resampling.bootstrap_t_ci_1d (SE = std/sqrt(n) per
    replicate), applied directly (no PPI correction) as the baseline
    _ppi_paired_bootstrap_t's corrected version is compared against."""
    n = len(diffs)
    theta_hat = float(np.mean(diffs))
    se_hat = float(np.std(diffs, ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    if not np.isfinite(se_hat) or se_hat <= 0.0:
        return 1.0
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = diffs[idx]
    boot_theta = samples.mean(axis=1)
    boot_se = np.std(samples, ddof=1, axis=1) / np.sqrt(n)
    valid = np.isfinite(boot_se) & (boot_se > 0.0)
    if not np.any(valid):
        return 1.0
    t_stats = (boot_theta[valid] - theta_hat) / boot_se[valid]
    t_obs = theta_hat / se_hat
    p = float(2.0 * min(np.mean(t_stats <= t_obs), np.mean(t_stats >= t_obs)))
    return min(max(p, 0.0), 1.0)


def _uncorrected_tango_paired_p_value(diffs: np.ndarray) -> float:
    """LLM-only (uncorrected) two-sided p-value for H0: mean(diffs) = 0,
    using the SAME per-item variance evalstats.core.resampling.
    tango_paired_ci's score interval is built from (V_hat = Var(diffs,
    ddof=0) / n, i.e. (n10+n01)/n^2 - (n10-n01)^2/n^3 for binary diffs) --
    applied directly (no PPI correction) as the baseline
    _ppi_paired_tango's corrected version is compared against. Closed-form,
    no bootstrap needed."""
    n = len(diffs)
    d_hat = float(np.mean(diffs))
    v_hat = float(np.mean((diffs - d_hat) ** 2)) / n if n > 0 else 0.0
    if v_hat <= 0.0 or not np.isfinite(v_hat):
        return 1.0
    z_obs = d_hat / np.sqrt(v_hat)
    p = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(z_obs))))
    return min(max(p, 0.0), 1.0)


def _lmm_wald_f_pvalue_from_fit(sm_result, k: int) -> float:
    """Wald-to-F omnibus p-value for template fixed effects, given an
    already-fitted MixedLM result (see _fit_lmm_general).

    Factored out of _uncorrected_lmm_p_value so callers that separately need
    the SAME LLM-only fit for the PPI correction (_ppi_lmm_p_value, via
    precomputed_fit=) can reuse it instead of fitting the identical model
    twice -- MixedLM's iterative MLE fit is by far the dominant cost of the
    ppi mode's lmm/lmm_factorial/lmm_runs tests (profiled at ~70% of total
    runtime), so this halves their cost.
    """
    beta = sm_result.fe_params.to_numpy()
    cov = _get_fe_vcov_sm(sm_result)
    df1 = k - 1
    df2 = float(sm_result.df_resid)
    beta_t, cov_t = beta[1:], cov[1:, 1:]
    wald = float(beta_t @ np.linalg.solve(cov_t, beta_t))
    f_stat = wald / df1
    return float(scipy_stats.f.sf(f_stat, df1, df2)) if f_stat > 0 else 1.0


def _uncorrected_lmm_p_value(groups: list[np.ndarray], factors=None) -> float:
    """Uncorrected (LLM-only) Wald F-test for score ~ <fixed factors> + (1|input),
    fit via statsmodels MixedLM (REML); _fit_lmm_general handles single-factor,
    multi-factor, and nested-run groups alike."""
    k = len(groups)
    template_labels = [f"T{i}" for i in range(k)]
    sm_result, _df_full, _x_row, _r = _fit_lmm_general(groups, template_labels, factors)
    return _lmm_wald_f_pvalue_from_fit(sm_result, k)


_ALPHA = ALPHA_DEFAULT

# Binary judge-bias data only supports the mean-based tests (a proportion is
# just the mean of a 0/1 variable, so PPI's rectifier applies unchanged --
# see scenarios.synthetic's binary judge-bias comment). The rank-based
# family (mw/wilcoxon/friedman/kruskal) and ANOVA/LMM assume a scale that
# doesn't hold up under binary's massive ties, and generate_judge_bias_cell
# doesn't extend its additive noise/bias/slope judge model to a 0/1
# judgment for those structures either.
_PPI_BINARY_COMPATIBLE_TESTS = {TTEST.name, TTEST_WELCH.name, PAIRED_T.name, BAYES_BOOTSTRAP.name, TANGO.name}

# The mirror-image restriction: tests whose estimand/formula is specific to
# paired BINARY data (Tango's discordant-pair-rate score interval, with a
# continuity correction that only makes sense for a discrete difference) and
# so should be excluded everywhere else, the same way BOOTSTRAP_T (numeric-
# only, see its Method-registry comment) is excluded FROM binary.
_PPI_BINARY_ONLY_TESTS = {TANGO.name}


def _ppi_effective_tests(sc: JudgeBiasSource, active_tests: list[str]) -> list[str]:
    """Restrict active_tests to what this scenario's eval_type actually
    supports: binary scenarios only run _PPI_BINARY_COMPATIBLE_TESTS;
    non-binary scenarios run everything except _PPI_BINARY_ONLY_TESTS."""
    if sc.eval_type == "binary":
        return [t for t in active_tests if t in _PPI_BINARY_COMPATIBLE_TESTS]
    return [t for t in active_tests if t not in _PPI_BINARY_ONLY_TESTS]


def _run_ppi_cell(
    sc: JudgeBiasSource, active_tests: list[str], n_reps: int, n_boot: int, seed,
    progress_dict=None, progress_key: str | None = None,
) -> list[PPIResult]:
    """Run all n_reps reps for one JudgeBiasSource.

    progress_dict / progress_key : optional
        When given, ``progress_dict[progress_key]`` is updated to ``(rep,
        n_reps)`` periodically (rate-limited to ~2/sec) as this cell runs.
        Lets a caller peek at a long-running cell's rep-level progress
        instead of it looking stalled until the whole cell returns -- some
        scenarios (large sample size, or hard-to-converge LMM fits) can
        take minutes on their own; see run_ppi_simulation's in-flight
        reporter thread. ``progress_dict`` may be a plain dict (serial
        mode) or a multiprocessing.Manager().dict() proxy (parallel mode)
        -- both support the same __setitem__ interface, so this function
        doesn't need to know which.
    """
    active_tests = _ppi_effective_tests(sc, active_tests)
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in active_tests}
    uncorrected: dict[str, int] = {t: 0 for t in active_tests}
    failed: dict[str, int] = {t: 0 for t in active_tests}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    _last_progress_t = 0.0
    for _rep_i in range(n_reps):
        cell = generate_judge_bias_cell(sc, rng)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if TTEST.name in active_tests:
                try:
                    p_u = float(scipy_stats.ttest_ind(cell.llm_a2, cell.llm_b2, equal_var=True).pvalue)
                    uncorrected[TTEST.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    corrected[TTEST.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[TTEST.name] += 1

            if TTEST_WELCH.name in active_tests:
                try:
                    p_u = float(scipy_stats.ttest_ind(cell.llm_a2, cell.llm_b2, equal_var=False).pvalue)
                    uncorrected[TTEST_WELCH.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    corrected[TTEST_WELCH.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[TTEST_WELCH.name] += 1

            if MW.name in active_tests:
                try:
                    p_u = float(scipy_stats.mannwhitneyu(cell.llm_a2, cell.llm_b2, alternative="two-sided").pvalue)
                    uncorrected[MW.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda xa, ya: _p_x_gt_y_midrank(xa, ya) - 0.5, _ALPHA, n_boot, _rng_seed())
                    corrected[MW.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[MW.name] += 1

            if WILCOXON.name in active_tests:
                try:
                    p_u = float(scipy_stats.wilcoxon(cell.llm_x, cell.llm_y, alternative="two-sided").pvalue)
                    uncorrected[WILCOXON.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_arrays(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, np.median, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.mean)
                    corrected[WILCOXON.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[WILCOXON.name] += 1

            if PAIRED_T.name in active_tests:
                try:
                    p_u = float(scipy_stats.ttest_rel(cell.llm_x, cell.llm_y).pvalue)
                    uncorrected[PAIRED_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_arrays(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, np.mean, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.mean)
                    corrected[PAIRED_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[PAIRED_T.name] += 1

            if BAYES_BOOTSTRAP.name in active_tests:
                try:
                    p_u = _uncorrected_bayes_bootstrap_paired_p_value(cell.llm_x - cell.llm_y, n_boot, np.random.default_rng(_rng_seed()))
                    uncorrected[BAYES_BOOTSTRAP.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bayes_bootstrap(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, n_boot, _rng_seed())
                    corrected[BAYES_BOOTSTRAP.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[BAYES_BOOTSTRAP.name] += 1

            if BOOTSTRAP_T.name in active_tests:
                try:
                    p_u = _uncorrected_bootstrap_t_paired_p_value(cell.llm_x - cell.llm_y, n_boot, np.random.default_rng(_rng_seed()))
                    uncorrected[BOOTSTRAP_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bootstrap_t(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, n_boot, _rng_seed())
                    corrected[BOOTSTRAP_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[BOOTSTRAP_T.name] += 1

            if TANGO.name in active_tests:
                try:
                    p_u = _uncorrected_tango_paired_p_value(cell.llm_x - cell.llm_y)
                    uncorrected[TANGO.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_tango(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA)
                    corrected[TANGO.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[TANGO.name] += 1

            if ANOVA_IND.name in active_tests:
                try:
                    groups_ind = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_ind_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    p_u = _uncorrected_anova_independent_p_value(groups_ind)
                    uncorrected[ANOVA_IND.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_independent_p_value(groups_ind, groups_ind_lab, k=len(groups_ind))
                    corrected[ANOVA_IND.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[ANOVA_IND.name] += 1

            if ANOVA_REP.name in active_tests:
                try:
                    groups_rep = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_rep_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    p_u = _uncorrected_anova_repeated_p_value(groups_rep)
                    uncorrected[ANOVA_REP.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_repeated_p_value(groups_rep, groups_rep_lab, k=len(groups_rep))
                    corrected[ANOVA_REP.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[ANOVA_REP.name] += 1

            if FRIEDMAN.name in active_tests:
                try:
                    groups_fr = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_fr_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    p_u = _uncorrected_friedman_p_value(groups_fr)
                    uncorrected[FRIEDMAN.name] += int(p_u < _ALPHA)
                    p = _ppi_friedman_p_value(groups_fr, groups_fr_lab, k=len(groups_fr))
                    corrected[FRIEDMAN.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[FRIEDMAN.name] += 1

            if KRUSKAL.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    p_u = _uncorrected_kruskal_p_value(groups_kw)
                    uncorrected[KRUSKAL.name] += int(p_u < _ALPHA)
                    pw = _ppi_kruskal_wallis_pairwise(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    corrected[KRUSKAL.name] += int(pw["wald_p"] < _ALPHA)
                except Exception:
                    failed[KRUSKAL.name] += 1

            if LMM.name in active_tests:
                try:
                    groups_lmm = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_lmm_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    k = len(groups_lmm)
                    # Fit once, reuse for both the uncorrected Wald F-test and
                    # the PPI correction (which needs the identical LLM-only
                    # fit as its nuisance-parameter/reference point) -- see
                    # _lmm_wald_f_pvalue_from_fit's docstring.
                    fit = _fit_lmm_general(groups_lmm, [f"T{i}" for i in range(k)])
                    p_u = _lmm_wald_f_pvalue_from_fit(fit[0], k)
                    uncorrected[LMM.name] += int(p_u < _ALPHA)
                    p = _ppi_lmm_p_value(groups_lmm, groups_lmm_lab, k=k, precomputed_fit=fit)
                    corrected[LMM.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[LMM.name] += 1

            if LMM_FACTORIAL.name in active_tests:
                try:
                    groups_lf = [cell.llm_W, cell.llm_X, cell.llm_Y, cell.llm_Z]
                    groups_lf_lab = [cell.lab_W, cell.lab_X, cell.lab_Y, cell.lab_Z]
                    k = len(groups_lf)
                    fit = _fit_lmm_general(groups_lf, [f"T{i}" for i in range(k)], JUDGE_BIAS_LMM_FACTORIAL_FACTORS)
                    p_u = _lmm_wald_f_pvalue_from_fit(fit[0], k)
                    uncorrected[LMM_FACTORIAL.name] += int(p_u < _ALPHA)
                    p = _ppi_lmm_p_value(
                        groups_lf, groups_lf_lab, k=k, factors=JUDGE_BIAS_LMM_FACTORIAL_FACTORS, precomputed_fit=fit,
                    )
                    corrected[LMM_FACTORIAL.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[LMM_FACTORIAL.name] += 1

            if LMM_RUNS.name in active_tests:
                try:
                    groups_runs = [cell.llm_A_runs, cell.llm_B_runs, cell.llm_C_runs]
                    groups_runs_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    k = len(groups_runs)
                    fit = _fit_lmm_general(groups_runs, [f"T{i}" for i in range(k)])
                    p_u = _lmm_wald_f_pvalue_from_fit(fit[0], k)
                    uncorrected[LMM_RUNS.name] += int(p_u < _ALPHA)
                    p = _ppi_lmm_p_value(groups_runs, groups_runs_lab, k=k, precomputed_fit=fit)
                    corrected[LMM_RUNS.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[LMM_RUNS.name] += 1

        if progress_dict is not None:
            _now = time.time()
            if _now - _last_progress_t >= 0.5 or _rep_i + 1 == n_reps:
                progress_dict[progress_key] = (_rep_i + 1, n_reps)
                _last_progress_t = _now

    return [
        PPIResult(
            name=sc.name, tag=sc.tag, test=t, n_reps=n_reps,
            corrected_rejects=corrected[t], uncorrected_rejects=uncorrected[t], n_failed=failed[t],
            n=sc.n,
        )
        for t in active_tests
    ]


def _ppi_in_flight_line(progress_dict, done_keys: set) -> str | None:
    """Format a one-line snapshot of currently in-progress (i.e. reporting
    rep-level progress but not yet returned) ppi cells, or None if there's
    nothing worth showing. Factored out of _run_in_flight_reporter so it's
    independently testable."""
    snapshot = dict(progress_dict)
    active = {name: rep_total for name, rep_total in snapshot.items() if name not in done_keys}
    if not active:
        return None
    parts = [
        f"{name}: {rep}/{total} ({100.0 * rep / total:.0f}%)"
        for name, (rep, total) in sorted(active.items())
    ]
    return "  [in-flight] " + "  |  ".join(parts)


def _run_in_flight_reporter(progress_dict, done_keys: set, done_lock, stop_event, interval: float = 8.0) -> None:
    """Background-thread body for run_ppi_simulation's parallel path.

    Some ppi scenarios (large sample size, or hard-to-converge LMM fits --
    see cases/pvalues.py module docstring / harness README) can take
    several minutes on their own; with imap_unordered, the main progress
    bar only advances when a WHOLE cell returns, so a long cell looks
    identical to a hang from the outside. This periodically prints a
    snapshot of every currently in-flight cell's rep-level progress
    (populated by _run_ppi_cell's progress_dict writes) so it's visible
    that work is still happening, and roughly how far along it is.
    """
    while not stop_event.wait(interval):
        with done_lock:
            done_snapshot = set(done_keys)
        line = _ppi_in_flight_line(progress_dict, done_snapshot)
        if line:
            print(f"\n{line}", flush=True)


def run_ppi_simulation(
    sources: list[JudgeBiasSource], active_tests: list[str], n_reps: int, n_boot: int,
    progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
) -> list[PPIResult]:
    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(sources))]

    reporter = _ProgressReporter(len(sources), mode=progress_mode, label="pvalues-ppi")
    results: list[PPIResult] = []

    if n_workers <= 1:
        for i, (sc, child_seed) in enumerate(zip(sources, child_seeds)):
            results.extend(_run_ppi_cell(sc, active_tests, n_reps, n_boot, child_seed))
            reporter.update(i + 1, detail=f"{sources[i].name}")
        reporter.update(len(sources), detail="done")
        return results

    # Parallel path: a shared Manager dict lets each worker report rep-level
    # progress while it's mid-cell (see _run_ppi_cell's progress_dict), and
    # a background thread periodically prints an in-flight snapshot -- see
    # _run_in_flight_reporter's docstring for why this matters here
    # specifically (long individual cells + imap_unordered's coarse
    # per-cell-only progress signal).
    manager = _mp.Manager()
    progress_dict = manager.dict()
    args_list = [
        (sc, active_tests, n_reps, n_boot, child_seed, progress_dict)
        for sc, child_seed in zip(sources, child_seeds)
    ]

    done_keys: set = set()
    done_lock = threading.Lock()
    stop_event = threading.Event()
    reporter_thread = None
    if progress_mode != "off":
        reporter_thread = threading.Thread(
            target=_run_in_flight_reporter,
            args=(progress_dict, done_keys, done_lock, stop_event),
            daemon=True,
        )
        reporter_thread.start()

    try:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_ppi_cell_worker, args_list)):
                if cell_results:
                    with done_lock:
                        done_keys.add(cell_results[0].name)
                results.extend(cell_results)
                reporter.update(i + 1)
    finally:
        stop_event.set()
        if reporter_thread is not None:
            reporter_thread.join(timeout=1.0)

    reporter.update(len(sources), detail="done")
    return results


# ---------------------------------------------------------------------------
# PPI mode, effect-size calibration: bias and CI coverage of the PPI-
# corrected point estimate itself, complementing run_ppi_simulation's Type-I
# check (does the p-value stay calibrated). Ported from
# sim_type_i_calibration.py's effect_results/_gold_null_values check. lmm/
# lmm_factorial/lmm_runs are intentionally excluded (same as the legacy
# script): their headline estimand is a quadratic form in the fixed effects
# with no valid CI by design -- see es.tests.lmm()'s docstring.
# ---------------------------------------------------------------------------

_PPI_EFFECT_TESTS = (
    TTEST.name, TTEST_WELCH.name, MW.name, WILCOXON.name, PAIRED_T.name, BAYES_BOOTSTRAP.name, BOOTSTRAP_T.name,
    TANGO.name, ANOVA_IND.name, ANOVA_REP.name, FRIEDMAN.name, KRUSKAL.name,
)

# bayes_bootstrap/bootstrap_t/tango_score are excluded from the main ppi
# Type-I/effect plots and reported in a separate plot instead: they read
# differently to reviewers than the rest of PPI_TEST_METHODS (which are all
# textbook tests -- t-test, Wilcoxon, ANOVA, Friedman, Kruskal, LMM). These
# three are bootstrap/CI-based constructions (Bayesian bootstrap, studentized
# bootstrap, Tango's score interval) that would read as unfamiliar or
# confusing mixed in with the standard-methods plot -- tango_score
# specifically is fundamentally a CI construction for binary paired
# differences (see evalstats.tests._ppi_paired_tango), not a p-value test in
# its own right, and it's also the only one of the three restricted to a
# single binary scenario (_PPI_BINARY_ONLY_TESTS) rather than swept across
# the full catalog, so it would look sparse/broken next to tests with ~44x
# more scenarios' worth of points.
_PPI_NONSTANDARD_TESTS = {BAYES_BOOTSTRAP.name, BOOTSTRAP_T.name, TANGO.name}


def _ppi_tests_present(results, *, nonstandard: bool) -> list[str]:
    """Test names present in results, in PPI_TEST_METHODS' canonical order,
    filtered to the standard (textbook) subset or the nonstandard
    (bootstrap/CI-based) subset -- see _PPI_NONSTANDARD_TESTS."""
    present = {r.test for r in results}
    if nonstandard:
        return [m.name for m in PPI_TEST_METHODS if m.name in present and m.name in _PPI_NONSTANDARD_TESTS]
    return [m.name for m in PPI_TEST_METHODS if m.name in present and m.name not in _PPI_NONSTANDARD_TESTS]


def _run_ppi_effect_cell(
    sc: JudgeBiasSource, active_tests: list[str], n_reps: int, n_boot: int, seed,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Draw n_reps fresh replicates and capture each active effect-check
    test's PPI-corrected (estimate, ci_low, ci_high, llm_estimate) per rep.

    Runs as its OWN dedicated pass (with its own, typically much smaller,
    --effect-reps count) rather than piggybacking on run_ppi_simulation's
    Type-I sweep the way sim_type_i_calibration.py's _run_one does for its
    "free" tests -- this keeps _run_ppi_cell's Type-I return type/call site
    completely unchanged, at the cost of redrawing ttest/ttest_welch/mw/
    wilcoxon/kruskal's bootstrap a second time (cheap at the smaller
    effect-reps count this is meant to run at). anova_ind/anova_rep/friedman
    call the bootstrap-based SCALAR-estimate functions here (_ppi_anova_
    independent/_ppi_anova_repeated/_ppi_friedman) instead of the closed-form
    p-value-only ones run_ppi_simulation uses, since only the former carry an
    estimate/CI -- the same reason sim_type_i_calibration.py's anova family
    needed a separate pass (_run_one_effect_anova) too.
    """
    active_tests = _ppi_effective_tests(sc, active_tests)
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        cell = generate_judge_bias_cell(sc, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if TTEST.name in active_tests:
                try:
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    out[TTEST.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if TTEST_WELCH.name in active_tests:
                try:
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    out[TTEST_WELCH.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if MW.name in active_tests:
                try:
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda xa, ya: _p_x_gt_y_midrank(xa, ya) - 0.5, _ALPHA, n_boot, _rng_seed())
                    out[MW.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if WILCOXON.name in active_tests:
                try:
                    r = _ppi_paired_arrays(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, np.median, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.mean)
                    out[WILCOXON.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PAIRED_T.name in active_tests:
                try:
                    r = _ppi_paired_arrays(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, np.mean, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.mean)
                    out[PAIRED_T.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if BAYES_BOOTSTRAP.name in active_tests:
                try:
                    r = _ppi_paired_bayes_bootstrap(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, n_boot, _rng_seed())
                    out[BAYES_BOOTSTRAP.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if BOOTSTRAP_T.name in active_tests:
                try:
                    r = _ppi_paired_bootstrap_t(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA, n_boot, _rng_seed())
                    out[BOOTSTRAP_T.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if TANGO.name in active_tests:
                try:
                    r = _ppi_paired_tango(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, _ALPHA)
                    out[TANGO.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if ANOVA_IND.name in active_tests:
                try:
                    r = _ppi_anova_independent([cell.llm_a3, cell.llm_b3, cell.llm_c3], [cell.lab_a3, cell.lab_b3, cell.lab_c3], _ALPHA, n_boot, _rng_seed())
                    out[ANOVA_IND.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if ANOVA_REP.name in active_tests:
                try:
                    r = _ppi_anova_repeated([cell.llm_A, cell.llm_B, cell.llm_C], [cell.lab_A, cell.lab_B, cell.lab_C], _ALPHA, n_boot, _rng_seed())
                    out[ANOVA_REP.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if FRIEDMAN.name in active_tests:
                try:
                    r = _ppi_friedman([cell.llm_A, cell.llm_B, cell.llm_C], [cell.lab_A, cell.lab_B, cell.lab_C], _ALPHA, n_boot, _rng_seed())
                    out[FRIEDMAN.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if KRUSKAL.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    pw = _ppi_kruskal_wallis_pairwise(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    llm_theta = _kw_pairwise_thetas(groups_kw, pw["pairs"])
                    out[KRUSKAL.name].append((
                        float(np.mean(pw["theta_hat"])), float(np.mean(pw["ci_lo"])),
                        float(np.mean(pw["ci_hi"])), float(np.mean(llm_theta)),
                    ))
                except Exception:
                    pass

    return dict(out)


def _effect_cell_stats(
    samples: list[tuple[float, float, float, float]], null_val: float,
) -> tuple[float, float, float, float, int]:
    """(mean_bias, z, coverage_rate, mean_ci_width, n) for one (scenario,
    test) cell -- ported from sim_type_i_calibration.py's helper of the same
    name."""
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    estimates = np.array([s[0] for s in samples]) - null_val
    contains = np.array([(s[1] <= null_val <= s[2]) for s in samples])
    ci_widths = np.array([s[2] - s[1] for s in samples])
    bias_mean = float(estimates.mean())
    bias_se = float(estimates.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = bias_mean / bias_se if bias_se and bias_se > 0 else float("nan")
    coverage = float(contains.mean())
    mean_ci_width = float(ci_widths.mean())
    return bias_mean, z, coverage, mean_ci_width, n


def _uncorrected_bias_z(samples: list[tuple[float, float, float, float]], null_val: float) -> float:
    """z-score of the RAW (pre-PPI) LLM-only estimate's bias -- a contrast for
    how much PPI correction reduced bias. No CI exists for the raw estimate,
    so this is bias only, not coverage."""
    n = len(samples)
    if n < 2:
        return float("nan")
    raw = np.array([s[3] for s in samples]) - null_val
    se = float(raw.std(ddof=1) / np.sqrt(n))
    return float(raw.mean() / se) if se > 0 else float("nan")


def run_ppi_effect_check(
    sources: list[JudgeBiasSource], active_tests: list[str], n_reps: int, n_boot: int,
    gold_null_mc: int = 3000, progress_mode: str = "bar", seed: int = 44, n_workers: int = 1,
) -> list[PPIEffectResult]:
    """Bias and CI-coverage check for the PPI-corrected point estimate itself.

    Complements run_ppi_simulation's Type-I check ("does the p-value stay
    calibrated") with "is the estimate centered at the truth, and does its CI
    cover that truth at the nominal rate" -- ported from
    sim_type_i_calibration.py's second check.
    """
    effect_tests = [t for t in active_tests if t in _PPI_EFFECT_TESTS]
    if not effect_tests:
        return []
    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(sources))]

    reporter = _ProgressReporter(len(sources), mode=progress_mode, label="pvalues-ppi-effect")
    results: list[PPIEffectResult] = []

    if n_workers > 1:
        args_list = [(i, sc, effect_tests, n_reps, n_boot, seed) for i, (sc, seed) in enumerate(zip(sources, child_seeds))]
        ctx = _mp.get_context("fork")
        gold_nulls = [estimate_judge_bias_gold_null_values(sc, n_mc=gold_null_mc, seed=int(child_seeds[i][0]))
                      for i, sc in enumerate(sources)]
        with ctx.Pool(n_workers) as pool:
            for i, (sc_idx, samples_by_test) in enumerate(pool.imap_unordered(_run_ppi_effect_cell_worker, args_list)):
                sc = sources[sc_idx]
                gold_null = gold_nulls[sc_idx]
                for t in effect_tests:
                    samples = samples_by_test.get(t, [])
                    null_val = gold_null.get(t, 0.0)
                    bias_mean, z, coverage, mean_width, n = _effect_cell_stats(samples, null_val)
                    if n == 0:
                        continue  # test not valid for this scenario's eval_type (e.g. binary) -- see _ppi_effective_tests
                    unc_z = _uncorrected_bias_z(samples, null_val)
                    results.append(PPIEffectResult(
                        name=sc.name, tag=sc.tag, test=t, n=sc.n, n_samples=n,
                        null_value=null_val, mean_bias=bias_mean, bias_z=z,
                        uncorrected_bias_z=unc_z, coverage=coverage, mean_ci_width=mean_width,
                    ))
                reporter.update(i + 1)
        reporter.update(len(sources), detail="done")
        return results

    for i, sc in enumerate(sources):
        gold_null = estimate_judge_bias_gold_null_values(sc, n_mc=gold_null_mc, seed=int(child_seeds[i][0]))
        samples_by_test = _run_ppi_effect_cell(sc, effect_tests, n_reps, n_boot, child_seeds[i])
        for t in effect_tests:
            samples = samples_by_test.get(t, [])
            null_val = gold_null.get(t, 0.0)
            bias_mean, z, coverage, mean_width, n = _effect_cell_stats(samples, null_val)
            if n == 0:
                continue  # test not valid for this scenario's eval_type (e.g. binary) -- see _ppi_effective_tests
            unc_z = _uncorrected_bias_z(samples, null_val)
            results.append(PPIEffectResult(
                name=sc.name, tag=sc.tag, test=t, n=sc.n, n_samples=n,
                null_value=null_val, mean_bias=bias_mean, bias_z=z,
                coverage=coverage, mean_ci_width=mean_width, uncorrected_bias_z=unc_z,
            ))
        reporter.update(i + 1, detail=f"{sc.name}")
    reporter.update(len(sources), detail="done")
    return results


def _ppi_wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli rate (ported from
    sim_type_i_calibration.py's ``_wilson_interval``)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    radius = (z / denom) * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
    lo = max(0.0, center - radius)
    hi = min(1.0, center + radius)
    return lo, hi


def _ppi_holm_rejections(pvals: list[tuple[tuple[str, str], float]], alpha: float = 0.05) -> set[tuple[str, str]]:
    """Rejected (scenario, test) cells under Holm-Bonferroni family-wise error
    control (ported from sim_type_i_calibration.py's ``_holm_rejections``)."""
    ordered = sorted(pvals, key=lambda x: x[1])
    m = len(ordered)
    rejected: set[tuple[str, str]] = set()
    for i, (cell, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        if p <= thresh:
            rejected.add(cell)
        else:
            break
    return rejected


def _fmt_ppi_rate(rate: float | None, flag2: float, flag3: float) -> str:
    if rate is None:
        return "  n/a  "
    s = f"{rate:.3f}"
    if rate > flag3:
        return s + "●●"
    if rate > flag2:
        return s + "● "
    return s + "  "


def print_ppi_report(results: list[PPIResult], alpha: float) -> None:
    """Scenario x test calibration table, mirroring sim_type_i_calibration.py's
    ``_print_table``: tag-grouped scenario rows, one column per test, per-cell
    2-sigma/3-sigma inflation flags, a Wilson-CI miscalibration flag (dagger),
    a Holm-Bonferroni family-wise flag (double-dagger), and a SUMMARY section
    with flag counts plus per-test corrected/uncorrected max/mean/median --
    instead of one flat row per (scenario, test) cell and a single
    averaged-rate table.
    """
    if not results:
        print("\n  (no PPI results)")
        return

    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    n_reps = results[0].n_reps
    sigma = (alpha * (1 - alpha) / n_reps) ** 0.5 if n_reps > 0 else float("nan")
    flag2 = alpha + 2 * sigma
    flag3 = alpha + 3 * sigma

    width = 90
    bar = "-" * width
    dbar = "=" * width
    col_w = max(9, max((len(t) for t in tests), default=9) + 1)

    cell: dict[tuple[str, str], PPIResult] = {(r.name, r.test): r for r in results}
    scenario_order: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in results:
        if r.name not in seen:
            seen.add(r.name)
            scenario_order.append((r.name, r.tag))
    name_w = max((len(name) for name, _tag in scenario_order), default=30) + 2

    tag_order: list[str] = []
    for _, tag in scenario_order:
        if tag not in tag_order:
            tag_order.append(tag)

    def rate_of(name: str, test: str) -> float | None:
        r = cell.get((name, test))
        if r is None or r.n_reps <= 0:
            return None
        return r.corrected_rejects / r.n_reps

    def uncorrected_rate_of(name: str, test: str) -> float | None:
        r = cell.get((name, test))
        if r is None or r.n_reps <= 0:
            return None
        return r.uncorrected_rejects / r.n_reps

    def wilson_outside(name: str, test: str) -> bool:
        r = cell.get((name, test))
        if r is None or r.n_reps <= 0:
            return False
        lo, hi = _ppi_wilson_interval(r.corrected_rejects, r.n_reps)
        return (alpha < lo) or (alpha > hi)

    print()
    print(dbar)
    print("  PVALUES (PPI-CORRECTED) -- TYPE I ERROR CALIBRATION")
    print(f"  n_reps={n_reps}  alpha={alpha}")
    print(f"  2σ flag (●): rate > {flag2:.3f}    3σ flag (●●): rate > {flag3:.3f}")
    print("  Wilson flag (†): 95% CI for rejection rate excludes alpha")
    print("  Holm flag (‡): exact binomial miscalibration survives family-wise correction")
    print(dbar)

    print()
    print(f"  {'Scenario':<{name_w}}" + "".join(f"{t:^{col_w}}" for t in tests))
    print(bar)

    pvals: list[tuple[tuple[str, str], float]] = []
    for name, _tag in scenario_order:
        for t in tests:
            r = cell.get((name, t))
            if r is not None and r.n_reps > 0:
                p = float(scipy_stats.binomtest(r.corrected_rejects, r.n_reps, alpha, alternative="two-sided").pvalue)
                pvals.append(((name, t), p))
    holm_bad = _ppi_holm_rejections(pvals, alpha=0.05)

    for tag in tag_order:
        print(f"\n[{tag}]")
        for name, sc_tag in scenario_order:
            if sc_tag != tag:
                continue
            row = f"  {name:<{name_w - 2}}"
            for t in tests:
                row += f" {_fmt_ppi_rate(rate_of(name, t), flag2, flag3):<{col_w - 1}}"
            n_failed_row = sum(cell[(name, t)].n_failed for t in tests if (name, t) in cell)
            if n_failed_row:
                row += f" ✗{n_failed_row}"
            if any(wilson_outside(name, t) for t in tests):
                row += "  †"
            if any((name, t) in holm_bad for t in tests):
                row += "‡"
            print(row)

    # -- Summary --------------------------------------------------------------
    print()
    print(bar)
    print("SUMMARY")
    print()

    n_scenarios = len(scenario_order)
    n_conditions = sum(1 for name, _tag in scenario_order for t in tests if (name, t) in cell)
    total_failed = sum(r.n_failed for r in results)

    all_corr = [rate_of(name, t) for name, _tag in scenario_order for t in tests]
    all_unc = [uncorrected_rate_of(name, t) for name, _tag in scenario_order for t in tests]

    flags2 = sum(1 for r in all_corr if r is not None and r > flag2)
    flags3 = sum(1 for r in all_corr if r is not None and r > flag3)
    wilson_miscal = sum(1 for name, _tag in scenario_order for t in tests if wilson_outside(name, t))
    nominal_miscal = sum(1 for _, p in pvals if p < 0.05)
    uncorrected_flags2 = sum(1 for r in all_unc if r is not None and r > flag2)
    uncorrected_flags3 = sum(1 for r in all_unc if r is not None and r > flag3)

    print(f"  Scenarios: {n_scenarios}  |  Tests: {len(tests)}  |  Conditions: {n_conditions}  |  Failed reps: {total_failed}")
    print(f"  Inflated at 2σ (rate > {flag2:.3f}):  {flags2}/{n_conditions}")
    print(f"  Inflated at 3σ (rate > {flag3:.3f}):  {flags3}/{n_conditions}")
    print(f"  Wilson miscalibrated (alpha outside 95% CI): {wilson_miscal}/{n_conditions}")
    print(f"  Exact-binomial p<0.05 (corrected rates, unadjusted): {nominal_miscal}/{n_conditions}")
    print(f"  Holm-confirmed miscalibrated cells: {len(holm_bad)}/{n_conditions}")
    print()
    print("  Uncorrected aggregate")
    print(f"  Inflated at 2σ (rate > {flag2:.3f}):  {uncorrected_flags2}/{n_conditions}")
    print(f"  Inflated at 3σ (rate > {flag3:.3f}):  {uncorrected_flags3}/{n_conditions}")
    print()
    print(f"  {'Test':<14}  {'corr max':>9}  {'corr mean':>9}  {'corr med':>9}  {'unc max':>9}  {'unc mean':>9}  {'unc med':>9}")
    for t in tests:
        col_rates = [r for r in (rate_of(name, t) for name, _tag in scenario_order) if r is not None]
        col_uncorrected = [r for r in (uncorrected_rate_of(name, t) for name, _tag in scenario_order) if r is not None]
        if col_rates or col_uncorrected:
            corr_max = max(col_rates) if col_rates else float("nan")
            corr_mean = float(np.mean(col_rates)) if col_rates else float("nan")
            corr_median = float(np.median(col_rates)) if col_rates else float("nan")
            unc_max = max(col_uncorrected) if col_uncorrected else float("nan")
            unc_mean = float(np.mean(col_uncorrected)) if col_uncorrected else float("nan")
            unc_median = float(np.median(col_uncorrected)) if col_uncorrected else float("nan")
            print(
                f"  {t:<14}  {corr_max:>9.3f}  {corr_mean:>9.3f}  {corr_median:>9.3f}  "
                f"{unc_max:>9.3f}  {unc_mean:>9.3f}  {unc_median:>9.3f}"
            )
    print()


def latex_ppi_overall_summary(results: list[PPIResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-test corrected/uncorrected Type-I
    rate (each with its 95% MC band), averaged across scenarios, plus one
    corrected-rate column per sample size actually swept by the
    'sample_size' tag (n=60/100/200/400 -- the only scenarios that
    deliberately vary n; every other scenario shares the fixed baseline),
    appended to the right.

    PPIResult has no eval_type axis (scenarios are judge-bias/noise sweeps,
    not distribution shapes), so there's no "Eval types" column here.
    """
    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    sizes_present = sorted({r.n for r in results if r.tag == "sample_size"})
    rows = []
    for t in tests:
        t_rows = [r for r in results if r.test == t]
        c_tot = sum(r.corrected_rejects for r in t_rows)
        u_tot = sum(r.uncorrected_rejects for r in t_rows)
        n_tot = sum(r.n_reps for r in t_rows)
        rate_c = c_tot / n_tot if n_tot > 0 else float("nan")
        rate_u = u_tot / n_tot if n_tot > 0 else float("nan")
        _, _, lo_c, hi_c = _mc_proportion_stats(c_tot, n_tot)
        _, _, lo_u, hi_u = _mc_proportion_stats(u_tot, n_tot)
        row = [
            escape_latex(t),
            f"{rate_c:.3f}" if np.isfinite(rate_c) else "-",
            f"${lo_c:.3f}\\text{{--}}{hi_c:.3f}$" if np.isfinite(lo_c) else "-",
            f"{rate_u:.3f}" if np.isfinite(rate_u) else "-",
            f"${lo_u:.3f}\\text{{--}}{hi_u:.3f}$" if np.isfinite(lo_u) else "-",
        ]
        for n in sizes_present:
            n_rows = [r for r in t_rows if r.tag == "sample_size" and r.n == n]
            c_n = sum(r.corrected_rejects for r in n_rows)
            t_n = sum(r.n_reps for r in n_rows)
            rate_n = c_n / t_n if t_n > 0 else float("nan")
            row.append(f"{rate_n:.3f}" if np.isfinite(rate_n) else "-")
        rows.append(row)

    return booktabs_table(
        caption=f"pvalues (PPI-corrected): corrected vs. uncorrected Type-I rate (nominal alpha={alpha}).",
        label="tab:pvalues_ppi_overall",
        columns=["Test", "Corrected", "95\\% MC band", "Uncorrected", "95\\% MC band"]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
    )


def save_results_artifacts_ppi(*, results: list[PPIResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "tag", "n", "test", "n_reps", "corrected_rejects", "uncorrected_rejects", "n_failed", "corrected_rate", "uncorrected_rate"])
        for r in results:
            writer.writerow([
                r.name, r.tag, r.n, r.test, r.n_reps, r.corrected_rejects, r.uncorrected_rejects, r.n_failed,
                f"{r.corrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.uncorrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
            ])
    summary_path = out_base / f"{run_stem}_ppi_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_ppi_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_typeI_plot(*, results: list[PPIResult], alpha: float, out_path: str, nonstandard: bool = False) -> str:
    """Per-scenario corrected vs. uncorrected Type-I rate scatter, one jittered
    column per test. Mirrors sim_type_i_calibration.py's ``_plot_results``
    scatter (gray uncorrected dots behind colored corrected dots, one dot per
    scenario, dashed alpha line) rather than collapsing every scenario into a
    single averaged bar, which hid per-scenario miscalibration entirely.

    nonstandard : bool
        When False (default), plots only the standard/textbook tests
        (excludes bayes_bootstrap/bootstrap_t/tango_score). When True,
        plots ONLY those three bootstrap/CI-based methods instead -- see
        _PPI_NONSTANDARD_TESTS for why they're kept out of the main plot.
    """
    import matplotlib.pyplot as plt

    tests = _ppi_tests_present(results, nonstandard=nonstandard)
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    rng = np.random.default_rng(0)
    unc_label_added = False
    all_rates: list[np.ndarray] = []

    for j, t in enumerate(tests):
        t_rows = [r for r in results if r.test == t]
        rates_u = np.array([r.uncorrected_rejects / r.n_reps if r.n_reps else float("nan") for r in t_rows])
        rates_c = np.array([r.corrected_rejects / r.n_reps if r.n_reps else float("nan") for r in t_rows])
        all_rates.append(rates_u)
        all_rates.append(rates_c)

        keep_u = np.isfinite(rates_u)
        x_u = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_u)))
        ax.scatter(
            x_u, rates_u[keep_u], s=18, alpha=0.35, color="#808080",
            label="uncorrected" if not unc_label_added else None, zorder=1,
        )
        unc_label_added = True

        keep_c = np.isfinite(rates_c)
        x_c = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_c)))
        ax.scatter(x_c, rates_c[keep_c], s=20, alpha=0.65, color=get_method_color(t), label=t, zorder=2)

    ax.axhline(alpha, color="black", ls="--", lw=1.1, label=f"alpha={alpha}")
    ax.set_xlim(-0.5, len(tests) - 0.5)
    scatter_max = np.nanmax(np.concatenate(all_rates)) if all_rates else float("nan")
    if not np.isfinite(scatter_max):
        scatter_max = 0.2
    ax.set_ylim(0.0, max(0.2, float(scatter_max) * 1.05))
    ax.set_xticks(np.arange(len(tests)))
    ax.set_xticklabels(tests, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Observed rejection rate")
    ax.set_xlabel("Test")
    title_suffix = " -- bootstrap/CI-based methods" if nonstandard else ""
    ax.set_title(f"pvalues (PPI-corrected): Type-I calibration scatter (per-scenario cells){title_suffix}")
    ax.grid(axis="y", alpha=0.25, lw=0.8)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_ppi_effect_report(results: list[PPIEffectResult], alpha: float) -> None:
    """Bias & CI-coverage summary, mirroring sim_type_i_calibration.py's
    ``_print_effect_table``: per-test mean bias, worst |z|, coverage, worst
    coverage scenario, and mean CI width, plus a flagged-cells list (|bias
    z| > 3, or coverage meaningfully under the 1-alpha target)."""
    if not results:
        print(
            "\n  (no PPI effect-check results -- active --tests must include at least one of "
            f"{', '.join(_PPI_EFFECT_TESTS)})"
        )
        return

    target_cov = 1.0 - alpha
    cov_flag_margin = 0.02
    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]

    width = 96
    bar = "-" * width
    dbar = "=" * width
    print()
    print(dbar)
    print("  PVALUES (PPI-CORRECTED) -- EFFECT-SIZE CALIBRATION (bias & CI coverage)")
    print("  (vs. Monte Carlo gold-reference null per scenario/test -- see estimate_judge_bias_gold_null_values)")
    print(dbar)
    print()
    header = (
        f"  {'Test':<12} {'n':>7} {'mean bias':>10} {'worst |z|':>10} "
        f"{'worst scen (bias)':<26} {'coverage':>9} {'cov min':>8} {'worst scen (cov)':<24} "
        f"{'mean width':>11}"
    )
    print(header)
    print(bar)

    flagged: list[str] = []
    for t in tests:
        t_rows = [r for r in results if r.test == t and r.n_samples > 0]
        if not t_rows:
            continue
        n_total = sum(r.n_samples for r in t_rows)
        weights = [r.n_samples for r in t_rows]
        mean_bias = float(np.average([r.mean_bias for r in t_rows], weights=weights))
        mean_width = float(np.average([r.mean_ci_width for r in t_rows], weights=weights))
        coverage = float(np.average([r.coverage for r in t_rows], weights=weights))
        worst_bias = max(t_rows, key=lambda r: abs(r.bias_z) if np.isfinite(r.bias_z) else 0.0)
        worst_cov = min(t_rows, key=lambda r: r.coverage if np.isfinite(r.coverage) else 1.0)

        print(
            f"  {t:<12} {n_total:>7} {mean_bias:>+10.4f} {abs(worst_bias.bias_z) if np.isfinite(worst_bias.bias_z) else 0.0:>10.2f} "
            f"{worst_bias.name:<26} {coverage:>9.3f} {worst_cov.coverage:>8.3f} {worst_cov.name:<24} "
            f"{mean_width:>11.4f}"
        )

        for r in t_rows:
            if np.isfinite(r.bias_z) and abs(r.bias_z) > 3.0:
                flagged.append(
                    f"    bias    {r.name:<28} {t:<10} mean={r.mean_bias:+.4f}  z={r.bias_z:+.2f}  (n={r.n_samples})"
                )
            lo_cov, hi_cov = _ppi_wilson_interval(int(round(r.coverage * r.n_samples)), r.n_samples)
            if hi_cov < target_cov - cov_flag_margin:
                flagged.append(
                    f"    cover   {r.name:<28} {t:<10} coverage={r.coverage:.3f}  "
                    f"Wilson=[{lo_cov:.3f},{hi_cov:.3f}]  (n={r.n_samples})"
                )

    print()
    if flagged:
        print(f"  Flagged cells (|bias z| > 3, or coverage Wilson upper bound < {target_cov - cov_flag_margin:.2f}):")
        for line in flagged:
            print(line)
    else:
        print("  No scenario x test cells flagged for bias or under-coverage.")
    print()


def latex_ppi_effect_overall_summary(results: list[PPIEffectResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-test bias and CI coverage (with
    its 95% MC band) and mean CI width of the PPI-corrected point estimate,
    averaged across scenarios -- complements latex_ppi_overall_summary's
    Type-I table with "is the estimate itself trustworthy," not just "does
    the p-value stay calibrated."""
    target_cov = 1.0 - alpha
    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    rows = []
    for t in tests:
        t_rows = [r for r in results if r.test == t and r.n_samples > 0]
        if not t_rows:
            continue
        n_tot = sum(r.n_samples for r in t_rows)
        weights = [r.n_samples for r in t_rows]
        mean_bias = float(np.average([r.mean_bias for r in t_rows], weights=weights))
        mean_width = float(np.average([r.mean_ci_width for r in t_rows], weights=weights))
        cov_count = sum(int(round(r.coverage * r.n_samples)) for r in t_rows)
        coverage = cov_count / n_tot if n_tot > 0 else float("nan")
        _, _, lo, hi = _mc_proportion_stats(cov_count, n_tot)
        rows.append([
            escape_latex(t),
            f"{mean_bias:+.4f}",
            f"{coverage:.3f}" if np.isfinite(coverage) else "-",
            f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
            f"{mean_width:.4f}" if np.isfinite(mean_width) else "-",
        ])

    return booktabs_table(
        caption=f"pvalues (PPI-corrected): bias and CI coverage of the corrected point estimate (nominal {target_cov:.0%}).",
        label="tab:pvalues_ppi_effect_overall",
        columns=["Test", "Mean bias", "Coverage", "95\\% MC band", "Mean CI width"],
        rows=rows,
    )


def save_results_artifacts_ppi_effect(*, results: list[PPIEffectResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_effect_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "tag", "n", "test", "n_samples", "null_value",
            "mean_bias", "bias_z", "coverage", "mean_ci_width", "uncorrected_bias_z",
        ])
        for r in results:
            writer.writerow([
                r.name, r.tag, r.n, r.test, r.n_samples, r.null_value,
                r.mean_bias, r.bias_z, r.coverage, r.mean_ci_width, r.uncorrected_bias_z,
            ])
    summary_path = out_base / f"{run_stem}_ppi_effect_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_effect_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_ppi_effect_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_effect_plot(
    *, results: list[PPIEffectResult], alpha: float, out_path: str, nonstandard: bool = False,
) -> str:
    """Bias-z / CI-coverage / CI-width scatter, one jittered column per test
    -- mirrors sim_type_i_calibration.py's ``_plot_effect_results`` (3
    panels), reading directly off PPIEffectResult's already-aggregated
    per-scenario stats rather than raw bootstrap samples.

    nonstandard : bool
        When False (default), plots only the standard/textbook tests
        (excludes bayes_bootstrap/bootstrap_t/tango_score). When True,
        plots ONLY those three bootstrap/CI-based methods instead -- see
        _PPI_NONSTANDARD_TESTS for why they're kept out of the main plot.
    """
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("save_ppi_effect_plot: no PPI effect-check results to plot.")

    tests = _ppi_tests_present(results, nonstandard=nonstandard)
    target_cov = 1.0 - alpha
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.0, 5.5))
    rng = np.random.default_rng(0)

    for j, t in enumerate(tests):
        t_rows = [r for r in results if r.test == t and r.n_samples > 0]
        if not t_rows:
            continue
        x = j + rng.uniform(-0.16, 0.16, size=len(t_rows))
        color = get_method_color(t)

        z = np.array([r.bias_z for r in t_rows])
        keep_z = np.isfinite(z)
        ax1.scatter(x[keep_z], z[keep_z], s=22, alpha=0.7, color=color, label=t)

        cov = np.array([r.coverage for r in t_rows])
        keep_c = np.isfinite(cov)
        ax2.scatter(x[keep_c], cov[keep_c], s=22, alpha=0.7, color=color)

        wid = np.array([r.mean_ci_width for r in t_rows])
        keep_w = np.isfinite(wid)
        ax3.scatter(x[keep_w], wid[keep_w], s=22, alpha=0.7, color=color)

    ax1.axhline(0.0, color="black", ls="--", lw=1.0)
    ax1.axhline(3.0, color="red", ls=":", lw=0.9)
    ax1.axhline(-3.0, color="red", ls=":", lw=0.9)
    ax1.set_xticks(np.arange(len(tests)))
    ax1.set_xticklabels(tests, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Bias z-score")
    ax1.set_title("Estimate bias (z vs. gold null)")
    ax1.grid(axis="y", alpha=0.25, lw=0.8)

    ax2.axhline(target_cov, color="black", ls="--", lw=1.1, label=f"target={target_cov:.2f}")
    ax2.set_xticks(np.arange(len(tests)))
    ax2.set_xticklabels(tests, rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0.0, 1.02)
    ax2.set_ylabel("CI coverage of gold null")
    ax2.set_title("CI coverage")
    ax2.grid(axis="y", alpha=0.25, lw=0.8)
    ax2.legend(loc="lower left", fontsize=8)

    ax3.set_xticks(np.arange(len(tests)))
    ax3.set_xticklabels(tests, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Mean CI width")
    ax3.set_title("CI width")
    ax3.grid(axis="y", alpha=0.25, lw=0.8)

    handles, labels = ax1.get_legend_handles_labels()
    title_suffix = " -- bootstrap/CI-based methods" if nonstandard else ""
    fig.suptitle(f"pvalues (PPI-corrected): effect-size calibration (bias, coverage, width){title_suffix}", y=1.12, fontsize=12)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(tests), 8), fontsize=8, bbox_to_anchor=(0.5, 1.0))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=MODES, default="all",
                         help="'pairwise' (non-PPI A/B), 'multiarm' (non-PPI k-arm), "
                              "'ppi' (PPI-corrected calibration), 'simultaneous_ci' (none vs. Bonferroni vs. "
                              "max-T simultaneous-CI coverage/width, on the same k-arm sources as multiarm), "
                              "'pairwise_multiarm' (just pairwise+multiarm), or 'all' (default: every mode "
                              "applicable to --data-source -- pairwise+multiarm+ppi+simultaneous_ci for "
                              "synthetic, pairwise+multiarm+simultaneous_ci for real data)")
    parser.add_argument("--reps", type=int, default=200, metavar="N")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--latex", action="store_true", default=False,
                         help="Append a LaTeX booktabs overall-summary table to each saved summary .log file.")

    # pairwise mode
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="synthetic",
                         help="pairwise/multiarm modes: 'synthetic' (default), or a real-data source: " + ", ".join(REAL_PAIR_SOURCES))
    parser.add_argument("--scenario-suite", choices=SCENARIO_SUITES, default="expanded",
                         help="pairwise mode: synthetic scenario breadth for build_pair_sources (ignored for real data sources)")
    parser.add_argument("--eval-types", nargs="+", choices=EVAL_TYPES, default=None, metavar="TYPE",
                         help="pairwise/multiarm modes: restrict to these eval types")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50, 100], metavar="N",
                         help="pairwise/multiarm modes: sample sizes to sweep")
    parser.add_argument("--runs", type=int, default=1, metavar="R",
                         help="pairwise/multiarm modes: runs per input (R>1 activates binary majority-vote/nested paths; "
                              "real-data pairwise sources only support --runs 1)")
    parser.add_argument("--statistic", choices=["mean", "median"], default="mean",
                         help="pairwise/multiarm modes: statistic passed to evalstats.core.paired")
    parser.add_argument("--bootstrap-n", type=int, default=500, metavar="N",
                         help="pairwise/multiarm modes: bootstrap resample count")
    parser.add_argument("--icc-values", type=float, nargs="+", default=[0.05, 0.20, 0.40, 0.60, 0.80], metavar="ICC",
                         help="pairwise mode: ICC sweep for build_pair_sources (ignored for real data sources)")
    parser.add_argument("--cohens-d-values", type=float, nargs="+", default=[0.2, 0.4], metavar="D",
                         help="pairwise mode: alt-condition effect sizes for build_pair_sources (ignored for real data sources)")
    parser.add_argument("--benchmarks", nargs="+", default=None, metavar="ID",
                         help="pairwise/multiarm modes, real data: benchmark IDs to filter to")
    parser.add_argument("--models", nargs="+", default=None, metavar="NAME",
                         help="pairwise/multiarm modes, real data: model names to filter to")
    parser.add_argument("--hf-token", default=None, help="pairwise/multiarm modes, real data")
    parser.add_argument("--cache-dir", default=None, help="pairwise/multiarm modes, real data")
    parser.add_argument("--min-pair-size", type=int, default=50,
                         help="pairwise/multiarm modes, real data: minimum shared items required "
                              "(multiarm: across ALL aligned models for a benchmark, not just a pair)")
    parser.add_argument("--inspect-csv", default=None,
                         help=f"pairwise/multiarm modes, real data: path to CSV from collect_inspect_benchmarks.py "
                              f"(used by --data-source inspect/real; defaults to {DEFAULT_INSPECT_CSV!r})")

    # multiarm mode (also used by simultaneous_ci mode -- same k-arm sources/grid)
    parser.add_argument("--k-arms", nargs="+", type=int, default=[4], metavar="K",
                        help="multiarm/simultaneous_ci modes: number of arms to sweep (one or more values, e.g. --k-arms 3 5 10); "
                             "max-T and post-hoc corrections are compared at each k. Real-data sources cap k at "
                             "however many aligned real models a benchmark has; larger k values are skipped with a warning.")
    parser.add_argument("--multiarm-method", default=BOOTSTRAP_T.name, metavar="METHOD",
                         choices=[BOOTSTRAP.name, BCA.name, BAYES_BOOTSTRAP.name, SMOOTH_BOOTSTRAP.name, PERMUTATION.name, BOOTSTRAP_T.name],
                         help="multiarm/simultaneous_ci modes: pairwise_differences-family method used to derive "
                              "raw p-values (multiarm) / the point-estimate + bootstrap draws that both simultaneous-CI "
                              "constructions share (simultaneous_ci) -- must be bootstrap-compatible for max-T to apply")
    parser.add_argument("--multiarm-icc", type=float, default=0.20, metavar="ICC",
                         help="multiarm/simultaneous_ci modes: ICC for build_multiarm_sources' shared truth/noise model "
                              "(same meaning as --icc-values in pairwise mode)")
    parser.add_argument("--multiarm-cohens-d", type=float, default=0.3, metavar="D",
                         help="multiarm/simultaneous_ci modes: alt-condition effect size (Cohen's d) for build_multiarm_sources")

    # ppi mode
    parser.add_argument("--tests", nargs="+", choices=[m.name for m in PPI_TEST_METHODS], default=None, metavar="TEST",
                         help="ppi mode: restrict to these evalstats.tests names (default: all)")
    parser.add_argument("--ppi-n-boot", type=int, default=1000, metavar="N",
                         help="ppi mode: PPI bootstrap resample count")
    parser.add_argument("--effect-reps", type=int, default=200, metavar="N",
                         help="ppi mode: reps for the bias/CI-coverage effect-size check of the corrected "
                              "point estimate (separate, typically smaller, pass from --reps' Type-I sweep)")
    parser.add_argument("--effect-gold-mc", type=int, default=3000, metavar="N",
                         help="ppi mode: Monte Carlo reps used to estimate each scenario/test's gold-reference "
                              "null value (estimate_judge_bias_gold_null_values)")
    parser.add_argument("--no-effect-check", action="store_true", default=False,
                         help="ppi mode: skip the bias/CI-coverage effect-size check, running Type-I calibration only")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N",
                         help="Parallel worker processes (default: cpu_count-1; 1=sequential).")


def official_args(base_seed: int = 42) -> argparse.Namespace:
    """Canonical official-test preset: pairwise + multiarm, synthetic data.
    PPI calibration is a separate, much slower preset (official_args_ppi)
    split out so it can be run/skipped independently in the
    --official-tests menu -- see official_variants().

    ``k_arms`` sweeps up to k=20 (190 pairwise comparisons -- pairs grow as
    k(k-1)/2, so this is roughly 4x the comparisons of k=10's 45) rather
    than stopping at 10. This is deliberate, not just "more thorough": a
    real multi-model comparison (an LLM leaderboard slice, an ablation with
    many variants) routinely has 10-20+ arms, and Bonferroni's per-comparison
    alpha budget (alpha/pairs) shrinks toward that same rate, which is
    exactly the regime where its extra conservativeness over max-T
    (--mode simultaneous_ci) -- and, on the p-value side, over max_t/holm/
    fdr_bh's better power at matched FWER (--mode multiarm) -- becomes most
    visible. Costs ~3.8x the k-sweep's compute vs. stopping at k=10, since
    per-cell cost tracks the pair count; both --mode multiarm and
    --mode simultaneous_ci reuse this same official_args() preset (see
    official_args_simultaneous_ci) and its k_arms sweep.

    Excludes "grades" from eval_types (also inherited by official_args_ppi/
    official_args_simultaneous_ci, which derive from this preset):
    "continuous" already covers the [0, 1]-scale case well (grades is just
    continuous rescaled to 0-100), while "likert" is kept as a genuinely
    distinct limiting case (integer-valued, few levels). Dropping grades
    cuts a third eval type out of the official sweep's runtime for no real
    loss of coverage."""
    return argparse.Namespace(
        mode="pairwise_multiarm", reps=300, alpha=0.05, seed=base_seed,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        data_source="synthetic", scenario_suite="expanded", eval_types=["binary", "continuous", "likert"], sizes=[10, 20, 30, 50, 75, 100],
        runs=1, statistic="mean",
        bootstrap_n=2000, icc_values=[0.05, 0.20, 0.40, 0.60, 0.80], cohens_d_values=[0.2, 0.4],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        k_arms=[3, 5, 10, 20], multiarm_method=BOOTSTRAP_T.name, multiarm_icc=0.20, multiarm_cohens_d=0.3,
        tests=None, ppi_n_boot=2000, effect_reps=200, effect_gold_mc=3000, no_effect_check=False,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def official_args_multiarm(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for multiarm-only calibration (synthetic data).
    Split out from official_args() (which runs mode="pairwise_multiarm",
    i.e. both pairwise and multiarm together) so the multiarm sweep alone
    can be re-run on its own -- e.g. after a multiarm-specific performance
    or correctness fix -- without paying for the separate (and unrelated)
    pairwise sweep every time."""
    args = official_args(base_seed)
    args.mode = "multiarm"
    return args


def official_args_ppi(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for PPI-corrected calibration only (synthetic
    data -- ppi has no real-data variant). Split out from official_args()
    since it's by far the slowest of the pvalues sub-modes (43 judge-bias
    scenarios x ~11 tests x reps, plus a separate bias/coverage effect-size
    check), so it can be selected or skipped on its own in the
    --official-tests menu instead of always riding along with the faster
    pairwise/multiarm sweep."""
    args = official_args(base_seed)
    args.mode = "ppi"
    return args


def official_args_simultaneous_ci(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for simultaneous-CI calibration only (synthetic
    data). Split out from official_args() for the same reason as
    official_args_ppi -- lets --official-tests select/skip it independently
    of the faster pairwise/multiarm sweep, even though it shares those
    modes' k-arm sources.

    Overrides two of official_args()'s defaults:
    - scenario_suite="standard" (not "expanded"): the effect this mode
      exists to show -- max-T's bootstrap_t studentization developing a
      random-denominator instability at small N combined with large k (see
      print_simultaneous_ci_report's LOW N / HIGH N split, and the
      coverage/width/violin plots) -- is consistent across scenario shapes
      within an eval type, so the smaller "standard" catalog (23 shapes vs.
      "expanded"'s 39) still demonstrates it clearly at a fraction of the
      compute; this mode's per-cell cost (bootstrap_t's nested double
      bootstrap, k(k-1)/2 marginal pairs plus the shared max-T resample) is
      high enough that this matters much more here than in the
      pairwise/multiarm modes official_args() also serves.
    - sizes adds n=125 on top of the default sweep, extending the high-N
      anchor a bit further past the ~30 crossover found in practice.
    """
    args = official_args(base_seed)
    args.mode = "simultaneous_ci"
    args.scenario_suite = "expanded"
    args.sizes = [10, 20, 30, 50, 75, 100, 125]
    return args


def real_official_args(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for real data sources (pairwise + multiarm only;
    ppi has no real-data variant, and simultaneous_ci is its own preset --
    see real_official_args_simultaneous_ci). Requires network/HF access."""
    return argparse.Namespace(
        mode="pairwise_multiarm", reps=300, alpha=0.05, seed=base_seed,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        data_source="real", scenario_suite="expanded", eval_types=None, sizes=[10, 20, 30, 50, 75, 100],
        runs=1, statistic="mean",
        bootstrap_n=2000, icc_values=[0.05, 0.20, 0.40, 0.60, 0.80], cohens_d_values=[0.2, 0.4],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        k_arms=[3, 5, 10], multiarm_method=BOOTSTRAP_T.name, multiarm_icc=0.20, multiarm_cohens_d=0.3,
        tests=None, ppi_n_boot=2000, effect_reps=200, effect_gold_mc=3000, no_effect_check=False,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def real_official_args_simultaneous_ci(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for simultaneous-CI calibration only, real data
    (real multi-arm sources -- see build_real_multiarm_sources). Split out
    from real_official_args() the same way official_args_simultaneous_ci is."""
    args = real_official_args(base_seed)
    args.mode = "simultaneous_ci"
    return args


def official_variants(base_seed: int = 42) -> list[tuple[str, argparse.Namespace]]:
    """All official-test variants for this case, as (label, args) pairs."""
    return [
        ("synthetic (pairwise + multiarm)", official_args(base_seed)),
        ("synthetic (multiarm)", official_args_multiarm(base_seed)),
        ("synthetic (ppi)", official_args_ppi(base_seed)),
        ("synthetic (simultaneous CI)", official_args_simultaneous_ci(base_seed)),
        ("real data (pairwise + multiarm)", real_official_args(base_seed)),
        ("real data (simultaneous CI)", real_official_args_simultaneous_ci(base_seed)),
    ]


def quick_args(base_seed: int = 43, data_source: str = "synthetic") -> argparse.Namespace:
    """Fast sanity-check preset for --quick-test: runs every mode applicable
    to `data_source` (mode="all" -- see run()'s "all" handling) with cut-down
    sweeps/reps/tests for a quick pass that confirms the pipeline (incl.
    --latex output) still works. Restricts eval_types/tests/k_arms rather
    than sweeping the full catalog -- this is for pipeline confidence, not a
    representative result.
    ``data_source="real"`` (or 'openeval'/'inspect') runs pairwise + multiarm
    + simultaneous_ci only -- ppi has no real-data variant
    (build_judge_bias_sources is synthetic-only, see README's "known
    exceptions"). It also switches eval_types to 'binary': real-data
    pairwise/multiarm sources are binary-only by construction
    (corpus_pair_to_ci_pair_source / multiarm_corpus_to_source hardcode
    eval_type="binary" for both openeval and inspect), so the synthetic
    variant's 'continuous' filter would leave zero sources. --quick-test
    calls this twice per case (synthetic, then real) so the real-data paths
    don't go unexercised between --official-tests runs."""
    eval_types = ["binary", "continuous"] if data_source == "synthetic" else ["binary"]
    return argparse.Namespace(
        mode="all", reps=3, alpha=0.05, seed=base_seed,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        data_source=data_source, scenario_suite="standard", eval_types=eval_types, sizes=[10, 30, 50],
        runs=1, statistic="mean",
        bootstrap_n=200, icc_values=[0.20], cohens_d_values=[0.3],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        k_arms=[3], multiarm_method=BOOTSTRAP_T.name, multiarm_icc=0.20, multiarm_cohens_d=0.3,
        tests=[TTEST.name, MW.name, PAIRED_T.name, BAYES_BOOTSTRAP.name, BOOTSTRAP_T.name, TANGO.name], ppi_n_boot=200, latex=True,
        effect_reps=5, effect_gold_mc=200, no_effect_check=False,
        workers=1,
    )


def run(args: argparse.Namespace) -> CaseResult:
    t0 = time.time()
    try:
        plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_paths: list[str] = []
        key_metrics: dict = {}
        # "all" means "every mode applicable to this data source" -- ppi has
        # no real-data variant (build_judge_bias_sources is synthetic-only),
        # so real/openeval/inspect data sources skip it rather than silently
        # re-running the synthetic PPI sweep under a real-data preset.
        # "pairwise_multiarm" is "all" minus ppi regardless of data source --
        # lets --official-tests offer synthetic ppi as its own (slow) menu
        # entry, separate from the faster pairwise+multiarm sweep.
        if args.mode == "pairwise_multiarm":
            modes = ["pairwise", "multiarm"]
        elif args.mode != "all":
            modes = [args.mode]
        elif args.data_source == "synthetic":
            modes = ["pairwise", "multiarm", "ppi", "simultaneous_ci"]
        else:
            modes = ["pairwise", "multiarm", "simultaneous_ci"]

        if "pairwise" in modes:
            print(f"\npvalues simulation (pairwise, non-PPI) -- data_source={args.data_source}, statistic={args.statistic}")
            if args.data_source == "synthetic":
                sources = build_pair_sources(
                    suite=args.scenario_suite, icc_values=args.icc_values,
                    cohens_d_values=args.cohens_d_values, include_null=True,
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
                    inspect_csv=args.inspect_csv, include_null=True,
                )
            if args.eval_types:
                requested = set(args.eval_types)
                sources = [s for s in sources if s.eval_type in requested]
            if not sources:
                raise ValueError("No CIPairSources left after filtering.")
            print(f"  {len(sources)} sources, sizes={args.sizes}, reps={args.reps}, alpha={args.alpha}")

            pw_results = run_pairwise_simulation(
                sources, sample_sizes=args.sizes, runs=args.runs, n_reps=args.reps, n_bootstrap=args.bootstrap_n,
                alpha=args.alpha, statistic=args.statistic, progress_mode=args.progress, seed=args.seed,
                n_workers=getattr(args, "workers", 1),
            )
            print_pairwise_report(pw_results, alpha=args.alpha)

            run_stem = f"pvalues_pairwise_{args.data_source}_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_pairwise(results=pw_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False))
            if args.plots == "save":
                plot_paths = save_pairwise_typeI_power_plot(results=pw_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_typeI_power.png"))
                for plot_path in plot_paths:
                    output_paths.append(plot_path)
                    print(f"Saved plot: {plot_path}")
                reliability_path = save_pairwise_reliability_violin_plot(
                    results=pw_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_reliability_violin.png"),
                )
                output_paths.append(reliability_path)
                print(f"Saved plot: {reliability_path}")

            null_rows = [r for r in pw_results if r.condition == "null"]
            type1 = float(np.mean([r.rejects / r.n_reps for r in null_rows])) if null_rows else float("nan")
            key_metrics["pairwise_n_results"] = len(pw_results)
            key_metrics["pairwise_mean_type1"] = type1

        if "multiarm" in modes or "simultaneous_ci" in modes:
            # Shared by both modes -- they sweep the identical k-arm source
            # list/grid, just measuring something different per rep (reject/
            # best-arm vs. CI coverage/width), so build it once even when
            # --mode all runs both (avoids a second, possibly network-bound,
            # real-data fetch).
            k_values = args.k_arms if isinstance(args.k_arms, list) else [args.k_arms]
            print(f"\npvalues simulation (multi-arm sources) -- data_source={args.data_source}, k={k_values}")
            if args.data_source == "synthetic":
                ma_sources = build_multiarm_sources(
                    suite=args.scenario_suite, icc=args.multiarm_icc, cohens_d=args.multiarm_cohens_d,
                    eval_types=args.eval_types,
                )
            else:
                runs = args.runs
                if runs != 1:
                    print("  Warning: real-data sources only support --runs 1 in this pass; forcing runs=1.")
                    runs = 1
                args = argparse.Namespace(**{**vars(args), "runs": runs})
                ma_sources = build_real_multiarm_sources(
                    args.data_source, benchmarks=args.benchmarks, models=args.models,
                    hf_token=args.hf_token, cache_dir=args.cache_dir, min_arm_size=args.min_pair_size,
                    inspect_csv=args.inspect_csv,
                )
                if args.eval_types:
                    requested = set(args.eval_types)
                    ma_sources = [s for s in ma_sources if s.eval_type in requested]
            if not ma_sources:
                raise ValueError("No MultiArmSources left after filtering.")
            print(f"  {len(ma_sources)} sources, sizes={args.sizes}, k_values={k_values}, reps={args.reps}, alpha={args.alpha}")

        if "multiarm" in modes:
            print(f"\npvalues simulation (multi-arm, non-PPI) -- method={args.multiarm_method}")
            ma_results = run_multiarm_simulation(
                ma_sources, sample_sizes=args.sizes, runs=args.runs, k_values=k_values, n_reps=args.reps,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, multiarm_method=args.multiarm_method,
                statistic=args.statistic, progress_mode=args.progress, seed=args.seed,
                n_workers=getattr(args, "workers", 1),
            )
            print_multiarm_report(ma_results, alpha=args.alpha)

            run_stem = f"pvalues_multiarm_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_multiarm(results=ma_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False))
            if args.plots == "save":
                plot_path = save_multiarm_fwer_power_plot(results=ma_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_fwer_power.png"))
                output_paths.append(plot_path)
                print(f"Saved plot: {plot_path}")
                vs_k_path = save_multiarm_fwer_vs_k_plot(results=ma_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_fwer_vs_k.png"))
                if Path(vs_k_path).exists():
                    output_paths.append(vs_k_path)
                    print(f"Saved plot: {vs_k_path}")
                reliability_path = save_multiarm_reliability_violin_plot(
                    results=ma_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_reliability_violin.png"),
                )
                output_paths.append(reliability_path)
                print(f"Saved plot: {reliability_path}")

            null_rows = [r for r in ma_results if r.condition == "null"]
            fwer = sum(r.any_reject for r in null_rows) / sum(r.n_reps for r in null_rows) if null_rows else float("nan")
            key_metrics["multiarm_n_results"] = len(ma_results)
            key_metrics["multiarm_mean_fwer"] = float(fwer)

        if "simultaneous_ci" in modes:
            print(f"\npvalues simulation (simultaneous CI, non-PPI) -- method={args.multiarm_method}")
            sci_results = run_simultaneous_ci_simulation(
                ma_sources, sample_sizes=args.sizes, runs=args.runs, k_values=k_values, n_reps=args.reps,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, multiarm_method=args.multiarm_method,
                statistic=args.statistic, progress_mode=args.progress, seed=args.seed,
                n_workers=getattr(args, "workers", 1),
            )
            print_simultaneous_ci_report(sci_results, alpha=args.alpha)

            run_stem = f"pvalues_simultaneous_ci_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_simultaneous_ci(
                    results=sci_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem,
                    latex=getattr(args, "latex", False),
                )
            if args.plots == "save":
                plot_path = save_simultaneous_ci_coverage_width_plot(
                    results=sci_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_coverage_width.png"),
                )
                output_paths.append(plot_path)
                print(f"Saved plot: {plot_path}")
                vs_k_path = save_simultaneous_ci_coverage_width_vs_k_plot(
                    results=sci_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_vs_k.png"),
                )
                if Path(vs_k_path).exists():
                    output_paths.append(vs_k_path)
                    print(f"Saved plot: {vs_k_path}")
                reliability_path = save_simultaneous_ci_reliability_violin_plot(
                    results=sci_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_reliability_violin.png"),
                )
                output_paths.append(reliability_path)
                print(f"Saved plot: {reliability_path}")
                violin_vs_n_path = save_simultaneous_ci_violin_vs_n_plot(
                    results=sci_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_violin_vs_n.png"),
                )
                output_paths.append(violin_vs_n_path)
                print(f"Saved plot: {violin_vs_n_path}")

            for cm_name in ("none", "bonferroni", "max_t"):
                cm_null_rows = [r for r in sci_results if r.ci_method == cm_name and r.condition == "null"]
                if not cm_null_rows:
                    continue
                cov = sum(r.all_covered for r in cm_null_rows) / sum(r.n_reps for r in cm_null_rows)
                width = sum(r.total_width for r in cm_null_rows) / sum(r.n_reps for r in cm_null_rows)
                score = sum(r.total_score for r in cm_null_rows) / sum(r.n_reps for r in cm_null_rows)
                key_metrics[f"simultaneous_ci_{cm_name}_coverage"] = float(cov)
                key_metrics[f"simultaneous_ci_{cm_name}_avg_width"] = float(width)
                key_metrics[f"simultaneous_ci_{cm_name}_avg_score"] = float(score)
            key_metrics["simultaneous_ci_n_results"] = len(sci_results)

        if "ppi" in modes:
            active_tests = args.tests if args.tests else [m.name for m in PPI_TEST_METHODS]
            print(f"\npvalues simulation (PPI-corrected) -- tests={active_tests}")
            jb_sources = build_judge_bias_sources()
            if args.eval_types:
                requested = set(args.eval_types)
                jb_sources = [s for s in jb_sources if s.eval_type in requested]
            if not jb_sources:
                raise ValueError("No JudgeBiasSources left after filtering.")
            print(f"  {len(jb_sources)} scenarios, reps={args.reps}, n_boot={args.ppi_n_boot}, alpha={args.alpha}")

            ppi_results = run_ppi_simulation(
                jb_sources, active_tests=active_tests, n_reps=args.reps, n_boot=args.ppi_n_boot,
                progress_mode=args.progress, seed=args.seed, n_workers=getattr(args, "workers", 1),
            )
            print_ppi_report(ppi_results, alpha=args.alpha)

            run_stem = f"pvalues_ppi_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_ppi(results=ppi_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False))
            if args.plots == "save":
                plot_path = save_ppi_typeI_plot(results=ppi_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected.png"))
                output_paths.append(plot_path)
                print(f"Saved plot: {plot_path}")
                if any(r.test in _PPI_NONSTANDARD_TESTS for r in ppi_results):
                    nonstd_plot_path = save_ppi_typeI_plot(
                        results=ppi_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_plot_path)
                    print(f"Saved plot: {nonstd_plot_path}")

            c_tot = sum(r.corrected_rejects for r in ppi_results)
            u_tot = sum(r.uncorrected_rejects for r in ppi_results)
            n_tot = sum(r.n_reps for r in ppi_results)
            key_metrics["ppi_n_results"] = len(ppi_results)
            key_metrics["ppi_mean_corrected_type1"] = float(c_tot / n_tot) if n_tot else float("nan")
            key_metrics["ppi_mean_uncorrected_type1"] = float(u_tot / n_tot) if n_tot else float("nan")

            if not getattr(args, "no_effect_check", False):
                effect_reps = getattr(args, "effect_reps", 200)
                effect_gold_mc = getattr(args, "effect_gold_mc", 3000)
                print(f"\npvalues simulation (PPI-corrected, effect-size check) -- effect_reps={effect_reps}, gold_mc={effect_gold_mc}")
                effect_results = run_ppi_effect_check(
                    jb_sources, active_tests=active_tests, n_reps=effect_reps, n_boot=args.ppi_n_boot,
                    gold_null_mc=effect_gold_mc, progress_mode=args.progress, seed=args.seed + 1,
                    n_workers=getattr(args, "workers", 1),
                )
                print_ppi_effect_report(effect_results, alpha=args.alpha)

                effect_stem = f"pvalues_ppi_effect_reps{effect_reps}_{stamp}"
                if effect_results:
                    if args.save_results == "save":
                        output_paths += save_results_artifacts_ppi_effect(
                            results=effect_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=effect_stem,
                            latex=getattr(args, "latex", False),
                        )
                    if args.plots == "save":
                        effect_plot_path = save_ppi_effect_plot(
                            results=effect_results, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{effect_stem}_bias_coverage_width.png"),
                        )
                        output_paths.append(effect_plot_path)
                        print(f"Saved plot: {effect_plot_path}")
                        if any(r.test in _PPI_NONSTANDARD_TESTS for r in effect_results):
                            nonstd_effect_plot_path = save_ppi_effect_plot(
                                results=effect_results, alpha=args.alpha,
                                out_path=str(Path(plots_dir) / f"{effect_stem}_bias_coverage_width_nonstandard.png"),
                                nonstandard=True,
                            )
                            output_paths.append(nonstd_effect_plot_path)
                            print(f"Saved plot: {nonstd_effect_plot_path}")

                    key_metrics["ppi_effect_n_results"] = len(effect_results)
                    finite_z = [r.bias_z for r in effect_results if np.isfinite(r.bias_z)]
                    key_metrics["ppi_effect_mean_abs_bias_z"] = float(np.mean(np.abs(finite_z))) if finite_z else float("nan")
                    finite_cov = [r.coverage for r in effect_results if np.isfinite(r.coverage)]
                    key_metrics["ppi_effect_mean_coverage"] = float(np.mean(finite_cov)) if finite_cov else float("nan")

        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics=key_metrics, duration_s=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return CaseResult(case_name=CASE_NAME, status="error", error=str(exc), duration_s=time.time() - t0)
