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
  across p-value correction strategies (holm/bonferroni/fdr_bh/hochberg/
  shaffer/friedman_nemenyi/max_t/romano_wolf/westfall_young), via
  ``scenarios.synthetic.build_multiarm_sources``, sweeping the SAME shape
  catalog ``build_pair_sources`` uses, generalized to k arms. hochberg/
  shaffer are closed-form refinements of holm (see
  ``evalstats.core.stats_utils.correct_pvalues``); romano_wolf/
  westfall_young are step-down max-T procedures (see
  ``_stepdown_max_t_pvalues``) that refine max_t's single-step joint
  critical value by recomputing the max only over not-yet-rejected pairs at
  each step -- all four exist to recover power lost to holm/bonferroni when
  pairwise comparisons are positively correlated, which repeated-measures/
  shared-item designs (the same participants or evaluation items
  contributing to multiple comparisons) produce routinely.
- ``simultaneous_ci``: family-wise CI coverage and average per-comparison
  width for the three simultaneous-CI constructions with a well-established
  dual to multiarm's p-value corrections -- ``none`` (naive per-pair CI, no
  simultaneous adjustment -- the "why do you need any correction?"
  baseline), Bonferroni t-intervals, and max-T (single-step studentized
  bootstrap, what ``evalstats.core.paired.all_pairwise`` uses by default)
  -- forced side by side on the SAME draw with the SAME point-estimate
  method held fixed (bypassing ``all_pairwise``'s own automatic
  method-based routing for max-T/Bonferroni), on the identical k-arm
  sources ``multiarm`` uses. multiarm's other corrections (holm/fdr_bh/
  hochberg/shaffer/friedman_nemenyi/romano_wolf/westfall_young) have no
  such CI dual -- holm/fdr_bh/hochberg/shaffer are p-value-only
  adjustments, friedman_nemenyi operates on rank differences rather than
  the raw mean-difference scale a CI needs, and romano_wolf/westfall_young's
  step-down critical value varies per rejection step rather than being one
  fixed value a CI could be built from. This is the evidence for why max-T
  is the harness's default simultaneous-CI method: it should hit nominal
  coverage same as Bonferroni (unlike ``none``, which should visibly
  under-cover as k grows), so a narrower average width at matching coverage
  is what actually distinguishes it from Bonferroni.

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
import re
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
    from evalstats.core.paired import (
        pairwise_differences, all_pairwise, friedman_nemenyi,
        _bonferroni_simultaneous_cis, _simultaneous_cis_router, _max_stat_simultaneous_cis,
        _sidak_simultaneous_cis, _joint_bootstrap_scaled_simultaneous_cis,
    )
    from evalstats.core.stats_utils import correct_pvalues, rescaled_ci
    from evalstats.core.resampling import (
        bayes_bootstrap_means_1d, tango_paired_ci_mean, tango_paired_ci_from_diffs, logit_t_ci_1d,
    )
    from evalstats.tests import (
        _ppi_two_sample,
        _ppi_two_sample_midrank_corrected,
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
        _ppi_kruskal_wallis_pairwise_corrected,
        _ppi_lmm_p_value,
        _kw_pairwise_thetas,
        _mcnemar_p,
    )
    from evalstats.core.mixed_effects import _fit_lmm_general, _get_fe_vcov_sm

from ..latex_tables import booktabs_table, escape_latex, eval_type_label
from ..scenarios import CIPairSource, MultiArmSource, JudgeBiasSource, EVAL_TYPES, EVAL_TYPE_SCALE_BOUNDS
from ..scenarios.synthetic import (
    SCENARIO_SUITES,
    build_pair_sources,
    build_multiarm_sources,
    build_judge_bias_sources,
    build_ppi_power_sources,
    build_ppi_power_reinforcing_sources,
    build_ppi_power_nobias_sources,
    build_ppi_comparison_label_frac_sources,
    build_ppi_nlab_grid_sources,
    build_ppi_factorial_sources,
    PPI_COMPARISON_MODERATE_EFFECT_FRAC,
    PPI_FACTORIAL_EFFECT_FRACS,
    PPI_FACTORIAL_N_VALUES,
    PPI_FACTORIAL_NLAB_VALUES,
    PPI_ALIGNMENT_HUMAN_NOISE_LEVELS,
    generate_judge_bias_cell,
    measure_judge_alignment,
    measure_human_human_alignment,
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
    CORR_SIDAK,
    CORR_BOOT,
    CANONICAL_SIMULTANEOUS_CI_METHODS,
    CORR_NONE,
    PPI_TEST_METHODS,
    PPI_OFFICIAL_TEST_METHODS,
    TTEST,
    TTEST_WELCH,
    MW_NAIVE,
    MWU_CORR,
    ANOVA_IND,
    ANOVA_REP,
    FRIEDMAN,
    KRUSKAL_NAIVE,
    KRUSKAL_CORR,
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

# Multiarm analogue of SIMULTANEOUS_CI_PLOT_METHODS below: `none`'s FWER is
# so far above nominal alpha (no correction at all) that plotting it on the
# same linear axis as every other correction (which all cluster near alpha)
# squashes the comparison save_multiarm_fwer_vs_k_plot /
# save_multiarm_fwer_vs_n_plot exist to show; `none` is still in the
# printed/logged report tables and the CSV.
MULTIARM_PLOT_METHODS = [m for m in MULTIARM_CORRECTION_METHODS if m.name != CORR_NONE.name]

# Every simultaneous-CI construction _run_simultaneous_ci_cell can produce:
# `none`/`bonferroni`/`max_t` (see SIMULTANEOUS_CI_METHODS) plus `sidak`/
# `boot` (see CANONICAL_SIMULTANEOUS_CI_METHODS' comment in methods.py).
# `none`/`bonferroni`/`sidak`/`boot` are all built on the scenario's
# eval-type-canonical CI method (Tango for binary, Logit-t for continuous/
# likert -- see _canonical_ci_func below), NOT
# --multiarm-method; `max_t` is the one exception, since it needs a
# bootstrap-compatible method to resample from and keeps using
# --multiarm-method (bootstrap_t by default) regardless of eval type. Report/
# plot functions filter this down to whichever names are actually present in
# a given results list, so a `grades`-only sweep (no canonical CI wired up --
# see _canonical_ci_func) simply never shows `sidak`/`boot`.
ALL_SIMULTANEOUS_CI_METHODS = SIMULTANEOUS_CI_METHODS + CANONICAL_SIMULTANEOUS_CI_METHODS

# `none` stays in _run_simultaneous_ci_cell's data collection and in
# print_simultaneous_ci_report's tables (it's the "why do you need any
# correction at all" baseline there), but is dropped from every
# simultaneous_ci *plot* -- it's so far below nominal family-wise coverage
# (even built on the canonical per-pair CI, which is well-calibrated
# per-comparison but never adjusted for multiplicity) that it squashes the
# Bonferroni-vs-max-T-vs-Sidak-vs-bootstrap comparison those plots exist to
# show.
SIMULTANEOUS_CI_PLOT_METHODS = [m for m in ALL_SIMULTANEOUS_CI_METHODS if m.name != CORR_NONE.name]


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
    """Wilcoxon signed-rank p-value via scipy's default method="auto".

    Deliberately does NOT override method= for speed. scipy's "auto" does
    genuinely different (not just slower) work for small, tied/discrete-
    valued samples (binary 0/1 diffs, Likert-scale integer diffs, etc.) at
    roughly n<=13: it runs exhaustive permutation enumeration for a
    rigorously tie-corrected exact p-value, which is legitimately expensive
    (up to ~300ms/call at n=13 in scipy 1.17.x) but is the CORRECT p-value.
    Forcing method="exact" instead is much faster, but per scipy's own docs
    "method='exact' no longer calculates the exact p-value" once ties/zeros
    are present -- empirically this shifted individual p-values by up to
    0.125 and measurably changed small-n FWER calibration in this harness's
    own null-hypothesis checks, not just decision-level noise that would
    wash out. Given every pair/rep goes through this function as evalstats'
    canonical raw p-value (see _compute_multiarm_metrics), correctness at
    small n matters more than the wall-clock cost -- eat the cost rather
    than risk silently drifting already-reported simulation results.
    """
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
    sc_idx, n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed, corrections = args
    return _run_multiarm_cell(
        _MULTIARM_SOURCES[sc_idx], n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed,
        corrections=corrections,
    )


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
    total_time: float = 0.0
    """Total wall-clock seconds for THIS correction's own computation, summed
    across all n_reps of this condition -- e.g. romano_wolf/westfall_young's
    own step-down resampling, max_t's own router call, etc. Does NOT include
    shared per-rep setup (score generation, Wilcoxon p-values) except for
    `none`, which that setup is attributed to (see _run_multiarm_cell /
    _compute_multiarm_metrics)."""


def _bootstrap_t_matrix(
    diffs_mat: np.ndarray, n_bootstrap: int, rng: np.random.Generator,
    resample_mode: str, batch_size: int = 256,
    arm_scores: np.ndarray | None = None, pair_indices: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint resampling core shared by _stepdown_max_t_pvalues (Romano-Wolf/
    Westfall-Young step-down) and max_t's single-step critical value.

    Draws n_bootstrap joint resamples of the studentized per-pair statistic
    ``t_p = mean(diffs_p) / se(diffs_p)`` and returns ``(t_abs, t_obs)``:
    ``t_abs`` is ``(k_pairs, n_bootstrap)`` -- ``|T|`` per pair per replicate
    -- and ``t_obs`` is ``(k_pairs,)`` -- the observed ``|T|``. Two
    resample_mode options generate the joint null distribution differently:

    - ``"bootstrap"`` (Romano-Wolf / max_t): resample items/participants
      (rows of ``diffs_mat``, shared across pairs) with replacement, then
      studentize and recenter each bootstrap draw at its own pair's
      *observed* statistic -- the same nonparametric bootstrap-t null
      evalstats.core.paired's ``_max_stat_simultaneous_cis`` uses for
      max_t's single-step critical value (bootstrap_t branch): identical
      formula, so max_t's critical value is exactly ``quantile(t_abs.max(
      axis=0), ci)`` -- i.e. step 0 of the step-down procedure, before any
      rejections -- computed from this SAME matrix. See
      _compute_multiarm_metrics for where that sharing happens.
    - ``"permutation"`` (Westfall-Young): independently, per item/
      participant, draw a uniformly random relabeling of the ``k`` arms
      (``arm_scores``' rows) and recompute every pair's diff from the
      relabeled scores for that item -- the genuine label-permutation null,
      not merely a sign flip of the diffs. This is exact under
      exchangeability of the k arms for any number of arms: unlike sign-
      flipping the diff vector (which only coincides with a real relabeling
      when k=2, since the group of relabelings has k! elements but sign
      flips only give 2), permuting the raw per-item scores and re-deriving
      every pairwise diff is, by construction, always one of the k!
      relabelings, so the joint null distribution it draws from is the
      correct one for any k. Requires ``arm_scores`` (k arms x m items) and
      ``pair_indices`` (row-index pairs into ``arm_scores``, parallel to
      ``diffs_mat``'s rows).

    Both branches avoid ever materializing a ``(k_pairs, b, m)`` gathered
    array (the naive "resample then reduce" approach), which dominates
    memory/time at this harness's largest cells (k=20 -> 190 pairs, n up to
    500): "bootstrap" resamples the SAME item index across every pair, so
    each replicate's per-pair mean/variance is a fixed linear combination of
    the original per-item diffs (weighted by "how many times item i was
    drawn"), computable via a ``(b, m)`` counts matrix (one ``bincount``)
    and two BLAS matmuls (``diffs_mat @ counts.T`` for the mean,
    ``diffs_mat**2 @ counts.T`` for the second moment) instead of a
    ``diffs_mat[:, idx]`` gather. "permutation" similarly replaces the
    ``relabeled[:, :, pair_i] - relabeled[:, :, pair_j]`` fancy-index
    differencing with one matmul against a fixed signed ``(k_arms,
    k_pairs)`` pairing matrix, since taking a pairwise diff is itself a
    linear map of the relabeled per-item arm vector. Verified bit-close
    (``np.allclose``) against the original gather-based formulas; ~12-27x
    faster for "bootstrap" and ~2.3x faster for "permutation" at k=20/n=500,
    with no regression at small k/n.
    """
    k_pairs, m = diffs_mat.shape
    means = diffs_mat.mean(axis=1)
    ses = diffs_mat.std(axis=1, ddof=1) / np.sqrt(m)
    ses_safe = np.where(ses > 1e-12, ses, 1.0)
    t_obs = np.abs(means) / ses_safe
    # m is this harness's smallest swept sample size (always >= 10), so
    # m == 1 never happens in practice -- guarded only so this never raises
    # (matching np.std(ddof=1)'s own non-crashing nan-on-degenerate-df
    # behavior) instead of Python's ZeroDivisionError on m/(m-1).
    ddof1_factor = m / (m - 1) if m > 1 else 1.0

    if resample_mode == "permutation":
        if arm_scores is None or pair_indices is None:
            raise ValueError("'permutation' resample_mode requires arm_scores and pair_indices")
        k_arms = arm_scores.shape[0]
        arm_scores_t = arm_scores.T  # (m, k_arms)
        pair_i = np.array([p[0] for p in pair_indices])
        pair_j = np.array([p[1] for p in pair_indices])
        # Signed pairing matrix: diff_perm[..., p] = relabeled[..., pair_i[p]]
        # - relabeled[..., pair_j[p]] -- see docstring.
        pairing_matrix = np.zeros((k_arms, k_pairs))
        pairing_matrix[pair_i, np.arange(k_pairs)] = 1.0
        pairing_matrix[pair_j, np.arange(k_pairs)] = -1.0
    else:
        diffs_sq = diffs_mat**2  # hoisted -- fixed across every batch below

    t_abs_chunks: list[np.ndarray] = []
    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        b = end - start
        if resample_mode == "bootstrap":
            idx = rng.integers(0, m, size=(b, m))
            # counts[draw, item] = how many times `item` was drawn in that
            # replicate -- see docstring for why this replaces the gather.
            flat_idx = idx + (np.arange(b)[:, None] * m)
            counts = np.bincount(flat_idx.ravel(), minlength=b * m).reshape(b, m).astype(diffs_mat.dtype)
            b_means = (diffs_mat @ counts.T) / m  # (k_pairs, b)
            sq_means = (diffs_sq @ counts.T) / m  # (k_pairs, b)
            var_unbiased = np.maximum(sq_means - b_means**2, 0.0) * ddof1_factor
            b_ses = np.sqrt(var_unbiased) / np.sqrt(m)
            b_ses_safe = np.where(b_ses > 1e-12, b_ses, 1.0)
            t_vals = (b_means - means[:, None]) / b_ses_safe
        else:  # "permutation" -- per-item random relabeling of the k arms
            # perm[b, j] is a uniformly random permutation of {0, ..., k_arms-1}
            # for bootstrap draw b, item j (argsort of iid uniforms is the
            # standard trick for batched random permutations).
            perm = np.argsort(rng.random(size=(b, m, k_arms)), axis=2)
            relabeled = np.take_along_axis(
                np.broadcast_to(arm_scores_t[None, :, :], (b, m, k_arms)), perm, axis=2,
            )  # (b, m, k_arms): item j's scores relabeled across arms
            diff_perm = (relabeled.reshape(b * m, k_arms) @ pairing_matrix).reshape(b, m, k_pairs)
            b_means = diff_perm.mean(axis=1).T  # (k_pairs, b)
            b_ses = (diff_perm.std(axis=1, ddof=1) / np.sqrt(m)).T
            b_ses_safe = np.where(b_ses > 1e-12, b_ses, 1.0)
            t_vals = b_means / b_ses_safe
        t_abs_chunks.append(np.abs(t_vals))
    t_abs = np.concatenate(t_abs_chunks, axis=1)  # (k_pairs, n_bootstrap)
    return t_abs, t_obs


def _single_step_max_t_pvalues(t_abs: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
    """max_t's single-step FWER p-value per pair, from an already-computed
    _bootstrap_t_matrix() draw -- the max-over-all-pairs-per-replicate
    distribution (``t_abs.max(axis=0)``) IS max_t's joint null, identical to
    what evalstats.core.paired._apply_max_t_cis derives independently from
    its own separate resample. Letting _compute_multiarm_metrics reuse one
    shared draw for both max_t and romano_wolf (whose step-down procedure
    needs this same matrix anyway) avoids drawing and reducing a second
    n_bootstrap x k_pairs resample just to recompute the same statistic."""
    m_b = t_abs.max(axis=0)  # (n_bootstrap,) -- max over ALL pairs per replicate
    b_total = t_abs.shape[1]
    extreme = (m_b[np.newaxis, :] >= t_obs[:, np.newaxis]).sum(axis=1)  # (k_pairs,)
    return (extreme + 1) / (b_total + 1)


def _stepdown_max_t_pvalues(
    diffs_mat: np.ndarray, n_bootstrap: int, rng: np.random.Generator,
    resample_mode: str, batch_size: int = 256,
    arm_scores: np.ndarray | None = None, pair_indices: list[tuple[int, int]] | None = None,
    precomputed: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Step-down max-|T| FWER p-values: Romano & Wolf (2005)'s bootstrap
    step-down, or its permutation analogue, Westfall & Young (1993)'s
    step-down min-P/max-T.

    Unlike single-step max-T (this harness's `max_t`), which uses ONE joint
    critical value for every pair, the step-down refinement here recomputes
    the max only over pairs not yet rejected at each step (starting from the
    pair with the largest observed |t|, working down), which strictly
    dominates single-step max-T in power for the same strong FWER guarantee
    -- exactly the "recover power lost to Holm/Bonferroni when comparisons
    are positively correlated" case repeated-measures designs create, since
    shared items/participants make every pair's diffs correlated.

    Returns one FWER-adjusted p-value per row of ``diffs_mat`` (same pair
    order), monotonized via a running max along the testing order (the same
    reformulation Holm's own adjusted p-values use) so they are directly
    comparable to alpha.

    Parameters
    ----------
    precomputed : tuple[np.ndarray, np.ndarray], optional
        ``(t_abs, t_obs)`` already computed by _bootstrap_t_matrix -- pass
        this to skip resampling entirely and reuse an existing draw (e.g.
        shared with max_t's single-step p-value; see
        _compute_multiarm_metrics). When ``None`` (default), resamples
        fresh via _bootstrap_t_matrix exactly as before this parameter
        existed -- passing nothing changes nothing.
    """
    if precomputed is not None:
        t_abs, t_obs = precomputed
        k_pairs = t_abs.shape[0]
    else:
        k_pairs = diffs_mat.shape[0]
        t_abs, t_obs = _bootstrap_t_matrix(
            diffs_mat, n_bootstrap, rng, resample_mode, batch_size, arm_scores, pair_indices,
        )
    b_total = t_abs.shape[1]

    order = np.argsort(-t_obs)  # descending observed |t|: tested first
    t_abs_sorted = t_abs[order]
    # suffix_max[step] = max over pairs tested at or after `step` -- the
    # step-down "remaining hypotheses" set, per bootstrap draw.
    suffix_max = np.maximum.accumulate(t_abs_sorted[::-1], axis=0)[::-1]

    # Both loops below are pure functions of `order` (testing sequence), so
    # they vectorize directly: compare/count and the running max are taken
    # along that sequence, then scattered back to original pair indices in
    # one assignment instead of a per-pair Python loop (k_pairs up to 190).
    t_obs_sorted = t_obs[order]
    extreme_counts = (suffix_max >= t_obs_sorted[:, None]).sum(axis=1)
    raw_step_p_sorted = (extreme_counts + 1) / (b_total + 1)
    adjusted_sorted = np.minimum(np.maximum.accumulate(raw_step_p_sorted), 1.0)

    adjusted = np.empty(k_pairs)
    adjusted[order] = adjusted_sorted
    return adjusted


def _compute_multiarm_metrics(
    *, scores: np.ndarray, labels: list[str], method: str, corrections: list[str],
    n_bootstrap: int, alpha: float, statistic: str, rng: np.random.Generator,
) -> tuple[dict[str, tuple[bool, bool]], dict[str, float]]:
    """Compute (any_reject, best_selected) for every correction strategy.

    none/holm/bonferroni/fdr_bh/hochberg/shaffer correct the Wilcoxon
    signed-rank p-value (evalstats' canonical, eval-type-agnostic paired
    test -- unlike Tango/Logit-t in --mode simultaneous_ci, one test covers
    binary/continuous/likert/grades alike, so no per-eval-type branching is
    needed here) via _safe_wilcoxon_p on each pair's per_input_diffs, rather
    than --multiarm-method's raw p-value (bootstrap_t by default).
    per_input_diffs/point_diff are built directly from `scores` (a plain
    per-input difference and its mean/median -- no resampling involved), not
    via all_pairwise(method=method, ...): that used to run the *full*
    method-specific bootstrap per pair (e.g. bootstrap_t's two independent
    n_bootstrap-sized resamples, one for the CI and one for the p-value)
    purely to obtain per_input_diffs/point_diff, which need no resampling at
    all -- wasting O(pairs * n_bootstrap) draws every rep/condition/cell for
    results this function never read. hochberg/shaffer are closed-form
    reweightings of the same Wilcoxon p-values (see
    evalstats.core.stats_utils.correct_pvalues; shaffer additionally needs
    `n_groups=k`, the number of arms, to derive its all-pairwise divisor
    sequence).

    max_t/romano_wolf/westfall_young are the exception: they still need
    genuine resampling, since Wilcoxon has no joint max-T analogue. `max_t`
    resamples from `scores` directly via evalstats.core.paired's
    _max_stat_simultaneous_cis, called directly rather than through
    _simultaneous_cis_router -- the router's only other job is falling back
    to a Bonferroni CI on degenerate bootstrap results, but this harness
    never reads that CI (only whether the max-T draw succeeded), so there's
    nothing for the router's required `results` argument to feed; skipping
    it avoids building an unused dict[pair, PairedDiffResult] stand-in on
    every call just to satisfy that argument. For bootstrap-compatible
    methods the resulting p-values are *single-step* max-T FWER-controlled
    p-values -- each is the min p-value commensurate with the simultaneous
    CI that was reported to the user. For non-bootstrap methods
    (permutation) max_t falls back to the raw (marginal) p-value, matching
    what all_pairwise itself does when its router falls back to Bonferroni
    (it only widens the CI, not `.p_value`).

    `romano_wolf`/`westfall_young` are the genuine *step-down* max-T
    procedures (see _stepdown_max_t_pvalues) -- unlike max_t, they don't go
    through all_pairwise/_simultaneous_cis_router at all, since step-down
    needs the full per-bootstrap-draw statistic matrix (not just the single
    joint critical value the router returns) to recompute the max over
    shrinking "not yet rejected" subsets. They're built directly off the
    same method-invariant per_input_diffs hochberg/shaffer/etc. use.

    `boot` is the multiarm analogue of --mode simultaneous_ci's `boot`:
    unlike max_t/romano_wolf/westfall_young, it is NOT tied to
    --multiarm-method -- like none/holm/bonferroni/etc. it always widens the
    canonical Wilcoxon p-value, but using a joint bootstrap critical value
    (the max-over-all-pairs studentized-mean resample -- same construction
    max_t/romano_wolf use) rather than a fixed, correlation-blind factor.
    Concretely: the joint critical value is translated to an equivalent
    alpha_eff (mirroring evalstats.core.paired._joint_bootstrap_scaled_
    simultaneous_cis's z<->alpha translation for CIs), and raw_p is rescaled
    by alpha/alpha_eff -- the same "scale the raw p-value by the correction
    factor" pattern Bonferroni's own adjustment uses, just with a
    resampled, correlation-aware factor instead of a fixed k.

    `boot` always uses the exact "bootstrap" resample (mean-based item
    bootstrap, romano_wolf's own resample_mode regardless of
    --multiarm-method), so it and `romano_wolf` always share ONE draw when
    both are requested; `max_t` additionally joins that shared draw when
    its own construction happens to match (method="bootstrap_t"/
    statistic="mean", the defaults) -- see _bootstrap_t_matrix's docstring.

    `friedman_nemenyi` is unaffected either way -- already its own
    rank-based omnibus + post-hoc test, unrelated to `method`.

    Returns
    -------
    tuple[dict[str, tuple[bool, bool]], dict[str, float]]
        ``(results, timings)`` -- *timings* maps each correction to its own
        wall-clock seconds (so e.g. romano_wolf/westfall_young's genuine
        step-down resampling shows up as slower than none/holm/bonferroni's
        closed-form reweighting in the report's Time(ms) column, instead of
        every correction row displaying the same aggregate "whole call"
        time). The one-time shared setup (diffs_by_pair/point_diff_by_pair/
        raw_p, and stepdown_corrections' diffs_mat) is folded into `none`'s
        and the first stepdown correction's timing respectively, since it's
        not fairly attributable to any other single correction.
    """
    results: dict[str, tuple[bool, bool]] = {}
    timings: dict[str, float] = {}
    k = len(labels)
    pairs = [(labels[i], labels[j]) for i in range(k) for j in range(i + 1, k)]

    _STEPDOWN_RESAMPLE_MODE = {"romano_wolf": "bootstrap", "westfall_young": "permutation"}
    non_friedman_non_maxt = [
        c for c in corrections
        if c not in ("friedman_nemenyi", "max_t", "boot") and c not in _STEPDOWN_RESAMPLE_MODE
    ]
    include_max_t = "max_t" in corrections
    include_boot = "boot" in corrections
    stepdown_corrections = [c for c in _STEPDOWN_RESAMPLE_MODE if c in corrections]

    if non_friedman_non_maxt or include_max_t or include_boot or stepdown_corrections:
        # Plain per-input differences and their mean/median -- no resampling
        # needed, unlike the method-specific bootstrap all_pairwise(method=
        # method, ...) used to run here just to throw away everything but
        # these two quantities (see docstring above).
        _t_setup0 = time.perf_counter()
        flat = scores.mean(axis=2) if scores.ndim == 3 else scores  # (k_arms, n)
        label_to_idx = {label: i for i, label in enumerate(labels)}
        diffs_by_pair: dict[tuple[str, str], np.ndarray] = {}
        point_diff_by_pair: dict[tuple[str, str], float] = {}
        for pair in pairs:
            a, b = pair
            d = flat[label_to_idx[a]] - flat[label_to_idx[b]]
            diffs_by_pair[pair] = d
            point_diff_by_pair[pair] = float(d.mean()) if statistic == "mean" else float(np.median(d))

        raw_p = np.array([_safe_wilcoxon_p(diffs_by_pair[pair]) for pair in pairs])
        pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}
        _setup_elapsed = time.perf_counter() - _t_setup0

        for correction in non_friedman_non_maxt:
            _t0 = time.perf_counter()
            if correction == "none":
                adj_p = raw_p
            elif correction == "shaffer":
                adj_p = correct_pvalues(raw_p, correction, n_groups=k)
            else:
                adj_p = correct_pvalues(raw_p, correction)
            has_any = bool(np.any(adj_p < alpha))
            best = labels[0]
            best_selected = True
            for other in labels[1:]:
                pair_idx = pair_to_idx.get((best, other))
                if pair_idx is None or not (adj_p[pair_idx] < alpha and point_diff_by_pair[(best, other)] > 0.0):
                    best_selected = False
                    break
            results[correction] = (has_any, best_selected)
            timings[correction] = time.perf_counter() - _t0
        if "none" in timings:
            # The one-time shared setup (diffs/point-diffs/Wilcoxon p-values)
            # is attributed to `none`, mirroring _run_simultaneous_ci_cell's
            # equivalent choice -- it's the row that most directly needs it,
            # not fairly split across every other correction.
            timings["none"] += _setup_elapsed

        # `boot` and `romano_wolf` always use the exact same fixed,
        # mean-based item-bootstrap construction (romano_wolf's
        # resample_mode is hardcoded to "bootstrap" regardless of
        # --multiarm-method/--statistic; `boot` -- evalstats' canonical-
        # Wilcoxon analogue of --mode simultaneous_ci's `boot` -- is
        # deliberately the same fixed construction for the same "canonical,
        # not --multiarm-method-tied" reason none/holm/bonferroni/etc. are),
        # so whenever both are requested they always share one draw, no
        # condition needed. `max_t` only joins that shared draw when its
        # OWN construction (tied to --multiarm-method) happens to be the
        # identical thing -- method="bootstrap_t" and statistic="mean" (the
        # CLI defaults) -- see _bootstrap_t_matrix's docstring. Falls back
        # to each computing its own resample independently (unchanged
        # behavior) whenever nothing needs to share.
        need_shared_matrix = include_boot or "romano_wolf" in stepdown_corrections
        max_t_matches_shared = method == "bootstrap_t" and statistic == "mean"
        share_max_t = include_max_t and need_shared_matrix and max_t_matches_shared
        diffs_mat = None
        pair_indices = None
        shared_t_abs = None
        shared_t_obs = None
        _shared_elapsed = 0.0
        _shared_owner = None  # which correction's timing bucket absorbs the shared resample
        if stepdown_corrections or include_boot:
            _t_stack0 = time.perf_counter()
            diffs_mat = np.stack([diffs_by_pair[pair] for pair in pairs], axis=0)
            pair_indices = [(label_to_idx[a], label_to_idx[b]) for a, b in pairs]
            _stack_elapsed = time.perf_counter() - _t_stack0
            if need_shared_matrix:
                _t_shared0 = time.perf_counter()
                shared_t_abs, shared_t_obs = _bootstrap_t_matrix(diffs_mat, n_bootstrap, rng, "bootstrap")
                _shared_elapsed = _stack_elapsed + (time.perf_counter() - _t_shared0)
                _shared_owner = "romano_wolf" if "romano_wolf" in stepdown_corrections else "boot"

        if include_max_t:
            _t0 = time.perf_counter()
            try:
                if share_max_t:
                    maxt_p = _single_step_max_t_pvalues(shared_t_abs, shared_t_obs)
                else:
                    # _max_stat_simultaneous_cis called directly (not via
                    # _simultaneous_cis_router) -- the router's only other
                    # job is falling back to _bonferroni_simultaneous_cis on
                    # degenerate bootstrap results, but this harness never
                    # reads the resulting CI values (`cis`), only whether
                    # the max-T draw succeeded (`sim_pvalues`) -- so there's
                    # nothing for a `results`/PairedDiffResult stand-in
                    # (formerly `results_stub`, built eagerly here on every
                    # call) to actually feed. Falling straight back to
                    # `raw_p` (unadjusted Wilcoxon) when the bootstrap
                    # returns empty matches what the router's own fallback
                    # would have produced as far as this harness could tell
                    # anyway (a safe placeholder, not a real Bonferroni CI
                    # the code ever used).
                    cis, max_t_pvalues = _max_stat_simultaneous_cis(
                        scores=scores, pairs=pairs, labels=labels, method=method,
                        ci=1.0 - alpha, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
                    )
                    maxt_p = np.array([max_t_pvalues[pair] for pair in pairs]) if cis else raw_p
                has_any = bool(np.any(maxt_p < alpha))
                best = labels[0]
                best_selected = True
                for other in labels[1:]:
                    pair_idx = pair_to_idx.get((best, other))
                    if pair_idx is None or not (maxt_p[pair_idx] < alpha and point_diff_by_pair[(best, other)] > 0.0):
                        best_selected = False
                        break
                results["max_t"] = (has_any, best_selected)
            except Exception:
                results["max_t"] = (False, False)
            # When shared, max_t's own incremental cost on top of the
            # already-computed matrix really is this small -- the resample
            # itself is charged to whichever of romano_wolf/boot is present
            # (see _shared_owner below), the correction(s) that intrinsically
            # need the full matrix regardless of whether max_t joins in.
            timings["max_t"] = time.perf_counter() - _t0

        if include_boot:
            _t0 = time.perf_counter()
            try:
                # Joint bootstrap critical value -- the max-over-all-pairs
                # studentized-mean distribution, the exact same construction
                # evalstats.core.paired._joint_bootstrap_critical_value uses
                # for --mode simultaneous_ci's `boot` -- translated to an
                # equivalent alpha (matching that construction's own
                # z<->alpha translation) and used to rescale the canonical
                # Wilcoxon p-value the same way Bonferroni rescales it with
                # a fixed, correlation-blind factor (alpha/k), except this
                # factor comes from a resampled joint null that DOES account
                # for correlation between comparisons.
                c = float(np.quantile(shared_t_abs.max(axis=0), 1.0 - alpha))
                alpha_eff = float(2.0 * (1.0 - scipy_stats.norm.cdf(c)))
                alpha_eff = min(max(alpha_eff, 1e-9), 1.0 - 1e-9)
                adj_p = np.minimum(raw_p * (alpha / alpha_eff), 1.0)
                has_any = bool(np.any(adj_p < alpha))
                best = labels[0]
                best_selected = True
                for other in labels[1:]:
                    pair_idx = pair_to_idx.get((best, other))
                    if pair_idx is None or not (adj_p[pair_idx] < alpha and point_diff_by_pair[(best, other)] > 0.0):
                        best_selected = False
                        break
                results["boot"] = (has_any, best_selected)
            except Exception:
                results["boot"] = (False, False)
            elapsed = time.perf_counter() - _t0
            if _shared_owner == "boot":
                elapsed += _shared_elapsed
            timings["boot"] = elapsed

        if stepdown_corrections:
            for i, correction in enumerate(stepdown_corrections):
                _t0 = time.perf_counter()
                try:
                    resample_mode = _STEPDOWN_RESAMPLE_MODE[correction]
                    if correction == "romano_wolf" and shared_t_abs is not None:
                        adj_p = _stepdown_max_t_pvalues(
                            diffs_mat, n_bootstrap, rng, resample_mode,
                            precomputed=(shared_t_abs, shared_t_obs),
                        )
                    else:
                        adj_p = _stepdown_max_t_pvalues(
                            diffs_mat, n_bootstrap, rng, resample_mode,
                            arm_scores=flat if resample_mode == "permutation" else None,
                            pair_indices=pair_indices if resample_mode == "permutation" else None,
                        )
                    has_any = bool(np.any(adj_p < alpha))
                    best = labels[0]
                    best_selected = True
                    for other in labels[1:]:
                        pair_idx = pair_to_idx.get((best, other))
                        if pair_idx is None or not (adj_p[pair_idx] < alpha and point_diff_by_pair[(best, other)] > 0.0):
                            best_selected = False
                            break
                    results[correction] = (has_any, best_selected)
                except Exception:
                    results[correction] = (False, False)
                elapsed = time.perf_counter() - _t0
                if correction == "romano_wolf" and _shared_owner == "romano_wolf":
                    # Includes the shared resample (see above) -- the one
                    # place its real cost is now charged.
                    elapsed += _shared_elapsed
                elif i == 0 and _shared_owner is None:
                    # np.stack's construction cost is shared setup for every
                    # stepdown correction; attributed to the first one rather
                    # than double-counted or arbitrarily split -- only when
                    # nothing else already absorbed it (_shared_owner is set
                    # whenever boot/romano_wolf triggered a shared matrix,
                    # which already includes this same stack cost).
                    elapsed += _stack_elapsed
                timings[correction] = elapsed

    if "friedman_nemenyi" in corrections:
        _t0 = time.perf_counter()
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
        timings["friedman_nemenyi"] = time.perf_counter() - _t0

    return results, timings


def _run_multiarm_cell(
    source: MultiArmSource, n: int, runs: int, k_arms: int, n_reps: int, n_bootstrap: int,
    alpha: float, multiarm_method: str, statistic: str, seed, corrections: list[str] | None = None,
) -> list[MultiArmResult]:
    labels = [f"arm_{i}" for i in range(k_arms)]
    if corrections is None:
        corrections = [m.name for m in MULTIARM_CORRECTION_METHODS]
    rng = np.random.default_rng(seed)

    agg_any: dict[tuple[str, str], int] = {(c, cond): 0 for c in corrections for cond in ("null", "alt")}
    agg_best: dict[tuple[str, str], int] = {(c, cond): 0 for c in corrections for cond in ("null", "alt")}
    # Per-(correction, condition), not a single per-condition total -- each
    # correction's own wall-clock cost, so e.g. romano_wolf/westfall_young's
    # genuine step-down resampling shows up as slower than none/holm/
    # bonferroni's closed-form reweighting in the report's Time(ms) column,
    # instead of every correction row displaying the same aggregate "whole
    # rep" time (see _compute_multiarm_metrics's per-correction timings).
    agg_time: dict[tuple[str, str], float] = {(c, cond): 0.0 for c in corrections for cond in ("null", "alt")}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_reps):
            for condition, delta in (("null", 0.0), ("alt", source.alt_delta)):
                # Score generation is shared setup, attributed to `none` --
                # not fairly attributable to any other single correction.
                _t_none0 = time.perf_counter()
                scores = source.generate_scores(rng, n, runs, k_arms, delta)
                _scores_elapsed = time.perf_counter() - _t_none0
                metrics, timings = _compute_multiarm_metrics(
                    scores=scores, labels=labels, method=multiarm_method, corrections=corrections,
                    n_bootstrap=n_bootstrap, alpha=alpha, statistic=statistic, rng=rng,
                )
                if "none" in timings:
                    timings["none"] += _scores_elapsed
                for correction in corrections:
                    any_reject, best_selected = metrics.get(correction, (False, False))
                    if any_reject:
                        agg_any[(correction, condition)] += 1
                    if best_selected:
                        agg_best[(correction, condition)] += 1
                    agg_time[(correction, condition)] += timings.get(correction, 0.0)

    return [
        MultiArmResult(
            eval_type=source.eval_type, label=source.label, n=n, k=k_arms, correction=correction,
            condition=condition, n_reps=n_reps, any_reject=agg_any[(correction, condition)],
            best_selected=agg_best[(correction, condition)],
            total_time=agg_time[(correction, condition)],
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
    seed: int = 42, n_workers: int = 1, corrections: list[str] | None = None,
) -> list[MultiArmResult]:
    global _MULTIARM_SOURCES
    _MULTIARM_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = _multiarm_style_cells(sources, sample_sizes, k_values)

    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, runs, k, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed, corrections)
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
        powers: list[float] = []
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
            powers.append(power)
        ax.set_xlabel("FWER (null)")
        ax.set_ylabel("Best-arm selection power (alt)")
        ax.set_title(f"eval type: {et}")
        ax.set_xlim(-0.02, max(0.3, alpha * 4))
        # Zoom to the actual power spread rather than a fixed [0, 1] -- power
        # can cluster near either end (uniformly low under a strict per-pair
        # rejection requirement, or uniformly high at large n), and a full
        # [0, 1] axis squashes that spread into an unreadable sliver either
        # way. No artificial 0.0 floor seed -- that would defeat the zoom
        # whenever power clusters near 1.0.
        if powers:
            pow_lo, pow_hi = min(powers), max(powers)
            pow_pad = max(0.01, (pow_hi - pow_lo) * 0.15)
            ax.set_ylim(max(-0.02, pow_lo - pow_pad), min(1.02, pow_hi + pow_pad))
        else:
            ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Family-Wise Error Rate vs. Best-Arm Selection Power\nNominal alpha = {alpha}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_multiarm_fwer_vs_k_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """FWER and best-arm power as a function of k (number of arms), one curve per
    correction method, collapsed across eval types and sample sizes -- the
    multiarm analogue of save_simultaneous_ci_coverage_width_vs_k_plot (same
    two-panel line-plot style: exact integer x-ticks pinned to the k values
    actually swept, FWER y-axis zoomed to the actual spread rather than a
    fixed [0, ...]). Only plots MULTIARM_PLOT_METHODS (every registered
    correction except `none` -- see that list's comment for why; `none` is
    still in the printed/logged report tables and the CSV).
    Only produced when more than one k value was swept; returns out_path
    unchanged (without writing) if all results share the same k."""
    import matplotlib.pyplot as plt

    ks_present = sorted({r.k for r in results})
    if len(ks_present) < 2:
        return out_path

    fig, (ax_fwer, ax_pow) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    ax_fwer.axhline(alpha, color="black", linewidth=1.0, linestyle="--", label=f"α={alpha}")
    ax_fwer.axhspan(max(0.0, alpha - 0.02), alpha + 0.02, color="#DDDDDD", alpha=0.4, zorder=0)

    all_fwer_vals: list[float] = [alpha]
    all_pow_vals: list[float] = []
    for m in MULTIARM_PLOT_METHODS:
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
            all_fwer_vals.extend(ys_fwer)
            all_pow_vals.extend(ys_pow)

    ax_fwer.set_xlabel("k (number of arms)")
    ax_fwer.set_ylabel("FWER (null)")
    ax_fwer.set_title("FWER vs. number of arms")
    # Zoom to the actual FWER spread (plus the nominal alpha line) rather
    # than a fixed [0, ...] -- with `none` dropped from this plot
    # (MULTIARM_PLOT_METHODS), every remaining curve usually clusters near
    # alpha, and a floor of 0.0 squashes that spread into an unreadable
    # sliver at the bottom (see save_simultaneous_ci_coverage_width_vs_k_plot's
    # identical fix for coverage).
    fwer_lo, fwer_hi = min(all_fwer_vals), max(all_fwer_vals)
    fwer_pad = max(0.005, (fwer_hi - fwer_lo) * 0.15)
    ax_fwer.set_ylim(max(0.0, fwer_lo - fwer_pad), fwer_hi + fwer_pad)
    ax_fwer.set_xticks(ks_present)

    ax_pow.set_xlabel("k (number of arms)")
    ax_pow.set_ylabel("Best-arm selection power (alt)")
    ax_pow.set_title("Power vs. number of arms")
    # Zoom to the actual power spread rather than a fixed [0, 1] -- power is
    # often concentrated near one end (uniformly low, or uniformly high as
    # with best-arm selection at large n), and a full [0, 1] axis squashes
    # that spread the same way an unzoomed FWER axis would (see above). No
    # artificial 0.0 floor seed (unlike FWER's alpha-line seed) -- that would
    # defeat the zoom whenever power clusters near 1.0.
    if all_pow_vals:
        pow_lo, pow_hi = min(all_pow_vals), max(all_pow_vals)
        pow_pad = max(0.01, (pow_hi - pow_lo) * 0.15)
        ax_pow.set_ylim(max(0.0, pow_lo - pow_pad), min(1.02, pow_hi + pow_pad))
    ax_pow.set_xticks(ks_present)

    # One shared legend for both panels (FWER's nominal-alpha line plus every
    # method, which both panels plot identically) instead of a separate
    # legend per panel, placed outside the axes to the right.
    handles, labels = ax_fwer.get_legend_handles_labels()
    ax_pow.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=7)

    fig.suptitle(
        "Family-Wise Error Rate and Best-Arm Selection Power vs. Number of Systems Compared\n"
        f"Nominal alpha = {alpha}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_multiarm_fwer_vs_n_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """FWER and best-arm power as a function of n (sample size), one curve
    per correction method, collapsed across eval types and k -- the
    sample-size analogue of save_multiarm_fwer_vs_k_plot (same two-panel
    line-plot style: FWER y-axis zoomed to the actual spread rather than a
    fixed [0, ...]). X-axis is log-scaled, unlike the vs-k plot's linear
    one: n sweeps span an order of magnitude or more, so a linear axis
    crams the small-n tick labels into an unreadable overlapping cluster
    (see save_simultaneous_ci_coverage_width_vs_n_plot's identical fix).
    Only plots MULTIARM_PLOT_METHODS (every registered correction except
    `none` -- see that list's comment for why; `none` is still in the
    printed/logged report tables and the CSV).
    Only produced when more than one n value was swept; returns out_path
    unchanged (without writing) if all results share the same n."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    if len(sizes_present) < 2:
        return out_path

    fig, (ax_fwer, ax_pow) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    ax_fwer.axhline(alpha, color="black", linewidth=1.0, linestyle="--", label=f"α={alpha}")
    ax_fwer.axhspan(max(0.0, alpha - 0.02), alpha + 0.02, color="#DDDDDD", alpha=0.4, zorder=0)

    all_fwer_vals: list[float] = [alpha]
    all_pow_vals: list[float] = []
    for m in MULTIARM_PLOT_METHODS:
        c_rows = [r for r in results if r.correction == m.name]
        if not c_rows:
            continue
        xs, ys_fwer, ys_pow = [], [], []
        for n in sizes_present:
            n_rows = [r for r in c_rows if r.n == n]
            null_rows = [r for r in n_rows if r.condition == "null"]
            alt_rows = [r for r in n_rows if r.condition == "alt"]
            fwer_t = sum(r.n_reps for r in null_rows)
            fwer_c = sum(r.any_reject for r in null_rows)
            power_t = sum(r.n_reps for r in alt_rows)
            power_c = sum(r.best_selected for r in alt_rows)
            if fwer_t == 0 or power_t == 0:
                continue
            xs.append(n)
            ys_fwer.append(fwer_c / fwer_t)
            ys_pow.append(power_c / power_t)
        if xs:
            ax_fwer.plot(xs, ys_fwer, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            ax_pow.plot(xs, ys_pow, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            all_fwer_vals.extend(ys_fwer)
            all_pow_vals.extend(ys_pow)

    ax_fwer.set_xlabel("n (sample size)")
    ax_fwer.set_ylabel("FWER (null)")
    ax_fwer.set_title("FWER vs. sample size")
    fwer_lo, fwer_hi = min(all_fwer_vals), max(all_fwer_vals)
    fwer_pad = max(0.005, (fwer_hi - fwer_lo) * 0.15)
    ax_fwer.set_ylim(max(0.0, fwer_lo - fwer_pad), fwer_hi + fwer_pad)

    ax_pow.set_xlabel("n (sample size)")
    ax_pow.set_ylabel("Best-arm selection power (alt)")
    ax_pow.set_title("Power vs. sample size")
    # Zoom to the actual power spread -- see save_multiarm_fwer_vs_k_plot's
    # identical fix (no artificial 0.0 floor seed).
    if all_pow_vals:
        pow_lo, pow_hi = min(all_pow_vals), max(all_pow_vals)
        pow_pad = max(0.01, (pow_hi - pow_lo) * 0.15)
        ax_pow.set_ylim(max(0.0, pow_lo - pow_pad), min(1.02, pow_hi + pow_pad))

    # One shared legend for both panels, placed outside the axes to the
    # right -- see save_multiarm_fwer_vs_k_plot's identical fix.
    handles, labels = ax_fwer.get_legend_handles_labels()
    ax_pow.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=7)

    # Log-scale x-axis (see docstring) with exact tick labels at the swept
    # sizes instead of matplotlib's default log-scale power-of-ten ticks.
    for ax in (ax_fwer, ax_pow):
        ax.set_xscale("log")
        ax.set_xticks(sizes_present)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    fig.suptitle(
        "Family-Wise Error Rate and Best-Arm Selection Power vs. Sample Size\n"
        f"Nominal alpha = {alpha}",
        fontsize=12,
    )
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
    import matplotlib.patches as mpatches
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

    # x-tick labels already name each correction, but a color-key legend
    # (matching the palette used across every other multiarm plot) makes it
    # easy to cross-reference colors against those plots without having to
    # read the rotated tick labels here. Built manually via mpatches (rather
    # than pulled from the violin/strip plots, which are legend=False --
    # seaborn's own hue legend duplicates each color once per subplot,
    # which is redundant here since every subplot shares the same palette).
    legend_handles = [mpatches.Patch(facecolor=palette[c], alpha=0.5, label=c) for c in corrections]
    axes[0][-1].legend(
        handles=legend_handles, title="Correction", fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )

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
# Simultaneous-CI mode (non-PPI): calibration check for simultaneous
# (family-wise) confidence intervals, built two different ways:
#
# 1. `none`/`bonferroni`/`max_t` (SIMULTANEOUS_CI_METHODS) -- the three of
#    multiarm's six p-value correction strategies that have an established
#    simultaneous-CI dual: `none` (naive per-pair CI, no adjustment -- the
#    uncorrected baseline), Bonferroni t-intervals, and max-T (studentized
#    bootstrap, Romano-Wolf) -- the two non-naive constructions
#    all_pairwise's own router (_simultaneous_cis_router) picks between
#    automatically based on whether `method` is bootstrap-compatible.
#    (holm/fdr_bh/friedman_nemenyi have no CI dual -- holm/fdr_bh are
#    p-value-only adjustments, friedman_nemenyi is on the rank scale -- so
#    they're multiarm-only.) `none`/`bonferroni` are built on the scenario's
#    eval-type-canonical CI method (see point 2); `max_t` is the exception,
#    kept on --multiarm-method (bootstrap_t by default) since it needs a
#    bootstrap-compatible method to resample from, and neither Tango nor
#    Logit-t is one.
#
# 2. `sidak`/`boot` (CANONICAL_SIMULTANEOUS_CI_METHODS) -- does adjusting
#    evalstats' actual production-default pairwise CI formula for
#    multiplicity (rather than max-T/Bonferroni's generic bootstrap_t-based
#    constructions) do better? _canonical_ci_func below maps
#    each eval type to its evalstats.config.AUTO_ANALYZE_METHOD_TABLE
#    default: Tango for binary (N>=50 row; small-N bayes_binary isn't
#    alpha-parameterized the same way, so isn't modeled here), Logit-t for
#    continuous/likert (both count as the "bounded_01" data_kind once this
#    harness's own EVAL_TYPE_SCALE_BOUNDS supplies the range Logit-t needs).
#    `grades` has no entry (out of scope -- not swept by default anyway; see
#    official_args()'s eval_types). `sidak` (closed-form Sidak-adjusted
#    per-comparison alpha) and `boot` (a joint bootstrap critical value
#    substituted for the canonical CI's marginal normal quantile, which
#    accounts for correlation between comparisons the way max-T does for a
#    generic statistic) are the two ways of widening it to hold
#    family-wise. Sidak/bootstrap-scaling themselves are NOT method-specific
#    -- see evalstats.core.paired's _sidak_simultaneous_cis /
#    _joint_bootstrap_scaled_simultaneous_cis, which take any
#    alpha-parameterized per-pair CI as a `ci_func` argument; the canonical
#    formula is just the ci_func passed in below.
#
# Both reuse the SAME k-arm MultiArmSource scenarios (synthetic and real)
# that --mode multiarm sweeps, since all these questions ("which p-value
# correction", "which CI construction", "does the canonical CI benefit from
# multiplicity adjustment") share the identical underlying k-arm generative
# model -- just a different measurement per rep (coverage + width of the
# constructed simultaneous CI, instead of reject/best-arm).
# ---------------------------------------------------------------------------


def _canonical_ci_func(eval_type: str):
    """The alpha-parameterized ci_func for evalstats' canonical pairwise CI
    at this eval type, or ``None`` if there isn't one wired up here.

    Mirrors evalstats.config.AUTO_ANALYZE_METHOD_TABLE's N>=50 binary row
    (Tango) and "bounded_01" row (Logit-t, for continuous/likert -- both are
    on a known bounded numeric scale via EVAL_TYPE_SCALE_BOUNDS, which is
    exactly what makes a "bounded_01" data_kind determination valid).
    `logit_t_ci_1d` assumes its input is a [0, 1]-scaled MEAN, not a signed
    difference of two such means (which ranges over [-span, span], centred
    at 0, not [0, 1]) -- rescaled_ci handles that remapping, matching
    evalstats.core.paired's own "logit_t" pairwise-CI branch and
    cases/ci_paired.py's PAIRWISE_EXTRA_METHODS treatment of the same
    formula. Returns ``None`` for "grades" (and anything else) -- no
    canonical default is modeled for it here.
    """
    if eval_type == "binary":
        return tango_paired_ci_from_diffs
    if eval_type in ("continuous", "likert"):
        scale_lo, scale_hi = EVAL_TYPE_SCALE_BOUNDS[eval_type]
        diff_span = scale_hi - scale_lo
        diff_lo, diff_hi = -diff_span, diff_span

        def _logit_t_diff_ci(diffs: np.ndarray, alpha: float, _lo: float = diff_lo, _hi: float = diff_hi) -> tuple[float, float]:
            return rescaled_ci(logit_t_ci_1d, diffs, alpha, _lo, _hi)

        return _logit_t_diff_ci
    return None


@dataclass
class SimultaneousCIResult:
    eval_type: str
    label: str
    n: int
    k: int
    ci_method: str  # "none" | "bonferroni" | "max_t" | "sidak" | "boot" ("sidak"/"boot" absent when _canonical_ci_func(eval_type) is None, e.g. "grades")
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
    total_time: float = 0.0
    """Total wall-clock seconds for THIS method's own construction, summed
    across all n_reps of this condition -- e.g. bonferroni's own
    _bonferroni_simultaneous_cis() call, max_t's own _simultaneous_cis_router()
    call (which includes its bootstrap resampling), etc. Does NOT include
    shared per-rep setup (score generation, building matrix_raw) except for
    `none`, which that setup is attributed to (see _run_simultaneous_ci_cell)."""


def _run_simultaneous_ci_cell(
    source: MultiArmSource, n: int, runs: int, k_arms: int, n_reps: int, n_bootstrap: int,
    alpha: float, multiarm_method: str, statistic: str, seed, ci_methods: list[str] | None = None,
) -> list[SimultaneousCIResult]:
    labels = [f"arm_{i}" for i in range(k_arms)]
    pairs = [(labels[i], labels[j]) for i in range(k_arms) for j in range(i + 1, k_arms)]
    ci = 1.0 - alpha
    rng = np.random.default_rng(seed)

    # ci_func is the eval-type-canonical CI formula (Tango for binary,
    # Logit-t for continuous/likert; None for grades or anything else --
    # see _canonical_ci_func). When present, it replaces --multiarm-method
    # as the basis for `none` (and feeds `sidak`/`boot`, which don't exist
    # without a canonical formula to widen); `bonferroni` is unaffected
    # either way (_bonferroni_simultaneous_cis always builds its own
    # generic t-interval from per_input_diffs, never a per-method formula,
    # so it never depended on --multiarm-method to begin with); `max_t`
    # keeps using --multiarm-method (bootstrap_t by default) always, since
    # it needs a bootstrap-compatible method to resample from and neither
    # Tango nor Logit-t is one.
    ci_func = _canonical_ci_func(source.eval_type)
    has_canonical = ci_func is not None
    base_methods = [m.name for m in SIMULTANEOUS_CI_METHODS]
    canonical_methods = [m.name for m in CANONICAL_SIMULTANEOUS_CI_METHODS] if has_canonical else []
    all_methods = base_methods + canonical_methods
    if ci_methods is not None:
        requested = set(ci_methods)
        all_methods = [m for m in all_methods if m in requested]
    # `max_t`/`boot` each pay for their own independent bootstrap resample
    # (unlike --mode multiarm, they don't share one here -- see
    # _joint_bootstrap_critical_value vs _max_stat_simultaneous_cis), so
    # skipping either one when it's not requested is a real cost saving, not
    # just a bookkeeping nicety. `none`/`bonferroni`/`sidak` are all cheap
    # closed-form constructions regardless, but are gated the same way for
    # consistency -- a method absent from `all_methods` never appears in
    # the returned SimultaneousCIResult rows either way (see the loop over
    # `all_methods` below), so gating its computation here changes runtime,
    # not results.
    need = {m: (m in all_methods) for m in ("none", "bonferroni", "max_t", CORR_SIDAK.name, CORR_BOOT.name)}
    agg_covered: dict[tuple[str, str], int] = {(m, cond): 0 for m in all_methods for cond in ("null", "alt")}
    agg_width: dict[tuple[str, str], float] = {(m, cond): 0.0 for m in all_methods for cond in ("null", "alt")}
    agg_score: dict[tuple[str, str], float] = {(m, cond): 0.0 for m in all_methods for cond in ("null", "alt")}
    # Per-(method, condition), not a single per-condition total -- each
    # construction's own wall-clock cost, so e.g. `boot`'s extra joint
    # bootstrap resampling actually shows up as slower than `sidak`'s
    # closed-form widening in the report's Time(ms) column, instead of every
    # method row displaying the same aggregate "whole rep" time.
    agg_time: dict[tuple[str, str], float] = {(m, cond): 0.0 for m in all_methods for cond in ("null", "alt")}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_reps):
            for condition, delta in (("null", 0.0), ("alt", source.alt_delta)):
                # Score generation + matrix_raw + `none` share one timer,
                # attributed to `none` -- matrix_raw is the shared
                # prerequisite `none` (and, when there's no canonical
                # ci_func, only `none`) actually needs to exist; it's not
                # fairly attributable to any other single method.
                _t_none0 = time.perf_counter()
                scores = source.generate_scores(rng, n, runs, k_arms, delta)
                true_means = source.true_means(k_arms, delta)

                # Raw per-pair results, for per_input_diffs (method-invariant:
                # always scores.mean(axis=2)[a] - scores.mean(axis=2)[b],
                # identical bit-for-bit regardless of which method computes
                # it -- verified against bootstrap_t's own seeded/non-seeded
                # construction) feeding Bonferroni's t-interval formula and,
                # when there's no canonical ci_func for this eval type,
                # `none` too. When a canonical ci_func DOES exist, built with
                # the cheapest closed-form method (t_interval -- no bootstrap
                # resampling at all) rather than --multiarm-method (bootstrap_t
                # by default): none/bonferroni/sidak/boot below never read
                # this matrix's own .ci_low/.ci_high in that case (only
                # per_input_diffs), and max_t's resampling reads `scores`
                # directly (see below), so paying for --multiarm-method's
                # expensive k(k-1)/2 independent nested double bootstrap here
                # would be pure waste. Falls back to --multiarm-method when
                # there's no canonical ci_func (e.g. "grades"), since `none`
                # genuinely needs that method's own CI there. Always built
                # (whenever anything at all is requested) since it's cheap
                # (t_interval, no bootstrap) and max_t's router below wants
                # it as a fallback safety net regardless of whether `none`
                # itself was requested.
                none_cis: dict = {}
                if any(need.values()):
                    matrix_raw = all_pairwise(
                        scores=scores, labels=labels, method=("t_interval" if has_canonical else multiarm_method), ci=ci,
                        n_bootstrap=n_bootstrap, correction="none", rng=rng, statistic=statistic,
                        simultaneous_ci=False,
                    )
                    if need["none"]:
                        if has_canonical:
                            # The canonical formula's own naive CI at the plain
                            # (unadjusted) alpha IS the "none" construction here --
                            # mathematically identical to what all_pairwise(method=
                            # "tango"/"logit_t") would compute for .ci_low/.ci_high,
                            # but derived directly from ci_func so continuous/likert
                            # get this harness's own EVAL_TYPE_SCALE_BOUNDS-derived
                            # diff span rather than all_pairwise's [0, 1]-diff-span
                            # default (which would silently mis-scale likert's
                            # [1, 5] range without an explicit score_range).
                            none_cis = {
                                pair: ci_func(matrix_raw.results[pair].per_input_diffs, alpha)
                                for pair in pairs
                            }
                        else:
                            none_cis = {pair: (matrix_raw.get(*pair).ci_low, matrix_raw.get(*pair).ci_high) for pair in pairs}
                if need["none"]:
                    agg_time[("none", condition)] += time.perf_counter() - _t_none0

                bonf_cis: dict = {}
                if need["bonferroni"]:
                    _t0 = time.perf_counter()
                    bonf_cis = _bonferroni_simultaneous_cis(results=matrix_raw.results, pairs=pairs, ci=ci)
                    agg_time[("bonferroni", condition)] += time.perf_counter() - _t0

                # max-T: call _simultaneous_cis_router directly (the same
                # function all_pairwise(simultaneous_ci=True) would call
                # internally). Its actual max-T computation
                # (_max_stat_simultaneous_cis) resamples straight from
                # `scores` under --multiarm-method and never reads `results`
                # at all -- `results` is only consulted for the router's
                # rare Bonferroni-fallback safety net on degenerate data,
                # which matrix_raw.results above already supplies (cheaply,
                # when has_canonical). This is --multiarm-method's (bootstrap_t
                # by default) one remaining unavoidable cost in this cell:
                # a single shared max-T resample, not the k(k-1)/2
                # independent nested double bootstraps matrix_raw used to
                # pay for before this cheap-when-canonical rework. Skipped
                # entirely (its own independent bootstrap resample, the
                # single most expensive part of this cell alongside boot's)
                # when max_t isn't requested via --ci-methods.
                maxt_cis: dict = {}
                if need["max_t"]:
                    _t0 = time.perf_counter()
                    sim_cis, sim_method, _ = _simultaneous_cis_router(
                        scores=scores, results=matrix_raw.results, pairs=pairs, labels=labels,
                        method=multiarm_method, ci=ci, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
                        prefer="max_t",
                    )
                    if sim_method == "max_t":
                        maxt_cis = sim_cis
                    agg_time[("max_t", condition)] += time.perf_counter() - _t0

                # sidak/boot: widen the canonical formula (ci_func) itself
                # for multiplicity, instead of Bonferroni/max-T's generic
                # bootstrap_t-based constructions. The widening machinery
                # (_sidak_simultaneous_cis / _joint_bootstrap_scaled_
                # simultaneous_cis) has no idea what "Tango" or "Logit-t"
                # is -- it's generic over any alpha-parameterized ci_func;
                # ci_func is just whichever canonical formula this
                # scenario's eval type resolves to. per_input_diffs is
                # method-agnostic, so matrix_raw.results -- built above
                # under multiarm_method -- is reused here rather than
                # re-running all_pairwise a second time.
                sidak_cis: dict = {}
                if has_canonical and need[CORR_SIDAK.name]:
                    _t0 = time.perf_counter()
                    sidak_cis = _sidak_simultaneous_cis(
                        results=matrix_raw.results, pairs=pairs, ci=ci, ci_func=ci_func,
                    )
                    agg_time[(CORR_SIDAK.name, condition)] += time.perf_counter() - _t0

                # boot: its own independent joint bootstrap resample (see
                # _joint_bootstrap_critical_value) -- unlike --mode multiarm,
                # this is NOT shared with max_t here, so skipping it when not
                # requested (and max_t IS requested, or vice versa) is a real
                # saving, not just symmetry with the gating above.
                boot_cis: dict = {}
                if has_canonical and need[CORR_BOOT.name]:
                    _t0 = time.perf_counter()
                    boot_cis = _joint_bootstrap_scaled_simultaneous_cis(
                        scores=scores, results=matrix_raw.results, pairs=pairs, labels=labels,
                        ci=ci, n_bootstrap=n_bootstrap, rng=rng, ci_func=ci_func, statistic=statistic,
                    )
                    agg_time[(CORR_BOOT.name, condition)] += time.perf_counter() - _t0

                for method_name, cis in (
                    ("none", none_cis), ("bonferroni", bonf_cis), ("max_t", maxt_cis),
                    (CORR_SIDAK.name, sidak_cis), (CORR_BOOT.name, boot_cis),
                ):
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
            total_time=agg_time[(method_name, condition)],
        )
        for method_name in all_methods
        for condition in ("null", "alt")
    ]


_SIMULTANEOUS_CI_SOURCES: list = []  # fork-inherited worker state for run_simultaneous_ci_simulation


def _run_simultaneous_ci_cell_worker(args: tuple) -> list[SimultaneousCIResult]:
    sc_idx, n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed, ci_methods = args
    return _run_simultaneous_ci_cell(
        _SIMULTANEOUS_CI_SOURCES[sc_idx], n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed,
        ci_methods=ci_methods,
    )


def run_simultaneous_ci_simulation(
    sources: list[MultiArmSource], sample_sizes: list[int], runs: int, k_values: list[int], n_reps: int,
    n_bootstrap: int, alpha: float, multiarm_method: str, statistic: str, progress_mode: str = "bar",
    seed: int = 42, n_workers: int = 1, ci_methods: list[str] | None = None,
) -> list[SimultaneousCIResult]:
    global _SIMULTANEOUS_CI_SOURCES
    _SIMULTANEOUS_CI_SOURCES = list(sources)
    ss = np.random.SeedSequence(seed)
    cells = _multiarm_style_cells(sources, sample_sizes, k_values)

    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [(sc_idx, n, runs, k, n_reps, n_bootstrap, alpha, multiarm_method, statistic, seed, ci_methods)
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
    print(f"\n{'='*78}\n  PVALUES (SIMULTANEOUS CI) -- none vs. BONFERRONI vs. max-T vs. Tango variants\n"
          f"  Nominal family-wise coverage: {target:.0%}\n{'='*78}")
    ci_methods = [m.name for m in ALL_SIMULTANEOUS_CI_METHODS if m.name in {r.ci_method for r in results}]
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


def latex_simultaneous_ci_overall_summary(
    results: list[SimultaneousCIResult], alpha: float, *,
    label_suffix: str = "", caption_suffix: str = "",
) -> str:
    """LaTeX booktabs overall summary: per-CI-method family-wise coverage
    (null, with its 95% MC band) + average width (null and alt), collapsed
    across eval types, plus one coverage column per sample size actually
    swept. `none` should visibly under-cover (no simultaneous adjustment at
    all, even though it's already built on evalstats' canonical per-eval-type
    CI -- see _canonical_ci_func); `bonferroni`/`max_t`/`sidak`/`boot` should
    all hit nominal coverage, so the tie-breaker between them is which gets
    there with a narrower average CI/interval score. *results* may be a
    filtered subset (e.g. n<=30 / n>=30 -- see latex_simultaneous_ci_full_report)
    with *label_suffix*/*caption_suffix* set so multiple calls in one
    document don't collide on \\label{}."""
    target = 1.0 - alpha
    ci_methods = [m.name for m in ALL_SIMULTANEOUS_CI_METHODS if m.name in {r.ci_method for r in results}]
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
                f"and average per-comparison interval score -- none/bonferroni/max\\_t (generic, "
                f"\\texttt{{--multiarm-method}}-based, bootstrap\\_t by default) vs. sidak/boot "
                f"(Sidak- and joint-bootstrap-scaled widenings of evalstats' canonical per-eval-type "
                f"CI: Tango for binary, Logit-t for continuous/likert){caption_suffix} "
                f"(nominal coverage={target:.0%}).",
        label=f"tab:pvalues_simultaneous_ci_overall{label_suffix}",
        columns=["CI method", "Cov(null)", "95\\% MC band", "Width(null)", "Score(null)",
                 "Cov(alt)", "Width(alt)", "Score(alt)", "Eval types"]
                + [f"n={n}" for n in sizes_present],
        rows=rows,
    )


def latex_simultaneous_ci_by_eval_type_summary(results: list[SimultaneousCIResult], alpha: float) -> str:
    """LaTeX booktabs summary faceted by eval type instead of collapsed
    across them: one row per (eval type, CI method), collapsed across n and
    k. Complements latex_simultaneous_ci_overall_summary -- `sidak`/`boot`
    widen a DIFFERENT canonical CI per eval type (Tango for binary, Logit-t
    for continuous/likert; see _canonical_ci_func), so this is the table
    that shows the effect holds for each formula separately rather than
    only in a pooled average that could be dominated by whichever eval type
    has the most scenarios/sizes swept."""
    target = 1.0 - alpha
    ci_methods = [m.name for m in ALL_SIMULTANEOUS_CI_METHODS if m.name in {r.ci_method for r in results}]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]

    rows = []
    for et in eval_types_present:
        et_results = [r for r in results if r.eval_type == et]
        et_methods = [cm for cm in ci_methods if any(r.ci_method == cm for r in et_results)]
        for cm in et_methods:
            c_rows = [r for r in et_results if r.ci_method == cm]
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
            rows.append([
                escape_latex(et), escape_latex(cm),
                f"{cov_null:.3f}" if np.isfinite(cov_null) else "-",
                f"${lo:.3f}\\text{{--}}{hi:.3f}$" if np.isfinite(lo) else "-",
                f"{w_null:.4f}" if np.isfinite(w_null) else "-",
                f"{s_null:.4f}" if np.isfinite(s_null) else "-",
                f"{cov_alt:.3f}" if np.isfinite(cov_alt) else "-",
                f"{w_alt:.4f}" if np.isfinite(w_alt) else "-",
                f"{s_alt:.4f}" if np.isfinite(s_alt) else "-",
            ])

    return booktabs_table(
        caption=f"pvalues (simultaneous CI): family-wise coverage, average per-comparison width, "
                f"and average per-comparison interval score, faceted by eval type "
                f"(nominal coverage={target:.0%}).",
        label="tab:pvalues_simultaneous_ci_by_eval_type",
        columns=["Eval type", "CI method", "Cov(null)", "95\\% MC band", "Width(null)", "Score(null)",
                 "Cov(alt)", "Width(alt)", "Score(alt)"],
        rows=rows,
    )


def latex_simultaneous_ci_full_report(results: list[SimultaneousCIResult], alpha: float) -> str:
    """All simultaneous_ci LaTeX tables for this run, concatenated and ready
    to paste into a paper: the pooled overall summary, the same summary
    split into low-N (n<=30) / high-N (n>=30) subsets (the mode's headline
    max-T-crossover finding -- see print_simultaneous_ci_report's LOW N /
    HIGH N split), and the by-eval-type facet (showing sidak/boot's effect
    holds separately for Tango (binary) and Logit-t (continuous/likert), not
    just in a pooled average)."""
    return "\n\n".join([
        latex_simultaneous_ci_overall_summary(results, alpha),
        latex_simultaneous_ci_overall_summary(
            [r for r in results if r.n <= 30], alpha,
            label_suffix="_lown", caption_suffix=", low-N ($n \\le 30$) subset",
        ),
        latex_simultaneous_ci_overall_summary(
            [r for r in results if r.n >= 30], alpha,
            label_suffix="_highn", caption_suffix=", high-N ($n \\ge 30$) subset",
        ),
        latex_simultaneous_ci_by_eval_type_summary(results, alpha),
    ])


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
        summary_text += "\n% --- LaTeX tables (--latex): overall, low-N, high-N, by-eval-type ---\n" + latex_simultaneous_ci_full_report(results, alpha=alpha)
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
        "Simultaneous Confidence Interval Calibration: Coverage vs. Width\n"
        f"One point per scenario, averaged across $n$ and $k$ (nominal coverage = {target:.0%})",
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

    all_cov_vals: list[float] = [target]
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
            all_cov_vals.extend(ys_cov)

    ax_cov.set_xlabel("k (number of arms)")
    ax_cov.set_ylabel("Family-wise coverage (null)")
    ax_cov.set_title("Coverage vs. number of arms")
    # Zoom to the actual coverage spread (plus the nominal line) rather than
    # a fixed [0, 1] -- with `none` dropped from this plot (SIMULTANEOUS_CI_
    # PLOT_METHODS), every remaining curve usually clusters near nominal, and
    # a full [0, 1] axis squashes that spread into an unreadable sliver at
    # the top (see save_simultaneous_ci_coverage_width_plot's identical fix).
    cov_lo, cov_hi = min(all_cov_vals), max(all_cov_vals)
    cov_pad = max(0.01, (cov_hi - cov_lo) * 0.15)
    ax_cov.set_ylim(max(0.0, cov_lo - cov_pad), min(1.02, cov_hi + cov_pad))
    ax_cov.set_xticks(ks_present)

    ax_width.set_xlabel("k (number of arms)")
    ax_width.set_ylabel("Average per-comparison CI width (null)")
    ax_width.set_title("Width vs. number of arms")
    ax_width.set_ylim(bottom=0.0)
    ax_width.set_xticks(ks_present)

    # One shared legend for both panels (coverage's nominal line plus every
    # method, which both panels plot identically) instead of a separate
    # legend per panel, placed outside the axes to the right.
    handles, labels = ax_cov.get_legend_handles_labels()
    ax_width.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=7)

    fig.suptitle(
        "Simultaneous Confidence Interval Calibration vs. Number of Systems Compared\n"
        f"Nominal coverage = {target:.0%}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_simultaneous_ci_coverage_width_vs_n_plot(*, results: list[SimultaneousCIResult], alpha: float, out_path: str) -> str:
    """Family-wise coverage and average width as a function of n (sample
    size), one curve per CI method, collapsed across eval types and k --
    the sample-size analogue of save_simultaneous_ci_coverage_width_vs_k_plot
    (same two-panel line-plot style: exact x-ticks pinned to the sizes
    actually swept, coverage y-axis zoomed to the actual spread rather than
    a fixed [0, 1]). X-axis is log-scaled, unlike the vs-k plot's linear one:
    n sweeps span an order of magnitude or more (e.g. the official preset's
    15..500), so a linear axis crams the small-n tick labels into an
    unreadable overlapping cluster -- log-scale is the standard convention
    for coverage-vs-sample-size plots for exactly this reason, and still
    shows exact tick labels (via a ScalarFormatter override) rather than
    scientific notation. Complements save_simultaneous_ci_violin_vs_n_plot
    (full per-cell distribution, faceted by eval type) with the single
    pooled-mean curve per method that's easier to read at a glance across
    the whole n sweep. Only plots Bonferroni/max-T/sidak/boot
    (SIMULTANEOUS_CI_PLOT_METHODS excludes `none` -- see that plot's
    docstring for why; `none` is still in the printed/logged report tables
    and the CSV). Only produced when more than one n value was swept;
    returns out_path unchanged (without writing) if all results share the
    same n."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    target = 1.0 - alpha
    sizes_present = sorted({r.n for r in results if r.condition == "null"})
    if len(sizes_present) < 2:
        return out_path

    fig, (ax_cov, ax_width) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    ax_cov.axhline(target, color="black", linewidth=1.0, linestyle="--", label=f"nominal={target:.0%}")

    all_cov_vals: list[float] = [target]
    for m in SIMULTANEOUS_CI_PLOT_METHODS:
        c_rows = [r for r in results if r.ci_method == m.name]
        if not c_rows:
            continue
        xs, ys_cov, ys_width = [], [], []
        for n in sizes_present:
            n_rows = [r for r in c_rows if r.n == n]
            null_rows = [r for r in n_rows if r.condition == "null"]
            t_null = sum(r.n_reps for r in null_rows)
            c_null = sum(r.all_covered for r in null_rows)
            w_null = sum(r.total_width for r in null_rows)
            if t_null == 0:
                continue
            xs.append(n)
            ys_cov.append(c_null / t_null)
            ys_width.append(w_null / t_null)
        if xs:
            ax_cov.plot(xs, ys_cov, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            ax_width.plot(xs, ys_width, marker="o", color=m.color, markersize=5, linewidth=1.4, label=m.name, alpha=0.85)
            all_cov_vals.extend(ys_cov)

    ax_cov.set_xlabel("n (sample size)")
    ax_cov.set_ylabel("Family-wise coverage (null)")
    ax_cov.set_title("Coverage vs. sample size")
    # Zoom to the actual coverage spread (plus the nominal line) rather than
    # a fixed [0, 1] -- see save_simultaneous_ci_coverage_width_vs_k_plot's
    # identical fix.
    cov_lo, cov_hi = min(all_cov_vals), max(all_cov_vals)
    cov_pad = max(0.01, (cov_hi - cov_lo) * 0.15)
    ax_cov.set_ylim(max(0.0, cov_lo - cov_pad), min(1.02, cov_hi + cov_pad))

    ax_width.set_xlabel("n (sample size)")
    ax_width.set_ylabel("Average per-comparison CI width (null)")
    ax_width.set_title("Width vs. sample size")
    ax_width.set_ylim(bottom=0.0)

    # One shared legend for both panels, placed outside the axes to the
    # right -- see save_simultaneous_ci_coverage_width_vs_k_plot's identical
    # fix.
    handles, labels = ax_cov.get_legend_handles_labels()
    ax_width.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=7)

    # Log-scale x-axis (see docstring) with exact tick labels at the swept
    # sizes instead of matplotlib's default log-scale power-of-ten ticks.
    for ax in (ax_cov, ax_width):
        ax.set_xscale("log")
        ax.set_xticks(sizes_present)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    fig.suptitle(
        "Simultaneous Confidence Interval Calibration vs. Sample Size\n"
        f"Nominal coverage = {target:.0%}",
        fontsize=12,
    )
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
    plots bonferroni/max_t/sidak/boot (`none` is dropped -- see
    SIMULTANEOUS_CI_PLOT_METHODS -- since it's so far below nominal
    coverage that it squashes the comparison this plot exists to show;
    it's still in the printed/logged report tables and the CSV)."""
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

    # x-tick labels already name each method, but a color-key legend (see
    # save_multiarm_reliability_violin_plot's identical fix) makes it easy
    # to cross-reference colors against the other simultaneous_ci plots.
    legend_handles = [mpatches.Patch(facecolor=palette[m], alpha=0.5, label=m) for m in ci_methods]
    axes[0][-1].legend(
        handles=legend_handles, title="Simult. CI method", fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )

    fig.suptitle(
        "Simultaneous Confidence Interval Reliability Across Evaluation Scenarios\n"
        f"One point per scenario, nominal coverage = {target:.0%}",
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

    Only plots bonferroni/max_t/sidak/boot (`none` is dropped -- see
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
        handles=legend_handles, title="Simult. CI method", fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )

    fig.suptitle(
        "Simultaneous Confidence Interval Coverage and Interval Score vs. Sample Size\n"
        f"Nominal coverage = {target:.0%}; each violin pools all $k$ and scenarios at that $n$",
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

            if MW_NAIVE.name in active_tests:
                try:
                    p_u = float(scipy_stats.mannwhitneyu(cell.llm_a2, cell.llm_b2, alternative="two-sided").pvalue)
                    uncorrected[MW_NAIVE.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda xa, ya: _p_x_gt_y_midrank(xa, ya) - 0.5, _ALPHA, n_boot, _rng_seed())
                    corrected[MW_NAIVE.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[MW_NAIVE.name] += 1

            if MWU_CORR.name in active_tests:
                try:
                    p_u = float(scipy_stats.mannwhitneyu(cell.llm_a2, cell.llm_b2, alternative="two-sided").pvalue)
                    uncorrected[MWU_CORR.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample_midrank_corrected(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, _ALPHA, n_boot, _rng_seed())
                    corrected[MWU_CORR.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[MWU_CORR.name] += 1

            if WILCOXON.name in active_tests:
                try:
                    # Deliberately left at scipy's default method="auto" --
                    # see _safe_wilcoxon_p's docstring: it's slower for
                    # small tied/discrete samples but computes a genuinely
                    # different (rigorously tie-corrected exact), not just
                    # slower, p-value than forcing method="exact" would.
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

            if KRUSKAL_NAIVE.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    p_u = _uncorrected_kruskal_p_value(groups_kw)
                    uncorrected[KRUSKAL_NAIVE.name] += int(p_u < _ALPHA)
                    pw = _ppi_kruskal_wallis_pairwise(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    corrected[KRUSKAL_NAIVE.name] += int(pw["wald_p"] < _ALPHA)
                except Exception:
                    failed[KRUSKAL_NAIVE.name] += 1

            if KRUSKAL_CORR.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    p_u = _uncorrected_kruskal_p_value(groups_kw)
                    uncorrected[KRUSKAL_CORR.name] += int(p_u < _ALPHA)
                    pw = _ppi_kruskal_wallis_pairwise_corrected(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    corrected[KRUSKAL_CORR.name] += int(pw["wald_p"] < _ALPHA)
                except Exception:
                    failed[KRUSKAL_CORR.name] += 1

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
    TTEST.name, TTEST_WELCH.name, MW_NAIVE.name, MWU_CORR.name, WILCOXON.name, PAIRED_T.name, BAYES_BOOTSTRAP.name,
    BOOTSTRAP_T.name, TANGO.name, ANOVA_IND.name, ANOVA_REP.name, FRIEDMAN.name, KRUSKAL_NAIVE.name, KRUSKAL_CORR.name,
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
    completely unchanged, at the cost of redrawing ttest/ttest_welch/mwu_corr/
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

            if MW_NAIVE.name in active_tests:
                try:
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda xa, ya: _p_x_gt_y_midrank(xa, ya) - 0.5, _ALPHA, n_boot, _rng_seed())
                    out[MW_NAIVE.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if MWU_CORR.name in active_tests:
                try:
                    r = _ppi_two_sample_midrank_corrected(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, _ALPHA, n_boot, _rng_seed())
                    out[MWU_CORR.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
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

            if KRUSKAL_NAIVE.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    pw = _ppi_kruskal_wallis_pairwise(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    llm_theta = _kw_pairwise_thetas(groups_kw, pw["pairs"])
                    out[KRUSKAL_NAIVE.name].append((
                        float(np.mean(pw["theta_hat"])), float(np.mean(pw["ci_lo"])),
                        float(np.mean(pw["ci_hi"])), float(np.mean(llm_theta)),
                    ))
                except Exception:
                    pass

            if KRUSKAL_CORR.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    pw = _ppi_kruskal_wallis_pairwise_corrected(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    llm_theta = _kw_pairwise_thetas(groups_kw, pw["pairs"])
                    out[KRUSKAL_CORR.name].append((
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


# ---------------------------------------------------------------------------
# PPI mode, estimator comparison: for ONE representative paired-mean estimand
# (paired_t -- generalizes to binary too since a proportion is just the mean
# of a 0/1 variable, and it's already documented elsewhere in this file as
# the "reasonable default" for that estimand), five ways of turning (sparse
# human labels + biased LLM-judge scores) into a hypothesis test, compared
# head to head on the SAME draws:
#   all_human      -- oracle: classical paired t-test on the FULL, dense
#                      ground truth (as if every item had a human label).
#   human_subset   -- classical paired t-test on ONLY the labeled subset's
#                      ground truth (small n_lab) -- the "why not just
#                      collect more human labels instead of trusting a
#                      correction" baseline a skeptical reviewer will ask
#                      about.
#   llm_only       -- classical paired t-test on the full LLM-judge scores,
#                      uncorrected (same number as run_ppi_cell's
#                      "uncorrected" arm, recomputed here for encapsulation).
#   llm_impute     -- classical paired t-test on the LLM-judge scores with
#                      labeled positions' values OVERWRITTEN by the true
#                      human label (a naive missing-data-imputation
#                      baseline with NO PPI rectifier) -- shows that simply
#                      "filling in what you know" is not the same as
#                      properly correcting for what you don't.
#   ppi            -- PPI-corrected (evalstats.tests._ppi_paired_arrays).
# Only paired_t's structure (cell.llm_x/llm_y/lab_x/lab_y/truth_x/truth_y) is
# used -- extending this same comparison to every PPI_TEST_METHODS estimand
# would multiply the plot count for no real gain in what it demonstrates;
# one clear, representative estimand is the point of this check.
# ---------------------------------------------------------------------------


@dataclass
class PPIComparisonResult:
    name: str
    tag: str  # "power" (vs. effect_size, reusing build_ppi_power_sources) | "compare_label_frac" (vs. label_frac)
    eval_type: str
    n: int
    n_reps: int
    effect_size: float
    """The eval-type-RELATIVE effect-size fraction (see
    build_ppi_power_sources/_jb_effect_magnitude), not JudgeBiasSource's raw
    absolute effect_size field -- the raw value is scaled by each eval
    type's own EVAL_TYPE_SCALE_BOUNDS span, so it isn't comparable across
    eval types (continuous vs. likert would show different x-axis values
    for "the same" relative effect). This field IS comparable across eval
    types; see _ppi_source_effect_frac."""
    label_frac: float
    n_lab: int
    """REALIZED labeled-item count (measured off the actual mask each
    replicate produces, not the nominal `n * label_frac`) -- see
    _JB_MIN_LAB: label_frac alone can be misleading once the floor binds
    (e.g. label_frac=0.05 and 0.10 both floor to n_lab=15 at n=100), so this
    is the field to plot/group by, not label_frac, whenever comparing
    across different n. For "independent"-mask structures (group/group3:
    ttest_welch, mwu_corr, anova_ind, kruskal) this is the FIRST group's
    labeled count specifically -- see _run_ppi_comparison_cell's docstring
    for why every group is expected to match under this harness's scenario
    construction."""
    method: str = PAIRED_T.name
    """Which classical test this result is for -- see _COMPARISON_METHODS.
    Defaults to paired_t for backward compatibility; every
    _run_ppi_comparison_cell call now sets this explicitly to one of
    ttest_welch/paired_t/mwu_corr/wilcoxon (never a pooled "average" tag --
    pooling across methods happens downstream, over a list of these, via
    pool_ppi_comparison_across_methods)."""
    rejects_all_human: int = 0
    rejects_human_subset: int = 0
    rejects_llm_only: int = 0
    rejects_llm_impute: int = 0
    rejects_ppi: int = 0
    n_failed: int = 0


def _ppi_source_effect_frac(sc: JudgeBiasSource) -> float:
    """Eval-type-relative effect-size fraction for a comparison-sweep
    source -- see PPIComparisonResult.effect_size's docstring. Every tag
    besides "power" must be listed explicitly here (not defaulted to
    PPI_COMPARISON_MODERATE_EFFECT_FRAC): that fallback happens to be
    correct for "compare_label_frac"/"nlab_grid_power" (both built at
    exactly that fraction) but was silently WRONG for "nlab_grid"
    (build_ppi_nlab_grid_sources' effect_frac=0.0 calibration grid) before
    this was caught -- every nlab_grid PPIComparisonResult reported
    effect_size=0.20 instead of 0.0 in its CSV/log output, even though the
    underlying simulation itself used effect_size=0.0 correctly (this field
    is metadata only; JudgeBiasSource.effect_size, not this function, is
    what generate_judge_bias_cell actually reads)."""
    if sc.tag == "power":
        return _parse_ppi_power_name(sc.name)[1]
    if sc.tag == "nlab_grid":
        return 0.0
    if sc.tag in ("compare_label_frac", "nlab_grid_power"):
        return PPI_COMPARISON_MODERATE_EFFECT_FRAC
    if sc.tag == "factorial":
        m = re.search(r"\.es=([a-z]+)\.", sc.name)
        if not m:
            raise ValueError(f"_ppi_source_effect_frac: could not parse es label from {sc.name!r}")
        return PPI_FACTORIAL_EFFECT_FRACS[m.group(1)]
    raise ValueError(f"_ppi_source_effect_frac: unrecognized tag {sc.tag!r}")


_COMPARISON_METHODS = (TTEST_WELCH.name, PAIRED_T.name, MWU_CORR.name, WILCOXON.name)
"""The four classical two-sample/paired tests the PPI estimator-comparison
sweep (and everything downstream: N x N_lab grid, full factorial, the
null-effect bar chart) runs and, by default, averages across -- rather than
paired_t alone. ttest_welch/paired_t (mean-based) and mwu_corr/wilcoxon (rank-
based) cover both the independent-two-group and paired structures, and all
four test the SAME two-group mean/location-shift question via different
classical machinery, so averaging their rejection rates is a coherent
summary of "does this hold across reasonable test choices." Uses mwu_corr
(the per-group locally-calibrated PPI midrank correction), not mw_naive
(single-global-rectifier) -- the latter was found badly miscalibrated under
MNAR-like labeling x real judge bias x coarse/discrete scales specifically
in this sweep (see mw_naive's Method docstring in methods.py), which is why
it was replaced here rather than kept alongside it. Deliberately
excludes the omnibus/multi-group tests (anova_ind/anova_rep/friedman/
kruskal/lmm*) and the non-standard bootstrap-CI constructions
(bayes_bootstrap/bootstrap_t/tango_score) -- those answer different
questions (multi-group omnibus effects, CI-based constructions), so
folding them into the SAME "pooled false-positive rate" would blend
apples with oranges rather than checking robustness across reasonable
alternatives, the same way build_ppi_factorial_sources/build_ppi_nlab_
grid_sources' paired_t-only scoping was never meant to claim the OTHER
PPI_TEST_METHODS behave identically."""
_COMPARISON_METHODS_OMNIBUS = (ANOVA_IND.name, ANOVA_REP.name, FRIEDMAN.name, KRUSKAL_CORR.name)
"""The four omnibus/multi-group tests -- run alongside _COMPARISON_METHODS
against the SAME factorial sources (build_ppi_factorial_sources), using the
SAME 5-way (all_human/human_subset/llm_only/llm_impute/ppi) machinery, but
NEVER pooled together with _COMPARISON_METHODS into one averaged rate: these
answer a genuinely different question (are the 3 groups/conditions
different at all, vs. _COMPARISON_METHODS' specific two-group location-shift
question) -- see _COMPARISON_METHODS' own docstring for why blending the two
would be apples-with-oranges. anova_ind/kruskal_corr use the
independent-3-group structure (a3/b3/c3); anova_rep/friedman use the
repeated-3-group structure (A/B/C) -- see _COMPARISON_METHOD_STRUCTURE's
"group3"/"pair3" entries. Uses kruskal_corr (the per-group, per-score-bin
locally-corrected Wald test -- evalstats.tests.
_ppi_kruskal_wallis_pairwise_corrected), not kruskal_naive
(single-global-rectifier) -- the latter was found badly miscalibrated under
the same combined bias x MNAR-labeling x coarse-scale x large-N stress that
broke mw_naive, once this factorial sweep was extended to the omnibus tests
(see KRUSKAL_NAIVE/KRUSKAL_CORR's Method docstrings in methods.py), which is
why it was replaced here rather than kept alongside it -- the same
reasoning _COMPARISON_METHODS already applied to mwu_corr vs. mw_naive.
Pool these among THEMSELVES (pool_ppi_comparison_across_methods, or a
filtered subset of `results`) for their own "mean_of_4_omnibus" summary,
kept in its own report section/log rather than merged into the headline
_COMPARISON_METHODS one."""
_COMPARISON_METHOD_STRUCTURE = {
    TTEST_WELCH.name: "group", MWU_CORR.name: "group", MW_NAIVE.name: "group",
    PAIRED_T.name: "pair", WILCOXON.name: "pair",
    ANOVA_IND.name: "group3", KRUSKAL_NAIVE.name: "group3", KRUSKAL_CORR.name: "group3",
    ANOVA_REP.name: "pair3", FRIEDMAN.name: "pair3",
}
_COMPARISON_METHODS_LABEL = "ttest_welch/paired_t/mwu_corr/wilcoxon"
_COMPARISON_METHODS_OMNIBUS_LABEL = "anova_ind/anova_rep/friedman/kruskal"
POOLED_METHOD_LABEL = "mean_of_4"
"""PPIComparisonResult.method value for a row produced by
pool_ppi_comparison_across_methods -- distinguishes a pooled/averaged row
from a genuine single-method one (never a value _run_ppi_comparison_cell
itself produces). Used for both the _COMPARISON_METHODS pool and the
_COMPARISON_METHODS_OMNIBUS pool -- callers keep the two separate by never
pooling a `results` list that mixes both method sets together (see
_COMPARISON_METHODS_OMNIBUS' docstring)."""


def _classical_pvalue(a: np.ndarray, b: np.ndarray, method: str, structure: str) -> float:
    """The SAME classical-test call _run_ppi_cell uses for this method
    (ttest_ind/mannwhitneyu for "group", ttest_rel/wilcoxon for "pair"),
    factored out here so it can be reused identically for all_human,
    human_subset, llm_only, and llm_impute -- every arm of the comparison
    for a given method uses the SAME test, just on different input arrays,
    so the comparison is apples-to-apples per method (e.g. the "oracle"
    all_human/human_subset arms run Mann-Whitney on truth for the
    "mwu_corr"/"mw_naive" method-rows, not always a t-test)."""
    if structure == "group":
        if method == TTEST_WELCH.name:
            return float(scipy_stats.ttest_ind(a, b, equal_var=False).pvalue)
        return float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    if method == PAIRED_T.name:
        return float(scipy_stats.ttest_rel(a, b).pvalue)
    return float(scipy_stats.wilcoxon(a, b, alternative="two-sided").pvalue)


def _ppi_comparison_pvalue(a: np.ndarray, b: np.ndarray, a_lab: np.ndarray, b_lab: np.ndarray, method: str, structure: str, n_boot: int, seed: int) -> float:
    """The SAME PPI-corrected call _run_ppi_cell uses for this method
    (_ppi_two_sample / _ppi_two_sample_midrank_corrected for "group"
    methods, _ppi_paired_arrays for "pair" methods -- see _run_ppi_cell's
    ttest_welch/mw_naive/mwu_corr/paired_t/wilcoxon blocks, which this
    mirrors exactly)."""
    if structure == "group":
        if method == TTEST_WELCH.name:
            estimator = lambda ya, yb: float(ya.mean() - yb.mean())  # noqa: E731
            return _ppi_two_sample(a, b, a_lab, b_lab, estimator, _ALPHA, n_boot, seed).p_value
        if method == MWU_CORR.name:
            return _ppi_two_sample_midrank_corrected(a, b, a_lab, b_lab, _ALPHA, n_boot, seed).p_value
        # mw_naive: single-global-rectifier midrank correction -- kept for
        # direct comparison against mwu_corr, not used by _COMPARISON_METHODS
        # itself (see that constant's docstring for why it was replaced there).
        estimator = lambda xa, ya: _p_x_gt_y_midrank(xa, ya) - 0.5  # noqa: E731
        return _ppi_two_sample(a, b, a_lab, b_lab, estimator, _ALPHA, n_boot, seed).p_value
    statistic = np.mean if method == PAIRED_T.name else np.median
    return _ppi_paired_arrays(a, b, a_lab, b_lab, statistic, _ALPHA, n_boot, seed, rectifier_func=np.mean).p_value


def _classical_pvalue_omnibus(groups: list[np.ndarray], method: str) -> float:
    """Omnibus counterpart to _classical_pvalue, for the 3-group
    _COMPARISON_METHODS_OMNIBUS methods -- the SAME uncorrected-p-value
    calls _run_ppi_cell's anova_ind/anova_rep/friedman/kruskal blocks use
    (_uncorrected_anova_independent_p_value etc.), reused identically here
    for all_human/human_subset/llm_only/llm_impute."""
    if method == ANOVA_IND.name:
        return _uncorrected_anova_independent_p_value(groups)
    if method == ANOVA_REP.name:
        return _uncorrected_anova_repeated_p_value(groups)
    if method == FRIEDMAN.name:
        return _uncorrected_friedman_p_value(groups)
    return _uncorrected_kruskal_p_value(groups)  # KRUSKAL_NAIVE.name / KRUSKAL_CORR.name (same uncorrected test)


def _ppi_comparison_pvalue_omnibus(
    groups: list[np.ndarray], groups_lab: list[np.ndarray], method: str, n_boot: int, seed: int,
) -> float | None:
    """Omnibus counterpart to _ppi_comparison_pvalue -- the SAME
    PPI-corrected calls _run_ppi_cell's anova_ind/anova_rep/friedman/kruskal
    blocks use, reused identically here. May return None (anova_ind/
    anova_rep/friedman's PPI-corrected p-value functions can return None on
    a degenerate fit -- see their own docstrings); the caller must treat
    that as "not rejected," matching _run_ppi_cell's `p is not None and p <
    alpha` pattern."""
    k = len(groups)
    if method == ANOVA_IND.name:
        return _ppi_anova_independent_p_value(groups, groups_lab, k=k)
    if method == ANOVA_REP.name:
        return _ppi_anova_repeated_p_value(groups, groups_lab, k=k)
    if method == FRIEDMAN.name:
        return _ppi_friedman_p_value(groups, groups_lab, k=k)
    if method == KRUSKAL_NAIVE.name:
        pw = _ppi_kruskal_wallis_pairwise(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=seed)
        return pw["wald_p"]
    # KRUSKAL_CORR.name
    pw = _ppi_kruskal_wallis_pairwise_corrected(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=seed)
    return pw["wald_p"]


_COMPARISON_CELL_FIELDS = {
    "group": (("llm_a2", "llm_b2"), ("lab_a2", "lab_b2"), ("truth_a2", "truth_b2"), "independent"),
    "pair": (("llm_x", "llm_y"), ("lab_x", "lab_y"), ("truth_x", "truth_y"), "shared"),
    "group3": (("llm_a3", "llm_b3", "llm_c3"), ("lab_a3", "lab_b3", "lab_c3"), ("truth_a3", "truth_b3", "truth_c3"), "independent"),
    "pair3": (("llm_A", "llm_B", "llm_C"), ("lab_A", "lab_B", "lab_C"), ("truth_A", "truth_B", "truth_C"), "shared"),
}
"""Maps each _COMPARISON_METHOD_STRUCTURE value to the JudgeBiasCellData
field names it reads, and whether its labeling mask is "independent" (each
group masked separately, e.g. group/group3's _jb_labels_independent) or
"shared" (one mask reused across every group, e.g. pair/pair3's
_jb_labels_shared) -- see _run_ppi_comparison_cell."""


def _run_ppi_comparison_cell(sc: JudgeBiasSource, n_reps: int, n_boot: int, seed, method: str) -> PPIComparisonResult:
    """Runs the 5-way comparison (all_human/human_subset/llm_only/
    llm_impute/ppi) for ONE classical `method` (see _COMPARISON_METHODS/
    _COMPARISON_METHODS_OMNIBUS). Dispatches on
    _COMPARISON_METHOD_STRUCTURE[method] via _COMPARISON_CELL_FIELDS:
    "group"/"group3" methods (ttest_welch, mwu_corr, anova_ind, kruskal) use
    generate_judge_bias_cell's independent-group structure (2 or 3 groups);
    "pair"/"pair3" methods (paired_t, wilcoxon, anova_rep, friedman) use its
    paired/repeated structure. generate_judge_bias_cell draws EVERY
    structure every replicate regardless of which this call's method needs
    (one rng stream, unused structures simply unused, not skipped -- keeps a
    given scenario/seed's draws identical regardless of which method is
    requested, the same reproducibility property --tests relies on in
    _run_ppi_cell).

    Each of the four arms is computed in its OWN try/except: a classical-
    test failure (e.g. wilcoxon raising on an all-zero-difference sample)
    just skips incrementing that arm for that replicate (same semantics as
    before this function supported rank-based tests, which never failed);
    only a PPI bootstrap-correction failure increments n_failed, preserving
    that field's original meaning. For "group3"/"pair3" methods, the
    PPI-corrected p-value can also be None on a degenerate fit (anova_ind/
    anova_rep/friedman -- see _ppi_comparison_pvalue_omnibus) -- treated as
    "not rejected," not a failure, matching _run_ppi_cell.

    n_lab (the realized labeled-item count) is the FIRST group's count --
    for "independent"-mask structures (group/group3) since every
    JudgeBiasSource this comparison-sweep machinery builds leaves n2/n3
    unset (so n2==n3==n) and applies the SAME label_frac to every group, so
    all groups are expected to match; for "shared"-mask structures
    (pair/pair3) every group shares one mask anyway, so the first group's
    count IS the shared count."""
    rng = np.random.default_rng(seed)
    rejects = {"all_human": 0, "human_subset": 0, "llm_only": 0, "llm_impute": 0, "ppi": 0}
    n_failed = 0
    n_lab_realized = 0
    structure = _COMPARISON_METHOD_STRUCTURE[method]
    llm_fields, lab_fields, truth_fields, mask_kind = _COMPARISON_CELL_FIELDS[structure]
    is_omnibus = structure in ("group3", "pair3")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_reps):
            cell = generate_judge_bias_cell(sc, rng)
            llm_groups = [getattr(cell, f) for f in llm_fields]
            lab_groups = [getattr(cell, f) for f in lab_fields]
            truth_groups = [getattr(cell, f) for f in truth_fields]

            if mask_kind == "independent":
                masks = [~np.isnan(lab) for lab in lab_groups]
                subset_ok = all(int(m.sum()) >= 2 for m in masks)
            else:
                shared_mask = np.logical_and.reduce([~np.isnan(lab) for lab in lab_groups])
                masks = [shared_mask] * len(lab_groups)
                subset_ok = int(shared_mask.sum()) >= 2
            n_lab_realized = int(masks[0].sum())
            truth_subset_groups = [t[m] for t, m in zip(truth_groups, masks)]
            filled_groups = []
            for llm, lab, m in zip(llm_groups, lab_groups, masks):
                filled = llm.copy()
                filled[m] = lab[m]
                filled_groups.append(filled)

            if is_omnibus:
                classical = lambda groups: _classical_pvalue_omnibus(groups, method)  # noqa: E731
            else:
                classical = lambda groups: _classical_pvalue(groups[0], groups[1], method, structure)  # noqa: E731

            try:
                p_all_human = classical(truth_groups)
                rejects["all_human"] += int(p_all_human < _ALPHA)
            except Exception:
                pass

            try:
                p_llm_only = classical(llm_groups)
                rejects["llm_only"] += int(p_llm_only < _ALPHA)
            except Exception:
                pass

            try:
                p_llm_impute = classical(filled_groups)
                rejects["llm_impute"] += int(p_llm_impute < _ALPHA)
            except Exception:
                pass

            if subset_ok:
                try:
                    p_human_subset = classical(truth_subset_groups)
                    rejects["human_subset"] += int(p_human_subset < _ALPHA)
                except Exception:
                    pass

            try:
                ppi_seed = int(rng.integers(0, 2 ** 31))
                if is_omnibus:
                    p_ppi = _ppi_comparison_pvalue_omnibus(llm_groups, lab_groups, method, n_boot, ppi_seed)
                    rejects["ppi"] += int(p_ppi is not None and p_ppi < _ALPHA)
                else:
                    p_ppi = _ppi_comparison_pvalue(
                        llm_groups[0], llm_groups[1], lab_groups[0], lab_groups[1], method, structure, n_boot, ppi_seed,
                    )
                    rejects["ppi"] += int(p_ppi < _ALPHA)
            except Exception:
                n_failed += 1

    return PPIComparisonResult(
        name=sc.name, tag=sc.tag, eval_type=sc.eval_type, n=sc.n, n_reps=n_reps,
        effect_size=_ppi_source_effect_frac(sc), label_frac=sc.label_frac, n_lab=n_lab_realized, method=method,
        rejects_all_human=rejects["all_human"], rejects_human_subset=rejects["human_subset"],
        rejects_llm_only=rejects["llm_only"], rejects_llm_impute=rejects["llm_impute"],
        rejects_ppi=rejects["ppi"], n_failed=n_failed,
    )


def _run_ppi_comparison_cell_worker(args: tuple) -> PPIComparisonResult:
    sc, n_reps, n_boot, seed, method = args
    return _run_ppi_comparison_cell(sc, n_reps, n_boot, seed, method)


def run_ppi_comparison_simulation(
    sources: list[JudgeBiasSource], n_reps: int, n_boot: int,
    progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
    methods: tuple = _COMPARISON_METHODS,
) -> list[PPIComparisonResult]:
    """Runs _run_ppi_comparison_cell for every (source, method) pair --
    len(sources) x len(methods) cells total -- returning a FLAT list (each
    PPIComparisonResult.method identifies which). Pool across methods with
    pool_ppi_comparison_across_methods for a single averaged row per
    scenario; group by .method for the per-method breakdown.

    n_workers=1 (the default) runs sequentially -- fine for the original
    ~24-scenario comparison grid (build_ppi_power_sources +
    build_ppi_comparison_label_frac_sources), where forking a worker pool
    would be pure overhead relative to the work itself. build_ppi_nlab_grid_
    sources (~44 scenarios), build_ppi_factorial_sources (~312), and now
    the x4 method sweep push this well past that point, so this supports
    the same fork-pool-over-sources pattern as run_ppi_simulation/
    run_multiarm_simulation (no in-cell progress-dict machinery, unlike
    run_ppi_simulation -- _run_ppi_comparison_cell is fast enough per
    (source, method) cell, seconds not minutes, that per-cell granularity
    isn't needed)."""
    cells = [(sc, m) for sc in sources for m in methods]
    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pvalues-ppi-compare")
    results: list[PPIComparisonResult] = []
    if n_workers <= 1:
        for i, ((sc, m), child_seed) in enumerate(zip(cells, child_seeds)):
            results.append(_run_ppi_comparison_cell(sc, n_reps, n_boot, child_seed, m))
            reporter.update(i + 1, detail=f"{sc.name} [{m}]")
    else:
        args_list = [(sc, n_reps, n_boot, child_seed, m) for (sc, m), child_seed in zip(cells, child_seeds)]
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_ppi_comparison_cell_worker, args_list)):
                results.append(result)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


def pool_ppi_comparison_across_methods(results: list[PPIComparisonResult]) -> list[PPIComparisonResult]:
    """Pool PPIComparisonResult rows across _COMPARISON_METHODS, one output
    row per distinct scenario `name` (summing rejects/n_reps across
    whichever methods are present for that name -- equivalent to averaging
    each method's rate since every method shares the same n_reps per
    scenario). Output rows carry method=POOLED_METHOD_LABEL. This is the
    "average across ttest_welch/paired_t/mwu_corr/wilcoxon" pooling requested for
    the headline figures (null-effect bar chart, 5-way comparison, N x
    N_lab grid, factorial slices) -- the per-method rows in `results`
    remain available (e.g. in the raw CSV) as the supplementary robustness
    breakdown, so a reviewer can check the average isn't hiding one method
    behaving badly."""
    by_name: dict[str, list[PPIComparisonResult]] = defaultdict(list)
    for r in results:
        by_name[r.name].append(r)

    pooled: list[PPIComparisonResult] = []
    for name, rows in by_name.items():
        r0 = rows[0]
        pooled.append(PPIComparisonResult(
            name=name, tag=r0.tag, eval_type=r0.eval_type, n=r0.n,
            n_reps=sum(r.n_reps for r in rows),
            effect_size=r0.effect_size, label_frac=r0.label_frac, n_lab=r0.n_lab,
            method=POOLED_METHOD_LABEL,
            rejects_all_human=sum(r.rejects_all_human for r in rows),
            rejects_human_subset=sum(r.rejects_human_subset for r in rows),
            rejects_llm_only=sum(r.rejects_llm_only for r in rows),
            rejects_llm_impute=sum(r.rejects_llm_impute for r in rows),
            rejects_ppi=sum(r.rejects_ppi for r in rows),
            n_failed=sum(r.n_failed for r in rows),
        ))
    return pooled


def _pool_ppi_comparison_rows(rows: list[PPIComparisonResult]) -> PPIComparisonResult | None:
    """Pool an arbitrary list of PPIComparisonResult rows -- possibly
    differing by method, N, N_lab, or anything else -- into ONE combined
    row: sums rejects/n_reps (equivalent to an unweighted average of each
    row's rate, since every row shares the same n_reps by construction in
    this harness), keeps the first row's descriptive metadata (name/tag/
    eval_type/n/n_lab/etc., which are display fields here, not recomputed
    as an average across whatever heterogeneous scenarios were pooled).
    Used for save_ppi_null_comparison_plot's continuous-eval-type panel,
    which pools ACROSS THE N x N_lab GRID on top of pool_ppi_comparison_
    across_methods' across-methods pooling -- two independent pooling
    axes, scenario and method, both folded into a single number/CI."""
    if not rows:
        return None
    r0 = rows[0]
    return PPIComparisonResult(
        name=r0.name, tag=r0.tag, eval_type=r0.eval_type, n=r0.n, n_reps=sum(r.n_reps for r in rows),
        effect_size=r0.effect_size, label_frac=r0.label_frac, n_lab=r0.n_lab, method=POOLED_METHOD_LABEL,
        rejects_all_human=sum(r.rejects_all_human for r in rows),
        rejects_human_subset=sum(r.rejects_human_subset for r in rows),
        rejects_llm_only=sum(r.rejects_llm_only for r in rows),
        rejects_llm_impute=sum(r.rejects_llm_impute for r in rows),
        rejects_ppi=sum(r.rejects_ppi for r in rows),
        n_failed=sum(r.n_failed for r in rows),
    )


def print_ppi_comparison_report(results: list[PPIComparisonResult], alpha: float) -> None:
    """Five-way rejection-rate table (paired_t estimand): all_human /
    human_subset / llm_only / llm_impute / ppi, grouped by tag (vs.
    effect_size, tag="power"; vs. label_frac, tag="compare_label_frac") then
    eval_type."""
    if not results:
        print("\n  (no PPI comparison results)")
        return
    print(f"\n{'='*96}\n  PVALUES (PPI-CORRECTED) -- ESTIMATOR COMPARISON (paired_t)\n"
          f"  all_human=oracle full-N truth | human_subset=labeled-only truth | llm_only=uncorrected |\n"
          f"  llm_impute=LLM+label-overwrite, no PPI rectifier | ppi=PPI-corrected | alpha={alpha}\n{'='*96}")

    for tag, x_field, x_label, x_fmt in [
        ("power", "effect_size", "es", "{:.2f}"), ("compare_label_frac", "n_lab", "nlab", "{:d}"),
    ]:
        tag_rows = [r for r in results if r.tag == tag]
        if not tag_rows:
            continue
        x_values = sorted({getattr(r, x_field) for r in tag_rows})
        eval_types = sorted({r.eval_type for r in tag_rows})
        print(f"\n  -- vs. {x_field} --")
        for et in eval_types:
            print(f"\n  [{et}]")
            print(f"    {'':<12}" + "".join((x_label + "=" + x_fmt.format(v)).rjust(11) for v in x_values))
            for col, label in [
                ("rejects_all_human", "all_human"), ("rejects_human_subset", "human_subset"),
                ("rejects_llm_only", "llm_only"), ("rejects_llm_impute", "llm_impute"), ("rejects_ppi", "ppi"),
            ]:
                row = f"    {label:<12}"
                for v in x_values:
                    r = next((r for r in tag_rows if r.eval_type == et and getattr(r, x_field) == v), None)
                    rate = getattr(r, col) / r.n_reps if r is not None and r.n_reps > 0 else float("nan")
                    row += f"  {rate:>9.3f}" if np.isfinite(rate) else f"  {'-':>9}"
                print(row)
    print()


def save_results_artifacts_ppi_comparison(
    *, results: list[PPIComparisonResult], alpha: float, out_dir: str, run_stem: str,
    pooled_results: list[PPIComparisonResult] | None = None,
) -> list[str]:
    """`results` is the RAW (per-method, len(sources)*len(methods) rows) data
    -- saved verbatim to the CSV (the per-method breakdown, for reviewers to
    check the pooled average isn't hiding one method behaving badly).

    The saved .log, however, must be built from POOLED data (one row per
    scenario), matching what run()'s own console output already prints via
    print_ppi_comparison_report(comparison_results_pooled, ...) -- pass that
    same pooled list as `pooled_results`. Calling print_ppi_comparison_report
    on the raw rows instead (as an earlier version of this function did) is
    NOT just a cosmetic difference: it doesn't affect the GLM-based factorial
    report's coefficients (grouped-binomial log-likelihood is additive over
    rows sharing the same covariates, so pooled vs. unpooled fits are
    numerically identical there), but THIS function's report picks single
    rows via `next(...)` lookups keyed on eval_type/x_field alone -- fed raw
    data, that silently returns whichever METHOD happens to appear first for
    a given cell instead of the 4-method-averaged rate, discarding the other
    3 methods' data entirely. `pooled_results=None` (the default) falls back
    to pooling `results` internally so old call sites don't silently regress,
    but new callers should pass the already-pooled list run() computes
    anyway, rather than pay to re-derive it here."""
    if pooled_results is None:
        pooled_results = pool_ppi_comparison_across_methods(results)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_comparison_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "tag", "eval_type", "method", "n", "n_reps", "effect_size", "label_frac", "n_lab",
            "rate_all_human", "rate_human_subset", "rate_llm_only", "rate_llm_impute", "rate_ppi", "n_failed",
        ])
        for r in results:
            writer.writerow([
                r.name, r.tag, r.eval_type, r.method, r.n, r.n_reps, f"{r.effect_size:.4f}", f"{r.label_frac:.4f}", r.n_lab,
                f"{r.rejects_all_human / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_human_subset / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_only / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_impute / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_ppi / r.n_reps:.8f}" if r.n_reps else "",
                r.n_failed,
            ])
    summary_path = out_base / f"{run_stem}_ppi_comparison_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_comparison_report(pooled_results, alpha=alpha)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


_PPI_COMPARISON_STYLE = {
    "all_human":    dict(color="#1b9e77", marker="o", ls="-",  label="all human (oracle)"),
    "ppi":          dict(color="#d95f02", marker="o", ls="-",  label="PPI-corrected"),
    "llm_impute":   dict(color="#7570b3", marker="s", ls="--", label="LLM + label overwrite (no PPI)"),
    "llm_only":     dict(color="#e7298a", marker="^", ls="--", label="LLM only (uncorrected)"),
    "human_subset": dict(color="#666666", marker="d", ls=":",  label="human subset only"),
}
_PPI_COMPARISON_COLS = [
    ("all_human", "rejects_all_human"), ("ppi", "rejects_ppi"), ("llm_impute", "rejects_llm_impute"),
    ("llm_only", "rejects_llm_only"), ("human_subset", "rejects_human_subset"),
]


def save_ppi_comparison_plot(*, results: list[PPIComparisonResult], alpha: float, out_path: str) -> str:
    """The flagship 5-way estimator-comparison figure: rejection rate for
    all_human/human_subset/llm_only/llm_impute/ppi, one row per x-axis
    (effect_size, then label_frac), one column per eval type. The story this
    is built to show: human_subset and ppi should share all_human's Type-I
    error at effect_size=0 (all three are unbiased there) while llm_only/
    llm_impute are inflated; as effect_size grows, ppi's power curve should
    track much closer to all_human's than human_subset's flatter, small-N
    curve does -- i.e. PPI recovers most of all_human's power at a fraction
    of its labeling cost, which plain human-only subsetting cannot. The
    n_lab row shows the SAME story from the budget side: ppi's power should
    approach all_human's ceiling as N_lab grows, while human_subset's power
    stays low even at higher N_lab (still small relative to all_human's full
    N). Plotted against the REALIZED N_lab (PPIComparisonResult.n_lab), not
    the nominal label_frac -- see PPI_COMPARISON_LABEL_FRACS' docstring for
    why label_frac alone can be misleading once _JB_MIN_LAB's floor binds."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No PPI comparison results to plot.")
    rows_spec = [
        ("power", "effect_size", "Effect size", "label budget fixed at N_lab/N = 20%"),
        ("compare_label_frac", "n_lab", "N_lab (labeled items)", f"effect size fixed at {PPI_COMPARISON_MODERATE_EFFECT_FRAC:.0%} of scale (moderate)"),
    ]
    rows_spec = [(tag, field, xlabel, fixed) for tag, field, xlabel, fixed in rows_spec if any(r.tag == tag for r in results)]
    eval_types = sorted({r.eval_type for r in results})

    fig, axes = plt.subplots(
        len(rows_spec), len(eval_types), figsize=(4.8 * len(eval_types), 4.0 * len(rows_spec)), squeeze=False,
    )
    for row_idx, (tag, field, xlabel, fixed) in enumerate(rows_spec):
        tag_rows = [r for r in results if r.tag == tag]
        x_values = sorted({getattr(r, field) for r in tag_rows})
        for col_idx, et in enumerate(eval_types):
            ax = axes[row_idx][col_idx]
            ax.axhline(
                alpha, color="black", ls="--", lw=1.0, alpha=0.5,
                label=f"Nominal {_alpha_label(alpha)}" if row_idx == 0 and col_idx == 0 else None,
            )
            et_rows = {getattr(r, field): r for r in tag_rows if r.eval_type == et}
            for key, rejects_field in _PPI_COMPARISON_COLS:
                style = _PPI_COMPARISON_STYLE[key]
                ys = [
                    (getattr(et_rows[x], rejects_field) / et_rows[x].n_reps) if x in et_rows and et_rows[x].n_reps else float("nan")
                    for x in x_values
                ]
                ax.plot(
                    x_values, ys, color=style["color"], marker=style["marker"], linestyle=style["ls"],
                    linewidth=1.8, markersize=5, label=style["label"] if row_idx == 0 and col_idx == 0 else None,
                )
            ax.set_ylim(-0.02, 1.02)
            if row_idx == 0:
                ax.set_title(et.capitalize())
            if col_idx == 0:
                ax.set_ylabel("Rejection rate")
            ax.set_xlabel(f"{xlabel}\n({fixed})", fontsize=9)
    fig.legend(loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.08 if len(rows_spec) > 1 else -0.14))
    fig.suptitle("PPI-Corrected Estimator Comparison (Paired-Mean Estimand)", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


_PPI_NULL_COMPARISON_ORDER = ["all_human", "human_subset", "ppi", "llm_impute", "llm_only"]
"""Deliberately NOT alphabetical or the same left-to-right order as
save_ppi_comparison_plot's legend: groups the three arms that SHOULD be
well-calibrated at the null (all_human, human_subset, ppi) together on the
left, then the two that are biased by construction (llm_impute, llm_only)
on the right -- the grouping itself is part of what makes the bar chart
read at a glance."""


def save_ppi_null_comparison_plot(
    *, results: list[PPIComparisonResult], alpha: float, out_path: str,
    nlab_cal_results: list[PPIComparisonResult] | None = None,
) -> str:
    """Bar chart isolating JUST the null (no real effect) case from
    save_ppi_comparison_plot's line plot -- one bar per estimator arm, one
    panel per eval type. save_ppi_comparison_plot's effect_size=0 point
    carries the same numbers, but buried as one of several x-values on a
    line plot built to tell a POWER story -- easy to misread llm_only/
    llm_impute's high rejection rate at small real effect sizes as "more
    powerful than PPI" there, when it's actually inflated false positives
    from judge bias, not power (build_ppi_power_sources fixes bias
    direction to OPPOSE the injected effect; the observed uncorrected
    difference is `effect_size - bias_delta`, so llm_only/llm_impute
    already reject at ~100% at effect_size=0, before any real effect exists
    at all -- see save_ppi_power_direction_plot for the reinforcing-bias
    mirror image, where the same arms would instead overstate a real effect
    that IS present). This plot has no effect_size axis to be misread
    against: every bar here is, by construction, a false-positive rate.

    Every bar pools across _COMPARISON_METHODS (ttest_welch/paired_t/mwu_corr/
    wilcoxon -- `results`/`nlab_cal_results` are expected to already be
    pool_ppi_comparison_across_methods output, one row per scenario, not
    the raw per-method rows). For continuous and likert specifically,
    passing `nlab_cal_results` (build_ppi_nlab_grid_sources' calibration
    grid, ALSO pre-pooled across methods, now itself crossing continuous/
    likert) pools a SECOND axis on top, per eval type: every N x N_lab cell
    in that eval type's slice of the grid, not just the single (N=100,
    N_lab=20) baseline scenario `results` alone would give -- the
    "defensible for a paper" version of this chart, an average over 4 tests
    x ~22 (N, N_lab) conditions rather than one arbitrarily-chosen scenario.
    grades has no such sweep available (build_ppi_nlab_grid_sources
    deliberately excludes it as redundant with continuous), so it falls
    back to `results`' single scenario -- still pooled across the 4
    methods, just not across N/N_lab. Each panel's subtitle states which
    pooling applies.

    Error bars are the 95% Wilson score interval for each bar's underlying
    binomial proportion (_ppi_wilson_interval, the same interval
    print_ppi_report's Type-I flagging already uses), computed on the
    POOLED rejects/n_reps -- i.e. treating every pooled replicate as one
    more independent Bernoulli draw at the same rate. That's exact for
    pooling across methods/conditions that are truly identically
    calibrated, and a standard, if slightly optimistic, simplification
    if there's real heterogeneity across the pooled methods/(N, N_lab)
    cells (the same simplification this file's pooled Type-I metrics
    already use, e.g. key_metrics["ppi_mean_corrected_type1"]) -- called
    out here rather than presented as more rigorous than it is."""
    import matplotlib.pyplot as plt

    null_rows = [r for r in results if r.tag == "power" and abs(r.effect_size) < 1e-9]
    if not null_rows:
        raise ValueError("No null-effect (effect_size=0) comparison results to plot.")
    eval_types = sorted({r.eval_type for r in null_rows})
    nlab_null_pool_by_et: dict[str, PPIComparisonResult | None] = {}
    if nlab_cal_results:
        for et in {r.eval_type for r in nlab_cal_results}:
            nlab_null_pool_by_et[et] = _pool_ppi_comparison_rows(
                [r for r in nlab_cal_results if r.tag == "nlab_grid" and r.eval_type == et]
            )

    fig, axes = plt.subplots(1, len(eval_types), figsize=(3.4 * len(eval_types), 4.9), squeeze=False)
    x = np.arange(len(_PPI_NULL_COMPARISON_ORDER))
    for col, et in enumerate(eval_types):
        ax = axes[0][col]
        if nlab_null_pool_by_et.get(et) is not None:
            r = nlab_null_pool_by_et[et]
            subtitle = f"pooled: 4 tests x N x N_lab\n(n_reps={r.n_reps})"
        else:
            r = next(r for r in null_rows if r.eval_type == et)
            subtitle = f"pooled: 4 tests\n(N={r.n}, N_lab={r.n_lab}, n_reps={r.n_reps})"
        rejects = [getattr(r, f"rejects_{key}") for key in _PPI_NULL_COMPARISON_ORDER]
        rates = [k / r.n_reps if r.n_reps else float("nan") for k in rejects]
        ci_lo_hi = [_ppi_wilson_interval(k, r.n_reps) for k in rejects]
        yerr = [
            [max(0.0, rate - lo) for rate, (lo, _hi) in zip(rates, ci_lo_hi)],
            [max(0.0, hi - rate) for rate, (_lo, hi) in zip(rates, ci_lo_hi)],
        ]
        colors = [_PPI_COMPARISON_STYLE[key]["color"] for key in _PPI_NULL_COMPARISON_ORDER]
        ax.bar(x, rates, color=colors, width=0.65, zorder=2)
        ax.errorbar(
            x, rates, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=4,
            label="95% Wilson CI" if col == 0 else None,
        )
        ax.axhline(
            alpha, color="black", ls="--", lw=1.2, zorder=3,
            label=f"Nominal {_alpha_label(alpha)}" if col == 0 else None,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_PPI_COMPARISON_STYLE[key]["label"] for key in _PPI_NULL_COMPARISON_ORDER],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylim(0.0, 1.08)
        ax.set_title(f"{et.capitalize()}\n{subtitle}", fontsize=10)
        ax.set_ylabel("False positive rate" if col == 0 else "")
        ax.grid(axis="y", alpha=0.25, lw=0.8, zorder=1)
        for xi, rate in zip(x, rates):
            if np.isfinite(rate):
                ax.text(xi, rate + 0.02, f"{rate:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("False-Positive Rate Under the Null (No Real Effect)", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_ppi_nlab_grid_report(
    results: list[PPIComparisonResult], alpha: float, header: str = "N x N_LAB GRID (calibration)",
) -> None:
    """N (columns) x N_lab (rows) grid table, one mini-table per arm
    (all_human / human_subset / ppi -- the three arms build_ppi_nlab_grid_
    sources' question is actually about; llm_only/llm_impute are omitted
    here since they aren't valid tests regardless of N/N_lab, so their rate
    doesn't answer a calibration-vs-(N,N_lab) question). Reading ACROSS a
    row (fixed N_lab, N varying) isolates N's effect; reading DOWN a column
    (fixed N, N_lab varying) isolates N_lab's effect -- directly separating
    "it's the ratio N_lab/N that matters" from "it's the absolute N_lab
    count that matters," which build_ppi_comparison_label_frac_sources'
    N=100-only sweep can't do on its own. One grid per eval type
    (build_ppi_nlab_grid_sources now crosses continuous/likert): grouping
    by `.eval_type` here isn't optional once more than one eval type is
    present -- (N, N_lab) pairs collide across eval types (each combination
    is generated for every eval type), so reading `results` without
    grouping would silently pick whichever eval type's row happened to come
    first for each cell."""
    if not results:
        print(f"\n  (no {header} results)")
        return
    eval_types = sorted({r.eval_type for r in results})
    print(f"\n{'='*88}\n  PVALUES (PPI-CORRECTED) -- {header}\n"
          f"  Rows = N_lab (labeled items), columns = N (total items); nominal alpha={alpha}\n{'='*88}")
    for et in eval_types:
        et_results = [r for r in results if r.eval_type == et]
        n_values = sorted({r.n for r in et_results})
        nlab_values = sorted({r.n_lab for r in et_results})
        print(f"\n  === {et.capitalize()} ===")
        for label, rejects_field in [
            ("all_human", "rejects_all_human"), ("human_subset", "rejects_human_subset"), ("ppi", "rejects_ppi"),
        ]:
            print(f"\n  [{label}]")
            print(f"    {'N_lab \\ N':<10}" + "".join(f"n={n}".rjust(9) for n in n_values))
            for nlab in nlab_values:
                row = f"    {nlab:<10}"
                for n in n_values:
                    r = next((r for r in et_results if r.n == n and r.n_lab == nlab), None)
                    if r is None or r.n_reps == 0:
                        row += f"{'-':>9}"
                        continue
                    rate = getattr(r, rejects_field) / r.n_reps
                    row += f"{rate:>9.3f}"
                print(row)
    print()


def save_results_artifacts_ppi_nlab_grid(
    *, results: list[PPIComparisonResult], alpha: float, out_dir: str, run_stem: str, header: str,
    pooled_results: list[PPIComparisonResult] | None = None,
) -> list[str]:
    """Same CSV shape as save_results_artifacts_ppi_comparison, but logs via
    print_ppi_nlab_grid_report instead -- that function's tag-based grouping
    (tag "power" / "compare_label_frac") doesn't match this grid's tags
    ("nlab_grid" / "nlab_grid_power"), so reusing it directly would produce
    an empty-looking log.

    `results` is the RAW (per-method) data, saved verbatim to the CSV.
    `pooled_results` (falls back to pooling `results` if omitted) feeds the
    saved .log instead -- print_ppi_nlab_grid_report's `next((r for r in
    et_results if r.n == n and r.n_lab == nlab), None)` cell lookup picks
    the first matching row for each (N, N_lab) cell, so fed raw data it
    silently reports whichever METHOD happens to appear first instead of
    the 4-method-averaged rate. See save_results_artifacts_ppi_comparison's
    docstring for the same issue there."""
    if pooled_results is None:
        pooled_results = pool_ppi_comparison_across_methods(results)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_nlab_grid_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "tag", "eval_type", "method", "n", "n_lab", "n_reps", "effect_size",
            "rate_all_human", "rate_human_subset", "rate_llm_only", "rate_llm_impute", "rate_ppi", "n_failed",
        ])
        for r in results:
            writer.writerow([
                r.name, r.tag, r.eval_type, r.method, r.n, r.n_lab, r.n_reps, f"{r.effect_size:.4f}",
                f"{r.rejects_all_human / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_human_subset / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_only / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_llm_impute / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_ppi / r.n_reps:.8f}" if r.n_reps else "",
                r.n_failed,
            ])
    summary_path = out_base / f"{run_stem}_ppi_nlab_grid_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_nlab_grid_report(pooled_results, alpha=alpha, header=header)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_nlab_grid_plot(
    *, calibration_results: list[PPIComparisonResult] | None, power_results: list[PPIComparisonResult] | None,
    alpha: float, out_path: str,
) -> str:
    """Heatmap(s) of the PPI-corrected rejection rate over the (N, N_lab)
    plane -- calibration (effect_size=0, diverging colormap centered on
    alpha so under/over-rejection are visually distinct) and power
    (moderate effect_size, sequential colormap), side by side when both are
    given. This is the direct visual answer to "is it the ratio N_lab/N or
    the absolute N_lab that drives calibration/power": scanning a ROW shows
    N's effect at fixed N_lab, scanning a COLUMN shows N_lab's effect at
    fixed N -- the line plots elsewhere in this module can't show this
    since they never vary N and N_lab independently (build_ppi_power_sources
    fixes N=100; build_ppi_comparison_label_frac_sources also fixes N=100
    and only varies the ratio).

    One ROW per eval type present across calibration_results/power_results
    (build_ppi_nlab_grid_sources now crosses continuous/likert), one COLUMN
    per panel (calibration and/or power) -- (N, N_lab) pairs collide across
    eval types the same way they do in print_ppi_nlab_grid_report, so each
    panel's grid is built from that eval type's rows only."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    panels = []
    if calibration_results:
        panels.append(("Type-I Error\n(no real effect)", calibration_results, "RdBu_r", alpha))
    if power_results:
        panels.append(("Power\n(moderate real effect)", power_results, "viridis", None))
    if not panels:
        raise ValueError("No N x N_lab grid results to plot.")
    eval_types = sorted({r.eval_type for _title, results, _cmap, _center in panels for r in results})

    fig, axes = plt.subplots(
        len(eval_types), len(panels), figsize=(6.0 * len(panels), 5.0 * len(eval_types)), squeeze=False,
    )
    for row, et in enumerate(eval_types):
        for col, (title, results, cmap, center) in enumerate(panels):
            ax = axes[row][col]
            et_results = [r for r in results if r.eval_type == et]
            n_values = sorted({r.n for r in et_results})
            nlab_values = sorted({r.n_lab for r in et_results})
            grid = np.full((len(nlab_values), len(n_values)), np.nan)
            for r in et_results:
                if r.n_reps == 0:
                    continue
                grid[nlab_values.index(r.n_lab), n_values.index(r.n)] = r.rejects_ppi / r.n_reps

            if center is not None:
                vmax = max(2.0 * center, float(np.nanmax(grid)) * 1.1 if np.isfinite(np.nanmax(grid)) else 2.0 * center)
                im = ax.imshow(grid, origin="lower", cmap=cmap, norm=TwoSlopeNorm(vmin=0.0, vcenter=center, vmax=vmax), aspect="auto")
            else:
                im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
            for i in range(len(nlab_values)):
                for j in range(len(n_values)):
                    val = grid[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black",
                            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.0),
                        )
            ax.set_xticks(range(len(n_values)))
            ax.set_xticklabels([str(n) for n in n_values])
            ax.set_yticks(range(len(nlab_values)))
            ax.set_yticklabels([str(nl) for nl in nlab_values])
            ax.set_xlabel("N (total items)")
            ax.set_ylabel("N_lab (labeled items)")
            ax.set_title(f"[{et.capitalize()}] {title}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"PPI-Corrected Rejection Rate over N × N_lab (nominal {_alpha_label(alpha)})", y=1.02, fontsize=12)
    fig.text(0.5, -0.02, "Paired-mean estimand", ha="center", fontsize=8, color="#555555")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PPI mode, full factorial: build_ppi_factorial_sources' 7-factor cross
# (bias_magnitude x N x N_lab x label_mechanism x effect_size x
# bias_direction x llm_noise), analyzed three ways -- a pooled binomial GLM
# (which factors/interactions actually move the PPI-corrected rejection
# rate, with real coefficients/p-values, fit at the llm_noise=0.20 baseline
# -- see _PPI_FACTORIAL_FORMULA's docstring for why noise isn't itself a GLM
# term), a curated set of 2D heatmap slices (visual, for the paper's main
# figure, also at the noise=0.20 baseline), and the judge-human alignment-
# bucketed false-positive-rate view (build_ppi_alignment_results_from_
# factorial/save_ppi_alignment_sweep_plot, further down this section), which
# is the one place llm_noise's other 10 levels get used. Reuses
# _run_ppi_comparison_cell/run_ppi_comparison_simulation unchanged -- this
# section is entirely new sources + new analysis, no new execution path.
# ---------------------------------------------------------------------------

_PPI_FACTORIAL_NAME_RE = re.compile(
    r"^fact\.(?P<et>[a-z]+)\.bm=(?P<bm>[a-z]+)\.n=(?P<n>\d+)\.nlab=(?P<nlab>\d+)\.lm=(?P<lm>[a-z_]+)\.es=(?P<es>[a-z]+)\.bd=(?P<bd>[a-z]+)\.noise=(?P<noise>[\d.]+)$"
)


def _parse_ppi_factorial_name(name: str) -> dict:
    m = _PPI_FACTORIAL_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized factorial scenario name: {name!r}")
    d = m.groupdict()
    d["n"] = int(d["n"])
    d["nlab"] = int(d["nlab"])
    d["noise"] = float(d["noise"])
    return d


def _ppi_factorial_dataframe(results: list[PPIComparisonResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        d = _parse_ppi_factorial_name(r.name)
        rows.append({
            **d, "n_reps": r.n_reps,
            "rejects_ppi": r.rejects_ppi, "fails_ppi": r.n_reps - r.rejects_ppi,
            "rate_ppi": r.rejects_ppi / r.n_reps if r.n_reps else float("nan"),
            "rejects_all_human": r.rejects_all_human, "rejects_human_subset": r.rejects_human_subset,
        })
    return pd.DataFrame(rows)


_PPI_FACTORIAL_FORMULA = (
    "rejects_ppi + fails_ppi ~ "
    "C(bm, Treatment('none')) + C(n) + C(nlab) + C(lm, Treatment('mcar')) "
    "+ C(es, Treatment('null')) + C(bd, Treatment('opposing')) "
    "+ C(et, Treatment('continuous')) "
    "+ C(bm, Treatment('none')):C(es, Treatment('null')) "
    "+ C(bd, Treatment('opposing')):C(es, Treatment('null'))"
)
"""Grouped-binomial GLM formula (statsmodels/patsy's "successes + failures ~
..." syntax, the standard encoding for aggregate count data -- equivalent to
a per-replicate logistic regression here since there are no per-replicate
covariates beyond the factors themselves). Main effects for all seven
factors (the original six, plus eval_type now that build_ppi_factorial_
sources crosses continuous/likert), plus two theoretically-motivated 2-way
interactions: bias_magnitude:effect_size (does bias severity change how
power grows with effect size) and bias_direction:effect_size (does the
opposing/reinforcing asymmetry itself depend on effect size). N:N_lab is
deliberately NOT included as an interaction term here despite
build_ppi_nlab_grid_sources' finding that N_lab matters far more than N at
small N_lab -- that finding is a statement about MAGNITUDE (visible
directly in the heatmap), not really a linear-interaction question, and
the seven main effects plus two interactions already leave ample residual
df on ~624 cells (continuous+likert combined); adding more interaction
terms than the sample supports would just widen every coefficient's CI
without adding information. et is likewise a main effect only, not crossed
with the other six factors -- this treats "does the whole 6-factor picture
shift up/down for likert vs. continuous" as the question worth asking here,
not "does every individual factor interact differently with eval_type,"
which would need a fractional design to stay estimable. bm/bd/et are
Treatment-coded at their "no bias"/"opposing"/"continuous" reference levels
so every coefficient reads as "vs. no bias" / "vs. opposing" / "vs.
continuous," matching how the rest of the PPI mode's plots and reports are
already framed.

llm_noise (build_ppi_factorial_sources' 7th factor) is deliberately NOT a
term here, and `results` fed to this function should be pre-filtered to
noise=0.20 (the baseline every non-alignment factorial output already used
before llm_noise joined the source grid) -- adding it as an eighth main
effect would run into a real confound, not just added complexity: llm_noise
only varies away from 0.20 on es="null" cells (see build_ppi_factorial_
sources' docstring), so any non-baseline noise level implies es="null" with
perfect collinearity against the es term already in the formula, making the
two effects statistically inseparable. The full noise-swept es="null"
subset is exactly what feeds the separate alignment-bucketed view instead
(build_ppi_alignment_results_from_factorial), which bypasses this GLM
entirely and reports realized-alignment buckets directly rather than a
fitted noise coefficient."""


def fit_ppi_factorial_model(results: list[PPIComparisonResult]) -> tuple[str, pd.DataFrame]:
    """Fit _PPI_FACTORIAL_FORMULA and return (summary_text, raw dataframe).

    Caveat worth keeping in the write-up: at es="large" (or any stratum
    where the corrected rate saturates to ~0 or ~1 for every level of
    another factor), the GLM can show quasi-complete separation -- huge
    coefficients/standard errors for terms involving that stratum. This
    isn't a fitting bug; it's the correct signal that once power has
    saturated, that stratum carries no further information about which
    factor moved it there. statsmodels still converges and the OTHER
    (non-saturated) coefficients remain informative; treat any
    coefficient with a standard error orders of magnitude larger than its
    neighbors as "this stratum saturated," not as a real, enormous effect.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = _ppi_factorial_dataframe(results)
    fit = smf.glm(formula=_PPI_FACTORIAL_FORMULA, data=df, family=sm.families.Binomial()).fit()
    return fit.summary().as_text(), df


def print_ppi_factorial_report(results: list[PPIComparisonResult], alpha: float, label: str = "paired_t") -> None:
    """Regression summary (fit_ppi_factorial_model) plus two quotable
    headline numbers: the worst observed Type-I inflation (among es="null"
    cells) and the largest all_human-vs-ppi power gap (among non-null
    cells) -- the single-number "worst case across N x N_lab" claims a
    paper would want, pulled directly from the factorial grid rather than
    eyeballed off a table.

    `label` names the estimand(s) `results` was pooled across in the header
    (default "paired_t", the original single-estimand factorial) -- pass
    _COMPARISON_METHODS_LABEL/_COMPARISON_METHODS_OMNIBUS_LABEL for the
    2-group/omnibus pooled reports respectively (see run()'s factorial_check
    block); this function itself is agnostic to which methods `results` was
    pooled across, it just needs a name for the header text."""
    if not results:
        print("\n  (no PPI factorial results)")
        return
    summary_text, df = fit_ppi_factorial_model(results)
    eval_types = sorted(df["et"].unique())
    print(f"\n{'='*96}\n  PVALUES (PPI-CORRECTED) -- FULL FACTORIAL "
          f"(bias_magnitude x N x N_lab x label_mechanism x effect_size x bias_direction x eval_type)\n"
          f"  {len(results)} cells, {'/'.join(eval_types)}/{label}; nominal alpha={alpha}\n{'='*96}\n")
    print(summary_text)

    null_rows = df[df["es"] == "null"]
    if len(null_rows):
        worst = null_rows.loc[(null_rows["rate_ppi"] - alpha).abs().idxmax()]
        print(f"\n  Worst Type-I cell: rate={worst['rate_ppi']:.3f} (nominal alpha={alpha}) at "
              f"et={worst['et']} bm={worst['bm']} n={worst['n']} nlab={worst['nlab']} lm={worst['lm']} bd={worst['bd']}")

    nonnull_rows = df[df["es"] != "null"].copy()
    if len(nonnull_rows):
        nonnull_rows["power_gap"] = (nonnull_rows["rejects_all_human"] - nonnull_rows["rejects_ppi"]) / nonnull_rows["n_reps"]
        worst_gap = nonnull_rows.loc[nonnull_rows["power_gap"].idxmax()]
        print(f"  Largest all_human-vs-ppi power gap: {worst_gap['power_gap']:.3f} at "
              f"et={worst_gap['et']} bm={worst_gap['bm']} n={worst_gap['n']} nlab={worst_gap['nlab']} "
              f"lm={worst_gap['lm']} es={worst_gap['es']} bd={worst_gap['bd']}")
    print()


def save_results_artifacts_ppi_factorial(
    *, results: list[PPIComparisonResult], alpha: float, out_dir: str, run_stem: str,
    pooled_results: list[PPIComparisonResult] | None = None, write_csv: bool = True, label: str = "paired_t",
) -> list[str]:
    """`results` is the RAW (per-method) data, saved verbatim to the CSV
    (unless `write_csv=False` -- see below).

    `pooled_results` (falls back to pooling `results` if omitted) feeds
    the saved .log's GLM fit and headline numbers instead. The GLM
    coefficients themselves are numerically IDENTICAL either way (grouped-
    binomial log-likelihood is additive over rows sharing the same
    covariates), but print_ppi_factorial_report's two "worst cell" headline
    numbers are NOT: they pick the single most extreme row via `idxmax()`,
    and fed the raw per-method rows (4x as many, each nosier at 1/4 the
    pooled n_reps) that max is mechanically more extreme than the properly
    pooled one -- confirmed on a real official run: the raw-fed log claimed
    a "worst Type-I cell" of 0.445 (nominal alpha=0.05) where the correctly
    pooled figure for that same cell was 0.154, and a different "largest
    power gap" cell entirely (0.715 vs. the pooled 0.416). See
    save_results_artifacts_ppi_comparison's docstring for the same
    raw-vs-pooled issue in the other two saved logs.

    `write_csv=False` skips the CSV entirely, writing only the .log -- for a
    SECOND call against the SAME `run_stem` that should append another
    pooled summary (e.g. _COMPARISON_METHODS_OMNIBUS' own report) without
    re-writing (or worse, silently truncating to a different method subset)
    the CSV the first call already wrote for the combined raw data. `label`
    is forwarded to print_ppi_factorial_report's header text -- see its
    own docstring."""
    if pooled_results is None:
        pooled_results = pool_ppi_comparison_across_methods(results)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_factorial_results.csv"
    if write_csv:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "name", "method", "et", "bm", "n", "nlab", "lm", "es", "bd", "n_reps",
                "rate_all_human", "rate_human_subset", "rate_llm_only", "rate_llm_impute", "rate_ppi", "n_failed",
            ])
            for r in results:
                d = _parse_ppi_factorial_name(r.name)
                writer.writerow([
                    r.name, r.method, d["et"], d["bm"], d["n"], d["nlab"], d["lm"], d["es"], d["bd"], r.n_reps,
                    f"{r.rejects_all_human / r.n_reps:.8f}" if r.n_reps else "",
                    f"{r.rejects_human_subset / r.n_reps:.8f}" if r.n_reps else "",
                    f"{r.rejects_llm_only / r.n_reps:.8f}" if r.n_reps else "",
                    f"{r.rejects_llm_impute / r.n_reps:.8f}" if r.n_reps else "",
                    f"{r.rejects_ppi / r.n_reps:.8f}" if r.n_reps else "",
                    r.n_failed,
                ])
        print(f"Saved results: {csv_path}")
    summary_path = out_base / f"{run_stem}_ppi_factorial_summary.log"
    write_mode = "w" if write_csv else "a"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_factorial_report(pooled_results, alpha=alpha, label=label)
    with summary_path.open(write_mode, encoding="utf-8") as handle:
        handle.write(buf.getvalue())
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)] if write_csv else [str(summary_path)]


def save_ppi_factorial_heatmap_plot(*, results: list[PPIComparisonResult], alpha: float, out_path: str) -> str:
    """Three flagship 2D heatmap slices through the 6D factorial cube, each
    fixing the other four factors at a moderate/representative level (bm=
    severe, n=200, nlab=30, lm=mcar, es=moderate, bd=opposing -- the same
    "severe" bias/moderate-effect severity used throughout the rest of
    this file's checks) so a reader can see two factors' effect on the
    PPI-corrected rate at a glance, the same way save_ppi_nlab_grid_plot
    does for N x N_lab alone. One ROW per eval type (build_ppi_factorial_
    sources now crosses continuous/likert), one COLUMN per slice, the same
    row-per-facet/column-per-slice convention save_ppi_nlab_grid_plot uses
    for its own eval-type faceting:
      1. N x N_lab (bm/lm/es/bd fixed) -- reproduces build_ppi_nlab_grid_
         sources' own heatmap as a consistency check, now inside the
         broader factorial's own data.
      2. bias_magnitude x label_mechanism (n/nlab/es/bd fixed) -- does a
         biased labeling PROCESS compound with judge bias severity.
      3. effect_size x bias_direction (n/nlab/bm/lm fixed) -- the
         opposing/reinforcing asymmetry (save_ppi_power_direction_plot) at
         a fixed, moderate bias/label setting instead of the cross-eval-type
         line-plot framing.
    The pooled GLM (fit_ppi_factorial_model) is the rigorous backing for
    what these slices show; these are the visual/narrative complement."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if not results:
        raise ValueError("No PPI factorial results to plot.")
    df = _ppi_factorial_dataframe(results)
    eval_types = sorted(df["et"].unique())

    _CATEGORICAL_FACTORS = ("bm", "lm", "es", "bd")
    slices = [
        ("n", "nlab", dict(bm="severe", lm="mcar", es="moderate", bd="opposing")),
        ("bm", "lm", dict(n=200, nlab=30, es="moderate", bd="opposing")),
        ("es", "bd", dict(n=200, nlab=30, bm="severe", lm="mcar")),
    ]
    order = {
        "bm": ["none", "moderate", "severe"], "lm": ["mcar", "mnar_mild", "mnar_strong"],
        "es": ["null", "moderate", "large"], "bd": ["opposing", "reinforcing"],
        "n": PPI_FACTORIAL_N_VALUES, "nlab": PPI_FACTORIAL_NLAB_VALUES,
    }

    fig, axes = plt.subplots(
        len(eval_types), len(slices), figsize=(6.0 * len(slices), 5.0 * len(eval_types)), squeeze=False,
    )
    for row, et in enumerate(eval_types):
        et_df = df[df["et"] == et]
        for col, (x_field, y_field, fixed) in enumerate(slices):
            ax = axes[row][col]
            sub = et_df
            for k, v in fixed.items():
                sub = sub[sub[k] == v]
            x_values = [v for v in order[x_field] if v in set(sub[x_field])]
            y_values = [v for v in order[y_field] if v in set(sub[y_field])]
            grid = np.full((len(y_values), len(x_values)), np.nan)
            for _, r in sub.iterrows():
                if r["n_reps"] == 0 or r[x_field] not in x_values or r[y_field] not in y_values:
                    continue
                grid[y_values.index(r[y_field]), x_values.index(r[x_field])] = r["rate_ppi"]

            vmax = max(2.0 * alpha, float(np.nanmax(grid)) * 1.1 if np.isfinite(np.nanmax(grid)) else 2.0 * alpha)
            im = ax.imshow(grid, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=alpha, vmax=vmax), aspect="auto")
            for i in range(len(y_values)):
                for j in range(len(x_values)):
                    val = grid[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black",
                            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.0),
                        )
            x_tick_labels = [_pretty_factorial_level(v) if x_field in _CATEGORICAL_FACTORS else str(v) for v in x_values]
            y_tick_labels = [_pretty_factorial_level(v) if y_field in _CATEGORICAL_FACTORS else str(v) for v in y_values]
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels(x_tick_labels, rotation=20 if x_field in _CATEGORICAL_FACTORS else 0)
            ax.set_yticks(range(len(y_values)))
            ax.set_yticklabels(y_tick_labels)
            ax.set_xlabel(_PPI_FACTORIAL_FACTOR_LABELS.get(x_field, x_field))
            ax.set_ylabel(_PPI_FACTORIAL_FACTOR_LABELS.get(y_field, y_field))
            x_name = _PPI_FACTORIAL_FACTOR_LABELS.get(x_field, x_field)
            y_name = _PPI_FACTORIAL_FACTOR_LABELS.get(y_field, y_field)
            fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
            ax.set_title(f"[{et.capitalize()}] {x_name} × {y_name}\n({fixed_str})", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"PPI-Corrected Rejection Rate: Full-Factorial Slices (nominal {_alpha_label(alpha)})", y=1.02, fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PPI mode, alignment view: false-positive rate (uncorrected vs PPI-corrected)
# as a function of REALIZED judge-human alignment -- derived from build_ppi_
# factorial_sources' own es="null" cells (which cross llm_noise x
# bias_magnitude, among the other factors) rather than a separate simulation
# run -- see scenarios.synthetic's build_ppi_factorial_sources/
# measure_judge_alignment for the design and the "percent aligned conflates
# noise and bias" motivation.
# ---------------------------------------------------------------------------

@dataclass
class PPIAlignmentSweepResult:
    name: str
    eval_type: str
    noise: float
    bias_label: str
    """One of PPI_FACTORIAL_BIAS_MAGNITUDES' keys (none/moderate/severe) --
    "none" is this result's REGIME marker (see _alignment_regime): every
    other label means real judge bias was present, regardless of
    magnitude."""
    alignment_metrics: dict[str, float]
    """Raw (not rescaled) alignment metrics from measure_judge_alignment --
    e.g. {"pearson_r": ..., "spearman_r": ...} for continuous, or
    {"weighted_kappa": ..., "spearman_r": ..., "percent_agreement": ...} for
    likert. Which one to bucket/plot by is a presentational choice made by
    callers (see _ALIGNMENT_VIEWS), not baked in here -- e.g. likert gets
    reported/plotted against BOTH weighted_kappa and spearman_r, since some
    work recommends one over the other for Likert-scored judges."""
    n_reps: int
    rejects_llm_only: int
    """Uncorrected false-positive count -- this sweep's es=0.0 throughout, so
    rejects_llm_only/n_reps IS the uncorrected Type-I rate directly (no
    effect-size framing needed the way _COMPARISON_METHODS' other consumers
    have to guard against, e.g. save_ppi_null_comparison_plot's docstring)."""
    rejects_ppi: int
    n_failed: int


def _alignment_regime(bias_label: str) -> str:
    """The qualitative "why is this cell at this alignment level" split the
    whole sweep is built to make visible: "no_bias" (bias_label == "none" --
    whatever alignment level this cell landed at came purely from llm_noise)
    vs. "bias_present" (any nonzero bias_delta, regardless of magnitude --
    lumped into one regime rather than none/mild/moderate/severe/extreme
    sub-bins, since the point is the qualitative presence of bias, not its
    size, and finer sub-bins would fragment already-sparse per-bucket cell
    counts further)."""
    return "no_bias" if bias_label == "none" else "bias_present"


_PPI_ALIGNMENT_REGIME_LABEL = {"no_bias": "no judge bias (noise only)", "bias_present": "judge bias present"}
_PPI_ALIGNMENT_BUCKET_WIDTH = 10


def _metric_pct(raw_value: float) -> float:
    """Rescale a correlation-or-kappa-like alignment metric (nominally
    -1..1, essentially always >= 0 for a judge at least weakly related to
    truth) to a 0-100 bucketing percentage, clipping at 0 on the rare
    below-chance/negative draw."""
    return float(np.clip(raw_value, 0.0, 1.0) * 100.0)


def _alignment_bucket(pct: float, width: int = _PPI_ALIGNMENT_BUCKET_WIDTH) -> tuple[int, str]:
    """(bucket_lo, label) for a 0-100 alignment percentage, in `width`-point
    buckets -- e.g. pct=73.2 -> (70, "70-80%") at width=10. Clamps into
    [0, 100-width] first so a pct of exactly 100.0 lands in the last bucket
    instead of spilling into a width-0 "100-110%" one."""
    lo = int(pct // width) * width
    lo = max(0, min(lo, 100 - width))
    return lo, f"{lo}-{lo + width}%"


def _kappa_band(x: float) -> str:
    """Landis & Koch (1977) benchmarks for kappa-type statistics -- same
    bands evalstats.alignment._interpret_kappa uses for the public alignment
    report, reused here so a bucket's qualitative label matches what a user
    would see calling validate_alignment() on the same kind of judge."""
    if x < 0:
        return "poor"
    if x <= 0.20:
        return "slight"
    if x <= 0.40:
        return "fair"
    if x <= 0.60:
        return "moderate"
    if x <= 0.80:
        return "substantial"
    return "almost perfect"


def _corr_band(x: float) -> str:
    """Cohen (1988) conventions for correlation-coefficient magnitude -- same
    bands evalstats.alignment._interpret_corr uses."""
    a = abs(x)
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "medium"
    return "large"


_ALIGNMENT_VIEWS = [
    ("continuous", "pearson_r", "Pearson r", _corr_band, "Cohen, 1988", "r"),
    ("likert", "weighted_kappa", "weighted κ", _kappa_band, "Landis & Koch, 1977", "κ"),
    ("likert", "spearman_r", "Spearman r", _corr_band, "Cohen, 1988", "ρ"),
]
"""The three (eval_type, metric, display_label, qualitative-band function,
citation, symbol) views the alignment sweep reports/plots -- one per
eval_type for the metric most commonly reported for that data type in
practice (Pearson r for continuous, weighted Cohen's kappa for likert),
PLUS a second view of likert bucketed by Spearman r, since some work
recommends rank correlation over weighted kappa for Likert-scored judges
(it doesn't require picking a tie-weighting scheme -- though empirically,
Spearman turned out MORE prone to masking a biased judge than weighted
kappa is, not less: at "large"/"almost perfect" alignment, Spearman's
bucket showed materially higher uncorrected false-positive rates than
kappa's -- both being rank/order-based to some degree, but kappa's
near-exact-match requirement is more bias-sensitive than pure rank
preservation is). `symbol` is the conventional single-character notation
used in bucket subplot titles (e.g. "κ=0.40-0.50"). Drives
print_ppi_alignment_sweep_report/save_ppi_alignment_sweep_plot/the
human-human companion uniformly -- one call per entry -- so all three stay
in sync and none can silently drift out of step with the others."""


def build_ppi_alignment_results_from_factorial(
    factorial_sources: list[JudgeBiasSource], factorial_results: list[PPIComparisonResult],
    n_align_mc: int, seed: int = 0,
) -> list[PPIAlignmentSweepResult]:
    """Derives the judge-human alignment-bucketed view (_ALIGNMENT_VIEWS) from
    build_ppi_factorial_sources' own es="null" cells, rather than a separate
    simulation run against a separate, narrower source grid -- `factorial_results`
    should be the FULL pooled-across-_COMPARISON_METHODS factorial results,
    covering every llm_noise level (not the noise=0.20-only subset fed to
    fit_ppi_factorial_model/save_ppi_factorial_heatmap_plot -- see
    _PPI_FACTORIAL_FORMULA's docstring for why those two views need disjoint
    slices of the same data).

    Realized alignment (measure_judge_alignment) depends only on
    (eval_type, llm_noise, bias_delta, likert_max) -- not on N, N_lab,
    label_mechanism, or bias_direction (bias_direction is moot here anyway:
    build_ppi_factorial_sources skips bias_direction="reinforcing" whenever
    es="null") -- so this memoizes that measurement across the handful of
    distinct (eval_type, llm_noise, bias_delta, likert_max) combinations the
    null-effect subset actually contains (2 eval types x 11 noise levels x 3
    bias magnitudes = 66, at the default noise grid), instead of recomputing
    an identical large-sample calibration draw once per factorial cell
    (~1,584 of them at the default grid). This is also what gives each
    alignment bucket its richer N/N_lab/label_mechanism spread versus the
    original standalone sweep's one-baseline-value-each design: every
    es="null" cell sharing a (eval_type, llm_noise, bias_delta, likert_max)
    combo lands in the SAME bucket regardless of its own N/N_lab/
    label_mechanism, so a bucket now pools rejection counts across whichever
    of those combinations survive build_ppi_factorial_sources' n_lab>=n skip
    at that noise/bias/eval_type slice."""
    by_name = {sc.name: sc for sc in factorial_sources}
    align_cache: dict[tuple, dict] = {}
    results: list[PPIAlignmentSweepResult] = []
    for r in factorial_results:
        d = _parse_ppi_factorial_name(r.name)
        if d["es"] != "null":
            continue
        sc = by_name[r.name]
        key = (sc.eval_type, round(sc.llm_noise, 8), round(sc.bias_delta, 8), sc.likert_max)
        if key not in align_cache:
            align_cache[key] = measure_judge_alignment(sc, n_mc=n_align_mc, seed=seed + len(align_cache))
        results.append(PPIAlignmentSweepResult(
            name=r.name, eval_type=d["et"], noise=d["noise"], bias_label=d["bm"],
            alignment_metrics=align_cache[key],
            n_reps=r.n_reps, rejects_llm_only=r.rejects_llm_only, rejects_ppi=r.rejects_ppi,
            n_failed=r.n_failed,
        ))
    return results


def print_ppi_alignment_sweep_report(results: list[PPIAlignmentSweepResult], alpha: float) -> None:
    """One table per _ALIGNMENT_VIEWS entry (uncorrected/PPI-corrected
    false-positive rate by alignment bucket x regime) -- the console/log-file
    counterpart to save_ppi_alignment_sweep_plot's bar charts, in text form.
    Each bucket's row also prints that bucket's qualitative interpretation
    band (Landis & Koch for kappa, Cohen for correlations), evaluated at the
    bucket's midpoint."""
    if not results:
        print("\n  (no PPI alignment-sweep results)")
        return
    print(f"\n{'='*96}\n  PVALUES (PPI-CORRECTED) -- ALIGNMENT SWEEP "
          f"(false-positive rate vs. realized judge-human alignment)\n"
          f"  {len(results)} cells, {len({r.eval_type for r in results})} eval type(s); nominal alpha={alpha}\n{'='*96}\n\n"
          f"  READ THE WITHIN-BUCKET COMPARISON, not the across-bucket trend: within the 'bias present' regime,\n"
          f"  alignment here is driven almost entirely by judge NOISE, not bias (a pure additive bias barely\n"
          f"  moves these metrics) -- so higher buckets mostly mean lower noise, and lower noise makes the SAME\n"
          f"  fixed bias easier to detect, which is why 'bias present' rows can rise across buckets. That's not\n"
          f"  alignment causing miscalibration -- see measure_judge_alignment's docstring for the full mechanism.")
    for et, metric, display, band_fn, band_source, _symbol in _ALIGNMENT_VIEWS:
        et_rows = [r for r in results if r.eval_type == et and metric in r.alignment_metrics]
        if not et_rows:
            continue
        print(f"\n  [{et}, bucketed by {display} ({band_source} bands)]")
        print(f"    {'bucket':<10} {'band':<16} {'regime':<22} {'n_cells':>7} {'uncorrected':>12} {'ppi-corrected':>14}")
        buckets = sorted({_alignment_bucket(_metric_pct(r.alignment_metrics[metric])) for r in et_rows})
        for lo, label in buckets:
            band = band_fn((lo + _PPI_ALIGNMENT_BUCKET_WIDTH / 2) / 100.0)
            for regime in ("no_bias", "bias_present"):
                cells = [
                    r for r in et_rows
                    if _alignment_bucket(_metric_pct(r.alignment_metrics[metric])) == (lo, label)
                    and _alignment_regime(r.bias_label) == regime
                ]
                if not cells:
                    continue
                n_reps_tot = sum(c.n_reps for c in cells)
                unc_rate = sum(c.rejects_llm_only for c in cells) / n_reps_tot if n_reps_tot else float("nan")
                ppi_rate = sum(c.rejects_ppi for c in cells) / n_reps_tot if n_reps_tot else float("nan")
                print(f"    {label:<10} {band:<16} {_PPI_ALIGNMENT_REGIME_LABEL[regime]:<22} {len(cells):>7d} "
                      f"{unc_rate:>12.3f} {ppi_rate:>14.3f}")
    print()


def save_results_artifacts_ppi_alignment_sweep(
    *, results: list[PPIAlignmentSweepResult], alpha: float, out_dir: str, run_stem: str,
    human_human_rows: list[dict] | None = None,
) -> list[str]:
    """`human_human_rows` (run_human_human_alignment_sweep's output), if
    given, is appended as a trailer section to the SAME .log file -- see
    print_human_human_alignment_report's docstring for why it isn't its own
    section header. The CSV has one column per possible metric (blank where
    an eval type doesn't compute it) rather than a single "primary" column,
    so the raw data supports re-bucketing by any metric later without
    rerunning the simulation."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    metric_cols = ["pearson_r", "spearman_r", "weighted_kappa", "percent_agreement"]
    csv_path = out_base / f"{run_stem}_ppi_alignment_sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "eval_type", "noise", "bias_label", "regime", *metric_cols,
            "n_reps", "rate_llm_only", "rate_ppi", "n_failed",
        ])
        for r in results:
            writer.writerow([
                r.name, r.eval_type, f"{r.noise:.4f}", r.bias_label, _alignment_regime(r.bias_label),
                *[f"{r.alignment_metrics[c]:.4f}" if c in r.alignment_metrics else "" for c in metric_cols],
                r.n_reps,
                f"{r.rejects_llm_only / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.rejects_ppi / r.n_reps:.8f}" if r.n_reps else "",
                r.n_failed,
            ])
    summary_path = out_base / f"{run_stem}_ppi_alignment_sweep_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_alignment_sweep_report(results, alpha=alpha)
        if human_human_rows:
            print_human_human_alignment_report(human_human_rows)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    output_paths = [str(csv_path), str(summary_path)]
    if human_human_rows:
        output_paths.append(save_human_human_alignment_csv(rows=human_human_rows, out_dir=out_dir, run_stem=run_stem))
    return output_paths


def save_ppi_alignment_sweep_plot(
    *, results: list[PPIAlignmentSweepResult], eval_type: str, metric: str, display_label: str,
    band_fn, band_source: str, symbol: str, alpha: float, out_path: str,
) -> str:
    """Array of bar charts for ONE (eval_type, metric) view from
    _ALIGNMENT_VIEWS -- one COLUMN per alignment bucket present, each panel
    ALWAYS showing both (no_bias, bias_present) regimes x (uncorrected,
    PPI-corrected) arms (4 bars total), even when one regime has zero cells
    in that bucket -- drawn as a zero-height bar with "(n=0)" in its tick
    label, rather than omitted, so every panel has the SAME x-axis layout
    and bar width instead of the layout stretching/shrinking per panel based
    on which regimes happen to have data (visually misleading side by side).
    Every panel also shares the same fixed y-axis (0-1.05) for the same
    reason -- direct visual comparability across buckets AND across the
    other _ALIGNMENT_VIEWS figures this is called for.

    Each bucket's title is the metric's own notation over its range (e.g.
    "κ=0.40-0.50"), with its qualitative interpretation band underneath
    (band_fn, evaluated at the bucket midpoint -- e.g. "substantial" per
    Landis & Koch, 1977) -- publication-style notation rather than a raw
    percentage, so a reader isn't left to separately look up what the number
    means for this metric.

    Called once per _ALIGNMENT_VIEWS entry (see run()'s factorial_check
    block) -- deliberately separate figures rather than one combined grid,
    since continuous/likert use different metrics with different natural
    ranges and interpretation bands, and likert gets shown against two
    different metrics that deserve their own titles rather than sharing one.

    Error bars are the 95% Wilson score interval on each bar's pooled
    rejects/n_reps (same convention/caveat as save_ppi_null_comparison_plot:
    exact for a truly homogeneous pool, a standard mild simplification if the
    (noise, bias) cells landing in the same bucket/regime aren't perfectly
    identically calibrated). The WITHIN-bucket-vs-across-bucket reading
    caveat (see measure_judge_alignment's docstring) is deliberately left out
    of the figure itself -- that belongs in the paper's caption/prose, not
    baked into the image."""
    import matplotlib.pyplot as plt

    et_rows = [r for r in results if r.eval_type == eval_type and metric in r.alignment_metrics]
    if not et_rows:
        raise ValueError(f"No PPI alignment-sweep results for eval_type={eval_type!r}, metric={metric!r}.")
    bar_width = 0.35
    group_gap = 0.25
    regimes = ("no_bias", "bias_present")
    arm_colors = {"llm_only": "#e7298a", "ppi": "#FFD400"}
    arm_edgecolors = {"llm_only": "none", "ppi": "#8a6d00"}
    arm_labels = {"llm_only": "uncorrected", "ppi": "PPI-corrected"}

    buckets = sorted({_alignment_bucket(_metric_pct(r.alignment_metrics[metric])) for r in et_rows})

    fig, axes = plt.subplots(1, len(buckets), figsize=(2.9 * len(buckets), 4.3), squeeze=False, sharey=True)
    for col, (lo, label) in enumerate(buckets):
        ax = axes[0][col]
        ax.axhline(
            alpha, color="black", ls="--", lw=1.0, alpha=0.6, zorder=1,
            label="nominal α" if col == 0 else None,
        )
        xticks, xticklabels = [], []
        for gi, regime in enumerate(regimes):  # ALWAYS both, even if empty
            cells = [
                r for r in et_rows
                if _alignment_bucket(_metric_pct(r.alignment_metrics[metric])) == (lo, label)
                and _alignment_regime(r.bias_label) == regime
            ]
            n_reps_tot = sum(c.n_reps for c in cells)
            for ai, arm in enumerate(("llm_only", "ppi")):
                x = gi * (2 * bar_width + group_gap) + ai * bar_width
                if n_reps_tot == 0:
                    continue  # nothing to draw; tick/slot still allocated below
                rejects_tot = sum(getattr(c, f"rejects_{arm}") for c in cells)
                rate = rejects_tot / n_reps_tot
                lo_ci, hi_ci = _ppi_wilson_interval(rejects_tot, n_reps_tot)
                ax.bar(
                    x, rate, width=bar_width, color=arm_colors[arm], edgecolor=arm_edgecolors[arm],
                    linewidth=1.0, zorder=2,
                    label=arm_labels[arm] if (col == 0 and gi == 0) else None,
                )
                ax.errorbar(
                    x, rate, yerr=[[max(0.0, rate - lo_ci)], [max(0.0, hi_ci - rate)]],
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=4,
                )
            mid = gi * (2 * bar_width + group_gap) + bar_width / 2
            xticks.append(mid)
            xticklabels.append(f"{_PPI_ALIGNMENT_REGIME_LABEL[regime]}\n(n={len(cells)})")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_xlim(-0.3, (2 * bar_width + group_gap) * len(regimes) - group_gap + 0.05)
        ax.set_ylim(0.0, 1.05)
        band = band_fn((lo + _PPI_ALIGNMENT_BUCKET_WIDTH / 2) / 100.0)
        ax.set_title(f"{symbol}={lo / 100:.2f}-{(lo + _PPI_ALIGNMENT_BUCKET_WIDTH) / 100:.2f}\n({band})", fontsize=10)
        if col == 0:
            ax.set_ylabel("False positive rate", fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.8, zorder=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=True)
    fig.suptitle(f"{eval_type.capitalize()}: False-Positive Rate by Judge-Human Alignment ({display_label})", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_human_human_alignment_sweep(n_mc: int = 20_000, seed: int = 42) -> list[dict]:
    """Companion measurement (not a hypothesis-test sweep -- see
    scenarios.synthetic.measure_human_human_alignment's docstring): for each
    eval_type x PPI_ALIGNMENT_HUMAN_NOISE_LEVELS combination, the realized
    alignment (every metric measure_human_human_alignment computes) between
    two independently-noisy synthetic human raters. Used as context alongside
    the main alignment sweep's judge-vs-human buckets, not merged into the
    same plot -- see save_human_human_alignment_plot."""
    rows = []
    for et in ("continuous", "likert"):
        for i, hn in enumerate(PPI_ALIGNMENT_HUMAN_NOISE_LEVELS):
            m = measure_human_human_alignment(et, hn, n_mc=n_mc, seed=seed + i)
            rows.append({"eval_type": et, "human_noise": hn, "metrics": m})
    return rows


def print_human_human_alignment_report(rows: list[dict]) -> None:
    """Text counterpart to save_human_human_alignment_plot -- printed as a
    trailer to print_ppi_alignment_sweep_report's own log, not a separate
    section header, since it's context for reading that report's buckets,
    not an independent result. One line per _ALIGNMENT_VIEWS entry, matching
    the main report's per-view breakdown."""
    if not rows:
        return
    print("  -- Human-human alignment range (context, NOT a claimed ceiling) --")
    for et, metric, display, _band_fn, _src, _symbol in _ALIGNMENT_VIEWS:
        et_rows = sorted(
            [r for r in rows if r["eval_type"] == et and metric in r["metrics"]], key=lambda r: r["human_noise"],
        )
        if not et_rows:
            continue
        vals = ", ".join(f"noise={r['human_noise']:.2f}: {_metric_pct(r['metrics'][metric]):.0f}%" for r in et_rows)
        print(f"    [{et}, {display}] {vals}")
    print()


def save_human_human_alignment_csv(*, rows: list[dict], out_dir: str, run_stem: str) -> str:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    metric_cols = ["pearson_r", "spearman_r", "weighted_kappa", "percent_agreement"]
    csv_path = out_base / f"{run_stem}_human_human_alignment.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_type", "human_noise", *metric_cols])
        for r in rows:
            writer.writerow([
                r["eval_type"], f"{r['human_noise']:.4f}",
                *[f"{r['metrics'][c]:.4f}" if c in r["metrics"] else "" for c in metric_cols],
            ])
    print(f"Saved results: {csv_path}")
    return str(csv_path)


def save_human_human_alignment_plot(*, rows: list[dict], out_path: str) -> str:
    """Small companion figure: realized human-human alignment (%) across
    PPI_ALIGNMENT_HUMAN_NOISE_LEVELS, one panel per _ALIGNMENT_VIEWS entry
    (matching the main sweep's three views) -- a rough benchmark range for
    reading the main alignment sweep's buckets against (a judge landing well
    below where two independent humans typically land with each other is a
    materially different finding than one landing within that range).
    Deliberately a RANGE across several human_noise values, not one
    bar/number -- there's no canonical "true" human-human noise level to
    assert here, and presenting a single anchored value would repeat the
    same overfitting-to-one-number problem already avoided in the main
    sweep's design."""
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No human-human alignment rows to plot.")
    views = [
        (et, metric, display) for et, metric, display, _bf, _src, _symbol in _ALIGNMENT_VIEWS
        if any(r["eval_type"] == et and metric in r["metrics"] for r in rows)
    ]
    fig, axes = plt.subplots(1, len(views), figsize=(3.6 * len(views), 3.6), squeeze=False)
    for col, (et, metric, display) in enumerate(views):
        ax = axes[0][col]
        et_rows = sorted(
            [r for r in rows if r["eval_type"] == et and metric in r["metrics"]], key=lambda r: r["human_noise"],
        )
        x = np.arange(len(et_rows))
        pcts = [_metric_pct(r["metrics"][metric]) for r in et_rows]
        ax.bar(x, pcts, width=0.6, color="#4d4d4d", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"noise={r['human_noise']:.2f}" for r in et_rows], fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(f"{et.capitalize()} ({display})", fontsize=10)
        ax.set_ylabel("Alignment %" if col == 0 else "")
        ax.grid(axis="y", alpha=0.25, lw=0.8, zorder=0)
        for xi, pct in zip(x, pcts):
            ax.text(xi, pct + 1.5, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "Human-Human Alignment Range (two independently-noisy synthetic raters)\n"
        "context for the judge-alignment sweep's buckets -- not a claimed ceiling",
        fontsize=11,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


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


# ---------------------------------------------------------------------------
# Publication-facing label helpers for PPI mode's plots. Method.name
# (snake_case test identifiers) is fine for CSV columns and code but
# unreadable dropped cold into a figure, so every save_ppi_*_plot function
# below routes test names through _pretty_test. The factorial sweep's
# axis/tick labels go through _PPI_FACTORIAL_FACTOR_LABELS/
# _pretty_factorial_level the same way -- but its per-panel "what's held
# fixed" annotation deliberately stays in short raw-code form (bm=severe,
# not "Bias magnitude = Severe"): spelled out, four of those wrap across
# several lines and read as MORE cluttered, not less: the terse form
# assumes the paper's caption/main text defines what bm/n/nlab/lm/es/bd
# mean once, which is the same assumption the codes bm/lm/es/bd already
# require in cases/pvalues.py itself.
# ---------------------------------------------------------------------------

_PPI_PRETTY_TEST_NAMES: dict[str, str] = {
    TTEST.name: "t-test", TTEST_WELCH.name: "Welch's t-test", MWU_CORR.name: "Mann-Whitney U (corrected)",
    MW_NAIVE.name: "Simple MWU",
    WILCOXON.name: "Wilcoxon", PAIRED_T.name: "Paired t-test", BAYES_BOOTSTRAP.name: "Bayes bootstrap",
    BOOTSTRAP_T.name: "Bootstrap-t", TANGO.name: "Tango score", ANOVA_IND.name: "ANOVA (indep.)",
    ANOVA_REP.name: "ANOVA (repeated)", FRIEDMAN.name: "Friedman",
    KRUSKAL_CORR.name: "Kruskal-Wallis (corrected)", KRUSKAL_NAIVE.name: "Kruskal-Wallis (naive)",
    LMM.name: "LMM", LMM_FACTORIAL.name: "LMM (factorial)", LMM_RUNS.name: "LMM (nested runs)",
}


def _pretty_test(name: str) -> str:
    return _PPI_PRETTY_TEST_NAMES.get(name, name)


_PPI_FACTORIAL_FACTOR_LABELS: dict[str, str] = {
    "bm": "Bias magnitude", "n": "N (total items)", "nlab": "N_lab (labeled items)",
    "lm": "Label mechanism", "es": "Effect size", "bd": "Bias direction",
}
_PPI_FACTORIAL_LEVEL_LABELS: dict[str, str] = {
    "none": "None", "moderate": "Moderate", "severe": "Severe",
    "mcar": "MCAR", "mnar_mild": "MNAR (mild)", "mnar_strong": "MNAR (strong)",
    "null": "Null", "large": "Large", "opposing": "Opposing", "reinforcing": "Reinforcing",
}


def _pretty_factorial_level(value) -> str:
    return _PPI_FACTORIAL_LEVEL_LABELS.get(str(value), str(value))


def _alpha_label(alpha: float) -> str:
    return f"α = {alpha:g}"


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
            label="Uncorrected (any test)" if not unc_label_added else None, zorder=1,
        )
        unc_label_added = True

        keep_c = np.isfinite(rates_c)
        x_c = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_c)))
        ax.scatter(x_c, rates_c[keep_c], s=20, alpha=0.65, color=get_method_color(t), label=_pretty_test(t), zorder=2)

    ax.axhline(alpha, color="black", ls="--", lw=1.1, label=f"Nominal {_alpha_label(alpha)}")
    ax.set_xlim(-0.5, len(tests) - 0.5)
    scatter_max = np.nanmax(np.concatenate(all_rates)) if all_rates else float("nan")
    if not np.isfinite(scatter_max):
        scatter_max = 0.2
    ax.set_ylim(0.0, max(0.2, float(scatter_max) * 1.05))
    ax.set_xticks(np.arange(len(tests)))
    ax.set_xticklabels([_pretty_test(t) for t in tests], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Observed rejection rate")
    ax.set_xlabel("Test")
    title_suffix = " -- Bootstrap/CI-Based Methods" if nonstandard else ""
    ax.set_title(f"PPI-Corrected Type-I Error, by Test{title_suffix}\n(each point: one judge-bias scenario)", fontsize=12)
    ax.grid(axis="y", alpha=0.25, lw=0.8)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


_PPI_POWER_NAME_RE = re.compile(r"^[a-z]+\.([a-z]+)\.es=([\d.]+)$")
"""Matches every power-family scenario name regardless of prefix -- "power."
(build_ppi_power_sources, bias opposing the effect), "powerrf."
(build_ppi_power_reinforcing_sources, bias reinforcing the effect), and
"powernb." (build_ppi_power_nobias_sources, bias_type="none") all share the
same "<prefix>.<eval_type>.es=<frac>" shape, so one parser/report/plot
pipeline serves all three variants."""


def _parse_ppi_power_name(name: str) -> tuple[str, float]:
    m = _PPI_POWER_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized power scenario name: {name!r}")
    return m.group(1), float(m.group(2))


def print_ppi_power_report(results: list[PPIResult], alpha: float, header: str = "POWER UNDER JUDGE BIAS") -> None:
    """Corrected vs. uncorrected rejection rate (POWER, not Type-I) as a
    real effect_size grows, per eval type -- the complement to
    print_ppi_report's null-only Type-I table (build_judge_bias_sources
    never sets effect_size above 0, so that table can only show whether
    PPI correction controls false positives, never whether it retains the
    power to detect a genuine difference under the SAME bias severity).
    es=0.00 doubles as a Type-I cross-check against build_judge_bias_sources'
    'eval_type.*' scenarios (same settings -- should read ~alpha here too).
    ``header`` distinguishes the bias-direction/no-bias variants (see
    _PPI_POWER_NAME_RE) when this same function is reused for them."""
    if not results:
        print("\n  (no PPI power results)")
        return
    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    parsed = {r.name: _parse_ppi_power_name(r.name) for r in results}
    eval_types = sorted({et for et, _ in parsed.values()})
    es_values = sorted({es for _, es in parsed.values()})

    print(f"\n{'='*88}\n  PVALUES (PPI-CORRECTED) -- {header}\n"
          f"  Same bias severity as build_judge_bias_sources' eval_type.* baseline; nominal alpha={alpha}\n"
          f"  es=0.00 column is a Type-I cross-check (should read ~alpha)\n{'='*88}")

    for et in eval_types:
        print(f"\n  [{et}]")
        hdr = f"    {'Test':<14}" + "".join(f"    c({es:.2f})".rjust(11) + f"  u({es:.2f})".rjust(11) for es in es_values)
        print(hdr)
        for t in tests:
            t_rows = [r for r in results if r.test == t]
            if not any(parsed[r.name][0] == et for r in t_rows):
                continue
            row = f"    {t:<14}"
            for es in es_values:
                cell_rows = [r for r in t_rows if parsed[r.name] == (et, es)]
                c_tot = sum(r.corrected_rejects for r in cell_rows)
                u_tot = sum(r.uncorrected_rejects for r in cell_rows)
                n_tot = sum(r.n_reps for r in cell_rows)
                rc = c_tot / n_tot if n_tot > 0 else float("nan")
                ru = u_tot / n_tot if n_tot > 0 else float("nan")
                row += f"  {rc:>9.3f}  {ru:>9.3f}"
            print(row)
    print()


def save_results_artifacts_ppi_power(*, results: list[PPIResult], alpha: float, out_dir: str, run_stem: str) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_power_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "tag", "eval_type", "effect_size", "n", "test", "n_reps",
            "corrected_rejects", "uncorrected_rejects", "n_failed", "corrected_rate", "uncorrected_rate",
        ])
        for r in results:
            et, es = _parse_ppi_power_name(r.name)
            writer.writerow([
                r.name, r.tag, et, f"{es:.4f}", r.n, r.test, r.n_reps, r.corrected_rejects, r.uncorrected_rejects, r.n_failed,
                f"{r.corrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.uncorrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
            ])
    summary_path = out_base / f"{run_stem}_ppi_power_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_power_report(results, alpha=alpha)
    summary_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_power_plot(
    *, results: list[PPIResult], alpha: float, out_path: str, title_suffix: str = "",
) -> str:
    """Power curve (rejection rate vs. real effect_size) -- TWO rows (top:
    corrected, bottom: uncorrected), one column per eval type, rather than
    overlaying both in one set of axes: with up to 13 tests' worth of
    same-colored solid+dashed lines sharing one plot, superimposing
    corrected and uncorrected became unreadable. Uncorrected keeps its
    dashed linestyle in its own row, consistent with save_ppi_typeI_plot/
    save_ppi_power_direction_plot's convention.

    (An earlier version of this function also accepted an ``ideal_results``
    list -- build_ppi_power_nobias_sources' results, overlaid as a dotted
    "ideal" reference line on the corrected row. Removed on request as
    unneeded clutter; run() still computes the no-bias check before the
    main power plot, since it also feeds its own standalone
    ``..._power_vs_effect_size_nobias.png`` plot via a separate call.)"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not results:
        raise ValueError("No PPI power results to plot.")
    tests = _ppi_tests_present(results, nonstandard=False)
    parsed = {r.name: _parse_ppi_power_name(r.name) for r in results}
    eval_types = sorted({et for et, _ in parsed.values()})
    es_values = sorted({es for _, es in parsed.values()})

    fig, axes = plt.subplots(2, len(eval_types), figsize=(4.6 * len(eval_types), 7.6), squeeze=False)
    for col, et in enumerate(eval_types):
        ax_c, ax_u = axes[0][col], axes[1][col]
        ax_c.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6)
        ax_u.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6)
        for t in tests:
            t_rows = [r for r in results if r.test == t]
            ys_c, ys_u = [], []
            for es in es_values:
                cell_rows = [r for r in t_rows if parsed[r.name] == (et, es)]
                c_tot = sum(r.corrected_rejects for r in cell_rows)
                u_tot = sum(r.uncorrected_rejects for r in cell_rows)
                n_tot = sum(r.n_reps for r in cell_rows)
                ys_c.append(c_tot / n_tot if n_tot > 0 else float("nan"))
                ys_u.append(u_tot / n_tot if n_tot > 0 else float("nan"))
            if not any(np.isfinite(ys_c)):
                continue
            color = get_method_color(t)
            ax_c.plot(es_values, ys_c, marker="o", color=color, linewidth=1.6, markersize=4, label=_pretty_test(t), zorder=2)
            ax_u.plot(es_values, ys_u, marker="x", color=color, linewidth=1.4, linestyle="--", markersize=4, zorder=2)

        ax_c.set_title(et.capitalize())
        ax_c.set_ylabel("Rejection rate\n(corrected)" if col == 0 else "")
        ax_c.set_ylim(-0.02, 1.02)
        ax_c.set_xticklabels([])

        ax_u.set_xlabel("Effect size")
        ax_u.set_ylabel("Rejection rate\n(uncorrected)" if col == 0 else "")
        ax_u.set_ylim(-0.02, 1.02)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", alpha=0.6))
    labels.append(f"Nominal {_alpha_label(alpha)}")
    fig.legend(handles, labels, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.5)
    fig.suptitle(f"PPI-Corrected Power vs. Effect Size{title_suffix}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 0.83, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ppi_power_direction_plot(
    *, opposing: list[PPIResult], reinforcing: list[PPIResult], alpha: float, out_path: str,
) -> str:
    """Power curve comparison: judge bias OPPOSING the injected real effect
    (build_ppi_power_sources) vs. REINFORCING it (build_ppi_power_reinforcing_
    sources) -- one row per direction, one column per eval type. Opposing
    bias produces the "cancellation dip" visible in save_ppi_power_plot's
    uncorrected line (bias and effect partially cancel as effect_size
    grows). Reinforcing bias instead pushes the uncorrected line ABOVE the
    corrected one with no dip at all -- arguably the more dangerous failure
    mode in practice, since nothing about the SHAPE of the uncorrected curve
    alone would flag it as wrong; it just quietly overstates the effect."""
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D

    row_titles = {"opposing": "Bias Opposes True Effect", "reinforcing": "Bias Reinforces True Effect"}
    rows = [(label, res) for label, res in [("opposing", opposing), ("reinforcing", reinforcing)] if res]
    if not rows:
        raise ValueError("No PPI power-direction results to plot.")
    all_results = [r for _, res in rows for r in res]
    tests = _ppi_tests_present(all_results, nonstandard=False)
    eval_types = sorted({_parse_ppi_power_name(r.name)[0] for r in all_results})

    fig, axes = plt.subplots(
        len(rows), len(eval_types), figsize=(5.2 * len(eval_types), 4.0 * len(rows)), squeeze=False,
    )
    for row_idx, (row_label, res) in enumerate(rows):
        parsed = {r.name: _parse_ppi_power_name(r.name) for r in res}
        es_values = sorted({es for _, es in parsed.values()})
        for col, et in enumerate(eval_types):
            ax = axes[row_idx][col]
            ax.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6)
            for t in tests:
                t_rows = [r for r in res if r.test == t]
                ys_c, ys_u = [], []
                for es in es_values:
                    cell_rows = [r for r in t_rows if parsed[r.name] == (et, es)]
                    c_tot = sum(r.corrected_rejects for r in cell_rows)
                    u_tot = sum(r.uncorrected_rejects for r in cell_rows)
                    n_tot = sum(r.n_reps for r in cell_rows)
                    ys_c.append(c_tot / n_tot if n_tot > 0 else float("nan"))
                    ys_u.append(u_tot / n_tot if n_tot > 0 else float("nan"))
                if not any(np.isfinite(ys_c)):
                    continue
                color = get_method_color(t)
                ax.plot(
                    es_values, ys_c, marker="o", color=color, linewidth=1.6,
                    label=_pretty_test(t) if row_idx == 0 else None, zorder=2,
                )
                ax.plot(es_values, ys_u, marker="x", color=color, linewidth=1.0, linestyle="--", alpha=0.5, zorder=1)
            if row_idx == 0:
                ax.set_title(et.capitalize())
            if col == 0:
                ax.set_ylabel(f"{row_titles.get(row_label, row_label)}\nRejection rate")
            ax.set_xlabel("Effect size")
            ax.set_ylim(-0.02, 1.02)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="#333333", marker="o", linewidth=1.6, linestyle="-"),
        Line2D([0], [0], color="#333333", marker="x", linewidth=1.0, linestyle="--", alpha=0.7),
        Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", alpha=0.6),
    ]
    labels += ["Corrected", "Uncorrected", f"Nominal {_alpha_label(alpha)}"]
    axes[0][0].legend(handles, labels, fontsize=7, loc="lower right", ncol=2)
    fig.suptitle("PPI-Corrected Power: Bias Opposing vs. Reinforcing the True Effect", y=1.03, fontsize=12)
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
        ax1.scatter(x[keep_z], z[keep_z], s=22, alpha=0.7, color=color, label=_pretty_test(t))

        cov = np.array([r.coverage for r in t_rows])
        keep_c = np.isfinite(cov)
        ax2.scatter(x[keep_c], cov[keep_c], s=22, alpha=0.7, color=color)

        wid = np.array([r.mean_ci_width for r in t_rows])
        keep_w = np.isfinite(wid)
        ax3.scatter(x[keep_w], wid[keep_w], s=22, alpha=0.7, color=color)

    ax1.axhline(0.0, color="black", ls="--", lw=1.0)
    ax1.axhline(3.0, color="red", ls=":", lw=0.9, label="|z| = 3 (flagged)")
    ax1.axhline(-3.0, color="red", ls=":", lw=0.9)
    ax1.set_xticks(np.arange(len(tests)))
    ax1.set_xticklabels([_pretty_test(t) for t in tests], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Bias z-score")
    ax1.set_title("Estimate Bias (vs. Gold-Reference Null)")
    ax1.grid(axis="y", alpha=0.25, lw=0.8)

    ax2.axhline(target_cov, color="black", ls="--", lw=1.1, label=f"Target = {target_cov:.2f}")
    ax2.set_xticks(np.arange(len(tests)))
    ax2.set_xticklabels([_pretty_test(t) for t in tests], rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0.0, 1.02)
    ax2.set_ylabel("CI coverage of gold-reference null")
    ax2.set_title("CI Coverage")
    ax2.grid(axis="y", alpha=0.25, lw=0.8)
    ax2.legend(loc="lower left", fontsize=8)

    ax3.set_xticks(np.arange(len(tests)))
    ax3.set_xticklabels([_pretty_test(t) for t in tests], rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Mean CI width")
    ax3.set_title("CI Width")
    ax3.grid(axis="y", alpha=0.25, lw=0.8)

    handles, labels = ax1.get_legend_handles_labels()
    title_suffix = " -- Bootstrap/CI-Based Methods" if nonstandard else ""
    fig.suptitle(f"PPI-Corrected Effect-Size Calibration: Bias, Coverage, and Width{title_suffix}", y=1.12, fontsize=12)
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
                         help="multiarm mode: only affects max_t's point estimate + bootstrap draws (none/holm/"
                              "bonferroni/fdr_bh correct the canonical Wilcoxon signed-rank p-value regardless) / "
                              "simultaneous_ci mode: only affects max_t's construction (none/bonferroni/sidak/boot "
                              "build on the canonical per-eval-type CI regardless) -- must be bootstrap-compatible "
                              "for max-T to apply")
    parser.add_argument("--multiarm-icc", type=float, default=0.20, metavar="ICC",
                         help="multiarm/simultaneous_ci modes: ICC for build_multiarm_sources' shared truth/noise model "
                              "(same meaning as --icc-values in pairwise mode)")
    parser.add_argument("--multiarm-cohens-d", type=float, default=0.3, metavar="D",
                         help="multiarm/simultaneous_ci modes: alt-condition effect size (Cohen's d) for build_multiarm_sources")
    parser.add_argument("--corrections", nargs="+", choices=[m.name for m in MULTIARM_CORRECTION_METHODS], default=None, metavar="CORRECTION",
                         help="multiarm mode: restrict to these correction strategies (default: all of "
                              f"{[m.name for m in MULTIARM_CORRECTION_METHODS]}) -- e.g. for a fast targeted re-run "
                              "of just the resampling-based corrections (max_t/romano_wolf/westfall_young) at "
                              "larger n without paying for the full correction set")
    parser.add_argument("--ci-methods", nargs="+", choices=[m.name for m in ALL_SIMULTANEOUS_CI_METHODS], default=None, metavar="METHOD",
                         help="simultaneous_ci mode: restrict to these CI methods (default: all of "
                              f"{[m.name for m in ALL_SIMULTANEOUS_CI_METHODS]}) -- e.g. --ci-methods boot sidak "
                              "to skip max_t's and none's/bonferroni's independent bootstrap/construction cost "
                              "entirely, not just skip reporting them (max_t and boot each pay for their own "
                              "separate bootstrap resample in this mode, unlike --mode multiarm's sharing)")

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
    parser.add_argument("--no-typeI-check", action="store_true", default=False,
                         help="ppi mode: skip the base Type-I calibration sweep (build_judge_bias_sources, "
                              "by far the slowest single piece of --mode ppi). The other checks (effect/power/"
                              "comparison/factorial) don't consume its results, so this + --no-effect-check "
                              "--no-power-check --no-comparison-check --factorial-check runs JUST the factorial "
                              "sweep -- see official_args_ppi_factorial")
    parser.add_argument("--no-effect-check", action="store_true", default=False,
                         help="ppi mode: skip the bias/CI-coverage effect-size check, running Type-I calibration only")
    parser.add_argument("--no-power-check", action="store_true", default=False,
                         help="ppi mode: skip the power-under-bias check (build_ppi_power_sources), running "
                              "Type-I calibration (and, unless also disabled, the effect-size check) only")
    parser.add_argument("--no-comparison-check", action="store_true", default=False,
                         help="ppi mode: skip the 5-way estimator comparison (all_human/human_subset/llm_only/"
                              "llm_impute/ppi rejection rate vs. effect_size and label_frac, paired_t estimand)")
    parser.add_argument("--factorial-check", action="store_true", default=False,
                         help="ppi mode: run the full 7-factor factorial (bias_magnitude x N x N_lab x "
                              "label_mechanism x effect_size x bias_direction x llm_noise, continuous/paired_t) "
                              "-- opt-in (default off) since it's substantially more scenarios than the other "
                              "checks; see build_ppi_factorial_sources. Also produces the judge-human ALIGNMENT-"
                              "bucketed false-positive-rate view (weighted Cohen's kappa for likert, Pearson r for "
                              "continuous), derived from this same run's es=\"null\" cells rather than a separate "
                              "sweep -- see build_ppi_alignment_results_from_factorial/"
                              "save_ppi_alignment_sweep_plot's docstrings.")
    parser.add_argument("--factorial-reps", type=int, default=100, metavar="N",
                         help="ppi mode: reps for --factorial-check (default 100, a screening-tier rep count -- "
                              "bump toward --reps for a publication-precision confirmation pass)")
    parser.add_argument("--factorial-n-boot", type=int, default=500, metavar="N",
                         help="ppi mode: PPI bootstrap resample count for --factorial-check (default 500, "
                              "screening-tier -- bump toward --ppi-n-boot for a confirmation pass)")
    parser.add_argument("--factorial-likert-max", type=int, default=5, metavar="N",
                         help="ppi mode: top of the Likert scale's integer range for --factorial-check's likert "
                              "scenarios (default 5, the standard scale). A non-default value (e.g. 7) rescales "
                              "the SAME underlying distribution/bias/effect magnitudes onto a wider integer grid, "
                              "rather than generating a different one -- see scenarios.synthetic."
                              "build_ppi_factorial_sources' likert_max parameter. Ignored for continuous scenarios.")
    parser.add_argument("--factorial-omnibus", action="store_true", default=False,
                         help="ppi mode: also run the 4 omnibus/multi-group tests (anova_ind, anova_rep, friedman, "
                              "kruskal -- _COMPARISON_METHODS_OMNIBUS) against --factorial-check's SAME sources, "
                              "on top of the default 4 two-group tests (ttest_welch/paired_t/mwu_corr/wilcoxon). "
                              "Opt-in: uses generate_judge_bias_cell (the full generator, needed for the 3-group "
                              "structures anova_ind/kruskal and anova_rep/friedman read) for EVERY method now, not "
                              "just these 4, so enabling this meaningfully increases --factorial-check's runtime. "
                              "Reported and saved as its OWN pooled summary/log section (mean_of_4_omnibus), never "
                              "blended into the two-group tests' pooled rate -- see _COMPARISON_METHODS_OMNIBUS' "
                              "docstring for why (different hypothesis: 3-group omnibus vs. two-group location-shift).")
    parser.add_argument("--factorial-alignment-mc", type=int, default=20000, metavar="N",
                         help="ppi mode: Monte Carlo sample size for --factorial-check's alignment-bucketed view's "
                              "per-(eval_type, llm_noise, bias_delta) alignment measurement (measure_judge_alignment "
                              "-- a separate, large, effectively noise-free calibration draw, not the small "
                              "labeled-subset the Type-I sweep itself uses; default 20000 keeps the realized "
                              "alignment percentage stable to within ~1 point). Ignored unless --factorial-check "
                              "produces es=\"null\" cells.")
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


def official_args_pairwise(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for pairwise-only calibration (synthetic data).
    Split out from official_args() (which runs mode="pairwise_multiarm",
    i.e. both pairwise and multiarm together) so the pairwise sweep alone
    can be re-run on its own -- e.g. after a pairwise-specific performance
    or correctness fix -- without paying for the separate (and unrelated,
    much slower at high k) multiarm sweep every time."""
    args = official_args(base_seed)
    args.mode = "pairwise"
    return args


def official_args_multiarm(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for multiarm-only calibration (synthetic data).
    Split out from official_args() (which runs mode="pairwise_multiarm",
    i.e. both pairwise and multiarm together) so the multiarm sweep alone
    can be re-run on its own -- e.g. after a multiarm-specific performance
    or correctness fix -- without paying for the separate (and unrelated)
    pairwise sweep every time.

    Overrides official_args()'s sizes with official_args_simultaneous_ci's
    coarser 6-point n=15..500 sweep (rather than official_args()'s denser
    6-point sweep stopping at 100): multiarm's resampling-based FWER
    corrections (max_t/romano_wolf/westfall_young) are the direct p-value-
    side analogue of simultaneous_ci's CI constructions (same bootstrap-t/
    step-down machinery, same k-arm sources), so sweeping the same n range
    makes the two modes' small-N-to-large-N comparisons directly comparable
    instead of stopping multiarm's sweep short of the large-N regime
    simultaneous_ci's sweep was chosen to cover.

    Also overrides official_args()'s bootstrap_n=2000 with 5000:
    romano_wolf/westfall_young/boot's FWER ran consistently ~0.001-0.002
    above nominal alpha at bootstrap_n=500-2000 (confirmed via direct
    n=500-2000 sweeps, holding even at small k), traced to Monte Carlo noise
    in estimating the joint max-statistic's upper-tail quantile from too few
    draws -- not a correction-logic bug or a k-dependent effect (ruled out by
    `boot`, a structurally different bootstrap-based correction, showing the
    same excess). 5000 draws resolved it. This is no longer as costly as it
    used to be -- _bootstrap_t_matrix's resample construction was rewritten
    from an O(k_pairs*n_bootstrap*n) gather to a counts/matmul formulation
    (~12-27x faster for the "bootstrap" mode max_t/romano_wolf/boot share,
    ~2.3x for westfall_young's "permutation" mode). Left at 2000 for
    official_args()'s other consumers (pairwise, simultaneous_ci, ppi) --
    this finding is specific to multiarm's resampling-based FWER
    corrections, not verified to generalize to simultaneous_ci's CI coverage
    calibration."""
    args = official_args(base_seed)
    args.mode = "multiarm"
    args.sizes = [15, 30, 50, 100, 200, 500, 1000]
    args.bootstrap_n = 5000
    args.reps = 500
    return args


def official_args_ppi(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for PPI-corrected calibration (synthetic data --
    ppi has no real-data variant). Split out from official_args() since it's
    by far the slowest of the pvalues sub-modes (43 judge-bias scenarios x
    ~11 tests x reps), so it can be selected or skipped on its own in the
    --official-tests menu instead of always riding along with the faster
    pairwise/multiarm sweep. Runs the FULL rigorous PPI evaluation built up
    over this harness's development: Type-I calibration, the bias/coverage
    effect-size check, the power-under-bias check (plus its bias-direction
    and no-bias companions), the 5-way estimator comparison, the N x N_lab
    grid, and -- via factorial_check below -- the full 7-factor factorial
    (build_ppi_factorial_sources) plus the judge-human alignment-bucketed
    view it now also produces (build_ppi_alignment_results_from_factorial).
    All of these except factorial_check are
    already on by default in run() (--no-power-check/--no-comparison-check
    are opt-OUT), so this preset only needs to explicitly enable
    factorial_check (opt-IN by default, given its larger scenario count)
    and give it the SAME precision tier the other secondary checks already
    run at here -- effect_reps/ppi_n_boot (200/2000) -- rather than
    inventing a third reps/n_boot tier alongside --reps and
    --factorial-check's own screening-tier CLI default (100/500, meant for
    fast interactive iteration, not a result worth citing)."""
    args = official_args(base_seed)
    args.mode = "ppi"
    args.factorial_check = True
    args.factorial_reps = args.effect_reps
    args.factorial_n_boot = args.ppi_n_boot
    return args


def official_args_ppi_no_lmm(base_seed: int = 42) -> argparse.Namespace:
    """Same as official_args_ppi (same checks, same scenarios, same reps/
    n_boot), except --tests excludes the three LMM-based methods (lmm/
    lmm_factorial/lmm_runs). LMM is profiled at ~70% of --mode ppi's total
    runtime (its mixed-model fits dominate build_judge_bias_sources' Type-I
    sweep and the power check, both of which iterate active_tests over
    every scenario) -- see run_ppi_simulation/_run_ppi_cell's docstrings.
    This preset exists purely for a faster quality-check pass (e.g. after a
    change to one of the OTHER PPI tests, or before a merge) when LMM's own
    calibration isn't what's being verified, at a fraction of the
    wall-clock cost.

    The factorial/N x N_lab comparison sweep (_COMPARISON_METHODS) never
    ran LMM to begin with (it's ttest_welch/paired_t/mwu_corr/wilcoxon
    only), so this only changes the main Type-I sweep and power check --
    it does NOT skip factorial_check itself; pair with --no-factorial-check
    (or official_args_ppi_factorial's already-LMM-free scope) if the
    factorial sweep's own cost also needs trimming."""
    args = official_args_ppi(base_seed)
    args.tests = [
        m.name for m in PPI_OFFICIAL_TEST_METHODS
        if m.name not in (LMM.name, LMM_FACTORIAL.name, LMM_RUNS.name)
    ]
    return args


def official_args_ppi_factorial(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for JUST the full 7-factor factorial sweep
    (build_ppi_factorial_sources, including its judge-human alignment-
    bucketed view -- build_ppi_alignment_results_from_factorial), split out
    from official_args_ppi the same way official_args_ppi itself is split
    from official_args -- lets --official-tests run/skip it independently,
    e.g. to iterate on the factorial analysis alone without re-paying for
    the base Type-I sweep (build_judge_bias_sources' ~85+ scenarios x ~11
    tests x reps -- by far the slowest piece of --mode ppi). Safe to isolate
    this way because the factorial sweep is fully self-contained: its own
    sources (build_ppi_factorial_sources), its own run_ppi_comparison_
    simulation call, no dependency on the Type-I/effect/power/comparison
    checks' results -- this is a real subset of official_args_ppi's work,
    not an approximation of it. Disables every other --mode ppi check via
    --no-typeI-check/--no-effect-check/--no-power-check/
    --no-comparison-check (all opt-out; harmless to set even though
    official_args_ppi doesn't set them, since their defaults already run).

    factorial_omnibus=True: also runs the 4 omnibus/multi-group tests
    (anova_ind/anova_rep/friedman/kruskal -- _COMPARISON_METHODS_OMNIBUS)
    against these same factorial sources, not just the original 4 two-group
    tests -- added once the main OFAT sweep (build_judge_bias_sources) and
    this factorial sweep's own two-group tests confirmed those 4 held up
    reasonably well under the combined-factor stress test, making it worth
    checking whether anova/friedman/kruskal (kruskal in particular already
    flagged as a milder, more diffuse Type-I outlier in the OFAT sweep) also
    hold up here, or blow up the way mw_naive did before mwu_corr replaced
    it. NOT set on official_args_ppi/official_args_ppi_no_lmm (the "run
    everything" presets, already by far the slowest --mode ppi variants) --
    only this standalone factorial-only preset, so the extra cost (roughly
    2x the method count, using the full generator now for every method) is
    opt-in at the granularity where it's easiest to run/iterate on its own."""
    args = official_args_ppi(base_seed)
    args.no_typeI_check = True
    args.no_effect_check = True
    args.no_power_check = True
    args.no_comparison_check = True
    args.factorial_omnibus = True
    return args


def official_args_ppi_factorial_likert7(base_seed: int = 42) -> argparse.Namespace:
    """Same as official_args_ppi_factorial, except likert scenarios are
    generated on a 1-7 scale instead of the standard 1-5 (factorial_likert_max
    = 7 -- see build_ppi_factorial_sources' likert_max parameter). Continuous
    scenarios are unaffected (likert_max is a no-op for them).

    Exists to test a specific hypothesis raised after the first factorial run
    (see simulations/out/official_20260718_213255): PPI-corrected
    Mann-Whitney's Type-I rate blew up specifically for likert scenarios
    under severe MNAR labeling (up to 0.445 at et=likert/bm=severe/n=400/
    nlab=80/lm=mnar_strong), while paired_t/wilcoxon/ttest_welch stayed
    well-calibrated in that exact same scenario, AND mw itself stayed
    well-calibrated on continuous (effectively tie-free) data under the same
    severe MNAR mechanism -- pointing at Likert's coarse, heavily-tied 5-level
    discretization (not MNAR alone, and not rank tests generally) as the
    likely aggravating factor for mw's independent-groups midrank
    construction specifically. Comparing this run's likert Type-I/power
    numbers against the 1-5 run's is the intended follow-up analysis."""
    args = official_args_ppi_factorial(base_seed)
    args.factorial_likert_max = 7
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
    - sizes is a coarser 6-point sweep spanning n=15 to n=500 (rather than
      official_args()'s denser 6-point sweep stopping at 100):
      save_simultaneous_ci_violin_vs_n_plot's per-n grouped violins
      (tango_naive/sidak/boot constructions alongside none/Bonferroni/
      max-T) are most informative for deciding a real default when they
      span the full small-N (where multiplicity eats the most power) to
      large-N (where all constructions should converge) range a real
      evaluation might have, not just the ~30 crossover this preset
      historically anchored on -- kept to 6 points, not official_args()'s
      density, since this mode's per-cell cost (bootstrap_t's nested double
      bootstrap, k(k-1)/2 marginal pairs plus the shared max-T resample,
      times the tango/sidak/boot rows on top for binary sources) is already
      the most expensive of the pvalues sub-modes.
    """
    args = official_args(base_seed)
    args.mode = "simultaneous_ci"
    args.scenario_suite = "expanded"
    args.sizes = [15, 30, 50, 100, 200, 500]
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


def real_official_args_pairwise(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for pairwise-only calibration, real data. Split
    out from real_official_args() (mode="pairwise_multiarm") the same way
    official_args_pairwise is split from official_args() -- lets the
    (network/HF-dependent) real-data pairwise sweep be re-run on its own
    without also paying for the real-data multiarm sweep."""
    args = real_official_args(base_seed)
    args.mode = "pairwise"
    return args


def real_official_args_multiarm(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for multiarm-only calibration, real data. Split
    out from real_official_args() (mode="pairwise_multiarm") the same way
    real_official_args_pairwise is -- lets the (network/HF-dependent)
    real-data multiarm sweep (FWER + best-arm power across
    none/holm/bonferroni/fdr_bh/hochberg/shaffer/friedman_nemenyi/max_t/
    romano_wolf/westfall_young -- see MULTIARM_CORRECTION_METHODS) be
    re-run on its own without also paying for the real-data pairwise sweep.

    Also bumps bootstrap_n from real_official_args()'s 2000 to 5000 -- see
    official_args_multiarm's docstring for why (Monte Carlo noise in the
    joint max-statistic's quantile at low bootstrap_n, not a k-dependent or
    correction-logic issue); applies identically regardless of data source."""
    args = real_official_args(base_seed)
    args.mode = "multiarm"
    args.bootstrap_n = 5000
    args.sizes=[10, 15, 20, 30, 50, 75, 100]
    args.k_arms = [3, 5]
    args.reps = 500
    return args


def real_official_args_simultaneous_ci(base_seed: int = 42) -> argparse.Namespace:
    """Official-test preset for simultaneous-CI calibration only, real data
    (real multi-arm sources -- see build_real_multiarm_sources). Split out
    from real_official_args() the same way official_args_simultaneous_ci is."""
    args = real_official_args(base_seed)
    args.mode = "simultaneous_ci"
    args.bootstrap_n = 5000
    args.sizes=[10, 15, 20, 30, 50, 75, 100]
    args.k_arms = [3, 5]
    return args


def official_variants(base_seed: int = 42) -> list[tuple[str, argparse.Namespace]]:
    """All official-test variants for this case, as (label, args) pairs."""
    return [
        ("synthetic (pairwise + multiarm)", official_args(base_seed)),
        ("synthetic (pairwise)", official_args_pairwise(base_seed)),
        ("synthetic (multiarm)", official_args_multiarm(base_seed)),
        ("synthetic (ppi)", official_args_ppi(base_seed)),
        ("synthetic (ppi, no LMM)", official_args_ppi_no_lmm(base_seed)),
        ("synthetic (ppi factorial only)", official_args_ppi_factorial(base_seed)),
        ("synthetic (ppi factorial only, likert 1-7)", official_args_ppi_factorial_likert7(base_seed)),
        ("synthetic (simultaneous CI)", official_args_simultaneous_ci(base_seed)),
        ("real data (pairwise + multiarm)", real_official_args(base_seed)),
        ("real data (pairwise)", real_official_args_pairwise(base_seed)),
        ("real data (multiarm)", real_official_args_multiarm(base_seed)),
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
    don't go unexercised between --official-tests runs. factorial_check=True
    (with trivial factorial_reps/factorial_n_boot) so build_ppi_factorial_
    sources/fit_ppi_factorial_model/save_ppi_factorial_heatmap_plot stay
    exercised here too -- it's opt-in in run() (unlike power_check/
    comparison_check, which are opt-OUT and so already covered by this
    preset's defaults), so a regression there would otherwise go completely
    uncaught between --official-tests runs. factorial_alignment_mc=200 (well
    below the 20000 default) keeps the alignment-bucketed view's per-(eval_
    type, noise, bias) calibration draws cheap here too -- this preset only
    needs the code path exercised, not a precise alignment percentage."""
    eval_types = ["binary", "continuous"] if data_source == "synthetic" else ["binary"]
    return argparse.Namespace(
        mode="all", reps=3, alpha=0.05, seed=base_seed,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        data_source=data_source, scenario_suite="standard", eval_types=eval_types, sizes=[10, 30, 50],
        runs=1, statistic="mean",
        bootstrap_n=200, icc_values=[0.20], cohens_d_values=[0.3],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        k_arms=[3], multiarm_method=BOOTSTRAP_T.name, multiarm_icc=0.20, multiarm_cohens_d=0.3,
        tests=[TTEST.name, MW_NAIVE.name, MWU_CORR.name, PAIRED_T.name, BAYES_BOOTSTRAP.name, BOOTSTRAP_T.name, TANGO.name], ppi_n_boot=200, latex=True,
        effect_reps=5, effect_gold_mc=200, no_effect_check=False,
        factorial_check=True, factorial_reps=2, factorial_n_boot=50, factorial_alignment_mc=200,
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
                n_workers=getattr(args, "workers", 1), corrections=getattr(args, "corrections", None),
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
                vs_n_path = save_multiarm_fwer_vs_n_plot(results=ma_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_fwer_vs_n.png"))
                if Path(vs_n_path).exists():
                    output_paths.append(vs_n_path)
                    print(f"Saved plot: {vs_n_path}")
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
                n_workers=getattr(args, "workers", 1), ci_methods=getattr(args, "ci_methods", None),
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
                vs_n_path = save_simultaneous_ci_coverage_width_vs_n_plot(
                    results=sci_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_vs_n.png"),
                )
                if Path(vs_n_path).exists():
                    output_paths.append(vs_n_path)
                    print(f"Saved plot: {vs_n_path}")
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

            for cm_name in ("none", "bonferroni", "max_t", CORR_SIDAK.name, CORR_BOOT.name):
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
            # Default (no --tests) runs the OFFICIAL subset -- excludes
            # mw_naive (superseded by mwu_corr; see methods.py) but still
            # selectable explicitly via --tests mw_naive for comparison.
            active_tests = args.tests if args.tests else [m.name for m in PPI_OFFICIAL_TEST_METHODS]
            print(f"\npvalues simulation (PPI-corrected) -- tests={active_tests}")
            jb_sources = build_judge_bias_sources()
            if args.eval_types:
                requested = set(args.eval_types)
                jb_sources = [s for s in jb_sources if s.eval_type in requested]
            if not jb_sources:
                raise ValueError("No JudgeBiasSources left after filtering.")

            if not getattr(args, "no_typeI_check", False):
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

            power_sources = build_ppi_power_sources()
            if args.eval_types:
                requested = set(args.eval_types)
                power_sources = [s for s in power_sources if s.eval_type in requested]

            if not getattr(args, "no_power_check", False) and power_sources:
                power_reps = getattr(args, "effect_reps", 200)
                print(f"\npvalues simulation (PPI-corrected, power check) -- {len(power_sources)} scenarios, "
                      f"reps={power_reps}, n_boot={args.ppi_n_boot}")
                power_results = run_ppi_simulation(
                    power_sources, active_tests=active_tests, n_reps=power_reps, n_boot=args.ppi_n_boot,
                    progress_mode=args.progress, seed=args.seed + 2, n_workers=getattr(args, "workers", 1),
                )
                print_ppi_power_report(power_results, alpha=args.alpha)

                # No-bias baseline computed BEFORE the main power plot (not
                # after, as originally ordered) so its corrected rate can be
                # overlaid there as an "ideal" reference line -- does PPI
                # correction cost power for nothing when there's no judge
                # bias to correct for, and how close does the biased-
                # condition line above track that ceiling? See
                # build_ppi_power_nobias_sources' docstring.
                nobias_sources = build_ppi_power_nobias_sources()
                if args.eval_types:
                    nobias_sources = [s for s in nobias_sources if s.eval_type in requested]
                nobias_results: list[PPIResult] = []
                if nobias_sources:
                    print(f"\npvalues simulation (PPI-corrected, power check -- no bias) -- "
                          f"{len(nobias_sources)} scenarios")
                    nobias_results = run_ppi_simulation(
                        nobias_sources, active_tests=active_tests, n_reps=power_reps, n_boot=args.ppi_n_boot,
                        progress_mode=args.progress, seed=args.seed + 5, n_workers=getattr(args, "workers", 1),
                    )
                    print_ppi_power_report(
                        nobias_results, alpha=args.alpha, header="POWER, NO JUDGE BIAS (bias_type=none)",
                    )
                    if args.save_results == "save":
                        output_paths += save_results_artifacts_ppi_power(
                            results=nobias_results, alpha=args.alpha, out_dir=args.out_dir,
                            run_stem=f"pvalues_ppi_power_nobias_reps{power_reps}_{stamp}",
                        )
                    key_metrics["ppi_power_nobias_n_results"] = len(nobias_results)

                power_stem = f"pvalues_ppi_power_reps{power_reps}_{stamp}"
                if power_results:
                    if args.save_results == "save":
                        output_paths += save_results_artifacts_ppi_power(
                            results=power_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=power_stem,
                        )
                    if args.plots == "save":
                        power_plot_path = save_ppi_power_plot(
                            results=power_results, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{power_stem}_power_vs_effect_size.png"),
                        )
                        output_paths.append(power_plot_path)
                        print(f"Saved plot: {power_plot_path}")
                        if nobias_results:
                            nobias_plot_path = save_ppi_power_plot(
                                results=nobias_results, alpha=args.alpha,
                                out_path=str(Path(plots_dir) / f"{power_stem}_power_vs_effect_size_nobias.png"),
                                title_suffix=" -- No Judge Bias",
                            )
                            output_paths.append(nobias_plot_path)
                            print(f"Saved plot: {nobias_plot_path}")

                    key_metrics["ppi_power_n_results"] = len(power_results)
                    top_es = max({_parse_ppi_power_name(r.name)[1] for r in power_results}, default=0.0)
                    top_rows = [r for r in power_results if _parse_ppi_power_name(r.name)[1] == top_es]
                    c_tot = sum(r.corrected_rejects for r in top_rows)
                    n_tot = sum(r.n_reps for r in top_rows)
                    key_metrics["ppi_power_mean_corrected_at_max_es"] = float(c_tot / n_tot) if n_tot else float("nan")

                # Bias-direction check: does the "cancellation dip" (opposing
                # bias vs. effect, already run above as power_results) look
                # different from the reinforcing-bias case, where an
                # uncorrected test would just quietly overstate the effect
                # instead of showing a visible anomaly? See
                # build_ppi_power_reinforcing_sources' docstring.
                reinforcing_sources = build_ppi_power_reinforcing_sources()
                if args.eval_types:
                    reinforcing_sources = [s for s in reinforcing_sources if s.eval_type in requested]
                reinforcing_results: list[PPIResult] = []
                if reinforcing_sources:
                    print(f"\npvalues simulation (PPI-corrected, power check -- bias reinforcing effect) -- "
                          f"{len(reinforcing_sources)} scenarios")
                    reinforcing_results = run_ppi_simulation(
                        reinforcing_sources, active_tests=active_tests, n_reps=power_reps, n_boot=args.ppi_n_boot,
                        progress_mode=args.progress, seed=args.seed + 4, n_workers=getattr(args, "workers", 1),
                    )
                    print_ppi_power_report(
                        reinforcing_results, alpha=args.alpha,
                        header="POWER UNDER JUDGE BIAS (reinforcing the real effect)",
                    )
                    if args.save_results == "save":
                        output_paths += save_results_artifacts_ppi_power(
                            results=reinforcing_results, alpha=args.alpha, out_dir=args.out_dir,
                            run_stem=f"pvalues_ppi_power_reinforcing_reps{power_reps}_{stamp}",
                        )
                    if args.plots == "save" and power_results:
                        direction_plot_path = save_ppi_power_direction_plot(
                            opposing=power_results, reinforcing=reinforcing_results, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{power_stem}_power_direction.png"),
                        )
                        output_paths.append(direction_plot_path)
                        print(f"Saved plot: {direction_plot_path}")
                    key_metrics["ppi_power_reinforcing_n_results"] = len(reinforcing_results)

            comparison_results_pooled: list[PPIComparisonResult] = []
            nlab_cal_pooled: list[PPIComparisonResult] = []
            if not getattr(args, "no_comparison_check", False):
                comparison_sources = power_sources + build_ppi_comparison_label_frac_sources()
                if args.eval_types:
                    requested = set(args.eval_types)
                    comparison_sources = [s for s in comparison_sources if s.eval_type in requested]
                if comparison_sources:
                    comparison_reps = getattr(args, "effect_reps", 200)
                    print(f"\npvalues simulation (PPI-corrected, estimator comparison) -- "
                          f"{len(comparison_sources)} scenarios x {len(_COMPARISON_METHODS)} methods "
                          f"({_COMPARISON_METHODS_LABEL}), reps={comparison_reps}, n_boot={args.ppi_n_boot}")
                    comparison_results_raw = run_ppi_comparison_simulation(
                        comparison_sources, n_reps=comparison_reps, n_boot=args.ppi_n_boot,
                        progress_mode=args.progress, seed=args.seed + 3, n_workers=getattr(args, "workers", 1),
                        methods=_COMPARISON_METHODS,
                    )
                    comparison_results_pooled = pool_ppi_comparison_across_methods(comparison_results_raw)
                    print_ppi_comparison_report(comparison_results_pooled, alpha=args.alpha)

                    comparison_stem = f"pvalues_ppi_comparison_reps{comparison_reps}_{stamp}"
                    if args.save_results == "save":
                        output_paths += save_results_artifacts_ppi_comparison(
                            results=comparison_results_raw, pooled_results=comparison_results_pooled,
                            alpha=args.alpha, out_dir=args.out_dir, run_stem=comparison_stem,
                        )
                    if args.plots == "save":
                        comparison_plot_path = save_ppi_comparison_plot(
                            results=comparison_results_pooled, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{comparison_stem}_five_way_comparison.png"),
                        )
                        output_paths.append(comparison_plot_path)
                        print(f"Saved plot: {comparison_plot_path}")

                    key_metrics["ppi_comparison_n_results"] = len(comparison_results_pooled)
                    max_es_rows = [r for r in comparison_results_pooled if r.tag == "power" and r.effect_size == max((r.effect_size for r in comparison_results_pooled if r.tag == "power"), default=0.0)]
                    if max_es_rows:
                        key_metrics["ppi_comparison_power_all_human_at_max_es"] = float(
                            sum(r.rejects_all_human for r in max_es_rows) / sum(r.n_reps for r in max_es_rows)
                        )
                        key_metrics["ppi_comparison_power_human_subset_at_max_es"] = float(
                            sum(r.rejects_human_subset for r in max_es_rows) / sum(r.n_reps for r in max_es_rows)
                        )
                        key_metrics["ppi_comparison_power_ppi_at_max_es"] = float(
                            sum(r.rejects_ppi for r in max_es_rows) / sum(r.n_reps for r in max_es_rows)
                        )

                # N x N_lab grid: does calibration/power depend on the RATIO
                # N_lab/N or the ABSOLUTE N_lab count? build_ppi_nlab_grid_sources
                # covers continuous and likert (see its docstring); filter
                # per-source by eval_type against --eval-types rather than an
                # all-or-nothing check, so e.g. --eval-types likert alone
                # still produces likert cells.
                nlab_cal_sources = build_ppi_nlab_grid_sources(effect_frac=0.0)
                nlab_pow_sources = build_ppi_nlab_grid_sources(effect_frac=PPI_COMPARISON_MODERATE_EFFECT_FRAC)
                if args.eval_types:
                    requested = set(args.eval_types)
                    nlab_cal_sources = [s for s in nlab_cal_sources if s.eval_type in requested]
                    nlab_pow_sources = [s for s in nlab_pow_sources if s.eval_type in requested]
                if nlab_cal_sources or nlab_pow_sources:
                    nlab_reps = getattr(args, "effect_reps", 200)
                    print(f"\npvalues simulation (PPI-corrected, N x N_lab grid) -- "
                          f"{len(nlab_cal_sources)} calibration + {len(nlab_pow_sources)} power scenarios "
                          f"x {len(_COMPARISON_METHODS)} methods, reps={nlab_reps}, n_boot={args.ppi_n_boot}")
                    nlab_cal_raw = run_ppi_comparison_simulation(
                        nlab_cal_sources, n_reps=nlab_reps, n_boot=args.ppi_n_boot,
                        progress_mode=args.progress, seed=args.seed + 6, n_workers=getattr(args, "workers", 1),
                        methods=_COMPARISON_METHODS,
                    ) if nlab_cal_sources else []
                    nlab_pow_raw = run_ppi_comparison_simulation(
                        nlab_pow_sources, n_reps=nlab_reps, n_boot=args.ppi_n_boot,
                        progress_mode=args.progress, seed=args.seed + 7, n_workers=getattr(args, "workers", 1),
                        methods=_COMPARISON_METHODS,
                    ) if nlab_pow_sources else []
                    nlab_cal_pooled = pool_ppi_comparison_across_methods(nlab_cal_raw) if nlab_cal_raw else []
                    nlab_pow_pooled = pool_ppi_comparison_across_methods(nlab_pow_raw) if nlab_pow_raw else []
                    print_ppi_nlab_grid_report(
                        nlab_cal_pooled, alpha=args.alpha, header="N x N_LAB GRID (calibration, effect_size=0)",
                    )
                    print_ppi_nlab_grid_report(
                        nlab_pow_pooled, alpha=args.alpha, header="N x N_LAB GRID (power, moderate effect_size)",
                    )

                    nlab_stem = f"pvalues_ppi_nlab_grid_reps{nlab_reps}_{stamp}"
                    if args.save_results == "save":
                        if nlab_cal_raw:
                            output_paths += save_results_artifacts_ppi_nlab_grid(
                                results=nlab_cal_raw, pooled_results=nlab_cal_pooled,
                                alpha=args.alpha, out_dir=args.out_dir,
                                run_stem=f"{nlab_stem}_calibration", header="N x N_LAB GRID (calibration, effect_size=0)",
                            )
                        if nlab_pow_raw:
                            output_paths += save_results_artifacts_ppi_nlab_grid(
                                results=nlab_pow_raw, pooled_results=nlab_pow_pooled,
                                alpha=args.alpha, out_dir=args.out_dir,
                                run_stem=f"{nlab_stem}_power", header="N x N_LAB GRID (power, moderate effect_size)",
                            )
                    if args.plots == "save":
                        nlab_plot_path = save_ppi_nlab_grid_plot(
                            calibration_results=nlab_cal_pooled or None, power_results=nlab_pow_pooled or None,
                            alpha=args.alpha, out_path=str(Path(plots_dir) / f"{nlab_stem}_heatmap.png"),
                        )
                        output_paths.append(nlab_plot_path)
                        print(f"Saved plot: {nlab_plot_path}")

                    key_metrics["ppi_nlab_grid_n_calibration_results"] = len(nlab_cal_pooled)
                    key_metrics["ppi_nlab_grid_n_power_results"] = len(nlab_pow_pooled)

                # Null-effect 5-way bar chart: pools across BOTH _COMPARISON_METHODS
                # (already done above) AND, for continuous, the N x N_lab grid just
                # computed -- see save_ppi_null_comparison_plot's docstring for why
                # this is more defensible than reading off a single scenario.
                if args.plots == "save" and comparison_results_pooled:
                    null_comparison_plot_path = save_ppi_null_comparison_plot(
                        results=comparison_results_pooled, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"pvalues_ppi_comparison_reps{getattr(args, 'effect_reps', 200)}_{stamp}_null_false_positive_rate.png"),
                        nlab_cal_results=nlab_cal_pooled or None,
                    )
                    output_paths.append(null_comparison_plot_path)
                    print(f"Saved plot: {null_comparison_plot_path}")

            if getattr(args, "factorial_check", False):
                factorial_likert_max = getattr(args, "factorial_likert_max", 5)
                factorial_omnibus = getattr(args, "factorial_omnibus", False)
                factorial_sources = build_ppi_factorial_sources(likert_max=factorial_likert_max)
                if args.eval_types:
                    requested = set(args.eval_types)
                    factorial_sources = [s for s in factorial_sources if s.eval_type in requested]
                if factorial_sources:
                    factorial_reps = getattr(args, "factorial_reps", 100)
                    factorial_n_boot = getattr(args, "factorial_n_boot", 500)
                    likert_note = f", likert_max={factorial_likert_max}" if factorial_likert_max != 5 else ""
                    factorial_methods = _COMPARISON_METHODS + (_COMPARISON_METHODS_OMNIBUS if factorial_omnibus else ())
                    omnibus_note = f" + {len(_COMPARISON_METHODS_OMNIBUS)} omnibus tests" if factorial_omnibus else ""
                    print(f"\npvalues simulation (PPI-corrected, full factorial) -- "
                          f"{len(factorial_sources)} scenarios x {len(_COMPARISON_METHODS)} methods{omnibus_note}, "
                          f"reps={factorial_reps}, n_boot={factorial_n_boot}{likert_note}")
                    factorial_results_raw = run_ppi_comparison_simulation(
                        factorial_sources, n_reps=factorial_reps, n_boot=factorial_n_boot,
                        progress_mode=args.progress, seed=args.seed + 8, n_workers=getattr(args, "workers", 1),
                        methods=factorial_methods,
                    )
                    factorial_results = pool_ppi_comparison_across_methods(
                        [r for r in factorial_results_raw if r.method in _COMPARISON_METHODS]
                    )
                    # GLM/heatmap/headline-report stay scoped to the llm_noise=0.20
                    # baseline (the only noise level non-null cells even have) --
                    # see _PPI_FACTORIAL_FORMULA's docstring for why llm_noise
                    # can't safely join that model as an eighth term. The FULL
                    # factorial_results (every noise level) is reserved for the
                    # alignment-bucketed view below.
                    factorial_results_baseline = [
                        r for r in factorial_results if _parse_ppi_factorial_name(r.name)["noise"] == 0.20
                    ]
                    print_ppi_factorial_report(
                        factorial_results_baseline, alpha=args.alpha, label=_COMPARISON_METHODS_LABEL,
                    )

                    omnibus_results = None
                    omnibus_results_baseline = None
                    if factorial_omnibus:
                        omnibus_results = pool_ppi_comparison_across_methods(
                            [r for r in factorial_results_raw if r.method in _COMPARISON_METHODS_OMNIBUS]
                        )
                        omnibus_results_baseline = [
                            r for r in omnibus_results if _parse_ppi_factorial_name(r.name)["noise"] == 0.20
                        ]
                        print_ppi_factorial_report(
                            omnibus_results_baseline, alpha=args.alpha, label=_COMPARISON_METHODS_OMNIBUS_LABEL,
                        )

                    stem_lmax_suffix = f"_lmax{factorial_likert_max}" if factorial_likert_max != 5 else ""
                    factorial_stem = f"pvalues_ppi_factorial_reps{factorial_reps}{stem_lmax_suffix}_{stamp}"
                    if args.save_results == "save":
                        output_paths += save_results_artifacts_ppi_factorial(
                            results=factorial_results_raw, pooled_results=factorial_results_baseline,
                            alpha=args.alpha, out_dir=args.out_dir, run_stem=factorial_stem,
                            label=_COMPARISON_METHODS_LABEL,
                        )
                        if omnibus_results_baseline is not None:
                            output_paths += save_results_artifacts_ppi_factorial(
                                results=factorial_results_raw, pooled_results=omnibus_results_baseline,
                                alpha=args.alpha, out_dir=args.out_dir, run_stem=factorial_stem,
                                write_csv=False, label=_COMPARISON_METHODS_OMNIBUS_LABEL,
                            )
                    if args.plots == "save":
                        factorial_plot_path = save_ppi_factorial_heatmap_plot(
                            results=factorial_results_baseline, alpha=args.alpha,
                            out_path=str(Path(plots_dir) / f"{factorial_stem}_slices.png"),
                        )
                        output_paths.append(factorial_plot_path)
                        print(f"Saved plot: {factorial_plot_path}")

                    key_metrics["ppi_factorial_n_results"] = len(factorial_results_baseline)
                    key_metrics["ppi_factorial_likert_max"] = factorial_likert_max
                    null_results = [
                        r for r in factorial_results_baseline if _parse_ppi_factorial_name(r.name)["es"] == "null"
                    ]
                    if null_results:
                        c_tot = sum(r.rejects_ppi for r in null_results)
                        n_tot = sum(r.n_reps for r in null_results)
                        key_metrics["ppi_factorial_mean_type1"] = float(c_tot / n_tot) if n_tot else float("nan")

                    if omnibus_results_baseline is not None:
                        key_metrics["ppi_factorial_omnibus_n_results"] = len(omnibus_results_baseline)
                        null_omnibus = [
                            r for r in omnibus_results_baseline if _parse_ppi_factorial_name(r.name)["es"] == "null"
                        ]
                        if null_omnibus:
                            c_tot_o = sum(r.rejects_ppi for r in null_omnibus)
                            n_tot_o = sum(r.n_reps for r in null_omnibus)
                            key_metrics["ppi_factorial_omnibus_mean_type1"] = (
                                float(c_tot_o / n_tot_o) if n_tot_o else float("nan")
                            )

                    # Judge-human alignment-bucketed view, derived from this SAME
                    # factorial run's es="null" cells (all llm_noise levels) --
                    # see build_ppi_alignment_results_from_factorial's docstring.
                    align_mc = getattr(args, "factorial_alignment_mc", 20000)
                    alignment_results = build_ppi_alignment_results_from_factorial(
                        factorial_sources, factorial_results, n_align_mc=align_mc, seed=args.seed + 9,
                    )
                    if alignment_results:
                        print_ppi_alignment_sweep_report(alignment_results, alpha=args.alpha)

                        hh_rows = run_human_human_alignment_sweep(n_mc=align_mc, seed=args.seed + 10)
                        print_human_human_alignment_report(hh_rows)

                        if args.save_results == "save":
                            output_paths += save_results_artifacts_ppi_alignment_sweep(
                                results=alignment_results, alpha=args.alpha, out_dir=args.out_dir,
                                run_stem=f"{factorial_stem}_alignment", human_human_rows=hh_rows,
                            )
                        if args.plots == "save":
                            for view_et, view_metric, view_label, view_band_fn, view_band_source, view_symbol in _ALIGNMENT_VIEWS:
                                if not any(r.eval_type == view_et and view_metric in r.alignment_metrics for r in alignment_results):
                                    continue
                                align_plot_path = save_ppi_alignment_sweep_plot(
                                    results=alignment_results, eval_type=view_et, metric=view_metric,
                                    display_label=view_label, band_fn=view_band_fn, band_source=view_band_source,
                                    symbol=view_symbol, alpha=args.alpha,
                                    out_path=str(Path(plots_dir) / f"{factorial_stem}_alignment_{view_et}_{view_metric}.png"),
                                )
                                output_paths.append(align_plot_path)
                                print(f"Saved plot: {align_plot_path}")

                            hh_plot_path = save_human_human_alignment_plot(
                                rows=hh_rows, out_path=str(Path(plots_dir) / f"{factorial_stem}_alignment_human_human.png"),
                            )
                            output_paths.append(hh_plot_path)
                            print(f"Saved plot: {hh_plot_path}")

                        key_metrics["ppi_alignment_sweep_n_results"] = len(alignment_results)

        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics=key_metrics, duration_s=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return CaseResult(case_name=CASE_NAME, status="error", error=str(exc), duration_s=time.time() - t0)
