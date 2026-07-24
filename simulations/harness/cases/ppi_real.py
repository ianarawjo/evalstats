"""ppi_real -- PPI-correction calibration checks against REAL (human_label,
judge_score) data collected by simulations/collect_judge_bias_data.py, as a
real-data complement to pvalues.py's ``--mode ppi`` (which is synthetic-only
-- see that file's "Known exceptions" note).

A separate case module (rather than growing pvalues.py further, which is
already very large) that stays thin by importing pvalues.py's own PPI
result types, test-battery helpers, and report/plot functions directly --
they're fully generic over (name, tag, test) labeled results and have no
JudgeBiasSource-specific coupling, so they work unmodified here. See
scenarios/real_judge_bias.py's module docstring for the data-loading side.

Four checks:

  single-sample bias/coverage
      Per judge model: does the PPI-corrected estimate of THIS corpus's
      true mean human_label recover it, with correct CI coverage, as
      label_frac (how much of the human-labeled subset is "revealed") and
      n vary? Uses evalstats.tests' _ppi_single_bootstrap_t (+
      _ppi_single_wilson for binary) -- functions pvalues.py's synthetic
      PPI sweep never exercises at all (it only ever tests group
      *comparisons*, never a plain one-sample mean estimate). Not a
      p-value/hypothesis-test check -- these functions return
      p_value=None.

  two-group Type-I null (random split)
      Per judge model: bisect the corpus into two disjoint random
      subsamples -- a valid null by construction (both are random draws
      from the identical population, so their true means are equal) -- and
      run the independent-samples tests (ttest/ttest_welch/mwu/
      mwu_mnar_experimental) PPI correction is supposed to keep calibrated, with real
      noise/skew/judge-bias characteristics instead of synthetic ones.

  paired Type-I null (cross-judge)
      For every unique PAIR of judge models scoring the SAME items: an
      EXACT null (stronger than the two-group check's "equal in
      distribution" -- here the true paired difference is exactly zero for
      every item, since both judges are noisy/biased reads of the
      IDENTICAL human_label). Runs the paired-samples tests (wilcoxon/
      paired_t/bayes_bootstrap/bootstrap_t/tango) that a single judge model
      alone can't exercise at all. With k judge models collected, all
      C(k, 2) pairs are checked (capped by --max-pairs) -- more judges
      means more independent (dataset, label_frac, n, pair) cells feeding
      the SAME calibration question, i.e. more power to catch a
      miscalibration than any single pair would give alone.

  within-item paired bias/coverage (wmt_da_paired, additive/optional)
      The paired-null check above pits two DIFFERENT judge models against
      each other as a proxy for real paired A/B data -- a real deployment
      more often has ONE judge scoring TWO conditions for the SAME item
      (e.g. response A vs response B for the same prompt), which is a
      different disagreement structure (inter-judge vs within-judge/
      across-condition) that the proxy can't exercise. This check instead
      uses simulations/collect_judge_bias_data.py's --types
      continuous_paired collection (RicardoRei/wmt-da-human-evaluation
      regrouped by source segment: the SAME source sentence translated by
      >=2 different MT systems, each independently human-DA-scored) --
      genuine within-item paired data for a single judge. Unlike the
      Type-I null checks above, system A and system B have REAL, generally
      DIFFERENT human labels (not a constructed identical label), so this
      is a bias/coverage check (does the PPI-corrected paired estimate
      recover the corpus's TRUE population paired difference, with correct
      CI coverage?) -- the paired-structure analogue of the single-sample
      check, sharing its PPIEffectResult/report/plot machinery. Entirely
      ADDITIVE to the three checks above (skipped with a warning, not a
      hard failure, if judge_bias_wmt_da_paired.csv hasn't been collected
      yet -- see --no-wmt-paired-check to disable explicitly).

Explicitly OUT of scope for v1 (see scoping discussion): a "real" (non-
null) group comparison via a natural categorical split (e.g. WMT by
language pair, App Store by app_id) -- deferred since it conflates a real
content difference with judge bias, a dirtier signal than the clean
synthetic power check -- and any 3-group/repeated/factorial/nested-run
structure (anova/friedman/kruskal/lmm), which would need 3+ judges treated
as a genuine repeated-measures design rather than an arbitrary combination.
"""

from __future__ import annotations

import argparse
import multiprocessing as _mp
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.stats as scipy_stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.tests import (
        _ppi_single_bootstrap_t,
        _ppi_single_wilson,
        _ppi_two_sample,
        _ppi_two_sample_midrank_corrected,
        _ppi_paired_arrays,
        _ppi_paired_bayes_bootstrap,
        _ppi_paired_bootstrap_t,
        _ppi_paired_tango,
        _p_x_gt_y_midrank,
    )

from ..methods import TTEST, TTEST_WELCH, MWU, MWU_MNAR_EXPERIMENTAL, WILCOXON, PAIRED_T, BAYES_BOOTSTRAP, BOOTSTRAP_T, TANGO
from ..scenarios.real_judge_bias import (
    REAL_JUDGE_BIAS_DATASETS,
    DEFAULT_DATA_DIR,
    RealJudgeBiasCorpus,
    load_real_judge_bias_corpus,
    default_size_grid,
    all_judge_pairs,
    generate_real_single_cell,
    generate_real_twogroup_null_cell,
    generate_real_paired_null_cell,
    RealWithinItemPairedCorpus,
    load_real_wmt_paired_corpus,
    generate_real_wmt_paired_bias_cell,
)
from .pvalues import (
    PPIResult,
    PPIEffectResult,
    print_ppi_report,
    print_ppi_effect_report,
    save_ppi_typeI_plot,
    save_ppi_effect_plot,
    save_results_artifacts_ppi,
    save_results_artifacts_ppi_effect,
    _effect_cell_stats,
    _uncorrected_bias_z,
    _uncorrected_bayes_bootstrap_paired_p_value,
    _uncorrected_bootstrap_t_paired_p_value,
    _uncorrected_tango_paired_p_value,
    _ProgressReporter,
    _ALPHA,
    _PPI_NONSTANDARD_TESTS,
)
from . import CaseResult

CASE_NAME = "ppi_real"

DEFAULT_LABEL_FRACS = [0.05, 0.10, 0.20, 0.40]
_SINGLE_METHOD_BOOTSTRAP_T = "bootstrap_t"
_SINGLE_METHOD_WILSON = "ppi_wilson"
"""Deliberately not "wilson" -- see methods.PPI_WILSON's docstring: that
name is already taken by ci_single.py's plain (non-PPI) Wilson CI."""


# ---------------------------------------------------------------------------
# Per-cell test batteries
# ---------------------------------------------------------------------------


def _run_real_single_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float, judge_model: str,
    n_reps: int, n_boot: int, seed: int,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """n_reps replicates of the single-sample bias/coverage check, capturing
    each method's (estimate, ci_low, ci_high, llm_estimate) per rep -- same
    tuple shape pvalues.py's _run_ppi_effect_cell collects, so
    _effect_cell_stats/_uncorrected_bias_z apply unchanged."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for _ in range(n_reps):
        judge, lab = generate_real_single_cell(corpus, rng, n, label_frac, judge_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if _SINGLE_METHOD_BOOTSTRAP_T in methods:
                try:
                    r = _ppi_single_bootstrap_t(judge, lab, _ALPHA, n_boot, int(rng.integers(0, 2 ** 31)))
                    out[_SINGLE_METHOD_BOOTSTRAP_T].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if _SINGLE_METHOD_WILSON in methods:
                try:
                    r = _ppi_single_wilson(judge, lab, _ALPHA)
                    out[_SINGLE_METHOD_WILSON].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

    return dict(out)


def _run_real_twogroup_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float, judge_model: str,
    n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the two-group Type-I null check (random-split
    real data), mirroring pvalues.py's _run_ppi_cell independent-groups
    branches (ttest/ttest_welch/mwu/mwu_mnar_experimental only -- see
    _run_real_paired_cell for the paired-samples family)."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        a, b, lab_a, lab_b = generate_real_twogroup_null_cell(corpus, rng, n, label_frac, judge_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if TTEST.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_ind(a, b, equal_var=True).pvalue)
                    uncorrected[TTEST.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(a, b, lab_a, lab_b, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    corrected[TTEST.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if TTEST_WELCH.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_ind(a, b, equal_var=False).pvalue)
                    uncorrected[TTEST_WELCH.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(a, b, lab_a, lab_b, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    corrected[TTEST_WELCH.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if MWU.name in methods:
                try:
                    p_u = float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                    uncorrected[MWU.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(a, b, lab_a, lab_b, lambda xa, ya: _p_x_gt_y_midrank(xa, ya) - 0.5, _ALPHA, n_boot, _rng_seed())
                    corrected[MWU.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if MWU_MNAR_EXPERIMENTAL.name in methods:
                try:
                    p_u = float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                    uncorrected[MWU_MNAR_EXPERIMENTAL.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample_midrank_corrected(a, b, lab_a, lab_b, _ALPHA, n_boot, _rng_seed())
                    corrected[MWU_MNAR_EXPERIMENTAL.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

    return corrected, uncorrected


def _run_real_paired_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the paired Type-I null check (same items,
    scored by two DIFFERENT judges) -- mirrors pvalues.py's _run_ppi_cell
    paired-groups branches (wilcoxon/paired_t/bayes_bootstrap/bootstrap_t/
    tango). See generate_real_paired_null_cell for why this is an exact
    null rather than merely an equal-in-distribution one."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        llm_x, llm_y, lab_x, lab_y = generate_real_paired_null_cell(corpus, rng, n, label_frac, judge_a, judge_b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if WILCOXON.name in methods:
                try:
                    p_u = float(scipy_stats.wilcoxon(llm_x, llm_y, alternative="two-sided").pvalue)
                    uncorrected[WILCOXON.name] += int(p_u < _ALPHA)
                    # rectifier_func MUST match the main statistic (median here, not
                    # mean) -- a mismatched rectifier is only unbiased when the
                    # population median equals the population mean of the paired
                    # judge-score diffs. When they diverge (common for real judge
                    # pairs), it introduces a FIXED bias that doesn't shrink with n
                    # while the SE does, so Type-I error climbs toward 100% as n/n_lab
                    # grow instead of converging to alpha. Root-caused on real wmt_da
                    # data (2026-07-23): |median-mean|/SD of the diffs predicted the
                    # blowup almost monotonically across 6 judge pairs (0.006 -> 2.5%
                    # Type-I, up to 0.26 -> 77.5%).
                    #
                    # Matching rectifier_func=np.median alone (same day) traded that
                    # for a WORSE problem on this same real data: percentile-
                    # bootstrapping a MEDIAN degenerates under real ties (92.6% of
                    # bootstrap replicate differences collapsed to exactly 0 on
                    # wmt_da), driving Type-I to ~0 (near-zero power) across the full
                    # sweep and NOT improving with n/n_lab (flat up to n_lab=400),
                    # since tie density is a property of the score scale, not sample
                    # size. evalstats.ppi.correct() now smooths the bootstrap with
                    # tiny sub-resolution jitter before each resample's median (see
                    # _tie_jitter_scale in evalstats/ppi.py) specifically to fix that
                    # degeneracy, which is what makes rectifier_func=np.median usable
                    # here -- re-verified via this same wmt_da paired check that both
                    # failure modes (bias AND degeneracy) are gone together.
                    r = _ppi_paired_arrays(llm_x, llm_y, lab_x, lab_y, np.median, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.median)
                    corrected[WILCOXON.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if PAIRED_T.name in methods:
                try:
                    p_u = float(scipy_stats.ttest_rel(llm_x, llm_y).pvalue)
                    uncorrected[PAIRED_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_arrays(llm_x, llm_y, lab_x, lab_y, np.mean, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.mean)
                    corrected[PAIRED_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if BAYES_BOOTSTRAP.name in methods:
                try:
                    p_u = _uncorrected_bayes_bootstrap_paired_p_value(llm_x - llm_y, n_boot, np.random.default_rng(_rng_seed()))
                    uncorrected[BAYES_BOOTSTRAP.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bayes_bootstrap(llm_x, llm_y, lab_x, lab_y, _ALPHA, n_boot, _rng_seed())
                    corrected[BAYES_BOOTSTRAP.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if BOOTSTRAP_T.name in methods:
                try:
                    p_u = _uncorrected_bootstrap_t_paired_p_value(llm_x - llm_y, n_boot, np.random.default_rng(_rng_seed()))
                    uncorrected[BOOTSTRAP_T.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_bootstrap_t(llm_x, llm_y, lab_x, lab_y, _ALPHA, n_boot, _rng_seed())
                    corrected[BOOTSTRAP_T.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

            if TANGO.name in methods:
                try:
                    p_u = _uncorrected_tango_paired_p_value(llm_x - llm_y)
                    uncorrected[TANGO.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_tango(llm_x, llm_y, lab_x, lab_y, _ALPHA)
                    corrected[TANGO.name] += int(r.p_value < _ALPHA)
                except Exception:
                    pass

    return corrected, uncorrected


_WMT_PAIRED_BIAS_METHODS = [WILCOXON.name, PAIRED_T.name, BAYES_BOOTSTRAP.name]
"""_paired_methods_for("continuous") minus BOOTSTRAP_T -- its test name
("bootstrap_t") is IDENTICAL to _SINGLE_METHOD_BOOTSTRAP_T, which the
single-sample bias/coverage check already uses for a totally different
estimand (a population MEAN, not a population MEAN PAIRED DIFFERENCE).
print_ppi_effect_report pools rows by test name only (not tag), so
including it here would silently merge two unrelated checks' bias/
coverage stats into one misleading "bootstrap_t" row. TANGO excluded for
the same reason it's excluded from _paired_methods_for("continuous")."""


def _run_real_wmt_paired_bias_cell(
    corpus: RealWithinItemPairedCorpus, methods: list[str], n: int, label_frac: float, judge_model: str,
    n_reps: int, n_boot: int, seed: int,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """n_reps replicates of the within-item paired bias/coverage check
    (genuine single-judge, two-condition paired data -- see
    scenarios/real_judge_bias.py's generate_real_wmt_paired_bias_cell),
    capturing each method's (estimate, ci_low, ci_high, llm_estimate) per
    rep -- same tuple shape _run_real_single_cell collects, so
    _effect_cell_stats/_uncorrected_bias_z apply unchanged. Unlike that
    function, the correct null value differs PER TEST (corpus.
    true_paired_median for wilcoxon's median estimand, corpus.
    true_paired_mean for the other three's mean estimand) -- see run()'s
    _consume for where that per-test split happens."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for _ in range(n_reps):
        llm_x, llm_y, lab_x, lab_y = generate_real_wmt_paired_bias_cell(corpus, rng, n, label_frac, judge_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if WILCOXON.name in methods:
                try:
                    r = _ppi_paired_arrays(llm_x, llm_y, lab_x, lab_y, np.median, _ALPHA, n_boot,
                                            int(rng.integers(0, 2 ** 31)), rectifier_func=np.median)
                    out[WILCOXON.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if PAIRED_T.name in methods:
                try:
                    r = _ppi_paired_arrays(llm_x, llm_y, lab_x, lab_y, np.mean, _ALPHA, n_boot,
                                            int(rng.integers(0, 2 ** 31)), rectifier_func=np.mean)
                    out[PAIRED_T.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            if BAYES_BOOTSTRAP.name in methods:
                try:
                    r = _ppi_paired_bayes_bootstrap(llm_x, llm_y, lab_x, lab_y, _ALPHA, n_boot,
                                                     int(rng.integers(0, 2 ** 31)))
                    out[BAYES_BOOTSTRAP.name].append((r.estimate, r.ci_low, r.ci_high, r.llm_estimate))
                except Exception:
                    pass

            # BOOTSTRAP_T deliberately NOT included here -- see
            # _WMT_PAIRED_BIAS_METHODS' docstring (its test name collides
            # with the single-sample check's _SINGLE_METHOD_BOOTSTRAP_T,
            # which would silently pool two different estimands' stats
            # together in print_ppi_effect_report). Not wired up at all,
            # rather than left reachable-but-unused, since nothing calls
            # this function with a methods list that would ever hit it.

    return dict(out)


_REAL_CORPORA: list[RealJudgeBiasCorpus] = []
"""Set by run() right before creating the worker Pool (fork context), so
worker processes inherit it via copy-on-write instead of the (potentially
large) corpus arrays being re-pickled through the task queue on every
work item -- same pattern as pvalues.py's _PAIRWISE_SOURCES /
_MULTIARM_SOURCES globals. Only meaningful in the child processes; the
parent addresses corpora directly."""

_REAL_WMT_PAIRED_CORPORA: list[RealWithinItemPairedCorpus] = []
"""Same fork-inheritance pattern as _REAL_CORPORA, kept as a SEPARATE list
(rather than folded into _REAL_CORPORA) since RealWithinItemPairedCorpus is
a different type loaded from a different file -- there's exactly one
wmt_da_paired corpus (index 0) when present, zero when the data hasn't
been collected yet (see run()'s try/except around load_real_wmt_paired_corpus)."""


def _run_ppi_real_cell_worker(args: tuple) -> dict:
    """Runs ONE (single|twogroup|paired|wmt_paired_bias) cell and returns
    everything run() needs to fold it into effect_results/twogroup_results/
    paired_results, with no dependency on the original work-item's position
    -- required since pool.imap_unordered does not preserve submission order."""
    check_type, corpus_idx, name, dataset, methods, n, label_frac, judge_or_pair, n_reps, n_boot, seed = args

    if check_type == "wmt_paired_bias":
        corpus = _REAL_WMT_PAIRED_CORPORA[corpus_idx]
        samples_by_test = _run_real_wmt_paired_bias_cell(corpus, methods, n, label_frac, judge_or_pair, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n,
            "true_paired_mean": corpus.true_paired_mean, "true_paired_median": corpus.true_paired_median,
            "samples_by_test": samples_by_test,
        }

    corpus = _REAL_CORPORA[corpus_idx]

    if check_type == "single":
        samples_by_test = _run_real_single_cell(corpus, methods, n, label_frac, judge_or_pair, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n,
            "corpus_mean": corpus.corpus_mean, "samples_by_test": samples_by_test,
        }
    if check_type == "twogroup":
        corrected, uncorrected = _run_real_twogroup_cell(corpus, methods, n, label_frac, judge_or_pair, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n, "n_reps": n_reps,
            "methods": methods, "corrected": corrected, "uncorrected": uncorrected,
        }
    judge_a, judge_b = judge_or_pair
    corrected, uncorrected = _run_real_paired_cell(corpus, methods, n, label_frac, judge_a, judge_b, n_reps, n_boot, seed)
    return {
        "check_type": check_type, "name": name, "dataset": dataset, "n": n, "n_reps": n_reps,
        "methods": methods, "corrected": corrected, "uncorrected": uncorrected,
    }


def _single_methods_for(eval_type: str) -> list[str]:
    return [_SINGLE_METHOD_BOOTSTRAP_T] + ([_SINGLE_METHOD_WILSON] if eval_type == "binary" else [])


def _twogroup_methods_for(eval_type: str) -> list[str]:
    # MWU/MWU_MNAR_EXPERIMENTAL (rank-based) aren't in pvalues.py's
    # _PPI_BINARY_COMPATIBLE_TESTS -- binary's massive ties break the
    # rank-based judge-bias noise model there, same restriction applies here.
    base = [TTEST.name, TTEST_WELCH.name]
    return base if eval_type == "binary" else base + [MWU.name, MWU_MNAR_EXPERIMENTAL.name]


def _has_standard_test(results: list) -> bool:
    return any(r.test not in _PPI_NONSTANDARD_TESTS for r in results)


def _has_nonstandard_test(results: list) -> bool:
    return any(r.test in _PPI_NONSTANDARD_TESTS for r in results)


def _paired_methods_for(eval_type: str) -> list[str]:
    # Same _PPI_BINARY_COMPATIBLE_TESTS restriction, intersected with the
    # paired-samples family: {ttest, ttest_welch, paired_t, bayes_bootstrap,
    # tango} ∩ {wilcoxon, paired_t, bayes_bootstrap, bootstrap_t, tango}.
    if eval_type == "binary":
        return [PAIRED_T.name, BAYES_BOOTSTRAP.name, TANGO.name]
    return [WILCOXON.name, PAIRED_T.name, BAYES_BOOTSTRAP.name, BOOTSTRAP_T.name]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                         help="Directory containing judge_bias_<dataset>.csv (from collect_judge_bias_data.py).")
    parser.add_argument("--datasets", choices=list(REAL_JUDGE_BIAS_DATASETS), nargs="+", default=None,
                         help="Which real datasets to check (default: all three; a dataset with no "
                              "judge_score data yet is skipped with a warning, not a hard failure).")
    parser.add_argument("--judge-models", nargs="+", default=None,
                         help="Which judge models to use (default: auto-discover every judge model "
                              "collected for each dataset, then apply --min-judge-coverage). Passing "
                              "this explicitly is trusted as-is, with NO coverage filtering -- only "
                              "items scored by EVERY selected judge are used either way.")
    parser.add_argument("--min-judge-coverage", type=float, default=0.9,
                         help="Auto-discovery only (ignored if --judge-models is given explicitly): "
                              "drop any judge whose distinct-item coverage of the dataset is below this "
                              "fraction (default 0.9) before aligning. Alignment takes the INTERSECTION "
                              "of items across every selected judge, so one incompletely-collected judge "
                              "(e.g. a --limit'd or still-running collect-judge-scores) otherwise drags "
                              "the usable corpus size down for every OTHER judge too, not just itself.")
    parser.add_argument("--max-pairs", type=int, default=None,
                         help="Cap on unique judge PAIRS checked by the paired-null test (default: all "
                              "C(k, 2) pairs for k judge models -- e.g. 28 for 8 judges. Caps by random "
                              "subset if k is large enough that C(k, 2) would be impractically slow.")
    parser.add_argument("--label-fracs", type=float, nargs="+", default=DEFAULT_LABEL_FRACS)
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                         help="Sample sizes per rep (default: 3 sizes bounded by each dataset's actual corpus size).")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--ppi-n-boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--progress", choices=["bar", "cell", "off"], default="bar")
    parser.add_argument("--save-results", choices=["save", "none"], default="save")
    parser.add_argument("--plots", choices=["save", "none"], default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--no-single-check", action="store_true", help="Skip the single-sample bias/coverage check.")
    parser.add_argument("--no-twogroup-check", action="store_true", help="Skip the two-group Type-I null check.")
    parser.add_argument("--no-paired-check", action="store_true", help="Skip the cross-judge paired Type-I null check.")
    parser.add_argument("--no-wmt-paired-check", action="store_true",
                         help="Skip the within-item (wmt_da_paired) bias/coverage check. Runs by default "
                              "IF judge_bias_wmt_da_paired.csv has been collected (see "
                              "collect_judge_bias_data.py --types continuous_paired); skipped with a "
                              "warning, not a hard failure, if it hasn't.")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N",
                         help="Parallel worker processes (default: cpu_count-1; 1=sequential). Parallelizes "
                              "across single/twogroup/paired cells (all corpora, label_fracs, sizes, judge "
                              "models/pairs flattened into one work queue) using a multiprocessing.Pool "
                              "with the fork start method, same as pvalues.py's run_*_simulation functions. "
                              "Cell seeds are fixed up front from --seed regardless of --workers, so "
                              "results are identical either way -- only completion order (and hence "
                              "progress-bar interleaving) differs.")


def official_args(base_seed: int = 46) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=DEFAULT_DATA_DIR, datasets=None, judge_models=None, min_judge_coverage=0.9, max_pairs=None,
        label_fracs=list(DEFAULT_LABEL_FRACS), sizes=None, reps=200, ppi_n_boot=2000,
        alpha=0.05, seed=base_seed, progress="bar", save_results="save", plots="save",
        out_dir="simulations/out", plots_dir=None,
        no_single_check=False, no_twogroup_check=False, no_paired_check=False, no_wmt_paired_check=False,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def official_variants(base_seed: int = 46) -> list[tuple[str, argparse.Namespace]]:
    return [("real judge-bias data (single-sample + two-group + paired null + within-item paired bias/coverage)", official_args(base_seed))]


def quick_args(base_seed: int = 47, data_source: str = "synthetic") -> argparse.Namespace:
    """`data_source` accepted only for --quick-test's uniform
    (module.quick_args(), module.quick_args(data_source="real")) calling
    convention -- this case has no synthetic variant of its own (it's
    always real data), so both calls run the identical fast preset."""
    return argparse.Namespace(
        data_dir=DEFAULT_DATA_DIR, datasets=None, judge_models=None, min_judge_coverage=0.9, max_pairs=10,
        label_fracs=[0.20], sizes=None, reps=5, ppi_n_boot=200,
        alpha=0.05, seed=base_seed, progress="bar", save_results="save", plots="save",
        out_dir="simulations/out", plots_dir=None,
        no_single_check=False, no_twogroup_check=False, no_paired_check=False, no_wmt_paired_check=False,
        latex=True, workers=1,
    )


def run(args: argparse.Namespace) -> CaseResult:
    t0 = time.time()
    try:
        plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_paths: list[str] = []
        key_metrics: dict = {}

        requested = args.datasets or list(REAL_JUDGE_BIAS_DATASETS)
        corpora: list[RealJudgeBiasCorpus] = []
        for ds in requested:
            try:
                corpus = load_real_judge_bias_corpus(
                    ds, data_dir=args.data_dir, judge_models=args.judge_models,
                    min_coverage=getattr(args, "min_judge_coverage", 0.9),
                )
                corpora.append(corpus)
                print(f"  Loaded {ds}: N={corpus.corpus_size}, judge_models={corpus.judge_models}, "
                      f"corpus_mean={corpus.corpus_mean:.4f}")
            except (FileNotFoundError, ValueError) as e:
                print(f"  Skipping {ds}: {e}")

        wmt_paired_corpus: RealWithinItemPairedCorpus | None = None
        if not getattr(args, "no_wmt_paired_check", False):
            try:
                wmt_paired_corpus = load_real_wmt_paired_corpus(
                    data_dir=args.data_dir, judge_models=args.judge_models,
                    min_coverage=getattr(args, "min_judge_coverage", 0.9),
                )
                print(f"  Loaded wmt_da_paired: N={wmt_paired_corpus.corpus_size} pairs, "
                      f"judge_models={wmt_paired_corpus.judge_models}, "
                      f"true_paired_mean={wmt_paired_corpus.true_paired_mean:.4f}, "
                      f"true_paired_median={wmt_paired_corpus.true_paired_median:.4f}")
            except (FileNotFoundError, ValueError) as e:
                print(f"  Skipping wmt_da_paired (within-item paired check): {e}")

        if not corpora and wmt_paired_corpus is None:
            raise ValueError(
                "No real judge-bias datasets with collected scores found under "
                f"{args.data_dir!r}. Run simulations/collect_judge_bias_data.py's "
                "collect-data then collect-judge-scores first."
            )

        effect_results: list[PPIEffectResult] = []
        twogroup_results: list[PPIResult] = []
        paired_results: list[PPIResult] = []
        seed_counter = args.seed

        # Flatten every (corpus, label_frac, size, judge_model/pair, check
        # type) cell into one work list up front, in the SAME nested order
        # the old inline loop used to run them in, so seed_counter assigns
        # the identical seed to the identical cell regardless of --workers
        # -- results are therefore reproducible for a given --seed whether
        # this runs serially or in parallel; only completion order (and
        # hence progress-bar interleaving) can differ.
        work_items: list[tuple] = []

        for corpus_idx, corpus in enumerate(corpora):
            sizes = args.sizes or default_size_grid(corpus.corpus_size)
            single_methods = _single_methods_for(corpus.eval_type)
            twogroup_methods = _twogroup_methods_for(corpus.eval_type)
            paired_methods = _paired_methods_for(corpus.eval_type)
            pairs = all_judge_pairs(corpus, max_pairs=args.max_pairs, rng=np.random.default_rng(args.seed))

            for label_frac in args.label_fracs:
                for n in sizes:
                    for judge_model in corpus.judge_models:
                        name = f"real.{corpus.dataset}.labfrac={label_frac:.2f}.n={n}.judge={judge_model}"

                        if not args.no_single_check:
                            seed_counter += 1
                            work_items.append((
                                "single", corpus_idx, name, corpus.dataset, single_methods,
                                n, label_frac, judge_model, args.reps, args.ppi_n_boot, seed_counter,
                            ))

                        if not args.no_twogroup_check:
                            seed_counter += 1
                            work_items.append((
                                "twogroup", corpus_idx, name, corpus.dataset, twogroup_methods,
                                n, label_frac, judge_model, args.reps, args.ppi_n_boot, seed_counter,
                            ))

                    if not args.no_paired_check:
                        for judge_a, judge_b in pairs:
                            seed_counter += 1
                            name = f"real.{corpus.dataset}.labfrac={label_frac:.2f}.n={n}.pair={judge_a}~{judge_b}"
                            work_items.append((
                                "paired", corpus_idx, name, corpus.dataset, paired_methods,
                                n, label_frac, (judge_a, judge_b), args.reps, args.ppi_n_boot, seed_counter,
                            ))

        if wmt_paired_corpus is not None:
            _REAL_WMT_PAIRED_CORPORA.append(wmt_paired_corpus)
            wmt_paired_idx = 0
            sizes = args.sizes or default_size_grid(wmt_paired_corpus.corpus_size)
            for label_frac in args.label_fracs:
                for n in sizes:
                    for judge_model in wmt_paired_corpus.judge_models:
                        seed_counter += 1
                        name = f"real.{wmt_paired_corpus.dataset}.labfrac={label_frac:.2f}.n={n}.judge={judge_model}"
                        work_items.append((
                            "wmt_paired_bias", wmt_paired_idx, name, wmt_paired_corpus.dataset,
                            _WMT_PAIRED_BIAS_METHODS, n, label_frac, judge_model,
                            args.reps, args.ppi_n_boot, seed_counter,
                        ))

        def _consume(result: dict) -> None:
            ct = result["check_type"]
            if ct == "single":
                for t, samples in result["samples_by_test"].items():
                    bias_mean, z, coverage, mean_width, n_ok = _effect_cell_stats(samples, result["corpus_mean"])
                    if n_ok == 0:
                        continue
                    unc_z = _uncorrected_bias_z(samples, result["corpus_mean"])
                    effect_results.append(PPIEffectResult(
                        name=result["name"], tag=f"real_{result['dataset']}", test=t, n=result["n"], n_samples=n_ok,
                        null_value=result["corpus_mean"], mean_bias=bias_mean, bias_z=z,
                        coverage=coverage, mean_ci_width=mean_width, uncorrected_bias_z=unc_z,
                    ))
            elif ct == "wmt_paired_bias":
                # Same PPIEffectResult shape as "single" above (reusing its
                # report/plot machinery unmodified), but the correct null
                # value differs PER TEST -- wilcoxon targets the population
                # MEDIAN paired difference, the other three target the MEAN
                # (see _run_real_wmt_paired_bias_cell's docstring).
                for t, samples in result["samples_by_test"].items():
                    null_value = (result["true_paired_median"] if t == WILCOXON.name
                                  else result["true_paired_mean"])
                    bias_mean, z, coverage, mean_width, n_ok = _effect_cell_stats(samples, null_value)
                    if n_ok == 0:
                        continue
                    unc_z = _uncorrected_bias_z(samples, null_value)
                    effect_results.append(PPIEffectResult(
                        name=result["name"], tag=f"real_{result['dataset']}", test=t, n=result["n"], n_samples=n_ok,
                        null_value=null_value, mean_bias=bias_mean, bias_z=z,
                        coverage=coverage, mean_ci_width=mean_width, uncorrected_bias_z=unc_z,
                    ))
            else:
                bucket = twogroup_results if ct == "twogroup" else paired_results
                for t in result["methods"]:
                    bucket.append(PPIResult(
                        name=result["name"], tag=f"real_{result['dataset']}", test=t, n_reps=result["n_reps"],
                        corrected_rejects=result["corrected"][t], uncorrected_rejects=result["uncorrected"][t],
                        n=result["n"],
                    ))

        reporter = _ProgressReporter(max(len(work_items), 1), mode=args.progress, label="ppi_real")
        n_workers = max(1, getattr(args, "workers", 1))

        global _REAL_CORPORA
        _REAL_CORPORA = corpora

        if n_workers <= 1:
            for i, item in enumerate(work_items):
                result = _run_ppi_real_cell_worker(item)
                _consume(result)
                reporter.update(i + 1, detail=f"{result['dataset']} {result['check_type']}")
        else:
            ctx = _mp.get_context("fork")
            with ctx.Pool(n_workers) as pool:
                for i, result in enumerate(pool.imap_unordered(_run_ppi_real_cell_worker, work_items)):
                    _consume(result)
                    reporter.update(i + 1, detail=f"{result['dataset']} {result['check_type']}")

        reporter.update(max(len(work_items), 1), detail="done")

        if effect_results:
            print_ppi_effect_report(effect_results, alpha=args.alpha)
            run_stem = f"ppi_real_effect_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_ppi_effect(
                    results=effect_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=args.latex,
                )
            if args.plots == "save":
                # bootstrap_t is classified "nonstandard" by pvalues.py's own
                # convention (_PPI_NONSTANDARD_TESTS); ppi_wilson (single-
                # sample, closed-form) isn't, and both are frequently present
                # together for binary datasets (arena) -- so, exactly like
                # pvalues.py's own run() does for its main PPI mode, this
                # calls save_ppi_effect_plot twice rather than once, so
                # bootstrap_t doesn't silently vanish from a single combined
                # plot (see save_ppi_effect_plot's nonstandard= filtering).
                # Each call is guarded by whether ITS bucket is non-empty --
                # a non-binary dataset's single-sample check has ONLY
                # bootstrap_t (no ppi_wilson), which is entirely
                # "nonstandard", so the plain (nonstandard=False) call would
                # otherwise get zero tests and crash building an empty legend.
                if _has_standard_test(effect_results):
                    plot_path = save_ppi_effect_plot(
                        results=effect_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_bias_coverage_width.png"),
                    )
                    output_paths.append(plot_path)
                    print(f"Saved plot: {plot_path}")
                if _has_nonstandard_test(effect_results):
                    nonstd_plot_path = save_ppi_effect_plot(
                        results=effect_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_bias_coverage_width_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_plot_path)
                    print(f"Saved plot: {nonstd_plot_path}")
            key_metrics["ppi_real_effect_n_results"] = len(effect_results)
            finite_z = [r.bias_z for r in effect_results if np.isfinite(r.bias_z)]
            key_metrics["ppi_real_effect_mean_abs_bias_z"] = float(np.mean(np.abs(finite_z))) if finite_z else float("nan")
            finite_cov = [r.coverage for r in effect_results if np.isfinite(r.coverage)]
            key_metrics["ppi_real_effect_mean_coverage"] = float(np.mean(finite_cov)) if finite_cov else float("nan")

        if twogroup_results:
            print_ppi_report(twogroup_results, alpha=args.alpha)
            run_stem = f"ppi_real_twogroup_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_ppi(
                    results=twogroup_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=args.latex,
                )
            if args.plots == "save":
                plot_path = save_ppi_typeI_plot(
                    results=twogroup_results, alpha=args.alpha,
                    out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected.png"),
                )
                output_paths.append(plot_path)
                print(f"Saved plot: {plot_path}")
            c_tot = sum(r.corrected_rejects for r in twogroup_results)
            u_tot = sum(r.uncorrected_rejects for r in twogroup_results)
            n_tot = sum(r.n_reps for r in twogroup_results)
            key_metrics["ppi_real_twogroup_n_results"] = len(twogroup_results)
            key_metrics["ppi_real_twogroup_mean_corrected_type1"] = float(c_tot / n_tot) if n_tot else float("nan")
            key_metrics["ppi_real_twogroup_mean_uncorrected_type1"] = float(u_tot / n_tot) if n_tot else float("nan")

        if paired_results:
            print_ppi_report(paired_results, alpha=args.alpha)
            run_stem = f"ppi_real_paired_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_ppi(
                    results=paired_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=args.latex,
                )
            if args.plots == "save":
                # Same empty-bucket guard as the effect plot above -- a
                # binary dataset's paired methods are {paired_t,
                # bayes_bootstrap, tango}, i.e. only ONE standard test
                # (paired_t) but TWO nonstandard ones, so neither bucket is
                # reliably non-empty across eval_types.
                if _has_standard_test(paired_results):
                    plot_path = save_ppi_typeI_plot(
                        results=paired_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected.png"),
                    )
                    output_paths.append(plot_path)
                    print(f"Saved plot: {plot_path}")
                if _has_nonstandard_test(paired_results):
                    nonstd_plot_path = save_ppi_typeI_plot(
                        results=paired_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_plot_path)
                    print(f"Saved plot: {nonstd_plot_path}")
            c_tot = sum(r.corrected_rejects for r in paired_results)
            u_tot = sum(r.uncorrected_rejects for r in paired_results)
            n_tot = sum(r.n_reps for r in paired_results)
            key_metrics["ppi_real_paired_n_results"] = len(paired_results)
            key_metrics["ppi_real_paired_mean_corrected_type1"] = float(c_tot / n_tot) if n_tot else float("nan")
            key_metrics["ppi_real_paired_mean_uncorrected_type1"] = float(u_tot / n_tot) if n_tot else float("nan")

        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics=key_metrics, duration_s=time.time() - t0,
        )
    except Exception as e:
        return CaseResult(case_name=CASE_NAME, status="error", error=str(e), duration_s=time.time() - t0)
