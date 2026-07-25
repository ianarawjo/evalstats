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

Six checks:

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

  two-group Type-I null (random split, cross-judge)
      For every unique PAIR of judge models: bisect the corpus into two
      disjoint random subsamples -- a valid null by construction (both are
      random draws from the identical population, so their TRUE, human-
      label means are equal) -- but read group A through judge_a and group
      B through judge_b, not the same judge reading both. Runs the
      independent-samples tests (ttest/ttest_welch/mwu/
      mwu_mnar_experimental) PPI correction is supposed to keep calibrated,
      with real noise/skew/judge-bias characteristics instead of synthetic
      ones. Cross-judge is a deliberate redesign (2026-07-23) from an
      original same-judge-reads-both version: since a single judge reading
      two random halves of the SAME population applies its bias identically
      to both, the bias cancels out of the A-vs-B difference regardless of
      its magnitude -- making "uncorrected" (a classical test on raw judge
      scores, no human labels) structurally unable to fail, which is
      uninformative about whether skipping PPI correction is actually
      risky. Two DIFFERENT judges reintroduces a genuine asymmetry
      (confirmed on real data: judges commonly differ by several
      percentage points of mean bias on a [0,1]-rescaled scale) an
      uncorrected test IS vulnerable to -- see
      scenarios/real_judge_bias.py's generate_real_twogroup_null_cell for
      the full reasoning and the specific real-data evidence that
      motivated it.

  paired Type-I null (cross-judge)
      For every unique PAIR of judge models scoring the SAME items: an
      EXACT null (stronger than the two-group check's "equal in
      distribution" -- here the true paired difference is exactly zero for
      every item, since both judges are noisy/biased reads of the
      IDENTICAL human_label). Runs the paired-samples tests (wilcoxon/
      paired_t/bayes_bootstrap/bootstrap_t/tango) that a single judge model
      alone can't exercise at all. Already cross-judge from the start (see
      the two-group check above, which was redesigned to match this one).
      With k judge models collected, all C(k, 2) pairs are checked (capped
      by --max-pairs) -- more judges means more independent (dataset,
      label_frac, n, pair) cells feeding the SAME calibration question,
      i.e. more power to catch a miscalibration than any single pair would
      give alone.

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

  omnibus-independent Type-I null (3-group, cross-judge)
      The 3-group generalization of the two-group check: for every unique
      TRIPLE of judge models, bisect the corpus into three disjoint random
      subsamples read through three DIFFERENT judges -- same cross-judge
      reasoning as the two-group check (see generate_real_
      omnibus_independent_null_cell). Runs anova_ind and kruskal (NOT
      kruskal_mnar_experimental -- see _omnibus_independent_methods_for's
      docstring for why, same reasoning as dropping mwu_mnar_experimental
      from the two-group check). With k judge models, all C(k, 3) triples
      are checked (capped by --max-triples).

  omnibus-repeated Type-I null (3-condition, cross-judge)
      The 3-condition generalization of the paired check: for every unique
      TRIPLE of judge models scoring the SAME items, an EXACT null (the
      true value is identical across all three conditions for every item).
      Runs anova_rep and friedman -- the repeated-measures tests a single
      judge model, or even a pair, can't exercise (see generate_real_
      omnibus_repeated_null_cell).

Explicitly OUT of scope for v1 (see scoping discussion): a "real" (non-
null) group comparison via a natural categorical split (e.g. WMT by
language pair, App Store by app_id) -- deferred since it conflates a real
content difference with judge bias, a dirtier signal than the clean
synthetic power check -- and LMM/LMM_FACTORIAL/LMM_RUNS specifically
(deferred for now, unlike the other omnibus tests above).
"""

from __future__ import annotations

import argparse
import multiprocessing as _mp
import os
import re
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
        _ppi_anova_independent_p_value,
        _ppi_anova_repeated_p_value,
        _ppi_friedman_p_value,
        _ppi_kruskal_wallis_pairwise,
    )

from ..methods import (
    TTEST, TTEST_WELCH, MWU, MWU_MNAR_EXPERIMENTAL, WILCOXON, PAIRED_T, BAYES_BOOTSTRAP, BOOTSTRAP_T, TANGO,
    ANOVA_IND, ANOVA_REP, FRIEDMAN, KRUSKAL, PPI_TEST_METHODS, get_method_color,
)
from ..scenarios.real_judge_bias import (
    REAL_JUDGE_BIAS_DATASETS,
    DEFAULT_DATA_DIR,
    RealJudgeBiasCorpus,
    load_real_judge_bias_corpus,
    default_size_grid,
    all_judge_pairs,
    all_judge_triples,
    generate_real_single_cell,
    generate_real_twogroup_null_cell,
    generate_real_paired_null_cell,
    generate_real_omnibus_independent_null_cell,
    generate_real_omnibus_repeated_null_cell,
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
    _uncorrected_anova_independent_p_value,
    _uncorrected_anova_repeated_p_value,
    _uncorrected_friedman_p_value,
    _uncorrected_kruskal_p_value,
    _ProgressReporter,
    _ALPHA,
    _PPI_NONSTANDARD_TESTS,
    _ppi_wilson_interval,
    _alpha_label,
    _pretty_test,
    _metric_pct,
    _alignment_bucket,
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
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the two-group Type-I null check (random-split
    real data, cross-judge -- see generate_real_twogroup_null_cell's
    docstring for why group A and group B are read through two DIFFERENT
    judges, not the same one), mirroring pvalues.py's _run_ppi_cell
    independent-groups branches (ttest/ttest_welch/mwu/
    mwu_mnar_experimental only -- see _run_real_paired_cell for the
    paired-samples family)."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        a, b, lab_a, lab_b = generate_real_twogroup_null_cell(corpus, rng, n, label_frac, judge_a, judge_b)
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


def _run_real_omnibus_independent_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, judge_c: str, n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the omnibus-independent Type-I null check (3-way
    random split, cross-judge -- see generate_real_omnibus_independent_
    null_cell's docstring), mirroring pvalues.py's _run_ppi_cell's
    ANOVA_IND/KRUSKAL branches. See _omnibus_independent_methods_for for
    why KRUSKAL_MNAR_EXPERIMENTAL is not wired up here."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        groups, groups_lab = generate_real_omnibus_independent_null_cell(
            corpus, rng, n, label_frac, judge_a, judge_b, judge_c,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if ANOVA_IND.name in methods:
                try:
                    p_u = _uncorrected_anova_independent_p_value(groups)
                    uncorrected[ANOVA_IND.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_independent_p_value(groups, groups_lab, k=len(groups))
                    corrected[ANOVA_IND.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    pass

            if KRUSKAL.name in methods:
                try:
                    p_u = _uncorrected_kruskal_p_value(groups)
                    uncorrected[KRUSKAL.name] += int(p_u < _ALPHA)
                    pw = _ppi_kruskal_wallis_pairwise(groups, groups_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    corrected[KRUSKAL.name] += int(pw["wald_p"] < _ALPHA)
                except Exception:
                    pass

    return corrected, uncorrected


def _run_real_omnibus_repeated_cell(
    corpus: RealJudgeBiasCorpus, methods: list[str], n: int, label_frac: float,
    judge_a: str, judge_b: str, judge_c: str, n_reps: int, n_boot: int, seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """n_reps replicates of the omnibus-repeated Type-I null check (same
    items, 3 different judges -- see generate_real_omnibus_repeated_
    null_cell's docstring), mirroring pvalues.py's _run_ppi_cell's
    ANOVA_REP/FRIEDMAN branches."""
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in methods}
    uncorrected: dict[str, int] = {t: 0 for t in methods}

    for _ in range(n_reps):
        groups, groups_lab = generate_real_omnibus_repeated_null_cell(
            corpus, rng, n, label_frac, judge_a, judge_b, judge_c,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if ANOVA_REP.name in methods:
                try:
                    p_u = _uncorrected_anova_repeated_p_value(groups)
                    uncorrected[ANOVA_REP.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_repeated_p_value(groups, groups_lab, k=len(groups))
                    corrected[ANOVA_REP.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    pass

            if FRIEDMAN.name in methods:
                try:
                    p_u = _uncorrected_friedman_p_value(groups)
                    uncorrected[FRIEDMAN.name] += int(p_u < _ALPHA)
                    p = _ppi_friedman_p_value(groups, groups_lab, k=len(groups))
                    corrected[FRIEDMAN.name] += int(p is not None and p < _ALPHA)
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
    """Runs ONE (single|twogroup|paired|omnibus_independent|omnibus_repeated|
    wmt_paired_bias) cell and returns everything run() needs to fold it into
    effect_results/twogroup_results/paired_results/omnibus_results, with no
    dependency on the original work-item's position -- required since
    pool.imap_unordered does not preserve submission order."""
    check_type, corpus_idx, name, dataset, methods, n, label_frac, judge_or_group, n_reps, n_boot, seed = args

    if check_type == "wmt_paired_bias":
        corpus = _REAL_WMT_PAIRED_CORPORA[corpus_idx]
        samples_by_test = _run_real_wmt_paired_bias_cell(corpus, methods, n, label_frac, judge_or_group, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n,
            "true_paired_mean": corpus.true_paired_mean, "true_paired_median": corpus.true_paired_median,
            "samples_by_test": samples_by_test,
        }

    corpus = _REAL_CORPORA[corpus_idx]

    if check_type == "single":
        samples_by_test = _run_real_single_cell(corpus, methods, n, label_frac, judge_or_group, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n,
            "corpus_mean": corpus.corpus_mean, "samples_by_test": samples_by_test,
        }

    if check_type in ("omnibus_independent", "omnibus_repeated"):
        judge_a, judge_b, judge_c = judge_or_group
        runner = _run_real_omnibus_independent_cell if check_type == "omnibus_independent" else _run_real_omnibus_repeated_cell
        corrected, uncorrected = runner(corpus, methods, n, label_frac, judge_a, judge_b, judge_c, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n, "n_reps": n_reps,
            "methods": methods, "corrected": corrected, "uncorrected": uncorrected,
        }

    judge_a, judge_b = judge_or_group
    if check_type == "twogroup":
        corrected, uncorrected = _run_real_twogroup_cell(corpus, methods, n, label_frac, judge_a, judge_b, n_reps, n_boot, seed)
        return {
            "check_type": check_type, "name": name, "dataset": dataset, "n": n, "n_reps": n_reps,
            "methods": methods, "corrected": corrected, "uncorrected": uncorrected,
        }
    corrected, uncorrected = _run_real_paired_cell(corpus, methods, n, label_frac, judge_a, judge_b, n_reps, n_boot, seed)
    return {
        "check_type": check_type, "name": name, "dataset": dataset, "n": n, "n_reps": n_reps,
        "methods": methods, "corrected": corrected, "uncorrected": uncorrected,
    }


def _single_methods_for(eval_type: str) -> list[str]:
    return [_SINGLE_METHOD_BOOTSTRAP_T] + ([_SINGLE_METHOD_WILSON] if eval_type == "binary" else [])


def _twogroup_methods_for(eval_type: str) -> list[str]:
    # MWU (rank-based) isn't in pvalues.py's _PPI_BINARY_COMPATIBLE_TESTS --
    # binary's massive ties break the rank-based judge-bias noise model
    # there, same restriction applies here.
    #
    # MWU_MNAR_EXPERIMENTAL (the local-rectifier MWU variant) deliberately
    # NOT included (as of 2026-07-24) -- it exists specifically to trade
    # some MCAR calibration for MNAR robustness (see methods.py's
    # MWU_MNAR_EXPERIMENTAL comment for the full history), but this check's
    # real-data labeling is MCAR by construction (generate_real_twogroup_
    # null_cell's _reveal_labels call), so there's no MNAR risk here for the
    # local rectifier to buy anything against -- it would only ever look
    # worse than plain MWU on this check, never better, for reasons that
    # have nothing to do with either method's actual quality.
    base = [TTEST.name, TTEST_WELCH.name]
    return base if eval_type == "binary" else base + [MWU.name]


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


def _omnibus_independent_methods_for(eval_type: str) -> list[str]:
    # anova_ind/kruskal/kruskal_mnar_experimental aren't in pvalues.py's
    # _PPI_BINARY_COMPATIBLE_TESTS -- binary's massive ties break these
    # rank/variance-based judge-bias models entirely, same restriction as
    # the twogroup/paired families.
    #
    # KRUSKAL_MNAR_EXPERIMENTAL deliberately NOT included, for the identical
    # reason MWU_MNAR_EXPERIMENTAL was dropped from the twogroup check (see
    # _twogroup_methods_for's docstring): it trades some MCAR calibration
    # for MNAR robustness, but this check's real-data labeling is MCAR by
    # construction, so there's no MNAR risk here for its local rectifier to
    # buy anything against.
    if eval_type == "binary":
        return []
    return [ANOVA_IND.name, KRUSKAL.name]


def _omnibus_repeated_methods_for(eval_type: str) -> list[str]:
    # Same binary restriction as _omnibus_independent_methods_for --
    # anova_rep/friedman are equally rank/variance-based.
    if eval_type == "binary":
        return []
    return [ANOVA_REP.name, FRIEDMAN.name]


# ---------------------------------------------------------------------------
# Extra plots for hypothesis_results (twogroup + paired + omnibus): a
# dataset x label_frac heatmap per test, and a judge-alignment-bucketed bar
# chart per test -- real-data analogues of pvalues.py's save_ppi_nlab_grid_
# plot / save_ppi_alignment_sweep_plot, adapted to what real data actually
# varies (there's no synthetic bias_magnitude/label_mechanism/effect_size
# knob here, just which real dataset/label_frac/judges a cell used).
# ---------------------------------------------------------------------------

_PPI_REAL_LABFRAC_N_RE = re.compile(r"\.labfrac=([0-9.]+)\.n=(\d+)")
_PPI_REAL_JUDGES_RE = re.compile(r"\.(?:judge|pair|triple)=(.+)$")


def _parse_real_cell_labfrac_n(name: str) -> tuple[float, int]:
    """`name` is one of run()'s work_items names, e.g.
    "real.arena.labfrac=0.20.n=100.pair=judge_a~judge_b" -- pulls
    (label_frac, n) back out for the heatmap's axes."""
    m = _PPI_REAL_LABFRAC_N_RE.search(name)
    if not m:
        raise ValueError(f"Unrecognized ppi_real cell name: {name!r}")
    return float(m.group(1)), int(m.group(2))


def _parse_real_cell_judges(name: str) -> list[str]:
    """The 1 (single), 2 (twogroup/paired), or 3 (omnibus) judge_model
    names a cell drew on, in the order they were assigned to
    group/condition A/B/C -- everything after the final .judge=/.pair=/
    .triple= marker, split on "~" (see run()'s name-building f-strings)."""
    m = _PPI_REAL_JUDGES_RE.search(name)
    if not m:
        return []
    return m.group(1).split("~")


def save_ppi_real_labfrac_dataset_heatmap(*, results: list[PPIResult], alpha: float, out_path: str) -> str:
    """Heatmap(s) of the PPI-corrected Type-I rate over (dataset, label_frac),
    one small-multiple panel per test -- pooled across every N and judge
    pair/triple that test's cells span, since (unlike pvalues.py's synthetic
    N x N_lab grid) real data has only 3-4 datasets and a handful of label
    fracs to show, so N/judge-pair are collapsed rather than given their own
    axis. Same TwoSlopeNorm diverging colormap centered on alpha as
    save_ppi_nlab_grid_plot, for the same reason: under- vs over-rejection
    should be visually distinct, not just "far from 0"."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if not results:
        raise ValueError("No ppi_real hypothesis results to plot.")
    tests = [m.name for m in PPI_TEST_METHODS if any(r.test == m.name for r in results)]
    datasets = sorted({r.tag.removeprefix("real_") for r in results})
    label_fracs = sorted({_parse_real_cell_labfrac_n(r.name)[0] for r in results})

    n_cols = min(4, len(tests))
    n_rows = -(-len(tests) // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)
    for idx, test in enumerate(tests):
        ax = axes[idx // n_cols][idx % n_cols]
        rejects = np.zeros((len(datasets), len(label_fracs)))
        totals = np.zeros((len(datasets), len(label_fracs)))
        for r in results:
            if r.test != test:
                continue
            ds = r.tag.removeprefix("real_")
            lf, _n = _parse_real_cell_labfrac_n(r.name)
            i, j = datasets.index(ds), label_fracs.index(lf)
            rejects[i, j] += r.corrected_rejects
            totals[i, j] += r.n_reps
        grid = np.where(totals > 0, np.divide(rejects, totals, out=np.full_like(rejects, np.nan), where=totals > 0), np.nan)

        vmax = max(2.0 * alpha, float(np.nanmax(grid)) * 1.1 if np.isfinite(np.nanmax(grid)) else 2.0 * alpha)
        im = ax.imshow(grid, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=alpha, vmax=vmax), aspect="auto")
        for i in range(len(datasets)):
            for j in range(len(label_fracs)):
                val = grid[i, j]
                if np.isfinite(val):
                    ax.text(
                        j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black",
                        bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.0),
                    )
        ax.set_xticks(range(len(label_fracs)))
        ax.set_xticklabels([f"{lf:.2f}" for lf in label_fracs], fontsize=8)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=8)
        ax.set_xlabel("label_frac", fontsize=8)
        ax.set_title(_pretty_test(test), fontsize=10, color=get_method_color(test))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for idx in range(len(tests), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle(f"PPI-Corrected Type-I Rate over Dataset x Label Fraction ({_alpha_label(alpha)})", y=1.02, fontsize=12)
    fig.text(0.5, -0.01, "Pooled across N and judge pair/triple", ha="center", fontsize=8, color="#555555")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ppi_real_alignment_plot(
    *, results: list[PPIResult], alignment_by_dataset_judge: dict[tuple[str, str], float],
    alpha: float, out_path: str,
) -> str:
    """Bar chart of uncorrected vs. PPI-corrected Type-I rate bucketed by
    realized judge-human alignment, one panel per test -- the real-data
    counterpart to save_ppi_alignment_sweep_plot, but using DIRECTLY
    measured alignment (np.corrcoef(judge_scores, human_label) on each
    corpus's already-[0,1]-rescaled arrays -- see RealJudgeBiasCorpus) for
    each cell's actual judge(s), not a simulated sweep over synthetic noise
    levels. A cell's alignment is the mean of its 1-3 involved judges'
    individual alignment (single/twogroup/paired/omnibus cells draw on 1/2/
    2/3 judges respectively -- see _parse_real_cell_judges). No no_bias/
    bias_present regime split like the synthetic version: real judges are
    always whatever bias they actually have, there's no "off" switch to
    contrast against."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No ppi_real hypothesis results to plot.")
    tests = [m.name for m in PPI_TEST_METHODS if any(r.test == m.name for r in results)]

    def _cell_alignment(r: PPIResult) -> float | None:
        ds = r.tag.removeprefix("real_")
        judges = _parse_real_cell_judges(r.name)
        vals = [alignment_by_dataset_judge[(ds, jm)] for jm in judges if (ds, jm) in alignment_by_dataset_judge]
        return float(np.mean(vals)) if vals else None

    cell_align = {id(r): _cell_alignment(r) for r in results}
    buckets = sorted({_alignment_bucket(_metric_pct(a)) for a in cell_align.values() if a is not None})
    if not buckets:
        raise ValueError("No alignment data available for any ppi_real hypothesis cell.")

    n_cols = min(4, len(tests))
    n_rows = -(-len(tests) // n_cols)
    panel_w = max(2.2, 0.85 * len(buckets))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, 4.0 * n_rows), squeeze=False)
    bar_width, gap = 0.35, 0.25
    for idx, test in enumerate(tests):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.axhline(alpha, color="black", ls="--", lw=1.0, alpha=0.6, zorder=1, label="nominal α" if idx == 0 else None)
        xticks, xticklabels = [], []
        for bi, (lo, label) in enumerate(buckets):
            cells = [r for r in results if r.test == test and cell_align.get(id(r)) is not None
                     and _alignment_bucket(_metric_pct(cell_align[id(r)])) == (lo, label)]
            n_reps_tot = sum(c.n_reps for c in cells)
            for ai, (arm, color, ec) in enumerate([("uncorrected", "#999999", "none"), ("corrected", get_method_color(test), "#333333")]):
                x = bi * (2 * bar_width + gap) + ai * bar_width
                if n_reps_tot == 0:
                    continue
                rejects_tot = sum((c.uncorrected_rejects if arm == "uncorrected" else c.corrected_rejects) for c in cells)
                rate = rejects_tot / n_reps_tot
                lo_ci, hi_ci = _ppi_wilson_interval(rejects_tot, n_reps_tot)
                ax.bar(x, rate, width=bar_width, color=color, edgecolor=ec, linewidth=1.0, zorder=2,
                       label=arm if (idx == 0 and bi == 0) else None)
                ax.errorbar(x, rate, yerr=[[max(0.0, rate - lo_ci)], [max(0.0, hi_ci - rate)]],
                             fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=4)
            mid = bi * (2 * bar_width + gap) + bar_width / 2
            xticks.append(mid)
            xticklabels.append(f"{label}\n(n={len(cells)})")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_xlim(-0.3, (2 * bar_width + gap) * len(buckets) - gap + 0.05)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(_pretty_test(test), fontsize=10, color=get_method_color(test))
        if idx % n_cols == 0:
            ax.set_ylabel("Type-I rate", fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.8, zorder=0)
    for idx in range(len(tests), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=True)
    fig.suptitle(f"PPI-Corrected Type-I Rate by Judge-Human Alignment ({_alpha_label(alpha)})", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
                         help="Cap on unique judge PAIRS checked by the twogroup/paired-null tests "
                              "(default: all C(k, 2) pairs for k judge models -- e.g. 28 for 8 judges. "
                              "Caps by random subset if k is large enough that C(k, 2) would be "
                              "impractically slow.")
    parser.add_argument("--max-triples", type=int, default=None,
                         help="Cap on unique judge TRIPLES checked by the omnibus-independent/omnibus-"
                              "repeated tests (default: all C(k, 3) triples for k judge models -- e.g. "
                              "56 for 8 judges. Caps by random subset if k is large enough that C(k, 3) "
                              "would be impractically slow.")
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
    parser.add_argument("--no-omnibus-independent-check", action="store_true",
                         help="Skip the 3-group cross-judge omnibus Type-I null check (anova_ind/kruskal).")
    parser.add_argument("--no-omnibus-repeated-check", action="store_true",
                         help="Skip the 3-condition cross-judge omnibus Type-I null check "
                              "(anova_rep/friedman).")
    parser.add_argument("--no-wmt-paired-check", action="store_true",
                         help="Skip the within-item (wmt_da_paired) bias/coverage check. Runs by default "
                              "IF judge_bias_wmt_da_paired.csv has been collected (see "
                              "collect_judge_bias_data.py --types continuous_paired); skipped with a "
                              "warning, not a hard failure, if it hasn't.")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N",
                         help="Parallel worker processes (default: cpu_count-1; 1=sequential). Parallelizes "
                              "across single/twogroup/paired/omnibus cells (all corpora, label_fracs, "
                              "sizes, judge models/pairs/triples flattened into one work queue) using a "
                              "multiprocessing.Pool "
                              "with the fork start method, same as pvalues.py's run_*_simulation functions. "
                              "Cell seeds are fixed up front from --seed regardless of --workers, so "
                              "results are identical either way -- only completion order (and hence "
                              "progress-bar interleaving) differs.")


def official_args(base_seed: int = 46) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=DEFAULT_DATA_DIR, datasets=None, judge_models=None, min_judge_coverage=0.9,
        max_pairs=None, max_triples=None,
        label_fracs=list(DEFAULT_LABEL_FRACS), sizes=None, reps=200, ppi_n_boot=2000,
        alpha=0.05, seed=base_seed, progress="bar", save_results="save", plots="save",
        out_dir="simulations/out", plots_dir=None,
        no_single_check=False, no_twogroup_check=False, no_paired_check=False,
        no_omnibus_independent_check=False, no_omnibus_repeated_check=False, no_wmt_paired_check=False,
        latex=True, workers=max(1, (os.cpu_count() or 2) - 1),
    )


def official_variants(base_seed: int = 46) -> list[tuple[str, argparse.Namespace]]:
    return [("real judge-bias data (single-sample + two-group + paired null + omnibus null + within-item paired bias/coverage)", official_args(base_seed))]


def quick_args(base_seed: int = 47, data_source: str = "synthetic") -> argparse.Namespace:
    """`data_source` accepted only for --quick-test's uniform
    (module.quick_args(), module.quick_args(data_source="real")) calling
    convention -- this case has no synthetic variant of its own (it's
    always real data), so both calls run the identical fast preset."""
    return argparse.Namespace(
        data_dir=DEFAULT_DATA_DIR, datasets=None, judge_models=None, min_judge_coverage=0.9,
        max_pairs=10, max_triples=10,
        label_fracs=[0.20], sizes=None, reps=5, ppi_n_boot=200,
        alpha=0.05, seed=base_seed, progress="bar", save_results="save", plots="save",
        out_dir="simulations/out", plots_dir=None,
        no_single_check=False, no_twogroup_check=False, no_paired_check=False,
        no_omnibus_independent_check=False, no_omnibus_repeated_check=False, no_wmt_paired_check=False,
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

        # Directly-measured judge-human alignment per (dataset, judge_model)
        # -- corpus.judge_scores/human_label are already rescaled to [0,1]
        # for every eval_type (see RealJudgeBiasCorpus), so a plain Pearson
        # r is comparable across binary/continuous/likert here, unlike the
        # synthetic sweep's per-eval_type metric choice (_ALIGNMENT_VIEWS).
        # Feeds save_ppi_real_alignment_plot below.
        alignment_by_dataset_judge: dict[tuple[str, str], float] = {
            (corpus.dataset, jm): float(np.corrcoef(scores, corpus.human_label)[0, 1])
            for corpus in corpora for jm, scores in corpus.judge_scores.items()
        }

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
        omnibus_results: list[PPIResult] = []
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
            omnibus_ind_methods = _omnibus_independent_methods_for(corpus.eval_type)
            omnibus_rep_methods = _omnibus_repeated_methods_for(corpus.eval_type)
            pairs = all_judge_pairs(corpus, max_pairs=args.max_pairs, rng=np.random.default_rng(args.seed))
            triples = all_judge_triples(corpus, max_triples=args.max_triples, rng=np.random.default_rng(args.seed))

            for label_frac in args.label_fracs:
                for n in sizes:
                    if not args.no_single_check:
                        for judge_model in corpus.judge_models:
                            name = f"real.{corpus.dataset}.labfrac={label_frac:.2f}.n={n}.judge={judge_model}"
                            seed_counter += 1
                            work_items.append((
                                "single", corpus_idx, name, corpus.dataset, single_methods,
                                n, label_frac, judge_model, args.reps, args.ppi_n_boot, seed_counter,
                            ))

                    # twogroup/paired both need judge PAIRS (cross-judge --
                    # see generate_real_twogroup_null_cell's docstring for
                    # why twogroup is cross-judge too, not one judge reading
                    # both random-split halves), so both loop over `pairs`.
                    if not args.no_twogroup_check or not args.no_paired_check:
                        for judge_a, judge_b in pairs:
                            name = f"real.{corpus.dataset}.labfrac={label_frac:.2f}.n={n}.pair={judge_a}~{judge_b}"

                            if not args.no_twogroup_check:
                                seed_counter += 1
                                work_items.append((
                                    "twogroup", corpus_idx, name, corpus.dataset, twogroup_methods,
                                    n, label_frac, (judge_a, judge_b), args.reps, args.ppi_n_boot, seed_counter,
                                ))

                            if not args.no_paired_check:
                                seed_counter += 1
                                work_items.append((
                                    "paired", corpus_idx, name, corpus.dataset, paired_methods,
                                    n, label_frac, (judge_a, judge_b), args.reps, args.ppi_n_boot, seed_counter,
                                ))

                    # omnibus-independent/omnibus-repeated both need judge
                    # TRIPLES (cross-judge, same reasoning as twogroup/paired
                    # above, just extended to 3 groups/conditions), so both
                    # loop over `triples`.
                    if not args.no_omnibus_independent_check or not args.no_omnibus_repeated_check:
                        for judge_a, judge_b, judge_c in triples:
                            name = (f"real.{corpus.dataset}.labfrac={label_frac:.2f}.n={n}"
                                    f".triple={judge_a}~{judge_b}~{judge_c}")

                            if not args.no_omnibus_independent_check:
                                seed_counter += 1
                                work_items.append((
                                    "omnibus_independent", corpus_idx, name, corpus.dataset, omnibus_ind_methods,
                                    n, label_frac, (judge_a, judge_b, judge_c), args.reps, args.ppi_n_boot, seed_counter,
                                ))

                            if not args.no_omnibus_repeated_check:
                                seed_counter += 1
                                work_items.append((
                                    "omnibus_repeated", corpus_idx, name, corpus.dataset, omnibus_rep_methods,
                                    n, label_frac, (judge_a, judge_b, judge_c), args.reps, args.ppi_n_boot, seed_counter,
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
                if ct == "twogroup":
                    bucket = twogroup_results
                elif ct == "paired":
                    bucket = paired_results
                else:
                    bucket = omnibus_results
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

        # twogroup + paired + omnibus combined into ONE Type-I report/plot,
        # matching pvalues.py's synthetic PPI sweep (run_ppi_simulation's
        # single combined `ppi_results` list, one print_ppi_report/
        # save_ppi_typeI_plot call) instead of separate ones per family --
        # safe to merge since all three families use entirely disjoint test
        # names (ttest/ttest_welch/mwu vs wilcoxon/paired_t/bayes_bootstrap/
        # bootstrap_t/tango vs anova_ind/anova_rep/friedman/kruskal, no
        # collision like the wmt_paired_bias/single-sample bootstrap_t one
        # elsewhere in this file), so this is purely additive -- no result
        # gets double-counted or aliased with another. Colors/legend come
        # from methods.py's Method.color per test, same registry pvalues.py's
        # plot uses, so a test keeps the same color whether it's plotted
        # from a synthetic or a real-data run.
        hypothesis_results = twogroup_results + paired_results + omnibus_results
        if hypothesis_results:
            print_ppi_report(hypothesis_results, alpha=args.alpha)
            run_stem = f"ppi_real_hypothesis_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_ppi(
                    results=hypothesis_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=args.latex,
                )
            if args.plots == "save":
                # Same empty-bucket guard as the effect plot above -- a
                # binary dataset's paired methods are {paired_t,
                # bayes_bootstrap, tango}, i.e. only ONE standard test
                # (paired_t) but TWO nonstandard ones, so neither bucket is
                # reliably non-empty across eval_types.
                if _has_standard_test(hypothesis_results):
                    plot_path = save_ppi_typeI_plot(
                        results=hypothesis_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected.png"),
                    )
                    output_paths.append(plot_path)
                    print(f"Saved plot: {plot_path}")
                if _has_nonstandard_test(hypothesis_results):
                    nonstd_plot_path = save_ppi_typeI_plot(
                        results=hypothesis_results, alpha=args.alpha,
                        out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected_nonstandard.png"),
                        nonstandard=True,
                    )
                    output_paths.append(nonstd_plot_path)
                    print(f"Saved plot: {nonstd_plot_path}")
                heatmap_path = save_ppi_real_labfrac_dataset_heatmap(
                    results=hypothesis_results, alpha=args.alpha,
                    out_path=str(Path(plots_dir) / f"{run_stem}_dataset_labfrac_heatmap.png"),
                )
                output_paths.append(heatmap_path)
                print(f"Saved plot: {heatmap_path}")
                alignment_plot_path = save_ppi_real_alignment_plot(
                    results=hypothesis_results, alignment_by_dataset_judge=alignment_by_dataset_judge,
                    alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_alignment.png"),
                )
                output_paths.append(alignment_plot_path)
                print(f"Saved plot: {alignment_plot_path}")
            c_tot = sum(r.corrected_rejects for r in hypothesis_results)
            u_tot = sum(r.uncorrected_rejects for r in hypothesis_results)
            n_tot = sum(r.n_reps for r in hypothesis_results)
            key_metrics["ppi_real_hypothesis_n_results"] = len(hypothesis_results)
            key_metrics["ppi_real_hypothesis_mean_corrected_type1"] = float(c_tot / n_tot) if n_tot else float("nan")
            key_metrics["ppi_real_hypothesis_mean_uncorrected_type1"] = float(u_tot / n_tot) if n_tot else float("nan")

        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics=key_metrics, duration_s=time.time() - t0,
        )
    except Exception as e:
        return CaseResult(case_name=CASE_NAME, status="error", error=str(e), duration_s=time.time() - t0)
