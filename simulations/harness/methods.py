"""Canonical CI-method registry, shared across cases.

Ported from the ``_METHOD_COLORS`` dict and method-name constants duplicated
across ``sim_compare_boot.py`` / ``sim_compare_boot_nested.py`` /
``sim_dove.py`` / ``sim_tango_real.py``. Each method is one ``Method``
instance carrying its name and plot color; callers work with the instance
directly (``method.name``, ``method.color``) instead of threading a bare
string through one lookup table per case. Centralizing this means a method
keeps the same name and the same plot color in every case's tables and
figures, instead of each case file redefining its own copy that can
silently drift out of sync.

There is no separate ``ci_nested`` case: ``sim_compare_boot_nested.py`` was
folded into ``cases/ci_single.py``'s and ``cases/ci_paired.py``'s
``--nested-mode`` flag instead, so single- and paired-sample CI logic stays
in one place per estimand. The "flat" vs "*_nested" method pairs below exist
because nested mode reports both side by side (flat = cell-mean reduction,
nested = full N×R matrix) to show what ignoring run structure costs.

Add new Method instances/groupings here as new cases are ported -- don't
predeclare methods for cases that don't exist as code yet.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_METHOD_COLOR = "#333333"


@dataclass(frozen=True)
class Method:
    name: str
    color: str = DEFAULT_METHOD_COLOR

    def __str__(self) -> str:
        return self.name

    def __format__(self, format_spec: str) -> str:
        return format(self.name, format_spec)


# ---------------------------------------------------------------------------
# Bootstrap-family methods (apply to single-sample means and paired diffs,
# flat or nested)
# ---------------------------------------------------------------------------
BOOTSTRAP = Method("bootstrap", "#1f77b4")
BCA = Method("bca", "#2ca02c")
BAYES_BOOTSTRAP = Method("bayes_bootstrap", "#ff7f0e")
SMOOTH_BOOTSTRAP = Method("smooth_bootstrap", "#9467bd")
BOOTSTRAP_T = Method("bootstrap_t", "#d62728")
BOOTSTRAP_METHODS = [BOOTSTRAP, BCA, BAYES_BOOTSTRAP, SMOOTH_BOOTSTRAP, BOOTSTRAP_T]

# ---------------------------------------------------------------------------
# Single-sample extras
# ---------------------------------------------------------------------------
T_INTERVAL = Method("t_interval", "#8c564b")
WILSON = Method("wilson", "#e377c2")
JEFFREYS = Method("jeffreys", "#e9cd14")
WALD = Method("wald", "#7f7f7f")
CLOPPER_PEARSON = Method("clopper_pearson", "#bcbd22")
BAYES_SINGLE = Method("bayes_indep", "#17becf")
BETA = Method("beta", "#f0027f")
LOGIT_T = Method("logit_t", "#a6761d")
NIG = Method("nig", "#888888")
EL = Method("el", "#00441b")

BINARY_SINGLE_EXTRA_METHODS = [WILSON, JEFFREYS, WALD, CLOPPER_PEARSON, BAYES_SINGLE]
CONTINUOUS_EXTRA_METHODS = [BETA, LOGIT_T, NIG, EL]

# ---------------------------------------------------------------------------
# Paired (pairwise-difference) extras -- for cases/ci_paired.py once ported
# ---------------------------------------------------------------------------
NEWCOMBE = Method("newcombe_score", "#aec7e8")
TANGO = Method("tango_score")  # no color in the legacy palette; uses the default
PPI_WILSON = Method("ppi_wilson", "#c49c94")
"""PPI-corrected single-sample Wilson score interval (evalstats.tests.
_ppi_single_wilson) -- a binary-proportion analogue of TANGO's paired
Wilson-style effective-n trick, for a single-sample (not two/paired-group)
mean estimand. Deliberately NOT named "wilson" -- that name is already
BINARY_SINGLE_EXTRA_METHODS' plain (non-PPI-corrected) Wilson CI for
ci_single.py, a different statistical procedure that happens to share the
same textbook name; reusing the same Method name here would have silently
overwritten that entry (same module-level name, last assignment wins).
Only exercised by cases/ppi_real.py's real-data single-sample bias/coverage
check so far (pvalues.py's synthetic PPI sweep never runs a plain one-sample
estimation check) -- registered here anyway so the shared report/plot
functions (_ppi_tests_present etc., which filter by PPI_TEST_METHODS
membership) recognize it instead of silently dropping it."""
TANGO_SCC = Method("tango_scc", "#b15928")
BAYES_PAIR_INDEP = Method("bayes_indep_comp", "#ffbb78")
BAYES_PAIR_PAIRED = Method("bayes_paired_comp", "#98df8a")
WALD_PAIR_INDEP = Method("wald_indep", "#7f7f7f")  # same grey as ci_single's WALD -- both are the naive baseline
PAIRWISE_EXTRA_METHODS = [T_INTERVAL, LOGIT_T, NIG, EL]
BINARY_PAIRWISE_EXTRA_METHODS = [NEWCOMBE, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED, WALD_PAIR_INDEP]

# ---------------------------------------------------------------------------
# Nested-mode methods -- for ci_single.py's/ci_paired.py's --nested-mode,
# ported from sim_compare_boot_nested.py. "Flat" methods here always reduce
# an (n, runs) matrix to per-input cell means first; "nested" methods apply
# directly to the full (n, runs) matrix. Both are reported side by side so
# the gap between them (the cost of ignoring run structure) is visible.
# ---------------------------------------------------------------------------
BOOTSTRAP_NESTED = Method("bootstrap_nested", "#aec7e8")
BAYES_NESTED = Method("bayes_bootstrap_nested", "#ffbb78")
SMOOTH_NESTED = Method("smooth_bootstrap_nested", "#c5b0d5")
BCA_NESTED = Method("bca_nested", "#98df8a")
BOOTSTRAP_T_NESTED = Method("bootstrap_t_nested", "#ff9896")
NESTED_METHODS = [BOOTSTRAP_NESTED, BAYES_NESTED, SMOOTH_NESTED, BCA_NESTED, BOOTSTRAP_T_NESTED]

WILSON_FLAT = Method("wilson_flat", "#e7298a")
WALD_FLAT = Method("wald_flat", "#66a61e")
CP_FLAT = Method("clopper_pearson_flat", "#e6ab02")
BAYES_INDEP_FLAT = Method("bayes_indep_flat", "#1b9e77")
BINARY_FLAT_METHODS = [WILSON_FLAT, WALD_FLAT, CP_FLAT, BAYES_INDEP_FLAT]

WILSON_OD = Method("wilson_od", "#666666")
WILSON_OD_BC = Method("wilson_od_bc", "#e31a1c")
WILSON_OD_T = Method("wilson_od_t", "#6a3d9a")
JEFFREYS_OD = Method("jeffreys_od", "#b2df8a")
CP_OD = Method("cp_od", "#fb9a99")
BB_BAYES = Method("bb_bayes", "#33a02c")
BB_BAYES_ROBUST = Method("bb_bayes_robust", "#ff7f00")
BINARY_NESTED_METHODS = [WILSON_OD, WILSON_OD_BC, WILSON_OD_T, JEFFREYS_OD, CP_OD, BB_BAYES, BB_BAYES_ROBUST]

BOOTSTRAP_DIFF_NESTED = Method("bootstrap_diff_nested", "#1b9e77")
BAYES_DIFF_NESTED = Method("bayes_diff_nested", "#d95f02")
SMOOTH_DIFF_NESTED = Method("smooth_diff_nested", "#7570b3")
PAIR_DIFF_NESTED_METHODS = [BOOTSTRAP_DIFF_NESTED, BAYES_DIFF_NESTED, SMOOTH_DIFF_NESTED]

TANGO_FLAT = Method("tango_flat", "#e7298a")
TANGO_MEAN = Method("tango_mean", "#8c564b")
NEWCOMBE_FLAT = Method("newcombe_flat", "#66a61e")
BINARY_PAIR_FLAT_METHODS = [TANGO_FLAT, NEWCOMBE_FLAT, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED, WALD_PAIR_INDEP]

TANGO_MULTIRUN_EFFECTIVE = Method("tango_multirun_effective", "#a6761d")
TANGO_MULTIRUN_MOMENTS = Method("tango_multirun_mmnt", "#1b9e77")
BINARY_PAIR_NESTED_METHODS = [TANGO_MULTIRUN_EFFECTIVE, TANGO_MULTIRUN_MOMENTS]

# ---------------------------------------------------------------------------
# cases/pvalues.py -- raw pairwise p-value/rejection procedures (non-PPI
# path), ported from sim_compare_pvalues.py's PAIRWISE_METHODS. These compute
# a p-value/rejection decision via evalstats.core.paired.pairwise_differences,
# not a CI -- distinct from the CI-coverage methods above even where a name
# overlaps conceptually (e.g. BOOTSTRAP/BCA/BAYES_BOOTSTRAP/SMOOTH_BOOTSTRAP
# are reused as-is; "newcombe"/"bayes_binary" are NOT the same underlying
# computation as ci_paired's "newcombe_score"/"bayes_indep_comp", so they get
# distinct Method instances despite the conceptual overlap).
# ---------------------------------------------------------------------------
MCNEMAR = Method("mcnemar", "#393b79")
PERMUTATION = Method("permutation", "#8c6d31")
SIGN_TEST = Method("sign_test", "#843c39")
NEWCOMBE_PVAL = Method("newcombe", "#7b4173")
BAYES_BINARY = Method("bayes_binary", "#5254a3")
WILCOXON = Method("wilcoxon", "#8ca252")
PAIRED_T = Method("paired_t", "#bd9e39")
PAIRWISE_PVALUE_METHODS = [
    MCNEMAR, BOOTSTRAP, BCA, BAYES_BOOTSTRAP, SMOOTH_BOOTSTRAP, BOOTSTRAP_T,
    PERMUTATION, SIGN_TEST, NEWCOMBE_PVAL, BAYES_BINARY, WILCOXON, PAIRED_T,
]

# ---------------------------------------------------------------------------
# cases/pvalues.py -- multi-arm multiplicity-correction strategies (non-PPI
# path), ported from sim_compare_pvalues.py's MULTIARM_CORRECTIONS.
# NONE/HOLM/BONFERRONI/FDR_BH/HOCHBERG/SHAFFER correct the Wilcoxon
# signed-rank p-value -- evalstats' canonical, eval-type-agnostic paired
# test (unlike Tango/Logit-t in --mode simultaneous_ci's
# CANONICAL_SIMULTANEOUS_CI_METHODS, one test covers binary/continuous/
# likert/grades alike, so there's no per-eval-type split here) -- rather
# than --multiarm-method's raw p-value; see cases/pvalues.py's
# _compute_multiarm_metrics. HOCHBERG is a closed-form step-up refinement of
# HOLM (never more conservative, valid under the non-negative dependence
# repeated-measures/shared-item designs produce -- see
# evalstats.core.stats_utils.correct_pvalues). SHAFFER is a closed-form
# step-down refinement of HOLM specific to *all-pairwise* comparisons among
# k arms, exploiting the transitivity of equality (if A=B and B=C then
# A=C) to use a smaller, non-constant divisor sequence instead of HOLM's
# plain (m, m-1, ..., 1) -- see
# evalstats.core.stats_utils._shaffer_adjusted_pvalues.
#
# MAX_T/ROMANO_WOLF/WESTFALL_YOUNG are the resampling-based family: MAX_T is
# *single-step* studentized-bootstrap max-T (one joint critical value for
# every pair); ROMANO_WOLF is the *step-down* refinement of the same
# bootstrap-t null (recomputing the max only over not-yet-rejected pairs at
# each step, so it dominates MAX_T in power for the same FWER guarantee);
# WESTFALL_YOUNG is the permutation-based (per-item sign-flip) analogue of
# ROMANO_WOLF's step-down algorithm, exact under exchangeability of the
# paired design rather than relying on the bootstrap's asymptotic
# justification. All three need a bootstrap/permutation-compatible
# resampling scheme (Wilcoxon has no joint max-T analogue), so they stay on
# --multiarm-method (bootstrap_t by default) / per-item paired differences
# regardless of eval type -- see cases/pvalues.py's
# _stepdown_max_t_pvalues. FRIEDMAN_NEMENYI is unaffected either way --
# already its own rank-based omnibus + post-hoc test.
#
# BOOT is the multiarm analogue of --mode simultaneous_ci's `boot` (see
# CANONICAL_SIMULTANEOUS_CI_METHODS below, whose CORR_BOOT instance this
# reuses as-is): it widens the canonical Wilcoxon p-value using a joint
# bootstrap critical value (the same max-over-all-pairs studentized-mean
# resample MAX_T/ROMANO_WOLF use -- see cases/pvalues.py's
# _bootstrap_t_matrix) instead of a fixed, correlation-blind factor the way
# HOLM/BONFERRONI/HOCHBERG/SHAFFER do -- rescaling raw_p by alpha/alpha_eff,
# where alpha_eff is that joint critical value translated back to an
# equivalent significance level (mirroring
# evalstats.core.paired._joint_bootstrap_scaled_simultaneous_cis's z<->alpha
# translation). Unlike MAX_T/ROMANO_WOLF/WESTFALL_YOUNG, BOOT is NOT tied to
# --multiarm-method -- like NONE/HOLM/etc. it always operates on the
# canonical Wilcoxon statistic, so it directly tests whether the small FWER
# excess observed for MAX_T/ROMANO_WOLF at n=500-2000 is specific to their
# studentized-mean bootstrap-t construction or a more general property of
# bootstrap-based FWER correction.
# ---------------------------------------------------------------------------
CORR_NONE = Method("none", "#9c9ede")
CORR_HOLM = Method("holm", "#cedb9c")
CORR_BONFERRONI = Method("bonferroni", "#e7ba52")
CORR_FDR_BH = Method("fdr_bh", "#ad494a")
CORR_HOCHBERG = Method("hochberg", "#e6550d")
CORR_SHAFFER = Method("shaffer", "#9e9ac8")
CORR_FRIEDMAN_NEMENYI = Method("friedman_nemenyi", "#a55194")
CORR_MAX_T = Method("max_t", "#5254a3")
CORR_ROMANO_WOLF = Method("romano_wolf", "#6baed6")
CORR_WESTFALL_YOUNG = Method("westfall_young", "#74c476")
CORR_BOOT = Method("boot", "#3182bd")
MULTIARM_CORRECTION_METHODS = [
    CORR_NONE, CORR_HOLM, CORR_BONFERRONI, CORR_FDR_BH, CORR_HOCHBERG, CORR_SHAFFER,
    CORR_FRIEDMAN_NEMENYI, CORR_MAX_T, CORR_ROMANO_WOLF, CORR_WESTFALL_YOUNG, CORR_BOOT,
]

# ---------------------------------------------------------------------------
# cases/pvalues.py -- simultaneous-CI construction methods (non-PPI path),
# for --mode simultaneous_ci. Reuses CORR_NONE/CORR_MAX_T/CORR_BONFERRONI
# as-is (same name/color as their multiarm p-value-correction counterparts):
# CORR_NONE is the naive per-pair CI with no simultaneous adjustment at all
# (the "why do you need any correction?" baseline, same role as multiarm's
# own `none` row); CORR_MAX_T/CORR_BONFERRONI are the two constructions
# evalstats.core.paired's _simultaneous_cis_router picks between. Only these
# three of multiarm's six correction strategies have a well-established
# simultaneous-CI dual -- holm/fdr_bh are p-value-only adjustments with no
# associated CI, and friedman_nemenyi operates on rank differences rather
# than the raw mean-difference scale a CI here would need.
# ---------------------------------------------------------------------------
SIMULTANEOUS_CI_METHODS = [CORR_NONE, CORR_BONFERRONI, CORR_MAX_T]

# ---------------------------------------------------------------------------
# cases/pvalues.py -- canonical-CI-based simultaneous-CI constructions
# (non-PPI path), for --mode simultaneous_ci. `none`/`bonferroni` (built on
# `matrix_raw.results`) and `sidak`/`boot` are all built on evalstats'
# *canonical* pairwise CI method for the scenario's eval type -- Tango for
# binary, Logit-t for continuous/likert (bounded [0, 1]/[lo, hi] numeric
# data; see evalstats.config.AUTO_ANALYZE_METHOD_TABLE's "binary"/
# "bounded_01" rows) -- rather than whatever --multiarm-method is in force.
# max_t is the one exception: it needs a bootstrap-compatible method to
# resample from (neither Tango nor Logit-t is), so it keeps using
# --multiarm-method (bootstrap_t by default) regardless of eval type.
# `grades` has no canonical default wired up here (deliberately out of
# scope -- see cases/pvalues.py's _CANONICAL_CI_FUNC_BY_EVAL_TYPE), so these
# rows are simply absent for grades scenarios.
# SIDAK ("sidak") and BOOT ("boot") widen the canonical CI to hold
# family-wise via evalstats.core.paired's generic (not method-specific)
# _sidak_simultaneous_cis / _joint_bootstrap_scaled_simultaneous_cis, called
# with whichever ci_func matches the scenario's eval type: Sidak's
# closed-form per-comparison alpha adjustment, and a joint-bootstrap
# critical value that accounts for correlation between comparisons and is
# substituted for the canonical CI's marginal normal quantile -- the two
# options a from-scratch multiplicity-correction analysis would reach for,
# per Gemini's Sidak/bootstrap-scaling suggestion this mode's docstring
# discusses. Named plain "sidak"/"boot" (not e.g. "tango_sidak") since which
# base CI they widen is now scenario-dependent, not fixed to one method.
# CORR_BOOT (not CORR_SIDAK) is reused as-is by MULTIARM_CORRECTION_METHODS
# above -- see its comment for the p-value-side analogue.
# ---------------------------------------------------------------------------
CORR_SIDAK = Method("sidak", "#31a354")
CANONICAL_SIMULTANEOUS_CI_METHODS = [CORR_SIDAK, CORR_BOOT]

# ---------------------------------------------------------------------------
# cases/pvalues.py -- evalstats.tests wrapper names (PPI-corrected path),
# ported from sim_type_i_calibration.py's TEST_NAMES. Each is run twice per
# scenario (uncorrected -- the scipy-equivalent test on LLM-only scores -- and
# PPI-corrected -- the same test with sparse human labels) to check whether
# PPI correction fixes the Type-I inflation judge bias/miscalibration causes.
# WILCOXON is shared with the pairwise non-PPI registry above: same
# underlying paired-difference test, just on different data structures.
# PAIRED_T is likewise shared with the pairwise non-PPI registry above: the
# mean-based paired-difference sibling to WILCOXON's median-based one --
# also the entry point for binary (a proportion is just a mean), alongside
# TTEST/TTEST_WELCH -- see scenarios.synthetic's binary judge-bias comment.
# BAYES_BOOTSTRAP is shared with the bootstrap-family registry above: the
# same paired-mean-difference estimand as PAIRED_T, but PPI-corrected via a
# Dirichlet-weighted (Bayesian) bootstrap instead of evalstats.ppi.correct's
# classical one -- see evalstats.tests._ppi_paired_bayes_bootstrap -- kept
# as a validated alternative, not a recommended default (real-data testing
# found it underperforms; PAIRED_T remains the reasonable default for
# binary p-values). BOOTSTRAP_T is likewise shared: the same paired-mean
# estimand, PPI-corrected via a studentized-bootstrap pivot generalizing
# evalstats.core.resampling.bootstrap_t_ci_1d's per-replicate SE to PPI's
# two-term variance -- see evalstats.tests._ppi_paired_bootstrap_t. Numeric
# (continuous/likert/grades) only -- unlike PAIRED_T/BAYES_BOOTSTRAP, not
# extended to binary, since bootstrap_t's value is specifically for
# resampling-based CI estimation on numeric data at N>=50 (ci_paired.py).
# TANGO (reusing ci_paired's existing "tango_score" Method instance) is the
# mirror image: binary paired data ONLY, not numeric -- PPI-corrects
# evalstats.core.resampling.tango_paired_ci's score interval by substituting
# an effective-n derived from PPI's two-term variance into its Wilson-style
# shrinkage formula (see evalstats.tests._ppi_paired_tango); fully
# closed-form, no bootstrap resampling.
# ---------------------------------------------------------------------------
TTEST = Method("ttest", "#3182bd")
TTEST_WELCH = Method("ttest_welch", "#6baed6")
# MW_NAIVE/MWU_CORR: two PPI corrections for the SAME classical test
# (Mann-Whitney U / independent two-group mid-rank estimand P_mid(A>B)-0.5),
# not two different classical tests. MW_NAIVE (formerly just "MW") applies
# evalstats.tests._ppi_two_sample's single GLOBAL rectifier -- exactly
# correct for a MEAN, but simulation (simulations/harness/cases/pvalues.py
# --mode ppi, judge-bias sweep, tag "label_mechanism" x "eval_type") showed
# it badly miscalibrated for this rank estimand specifically when labeling
# is non-uniform with respect to score (e.g. "double-check the highest-
# scoring items") combined with real judge bias and a coarse/discrete
# (Likert) scale: Type-I error 3-9x nominal in the worst identified cell
# (likert, severe bias, strong such labeling: 0.36-0.41 vs nominal 0.05).
# MWU_CORR (evalstats.tests._ppi_two_sample_midrank_corrected) fixes this
# with a per-group, per-score-bin LOCAL rectifier instead (see that
# function's docstring for the full mechanism/validation) and is what the
# official PPI test sweep now runs by default in mw_naive's old slot --
# see PPI_OFFICIAL_TEST_METHODS below. MW_NAIVE is kept (not deleted) for
# direct comparison and reproducing pre-fix results, selectable explicitly
# via --tests mw_naive. Colors stay in the same Blues family as
# ttest/ttest_welch (mwu_corr inherits mw's original shade, since it now
# occupies that slot in the standard-tests plot; mw_naive gets a visually
# lighter/secondary shade of the same family).
MW_NAIVE = Method("mw_naive", "#c6dbef")
MWU_CORR = Method("mwu_corr", "#9ecae1")
ANOVA_IND = Method("anova_ind", "#e6550d")
ANOVA_REP = Method("anova_rep", "#fd8d3c")
FRIEDMAN = Method("friedman", "#756bb1")  # purple -- distinct from the anova_*/lmm_* families
# KRUSKAL_NAIVE/KRUSKAL_CORR: same MW_NAIVE/MWU_CORR story, one level up (k
# independent groups instead of 2). KRUSKAL_NAIVE (formerly just "KRUSKAL")
# is evalstats.tests._ppi_kruskal_wallis_pairwise's single GLOBAL rectifier
# per pairwise dominance estimate theta_ab = theta_unlab + (theta_lab_human -
# theta_lab_llm) -- the SAME global-rectifier pattern that broke mw_naive,
# just extended from one pair to all C(k,2) pairs. Once the PPI factorial
# sweep was extended to the omnibus tests (_COMPARISON_METHODS_OMNIBUS,
# simulations/harness/cases/pvalues.py --factorial-omnibus), it turned up
# the exact same combined-factor blowup mw_naive had (severe judge bias x
# MNAR-like labeling x coarse/discrete scale x large N): Type-I 0.32-0.49 vs
# nominal 0.05 in the worst identified cells (likert, severe bias, strong
# such labeling, n=200-400). KRUSKAL_CORR (evalstats.tests.
# _ppi_kruskal_wallis_pairwise_corrected) fixes this the same way MWU_CORR
# does -- a per-group, per-score-bin LOCAL rectifier, generalized from 2
# groups to k groups / all C(k,2) pairs jointly (see that function's
# docstring for the full mechanism) -- reducing those same cells to ~0.09-0.11
# (a real, if imperfect, fix: it inherits the same modest residual MWU_CORR
# also has under strong MNAR labeling with NO judge bias present at certain
# N/N_lab combinations, confirmed to be pre-existing in MWU_CORR too, not
# introduced by this fix), with no power loss and no regression under MCAR
# labeling (with or without judge bias) or continuous eval_type. Kept
# alongside (not deleted) for direct comparison and reproducing pre-fix
# results, selectable explicitly via --tests kruskal_naive. Same color
# convention as MW_NAIVE/MWU_CORR: kruskal_corr inherits kruskal's original
# shade (it now occupies that slot), kruskal_naive gets a lighter tint.
KRUSKAL_NAIVE = Method("kruskal_naive", "#f2b6d4")  # lighter tint of kruskal's original pink
KRUSKAL_CORR = Method("kruskal_corr", "#e377c2")  # pink -- distinct from the anova_*/lmm_* families
LMM = Method("lmm", "#74c476")
LMM_FACTORIAL = Method("lmm_factorial", "#a1d99b")
LMM_RUNS = Method("lmm_runs", "#c7e9c0")
PPI_TEST_METHODS = [
    TTEST, TTEST_WELCH, MW_NAIVE, MWU_CORR, WILCOXON, PAIRED_T, BAYES_BOOTSTRAP, BOOTSTRAP_T, TANGO, ANOVA_IND,
    ANOVA_REP, FRIEDMAN, KRUSKAL_NAIVE, KRUSKAL_CORR, LMM, LMM_FACTORIAL, LMM_RUNS, PPI_WILSON,
]
"""Every PPI test method the harness knows how to run -- the full set
selectable via --tests. NOT what runs by default; see
PPI_OFFICIAL_TEST_METHODS for that."""
PPI_OFFICIAL_TEST_METHODS = [m for m in PPI_TEST_METHODS if m not in (MW_NAIVE, KRUSKAL_NAIVE, PPI_WILSON)]
"""The default (--tests unset) active-test set for --mode ppi -- every
PPI_TEST_METHODS entry except mw_naive/kruskal_naive, which simulation showed
badly miscalibrated under MNAR-like labeling x real judge bias x
coarse/discrete scales (see MW_NAIVE/KRUSKAL_NAIVE's comments above) and
which mwu_corr/kruskal_corr now cover instead (mw_naive/kruskal_naive still
run if explicitly requested via --tests mw_naive/kruskal_naive); and except
ppi_wilson, excluded not because it's miscalibrated but because pvalues.py's
synthetic PPI sweep has no single-sample scenario to run it against at all
(see PPI_WILSON's docstring) -- only cases/ppi_real.py's real-data single-
sample check uses it, selecting it explicitly rather than through this
"default active set"."""

# ---------------------------------------------------------------------------
# Registry -- canonical ordering for tables/legends, and name -> Method lookup
# ---------------------------------------------------------------------------
REPORT_METHOD_ORDER: list[Method] = BOOTSTRAP_METHODS + [
    T_INTERVAL, WILSON, JEFFREYS, NEWCOMBE, TANGO, TANGO_SCC,
    WALD, CLOPPER_PEARSON, BAYES_SINGLE, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED, WALD_PAIR_INDEP,
] + CONTINUOUS_EXTRA_METHODS + NESTED_METHODS + BINARY_FLAT_METHODS + BINARY_NESTED_METHODS + (
    PAIR_DIFF_NESTED_METHODS
    + [TANGO_FLAT, NEWCOMBE_FLAT] + BINARY_PAIR_NESTED_METHODS
) + [
    MCNEMAR, PERMUTATION, SIGN_TEST, NEWCOMBE_PVAL, BAYES_BINARY, WILCOXON, PAIRED_T,
] + MULTIARM_CORRECTION_METHODS + CANONICAL_SIMULTANEOUS_CI_METHODS + [
    TTEST, TTEST_WELCH, MW_NAIVE, MWU_CORR, ANOVA_IND, ANOVA_REP, FRIEDMAN, KRUSKAL_NAIVE, KRUSKAL_CORR,
    LMM, LMM_FACTORIAL, LMM_RUNS,
]

METHODS_BY_NAME: dict[str, Method] = {m.name: m for m in REPORT_METHOD_ORDER}


def get_method(name: str) -> Method:
    """Look up a Method by name, falling back to a default-colored Method for unknown names."""
    return METHODS_BY_NAME.get(name, Method(name))


def get_method_color(name: str) -> str:
    return get_method(name).color


def order_present_methods(present_names: set[str]) -> list[Method]:
    """Filter REPORT_METHOD_ORDER down to methods actually present, preserving canonical order."""
    return [m for m in REPORT_METHOD_ORDER if m.name in present_names]
