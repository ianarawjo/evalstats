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
LOGIT_T_2ND = Method("logit_t_2nd", "#bd5b17")
"""evalstats.core.resampling.logit_t_ci_1d(..., order=2) -- the optional
2nd-order (curvature) bias-corrected variant, registered here purely for
direct comparison against the order=1 default (LOGIT_T, used everywhere --
both ci_single.py and ci_paired.py). A 2026-08-04 investigation (see
logit_t_boundary_investigation memory) found order=2 gives a real, if
modest, coverage improvement on boundary-hugging/right-skewed single-sample
data at negligible width cost -- ci_single.py briefly defaulted LOGIT_T to
order=2 over that finding, then reverted (2026-08-04) since the gain wasn't
judged worth re-running the paper's simulations/rewriting results over.
Kept here as an opt-in comparison variant in case that tradeoff is
revisited later; order=2 was also confirmed to NOT help ci_paired's use of
logit_t (rescaled_ci recentres a paired diff near 0.5 regardless of raw
skew, where order=2's boundary-only correction never activates)."""

BINARY_SINGLE_EXTRA_METHODS = [WILSON, JEFFREYS, WALD, CLOPPER_PEARSON, BAYES_SINGLE]
CONTINUOUS_EXTRA_METHODS = [BETA, LOGIT_T, NIG, EL]
CONTINUOUS_EXTRA_METHODS_WITH_LOGIT_T_2ND = [BETA, LOGIT_T, LOGIT_T_2ND, NIG, EL]
"""CONTINUOUS_EXTRA_METHODS plus logit_t_2nd -- opt-in via --methods, not
part of the default battery (LOGIT_T_2ND is a validation-only comparison
variant, not a distinct recommended method)."""

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
Exercised both by cases/ppi_real.py's real-data single-sample bias/coverage
check and (as of the single-sample effect-check addition) pvalues.py's
synthetic PPI sweep -- see PPI_BOOTSTRAP_T_SINGLE below and
cases/pvalues.py's _run_ppi_effect_cell single-arm blocks."""
PPI_BOOTSTRAP_T_SINGLE = Method("bootstrap_t_single", "#9edae5")
"""PPI-corrected single-sample studentized-bootstrap CI (evalstats.tests.
_ppi_single_bootstrap_t) -- the bounded_01/continuous analogue of PPI_WILSON,
targeting the same single-sample (not paired) mean estimand PPI_AUTO_METHOD_
TABLE routes non-binary robustness CIs to. Deliberately distinct from
BOOTSTRAP_T (the paired/two-sample PPI method of the same underlying
construction) -- same reason PPI_WILSON isn't named "wilson": different
estimand, would silently collide if given the same Method name."""
PPI_T_INTERVAL = Method("ppi_t_interval", "#08519c")
"""PPI-corrected single-/paired-sample closed-form (no-bootstrap) t-interval
for an UNBOUNDED numeric mean/mean-difference estimand (evalstats.tests.
_ppi_single_t_interval / _ppi_paired_t_interval, both thin wrappers around
evalstats.ppi._analytic_mean_correct). The closed-form replacement for
BOOTSTRAP_T's role in PPI_AUTO_METHOD_TABLE's "unbounded" row (see
evalstats.config) -- resolved 2026-08-05, closing a previously-documented
gap. Uses the analytic construction at EVERY n_lab, not just below
_MIN_LAB_RECOMMENDED, mirroring the precedent evalstats.ppi.
_ANALYTIC_ALWAYS_PREFERRED set for the Wilcoxon estimand (analytic beat
bootstrap at every n_lab tested, 30-200). Run as a PAIRED mean-difference
test in this harness (cell.llm_x/llm_y/lab_x/lab_y), the same structural
role BOOTSTRAP_T/PAIRED_T occupy -- see PPI_LOGIT_T below for its
[0,1]-bounded sibling. Deliberately NOT named "t_interval" -- that's
already the plain (non-PPI-corrected) classical T_INTERVAL method above,
a different statistical procedure that happens to share the textbook
name (same reason PPI_WILSON isn't named "wilson")."""
PPI_LOGIT_T = Method("ppi_logit_t", "#8dd3c7")
"""PPI-corrected single-/paired-sample closed-form (no-bootstrap) logit-t
CI for a [lo, hi]-BOUNDED numeric mean/mean-difference estimand
(evalstats.tests._ppi_single_logit_t / _ppi_paired_logit_t, wrapping
evalstats.ppi._analytic_logit_t_correct -- the PPI analogue of
evalstats.core.resampling.logit_t_ci_1d's delta-method construction,
reusing the SAME point estimate/p-value as PPI_T_INTERVAL and differing
only in the CI's shape). The closed-form replacement for BOOTSTRAP_T's
role in PPI_AUTO_METHOD_TABLE's "bounded_01" row -- resolved 2026-08-05.
Run as a PAIRED mean-difference test here, rescaled onto [0, 1] via this
harness's own EVAL_TYPE_SCALE_BOUNDS[eval_type] (continuous/likert/grades;
excluded from binary scenarios the same way BOOTSTRAP_T is). Deliberately
NOT named "logit_t" -- see PPI_T_INTERVAL's docstring for why."""
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
TTEST = Method("ttest", "#1f77b4")
TTEST_WELCH = Method("ttest_welch", "#d62728")
# MWU/MWU_MNAR_EXPERIMENTAL: two PPI corrections for the SAME classical test
# (Mann-Whitney U / independent two-group mid-rank estimand P_mid(A>B)-0.5),
# not two different classical tests. MWU applies evalstats.tests.
# _ppi_two_sample's single GLOBAL rectifier -- exactly correct for a MEAN,
# but simulation (simulations/harness/cases/pvalues.py --mode ppi,
# judge-bias sweep, tag "label_mechanism" x "eval_type") showed it badly
# miscalibrated for this rank estimand specifically when labeling is
# non-uniform with respect to score (e.g. "double-check the highest-
# scoring items") combined with real judge bias and a coarse/discrete
# (Likert) scale: Type-I error 3-9x nominal in the worst identified cell
# (likert, severe bias, strong such labeling: 0.36-0.41 vs nominal 0.05).
# MWU_MNAR_EXPERIMENTAL (evalstats.tests._ppi_two_sample_midrank_corrected)
# fixes this with a per-group, per-score-bin LOCAL rectifier instead (see
# that function's docstring for the full mechanism/validation) -- and WAS
# the official PPI test sweep's default (as "mwu_corr") until 2026-07-22,
# when a controlled naive-vs-corrected comparison on matched draws (the
# same methodology that caught kruskal_corr's analogous regression -- see
# KRUSKAL/KRUSKAL_MNAR_EXPERIMENTAL below) found the SAME failure mode here
# at a smaller magnitude: under MCAR labeling with real judge bias present
# and a small labeled sample, Type-I climbed from ~3.6-5.6% (global) to
# ~5.4-7.2% (local) -- worst cells showing a clean 2x multiplier (e.g. 3.6%
# -> 7.2% at n_lab=15, severe bias, noise=0.025), replicated across two
# noise levels with the same n_lab/bias combination (not sampling noise:
# ~5 SEs at n_reps=1000). Given this project's stance that PPI requires
# MCAR labeling and treats MNAR as a documented, out-of-scope limitation
# (see evalstats.ppi.correct's docstring) rather than something to actively
# correct for, paying an MCAR cost for MNAR robustness in a regime users
# are already told not to rely on is the wrong trade -- so MWU (the global
# rectifier) is the default/official method again, and MWU_MNAR_EXPERIMENTAL
# is kept (not deleted) for direct comparison, reproducing pre-2026-07-22
# results, or anyone deliberately studying MNAR robustness, selectable
# explicitly via --tests mwu_mnar_experimental. cases/ppi_real.py's twogroup
# check dropped it entirely as of 2026-07-24 (real-data judge bias isn't
# MNAR, so there's nothing there for the local rectifier to buy over MWU --
# see _twogroup_methods_for's docstring), so it no longer needs to share a
# same-hue "Blues family" with ttest/ttest_welch/mwu the way it did when all
# four were expected to appear together in one plot (that family made a
# same-hue-different-shade grouping hard to tell apart at a glance, which is
# why ttest/ttest_welch/mwu moved to distinct tab10 hues instead) -- kept
# distinct here too so it isn't left visually stranded wherever it's still
# selected explicitly.
MWU = Method("mwu", "#2ca02c")
MWU_MNAR_EXPERIMENTAL = Method("mwu_mnar_experimental", "#9467bd")
# MWU_MNAR_POOLED (evalstats.tests._ppi_two_sample_midrank_corrected_pooled):
# a candidate replacement for MWU_MNAR_EXPERIMENTAL's bootstrap, NOT its
# rectifier. Investigated 2026-08-01 after tracing MWU_MNAR_EXPERIMENTAL's
# MCAR cost to something other than the local-vs-global rectifier choice:
# even at n_strata=1 (mathematically the SAME point estimate as MWU's
# global rectifier), MWU_MNAR_EXPERIMENTAL's bootstrap still ran hotter
# than MWU under MCAR, and isolating power-tuning (MWU defaults to it,
# MWU_MNAR_EXPERIMENTAL has none) didn't explain the gap either (MWU with
# power_tune=False stayed close to nominal, unlike the local rectifier at
# n_strata=1). The remaining difference: MWU_MNAR_EXPERIMENTAL draws FOUR
# separate bootstrap resamples (group A/B x labeled/unlabeled, each fixing
# that group's own count every replicate); evalstats.ppi.correct (what MWU
# actually uses) instead pools group A + B's unlabeled items into ONE
# array and draws a SINGLE resample from the pool (likewise for labeled
# items), letting the group split vary replicate-to-replicate. Switching
# ONLY that (same rectifier, same truth-binning collider fix, same hard
# min_lab_per_bin cutoff) improved BOTH axes on matched draws across 5
# MCAR + 4 MNAR corners (n_reps=1500, likert): mean MCAR |gap to naive|
# 0.0113 -> 0.0092, mean MNAR-corner rate (vs. nominal 0.05) 0.072 -> 0.056
# -- better than MWU_MNAR_EXPERIMENTAL in every corner tested, not just on
# average. Confirmed at harness scale (2064-scenario factorial grid,
# reps=300/n_boot=500, zero bootstrap failures): MCAR-only mean Type-I
# 0.0486 (vs. MWU's 0.0518, MWU_MNAR_EXPERIMENTAL's 0.0517), by-label-
# mechanism mean 0.049/0.048/0.048 (mcar/mnar_mild/mnar_strong, essentially
# flat) vs. 0.052/0.067/0.096 (MWU) and 0.052/0.056/0.067 (MWU_MNAR_
# EXPERIMENTAL), worst cell across the whole grid 0.097 vs. 0.630 (MWU) /
# 0.200 (MWU_MNAR_EXPERIMENTAL). Promoted to evalstats.tests.mannwhitney's
# default (method="local") 2026-08-01. See MWU_ADAPTIVE below, however --
# a follow-up 5-way estimator comparison (all_human/human_subset/llm_only/
# llm_impute/ppi) found this rectifier costs real POWER for CONTINUOUS
# data specifically (corrected power falls BELOW human_subset-only at
# every effect size tested, e.g. es=0.10: human_subset=0.730 vs. this=
# 0.417) -- inherited from the local rectifier's construction itself
# (shared identically by MWU_MNAR_EXPERIMENTAL), not something this
# resampling change introduced, and not fixable by tuning n_strata (the
# deficit persists even at n_strata=1). Selectable via --tests
# mwu_mnar_pooled; not part of PPI_OFFICIAL_TEST_METHODS (kept for direct
# comparison against MWU_ADAPTIVE, which is expected to supersede it as
# the recommended choice pending its own harness-scale validation).
MWU_MNAR_POOLED = Method("mwu_mnar_pooled", "#c5b0d5")  # lighter tint of MWU_MNAR_EXPERIMENTAL's purple
# MWU_ADAPTIVE (evalstats.tests._ppi_two_sample_adaptive): dispatches
# between MWU_MNAR_POOLED's local rectifier and MWU's global one based on
# how discrete the LABELED (truth) values look (unique_fraction =
# n_unique/n_labeled on the combined group A+B labeled sample; below 0.7
# -> local, at or above -> global -- see _ADAPTIVE_DISCRETENESS_THRESHOLD's
# docstring for the empirical separation, essentially deterministic:
# continuous always exactly 1.0, grades always >=0.967, likert always
# <=0.333). Exists because MWU_MNAR_POOLED's continuous-data power deficit
# (see its comment above) is real but avoidable: the deficit is specific
# to continuous/smooth data, where the global rectifier was already
# theoretically sufficient and the local one's per-bin/combine-then-rank
# construction only adds cost; Likert genuinely needs the local
# construction. Validated (matched draws, continuous + likert, vs.
# effect_size AND vs. n_lab, plus a small-n_lab MCAR/MNAR Type-I stress
# check): branch choice correct in every condition tested, "adaptive"
# exactly matching whichever of MWU/MWU_MNAR_POOLED was actually better --
# e.g. continuous es=0.10: adaptive=MWU=0.847 (vs. MWU_MNAR_POOLED alone
# at 0.433); likert es=0.20: adaptive=MWU_MNAR_POOLED=0.967 (vs. MWU alone
# at 0.620); likert MNAR Type-I: adaptive=MWU_MNAR_POOLED=0.047 (vs. MWU
# alone at 0.180, badly miscalibrated). See evalstats.tests.mannwhitney's
# method="adaptive" docstring for the full validation.
#
# Was briefly promoted to mannwhitney's default (method="adaptive")
# 2026-08-01/02, REVERTED to MWU (method="global") a few hours later: the
# validation above is synthetic-only. Running MWU_ADAPTIVE against
# cases/ppi_real.py's real judge-bias data (wmt_da dataset specifically)
# found a severe real-data MCAR calibration failure the synthetic grid
# never caught -- real (rounded/averaged) continuous truth values land at
# unique_fraction ~0.4-0.5, below the 0.7 threshold, so adaptive dispatches
# to the local rectifier; on real biased-judge data that rectifier's Type-I
# cost (0.25, isolated 300-rep check on one wmt_da cell) far exceeds the
# 0.093 synthetic worst-case above. A correctness regression, not a tuning
# nitpick -- see evalstats.tests._ppi_two_sample_adaptive's docstring for
# the full account. Kept available via --tests mwu_adaptive; not the
# default pending a better discreteness signal.
MWU_ADAPTIVE = Method("mwu_adaptive", "#98df8a")  # light tint of MWU's green
# MWU_RIDGE (evalstats.tests._ppi_two_sample_ridge_corrected): the user's
# requested next step
# after MWU_ADAPTIVE's revert (2026-08-02): "mix the ideas from global and
# local... dynamic adaptation of the pooling to the data" instead of a hard
# threshold dispatch. Root-caused MWU_ADAPTIVE's wmt_da failure precisely:
# MWU_MNAR_POOLED's per-bin STEP-FUNCTION rectifier has a real point-
# estimate BIAS on that cell (+0.042 under a true null, vs. MWU's +0.0003 --
# confirmed it's bias, not bootstrap variance-underestimation), because
# within a bin the judge's bias (truth-llm) is strongly correlated with the
# item's own score (corr -0.8 to -0.99 on that cell) -- a real slope a flat
# per-bin mean can't capture, and the STEP boundary amplifies it (bias jumps
# from -0.011 at n_strata=1 to +0.04-0.05 the moment n_strata>=2, and does
# NOT improve with finer bins). MWU_RIDGE replaces the step function with a
# ridge-shrunk LINEAR rectifier (fit per group: diff=truth-llm ~ beta0 +
# beta1*llm_score, ridge penalty lam=ridge_k*Sxx, ridge_k dimensionless) --
# on the exact failing wmt_da cell this brings Type-I from 0.25 (local) down
# to 0.047 (nominal, beating MWU's own 0.07). ridge_k=2.0 was picked from
# that ONE cell's bias/variance tradeoff curve (not a fully-automatic
# selection rule -- two candidates, empirical-Bayes/SE-based and closed-form
# LOOCV, were tried and rejected, both barely shrinking since they only see
# IN-SAMPLE/interpolation risk, not the EXTRAPOLATION risk to wherever
# unlabeled items' scores actually sit), but generalizes well: validated at
# full factorial-grid scale (2064 scenarios/8256 cells, reps=150/n_boot=300)
# -- MCAR mean/worst 0.056/0.113 (matches MWU's 0.054/0.113, no regression);
# MNAR-mild/strong worst-case 0.107/0.127, close to MWU_MNAR_POOLED's
# 0.107/0.120, vs. MWU's catastrophic 0.467/0.613 or MWU_ADAPTIVE's
# 0.520/0.440; power at moderate effect: continuous 0.975 (near MWU's 0.996
# ceiling), likert 0.871 -- the BEST of all four methods, beating even
# MWU_MNAR_POOLED's own 0.827 -- and against REAL data (cases/ppi_real.py,
# all 5 datasets): Type-I max 0.120 (== MWU's own max, vs. MWU_ADAPTIVE's
# 0.380), zero NEW Holm-confirmed cells, fixes wmt_da across multiple judge
# pairs (not just the tuning cell), real power unchanged at 1.000. See
# evalstats.tests._ppi_two_sample_ridge_corrected's docstring and
# simulations/out/mwu_ridge_validation/VALIDATION_SUMMARY.md for full
# numbers. NOT wired into mannwhitney() yet -- selectable via --tests
# mwu_ridge; pending a decision on whether/how to expose it as a method.
MWU_RIDGE = Method("mwu_ridge", "#c49c94")  # muted brown -- distinct from the MWU family's greens/purples
ANOVA_IND = Method("anova_ind", "#e6550d")
ANOVA_REP = Method("anova_rep", "#fd8d3c")
FRIEDMAN = Method("friedman", "#756bb1")  # purple -- distinct from the anova_*/lmm_* families
# KRUSKAL/KRUSKAL_MNAR_EXPERIMENTAL: two PPI corrections for the SAME
# omnibus test, generalizing the MWU/MWU_MNAR_EXPERIMENTAL story one level up (k
# independent groups instead of 2) -- and, as of 2026-07-22, the SAME
# conclusion. KRUSKAL applies evalstats.tests._ppi_kruskal_wallis_pairwise's
# single GLOBAL rectifier per pairwise dominance estimate theta_ab =
# theta_unlab + (theta_lab_human - theta_lab_llm) -- the SAME global-
# rectifier pattern that broke MWU, just extended from one pair to all
# C(k,2) pairs, and genuinely miscalibrated the same way under severe judge
# bias x MNAR-like labeling x coarse/discrete scale x large N (Type-I
# 0.32-0.49 vs nominal 0.05 in the worst identified cells). A per-group,
# per-score-bin LOCAL rectifier (evalstats.tests.
# _ppi_kruskal_wallis_pairwise_mnar_experimental, generalizing MW_MNAR_
# EXPERIMENTAL's fix from 2 groups to k) was built and validated to fix
# that MNAR regime (down to ~0.09-0.11 in the same cells) -- but was
# confirmed (2026-07-22 screening + a controlled naive-vs-corrected
# comparison on matched draws) to cost real MCAR calibration in exchange:
# worst found cell went from 7.9% (global, already-imperfect) to 11.1%
# (local) at small n_lab + high llm_noise, a regression the fix itself
# introduces, not one it inherits. A shrinkage/partial-pooling variant was
# prototyped to try to recover that MCAR cost without losing the MNAR fix
# and made BOTH worse, not just MCAR (see scratch prototypes in the
# ppi-welch-paired-t-calibration worktree). The SAME controlled comparison
# run against MWU/MWU_MNAR_EXPERIMENTAL (same day) found the identical
# failure mode at a smaller magnitude (~2x multiplier in the worst cells,
# e.g. 3.6% -> 7.2%, vs. kruskal's ~1.4x) -- see MWU/MWU_MNAR_EXPERIMENTAL's
# comment above for that data. Given this project's stance that PPI
# requires random (MCAR) label sampling and treats MNAR as a documented,
# out-of-scope limitation rather than something to actively correct for
# (see evalstats.ppi.correct's docstring), paying an MCAR cost to partially
# fix a regime users are already told not to rely on is the wrong trade for
# either -- so KRUSKAL (the global rectifier) is the default/official
# method as of 2026-07-22, and the local rectifier is demoted to
# KRUSKAL_MNAR_EXPERIMENTAL: kept for direct comparison and for anyone
# deliberately studying the MNAR-robustness question, selectable explicitly
# via --tests kruskal_mnar_experimental, but not part of the official/
# validated result set. Same color convention as MWU/MWU_MNAR_EXPERIMENTAL:
# the default occupies the original primary shade, the alternate gets a
# lighter tint.
KRUSKAL = Method("kruskal", "#e377c2")  # pink -- distinct from the anova_*/lmm_* families
KRUSKAL_MNAR_EXPERIMENTAL = Method("kruskal_mnar_experimental", "#f2b6d4")  # lighter tint
LMM = Method("lmm", "#74c476")
LMM_FACTORIAL = Method("lmm_factorial", "#a1d99b")
LMM_RUNS = Method("lmm_runs", "#c7e9c0")
PPI_TEST_METHODS = [
    TTEST, TTEST_WELCH, MWU, MWU_MNAR_EXPERIMENTAL, MWU_MNAR_POOLED, MWU_ADAPTIVE, MWU_RIDGE, WILCOXON, PAIRED_T, BAYES_BOOTSTRAP, BOOTSTRAP_T, TANGO, ANOVA_IND,
    ANOVA_REP, FRIEDMAN, KRUSKAL, KRUSKAL_MNAR_EXPERIMENTAL, LMM, LMM_FACTORIAL, LMM_RUNS, PPI_WILSON,
    PPI_BOOTSTRAP_T_SINGLE, PPI_T_INTERVAL, PPI_LOGIT_T,
]
"""Every PPI test method the harness knows how to run -- the full set
selectable via --tests. NOT what runs by default; see
PPI_OFFICIAL_TEST_METHODS for that."""
PPI_OFFICIAL_TEST_METHODS = [
    m for m in PPI_TEST_METHODS
    if m not in (
        MWU_MNAR_EXPERIMENTAL, MWU_MNAR_POOLED, MWU_ADAPTIVE, MWU_RIDGE, KRUSKAL_MNAR_EXPERIMENTAL,
        LMM, LMM_FACTORIAL, LMM_RUNS,
    )
]
"""The default (--tests unset) active-test set for --mode ppi -- every
PPI_TEST_METHODS entry except mwu_mnar_experimental/kruskal_mnar_experimental
(both fix real MNAR-labeling miscalibration in their global-rectifier
sibling, but were found to cost real MCAR calibration doing so -- see
MWU/MWU_MNAR_EXPERIMENTAL's and KRUSKAL/KRUSKAL_MNAR_EXPERIMENTAL's comments
above). Both remain selectable via --tests mwu_mnar_experimental / --tests
kruskal_mnar_experimental for direct comparison, reproducing pre-2026-07-22
results, or studying MNAR robustness deliberately.

lmm/lmm_factorial/lmm_runs EXCLUDED from the official set as of 2026-08-03
at the user's explicit request: not being reported in the current paper
iteration (may be added back for a later one) -- no point paying their
runtime cost (and reviewing their output) in every official pass while
that's true. Still fully selectable via --tests lmm/lmm_factorial/
lmm_runs for anyone who wants them.

PPI_WILSON/
PPI_BOOTSTRAP_T_SINGLE (the single-sample robustness-CI methods
PPI_AUTO_METHOD_TABLE routes to) are now part of the default set too --
pvalues.py's synthetic PPI sweep gained a single-arm effect-check scenario
(see cases/pvalues.py's _run_ppi_effect_cell single-arm blocks) precisely so
these wouldn't be validated ONLY by cases/ppi_real.py's real-data check."""

# ---------------------------------------------------------------------------
# Registry -- canonical ordering for tables/legends, and name -> Method lookup
# ---------------------------------------------------------------------------
REPORT_METHOD_ORDER: list[Method] = BOOTSTRAP_METHODS + [
    T_INTERVAL, WILSON, JEFFREYS, NEWCOMBE, TANGO, TANGO_SCC,
    WALD, CLOPPER_PEARSON, BAYES_SINGLE, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED, WALD_PAIR_INDEP,
] + CONTINUOUS_EXTRA_METHODS + [LOGIT_T_2ND] + NESTED_METHODS + BINARY_FLAT_METHODS + BINARY_NESTED_METHODS + (
    PAIR_DIFF_NESTED_METHODS
    + [TANGO_FLAT, NEWCOMBE_FLAT] + BINARY_PAIR_NESTED_METHODS
) + [
    MCNEMAR, PERMUTATION, SIGN_TEST, NEWCOMBE_PVAL, BAYES_BINARY, WILCOXON, PAIRED_T, PPI_T_INTERVAL, PPI_LOGIT_T,
] + MULTIARM_CORRECTION_METHODS + CANONICAL_SIMULTANEOUS_CI_METHODS + [
    TTEST, TTEST_WELCH, MWU, MWU_MNAR_EXPERIMENTAL, MWU_MNAR_POOLED, MWU_ADAPTIVE, MWU_RIDGE, ANOVA_IND, ANOVA_REP, FRIEDMAN, KRUSKAL, KRUSKAL_MNAR_EXPERIMENTAL,
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
