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
both ci_single.py and ci_paired.py). Gives a modest coverage improvement on
boundary-hugging/right-skewed single-sample data at negligible width cost,
but not part of the default battery: the gain wasn't judged worth changing
recorded results over. Does not help ci_paired's use of
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
mean estimand. Deliberately not named "wilson" -- that name is already
BINARY_SINGLE_EXTRA_METHODS' plain (non-PPI-corrected) Wilson CI for
ci_single.py, a different statistical procedure that happens to share the
same textbook name; reusing it here would silently overwrite that entry
(same module-level name, last assignment wins). Exercised by both
cases/ppi_real.py's real-data single-sample bias/coverage check and
pvalues.py's synthetic PPI sweep -- see PPI_BOOTSTRAP_T_SINGLE below and
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
"""PPI-corrected closed-form (no-bootstrap) t-interval for an unbounded
numeric mean/mean-difference estimand (evalstats.tests._ppi_single_t_interval
/ _ppi_paired_t_interval, both thin wrappers around evalstats.ppi.
_analytic_mean_correct). The closed-form replacement for BOOTSTRAP_T's role
in PPI_AUTO_METHOD_TABLE's "unbounded" row (see evalstats.config). Uses the
analytic construction at every n_lab, not just below _MIN_LAB_RECOMMENDED,
mirroring the precedent evalstats.ppi._ANALYTIC_ALWAYS_PREFERRED set for
the Wilcoxon estimand. Run as a paired mean-difference test in this harness
(cell.llm_x/llm_y/lab_x/lab_y), the same structural role BOOTSTRAP_T/PAIRED_T
occupy -- see PPI_LOGIT_T below for its [0,1]-bounded sibling. Deliberately
not named "t_interval" -- that's already the plain (non-PPI-corrected)
classical T_INTERVAL method above, a different statistical procedure that
happens to share the textbook name (same reason PPI_WILSON isn't named
"wilson").

Paired estimand only -- see PPI_T_INTERVAL_SINGLE below for the
single-sample sibling, split out for the same reason PPI_BOOTSTRAP_T_SINGLE
is split from BOOTSTRAP_T: this Method's name is pooled across every
paired-family check that uses it (print_ppi_effect_report groups by test
name only, not by estimand), so a single-sample usage under the same name
would silently merge two different estimands' bias/coverage stats
together."""
PPI_T_INTERVAL_SINGLE = Method("ppi_t_interval_single", "#6baed6")  # lighter tint of PPI_T_INTERVAL
"""Single-sample sibling of PPI_T_INTERVAL (evalstats.tests.
_ppi_single_t_interval), split out for the same reason
PPI_BOOTSTRAP_T_SINGLE is split from BOOTSTRAP_T -- see PPI_T_INTERVAL's
docstring. Targets an unbounded numeric single-sample mean estimand, the
non-binary/non-[0,1]-bounded counterpart to PPI_WILSON's role."""
PPI_LOGIT_T = Method("ppi_logit_t", "#8dd3c7")
"""PPI-corrected closed-form (no-bootstrap) logit-t CI for a [lo, hi]-bounded
numeric mean/mean-difference estimand (evalstats.tests._ppi_single_logit_t /
_ppi_paired_logit_t, wrapping evalstats.ppi._analytic_logit_t_correct -- the
PPI analogue of evalstats.core.resampling.logit_t_ci_1d's delta-method construction,
reusing the same point estimate/p-value as PPI_T_INTERVAL and differing
only in the CI's shape). The closed-form replacement for BOOTSTRAP_T's
role in PPI_AUTO_METHOD_TABLE's "bounded_01" row. Run as a paired
mean-difference test here, rescaled onto [0, 1] via this harness's own
EVAL_TYPE_SCALE_BOUNDS[eval_type] (continuous/likert/grades; excluded from
binary scenarios the same way BOOTSTRAP_T is). Deliberately not named
"logit_t" -- see PPI_T_INTERVAL's docstring for why.

Paired estimand only -- see PPI_LOGIT_T_SINGLE below and PPI_T_INTERVAL's
matching docstring addendum for why (same name-pooling collision risk)."""
PPI_LOGIT_T_SINGLE = Method("ppi_logit_t_single", "#ccebc5")  # lighter tint of PPI_LOGIT_T
"""Single-sample sibling of PPI_LOGIT_T (evalstats.tests._ppi_single_logit_t),
split out for the same reason PPI_BOOTSTRAP_T_SINGLE is split from
BOOTSTRAP_T -- see PPI_T_INTERVAL_SINGLE's docstring (identical reasoning,
[0,1]-bounded instead of unbounded). PPI_AUTO_METHOD_TABLE's "bounded_01"
robustness method -- the non-binary counterpart to PPI_WILSON's binary
role; every real dataset ppi_real.py checks is already rescaled to [0, 1]
(see RealJudgeBiasCorpus), so this applies uniformly there."""
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
# MWU family: five PPI corrections for the same classical test (Mann-Whitney
# U / independent two-group mid-rank estimand P_mid(A>B)-0.5), matching
# evalstats.tests.mannwhitney's "method" values one-to-one -- see that
# function's docstring for the full mechanism/tradeoff of each. MWU="global"
# (the default), MWU_MNAR_EXPERIMENTAL="mnar_experimental",
# MWU_MNAR_POOLED (not directly selectable via mannwhitney(), the pooled-
# resampling variant "local" is built on), MWU_ADAPTIVE="adaptive",
# MWU_RIDGE="ridge". cases/ppi_real.py's twogroup check dropped
# MWU_MNAR_EXPERIMENTAL from its default plots since real-data judge bias
# isn't MNAR, so there's nothing there for the local rectifier to buy over
# MWU -- see _twogroup_methods_for's docstring.
MWU = Method("mwu", "#2ca02c")
MWU_MNAR_EXPERIMENTAL = Method("mwu_mnar_experimental", "#9467bd")
MWU_MNAR_POOLED = Method("mwu_mnar_pooled", "#c5b0d5")  # lighter tint of MWU_MNAR_EXPERIMENTAL's purple
MWU_ADAPTIVE = Method("mwu_adaptive", "#98df8a")  # light tint of MWU's green
MWU_RIDGE = Method("mwu_ridge", "#c49c94")  # muted brown -- distinct from the MWU family's greens/purples
ANOVA_IND = Method("anova_ind", "#e6550d")
ANOVA_REP = Method("anova_rep", "#fd8d3c")
FRIEDMAN = Method("friedman", "#756bb1")  # purple -- distinct from the anova_*/lmm_* families
# KRUSKAL/KRUSKAL_MNAR_EXPERIMENTAL: two PPI corrections for the same
# omnibus test, the MWU/MWU_MNAR_EXPERIMENTAL story generalized one level up
# (k independent groups instead of 2) -- see
# evalstats.tests.kruskalwallis's "method" docstring for the full
# mechanism/tradeoff. KRUSKAL="global" (the default, global rectifier),
# KRUSKAL_MNAR_EXPERIMENTAL="mnar_experimental" (local rectifier: fixes MNAR
# labeling at the cost of MCAR calibration, kept for direct comparison and
# for anyone deliberately studying MNAR robustness). Same color convention
# as MWU/MWU_MNAR_EXPERIMENTAL: the default occupies the original primary
# shade, the alternate gets a lighter tint.
KRUSKAL = Method("kruskal", "#e377c2")  # pink -- distinct from the anova_*/lmm_* families
KRUSKAL_MNAR_EXPERIMENTAL = Method("kruskal_mnar_experimental", "#f2b6d4")  # lighter tint
LMM = Method("lmm", "#74c476")
LMM_FACTORIAL = Method("lmm_factorial", "#a1d99b")
LMM_RUNS = Method("lmm_runs", "#c7e9c0")
PPI_TEST_METHODS = [
    TTEST, TTEST_WELCH, MWU, MWU_MNAR_EXPERIMENTAL, MWU_MNAR_POOLED, MWU_ADAPTIVE, MWU_RIDGE, WILCOXON, PAIRED_T, BAYES_BOOTSTRAP, BOOTSTRAP_T, TANGO, ANOVA_IND,
    ANOVA_REP, FRIEDMAN, KRUSKAL, KRUSKAL_MNAR_EXPERIMENTAL, LMM, LMM_FACTORIAL, LMM_RUNS, PPI_WILSON,
    PPI_BOOTSTRAP_T_SINGLE, PPI_T_INTERVAL, PPI_LOGIT_T, PPI_T_INTERVAL_SINGLE, PPI_LOGIT_T_SINGLE,
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
sibling, but cost real MCAR calibration doing so -- see
evalstats.tests.mannwhitney's and kruskalwallis's "method" docstrings).
Both remain selectable via --tests mwu_mnar_experimental / --tests
kruskal_mnar_experimental for direct comparison or studying MNAR robustness
deliberately.

lmm/lmm_factorial/lmm_runs are excluded from the official set: not
currently part of the reported result set, so there's no point paying their
runtime cost (and reviewing their output) in every official pass. Still
fully selectable via --tests lmm/lmm_factorial/lmm_runs for anyone who
wants them.

PPI_WILSON/PPI_BOOTSTRAP_T_SINGLE (the single-sample robustness-CI methods
PPI_AUTO_METHOD_TABLE routes to) are part of the default set too --
pvalues.py's synthetic PPI sweep has a single-arm effect-check scenario
(see cases/pvalues.py's _run_ppi_effect_cell single-arm blocks) so these
aren't validated only by cases/ppi_real.py's real-data check."""

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
    PPI_WILSON, PPI_BOOTSTRAP_T_SINGLE, PPI_T_INTERVAL_SINGLE, PPI_LOGIT_T_SINGLE,
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
