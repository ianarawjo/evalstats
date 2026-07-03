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
BAYES_PAIR_INDEP = Method("bayes_indep_comp", "#ffbb78")
BAYES_PAIR_PAIRED = Method("bayes_paired_comp", "#98df8a")
PAIRWISE_EXTRA_METHODS = [T_INTERVAL, NIG, EL]
BINARY_PAIRWISE_EXTRA_METHODS = [NEWCOMBE, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED]

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

WILSON_DE = Method("wilson_de", "#a6761d")
WILSON_OD = Method("wilson_od", "#666666")
BETA_BINOMIAL = Method("beta_binomial", "#17becf")
BINARY_NESTED_METHODS = [WILSON_DE, WILSON_OD, BETA_BINOMIAL]

NIG_NESTED = Method("nig_nested", "#f7b6d2")
CONTINUOUS_NESTED_METHODS = [NIG_NESTED]

BOOTSTRAP_DIFF_NESTED = Method("bootstrap_diff_nested", "#1b9e77")
BAYES_DIFF_NESTED = Method("bayes_diff_nested", "#d95f02")
SMOOTH_DIFF_NESTED = Method("smooth_diff_nested", "#7570b3")
PAIR_DIFF_NESTED_METHODS = [BOOTSTRAP_DIFF_NESTED, BAYES_DIFF_NESTED, SMOOTH_DIFF_NESTED]

TANGO_FLAT = Method("tango_flat", "#e7298a")
NEWCOMBE_FLAT = Method("newcombe_flat", "#66a61e")
BINARY_PAIR_FLAT_METHODS = [TANGO_FLAT, NEWCOMBE_FLAT, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED]

TANGO_MULTIRUN_CLUSTER = Method("tango_multirun_cluster", "#e6ab02")
TANGO_MULTIRUN_EFFECTIVE = Method("tango_multirun_effective", "#a6761d")
TANGO_MULTIRUN_MOMENTS = Method("tango_multirun_mmnt", "#1b9e77")
BINARY_PAIR_NESTED_METHODS = [TANGO_MULTIRUN_CLUSTER, TANGO_MULTIRUN_EFFECTIVE, TANGO_MULTIRUN_MOMENTS]

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
# ---------------------------------------------------------------------------
CORR_NONE = Method("none", "#9c9ede")
CORR_HOLM = Method("holm", "#cedb9c")
CORR_BONFERRONI = Method("bonferroni", "#e7ba52")
CORR_FDR_BH = Method("fdr_bh", "#ad494a")
CORR_FRIEDMAN_NEMENYI = Method("friedman_nemenyi", "#a55194")
CORR_MAX_T = Method("max_t", "#5254a3")
MULTIARM_CORRECTION_METHODS = [CORR_NONE, CORR_HOLM, CORR_BONFERRONI, CORR_FDR_BH, CORR_FRIEDMAN_NEMENYI, CORR_MAX_T]

# ---------------------------------------------------------------------------
# cases/pvalues.py -- evalstats.tests wrapper names (PPI-corrected path),
# ported from sim_type_i_calibration.py's TEST_NAMES. Each is run twice per
# scenario (uncorrected -- the scipy-equivalent test on LLM-only scores -- and
# PPI-corrected -- the same test with sparse human labels) to check whether
# PPI correction fixes the Type-I inflation judge bias/miscalibration causes.
# WILCOXON is shared with the pairwise non-PPI registry above: same
# underlying paired-difference test, just on different data structures.
# ---------------------------------------------------------------------------
TTEST = Method("ttest", "#3182bd")
TTEST_WELCH = Method("ttest_welch", "#6baed6")
MW = Method("mw", "#9ecae1")
ANOVA_IND = Method("anova_ind", "#e6550d")
ANOVA_REP = Method("anova_rep", "#fd8d3c")
FRIEDMAN = Method("friedman", "#fdae6b")
KRUSKAL = Method("kruskal", "#31a354")
LMM = Method("lmm", "#74c476")
LMM_FACTORIAL = Method("lmm_factorial", "#a1d99b")
LMM_RUNS = Method("lmm_runs", "#c7e9c0")
PPI_TEST_METHODS = [
    TTEST, TTEST_WELCH, MW, WILCOXON, ANOVA_IND, ANOVA_REP, FRIEDMAN, KRUSKAL,
    LMM, LMM_FACTORIAL, LMM_RUNS,
]

# ---------------------------------------------------------------------------
# Registry -- canonical ordering for tables/legends, and name -> Method lookup
# ---------------------------------------------------------------------------
REPORT_METHOD_ORDER: list[Method] = BOOTSTRAP_METHODS + [
    T_INTERVAL, WILSON, JEFFREYS, NEWCOMBE, TANGO,
    WALD, CLOPPER_PEARSON, BAYES_SINGLE, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED,
] + CONTINUOUS_EXTRA_METHODS + NESTED_METHODS + BINARY_FLAT_METHODS + BINARY_NESTED_METHODS + (
    CONTINUOUS_NESTED_METHODS + PAIR_DIFF_NESTED_METHODS
    + [TANGO_FLAT, NEWCOMBE_FLAT] + BINARY_PAIR_NESTED_METHODS
) + [
    MCNEMAR, PERMUTATION, SIGN_TEST, NEWCOMBE_PVAL, BAYES_BINARY, WILCOXON, PAIRED_T,
] + MULTIARM_CORRECTION_METHODS + [
    TTEST, TTEST_WELCH, MW, ANOVA_IND, ANOVA_REP, FRIEDMAN, KRUSKAL, LMM, LMM_FACTORIAL, LMM_RUNS,
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
