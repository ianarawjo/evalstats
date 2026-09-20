"""evalstats: Statistical analysis and visualization for prompt benchmarking."""

# ── Core types and analysis engine ───────────────────────────────────────────
from evalstats.core.types import BenchmarkResult, MultiModelBenchmark
from evalstats.core.paired import pairwise_differences, all_pairwise, vs_baseline, friedman_nemenyi, FriedmanResult
from evalstats.core.ranking import bootstrap_ranks
from evalstats.core.variance import (
    robustness_metrics,
    seed_variance_decomposition,
    SeedVarianceResult,
)
from evalstats.core.router import (
    analyze,
    analyze_factorial,
    AnalysisBundle,
    AnalysisResult,
    BenchmarkShape,
    MultiModelBundle,
)
from evalstats.core.summary import print_analysis_summary, print_brief_summary
from evalstats.io import from_dataframe, DataLoadReport
from evalstats.errors import (
    MIN_ITEMS,
    AmbiguousLabelsError,
    InsufficientItemsError,
    MissingCellsError,
    TooFewGroupsError,
)
from evalstats._notes import Note
from evalstats.core.resampling import bayes_binary_ci_1d, bayes_paired_diff_ci
from evalstats.core import bayes_evals
from evalstats.config import set_alpha_ci, get_alpha_ci

# ── High-level spec API ───────────────────────────────────────────────────────
# Must come after all other imports: importing evalstats.api triggers
# evalstats.compare submodule registration, which would shadow a bare
# "compare" name if it were imported before the submodule.
from evalstats.loader import load_from, EvalResults, EvalLoadError
from evalstats.completeness import complete_items, CompletenessReport
from evalstats.api import compare, compare_models, compare_prompts, ComparisonResult
from evalstats.alignment import judge_alignment, AlignmentResult
from evalstats import ppi
from evalstats import tests
from evalstats.quick import (
    mean_ci,
    MeanCI,
    summarize,
    GroupSummary,
    stability,
    StabilityResult,
    tradeoff,
    TradeoffResult,
    judge_debias_mean_ci,
    DebiasedMeanCI,
)

__version__ = "0.3.3"

__all__ = [
    # High-level spec API
    "load_from",
    "judge_alignment",
    "AlignmentResult",
    "ppi",
    "tests",
    "EvalResults",
    "EvalLoadError",
    "complete_items",
    "CompletenessReport",
    "compare",
    "compare_models",
    "compare_prompts",
    "ComparisonResult",
    # Quick primitives
    "mean_ci",
    "MeanCI",
    "summarize",
    "GroupSummary",
    "stability",
    "StabilityResult",
    "tradeoff",
    "TradeoffResult",
    "judge_debias_mean_ci",
    "DebiasedMeanCI",
    # Core types
    "BenchmarkResult",
    "MultiModelBenchmark",
    "pairwise_differences",
    "all_pairwise",
    "vs_baseline",
    "friedman_nemenyi",
    "FriedmanResult",
    "bootstrap_ranks",
    "robustness_metrics",
    "seed_variance_decomposition",
    "SeedVarianceResult",
    "analyze",
    "AnalysisBundle",
    "AnalysisResult",
    "BenchmarkShape",
    "MultiModelBundle",
    "print_analysis_summary",
    "print_brief_summary",
    "MIN_ITEMS",
    "AmbiguousLabelsError",
    "InsufficientItemsError",
    "MissingCellsError",
    "TooFewGroupsError",
    "Note",
    "plot_point_estimates",
    "plot_critical_difference",
    "plot_ci_forest",
    "plot_accuracy_bar",
    "from_dataframe",
    "DataLoadReport",
    "bayes_binary_ci_1d",
    "bayes_paired_diff_ci",
    "bayes_evals",
    "analyze_factorial",
    "set_alpha_ci",
    "get_alpha_ci",
]

# Plotting functions load matplotlib on first use, not at import.
_LAZY_PLOTS = {
    "plot_point_estimates": "evalstats.vis.point_estimates",
    "plot_critical_difference": "evalstats.vis.critical_difference",
    "plot_ci_forest": "evalstats.vis.forest",
    "plot_accuracy_bar": "evalstats.vis.scoreboard",
}


def __getattr__(name):
    if name in _LAZY_PLOTS:
        import importlib
        func = getattr(importlib.import_module(_LAZY_PLOTS[name]), name)
        globals()[name] = func
        return func
    raise AttributeError(f"module 'evalstats' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_PLOTS))


# LMMInfo and FactorialLMMInfo are exported lazily so that statsmodels/pymer4
# are not hard dependencies.  Access via:
#   from evalstats.core.mixed_effects import LMMInfo, FactorialLMMInfo
# or inspect bundle.lmm_info / bundle.factorial_lmm_info at runtime.
try:
    from evalstats.core.mixed_effects import LMMInfo, FactorialLMMInfo
    __all__ = __all__ + ["LMMInfo", "FactorialLMMInfo"]
except ImportError:
    pass
