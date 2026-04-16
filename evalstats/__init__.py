"""evalstats: Statistical analysis and visualization for prompt benchmarking."""

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
from evalstats.vis.point_estimates import plot_point_estimates
from evalstats.vis.critical_difference import plot_critical_difference
from evalstats.vis.forest import plot_ci_forest
from evalstats.vis.scoreboard import plot_accuracy_bar
from evalstats.io import from_dataframe, DataLoadReport
from evalstats.core.resampling import bayes_binary_ci_1d, bayes_paired_diff_ci
from evalstats.core import bayes_evals
from evalstats.compare import (
    compare_prompts,
    compare_models,
    CompareReport,
    EntityStats,
)
from evalstats.config import set_alpha_ci, get_alpha_ci

__version__ = "0.1.9"

__all__ = [
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
    "plot_point_estimates",
    "plot_critical_difference",
    "plot_ci_forest",
    "plot_accuracy_bar",
    "from_dataframe",
    "DataLoadReport",
    "bayes_binary_ci_1d",
    "bayes_paired_diff_ci",
    "bayes_evals",
    "compare_prompts",
    "compare_models",
    "CompareReport",
    "EntityStats",
    "analyze_factorial",
    "set_alpha_ci",
    "get_alpha_ci",
]

# LMMInfo and FactorialLMMInfo are exported lazily so that statsmodels/pymer4
# are not hard dependencies.  Access via:
#   from evalstats.core.mixed_effects import LMMInfo, FactorialLMMInfo
# or inspect bundle.lmm_info / bundle.factorial_lmm_info at runtime.
try:
    from evalstats.core.mixed_effects import LMMInfo, FactorialLMMInfo
    __all__ = __all__ + ["LMMInfo", "FactorialLMMInfo"]
except ImportError:
    pass
