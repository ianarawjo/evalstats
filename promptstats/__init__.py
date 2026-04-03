"""promptstats: Statistical analysis and visualization for prompt benchmarking."""

from promptstats.core.types import BenchmarkResult, MultiModelBenchmark
from promptstats.core.paired import pairwise_differences, all_pairwise, vs_baseline, friedman_nemenyi, FriedmanResult
from promptstats.core.ranking import bootstrap_ranks, bootstrap_point_advantage
from promptstats.core.variance import (
    robustness_metrics,
    seed_variance_decomposition,
    SeedVarianceResult,
)
from promptstats.core.router import (
    analyze,
    analyze_factorial,
    AnalysisBundle,
    AnalysisResult,
    BenchmarkShape,
    MultiModelBundle,
)
from promptstats.core.summary import print_analysis_summary, print_brief_summary
from promptstats.vis.advantage import plot_point_advantage
from promptstats.vis.critical_difference import plot_critical_difference
from promptstats.vis.forest import plot_ci_forest
from promptstats.vis.scoreboard import plot_accuracy_bar
from promptstats.io import from_dataframe, DataLoadReport
from promptstats.core.resampling import bayes_binary_ci_1d, bayes_paired_diff_ci
from promptstats.core import bayes_evals
from promptstats.compare import (
    compare_prompts,
    compare_models,
    CompareReport,
    EntityStats,
)
from promptstats.config import set_alpha_ci, get_alpha_ci

__version__ = "0.1.7"

__all__ = [
    "BenchmarkResult",
    "MultiModelBenchmark",
    "pairwise_differences",
    "all_pairwise",
    "vs_baseline",
    "friedman_nemenyi",
    "FriedmanResult",
    "bootstrap_ranks",
    "bootstrap_point_advantage",
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
    "plot_point_advantage",
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
#   from promptstats.core.mixed_effects import LMMInfo, FactorialLMMInfo
# or inspect bundle.lmm_info / bundle.factorial_lmm_info at runtime.
try:
    from promptstats.core.mixed_effects import LMMInfo, FactorialLMMInfo
    __all__ = __all__ + ["LMMInfo", "FactorialLMMInfo"]
except ImportError:
    pass
