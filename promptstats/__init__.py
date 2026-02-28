"""promptstats: Statistical analysis and visualization for prompt benchmarking."""

from promptstats.core.types import BenchmarkResult
from promptstats.core.paired import pairwise_differences, all_pairwise, vs_baseline
from promptstats.core.ranking import bootstrap_ranks, bootstrap_mean_advantage
from promptstats.core.variance import robustness_metrics
from promptstats.core.router import (
    analyze,
    AnalysisBundle,
    AnalysisResult,
    BenchmarkShape,
    print_analysis_summary,
)
from promptstats.vis.advantage import plot_mean_advantage

__version__ = "0.1.0"

__all__ = [
    "BenchmarkResult",
    "pairwise_differences",
    "all_pairwise",
    "vs_baseline",
    "bootstrap_ranks",
    "bootstrap_mean_advantage",
    "robustness_metrics",
    "analyze",
    "AnalysisBundle",
    "AnalysisResult",
    "BenchmarkShape",
    "print_analysis_summary",
    "plot_mean_advantage",
]
