"""Shared scenario / data-source interface for the evalstats simulation harness.

A ``CISource`` represents one "population" with a known (or precisely
estimated) true mean from which samples of size ``n`` can be drawn. Synthetic
sources (``synthetic.py``) generate i.i.d. draws from a parametric
distribution; real-data sources (``real_data.py``) draw without replacement
from a fixed corpus of real eval scores (DOVE_Lite / OpenEval). Both expose
the same ``generate(rng, n) -> ndarray`` interface so a single simulation
loop (e.g. ``cases/ci_single.py``) can run over either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EVAL_TYPES = ["binary", "continuous", "likert", "grades"]


@dataclass
class CISource:
    label: str
    eval_type: str
    true_mean: float
    generate: Callable[[np.random.Generator, int], np.ndarray]
    source: str = "synthetic"  # "synthetic" | "dove" | "openeval"
    max_n: int | None = None  # corpus size bound for real-data sources; None = unbounded
    model: str | None = None
    benchmark_id: str | None = None
