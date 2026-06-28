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


@dataclass
class CIPairSource:
    """Paired-difference analogue of CISource (used by cases/ci_paired.py).

    ``generate_pair(rng, n, runs)`` returns ``(a, b)``, each shape ``(n,
    runs)`` -- one row per sampled input, one column per run. Synthetic
    sources draw i.i.d.; real-data sources (``scenarios/real_data.py``) draw
    without replacement from a fixed corpus pair and only support ``runs=1``
    in this pass (see harness README known exceptions).
    """

    label: str
    eval_type: str
    true_diff: float
    generate_pair: Callable[[np.random.Generator, int, int], tuple[np.ndarray, np.ndarray]]
    source: str = "synthetic"  # "synthetic" | "dove" | "openeval" | "inspect"
    max_n: int | None = None
    icc: float = 0.0
    cohens_d: float = 0.0
    is_null: bool = False
    model_a: str | None = None
    model_b: str | None = None
    benchmark_id: str | None = None
