"""Shared scenario / data-source interface for the evalstats simulation harness.

A ``CISource`` represents one "population" with a known (or precisely
estimated) true mean from which samples of size ``n`` can be drawn. Synthetic
sources (``synthetic.py``) generate i.i.d. draws from a parametric
distribution; real-data sources (``real_data.py``) draw without replacement
from a fixed corpus of real eval scores (OpenEval / Inspect AI). Both expose
the same ``generate(rng, n) -> ndarray`` interface so a single simulation
loop (e.g. ``cases/ci_single.py``) can run over either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EVAL_TYPES = ["binary", "continuous", "likert", "grades"]

# Canonical (lo, hi) natural-scale bounds per eval type -- see synthetic.py's
# per-shape generators (grades/likert are clipped to exactly these ranges).
# Used to rescale data onto [0, 1] before calling CI methods whose domain or
# prior (e.g. nig_ci_1d, logit_t_ci_1d) assumes a [0, 1]-bounded scale, since
# only "binary"/"continuous" are natively [0, 1] -- "likert" (1-5) and
# "grades" (0-100) are not.
EVAL_TYPE_SCALE_BOUNDS: dict[str, tuple[float, float]] = {
    "binary": (0.0, 1.0),
    "continuous": (0.0, 1.0),
    "likert": (1.0, 5.0),
    "grades": (0.0, 100.0),
}


@dataclass
class CISource:
    label: str
    eval_type: str
    true_mean: float
    generate: Callable[[np.random.Generator, int], np.ndarray]
    source: str = "synthetic"  # "synthetic" | "openeval" | "inspect" | "real"
    max_n: int | None = None  # corpus size bound for real-data sources; None = unbounded
    model: str | None = None
    benchmark_id: str | None = None
    generate_runs: Callable[[np.random.Generator, int, int], np.ndarray] | None = None
    """Multi-run analogue of ``generate``: ``generate_runs(rng, n, runs) -> (n, runs)``
    ndarray. Set only by ``scenarios.synthetic.build_single_sample_sources``'s
    ``run_noise_fracs`` param (used by ``cases/ci_single.py``'s
    ``--nested-mode``); ``None`` for ordinary flat sources."""
    run_noise_frac: float | None = None
    """f_run = sigma^2_run / (sigma^2_input + sigma^2_run); set alongside ``generate_runs``."""


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
    source: str = "synthetic"  # "synthetic" | "openeval" | "inspect" | "real"
    max_n: int | None = None
    icc: float = 0.0
    cohens_d: float = 0.0
    is_null: bool = False
    model_a: str | None = None
    model_b: str | None = None
    benchmark_id: str | None = None
    run_noise_frac: float = 0.0
    run_noise_frac_a: float = 0.0
    run_noise_frac_b: float = 0.0
    """Set (alongside ``icc``) by ``scenarios.synthetic.build_pair_sources``'s
    ``run_noise_fracs`` param, used by ``cases/ci_paired.py``'s
    ``--nested-mode``; 0.0 for ordinary sources."""


@dataclass
class MultiArmSource:
    """K-arm analogue of CIPairSource (used by cases/pvalues.py's multi-arm mode).

    ``generate_scores(rng, n, runs, k, delta)`` returns an ndarray of shape
    ``(k, n, runs)`` -- arm 0 carries the alternative-hypothesis shift
    ``delta``; arms ``1..k-1`` stay at baseline. Built by
    ``scenarios.synthetic.build_multiarm_sources`` (synthetic) or
    ``scenarios.real_data.build_real_multiarm_sources`` (real data: arm 0 is
    instead the empirically best-performing real model on that benchmark,
    ``delta`` is only used as a null/alt switch -- 0.0 selects a per-item
    permutation null, anything else selects the real, unpermuted arms -- and
    R=1 only; see harness README known exceptions).
    """

    label: str
    eval_type: str
    generate_scores: Callable[[np.random.Generator, int, int, int, float], np.ndarray]
    alt_delta: float
    source: str = "synthetic"  # "synthetic" | "openeval" | "inspect" | "real"
    max_n: int | None = None  # corpus size bound for real-data sources; None = unbounded
    max_k: int | None = None  # max real arms (models) available; None = unbounded
    benchmark_id: str | None = None
    true_means: Callable[[int, float], np.ndarray] | None = None
    """``true_means(k, delta) -> ndarray`` of shape ``(k,)``: the ground-truth
    per-arm mean for the SAME ``(k, delta)`` a ``generate_scores`` call would
    use, in the same arm-index order. Used by ``cases/pvalues.py``'s
    ``--mode simultaneous_ci`` to check whether a constructed simultaneous CI
    actually contains each pair's true difference (``true_means[i] -
    true_means[j]``). Synthetic sources Monte-Carlo-estimate this (same
    technique as ``scenarios.synthetic._estimate_true_pair_diff``, since
    clipping/rounding can shift the realized mean away from the raw additive
    ``effects`` shift); real sources return the exact aligned-corpus means
    (or an all-equal vector for the ``delta == 0.0`` permutation null, whose
    construction makes every arm's true value identical by design). ``None``
    for sources that don't support this (there are none as of this writing --
    every ``MultiArmSource`` builder sets it)."""


@dataclass(frozen=True)
class JudgeBiasSource:
    """One judge-bias/miscalibration scenario for cases/pvalues.py's PPI mode.

    Originally mirrored ``sim_type_i_calibration.py``'s ``Scenario`` dataclass
    field for field, including its own narrower truth-distribution axis
    ("normal"/"likert"/"skewed", unrelated to ``EVAL_TYPES``). That axis was
    later unified onto ``EVAL_TYPES`` (binary excluded -- ttest/MWU aren't
    designed for 0/1 outcomes) so the harness has ONE truth-generating
    process per eval type shared by ``build_pair_sources``,
    ``build_multiarm_sources``, and this scenario family (one representative
    ``ShapeSpec`` per eval type, via ``_ppi_shape``), with judge bias/noise/
    MNAR-labels layered on top as an orthogonal sweep -- see the harness
    README's Known exceptions for the rationale. Built by
    ``scenarios.synthetic.build_judge_bias_sources``; consumed via
    ``scenarios.synthetic.generate_judge_bias_cell(source, rng)``, which
    draws ground truth (via ``sample_group_truth``), LLM-judge scores (with
    bias/slope distortion), and sparse (possibly MNAR) human labels for
    every test structure (two independent groups, paired/repeated, three
    independent groups, three repeated conditions, a 2x2 crossed-factor
    design, and nested LLM runs)
    in one call, in a fixed draw order, so a single ``rng`` stream feeds every
    active test deterministically.
    """

    name: str
    tag: str
    eval_type: str = "continuous"  # "continuous" | "likert" | "grades" | "binary" (binary: mean-based tests only, see above)
    icc: float = 0.20
    """Intraclass correlation for the paired/repeated test structures' truth
    model (same meaning as build_pair_sources'/build_multiarm_sources' icc:
    between-subject variance / total variance). Not currently swept across
    scenarios -- fixed at a baseline consistent with the rest of the harness."""
    n: int = 100
    n2: int | None = None
    n3: int | None = None
    label_frac: float = 0.20
    llm_noise: float = 0.20
    llm_noise2: float | None = None
    llm_noise3: float | None = None
    bias_type: str = "differential"  # "none" | "constant" | "differential"
    bias_delta: float = 0.30
    bias_const: float = 0.40
    bias_extra_a: float = 0.0
    bias_extra_b: float = 0.0
    bias_extra_c: float = 0.0
    bias_extra_d: float = 0.0
    slope_a: float = 1.0
    slope_b: float = 1.0
    slope_c: float = 1.0
    slope_d: float = 1.0
    label_mnar: bool = False
    mnar_strength: float = 1.0
    mnar_mode: str = "high"  # "high" | "extreme"
    repeated_corr: float = 0.0
    effect_size: float = 0.0
    shape_label: str | None = None
    """Override for which ShapeSpec (from scenarios.synthetic.SHAPES_BY_EVAL_TYPE[eval_type])
    to draw truth from -- None (the default) uses the eval type's fixed
    representative "param" shape (scenarios.synthetic._PPI_REPRESENTATIVE_SHAPE_LABEL);
    set explicitly to sweep a "custom" (mixture/inflated/heavy-tail) shape
    instead, so PPI-correction calibration gets checked against a
    pathological truth distribution too, not just smooth Beta/Normal ones."""
    noise_family: str = "gaussian"  # "gaussian" | "contaminated"
    """LLM-judge measurement-error noise family (scenarios.synthetic._jb_llm/
    _jb_llm_repeated). "gaussian" (the default, used everywhere except the
    noise_family.* scenarios below) is symmetric, uniform-width measurement
    error. "contaminated" is a zero-mean two-component variance-mixture --
    mostly a small width, occasionally (probability contam_frac) a much
    larger one -- modeling "judge is mostly right, occasionally
    catastrophically wrong" instead. Total noise variance (and therefore
    llm_noise's calibrated meaning) is identical between the two families;
    only how that variance is distributed across items differs."""
    contam_frac: float = 0.10
    """"contaminated" noise_family only: probability an item's judge error
    is drawn from the large-width (catastrophic) component."""
    contam_scale: float = 4.0
    """"contaminated" noise_family only: the catastrophic component's std is
    this multiple of the normal component's std (solved jointly with
    contam_frac so total variance stays fixed -- see _contaminated_noise_stds)."""
