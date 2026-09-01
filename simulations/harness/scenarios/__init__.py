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

DEFAULT_EVAL_TYPES = ["binary", "continuous", "likert"]
"""The eval types the official presets actually sweep. "grades" is a valid
scenario type (EVAL_TYPES above) but is deliberately NOT swept: it is
"continuous" rescaled to 0-100, so it adds a third of the runtime for no
coverage the continuous column does not already give, while "likert" is kept
as the genuinely distinct integer/few-level case.

Cases must use this as their --eval-types default rather than falling through
to EVAL_TYPES. Leaving it to None meant a bare CLI run swept four types while
the official preset swept three, with nothing in the output saying which you
got -- a multi-run pairwise table in the paper carried 15 grades rows that
the official test would never have produced."""

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
    scale_bounds: tuple[float, float] | None = None
    """The (lo, hi) range this source's scores actually live on, when it is
    NOT the eval type's canonical range (``EVAL_TYPE_SCALE_BOUNDS``).

    Needed by ``cases/ci_unpaired.py``'s bounded-scale methods (mover_logit_t,
    mover_nig), which have to map an arm onto [0, 1] before applying a
    logit/NIG interval. Real corpora do not respect the synthetic ranges --
    e.g. the privacy_judge human labels are continuous but live on ~[1, 5],
    not the synthetic "continuous" [0, 1] -- and rescaling one of those with
    the wrong bounds silently produces garbage rather than an error.
    ``None`` means "use the eval type's canonical range"."""


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

    ``bias_delta``/``bias_extra_*``/``slope_*`` model judge miscalibration as
    an AFFINE function of the group's true score -- a single constant shift
    plus a single constant stretch/compression, the same for every item in
    a group. ``confound_weight``/``confound_truth_corr``/``confound_shift_*``
    are a structurally different (not a replacement) bias mechanism: a
    per-item nuisance covariate -- e.g. response length -- correlated with,
    but distinct from, true quality, that can differ systematically between
    compared groups for reasons that have nothing to do with quality. An
    affine-in-truth model can only ever move a group as a whole; it can't
    reproduce "this group's responses happen to be longer, and the judge
    rewards length regardless of actual quality." See
    ``scenarios.synthetic._confound_latent`` for the generative model and
    ``build_judge_bias_sources``' ``"confound.*"`` scenarios for the sweep.
    """

    name: str
    tag: str
    eval_type: str = "continuous"  # "continuous" | "likert" | "grades" | "binary" (binary: mean-based tests only, see above)
    likert_max: int = 5
    """Top of the Likert scale's integer range (bottom is always fixed at 1)
    -- only meaningful when eval_type=="likert", ignored otherwise. The
    default (5) reproduces every existing Likert scenario exactly. Setting
    this to e.g. 7 rescales the SAME underlying "param"-shape distribution
    (same relative mean position, spread, judge bias, and effect size as the
    1-5 version -- see scenarios.synthetic.sample_group_truth's likert_max
    parameter and _jb_bias_magnitude/_jb_effect_magnitude's scale_bounds
    override) onto a wider integer grid, rather than generating a different
    distribution -- lets PPI judge-bias scenarios test whether a coarser
    (fewer-level) Likert scale is itself responsible for a calibration
    issue. Only "param" Likert shapes support this (sample_group_truth
    raises if combined with a "custom" shape like likert-bimodal). Deliberately
    NOT threaded through EVAL_TYPE_SCALE_BOUNDS -- that harness-wide constant
    still assumes 1-5 for the pairwise/multiarm/CI code that also reads it."""
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
    confound_weight: float = 0.0
    """How strongly a per-item nuisance covariate (e.g. response length --
    correlated with, but distinct from, true quality) drives the judge's
    score, independent of bias_delta/slope_*. 0.0 (the default) disables
    the mechanism entirely -- every pre-existing scenario is bit-for-bit
    unaffected. See scenarios.synthetic._confound_latent for the generative
    model and JudgeBiasSource's class docstring's "confound-driven bias"
    note for why this is a structurally different stress test than
    bias_delta/slope_* (those two can only stretch or shift a group as a
    whole; this varies item-to-item). Units differ by eval_type, mirroring
    bias_delta's own split: an additive score-scale term for continuous/
    likert/grades (typically set via _jb_bias_magnitude, same convention
    bias_delta uses), but for binary it sits inside _jb_llm_binary's
    bias/2 flip-probability-skew term instead (see that function's `extra`
    parameter) -- use PPI_BINARY_BIAS_MAGNITUDES' ~0.10-0.30 scale there,
    not _jb_bias_magnitude."""
    confound_truth_corr: float = 0.0
    """Gaussian-copula correlation between the confound and standardized
    truth (0 = confound is pure nuisance, unrelated to quality; nonzero,
    e.g. "longer answers tend to be a bit better too," makes the confound
    harder to distinguish from ordinary judge bias). Linear/copula only --
    see _confound_latent's docstring for why this harness doesn't attempt
    non-monotonic (e.g. U-shaped) truth-confound relationships. Not
    swept independently of confound_weight in build_judge_bias_sources'
    "confound.*" scenarios -- see that group's comment."""
    confound_shift_a: float = 0.0
    confound_shift_b: float = 0.0
    confound_shift_c: float = 0.0
    confound_shift_d: float = 0.0
    """Per-group mean shift in the confound itself -- same group-letter
    convention bias_extra_a/b/c/d already uses. This is what turns the
    confound from inert per-item noise (which averages out across N and
    can't miscalibrate a mean-comparison test on its own) into a genuine
    between-group nuisance difference: e.g. "group A's responses average
    longer than group B's, for reasons unrelated to quality" -- something
    an affine-in-truth bias_delta/slope model structurally cannot express,
    since it only ever moves a group as a whole, uniformly."""
