"""Canonical synthetic single-sample scenario library.

Ported from ``simulations/sim_compare_boot.py``'s ``build_scenarios()``
(non-pairwise branch) — see that module's history for provenance. This is
the canonical synthetic distribution suite used by ``cases/ci_single.py``
and intended to be the *one* place future cases (``ci_paired``, ``ci_nested``,
``pvalues``) pull single-sample-equivalent generators from, so the paper can
describe these distributions once.

Eval types
----------
binary      Bernoulli 0/1 (pass/fail judgements)
continuous  Continuous floats in [0, 1] via Beta / logit-normal / inflated mixtures
likert      Integer scores 1-5 (Likert-scale rubrics), latent-normal model
grades      Scores 0-100 (test-like), truncated normal / mixture / heavy-tail

Suites
------
``standard`` is a compact suite covering the main shape families per eval
type. ``expanded`` adds boundary/extreme/mixture variants. ``extreme`` is
reserved for future stress-test scenarios (currently identical to
``expanded`` — see [[Known exceptions]] in the harness README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from . import CISource, CIPairSource, MultiArmSource, JudgeBiasSource, EVAL_TYPES

SCENARIO_SUITES = ["standard", "expanded", "extreme"]
RUN_NOISE_FRACS_DEFAULT = [0.01, 0.1, 0.3, 0.5]

# Latent-scale total standard deviations for Likert / Grades, shared by
# build_pair_sources, build_multiarm_sources, and build_judge_bias_sources'
# truth model -- fixes the total marginal std for each score type so a
# given (icc, cohens_d) means the same thing everywhere it's used.
_LIKERT_TOTAL_STD = 1.2  # latent scale, maps to {1,...,5} after rounding
_GRADES_TOTAL_STD = 20.0  # [0, 100] scale


def _true_mean_clipped_normal(
    mu: float, sigma: float, lo: float = 0.0, hi: float = 100.0
) -> float:
    """Population mean of Normal(mu, sigma) clipped to [lo, hi] via large sample."""
    rng = np.random.default_rng(0)
    return float(np.clip(rng.normal(mu, sigma, size=2_000_000), lo, hi).mean())


def _estimate_true_mean_mc(
    generate: Callable[[np.random.Generator, int], np.ndarray],
    *,
    seed: int = 0,
    n_mc: int = 500_000,
) -> float:
    """Estimate population mean via large Monte Carlo draw for complex generators.

    Uses n_mc=500,000 samples. For all scenarios used here the resulting MC
    standard error is < 0.001 (well below the CI widths under study), so
    estimand error is negligible relative to method-comparison noise.
    """
    rng = np.random.default_rng(seed)
    return float(np.mean(generate(rng, n_mc)))


def _binary_conc_from_icc(icc: float) -> float:
    """Beta concentration for Bernoulli input probabilities giving target ICC.

    For a Bernoulli-Beta hierarchical model where per-input success probabilities
    are drawn from Beta(conc*p0, conc*(1-p0)), the intraclass correlation of
    observed scores is ICC = 1/(conc + 1). Inverted: conc = 1/ICC - 1.
    """
    return max(1.0 / max(icc, 1e-9) - 1.0, 0.1)


def _beta_var(a: float, b: float) -> float:
    """Variance of Beta(a, b)."""
    return a * b / ((a + b) ** 2 * (a + b + 1))


def _estimate_true_pair_diff(
    generate_pair: Callable[[np.random.Generator, int, int], tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = 0,
    n_mc: int = 300_000,
) -> float:
    """Estimate E[cell_mean(A) - cell_mean(B)] via a large synthetic sample.

    Uses n_mc=300,000 items (single run). The resulting MC standard error on
    the true diff is negligible relative to the CI widths under study.
    """
    rng = np.random.default_rng(seed)
    a, b = generate_pair(rng, n_mc, 1)
    return float(np.mean(a[:, 0] - b[:, 0]))


# ---------------------------------------------------------------------------
# Unified per-eval-type shape catalog + truth/noise generator
# ---------------------------------------------------------------------------
# ONE shape catalog (binary/continuous/likert/grades) and ONE generator
# function (sample_group_truth) shared by build_single_sample_sources (k=1),
# build_pair_sources (k=2), build_multiarm_sources (k>=2), and
# build_judge_bias_sources' truth model (picks one representative shape) --
# so a given (icc, cohens_d) means the same thing everywhere, and the paper
# can describe one truth-generating process per eval type instead of one
# per simulation mode. Merges and supersedes both this file's previous,
# independently-evolved build_single_sample_sources catalog (richer shape
# variety: mixtures, inflated, heavy-tailed, bimodal) and build_pair_sources
# catalog (icc-parameterized, but narrower shape variety) -- the merged
# catalog keeps every shape family from both (see _legacy_* functions above
# for the pre-merge originals).
#
# "param" shapes are smooth parametric families that support ANY k via the
# icc/corr noise-layering below. "custom" shapes (mixtures, inflated, heavy-
# tailed, bimodal) are bespoke one-off generators that only make sense as a
# single population characteristic -- usable only at k=1, matching their
# scope before this unification (sim_compare_boot.py never had pair/multi-
# arm analogues for them either).


@dataclass(frozen=True)
class ShapeSpec:
    label: str
    eval_type: str
    kind: str  # "param" | "custom"
    params: tuple | float | None = None
    """binary: base_p (float); continuous: (a, b); likert/grades: (mu, total_std)."""
    custom_sampler: Callable[[np.random.Generator, int], np.ndarray] | None = None
    """For kind="custom": draws n FINAL (already clipped/rounded) values
    directly for one population/item draw. Any k and any base_corr are
    supported via sample_group_truth (base_corr<1.0 partial agreement uses
    an empirical-quantile copula -- see _custom_shape_inverse_cdf -- since
    these bespoke samplers have no closed-form .ppf())."""
    suite_tier: str = "standard"  # "standard" | "expanded"


def _tier_shapes(catalog: list[ShapeSpec], suite: str) -> list[ShapeSpec]:
    if suite in ("expanded", "extreme"):
        return catalog
    return [s for s in catalog if s.suite_tier == "standard"]


BINARY_SHAPES: list[ShapeSpec] = [
    ShapeSpec("p=0.10", "binary", "param", 0.10),
    ShapeSpec("p=0.20", "binary", "param", 0.20),
    ShapeSpec("p=0.30", "binary", "param", 0.30),
    ShapeSpec("p=0.50", "binary", "param", 0.50),
    ShapeSpec("p=0.70", "binary", "param", 0.70),
    ShapeSpec("p=0.80", "binary", "param", 0.80),
    ShapeSpec("p=0.90", "binary", "param", 0.90),
    ShapeSpec("p=0.92", "binary", "param", 0.92),
    ShapeSpec("p=0.02", "binary", "param", 0.02, suite_tier="expanded"),
    ShapeSpec("p=0.05", "binary", "param", 0.05, suite_tier="expanded"),
    ShapeSpec("p=0.95", "binary", "param", 0.95, suite_tier="expanded"),
    ShapeSpec("p=0.98", "binary", "param", 0.98, suite_tier="expanded"),
]

CONTINUOUS_SHAPES: list[ShapeSpec] = [
    ShapeSpec("cont-uniform", "continuous", "param", (1.0, 1.0)),
    ShapeSpec("cont-u-shaped", "continuous", "param", (0.5, 0.5)),
    ShapeSpec("cont-right-skew", "continuous", "param", (2.0, 8.0)),
    ShapeSpec("cont-left-skew", "continuous", "param", (8.0, 2.0)),
    ShapeSpec("cont-moderate-skew", "continuous", "param", (2.0, 5.0), suite_tier="expanded"),
    ShapeSpec("cont-boundary", "continuous", "param", (0.6, 0.6), suite_tier="expanded"),
    ShapeSpec("cont-extreme-right", "continuous", "param", (0.35, 6.0), suite_tier="expanded"),
    ShapeSpec("cont-extreme-left", "continuous", "param", (6.0, 0.35), suite_tier="expanded"),
    ShapeSpec("cont-near-boundaries", "continuous", "param", (0.3, 0.3), suite_tier="expanded"),
    ShapeSpec("cont-near-center", "continuous", "param", (6.0, 6.0), suite_tier="expanded"),
    ShapeSpec("cont-logit-normal", "continuous", "custom", custom_sampler=lambda rng, n: 1.0 / (1.0 + np.exp(-rng.normal(-0.35, 1.35, size=n)))),
    ShapeSpec(
        "cont-zero-inflated", "continuous", "custom",
        custom_sampler=lambda rng, n: np.where(rng.random(n) < 0.70, 0.0, rng.beta(2.0, 4.0, n)),
    ),
    ShapeSpec(
        "cont-one-inflated", "continuous", "custom",
        custom_sampler=lambda rng, n: np.where(rng.random(n) < 0.70, 1.0, rng.beta(4.0, 2.0, n)),
    ),
    ShapeSpec(
        "cont-mixture", "continuous", "custom", suite_tier="expanded",
        custom_sampler=lambda rng, n: np.where(
            rng.binomial(1, 0.55, size=n).astype(bool), rng.beta(0.5, 4.0, n), rng.beta(5.5, 1.2, n),
        ),
    ),
]

LIKERT_SHAPES: list[ShapeSpec] = [
    ShapeSpec("likert-mid", "likert", "param", (3.0, 1.2)),
    ShapeSpec("likert-low", "likert", "param", (2.2, 1.2)),
    ShapeSpec("likert-high", "likert", "param", (3.8, 1.2)),
    ShapeSpec("likert-uniform", "likert", "param", (3.0, 2.0)),
    ShapeSpec("likert-skewed-low", "likert", "param", (2.0, 1.1)),
    ShapeSpec("likert-skewed-high", "likert", "param", (4.0, 1.1)),
    ShapeSpec("likert-center-peaked", "likert", "param", (3.0, 0.55)),
    ShapeSpec("likert-floor", "likert", "param", (1.8, 1.2), suite_tier="expanded"),
    ShapeSpec("likert-near-floor", "likert", "param", (1.5, 0.65), suite_tier="expanded"),
    ShapeSpec("likert-near-ceiling", "likert", "param", (4.5, 0.65), suite_tier="expanded"),
    ShapeSpec("likert-flat-middle", "likert", "param", (3.0, 1.4), suite_tier="expanded"),
    ShapeSpec(
        "likert-bimodal", "likert", "custom",
        custom_sampler=lambda rng, n: np.clip(np.rint(np.where(rng.random(n) < 0.5, rng.normal(1.5, 0.65, n), rng.normal(4.5, 0.65, n))), 1.0, 5.0),
    ),
    ShapeSpec(
        "likert-bimodal-extreme", "likert", "custom", suite_tier="expanded",
        custom_sampler=lambda rng, n: np.clip(np.rint(np.where(rng.random(n) < 0.5, rng.normal(1.3, 0.50, n), rng.normal(4.7, 0.50, n))), 1.0, 5.0),
    ),
]

GRADES_SHAPES: list[ShapeSpec] = [
    ShapeSpec("grades-mid", "grades", "param", (55.0, 20.0)),
    ShapeSpec("grades-low", "grades", "param", (35.0, 20.0)),
    ShapeSpec("grades-high", "grades", "param", (78.0, 20.0)),
    ShapeSpec("grades-high-scoring", "grades", "param", (75.0, 15.0)),
    ShapeSpec("grades-ceiling-heavy", "grades", "param", (88.0, 10.0)),
    ShapeSpec("grades-floor-heavy", "grades", "param", (12.0, 10.0)),
    ShapeSpec("grades-ceiling", "grades", "param", (86.0, 20.0), suite_tier="expanded"),
    ShapeSpec("grades-floor", "grades", "param", (20.0, 20.0), suite_tier="expanded"),
    ShapeSpec("grades-very-high", "grades", "param", (92.0, 7.0), suite_tier="expanded"),
    ShapeSpec("grades-very-low", "grades", "param", (8.0, 7.0), suite_tier="expanded"),
    ShapeSpec("grades-high-variance", "grades", "param", (50.0, 34.0), suite_tier="expanded"),
    ShapeSpec(
        "grades-mixture", "grades", "custom",
        custom_sampler=lambda rng, n: np.clip(_grade_mixture_sampler(rng, n), 0.0, 100.0),
    ),
    ShapeSpec("grades-heavy-tail", "grades", "custom", custom_sampler=lambda rng, n: np.clip(52.0 + 16.0 * rng.standard_t(df=3.0, size=n), 0.0, 100.0)),
    ShapeSpec(
        "grades-zero-spiked", "grades", "custom", suite_tier="expanded",
        custom_sampler=lambda rng, n: np.where(rng.random(n) < 0.40, 0.0, np.clip(rng.normal(45.0, 20.0, n), 0.0, 100.0)),
    ),
    ShapeSpec(
        "grades-hundred-spiked", "grades", "custom", suite_tier="expanded",
        custom_sampler=lambda rng, n: np.where(rng.random(n) < 0.40, 100.0, np.clip(rng.normal(65.0, 18.0, n), 0.0, 100.0)),
    ),
]

SHAPES_BY_EVAL_TYPE: dict[str, list[ShapeSpec]] = {
    "binary": BINARY_SHAPES, "continuous": CONTINUOUS_SHAPES, "likert": LIKERT_SHAPES, "grades": GRADES_SHAPES,
}


def _grade_mixture_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    flags = rng.choice(3, size=n, p=[0.20, 0.50, 0.30])
    vals = np.empty(n, dtype=float)
    for bucket, mu, sigma in [(0, 22.0, 11.0), (1, 58.0, 14.0), (2, 88.0, 8.0)]:
        mask = flags == bucket
        vals[mask] = rng.normal(mu, sigma, size=int(np.sum(mask)))
    return vals


def _group_noise_var(eval_type: str, params, icc: float) -> float:
    """Total per-group noise variance (Var(shared)+Var(indiv) at corr=0.5)
    implied by icc, for one "param" shape. Binary has no separate Gaussian
    noise layer (its within-item variability is the Bernoulli draw itself)."""
    if eval_type == "continuous":
        a_beta, b_beta = params
        var_base = _beta_var(a_beta, b_beta)
        return var_base * (1.0 / max(icc, 1e-9) - 1.0)
    if eval_type in ("likert", "grades"):
        _mu, total_std = params
        return (1.0 - icc) * total_std ** 2
    raise ValueError(f"_group_noise_var: unsupported eval_type {eval_type!r}")


def group_total_std(shape: ShapeSpec, icc: float) -> float:
    """Total marginal std of one shape's truth+noise model at a given icc --
    so cohens_d * group_total_std(...) is the same effect-size convention
    build_pair_sources/build_multiarm_sources/PPI all share. Custom shapes
    use the same "fixed base variance, noise grows as icc shrinks"
    convention as continuous param shapes, with an MC-estimated base
    variance (_custom_shape_var) standing in for the closed-form one."""
    if shape.kind == "custom":
        return float(np.sqrt(_custom_shape_var(shape) / max(icc, 1e-9)))
    eval_type, params = shape.eval_type, shape.params
    if eval_type == "binary":
        p0 = params
        return float(np.sqrt(p0 * (1.0 - p0)))
    if eval_type == "continuous":
        a_beta, b_beta = params
        return float(np.sqrt(_beta_var(a_beta, b_beta) / max(icc, 1e-9)))
    if eval_type in ("likert", "grades"):
        _mu, total_std = params
        return float(total_std)
    raise ValueError(f"group_total_std: unsupported eval_type {eval_type!r}")


_CUSTOM_SHAPE_VAR_CACHE: dict[str, float] = {}
_CUSTOM_SHAPE_MEAN_CACHE: dict[str, float] = {}
_CUSTOM_SHAPE_RANGE: dict[str, tuple[float, float, bool]] = {
    "continuous": (0.0, 1.0, False), "likert": (1.0, 5.0, True), "grades": (0.0, 100.0, False),
}


def _custom_shape_var(shape: ShapeSpec, *, n_mc: int = 200_000, seed: int = 0) -> float:
    """MC-estimated population variance of a "custom" shape's bare sampler
    (no run-noise added) -- the analogue of _beta_var(a,b)/total_std**2 for
    shapes with no closed-form variance. Memoized by label (custom_sampler
    is a pure function of (rng, n) for every shape in the catalog)."""
    if shape.label not in _CUSTOM_SHAPE_VAR_CACHE:
        rng = np.random.default_rng(seed)
        _CUSTOM_SHAPE_VAR_CACHE[shape.label] = float(np.var(shape.custom_sampler(rng, n_mc)))
    return _CUSTOM_SHAPE_VAR_CACHE[shape.label]


def _custom_shape_mean(shape: ShapeSpec, *, n_mc: int = 200_000, seed: int = 1) -> float:
    """MC-estimated population mean of a "custom" shape's bare sampler --
    the analogue of a/(a+b) (continuous) or mu (likert/grades) for shapes
    with no closed-form mean. Memoized by label."""
    if shape.label not in _CUSTOM_SHAPE_MEAN_CACHE:
        rng = np.random.default_rng(seed)
        _CUSTOM_SHAPE_MEAN_CACHE[shape.label] = float(np.mean(shape.custom_sampler(rng, n_mc)))
    return _CUSTOM_SHAPE_MEAN_CACHE[shape.label]


_CUSTOM_SHAPE_SORTED_SAMPLE_CACHE: dict[str, np.ndarray] = {}
_CUSTOM_SHAPE_QUANTILE_GRID_CACHE: dict[int, np.ndarray] = {}


def _custom_shape_inverse_cdf(shape: ShapeSpec, q: np.ndarray, *, n_mc: int = 200_000, seed: int = 2) -> np.ndarray:
    """Empirical inverse-CDF of a "custom" shape's bare sampler: maps
    quantiles in (0, 1) to values, standing in for a closed-form .ppf() that
    doesn't exist for bespoke samplers (mixtures, inflated, heavy-tail).
    Used by sample_group_truth's base_corr < 1.0 path for custom shapes --
    the same Gaussian-copula trick binary/continuous "param" shapes use
    (norm.cdf -> .ppf), just with this in place of stats.beta.ppf.

    The MC sample is cached and sorted once per shape (memoized by label,
    same pattern as _custom_shape_var/_custom_shape_mean); np.interp linearly
    interpolates between its order statistics, treated as a piecewise-linear
    empirical quantile function. Point masses (e.g. zero-inflated's 70%
    spike at 0) survive exactly: they occupy a contiguous run of the sorted
    sample, so any quantile landing inside that run interpolates between two
    copies of the same value and returns it unchanged.

    Caveat (documented in the harness README): this preserves *rank*
    correlation between groups exactly equal to base_corr (rank correlation
    is invariant under the monotonic copula-quantile mapping), but Pearson
    correlation only approximately, since the empirical quantile function is
    a nonlinear map for non-uniform shapes.
    """
    if shape.label not in _CUSTOM_SHAPE_SORTED_SAMPLE_CACHE:
        rng = np.random.default_rng(seed)
        _CUSTOM_SHAPE_SORTED_SAMPLE_CACHE[shape.label] = np.sort(shape.custom_sampler(rng, n_mc))
    if n_mc not in _CUSTOM_SHAPE_QUANTILE_GRID_CACHE:
        _CUSTOM_SHAPE_QUANTILE_GRID_CACHE[n_mc] = np.linspace(0.0, 1.0, n_mc)
    sorted_sample = _CUSTOM_SHAPE_SORTED_SAMPLE_CACHE[shape.label]
    grid = _CUSTOM_SHAPE_QUANTILE_GRID_CACHE[n_mc]
    return np.interp(q, grid, sorted_sample)


def _hetero_noise_scale(base: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Per-item heteroscedastic noise-std multiplier: scales toward 0 near
    the [lo, hi] boundaries (where item-level variance is naturally squeezed)
    and peaks at 1.0 at the midpoint -- same shape _legacy_build_multirun_sources'/
    _legacy_build_pair_multirun_sources' per-eval-type "_hetero" branches used."""
    p = np.clip((base - lo) / (hi - lo), 0.0, 1.0)
    return 2.0 * np.sqrt(p * (1.0 - p))


def _equicorrelated_noise(
    rng: np.random.Generator, n: int, runs: int, k: int, group_vars: np.ndarray, corr: float,
) -> np.ndarray:
    """k correlated mean-zero Gaussian noise streams, shape (k, n, runs):
    Var(noise_j) = group_vars[j], Corr(noise_a, noise_b) = corr for all
    a != b. Generalizes the shared/indiv split (equal variance per group) to
    per-group variances via a shared standard-normal factor scaled by each
    group's own std."""
    shared_z = rng.normal(0.0, 1.0, size=(n, runs))
    out = np.empty((k, n, runs), dtype=float)
    c = max(corr, 0.0)
    for j in range(k):
        std_j = float(np.sqrt(max(group_vars[j], 0.0)))
        indiv_z = rng.normal(0.0, 1.0, size=(n, runs))
        out[j] = std_j * (np.sqrt(c) * shared_z + np.sqrt(max(1.0 - c, 0.0)) * indiv_z)
    return out


def _equicorrelated_std_normal(rng: np.random.Generator, n: int, k: int, base_corr: float) -> np.ndarray:
    """k equicorrelated standard-normal columns, shape (k, n, 1):
    Var=1, Corr(z_a, z_b) = base_corr for all a != b."""
    shared_z = rng.normal(0.0, 1.0, size=(n, 1))
    bc = max(base_corr, 0.0)
    out = np.empty((k, n, 1), dtype=float)
    for j in range(k):
        indiv_z = rng.normal(0.0, 1.0, size=(n, 1))
        out[j] = np.sqrt(bc) * shared_z + np.sqrt(max(1.0 - bc, 0.0)) * indiv_z
    return out


def sample_group_truth(
    shape: ShapeSpec, n: int, runs: int, k: int, icc: float | Sequence[float], rng: np.random.Generator,
    *, corr: float = 0.5, effects: np.ndarray | None = None,
    heteroscedastic: bool = False, base_corr: float = 1.0,
) -> np.ndarray:
    """Truth+within-item-noise model shared by build_single_sample_sources
    (k=1, optionally multi-run), build_pair_sources (k=2, optionally
    multi-run/partial-agreement), build_multiarm_sources (k>=2), and
    build_judge_bias_sources' repeated-measures truth (k=2,3,4). Returns an
    ndarray of shape (k, n, runs).

    icc: between-item variance / total variance (same meaning everywhere
    it's used in the harness). May be a single float (shared by all k
    groups -- the common case) or a length-k sequence of per-group values
    (asymmetric reliability, e.g. build_pair_sources' run_noise_fracs sweep
    with different f_a/f_b for the two groups).

    corr: correlation between any two of the k groups' within-item NOISE (0
    = fully independent runs; 1 = identical noise draw shared across all
    groups). k=2 at corr=0.5, base_corr=1.0, scalar icc reproduces
    build_pair_sources' original fixed shared/individual noise split
    exactly (see _legacy_build_pair_sources above).

    base_corr: correlation of the GAUSSIAN-COPULA LATENT VARIABLE underlying
    any two of the k groups' BASE/item-level values (1.0 = fully shared base
    -- every group sees the literal same item, the harness default
    everywhere except build_pair_sources' run_noise_fracs multi-run mode;
    <1.0 = partial agreement, a Gaussian-copula "different items/judges,
    correlated quality" model, generalizing
    _legacy_build_pair_multirun_sources' cross_item_rho to general k). For
    binary/continuous "param" shapes and all "custom" shapes, base_corr is
    NOT the realized Pearson or Spearman correlation of the final base
    values -- it's the correlation BEFORE the (generally nonlinear)
    .ppf()/empirical-quantile mapping. The realized Spearman correlation
    follows the standard Gaussian-copula identity (6/pi)*arcsin(base_corr/2)
    for shapes with continuous marginals (verified numerically for
    cont-mixture); shapes with large point masses (zero/one-inflated,
    floor/ceiling-spiked) compress it further below even that, since tied
    values get averaged ranks (verified: cont-zero-inflated's 70% spike at 0
    realizes Spearman ~0.54 at base_corr=0.7, vs. the ~0.68 the tie-free
    identity would predict). likert/grades "param" shapes are the one
    exception: their base is additively decomposed in Gaussian space
    directly (no .ppf()), so base_corr matches the realized Pearson
    correlation of the underlying continuous latent exactly (verified:
    grades-mid, no boundary clipping, base_corr=0.3/0.7 -> realized
    0.2989/0.6985); the FINAL output can still differ slightly due to
    boundary clipping (any shape near its range edges) and, for likert
    specifically, rounding to the nearest integer score (verified: likert
    rounding alone pulls realized correlation down by ~0.03-0.04 at
    base_corr=0.3/0.7). At base_corr == 1.0 and scalar icc, this is a no-op
    vs. the
    pre-existing shared-base code path (kept as its own branch below to
    preserve exact-parity with everything that doesn't pass base_corr/
    per-group icc).

    k=1 has no inter-group correlation concept -- the single group gets the
    FULL within-item noise variance directly (same marginal variance as any
    one of the k>=2 groups at the same icc), reproducing
    build_single_sample_sources' simple "param" shapes exactly when icc=1.0
    (no separate noise layer at all); icc<1.0 at k=1, runs>1 is the
    run_noise_frac multi-run case (f_run = 1 - icc).

    shape.kind == "custom" supports k >= 1, including icc<1.0, runs>1
    (multi-run): the bare custom_sampler draw is treated as the per-item
    "base", with the same additive-Gaussian within-item noise model as
    "param" shapes, using an MC-estimated (not closed-form) base variance
    (_custom_shape_var). At k >= 2 and base_corr < 1.0 (partial agreement),
    custom shapes use an empirical-quantile copula in place of a closed-form
    .ppf() (see _custom_shape_inverse_cdf) -- this preserves rank
    correlation between groups exactly equal to base_corr, but Pearson
    correlation only approximately (the empirical quantile function is a
    nonlinear map for non-uniform shapes).
    """
    if effects is None:
        effects = np.zeros(k)
    et = shape.eval_type
    icc_arr = np.full(k, float(icc)) if np.ndim(icc) == 0 else np.asarray(icc, dtype=float)
    if icc_arr.shape != (k,):
        raise ValueError(f"icc must be a scalar or a length-k sequence, got shape {icc_arr.shape} for k={k}")
    icc_scalar = bool(np.ndim(icc) == 0)

    if shape.kind == "custom":
        if k == 1:
            base = shape.custom_sampler(rng, n)[:, None]  # (n, 1)
            lo, hi, round_to_int = _CUSTOM_SHAPE_RANGE[et]
            ic = float(icc_arr[0])
            if ic >= 1.0 - 1e-12:
                noise = np.zeros((n, runs))
            else:
                noise_var = _custom_shape_var(shape) * (1.0 / max(ic, 1e-9) - 1.0)
                sigma = float(np.sqrt(max(noise_var, 0.0)))
                if heteroscedastic:
                    noise = rng.normal(0.0, 1.0, size=(n, runs)) * (sigma * _hetero_noise_scale(base, lo, hi))
                else:
                    noise = rng.normal(0.0, sigma, size=(n, runs))
            vals = base + effects[0] + noise
            out = np.empty((1, n, runs), dtype=float)
            out[0] = np.rint(np.clip(vals, lo, hi)) if round_to_int else np.clip(vals, lo, hi)
            return out

        lo, hi, round_to_int = _CUSTOM_SHAPE_RANGE[et]
        if base_corr >= 1.0 - 1e-12:
            base = np.broadcast_to(shape.custom_sampler(rng, n)[None, :, None], (k, n, 1)).copy()
        else:
            # Partial-agreement path: equicorrelated Gaussian copula on
            # quantiles, mapped through the shape's own empirical inverse-CDF
            # (no closed-form .ppf() to invert through, unlike binary/
            # continuous "param" shapes -- see _custom_shape_inverse_cdf).
            z = _equicorrelated_std_normal(rng, n, k, base_corr)
            q = np.clip(norm.cdf(z), 1e-9, 1.0 - 1e-9)
            base = _custom_shape_inverse_cdf(shape, q)
        group_noise_vars = np.array([
            _custom_shape_var(shape) * (1.0 / max(ic, 1e-9) - 1.0) if ic < 1.0 - 1e-12 else 0.0
            for ic in icc_arr
        ])
        # falls through to the shared noise-injection tail below (same code
        # path "param" continuous/likert/grades shapes use once base/lo/hi/
        # round_to_int/group_noise_vars are computed).

    elif et == "binary":
        base_p = shape.params
        if base_corr >= 1.0 - 1e-12 and icc_scalar:
            conc = _binary_conc_from_icc(float(icc))
            base = rng.beta(base_p * conc, (1.0 - base_p) * conc, size=(n, 1))
            out = np.empty((k, n, runs), dtype=float)
            for j in range(k):
                p = np.clip(base + effects[j], 0.0, 1.0)
                out[j] = rng.binomial(1, p, size=(n, runs)).astype(float)
            return out
        # Partial-agreement / per-group-icc path: equicorrelated Gaussian
        # copula on quantiles, mapped through each group's own Beta(p, icc_j).
        z = _equicorrelated_std_normal(rng, n, k, base_corr)
        out = np.empty((k, n, runs), dtype=float)
        for j in range(k):
            conc_j = _binary_conc_from_icc(float(icc_arr[j]))
            q_j = np.clip(norm.cdf(z[j]), 1e-9, 1.0 - 1e-9)
            p_j = np.clip(stats.beta.ppf(q_j, base_p * conc_j, (1.0 - base_p) * conc_j) + effects[j], 0.0, 1.0)
            out[j] = rng.binomial(1, p_j, size=(n, runs)).astype(float)
        return out

    elif et == "continuous":
        a_beta, b_beta = shape.params
        lo, hi, round_to_int = 0.0, 1.0, False
        if base_corr >= 1.0 - 1e-12:
            base = np.broadcast_to(rng.beta(a_beta, b_beta, size=(n, 1)), (k, n, 1)).copy()
        else:
            z = _equicorrelated_std_normal(rng, n, k, base_corr)
            q = np.clip(norm.cdf(z), 1e-9, 1.0 - 1e-9)
            base = stats.beta.ppf(q, a_beta, b_beta)
    else:
        mu_lat, total_std = shape.params
        lo, hi = (1.0, 5.0) if et == "likert" else (0.0, 100.0)
        round_to_int = et == "likert"
        if base_corr >= 1.0 - 1e-12 and icc_scalar:
            base_std = float(np.sqrt(float(icc))) * total_std
            base = np.broadcast_to(rng.normal(mu_lat, base_std, size=(n, 1)), (k, n, 1)).copy()
        else:
            z = _equicorrelated_std_normal(rng, n, k, base_corr)
            base = mu_lat + total_std * np.sqrt(np.clip(icc_arr, 0.0, 1.0))[:, None, None] * z

    def _finish(vals: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(vals, lo, hi)) if round_to_int else np.clip(vals, lo, hi)

    if shape.kind == "param":
        group_noise_vars = np.array([_group_noise_var(et, shape.params, ic) for ic in icc_arr])
    # else: shape.kind == "custom" at k >= 2 already computed group_noise_vars above.
    out = np.empty((k, n, runs), dtype=float)
    if k == 1:
        if heteroscedastic and group_noise_vars[0] > 0.0:
            scale = _hetero_noise_scale(base[0], lo, hi)
            noise = rng.normal(0.0, 1.0, size=(n, runs)) * (float(np.sqrt(group_noise_vars[0])) * scale)
        else:
            noise = rng.normal(0.0, float(np.sqrt(max(group_noise_vars[0], 0.0))), size=(n, runs))
        out[0] = _finish(base[0] + effects[0] + noise)
        return out

    if heteroscedastic:
        # Heteroscedastic noise can't share the equicorrelated-noise helper's
        # single homoscedastic std per group; fall back to corr-mixed draws
        # with a per-item, per-group scale (matches _legacy_build_pair_multirun_sources'
        # "_hetero" branches, generalized to k groups / per-group icc). One
        # shared_z draw is reused across all k groups so corr means the same
        # thing here as in the homoscedastic _equicorrelated_noise path.
        shared_z = rng.normal(0.0, 1.0, size=(n, runs))
        c = max(corr, 0.0)
        for j in range(k):
            scale_j = _hetero_noise_scale(base[j], lo, hi)
            indiv_z = rng.normal(0.0, 1.0, size=(n, runs))
            std_j = float(np.sqrt(group_noise_vars[j])) * scale_j
            noise_j = std_j * (np.sqrt(c) * shared_z + np.sqrt(max(1.0 - c, 0.0)) * indiv_z)
            out[j] = _finish(base[j] + effects[j] + noise_j)
        return out

    noise = _equicorrelated_noise(rng, n, runs, k, group_noise_vars, corr)
    for j in range(k):
        out[j] = _finish(base[j] + effects[j] + noise[j])
    return out


def build_single_sample_sources(
    suite: str = "standard", *, run_noise_fracs: list[float] | None = None, heteroscedastic: bool = False,
) -> list[CISource]:
    """Canonical synthetic single-sample CISources, drawn from the shared
    shape catalog above via sample_group_truth(k=1).

    run_noise_fracs : sequence of float, optional
        If given, switches to multi-run mode (for cases/ci_single.py's
        ``--nested-mode``): for every shape, builds one CISource per
        ``f_run`` in this list, with ``generate_runs(rng, n, runs) -> (n,
        runs)`` set (icc = 1 - f_run fed through sample_group_truth(k=1,
        runs>1) -- the within-item run-noise variance grows as f_run grows).
        If ``None`` (the default), every shape gets exactly one flat,
        single-run CISource (generate_runs=None), matching this function's
        pre-multi-run behavior exactly.
    heteroscedastic : bool
        Only meaningful when run_noise_fracs is given: scales each item's
        run-noise std down near the eval type's score-range boundaries
        (where item-level variance is naturally squeezed) instead of using
        one fixed std for every item -- see sample_group_truth.
    """
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")

    sources: list[CISource] = []
    for eval_type, catalog in SHAPES_BY_EVAL_TYPE.items():
        for shape in _tier_shapes(catalog, suite):
            if run_noise_fracs is None:
                def _gen(rng: np.random.Generator, n: int, _shape: ShapeSpec = shape) -> np.ndarray:
                    return sample_group_truth(_shape, n, 1, 1, 1.0, rng)[0, :, 0]

                true_mean = _estimate_true_mean_mc(_gen)
                sources.append(CISource(label=shape.label, eval_type=eval_type, generate=_gen, true_mean=true_mean))
                continue

            for f in run_noise_fracs:
                icc_ = 1.0 - f

                def _gen_runs(
                    rng: np.random.Generator, n: int, runs: int,
                    _shape: ShapeSpec = shape, _icc: float = icc_,
                ) -> np.ndarray:
                    return sample_group_truth(_shape, n, runs, 1, _icc, rng, heteroscedastic=heteroscedastic)[0]

                true_mean = _estimate_true_mean_mc_runs(_gen_runs)
                sources.append(CISource(
                    label=f"{shape.label}|f={f:.2f}", eval_type=eval_type,
                    generate=lambda rng, n, _g=_gen_runs: _g(rng, n, 1)[:, 0],
                    generate_runs=_gen_runs, true_mean=true_mean, run_noise_frac=f,
                ))

    return sources


# Explicit stress-test binary regimes with highly one-sided discordance --
# not part of the shape catalog (no icc/shape decomposition; these are
# literally pre-specified joint discordance probabilities). Shared by
# build_pair_sources' flat and multi-run (run_noise_fracs) paths.
_ASYM_BINARY_SPECS: list[tuple[str, float, float, float]] = [
    ("binary-onesided-neg-extreme", 0.001, 0.384, 0.000),
    ("binary-onesided-pos-extreme", 0.384, 0.001, 0.000),
    ("binary-onesided-neg-strong", 0.020, 0.300, 0.050),
    ("binary-onesided-pos-strong", 0.300, 0.020, 0.050),
    ("binary-onesided-neg-ultra", 0.000, 0.520, 0.000),
    ("binary-onesided-pos-ultra", 0.520, 0.000, 0.000),
    ("binary-onesided-neg-sparse", 0.001, 0.090, 0.030),
    ("binary-onesided-pos-sparse", 0.090, 0.001, 0.030),
    ("binary-onesided-neg-near-ceil", 0.000, 0.080, 0.900),
    ("binary-onesided-pos-near-floor", 0.080, 0.000, 0.020),
    ("binary-onesided-neg-moderate", 0.050, 0.220, 0.150),
    ("binary-onesided-pos-moderate", 0.220, 0.050, 0.150),
]


def _build_pair_sources_multirun(
    suite: str, d_list: list[float], run_noise_fracs: list[float], *,
    heteroscedastic: bool, pairwise_noise_grid: bool, pairwise_noise_grid_max: int | None,
    pairwise_noise_grid_seed: int, cross_item_rho: float,
) -> list[CIPairSource]:
    """build_pair_sources' run_noise_fracs branch: the same (eval_type,
    shape, d) sweep as the flat path, but over (f_a, f_b) run-noise-frac
    pairs instead of icc_values, routed through sample_group_truth(k=2,
    base_corr=cross_item_rho, icc=[1-f_a, 1-f_b])."""
    if pairwise_noise_grid:
        noise_pairs = [(float(fa), float(fb)) for fa in run_noise_fracs for fb in run_noise_fracs]
    else:
        noise_pairs = [(float(f), float(f)) for f in run_noise_fracs]
    if pairwise_noise_grid_max is not None and pairwise_noise_grid_max > 0 and len(noise_pairs) > pairwise_noise_grid_max:
        rng_grid = np.random.default_rng(pairwise_noise_grid_seed)
        keep_idx = np.sort(rng_grid.choice(len(noise_pairs), size=pairwise_noise_grid_max, replace=False))
        noise_pairs = [noise_pairs[int(i)] for i in keep_idx]

    sources: list[CIPairSource] = []
    for eval_type, catalog in SHAPES_BY_EVAL_TYPE.items():
        # Custom shapes are included too -- sample_group_truth's Phase 2
        # empirical-quantile copula supports base_corr < 1.0 for them now.
        shapes_here = _tier_shapes(catalog, suite)
        for f_a, f_b in noise_pairs:
            icc_a, icc_b = 1.0 - f_a, 1.0 - f_b
            icc_eff = 0.5 * (icc_a + icc_b)
            for shape in shapes_here:
                total_std = group_total_std(shape, icc_eff)
                for d in d_list:
                    delta = d * total_std
                    is_null = d == 0.0
                    effect_tag = "null" if is_null else f"d={d:.2f}"
                    label = (
                        f"{shape.label}|fA={f_a:.2f}|fB={f_b:.2f}|{effect_tag}" if pairwise_noise_grid
                        else f"{shape.label}|f={f_a:.2f}|{effect_tag}"
                    )
                    shape_, icc_pair_, delta_ = shape, [icc_a, icc_b], delta

                    def _gen_pair_runs(
                        rng: np.random.Generator, n: int, runs: int,
                        _shape: ShapeSpec = shape_, _icc: list[float] = icc_pair_, _d: float = delta_,
                    ) -> tuple[np.ndarray, np.ndarray]:
                        # corr=0.0: run-noise is independent between A and B
                        # here (matches _legacy_build_pair_multirun_sources,
                        # which never correlated noise across groups -- only
                        # the base/item-level value is rho-correlated, via
                        # base_corr). The flat (non-multirun) path's corr=0.5
                        # default is a separate, unrelated axis.
                        out = sample_group_truth(
                            _shape, n, runs, 2, _icc, rng, effects=np.array([0.0, _d]),
                            heteroscedastic=heteroscedastic, base_corr=cross_item_rho, corr=0.0,
                        )
                        return out[0], out[1]

                    true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_pair_runs)
                    sources.append(CIPairSource(
                        label=label, eval_type=eval_type, generate_pair=_gen_pair_runs, true_diff=true_diff,
                        icc=icc_eff, cohens_d=d, is_null=is_null,
                        run_noise_frac=0.5 * (f_a + f_b), run_noise_frac_a=f_a, run_noise_frac_b=f_b,
                    ))

    # Asym binary stress test, generalized with a redraw-frac run-noise
    # mechanism: each run independently redraws the joint (A, B) state with
    # probability f_eff instead of keeping the item's fixed joint state.
    if suite in ("expanded", "extreme"):
        for shape_label, p10, p01, p11 in _ASYM_BINARY_SPECS:
            p00 = 1.0 - (p11 + p10 + p01)
            if p00 <= 0.0:
                raise ValueError(f"Invalid asymmetric binary scenario {shape_label}: probabilities sum to >= 1.0")
            probs = np.array([p11, p10, p01, p00], dtype=float)
            true_diff = float(p10 - p01)

            for f_a, f_b in noise_pairs:
                f_eff = 0.5 * (f_a + f_b)
                label = (
                    f"{shape_label}|p10={p10:.3f}|p01={p01:.3f}|p11={p11:.3f}|p00={p00:.3f}|fA={f_a:.2f}|fB={f_b:.2f}"
                    if pairwise_noise_grid
                    else f"{shape_label}|p10={p10:.3f}|p01={p01:.3f}|p11={p11:.3f}|p00={p00:.3f}|f={f_eff:.2f}"
                )
                probs_, f_ = probs, f_eff

                def _gen_binary_asym_runs(
                    rng: np.random.Generator, n: int, runs: int, _probs: np.ndarray = probs_, _f: float = f_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    z_item = rng.choice(4, size=(n, 1), p=_probs)
                    z_run = rng.choice(4, size=(n, runs), p=_probs)
                    redraw = rng.random((n, runs)) < _f
                    z = np.where(redraw, z_run, z_item)
                    a = np.isin(z, (0, 1)).astype(float)
                    b = np.isin(z, (0, 2)).astype(float)
                    return a, b

                sources.append(CIPairSource(
                    label=label, eval_type="binary", generate_pair=_gen_binary_asym_runs, true_diff=true_diff,
                    icc=1.0 - f_eff, cohens_d=0.0, is_null=False,
                    run_noise_frac=f_eff, run_noise_frac_a=f_a, run_noise_frac_b=f_b,
                ))

    return sources


def build_pair_sources(
    suite: str = "standard",
    icc_values: list[float] | tuple[float, ...] = (0.10, 0.25, 0.40),
    cohens_d_values: list[float] | tuple[float, ...] = (0.3,),
    include_null: bool = False,
    *,
    run_noise_fracs: list[float] | None = None,
    heteroscedastic: bool = False,
    pairwise_noise_grid: bool = False,
    pairwise_noise_grid_max: int | None = None,
    pairwise_noise_grid_seed: int = 42,
    cross_item_rho: float = 0.7,
) -> list[CIPairSource]:
    """Return canonical synthetic paired-difference CIPairSources, parameterised
    by ICC and Cohen's d -- a thin (icc, shape, d) sweep over
    sample_group_truth(k=2) using the full shape catalog above, "custom"
    shapes (mixtures, inflated, heavy-tail) included. The flat path here
    uses base_corr=1.0 (the default); the run_noise_fracs multi-run path
    (_build_pair_sources_multirun) uses base_corr=cross_item_rho < 1.0,
    which custom shapes now support via an empirical-quantile copula (see
    ShapeSpec/sample_group_truth docs).

    ICC = between-input variance / total variance; Cohen's d = delta /
    total_std (group_total_std(shape, icc)).

    Parameters
    ----------
    suite : str
        'standard', 'expanded', or 'extreme'.
    icc_values : sequence of float
        ICC values to sweep. Each value generates a separate scenario batch.
        Ignored when run_noise_fracs is given.
    cohens_d_values : sequence of float
        Non-null standardised effect sizes. A null (d=0) variant is
        automatically prepended when include_null=True.
    include_null : bool
        If True, prepend d=0 scenarios for every (eval_type, shape, icc)
        combination, flagged is_null=True (used to measure Type I error).
    run_noise_fracs : sequence of float, optional
        If given, switches to multi-run mode (for cases/ci_paired.py's
        ``--nested-mode``): icc_values is ignored, and instead every (f_a,
        f_b) pair from this list (f_a == f_b unless pairwise_noise_grid)
        sweeps icc_a = 1 - f_a, icc_b = 1 - f_b through
        sample_group_truth(k=2, base_corr=cross_item_rho). If ``None`` (the
        default), every scenario is flat/single-run with base_corr=1.0
        (fully shared base), matching this function's pre-multi-run
        behavior exactly.
    heteroscedastic : bool
        Only meaningful when run_noise_fracs is given -- see sample_group_truth.
    pairwise_noise_grid : bool
        Only meaningful when run_noise_fracs is given: sweep the full (f_a,
        f_b) cross product instead of pairing each f with itself.
    pairwise_noise_grid_max : int, optional
        Only meaningful with pairwise_noise_grid: randomly subsample the
        (f_a, f_b) grid down to this many pairs (deterministic given
        pairwise_noise_grid_seed) if the full grid would be larger.
    cross_item_rho : float
        Only meaningful when run_noise_fracs is given: the base_corr fed to
        sample_group_truth -- correlation between A's and B's item-level
        truth (1.0 = identical item; <1.0 = "different items/judges,
        correlated quality").
    """
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")

    d_list = list(cohens_d_values)
    if include_null:
        d_list = [0.0] + [d for d in d_list if d > 0.0]

    if run_noise_fracs is not None:
        return _build_pair_sources_multirun(
            suite, d_list, run_noise_fracs, heteroscedastic=heteroscedastic,
            pairwise_noise_grid=pairwise_noise_grid, pairwise_noise_grid_max=pairwise_noise_grid_max,
            pairwise_noise_grid_seed=pairwise_noise_grid_seed, cross_item_rho=cross_item_rho,
        )

    sources: list[CIPairSource] = []
    icc_list = list(icc_values)

    for eval_type, catalog in SHAPES_BY_EVAL_TYPE.items():
        # Custom shapes (mixtures, inflated, heavy-tail) get swept here too --
        # sample_group_truth supports them at k=2/base_corr=1.0 (the default
        # here). This is the riskiest, closest-to-real-world part of the
        # catalog, so it shouldn't be exempt from the pairwise power/Type-I
        # checks the way it used to be (param-only).
        shapes_here = _tier_shapes(catalog, suite)
        for icc in icc_list:
            for shape in shapes_here:
                total_std = group_total_std(shape, icc)
                for d in d_list:
                    delta = d * total_std
                    is_null = d == 0.0
                    effect_tag = "null" if is_null else f"d={d:.2f}"
                    label = f"{shape.label}|icc={icc:.2f}|{effect_tag}"
                    shape_, icc_, delta_ = shape, icc, delta

                    def _gen_pair(
                        rng: np.random.Generator, n: int, runs: int,
                        _shape: ShapeSpec = shape_, _icc: float = icc_, _d: float = delta_,
                    ) -> tuple[np.ndarray, np.ndarray]:
                        out = sample_group_truth(_shape, n, runs, 2, _icc, rng, effects=np.array([0.0, _d]))
                        return out[0], out[1]

                    true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_pair)
                    sources.append(CIPairSource(
                        label=label, eval_type=eval_type, generate_pair=_gen_pair, true_diff=true_diff,
                        icc=icc, cohens_d=d, is_null=is_null,
                    ))

    # Explicit stress-test binary regimes with highly one-sided discordance --
    # not part of the shape catalog (no icc/shape decomposition; these are
    # literally pre-specified joint discordance probabilities), kept as-is.
    if suite in ("expanded", "extreme"):
        for shape_label, p10, p01, p11 in _ASYM_BINARY_SPECS:
            p00 = 1.0 - (p11 + p10 + p01)
            if p00 <= 0.0:
                raise ValueError(f"Invalid asymmetric binary scenario {shape_label}: probabilities sum to >= 1.0")

            probs = np.array([p11, p10, p01, p00], dtype=float)
            true_diff = float(p10 - p01)
            label = f"{shape_label}|p10={p10:.3f}|p01={p01:.3f}|p11={p11:.3f}|p00={p00:.3f}"

            def _gen_binary_asym(rng: np.random.Generator, n: int, runs: int, _probs: np.ndarray = probs) -> tuple[np.ndarray, np.ndarray]:
                z = rng.choice(4, size=(n, runs), p=_probs)
                a = np.isin(z, (0, 1)).astype(float)
                b = np.isin(z, (0, 2)).astype(float)
                return a, b

            sources.append(CIPairSource(
                label=label, eval_type="binary", generate_pair=_gen_binary_asym, true_diff=true_diff,
                icc=0.0, cohens_d=0.0, is_null=False,
            ))

    return sources


def _estimate_true_mean_mc_runs(
    generate_runs: Callable[[np.random.Generator, int, int], np.ndarray],
    *,
    seed: int = 0,
    n_mc: int = 500_000,
) -> float:
    rng = np.random.default_rng(seed)
    return float(np.mean(generate_runs(rng, n_mc, 1)))


# ---------------------------------------------------------------------------
# Multi-arm sources (for cases/pvalues.py's multi-arm multiplicity benchmark)
# ---------------------------------------------------------------------------
# Sweeps the SAME shape catalog (BINARY_SHAPES/CONTINUOUS_SHAPES/
# LIKERT_SHAPES/GRADES_SHAPES) build_pair_sources uses, generalized from k=2
# to k arms via sample_group_truth -- so multi-arm benchmarking gets the
# same shape-robustness as the pairwise comparison, not a separate, narrower
# bespoke catalog. "custom" shapes (mixtures, inflated, heavy-tail) are
# included too, at base_corr=1.0 (the default here). Arm 0 carries the
# alternative-hypothesis shift (cohens_d * group_total_std); arms 1..k-1
# stay at the shared baseline.


def build_multiarm_sources(
    *, suite: str = "standard", icc: float = 0.20, cohens_d: float = 0.3, eval_types: list[str] | None = None,
) -> list[MultiArmSource]:
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")
    eval_types = list(eval_types) if eval_types is not None else list(EVAL_TYPES)

    def _make_generator(shape: ShapeSpec):
        def _gen(rng: np.random.Generator, n: int, runs: int, k: int, delta: float) -> np.ndarray:
            effects = np.zeros(k)
            effects[0] = delta
            return sample_group_truth(shape, n, runs, k, icc, rng, effects=effects)
        return _gen

    sources: list[MultiArmSource] = []
    for eval_type in eval_types:
        # Custom shapes (mixtures, inflated, heavy-tail) are included too --
        # sample_group_truth supports them at k>=2/base_corr=1.0 (the
        # default here), so the multi-arm FWER/power sweep no longer skips
        # the catalog's most realistic-pathology shapes.
        for shape in _tier_shapes(SHAPES_BY_EVAL_TYPE[eval_type], suite):
            sources.append(MultiArmSource(
                label=shape.label, eval_type=eval_type, generate_scores=_make_generator(shape),
                alt_delta=cohens_d * group_total_std(shape, icc),
            ))
    return sources



# ---------------------------------------------------------------------------
# Judge-bias sources (for cases/pvalues.py's PPI-correction calibration mode)
# ---------------------------------------------------------------------------
# Ported from sim_type_i_calibration.py's Scenario/_build_scenarios/_run_one
# data-generation helpers. Sweeps JUDGE-MEASUREMENT-ERROR parameters (bias,
# scale-calibration slope, label sparsity/MNAR-ness, repeated-measures error
# correlation) layered on top of ONE representative shape per eval type,
# drawn from the SAME shape catalog (BINARY_SHAPES/CONTINUOUS_SHAPES/
# LIKERT_SHAPES/GRADES_SHAPES) build_pair_sources/build_multiarm_sources
# use, via sample_group_truth -- so judge bias/noise is an orthogonal sweep
# on the same truth-generating process, not its own separate distribution
# family. `eval_type` excludes "binary": ttest/MWU/ANOVA aren't designed
# for 0/1 outcomes, so PPI mode only sweeps continuous/likert/grades.

_PPI_REPRESENTATIVE_SHAPE_LABEL: dict[str, str] = {
    "continuous": "cont-right-skew", "likert": "likert-mid", "grades": "grades-mid",
}


def _ppi_shape(eval_type: str, shape_label: str | None = None) -> ShapeSpec:
    label = shape_label if shape_label is not None else _PPI_REPRESENTATIVE_SHAPE_LABEL[eval_type]
    return next(s for s in SHAPES_BY_EVAL_TYPE[eval_type] if s.label == label)


def _ppi_shape_anchor(shape: ShapeSpec) -> float:
    """Mean of a ShapeSpec's truth distribution -- used both as the
    judge-bias model's mu_null and as the slope-distortion anchor point."""
    if shape.kind == "custom":
        return _custom_shape_mean(shape)
    if shape.eval_type == "continuous":
        a, b = shape.params
        return a / (a + b)
    return float(shape.params[0])  # likert/grades: (mu, total_std)


_JB_MIN_LAB = 15
JUDGE_BIAS_LMM_RUNS_R = 3
JUDGE_BIAS_LMM_FACTORIAL_FACTORS = pd.DataFrame({
    "f1": ["a", "a", "b", "b"],
    "f2": ["x", "y", "x", "y"],
})


def _jb_biases(sc: JudgeBiasSource) -> tuple[float, float, float]:
    if sc.bias_type == "none":
        return 0.0, 0.0, 0.0
    if sc.bias_type == "constant":
        return sc.bias_const, sc.bias_const, sc.bias_const
    if sc.bias_type == "differential":
        return sc.bias_delta, 0.0, 0.0
    raise ValueError(f"Unknown bias_type: {sc.bias_type!r}")


def _jb_biases_4(sc: JudgeBiasSource) -> tuple[float, float, float, float]:
    if sc.bias_type == "none":
        return 0.0, 0.0, 0.0, 0.0
    if sc.bias_type == "constant":
        return (sc.bias_const,) * 4
    if sc.bias_type == "differential":
        return sc.bias_delta, 0.0, 0.0, 0.0
    raise ValueError(f"Unknown bias_type: {sc.bias_type!r}")


def _jb_judge_params_3(sc: JudgeBiasSource) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    b1, b2, b3 = _jb_biases(sc)
    biases = (b1 + sc.bias_extra_a, b2 + sc.bias_extra_b, b3 + sc.bias_extra_c)
    slopes = (sc.slope_a, sc.slope_b, sc.slope_c)
    return biases, slopes


def _jb_judge_params_4(sc: JudgeBiasSource) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    b1, b2, b3, b4 = _jb_biases_4(sc)
    biases = (b1 + sc.bias_extra_a, b2 + sc.bias_extra_b, b3 + sc.bias_extra_c, b4 + sc.bias_extra_d)
    slopes = (sc.slope_a, sc.slope_b, sc.slope_c, sc.slope_d)
    return biases, slopes


def _contaminated_noise_stds(noise_sd: float, contam_frac: float, contam_scale: float) -> tuple[float, float]:
    """Solve for (sigma_normal, sigma_catastrophic) of a two-component,
    zero-mean Gaussian variance-mixture -- weight (1 - contam_frac) at
    sigma_normal, weight contam_frac at sigma_catastrophic = contam_scale *
    sigma_normal -- whose TOTAL variance equals noise_sd**2 exactly. This is
    what keeps noise_family="contaminated" meaning the same noise_sd/icc
    everywhere noise_family="gaussian" does: it only redistributes WHERE
    that fixed total variance comes from (mostly a small width, occasionally
    a much larger one), modeling "judge is mostly right, occasionally
    catastrophically wrong" instead of symmetric/uniform measurement error.
    """
    noise_var = noise_sd ** 2
    denom = (1.0 - contam_frac) + contam_frac * contam_scale ** 2
    sigma_normal = float(np.sqrt(noise_var / max(denom, 1e-12)))
    sigma_catastrophic = contam_scale * sigma_normal
    return sigma_normal, sigma_catastrophic


def _jb_llm(
    truth: np.ndarray, bias: float, noise_sd: float, rng: np.random.Generator,
    slope: float = 1.0, anchor: float = 0.0,
    noise_family: str = "gaussian", contam_frac: float = 0.10, contam_scale: float = 4.0,
) -> np.ndarray:
    pred = anchor + slope * (truth - anchor) + bias
    if noise_sd == 0.0:
        return pred
    if noise_family == "gaussian":
        return pred + rng.normal(0.0, noise_sd, len(truth))
    if noise_family == "contaminated":
        sigma_normal, sigma_catastrophic = _contaminated_noise_stds(noise_sd, contam_frac, contam_scale)
        is_catastrophic = rng.random(len(truth)) < contam_frac
        sd_per_item = np.where(is_catastrophic, sigma_catastrophic, sigma_normal)
        return pred + rng.normal(0.0, 1.0, len(truth)) * sd_per_item
    raise ValueError(f"Unknown noise_family: {noise_family!r}")


def _jb_llm_repeated(
    truths: list[np.ndarray], biases: list[float], noise_sds: list[float], slopes: list[float],
    rng: np.random.Generator, anchor: float, corr: float,
    noise_family: str = "gaussian", contam_frac: float = 0.10, contam_scale: float = 4.0,
) -> list[np.ndarray]:
    if len(truths) == 0:
        return []
    n = len(truths[0])
    if corr <= 0.0:
        return [
            _jb_llm(tr, b, s, rng, slope=m, anchor=anchor, noise_family=noise_family,
                    contam_frac=contam_frac, contam_scale=contam_scale)
            for tr, b, s, m in zip(truths, biases, noise_sds, slopes)
        ]

    corr_clamped = min(max(corr, 0.0), 0.999)
    w_shared = float(np.sqrt(corr_clamped))
    w_ind = float(np.sqrt(1.0 - corr_clamped))

    if noise_family == "gaussian":
        shared = rng.normal(0.0, 1.0, n)
        out: list[np.ndarray] = []
        for tr, b, sd, m in zip(truths, biases, noise_sds, slopes):
            pred = anchor + m * (tr - anchor) + b
            if sd == 0.0:
                out.append(pred)
                continue
            err = sd * (w_shared * shared + w_ind * rng.normal(0.0, 1.0, n))
            out.append(pred + err)
        return out

    if noise_family == "contaminated":
        # shared_regime drives a CORRELATED "is this a catastrophic-error
        # item" event across groups (via the same copula-style threshold
        # trick base_corr uses for custom shapes: norm.cdf(u) < contam_frac
        # gives marginal P(catastrophic)=contam_frac with cross-group regime
        # correlation = corr) -- representing e.g. "the judge has a bad day
        # on this specific item" being a shared, not independent, event
        # across paired/repeated conditions. The ERROR MAGNITUDE within a
        # regime is still drawn independently per group/condition.
        shared_regime = rng.normal(0.0, 1.0, n)
        out = []
        for tr, b, sd, m in zip(truths, biases, noise_sds, slopes):
            pred = anchor + m * (tr - anchor) + b
            if sd == 0.0:
                out.append(pred)
                continue
            sigma_normal, sigma_catastrophic = _contaminated_noise_stds(sd, contam_frac, contam_scale)
            u = w_shared * shared_regime + w_ind * rng.normal(0.0, 1.0, n)
            is_catastrophic = norm.cdf(u) < contam_frac
            sd_per_item = np.where(is_catastrophic, sigma_catastrophic, sigma_normal)
            err = rng.normal(0.0, 1.0, n) * sd_per_item
            out.append(pred + err)
        return out

    raise ValueError(f"Unknown noise_family: {noise_family!r}")


def _jb_label_indices(
    signal: np.ndarray, n_lab: int, rng: np.random.Generator,
    mnar: bool, mnar_strength: float, mnar_mode: str,
) -> np.ndarray:
    n = len(signal)
    if (not mnar) or mnar_strength <= 0.0:
        return rng.choice(n, n_lab, replace=False)

    scale = float(np.std(signal, ddof=0))
    if scale <= 1e-12:
        return rng.choice(n, n_lab, replace=False)
    z = (signal - float(np.mean(signal))) / scale
    if mnar_mode == "high":
        score = z
    elif mnar_mode == "extreme":
        score = np.abs(z)
    else:
        raise ValueError(f"Unknown mnar_mode: {mnar_mode!r}")

    logits = mnar_strength * score
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    probs = weights / float(np.sum(weights))
    return rng.choice(n, n_lab, replace=False, p=probs)


def _jb_labels_independent(
    truth: np.ndarray, frac: float, rng: np.random.Generator,
    mnar: bool = False, mnar_strength: float = 1.0, mnar_mode: str = "high",
) -> np.ndarray:
    n = len(truth)
    if n < _JB_MIN_LAB:
        raise ValueError(f"Need n >= {_JB_MIN_LAB} to enforce n_lab >= {_JB_MIN_LAB}; got n={n}")
    n_lab = min(n, max(_JB_MIN_LAB, int(round(n * frac))))
    lab = np.full(n, np.nan)
    idx = _jb_label_indices(truth, n_lab, rng, mnar, mnar_strength, mnar_mode)
    lab[idx] = truth[idx]
    return lab


def _jb_labels_shared(
    truths: list[np.ndarray], frac: float, rng: np.random.Generator,
    mnar: bool = False, mnar_strength: float = 1.0, mnar_mode: str = "high",
) -> list[np.ndarray]:
    n = len(truths[0])
    if n < _JB_MIN_LAB:
        raise ValueError(f"Need n >= {_JB_MIN_LAB} to enforce n_lab >= {_JB_MIN_LAB}; got n={n}")
    n_lab = min(n, max(_JB_MIN_LAB, int(round(n * frac))))
    signal = np.mean(np.column_stack(truths), axis=1)
    idx = _jb_label_indices(signal, n_lab, rng, mnar, mnar_strength, mnar_mode)
    labs = []
    for truth in truths:
        lab = np.full(n, np.nan)
        lab[idx] = truth[idx]
        labs.append(lab)
    return labs


def build_judge_bias_sources() -> list[JudgeBiasSource]:
    """Curated, factor-at-a-time judge-bias scenario sweep, ported from
    sim_type_i_calibration.py's _build_scenarios(). Each scenario varies
    exactly one factor from a fixed baseline (eval type, sample size, group
    balance, label fraction, label mechanism, LLM noise, bias type,
    scale/slope calibration, group-specific calibration, repeated-measures
    error correlation, heteroskedastic noise), plus a few multi-factor
    interaction stress tests."""
    B: dict = dict(
        eval_type="continuous", icc=0.20, n=100, n2=None, n3=None,
        label_frac=0.20, llm_noise=0.20, llm_noise2=None, llm_noise3=None,
        bias_type="differential", bias_delta=0.30, bias_const=0.40,
        bias_extra_a=0.0, bias_extra_b=0.0, bias_extra_c=0.0, bias_extra_d=0.0,
        slope_a=1.0, slope_b=1.0, slope_c=1.0, slope_d=1.0,
        label_mnar=False, mnar_strength=1.0, mnar_mode="high",
        repeated_corr=0.0,
    )

    S: list[JudgeBiasSource] = []

    def sc(name, tag, **kw):
        return JudgeBiasSource(name=name, tag=tag, **{**B, **kw})

    for eval_type in ["continuous", "likert", "grades"]:
        S.append(sc(f"eval_type.{eval_type}", "eval_type", eval_type=eval_type))

    # Truth-distribution shape factor: does PPI correction still fix Type-I
    # inflation when the underlying truth is a pathological ("custom")
    # shape -- refusal-like zero-inflation, bimodal pass/fail, multi-modal
    # scoring clusters -- instead of the smooth Beta/Normal representative
    # shape every other scenario in this sweep uses? See ShapeSpec/
    # _ppi_shape/JudgeBiasSource.shape_label.
    S.append(sc("shape.cont-zero-inflated", "shape", eval_type="continuous", shape_label="cont-zero-inflated"))
    S.append(sc("shape.likert-bimodal", "shape", eval_type="likert", shape_label="likert-bimodal"))
    S.append(sc("shape.grades-mixture", "shape", eval_type="grades", shape_label="grades-mixture"))

    for n in [60, 100, 200, 400]:
        S.append(sc(f"n={n}", "sample_size", n=n))

    S.append(sc("balance.1:1", "balance", n=100, n2=100, n3=100))
    S.append(sc("balance.2:1", "balance", n=67, n2=133, n3=100))
    S.append(sc("balance.4:1", "balance", n=40, n2=160, n3=100))

    for lab in [0.05, 0.10, 0.20, 0.40]:
        S.append(sc(f"lab.{lab:.0%}", "label_frac", label_frac=lab))

    S.append(sc("label.mcar", "label_mechanism", label_mnar=False, mnar_strength=0.0))
    S.append(sc("label.mnar-mild", "label_mechanism", label_mnar=True, mnar_strength=0.8, mnar_mode="high"))
    S.append(sc("label.mnar-strong", "label_mechanism", label_mnar=True, mnar_strength=1.6, mnar_mode="high"))

    for noise in [0.0, 0.10, 0.35, 0.70]:
        S.append(sc(f"noise.{noise}", "llm_noise", llm_noise=noise))

    for bt in ["none", "constant", "differential"]:
        S.append(sc(f"bias.{bt}", "bias_type", bias_type=bt))

    S.append(sc("scale.none", "scale_bias", bias_type="none", slope_a=1.0, slope_b=1.0, slope_c=1.0, slope_d=1.0))
    S.append(sc("scale.compress", "scale_bias", bias_type="none", slope_a=0.80, slope_b=0.80, slope_c=0.80, slope_d=0.80))
    S.append(sc("scale.expand", "scale_bias", bias_type="none", slope_a=1.20, slope_b=1.20, slope_c=1.20, slope_d=1.20))

    S.append(sc("gcal.none", "group_calibration", bias_type="none"))
    S.append(sc(
        "gcal.mild", "group_calibration", bias_type="none",
        bias_extra_a=0.15, bias_extra_b=-0.05, bias_extra_c=0.05, bias_extra_d=-0.10,
        slope_a=1.15, slope_b=0.95, slope_c=1.05, slope_d=0.90,
    ))
    S.append(sc(
        "gcal.strong", "group_calibration", bias_type="none",
        bias_extra_a=0.30, bias_extra_b=-0.10, bias_extra_c=0.10, bias_extra_d=-0.20,
        slope_a=1.30, slope_b=0.90, slope_c=1.10, slope_d=0.80,
    ))

    S.append(sc("corr.0.0", "repeated_corr", repeated_corr=0.0, bias_type="none"))
    S.append(sc("corr.0.3", "repeated_corr", repeated_corr=0.3, bias_type="none"))
    S.append(sc("corr.0.7", "repeated_corr", repeated_corr=0.7, bias_type="none"))

    S.append(sc("hetero.mild", "heteroskedastic", llm_noise=0.05, llm_noise2=0.50, llm_noise3=0.25))
    S.append(sc("hetero.extreme", "heteroskedastic", llm_noise=0.02, llm_noise2=0.80, llm_noise3=0.40))

    # Judge-noise FAMILY factor: does PPI correction still fix Type-I
    # inflation when judge measurement error is "mostly right, occasionally
    # catastrophically wrong" (a contaminated/heavy-tailed Gaussian mixture)
    # instead of symmetric, uniform-width Gaussian noise? Same total noise
    # variance (noise_sd**2) as the gaussian baseline either way -- only
    # where that variance comes from differs. See JudgeBiasSource.noise_family.
    S.append(sc("noise_family.contaminated-mild", "noise_family", noise_family="contaminated", contam_frac=0.10, contam_scale=3.0))
    S.append(sc("noise_family.contaminated-strong", "noise_family", noise_family="contaminated", contam_frac=0.10, contam_scale=6.0))
    S.append(sc("noise_family.contaminated+corr", "noise_family", noise_family="contaminated", contam_frac=0.15, contam_scale=5.0, repeated_corr=0.5))

    S.append(sc("stress.small+sparse", "stress", n=30, label_frac=0.07))
    S.append(sc("stress.large+noisy", "stress", n=300, llm_noise=0.70))
    S.append(sc("stress.unbal+diff", "stress", n=40, n2=200, label_frac=0.08))
    S.append(sc("stress.tiny_lab", "stress", n=200, label_frac=0.02))

    S.append(sc(
        "interact.small+unbal+hetero+diff", "interaction",
        n=30, n2=120, n3=80, label_frac=0.05, llm_noise=0.05, llm_noise2=0.70, llm_noise3=0.35,
        bias_type="differential",
    ))
    S.append(sc(
        "interact.small+sparse+noisy+const", "interaction",
        n=40, label_frac=0.05, llm_noise=0.80, bias_type="constant",
    ))
    S.append(sc(
        "interact.large+sparse+hetero+diff", "interaction",
        n=250, n2=400, n3=300, label_frac=0.03, llm_noise=0.02, llm_noise2=0.50, llm_noise3=0.90,
        bias_type="differential",
    ))
    S.append(sc(
        "interact.mid+extreme-hetero+none", "interaction",
        n=100, n2=150, n3=60, label_frac=0.08, llm_noise=0.01, llm_noise2=0.90, llm_noise3=0.40,
        bias_type="none",
    ))

    return S


@dataclass
class JudgeBiasCellData:
    """Output of generate_judge_bias_cell: every test-structure's truth/LLM/label
    arrays for one (JudgeBiasSource, rep) draw."""

    llm_a2: np.ndarray
    llm_b2: np.ndarray
    lab_a2: np.ndarray
    lab_b2: np.ndarray
    llm_x: np.ndarray
    llm_y: np.ndarray
    lab_x: np.ndarray
    lab_y: np.ndarray
    llm_a3: np.ndarray
    llm_b3: np.ndarray
    llm_c3: np.ndarray
    lab_a3: np.ndarray
    lab_b3: np.ndarray
    lab_c3: np.ndarray
    llm_A: np.ndarray
    llm_B: np.ndarray
    llm_C: np.ndarray
    lab_A: np.ndarray
    lab_B: np.ndarray
    lab_C: np.ndarray
    llm_W: np.ndarray
    llm_X: np.ndarray
    llm_Y: np.ndarray
    llm_Z: np.ndarray
    lab_W: np.ndarray
    lab_X: np.ndarray
    lab_Y: np.ndarray
    lab_Z: np.ndarray
    llm_A_runs: np.ndarray
    llm_B_runs: np.ndarray
    llm_C_runs: np.ndarray


def generate_judge_bias_cell(
    sc: JudgeBiasSource, rng: np.random.Generator, *, lmm_runs_r: int = JUDGE_BIAS_LMM_RUNS_R,
) -> JudgeBiasCellData:
    """Draw one replicate's worth of truth/LLM-judge/human-label arrays for
    every test structure (two independent groups, paired/repeated, three
    independent groups, three repeated conditions, a 2x2 crossed-factor
    design, and nested LLM runs), from one continuing rng stream.

    Truth comes from ONE representative shape per eval type (_ppi_shape),
    drawn via sample_group_truth -- k=1 (icc=1.0) for the independent-groups
    structures, k=2/3/4 at sc.icc for the paired/repeated structures -- the
    same shape catalog and generator build_pair_sources/build_multiarm_sources
    use, instead of this scenario family's own bespoke distribution.

    Effect size: ``sc.effect_size`` injects a real, monotonic mean shift into
    the truth itself (group/condition 0 unshifted, 1 gets +effect_size, 2 gets
    +2*effect_size, ...) -- 0.0 (the Type-I sweep default) makes this a no-op;
    >0 turns the same scenario grid into a power sweep.
    """
    shape = _ppi_shape(sc.eval_type, sc.shape_label)
    n1 = sc.n
    n2 = sc.n2 if sc.n2 is not None else sc.n
    n3 = sc.n3 if sc.n3 is not None else sc.n
    noise1 = sc.llm_noise
    noise2 = sc.llm_noise2 if sc.llm_noise2 is not None else sc.llm_noise
    noise3 = sc.llm_noise3 if sc.llm_noise3 is not None else sc.llm_noise
    anchor = _ppi_shape_anchor(shape)
    (bias_a, bias_b, bias_c), (slope_a, slope_b, slope_c) = _jb_judge_params_3(sc)
    es = sc.effect_size

    def _marginal(n: int) -> np.ndarray:
        return sample_group_truth(shape, n, 1, 1, 1.0, rng)[0, :, 0]

    def _repeated(n: int, n_conditions: int, effects: np.ndarray) -> np.ndarray:
        return sample_group_truth(shape, n, 1, n_conditions, sc.icc, rng, effects=effects)[:, :, 0]

    # -- Independent two-group data (ttest, mannwhitney) --
    truth_a2 = _marginal(n1)
    truth_b2 = _marginal(n2) + es
    llm_a2 = _jb_llm(truth_a2, bias_a, noise1, rng, slope=slope_a, anchor=anchor, noise_family=sc.noise_family, contam_frac=sc.contam_frac, contam_scale=sc.contam_scale)
    llm_b2 = _jb_llm(truth_b2, bias_b, noise2, rng, slope=slope_b, anchor=anchor, noise_family=sc.noise_family, contam_frac=sc.contam_frac, contam_scale=sc.contam_scale)
    lab_a2 = _jb_labels_independent(truth_a2, sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)
    lab_b2 = _jb_labels_independent(truth_b2, sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)

    # -- Paired data (wilcoxon) --
    truth_x, truth_y = _repeated(n1, 2, np.array([0.0, es]))
    llm_x, llm_y = _jb_llm_repeated(
        [truth_x, truth_y], [bias_a, bias_b], [noise1, noise2], [slope_a, slope_b],
        rng, anchor=anchor, corr=sc.repeated_corr,
        noise_family=sc.noise_family, contam_frac=sc.contam_frac, contam_scale=sc.contam_scale,
    )
    lab_x, lab_y = _jb_labels_shared([truth_x, truth_y], sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)

    # -- Independent three-group data (anova_ind) --
    truth_a3 = _marginal(n1)
    truth_b3 = _marginal(n2) + es
    truth_c3 = _marginal(n3) + 2 * es
    llm_a3 = _jb_llm(truth_a3, bias_a, noise1, rng, slope=slope_a, anchor=anchor, noise_family=sc.noise_family, contam_frac=sc.contam_frac, contam_scale=sc.contam_scale)
    llm_b3 = _jb_llm(truth_b3, bias_b, noise2, rng, slope=slope_b, anchor=anchor, noise_family=sc.noise_family, contam_frac=sc.contam_frac, contam_scale=sc.contam_scale)
    llm_c3 = _jb_llm(truth_c3, bias_c, noise3, rng, slope=slope_c, anchor=anchor, noise_family=sc.noise_family, contam_frac=sc.contam_frac, contam_scale=sc.contam_scale)
    lab_a3 = _jb_labels_independent(truth_a3, sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)
    lab_b3 = _jb_labels_independent(truth_b3, sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)
    lab_c3 = _jb_labels_independent(truth_c3, sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)

    # -- Repeated-measures three-group data (anova_rep, friedman, lmm) --
    truth_A, truth_B, truth_C = _repeated(n1, 3, np.array([0.0, es, 2 * es]))
    llm_A, llm_B, llm_C = _jb_llm_repeated(
        [truth_A, truth_B, truth_C], [bias_a, bias_b, bias_c], [noise1, noise2, noise3], [slope_a, slope_b, slope_c],
        rng, anchor=anchor, corr=sc.repeated_corr,
        noise_family=sc.noise_family, contam_frac=sc.contam_frac, contam_scale=sc.contam_scale,
    )
    lab_A, lab_B, lab_C = _jb_labels_shared([truth_A, truth_B, truth_C], sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)

    # -- 2x2 crossed fixed-factor data (lmm_factorial) --
    (bias_w, bias_x, bias_y, bias_z), (slope_w, slope_x, slope_y, slope_z) = _jb_judge_params_4(sc)
    truth_W, truth_X, truth_Y, truth_Z = _repeated(n1, 4, np.array([0.0, es, 2 * es, 3 * es]))
    llm_W, llm_X, llm_Y, llm_Z = _jb_llm_repeated(
        [truth_W, truth_X, truth_Y, truth_Z], [bias_w, bias_x, bias_y, bias_z],
        [noise1, noise2, noise3, noise1], [slope_w, slope_x, slope_y, slope_z],
        rng, anchor=anchor, corr=sc.repeated_corr,
        noise_family=sc.noise_family, contam_frac=sc.contam_frac, contam_scale=sc.contam_scale,
    )
    lab_W, lab_X, lab_Y, lab_Z = _jb_labels_shared([truth_W, truth_X, truth_Y, truth_Z], sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)

    # -- Nested run replication (lmm_runs) --
    a_cols: list[np.ndarray] = []
    b_cols: list[np.ndarray] = []
    c_cols: list[np.ndarray] = []
    for _ in range(lmm_runs_r):
        rA, rB, rC = _jb_llm_repeated(
            [truth_A, truth_B, truth_C], [bias_a, bias_b, bias_c], [noise1, noise2, noise3], [slope_a, slope_b, slope_c],
            rng, anchor=anchor, corr=sc.repeated_corr,
            noise_family=sc.noise_family, contam_frac=sc.contam_frac, contam_scale=sc.contam_scale,
        )
        a_cols.append(rA)
        b_cols.append(rB)
        c_cols.append(rC)
    llm_A_runs = np.column_stack(a_cols)
    llm_B_runs = np.column_stack(b_cols)
    llm_C_runs = np.column_stack(c_cols)

    return JudgeBiasCellData(
        llm_a2=llm_a2, llm_b2=llm_b2, lab_a2=lab_a2, lab_b2=lab_b2,
        llm_x=llm_x, llm_y=llm_y, lab_x=lab_x, lab_y=lab_y,
        llm_a3=llm_a3, llm_b3=llm_b3, llm_c3=llm_c3, lab_a3=lab_a3, lab_b3=lab_b3, lab_c3=lab_c3,
        llm_A=llm_A, llm_B=llm_B, llm_C=llm_C, lab_A=lab_A, lab_B=lab_B, lab_C=lab_C,
        llm_W=llm_W, llm_X=llm_X, llm_Y=llm_Y, llm_Z=llm_Z, lab_W=lab_W, lab_X=lab_X, lab_Y=lab_Y, lab_Z=lab_Z,
        llm_A_runs=llm_A_runs, llm_B_runs=llm_B_runs, llm_C_runs=llm_C_runs,
    )


def estimate_judge_bias_gold_null_values(sc: JudgeBiasSource, *, n_mc: int = 3000, seed: int = 0) -> dict[str, float]:
    """Monte Carlo expectation of each test's raw estimand on pure-truth (no
    LLM bias/noise) data at this scenario's exact sample sizes -- the
    bias/coverage effect-size check's "true null value" (not always exactly
    0; see sim_type_i_calibration.py's _gold_null_values docstring for why).
    """
    from evalstats.tests import _p_x_gt_y_midrank, _anova_between_variance_from_groups, _repeated_condition_variance, _friedman_rank_variance

    rng = np.random.default_rng(seed)
    shape = _ppi_shape(sc.eval_type, sc.shape_label)
    n1 = sc.n
    n2 = sc.n2 if sc.n2 is not None else sc.n
    n3 = sc.n3 if sc.n3 is not None else sc.n

    def _marginal(n: int) -> np.ndarray:
        return sample_group_truth(shape, n, 1, 1, 1.0, rng)[0, :, 0]

    def _repeated(n: int, n_conditions: int) -> np.ndarray:
        return sample_group_truth(shape, n, 1, n_conditions, sc.icc, rng)[:, :, 0]

    diffs2 = np.empty(n_mc)
    thetas2 = np.empty(n_mc)
    for i in range(n_mc):
        a = _marginal(n1)
        b = _marginal(n2)
        diffs2[i] = a.mean() - b.mean()
        thetas2[i] = _p_x_gt_y_midrank(a, b) - 0.5

    meds = np.empty(n_mc)
    for i in range(n_mc):
        x, y = _repeated(n1, 2)
        meds[i] = float(np.median(x - y))

    bv = np.empty(n_mc)
    for i in range(n_mc):
        a3 = _marginal(n1)
        b3 = _marginal(n2)
        c3 = _marginal(n3)
        bv[i] = _anova_between_variance_from_groups([a3, b3, c3])

    rv = np.empty(n_mc)
    frv = np.empty(n_mc)
    for i in range(n_mc):
        mat = _repeated(n1, 3).T  # (n1, 3)
        rv[i] = _repeated_condition_variance(mat)
        frv[i] = _friedman_rank_variance(mat)

    return {
        "ttest": float(diffs2.mean()),
        "ttest_welch": float(diffs2.mean()),
        "mw": float(thetas2.mean()),
        "wilcoxon": float(meds.mean()),
        "anova_ind": float(bv.mean()),
        "anova_rep": float(rv.mean()),
        "friedman": float(frv.mean()),
        "kruskal": 0.5,
    }
