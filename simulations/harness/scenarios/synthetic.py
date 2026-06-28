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


def _legacy_build_single_sample_sources(suite: str = "standard") -> list["CISource"]:
    """UNUSED -- kept for reference only. Superseded by the new
    build_single_sample_sources below, which sources its "param" shapes
    (Beta/latent-Normal families) from the SAME module-level shape catalogs
    (BINARY_SHAPES/CONTINUOUS_SHAPES/LIKERT_SHAPES/GRADES_SHAPES) that
    build_pair_sources and build_multiarm_sources use, via sample_group_truth
    with k=1. Not imported by any case; left in place only so the original,
    independently-evolved single-sample catalog is still visible for
    comparison/audit."""
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")

    sources: list[CISource] = []

    # ------------------------------------------------------------------
    # Binary: Bernoulli(p) -> 0/1 pass-fail judgements
    # ------------------------------------------------------------------
    binary_ps = [0.1, 0.3, 0.5, 0.7, 0.9]
    if suite in ("expanded", "extreme"):
        binary_ps += [0.02, 0.05, 0.95, 0.98]

    for p in binary_ps:
        p_ = p
        sources.append(
            CISource(
                label=f"p={p_}",
                eval_type="binary",
                generate=lambda rng, n, _p=p_: rng.binomial(1, _p, n).astype(float),
                true_mean=p_,
            )
        )

    # ------------------------------------------------------------------
    # Continuous [0, 1]: Beta distributions of varying shape
    # ------------------------------------------------------------------
    continuous_specs = [
        ("Uniform", 1.0, 1.0),  # flat
        ("U-shaped", 0.5, 0.5),  # bimodal-ish extremes
        ("right-skewed", 2.0, 8.0),  # mass near 0
        ("left-skewed", 8.0, 2.0),  # mass near 1
        ("moderate-skew", 2.0, 5.0),  # asymmetric centre
    ]
    if suite in ("expanded", "extreme"):
        continuous_specs.extend(
            [
                ("extreme-right", 0.35, 6.0),
                ("extreme-left", 6.0, 0.35),
                ("near-boundaries", 0.3, 0.3),
                ("near-center", 6.0, 6.0),
            ]
        )

    for label, a, b in continuous_specs:
        a_, b_ = a, b
        sources.append(
            CISource(
                label=f"{label} Beta({a_},{b_})",
                eval_type="continuous",
                generate=lambda rng, n, _a=a_, _b=b_: rng.beta(_a, _b, n),
                true_mean=a_ / (a_ + b_),
            )
        )

    def _gen_logit_normal(rng: np.random.Generator, n: int) -> np.ndarray:
        logits = rng.normal(-0.35, 1.35, size=n)
        return 1.0 / (1.0 + np.exp(-logits))

    sources.append(
        CISource(
            label="logit-normal(mu=-0.35,s=1.35)",
            eval_type="continuous",
            generate=_gen_logit_normal,
            true_mean=_estimate_true_mean_mc(_gen_logit_normal),
        )
    )

    def _gen_zero_inflated(rng: np.random.Generator, n: int) -> np.ndarray:
        spike = rng.random(n) < 0.70
        return np.where(spike, 0.0, rng.beta(2.0, 4.0, n))

    sources.append(
        CISource(
            label="zero-inflated(pi=0.70,Beta(2,4))",
            eval_type="continuous",
            generate=_gen_zero_inflated,
            true_mean=0.30 * (2.0 / 6.0),
        )
    )

    def _gen_one_inflated(rng: np.random.Generator, n: int) -> np.ndarray:
        spike = rng.random(n) < 0.70
        return np.where(spike, 1.0, rng.beta(4.0, 2.0, n))

    sources.append(
        CISource(
            label="one-inflated(pi=0.70,Beta(4,2))",
            eval_type="continuous",
            generate=_gen_one_inflated,
            true_mean=0.70 + 0.30 * (4.0 / 6.0),
        )
    )

    if suite in ("expanded", "extreme"):
        def _gen_mix_continuous(rng: np.random.Generator, n: int) -> np.ndarray:
            selector = rng.binomial(1, 0.55, size=n).astype(bool)
            vals = np.empty(n, dtype=float)
            vals[selector] = rng.beta(0.5, 4.0, size=int(np.sum(selector)))
            vals[~selector] = rng.beta(5.5, 1.2, size=int(np.sum(~selector)))
            return vals

        sources.append(
            CISource(
                label="mixture Beta(0.5,4.0)/(5.5,1.2)",
                eval_type="continuous",
                generate=_gen_mix_continuous,
                true_mean=_estimate_true_mean_mc(_gen_mix_continuous),
            )
        )

    def _make_likert_normal(mu: float, sigma: float):
        def _gen(rng: np.random.Generator, n: int, _m: float = mu, _s: float = sigma) -> np.ndarray:
            return np.clip(np.rint(rng.normal(_m, _s, n)), 1.0, 5.0)
        return _gen

    def _make_likert_bimodal(mu1: float, mu2: float, sigma: float):
        def _gen(rng: np.random.Generator, n: int, _m1: float = mu1, _m2: float = mu2, _s: float = sigma) -> np.ndarray:
            sel = rng.random(n) < 0.5
            latents = np.where(sel, rng.normal(_m1, _s, n), rng.normal(_m2, _s, n))
            return np.clip(np.rint(latents), 1.0, 5.0)
        return _gen

    _likert_standard = [
        ("uniform", _make_likert_normal(3.0, 2.0)),
        ("skewed-low", _make_likert_normal(2.0, 1.1)),
        ("skewed-high", _make_likert_normal(4.0, 1.1)),
        ("bimodal", _make_likert_bimodal(1.5, 4.5, 0.65)),
        ("center-peaked", _make_likert_normal(3.0, 0.55)),
    ]
    for label, gen in _likert_standard:
        sources.append(CISource(label=label, eval_type="likert", generate=gen, true_mean=_estimate_true_mean_mc(gen)))

    if suite in ("expanded", "extreme"):
        _likert_expanded = [
            ("near-floor", _make_likert_normal(1.5, 0.65)),
            ("near-ceiling", _make_likert_normal(4.5, 0.65)),
            ("polarized", _make_likert_bimodal(1.3, 4.7, 0.50)),
            ("flat-middle", _make_likert_normal(3.0, 1.4)),
        ]
        for label, gen in _likert_expanded:
            sources.append(CISource(label=label, eval_type="likert", generate=gen, true_mean=_estimate_true_mean_mc(gen)))

    grade_specs = [
        ("symmetric", 50, 20),
        ("high-scoring", 75, 15),
        ("low-scoring", 35, 20),
        ("ceiling-heavy", 88, 10),
        ("floor-heavy", 12, 10),
    ]
    if suite in ("expanded", "extreme"):
        grade_specs.extend([("very-high", 92, 7), ("very-low", 8, 7), ("high-variance", 50, 34)])

    for label, mu, sigma in grade_specs:
        mu_, sigma_ = mu, sigma
        true_mean_ = _true_mean_clipped_normal(mu_, sigma_)
        sources.append(
            CISource(
                label=f"{label} N({mu_},{sigma_})",
                eval_type="grades",
                generate=lambda rng, n, _m=mu_, _s=sigma_: np.clip(rng.normal(_m, _s, n), 0.0, 100.0),
                true_mean=true_mean_,
            )
        )

    def _gen_grade_mixture(rng: np.random.Generator, n: int) -> np.ndarray:
        flags = rng.choice(3, size=n, p=[0.20, 0.50, 0.30])
        vals = np.empty(n, dtype=float)
        for bucket, mu, sigma in [(0, 22.0, 11.0), (1, 58.0, 14.0), (2, 88.0, 8.0)]:
            mask = flags == bucket
            vals[mask] = rng.normal(mu, sigma, size=int(np.sum(mask)))
        return np.clip(vals, 0.0, 100.0)

    sources.append(
        CISource(
            label="mixture-truncnorm(3 components)",
            eval_type="grades",
            generate=_gen_grade_mixture,
            true_mean=_estimate_true_mean_mc(_gen_grade_mixture),
        )
    )

    def _gen_grade_heavy_tail(rng: np.random.Generator, n: int) -> np.ndarray:
        vals = 52.0 + 16.0 * rng.standard_t(df=3.0, size=n)
        return np.clip(vals, 0.0, 100.0)

    sources.append(
        CISource(
            label="heavy-tail t(df=3)",
            eval_type="grades",
            generate=_gen_grade_heavy_tail,
            true_mean=_estimate_true_mean_mc(_gen_grade_heavy_tail),
        )
    )

    if suite in ("expanded", "extreme"):
        def _gen_grade_zero_spiked(rng: np.random.Generator, n: int) -> np.ndarray:
            spike = rng.random(n) < 0.40
            body = np.clip(rng.normal(45.0, 20.0, n), 0.0, 100.0)
            return np.where(spike, 0.0, body)

        sources.append(
            CISource(
                label="zero-spiked(pi=0.40,N(45,20))",
                eval_type="grades",
                generate=_gen_grade_zero_spiked,
                true_mean=_estimate_true_mean_mc(_gen_grade_zero_spiked),
            )
        )

        def _gen_grade_hundred_spiked(rng: np.random.Generator, n: int) -> np.ndarray:
            spike = rng.random(n) < 0.40
            body = np.clip(rng.normal(65.0, 18.0, n), 0.0, 100.0)
            return np.where(spike, 100.0, body)

        sources.append(
            CISource(
                label="hundred-spiked(pi=0.40,N(65,18))",
                eval_type="grades",
                generate=_gen_grade_hundred_spiked,
                true_mean=_estimate_true_mean_mc(_gen_grade_hundred_spiked),
            )
        )

    return sources


def _legacy_build_pair_sources(
    suite: str = "standard",
    icc_values: list[float] | tuple[float, ...] = (0.10, 0.25, 0.40),
    cohens_d_values: list[float] | tuple[float, ...] = (0.3,),
    include_null: bool = False,
) -> list["CIPairSource"]:
    """UNUSED -- kept for reference only. Superseded by the new
    build_pair_sources below, which is a thin (icc, shape, d) sweep over
    sample_group_truth(k=2) using the SAME module-level shape catalogs
    build_single_sample_sources (k=1) and build_multiarm_sources (k>=2) use.
    Not imported by any case."""
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")

    sources: list[CIPairSource] = []
    icc_list = list(icc_values)
    d_list = list(cohens_d_values)
    if include_null:
        d_list = [0.0] + [d for d in d_list if d > 0.0]

    binary_shapes: list[tuple[str, float]] = [
        ("binary-balanced", 0.5), ("binary-high", 0.8), ("binary-low", 0.2), ("binary-near-ceil", 0.92),
    ]
    if suite in ("expanded", "extreme"):
        binary_shapes += [("binary-rare", 0.05), ("binary-near-ceil-hi", 0.95)]

    for icc in icc_list:
        conc = _binary_conc_from_icc(icc)
        for shape_label, base_p in binary_shapes:
            total_std = float(np.sqrt(base_p * (1.0 - base_p)))
            for d in d_list:
                delta = d * total_std
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|icc={icc:.2f}|{effect_tag}"
                bp_, delta_, conc_ = base_p, delta, conc

                def _gen_binary(rng, n, runs, _bp=bp_, _d=delta_, _c=conc_):
                    p_a = rng.beta(_bp * _c, (1.0 - _bp) * _c, size=(n, 1))
                    p_b = np.clip(p_a + _d, 0.0, 1.0)
                    a = rng.binomial(1, p_a, size=(n, runs)).astype(float)
                    b = rng.binomial(1, p_b, size=(n, runs)).astype(float)
                    return a, b

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_binary)
                sources.append(CIPairSource(label=label, eval_type="binary", generate_pair=_gen_binary, true_diff=true_diff, icc=icc, cohens_d=d, is_null=is_null))

    continuous_shapes: list[tuple[str, float, float]] = [
        ("cont-uniform", 1.0, 1.0), ("cont-right-skew", 2.0, 8.0), ("cont-left-skew", 8.0, 2.0),
    ]
    if suite in ("expanded", "extreme"):
        continuous_shapes += [("cont-moderate-skew", 2.0, 5.0), ("cont-boundary", 0.6, 0.6)]

    for icc in icc_list:
        for shape_label, a_beta, b_beta in continuous_shapes:
            var_base = _beta_var(a_beta, b_beta)
            noise_std = float(np.sqrt(max(var_base * (1.0 / max(icc, 1e-9) - 1.0) / 2.0, 0.0)))
            total_std = float(np.sqrt(var_base / max(icc, 1e-9)))
            for d in d_list:
                delta = d * total_std
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|icc={icc:.2f}|{effect_tag}"
                a_, b_, ns_, delta_ = a_beta, b_beta, noise_std, delta

                def _gen_continuous(rng, n, runs, _a=a_, _b=b_, _ns=ns_, _d=delta_):
                    base = rng.beta(_a, _b, size=(n, 1))
                    shared = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_a = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_b = rng.normal(0.0, _ns, size=(n, runs))
                    a_vals = np.clip(base + shared + indiv_a, 0.0, 1.0)
                    b_vals = np.clip(base + _d + shared + indiv_b, 0.0, 1.0)
                    return a_vals, b_vals

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_continuous)
                sources.append(CIPairSource(label=label, eval_type="continuous", generate_pair=_gen_continuous, true_diff=true_diff, icc=icc, cohens_d=d, is_null=is_null))

    likert_shapes: list[tuple[str, float]] = [("likert-mid", 3.0), ("likert-low", 2.2), ("likert-high", 3.8)]
    if suite in ("expanded", "extreme"):
        likert_shapes += [("likert-polarized", 3.0), ("likert-floor", 1.8)]

    for icc in icc_list:
        base_std_l = float(np.sqrt(icc)) * _LIKERT_TOTAL_STD
        noise_std_l = float(np.sqrt(max((1.0 - icc) / 2.0, 0.0))) * _LIKERT_TOTAL_STD
        for shape_label, mu_lat in likert_shapes:
            for d in d_list:
                delta = d * _LIKERT_TOTAL_STD
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|icc={icc:.2f}|{effect_tag}"
                m_, bs_, ns_, delta_ = mu_lat, base_std_l, noise_std_l, delta

                def _gen_likert(rng, n, runs, _m=m_, _bs=bs_, _ns=ns_, _d=delta_):
                    base = rng.normal(_m, _bs, size=(n, 1))
                    shared = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_a = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_b = rng.normal(0.0, _ns, size=(n, runs))
                    a_vals = np.rint(np.clip(base + shared + indiv_a, 1.0, 5.0))
                    b_vals = np.rint(np.clip(base + _d + shared + indiv_b, 1.0, 5.0))
                    return a_vals, b_vals

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_likert)
                sources.append(CIPairSource(label=label, eval_type="likert", generate_pair=_gen_likert, true_diff=true_diff, icc=icc, cohens_d=d, is_null=is_null))

    grades_shapes: list[tuple[str, float]] = [("grades-mid", 55.0), ("grades-low", 35.0), ("grades-high", 78.0)]
    if suite in ("expanded", "extreme"):
        grades_shapes += [("grades-ceiling", 86.0), ("grades-floor", 20.0)]

    for icc in icc_list:
        base_std_g = float(np.sqrt(icc)) * _GRADES_TOTAL_STD
        noise_std_g = float(np.sqrt(max((1.0 - icc) / 2.0, 0.0))) * _GRADES_TOTAL_STD
        for shape_label, mu_g in grades_shapes:
            for d in d_list:
                delta = d * _GRADES_TOTAL_STD
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = f"{shape_label}|icc={icc:.2f}|{effect_tag}"
                m_, bs_, ns_, delta_ = mu_g, base_std_g, noise_std_g, delta

                def _gen_grades(rng, n, runs, _m=m_, _bs=bs_, _ns=ns_, _d=delta_):
                    base = rng.normal(_m, _bs, size=(n, 1))
                    shared = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_a = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_b = rng.normal(0.0, _ns, size=(n, runs))
                    a_vals = np.clip(base + shared + indiv_a, 0.0, 100.0)
                    b_vals = np.clip(base + _d + shared + indiv_b, 0.0, 100.0)
                    return a_vals, b_vals

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_grades)
                sources.append(CIPairSource(label=label, eval_type="grades", generate_pair=_gen_grades, true_diff=true_diff, icc=icc, cohens_d=d, is_null=is_null))

    return sources


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
    """For kind="custom": draws n FINAL (already clipped/rounded) values directly. k=1 only."""
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


def group_total_std(eval_type: str, params, icc: float) -> float:
    """Total marginal std of one (eval_type, params) truth+noise model at a
    given icc -- so cohens_d * group_total_std(...) is the same effect-size
    convention build_pair_sources/build_multiarm_sources/PPI all share."""
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

    base_corr: correlation between any two of the k groups' BASE/item-level
    values (1.0 = fully shared base -- every group sees the literal same
    item, the harness default everywhere except build_pair_sources'
    run_noise_fracs multi-run mode; <1.0 = partial agreement, a Gaussian-
    copula "different items/judges, correlated quality" model, generalizing
    _legacy_build_pair_multirun_sources' cross_item_rho to general k). At base_corr
    == 1.0 and scalar icc, this is a no-op vs. the pre-existing shared-base
    code path (kept as its own branch below to preserve exact-parity with
    everything that doesn't pass base_corr/per-group icc).

    k=1 has no inter-group correlation concept -- the single group gets the
    FULL within-item noise variance directly (same marginal variance as any
    one of the k>=2 groups at the same icc), reproducing
    build_single_sample_sources' simple "param" shapes exactly when icc=1.0
    (no separate noise layer at all); icc<1.0 at k=1, runs>1 is the
    run_noise_frac multi-run case (f_run = 1 - icc).

    shape.kind == "custom" requires k == 1 (these bespoke single-population
    generators have no pair/multi-arm analogue), but DOES support icc<1.0,
    runs>1 (multi-run): the bare custom_sampler draw is treated as the
    per-item "base", with the same additive-Gaussian within-item noise model
    as "param" shapes, using an MC-estimated (not closed-form) base variance.
    """
    if effects is None:
        effects = np.zeros(k)
    et = shape.eval_type
    icc_arr = np.full(k, float(icc)) if np.ndim(icc) == 0 else np.asarray(icc, dtype=float)
    if icc_arr.shape != (k,):
        raise ValueError(f"icc must be a scalar or a length-k sequence, got shape {icc_arr.shape} for k={k}")
    icc_scalar = bool(np.ndim(icc) == 0)

    if shape.kind == "custom":
        if k != 1:
            raise ValueError(f"Custom shape {shape.label!r} only supports k=1 (single-sample), got k={k}")
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

    if et == "binary":
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

    if et == "continuous":
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

    group_noise_vars = np.array([_group_noise_var(et, shape.params, ic) for ic in icc_arr])
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
        param_shapes = [s for s in _tier_shapes(catalog, suite) if s.kind == "param"]
        for f_a, f_b in noise_pairs:
            icc_a, icc_b = 1.0 - f_a, 1.0 - f_b
            icc_eff = 0.5 * (icc_a + icc_b)
            for shape in param_shapes:
                total_std = group_total_std(eval_type, shape.params, icc_eff)
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
    sample_group_truth(k=2) using the shape catalog above ("param" shapes
    only; "custom" shapes have no pair analogue, see ShapeSpec docs).

    ICC = between-input variance / total variance; Cohen's d = delta /
    total_std (group_total_std(eval_type, shape.params, icc)).

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
        param_shapes = [s for s in _tier_shapes(catalog, suite) if s.kind == "param"]
        for icc in icc_list:
            for shape in param_shapes:
                total_std = group_total_std(eval_type, shape.params, icc)
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




# ---------------------------------------------------------------------------
# Multi-run (nested-mode) single-sample sources
# ---------------------------------------------------------------------------
# Ported from sim_compare_boot_nested.py's build_multirun_scenarios(). Used by
# cases/ci_single.py's --nested-mode, parameterized by run_noise_frac rather
# than ICC: f_run = sigma^2_run / (sigma^2_input + sigma^2_run). Low f_run =
# high ICC = clustered runs; f_run ~= 1 = run noise dominates.


def _estimate_true_mean_mc_runs(
    generate_runs: Callable[[np.random.Generator, int, int], np.ndarray],
    *,
    seed: int = 0,
    n_mc: int = 500_000,
) -> float:
    rng = np.random.default_rng(seed)
    return float(np.mean(generate_runs(rng, n_mc, 1)))


def _legacy_build_multirun_sources(
    run_noise_fracs: list[float],
    suite: str = "standard",
    heteroscedastic: bool = False,
) -> list[CISource]:
    """UNUSED -- kept for reference only. Pre-unification implementation,
    superseded by build_single_sample_sources' run_noise_fracs param (which
    sweeps the same icc = 1 - f_run axis through sample_group_truth(k=1)
    instead of a separate bespoke per-shape generator catalog).

    Build single-sample multi-run sources parameterised by run_noise_frac.

    DGPs (matching build_single_sample_sources' shape families):
      binary:     Bernoulli-Beta hierarchical. p_i ~ Beta(conc*p0, conc*(1-p0)),
                  ICC = 1/(conc+1), f_run = conc/(conc+1).
      continuous: base_i ~ Beta(a,b), x_{i,r} = clip(base_i + N(0,sigma_run), 0,1).
      likert:     latent_i ~ N(mu, sigma_input), x_{i,r} = round(clip(latent_i + N(0,sigma_run),1,5)).
      grades:     same as likert, clipped to [0, 100].
    """
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")

    sources: list[CISource] = []

    def _add(label: str, eval_type: str, gen_fn, true_mean: float, f: float) -> None:
        sources.append(CISource(
            label=label, eval_type=eval_type, true_mean=true_mean,
            generate=lambda rng, n, _g=gen_fn: _g(rng, n, 1)[:, 0],
            generate_runs=gen_fn, run_noise_frac=f,
        ))

    # -- Binary --
    binary_ps = [0.1, 0.3, 0.5, 0.7, 0.9]
    if suite in ("expanded", "extreme"):
        binary_ps += [0.02, 0.05, 0.95, 0.98]

    for p0 in binary_ps:
        for f in run_noise_fracs:
            conc = float(max(f, 1e-6)) / float(max(1.0 - f, 1e-6))
            p_, c_ = p0, conc

            def _gen_bin(rng: np.random.Generator, n: int, runs: int, _p: float = p_, _c: float = c_) -> np.ndarray:
                p_i = rng.beta(_c * _p, _c * (1.0 - _p), size=(n, 1))
                return rng.binomial(1, p_i, size=(n, runs)).astype(float)

            _add(f"p={p0}|f={f:.2f}", "binary", _gen_bin, p0, f)

    # -- Continuous [0, 1] --
    continuous_specs: list[tuple[str, float, float]] = [
        ("Uniform", 1.0, 1.0), ("U-shaped", 0.5, 0.5), ("right-skewed", 2.0, 8.0),
        ("left-skewed", 8.0, 2.0), ("moderate-skew", 2.0, 5.0),
    ]
    if suite in ("expanded", "extreme"):
        continuous_specs.extend([
            ("extreme-right", 0.35, 6.0), ("extreme-left", 6.0, 0.35),
            ("near-boundaries", 0.3, 0.3), ("near-center", 6.0, 6.0),
        ])

    for shape_label, a_b, b_b in continuous_specs:
        var_base = _beta_var(a_b, b_b)
        for f in run_noise_fracs:
            sigma_run = float(np.sqrt(var_base * f / max(1.0 - f, 1e-9)))
            a_, b_, sr_ = a_b, b_b, sigma_run

            def _gen_cont(
                rng: np.random.Generator, n: int, runs: int,
                _a: float = a_, _b: float = b_, _sr: float = sr_, _hetero: bool = heteroscedastic,
            ) -> np.ndarray:
                base = rng.beta(_a, _b, size=(n, 1))
                if _sr > 0.0:
                    if _hetero:
                        sigma_i = _sr * 2.0 * np.sqrt(base * (1.0 - base))
                        noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                    else:
                        noise = rng.normal(0.0, _sr, size=(n, runs))
                else:
                    noise = np.zeros((n, runs))
                return np.clip(base + noise, 0.0, 1.0)

            true_mean = _estimate_true_mean_mc_runs(_gen_cont)
            _add(f"{shape_label}|f={f:.2f}", "continuous", _gen_cont, true_mean, f)

    # logit-normal base (always included)
    for f in run_noise_fracs:
        rng_tmp = np.random.default_rng(1)
        base_logit = rng_tmp.normal(-0.35, 1.35, 200_000)
        base_vals = 1.0 / (1.0 + np.exp(-base_logit))
        var_logit = float(np.var(base_vals))
        sigma_run = float(np.sqrt(var_logit * f / max(1.0 - f, 1e-9)))
        sr_ = sigma_run

        def _gen_logit(
            rng: np.random.Generator, n: int, runs: int, _sr: float = sr_, _hetero: bool = heteroscedastic,
        ) -> np.ndarray:
            logits = rng.normal(-0.35, 1.35, size=(n, 1))
            base = 1.0 / (1.0 + np.exp(-logits))
            if _sr > 0.0:
                if _hetero:
                    sigma_i = _sr * 2.0 * np.sqrt(base * (1.0 - base))
                    noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                else:
                    noise = rng.normal(0.0, _sr, size=(n, runs))
            else:
                noise = np.zeros((n, runs))
            return np.clip(base + noise, 0.0, 1.0)

        true_mean = _estimate_true_mean_mc_runs(_gen_logit)
        _add(f"logit-normal|f={f:.2f}", "continuous", _gen_logit, true_mean, f)

    # zero-inflated and one-inflated
    for shape_name, spike_val, beta_a, beta_b, spike_prob in [
        ("zero-inflated", 0.0, 2.0, 4.0, 0.70), ("one-inflated", 1.0, 4.0, 2.0, 0.70),
    ]:
        rng_tmp = np.random.default_rng(2)
        spike_mask = rng_tmp.random(200_000) < spike_prob
        body = rng_tmp.beta(beta_a, beta_b, 200_000)
        base_vals_zi = np.where(spike_mask, spike_val, body)
        var_zi = float(np.var(base_vals_zi))

        for f in run_noise_fracs:
            sigma_run = float(np.sqrt(var_zi * f / max(1.0 - f, 1e-9)))
            sv_, ba_, bb_, sp_, sr_ = spike_val, beta_a, beta_b, spike_prob, sigma_run

            def _gen_infl(
                rng: np.random.Generator, n: int, runs: int,
                _sv: float = sv_, _ba: float = ba_, _bb: float = bb_, _sp: float = sp_, _sr: float = sr_,
                _hetero: bool = heteroscedastic,
            ) -> np.ndarray:
                spike_i = rng.random((n, 1)) < _sp
                base = np.where(spike_i, _sv, rng.beta(_ba, _bb, size=(n, 1)))
                if _sr > 0.0:
                    if _hetero:
                        sigma_i = _sr * 2.0 * np.sqrt(base * (1.0 - base))
                        noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                    else:
                        noise = rng.normal(0.0, _sr, size=(n, runs))
                else:
                    noise = np.zeros((n, runs))
                return np.clip(base + noise, 0.0, 1.0)

            true_mean = _estimate_true_mean_mc_runs(_gen_infl)
            _add(f"{shape_name}|f={f:.2f}", "continuous", _gen_infl, true_mean, f)

    if suite in ("expanded", "extreme"):
        rng_tmp = np.random.default_rng(3)
        sel = rng_tmp.binomial(1, 0.55, size=200_000).astype(bool)
        v = np.empty(200_000, dtype=float)
        v[sel] = rng_tmp.beta(0.5, 4.0, size=int(np.sum(sel)))
        v[~sel] = rng_tmp.beta(5.5, 1.2, size=int(np.sum(~sel)))
        var_mix = float(np.var(v))

        for f in run_noise_fracs:
            sigma_run = float(np.sqrt(var_mix * f / max(1.0 - f, 1e-9)))
            sr_ = sigma_run

            def _gen_mix(
                rng: np.random.Generator, n: int, runs: int, _sr: float = sr_, _hetero: bool = heteroscedastic,
            ) -> np.ndarray:
                selector = rng.binomial(1, 0.55, size=(n, 1)).astype(bool)
                base = np.where(selector, rng.beta(0.5, 4.0, size=(n, 1)), rng.beta(5.5, 1.2, size=(n, 1)))
                if _sr > 0.0:
                    if _hetero:
                        sigma_i = _sr * 2.0 * np.sqrt(base * (1.0 - base))
                        noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                    else:
                        noise = rng.normal(0.0, _sr, size=(n, runs))
                else:
                    noise = np.zeros((n, runs))
                return np.clip(base + noise, 0.0, 1.0)

            true_mean = _estimate_true_mean_mc_runs(_gen_mix)
            _add(f"mix-Beta|f={f:.2f}", "continuous", _gen_mix, true_mean, f)

    # -- Likert 1-5 --
    _likert_standard: list[tuple[str, float, float, bool]] = [
        ("uniform", 3.0, 2.0, False), ("skewed-low", 2.0, 1.1, False), ("skewed-high", 4.0, 1.1, False),
        ("bimodal", 3.0, 0.65, True), ("center-peaked", 3.0, 0.55, False),
    ]
    if suite in ("expanded", "extreme"):
        _likert_standard += [
            ("near-floor", 1.5, 0.65, False), ("near-ceiling", 4.5, 0.65, False), ("flat-middle", 3.0, 1.4, False),
        ]

    for shape_label, mu_lat, s0, is_bimodal in _likert_standard:
        for f in run_noise_fracs:
            sigma_input_l = float(np.sqrt(max(1.0 - f, 0.0))) * s0
            sigma_run_l = float(np.sqrt(f)) * s0
            m_, si_, sr_, bim_ = mu_lat, sigma_input_l, sigma_run_l, is_bimodal

            if is_bimodal:
                def _gen_likert_bim(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _si: float = si_, _sr: float = sr_, _hetero: bool = heteroscedastic,
                ) -> np.ndarray:
                    mode = rng.integers(0, 2, size=(n, 1))
                    mu_mode = np.where(mode == 0, _m - 1.5, _m + 1.5)
                    latent_i = mu_mode + rng.normal(0.0, _si, size=(n, 1))
                    if _sr > 0.0:
                        if _hetero:
                            p_i = np.clip((latent_i - 1.0) / 4.0, 0.0, 1.0)
                            sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                            noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                        else:
                            noise = rng.normal(0.0, _sr, size=(n, runs))
                    else:
                        noise = np.zeros((n, runs))
                    return np.rint(np.clip(latent_i + noise, 1.0, 5.0))

                gen_fn = _gen_likert_bim
            else:
                def _gen_likert_norm(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _si: float = si_, _sr: float = sr_, _hetero: bool = heteroscedastic,
                ) -> np.ndarray:
                    latent_i = rng.normal(_m, _si, size=(n, 1)) if _si > 0.0 else np.full((n, 1), _m)
                    if _sr > 0.0:
                        if _hetero:
                            p_i = np.clip((latent_i - 1.0) / 4.0, 0.0, 1.0)
                            sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                            noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                        else:
                            noise = rng.normal(0.0, _sr, size=(n, runs))
                    else:
                        noise = np.zeros((n, runs))
                    return np.rint(np.clip(latent_i + noise, 1.0, 5.0))

                gen_fn = _gen_likert_norm

            true_mean = _estimate_true_mean_mc_runs(gen_fn)
            _add(f"{shape_label}|f={f:.2f}", "likert", gen_fn, true_mean, f)

    if suite in ("expanded", "extreme"):
        for f in run_noise_fracs:
            sigma_input_l = float(np.sqrt(max(1.0 - f, 0.0))) * 0.50
            sigma_run_l = float(np.sqrt(f)) * 0.50
            si_, sr_ = sigma_input_l, sigma_run_l

            def _gen_likert_polarized(
                rng: np.random.Generator, n: int, runs: int,
                _si: float = si_, _sr: float = sr_, _hetero: bool = heteroscedastic,
            ) -> np.ndarray:
                sel = rng.random((n, 1)) < 0.5
                mu_i = np.where(sel, 1.3, 4.7)
                latent_i = mu_i + rng.normal(0.0, _si, size=(n, 1)) if _si > 0.0 else mu_i.astype(float)
                if _sr > 0.0:
                    if _hetero:
                        p_i = np.clip((latent_i - 1.0) / 4.0, 0.0, 1.0)
                        sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                        noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                    else:
                        noise = rng.normal(0.0, _sr, size=(n, runs))
                else:
                    noise = np.zeros((n, runs))
                return np.rint(np.clip(latent_i + noise, 1.0, 5.0))

            true_mean = _estimate_true_mean_mc_runs(_gen_likert_polarized)
            _add(f"polarized|f={f:.2f}", "likert", _gen_likert_polarized, true_mean, f)

    # -- Grades 0-100 --
    _grades_standard: list[tuple[str, float, float]] = [
        ("symmetric", 50.0, 20.0), ("high-scoring", 75.0, 15.0), ("low-scoring", 35.0, 20.0),
        ("ceiling-heavy", 88.0, 10.0), ("floor-heavy", 12.0, 10.0),
    ]
    if suite in ("expanded", "extreme"):
        _grades_standard += [
            ("very-high", 92.0, 7.0), ("very-low", 8.0, 7.0), ("high-variance", 50.0, 34.0),
        ]

    for shape_label, mu_g, s0_g in _grades_standard:
        for f in run_noise_fracs:
            sigma_input_g = float(np.sqrt(max(1.0 - f, 0.0))) * s0_g
            sigma_run_g = float(np.sqrt(f)) * s0_g
            m_, si_, sr_ = mu_g, sigma_input_g, sigma_run_g

            def _gen_grades(
                rng: np.random.Generator, n: int, runs: int,
                _m: float = m_, _si: float = si_, _sr: float = sr_, _hetero: bool = heteroscedastic,
            ) -> np.ndarray:
                latent_i = rng.normal(_m, _si, size=(n, 1)) if _si > 0.0 else np.full((n, 1), _m)
                if _sr > 0.0:
                    if _hetero:
                        p_i = np.clip(latent_i / 100.0, 0.0, 1.0)
                        sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                        noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                    else:
                        noise = rng.normal(0.0, _sr, size=(n, runs))
                else:
                    noise = np.zeros((n, runs))
                return np.clip(latent_i + noise, 0.0, 100.0)

            true_mean = _estimate_true_mean_mc_runs(_gen_grades)
            _add(f"{shape_label}|f={f:.2f}", "grades", _gen_grades, true_mean, f)

    # Grades mixture (always included)
    for f in run_noise_fracs:
        rng_tmp = np.random.default_rng(4)
        flags_tmp = rng_tmp.choice(3, size=200_000, p=[0.20, 0.50, 0.30])
        base_tmp = np.empty(200_000, dtype=float)
        for bucket, mu_b, sig_b in [(0, 22.0, 11.0), (1, 58.0, 14.0), (2, 88.0, 8.0)]:
            msk = flags_tmp == bucket
            base_tmp[msk] = rng_tmp.normal(mu_b, sig_b, size=int(np.sum(msk)))
        var_mix = float(np.var(np.clip(base_tmp, 0.0, 100.0)))
        sigma_run_m = float(np.sqrt(var_mix * f / max(1.0 - f, 1e-9)))
        sr_ = sigma_run_m

        def _gen_grade_mix(
            rng: np.random.Generator, n: int, runs: int, _sr: float = sr_, _hetero: bool = heteroscedastic,
        ) -> np.ndarray:
            flags = rng.choice(3, size=(n, 1), p=[0.20, 0.50, 0.30])
            mu_i = np.where(flags == 0, 22.0, np.where(flags == 1, 58.0, 88.0)).astype(float)
            sig_i = np.where(flags == 0, 11.0, np.where(flags == 1, 14.0, 8.0)).astype(float)
            base_i = np.clip(mu_i + rng.normal(0.0, 1.0, size=(n, 1)) * sig_i, 0.0, 100.0)
            if _sr > 0.0:
                if _hetero:
                    p_i = np.clip(base_i / 100.0, 0.0, 1.0)
                    sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                    noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                else:
                    noise = rng.normal(0.0, _sr, size=(n, runs))
            else:
                noise = np.zeros((n, runs))
            return np.clip(base_i + noise, 0.0, 100.0)

        true_mean = _estimate_true_mean_mc_runs(_gen_grade_mix)
        _add(f"mixture-truncnorm(3 components)|f={f:.2f}", "grades", _gen_grade_mix, true_mean, f)

    # Heavy-tail t(df=3) grades scenario (always included)
    for f in run_noise_fracs:
        rng_tmp = np.random.default_rng(5)
        base_ht = np.clip(52.0 + 16.0 * rng_tmp.standard_t(df=3.0, size=200_000), 0.0, 100.0)
        var_ht = float(np.var(base_ht))
        sigma_run_h = float(np.sqrt(var_ht * f / max(1.0 - f, 1e-9)))
        sr_ = sigma_run_h

        def _gen_grade_heavy_tail(
            rng: np.random.Generator, n: int, runs: int, _sr: float = sr_, _hetero: bool = heteroscedastic,
        ) -> np.ndarray:
            base_i = np.clip(52.0 + 16.0 * rng.standard_t(df=3.0, size=(n, 1)), 0.0, 100.0)
            if _sr > 0.0:
                if _hetero:
                    p_i = np.clip(base_i / 100.0, 0.0, 1.0)
                    sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                    noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                else:
                    noise = rng.normal(0.0, _sr, size=(n, runs))
            else:
                noise = np.zeros((n, runs))
            return np.clip(base_i + noise, 0.0, 100.0)

        true_mean = _estimate_true_mean_mc_runs(_gen_grade_heavy_tail)
        _add(f"heavy-tail t(df=3)|f={f:.2f}", "grades", _gen_grade_heavy_tail, true_mean, f)

    # Expanded-suite floor/ceiling spiked grades scenarios
    if suite in ("expanded", "extreme"):
        for spike_name, spike_value, body_mu, body_sigma in [
            ("zero-spiked(pi=0.40,N(45,20))", 0.0, 45.0, 20.0),
            ("hundred-spiked(pi=0.40,N(65,18))", 100.0, 65.0, 18.0),
        ]:
            rng_tmp = np.random.default_rng(6)
            spike_mask = rng_tmp.random(200_000) < 0.40
            body = np.clip(rng_tmp.normal(body_mu, body_sigma, size=200_000), 0.0, 100.0)
            base_sp = np.where(spike_mask, spike_value, body)
            var_sp = float(np.var(base_sp))

            for f in run_noise_fracs:
                sigma_run_sp = float(np.sqrt(var_sp * f / max(1.0 - f, 1e-9)))
                sv_, bm_, bs_, sr_ = spike_value, body_mu, body_sigma, sigma_run_sp

                def _gen_grade_spiked(
                    rng: np.random.Generator, n: int, runs: int,
                    _sv: float = sv_, _bm: float = bm_, _bs: float = bs_, _sr: float = sr_,
                    _hetero: bool = heteroscedastic,
                ) -> np.ndarray:
                    spike_i = rng.random((n, 1)) < 0.40
                    body_i = np.clip(rng.normal(_bm, _bs, size=(n, 1)), 0.0, 100.0)
                    base_i = np.where(spike_i, _sv, body_i)
                    if _sr > 0.0:
                        if _hetero:
                            p_i = np.clip(base_i / 100.0, 0.0, 1.0)
                            sigma_i = _sr * 2.0 * np.sqrt(p_i * (1.0 - p_i))
                            noise = rng.normal(0.0, 1.0, size=(n, runs)) * sigma_i
                        else:
                            noise = rng.normal(0.0, _sr, size=(n, runs))
                    else:
                        noise = np.zeros((n, runs))
                    return np.clip(base_i + noise, 0.0, 100.0)

                true_mean = _estimate_true_mean_mc_runs(_gen_grade_spiked)
                _add(f"{spike_name}|f={f:.2f}", "grades", _gen_grade_spiked, true_mean, f)

    return sources


# ---------------------------------------------------------------------------
# Multi-run (nested-mode) paired-difference sources
# ---------------------------------------------------------------------------
# Ported from sim_compare_boot_nested.py's build_pair_multirun_scenarios().
# Used by cases/ci_paired.py's --nested-mode. Unlike build_pair_sources
# (which shares a single between-item draw for both A and B), these use a
# Gaussian-copula correlation (cross_item_rho) between A's and B's item-level
# latent scores -- a more realistic "partial agreement" DGP -- and let A and B
# have independently swept run-noise fractions (f_A, f_B).


def _legacy_build_pair_multirun_sources(
    # UNUSED -- kept for reference only. Pre-unification implementation,
    # superseded by build_pair_sources' run_noise_fracs/cross_item_rho/
    # pairwise_noise_grid params (which route through sample_group_truth's
    # base_corr/per-group-icc generalization instead of a separate bespoke
    # per-shape generator catalog).
    run_noise_fracs: list[float],
    suite: str = "standard",
    cohens_d_values: list[float] | None = None,
    include_null: bool = False,
    heteroscedastic: bool = False,
    pairwise_noise_grid: bool = False,
    pairwise_noise_grid_max: int | None = None,
    pairwise_noise_grid_seed: int = 42,
    cross_item_rho: float = 0.7,
) -> list[CIPairSource]:
    if cohens_d_values is None:
        cohens_d_values = [0.3]

    d_list: list[float] = []
    if include_null:
        d_list.append(0.0)
    d_list.extend(d for d in cohens_d_values if d > 0.0)

    if pairwise_noise_grid:
        noise_pairs = [(float(fa), float(fb)) for fa in run_noise_fracs for fb in run_noise_fracs]
    else:
        noise_pairs = [(float(f), float(f)) for f in run_noise_fracs]

    if pairwise_noise_grid_max is not None and pairwise_noise_grid_max > 0 and len(noise_pairs) > pairwise_noise_grid_max:
        rng_grid = np.random.default_rng(pairwise_noise_grid_seed)
        keep_idx = np.sort(rng_grid.choice(len(noise_pairs), size=pairwise_noise_grid_max, replace=False))
        noise_pairs = [noise_pairs[int(i)] for i in keep_idx]

    sources: list[CIPairSource] = []

    def _add(label, eval_type, gen_pair, true_diff, f_a, f_b, icc, is_null) -> None:
        sources.append(CIPairSource(
            label=label, eval_type=eval_type, true_diff=true_diff, generate_pair=gen_pair,
            icc=icc, is_null=is_null, run_noise_frac=0.5 * (f_a + f_b),
            run_noise_frac_a=f_a, run_noise_frac_b=f_b,
        ))

    # -- Binary --
    binary_shapes: list[tuple[str, float]] = [
        ("binary-balanced", 0.5), ("binary-high", 0.8), ("binary-low", 0.2),
    ]
    if suite in ("expanded", "extreme"):
        binary_shapes += [("binary-rare", 0.05), ("binary-near-ceil", 0.93)]

    for shape_label, p0 in binary_shapes:
        total_std = float(np.sqrt(p0 * (1.0 - p0)))
        for f_a, f_b in noise_pairs:
            conc_a = float(max(f_a, 1e-6)) / float(max(1.0 - f_a, 1e-6))
            conc_b = float(max(f_b, 1e-6)) / float(max(1.0 - f_b, 1e-6))
            icc_a = 1.0 / (conc_a + 1.0)
            icc_b = 1.0 / (conc_b + 1.0)
            icc = 0.5 * (icc_a + icc_b)
            for d in d_list:
                delta = d * total_std
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = (
                    f"{shape_label}|fA={f_a:.2f}|fB={f_b:.2f}|{effect_tag}" if pairwise_noise_grid
                    else f"{shape_label}|f={f_a:.2f}|{effect_tag}"
                )
                p_, ca_, cb_, delta_, rho_ = p0, conc_a, conc_b, delta, cross_item_rho

                def _gen_bin_pair(
                    rng: np.random.Generator, n: int, runs: int,
                    _p: float = p_, _ca: float = ca_, _cb: float = cb_, _d: float = delta_, _rho: float = rho_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    z_shared = rng.normal(size=(n, 1))
                    z_indep = rng.normal(size=(n, 1))
                    q_a = np.clip(norm.cdf(z_shared), 1e-9, 1.0 - 1e-9)
                    q_b = np.clip(norm.cdf(_rho * z_shared + np.sqrt(1.0 - _rho ** 2) * z_indep), 1e-9, 1.0 - 1e-9)
                    p_a = stats.beta.ppf(q_a, _ca * _p, _ca * (1.0 - _p))
                    p_b0 = stats.beta.ppf(q_b, _cb * _p, _cb * (1.0 - _p))
                    p_b = np.clip(p_b0 + _d, 0.0, 1.0)
                    a = rng.binomial(1, p_a, size=(n, runs)).astype(float)
                    b = rng.binomial(1, p_b, size=(n, runs)).astype(float)
                    return a, b

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_bin_pair)
                _add(label, "binary", _gen_bin_pair, true_diff, f_a, f_b, icc, is_null)

    # Explicit stress-test binary regimes with highly one-sided discordance.
    if suite in ("expanded", "extreme"):
        asym_binary_specs: list[tuple[str, float, float, float]] = [
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

        for shape_label, p10, p01, p11 in asym_binary_specs:
            p00 = 1.0 - (p11 + p10 + p01)
            if p00 <= 0.0:
                raise ValueError(f"Invalid asymmetric binary scenario {shape_label}: probabilities sum to >= 1.0")

            probs = np.array([p11, p10, p01, p00], dtype=float)
            true_diff = float(p10 - p01)

            for f_a, f_b in noise_pairs:
                f_eff = 0.5 * (f_a + f_b)
                icc = 1.0 - f_eff
                label = (
                    f"{shape_label}|p10={p10:.3f}|p01={p01:.3f}|p11={p11:.3f}|p00={p00:.3f}|fA={f_a:.2f}|fB={f_b:.2f}"
                    if pairwise_noise_grid
                    else f"{shape_label}|p10={p10:.3f}|p01={p01:.3f}|p11={p11:.3f}|p00={p00:.3f}|f={f_eff:.2f}"
                )
                probs_, f_ = probs, f_eff

                def _gen_bin_pair_asym(
                    rng: np.random.Generator, n: int, runs: int, _probs: np.ndarray = probs_, _f: float = f_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    z_item = rng.choice(4, size=(n, 1), p=_probs)
                    z_run = rng.choice(4, size=(n, runs), p=_probs)
                    redraw = rng.random((n, runs)) < _f
                    z = np.where(redraw, z_run, z_item)
                    a = np.isin(z, (0, 1)).astype(float)
                    b = np.isin(z, (0, 2)).astype(float)
                    return a, b

                _add(label, "binary", _gen_bin_pair_asym, true_diff, f_a, f_b, icc, False)

    # -- Continuous [0, 1] --
    continuous_shapes: list[tuple[str, float, float]] = [
        ("cont-uniform", 1.0, 1.0), ("cont-right-skew", 2.0, 8.0), ("cont-left-skew", 8.0, 2.0),
    ]
    if suite in ("expanded", "extreme"):
        continuous_shapes += [("cont-moderate-skew", 2.0, 5.0), ("cont-boundary", 0.6, 0.6)]

    for shape_label, a_b, b_b in continuous_shapes:
        var_base = _beta_var(a_b, b_b)
        total_std = float(np.sqrt(var_base))
        for f_a, f_b in noise_pairs:
            sigma_run_a = float(np.sqrt(var_base * f_a / max(1.0 - f_a, 1e-9)))
            sigma_run_b = float(np.sqrt(var_base * f_b / max(1.0 - f_b, 1e-9)))
            icc_a = 1.0 / (1.0 + sigma_run_a ** 2 / max(var_base, 1e-12))
            icc_b = 1.0 / (1.0 + sigma_run_b ** 2 / max(var_base, 1e-12))
            icc = 0.5 * (icc_a + icc_b)
            for d in d_list:
                delta = d * total_std
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = (
                    f"{shape_label}|fA={f_a:.2f}|fB={f_b:.2f}|{effect_tag}" if pairwise_noise_grid
                    else f"{shape_label}|f={f_a:.2f}|{effect_tag}"
                )
                a_, b_, sra_, srb_, delta_, rho_ = a_b, b_b, sigma_run_a, sigma_run_b, delta, cross_item_rho

                def _gen_cont_pair(
                    rng: np.random.Generator, n: int, runs: int,
                    _a: float = a_, _b: float = b_, _sra: float = sra_, _srb: float = srb_, _d: float = delta_,
                    _hetero: bool = heteroscedastic, _rho: float = rho_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    z_shared = rng.normal(size=(n, 1))
                    z_indep = rng.normal(size=(n, 1))
                    q_a = np.clip(norm.cdf(z_shared), 1e-9, 1.0 - 1e-9)
                    q_b = np.clip(norm.cdf(_rho * z_shared + np.sqrt(1.0 - _rho ** 2) * z_indep), 1e-9, 1.0 - 1e-9)
                    base_a = stats.beta.ppf(q_a, _a, _b)
                    base_b = stats.beta.ppf(q_b, _a, _b)
                    if _sra > 0.0 or _srb > 0.0:
                        if _hetero:
                            noise_a = rng.normal(0.0, 1.0, size=(n, runs)) * (_sra * 2.0 * np.sqrt(base_a * (1.0 - base_a)))
                            noise_b = rng.normal(0.0, 1.0, size=(n, runs)) * (_srb * 2.0 * np.sqrt(base_b * (1.0 - base_b)))
                        else:
                            noise_a = rng.normal(0.0, _sra, size=(n, runs))
                            noise_b = rng.normal(0.0, _srb, size=(n, runs))
                    else:
                        noise_a = np.zeros((n, runs))
                        noise_b = np.zeros((n, runs))
                    a_sc = np.clip(base_a + noise_a, 0.0, 1.0)
                    b_sc = np.clip(base_b + _d + noise_b, 0.0, 1.0)
                    return a_sc, b_sc

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_cont_pair)
                _add(label, "continuous", _gen_cont_pair, true_diff, f_a, f_b, icc, is_null)

    # -- Likert 1-5 --
    _LIKERT_TOTAL_STD = 1.2
    likert_shapes: list[tuple[str, float]] = [
        ("likert-mid", 3.0), ("likert-low", 2.2), ("likert-high", 3.8),
    ]
    if suite in ("expanded", "extreme"):
        likert_shapes += [("likert-floor", 1.8), ("likert-ceil", 4.2)]

    for shape_label, mu_lat in likert_shapes:
        for f_a, f_b in noise_pairs:
            f_eff = 0.5 * (f_a + f_b)
            sigma_input_l = float(np.sqrt(max(1.0 - f_eff, 0.0))) * _LIKERT_TOTAL_STD
            sigma_run_a = float(np.sqrt(f_a)) * _LIKERT_TOTAL_STD
            sigma_run_b = float(np.sqrt(f_b)) * _LIKERT_TOTAL_STD
            icc = 1.0 - f_eff
            for d in d_list:
                delta = d * _LIKERT_TOTAL_STD
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = (
                    f"{shape_label}|fA={f_a:.2f}|fB={f_b:.2f}|{effect_tag}" if pairwise_noise_grid
                    else f"{shape_label}|f={f_a:.2f}|{effect_tag}"
                )
                m_, si_, sra_, srb_, delta_, rho_ = mu_lat, sigma_input_l, sigma_run_a, sigma_run_b, delta, cross_item_rho

                def _gen_likert_pair(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _si: float = si_, _sra: float = sra_, _srb: float = srb_, _d: float = delta_,
                    _hetero: bool = heteroscedastic, _rho: float = rho_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    if _si > 0.0:
                        z_shared = rng.normal(size=(n, 1))
                        z_indep = rng.normal(size=(n, 1))
                        base_a = _m + _si * z_shared
                        base_b = _m + _si * (_rho * z_shared + np.sqrt(1.0 - _rho ** 2) * z_indep)
                    else:
                        base_a = np.full((n, 1), _m)
                        base_b = np.full((n, 1), _m)
                    if _sra > 0.0 or _srb > 0.0:
                        if _hetero:
                            p_i_a = np.clip((base_a - 1.0) / 4.0, 0.0, 1.0)
                            p_i_b = np.clip((base_b - 1.0) / 4.0, 0.0, 1.0)
                            noise_a = rng.normal(0.0, 1.0, size=(n, runs)) * (_sra * 2.0 * np.sqrt(p_i_a * (1.0 - p_i_a)))
                            noise_b = rng.normal(0.0, 1.0, size=(n, runs)) * (_srb * 2.0 * np.sqrt(p_i_b * (1.0 - p_i_b)))
                        else:
                            noise_a = rng.normal(0.0, _sra, size=(n, runs))
                            noise_b = rng.normal(0.0, _srb, size=(n, runs))
                    else:
                        noise_a = np.zeros((n, runs))
                        noise_b = np.zeros((n, runs))
                    a_sc = np.rint(np.clip(base_a + noise_a, 1.0, 5.0))
                    b_sc = np.rint(np.clip(base_b + _d + noise_b, 1.0, 5.0))
                    return a_sc, b_sc

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_likert_pair)
                _add(label, "likert", _gen_likert_pair, true_diff, f_a, f_b, icc, is_null)

    # -- Grades 0-100 --
    _GRADES_TOTAL_STD = 20.0
    grades_shapes: list[tuple[str, float]] = [
        ("grades-mid", 55.0), ("grades-low", 35.0), ("grades-high", 78.0),
    ]
    if suite in ("expanded", "extreme"):
        grades_shapes += [("grades-ceiling", 86.0), ("grades-floor", 20.0)]

    for shape_label, mu_g in grades_shapes:
        for f_a, f_b in noise_pairs:
            f_eff = 0.5 * (f_a + f_b)
            sigma_input_g = float(np.sqrt(max(1.0 - f_eff, 0.0))) * _GRADES_TOTAL_STD
            sigma_run_a = float(np.sqrt(f_a)) * _GRADES_TOTAL_STD
            sigma_run_b = float(np.sqrt(f_b)) * _GRADES_TOTAL_STD
            icc = 1.0 - f_eff
            for d in d_list:
                delta = d * _GRADES_TOTAL_STD
                is_null = d == 0.0
                effect_tag = "null" if is_null else f"d={d:.2f}"
                label = (
                    f"{shape_label}|fA={f_a:.2f}|fB={f_b:.2f}|{effect_tag}" if pairwise_noise_grid
                    else f"{shape_label}|f={f_a:.2f}|{effect_tag}"
                )
                m_, si_, sra_, srb_, delta_, rho_ = mu_g, sigma_input_g, sigma_run_a, sigma_run_b, delta, cross_item_rho

                def _gen_grades_pair(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _si: float = si_, _sra: float = sra_, _srb: float = srb_, _d: float = delta_,
                    _hetero: bool = heteroscedastic, _rho: float = rho_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    if _si > 0.0:
                        z_shared = rng.normal(size=(n, 1))
                        z_indep = rng.normal(size=(n, 1))
                        base_a = _m + _si * z_shared
                        base_b = _m + _si * (_rho * z_shared + np.sqrt(1.0 - _rho ** 2) * z_indep)
                    else:
                        base_a = np.full((n, 1), _m)
                        base_b = np.full((n, 1), _m)
                    if _sra > 0.0 or _srb > 0.0:
                        if _hetero:
                            p_i_a = np.clip(base_a / 100.0, 0.0, 1.0)
                            p_i_b = np.clip(base_b / 100.0, 0.0, 1.0)
                            noise_a = rng.normal(0.0, 1.0, size=(n, runs)) * (_sra * 2.0 * np.sqrt(p_i_a * (1.0 - p_i_a)))
                            noise_b = rng.normal(0.0, 1.0, size=(n, runs)) * (_srb * 2.0 * np.sqrt(p_i_b * (1.0 - p_i_b)))
                        else:
                            noise_a = rng.normal(0.0, _sra, size=(n, runs))
                            noise_b = rng.normal(0.0, _srb, size=(n, runs))
                    else:
                        noise_a = np.zeros((n, runs))
                        noise_b = np.zeros((n, runs))
                    a_sc = np.clip(base_a + noise_a, 0.0, 100.0)
                    b_sc = np.clip(base_b + _d + noise_b, 0.0, 100.0)
                    return a_sc, b_sc

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_grades_pair)
                _add(label, "grades", _gen_grades_pair, true_diff, f_a, f_b, icc, is_null)

    return sources


# ---------------------------------------------------------------------------
# Multi-arm sources (for cases/pvalues.py's multi-arm multiplicity benchmark)
# ---------------------------------------------------------------------------
# Sweeps the SAME shape catalog (BINARY_SHAPES/CONTINUOUS_SHAPES/
# LIKERT_SHAPES/GRADES_SHAPES) build_pair_sources uses, generalized from k=2
# to k arms via sample_group_truth -- so multi-arm benchmarking gets the
# same shape-robustness as the pairwise comparison, not a separate, narrower
# bespoke catalog. "param" shapes only (custom shapes have no k-arm
# analogue). Arm 0 carries the alternative-hypothesis shift (cohens_d *
# group_total_std); arms 1..k-1 stay at the shared baseline.


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
        for shape in _tier_shapes(SHAPES_BY_EVAL_TYPE[eval_type], suite):
            if shape.kind != "param":
                continue
            sources.append(MultiArmSource(
                label=shape.label, eval_type=eval_type, generate_scores=_make_generator(shape),
                alt_delta=cohens_d * group_total_std(eval_type, shape.params, icc),
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


def _ppi_shape(eval_type: str) -> ShapeSpec:
    label = _PPI_REPRESENTATIVE_SHAPE_LABEL[eval_type]
    return next(s for s in SHAPES_BY_EVAL_TYPE[eval_type] if s.label == label)


def _ppi_shape_anchor(shape: ShapeSpec) -> float:
    """Mean of a 'param' ShapeSpec's truth distribution -- used both as the
    judge-bias model's mu_null and as the slope-distortion anchor point."""
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


def _jb_llm(
    truth: np.ndarray, bias: float, noise_sd: float, rng: np.random.Generator,
    slope: float = 1.0, anchor: float = 0.0,
) -> np.ndarray:
    pred = anchor + slope * (truth - anchor) + bias
    if noise_sd == 0.0:
        return pred
    return pred + rng.normal(0.0, noise_sd, len(truth))


def _jb_llm_repeated(
    truths: list[np.ndarray], biases: list[float], noise_sds: list[float], slopes: list[float],
    rng: np.random.Generator, anchor: float, corr: float,
) -> list[np.ndarray]:
    if len(truths) == 0:
        return []
    n = len(truths[0])
    if corr <= 0.0:
        return [_jb_llm(tr, b, s, rng, slope=m, anchor=anchor) for tr, b, s, m in zip(truths, biases, noise_sds, slopes)]

    corr_clamped = min(max(corr, 0.0), 0.999)
    w_shared = float(np.sqrt(corr_clamped))
    w_ind = float(np.sqrt(1.0 - corr_clamped))
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
    shape = _ppi_shape(sc.eval_type)
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
    llm_a2 = _jb_llm(truth_a2, bias_a, noise1, rng, slope=slope_a, anchor=anchor)
    llm_b2 = _jb_llm(truth_b2, bias_b, noise2, rng, slope=slope_b, anchor=anchor)
    lab_a2 = _jb_labels_independent(truth_a2, sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)
    lab_b2 = _jb_labels_independent(truth_b2, sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)

    # -- Paired data (wilcoxon) --
    truth_x, truth_y = _repeated(n1, 2, np.array([0.0, es]))
    llm_x, llm_y = _jb_llm_repeated(
        [truth_x, truth_y], [bias_a, bias_b], [noise1, noise2], [slope_a, slope_b],
        rng, anchor=anchor, corr=sc.repeated_corr,
    )
    lab_x, lab_y = _jb_labels_shared([truth_x, truth_y], sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)

    # -- Independent three-group data (anova_ind) --
    truth_a3 = _marginal(n1)
    truth_b3 = _marginal(n2) + es
    truth_c3 = _marginal(n3) + 2 * es
    llm_a3 = _jb_llm(truth_a3, bias_a, noise1, rng, slope=slope_a, anchor=anchor)
    llm_b3 = _jb_llm(truth_b3, bias_b, noise2, rng, slope=slope_b, anchor=anchor)
    llm_c3 = _jb_llm(truth_c3, bias_c, noise3, rng, slope=slope_c, anchor=anchor)
    lab_a3 = _jb_labels_independent(truth_a3, sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)
    lab_b3 = _jb_labels_independent(truth_b3, sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)
    lab_c3 = _jb_labels_independent(truth_c3, sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)

    # -- Repeated-measures three-group data (anova_rep, friedman, lmm) --
    truth_A, truth_B, truth_C = _repeated(n1, 3, np.array([0.0, es, 2 * es]))
    llm_A, llm_B, llm_C = _jb_llm_repeated(
        [truth_A, truth_B, truth_C], [bias_a, bias_b, bias_c], [noise1, noise2, noise3], [slope_a, slope_b, slope_c],
        rng, anchor=anchor, corr=sc.repeated_corr,
    )
    lab_A, lab_B, lab_C = _jb_labels_shared([truth_A, truth_B, truth_C], sc.label_frac, rng, mnar=sc.label_mnar, mnar_strength=sc.mnar_strength, mnar_mode=sc.mnar_mode)

    # -- 2x2 crossed fixed-factor data (lmm_factorial) --
    (bias_w, bias_x, bias_y, bias_z), (slope_w, slope_x, slope_y, slope_z) = _jb_judge_params_4(sc)
    truth_W, truth_X, truth_Y, truth_Z = _repeated(n1, 4, np.array([0.0, es, 2 * es, 3 * es]))
    llm_W, llm_X, llm_Y, llm_Z = _jb_llm_repeated(
        [truth_W, truth_X, truth_Y, truth_Z], [bias_w, bias_x, bias_y, bias_z],
        [noise1, noise2, noise3, noise1], [slope_w, slope_x, slope_y, slope_z],
        rng, anchor=anchor, corr=sc.repeated_corr,
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
    shape = _ppi_shape(sc.eval_type)
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
