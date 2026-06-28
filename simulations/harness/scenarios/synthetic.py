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

from typing import Callable

import numpy as np

from . import CISource

SCENARIO_SUITES = ["standard", "expanded", "extreme"]


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


def build_single_sample_sources(suite: str = "standard") -> list[CISource]:
    """Return canonical synthetic single-sample CISources across the four eval types."""
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

    # logit-normal is a principled model for bounded continuous scores
    # (common in LLM rubric outputs); always included in standard suite.
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

    # Zero-inflated and one-inflated -- point-mass spike at the boundary mixed
    # with a Beta component. Models extreme low/high LLM performance: a model
    # that almost always fails produces a spike at 0; a near-ceiling model
    # produces a spike at 1. True means are computed analytically.
    def _gen_zero_inflated(rng: np.random.Generator, n: int) -> np.ndarray:
        spike = rng.random(n) < 0.70
        return np.where(spike, 0.0, rng.beta(2.0, 4.0, n))

    sources.append(
        CISource(
            label="zero-inflated(pi=0.70,Beta(2,4))",
            eval_type="continuous",
            generate=_gen_zero_inflated,
            # E[X] = 0.70*0 + 0.30*Beta_mean = 0.30 * 2/(2+4)
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
            # E[X] = 0.70*1 + 0.30*Beta_mean = 0.70 + 0.30 * 4/(4+2)
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

    # ------------------------------------------------------------------
    # Likert 1-5: latent-normal model
    # ------------------------------------------------------------------
    # Each generator draws from N(mu_lat, sigma_lat) on a latent scale, then
    # rounds to integers in {1,...,5} via rint+clip. True means are estimated
    # via MC (n_mc=500,000; SE < 0.001).

    def _make_likert_normal(mu: float, sigma: float) -> Callable[[np.random.Generator, int], np.ndarray]:
        def _gen(rng: np.random.Generator, n: int, _m: float = mu, _s: float = sigma) -> np.ndarray:
            return np.clip(np.rint(rng.normal(_m, _s, n)), 1.0, 5.0)
        return _gen

    def _make_likert_bimodal(mu1: float, mu2: float, sigma: float) -> Callable[[np.random.Generator, int], np.ndarray]:
        def _gen(rng: np.random.Generator, n: int, _m1: float = mu1, _m2: float = mu2, _s: float = sigma) -> np.ndarray:
            sel = rng.random(n) < 0.5
            latents = np.where(sel, rng.normal(_m1, _s, n), rng.normal(_m2, _s, n))
            return np.clip(np.rint(latents), 1.0, 5.0)
        return _gen

    # Standard suite: five shapes covering the main families
    _likert_standard = [
        ("uniform", _make_likert_normal(3.0, 2.0)),  # high variance -> flat
        ("skewed-low", _make_likert_normal(2.0, 1.1)),  # mass at 1-2
        ("skewed-high", _make_likert_normal(4.0, 1.1)),  # mass at 4-5
        ("bimodal", _make_likert_bimodal(1.5, 4.5, 0.65)),  # peaks at extremes
        ("center-peaked", _make_likert_normal(3.0, 0.55)),  # sharp peak at 3
    ]
    for label, gen in _likert_standard:
        sources.append(
            CISource(
                label=label,
                eval_type="likert",
                generate=gen,
                true_mean=_estimate_true_mean_mc(gen),
            )
        )

    if suite in ("expanded", "extreme"):
        _likert_expanded = [
            ("near-floor", _make_likert_normal(1.5, 0.65)),  # mostly 1s
            ("near-ceiling", _make_likert_normal(4.5, 0.65)),  # mostly 5s
            ("polarized", _make_likert_bimodal(1.3, 4.7, 0.50)),  # extreme bimodal
            ("flat-middle", _make_likert_normal(3.0, 1.4)),  # moderate spread
        ]
        for label, gen in _likert_expanded:
            sources.append(
                CISource(
                    label=label,
                    eval_type="likert",
                    generate=gen,
                    true_mean=_estimate_true_mean_mc(gen),
                )
            )

    # ------------------------------------------------------------------
    # Grades 0-100: truncated normals of varying centre and spread
    # ------------------------------------------------------------------
    grade_specs = [
        ("symmetric", 50, 20),  # centred, moderate spread
        ("high-scoring", 75, 15),  # near ceiling
        ("low-scoring", 35, 20),  # near floor
        ("ceiling-heavy", 88, 10),  # mass near 100 -- heavy clipping
        ("floor-heavy", 12, 10),  # mass near 0 -- heavy clipping
    ]
    if suite in ("expanded", "extreme"):
        grade_specs.extend(
            [
                ("very-high", 92, 7),
                ("very-low", 8, 7),
                ("high-variance", 50, 34),
            ]
        )

    for label, mu, sigma in grade_specs:
        mu_, sigma_ = mu, sigma
        true_mean_ = _true_mean_clipped_normal(mu_, sigma_)
        sources.append(
            CISource(
                label=f"{label} N({mu_},{sigma_})",
                eval_type="grades",
                generate=lambda rng, n, _m=mu_, _s=sigma_: np.clip(
                    rng.normal(_m, _s, n), 0.0, 100.0
                ),
                true_mean=true_mean_,
            )
        )

    # Mixture of 3 normal components -- captures bimodal/trimodal grade
    # distributions common in LLM evals (fail / partial / full credit).
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

    # Heavy-tailed t(df=3) -- models grade distributions with occasional
    # outlier runs (model crashes, OOD inputs, etc.).
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
        # Zero-spiked and hundred-spiked grades -- point-mass spike at floor/ceiling
        # mixed with a truncated-normal body. Models complete failure (spike at 0)
        # and near-perfect performance (spike at 100) common in LLM coding/math evals.
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
