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

from . import CISource, CIPairSource

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


def build_pair_sources(
    suite: str = "standard",
    icc_values: list[float] | tuple[float, ...] = (0.10, 0.25, 0.40),
    cohens_d_values: list[float] | tuple[float, ...] = (0.3,),
    include_null: bool = False,
) -> list[CIPairSource]:
    """Return canonical synthetic paired-difference CIPairSources, parameterised
    by ICC and Cohen's d.

    Data-generating processes are reparameterised so that the intraclass
    correlation (ICC = between-input variance / total variance) and the
    standardised effect size (Cohen's d = delta / total_std) are explicit
    inputs. This makes the sweep principled and reviewer-defensible.

    ICC definitions per eval type
    ------------------------------
    Binary    : Bernoulli-Beta hierarchical model. Per-input success probs
                p_i ~ Beta(conc*p0, conc*(1-p0)), giving ICC = 1/(conc+1).
    Continuous: Beta base distribution with ICC-derived Gaussian noise.
                ICC = Var(base) / (Var(base) + 2*noise_std^2).
    Likert    : Latent-normal model. ICC = base_std^2 / total_var,
                total_std fixed at _LIKERT_TOTAL_STD on the latent scale.
    Grades    : Same latent-normal structure; total_std = _GRADES_TOTAL_STD.

    Parameters
    ----------
    suite : str
        'standard', 'expanded', or 'extreme'.
    icc_values : sequence of float
        ICC values to sweep. Each value generates a separate scenario batch.
    cohens_d_values : sequence of float
        Non-null standardised effect sizes. A null (d=0) variant is
        automatically prepended when include_null=True.
    include_null : bool
        If True, prepend d=0 scenarios for every (eval_type, shape, icc)
        combination, flagged is_null=True (used to measure Type I error).
    """
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")

    sources: list[CIPairSource] = []

    icc_list = list(icc_values)
    d_list = list(cohens_d_values)
    if include_null:
        d_list = [0.0] + [d for d in d_list if d > 0.0]

    # Latent-scale standard deviations for Likert / Grades -- fix the total
    # marginal std for each score type so Cohen's d has a concrete, consistent
    # meaning across ICC levels.
    _LIKERT_TOTAL_STD = 1.2  # latent scale, maps to {1,...,5} after rounding
    _GRADES_TOTAL_STD = 20.0  # [0, 100] scale

    # ------------------------------------------------------------------
    # Binary: ICC = 1/(conc+1) <-> conc = 1/ICC - 1
    # total_std ~ sqrt(p0*(1-p0)); delta = d * total_std
    # ------------------------------------------------------------------
    binary_shapes: list[tuple[str, float]] = [
        ("binary-balanced", 0.5),
        ("binary-high", 0.8),
        ("binary-low", 0.2),
        ("binary-near-ceil", 0.92),
    ]
    if suite in ("expanded", "extreme"):
        binary_shapes += [
            ("binary-rare", 0.05),
            ("binary-near-ceil-hi", 0.95),
        ]

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

                def _gen_binary(
                    rng: np.random.Generator, n: int, runs: int,
                    _bp: float = bp_, _d: float = delta_, _c: float = conc_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    p_a = rng.beta(_bp * _c, (1.0 - _bp) * _c, size=(n, 1))
                    p_b = np.clip(p_a + _d, 0.0, 1.0)
                    a = rng.binomial(1, p_a, size=(n, runs)).astype(float)
                    b = rng.binomial(1, p_b, size=(n, runs)).astype(float)
                    return a, b

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_binary)
                sources.append(CIPairSource(
                    label=label, eval_type="binary",
                    generate_pair=_gen_binary, true_diff=true_diff,
                    icc=icc, cohens_d=d, is_null=is_null,
                ))

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
                raise ValueError(
                    f"Invalid asymmetric binary scenario {shape_label}: probabilities sum to >= 1.0"
                )

            probs = np.array([p11, p10, p01, p00], dtype=float)
            true_diff = float(p10 - p01)
            label = f"{shape_label}|p10={p10:.3f}|p01={p01:.3f}|p11={p11:.3f}|p00={p00:.3f}"

            def _gen_binary_asym(
                rng: np.random.Generator, n: int, runs: int, _probs: np.ndarray = probs,
            ) -> tuple[np.ndarray, np.ndarray]:
                z = rng.choice(4, size=(n, runs), p=_probs)
                a = np.isin(z, (0, 1)).astype(float)
                b = np.isin(z, (0, 2)).astype(float)
                return a, b

            sources.append(CIPairSource(
                label=label, eval_type="binary",
                generate_pair=_gen_binary_asym, true_diff=true_diff,
                icc=0.0, cohens_d=0.0, is_null=False,
            ))

    # ------------------------------------------------------------------
    # Continuous [0, 1]: Beta(a, b) base + ICC-derived Gaussian noise.
    # ICC = Var(base) / (Var(base) + 2*noise_std^2)
    #   -> noise_std = sqrt(Var(base) * (1/ICC - 1) / 2)
    # total_std ~ sqrt(Var(base)/ICC); delta = d * total_std.
    # ------------------------------------------------------------------
    continuous_shapes: list[tuple[str, float, float]] = [
        ("cont-uniform", 1.0, 1.0),
        ("cont-right-skew", 2.0, 8.0),
        ("cont-left-skew", 8.0, 2.0),
    ]
    if suite in ("expanded", "extreme"):
        continuous_shapes += [
            ("cont-moderate-skew", 2.0, 5.0),
            ("cont-boundary", 0.6, 0.6),
        ]

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

                def _gen_continuous(
                    rng: np.random.Generator, n: int, runs: int,
                    _a: float = a_, _b: float = b_, _ns: float = ns_, _d: float = delta_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    base = rng.beta(_a, _b, size=(n, 1))
                    shared = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_a = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_b = rng.normal(0.0, _ns, size=(n, runs))
                    a_vals = np.clip(base + shared + indiv_a, 0.0, 1.0)
                    b_vals = np.clip(base + _d + shared + indiv_b, 0.0, 1.0)
                    return a_vals, b_vals

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_continuous)
                sources.append(CIPairSource(
                    label=label, eval_type="continuous",
                    generate_pair=_gen_continuous, true_diff=true_diff,
                    icc=icc, cohens_d=d, is_null=is_null,
                ))

    # ------------------------------------------------------------------
    # Likert 1-5: latent-normal model rounded to {1,...,5}.
    # ICC = base_std^2 / total_var with total_var = _LIKERT_TOTAL_STD^2.
    # ------------------------------------------------------------------
    likert_shapes: list[tuple[str, float]] = [
        ("likert-mid", 3.0),
        ("likert-low", 2.2),
        ("likert-high", 3.8),
    ]
    if suite in ("expanded", "extreme"):
        likert_shapes += [
            ("likert-polarized", 3.0),
            ("likert-floor", 1.8),
        ]

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

                def _gen_likert(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _bs: float = bs_, _ns: float = ns_, _d: float = delta_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    base = rng.normal(_m, _bs, size=(n, 1))
                    shared = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_a = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_b = rng.normal(0.0, _ns, size=(n, runs))
                    a_vals = np.rint(np.clip(base + shared + indiv_a, 1.0, 5.0))
                    b_vals = np.rint(np.clip(base + _d + shared + indiv_b, 1.0, 5.0))
                    return a_vals, b_vals

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_likert)
                sources.append(CIPairSource(
                    label=label, eval_type="likert",
                    generate_pair=_gen_likert, true_diff=true_diff,
                    icc=icc, cohens_d=d, is_null=is_null,
                ))

    # ------------------------------------------------------------------
    # Grades 0-100: same latent-normal structure; total_std = _GRADES_TOTAL_STD.
    # ------------------------------------------------------------------
    grades_shapes: list[tuple[str, float]] = [
        ("grades-mid", 55.0),
        ("grades-low", 35.0),
        ("grades-high", 78.0),
    ]
    if suite in ("expanded", "extreme"):
        grades_shapes += [
            ("grades-ceiling", 86.0),
            ("grades-floor", 20.0),
        ]

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

                def _gen_grades(
                    rng: np.random.Generator, n: int, runs: int,
                    _m: float = m_, _bs: float = bs_, _ns: float = ns_, _d: float = delta_,
                ) -> tuple[np.ndarray, np.ndarray]:
                    base = rng.normal(_m, _bs, size=(n, 1))
                    shared = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_a = rng.normal(0.0, _ns, size=(n, runs))
                    indiv_b = rng.normal(0.0, _ns, size=(n, runs))
                    a_vals = np.clip(base + shared + indiv_a, 0.0, 100.0)
                    b_vals = np.clip(base + _d + shared + indiv_b, 0.0, 100.0)
                    return a_vals, b_vals

                true_diff = 0.0 if is_null else _estimate_true_pair_diff(_gen_grades)
                sources.append(CIPairSource(
                    label=label, eval_type="grades",
                    generate_pair=_gen_grades, true_diff=true_diff,
                    icc=icc, cohens_d=d, is_null=is_null,
                ))

    return sources
