"""Front-facing API for factorial LMM analysis on long-form DataFrames.

Use :func:`analyze_factorial` when you have post-hoc tagged outputs (e.g.
a RAG pipeline where each output is labelled with the chunker and retrieval
method used) or a designed factorial experiment, and want to fit a mixed
model of the form::

    score ~ C(F1) * C(F2) * ... + (1 | random_effect)

The function accepts a plain :class:`pandas.DataFrame`, builds the internal
:class:`~promptstats.core.types.BenchmarkResult` automatically, runs the
statsmodels factorial LMM, and returns a fully populated
:class:`~promptstats.core.router.AnalysisBundle`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def analyze_factorial(
    data: "pd.DataFrame",
    factors: list[str],
    random_effect: str = "input_id",
    score_col: str = "score",
    *,
    ci: float = 0.95,
    correction: str = "holm",
    reference: str = "grand_mean",
    spread_percentiles: tuple[float, float] = (10, 90),
    failure_threshold: Optional[float] = None,
    n_sim: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> "AnalysisBundle":
    """Run factorial LMM analysis on a long-form DataFrame.

    Fits the mixed model::

        score ~ C(F1) * C(F2) * ... + (1 | random_effect)

    where ``F1, F2, ...`` are the factor columns specified in *factors*.
    The random effect (e.g. question ID, input document) absorbs
    between-input variation, producing cleaner estimates of each factor
    combination's performance and their main effects / interactions.

    This is the recommended entry point for:

    * **Post-hoc tagged pipelines** — e.g. RAG experiments where each
      output row records the ``chunker`` and ``retrieval_method`` used.
    * **Designed factorial experiments** — e.g. prompt templates that vary
      ``persona`` and ``few_shots`` simultaneously; see also
      :class:`~promptstats.core.types.BenchmarkResult` with
      ``template_factors``.

    Parameters
    ----------
    data : pd.DataFrame
        Long-form DataFrame.  Must contain:

        * One column per factor name in *factors*.
        * *random_effect* column — a unique identifier for each benchmark
          input (e.g. question ID).  At least 2 distinct values required.
        * *score_col* column — numeric evaluation score.

        Multiple rows with the same ``(random_effect, factor_combo)`` key
        are averaged (cell means) before fitting the LMM.

    factors : list[str]
        Column names of the fixed-effect factors.  Each must be a valid
        Python identifier and have at least 2 unique values in *data*.

    random_effect : str
        Column that identifies benchmark inputs (default ``"input_id"``).

    score_col : str
        Column containing the numeric score (default ``"score"``).

    ci : float
        Confidence level for Wald intervals (default 0.95).

    correction : str
        Multiple-comparisons correction for pairwise tests:
        ``'holm'`` (default), ``'bonferroni'``, ``'fdr_bh'``, or ``'none'``.

    reference : str
        Reference for mean advantage: ``'grand_mean'`` (default) or the
        label of a specific factor-combination cell (e.g. ``'fixed_512|bm25'``
        when factors are ``['chunker', 'retrieval']``).

    spread_percentiles : tuple[float, float]
        Percentile bands for the point-advantage spread (default ``(10, 90)``).

    failure_threshold : float, optional
        Score threshold below which an observation counts as a failure
        for robustness metrics.

    n_sim : int
        Monte Carlo simulations for the rank distribution (default 10 000).

    rng : np.random.Generator, optional
        Random-number generator for reproducibility.

    Returns
    -------
    AnalysisBundle
        All the usual fields (``pairwise``, ``point_advantage``,
        ``robustness``, ``rank_dist``) are populated.
        ``bundle.factorial_lmm_info`` additionally contains:

        * ``factor_tests`` — Wald χ² tests per main effect and interaction.
        * ``marginal_means`` — estimated marginal means per factor.
        * ``icc``, ``sigma_input``, ``sigma_resid`` — variance components.

    Raises
    ------
    TypeError
        If *data* is not a :class:`pandas.DataFrame`.
    ValueError
        If required columns are missing, factor names are not valid Python
        identifiers, or any factor has fewer than 2 unique levels.

    Examples
    --------
    RAG pipeline with two factors (chunker × retrieval method):

    >>> import pandas as pd
    >>> import promptstats as ps
    >>> data = pd.DataFrame([
    ...     {"input_id": "q1", "chunker": "fixed_512", "retrieval": "bm25",  "score": 0.72},
    ...     {"input_id": "q1", "chunker": "fixed_512", "retrieval": "dense", "score": 0.85},
    ...     {"input_id": "q1", "chunker": "semantic",  "retrieval": "bm25",  "score": 0.78},
    ...     {"input_id": "q1", "chunker": "semantic",  "retrieval": "dense", "score": 0.91},
    ...     {"input_id": "q2", "chunker": "fixed_512", "retrieval": "bm25",  "score": 0.61},
    ...     {"input_id": "q2", "chunker": "fixed_512", "retrieval": "dense", "score": 0.74},
    ...     {"input_id": "q2", "chunker": "semantic",  "retrieval": "bm25",  "score": 0.65},
    ...     {"input_id": "q2", "chunker": "semantic",  "retrieval": "dense", "score": 0.82},
    ... ])
    >>> bundle = ps.analyze_factorial(data, factors=["chunker", "retrieval"])
    >>> print(bundle.factorial_lmm_info.factor_tests)
    """
    import pandas as pd

    from .core.types import BenchmarkResult
    from .core.router import AnalysisBundle, BenchmarkShape
    from .core.mixed_effects import _lmm_analyze_factorial_sm

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame; got {type(data).__name__}."
        )

    if not factors:
        raise ValueError("factors must be a non-empty list of column names.")

    missing = [c for c in [*factors, random_effect, score_col] if c not in data.columns]
    if missing:
        raise ValueError(
            f"Columns not found in data: {missing}. "
            f"Available columns: {list(data.columns)}."
        )

    for factor in factors:
        if not str(factor).isidentifier():
            raise ValueError(
                f"Factor name '{factor}' is not a valid Python identifier. "
                "Rename it (e.g., replace spaces with underscores) so that "
                "it can be used in model formulas."
            )
        n_unique = data[factor].nunique(dropna=True)
        if n_unique < 2:
            raise ValueError(
                f"Factor '{factor}' has {n_unique} unique level(s). "
                "Each factor must have at least 2 distinct levels."
            )

    n_inputs = data[random_effect].nunique(dropna=True)
    if n_inputs < 2:
        raise ValueError(
            f"random_effect column '{random_effect}' has {n_inputs} unique value(s). "
            "At least 2 distinct inputs are required to fit the random intercept."
        )

    # ------------------------------------------------------------------
    # Build unique factor-combination → template label mapping
    # Template label = "v1|v2|..." (sorted by factor values).
    # Note: factor values must not contain the separator character "|".
    # ------------------------------------------------------------------
    _SEP = "|"

    combos = (
        data[factors]
        .drop_duplicates()
        .sort_values(factors)
        .reset_index(drop=True)
    )
    template_labels: list[str] = [
        _SEP.join(str(row[f]) for f in factors)
        for _, row in combos.iterrows()
    ]

    if len(set(template_labels)) < len(template_labels):
        raise ValueError(
            "Some factor-level combinations produce identical template labels "
            f"when joined with the separator '{_SEP}'. This can happen when "
            f"factor values themselves contain '{_SEP}'. "
            f"Ensure factor values do not contain '{_SEP}'."
        )

    # template_factors_df: rows aligned with template_labels
    template_factors_df = combos.copy()

    # ------------------------------------------------------------------
    # Normalise the random-effect column to strings so that pivot and
    # BenchmarkResult.input_labels are consistent.
    # ------------------------------------------------------------------
    data_work = data.copy()
    data_work["_ps_input"] = data_work[random_effect].astype(str)
    data_work["_ps_template"] = data_work[factors].apply(
        lambda row: _SEP.join(str(row[f]) for f in factors), axis=1
    )

    input_labels: list[str] = sorted(
        data_work["_ps_input"].dropna().unique().tolist()
    )

    # ------------------------------------------------------------------
    # Pivot to (N_templates × M_inputs) cell-mean matrix (NaN = missing)
    # ------------------------------------------------------------------
    pivot = data_work.pivot_table(
        index="_ps_input",
        columns="_ps_template",
        values=score_col,
        aggfunc="mean",
        observed=True,
    )
    # Reindex to guarantee consistent ordering with template_labels / input_labels.
    pivot = pivot.reindex(index=input_labels, columns=template_labels)
    cell_means_2d = pivot.to_numpy().T  # (N_templates, M_inputs)

    # ------------------------------------------------------------------
    # Build BenchmarkResult (NaN-safe — missing cells are supported)
    # ------------------------------------------------------------------
    if rng is None:
        rng = np.random.default_rng()

    benchmark = BenchmarkResult(
        scores=cell_means_2d,
        template_labels=template_labels,
        input_labels=input_labels,
        template_factors=template_factors_df,
    )

    # ------------------------------------------------------------------
    # Factorial LMM analysis (statsmodels backend only)
    # ------------------------------------------------------------------
    pairwise, mean_adv, rank_dist, robustness, seed_var, lmm_info = (
        _lmm_analyze_factorial_sm(
            benchmark,
            reference=reference,
            ci=ci,
            correction=correction,
            spread_percentiles=spread_percentiles,
            failure_threshold=failure_threshold,
            n_sim=n_sim,
            rng=rng,
        )
    )

    shape = BenchmarkShape(
        n_models=1,
        n_prompts=benchmark.n_templates,
        n_input_vars=1,
        n_evaluators=1,
        n_runs=1,
    )

    return AnalysisBundle(
        benchmark=benchmark,
        shape=shape,
        pairwise=pairwise,
        point_advantage=mean_adv,
        robustness=robustness,
        rank_dist=rank_dist,
        seed_variance=seed_var,
        factorial_lmm_info=lmm_info,
    )
