"""High-level comparison API: compare(), ComparisonResult, and thin wrappers.

Provides the new spec API on top of the existing statistical engine:

    evaldata = load_from(df)
    result   = compare(evaldata, factors="model", metric="score")
    result.summary()
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from evalstats.loader import EvalResults, EvalLoadError, load_from, _scores_dict_to_df, _is_nested_scores_dict
from evalstats.io import from_dataframe
from evalstats.core.router import analyze, analyze_factorial
from evalstats.core.bundles import AnalysisBundle, MultiModelBundle, AnalysisResult
from evalstats.core.summary import print_analysis_summary, print_brief_summary


# ─────────────────────────────────────────────────────────────────────────────
# ComparisonResult
# ─────────────────────────────────────────────────────────────────────────────

class ComparisonResult:
    """Statistical comparison results returned by :func:`compare`.

    Wraps the underlying :class:`~evalstats.core.bundles.AnalysisBundle` or
    :class:`~evalstats.core.bundles.MultiModelBundle` with the new spec API.

    Call :meth:`summary` to print the full terminal output (with gradient CI
    plots), :meth:`to_frame` to get DataFrames for downstream work, or
    :meth:`to_dict` for a JSON-friendly representation.
    """

    def __init__(
        self,
        analysis: AnalysisResult,
        *,
        factors: Union[str, list[str]],
        metric: str,
        baseline: Optional[str],
        alpha: float,
        filtered_df: pd.DataFrame,
        _mmb_view: Literal["model_level", "template_level", "cross_model"] = "model_level",
    ):
        self._analysis = analysis
        self._factors = factors
        self._metric = metric
        self._baseline = baseline
        self._alpha = alpha
        self._df = filtered_df
        self._mmb_view = _mmb_view  # which MultiModelBundle view is primary

    # ── print methods ────────────────────────────────────────────────────────

    def summary(
        self,
        *,
        top_pairwise: Optional[int] = None,
        style: Literal["gradient", "line"] = "gradient",
    ) -> None:
        """Print the full terminal summary with gradient CI plots.

        This delegates directly to the existing ``print_analysis_summary``
        which produces the gradient multi-band CI plots that are the
        signature output of evalstats.

        Parameters
        ----------
        top_pairwise : int, optional
            Number of pairwise comparisons to show. None shows all.
        style : {"gradient", "line"}
            CI plot style. ``"gradient"`` (default) renders multi-band opacity
            plots; ``"line"`` uses the classic dot-and-line style.
        """
        print_analysis_summary(self._analysis, top_pairwise=top_pairwise, style=style)

    def print(self, **kwargs) -> None:
        """Alias for :meth:`summary`."""
        self.summary(**kwargs)

    def brief(self) -> None:
        """Print a compact one-line-per-entity summary."""
        print_brief_summary(self._analysis)

    def report(self, format: str = "markdown") -> str:
        """[Deferred] Export formatted report.

        Not yet implemented. Use :meth:`summary` for terminal output or
        :meth:`to_dict` / :meth:`to_frame` for programmatic access.
        """
        raise NotImplementedError(
            "report() export is not yet implemented. "
            "Use summary() for terminal output or to_dict()/to_frame() for data."
        )

    # ── data access ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict with CIs, p-values, and pairwise diffs.

        Returns a dict with structure::

            {
                "factors": ...,
                "metric": ...,
                "alpha": ...,
                "entities": {
                    name: {
                        "mean": float,
                        "ci_low": float,
                        "ci_high": float,
                        "p_best": float,  # P(rank 1) from bootstrap
                    }
                },
                "pairwise": [
                    {"a": str, "b": str, "diff": float, "ci_low": float,
                     "ci_high": float, "p_value": float | None}
                ],
            }
        """
        bundle = self._primary_bundle()
        if bundle is None:
            return {
                "factors": self._factors,
                "metric": self._metric,
                "alpha": self._alpha,
                "note": "Multi-bundle result; use to_frame() for structured access.",
            }

        rob = bundle.robustness
        rank = bundle.rank_dist
        pairwise = bundle.pairwise

        entities: dict[str, dict] = {}
        labels = bundle.benchmark.template_labels
        for i, name in enumerate(labels):
            entry: dict[str, Any] = {
                "mean": float(rob.mean[i]),
                "ci_low": float(rob.ci_low[i]) if rob.ci_low is not None else None,
                "ci_high": float(rob.ci_high[i]) if rob.ci_high is not None else None,
            }
            if rank is not None:
                entry["p_best"] = float(rank.p_best[i])
            entities[str(name)] = entry

        pw_list: list[dict] = []
        for (a, b), pair_result in pairwise.results.items():
            pw_entry: dict[str, Any] = {
                "a": str(a),
                "b": str(b),
                "diff": float(pair_result.point_diff),
                "ci_low": float(pair_result.ci_low),
                "ci_high": float(pair_result.ci_high),
            }
            if pair_result.p_value is not None:
                pw_entry["p_value"] = float(pair_result.p_value)
            pw_list.append(pw_entry)

        return {
            "factors": self._factors,
            "metric": self._metric,
            "alpha": self._alpha,
            "entities": entities,
            "pairwise": pw_list,
        }

    def to_frame(self) -> dict[str, pd.DataFrame]:
        """Return analysis results as a dict of DataFrames.

        Keys:

        * ``"entities"`` — one row per entity with mean, CI bounds, P(best).
        * ``"pairwise"`` — one row per pairwise comparison.
        * ``"raw"`` — the filtered input data that was analyzed.
        """
        bundle = self._primary_bundle()
        frames: dict[str, pd.DataFrame] = {"raw": self._df.copy()}

        if bundle is None:
            return frames

        rob = bundle.robustness
        rank = bundle.rank_dist
        pairwise = bundle.pairwise
        labels = bundle.benchmark.template_labels

        entity_rows: list[dict] = []
        for i, name in enumerate(labels):
            row: dict[str, Any] = {
                "entity": str(name),
                "mean": float(rob.mean[i]),
                "ci_low": float(rob.ci_low[i]) if rob.ci_low is not None else None,
                "ci_high": float(rob.ci_high[i]) if rob.ci_high is not None else None,
            }
            if rank is not None:
                row["p_best"] = float(rank.p_best[i])
            entity_rows.append(row)
        frames["entities"] = pd.DataFrame(entity_rows)

        pw_rows: list[dict] = []
        for (a, b), pair_result in pairwise.results.items():
            row_pw: dict[str, Any] = {
                "a": str(a),
                "b": str(b),
                "diff": float(pair_result.point_diff),
                "ci_low": float(pair_result.ci_low),
                "ci_high": float(pair_result.ci_high),
            }
            if pair_result.p_value is not None:
                row_pw["p_value"] = float(pair_result.p_value)
            pw_rows.append(row_pw)
        frames["pairwise"] = pd.DataFrame(pw_rows)

        return frames

    def disagreements(
        self,
        by: Optional[str] = None,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return items where models/prompts disagree most.

        Returns rows from the raw filtered data where the score variance
        across the compared entities is highest — useful for finding examples
        where models diverge strongly.

        Parameters
        ----------
        by : str, optional
            Column to aggregate over (default: the item column).
        threshold : float, optional
            Only include items with score std ≥ threshold.
        top_n : int, optional
            Return only the top-N most disagreed-upon items.
        """
        df = self._df.copy()

        # Determine the entity column (what we're comparing)
        factors = [self._factors] if isinstance(self._factors, str) else self._factors
        item_col = by

        # Identify item col from the EvalResults column mapping — not stored directly,
        # so we check common item column names in df.columns.
        if item_col is None:
            for candidate in ["item", "input", "example", "id", "input_label"]:
                if candidate in df.columns:
                    item_col = candidate
                    break

        if item_col is None:
            raise ValueError(
                "Could not detect item column for disagreement analysis. "
                "Pass by='your_item_column'."
            )

        score_col = self._metric
        if score_col not in df.columns:
            return pd.DataFrame()

        key_cols = [item_col] + [
            f for f in factors if f in df.columns and f != item_col
        ]

        try:
            agg = (
                df.groupby(item_col)[score_col]
                .std()
                .reset_index()
                .rename(columns={score_col: "score_std"})
            )
        except Exception:
            return pd.DataFrame()

        agg = agg.sort_values("score_std", ascending=False)

        if threshold is not None:
            agg = agg[agg["score_std"] >= threshold]

        if top_n is not None:
            agg = agg.head(top_n)

        return agg.reset_index(drop=True)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _primary_bundle(self) -> Optional[AnalysisBundle]:
        """Return a single AnalysisBundle from the underlying analysis, if available."""
        if isinstance(self._analysis, AnalysisBundle):
            return self._analysis
        if isinstance(self._analysis, MultiModelBundle):
            return getattr(self._analysis, self._mmb_view)
        # Per-evaluator dicts: return first
        if isinstance(self._analysis, dict):
            first = next(iter(self._analysis.values()), None)
            if isinstance(first, AnalysisBundle):
                return first
            if isinstance(first, MultiModelBundle):
                return getattr(first, self._mmb_view)
        return None

    def __repr__(self) -> str:
        bundle = self._primary_bundle()
        n_entities = "?" if bundle is None else len(bundle.benchmark.template_labels)
        return (
            f"ComparisonResult("
            f"factors={self._factors!r}, "
            f"metric={self._metric!r}, "
            f"n_entities={n_entities})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal: bridge EvalResults → existing analysis engine
# ─────────────────────────────────────────────────────────────────────────────

def _apply_kwarg_filters(
    df: pd.DataFrame,
    kwargs: dict[str, Any],
    known_params: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Split kwargs into column filters vs. engine kwargs, then filter."""
    col_filters: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in known_params:
            engine_kwargs[k] = v
        elif k in df.columns:
            col_filters[k] = v
        else:
            warnings.warn(
                f"compare(): unknown keyword argument '{k}'. "
                "If this is a column filter, the column was not found in the data. "
                "If it's an analysis parameter, check the spelling.",
                UserWarning,
                stacklevel=3,
            )

    for col, val in col_filters.items():
        if isinstance(val, (list, tuple)):
            df = df[df[col].isin(val)]
        else:
            df = df[df[col] == val]

    if df.empty:
        raise EvalLoadError(
            "After applying column filters from keyword arguments, no data remains."
        )

    return df, engine_kwargs


_ANALYZE_PARAMS = {
    "evaluator_mode", "reference", "method", "backend", "n_bootstrap",
    "correction", "spread_percentiles", "failure_threshold", "rng", "statistic",
    "template_model_collapse", "simultaneous_ci", "omnibus", "p_values",
    "pairwise_test", "ci_style",
}


def _bridge_to_io(
    df: pd.DataFrame,
    *,
    factor_col: str,
    item_col: str,
    metric_col: str,
    run_col: Optional[str] = None,
    block_col: Optional[str] = None,
) -> pd.DataFrame:
    """Rename columns to the canonical names expected by from_dataframe()."""
    rename: dict[str, str] = {}
    if factor_col != "template":
        rename[factor_col] = "template"
    if item_col != "input":
        rename[item_col] = "input"
    if metric_col != "score":
        rename[metric_col] = "score"
    if run_col and run_col != "run":
        rename[run_col] = "run"

    # When block is a second key (e.g. model when comparing prompts), pass as "model"
    if block_col and block_col not in rename and block_col != "model":
        rename[block_col] = "model"

    df_io = df.copy()
    df_io = df_io.rename(columns=rename)

    keep_cols = {"template", "input", "score"}
    if run_col:
        keep_cols.add(rename.get(run_col, run_col))
    if block_col:
        keep_cols.add(rename.get(block_col, block_col))

    keep_cols = keep_cols & set(df_io.columns)
    return df_io[list(keep_cols)]


# ─────────────────────────────────────────────────────────────────────────────
# compare()
# ─────────────────────────────────────────────────────────────────────────────

def compare(
    evaldata: EvalResults,
    *,
    factors: Union[str, list[str]],
    metric: Optional[str] = None,
    baseline: Optional[str] = None,
    block: Union[str, list[str], Literal["auto"]] = "auto",
    slices=None,         # deferred
    secondary=None,      # deferred
    alignment=None,      # deferred
    min_meaningful_diff=None,  # deferred
    alpha: float = 0.05,
    **kwargs: Any,
) -> ComparisonResult:
    """Compare entities along one or more factor axes.

    Parameters
    ----------
    evaldata : EvalResults
        Evaluation data from :func:`load_from`.
    factors : str or list[str]
        What to compare. Common values:

        * ``"model"`` — compare models
        * ``"prompt"`` — compare prompt templates
        * ``["model", "prompt"]`` — factorial design (uses LMM backend)
        * Any other column name — compares levels of that column

    metric : str, optional
        Metric column to analyze. Defaults to the first metric column
        detected by ``load_from``.
    baseline : str, optional
        Name of the baseline entity to compare all others against.
        When ``None``, uses grand-mean reference.
    block : str, list[str], or "auto"
        Blocking variable(s) — typically ``"item"`` or ``"input"``.
        ``"auto"`` (default) uses the item column detected by ``load_from``.
    alpha : float
        Significance level / CI width: ``alpha=0.05`` → 95 % CIs (default).
    **kwargs
        Two uses:

        1. **Column filters** — keyword matching a column name in the data
           filters rows before analysis.
           E.g. ``compare(evaldata, factors="model", split="test")``
           keeps only rows where ``split == "test"``.
           Pass a list to select multiple values:
           ``model=["gpt-4o", "claude-3-5-sonnet"]``.

        2. **Analysis engine overrides** — any of the keyword arguments
           accepted by :func:`~evalstats.core.router.analyze` (e.g.
           ``method="bca"``, ``n_bootstrap=5000``).

    Returns
    -------
    ComparisonResult

    Examples
    --------
    >>> import evalstats as es
    >>> evaldata = es.load_from(df, col_map={"llm": "model", "q_id": "item"})
    >>> result = es.compare(evaldata, factors="model")
    >>> result.summary()

    >>> result = es.compare(evaldata, factors="prompt", method="bca")
    >>> result.to_frame()["entities"]
    """
    if slices is not None:
        warnings.warn("slices= is not yet implemented and will be ignored.", UserWarning, stacklevel=2)
    if secondary is not None:
        warnings.warn("secondary= is not yet implemented and will be ignored.", UserWarning, stacklevel=2)
    if alignment is not None:
        warnings.warn("alignment= is not yet implemented and will be ignored.", UserWarning, stacklevel=2)
    if min_meaningful_diff is not None:
        warnings.warn("min_meaningful_diff= is not yet implemented and will be ignored.", UserWarning, stacklevel=2)

    # ── coerce non-EvalResults input types ────────────────────────────────────
    if isinstance(evaldata, list):
        # list[dict] in long format — delegate directly to load_from
        evaldata = load_from(evaldata)
    elif isinstance(evaldata, dict):
        # dict-of-arrays (flat or nested) — convert via scores-dict helper
        df_from_dict = _scores_dict_to_df(evaldata, factors=factors)
        evaldata = load_from(df_from_dict)
    elif not isinstance(evaldata, EvalResults):
        raise TypeError(
            f"compare() expects EvalResults, list[dict], or dict-of-arrays; "
            f"got {type(evaldata).__name__}. "
            "Use load_from() to construct an EvalResults object from a DataFrame."
        )

    # ── get raw DataFrame and column roles ────────────────────────────────────
    df = evaldata._df.copy()
    col = evaldata._col
    metric_cols = evaldata._metric_cols

    # Resolve metric column
    if metric is None:
        metric_col = metric_cols[0]
    else:
        if metric not in df.columns:
            raise EvalLoadError(
                f"metric column '{metric}' not found in data. "
                f"Available metric columns: {metric_cols}"
            )
        metric_col = metric

    # Resolve item (blocking) column
    if block == "auto":
        item_col = col.get("item")
    elif isinstance(block, str):
        item_col = block
    else:
        item_col = block[0] if block else col.get("item")

    if not item_col or item_col not in df.columns:
        raise EvalLoadError(
            "Could not determine the item/blocking column. "
            "Specify block='your_item_column' or ensure your data has an 'item' column."
        )

    run_col = col.get("run")

    # ── split kwargs into column filters vs. engine kwargs ────────────────────
    df, engine_kwargs = _apply_kwarg_filters(df, kwargs, _ANALYZE_PARAMS)

    # ── set CI level from alpha ───────────────────────────────────────────────
    ci_level = 1.0 - alpha

    # ── dispatch by factor type ───────────────────────────────────────────────
    factors_list = [factors] if isinstance(factors, str) else list(factors)

    # Detect canonical mappings
    model_col  = col.get("model")
    prompt_col = col.get("prompt")

    is_model_comparison  = (len(factors_list) == 1 and
                             factors_list[0] in {"model"} and model_col and model_col in df)
    is_prompt_comparison = (len(factors_list) == 1 and
                             factors_list[0] in {"prompt", "template"} and prompt_col and prompt_col in df)
    is_canonical_col = (len(factors_list) == 1 and factors_list[0] in df.columns and
                        not is_model_comparison and not is_prompt_comparison)
    is_factorial = len(factors_list) >= 2

    # Also handle the case where factor is neither "model" nor "prompt" but names
    # a canonical-alias column directly (e.g. user mapped "llm" → "model", then
    # passes factors="model" which now IS model_col).
    if not is_model_comparison and not is_prompt_comparison and not is_factorial:
        factor_col_name = factors_list[0]
        if factor_col_name in df.columns:
            is_canonical_col = True

    # ── path A: model comparison ──────────────────────────────────────────────
    if is_model_comparison:
        factor_col_name = model_col
        block_col = prompt_col  # prompts become the template axis if present

        if block_col and block_col in df.columns:
            # Multi-model path: map model→"model" axis and prompt→"template" axis.
            # This keeps labels natural: MultiModelBundle.model_level compares models,
            # template_level compares prompts, per_model shows per-model prompt analysis.
            df_multi = df[[factor_col_name, block_col, item_col, metric_col]
                          + ([run_col] if run_col and run_col in df.columns else [])].copy()
            rename_multi = {
                factor_col_name: "model",     # actual models → "model" axis
                block_col: "template",        # prompts → "template" axis
                item_col: "input",
                metric_col: "score",
            }
            if run_col and run_col in df.columns:
                rename_multi[run_col] = "run"
            df_multi = df_multi.rename(columns={k: v for k, v in rename_multi.items() if k != v})
            bench = from_dataframe(df_multi, format="long", strict_complete_design=False)
        else:
            # No prompt col — single-model BenchmarkResult with model as template axis
            df_io_keep = df[[factor_col_name, item_col, metric_col]
                            + ([run_col] if run_col and run_col in df.columns else [])].copy()
            rename_io = {factor_col_name: "template", item_col: "input", metric_col: "score"}
            if run_col and run_col in df.columns:
                rename_io[run_col] = "run"
            df_io_keep = df_io_keep.rename(columns={k: v for k, v in rename_io.items() if k != v})
            bench = from_dataframe(df_io_keep, format="long", strict_complete_design=False)

        reference = baseline if baseline else "grand_mean"
        analysis = analyze(bench, ci=ci_level, reference=reference, **engine_kwargs)

        # model→"model" axis means model_level compares models (what the user requested).
        # When bench is a BenchmarkResult (no prompts), the analysis is an AnalysisBundle
        # and _mmb_view is irrelevant.
        return ComparisonResult(
            analysis,
            factors=factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            _mmb_view="model_level",
        )

    # ── path B: prompt/template comparison ───────────────────────────────────
    if is_prompt_comparison:
        factor_col_name = prompt_col
        block_col = model_col  # if models present, they become block axis

        if block_col and block_col in df.columns:
            # Multi-model path: prompt as template, model as model.
            df_multi = df[[block_col, factor_col_name, item_col, metric_col]
                          + ([run_col] if run_col and run_col in df.columns else [])].copy()
            rename_multi = {
                factor_col_name: "template",
                block_col: "model",
                item_col: "input",
                metric_col: "score",
            }
            if run_col and run_col in df.columns:
                rename_multi[run_col] = "run"
            df_multi = df_multi.rename(columns={k: v for k, v in rename_multi.items() if k != v})
            bench = from_dataframe(df_multi, format="long", strict_complete_design=False)
        else:
            df_io_keep = df[[factor_col_name, item_col, metric_col]
                            + ([run_col] if run_col and run_col in df.columns else [])].copy()
            rename_io = {factor_col_name: "template", item_col: "input", metric_col: "score"}
            if run_col and run_col in df.columns:
                rename_io[run_col] = "run"
            df_io_keep = df_io_keep.rename(columns={k: v for k, v in rename_io.items() if k != v})
            bench = from_dataframe(df_io_keep, format="long", strict_complete_design=False)

        reference = baseline if baseline else "grand_mean"
        analysis = analyze(bench, ci=ci_level, reference=reference, **engine_kwargs)

        return ComparisonResult(
            analysis,
            factors=factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            _mmb_view="template_level",
        )

    # ── path C: arbitrary single factor column ────────────────────────────────
    if is_canonical_col or (not is_factorial and factors_list[0] in df.columns):
        factor_col_name = factors_list[0]

        df_io_keep = df[[factor_col_name, item_col, metric_col]
                        + ([run_col] if run_col and run_col in df.columns else [])].copy()
        rename_io = {factor_col_name: "template", item_col: "input", metric_col: "score"}
        if run_col and run_col in df.columns:
            rename_io[run_col] = "run"
        df_io_keep = df_io_keep.rename(columns={k: v for k, v in rename_io.items() if k != v})
        bench = from_dataframe(df_io_keep, format="long", strict_complete_design=False)

        reference = baseline if baseline else "grand_mean"
        analysis = analyze(bench, ci=ci_level, reference=reference, **engine_kwargs)

        return ComparisonResult(
            analysis,
            factors=factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
        )

    # ── path D: factorial (multiple factors → LMM) ────────────────────────────
    if is_factorial:
        # Validate all factor columns exist
        missing_factors = [f for f in factors_list if f not in df.columns]
        if missing_factors:
            raise EvalLoadError(
                f"Factor column(s) {missing_factors} not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        # Rename metric and item cols to what analyze_factorial expects
        df_fact = df.copy()
        rename_fact: dict[str, str] = {}
        if metric_col != "score":
            rename_fact[metric_col] = "score"
            df_fact = df_fact.rename(columns=rename_fact)
            score_col_name = "score"
        else:
            score_col_name = "score"

        factorial_kwargs = {
            k: v for k, v in engine_kwargs.items()
            if k in {"backend", "ci", "correction", "reference",
                     "spread_percentiles", "failure_threshold", "n_sim", "rng"}
        }
        if "ci" not in factorial_kwargs:
            factorial_kwargs["ci"] = ci_level

        run_col_fact = run_col if run_col and run_col in df_fact.columns else None

        analysis = analyze_factorial(
            df_fact,
            factors=factors_list,
            random_effect=item_col,
            score_col=score_col_name,
            run_col=run_col_fact,
            **factorial_kwargs,
        )

        return ComparisonResult(
            analysis,
            factors=factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
        )

    raise EvalLoadError(
        f"Could not dispatch compare() for factors={factors!r}. "
        f"Factor column(s) not found in data. Available columns: {list(df.columns)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Thin wrappers
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(evaldata: EvalResults, **kwargs) -> ComparisonResult:
    """Compare models — equivalent to ``compare(evaldata, factors="model", ...)``.

    All keyword arguments are forwarded to :func:`compare`.
    """
    return compare(evaldata, factors="model", **kwargs)


def compare_prompts(evaldata: EvalResults, **kwargs) -> ComparisonResult:
    """Compare prompt templates — equivalent to ``compare(evaldata, factors="prompt", ...)``.

    All keyword arguments are forwarded to :func:`compare`.
    """
    return compare(evaldata, factors="prompt", **kwargs)
