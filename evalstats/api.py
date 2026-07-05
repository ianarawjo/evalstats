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
from scipy.stats import norm as _scipy_norm
from scipy.stats import t as _scipy_t

from evalstats.loader import EvalResults, EvalLoadError, load_from, _scores_dict_to_df, _is_nested_scores_dict
from evalstats.io import from_dataframe
from evalstats.config import get_alpha_ci, GRADIENT_CI_ALPHAS
from evalstats.core.router import analyze, analyze_factorial, _analyze_single_lightweight
from evalstats.core.bundles import AnalysisBundle, MultiModelBundle, AnalysisResult
from evalstats.core.stats_utils import correct_pvalues
from evalstats.core.summary import print_analysis_summary, print_brief_summary, _UNSET as _SUMMARY_UNSET


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
        min_meaningful_diff: Optional[float] = None,
    ):
        self._analysis = analysis
        self._factors = factors
        self._metric = metric
        self._baseline = baseline
        self._alpha = alpha
        self._df = filtered_df
        self._mmb_view = _mmb_view  # which MultiModelBundle view is primary
        self._min_meaningful_diff = min_meaningful_diff
        self._variance_components: Optional[dict] = None  # set by MC alignment loop

    # ── print methods ────────────────────────────────────────────────────────

    _UNSET = object()  # sentinel distinguishing "not passed" from None

    def summary(
        self,
        *,
        top_pairwise: Optional[int] = None,
        style: Literal["gradient", "line"] = "gradient",
        p_value_method=_UNSET,
        pairwise_sort: Literal["grouped", "significance"] = "grouped",
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
        p_value_method : str or None, optional
            Override the p-value display method.  When not passed (default),
            uses the method stored in the bundle (from ``p_values=`` /
            ``pairwise_test=`` kwargs).  Pass ``None`` to explicitly suppress,
            or a string like ``"wsr"`` / ``"boot"`` to force a column.
        pairwise_sort : {"grouped", "significance"}
            Pairwise row ordering. ``"grouped"`` (default) keeps related pairs
            together; ``"significance"`` sorts by p-value then effect size.
        """
        # Map our sentinel to the summary module's _UNSET so it reads from bundle.
        pvm = _SUMMARY_UNSET if p_value_method is ComparisonResult._UNSET else p_value_method
        item_singular, item_plural = _factor_item_labels(self._factors)
        print_analysis_summary(
            self._analysis,
            top_pairwise=top_pairwise,
            style=style,
            p_value_method=pvm,
            pairwise_sort=pairwise_sort,
            min_meaningful_diff=self._min_meaningful_diff,
            item_singular=item_singular,
            item_plural=item_plural,
        )

    def brief(self) -> None:
        """Print a compact one-line-per-entity summary."""
        item_singular, item_plural = _factor_item_labels(self._factors)
        print_brief_summary(self._analysis, item_singular=item_singular, item_plural=item_plural)

    def print_ci_table(self, sort_by: str = "mean", as_percent: bool = True) -> None:
        """Print a compact table of entity means and confidence intervals.

        Parameters
        ----------
        sort_by : str
            Row ordering: ``"mean"`` (default, descending), ``"label"``
            (alphabetical), or ``"input_order"`` (preserves label order).
        as_percent : bool
            When ``True`` (default), display values as percentages (0–100).
        """
        bundle = self._primary_bundle()
        if bundle is None:
            print("No analysis available.")
            return

        ci_pct = int((1 - self._alpha) * 100)
        entity_name = self.entity_name_singular.capitalize()
        labels = list(bundle.benchmark.template_labels)
        rob = bundle.robustness
        unbeaten_set = set(self.unbeaten or [])

        rows = []
        for i, lbl in enumerate(labels):
            mean = float(rob.mean[i])
            ci_lo = float(rob.ci_low[i]) if rob.ci_low is not None else None
            ci_hi = float(rob.ci_high[i]) if rob.ci_high is not None else None
            rows.append((str(lbl), mean, ci_lo, ci_hi))

        if sort_by == "mean":
            rows.sort(key=lambda r: r[1], reverse=True)
        elif sort_by == "label":
            rows.sort(key=lambda r: r[0])

        scale = 100.0 if as_percent else 1.0
        pct = "%" if as_percent else ""
        col_w = max((len(r[0]) for r in rows), default=8)
        col_w = max(col_w, len(entity_name))

        header = f"  {entity_name:<{col_w}}  {'Mean':>7}  {ci_pct}% CI                  Status"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for lbl, mean, lo, hi in rows:
            if lo is not None and hi is not None:
                ci_str = f"[{lo*scale:.1f}{pct}, {hi*scale:.1f}{pct}]"
            else:
                ci_str = "—"
            status = "✓" if lbl in unbeaten_set else "—"
            print(f"  {lbl:<{col_w}}  {mean*scale:>6.1f}{pct}  {ci_str:<22}  {status}")

    def print_pair(self, entity_a: str, entity_b: str) -> None:
        """Print the pairwise comparison summary for a specific pair.

        Parameters
        ----------
        entity_a, entity_b : str
            Labels of the two entities to compare.  The pair must be present
            in the pairwise matrix (order does not matter).
        """
        pw = self.pairwise
        if pw is None:
            print("No pairwise analysis available.")
            return
        pair = pw.get(entity_a, entity_b)
        if pair is None:
            print(f"No pairwise comparison found for ({entity_a!r}, {entity_b!r}).")
            return
        pair.summary()

    def plot(self, method: str = "bar", **kwargs):
        """Visualize comparison results.

        Parameters
        ----------
        method : str
            Plot type:

            * ``"bar"`` (default) — accuracy bar chart via
              :func:`~evalstats.vis.scoreboard.plot_accuracy_bar`.
            * ``"forest"`` — horizontal CI forest plot via
              :func:`~evalstats.vis.forest.plot_ci_forest`.
            * ``"cd"`` — critical difference diagram via
              :func:`~evalstats.vis.critical_difference.plot_critical_difference`.

        **kwargs
            Forwarded to the underlying plot function.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if method == "bar":
            from evalstats.vis.scoreboard import plot_accuracy_bar
            return plot_accuracy_bar(self, **kwargs)
        elif method == "forest":
            from evalstats.vis.forest import plot_ci_forest
            return plot_ci_forest(self, **kwargs)
        elif method == "cd":
            from evalstats.vis.critical_difference import plot_critical_difference
            return plot_critical_difference(self, **kwargs)
        else:
            raise ValueError(
                f"Unknown plot method: {method!r}. "
                "Expected 'bar', 'forest', or 'cd'."
            )

    def report(self, format: str = "markdown") -> str:
        """[Deferred] Export formatted report.

        Not yet implemented. Use :meth:`summary` for terminal output or
        :meth:`to_dict` / :meth:`to_frame` for programmatic access.
        """
        raise NotImplementedError(
            "report() export is not yet implemented. "
            "Use summary() for terminal output or to_dict()/to_frame() for data."
        )

    # ── duck-type compatibility properties ───────────────────────────────────
    # These allow ComparisonResult to be passed directly to vis functions
    # (plot_ci_forest, plot_accuracy_bar, plot_critical_difference) that
    # previously accepted CompareReport objects.

    @property
    def labels(self) -> list:
        """Entity labels (for vis function compatibility)."""
        bundle = self._primary_bundle()
        return list(bundle.benchmark.template_labels) if bundle else []

    @property
    def entity_stats(self) -> dict:
        """Per-entity stats dict (for vis function compatibility)."""
        from types import SimpleNamespace
        bundle = self._primary_bundle()
        if bundle is None:
            return {}
        rob = bundle.robustness
        return {
            str(lbl): SimpleNamespace(
                mean=float(rob.mean[i]),
                ci_low=float(rob.ci_low[i]) if rob.ci_low is not None else 0.0,
                ci_high=float(rob.ci_high[i]) if rob.ci_high is not None else 1.0,
                median=float(rob.median[i]),
                std=float(rob.std[i]),
            )
            for i, lbl in enumerate(bundle.benchmark.template_labels)
        }

    @property
    def unbeaten(self) -> Optional[list]:
        """Entities not significantly beaten at the stored alpha level.

        Returns ``None`` when no pairwise differences are significant (the
        concept of "unbeaten" does not apply when there is no winner).
        """
        bundle = self._primary_bundle()
        if bundle is None:
            return None
        labels = list(bundle.benchmark.template_labels)
        beaten: set = set()
        any_sig = False
        for (a, b), pair in bundle.pairwise.results.items():
            if pair.p_value is not None and pair.p_value < self._alpha:
                any_sig = True
                if pair.point_diff > 0:
                    beaten.add(str(b))
                else:
                    beaten.add(str(a))
        if not any_sig:
            return None
        result_unbeaten = [str(lbl) for lbl in labels if str(lbl) not in beaten]
        return result_unbeaten if result_unbeaten else None

    @property
    def entity_name_singular(self) -> str:
        """Entity type name (for vis function compatibility)."""
        f = self._factors
        if isinstance(f, str) and f in ("model", "prompt", "entity"):
            return f
        return "entity"

    @property
    def pairwise(self):
        """Pairwise matrix from the primary bundle."""
        bundle = self._primary_bundle()
        return bundle.pairwise if bundle else None

    @property
    def rank_dist(self):
        """Rank distribution from the primary bundle."""
        bundle = self._primary_bundle()
        return bundle.rank_dist if bundle else None

    @property
    def alpha(self) -> float:
        """Significance level used for this comparison (e.g. 0.05 → 95% CIs)."""
        return self._alpha

    @property
    def p_value_method(self) -> Optional[str]:
        """Stored p-value method from the underlying bundle (or None if not requested)."""
        bundle = self._primary_bundle()
        return bundle.p_value_method if bundle is not None else None

    @property
    def simultaneous_ci(self) -> bool:
        """Whether simultaneous (family-wise) CIs were used for pairwise comparisons."""
        bundle = self._primary_bundle()
        if bundle is None:
            return False
        return bool(bundle.pairwise.simultaneous_ci)

    @property
    def full_analysis(self):
        """Underlying AnalysisBundle (for vis function compatibility)."""
        return self._primary_bundle()

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

        result: dict[str, Any] = {
            "factors": self._factors,
            "metric": self._metric,
            "alpha": self._alpha,
            "entities": entities,
            "pairwise": pw_list,
        }
        if self._variance_components is not None:
            result["variance_components"] = self._variance_components
        return result

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

_STANDARD_FACTOR_NAMES = {"model", "prompt", "template"}
_FACTOR_STD_SLOTS = ["model", "prompt"]  # canonical names for the first two custom factors


def _factor_item_labels(factors) -> tuple[str, str]:
    """Derive (item_singular, item_plural) from a factors argument."""
    if factors is None:
        return "template", "templates"
    if isinstance(factors, str):
        singular = factors
    else:
        singular = "|".join(factors)
    if "|" in singular:
        plural = singular + " combinations"
    elif singular.endswith("s"):
        plural = singular
    else:
        plural = singular + "s"
    return singular, plural


def _custom_factor_col_map(
    factors_list: list[str], df: pd.DataFrame
) -> dict[str, str]:
    """Return a col_map that remaps non-standard factor columns to standard slots.

    Non-standard factor columns (not "model", "prompt", or "template") that
    exist in *df* are mapped to "model" then "prompt" in order.  This allows
    load_from() to include them in the uniqueness key so the duplicate check
    passes.
    """
    col_map: dict[str, str] = {}
    slot_idx = 0
    for f in factors_list:
        if f not in _STANDARD_FACTOR_NAMES and f in df.columns and slot_idx < len(_FACTOR_STD_SLOTS):
            col_map[f] = _FACTOR_STD_SLOTS[slot_idx]
            slot_idx += 1
    return col_map


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
# PPI alignment correction
# ─────────────────────────────────────────────────────────────────────────────

_PPI_PAIRWISE_SUPPORTED = ("tango", "t_interval", "bootstrap", "wilcoxon", "mannwhitney", "bootstrap_t", "bayes_bootstrap")
_PPI_ROBUSTNESS_SUPPORTED = ("wilson", "bootstrap", "bootstrap_t")


def _ppi_pairwise_dispatch(method: str, a, b, a_lab, b_lab, alpha: float, n_boot: int, rng):
    """Dispatch to the PPI-corrected pairwise implementation of *method*.

    Only methods with a validated PPI-corrected counterpart (see
    ``evalstats.tests``'s ``_ppi_paired_*``/``_ppi_two_sample`` functions,
    calibrated via ``simulations/harness --mode ppi``) are supported here.
    """
    from evalstats.tests import (
        _ppi_paired_tango, _ppi_paired_bootstrap_t, _ppi_paired_bayes_bootstrap,
        _ppi_paired_arrays, _ppi_two_sample, _p_x_gt_y_midrank,
    )
    if method == "tango":
        return _ppi_paired_tango(a, b, a_lab, b_lab, alpha)
    if method == "bootstrap_t":
        return _ppi_paired_bootstrap_t(a, b, a_lab, b_lab, alpha, n_boot, rng)
    if method == "bayes_bootstrap":
        return _ppi_paired_bayes_bootstrap(a, b, a_lab, b_lab, alpha, n_boot, rng)
    if method in ("t_interval", "bootstrap"):
        return _ppi_paired_arrays(a, b, a_lab, b_lab, np.mean, alpha, n_boot, rng, rectifier_func=np.mean)
    if method == "wilcoxon":
        return _ppi_paired_arrays(a, b, a_lab, b_lab, np.median, alpha, n_boot, rng, rectifier_func=np.mean)
    if method == "mannwhitney":
        return _ppi_two_sample(
            a, b, a_lab, b_lab,
            lambda ya, yb: float(_p_x_gt_y_midrank(ya, yb) - 0.5),
            alpha, n_boot, rng,
        )
    raise ValueError(
        f"PPI alignment correction has no validated implementation for pairwise "
        f"method {method!r}. Supported pairwise methods: "
        f"{', '.join(repr(m) for m in _PPI_PAIRWISE_SUPPORTED)}. "
        "Pass method=\"auto\" to let PPI correction pick a supported method "
        "automatically, or choose one of the methods above explicitly."
    )


def _ppi_pairwise_unpaired_fallback(a, b, a_lab, b_lab, alpha: float, n_boot: int, rng):
    """Independent-groups PPI fallback for a paired-mean-diff estimand.

    Used when two entities don't have enough commonly-labeled items (same
    item labeled for both) for a proper paired PPI correction, but each
    individually has enough of its own labels. Mathematically valid because
    ``mean(a) - mean(b)`` decomposes into two independent rectifiers — this
    is exactly ``evalstats.tests._ppi_two_sample``'s validated "TTEST" PPI
    form (see ``simulations/harness/cases/pvalues.py``), just applied to
    what would otherwise be a paired comparison.
    """
    from evalstats.tests import _ppi_two_sample
    return _ppi_two_sample(a, b, a_lab, b_lab, lambda ya, yb: float(ya.mean() - yb.mean()), alpha, n_boot, rng)


def _ppi_robustness_dispatch(method: str, a, a_lab, alpha: float, n_boot: int, rng):
    """Dispatch to the PPI-corrected single-sample implementation of *method*."""
    from evalstats.tests import _ppi_single_wilson, _ppi_single_bootstrap_t
    if method == "wilson":
        return _ppi_single_wilson(a, a_lab, alpha)
    if method == "bootstrap_t":
        return _ppi_single_bootstrap_t(a, a_lab, alpha, n_boot, rng)
    if method == "bootstrap":
        from evalstats.ppi import correct as _ppi_correct
        mask = ~np.isnan(a_lab)
        if mask.sum() == 0:
            raise ValueError("No positions have human labels in a_lab.")
        return _ppi_correct(
            np.mean, Y_lab=a_lab[mask], Y_hat_lab=a[mask], Y_hat_unlab=a,
            alpha=alpha, n_boot=n_boot, rng=rng, compute_pvalue=False,
        )
    raise ValueError(
        f"PPI alignment correction has no validated implementation for robustness "
        f"(single-entity) method {method!r}. Supported robustness methods: "
        f"{', '.join(repr(m) for m in _PPI_ROBUSTNESS_SUPPORTED)}. "
        "Pass method=\"auto\" to let PPI correction pick a supported method "
        "automatically, or choose one of the methods above explicitly."
    )


def _run_alignment_ppi(
    cr: "ComparisonResult",
    *,
    df: pd.DataFrame,
    metric_col: str,
    factor_col: str,
    item_col: str,
    alignment_result,
    alpha: float,
    n_boot: int,
    correction: str,
    method: str,
    rng,
) -> None:
    """Override CIs/p-values in ``cr._analysis`` in-place using Prediction-Powered
    Inference (PPI), by dispatching to the specific PPI-corrected implementation
    of whichever pairwise/robustness method applies (see
    ``_ppi_pairwise_dispatch``/``_ppi_robustness_dispatch`` above).

    For ``method="auto"`` the PPI-specific auto table
    (``evalstats.config.resolve_ppi_auto_methods``) picks a method validated
    for PPI use, which need not match the non-aligned auto default for the
    same data (e.g. numeric data defaults to ``t_interval`` without alignment,
    but ``bootstrap_t`` once PPI correction is in play). When the user passes
    an explicit ``method=``, that exact method's PPI-corrected counterpart is
    used, and a clear ``ValueError`` is raised if none exists — PPI correction
    never silently substitutes or falls back to an unvalidated method.

    Every comparison in evalstats is paired by input (see
    ``evalstats.core.paired``'s module docstring), so this pairs items by
    their shared position in ``bundle.benchmark``'s item-aligned score matrix
    (``get_2d_scores()``), rather than treating LLM scores as independent
    per-entity groups.
    """
    bundle = cr._primary_bundle()
    if bundle is None:
        warnings.warn(
            "PPI alignment correction is not yet supported for multi-bundle "
            "(multi-model) results. alignment= will be ignored.",
            UserWarning,
            stacklevel=4,
        )
        return

    if bundle.benchmark.is_seeded:
        raise ValueError(
            "PPI alignment correction does not yet support seeded benchmarks "
            "(R >= 3 repeated runs). Aggregate runs to a single score per "
            "(template, input) cell before passing alignment=."
        )

    rng = np.random.default_rng(rng)

    labels = list(bundle.benchmark.template_labels)
    input_labels = list(bundle.benchmark.input_labels)
    n_entities = len(labels)
    n_items = len(input_labels)
    entity_idx = {e: i for i, e in enumerate(labels)}
    item_idx = {it: j for j, it in enumerate(input_labels)}

    # Item-aligned LLM score matrix: scores_2d[i, j] = entity i's score on
    # item j (NaN for incomplete-design cells; averaged over runs/evaluators).
    scores_2d = bundle.benchmark.get_2d_scores()

    # ── Extract labeled/unlabeled arrays (dataset-level counts only) ──────────
    from evalstats.ppi import resolve_arrays
    Y_hat_unlab, X_unlab, Y_lab, Y_hat_lab, X_lab = resolve_arrays(
        df, metric_col=metric_col, group_col=factor_col, alignment_result=alignment_result
    )

    n_all = len(Y_hat_unlab)
    n_lab = len(Y_lab)

    # ── Minimum sample-size requirements ─────────────────────────────────────
    if n_lab < 15:
        raise ValueError(
            f"PPI alignment requires at least 15 human-labeled items; "
            f"got n_lab={n_lab}. Expand the alignment set and re-run "
            "validate_alignment()."
        )
    if n_all < 50:
        raise ValueError(
            f"PPI alignment requires at least 50 items in the full dataset; "
            f"got N={n_all}. PPI is only beneficial at scale."
        )
    if n_lab < 30:
        warnings.warn(
            f"PPI alignment: only {n_lab} human-labeled items (recommend ≥ 30). "
            "Confidence intervals may under-cover at this sample size.",
            UserWarning,
            stacklevel=4,
        )
    if n_all < 100:
        warnings.warn(
            f"PPI alignment: only {n_all} total items (recommend ≥ 100). "
            "Confidence intervals may under-cover at this sample size.",
            UserWarning,
            stacklevel=4,
        )

    # ── Item-aligned human-label matrix (n_entities x n_items) ───────────────
    human_col = alignment_result.human_col
    lab_rows = df.loc[df[human_col].notna(), [factor_col, item_col, human_col]]
    lab_matrix = np.full((n_entities, n_items), np.nan)
    for e_val, it_val, h_val in zip(
        lab_rows[factor_col].astype(str).to_numpy(),
        lab_rows[item_col].astype(str).to_numpy(),
        lab_rows[human_col].to_numpy(dtype=float),
    ):
        ei = entity_idx.get(e_val)
        ij = item_idx.get(it_val)
        if ei is not None and ij is not None:
            lab_matrix[ei, ij] = h_val

    missing_entities = [e for i, e in enumerate(labels) if np.all(np.isnan(lab_matrix[i]))]
    if missing_entities:
        warnings.warn(
            f"PPI alignment: the following entities have no human-labeled items and "
            f"will keep their uncorrected LLM-only estimate: {missing_entities}. "
            "Consider expanding the alignment set to cover all entities.",
            UserWarning,
            stacklevel=4,
        )

    # ── Resolve the PPI-specific pairwise/robustness method ──────────────────
    from evalstats.core.resampling import is_binary_scores, is_bounded_01_scores
    from evalstats.config import resolve_ppi_auto_methods

    if is_binary_scores(scores_2d):
        data_kind = "binary"
    elif is_bounded_01_scores(scores_2d):
        data_kind = "bounded_01"
    else:
        data_kind = "continuous"

    if method == "auto":
        pairwise_method, robustness_method = resolve_ppi_auto_methods(data_kind)
    else:
        pairwise_method = bundle.resolved_method or method
        robustness_method = bundle.resolved_ci_method or method

    # ── Point estimates (per entity) ──────────────────────────────────────────
    final_means  = np.array(bundle.robustness.mean, dtype=float, copy=True)
    final_ci_low = np.array(bundle.robustness.ci_low, dtype=float, copy=True)
    final_ci_high = np.array(bundle.robustness.ci_high, dtype=float, copy=True)
    multi_ci_lo = {a: np.array(bundle.robustness.multi_ci[a][0], dtype=float, copy=True) for a in GRADIENT_CI_ALPHAS}
    multi_ci_hi = {a: np.array(bundle.robustness.multi_ci[a][1], dtype=float, copy=True) for a in GRADIENT_CI_ALPHAS}
    entity_rectifier = {e: 0.0 for e in labels}

    for i, e in enumerate(labels):
        if e in missing_entities:
            continue  # keep the uncorrected LLM-only estimate already copied above
        valid = ~np.isnan(scores_2d[i])
        arr = scores_2d[i, valid]
        lab_arr = lab_matrix[i, valid]

        res = _ppi_robustness_dispatch(robustness_method, arr, lab_arr, alpha, n_boot, rng)
        final_means[i] = res.estimate
        final_ci_low[i] = res.ci_low
        final_ci_high[i] = res.ci_high
        entity_rectifier[e] = res.rectifier
        for a in GRADIENT_CI_ALPHAS:
            g = _ppi_robustness_dispatch(robustness_method, arr, lab_arr, a, n_boot, rng)
            multi_ci_lo[a][i] = g.ci_low
            multi_ci_hi[a][i] = g.ci_high

    final_multi_ci = {a: (multi_ci_lo[a], multi_ci_hi[a]) for a in GRADIENT_CI_ALPHAS}

    # ── Pairwise diffs ────────────────────────────────────────────────────────
    pair_keys = list(bundle.pairwise.results.keys())
    n_pairs = len(pair_keys)

    # Simultaneous (family-wise) CIs: PPI's per-method distributions (bootstrap-t
    # pivots, closed-form score intervals) don't share a joint null distribution
    # the way all_pairwise()'s single-method bootstrap does for max-T, so a
    # Bonferroni alpha adjustment is used instead when simultaneous_ci was requested.
    use_simultaneous = bool(bundle.pairwise.simultaneous_ci) and n_pairs > 1
    pair_alpha = alpha / n_pairs if use_simultaneous else alpha

    final_diffs = np.empty(n_pairs, dtype=float)
    pair_ci_lo  = np.empty(n_pairs, dtype=float)
    pair_ci_hi  = np.empty(n_pairs, dtype=float)
    pair_pvals  = np.empty(n_pairs, dtype=float)
    pair_multi_ci: dict = {a: {} for a in GRADIENT_CI_ALPHAS}
    pair_test_method: dict = {}
    skipped_pairs = []
    fallback_pairs = []

    for k, (ea, eb) in enumerate(pair_keys):
        pr = bundle.pairwise.results[(ea, eb)]
        ia, ib = entity_idx[ea], entity_idx[eb]
        valid = ~np.isnan(scores_2d[ia]) & ~np.isnan(scores_2d[ib])
        a_arr, b_arr = scores_2d[ia, valid], scores_2d[ib, valid]
        a_lab_arr, b_lab_arr = lab_matrix[ia, valid], lab_matrix[ib, valid]

        n_overlap = int(np.sum(~np.isnan(a_lab_arr) & ~np.isnan(b_lab_arr)))
        n_a_only  = int(np.sum(~np.isnan(a_lab_arr)))
        n_b_only  = int(np.sum(~np.isnan(b_lab_arr)))

        if n_overlap >= 15:
            dispatch = lambda a_, n_boot_, rng_: _ppi_pairwise_dispatch(
                pairwise_method, a_arr, b_arr, a_lab_arr, b_lab_arr, a_, n_boot_, rng_
            )
            pair_test_method[(ea, eb)] = f"PPI {pairwise_method}"
        elif n_a_only >= 15 and n_b_only >= 15:
            # Not enough items are labeled for *both* entities to run the
            # paired PPI method, but each entity individually has enough of
            # its own labels — fall back to independent-groups PPI (see
            # _ppi_pairwise_unpaired_fallback), which only needs that.
            dispatch = lambda a_, n_boot_, rng_: _ppi_pairwise_unpaired_fallback(
                a_arr, b_arr, a_lab_arr, b_lab_arr, a_, n_boot_, rng_
            )
            pair_test_method[(ea, eb)] = "PPI mean-diff (unpaired fallback, insufficient item overlap)"
            fallback_pairs.append((ea, eb))
        else:
            skipped_pairs.append((ea, eb))
            final_diffs[k] = pr.point_diff
            pair_ci_lo[k]  = pr.ci_low
            pair_ci_hi[k]  = pr.ci_high
            pair_pvals[k]  = pr.p_value
            pair_test_method[(ea, eb)] = pr.test_method
            for a in GRADIENT_CI_ALPHAS:
                pair_multi_ci[a][(ea, eb)] = pr.multi_ci.get(a, (pr.ci_low, pr.ci_high)) if pr.multi_ci else (pr.ci_low, pr.ci_high)
            continue

        res = dispatch(pair_alpha, n_boot, rng)
        final_diffs[k] = res.estimate
        pair_ci_lo[k]  = res.ci_low
        pair_ci_hi[k]  = res.ci_high
        pair_pvals[k]  = res.p_value if res.p_value is not None else 1.0

        for a in GRADIENT_CI_ALPHAS:
            g_alpha = a / n_pairs if use_simultaneous else a
            g = dispatch(g_alpha, n_boot, rng)
            pair_multi_ci[a][(ea, eb)] = (g.ci_low, g.ci_high)

    if fallback_pairs:
        warnings.warn(
            f"PPI alignment: the following pairs don't have enough commonly-labeled "
            f"items for a paired PPI correction and used an independent-groups "
            f"fallback instead: {fallback_pairs}. Consider labeling the same items "
            "for every entity to enable the (more efficient) paired correction.",
            UserWarning,
            stacklevel=4,
        )
    if skipped_pairs:
        warnings.warn(
            f"PPI alignment: the following pairs have fewer than 15 labeled items "
            f"for either entity and keep their uncorrected estimate: {skipped_pairs}. "
            "Consider expanding the alignment set to cover more entities.",
            UserWarning,
            stacklevel=4,
        )

    # Multiple-comparison correction on marginal p-values (unaffected by the
    # Bonferroni simultaneous-CI adjustment above, which only widens CIs).
    if correction != "none" and n_pairs > 1:
        pair_pvals = correct_pvalues(pair_pvals, correction)

    # ── Override _analysis in-place ───────────────────────────────────────────
    bundle.robustness.mean     = final_means
    bundle.robustness.ci_low   = final_ci_low
    bundle.robustness.ci_high  = final_ci_high
    bundle.robustness.multi_ci = final_multi_ci

    for k, key in enumerate(pair_keys):
        pr = bundle.pairwise.results[key]
        pr.point_diff = float(final_diffs[k])
        pr.p_value    = float(pair_pvals[k])
        pr.ci_low     = float(pair_ci_lo[k])
        pr.ci_high    = float(pair_ci_hi[k])
        pr.multi_ci   = {a: pair_multi_ci[a][key] for a in GRADIENT_CI_ALPHAS}
        pr.test_method = pair_test_method[key]

    # Update bundle method metadata so summary() headers reflect the PPI method.
    bundle.resolved_method = pairwise_method
    bundle.resolved_ci_method = robustness_method
    bundle.pairwise.simultaneous_ci_method = "bonferroni" if use_simultaneous else None

    # ── Diagnostics ───────────────────────────────────────────────────────────
    cr._variance_components = {
        "method": "ppi",
        "n_all": n_all,
        "n_lab": n_lab,
        "n_boot": n_boot,
        "pairwise_method": pairwise_method,
        "robustness_method": robustness_method,
        "data_kind": data_kind,
        "entities": {
            lbl: {
                "n_labeled": int(np.sum(~np.isnan(lab_matrix[i]))),
                "llm_mean": float(np.nanmean(scores_2d[i])) if not np.all(np.isnan(scores_2d[i])) else None,
                "rectifier": entity_rectifier[lbl],
                "ppi_mean": float(final_means[i]),
            }
            for i, lbl in enumerate(labels)
        },
        "note": (
            "Prediction-Powered Inference (PPI), dispatched per resolved "
            f"method (pairwise={pairwise_method!r}, robustness={robustness_method!r}). "
            "Items are paired by shared position in the benchmark's item-aligned "
            "score matrix, matching evalstats' paired-by-input design."
        ),
    }


def _run_judge_alignment_if_needed(
    cr: "ComparisonResult",
    *,
    alignment,
    metric_col: str,
    n_mc: int,
    alpha: float,
    ci_level: float,
    engine_kwargs: dict,
    df: pd.DataFrame,
    factor_col: str,
    item_col: str,
    run_col: Optional[str],
) -> None:
    """Validate alignment= and dispatch to _run_alignment_ppi when appropriate."""
    if alignment is None:
        return
    if not isinstance(alignment, dict):
        warnings.warn(
            "alignment= must be a dict mapping metric column names to AlignmentResult objects. "
            "Example: alignment={'score': my_alignment_result}. alignment= will be ignored.",
            UserWarning,
            stacklevel=4,
        )
        return
    ar = alignment.get(metric_col)
    if ar is None:
        warnings.warn(
            f"alignment= dict has no entry for metric column '{metric_col}'. "
            f"Keys present: {list(alignment.keys())}. alignment= will be ignored.",
            UserWarning,
            stacklevel=4,
        )
        return
    if not isinstance(cr._analysis, AnalysisBundle):
        warnings.warn(
            "PPI alignment correction is not yet supported for multi-model or factorial "
            "analyses. alignment= will be ignored for this comparison.",
            UserWarning,
            stacklevel=4,
        )
        return
    _run_alignment_ppi(
        cr,
        df=df,
        metric_col=metric_col,
        factor_col=factor_col,
        item_col=item_col,
        alignment_result=ar,
        alpha=alpha,
        n_boot=max(n_mc, 1000),
        correction=engine_kwargs.get("correction", "fdr_bh"),
        method=engine_kwargs.get("method", "auto"),
        rng=engine_kwargs.get("rng"),
    )


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
    alignment=None,
    n_mc: int = 200,
    min_meaningful_diff=None,  # deferred
    alpha: Optional[float] = None,
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
    alpha : float, optional
        Significance level / CI width: ``alpha=0.05`` → 95 % CIs.
        When ``None`` (default), uses the global value set by
        :func:`~evalstats.config.set_alpha_ci` (default 0.05).
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
    # ── resolve alpha (explicit > global default) ─────────────────────────────
    if alpha is None:
        alpha = get_alpha_ci()

    # Preserve the original factor names as provided by the caller; internal
    # dispatch may remap them to standard column names ("model", "prompt") so
    # that load_from() can detect uniqueness constraints correctly.
    _user_factors = factors

    # ── coerce non-EvalResults input types ────────────────────────────────────
    if isinstance(evaldata, list):
        # list[dict] in long format — detect custom factor columns and remap
        # them to standard names so load_from() includes them in the uniqueness
        # key (otherwise the duplicate check rejects multi-factor long tables).
        _f_list = [factors] if isinstance(factors, str) else list(factors)
        _tmp_df = pd.DataFrame(evaldata) if evaldata else pd.DataFrame()
        _cmap = _custom_factor_col_map(_f_list, _tmp_df)
        if _cmap:
            evaldata = load_from(evaldata, col_map=_cmap)
            _f_list = [_cmap.get(f, f) for f in _f_list]
            factors = _f_list[0] if len(_f_list) == 1 else _f_list
        else:
            evaldata = load_from(evaldata)
    elif isinstance(evaldata, dict):
        # dict-of-arrays (flat or nested) — convert via scores-dict helper,
        # then ensure factor column names resolve to standard slot names.
        # _scores_dict_to_df normalizes nested-dict keys to "model"/"prompt"
        # regardless of factors; flat-dict custom names stay as-is and need
        # renaming so load_from() includes them in the uniqueness key.
        _f_list = [factors] if isinstance(factors, str) else list(factors)
        df_from_dict = _scores_dict_to_df(evaldata, factors=factors)
        _f_actual: list[str] = []
        for _i, _f in enumerate(_f_list):
            _std = _FACTOR_STD_SLOTS[_i] if _i < len(_FACTOR_STD_SLOTS) else _f
            if _f in df_from_dict.columns and _f not in _STANDARD_FACTOR_NAMES:
                # Flat-dict custom column — rename to standard slot in place
                df_from_dict = df_from_dict.rename(columns={_f: _std})
                _f_actual.append(_std)
            elif _std in df_from_dict.columns:
                # Nested-dict already produced the standard slot name
                _f_actual.append(_std)
            else:
                _f_actual.append(_f)
        factors = _f_actual[0] if len(_f_actual) == 1 else _f_actual
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
        cr = ComparisonResult(
            analysis,
            factors=_user_factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            _mmb_view="model_level",
            min_meaningful_diff=min_meaningful_diff,
        )
        _run_judge_alignment_if_needed(
            cr, alignment=alignment, metric_col=metric_col, n_mc=n_mc,
            alpha=alpha, ci_level=ci_level, engine_kwargs=engine_kwargs,
            df=df, factor_col=factor_col_name, item_col=item_col, run_col=run_col,
        )
        return cr

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

        cr = ComparisonResult(
            analysis,
            factors=_user_factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            _mmb_view="template_level",
            min_meaningful_diff=min_meaningful_diff,
        )
        _run_judge_alignment_if_needed(
            cr, alignment=alignment, metric_col=metric_col, n_mc=n_mc,
            alpha=alpha, ci_level=ci_level, engine_kwargs=engine_kwargs,
            df=df, factor_col=factor_col_name, item_col=item_col, run_col=run_col,
        )
        return cr

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

        cr = ComparisonResult(
            analysis,
            factors=_user_factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            min_meaningful_diff=min_meaningful_diff,
        )
        _run_judge_alignment_if_needed(
            cr, alignment=alignment, metric_col=metric_col, n_mc=n_mc,
            alpha=alpha, ci_level=ci_level, engine_kwargs=engine_kwargs,
            df=df, factor_col=factor_col_name, item_col=item_col, run_col=run_col,
        )
        return cr

    # ── path D-pre: canonical 2-factor ["model","prompt"] → multi-model path ──
    # When the user passes exactly the standard factor names (not custom names
    # that were remapped), prefer the richer MultiModelBundle path over factorial
    # LMM — it's simpler, faster, and shows the cross-model ranking output the
    # user usually wants.  An explicit LMM backend opt-in bypasses this.
    _user_factors_list = [_user_factors] if isinstance(_user_factors, str) else list(_user_factors)
    _is_canonical_2factor = (
        is_factorial
        and len(_user_factors_list) == 2
        and all(f in _STANDARD_FACTOR_NAMES for f in _user_factors_list)
        and model_col and model_col in df.columns
        and prompt_col and prompt_col in df.columns
        and engine_kwargs.get("backend") not in {"lmm", "factorial_lmm"}
    )
    if _is_canonical_2factor:
        df_multi = df[
            [model_col, prompt_col, item_col, metric_col]
            + ([run_col] if run_col and run_col in df.columns else [])
        ].copy()
        rename_multi = {
            model_col: "model",
            prompt_col: "template",
            item_col: "input",
            metric_col: "score",
        }
        if run_col and run_col in df.columns:
            rename_multi[run_col] = "run"
        df_multi = df_multi.rename(columns={k: v for k, v in rename_multi.items() if k != v})
        bench = from_dataframe(df_multi, format="long", strict_complete_design=False)
        reference = baseline if baseline else "grand_mean"
        analysis = analyze(bench, ci=ci_level, reference=reference, **engine_kwargs)
        return ComparisonResult(
            analysis,
            factors=_user_factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            _mmb_view="model_level",
            min_meaningful_diff=min_meaningful_diff,
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
            factors=_user_factors,
            metric=metric_col,
            baseline=baseline,
            alpha=alpha,
            filtered_df=df,
            min_meaningful_diff=min_meaningful_diff,
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
