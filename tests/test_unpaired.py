"""Tests for the between-subjects comparison engine:
evalstats.core.unpaired.compare_unpaired(), GroupComparisonResult, and
compare(design=...) routing in evalstats/api.py.

See PLAN_between_subjects_extension.md for the design this implements.
"""
from __future__ import annotations

import io
import warnings as warnings_lib
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

import evalstats as es
from evalstats.core.unpaired import compare_unpaired, GroupComparisonResult, SYNTHETIC_ITEM_COL
from evalstats.alignment import judge_alignment


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_unpaired_df(
    group_means: dict[str, float],
    n_per_group: int | dict[str, int] = 40,
    std: float = 0.15,
    seed: int = 0,
    item_col: str | None = "item",
) -> pd.DataFrame:
    """Disjoint-item long-format continuous data, one independent cohort per group."""
    rng = _rng(seed)
    rows = []
    for g, mean in group_means.items():
        n = n_per_group[g] if isinstance(n_per_group, dict) else n_per_group
        for i in range(n):
            row = {"model": g, "score": float(np.clip(rng.normal(mean, std), 0, 1))}
            if item_col is not None:
                row[item_col] = f"{g}_{i}"
            rows.append(row)
    return pd.DataFrame(rows)


def _make_unpaired_binary_df(
    group_p: dict[str, float], n_per_group: int = 40, seed: int = 1,
) -> pd.DataFrame:
    rng = _rng(seed)
    rows = []
    for g, p in group_p.items():
        for i in range(n_per_group):
            rows.append({"model": g, "item": f"{g}_{i}", "score": float(rng.binomial(1, p))})
    return pd.DataFrame(rows)


def _make_unpaired_with_alignment(
    group_means: dict[str, float], n_per_group: int = 60, n_labeled_per_group: int = 20, seed: int = 2,
):
    """Disjoint-item continuous data with a sparse human_score column, for PPI tests."""
    rng = _rng(seed)
    rows = []
    for g, mean in group_means.items():
        for i in range(n_per_group):
            rows.append({"model": g, "item": f"{g}_{i}", "llm_score": float(np.clip(rng.normal(mean, 0.15), 0, 1))})
    df = pd.DataFrame(rows)
    human = np.full(len(df), np.nan)
    for g in group_means:
        idx = df.index[df["model"] == g].to_numpy()
        chosen = rng.choice(idx, size=min(n_labeled_per_group, len(idx)), replace=False)
        for j in chosen:
            human[j] = float(np.clip(df.loc[j, "llm_score"] + rng.normal(0, 0.05), 0, 1))
    df["human_score"] = human
    return df


# ─────────────────────────────────────────────────────────────────────────────
# compare_unpaired() -- direct engine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareUnpairedBasics:
    def test_k2_continuous_rank_based_no_omnibus(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        r = compare_unpaired(df, factor_col="model", metric_col="score")
        assert isinstance(r, GroupComparisonResult)
        assert r.family == "rank_based"
        assert r.score_type == "continuous"
        assert len(r.groups) == 2
        assert r.n_pairs == 1
        assert r.omnibus_test_name is None
        assert r.ci_correction == "none"
        assert r.pvalue_correction == "none"
        assert len(r.pairwise) == 1
        pair = r.pairwise[0]
        assert pair.estimand == "dominance"
        assert pair.null_value == 0.5
        # B has a clearly higher mean; dominance should reflect that direction.
        assert pair.significant

    def test_k3_continuous_has_omnibus_and_corrections(self):
        df = _make_unpaired_df({"A": 0.3, "B": 0.5, "C": 0.7})
        r = compare_unpaired(df, factor_col="model", metric_col="score")
        assert len(r.groups) == 3
        assert r.n_pairs == 3
        assert r.omnibus_test_name == "Kruskal-Wallis test"
        assert r.omnibus_statistic is not None
        assert r.omnibus_p_value is not None
        assert r.ci_correction == "bonferroni"
        assert r.pvalue_correction == "holm"
        assert len(r.pairwise) == 3
        # Widely separated means -> omnibus should reject at alpha=0.05.
        assert r.omnibus_p_value < 0.05

    def test_binary_family_uses_anova_and_ttest(self):
        df = _make_unpaired_binary_df({"A": 0.3, "B": 0.5, "C": 0.8})
        r = compare_unpaired(df, factor_col="model", metric_col="score")
        assert r.score_type == "binary"
        assert r.family == "binary_proportion"
        assert r.omnibus_test_name == "One-way ANOVA (independent)"
        for pair in r.pairwise:
            assert pair.estimand == "mean_diff"
            assert pair.null_value == 0.0

    def test_unbalanced_group_sizes(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6}, n_per_group={"A": 15, "B": 55})
        r = compare_unpaired(df, factor_col="model", metric_col="score")
        n_a = next(g.n for g in r.groups if g.label == "A")
        n_b = next(g.n for g in r.groups if g.label == "B")
        assert n_a == 15
        assert n_b == 55

    def test_synthetic_item_column_fallback(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6}, item_col=None)
        assert "item" not in df.columns
        r = compare_unpaired(df, factor_col="model", metric_col="score")
        assert r.item_col_synthetic is True
        assert r.item_col == SYNTHETIC_ITEM_COL

    def test_explicit_item_col_not_in_data_raises(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        with pytest.raises(ValueError, match="not found"):
            compare_unpaired(df, factor_col="model", metric_col="score", item_col="nonexistent")

    def test_unknown_factor_col_raises(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        with pytest.raises(ValueError, match="not found"):
            compare_unpaired(df, factor_col="nonexistent", metric_col="score")

    def test_single_group_raises(self):
        df = _make_unpaired_df({"A": 0.4})
        with pytest.raises(ValueError, match="at least 2 groups"):
            compare_unpaired(df, factor_col="model", metric_col="score")


class TestCompareUnpairedNaNAndPPIGuards:
    """Regression tests for bugs found by an independent integration review
    of the between-subjects engine (2026-08-15): NaN handling in the metric
    column, and PPI label-sanitization bypass.
    """

    def test_nan_scores_dropped_with_warning_not_poisoning_result(self):
        rng = _rng(10)
        rows = []
        for g, mean in [("A", 0.4), ("B", 0.6), ("C", 0.5)]:
            for i in range(30):
                score = float(np.clip(rng.normal(mean, 0.15), 0, 1))
                if rng.random() < 0.1:
                    score = float("nan")
                rows.append({"group": g, "item": f"{g}_{i}", "score": score})
        df = pd.DataFrame(rows)
        assert df["score"].isna().sum() > 0
        with pytest.warns(UserWarning, match="dropped"):
            r = compare_unpaired(df, factor_col="group", metric_col="score", n_boot=300, rng=10)
        for g in r.groups:
            assert not np.isnan(g.mean)
            assert not np.isnan(g.ci_low) and not np.isnan(g.ci_high)
        assert any(g.n < 30 for g in r.groups)  # some rows were dropped somewhere
        assert not np.isnan(r.omnibus_p_value)

    def test_nan_scores_dont_crash_binary_family(self):
        rng = _rng(11)
        rows = []
        for g, p in [("A", 0.3), ("B", 0.6)]:
            for i in range(30):
                score = float(rng.binomial(1, p))
                if rng.random() < 0.1:
                    score = float("nan")
                rows.append({"group": g, "item": f"{g}_{i}", "score": score})
        df = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match="dropped"):
            r = compare_unpaired(df, factor_col="group", metric_col="score", n_boot=300, rng=11)
        assert r.score_type == "binary"

    def test_all_nan_group_raises_clear_error(self):
        rows = [{"group": "A", "item": f"A_{i}", "score": float("nan")} for i in range(10)]
        rows += [{"group": "B", "item": f"B_{i}", "score": 0.5} for i in range(10)]
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="no valid"):
            compare_unpaired(df, factor_col="group", metric_col="score")

    def test_ppi_zero_labeled_group_raises_clear_error(self):
        df = _make_unpaired_with_alignment({"A": 0.4, "B": 0.6}, n_per_group=30, n_labeled_per_group=20)
        # Wipe out B's labels entirely after generation -- zero-labeled group.
        df = df.copy()
        df.loc[df["model"] == "B", "human_score"] = np.nan
        assert df.loc[df["model"] == "B", "human_score"].notna().sum() == 0
        assert df.loc[df["model"] == "A", "human_score"].notna().sum() > 0
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with warnings_lib.catch_warnings():
            warnings_lib.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        with pytest.raises(ValueError, match="zero labeled"):
            compare_unpaired(df, factor_col="model", metric_col="llm_score", alignment={"llm_score": ar})

    def test_ppi_too_few_total_labels_raises_clear_error(self):
        df = _make_unpaired_with_alignment({"A": 0.4, "B": 0.6}, n_per_group=30, n_labeled_per_group=2)
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with warnings_lib.catch_warnings():
            warnings_lib.simplefilter("ignore")
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        with pytest.raises(ValueError, match="At least 15 human labels"):
            compare_unpaired(df, factor_col="model", metric_col="llm_score", alignment={"llm_score": ar})

    def test_score_range_threaded_through_and_suppresses_autodetect_warning(self):
        rng = _rng(12)
        rows = []
        for g, mean in [("A", 2.0), ("B", 3.5)]:
            for i in range(30):
                rows.append({"group": g, "item": f"{g}_{i}",
                             "score": float(np.clip(rng.normal(mean, 1.0), 1, 5))})
        df = pd.DataFrame(rows)
        with warnings_lib.catch_warnings(record=True) as caught:
            warnings_lib.simplefilter("always")
            r = compare_unpaired(df, factor_col="group", metric_col="score", score_range=(1, 5))
        assert not any("score_range" in str(w.message) for w in caught)
        assert r.groups[0].method != "t_interval"  # bounds-agnostic fallback shouldn't fire

    def test_method_override_raises_via_compare(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        with pytest.raises(ValueError, match="method='bca'"):
            es.compare(evaldata, factors="model", metric="score", design="unpaired", method="bca")

    def test_k2_point_estimate_matches_public_mannwhitney(self):
        """_rank_based_pairwise_uncorrected reuses the private
        _kw_pairwise_thetas machinery at k=2 rather than routing through
        the public mannwhitney() wrapper (justified by kruskalwallis's own
        docstring: Kruskal-Wallis reduces to Mann-Whitney at k=2) -- verify
        that claim numerically rather than trusting the docstring alone.
        mannwhitney()'s raw U-statistic / (n_x*n_y) is P_mid(X>Y), the same
        quantity _rank_based_pairwise_uncorrected reports as theta_hat.
        """
        from evalstats.tests import mannwhitney
        from evalstats.core.unpaired import _rank_based_pairwise_uncorrected

        rng = _rng(99)
        x = rng.normal(0.4, 0.15, 40)
        y = rng.normal(0.6, 0.15, 35)
        mw = mannwhitney(x, y, alpha=0.05, print_result=False)
        theta_from_mw = mw.statistic / (len(x) * len(y))
        out = _rank_based_pairwise_uncorrected([x, y], alpha=0.05, n_boot=1, rng=1)
        assert np.isclose(theta_from_mw, out["point"][0])

    def test_routing_table_family_drives_dispatch(self):
        from evalstats.config import resolve_auto_unpaired_methods
        for score_type in ["binary", "continuous", "likert", "grade"]:
            family, omnibus_method, pairwise_method = resolve_auto_unpaired_methods(score_type)
            assert family in ("binary_proportion", "rank_based")
            if score_type == "binary":
                assert family == "binary_proportion"
                assert omnibus_method == "anova_oneway"
            else:
                assert family == "rank_based"
                assert omnibus_method == "kruskalwallis"


class TestGroupComparisonResultReporting:
    def _result(self) -> GroupComparisonResult:
        df = _make_unpaired_df({"A": 0.3, "B": 0.5, "C": 0.7})
        return compare_unpaired(df, factor_col="model", metric_col="score")

    def test_summary_runs_without_error(self):
        r = self._result()
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Between-subjects comparison" in out
        assert "Kruskal-Wallis" in out

    def test_plot_not_implemented(self):
        r = self._result()
        with pytest.raises(NotImplementedError):
            r.plot()

    def test_pairwise_table_uses_shared_print_pairwise_section(self):
        """The pairwise comparison table is rendered by the SAME function
        the paired path uses (core.summary._print_pairwise_section), not a
        parallel reimplementation -- see core/summary_unpaired.py's module
        docstring. This changed the unpaired table's format: an interval-
        plot bar per pair (previously text-only), the estimand shown as a
        signed deviation from null (Δθ for the dominance family, unchanged
        for Δp since its null is already 0), and p-values with significance
        stars -- replacing the old verbal "Verdict: significant (A < B)"
        column, which doesn't exist in the shared renderer.
        """
        from evalstats.core.summary import _print_pairwise_section
        import evalstats.core.summary_unpaired as _mod
        assert _mod._print_pairwise_section.__module__ == "evalstats.core.summary"

        r = self._result()
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "effect: Left - Right" in out  # shared axis/legend line
        assert "Δθ" in out  # dominance family shown as a deviation from null=0.5
        # Old per-row verbal verdict cell ("significant (A < B)" / "not
        # significant") is gone -- replaced by the shared table's numeric
        # CI + p + stars. The unrelated footer sentence ("Verdict reflects
        # the ...-corrected CI...") is intentionally still present.
        assert "significant (" not in out

    def test_means_table_uses_shared_print_mean_advantage(self):
        """The per-group means table is rendered by the SAME function the
        paired path uses (core.summary._print_mean_advantage), not a
        parallel reimplementation -- see core/summary_unpaired.py's module
        docstring. A change to that shared function's section header
        renders identically for both paths; assert the literal header text
        here as a tripwire against that sharing silently regressing back
        into two independent implementations.
        """
        from evalstats.core.summary_unpaired import print_group_comparison_summary
        import evalstats.core.summary_unpaired as _mod
        assert _mod._print_mean_advantage.__module__ == "evalstats.core.summary"

        r = self._result()
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        # Exact text _print_mean_advantage prints -- same string the paired
        # path's own summary shows for its equivalent section.
        assert "--- Mean Performance (" in out

    def test_to_dict_shape(self):
        r = self._result()
        d = r.to_dict()
        assert d["design"] == "unpaired"
        assert set(d["groups"].keys()) == {"A", "B", "C"}
        assert d["omnibus"]["test_name"] == "Kruskal-Wallis test"
        assert len(d["pairwise"]) == 3

    def test_to_frame_shape(self):
        r = self._result()
        frame = r.to_frame()
        assert len(frame) == 3
        assert {"a", "b", "point_estimate", "ci_low", "ci_high", "p_value", "significant"} <= set(frame.columns)

    def test_groups_to_frame_shape(self):
        r = self._result()
        frame = r.groups_to_frame()
        assert len(frame) == 3
        assert frame.index.name == "label"
        assert set(frame.index) == {"A", "B", "C"}

    def test_labels_property(self):
        r = self._result()
        assert set(r.labels) == {"A", "B", "C"}


class TestCompareUnpairedWithPPI:
    def test_ppi_applied_and_single_alignment_banner(self):
        df = _make_unpaired_with_alignment({"A": 0.35, "B": 0.65})
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with pytest.warns(UserWarning):
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        r = compare_unpaired(
            df, factor_col="model", metric_col="llm_score",
            alignment={"llm_score": ar},
        )
        assert r.ppi_applied is True
        assert r.alignment_result is ar

        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        # Exactly one alignment report should be printed, not a duplicate/stale second one.
        assert out.count("PPI-CORRECTED") == 1

    def test_ppi_k2_pairwise_survives_degenerate_covariance_seeds(self):
        """Regression test for a ZeroDivisionError in
        evalstats.tests._ppi_kruskal_wallis_pairwise (found via
        simulations/investigate_unpaired_battle_test.py's crash grid): at
        k=2 there is exactly one pair, so the pairwise Wald covariance is a
        1x1 matrix that can come back numerically rank-0 (all bootstrap
        replicates ~identical) for some data/seed combinations -- 6 of 192
        battle-test grid cells hit this (continuous/grade score types,
        k=2, ppi=True, specific seeds). `_ppi_kruskal_wallis_pairwise` now
        guards `df == 0` explicitly instead of dividing by `nu * df`.
        Sweep several seeds here since the failure is seed-dependent and a
        single fixed seed previously happened not to trigger it.
        """
        for seed in range(6):
            df = _make_unpaired_with_alignment(
                {"A": 0.4, "B": 0.6}, n_per_group=30, n_labeled_per_group=8, seed=seed,
            )
            evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
            with warnings_lib.catch_warnings():
                warnings_lib.simplefilter("ignore")
                ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
            r = compare_unpaired(
                df, factor_col="model", metric_col="llm_score",
                alignment={"llm_score": ar}, n_boot=400, rng=seed,
            )
            assert 0.0 <= r.pairwise[0].p_value <= 1.0
            assert 0.0 <= r.pairwise[0].raw_p_value <= 1.0

    def test_ppi_three_groups_omnibus_and_pairwise_work(self):
        df = _make_unpaired_with_alignment({"A": 0.3, "B": 0.5, "C": 0.7}, n_per_group=60, n_labeled_per_group=20)
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with pytest.warns(UserWarning):
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        r = compare_unpaired(
            df, factor_col="model", metric_col="llm_score",
            alignment={"llm_score": ar},
        )
        assert r.ppi_applied is True
        assert r.omnibus_test_name == "Kruskal-Wallis test"
        assert r.omnibus_corrected_p_value is not None
        assert len(r.pairwise) == 3


def _make_unpaired_pareto_df(
    score_means: dict[str, float],
    n_per_group: int | dict[str, int] = 50,
    seed: int = 30,
    cost_means: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Disjoint-item data with a primary ("score", higher=better) and a
    secondary ("cost", lower=better) metric, row-aligned within each group.
    cost_means defaults to the same value (200) for every group -- tests
    that care about a specific dominance pattern pass cost_means explicitly
    (or override df["cost"] afterward).
    """
    rng = _rng(seed)
    rows = []
    for g, score_mean in score_means.items():
        cost_mean = cost_means[g] if cost_means else 200.0
        n = n_per_group[g] if isinstance(n_per_group, dict) else n_per_group
        for i in range(n):
            rows.append({
                "model": g, "item": f"{g}_{i}",
                "score": float(np.clip(rng.normal(score_mean, 0.08), 0, 1)),
                "cost": float(rng.normal(cost_mean, 15)),
            })
    return pd.DataFrame(rows)


class TestCompareUnpairedPareto:
    def test_clear_dominator_is_frontier_others_dominated(self):
        # B has both the best score AND the lowest cost -- unambiguous dominator.
        df = _make_unpaired_pareto_df({"A": 0.6, "B": 0.85, "C": 0.5}, n_per_group={"A": 50, "B": 50, "C": 50})
        # cost means chosen so B < A < C on cost too (B dominates both on both axes)
        df.loc[df["model"] == "A", "cost"] = _rng(31).normal(200, 15, (df["model"] == "A").sum())
        df.loc[df["model"] == "B", "cost"] = _rng(32).normal(150, 15, (df["model"] == "B").sum())
        df.loc[df["model"] == "C", "cost"] = _rng(33).normal(260, 15, (df["model"] == "C").sum())
        r = compare_unpaired(
            df, factor_col="model", metric_col="score",
            secondary_metric={"cost": "min"}, n_boot=800, rng=30,
        )
        assert r.pareto is not None
        assert r.pareto_status["B"].status == "frontier"
        assert r.pareto_status["A"].status == "dominated"
        assert r.pareto_status["C"].status == "dominated"
        assert "B" in r.pareto_status["A"].dominated_by
        assert r.pareto_frontier_probability["B"] == pytest.approx(1.0)
        assert r.pareto_frontier_probability["A"] == pytest.approx(0.0)

    def test_k2_pareto_works(self):
        df = _make_unpaired_pareto_df({"A": 0.6, "B": 0.6}, n_per_group=50)
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min"}, n_boot=500, rng=30)
        assert r.pareto is not None
        assert set(r.pareto_status.keys()) == {"A", "B"}

    def test_unbalanced_groups_pareto_works(self):
        df = _make_unpaired_pareto_df(
            {"A": 0.5, "B": 0.7, "C": 0.6}, n_per_group={"A": 15, "B": 60, "C": 30}, seed=34,
        )
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min"}, n_boot=500, rng=34)
        assert r.pareto is not None
        assert len(r.pareto["result"].labels) == 3

    def test_max_direction(self):
        # secondary metric where higher is also better (e.g. throughput).
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.7}, seed=35)
        df = df.rename(columns={"cost": "throughput"})
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"throughput": "max"}, n_boot=500, rng=35)
        assert r.pareto is not None
        assert r.pareto["direction"] == "max"

    def test_malformed_secondary_metric_raises(self):
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.7})
        with pytest.raises(ValueError, match="exactly one entry"):
            compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min", "extra": "max"})
        with pytest.raises(ValueError, match="min.*or.*max"):
            compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "sideways"})
        with pytest.raises(ValueError, match="not found"):
            compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"nonexistent_col": "min"})

    def test_row_level_nan_in_either_metric_drops_jointly(self):
        rng = _rng(36)
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.7}, n_per_group=40, seed=36)
        # NaN the cost for a few rows in A, and score for a few rows in B --
        # both should be dropped from BOTH arrays to preserve row alignment.
        idx_a = df.index[df["model"] == "A"][:3]
        idx_b = df.index[df["model"] == "B"][:2]
        df.loc[idx_a, "cost"] = np.nan
        df.loc[idx_b, "score"] = np.nan
        with pytest.warns(UserWarning, match="dropped"):
            r = compare_unpaired(df, factor_col="model", metric_col="score",
                                  secondary_metric={"cost": "min"}, n_boot=500, rng=36)
        assert r.pareto is not None
        a_group = r._group("A")
        b_group = r._group("B")
        assert a_group.n == 37  # 40 - 3
        assert b_group.n == 38  # 40 - 2

    def test_to_dict_includes_pareto(self):
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.7}, seed=37)
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min"}, n_boot=500, rng=37)
        d = r.to_dict()
        assert "pareto" in d
        assert d["pareto"]["secondary_metric"] == "cost"
        assert d["pareto"]["direction"] == "min"
        assert set(d["pareto"]["groups"].keys()) == {"A", "B"}
        for entry in d["pareto"]["groups"].values():
            assert "status" in entry and "p_pareto_optimal" in entry

    def test_to_dict_omits_pareto_when_not_requested(self):
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.7}, seed=38)
        r = compare_unpaired(df, factor_col="model", metric_col="score", n_boot=500, rng=38)
        assert r.pareto is None
        assert r.pareto_status is None
        assert r.pareto_frontier_probability is None
        assert "pareto" not in r.to_dict()

    def test_summary_prints_pareto_section_using_shared_paired_renderer(self):
        """The Pareto section is rendered by the SAME function the paired
        path uses (core.summary._print_pareto_section), including its ASCII
        scatterplot -- see core/summary_unpaired.py's module docstring and
        evalstats.core.unpaired._GroupStatsAsRobustness.
        """
        from evalstats.core.summary import _print_pareto_section
        import evalstats.core.summary_unpaired as _mod
        assert _mod._print_pareto_section.__module__ == "evalstats.core.summary"

        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.8, "C": 0.4}, seed=39)
        r = compare_unpaired(df, factor_col="model", metric_col="score",
                              secondary_metric={"cost": "min"}, n_boot=800, rng=39)
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Trade-off" in out
        assert "Pareto Front" in out

    def test_design_unpaired_via_compare_with_secondary_metric(self):
        df = _make_unpaired_pareto_df({"A": 0.5, "B": 0.8}, seed=40)
        evaldata = es.load_from(df)
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired",
                        secondary_metric={"cost": "min"}, rng=40)
        assert isinstance(r, GroupComparisonResult)
        assert r.pareto is not None
        assert r.pareto_status["B"].status == "frontier"


# ─────────────────────────────────────────────────────────────────────────────
# compare(design=...) routing -- api.py integration
# ─────────────────────────────────────────────────────────────────────────────

class TestPValuesOmnibusToggles:
    """p_values=/omnibus= are unpaired-specific opt-outs (default True, not
    compare()'s own False) -- see PLAN discussion + api.py's design=
    docstring. Verifies both the default (unset) preserves the always-shown
    behavior this path was built and battle-tested with, and that explicit
    False actually suppresses.
    """

    def _df(self):
        rng = _rng(50)
        rows = []
        for g, mean in [("A", 0.3), ("B", 0.5), ("C", 0.7)]:
            for i in range(30):
                rows.append({"model": g, "item": f"{g}_{i}",
                             "score": float(np.clip(rng.normal(mean, 0.15), 0, 1))})
        return pd.DataFrame(rows)

    def test_default_shows_both(self):
        evaldata = es.load_from(self._df())
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired", rng=1)
        assert r.show_p_values is True
        assert r.omnibus_test_name is not None
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Omnibus Test" in out
        assert "  p" in out or "p " in out

    def test_p_values_false_hides_column_keeps_data(self):
        evaldata = es.load_from(self._df())
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired",
                        p_values=False, rng=1)
        assert r.show_p_values is False
        # underlying values still computed and accessible programmatically
        assert all(p.p_value is not None for p in r.pairwise)
        assert "p_value" in r.to_frame().columns
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        out = buf.getvalue()
        assert "Verdict reflects" not in out  # p-correction footnote suppressed

    def test_omnibus_false_skips_computation_entirely(self):
        evaldata = es.load_from(self._df())
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired",
                        omnibus=False, rng=1)
        assert r.omnibus_test_name is None
        assert r.omnibus_statistic is None
        assert r.omnibus_p_value is None
        buf = io.StringIO()
        with redirect_stdout(buf):
            r.summary()
        assert "Omnibus Test" not in buf.getvalue()
        # pairwise table is untouched
        assert len(r.pairwise) == 3

    def test_paired_path_p_values_omnibus_unaffected_by_none_default(self):
        # compare()'s own p_values=/omnibus= defaults changed from False to
        # None (a sentinel distinguishing "unset" from "explicitly False")
        # -- both are falsy, so paired-path behavior must be identical.
        rng = _rng(51)
        rows = []
        for m in ["A", "B", "C"]:
            for i in range(20):
                rows.append({"model": m, "item": i,
                             "score": float(np.clip(rng.normal(0.5, 0.15), 0, 1))})
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        r_default = es.compare(evaldata, factors="model", metric="score")
        r_explicit_false = es.compare(evaldata, factors="model", metric="score",
                                       p_values=False, omnibus=False)
        assert r_default.to_dict() == r_explicit_false.to_dict()


class TestCompareDesignRouting:
    def test_design_auto_on_paired_data_is_unchanged(self):
        rows = []
        rng = _rng(3)
        for m in ["A", "B"]:
            for i in range(30):
                rows.append({"model": m, "item": i, "score": float(np.clip(rng.normal(0.5, 0.15), 0, 1))})
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        r = es.compare(evaldata, factors="model", metric="score")
        from evalstats.api import ComparisonResult
        assert isinstance(r, ComparisonResult)

    def test_design_auto_raises_on_unpaired_data(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        with pytest.raises(ValueError, match="between-subjects"):
            es.compare(evaldata, factors="model", metric="score")

    def test_design_unpaired_dispatches_to_group_comparison_result(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired")
        assert isinstance(r, GroupComparisonResult)

    def test_design_paired_forces_old_path_on_unpaired_data(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        # Forcing the paired path on genuinely disjoint items must still hit the
        # existing (pre-existing, unchanged) has_missing crash -- not a new error.
        with pytest.raises(ValueError, match="NaN"):
            es.compare(evaldata, factors="model", metric="score", design="paired")

    def test_design_unpaired_not_supported_for_factorial(self):
        rng = _rng(4)
        rows = []
        for m in ["A", "B"]:
            for p in ["p1", "p2"]:
                for i in range(20):
                    rows.append({"model": m, "prompt": p, "item": f"{m}_{p}_{i}",
                                 "score": float(np.clip(rng.normal(0.5, 0.15), 0, 1))})
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        with pytest.raises(ValueError, match="not supported"):
            es.compare(evaldata, factors=["model", "prompt"], metric="score", design="unpaired")

    def test_design_unpaired_not_supported_for_lmm(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        with pytest.raises(ValueError, match="not supported"):
            es.compare(evaldata, factors="model", metric="score", design="unpaired", method="lmm")

    def test_design_auto_exempt_for_lmm_on_unpaired_data(self):
        df = _make_unpaired_df({"A": 0.4, "B": 0.6})
        evaldata = es.load_from(df)
        # method="lmm" tolerates disjoint items natively -- design="auto" must not
        # raise the between-subjects ValueError for this call.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = es.compare(evaldata, factors="model", metric="score", method="lmm")
        from evalstats.api import ComparisonResult
        assert isinstance(r, ComparisonResult)

    def test_design_unpaired_with_secondary_metric_runs_pareto(self):
        # secondary_metric= is supported under design="unpaired" -- see
        # TestCompareUnpairedPareto for the full engine-level coverage; this
        # just confirms compare()'s own dispatch threads it through.
        rng = _rng(6)
        rows = []
        for m, mean in [("A", 0.4), ("B", 0.7)]:
            for i in range(30):
                rows.append({"model": m, "item": f"{m}_{i}",
                             "score": float(np.clip(rng.normal(mean, 0.15), 0, 1)),
                             "latency_ms": float(rng.normal(100, 10))})
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        r = es.compare(evaldata, factors="model", metric="score", design="unpaired",
                        secondary_metric={"latency_ms": "min"}, rng=6)
        assert r.pareto is not None

    def test_design_unpaired_with_multirun_data_not_supported(self):
        rng = _rng(8)
        rows = []
        for m in ["A", "B"]:
            for i in range(20):
                item_noise = rng.normal(0, 0.1)
                for run in range(3):
                    rows.append({
                        "model": m, "item": f"{m}_{i}", "run": run,
                        "score": float(np.clip(0.5 + (0.15 if m == "B" else 0.0) + item_noise + rng.normal(0, 0.05), 0, 1)),
                    })
        df = pd.DataFrame(rows)
        evaldata = es.load_from(df)
        with pytest.raises(ValueError, match="multi-run"):
            es.compare(evaldata, factors="model", metric="score", design="unpaired")

        # Single-run (R=1) slice of the same data should work fine -- the guard
        # only fires when run_col genuinely has >1 distinct value.
        df_single_run = df[df["run"] == 0].drop(columns=["run"])
        evaldata_single = es.load_from(df_single_run)
        r = es.compare(evaldata_single, factors="model", metric="score", design="unpaired")
        assert isinstance(r, GroupComparisonResult)

    def test_design_unpaired_with_alignment_end_to_end(self):
        df = _make_unpaired_with_alignment({"A": 0.35, "B": 0.65})
        evaldata = es.load_from(df, col_map={"model": "model", "item": "item"})
        with pytest.warns(UserWarning):
            ar = judge_alignment(evaldata, llm_metric="llm_score", human_groundtruth="human_score")
        r = es.compare(evaldata, factors="model", metric="llm_score", design="unpaired",
                        alignment={"llm_score": ar})
        assert isinstance(r, GroupComparisonResult)
        assert r.ppi_applied is True
