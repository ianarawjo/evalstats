import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer

import promptstats as ps
from promptstats.compare import CompareReport, EntityStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_compare_prompts_requires_dict():
    with pytest.raises(TypeError, match="dict"):
        ps.compare_prompts([[1, 0, 1], [1, 1, 1]])


def test_compare_prompts_requires_at_least_two_prompts():
    with pytest.raises(ValueError, match="at least 2"):
        ps.compare_prompts({"only_one": [1, 0, 1]})


def test_compare_prompts_mismatched_input_lengths():
    with pytest.raises(ValueError, match="same number of inputs"):
        ps.compare_prompts(
            {"a": [1, 0, 1], "b": [1, 1]},
            rng=_rng(),
        )


def test_compare_prompts_mixed_1d_2d_raises():
    with pytest.raises(ValueError, match="mix of 1-D and 2-D"):
        ps.compare_prompts(
            {"a": [1, 0, 1], "b": [[1, 0], [0, 1], [1, 1]]},
            rng=_rng(),
        )


def test_compare_prompts_mismatched_run_counts():
    with pytest.raises(ValueError, match="same number of runs"):
        ps.compare_prompts(
            {
                "a": [[1, 0, 1], [0, 1, 1]],   # R=3
                "b": [[1, 0], [0, 1]],           # R=2
            },
            rng=_rng(),
        )


def test_compare_prompts_wrong_ndim():
    with pytest.raises(ValueError, match="dimensions"):
        ps.compare_prompts(
            {"a": np.ones((3, 2, 2)), "b": np.ones((3, 2, 2))},
            rng=_rng(),
        )


@pytest.mark.parametrize("method", ["wilson", "newcombe", "fisher_exact"])
def test_compare_prompts_accepts_explicit_binary_methods(method: str):
    report = ps.compare_prompts(
        {
            "a": [1, 0, 1, 1, 0, 1, 0, 1],
            "b": [0, 0, 1, 0, 0, 1, 0, 0],
        },
        method=method,
        n_bootstrap=300,
        rng=_rng(101),
    )
    pair = report.pairwise.get("a", "b")
    if method == "fisher_exact":
        assert "fisher exact" in pair.test_method.lower()
    else:
        assert "newcombe" in pair.test_method


@pytest.mark.parametrize("method", ["wilson", "newcombe", "fisher_exact"])
def test_compare_prompts_explicit_binary_methods_reject_non_binary(method: str):
    with pytest.raises(ValueError, match=r"requires binary \(0/1\) data"):
        ps.compare_prompts(
            {
                "a": [0.1, 0.4, 0.8, 0.6, 0.2],
                "b": [0.2, 0.5, 0.7, 0.4, 0.3],
            },
            method=method,
            rng=_rng(102),
        )


def test_compare_prompts_accepts_sign_test_for_non_binary_data():
    report = ps.compare_prompts(
        {
            "a": [0.72, 0.75, 0.78, 0.74, 0.71, 0.73, 0.77, 0.76],
            "b": [0.66, 0.69, 0.70, 0.68, 0.65, 0.67, 0.71, 0.70],
        },
        method="sign_test",
        n_bootstrap=500,
        rng=_rng(103),
    )

    pair = report.pairwise.get("a", "b")
    assert "sign test" in pair.test_method.lower()


def test_friedman_nemenyi_all_ties_returns_stable_values():
    scores = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    result = ps.friedman_nemenyi(scores, ["a", "b", "c"])
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0, abs=1e-12)
    assert all(p == pytest.approx(1.0, abs=1e-12) for p in result.nemenyi_p.values())


def test_friedman_nemenyi_label_length_must_match_templates():
    scores = np.array(
        [
            [0.9, 0.8, 0.7],
            [0.8, 0.7, 0.6],
            [0.7, 0.6, 0.5],
        ]
    )
    with pytest.raises(ValueError, match="labels length"):
        ps.friedman_nemenyi(scores, ["a", "b"])


def test_friedman_nemenyi_matches_r_reference_values():
    """
        Example matrix and output from R's friedman.test and frdAllPairsNemenyiTest:
        > scores <- matrix(c(
        +   0.82, 0.79, 0.76,
        +   0.91, 0.88, 0.86,
        +   0.73, 0.71, 0.69,
        +   0.65, 0.67, 0.63,
        +   0.84, 0.82, 0.80,
        +   0.77, 0.74, 0.72
        + ), nrow = 6, byrow = TRUE)
        > 
        > colnames(scores) <- c("template_A","template_B","template_C")
        > 
        > # Convert to long format
        > df <- data.frame(
        +   score = as.vector(scores),
        +   template = factor(rep(colnames(scores), each=nrow(scores))),
        +   input = factor(rep(1:nrow(scores), times=ncol(scores)))
        + )
        > 
        > # Friedman test
        > friedman_res <- friedman.test(score ~ template | input, data=df)
        > 
        > # Nemenyi posthoc
        > nemenyi_res <- frdAllPairsNemenyiTest(score ~ template | input, data=df)
        > 
        > print(friedman_res)

            Friedman rank sum test

        data:  score and template and input
        Friedman chi-squared = 10.333, df = 2, p-value = 0.005704

        > print(nemenyi_res$p.value)
                    template_A template_B
        template_B 0.480432575         NA
        template_C 0.004269786  0.1072232
    """
    scores = np.array(
        [
            [0.82, 0.79, 0.76],
            [0.91, 0.88, 0.86],
            [0.73, 0.71, 0.69],
            [0.65, 0.67, 0.63],
            [0.84, 0.82, 0.80],
            [0.77, 0.74, 0.72],
        ],
        dtype=float,
    )
    labels = ["template_A", "template_B", "template_C"]

    # R matrix is (N inputs, k templates); API expects (k, N).
    result = ps.friedman_nemenyi(scores.T, labels)

    # We expect the Friedman statistic, df, and p-value to match R's output
    assert result.statistic == pytest.approx(10.333333333333329, abs=1e-6)
    assert result.df == 2
    assert result.p_value == pytest.approx(0.005703548998007417, abs=1e-6)

    # We expect the Nemenyi pairwise p-values to match R's output (with some tolerance for floating-point differences)
    assert result.get_nemenyi_p("template_A", "template_B") == pytest.approx(0.480432575, abs=1e-6)
    assert result.get_nemenyi_p("template_A", "template_C") == pytest.approx(0.004269786, abs=1e-6)
    assert result.get_nemenyi_p("template_B", "template_C") == pytest.approx(0.1072232, abs=1e-6)


# ---------------------------------------------------------------------------
# Return type and attribute shapes
# ---------------------------------------------------------------------------

def test_compare_prompts_returns_report():
    report = ps.compare_prompts(
        {"a": [1, 1, 0, 1, 0], "b": [1, 1, 1, 1, 0]},
        n_bootstrap=500,
        rng=_rng(),
    )
    assert isinstance(report, CompareReport)


def test_report_has_expected_attributes():
    report = ps.compare_prompts(
        {"a": [1, 1, 0, 1, 0], "b": [1, 1, 1, 1, 0]},
        n_bootstrap=500,
        rng=_rng(),
    )
    assert set(report.labels) == {"a", "b"}
    assert set(report.means.keys()) == {"a", "b"}
    assert set(report.prompt_stats.keys()) == {"a", "b"}
    assert report.unbeaten in (None, ["a"], ["b"])
    assert isinstance(report.significant, bool)
    assert isinstance(report.quick_summary(), str) and len(report.quick_summary()) > 0
    assert isinstance(report.full_analysis, ps.AnalysisBundle)


def test_report_means_are_correct():
    report = ps.compare_prompts(
        {"a": [0.0, 1.0, 0.0], "b": [1.0, 1.0, 1.0]},
        n_bootstrap=500,
        rng=_rng(),
    )
    assert report.means["a"] == pytest.approx(1 / 3, abs=1e-9)
    assert report.means["b"] == pytest.approx(1.0, abs=1e-9)


def test_report_plot_bars_adds_ci_error_bars():
    report = ps.compare_prompts(
        {
            "a": [1, 0, 1, 1, 0, 1, 0, 1],
            "b": [1, 1, 1, 1, 0, 1, 1, 1],
        },
        n_bootstrap=300,
        rng=_rng(17),
    )

    fig = report.plot_bars(as_percent=False)

    assert fig is not None
    assert any(isinstance(c, ErrorbarContainer) for c in fig.axes[0].containers)


def test_report_plot_bars_can_disable_error_bars():
    report = ps.compare_prompts(
        {
            "a": [1, 0, 1, 1, 0, 1, 0, 1],
            "b": [1, 1, 1, 1, 0, 1, 1, 1],
        },
        n_bootstrap=300,
        rng=_rng(23),
    )

    fig = report.plot_bars(as_percent=False, error_bars=False)

    assert fig is not None
    assert not any(isinstance(c, ErrorbarContainer) for c in fig.axes[0].containers)


# ---------------------------------------------------------------------------
# prompt_stats: PromptStats per template
# ---------------------------------------------------------------------------

def test_prompt_stats_fields_present():
    report = ps.compare_prompts(
        {"a": [0.8, 0.9, 0.7, 0.85] * 5, "b": [0.85, 0.92, 0.75, 0.88] * 5},
        n_bootstrap=500,
        rng=_rng(),
    )
    for label in ("a", "b"):
        s = report.prompt_stats[label]
        assert isinstance(s, EntityStats)
        assert isinstance(s.mean, float)
        assert isinstance(s.median, float)
        assert isinstance(s.std, float)
        assert isinstance(s.ci_low, float)
        assert isinstance(s.ci_high, float)
        # CI should straddle the mean
        assert s.ci_low <= s.mean <= s.ci_high


def test_prompt_stats_ci_ordering():
    """ci_low < ci_high for all templates."""
    report = ps.compare_prompts(
        {"a": [0.8, 0.9, 0.7, 0.85] * 5, "b": [0.6, 0.5, 0.65, 0.55] * 5},
        n_bootstrap=500,
        rng=_rng(),
    )
    for label in report.labels:
        s = report.prompt_stats[label]
        assert s.ci_low < s.ci_high


def test_pairwise_p_values_populated_for_two_way():
    """2-way comparison: pairwise p-values should be accessible via PairwiseMatrix."""
    report = ps.compare_prompts(
        {"a": [0.8, 0.9, 0.7, 0.85] * 5, "b": [0.6, 0.5, 0.65, 0.55] * 5},
        n_bootstrap=500,
        rng=_rng(),
    )
    p = report.pairwise.get("a", "b").p_value
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_pairwise_p_values_match_pairwise_matrix():
    """pairwise p-value should be accessible via PairwiseMatrix both directions."""
    report = ps.compare_prompts(
        {"a": [0.8, 0.9, 0.7, 0.85] * 5, "b": [0.6, 0.5, 0.65, 0.55] * 5},
        n_bootstrap=500,
        rng=_rng(),
    )
    ab = report.pairwise.get("a", "b")
    ba = report.pairwise.get("b", "a")
    assert ab.p_value == pytest.approx(ba.p_value, abs=1e-12)


def test_pairwise_p_values_present_for_all_nway_pairs():
    """N-way comparison: pairwise results should be present for all pairs."""
    report = ps.compare_prompts(
        {
            "a": [0.8, 0.9, 0.7] * 5,
            "b": [0.6, 0.5, 0.65] * 5,
            "c": [0.7, 0.75, 0.68] * 5,
        },
        n_bootstrap=500,
        rng=_rng(),
    )
    assert len(report.pairwise.results) == 3


def test_get_pairwise_p_values_works_both_directions():
    report = ps.compare_prompts(
        {"a": [0.8, 0.9, 0.7, 0.85] * 5, "b": [0.6, 0.5, 0.65, 0.55] * 5},
        n_bootstrap=500,
        rng=_rng(),
    )
    ab = report.pairwise.get("a", "b")
    ba = report.pairwise.get("b", "a")
    assert ab.p_value == pytest.approx(ba.p_value, abs=1e-12)


# ---------------------------------------------------------------------------
# Two-prompt comparisons
# ---------------------------------------------------------------------------

def test_significant_difference_detected():
    """Large mean difference on many inputs should yield a significant result."""
    rng = _rng(42)
    a_scores = rng.normal(loc=0.5, scale=0.1, size=100)
    b_scores = rng.normal(loc=0.8, scale=0.1, size=100)

    report = ps.compare_prompts(
        {"a": a_scores, "b": b_scores},
        n_bootstrap=2_000,
        rng=_rng(7),
    )
    assert report.unbeaten == ["b"]
    assert report.significant is True
    assert report.pairwise.get("a", "b").p_value < 0.05


def test_no_significant_difference_detected():
    """Identical scores should yield no unbeaten (all tied)."""
    scores = [0.8, 0.7, 0.9, 0.6, 0.8]
    report = ps.compare_prompts(
        {"a": scores, "b": scores},
        n_bootstrap=500,
        rng=_rng(),
    )
    assert report.unbeaten is None
    assert report.significant is False


def test_alpha_controls_winner_threshold():
    """With a very strict alpha the same data should not produce a top tier."""
    rng = _rng(42)
    a = rng.normal(0.5, 0.1, 50)
    b = rng.normal(0.55, 0.1, 50)  # small difference

    report_strict = ps.compare_prompts(
        {"a": a, "b": b},
        alpha=0.001,
        n_bootstrap=1_000,
        rng=_rng(1),
    )
    p = report_strict.pairwise.get("a", "b").p_value
    assert p > report_strict.alpha or report_strict.unbeaten is None


# ---------------------------------------------------------------------------
# N-way comparisons (N > 2)
# ---------------------------------------------------------------------------

def test_three_way_comparison_returns_report():
    report = ps.compare_prompts(
        {
            "zero-shot": [0.80, 0.90, 0.70, 0.85, 0.75] * 4,
            "few-shot":  [0.75, 0.88, 0.65, 0.80, 0.70] * 4,
            "cot":       [0.82, 0.91, 0.73, 0.87, 0.78] * 4,
        },
        n_bootstrap=500,
        rng=_rng(),
    )
    assert len(report.labels) == 3
    assert len(report.pairwise.results) == 3


def test_three_way_single_winner_has_highest_mean():
    """When a single top-tier is declared it must be the highest-mean prompt."""
    rng = _rng(99)
    a = rng.normal(0.5, 0.05, 200)
    b = rng.normal(0.6, 0.05, 200)
    c = rng.normal(0.4, 0.05, 200)

    report = ps.compare_prompts(
        {"a": a, "b": b, "c": c},
        n_bootstrap=2_000,
        rng=_rng(3),
    )
    if report.unbeaten is not None and len(report.unbeaten) == 1:
        assert report.unbeaten[0] == max(report.means, key=report.means.get)


def test_winners_can_include_multiple_top_prompts():
    """Top prompts can tie with each other while beating lower prompts."""
    rng = _rng(123)
    a = rng.normal(0.80, 0.02, 160)
    b = rng.normal(0.80, 0.02, 160)
    c = rng.normal(0.60, 0.02, 160)

    report = ps.compare_prompts(
        {"a": a, "b": b, "c": c},
        n_bootstrap=2_000,
        rng=_rng(8),
    )

    assert report.unbeaten is not None
    assert set(report.unbeaten) == {"a", "b"}


# ---------------------------------------------------------------------------
# Multi-run (nested bootstrap)
# ---------------------------------------------------------------------------

def test_multirun_1d_equivalent_shape():
    """1-D and (M, 1) inputs should produce the same means."""
    scores_1d = {"a": [0.8, 0.7, 0.9], "b": [0.85, 0.75, 0.95]}
    scores_2d = {"a": [[0.8], [0.7], [0.9]], "b": [[0.85], [0.75], [0.95]]}

    r1 = ps.compare_prompts(scores_1d, n_bootstrap=200, rng=_rng())
    r2 = ps.compare_prompts(scores_2d, n_bootstrap=200, rng=_rng())

    assert r1.means == pytest.approx(r2.means, abs=1e-9)


def test_multirun_nested_bootstrap_activates():
    """R >= 3 runs should engage the nested bootstrap path (smoke test)."""
    rng = _rng(5)
    a = rng.normal(0.5, 0.1, (20, 3))
    b = rng.normal(0.7, 0.1, (20, 3))

    report = ps.compare_prompts(
        {"a": a, "b": b},
        n_bootstrap=500,
        rng=_rng(6),
    )
    assert report.full_analysis.benchmark.n_runs == 3
    assert isinstance(report.pairwise.get("a", "b").p_value, float)


def test_multirun_mean_matches_flattened():
    """Reported mean should equal the mean of per-input cell means."""
    a_runs = np.array([[0.8, 0.9, 0.7], [0.6, 0.5, 0.7]])  # (2 inputs, 3 runs)
    b_runs = np.array([[0.9, 0.8, 0.85], [0.7, 0.65, 0.72]])

    report = ps.compare_prompts(
        {"a": a_runs, "b": b_runs},
        n_bootstrap=200,
        rng=_rng(),
    )
    expected_a = float(np.mean(a_runs.mean(axis=1)))
    expected_b = float(np.mean(b_runs.mean(axis=1)))
    assert report.means["a"] == pytest.approx(expected_a, abs=1e-9)
    assert report.means["b"] == pytest.approx(expected_b, abs=1e-9)


# ---------------------------------------------------------------------------
# Summary and print (smoke tests)
# ---------------------------------------------------------------------------

def test_summary_is_nonempty_string():
    report = ps.compare_prompts(
        {"a": [1, 0, 1, 1, 0], "b": [1, 1, 1, 0, 1]},
        n_bootstrap=200,
        rng=_rng(),
    )
    assert isinstance(report.quick_summary(), str)
    assert len(report.quick_summary()) > 10


def test_summary_mentions_selected_statistic_and_correction():
    report = ps.compare_prompts(
        {"a": [1, 0, 1, 1, 0], "b": [1, 1, 1, 0, 1]},
        statistic="median",
        correction="bonferroni",
        n_bootstrap=200,
        rng=_rng(),
    )
    assert "Δmedian" in report.quick_summary() or "median=" in report.quick_summary()
    assert "bonferroni-corrected" in report.quick_summary()


def test_summary_delegates_to_compare_summary(capsys):
    """summary() should call print_compare_summary and produce focused output."""
    report = ps.compare_prompts(
        {"baseline": [0.8, 0.7, 0.9, 0.85] * 5, "new": [0.82, 0.75, 0.91, 0.87] * 5},
        n_bootstrap=200,
        rng=_rng(),
    )
    report.summary()
    captured = capsys.readouterr()
    # print_compare_summary emits pairwise and executive summary sections
    assert "Pairwise Comparisons" in captured.out
    assert "Executive Summary" in captured.out


def test_print_is_alias_for_summary(capsys):
    report = ps.compare_prompts(
        {"baseline": [0.8, 0.7, 0.9, 0.85] * 5, "new": [0.82, 0.75, 0.91, 0.87] * 5},
        n_bootstrap=200,
        rng=_rng(),
    )
    report.print()
    captured = capsys.readouterr()
    assert "Pairwise Comparisons" in captured.out
    assert "Executive Summary" in captured.out


# ---------------------------------------------------------------------------
# Pairwise access
# ---------------------------------------------------------------------------

def test_pairwise_get_works_both_directions():
    report = ps.compare_prompts(
        {"a": [1, 0, 1, 1], "b": [0, 1, 0, 1]},
        n_bootstrap=200,
        rng=_rng(),
    )
    ab = report.pairwise.get("a", "b")
    ba = report.pairwise.get("b", "a")
    assert ab.p_value == pytest.approx(ba.p_value, abs=1e-12)
    assert ab.point_diff == pytest.approx(-ba.point_diff, abs=1e-12)
