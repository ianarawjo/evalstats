"""Tests for the global set_alpha_ci / get_alpha_ci configuration."""

import pytest
import numpy as np

import promptstats as ps
from promptstats.config import set_alpha_ci, get_alpha_ci


# ---------------------------------------------------------------------------
# Fixture: restore default alpha after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_alpha():
    """Reset the global alpha to its default (0.01) after every test."""
    original = get_alpha_ci()
    yield
    set_alpha_ci(original)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Basic config API
# ---------------------------------------------------------------------------

def test_set_alpha_ci_updates_get():
    set_alpha_ci(0.05)
    assert get_alpha_ci() == 0.05


def test_set_alpha_ci_rejects_zero():
    with pytest.raises(ValueError):
        set_alpha_ci(0.0)


def test_set_alpha_ci_rejects_one():
    with pytest.raises(ValueError):
        set_alpha_ci(1.0)


def test_set_alpha_ci_rejects_negative():
    with pytest.raises(ValueError):
        set_alpha_ci(-0.05)


def test_set_alpha_ci_rejects_greater_than_one():
    with pytest.raises(ValueError):
        set_alpha_ci(1.5)


# ---------------------------------------------------------------------------
# compare_prompts: stored alpha reflects global setting
# ---------------------------------------------------------------------------

def test_compare_prompts_stores_global_alpha():
    """report.alpha should reflect the global alpha when none is passed."""
    set_alpha_ci(0.03)
    report = ps.compare_prompts(
        {"a": [1, 1, 0, 1, 0], "b": [0, 0, 1, 0, 0]},
        method="fisher_exact",
        correction="none",
        rng=_rng(),
    )
    assert report.alpha == pytest.approx(0.03)


def test_compare_prompts_explicit_alpha_overrides_global():
    """An explicit alpha= kwarg should take precedence over set_alpha_ci."""
    set_alpha_ci(0.03)
    report = ps.compare_prompts(
        {"a": [1, 1, 0, 1, 0], "b": [0, 0, 1, 0, 0]},
        method="fisher_exact",
        correction="none",
        alpha=0.10,
        rng=_rng(),
    )
    assert report.alpha == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# compare_models: stored alpha reflects global setting
# ---------------------------------------------------------------------------

def test_compare_models_stores_global_alpha():
    """report.alpha should reflect the global alpha when none is passed."""
    set_alpha_ci(0.03)
    report = ps.compare_models(
        {
            "A": np.array([[1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
                           [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]]),
            "B": np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]),
        },
        n_bootstrap=500,
        rng=_rng(),
    )
    assert report.alpha == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Behavioral: alpha threshold changes significance outcome (unbeaten set)
# ---------------------------------------------------------------------------

def test_alpha_affects_unbeaten_in_compare_prompts():
    """Changing the significance threshold should change which prompts are 'unbeaten'.

    Uses the sign test (deterministic) with data that gives an exact p-value of
    2*(0.5)^6 = 0.03125: 6 non-tied inputs all won by 'a', 0 won by 'b'.
    This is significant at alpha=0.05 but not at alpha=0.01.
    """
    # a wins 6 of 10 inputs (1 tie at index 0, 3 ties at indices 7-9).
    # sign test: 6 positives, 0 negatives → p = 2*(0.5)^6 ≈ 0.03125
    scores = {
        "a": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "b": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    # Verify the known p-value (sanity check)
    ref = ps.compare_prompts(
        scores, method="sign_test", correction="none", simultaneous_ci=False, alpha=0.05, rng=_rng(),
    )
    p_val = ref.pairwise.get("a", "b").p_value
    assert p_val == pytest.approx(2 * (0.5 ** 6), abs=1e-9), (
        f"Unexpected sign-test p-value: {p_val}"
    )

    # Loose alpha (> p_val): a significantly beats b → only a is unbeaten.
    # Use simultaneous_ci=False to test the p-value-based significance path.
    report_loose = ps.compare_prompts(
        scores, method="sign_test", correction="none", simultaneous_ci=False, alpha=0.05, rng=_rng(),
    )
    assert report_loose.unbeaten == ["a"]

    # Strict alpha (< p_val): no significant edge → unbeaten is None
    report_strict = ps.compare_prompts(
        scores, method="sign_test", correction="none", simultaneous_ci=False, alpha=0.01, rng=_rng(),
    )
    assert report_strict.unbeaten is None


def test_global_alpha_affects_unbeaten_outcome():
    """set_alpha_ci should produce the same behavioral change as passing alpha= explicitly."""
    # Same dataset as above: sign-test p ≈ 0.03125 (significant at 0.05, not at 0.01)
    scores = {
        "a": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "b": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    # Via global: loose alpha → a is unbeaten.
    # Use simultaneous_ci=False to test the p-value-based significance path.
    set_alpha_ci(0.05)
    report_global_loose = ps.compare_prompts(
        scores, method="sign_test", correction="none", simultaneous_ci=False, rng=_rng(),
    )
    assert report_global_loose.unbeaten == ["a"]
    assert report_global_loose.alpha == pytest.approx(0.05)

    # Via global: strict alpha → no winner
    set_alpha_ci(0.01)
    report_global_strict = ps.compare_prompts(
        scores, method="sign_test", correction="none", simultaneous_ci=False, rng=_rng(),
    )
    assert report_global_strict.unbeaten is None
    assert report_global_strict.alpha == pytest.approx(0.01)
