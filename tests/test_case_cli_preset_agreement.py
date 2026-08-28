"""A bare CLI run and the official preset must sweep the same eval types.

They did not: --eval-types defaulted to None in ci_paired/ci_single, which is
falsy, so the filter never applied and the run swept all of EVAL_TYPES --
including "grades", which every official preset deliberately excludes. The
symptom is not an error but a silently larger sweep: a multi-run pairwise
table in the paper carried 15 grades rows the official test would never have
produced.
"""
import argparse
import pytest

from simulations.harness.scenarios import DEFAULT_EVAL_TYPES, EVAL_TYPES


CASES = [
    ("ci_paired", "nested_official_args"),
    ("ci_paired", "official_args"),
    ("ci_single", "official_args"),
    ("compare_e2e", "official_args"),
]


def _load(case):
    import importlib
    return importlib.import_module(f"simulations.harness.cases.{case}")


@pytest.mark.parametrize("case,preset_name", CASES)
def test_cli_eval_types_default_matches_preset(case, preset_name):
    mod = _load(case)
    preset = getattr(mod, preset_name, None)
    if preset is None:
        pytest.skip(f"{case} has no {preset_name}")
    args = preset(42)
    if getattr(args, "eval_types", None) is None:
        pytest.skip(f"{case}.{preset_name} does not pin eval_types")

    parser = argparse.ArgumentParser()
    mod.add_arguments(parser)
    cli = parser.parse_args([]).eval_types
    assert cli is not None, f"{case}: --eval-types defaults to None, so a bare run sweeps all of EVAL_TYPES"
    assert list(cli) == list(args.eval_types), (
        f"{case}: CLI default {list(cli)} != {preset_name} {list(args.eval_types)}"
    )


@pytest.mark.parametrize("case,_p", CASES)
def test_grades_is_never_a_default(case, _p):
    """'grades' stays opt-in: it is continuous rescaled, and no official
    preset sweeps it."""
    mod = _load(case)
    parser = argparse.ArgumentParser()
    mod.add_arguments(parser)
    cli = parser.parse_args([]).eval_types
    assert "grades" not in list(cli or []), f"{case} sweeps grades by default"


def test_default_eval_types_is_a_strict_subset_of_eval_types():
    assert set(DEFAULT_EVAL_TYPES) < set(EVAL_TYPES)
    assert "grades" in EVAL_TYPES and "grades" not in DEFAULT_EVAL_TYPES
