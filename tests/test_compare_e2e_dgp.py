"""compare_e2e's non-PPI arm must draw from the SAME generator as the
multi-arm scenarios cases/pvalues.py sweeps.

This is the claim the paper makes to justify reading compare_e2e's non-PPI
columns next to the ci_paired / simultaneous-CI sweeps: they are the same
data-generating process, differing only in what is measured on top of it.
Before this was enforced, compare_e2e layered a second rater-noise pass over
sample_group_truth's own icc noise, which pushed the arm to
r(arm_i, arm_j)=0.37 -- below the lowest point (0.49) ci_paired's official icc
sweep validates -- and cost the Likert NIG interval ~1pp of coverage.
"""
import numpy as np
import pytest

from simulations.harness.cases import compare_e2e as C
from simulations.harness.scenarios.synthetic import (
    build_multiarm_sources, sample_group_truth,
)


@pytest.mark.parametrize("eval_type", ["binary", "continuous", "likert"])
@pytest.mark.parametrize("k", [3, 5])
def test_non_ppi_draw_matches_multiarm_ramp_source(eval_type, k):
    """Bit-for-bit: compare_e2e's non-PPI draw == build_multiarm_sources(ramp)."""
    shape = C.SHAPES_BY_EVAL_TYPE[eval_type][0]
    src = next(s for s in build_multiarm_sources(
        suite="standard", eval_types=[eval_type], icc=C.DEFAULT_ICC,
        effect_mode="ramp") if s.label == shape.label)
    step = C._cell_effect_step(eval_type, C._effect_frac_for(eval_type), k, C.DEFAULT_ICC)

    from_source = src.generate_scores(np.random.default_rng(7), 150, 1, k, step)[:, :, 0]
    from_case = sample_group_truth(
        shape, 150, 1, k, C.DEFAULT_ICC, np.random.default_rng(7),
        effects=np.arange(k, dtype=float) * step,
    )[:, :, 0]
    assert np.array_equal(from_source, from_case)


def test_non_ppi_cells_apply_no_judge_noise():
    """The no-PPI arm analyses the truth draw; only PPI cells get judge noise.

    Guards the actual branch in _run_cell rather than the generator: a
    regression there would reintroduce the second noise layer without
    breaking the equivalence test above.
    """
    shape = C.SHAPES_BY_EVAL_TYPE["likert"][0]
    seen = {}
    orig = C._apply_judge_noise

    def spy(truth, eval_type, rng, agreement_rate, biases=None):
        seen["called"] = seen.get("called", 0) + 1
        return orig(truth, eval_type, rng, agreement_rate, biases)

    C._apply_judge_noise = spy
    try:
        seen.clear()
        C._run_cell(eval_type="likert", shape=shape, k=3, n_items=40, ppi_frac=None,
                    is_null=True, n_reps=2, alpha=0.05, seed=1, n_bootstrap=200,
                    reference_estimator_k=None)
        # _reference_means_for still calls it once to compute llm_means; what
        # must NOT happen is a per-rep call on the analysed scores.
        assert seen.get("called", 0) <= 1, f"judge noise applied per-rep on the no-PPI arm: {seen}"
    finally:
        C._apply_judge_noise = orig


def test_icc_is_swept_on_non_ppi_arm_only():
    fracs = C._parse_ppi_fracs(["none", "0.20"])
    # n=250: at smaller n the PPI cells are filtered out entirely by
    # _ppi_applicable (n_lab < 15), leaving nothing to assert about.
    cells, _ = C.build_cells(["likert"], "standard", [3], [250], fracs,
                             icc_values=[0.05, 0.20, 0.60])
    non_ppi = {c["icc"] for c in cells if c["ppi_frac"] is None}
    ppi = {c["icc"] for c in cells if c["ppi_frac"] is not None}
    assert non_ppi == {0.05, 0.20, 0.60}
    assert ppi == {C.DEFAULT_ICC}


def test_csv_is_lossless_for_every_result_field():
    """Every CompareE2EResult field must appear in the saved CSV.

    The rate columns (coverage/type1/power) are derived and lossy; the raw
    counts and sums are what a plot regenerated from the CSV needs. A field
    added to the dataclass but not the writer silently produces a CSV whose
    plots cannot be rebuilt -- and, worse, rebuilds that quietly default the
    missing field rather than failing.
    """
    import dataclasses, re
    from simulations.harness.cases import compare_e2e as ce

    src = open(ce.__file__).read()
    header = re.search(r"csv_path\.open.*?writer\.writerow\(\[(.*?)\]\)", src, re.S)
    assert header, "could not locate the results CSV header"
    cols = set(re.findall(r'"([A-Za-z_0-9]+)"', header.group(1)))
    fields = {f.name for f in dataclasses.fields(ce.CompareE2EResult)}
    assert not (fields - cols), f"fields missing from results CSV: {sorted(fields - cols)}"


def test_float_columns_are_written_as_plain_numbers():
    """repr() on a numpy scalar emits 'np.float64(...)', which pandas reads
    back as a string, not a number -- a rebuild then silently coerces it to a
    default and the regenerated plot differs from the run's own."""
    import re
    from simulations.harness.cases import compare_e2e as ce

    src = open(ce.__file__).read()
    row = re.search(r"csv_path\.open.*?for r in results:(.*?)\n\n", src, re.S)
    assert row, "could not locate the results CSV row writer"
    bare = re.findall(r"repr\(r\.[a-z_]+\)", row.group(1))
    assert not bare, f"repr() on a possibly-numpy field: {bare}; wrap in float()"


def test_cli_default_and_official_preset_agree_on_icc_sweep():
    """A bare CLI run and the official preset must sweep the same icc values.

    They briefly disagreed: official_args carried the sweep while the
    argparse default was None, so a full-grid CLI run silently produced a
    single-icc grid -- valid numbers, but missing the robustness check the
    sweep exists for, and with nothing in the output saying so.
    """
    import argparse
    from simulations.harness.cases import compare_e2e as ce

    parser = argparse.ArgumentParser()
    ce.add_arguments(parser)
    cli = parser.parse_args([])
    preset = ce.official_args(42)
    assert list(cli.icc_values) == list(preset.icc_values) == list(ce.DEFAULT_ICC_VALUES)


def test_effect_ramp_span_is_capped_above_k3():
    """The arm-0..arm-(k-1) ramp must not outgrow the scale as k rises.

    An uncapped `arange(k)*step` put the k=10 continuous ramp at 1.031 on a
    [0, 1] scale, saturating the ceiling-hugging shapes and collapsing family
    coverage to ~79%. k=2/k=3 keep their previous effects exactly.
    """
    for eval_type in ("binary", "continuous", "likert"):
        frac = C._effect_frac_for(eval_type)
        base = C._effect_step_for(eval_type, frac)
        for k in (2, 3):
            assert C._cell_effect_step(eval_type, frac, k, C.DEFAULT_ICC) == base
        cap = C.MAX_EFFECT_SPAN_STEPS * base
        for k in (5, 10):
            span = (k - 1) * C._cell_effect_step(eval_type, frac, k, C.DEFAULT_ICC)
            assert span == pytest.approx(cap), (eval_type, k, span, cap)


def test_continuous_effect_holds_cohens_d_across_icc():
    """icc must not silently sweep the standardized effect size.

    continuous's `_group_noise_var` ADDS noise variance, so its realized SD
    moves with icc while likert/binary decompose a fixed total and stay flat.
    With a fixed denominator that made the icc axis a hidden effect-size axis,
    for continuous and only continuous.
    """
    # continuous genuinely needs the correction, and in the right direction:
    # lower icc = more added noise = larger realized SD = larger raw step.
    assert C._icc_sd_ratio("continuous", 0.05) > 1.2
    assert C._icc_sd_ratio("continuous", 0.60) < 0.8
    assert C._icc_sd_ratio("continuous", C.DEFAULT_ICC) == 1.0
    # ...and the correction actually equalizes d, which is the whole point.
    frac = C._effect_frac_for("continuous")
    ds = []
    for icc in (0.05, 0.20, 0.60):
        step = C._cell_effect_step("continuous", frac, 3, icc)
        ds.append(step / C._representative_sd_at_icc("continuous", icc))
    assert max(ds) - min(ds) < 0.01 * max(ds), ds
    # likert/binary are structurally invariant -> exactly a no-op there.
    for eval_type in ("likert", "binary"):
        for icc in (0.05, 0.20, 0.60):
            assert C._icc_sd_ratio(eval_type, icc) == 1.0
