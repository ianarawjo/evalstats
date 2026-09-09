"""Scoped test for JudgeBiasSource.truth_scale_b/truth_scale_c (see
simulations/out/PLAN_hetero_null_check.md Step 1.4). Run only this file:

    pytest simulations/test_hetero_truth_scale.py -q
"""
import numpy as np

from simulations.harness.scenarios import JudgeBiasSource
from simulations.harness.scenarios.synthetic import generate_judge_bias_cell

BASE = dict(
    eval_type="continuous", shape_label="cont-uniform", icc=0.20, n=200, n2=200, n3=200,
    label_frac=0.20, llm_noise=0.20, bias_type="none", effect_size=0.0,
)


def test_default_truth_scale_is_bitwise_unchanged():
    baseline = JudgeBiasSource(name="baseline", tag="t", **BASE)
    scaled = JudgeBiasSource(name="scaled", tag="t", **BASE, truth_scale_b=1.0, truth_scale_c=1.0)
    d1 = generate_judge_bias_cell(baseline, np.random.default_rng(7))
    d2 = generate_judge_bias_cell(scaled, np.random.default_rng(7))
    for field in ("truth_a2", "truth_b2", "truth_a3", "truth_b3", "truth_c3", "truth_x", "truth_y",
                  "truth_A", "truth_B", "truth_C", "llm_a2", "llm_b2", "lab_a2", "lab_b2"):
        np.testing.assert_array_equal(getattr(d1, field), getattr(d2, field))


def test_truth_scale_b_roughly_doubles_sample_sd():
    # Uses the default representative continuous shape (cont-right-skew),
    # not BASE's cont-uniform override -- cont-uniform is centered at 0.5
    # with little headroom before [0, 1] clipping, so a 2x stretch there
    # saturates hard (empirically ~1.4x, not ~2x). cont-right-skew sits
    # further from the boundary and reproduces the ~2x target within ~6%.
    n = 20_000
    scenario = JudgeBiasSource(
        name="scaled2x", tag="t", eval_type="continuous", icc=0.20, n=n, n2=n,
        label_frac=0.20, llm_noise=0.20, bias_type="none", effect_size=0.0, truth_scale_b=2.0,
    )
    data = generate_judge_bias_cell(scenario, np.random.default_rng(3))
    baseline_sd = np.std(data.truth_a2)
    scaled_sd = np.std(data.truth_b2)
    ratio = scaled_sd / baseline_sd
    assert abs(ratio - 2.0) < 0.10 * 2.0, f"expected ~2x SD, got ratio={ratio:.3f}"


def test_truth_scale_preserves_pairing_correlation():
    n = 20_000
    overrides = {**BASE, "n": n, "n2": n, "icc": 0.6}
    scenario = JudgeBiasSource(name="paired_scaled", tag="t", **overrides, truth_scale_b=2.0)
    data = generate_judge_bias_cell(scenario, np.random.default_rng(11))
    corr = np.corrcoef(data.truth_x, data.truth_y)[0, 1]
    assert corr > 0.3, f"expected the rescale to preserve strong positive pairing correlation, got {corr:.3f}"


def test_unsupported_eval_type_raises():
    scenario = JudgeBiasSource(name="binary_scaled", tag="t", eval_type="binary", n=50, truth_scale_b=2.0)
    try:
        generate_judge_bias_cell(scenario, np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for truth_scale_b on eval_type='binary'")
