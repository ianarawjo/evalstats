import numpy as np

import promptstats as ps
from promptstats.core.paired import pairwise_differences


def test_pairwise_permutation_detects_nonzero_paired_effect():
    rng = np.random.default_rng(123)
    n_inputs = 80

    a = rng.normal(loc=0.7, scale=0.15, size=n_inputs)
    b = a - 0.08 + rng.normal(loc=0.0, scale=0.02, size=n_inputs)
    scores = np.vstack([a, b])

    res = pairwise_differences(
        scores,
        0,
        1,
        "A",
        "B",
        method="permutation",
        n_bootstrap=4000,
        rng=np.random.default_rng(7),
    )

    assert "permutation" in res.test_method
    assert res.point_diff > 0
    assert res.p_value < 0.05


def test_pairwise_permutation_is_symmetric_when_swapped():
    scores = np.array(
        [
            [0.8, 0.9, 0.7, 0.85, 0.88, 0.81, 0.79, 0.84],
            [0.7, 0.8, 0.65, 0.74, 0.79, 0.73, 0.72, 0.77],
        ],
        dtype=float,
    )

    ab = pairwise_differences(
        scores,
        0,
        1,
        "A",
        "B",
        method="permutation",
        n_bootstrap=3000,
        rng=np.random.default_rng(11),
    )
    ba = pairwise_differences(
        scores,
        1,
        0,
        "B",
        "A",
        method="permutation",
        n_bootstrap=3000,
        rng=np.random.default_rng(11),
    )

    assert ab.p_value == ba.p_value
    assert ab.point_diff == -ba.point_diff


def test_analyze_accepts_permutation_method_with_runs():
    rng = np.random.default_rng(2026)
    n_templates, n_inputs, n_runs = 2, 30, 3

    scores = np.empty((n_templates, n_inputs, n_runs), dtype=float)
    base = rng.normal(0.6, 0.10, size=(n_inputs, n_runs))
    scores[0] = base + rng.normal(0.03, 0.02, size=(n_inputs, n_runs))
    scores[1] = base + rng.normal(-0.03, 0.02, size=(n_inputs, n_runs))

    result = ps.BenchmarkResult(
        scores=scores,
        template_labels=["A", "B"],
        input_labels=[f"i{i:02d}" for i in range(n_inputs)],
    )

    analysis = ps.analyze(
        result,
        method="permutation",
        n_bootstrap=2000,
        rng=np.random.default_rng(99),
    )

    pair = analysis.pairwise.get("A", "B")
    assert "permutation" in pair.test_method
    assert pair.n_runs == 3
