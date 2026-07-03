import numpy as np

from evalstats.core.paired import pairwise_differences
from evalstats.core.ranking import bootstrap_ranks


def test_pairwise_differences_seeded_accepts_bootstrap_t() -> None:
    rng = np.random.default_rng(123)
    scores = rng.normal(size=(2, 48, 4))  # N templates, M inputs, R runs (R >= 3)

    result = pairwise_differences(
        scores,
        0,
        1,
        method="bootstrap_t",
        n_bootstrap=300,
        rng=np.random.default_rng(7),
    )

    assert np.isfinite(result.ci_low)
    assert np.isfinite(result.ci_high)
    assert result.test_method.startswith("nested bootstrap-t")


def test_bootstrap_ranks_accepts_bootstrap_t() -> None:
    rng = np.random.default_rng(456)
    scores = rng.normal(size=(3, 60))

    rank_dist = bootstrap_ranks(
        scores,
        ["A", "B", "C"],
        method="bootstrap_t",
        n_bootstrap=400,
        rng=np.random.default_rng(8),
    )

    assert rank_dist.rank_probs.shape == (3, 3)
    assert np.all(rank_dist.rank_probs >= 0.0)
    assert np.allclose(rank_dist.rank_probs.sum(axis=1), 1.0)
