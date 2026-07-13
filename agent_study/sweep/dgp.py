"""Maps (data_type, dgp_regime) to a concrete ShapeSpec + icc_spread flag
for simulations.harness.scenarios.synthetic.sample_group_truth.

dgp_regime is the axis this sweep adds on top of the existing agent_study
families: whether the data-generating process itself is the kind of thing
that breaks a naive, normality-assumed analysis (a plain t-test assuming iid
Gaussian, equal variance) -- as opposed to the ground_truth_regime axis
(true_null/borderline/clear_winner), which is about effect size, not shape.

Levels are restricted per data type rather than a full cross, since not all
of them are meaningful for every data type: binary data has no real concept
of skew or heavy tails (a Bernoulli is fully described by its rate), so it
only gets "clean". Likert (bounded, 1-5, discrete) gets skew and mixture
(both present in the existing shape catalog) but not a separate heavy-tail
regime, which would mostly duplicate what its bimodal/skewed shapes already
stress. Continuous gets the full 5-level treatment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulations.harness.scenarios.synthetic import SHAPES_BY_EVAL_TYPE, ShapeSpec

DATA_TYPES = ["binary", "continuous", "likert"]

# data_type -> ordered list of dgp_regime names this sweep runs for it.
DGP_REGIMES_BY_DATA_TYPE: dict[str, list[str]] = {
    "continuous": ["clean", "unequal_variance", "skewed", "heavy_tail", "mixture"],
    "likert": ["clean", "skewed", "mixture"],
    "binary": ["clean"],
}

# data_type -> valid score range, for baseline-level shifting (below) and
# for sizing that shift relative to each type's own scale.
DATA_TYPE_RANGE: dict[str, tuple[float, float]] = {
    "continuous": (0.0, 1.0),
    "binary": (0.0, 1.0),
    "likert": (1.0, 5.0),
}

# How far a cell's baseline can drift from a shape's own natural center, as a
# fraction of the type's total range. Without this, every item sits at
# whatever level its chosen shape happens to be centered at (~0.4-0.5 for our
# continuous/binary "clean" shapes, ~3.0 for likert-mid) -- realistic eval
# data spans the whole scale, and evalstats' bootstrap CI behavior isn't
# guaranteed to be uniform across it (variance mechanically shrinks near a
# bounded scale's edges), so a benchmark that only ever tests the middle
# can't tell you anything about the edges.
BASELINE_SHIFT_FRACTION = 0.4


def sample_baseline_shift(data_type: str, rng: np.random.Generator) -> float:
    lo, hi = DATA_TYPE_RANGE[data_type]
    span = (hi - lo) * BASELINE_SHIFT_FRACTION
    return float(rng.uniform(-span, span))


# Range each arm's own icc (reliability) is drawn from independently when
# icc_spread=True, giving genuine cross-arm variance heterogeneity -- see
# sample_icc_per_arm's docstring for why this replaced sample_group_truth's
# own heteroscedastic= flag for this purpose.
ICC_SPREAD_RANGE = (0.1, 0.5)


def sample_icc_per_arm(k: int, rng: np.random.Generator) -> list[float]:
    """A different icc drawn independently for every arm -- unlike
    sample_group_truth's heteroscedastic= flag (which scales noise by how
    close an item's value sits to the scale's boundary, so arms only end up
    with visibly different variance if they land at very different absolute
    positions), this creates real, position-independent variance
    differences between arms: verified empirically that heteroscedastic=True
    alone left arms at ~0.35 std regardless, while independently-varied icc
    produced std ranging 0.29-0.41 in a test. Drawn independently of
    winner_idx, so the winning arm isn't systematically more or less
    reliable than the rest -- only that arms genuinely differ from each
    other, which is what "unequal variance" is supposed to mean."""
    return rng.uniform(*ICC_SPREAD_RANGE, size=k).tolist()


@dataclass
class DGPConfig:
    shape: ShapeSpec
    icc_spread: bool = False


def _shape(data_type: str, label: str) -> ShapeSpec:
    return next(s for s in SHAPES_BY_EVAL_TYPE[data_type] if s.label == label)


def _outlier_contaminated_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    """~88% the same logit-normal base as "cont-logit-normal", ~12% slammed
    to an exact 0 or 1 -- the textbook "contaminated normal" robust-stats
    setup (a few catastrophic failures or perfect scores mixed into
    otherwise well-behaved data), and a pattern our own generated eval data
    already shows naturally at the boundaries, just turned up further."""
    base = 1.0 / (1.0 + np.exp(-rng.normal(-0.35, 1.35, size=n)))
    contaminated = rng.random(n) < 0.12
    n_contaminated = int(contaminated.sum())
    if n_contaminated:
        base[contaminated] = rng.choice([0.0, 1.0], size=n_contaminated)
    return base


_CONT_HEAVY_TAIL_SHAPE = ShapeSpec(
    "cont-heavy-tail-outlier", "continuous", "custom", custom_sampler=_outlier_contaminated_sampler,
)


def get_dgp_config(data_type: str, dgp_regime: str) -> DGPConfig:
    if dgp_regime not in DGP_REGIMES_BY_DATA_TYPE[data_type]:
        raise ValueError(f"dgp_regime {dgp_regime!r} not valid for data_type {data_type!r}")

    if data_type == "continuous":
        mapping = {
            "clean": (_shape("continuous", "cont-logit-normal"), False),
            "unequal_variance": (_shape("continuous", "cont-logit-normal"), True),
            "skewed": (_shape("continuous", "cont-right-skew"), False),
            "heavy_tail": (_CONT_HEAVY_TAIL_SHAPE, False),
            "mixture": (_shape("continuous", "cont-mixture"), False),
        }
    elif data_type == "likert":
        mapping = {
            "clean": (_shape("likert", "likert-mid"), False),
            "skewed": (_shape("likert", "likert-skewed-high"), False),
            "mixture": (_shape("likert", "likert-bimodal"), False),
        }
    else:  # binary
        mapping = {
            "clean": (_shape("binary", "p=0.50"), False),
        }

    shape, icc_spread = mapping[dgp_regime]
    return DGPConfig(shape=shape, icc_spread=icc_spread)
