"""Real-data unpaired (between-subjects) sources for cases/ci_unpaired.py.

Built from the HUMAN LABELS of the judge-bias corpora, not the judge scores.
Each corpus stores one human label per item, so splitting items into two
disjoint sets gives a genuine two-independent-groups structure -- the same
construction ``real_judge_bias.generate_real_twogroup_null_cell`` already
uses ("only independent-samples tests apply to this structure").

What this buys over synthetic sources: real score marginals. Human ratings
are lumpy, skewed, ceiling/floor-heavy and coarsely quantised in ways the
synthetic shape catalog only approximates, and a CI method that holds up on
a smooth Beta can still fail on a real 1-5 star distribution with 40% of its
mass on "5".

Coverage of the three eval types (all from human labels):

    binary      iclr_metareview (4,000 items, p=0.283)
                appstore stars>=4 (1,200 items, 4 disjoint apps)
    continuous  privacy_judge (250 items, ~[1, 5] continuous ratings)
    likert      appstore star ratings (1,200 items, 1-5 integer, 4 apps)

Null AND non-null sources, deliberately
---------------------------------------
Null sources draw both groups from the SAME pool, so the true difference is
exactly 0 -- no Monte Carlo estimate, no finite-population correction, no
approximation anywhere in the coverage target.

But an all-null real-data catalog would repeat a mistake this project has
already made once: a coverage catalog swept only at the null hid paired-PPI
defects that only appear once the estimand moves off zero. So every eval
type also gets non-null sources built from a REAL covariate:

    appstore      split by app_id -- 6 pairs, true diffs -0.24 to +0.21
                  (likert), and +0.010 to +0.067 (binarised)
    privacy_judge split at the median judge_input length -- longer texts
                  really are rated more privacy-sensitive (2.72 vs 2.08,
                  true diff -0.645, d=0.75)

Why content-based splits are fine HERE
--------------------------------------
``cases/ppi_real.py`` explicitly rules out content-based splits (by language
pair, by app id) because for JUDGE-BIAS work a real content difference
contaminates the very signal being measured -- you can no longer tell judge
bias from a true group difference. That reasoning does not apply to this
case. ci_unpaired asks only whether an interval on a mean difference covers
its own truth; a real content difference is precisely the asset, because it
supplies a real, exactly computable, NONZERO target. Do not "fix" these
sources by removing them.

Sampling is WITH REPLACEMENT
----------------------------
``real_judge_bias`` draws without replacement, which is right when the
corpus IS the population of interest. Here it would corrupt the measurement:
drawing n=30 without replacement from privacy_judge's 250 items shrinks the
true sampling variance by a factor of 0.76, so an iid-based interval is ~14%
wider than it needs to be and posts over-coverage that looks like a
conservative method rather than a sampling artifact. Drawing with
replacement makes each pool an exact iid population whose mean is exactly
the coverage target.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import CIPairSource

DEFAULT_DATA_DIR = "simulations/out"

REAL_UNPAIRED_DATASETS = ["iclr_metareview", "privacy_judge", "appstore"]

_APPSTORE_REVIEWS = "appstore_scenario_reviews.csv"
_APPSTORE_POSITIVE_THRESHOLD = 4.0
"""A 4- or 5-star review counts as "positive" for the binarised source --
the usual split for star ratings, and it lands near p=0.44 here, which is
a more informative regime than a threshold that pushes p to an extreme."""

_PRIVACY_BOUNDS = (1.0, 5.0)
"""privacy_judge's human labels are continuous on an observed [1.04, 4.58].
The underlying instrument is a 1-5 rating averaged over raters, so [1, 5] is
the scale, not the observed min/max -- using the observed range instead
would put the extreme observations exactly on the boundary, where the
logit arm of mover_logit_t is undefined."""


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pool_source(
    label: str, eval_type: str, pool_a: np.ndarray, pool_b: np.ndarray,
    *, scale_bounds: tuple[float, float] | None = None, is_null: bool = False,
) -> CIPairSource:
    """Wrap two fixed pools of real scores as a CIPairSource.

    ``generate_pair`` draws n items from each pool independently and WITH
    replacement, so each pool acts as an exact iid population and
    ``true_diff`` (the difference of the pool means) is exactly the
    coverage target. ci_unpaired's ``_draw_unpaired`` calls this twice and
    keeps arm A from one call and arm B from the other, which is what makes
    the two groups share no items.
    """
    a_pool = np.asarray(pool_a, dtype=float)
    b_pool = np.asarray(pool_b, dtype=float)

    def _gen(rng: np.random.Generator, n: int, runs: int,
             _a: np.ndarray = a_pool, _b: np.ndarray = b_pool):
        a = rng.choice(_a, size=n, replace=True).reshape(n, 1)
        b = rng.choice(_b, size=n, replace=True).reshape(n, 1)
        if runs > 1:
            a = np.repeat(a, runs, axis=1)
            b = np.repeat(b, runs, axis=1)
        return a, b

    return CIPairSource(
        label=label, eval_type=eval_type, generate_pair=_gen,
        true_diff=float(a_pool.mean() - b_pool.mean()),
        source="real_unpaired", is_null=is_null, scale_bounds=scale_bounds,
    )


def _load_judge_bias_items(data_dir: str, dataset: str) -> tuple[np.ndarray, list[str]]:
    """Human labels (one per item) and the matching judge_input texts."""
    path = Path(data_dir) / f"judge_bias_{dataset}_items.csv"
    if not path.exists():
        return np.array([]), []
    rows = _read_rows(path)
    labels, texts = [], []
    for r in rows:
        raw = r.get("human_label", "")
        if raw in ("", None):
            continue
        labels.append(float(raw))
        texts.append(r.get("judge_input") or "")
    return np.asarray(labels, dtype=float), texts


def _load_appstore(data_dir: str) -> dict[str, np.ndarray]:
    """Star ratings grouped by app_id."""
    path = Path(data_dir) / _APPSTORE_REVIEWS
    if not path.exists():
        return {}
    by_app: dict[str, list[float]] = defaultdict(list)
    for r in _read_rows(path):
        raw = r.get("human_label", "")
        if raw in ("", None):
            continue
        by_app[r["app_id"]].append(float(raw))
    return {k: np.asarray(v, dtype=float) for k, v in sorted(by_app.items())}


def build_real_unpaired_sources(
    data_dir: str = DEFAULT_DATA_DIR,
    datasets: list[str] | None = None,
    include_null: bool = True,
) -> list[CIPairSource]:
    """Build every available real unpaired source. Missing corpora are
    skipped with a warning rather than raising, so the case still runs on a
    checkout that has only some of the data files."""
    wanted = set(datasets or REAL_UNPAIRED_DATASETS)
    sources: list[CIPairSource] = []

    # --- iclr_metareview: binary human labels, 4,000 items ---------------
    if "iclr_metareview" in wanted:
        labels, _ = _load_judge_bias_items(data_dir, "iclr_metareview")
        if labels.size == 0:
            print("  Warning: judge_bias_iclr_metareview_items.csv not found; skipping.")
        elif include_null:
            sources.append(_pool_source(
                f"iclr_metareview|null|N={labels.size}", "binary", labels, labels,
                scale_bounds=(0.0, 1.0), is_null=True,
            ))

    # --- privacy_judge: continuous human labels, 250 items ---------------
    if "privacy_judge" in wanted:
        labels, texts = _load_judge_bias_items(data_dir, "privacy_judge")
        if labels.size == 0:
            print("  Warning: judge_bias_privacy_judge_items.csv not found; skipping.")
        else:
            if include_null:
                sources.append(_pool_source(
                    f"privacy_judge|null|N={labels.size}", "continuous", labels, labels,
                    scale_bounds=_PRIVACY_BOUNDS, is_null=True,
                ))
            # Real covariate split: longer texts are rated more sensitive.
            lengths = np.array([len(t) for t in texts], dtype=float)
            med = float(np.median(lengths))
            short, long_ = labels[lengths <= med], labels[lengths > med]
            if short.size >= 20 and long_.size >= 20:
                sources.append(_pool_source(
                    f"privacy_judge|textlen|n={short.size}v{long_.size}",
                    "continuous", short, long_, scale_bounds=_PRIVACY_BOUNDS,
                ))

    # --- appstore: 1-5 star ratings, 4 disjoint apps ---------------------
    if "appstore" in wanted:
        by_app = _load_appstore(data_dir)
        if not by_app:
            print(f"  Warning: {_APPSTORE_REVIEWS} not found; skipping.")
        else:
            apps = list(by_app)
            all_stars = np.concatenate([by_app[a] for a in apps])
            all_bin = (all_stars >= _APPSTORE_POSITIVE_THRESHOLD).astype(float)
            if include_null:
                sources.append(_pool_source(
                    f"appstore_stars|null|N={all_stars.size}", "likert",
                    all_stars, all_stars, scale_bounds=(1.0, 5.0), is_null=True,
                ))
                sources.append(_pool_source(
                    f"appstore_pos|null|N={all_bin.size}", "binary",
                    all_bin, all_bin, scale_bounds=(0.0, 1.0), is_null=True,
                ))
            for i in range(len(apps)):
                for j in range(i + 1, len(apps)):
                    a_id, b_id = apps[i], apps[j]
                    sa, sb = by_app[a_id], by_app[b_id]
                    sources.append(_pool_source(
                        f"appstore_stars|{a_id}v{b_id}", "likert", sa, sb,
                        scale_bounds=(1.0, 5.0),
                    ))
                    sources.append(_pool_source(
                        f"appstore_pos|{a_id}v{b_id}", "binary",
                        (sa >= _APPSTORE_POSITIVE_THRESHOLD).astype(float),
                        (sb >= _APPSTORE_POSITIVE_THRESHOLD).astype(float),
                        scale_bounds=(0.0, 1.0),
                    ))

    if not sources:
        raise ValueError(
            f"No real unpaired sources could be built from {data_dir!r}. "
            f"Expected judge_bias_<dataset>_items.csv and/or {_APPSTORE_REVIEWS}."
        )
    return sources
