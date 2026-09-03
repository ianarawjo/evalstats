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

REAL_UNPAIRED_DATASETS = ["iclr_metareview", "privacy_judge", "appstore", "inspect", "socsci210"]

_APPSTORE_REVIEWS = "appstore_scenario_reviews.csv"
_APPSTORE_POSITIVE_THRESHOLD = 4.0
"""A 4- or 5-star review counts as "positive" for the binarised source --
the usual split for star ratings, and it lands near p=0.44 here, which is
a more informative regime than a threshold that pushes p to an extreme."""

_INSPECT_CSV = "inspect_benchmarks.csv"
_INSPECT_RUN_IDX = 0
"""Single run only. Averaging the 5 available runs per item would turn a
strictly 0/1 score into one of six levels, which is no longer binary and
would silently be scored by the wrong method family."""
_INSPECT_PAIRS_PER_BENCHMARK = 3
"""Model pairs kept per benchmark, chosen to span that benchmark's range of
true accuracy gaps (smallest, median, largest) rather than taking all 130
available pairs -- 9 benchmarks x 3 keeps the sweep proportionate while still
covering gaps from ~0 to 0.66."""

_SOCSCI_DIR = "socsci210"
"""Subdirectory of the data dir holding the SocSci210 parquet shards.

NOT redistributable with this repo and not committed: the HuggingFace dataset
card (socratesft/SocSci210) carries no license field, so we use it locally
without republishing any of it. Fetch it yourself into
``simulations/out/socsci210/`` -- that path is gitignored -- and these sources
appear; leave it absent and they are skipped with a warning. The underlying
experiments come from NSF's TESS programme, whose studies are individually
archived and citable, which is what a paper should cite rather than this
convenience packaging."""

_SOCSCI_COLUMNS = ["study_id", "task_num", "condition_num", "participant", "response"]
"""Only these are read. The corpus is ~1.45 GB almost entirely because of the
`stimuli`, `prompt` and `reasoning` free-text columns, none of which this case
needs -- projecting at read time keeps the load to a few hundred MB."""

_SOCSCI_MIN_PER_CONDITION = 400
"""Minimum participants per condition for an outcome to be usable.

Set well above what validity strictly requires. Sampling is WITH replacement,
so the pool is an exact iid population at any size and there is no
finite-population correction to worry about -- but `true_diff` is the POOL
mean, and a small pool estimates the real study population's mean poorly. A
117-participant pool pins a 5-level item's proportions to only about +-4.6%,
and at the largest sampled n=100 we would be drawing nearly the whole pool.
At 400 that tightens to about +-2.5% with 4x headroom, and it costs nothing:
115 between-subjects studies still qualify and only 15 are ever used."""
_SOCSCI_MAX_LEVELS = 11
"""Widest ordinal scale accepted (a 0-10 or 1-11 item). SocSci210's `response`
column mixes scales wildly -- its observed range spans -5 to 6e7, because some
outcomes are dollar amounts or counts rather than rating scales -- so anything
that is not a small integer scale is dropped rather than guessed at."""
_SOCSCI_MAX_SOURCES = 30
_SOCSCI_OUTCOMES_PER_STUDY = 1
"""At most one outcome per study, so the cap spreads across STUDIES rather
than being consumed by whichever study sorts first. The whole reason this
corpus is worth having is independent replication -- 30 outcomes drawn from
2 studies would be barely better than the single App Store corpus it is meant
to complement."""

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


def _load_inspect_pools(data_dir: str) -> dict[tuple[str, str], np.ndarray]:
    """(benchmark, model) -> that model's 0/1 item scores on that benchmark.

    Real LLM benchmark results, and the one corpus here that supplies
    genuinely INDEPENDENT replications: 9 separate benchmarks, ~1000 items
    each, rather than several contrasts carved out of a single dataset the
    way the App Store app pairs are.
    """
    path = Path(data_dir) / _INSPECT_CSV
    if not path.exists():
        return {}
    pools: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in _read_rows(path):
        if int(float(r.get("run_idx", 0))) != _INSPECT_RUN_IDX:
            continue
        raw = r.get("score", "")
        if raw in ("", None):
            continue
        pools[(r["benchmark"], r["model"])].append(float(raw))
    return {k: np.asarray(v, dtype=float) for k, v in pools.items() if len(v) >= 200}


def _inspect_sources(data_dir: str, include_null: bool) -> list[CIPairSource]:
    """Null and non-null binary sources from the Inspect benchmark corpus.

    Two models' scores on the same benchmark, drawn independently, is a
    genuine unpaired contrast whose true difference is the two models' real
    accuracy gap. Drawing each arm's indices independently makes the two
    sample means independent regardless of how correlated the models are
    item-by-item, so the per-item correlation that makes this corpus PAIRED
    for ci_paired does not leak into this construction.
    """
    pools = _load_inspect_pools(data_dir)
    if not pools:
        print(f"  Warning: {_INSPECT_CSV} not found; skipping inspect sources.")
        return []
    by_bench: dict[str, list[str]] = defaultdict(list)
    for (bench, model) in pools:
        by_bench[bench].append(model)

    out: list[CIPairSource] = []
    for bench in sorted(by_bench):
        models = sorted(by_bench[bench])
        if include_null:
            # Both arms from ONE model's pool: true difference is exactly 0.
            pool = pools[(bench, models[0])]
            out.append(_pool_source(
                f"inspect_{bench}|null|N={pool.size}", "binary", pool, pool,
                scale_bounds=(0.0, 1.0), is_null=True,
            ))
        cand = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                pa, pb = pools[(bench, models[i])], pools[(bench, models[j])]
                cand.append((abs(pa.mean() - pb.mean()), models[i], models[j]))
        cand.sort()
        if not cand:
            continue
        # Smallest, median and largest true gap on this benchmark.
        picks = {0, len(cand) // 2, len(cand) - 1}
        for idx in sorted(picks)[:_INSPECT_PAIRS_PER_BENCHMARK]:
            _gap, ma, mb = cand[idx]
            out.append(_pool_source(
                f"inspect_{bench}|{ma.split('/')[-1]}v{mb.split('/')[-1]}", "binary",
                pools[(bench, ma)], pools[(bench, mb)], scale_bounds=(0.0, 1.0),
            ))
    return out


def _socsci_sources(data_dir: str, include_null: bool) -> list[CIPairSource]:
    """Between-subjects human-subjects sources from SocSci210 (NSF TESS).

    The point of this corpus for us is provenance: every other real source
    here is AI-eval or product-review data, whereas between-subjects designs
    are overwhelmingly an HCI/social-science practice, so a recommendation
    aimed at that setting should be checked against actual human-subjects
    experiments.

    A (study, outcome) is kept only when it is genuinely between-subjects --
    every participant appears under exactly one condition within the study --
    and its responses form a small integer scale. Both are verified from the
    data rather than assumed, since the corpus mixes between- and
    within-subjects studies and mixes rating scales with unbounded numeric
    outcomes.
    """
    import pandas as pd

    root = Path(data_dir) / _SOCSCI_DIR
    shards = sorted(root.rglob("*.parquet"))
    if not shards:
        print(f"  Warning: no parquet files under {root}; skipping socsci210 sources. "
              f"Download socratesft/SocSci210 into that directory to enable them.")
        return []

    frames = []
    for f in shards:
        try:
            frames.append(pd.read_parquet(f, columns=_SOCSCI_COLUMNS))
        except Exception as exc:  # a shard with an unexpected schema shouldn't kill the run
            print(f"  Warning: could not read {f.name} ({exc}); skipping that shard.")
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["response", "condition_num", "participant", "study_id", "task_num"])

    # Between-subjects check, per STUDY: a participant must sit in one condition.
    per_study = df.groupby("study_id").participant.nunique()
    cond_per_p = df.groupby(["study_id", "participant"]).condition_num.nunique()
    between = {sid for sid, sub in cond_per_p.groupby(level=0) if int(sub.max()) == 1}
    print(f"  socsci210: {len(shards)} shards, {df.study_id.nunique()} studies, "
          f"{len(between)} between-subjects ({int(per_study.sum())} participant-study rows)")

    # Score every candidate outcome first, then take the best one per study,
    # so the source cap spans as many independent studies as possible.
    cands: dict[str, list] = defaultdict(list)
    for (sid, task), sub in df[df.study_id.isin(between)].groupby(["study_id", "task_num"], sort=True):
        vals = sub.response.to_numpy(dtype=float)
        uniq = np.unique(vals)
        if uniq.size < 2 or uniq.size > _SOCSCI_MAX_LEVELS:
            continue
        if not np.allclose(uniq, np.round(uniq)):
            continue                      # not an ordinal rating scale
        # A rating scale's levels are CONSECUTIVE integers. Requiring that
        # rejects items where sentinel codes masquerade as scale points --
        # study 5an26's outcome 3 reports levels [1,2,3,4,5,1000,2000,3000,
        # 4000,5000], i.e. a 5-point rating plus refused/don't-know codes,
        # and passed a levels-count filter easily. Left in, it was the single
        # source of mover_logit_t's worst-case Likert coverage (0.833), which
        # would have read as a defect in the method rather than in the data.
        # Two-level items are exempt: a binary outcome is binary whatever
        # codes it uses, and is rescaled onto {0, 1} below.
        if uniq.size > 2 and not np.allclose(np.diff(uniq), 1.0):
            continue
        lo_v, hi_v = float(uniq.min()), float(uniq.max())
        pools = {int(c): g.response.to_numpy(dtype=float)
                 for c, g in sub.groupby("condition_num")
                 if len(g) >= _SOCSCI_MIN_PER_CONDITION}
        if len(pools) < 2:
            continue
        # A two-level item is a binary outcome however it happens to be coded
        # (TESS uses 1/2 as often as 0/1), so rescale it onto {0, 1} -- the
        # binary interval methods threshold at 0.5 and would read 1/2 coding
        # as "every response is a success".
        if uniq.size == 2:
            eval_type, bounds = "binary", (0.0, 1.0)
            pools = {c: (v - lo_v) / (hi_v - lo_v) for c, v in pools.items()}
        else:
            eval_type, bounds = "likert", (lo_v, hi_v)
        conds = sorted(pools)
        gap, a, b = max((abs(pools[x].mean() - pools[y].mean()), x, y)
                        for i, x in enumerate(conds) for y in conds[i + 1:])
        cands[sid].append((gap, int(task), eval_type, bounds, pools, a, b))

    out: list[CIPairSource] = []
    for sid in sorted(cands):
        if len(out) >= _SOCSCI_MAX_SOURCES:
            break
        for gap, task, eval_type, bounds, pools, a, b in sorted(
                cands[sid], reverse=True)[:_SOCSCI_OUTCOMES_PER_STUDY]:
            tag = f"socsci{sid}t{task}"
            if include_null:
                p0 = pools[sorted(pools)[0]]
                out.append(_pool_source(f"{tag}|null|N={p0.size}", eval_type, p0, p0,
                                        scale_bounds=bounds, is_null=True))
            out.append(_pool_source(f"{tag}|c{a}vc{b}", eval_type, pools[a], pools[b],
                                    scale_bounds=bounds))
    print(f"  socsci210: built {len(out)} sources")
    return out


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

    if "inspect" in wanted:
        sources += _inspect_sources(data_dir, include_null)

    if "socsci210" in wanted:
        sources += _socsci_sources(data_dir, include_null)

    if not sources:
        raise ValueError(
            f"No real unpaired sources could be built from {data_dir!r}. "
            f"Expected judge_bias_<dataset>_items.csv and/or {_APPSTORE_REVIEWS}."
        )
    return sources
