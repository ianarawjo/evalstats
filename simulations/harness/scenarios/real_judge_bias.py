"""Real-data adapter for cases/ppi_real.py: loads the (human_label,
judge_score) pairs collected by simulations/collect_judge_bias_data.py --
one dataset per eval_type, one row per (item, judge_model, run) --
into the array shapes evalstats.tests' PPI-corrected functions expect.

Unlike scenarios/synthetic.py's JudgeBiasSource (which draws a fresh truth
distribution every replicate), a RealJudgeBiasCorpus already knows both
human_label and every judge's judge_score for every collected item -- there
is no "draw truth" step, only "draw which items this replicate sees" and
"reveal a label_frac fraction of their human labels". That second step
reuses the exact same MCAR masking convention
scenarios.synthetic._jb_labels_independent uses (NaN for unrevealed items,
floor of _MIN_LAB labeled items) -- just without synthetic's MNAR option,
which real data doesn't need to simulate since we're not testing the MNAR
mechanism itself here, only whether real judge-bias patterns break
calibration under ordinary (MCAR) sparse labeling.

A corpus can hold MULTIPLE judge models' scores at once, all aligned on the
shared item_id intersection (mirroring real_data.py's multiarm alignment,
which aligns ALL requested real models on shared item_id). This unlocks a
genuine PAIRED structure with no synthetic data-generating process needed:
any two judges scoring the SAME items are both noisy/biased reads of the
IDENTICAL human_label per item, so the true paired difference between them
is EXACTLY zero (stronger than generate_real_twogroup_null_cell's "equal in
distribution", since it's the SAME items on both sides) -- see
generate_real_paired_null_cell. With k judge models collected, all C(k, 2)
unique pairs are available as independent (dataset, label_frac, n, pair)
cells, each carrying that pair's own real noise/bias characteristics --
more judges means more such cells, i.e. more power to catch a paired-test
miscalibration than a single pair alone would give.

generate_real_twogroup_null_cell also reads its two groups through two
DIFFERENT judges (not the same judge reading both, which was tried first
and found to make an "uncorrected" comparison structurally unable to
fail -- see that function's docstring for the full reasoning), so both
the independent-groups and paired checks now depend on a corpus holding
multiple judge models to be meaningful, not just the paired one.

Unlike scenarios/real_data.py's Corpus (ground-truth accuracy corpora, no
judge in the loop at all -- OpenEval/Inspect benchmark scores), a
RealJudgeBiasCorpus is fundamentally about a JUDGE's relationship to human
labels, which is why this lives in its own file rather than folded into
real_data.py.

Sampling is without replacement (WOR), bounded by corpus size -- matching
real_data.py's corpus_to_ci_source convention (not corpus_to_shape_spec's
bootstrap one), since these are real, once-only human/judge comparisons and
we'd rather cap `n` at the corpus size than resample the same item twice
within one replicate.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = "simulations/out"

_MIN_LAB = 15  # same floor scenarios.synthetic._jb_labels_independent uses

# dataset key -> (eval_type, native (lo, hi) scale bounds to rescale onto [0, 1])
REAL_JUDGE_BIAS_DATASETS: dict[str, tuple[str, tuple[float, float]]] = {
    "arena": ("binary", (0.0, 1.0)),
    "wmt_da": ("continuous", (0.0, 100.0)),
    "appstore": ("likert", (1.0, 5.0)),
}


@dataclass
class RealJudgeBiasCorpus:
    """One dataset's full collected human_label + (possibly many judges')
    judge_score arrays, all rescaled to [0, 1] (matching
    scenarios.EVAL_TYPE_SCALE_BOUNDS -- the scale evalstats' PPI functions
    assume) so they're directly usable regardless of eval_type. Every
    judge_model's array is aligned to the SAME item order (the intersection
    of items every requested judge has scored), so judge_scores[a][i] and
    judge_scores[b][i] are always the same item for any two judges a, b.
    """
    dataset: str  # "arena" | "wmt_da" | "appstore"
    eval_type: str  # "binary" | "continuous" | "likert"
    judge_models: list[str]
    human_label: np.ndarray  # shape (N,), rescaled to [0, 1]
    judge_scores: dict[str, np.ndarray]  # judge_model -> shape (N,), rescaled to [0, 1], item-aligned
    corpus_size: int
    corpus_mean: float
    """True population mean of human_label (rescaled) -- the whole
    collected corpus stands in as "the population" (same convention
    real_data.py's Corpus.corpus_mean uses), and is what this module's PPI
    checks compare corrected/uncorrected estimates against."""


def _rescale(x: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    if lo == 0.0 and hi == 1.0:
        return x
    return (x - lo) / (hi - lo)


def load_real_judge_bias_corpus(
    dataset: str, *, data_dir: str = DEFAULT_DATA_DIR, judge_models: list[str] | None = None,
    min_coverage: float = 0.0,
) -> RealJudgeBiasCorpus:
    """Load simulations/out/judge_bias_<dataset>.csv (the merged
    item+judge-score view collect_judge_bias_data.py's collect-judge-scores
    step writes) into a RealJudgeBiasCorpus.

    `judge_models`, if given, restricts to that subset (all must be
    present, or this raises) -- an explicit request is trusted as-is, with
    no coverage filtering applied. Default (None) auto-discovers every
    judge model present in the file, THEN drops any whose coverage (its
    distinct-item count / the dataset's total item count, from
    _items.csv) is below `min_coverage`.

    This matters because alignment takes the INTERSECTION of items across
    every selected judge (mirroring real_data.py's
    build_openeval_multiarm_corpora) -- one incompletely-collected judge
    (e.g. a --limit'd or still-in-progress collect-judge-scores run) drags
    the usable corpus size down for EVERY judge, not just itself. Observed
    on real data: including one judge at ~65% coverage shrank a 5-judge
    arena corpus from 1241 aligned items down to 811, purely because of
    that one judge's gaps.
    """
    if dataset not in REAL_JUDGE_BIAS_DATASETS:
        raise ValueError(f"Unknown real judge-bias dataset {dataset!r}. Choices: {list(REAL_JUDGE_BIAS_DATASETS)}")
    eval_type, bounds = REAL_JUDGE_BIAS_DATASETS[dataset]

    path = Path(data_dir) / f"judge_bias_{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:\n"
            f"  python simulations/collect_judge_bias_data.py collect-data --types <type>\n"
            f"  python simulations/collect_judge_bias_data.py collect-judge-scores --types <type> "
            f"--backend <openrouter|ollama> --model <model>\n"
            f"first to produce it."
        )
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows -- run collect-judge-scores first (collect-data alone "
                          f"only fetches items + human labels, no judge scores).")

    models_present = sorted({r["judge_model"] for r in rows})
    if judge_models is None:
        if min_coverage > 0.0:
            items_path = Path(data_dir) / f"judge_bias_{dataset}_items.csv"
            with items_path.open(newline="", encoding="utf-8") as f:
                total_items = sum(1 for _ in csv.DictReader(f))
            items_by_model: dict[str, set[str]] = defaultdict(set)
            for r in rows:
                items_by_model[r["judge_model"]].add(r["item_id"])
            kept, dropped = [], []
            for jm in models_present:
                cov = len(items_by_model[jm]) / total_items if total_items else 0.0
                (kept if cov >= min_coverage else dropped).append((jm, cov))
            if dropped:
                print(f"  [{dataset}] excluding {len(dropped)} judge(s) below --min-judge-coverage "
                      f"{min_coverage:.0%}: " + ", ".join(f"{jm} ({cov:.0%})" for jm, cov in dropped))
            if not kept:
                raise ValueError(
                    f"No judge models in {path} meet min_coverage={min_coverage:.0%} of {total_items} "
                    f"items. Present (with coverage): "
                    f"{[(jm, f'{len(items_by_model[jm]) / total_items:.0%}') for jm in models_present]}"
                )
            judge_models = [jm for jm, _cov in kept]
        else:
            judge_models = models_present
    else:
        missing = [m for m in judge_models if m not in models_present]
        if missing:
            raise ValueError(f"judge_models {missing} not found in {path}. Present: {models_present}")

    # (item_id, judge_model) -> list of (human_label, judge_score) across runs
    by_item_judge: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        jm = r["judge_model"]
        if jm not in judge_models:
            continue
        by_item_judge[(r["item_id"], jm)].append((float(r["human_label"]), float(r["judge_score"])))

    all_items = sorted({item for (item, _jm) in by_item_judge.keys()})
    aligned_items = [item for item in all_items if all((item, jm) in by_item_judge for jm in judge_models)]
    if not aligned_items:
        raise ValueError(
            f"No items in {path} have scores from every requested judge model {judge_models} -- "
            f"pass a smaller judge_models= subset, or collect-judge-scores for the missing ones."
        )

    # human_label is identical across a given item's rows regardless of
    # judge_model/run_idx (it's the same ground truth) -- averaging is a
    # no-op, just a convenient way to collapse duplicate rows.
    human = np.array([
        np.mean([hl for hl, _ in by_item_judge[(item, judge_models[0])]]) for item in aligned_items
    ])
    judge_scores = {
        jm: np.array([np.mean([js for _, js in by_item_judge[(item, jm)]]) for item in aligned_items])
        for jm in judge_models
    }

    human = _rescale(human, bounds)
    judge_scores = {jm: _rescale(arr, bounds) for jm, arr in judge_scores.items()}

    return RealJudgeBiasCorpus(
        dataset=dataset, eval_type=eval_type, judge_models=list(judge_models),
        human_label=human, judge_scores=judge_scores, corpus_size=len(aligned_items),
        corpus_mean=float(np.mean(human)),
    )


def all_judge_pairs(
    corpus: RealJudgeBiasCorpus, *, max_pairs: int | None = None, rng: np.random.Generator | None = None,
) -> list[tuple[str, str]]:
    """Every unique (judge_a, judge_b) pair -- C(k, 2) for k judge models --
    for the paired-null check. If max_pairs caps the count below C(k, 2), a
    random subset is used instead (a practical safety valve: C(k, 2) grows
    quadratically, e.g. 8 judges -> 28 pairs, 20 judges -> 190)."""
    pairs = list(combinations(corpus.judge_models, 2))
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = rng or np.random.default_rng(0)
        keep = sorted(rng.choice(len(pairs), size=max_pairs, replace=False))
        pairs = [pairs[i] for i in keep]
    return pairs


def default_size_grid(corpus_size: int, n_points: int = 3) -> list[int]:
    """A small size grid bounded by (and including) the corpus's actual
    size, rather than one fixed grid applied blindly across very
    differently-sized corpora (e.g. arena's ~1300 items vs. app store's
    ~300)."""
    if corpus_size < _MIN_LAB:
        raise ValueError(f"Corpus too small ({corpus_size} < {_MIN_LAB}) to run any PPI check.")
    fracs = np.linspace(1.0 / n_points, 1.0, n_points)
    sizes = sorted({max(_MIN_LAB, int(round(corpus_size * f))) for f in fracs})
    return sizes


def _reveal_mask(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Indices to reveal (floored at _MIN_LAB) -- factored out of
    _reveal_labels so generate_real_wmt_paired_bias_cell can apply the SAME
    revealed-index set to both sides of a pair (an item's pair either gets
    both its real human labels revealed together or neither, mirroring
    generate_real_paired_null_cell's "one ground truth per item" framing)."""
    if n < _MIN_LAB:
        raise ValueError(f"Need n >= {_MIN_LAB} to reveal at least {_MIN_LAB} labels; got n={n}")
    n_lab = min(n, max(_MIN_LAB, int(round(n * frac))))
    return rng.choice(n, size=n_lab, replace=False)


def _reveal_labels(values: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """MCAR label-reveal: NaN out all but a `frac` fraction of `values`
    (floored at _MIN_LAB revealed items) -- the real-data analogue of
    scenarios.synthetic._jb_labels_independent, without its MNAR option."""
    idx = _reveal_mask(len(values), frac, rng)
    lab = np.full(len(values), np.nan)
    lab[idx] = values[idx]
    return lab


def generate_real_single_cell(
    corpus: RealJudgeBiasCorpus, rng: np.random.Generator, n: int, label_frac: float, judge_model: str,
) -> tuple[np.ndarray, np.ndarray]:
    """WOR-draw n items and reveal a label_frac fraction of their human
    labels -- the (llm, lab) shape evalstats.tests' _ppi_single_wilson /
    _ppi_single_bootstrap_t expect, for the single-sample bias/coverage
    check (does the PPI-corrected estimate of this real corpus's true mean,
    as seen through `judge_model`, recover it, with correct CI coverage)."""
    n = min(n, corpus.corpus_size)
    idx = rng.choice(corpus.corpus_size, size=n, replace=False)
    judge = corpus.judge_scores[judge_model][idx]
    human = corpus.human_label[idx]
    lab = _reveal_labels(human, label_frac, rng)
    return judge, lab


def generate_real_twogroup_null_cell(
    corpus: RealJudgeBiasCorpus, rng: np.random.Generator, n: int, label_frac: float,
    judge_a: str, judge_b: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """WOR-draw 2n DISJOINT items, split into two independent groups A/B --
    a valid Type-I null by construction (both groups are random subsamples
    of the SAME population, so their TRUE, human-label means are equal) --
    but group A is read through `judge_a` and group B through `judge_b`,
    two DIFFERENT judge models, not the same one reading both.

    This is a deliberate redesign (2026-07-23) from the original single-
    judge version, which scored both A and B with the SAME judge_model.
    That construction made the "uncorrected" comparison (a classical test
    run directly on judge scores, no human labels) structurally unable to
    fail: since A and B are random draws from the identical population read
    by the identical instrument, ANY judge bias -- however large -- is
    identical in expectation on both sides and cancels out of an A-vs-B
    difference completely. Confirmed directly on real data: a badly-biased,
    weakly-human-correlated judge (e.g. arena's judges correlate with human
    labels at only r=0.19-0.29) still passed the same-judge uncorrected
    check at essentially nominal Type-I error, which made the check
    uninformative about whether skipping PPI correction is actually risky --
    it could only ever demonstrate "there was nothing to correct here,"
    never illustrate a real failure mode the way the synthetic PPI sweep's
    twogroup checks do (see pvalues.py's confound/noise_family scenarios).

    Using two DIFFERENT real judges instead reintroduces a genuine
    asymmetry an uncorrected test IS vulnerable to: judge_a and judge_b
    generally have different bias/noise characteristics (confirmed
    directly on real data: arena's 4 judges range from +0.005 to +0.066
    mean bias on a [0,1]-rescaled scale, wmt_da's from -0.060 to +0.056),
    so a naive "just trust the judge" comparison between a judge_a-scored
    group and a judge_b-scored group can genuinely diverge even though the
    TRUE (human-label) difference is exactly zero -- the real-data
    analogue of a synthetic confound scenario, without fabricating one.
    Deliberately NOT a content-based split (e.g. by language pair or app
    id) -- see cases/ppi_real.py's module docstring for why that's
    explicitly out of scope (conflates a real content difference with
    judge bias, a dirtier signal).

    Only independent-samples tests apply to this structure (ttest,
    ttest_welch, mwu, mwu_mnar_experimental) -- for a genuine PAIRED
    structure (same items, not disjoint groups), see
    generate_real_paired_null_cell below, which already used two different
    judges from the start."""
    n = min(n, corpus.corpus_size // 2)
    idx = rng.choice(corpus.corpus_size, size=2 * n, replace=False)
    idx_a, idx_b = idx[:n], idx[n:]
    judge_a_scores = corpus.judge_scores[judge_a][idx_a]
    judge_b_scores = corpus.judge_scores[judge_b][idx_b]
    human_a, human_b = corpus.human_label[idx_a], corpus.human_label[idx_b]
    lab_a = _reveal_labels(human_a, label_frac, rng)
    lab_b = _reveal_labels(human_b, label_frac, rng)
    return judge_a_scores, judge_b_scores, lab_a, lab_b


def generate_real_paired_null_cell(
    corpus: RealJudgeBiasCorpus, rng: np.random.Generator, n: int, label_frac: float,
    judge_a: str, judge_b: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """WOR-draw n items ONCE (not 2n -- both "conditions" are the SAME
    items) and read them through two DIFFERENT judges -- an EXACT Type-I
    null (stronger than generate_real_twogroup_null_cell's "equal in
    distribution": here the true paired difference is exactly zero for
    every single item, since judge_a and judge_b are both noisy/biased
    reads of the IDENTICAL human_label -- there is no separate "ground
    truth of condition B" distinct from condition A's), with each judge's
    own real noise/bias driving the comparison instead of a synthetic
    model. label_frac reveals the SAME items' human labels for both
    conditions (there's one ground truth per item, not one per judge), so
    lab_x and lab_y are identical (copied, not aliased, so nothing
    downstream can mutate one and silently affect the other).

    Feeds cases/ppi_real.py's paired-null check (wilcoxon/paired_t/
    bayes_bootstrap/bootstrap_t/tango -- the PPI test methods that need a
    genuine paired/repeated structure, which a single judge model alone
    can't provide)."""
    n = min(n, corpus.corpus_size)
    idx = rng.choice(corpus.corpus_size, size=n, replace=False)
    llm_x = corpus.judge_scores[judge_a][idx]
    llm_y = corpus.judge_scores[judge_b][idx]
    human = corpus.human_label[idx]
    lab = _reveal_labels(human, label_frac, rng)
    return llm_x, llm_y, lab, lab.copy()


# ---------------------------------------------------------------------------
# Genuine within-item paired data (wmt_da_paired): ONE judge scoring TWO
# DIFFERENT MT systems' output for the SAME source segment, with REAL
# (not artificially-identical) human DA scores on each side. Complements
# generate_real_paired_null_cell's cross-judge proxy pairing (see that
# function's docstring, and the "why this needs its own fetcher" discussion
# in collect_judge_bias_data.py's fetch_wmt_da_paired_items) -- this is the
# structure a real single-judge Wilcoxon-style paired comparison actually
# has: one judge, two conditions for the same item, not two different
# judges reading the same condition.
# ---------------------------------------------------------------------------

_ITEM_ID_PAIRED_RE = re.compile(r"^wmt_da_paired_seg(\d+)_sys([01])$")

WMT_PAIRED_DATASET = "wmt_da_paired"
WMT_PAIRED_BOUNDS = (0.0, 100.0)  # same 0-100 DA scale as plain wmt_da


@dataclass
class RealWithinItemPairedCorpus:
    """collect_judge_bias_data.py's fetch_wmt_da_paired_items output,
    loaded and aligned: for each of ``corpus_size`` source segments, the
    REAL human DA score and every requested judge's score for its two
    different MT systems' outputs ("A" = sys0, "B" = sys1 -- an arbitrary
    but fixed labeling per segment, not "the better one" or "the worse
    one"). All arrays rescaled to [0, 1] (matching
    scenarios.EVAL_TYPE_SCALE_BOUNDS's convention). Unlike
    RealJudgeBiasCorpus (aligned across judge models, one human_label per
    item), this is aligned across BOTH SIDES of a pair for each judge --
    judge_scores_a[jm][i] and judge_scores_b[jm][i] are always the two
    systems of the SAME segment, for any requested judge jm.
    """
    dataset: str  # "wmt_da_paired"
    eval_type: str  # "continuous_paired"
    judge_models: list[str]
    human_a: np.ndarray  # shape (corpus_size,), system A's real human DA score, rescaled
    human_b: np.ndarray  # shape (corpus_size,), system B's real human DA score, rescaled
    judge_scores_a: dict[str, np.ndarray]  # judge_model -> shape (corpus_size,), rescaled
    judge_scores_b: dict[str, np.ndarray]  # judge_model -> shape (corpus_size,), rescaled
    corpus_size: int  # number of segment-pairs
    true_paired_mean: float
    """Population mean of (human_a - human_b) across every aligned segment
    -- the REAL target a PPI-corrected paired mean estimate (paired_t/
    bayes_bootstrap/bootstrap_t) should recover, not an artificial zero
    (contrast generate_real_paired_null_cell's exact-null construction)."""
    true_paired_median: float
    """Population median of (human_a - human_b) -- the analogous real
    target for wilcoxon's median estimand."""


def load_real_wmt_paired_corpus(
    *, data_dir: str = DEFAULT_DATA_DIR, judge_models: list[str] | None = None, min_coverage: float = 0.0,
) -> RealWithinItemPairedCorpus:
    """Load simulations/out/judge_bias_wmt_da_paired.csv (the merged view
    collect_judge_bias_data.py's collect-judge-scores writes for
    --types continuous_paired) into a RealWithinItemPairedCorpus.

    Unlike load_real_judge_bias_corpus's exact-null construction (the SAME
    human_label copied onto both sides of a pair, forcing a KNOWN-zero true
    difference for a Type-I check), system A and system B here genuinely
    have DIFFERENT real human labels -- true_paired_mean/true_paired_median
    are actual population values, not an artificial zero, so this feeds a
    bias/coverage check (generate_real_wmt_paired_bias_cell, the paired-
    structure analogue of generate_real_single_cell) rather than a Type-I
    null check.

    `judge_models`/`min_coverage` behave exactly as in
    load_real_judge_bias_corpus (auto-discover + coverage filtering when
    `judge_models` is None; trusted as-is with no filtering when given
    explicitly) -- see that function's docstring for the coverage-filtering
    rationale, which applies identically here.
    """
    path = Path(data_dir) / f"judge_bias_{WMT_PAIRED_DATASET}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:\n"
            f"  python simulations/collect_judge_bias_data.py collect-data --types continuous_paired\n"
            f"  python simulations/collect_judge_bias_data.py collect-judge-scores "
            f"--types continuous_paired --backend <openrouter|ollama> --models <model>\n"
            f"first to produce it."
        )
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows -- run collect-judge-scores first (collect-data alone "
                          f"only fetches items + human labels, no judge scores).")

    parsed: list[tuple[int, int, dict]] = []
    for r in rows:
        m = _ITEM_ID_PAIRED_RE.match(r["item_id"])
        if m is None:
            raise ValueError(
                f"Unexpected item_id {r['item_id']!r} in {path} -- expected "
                f"'wmt_da_paired_seg<N>_sys<0|1>' (from fetch_wmt_da_paired_items)."
            )
        parsed.append((int(m.group(1)), int(m.group(2)), r))

    models_present = sorted({r["judge_model"] for _seg, _sys, r in parsed})
    if judge_models is None:
        if min_coverage > 0.0:
            items_path = Path(data_dir) / f"judge_bias_{WMT_PAIRED_DATASET}_items.csv"
            with items_path.open(newline="", encoding="utf-8") as f:
                total_items = sum(1 for _ in csv.DictReader(f))
            items_by_model: dict[str, set[str]] = defaultdict(set)
            for _seg, _sys, r in parsed:
                items_by_model[r["judge_model"]].add(r["item_id"])
            kept, dropped = [], []
            for jm in models_present:
                cov = len(items_by_model[jm]) / total_items if total_items else 0.0
                (kept if cov >= min_coverage else dropped).append((jm, cov))
            if dropped:
                print(f"  [{WMT_PAIRED_DATASET}] excluding {len(dropped)} judge(s) below "
                      f"--min-judge-coverage {min_coverage:.0%}: "
                      + ", ".join(f"{jm} ({cov:.0%})" for jm, cov in dropped))
            if not kept:
                raise ValueError(
                    f"No judge models in {path} meet min_coverage={min_coverage:.0%} of {total_items} "
                    f"items. Present (with coverage): "
                    f"{[(jm, f'{len(items_by_model[jm]) / total_items:.0%}') for jm in models_present]}"
                )
            judge_models = [jm for jm, _cov in kept]
        else:
            judge_models = models_present
    else:
        missing = [m for m in judge_models if m not in models_present]
        if missing:
            raise ValueError(f"judge_models {missing} not found in {path}. Present: {models_present}")

    # (seg_idx, sys_idx, judge_model) -> list of (human_label, judge_score) across runs
    by_seg_sys_judge: dict[tuple[int, int, str], list[tuple[float, float]]] = defaultdict(list)
    for seg_idx, sys_idx, r in parsed:
        jm = r["judge_model"]
        if jm not in judge_models:
            continue
        by_seg_sys_judge[(seg_idx, sys_idx, jm)].append((float(r["human_label"]), float(r["judge_score"])))

    all_segs = sorted({seg for (seg, _sys, _jm) in by_seg_sys_judge.keys()})
    aligned_segs = [
        seg for seg in all_segs
        if all((seg, sys_idx, jm) in by_seg_sys_judge for sys_idx in (0, 1) for jm in judge_models)
    ]
    if not aligned_segs:
        raise ValueError(
            f"No segments in {path} have BOTH systems scored by every requested judge model "
            f"{judge_models} -- pass a smaller judge_models= subset, or collect-judge-scores "
            f"for the missing item(s)."
        )

    def _avg_human(seg: int, sys_idx: int) -> float:
        # Identical across judge_model/run_idx for a given (seg, sys_idx)
        # (same ground truth) -- averaging is a no-op, just collapses
        # duplicate rows, mirroring load_real_judge_bias_corpus's human_label.
        return float(np.mean([hl for hl, _js in by_seg_sys_judge[(seg, sys_idx, judge_models[0])]]))

    human_a = np.array([_avg_human(seg, 0) for seg in aligned_segs])
    human_b = np.array([_avg_human(seg, 1) for seg in aligned_segs])
    judge_scores_a = {
        jm: np.array([np.mean([js for _hl, js in by_seg_sys_judge[(seg, 0, jm)]]) for seg in aligned_segs])
        for jm in judge_models
    }
    judge_scores_b = {
        jm: np.array([np.mean([js for _hl, js in by_seg_sys_judge[(seg, 1, jm)]]) for seg in aligned_segs])
        for jm in judge_models
    }

    human_a = _rescale(human_a, WMT_PAIRED_BOUNDS)
    human_b = _rescale(human_b, WMT_PAIRED_BOUNDS)
    judge_scores_a = {jm: _rescale(a, WMT_PAIRED_BOUNDS) for jm, a in judge_scores_a.items()}
    judge_scores_b = {jm: _rescale(a, WMT_PAIRED_BOUNDS) for jm, a in judge_scores_b.items()}

    diffs = human_a - human_b
    return RealWithinItemPairedCorpus(
        dataset=WMT_PAIRED_DATASET, eval_type="continuous_paired", judge_models=list(judge_models),
        human_a=human_a, human_b=human_b, judge_scores_a=judge_scores_a, judge_scores_b=judge_scores_b,
        corpus_size=len(aligned_segs),
        true_paired_mean=float(np.mean(diffs)), true_paired_median=float(np.median(diffs)),
    )


def generate_real_wmt_paired_bias_cell(
    corpus: RealWithinItemPairedCorpus, rng: np.random.Generator, n: int, label_frac: float, judge_model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """WOR-draw n segment-pairs and reveal a label_frac fraction of their
    REAL (genuinely different, not copied) human labels -- the (llm_x,
    llm_y, lab_x, lab_y) shape evalstats.tests' paired-samples PPI
    functions expect, for a bias/coverage check: does the PPI-corrected
    paired estimate (as seen through ONE judge scoring both systems
    independently) recover corpus.true_paired_mean/true_paired_median, with
    correct CI coverage, as label_frac/n vary? The paired-structure
    analogue of generate_real_single_cell -- NOT a Type-I null check like
    generate_real_paired_null_cell (there is no artificial zero here)."""
    n = min(n, corpus.corpus_size)
    idx = rng.choice(corpus.corpus_size, size=n, replace=False)
    llm_x = corpus.judge_scores_a[judge_model][idx]
    llm_y = corpus.judge_scores_b[judge_model][idx]
    human_a = corpus.human_a[idx]
    human_b = corpus.human_b[idx]
    # SHARED revealed-index set across both sides -- a segment's pair either
    # gets both its real human labels revealed together or neither (one
    # ground truth pair per item, mirroring generate_real_paired_null_cell's
    # framing, just with genuinely different values on each side now).
    reveal_idx = _reveal_mask(n, label_frac, rng)
    lab_x = np.full(n, np.nan)
    lab_y = np.full(n, np.nan)
    lab_x[reveal_idx] = human_a[reveal_idx]
    lab_y[reveal_idx] = human_b[reveal_idx]
    return llm_x, llm_y, lab_x, lab_y
