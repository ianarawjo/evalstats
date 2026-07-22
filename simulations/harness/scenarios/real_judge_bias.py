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
is EXACTLY zero (not just equal in distribution, like the independent-group
null below) -- see generate_real_paired_null_cell. With k judge models
collected, all C(k, 2) unique pairs are available as independent (dataset,
label_frac, n, pair) cells, each carrying that pair's own real noise/bias
characteristics -- more judges means more such cells, i.e. more power to
catch a paired-test miscalibration than a single pair alone would give.

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
) -> RealJudgeBiasCorpus:
    """Load simulations/out/judge_bias_<dataset>.csv (the merged
    item+judge-score view collect_judge_bias_data.py's collect-judge-scores
    step writes) into a RealJudgeBiasCorpus.

    `judge_models`, if given, restricts to that subset (all must be
    present, or this raises). Default (None) uses every judge model
    present in the file. Only items scored by EVERY selected judge model
    are kept (mirroring real_data.py's build_openeval_multiarm_corpora,
    which aligns all requested real models on shared item_id).
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


def _reveal_labels(values: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """MCAR label-reveal: NaN out all but a `frac` fraction of `values`
    (floored at _MIN_LAB revealed items) -- the real-data analogue of
    scenarios.synthetic._jb_labels_independent, without its MNAR option."""
    n = len(values)
    if n < _MIN_LAB:
        raise ValueError(f"Need n >= {_MIN_LAB} to reveal at least {_MIN_LAB} labels; got n={n}")
    n_lab = min(n, max(_MIN_LAB, int(round(n * frac))))
    lab = np.full(n, np.nan)
    idx = rng.choice(n, size=n_lab, replace=False)
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
    corpus: RealJudgeBiasCorpus, rng: np.random.Generator, n: int, label_frac: float, judge_model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """WOR-draw 2n DISJOINT items (both scored by `judge_model`), split
    into two independent groups A/B -- a valid Type-I null by construction
    (both groups are random subsamples of the SAME population, so their
    true means are equal), carrying real noise/skew/judge-bias
    characteristics instead of synthetic ones. Mirrors real_data.py's
    corpus_pair_to_null_ci_pair_source permutation-null trick, simplified
    since there's only one real arm here (not two distinct real models
    needing symmetrizing).

    Only independent-samples tests apply to this structure (ttest,
    ttest_welch, mw_naive, mwu_corr) -- for a genuine PAIRED structure, see
    generate_real_paired_null_cell below."""
    n = min(n, corpus.corpus_size // 2)
    idx = rng.choice(corpus.corpus_size, size=2 * n, replace=False)
    idx_a, idx_b = idx[:n], idx[n:]
    js = corpus.judge_scores[judge_model]
    judge_a, judge_b = js[idx_a], js[idx_b]
    human_a, human_b = corpus.human_label[idx_a], corpus.human_label[idx_b]
    lab_a = _reveal_labels(human_a, label_frac, rng)
    lab_b = _reveal_labels(human_b, label_frac, rng)
    return judge_a, judge_b, lab_a, lab_b


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
