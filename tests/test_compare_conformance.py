"""compare() reports the intervals of the methods it names.

Every CI is recomputed from the low-level formula behind its label, with the
family adjustment (Sidak for paired, Bonferroni for unpaired) applied here
rather than taken from the router.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import evalstats as es
from evalstats.core.paired import _NIG_PAIRED_DIFF_B0
from evalstats.core.resampling import (
    bonett_price_paired_ci,
    bonett_price_paired_ci_multirun_shrunk,
    logit_t_ci_1d,
    nig_ci_1d,
    t_interval_ci_1d,
    wilson_ci_1d,
)
from evalstats.core.stats_utils import rescaled_ci
from evalstats.core.unpaired import _agresti_caffo_ci
from evalstats.tests import _mcnemar_midp_p

ALPHA = 0.05
RANGE = {"binary": (0, 1), "likert": (1, 5), "cont01": (0, 1), "unb_range": (0, 100),
         "unb_norange": None, "avg_binary": (0, 1)}
COMPARE_KW = {"binary": {}, "likert": {"score_range": (1, 5)}, "cont01": {},
              "unb_range": {"score_range": (0, 100)}, "unb_norange": {}}
KINDS = list(COMPARE_KW)


def _gen(kind, rng, j, n):
    if kind == "binary":
        return (rng.random(n) < 0.45 + 0.05 * j).astype(float)
    if kind == "likert":
        return np.clip(np.round(rng.normal(3 + 0.2 * j, 1.0, n)), 1, 5)
    if kind == "cont01":
        return np.clip(rng.beta(4 + j, 4, n), 0, 1)
    return np.clip(rng.normal(60 + 3 * j, 15, n), 0, 100)


def _sidak(n_pairs, alpha=ALPHA):
    return alpha if n_pairs <= 1 else 1 - (1 - alpha) ** (1 / n_pairs)


def _assert_bands_match(pw, n_pairs, reference):
    """Every gradient band is the pair's interval at its own adjusted level,
    and the headline band is the printed CI, so the plot agrees with the numbers."""
    assert pw.multi_ci[ALPHA] == pytest.approx((pw.ci_low, pw.ci_high))
    for band_alpha, band in pw.multi_ci.items():
        assert band == pytest.approx(reference(_sidak(n_pairs, band_alpha)))


def _ref_marginal(kind, x):
    if kind == "binary":
        return wilson_ci_1d(x, ALPHA)
    r = RANGE[kind]
    return t_interval_ci_1d(x, ALPHA) if r is None else rescaled_ci(logit_t_ci_1d, x, ALPHA, *r)


def _ref_pair(kind, a, b, alpha):
    """a, b: (n_items, n_runs) score matrices."""
    if kind == "binary":
        if a.shape[1] == 1:
            return bonett_price_paired_ci(a[:, 0], b[:, 0], alpha)
        return bonett_price_paired_ci_multirun_shrunk(a, b, alpha)
    d = a.mean(1) - b.mean(1)
    r = RANGE[kind]
    if r is None:
        return t_interval_ci_1d(d, alpha)
    span = r[1] - r[0]
    if kind in ("likert", "avg_binary"):
        return rescaled_ci(nig_ci_1d, d, alpha, -span, span, b0=_NIG_PAIRED_DIFF_B0)
    return rescaled_ci(logit_t_ci_1d, d, alpha, -span, span)


def _compare(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return es.compare(es.load_from(df), **kw)


@pytest.mark.parametrize("kind,runs,k,n", list(itertools.product(KINDS, [1, 3], [2, 3, 5], [15, 60])))
def test_paired_intervals_are_the_named_methods(kind, runs, k, n):
    rng = np.random.default_rng(7)
    mats = {f"P{j}": np.stack([_gen(kind, rng, j, n) for _ in range(runs)], axis=1) for j in range(k)}
    df = pd.DataFrame([
        {"prompt": p, "item": f"q{i}", "score": float(m[i, r]), **({"run": r} if runs > 1 else {})}
        for p, m in mats.items() for r in range(runs) for i in range(n)
    ])
    res = _compare(df, factors="prompt", **COMPARE_KW[kind])

    for label, m in mats.items():
        s = res.entity_stats[label]
        assert (s.ci_low, s.ci_high) == pytest.approx(_ref_marginal(kind, m.mean(1)))

    n_pairs = k * (k - 1) // 2
    for (a, b), pw in res.pairwise.results.items():
        assert (pw.ci_low, pw.ci_high) == pytest.approx(_ref_pair(kind, mats[a], mats[b], _sidak(n_pairs)))
        _assert_bands_match(pw, n_pairs, lambda al, a=a, b=b: _ref_pair(kind, mats[a], mats[b], al))
        if k == 2 and runs == 1:
            if kind == "binary":
                assert pw.p_value == pytest.approx(_mcnemar_midp_p(mats[a][:, 0], mats[b][:, 0]))
            else:
                assert pw.wilcoxon_p == pytest.approx(stats.wilcoxon(mats[a][:, 0], mats[b][:, 0]).pvalue)


@pytest.mark.parametrize("kind,k,n", list(itertools.product(KINDS, [2, 3, 5], [15, 60])))
def test_unpaired_intervals_are_the_named_methods(kind, k, n):
    rng = np.random.default_rng(11)
    groups = {f"P{j}": _gen(kind, rng, j, n) for j in range(k)}
    df = pd.DataFrame([{"prompt": p, "item": f"{p}_q{i}", "score": float(v[i])}
                       for p, v in groups.items() for i in range(n)])
    res = _compare(df, factors="prompt", design="unpaired", **COMPARE_KW[kind])

    for g in res.groups:
        assert (g.ci_low, g.ci_high) == pytest.approx(_ref_marginal(kind, groups[g.label]))

    alpha_pair = ALPHA / (k * (k - 1) // 2)
    for p in res.pairwise:
        ga, gb = groups[p.label_a], groups[p.label_b]
        if kind == "binary":
            expected = _agresti_caffo_ci(ga, gb, alpha_pair)
        else:
            expected = tuple(stats.ttest_ind(ga, gb, equal_var=False).confidence_interval(1 - alpha_pair))
        assert (p.ci_low, p.ci_high) == pytest.approx(expected)


@pytest.mark.parametrize("kind,shape", list(itertools.product(["binary", "cont01"], [(2, 2), (3, 2), (2, 3)])))
def test_model_by_prompt_intervals_are_the_named_methods(kind, shape):
    n_models, n_prompts = shape
    n = 40
    rng = np.random.default_rng(3)
    cells = {(f"m{i}", f"p{j}"): _gen(kind, rng, i + j, n) for i in range(n_models) for j in range(n_prompts)}
    df = pd.DataFrame([{"model": m, "prompt": p, "item": f"q{t}", "score": float(v[t])}
                       for (m, p), v in cells.items() for t in range(n)])
    res = _compare(df, factors=["model", "prompt"])

    n_cells = n_models * n_prompts
    n_cell_pairs = n_cells * (n_cells - 1) // 2
    for (a, b), pw in res.pairwise.results.items():
        ca, cb = cells[tuple(a.split(" / "))], cells[tuple(b.split(" / "))]
        expected = _ref_pair(kind, ca[:, None], cb[:, None], _sidak(n_cell_pairs))
        assert (pw.ci_low, pw.ci_high) == pytest.approx(expected)
        _assert_bands_match(pw, n_cell_pairs,
                            lambda al, ca=ca, cb=cb: _ref_pair(kind, ca[:, None], cb[:, None], al))

    view_kind = "avg_binary" if kind == "binary" else kind
    models = [f"m{i}" for i in range(n_models)]
    prompts = [f"p{j}" for j in range(n_prompts)]
    for axis, levels, others in (("model", models, prompts), ("prompt", prompts, models)):
        key = (lambda lv, o: (lv, o)) if axis == "model" else (lambda lv, o: (o, lv))
        avg = {lv: np.mean([cells[key(lv, o)] for o in others], axis=0) for lv in levels}
        view = res.as_view(axis)
        for lv in levels:
            s = view.entity_stats[lv]
            assert (s.ci_low, s.ci_high) == pytest.approx(_ref_marginal(view_kind, avg[lv]))
        n_level_pairs = len(levels) * (len(levels) - 1) // 2
        for (a, b), pw in view.pairwise.results.items():
            expected = _ref_pair(view_kind, avg[a][:, None], avg[b][:, None], _sidak(n_level_pairs))
            assert (pw.ci_low, pw.ci_high) == pytest.approx(expected)
            _assert_bands_match(pw, n_level_pairs,
                                lambda al, a=a, b=b: _ref_pair(view_kind, avg[a][:, None], avg[b][:, None], al))
