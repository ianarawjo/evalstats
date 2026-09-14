"""The public, JSON-safe result surface front ends build on."""

from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import sys
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

import evalstats as es
from evalstats.core.report import leaderboard_order


def _df(shifts: dict, n: int = 40, seed: int = 0, sd: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.3, 0.7, n)
    return pd.DataFrame([
        {"model": name, "item": f"i{i}", "score": float(np.clip(base[i] + shift + rng.normal(0, sd), 0, 1))}
        for name, shift in shifts.items()
        for i in range(n)
    ])


def _compare(df: pd.DataFrame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return es.compare(es.load_from(df), design="paired", **kwargs)


def _executive_rows(result) -> list[tuple[str, str, str]]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result.summary()
    lines = buf.getvalue().splitlines()
    start = next(i for i, l in enumerate(lines) if "Executive Summary" in l)
    rows = []
    for line in lines[start + 3:]:
        m = re.match(r"^\s+(\S+)\s+(#\d+)\s+[\d.]+\s+\[[^\]]+\]\s+(.*\S)\s*$", line)
        if not m:
            break
        rows.append(m.groups())
    return rows


# ── rank bands ──────────────────────────────────────────────────────────────

def test_leaderboard_order_is_stable_on_ties():
    assert leaderboard_order([0.4, 0.9, 0.4, 0.1]) == [1, 0, 2, 3]


def test_rank_bands_match_executive_summary():
    result = _compare(_df({"a": 0.0, "b": 0.08, "c": 0.1}), factors="model")
    bands = result.rank_bands()
    assert [(r["label"], f"#{r['band']}", r["verdict_text"]) for r in bands] == _executive_rows(result)
    assert [r["rank"] for r in bands] == [1, 2, 3]


def test_tied_means_keep_data_order():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 30)
    scores = {"zeta": np.roll(y, 3), "alpha": y, "mid": np.zeros(30, int)}
    df = pd.DataFrame([
        {"model": m, "item": f"i{i}", "score": int(s[i])} for m, s in scores.items() for i in range(30)
    ])
    result = _compare(df, factors="model")
    assert [r["label"] for r in result.rank_bands()] == ["zeta", "alpha", "mid"]
    assert result.to_dict()["order"] == ["zeta", "alpha", "mid"]
    assert [row[0] for row in _executive_rows(result)] == ["zeta", "alpha", "mid"]


def test_significant_unbeaten_and_bands_share_the_ci_criterion():
    result = _compare(_df({"a": 0.0, "b": 0.08, "c": 0.1}), factors="model")
    d = result.to_dict()
    beaten = set()
    for pair in d["pairwise"]:
        assert pair["significant"] == (pair["ci_low"] > 0 or pair["ci_high"] < 0)
        if pair["significant"]:
            beaten.add(pair["b"] if pair["diff"] > 0 else pair["a"])
    expected_unbeaten = [l for l in result.labels if l not in beaten] if beaten else None
    assert result.unbeaten == expected_unbeaten


def test_verdict_codes_and_tied_with():
    result = _compare(_df({"a": 0.0, "b": 0.08, "c": 0.1}), factors="model")
    by_label = {r["label"]: r for r in result.rank_bands()}
    for row in by_label.values():
        assert row["verdict"] in {"likely_best", "tied_for_best", "significant_drop_off"}
        if row["verdict"] == "tied_for_best":
            assert row["tied_with"] and all(by_label[t]["band"] == 1 for t in row["tied_with"])
        else:
            assert row["tied_with"] == []


# ── two-factor levels ───────────────────────────────────────────────────────

def _two_factor_df(models, prompts, n=20, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([
        {"model": m, "prompt": p, "item": f"i{i}", "score": float(rng.uniform(0.2 + 0.1 * mi, 0.8))}
        for mi, m in enumerate(models) for p in prompts for i in range(n)
    ])


def test_two_factor_levels_survive_the_label_separator():
    result = _compare(_two_factor_df(["gpt / 4o", "claude"], ["terse", "long / detailed"]),
                      factors=["model", "prompt"])
    d = result.to_dict()
    assert d["entities"]["gpt / 4o / long / detailed"]["levels"] == {"model": "gpt / 4o", "prompt": "long / detailed"}
    for pair in d["pairwise"]:
        assert pair["a_levels"] == d["entities"][pair["a"]]["levels"]
    for model, prompt in result.best_pairs or []:
        assert model in {"gpt / 4o", "claude"} and prompt in {"terse", "long / detailed"}
    assert result.as_view("model").to_dict()["entities"]["gpt / 4o"]["levels"] == {"model": "gpt / 4o"}


def test_two_factor_label_collision_raises():
    with pytest.raises(ValueError, match="ambiguous"):
        _compare(_two_factor_df(["a / b", "a"], ["c", "b / c"]), factors=["model", "prompt"])


# ── JSON safety ─────────────────────────────────────────────────────────────

def test_to_dict_is_strict_json():
    for result in (
        _compare(_df({"a": 0.0, "b": 0.05, "c": 0.1}), factors="model"),
        _compare(_two_factor_df(["m1", "m2"], ["p1", "p2"]), factors=["model", "prompt"]),
    ):
        json.dumps(result.to_dict(), allow_nan=False)
        json.dumps(result.rank_bands(), allow_nan=False)


def test_to_dict_turns_non_finite_values_into_none():
    result = _compare(_df({"a": 0.0, "b": 0.05, "c": 0.1}), factors="model")
    (key, pair), *_ = result.pairwise.results.items()
    pair.ci_low, pair.ci_high = float("-inf"), float("inf")
    result.full_analysis.robustness.ci_low[0] = float("nan")
    d = result.to_dict()
    json.dumps(d, allow_nan=False)
    entry = next(p for p in d["pairwise"] if (p["a"], p["b"]) == key)
    assert entry["ci_low"] is None and entry["ci_high"] is None
    assert entry["significant"] is False
    assert d["entities"][result.labels[0]]["ci_low"] is None


# ── notes ───────────────────────────────────────────────────────────────────

def _with_constant_entity(seed: int) -> pd.DataFrame:
    df = _df({"a": 0.0, "b": 0.1}, n=20, seed=seed)
    return pd.concat([df, pd.DataFrame([{"model": "z", "item": f"i{i}", "score": 0.5} for i in range(20)])])


def test_notes_are_collected_on_the_result():
    result = _compare(_with_constant_entity(0), factors="model")
    codes = {n.code for n in result.notes}
    assert {"zero_variance", "bounded_01_autodetected"} <= codes
    zero = next(n for n in result.notes if n.code == "zero_variance")
    assert zero.entities == ("z",)
    assert {"code", "message", "severity", "entities"} <= set(result.to_dict()["notes"][0])


def test_notes_still_emit_warnings():
    with pytest.warns(UserWarning, match="zero variance"):
        es.compare(es.load_from(_with_constant_entity(0)), factors="model", design="paired")


def test_notes_are_isolated_between_threads():
    results: dict[str, object] = {}
    barrier = threading.Barrier(2)

    def run(name, df):
        barrier.wait()
        results[name] = es.compare(es.load_from(df), factors="model", design="paired", score_range=(0, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        threads = [
            threading.Thread(target=run, args=("constant", _with_constant_entity(1))),
            threading.Thread(target=run, args=("clean", _df({"a": 0.0, "b": 0.1}, n=20, seed=2))),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert "zero_variance" in {n.code for n in results["constant"].notes}
    assert "zero_variance" not in {n.code for n in results["clean"].notes}


# ── typed errors ────────────────────────────────────────────────────────────

def test_typed_errors():
    df = _df({"a": 0.0, "b": 0.1}, n=20)
    with pytest.raises(es.InsufficientItemsError) as floor:
        _compare(df[df["item"].isin([f"i{i}" for i in range(10)])], factors="model")
    assert floor.value.n_items == 10 and floor.value.min_items == es.MIN_ITEMS == 15

    with pytest.raises(es.MissingCellsError) as missing:
        _compare(df[~((df["model"] == "b") & df["item"].isin(["i0", "i1"]))], factors="model")
    assert missing.value.missing == [("b", "i0"), ("b", "i1")]
    assert missing.value.n_missing == 2

    with pytest.raises(es.TooFewGroupsError) as groups:
        _compare(df[df["model"] == "a"], factors="model")
    assert groups.value.n_groups == 1

    for err in (floor.value, missing.value, groups.value):
        assert isinstance(err, ValueError)


def test_two_factor_missing_cells_name_the_cell():
    df = _two_factor_df(["gpt / 4o", "claude"], ["terse", "long"])
    df = df[~((df["model"] == "claude") & (df["prompt"] == "terse") & (df["item"] == "i3"))]
    with pytest.raises(es.MissingCellsError) as missing:
        _compare(df, factors=["model", "prompt"])
    assert missing.value.missing == [("claude / terse", "i3")]
    assert missing.value.n_missing == 1


# ── complete_items ──────────────────────────────────────────────────────────

def test_complete_items_drops_whole_items_and_reports_cells():
    df = _df({"a": 0.0, "b": 0.1}, n=20)
    df = df[~((df["model"] == "b") & (df["item"] == "i3"))].copy()
    df.loc[(df["model"] == "a") & (df["item"] == "i5"), "score"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complete, report = es.complete_items(es.load_from(df), factors="model")
    assert report.excluded_items == ["i3", "i5"]
    assert report.n_items == 18 and report.n_excluded == 2
    assert report.missing_cells["i3"] == [{"model": "b"}]
    json.dumps(report.to_dict(), allow_nan=False)
    result = _compare(complete._df, factors="model")
    assert result.full_analysis.benchmark.n_inputs == 18


# ── methods record ──────────────────────────────────────────────────────────

def test_methods_record_names_what_ran():
    result = _compare(_df({"a": 0.0, "b": 0.05, "c": 0.1}, n=20), factors="model")
    m = result.to_dict()["methods"]
    json.dumps(m, allow_nan=False)
    assert m == result.methods()
    assert m["evalstats_version"] == es.__version__
    assert m["design"] == {"code": "paired", "n_items": 20, "n_runs": 1, "runs_averaged": False}
    assert m["data_kind"]["code"] == "bounded_01"
    assert m["mean_ci"] == {"code": "logit_t", "name": "Logit-t"}
    assert m["pairwise_ci"]["simultaneous"]["code"] == "sidak"
    assert m["p_values"]["test"] == {"code": "wilcoxon_signed_rank", "name": "Wilcoxon signed-rank"}
    assert m["p_values"]["correction"] == {"code": "shaffer", "name": "Shaffer"}
    assert m["omnibus"]["test"] == "friedman" and m["omnibus"]["df"] == 2
    assert m["rank_bands"] == {"criterion": "simultaneous_ci_excludes_zero"}
    assert m["resampling"]["n_bootstrap"] > 0 and m["resampling"]["rng_seed"] == result.rng_seed


def _summary_text(result) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result.summary()
    return buf.getvalue()


def test_methods_names_match_the_printed_summary():
    result = _compare(_df({"a": 0.0, "b": 0.05, "c": 0.1}, n=20), factors="model")
    m, text = result.methods(), _summary_text(result)
    assert f"--- Pairwise Comparisons (95% {m['pairwise_ci']['name']} CIs) ---" in text
    assert f"95% CI method: {m['mean_ci']['name']}\n" in text
    assert f"Simultaneous CI method: {m['pairwise_ci']['simultaneous']['name']}" in text
    assert f"p-value method: {m['p_values']['test']['name']}" in text
    assert f"FWER correction for p-values: {m['p_values']['correction']['name']}" in text


def test_nemenyi_with_two_entities_falls_back_with_a_note():
    result = _compare(_df({"a": 0.0, "b": 0.1}, n=20), factors="model", pairwise_test="nemenyi")
    assert "p (wsr)" in _summary_text(result)
    assert result.methods()["p_values"]["test"]["code"] == "wilcoxon_signed_rank"
    assert "nemenyi_unavailable" in {n.code for n in result.notes}


def test_summary_labels_use_the_result_alpha():
    single = _compare(_df({"a": 0.0, "b": 0.05, "c": 0.1}, n=20), factors="model", alpha=0.1)
    text = _summary_text(single)
    assert "computed from 90% CI" in text and "95% CI" not in text

    two = _compare(_two_factor_df(["m1", "m2", "m3"], ["p1", "p2"]), factors=["model", "prompt"], alpha=0.1)
    text = _summary_text(two)
    assert "90% CI method: Logit-t" in text and "95% CI method" not in text
    assert "α=0.1" in text and "α=0.05" not in text
    assert "(90% CI, not significantly beaten)" in text


# ── one pairwise p-value ────────────────────────────────────────────────────

def test_pairs_carry_one_p_value_named_by_p_test():
    result = _compare(_df({"a": 0.0, "b": 0.05, "c": 0.1}, n=20), factors="model")
    pw = result.pairwise
    assert pw.p_value_test == "wilcoxon_signed_rank"
    assert pw.correction_method == "shaffer"
    for pair in pw.results.values():
        assert not hasattr(pair, "wilcoxon_p")
    assert all("wilcoxon_p" not in p and 0 <= p["p_value"] <= 1 for p in result.to_dict()["pairwise"])


def test_romano_wolf_p_value_is_what_the_table_prints():
    result = _compare(_df({"a": 0.0, "b": 0.05, "c": 0.1}, n=40), factors="model")
    assert result.pairwise.p_value_test == "romano_wolf"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result.summary()
    assert "p (RW)" in buf.getvalue()


def test_print_time_p_value_method_must_match_the_stored_test():
    result = _compare(_df({"a": 0.0, "b": 0.05, "c": 0.1}, n=20), factors="model")
    with pytest.raises(ValueError, match="does not match"):
        result.summary(p_value_method="nem")


@pytest.mark.parametrize("test_method,code", [
    ("auto→paired logit-t", "paired_t"),
    ("paired NIG", "paired_t"),
    ("mj_floor cluster", "ci_inversion"),
    ("bonett_price", "mcnemar_midp"),
    ("newcombe (mcnemar_midp p-value)", "mcnemar_midp"),
    ("nested bootstrap-t (n=10, R=3)", "bootstrap_t"),
    ("paired sign test + bootstrap ci (n=10)", "sign_test"),
    ("paired permutation + bootstrap ci (n=10)", "sign_flip_permutation"),
    ("lmm wald (statsmodels, df=3)", "lmm_wald"),
])
def test_own_p_test_codes(test_method, code):
    from evalstats.core.paired import P_TEST_NAMES, own_p_test
    assert own_p_test(test_method) == code
    assert code in P_TEST_NAMES


# ── import cost ─────────────────────────────────────────────────────────────

def test_import_does_not_load_matplotlib():
    code = "import sys, evalstats; assert 'matplotlib' not in sys.modules; evalstats.plot_ci_forest"
    subprocess.run([sys.executable, "-c", code], check=True)
