"""The printed summary and the structured API describe the same analysis.

Each case runs ``compare()``, parses ``.summary()`` and checks it against
``to_dict()``, ``methods()`` and ``rank_bands()``: method names, correction,
omnibus test, rank-band source, every pairwise row and every leaderboard row.
Wilcoxon, McNemar and Nemenyi p-values are also recomputed independently.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import evalstats as es
from evalstats.core.paired import friedman_nemenyi
from evalstats.core.report import verdict_text
from evalstats.core.stats_utils import correct_pvalues
from evalstats.core.summary import _format_p_value
from evalstats.tests import _mcnemar_midp_p

N_BOOT = 200
ANSI = re.compile(r"\x1b\[[0-9;]*m")
P_COL = {"wilcoxon_signed_rank": "p ({t}wsr)", "nemenyi": "p (nem)",
         "mcnemar_midp": "p (mcnemar)", "ci_inversion": "p (CI)"}


# ── data and running ────────────────────────────────────────────────────────

def _make_df(kind: str, k: int, n: int, runs: int, seed: int, prompts=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.25, 0.75, n)
    shifts = rng.uniform(0, 0.18, k)
    rows = []
    for j in range(k):
        for pi, p in enumerate(prompts or [None]):
            for i in range(n):
                for r in range(runs):
                    v = base[i] + shifts[j] + 0.03 * pi + rng.normal(0, 0.12)
                    if kind == "bin":
                        v = float(rng.random() < np.clip(v, 0.02, 0.98))
                    elif kind == "likert":
                        v = float(np.clip(np.round(1 + 4 * v), 1, 5))
                    elif kind == "unbounded":
                        v = float(100 * v)
                    else:
                        v = float(np.clip(v, 0, 1))
                    row = {"model": f"m{j}", "item": f"i{i}", "score": v}
                    if p is not None:
                        row["prompt"] = p
                    if runs > 1:
                        row["run"] = r
                    rows.append(row)
    return pd.DataFrame(rows)


def _run(shape: str, kind: str, k: int, n: int, runs: int, kw: dict, seed: int = 7):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if shape == "ppi":
            from test_compound_ppi_fwer import _make_multiarm_binary, _make_multiarm_continuous
            from evalstats.alignment import judge_alignment
            maker = _make_multiarm_binary if kind == "bin" else _make_multiarm_continuous
            ev = maker(n_entities=k, seed=seed)
            ar = judge_alignment(ev, llm_metric="llm_score", human_groundtruth="human_score")
            result = es.compare(ev, factors="model", metric="llm_score", alignment={"llm_score": ar}, n_mc=30, **kw)
        else:
            prompts = ["p0", "p1"] if shape in {"two", "implicit"} else None
            df = _make_df(kind, k, n, runs, seed, prompts)
            factors = ["model", "prompt"] if shape == "two" else "model"
            result = es.compare(es.load_from(df), factors=factors, design="paired", n_bootstrap=N_BOOT, **kw)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result.summary()
    return result, ANSI.sub("", buf.getvalue())


# ── parsing ─────────────────────────────────────────────────────────────────

def _first(lines, rx):
    return next((m for m in (re.search(rx, l) for l in lines) if m), None)


def _loud_sections(text: str) -> dict[str, str]:
    lines, out, cur, buf, i = text.splitlines(), {}, "_top", [], 0
    while i < len(lines):
        if i + 2 < len(lines) and re.fullmatch(r"=+", lines[i]) and re.fullmatch(r"=+", lines[i + 2]):
            out[cur] = "\n".join(buf)
            cur, buf, i = lines[i + 1].strip(), [], i + 3
            continue
        buf.append(lines[i])
        i += 1
    out[cur] = "\n".join(buf)
    return out


def _header_cols(header: str) -> list[str]:
    tokens, cols, i = header.split("Interval Plot", 1)[1].split(), [], 0
    while i < len(tokens):
        if tokens[i] == "CI" and i + 1 < len(tokens):
            cols.append("CI " + tokens[i + 1]); i += 2
        elif tokens[i] == "p" and i + 1 < len(tokens) and tokens[i + 1].startswith("("):
            cols.append("p " + tokens[i + 1]); i += 2
        else:
            cols.append(tokens[i]); i += 1
    return cols


def _row_cols(tokens: list[str], cols: list[str]):
    j = next((i for i, t in enumerate(tokens[2:], 2) if re.match(r"^[+-]\d", t)), None)
    return None if j is None or len(tokens) - j < len(cols) else dict(zip(cols, tokens[j:j + len(cols)]))


def _exec_rows(lines, cross: bool = False):
    i = next((i for i, l in enumerate(lines) if "--- Executive Summary (" in l and (("Cross-model" in l) == cross)), None)
    if i is None:
        return None
    rx = (r"^\s+(\S+)\s+(\S+)\s+(#\d+)\s+([-\d.]+)\s+\[([-\d.]+), ([-\d.]+)\]" if cross
          else r"^\s+(\S+)\s+(#\d+)\s+([-\d.]+)\s+\[([-\d.]+), ([-\d.]+)\]")
    rows = []
    for l in lines[i + 3:]:
        if not l.strip() or l.strip().startswith("─"):
            break
        m, v = re.match(rx, l), re.search(r"(Likely best|Tied with .+ as best|Significant drop-off)\s*$", l)
        rows.append((m.groups() if m else None, v.group(1) if v else None))
    return rows


def _close(s: str, val) -> bool:
    if val is None:
        return True
    try:
        x = float(s)
    except (TypeError, ValueError):
        return False
    dec = len(s.split(".")[1]) if "." in s else 0
    return abs(x - val) <= 0.5 * 10 ** -dec + 1e-9


# ── agreement checks ────────────────────────────────────────────────────────

def _block_mismatches(result, text: str) -> list[str]:
    """Mismatches between one printed bundle section and the result's API."""
    bad: list[str] = []

    def expect(ok, what):
        if not ok:
            bad.append(what)

    m, d, bundle = result.methods(), result.to_dict(), result.full_analysis
    pw, alpha, L = bundle.pairwise, result.alpha, text.splitlines()
    pct, ppi, n_pairs = int(round((1 - alpha) * 100)), m["ppi"]["applied"], len(bundle.pairwise.results)

    mh = _first(L, r"--- Mean Performance \(marginal (\d+)% CIs\) ---")
    expect(mh and int(mh.group(1)) == pct, f"mean header pct {mh and mh.group(1)} != {pct}")
    mean_line = next((l for l in L if re.match(r"\s+\d+% CI method: ", l) and "|" not in l), None)
    expect(mean_line and mean_line.strip() == f"{pct}% CI method: {m['mean_ci']['name']}",
           f"mean CI line {mean_line!r} vs {m['mean_ci']}")

    ph = _first(L, r"--- Pairwise Comparisons \((\d+)% (.+) CIs\) ---")
    if ph is None:
        return bad + ["no pairwise section"]
    expect((int(ph.group(1)), ph.group(2)) == (pct, m["pairwise_ci"]["name"]),
           f"pairwise header {ph.groups()} vs {m['pairwise_ci']['name']!r}")
    f1 = next((re.match(r"\s+(\d+)% CI method: (.+?)(?:  \|  p-value method: (.+?))?  \|  α=(\S+)$", l)
               for l in L if "|  α=" in l), None)
    f2 = _first(L, r"^\s+Simultaneous CI method: (.+?)(?:  \|  FWER correction for p-values: (.+))?$")
    expect(f1 and (int(f1.group(1)), f1.group(2)) == (pct, m["pairwise_ci"]["name"]) and math.isclose(float(f1.group(4)), alpha),
           f"footer {f1 and f1.groups()} vs {pct}% {m['pairwise_ci']['name']!r} α={alpha}")
    expect(f2 and f2.group(1) == m["pairwise_ci"]["simultaneous"]["name"],
           f"simultaneous {f2 and f2.group(1)!r} vs {m['pairwise_ci']['simultaneous']}")

    hi = next((i for i, l in enumerate(L) if re.search(r"^\s+Left\s+Right\s+Interval Plot", l)), None)
    cols = _header_cols(L[hi]) if hi is not None else []
    p_col = next((c for c in cols if c.startswith("p (")), None)
    code, corr = m["p_values"]["test"]["code"], m["p_values"]["correction"]
    expect((p_col is not None) == m["p_values"]["shown"], f"p column {p_col!r} vs shown={m['p_values']['shown']}")
    expect({p.p_test for p in pw.results.values()} == {code}, f"pair p_tests vs {code}")
    if p_col:
        label = f1.group(3) if f1 else None
        expect(label is not None and (label.endswith(m["p_values"]["test"]["name"])
                                      or (code == "ci_inversion" and label.startswith("inverted from"))
                                      or (code == "paired_t" and ppi and "paired t-test" in label)),
               f"p method {label!r} vs {m['p_values']['test']}")
        expect(f2 and f2.group(2) == corr["name"], f"correction {f2 and f2.group(2)!r} vs {corr}")
        note = ("FWER-controlled" if code == "nemenyi"
                else "one comparison, uncorrected" if n_pairs == 1 else f"{corr['name']}-corrected")
        expect(f"  {p_col} = {label} ({note})" in L, f"p detail line for {p_col} = {label} ({note})")
        binary = str(bundle.resolved_data_kind) == "binary"
        rw = code == "bootstrap_t" and corr["code"] == "romano_wolf"
        want = ("p ({t}RW)" if rw else P_COL.get(code, "p (PPI-paired-t)" if (ppi and binary) else "p ({t}boot)")).format(
            t="PPI-" if ppi else "")
        expect(p_col == want, f"p column {p_col!r} vs {want!r}")

    omni = _first(L, r"--- Omnibus Test: (PPI-)?Friedman ---")
    expect(bool(omni) == (m["omnibus"] is not None), f"omnibus printed={bool(omni)} vs {m['omnibus']}")
    if omni and m["omnibus"] and not ppi:
        o, s = m["omnibus"], _first(L, r"statistic = ([-\d.]+)\((\d+)\)\s+p = (\S+)")
        expect(s and s.groups() == (f"{o['statistic']:.4f}", str(o["df"]), _format_p_value(o["p_value"])),
               f"omnibus {s and s.groups()} vs {o}")

    src = _first(L, r"rank bands .*computed from (.+):$") or _first(L, r"rank bands \((.+)\): none")
    want_src = f"{pct}% CI" if m["rank_bands"]["criterion"] == "simultaneous_ci_excludes_zero" else "corrected p"
    expect(src and src.group(1) == want_src, f"rank band source {src and src.group(1)!r} vs {want_src!r}")

    pairs = {frozenset((p["a"], p["b"])): p for p in d["pairwise"]}
    rows = []
    for l in L[hi + 1:] if hi is not None else []:
        if not l.strip() or l.strip().startswith("ES ="):
            break
        rows.append((l.split()[0], l.split()[1], _row_cols(l.split(), cols)))
    expect(len(rows) == len(pairs), f"{len(rows)} pairwise rows vs {len(pairs)} pairs")
    for left, right, c in rows:
        p = pairs.get(frozenset((left, right)))
        if p is None or c is None:
            bad.append(f"unparsed pairwise row {left} {right}")
            continue
        sign = 1 if (p["a"], p["b"]) == (left, right) else -1
        lo, hi_ = (p["ci_low"], p["ci_high"]) if sign == 1 else (-p["ci_high"], -p["ci_low"])
        expect(_close(c["Mean"], sign * p["diff"]) and _close(c["CI Low"], lo) and _close(c["CI High"], hi_),
               f"row {left}-{right} {c} vs {p}")
        if p_col:
            expect(c[p_col] == _format_p_value(p.get("p_value")), f"row {left}-{right} p {c[p_col]} vs {p.get('p_value')}")
        if "ES" in c:
            # The printed effect size must be the one the API returns for the
            # same orientation -- under PPI these came from two different
            # places and could disagree.
            es_api = pw.get(left, right).rank_biserial
            expect(_close(c["ES"], es_api), f"row {left}-{right} ES {c['ES']} vs {es_api}")

    ex, bands = _exec_rows(L), result.rank_bands()
    expect(ex is not None and len(ex) == len(bands), "leaderboard rows")
    for (g, verdict), row in zip(ex or [], bands):
        want_v = verdict_text(row["verdict"], row["tied_with"], max_name_len=20)
        expect(g is not None and (g[0], g[1]) == (row["label"], f"#{row['band']}") and verdict == want_v
               and _close(g[2], row["mean"]) and _close(g[3], row["ci_low"]) and _close(g[4], row["ci_high"]),
               f"leaderboard {g} {verdict!r} vs {row}")
    return bad


def _engine_p_mismatches(result) -> list[str]:
    """Stored p-values that don't match an independent recomputation of the named test."""
    m, bundle = result.methods(), result.full_analysis
    if m["ppi"]["applied"]:
        return []
    pw, labels = bundle.pairwise, [str(l) for l in bundle.labels]
    code, corr, keys = m["p_values"]["test"]["code"], m["p_values"]["correction"]["code"], list(bundle.pairwise.results)
    if code == "nemenyi":
        fr = friedman_nemenyi(bundle.benchmark.get_2d_scores(), labels)
        return [f"nemenyi {k}" for k in keys if not math.isclose(pw.results[k].p_value, fr.get_nemenyi_p(*k), abs_tol=1e-12)]
    if code == "wilcoxon_signed_rank":
        raw = [1.0 if not np.any(pw.results[k].per_input_diffs != 0) else stats.wilcoxon(pw.results[k].per_input_diffs).pvalue
               for k in keys]
    elif code == "mcnemar_midp" and bundle.benchmark.n_runs == 1:
        scores, idx = bundle.benchmark.get_2d_scores(), {l: i for i, l in enumerate(labels)}
        raw = [_mcnemar_midp_p(scores[idx[a]], scores[idx[b]]) for a, b in keys]
    else:
        return []
    adj = correct_pvalues(np.array(raw, dtype=float), corr, n_groups=len(labels)) if (corr != "none" and len(keys) > 1) else raw
    return [f"{code} {k}: stored {pw.results[k].p_value} vs {a}" for k, a in zip(keys, adj)
            if not math.isclose(pw.results[k].p_value, float(a), abs_tol=1e-10)]


def _header_mismatches(result, text: str, runs: int) -> list[str]:
    m, L, bad = result.methods(), text.splitlines(), []
    json.dumps(result.to_dict(), allow_nan=False)
    inputs, seed = _first(L, r"Inputs: (\d+)"), _first(L, r"seed: (\d+)")
    if inputs and int(inputs.group(1)) != m["design"]["n_items"]:
        bad.append(f"inputs {inputs.group(1)} vs {m['design']}")
    if (int(seed.group(1)) if seed else None) != m["resampling"]["rng_seed"]:
        bad.append(f"seed {seed and seed.group(1)} vs {m['resampling']}")
    if ("WARNING: only 2 runs" in text) != m["design"]["runs_averaged"] or m["design"]["n_runs"] != runs:
        bad.append(f"runs warning vs {m['design']}")
    return bad


# ── cases ───────────────────────────────────────────────────────────────────

CASES = [
    ("single", "cont", 3, 20, 1, {}),
    ("single", "cont", 3, 40, 1, {}),
    ("single", "cont", 2, 30, 1, {}),
    ("single", "bin", 3, 20, 1, {}),
    ("single", "bin", 3, 40, 1, {}),
    ("single", "likert", 3, 20, 3, {"score_range": (1, 5)}),
    ("single", "unbounded", 3, 20, 1, {}),
    ("single", "cont", 4, 20, 1, {"pairwise_test": "nemenyi"}),
    ("single", "cont", 2, 20, 1, {"pairwise_test": "nemenyi"}),
    ("single", "cont", 3, 40, 1, {"pairwise_test": "wilcoxon"}),
    ("single", "cont", 3, 40, 1, {"pairwise_test": "bootstrap"}),
    ("single", "cont", 3, 20, 1, {"p_values": False, "omnibus": False}),
    ("single", "cont", 3, 20, 1, {"simultaneous_ci": False, "alpha": 0.1}),
    ("single", "cont", 4, 20, 1, {"correction": "fdr_bh", "alpha": 0.01}),
    ("single", "cont", 3, 20, 2, {}),
    ("single", "bin", 3, 20, 3, {}),
    ("single", "bin", 3, 20, 1, {"method": "bootstrap"}),
    ("implicit", "cont", 3, 20, 1, {}),
    ("two", "cont", 3, 20, 1, {"alpha": 0.1}),
    ("two", "bin", 2, 40, 1, {}),
    ("ppi", "bin", 3, 0, 1, {}),
    ("ppi", "cont", 3, 0, 1, {}),
]


@pytest.mark.parametrize("shape,kind,k,n,runs,kw", CASES,
                         ids=[f"{c[0]}-{c[1]}-k{c[2]}-n{c[3]}-R{c[4]}-" + "-".join(f"{a}={b}" for a, b in c[5].items()) for c in CASES])
def test_summary_and_api_agree(shape, kind, k, n, runs, kw):
    result, text = _run(shape, kind, k, n, runs, kw)
    bad = _header_mismatches(result, text, runs)
    if shape == "two":
        sections = _loud_sections(text)
        for title, view in (("COMPARISON ACROSS", "model"), ("CROSS-MODEL PER-TEMPLATE", "prompt")):
            section = next(v for t, v in sections.items() if t.startswith(title))
            bad += [f"{view} view: {b}" for b in _block_mismatches(result.as_view(view), section)]
        cross = next(v for t, v in sections.items() if t.startswith("CROSS-MODEL RANKING")).splitlines()
        rows, bands = _exec_rows(cross, cross=True), result.rank_bands()
        assert rows is not None and len(rows) == len(bands)
        for (g, verdict), row in zip(rows, bands):
            want = (row["levels"]["model"], row["levels"]["prompt"], f"#{row['band']}")
            if g is None or g[:3] != want or verdict != verdict_text(row["verdict"], row["tied_with"], max_name_len=20):
                bad.append(f"cross leaderboard {g} {verdict!r} vs {row}")
        bad += _engine_p_mismatches(result.as_view("model"))
    else:
        section = next(v for t, v in _loud_sections(text).items() if t.startswith("COMPARISON ACROSS")) if shape == "implicit" else text
        bad += _block_mismatches(result, section) + _engine_p_mismatches(result)
    assert not bad, "\n".join(bad)
