"""pvalues case: p-value/rejection-decision calibration, non-PPI and PPI-corrected.

Consolidates two complementary benchmarks into one file with three modes
(``--mode {pairwise,multiarm,ppi,all}``):

Non-PPI path (ported from ``simulations/sim_compare_pvalues.py``)
------------------------------------------------------------------
Given LLM-only scores (no human labels), which raw p-value/rejection
procedure is best calibrated?

- ``pairwise``: Type-I error and power for pairwise A-vs-B comparisons,
  via ``scenarios.synthetic.build_pair_sources``'s ICC x Cohen's-d grid
  (the same paired-difference scenario library ``cases/ci_paired.py`` uses)
  as the null/weak-alt/full-alt conditions.
- ``multiarm``: family-wise false-positive rate and best-arm selection power
  across p-value correction strategies (holm/bonferroni/fdr_bh/
  friedman_nemenyi), via ``scenarios.synthetic.build_multiarm_sources``,
  sweeping the SAME shape catalog ``build_pair_sources`` uses, generalized
  to k arms.

PPI-corrected path (ported from ``simulations/sim_type_i_calibration.py``)
---------------------------------------------------------------------------
- ``ppi``: given LLM scores plus sparse (possibly MNAR) human labels, does
  PPI correction (``evalstats.tests``' internal machinery) fix the Type-I
  inflation that judge bias/miscalibration causes in the uncorrected
  (scipy-equivalent) version? Sweeps judge-bias parameters via
  ``scenarios.synthetic.build_judge_bias_sources`` /
  ``generate_judge_bias_cell``, one factor at a time from a fixed baseline,
  layered on top of ONE representative shape per eval type from the SAME
  catalog the other two modes use. Includes a ``noise_family.*`` factor
  (``JudgeBiasSource.noise_family="contaminated"``) checking the same
  question under "judge mostly right, occasionally catastrophically wrong"
  measurement error instead of the default symmetric Gaussian -- same total
  noise variance either way, just redistributed.

There is no separate ``ppi_calibration`` case: it was folded in here instead,
since both halves answer "is this statistical decision trustworthy" at
different levels of the stack (raw CI/p-value procedures vs. high-level
``evalstats.tests`` wrappers with PPI), sharing report/plot/CLI scaffolding
AND, as of the shape-catalog unification, the underlying truth-generating
process too (``scenarios.synthetic.sample_group_truth`` -- see that
module's docstring and the harness README's "Shared scenario library"
section).

Known exceptions (see simulations/harness/README.md):
- ``ppi`` mode sweeps ``eval_type`` in ``{continuous, likert, grades}`` (no
  binary -- current PPI tests like ttest/MWU/ANOVA aren't designed for 0/1
  outcomes), picking one representative shape per eval type rather than
  sweeping the full catalog the way ``multiarm`` does -- judge-bias/noise/
  MNAR-label parameters are PPI's actual axis of interest, not shape.
- None of the three modes numerically matches its legacy script anymore
  (a deliberate trade: cross-mode truth-distribution consistency over
  per-mode legacy-script parity) -- verification is by sanity check
  (Type-I ~ alpha, power increasing with effect size/n) for all three.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.paired import pairwise_differences, all_pairwise, friedman_nemenyi, _mcnemar_p
    from evalstats.core.stats_utils import correct_pvalues
    from evalstats.tests import (
        _ppi_two_sample,
        _ppi_paired_arrays,
        _p_x_gt_y_midrank,
        _ppi_anova_independent_p_value,
        _ppi_anova_repeated_p_value,
        _ppi_friedman_p_value,
        _ppi_kruskal_wallis_pairwise,
        _ppi_lmm_p_value,
        _kw_pairwise_thetas,
    )
    from evalstats.core.mixed_effects import _fit_lmm_general, _get_fe_vcov_sm

from ..latex_tables import booktabs_table, escape_latex, eval_type_label
from ..scenarios import CIPairSource, MultiArmSource, JudgeBiasSource, EVAL_TYPES
from ..scenarios.synthetic import (
    SCENARIO_SUITES,
    build_pair_sources,
    build_multiarm_sources,
    build_judge_bias_sources,
    generate_judge_bias_cell,
    estimate_judge_bias_gold_null_values,
    JUDGE_BIAS_LMM_FACTORIAL_FACTORS,
)
from ..scenarios.real_data import DEFAULT_INSPECT_CSV, PAIR_SOURCES as REAL_PAIR_SOURCES, build_real_pair_sources
from ..methods import (
    PAIRWISE_PVALUE_METHODS,
    MCNEMAR,
    FISHER_EXACT,
    BOOTSTRAP,
    BCA,
    BAYES_BOOTSTRAP,
    SMOOTH_BOOTSTRAP,
    PERMUTATION,
    SIGN_TEST,
    NEWCOMBE_PVAL,
    BAYES_BINARY,
    WILCOXON,
    PAIRED_T,
    MULTIARM_CORRECTION_METHODS,
    PPI_TEST_METHODS,
    TTEST,
    TTEST_WELCH,
    MW,
    ANOVA_IND,
    ANOVA_REP,
    FRIEDMAN,
    KRUSKAL,
    LMM,
    LMM_FACTORIAL,
    LMM_RUNS,
    get_method_color,
    order_present_methods,
)
from . import CaseResult

CASE_NAME = "pvalues"

MODES = ["pairwise", "multiarm", "ppi", "all"]
DATA_SOURCES = ["synthetic"] + REAL_PAIR_SOURCES
PROGRESS_MODES = ["bar", "cell", "off"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]
ALPHA_DEFAULT = 0.05

_BINARY_ONLY_PVAL_METHODS = {NEWCOMBE_PVAL.name, BAYES_BINARY.name, MCNEMAR.name, FISHER_EXACT.name}


class _ProgressReporter:
    def __init__(self, total: int, *, mode: str = "bar", label: str = "") -> None:
        self.total = max(int(total), 1)
        self.mode = mode
        self.label = label
        self.start = time.time()
        self.last_print = 0.0

    def update(self, step: int, detail: str = "") -> None:
        if self.mode == "off":
            return
        now = time.time()
        is_final = step >= self.total
        if not is_final and (now - self.last_print) < 0.2:
            return
        self.last_print = now
        if self.mode == "cell":
            pct = 100.0 * min(step, self.total) / self.total
            print(f"\r  [{step:>7d}/{self.total:<7d}] {pct:6.2f}%  {detail:<55s}", end="", flush=True)
            if is_final:
                print()
            return
        frac = min(step, self.total) / self.total
        filled = int(28 * frac)
        bar = "█" * filled + "░" * (28 - filled)
        elapsed = max(now - self.start, 1e-9)
        rate = step / elapsed
        eta_sec = max(self.total - step, 0) / max(rate, 1e-12)
        eta_m, eta_s = divmod(int(round(eta_sec)), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        prefix = f"{self.label}: " if self.label else ""
        print(
            f"\r  {prefix}[{bar}] {100.0*frac:6.2f}%  {step:>7d}/{self.total:<7d}  "
            f"ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}  {detail[:40]:<40s}",
            end="", flush=True,
        )
        if is_final:
            print()


def _mc_proportion_stats(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    if total <= 0:
        return (float("nan"),) * 4
    p_hat = successes / total
    mcse = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    return float(p_hat), mcse, max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)


# ---------------------------------------------------------------------------
# Pairwise mode (non-PPI): Type-I error and power across raw p-value
# procedures, ported from sim_compare_pvalues.py's pairwise phase. Reuses
# scenarios.synthetic.build_pair_sources -- its is_null rows are the "null"
# condition, its cohens_d_values sweep is the alt-condition power curve.
# ---------------------------------------------------------------------------


@dataclass
class PairwiseResult:
    eval_type: str
    label: str
    n: int
    method: str
    condition: str  # "null" | f"d={cohens_d:.2f}"
    n_reps: int
    rejects: int
    p_sum: float
    cohens_d: float = 0.0


def _safe_wilcoxon_p(diffs: np.ndarray) -> float:
    if int(np.sum(diffs != 0)) < 1:
        return 1.0
    try:
        with np.errstate(all="ignore"):
            w = scipy_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        p = float(w.pvalue)
        return min(max(p, 0.0), 1.0) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _safe_paired_t_p(diffs: np.ndarray) -> float:
    if len(diffs) <= 1:
        return 1.0
    try:
        with np.errstate(all="ignore"):
            t = scipy_stats.ttest_1samp(diffs, popmean=0.0, nan_policy="omit")
        p = float(t.pvalue)
        return min(max(p, 0.0), 1.0) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _pairwise_pvalue(a: np.ndarray, b: np.ndarray, method: str, n_bootstrap: int, rng: np.random.Generator, statistic: str) -> float:
    """Compute one method's p-value for paired A-vs-B data of shape (n, runs)."""
    diffs = a.mean(axis=1) - b.mean(axis=1)

    if method == WILCOXON.name:
        return _safe_wilcoxon_p(diffs)
    if method == PAIRED_T.name:
        return _safe_paired_t_p(diffs)

    if method in _BINARY_ONLY_PVAL_METHODS:
        aa = (a.mean(axis=1) >= 0.5).astype(float)
        bb = (b.mean(axis=1) >= 0.5).astype(float)
        if method == MCNEMAR.name:
            return _mcnemar_p(aa, bb)
        scores = np.stack([aa, bb], axis=0)
    else:
        scores = np.stack([a[:, 0], b[:, 0]], axis=0) if a.shape[1] == 1 else np.stack([a, b], axis=0)

    result = pairwise_differences(
        scores=scores, idx_a=0, idx_b=1, label_a="A", label_b="B",
        method=method, ci=0.95, n_bootstrap=n_bootstrap, rng=rng, statistic=statistic,
    )
    p = float(result.p_value)
    return min(max(p, 0.0), 1.0) if np.isfinite(p) else 1.0


def _pairwise_methods_allowed(eval_type: str) -> list:
    return [m for m in PAIRWISE_PVALUE_METHODS if m.name not in _BINARY_ONLY_PVAL_METHODS or eval_type == "binary"]


def _run_pairwise_cell(
    source: CIPairSource, n: int, runs: int, n_reps: int, n_bootstrap: int, alpha: float, statistic: str, seed,
) -> list[PairwiseResult]:
    methods = _pairwise_methods_allowed(source.eval_type)
    condition = "null" if source.is_null else f"d={source.cohens_d:.2f}"

    ss = np.random.SeedSequence(seed)
    data_rng = np.random.default_rng(ss.spawn(1)[0])
    method_rngs = {m.name: np.random.default_rng(s) for m, s in zip(methods, ss.spawn(len(methods)))}

    rejects: dict[str, int] = {m.name: 0 for m in methods}
    p_sums: dict[str, float] = {m.name: 0.0 for m in methods}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_reps):
            a, b = source.generate_pair(data_rng, n, runs)
            for m in methods:
                p = _pairwise_pvalue(a, b, method=m.name, n_bootstrap=n_bootstrap, rng=method_rngs[m.name], statistic=statistic)
                p_sums[m.name] += p
                if p <= alpha:
                    rejects[m.name] += 1

    return [
        PairwiseResult(
            eval_type=source.eval_type, label=source.label, n=n, method=m.name, condition=condition,
            n_reps=n_reps, rejects=rejects[m.name], p_sum=p_sums[m.name], cohens_d=source.cohens_d,
        )
        for m in methods
    ]


def run_pairwise_simulation(
    sources: list[CIPairSource], sample_sizes: list[int], runs: int, n_reps: int, n_bootstrap: int,
    alpha: float, statistic: str, progress_mode: str = "bar", seed: int = 42,
) -> list[PairwiseResult]:
    ss = np.random.SeedSequence(seed)
    cells = [(s, n) for s in sources for n in sample_sizes]
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pvalues-pairwise")
    results: list[PairwiseResult] = []
    for i, (source, n) in enumerate(cells):
        results.extend(_run_pairwise_cell(source, n, runs, n_reps, n_bootstrap, alpha, statistic, child_seeds[i]))
        reporter.update(i + 1, detail=f"{source.eval_type} {source.label} n={n}")
    reporter.update(len(cells), detail="done")
    return results


def print_pairwise_report(results: list[PairwiseResult], alpha: float) -> None:
    print(f"\n{'='*78}\n  PVALUES (PAIRWISE, NON-PPI) -- TYPE I ERROR + POWER\n  Nominal alpha: {alpha}\n{'='*78}")
    present_methods = {r.method for r in results}
    method_labels = [m.name for m in order_present_methods(present_methods)]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    conditions = sorted({r.condition for r in results if r.condition != "null"})

    for et in eval_types_present:
        et_rows = [r for r in results if r.eval_type == et]
        print(f"\n  [{et}]")
        hdr = f"    {'Method':<14} {'typeI':>8}" + "".join(f"  power({c})".rjust(14) for c in conditions)
        print(hdr)
        for m in method_labels:
            m_rows = [r for r in et_rows if r.method == m]
            if not m_rows:
                continue
            null_rows = [r for r in m_rows if r.condition == "null"]
            c_tot = sum(r.rejects for r in null_rows)
            t_tot = sum(r.n_reps for r in null_rows)
            type1 = c_tot / t_tot if t_tot > 0 else float("nan")
            row = f"    {m:<14} {type1:>8.3f}"
            for c in conditions:
                c_rows = [r for r in m_rows if r.condition == c]
                cr = sum(r.rejects for r in c_rows)
                ct = sum(r.n_reps for r in c_rows)
                pw = cr / ct if ct > 0 else float("nan")
                row += f"  {pw:>12.3f}"
            print(row)
    print()


def latex_pairwise_overall_summary(results: list[PairwiseResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-method Type-I error + power, collapsed across eval types/sizes."""
    present_methods = {r.method for r in results}
    method_labels = [m.name for m in order_present_methods(present_methods)]
    eval_types_present = {et for et in EVAL_TYPES if any(r.eval_type == et for r in results)}
    conditions = sorted({r.condition for r in results if r.condition != "null"})

    rows = []
    for m in method_labels:
        m_rows = [r for r in results if r.method == m]
        if not m_rows:
            continue
        covered = {r.eval_type for r in m_rows}
        null_rows = [r for r in m_rows if r.condition == "null"]
        c_tot = sum(r.rejects for r in null_rows)
        t_tot = sum(r.n_reps for r in null_rows)
        type1 = c_tot / t_tot if t_tot > 0 else float("nan")
        power_cells = []
        for c in conditions:
            c_rows = [r for r in m_rows if r.condition == c]
            cr = sum(r.rejects for r in c_rows)
            ct = sum(r.n_reps for r in c_rows)
            power_cells.append(cr / ct if ct > 0 else float("nan"))
        mean_power = float(np.mean([p for p in power_cells if np.isfinite(p)])) if power_cells else float("nan")
        rows.append([
            escape_latex(m),
            f"{type1:.3f}" if np.isfinite(type1) else "-",
            f"{mean_power:.3f}" if np.isfinite(mean_power) else "-",
            eval_type_label(covered, eval_types_present),
        ])

    return booktabs_table(
        caption=f"pvalues (pairwise, non-PPI): Type-I error and mean power across conditions (nominal alpha={alpha}).",
        label="tab:pvalues_pairwise_overall",
        columns=["Method", "Type-I error", "Mean power", "Eval types"],
        rows=rows,
    )


def save_results_artifacts_pairwise(*, results: list[PairwiseResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_pairwise_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_type", "label", "n", "method", "condition", "n_reps", "rejects", "reject_rate", "mean_p", "cohens_d"])
        for r in results:
            writer.writerow([
                r.eval_type, r.label, r.n, r.method, r.condition, r.n_reps, r.rejects,
                f"{r.rejects / r.n_reps:.8f}", f"{r.p_sum / r.n_reps:.8f}", f"{r.cohens_d:.6f}",
            ])
    summary_path = out_base / f"{run_stem}_pairwise_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_pairwise_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_pairwise_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_pairwise_typeI_power_plot(*, results: list[PairwiseResult], alpha: float, out_path: str) -> str:
    """Type-I error (left) and power-vs-n curves (right) per eval type."""
    import matplotlib.pyplot as plt

    present_methods = {r.method for r in results}
    method_objs = order_present_methods(present_methods)
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    sample_sizes = sorted({r.n for r in results})

    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(11.0, 4.2 * nrows), squeeze=False, gridspec_kw={"hspace": 0.5})

    for row_idx, et in enumerate(eval_types_present):
        ax_t1, ax_pw = axes[row_idx][0], axes[row_idx][1]
        et_rows = [r for r in results if r.eval_type == et]
        ax_t1.axhline(alpha, color="black", linewidth=1.0, linestyle="--")
        ax_t1.axhspan(max(0.0, alpha - 0.02), alpha + 0.02, color="#DDDDDD", alpha=0.4, zorder=0)

        for m in method_objs:
            m_rows = [r for r in et_rows if r.method == m.name]
            if not m_rows:
                continue
            null_rows = [r for r in m_rows if r.condition == "null"]
            xs, ys = [], []
            for n in sample_sizes:
                subset = [r for r in null_rows if r.n == n]
                if not subset:
                    continue
                c = sum(r.rejects for r in subset)
                t = sum(r.n_reps for r in subset)
                xs.append(n)
                ys.append(c / t if t > 0 else float("nan"))
            if xs:
                ax_t1.plot(xs, ys, marker="o", color=m.color, markersize=4, linewidth=1.2, label=m.name, alpha=0.85)

            alt_rows = [r for r in m_rows if r.condition != "null"]
            xs2, ys2 = [], []
            for n in sample_sizes:
                subset = [r for r in alt_rows if r.n == n]
                if not subset:
                    continue
                c = sum(r.rejects for r in subset)
                t = sum(r.n_reps for r in subset)
                xs2.append(n)
                ys2.append(c / t if t > 0 else float("nan"))
            if xs2:
                ax_pw.plot(xs2, ys2, marker="o", color=m.color, markersize=4, linewidth=1.2, label=m.name, alpha=0.85)

        ax_t1.set_title(f"{et}: Type-I error")
        ax_t1.set_xlabel("n")
        ax_t1.set_ylabel("Rejection rate (null)")
        ax_t1.set_xscale("log")
        ax_pw.set_title(f"{et}: power (mean over alt conditions)")
        ax_pw.set_xlabel("n")
        ax_pw.set_ylabel("Rejection rate (alt)")
        ax_pw.set_xscale("log")
        ax_t1.legend(fontsize=6.5, loc="upper right")

    fig.suptitle(f"pvalues (pairwise, non-PPI): Type-I + Power | alpha={alpha}", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Multi-arm mode (non-PPI): family-wise error rate and best-arm selection
# power across p-value correction strategies, ported from
# sim_compare_pvalues.py's multi-arm phase.
# ---------------------------------------------------------------------------


@dataclass
class MultiArmResult:
    eval_type: str
    label: str
    n: int
    k: int
    correction: str
    condition: str  # "null" | "alt"
    n_reps: int
    any_reject: int
    best_selected: int


def _compute_multiarm_metrics(
    *, scores: np.ndarray, labels: list[str], method: str, corrections: list[str],
    n_bootstrap: int, alpha: float, statistic: str, rng: np.random.Generator,
) -> dict[str, tuple[bool, bool]]:
    """Compute (any_reject, best_selected) for every correction strategy.

    Bootstrap/permutation p-values are computed once with correction='none';
    each non-Friedman correction is applied to the same raw p-values.
    """
    results: dict[str, tuple[bool, bool]] = {}
    k = len(labels)
    pairs = [(labels[i], labels[j]) for i in range(k) for j in range(i + 1, k)]
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    non_friedman = [c for c in corrections if c != "friedman_nemenyi"]
    if non_friedman:
        matrix_none = all_pairwise(
            scores=scores, labels=labels, method=method, ci=1.0 - alpha,
            n_bootstrap=n_bootstrap, correction="none", rng=rng, statistic=statistic,
        )
        raw_p = np.array([matrix_none.get(a, b).p_value for (a, b) in pairs])
        pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}

        for correction in non_friedman:
            adj_p = raw_p if correction == "none" else correct_pvalues(raw_p, correction)
            has_any = bool(np.any(adj_p < alpha))

            best = labels[0]
            best_selected = True
            for other in labels[1:]:
                pair_idx = pair_to_idx.get((best, other))
                if pair_idx is None or not (adj_p[pair_idx] < alpha and matrix_none.get(best, other).point_diff > 0.0):
                    best_selected = False
                    break

            results[correction] = (has_any, best_selected)

    if "friedman_nemenyi" in corrections:
        try:
            fr = friedman_nemenyi(scores, labels)
            has_any = any(
                (p is not None and p < alpha) for (a, b) in pairs for p in [fr.get_nemenyi_p(a, b)]
            )
            best = labels[0]
            best_selected = True
            for other in labels[1:]:
                nem_p = fr.get_nemenyi_p(best, other)
                if nem_p is None:
                    best_selected = False
                    break
                if not (nem_p < alpha and fr.avg_ranks[best] < fr.avg_ranks[other]):
                    best_selected = False
                    break
            results["friedman_nemenyi"] = (has_any, best_selected)
        except Exception:
            results["friedman_nemenyi"] = (False, False)

    return results


def _run_multiarm_cell(
    source: MultiArmSource, n: int, runs: int, k_arms: int, n_reps: int, n_bootstrap: int,
    alpha: float, multiarm_method: str, statistic: str, seed,
) -> list[MultiArmResult]:
    labels = [f"arm_{i}" for i in range(k_arms)]
    corrections = [m.name for m in MULTIARM_CORRECTION_METHODS]
    rng = np.random.default_rng(seed)

    agg_any: dict[tuple[str, str], int] = {(c, cond): 0 for c in corrections for cond in ("null", "alt")}
    agg_best: dict[tuple[str, str], int] = {(c, cond): 0 for c in corrections for cond in ("null", "alt")}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_reps):
            for condition, delta in (("null", 0.0), ("alt", source.alt_delta)):
                scores = source.generate_scores(rng, n, runs, k_arms, delta)
                metrics = _compute_multiarm_metrics(
                    scores=scores, labels=labels, method=multiarm_method, corrections=corrections,
                    n_bootstrap=n_bootstrap, alpha=alpha, statistic=statistic, rng=rng,
                )
                for correction in corrections:
                    any_reject, best_selected = metrics.get(correction, (False, False))
                    if any_reject:
                        agg_any[(correction, condition)] += 1
                    if best_selected:
                        agg_best[(correction, condition)] += 1

    return [
        MultiArmResult(
            eval_type=source.eval_type, label=source.label, n=n, k=k_arms, correction=correction,
            condition=condition, n_reps=n_reps, any_reject=agg_any[(correction, condition)],
            best_selected=agg_best[(correction, condition)],
        )
        for correction in corrections
        for condition in ("null", "alt")
    ]


def run_multiarm_simulation(
    sources: list[MultiArmSource], sample_sizes: list[int], runs: int, k_arms: int, n_reps: int,
    n_bootstrap: int, alpha: float, multiarm_method: str, statistic: str, progress_mode: str = "bar", seed: int = 42,
) -> list[MultiArmResult]:
    ss = np.random.SeedSequence(seed)
    cells = [(s, n) for s in sources for n in sample_sizes]
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="pvalues-multiarm")
    results: list[MultiArmResult] = []
    for i, (source, n) in enumerate(cells):
        results.extend(_run_multiarm_cell(source, n, runs, k_arms, n_reps, n_bootstrap, alpha, multiarm_method, statistic, child_seeds[i]))
        reporter.update(i + 1, detail=f"{source.eval_type} n={n}")
    reporter.update(len(cells), detail="done")
    return results


def print_multiarm_report(results: list[MultiArmResult], alpha: float) -> None:
    print(f"\n{'='*78}\n  PVALUES (MULTI-ARM, NON-PPI) -- FWER + BEST-ARM POWER\n  Nominal alpha: {alpha}\n{'='*78}")
    corrections = [m.name for m in MULTIARM_CORRECTION_METHODS if m.name in {r.correction for r in results}]
    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    for et in eval_types_present:
        et_rows = [r for r in results if r.eval_type == et]
        print(f"\n  [{et}]")
        print(f"    {'Correction':<18} {'FWER':>8} {'BestPower':>10}")
        for corr in corrections:
            c_rows = [r for r in et_rows if r.correction == corr]
            null_rows = [r for r in c_rows if r.condition == "null"]
            alt_rows = [r for r in c_rows if r.condition == "alt"]
            fwer_t = sum(r.n_reps for r in null_rows)
            fwer_c = sum(r.any_reject for r in null_rows)
            power_t = sum(r.n_reps for r in alt_rows)
            power_c = sum(r.best_selected for r in alt_rows)
            fwer = fwer_c / fwer_t if fwer_t > 0 else float("nan")
            power = power_c / power_t if power_t > 0 else float("nan")
            print(f"    {corr:<18} {fwer:>8.3f} {power:>10.3f}")
    print()


def latex_multiarm_overall_summary(results: list[MultiArmResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-correction FWER + best-arm power, collapsed across eval types."""
    corrections = [m.name for m in MULTIARM_CORRECTION_METHODS if m.name in {r.correction for r in results}]
    eval_types_present = {et for et in EVAL_TYPES if any(r.eval_type == et for r in results)}

    rows = []
    for corr in corrections:
        c_rows = [r for r in results if r.correction == corr]
        covered = {r.eval_type for r in c_rows}
        null_rows = [r for r in c_rows if r.condition == "null"]
        alt_rows = [r for r in c_rows if r.condition == "alt"]
        fwer_t = sum(r.n_reps for r in null_rows)
        fwer_c = sum(r.any_reject for r in null_rows)
        power_t = sum(r.n_reps for r in alt_rows)
        power_c = sum(r.best_selected for r in alt_rows)
        fwer = fwer_c / fwer_t if fwer_t > 0 else float("nan")
        power = power_c / power_t if power_t > 0 else float("nan")
        rows.append([
            escape_latex(corr),
            f"{fwer:.3f}" if np.isfinite(fwer) else "-",
            f"{power:.3f}" if np.isfinite(power) else "-",
            eval_type_label(covered, eval_types_present),
        ])

    return booktabs_table(
        caption=f"pvalues (multi-arm, non-PPI): FWER and best-arm selection power (nominal alpha={alpha}).",
        label="tab:pvalues_multiarm_overall",
        columns=["Correction", "FWER", "Best-arm power", "Eval types"],
        rows=rows,
    )


def save_results_artifacts_multiarm(*, results: list[MultiArmResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_multiarm_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_type", "label", "n", "k", "correction", "condition", "n_reps", "any_reject", "best_selected", "any_reject_rate", "best_selected_rate"])
        for r in results:
            writer.writerow([
                r.eval_type, r.label, r.n, r.k, r.correction, r.condition, r.n_reps, r.any_reject, r.best_selected,
                f"{r.any_reject / r.n_reps:.8f}", f"{r.best_selected / r.n_reps:.8f}",
            ])
    summary_path = out_base / f"{run_stem}_multiarm_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_multiarm_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_multiarm_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_multiarm_fwer_power_plot(*, results: list[MultiArmResult], alpha: float, out_path: str) -> str:
    """FWER vs. best-arm-selection power, one point per correction strategy per eval type."""
    import matplotlib.pyplot as plt

    eval_types_present = [et for et in EVAL_TYPES if any(r.eval_type == et for r in results)]
    nrows = max(len(eval_types_present), 1)
    fig, axes = plt.subplots(nrows=1, ncols=nrows, figsize=(5.0 * nrows, 5.0), squeeze=False)

    for col_idx, et in enumerate(eval_types_present):
        ax = axes[0][col_idx]
        et_rows = [r for r in results if r.eval_type == et]
        ax.axvline(alpha, color="black", linestyle="--", linewidth=1.0)
        for m in MULTIARM_CORRECTION_METHODS:
            c_rows = [r for r in et_rows if r.correction == m.name]
            null_rows = [r for r in c_rows if r.condition == "null"]
            alt_rows = [r for r in c_rows if r.condition == "alt"]
            t1 = sum(r.n_reps for r in null_rows)
            c1 = sum(r.any_reject for r in null_rows)
            t2 = sum(r.n_reps for r in alt_rows)
            c2 = sum(r.best_selected for r in alt_rows)
            if t1 == 0 or t2 == 0:
                continue
            fwer = c1 / t1
            power = c2 / t2
            ax.scatter([fwer], [power], color=m.color, s=60, label=m.name, edgecolors="white", linewidths=0.6)
        ax.set_xlabel("FWER (null)")
        ax.set_ylabel("Best-arm selection power (alt)")
        ax.set_title(f"eval type: {et}")
        ax.set_xlim(-0.02, max(0.3, alpha * 4))
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("pvalues (multi-arm, non-PPI): FWER vs. best-arm power", fontsize=12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PPI mode: Type-I error calibration for evalstats.tests' PPI-corrected
# wrappers under judge bias/miscalibration, ported from
# sim_type_i_calibration.py's _run_one. Calls evalstats.tests' internal PPI
# functions directly (the same functions back the public es.tests.* API) to
# skip validate_alignment overhead, exactly as the legacy script does.
# ---------------------------------------------------------------------------


@dataclass
class PPIResult:
    name: str
    tag: str
    test: str
    n_reps: int
    corrected_rejects: int
    uncorrected_rejects: int
    n_failed: int = 0


def _uncorrected_anova_independent_p_value(groups: list[np.ndarray]) -> float:
    return float(scipy_stats.f_oneway(*groups).pvalue)


def _uncorrected_anova_repeated_p_value(groups: list[np.ndarray]) -> float:
    from statsmodels.stats.anova import AnovaRM

    k = len(groups)
    n_subjects = len(groups[0])
    stacked = np.column_stack(groups)
    df_long = pd.DataFrame({
        "subject": np.repeat(np.arange(n_subjects), k),
        "condition": np.tile(np.arange(k), n_subjects),
        "score": stacked.reshape(-1),
    })
    rm = AnovaRM(df_long, depvar="score", subject="subject", within=["condition"]).fit()
    return float(rm.anova_table.iloc[0]["Pr > F"])


def _uncorrected_friedman_p_value(groups: list[np.ndarray]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(scipy_stats.friedmanchisquare(*groups).pvalue)


def _uncorrected_kruskal_p_value(groups: list[np.ndarray]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(scipy_stats.kruskal(*groups).pvalue)


def _uncorrected_lmm_p_value(groups: list[np.ndarray], factors=None) -> float:
    """Uncorrected (LLM-only) Wald F-test for score ~ <fixed factors> + (1|input),
    fit via statsmodels MixedLM (REML); _fit_lmm_general handles single-factor,
    multi-factor, and nested-run groups alike."""
    k = len(groups)
    template_labels = [f"T{i}" for i in range(k)]
    sm_result, _df_full, _x_row, _r = _fit_lmm_general(groups, template_labels, factors)
    beta = sm_result.fe_params.to_numpy()
    cov = _get_fe_vcov_sm(sm_result)
    df1 = k - 1
    df2 = float(sm_result.df_resid)
    beta_t, cov_t = beta[1:], cov[1:, 1:]
    wald = float(beta_t @ np.linalg.solve(cov_t, beta_t))
    f_stat = wald / df1
    return float(scipy_stats.f.sf(f_stat, df1, df2)) if f_stat > 0 else 1.0


_ALPHA = ALPHA_DEFAULT


def _run_ppi_cell(sc: JudgeBiasSource, active_tests: list[str], n_reps: int, n_boot: int, seed) -> list[PPIResult]:
    rng = np.random.default_rng(seed)
    corrected: dict[str, int] = {t: 0 for t in active_tests}
    uncorrected: dict[str, int] = {t: 0 for t in active_tests}
    failed: dict[str, int] = {t: 0 for t in active_tests}

    def _rng_seed() -> int:
        return int(rng.integers(0, 2 ** 31))

    for _ in range(n_reps):
        cell = generate_judge_bias_cell(sc, rng)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if TTEST.name in active_tests:
                try:
                    p_u = float(scipy_stats.ttest_ind(cell.llm_a2, cell.llm_b2, equal_var=True).pvalue)
                    uncorrected[TTEST.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    corrected[TTEST.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[TTEST.name] += 1

            if TTEST_WELCH.name in active_tests:
                try:
                    p_u = float(scipy_stats.ttest_ind(cell.llm_a2, cell.llm_b2, equal_var=False).pvalue)
                    uncorrected[TTEST_WELCH.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda ya, yb: float(ya.mean() - yb.mean()), _ALPHA, n_boot, _rng_seed())
                    corrected[TTEST_WELCH.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[TTEST_WELCH.name] += 1

            if MW.name in active_tests:
                try:
                    p_u = float(scipy_stats.mannwhitneyu(cell.llm_a2, cell.llm_b2, alternative="two-sided").pvalue)
                    uncorrected[MW.name] += int(p_u < _ALPHA)
                    r = _ppi_two_sample(cell.llm_a2, cell.llm_b2, cell.lab_a2, cell.lab_b2, lambda xa, ya: _p_x_gt_y_midrank(xa, ya) - 0.5, _ALPHA, n_boot, _rng_seed())
                    corrected[MW.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[MW.name] += 1

            if WILCOXON.name in active_tests:
                try:
                    p_u = float(scipy_stats.wilcoxon(cell.llm_x, cell.llm_y, alternative="two-sided").pvalue)
                    uncorrected[WILCOXON.name] += int(p_u < _ALPHA)
                    r = _ppi_paired_arrays(cell.llm_x, cell.llm_y, cell.lab_x, cell.lab_y, np.median, _ALPHA, n_boot, _rng_seed(), rectifier_func=np.mean)
                    corrected[WILCOXON.name] += int(r.p_value < _ALPHA)
                except Exception:
                    failed[WILCOXON.name] += 1

            if ANOVA_IND.name in active_tests:
                try:
                    groups_ind = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_ind_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    p_u = _uncorrected_anova_independent_p_value(groups_ind)
                    uncorrected[ANOVA_IND.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_independent_p_value(groups_ind, groups_ind_lab, k=len(groups_ind))
                    corrected[ANOVA_IND.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[ANOVA_IND.name] += 1

            if ANOVA_REP.name in active_tests:
                try:
                    groups_rep = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_rep_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    p_u = _uncorrected_anova_repeated_p_value(groups_rep)
                    uncorrected[ANOVA_REP.name] += int(p_u < _ALPHA)
                    p = _ppi_anova_repeated_p_value(groups_rep, groups_rep_lab, k=len(groups_rep))
                    corrected[ANOVA_REP.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[ANOVA_REP.name] += 1

            if FRIEDMAN.name in active_tests:
                try:
                    groups_fr = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_fr_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    p_u = _uncorrected_friedman_p_value(groups_fr)
                    uncorrected[FRIEDMAN.name] += int(p_u < _ALPHA)
                    p = _ppi_friedman_p_value(groups_fr, groups_fr_lab, k=len(groups_fr))
                    corrected[FRIEDMAN.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[FRIEDMAN.name] += 1

            if KRUSKAL.name in active_tests:
                try:
                    groups_kw = [cell.llm_a3, cell.llm_b3, cell.llm_c3]
                    groups_kw_lab = [cell.lab_a3, cell.lab_b3, cell.lab_c3]
                    p_u = _uncorrected_kruskal_p_value(groups_kw)
                    uncorrected[KRUSKAL.name] += int(p_u < _ALPHA)
                    pw = _ppi_kruskal_wallis_pairwise(groups_kw, groups_kw_lab, alpha=_ALPHA, n_boot=n_boot, rng=_rng_seed())
                    corrected[KRUSKAL.name] += int(pw["wald_p"] < _ALPHA)
                except Exception:
                    failed[KRUSKAL.name] += 1

            if LMM.name in active_tests:
                try:
                    groups_lmm = [cell.llm_A, cell.llm_B, cell.llm_C]
                    groups_lmm_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    p_u = _uncorrected_lmm_p_value(groups_lmm)
                    uncorrected[LMM.name] += int(p_u < _ALPHA)
                    p = _ppi_lmm_p_value(groups_lmm, groups_lmm_lab, k=len(groups_lmm))
                    corrected[LMM.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[LMM.name] += 1

            if LMM_FACTORIAL.name in active_tests:
                try:
                    groups_lf = [cell.llm_W, cell.llm_X, cell.llm_Y, cell.llm_Z]
                    groups_lf_lab = [cell.lab_W, cell.lab_X, cell.lab_Y, cell.lab_Z]
                    p_u = _uncorrected_lmm_p_value(groups_lf, factors=JUDGE_BIAS_LMM_FACTORIAL_FACTORS)
                    uncorrected[LMM_FACTORIAL.name] += int(p_u < _ALPHA)
                    p = _ppi_lmm_p_value(groups_lf, groups_lf_lab, k=len(groups_lf), factors=JUDGE_BIAS_LMM_FACTORIAL_FACTORS)
                    corrected[LMM_FACTORIAL.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[LMM_FACTORIAL.name] += 1

            if LMM_RUNS.name in active_tests:
                try:
                    groups_runs = [cell.llm_A_runs, cell.llm_B_runs, cell.llm_C_runs]
                    groups_runs_lab = [cell.lab_A, cell.lab_B, cell.lab_C]
                    p_u = _uncorrected_lmm_p_value(groups_runs)
                    uncorrected[LMM_RUNS.name] += int(p_u < _ALPHA)
                    p = _ppi_lmm_p_value(groups_runs, groups_runs_lab, k=len(groups_runs))
                    corrected[LMM_RUNS.name] += int(p is not None and p < _ALPHA)
                except Exception:
                    failed[LMM_RUNS.name] += 1

    return [
        PPIResult(
            name=sc.name, tag=sc.tag, test=t, n_reps=n_reps,
            corrected_rejects=corrected[t], uncorrected_rejects=uncorrected[t], n_failed=failed[t],
        )
        for t in active_tests
    ]


def run_ppi_simulation(
    sources: list[JudgeBiasSource], active_tests: list[str], n_reps: int, n_boot: int,
    progress_mode: str = "bar", seed: int = 42,
) -> list[PPIResult]:
    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(sources))]

    reporter = _ProgressReporter(len(sources), mode=progress_mode, label="pvalues-ppi")
    results: list[PPIResult] = []
    for i, sc in enumerate(sources):
        results.extend(_run_ppi_cell(sc, active_tests, n_reps, n_boot, child_seeds[i]))
        reporter.update(i + 1, detail=f"{sc.name}")
    reporter.update(len(sources), detail="done")
    return results


def _ppi_wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli rate (ported from
    sim_type_i_calibration.py's ``_wilson_interval``)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    radius = (z / denom) * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
    lo = max(0.0, center - radius)
    hi = min(1.0, center + radius)
    return lo, hi


def _ppi_holm_rejections(pvals: list[tuple[tuple[str, str], float]], alpha: float = 0.05) -> set[tuple[str, str]]:
    """Rejected (scenario, test) cells under Holm-Bonferroni family-wise error
    control (ported from sim_type_i_calibration.py's ``_holm_rejections``)."""
    ordered = sorted(pvals, key=lambda x: x[1])
    m = len(ordered)
    rejected: set[tuple[str, str]] = set()
    for i, (cell, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        if p <= thresh:
            rejected.add(cell)
        else:
            break
    return rejected


def _fmt_ppi_rate(rate: float | None, flag2: float, flag3: float) -> str:
    if rate is None:
        return "  n/a  "
    s = f"{rate:.3f}"
    if rate > flag3:
        return s + "●●"
    if rate > flag2:
        return s + "● "
    return s + "  "


def print_ppi_report(results: list[PPIResult], alpha: float) -> None:
    """Scenario x test calibration table, mirroring sim_type_i_calibration.py's
    ``_print_table``: tag-grouped scenario rows, one column per test, per-cell
    2-sigma/3-sigma inflation flags, a Wilson-CI miscalibration flag (dagger),
    a Holm-Bonferroni family-wise flag (double-dagger), and a SUMMARY section
    with flag counts plus per-test corrected/uncorrected max/mean/median --
    instead of one flat row per (scenario, test) cell and a single
    averaged-rate table.
    """
    if not results:
        print("\n  (no PPI results)")
        return

    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    n_reps = results[0].n_reps
    sigma = (alpha * (1 - alpha) / n_reps) ** 0.5 if n_reps > 0 else float("nan")
    flag2 = alpha + 2 * sigma
    flag3 = alpha + 3 * sigma

    width = 90
    bar = "-" * width
    dbar = "=" * width
    col_w = max(9, max((len(t) for t in tests), default=9) + 1)

    cell: dict[tuple[str, str], PPIResult] = {(r.name, r.test): r for r in results}
    scenario_order: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in results:
        if r.name not in seen:
            seen.add(r.name)
            scenario_order.append((r.name, r.tag))
    name_w = max((len(name) for name, _tag in scenario_order), default=30) + 2

    tag_order: list[str] = []
    for _, tag in scenario_order:
        if tag not in tag_order:
            tag_order.append(tag)

    def rate_of(name: str, test: str) -> float | None:
        r = cell.get((name, test))
        if r is None or r.n_reps <= 0:
            return None
        return r.corrected_rejects / r.n_reps

    def uncorrected_rate_of(name: str, test: str) -> float | None:
        r = cell.get((name, test))
        if r is None or r.n_reps <= 0:
            return None
        return r.uncorrected_rejects / r.n_reps

    def wilson_outside(name: str, test: str) -> bool:
        r = cell.get((name, test))
        if r is None or r.n_reps <= 0:
            return False
        lo, hi = _ppi_wilson_interval(r.corrected_rejects, r.n_reps)
        return (alpha < lo) or (alpha > hi)

    print()
    print(dbar)
    print("  PVALUES (PPI-CORRECTED) -- TYPE I ERROR CALIBRATION")
    print(f"  n_reps={n_reps}  alpha={alpha}")
    print(f"  2σ flag (●): rate > {flag2:.3f}    3σ flag (●●): rate > {flag3:.3f}")
    print("  Wilson flag (†): 95% CI for rejection rate excludes alpha")
    print("  Holm flag (‡): exact binomial miscalibration survives family-wise correction")
    print(dbar)

    print()
    print(f"  {'Scenario':<{name_w}}" + "".join(f"{t:^{col_w}}" for t in tests))
    print(bar)

    pvals: list[tuple[tuple[str, str], float]] = []
    for name, _tag in scenario_order:
        for t in tests:
            r = cell.get((name, t))
            if r is not None and r.n_reps > 0:
                p = float(scipy_stats.binomtest(r.corrected_rejects, r.n_reps, alpha, alternative="two-sided").pvalue)
                pvals.append(((name, t), p))
    holm_bad = _ppi_holm_rejections(pvals, alpha=0.05)

    for tag in tag_order:
        print(f"\n[{tag}]")
        for name, sc_tag in scenario_order:
            if sc_tag != tag:
                continue
            row = f"  {name:<{name_w - 2}}"
            for t in tests:
                row += f" {_fmt_ppi_rate(rate_of(name, t), flag2, flag3):<{col_w - 1}}"
            n_failed_row = sum(cell[(name, t)].n_failed for t in tests if (name, t) in cell)
            if n_failed_row:
                row += f" ✗{n_failed_row}"
            if any(wilson_outside(name, t) for t in tests):
                row += "  †"
            if any((name, t) in holm_bad for t in tests):
                row += "‡"
            print(row)

    # -- Summary --------------------------------------------------------------
    print()
    print(bar)
    print("SUMMARY")
    print()

    n_scenarios = len(scenario_order)
    n_conditions = sum(1 for name, _tag in scenario_order for t in tests if (name, t) in cell)
    total_failed = sum(r.n_failed for r in results)

    all_corr = [rate_of(name, t) for name, _tag in scenario_order for t in tests]
    all_unc = [uncorrected_rate_of(name, t) for name, _tag in scenario_order for t in tests]

    flags2 = sum(1 for r in all_corr if r is not None and r > flag2)
    flags3 = sum(1 for r in all_corr if r is not None and r > flag3)
    wilson_miscal = sum(1 for name, _tag in scenario_order for t in tests if wilson_outside(name, t))
    nominal_miscal = sum(1 for _, p in pvals if p < 0.05)
    uncorrected_flags2 = sum(1 for r in all_unc if r is not None and r > flag2)
    uncorrected_flags3 = sum(1 for r in all_unc if r is not None and r > flag3)

    print(f"  Scenarios: {n_scenarios}  |  Tests: {len(tests)}  |  Conditions: {n_conditions}  |  Failed reps: {total_failed}")
    print(f"  Inflated at 2σ (rate > {flag2:.3f}):  {flags2}/{n_conditions}")
    print(f"  Inflated at 3σ (rate > {flag3:.3f}):  {flags3}/{n_conditions}")
    print(f"  Wilson miscalibrated (alpha outside 95% CI): {wilson_miscal}/{n_conditions}")
    print(f"  Exact-binomial p<0.05 (corrected rates, unadjusted): {nominal_miscal}/{n_conditions}")
    print(f"  Holm-confirmed miscalibrated cells: {len(holm_bad)}/{n_conditions}")
    print()
    print("  Uncorrected aggregate")
    print(f"  Inflated at 2σ (rate > {flag2:.3f}):  {uncorrected_flags2}/{n_conditions}")
    print(f"  Inflated at 3σ (rate > {flag3:.3f}):  {uncorrected_flags3}/{n_conditions}")
    print()
    print(f"  {'Test':<14}  {'corr max':>9}  {'corr mean':>9}  {'corr med':>9}  {'unc max':>9}  {'unc mean':>9}  {'unc med':>9}")
    for t in tests:
        col_rates = [r for r in (rate_of(name, t) for name, _tag in scenario_order) if r is not None]
        col_uncorrected = [r for r in (uncorrected_rate_of(name, t) for name, _tag in scenario_order) if r is not None]
        if col_rates or col_uncorrected:
            corr_max = max(col_rates) if col_rates else float("nan")
            corr_mean = float(np.mean(col_rates)) if col_rates else float("nan")
            corr_median = float(np.median(col_rates)) if col_rates else float("nan")
            unc_max = max(col_uncorrected) if col_uncorrected else float("nan")
            unc_mean = float(np.mean(col_uncorrected)) if col_uncorrected else float("nan")
            unc_median = float(np.median(col_uncorrected)) if col_uncorrected else float("nan")
            print(
                f"  {t:<14}  {corr_max:>9.3f}  {corr_mean:>9.3f}  {corr_median:>9.3f}  "
                f"{unc_max:>9.3f}  {unc_mean:>9.3f}  {unc_median:>9.3f}"
            )
    print()


def latex_ppi_overall_summary(results: list[PPIResult], alpha: float) -> str:
    """LaTeX booktabs overall summary: per-test corrected/uncorrected Type-I rate, averaged across scenarios.

    PPIResult has no eval_type axis (scenarios are judge-bias/noise sweeps, not
    distribution shapes), so there's no "Eval types" column here.
    """
    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    rows = []
    for t in tests:
        t_rows = [r for r in results if r.test == t]
        c_tot = sum(r.corrected_rejects for r in t_rows)
        u_tot = sum(r.uncorrected_rejects for r in t_rows)
        n_tot = sum(r.n_reps for r in t_rows)
        rate_c = c_tot / n_tot if n_tot > 0 else float("nan")
        rate_u = u_tot / n_tot if n_tot > 0 else float("nan")
        rows.append([
            escape_latex(t),
            f"{rate_c:.3f}" if np.isfinite(rate_c) else "-",
            f"{rate_u:.3f}" if np.isfinite(rate_u) else "-",
        ])

    return booktabs_table(
        caption=f"pvalues (PPI-corrected): corrected vs. uncorrected Type-I rate (nominal alpha={alpha}).",
        label="tab:pvalues_ppi_overall",
        columns=["Test", "Corrected", "Uncorrected"],
        rows=rows,
    )


def save_results_artifacts_ppi(*, results: list[PPIResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False) -> list[str]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / f"{run_stem}_ppi_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "tag", "test", "n_reps", "corrected_rejects", "uncorrected_rejects", "n_failed", "corrected_rate", "uncorrected_rate"])
        for r in results:
            writer.writerow([
                r.name, r.tag, r.test, r.n_reps, r.corrected_rejects, r.uncorrected_rejects, r.n_failed,
                f"{r.corrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
                f"{r.uncorrected_rejects / r.n_reps:.8f}" if r.n_reps else "",
            ])
    summary_path = out_base / f"{run_stem}_ppi_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_ppi_report(results, alpha=alpha)
    summary_text = buf.getvalue()
    if latex:
        summary_text += "\n% --- LaTeX table (--latex) ---\n" + latex_ppi_overall_summary(results, alpha=alpha)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_ppi_typeI_plot(*, results: list[PPIResult], alpha: float, out_path: str) -> str:
    """Per-scenario corrected vs. uncorrected Type-I rate scatter, one jittered
    column per test. Mirrors sim_type_i_calibration.py's ``_plot_results``
    scatter (gray uncorrected dots behind colored corrected dots, one dot per
    scenario, dashed alpha line) rather than collapsing every scenario into a
    single averaged bar, which hid per-scenario miscalibration entirely.
    """
    import matplotlib.pyplot as plt

    tests = [m.name for m in PPI_TEST_METHODS if m.name in {r.test for r in results}]
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    rng = np.random.default_rng(0)
    colors = [plt.cm.tab10(i) for i in range(len(tests))]
    unc_label_added = False
    all_rates: list[np.ndarray] = []

    for j, t in enumerate(tests):
        t_rows = [r for r in results if r.test == t]
        rates_u = np.array([r.uncorrected_rejects / r.n_reps if r.n_reps else float("nan") for r in t_rows])
        rates_c = np.array([r.corrected_rejects / r.n_reps if r.n_reps else float("nan") for r in t_rows])
        all_rates.append(rates_u)
        all_rates.append(rates_c)

        keep_u = np.isfinite(rates_u)
        x_u = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_u)))
        ax.scatter(
            x_u, rates_u[keep_u], s=18, alpha=0.35, color="#808080",
            label="uncorrected" if not unc_label_added else None, zorder=1,
        )
        unc_label_added = True

        keep_c = np.isfinite(rates_c)
        x_c = j + rng.uniform(-0.16, 0.16, size=int(np.sum(keep_c)))
        ax.scatter(x_c, rates_c[keep_c], s=20, alpha=0.65, color=colors[j], label=t, zorder=2)

    ax.axhline(alpha, color="black", ls="--", lw=1.1, label=f"alpha={alpha}")
    ax.set_xlim(-0.5, len(tests) - 0.5)
    scatter_max = np.nanmax(np.concatenate(all_rates)) if all_rates else float("nan")
    if not np.isfinite(scatter_max):
        scatter_max = 0.2
    ax.set_ylim(0.0, max(0.2, float(scatter_max) * 1.05))
    ax.set_xticks(np.arange(len(tests)))
    ax.set_xticklabels(tests, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Observed rejection rate")
    ax.set_xlabel("Test")
    ax.set_title("pvalues (PPI-corrected): Type-I calibration scatter (per-scenario cells)")
    ax.grid(axis="y", alpha=0.25, lw=0.8)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=MODES, default="all",
                         help="'pairwise' (non-PPI A/B), 'multiarm' (non-PPI k-arm), "
                              "'ppi' (PPI-corrected calibration), or 'all' (default: run all three)")
    parser.add_argument("--reps", type=int, default=200, metavar="N")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--latex", action="store_true", default=False,
                         help="Append a LaTeX booktabs overall-summary table to each saved summary .log file.")

    # pairwise mode
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="synthetic",
                         help="pairwise mode: 'synthetic' (default), or a real-data source: " + ", ".join(REAL_PAIR_SOURCES))
    parser.add_argument("--scenario-suite", choices=SCENARIO_SUITES, default="expanded",
                         help="pairwise mode: synthetic scenario breadth for build_pair_sources (ignored for real data sources)")
    parser.add_argument("--eval-types", nargs="+", choices=EVAL_TYPES, default=None, metavar="TYPE",
                         help="pairwise/multiarm modes: restrict to these eval types")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50, 100], metavar="N",
                         help="pairwise/multiarm modes: sample sizes to sweep")
    parser.add_argument("--runs", type=int, default=1, metavar="R",
                         help="pairwise/multiarm modes: runs per input (R>1 activates binary majority-vote/nested paths; "
                              "real-data pairwise sources only support --runs 1)")
    parser.add_argument("--statistic", choices=["mean", "median"], default="mean",
                         help="pairwise/multiarm modes: statistic passed to evalstats.core.paired")
    parser.add_argument("--bootstrap-n", type=int, default=500, metavar="N",
                         help="pairwise/multiarm modes: bootstrap resample count")
    parser.add_argument("--icc-values", type=float, nargs="+", default=[0.05, 0.20, 0.40, 0.60, 0.80], metavar="ICC",
                         help="pairwise mode: ICC sweep for build_pair_sources (ignored for real data sources)")
    parser.add_argument("--cohens-d-values", type=float, nargs="+", default=[0.2, 0.4], metavar="D",
                         help="pairwise mode: alt-condition effect sizes for build_pair_sources (ignored for real data sources)")
    parser.add_argument("--benchmarks", nargs="+", default=None, metavar="ID",
                         help="pairwise mode, real data: benchmark IDs to filter to")
    parser.add_argument("--models", nargs="+", default=None, metavar="NAME",
                         help="pairwise mode, real data: model names to filter to")
    parser.add_argument("--hf-token", default=None, help="pairwise mode, real data")
    parser.add_argument("--cache-dir", default=None, help="pairwise mode, real data")
    parser.add_argument("--min-pair-size", type=int, default=50, help="pairwise mode, real data")
    parser.add_argument("--inspect-csv", default=None,
                         help=f"pairwise mode, real data: path to CSV from collect_inspect_benchmarks.py "
                              f"(used by --data-source inspect/real; defaults to {DEFAULT_INSPECT_CSV!r})")

    # multiarm mode
    parser.add_argument("--k-arms", type=int, default=4, metavar="K", help="multiarm mode: number of arms")
    parser.add_argument("--multiarm-method", default=SMOOTH_BOOTSTRAP.name, metavar="METHOD",
                         choices=[BOOTSTRAP.name, BCA.name, BAYES_BOOTSTRAP.name, SMOOTH_BOOTSTRAP.name, PERMUTATION.name],
                         help="multiarm mode: pairwise_differences-family method used to derive raw p-values before correction")
    parser.add_argument("--multiarm-icc", type=float, default=0.20, metavar="ICC",
                         help="multiarm mode: ICC for build_multiarm_sources' shared truth/noise model "
                              "(same meaning as --icc-values in pairwise mode)")
    parser.add_argument("--multiarm-cohens-d", type=float, default=0.3, metavar="D",
                         help="multiarm mode: alt-condition effect size (Cohen's d) for build_multiarm_sources")

    # ppi mode
    parser.add_argument("--tests", nargs="+", choices=[m.name for m in PPI_TEST_METHODS], default=None, metavar="TEST",
                         help="ppi mode: restrict to these evalstats.tests names (default: all)")
    parser.add_argument("--ppi-n-boot", type=int, default=1000, metavar="N",
                         help="ppi mode: PPI bootstrap resample count")


def official_args(base_seed: int = 42) -> argparse.Namespace:
    """Canonical official-test preset: runs all three modes (pairwise, multiarm,
    ppi) back to back, mirroring sim_compare_pvalues.py's combined
    pairwise+multiarm run and sim_type_i_calibration.py's full 43-scenario
    11-test sweep. Synthetic only."""
    return argparse.Namespace(
        mode="all", reps=1000, alpha=0.05, seed=base_seed,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        data_source="synthetic", scenario_suite="expanded", eval_types=None, sizes=[10, 20, 30, 50, 75, 100],
        runs=1, statistic="mean",
        bootstrap_n=2000, icc_values=[0.05, 0.20, 0.40, 0.60, 0.80], cohens_d_values=[0.2, 0.4],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        k_arms=4, multiarm_method=SMOOTH_BOOTSTRAP.name, multiarm_icc=0.20, multiarm_cohens_d=0.3,
        tests=None, ppi_n_boot=2000,
    )


def quick_args(base_seed: int = 43, data_source: str = "synthetic") -> argparse.Namespace:
    """Fast sanity-check preset for --quick-test: runs all three modes
    (pairwise, multiarm, ppi) with cut-down sweeps/reps/tests for a quick pass
    that confirms the pipeline (incl. --latex output) still works. Restricts
    eval_types/tests/k_arms rather than sweeping the full catalog -- this is
    for pipeline confidence, not a representative result.
    ``data_source="real"`` (or 'openeval'/'inspect') restricts mode to
    'pairwise' -- the only mode whose sources come from --data-source;
    multiarm/ppi are synthetic-only (see README), so re-running them here
    would just recompute the identical synthetic sweep a second time. It also
    switches eval_types to 'binary': real-data pairwise sources are binary-
    only by construction (corpus_pair_to_ci_pair_source hardcodes
    eval_type="binary" for both openeval and inspect -- see README's "known
    exceptions"), so the synthetic variant's 'continuous' filter would leave
    zero sources. --quick-test calls this twice per case (synthetic, then
    real) so the real-data pairwise path doesn't go unexercised between
    --official-tests runs."""
    mode = "all" if data_source == "synthetic" else "pairwise"
    eval_types = ["continuous"] if data_source == "synthetic" else ["binary"]
    return argparse.Namespace(
        mode=mode, reps=3, alpha=0.05, seed=base_seed,
        progress="bar", plots="save", save_results="save", out_dir="simulations/out", plots_dir=None,
        data_source=data_source, scenario_suite="standard", eval_types=eval_types, sizes=[10, 30, 50],
        runs=1, statistic="mean",
        bootstrap_n=200, icc_values=[0.20], cohens_d_values=[0.3],
        benchmarks=None, models=None, hf_token=None, cache_dir=None, min_pair_size=50, inspect_csv=None,
        k_arms=3, multiarm_method=SMOOTH_BOOTSTRAP.name, multiarm_icc=0.20, multiarm_cohens_d=0.3,
        tests=[TTEST.name, MW.name], ppi_n_boot=200, latex=True,
    )


def run(args: argparse.Namespace) -> CaseResult:
    t0 = time.time()
    try:
        plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_paths: list[str] = []
        key_metrics: dict = {}
        modes = [args.mode] if args.mode != "all" else ["pairwise", "multiarm", "ppi"]

        if "pairwise" in modes:
            print(f"\npvalues simulation (pairwise, non-PPI) -- data_source={args.data_source}, statistic={args.statistic}")
            if args.data_source == "synthetic":
                sources = build_pair_sources(
                    suite=args.scenario_suite, icc_values=args.icc_values,
                    cohens_d_values=args.cohens_d_values, include_null=True,
                )
            else:
                runs = args.runs
                if runs != 1:
                    print("  Warning: real-data sources only support --runs 1 in this pass; forcing runs=1.")
                    runs = 1
                args = argparse.Namespace(**{**vars(args), "runs": runs})
                sources = build_real_pair_sources(
                    args.data_source, benchmarks=args.benchmarks, models=args.models,
                    hf_token=args.hf_token, cache_dir=args.cache_dir, min_pair_size=args.min_pair_size,
                    inspect_csv=args.inspect_csv,
                )
            if args.eval_types:
                requested = set(args.eval_types)
                sources = [s for s in sources if s.eval_type in requested]
            if not sources:
                raise ValueError("No CIPairSources left after filtering.")
            print(f"  {len(sources)} sources, sizes={args.sizes}, reps={args.reps}, alpha={args.alpha}")

            pw_results = run_pairwise_simulation(
                sources, sample_sizes=args.sizes, runs=args.runs, n_reps=args.reps, n_bootstrap=args.bootstrap_n,
                alpha=args.alpha, statistic=args.statistic, progress_mode=args.progress, seed=args.seed,
            )
            print_pairwise_report(pw_results, alpha=args.alpha)

            run_stem = f"pvalues_pairwise_{args.data_source}_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_pairwise(results=pw_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False))
            if args.plots == "save":
                plot_path = save_pairwise_typeI_power_plot(results=pw_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_typeI_power.png"))
                output_paths.append(plot_path)
                print(f"Saved plot: {plot_path}")

            null_rows = [r for r in pw_results if r.condition == "null"]
            type1 = float(np.mean([r.rejects / r.n_reps for r in null_rows])) if null_rows else float("nan")
            key_metrics["pairwise_n_results"] = len(pw_results)
            key_metrics["pairwise_mean_type1"] = type1

        if "multiarm" in modes:
            print(f"\npvalues simulation (multi-arm, non-PPI) -- k={args.k_arms}, method={args.multiarm_method}")
            ma_sources = build_multiarm_sources(
                suite=args.scenario_suite, icc=args.multiarm_icc, cohens_d=args.multiarm_cohens_d,
                eval_types=args.eval_types,
            )
            if not ma_sources:
                raise ValueError("No MultiArmSources left after filtering.")
            print(f"  {len(ma_sources)} sources, sizes={args.sizes}, reps={args.reps}, alpha={args.alpha}")

            ma_results = run_multiarm_simulation(
                ma_sources, sample_sizes=args.sizes, runs=args.runs, k_arms=args.k_arms, n_reps=args.reps,
                n_bootstrap=args.bootstrap_n, alpha=args.alpha, multiarm_method=args.multiarm_method,
                statistic=args.statistic, progress_mode=args.progress, seed=args.seed,
            )
            print_multiarm_report(ma_results, alpha=args.alpha)

            run_stem = f"pvalues_multiarm_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_multiarm(results=ma_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False))
            if args.plots == "save":
                plot_path = save_multiarm_fwer_power_plot(results=ma_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_fwer_power.png"))
                output_paths.append(plot_path)
                print(f"Saved plot: {plot_path}")

            null_rows = [r for r in ma_results if r.condition == "null"]
            fwer = sum(r.any_reject for r in null_rows) / sum(r.n_reps for r in null_rows) if null_rows else float("nan")
            key_metrics["multiarm_n_results"] = len(ma_results)
            key_metrics["multiarm_mean_fwer"] = float(fwer)

        if "ppi" in modes:
            active_tests = args.tests if args.tests else [m.name for m in PPI_TEST_METHODS]
            print(f"\npvalues simulation (PPI-corrected) -- tests={active_tests}")
            jb_sources = build_judge_bias_sources()
            print(f"  {len(jb_sources)} scenarios, reps={args.reps}, n_boot={args.ppi_n_boot}, alpha={args.alpha}")

            ppi_results = run_ppi_simulation(
                jb_sources, active_tests=active_tests, n_reps=args.reps, n_boot=args.ppi_n_boot,
                progress_mode=args.progress, seed=args.seed,
            )
            print_ppi_report(ppi_results, alpha=args.alpha)

            run_stem = f"pvalues_ppi_reps{args.reps}_{stamp}"
            if args.save_results == "save":
                output_paths += save_results_artifacts_ppi(results=ppi_results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem, latex=getattr(args, "latex", False))
            if args.plots == "save":
                plot_path = save_ppi_typeI_plot(results=ppi_results, alpha=args.alpha, out_path=str(Path(plots_dir) / f"{run_stem}_typeI_corrected_vs_uncorrected.png"))
                output_paths.append(plot_path)
                print(f"Saved plot: {plot_path}")

            c_tot = sum(r.corrected_rejects for r in ppi_results)
            u_tot = sum(r.uncorrected_rejects for r in ppi_results)
            n_tot = sum(r.n_reps for r in ppi_results)
            key_metrics["ppi_n_results"] = len(ppi_results)
            key_metrics["ppi_mean_corrected_type1"] = float(c_tot / n_tot) if n_tot else float("nan")
            key_metrics["ppi_mean_uncorrected_type1"] = float(u_tot / n_tot) if n_tot else float("nan")

        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics=key_metrics, duration_s=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return CaseResult(case_name=CASE_NAME, status="error", error=str(exc), duration_s=time.time() - t0)
