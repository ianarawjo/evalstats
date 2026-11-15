"""Judge alignment validation and MC-based uncertainty propagation.

Provides :func:`judge_alignment` and :class:`AlignmentResult` for
characterising how well an LLM judge aligns with human graders, and for
propagating that uncertainty into downstream comparisons via Monte-Carlo
imputation of latent human labels.
"""
from __future__ import annotations

import math
import warnings
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, pearsonr, spearmanr, norm
from scipy.stats import ConstantInputWarning


def _quiet_corr(fn, a, b) -> float:
    """Correlation without scipy's constant-input warning: a bootstrap
    resample can be constant, and NaN is the right answer there."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        r, _ = fn(a, b)
    return float(r)


def _quiet_ks_2samp(a, b):
    """ks_2samp without the 'exact calculation unsuccessful, switching to
    asymp' RuntimeWarning; the asymptotic p-value is what gets used."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ks_2samp(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# AlignmentResult
# ─────────────────────────────────────────────────────────────────────────────

class AlignmentResult:
    """Carries a fitted calibration model and alignment diagnostics.

    Created by :func:`judge_alignment`.  Pass it to
    ``compare(alignment={metric_col: result})`` to widen confidence intervals
    to account for LLM-judge measurement uncertainty via Monte-Carlo imputation.

    Attributes
    ----------
    llm_metric : str
        Column name of the LLM judge scores.
    human_col : str
        Column name of the human-label scores.
    score_type : str
        Detected score type: ``"binary"``, ``"likert"``, ``"continuous"``,
.
    n_labeled : int
        Number of items with human labels (alignment set size).
    n_total : int
        Total number of items in the dataset.
    selection : str
        How the labeled subset was chosen, as declared by the caller via
        :func:`judge_alignment`'s ``selection=`` -- ``"random"``,
        ``"stratified"``, ``"manual"``, or ``"unknown"`` (the default when
        not specified). Every correction :func:`judge_alignment` /
        ``compare(alignment=...)`` applies assumes the labeled subset is a
        random sample of the full item pool (MCAR, "missing completely at
        random"); anything other than ``"random"`` means that assumption
        is either known-violated or unconfirmed, and a warning is raised
        at call time -- see :attr:`representativeness` and
        :meth:`summary` for the diagnostics that check for this in practice.
    alignment_metrics : dict
        Point estimates and bootstrap CIs for each alignment metric.
    representativeness : dict
        Representativeness check results (distribution, slice columns, and
        label-position contiguity).
    bias_check : dict or None
        For likert/continuous score types, compares the correlation-type
        metric (weighted κ or Pearson r) against ICC(2,1) to flag whether the
        judge is systematically biased in absolute scale despite tracking
        human relative ordering.  ``None`` for binary score types, where ICC
        isn't computed.
    test : str or None
        The test named via ``test=``, if any -- see :func:`judge_alignment`.
        For a single condition, only ``"mean_estimate"`` is valid (no
        comparison to linearize against).
    test_metric : dict or None
        Set iff ``test`` was given: the correlation entry (same shape as
        ``alignment_metrics``' entries, with ``multiplier``/``n_eff`` added)
        that governs ``test``'s PPI variance reduction. For
        ``test="mean_estimate"`` this is identical to ``alignment_metrics
        ["pearson_r"]``. :attr:`n_eff`/:attr:`multiplier` read from here.
    per_condition_metrics : dict or None
        Only set for form 1 (an ``EvalResults`` with a ``model``/condition
        column). The same headline metric ``bias_check`` uses (e.g.
        weighted κ for likert), recomputed separately within each
        condition's labeled items -- point estimates only, no CI (see
        :func:`_compute_per_condition_alignment`'s docstring for why).
        Shape: ``{"column": str, "spread": float, "conditions": {label:
        {"n": int, "too_few": bool, "label": str, "estimate": float}}}``.
        A pooled metric can look fine while a judge is biased *differently*
        per condition -- generous to one, stingy to another -- which is
        invisible to any pooled statistic; this is printed automatically in
        :meth:`summary` so that risk isn't discovered only after a
        PPI-corrected ``compare()`` disagrees with the naive one. ``None``
        when there's no condition column, or fewer than 2 conditions have
        any labeled items.
    """

    def __init__(
        self,
        *,
        llm_metric: str,
        human_col: str,
        score_type: str,
        n_labeled: int,
        n_total: int,
        calibration: dict,
        alignment_metrics: dict,
        representativeness: dict,
        bias_check: Optional[dict] = None,
        selection: str = "unknown",
        test: Optional[str] = None,
        test_metric: Optional[dict] = None,
        per_condition_metrics: Optional[dict] = None,
    ) -> None:
        self.llm_metric = llm_metric
        self.human_col = human_col
        self.score_type = score_type
        self.n_labeled = n_labeled
        self.n_total = n_total
        self._calibration = calibration
        self.alignment_metrics = alignment_metrics
        self.representativeness = representativeness
        self.bias_check = bias_check
        self.selection = selection
        self.test = test
        self.test_metric = test_metric
        self.per_condition_metrics = per_condition_metrics
        self.comparison: Optional[dict] = None
        self.comparison_error: Optional[str] = None
        self.score_type_source: str = "inferred"   # or "declared"
        self.grid_step: Optional[float] = None

    @property
    def n_eff(self) -> float:
        """Effective human-label sample size for the ``test=`` you
        specified -- see :func:`judge_alignment`'s ``test=`` docs. Raises
        if you didn't pass ``test=``."""
        return self._require_test()["n_eff"]

    @property
    def multiplier(self) -> float:
        """Label-efficiency savings multiplier for the ``test=`` you
        specified. Raises if you didn't pass ``test=``."""
        return self._require_test()["multiplier"]

    def _require_test(self) -> dict:
        if self.test_metric is None:
            raise ValueError(
                "No test= was given to judge_alignment(), so there's no single "
                "n_eff/multiplier answer -- inspect .alignment_metrics directly "
                "(raw Pearson/Spearman r, not test-specific), or re-call with "
                "test='mean_estimate'."
            )
        return self.test_metric

    # ── sampling ─────────────────────────────────────────────────────────────

    def _sample_imputed_scores(
        self,
        llm_scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample one realisation of latent human labels from the calibration posterior.

        Draws calibration parameters from their Bayesian posterior, then
        samples a human label for each item conditioned on its LLM score.
        Both sources of uncertainty (parameter uncertainty and item-level
        noise) are propagated.

        Parameters
        ----------
        llm_scores : np.ndarray
            1-D array of LLM judge scores for all items, in dataset row order.
        rng : np.random.Generator
            Random number generator (advanced each call).

        Returns
        -------
        np.ndarray
            Imputed human scores, same shape as ``llm_scores``.
        """
        cal = self._calibration
        n = len(llm_scores)
        imputed = np.empty(n, dtype=float)

        if cal["type"] == "binary":
            # Sample Bernoulli probability for each LLM class from Beta posterior
            p = {l: rng.beta(cal["alpha"][l], cal["beta"][l]) for l in cal["classes"]}
            fallback_p = float(np.mean(list(p.values())))
            for i, s in enumerate(llm_scores):
                prob = p.get(int(round(s)), fallback_p)
                imputed[i] = float(rng.binomial(1, prob))

        elif cal["type"] == "likert":
            # Sample category probabilities for each LLM class from Dirichlet posterior
            probs = {l: rng.dirichlet(cal["concentration"][l]) for l in cal["llm_cats"]}
            human_cats = np.array(cal["human_cats"])
            n_cats = len(human_cats)
            fallback = np.ones(n_cats) / n_cats
            for i, s in enumerate(llm_scores):
                cat_probs = probs.get(s, fallback)
                idx = rng.choice(n_cats, p=cat_probs)
                imputed[i] = float(human_cats[idx])

        else:  # "continuous"
            # Sample (intercept, slope, σ²) from Normal-Inverse-Gamma posterior
            # σ² ~ InvGamma(an, bn); coefs | σ² ~ N(mun, σ² * Vn)
            sigma2 = 1.0 / rng.gamma(shape=cal["an"], scale=1.0 / cal["bn"])
            sigma2 = max(float(sigma2), 1e-10)
            cov = sigma2 * cal["Vn"] + np.eye(2) * 1e-12
            L = np.linalg.cholesky(cov)
            coefs = cal["mun"] + L @ rng.standard_normal(2)
            mu_pred = coefs[0] + coefs[1] * llm_scores
            imputed = mu_pred + np.sqrt(sigma2) * rng.standard_normal(n)

        return imputed

    # ── display ──────────────────────────────────────────────────────────────

    def summary(self, verbose: bool = False) -> None:
        """Print an alignment and representativeness report.

        Parameters
        ----------
        verbose : bool
            If ``False`` (default), print a short, plain-language report:
            one line per check/metric, with an explanation only where
            something looks off. Aimed at readers who don't need the
            statistical background spelled out every time.
            If ``True``, print the full report: every metric's definition,
            why it was chosen, how to interpret it, and citation-ready
            wording for a paper.
        """
        if verbose:
            self._summary_verbose()
        else:
            self._summary_simple()

    def _header(self) -> None:
        pct = 100.0 * self.n_labeled / self.n_total if self.n_total > 0 else 0.0
        print("Judge alignment report")
        print("─" * 58)
        print(
            f"Alignment set  : {self.n_labeled} of {self.n_total} items "
            f"have human labels ({pct:.1f}%)"
        )
        sel_icon = "✓" if self.selection == "random" else "⚠ "
        print(f"Label selection: {sel_icon} {self.selection}")
        print(
            "Note: corrections below assume the labeled subset is a random "
            "sample of the full item pool (MCAR). See 'Representativeness'."
        )
        print()

    def _summary_simple(self) -> None:
        self._header()

        # This check compares a correlation against ICC(2,1), so the only thing
        # it can see is a systematic shift or compression of the judge's raw
        # scores. PPI corrects for that regardless, so the result never changes
        # whether to correct. Only the FAILING branch is printed here, because it
        # says something about the judge worth knowing; a passing result is not
        # evidence that correction can be skipped -- the bias PPI is most needed
        # for errs in different directions across conditions, which is invisible
        # to any pooled statistic including this one. Both branches stay in
        # summary(verbose=True), where the surrounding text supplies that context.
        if self.bias_check is not None and not self.bias_check["passed"]:
            bc = self.bias_check
            print(
                f"⚠ Possible judge scale bias: {bc['corr_label']} = "
                f"{bc['corr_estimate']:.2f} but ICC(2,1) = {bc['icc_estimate']:.2f}. "
                "The judge ranks items like humans do, but its raw scores "
                "look shifted or compressed relative to human scores."
            )
            print(
                "  This affects raw judge scores only; a PPI-corrected "
                "comparison (compare(alignment=...)) already absorbs it. "
                "Run .summary(verbose=True) for the full check."
            )
            print()

        rep = self.representativeness
        rep_failed = [
            (k, v) for k, v in rep.items() if not v["passed"]
        ]
        if rep_failed:
            print("⚠ Representativeness: the labeled sample may not be representative")
            for key, val in rep_failed:
                print(f"    - {_rep_check_display_name(key)}: {val['message']}")
        else:
            print("✓ Representativeness: labeled items look like the full item pool")
        print()

        self._print_overall_block()
        self._print_condition_agreement_block()
        self._print_efficiency_block()
        print("Run .summary(verbose=True) for definitions, rationale, and")
        print("citation-ready wording for each check above.")
        print("─" * 58)

    # ── report blocks ────────────────────────────────────────────────────────

    def _score_type_phrase(self) -> str:
        """What score type was used and whether it was inferred or declared."""
        base = self.score_type
        if base == "likert":
            step = self.grid_step
            if step is None or abs(step - 1.0) < 1e-9:
                base = "likert, whole-number steps"
            elif abs(step - 0.5) < 1e-9:
                base = "likert, half-point steps"
            else:
                base = f"likert, steps of {step:g}"
        return f"score type {self.score_type_source}: {base}"

    def _print_overall_block(self) -> None:
        pooled = self.per_condition_metrics is not None
        lead = "Overall agreement (all conditions pooled; " if pooled else "Agreement ("
        print(f"{lead}{self._score_type_phrase()}):")
        label_width = max(
            (len(entry.get("label", "")) for entry in self.alignment_metrics.values()),
            default=20,
        )
        for entry in self.alignment_metrics.values():
            label = entry.get("label", "")
            band = entry.get("band")
            tail = f"  {band}" if band else ""
            print(f"  {label:<{label_width}} {entry['estimate']:6.2f}  "
                  f"[{entry['ci_low']:5.2f}, {entry['ci_high']:5.2f}]{tail}")
        print()

    def _print_condition_agreement_block(self, *, verbose: bool = False) -> None:
        """Agreement by condition. See :func:`_compute_per_condition_alignment`: a pooled
        number can look fine while hiding a judge biased differently per condition,
        which no pooled statistic -- including the scale-bias check -- can catch."""
        pc = self.per_condition_metrics
        if pc is None:
            return
        conds = pc["conditions"]
        first = next((e for e in conds.values() if not e.get("too_few")), None)
        label = first["label"] if first else "metric"
        short = "κ" if "κ" in label else label
        with_r = short != "Pearson r"
        width = max((len(str(c)) for c in conds), default=9)
        width = max(width, len("condition"))
        print(f"Agreement by condition ({pc['column']!r}):")
        if verbose:
            print("  The pooled numbers above stay high even if the judge is generous to one "
                  "condition and stingy to another, as long as the two roughly cancel out.")
        print(f"  {'condition':<{width}}  {'labeled':>7}  {short:>6}" + (f"  {'r':>5}" if with_r else ""))
        for cond, entry in conds.items():
            if entry.get("too_few"):
                print(f"  {str(cond):<{width}}  {entry['n']:>7}  too few human labels to estimate")
            else:
                row = f"  {str(cond):<{width}}  {entry['n']:>7}  {entry['estimate']:>6.2f}"
                if with_r:
                    row += f"  {entry['r']:>5.2f}"
                print(row)
        spread = pc["spread"]
        if spread >= _PER_CONDITION_SPREAD_FLAG:
            print(f"  ⚠ Spread of {spread:.2f} across conditions on the same scale as the "
                  "pooled metric above. The judge may be tracking humans unevenly.")
        else:
            print("  ✓ No condition stands out from the others on this metric.")
        print()

    def _print_efficiency_block(self, *, verbose: bool = False) -> None:
        """rho^2 and N_eff for each result compare() will report: each condition's mean,
        the omnibus test, and each pair. rho^2 is a squared judge-human correlation on
        the quantity being corrected, so it differs by result and each line names it."""
        pc, c, tm = self.per_condition_metrics, self.comparison, self.test_metric
        if pc is None and c is None and tm is None and not self.comparison_error:
            return
        print("Label efficiency: what the judge adds to each result compare() will report")
        print("  rho^2 is the squared judge-human correlation on the quantity being estimated, so it differs by result.")

        projects = False
        if pc is not None:
            rows = [(k, e) for k, e in pc["conditions"].items() if not e.get("too_few")]
            projects = any(e["n_total"] > e["n"] for _, e in rows)
            title = "Mean of each condition"
            w = max([len(title)] + [len(str(k)) + 2 for k, _ in rows])
            print(f"  {title:<{w}}  {'rho^2':>6}" + (f"  {'N_eff':>6}" if projects else ""))
            for k, e in rows:
                line = f"    {str(k):<{w - 2}}  {e['rho2']:>6.2f}"
                if projects:
                    line += f"  {e['n_eff']:>6.0f}"
                print(line)
        elif tm is not None and "rho2" in tm:
            projects = self.n_total > self.n_labeled
            ci = ""
            if np.isfinite(tm["rho2_ci_low"]) and np.isfinite(tm["rho2_ci_high"]):
                ci = f"  95% CI [{tm['rho2_ci_low']:.2f}, {tm['rho2_ci_high']:.2f}]"
            line = f"  Mean:  rho^2 {tm['rho2']:.2f}{ci}"
            if projects:
                line += f"   N_eff {tm['n_eff']:.0f}"
            print(line)

        if c is not None:
            projects = projects or c["projects_n_eff"]
            for t in c["tests"]:
                name = _TEST_NAMES.get(t["test"], t["test"])
                if "metric" in t:
                    m = t["metric"]
                    ci = ""
                    if np.isfinite(m["rho2_ci_low"]) and np.isfinite(m["rho2_ci_high"]):
                        ci = f"  95% CI [{m['rho2_ci_low']:.2f}, {m['rho2_ci_high']:.2f}]"
                    line = f"  Omnibus, {name}:  rho^2 {m['rho2']:.2f}{ci}"
                    if c["projects_n_eff"]:
                        line += f"   N_eff {m['n_eff_per_condition']:.0f} per condition"
                    print(line)
                else:
                    pairs = t["pairs"]
                    proj = c["projects_n_eff"]
                    if len(pairs) <= _MAX_PAIRS_LISTED or verbose:
                        title = f"Pairs, {name}"
                        labels = [f"{a} vs {b}" for a, b in pairs]
                        w = max([len(title)] + [len(x) + 2 for x in labels])
                        print(f"  {title:<{w}}  {'rho^2':>6}" + (f"  {'N_eff':>6}" if proj else ""))
                        for lab, m in zip(labels, pairs.values()):
                            line = f"    {lab:<{w - 2}}  {m['rho2']:>6.2f}"
                            if proj:
                                line += f"  {m['n_eff_per_condition']:>6.0f}"
                            print(line)
                    else:
                        r2 = [m["rho2"] for m in pairs.values() if np.isfinite(m["rho2"])]
                        line = (f"  Pairs, {name}:  rho^2 {min(r2):.2f} to {max(r2):.2f} "
                                f"across {len(pairs)} pairs (summary(verbose=True) lists each)")
                        print(line)
        elif self.comparison_error:
            print(f"  Tests compare() will run: not estimated. {self.comparison_error}")

        if projects:
            counts = (list(c["n_labeled"].values()) if c is not None
                      else [e["n"] for e in pc["conditions"].values()] if pc is not None
                      else [self.n_labeled])
            same = len(set(counts)) == 1
            had = (f"{counts[0]} per condition" if same and c is not None
                   else f"{min(counts)} to {max(counts)} per condition" if c is not None else f"{counts[0]}")
            print(f"  N_eff: human labels per condition with the same power; you labeled {had}.")
        print()

    def _summary_verbose(self) -> None:
        self._header()

        if self.bias_check is not None and not self.bias_check["passed"]:
            print(f"⚠ Judge bias flag: {self.bias_check['message']}")
            print("   (see 'Bias diagnostics' below for details)")
            print()

        # Representativeness
        rep = self.representativeness

        def _print_check(title: str, val: dict) -> None:
            icon = "✓" if val["passed"] else "⚠ "
            print(f"  {title}: {icon}  {val['message']}")
            what = val.get("what")
            why = val.get("why")
            interpretation = val.get("interpretation")
            if what:
                print(f"      -> What this checks: {what}")
            if why:
                print(f"      -> Why it was computed in this case: {why}")
            if interpretation:
                print(f"      -> How to interpret this result: {interpretation}")
            print()

        print("Representativeness diagnostics:")
        dist = rep.get("score_distribution")
        if dist:
            _print_check("Score distribution", dist)
        contiguity = rep.get("label_contiguity")
        if contiguity:
            _print_check("Label position contiguity", contiguity)
        for key, val in rep.items():
            if key in ("score_distribution", "label_contiguity"):
                continue
            if key.startswith("slice_"):
                col = key[len("slice_"):]
                _print_check(f"{col!r}", val)

        # Alignment metrics
        pooled = self.per_condition_metrics is not None
        lead = "Overall agreement (all conditions pooled; " if pooled else "Agreement ("
        print(f"{lead}{self._score_type_phrase()}):")
        print()
        verbose_label_width = max(
            (len(entry.get("label", "")) for entry in self.alignment_metrics.values()),
            default=24,
        )
        for entry in self.alignment_metrics.values():
            label = entry.get("label", "")
            est = entry["estimate"]
            lo = entry["ci_low"]
            hi = entry["ci_high"]
            print(f"  {label:<{verbose_label_width}}: {est:.3f}  [{lo:.3f}, {hi:.3f}]")
            what = entry.get("what")
            why = entry.get("why")
            interpretation = entry.get("interpretation")
            example = entry.get("example")
            if what:
                print(f"      -> What this metric is: {what}")
            if why:
                print(f"      -> Why it was computed in this case: {why}")
            if interpretation:
                print(f"      -> How to interpret this result: {interpretation}")
            if example:
                print(f"      -> Example paper reporting: {example}")
            print()

        self._print_condition_agreement_block(verbose=True)
        self._print_efficiency_block(verbose=True)

        if self.bias_check is not None:
            print("Bias diagnostics:")
            _print_check("Judge scale bias (correlation vs. ICC)", self.bias_check)

        print("─" * 58)

    def __repr__(self) -> str:
        return (
            f"AlignmentResult(metric={self.llm_metric!r}, "
            f"score_type={self.score_type!r}, "
            f"n_labeled={self.n_labeled}/{self.n_total})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Calibration model fitting
# ─────────────────────────────────────────────────────────────────────────────

def _fit_calibration(
    llm_labels: np.ndarray,
    human_labels: np.ndarray,
    score_type: str,
) -> dict:
    if score_type == "binary":
        return _fit_binary(llm_labels, human_labels)
    elif score_type == "likert":
        return _fit_likert(llm_labels, human_labels)
    else:
        return _fit_continuous(llm_labels, human_labels, score_type)


def _fit_binary(llm: np.ndarray, human: np.ndarray) -> dict:
    prior = 1.0  # Beta(1,1) uniform prior
    alpha_post: dict = {}
    beta_post: dict = {}
    for l in [0, 1]:
        mask = (llm == l)
        n_l = int(mask.sum())
        n_pos = int((human[mask] == 1).sum()) if n_l > 0 else 0
        alpha_post[l] = float(n_pos + prior)
        beta_post[l] = float((n_l - n_pos) + prior)
    return {"type": "binary", "classes": [0, 1], "alpha": alpha_post, "beta": beta_post}


def _fit_likert(llm: np.ndarray, human: np.ndarray) -> dict:
    llm_cats = sorted(set(llm.tolist()))
    human_cats = sorted(set(llm.tolist()) | set(human.tolist()))
    n_human = len(human_cats)
    prior_conc = 1.0 / n_human  # near-uniform Dirichlet
    concentration: dict = {}
    for l in llm_cats:
        mask = (llm == l)
        counts = np.array(
            [(human[mask] == k).sum() for k in human_cats], dtype=float
        )
        concentration[l] = counts + prior_conc
    return {
        "type": "likert",
        "llm_cats": llm_cats,
        "human_cats": human_cats,
        "concentration": concentration,
    }


def _fit_continuous(llm: np.ndarray, human: np.ndarray, score_type: str) -> dict:
    n = len(llm)
    X = np.column_stack([np.ones(n), llm])
    y = human.astype(float)

    # Uninformative Normal-Inverse-Gamma prior
    V0_inv = np.eye(2) * 1e-6
    mu0 = np.zeros(2)
    a0, b0 = 1.0, 1e-6

    XtX = X.T @ X
    Xty = X.T @ y
    Vn_inv = V0_inv + XtX
    Vn = np.linalg.inv(Vn_inv)
    mun = Vn @ (V0_inv @ mu0 + Xty)
    an = a0 + n / 2.0
    bn = float(b0 + 0.5 * (y @ y + mu0 @ V0_inv @ mu0 - mun @ Vn_inv @ mun))
    bn = max(bn, 1e-10)
    return {"type": score_type, "Vn": Vn, "mun": mun, "an": an, "bn": bn}


# ─────────────────────────────────────────────────────────────────────────────
# Alignment metrics
# ─────────────────────────────────────────────────────────────────────────────

def _interpret_kappa(est: float, lo: float, hi: float, n: int, label: str) -> tuple[str, str, str]:
    """Landis & Koch (1977) benchmarks for kappa-type statistics."""
    if est < 0:
        band = "poor"
    elif est <= 0.20:
        band = "slight"
    elif est <= 0.40:
        band = "fair"
    elif est <= 0.60:
        band = "moderate"
    elif est <= 0.80:
        band = "substantial"
    else:
        band = "almost perfect"
    band_phrase = f"{band} agreement"
    interpretation = f"{band} agreement (Landis & Koch, 1977 benchmarks)"
    example = (
        f'"{label} = {est:.2f}, 95% CI [{lo:.2f}, {hi:.2f}] (n={n}), indicating '
        f'{band} agreement between the LLM judge and human raters, per the Landis '
        f'& Koch (1977) benchmarks."'
    )
    return band_phrase, interpretation, example


def _interpret_corr(est: float, lo: float, hi: float, n: int, label: str) -> tuple[str, str, str]:
    """Cohen (1988) conventions for correlation-coefficient magnitude."""
    a = abs(est)
    if a < 0.10:
        band = "negligible"
    elif a < 0.30:
        band = "small"
    elif a < 0.50:
        band = "medium"
    else:
        band = "large"
    direction = "positive" if est >= 0 else "negative"
    band_phrase = f"{band} {direction} correlation"
    interpretation = f"{band_phrase} (Cohen, 1988 conventions)"
    example = (
        f'"{label} = {est:.2f}, 95% CI [{lo:.2f}, {hi:.2f}] (n={n}), a {band} '
        f'{direction} correlation between the LLM judge and human scores (Cohen, '
        f'1988 conventions)."'
    )
    return band_phrase, interpretation, example


def _interpret_icc(est: float, lo: float, hi: float, n: int, label: str) -> tuple[str, str, str]:
    """Koo & Li (2016) benchmarks for ICC magnitude."""
    if est < 0.50:
        band = "poor"
    elif est < 0.75:
        band = "moderate"
    elif est < 0.90:
        band = "good"
    else:
        band = "excellent"
    band_phrase = f"{band} absolute agreement"
    interpretation = f"{band_phrase} (Koo & Li, 2016 benchmarks)"
    example = (
        f'"{label} = {est:.2f}, 95% CI [{lo:.2f}, {hi:.2f}] (n={n}), indicating '
        f'{band} absolute agreement between the LLM judge and human raters, per '
        f'Koo & Li (2016) benchmarks."'
    )
    return band_phrase, interpretation, example


def _interpret_pct_agreement(est: float, lo: float, hi: float, n: int, label: str) -> tuple[Optional[str], str, str]:
    interpretation = (
        "no universally-agreed threshold exists for raw percent agreement. Read it "
        "alongside Cohen's κ, since it does not correct for chance and can look high "
        "purely from imbalanced label classes"
    )
    example = (
        f'"the LLM judge matched human labels on {est * 100:.1f}% of items, 95% CI '
        f'[{lo * 100:.1f}%, {hi * 100:.1f}%] (n={n})."'
    )
    return None, interpretation, example


def _bootstrap_ci_2(
    fn,
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n = len(a)
    obs = float(fn(a, b))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = fn(a[idx], b[idx])
    lo = float(np.percentile(boot, 100.0 * alpha / 2))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2)))
    return obs, lo, hi


def _icc_21(a: np.ndarray, b: np.ndarray) -> float:
    """Shrout & Fleiss (1979) ICC(2,1): two-way random effects, single rater,
    absolute agreement, for exactly two raters (``a``, ``b``).

    Unlike Pearson/Spearman r or weighted kappa's category-index distance,
    this is sensitive to a systematic offset or scale mismatch between the
    two raters — it measures whether they land on the same absolute values,
    not just whether they move together.
    """
    n = len(a)
    data = np.column_stack([a, b]).astype(float)
    k = 2
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    df_row = max(n - 1, 1)
    SSR = k * np.sum((row_means - grand_mean) ** 2)
    SSC = n * np.sum((col_means - grand_mean) ** 2)  # (k-1) == 1
    SST = np.sum((data - grand_mean) ** 2)
    SSE = SST - SSR - SSC

    MSR = SSR / df_row
    MSC = SSC
    MSE = SSE / df_row  # (n-1)(k-1) == n-1

    denom = MSR + MSE + 2.0 * (MSC - MSE) / n
    if denom <= 1e-12:
        return 1.0
    return float((MSR - MSE) / denom)


def _bootstrap_ci_gap(
    fn_corr,
    fn_icc,
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for (correlation-type metric − ICC(2,1)).

    Resamples items once per draw and evaluates both statistics on the same
    resample, so the CI reflects the sampling distribution of the *gap*
    itself, not the (looser, more conservative) union of two independently
    bootstrapped CIs.
    """
    n = len(a)
    obs = float(fn_corr(a, b) - fn_icc(a, b))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = fn_corr(a[idx], b[idx]) - fn_icc(a[idx], b[idx])
    lo = float(np.percentile(boot, 100.0 * alpha / 2))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2)))
    return obs, lo, hi


def _build_bias_check(
    corr_label: str,
    corr_est: float,
    icc_est: float,
    gap_est: float,
    gap_lo: float,
    gap_hi: float,
) -> dict:
    """Package the correlation-vs-ICC(2,1) discrepancy check into a dict with
    the same shape as the representativeness checks, so it can be rendered
    with the same ``_print_check`` helper in ``AlignmentResult.summary()``.
    """
    flagged = gap_lo > 0.0
    if flagged:
        message = (
            f"possible judge bias: {corr_label} = {corr_est:.3f} but "
            f"ICC(2,1) = {icc_est:.3f} (gap = {gap_est:.3f}, 95% CI "
            f"[{gap_lo:.3f}, {gap_hi:.3f}], excludes 0)"
        )
    else:
        message = (
            f"no evidence of scale bias: {corr_label} = {corr_est:.3f} and "
            f"ICC(2,1) = {icc_est:.3f} are consistent (gap 95% CI "
            f"[{gap_lo:.3f}, {gap_hi:.3f}] includes 0)"
        )
    what = (
        f"Compares {corr_label}, which is insensitive to a systematic offset "
        "or scale difference between judge and human scores, against "
        "ICC(2,1), which penalizes exactly that. A paired bootstrap CI is "
        "used so the comparison reflects the sampling distribution of the "
        "gap itself rather than the union of two separate CIs."
    )
    why = (
        "A correlation-type metric can look strong even when a judge is "
        "systematically shifted or compressed relative to human scores, "
        "since it only requires the two to move together. This check exists "
        "to catch that failure mode before it's mistaken for genuine "
        "agreement."
    )
    if flagged:
        interpretation = (
            "the judge tracks human relative ordering but disagrees on "
            "absolute scale. Treat raw judge scores as biased; consider "
            "using the Bayesian calibration model fit by judge_alignment "
            "(e.g. via compare(alignment=...)) to correct for it before "
            "drawing conclusions from raw judge scores"
        )
    else:
        interpretation = (
            "the correlation and absolute-agreement metrics tell a "
            "consistent story. No sign that the judge's ranking ability is "
            "masking a scale or offset problem"
        )
    return {
        "passed": not flagged,
        "message": message,
        "corr_label": corr_label,
        "corr_estimate": corr_est,
        "icc_estimate": icc_est,
        "gap": gap_est,
        "gap_ci_low": gap_lo,
        "gap_ci_high": gap_hi,
        "what": what,
        "why": why,
        "interpretation": interpretation,
    }


_VALID_SCORE_TYPES = ("binary", "likert", "continuous")


def _resolve_score_type(
    declared: Optional[str], auto: str, llm: np.ndarray, human: np.ndarray,
) -> tuple[str, Optional[float]]:
    """Pick the score type and, for ordinal data, the grid step it sits on.

    ``declared`` (the caller's ``score_type=``) always wins. Otherwise whole-number
    scales are ``likert`` as before, and so is any other bounded quantization grid
    (e.g. half-point scores) -- the same detector ``compare()`` uses -- so that
    half-point ratings get an ordinal metric rather than being demoted to
    ``continuous``. The returned step is ``None`` unless the type is ``likert``.
    """
    from evalstats.config import MAX_DISCRETE_LEVELS
    from evalstats.core.resampling import detect_quantization_step

    if declared is not None and declared not in _VALID_SCORE_TYPES:
        raise ValueError(
            f"score_type={declared!r} -- expected one of {list(_VALID_SCORE_TYPES)}."
        )
    score_type = declared or auto
    if score_type != "likert" and not (declared is None and auto == "continuous"):
        return score_type, None

    pooled = np.concatenate([np.asarray(llm, dtype=float), np.asarray(human, dtype=float)])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size and bool(np.all(pooled == np.floor(pooled))):
        step: Optional[float] = 1.0
    else:
        step = detect_quantization_step(pooled)
    if step is not None and pooled.size:
        levels = (float(pooled.max()) - float(pooled.min())) / step + 1
        if levels > MAX_DISCRETE_LEVELS:
            step = None

    if declared == "likert" and step is None:
        raise ValueError(
            "score_type='likert' but the judge and human scores don't sit on a "
            f"grid of at most {MAX_DISCRETE_LEVELS} levels (e.g. whole numbers or "
            "half points). Round the scores to the scale's grid first."
        )
    if declared is None and step is None:
        return "continuous", None
    return "likert", step



def _compute_alignment_metrics(
    llm: np.ndarray,
    human: np.ndarray,
    score_type: str,
    *,
    alpha: float = 0.05,
    rng: np.random.Generator,
    ci: bool = True,
    grid_step: Optional[float] = None,
) -> dict:
    metrics: dict = {}

    # ci=False skips every bootstrap CI on the alignment metrics and reports
    # NaN bounds, keeping only the (deterministic, closed-form) point
    # estimates. Each CI is 2000 resamples of its metric, so this is the bulk
    # of judge_alignment()'s cost. Intended for callers that consume only the
    # estimates -- notably compare(alignment=...), whose PPI correction reads
    # the point estimates alone -- and for large simulation sweeps.
    if ci:
        _ci2, _cigap = _bootstrap_ci_2, _bootstrap_ci_gap
    else:
        def _ci2(fn, a, b, **_kw):
            return float(fn(a, b)), float("nan"), float("nan")

        def _cigap(fn_corr, fn_icc, a, b, **_kw):
            return float(fn_corr(a, b)) - float(fn_icc(a, b)), float("nan"), float("nan")

    if score_type == "binary":
        def agree(a, b):
            return float(np.mean(a == b))

        def kappa(a, b):
            p_o = float(np.mean(a == b))
            p_e = float(
                np.mean(a == 1) * np.mean(b == 1)
                + np.mean(a == 0) * np.mean(b == 0)
            )
            return (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0

        est, lo, hi = _ci2(agree, llm, human, alpha=alpha, rng=rng)
        band, interp, example = _interpret_pct_agreement(est, lo, hi, len(llm), "Percent agreement")
        metrics["percent_agreement"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Percent agreement",
            "band": band,
            "what": (
                "The fraction of items where the LLM judge's label exactly matches "
                "the human label."
            ),
            "why": (
                "Included as an intuitive baseline agreement measure alongside "
                "Cohen's κ, since your judge produces binary (0/1) labels."
            ),
            "interpretation": interp,
            "example": example,
        }
        est, lo, hi = _ci2(kappa, llm, human, alpha=alpha, rng=rng)
        band, interp, example = _interpret_kappa(est, lo, hi, len(llm), "Cohen's κ")
        metrics["cohens_kappa"] = {
            "estimate": est, "ci_low": lo, "ci_high": hi,
            "label": "Cohen's κ",
            "band": band,
            "what": (
                "Percent agreement adjusted for the rate of agreement expected from "
                "two raters guessing at random, given the observed marginal label "
                "rates (Cohen, 1960)."
            ),
            "why": (
                "Your judge produces binary labels, so this nominal-data reliability "
                "statistic is the standard choice for reporting judge-human "
                "agreement in a paper."
            ),
            "interpretation": interp,
            "example": example,
        }

        metrics.update(_pearson_spearman_metrics(
            llm, human, alpha=alpha, rng=rng, ci2_fn=_ci2,
            pearson_label="Pearson r", spearman_label="Spearman r",
            pearson_what=(
                "Linear correlation coefficient between judge and human labels -- "
                "for two binary (0/1) variables this is the phi coefficient, "
                "algebraically equivalent to Cohen's κ's numerator rescaled by "
                "the marginal proportions."
            ),
            pearson_why=(
                "Reported alongside Cohen's κ/percent agreement because a "
                "PPI-corrected hypothesis test's variance reduction is governed "
                "by this correlation (or its rank-based counterpart below), not "
                "by κ -- see the label-efficiency guidance in the package docs "
                "for which one your test needs."
            ),
            spearman_what=(
                "Rank correlation between judge and human labels -- for two "
                "binary (0/1) variables this is numerically identical to "
                "Pearson r above (rank-transforming a two-valued variable is "
                "just an increasing affine rescaling of it, which Pearson r is "
                "invariant to)."
            ),
            spearman_why=(
                "Reported for consistency with the continuous/likert score "
                "types, and because rank-based hypothesis tests (e.g. "
                "Mann-Whitney) predict their PPI variance reduction from this "
                "correlation, not Pearson's."
            ),
        ))

    elif score_type == "likert":
        # Categories are positions on the scale's grid (step 1 for whole-number
        # scales, the detected step for e.g. half-point ones), not the ranks of
        # the values that happen to be observed: a level nobody used must still
        # count toward the distance between its neighbours.
        step = float(grid_step) if grid_step else 1.0
        origin = float(min(np.min(llm), np.min(human)))

        def _pos(v):
            return np.rint((np.asarray(v, dtype=float) - origin) / step).astype(int)

        k = int(max(_pos(llm).max(), _pos(human).max())) + 1
        ii = np.arange(k, dtype=float)
        wm = 1.0 - (ii[:, None] - ii[None, :])**2 / max((k - 1)**2, 1)

        def wk(a, b):
            ai, bi = _pos(a), _pos(b)
            p_o = float(np.mean(1.0 - (ai - bi)**2 / max((k - 1)**2, 1)))
            p_a = np.bincount(ai, minlength=k) / len(ai)
            p_b = np.bincount(bi, minlength=k) / len(bi)
            p_e = float((p_a[:, None] * p_b[None, :] * wm).sum())
            return (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0

        if k >= 2:
            est, lo, hi = _ci2(wk, llm, human, alpha=alpha, rng=rng)
            band, interp, example = _interpret_kappa(est, lo, hi, len(llm), "Quadratic-weighted Cohen's κ")
            metrics["weighted_kappa"] = {
                "estimate": est, "ci_low": lo, "ci_high": hi,
                "label": "Quadratic-weighted Cohen's κ",
                "band": band,
                "what": (
                    "Cohen's κ with quadratic weights, so disagreements are penalized "
                    "in proportion to the square of their distance on the ordinal "
                    "scale (Cohen, 1968). The other common convention, linear "
                    "weighting, penalizes distance directly rather than its square."
                ),
                "why": (
                    "Your judge produces ordered categorical (Likert) labels, so an "
                    "ordinal-aware kappa is used instead of the unweighted version, "
                    "which would penalize a near-miss (e.g. judge=4 vs human=5) as "
                    "harshly as a large disagreement. Quadratic weighting is the more "
                    "common convention for Likert-scale IRR and is used here."
                ),
                "interpretation": interp,
                "example": example,
            }
        metrics.update(_pearson_spearman_metrics(
            llm, human, alpha=alpha, rng=rng, ci2_fn=_ci2,
            pearson_label="Pearson r", spearman_label="Spearman r",
            pearson_what=(
                "Linear correlation coefficient between judge and human scores, "
                "treating the Likert categories as equally-spaced numeric values."
            ),
            pearson_why=(
                "Reported alongside weighted κ/Spearman r because a PPI-corrected "
                "parametric or mean-based test (e.g. a $t$-test on Likert scores "
                "treated as numeric) draws its variance reduction from this "
                "correlation, not from weighted κ or Spearman's rank-based one -- "
                "see the label-efficiency guidance in the package docs for which "
                "one your test needs."
            ),
            spearman_what=(
                "Rank correlation between judge and human scores. Checks whether "
                "higher judge scores correspond to higher human scores, without "
                "assuming the categories are equally spaced."
            ),
            spearman_why=(
                "Reported alongside weighted κ to show whether the judge preserves "
                "relative ordering, which matters if judge scores are mainly used "
                "to rank or compare outputs."
            ),
        ))

        if k >= 2:
            icc_est, icc_lo, icc_hi = _ci2(_icc_21, llm, human, alpha=alpha, rng=rng)
            band, interp, example = _interpret_icc(icc_est, icc_lo, icc_hi, len(llm), "ICC(2,1)")
            metrics["icc_21"] = {
                "estimate": icc_est, "ci_low": icc_lo, "ci_high": icc_hi,
                "label": "ICC(2,1)",
                "band": band,
                "what": (
                    "Two-way random-effects intraclass correlation for absolute "
                    "agreement (Shrout & Fleiss, 1979): unlike weighted κ's "
                    "category-index distance or Spearman r's rank comparison, it "
                    "is sensitive to a systematic offset between the judge and "
                    "human scale, not just whether they move together."
                ),
                "why": (
                    "Computed alongside weighted κ to check for absolute-scale "
                    "bias: a judge that ranks items correctly but is shifted or "
                    "compressed relative to human scores can still get a high "
                    "weighted κ / Spearman r while scoring poorly here."
                ),
                "interpretation": interp,
                "example": example,
            }

            gap_est, gap_lo, gap_hi = _cigap(wk, _icc_21, llm, human, alpha=alpha, rng=rng)
            metrics["_bias_check"] = _build_bias_check(
                "Quadratic-weighted Cohen's κ", metrics["weighted_kappa"]["estimate"],
                icc_est, gap_est, gap_lo, gap_hi,
            )

    else:  # continuous
        def pe(a, b):
            return _quiet_corr(pearsonr, a, b)

        metrics.update(_pearson_spearman_metrics(
            llm, human, alpha=alpha, rng=rng, ci2_fn=_ci2,
            pearson_label="Pearson r", spearman_label="Spearman r",
            pearson_what="Linear correlation coefficient between judge and human scores.",
            pearson_why=(
                "Your judge produces continuous/numeric scores, so a correlation "
                "coefficient is the standard way to summarize agreement."
            ),
            spearman_what="Rank correlation between judge and human scores.",
            spearman_why=(
                "Reported alongside Pearson r to check whether agreement holds even "
                "if the judge-human relationship is monotonic but non-linear (e.g. "
                "the judge saturates at high scores)."
            ),
        ))

        icc_est, icc_lo, icc_hi = _ci2(_icc_21, llm, human, alpha=alpha, rng=rng)
        band, interp, example = _interpret_icc(icc_est, icc_lo, icc_hi, len(llm), "ICC(2,1)")
        metrics["icc_21"] = {
            "estimate": icc_est, "ci_low": icc_lo, "ci_high": icc_hi,
            "label": "ICC(2,1)",
            "band": band,
            "what": (
                "Two-way random-effects intraclass correlation for absolute "
                "agreement (Shrout & Fleiss, 1979): unlike Pearson/Spearman r, "
                "which are invariant to any linear rescaling of one variable, "
                "this is sensitive to a systematic offset or scale mismatch "
                "between the judge and human scale."
            ),
            "why": (
                "Computed alongside Pearson r to check for absolute-scale bias: "
                "a judge that is consistently shifted or compressed relative to "
                "human scores can still score a perfect Pearson r while "
                "disagreeing badly here."
            ),
            "interpretation": interp,
            "example": example,
        }

        gap_est, gap_lo, gap_hi = _cigap(pe, _icc_21, llm, human, alpha=alpha, rng=rng)
        metrics["_bias_check"] = _build_bias_check(
            "Pearson r", metrics["pearson_r"]["estimate"],
            icc_est, gap_est, gap_lo, gap_hi,
        )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Representativeness checks
# ─────────────────────────────────────────────────────────────────────────────

# Why representativeness is checked at all — shared across the score-distribution,
# slice-column, and label-contiguity checks, since all three exist to answer the
# same question. Named explicitly (not just "representative") because the
# developer needs the actual causal mechanism to avoid it next time: the natural
# QA instinct is to hand-label the items you're *unsure about* (borderline
# scores, ones the judge seemed shaky on) -- which is exactly the kind of
# selection that breaks this assumption.
_REPRESENTATIVENESS_WHY = (
    "The calibration model and alignment metrics above are fit only on the "
    "labeled subset, and assume it's a random sample of the full item pool "
    "(\"missing completely at random\", MCAR, in the statistics literature) -- "
    "not, for example, the items you were most unsure about, or the "
    "lowest-scoring ones. If that assumption doesn't hold, statistical "
    "inference may not generalize to unlabeled items."
)

# Significance threshold for every representativeness "passed" verdict
# (score distribution, slice columns, label contiguity). Deliberately lower
# than the conventional 0.05: these are diagnostic tripwires meant to catch
# real MNAR violations, not confirmatory hypothesis tests -- and a
# well-calibrated test fires on true-null (genuinely random) data at
# whatever rate this is set to, so 0.05 means real random samples get
# flagged 1-in-20 times. 0.02 trades a bit of detection power for fewer
# false alarms crying wolf on real random data. Alignment-metric CIs
# (Pearson r, kappa, etc.) use their own, separate alpha= (default 0.05)
# and are unaffected by this constant.
_REP_ALPHA = 0.02


# Which alignment_metrics key is "the" headline metric per score type --
# the same one _build_bias_check compares against ICC(2,1) (see the
# score-type branches inside _compute_alignment_metrics above). Reused here
# so the per-condition breakdown reports the same metric a reader already
# saw pooled, rather than introducing a second unfamiliar number.
_PRIMARY_ALIGNMENT_KEY = {
    "binary": "cohens_kappa",
    "likert": "weighted_kappa",
    "continuous": "pearson_r",
}

# Below this many labeled items in a single condition, the primary metric
# is little more than noise (e.g. a single disagreement can swing Cohen's
# kappa by 0.3+) -- flagged as "too few" rather than shown with a
# misleadingly precise-looking number.
_PER_CONDITION_MIN_N = 5

# A same-direction-for-everyone judge is the ONLY failure mode a pooled
# metric can catch; a spread this large across conditions in the metric's
# own [-1, 1]-ish scale is large enough to be worth a reader's attention
# even without a formal test (no CI is computed per condition -- see
# _compute_per_condition_alignment's docstring for why).
_PER_CONDITION_SPREAD_FLAG = 0.15


def _compute_per_condition_alignment(
    df: pd.DataFrame,
    labeled_mask: "pd.Series[bool]",
    model_col: str,
    llm_metric: str,
    human_col: str,
    score_type: str,
    *,
    alpha: float,
    grid_step: Optional[float] = None,
    totals: Optional[dict] = None,
) -> Optional[dict]:
    """Per-condition/model breakdown of the headline alignment metric.

    A single *pooled* IRR number can look perfectly fine while hiding a
    judge that is biased *differently* per condition -- generous to one
    model, stingy to another -- which is exactly the failure mode PPI
    correction exists to catch, and exactly what no pooled statistic
    (including this function's own bias_check) can see. Surfacing the
    per-condition numbers here means a user sees that risk at
    judge_alignment() time, rather than discovering it only after a
    PPI-corrected compare() call disagrees with the naive one.

    Point estimates only, no bootstrap CI: each condition's labeled subset
    is a fraction of an already-small alignment set split across k
    conditions, so a per-condition CI would mostly be noise, and computing
    a full 2000-resample CI per condition would multiply
    judge_alignment()'s cost by k for little benefit. Returns None when
    there's no model column in the data, or fewer than 2 conditions have
    any labeled items at all.
    """
    if model_col not in df.columns:
        return None
    primary_key = _PRIMARY_ALIGNMENT_KEY.get(score_type)
    if primary_key is None:
        return None

    labeled_df = df.loc[labeled_mask]
    conditions = labeled_df[model_col].unique().tolist()
    if len(conditions) < 2 or len(conditions) > 20:
        # >20: model_col almost certainly isn't a small factor of interest
        # here (same cardinality cap the categorical slice-column check
        # uses elsewhere in this file) -- a per-condition table that long
        # would bury the signal it's meant to surface, not highlight it.
        return None

    rng = np.random.default_rng(42)
    out: dict = {}
    for cond in conditions:
        cond_mask = (labeled_df[model_col] == cond).to_numpy()
        n = int(cond_mask.sum())
        if n < _PER_CONDITION_MIN_N:
            out[cond] = {"n": n, "too_few": True}
            continue
        cond_llm = labeled_df.loc[cond_mask, llm_metric].to_numpy(dtype=float)
        cond_human = labeled_df.loc[cond_mask, human_col].to_numpy(dtype=float)
        metrics = _compute_alignment_metrics(
            cond_llm, cond_human, score_type, alpha=alpha, rng=rng, ci=False,
            grid_step=grid_step,
        )
        entry = metrics.get(primary_key)
        if entry is None:
            continue
        # rho^2 for this condition's MEAN: a mean's influence function is the score
        # itself, so it is the squared Pearson r of the labeled pairs. N_eff projects it
        # onto the condition's full size.
        r = (float(np.corrcoef(cond_llm, cond_human)[0, 1])
             if np.std(cond_llm) > 0 and np.std(cond_human) > 0 else float("nan"))
        n_all = int((totals or {}).get(cond) or (df[model_col] == cond).sum())
        mult, n_eff = _n_eff(r, n, n_all)
        out[cond] = {
            "n": n, "too_few": False,
            "label": entry["label"], "estimate": entry["estimate"],
            "r": r, "rho2": r * r if np.isfinite(r) else float("nan"),
            "n_total": n_all, "multiplier": mult, "n_eff": n_eff,
        }

    if not out:
        return None
    estimates = [v["estimate"] for v in out.values() if not v.get("too_few")]
    spread = (max(estimates) - min(estimates)) if len(estimates) >= 2 else 0.0
    return {"conditions": out, "spread": spread, "column": model_col}


def _rep_check_display_name(key: str) -> str:
    """Human-readable label for a representativeness-check dict key, for
    the short/simple summary (:meth:`AlignmentResult._summary_simple`)."""
    if key == "score_distribution":
        return "score distribution"
    if key == "label_contiguity":
        return "label position"
    if key.startswith("slice_"):
        return key[len("slice_"):]
    return key


def _interpret_representativeness(passed: bool, subject: str) -> str:
    if passed:
        return (
            f"no evidence (p ≥ {_REP_ALPHA:g}) that {subject} differs between the "
            "labeled subset and the full pool. Alignment estimates should "
            "generalize reasonably well"
        )
    return (
        f"{subject} differs between the labeled subset and the full pool "
        f"(p < {_REP_ALPHA:g}). Treat alignment estimates as potentially biased "
        "for unlabeled items; consider expanding or re-sampling the alignment set"
    )


def _check_score_distribution(
    all_scores: np.ndarray,
    labeled_scores: np.ndarray,
    score_type: str,
    unlabeled_scores: Optional[np.ndarray] = None,
) -> dict:
    """``unlabeled_scores``, when available, is used as the KS-test comparison
    target instead of ``all_scores``. ``all_scores`` includes the labeled
    subset by construction, so comparing against it (rather than the
    unlabeled complement) dilutes any real divergence -- the more of the
    pool is labeled, the more the sample resembles the thing it's being
    compared to. The binary branch is unaffected: it already derives
    unlabeled counts by subtraction, which is exact regardless.
    """
    if score_type == "binary":
        what = (
            "Chi-square test comparing the labeled subset's 0/1 score distribution "
            "to the unlabeled pool's."
        )
        labeled_0 = int((labeled_scores == 0).sum())
        labeled_1 = int((labeled_scores == 1).sum())
        all_0 = int((all_scores == 0).sum())
        all_1 = int((all_scores == 1).sum())
        unlabeled_0 = all_0 - labeled_0
        unlabeled_1 = all_1 - labeled_1
        if unlabeled_0 + unlabeled_1 == 0:
            return {
                "passed": True, "message": "all items are labeled", "p_value": None,
                "what": what,
                "why": _REPRESENTATIVENESS_WHY,
                "interpretation": (
                    "not applicable: every item already has a human label, so "
                    "there is no unlabeled pool to generalize to"
                ),
            }
        contingency = [[labeled_0, labeled_1], [unlabeled_0, unlabeled_1]]
        try:
            _, p, _, _ = chi2_contingency(contingency)
            p = float(p)
        except ValueError:
            p = 1.0
        passed = p >= _REP_ALPHA
        msg = f"χ² p={p:.3f}"
        if not passed:
            msg += ": labeled 0/1 distribution differs from unlabeled pool"
    else:
        compare_target = unlabeled_scores if unlabeled_scores is not None and len(unlabeled_scores) > 0 else all_scores
        what = (
            "Kolmogorov–Smirnov test comparing the labeled subset's score "
            "distribution to "
            + ("the unlabeled complement's." if unlabeled_scores is not None and len(unlabeled_scores) > 0
               else "the full item pool's (unlabeled-only comparison unavailable in this call form).")
        )
        if len(np.unique(labeled_scores)) < 2:
            return {
                "passed": True, "message": "insufficient labeled variation to test", "p_value": None,
                "what": what,
                "why": _REPRESENTATIVENESS_WHY,
                "interpretation": (
                    "not applicable: the labeled scores don't vary enough to run "
                    "this test"
                ),
            }
        _, p = _quiet_ks_2samp(labeled_scores, compare_target)
        p = float(p)
        passed = p >= _REP_ALPHA
        msg = f"KS p={p:.3f}"
        if not passed:
            msg += ": labeled subset appears non-representative of full score range"
    return {
        "passed": passed, "message": msg, "p_value": p,
        "what": what,
        "why": _REPRESENTATIVENESS_WHY,
        "interpretation": _interpret_representativeness(passed, "the score distribution"),
    }


def _check_slice_column(
    df: pd.DataFrame,
    labeled_mask: pd.Series,
    col: str,
) -> dict:
    what = (
        f"Chi-square test comparing the distribution of {col!r} between labeled "
        "and unlabeled items."
    )
    why = (
        "Checks whether the alignment set is representative across this "
        "categorical variable. Important if judge accuracy might vary by "
        "subgroup (e.g. domain, difficulty, model)."
    )
    labeled = df.loc[labeled_mask, col].dropna()
    unlabeled = df.loc[~labeled_mask, col].dropna()
    if len(unlabeled) == 0:
        return {
            "passed": True, "message": "no unlabeled items", "p_value": None,
            "what": what, "why": why,
            "interpretation": (
                "not applicable: there are no unlabeled items to compare against"
            ),
        }
    cats = sorted(df[col].dropna().unique())
    lab_counts = [(labeled == c).sum() for c in cats]
    unlab_counts = [(unlabeled == c).sum() for c in cats]
    contingency = [lab_counts, unlab_counts]
    try:
        _, p, _, _ = chi2_contingency(contingency)
        p = float(p)
    except ValueError:
        p = 1.0
    passed = p >= _REP_ALPHA
    msg = f"χ² p={p:.3f}"
    if not passed:
        msg += ": labeled subset is over/under-represented in some categories"
    return {
        "passed": passed, "message": msg, "p_value": p,
        "what": what, "why": why,
        "interpretation": _interpret_representativeness(passed, f"{col!r}"),
    }


def _check_slice_column_numeric(
    df: pd.DataFrame,
    labeled_mask: pd.Series,
    col: str,
) -> dict:
    """KS-test analogue of :func:`_check_slice_column` for numeric covariates
    (e.g. difficulty, length, latency) -- these are never string dtype, so
    the chi-square categorical check above silently skips them entirely.
    """
    what = (
        f"Kolmogorov–Smirnov test comparing the distribution of numeric "
        f"column {col!r} between labeled and unlabeled items."
    )
    why = (
        "Checks whether the alignment set is representative across this "
        "numeric covariate. Important if judge accuracy might vary with it "
        "(e.g. difficulty, length, latency). Categorical (string) columns are "
        "checked with a chi-square test instead; this covers the numeric "
        "columns that check silently skips."
    )
    labeled = df.loc[labeled_mask, col].dropna().to_numpy(dtype=float)
    unlabeled = df.loc[~labeled_mask, col].dropna().to_numpy(dtype=float)
    if len(unlabeled) == 0:
        return {
            "passed": True, "message": "no unlabeled items", "p_value": None,
            "what": what, "why": why,
            "interpretation": (
                "not applicable: there are no unlabeled items to compare against"
            ),
        }
    if len(np.unique(labeled)) < 2:
        return {
            "passed": True, "message": "insufficient labeled variation to test", "p_value": None,
            "what": what, "why": why,
            "interpretation": (
                "not applicable: the labeled values don't vary enough to run "
                "this test"
            ),
        }
    _, p = _quiet_ks_2samp(labeled, unlabeled)
    p = float(p)
    passed = p >= _REP_ALPHA
    msg = f"KS p={p:.3f}"
    if not passed:
        msg += ": labeled subset differs from unlabeled pool on this covariate"
    return {
        "passed": passed, "message": msg, "p_value": p,
        "what": what, "why": why,
        "interpretation": _interpret_representativeness(passed, f"{col!r}"),
    }


def _apply_family_correction(results: dict[str, dict], method: str = "holm") -> dict[str, dict]:
    """Apply a family-wise multiple-testing correction across a set of
    representativeness checks (one per covariate), keyed by name.

    Without this, testing many slice columns inflates the chance of at least
    one spurious "not representative" flag well above the nominal alpha --
    e.g. ~63% with 20 unrelated covariates under a true null at alpha=0.05,
    empirically (worse at looser alpha, better at the stricter _REP_ALPHA
    this module actually uses). Entries with no ``p_value`` (not-applicable
    checks) pass through untouched and aren't counted in the correction
    family. Only annotates the message for entries whose *raw* p was below
    ``_REP_ALPHA`` (Holm-adjusted p is never smaller than the raw p, so a
    raw-passing entry always still passes -- nothing to say there).
    """
    testable = [k for k, v in results.items() if v.get("p_value") is not None]
    if len(testable) <= 1:
        return results
    from evalstats.core.stats_utils import correct_pvalues
    raw_p = np.array([results[k]["p_value"] for k in testable])
    adj_p = correct_pvalues(raw_p, method=method)
    out = dict(results)
    for k, p_adj in zip(testable, adj_p):
        p_adj = float(p_adj)
        res = dict(out[k])
        raw_p_k = res["p_value"]
        passed = bool(p_adj >= _REP_ALPHA)
        res["p_value_adjusted"] = p_adj
        res["passed"] = passed
        if raw_p_k < _REP_ALPHA:
            if passed:
                res["message"] += (
                    f". No longer significant after Holm correction across "
                    f"{len(testable)} covariates (adjusted p={p_adj:.3f})"
                )
            else:
                res["message"] += (
                    f". Still significant after Holm correction across "
                    f"{len(testable)} covariates (adjusted p={p_adj:.3f})"
                )
            res["interpretation"] = _interpret_representativeness(passed, "this covariate")
        out[k] = res
    return out


def _safe_comb(n: int, k: int) -> int:
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)


def _count_runs(mask: np.ndarray) -> int:
    """Number of maximal contiguous same-value stretches in a boolean sequence."""
    if len(mask) == 0:
        return 0
    return int(1 + np.sum(mask[1:] != mask[:-1]))


def _runs_test_pvalue(n1: int, n2: int, r_obs: int) -> float:
    """Two-sided Wald–Wolfowitz runs-test p-value for ``r_obs`` runs among
    ``n1`` items of one kind and ``n2`` of another, arranged uniformly at
    random. Flags both too few runs (clustering, e.g. a contiguous block or
    a couple of blocks) and too many runs (suspicious regularity, e.g. every
    Kth position).

    Uses the exact distribution (summed directly, cheap for realistic
    dataset sizes) below ``n1 + n2 <= 4000``; falls back to the standard
    normal approximation with continuity correction above that, since the
    exact pmf's binomial-coefficient terms grow expensive to sum one-by-one
    at that scale while the normal approximation is already excellent there.
    """
    n = n1 + n2
    if n1 == 0 or n2 == 0:
        return 1.0
    if n <= 4000:
        total = math.comb(n, n1)

        def pmf(r: int) -> float:
            if r % 2 == 0:
                k = r // 2
                return 2 * _safe_comb(n1 - 1, k - 1) * _safe_comb(n2 - 1, k - 1) / total
            k = (r - 1) // 2
            return (
                _safe_comb(n1 - 1, k) * _safe_comb(n2 - 1, k - 1)
                + _safe_comb(n1 - 1, k - 1) * _safe_comb(n2 - 1, k)
            ) / total

        p_le = sum(pmf(r) for r in range(2, r_obs + 1))
        p_ge = sum(pmf(r) for r in range(r_obs, n + 1))
        return float(min(1.0, 2 * min(p_le, p_ge)))

    mu = 1.0 + 2.0 * n1 * n2 / n
    var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)) / (n**2 * (n - 1))
    if var <= 0:
        return 1.0
    sd = math.sqrt(var)
    cc = 0.5 if r_obs < mu else -0.5
    z = (r_obs - mu + cc) / sd
    return float(2 * norm.sf(abs(z)))


def _check_label_contiguity(
    n_total: int, labeled_mask: np.ndarray, item_ids: Optional[np.ndarray] = None,
) -> dict:
    """Runs test on where the labeled items sit in the dataset.

    Unlike the distribution-based checks above, this doesn't look at scores
    at all -- it only looks at *where in the dataset* the labeled items sit.
    A single contiguous block (e.g. "the first N" or "the last N" items) is
    the most common way evalstats has seen this assumption broken in
    practice, but it's just the most extreme case of a broader failure mode:
    labeled items clustered into a small number of blocks (e.g. first-N-
    plus-last-N), or laid out with suspicious regularity (e.g. every Kth
    row). The Wald-Wolfowitz runs test catches all of these by comparing the
    observed number of contiguous same-label runs against what genuine
    uniform-random sampling would produce -- even in the case where such a
    selection happens to produce a labeled subset whose score distribution
    passes the other checks by chance.
    """
    what = (
        "Runs test on the labeled/unlabeled sequence: checks whether the "
        "labeled items form too few contiguous blocks (clustering, e.g. "
        "rows 0-14, or first-15-plus-last-15) or too many (suspicious "
        "regularity, e.g. every 10th row) to be a uniformly random subset."
    )
    why = (
        "The most common way evalstats has seen this assumption broken in "
        "practice isn't a subtle score-distribution skew -- it's literally "
        "labeling \"the first N\" or \"the last N\" items, often just because "
        "that's what a spreadsheet or a `.head()` call hands you first. A "
        "runs test catches that pattern and its variants (e.g. a couple of "
        "blocks, or artificially regular spacing) in one check, rather than "
        "only the single-contiguous-block special case."
    )
    mask = np.asarray(labeled_mask).astype(bool)
    if item_ids is not None and len(item_ids) == len(mask):
        # One entry per distinct item, in first-appearance order. In a paired
        # design every condition's row for a labeled item is labeled, so the
        # row sequence has runs of length k by construction; the question is
        # whether the labeled ITEMS cluster.
        codes, uniques = pd.factorize(pd.Series(np.asarray(item_ids)), sort=False)
        if len(uniques) < len(mask):
            item_mask = np.zeros(len(uniques), dtype=bool)
            np.logical_or.at(item_mask, codes, mask)
            mask = item_mask
            n_total = len(uniques)
    n_labeled = int(mask.sum())
    n_unlabeled = n_total - n_labeled
    if n_labeled < 2 or n_unlabeled < 2:
        return {
            "passed": True, "message": "not applicable", "p_value": None,
            "what": what, "why": why,
            "interpretation": (
                "not applicable -- fewer than 2 labeled or 2 unlabeled items, "
                "so there's no position pattern to check"
            ),
        }
    r_obs = _count_runs(mask)
    p = _runs_test_pvalue(n_labeled, n_unlabeled, r_obs)
    mu = 1.0 + 2.0 * n_labeled * n_unlabeled / n_total
    passed = p >= _REP_ALPHA

    positions = np.flatnonzero(mask)
    span = int(positions.max() - positions.min() + 1)
    is_single_block = span == n_labeled and r_obs <= 2

    if not passed:
        if is_single_block:
            start, end = int(positions.min()), int(positions.max())
            msg = (
                f"the {n_labeled} labeled items are exactly rows {start}-{end} "
                f"of {n_total} -- a single contiguous block ({r_obs} run(s) vs. "
                f"~{mu:.0f} expected under random selection, p={p:.2e})"
            )
        elif r_obs < mu:
            msg = (
                f"labeled item positions form only {r_obs} contiguous run(s), "
                f"vs. ~{mu:.0f} expected under random selection (p={p:.2e}) -- "
                "looks like a small number of blocks (e.g. first-N-plus-"
                "last-N) rather than a scattered random sample"
            )
        else:
            msg = (
                f"labeled item positions form {r_obs} runs, far more than "
                f"the ~{mu:.0f} expected under random selection (p={p:.2e}) "
                "-- looks like an artificially regular pattern (e.g. every "
                "Kth row) rather than genuine random sampling"
            )
    else:
        msg = (
            f"labeled item positions look scattered ({r_obs} runs, "
            f"~{mu:.0f} expected under random selection, p={p:.2f})"
        )

    if passed:
        interpretation = (
            "the labeled items' positions don't form a suspicious clustered "
            "or artificially regular pattern -- doesn't confirm random "
            "selection, but rules out the most common non-random patterns"
        )
    else:
        interpretation = (
            "the labeled items' positions are essentially impossible from "
            "real random sampling -- treat alignment estimates as unreliable "
            "for unlabeled items unless this was deliberate (e.g. the "
            "dataset itself was already shuffled before labeling); consider "
            "re-sampling the alignment set uniformly at random instead"
        )
    return {
        "passed": passed, "message": msg, "p_value": p,
        "what": what, "why": why,
        "interpretation": interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# judge_alignment
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SELECTIONS = ("random", "stratified", "manual", "unknown")


def _validate_and_warn_selection(selection: str, warn_stacklevel: int) -> None:
    """Validate ``selection=`` and warn about its MCAR implications.

    Shared by :func:`_judge_alignment_core` and :func:`_judge_alignment_pairwise`.
    """
    if selection not in _VALID_SELECTIONS:
        raise ValueError(
            f"selection={selection!r} -- must be one of {_VALID_SELECTIONS}."
        )
    if selection == "unknown":
        warnings.warn(
            "judge_alignment() was not told how the labeled subset was "
            "selected (selection=). Every correction it and "
            "compare(alignment=...) apply assumes the labeled items are a "
            "random sample of the full item pool -- pass selection='random' "
            "to confirm that's the case, or selection='manual'/'stratified' "
            "if not, so this is a deliberate acknowledgment rather than an "
            "unexamined default.",
            UserWarning, stacklevel=warn_stacklevel,
        )
    elif selection == "manual":
        warnings.warn(
            "selection='manual': the labeled subset was NOT randomly "
            "sampled. PPI/alignment correction assumes random sampling "
            "(MCAR) to be valid -- with a manually-chosen subset, the "
            "corrected estimates and CIs compare()/judge_alignment() report "
            "may be miscalibrated, not just imprecise. Treat them as "
            "informal unless the alignment set is re-sampled at random.",
            UserWarning, stacklevel=warn_stacklevel,
        )
    elif selection == "stratified":
        warnings.warn(
            "selection='stratified': evalstats' current correction doesn't "
            "account for stratification weights, so this is only valid if "
            "each stratum was itself sampled uniformly at random and the "
            "strata are otherwise ignorable for the metric being judged. "
            "If items were hand-picked within strata, treat corrected "
            "estimates as potentially biased, same as selection='manual'.",
            UserWarning, stacklevel=warn_stacklevel,
        )


def _judge_alignment_core(
    llm_aligned: np.ndarray,
    human_aligned: np.ndarray,
    score_type: str,
    *,
    llm_metric: str,
    human_groundtruth: str,
    alpha: float,
    n_total: int,
    ci: bool = True,
    grid_step: Optional[float] = None,
    all_llm: Optional[np.ndarray] = None,
    slice_df: Optional[pd.DataFrame] = None,
    slice_labeled_mask: Optional[pd.Series] = None,
    slice_exclude_cols: frozenset = frozenset(),
    labeled_mask: Optional[np.ndarray] = None,
    item_ids: Optional[np.ndarray] = None,
    selection: str = "unknown",
    test: Optional[str] = None,
    per_condition_metrics: Optional[dict] = None,
    warn_stacklevel: int = 3,
) -> AlignmentResult:
    """Shared core behind both :func:`judge_alignment` call forms: fits the
    calibration model, computes alignment metrics, and (only when the
    relevant context is available) runs representativeness diagnostics.

    ``all_llm`` enables the score-distribution check; ``slice_df`` +
    ``slice_labeled_mask`` enable the categorical slice-column checks; a
    non-``None`` ``labeled_mask`` (positions of labeled items within the
    ``n_total``-length item pool, in dataset row order) enables the
    label-contiguity check. All three require the full item pool / other
    columns, so they're skipped entirely -- not silently approximated --
    when this is called from raw paired arrays with no further context,
    see :func:`judge_alignment`.
    """
    _validate_and_warn_selection(selection, warn_stacklevel)
    n_labeled = int(len(llm_aligned))

    calibration = _fit_calibration(llm_aligned, human_aligned, score_type)

    rng = np.random.default_rng(42)
    alignment_metrics = _compute_alignment_metrics(
        llm_aligned, human_aligned, score_type, alpha=alpha, rng=rng, ci=ci,
        grid_step=grid_step,
    )
    bias_check = alignment_metrics.pop("_bias_check", None)

    rep: dict = {}
    if all_llm is not None:
        unlabeled_llm = None
        if labeled_mask is not None and len(labeled_mask) == len(all_llm):
            unlabeled_llm = all_llm[~labeled_mask.astype(bool)]
        dist_result = _check_score_distribution(
            all_llm, llm_aligned, score_type, unlabeled_scores=unlabeled_llm
        )
        rep["score_distribution"] = dist_result
        if not dist_result["passed"]:
            warnings.warn(
                f"Representativeness warning: the {n_labeled} labeled items appear to have "
                f"a different {llm_metric} distribution than the full item pool "
                f"({dist_result['message']}). "
                "Alignment uncertainty estimates may not generalise to all items. "
                "Consider sampling human labels more broadly across the score range.",
                UserWarning,
                stacklevel=warn_stacklevel,
            )

    if slice_df is not None and slice_labeled_mask is not None:
        cat_cols = [
            c for c in slice_df.columns
            if c not in slice_exclude_cols
            and pd.api.types.is_string_dtype(slice_df[c])
            and 1 < slice_df[c].nunique() <= 20
        ]
        num_cols = [
            c for c in slice_df.columns
            if c not in slice_exclude_cols
            and pd.api.types.is_numeric_dtype(slice_df[c])
            and not pd.api.types.is_bool_dtype(slice_df[c])
            and slice_df[c].nunique() > 1
        ]
        slice_results: dict = {}
        for col in cat_cols:
            slice_results[col] = _check_slice_column(slice_df, slice_labeled_mask, col)
        for col in num_cols:
            slice_results[col] = _check_slice_column_numeric(slice_df, slice_labeled_mask, col)

        # Correct across the whole covariate family jointly (not per-column) --
        # testing many slice columns otherwise inflates the false-alarm rate
        # well above the nominal 5% (empirically ~63% at 20 columns).
        slice_results = _apply_family_correction(slice_results, method="holm")

        for col, col_result in slice_results.items():
            rep[f"slice_{col}"] = col_result
            if not col_result["passed"]:
                warnings.warn(
                    f"Representativeness warning for column '{col}': the labeled subset "
                    f"appears unevenly distributed across categories "
                    f"({col_result['message']}). "
                    "Consider stratified sampling of human labels.",
                    UserWarning,
                    stacklevel=warn_stacklevel,
                )

    # Position says something only when the rows passed in ARE the pool. When n_total
    # declares a larger dataset, they are just the labeled validation set.
    if labeled_mask is not None and n_total == len(labeled_mask):
        contiguity_result = _check_label_contiguity(n_total, labeled_mask, item_ids=item_ids)
        rep["label_contiguity"] = contiguity_result
        if not contiguity_result["passed"]:
            warnings.warn(
                f"Representativeness warning: {contiguity_result['message']}. "
                "This looks like 'the first N' or 'the last N' items were "
                "labeled rather than a random sample. Consider re-sampling "
                "the alignment set uniformly at random.",
                UserWarning,
                stacklevel=warn_stacklevel,
            )

    for key in ("pearson_r", "spearman_r"):
        if key in alignment_metrics:
            mult, n_eff = _n_eff(alignment_metrics[key]["estimate"], n_labeled, n_total)
            alignment_metrics[key]["multiplier"] = mult
            alignment_metrics[key]["n_eff"] = n_eff

    test_metric = None
    if test is not None:
        if test != "mean_estimate":
            raise ValueError(
                f"test={test!r} needs a comparison (2+ conditions) -- pass evaldata "
                "with a condition column (from load_from) instead of plain "
                "arrays, or use test='mean_estimate' for a single-condition "
                "estimate (no comparison)."
            )
        test_metric = _add_rho2(dict(alignment_metrics["pearson_r"]))
        test_metric["label"] = "mean_estimate rho"

    return AlignmentResult(
        llm_metric=llm_metric,
        human_col=human_groundtruth,
        score_type=score_type,
        n_labeled=n_labeled,
        n_total=n_total,
        calibration=calibration,
        alignment_metrics=alignment_metrics,
        representativeness=rep,
        bias_check=bias_check,
        selection=selection,
        test=test,
        test_metric=test_metric,
        per_condition_metrics=per_condition_metrics,
    )


def _resolve_per_condition_col(evaldata, df, factors) -> Optional[str]:
    """Pick the column whose per-condition alignment breakdown to report.

    In order: an explicit ``factors=``, the column declared to ``load_from``,
    the "model" role column, then the sole factor column. Raises when that
    leaves two or more candidates, since there is then no right grouping to pick.
    """
    if factors is not None:
        if isinstance(factors, (list, tuple)):
            if len(factors) != 1:
                raise ValueError(
                    "judge_alignment(factors=...) takes a single column name; "
                    f"got {list(factors)!r}."
                )
            factors = factors[0]
        if factors not in df.columns:
            raise ValueError(
                f"factors='{factors}' is not a column in this data. "
                f"Available columns: {list(df.columns)}"
            )
        return factors

    declared = [c for c in (getattr(evaldata, "_declared_factors", None) or [])
                if c in df.columns]
    if len(declared) == 1:
        return declared[0]

    model_col = evaldata._col.get("model")
    if model_col is not None:
        return model_col

    factor_cols = [c for c in (declared or getattr(evaldata, "_factor_cols", None) or [])
                   if c in df.columns]
    if len(factor_cols) == 1:
        return factor_cols[0]
    if len(factor_cols) > 1:
        raise ValueError(
            "This data has more than one factor column "
            f"({', '.join(repr(c) for c in factor_cols)}) and no 'model' role, so "
            "there is no single grouping to report per-condition alignment over. "
            "Pass the one you are comparing, e.g. "
            f"judge_alignment(..., factors='{factor_cols[0]}')."
        )
    return None


_DESIGN_ALIASES = {"paired": "within", "unpaired": "between", "within": "within", "between": "between"}


def _normalize_design(design: Optional[str]) -> Optional[str]:
    if design is None:
        return None
    if design not in _DESIGN_ALIASES:
        raise ValueError(
            f"design={design!r} -- expected 'within' (or 'paired') or 'between' (or 'unpaired')."
        )
    return _DESIGN_ALIASES[design]


def _comparison_conditions(df, group_col, item_col, llm_metric, human_col, design):
    """Split the evaldata into ``{condition: (judge, human)}`` and settle the design.

    The design is read off the item ids when not given: the same items in every
    condition is within-subjects, items that never repeat across conditions is
    between-subjects, and anything in between is ambiguous and raises."""
    groups = list(dict.fromkeys(df[group_col].tolist()))
    inferred = design is None
    if inferred:
        if item_col is None or item_col not in df.columns:
            raise ValueError(
                "Can't tell whether this is a within- or between-subjects comparison "
                "without item ids. Pass design='within' (the same items in every "
                "condition) or design='between' (different items per condition)."
            )
        item_sets = {g: set(df.loc[df[group_col] == g, item_col].tolist()) for g in groups}
        total = sum(len(v) for v in item_sets.values())
        union = set().union(*item_sets.values())
        if all(v == item_sets[groups[0]] for v in item_sets.values()) and len(groups) > 1:
            design = "within"
        elif total == len(union):
            design = "between"
        else:
            raise ValueError(
                f"Some items, but not all, appear in more than one {group_col!r} "
                "condition, so the design can't be inferred. Pass design='within' or "
                "design='between'."
            )

    conditions = {}
    for g in groups:
        sub = df[df[group_col] == g]
        if design == "within":
            if item_col is None or item_col not in df.columns:
                raise ValueError("design='within' needs an item column to line the conditions up.")
            if sub[item_col].duplicated().any():
                raise ValueError(
                    f"Condition {g!r} has several rows for the same item, so its scores "
                    "can't be lined up across conditions. Aggregate to one row per "
                    "item and condition first."
                )
            sub = sub.sort_values(item_col)
        conditions[str(g)] = (sub[llm_metric].to_numpy(dtype=float),
                              sub[human_col].to_numpy(dtype=float))
    if design == "within":
        lens = {len(v[0]) for v in conditions.values()}
        if len(lens) > 1:
            raise ValueError("design='within' needs the same items in every condition.")
    return conditions, design, inferred


def _per_condition_n_eff(design: str, role: str, n_eff_total: float, k: int) -> float:
    """compare() quotes N_eff as labels per condition; judge_alignment's totals span the
    conditions a correlation pools, so divide the same way compare() does."""
    if not np.isfinite(n_eff_total):
        return float("nan")
    if role == "omnibus":
        return n_eff_total / k
    return n_eff_total / 2 if design == "between" else n_eff_total


def _comparison_report(
    conditions: dict, design: str, inferred: bool, score_type: str, test: Optional[str],
    *, alpha: float, n_total, ci: bool,
) -> dict:
    """rho^2 and N_eff for the tests compare() would run on these conditions."""
    names = list(conditions)
    k = len(names)
    if test is not None:
        if test not in _TEST_STRUCTURE or _TEST_STRUCTURE[test]["min_k"] < 2:
            known = sorted(t for t, v in _TEST_STRUCTURE.items() if v["min_k"] >= 2)
            raise ValueError(f"test={test!r} -- expected one of {known}.")
        implied = _TEST_STRUCTURE[test]["design"]
        if implied is not None and implied != design:
            raise ValueError(f"test={test!r} is a {implied}-subjects test, but this data is {design}-subjects.")
        plan = [(test, "omnibus" if _TEST_STRUCTURE[test]["max_k"] is None else "pairs")]
    else:
        plan = _comparison_tests(design, score_type, k)

    totals = _resolve_condition_totals(conditions, n_total, design)
    n_lab = {n: int(np.sum(~np.isnan(np.asarray(conditions[n][1], dtype=float)))) for n in names}
    n_all = {n: (totals[n] if totals else len(conditions[n][0])) for n in names}

    entries = []
    for t, role in plan:
        # The caller has already warned about how the labels were selected; saying so
        # again here would print the same warning once per test.
        res = _judge_alignment_pairwise(
            conditions, design=design, alpha=alpha, test=t, selection="random",
            n_total=n_total, ci=ci, include_raw=False,
        )
        if role == "omnibus":
            m = res.omnibus_metric
            metric = dict(m)
            metric["n_eff_per_condition"] = _per_condition_n_eff(design, role, m["n_eff"], k)
            entries.append({"test": t, "role": role, "label": _TEST_LABELS.get(t, t), "metric": metric})
        else:
            pairs = {}
            for key, m in res.test_pairwise_metrics.items():
                metric = dict(m)
                metric["n_eff_per_condition"] = _per_condition_n_eff(design, role, m["n_eff"], k)
                pairs[key] = metric
            entries.append({"test": t, "role": role, "label": _TEST_LABELS.get(t, t), "pairs": pairs})
    return {
        "design": design, "design_inferred": inferred, "conditions": names,
        "n_labeled": n_lab, "n_total": n_all,
        "projects_n_eff": any(n_all[n] > n_lab[n] for n in names),
        "tests": entries,
    }


def _condition_totals_from_df(df, group_col, by_condition: Optional[dict], n_total) -> dict:
    """Full size of each condition, keyed like the column's values: the rows present,
    unless ``n_total=`` says the data is only a validation subset of a larger dataset."""
    counts = df[group_col].value_counts()
    totals = {c: int(counts[c]) for c in counts.index}
    if by_condition:
        by_str = {str(k): int(v) for k, v in by_condition.items()}
        for c in totals:
            if str(c) in by_str:
                totals[c] = by_str[str(c)]
    elif n_total is not None and int(n_total) > len(df):
        totals = {c: int(round(int(n_total) * v / len(df))) for c, v in totals.items()}
    return totals


def _check_n_total(override: Optional[int], actual: int) -> int:
    """The full-dataset size to report: the caller's ``n_total=`` if given, else
    the number of items actually passed in."""
    if override is None:
        return int(actual)
    if int(override) < int(actual):
        raise ValueError(
            f"n_total={override}, but {actual} items were passed in; n_total is "
            "the size of the full dataset and can't be smaller."
        )
    return int(override)


def _judge_alignment_from_evaldata(
    evaldata,
    *,
    llm_metric: str,
    human_groundtruth: str,
    alpha: float,
    selection: str = "unknown",
    ci: bool = True,
    factors=None,
    score_type: Optional[str] = None,
    n_total=None,
    design: Optional[str] = None,
    test: Optional[str] = None,
) -> AlignmentResult:
    df = evaldata._df

    if llm_metric not in df.columns:
        raise ValueError(
            f"llm_metric column '{llm_metric}' not found in evaldata. "
            f"Available columns: {list(df.columns)}"
        )
    if human_groundtruth not in df.columns:
        raise ValueError(
            f"human_groundtruth column '{human_groundtruth}' not found in evaldata. "
            f"Available columns: {list(df.columns)}"
        )

    labeled_mask = df[human_groundtruth].notna()
    n_labeled = int(labeled_mask.sum())
    design_arg = _normalize_design(design)
    if isinstance(n_total, dict):
        n_total_by_condition = {str(k): int(v) for k, v in n_total.items()}
        n_total = sum(n_total_by_condition.values())
    else:
        n_total_by_condition = None
    n_total = _check_n_total(n_total, len(df))

    if n_labeled == 0:
        raise ValueError(
            f"No rows have human labels in '{human_groundtruth}'. "
            "Ensure it is NaN for unlabeled items and non-NaN for the alignment subset."
        )

    if n_labeled < 30:
        warnings.warn(
            f"Only {n_labeled} items have human labels. "
            "Alignment estimates will be imprecise with fewer than ~30 labeled items; "
            "consider expanding the alignment set for reliable uncertainty propagation.",
            UserWarning,
            stacklevel=3,
        )

    auto_type = evaldata._score_types.get(llm_metric)
    if auto_type is None:
        from evalstats.loader import _detect_score_type
        auto_type = _detect_score_type(df[llm_metric].dropna())

    llm_aligned = df.loc[labeled_mask, llm_metric].to_numpy(dtype=float)
    human_aligned = df.loc[labeled_mask, human_groundtruth].to_numpy(dtype=float)
    all_llm = df[llm_metric].to_numpy(dtype=float)
    declared_type = score_type is not None
    score_type, grid_step = _resolve_score_type(score_type, auto_type, llm_aligned, human_aligned)

    # Structural role columns (model/item/run) are row/group identifiers, not
    # domain covariates -- an "item" column is frequently just a sequential
    # index (or unique per row), which the numeric-covariate check would
    # otherwise happily test, redundantly rediscovering (in a noisier form)
    # exactly what the position-based label-contiguity check already covers.
    group_col = _resolve_per_condition_col(evaldata, df, factors)

    structural_cols = {
        c for c in (evaldata._col.get("model"), evaldata._col.get("item"),
                    evaldata._col.get("run"), group_col)
        if c is not None
    }

    per_condition_metrics = None
    if group_col is not None:
        per_condition_metrics = _compute_per_condition_alignment(
            df, labeled_mask, group_col, llm_metric, human_groundtruth, score_type,
            alpha=alpha, grid_step=grid_step,
            totals=_condition_totals_from_df(df, group_col, n_total_by_condition, n_total),
        )

    n_conditions = int(df[group_col].nunique()) if group_col is not None else 0
    compares = n_conditions >= 2 and test != "mean_estimate"

    result = _judge_alignment_core(
        llm_aligned, human_aligned, score_type,
        llm_metric=llm_metric, human_groundtruth=human_groundtruth,
        alpha=alpha, n_total=n_total, all_llm=all_llm, grid_step=grid_step,
        slice_df=df, slice_labeled_mask=labeled_mask,
        slice_exclude_cols=frozenset({llm_metric, human_groundtruth}) | structural_cols,
        labeled_mask=labeled_mask.to_numpy(),
        item_ids=(df[evaldata._col["item"]].to_numpy()
                  if evaldata._col.get("item") in df.columns else None),
        selection=selection, ci=ci,
        per_condition_metrics=per_condition_metrics,
        # Only a single condition has no comparison to describe; its efficiency is that
        # of a plain mean.
        test=None if compares else "mean_estimate",
        warn_stacklevel=4,
    )

    result.score_type_source = "declared" if declared_type else "inferred"
    result.grid_step = grid_step

    if compares:
        # rho^2 and N_eff for the tests compare() will run on these conditions. A design
        # that can't be worked out is reported in the summary rather than raised, unless
        # the caller named the design or test themselves.
        try:
            conditions, resolved, inferred = _comparison_conditions(
                df, group_col, evaldata._col.get("item"), llm_metric, human_groundtruth, design_arg)
            result.comparison = _comparison_report(
                conditions, resolved, inferred, score_type, test, alpha=alpha,
                n_total=n_total_by_condition if n_total_by_condition is not None
                else (n_total if n_total != len(df) else None),
                ci=ci)
        except ValueError as exc:
            if design_arg is not None or test is not None:
                raise
            result.comparison_error = str(exc)
    return result


def _judge_alignment_from_arrays(
    judge_scores: np.ndarray,
    human_scores: np.ndarray,
    *,
    all_judge_scores: Optional[np.ndarray],
    score_type: Optional[str],
    llm_metric: Optional[str],
    human_groundtruth: Optional[str],
    alpha: float,
    selection: str = "unknown",
    test: Optional[str] = None,
    ci: bool = True,
    n_total: Optional[int] = None,
) -> AlignmentResult:
    judge_full = np.asarray(judge_scores, dtype=float)
    human_full = np.asarray(human_scores, dtype=float)
    if judge_full.shape != human_full.shape:
        raise ValueError(
            "judge_scores and human_scores must be the same length -- one "
            "judge score + one (possibly NaN) human score per item; got "
            f"shapes {judge_full.shape} and {human_full.shape}."
        )
    if judge_full.ndim != 1:
        raise ValueError(
            f"judge_scores/human_scores must be 1-D; got shape {judge_full.shape}."
        )

    labeled_mask = ~np.isnan(human_full)
    n_labeled = int(labeled_mask.sum())
    if n_labeled == 0:
        raise ValueError(
            "No labeled items -- human_scores is all NaN. It should be "
            "non-NaN for the alignment subset and NaN elsewhere (or, if "
            "every item is labeled, contain no NaN at all)."
        )
    llm_aligned = judge_full[labeled_mask]
    human_aligned = human_full[labeled_mask]

    if n_labeled < 30:
        warnings.warn(
            f"Only {n_labeled} items have human labels. "
            "Alignment estimates will be imprecise with fewer than ~30 labeled items; "
            "consider expanding the alignment set for reliable uncertainty propagation.",
            UserWarning,
            stacklevel=3,
        )

    # judge_scores doubles as "every item's judge score" for the
    # representativeness check for free -- but only when there's actual
    # evidence it's the full pool (some items weren't labeled). When
    # n_labeled == judge_full.size (no NaN at all in human_scores), there's
    # no way to tell "this is the full pool, 100% labeled" apart from "this
    # is just the labeled subset the caller already extracted" -- stay
    # conservative and skip the check rather than silently comparing a set
    # against itself (which would trivially "pass" and could read as false
    # confidence). An explicit all_judge_scores= always wins either way.
    # Position-based (label-contiguity) check needs labeled_mask to actually
    # index into all_llm -- only true when all_llm *is* judge_full itself.
    # An explicit all_judge_scores= has no known positional correspondence
    # to judge_scores/human_scores, so the check is skipped rather than
    # guessed at.
    position_mask = None
    if all_judge_scores is not None:
        all_llm = np.asarray(all_judge_scores, dtype=float)
    elif n_labeled < judge_full.size:
        all_llm = judge_full
        position_mask = labeled_mask
    else:
        all_llm = None
    n_total = _check_n_total(n_total, int(all_llm.size) if all_llm is not None else n_labeled)

    from evalstats.loader import _detect_score_type
    declared_type = score_type is not None
    score_type, grid_step = _resolve_score_type(
        score_type, _detect_score_type(pd.Series(llm_aligned)), llm_aligned, human_aligned)

    result = _judge_alignment_core(
        llm_aligned, human_aligned, score_type,
        llm_metric=llm_metric or "judge", human_groundtruth=human_groundtruth or "human",
        alpha=alpha, n_total=n_total, all_llm=all_llm, grid_step=grid_step,
        slice_df=None, slice_labeled_mask=None,
        labeled_mask=position_mask, selection=selection, test=test, ci=ci,
        warn_stacklevel=4,
    )
    result.score_type_source = "declared" if declared_type else "inferred"
    result.grid_step = grid_step
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Multi-condition (pairwise) alignment -- within-subjects and between-subjects
# comparisons across 2+ named conditions, and the label-efficiency numbers a
# PPI-corrected hypothesis test's savings depend on.
# ─────────────────────────────────────────────────────────────────────────────

_VALID_DESIGNS = ("within", "between")

# Which correlation governs each evalstats.tests function's PPI variance
# reduction, precisely -- not a fixed "Pearson for mean tests, Spearman for
# rank tests" recipe (wrong for rank tests). Every test's rho is actually a
# Pearson correlation on a test-specific LINEARIZATION of the raw values --
# identity for mean-type tests (whose influence function psi(y)=y-mu is
# already linear, hence exactly effect-size-invariant), but a genuine
# transform for rank-type tests, whose named/raw-Spearman recipe drifts with
# effect size instead (see notes/omnibus_label_efficiency.html). See
# _linearize_for_test for the dispatch and each _linearize_* function for
# the actual recipe.
#
# design: the design each test implies, or None if the caller must say
# ("within"/"between" both valid, e.g. ttest/anova_oneway paired vs
# independent). min_k/max_k: condition-count bounds (None = unbounded).
# The tests compare() runs when a metric is judge-corrected, in judge_alignment's
# vocabulary. compare() and judge_alignment both read these, so the efficiency a
# report quotes is always for the test that was actually run. A mismatch would
# describe the power of a test the user never ran.
_BETWEEN_TESTS = {
    # kind of scores -> (omnibus test, pairwise test)
    "binary": ("anova_oneway", "ttest"),
    "rank": ("kruskalwallis", "mannwhitney"),
}
_WITHIN_TESTS = {"pairwise_ci": "ttest", "pairwise_p": "wilcoxon", "omnibus": "friedman"}

# Pairs are listed individually up to this many; beyond it the summary gives a range
# (summary(verbose=True) always lists them).
_MAX_PAIRS_LISTED = 10

_TEST_NAMES = {
    "kruskalwallis": "Kruskal-Wallis", "anova_oneway": "one-way ANOVA", "friedman": "Friedman",
    "mannwhitney": "Mann-Whitney U", "ttest": "t-test", "wilcoxon": "Wilcoxon",
}

_TEST_LABELS = {
    "kruskalwallis": "Kruskal-Wallis (omnibus)", "anova_oneway": "one-way ANOVA (omnibus)",
    "friedman": "Friedman (omnibus)", "mannwhitney": "Mann-Whitney U (pairs)",
    "ttest": "t-test (pairs)", "wilcoxon": "Wilcoxon (pairs)",
    "mean_estimate": "mean estimate",
}


def _comparison_tests(design: str, score_type: str, k: int) -> list:
    """``[(test, role), ...]`` for the tests compare() runs on ``k`` conditions."""
    if design == "within":
        out = [(_WITHIN_TESTS["pairwise_ci"], "pairs"), (_WITHIN_TESTS["pairwise_p"], "pairs")]
        if k >= 3:
            out.insert(0, (_WITHIN_TESTS["omnibus"], "omnibus"))
        return out
    omnibus, pairwise = _BETWEEN_TESTS["binary" if score_type == "binary" else "rank"]
    out = [(pairwise, "pairs")]
    if k >= 3:
        out.insert(0, (omnibus, "omnibus"))
    return out


_TEST_STRUCTURE = {
    "ttest":         {"design": None,      "min_k": 2, "max_k": 2},
    "wilcoxon":      {"design": "within",  "min_k": 2, "max_k": 2},
    "mannwhitney":   {"design": "between", "min_k": 2, "max_k": 2},
    "anova_oneway":  {"design": None,      "min_k": 2, "max_k": None},
    "kruskalwallis": {"design": "between", "min_k": 2, "max_k": None},
    "friedman":      {"design": "within",  "min_k": 2, "max_k": None},
    "mean_estimate": {"design": None,      "min_k": 1, "max_k": 1},
}


def _linearize_mean(conditions: dict, design: str) -> tuple[np.ndarray, np.ndarray]:
    """Identity linearization for mean-type tests (ttest, anova_oneway) --
    Pearson r on (possibly centered/differenced) raw scores IS the governing
    correlation, since a mean's influence function psi(y)=y-mu is linear and
    hence exactly effect-size-invariant; no rank/placement transform needed.
    Generalizes the 2-condition pairwise recipe to k conditions:

    design="within": a plain paired difference at k=2 (same as
    _condition_pair_arrays); DOUBLY centered (each participant's own mean
    AND each condition's mean removed) at k>2 -- row-centering alone leaks
    the condition effect into the correlation (the shared between-condition
    mean judge and humans both track contributes no cross-participant
    variance, but a row-only-centered recipe still credits it), confirmed to
    be exactly what makes repeated-measures ANOVA's recipe effect-invariant
    in notes/omnibus_label_efficiency.html's Method 3 (flat at rho^2=0.646
    for d=0..1.0 there; row-centering alone climbs 0.640->0.862).

    design="between": each condition centered on its own mean, then pooled
    (concatenated) -- the within-group pooled correlation validated for
    anova_oneway in the same note's Method 1; reduces to the existing
    2-condition recipe at k=2.
    """
    names = list(conditions.keys())
    arrs = {n: (np.asarray(j, dtype=float), np.asarray(h, dtype=float)) for n, (j, h) in conditions.items()}

    if design == "within":
        lengths = {len(j) for j, h in arrs.values()}
        if len(lengths) != 1:
            raise ValueError(
                "design='within' requires every condition to have the same "
                "length (same items/participants in the same order)."
            )
        judge_mat = np.column_stack([arrs[n][0] for n in names])
        human_mat = np.column_stack([arrs[n][1] for n in names])
        overlap = ~np.isnan(human_mat).any(axis=1)
        judge_mat, human_mat = judge_mat[overlap], human_mat[overlap]
        if len(names) == 2:
            judge = judge_mat[:, 0] - judge_mat[:, 1]
            human = human_mat[:, 0] - human_mat[:, 1]
        else:
            def double_center(m: np.ndarray) -> np.ndarray:
                return m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + m.mean()
            judge = double_center(judge_mat).ravel()
            human = double_center(human_mat).ravel()
    else:
        judge_parts, human_parts = [], []
        for n in names:
            j, h = arrs[n]
            mask = ~np.isnan(h)
            jj, hh = j[mask], h[mask]
            if len(jj) == 0:
                continue
            judge_parts.append(jj - jj.mean())
            human_parts.append(hh - hh.mean())
        judge = np.concatenate(judge_parts) if judge_parts else np.array([])
        human = np.concatenate(human_parts) if human_parts else np.array([])
    return judge, human


def _linearize_wilcoxon(conditions: dict) -> tuple[np.ndarray, np.ndarray]:
    """Hajek-projection linearization for Wilcoxon signed-rank (paired,
    exactly 2 conditions): the judge-side and human-side paired differences
    are each mapped through ``evalstats.ppi._walsh_theta_h1_components``,
    the per-item empirical Hajek projection ``h1(d) = P(D > -d)`` (mid-ranks
    for ties) of the Walsh/Hodges-Lehmann estimand ``wilcoxon()`` actually
    uses. That is the SAME production function
    ``_analytic_walsh_theta_correct``/``_walsh_theta_analytic_variance``
    already build their variance estimates from, so the correlation
    reported here is taken against the very quantity the correction's own
    variance is computed on, rather than a re-derived lookalike.

    Deliberately not ``sign(d) * (2*F_{|D|}(|d|) - 1)``: that expands to
    ``4*F_D(d) - sign(d) - 2``, which is not affine in ``F_D(d)`` (the
    ``sign`` term survives) and is non-monotonic in ``d``, returning about
    -1 just above zero and about +1 just below it.

    Replaces the raw-Spearman-of-differences recipe, which drifts with
    effect size (see notes/omnibus_label_efficiency.html)."""
    from evalstats.ppi import _walsh_theta_h1_components

    names = list(conditions.keys())
    if len(names) != 2:
        raise ValueError(f"wilcoxon needs exactly 2 conditions, got {len(names)}.")
    (ja, ha), (jb, hb) = conditions[names[0]], conditions[names[1]]
    ja, ha = np.asarray(ja, dtype=float), np.asarray(ha, dtype=float)
    jb, hb = np.asarray(jb, dtype=float), np.asarray(hb, dtype=float)
    if not (len(ja) == len(ha) == len(jb) == len(hb)):
        raise ValueError("wilcoxon requires both conditions to have the same length (same items in the same order).")
    mask = ~np.isnan(ha) & ~np.isnan(hb)
    judge = _walsh_theta_h1_components(ja[mask] - jb[mask])
    human = _walsh_theta_h1_components(ha[mask] - hb[mask])
    return judge, human


def _linearize_mannwhitney(conditions: dict) -> tuple[np.ndarray, np.ndarray]:
    """Empirical placement-value linearization for Mann-Whitney/Wilcoxon
    rank-sum (independent groups, exactly 2 conditions), the influence
    function of the ``theta = P(X > Y)`` estimand ``mannwhitney()`` uses.

    For item ``x_i`` in group A the score is ``F_Y(x_i)``, its mid-rank
    placement within group B; for item ``y_j`` in group B it is
    ``P(X > y_j) = 1 - F_X(y_j)`` -- not ``-F_X(y_j)``: both have the same
    spread, but their means differ by 1 (``theta - 1`` vs ``theta``), so
    pooling would put the two halves a constant ~1.0 apart on both the
    judge and human side, a lockstep offset that Pearson would then score
    as agreement. Built on the same searchsorted mid-rank construction
    already used and tested in ``evalstats.tests._p_x_gt_y_midrank`` for
    the point estimate itself, extracted per item instead of summed to one
    ``P(X > Y)`` number.

    Both halves are then centered on their own mean before pooling: the
    pooled correlation must be the within-group one, since any
    between-group difference in mean placement is shared by judge and
    humans and would again be counted as agreement -- the same fix as the
    uncentered pooling corrected in ``_pooled_two_group_lambda``, and as
    ``_linearize_mean``'s "between" branch.

    Replaces the raw-Spearman recipe, which drifts with effect size (see
    notes/omnibus_label_efficiency.html).

    Coarse-scale caveat (likert): placement values take only ~k distinct
    levels on a k-point scale, so the influence function loses most of its
    spread and MWU/kruskal run conservative. Paired rank tests
    (wilcoxon/friedman) are unaffected -- they score differences, not
    cross-group comparisons. See api._ppi_pairwise's mannwhitney branch."""
    names = list(conditions.keys())
    if len(names) != 2:
        raise ValueError(f"mannwhitney needs exactly 2 conditions, got {len(names)}.")
    (ja, ha), (jb, hb) = conditions[names[0]], conditions[names[1]]
    ja, ha = np.asarray(ja, dtype=float), np.asarray(ha, dtype=float)
    jb, hb = np.asarray(jb, dtype=float), np.asarray(hb, dtype=float)
    mask_a, mask_b = ~np.isnan(ha), ~np.isnan(hb)
    ja, ha = ja[mask_a], ha[mask_a]
    jb, hb = jb[mask_b], hb[mask_b]

    def placement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(y) == 0:
            return np.zeros_like(x)
        y_sorted = np.sort(y)
        n_lt = np.searchsorted(y_sorted, x, side="left")
        n_le = np.searchsorted(y_sorted, x, side="right")
        return (n_lt + 0.5 * (n_le - n_lt)) / len(y)

    def _pool(a_scores: np.ndarray, b_scores: np.ndarray) -> np.ndarray:
        if len(a_scores) == 0 or len(b_scores) == 0:
            return np.array([])
        return np.concatenate([a_scores - a_scores.mean(), b_scores - b_scores.mean()])

    judge = _pool(placement(ja, jb), 1.0 - placement(jb, ja))
    human = _pool(placement(ha, hb), 1.0 - placement(hb, ha))
    return judge, human


def _linearize_kruskal(conditions: dict) -> tuple[np.ndarray, np.ndarray]:
    """Spearman of within-condition-centered, pooled values -- the
    validated recipe for Kruskal-Wallis (notes/omnibus_label_efficiency.html
    Method 2): each condition's judge/human values centered on that
    condition's own mean (removing the between-condition location signal,
    exactly like ``_linearize_mean``'s "between" branch), then pooled
    (concatenated) across conditions, then rank-transformed as one combined
    array -- not ranked within each condition separately first, which would
    discard the between-condition-relative information centering-then-
    pooling is meant to preserve. Spearman correlation of the
    pooled-then-globally-ranked residuals is, by definition, Pearson
    correlation of their ranks; that's what's returned here.

    Inherited caveat, documented in the note and not fixed here: this
    recipe is effect-invariant while the true implied rho^2 falls with
    effect size, so it runs mildly optimistic -- treat Kruskal-Wallis's
    number as a ceiling rather than a point estimate when a large effect is
    expected. Same rank-drift phenomenon that hits Friedman harder (see
    :func:`_linearize_friedman`), in mild form; no doubly-centred/plug-in
    replacement for it has been validated."""
    from scipy.stats import rankdata

    judge_parts, human_parts = [], []
    for j, h in conditions.values():
        j, h = np.asarray(j, dtype=float), np.asarray(h, dtype=float)
        mask = ~np.isnan(h)
        jj, hh = j[mask], h[mask]
        if len(jj) == 0:
            continue
        judge_parts.append(jj - jj.mean())
        human_parts.append(hh - hh.mean())
    judge_pooled = np.concatenate(judge_parts) if judge_parts else np.array([])
    human_pooled = np.concatenate(human_parts) if human_parts else np.array([])
    return rankdata(judge_pooled), rankdata(human_pooled)


def _linearize_friedman(conditions: dict) -> tuple[np.ndarray, np.ndarray]:
    """Doubly-centered within-subject ranks -- the validated recipe for
    Friedman (notes/omnibus_label_efficiency.html Method 4): rank each
    participant's k conditions (row-wise) for judge and humans alike,
    subtract each condition's (column) mean rank, correlate the pooled
    residuals. Not the average per-participant Spearman -- that recipe
    moves in the opposite direction from the truth as effect size grows
    (rising while the truth falls). The row-wise rank transform substitutes
    for row-centering (ranks are already row-normalized by construction);
    only the column (condition) mean needs explicit removal."""
    from scipy.stats import rankdata

    names = list(conditions.keys())
    arrs = {n: (np.asarray(j, dtype=float), np.asarray(h, dtype=float)) for n, (j, h) in conditions.items()}
    lengths = {len(j) for j, h in arrs.values()}
    if len(lengths) != 1:
        raise ValueError(
            "friedman requires every condition to have the same length "
            "(same items/participants in the same order)."
        )
    judge_mat = np.column_stack([arrs[n][0] for n in names])
    human_mat = np.column_stack([arrs[n][1] for n in names])
    overlap = ~np.isnan(human_mat).any(axis=1)
    judge_mat, human_mat = judge_mat[overlap], human_mat[overlap]
    judge_ranks = rankdata(judge_mat, axis=1, method="average")
    human_ranks = rankdata(human_mat, axis=1, method="average")
    judge = (judge_ranks - judge_ranks.mean(axis=0, keepdims=True)).ravel()
    human = (human_ranks - human_ranks.mean(axis=0, keepdims=True)).ravel()
    return judge, human


def _linearize_for_test(
    conditions: dict, *, test: str, design: Optional[str],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Dispatch to the right _linearize_* function for `test`, validating
    condition count and design against _TEST_STRUCTURE first. Returns
    (judge_linearized, human_linearized, resolved_design)."""
    if test not in _TEST_STRUCTURE:
        raise ValueError(f"Unrecognized test={test!r}. Known: {sorted(_TEST_STRUCTURE)}.")
    spec = _TEST_STRUCTURE[test]
    k = len(conditions)
    if k < spec["min_k"] or (spec["max_k"] is not None and k > spec["max_k"]):
        bound = f"exactly {spec['min_k']}" if spec["min_k"] == spec["max_k"] else f"at least {spec['min_k']}"
        raise ValueError(f"test={test!r} needs {bound} condition(s), got {k}.")

    implied = spec["design"]
    if implied is not None:
        if design is not None and design != implied:
            raise ValueError(f"test={test!r} is always design={implied!r}; design={design!r} conflicts.")
        design = implied
    elif design is None and test != "mean_estimate":
        raise ValueError(f"test={test!r} needs an explicit design= ('within' or 'between').")

    if test in ("ttest", "anova_oneway"):
        judge, human = _linearize_mean(conditions, design)
    elif test == "wilcoxon":
        judge, human = _linearize_wilcoxon(conditions)
    elif test == "mannwhitney":
        judge, human = _linearize_mannwhitney(conditions)
    elif test == "kruskalwallis":
        judge, human = _linearize_kruskal(conditions)
    elif test == "friedman":
        judge, human = _linearize_friedman(conditions)
    else:  # mean_estimate
        (j, h), = conditions.values()
        j, h = np.asarray(j, dtype=float), np.asarray(h, dtype=float)
        mask = ~np.isnan(h)
        judge, human = j[mask], h[mask]
    return judge, human, design


def _pearson_spearman_metrics(
    judge: np.ndarray, human: np.ndarray, *, alpha: float, rng: np.random.Generator,
    pearson_label: str, spearman_label: str, what_suffix: str = "",
    ci2_fn=_bootstrap_ci_2,
    pearson_what: Optional[str] = None,
    pearson_why: Optional[str] = None,
    spearman_what: Optional[str] = None,
    spearman_why: Optional[str] = None,
) -> dict:
    """Pearson r and Spearman r (point estimate + bootstrap CI) between two
    already-prepared 1-D arrays -- the shared low-level computation behind
    both the multi-condition pairwise path and the single-condition
    :func:`_compute_alignment_metrics`, so there is exactly one place this
    math lives. Callers are responsible for whatever differencing/pooling/
    masking the two arrays need before calling this (see
    :func:`_condition_pair_arrays`).

    ``ci2_fn`` defaults to :func:`_bootstrap_ci_2` but can be swapped for a
    cheaper stand-in (e.g. skipping the bootstrap entirely) by callers that
    don't need a CI, matching :func:`_compute_alignment_metrics`'s ``ci=``
    parameter. ``pearson_what``/``pearson_why``/``spearman_what``/
    ``spearman_why`` override the generic "what"/"why" text below when a
    caller has more specific, score-type-tailored wording to show instead.
    """
    n = len(judge)

    def pe(a, b):
        return _quiet_corr(pearsonr, a, b)

    def sp(a, b):
        return _quiet_corr(spearmanr, a, b)

    est, lo, hi = ci2_fn(pe, judge, human, alpha=alpha, rng=rng)
    band, interp, example = _interpret_corr(est, lo, hi, n, pearson_label)
    pearson_entry = {
        "estimate": est, "ci_low": lo, "ci_high": hi, "label": pearson_label, "band": band, "n": n,
        "what": pearson_what or f"Linear correlation coefficient between judge and human values{what_suffix}.",
        "why": pearson_why or (
            "Governs the label-efficiency multiplier for parametric/mean-based "
            "tests (t-test, ANOVA, mean estimation) -- see judge_alignment()'s test=."
        ),
        "interpretation": interp, "example": example,
    }
    est, lo, hi = ci2_fn(sp, judge, human, alpha=alpha, rng=rng)
    band, interp, example = _interpret_corr(est, lo, hi, n, spearman_label)
    spearman_entry = {
        "estimate": est, "ci_low": lo, "ci_high": hi, "label": spearman_label, "band": band, "n": n,
        "what": spearman_what or f"Rank correlation coefficient between judge and human values{what_suffix}.",
        "why": spearman_why or (
            "Governs the label-efficiency multiplier for rank-based tests "
            "(Mann-Whitney, Wilcoxon, Friedman) -- see judge_alignment()'s test=."
        ),
        "interpretation": interp, "example": example,
    }
    return {"pearson_r": pearson_entry, "spearman_r": spearman_entry}


def _condition_pair_arrays(
    judge_a, human_a, judge_b, human_b, *, design: str, label_a: str, label_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one pair of conditions to the two 1-D arrays whose correlation
    actually governs that pair's PPI variance reduction, per `design`:

    "within" (paired/repeated-measures): the estimand is a function of the
    per-item DIFFERENCE between conditions, so the governing correlation is
    between the two conditions' differences -- Corr(judge_a - judge_b,
    human_a - human_b) -- not between either condition's raw scores. Uses
    only items labeled in BOTH conditions (the overlap).

    "between" (independent groups): the estimand spans both groups, so the
    governing correlation is the WITHIN-GROUP pooled one -- each condition's
    judge/human values centered on their own mean, then concatenated. Plain
    pooling without per-group centering would inflate/deflate the
    correlation by the between-group difference itself, which isn't part of
    what the control variate is being credited for.
    """
    judge_a = np.asarray(judge_a, dtype=float)
    human_a = np.asarray(human_a, dtype=float)
    judge_b = np.asarray(judge_b, dtype=float)
    human_b = np.asarray(human_b, dtype=float)

    if design == "within":
        if not (len(judge_a) == len(human_a) == len(judge_b) == len(human_b)):
            raise ValueError(
                f"design='within' requires {label_a!r} and {label_b!r} to have the "
                "same length (same items in the same order) -- pass NaN in the "
                "human array for items without a label, not a shorter array."
            )
        mask = ~np.isnan(human_a) & ~np.isnan(human_b)
        judge = judge_a[mask] - judge_b[mask]
        human = human_a[mask] - human_b[mask]
    else:
        mask_a = ~np.isnan(human_a)
        mask_b = ~np.isnan(human_b)
        ja, ha = judge_a[mask_a], human_a[mask_a]
        jb, hb = judge_b[mask_b], human_b[mask_b]
        judge = np.concatenate([ja - ja.mean(), jb - jb.mean()]) if len(ja) and len(jb) else np.array([])
        human = np.concatenate([ha - ha.mean(), hb - hb.mean()]) if len(ha) and len(hb) else np.array([])

    if len(judge) < 3:
        raise ValueError(
            f"Not enough overlapping labeled items between {label_a!r} and "
            f"{label_b!r} (n={len(judge)}) to compute a correlation."
        )
    return judge, human


def _add_rho2(metric: dict) -> dict:
    """Add ``rho2`` (and its CI) to a test-specific correlation metric dict.

    rho^2, not rho, is the quantity the label-efficiency multiplier and the
    reporting guidance are stated in, so it is the number that gets printed.
    The CI is the correlation's CI squared; when it straddles zero the lower
    bound is 0."""
    r, lo, hi = metric["estimate"], metric["ci_low"], metric["ci_high"]
    metric["rho2"] = float(r) ** 2
    if np.isfinite(lo) and np.isfinite(hi):
        if lo >= 0:
            metric["rho2_ci_low"], metric["rho2_ci_high"] = lo ** 2, hi ** 2
        elif hi <= 0:
            metric["rho2_ci_low"], metric["rho2_ci_high"] = hi ** 2, lo ** 2
        else:
            metric["rho2_ci_low"], metric["rho2_ci_high"] = 0.0, max(lo ** 2, hi ** 2)
    else:
        metric["rho2_ci_low"] = metric["rho2_ci_high"] = float("nan")
    return metric


def _single_metric(
    judge: np.ndarray, human: np.ndarray, *, alpha: float, rng: np.random.Generator,
    label: str, what: str = "", why: str = "", ci: bool = True,
) -> dict:
    """One Pearson-r metric dict (point estimate + bootstrap CI) from two
    already-linearized 1-D arrays -- the shared low-level computation
    behind every _linearize_* function's reported rho. Always Pearson: once
    the test-specific linearization has been applied (identity for
    mean-type tests, Hajek/placement/rank-based for the others), Pearson r
    of the two linearized arrays IS the governing correlation -- see
    _TEST_STRUCTURE's docstring."""
    n = len(judge)

    def pe(a, b):
        return _quiet_corr(pearsonr, a, b)

    if ci:
        est, lo, hi = _bootstrap_ci_2(pe, judge, human, alpha=alpha, rng=rng)
    else:
        est, lo, hi = float(pe(judge, human)), float("nan"), float("nan")
    band, interp, example = _interpret_corr(est, lo, hi, n, label)
    return _add_rho2({
        "estimate": est, "ci_low": lo, "ci_high": hi, "label": label, "band": band, "n": n,
        "what": what, "why": why, "interpretation": interp, "example": example,
    })


class PairwiseAlignmentResult:
    """Judge-human correlation for every pair among 2+ named conditions.

    Internal: the per-condition machinery behind the comparison report that
    :func:`judge_alignment` builds from evaldata (``AlignmentResult.comparison``),
    and that ``compare()`` asks for directly. Answers "how well does my judge
    track human labels for the comparison I'm about to run" across every
    pairwise comparison, rather than a single item-level number -- see
    :attr:`pairwise_metrics`.

    Attributes
    ----------
    conditions : list[str]
        Condition names, in input order.
    design : {"within", "between"}
        Whether each pair's correlation was computed on within-subject
        differences or between-subjects pooled values -- see
        :func:`_condition_pair_arrays`.
    pairwise_metrics : dict[tuple[str, str], dict]
        ``(condition_a, condition_b) -> {"pearson_r": {...}, "spearman_r": {...}}``,
        one entry per unordered pair -- the RAW correlations, for
        comparability to prior work. NOT necessarily the correlation that
        governs your test's PPI variance reduction for rank-based tests
        (see :attr:`test`/:attr:`test_pairwise_metrics`/:attr:`omnibus_metric`).
    test : str or None
        The test named via ``test=``, if any -- see :func:`judge_alignment`.
    test_pairwise_metrics : dict[tuple[str, str], dict] or None
        Only set when ``test`` needs exactly 2 conditions (ttest, wilcoxon,
        mannwhitney): that test's CORRECT, test-specific linearized rho for
        every pair -- e.g. the Hajek-projection correlation for wilcoxon,
        not raw Spearman. This is the number to use for planning/reporting
        a pairwise (e.g. post-hoc) run of that test between two conditions.
    omnibus_metric : dict or None
        Only set when ``test`` can span 2+ conditions at once
        (anova_oneway, kruskalwallis, friedman): that test's own validated
        whole-design rho, computed once across ALL conditions together
        (not decomposable into pairs) -- see the relevant
        ``_linearize_*`` function's docstring for the recipe and its
        Monte-Carlo validation in notes/omnibus_label_efficiency.html.
    condition_counts : dict[str, tuple[int, int]]
        ``name -> (n_labeled, n_total)`` for each condition, as passed --
        for reporting "N_lab and N per condition/measure" alongside the
        correlations above. Not used in any correlation/N_eff computation
        itself (those use the labeled OVERLAP / pooled totals actually
        involved in each specific pair or the whole design).
    selection : str
        How the labeled subset was chosen -- see :func:`judge_alignment`'s
        ``selection=``. Same MCAR-assumption warning as the single-
        condition form when left at ``"unknown"``.

    Every metric dict in ``pairwise_metrics``/``test_pairwise_metrics``/
    ``omnibus_metric`` also carries ``multiplier``/``n_eff`` (the
    label-efficiency savings implied by that correlation, at the N/N_lab
    actually spanned by that specific correlation -- summed across the
    conditions it pools, for a "between" pair/omnibus, or a single
    condition's N for a "within" one, where every condition shares the
    same items by construction). These are the ORACLE bound -- the
    efficiency available at the variance-minimizing lambda, validated to
    within 1.7% against direct oracle-lambda simulation. A particular
    corrected test may realize less, since ``evalstats.tests`` sometimes
    trades efficiency for calibration when choosing lambda; see
    :func:`_attach_savings`'s docstring for the measured size of that gap
    for ``wilcoxon(power_tune=True)``.

    Notes
    -----
    With 3+ conditions, ``pairwise_metrics``/``test_pairwise_metrics`` are
    NOT statistically independent of each other across pairs (e.g. "post vs
    pre" and "post vs mid" both involve the "post" condition's data) --
    fine to report each pair's own number, but don't average them across
    pairs as if they were independent samples.
    """

    def __init__(
        self, *, conditions: list, design: str, pairwise_metrics: dict,
        condition_counts: dict, selection: str = "unknown",
        test: Optional[str] = None, test_pairwise_metrics: Optional[dict] = None,
        omnibus_metric: Optional[dict] = None,
    ) -> None:
        self.conditions = conditions
        self.design = design
        self.pairwise_metrics = pairwise_metrics
        self.condition_counts = condition_counts
        self.selection = selection
        self.test = test
        self.test_pairwise_metrics = test_pairwise_metrics
        self.omnibus_metric = omnibus_metric

    @staticmethod
    def _print_metric(d: dict) -> None:
        if "rho2" in d:
            # A test-specific correlation is reported as rho^2.
            print(
                f"    {d['label'].replace(' rho', ' rho^2'):<28} {d['rho2']:.3f}  "
                f"95% CI [{d['rho2_ci_low']:.3f}, {d['rho2_ci_high']:.3f}]  (n={d['n']})"
            )
        else:
            print(
                f"    {d['label']:<28} {d['estimate']:+.3f}  "
                f"95% CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  (n={d['n']})"
            )
        # With every item labeled there is nothing to extrapolate to, so N_eff would
        # just restate n. Pass n_total= to judge_alignment to project it onto a larger set.
        if "n_eff" in d and d["N"] > d["n"]:
            print(f"        multiplier = {d['multiplier']:.2f}x   N_eff = {d['n_eff']:.0f}  (N={d['N']})")

    def summary(self) -> None:
        """Print one line per pair per metric."""
        print("Pairwise judge alignment report")
        print("─" * 58)
        print(f"Conditions : {', '.join(self.conditions)}")
        print(f"Design     : {self.design}-subjects")
        for name in self.conditions:
            n_lab, n_tot = self.condition_counts[name]
            print(f"  {name}: N_lab={n_lab}, N={n_tot} ({100.0*n_lab/n_tot:.1f}% labeled)")
        if len(self.conditions) > 2:
            print(
                "Note: pairwise correlations below are not independent of "
                "each other (they share conditions) -- see class docstring."
            )
        print()
        if self.omnibus_metric is not None:
            print(f"test={self.test!r} (whole-design, all {len(self.conditions)} conditions):")
            self._print_metric(self.omnibus_metric)
            print()
        for (a, b), metrics in self.pairwise_metrics.items():
            print(f"{a} vs {b}:")
            for entry in metrics.values():
                self._print_metric(entry)
            if self.test_pairwise_metrics is not None:
                self._print_metric(self.test_pairwise_metrics[(a, b)])
            print()


def _resolve_condition_totals(conditions: dict, n_total, design: str) -> Optional[dict]:
    """Per-condition full-dataset sizes from ``judge_alignment(n_total=...)``.

    ``None`` -> ``None`` (use the lengths of the arrays as passed). A dict gives
    each condition's size directly. An int is the whole design's size: shared by
    every condition for ``design="within"`` (the same items sit in all of them),
    or split across conditions in proportion to the sizes passed in for
    ``"between"``. Every size must be at least the number of items passed in."""
    if n_total is None:
        return None
    lens = {n: len(np.asarray(conditions[n][0])) for n in conditions}
    if isinstance(n_total, dict):
        missing = [n for n in conditions if n not in n_total]
        if missing:
            raise ValueError(f"n_total is missing conditions {missing}; expected one entry per condition.")
        totals = {n: int(n_total[n]) for n in conditions}
    elif design == "within":
        totals = {n: int(n_total) for n in conditions}
    else:
        whole = float(sum(lens.values()))
        if design == "between" and int(n_total) < whole:
            raise ValueError(
                f"n_total={n_total}, but {int(whole)} items were passed in across the conditions; "
                "n_total is the size of the full dataset and can't be smaller."
            )
        totals = {n: int(round(int(n_total) * lens[n] / whole)) for n in conditions}
    for n, t in totals.items():
        if t < lens[n]:
            raise ValueError(
                f"n_total for {n!r} is {t}, but {lens[n]} items were passed in for it; "
                "n_total is the size of the full dataset and can't be smaller."
            )
    if design == "within" and len(set(totals.values())) > 1:
        raise ValueError("With design='within' every condition shares the same items, so n_total must be one number.")
    return totals


def _pair_total_n(conditions: dict, names: list, design: str, totals: Optional[dict] = None) -> int:
    """Total (labeled+unlabeled) item count spanned by a correlation over
    `names`' conditions: one condition's length for design="within" (every
    condition shares the same items by construction, enforced elsewhere),
    or the sum across conditions for design="between" (independent groups,
    pooled)."""
    size = (lambda n: totals[n]) if totals else (lambda n: len(np.asarray(conditions[n][0])))
    if design == "within":
        return size(names[0])
    return sum(size(n) for n in names)


def _full_linearized_n(conditions: dict, *, test: str, design: Optional[str]) -> int:
    """The savings formula's total item count, in the SAME units as the
    linearized arrays whose correlation feeds it.

    ``_pair_total_n`` counts items, which is right whenever a test's
    linearization emits one number per item (``ttest``/``wilcoxon`` score
    differences; the between-design tests emit one per observation, and its
    ``design="between"`` branch sums to match). ``friedman`` is the exception:
    ``_linearize_friedman`` ravels an (items x k) matrix, so its labeled count
    is ``k`` times the item count. Pairing that with an item-scale total made
    ``n_lab/N`` about ``k`` times too large, collapsing the multiplier toward 1
    and reporting an ``n_eff`` on the wrong scale.

    Relabeling every human score as observed and re-linearizing gives the
    length the arrays would have under full labeling -- correct for any test,
    without a per-test table of emission rates. Only lengths are read, so the
    placeholder value never reaches a correlation.
    """
    full = {}
    for name, (judge, human) in conditions.items():
        human = np.asarray(human, dtype=float)
        full[name] = (np.asarray(judge, dtype=float), np.zeros_like(human))
    judge_full, _, _ = _linearize_for_test(full, test=test, design=design)
    return int(len(judge_full))


def _attach_savings(metric: dict, N: int) -> dict:
    """Attach multiplier/n_eff to a correlation metric dict, using `N`
    (see :func:`_pair_total_n`) as the savings formula's total item count.

    With ``lam*`` the variance-minimizing PPI++ weight, the algebra gives
    ``Var_min = (V_h/n_lab) * [1 - rho^2 * (n_unlab/N)]``, i.e. exactly
    ``multiplier = 1/(1 - rho^2*(1 - n_lab/N))`` with ``rho`` the
    correlation of the two sides' influence functions. The multiplier
    depends on ``N`` and ``n_lab`` only through their ratio, so pooling
    equal-sized groups preserves it exactly.

    One standing caveat, about the library's lambda rather than this
    formula: ``multiplier``/``n_eff`` are the oracle bound -- what is
    achievable at the variance-minimizing lambda. ``evalstats.tests``
    deliberately does not always use that lambda: ``wilcoxon(power_tune=True)``
    evaluates the human term's variance under H0 (sign-flip null variance)
    rather than plug-in, a deliberate calibration trade documented in
    ``evalstats.ppi._analytic_walsh_theta_correct``, which is intentionally
    sub-optimal and drifts further from optimal as the true effect moves
    away from H0. So report these as "the efficiency this judge makes
    available", not as a guarantee of what a particular corrected test
    will realize."""
    mult, n_eff = _n_eff(metric["estimate"], metric["n"], N)
    metric["N"] = N
    metric["multiplier"] = mult
    metric["n_eff"] = n_eff
    return metric


def _judge_alignment_pairwise(
    conditions: dict, *, design: Optional[str], alpha: float,
    test: Optional[str] = None, selection: str = "unknown", warn_stacklevel: int = 3,
    n_total=None, ci: bool = True, include_raw: bool = True,
) -> PairwiseAlignmentResult:
    if len(conditions) < 2:
        raise ValueError(
            "judge_alignment(conditions_dict) needs at least 2 conditions. "
            "For a single condition, call judge_alignment(judge_scores, human_scores) "
            "with plain arrays instead."
        )
    _validate_and_warn_selection(selection, warn_stacklevel)

    resolved_design = design
    if test is not None:
        if test not in _TEST_STRUCTURE or _TEST_STRUCTURE[test]["min_k"] < 2:
            multi_cond_tests = sorted(t for t, s in _TEST_STRUCTURE.items() if s["min_k"] >= 2)
            raise ValueError(f"Unrecognized or single-condition-only test={test!r}. Known: {multi_cond_tests}.")
        implied = _TEST_STRUCTURE[test]["design"]
        if implied is not None:
            if design is not None and design != implied:
                raise ValueError(f"test={test!r} is always design={implied!r}; design={design!r} conflicts.")
            resolved_design = implied

    if resolved_design not in _VALID_DESIGNS:
        raise ValueError(
            f"design={resolved_design!r} -- with 2+ conditions, pass design='within' "
            "(paired/repeated-measures: the same items/participants in every "
            "condition) or design='between' (independent groups) explicitly, "
            "or a test= that implies one. This can't be inferred from the data "
            "alone -- two conditions look the same positionally whether they're "
            "a paired comparison or two independent groups, and each needs "
            "different math."
        )

    names = list(conditions.keys())
    totals = _resolve_condition_totals(conditions, n_total, resolved_design)
    condition_counts = {}
    for n in names:
        j, h = conditions[n]
        h = np.asarray(h, dtype=float)
        condition_counts[n] = (int((~np.isnan(h)).sum()), totals[n] if totals else len(h))

    rng = np.random.default_rng(42)
    pairwise_metrics = {}
    for name_a, name_b in (combinations(names, 2) if include_raw else ()):
        judge_a, human_a = conditions[name_a]
        judge_b, human_b = conditions[name_b]
        judge, human = _condition_pair_arrays(
            judge_a, human_a, judge_b, human_b, design=resolved_design, label_a=name_a, label_b=name_b,
        )
        pair_n = _pair_total_n(conditions, [name_a, name_b], resolved_design, totals)
        metrics = _pearson_spearman_metrics(
            judge, human, alpha=alpha, rng=rng,
            pearson_label="Pearson r", spearman_label="Spearman r",
            what_suffix=(
                f" between {name_a} and {name_b}'s differences" if resolved_design == "within"
                else f" between {name_a} and {name_b}, within-group centered"
            ),
        )
        for m in metrics.values():
            _attach_savings(m, pair_n)
        pairwise_metrics[(name_a, name_b)] = metrics

    test_pairwise_metrics = None
    omnibus_metric = None
    if test is not None:
        spec = _TEST_STRUCTURE[test]
        if spec["max_k"] == 2:
            test_pairwise_metrics = {}
            for name_a, name_b in combinations(names, 2):
                pair = {name_a: conditions[name_a], name_b: conditions[name_b]}
                jl, hl, _ = _linearize_for_test(pair, test=test, design=resolved_design)
                pair_n = _pair_total_n(conditions, [name_a, name_b], resolved_design, totals)
                m = _single_metric(
                    jl, hl, alpha=alpha, rng=rng, label=f"{test} rho (test-correct)",
                    why=f"The correlation that actually governs {test}'s PPI variance reduction, not raw Pearson/Spearman.",
                    ci=ci,
                )
                test_pairwise_metrics[(name_a, name_b)] = _attach_savings(m, pair_n)
        else:
            jl, hl, _ = _linearize_for_test(conditions, test=test, design=resolved_design)
            whole_n = _full_linearized_n(conditions, test=test, design=resolved_design)
            if totals:   # rescale from the items passed in to the full dataset
                whole_n = int(round(whole_n * _pair_total_n(conditions, names, resolved_design, totals)
                                    / _pair_total_n(conditions, names, resolved_design)))
            m = _single_metric(
                jl, hl, alpha=alpha, rng=rng, label=f"{test} rho (whole-design)",
                why=f"{test}'s own validated correlation across all conditions at once -- not decomposable into pairs.",
                ci=ci,
            )
            omnibus_metric = _attach_savings(m, whole_n)

    return PairwiseAlignmentResult(
        conditions=names, design=resolved_design, pairwise_metrics=pairwise_metrics,
        condition_counts=condition_counts, selection=selection,
        test=test, test_pairwise_metrics=test_pairwise_metrics, omnibus_metric=omnibus_metric,
    )


def judge_alignment(
    judge_scores_or_evaldata,
    human_scores=None,
    *,
    llm_metric: Optional[str] = None,
    human_groundtruth: Optional[str] = None,
    all_judge_scores=None,
    score_type: Optional[str] = None,
    design: Optional[str] = None,
    test: Optional[str] = None,
    alpha: float = 0.05,
    selection: str = "unknown",
    ci: bool = True,
    factors=None,
    n_total=None,
) -> AlignmentResult:
    """Validate how well an LLM judge aligns with human graders.

    Two call forms:

    1. ``judge_alignment(evaldata, llm_metric=..., human_groundtruth=...)`` --
       the form to use. ``evaldata`` is what you pass to :func:`compare`
       (from :func:`load_from`); judge scores exist for every item and human
       labels for a subset, identified by column name. The result holds the
       agreement metrics (quadratic-weighted kappa for ordinal scores,
       correlations, ICC), a per-condition breakdown, the representativeness
       checks, and -- when the data has 2+ conditions -- rho^2 and N_eff for the
       tests :func:`compare` will run on it (see ``comparison``). Pass it on as
       ``compare(alignment={metric: result})``.
    2. ``judge_alignment(judge_scores, human_scores)`` -- a quick form for two
       plain arrays of the same length, ``NaN`` in ``human_scores`` for items
       with no label. One condition, so there is no comparison to describe;
       ``test="mean_estimate"`` gives the rho^2 of a plain mean. The result
       can't be passed to ``compare(alignment=...)``: it has no underlying
       DataFrame to look values up in.

    Every form fits a Bayesian calibration model that is later used to
    propagate judge uncertainty into downstream comparisons.

    Parameters
    ----------
    judge_scores_or_evaldata : EvalResults or array-like
        Evaluation data from :func:`load_from` (form 1), or every item's judge
        score (form 2).
    human_scores : array-like, optional
        Form 2 only: same length as the judge scores, ``NaN`` for unlabeled
        items.
    llm_metric : str, optional
        Form 1: column name of the LLM judge scores (required). Form 2:
        optional display name for the judge, used only in printed reports.
    human_groundtruth : str, optional
        Form 1: column name of the human rater scores (required), non-null for
        the alignment subset and ``NaN`` elsewhere. Form 2: optional display
        name for the human rater, used only in printed reports.
    all_judge_scores : array-like, optional
        Form 2 only: override which array is treated as "every item's judge
        score" for the representativeness check.
    score_type : str, optional
        Declare the score type (``"binary"``, ``"likert"``, or ``"continuous"``)
        instead of having it detected. Detected from the labeled judge and
        human scores when not given: 0/1 is binary, whole numbers or any other
        bounded quantization grid of at most ``MAX_DISCRETE_LEVELS`` levels
        (e.g. half-point scores) is likert, and anything else is continuous.
        Likert scores get quadratic-weighted Cohen's kappa, computed on grid
        positions. Declaring ``"likert"`` for scores that don't sit on a grid
        raises -- round them first.
    design : {"within", "between"}, optional
        Form 1: whether the conditions share the same items (within-subjects,
        also spelled ``"paired"``) or have different items (between-subjects,
        ``"unpaired"``). Worked out from the item ids when not given; pass it
        when the ids don't settle it.
    test : str, optional
        Which test's rho^2 and N_eff to report. By default, the tests
        :func:`compare` runs: Kruskal-Wallis (omnibus, 3+ conditions) and
        Mann-Whitney for between-subjects, or binary data's ANOVA and t-test;
        Friedman, t-test and Wilcoxon for within-subjects. Name one of
        ``"ttest"``, ``"wilcoxon"``, ``"mannwhitney"``, ``"anova_oneway"``,
        ``"kruskalwallis"``, ``"friedman"`` to report only that one. rho^2 depends
        on the test: a rank test's governing correlation is computed on a
        test-specific transform of the scores, not on the raw scores, so the
        raw Pearson and Spearman r above are not a substitute. Form 2 accepts
        ``"mean_estimate"``.
    n_total : int or dict, optional
        The size of the full dataset, for when the items you pass in are only
        the human-labeled validation set (nothing left unlabeled). With every
        item labeled there is no larger pool to project onto, so the report
        leaves out N_eff; give the size of the dataset the judge will actually
        score and it comes back, computed for labeling these items out of
        ``n_total``. An int, or for form 1 with several conditions a
        ``{condition: size}`` dict (an int is split across conditions in
        proportion to their sizes when between-subjects, and shared by all of
        them when within-subjects). Must be at least the number of items
        passed in.
    alpha : float
        Significance level for alignment metric CIs.  Default ``0.05``.
    selection : {"random", "stratified", "manual", "unknown"}
        How the labeled subset was chosen. Every correction this function
        (and ``compare(alignment=...)``) applies assumes the labeled items
        are a random sample of the full item pool ("missing completely at
        random", MCAR) -- pass ``"random"`` to confirm that's the case.
        Anything else (including the ``"unknown"`` default) raises a
        ``UserWarning`` explaining the risk, since the natural QA instinct
        -- hand-labeling the items you're least sure about -- is exactly
        the kind of selection that breaks this assumption. ``"manual"``
        and ``"stratified"`` are for acknowledging a known-non-random
        selection explicitly rather than leaving it unexamined.
    ci : bool, default True
        Whether to bootstrap confidence intervals for the alignment metrics.
        ``ci=False`` returns the point estimates with NaN bounds and skips
        roughly 2000 resamples per metric -- the bulk of this function's
        runtime. Use it when only the estimates are consumed, or in large
        sweeps.
    factors : str, optional
        Form 1: the column that defines the conditions, for the per-condition
        report and the comparison's rho^2. Defaults to the ``model`` role
        column, then to the sole factor column. Required when the data has more
        than one factor and no ``model`` role.

    Returns
    -------
    AlignmentResult
        ``.alignment_metrics`` holds the agreement metrics; ``.comparison``
        holds rho^2 and N_eff per test for a multi-condition form 1 (``None``
        otherwise, with ``.comparison_error`` saying why when it could not be
        worked out); ``.summary()`` prints it all.

    Notes
    -----
    The correlation that governs a test's PPI variance reduction is a Pearson
    correlation on a TEST-SPECIFIC linearization of the raw values (identity
    for mean-type tests; a Hajek projection, empirical placements or centered
    ranks for rank-type ones). It is NOT simply "Pearson for mean-based tests,
    Spearman for rank-based ones": for every rank-based test that split drifts
    with effect size, overstating N_eff by -13% to +89% in simulation (see
    :data:`_TEST_STRUCTURE` and notes/omnibus_label_efficiency.html). That is
    why ``comparison`` is separate from the raw Pearson/Spearman metrics, which
    are reported for comparability to prior work.
    """
    if isinstance(judge_scores_or_evaldata, dict):
        raise TypeError(
            "judge_alignment() no longer takes a {condition: (judge, human)} dict. Pass the "
            "same evaldata you give compare() (from load_from), with llm_metric= and "
            "human_groundtruth=; the conditions, the design and the tests are read from it."
        )
    from evalstats.loader import EvalResults

    if isinstance(judge_scores_or_evaldata, EvalResults):
        evaldata = judge_scores_or_evaldata
        if human_scores is not None:
            raise TypeError(
                "judge_alignment(evaldata, ...) doesn't take a second "
                "positional argument; pass llm_metric= and "
                "human_groundtruth= as column names instead. (For the "
                "raw-array form, pass two arrays: "
                "judge_alignment(judge_scores, human_scores).)"
            )
        if llm_metric is None or human_groundtruth is None:
            raise TypeError(
                "judge_alignment(evaldata, ...) requires llm_metric= and "
                "human_groundtruth= (column names)."
            )
        return _judge_alignment_from_evaldata(
            evaldata, llm_metric=llm_metric, human_groundtruth=human_groundtruth, alpha=alpha,
            selection=selection, ci=ci, factors=factors, score_type=score_type,
            n_total=n_total, design=design, test=test,
        )

    if human_scores is None:
        raise TypeError(
            "judge_alignment(judge_scores, human_scores) requires both "
            "arrays; or pass an EvalResults (from load_from()) as the "
            "first argument for the column-name-based form: "
            "judge_alignment(evaldata, llm_metric=..., human_groundtruth=...)."
        )
    return _judge_alignment_from_arrays(
        judge_scores_or_evaldata, human_scores,
        all_judge_scores=all_judge_scores, score_type=score_type,
        llm_metric=llm_metric, human_groundtruth=human_groundtruth, alpha=alpha,
        selection=selection, test=test, ci=ci, n_total=n_total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Label-efficiency savings formula, shared by AlignmentResult.n_eff/
# .multiplier and PairwiseAlignmentResult's per-metric multiplier/n_eff.
# ─────────────────────────────────────────────────────────────────────────────

def _n_eff(r: float, n_lab: int, N: int) -> tuple[float, float]:
    """(multiplier, N_eff) from the control-variate savings formula
    ``1 / (1 - rho^2 * (1 - N_lab/N))`` -- see
    simulations/harness/cases/pvalues.py's ``_ppi_predicted_savings`` for
    the derivation and validation (R^2=0.9968 against measured variance
    ratios over a 48-cell grid at 3000 reps/cell)."""
    if not np.isfinite(r) or N <= 0:
        return float("nan"), float("nan")
    rho2 = float(np.clip(r, -1.0, 1.0)) ** 2
    k = max(0.0, 1.0 - float(n_lab) / float(N))
    denom = 1.0 - rho2 * k
    mult = 1.0 / denom if denom > 1e-9 else float("inf")
    return mult, n_lab * mult


# ── Label efficiency for a comparison that has already run ─────────────────
#
# compare() knows which tests it ran; these turn that knowledge into the
# rho^2/N_eff a reader needs beside each estimate. Kept here, next to
# judge_alignment, because they are thin wrappers over it rather than new
# statistics -- and shared so the paired and unpaired paths cannot drift into
# reporting the same quantity two different ways.


def _efficiency_metric(conds, *, test, design, want_pairs):
    """One judge_alignment call, reduced to what a summary table needs.

    Returns ``(omnibus, pairs)`` where omnibus is ``(rho2, n_eff_total)`` or
    None, and pairs maps ``(a, b) -> (rho2, n_eff_total)``. n_eff is left as
    the TOTAL judge_alignment returns; dividing it by the conditions a given
    correlation spans is the caller's job, since only the caller knows whether
    a row is an omnibus (k conditions), a pair (2), or a single mean (1).

    Never raises: this is reporting, and a failure must cost a column rather
    than the comparison that produced it.
    """
    import contextlib as _c, io as _io
    try:
        with _c.redirect_stdout(_io.StringIO()):
            res = _judge_alignment_pairwise(conds, design=design, alpha=0.05, test=test,
                                            selection="random", ci=False, include_raw=False)
    except Exception:
        return None, {}
    om = None
    m = getattr(res, "omnibus_metric", None)
    if m is not None:
        try:
            om = (float(m["estimate"]) ** 2, float(m["n_eff"]))
        except Exception:
            om = None
    pairs = {}
    if want_pairs:
        for key, mm in (getattr(res, "test_pairwise_metrics", None) or {}).items():
            try:
                pairs[(str(key[0]), str(key[1]))] = (float(mm["estimate"]) ** 2,
                                                     float(mm["n_eff"]))
            except Exception:
                pass
    return om, pairs


def _marginal_efficiency(judge, human):
    """rho^2 and N_eff for ONE entity's marginal mean.

    The estimand is a plain mean, whose influence function is the identity, so
    this is exactly Pearson r^2 on the labeled pairs (verified bit-identical
    against scipy). Routed through judge_alignment anyway so the number a user
    sees here is produced by the same code path as every other rho^2 we report.

    n_eff comes back against this entity's own item count, so it needs no
    division: one condition spans itself.
    """
    import contextlib as _c, io as _io
    try:
        with _c.redirect_stdout(_io.StringIO()):
            res = judge_alignment(np.asarray(judge, dtype=float),
                                  np.asarray(human, dtype=float),
                                  test="mean_estimate", selection="random", ci=False)
        return float(res.test_metric["estimate"]) ** 2, float(res.n_eff)
    except Exception:
        return None, None
