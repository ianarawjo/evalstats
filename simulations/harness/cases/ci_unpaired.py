"""ci_unpaired case: between-subjects (unpaired) pairwise CI coverage.

The missing validation behind ``compare(design="unpaired")``. ``ci_paired.py``
answers "which CI method for a paired difference?"; nothing yet answers it for
two *independent* groups, which is why the paper currently has to disclaim
pairwise CIs for between-subjects designs.

What the shipped code does today (evalstats/config.py's
AUTO_UNPAIRED_METHOD_TABLE, evalstats/core/unpaired.py):

  binary                  -> Welch's t-interval on the raw 0/1 values, i.e.
                             a linear-probability-model patch, explicitly
                             flagged in config.py as "a deliberate patch, not
                             a clean solution".
  continuous/likert/grade -> percentile bootstrap on theta_ab, the
                             stochastic-dominance probability from the
                             Mann-Whitney / Kruskal-Wallis path -- NOT a mean
                             difference.

So there are two distinct estimands in play, and a validation that only
covered one of them would be validating a method the library does not use:

  mean_diff : mean(A) - mean(B)  (a.k.a. Delta-p on binary)
  theta     : P(A > B) + 0.5 * P(A = B)

Every row this case emits is tagged with its estimand, and the two are never
pooled -- their coverage targets are different quantities.

Data generation
---------------
Reuses ``scenarios.CIPairSource`` unchanged, but *breaks the pairing*: arm A
comes from one ``generate_pair`` call and arm B from a second, independent
one, so the two groups share no items. Two consequences worth stating
explicitly:

  1. ``source.true_diff`` remains exactly correct. It is
     E[mean(a)] - E[mean(b)], a difference of marginals, and expectation is
     linear -- whether a and b were drawn jointly or independently does not
     change it. This holds for the hand-built 2x2 binary scenarios too
     (true_diff = p10 - p01 = p_A - p_B).
  2. ``true_theta`` does NOT follow from true_diff and has to be estimated by
     Monte Carlo per source (``_estimate_true_theta``), from independent
     draws, with ties handled at 0.5.

Unequal group sizes are the norm in between-subjects work, so ``--size-ratios``
sweeps n_B / n_A (default 1.0; e.g. ``--size-ratios 1.0 2.0``).

Method slate
------------
Deliberately a *starting* slate, not a settled recommendation -- picking the
right candidates from the two-independent-proportions and stochastic-dominance
literatures is the open question this case exists to answer. Currently:

  mean_diff, all eval types : bootstrap, bca, bayes_bootstrap,
                              smooth_bootstrap, bootstrap_t (all two-sample
                              forms), welch_t, student_t
  mean_diff, binary only    : wald_unpaired (naive baseline), agresti_caffo,
                              newcombe_hybrid, miettinen_nurminen,
                              bayes_beta_indep
  theta, all eval types     : theta_bootstrap (the shipped behavior),
                              theta_bca, brunner_munzel, brunner_munzel_logit

Known limitations of this first pass (deliberate, to keep the engine small):
synthetic sources only (no real-data corpora), runs=1 only (no multi-run /
nested variants), statistic=mean only.

Run:
    python -m simulations.harness.cli ci_unpaired --quick
"""
from __future__ import annotations

import argparse
import csv
import io
import multiprocessing as _mp
import os
import time
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import scipy.stats as stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evalstats.core.resampling import (
        bootstrap_means_1d,
        bayes_bootstrap_means_1d,
        smooth_bootstrap_means_1d,
        logit_t_ci_1d,
        nig_ci_1d,
        t_interval_ci_1d,
    )
    from evalstats.core.stats_utils import interval_score, rescaled_ci

from ..scenarios import CIPairSource, EVAL_TYPES, DEFAULT_EVAL_TYPES, EVAL_TYPE_SCALE_BOUNDS
from ..scenarios.synthetic import SCENARIO_SUITES, build_pair_sources
from ..scenarios.real_unpaired import (
    REAL_UNPAIRED_DATASETS, DEFAULT_DATA_DIR as REAL_DEFAULT_DATA_DIR,
    build_real_unpaired_sources,
)
from ..latex_tables import (
    booktabs_table, coverage_cell, escape_latex,
    mark_best_and_runnerup, report_eval_type_group,
)
from ..methods import (
    BOOTSTRAP_METHODS,
    BOOTSTRAP, BCA, BAYES_BOOTSTRAP, SMOOTH_BOOTSTRAP, BOOTSTRAP_T,
    WELCH_T, STUDENT_T, MOVER_T, MOVER_LOGIT_T, MOVER_NIG,
    WALD_UNPAIRED, AGRESTI_CAFFO, NEWCOMBE_HYBRID, MIETTINEN_NURMINEN, BAYES_BETA_INDEP,
    AGRESTI_MIN,
    THETA_BOOTSTRAP, THETA_BCA, BRUNNER_MUNZEL, BRUNNER_MUNZEL_LOGIT,
    UNPAIRED_MEAN_EXTRA_METHODS, UNPAIRED_BINARY_METHODS, UNPAIRED_THETA_METHODS,
    order_present_methods,
)
from . import CaseResult

CASE_NAME = "ci_unpaired"

DATA_SOURCES = ["synthetic", "real"]
ESTIMANDS = ["mean_diff", "theta", "both"]
PROGRESS_MODES = ["bar", "cell", "off"]
PLOT_MODES = ["save", "off"]
RESULTS_MODES = ["save", "off"]

MEAN_ESTIMAND = "mean_diff"
THETA_ESTIMAND = "theta"

_BINARY_INELIGIBLE = (MOVER_NIG,)
"""MOVER arms that are not valid on binary data, so they are not run there --
matching ci_paired, which gates its own nig off binary for the same reason.

Not a convention but a validity failure, verified directly on 0/1 samples at
n=10: NIG returns [0.852, 1.0571] for a 10/10 sample -- an upper limit above
1 for a proportion.

mover_logit_t is deliberately NOT gated: the logit transform respects [0, 1]
by construction, and on binary it is the strongest method in the exact table
(0.954 minimum coverage, holding nominal), so excluding it would discard a
real result rather than avoid a spurious one."""

_THETA_NULL_VALUE = 0.5
"""theta's "no difference" point -- the analogue of 0 for a mean difference.
Used for the Type I error / power counter, not for coverage."""


@dataclass
class SimResult:
    source: str
    label: str
    eval_type: str
    estimand: str  # "mean_diff" | "theta"
    n_a: int
    n_b: int
    method: str
    n_reps: int
    covered: int
    total_width: float
    total_score: float = 0.0
    total_pen_under: float = 0.0
    total_pen_over: float = 0.0
    rejects: int = 0
    """Reps whose CI excluded the estimand's null value (0 for mean_diff, 0.5
    for theta): Type I error on is_null rows, power elsewhere."""
    total_time: float = 0.0
    total_time_sq: float = 0.0
    is_null: bool = False
    true_value: float = 0.0
    icc: float = 0.0
    cohens_d: float = 0.0


# ---------------------------------------------------------------------------
# Delta-mean / Delta-p intervals for two INDEPENDENT samples
# ---------------------------------------------------------------------------
# Local to this case on purpose: none of these are shipped by evalstats today,
# and which of them (if any) should be is exactly what this sweep is for.
# Promote the winners into evalstats.core.resampling once the slate settles.


def _welch_t_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Welch's unequal-variance t-interval on mean(a) - mean(b).

    This is the shipped behavior for binary score types
    (evalstats.core.unpaired._binary_pairwise_uncorrected), reached via
    scipy's ttest_ind(equal_var=False).confidence_interval.
    """
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        d = float(np.mean(a) - np.mean(b))
        return d, d
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se2 = va / na + vb / nb
    d = float(np.mean(a) - np.mean(b))
    if se2 <= 0.0:
        # Both arms constant -- no variance to estimate from, so a
        # variance-based interval has nothing to say and collapses to a
        # point. NOT a harness bug: on binary data at small n and extreme p
        # this is common and real (measured: 17% of draws at n=5, 4.7% at
        # n=10, 1.25% at n=20, none by n=30), and it is a genuine part of
        # why the t-based methods bottom out near 0.64 exact coverage while
        # the dedicated binary intervals -- which handle a degenerate sample
        # rather than dividing by its variance -- never do this at all.
        return d, d
    se = float(np.sqrt(se2))
    # Welch-Satterthwaite degrees of freedom.
    df = se2**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return d - t * se, d + t * se


def _student_t_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Pooled-variance (equal-variance) two-sample t-interval."""
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        d = float(np.mean(a) - np.mean(b))
        return d, d
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    df = na + nb - 2
    sp2 = ((na - 1) * va + (nb - 1) * vb) / df
    se = float(np.sqrt(sp2 * (1.0 / na + 1.0 / nb)))
    d = float(np.mean(a) - np.mean(b))
    if se <= 0.0:
        return d, d
    t = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return d - t * se, d + t * se


def _wald_unpaired_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Naive normal-approximation Wald interval for p_A - p_B.

    The textbook bad baseline for two independent proportions: it degenerates
    to zero width when either arm is at 0 or 1, and under-covers badly at
    small n or extreme p. Kept for the same reason ci_single keeps `wald` --
    a floor to measure the real methods against.
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    pa, pb = float(np.mean(a_bin)), float(np.mean(b_bin))
    se = float(np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    d = pa - pb
    return max(-1.0, d - z * se), min(1.0, d + z * se)


def _agresti_caffo_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Agresti & Caffo (2000), "Simple and effective confidence intervals for
    proportions and differences of proportions result from adding two
    successes and two failures", The American Statistician 54(4):280-288.

    Add one success and one failure to EACH arm, then apply the plain Wald
    formula to the adjusted counts. Motivated as the two-sample analogue of
    the Agresti-Coull single-proportion adjustment; the appeal is that it is
    a one-line change to Wald that fixes most of Wald's small-sample
    undercoverage.
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    pa = (float(np.sum(a_bin)) + 1.0) / (na + 2.0)
    pb = (float(np.sum(b_bin)) + 1.0) / (nb + 2.0)
    se = float(np.sqrt(pa * (1 - pa) / (na + 2.0) + pb * (1 - pb) / (nb + 2.0)))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    d = pa - pb
    return max(-1.0, d - z * se), min(1.0, d + z * se)


def _wilson_bounds(successes: float, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score interval for a single proportion (helper for MOVER)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt(p * (1 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, float(centre - radius)), min(1.0, float(centre + radius))


def _mover_combine(
    ta: float, tb: float, arm_a: tuple[float, float], arm_b: tuple[float, float],
) -> tuple[float, float]:
    """MOVER ("method of variance estimates recovery") combination of two
    independent one-sample intervals into an interval for their difference.

    Zou & Donner (2008), "Construction of confidence limits about effect
    measures: a general approach", Statistics in Medicine 27(10):1693-1702.
    Given point estimates ta, tb and separate intervals (la, ua), (lb, ub):

        lower = (ta - tb) - sqrt( (ta - la)^2 + (ub - tb)^2 )
        upper = (ta - tb) + sqrt( (ua - ta)^2 + (tb - lb)^2 )

    Each arm contributes its variance estimate *at the tail that matters*
    for that endpoint, rather than a single symmetric SE at the point
    estimate -- which is what lets a skewed or boundary-respecting one-sample
    interval carry its good behavior through to the difference.

    This is the general form of Newcombe's hybrid score interval (which is
    exactly this with Wilson arms), so it lets the unpaired path reuse the
    very same one-sample methods the paired path already recommends.
    """
    d = ta - tb
    lo = d - float(np.sqrt((ta - arm_a[0]) ** 2 + (arm_b[1] - tb) ** 2))
    hi = d + float(np.sqrt((arm_a[1] - ta) ** 2 + (tb - arm_b[0]) ** 2))
    return lo, hi


def _newcombe_hybrid_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Newcombe (1998) method 10, the "hybrid score" / square-and-add
    interval, Statistics in Medicine 17(8):873-890 ("Interval estimation for
    the difference between independent proportions: comparison of eleven
    methods").

    Build a Wilson interval (l_i, u_i) separately for each arm, then combine:

        lower = (pa - pb) - sqrt( (pa - la)^2 + (ub - pb)^2 )
        upper = (pa - pb) + sqrt( (ua - pa)^2 + (pb - lb)^2 )

    This is the MOVER (method of variance estimates recovery) construction:
    it takes each arm's variance estimate *at the relevant tail* rather than
    at the point estimate, which is what keeps it honest as either arm
    approaches a boundary. Same family as the paired ``newcombe_mover``
    already in the harness.
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    ka, kb = float(np.sum(a_bin)), float(np.sum(b_bin))
    pa, pb = ka / na, kb / nb
    lo, hi = _mover_combine(pa, pb, _wilson_bounds(ka, na, alpha), _wilson_bounds(kb, nb, alpha))
    return max(-1.0, lo), min(1.0, hi)


def _mover_one_sample_ci(
    values: np.ndarray, alpha: float, fn, bounds: tuple[float, float],
) -> tuple[float, float]:
    """One arm's mean CI via a shipped [0, 1]-domain method, rescaled onto
    the eval type's own scale.

    Note this is a *cleaner* use of these methods than the paired path gets.
    ci_paired has to rescale onto [-span, span] so a zero difference lands at
    the methods' own centre, which is what forced its b0/4 correction to
    nig's prior (see ci_paired._NIG_PAIRED_DIFF_B0). Here each arm is a plain
    mean on the original [lo, hi] scale -- exactly ci_single's usage -- so
    the shipped priors apply unmodified.
    """
    return rescaled_ci(fn, values, alpha, bounds[0], bounds[1])


def _mover_t_ci(
    a: np.ndarray, b: np.ndarray, alpha: float, bounds: tuple[float, float],
) -> tuple[float, float]:
    """MOVER with plain t-interval arms -- the CONTROL for the MOVER family.

    Holds the arm interval fixed at an ordinary t-interval and varies only
    the combination rule, so comparing this against welch_t isolates what
    MOVER's square-and-add buys, and comparing mover_logit_t against THIS
    isolates what the logit arm buys. Without it, mover_logit_t vs welch_t
    confounds the two changes.

    ``bounds`` is accepted and unused (a t-interval needs no rescaling);
    it keeps the signature uniform across the MOVER family.
    """
    ci_a = t_interval_ci_1d(a, alpha)
    ci_b = t_interval_ci_1d(b, alpha)
    return _mover_combine(float(np.mean(a)), float(np.mean(b)), ci_a, ci_b)


def _mover_logit_t_ci(
    a: np.ndarray, b: np.ndarray, alpha: float, bounds: tuple[float, float],
) -> tuple[float, float]:
    """MOVER with logit-t arms -- the unpaired sibling of the paired path's
    ``logit_t`` recommendation for bounded_01 data."""
    ci_a = _mover_one_sample_ci(a, alpha, logit_t_ci_1d, bounds)
    ci_b = _mover_one_sample_ci(b, alpha, logit_t_ci_1d, bounds)
    return _mover_combine(float(np.mean(a)), float(np.mean(b)), ci_a, ci_b)


def _mover_nig_ci(
    a: np.ndarray, b: np.ndarray, alpha: float, bounds: tuple[float, float],
) -> tuple[float, float]:
    """MOVER with NIG arms -- the unpaired sibling of the paired path's
    ``nig`` recommendation for likert data."""
    ci_a = _mover_one_sample_ci(a, alpha, nig_ci_1d, bounds)
    ci_b = _mover_one_sample_ci(b, alpha, nig_ci_1d, bounds)
    return _mover_combine(float(np.mean(a)), float(np.mean(b)), ci_a, ci_b)


def _fm_constrained_mle(pa: float, pb: float, na: int, nb: int, delta: float) -> tuple[float, float]:
    """Farrington & Manning (1990) closed-form constrained MLEs of (p_A, p_B)
    subject to p_A - p_B = delta, used by the Miettinen-Nurminen score
    interval. Statistics in Medicine 9(12):1447-1454.

    Solves the cubic that the constrained likelihood equations reduce to. The
    trigonometric branch below is the standard three-real-roots form; the
    guarded fallbacks handle the degenerate cases (u == 0, |v/u^3| > 1) that
    arise at the boundaries, where the cubic has a repeated root.
    """
    theta = nb / na
    aa = 1.0 + theta
    bb = -(1.0 + theta + pa + theta * pb + delta * (theta + 2.0))
    cc = delta * delta + delta * (2.0 * pa + theta + 1.0) + pa + theta * pb
    dd = -pa * delta * (1.0 + delta)

    v = bb**3 / (27.0 * aa**3) - bb * cc / (6.0 * aa**2) + dd / (2.0 * aa)
    inner = bb**2 / (9.0 * aa**2) - cc / (3.0 * aa)
    if inner <= 0.0:
        p1 = float(np.clip(pa, 0.0, 1.0))
        return p1, float(np.clip(p1 - delta, 0.0, 1.0))
    u = float(np.sign(v) if v != 0 else 1.0) * np.sqrt(inner)
    if u == 0.0:
        p1 = float(np.clip(-bb / (3.0 * aa), 0.0, 1.0))
        return p1, float(np.clip(p1 - delta, 0.0, 1.0))
    ratio = float(np.clip(v / u**3, -1.0, 1.0))
    w = (np.pi + np.arccos(ratio)) / 3.0
    p1 = float(np.clip(2.0 * u * np.cos(w) - bb / (3.0 * aa), 0.0, 1.0))
    p2 = float(np.clip(p1 - delta, 0.0, 1.0))
    return p1, p2


def _miettinen_nurminen_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Miettinen & Nurminen (1985) score interval for p_A - p_B, Statistics in
    Medicine 4(2):213-226.

    Inverts the score test: the interval is the set of delta for which the
    score statistic

        Z(delta) = (pa - pb - delta) / sqrt( V(delta) )
        V(delta) = ( p1~(1-p1~)/na + p2~(1-p2~)/nb ) * N/(N-1)

    (with p1~, p2~ the constrained MLEs at that delta, and the N/(N-1) term
    Miettinen-Nurminen's small-sample variance correction) satisfies
    |Z| <= z_{1-alpha/2}. Z is monotone decreasing in delta, so each endpoint
    is a single root, found here by bisection rather than the closed-form
    quartic -- slower per call but far harder to get subtly wrong, and the
    per-call cost is still negligible next to any bootstrap.

    Widely reported as the best-performing interval in the two-independent-
    proportions comparisons (it is what statsmodels calls method="score").
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    pa, pb = float(np.mean(a_bin)), float(np.mean(b_bin))
    n_tot = na + nb
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    d_hat = pa - pb

    def _score(delta: float) -> float:
        p1, p2 = _fm_constrained_mle(pa, pb, na, nb, delta)
        var = (p1 * (1 - p1) / na + p2 * (1 - p2) / nb) * (n_tot / (n_tot - 1.0))
        if var <= 1e-14:
            # Degenerate constrained variance -- both arms pinned to a
            # boundary, e.g. 0/8 vs 8/8 evaluated at delta = -1. The score is
            # 0/0 exactly at delta = d_hat, and that case must return 0 (not
            # +-inf): d_hat is always inside its own interval, and reporting
            # +-inf there collapses the interval to the single point d_hat.
            # Away from d_hat the numerator is nonzero over a vanishing
            # variance, so delta really is decisively excluded.
            if abs(d_hat - delta) <= 1e-12:
                return 0.0
            return float(np.inf) if d_hat > delta else float(-np.inf)
        return (d_hat - delta) / float(np.sqrt(var))

    def _bisect(target: float, lo: float, hi: float) -> float:
        # _score is decreasing in delta; find where it crosses `target`.
        f_lo, f_hi = _score(lo) - target, _score(hi) - target
        if f_lo <= 0:
            return lo
        if f_hi >= 0:
            return hi
        # Tolerance-based rather than a fixed iteration count: the bracket
        # starts at width <= 2, so 1e-11 is reached in ~38 halvings, and
        # capping at 60 bounds the worst case. (A flat 80 iterations cost
        # ~1 ms/call, which made this the most expensive method in the
        # sweep by 20x for precision nobody uses.)
        for _ in range(60):
            if hi - lo < 1e-11:
                break
            mid = 0.5 * (lo + hi)
            if _score(mid) - target > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    eps = 1e-9
    lower = _bisect(z, -1.0 + eps, d_hat)
    upper = _bisect(-z, d_hat, 1.0 - eps)
    return max(-1.0, lower), min(1.0, upper)


def _fm_constrained_mle_vec(
    pa: np.ndarray, pb: np.ndarray, na: int, nb: int, delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``_fm_constrained_mle`` over arrays of observed proportions.

    Same cubic, same branch; exists because Agresti-Min needs the constrained
    MLE at every one of the (na+1)*(nb+1) possible tables for each candidate
    delta, and doing that with the scalar version is ~100x slower.
    """
    theta = nb / na
    aa = 1.0 + theta
    bb = -(1.0 + theta + pa + theta * pb + delta * (theta + 2.0))
    cc = delta * delta + delta * (2.0 * pa + theta + 1.0) + pa + theta * pb
    dd = -pa * delta * (1.0 + delta)

    v = bb**3 / (27.0 * aa**3) - bb * cc / (6.0 * aa**2) + dd / (2.0 * aa)
    inner = bb**2 / (9.0 * aa**2) - cc / (3.0 * aa)
    inner_safe = np.where(inner > 0, inner, 1.0)
    sign = np.where(v != 0, np.sign(v), 1.0)
    u = sign * np.sqrt(inner_safe)
    u_safe = np.where(np.abs(u) < 1e-300, 1.0, u)
    ratio = np.clip(v / u_safe**3, -1.0, 1.0)
    w = (np.pi + np.arccos(ratio)) / 3.0
    p1 = 2.0 * u * np.cos(w) - bb / (3.0 * aa)
    # Degenerate branches, matching the scalar version's fallbacks.
    p1 = np.where(inner > 0, p1, pa)
    p1 = np.where(np.abs(u) < 1e-300, -bb / (3.0 * aa), p1)
    p1 = np.clip(p1, 0.0, 1.0)
    return p1, np.clip(p1 - delta, 0.0, 1.0)


def _mn_score_all_tables(na: int, nb: int, delta: float) -> np.ndarray:
    """|Miettinen-Nurminen score statistic| at ``delta`` for every possible
    2x2 table, as an (na+1, nb+1) array indexed by (successes in A,
    successes in B). The test statistic Agresti-Min inverts."""
    ka = np.arange(na + 1)[:, None] / na
    kb = np.arange(nb + 1)[None, :] / nb
    pa = np.broadcast_to(ka, (na + 1, nb + 1))
    pb = np.broadcast_to(kb, (na + 1, nb + 1))
    p1, p2 = _fm_constrained_mle_vec(pa, pb, na, nb, delta)
    n_tot = na + nb
    var = (p1 * (1 - p1) / na + p2 * (1 - p2) / nb) * (n_tot / (n_tot - 1.0))
    num = pa - pb - delta
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 1e-14, num / np.sqrt(np.where(var > 1e-14, var, 1.0)), 0.0)
    # var == 0 happens only when both constrained MLEs sit on a boundary; the
    # table is then either exactly consistent with delta (num == 0, Z = 0) or
    # impossible under it (Z infinite).
    z = np.where((var <= 1e-14) & (np.abs(num) > 1e-12), np.inf, z)
    return np.abs(z)


_AGRESTI_MIN_GRID = 80
_AGRESTI_MIN_SCAN = 161
"""Delta values scanned to locate the OUTERMOST crossing of R = alpha before
refining. R is not monotone in delta, so a bracket has to be found by scan
rather than assumed; 161 points is a step of 0.0125, which resolves the
~0.02-wide discrepancies plain bisection was observed to miss. A gap in
{delta : R > alpha} narrower than one step could still be stepped over --
the interval would then be slightly short, so this is approximately rather
than provably exact."""
_AGRESTI_MIN_GAMMA = 1e-4
"""Berger-Boos confidence level for restricting the nuisance-parameter sup.
Berger & Boos (1994) show that taking the supremum over a 100(1-gamma)%
confidence set for the nuisance parameter and then adding gamma back keeps
the test valid while removing the far-tail nuisance values that otherwise
dominate the sup and make the interval needlessly wide."""


def _agresti_min_ci(
    a: np.ndarray, b: np.ndarray, alpha: float,
    n_grid: int = _AGRESTI_MIN_GRID, gamma: float = _AGRESTI_MIN_GAMMA,
) -> tuple[float, float]:
    """Agresti & Min (2001) exact unconditional confidence interval for
    p_A - p_B, inverting the Miettinen-Nurminen score test.

    Agresti & Min, "On small-sample confidence intervals for parameters in
    discrete distributions", Biometrics 57(3):963-971. This is Fagerland,
    Lydersen & Laake's *prime recommendation for small samples* (their
    Section 8.1 / Table 7): unlike the asymptotic score intervals it never
    dips below nominal coverage, and it behaves better than they do when a
    proportion sits near 0 or 1.

    For a candidate delta, the exact unconditional p-value is

        R(delta) = sup_{p1} P( |Z(X | delta)| >= |Z(x_obs | delta)| )

    where X ranges over all (na+1)(nb+1) possible tables, each weighted by
    Binom(na, p1) x Binom(nb, p1 - delta), and the supremum eliminates the
    nuisance parameter p1. The interval is {delta : R(delta) > alpha}.

    Two practical notes, both standard:

    * The sup is taken over a Berger-Boos 100(1-gamma)% confidence set for
      p1 rather than all of [0, 1], with gamma added back to R.
    * R is not guaranteed monotone in delta, so bisecting each side of the
      point estimate (as here, and as the standard implementations do) finds
      *a* crossing rather than provably the outermost one. In exchange the
      whole interval costs ~40 evaluations instead of a full grid sweep.

    This is by far the most expensive method in the slate -- roughly 100x a
    Welch interval -- because each evaluation rebuilds the score statistic
    for every possible table. That cost is the reason it is not in most
    software (Fagerland et al.'s Table 8 lists it only in StatXact).
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    ka, kb = int(np.sum(a_bin)), int(np.sum(b_bin))
    d_hat = ka / na - kb / nb

    # Berger-Boos confidence set for the nuisance parameter p1, from arm A.
    if gamma > 0:
        bb_lo = stats.beta.ppf(gamma / 2, ka, na - ka + 1) if ka > 0 else 0.0
        bb_hi = stats.beta.ppf(1 - gamma / 2, ka + 1, na - ka) if ka < na else 1.0
        bb_lo = 0.0 if not np.isfinite(bb_lo) else float(bb_lo)
        bb_hi = 1.0 if not np.isfinite(bb_hi) else float(bb_hi)
    else:
        bb_lo, bb_hi = 0.0, 1.0

    def _r(delta: float) -> float:
        z_tab = _mn_score_all_tables(na, nb, delta)
        z_obs = z_tab[ka, kb]
        if not np.isfinite(z_obs):
            return 0.0
        mask = (z_tab >= z_obs - 1e-9).astype(float)
        lo_p1 = max(bb_lo, max(0.0, delta))
        hi_p1 = min(bb_hi, min(1.0, 1.0 + delta))
        if hi_p1 < lo_p1:
            return 0.0
        p1s = np.linspace(lo_p1, hi_p1, n_grid)
        p2s = np.clip(p1s - delta, 0.0, 1.0)
        pmf_a = stats.binom.pmf(np.arange(na + 1)[:, None], na, p1s[None, :])
        pmf_b = stats.binom.pmf(np.arange(nb + 1)[:, None], nb, p2s[None, :])
        probs = np.einsum("ig,ij,jg->g", pmf_a, mask, pmf_b)
        return float(np.max(probs)) + gamma

    def _refine(inside: float, outside: float) -> float:
        """Bisect a bracket that is already known to straddle the OUTERMOST
        crossing. Returns ``outside``, the last delta known to be excluded,
        so any residual search error widens the interval rather than
        narrowing it."""
        for _ in range(40):
            if abs(outside - inside) < 1e-6:
                break
            mid = 0.5 * (inside + outside)
            if _r(mid) > alpha:
                inside = mid
            else:
                outside = mid
        return outside

    # R(delta) is NOT monotone, so bisecting straight out from d_hat finds
    # *a* crossing rather than the outermost one, and returns an interval
    # that is too short. Measured directly: at n_A=n_B=20 plain bisection
    # under-covered (exact minimum 0.9490 < 0.95) because on tables like
    # k=(20,0) and k=(18,1) it stopped 0.02 short of the true limit -- and
    # those are exactly the tables carrying the mass at the worst (p_A, p_B).
    # So scan a grid first, take the outermost delta that is still inside,
    # and only then refine within that one cell.
    eps = 1e-9
    grid = np.linspace(-1.0 + eps, 1.0 - eps, _AGRESTI_MIN_SCAN)
    inside_flags = np.array([_r(d) > alpha for d in grid])
    idx = np.flatnonzero(inside_flags)
    if idx.size == 0:
        return d_hat, d_hat  # degenerate; d_hat is always inside in exact arithmetic
    i_lo, i_hi = int(idx[0]), int(idx[-1])
    lower = -1.0 if i_lo == 0 else _refine(float(grid[i_lo]), float(grid[i_lo - 1]))
    upper = 1.0 if i_hi == grid.size - 1 else _refine(float(grid[i_hi]), float(grid[i_hi + 1]))
    return max(-1.0, min(lower, d_hat)), min(1.0, max(upper, d_hat))


def _bayes_beta_indep_ci(
    a: np.ndarray, b: np.ndarray, alpha: float, num_samples: int, rng: np.random.Generator,
) -> tuple[float, float]:
    """Independent Jeffreys Beta(1/2, 1/2) posteriors on p_A and p_B, sampled
    and subtracted.

    Legitimate here in a way it is not for paired data: the two groups really
    are independent, so the independence the model assumes is a fact of the
    design rather than an error. (Contrast ci_paired's ``bayes_indep_comp``,
    where the same construction is deliberately the *wrong* answer.)
    """
    a_bin = (np.asarray(a) >= 0.5).astype(float)
    b_bin = (np.asarray(b) >= 0.5).astype(float)
    na, nb = a_bin.size, b_bin.size
    if na == 0 or nb == 0:
        return (0.0, 0.0)
    ka, kb = float(np.sum(a_bin)), float(np.sum(b_bin))
    post_a = rng.beta(ka + 0.5, na - ka + 0.5, size=num_samples)
    post_b = rng.beta(kb + 0.5, nb - kb + 0.5, size=num_samples)
    diff = post_a - post_b
    return (float(np.percentile(diff, 100.0 * alpha / 2.0)),
            float(np.percentile(diff, 100.0 * (1.0 - alpha / 2.0))))


# ---------------------------------------------------------------------------
# Two-sample bootstrap family
# ---------------------------------------------------------------------------


def _two_sample_boot_diffs(
    a: np.ndarray, b: np.ndarray, method: str, n_boot: int, rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap distribution of mean(a) - mean(b) under independent resampling.

    Because the arms are independent, resampling each one separately and
    subtracting the resampled means is exactly the two-sample bootstrap -- so
    this delegates to the shipped single-sample resamplers rather than
    reimplementing them, and any fix to those propagates here.
    """
    if method == "bayes_bootstrap":
        fn = bayes_bootstrap_means_1d
    elif method == "smooth_bootstrap":
        fn = smooth_bootstrap_means_1d
    else:
        fn = bootstrap_means_1d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ma = fn(a, n_boot, rng, statistic="mean")
        mb = fn(b, n_boot, rng, statistic="mean")
    return np.asarray(ma) - np.asarray(mb)


def _two_sample_bca_ci(
    a: np.ndarray, b: np.ndarray, boot_diffs: np.ndarray, alpha: float,
) -> tuple[float, float]:
    """BCa for a two-sample difference of means.

    Bias correction z0 from the bootstrap distribution as usual; the
    acceleration comes from a jackknife that deletes one observation at a
    time from the *pooled* set of observations across both arms (deleting
    from A, then from B), which is the standard two-sample extension.
    """
    observed = float(np.mean(a) - np.mean(b))
    boot = np.asarray(boot_diffs, dtype=float)
    n_boot = boot.size
    prop = float(np.mean(boot < observed))
    if prop <= 0.0 or prop >= 1.0 or n_boot == 0:
        return (float(np.percentile(boot, 100 * alpha / 2)),
                float(np.percentile(boot, 100 * (1 - alpha / 2))))
    z0 = float(stats.norm.ppf(prop))

    na, nb = a.size, b.size
    sum_a, sum_b = float(np.sum(a)), float(np.sum(b))
    jack = np.empty(na + nb, dtype=float)
    if na > 1:
        jack[:na] = (sum_a - a) / (na - 1) - sum_b / nb
    else:
        jack[:na] = observed
    if nb > 1:
        jack[na:] = sum_a / na - (sum_b - b) / (nb - 1)
    else:
        jack[na:] = observed
    jack_mean = float(np.mean(jack))
    diffs = jack_mean - jack
    denom = 6.0 * (float(np.sum(diffs**2)) ** 1.5)
    acc = float(np.sum(diffs**3)) / denom if denom > 0 else 0.0

    def _adj(q: float) -> float:
        zq = float(stats.norm.ppf(q))
        num = z0 + zq
        den = 1.0 - acc * num
        if abs(den) < 1e-12:
            return q
        return float(stats.norm.cdf(z0 + num / den))

    lo_q = float(np.clip(_adj(alpha / 2), 1e-6, 1 - 1e-6))
    hi_q = float(np.clip(_adj(1 - alpha / 2), 1e-6, 1 - 1e-6))
    return float(np.percentile(boot, 100 * lo_q)), float(np.percentile(boot, 100 * hi_q))


def _two_sample_bootstrap_t_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator,
) -> tuple[float, float]:
    """Studentized (bootstrap-t) interval for a two-sample mean difference.

    Each bootstrap replicate is studentized by its OWN Welch standard error,
    so the resulting quantiles are of a pivotal quantity rather than of the
    raw difference -- the reason bootstrap-t usually beats the percentile
    bootstrap at small n.
    """
    na, nb = a.size, b.size
    observed = float(np.mean(a) - np.mean(b))
    if na < 2 or nb < 2:
        return observed, observed
    se_obs = float(np.sqrt(np.var(a, ddof=1) / na + np.var(b, ddof=1) / nb))
    if se_obs <= 0:
        return observed, observed

    idx_a = rng.integers(0, na, size=(n_boot, na))
    idx_b = rng.integers(0, nb, size=(n_boot, nb))
    sa, sb = a[idx_a], b[idx_b]
    ma, mb = sa.mean(axis=1), sb.mean(axis=1)
    va, vb = sa.var(axis=1, ddof=1), sb.var(axis=1, ddof=1)
    se_boot = np.sqrt(va / na + vb / nb)
    # Replicates with zero resampled variance carry no information about the
    # pivot's tails; drop them rather than letting them become +-inf.
    ok = se_boot > 0
    if not np.any(ok):
        return observed, observed
    t_stats = (ma[ok] - mb[ok] - observed) / se_boot[ok]
    t_lo = float(np.percentile(t_stats, 100 * (1 - alpha / 2)))
    t_hi = float(np.percentile(t_stats, 100 * (alpha / 2)))
    return observed - t_lo * se_obs, observed - t_hi * se_obs


# ---------------------------------------------------------------------------
# theta = P(A > B) + 0.5 * P(A = B)
# ---------------------------------------------------------------------------


def _theta_hat(a: np.ndarray, b: np.ndarray) -> float:
    """Point estimate of theta, ties counted at one half.

    O((na+nb) log nb) via binary search into a sorted copy of b, rather than
    the O(na*nb) pairwise comparison -- this is called once per bootstrap
    replicate, so the difference matters.
    """
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return 0.5
    bs = np.sort(b)
    wins = np.searchsorted(bs, a, side="left")
    wins_or_ties = np.searchsorted(bs, a, side="right")
    ties = wins_or_ties - wins
    return float((wins.sum() + 0.5 * ties.sum()) / (na * nb))


def _theta_bootstrap_stats(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap distribution of theta_hat -- resample each arm independently.

    Mirrors what evalstats.core.unpaired._rank_based_pairwise_uncorrected
    does for the shipped continuous/likert path.
    """
    na, nb = a.size, b.size
    out = np.empty(n_boot, dtype=float)
    idx_a = rng.integers(0, na, size=(n_boot, na))
    idx_b = rng.integers(0, nb, size=(n_boot, nb))
    for i in range(n_boot):
        out[i] = _theta_hat(a[idx_a[i]], b[idx_b[i]])
    return out


def _theta_bca_ci(
    a: np.ndarray, b: np.ndarray, boot_thetas: np.ndarray, alpha: float,
) -> tuple[float, float]:
    """BCa applied to theta_hat, with the same pooled jackknife as
    _two_sample_bca_ci."""
    observed = _theta_hat(a, b)
    boot = np.asarray(boot_thetas, dtype=float)
    prop = float(np.mean(boot < observed))
    if prop <= 0.0 or prop >= 1.0 or boot.size == 0:
        return (float(np.percentile(boot, 100 * alpha / 2)),
                float(np.percentile(boot, 100 * (1 - alpha / 2))))
    z0 = float(stats.norm.ppf(prop))

    na, nb = a.size, b.size
    jack = np.empty(na + nb, dtype=float)
    for i in range(na):
        jack[i] = _theta_hat(np.delete(a, i), b) if na > 1 else observed
    for j in range(nb):
        jack[na + j] = _theta_hat(a, np.delete(b, j)) if nb > 1 else observed
    diffs = float(np.mean(jack)) - jack
    denom = 6.0 * (float(np.sum(diffs**2)) ** 1.5)
    acc = float(np.sum(diffs**3)) / denom if denom > 0 else 0.0

    def _adj(q: float) -> float:
        zq = float(stats.norm.ppf(q))
        num = z0 + zq
        den = 1.0 - acc * num
        if abs(den) < 1e-12:
            return q
        return float(stats.norm.cdf(z0 + num / den))

    lo_q = float(np.clip(_adj(alpha / 2), 1e-6, 1 - 1e-6))
    hi_q = float(np.clip(_adj(1 - alpha / 2), 1e-6, 1 - 1e-6))
    return float(np.percentile(boot, 100 * lo_q)), float(np.percentile(boot, 100 * hi_q))


def _brunner_munzel_pieces(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return (theta_hat, se, df) for the Brunner-Munzel estimator of
    theta = P(A > B) + 0.5 P(A = B).

    Brunner & Munzel (2000), "The nonparametric Behrens-Fisher problem:
    asymptotic theory and a small-sample approximation", Biometrical Journal
    42(1):17-25. Unlike Mann-Whitney's usual variance, this does not assume
    the two groups share a distribution shape, which is the whole point --
    it is the rank analogue of Welch's t.

    df is the Satterthwaite-style approximation from the same paper.
    """
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return _theta_hat(a, b), float("nan"), float("nan")
    combined = np.concatenate([b, a])  # b first: theta is oriented as P(b < a)
    rank_c = stats.rankdata(combined)
    rc_b, rc_a = rank_c[:nb], rank_c[nb:]
    r_b, r_a = stats.rankdata(b), stats.rankdata(a)
    mb_c, ma_c = float(np.mean(rc_b)), float(np.mean(rc_a))

    theta = (ma_c - (na + 1) / 2.0) / nb
    s_b = float(np.sum((rc_b - r_b - mb_c + (nb + 1) / 2.0) ** 2)) / (nb - 1)
    s_a = float(np.sum((rc_a - r_a - ma_c + (na + 1) / 2.0) ** 2)) / (na - 1)
    pooled = nb * s_b + na * s_a
    if pooled <= 0:
        return theta, 0.0, float(na + nb - 2)
    se = float(np.sqrt(pooled)) / (na * nb)
    num = pooled**2
    den = (nb * s_b) ** 2 / (nb - 1) + (na * s_a) ** 2 / (na - 1)
    df = num / den if den > 0 else float(na + nb - 2)
    return theta, se, float(df)


def _brunner_munzel_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Brunner-Munzel interval for theta on the natural (untransformed) scale."""
    theta, se, df = _brunner_munzel_pieces(a, b)
    if not np.isfinite(se) or not np.isfinite(df) or se <= 0:
        return theta, theta
    t = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return max(0.0, theta - t * se), min(1.0, theta + t * se)


def _brunner_munzel_logit_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    """Brunner-Munzel on the logit scale, back-transformed.

    theta is a probability, so the untransformed interval can run past 0 or 1
    and its sampling distribution is skewed near the boundaries. Building the
    interval for logit(theta) (delta-method SE = se / (theta(1-theta))) and
    mapping back keeps it inside [0, 1] by construction and is the usual
    small-sample recommendation for this estimator.
    """
    theta, se, df = _brunner_munzel_pieces(a, b)
    if not np.isfinite(se) or not np.isfinite(df) or se <= 0:
        return theta, theta
    eps = 1e-6
    th = float(np.clip(theta, eps, 1 - eps))
    se_logit = se / (th * (1.0 - th))
    t = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    lg = np.log(th / (1.0 - th))
    lo, hi = lg - t * se_logit, lg + t * se_logit
    return float(1.0 / (1.0 + np.exp(-lo))), float(1.0 / (1.0 + np.exp(-hi)))


# ---------------------------------------------------------------------------
# True theta per source (Monte Carlo)
# ---------------------------------------------------------------------------

_TRUE_THETA_MC_N = 200_000


def _estimate_true_theta(source: CIPairSource, *, seed: int = 0, n_mc: int = _TRUE_THETA_MC_N) -> float:
    """The population theta for a source under INDEPENDENT arms.

    Unlike true_diff, theta is not a difference of marginals, so in general it
    cannot be read off the source -- it depends on the two marginal
    distributions jointly, and is estimated by Monte Carlo from two separate
    generate_pair calls (independent arms, matching how the simulation itself
    samples).

    Null sources are the exception and are returned EXACTLY. When both arms
    share a distribution, P(A>B) = P(B>A) by symmetry, so theta is exactly
    1/2. Estimating that by Monte Carlo instead was measurably wrong: over
    the null sources it missed 0.5 by up to 0.0011, and since coverage is
    scored against this number, that error goes straight into every theta
    method's coverage on every null row. Both source families qualify --
    synthetic d=0 draws both arms from one distribution, and
    real_unpaired's null sources draw both arms from one pool.
    """
    if source.is_null:
        return _THETA_NULL_VALUE
    rng = np.random.default_rng(seed)
    a, _ = source.generate_pair(rng, n_mc, 1)
    _, b = source.generate_pair(rng, n_mc, 1)
    return _theta_hat(a[:, 0], b[:, 0])


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


def _draw_unpaired(
    source: CIPairSource, rng: np.random.Generator, n_a: int, n_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw two INDEPENDENT groups from a paired source.

    Arm A is taken from one generate_pair call and arm B from a second,
    so no item is shared. Each arm keeps its own marginal distribution
    (including the effect shift on B), which is what makes source.true_diff
    still the right coverage target.
    """
    a, _ = source.generate_pair(rng, n_a, 1)
    _, b = source.generate_pair(rng, n_b, 1)
    return a[:, 0], b[:, 0]


def _run_cell(
    source: CIPairSource, n_a: int, n_b: int, n_reps: int, n_bootstrap: int, bayes_n: int,
    alpha: float, estimand: str, seed, true_theta: float,
    method_names: frozenset[str] | None = None,
) -> list[SimResult]:
    """Run all reps for one (source, n_a, n_b) cell."""
    rng = np.random.default_rng(seed)
    is_binary = source.eval_type == "binary"
    scale_bounds = source.scale_bounds or EVAL_TYPE_SCALE_BOUNDS[source.eval_type]

    def _want(m) -> bool:
        return method_names is None or m.name in method_names

    do_mean = estimand in (MEAN_ESTIMAND, "both")
    do_theta = estimand in (THETA_ESTIMAND, "both")

    mean_methods = []
    if do_mean:
        mean_methods += [m for m in BOOTSTRAP_METHODS if _want(m)]
        mean_methods += [m for m in UNPAIRED_MEAN_EXTRA_METHODS
                         if _want(m) and not (is_binary and m in _BINARY_INELIGIBLE)]
        if is_binary:
            mean_methods += [m for m in UNPAIRED_BINARY_METHODS if _want(m)]
    theta_methods = [m for m in UNPAIRED_THETA_METHODS if _want(m)] if do_theta else []

    all_methods = [(m, MEAN_ESTIMAND) for m in mean_methods] + [(m, THETA_ESTIMAND) for m in theta_methods]
    covered = {k: 0 for k in all_methods}
    total_w = {k: 0.0 for k in all_methods}
    total_score = {k: 0.0 for k in all_methods}
    pen_under = {k: 0.0 for k in all_methods}
    pen_over = {k: 0.0 for k in all_methods}
    rejects = {k: 0 for k in all_methods}
    total_t = {k: 0.0 for k in all_methods}
    total_t_sq = {k: 0.0 for k in all_methods}

    truth = {MEAN_ESTIMAND: float(source.true_diff), THETA_ESTIMAND: float(true_theta)}
    null_value = {MEAN_ESTIMAND: 0.0, THETA_ESTIMAND: _THETA_NULL_VALUE}

    def _record(key, lo: float, hi: float, elapsed: float) -> None:
        _, est = key
        target, null_v = truth[est], null_value[est]
        if lo <= target <= hi:
            covered[key] += 1
        total_w[key] += hi - lo
        total_score[key] += interval_score(lo, hi, target, alpha)
        if target < lo:
            pen_under[key] += (2.0 / alpha) * (lo - target)
        elif target > hi:
            pen_over[key] += (2.0 / alpha) * (target - hi)
        if lo > null_v or hi < null_v:
            rejects[key] += 1
        total_t[key] += elapsed
        total_t_sq[key] += elapsed * elapsed

    for _rep in range(n_reps):
        a, b = _draw_unpaired(source, rng, n_a, n_b)
        obs_mean = float(np.mean(a) - np.mean(b))

        # --- mean-difference family ---------------------------------------
        # bootstrap and bca share one set of resampled differences (they are
        # the same draws, read off differently). That shared draw is timed
        # once and its cost added to BOTH consumers -- charging it only to
        # whichever method happened to run first would make the other look
        # ~20x cheaper than it is, and the cost/coverage tradeoff is one of
        # the things this sweep is meant to inform.
        shared_boot: np.ndarray | None = None
        shared_boot_t = 0.0
        if any(m in (BOOTSTRAP, BCA) for m in mean_methods):
            _t0 = time.perf_counter()
            shared_boot = _two_sample_boot_diffs(a, b, "bootstrap", n_bootstrap, rng)
            shared_boot_t = time.perf_counter() - _t0

        for method in mean_methods:
            key = (method, MEAN_ESTIMAND)
            extra_t = shared_boot_t if method in (BOOTSTRAP, BCA) else 0.0
            t0 = time.perf_counter()
            try:
                if method is BOOTSTRAP:
                    lo = float(np.percentile(shared_boot, 100 * alpha / 2))
                    hi = float(np.percentile(shared_boot, 100 * (1 - alpha / 2)))
                elif method is BCA:
                    lo, hi = _two_sample_bca_ci(a, b, shared_boot, alpha)
                elif method is BAYES_BOOTSTRAP:
                    bd = _two_sample_boot_diffs(a, b, "bayes_bootstrap", n_bootstrap, rng)
                    lo = float(np.percentile(bd, 100 * alpha / 2))
                    hi = float(np.percentile(bd, 100 * (1 - alpha / 2)))
                elif method is SMOOTH_BOOTSTRAP:
                    bd = _two_sample_boot_diffs(a, b, "smooth_bootstrap", n_bootstrap, rng)
                    lo = float(np.percentile(bd, 100 * alpha / 2))
                    hi = float(np.percentile(bd, 100 * (1 - alpha / 2)))
                elif method is BOOTSTRAP_T:
                    lo, hi = _two_sample_bootstrap_t_ci(a, b, n_bootstrap, alpha, rng)
                elif method is WELCH_T:
                    lo, hi = _welch_t_ci(a, b, alpha)
                elif method is STUDENT_T:
                    lo, hi = _student_t_ci(a, b, alpha)
                elif method is MOVER_T:
                    lo, hi = _mover_t_ci(a, b, alpha, scale_bounds)
                elif method is MOVER_LOGIT_T:
                    lo, hi = _mover_logit_t_ci(a, b, alpha, scale_bounds)
                elif method is MOVER_NIG:
                    lo, hi = _mover_nig_ci(a, b, alpha, scale_bounds)
                elif method is WALD_UNPAIRED:
                    lo, hi = _wald_unpaired_ci(a, b, alpha)
                elif method is AGRESTI_CAFFO:
                    lo, hi = _agresti_caffo_ci(a, b, alpha)
                elif method is NEWCOMBE_HYBRID:
                    lo, hi = _newcombe_hybrid_ci(a, b, alpha)
                elif method is MIETTINEN_NURMINEN:
                    lo, hi = _miettinen_nurminen_ci(a, b, alpha)
                elif method is AGRESTI_MIN:
                    lo, hi = _agresti_min_ci(a, b, alpha)
                elif method is BAYES_BETA_INDEP:
                    lo, hi = _bayes_beta_indep_ci(a, b, alpha, bayes_n, rng)
                else:
                    raise AssertionError(f"unhandled mean-family method {method.name!r}")
            except Exception:
                lo = hi = obs_mean
            _record(key, lo, hi, time.perf_counter() - t0 + extra_t)

        # --- theta family --------------------------------------------------
        if theta_methods:
            theta_boot: np.ndarray | None = None
            theta_boot_t = 0.0
            if any(m in (THETA_BOOTSTRAP, THETA_BCA) for m in theta_methods):
                _t0 = time.perf_counter()
                theta_boot = _theta_bootstrap_stats(a, b, n_bootstrap, rng)
                theta_boot_t = time.perf_counter() - _t0  # charged to both consumers, as above
            obs_theta = _theta_hat(a, b)
            for method in theta_methods:
                key = (method, THETA_ESTIMAND)
                extra_t = theta_boot_t if method in (THETA_BOOTSTRAP, THETA_BCA) else 0.0
                t0 = time.perf_counter()
                try:
                    if method is THETA_BOOTSTRAP:
                        lo = float(np.percentile(theta_boot, 100 * alpha / 2))
                        hi = float(np.percentile(theta_boot, 100 * (1 - alpha / 2)))
                    elif method is THETA_BCA:
                        lo, hi = _theta_bca_ci(a, b, theta_boot, alpha)
                    elif method is BRUNNER_MUNZEL:
                        lo, hi = _brunner_munzel_ci(a, b, alpha)
                    elif method is BRUNNER_MUNZEL_LOGIT:
                        lo, hi = _brunner_munzel_logit_ci(a, b, alpha)
                    else:
                        raise AssertionError(f"unhandled theta method {method.name!r}")
                except Exception:
                    lo = hi = obs_theta
                _record(key, lo, hi, time.perf_counter() - t0 + extra_t)

    out: list[SimResult] = []
    for key in all_methods:
        method, est = key
        out.append(SimResult(
            source=source.source, label=source.label, eval_type=source.eval_type,
            estimand=est, n_a=n_a, n_b=n_b, method=method.name, n_reps=n_reps,
            covered=covered[key], total_width=total_w[key], total_score=total_score[key],
            total_pen_under=pen_under[key], total_pen_over=pen_over[key],
            rejects=rejects[key], total_time=total_t[key], total_time_sq=total_t_sq[key],
            is_null=source.is_null, true_value=truth[est],
            icc=source.icc, cohens_d=source.cohens_d,
        ))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


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
        eta_sec = max(self.total - step, 0) / max(step / elapsed, 1e-12)
        eta_m, eta_s = divmod(int(round(eta_sec)), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        prefix = f"{self.label}: " if self.label else ""
        print(f"\r  {prefix}[{bar}] {100.0*frac:6.2f}%  {step:>7d}/{self.total:<7d}  "
              f"ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}  {detail[:40]:<40s}", end="", flush=True)
        if is_final:
            print()


_CELL_SOURCES: list = []
_CELL_TRUE_THETAS: list = []


def _run_cell_worker(args: tuple) -> list[SimResult]:
    (sc_idx, n_a, n_b, n_reps, n_bootstrap, bayes_n, alpha, estimand, seed, method_names) = args
    return _run_cell(
        _CELL_SOURCES[sc_idx], n_a, n_b, n_reps, n_bootstrap, bayes_n, alpha, estimand,
        seed, _CELL_TRUE_THETAS[sc_idx], method_names,
    )


def run_simulation(
    sources: list[CIPairSource], sample_sizes: list[int], size_ratios: list[float],
    n_reps: int, n_bootstrap: int, bayes_n: int, alpha: float, estimand: str,
    true_thetas: list[float], progress_mode: str = "bar", seed: int = 42, n_workers: int = 1,
    method_names: frozenset[str] | None = None,
) -> list[SimResult]:
    global _CELL_SOURCES, _CELL_TRUE_THETAS
    _CELL_SOURCES = list(sources)
    _CELL_TRUE_THETAS = list(true_thetas)

    cells = [
        (i, n, max(2, int(round(n * ratio))))
        for i, s in enumerate(sources)
        for n in sample_sizes
        for ratio in size_ratios
    ]
    ss = np.random.SeedSequence(seed)
    child_seeds = [seq.generate_state(4).tolist() for seq in ss.spawn(len(cells))]
    args_list = [
        (sc_idx, n_a, n_b, n_reps, n_bootstrap, bayes_n, alpha, estimand, cseed, method_names)
        for (sc_idx, n_a, n_b), cseed in zip(cells, child_seeds)
    ]

    reporter = _ProgressReporter(len(cells), mode=progress_mode, label="ci_unpaired")
    results: list[SimResult] = []
    if n_workers <= 1:
        for i, a in enumerate(args_list):
            results.extend(_run_cell_worker(a))
            sc_idx, n_a, n_b = cells[i]
            reporter.update(i + 1, detail=f"{sources[sc_idx].eval_type} n={n_a}/{n_b}")
    else:
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, cell_results in enumerate(pool.imap_unordered(_run_cell_worker, args_list)):
                results.extend(cell_results)
                reporter.update(i + 1)
    reporter.update(len(cells), detail="done")
    return results


# ---------------------------------------------------------------------------
# Exact coverage (binary only) -- no Monte Carlo
# ---------------------------------------------------------------------------
# For a binary outcome every interval is a function of the two success counts
# alone, and (k_A, k_B) has a known joint distribution. So coverage can be
# summed exactly over all (n_A+1)(n_B+1) tables instead of estimated by
# simulation: no reps, no seeds, no Monte Carlo band. This is also how the
# two-independent-proportions literature reports coverage (Fagerland,
# Lydersen & Laake's figures are exact curves, not simulations), so it makes
# our binary numbers directly comparable to theirs.
#
# It matters practically: at 200-300 reps the Monte Carlo band on a coverage
# estimate is about +-0.03, which is the same size as the entire spread
# between the good binary methods. The sweep's `minCov` column is dominated
# by that noise -- it reported agresti_min dipping below agresti_caffo, which
# exact enumeration shows cannot happen.


@dataclass
class ExactResult:
    method: str
    n_a: int
    n_b: int
    min_coverage: float
    mean_coverage: float
    worst_p_a: float
    worst_p_b: float
    mean_width: float


_EXACT_BINARY_METHODS = (
    WALD_UNPAIRED, AGRESTI_CAFFO, NEWCOMBE_HYBRID, MIETTINEN_NURMINEN,
    AGRESTI_MIN, WELCH_T, STUDENT_T, MOVER_T, MOVER_LOGIT_T,
)
"""Deterministic binary methods only, i.e. every method whose interval is a
function of (k_A, k_B) alone -- verified deterministic across repeated calls,
which is what enumeration requires. bayes_beta_indep and the bootstrap family
are excluded because they are randomised, so no finite table of intervals
represents them and exact enumeration does not apply to them at all."""


def _exact_binary_ci_table(method, n_a: int, n_b: int, alpha: float) -> dict:
    """Every interval this method produces at (n_a, n_b), keyed by the two
    success counts. Computed once and reused across the whole (p_a, p_b)
    grid -- the interval does not depend on p."""
    fns = {
        WALD_UNPAIRED: _wald_unpaired_ci, AGRESTI_CAFFO: _agresti_caffo_ci,
        NEWCOMBE_HYBRID: _newcombe_hybrid_ci, MIETTINEN_NURMINEN: _miettinen_nurminen_ci,
        AGRESTI_MIN: _agresti_min_ci, WELCH_T: _welch_t_ci, STUDENT_T: _student_t_ci,
    }
    out = {}
    for ka in range(n_a + 1):
        a = np.r_[np.ones(ka), np.zeros(n_a - ka)]
        for kb in range(n_b + 1):
            b = np.r_[np.ones(kb), np.zeros(n_b - kb)]
            try:
                mover = {MOVER_T: _mover_t_ci, MOVER_LOGIT_T: _mover_logit_t_ci,
                         MOVER_NIG: _mover_nig_ci}.get(method)
                if mover is not None:
                    out[(ka, kb)] = mover(a, b, alpha, (0.0, 1.0))
                else:
                    out[(ka, kb)] = fns[method](a, b, alpha)
            except Exception:
                d = ka / n_a - kb / n_b
                out[(ka, kb)] = (d, d)
    return out


def run_exact_coverage(
    size_pairs: list[tuple[int, int]], p_grid: np.ndarray, alpha: float,
    method_names: frozenset[str] | None = None, progress_mode: str = "bar",
) -> list[ExactResult]:
    methods = [m for m in _EXACT_BINARY_METHODS
               if method_names is None or m.name in method_names]
    results: list[ExactResult] = []
    reporter = _ProgressReporter(len(size_pairs) * len(methods),
                                 mode=progress_mode, label="ci_unpaired/exact")
    step = 0
    for n_a, n_b in size_pairs:
        pmf_a = {p: stats.binom.pmf(np.arange(n_a + 1), n_a, p) for p in p_grid}
        pmf_b = {p: stats.binom.pmf(np.arange(n_b + 1), n_b, p) for p in p_grid}
        for method in methods:
            table = _exact_binary_ci_table(method, n_a, n_b, alpha)
            los = np.array([[table[(ka, kb)][0] for kb in range(n_b + 1)] for ka in range(n_a + 1)])
            his = np.array([[table[(ka, kb)][1] for kb in range(n_b + 1)] for ka in range(n_a + 1)])
            widths = his - los
            best = (1.0, 0.0, 0.0)
            covs, wmeans = [], []
            for pa in p_grid:
                wa = pmf_a[pa]
                for pb in p_grid:
                    wb = pmf_b[pb]
                    w = np.outer(wa, wb)
                    cov = float(np.sum(w * ((los <= pa - pb) & (pa - pb <= his))))
                    covs.append(cov)
                    wmeans.append(float(np.sum(w * widths)))
                    if cov < best[0]:
                        best = (cov, float(pa), float(pb))
            results.append(ExactResult(
                method=method.name, n_a=n_a, n_b=n_b,
                min_coverage=best[0], mean_coverage=float(np.mean(covs)),
                worst_p_a=best[1], worst_p_b=best[2], mean_width=float(np.mean(wmeans)),
            ))
            step += 1
            reporter.update(step, detail=f"{method.name} n={n_a}/{n_b}")
    reporter.update(len(size_pairs) * len(methods), detail="done")
    return results


def print_exact_report(results: list[ExactResult], alpha: float, n_grid_points: int) -> None:
    target = 1.0 - alpha
    print()
    print("=" * 100)
    print(f"EXACT binary coverage -- enumerated over all tables, {n_grid_points}x{n_grid_points} "
          f"(p_A, p_B) grid. No Monte Carlo.")
    print("=" * 100)
    by_size: dict = defaultdict(list)
    for r in results:
        by_size[(r.n_a, r.n_b)].append(r)
    for (n_a, n_b) in sorted(by_size):
        print(f"\n  n_A={n_a}, n_B={n_b}   (target {target:.2f})")
        print(f"  {'method':<22s} {'min cov':>9s} {'mean cov':>9s} {'width':>8s}  "
              f"{'worst at (p_A,p_B)':>20s}   verdict")
        rows = sorted(by_size[(n_a, n_b)], key=lambda r: -r.min_coverage)
        for r in rows:
            verdict = "holds nominal" if r.min_coverage >= target - 1e-9 else f"dips {r.min_coverage:.4f}"
            print(f"  {r.method:<22s} {r.min_coverage:9.4f} {r.mean_coverage:9.4f} "
                  f"{r.mean_width:8.4f}  {'(' + format(r.worst_p_a, '.2f') + ', ' + format(r.worst_p_b, '.2f') + ')':>20s}   {verdict}")


def save_exact_csv(results: list[ExactResult], out_dir: str, run_stem: str) -> str:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_stem}.csv"
    rows = [asdict(r) for r in results]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nSaved exact-coverage CSV: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _cov_marker(cov: float, target: float, tol: float = 0.04) -> str:
    if cov < target - tol:
        return " !"
    if cov > target + tol:
        return " +"
    return "  "


def print_report(results: list[SimResult], alpha: float, sample_sizes: list[int]) -> None:
    target = 1.0 - alpha
    by_est_type: dict = defaultdict(list)
    for r in results:
        by_est_type[(r.estimand, r.eval_type)].append(r)

    for (estimand, eval_type) in sorted(by_est_type):
        rows = by_est_type[(estimand, eval_type)]
        non_null = [r for r in rows if not r.is_null]
        nulls = [r for r in rows if r.is_null]
        print()
        print("=" * 100)
        print(f"estimand={estimand}  eval_type={eval_type}   "
              f"({len(rows)} rows, {len(nulls)} null / {len(non_null)} non-null)")
        print("=" * 100)
        print(f"{'method':<22s} {'meanCov':>8s} {'minCov':>8s} {'p05Cov':>8s} "
              f"{'width':>9s} {'score':>9s} {'penalty':>9s} {'TypeI':>8s} {'ms/call':>8s}")
        print("-" * 100)

        by_method: dict = defaultdict(list)
        for r in rows:
            by_method[r.method].append(r)
        for method in order_present_methods(set(by_method)):
            sub = by_method[method.name]
            sub_nn = [r for r in sub if not r.is_null] or sub
            covs = np.array([r.covered / r.n_reps for r in sub_nn])
            widths = np.array([r.total_width / r.n_reps for r in sub_nn])
            scores = np.array([r.total_score / r.n_reps for r in sub_nn])
            pens = np.array([(r.total_pen_under + r.total_pen_over) / r.n_reps for r in sub_nn])
            sub_null = [r for r in sub if r.is_null]
            type1 = (float(np.mean([r.rejects / r.n_reps for r in sub_null]))
                     if sub_null else float("nan"))
            n_calls = sum(r.n_reps for r in sub)
            ms = 1000.0 * sum(r.total_time for r in sub) / max(n_calls, 1)
            print(f"{method.name:<22s} {covs.mean():8.4f}{_cov_marker(covs.mean(), target)}"
                  f"{covs.min():7.4f}{_cov_marker(covs.min(), target)}"
                  f"{np.percentile(covs, 5):7.4f}  "
                  f"{widths.mean():8.4f} {scores.mean():9.4f} {pens.mean():9.4f} "
                  f"{type1:8.4f} {ms:8.3f}")

        # Coverage vs n, for the equal-size cells only (n_a == n_b), so the
        # column headers mean one thing.
        eq = [r for r in rows if r.n_a == r.n_b and not r.is_null]
        if eq:
            ns = sorted({r.n_a for r in eq})
            print()
            print(f"  coverage vs n (equal group sizes, non-null; target {target:.2f})")
            print("  " + f"{'method':<22s}" + "".join(f"{('n=' + str(n)):>9s}" for n in ns))
            by_m: dict = defaultdict(lambda: defaultdict(list))
            for r in eq:
                by_m[r.method][r.n_a].append(r.covered / r.n_reps)
            for method in order_present_methods(set(by_m)):
                cells = "".join(
                    f"{np.mean(by_m[method.name][n]):9.4f}" if by_m[method.name].get(n) else f"{'--':>9s}"
                    for n in ns
                )
                print("  " + f"{method.name:<22s}" + cells)


def _mc_proportion_stats(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    if total <= 0:
        return (float("nan"),) * 4
    p_hat = successes / total
    mcse = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total))
    return float(p_hat), mcse, max(0.0, p_hat - z * mcse), min(1.0, p_hat + z * mcse)


def _time_stats(subset: list[SimResult]) -> tuple[float, float]:
    total_reps = sum(r.n_reps for r in subset)
    if total_reps <= 0:
        return float("nan"), float("nan")
    sum_t = sum(r.total_time for r in subset)
    sum_t2 = sum(r.total_time_sq for r in subset)
    avg = sum_t / total_reps
    var = max(0.0, sum_t2 / total_reps - avg * avg)
    return avg * 1000.0, float(np.sqrt(var / total_reps)) * 1000.0


def latex_summary(results: list[SimResult], alpha: float, estimand: str) -> str:
    r"""Booktabs summary for ONE estimand, mirroring ci_paired.latex_overall_summary.

    One table per estimand rather than a combined one: a mean difference and a
    dominance probability live on different scales, so a Width or Score column
    spanning both would invite exactly the comparison that is not meaningful.
    Within a table, rows are blocked by eval type (bin/cont/lik) with a midrule
    between blocks, and per-n coverage columns are appended on the right
    because the aggregate column can hide miscalibration that only shows up at
    one end of the size range.
    """
    target = 1.0 - alpha
    rows_in = [r for r in results if r.estimand == estimand and not r.is_null]
    if not rows_in:
        return ""
    method_labels = [m.name for m in order_present_methods({r.method for r in rows_in})]
    sizes_present = sorted({r.n_a for r in rows_in})

    agg: dict = defaultdict(list)
    counts: dict = defaultdict(lambda: (0, 0))
    per_n: dict = defaultdict(lambda: (0, 0))
    for r in rows_in:
        g = report_eval_type_group(r.eval_type)
        n = max(r.n_reps, 1)
        agg[(g, r.method)].append((
            r.covered / n, r.total_width / n, r.total_score / n,
            (r.total_pen_under + r.total_pen_over) / n,
        ))
        c, t = counts[(g, r.method)]
        counts[(g, r.method)] = (c + r.covered, t + r.n_reps)
        c2, t2 = per_n[(g, r.method, r.n_a)]
        per_n[(g, r.method, r.n_a)] = (c2 + r.covered, t2 + r.n_reps)

    # Type-I error lives on the null rows, which are excluded from `rows_in`.
    null_rate: dict = defaultdict(lambda: (0, 0))
    for r in results:
        if r.estimand != estimand or not r.is_null:
            continue
        g = report_eval_type_group(r.eval_type)
        c, t = null_rate[(g, r.method)]
        null_rate[(g, r.method)] = (c + r.rejects, t + r.n_reps)

    groups_present = [g for g in ("bin", "cont", "lik", "grades")
                      if any(k[0] == g for k in agg)]
    columns = (["Method", "Coverage", "MinCov", "Width", "Score", "Penalty", "TypeI"]
               + [f"$n{{=}}{n}$" for n in sizes_present])
    rows: list[list[str]] = []
    rule_before: set[int] = set()
    for g in groups_present:
        if rows:
            rule_before.add(len(rows))
        block_start = len(rows)
        score_vals: list[float] = []
        for m in method_labels:
            if (g, m) not in agg:
                continue
            vals = agg[(g, m)]
            covs = np.array([v[0] for v in vals])
            width = float(np.mean([v[1] for v in vals]))
            score = float(np.mean([v[2] for v in vals]))
            pen = float(np.mean([v[3] for v in vals]))
            c, t = counts[(g, m)]
            pooled = c / t if t else float("nan")
            rc, rt = null_rate[(g, m)]
            t1 = rc / rt if rt else float("nan")
            cells = [
                f"{escape_latex(m)} ({g})",
                coverage_cell(pooled, target),
                coverage_cell(float(covs.min()), target),
                f"{width:.3f}", f"{score:.3f}", f"{pen:.3f}",
                f"{t1:.3f}" if np.isfinite(t1) else "--",
            ]
            for n in sizes_present:
                c2, t2 = per_n[(g, m, n)]
                cells.append(coverage_cell(c2 / t2, target) if t2 else "--")
            rows.append(cells)
            score_vals.append(score)
        block = [row[4] for row in rows[block_start:]]
        marked = mark_best_and_runnerup(block, score_vals, higher_is_better=False)
        for i, cell in enumerate(marked):
            rows[block_start + i][4] = cell

    est_desc = (r"mean difference $\bar{A}-\bar{B}$" if estimand == MEAN_ESTIMAND
                else r"$\theta = P(A{>}B) + \tfrac12 P(A{=}B)$")
    return booktabs_table(
        caption=(f"Between-subjects (unpaired) pairwise CI calibration, estimand: {est_desc}. "
                 f"Non-null cells only; TypeI is the rejection rate on null cells. "
                 f"Coverage cells are shaded when they fall outside the acceptable band "
                 f"around {target:.2f}. Best Score per block in bold, runner-up underlined."),
        label=f"tab:ci-unpaired-{estimand.replace('_', '-')}",
        columns=columns, rows=rows, rule_before=rule_before,
    )


def latex_exact_summary(results: list, alpha: float) -> str:
    """Booktabs table for the exact binary coverage mode."""
    if not results:
        return ""
    target = 1.0 - alpha
    by_size = sorted({(r.n_a, r.n_b) for r in results})
    methods = [m.name for m in order_present_methods({r.method for r in results})]
    lookup = {(r.method, r.n_a, r.n_b): r for r in results}
    columns = ["Method"] + [f"$n{{=}}{a}/{b}$" for a, b in by_size] + ["Width"]
    rows = []
    for m in methods:
        cells = [escape_latex(m)]
        widths = []
        for a, b in by_size:
            r = lookup.get((m, a, b))
            cells.append(coverage_cell(r.min_coverage, target) if r else "--")
            if r:
                widths.append(r.mean_width)
        cells.append(f"{np.mean(widths):.3f}" if widths else "--")
        rows.append(cells)
    return booktabs_table(
        caption=(f"Exact MINIMUM coverage for a difference of two independent proportions, "
                 f"enumerated over every $(k_A, k_B)$ table and a grid of $(p_A, p_B)$. "
                 f"No Monte Carlo, so these are exact values rather than estimates with a "
                 f"simulation band, and directly comparable to the published coverage curves "
                 f"in the two-independent-proportions literature. Target {target:.2f}."),
        label="tab:ci-unpaired-exact",
        columns=columns, rows=rows,
    )


def save_results_artifacts(
    results: list[SimResult], alpha: float, out_dir: str, run_stem: str, latex: bool = False,
) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{run_stem}_results.csv"
    rows = []
    for r in results:
        n = max(r.n_reps, 1)
        _, mcse, lo, hi = _mc_proportion_stats(r.covered, r.n_reps)
        avg_ms, se_ms = _time_stats([r])
        d = asdict(r)
        d.update({
            "coverage": r.covered / n, "mean_width": r.total_width / n,
            "mean_score": r.total_score / n,
            "mean_penalty": (r.total_pen_under + r.total_pen_over) / n,
            "reject_rate": r.rejects / n, "mcse": mcse,
            "band95_low": lo, "band95_high": hi,
            "avg_time_ms": avg_ms, "se_time_ms": se_ms,
        })
        rows.append(d)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary_path = out / f"{run_stem}_summary.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(results, alpha=alpha, sample_sizes=sorted({r.n_a for r in results}))
    text = buf.getvalue()
    if latex:
        text += "\n% --- LaTeX tables (--latex) ---\n"
        for est in (MEAN_ESTIMAND, THETA_ESTIMAND):
            text += latex_summary(results, alpha=alpha, estimand=est)
    summary_path.write_text(text, encoding="utf-8")
    print(f"\nSaved results: {csv_path}")
    print(f"Saved log: {summary_path}")
    return [str(csv_path), str(summary_path)]


def save_coverage_vs_n_plot(results: list[SimResult], alpha: float, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..methods import get_method_color

    eq = [r for r in results if r.n_a == r.n_b and not r.is_null]
    if not eq:
        eq = [r for r in results if not r.is_null]
    keys = sorted({(r.estimand, r.eval_type) for r in eq})
    if not keys:
        raise ValueError("no non-null results to plot")
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.8), squeeze=False)
    target = 1.0 - alpha
    for ax, (estimand, eval_type) in zip(axes[0], keys):
        rows = [r for r in eq if r.estimand == estimand and r.eval_type == eval_type]
        by_m: dict = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by_m[r.method][r.n_a].append(r.covered / r.n_reps)
        for method in order_present_methods(set(by_m)):
            ns = sorted(by_m[method.name])
            ys = [float(np.mean(by_m[method.name][n])) for n in ns]
            ax.plot(ns, ys, marker="o", ms=3, lw=1.3, label=method.name,
                    color=get_method_color(method.name))
        ax.axhline(target, color="k", ls="--", lw=1)
        ax.set_title(f"{estimand} / {eval_type}", fontsize=10)
        ax.set_xlabel("n per group")
        ax.set_ylabel("coverage")
        ax.set_ylim(min(0.80, target - 0.12), 1.005)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_panels(results: list[SimResult]) -> list[tuple[str, str]]:
    """(estimand, eval_type) panels present in the non-null results."""
    return sorted({(r.estimand, r.eval_type) for r in results if not r.is_null})


def save_width_vs_n_plot(results: list[SimResult], alpha: float, out_path: str) -> str:
    """Mean interval width vs. n -- the sharpness half of the tradeoff the
    coverage plot shows the validity half of. A method that covers by being
    wide is visible only here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..methods import get_method_color

    eq = [r for r in results if r.n_a == r.n_b and not r.is_null]
    keys = _plot_panels(eq)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.8), squeeze=False)
    for ax, (estimand, eval_type) in zip(axes[0], keys):
        rows = [r for r in eq if r.estimand == estimand and r.eval_type == eval_type]
        by_m: dict = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by_m[r.method][r.n_a].append(r.total_width / max(r.n_reps, 1))
        for method in order_present_methods(set(by_m)):
            ns = sorted(by_m[method.name])
            ax.plot(ns, [float(np.mean(by_m[method.name][n])) for n in ns],
                    marker="o", ms=3, lw=1.3, label=method.name,
                    color=get_method_color(method.name))
        ax.set_title(f"{estimand} / {eval_type}", fontsize=10)
        ax.set_xlabel("n per group")
        ax.set_ylabel("mean CI width")
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_cost_plot(results: list[SimResult], alpha: float, out_path: str) -> str:
    """Coverage against compute cost. Some of these methods differ by three
    orders of magnitude in runtime (agresti_min's exact enumeration vs. a
    closed-form Wald), so "is the extra coverage worth the wait" is a real
    question a reader will ask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..methods import get_method_color

    non_null = [r for r in results if not r.is_null]
    keys = _plot_panels(non_null)
    target = 1.0 - alpha
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.8), squeeze=False)
    for ax, (estimand, eval_type) in zip(axes[0], keys):
        rows = [r for r in non_null if r.estimand == estimand and r.eval_type == eval_type]
        by_m: dict = defaultdict(list)
        for r in rows:
            by_m[r.method].append(r)
        for method in order_present_methods(set(by_m)):
            sub = by_m[method.name]
            cov = float(np.mean([r.covered / max(r.n_reps, 1) for r in sub]))
            ms, _ = _time_stats(sub)
            ax.scatter(max(ms, 1e-4), cov, s=34, color=get_method_color(method.name),
                       label=method.name, zorder=3)
        ax.axhline(target, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"{estimand} / {eval_type}", fontsize=10)
        ax.set_xlabel("mean time per interval (ms, log)")
        ax.set_ylabel("mean coverage")
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_reliability_violin_plot(results: list[SimResult], alpha: float, out_path: str) -> str:
    """Distribution of per-scenario coverage, one violin per method.

    The mean coverage column hides the shape: a method averaging 0.95 by
    over-covering on easy scenarios and under-covering badly on hard ones is
    not the same as one sitting at 0.95 everywhere, and only the spread
    distinguishes them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..methods import get_method_color

    non_null = [r for r in results if not r.is_null]
    keys = _plot_panels(non_null)
    target = 1.0 - alpha
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 3.1 * len(keys)), squeeze=False)
    for ax, (estimand, eval_type) in zip(axes[:, 0], keys):
        rows = [r for r in non_null if r.estimand == estimand and r.eval_type == eval_type]
        by_m: dict = defaultdict(list)
        for r in rows:
            by_m[r.method].append(r.covered / max(r.n_reps, 1))
        methods = order_present_methods(set(by_m))
        data = [by_m[m.name] for m in methods]
        parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.85)
        for body, m in zip(parts["bodies"], methods):
            body.set_facecolor(get_method_color(m.name))
            body.set_alpha(0.65)
        ax.axhline(target, color="k", ls="--", lw=1)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([m.name for m in methods], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("coverage")
        ax.set_title(f"{estimand} / {eval_type}", fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="synthetic",
                        help="'synthetic' (default) or 'real' -- human labels from the "
                             "judge-bias corpora and the App Store review corpus, split "
                             "into disjoint groups (see scenarios/real_unpaired.py).")
    parser.add_argument("--real-datasets", nargs="+", choices=REAL_UNPAIRED_DATASETS,
                        default=None, metavar="NAME",
                        help="Real data: restrict to these corpora (default: all available).")
    parser.add_argument("--data-dir", default=REAL_DEFAULT_DATA_DIR,
                        help=f"Real data: directory holding the corpus CSVs (default: {REAL_DEFAULT_DATA_DIR!r}).")
    parser.add_argument("--scenario-suite", choices=SCENARIO_SUITES, default="expanded")
    parser.add_argument("--eval-types", nargs="+", choices=EVAL_TYPES,
                        default=list(DEFAULT_EVAL_TYPES), metavar="TYPE")
    parser.add_argument("--estimand", choices=ESTIMANDS, default="both",
                        help="'mean_diff' (mean(A)-mean(B)), 'theta' (P(A>B)+.5P(A=B)), "
                             "or 'both' (default). The shipped unpaired path uses mean_diff "
                             "for binary and theta for continuous/likert, so 'both' is what "
                             "actually covers compare(design='unpaired').")
    parser.add_argument("--methods", nargs="+", default=None, metavar="NAME",
                        help="Restrict computation to these method names.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50], metavar="N",
                        help="Group-A sizes to sweep.")
    parser.add_argument("--size-ratios", type=float, nargs="+", default=[1.0], metavar="R",
                        help="n_B / n_A ratios (default 1.0). Unequal group sizes are the "
                             "norm between subjects, so e.g. --size-ratios 1.0 2.0.")
    parser.add_argument("--reps", type=int, default=200, metavar="N")
    parser.add_argument("--bootstrap-n", type=int, default=500, metavar="N")
    parser.add_argument("--bayes-n", type=int, default=2000, metavar="N")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--icc-values", type=float, nargs="+", default=None, metavar="ICC")
    parser.add_argument("--cohens-d-values", type=float, nargs="+", default=[0.3], metavar="D")
    parser.add_argument("--include-null", action="store_true", default=False)
    parser.add_argument("--theta-mc-n", type=int, default=_TRUE_THETA_MC_N, metavar="N",
                        help="Monte Carlo draws for the per-source true theta.")
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="bar")
    parser.add_argument("--plots", choices=PLOT_MODES, default="save")
    parser.add_argument("--save-results", choices=RESULTS_MODES, default="save")
    parser.add_argument("--out-dir", default="simulations/out")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--latex", action="store_true", default=False,
                        help="Append booktabs LaTeX tables to the saved summary .log "
                             "(one per estimand), and write a .tex for --exact-coverage.")
    parser.add_argument("--exact-coverage", action="store_true", default=False,
                        help="Binary only: compute coverage EXACTLY by enumerating every "
                             "possible pair of success counts, instead of estimating it by "
                             "simulation. No reps, no seed, no Monte Carlo error -- and it is "
                             "how the two-independent-proportions literature reports coverage, "
                             "so the numbers are directly comparable to Fagerland et al. "
                             "Ignores --reps/--sizes-as-pairs semantics: use --exact-sizes.")
    parser.add_argument("--exact-sizes", type=int, nargs="+", default=[10, 20, 30], metavar="N",
                        help="--exact-coverage: group-A sizes (combined with --size-ratios).")
    parser.add_argument("--exact-p-grid", type=int, default=19, metavar="K",
                        help="--exact-coverage: number of p values per axis, spread over "
                             "(0, 1) exclusive (default 19 -> 0.05..0.95 in steps of 0.05).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), metavar="N")


def official_args(base_seed: int = 42) -> argparse.Namespace:
    """Canonical preset -- mirrors ci_paired.official_args' breadth."""
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="expanded",
        eval_types=["binary", "continuous", "likert"], estimand="both", methods=None,
        real_datasets=None, data_dir=REAL_DEFAULT_DATA_DIR,
        sizes=[10, 15, 20, 30, 40, 50, 60, 80, 100], size_ratios=[1.0, 2.0],
        reps=300, bootstrap_n=2000, bayes_n=10000, alpha=0.05, seed=base_seed,
        icc_values=[0.01, 0.3, 0.5, 0.65, 0.75, 0.85, 0.95], cohens_d_values=[0.2, 0.4],
        include_null=True, theta_mc_n=_TRUE_THETA_MC_N,
        exact_coverage=False, exact_sizes=[10, 20, 30], exact_p_grid=19, latex=True,
        progress="bar", plots="save", save_results="save",
        out_dir="simulations/out", plots_dir=None,
        workers=max(1, (os.cpu_count() or 2) - 1),
    )


def official_variants(base_seed: int = 42) -> list[tuple[str, argparse.Namespace]]:
    """Entries offered by ``--official-tests``' interactive menu.

    Listed but flagged PROVISIONAL: the method slate here has not been
    adjudicated against the literature yet, so this is not something to
    quote in the paper. Selecting it from the menu is opt-in, so listing it
    costs nothing and keeps the case reachable through the same path as
    every other case.
    """
    real = argparse.Namespace(**{**vars(official_args(base_seed + 1)),
                                 "data_source": "real",
                                 # Real pools are fixed corpora, so the synthetic
                                 # shape/icc/effect sweep does not apply to them.
                                 "icc_values": None, "cohens_d_values": [0.3]})
    return [
        ("PROVISIONAL: between-subjects pairwise CIs, synthetic (mean diff + theta)",
         official_args(base_seed)),
        ("PROVISIONAL: between-subjects pairwise CIs, real human labels (mean diff + theta)",
         real),
        ("PROVISIONAL: between-subjects binary CIs, EXACT coverage (no Monte Carlo)",
         argparse.Namespace(**{**vars(official_args(base_seed)),
                               "exact_coverage": True, "estimand": MEAN_ESTIMAND})),
    ]


def quick_args(base_seed: int = 43, data_source: str = "synthetic") -> argparse.Namespace:
    """Small smoke-test preset -- runs in well under a minute.

    ``data_source`` is accepted and ignored (this case is synthetic-only in
    this pass), matching the convention the other synthetic-only cases use
    so ``--quick-test``'s synthetic+real double call still works.
    """
    return argparse.Namespace(
        data_source="synthetic", scenario_suite="standard",
        eval_types=["binary", "continuous", "likert"], estimand="both", methods=None,
        real_datasets=None, data_dir=REAL_DEFAULT_DATA_DIR,
        sizes=[10, 30], size_ratios=[1.0], reps=40, bootstrap_n=200, bayes_n=800,
        alpha=0.05, seed=base_seed, icc_values=[0.5], cohens_d_values=[0.3],
        include_null=True, theta_mc_n=40_000,
        exact_coverage=False, exact_sizes=[10, 20, 30], exact_p_grid=19, latex=True,
        progress="bar", plots="off", save_results="off",
        out_dir="simulations/out", plots_dir=None,
        workers=max(1, (os.cpu_count() or 2) - 1),
    )


def run(args: argparse.Namespace) -> CaseResult:
    t0 = time.time()
    try:
        plots_dir = args.plots_dir or str(Path(args.out_dir) / "plots")

        if getattr(args, "exact_coverage", False):
            k = args.exact_p_grid
            p_grid = np.linspace(0.0, 1.0, k + 2)[1:-1]
            size_pairs = [(n, max(2, int(round(n * r))))
                          for n in args.exact_sizes for r in args.size_ratios]
            method_names = frozenset(args.methods) if getattr(args, "methods", None) else None
            print(f"\nci_unpaired EXACT binary coverage -- sizes={size_pairs}, "
                  f"{k}x{k} p-grid, alpha={args.alpha}")
            ex = run_exact_coverage(size_pairs, p_grid, args.alpha,
                                    method_names=method_names, progress_mode=args.progress)
            print_exact_report(ex, alpha=args.alpha, n_grid_points=k)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            paths = []
            if args.save_results == "save":
                paths.append(save_exact_csv(ex, args.out_dir, f"ci_unpaired_exact_{stamp}"))
                if getattr(args, "latex", False):
                    tex = Path(args.out_dir) / f"ci_unpaired_exact_{stamp}.tex"
                    tex.write_text(latex_exact_summary(ex, alpha=args.alpha), encoding="utf-8")
                    paths.append(str(tex))
                    print(f"Saved LaTeX table: {tex}")
            return CaseResult(
                case_name=CASE_NAME, status="ok", output_paths=paths,
                key_metrics={"n_results": len(ex),
                             "worst_min_coverage": min(r.min_coverage for r in ex)},
                duration_s=time.time() - t0,
            )

        print(f"\nci_unpaired simulation -- data_source={args.data_source}, "
              f"estimand={args.estimand}, sizes={args.sizes}, size_ratios={args.size_ratios}")

        if args.data_source == "real":
            sources = build_real_unpaired_sources(
                data_dir=getattr(args, "data_dir", REAL_DEFAULT_DATA_DIR),
                datasets=getattr(args, "real_datasets", None),
                include_null=args.include_null,
            )
        else:
            icc_values = args.icc_values if args.icc_values is not None else [0.01, 0.3, 0.5, 0.65, 0.75, 0.85, 0.95]
            sources = build_pair_sources(
                suite=args.scenario_suite, icc_values=icc_values,
                cohens_d_values=args.cohens_d_values, include_null=args.include_null,
            )
        if args.eval_types:
            requested = set(args.eval_types)
            sources = [s for s in sources if s.eval_type in requested]
        if not sources:
            raise ValueError("No CIPairSources left after filtering.")

        need_theta = args.estimand in (THETA_ESTIMAND, "both")
        if need_theta:
            print(f"  estimating true theta for {len(sources)} sources "
                  f"({args.theta_mc_n:,} MC draws each) ...")
            true_thetas = [
                _estimate_true_theta(s, seed=1000 + i, n_mc=args.theta_mc_n)
                for i, s in enumerate(sources)
            ]
        else:
            true_thetas = [float("nan")] * len(sources)

        method_names = frozenset(args.methods) if getattr(args, "methods", None) else None
        n_cells = len(sources) * len(args.sizes) * len(args.size_ratios)
        print(f"  {len(sources)} sources, {n_cells} cells, reps={args.reps}, alpha={args.alpha}")

        results = run_simulation(
            sources, sample_sizes=args.sizes, size_ratios=list(args.size_ratios),
            n_reps=args.reps, n_bootstrap=args.bootstrap_n, bayes_n=args.bayes_n,
            alpha=args.alpha, estimand=args.estimand, true_thetas=true_thetas,
            progress_mode=args.progress, seed=args.seed,
            n_workers=getattr(args, "workers", 1), method_names=method_names,
        )

        print_report(results, alpha=args.alpha, sample_sizes=args.sizes)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_stem = f"ci_unpaired_{args.data_source}_{args.estimand}_reps{args.reps}_{stamp}"
        output_paths: list[str] = []
        if args.save_results == "save":
            output_paths += save_results_artifacts(
                results, alpha=args.alpha, out_dir=args.out_dir, run_stem=run_stem,
                latex=getattr(args, "latex", False),
            )
        if args.plots == "save":
            for suffix, fn in (
                ("coverage_vs_n", save_coverage_vs_n_plot),
                ("width_vs_n", save_width_vs_n_plot),
                ("cost_coverage", save_cost_plot),
                ("reliability_violin", save_reliability_violin_plot),
            ):
                output_paths.append(fn(
                    results, alpha=args.alpha,
                    out_path=str(Path(plots_dir) / f"{run_stem}_{suffix}.png"),
                ))
            print(f"Saved plots: {output_paths[-4:]}")

        non_null = [r for r in results if not r.is_null]
        overall_cov = float(np.mean([r.covered / r.n_reps for r in non_null])) if non_null else float("nan")
        return CaseResult(
            case_name=CASE_NAME, status="ok", output_paths=output_paths,
            key_metrics={"n_results": len(results), "overall_mean_coverage": overall_cov},
            duration_s=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001 -- harness contract: report, don't crash the batch
        return CaseResult(
            case_name=CASE_NAME, status="error", duration_s=time.time() - t0, error=repr(exc),
        )
