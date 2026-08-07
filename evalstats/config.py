"""Global configuration for evalstats."""

import os
import sys
from dataclasses import dataclass
from typing import Literal, Optional

# We use a global variable to store the default alpha for CI analyses,
# which can be set by the user via set_alpha_ci() and is used by default in
# all CI analyses across the library (but can be overridden on a per-analysis basis
# by passing an explicit alpha).
_alpha: float = 0.05

# Alpha levels used to build the gradient CI bands in terminal plots.
# Ordered narrowest→widest: 90%, 95%, 99%, 99.9% CI.
GRADIENT_CI_ALPHAS: tuple[float, ...] = (0.32, 0.10, 0.05, 0.01)


def supports_ansi_color() -> bool:
    """Whether ANSI color escape codes should be emitted for console output.

    True for a real terminal. ``sys.stdout.isatty()`` is False for
    Jupyter's ipykernel output stream, but JupyterLab/Notebook still render
    ANSI codes (converted to HTML) in the output cell, so it's detected
    separately here rather than being treated like a plain redirected/piped
    stream. Respects the NO_COLOR / FORCE_COLOR conventions.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if sys.stdout.isatty():
        return True
    return type(sys.stdout).__module__.startswith("ipykernel")


# ---------------------------------------------------------------------------
# method="auto" resolution
# ---------------------------------------------------------------------------
# Every public entry point that accepts method="auto" (analyze(),
# resolve_resampling_method(), the max-T simultaneous-CI path, ...)
# resolves to a concrete method by consulting the tables below. Keeping
# them here means the full "what runs automatically, and when" surface is
# visible in one place instead of scattered across router.py / resampling.py
# / paired.py.
#
# Note: this does *not* cover the R >= 3 "seeded" threshold (whether a
# two-level nested bootstrap is used), which is a structural property of
# the input data shape (see BenchmarkResult.is_seeded in core/types.py)
# rather than a method-selection choice.

DataKind = Literal["binary", "bounded_01", "unbounded"]

# --- Bootstrap resampling variant (resolve_resampling_method) --------------
# Plain (non-binary) bootstrap CIs: sample_size >= this -> "bootstrap"
# (simpler and at least as accurate at that scale); below it -> "bootstrap_t".
BOOTSTRAP_AUTO_MIN_N: int = 200

# --- max-T simultaneous CI bootstrap variant --------------------------------
# Bonferroni is the default simultaneous-CI construction (see
# _simultaneous_cis_router in evalstats/core/paired.py); max-T is opt-in via
# prefer="max_t". When max-T IS requested, 'auto' always resolves to this
# variant regardless of sample size — max-T's studentization needs a stable
# per-replicate SE estimate, which the studentized bootstrap provides more
# robustly at all N.
MAX_T_AUTO_METHOD: str = "bootstrap_t"


@dataclass(frozen=True)
class AutoAnalyzeRule:
    """One row of the ``analyze(method="auto")`` routing matrix.

    A rule applies when the observed data matches ``data_kind`` and (for
    binary data) the per-template sample size ``N`` is below ``max_n``
    (``None`` = no upper bound, i.e. applies at any N).
    """
    data_kind: DataKind
    max_n: Optional[int]
    pairwise_method: str
    robustness_method_single_run: str
    robustness_method_seeded: str
    reason: str


# Ordered (N, data-kind) x (single-run, seeded) matrix used by analyze() to
# resolve method="auto". Read as: for this data_kind, when N < max_n, use
# this pairwise_method; the robustness (single-sample marginal CI) method
# additionally depends on whether the benchmark is seeded (R >= 3 runs).
#
# This table is the code-level source of truth for the paper's CI
# decision-tree figure (fig:ci-decision-tree) -- every row below should have
# a matching leaf there. "Wilson flat" / "Logit-t on run means" in that
# figure are not separate implementations: robustness_metrics() already
# collapses (N, M, R) score arrays to per-input cell means before dispatching
# on marginal_method, so plain "wilson" / "logit_t" applied to that
# already-collapsed array *is* the flat/run-means variant. Same story for
# "Logit-t on run mean differences" in all_pairwise()'s "logit_t" path. And
# "ER-Tango" is not a separate method name either -- pairwise_method="tango"
# internally detects R >= 3 seeded runs and switches to the effective-N
# multirun variant (tango_paired_ci_multirun_effective), which *is* ER-Tango.
# (tango_paired_ci_multirun_moments is a related but distinct variant --
# still available in core/resampling.py, but not what "tango" routes to.)
#
# "bounded_01" (the data_kind label) no longer means the data is literally
# valued in [0, 1] -- it means router.py._analyze_single could establish a
# reliable [lo, hi] range for it via resolve_score_bounds() (core/
# resampling.py): either the caller's explicit score_range, or an exact
# [0, 1] match (still emits a UserWarning, since it's still an inference).
# There is deliberately no third option that approximates a range from the
# sample's own min/max -- that's not a safe substitute for a metric's true
# theoretical bounds (e.g. a 1-5 Likert scale sampled only between 2 and 4
# would silently produce a miscalibrated CI). Numeric data outside [0, 1]
# with no score_range given falls through to the "unbounded" row instead
# (also with a UserWarning, recommending an explicit score_range). Named
# "unbounded", not "continuous" -- the simulation harness's own eval_type
# taxonomy already uses "continuous" for a different concept (a bounded
# Beta-distributed shape), and reusing the same word here for an entirely
# different meaning (numeric data with NO reliable range at all) was a
# recurring source of confusion between the package and the harness.
AUTO_ANALYZE_METHOD_TABLE: tuple[AutoAnalyzeRule, ...] = (
    AutoAnalyzeRule(
        data_kind="binary", max_n=50,
        pairwise_method="bayes_binary",
        robustness_method_single_run="wilson",
        robustness_method_seeded="wilson",
        reason=(
            "Real-data simulations show Tango under-covers in "
            "dominated/jointly-sparse pairs at small N, regardless of run "
            "count, so small-N binary data uses the Bayesian paired model "
            "(citet{dontusetheclt}) instead. Cutoff N=50, per "
            "fig:ci-decision-tree. Marginal CIs use plain Wilson regardless "
            "of seeding ('Wilson flat' in the figure) -- NIG-nested was "
            "tried and dropped in favor of this simpler, better-calibrated "
            "choice."
        ),
    ),
    AutoAnalyzeRule(
        data_kind="binary", max_n=None,
        pairwise_method="tango",
        robustness_method_single_run="wilson",
        robustness_method_seeded="wilson",
        reason=(
            "N >= 50 binary data: Tango pairwise (ER-Tango via the "
            "effective-N multirun variant when seeded), Wilson-flat marginal."
        ),
    ),
    AutoAnalyzeRule(
        data_kind="bounded_01", max_n=None,
        pairwise_method="logit_t",
        robustness_method_single_run="logit_t",
        robustness_method_seeded="logit_t",
        reason=(
            "Numeric data with a reliable [lo, hi] range (e.g. normalised "
            "accuracy, ROUGE, or any scale declared via an explicit "
            "score_range -- a Likert scale, a percentage grade): Logit-t "
            "pairwise and marginal CIs, per fig:ci-decision-tree. The range "
            "is either the caller's explicit score_range or an exact [0, 1] "
            "match -- see resolve_score_bounds() in core/resampling.py. "
            "Supersedes the earlier t_interval (pairwise) / nig, nig_nested "
            "(marginal) defaults for data in this range."
        ),
    ),
    AutoAnalyzeRule(
        data_kind="unbounded", max_n=None,
        pairwise_method="t_interval",
        robustness_method_single_run="t_interval",
        robustness_method_seeded="t_interval",
        reason=(
            "Numeric data outside [0, 1] with no explicit score_range -- "
            "evalstats deliberately does not guess a [lo, hi] range from the "
            "sample's own min/max (unreliable; see resolve_score_bounds()), "
            "so this falls back to a plain t-interval, with a UserWarning "
            "recommending the caller pass score_range explicitly to get "
            "Logit-t instead (not covered by fig:ci-decision-tree, which "
            "assumes a known eval-metric range)."
        ),
    ),
)


def resolve_auto_analyze_methods(
    data_kind: DataKind, n: int, seeded: bool,
) -> tuple[str, str]:
    """Resolve ``analyze(method="auto")`` to concrete (pairwise, robustness) methods.

    Looks up :data:`AUTO_ANALYZE_METHOD_TABLE` for the first rule matching
    ``data_kind`` and ``n``, in table order.

    Parameters
    ----------
    data_kind : "binary", "bounded_01", or "unbounded"
        Detected data type (see ``is_binary_scores`` / ``is_bounded_01_scores``
        in ``core.resampling``).
    n : int
        Per-template sample size (number of inputs).
    seeded : bool
        Whether the benchmark carries R >= 3 runs (nested bootstrap path).

    Returns
    -------
    tuple[str, str]
        ``(pairwise_method, robustness_method)``.
    """
    for rule in AUTO_ANALYZE_METHOD_TABLE:
        if rule.data_kind != data_kind:
            continue
        if rule.max_n is not None and n >= rule.max_n:
            continue
        robustness = rule.robustness_method_seeded if seeded else rule.robustness_method_single_run
        return rule.pairwise_method, robustness
    raise AssertionError(
        f"no AUTO_ANALYZE_METHOD_TABLE rule matched data_kind={data_kind!r}, n={n}"
    )


@dataclass(frozen=True)
class PPIAutoMethodRule:
    """One row of the PPI-alignment-correction ``method="auto"`` routing table.

    A separate table from :data:`AUTO_ANALYZE_METHOD_TABLE`: the non-aligned
    auto default for a given ``data_kind`` (e.g. ``"t_interval"`` for
    continuous data) does not necessarily have a validated PPI-corrected
    counterpart, so PPI alignment correction resolves ``"auto"`` to whichever
    method *does* have one, instead of reusing the non-aligned default and
    failing.
    """
    data_kind: DataKind
    pairwise_method: str
    robustness_method: str
    reason: str


# PPI alignment correction requires N >= 50 (enforced in
# evalstats.api._run_alignment_ppi), so there is no small-N branch here the
# way AUTO_ANALYZE_METHOD_TABLE has one for binary data.
PPI_AUTO_METHOD_TABLE: tuple[PPIAutoMethodRule, ...] = (
    PPIAutoMethodRule(
        data_kind="binary",
        pairwise_method="tango",
        robustness_method="wilson",
        reason=(
            "Binary data: Tango (pairwise) and Wilson (marginal) both have "
            "closed-form PPI-corrected forms via an effective-n substitution "
            "(see evalstats.tests._ppi_paired_tango / _ppi_single_wilson)."
        ),
    ),
    PPIAutoMethodRule(
        data_kind="bounded_01",
        pairwise_method="ppi_logit_t",
        robustness_method="ppi_logit_t",
        reason=(
            "Numeric [0, 1]-bounded data: closed-form (no-bootstrap) PPI-corrected "
            "logit-t (see evalstats.tests._ppi_paired_logit_t / "
            "_ppi_single_logit_t, wrapping evalstats.ppi._analytic_logit_t_correct), "
            "matching the non-aligned default's own logit_t choice for this "
            "data_kind (see AUTO_ANALYZE_METHOD_TABLE). RESOLVED 2026-08-05: this "
            "row previously routed to bootstrap_t as a TEMPORARY stand-in, since no "
            "PPI-corrected logit_t existed -- that gap is now closed."
        ),
    ),
    PPIAutoMethodRule(
        data_kind="unbounded",
        pairwise_method="ppi_t_interval",
        robustness_method="ppi_t_interval",
        reason=(
            "Numeric data with no reliable [lo, hi] range: closed-form "
            "(no-bootstrap) PPI-corrected t-interval (see evalstats.tests."
            "_ppi_paired_t_interval / _ppi_single_t_interval, wrapping "
            "evalstats.ppi._analytic_mean_correct), matching the non-aligned "
            "default's own t_interval fallback for this data_kind (see "
            "AUTO_ANALYZE_METHOD_TABLE) -- logit_t requires known bounds to do "
            "the logit transform at all, which this data_kind by definition lacks."
        ),
    ),
)


def resolve_ppi_auto_methods(data_kind: DataKind) -> tuple[str, str]:
    """Resolve PPI alignment correction's ``method="auto"`` to concrete
    ``(pairwise_method, robustness_method)``, for use by
    ``evalstats.api._run_alignment_ppi``.

    Raises
    ------
    ValueError
        If no PPI-corrected method is defined for ``data_kind``.
    """
    for rule in PPI_AUTO_METHOD_TABLE:
        if rule.data_kind == data_kind:
            return rule.pairwise_method, rule.robustness_method
    raise ValueError(
        f"No PPI-corrected auto method is defined for data_kind={data_kind!r}."
    )


def set_alpha_ci(alpha: float) -> None:
    """Set the default significance level used across all CI analyses.

    Parameters
    ----------
    alpha:
        Significance level (e.g. 0.05 for 95% CI, 0.01 for 99% CI).
        Must be in the open interval (0, 1).
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    global _alpha
    _alpha = alpha


def get_alpha_ci() -> float:
    """Return the current default significance level."""
    return _alpha
