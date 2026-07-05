"""Global configuration for evalstats."""

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

DataKind = Literal["binary", "bounded_01", "continuous"]

# --- Bootstrap resampling variant (resolve_resampling_method) --------------
# Plain (non-binary) bootstrap CIs: sample_size >= this -> "bootstrap"
# (simpler and at least as accurate at that scale); below it -> "bootstrap_t".
BOOTSTRAP_AUTO_MIN_N: int = 200

# --- max-T simultaneous CI bootstrap variant --------------------------------
# 'auto' always resolves to this variant for simultaneous (max-T) CIs, regardless
# of sample size — max-T's studentization needs a stable per-replicate SE
# estimate, which the studentized bootstrap provides more robustly at all N.
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
AUTO_ANALYZE_METHOD_TABLE: tuple[AutoAnalyzeRule, ...] = (
    AutoAnalyzeRule(
        data_kind="binary", max_n=50,
        pairwise_method="bayes_binary",
        robustness_method_single_run="wilson",
        robustness_method_seeded="nig_nested",
        reason=(
            "Real-data simulations show Tango under-covers in "
            "dominated/jointly-sparse pairs at small N, regardless of run "
            "count, so small-N binary data uses the Bayesian paired model "
            "instead."
        ),
    ),
    AutoAnalyzeRule(
        data_kind="binary", max_n=None,
        pairwise_method="tango",
        robustness_method_single_run="wilson",
        robustness_method_seeded="nig_nested",
        reason="N >= 50 binary data: Tango pairwise, Wilson/NIG-nested marginal.",
    ),
    AutoAnalyzeRule(
        data_kind="bounded_01", max_n=None,
        pairwise_method="t_interval",
        robustness_method_single_run="nig",
        robustness_method_seeded="nig_nested",
        reason="Continuous [0, 1]-bounded data (e.g. normalised accuracy, ROUGE).",
    ),
    AutoAnalyzeRule(
        data_kind="continuous", max_n=None,
        pairwise_method="t_interval",
        robustness_method_single_run="t_interval",
        robustness_method_seeded="t_interval",
        reason="Arbitrary numeric data with no known bounds.",
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
    data_kind : "binary", "bounded_01", or "continuous"
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
        pairwise_method="bootstrap_t",
        robustness_method="bootstrap_t",
        reason=(
            "Numeric [0, 1]-bounded data: PPI-corrected studentized bootstrap "
            "(see evalstats.tests._ppi_paired_bootstrap_t / "
            "_ppi_single_bootstrap_t). The non-aligned default (t_interval) "
            "has no PPI-corrected form."
        ),
    ),
    PPIAutoMethodRule(
        data_kind="continuous",
        pairwise_method="bootstrap_t",
        robustness_method="bootstrap_t",
        reason="Arbitrary numeric data: same PPI-corrected studentized bootstrap as bounded_01.",
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
