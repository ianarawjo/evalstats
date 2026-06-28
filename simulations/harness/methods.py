"""Canonical CI-method registry, shared across cases.

Ported from the ``_METHOD_COLORS`` dict and method-name constants duplicated
across ``sim_compare_boot.py`` / ``sim_compare_boot_nested.py`` /
``sim_dove.py`` / ``sim_tango_real.py``. Each method is one ``Method``
instance carrying its name and plot color; callers work with the instance
directly (``method.name``, ``method.color``) instead of threading a bare
string through one lookup table per case. Centralizing this means a method
keeps the same name and the same plot color in every case's tables and
figures (``cases/ci_single.py`` today; ``cases/ci_paired.py`` and
``cases/ci_nested.py`` once ported), instead of each case file redefining
its own copy that can silently drift out of sync.

Add new Method instances/groupings here as new cases are ported -- don't
predeclare methods for cases that don't exist as code yet.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_METHOD_COLOR = "#333333"


@dataclass(frozen=True)
class Method:
    name: str
    color: str = DEFAULT_METHOD_COLOR

    def __str__(self) -> str:
        return self.name

    def __format__(self, format_spec: str) -> str:
        return format(self.name, format_spec)


# ---------------------------------------------------------------------------
# Bootstrap-family methods (apply to single-sample means and paired diffs,
# flat or nested)
# ---------------------------------------------------------------------------
BOOTSTRAP = Method("bootstrap", "#1f77b4")
BCA = Method("bca", "#2ca02c")
BAYES_BOOTSTRAP = Method("bayes_bootstrap", "#ff7f0e")
SMOOTH_BOOTSTRAP = Method("smooth_bootstrap", "#9467bd")
BOOTSTRAP_T = Method("bootstrap_t", "#d62728")
BOOTSTRAP_METHODS = [BOOTSTRAP, BCA, BAYES_BOOTSTRAP, SMOOTH_BOOTSTRAP, BOOTSTRAP_T]

# ---------------------------------------------------------------------------
# Single-sample extras
# ---------------------------------------------------------------------------
T_INTERVAL = Method("t_interval", "#8c564b")
WILSON = Method("wilson", "#e377c2")
JEFFREYS = Method("jeffreys", "#e9cd14")
WALD = Method("wald", "#7f7f7f")
CLOPPER_PEARSON = Method("clopper_pearson", "#bcbd22")
BAYES_SINGLE = Method("bayes_indep", "#17becf")
BETA = Method("beta", "#f0027f")
LOGIT_T = Method("logit_t", "#a6761d")
NIG = Method("nig", "#888888")
EL = Method("el", "#00441b")

BINARY_SINGLE_EXTRA_METHODS = [WILSON, JEFFREYS, WALD, CLOPPER_PEARSON, BAYES_SINGLE]
CONTINUOUS_EXTRA_METHODS = [BETA, LOGIT_T, NIG, EL]

# ---------------------------------------------------------------------------
# Paired (pairwise-difference) extras -- for cases/ci_paired.py once ported
# ---------------------------------------------------------------------------
NEWCOMBE = Method("newcombe_score", "#aec7e8")
TANGO = Method("tango_score")  # no color in the legacy palette; uses the default
BAYES_PAIR_INDEP = Method("bayes_indep_comp", "#ffbb78")
BAYES_PAIR_PAIRED = Method("bayes_paired_comp", "#98df8a")
PAIRWISE_EXTRA_METHODS = [T_INTERVAL, NIG, EL]
BINARY_PAIRWISE_EXTRA_METHODS = [NEWCOMBE, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED]

# ---------------------------------------------------------------------------
# Registry -- canonical ordering for tables/legends, and name -> Method lookup
# ---------------------------------------------------------------------------
REPORT_METHOD_ORDER: list[Method] = BOOTSTRAP_METHODS + [
    T_INTERVAL, WILSON, JEFFREYS, NEWCOMBE, TANGO,
    WALD, CLOPPER_PEARSON, BAYES_SINGLE, BAYES_PAIR_INDEP, BAYES_PAIR_PAIRED,
] + CONTINUOUS_EXTRA_METHODS

METHODS_BY_NAME: dict[str, Method] = {m.name: m for m in REPORT_METHOD_ORDER}


def get_method(name: str) -> Method:
    """Look up a Method by name, falling back to a default-colored Method for unknown names."""
    return METHODS_BY_NAME.get(name, Method(name))


def get_method_color(name: str) -> str:
    return get_method(name).color


def order_present_methods(present_names: set[str]) -> list[Method]:
    """Filter REPORT_METHOD_ORDER down to methods actually present, preserving canonical order."""
    return [m for m in REPORT_METHOD_ORDER if m.name in present_names]
