"""Display-free views of an analysis: entity order, rank bands, verdicts.

``.summary()`` prints these and ``ComparisonResult.rank_bands()`` /
``to_dict()`` return them, so both come from one definition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np

from ..config import get_alpha_ci
from .paired import PairedDiffResult, PairwiseMatrix

if TYPE_CHECKING:
    from .types import MultiModelBenchmark

BandCriterion = Literal["simultaneous_ci_excludes_zero", "corrected_p_below_alpha"]

VERDICT_LIKELY_BEST = "likely_best"
VERDICT_TIED_FOR_BEST = "tied_for_best"
VERDICT_DROP_OFF = "significant_drop_off"

CELL_LABEL_SEP = " / "


def leaderboard_order(means: Sequence[float]) -> list[int]:
    """Indices by descending mean; tied means keep their data order."""
    return [int(i) for i in np.argsort(-np.asarray(means, dtype=float), kind="stable")]


def band_criterion(pairwise: PairwiseMatrix) -> BandCriterion:
    """The rule deciding whether a pair differs: the simultaneous CI when there is one."""
    if pairwise.simultaneous_ci_method is not None:
        return "simultaneous_ci_excludes_zero"
    return "corrected_p_below_alpha"


def pair_significant(
    result: PairedDiffResult,
    *,
    criterion: BandCriterion,
    alpha: float,
) -> bool:
    """Whether a pair differs under ``criterion``."""
    if criterion == "simultaneous_ci_excludes_zero":
        return bool(result.ci_low > 0 or result.ci_high < 0)
    return result.p_value is not None and bool(result.p_value < alpha)


def critical_difference_groups(
    pairwise: PairwiseMatrix,
    labels_sorted: list[str],
    *,
    alpha: Optional[float] = None,
) -> list[list[str]]:
    """Contiguous, maximal rank bands whose members are pairwise indistinguishable."""
    if alpha is None:
        alpha = get_alpha_ci()
    if len(labels_sorted) < 2:
        return []
    criterion = band_criterion(pairwise)

    def _all_pairs_nonsignificant(group_labels: list[str]) -> bool:
        for i in range(len(group_labels)):
            for j in range(i + 1, len(group_labels)):
                try:
                    result = pairwise.get(group_labels[i], group_labels[j])
                except KeyError:
                    return False
                if pair_significant(result, criterion=criterion, alpha=alpha):
                    return False
        return True

    n_labels = len(labels_sorted)
    candidate_groups: list[list[str]] = []
    for start_idx in range(n_labels - 1):
        best_group: Optional[list[str]] = None
        for end_idx in range(start_idx + 1, n_labels):
            group = labels_sorted[start_idx : end_idx + 1]
            if _all_pairs_nonsignificant(group):
                best_group = group
            else:
                break
        if best_group is not None:
            candidate_groups.append(best_group)

    def _is_contiguous_subsequence(smaller: list[str], larger: list[str]) -> bool:
        if len(smaller) >= len(larger):
            return False
        for start in range(len(larger) - len(smaller) + 1):
            if larger[start : start + len(smaller)] == smaller:
                return True
        return False

    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in candidate_groups:
        if any(_is_contiguous_subsequence(group, other) for other in candidate_groups if other is not group):
            continue
        key = tuple(group)
        if key not in seen:
            seen.add(key)
            deduped.append(group)
    return deduped


def clear_winner(
    pairwise: PairwiseMatrix,
    labels_sorted: list[str],
    *,
    alpha: Optional[float] = None,
) -> Optional[str]:
    """The unique label that significantly beats every other label, if any."""
    if alpha is None:
        alpha = get_alpha_ci()
    if len(labels_sorted) < 2:
        return None
    criterion = band_criterion(pairwise)
    winners: list[str] = []
    for candidate in labels_sorted:
        beats_all = True
        for other in labels_sorted:
            if other == candidate:
                continue
            try:
                result = pairwise.get(candidate, other)
            except KeyError:
                beats_all = False
                break
            if not (pair_significant(result, criterion=criterion, alpha=alpha) and result.point_diff > 0):
                beats_all = False
                break
        if beats_all:
            winners.append(candidate)
            if len(winners) > 1:
                return None
    return winners[0] if len(winners) == 1 else None


def significance_bands(
    pairwise: PairwiseMatrix,
    labels_sorted: list[str],
    *,
    alpha: Optional[float] = None,
) -> dict[str, int]:
    """Band number (1 = top) for each label, as the executive summary shows it.

    Labels in the same maximal non-significant rank band share a band, and
    band numbers never decrease down the sorted list. From band 2 on,
    overlapping maximal bands (A~B and B~C but not A~C, the transitivity
    caveat of critical-difference diagrams, Demsar 2006) are merged into one
    band, since each label carries a single number.

    Band 1 is not extended this way. It becomes the "tied for best" verdict,
    so it holds only the single maximal band containing the top label: labels
    provably indistinguishable from the top performer, not merely reachable
    from it through a chain of non-significant neighbours.
    """
    if alpha is None:
        alpha = get_alpha_ci()
    if not labels_sorted:
        return {}
    groups = critical_difference_groups(pairwise, labels_sorted, alpha=alpha)
    rank_of = {label: i for i, label in enumerate(labels_sorted)}

    reach = {label: rank_of[label] for label in labels_sorted}
    for group in groups:
        end_idx = max(rank_of[l] for l in group)
        for label in group:
            reach[label] = max(reach[label], end_idx)

    top_reach = 0
    for group in groups:
        if labels_sorted[0] in group:
            top_reach = max(top_reach, max(rank_of[l] for l in group))

    bands: dict[str, int] = {}
    band = 0
    current_end_idx = -1
    for idx, label in enumerate(labels_sorted):
        if idx > current_end_idx:
            band += 1
            current_end_idx = top_reach if band == 1 else reach[label]
        elif band > 1:
            current_end_idx = max(current_end_idx, reach[label])
        bands[label] = band
    return bands


def verdict(label: str, bands: dict[str, int], labels_sorted: list[str]) -> tuple[str, list[str]]:
    """``(code, tied_with)`` for one label: its verdict and the other band-1 labels."""
    if bands.get(label) != 1:
        return VERDICT_DROP_OFF, []
    tied_with = [l for l in labels_sorted if bands.get(l) == 1 and l != label]
    return (VERDICT_TIED_FOR_BEST if tied_with else VERDICT_LIKELY_BEST), tied_with


def verdict_text(code: str, tied_with: Sequence[str], *, max_name_len: Optional[int] = None) -> str:
    """The executive summary's wording for a verdict."""
    if code == VERDICT_DROP_OFF:
        return "Significant drop-off"
    if code == VERDICT_LIKELY_BEST:
        return "Likely best"
    if len(tied_with) == 1:
        name = tied_with[0]
        if max_name_len is not None and len(name) > max_name_len:
            name = name[: max_name_len - 1] + "…"
        return f"Tied with {name} as best"
    return f"Tied with {len(tied_with)} others as best"


def json_safe(obj):
    """Copy of ``obj`` holding only JSON types; non-finite floats become ``None``."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def cell_levels(benchmark: "MultiModelBenchmark") -> dict[str, tuple[str, str]]:
    """``(model, template)`` for each flat cross-model label."""
    return {
        f"{m}{CELL_LABEL_SEP}{t}": (str(m), str(t))
        for m in benchmark.model_labels
        for t in benchmark.template_labels
    }
