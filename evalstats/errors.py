"""Typed exceptions raised by :func:`evalstats.compare`.

Each subclasses ``ValueError``, so existing ``except ValueError`` handlers keep
working.
"""

from __future__ import annotations

from typing import Optional

from evalstats.config import MIN_SAMPLE_FLOOR

#: Fewest items per compared entity that ``compare()`` reports statistics for.
MIN_ITEMS: int = MIN_SAMPLE_FLOOR


class InsufficientItemsError(ValueError):
    """Fewer items per compared entity than :data:`MIN_ITEMS`."""

    def __init__(self, message: str, *, n_items: int, min_items: int = MIN_ITEMS):
        super().__init__(message)
        self.n_items = n_items
        self.min_items = min_items


class MissingCellsError(ValueError):
    """The paired design has (entity, item) cells with no score.

    ``missing`` lists up to the first 1000 ``(entity, item)`` pairs;
    ``n_missing`` is the exact count.
    """

    def __init__(self, message: str, *, missing: list[tuple[str, str]], n_missing: int):
        super().__init__(message)
        self.missing = missing
        self.n_missing = n_missing


class AmbiguousLabelsError(ValueError):
    """Two-factor cells whose names join to the same ``"model / prompt"`` label."""

    def __init__(self, message: str, *, labels: list[str]):
        super().__init__(message)
        self.labels = labels


class TooFewGroupsError(ValueError):
    """Fewer than two levels to compare."""

    def __init__(self, message: str, *, n_groups: int, factor: Optional[str] = None):
        super().__init__(message)
        self.n_groups = n_groups
        self.factor = factor
