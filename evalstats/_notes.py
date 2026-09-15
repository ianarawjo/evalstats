"""Structured analysis notes, collected per call without touching warning filters.

``warn()`` emits a normal ``warnings.warn`` and, inside ``collect_notes()``,
also records a :class:`Note`. The collector lives in a ``ContextVar``, so
concurrent ``compare()`` calls in different threads each see only their own
notes.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional, Sequence

Severity = Literal["info", "warning"]

_collector: ContextVar[Optional[list]] = ContextVar("evalstats_notes", default=None)
_suppressed: ContextVar[frozenset] = ContextVar("evalstats_suppressed_notes", default=frozenset())


@dataclass(frozen=True)
class Note:
    """One note about how an analysis ran."""

    code: str
    message: str
    severity: Severity = "warning"
    entities: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "entities": list(self.entities),
        }


def warn(
    message: str,
    *,
    code: str,
    entities: Sequence[str] = (),
    severity: Severity = "warning",
    category: type[Warning] = UserWarning,
    stacklevel: int = 2,
    emit: bool = True,
) -> None:
    """Record a note (when collecting) and emit it as a warning."""
    if code in _suppressed.get():
        return
    notes = _collector.get()
    if notes is not None:
        note = Note(code=code, message=message, severity=severity,
                    entities=tuple(str(e) for e in entities))
        if note not in notes:
            notes.append(note)
    if emit:
        warnings.warn(message, category, stacklevel=stacklevel + 1)


@contextmanager
def collect_notes() -> Iterator[list]:
    """Collect notes raised inside the block into the yielded list."""
    token = _collector.set([])
    try:
        yield _collector.get()
    finally:
        _collector.reset(token)


@contextmanager
def suppress_notes(*codes: str) -> Iterator[None]:
    """Neither record nor emit notes with these codes inside the block."""
    token = _suppressed.set(_suppressed.get() | frozenset(codes))
    try:
        yield
    finally:
        _suppressed.reset(token)
