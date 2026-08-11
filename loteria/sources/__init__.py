"""Indexes that enumerate available prize-list PDFs.

Both backends answer the same question — "which drawings exist and where do I fetch
them?" — so they are interchangeable behind :class:`Source`. Enumerating an index is
essential rather than convenient: URLs cannot be constructed from the drawing number
because the ``/uploads/YYYY/MM/`` folder is the upload month, which routinely
disagrees with the drawing month, and because special draws carry ``S``/``X`` suffixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

SITE = "https://loteriasdepuertorico.pr.gov"


@dataclass(frozen=True)
class DrawingRef:
    """One fetchable candidate for one drawing.

    A drawing may have several refs — the Wayback Machine often holds multiple
    captures of the same URL, and only some are complete — so the downloader treats
    refs sharing a ``canonical_id`` as ordered fallbacks.
    """

    canonical_id: str
    year: int
    seq: int
    suffix: str
    filename: str
    url: str
    source: str
    published: str | None = None
    wayback_timestamp: str | None = None
    size_hint: int | None = None
    low_confidence: bool = False

    @property
    def draw_key(self) -> tuple[int, int]:
        return (self.year, self.seq)

    def describe(self) -> str:
        where = f"{self.source}"
        if self.wayback_timestamp:
            where += f"@{self.wayback_timestamp}"
        return f"{self.canonical_id} [{where}] {self.filename}"


class Source(Protocol):
    """Anything that can list available drawings."""

    name: str

    def index(self) -> list[DrawingRef]:
        """Return every drawing this source can offer, best candidate first."""
        ...


# The live site is the canonical origin, so its copies win over archived ones.
SOURCE_PRIORITY = {"live": 0, "wayback": 1}


def rank(ref: DrawingRef) -> tuple[int, int, int]:
    """Sort key deciding which candidate to try first for a given drawing.

    Prefer the canonical origin, then a clean filename over a duplicate/typo
    artifact, then the largest known capture — since truncated Wayback captures are
    always smaller than the complete file.
    """
    return (
        SOURCE_PRIORITY.get(ref.source, 9),
        1 if ref.low_confidence else 0,
        -(ref.size_hint or 0),
    )
