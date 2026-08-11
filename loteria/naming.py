"""Decode ``Lista-Oficial-Web-*.pdf`` filenames into stable drawing identifiers.

The site's numbering looks like a running counter that jumps by 10, but it isn't.
The number is ``<SS><Y>``: ``SS`` is the weekly draw sequence *within the year* and
``Y`` is the last digit of the year. So ``326`` is draw 32 of 2026 and ``371`` is
draw 37 of 2021. Incrementing the weekly draw shifts the printed number by 10
because the year digit is pinned in the ones place.

A trailing letter marks a non-ordinary draw (``S`` special, ``X`` extraordinary),
and the archive also holds WordPress duplicate artifacts (``-1``, ``-copy``) and
the odd typo (``Lista-Oficial-Web-2831.pdf``).
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass

# Trailing ``-1`` / ``-copy`` are WordPress duplicate-upload artifacts, not part of the id.
FILENAME_RE = re.compile(
    r"Lista-Oficial-Web-(?P<num>\d{3,4})(?P<suffix>[A-Za-z]*?)"
    r"(?:-(?P<dup>\d+|copy))?\.pdf$",
    re.IGNORECASE,
)

_UPLOAD_YEAR_RE = re.compile(r"/uploads/(?P<year>\d{4})/")

# Draws are weekly, so a year holds at most 53.
MAX_SEQ = 53


@dataclass(frozen=True)
class ParsedName:
    """A filename decoded into the drawing it represents."""

    year: int
    seq: int
    suffix: str
    low_confidence: bool
    filename: str

    @property
    def canonical_id(self) -> str:
        """Stable, sortable identity for a drawing, e.g. ``2026-032`` or ``2026-019X``."""
        return f"{self.year}-{self.seq:03d}{self.suffix}"

    @property
    def draw_key(self) -> tuple[int, int]:
        """(year, seq) ignoring suffix — the slot a drawing occupies in the weekly calendar."""
        return (self.year, self.seq)


def hint_year_from_url(url: str) -> int | None:
    """Pull the ``/uploads/YYYY/`` upload year from a URL, if present.

    This is the *upload* year, which disambiguates the single year digit. It is not
    reliable as a drawing date — the upload month frequently disagrees with the draw
    month — but it is always within a year of the truth.
    """
    match = _UPLOAD_YEAR_RE.search(url)
    return int(match.group("year")) if match else None


def resolve_year(year_digit: str, hint_year: int | None) -> int:
    """Expand a single year digit to a full year using the closest plausible decade."""
    digit = int(year_digit)
    if hint_year is None:
        # Without a hint, take the most recent year ending in this digit.
        current = datetime.date.today().year
        return current - (current - digit) % 10
    base = hint_year - (hint_year % 10) + digit
    return min((base - 10, base, base + 10), key=lambda y: abs(y - hint_year))


def parse_filename(name: str, hint_year: int | None = None) -> ParsedName | None:
    """Decode a PDF filename, or return ``None`` if it isn't a prize list."""
    match = FILENAME_RE.search(name)
    if not match:
        return None

    num = match.group("num")
    suffix = (match.group("suffix") or "").upper()
    duplicate_artifact = match.group("dup") is not None

    # Primary reading: the last digit is the year. A 4-digit number can't be that
    # (it would imply seq >= 100, impossible for weekly draws), so also try reading
    # it as a 3-digit id with a stray trailing character.
    readings: list[tuple[int, str, bool]] = [(int(num[:-1]), num[-1], False)]
    if len(num) == 4:
        readings.append((int(num[:2]), num[2], True))

    for seq, year_digit, malformed in readings:
        if 1 <= seq <= MAX_SEQ:
            return ParsedName(
                year=resolve_year(year_digit, hint_year),
                seq=seq,
                suffix=suffix,
                low_confidence=duplicate_artifact or malformed,
                filename=name,
            )

    # No plausible reading. Keep the primary one so the file is still cataloged,
    # but flag it so it can never displace a clean match for the same id.
    seq, year_digit, _ = readings[0]
    return ParsedName(
        year=resolve_year(year_digit, hint_year),
        seq=seq,
        suffix=suffix,
        low_confidence=True,
        filename=name,
    )


def parse_url(url: str) -> ParsedName | None:
    """Decode the filename at the end of a URL, using its upload year as the hint."""
    filename = url.rstrip("/").rsplit("/", 1)[-1]
    return parse_filename(filename, hint_year_from_url(url))
