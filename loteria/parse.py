"""Extract structured results from a prize-list PDF.

The PDFs are vector text, but every glyph is positioned individually, so
``extract_text()`` returns kerning debris like ``"4 7 083"`` and ``"NÚMERO"`` doubled
into ``"NNÚÚMMEERROO"``. Everything here works from word coordinates instead.

Layout: a wide table of 14 column-groups, each a ``NÚMERO`` / ``PREMIO`` pair, read as
cells. A cell's rightmost numeric token is the prize and whatever precedes it is the
winning number, which handles all three separator styles the template uses —
``2429C-- 100``, ``37378 - 1200``, and the top prize's bare ``47083 250000``.

Column boundaries come from the actual ``NÚMERO`` header positions on each page, not
from a uniform pitch: the template's spacing alternates between 78.5 and 79.5 points,
so an assumed pitch accumulates enough drift across 14 columns to throw prizes into
the neighbouring column.

Page-0 headers are stamped twice, doubling every character (``DDEE`` for ``DE``), so
header words are de-doubled before being read. The table itself is not doubled.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

# A prize cell's number, optionally carrying a category letter.
NUMBER_RE = re.compile(r"^(\d{1,5})([A-Z]?)$")

# Some issues print "6- 49088" with the number pushed off the dash.
MAJORS_RE = re.compile(r"(\d{1,2})\s*-\s*(\d{4,6})")
LAST_DIGIT_RE = re.compile(r"ÚLTIMA CIFRA DEL PRIMER PREMIO\s*(\d)")
SERIES_RE = re.compile(r"series:\s*([A-Z](?:\s*,\s*[A-Z])*(?:\s*y\s*[A-Z])?)", re.I)
# The second "de" is missing in some issues ("9 DE FEBRERO 2023"), so it's optional.
EXPIRES_RE = re.compile(r"caduca el\s+(\d{1,2})\s+de\s+(\w+)\s+(?:de\s+)?(\d{4})", re.I)
CELEBRATED_RE = re.compile(r"CELEBRADO EL\s+(\d{1,2})\s+DE\s+(\w+)\s+(?:DE\s+)?(\d{4})", re.I)
DRAW_DATE_RE = re.compile(r"(\d{1,2})\s+DE\s+(\w+)\s+(?:DE\s+)?(\d{4})", re.I)
NEXT_DRAW_RE = re.compile(r"PRÓXIMO SORTEO", re.I)

MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

# Observed prize categories. Every category except `directo` is derived by rule from the
# major prizes, which is what :meth:`ParsedDrawing.invariants` checks.
CATEGORIES = {
    "": "directo",       # the number itself was pulled from the tumbler
    "A": "aproximación",  # numbers immediately above/below a major prize (2 per major)
    "C": "centena",       # rest of the hundred containing a major prize (99 per major)
    "T": "terminación",   # shares the major prize's last three digits (49 per major)
}

# There are always exactly this many numbered balls.
BALLS = 50_000

# Prize slots each major prize awards in each derived category.
CENTENA_PER_MAJOR = 99
TERMINACION_PER_MAJOR = 49


def derived_prize_sets(majors: list[int]) -> dict[str, tuple[set[int], int]]:
    """Predict the derived prize categories from the major prize numbers.

    Each rule applies **per major** and the results are unioned. That distinction is
    load-bearing: when two majors land in the same hundred-block, each awards centena to
    the other 99 numbers in it, so the union covers all 100 and the two majors each win
    centena off the other's prize. Subtracting all majors globally instead would predict
    98 and wrongly condemn a perfectly good drawing.

    Returns ``{category: (winning_numbers, prize_slots)}``. The slot count carries
    multiplicity — an overlapping number legitimately collects two prizes — so slots
    stay at 99/49/2 per major even when the sets overlap.
    """
    centena: set[int] = set()
    terminacion: set[int] = set()
    aproximacion: set[int] = set()
    centena_slots = terminacion_slots = aproximacion_slots = 0
    majors_set = set(majors)

    for major in majors:
        block_start = ((major - 1) // 100) * 100
        block = {block_start + offset for offset in range(1, 101)} - {major}
        centena |= block
        centena_slots += len(block)

        last_three = major % 1000
        tail = {
            thousand * 1000 + last_three
            for thousand in range(0, BALLS // 1000 + 1)
            if 1 <= thousand * 1000 + last_three <= BALLS
        } - {major}
        terminacion |= tail
        terminacion_slots += len(tail)

        neighbours = {x for x in (major - 1, major + 1) if 1 <= x <= BALLS} - majors_set
        aproximacion |= neighbours
        aproximacion_slots += len(neighbours)

    return {
        "C": (centena, centena_slots),
        "T": (terminacion, terminacion_slots),
        "A": (aproximacion, aproximacion_slots),
    }

ROW_TOLERANCE = 4.0
X_TOLERANCE = 1.2


@dataclass
class Prize:
    number: int
    category: str
    amount: int


@dataclass
class ParsedDrawing:
    drawing_label: str            # as printed, e.g. "326" or "196X"
    draw_date: datetime.date | None
    celebrated_on: datetime.date | None
    expires_on: datetime.date | None
    series: list[str]
    next_drawing: str | None
    first_prize_last_digit: int | None
    majors: list[tuple[int, int]]  # (rank, number)
    prizes: list[Prize]
    pages: int
    skipped_cells: list[str] = field(default_factory=list)

    @property
    def category_counts(self) -> dict[str, int]:
        """Row counts per category."""
        counts: dict[str, int] = {}
        for prize in self.prizes:
            counts[prize.category] = counts.get(prize.category, 0) + 1
        return counts

    @property
    def distinct_category_counts(self) -> dict[str, int]:
        """Distinct winning numbers per category.

        Lower than the slot count when two majors share a hundred-block or a
        terminación class, since the overlapping numbers collect two prizes each.
        """
        seen: dict[str, set[int]] = {}
        for prize in self.prizes:
            seen.setdefault(prize.category, set()).add(prize.number)
        return {category: len(numbers) for category, numbers in seen.items()}

    @property
    def duplicate_entries(self) -> list[tuple[int, str]]:
        """(number, category) pairs printed more than once.

        Legitimate, not an error: it means two major prizes shared the number's
        hundred-block or terminación class, so it genuinely wins that category twice.
        """
        counts: dict[tuple[int, str], int] = {}
        for prize in self.prizes:
            key = (prize.number, prize.category)
            counts[key] = counts.get(key, 0) + 1
        return sorted(key for key, count in counts.items() if count > 1)

    def invariants(self) -> list[str]:
        """Return a list of consistency problems; empty means the parse is sound.

        The prize structure is rigidly determined by the number of major prizes, which
        makes these strong checks — a misread column or a dropped row breaks them.
        """
        problems: list[str] = []
        majors = [number for _, number in self.majors]
        n = len(majors)

        if n == 0:
            problems.append("no major prizes found in header")
            return problems

        # The table itself says how many majors there should be: centena awards exactly
        # 99 slots per major regardless of how they overlap.
        centena_rows = self.category_counts.get("C", 0)
        if centena_rows % CENTENA_PER_MAJOR == 0:
            implied = centena_rows // CENTENA_PER_MAJOR
            if implied != n:
                problems.append(
                    f"table implies {implied} major prizes ({centena_rows} centena rows) "
                    f"but {n} were read from the header"
                )

        # Compare the derived categories against what the rules predict, exactly.
        predicted = derived_prize_sets(majors)
        for letter, (want_set, want_slots) in predicted.items():
            got = [p.number for p in self.prizes if p.category == letter]
            got_set = set(got)
            if got_set != want_set:
                missing = sorted(want_set - got_set)
                extra = sorted(got_set - want_set)
                detail = []
                if missing:
                    detail.append(f"{len(missing)} missing e.g. {missing[:3]}")
                if extra:
                    detail.append(f"{len(extra)} unexpected e.g. {extra[:3]}")
                problems.append(
                    f"category {letter} ({CATEGORIES[letter]}) does not match the rule: "
                    + "; ".join(detail)
                )
            elif len(got) != want_slots:
                problems.append(
                    f"category {letter} ({CATEGORIES[letter]}): {len(got)} prize slots, "
                    f"rule predicts {want_slots}"
                )

        # Every headline number must also appear in the table with a substantial prize.
        table = {(p.number, p.category): p.amount for p in self.prizes}
        for rank, number in self.majors:
            amount = table.get((number, ""))
            if amount is None:
                problems.append(f"major #{rank} {number:05d} missing from prize table")
            elif amount < 1000:
                problems.append(f"major #{rank} {number:05d} has implausible prize {amount}")

        if self.first_prize_last_digit is not None and self.majors:
            top = self.majors[0][1]
            if top % 10 != self.first_prize_last_digit:
                problems.append(
                    f"stated last digit {self.first_prize_last_digit} "
                    f"contradicts first prize {top:05d}"
                )

        return problems


def parse_pdf(path: Path) -> ParsedDrawing:
    """Parse one prize-list PDF into a :class:`ParsedDrawing`."""
    header_lines: list[str] = []
    prizes: list[Prize] = []
    skipped: list[str] = []

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for index, page in enumerate(pdf.pages):
            words = page.extract_words(x_tolerance=X_TOLERANCE, y_tolerance=2)
            bounds, header_top = _column_bounds(words)
            if bounds is None:
                continue

            if index == 0:
                header_lines.extend(_header_lines(words, header_top))
            # The "ÚLTIMA CIFRA" note sits below the table, on whichever page ends it.
            note = _trailing_note(words)
            if note:
                header_lines.append(note)

            page_prizes, page_skipped = _read_table(words, bounds, header_top)
            prizes.extend(page_prizes)
            skipped.extend(page_skipped)

    header = " \n".join(header_lines)
    label = _drawing_label(header, path)
    return ParsedDrawing(
        drawing_label=label,
        draw_date=_draw_date(header),
        celebrated_on=_spanish_date(CELEBRATED_RE.search(header)),
        expires_on=_spanish_date(EXPIRES_RE.search(header)),
        series=_series(header),
        next_drawing=_next_drawing(header_lines, label),
        first_prize_last_digit=_first_digit(header),
        majors=[(int(r), int(n)) for r, n in MAJORS_RE.findall(_majors_line(header_lines))],
        prizes=prizes,
        pages=page_count,
        skipped_cells=skipped,
    )


# -- geometry -----------------------------------------------------------------


def _column_bounds(words: list[dict]) -> tuple[list[float] | None, float]:
    """Derive column left edges from the repeated ``NÚMERO`` header cells."""
    headers = [w for w in words if w["text"] == "NÚMERO"]
    if len(headers) < 2:
        return None, 0.0
    edges = sorted(w["x0"] for w in headers)
    header_top = min(w["top"] for w in headers)
    # Nudge left so a token sitting marginally left of its header still lands right.
    return [e - 3.0 for e in edges], header_top


def _cluster_rows(words: list[dict]) -> list[list[dict]]:
    """Group words into table rows by vertical position."""
    rows: list[list[dict]] = []
    anchor = None
    for word in sorted(words, key=lambda w: w["top"]):
        if anchor is None or word["top"] - anchor > ROW_TOLERANCE:
            rows.append([])
            anchor = word["top"]
        rows[-1].append(word)
    return rows


def _read_table(
    words: list[dict], bounds: list[float], header_top: float
) -> tuple[list[Prize], list[str]]:
    """Read every prize cell on a page."""
    body = [w for w in words if w["top"] > header_top + 5]
    prizes: list[Prize] = []
    skipped: list[str] = []

    for row in _cluster_rows(body):
        cells: dict[int, list[dict]] = {}
        for word in row:
            column = _column_of(word["x0"], bounds)
            cells.setdefault(column, []).append(word)

        for tokens in cells.values():
            prize, raw = _read_cell(tokens)
            if prize is not None:
                prizes.append(prize)
            elif raw:
                skipped.append(raw)

    return prizes, skipped


def _column_of(x: float, bounds: list[float]) -> int:
    """Index of the column containing ``x``."""
    column = 0
    for index, edge in enumerate(bounds):
        if x >= edge:
            column = index
        else:
            break
    return column


def _read_cell(tokens: list[dict]) -> tuple[Prize | None, str]:
    """Turn one table cell into a prize.

    The rightmost numeric token is the amount; everything to its left is the winning
    number. Section labels (``CENTESIMALES``, ``CIENTOS``) and thousands markers
    (``18 MIL``) fail this shape and are reported as skipped instead.
    """
    parts = [t["text"].replace("-", "").strip() for t in sorted(tokens, key=lambda t: t["x0"])]
    parts = [p for p in parts if p]
    raw = " ".join(parts)
    if len(parts) < 2 or not parts[-1].isdigit():
        return None, raw

    match = NUMBER_RE.match("".join(parts[:-1]))
    if match is None:
        return None, raw

    return Prize(int(match.group(1)), match.group(2), int(parts[-1])), raw


# -- header -------------------------------------------------------------------


def _header_lines(words: list[dict], header_top: float | None) -> list[str]:
    """Read header text above the table, undoing page 0's double-stamping."""
    selected = [w for w in words if header_top is None or w["top"] < header_top]
    out = []
    for row in _cluster_rows(selected):
        text = " ".join(_dedouble(w["text"]) for w in sorted(row, key=lambda w: w["x0"]))
        if text.strip():
            out.append(text)
    return out


def _trailing_note(words: list[dict]) -> str | None:
    """Find the ``ÚLTIMA CIFRA DEL PRIMER PREMIO`` line printed below the table."""
    anchor = next((w for w in words if _dedouble(w["text"]).upper() == "CIFRA"), None)
    if anchor is None:
        return None
    row = [w for w in words if abs(w["top"] - anchor["top"]) <= ROW_TOLERANCE]
    return " ".join(_dedouble(w["text"]) for w in sorted(row, key=lambda w: w["x0"]))


def _dedouble(text: str) -> str:
    """``DDEE`` -> ``DE``. Leaves normal words untouched."""
    if len(text) < 2 or len(text) % 2:
        return text
    if all(text[i] == text[i + 1] for i in range(0, len(text), 2)):
        return text[::2]
    return text


def _majors_line(header_lines: list[str]) -> str:
    return next((line for line in header_lines if "MAYORES" in line), "")


def _drawing_label(header: str, path: Path) -> str:
    match = re.search(r"Sorteo\s+(\d{2,4}[A-Z]?)", header, re.I)
    if match:
        return match.group(1)
    # Fall back to the filename, which encodes the same identifier.
    match = re.search(r"Lista-Oficial-Web-(\d{2,4}[A-Za-z]?)", path.name)
    return match.group(1).upper() if match else path.stem


def _draw_date(header: str) -> datetime.date | None:
    """The scheduled drawing date — the first date printed in the header."""
    for match in DRAW_DATE_RE.finditer(header):
        # Skip the expiry and celebrated-on dates, which carry their own preamble.
        preceding = header[max(0, match.start() - 24):match.start()].lower()
        if "caduca" in preceding or "celebrado" in preceding:
            continue
        return _spanish_date(match)
    return None


def _spanish_date(match: re.Match | None) -> datetime.date | None:
    if match is None:
        return None
    day, month_name, year = match.group(1), match.group(2).lower(), match.group(3)
    month = MONTHS.get(month_name)
    if month is None:
        return None
    try:
        return datetime.date(int(year), month, int(day))
    except ValueError:
        return None


def _series(header: str) -> list[str]:
    """Series letters from "válida para las series: A, B, C y D".

    Case is load-bearing: the Spanish connector "y" must not be mistaken for a series.
    """
    match = SERIES_RE.search(header)
    if not match:
        return []
    return re.findall(r"\b([A-Z])\b", match.group(1))


def _next_drawing(header_lines: list[str], current: str) -> str | None:
    """The next drawing's label, printed alone on a line under ``PRÓXIMO SORTEO``.

    It has to be a line holding nothing else — the caption's neighbourhood also
    contains "SORTEO ORDINARIO 326", which would otherwise match the current drawing.
    """
    start = next((i for i, line in enumerate(header_lines) if NEXT_DRAW_RE.search(line)), None)
    if start is None:
        return None
    for line in header_lines[start:start + 4]:
        candidate = line.strip()
        if re.fullmatch(r"\d{3}[A-Z]?", candidate) and candidate != current:
            return candidate
    return None


def _first_digit(header: str) -> int | None:
    match = LAST_DIGIT_RE.search(header)
    return int(match.group(1)) if match else None
