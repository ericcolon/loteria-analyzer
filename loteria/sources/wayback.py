"""Index the Wayback Machine for prize lists the site has since deleted.

The live site retains roughly six months. The CDX API turns up ~150 additional
distinct files reaching back to drawing-year 2021, which is the difference between a
few months of data and a few years.

Two details matter. Captures must be filtered to ``statuscode:200``, because the
archive is full of crawls that hit already-deleted files and recorded the 404. And
each URL's captures are yielded largest-first: truncated captures are always smaller
than the complete file, so size ordering makes the downloader's fallback converge on
a good copy quickly.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from ..http import Fetcher
from ..naming import hint_year_from_url, parse_filename
from . import DrawingRef

log = logging.getLogger(__name__)

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
DOMAIN = "loteriasdepuertorico.pr.gov"

# `id_` asks for the archived bytes verbatim, with no Wayback HTML rewriting.
REPLAY_TEMPLATE = "https://web.archive.org/web/{timestamp}id_/{url}"

# More captures per URL than this adds retries without adding coverage.
MAX_CAPTURES_PER_URL = 4


class WaybackSource:
    """Enumerate archived prize-list PDFs, best capture per URL first."""

    name = "wayback"

    def __init__(self, fetcher: Fetcher, max_captures: int = MAX_CAPTURES_PER_URL):
        self.fetcher = fetcher
        self.max_captures = max_captures

    def index(self) -> list[DrawingRef]:
        rows = self._query()
        by_url: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for timestamp, original, length in rows:
            by_url[original].append((timestamp, length))

        refs: list[DrawingRef] = []
        for original, captures in by_url.items():
            filename = original.rsplit("/", 1)[-1]
            parsed = parse_filename(filename, hint_year_from_url(original))
            if parsed is None:
                continue

            # Largest first: a truncated capture can never be the biggest one.
            captures.sort(key=lambda c: (-c[1], c[0]))
            for timestamp, length in captures[: self.max_captures]:
                refs.append(
                    DrawingRef(
                        canonical_id=parsed.canonical_id,
                        year=parsed.year,
                        seq=parsed.seq,
                        suffix=parsed.suffix,
                        filename=filename,
                        url=REPLAY_TEMPLATE.format(timestamp=timestamp, url=original),
                        source=self.name,
                        wayback_timestamp=timestamp,
                        size_hint=length,
                        low_confidence=parsed.low_confidence,
                    )
                )

        distinct = len({r.canonical_id for r in refs})
        log.info(
            "wayback: %d captures of %d files covering %d drawings",
            len(refs),
            len(by_url),
            distinct,
        )
        return refs

    def _query(self) -> list[tuple[str, str, int]]:
        params = {
            "url": DOMAIN,
            "matchType": "domain",
            "fl": "timestamp,original,statuscode,length",
            "collapse": "digest",
            "filter": ["statuscode:200", "original:.*Lista-Oficial-Web.*"],
            "limit": 5000,
        }
        text = self.fetcher.get_text(CDX_ENDPOINT, params=params)

        rows: list[tuple[str, str, int]] = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 4:
                continue
            timestamp, original, _status, length = parts
            if not original.lower().endswith(".pdf"):
                continue
            rows.append((timestamp, original, _to_int(length)))
        return rows


def _to_int(value: str) -> int:
    """CDX occasionally reports a length of ``-``."""
    try:
        return int(value)
    except ValueError:
        return 0
