"""Index the live site through the WordPress REST media API.

The public site is JavaScript-rendered and its homepage contains no PDF links at all,
so HTML scraping is a dead end. ``/wp-json/wp/v2/media`` exposes the same uploads as
JSON — authoritative, paginated, and cheap.

Only about six months of files stay live; older drawings are deleted from the server
and must come from :mod:`loteria.sources.wayback`.
"""

from __future__ import annotations

import logging
from urllib.parse import urljoin

from ..http import Fetcher
from ..naming import hint_year_from_url, parse_filename
from . import SITE, DrawingRef

log = logging.getLogger(__name__)

MEDIA_ENDPOINT = f"{SITE}/wp-json/wp/v2/media"
SEARCH_TERM = "Lista-Oficial-Web"
PER_PAGE = 100


class LiveSource:
    """Enumerate prize-list PDFs currently published on the site."""

    name = "live"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def index(self) -> list[DrawingRef]:
        refs: list[DrawingRef] = []
        page = 1
        total_pages = 1

        while page <= total_pages:
            params = {
                "search": SEARCH_TERM,
                "per_page": PER_PAGE,
                "page": page,
                "orderby": "date",
                "order": "asc",
                "_fields": "id,date,title,source_url",
            }
            items, headers = self.fetcher.get_json(MEDIA_ENDPOINT, params=params)
            if page == 1:
                total_pages = int(headers.get("X-WP-TotalPages", 1) or 1)
                log.info(
                    "live: %s files across %d page(s)",
                    headers.get("X-WP-Total", "?"),
                    total_pages,
                )

            for item in items:
                ref = self._to_ref(item)
                if ref is not None:
                    refs.append(ref)
            page += 1

        log.info("live: indexed %d prize lists", len(refs))
        return refs

    def _to_ref(self, item: dict) -> DrawingRef | None:
        source_url = item.get("source_url") or ""
        if not source_url:
            return None
        # source_url comes back site-relative from this install.
        url = urljoin(SITE, source_url)
        filename = url.rsplit("/", 1)[-1]

        parsed = parse_filename(filename, hint_year_from_url(url))
        if parsed is None:
            return None  # some other document in the media library

        published = (item.get("date") or "")[:10] or None
        return DrawingRef(
            canonical_id=parsed.canonical_id,
            year=parsed.year,
            seq=parsed.seq,
            suffix=parsed.suffix,
            filename=filename,
            url=url,
            source=self.name,
            published=published,
            low_confidence=parsed.low_confidence,
        )
