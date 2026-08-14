"""Polite, retrying HTTP with atomic downloads.

Two hosts are involved and they behave differently: the lottery site is small and
should be treated gently, while the Wayback Machine rate-limits aggressively and
returns 429/503 as a matter of course. Both are handled by per-host pacing plus
exponential backoff that honors ``Retry-After``.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

log = logging.getLogger(__name__)

USER_AGENT = (
    "loteria-analyzer/1.0 (personal research archive of public prize lists; "
    "contact ericcolon@22blak.com)"
)

RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 5
CHUNK = 64 * 1024


class FetchError(Exception):
    """A request failed after exhausting retries."""


class Fetcher:
    """A shared session that paces itself per host and retries transient failures."""

    def __init__(self, delays: dict[str, float] | None = None, default_delay: float = 1.5):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.delays = delays or {}
        self.default_delay = default_delay
        self._last_request: dict[str, float] = {}

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- pacing ---------------------------------------------------------------

    def _host(self, url: str) -> str:
        return urlsplit(url).netloc

    def _wait_turn(self, url: str) -> None:
        """Sleep just long enough to keep this host's request rate under the limit."""
        host = self._host(url)
        interval = self.delays.get(host, self.default_delay)
        elapsed = time.monotonic() - self._last_request.get(host, 0.0)
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request[host] = time.monotonic()

    # -- requests -------------------------------------------------------------

    def _with_retries(self, url: str, consume, *, stream: bool = False, **kwargs):
        """GET ``url``, hand the response to ``consume``, and return what it returns.

        ``consume`` runs *inside* the retry loop. A connection dropped while the body
        is still arriving is transient in exactly the way a failed handshake is, and
        restarting the transfer is the only way to recover it, so both are retried.
        Reading the body outside this loop would make every mid-transfer reset fatal.
        """
        last_error: str = "unknown"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._wait_turn(url)
            try:
                response = self.session.get(url, timeout=(15, 120), stream=stream, **kwargs)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code in RETRY_STATUS:
                    last_error = f"HTTP {response.status_code}"
                    retry_after = _retry_after_seconds(response)
                    response.close()
                    if attempt < MAX_ATTEMPTS:
                        self._sleep_backoff(attempt, retry_after)
                    continue

                # Outside the try below: a non-retryable status is the caller's
                # problem, not something to spend four more attempts on.
                response.raise_for_status()
                try:
                    return consume(response)
                except requests.RequestException as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    response.close()

            if attempt < MAX_ATTEMPTS:
                self._sleep_backoff(attempt, None)

        raise FetchError(f"{last_error} after {MAX_ATTEMPTS} attempts: {url}")

    def request(self, url: str, **kwargs) -> requests.Response:
        """GET ``url``, retrying transient failures with exponential backoff."""
        return self._with_retries(url, lambda response: response, **kwargs)

    def get_json(self, url: str, **kwargs):
        response = self.request(url, **kwargs)
        return response.json(), response.headers

    def get_text(self, url: str, **kwargs) -> str:
        return self.request(url, **kwargs).text

    def _sleep_backoff(self, attempt: int, retry_after: float | None) -> None:
        delay = retry_after if retry_after is not None else 2.0**attempt
        delay += random.uniform(0, 0.75)  # jitter, so retries don't synchronize
        log.debug("backing off %.1fs (attempt %d/%d)", delay, attempt, MAX_ATTEMPTS)
        time.sleep(min(delay, 120.0))

    # -- downloads ------------------------------------------------------------

    def download(self, url: str, dest: Path) -> Path:
        """Stream ``url`` to a sibling ``.part`` file and return its path.

        The caller validates the temp file and calls :func:`commit` to move it into
        place, so a failed or truncated download never lands at the real path.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(dest.suffix + ".part")

        def write(response: requests.Response) -> Path:
            # Each attempt truncates and starts over, so a partial write left by a
            # reset connection is never mistaken for the head of a good file.
            try:
                with open(temp, "wb") as handle:
                    for chunk in response.iter_content(CHUNK):
                        handle.write(chunk)
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
            finally:
                response.close()
            return temp

        try:
            return self._with_retries(url, write, stream=True)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise


def commit(temp: Path, dest: Path) -> None:
    """Atomically move a validated temp file into its final location."""
    os.replace(temp, dest)


def _retry_after_seconds(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None  # HTTP-date form; fall back to exponential backoff
