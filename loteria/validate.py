"""Integrity checks for downloaded PDFs.

These are not paranoia. The Wayback Machine serves truncated captures that look
entirely healthy at the HTTP layer: capture ``20250120132115`` of
``Lista-Oficial-Web-025.pdf`` returns 200, ``content-type: application/pdf`` and a
valid ``%PDF-1.5`` header, but is cut off at exactly 1 MiB and holds 1 page instead
of 3. Only a structural check catches that.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pypdf

# Wayback truncates some captures at exactly this boundary.
TRUNCATION_SIZE = 1024 * 1024

# %%EOF lives at the very end of a well-formed PDF; allow slack for trailing bytes.
EOF_SEARCH_WINDOW = 4096


@dataclass(frozen=True)
class PdfInfo:
    """Result of validating a candidate PDF."""

    ok: bool
    reason: str
    bytes: int
    sha256: str | None = None
    pages: int | None = None


def validate_pdf(path: Path) -> PdfInfo:
    """Check that ``path`` is a complete, readable PDF and fingerprint it."""
    size = path.stat().st_size

    if size == 0:
        return PdfInfo(False, "empty file", size)

    with open(path, "rb") as handle:
        if handle.read(5)[:4] != b"%PDF":
            return PdfInfo(False, "not a PDF (bad magic bytes)", size)
        handle.seek(max(0, size - EOF_SEARCH_WINDOW))
        if b"%%EOF" not in handle.read():
            return PdfInfo(False, "truncated (no %%EOF trailer)", size)

    if size == TRUNCATION_SIZE:
        return PdfInfo(False, f"truncated (exactly {TRUNCATION_SIZE} bytes)", size)

    try:
        pages = len(pypdf.PdfReader(str(path)).pages)
    except Exception as exc:  # pypdf raises a wide variety of parse errors
        return PdfInfo(False, f"unreadable PDF ({type(exc).__name__}: {exc})", size)

    if pages < 1:
        return PdfInfo(False, "no pages", size)

    return PdfInfo(True, "ok", size, sha256=sha256_file(path), pages=pages)


def sha256_file(path: Path) -> str:
    """Hash a file in chunks, so multi-megabyte PDFs don't land in memory at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
