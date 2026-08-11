"""The catalog of what has been downloaded.

``manifest.json`` is the durable artifact of a run — it records where every PDF came
from, when, and its checksum. It is written after each successful file so an
interrupted run never loses work, and always via temp-file-plus-rename so a crash
mid-write can't corrupt it. ``manifest.csv`` is a flat mirror for easy loading later.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import os
from pathlib import Path
from typing import Any

from .validate import sha256_file

VERSION = 1

FIELDS = [
    "canonical_id",
    "year",
    "seq",
    "suffix",
    "published",
    "source",
    "source_url",
    "wayback_timestamp",
    "local_path",
    "sha256",
    "bytes",
    "pages",
    "low_confidence",
    "fetched_at",
]


class Manifest:
    """Load, query, and persist the download catalog."""

    def __init__(self, root: Path):
        self.root = root
        self.path = root / "manifest.json"
        self.csv_path = root / "manifest.csv"
        self.drawings: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.drawings = data.get("drawings", {})

    # -- queries --------------------------------------------------------------

    def __contains__(self, canonical_id: str) -> bool:
        return canonical_id in self.drawings

    def __len__(self) -> int:
        return len(self.drawings)

    def get(self, canonical_id: str) -> dict[str, Any] | None:
        return self.drawings.get(canonical_id)

    def holds(self, canonical_id: str) -> bool:
        """True if this drawing is already downloaded and the file still matches.

        Verifies by size first and only hashes when the size disagrees, so a re-run
        over ~180 multi-megabyte PDFs stays fast.
        """
        record = self.drawings.get(canonical_id)
        if record is None:
            return False
        path = self.root / record["local_path"]
        if not path.exists():
            return False
        if path.stat().st_size == record.get("bytes"):
            return True
        return sha256_file(path) == record.get("sha256")

    def held_draw_keys(self) -> set[tuple[int, int]]:
        """(year, seq) slots already held, ignoring suffix — used by the gap report."""
        return {(r["year"], r["seq"]) for r in self.drawings.values()}

    # -- mutation -------------------------------------------------------------

    def record(self, entry: dict[str, Any]) -> None:
        entry["fetched_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
        self.drawings[entry["canonical_id"]] = entry

    def save(self) -> None:
        """Persist both JSON and CSV forms atomically."""
        payload = {
            "version": VERSION,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "count": len(self.drawings),
            "drawings": dict(sorted(self.drawings.items())),
        }
        _atomic_write(self.path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(self.drawings.values(), key=lambda r: r["canonical_id"]))
        _atomic_write(self.csv_path, buffer.getvalue())


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)
