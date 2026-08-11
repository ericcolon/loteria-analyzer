#!/usr/bin/env python3
"""Download and catalog Lotería de Puerto Rico official prize-list PDFs.

Builds a local corpus from two sources — the live site's WordPress media API and the
Wayback Machine for drawings the site has deleted — verifying every file and
recording it in ``data/manifest.json``. Safe to re-run: files already held are
skipped, so a weekly run fetches only what is new.

    python download_results.py --dry-run --report      # see what's available
    python download_results.py --sources live          # just this season
    python download_results.py -v                      # full backfill
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from loteria.http import Fetcher, FetchError, commit
from loteria.manifest import Manifest
from loteria.sources import DrawingRef, rank
from loteria.sources.live import LiveSource
from loteria.sources.wayback import WaybackSource
from loteria.validate import validate_pdf

log = logging.getLogger("loteria")

# The lottery site is small; be gentle. Wayback rate-limits, so back off further.
HOST_DELAYS = {
    "loteriasdepuertorico.pr.gov": 1.5,
    "web.archive.org": 3.0,
}

DRAWS_PER_YEAR = 52


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    # urllib3 narrates every connection at DEBUG, which drowns out progress.
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    root = Path(args.out).resolve()
    pdf_root = root / "pdfs"
    manifest = Manifest(root)
    log.info("manifest holds %d drawing(s) at %s", len(manifest), root)

    delays = dict(HOST_DELAYS)
    if args.delay is not None:
        delays = {host: args.delay for host in delays}

    with Fetcher(delays=delays) as fetcher:
        candidates = collect_candidates(fetcher, args)
        if not candidates:
            log.error("no drawings found — check connectivity or --sources")
            return 1

        pending = [
            (cid, refs)
            for cid, refs in candidates
            if args.force or not manifest.holds(cid)
        ]
        skipped = len(candidates) - len(pending)
        if args.limit is not None:
            pending = pending[: args.limit]

        log.info(
            "%d drawing(s) indexed, %d already held, %d to fetch",
            len(candidates),
            skipped,
            len(pending),
        )

        if args.dry_run:
            for cid, refs in pending:
                log.info("  would fetch %s", refs[0].describe())
            report_gaps(candidates, manifest)
            return 0

        stats = download_all(fetcher, pending, manifest, pdf_root, root)

    log.info(
        "\ndone: %d downloaded, %d skipped, %d failed — %d drawing(s) in the corpus",
        stats["downloaded"],
        skipped,
        len(stats["failures"]),
        len(manifest),
    )
    write_failures(root, stats["failures"])
    if args.report or stats["failures"]:
        report_gaps(candidates, manifest)
    return 0


# -- indexing -----------------------------------------------------------------


def collect_candidates(
    fetcher: Fetcher, args: argparse.Namespace
) -> list[tuple[str, list[DrawingRef]]]:
    """Index every requested source and group fetch candidates by drawing.

    Each drawing gets an ordered list of refs to try: the live site before the
    archive, clean filenames before duplicate artifacts, larger captures before
    smaller ones. That ordering is what lets a truncated Wayback capture fall through
    to a good one.
    """
    refs: list[DrawingRef] = []
    for name in args.sources:
        source = LiveSource(fetcher) if name == "live" else WaybackSource(fetcher)
        try:
            refs.extend(source.index())
        except FetchError as exc:
            log.warning("could not index %s: %s", name, exc)

    grouped: dict[str, list[DrawingRef]] = defaultdict(list)
    for ref in refs:
        if args.since is not None and ref.year < args.since:
            continue
        if args.until is not None and ref.year > args.until:
            continue
        grouped[ref.canonical_id].append(ref)

    for group in grouped.values():
        group.sort(key=rank)

    return sorted(grouped.items())


# -- downloading --------------------------------------------------------------


def download_all(
    fetcher: Fetcher,
    pending: list[tuple[str, list[DrawingRef]]],
    manifest: Manifest,
    pdf_root: Path,
    root: Path,
) -> dict:
    """Fetch each drawing, trying its candidates in order until one validates."""
    downloaded = 0
    failures: list[dict] = []

    for position, (canonical_id, refs) in enumerate(pending, start=1):
        log.info("[%d/%d] %s", position, len(pending), canonical_id)
        attempts: list[str] = []

        for ref in refs:
            dest = pdf_root / str(ref.year) / ref.filename
            try:
                temp = fetcher.download(ref.url, dest)
            except FetchError as exc:
                attempts.append(f"{ref.describe()}: {exc}")
                log.warning("      fetch failed: %s", exc)
                continue

            info = validate_pdf(temp)
            if not info.ok:
                temp.unlink(missing_ok=True)
                attempts.append(f"{ref.describe()}: {info.reason}")
                log.warning("      rejected (%s): %s", info.reason, ref.describe())
                continue

            commit(temp, dest)
            manifest.record(
                {
                    "canonical_id": ref.canonical_id,
                    "year": ref.year,
                    "seq": ref.seq,
                    "suffix": ref.suffix,
                    "published": ref.published,
                    "source": ref.source,
                    "source_url": ref.url,
                    "wayback_timestamp": ref.wayback_timestamp,
                    "local_path": str(dest.relative_to(root)),
                    "sha256": info.sha256,
                    "bytes": info.bytes,
                    "pages": info.pages,
                    "low_confidence": ref.low_confidence,
                }
            )
            manifest.save()  # after every file, so an interrupted run loses nothing
            downloaded += 1
            log.info(
                "      ok %s (%.1f MB, %d pages) via %s",
                ref.filename,
                info.bytes / 1e6,
                info.pages,
                ref.source,
            )
            break
        else:
            log.error("      all %d candidate(s) failed for %s", len(refs), canonical_id)
            failures.append({"canonical_id": canonical_id, "attempts": attempts})

    return {"downloaded": downloaded, "failures": failures}


def write_failures(root: Path, failures: list[dict]) -> None:
    path = root / "failures.json"
    if not failures:
        path.unlink(missing_ok=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failures, indent=2) + "\n")
    log.warning("%d failure(s) written to %s", len(failures), path)


# -- reporting ----------------------------------------------------------------


def report_gaps(
    candidates: list[tuple[str, list[DrawingRef]]], manifest: Manifest
) -> None:
    """Show, per year, which weekly draws are held and which are missing.

    Coverage is uneven — the archive has no reason to hold every week — so knowing
    exactly which draws are absent is a precondition for trusting any later analysis.
    """
    held = manifest.held_draw_keys()
    indexed: dict[int, set[int]] = defaultdict(set)
    for _, refs in candidates:
        indexed[refs[0].year].add(refs[0].seq)

    current_year = datetime.date.today().year
    log.info("\ncoverage by drawing year (weekly draws)")

    for year in sorted(set(indexed) | {y for y, _ in held}):
        seen = indexed.get(year, set())
        held_seqs = {seq for y, seq in held if y == year}
        highest = max(seen | held_seqs, default=0)
        # A year in progress only has the draws that have happened so far.
        expected = highest if year >= current_year else max(DRAWS_PER_YEAR, highest)
        missing = sorted(set(range(1, expected + 1)) - held_seqs)

        log.info(
            "  %d: %2d/%d held%s",
            year,
            len(held_seqs),
            expected,
            f"  missing {_compress(missing)}" if missing else "  complete",
        )


def _compress(numbers: list[int]) -> str:
    """Render a sorted int list compactly: ``1-4, 9, 12-14``."""
    if not numbers:
        return ""
    spans: list[str] = []
    start = previous = numbers[0]
    for value in numbers[1:] + [None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        spans.append(str(start) if start == previous else f"{start}-{previous}")
        if value is not None:
            start = previous = value
    return ", ".join(spans)


# -- CLI ----------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        default="live,wayback",
        help="comma-separated: live, wayback (default: both)",
    )
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    parser.add_argument("--since", type=int, metavar="YEAR", help="earliest drawing year")
    parser.add_argument("--until", type=int, metavar="YEAR", help="latest drawing year")
    parser.add_argument("--limit", type=int, help="stop after this many downloads")
    parser.add_argument(
        "--delay", type=float, help="seconds between requests to a host (overrides defaults)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="resolve and list work without downloading"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download drawings already held"
    )
    parser.add_argument("--report", action="store_true", help="print the coverage report")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    args = parser.parse_args(argv)

    args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(args.sources) - {"live", "wayback"}
    if unknown:
        parser.error(f"unknown source(s): {', '.join(sorted(unknown))}")
    if not args.sources:
        parser.error("--sources needs at least one of: live, wayback")

    return args


if __name__ == "__main__":
    sys.exit(main())
