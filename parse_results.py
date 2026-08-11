#!/usr/bin/env python3
"""Parse downloaded prize-list PDFs into CSVs ready for loading.

Reads ``data/manifest.json``, parses each PDF, checks each result against the prize
structure's invariants, and writes ``data/parsed/{drawings,prizes,major_prizes}.csv``
plus a validation report.

Drawings that fail validation are written with ``valid=false`` and their problems
recorded, never silently dropped — a prize list that half-parsed is worse than one
that's visibly broken.

    python parse_results.py                 # parse everything
    python parse_results.py --limit 5 -v    # smoke test
    python parse_results.py --only 2026-032
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from loteria.manifest import Manifest
from loteria.parse import parse_pdf

log = logging.getLogger("parse")

DRAWING_FIELDS = [
    "canonical_id", "drawing_label", "year", "seq", "suffix", "kind",
    "draw_date", "celebrated_on", "expires_on", "series", "next_drawing",
    "first_prize_last_digit", "major_count", "prize_count", "pages",
    "pdf_filename", "pdf_sha256", "pdf_source", "source_url",
    "valid", "problems",
]
PRIZE_FIELDS = ["drawing_id", "number", "category", "amount"]
MAJOR_FIELDS = ["drawing_id", "rank", "number"]

KINDS = {"": "ordinary", "S": "special", "X": "extraordinary"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    # pdfminer logs every token it reads at DEBUG, which buries our own output.
    logging.getLogger("pdfminer").setLevel(logging.WARNING)

    root = Path(args.out).resolve()
    manifest = Manifest(root)
    if not len(manifest):
        log.error("no manifest at %s — run download_results.py first", root)
        return 1

    records = sorted(manifest.drawings.values(), key=lambda r: r["canonical_id"])
    if args.only:
        wanted = set(args.only.split(","))
        records = [r for r in records if r["canonical_id"] in wanted]
    if args.limit:
        records = records[: args.limit]

    out_dir = root / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    drawings: list[dict] = []
    prizes: list[dict] = []
    majors: list[dict] = []
    report: list[dict] = []

    for position, record in enumerate(records, start=1):
        pdf_path = root / record["local_path"]
        canonical_id = record["canonical_id"]
        try:
            parsed = parse_pdf(pdf_path)
        except Exception as exc:  # a broken PDF shouldn't stop the sweep
            log.error("[%d/%d] %s FAILED to parse: %s", position, len(records), canonical_id, exc)
            report.append({"canonical_id": canonical_id, "problems": [f"parse error: {exc}"]})
            continue

        problems = parsed.invariants()
        drawings.append(
            {
                "canonical_id": canonical_id,
                "drawing_label": parsed.drawing_label,
                "year": record["year"],
                "seq": record["seq"],
                "suffix": record["suffix"],
                "kind": KINDS.get(record["suffix"], "unknown"),
                "draw_date": parsed.draw_date or "",
                "celebrated_on": parsed.celebrated_on or "",
                "expires_on": parsed.expires_on or "",
                "series": "".join(parsed.series),
                "next_drawing": parsed.next_drawing or "",
                "first_prize_last_digit": (
                    "" if parsed.first_prize_last_digit is None
                    else parsed.first_prize_last_digit
                ),
                "major_count": len(parsed.majors),
                "prize_count": len(parsed.prizes),
                "pages": parsed.pages,
                "pdf_filename": pdf_path.name,
                "pdf_sha256": record["sha256"],
                "pdf_source": record["source"],
                "source_url": record["source_url"],
                "valid": not problems,
                "problems": json.dumps(problems),
            }
        )
        for prize in parsed.prizes:
            prizes.append(
                {
                    "drawing_id": canonical_id,
                    "number": prize.number,
                    "category": prize.category,
                    "amount": prize.amount,
                }
            )
        for rank, number in parsed.majors:
            majors.append({"drawing_id": canonical_id, "rank": rank, "number": number})

        if problems:
            report.append({"canonical_id": canonical_id, "problems": problems})
            log.warning(
                "[%d/%d] %s  %d prizes  INVALID: %s",
                position, len(records), canonical_id, len(parsed.prizes), problems[0],
            )
        else:
            log.info(
                "[%d/%d] %s  %d prizes  %d majors  %s  ok",
                position, len(records), canonical_id,
                len(parsed.prizes), len(parsed.majors), parsed.draw_date,
            )

    # Parsing a subset merges into the existing CSVs rather than replacing them, so
    # `--only 2026-033` after next week's download extends the corpus instead of
    # truncating it to one drawing.
    parsed_ids = {d["canonical_id"] for d in drawings}
    _merge_csv(out_dir / "drawings.csv", DRAWING_FIELDS, drawings, "canonical_id", parsed_ids)
    _merge_csv(out_dir / "prizes.csv", PRIZE_FIELDS, prizes, "drawing_id", parsed_ids)
    _merge_csv(out_dir / "major_prizes.csv", MAJOR_FIELDS, majors, "drawing_id", parsed_ids)
    (out_dir / "parse_report.json").write_text(json.dumps(report, indent=2) + "\n")

    valid = sum(1 for d in drawings if d["valid"])
    log.info(
        "\nparsed %d drawing(s): %d valid, %d invalid — %d prize rows -> %s",
        len(drawings), valid, len(drawings) - valid, len(prizes), out_dir,
    )
    if report:
        log.warning("%d drawing(s) need review; see parse_report.json", len(report))
    return 0


def _merge_csv(
    path: Path, fields: list[str], rows: list[dict], key: str, parsed_ids: set[str]
) -> None:
    """Replace rows for the drawings just parsed, keeping every other row intact."""
    kept: list[dict] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as handle:
            kept = [r for r in csv.DictReader(handle) if r.get(key) not in parsed_ids]

    combined = kept + [{f: r.get(f, "") for f in fields} for r in rows]
    combined.sort(key=lambda r: (str(r.get(key, "")), str(r.get("number", ""))))

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    log.debug("wrote %d rows to %s (%d kept, %d new)", len(combined), path, len(kept), len(rows))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data", help="data directory (default: data)")
    parser.add_argument("--only", help="comma-separated canonical ids to parse")
    parser.add_argument("--limit", type=int, help="parse at most this many")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
