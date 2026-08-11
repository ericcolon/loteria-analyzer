#!/usr/bin/env python3
"""Load parsed prize lists into Supabase (Postgres) via COPY.

Reads the CSVs written by ``parse_results.py`` and streams them into the database.
COPY is used rather than row-by-row inserts because the corpus is ~370k prize rows —
inserts would take minutes and thousands of round trips; COPY takes seconds.

The load is idempotent per drawing: each drawing's existing rows are deleted and
replaced inside a single transaction, so re-running after a re-parse converges instead
of duplicating. Nothing is committed unless the whole load succeeds.

Set DATABASE_URL in .env first (see .env.example), then:

    python load_supabase.py --dry-run    # verify connection + preview counts
    python load_supabase.py              # load everything
    python load_supabase.py --only 2026-032
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

log = logging.getLogger("load")

# Column order must match the COPY statements below.
DRAWING_COLUMNS = [
    "canonical_id", "drawing_label", "year", "seq", "suffix", "kind",
    "draw_date", "celebrated_on", "expires_on", "series", "next_drawing",
    "first_prize_last_digit", "major_count", "prize_count", "pages",
    "pdf_filename", "pdf_sha256", "pdf_source", "source_url", "valid", "problems",
]
PRIZE_COLUMNS = ["drawing_id", "number", "category", "amount"]
MAJOR_COLUMNS = ["drawing_id", "rank", "number"]

# Columns where an empty CSV field means SQL NULL rather than an empty string.
# This has to be per-column: a blanket `null ''` would also blank out `suffix`, which
# is NOT NULL DEFAULT '' for ordinary drawings and legitimately empty.
FORCE_NULL = {
    "drawings": [
        "draw_date", "celebrated_on", "expires_on",
        "first_prize_last_digit", "next_drawing", "series",
        "pdf_filename", "pdf_sha256", "pdf_source", "source_url",
    ],
    "prizes": [],
    "major_prizes": [],
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    load_dotenv()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        log.error(
            "DATABASE_URL is not set.\n"
            "  1. cp .env.example .env\n"
            "  2. paste your project's Postgres URI into it "
            "(Supabase dashboard -> Connect -> Session pooler)\n"
            "Nothing was sent anywhere."
        )
        return 1

    parsed_dir = Path(args.parsed).resolve()
    drawings = _read_csv(parsed_dir / "drawings.csv")
    prizes = _read_csv(parsed_dir / "prizes.csv")
    majors = _read_csv(parsed_dir / "major_prizes.csv")

    if args.only:
        wanted = set(args.only.split(","))
        drawings = [r for r in drawings if r["canonical_id"] in wanted]
        prizes = [r for r in prizes if r["drawing_id"] in wanted]
        majors = [r for r in majors if r["drawing_id"] in wanted]

    ids = [r["canonical_id"] for r in drawings]
    valid = sum(1 for r in drawings if r["valid"] == "True")
    log.info(
        "%d drawing(s) (%d valid, %d invalid), %d prize rows, %d major rows",
        len(drawings), valid, len(drawings) - valid, len(prizes), len(majors),
    )

    if args.dry_run:
        with psycopg.connect(dsn, connect_timeout=20) as conn:
            version = conn.execute("select version()").fetchone()[0]
            existing = conn.execute("select count(*) from drawings").fetchone()[0]
        log.info("connected: %s", version.split(",")[0])
        log.info("drawings already in database: %d", existing)
        log.info("dry run — nothing written")
        return 0

    with psycopg.connect(dsn, connect_timeout=20) as conn:
        with conn.transaction():
            deleted = _replace_drawings(conn, ids)
            if deleted:
                log.info("replacing %d existing drawing(s)", deleted)
            _copy(conn, "drawings", DRAWING_COLUMNS, drawings)
            _copy(conn, "prizes", PRIZE_COLUMNS, prizes)
            _copy(conn, "major_prizes", MAJOR_COLUMNS, majors)
        totals = _totals(conn)

    log.info(
        "\nloaded. database now holds %d drawings (%d valid), %d prizes, %d majors",
        totals["drawings"], totals["valid"], totals["prizes"], totals["majors"],
    )
    return 0


def _replace_drawings(conn: psycopg.Connection, ids: list[str]) -> int:
    """Delete the drawings we're about to load; cascade clears their child rows."""
    if not ids:
        return 0
    result = conn.execute("delete from drawings where canonical_id = any(%s)", (ids,))
    return result.rowcount


def _copy(conn: psycopg.Connection, table: str, columns: list[str], rows: list[dict]) -> None:
    """Stream rows into ``table`` with COPY."""
    if not rows:
        log.debug("nothing to copy into %s", table)
        return
    column_list = ", ".join(f'"{c}"' for c in columns)
    options = ["format csv"]
    nullable = [c for c in FORCE_NULL.get(table, []) if c in columns]
    if nullable:
        options.append("force_null (" + ", ".join(f'"{c}"' for c in nullable) + ")")
    statement = f"copy {table} ({column_list}) from stdin with ({', '.join(options)})"

    with conn.cursor().copy(statement) as copy:
        buffer = io.StringIO()
        # Quote everything, so an empty field is an explicit empty string. Only the
        # force_null columns above then turn that into NULL.
        writer = csv.writer(buffer, quoting=csv.QUOTE_ALL)
        for index, row in enumerate(rows, start=1):
            writer.writerow([_cell(row.get(c, ""), c) for c in columns])
            if index % 50_000 == 0:
                copy.write(buffer.getvalue())
                buffer.seek(0)
                buffer.truncate(0)
        copy.write(buffer.getvalue())
    log.info("copied %6d row(s) into %s", len(rows), table)


def _cell(value: str, column: str) -> str:
    """Normalise a CSV cell for Postgres."""
    if column == "problems" and not value:
        return "[]"  # jsonb is NOT NULL; an absent list is an empty one
    return value


def _totals(conn: psycopg.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        select (select count(*) from drawings),
               (select count(*) from drawings where valid),
               (select count(*) from prizes),
               (select count(*) from major_prizes)
        """
    ).fetchone()
    return {"drawings": row[0], "valid": row[1], "prizes": row[2], "majors": row[3]}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        log.error("missing %s — run parse_results.py first", path)
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parsed", default="data/parsed", help="directory of parsed CSVs")
    parser.add_argument("--only", help="comma-separated canonical ids to load")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the connection and report counts without writing")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
