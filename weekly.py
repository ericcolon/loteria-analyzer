#!/usr/bin/env python3
"""One command for the weekly update: download, parse, load, rebuild the card.

Figures out for itself which drawings are new by comparing the download manifest against
what has already been parsed, so it is safe to run any number of times — on a week with
no new drawing it does nothing and says so.

    python weekly.py              # the weekly run
    python weekly.py --dry-run    # show what it would do
    python weekly.py --publish    # also commit and push the card (see --publish notes)

Exit codes: 0 = fine (including "nothing new"), 1 = something failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
MANIFEST = ROOT / "data" / "manifest.json"
PARSED = ROOT / "data" / "parsed" / "drawings.csv"
CARD = ROOT / "docs" / "index.html"
LOG = ROOT / "data" / "weekly.log"


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(step: str, *args: str) -> None:
    """Run one pipeline stage, streaming its output into the log."""
    log(f"--- {step}: {' '.join(str(a) for a in args)}")
    result = subprocess.run(
        [str(PYTHON), *[str(a) for a in args]],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    for line in (result.stdout or "").splitlines():
        log(f"    {line}")
    if result.returncode != 0:
        for line in (result.stderr or "").splitlines()[-25:]:
            log(f"    !! {line}")
        raise SystemExit(f"{step} failed with exit code {result.returncode}")


def downloaded_ids() -> set[str]:
    if not MANIFEST.exists():
        return set()
    return set(json.loads(MANIFEST.read_text()).get("drawings", {}))


def parsed_ids() -> set[str]:
    if not PARSED.exists():
        return set()
    with open(PARSED, newline="", encoding="utf-8") as handle:
        return {row["canonical_id"] for row in csv.DictReader(handle)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is new without changing anything")
    parser.add_argument("--publish", action="store_true",
                        help="commit and push the rebuilt card (needs a git remote)")
    parser.add_argument("--force-card", action="store_true",
                        help="rebuild the card even when no new drawing arrived")
    args = parser.parse_args(argv)

    if not PYTHON.exists():
        print(f"missing virtualenv at {PYTHON} — run: python3 -m venv .venv && "
              ".venv/bin/pip install -r requirements.txt", file=sys.stderr)
        return 1

    log("=" * 62)
    log("weekly run starting")

    before = downloaded_ids()

    if args.dry_run:
        run("check for new drawings (dry run)",
            "download_results.py", "--sources", "live", "--dry-run")
        pending = before - parsed_ids()
        log(f"already downloaded but not yet parsed: {sorted(pending) or 'none'}")
        log("dry run — nothing written")
        return 0

    # 1. Fetch anything new from the live site. Cheap: already-held files are skipped.
    run("download", "download_results.py", "--sources", "live")

    # 2. Parse whatever the database has not seen yet. Comparing the manifest against the
    #    parsed CSV also picks up anything a previous run downloaded but failed to parse.
    new = sorted(downloaded_ids() - parsed_ids())
    if not new:
        log("no new drawings this week")
        if not args.force_card:
            log("card left as is (use --force-card to rebuild anyway)")
            log("done")
            return 0
    else:
        added = sorted(downloaded_ids() - before)
        log(f"new drawing(s): {', '.join(new)}"
            + (f"   (newly downloaded: {', '.join(added)})" if added else ""))
        run("parse", "parse_results.py", "--only", ",".join(new))
        run("load", "load_supabase.py", "--only", ",".join(new))

    # 3. Rebuild the card from the database so the zone ratings include the new drawing.
    CARD.parent.mkdir(parents=True, exist_ok=True)
    run("rebuild card", "make_card.py", "--out", str(CARD))

    if args.publish:
        publish()

    log("done")
    return 0


def publish() -> None:
    """Commit and push the rebuilt card so the hosted copy updates."""
    if not (ROOT / ".git").exists():
        log("!! --publish needs a git repo; skipping")
        return
    status = subprocess.run(["git", "status", "--porcelain", str(CARD)],
                            cwd=ROOT, capture_output=True, text=True)
    if not status.stdout.strip():
        log("card unchanged; nothing to publish")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for command in (["git", "add", str(CARD)],
                    ["git", "commit", "-m", f"Update zone ratings ({stamp})"],
                    ["git", "push"]):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"!! {' '.join(command)} failed: {result.stderr.strip()[:200]}")
            return
    log("card published")


if __name__ == "__main__":
    sys.exit(main())
