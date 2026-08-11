# loteria-analyzer

Downloads, parses, and warehouses the official prize-list PDFs (`Lista Oficial de Premios`)
published by the Lotería de Puerto Rico.

**Live card: https://ericcolon.github.io/loteria-analyzer/** — zone ratings, rebuilt weekly.

Five stages, each independently re-runnable:

```
download_results.py  →  verified PDFs + data/manifest.json
parse_results.py     →  data/parsed/*.csv  (structured, self-validated)
load_supabase.py     →  Postgres tables in Supabase
analyze.py           →  the shopping list, on the terminal
make_card.py         →  docs/index.html, served by GitHub Pages
```

`weekly.py` runs all of it in one command, and launchd fires it every Friday.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then paste your Supabase Postgres URI (load stage only)
```

## Weekly routine — automated

A new drawing is published each Thursday evening. **This runs itself every Friday at
8:00 AM** via a launchd agent; there is nothing to remember.

```bash
.venv/bin/python weekly.py              # what the schedule runs
.venv/bin/python weekly.py --dry-run    # show what's new, change nothing
.venv/bin/python weekly.py --publish    # also push the rebuilt card to Pages
```

`weekly.py` works out for itself which drawings are new — it compares the download
manifest against what has already been parsed — so it is safe to run repeatedly. On a week
with no new drawing it does nothing and says so. It also picks up anything a previous run
downloaded but failed to parse.

### The schedule

| | |
| --- | --- |
| Agent | `com.ericcolon.loteria-weekly` |
| Plist | `~/Library/LaunchAgents/com.ericcolon.loteria-weekly.plist` |
| Fires | Fridays, 8:00 AM local |
| Logs | `data/weekly.log`, plus `data/launchd.{out,err}.log` |

```bash
launchctl print gui/$(id -u)/com.ericcolon.loteria-weekly   # status, run count, last exit
launchctl kickstart -w gui/$(id -u)/com.ericcolon.loteria-weekly   # run it now
launchctl bootout gui/$(id -u)/com.ericcolon.loteria-weekly        # disable
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ericcolon.loteria-weekly.plist  # re-enable
```

Two details that matter for reliability, both verified rather than assumed:

- **The Mac has to be awake at some point.** If it's asleep at 8:00 Friday, launchd runs
  the job when it next wakes, so a week is never silently skipped — it just lands late.
- **The plist sets `PATH` explicitly.** launchd's default `PATH` omits Homebrew, and `git`
  lives in `/opt/homebrew/bin` here, so publishing would fail without it. Pushing over SSH
  was checked in a stripped environment (no ssh-agent) before relying on it.

### Deployment

`docs/index.html` is the card; GitHub Pages serves it from `main` at `/docs`. `weekly.py
--publish` commits and pushes it only when the content actually changed, and Pages rebuilds
within a minute or two. The card carries its own build date so a stale copy is obvious.

Manually: `.venv/bin/python make_card.py && git add docs/index.html && git commit && git push`.

Every stage is idempotent, so running any of them twice is harmless.

## Stage 1 — download

```bash
# See what's available without downloading anything
.venv/bin/python download_results.py --dry-run --report

# Just what's currently on the live site (~27 drawings)
.venv/bin/python download_results.py --sources live

# Everything, including archived drawings back to 2021 (~168 drawings, ~15 min)
.venv/bin/python download_results.py --report -v

# Weekly top-up — a new drawing is published each Thursday
.venv/bin/python download_results.py --sources live
```

Re-running is safe and cheap: drawings already downloaded are verified by size and skipped, so a
weekly run fetches only what's new.

| Flag | Effect |
| --- | --- |
| `--sources live,wayback` | Which indexes to use (default: both) |
| `--out DIR` | Output directory (default: `data`) |
| `--since YEAR` / `--until YEAR` | Restrict to a range of drawing years |
| `--limit N` | Stop after N downloads — useful for a first smoke test |
| `--delay SEC` | Seconds between requests to a host |
| `--dry-run` | Resolve and list the work without downloading |
| `--force` | Re-download drawings already held |
| `--report` | Print the per-year coverage report |
| `-v` | Debug logging |

## Output

```
data/
├── pdfs/<year>/Lista-Oficial-Web-<n>.pdf
├── manifest.json     # full catalog: provenance, checksum, page count
├── manifest.csv      # same, flat, for loading into pandas
└── failures.json     # only present if something failed
```

Each manifest entry records `canonical_id`, `year`, `seq`, `suffix`, `published`, `source`,
`source_url`, `wayback_timestamp`, `local_path`, `sha256`, `bytes`, `pages`, `low_confidence`, and
`fetched_at`. The manifest is saved after every file, so an interrupted run never loses work.

## Stage 2 — parse

```bash
.venv/bin/python parse_results.py              # parse the whole corpus (~35 min)
.venv/bin/python parse_results.py --limit 5    # smoke test
.venv/bin/python parse_results.py --only 2026-032
```

Writes `data/parsed/drawings.csv`, `prizes.csv`, `major_prizes.csv`, and `parse_report.json`.
Parsing a subset **merges** into the existing CSVs rather than replacing them, so `--only` is the
right tool for a weekly top-up.

### The PDFs fight back

They're vector text, so no OCR is needed, but every glyph is positioned individually.
`extract_text()` returns debris like `"4 7 083"` and `"NNÚÚMMEERROO"`. Four things the layout does
that break the obvious approaches:

1. **Three different separator styles.** Ordinary rows print `2429C-- 100`, four-digit amounts print
   `37378 - 1200`, and the top prize prints `47083 250000` with no separator at all. Rather than
   matching separators, the parser takes each cell's **rightmost numeric token** as the amount and
   everything left of it as the winning number.
2. **Column pitch alternates** between 78.5 and 79.5 points. Assuming a uniform pitch accumulates
   ~4pt of drift over the 14 column-groups — enough to throw prizes into the neighbouring column. This
   silently corrupted 506 cells on the 2022 files before it was fixed. Column boundaries now come from
   each page's actual `NÚMERO` header positions.
3. **Headers are double-stamped**, two different ways: per-character (`DDEE` → `DE`) and per-word
   (`SORTEO SORTEO ORDINARIO ORDINARIO 432 432`). The table body is not doubled.
4. **Section labels sit in data cells.** `CENTESIMALES` and `CIENTOS` occupy the first cell of a
   column, and thousands markers (`18 MIL`) sit mid-table. These fail the cell shape and are counted
   as skipped rather than parsed.

### It checks its own work

The prize structure is rigid arithmetic, which makes strong invariants possible. For a list with
`n` major prizes, there must be exactly:

| Category | Suffix | Count | Meaning |
| --- | --- | --- | --- |
| directo | *(none)* | — | the number itself won |
| aproximación | `A` | `2n` | the numbers either side of a major prize |
| centena | `C` | `99n` | the rest of the hundred containing a major prize |
| terminación | `T` | `49n` | shares a major prize's last three digits |

Every headline number must also reappear in the table with a large prize, and the printed
`ÚLTIMA CIFRA DEL PRIMER PREMIO` must match the top prize's last digit.

Counts are of **distinct numbers**, not rows — that distinction matters, see below. Any drawing that
fails is written with `valid = false` and its specific problems recorded, never dropped.

## Stage 3 — load into Supabase

```bash
.venv/bin/python load_supabase.py --dry-run    # verify connection, preview counts
.venv/bin/python load_supabase.py              # COPY everything (seconds)
.venv/bin/python load_supabase.py --only 2026-032
```

Needs `DATABASE_URL` in `.env` — use the **Session pooler** URI from the Supabase dashboard, not the
Transaction pooler, since `COPY` requires a session connection. `.env` is gitignored.

`COPY` is used rather than inserts because the corpus is ~364k prize rows; inserts would mean minutes
and thousands of round trips. The load is idempotent per drawing — each drawing's rows are deleted and
replaced inside one transaction, so nothing is committed unless the whole load succeeds.

### Schema

```
drawings       one row per drawing: dates, series, kind, provenance, valid + problems
prizes         one row per winning number: number, category, amount
major_prizes   the ranked headline numbers from PREMIOS MAYORES
prizes_detailed  view joining the above, with zero-padded number_text and category names
```

Row Level Security is on with an `authenticated`-read policy, so the data is not publicly readable.

Winning numbers run **00001–50000** — there is no `00000`, and `50000` does occur. Numbers are stored
as integers; use `lpad(number::text, 5, '0')` (or the view's `number_text`) for display.

There is deliberately **no unique constraint** on `(drawing_id, number, category)`, because a few
published PDFs genuinely print the same block twice. Those drawings are flagged instead.

## What's in the corpus

As loaded: **141 drawings, 364,096 prize rows**, spanning 2021-10-14 → 2026-08-06.
**137 are `valid = true`**; 4 are flagged.

| Year | Valid | Total |
| --- | --- | --- |
| 2021 | 1 | 1 |
| 2022 | 8 | 9 |
| 2023 | 31 | 32 |
| 2024 | 32 | 33 |
| 2025 | 36 | 37 |
| 2026 | 29 | 29 |

Coverage is partial for older years because the live site only retains ~6 months and the Wayback
archive has no obligation to hold every week. Run `download_results.py --report` for the exact
per-year gaps.

## Data quality: read this before analyzing

Filter on `valid = true` unless you are specifically investigating a defect. The 4 flagged drawings
are all defects in the **published PDFs**, not parser bugs:

| Defect | Drawings | Effect |
| --- | --- | --- |
| Page 2's glyphs flattened to vector outlines | `2024-030`, `2023-027S` | ~1,000 prizes unreadable without OCR |
| PDF contains two drawings' data | `2025-028` — two separate $250,000 top prizes | unusable |
| A headline number buried under an overlapping layer | `2022-047` (the only capture in existence) | 1 major unreadable; prize table intact |

The vector-outline case is recoverable in principle — those pages render correctly, the digits are
just outlined rather than encoded as text. It would need rasterising plus OCR, which is not
implemented.

`2022-047`'s missing major is actually *deducible*: the unexplained centena block and terminación
class intersect at exactly one number. That inference is deliberately not automated — it would put
a number in the database that the document does not state.

### A false positive worth remembering

An earlier version of the invariant flagged five additional drawings (`2023-004`, `2023-026`,
`2024-050`, `2026-029`, `2026-015`) as having "a block printed twice and another omitted". They are
all perfectly fine. When two majors land in the same hundred-block, **each awards centena to the
other 99 numbers in it**, so the union covers all 100 and the two majors collect centena off each
other — and the overlapping numbers legitimately appear twice, winning twice. The check was
subtracting all majors globally instead of applying the rule per major. It happened to agree with
reality for the 132 drawings where no majors collided, which is exactly what made it dangerous.

The lesson generalised: the invariant now compares the derived categories against a full
rule-based prediction (`derived_prize_sets` in `loteria/parse.py`), not against a count.

### Cross-checks that pass

- Re-deriving every centena / terminación / aproximación set from the rules finds **zero**
  mismatches across all 137 valid drawings.
- The printed `ÚLTIMA CIFRA DEL PRIMER PREMIO` agrees with the top prize's last digit in **every**
  case where it is printed (only `2021-041`, the oldest, omits the note).
- `analyze.py --verify` re-derives the model's structural constants from the database and confirms
  the prize pool and every category size is exactly constant per draw type.

## How the numbering works

The filenames look like a counter that jumps by 10 — `246`, `256`, `266` — but that's not what's
happening. The number is `<SS><Y>`, where `SS` is the **weekly drawing sequence within the year** and
`Y` is the **last digit of the year**:

| Filename | Decodes to | Canonical ID |
| --- | --- | --- |
| `Lista-Oficial-Web-326.pdf` | draw 32 of 2026 | `2026-032` |
| `Lista-Oficial-Web-066.pdf` | draw 6 of 2026 | `2026-006` |
| `Lista-Oficial-Web-371.pdf` | draw 37 of 2021 | `2021-037` |
| `Lista-Oficial-Web-012.pdf` | draw 1 of 2022 | `2022-001` |

Incrementing the weekly draw shifts the printed number by 10 because the year digit is pinned in the
ones place. Sequences below 10 are zero-padded to keep the number at three digits (`066`, not `66`).
A trailing letter marks a non-ordinary draw — `S` (special, e.g. `076S`) or `X` (extraordinary, e.g.
`196X`).

## Why it enumerates instead of guessing URLs

Constructing URLs from the drawing number does not work:

- **The upload folder isn't the drawing month.** Draw `226` was drawn 2026-05-28 but lives under
  `/uploads/2026/05/`, while `236` (2026-06-04) is under `/uploads/2026/06/`. Guessing the wrong
  month 404s.
- **Special draws carry suffixes** (`076S`, `196X`, `511S`) that a numeric loop won't produce.
- **The archive contains WordPress artifacts** — `Lista-Oficial-Web-472-4.pdf`,
  `Lista-Oficial-Web-285-copy.pdf`, a typo'd `Lista-Oficial-Web-2831.pdf`, and even a malformed
  `/uploads/2024//2/` path. These are cataloged but flagged `low_confidence` so they can never
  displace a clean copy of the same drawing.

Instead, two indexes are consulted:

1. **Live site** — `/wp-json/wp/v2/media?search=Lista-Oficial-Web`, the WordPress REST API, which
   lists uploads as JSON. (The public site is JavaScript-rendered and its homepage contains zero PDF
   links, so HTML scraping is a dead end.)
2. **Wayback Machine** — the CDX API, filtered to `statuscode:200`, for drawings the site has since
   deleted.

## Two things that will bite you if you rebuild this

**The live site only keeps about six months.** Right now it serves 27 drawings, 2026-02-05 through
2026-08-06. Everything older has been deleted — `Lista-Oficial-Web-056.pdf`, `016`, and `371` all
return 404. The Wayback backfill is what gets coverage back to 2021, roughly 168 drawings total.
Since the site prunes, the corpus is worth keeping and topping up weekly rather than re-derived on
demand.

**Wayback serves truncated captures that look perfectly healthy.** Capture `20250120132115` of
`Lista-Oficial-Web-025.pdf` returns HTTP 200, `content-type: application/pdf`, and a valid
`%PDF-1.5` header — but stops at exactly 1 MiB and holds 1 page instead of 2. A later capture of the
same URL is complete at 3.4 MB. So every download is structurally validated (`%PDF` magic, `%%EOF`
trailer, pypdf parses it, page count ≥ 1, size not exactly 1 MiB) before it's committed to its final
path, and each drawing's captures are tried largest-first so the fallback converges on a good copy.

## Coverage is uneven — check the report before analyzing

The Wayback archive has no obligation to hold every week, so some years are partial. Run with
`--report` to see exactly which weekly draws are present and which are missing per year. Knowing the
gaps is a precondition for trusting anything computed from the corpus.

## Stage 4 — analysis: which numbers should you buy?

```bash
.venv/bin/python analyze.py                          # all four sections
.venv/bin/python analyze.py --section frequency
.venv/bin/python analyze.py --section portfolio --budget 10
.venv/bin/python analyze.py --verify                 # re-derive model constants from the DB
```

### How the drawing actually works

50,000 balls, always. Two tumblers — one pulls a number, the other a prize. The prize structure
is **entirely predetermined**: an ordinary drawing awards exactly 2,629 prizes totalling exactly
$703,300, every time, with zero variation across 119 drawings. Only *which* numbers receive them
is random.

Only the first category is drawn. The rest are derived by rule from the major prizes, and every
rule below was verified against 132 drawings with **zero mismatches**:

| Category | How you win | Slots (ordinary, 6 majors) |
| --- | --- | --- |
| directo | your ball is pulled | 1,729 |
| centena (`C`) | a major landed in your hundred-block `[100k+1 … 100k+100]` | 99 per major |
| terminación (`T`) | a major shares your last three digits | 49 per major |
| aproximación (`A`) | a major is your number ± 1 | 2 per major |
| reintegro | your last digit = the first prize's last digit | 5,000 numbers |

The reintegro isn't listed as numbers — the PDF prints only the winning digit — so it's modelled
from the rule.

When two majors land in the same hundred-block, each awards centena to the other 99 numbers in
it, so the union covers all 100 and the two majors collect centena off each other. That's why
`prizes` has no unique constraint and why the invariant applies each rule **per major** before
unioning. Getting this wrong flags five perfectly valid drawings as corrupt.

### The answer

**P(any given number wins something) = 14.57% per ordinary drawing** — 5.17% from the prize table
plus 10% reintegro, less their measured overlap. Expected value is exact because the pool is
fixed: **$16.57 back on a $25 billete, a 66.3% return / 33.7% house edge.**

**The biggest lever is not which numbers but how many last digits you cover.** The reintegro goes
to every number ending in the first prize's last digit. Hold all ten last digits — ten pedazos is
enough — and one of your numbers wins it *every single drawing*, guaranteed. P(win something)
goes from ~71% (ten random numbers) to **100%**.

**Frequency does carry real information, but only regionally.** Per-*number* frequency is
hopeless: each ball has been drawn ~4.5 times, and against 50,000 simultaneous comparisons a ball
would need to be ~5× more likely than fair to stand out. The most-drawn ball (15 times vs a mean
of 4.5) is entirely unremarkable once you compare it against the distribution of the *maximum*
rather than the mean.

Pool numbers into 1,000-number regions, though, and a stable deviation appears:

| Test | Result |
| --- | --- |
| early-half ranking predicts late half | r = **+0.785**, p = 2e-11 |
| hot vs cold regions, out of sample | 36.00 vs 32.79 directo prizes per drawing |
| against a permutation null | **+9.9σ**, p = 9e-14 |

Regions around 26,000–34,000 run hot; 44,000–47,000 runs cold. The effect is a broad gradient, not
fine-grained lucky numbers — the achievable hit rate is nearly identical whether you bin into 250s
or 5,000s, so binning finer buys nothing.

It replicates out of sample and is not fragile. But it is **small**, and it lives in the wrong
place:

| Prize tier | Share of pool | Regional effect replicates? |
| --- | --- | --- |
| small ($100–200) | 31% | **yes** (t = 9.7, p < 0.0001) |
| mid ($400–2,000) | 5% | no (p = 0.83) |
| big (> $2,000) | 64% | no (p = 0.48) |

The edge is solid in the tier that barely moves the money and unproven in the tier that dominates
it — the major prizes test as uniform. Buying only from the highest-rate regions raises
P(winning something) from 14.57% to ~14.75% and the return from **66.3% to 66.9%**. You still
lose about 33 cents per dollar.

Pooling the tiers instead of splitting them makes this look far better than it is: an early version
of this analysis reported an 86.9% return for hot regions, which turned out to be three large
prizes landing there by luck. `report_money_edge` in `analyze.py` now counts only tiers whose
effect replicates out of sample.

So: cover all ten last digits, prefer the high-rate regions, use distinct hundred-blocks and
terminación classes so your numbers don't win or lose as a bloc. That is genuinely the optimum,
and it is still a losing bet — the house edge comes from the fixed pool being smaller than ticket
sales, and no selection strategy touches that.

## Example queries

```sql
-- Most frequently drawn numbers (clean drawings only)
select number, count(*) as wins
  from prizes_detailed
 where valid and category = ''
 group by number
 order by wins desc
 limit 20;

-- Total prize money paid out per drawing
select drawing_id, draw_date, sum(amount) as total_paid, count(*) as winners
  from prizes_detailed
 where valid
 group by drawing_id, draw_date
 order by draw_date desc;

-- Uniformity check: are top-prize last digits evenly distributed?
select first_prize_last_digit, count(*)
  from drawings
 where valid
 group by first_prize_last_digit
 order by first_prize_last_digit;

-- Which drawings failed validation, and why
select canonical_id, draw_date, problems from drawings where not valid;
```

## A note on the analysis goal

Lottery draws are independent random events, so frequency or gap analysis of past results won't
predict future draws — a number being "due" isn't a thing these data can support. What this corpus is
genuinely good for: auditing prize distributions, testing the draws for uniformity, and studying how
prize tiers and series are structured.
