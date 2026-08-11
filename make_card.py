#!/usr/bin/env python3
"""Generate the phone reference card from the database.

Writes a self-contained HTML page rating all 50 zones of 1,000 on both frequency and
prize money, plus a suggested set of numbers. Regenerate after each weekly load so the
card never drifts from the data:

    python make_card.py                    # writes card.html
    python make_card.py --out /tmp/x.html
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv

BALLS = 50_000
ZONE = 1_000
HOT_COUNT = 15

# Prize amounts are capped here before totalling a zone. 98.5% of prize slots are already
# at or below $400, so the cap barely touches the data while stopping a single jackpot from
# deciding a zone's rank. Uncapped totals do not repeat out of sample; capped ones do.
MONEY_CAP = 400


def fetch(conn, since=None, until=None):
    rows = conn.execute(
        """
        select d.canonical_id, p.number, p.amount, p.category
          from prizes p join drawings d on d.canonical_id = p.drawing_id
         where d.valid and d.kind = 'ordinary' and d.major_count = 6
           and (%s::int is null or d.year >= %s::int)
           and (%s::int is null or d.year <= %s::int)
         order by d.draw_date
        """,
        (since, since, until, until),
    ).fetchall()

    order, seen = [], set()
    for drawing_id, *_ in rows:
        if drawing_id not in seen:
            seen.add(drawing_id)
            order.append(drawing_id)
    index = {d: i for i, d in enumerate(order)}
    zones = BALLS // ZONE

    hits = np.zeros((len(order), zones))
    money = np.zeros((len(order), zones))
    for drawing_id, number, amount, category in rows:
        row, zone = index[drawing_id], (number - 1) // ZONE
        if category == "":
            hits[row, zone] += 1
            money[row, zone] += min(amount, MONEY_CAP)
    return len(order), hits.mean(axis=0), money.mean(axis=0)


def pick_numbers(ranked_zones, count=10, seed=7):
    """One number per last digit, drawn from the best-rated zones first."""
    rng = np.random.default_rng(seed)
    chosen: list[tuple[int, int]] = []
    used: dict[int, int] = {}
    for position in range(count):
        digit = position % 10
        for zone in ranked_zones:
            if used.get(zone, 0) >= 2:  # allow at most two per zone
                continue
            low = zone * ZONE + 1
            options = [n for n in range(low, low + ZONE) if n % 10 == digit]
            if not options:
                continue
            chosen.append((int(rng.choice(options)), zone))
            used[zone] = used.get(zone, 0) + 1
            break
    return sorted(chosen, key=lambda pair: pair[0] % 10)


def build(drawings, freq, cash) -> str:
    hot = set(np.argsort(freq)[::-1][:HOT_COUNT].tolist())
    cold = set(np.argsort(freq)[:HOT_COUNT].tolist())
    rich = set(np.argsort(cash)[::-1][:HOT_COUNT].tolist())
    both = sorted(hot & rich)

    # Best-rated first: double stars, then frequency-only stars, then the rest.
    ranked = sorted(
        range(len(freq)),
        key=lambda z: (0 if z in both else 1 if z in hot else 2, -freq[z]),
    )
    picks = pick_numbers(ranked)

    star_rows = "\n".join(
        f'      <div class="star-zone"><span class="zr">{z * ZONE + 1:,}</span>'
        f'<span class="dash">–</span><span class="zr">{(z + 1) * ZONE:,}</span></div>'
        for z in both
    )

    pick_cells = "\n".join(
        f'      <div class="pick"><span class="num">{n:05d}</span>'
        f'<span class="tail">ends {n % 10}</span></div>'
        for n, _ in picks
    )

    body_rows = []
    for zone in np.argsort(freq)[::-1]:
        low, high = zone * ZONE + 1, (zone + 1) * ZONE
        f_pct = (freq[zone] / freq.mean() - 1) * 100
        m_pct = (cash[zone] / cash.mean() - 1) * 100
        if zone in both:
            rating, cls = "★★", "s2"
        elif zone in hot:
            rating, cls = "★", "s1"
        elif zone in cold:
            rating, cls = "avoid", "s0"
        else:
            rating, cls = "", ""
        body_rows.append(
            f'        <tr class="{cls}">'
            f'<td class="z">{low:,}–{high:,}</td>'
            f'<td class="n">{freq[zone]:.1f}</td>'
            f'<td class="n {"pos" if f_pct > 0 else "neg"}">{f_pct:+.1f}%</td>'
            f'<td class="n">${cash[zone]:,.0f}</td>'
            f'<td class="n {"pos" if m_pct > 0 else "neg"}">{m_pct:+.1f}%</td>'
            f'<td class="r">{rating}</td></tr>'
        )

    return TEMPLATE.format(
        updated=date.today().isoformat(),
        drawings=drawings,
        star_count=len(both),
        star_rows=star_rows,
        pick_cells=pick_cells,
        table_rows="\n".join(body_rows),
        avg_freq=freq.mean(),
    )


TEMPLATE = """<title>Lotería Tradicional — Zone Ratings</title>

<style>
  :root {{
    --ink:#14261f; --ink-soft:#4a5a51; --paper:#f6f7f3; --card:#fff; --rule:#ccd3c9;
    --buy:#1f6f43; --buy-fill:#1f6f4318; --avoid:#9e3a2a; --avoid-fill:#9e3a2a14;
    --gold:#8f7220; --gold-fill:#a8862c1a; --neutral:#d8ddd5;
    --mono: ui-monospace, "SF Mono", SFMono-Regular, "Cascadia Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ink:#e6ece4; --ink-soft:#97a89b; --paper:#0e1410; --card:#161e18; --rule:#2b372d;
      --buy:#5fb381; --buy-fill:#5fb3811f; --avoid:#d97a66; --avoid-fill:#d97a6618;
      --gold:#d9b45f; --gold-fill:#d9b45f1a; --neutral:#263025;
    }}
  }}
  :root[data-theme="dark"] {{
    --ink:#e6ece4; --ink-soft:#97a89b; --paper:#0e1410; --card:#161e18; --rule:#2b372d;
    --buy:#5fb381; --buy-fill:#5fb3811f; --avoid:#d97a66; --avoid-fill:#d97a6618;
    --gold:#d9b45f; --gold-fill:#d9b45f1a; --neutral:#263025;
  }}
  :root[data-theme="light"] {{
    --ink:#14261f; --ink-soft:#4a5a51; --paper:#f6f7f3; --card:#fff; --rule:#ccd3c9;
    --buy:#1f6f43; --buy-fill:#1f6f4318; --avoid:#9e3a2a; --avoid-fill:#9e3a2a14;
    --gold:#8f7220; --gold-fill:#a8862c1a; --neutral:#d8ddd5;
  }}

  body {{ background:var(--paper); color:var(--ink); font-family:var(--sans);
         line-height:1.55; -webkit-text-size-adjust:100%; }}
  .wrap {{ max-width:36rem; margin:0 auto; padding:1.75rem 1.05rem 3rem;
           display:flex; flex-direction:column; gap:1.3rem; }}

  .eyebrow {{ font-family:var(--mono); font-size:.66rem; letter-spacing:.16em;
              text-transform:uppercase; color:var(--ink-soft); }}
  h1 {{ font-size:clamp(1.6rem,6.4vw,2.05rem); line-height:1.1; letter-spacing:-.02em;
        font-weight:640; margin:0; text-wrap:balance; }}
  .masthead {{ display:flex; flex-direction:column; gap:.4rem; }}

  section {{ background:var(--card); border:1px solid var(--rule); border-radius:3px;
             padding:1rem .95rem 1.1rem; display:flex; flex-direction:column; gap:.75rem; }}
  h2 {{ font-family:var(--mono); font-size:.7rem; letter-spacing:.14em;
        text-transform:uppercase; color:var(--ink-soft); font-weight:600; margin:0;
        padding-bottom:.6rem; border-bottom:1px solid var(--rule); }}
  p {{ margin:0; }}
  .note {{ font-size:.85rem; color:var(--ink-soft); }}

  .star-zones {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));
                 gap:.4rem; }}
  .star-zone {{ background:var(--gold-fill); border:1px solid var(--gold);
                border-radius:2px; padding:.5rem .4rem; text-align:center;
                font-family:var(--mono); font-size:.92rem;
                font-variant-numeric:tabular-nums; color:var(--gold);
                display:flex; justify-content:center; gap:.3rem; }}
  .star-zone .dash {{ opacity:.6; }}

  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:.8rem; }}
  thead th {{ font-family:var(--mono); font-size:.6rem; letter-spacing:.09em;
              text-transform:uppercase; color:var(--ink-soft); font-weight:600;
              text-align:right; padding:.3rem .45rem .5rem; border-bottom:1px solid var(--rule);
              white-space:nowrap; }}
  thead th:first-child {{ text-align:left; }}
  tbody td {{ padding:.28rem .45rem; border-bottom:1px solid var(--rule);
              font-variant-numeric:tabular-nums; white-space:nowrap; }}
  tbody tr:last-child td {{ border-bottom:none; }}
  td.z {{ font-family:var(--mono); font-size:.78rem; }}
  td.n {{ font-family:var(--mono); text-align:right; }}
  td.r {{ text-align:right; color:var(--gold); letter-spacing:-.04em; font-size:.72rem; }}
  td.pos {{ color:var(--buy); }}
  td.neg {{ color:var(--avoid); }}
  tr.s2 {{ background:var(--gold-fill); }}
  tr.s1 {{ background:var(--buy-fill); }}
  tr.s0 {{ background:var(--avoid-fill); }}
  tr.s0 td.r {{ color:var(--avoid); font-family:var(--mono); font-size:.62rem;
                letter-spacing:.06em; text-transform:uppercase; }}

  .picks {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(5rem,1fr)); gap:.4rem; }}
  .pick {{ background:var(--buy-fill); border:1px solid var(--buy); border-radius:2px;
           padding:.45rem .2rem .35rem; text-align:center; display:flex;
           flex-direction:column; gap:.1rem; }}
  .pick .num {{ font-family:var(--mono); font-size:1.08rem; font-weight:600;
                font-variant-numeric:tabular-nums; letter-spacing:.03em; }}
  .pick .tail {{ font-family:var(--mono); font-size:.55rem; letter-spacing:.09em;
                 text-transform:uppercase; color:var(--ink-soft); }}

  footer {{ font-size:.78rem; color:var(--ink-soft); border-top:1px solid var(--rule);
            padding-top:.9rem; display:flex; flex-direction:column; gap:.5rem; }}
  footer b {{ color:var(--ink); font-weight:600; }}
</style>

<div class="wrap">

  <header class="masthead">
    <div class="eyebrow">Sorteo Ordinario · {drawings} drawings · updated {updated}</div>
    <h1>Zone ratings</h1>
    <p class="note">
      All 50 zones of 1,000 numbers, rated two ways: how <b>often</b> numbers there win,
      and the <b>typical money</b> they collect — with each prize capped at $400 so one
      jackpot can't skew a zone. Same every week; a standing list, not a forecast.
    </p>
  </header>

  <section>
    <h2>★★ Best on both counts — {star_count} zones</h2>
    <div class="star-zones">
{star_rows}
    </div>
    <p class="note">
      Top 15 for frequency <i>and</i> top 15 for typical money. If a billete you're
      offered falls in one of these, that's your buy.
    </p>
  </section>

  <section>
    <h2>Ten numbers to play</h2>
    <div class="picks">
{pick_cells}
    </div>
    <p class="note">
      Drawn from the best-rated zones, covering every last digit 0–9 — so one of them
      collects a reintegro every single drawing, guaranteed.
    </p>
  </section>

  <section>
    <h2>All 50 zones</h2>
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th>Zone</th><th>Prizes<br>/draw</th><th>vs avg</th>
            <th>Typical $<br>/draw</th><th>vs avg</th><th></th>
          </tr>
        </thead>
        <tbody>
{table_rows}
        </tbody>
      </table>
    </div>
    <p class="note">Average is {avg_freq:.1f} prizes per zone per drawing.</p>
  </section>

  <footer>
    <div>
      <b>Why money is capped at $400.</b> 98.5% of prizes are already at or below it, so
      the cap barely touches the data — but without it, one jackpot decides a zone's rank
      and the ranking stops repeating. Uncapped, only 6 of the top 15 money zones are also
      top 15 for frequency; capped, 13 of 15 are.
    </div>
    <div>
      <b>What that agreement means.</b> Prize size is drawn from a separate tumbler, so it
      carries no zone information — a zone's hit count and its average prize size are
      unrelated. Money is "how often × how much", and only "how often" varies by zone. So
      ★★ is a sanity check that the two views agree, not a second edge on top of the first.
    </div>
    <div>
      <b>Frequency counts</b> only prizes where your number itself was drawn (1,729 of the
      2,629 each week). The rest are handed out by rule around the main numbers, which
      dumps 99 prizes into one random zone per drawing and swamps the pattern.
    </div>
  </footer>

</div>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="card.html")
    parser.add_argument("--since", type=int, metavar="YEAR")
    parser.add_argument("--until", type=int, metavar="YEAR")
    args = parser.parse_args(argv)

    load_dotenv()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set — see .env.example", file=sys.stderr)
        return 1

    with psycopg.connect(dsn, connect_timeout=20) as conn:
        drawings, freq, cash = fetch(conn, args.since, args.until)

    Path(args.out).write_text(build(drawings, freq, cash), encoding="utf-8")
    print(f"wrote {args.out} from {drawings} drawings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
