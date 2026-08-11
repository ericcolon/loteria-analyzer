#!/usr/bin/env python3
"""Generate the phone reference card from the database.

Writes a self-contained HTML page rating all 50 zones of 1,000 on both frequency and
prize money, plus a suggested set of numbers. Regenerate after each weekly load so the
card never drifts from the data:

    python make_card.py                    # writes docs/index.html (what GitHub Pages serves)
    python make_card.py --out /tmp/x.html
    python make_card.py --fragment         # body only, for embedding elsewhere
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
        f'<span class="tail">termina en {n % 10}</span></div>'
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
            rating, cls = "evitar", "s0"
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

    # The table is by far the tallest block. Splitting it in half lets the desktop
    # layout put the two halves side by side instead of leaving a column of dead space;
    # on a phone the halves simply stack and read as one continuous list.
    middle = (len(body_rows) + 1) // 2
    return TEMPLATE.format(
        updated=date.today().isoformat(),
        drawings=drawings,
        star_count=len(both),
        star_rows=star_rows,
        pick_cells=pick_cells,
        table_rows_a="\n".join(body_rows[:middle]),
        table_rows_b="\n".join(body_rows[middle:]),
        avg_freq=freq.mean(),
    )


# GitHub Pages serves this file as-is, so it needs a real document around it. The viewport
# line is the one that matters most: without it mobile Safari lays the page out at desktop
# width and zooms out, which is useless for a card you read on your phone at the vendor.
DOC_OPEN = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="description" content="Calificación de las 50 zonas de la Lotería Tradicional de Puerto Rico por frecuencia de premios y dinero típico.">
</head>
<body>
"""

DOC_CLOSE = """</body>
</html>
"""


TEMPLATE = """<title>Lotería Tradicional — Calificación de Zonas</title>

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

  /* Minimal reset: this page is served standalone, with no framework underneath it. */
  *, *::before, *::after {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink); font-family:var(--sans);
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

  /* Deliberately quieter than the data sections — a notice, not a headline. */
  .banner {{ font-size:.76rem; color:var(--ink-soft); border:1px dashed var(--rule);
             border-radius:2px; padding:.5rem .65rem; }}
  .aviso {{ background:transparent; border:1px solid var(--rule); border-left:3px solid var(--ink-soft);
            font-size:.8rem; color:var(--ink-soft); gap:.6rem; }}
  .aviso h2 {{ color:var(--ink); }}
  .aviso b {{ color:var(--ink); font-weight:600; }}
  .aviso .warn {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.03em;
                  color:var(--ink); background:var(--avoid-fill);
                  border-left:3px solid var(--avoid); padding:.55rem .65rem;
                  border-radius:2px; }}
  .aviso .en {{ font-size:.74rem; font-style:italic; padding-top:.15rem;
                border-top:1px solid var(--rule); }}
  .aviso a {{ color:var(--ink); }}

  /* Stacked by default — phone first. */
  .content, .bottom {{ display:flex; flex-direction:column; gap:1.3rem; }}
  .tables {{ display:flex; flex-direction:column; gap:1rem; }}

  /* On a desktop the single 36rem column left most of the window empty. The 50-row
     table is by far the tallest block, so it gets its own column beside the summary
     cards, and the two closing notes sit side by side underneath. */
  @media (min-width: 62rem) {{
    .wrap {{ max-width:72rem; padding:2.25rem 2rem 3.5rem; gap:1.6rem; }}
    h1 {{ font-size:2.45rem; }}
    /* Keep running text at a readable measure even though the page is now wide. */
    .masthead > .note, .masthead > .banner {{ max-width:46rem; }}
    .content {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr);
                align-items:start; gap:1.6rem; }}
    .tables {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr);
               align-items:start; gap:1.5rem; }}
    .bottom {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr);
               align-items:start; gap:1.6rem; }}
    /* The footer now sits beside the aviso, so give it a matching panel instead of a
       bare top rule. The aviso keeps its accent bar so it still reads as the notice. */
    footer {{ border:1px solid var(--rule); border-radius:3px;
              padding:1rem .95rem 1.1rem; }}
    /* The table has room to breathe now. */
    table {{ font-size:.84rem; }}
    tbody td {{ padding:.34rem .5rem; }}
  }}
</style>

<div class="wrap" lang="es">

  <header class="masthead">
    <div class="eyebrow">Sorteo Ordinario · {drawings} sorteos · actualizado {updated}</div>
    <h1>Qué zonas comprar</h1>
    <p class="note">
      Las 50 zonas de 1,000 números, calificadas de dos maneras: <b>con qué frecuencia</b>
      ganan los números de esa zona, y el <b>dinero típico</b> que cobran — con cada premio
      limitado a $400 para que un premio mayor no distorsione la zona. Es igual todas las
      semanas: una lista fija, no un pronóstico.
    </p>
    <p class="banner">
      Proyecto de ciencia de datos. <b>No es asesoría para apostar</b> — lee el aviso al
      final.
    </p>
  </header>

  <div class="content">

  <section>
    <h2>★★ Mejores en ambas columnas — {star_count} zonas</h2>
    <div class="star-zones">
{star_rows}
    </div>
    <p class="note">
      Están entre las 15 de mayor frecuencia <i>y</i> entre las 15 de más dinero típico. Si
      el billete que te ofrecen cae en una de estas, ese es el que compras.
    </p>
  </section>

  <section>
    <h2>Diez números para jugar</h2>
    <div class="picks">
{pick_cells}
    </div>
    <p class="note">
      Todos salen de las zonas mejor calificadas y entre ellos cubren todas las últimas
      cifras del 0 al 9 — así que uno siempre cobra <b>reintegro en cada sorteo</b>,
      garantizado.
    </p>
  </section>

  </div>

  <section>
    <h2>Las 50 zonas</h2>
    <div class="tables">
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Zona</th><th>Premios<br>/sorteo</th><th>vs prom.</th>
              <th>$ típico<br>/sorteo</th><th>vs prom.</th><th></th>
            </tr>
          </thead>
          <tbody>
{table_rows_a}
          </tbody>
        </table>
      </div>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Zona</th><th>Premios<br>/sorteo</th><th>vs prom.</th>
              <th>$ típico<br>/sorteo</th><th>vs prom.</th><th></th>
            </tr>
          </thead>
          <tbody>
{table_rows_b}
          </tbody>
        </table>
      </div>
    </div>
    <p class="note">El promedio es {avg_freq:.1f} premios por zona por sorteo.</p>
  </section>

  <div class="bottom">
  <footer>
    <div>
      <b>El límite de $400.</b> Existe porque el 98.5% de los premios ya son de $400 o
      menos: así el límite casi no altera los datos, pero sin él un solo premio mayor decide
      la posición de una zona y la calificación deja de repetirse. Sin límite, solo 6 de las
      15 zonas de más dinero están también entre las 15 de mayor frecuencia; con el límite,
      13 de 15.
    </div>
    <div>
      <b>Qué significa que ambas columnas coincidan.</b> El monto del premio sale de un bombo
      aparte, así que no dice nada sobre la zona: la cantidad de premios de una zona y el
      monto promedio no tienen relación. El dinero es "cuántas veces × cuánto", y solo
      "cuántas veces" cambia según la zona. Por eso ★★ confirma que las dos columnas están
      de acuerdo; no es una ventaja adicional encima de la otra.
    </div>
    <div>
      <b>La frecuencia cuenta</b> solo los premios donde salió tu número (1,729 de los 2,629
      de cada semana). Los demás se reparten por regla alrededor de los premios mayores, lo
      que deja 99 premios en una zona al azar en cada sorteo y tapa el patrón.
    </div>
  </footer>

  <section class="aviso">
    <h2>Aviso</h2>
    <p>
      Esta página es un <b>proyecto personal de ciencia de datos e inteligencia
      artificial</b>. No es asesoría para apostar, no promuevo el juego y no sugiero que
      nadie compre billetes.
    </p>
    <p>
      Los números y las zonas salen de un análisis estadístico de listas oficiales de
      premios ya publicadas. <b>No predicen nada.</b> Cada sorteo es independiente y ninguna
      selección de números cambia eso. Las diferencias entre zonas son pequeñas y solo
      aparecen a lo largo de años, no en un sorteo.
    </p>
    <p>
      La matemática sin adornos: la Lotería reparte alrededor de <b>66 centavos por cada
      dólar</b> que se juega. A la larga se pierde dinero, no importa qué números escojas.
      Nada en esta página cambia eso ni existe manera de ganarle.
    </p>
    <p>
      Tómalo como entretenimiento. El juego es solo para adultos, y solo con dinero que
      puedas perder sin que te afecte.
    </p>
    <p class="warn">
      LOS JUEGOS DE AZAR PUEDEN CREAR ADICCIÓN. Si jugar le causa problemas económicos,
      familiares y ocupacionales, llame a su proveedor de salud mental.
    </p>
    <p>
      En Puerto Rico, el <b>Programa de Ayuda a Jugadores Compulsivos (PAJC)</b> de ASSMCA
      atiende por la <b>Línea PAS</b> (Primera Ayuda Sicosocial):
      <b>1-800-981-0023</b> — 24 horas al día, los 7 días de la semana.
      Más información en <a href="https://www.assmca.pr.gov/">assmca.pr.gov</a> y en la
      página de <a href="https://loteriasdepuertorico.pr.gov/loteria-tradicional/juego-responsable/">juego
      responsable</a> de la Lotería.
    </p>
    <p>
      Sin afiliación alguna con la Lotería de Puerto Rico, el Negociado de las Loterías ni
      el Departamento de Hacienda. La lista oficial de premios publicada por la Lotería es
      la única referencia válida para cobrar.
    </p>
    <p class="en">
      This page is a personal data-science project, not gambling advice. It predicts
      nothing; the lottery returns about 66 cents per dollar wagered and no choice of
      numbers changes that. For entertainment only. Not affiliated with the Lotería de
      Puerto Rico. Gambling can be addictive — in Puerto Rico, ASSMCA's Línea PAS is
      1-800-981-0023, 24/7.
    </p>
  </section>
  </div>

</div>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/index.html",
                        help="output path (default: docs/index.html, served by Pages)")
    parser.add_argument("--fragment", action="store_true",
                        help="emit body content only, without the html/head wrapper")
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

    page = build(drawings, freq, cash)
    if not args.fragment:
        page = DOC_OPEN + page + DOC_CLOSE
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"wrote {args.out} from {drawings} drawings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
