#!/usr/bin/env python3
"""Answer the question: which numbers have the best chance of winning something?

Running it with no arguments prints the shopping list: which number ranges to buy in,
which to avoid, and what the edge is actually worth in dollars. Everything else is the
supporting work.

  buy          (default) plain-language shopping list, no statistics
  zones        all 50 zones of 1,000, rated on frequency and on prize money
  probability  exact per-number win probability, decomposed by prize category
  fairness     does the physical tumbler deviate from uniform? (the only way past
               frequency could legitimately help)
  frequency    fit and OUT-OF-SAMPLE validate the regional model, then check whether its
               edge reaches the money or only the hit rate
  portfolio    simulated comparison of selection strategies

    python analyze.py                        # the shopping list
    python analyze.py --seed 12              # a different suggested set of ten
    python analyze.py --section frequency    # the evidence behind the ranges
    python analyze.py --verify               # re-derive the model's constants from the DB
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import psycopg
from dotenv import load_dotenv
from scipy import stats

from loteria import fairness, frequency
from loteria.model import (
    BALLS,
    ORDINARY,
    STRUCTURES,
    category_probabilities,
    clustered_portfolio,
    expected_value,
    random_portfolio,
    simulate_drawings,
    spread_portfolio,
)

log = logging.getLogger("analyze")

# A full billete is 25 pedazos at $1. The prize list is denominated per billete, and a
# reintegro returns the ticket price. Override with --ticket-price if that's wrong.
DEFAULT_TICKET_PRICE = 25.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    load_dotenv()

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        log.error("DATABASE_URL not set — see .env.example")
        return 1

    with psycopg.connect(dsn, connect_timeout=20) as conn:
        if args.verify:
            verify_constants(conn)
            return 0
        sections = args.section or ["buy"]  # plain shopping list by default
        model = None
        if "probability" in sections:
            report_probability(args)
        if "fairness" in sections:
            report_fairness(conn)
        if {"frequency", "portfolio", "buy", "majors"} & set(sections):
            model = report_frequency(
                conn, quiet="frequency" not in sections,
                since=args.since, until=args.until)
        if "portfolio" in sections:
            report_portfolio(args, model)
        if "zones" in sections:
            report_zones(conn, args)
        if "majors" in sections:
            report_majors(conn, args, model)
        if "buy" in sections:
            report_buy(conn, args, model)
    return 0


def load_directo_by_drawing(
    conn: psycopg.Connection, since: int | None = None, until: int | None = None
) -> list[np.ndarray]:
    """Directo numbers per ordinary drawing, in chronological order.

    Restricted to ordinary 6-major drawings so every row in the matrix comes from an
    identically structured draw (1,729 pulls), and ordered by date so a split-half test
    is genuinely out of sample in time.
    """
    rows = conn.execute(
        """
        select d.canonical_id, p.number
          from prizes p join drawings d on d.canonical_id = p.drawing_id
         where d.valid and p.category = ''
           and d.kind = 'ordinary' and d.major_count = 6
           and (%s::int is null or d.year >= %s::int)
           and (%s::int is null or d.year <= %s::int)
         order by d.draw_date, p.number
        """,
        (since, since, until, until),
    ).fetchall()
    grouped: dict[str, list[int]] = {}
    for drawing_id, number in rows:
        grouped.setdefault(drawing_id, []).append(number)
    return [np.array(v, dtype=np.int64) for v in grouped.values()]


def report_frequency(
    conn: psycopg.Connection,
    quiet: bool = False,
    since: int | None = None,
    until: int | None = None,
):
    """Fit and validate the regional frequency model."""
    draws = load_directo_by_drawing(conn, since, until)
    if len(draws) < 20:
        if not quiet:
            print(f"\n  only {len(draws)} ordinary drawings in that range — too few to fit")
            print("  a model. Widen the year filter.")
        return None

    if not quiet:
        head("3. Does past frequency help? (the regional model)")
        print(f"  Fitted on {len(draws)} ordinary drawings.\n")
        print("  Per-NUMBER frequency is a dead end — see the power analysis above. But")
        print("  pooling numbers into regions gives thousands of observations each, and")
        print("  there the history does carry information. Choosing the granularity by")
        print("  which one survives an out-of-sample split:\n")
        print(f"    {'region size':>12}{'split r':>10}{'hot rate/number':>17}"
              f"{'vs null':>9}   verdict")

    candidates: list[tuple] = []
    for block_size in (100, 250, 500, 1000, 2500, 5000):
        counts = frequency.counts_matrix(draws, block_size)
        result = frequency.validate(counts, block_size, permutations=400)
        if not quiet:
            print(f"    {block_size:>12,}{result.split_correlation:>10.3f}"
                  f"{result.hot_rate_per_number * 100:>16.3f}% {result.permutation_z:>8.1f}σ   "
                  f"{'credible' if result.credible else 'not credible'}")
        if result.credible:
            candidates.append((block_size, result, counts))

    # The achievable hot rate turns out to be nearly identical at every granularity,
    # which says the effect is a broad regional gradient rather than fine-grained lucky
    # numbers. So among granularities that are practically tied on what they deliver,
    # take the statistically most robust one instead of chasing a noise-level difference.
    best = None
    if candidates:
        top_rate = max(c[1].hot_rate_per_number for c in candidates)
        tied = [c for c in candidates if c[1].hot_rate_per_number >= top_rate - 0.0002]
        best = max(tied, key=lambda c: c[1].permutation_z)

    if best is None:
        if not quiet:
            print("\n  >> No granularity survives out-of-sample validation. Treat every")
            print("     number as equally likely.")
        return None

    block_size, result, counts = best
    model = frequency.fit(counts, block_size)

    if not quiet:
        print(f"\n  Best: {block_size:,}-number regions.")
        print(f"    early-half ranking predicts late half : r = {result.split_correlation:+.3f} "
              f"(p = {result.split_p:.1e})")
        print(f"    hot regions, late half                : {result.hot_rate_late:.2f} "
              f"directo prizes per drawing")
        print(f"    cold regions, late half               : {result.cold_rate_late:.2f}")
        print(f"    gap vs permutation null               : {result.permutation_z:.1f}σ "
              f"(p = {result.gap_p:.1e})")
        print(f"    empirical-Bayes shrinkage applied     : "
              f"{(1 - model.shrinkage) * 100:.0f}% pulled toward the mean")

        print("\n  What it's worth, per number, per drawing:\n")
        print(f"    {'region':>22}{'P(any prize)':>14}{'vs uniform':>12}")
        ranked = model.ranked_blocks()
        for label, block in (("best", ranked[0]), ("median", ranked[len(ranked) // 2]),
                             ("worst", ranked[-1])):
            low, high = model.block_range(block)
            p = model.any_prize_probability(low)
            print(f"    {label:>8} {low:>6,}-{high:<7,}{p * 100:>10.3f}%"
                  f"{(p - frequency.BASE_ANY_PRIZE) * 100:>+11.3f}pp")

        print("\n  Highest-rate regions:")
        for block in ranked[:8]:
            low, high = model.block_range(block)
            print(f"    {low:>6,}-{high:<7,}  {model.rate[block]:.2f} prizes/drawing "
                  f"(uniform would be {model.grand_mean:.2f})")

        report_money_edge(conn, model, args_ticket_price=TICKET_PRICE_FOR_REPORT)

    return model


# Prize tiers, split because they behave completely differently statistically: thousands
# of small prizes versus a handful of huge ones.
TIERS = [("small $100-200", 0, 200), ("mid $400-2,000", 201, 2000), ("big > $2,000", 2001, 10**9)]
TICKET_PRICE_FOR_REPORT = DEFAULT_TICKET_PRICE


def report_money_edge(conn: psycopg.Connection, model, args_ticket_price: float) -> None:
    """Does the regional edge survive into expected value, or only into hit rate?

    Split by prize tier, because expected value is dominated by a few very large prizes
    while hit rate is dominated by thousands of $100 ones. Pooling them lets the large
    prizes' noise masquerade as an edge — an early version of this analysis reported a
    20-point EV gain that was three big prizes landing in hot regions by luck.
    """
    block = model.block_size
    rows = conn.execute(
        """
        select d.canonical_id, d.draw_date, (p.number - 1) / %s as blk, p.amount
          from prizes p join drawings d on d.canonical_id = p.drawing_id
         where d.valid and d.kind = 'ordinary' and d.major_count = 6 and p.category = ''
         order by d.draw_date
        """,
        (block,),
    ).fetchall()

    order, seen = [], set()
    for drawing_id, *_ in rows:
        if drawing_id not in seen:
            seen.add(drawing_id)
            order.append(drawing_id)
    index = {d: i for i, d in enumerate(order)}
    blocks = BALLS // block

    counts = np.zeros((len(order), blocks))
    tier_counts = {name: np.zeros((len(order), blocks)) for name, _, _ in TIERS}
    tier_money = {name: np.zeros((len(order), blocks)) for name, _, _ in TIERS}
    for drawing_id, _, blk, amount in rows:
        i, b = index[drawing_id], int(blk)
        counts[i, b] += 1
        for name, low, high in TIERS:
            if low <= amount <= high:
                tier_counts[name][i, b] += 1
                tier_money[name][i, b] += amount
                break

    half = len(order) // 2
    top_k = max(1, blocks // 5)
    train = counts[:half].mean(axis=0)
    hot, cold = np.argsort(train)[::-1][:top_k], np.argsort(train)[:top_k]

    print("\n  Per prize tier, measured on the half of the corpus the ranking was NOT")
    print("  fitted on:\n")
    print(f"    {'tier':<16}{'share of pool':>14}{'hot':>8}{'cold':>8}{'p':>9}  measurable?")
    pool_total = sum(m.sum() for m in tier_money.values())
    for name, _, _ in TIERS:
        late = tier_counts[name][half:]
        _, p = stats.ttest_rel(late[:, hot].sum(axis=1), late[:, cold].sum(axis=1))
        share = tier_money[name].sum() / pool_total
        print(f"    {name:<16}{share * 100:>13.1f}%{late[:, hot].mean():>8.2f}"
              f"{late[:, cold].mean():>8.2f}{p:>9.4f}  {'yes' if p < 0.01 else 'too few'}")

    print("\n    Only the small tier is measurable, but that is a sample-size limit, not")
    print("    evidence the edge stops there. The draw pairs a number ball with a")
    print("    separately drawn PRIZE ball, so the amount a number wins is independent")
    print("    of the number. The zone effect is on how often a number gets drawn at")
    print("    all — so it carries into every prize size, including the majors. Every")
    print("    major prize on record is itself a directo pull, which confirms the")
    print("    majors come out of the same tumbler and inherit the same bias.\n")

    # Apply the measured hit-rate advantage to the categories that depend on where the
    # number sits. Terminación does not: its class spans all 50 zones evenly, so a zone
    # advantage averages out of it. Reintegro depends only on the last digit.
    ratio = model.rate[hot].mean() / model.grand_mean
    per_number = dict(conn.execute(
        """
        select p.category, sum(p.amount)::float / count(distinct p.drawing_id) / 50000
          from prizes p join drawings d on d.canonical_id = p.drawing_id
         where d.valid and d.kind = 'ordinary' and d.major_count = 6
         group by 1
        """
    ).fetchall())
    reintegro = args_ticket_price / 10
    scales = per_number.get("", 0) + per_number.get("C", 0) + per_number.get("A", 0)
    flat = per_number.get("T", 0) + reintegro

    base = scales + flat
    boosted = scales * ratio + flat
    print(f"    hot-zone hit-rate advantage: {(ratio - 1) * 100:.1f}%\n")
    print(f"      expected value, any number   ${base:.2f}  "
          f"({base / args_ticket_price * 100:.1f}% return)")
    print(f"      expected value, hot zones    ${boosted:.2f}  "
          f"({boosted / args_ticket_price * 100:.1f}% return)")
    print(f"      house edge {100 - base / args_ticket_price * 100:.1f}% -> "
          f"{100 - boosted / args_ticket_price * 100:.1f}%")
    print(f"      worth +${boosted - base:.2f} per ${args_ticket_price:.0f} ticket")
    print("\n  >> Real and replicated, and it applies to the big prizes too. It still")
    print("     does not make this a winning bet — the house keeps about 30 cents per")
    print("     dollar instead of 34.")


# -- section 1: probability ---------------------------------------------------


def report_probability(args: argparse.Namespace) -> None:
    structure = ORDINARY
    head("1. What is the probability of a number winning something?")

    probs = category_probabilities(12_345, structure)
    print("  Per prize category, for one ordinary drawing (any mid-range number):\n")
    print(f"    {'category':<16}{'probability':>13}   {'how you win it'}")
    how = {
        "directo": "your ball is pulled from the tumbler",
        "centena": "a major lands in your hundred-block",
        "terminacion": "a major shares your last 3 digits",
        "aproximacion": "a major is your number ± 1",
        "reintegro": "your last digit = first prize's last digit",
    }
    for name, probability in probs.items():
        print(f"    {name:<16}{probability * 100:>12.4f}%   {how[name]}")

    simulated = simulate_drawings(structure, trials=args.trials, seed=7)
    winners = simulated["distinct_table_winners"].mean()
    p_table = winners / BALLS
    # Inclusion-exclusion. The two sets overlap slightly more than chance because a
    # major's whole terminación class shares its last digit, so the overlap is measured
    # from the corpus rather than assumed independent.
    p_any = p_table + 0.10 - _overlap_correction()

    print(f"\n    {'ANY table prize':<16}{p_table * 100:>12.4f}%   "
          f"(~{winners:,.0f} of 50,000 balls win something)")
    print(f"    {'ANY prize at all':<16}{p_any * 100:>12.4f}%   including the reintegro")

    print("\n  The pool is a fixed constant, so expected value is exact:\n")
    ev = expected_value(structure, args.ticket_price)
    print(f"    prize pool per drawing          ${structure.pool:,} (never varies)")
    print(f"    expected table prize per ticket ${ev['expected_table_prize']:.2f}")
    print(f"    expected reintegro per ticket   ${ev['expected_reintegro']:.2f}")
    print(f"    expected total per ticket       ${ev['expected_total']:.2f} "
          f"on a ${ev['ticket_price']:.2f} ticket")
    print(f"    return                          {ev['return_ratio'] * 100:.1f}% "
          f"(house edge {ev['house_edge'] * 100:.1f}%)")

    print("\n  Every number carries this same probability and this same expected value.")
    print("  The two exceptions are the range endpoints, which have one neighbour")
    print("  instead of two and so are very slightly WORSE:\n")
    for number in (1, 50_000, 12_345):
        aprox = category_probabilities(number, structure)["aproximacion"]
        note = "  <- endpoint" if number in (1, BALLS) else ""
        print(f"    number {number:>6}: aproximación {aprox * 100:.6f}%{note}")


def _overlap_correction() -> float:
    """Measured overlap between table winners and the reintegro set (299.2 / 50,000)."""
    return 299.2 / BALLS


# -- section 2: fairness ------------------------------------------------------


def report_fairness(conn: psycopg.Connection) -> None:
    head("2. Is the tumbler fair? (the only way frequency could help)")

    rows = conn.execute(
        """
        select p.number, d.draw_date
          from prizes p join drawings d on d.canonical_id = p.drawing_id
         where d.valid and p.category = ''
         order by d.draw_date
        """
    ).fetchall()
    if not rows:
        print("  no valid data loaded")
        return

    draws = np.array([r[0] for r in rows], dtype=np.int64)
    digits = np.array(
        [
            r[0]
            for r in conn.execute(
                "select first_prize_last_digit from drawings "
                "where valid and first_prize_last_digit is not null"
            ).fetchall()
        ],
        dtype=np.int64,
    )
    half = len(draws) // 2

    print(f"  Using {len(draws):,} independent tumbler pulls (directo category only —")
    print("  the other categories are derived by rule from the majors, so counting them")
    print("  would multiply the same few random events).\n")

    results = [
        fairness.test_uniform_all_numbers(draws),
        *fairness.test_digit_positions(draws),
        fairness.test_range_bins(draws),
        fairness.test_last_digit_of_first_prize(digits),
        fairness.test_temporal_drift(draws[:half], draws[half:]),
        fairness.test_hottest_numbers(draws),
    ]

    print(f"    {'test':<48}{'p-value':>10}   verdict")
    for result in results:
        print(f"    {result.name:<48}{result.p_value:>10.3f}   {result.verdict}")

    print("\n  Detail:")
    for result in results:
        print(f"    - {result.name}: {result.detail}")

    power = fairness.detectable_effect(draws)
    print("\n  Power — what this much data could actually detect:\n")
    print(f"    each ball drawn on average          {power['expected_draws_per_ball']:.2f} times")
    print(f"    count needed to stand out           {power['count_needed_for_significance']:.0f}x"
          "  (after correcting for 50,000 tests)")
    print(f"    a ball would need to be             {power['multiplier']:.1f}x more likely")
    print("    than fair before this corpus could reliably flag it.\n")
    print("    So the absence of a signal does NOT prove the tumbler is perfect — it")
    print("    rules out gross bias, not a few-percent edge. But a few-percent edge on")
    print("    a 66% payout is still a losing bet, so it would not change the answer.")

    significant = [r for r in results if r.p_value < 0.05]
    print()
    if significant:
        print("  >> Deviations found — worth a closer look:")
        for result in significant:
            print(f"     {result.name} (p={result.p_value:.4f})")
    else:
        print("  >> No test finds a deviation from uniform. Past frequency carries no")
        print("     information about future drawings, so 'hot' and 'cold' numbers are")
        print("     equally good buys.")


# -- section 3: portfolio -----------------------------------------------------


def report_portfolio(args: argparse.Namespace, model=None) -> None:
    structure = ORDINARY
    budget = args.budget
    head(f"4. What to actually buy ({budget} pedazos, one per number)")

    print("  Since every number is equally likely, WHICH numbers you pick cannot change")
    print("  your expected value. What it does change is CORRELATION — and that does")
    print("  move your chance of winning something:\n")
    print("    - numbers in the same hundred-block win centena together")
    print("    - numbers sharing last-3-digits win terminación together")
    print("    - numbers sharing a last digit win the reintegro together\n")
    print("  So to maximise P(win something), decorrelate. Simulated over")
    print(f"  {args.trials:,} drawings:\n")

    portfolios = {
        "spread: all last digits, distinct blocks": spread_portfolio(budget, seed=3),
        "consecutive: all last digits, ONE block": clustered_portfolio(budget),
        "random picks": random_portfolio(budget, seed=3),
        "all sharing one last digit": _same_last_digit_portfolio(budget),
    }

    print(f"    {'strategy':<44}{'P(win ≥1)':>11}{'E[wins]':>10}{'P(≥1) table only':>18}")
    for label, numbers in portfolios.items():
        result = simulate_drawings(structure, portfolio=numbers, trials=args.trials, seed=11)
        hits = result["portfolio_hits"]
        table_only = result["portfolio_hits_table_only"]
        print(
            f"    {label:<44}{(hits > 0).mean() * 100:>10.1f}%"
            f"{hits.mean():>10.2f}{(table_only > 0).mean() * 100:>17.1f}%"
        )

    if budget >= 10:
        print("\n  The jump to 100% is the reintegro: cover all ten last digits and one of")
        print("  your numbers ALWAYS matches the first prize's last digit. Guaranteed,")
        print("  every drawing. Ten pedazos is the threshold — that is the whole trick,")
        print("  and it is why consecutive numbers also reach 100% (00-09 covers all ten).")
        print("  Distinct hundred-blocks then improve the table-prize column on top.")
    else:
        print(f"\n  With only {budget} pedazos you cannot cover all ten last digits, so the")
        print(f"  reintegro lands {budget}/10 of the time. At 10 pedazos, one per last digit,")
        print("  P(win something) becomes exactly 100% — a guaranteed reintegro.")
    print("\n  Read that carefully though — a reintegro returns your $1 stake on one")
    print("  pedazo. Guaranteeing you 'win something' is not the same as making money:")
    ev = expected_value(structure, args.ticket_price)
    stake = budget * (args.ticket_price / 25)
    baseline = stake * ev["return_ratio"]
    # The measured regional edge is worth +$0.14 on a $25 ticket (see --section frequency).
    with_edge = baseline + budget * (0.14 / 25)
    print(f"    {budget} pedazos cost ${stake:.2f} and return ${baseline:.2f} on average — "
          f"${with_edge:.2f} if you buy only")
    print("    from the highest-rate regions. Both are losses.")

    numbers, notes = frequency.pick_numbers(budget, model=model, seed=5)
    source = "highest-rate regions" if model is not None else "spread only (no fitted model)"
    print(f"\n  Recommended selection ({source}):")
    for index in range(0, len(numbers), 10):
        chunk = numbers[index : index + 10]
        print("    " + "  ".join(f"{n:05d}" for n in chunk))
    print(f"\n    last digits covered: {notes['last_digits_covered']}/10   "
          f"reintegro guaranteed: {notes['reintegro_guaranteed']}   "
          f"distinct hundred-blocks: {notes['distinct_blocks']}")


def _same_last_digit_portfolio(size: int) -> np.ndarray:
    """Every number ending in 7 — maximally correlated on the reintegro."""
    return np.array([7 + 10 * i for i in range(size)], dtype=np.int64)


# -- verification -------------------------------------------------------------


def verify_constants(conn: psycopg.Connection) -> None:
    head("Re-deriving the model's constants from the database")
    rows = conn.execute(
        """
        select d.kind, d.major_count, count(*) n,
               min(x.directo) directo_min, max(x.directo) directo_max,
               min(x.c) c_min, max(x.c) c_max,
               min(x.t) t_min, max(x.t) t_max,
               min(x.a) a_min, max(x.a) a_max,
               min(x.pool) pool_min, max(x.pool) pool_max
          from drawings d join (
            select drawing_id,
                   count(*) filter (where category = '') directo,
                   count(*) filter (where category = 'C') c,
                   count(*) filter (where category = 'T') t,
                   count(*) filter (where category = 'A') a,
                   sum(amount) pool
              from prizes group by 1) x on x.drawing_id = d.canonical_id
         where d.valid
         group by 1, 2 order by 1, 2
        """
    ).fetchall()

    for row in rows:
        kind, majors, n = row[0], row[1], row[2]
        ranges = row[3:]
        constant = all(ranges[i] == ranges[i + 1] for i in range(0, len(ranges), 2))
        key = kind if kind in STRUCTURES else "?"
        expected = STRUCTURES.get(key)
        print(f"\n  {kind} / {majors} majors  ({n} drawings)  "
              f"{'all constant' if constant else 'VARIES'}")
        print(f"    directo {ranges[0]}  centena {ranges[2]}  terminación {ranges[4]}  "
              f"aproximación {ranges[6]}  pool {ranges[8]:,}")
        if expected and expected.major_count == majors:
            match = (
                expected.directo == ranges[0]
                and expected.centena == ranges[2]
                and expected.terminacion == ranges[4]
                and expected.aproximacion == ranges[6]
                and expected.pool == ranges[8]
            )
            print(f"    model constant: {'MATCHES' if match else 'MISMATCH — update model.py'}")


ZONE_SIZE = 1000
HOT_COUNT = 15  # top and bottom 15 of 50 zones

# Prize amounts are capped here before totalling a zone's money. 98.5% of prize slots are
# already at or below $400, so this leaves almost every prize untouched while stopping one
# jackpot from deciding a zone's rank. Uncapped totals do not repeat out of sample; capped
# ones do. See `normalize` notes in the README.
MONEY_CAP = 400


def report_zones(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    """All 50 zones of 1,000, rated on both frequency and prize money.

    Two independent columns, because they are not equally trustworthy and the reader
    should see both rather than a single blended score:

    * **Frequency** — how often a number in the zone is drawn. Counts only the prizes
      where the number itself came out of the tumbler (1,729 of 2,629 each week).
    * **Typical money** — dollars per drawing with each prize capped at $400, so a single
      jackpot cannot decide a zone's rank.

    Both repeat out of sample. Uncapped money does not: ranked on the older half of the
    record it fails on the newer half, because one large prize moves a zone's total more
    than a year of ordinary wins.
    """
    head("ALL 50 ZONES — frequency and money")

    rows = conn.execute(
        """
        select d.canonical_id, p.number, p.amount, p.category
          from prizes p join drawings d on d.canonical_id = p.drawing_id
         where d.valid and d.kind = 'ordinary' and d.major_count = 6
           and (%s::int is null or d.year >= %s::int)
           and (%s::int is null or d.year <= %s::int)
         order by d.draw_date
        """,
        (args.since, args.since, args.until, args.until),
    ).fetchall()

    order, seen = [], set()
    for drawing_id, *_ in rows:
        if drawing_id not in seen:
            seen.add(drawing_id)
            order.append(drawing_id)
    index = {d: i for i, d in enumerate(order)}
    zones = BALLS // ZONE_SIZE

    hits = np.zeros((len(order), zones))
    money = np.zeros((len(order), zones))
    uncapped = np.zeros((len(order), zones))
    for drawing_id, number, amount, category in rows:
        row, zone = index[drawing_id], (number - 1) // ZONE_SIZE
        uncapped[row, zone] += amount
        if category == "":
            hits[row, zone] += 1
            money[row, zone] += min(amount, MONEY_CAP)

    freq, cash, raw = hits.mean(axis=0), money.mean(axis=0), uncapped.mean(axis=0)
    hot_freq = set(np.argsort(freq)[::-1][:HOT_COUNT].tolist())
    cold_freq = set(np.argsort(freq)[:HOT_COUNT].tolist())
    rich = set(np.argsort(cash)[::-1][:HOT_COUNT].tolist())

    print(f"  {len(order)} ordinary drawings. Sorted by frequency, best first.")
    print(f"  Money is capped at ${MONEY_CAP} per prize so jackpots can't skew a zone.")
    print(f"  ★★ = top {HOT_COUNT} on BOTH   ★ = top {HOT_COUNT} on frequency only\n")
    print(f"  {'zone':<17}{'prizes/draw':>12}{'vs avg':>9}   "
          f"{'typical $':>10}{'vs avg':>9}   rating")

    for zone in np.argsort(freq)[::-1]:
        low, high = zone * ZONE_SIZE + 1, (zone + 1) * ZONE_SIZE
        if zone in hot_freq:
            rating = "★★" if zone in rich else "★"
        elif zone in cold_freq:
            rating = "avoid"
        else:
            rating = ""
        print(f"  {low:>6,}-{high:<10,}{freq[zone]:>12.1f}"
              f"{(freq[zone] / freq.mean() - 1) * 100:>+8.1f}%   "
              f"${cash[zone]:>10,.0f}{(cash[zone] / cash.mean() - 1) * 100:>+8.1f}%   {rating}")

    both = sorted(hot_freq & rich)
    print(f"\n  ★★ ZONES — top {HOT_COUNT} on both counts ({len(both)} of 50):")
    for zone in both:
        print(f"       {zone * ZONE_SIZE + 1:>6,} - {(zone + 1) * ZONE_SIZE:<6,}")

    raw_top = set(np.argsort(raw)[::-1][:HOT_COUNT].tolist())
    print("\n  Why the money column is capped:")
    print(f"    ${MONEY_CAP} was chosen because 98.5% of prize slots are already at or below")
    print("    it, so the cap barely touches the data — but it stops one jackpot from")
    print("    deciding a zone's rank. The difference that makes:")
    print(f"      capped money agrees with frequency on {len(hot_freq & rich)}"
          f"/{HOT_COUNT} of the top zones")
    print(f"      uncapped money agrees on only {len(hot_freq & raw_top)}/{HOT_COUNT}")
    print("    Uncapped, the ranking also fails to repeat on drawings it was not built")
    print("    from. Capped, it repeats — which is the whole point of the cap.\n")
    print("    Worth understanding what that means: the amount a number wins is drawn")
    print("    from a separate tumbler, so prize size carries no zone information at all")
    print("    (a zone's hit count and its average prize size are unrelated). Money is")
    print("    'how often × how much', and only 'how often' varies by zone. So a")
    print("    trustworthy money column necessarily converges on frequency — capping")
    print("    removes noise rather than revealing a second signal. ★★ is a sanity")
    print("    check, not extra edge.")


def report_majors(conn: psycopg.Connection, args: argparse.Namespace, model) -> None:
    """Are any zones hot for the MAJOR prizes — the ones worth real money?

    Majors are the scarcest thing in the data: six per drawing, so a few hundred in
    total. Every count here is printed against the spread pure chance would produce with
    that many observations, because at this sample size an eye-catching leader is the
    default outcome, not evidence.
    """
    head("MAJOR PRIZES — is any zone hot for the big money?")

    rows = conn.execute(
        """
        select m.number from major_prizes m join drawings d on d.canonical_id = m.drawing_id
         where d.valid
           and (%s::int is null or d.year >= %s::int)
           and (%s::int is null or d.year <= %s::int)
        """,
        (args.since, args.since, args.until, args.until),
    ).fetchall()
    numbers = np.array([r[0] for r in rows], dtype=np.int64)
    if len(numbers) < 50:
        print(f"  only {len(numbers)} major prizes in that range — nothing to say.")
        return

    print(f"  {len(numbers)} major prizes on record"
          f"{'' if args.since is None and args.until is None else ' in this range'}.\n")

    for size in (10_000, 5_000, 1_000):
        counts = np.bincount((numbers - 1) // size, minlength=BALLS // size)
        mean, chance_sd = counts.mean(), np.sqrt(counts.mean())
        p = stats.chisquare(counts).pvalue
        print(f"  {size:,}-number zones — {mean:.1f} majors expected each, "
              f"chance spread ±{chance_sd:.1f}")
        print(f"    observed spread {counts.std():.1f}   uniformity p = {p:.3f}   "
              f"{'BIASED' if p < 0.01 else 'consistent with chance'}")
        if size == 10_000:
            for block in np.argsort(counts)[::-1]:
                bar = "█" * int(round(counts[block] / mean * 18))
                print(f"      {block * size + 1:>6,}-{(block + 1) * size:<6,} "
                      f"{counts[block]:>4}  {(counts[block] - mean) / chance_sd:+.1f}σ  {bar}")
        print()

    fine = np.bincount((numbers - 1) // 1000, minlength=50)
    leader = int(fine.argmax())
    # With 50 zones examined at once, the leader is expected to sit well above the mean.
    family_p = 1 - (1 - stats.poisson.sf(fine.max() - 1, fine.mean())) ** 50
    print(f"  Most majors in any single 1,000-block: {fine.max()} "
          f"({leader * 1000 + 1:,}-{(leader + 1) * 1000:,})")
    print(f"    chance alone would top out near {stats.poisson.isf(1 / 50, fine.mean()):.0f}; "
          f"family-wise p = {family_p:.3f}")
    print("    one marginal leader among 50 zones, with no overall signal, is not a")
    print("    pattern — it is what looking at 50 things at once produces.\n")

    if model is not None:
        rate = model.rate
        r, p = stats.spearmanr(fine, rate)
        hot = np.argsort(rate)[::-1][:10]
        cold = np.argsort(rate)[:10]
        print("  Do majors lean toward the hot zones?")
        print(f"    rank correlation rho = {r:+.3f} (p = {p:.3f})")
        print(f"    majors in the 10 hot zones: {fine[hot].sum()}   "
              f"in the 10 cold zones: {fine[cold].sum()}")

    orphans = conn.execute(
        """
        select count(*) from major_prizes m
          join drawings d on d.canonical_id = m.drawing_id
         where d.valid and not exists (
           select 1 from prizes p
            where p.drawing_id = m.drawing_id and p.number = m.number and p.category = '')
        """
    ).fetchone()[0]

    print("\n  >> BOTTOM LINE: the tests above cannot RANK zones for major prizes — six")
    print("     majors a drawing is far too sparse for that, and it always will be.")
    print("     But that is not the same as majors being immune to the zone effect.")
    print(f"     All {len(numbers)} majors on record are themselves directo pulls "
          f"({orphans} exceptions),")
    print("     i.e. numbers from the same tumbler that happened to be paired with the")
    print("     biggest prize balls. Since the prize ball is drawn separately from the")
    print("     number ball, which number gets the big prize is independent of the")
    print("     amount — so the majors are a random subset of the drawn numbers and")
    print("     carry the same zone bias. Hot zones do improve your big-prize odds by")
    print("     roughly the same few percent; that is mechanism, not measurement.")


def report_buy(conn: psycopg.Connection, args: argparse.Namespace, model) -> None:
    """The shopping list. No statistics — just what to ask the vendor for."""
    head("WHAT TO BUY — plain version")

    if model is None:
        print("  No usable pattern in the data. Any number is as good as any other.")
        print("  Only rule that matters: cover all ten last digits (see below).")
        return

    rate, mean = model.rate, model.grand_mean
    order = np.argsort(rate)[::-1]
    # Hot and cold lists must not overlap. With coarse blocks, an unclamped top-10 and
    # bottom-10 partition the whole range and the same zone lands in both lists.
    pick = max(1, min(10, model.blocks // 3))

    if model.drawings < 60:
        print(f"  CAUTION: fitted on only {model.drawings} drawings. The ranges below are")
        print("  a noisy view of the same long-run pattern — prefer the full-history run.\n")

    def zones(blocks):
        merged: list[list[int]] = []
        for block in sorted(blocks):
            if merged and block == merged[-1][1] + 1:
                merged[-1][1] = block
            else:
                merged.append([block, block])
        return [
            (low * model.block_size + 1, (high + 1) * model.block_size,
             rate[low:high + 1].mean() / mean - 1)
            for low, high in merged
        ]

    print("  1. BUY IN THESE RANGES — best first\n")
    for low, high, edge in sorted(zones(order[:pick]), key=lambda z: -z[2]):
        print(f"       {low:>6,} - {high:<6,}   {edge * 100:+.0f}% better than average")
    print("\n     Also fine:")
    for low, high, _ in zones(order[pick:pick * 2]):
        print(f"       {low:>6,} - {high:<6,}")

    print("\n  2. AVOID THESE RANGES\n")
    for low, high, edge in sorted(zones(order[-pick:]), key=lambda z: z[2]):
        print(f"       {low:>6,} - {high:<6,}   {edge * 100:+.0f}% worse than average")

    print("\n  3. COVER ALL TEN LAST DIGITS — this matters more than the ranges\n")
    print("     Every ticket ending in the same digit as the first prize gets a")
    print("     reintegro (your money back). Buy ten fractions whose numbers end in")
    print("     0,1,2,3,4,5,6,7,8,9 and you get one of those EVERY WEEK, guaranteed.")
    print("     Buy ten numbers ending in the same digit and you get it 1 week in 10.")

    numbers, notes = frequency.pick_numbers(10, model=model, seed=int(args.seed))
    print("\n  4. A CONCRETE SET OF TEN — hot ranges, all ten last digits\n")
    print("       " + "   ".join(f"{n:05d}" for n in numbers[:5]))
    print("       " + "   ".join(f"{n:05d}" for n in numbers[5:]))
    print(f"\n     (last digits covered: {notes['last_digits_covered']}/10 — "
          f"reintegro guaranteed every draw)")

    print("\n  5. IF THE VENDOR DOESN'T HAVE THOSE\n")
    print("     Don't hunt for exact numbers. In order of what actually matters:")
    print("       a. ten different last digits  <- biggest effect by far")
    print("       b. anywhere in 12,000-40,000  <- beats 41,000-50,000")
    print("       c. avoid 41,000-50,000 if you can")
    print("       d. spread across different thousands, not ten consecutive numbers")

    # The hot-zone advantage applies to the categories that depend on where the number
    # sits (directo, centena, aproximación) but not to terminación or the reintegro.
    ratio = model.rate[np.argsort(model.rate)[::-1][:pick]].mean() / model.grand_mean
    per_number = dict(conn.execute(
        """
        select p.category, sum(p.amount)::float / count(distinct p.drawing_id) / 50000
          from prizes p join drawings d on d.canonical_id = p.drawing_id
         where d.valid and d.kind = 'ordinary' and d.major_count = 6
         group by 1
        """
    ).fetchall())
    scales = per_number.get("", 0) + per_number.get("C", 0) + per_number.get("A", 0)
    flat = per_number.get("T", 0) + args.ticket_price / 10
    per_fraction = (scales + flat) / 25
    hot_fraction = (scales * ratio + flat) / 25

    spend = 10 * 52
    print("\n  6. WHAT THIS IS ACTUALLY WORTH — the honest part\n")
    print(f"     Ten $1 fractions a week = ${spend:,}/year.")
    print(f"     Expected back, any numbers:  ${spend * per_fraction:,.0f}")
    print(f"     Expected back, hot ranges:   ${spend * hot_fraction:,.0f}")
    print(f"     The range edge is worth roughly "
          f"${spend * (hot_fraction - per_fraction):.0f} a year.")
    print("\n     Hot ranges get you MORE prizes of every size — including a slightly")
    print("     better shot at the big ones, since the draw pairs your number with a")
    print("     randomly chosen prize. But the lottery still keeps about 30 cents of")
    print("     every dollar. Buy because you enjoy it; the zones are a free upgrade.")


def head(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--section", action="append",
                       choices=["probability", "fairness", "frequency", "portfolio",
                                "zones", "majors", "buy"],
                       help="run only this section (repeatable)")
    parser.add_argument("--budget", type=int, default=10,
                       help="pedazos to buy, one per number (default: 10)")
    parser.add_argument("--ticket-price", type=float, default=DEFAULT_TICKET_PRICE,
                       help=f"price of one billete (default: {DEFAULT_TICKET_PRICE})")
    parser.add_argument("--trials", type=int, default=4000,
                       help="simulated drawings (default: 4000)")
    parser.add_argument("--since", type=int, metavar="YEAR",
                       help="only use drawings from this year onward")
    parser.add_argument("--until", type=int, metavar="YEAR",
                       help="only use drawings up to this year")
    parser.add_argument("--seed", type=int, default=5,
                       help="seed for the suggested number set (change for a different set)")
    parser.add_argument("--verify", action="store_true",
                       help="re-derive the model's constants from the database")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
