"""Probability model for the Lotería de Puerto Rico traditional drawing.

## How the drawing actually works

50,000 numbered balls, always. Two tumblers: one pulls a number, the other pulls a
prize. Every drawing awards a **fixed, predetermined** set of prizes — an ordinary
drawing pays out exactly $703,300 across exactly 2,629 prize slots, every single time
(verified across 119 ordinary drawings in the corpus, with zero variation). Only *which*
numbers receive them is random.

Prizes come in five kinds. The first is drawn; the other four are *derived by rule* from
the major prizes, and every rule below was verified against 132 drawings with zero
mismatches:

| Kind | How you win it | Count (ordinary) |
| --- | --- | --- |
| directo | your number was pulled from the tumbler | 1,729 |
| centena (C) | a major prize landed in your hundred-block `[100k+1 … 100k+100]` | 99 per major |
| terminación (T) | a major prize shares your last three digits | 49 per major |
| aproximación (A) | a major prize is your number ± 1 | 2 per major |
| reintegro | your last digit matches the first prize's last digit | 5,000 numbers |

The reintegro is not printed as a list of numbers — the PDF states only the winning
digit — so it is modelled from the rule rather than read from the data.

## The consequence

Because the pool is fixed and every rule is defined *relative to* uniformly drawn
numbers, every number carries an identical probability and an identical expected value.
The only exceptions are numbers 1 and 50,000, which have one neighbour instead of two and
so are very slightly *worse* (aproximación is ~0.024% of win probability, so halving it
costs about one part in 8,000).

Frequency of past wins therefore cannot improve number selection unless the physical
tumbler is biased. That is an empirical question, and :mod:`analyze` tests it.

What number selection *does* control is correlation: numbers in the same hundred-block
win centena together, numbers sharing last-three-digits win terminación together, and
numbers sharing a last digit win reintegro together. See :func:`simulate_drawings`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BALLS = 50_000
REINTEGRO_SET_SIZE = BALLS // 10  # one last digit in ten


@dataclass(frozen=True)
class DrawStructure:
    """The fixed prize-slot counts for one kind of drawing.

    These are measured from the corpus, not assumed. Every field is exactly constant
    across drawings of the same kind.
    """

    kind: str
    major_count: int
    directo: int         # numbers pulled from the tumbler, majors included
    centena: int         # 99 x major_count
    terminacion: int     # 49 x major_count
    aproximacion: int    # 2 x major_count, minus range-edge clipping
    pool: int            # total prize money awarded, in list units
    label: str = ""

    @property
    def prize_slots(self) -> int:
        return self.directo + self.centena + self.terminacion + self.aproximacion


# Measured from the corpus; `analyze.py --verify` re-derives these from the database.
ORDINARY = DrawStructure("ordinary", 6, 1729, 594, 294, 12, 703_300, "ordinary (6 majors)")
SPECIAL = DrawStructure("special", 5, 1388, 495, 245, 10, 1_406_500, "special (5 majors)")
EXTRAORDINARY = DrawStructure(
    "extraordinary", 8, 1399, 792, 392, 16, 2_813_700, "extraordinary (8 majors)"
)

STRUCTURES = {s.kind: s for s in (ORDINARY, SPECIAL, EXTRAORDINARY)}


# -- exact per-category probabilities -----------------------------------------


def _p_at_least_one_major_among(others: int, majors: int) -> float:
    """P(a given number is not a major, but at least one of `others` related numbers is).

    Exact hypergeometric counting: majors are `majors` distinct balls drawn from BALLS.
    """
    total = math.comb(BALLS, majors)
    not_me = math.comb(BALLS - 1, majors)                 # I'm not a major
    not_me_nor_them = math.comb(BALLS - 1 - others, majors)  # nor is anything related
    return (not_me - not_me_nor_them) / total


def neighbour_count(number: int) -> int:
    """How many of `number ± 1` exist. The range endpoints have only one."""
    return sum(1 for x in (number - 1, number + 1) if 1 <= x <= BALLS)


def category_probabilities(number: int, structure: DrawStructure) -> dict[str, float]:
    """Exact probability that `number` wins in each category, for one drawing."""
    majors = structure.major_count
    return {
        # The directo set is a uniform random subset of fixed size.
        "directo": structure.directo / BALLS,
        # 99 other numbers share my hundred-block.
        "centena": _p_at_least_one_major_among(99, majors),
        # 49 other numbers share my last three digits.
        "terminacion": _p_at_least_one_major_among(49, majors),
        # 1 or 2 neighbours, depending on where I sit in the range.
        "aproximacion": _p_at_least_one_major_among(neighbour_count(number), majors),
        # Independent of the number: one last digit in ten wins.
        "reintegro": REINTEGRO_SET_SIZE / BALLS,
    }


def expected_value(structure: DrawStructure, ticket_price: float) -> dict[str, float]:
    """Expected return per ticket for any number, per drawing.

    Exact, because the pool is fixed: the expected table prize is simply the pool
    divided across all 50,000 balls. `ticket_price` is what one ticket of the unit the
    prize list is denominated in costs (a full *billete*), and the reintegro is assumed
    to return that price.
    """
    table = structure.pool / BALLS
    reintegro = (REINTEGRO_SET_SIZE / BALLS) * ticket_price
    total = table + reintegro
    return {
        "expected_table_prize": table,
        "expected_reintegro": reintegro,
        "expected_total": total,
        "ticket_price": ticket_price,
        "return_ratio": total / ticket_price if ticket_price else float("nan"),
        "house_edge": 1 - (total / ticket_price) if ticket_price else float("nan"),
    }


# -- simulation ---------------------------------------------------------------


def simulate_drawings(
    structure: DrawStructure,
    portfolio: np.ndarray | None = None,
    trials: int = 10_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Simulate whole drawings under the verified rules.

    Needed because the categories overlap and, for a portfolio, are *correlated* —
    two numbers in one hundred-block win centena on the same event. Closed forms get
    unwieldy fast; faithful simulation does not.

    Returns per-trial arrays: how many distinct numbers won any table prize, and — if a
    portfolio is supplied — how many of its numbers won, split by whether the reintegro
    is counted.
    """
    rng = np.random.default_rng(seed)
    winners_per_trial = np.empty(trials, dtype=np.int32)
    portfolio_hits = np.zeros(trials, dtype=np.int32)
    portfolio_hits_no_reintegro = np.zeros(trials, dtype=np.int32)

    n_extra = structure.directo - structure.major_count
    member = np.zeros(BALLS + 1, dtype=bool)  # 1-indexed scratch buffer

    for trial in range(trials):
        member[:] = False

        # The tumbler pull: `directo` distinct balls. The majors are the ones that
        # happened to be paired with the big prize balls, so they are part of this set.
        drawn = rng.choice(BALLS, size=structure.directo, replace=False) + 1
        majors = drawn[: structure.major_count]
        member[drawn] = True

        # Derived categories. Rules verified against the corpus in analyze.py --verify.
        blocks = (majors - 1) // 100
        for block in blocks:
            member[block * 100 + 1 : block * 100 + 101] = True

        for major in majors:
            last3 = major % 1000
            tails = np.arange(0, 51) * 1000 + last3
            member[tails[(tails >= 1) & (tails <= BALLS)]] = True

        for major in majors:
            if major > 1:
                member[major - 1] = True
            if major < BALLS:
                member[major + 1] = True

        winners_per_trial[trial] = int(member[1:].sum())

        if portfolio is not None:
            table_hits = int(member[portfolio].sum())
            portfolio_hits_no_reintegro[trial] = table_hits
            # Reintegro: everyone sharing the first prize's last digit.
            digit = int(majors[0]) % 10
            union = member[portfolio] | (portfolio % 10 == digit)
            portfolio_hits[trial] = int(union.sum())

    result = {"distinct_table_winners": winners_per_trial}
    if portfolio is not None:
        result["portfolio_hits"] = portfolio_hits
        result["portfolio_hits_table_only"] = portfolio_hits_no_reintegro
    return result


# -- portfolio construction ---------------------------------------------------


def spread_portfolio(size: int, seed: int = 0) -> np.ndarray:
    """Pick `size` numbers that share as little correlation structure as possible.

    Maximises the chance of winning *something* by decorrelating the three shared-fate
    rules: give every number a different last digit where possible (reintegro), a
    different hundred-block (centena), and a different last-three-digits class
    (terminación).
    """
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    used_blocks: set[int] = set()
    used_tails: set[int] = set()

    for index in range(size):
        target_digit = index % 10  # cycle the last digit so all ten get covered
        for _ in range(10_000):
            candidate = int(rng.integers(1, BALLS + 1))
            if candidate % 10 != target_digit:
                continue
            block = (candidate - 1) // 100
            tail = candidate % 1000
            if block in used_blocks or tail in used_tails:
                continue
            chosen.append(candidate)
            used_blocks.add(block)
            used_tails.add(tail)
            break
        else:  # pragma: no cover - only if the constraints become unsatisfiable
            raise RuntimeError("could not find a decorrelated candidate")

    return np.array(sorted(chosen), dtype=np.int64)


def clustered_portfolio(size: int, start: int = 12_301) -> np.ndarray:
    """Consecutive numbers — the maximally *correlated* choice, for comparison."""
    return np.arange(start, start + size, dtype=np.int64)


def random_portfolio(size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(BALLS, size=size, replace=False) + 1).astype(np.int64)
