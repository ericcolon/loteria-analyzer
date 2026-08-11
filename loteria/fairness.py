"""Test whether the physical drawing deviates from uniform.

This is the only route by which past frequency could legitimately inform number
selection. The prize rules are all defined *relative to* uniformly drawn numbers and the
prize pool is a fixed constant, so if the tumbler is fair then every number has an
identical probability and an identical expected value, and frequency is pure noise.
If the tumbler is *not* fair, that is exploitable — so it is worth testing properly.

The tests are ordered by statistical power. Per-number tests have almost none (each
number has only been drawn ~4-5 times), while marginal tests pool tens of thousands of
observations into ten or fifty buckets and can detect much smaller distortions. A
power analysis accompanies the results, because "no significant deviation" is only
meaningful alongside "here is what we could have detected".

Only the `directo` category is used. The other categories are derived by rule from the
majors, so counting them would multiply the same few random events and badly overstate
the sample size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

BALLS = 50_000


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    detail: str

    @property
    def verdict(self) -> str:
        if self.p_value < 0.001:
            return "STRONG deviation"
        if self.p_value < 0.01:
            return "deviation"
        if self.p_value < 0.05:
            return "weak deviation"
        return "consistent with fair"


def counts_by_number(draws: np.ndarray) -> np.ndarray:
    """Times each ball 1..BALLS was drawn. Index 0 is unused."""
    return np.bincount(draws, minlength=BALLS + 1)


def test_uniform_all_numbers(draws: np.ndarray) -> TestResult:
    """Chi-square across all 50,000 balls.

    Low power for any individual ball, but sensitive to a broad distortion affecting
    many balls at once.
    """
    observed = counts_by_number(draws)[1:]
    statistic, p_value = stats.chisquare(observed)
    return TestResult(
        "uniformity across all 50,000 balls",
        statistic,
        p_value,
        f"{len(draws):,} draws, mean {observed.mean():.2f} per ball, "
        f"df={BALLS - 1}",
    )


def test_digit_positions(draws: np.ndarray) -> list[TestResult]:
    """Per-digit-position uniformity.

    High power: tens of thousands of observations across ten buckets. A tumbler with
    physically distinguishable balls (ink weight, wear) would most plausibly show up
    here. The leading digit is skipped because the range 1..50000 does not populate it
    uniformly by construction.
    """
    results = []
    for position in range(1, 5):
        divisor = 10 ** (4 - position)
        digits = (draws // divisor) % 10
        observed = np.bincount(digits, minlength=10)
        statistic, p_value = stats.chisquare(observed)
        spread = f"min {observed.min():,} max {observed.max():,}"
        results.append(
            TestResult(
                f"digit uniformity at position {position + 1} of 5",
                statistic,
                p_value,
                f"{spread}, expected {observed.mean():,.0f} each",
            )
        )
    return results


def test_range_bins(draws: np.ndarray, bins: int = 50) -> TestResult:
    """Uniformity across equal-width slices of the ball range.

    Catches a tumbler that favours part of its contents — e.g. if balls are loaded in
    numeric order and imperfectly mixed.
    """
    edges = np.linspace(0, BALLS, bins + 1)
    observed, _ = np.histogram(draws, bins=edges)
    statistic, p_value = stats.chisquare(observed)
    return TestResult(
        f"uniformity across {bins} equal slices of 1..50,000",
        statistic,
        p_value,
        f"min {observed.min():,} max {observed.max():,}, "
        f"expected {observed.mean():,.0f} each",
    )


def test_last_digit_of_first_prize(digits: np.ndarray) -> TestResult:
    """The reintegro digit. Small sample, but it decides a 10% prize every drawing."""
    observed = np.bincount(digits, minlength=10)
    statistic, p_value = stats.chisquare(observed)
    return TestResult(
        "uniformity of the first prize's last digit (reintegro)",
        statistic,
        p_value,
        f"{len(digits)} drawings, counts {observed.tolist()}",
    )


def test_temporal_drift(draws_early: np.ndarray, draws_late: np.ndarray) -> TestResult:
    """Do the first and second halves of the corpus agree?

    A tumbler that develops a bias over time — ball wear, a replaced set — would show a
    difference here even if each half looks uniform on its own.
    """
    bins = 50
    edges = np.linspace(0, BALLS, bins + 1)
    early, _ = np.histogram(draws_early, bins=edges)
    late, _ = np.histogram(draws_late, bins=edges)
    table = np.vstack([early, late])
    statistic, p_value, _, _ = stats.chi2_contingency(table)
    return TestResult(
        "early vs late half of the corpus (drift)",
        statistic,
        p_value,
        f"{len(draws_early):,} early draws vs {len(draws_late):,} late",
    )


def test_hottest_numbers(draws: np.ndarray) -> TestResult:
    """Is the most-drawn ball drawn more often than chance allows?

    This is the test that speaks directly to picking "hot" numbers. With 50,000 balls
    examined at once, the highest count among them is expected to be well above the mean
    purely by chance, so the maximum must be judged against the distribution of the
    *maximum*, not against the mean. Judging it against the mean is what makes random
    noise look like a pattern.
    """
    observed = counts_by_number(draws)[1:]
    hottest = int(observed.max())
    rate = len(draws) / BALLS

    # P(at least one of 50,000 Poisson(rate) balls reaches `hottest`)
    tail = stats.poisson.sf(hottest - 1, rate)
    p_family = 1 - (1 - tail) ** BALLS

    n_at_max = int((observed == hottest).sum())
    return TestResult(
        "hottest ball vs chance (family-wise)",
        float(hottest),
        float(p_family),
        f"most-drawn ball appeared {hottest}x vs mean {rate:.2f}; "
        f"{n_at_max} ball(s) tied at that count. p is the chance that *some* ball "
        f"among 50,000 reaches {hottest}x if the tumbler is fair",
    )


def detectable_effect(draws: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """How large a per-ball bias could this much data actually detect?

    Reports the multiplier on a single ball's draw rate that would be needed for it to
    stand out at `alpha` after correcting for testing all 50,000 balls. Without this,
    "no significant deviation" is an empty statement.
    """
    rate = len(draws) / BALLS
    corrected = alpha / BALLS
    # Smallest count that would clear the corrected threshold.
    threshold = int(stats.poisson.isf(corrected, rate)) + 1
    # A ball needs a true rate high enough that reaching `threshold` is likely (80%).
    needed_rate = threshold
    for candidate in np.arange(rate, rate * 20, rate / 50):
        if stats.poisson.sf(threshold - 1, candidate) >= 0.80:
            needed_rate = candidate
            break
    return {
        "expected_draws_per_ball": rate,
        "count_needed_for_significance": float(threshold),
        "true_rate_needed_for_80pct_power": float(needed_rate),
        "multiplier": float(needed_rate / rate),
    }
