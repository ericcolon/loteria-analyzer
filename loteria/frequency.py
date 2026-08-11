"""The frequency model: does past history change a number's chance of winning?

Per-*number* frequency is hopeless — each ball has only been drawn ~4.5 times, and to
stand out against 50,000 simultaneous comparisons a ball would need to be about 5x more
likely than fair. Nothing in the corpus comes close, and nothing could.

Per-*region* frequency is a different story. Pooling numbers into thousand-blocks gives
~4,500 observations per block, and there the corpus shows a stable, reproducible
deviation from uniform: some blocks yield noticeably more directo prizes than others,
the ranking learned from the first half of the corpus holds up in the second half, and
the gap sits far outside a permutation null.

Only the `directo` category is modelled. Centena, terminación and aproximación are
derived by rule from where the major prizes landed, so their spatial distribution just
echoes the majors' — and the majors test as uniform.

Estimates are shrunk toward the grand mean (empirical Bayes), because a raw per-block
average contains real sampling noise and taking it at face value would overstate the
spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

BALLS = 50_000

# Probability that any given number wins something in an ordinary drawing, measured from
# the corpus: distinct table winners plus the reintegro, less their overlap.
BASE_ANY_PRIZE = 0.14568


@dataclass
class BlockModel:
    """Per-region directo rates, fitted from drawing history."""

    block_size: int
    rate: np.ndarray          # shrunk expected directo hits per drawing per block
    raw_rate: np.ndarray      # unshrunk observed means
    drawings: int
    grand_mean: float
    shrinkage: float

    @property
    def blocks(self) -> int:
        return len(self.rate)

    def block_of(self, number: int) -> int:
        return (number - 1) // self.block_size

    def directo_probability(self, number: int) -> float:
        """P(this number is pulled from the tumbler) per drawing."""
        return float(self.rate[self.block_of(number)] / self.block_size)

    def any_prize_probability(self, number: int) -> float:
        """P(this number wins ANY prize) per drawing, adjusted for its region.

        The adjustment is the difference between this region's directo rate and the
        uniform rate. Everything else — reintegro, centena, terminación — is uniform
        across regions, so it contributes no differential.
        """
        uniform = self.grand_mean / self.block_size
        return BASE_ANY_PRIZE + (self.directo_probability(number) - uniform)

    def ranked_blocks(self) -> np.ndarray:
        """Block indices, best first."""
        return np.argsort(self.rate)[::-1]

    def block_range(self, block: int) -> tuple[int, int]:
        return block * self.block_size + 1, (block + 1) * self.block_size


def fit(counts: np.ndarray, block_size: int) -> BlockModel:
    """Fit a block model from a (drawings x blocks) matrix of directo hit counts.

    Shrinkage is empirical Bayes: the observed spread between block means mixes real
    signal with sampling noise, so the signal variance is estimated by subtracting the
    known sampling variance, and each block is pulled toward the grand mean accordingly.
    """
    drawings = counts.shape[0]
    raw = counts.mean(axis=0)
    grand = float(raw.mean())

    # Sampling variance of a block mean, from the within-block variance across drawings.
    within = counts.var(axis=0, ddof=1).mean() / drawings
    observed = float(raw.var(ddof=1))
    signal = max(observed - within, 0.0)
    shrinkage = signal / (signal + within) if (signal + within) > 0 else 0.0

    shrunk = grand + shrinkage * (raw - grand)
    return BlockModel(
        block_size=block_size,
        rate=shrunk,
        raw_rate=raw,
        drawings=drawings,
        grand_mean=grand,
        shrinkage=float(shrinkage),
    )


@dataclass
class Validation:
    """Out-of-sample evidence that a fitted model carries real information."""

    block_size: int
    split_correlation: float
    split_p: float
    hot_rate_late: float
    cold_rate_late: float
    gap: float
    gap_t: float
    gap_p: float
    permutation_sd: float
    permutation_z: float

    @property
    def credible(self) -> bool:
        return self.split_p < 0.01 and self.gap_p < 0.01 and self.permutation_z > 3

    @property
    def hot_rate_per_number(self) -> float:
        """Out-of-sample directo probability for a number in a hot region.

        The criterion for choosing granularity. Raw gaps are not comparable across
        block sizes — a coarser block always shows a bigger absolute gap simply because
        it holds more numbers — whereas this is the quantity a buyer actually gets.
        """
        return self.hot_rate_late / self.block_size


def validate(counts: np.ndarray, block_size: int, top_k: int = 10,
             permutations: int = 1000, seed: int = 0) -> Validation:
    """Split the corpus in half: does the early ranking predict the late half?

    This is the only test that matters for acting on the model. An in-sample fit will
    always find "hot" regions; the question is whether they stay hot.
    """
    # The hot and cold sets must be disjoint: with few blocks, an unclamped top_k would
    # select every block into both and report a meaningless gap of zero.
    top_k = max(1, min(top_k, counts.shape[1] // 3))

    half = counts.shape[0] // 2
    early, late = counts[:half], counts[half:]
    early_mean, late_mean = early.mean(axis=0), late.mean(axis=0)

    r, p = stats.pearsonr(early_mean, late_mean)
    hot = np.argsort(early_mean)[::-1][:top_k]
    cold = np.argsort(early_mean)[:top_k]
    gap = float(late_mean[hot].mean() - late_mean[cold].mean())
    t, gap_p = stats.ttest_rel(late[:, hot].sum(axis=1), late[:, cold].sum(axis=1))

    # Permutation null: shuffle each drawing's counts across blocks, destroying any
    # spatial structure while preserving each drawing's total.
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for index in range(permutations):
        shuffled = counts.copy()
        for row in shuffled:
            rng.shuffle(row)
        se = shuffled[:half].mean(axis=0)
        sl = shuffled[half:].mean(axis=0)
        h = np.argsort(se)[::-1][:top_k]
        c = np.argsort(se)[:top_k]
        null[index] = sl[h].mean() - sl[c].mean()

    sd = float(null.std()) or float("nan")
    return Validation(
        block_size=block_size,
        split_correlation=float(r),
        split_p=float(p),
        hot_rate_late=float(late_mean[hot].mean()),
        cold_rate_late=float(late_mean[cold].mean()),
        gap=gap,
        gap_t=float(t),
        gap_p=float(gap_p),
        permutation_sd=sd,
        permutation_z=float((gap - null.mean()) / sd),
    )


def counts_matrix(draws_by_drawing: list[np.ndarray], block_size: int) -> np.ndarray:
    """Build the (drawings x blocks) directo count matrix."""
    blocks = BALLS // block_size
    matrix = np.zeros((len(draws_by_drawing), blocks))
    for index, numbers in enumerate(draws_by_drawing):
        matrix[index] = np.bincount((numbers - 1) // block_size, minlength=blocks)
    return matrix


# -- selection ----------------------------------------------------------------


def pick_numbers(
    budget: int, model: BlockModel | None = None, seed: int = 0
) -> tuple[np.ndarray, dict[str, str]]:
    """Choose `budget` numbers to maximise the chance of winning something.

    Two levers, in order of how much they matter:

    1. **Cover all ten last digits.** The reintegro goes to every number sharing the
       first prize's last digit — 10% of all balls, and by far the largest single prize
       category. Hold all ten digits and at least one of your numbers wins it every
       single drawing, guaranteed. This dwarfs everything else.
    2. **Prefer high-rate regions, and use distinct ones.** Worth about +0.14
       percentage points, and spreading across blocks and terminación classes avoids
       numbers that can only win together.
    """
    rng = np.random.default_rng(seed)
    # Block arithmetic has to follow the model's own granularity — the fitted block size
    # is not always 1,000, and assuming it silently walks the wrong part of the range.
    size = model.block_size if model is not None else 1000
    if model is not None:
        preferred = [int(b) for b in model.ranked_blocks()]
    else:
        # No model: visit blocks in random order. Walking 0,1,2,… would quietly steer
        # every pick into the low numbers.
        preferred = list(range(BALLS // size))
        rng.shuffle(preferred)

    chosen: list[int] = []
    used_blocks: set[int] = set()
    used_tails: set[int] = set()

    for position in range(budget):
        digit = position % 10  # cycle so the ten last digits get covered first
        candidate = _find_candidate(rng, digit, preferred, size, used_blocks, used_tails)
        chosen.append(candidate)
        used_blocks.add((candidate - 1) // size)
        used_tails.add(candidate % 1000)

    notes = {
        "last_digits_covered": str(len({n % 10 for n in chosen})),
        "reintegro_guaranteed": "yes" if len({n % 10 for n in chosen}) == 10 else "no",
        "distinct_blocks": str(len(used_blocks)),
    }
    return np.array(sorted(chosen), dtype=np.int64), notes


def _find_candidate(
    rng: np.random.Generator,
    digit: int,
    preferred_blocks: list[int],
    block_size: int,
    used_blocks: set[int],
    used_tails: set[int],
) -> int:
    """A number ending in `digit`, in the best still-unused region available."""
    for block in preferred_blocks:
        if block in used_blocks:
            continue
        low = block * block_size + 1
        options = [n for n in range(low, low + block_size) if n % 10 == digit
                   and n % 1000 not in used_tails]
        if options:
            return int(rng.choice(options))
    # Constraints exhausted; fall back to any number with the right last digit.
    while True:
        candidate = int(rng.integers(1, BALLS + 1))
        if candidate % 10 == digit:
            return candidate
