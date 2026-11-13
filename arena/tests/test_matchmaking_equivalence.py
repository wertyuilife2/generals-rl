"""Weight normalization must preserve both choices and random-state consumption."""
import math
import random

import pytest

from generals_arena.competitions.matchmaking import choose_opponents
from generals_arena.competitions.rating import initial_rating


class CheckedRandom(random.Random):
    def __init__(self, seed, target, count, ratings, coverage):
        super().__init__(seed)
        self.reference = random.Random(seed)
        self.target, self.count, self.ratings, self.coverage = target, count, ratings, coverage

    def choices(self, population, weights, k):
        proximity, exploration = (.8, .05) if self.count == 2 else (.6, .15)
        default = initial_rating('1v1' if self.count == 2 else 'ffa')
        nearby = [math.exp(-min(700, abs(self.ratings.get(key, default) -
                    self.ratings.get(self.target, default)) / 300)) for key in population]
        underplayed = [1 / (1 + self.coverage.get(key, 0)) for key in population]
        # Original per-element sums, including the original floating-point order.
        expected = [proximity * n / sum(nearby) + (1-proximity-exploration) * c / sum(underplayed)
                    + exploration / len(population) for n, c in zip(nearby, underplayed)]
        assert weights == expected
        selected = super().choices(population, weights=weights, k=k)
        assert selected == self.reference.choices(population, weights=expected, k=k)
        assert self.getstate() == self.reference.getstate()
        return selected


@pytest.mark.parametrize('size', [1, 3, 16, 128, 1024])
@pytest.mark.parametrize('players', [2, 4, 8])
def test_normalization_preserves_exact_weights_choices_and_rng(size, players):
    candidates = [f'p{i:04}' for i in range(size)]
    ratings = {key: i * 17 for i, key in enumerate(candidates) if i % 3}
    coverage = {key: i % 11 for i, key in enumerate(candidates)}
    for seed in range(5):
        rng = CheckedRandom(seed, candidates[0], players, ratings, coverage)
        assert len(choose_opponents(candidates[0], candidates, players, ratings, coverage, rng)) == players - 1
