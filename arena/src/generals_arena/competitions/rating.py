"""One current rating rule: raw GoR duels and zero-sum Codeforces FFA.

WAIT participates normally. Only report presentation subtracts its raw GoR
rating. Distinct strategies receive one update; seat rotations share weight.
Pairwise comparisons below measure CI information, not CF update magnitude.
"""
from collections import defaultdict
import math

from generals_arena.util import integer

POLICY = 'gor-relative-codeforces-official-31.4.5-v10'
MODES = ('1v1', 'ffa')
PROVISIONAL_GAMES = 6


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError('rating mode must be 1v1 or ffa')
    return mode


def rating_mode(result):
    count = len(result['entrant_ids'])
    if not 2 <= count <= 8:
        raise ValueError('rating requires two to eight seats')
    return '1v1' if count == 2 else 'ffa'


def scores(result):
    """The environment supplies official placements, independent of Elo math."""
    return [-rank for rank in result['placements']]


def ranks(result):
    places = scores(result)
    return [1 + sum(other > place for other in places)
            + (sum(other == place for other in places) - 1) / 2 for place in places]


FFA_MEAN = 1000.0
CF_SCALE = 400.0
GO_MIN, GO_MAX = -900.0, 3299.0


def initial_rating(mode):
    validate_mode(mode)
    return 0.0 if mode == '1v1' else FFA_MEAN


def _go_beta(rating):
    return -7 * math.log(3300 - min(GO_MAX, max(GO_MIN, rating)))


def probability(mode, left, right):
    validate_mode(mode)
    difference = (_go_beta(left) - _go_beta(right) if mode == '1v1'
                  else math.log(10) * (left - right) / CF_SCALE)
    return 1 / (1 + math.exp(-max(-700, min(700, difference))))


def slope(mode, rating):
    validate_mode(mode)
    return 7 / (3300 - min(GO_MAX, max(GO_MIN, rating))) if mode == '1v1' else math.log(10) / CF_SCALE


def score(ratings, result, games):
    if result.get('rating_policy', POLICY) != POLICY:
        raise ValueError('unsupported recorded rating policy')
    mode = rating_mode(result)
    seats = defaultdict(list)
    for seat, key in enumerate(result['entrant_ids']):
        seats[key].append(seat)
    # Stable ordering makes duplicate/permuted seats exactly equivalent.
    keys = sorted(seats)
    places = scores(result)
    before = {key: ratings.get(key, initial_rating(mode)) for key in keys}
    weight = 1 / integer('rating_group_size', result.get('rating_group_size', 1), 1)
    changes = {}
    for key in keys:
        opponents = [other for other in keys if other != key]
        actual = 1 + sum(sum((places[j] > places[i]) + .5 * (places[j] == places[i])
                            for i in seats[key] for j in seats[other]) / (len(seats[key])*len(seats[other]))
                         for other in opponents)
        seed = 1 + sum(probability(mode, before[other], before[key]) for other in opponents)
        change = dict(before=before[key], after=before[key], delta=0., rank=actual,
                      expected_rank=seed, distinct_opponents=len(opponents),
                      rated_games=games.get(key, 0)+bool(opponents), weight=weight,
                      model='egf-gor' if mode == '1v1' else 'codeforces')
        if opponents:
            if mode == '1v1':
                gain = ((3300 - before[key]) / 200) ** 1.6
                delta = weight * gain * (seed - actual)
                after = min(GO_MAX, max(GO_MIN, before[key] + delta))
                change.update(after=after, delta=after-before[key], update_gain=gain)
            else:
                target = math.sqrt(actual * seed)
                low, high = min(before.values()) - 8000, max(before.values()) + 8000
                for _ in range(64):
                    mid = (low + high) / 2
                    predicted = 1 + sum(probability(mode, before[other], mid) for other in opponents)
                    if predicted > target:
                        low = mid
                    else:
                        high = mid
                performance = (low + high) / 2
                change.update(performance=performance, target_rank=target,
                              raw_delta=(performance-before[key])/2)
        changes[key] = change
    if mode == 'ffa' and len(keys) > 1:
        correction = -math.fsum(c['raw_delta'] for c in changes.values()) / len(keys)
        for change in changes.values():
            delta = weight * (change['raw_delta'] + correction)
            change.update(correction=correction, delta=delta, after=change['before']+delta)
    return changes


def comparisons(result):
    """Yield (left, right, weight, left rank score), once per distinct pair."""
    seats = defaultdict(list)
    places = scores(result)
    for seat, key in enumerate(result['entrant_ids']):
        seats[key].append(seat)
    ids = sorted(seats)
    rotations = integer('rating_group_size', result.get('rating_group_size', 1), 1)
    weight = (1.0 if len(result['entrant_ids']) == 2 else .5) / (max(1, len(ids)-1)*rotations)
    for index, left in enumerate(ids):
        for right in ids[index+1:]:
            score = sum((places[i] > places[j]) + .5*(places[i] == places[j])
                        for i in seats[left] for j in seats[right]) / (len(seats[left])*len(seats[right]))
            yield left, right, weight, score
