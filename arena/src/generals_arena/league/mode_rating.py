"""EGF/GoR duels and mean-preserving Codeforces multiplayer ratings.

EGF: beta=-7*log(3300-r), con=((3300-r)/200)**1.6. The human
improvement bonus is omitted for immutable policies. WAIT is rated normally;
the public 1v1 score subtracts its raw rating without changing model inputs.
CF: https://codeforces.com/blog/entry/20762 (seed, geometric rank, performance).
Arena replaces CF's deflation correction with an exactly zero-sum correction.
Repeated strategies count once; cyclic rotations share one update's weight.
"""
from collections import defaultdict
import math

from .modes import rating_mode
from .rating import scores

FFA_MEAN = 1000.0
CF_SCALE = 400.0
GO_MIN, GO_MAX = -900.0, 3299.0


def initial_rating(mode):
    return 0.0 if mode == '1v1' else FFA_MEAN


def go_beta(rating):
    return -7 * math.log(3300 - min(GO_MAX, max(GO_MIN, rating)))


def probability(mode, left, right):
    difference = (go_beta(left) - go_beta(right) if mode == '1v1'
                  else math.log(10) * (left - right) / CF_SCALE)
    return 1 / (1 + math.exp(-max(-700, min(700, difference))))


def slope(mode, rating):
    return 7 / (3300 - min(GO_MAX, max(GO_MIN, rating))) if mode == '1v1' else math.log(10) / CF_SCALE


def score_modes(ratings, result, anchor, games, *, fixed_anchor=False):
    mode = rating_mode(result)
    seats = defaultdict(list)
    for seat, key in enumerate(result['entrant_ids']):
        seats[key].append(seat)
    # Stable ordering makes duplicate/permuted seats exactly equivalent.
    keys = sorted(seats)
    places = scores(result)
    before = {key: (0.0 if fixed_anchor and mode == '1v1' and key == anchor else
                    ratings.get(key, initial_rating(mode))) for key in keys}
    weight = 1 / max(1, result.get('rating_group_size', 1))
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
                delta = 0. if fixed_anchor and key == anchor else weight * gain * (seed - actual)
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
