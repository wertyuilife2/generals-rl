"""Random is fixed at zero; a 500-point gap means 90% expected pair score."""
import math

SCALE = 500.0 / math.log10(9.0)


def expected(left, right):
    return 1 / (1 + 10 ** max(-300, min(300, (right - left) / SCALE)))


def scores(result):
    """Survivors outrank eliminated players; same-tick eliminations tie."""
    return [float("inf") if tick is None else tick for tick in result["elimination_ticks"]]


def update(ratings, result, anchor, k=32):
    ids = result["entrant_ids"]
    places = scores(result)
    before = [ratings.get(key, 0.0) for key in ids]
    after = dict(ratings)
    for i, key in enumerate(ids):
        delta = sum((1 if places[i] > places[j] else .5 if places[i] == places[j] else 0)
                    - expected(before[i], before[j]) for j in range(len(ids)) if i != j)
        after[key] = 0.0 if key == anchor else before[i] + k * delta / (len(ids) - 1)
    return after
