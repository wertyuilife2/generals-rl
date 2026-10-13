"""Rank-based, anchored ratings with fast provisional performance updates.

Like Codeforces, compare actual and expected rank. Use a damped Newton
performance estimate (rather than CF's geometric rank mean) so the expected
update remains zero under the advertised pairwise probability model.
"""
import math

SCALE = 500.0 / math.log10(9.0)
LOGISTIC_SLOPE = math.log(9.0) / 500.0
PROVISIONAL_GAMES = 6


def expected(left, right):
    return 1 / (1 + 10 ** max(-300, min(300, (right - left) / SCALE)))


def scores(result):
    """Survivors outrank eliminated players; same-tick eliminations tie."""
    return [float("inf") if tick is None else tick for tick in result["elimination_ticks"]]


def ranks(result):
    places = scores(result)
    return [1 + sum(other > place for other in places)
            + (sum(other == place for other in places) - 1) / 2 for place in places]


def learning_rate(games, k=32):
    # Six decisive provisional games, then gradually reduce rating volatility.
    provisional = (.50, .40, .32, .25, .20, .16)
    rate = provisional[games] if games < PROVISIONAL_GAMES else max(.025, .16 / math.sqrt(1 + (games - 5) / 12))
    return min(1.0, rate * k / 32)


def rate(ratings, result, anchor, k=32, games=None):
    """One update per algorithm, averaging its independent seat placements.

    Same-algorithm comparisons carry no information and never age provisional
    ratings. All deltas use the pre-match ratings, irrespective of seat order.
    The curvature floor limits sensitivity to extreme upsets symmetrically;
    it does not clip losses differently from wins.
    """
    games = games or {}
    ids, places = result['entrant_ids'], scores(result)
    changes = {}
    for key in dict.fromkeys(ids):
        own = [i for i, entrant in enumerate(ids) if entrant == key]
        opponents = [j for j, entrant in enumerate(ids) if entrant != key]
        before = 0.0 if key == anchor else ratings.get(key, 0.0)
        count = games.get(key, 0)
        actual = 1.0 + sum((places[j] > places[i]) + .5 * (places[j] == places[i])
                           for i in own for j in opponents) / len(own)
        probabilities = [expected(ratings.get(ids[j], 0.0) if ids[j] != anchor else 0.0, before) for j in opponents]
        seed = 1.0 + sum(probabilities)
        # -d(expected rank)/d(rating); floor keeps inverse slope bounded.
        slope = LOGISTIC_SLOPE * max(sum(p * (1-p) for p in probabilities), .04 * len(opponents))
        performance = before + (seed - actual) / slope if opponents else before
        gain = learning_rate(count, k) if opponents and key != anchor else 0.0
        after = before + gain * (performance - before)
        changes[key] = dict(before=before, after=after, delta=after-before,
                            rank=actual, expected_rank=seed, performance=performance,
                            learning_rate=gain, rated_games=count + bool(opponents))
    return changes


def update(ratings, result, anchor, k=32, games=None):
    return {**ratings, **{key: value['after'] for key, value in rate(ratings, result, anchor, k, games).items()}}
