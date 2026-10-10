"""Current evaluation weights; each recorded plan pins its scoring policy."""
from collections import defaultdict
import random

from .rating import LOGISTIC_SLOPE, expected, learning_rate, rate, scores
from .rank_model import calibrate, comparisons, fit, update
from ..util import digest

LEGACY_POLICY = 'rank-balanced-opponents-v4'
CUMULATIVE_POLICY = 'rank-cumulative-bt-v5'
ONLINE_POLICY = 'rank-online-information-v6'
POLICY = 'rank-calibrated-outcome-v7'
EVIDENCE_POLICIES = (CUMULATIVE_POLICY, ONLINE_POLICY, POLICY)
DUEL_SHARE = .5
DUEL_WEIGHT = 2.0


def player_count(config, job):
    enabled = config.player_counts
    if 2 not in enabled or len(enabled) == 1:
        weights = [1 / count for count in enabled]
    else:
        # The target is a share of games, not rotation groups. A P-player
        # group produces P games, so divide each desired share by P.
        weights = [(DUEL_SHARE if count == 2 else (1-DUEL_SHARE)/(len(enabled)-1)) / count
                   for count in enabled]
    rng = random.Random(int(digest([config.seed, job['id'], job['done'], POLICY])[:16], 16))
    return rng.choices(enabled, weights=weights)[0]


def score(ratings, result, anchor, k, games, evidence=None):
    policy = result.get('rating_policy')
    if policy is None:
        return rate(ratings, result, anchor, k, games)
    if policy not in (*EVIDENCE_POLICIES, LEGACY_POLICY):
        raise ValueError(f'unknown recorded rating policy: {policy}')
    ids, places = result['entrant_ids'], scores(result)
    seats = defaultdict(list)
    for seat, key in enumerate(ids):
        seats[key].append(seat)
    changes = {}
    weight = DUEL_WEIGHT if len(ids) == 2 else 1.0
    if policy in EVIDENCE_POLICIES:
        evidence = {} if evidence is None else evidence
        current = list(comparisons(result))
        for left, right, amount, outcome in current:
            total, wins = evidence.get((left, right), (0., 0.))
            evidence[left, right] = (total+amount, wins+amount*outcome)
        if policy in (ONLINE_POLICY, POLICY):
            fitted, gains = update(ratings, seats, anchor, evidence, current)
            if policy == POLICY:
                online = fitted
                # At most two full-surprise online steps, including the base
                # step. Calibration cannot create an unbounded late refit.
                limits = {key: 2*gains[key]*sum(w for left,right,w,_ in current if key in (left,right))
                          for key in seats}
                fitted = calibrate(ratings, seats, anchor, evidence, online, limits)
        else:
            fitted = fit(ratings, seats, anchor, evidence) if len(seats) > 1 else ratings
    for key, own in seats.items():
        opponents = [other for other in seats if other != key]
        before = 0.0 if key == anchor else ratings.get(key, 0.0)
        actual = 1.0 + sum(
            sum((places[j] > places[i]) + .5*(places[j] == places[i])
                for i in own for j in seats[other]) / (len(own)*len(seats[other]))
            for other in opponents)
        probabilities = [expected(0.0 if other == anchor else ratings.get(other, 0.0), before)
                         for other in opponents]
        seed = 1.0 + sum(probabilities)
        count = games.get(key, 0)
        if policy in EVIDENCE_POLICIES:
            after = 0.0 if key == anchor else fitted.get(key, before)
            changes[key] = dict(before=before, after=after, delta=after-before, rank=actual,
                                expected_rank=seed, rated_games=count+bool(opponents),
                                weight=weight, distinct_opponents=len(opponents),
                                effective_games=sum(total for pair, (total, _) in evidence.items() if key in pair))
            if policy in (ONLINE_POLICY, POLICY):
                changes[key]['update_gain'] = gains[key]
            if policy == POLICY:
                changes[key]['online_delta'] = online[key]-before
                changes[key]['calibration_delta'] = after-online[key]
                changes[key]['update_limit'] = limits[key]
            continue
        slope = LOGISTIC_SLOPE * max(sum(p*(1-p) for p in probabilities), .04*len(opponents))
        performance = before + (seed-actual)/slope if opponents else before
        gain = learning_rate(count, k*weight) if opponents and key != anchor else 0.0
        after = before + gain*(performance-before)
        changes[key] = dict(before=before, after=after, delta=after-before, rank=actual,
                            expected_rank=seed, performance=performance, learning_rate=gain,
                            rated_games=count+bool(opponents), weight=weight,
                            distinct_opponents=len(opponents))
    return changes
