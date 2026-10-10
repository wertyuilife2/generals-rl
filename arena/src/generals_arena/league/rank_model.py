"""Rank evidence, online steps, and outcome-constrained historical calibration."""
from collections import defaultdict
import math

import numpy as np

from .rating import LOGISTIC_SLOPE, expected, scores
from ..util import canonical, integer

PAIR_PRIOR = .5  # Half a win and half a loss, once per observed policy pair.
INITIAL_INFORMATION = .25
CURVATURE_FLOOR = .04


def rotation_key(result, job_id=None):
    if 'map_seed' not in result or result.get('human'):
        return result['game_id']
    return (job_id, result['map_seed'], canonical(result['map']),
            tuple(sorted(result['entrant_ids'])))


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


def update(ratings, participants, anchor, evidence, current):
    """Only this game's residual changes means; history sets positive gains.

    Information is outcome-independent for this game. Thus the update has zero
    conditional expectation under the advertised probability model, and its
    sign cannot be reversed by historical outcomes or other opponents' changes.
    The cumulative curvature floor bounds late upsets even after a perfect run.
    """
    before = {key: 0.0 if key == anchor else ratings.get(key, 0.0)
              for key in participants}
    residual = defaultdict(float)
    information = {key: INITIAL_INFORMATION for key in participants}
    for left, right, weight, outcome in current:
        probability = expected(before[left], before[right])
        residual[left] += weight*(outcome-probability)
        residual[right] -= weight*(outcome-probability)
    for (left, right), (weight, _) in sorted(evidence.items()):
        if left not in information and right not in information:
            continue
        probability = expected(0.0 if left == anchor else ratings.get(left, 0.0),
                               0.0 if right == anchor else ratings.get(right, 0.0))
        value = weight*max(CURVATURE_FLOOR, probability*(1-probability))
        for key in (left, right):
            if key in information:
                information[key] += value
    gain = {key: 0.0 if key == anchor else 1/(LOGISTIC_SLOPE*information[key])
            for key in participants}
    after = {key: before[key]+gain[key]*residual[key] for key in participants}
    return after, gain


def calibrate(ratings, participants, anchor, evidence, proposed, limits):
    """Fit historical comparisons without reversing this game's online step.

    Inactive means stay fixed. Each active mean is bounded on the side of its
    proposed online update: a winner can gain more, but cannot lose or have its
    positive step cancelled. A per-game trust region bounds the total change.
    This lets sparse reference evidence propagate on
    later appearances instead of freezing an incorrectly identified origin.

    Cyclic coordinate Newton steps minimize the convex, regularized pair loss.
    The half-win/half-loss per observed pair keeps perfect records finite. It
    is regularization, not additional data for uncertainty or effective games.
    """
    active = sorted(set(participants) - {anchor})
    theta = {key: value*LOGISTIC_SLOPE for key, value in ratings.items()}
    theta[anchor] = 0.0
    bounds, neighbors = {}, {key: [] for key in active}
    for key in active:
        before, after = ratings.get(key, 0.0), proposed[key]
        value = after*LOGISTIC_SLOPE
        theta[key] = value
        bounds[key] = ((value, (before+limits[key])*LOGISTIC_SLOPE) if after > before else
                       ((before-limits[key])*LOGISTIC_SLOPE, value) if after < before else (value, value))
    for (left, right), (total, wins) in sorted(evidence.items()):
        if left in neighbors:
            neighbors[left].append((right, total+2*PAIR_PRIOR, wins+PAIR_PRIOR))
        if right in neighbors:
            neighbors[right].append((left, total+2*PAIR_PRIOR, total-wins+PAIR_PRIOR))

    def loss(value, opponents):
        total = 0.0
        for key, weight, wins in opponents:
            difference = value-theta.get(key, 0.0)
            total += weight*(max(0.0, difference)+math.log1p(math.exp(-abs(difference)))) - wins*difference
        return total

    for _ in range(100):
        largest = 0.0
        for key in active:
            lower, upper = bounds[key]
            if lower == upper:
                continue
            value, gradient, information = theta[key], 0.0, 0.0
            for other, weight, wins in neighbors[key]:
                difference = value-theta.get(other, 0.0)
                p = 1/(1+math.exp(-max(-700, min(700, difference))))
                gradient += weight*p-wins
                information += weight*p*(1-p)
            if information < 1e-12:
                continue
            step = max(-1.0, min(1.0, -gradient/information))
            candidate = max(lower, min(upper, value+step))
            original_loss = loss(value, neighbors[key])
            for _ in range(30):
                if loss(candidate, neighbors[key]) <= original_loss+1e-12:
                    break
                candidate = (value+candidate)/2
            theta[key] = candidate
            largest = max(largest, abs(candidate-value))
        if largest < 1e-9:
            break
    fitted = {}
    for key in participants:
        before, online = ratings.get(key, 0.0), proposed[key]
        if key == anchor:
            fitted[key] = 0.0
        elif online == before:
            fitted[key] = before
        elif online > before:
            fitted[key] = max(online, theta[key]/LOGISTIC_SLOPE)
        else:
            fitted[key] = min(online, theta[key]/LOGISTIC_SLOPE)
    return fitted


def fit(ratings, participants, anchor, evidence):
    """Legacy v5 replay: minimize cumulative loss with inactive ratings fixed.

    Newton steps have a line search. Least squares preserves the common origin
    of components that have not yet met the anchor (their CI remains unknown).
    Numerical step damping is solely for the optimizer, not a rating K-factor.
    """
    active = sorted(set(participants) - {anchor})
    indices = {key: i for i, key in enumerate(active)}
    relevant = [(left, right, total+2*PAIR_PRIOR, wins+PAIR_PRIOR)
                for (left, right), (total, wins) in sorted(evidence.items())
                if left in indices or right in indices]
    if not active or not relevant:
        return {key: 0.0 if key == anchor else ratings.get(key, 0.0) for key in participants}
    theta = np.array([ratings.get(key, 0.0)*LOGISTIC_SLOPE for key in active])

    def evaluate(values, derivatives=False):
        loss = 0.0
        gradient, hessian = np.zeros(len(active)), np.zeros((len(active), len(active)))
        for left, right, total, wins in relevant:
            i, j = indices.get(left), indices.get(right)
            a = values[i] if i is not None else (0 if left == anchor else ratings.get(left, 0))*LOGISTIC_SLOPE
            b = values[j] if j is not None else (0 if right == anchor else ratings.get(right, 0))*LOGISTIC_SLOPE
            difference = a-b
            loss += total*np.logaddexp(0, difference) - wins*difference
            if not derivatives:
                continue
            probability = 1/(1+math.exp(-max(-700, min(700, difference))))
            residual, information = total*probability-wins, total*probability*(1-probability)
            if i is not None:
                gradient[i] += residual
                hessian[i, i] += information
            if j is not None:
                gradient[j] -= residual
                hessian[j, j] += information
            if i is not None and j is not None:
                hessian[i, j] -= information
                hessian[j, i] -= information
        return loss, gradient, hessian

    for _ in range(40):
        loss, gradient, hessian = evaluate(theta, True)
        if np.max(np.abs(gradient)) < 1e-8:
            break
        step = np.linalg.lstsq(hessian, gradient, rcond=1e-12)[0]
        step /= max(1, float(np.max(np.abs(step))))
        scale = 1.0
        while scale >= 1/4096:
            candidate = theta-scale*step
            if evaluate(candidate)[0] <= loss:
                break
            scale /= 2
        if scale < 1/4096:
            break
        theta = candidate
        if np.max(np.abs(scale*step)) < 1e-8:
            break
    fitted = {key: float(theta[index]/LOGISTIC_SLOPE) for key, index in indices.items()}
    if anchor in participants:
        fitted[anchor] = 0.0
    return fitted
