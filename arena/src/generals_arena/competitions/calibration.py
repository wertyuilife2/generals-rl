"""Order-independent GoR estimation for fixed-reference evaluation.

The probability model is the existing GoR model. League games still use its
online update rule; calibration and training evaluation fit all observations
together. One virtual draw per observed pair keeps undefeated policies finite.
"""
import math

import numpy as np

from generals_arena.competitions.rating import GO_MIN, GO_MAX, comparisons
from generals_arena.util import digest


ESTIMATOR = 'gor-batch-fit-v1'
PAIR_PRIOR = 1.0


def reference_identity(entry):
    """Pin policy semantics and artifacts without local paths or device ordinals."""
    return digest({key: value if key != 'kwargs' else {k: v for k, v in value.items() if k != 'device'}
                   for key, value in entry.items()
                   if key in ('name', 'kind', 'version', 'factory', 'kwargs', 'implementation')})


def fit_ratings(entrants, results, reference, fixed=None):
    """Fit free players jointly, conditional on immutable reference ratings."""
    ids = sorted(entrants)
    fixed = dict(fixed or {reference: 0.0})
    if reference not in fixed or not set(fixed) <= set(ids):
        raise ValueError('GoR fitting requires a fixed WAIT reference in the roster')
    if any(isinstance(r, bool) or not math.isfinite(r) or not GO_MIN <= r <= GO_MAX for r in fixed.values()):
        raise ValueError('calibrated ratings must be finite raw GoR values within its bounds')
    positions = {key: index for index, key in enumerate(ids)}
    observations, graph = {}, {key: set() for key in ids}
    for result in sorted(results, key=lambda row: row['game_id']):
        for a, b, weight, score in comparisons(result):
            totals = observations.setdefault((a, b), [0., 0.])
            totals[0] += weight
            totals[1] += weight * score
            graph[a].add(b)
            graph[b].add(a)
    connected, pending = set(fixed), list(fixed)
    while pending:
        for other in graph[pending.pop()] - connected:
            connected.add(other)
            pending.append(other)
    if connected != set(ids):
        raise ValueError('every fitted policy needs a match path to a calibrated reference')
    pairs = sorted(observations)
    design = np.zeros((len(pairs), len(ids)))
    for index, (a, b) in enumerate(pairs):
        design[index, positions[a]], design[index, positions[b]] = 1., -1.
    totals = np.array([observations[pair] for pair in pairs])
    games, wins = totals[:, 0] + PAIR_PRIOR, totals[:, 1] + PAIR_PRIOR / 2
    free = [positions[key] for key in ids if key not in fixed]
    x = np.zeros(len(ids))
    for key, raw in fixed.items():
        x[positions[key]] = -7 * math.log((3300 - raw) / 3300)
    lower, upper = (-7 * math.log((3300 - raw) / 3300) for raw in (GO_MIN, GO_MAX))
    a = design[:, free]

    def loss(values):
        logits = design @ values
        return float(np.sum(games * np.logaddexp(0, logits) - wins * logits))

    for iteration in range(100):
        p = 1 / (1 + np.exp(-(design @ x)))
        gradient = a.T @ (games * p - wins)
        active = ((x[free] > lower + 1e-10) | (gradient < 0)) & ((x[free] < upper - 1e-10) | (gradient > 0))
        if not np.any(active) or np.max(np.abs(gradient[active])) < 1e-9:
            break
        hessian = (a.T * (games * p * (1-p))) @ a
        step = np.zeros(len(free))
        step[active] = np.linalg.solve(hessian[np.ix_(active, active)], gradient[active])
        old = loss(x)
        scale = 1.
        while scale > 1e-10:
            proposal = x.copy()
            proposal[free] = np.clip(x[free] - scale * step, lower, upper)
            if loss(proposal) <= old + 1e-12:
                break
            scale *= .5
        if np.max(np.abs(proposal - x)) < 1e-10:
            x = proposal
            break
        x = proposal
    else:
        raise RuntimeError('GoR batch fit did not converge')
    raw = {key: float(3300 * (-np.expm1(x[index] / -7))) for index, key in enumerate(ids)}
    raw.update(fixed)  # Preserve the exact calibrated bytes, including WAIT.
    p = 1 / (1 + np.exp(-(design @ x)))
    covariance = np.linalg.inv((a.T * (games * p * (1-p))) @ a) if free else np.empty((0, 0))
    intervals = {key: None for key in ids}
    for column, index in enumerate(free):
        key = ids[index]
        if x[index] <= lower + 1e-8 or x[index] >= upper - 1e-8:
            continue
        error = 1.959963984540054 * math.sqrt(max(0, covariance[column, column])) * (3300-raw[key])/7
        elo = raw[key] - raw[reference]
        intervals[key] = dict(level=.95, lower=elo-error, upper=elo+error, half_width=error,
                              method='gor-fit-fisher-conditional-v1')
    return raw, intervals, {'estimator': ESTIMATOR, 'pair_prior': PAIR_PRIOR,
                           'fixed_ratings': fixed, 'iterations': iteration + 1,
                           'ci_scope': 'conditional_on_fixed_reference_ratings'}
