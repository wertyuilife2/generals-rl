"""Approximate rank-model uncertainty, counting independent maps, not seats.

Fisher information is evaluated at current online estimates. Update damping
and curvature floors are excluded: they cannot manufacture measured precision.
These are local model-based estimates, not exact-coverage intervals.
"""
from collections import Counter
from itertools import combinations
import math

import numpy as np

from .rating import expected, LOGISTIC_SLOPE, PROVISIONAL_GAMES
from .rating_policy import EVIDENCE_POLICIES
from .rank_model import comparisons
from .mode_rating import probability, slope
from ..util import canonical


class RatingUncertainty:
    def __init__(self):
        self.seen = set()
        self.pairs = Counter()
        self.blocks = Counter()
        self.games = set()

    def add(self, result, job_id=None):
        ids = tuple(sorted(set(result['entrant_ids'])))
        if len(ids) < 2:
            return  # Self-play contains no relative strength information.
        if 'map_seed' in result and not result.get('human'):
            block = (job_id, result['map_seed'], canonical(result['map']), ids)
        else:
            block = result['game_id']
        if result.get('rating_policy') in EVIDENCE_POLICIES:
            if result['game_id'] in self.games:
                return
            self.games.add(result['game_id'])
            if block not in self.seen:
                self.seen.add(block)
                self.blocks.update(ids)
            for left, right, weight, _ in comparisons(result):
                self.pairs[left, right] += weight
            return
        if block in self.seen:
            return
        self.seen.add(block)
        self.blocks.update(ids)
        # One effective comparison per strategy per independent map. Duplicate
        # seats and extra rotations cannot manufacture additional evidence.
        for pair in combinations(ids, 2):
            self.pairs[pair] += 1 / (len(ids) - 1)

    def annotate(self, report):
        rows = {row['entrant_id']: row for row in report['ranking']}
        anchor = report['anchor']
        graph = {key: set() for key in rows}
        for left, right in self.pairs:
            if left in graph and right in graph:
                graph[left].add(right)
                graph[right].add(left)
        mean_pool = report.get('rating_model') == 'codeforces'
        modern = report.get('rating_model') in ('egf-gor', 'codeforces')
        rated = {key for key in rows if self.blocks[key]}
        origin = next(iter(sorted(rated)), None) if mean_pool else anchor
        connected, pending = ({origin}, [origin]) if origin in graph else (set(), [])
        while pending:
            for key in graph[pending.pop()] - connected:
                connected.add(key)
                pending.append(key)
        # Multiple disconnected components have unidentified relative offsets.
        free = sorted(connected if mean_pool else connected - {anchor})
        indices = {key: i for i, key in enumerate(free)}
        information = np.zeros((len(free), len(free)))
        for (left, right), weight in self.pairs.items():
            if left not in connected or right not in connected:
                continue
            # GoR is nonlinear in its absolute coordinate. The displayed WAIT
            # offset must never be fed back into probabilities or derivatives.
            a, b = (rows[key].get('raw_elo', rows[key]['elo']) for key in (left,right))
            mode = 'ffa' if mean_pool else '1v1'
            p = probability(mode, a, b) if modern else expected(a, b)
            sa, sb = (slope(mode, a), slope(mode, b)) if modern else (LOGISTIC_SLOPE, LOGISTIC_SLOPE)
            value = weight * p * (1 - p)
            i, j = indices.get(left), indices.get(right)
            if i is not None:
                information[i, i] += value * sa * sa
            if j is not None:
                information[j, j] += value * sb * sb
            if i is not None and j is not None:
                information[i, j] -= value * sa * sb
                information[j, i] -= value * sa * sb
        variance = None
        identified = (len(free) > 1 and connected == rated) if mean_pool else self.blocks[anchor] >= PROVISIONAL_GAMES
        if free and identified:
            try:
                if mean_pool:
                    # Invert on the sum-zero subspace, keeping the 1000 mean
                    # fixed without treating any strategy as a perfect anchor.
                    constraint = np.ones_like(information) / len(free)
                    matrix = information + constraint
                    covariance = np.linalg.inv(matrix) - constraint
                else:
                    matrix = information
                    covariance = np.linalg.inv(matrix)
                if np.linalg.cond(matrix) < 1e12:
                    variance = np.diag(covariance)
            except np.linalg.LinAlgError:
                pass
        for key, row in rows.items():
            row['elo_ci'] = None
            row['ci_blocks'] = self.blocks[key]
            if key == anchor:
                error = 0.0  # The origin is fixed by definition.
            elif variance is not None and key in indices and self.blocks[key] >= PROVISIONAL_GAMES:
                error = 1.959963984540054 * math.sqrt(max(0, variance[indices[key]]))
            else:
                continue
            if math.isfinite(error):
                row['elo_ci'] = {'level': .95, 'lower': row['elo'] - error,
                                 'upper': row['elo'] + error, 'half_width': error,
                                 'method': ('mode-fisher-independent-maps-v3' if modern else 'rank-fisher-independent-maps-v2')}
