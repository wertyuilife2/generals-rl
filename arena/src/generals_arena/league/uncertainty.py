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
        connected, pending = {anchor}, [anchor]
        while pending:
            for key in graph[pending.pop()] - connected:
                connected.add(key)
                pending.append(key)
        free = sorted(connected - {anchor})
        indices = {key: i for i, key in enumerate(free)}
        information = np.zeros((len(free), len(free)))
        for (left, right), weight in self.pairs.items():
            if left not in connected or right not in connected:
                continue
            p = expected(rows[left]['elo'], rows[right]['elo'])
            value = weight * p * (1 - p) * LOGISTIC_SLOPE ** 2
            i, j = indices.get(left), indices.get(right)
            if i is not None:
                information[i, i] += value
            if j is not None:
                information[j, j] += value
            if i is not None and j is not None:
                information[i, j] -= value
                information[j, i] -= value
        variance = None
        if free and self.blocks[anchor] >= PROVISIONAL_GAMES:
            try:
                if np.linalg.cond(information) < 1e12:
                    variance = np.diag(np.linalg.inv(information))
            except np.linalg.LinAlgError:
                pass  # Disconnected or numerically unidentified comparisons.
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
                                 'method': 'rank-fisher-independent-maps-v2'}
