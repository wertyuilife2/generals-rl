"""Local Fisher intervals on raw GoR or the sum-zero Codeforces pool.

Independent maps determine sample coverage; seat rotations use fractional
pair information. These model-based intervals are not exact-coverage bounds.
"""
from collections import Counter
import math

import numpy as np

from generals_arena.competitions.rating import PROVISIONAL_GAMES
from generals_arena.competitions.rating import comparisons
from generals_arena.competitions.rating import probability
from generals_arena.competitions.rating import rating_mode
from generals_arena.competitions.rating import slope
from generals_arena.competitions.rating import validate_mode
from generals_arena.util import canonical


class RatingUncertainty:
    def __init__(self, mode):
        self.mode = validate_mode(mode)
        self.seen, self.games = set(), set()
        self.pairs, self.blocks = Counter(), Counter()

    def add(self, result, job_id=None):
        if rating_mode(result) != self.mode:
            raise ValueError('cannot mix different rating pools')
        ids = tuple(sorted(set(result['entrant_ids'])))
        if len(ids) < 2 or result['game_id'] in self.games:
            return
        self.games.add(result['game_id'])
        block = ((job_id, result['map_seed'], canonical(result['map']), ids)
                 if 'map_seed' in result and not result.get('human') else result['game_id'])
        if block not in self.seen:
            self.seen.add(block)
            self.blocks.update(ids)
        for left, right, weight, _ in comparisons(result):
            self.pairs[left, right] += weight

    def annotate(self, report):
        if report['rating_mode'] != self.mode:
            raise ValueError('report belongs to a different rating pool')
        rows = {row['entrant_id']: row for row in report['ranking']}
        for key, row in rows.items():
            row.update(elo_ci=None, ci_blocks=self.blocks[key])
        mean_pool = self.mode == 'ffa'
        anchor = report['anchor'] if not mean_pool else None
        if not mean_pool and anchor not in rows:
            return  # No WAIT reference was selected; do not invent an origin.
        graph = {key: set() for key in rows}
        for left, right in self.pairs:
            if left in graph and right in graph:
                graph[left].add(right)
                graph[right].add(left)
        rated = {key for key in rows if self.blocks[key]}
        origin = next(iter(sorted(rated)), None) if mean_pool else anchor
        connected, pending = ({origin}, [origin]) if origin in graph else (set(), [])
        while pending:
            for key in graph[pending.pop()] - connected:
                connected.add(key)
                pending.append(key)
        free = sorted(connected if mean_pool else connected - {anchor})
        indices = {key: index for index, key in enumerate(free)}
        information = np.zeros((len(free), len(free)))
        for (left, right), weight in self.pairs.items():
            if left not in connected or right not in connected:
                continue
            # The displayed WAIT translation must not enter GoR probabilities.
            a, b = rows[left]['raw_elo'], rows[right]['raw_elo']
            p = probability(self.mode, a, b)
            sa, sb = slope(self.mode, a), slope(self.mode, b)
            value = weight * p * (1-p)
            i, j = indices.get(left), indices.get(right)
            if i is not None:
                information[i, i] += value * sa * sa
            if j is not None:
                information[j, j] += value * sb * sb
            if i is not None and j is not None:
                information[i, j] -= value * sa * sb
                information[j, i] -= value * sa * sb
        variance = None
        identified = ((len(free) > 1 and connected == rated) if mean_pool
                      else self.blocks[anchor] >= PROVISIONAL_GAMES)
        if free and identified:
            try:
                if mean_pool:
                    constraint = np.ones_like(information) / len(free)
                    matrix = information + constraint
                    covariance = np.linalg.inv(matrix) - constraint
                else:
                    matrix = information
                    covariance = np.linalg.inv(matrix)
                if np.linalg.cond(matrix) < 1e12:
                    variance = np.diag(covariance)
            except np.linalg.LinAlgError:
                pass  # Relative origins of disconnected pools are unidentified.
        for key, row in rows.items():
            if key == anchor:
                error = 0.0
            elif variance is not None and key in indices and self.blocks[key] >= PROVISIONAL_GAMES:
                error = 1.959963984540054 * math.sqrt(max(0, variance[indices[key]]))
            else:
                continue
            if math.isfinite(error):
                row['elo_ci'] = {'level': .95, 'lower': row['elo']-error, 'upper': row['elo']+error,
                    'half_width': error, 'method': 'mode-fisher-independent-maps-v3'}
