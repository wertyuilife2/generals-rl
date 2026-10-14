"""Derived leaderboard statistics; never change match results or Elo history."""
from collections import defaultdict
import json
import re
import sqlite3
import threading


from ..execution import verified_replay
from ..util import digest
from .rating import ranks
from .uncertainty import RatingUncertainty


def replay_kills(result, blob):
    """Backfill old results using env's backend, without re-running any policy."""
    kills = [0] * len(result['entrant_ids'])
    eliminated = [None] * len(kills)
    for state, outcome in verified_replay(blob, result):
        if outcome is not None:
            for move in outcome.moves:
                if move.eliminated is not None:
                    kills[move.player_id] += 1
                    eliminated[move.eliminated] = state.tick
    if eliminated != result.get('capture_ticks', result['elimination_ticks']):
        raise ValueError('replay result mismatch')
    return kills


def match_statistics(result, kills):
    """Each strategy has one sample per game, regardless of its seat count."""
    ids = result['entrant_ids']
    if (len(kills) != len(ids) or
            any(type(k) is not int or not 0 <= k < len(ids) for k in kills) or
            sum(kills) != sum(t is not None for t in result.get('capture_ticks', result['elimination_ticks']))):
        raise ValueError('invalid kill counts')
    seats = defaultdict(list)
    for seat, entrant in enumerate(ids):
        seats[entrant].append(seat)
    winner = result['winner_seat']
    # Normalize survival placement, not elapsed time: in a duel the losing
    # general is normally captured on the final tick, but must still score 0.
    # Average ranks give tied eliminations / remaining survivors equal credit.
    survival = [(len(ids) - rank) / (len(ids) - 1) for rank in ranks(result)]
    return {entrant: {
        'win_rate': float(winner in group),
        'survival_index': sum(survival[seat] for seat in group) / len(group),
        'kill_index': sum(kills[seat] / (len(ids) - 1) for seat in group) / len(group),
    } for entrant, group in seats.items()}


def version_deltas(rows):
    """Compare current ratings of adjacent registered name-vN versions."""
    series = defaultdict(list)
    for row in rows:
        row.update(elo_delta=None, previous_version=None)
        match = re.fullmatch(r'(.+)-v(0|[1-9][0-9]*)', row['name'])
        if match and row['kind'] != 'human':
            series[row['kind'], match[1]].append((int(match[2]), row))
    for versions in series.values():
        versions.sort(key=lambda item: item[0])
        for (_, previous), (_, current) in zip(versions, versions[1:]):
            current['previous_version'] = previous['name']
            if current['games'] and previous['games']:
                current['elo_delta'] = current['elo'] - previous['elo']


class LeaderboardStatistics:
    """Incremental in-memory aggregates with a disposable on-disk replay cache."""

    def __init__(self, store):
        self.store = store
        self.path = store.directory / 'statistics.sqlite3'
        self.lock = threading.Lock()
        self.sequence = -1
        self.completed = 0
        self.totals = {}
        self.uncertainty = RatingUncertainty()
        self.revision = None
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS kills '
                       '(game_id TEXT PRIMARY KEY, signature TEXT NOT NULL, counts TEXT NOT NULL)')

    def _kills(self, result, cache):
        if 'kills' in result:
            return result['kills']
        signature = digest({key: result[key] for key in
                            ('initial_digest', 'final_digest', 'ticks', 'elimination_ticks')})
        cached = cache.execute('SELECT counts FROM kills WHERE game_id=? AND signature=?',
                               (result['game_id'], signature)).fetchone()
        if cached:
            return json.loads(cached[0])
        _, blob = self.store.replay(result['game_id'])
        kills = replay_kills(result, blob)
        cache.execute('INSERT OR REPLACE INTO kills VALUES (?,?,?)',
                      (result['game_id'], signature, json.dumps(kills)))
        cache.commit()  # Backfill is resumable even if the server is interrupted.
        return kills

    def leaderboard(self, progress=None):
        with self.lock:
            report = self.store.leaderboard()
            count = report['completed_games']
            with self.store.connect() as db:
                row = db.execute("SELECT value FROM meta WHERE key='rating_revision'").fetchone()
                revision = row[0] if row else None
            if count < self.completed or revision != self.revision:
                self.sequence, self.completed, self.totals = -1, 0, {}
                self.uncertainty = RatingUncertainty()
                self.revision = revision
            # Limit to the same completed prefix as the rating snapshot, even if
            # a worker commits another match while this request is running.
            with self.store.connect() as db:
                pending = db.execute("SELECT sequence,result,job_id FROM games WHERE status='complete' "
                                     'AND sequence>? ORDER BY sequence LIMIT ?',
                                     (self.sequence, count - self.completed)).fetchall()
            if pending:
                with sqlite3.connect(self.path, timeout=30) as cache:
                    for row in pending:
                        result = json.loads(row['result'])
                        samples = match_statistics(result, self._kills(result, cache))
                        self.uncertainty.add(result, row['job_id'])
                        for entrant, sample in samples.items():
                            total = self.totals.setdefault(entrant, dict(games=0, win_rate=0.,
                                                                       survival_index=0., kill_index=0.))
                            total['games'] += 1
                            for key, value in sample.items():
                                total[key] += value
                        self.sequence = row['sequence']
                        self.completed += 1
                        if progress:
                            progress(self.completed, count)
            for row in report['ranking']:
                total = self.totals.get(row['entrant_id'])
                for key in ('win_rate', 'survival_index', 'kill_index'):
                    row[key] = total[key] / total['games'] if total else None
            version_deltas(report['ranking'])
            self.uncertainty.annotate(report)
            return report
