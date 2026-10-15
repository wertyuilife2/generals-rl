"""Derived leaderboard statistics; never change match results or Elo history."""
from collections import defaultdict
from copy import deepcopy
import json
import math
import re
import sqlite3
import threading


from ..execution import verified_replay
from ..util import digest
from .rating import ranks
from .uncertainty import RatingUncertainty
from .modes import validate_mode, mode_predicate


class WinMatrix:
    """Empirical pair outcomes: average copies within a game, then games."""
    def __init__(self):
        self.pairs = {}

    def add(self, result):
        seats = defaultdict(list)
        places = ranks(result)
        for seat,key in enumerate(result['entrant_ids']):
            seats[key].append(seat)
        keys = sorted(seats)
        for index,left in enumerate(keys):
            for right in keys[index+1:]:
                outcomes = [(places[i],places[j]) for i in seats[left] for j in seats[right]]
                total = self.pairs.setdefault((left,right),[0,0.,0.])
                total[0] += 1
                total[1] += sum(a < b for a,b in outcomes)/len(outcomes)
                total[2] += sum(a == b for a,b in outcomes)/len(outcomes)

    def report(self, rows):
        ids = [row['entrant_id'] for row in rows]
        def cell(left,right):
            values = self.pairs.get(tuple(sorted((left,right)))) if left != right else None
            if values is None:
                return None
            games,wins,draws = values
            if left > right:
                wins = games-wins-draws
            return dict(games=games,win_rate=wins/games,draw_rate=draws/games)
        return dict(entrant_ids=ids,cells=[[cell(left,right) for right in ids] for left in ids])


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


class DuelStatistics:
    """Result mix, winning duration and per-game mean decision cost."""
    def __init__(self):
        self.totals = {}

    def add(self, result):
        ids = result['entrant_ids']
        if len(ids) != 2 or ids[0] == ids[1]:
            return  # Self-play cannot describe strength against another policy.
        timings = {record.get('seat'):record for record in result.get('decision_timing', [])}
        for seat,key in enumerate(ids):
            total = self.totals.setdefault(key, dict(games=0,wins=0,draws=0,win_ticks=0,
                                                    decision_ms=0.,timed_games=0))
            total['games'] += 1
            total['draws'] += result['winner_seat'] is None
            if result['winner_seat'] == seat:
                total['wins'] += 1
                total['win_ticks'] += result['ticks']
            timing = timings.get(seat)
            if timing and timing.get('calls',0) > 0 and timing.get('total_ns',-1) >= 0:
                mean = timing['total_ns']/timing['calls']/1e6
                if math.isfinite(mean):
                    total['decision_ms'] += mean
                    total['timed_games'] += 1

    def annotate(self, rows):
        for row in rows:
            total = self.totals.get(row['entrant_id'])
            row.update(duel_games=total['games'] if total else 0,
                       duel_wins=total['wins'] if total else 0,
                       timed_games=total['timed_games'] if total else 0,
                       duel_win_rate=total['wins']/total['games'] if total else None,
                       draw_rate=total['draws']/total['games'] if total else None,
                       mean_win_ticks=total['win_ticks']/total['wins'] if total and total['wins'] else None,
                       mean_decision_ms=total['decision_ms']/total['timed_games'] if total and total['timed_games'] else None)


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

    def __init__(self, store, mode=None):
        self.store = store
        self.mode = validate_mode(mode or store.default_mode())
        self.path = store.directory / 'statistics.sqlite3'
        self.lock = threading.Lock()
        self.sequence = -1
        self.completed = 0
        self.totals = {}
        self.uncertainty = RatingUncertainty()
        self.matrix = WinMatrix()
        self.duels = DuelStatistics()
        self.revision = None
        self.report_key = None
        self.report = None
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
            report = self.store.leaderboard(self.mode, include_retired=True)
            count = report['completed_games']
            with self.store.connect() as db:
                row = db.execute("SELECT value FROM meta WHERE key='rating_revision'").fetchone()
                revision = row[0] if row else None
            report_key = digest([report, revision])
            if report_key == self.report_key:
                return deepcopy(self.report)
            if count < self.completed or revision != self.revision:
                self.sequence, self.completed, self.totals = -1, 0, {}
                self.uncertainty = RatingUncertainty()
                self.matrix = WinMatrix()
                self.duels = DuelStatistics()
                self.revision = revision
            # Limit to the same completed prefix as the rating snapshot, even if
            # a worker commits another match while this request is running.
            with self.store.connect() as db:
                pending = db.execute("SELECT sequence,result,job_id FROM games WHERE status='complete' "
                                     'AND sequence>? AND ' + mode_predicate(self.mode) + ' ORDER BY sequence LIMIT ?',
                                     (self.sequence, count - self.completed)).fetchall()
            if pending:
                with sqlite3.connect(self.path, timeout=30) as cache:
                    for row in pending:
                        result = json.loads(row['result'])
                        samples = match_statistics(result, self._kills(result, cache))
                        self.uncertainty.add(result, row['job_id'])
                        self.matrix.add(result)
                        if self.mode == '1v1':
                            self.duels.add(result)
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
            self.uncertainty.annotate(report)
            report['ranking'] = [row for row in report['ranking'] if not row.get('retired')]
            version_deltas(report['ranking'])
            if self.mode == '1v1':
                self.duels.annotate(report['ranking'])
            report['win_matrix'] = self.matrix.report([row for row in report['ranking'] if row['kind'] != 'human'])
            self.report_key, self.report = report_key, deepcopy(report)
            return report
