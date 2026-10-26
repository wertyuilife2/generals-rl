"""Leaderboard metrics count games equally and use actual capture attribution."""
import gzip
import json
from unittest.mock import patch

import numpy as np
import pytest

from generals.core.state import Action, Direction, GameState, MapLayout
from generals_arena.matches.game import RecordedGame
from generals_arena.matches.game import close_game
from generals_arena.matches.replay import verified_replay
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.leaderboard import LeaderboardStatistics
from generals_arena.competitions.leaderboard import match_statistics
from generals_arena.competitions.leaderboard import version_deltas
from generals_arena.competitions.store import CompetitionStore


def test_multiplayer_metrics_average_seats_then_games_and_include_timeouts(tmp_path):
    store = CompetitionStore.create(tmp_path / 'league', reference='random')
    a = store.register({'name':'test-v1', 'kind':'baseline', 'policy':'aggressive'})['entrant_id']
    b = store.reference
    idle = store.account('not-yet-played')['entrant']['entrant_id']
    stats = LeaderboardStatistics(store, 'ffa')
    def complete(key, ids, ticks, eliminations, kills, winner):
        plan = {'game_id':key, 'entrant_ids':ids}
        store.insert_plans([plan])
        store.complete_result({**plan, 'ticks':ticks, 'elimination_ticks':eliminations,
                               'kills':kills, 'winner_seat':winner,
                               'final_alive':[tick is None for tick in eliminations]}, b'')

    complete('four', [a,a,b,b], 10, [2,None,8,6], [0,3,0,0], 1)
    first = stats.leaderboard()
    first_a = next(r for r in first['ranking'] if r['entrant_id'] == a)
    assert first_a['survival_index'] == pytest.approx(.6)
    assert first_a['kill_index'] == pytest.approx(.5)
    assert first_a['win_rate'] == 1  # Winner was the second duplicate seat.

    complete('two', [a,b], 20, [20,None], [0,1], 1)
    complete('timeout', [a,b], 30, [None,None], [0,0], None)
    store.insert_plans([{'game_id':'failed', 'entrant_ids':[a,b]}])
    store.fail_game({'game_id':'failed', 'entrant_ids':[a,b]}, b'')
    store.insert_plans([{'game_id':'live', 'entrant_ids':[a,b]}], status='live')
    report = stats.leaderboard()
    rows = {r['entrant_id']: r for r in report['ranking']}
    assert report['completed_games'] == 1
    assert rows[a]['survival_index'] == pytest.approx(.6)
    assert rows[a]['kill_index'] == pytest.approx(.5)
    assert rows[a]['win_rate'] == 1
    assert rows[b]['survival_index'] == pytest.approx(.7)
    assert rows[b]['kill_index'] == 0
    assert all(rows[idle][key] is None for key in ('win_rate','survival_index','kill_index'))
    assert stats.leaderboard() == report
    assert LeaderboardStatistics(store, 'ffa').leaderboard() == report
    duels = LeaderboardStatistics(store, '1v1').leaderboard()
    duel_rows = {row['entrant_id']:row for row in duels['ranking']}
    assert duels['completed_games'] == 2
    assert duel_rows[a]['survival_index'] == 1 and duel_rows[a]['kill_index'] == 0
    assert duel_rows[b]['kill_index'] == .5 and duel_rows[b]['win_rate'] == .5


def test_native_kills_do_not_transfer_and_statistics_use_recorded_counts(tmp_path):
    store = CompetitionStore.create(tmp_path / 'league', LeagueConfig(max_ticks=10), reference='random')
    a = store.account('a')['entrant']['entrant_id']
    b = store.account('b')['entrant']['entrant_id']
    state = GameState(MapLayout(np.zeros((1,3), dtype=np.uint8)),
                      np.array([[2,2,2]]), np.array([[0,1,2]]), np.array([[60,20,1]]),
                      np.ones(3, dtype=bool), np.array([0,1,2]))
    plan = {'game_id':'captures', 'entrant_ids':[a,b,store.reference],
            'controller_seeds':[1,2,3], 'map_seed':1,
            'map':{'height':3,'width':3,'players':3}}
    with patch('generals_arena.matches.game.generate_map', return_value=state):
        game = RecordedGame(plan, store.config, [None]*3)
    try:
        game.advance({1:Action.move(1, Direction.RIGHT)})  # P1 kills P2.
        game.advance({0:Action.move(0, Direction.RIGHT)})  # P0 kills P1.
        result, blob = game.result(), game.replay_blob()
        assert result['winner_seat'] == 0
        assert result['kills'] == [1,1,0]
        assert result['elimination_ticks'] == [None,2,1]
        assert list(verified_replay(blob, result))[-1][0].tick == 2
        samples = match_statistics(result, result['kills'])
        assert samples[a]['kill_index'] == samples[b]['kill_index'] == .5
        assert samples[store.reference]['survival_index'] == .5
        trace = json.loads(gzip.decompress(blob))
        trace['digests'][0] = 'corrupt'
        with pytest.raises(ValueError, match='diverged at tick 1'):
            list(verified_replay(gzip.compress(json.dumps(trace).encode()), result))
        store.insert_plans([plan])
        store.complete_result(result, blob)
        before = store.replay(plan['game_id']), store.leaderboard('ffa'), store.manifest
        stats = LeaderboardStatistics(store, 'ffa')
        report = stats.leaderboard()
        assert report['completed_games'] == 1
        assert stats.leaderboard() == report
        assert LeaderboardStatistics(store, 'ffa').leaderboard() == report
        assert not (store.directory/'statistics.sqlite3').exists()
        assert (store.replay(plan['game_id']), store.leaderboard('ffa'), store.manifest) == before
        store.verify_execution()
    finally:
        close_game(game)


def test_version_delta_uses_numeric_predecessor_and_current_elo():
    def row(name, elo, games=1, kind='cpp'):
        return dict(name=name, elo=elo, games=games, kind=kind)
    rows = [row('model-bird-v10', 400), row('model-bird-v1', 200),
            row('model-bird-v2', 150), row('other-bird-v9', 999),
            row('model-bird-v11', 0, games=0), row('player-v1', 10, kind='human'),
            row('player-v2', 20, kind='human'), row('random', 0, kind='baseline')]
    version_deltas(rows)
    assert rows[0]['elo_delta'] == 250
    assert rows[0]['previous_version'] == 'model-bird-v2'
    assert rows[2]['elo_delta'] == -50
    assert rows[2]['previous_version'] == 'model-bird-v1'
    for index in (1,3,4,5,6,7):
        assert rows[index]['elo_delta'] is None


def test_confidence_coverage_and_totals_are_isolated_by_rating_mode(tmp_path):
    store = CompetitionStore.create(tmp_path/'league', reference='random')
    a = store.register({'name':'a','kind':'baseline','policy':'aggressive'})['entrant_id']
    b = store.reference
    c = store.register({'name':'c','kind':'baseline','policy':'expansion'})['entrant_id']
    for index in range(6):
        plan = dict(game_id=f'ffa-{index}', entrant_ids=[a,b,c], map_seed=index,
                    map={'width':9,'height':9,'players':3})
        store.insert_plans([plan])
        store.complete_result({**plan, 'ticks':2, 'elimination_ticks':[None,2,1],
                               'kills':[2,0,0], 'winner_seat':0, 'final_alive':[True,False,False]}, b'')
    ffa = LeaderboardStatistics(store, 'ffa').leaderboard()
    duels = LeaderboardStatistics(store, '1v1').leaderboard()
    assert ffa['completed_games'] == 6 and duels['completed_games'] == 0
    assert all(row['ci_blocks'] == 6 and row['elo_ci'] is not None for row in ffa['ranking'])
    assert all(row['ci_blocks'] == 0 and row['win_rate'] is None for row in duels['ranking'])
    assert all(row['elo_ci'] is None for row in duels['ranking'] if not row['anchor'])
