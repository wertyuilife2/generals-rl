"""Leaderboard metrics count games equally and use actual capture attribution."""
import gzip
import json
from unittest.mock import patch

import numpy as np
import pytest

from generals.core.state import Action, Direction, GameState, MapLayout
from generals_arena.execution import RecordedGame, close_game
from generals_arena.league.config import LeagueConfig
from generals_arena.league.statistics import LeaderboardStatistics, match_statistics, replay_kills, version_deltas
from generals_arena.league.store import LeagueStore


def test_multiplayer_metrics_average_seats_then_games_and_include_timeouts(tmp_path):
    store = LeagueStore.create(tmp_path / 'league', reference='random')
    a = store.register({'name':'test-v1', 'kind':'baseline', 'policy':'aggressive'})['entrant_id']
    b = store.anchor
    idle = store.account('not-yet-played')['entrant']['entrant_id']
    stats = LeaderboardStatistics(store)
    def complete(key, ids, ticks, eliminations, kills, winner):
        plan = {'game_id':key, 'entrant_ids':ids}
        store.insert_plans([plan])
        store.complete_result({**plan, 'ticks':ticks, 'elimination_ticks':eliminations,
                               'kills':kills, 'winner_seat':winner}, b'')

    complete('four', [a,a,b,b], 10, [2,None,8,6], [0,3,0,0], 1)
    first = stats.leaderboard()
    first_a = next(r for r in first['ranking'] if r['entrant_id'] == a)
    assert first_a['survival_index'] == pytest.approx(.5)
    assert first_a['kill_index'] == pytest.approx(.5)
    assert first_a['win_rate'] == 1  # Winner was the second duplicate seat.

    complete('two', [a,b], 20, [20,None], [0,1], 1)
    complete('timeout', [a,b], 30, [None,None], [0,0], None)
    store.insert_plans([{'game_id':'failed', 'entrant_ids':[a,b]}])
    store.fail_game({'game_id':'failed', 'entrant_ids':[a,b]}, b'')
    store.insert_plans([{'game_id':'live', 'entrant_ids':[a,b]}], status='live')
    report = stats.leaderboard()
    rows = {r['entrant_id']: r for r in report['ranking']}
    assert report['completed_games'] == 3
    assert rows[a]['survival_index'] == pytest.approx((.5 + 0 + .5) / 3)
    assert rows[a]['kill_index'] == pytest.approx(.5 / 3)
    assert rows[a]['win_rate'] == pytest.approx(1 / 3)
    assert rows[b]['survival_index'] == pytest.approx((.5 + 1 + .5) / 3)
    assert rows[b]['kill_index'] == pytest.approx(1 / 3)
    assert all(rows[idle][key] is None for key in ('win_rate','survival_index','kill_index'))
    assert stats.leaderboard() == report
    assert LeaderboardStatistics(store).leaderboard() == report


def test_native_kills_do_not_transfer_to_the_winner_and_old_replay_is_cached(tmp_path):
    store = LeagueStore.create(tmp_path / 'league', LeagueConfig(max_ticks=10), reference='random')
    a = store.account('a')['entrant']['entrant_id']
    b = store.account('b')['entrant']['entrant_id']
    state = GameState(MapLayout(np.zeros((1,3), dtype=np.uint8)),
                      np.array([[2,2,2]]), np.array([[0,1,2]]), np.array([[60,20,1]]),
                      np.ones(3, dtype=bool), np.array([0,1,2]))
    plan = {'game_id':'captures', 'entrant_ids':[a,b,store.anchor],
            'controller_seeds':[1,2,3], 'map_seed':1,
            'map':{'height':3,'width':3,'players':3}}
    with patch('generals_arena.league.game.generate_map', return_value=state):
        game = RecordedGame(plan, store.config, [None]*3)
    try:
        game.advance({1:Action.move(1, Direction.RIGHT)})  # P1 kills P2.
        game.advance({0:Action.move(0, Direction.RIGHT)})  # P0 kills P1.
        result, blob = game.result(), game.replay_blob()
        assert result['winner_seat'] == 0
        assert result['kills'] == [1,1,0]
        assert result['elimination_ticks'] == [None,2,1]
        assert replay_kills(result, blob) == [1,1,0]
        samples = match_statistics(result, result['kills'])
        assert samples[a]['kill_index'] == samples[b]['kill_index'] == .5
        assert samples[store.anchor]['survival_index'] == 0
        trace = json.loads(gzip.decompress(blob))
        trace['digests'][0] = 'corrupt'
        with pytest.raises(ValueError, match='diverged at tick 1'):
            replay_kills(result, gzip.compress(json.dumps(trace).encode()))
        result.pop('kills')  # An existing record produced before tracking kills.
        store.insert_plans([plan])
        store.complete_result(result, blob)
        before = store.replay(plan['game_id']), store.leaderboard(), store.manifest
        with patch('generals_arena.league.statistics.replay_kills', wraps=replay_kills) as replay:
            stats = LeaderboardStatistics(store)
            report = stats.leaderboard()
            assert replay.call_count == 1
            assert stats.leaderboard() == report
            assert LeaderboardStatistics(store).leaderboard() == report
            assert replay.call_count == 1  # Survives restarting the web server.
        assert (store.replay(plan['game_id']), store.leaderboard(), store.manifest) == before
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


@pytest.mark.parametrize('ticks,eliminations,winner,expected', [
    (20, [20,None], 1, [0,1]),  # Final-tick loser is not a survivor.
    (0, [0,None], 1, [0,1]),  # Immediate surrender uses the same ranking.
    (30, [None,None], None, [.5,.5]),
    (30, [10,None,None], None, [0,.75,.75]),
    (20, [10,10,None], 2, [.25,.25,1]),
    (80, [10,20,30,40,50,60,80,None], 7, [i/7 for i in range(8)]),
])
def test_survival_index_uses_normalized_placement_and_average_ties(
        ticks, eliminations, winner, expected):
    ids = [f'p{i}' for i in range(len(eliminations))]
    kills = [0] * len(ids)
    kills[-1] = sum(t is not None for t in eliminations)
    result = dict(entrant_ids=ids, ticks=ticks, elimination_ticks=eliminations,
                  winner_seat=winner)
    actual = match_statistics(result, kills)
    assert [actual[key]['survival_index'] for key in ids] == pytest.approx(expected)
    # Increasing all elimination times without changing order is immaterial.
    stretched = {**result, 'ticks':ticks*3,
                 'elimination_ticks':[t*3 if t is not None else None for t in eliminations]}
    assert match_statistics(stretched, kills) == actual
