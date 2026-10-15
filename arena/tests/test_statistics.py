"""Leaderboard metrics count games equally and use actual capture attribution."""
import gzip
import json
from unittest.mock import patch

import numpy as np
import pytest

from generals.core.state import Action, Direction, GameState, MapLayout
from generals_arena.execution import RecordedGame, close_game
from generals_arena.league.config import LeagueConfig
from generals_arena.league.statistics import LeaderboardStatistics, WinMatrix, DuelStatistics, match_statistics, replay_kills, version_deltas
from generals_arena.league.store import LeagueStore


def test_cached_report_is_isolated_and_invalidates_on_roster_retirement_and_rebuild(tmp_path):
    store = LeagueStore.create(tmp_path/'league', reference='random')
    bot, _ = store.add({'name':'aggressive','kind':'baseline','policy':'aggressive'}, games=2)
    stats = LeaderboardStatistics(store,'1v1')
    first = stats.leaderboard()
    with patch.object(stats.uncertainty,'annotate',side_effect=AssertionError('should use cached report')):
        first['ranking'].clear()
        assert stats.leaderboard()['ranking']
    human = store.account('new human')['entrant']['entrant_id']
    assert human in [r['entrant_id'] for r in stats.leaderboard()['ranking']]
    store.set_retired(bot['entrant_id'])
    assert bot['entrant_id'] not in [r['entrant_id'] for r in stats.leaderboard()['ranking']]
    store.set_retired(bot['entrant_id'],False)
    assert bot['entrant_id'] in [r['entrant_id'] for r in stats.leaderboard()['ranking']]
    prior = stats.uncertainty
    store.rebuild()
    stats.leaderboard()
    assert stats.uncertainty is not prior


def test_duel_metrics_separate_draws_speed_and_compute_cost_without_counting_self_play():
    stats = DuelStatistics()
    stats.add(dict(entrant_ids=['a','b'],winner_seat=0,ticks=100,decision_timing=[
        dict(seat=0,calls=10,total_ns=10_000_000),dict(seat=1,calls=10,total_ns=0)]))
    stats.add(dict(entrant_ids=['a','b'],winner_seat=None,ticks=3000,decision_timing=[
        dict(seat=0,calls=1000,total_ns=9_000_000_000)]))
    stats.add(dict(entrant_ids=['a','b'],winner_seat=1,ticks=20,reason='surrender',
                   decision_timing=[dict(seat=0,calls=0,total_ns=0)]))
    stats.add(dict(entrant_ids=['a','a'],winner_seat=0,ticks=1))
    stats.add(dict(entrant_ids=['a','b','c'],winner_seat=0,ticks=1))
    rows = [dict(entrant_id=key) for key in ('a','b','idle')]
    stats.annotate(rows)
    a,b,idle = rows
    assert a['duel_games'] == b['duel_games'] == 3
    assert a['duel_win_rate'] == a['draw_rate'] == pytest.approx(1/3)
    assert a['mean_win_ticks'] == 100 and b['mean_win_ticks'] == 20
    assert a['mean_decision_ms'] == 5  # Mean of game means, not of all 1,010 calls.
    assert a['timed_games'] == 2
    assert b['mean_decision_ms'] == 0 and b['timed_games'] == 1
    assert idle['mean_decision_ms'] is None and idle['mean_win_ticks'] is None
    assert idle['duel_win_rate'] is None and idle['draw_rate'] is None


def test_duel_derived_metrics_reload_and_rebuild_without_rescoring(tmp_path):
    store=LeagueStore.create(tmp_path/'league',reference='random')
    a=store.register({'name':'a-v1','kind':'baseline','policy':'aggressive'})['entrant_id']
    plan=dict(game_id='duel',entrant_ids=[a,store.anchor])
    store.insert_plans([plan])
    store.complete_result({**plan,'ticks':80,'winner_seat':0,'kills':[1,0],
                           'elimination_ticks':[None,80]},b'original recording')
    saved=store.replay('duel')
    stats=LeaderboardStatistics(store,'1v1')
    board=stats.leaderboard()
    row=next(r for r in board['ranking'] if r['entrant_id']==a)
    assert row['duel_win_rate']==1 and row['draw_rate']==0 and row['mean_win_ticks']==80
    assert row['mean_decision_ms'] is None and row['timed_games']==0
    assert stats.leaderboard()==LeaderboardStatistics(store,'1v1').leaderboard()==board
    assert 'duel_win_rate' not in LeaderboardStatistics(store,'ffa').leaderboard()['ranking'][0]
    store.rebuild()
    assert stats.leaderboard()==board and store.replay('duel')==saved
    store.verify_execution()


def test_multiplayer_metrics_average_seats_then_games_and_include_timeouts(tmp_path):
    store = LeagueStore.create(tmp_path / 'league', reference='random')
    a = store.register({'name':'test-v1', 'kind':'baseline', 'policy':'aggressive'})['entrant_id']
    b = store.anchor
    idle = store.account('not-yet-played')['entrant']['entrant_id']
    stats = LeaderboardStatistics(store, 'ffa')
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
    complete('three', [a,b,b], 20, [10,None,20], [0,2,0], 1)
    complete('timeout', [a,b,b], 30, [None,None,None], [0,0,0], None)
    store.insert_plans([{'game_id':'failed', 'entrant_ids':[a,b]}])
    store.fail_game({'game_id':'failed', 'entrant_ids':[a,b]}, b'')
    store.insert_plans([{'game_id':'live', 'entrant_ids':[a,b]}], status='live')
    report = stats.leaderboard()
    rows = {r['entrant_id']: r for r in report['ranking']}
    assert report['completed_games'] == 3
    assert rows[a]['survival_index'] == pytest.approx((.5 + 0 + .5) / 3)
    assert rows[a]['kill_index'] == pytest.approx(.5 / 3)
    assert rows[a]['win_rate'] == pytest.approx(1 / 3)
    assert rows[b]['survival_index'] == pytest.approx((.5 + .75 + .5) / 3)
    assert rows[b]['kill_index'] == pytest.approx(.5 / 3)
    assert all(rows[idle][key] is None for key in ('win_rate','survival_index','kill_index'))
    assert stats.leaderboard() == report
    assert LeaderboardStatistics(store, 'ffa').leaderboard() == report
    ids = report['win_matrix']['entrant_ids']
    pair = report['win_matrix']['cells'][ids.index(a)][ids.index(b)]
    assert pair == {'games':3,'win_rate':pytest.approx(1/6),'draw_rate':pytest.approx(1/3)}
    assert idle not in ids  # Human accounts have no matrix row or column.
    store.abort_live()
    store.rebuild()
    assert stats.leaderboard()['win_matrix'] == report['win_matrix']


def test_pair_matrix_ties_copies_permutations_and_row_orientation():
    model = WinMatrix()
    game = dict(entrant_ids=['a','a','b','c'],elimination_ticks=[None,1,2,None])
    model.add(game)
    model.add(dict(entrant_ids=['a','b','c'],elimination_ticks=[3,2,None]))
    rows = [{'entrant_id':key} for key in ('c','b','a','unplayed')]
    matrix = model.report(rows)
    assert matrix['cells'][2][1] == dict(games=2,win_rate=.75,draw_rate=0.)
    assert matrix['cells'][1][2] == dict(games=2,win_rate=.25,draw_rate=0.)
    assert matrix['cells'][2][0] == dict(games=2,win_rate=0.,draw_rate=.25)
    assert matrix['cells'][0][2] == dict(games=2,win_rate=.75,draw_rate=.25)
    assert all(matrix['cells'][i][i] is None for i in range(4))
    assert matrix['cells'][3] == [None]*4
    other = WinMatrix()
    other.add(dict(entrant_ids=list(reversed(game['entrant_ids'])),elimination_ticks=list(reversed(game['elimination_ticks']))))
    single = WinMatrix(); single.add(game)
    assert other.report(rows) == single.report(rows)


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
            stats = LeaderboardStatistics(store, 'ffa')
            report = stats.leaderboard()
            assert replay.call_count == 1
            assert stats.leaderboard() == report
            assert LeaderboardStatistics(store, 'ffa').leaderboard() == report
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
