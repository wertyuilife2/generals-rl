"""League persistence, rating calibration, scheduling, replays and human privacy."""
import json
import gzip
import threading
import time
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from generals.core.mapgen import generate_map
from generals_arena.league.config import LeagueConfig
from generals_arena.league.game import RecordedGame, replay_frames
from generals_arena.league.rating import expected, update
from generals_arena.league.scheduler import make_block
from generals_arena.league.service import LiveArena, controllers_for, process_lock, run_jobs
from generals_arena.league.store import LeagueStore
from generals_arena.server import make_server


def config(**kwargs):
    return LeagueConfig(**{**dict(player_counts=(2, 4), min_side=3, max_side=5, max_ticks=8,
                                 mountain_min=0, mountain_max=.1, city_min=0, city_max=0,
                                 tick_seconds=.05), **kwargs})


def league(tmp_path, **kwargs):
    return LeagueStore.create(tmp_path / 'league', config(**kwargs))


def add(store, policy='aggressive', games=6):
    return store.add({'name': policy, 'kind': 'baseline', 'policy': policy}, games=games)


def test_variable_map_distribution_is_seeded_and_scales_with_players():
    rules = LeagueConfig(player_counts=tuple(range(2, 9)))
    assert rules.side_range(2) == (18, 23)
    assert rules.side_range(4) == (26, 33)
    assert rules.side_range(8) == (36, 46)
    shapes = set()
    for players in range(2, 9):
        for seed in range(5):
            map_config = rules.sample_map(players, seed)
            assert map_config == rules.sample_map(players, seed)
            low, high = rules.side_range(players)
            assert low <= map_config.width <= high and low <= map_config.height <= high
            assert .15 <= map_config.mountain_density <= .22
            state = generate_map(map_config, seed)
            state.validate()
            assert state.player_count == players
            shapes.add((map_config.height, map_config.width))
    assert len(shapes) > 15


def test_exact_rating_scale_anchor_and_multiplayer_outcomes():
    assert expected(500, 0) == pytest.approx(.9)
    assert expected(0, 500) == pytest.approx(.1)
    assert expected(1000, 0) == pytest.approx(81 / 82)
    result = {'entrant_ids': ['random', 'a', 'b', 'c'], 'elimination_ticks': [2, 4, 4, None]}
    rated = update(dict.fromkeys(result['entrant_ids'], 0), result, 'random')
    assert rated['random'] == 0
    assert rated['c'] == pytest.approx(16)
    assert rated['a'] == rated['b']
    assert rated['a'] == 0  # One win, one tie, one loss at equal pre-match ratings.
    reverse = {'entrant_ids': list(reversed(result['entrant_ids'])), 'elimination_ticks': list(reversed(result['elimination_ticks']))}
    assert update(dict.fromkeys(result['entrant_ids'], 0), reverse, 'random') == rated
    lose = {'entrant_ids': ['random', 'a'], 'elimination_ticks': [None, 10]}
    assert update({'random': 0, 'a': 0}, lose, 'random')['a'] == -16
    timeout = {'entrant_ids': ['random', 'a', 'b'], 'elimination_ticks': [10, None, None]}
    assert update({}, timeout, 'random')['a'] == 8


def test_seat_rotations_keep_map_and_rng_streams(tmp_path):
    store = league(tmp_path, player_counts=(4,))
    for name in ('aggressive','defensive','expansion'):
        add(store, name)
    job = store.jobs()[0]
    plans = make_block(store, job)
    assert len(plans) == 4
    assert len({p['game_id'] for p in plans}) == 4
    assert {p['map_seed'] for p in plans} == {plans[0]['map_seed']}
    streams = dict(zip(plans[0]['entrant_ids'], plans[0]['controller_seeds']))
    for plan in plans:
        assert plan['map'] == plans[0]['map']
        assert dict(zip(plan['entrant_ids'], plan['controller_seeds'])) == streams
        assert job['entrant_id'] in plan['entrant_ids']
    assert sorted(p['entrant_ids'].index(job['entrant_id']) for p in plans) == [0,1,2,3]
    assert make_block(store, job) == plans


def test_queue_budget_idempotency_resume_rebuild_and_replay(tmp_path):
    store = league(tmp_path)
    entrant, job_id = add(store, games=7)
    assert add(store, games=1000)[1] == job_id
    assert run_jobs(store, limit=3) == 3
    assert store.jobs()[0]['done'] == 3
    # One rotation remains; resume keeps the already scheduled map and seats.
    pending = store.pending(job_id)
    assert pending is not None
    reopened = LeagueStore(store.directory)
    assert run_jobs(reopened) == 4
    assert reopened.jobs()[0]['done'] == reopened.jobs()[0]['budget'] == 7
    assert reopened.jobs()[0]['status'] == 'complete'
    assert run_jobs(reopened) == 0
    report = reopened.leaderboard(2)
    assert report['completed_games'] == 7
    assert next(r for r in report['ranking'] if r['anchor'])['elo'] == 0
    assert reopened.rebuild()[2] == report
    matches = reopened.matches(players=2)
    assert matches['total'] == 7
    assert len({m['sequence'] for m in matches['matches']}) == 7
    for record in matches['matches']:
        result, blob = reopened.replay(record['game_id'])
        frames = replay_frames(blob, result)
        assert frames[-1]['tick'] == result['ticks']
        assert frames[-1]['alive'] == result['final_alive']
        compact = replay_frames(blob, result, compact=True)
        board = compact[0]
        for frame in compact[1:]:
            for cell, owner, army, structure in frame['cells']:
                y, x = divmod(cell, board['width'])
                for key, value in [('owner',owner),('army',army),('structure',structure)]:
                    board[key][y][x] = value
        for key in ('owner','army','structure'):
            assert board[key] == frames[-1][key]
    assert not any('rl' == e['kind'] for e in store.entrants().values())


def test_new_algorithm_only_schedules_its_budget(tmp_path):
    store = league(tmp_path)
    add(store, games=2)
    run_jobs(store)
    before = store.matches()['matches']
    newcomer, _ = add(store, 'defensive', games=3)
    run_jobs(store)
    after = store.matches()['matches']
    assert len(after) == 5
    assert {m['game_id'] for m in before}.issubset(m['game_id'] for m in after)
    assert all(newcomer['entrant_id'] in m['entrant_ids'] for m in after[:3])


@pytest.mark.parametrize('players', [3, 8])
def test_all_supported_seat_counts_can_finish_and_replay(tmp_path, players):
    store = league(tmp_path, player_counts=(players,), max_ticks=12)
    for i in range(players - 1):
        store.add({'name': f'bot-{i}', 'kind':'baseline', 'policy':'aggressive'}, games=1)
    assert run_jobs(store, limit=1) == 1
    record = store.matches()['matches'][0]
    assert len(record['entrant_ids']) == players
    assert len(record['elimination_ticks']) == players
    result, blob = store.replay(record['game_id'])
    assert len(replay_frames(blob, result)[-1]['alive']) == players
    assert store.leaderboard(players)['completed_games'] == 1
    corrupted = json.loads(gzip.decompress(blob))
    corrupted['digests'][-1] = 'wrong'
    with pytest.raises(ValueError, match='diverged'):
        replay_frames(gzip.compress(json.dumps(corrupted).encode()), result)


def test_source_change_cannot_silently_change_existing_league(tmp_path):
    store = league(tmp_path)
    with patch('generals_arena.league.config.LeagueConfig.manifest', return_value={'rules_id':'changed'}):
        with pytest.raises(ValueError, match='rules/source changed'):
            add(store)


def test_failure_is_unrated_and_retryable_and_workers_exclusive(tmp_path):
    store = league(tmp_path)
    add(store, games=2)
    with process_lock(store, '.worker.lock'):
        with pytest.raises(RuntimeError, match='another process'):
            run_jobs(store)
    with patch('generals_arena.league.service.controllers_for', side_effect=RuntimeError('bad policy')):
        with pytest.raises(RuntimeError, match='bad policy'):
            run_jobs(store)
    assert store.jobs()[0]['status'] == 'failed'
    assert store.leaderboard(2)['completed_games'] == 0
    assert store.matches()['total'] == 0
    assert run_jobs(store, retry=True) == 2
    assert store.jobs()[0]['status'] == 'complete'


def test_no_rl_registration_or_loading(tmp_path):
    store = league(tmp_path)
    with patch('generals_arena.league.store.prepare_entrant', side_effect=AssertionError('must not load RL')):
        with pytest.raises(ValueError, match='fixed-size'):
            store.add({'name': 'rl', 'kind': 'rl', 'checkpoint': 'anything'})
    assert not store.jobs()


def test_completion_is_atomic_and_duplicate_safe(tmp_path):
    store = league(tmp_path)
    add(store, games=2)
    plans = make_block(store, store.jobs()[0])
    store.insert_plans(plans, store.jobs()[0]['id'])
    game = RecordedGame(plans[0], store.config, controllers_for(store, plans[0]))
    while not game.done:
        game.advance()
    with patch.object(store, '_score', side_effect=RuntimeError('write failure')):
        with pytest.raises(RuntimeError):
            store.complete(game)
    assert store.matches()['total'] == 0
    assert store.jobs()[0]['done'] == 0
    first = store.complete(game)
    assert store.complete(game) == first
    assert store.matches()['total'] == 1
    assert store.jobs()[0]['done'] == 1


def test_human_clock_authorization_fog_and_rated_completion(tmp_path):
    store = league(tmp_path, max_ticks=4, min_side=8, max_side=8)
    account = store.account('Alice')
    other = store.account('Bob')
    human = store.authenticate(account['token'])
    with pytest.raises(PermissionError):
        store.authenticate('invalid')
    with pytest.raises(ValueError, match='registered'):
        store.account('Alice')
    arena = LiveArena(store)
    created = arena.create_game(human, 2)
    game_id = created['game_id']
    assert arena.create_game(human, 2)['game_id'] == game_id
    board = created['board']
    invisible = np.logical_not(board['visible'])
    assert invisible.any()
    assert np.all(np.asarray(board['owner'])[invisible] == -3)
    assert np.all(np.asarray(board['army'])[invisible] == -1)
    assert 'map_seed' not in created and 'initial' not in created
    with pytest.raises(PermissionError):
        arena.action(game_id, other['entrant']['entrant_id'], 0, 0)
    with pytest.raises(ValueError, match='stale'):
        arena.action(game_id, human, 0, 100)
    assert arena.action(game_id, human, 0, 0)['queued_for_tick'] == 1
    assert arena.view(game_id, human)['board']['tick'] == 0
    # Only the server clock can advance the board.
    arena.thread.start()
    try:
        deadline = time.monotonic() + 3
        while not arena.view(game_id, human)['done'] and time.monotonic() < deadline:
            time.sleep(.02)
        final = arena.view(game_id, human)
        assert final['done'] and not final['error']
        assert store.leaderboard(2)['completed_games'] == 1
        row = next(r for r in store.leaderboard(2)['ranking'] if r['entrant_id'] == human)
        assert row['games'] == 1
        assert store.rebuild()[2] == store.leaderboard(2)
        assert replay_frames(*reversed(store.replay(game_id)))[-1]['tick'] == 4
    finally:
        arena.stop.set()
        arena.thread.join(2)


def test_abandoned_server_game_is_visible_but_unrated(tmp_path):
    store = league(tmp_path)
    user = store.account('human')['entrant']['entrant_id']
    arena = LiveArena(store)
    game = arena.create_game(user, 2)
    store.abort_live()
    assert store.leaderboard(2)['completed_games'] == 0
    assert store.matches()['matches'][0]['status'] == 'aborted'
    with pytest.raises(ValueError, match='not found'):
        store.replay(game['game_id'])


def test_human_matchmaking_can_select_non_random_opponents(tmp_path):
    store = league(tmp_path)
    add(store)
    human = store.account('matchmaking')['entrant']['entrant_id']
    job = {'id':'first','entrant_id':human,'done':0,'budget':1}
    # After the first anchor calibration game, normal adaptive sampling applies.
    with patch.object(store, 'pair_counts', return_value={store.anchor:1}):
        opponents = set()
        for index in range(30):
            plan = make_block(store, {**job, 'id':str(index)}, human=True, players=2)[0]
            opponents.update(key for key in plan['entrant_ids'] if key != human)
    assert len(opponents) == 2


def test_http_frontend_and_privacy_boundaries(tmp_path):
    store = league(tmp_path)
    server = make_server(store, port=0)
    server.arena.start()
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = f'http://127.0.0.1:{server.server_address[1]}'
    def request(path, data=None, headers=None):
        request_headers = {'Content-Type':'application/json', **(headers or {})}
        body = None if data is None else json.dumps(data).encode()
        with urlopen(Request(url + path, data=body, headers=request_headers)) as response:
            return response, response.read()
    try:
        response, html = request('/')
        assert b'GENERALS' in html and response.headers['Content-Security-Policy']
        _, raw = request('/api/meta')
        assert 'factory_kwargs' not in raw.decode()
        with pytest.raises(HTTPError) as error:
            request('/api/play', {'players':2})
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            request('/api/players', {'name':'attack'}, {'Origin':'https://example.com'})
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            request('/api/meta', headers={'Host':'attacker.example'})
        assert error.value.code == 403
        _, raw = request('/api/players', {'name':'web-player'})
        token = json.loads(raw)['token']
        _, raw = request('/api/play', {'players':2}, {'Authorization':f'Bearer {token}'})
        game = json.loads(raw)
        with pytest.raises(HTTPError) as error:
            request('/api/replay/' + game['game_id'])
        assert error.value.code == 400
        assert game['board']['tick'] == 0
        auth = {'Authorization':f'Bearer {token}'}
        _, raw = request('/api/play/' + game['game_id'] + '/action', {'action':0, 'tick':0}, auth)
        assert json.loads(raw)['queued_for_tick'] == 1
        deadline = time.monotonic() + 3
        while not game['done'] and time.monotonic() < deadline:
            time.sleep(.02)
            _, raw = request('/api/play/' + game['game_id'], headers=auth)
            game = json.loads(raw)
        assert game['done'] and not game['error']
        _, raw = request('/api/replay/' + game['game_id'])
        replay = json.loads(raw)
        assert replay['frames'][-1]['tick'] == replay['result']['ticks']
        _, raw = request('/api/leaderboard?players=2')
        assert json.loads(raw)['completed_games'] == 1
        with pytest.raises(HTTPError) as error:
            request('/../league.sqlite3')
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.arena.close()
        server.server_close()
        server_thread.join(2)
