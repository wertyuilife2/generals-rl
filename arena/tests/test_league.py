"""Dual-rating league persistence, scheduling, replays and human privacy."""
from policy_test_utils import synthetic_placements
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
from generals_arena.competitions.config import LeagueConfig
from generals_arena.matches.game import RecordedGame
from generals_arena.matches.replay import replay_frames
from generals_arena.competitions.rating import initial_rating
from generals_arena.competitions.rating import probability
from generals_arena.competitions.rating import ranks
from generals_arena.competitions.rating import rating_mode
from generals_arena.competitions.rating import score
from generals_arena.competitions.matchmaking import make_block
from generals_arena.web.live import LiveArena
from generals_arena.util import process_lock
from generals_arena.competitions.jobs import run_jobs
from generals_arena.matches.runner import build_controllers
from generals_arena.competitions.store import CompetitionStore
from generals_arena.web.server import make_server


def config(**kwargs):
    return LeagueConfig(**{**dict(player_counts=(2,), min_side=25, max_side=27, max_ticks=8,
                                 mountain_min=0, mountain_max=.1, city_min=0, city_max=0,
                                 tick_seconds=.05), **kwargs})


def league(tmp_path, **kwargs):
    return CompetitionStore.create(tmp_path / 'league', config(**kwargs), reference='random')


def add(store, policy='aggressive', games=6):
    return store.add({'name': policy, 'kind': 'baseline', 'policy': policy}, games=games)


def test_retry_preserves_failed_attempts_without_reusing_game_ids(tmp_path):
    from generals_arena.competitions.jobs import prepare_batch
    store = league(tmp_path)
    _, job_id = add(store, games=1)
    original = prepare_batch(store, store.jobs()[0])[0]
    attempted = []
    for _ in range(2):
        plan = prepare_batch(store, store.jobs()[0])[0]
        attempted.append(plan['game_id'])
        store.fail_game({**plan, 'policy_error': 'temporary failure', 'error_detail': {}}, b'failed replay')
        store.fail(job_id, 'temporary failure')
        store.retry(job_id)
        retried = prepare_batch(store, store.jobs()[0])
        assert len(retried) == 1
        assert retried[0]['game_id'] not in attempted
        assert retried[0]['retry_of'] == plan['game_id']
        for key in ('entrant_ids', 'instance_ids', 'map', 'map_seed', 'controller_seeds'):
            assert retried[0][key] == original[key]
    assert run_jobs(store, job_id=job_id) == 1
    assert store.jobs()[0]['done'] == 1 and store.jobs()[0]['status'] == 'complete'
    assert run_jobs(store, retry=True) == 0
    with store.connect() as db:
        failed = db.execute("SELECT replay FROM games WHERE status='failed'").fetchall()
    assert len(failed) == 2 and all(row[0] == b'failed replay' for row in failed)


def test_ffa_tick_limit_counts_eliminated_seats_as_losses(tmp_path):
    from generals_arena.matches.results import aggregate
    store = league(tmp_path, player_counts=(3,))
    a = store.register({'name': 'a', 'kind': 'baseline', 'policy': 'aggressive'})['entrant_id']
    c = store.register({'name': 'c', 'kind': 'baseline', 'policy': 'expansion'})['entrant_id']
    plan = {'game_id': 'partial_elimination', 'entrant_ids': [a, store.reference, c]}
    store.insert_plans([plan])
    result = store.complete_result({'placements': synthetic_placements([2, None, None]), **plan, 'ticks': 8, 'winner_seat': None,
        'elimination_ticks': [2, None, None], 'final_alive': [False, True, True]}, b'')
    board = store.leaderboard('ffa')
    for row in board['ranking']:
        expected = aggregate([result], row['entrant_id'])
        assert {key: row[key] for key in ('wins', 'losses', 'draws')} == {
            key: expected[key] for key in ('wins', 'losses', 'draws')}
    store.rebuild()
    assert store.leaderboard('ffa') == board


def test_variable_map_distribution_is_seeded_and_scales_with_players():
    rules = LeagueConfig(player_counts=tuple(range(2, 9)))
    assert rules.side_range(2) == (18, 23)
    assert rules.side_range(4) == (20, 26)
    assert rules.side_range(8) == (25, 33)
    shapes = set()
    for players in range(2, 9):
        for seed in range(5):
            map_config = rules.sample_map(players, seed)
            assert map_config == rules.sample_map(players, seed)
            low, high = rules.side_range(players)
            assert map_config.width is map_config.height is None
            assert map_config.mountain_density == map_config.city_density == .5
            state = generate_map(map_config, seed)
            assert low <= state.layout.width <= high
            assert low <= state.layout.height <= high
            state.validate()
            assert state.player_count == players
            shapes.add((state.layout.height, state.layout.width))
    assert len(shapes) > 15


def test_rating_modes_have_distinct_origins_and_multiplayer_ties():
    assert initial_rating('1v1') == 0 and initial_rating('ffa') == 1000
    assert probability('ffa', 1400, 1000) == pytest.approx(10 / 11)
    assert probability('1v1', 0, 0) == .5
    result = {'placements': synthetic_placements([2, 4, 4, None]), 'entrant_ids': ['random', 'a', 'b', 'c'], 'elimination_ticks': [2, 4, 4, None]}
    changes = score({}, result, {})
    assert changes['a']['after'] == changes['b']['after']
    assert sum(value['after'] for value in changes.values()) == pytest.approx(4000)
    reverse = {'placements': synthetic_placements(list(reversed(result['elimination_ticks']))), 'entrant_ids': list(reversed(result['entrant_ids'])),
               'elimination_ticks': list(reversed(result['elimination_ticks']))}
    assert score({}, reverse, {}) == changes


def test_seat_rotations_keep_map_and_rng_streams(tmp_path):
    store = league(tmp_path, player_counts=(4,))
    for name in ('aggressive','defensive','expansion'):
        add(store, name)
    job = store.jobs()[0]
    plans = make_block(store, job)
    assert len(plans) == 4
    assert len({p['game_id'] for p in plans}) == 4
    assert {p['map_seed'] for p in plans} == {plans[0]['map_seed']}
    streams = dict(zip(plans[0]['instance_ids'], plans[0]['controller_seeds']))
    for plan in plans:
        assert plan['map'] == plans[0]['map']
        assert dict(zip(plan['instance_ids'], plan['controller_seeds'])) == streams
        assert job['entrant_id'] in plan['entrant_ids']
    assert sorted(p['instance_ids'].index(0) for p in plans) == [0,1,2,3]
    assert make_block(store, job) == plans


def test_queue_budget_idempotency_resume_rebuild_and_replay(tmp_path):
    store = league(tmp_path)
    entrant, job_id = add(store, games=7)
    assert add(store, games=1000)[1] == job_id
    assert run_jobs(store, limit=3) == 3
    assert store.jobs()[0]['done'] == 3
    # One rotation remains; resume keeps the already scheduled map and seats.
    with store.connect() as db:
        pending = db.execute("SELECT id FROM games WHERE job_id=? AND status='pending'", (job_id,)).fetchall()
    assert len(pending) == 4
    reopened = CompetitionStore(store.directory)
    assert run_jobs(reopened) == 4
    assert reopened.jobs()[0]['done'] == reopened.jobs()[0]['budget'] == 7
    assert reopened.jobs()[0]['status'] == 'complete'
    assert run_jobs(reopened) == 0
    report = reopened.leaderboard()
    assert report['completed_games'] == 7
    assert next(r for r in report['ranking'] if r['anchor'])['elo'] == 0
    assert reopened.rebuild() == report
    matches = reopened.matches()
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


def test_match_filters_combine_participants_counts_and_pagination(tmp_path):
    store = league(tmp_path, player_counts=(2,3))
    a, b, c = 'entrant-a', 'entrant-b', 'entrant-c'
    fixtures = [[a,b], [a,c], [b,c], [a,a,b], [a,b,c], [c,c]]
    for index, ids in enumerate(fixtures):
        store.insert_plans([{'game_id':str(index),'entrant_ids':ids}], status='live')
        store.abort_live(str(index))
    def keys(**filters):
        return {match['game_id'] for match in store.matches(**filters)['matches']}
    assert keys(entrant=a) == {'0','1','3','4'}
    assert keys(opponent=b) == {'0','2','3','4'}
    assert keys(entrant=a,opponent=b) == {'0','3','4'}
    assert keys(entrant=b,opponent=a,players=2) == {'0'}
    assert keys(entrant=a,opponent=a) == {'3'}
    assert keys(entrant=a,opponent=a,players=2) == set()
    assert keys(entrant="' OR 1=1 --") == set()
    first = store.matches(entrant=a,opponent=b,limit=2)
    second = store.matches(entrant=a,opponent=b,limit=2,offset=2)
    assert first['total'] == second['total'] == 3
    assert len(first['matches']) == 2 and len(second['matches']) == 1
    assert {m['game_id'] for m in first['matches']+second['matches']} == {'0','3','4'}
    server = make_server(store, port=0)
    server.arena.start(run_worker=False)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = f'http://127.0.0.1:{server.server_address[1]}/api/matches'
    try:
        with urlopen(url+f'?entrant={a}&opponent={b}&players=2') as response:
            assert {m['game_id'] for m in json.load(response)['matches']} == {'0'}
        with urlopen(url+f'?entrant={a}&opponent={a}') as response:
            assert json.load(response)['total'] == 1
    finally:
        server.shutdown()
        server.arena.close()
        server.server_close()
        server_thread.join(2)


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
    assert store.leaderboard()['completed_games'] == 1
    corrupted = json.loads(gzip.decompress(blob))
    corrupted['digests'][-1] = 'wrong'
    with pytest.raises(ValueError, match='diverged'):
        replay_frames(gzip.compress(json.dumps(corrupted).encode()), result)


def test_source_change_cannot_silently_change_existing_league(tmp_path):
    store = league(tmp_path)
    with patch('generals_arena.competitions.config.LeagueConfig.manifest', return_value={'rules_id':'changed'}):
        with pytest.raises(ValueError, match='rules/source changed'):
            add(store)


def test_failure_is_unrated_and_retryable_and_workers_exclusive(tmp_path):
    store = league(tmp_path)
    add(store, games=2)
    with process_lock(store.directory, '.worker.lock'):
        with pytest.raises(RuntimeError, match='another process'):
            run_jobs(store)
    with patch('generals_arena.matches.runner.build_controllers', side_effect=RuntimeError('bad policy')):
        with pytest.raises(RuntimeError, match='bad policy'):
            run_jobs(store)
    assert store.jobs()[0]['status'] == 'failed'
    assert store.leaderboard()['completed_games'] == 0
    assert store.matches()['total'] == 0
    assert run_jobs(store, retry=True) == 2
    assert store.jobs()[0]['status'] == 'complete'


def test_unknown_policy_kind_does_not_create_jobs(tmp_path):
    store = league(tmp_path)
    with pytest.raises(ValueError, match='invalid entrant kind'):
        store.add({'name': 'unknown', 'kind': 'unknown'})
    assert not store.jobs()


def test_completion_is_atomic_and_duplicate_safe(tmp_path):
    store = league(tmp_path)
    add(store, games=2)
    plans = make_block(store, store.jobs()[0])
    store.insert_plans(plans, store.jobs()[0]['id'])
    game = RecordedGame(plans[0], store.config, build_controllers(store.entrants(), plans[0]))
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
    store = league(tmp_path, max_ticks=4, min_side=25, max_side=25)
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
        arena.command(game_id, other['entrant']['entrant_id'], {'id':1,'op':'append','action':0})
    with pytest.raises(ValueError, match='in order'):
        arena.command(game_id, human, {'id':2,'op':'append','action':0})
    assert arena.command(game_id, human, {'id':1,'op':'append','action':0})['queue'] == [{'id':1,'action':0}]
    assert arena.view(game_id, human)['board']['tick'] == 0
    # Only the server clock can advance the board.
    arena.thread.start()
    try:
        deadline = time.monotonic() + 3
        while not arena.view(game_id, human)['done'] and time.monotonic() < deadline:
            time.sleep(.02)
        final = arena.view(game_id, human)
        assert final['done'] and not final['error']
        assert store.leaderboard()['completed_games'] == 1
        row = next(r for r in store.leaderboard()['ranking'] if r['entrant_id'] == human)
        assert row['games'] == 1
        assert store.rebuild() == store.leaderboard()
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
    assert store.leaderboard()['completed_games'] == 0
    assert store.matches()['matches'][0]['status'] == 'aborted'
    with pytest.raises(ValueError, match='not found'):
        store.replay(game['game_id'])


def test_human_matchmaking_can_select_non_random_opponents(tmp_path):
    store = league(tmp_path)
    add(store)
    human = store.account('matchmaking')['entrant']['entrant_id']
    job = {'id':'first','entrant_id':human,'done':0,'budget':1}
    # Coverage contributes to ordinary adaptive opponent selection.
    with patch.object(store, 'pair_counts', return_value={store.reference:1}):
        opponents = set()
        for index in range(30):
            plan = make_block(store, {**job, 'id':str(index)}, human=True, players=2)[0]
            opponents.update(key for key in plan['entrant_ids'] if key != human)
    assert len(opponents) == 2


def test_http_frontend_and_privacy_boundaries(tmp_path):
    # Leave enough time for HTTP assertions before submitting the first tick.
    store = league(tmp_path, tick_seconds=.2)
    server = make_server(store, port=0)
    server.arena.start()
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = f'http://127.0.0.1:{server.server_address[1]}'
    from urllib.request import build_opener, ProxyHandler
    direct_open = build_opener(ProxyHandler({})).open
    def request(path, data=None, headers=None):
        request_headers = {'Content-Type':'application/json', **(headers or {})}
        body = None if data is None else json.dumps(data).encode()
        with direct_open(Request(url + path, data=body, headers=request_headers)) as response:
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
        _, raw = request('/api/play/' + game['game_id'] + '/commands',
                         {'id':1, 'op':'append', 'action':0}, auth)
        assert json.loads(raw)['last_command_id'] == 1
        deadline = time.monotonic() + 3
        while not game['done'] and time.monotonic() < deadline:
            time.sleep(.02)
            _, raw = request('/api/play/' + game['game_id'], headers=auth)
            game = json.loads(raw)
        assert game['done'] and not game['error']
        _, raw = request('/api/replay/' + game['game_id'])
        replay = json.loads(raw)
        assert replay['frames'][-1]['tick'] == replay['result']['ticks']
        _, raw = request('/api/leaderboard?mode=1v1')
        leaderboard = json.loads(raw)
        assert leaderboard['completed_games'] == 1
        for row in leaderboard['ranking']:
            assert row['elo_delta'] is None
            assert 0 <= row['win_rate'] <= 1
            assert 0 <= row['survival_index'] <= 1
            assert 0 <= row['kill_index'] <= 1
        assert sum(replay['result']['kills']) == sum(t is not None for t in replay['result']['elimination_ticks'])
        with pytest.raises(HTTPError) as error:
            request('/../league.sqlite3')
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.arena.close()
        server.server_close()
        server_thread.join(2)


def test_duplicate_instances_average_cross_policy_scores_only():
    result = {'placements': synthetic_placements([None,2,1]), 'entrant_ids':['a','a','random'], 'elimination_ticks':[None,2,1]}
    rated = score({'a':1000, 'random':1000}, result, {})
    assert rated['a']['after'] > 1000 > rated['random']['after']
    assert rated['a']['delta'] == pytest.approx(-rated['random']['delta'])
    result['elimination_ticks'] = [None,1,2]
    result['placements'] = synthetic_placements(result['elimination_ticks'])
    assert score({'a':1000,'random':1000}, result, {})['a']['after'] == 1000
    only_self = {'placements': synthetic_placements([1,2,3,4,5,6,7,None]), 'entrant_ids':['a']*8, 'elimination_ticks':[1,2,3,4,5,6,7,None]}
    assert score({'a':137}, only_self, {})['a']['after'] == 137
    permutation = [2,0,1]
    permuted = {key:[result[key][i] for i in permutation] for key in result}
    assert score({'a':1080,'random':1000}, result, {}) == score({'a':1080,'random':1000}, permuted, {})


def test_single_bot_can_fill_eight_independent_seats(tmp_path):
    store = league(tmp_path, player_counts=(8,))
    human = store.account('solo')['entrant']['entrant_id']
    job = {'id':'human','entrant_id':human,'done':0,'budget':1}
    plan = make_block(store, job, human=True)[0]
    assert plan['entrant_ids'].count(store.reference) == 7
    assert plan['entrant_ids'].count(human) == 1
    controllers = build_controllers(store.entrants(), plan)
    bots = [c for c in controllers if c is not None]
    assert len(set(map(id, bots))) == 7
    assert len(set(plan['controller_seeds'])) == 8
    game = RecordedGame(plan, store.config, controllers)
    assert all(c.player_id == seat for seat,c in enumerate(controllers) if c is not None)
    assert not game.done


def test_mixed_player_counts_have_independent_ratings_and_duplicate_accounting(tmp_path):
    store = league(tmp_path, player_counts=(2,3))
    entrant,_ = add(store, games=3)
    target = entrant['entrant_id']
    for index, count in enumerate((2,3)):
        job = {'id':str(index),'entrant_id':target,'done':0,'budget':1}
        plan = make_block(store, job, players=count)[0]
        plan['entrant_ids'] = [target] * (count-1) + [store.reference]
        store.insert_plans([plan])
        game = RecordedGame(plan, store.config, build_controllers(store.entrants(), plan))
        try:
            while not game.done:
                game.advance()
            mode = rating_mode(plan)
            expected = score({}, game.result(), {})
            store.complete(game)
        finally:
            game.engine.close()
        report = store.leaderboard(mode)
        assert report['completed_games'] == 1
        row = next(row for row in report['ranking'] if row['entrant_id'] == target)
        assert row['raw_elo'] == expected[target]['after']
        assert row['games'] == 1 and row['appearances'] == count-1
        assert row['wins']+row['losses']+row['draws'] == count-1
    before = store.leaderboards()
    assert before['ffa']['anchor'] is None and before['ffa']['rating_mean'] == 1000
    assert before['1v1']['rating_model'] == 'egf-gor'
    assert store.rebuild() == before['1v1']
    assert store.leaderboards() == before
    assert store.pair_counts(target, '1v1') == store.pair_counts(target, 'ffa') == {store.reference:1}
    with store.connect() as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='rating_pairs'").fetchone() is None


def test_dual_pool_scores_keep_raw_gor_and_global_ffa_mean(tmp_path):
    store = league(tmp_path, player_counts=(2,3))
    a = store.register({'name':'a','kind':'baseline','policy':'aggressive'})['entrant_id']
    b = store.reference
    c = store.register({'name':'c','kind':'baseline','policy':'expansion'})['entrant_id']
    idle = store.account('idle')['entrant']['entrant_id']

    def complete(key, ids, eliminations):
        plan = dict(game_id=key, entrant_ids=ids)
        store.insert_plans([plan])
        return store.complete_result({'placements': synthetic_placements(eliminations), **plan, 'ticks':2, 'elimination_ticks':eliminations,
                                      'winner_seat':0, 'final_alive':[tick is None for tick in eliminations]}, b'unchanged')

    duel = complete('duel', [a,b], [None,2])
    gor = store.ratings('1v1')
    assert gor[a] > 0 > gor[b]
    assert duel['rating_mode'] == '1v1'
    assert all(row['elo'] == 1000 for row in store.leaderboard('ffa')['ranking'])
    ffa = complete('ffa', [a,b,c], [None,2,1])
    assert ffa['rating_mode'] == 'ffa' and store.ratings('1v1') == gor
    assert sum(store.ratings('ffa').values()) == pytest.approx(3000)
    board = store.leaderboard('1v1')
    rows = {row['entrant_id']:row for row in board['ranking']}
    assert rows[b]['elo'] == 0 and rows[b]['raw_elo'] < 0
    assert rows[idle]['raw_elo'] == 0 and rows[idle]['elo'] == -gor[b]
    assert board['rating_offset'] == gor[b]
    store.register({'name':'new','kind':'baseline','policy':'defensive'})
    ffa_rows = store.leaderboard('ffa')['ranking']
    assert sum(row['elo'] for row in ffa_rows) / len(ffa_rows) == pytest.approx(1000)
    assert store.matches(mode='1v1')['total'] == store.matches(mode='ffa')['total'] == 1
    before = store.leaderboards()
    assert CompetitionStore(store.directory).leaderboards() == before
    store.rebuild()
    assert store.leaderboards() == before


def test_fresh_rating_schema_does_not_silently_read_old_global_pool(tmp_path):
    store = league(tmp_path)
    manifest = {**store.manifest, 'schema':4}
    with store.connect(write=True) as db:
        db.execute("UPDATE meta SET value=? WHERE key='manifest'", (json.dumps(manifest),))
    with pytest.raises(ValueError, match='archived rules'):
        CompetitionStore(store.directory)


def test_parallel_and_sequential_execution_have_identical_ratings(tmp_path):
    import shutil
    store = league(tmp_path, player_counts=(4,))
    add(store, games=7)
    for policy in ('expansion','defensive'):
        add(store, policy, games=1)
    # Copy before execution so job IDs, schedules and algorithm versions match.
    shutil.copytree(store.directory, tmp_path/'parallel')
    other = CompetitionStore(tmp_path/'parallel')
    assert run_jobs(store, limit=5) == run_jobs(other, limit=5, workers=2) == 5
    assert store.leaderboard() == other.leaderboard()
    left = store.matches()['matches']
    right = other.matches()['matches']
    for a,b in zip(left,right):
        assert a['game_id'] == b['game_id'] and a['sequence'] == b['sequence']
        assert a['final_digest'] == b['final_digest']
    assert run_jobs(other, workers=2) == 4
    assert other.leaderboard()['completed_games'] == 9


def test_rating_uses_all_ranks_and_ties_not_just_winner():
    result = {'placements': synthetic_placements([None,30,20,10]), 'entrant_ids':['random','a','b','c'], 'elimination_ticks':[None,30,20,10]}
    changes = score({}, result, {})
    assert ranks(result) == [1,2,3,4]
    assert changes['a']['after'] > 1000 > changes['b']['after'] > changes['c']['after']
    assert changes['a']['rank'] == 2 and changes['a']['expected_rank'] == 2.5
    # The winner stays the same; swapping two losing players swaps their gain.
    other = {**result, 'placements': synthetic_placements([None,20,30,10]), 'elimination_ticks':[None,20,30,10]}
    assert score({},other,{})['a']['after'] == changes['b']['after']
    result['elimination_ticks'] = [None,None,10,10]
    result['placements'] = synthetic_placements(result['elimination_ticks'])
    assert ranks(result) == [1.5,1.5,3.5,3.5]


def test_self_play_does_not_change_ratings_or_age():
    changes = score({'a':300}, {'placements': synthetic_placements(list(range(8))), 'entrant_ids':['a']*8, 'elimination_ticks':list(range(8))}, {'a':2})
    assert changes['a']['after'] == 300 and changes['a']['rated_games'] == 2


def test_gor_probability_calibration_has_zero_expected_raw_rating_drift():
    win = {'placements': synthetic_placements([None,1]), 'entrant_ids':['a','random'],'elimination_ticks':[None,1]}
    lose = {**win, 'placements': synthetic_placements([1,None]),'elimination_ticks':[1,None]}
    for rating in (-800,-500,0,500,2000):
        dw = score({'a':rating}, win, {'a':100})['a']['delta']
        dl = score({'a':rating}, lose, {'a':100})['a']['delta']
        p = probability('1v1',rating,0)
        assert p*dw+(1-p)*dl == pytest.approx(0,abs=1e-9)


def test_human_queue_is_idempotent_ordered_and_clock_driven(tmp_path):
    from generals.core.state import Action, Direction
    store=league(tmp_path, min_side=25,max_side=25,max_ticks=10,tick_seconds=.1)
    human=store.account('queue')['entrant']['entrant_id']
    arena=LiveArena(store);view=arena.create_game(human,2);gid=view['game_id']
    value=arena.games[gid];state=value['game'].state
    source=view['board']['general_pos'];y,x=divmod(source,state.layout.width)
    direction=Direction.RIGHT if x+2<state.layout.width else Direction.LEFT
    target=source+(1 if direction==Direction.RIGHT else -1)
    # Future territory is intentionally not owned at submission time.
    commands=[{'id':1,'op':'append','action':Action.move(source,direction).encode()},
              {'id':2,'op':'append','action':Action.move(target,direction).encode()}]
    for command in commands: view=arena.command(gid,human,command)
    assert len(view['queue'])==2 and view['board']['tick']==0
    assert arena.command(gid,human,commands[1])==view
    assert len(value['queue'])==2
    with pytest.raises(ValueError,match='conflict'):
        arena.command(gid,human,{'id':2,'op':'clear'})
    with pytest.raises(ValueError,match='order'):
        arena.command(gid,human,{'id':4,'op':'clear'})
    # Long polling sleeps without holding the lock; commands wake it immediately.
    received=[]
    listener=threading.Thread(target=lambda:received.append(arena.view(gid,human,after=2)))
    listener.start()
    undone=arena.command(gid,human,{'id':3,'op':'undo'})
    listener.join(1)
    assert received and received[0]['revision']==3 and len(undone['queue'])==1
    assert arena.command(gid,human,{'id':4,'op':'clear'})['queue']==[]
    # Rapid input must never advance the game faster than the authoritative clock.
    for i in range(5,15):arena.command(gid,human,{'id':i,'op':'append','action':0})
    assert state.tick==0 and len(value['queue'])==10
    arena.thread.start()
    try:
        deadline=time.monotonic()+2
        while state.tick<2 and time.monotonic()<deadline:time.sleep(.01)
        with arena.lock:
            assert state.tick==2 and len(value['queue'])==8
    finally:
        arena.stop.set();arena.thread.join(1)


def test_human_fog_merges_cities_and_mountains_without_changing_policy_observations():
    from generals.core.state import GameState, MapLayout, Terrain, Structure, EMPTY, NEUTRAL, UNKNOWN, UNKNOWN_OWNER
    from generals.core.observation import observe
    from generals_arena.web.live import human_board
    terrain=np.zeros((7,7),dtype=np.uint8);terrain[4,4]=Terrain.MOUNTAIN
    structure=np.zeros((7,7),dtype=np.uint8);structure[0,0]=structure[6,6]=Structure.GENERAL
    structure[4,5]=Structure.CITY
    owner=np.full((7,7),EMPTY,dtype=np.int16);owner[0,0]=0;owner[6,6]=1;owner[4,5]=NEUTRAL
    army=np.zeros((7,7),dtype=np.int64);army[0,0]=army[6,6]=1;army[4,5]=40
    state=GameState(MapLayout(terrain),structure,owner,army,np.ones(2,dtype=bool),np.array([0,48],dtype=np.int32))
    state.validate()
    before=observe(state,0)
    view=human_board(state,0)
    assert view['fog_obstacles'][4][4] and view['fog_obstacles'][4][5]
    assert not view['fog_obstacles'][6][6]  # Enemy general cannot be inferred.
    assert not view['fog_obstacles'][3][3]  # Ordinary fog stays empty.
    for y,x in [(4,4),(4,5),(6,6)]:
        assert view['terrain'][y][x]==view['structure'][y][x]==UNKNOWN
        assert view['owner'][y][x]==UNKNOWN_OWNER and view['army'][y][x]==-1
    after=observe(state,0)
    for key in ('terrain','structure','owner','army','visible'):
        np.testing.assert_array_equal(getattr(before,key),getattr(after,key))
    # Moving into vision reveals the real types, and leaving hides them again.
    state.owner[3,4]=0;state.army[3,4]=1
    visible=human_board(state,0)
    assert not visible['fog_obstacles'][4][4] and not visible['fog_obstacles'][4][5]
    assert visible['terrain'][4][4]==Terrain.MOUNTAIN
    assert visible['structure'][4][5]==Structure.CITY and visible['army'][4][5]==40
    state.owner[3,4]=EMPTY;state.army[3,4]=0
    assert human_board(state,0)==view


def test_map_revision_preserves_players_results_scores_and_legacy_manifest(tmp_path):
    store=league(tmp_path)
    add(store,games=2);run_jobs(store)
    human=store.account('keep-my-rating')
    before=store.leaderboard();records=store.matches();previous=store.manifest
    store.resize_maps(15,20,35)
    assert store.config.side_range(8)==(30,35)
    assert store.manifest['rules_id']!=previous['rules_id']
    assert store.authenticate(human['token'])==human['entrant']['entrant_id']
    assert store.leaderboard()==before and store.matches()==records
    with store.connect() as db:
        history=json.loads(db.execute("SELECT value FROM meta WHERE key='rules_history'").fetchone()[0])
    assert history[0]['manifest']==json.loads(json.dumps(previous))
    assert history[0]['next_sequence']==2
    store=CompetitionStore(store.directory);store.verify_execution()
    assert store.rebuild()==before
    for record in records['matches']:
        result,blob=store.replay(record['game_id'])
        assert result['rules_id']==previous['rules_id']
        assert replay_frames(blob,result)[-1]['tick']==result['ticks']
    add(store,'defensive',games=1);run_jobs(store)
    latest=store.matches()['matches'][0]
    assert latest['rules_id']==store.manifest['rules_id']
    assert 15<=latest['map']['width']<=35 and 15<=latest['map']['height']<=35


def test_map_revision_rejects_pending_games_and_non_map_code_changes(tmp_path):
    store=league(tmp_path);add(store,games=2)
    from generals_arena.competitions.jobs import prepare_batch
    prepare_batch(store,store.jobs()[0])
    before=store.manifest
    with pytest.raises(ValueError,match='active and pending'):
        store.resize_maps(15,20,35)
    assert store.manifest==before
    store.manifest={**before,'rating':'different-algorithm'}
    with pytest.raises(ValueError,match='engine or rating'):
        store.resize_maps(15,20,35)
