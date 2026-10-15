"""League persistence, rating calibration, scheduling, replays and human privacy."""
import json
import math
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
from generals_arena.league.rating import expected, update, ranks, rate, learning_rate
from generals_arena.league.scheduler import make_block
from generals_arena.league.service import LiveArena, controllers_for, eligible_schedule, process_lock, run_jobs
from generals_arena.league.store import LeagueStore
from generals_arena.server import make_server


def config(**kwargs):
    return LeagueConfig(**{**dict(player_counts=(2, 4), map_generation=None, min_side=3, max_side=5, max_ticks=8,
                                 mountain_min=0, mountain_max=.1, city_min=0, city_max=0,
                                 tick_seconds=.05), **kwargs})


def league(tmp_path, **kwargs):
    return LeagueStore.create(tmp_path / 'league', config(**kwargs), reference='random')


def add(store, policy='aggressive', games=6):
    return store.add({'name': policy, 'kind': 'baseline', 'policy': policy}, games=games)


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
            assert 15 <= low <= map_config.width <= high <= 35
            assert low <= map_config.height <= high
            assert map_config.generation == rules.map_generation
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
    assert rated['c'] == pytest.approx(500 / math.log(9))
    assert rated['a'] == rated['b']
    assert rated['a'] == 0  # One win, one tie, one loss at equal pre-match ratings.
    reverse = {'entrant_ids': list(reversed(result['entrant_ids'])), 'elimination_ticks': list(reversed(result['elimination_ticks']))}
    assert update(dict.fromkeys(result['entrant_ids'], 0), reverse, 'random') == rated
    lose = {'entrant_ids': ['random', 'a'], 'elimination_ticks': [None, 10]}
    assert update({'random': 0, 'a': 0}, lose, 'random')['a'] == pytest.approx(-500 / math.log(9))
    timeout = {'entrant_ids': ['random', 'a', 'b'], 'elimination_ticks': [10, None, None]}
    assert update({}, timeout, 'random')['a'] == pytest.approx(250 / math.log(9))


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
    pending = store.pending(job_id)
    assert pending is not None
    reopened = LeagueStore(store.directory)
    assert run_jobs(reopened) == 4
    assert reopened.jobs()[0]['done'] == reopened.jobs()[0]['budget'] == 7
    assert reopened.jobs()[0]['status'] == 'complete'
    assert run_jobs(reopened) == 0
    report = reopened.leaderboard()
    assert sum(board['completed_games'] for board in reopened.leaderboards().values()) == 7
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
    assert keys(include=[a,b]) == {'0','3','4'}
    assert keys(include=[a,a,b], exclude=[c]) == {'0','3'}
    assert keys(include=[a], exclude=[a]) == set()
    assert keys(exclude=[a,b]) == {'5'}
    assert keys(mode='1v1', include=[a]) == {'0','1'}
    assert keys(mode='multiplayer', include=[a,b]) == {'3','4'}
    assert keys(include=["' OR 1=1 --"]) == set()
    filtered = store.matches(include=[a,b], exclude=[c], limit=1, offset=1)
    assert filtered['total'] == 2 and len(filtered['matches']) == 1
    for invalid in ({'mode':'duel'}, {'include':a}, {'exclude':[None]}):
        with pytest.raises(ValueError):
            store.matches(**invalid)
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
        with urlopen(url+f'?include={a}&include={b}&exclude={c}&mode=multiplayer') as response:
            data = json.load(response)
            assert data['total'] == 1 and data['matches'][0]['game_id'] == '3'
        with pytest.raises(HTTPError) as error:
            urlopen(url+'?mode=invalid')
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.arena.close()
        server.server_close()
        server_thread.join(2)


def test_history_index_tracks_completed_and_unsequenced_games(tmp_path):
    store = league(tmp_path)
    with store.connect(write=True) as db:
        # Existing leagues acquire the index when reopened.
        db.execute('DROP INDEX games_history')
        for name, status, sequence, finished in (
                ('complete', 'complete', 0, 1), ('pending', 'pending', 1, None),
                ('failed', 'failed', None, 3), ('aborted', 'aborted', None, 2)):
            plan = json.dumps({'game_id': name, 'entrant_ids': [store.anchor, store.anchor]})
            db.execute('INSERT INTO games(id,status,plan,sequence,created,finished) VALUES(?,?,?,?,0,?)',
                       (name, status, plan, sequence, finished))
    store = LeagueStore(store.directory)
    assert [m['game_id'] for m in store.matches()['matches']] == ['complete', 'failed', 'aborted']
    assert store.matches(entrant=store.anchor, opponent=store.anchor)['total'] == 3
    with store.connect(write=True) as db:
        db.execute("UPDATE games SET status='complete',finished=4 WHERE id='pending'")
    assert store.matches(limit=1)['matches'][0]['game_id'] == 'pending'
    assert store.matches(offset=1, limit=1)['matches'][0]['game_id'] == 'complete'
    assert store.matches(mode='1v1')['total'] == 4
    assert store.matches(mode='ffa')['total'] == 0


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


def test_retirement_hides_and_unschedules_without_deleting_history(tmp_path):
    store = league(tmp_path, player_counts=(2,))
    old, _ = add(store, 'aggressive', games=2)
    run_jobs(store)
    newcomer, _ = add(store, 'defensive', games=1)
    run_jobs(store)
    before_matches = store.matches(limit=100)
    before_rating = store.ratings()[old['entrant_id']]

    retired = store.set_retired('@aggressive', reason='superseded')
    assert retired['entrant_id'] == old['entrant_id'] and retired['retired']
    assert old['entrant_id'] not in {r['entrant_id'] for r in store.leaderboard()['ranking']}
    archived = store.leaderboard(include_retired=True)
    assert next(r for r in archived['ranking'] if r['entrant_id'] == old['entrant_id'])['retired']
    assert store.matches(limit=100) == before_matches
    assert store.ratings()[old['entrant_id']] == before_rating

    job = {'id': 'next', 'entrant_id': newcomer['entrant_id'], 'done': 0, 'budget': 20}
    assert all(old['entrant_id'] not in plan['entrant_ids'] for plan in eligible_schedule(store, job))
    with pytest.raises(ValueError, match='retired'):
        eligible_schedule(store, {**job, 'entrant_id': old['entrant_id']})
    with pytest.raises(ValueError, match='anchor'):
        store.set_retired(store.anchor)

    restored = store.set_retired(old['entrant_id'], False, reason='needed again')
    assert restored['retired'] is False
    assert old['entrant_id'] in {r['entrant_id'] for r in store.leaderboard()['ranking']}


def test_http_retirement_filters_history_metadata_and_cached_replays(tmp_path):
    store = league(tmp_path, player_counts=(2,))
    old, _ = add(store, games=2)
    run_jobs(store)
    before = store.matches()
    game_id = before['matches'][0]['game_id']
    server = make_server(store, port=0)
    server.arena.start(run_worker=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_address[1]}'
    def get(path):
        with urlopen(url + path) as response:
            return json.load(response)
    try:
        assert get('/api/replay/' + game_id)['frames']  # Prime the response cache.
        with urlopen(Request(url + '/api/replay/' + game_id, headers={'Accept-Encoding': 'gzip'})) as response:
            assert response.headers['Content-Encoding'] == 'gzip'
            assert json.loads(gzip.decompress(response.read())) == get('/api/replay/' + game_id)
        store.set_retired(old['entrant_id'])
        assert store.matches() == before  # The audit history is preserved.
        assert store.matches(include_retired=False)['total'] == 0
        assert not store.match_is_visible(game_id)
        assert get('/api/matches?include=' + old['entrant_id'])['total'] == 0
        meta = get('/api/meta')
        assert old['entrant_id'] not in {e['entrant_id'] for e in meta['entrants']}
        assert all(j['entrant_id'] != old['entrant_id'] for j in meta['jobs'])
        with pytest.raises(HTTPError) as error:
            get('/api/replay/' + game_id)
        assert error.value.code == 404
        store.set_retired(old['entrant_id'], False)
        assert get('/api/matches')['total'] == 2
        assert get('/api/replay/' + game_id)['frames']
    finally:
        server.shutdown()
        server.arena.close()
        server.server_close()
        thread.join(2)


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
    assert store.leaderboard()['completed_games'] == 0
    assert store.matches()['total'] == 0
    assert run_jobs(store, retry=True) == 2
    assert store.jobs()[0]['status'] == 'complete'


def test_unattributed_cpp_infrastructure_failure_is_aborted_and_retryable(tmp_path):
    store = league(tmp_path)
    with store.connect(write=True) as db:
        db.execute("INSERT INTO entrants VALUES (?,?,?,?)", (
            'cpp', 'cpp', 'cpp', json.dumps({'entrant_id':'cpp', 'name':'cpp', 'kind':'cpp'})))
        db.execute("INSERT INTO ratings VALUES (?,?,?,?,?,?,?,?,?)", ('1v1', 'cpp', 0, 0, 0, 0, 0, 0, 0))
        db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
                   ('job', 'cpp', 1, 0, 'failed', 'host cache vanished', time.time()))
    store.insert_plans([{'game_id':'game'}], 'job')
    store.fail_game({'game_id':'game', 'policy_error':'host cache vanished',
                     'error_detail':{}}, b'audit replay')
    store.retry('job')
    assert next(j for j in store.jobs() if j['id'] == 'job')['status'] == 'queued'
    with store.connect() as db:
        row = db.execute("SELECT status,result,replay FROM games WHERE id='game'").fetchone()
    assert row['status'] == 'aborted'
    assert json.loads(row['result'])['policy_error'] == 'host cache vanished'
    assert row['replay'] == b'audit replay'
    with store.connect(write=True) as db:
        db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
                   ('policy-job', 'cpp', 1, 0, 'failed', 'policy crashed', time.time()))
    store.insert_plans([{'game_id':'policy-game'}], 'policy-job')
    store.fail_game({'game_id':'policy-game', 'policy_error':'policy crashed',
                     'error_detail':{'player':0}}, b'policy replay')
    store.retry('policy-job')
    assert next(j for j in store.jobs() if j['id'] == 'policy-job')['status'] == 'failed'
    with store.connect() as db:
        assert db.execute("SELECT status FROM games WHERE id='policy-game'").fetchone()['status'] == 'failed'


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
    # After the first anchor calibration game, normal adaptive sampling applies.
    with patch.object(store, 'pair_counts', return_value={store.anchor:1}):
        opponents = set()
        for index in range(30):
            plan = make_block(store, {**job, 'id':str(index)}, human=True, players=2)[0]
            opponents.update(key for key in plan['entrant_ids'] if key != human)
    assert len(opponents) == 2


def test_human_selected_opponents_duplicates_random_slots_and_account_isolation(tmp_path):
    store = league(tmp_path, max_ticks=2)
    bot, _ = add(store)
    alice = store.account('select Alice')['entrant']['entrant_id']
    bob = store.account('select Bob')['entrant']['entrant_id']
    arena = LiveArena(store)
    a = arena.create_game(alice, 4, [bot['entrant_id'], bot['entrant_id'], None])
    assert [e['entrant_id'] for e in a['entrants']].count(bot['entrant_id']) >= 2
    assert len(a['entrants']) == 4
    b = arena.create_game(bob, opponents=[store.anchor])
    assert {e['entrant_id'] for e in b['entrants']} == {bob, store.anchor}
    assert arena.active(alice)['game_id'] == a['game_id']
    assert arena.active(bob)['game_id'] == b['game_id']
    assert arena.create_game(alice, 2, [store.anchor])['game_id'] == a['game_id']
    with pytest.raises(PermissionError):
        arena.command(a['game_id'], bob, {'id':1,'op':'append','action':0})
    arena.thread.start()
    try:
        deadline = time.monotonic()+3
        while any(not arena.active(key)['done'] for key in (alice,bob)) and time.monotonic()<deadline:
            time.sleep(.02)
        assert all(arena.active(key)['done'] and not arena.active(key)['error'] for key in (alice,bob))
        assert store.leaderboard('1v1')['completed_games'] == 1
        assert store.leaderboard('ffa')['completed_games'] == 1
    finally:
        arena.close()


@pytest.mark.parametrize('players,opponents', [
    (2,[]), (2,'random'), (2,[None,None]), (2,[{}]), (2,['missing']),
    (9,[None]*8), (3,[None]*2),
])
def test_invalid_human_opponents_do_not_create_games(tmp_path, players, opponents):
    store = league(tmp_path)
    human = store.account('validation')['entrant']['entrant_id']
    with pytest.raises(ValueError):
        LiveArena(store).create_game(human, players, opponents)
    assert store.matches()['total'] == 0


def test_human_opponents_exclude_humans_retired_and_disqualified_bots(tmp_path):
    store = league(tmp_path)
    human = store.account('validation')['entrant']['entrant_id']
    bot, _ = add(store)
    store.set_retired(bot['entrant_id'])
    arena = LiveArena(store)
    for key in (human,bot['entrant_id']):
        with pytest.raises(ValueError,match='eligible bot'):
            arena.create_game(human, 2, [key])
    with patch.object(store,'disabled_policy_ids',return_value={store.anchor}):
        with pytest.raises(ValueError,match='eligible bot'):
            arena.create_game(human, 2, [store.anchor])
    assert store.matches()['total'] == 0


def test_http_frontend_and_privacy_boundaries(tmp_path):
    # Leave enough time for HTTP assertions before submitting the first tick.
    store = league(tmp_path, tick_seconds=.2)
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
        response, javascript = request('/history.js')
        assert b'export function historyQuery' in javascript
        assert 'javascript' in response.headers['Content-Type']
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
        _, raw = request('/api/play', {'players':2,'opponents':[store.anchor]}, {'Authorization':f'Bearer {token}'})
        game = json.loads(raw)
        assert store.anchor in [entry['entrant_id'] for entry in game['entrants']]
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
    result = {'entrant_ids':['a','a','random'], 'elimination_ticks':[None,2,1]}
    rated = update({'a':0, 'random':0}, result, 'random')
    assert rated == pytest.approx({'a':500 / math.log(9), 'random':0})
    # A winning copy and a losing copy average to a draw against the reference.
    result['elimination_ticks'] = [None,1,2]
    assert update({'a':0,'random':0}, result, 'random')['a'] == 0
    only_self = {'entrant_ids':['a']*8, 'elimination_ticks':[1,2,3,4,5,6,7,None]}
    assert update({'a':137}, only_self, 'random')['a'] == 137
    permutation = [2,0,1]
    permuted = {key:[result[key][i] for i in permutation] for key in result}
    assert update({'a':80,'random':0}, result, 'random') == update({'a':80,'random':0}, permuted, 'random')


def test_single_bot_can_fill_eight_independent_seats(tmp_path):
    store = league(tmp_path, player_counts=(8,))
    human = store.account('solo')['entrant']['entrant_id']
    job = {'id':'human','entrant_id':human,'done':0,'budget':1}
    plan = make_block(store, job, human=True)[0]
    assert plan['entrant_ids'].count(store.anchor) == 7
    assert plan['entrant_ids'].count(human) == 1
    controllers = controllers_for(store, plan)
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
        plan['entrant_ids'] = [target] * (count-1) + [store.anchor]
        store.insert_plans([plan])
        game = RecordedGame(plan, store.config, controllers_for(store, plan))
        while not game.done:
            game.advance()
        expected_ratings = update({}, game.result(), store.anchor)
        store.complete(game)
        report = store.leaderboard('1v1' if count == 2 else 'ffa')
        assert report['completed_games'] == 1
        row = next(row for row in report['ranking'] if row['entrant_id'] == target)
        assert row['elo'] == expected_ratings[target]
        assert row['games'] == 1 and row['appearances'] == count-1
        assert row['wins']+row['losses']+row['draws'] == count-1
    before = store.leaderboards()
    store.rebuild()
    assert store.leaderboards() == before


def test_parallel_and_sequential_execution_have_identical_ratings(tmp_path):
    import shutil
    store = league(tmp_path, player_counts=(4,))
    add(store, games=7)
    for policy in ('expansion','defensive'):
        add(store, policy, games=1)
    # Copy before execution so job IDs, schedules and algorithm versions match.
    shutil.copytree(store.directory, tmp_path/'parallel')
    other = LeagueStore(tmp_path/'parallel')
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
    result = {'entrant_ids':['random','a','b','c'], 'elimination_ticks':[None,30,20,10]}
    changes = rate({}, result, 'random')
    assert ranks(result) == [1,2,3,4]
    assert changes['a']['after'] > 0 > changes['b']['after'] > changes['c']['after']
    assert changes['a']['rank'] == 2 and changes['a']['expected_rank'] == 2.5
    # The winner stays the same; swapping two losing players swaps their gain.
    other = {**result, 'elimination_ticks':[None,20,30,10]}
    assert rate({},other,'random')['a']['after'] == changes['b']['after']
    result['elimination_ticks'] = [None,None,10,10]
    assert ranks(result) == [1.5,1.5,3.5,3.5]


def test_fast_provisional_convergence_and_no_self_play_aging():
    result = {'entrant_ids':['random','a'], 'elimination_ticks':[1,None]}
    ratings = {}
    old_elo = 0
    for game in range(6):
        ratings = update(ratings,result,'random',games={'a':game})
        old_elo += 32 * (1-expected(old_elo,0))
    assert ratings['a'] > 500 and old_elo < 100
    assert learning_rate(0) > learning_rate(5) > learning_rate(30) > learning_rate(300)
    changes=rate({'a':300},{'entrant_ids':['a']*8,'elimination_ticks':list(range(8))},'random',games={'a':2})
    assert changes['a']['after'] == 300 and changes['a']['rated_games'] == 2


def test_probability_calibration_has_zero_expected_rating_drift():
    for rating in (-1000,-500,0,500,1000):
        for count in (2,4,8):
            ids=['a']+['random']*(count-1)
            win={'entrant_ids':ids,'elimination_ticks':[None]+[1]*(count-1)}
            lose={'entrant_ids':ids,'elimination_ticks':[1]+[None]*(count-1)}
            dw=rate({'a':rating},win,'random',games={'a':100})['a']['delta']
            dl=rate({'a':rating},lose,'random',games={'a':100})['a']['delta']
            assert expected(rating,0)*dw+(1-expected(rating,0))*dl == pytest.approx(0,abs=1e-9)


def test_human_queue_is_idempotent_ordered_and_clock_driven(tmp_path):
    from generals.core.state import Action, Direction
    store=league(tmp_path, min_side=8,max_side=8,max_ticks=10,tick_seconds=.1)
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
    from generals_arena.league.views import human_board
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
    store=LeagueStore(store.directory);store.verify_execution()
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
    from generals_arena.league.service import prepare_batch
    prepare_batch(store,store.jobs()[0])
    before=store.manifest
    with pytest.raises(ValueError,match='active and pending'):
        store.resize_maps(15,20,35)
    assert store.manifest==before
    store.manifest={**before,'rating':'different-algorithm'}
    with pytest.raises(ValueError,match='engine or rating'):
        store.resize_maps(15,20,35)


def test_configured_maps_upgrade_preserves_results_and_replays(tmp_path):
    from generals import MapGenerationConfig
    store = league(tmp_path)
    add(store, games=2)
    run_jobs(store)
    before, records, previous = store.leaderboard(), store.matches(), store.manifest
    # Old manifests must retain their old distribution until explicitly upgraded.
    legacy = json.loads(json.dumps(previous))
    legacy['config'].pop('map_generation')
    with store.connect(write=True) as db:
        db.execute("UPDATE meta SET value=? WHERE key='manifest'", (json.dumps(legacy),))
    store = LeagueStore(store.directory)
    assert store.config.map_generation is None
    options = MapGenerationConfig(city_army_min=50, city_army_max=50)
    store.configure_maps(options)
    store = LeagueStore(store.directory)
    store.verify_execution()
    assert store.config.map_generation == options
    assert store.config.side_range(2) == (18, 23)
    assert store.leaderboard() == before
    for record in records['matches']:
        result, blob = store.replay(record['game_id'])
        assert result['rules_id'] == previous['rules_id']
        assert replay_frames(blob, result)[-1]['tick'] == result['ticks']
    add(store, 'defensive', games=1)
    run_jobs(store)
    result, blob = store.replay(store.matches()['matches'][0]['game_id'])
    assert result['map']['generation']['city_army_min'] == 50
    assert result['rules_id'] == store.manifest['rules_id']
    assert replay_frames(blob, result)[-1]['tick'] == result['ticks']


def test_configured_maps_cannot_change_an_unfinished_evaluation(tmp_path):
    store = league(tmp_path)
    add(store, games=2)
    previous = store.manifest
    with pytest.raises(ValueError, match='queued evaluations'):
        store.configure_maps()
    assert store.manifest == previous
    store.manifest = {**previous, 'rating': 'different-algorithm'}
    with pytest.raises(ValueError, match='simulation or rating'):
        store.configure_maps()
