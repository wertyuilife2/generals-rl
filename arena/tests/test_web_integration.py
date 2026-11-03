"""Local neko-dev UI capabilities on the unified competition interfaces."""
import gzip
import json
import threading
import time
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from test_league import league, add
from generals_arena.web.server import make_server
from generals_arena.web.live import LiveArena
from generals_arena.competitions.jobs import run_jobs
from generals_arena.competitions.leaderboard import WinMatrix, DuelStatistics
from generals_arena.competitions.titles import rating_title, title_bands

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


def test_human_selected_opponents_duplicates_random_slots_and_account_isolation(tmp_path):
    store = league(tmp_path, player_counts=(2,4), max_ticks=2)
    bot, _ = add(store)
    alice = store.account('select Alice')['entrant']['entrant_id']
    bob = store.account('select Bob')['entrant']['entrant_id']
    arena = LiveArena(store)
    a = arena.create_game(alice, 4, [bot['entrant_id'], bot['entrant_id'], None])
    assert [e['entrant_id'] for e in a['entrants']].count(bot['entrant_id']) >= 2
    assert len(a['entrants']) == 4
    b = arena.create_game(bob, opponents=[store.reference])
    assert {e['entrant_id'] for e in b['entrants']} == {bob, store.reference}
    assert arena.active(alice)['game_id'] == a['game_id']
    assert arena.active(bob)['game_id'] == b['game_id']
    assert arena.create_game(alice, 2, [store.reference])['game_id'] == a['game_id']
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
    bot = store.register({'name':'retiring','kind':'baseline','policy':'aggressive'})
    store.set_retired(bot['entrant_id'])
    arena = LiveArena(store)
    for key in (human,bot['entrant_id']):
        with pytest.raises(ValueError,match='eligible bot'):
            arena.create_game(human, 2, [key])
    with patch.object(store,'disabled_policy_ids',return_value={store.reference}):
        with pytest.raises(ValueError,match='eligible bot'):
            arena.create_game(human, 2, [store.reference])
    assert store.matches()['total'] == 0


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


def test_pair_matrix_ties_copies_permutations_and_row_orientation():
    model = WinMatrix()
    game = dict(entrant_ids=['a','a','b','c'],elimination_ticks=[None,1,2,None],placements=[1,4,3,1])
    model.add(game)
    model.add(dict(entrant_ids=['a','b','c'],elimination_ticks=[3,2,None],placements=[2,3,1]))
    rows = [{'entrant_id':key} for key in ('c','b','a','unplayed')]
    matrix = model.report(rows)
    assert matrix['cells'][2][1] == dict(games=2,win_rate=.75,draw_rate=0.)
    assert matrix['cells'][1][2] == dict(games=2,win_rate=.25,draw_rate=0.)
    assert matrix['cells'][2][0] == dict(games=2,win_rate=0.,draw_rate=.25)
    assert matrix['cells'][0][2] == dict(games=2,win_rate=.75,draw_rate=.25)
    assert all(matrix['cells'][i][i] is None for i in range(4))
    assert matrix['cells'][3] == [None]*4
    other = WinMatrix()
    other.add(dict(entrant_ids=list(reversed(game['entrant_ids'])),elimination_ticks=list(reversed(game['elimination_ticks'])),placements=list(reversed(game['placements']))))
    single = WinMatrix(); single.add(game)
    assert other.report(rows) == single.report(rows)


def test_titles_use_numeric_boundaries_and_keep_unplayed_players_unrated():
    bands=title_bands()
    assert rating_title(4000,0)['key']=='unrated'
    assert rating_title(4000,5)['remaining_games']==1
    assert rating_title(-500,6)['key']=='newbie'
    for i,band in enumerate(bands[1:],1):
        assert rating_title(band['minimum'],6)==band
        assert rating_title(band['minimum']-.001,100)==bands[i-1]
    assert rating_title(1864.5,500)['key']=='expert'
    assert rating_title(2400,500)['key']=='grandmaster'
    assert rating_title(3000,500)['key']=='legendary-grandmaster'
