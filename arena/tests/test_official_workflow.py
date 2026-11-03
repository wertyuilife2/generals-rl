"""Current random geometry through persistent Arena, batch execution and replay."""
import json
import numpy as np
import pytest
from generals import MapConfig
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.store import CompetitionStore
from generals_arena.competitions.jobs import run_jobs
from generals_arena.competitions.tournament import tournament
from generals_arena.matches.config import MatchConfig
from generals_arena.matches.replay import replay_frames


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('tick', [0, 2])
def test_human_surrender_records_and_replays(tmp_path, monkeypatch, native, tick):
    from generals_arena.competitions.schedules import candidate_panel
    from generals_arena.policies.artifacts import snapshot_policy
    from generals_arena.web import live as service
    import time

    store = CompetitionStore.create(tmp_path/'league', LeagueConfig(player_counts=(2,), tick_seconds=.05))
    if native:
        source = tmp_path/'source'; source.mkdir()
        (source/'policy.cpp').write_text('''#include "arena_policy.h"
extern "C" void *arena_create(int, std::uint64_t, const char *) { return new int(0); }
extern "C" std::int64_t arena_act(void *, const ArenaObservation *) { return 0; }
extern "C" void arena_destroy(void *p) { delete static_cast<int *>(p); }
''')
        entrant = store.register(snapshot_policy(store.directory, source))['entrant_id']
    else:
        entrant = next(iter(store.entrants()))
    human = store.account('human')['entrant']['entrant_id']
    plan = candidate_panel(store.config, human, [entrant], 1, 8, 'surrender')[0]
    plan['human'] = True
    monkeypatch.setattr(service, 'eligible_schedule', lambda *a, **kw: [plan])
    live = service.LiveArena(store)
    live.start(run_worker=False)
    try:
        with live.lock:
            view = live.create_game(human, 2); gid = view['game_id']
            game = live.games[gid]['game']; seat = live.games[gid]['seat']
            for _ in range(tick): game.advance()
            command = {'id': 1, 'op': 'surrender'}
            view = live.command(gid, human, command)
            assert live.command(gid, human, command) == view
        deadline = time.monotonic() + 5
        while not view['done'] and time.monotonic() < deadline:
            time.sleep(.02); view = live.view(gid, human)
        assert view['done'] and view['error'] is None
        result, blob = store.replay(gid)
        frames = replay_frames(blob, result)
        assert result['ticks'] == max(1, tick)
        assert result['reason'] == 'surrender'
        assert result['winner_seat'] == 1-seat and result['placements'][seat] == 2
        assert len(result['forfeits']) == 1 and frames[-1]['forfeited'] == [seat]
        assert len(frames) == result['ticks'] + 1
    finally:
        live.close()


def test_invalid_human_queue_entries_are_drained_in_one_tick(tmp_path):
    from generals import Action, Direction
    from generals_arena.web.live import LiveArena
    import time

    store = CompetitionStore.create(tmp_path/'league', LeagueConfig(player_counts=(2,), max_ticks=1, tick_seconds=.05))
    human = store.account('queue')['entrant']['entrant_id']
    live = LiveArena(store); live.start(run_worker=False)
    try:
        with live.lock:
            view = live.create_game(human, 2); gid = view['game_id']
            source = view['board']['general_pos']  # One troop: all moves currently invalid.
            for i in range(3):
                view = live.command(gid, human, {'id': i+1, 'op': 'append',
                                   'action': Action.move(source, Direction.RIGHT).encode()})
            live.command(gid, human, {'id': 4, 'op': 'append', 'action': 0})
        deadline = time.monotonic() + 3
        while not view['done'] and time.monotonic() < deadline:
            time.sleep(.02); view = live.view(gid, human)
        assert view['done'] and view['error'] is None and view['queue'] == []
        assert view['board']['tick'] == 1
    finally:
        live.close()


def test_live_afk_cached_winner_scores_as_a_win():
    from generals_arena.matches.results import seat_outcome
    roster = [{'name': name, 'kind': 'custom', 'version': 'test-wait-v1',
               'factory': 'generals_arena.policies.builtin:create_wait_controller'} for name in ('wait1', 'wait2')]
    report = tournament(roster, MatchConfig(MapConfig(players=2), max_ticks=2000), maps_per_group=1, seed=42)
    for result in report['results']:
        assert result['reason'] == 'afk' and result['ticks'] == 71
        assert result['final_alive'] == [False, False] and result['winner_seat'] == 1
        assert result['placements'] == [2, 1]
        assert seat_outcome(result, 1) == 'win' and seat_outcome(result, 0) == 'loss'


def test_random_maps_batch_scalar_and_gpu_results_agree():
    import pytest
    torch=pytest.importorskip('torch')
    roster=[{'name':n,'kind':'baseline','policy':n} for n in ('aggressive','expansion')]
    cfg=MatchConfig(MapConfig(players=2),max_ticks=55)
    results=[]
    for options in ({'scalar':True},{'batch_size':0},*([{'backend':'gpu','device':'cuda:0','batch_size':0}] if torch.cuda.is_available() else [])):
        report=tournament(roster,cfg,maps_per_group=12,seed=3,**options)
        results.append([(r['map_seed'],r['initial_digest'],r['final_digest'],r['placements']) for r in report['results']])
    assert results[0]==results[1]
    if len(results)>2:assert results[0]==results[2]


def test_default_map_distribution_persists_and_replays(tmp_path):
    store=CompetitionStore.create(tmp_path/'league',LeagueConfig(player_counts=(2,),max_ticks=4))
    _,job=store.add({'name':'expansion','kind':'baseline','policy':'expansion'},games=6)
    assert run_jobs(store,job_id=job)==6
    for row in store.matches()['matches']:
        result,blob=store.replay(row['game_id'])
        frames=replay_frames(blob,result)
        assert len(frames)==5
        h,w=np.shape(frames[0]['army'])
        assert 18<=h<=23 and 18<=w<=23
    before=store.leaderboard('1v1');store.rebuild();assert store.leaderboard('1v1')==before
