"""Execution ownership, uniform timing, recovery and playback boundaries."""
from types import SimpleNamespace
import gzip
import json

import numpy as np

from generals_arena.matches.game import RecordedGame
from generals_arena.matches.replay import replay_frames, verified_replay
from generals_arena.policies.builtin import WaitController
from generals_baselines.registry import create_controller


def plan(players=3):
    return dict(entrant_ids=[str(i) for i in range(players)], controller_seeds=list(range(players)),
                map_seed=91, map=dict(width=15, height=15, players=players))


def test_single_game_and_recovery_never_dispatch_python_rules_or_observations(monkeypatch):
    from generals.backends.cpu.env import CPUEnv
    from generals.core import observation
    def forbidden(*args, **kwargs):
        raise AssertionError('Python game execution was reached')
    monkeypatch.setattr(CPUEnv, 'step', forbidden)
    monkeypatch.setattr(CPUEnv, 'observer', forbidden)
    monkeypatch.setattr(observation, 'observe', forbidden)
    controllers = lambda: [create_controller(kind='random'), WaitController(), WaitController()]
    game = RecordedGame(plan(), SimpleNamespace(max_ticks=8), controllers())
    try:
        game.advance()
        game.surrender(1)
        game.run()
        result, blob = game.result(), game.replay_blob()
        assert [row['seat'] for row in result['decision_timing']] == [0,1,2]
        assert [row['calls'] for row in result['decision_timing']] == [8,1,8]
        assert all(row['limit_ns'] == 10_000_000 for row in result['decision_timing'])
        assert result['elimination_ticks'][1] == 1
        assert len(list(verified_replay(blob, result))) == 9
        trace = json.loads(gzip.decompress(blob))
        from generals import Action
        actions = np.array([[Action(*a).encode() for a in row] for row in trace['actions']], np.int64)
        recovered = RecordedGame(plan(), SimpleNamespace(max_ticks=8), [None]*3, result['forfeits'])
        try:
            recovered.restore(actions)
            assert recovered.summary()['final_digest'] == result['final_digest']
            assert recovered.trace()['frames'] == trace['frames']
        finally:
            recovered.close()
    finally:
        game.close()


def test_playback_only_decodes_recorded_frames(monkeypatch):
    import generals_arena.matches.replay as replay
    game = RecordedGame(plan(2), SimpleNamespace(max_ticks=5), [WaitController(), WaitController()])
    try:
        game.run()
        result, blob = game.result(), game.replay_blob()
        def forbidden(*args, **kwargs):
            raise AssertionError('playback attempted to simulate the game')
        monkeypatch.setattr(replay, '_audit', forbidden)
        assert len(replay_frames(blob, result)) == 6
        assert replay_frames(blob, result, compact=True)[-1]['tick'] == 5
    finally:
        game.close()


def test_legacy_replay_is_converted_with_same_native_frames():
    from generals_arena.matches.replay import encode_replay
    game = RecordedGame(plan(2), SimpleNamespace(max_ticks=5), [WaitController(), WaitController()])
    try:
        game.run()
        trace, result = game.trace(), game.result()
        legacy = encode_replay(trace['initial'], trace['actions'], trace['digests'])
        assert replay_frames(legacy, result) == replay_frames(game.replay_blob(), result)
    finally:
        game.close()


def test_baseline_and_wait_both_supply_leaderboard_decision_time():
    from generals_arena.competitions.leaderboard import DuelStatistics
    game = RecordedGame(plan(2), SimpleNamespace(max_ticks=4),
                        [create_controller(kind='random'), WaitController()])
    try:
        game.run()
        result = game.result()
        assert [row['calls'] for row in result['decision_timing']] == [4,4]
        statistics = DuelStatistics()
        statistics.add(result)
        rows = [{'entrant_id':key} for key in result['entrant_ids']]
        statistics.annotate(rows)
        assert all(row['timed_games'] == 1 and row['mean_decision_ms'] is not None for row in rows)
    finally:
        game.close()


def test_browser_seeks_across_surrender_keyframes():
    from pathlib import Path
    import shutil
    import subprocess
    import pytest
    if not shutil.which('node'):
        pytest.skip('Node is required to exercise the browser replay module')
    game = RecordedGame(plan(), SimpleNamespace(max_ticks=5),
                        [WaitController(),WaitController(),WaitController()])
    try:
        game.advance()
        game.surrender(1)
        game.run()
        result, blob = game.result(), game.replay_blob()
        frames = replay_frames(blob, result, compact=True)
        full = replay_frames(blob, result)
        module = Path(__file__).parents[1]/'src/generals_arena/web/replay.js'
        script = f'''import {{ReplayTimeline}} from {json.dumps(module.as_uri())};
            let input='';for await (const chunk of process.stdin) input+=chunk;
            const frames=JSON.parse(input), timeline=new ReplayTimeline(frames,{{interval:2,capacity:2}});
            process.stdout.write(JSON.stringify([5,0,1,4,2,5].map(index=>structuredClone(timeline.seek(index)))));'''
        output = subprocess.run(['node','--input-type=module','-e',script],input=json.dumps(frames),
                                text=True,capture_output=True,check=True)
        actual = json.loads(output.stdout)
        assert actual == [full[i] for i in [5,0,1,4,2,5]]
    finally:
        game.close()
