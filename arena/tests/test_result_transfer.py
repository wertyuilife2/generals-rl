"""Results cross the native boundary as owned data, preserving public semantics."""
from types import SimpleNamespace

import pytest

from generals_arena.matches.game import RecordedGame
from generals_arena.policies.builtin import WaitController
from test_native_session import plan


@pytest.mark.parametrize('players', [2, 8])
def test_result_transfer_bypasses_json_and_preserves_previous_snapshots(players, monkeypatch):
    settings = plan(players)
    settings['map'].update(width=25, height=25)
    game = RecordedGame(settings, SimpleNamespace(max_ticks=3),
                        [WaitController() for _ in range(players)])
    try:
        def forbidden(*args, **kwargs):
            raise AssertionError('result used a JSON query')
        with monkeypatch.context() as patch:
            patch.setattr(game.execution, 'query_bytes', forbidden)
            initial = game.execution.query(0)
            game.advance()
            game.surrender(1)
            game.run()
            final = game.execution.query(0)
        assert initial['ticks'] == 0 and initial['winner_seat'] is None
        assert initial['final_alive'] == [True]*players
        assert initial['elimination_ticks'] == [None]*players
        assert final['initial_digest'] == initial['final_digest'] == initial['initial_digest']
        assert final['final_digest'] != initial['final_digest']
        assert final['final_alive'] == [i != 1 for i in range(players)]
        assert final['elimination_ticks'][1] == 1
        assert final['kills'] == game.state.kills.tolist()
        assert all(type(v) is int for v in final['placements'])
        assert all(len(v) == players for v in final['final_stats'].values())
        if players == 2:
            assert final['winner_seat'] == 0 and final['placements'] == [1, 2]
            assert initial['placements'] == [1.5, 1.5]
        else:
            assert final['winner_seat'] is None and final['reason'] == 'tick_limit'
        with pytest.raises(RuntimeError, match='invalid result'):
            game.execution.query(0, lane=1)
    finally:
        game.close()
    assert initial['final_alive'] == [True]*players
