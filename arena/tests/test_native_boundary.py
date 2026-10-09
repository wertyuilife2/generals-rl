"""Arena delegates simulation to env and versions native source changes."""
from types import SimpleNamespace

from generals.backends.cpu.engine import CPUEngine
from generals_arena.league.config import LeagueConfig
from generals_arena.league.game import RecordedGame
from generals_arena import util


def test_recorded_games_use_shared_native_environment():
    plan = {'entrant_ids': ['a','b'], 'controller_seeds': [1,2], 'map_seed': 3,
            'map': {'height':15, 'width':15, 'players':2}}
    game = RecordedGame(plan, LeagueConfig(max_ticks=2), [None,None])
    try:
        assert isinstance(game.engine, CPUEngine)
        game.advance()
        assert game.engine.env.state['tick'][0] == game.state.tick == 1
    finally:
        game.engine.close()


def test_native_sources_are_part_of_registered_version_identity(tmp_path, monkeypatch):
    (tmp_path/'__init__.py').write_text('')
    source = tmp_path/'policy.cpp'
    source.write_text('first implementation')
    monkeypatch.setattr(util, 'find_spec', lambda _: SimpleNamespace(submodule_search_locations=[str(tmp_path)]))
    before = util.source_digest('policy')
    source.write_text('different implementation')
    assert before != util.source_digest('policy')
