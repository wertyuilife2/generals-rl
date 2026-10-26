"""Arena delegates simulation to env and versions native source changes."""
from types import SimpleNamespace
from pathlib import Path

import pytest

from generals.backends.cpu.engine import CPUEngine
from generals_arena.competitions.config import LeagueConfig
from generals_arena.matches.game import RecordedGame
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


@pytest.mark.parametrize('filename', ['native.py', 'worker.py'])
def test_native_runtime_changes_invalidate_registered_execution(tmp_path, monkeypatch, filename):
    from generals_arena.policies import native
    from generals_arena.policies.artifacts import snapshot_policy
    from generals_arena.policies.entries import prepare_entrant, verify_artifacts
    source = tmp_path/'policy.cpp'
    source.write_bytes((Path(__file__).resolve().parents[1]/'examples'/'cpp_wait'/'policy.cpp').read_bytes())
    spec = snapshot_policy(tmp_path/'archive', source)
    entry = prepare_entrant(spec)
    original = native.file_digest
    monkeypatch.setattr(native, 'file_digest', lambda path: '0'*64 if Path(path).name == filename else original(path))
    changed = prepare_entrant(spec)
    assert changed['version'] == entry['version']
    assert changed['entrant_id'] != entry['entrant_id']
    with pytest.raises(ValueError, match='native host changed'):
        verify_artifacts(entry)


def test_only_explicit_decision_phase_is_a_decision_timeout():
    from generals_arena.policies.native import decision_timeout
    assert decision_timeout({'kind': 'timeout', 'phase': 'decision'})
    assert not decision_timeout({'kind': 'timeout', 'limit_ns': 10_000_000})
    assert not decision_timeout({'kind': 'timeout', 'phase': 'initialization'})
    assert not decision_timeout({'kind': 'policy_error', 'phase': 'decision'})


def test_native_match_does_not_charge_uncapped_package_against_cpp_wall_budget(tmp_path, monkeypatch):
    from dataclasses import asdict
    from generals import MapConfig
    from generals_arena.matches.runner import MatchRunner
    from generals_arena.policies.artifacts import snapshot_policy
    from generals_arena.policies.entries import prepare_entrant
    import policy_test_utils

    monkeypatch.setattr(policy_test_utils, 'SOURCE', policy_test_utils.SOURCE.replace(
        '    def act(self, obs):', '    def act(self, obs):\n        import time\n        time.sleep(.6)'))
    bundle = policy_test_utils.make_bundle(tmp_path / 'slow-package')
    package = prepare_entrant({'name': 'slow-package', 'kind': 'package', 'bundle': str(bundle)})
    native = prepare_entrant(snapshot_policy(tmp_path / 'archive',
        Path(__file__).resolve().parents[1] / 'examples/cpp_wait/policy.cpp'))
    entries = {entry['entrant_id']: entry for entry in (native, package)}
    plan = dict(game_id='mixed', entrant_ids=list(entries), instance_ids=[0, 1],
                controller_seeds=[2, 3], map_seed=42, map=asdict(MapConfig(9, 9, 2, .1, .05)))
    # Twenty .6s decisions exceed the former 11.2s match budget.
    with MatchRunner(entries, max_ticks=20) as runner:
        results, _ = runner.run([plan])
    assert results[0]['ticks'] == 20 and results[0]['reason'] == 'tick_limit'
    assert 'policy_error' not in results[0]
