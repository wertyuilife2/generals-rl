"""Default opponents retain source parameters and immutable bytes across runs."""
import json
from pathlib import Path

import pytest

from generals import MapConfig
from generals_arena.competitions.evaluation import make_profile, profile_id, expected_games
from generals_arena.policies import references
from generals_arena.policies.native import bundle_info
from generals_arena.util import file_digest


def test_training_reference_bundles_are_reused_and_portable(tmp_path, monkeypatch):
    first = references.training_opponents(tmp_path / 'first')
    assert [spec['name'] for spec in first] == list(references.BASELINE_NAMES + references.HEURISTIC_NAMES)
    expected = {}
    for spec in first[4:]:
        source = Path(references.__file__).with_name('reference_sources') / spec['name']
        assert bundle_info(spec['bundle'])['parameters'] == json.loads((source / 'params.json').read_text())
        expected[spec['name']] = file_digest(spec['bundle'])
    monkeypatch.setattr(references, 'build_bundle', lambda *a, **kw: pytest.fail('identical references rebuilt'))
    second = references.training_opponents(tmp_path / 'second')
    assert {spec['name']: file_digest(spec['bundle']) for spec in second[4:]} == expected
    left = make_profile(MapConfig(9, 9, 2), opponents=first, maps_per_opponent=100)
    right = make_profile(MapConfig(9, 9, 2), opponents=second, maps_per_opponent=100)
    assert profile_id(left) == profile_id(right)
    assert expected_games(left) == 1600
    assert references.training_opponents(tmp_path / 'first') == first
