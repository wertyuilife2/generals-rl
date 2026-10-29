"""Arena hashes opaque bundles and never interprets model features or training."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import pytest

from generals_arena.policies import artifacts as artifacts
from generals_arena.matches.config import MatchConfig
from generals_arena.policies.entries import prepare_entrant
from generals_arena.policies.entries import verify_artifacts
from generals_arena.policies.entries import load_controller
from generals_arena.util import file_digest
from policy_test_utils import make_bundle


def test_opaque_complete_bundle_is_portable_immutable_and_device_independent(tmp_path):
    source = make_bundle(tmp_path/'source')
    expected = artifacts.validate_bundle(source)
    first = artifacts.snapshot_bundle(tmp_path/'catalog', source)
    moved = tmp_path/'moved'
    shutil.copytree(source, moved)
    second = artifacts.snapshot_bundle(tmp_path/'other', moved, device='cuda:0')
    assert first['kind'] == 'package'
    assert first['version'] == second['version'] == expected['policy_id']
    assert artifacts.resolve_bundle(tmp_path/'catalog', first['version']) == first
    shutil.rmtree(source)
    entry = prepare_entrant(first)
    assert load_controller(entry, MatchConfig()).player_id == 0
    verify_artifacts(entry)
    (tmp_path/'catalog'/'policies'/first['version']/'actor.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        artifacts.resolve_bundle(tmp_path/'catalog', first['version'])


def test_declared_capabilities_are_generic_and_config_is_never_parsed(tmp_path):
    source = make_bundle(tmp_path/'source', width=11, height=7, players=(4, 8))
    info = artifacts.validate_bundle(source)
    caps = info['manifest']['capabilities']
    assert artifacts.capabilities_supports(caps, 11, 7, 4)
    assert not artifacts.capabilities_supports(caps, 9, 9, 2)
    manifest = info['manifest']
    manifest['capabilities']['maps'] = {'width': [7, 15], 'height': [5, 9]}
    (source/'manifest.json').write_text(json.dumps(manifest))
    assert artifacts.capabilities_supports(artifacts.validate_bundle(source)['manifest']['capabilities'], 9, 9, 8)


@pytest.mark.parametrize('role', ['weights', 'config', 'code'])
def test_mutation_is_rejected_before_publication(tmp_path, role):
    source = make_bundle(tmp_path/'source')
    manifest = json.loads((source/'manifest.json').read_text())
    manifest['artifacts'][role]['sha256'] = '0'*64
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='SHA256'):
        artifacts.snapshot_bundle(tmp_path/'catalog', source)
    assert not (tmp_path/'catalog').exists()


def test_alias_cannot_silently_change_weights_or_code(tmp_path):
    source = make_bundle(tmp_path/'source')
    artifacts.snapshot_bundle(tmp_path/'catalog', source)
    (source/'actor.pt').write_bytes(b'new')
    manifest = json.loads((source/'manifest.json').read_text())
    manifest['artifacts']['weights']['sha256'] = file_digest(source/'actor.pt')
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='alias'):
        artifacts.snapshot_bundle(tmp_path/'catalog', source)


def test_atomic_concurrent_import_is_idempotent(tmp_path):
    source = make_bundle(tmp_path/'source')
    with ThreadPoolExecutor(max_workers=4) as pool:
        specs = list(pool.map(lambda _: artifacts.snapshot_bundle(tmp_path/'catalog', source), range(8)))
    assert all(spec == specs[0] for spec in specs)
    assert len(list((tmp_path/'catalog'/'policies').iterdir())) == 1


@pytest.mark.parametrize('path', ['../outside', '/outside', 'policy/../weights', 'actor.pt/child'])
def test_manifest_paths_cannot_escape_or_overlap(tmp_path, path):
    source = make_bundle(tmp_path/'source')
    manifest = json.loads((source/'manifest.json').read_text())
    manifest['artifacts']['config']['path'] = path
    with pytest.raises(ValueError):
        artifacts.load_manifest(manifest)


def test_symlink_and_loader_mutations_are_rejected(tmp_path, monkeypatch):
    source = make_bundle(tmp_path/'source')
    path = source/'actor.pt'
    outside = tmp_path/'outside'
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        artifacts.validate_bundle(source)
    path.unlink()
    outside.rename(path)
    entry = prepare_entrant({'name':'test','kind':'package','bundle':str(source)})
    monkeypatch.setattr('generals_arena.policies.entries._policy_implementation', lambda _: {'changed': True})
    with pytest.raises(ValueError, match='loader changed'):
        verify_artifacts(entry)


def test_selection_uses_an_explicit_archive_path_and_supplied_entries(tmp_path):
    from generals_arena.policies.builtin import wait_spec
    source = make_bundle(tmp_path/'source')
    directory = tmp_path/'archive'
    selected = artifacts.select_policy(directory, source, device='cuda:0')
    entry = prepare_entrant(selected)
    entries = {entry['entrant_id']: entry}
    assert artifacts.select_policy(directory, 'policy/' + selected['version'], entries=entries)['device'] == 'cuda:0'
    assert artifacts.select_policy(directory, '@' + entry['name'], entries=entries)['device'] == 'cuda:0'
    assert artifacts.select_policy(directory, '@' + entry['name'], entries=entries, device='cpu')['device'] == 'cpu'
    assert artifacts.select_policy(directory, 'baseline/wait') == wait_spec()
    assert artifacts.select_policy(directory, 'baseline/aggressive')['policy'] == 'aggressive'
    with pytest.raises(ValueError, match='registered'):
        artifacts.select_policy(directory, '@missing', entries=entries)
    with pytest.raises(ValueError, match='native policies'):
        artifacts.select_policy(directory, source, parameters={})


def test_native_snapshot_is_path_based_immutable_and_selectable(tmp_path):
    source = tmp_path/'policy.cpp'
    example = Path(__file__).resolve().parents[1]/'examples'/'cpp_wait'/'policy.cpp'
    source.write_bytes(example.read_bytes())
    directory = tmp_path/'archive'
    first = artifacts.snapshot_policy(directory, source)
    assert artifacts.snapshot_policy(directory, source) == first
    entry = prepare_entrant(first)
    assert artifacts.select_policy(directory, 'policy/' + first['version']) == first
    assert artifacts.select_policy(directory, '@' + entry['name'], entries={entry['entrant_id']: entry}) == first
    source.write_text(source.read_text()+'\n// a new source version\n')
    assert artifacts.snapshot_policy(directory, source)['version'] != first['version']
    with Path(first['bundle']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        artifacts.select_policy(directory, 'policy/' + first['version'])


def test_failed_rule_admission_closes_the_created_controller(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    closed = []
    class Controller:
        def reset(self, player_id, seed): pass
        def act(self, observation): pass
        def on_result(self, move): pass
        def validate_rules(self, *args): raise ValueError('unsupported geometry')
        def close(self): closed.append(True)
    monkeypatch.setitem(sys.modules, 'admission_test_factory', SimpleNamespace(create=lambda **kwargs: Controller()))
    entry = prepare_entrant({'name': 'failure', 'kind': 'custom', 'version': 'v1',
                             'factory': 'admission_test_factory:create'})
    with pytest.raises(ValueError, match='unsupported geometry'):
        load_controller(entry, MatchConfig())
    assert closed == [True]
