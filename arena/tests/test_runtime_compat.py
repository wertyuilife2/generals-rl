"""A performance migration must preserve history and retain strict source gates."""
import json

import pytest

from generals import Action
from generals_arena import runtime_compat as compat
from generals_arena.league.config import LeagueConfig
from generals_arena.league.service import run_jobs
from generals_arena.league.store import LeagueStore


def completed_league(tmp_path):
    config = LeagueConfig(player_counts=(2,), map_generation=None, min_side=15,
                          max_side=15, max_ticks=8)
    store = LeagueStore.create(tmp_path/'league', config, reference='random')
    store.add({'name': 'aggressive', 'kind': 'baseline', 'policy': 'aggressive'}, games=2)
    run_jobs(store)
    store.account('Human account')
    return store


def history(store):
    with store.connect() as db:
        return {table: [tuple(row) for row in db.execute(f'SELECT * FROM {table} ORDER BY rowid')]
                for table in ('entrants', 'accounts', 'ratings', 'rating_pairs', 'games', 'jobs', 'job_modes')}


def test_install_preserves_history_accounts_ids_and_backup(tmp_path):
    store = completed_league(tmp_path)
    validation = compat.verify_replays(store, per_count=1, per_baseline=1)
    assert validation['ticks'] and validation['observations']
    assert len(validation['baseline_decisions']) == 2
    before = history(store)
    report = {'runtime': compat.runtime_identity(), 'replay_validation': validation}
    receipt = compat.install_verified(store, store.manifest, store.config.manifest(), report)
    assert history(store) == before
    assert receipt['package_aliases'] == []
    import sqlite3
    with sqlite3.connect(receipt['backup']) as backup:
        assert backup.execute('SELECT count(*) FROM games').fetchone()[0] == 2
    store.verify_execution()


def test_new_runtime_and_unfinished_jobs_cannot_be_adopted(tmp_path):
    store = completed_league(tmp_path)
    validation = compat.verify_replays(store, per_count=1, per_baseline=1)
    report = {'runtime': compat.runtime_identity(), 'replay_validation': validation}
    changed = {**report, 'runtime': {**report['runtime'], 'generals': 'unverified'}}
    with pytest.raises(ValueError, match='runtime changed'):
        compat.install_verified(store, store.manifest, store.config.manifest(), changed)
    store.add({'name': 'defensive', 'kind': 'baseline', 'policy': 'defensive'}, games=2)
    before = history(store)
    with pytest.raises(ValueError, match='queued games'):
        compat.install_verified(store, store.manifest, store.config.manifest(), report)
    assert history(store) == before


def test_package_alias_requires_exact_entrant_source_and_rules(tmp_path):
    store = completed_league(tmp_path)
    entrant = store.entrants()[store.anchor]
    alias = {'entrant_id': entrant['entrant_id'], 'package': 'generals_baselines',
             'from': 'old-source', 'to': 'verified-source'}
    receipt = {'to_rules_id': store.manifest['rules_id'], 'package_aliases': [alias]}
    with store.connect(write=True) as db:
        db.execute("INSERT INTO meta VALUES ('runtime_compatibility',?)", (json.dumps([receipt]),))
    assert compat.compatible_implementation(store, entrant, 'generals_baselines', 'old-source', 'verified-source')
    assert not compat.compatible_implementation(store, entrant, 'generals_baselines', 'old-source', 'another-source')
    assert not compat.compatible_implementation(store, {**entrant, 'entrant_id': 'other'}, 'generals_baselines', 'old-source', 'verified-source')
    store.manifest = {**store.manifest, 'rules_id': 'another-ruleset'}
    assert not compat.compatible_implementation(store, entrant, 'generals_baselines', 'old-source', 'verified-source')


def test_black_box_action_change_blocks_verification(tmp_path, monkeypatch):
    from generals_arena import registry
    store = completed_league(tmp_path)
    original = registry.load_controller
    def altered(*args, **kwargs):
        controller = original(*args, **kwargs)
        act = controller.act
        controller.act = lambda obs: Action.decode((act(obs).encode() + 1) % (obs.size * 8 + 1))
        return controller
    monkeypatch.setattr(registry, 'load_controller', altered)
    with pytest.raises(ValueError, match='black-box action changed'):
        compat.verify_replays(store, per_count=1, per_baseline=1)


@pytest.mark.parametrize('field,value', [('rating', 'other-rating'), ('config', {'max_ticks': 1}),
                                        ('decision_timing', 'after-growth')])
def test_semantic_manifest_changes_are_rejected(tmp_path, field, value):
    previous = LeagueConfig().manifest()
    with pytest.raises(ValueError, match=f'cannot change {field}'):
        compat.check_transition(previous, {**previous, field: value}, tmp_path)
