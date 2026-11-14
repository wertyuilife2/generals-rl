"""Budgets and hard watchdogs are independently configured per policy."""
import pytest

from generals_arena.policies.limits import configure, duration, execution_limits, freeze_plan
from generals_arena.policies.entries import prepare_entrant
from generals_arena.competitions.store import CompetitionStore
from generals_arena.competitions.config import LeagueConfig
from generals_arena.matches.runner import MatchRunner
from test_native_batch import policy, plans


def test_defaults_and_validation():
    assert execution_limits({'kind': 'cpp'}) == dict(decision_budget_ms=10, decision_timeout_ms=100)
    assert execution_limits({'kind': 'package'}) == dict(decision_budget_ms=50, decision_timeout_ms=100)
    assert execution_limits({'kind': 'baseline'}) == dict(decision_budget_ms=10, decision_timeout_ms=None)
    assert duration('off', allow_off=True) is None
    assert duration('50ms') == 50
    for settings in ({'decision_budget_ms': 0}, {'decision_timeout_ms': 9}, {'decision_budget_ms': True}, {'extra': 1}):
        with pytest.raises(ValueError):
            execution_limits({'execution': settings})


def test_configure_preserves_identity_and_frozen_plans(tmp_path):
    spec = dict(name='random', kind='baseline', policy='random')
    store = CompetitionStore.create(tmp_path/'league', LeagueConfig())
    entry = store.register(spec)
    if isinstance(entry, str):
        entry = store.entrants()[entry]
    original = prepare_entrant(spec)
    updated = prepare_entrant({**spec, 'execution': {'decision_timeout_ms': 100}})
    assert updated['entrant_id'] == original['entrant_id']
    key = original['entrant_id']
    plan = freeze_plan({'entrant_ids': [key, key]}, store.entrants())
    configure(store, '@random', {'decision_timeout_ms': 100})
    assert execution_limits(store.entrants()[key])['decision_timeout_ms'] == 100
    assert freeze_plan(plan, store.entrants()) == plan
    assert plan['decision_limits'][0]['decision_timeout_ms'] is None


@pytest.mark.parametrize('execution', ['isolated', 'batch'])
@pytest.mark.parametrize('hard', [None, 100])
def test_returned_over_budget_action_forfeits_with_or_without_watchdog(tmp_path, execution, hard):
    entry = policy(tmp_path, 'if(s.seed==42) std::this_thread::sleep_for(std::chrono::milliseconds(25)); return 0;')
    entry['execution'] = dict(decision_budget_ms=10, decision_timeout_ms=hard)
    with MatchRunner({entry['entrant_id']: entry}, max_ticks=4, execution=execution) as runner:
        result, _ = runner.run(plans(entry, 3, [[42, 2, 3]]))
    row = result[0]
    assert row['elimination_ticks'][0] == 1
    assert not row['final_alive'][0]
    assert row['ticks'] == 4
    assert len(row['forfeits']) == 1
    event = row['forfeits'][0]
    assert event['kind'] == 'timeout'
    assert 10_000_000 < event['elapsed_ns'] < 100_000_000
    assert event['hard_timeout_ns'] == (None if hard is None else hard * 1_000_000)


def test_private_override_is_recorded_without_changing_registration(tmp_path):
    from generals_arena.competitions.private import run
    store = CompetitionStore.create(tmp_path/'league', LeagueConfig(max_ticks=3))
    entry = store.register(dict(name='random', kind='baseline', policy='random'))
    configure(store, 'random', {'decision_budget_ms': 20, 'decision_timeout_ms': 100})
    report = run(['@random', 'baseline/wait'], league=store.directory, games=1, size='15x15', workers=1,
                 execution={'@random': {'decision_timeout_ms': None}})
    assert report['ok']
    random = next(p for p in report['players'] if p['kind'] == 'baseline')
    assert random['timing']['limit_ns'] == 20_000_000
    assert random['timing']['hard_timeout_ns'] is None
    assert store.entrants()[entry['entrant_id']]['execution']['decision_timeout_ms'] == 100
    private = CompetitionStore(report['league_directory'])
    import json
    with private.connect() as db:
        result = json.loads(db.execute('SELECT result FROM games LIMIT 1').fetchone()[0])
    assert result['decision_limits'][0] == dict(decision_budget_ms=20, decision_timeout_ms=None)


def test_same_strategy_seat_overrides_follow_rotations(tmp_path):
    from generals_arena.competitions.private import run
    import json
    store = CompetitionStore.create(tmp_path/'league', LeagueConfig(max_ticks=2))
    store.register(dict(name='random', kind='baseline', policy='random'))
    report = run(['@random', 'baseline/random'], league=store.directory, games=2, size='15x15', workers=1,
                 execution={'@random': {'decision_budget_ms': 20, 'decision_timeout_ms': 100}})
    assert report['ok']
    with CompetitionStore(report['league_directory']).connect() as db:
        results = [json.loads(row[0]) for row in db.execute('SELECT result FROM games ORDER BY rowid')]
    assert [[v['decision_budget_ms'] for v in r['decision_limits']] for r in results] == [[20, 10], [10, 20]]
