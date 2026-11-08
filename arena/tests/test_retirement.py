"""Roster changes retain ratings, histories and deterministic pending games."""
import pytest

from test_league import league, add
from generals_arena.competitions.jobs import eligible_schedule, prepare_batch, run_jobs
from generals_arena.competitions.leaderboard import LeaderboardStatistics


@pytest.mark.parametrize('players', [2, 3])
def test_retirement_only_changes_roster_and_future_scheduling(tmp_path, players):
    store = league(tmp_path, player_counts=(players,))
    entry, job = add(store, games=3)
    run_jobs(store)
    ratings, matches = store.ratings(), store.matches()
    before = store.leaderboard()
    key = entry['entrant_id']
    store.set_retired('@aggressive', reason='superseded')
    assert store.ratings() == ratings and store.matches() == matches
    assert key not in {r['entrant_id'] for r in store.leaderboard()['ranking']}
    assert key not in {r['entrant_id'] for r in LeaderboardStatistics(store).leaderboard()['ranking']}
    all_rows = store.leaderboard(include_retired=True)['ranking']
    assert next(r for r in all_rows if r['entrant_id'] == key)['retired']
    assert len(all_rows) == len(before['ranking'])
    newcomer, _ = add(store, 'defensive', games=3)
    new_job = next(j for j in store.jobs() if j['entrant_id'] == newcomer['entrant_id'])
    assert all(key not in p['entrant_ids'] for p in eligible_schedule(store, new_job))
    with pytest.raises(ValueError, match='retired'):
        eligible_schedule(store, next(j for j in store.jobs() if j['id'] == job))
    store.set_retired(key, False)
    assert store.ratings() == ratings
    restored = next(r for r in store.leaderboard()['ranking'] if r['entrant_id'] == key)
    assert restored == next(r for r in before['ranking'] if r['entrant_id'] == key)


def test_retirement_rejects_reference_and_scheduled_games(tmp_path):
    store = league(tmp_path)
    with pytest.raises(ValueError, match='reference'):
        store.set_retired(store.reference)
    entry, job = add(store, games=2)
    with pytest.raises(ValueError, match='queued'):
        store.set_retired(entry['entrant_id'])
    prepare_batch(store, store.jobs()[0])
    store.fail(job, 'paused')
    with pytest.raises(ValueError, match='pending/live'):
        store.set_retired(entry['entrant_id'])


def test_native_infrastructure_failure_retries_without_erasing_evidence(tmp_path):
    from pathlib import Path
    from generals_arena.policies.artifacts import snapshot_policy
    store = league(tmp_path)
    spec = snapshot_policy(tmp_path/'registry', Path(__file__).parent.parent/'examples/cpp_wait')
    _, job = store.add(spec, games=1)
    plan = prepare_batch(store, store.jobs()[0])[0]
    store.fail_game({**plan, 'policy_error': 'worker disconnected', 'error_detail': {}}, b'evidence')
    store.fail(job, 'worker disconnected')
    store.retry(job)
    replacement = prepare_batch(store, store.jobs()[0])[0]
    assert replacement['retry_of'] == plan['game_id']
    assert replacement['game_id'] != plan['game_id']
    for key in ('entrant_ids', 'instance_ids', 'map', 'map_seed', 'controller_seeds'):
        assert replacement[key] == plan[key]
    with store.connect() as db:
        assert db.execute('SELECT replay FROM games WHERE id=?', (plan['game_id'],)).fetchone()[0] == b'evidence'


def test_retirement_between_planning_and_insertion_rejects_stale_roster(tmp_path):
    store = league(tmp_path)
    old = store.register({'name': 'old', 'kind': 'baseline', 'policy': 'aggressive'})
    plan = {'game_id': 'stale', 'entrant_ids': [old['entrant_id'], store.reference]}
    store.set_retired(old['entrant_id'])
    with pytest.raises(ValueError, match='retired before scheduling'):
        store.insert_plans([plan])
    assert store.matches()['total'] == 0
