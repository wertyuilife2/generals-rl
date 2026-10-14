"""Independent mode evidence, exact resumable quotas and diverse FFA seats."""
from collections import Counter
import random

import pytest

from generals_arena.league.config import LeagueConfig
from generals_arena.league.modes import MODES, split_budget
from generals_arena.league.rating_policy import POLICY
from generals_arena.league.scheduler import choose_opponents
from generals_arena.league.service import prepare_batch
from generals_arena.league.statistics import LeaderboardStatistics
from generals_arena.league.store import LeagueStore
from generals_arena.util import canonical


def finish(store, plan, winner=0):
    n = len(plan['entrant_ids'])
    return store.complete_result({**plan, 'ticks':n, 'winner_seat':winner,
        'elimination_ticks':[None if i == winner else i+1 for i in range(n)],
        'kills':[n-1 if i == winner else 0 for i in range(n)]}, b'original replay')


def test_ffa_cannot_change_duel_ratings_evidence_or_statistics(tmp_path):
    store = LeagueStore.create(tmp_path/'league', reference='random')
    a = store.register({'name':'a-v1','kind':'baseline','policy':'aggressive'})['entrant_id']
    b = store.register({'name':'b-v1','kind':'baseline','policy':'defensive'})['entrant_id']
    stats = {mode: LeaderboardStatistics(store, mode) for mode in MODES}
    duel = dict(game_id='duel', entrant_ids=[a,store.anchor], rating_policy=POLICY)
    store.insert_plans([duel]); finish(store, duel)
    before = stats['1v1'].leaderboard()
    ffa = dict(game_id='ffa', entrant_ids=[a,b,store.anchor], rating_policy=POLICY)
    store.insert_plans([ffa]); finish(store, ffa, winner=2)
    assert stats['1v1'].leaderboard() == before
    assert store.ratings('1v1')[a] > 0 > store.ratings('ffa')[a]
    assert b not in store.ratings('1v1')
    assert store.pair_counts(a, '1v1') == {store.anchor:1}
    assert store.pair_counts(a, 'ffa') == {store.anchor:1,b:1}
    assert all(store.ratings(mode)[store.anchor] == 0 for mode in MODES)
    snapshot = store.leaderboards()
    store.rebuild()
    assert store.leaderboards() == snapshot
    assert stats['ffa'].leaderboard()['completed_games'] == 1


def test_exact_500_per_mode_even_across_partial_rotation_resumes(tmp_path):
    store = LeagueStore.create(tmp_path/'league')
    target, job_id = store.add({'name':'candidate','kind':'baseline','policy':'aggressive'},
                              mode_games=split_budget(1000, store.config.player_counts))
    for index in range(9):
        store.register({'name':f'opponent-{index}', 'kind':'baseline','policy':'random'})
    counts, reference = Counter(), Counter()
    while (job := next(j for j in store.jobs() if j['id'] == job_id))['status'] != 'complete':
        plans = prepare_batch(store, job)
        # Complete only a prefix, reopen the store, then resume the saved window.
        for plan in plans[:7]:
            mode = plan['rating_mode']
            counts[mode] += 1
            assert len(set(plan['entrant_ids'])) == len(plan['entrant_ids'])
            reference[mode] += store.anchor in plan['entrant_ids']
            finish(store, plan)
        store = LeagueStore(store.directory)
    assert counts == {'1v1':500,'ffa':500}
    assert all(0 < reference[mode] <= 20 for mode in MODES)
    assert all(value == {'budget':500,'done':500} for value in job['modes'].values())
    assert len(store.matches(limit=100)['matches']) == 100
    assert store.matches()['total'] == 1000


@pytest.mark.parametrize('available', [1,2,3,7,12])
def test_small_roster_spreads_repeats_evenly_and_large_roster_has_none(available):
    ids = [f'bot-{i}' for i in range(available)]
    for seed in range(20):
        seats = [ids[0], *choose_opponents(ids[0], ids, 8, {}, {}, ids[0], random.Random(seed), seed)]
        counts = Counter(seats)
        assert len(counts) == min(available, 8)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_schema_three_upgrade_preserves_recordings_and_replaces_unified_scores(tmp_path):
    store = LeagueStore.create(tmp_path/'league', reference='random')
    target, _ = store.add({'name':'test','kind':'baseline','policy':'aggressive'}, games=2)
    for i, count in enumerate((2,4)):
        plan = dict(game_id=str(i), entrant_ids=[target['entrant_id']]*(count-1)+[store.anchor], rating_policy=POLICY)
        store.insert_plans([plan]); finish(store, plan)
    expected = store.leaderboards()
    recordings = {str(i):store.replay(str(i))[1] for i in range(2)}
    jobs = store.jobs()
    # Recreate the persisted schema used before separate modes existed.
    with store.connect(write=True) as db:
        db.execute('DROP TABLE ratings')
        db.execute('CREATE TABLE ratings(entrant_id TEXT PRIMARY KEY, elo REAL, games INTEGER, rated_games INTEGER, appearances INTEGER, wins INTEGER, losses INTEGER, draws INTEGER)')
        db.execute('INSERT INTO ratings VALUES (?,?,?,?,?,?,?,?)', (target['entrant_id'],9999,2,2,4,2,0,0))
        db.execute('DROP TABLE rating_pairs')
        db.execute('CREATE TABLE rating_pairs(left_id TEXT, right_id TEXT, weight REAL, score REAL, PRIMARY KEY(left_id,right_id))')
        db.execute('DROP TABLE job_modes')
        old = {**store.manifest, 'schema':3, 'rating':'unified-anchored-rank-performance-v3'}
        db.execute("UPDATE meta SET value=? WHERE key='manifest'", (canonical(old),))
    migrated = LeagueStore(store.directory)
    with pytest.raises(ValueError, match='rules/source changed'):
        migrated.verify_execution()
    migrated.rebuild(upgrade=True)
    migrated.verify_execution()
    assert migrated.leaderboards() == expected
    assert migrated.jobs() == jobs
    assert {str(i):migrated.replay(str(i))[1] for i in range(2)} == recordings
    assert {m['rating_mode'] for m in migrated.matches()['matches']} == set(MODES)
    assert migrated.manifest['schema'] == 4
