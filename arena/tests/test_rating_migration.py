"""A scoring migration preserves games and can resume its pending rotations."""
import json

import pytest

from generals_arena.league.config import LeagueConfig
from generals_arena.league.rating_policy import LEGACY_POLICY, POLICY
from generals_arena.league.service import process_lock
from generals_arena.league.statistics import LeaderboardStatistics
from generals_arena.league.store import LeagueStore


def test_upgrade_preserves_records_jobs_replays_and_rebuilds_evidence(tmp_path):
    store = LeagueStore.create(tmp_path/'league', LeagueConfig(max_ticks=1))
    a, job = store.add({'name':'a','kind':'baseline','policy':'aggressive'}, games=2)
    plans = [dict(game_id=str(i),entrant_ids=[a['entrant_id'],store.anchor], map_seed=123,
                  map={'width':15,'height':15},rating_policy=LEGACY_POLICY) for i in range(2)]
    store.insert_plans(plans, job)
    store.complete_result({**plans[0],'ticks':1,'elimination_ticks':[None,1],
                           'winner_seat':0,'kills':[1,0]}, b'unchanged replay')
    stats = LeaderboardStatistics(store)
    stats.leaderboard()
    jobs = store.jobs()
    saved, blob = store.replay('0')
    with process_lock(store, '.worker.lock'):
        with pytest.raises(RuntimeError, match='another process'):
            store.rebuild(upgrade=True)
    publication = store.directory/'commits'/'test'/'publication.json'
    publication.parent.mkdir(parents=True)
    publication.write_text(json.dumps({'entrant_id':a['entrant_id'],'rating':-999}))
    store.rebuild(upgrade=True)
    result, replay = store.replay('0')
    ignored = {'rating_policy','rating_group_size','rating_changes'}
    assert {k:v for k,v in result.items() if k not in ignored} == {k:v for k,v in saved.items() if k not in ignored}
    assert replay == blob and store.jobs() == jobs
    assert result['rating_policy'] == POLICY and result['rating_group_size'] == 2
    assert json.loads(publication.read_text())['rating'] == store.ratings()[a['entrant_id']]
    # Existing in-memory derived statistics must notice a score-only revision.
    assert stats.leaderboard() == LeaderboardStatistics(store).leaderboard()
    pending = store.pending(job)
    assert pending['rating_policy'] == POLICY and pending['rating_group_size'] == 2
    store.complete_result({**pending,'ticks':1,'elimination_ticks':[1,None],
                           'winner_seat':1,'kills':[0,1]}, b'next replay')
    # A win and a loss are two chronological updates, not a batch refit.
    assert abs(store.ratings()[a['entrant_id']]) < 10
    with store.connect() as db:
        assert db.execute('SELECT sum(weight) FROM rating_pairs').fetchone()[0] == 1
    before = store.leaderboard()
    assert store.rebuild() == before
    assert store.rebuild(upgrade=True) == before
