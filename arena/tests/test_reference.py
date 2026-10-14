"""WAIT defines the public zero point without rewriting archived scores."""
import pytest
from generals_arena.league.config import LeagueConfig
from generals_arena.league.service import eligible_schedule
from generals_arena.league.store import LeagueStore
from generals_arena import research
from generals_arena.league.rating_policy import POLICY


def league(tmp_path):
    return LeagueStore.create(tmp_path/'league', LeagueConfig(max_ticks=2, min_side=15, max_side=15))


def test_wait_is_fixed_and_nonparticipants_never_change_rating(tmp_path):
    store = league(tmp_path)
    reference = store.reference
    assert reference == store.anchor
    random = next(e['entrant_id'] for e in store.entrants().values() if e['name']=='random')
    other = store.register({'name':'other','kind':'baseline','policy':'aggressive'})['entrant_id']
    idle = store.account('idle')['entrant']['entrant_id']
    def complete(key, ids):
        plan = {'game_id':key, 'entrant_ids':ids, 'rating_policy':POLICY}
        store.insert_plans([plan])
        store.complete_result({**plan, 'ticks':10, 'elimination_ticks':[None,10],
                               'winner_seat':0, 'kills':[1,0]}, b'')
    complete('initial', [idle, random])
    before = {r['entrant_id']:r['elo'] for r in store.leaderboard()['ranking']}
    complete('calibration', [other, reference])
    after = {r['entrant_id']:r['elo'] for r in store.leaderboard()['ranking']}
    assert after[reference] == before[reference] == 0
    assert after[idle] == before[idle] and after[random] == before[random]
    assert after[other] > before[other]
    assert store.ratings()[reference] == 0
    report = store.leaderboard()
    record = store.replay('calibration')
    assert store.rebuild() == report
    assert store.replay('calibration') == record
    assert LeagueStore(store.directory).anchor == reference
    with pytest.raises(ValueError, match='before rated games'):
        store.set_rating_reference(random)
    store.verify_execution()


def test_wait_is_rare_single_seat_and_absent_from_human_matchmaking(tmp_path):
    store = league(tmp_path)
    entrant, _ = store.add({'name':'test-v1','kind':'baseline','policy':'aggressive'}, games=1000)
    job = {'id':'coverage','entrant_id':entrant['entrant_id'],'done':0,'budget':2000}
    included, total, coverage = 0, 0, {}
    job['budget'] = 1000
    while total < job['budget']:
        plans = eligible_schedule(store, {**job, 'done':total}, coverage=coverage)
        counts = [p['entrant_ids'].count(store.reference) for p in plans]
        assert len(set(counts)) == 1
        assert max(counts) <= 1
        included += sum(counts)
        total += len(plans)
        for plan in plans:
            for key in set(plan['entrant_ids']) - {entrant['entrant_id']}:
                coverage[key] = coverage.get(key,0)+1
    assert .03 <= included / total <= .04
    assert eligible_schedule(store, job) == eligible_schedule(store, job)
    human = store.account('human')['entrant']['entrant_id']
    for block in range(100):
        plans = eligible_schedule(store, {**job,'id':str(block),'entrant_id':human}, human=True)
        assert all(store.reference not in p['entrant_ids'] for p in plans)
    for plan in eligible_schedule(store, {**job,'entrant_id':store.reference}):
        assert plan['entrant_ids'].count(store.reference) == 1
        assert len(plan['entrant_ids']) == 2


def test_explicit_wait_seats_work_after_private_directory_is_renamed(tmp_path):
    store = league(tmp_path)
    report = research.run(['baseline/wait','baseline/wait'], league=store.directory, games=2)
    assert report['ok'] and report['completed_games'] == 2
    private = LeagueStore(report['league_directory'])
    assert private.entrants()[private.reference]['name'] == 'WAIT'
    for match in private.matches()['matches']:
        assert match['entrant_ids'] == [private.reference]*2
        assert match['kills'] == [0,0]
    assert store.leaderboard()['completed_games'] == 0
