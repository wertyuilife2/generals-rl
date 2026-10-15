"""WAIT defines the public zero point without rewriting archived scores."""
import pytest
from generals_arena.league.config import LeagueConfig
from generals_arena.league.service import eligible_schedule
from generals_arena.league.store import LeagueStore
from generals_arena import research
from generals_arena.league.rating_policy import POLICY


def league(tmp_path):
    return LeagueStore.create(tmp_path/'league', LeagueConfig(max_ticks=2, min_side=15, max_side=15))


def test_wait_is_display_origin_and_nonparticipants_keep_raw_rating(tmp_path):
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
    raw_before = store.ratings()
    complete('calibration', [other, reference])
    after = {r['entrant_id']:r['elo'] for r in store.leaderboard()['ranking']}
    assert after[reference] == before[reference] == 0
    raw_after = store.ratings()
    assert raw_after[idle] == raw_before[idle] and raw_after[random] == raw_before[random]
    assert after[idle]-before[idle] == pytest.approx(-raw_after[reference])
    assert after[random]-before[random] == pytest.approx(-raw_after[reference])
    assert after[idle]-after[random] == pytest.approx(before[idle]-before[random])
    assert after[other] > before[other]
    assert raw_after[reference] < 0
    report = store.leaderboard()
    record = store.replay('calibration')
    assert store.rebuild() == report
    assert store.replay('calibration') == record
    assert LeagueStore(store.directory).anchor == reference
    with pytest.raises(ValueError, match='before rated games'):
        store.set_rating_reference(random)
    store.verify_execution()


def test_wait_uses_ordinary_matchmaking_including_humans_and_ffa(tmp_path):
    store = league(tmp_path)
    entrant, _ = store.add({'name':'test-v1','kind':'baseline','policy':'aggressive'}, games=1000)
    human = store.account('human')['entrant']['entrant_id']
    job = {'id':'ordinary','entrant_id':entrant['entrant_id'],'done':0,'budget':1000}
    included = 0
    for block in range(100):
        plans = eligible_schedule(store, {**job,'done':block}, players=2)
        included += store.reference in plans[0]['entrant_ids']
    assert 30 < included < 70  # WAIT and random have equal rating/coverage.
    plans = eligible_schedule(store, {**job,'entrant_id':human}, human=True, players=4)
    assert store.reference in plans[0]['entrant_ids']
    plans = eligible_schedule(store, {**job,'entrant_id':store.reference}, players=8)
    assert len(plans[0]['entrant_ids']) == 8
    assert plans[0]['entrant_ids'].count(store.reference) >= 2
    assert eligible_schedule(store, job) == eligible_schedule(store, job)


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
