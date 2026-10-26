"""WAIT is rated normally and only defines the displayed 1v1 origin."""
import pytest
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.jobs import eligible_schedule
from generals_arena.competitions.store import CompetitionStore
from generals_arena.competitions.private import run
from generals_arena.competitions.rating import MODES
from generals_arena.competitions.rating import POLICY


def league(tmp_path):
    return CompetitionStore.create(tmp_path/'league', LeagueConfig(max_ticks=2, min_side=15, max_side=15))


@pytest.mark.parametrize('reference', ['wait', 'random'])
def test_one_persisted_reference_without_anchor_fallback(tmp_path, reference):
    store = CompetitionStore.create(tmp_path / 'league', reference=reference)
    with store.connect() as db:
        keys = {row['key'] for row in db.execute('SELECT key FROM meta')}
    assert 'rating_reference' in keys and 'anchor' not in keys
    assert store.entrants()[store.reference]['name'] == ('WAIT' if reference == 'wait' else 'random')


def test_one_wait_identity_and_controller_serve_scalar_batch_and_store(tmp_path):
    import numpy as np
    from generals import Action
    from generals_arena.matches.config import MatchConfig
    from generals_arena.policies.builtin import wait_spec, with_wait_reference
    from generals_arena.policies.entries import load_controller, prepare_entrant
    spec = wait_spec()
    assert spec['kind'] == 'custom'
    entry = prepare_entrant(spec)
    store = league(tmp_path)
    assert entry['entrant_id'] == store.reference
    controller = load_controller(entry, MatchConfig())
    controller.reset(0, 7)
    assert controller.act(None) == Action.wait()
    controller.reset_batch([0, 1], [7, 8])
    np.testing.assert_array_equal(controller.act_batch({'active': np.array([True, False])}), [0, 0])
    assert with_wait_reference([], 2) == [spec]
    assert with_wait_reference([spec], 2) == [spec]
    assert with_wait_reference([], 3) == []
    with pytest.raises(ValueError, match='reserved'):
        with_wait_reference([{**spec, 'version': 'not-the-reference'}], 2)


def test_wait_display_is_zero_while_raw_rating_and_display_offset_move(tmp_path):
    store = league(tmp_path)
    reference = store.reference
    random = next(e['entrant_id'] for e in store.entrants().values() if e['name']=='random')
    other = store.register({'name':'other','kind':'baseline','policy':'aggressive'})['entrant_id']
    idle = store.account('idle')['entrant']['entrant_id']
    def complete(key, ids):
        plan = {'game_id':key, 'entrant_ids':ids, 'rating_policy':POLICY}
        store.insert_plans([plan])
        store.complete_result({**plan, 'ticks':10, 'elimination_ticks':[None,10],
                               'winner_seat':0, 'final_alive':[True, False], 'kills':[1,0]}, b'')
    complete('initial', [idle, random])
    before = {r['entrant_id']:r['raw_elo'] for r in store.leaderboard()['ranking']}
    complete('calibration', [other, reference])
    after = {r['entrant_id']:r['raw_elo'] for r in store.leaderboard()['ranking']}
    assert after[reference] < before[reference] == 0
    assert after[idle] == before[idle] and after[random] == before[random]
    assert after[other] > before[other]
    assert store.ratings()[reference] == after[reference]
    assert next(row for row in store.leaderboard()["ranking"] if row["anchor"])["elo"] == 0
    report = store.leaderboard()
    idle_row = next(row for row in report['ranking'] if row['entrant_id'] == idle)
    assert idle_row['elo'] == after[idle] - after[reference]
    record = store.replay('calibration')
    assert store.rebuild() == report
    assert store.replay('calibration') == record
    assert CompetitionStore(store.directory).reference == reference
    with pytest.raises(ValueError, match='before rated games'):
        store.set_rating_reference(random)
    store.verify_execution()


def test_wait_uses_ordinary_opponent_pool_in_both_modes_and_human_games(tmp_path):
    store = league(tmp_path)
    entrant, _ = store.add({'name':'test-v1','kind':'baseline','policy':'aggressive'}, games=1000)
    job = {'id':'coverage','entrant_id':entrant['entrant_id'],'done':0,'budget':1000}
    included, total = 0, 0
    coverage = {mode:{} for mode in MODES}
    while total < job['budget']:
        plans = eligible_schedule(store, {**job, 'done':total}, coverage=coverage)
        included += sum(plan['entrant_ids'].count(store.reference) for plan in plans)
        total += len(plans)
        for plan in plans:
            mode = plan['rating_mode']
            for key in set(plan['entrant_ids']) - {entrant['entrant_id']}:
                coverage[mode][key] = coverage[mode].get(key,0)+1
    assert included / total > .1  # No former 4% WAIT quota.
    assert eligible_schedule(store, job) == eligible_schedule(store, job)
    human = store.account('human')['entrant']['entrant_id']
    plans = eligible_schedule(store, {**job,'entrant_id':human}, human=True, players=8)
    assert store.reference in plans[0]['entrant_ids']
    for players in (2,8):
        plan = eligible_schedule(store, {**job,'entrant_id':store.reference}, players=players)[0]
        assert store.reference in plan['entrant_ids'] and len(plan['entrant_ids']) == players


def test_explicit_wait_seats_work_after_private_directory_is_renamed(tmp_path):
    store = league(tmp_path)
    report = run(['baseline/wait','baseline/wait'], league=store.directory, games=2)
    assert report['ok'] and report['completed_games'] == 2
    private = CompetitionStore(report['league_directory'])
    assert private.entrants()[private.reference]['name'] == 'WAIT'
    for match in private.matches()['matches']:
        assert match['entrant_ids'] == [private.reference]*2
        assert match['kills'] == [0,0]
    assert store.leaderboard()['completed_games'] == 0
