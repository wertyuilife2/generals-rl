"""Informative duels converge faster; multiplayer copies cannot overweight a policy."""
from collections import Counter
import math

import pytest

from generals_arena.league.config import LeagueConfig
from generals_arena.league.rating import LOGISTIC_SLOPE, expected, rate
from generals_arena.league.rating_policy import POLICY, LEGACY_POLICY, ONLINE_POLICY, player_count, score
from generals_arena.league.rank_model import fit
from generals_arena.league.service import eligible_schedule
from generals_arena.league.store import LeagueStore


def test_game_distribution_targets_half_duels_not_half_rotation_groups():
    config = LeagueConfig()
    counts = Counter()
    for block in range(12000):
        count = player_count(config, {'id':'distribution', 'done':block})
        counts[count] += count
    assert .48 < counts[2] / counts.total() < .52
    for players in range(3,9):
        assert .065 < counts[players] / counts.total() < .1
    assert player_count(LeagueConfig(player_counts=(4,)), {'id':'single','done':0}) == 4


def test_fast_placement_then_cumulative_evidence_controls_late_upsets():
    result = {'entrant_ids':['a','random'], 'elimination_ticks':[None,10]}
    new, evidence = {}, {}
    for index in range(6):
        games = {'a':index,'random':index}
        current = score(new, {**result,'rating_policy':POLICY}, 'random', 32, games, evidence)
        new = {key:c['after'] for key,c in current.items()}
    # Six wins, with half-win/half-loss smoothing, identify odds of 13:1.
    assert new['a'] == pytest.approx(math.log(13)/LOGISTIC_SLOPE, abs=1e-6)
    assert new['random'] == 0
    # 900 wins + 100 losses identify a 90% pair score, regardless of recency.
    evidence = {('a','random'):(1000.,900.)}
    rating = math.log(900.5/100.5)/LOGISTIC_SLOPE
    loss = {**result,'rating_policy':POLICY,'elimination_ticks':[10,None]}
    change = score({'a':rating}, loss, 'random', 32, {'a':1000}, evidence)['a']
    assert 490 < rating < 500
    assert -3 < change['delta'] < 0
    # Missing policy means a legacy record, so old replay/rebuild is exact.
    assert score(new, result, 'random', 32, games) == rate(new, result, 'random', 32, games)
    assert score({}, {**result,'rating_policy':LEGACY_POLICY}, 'random', 32, {})['a']['delta'] > 400


def test_joint_fit_recovers_strengths_and_freezes_inactive_players():
    truth = {'wait':0.,'a':200.,'b':500.,'c':-100.}
    evidence = {(a,b):(10000.,10000.*expected(truth[a],truth[b]))
                for a in truth for b in truth if a < b}
    fitted = fit({}, truth, 'wait', evidence)
    assert fitted == pytest.approx(truth, abs=1)
    assert expected(500,0) == pytest.approx(.9)
    assert set(fit(truth, ['a','b'], 'wait', evidence)) == {'a','b'}


def test_rotations_share_evidence_and_opposing_results_balance():
    original = {'entrant_ids':['a','wait'],'elimination_ticks':[None,1], 'rating_policy':POLICY}
    one = score({}, original, 'wait', 32, {})['a']['after']
    ratings, evidence = {}, {}
    for index in range(8):
        changes = score(ratings, {**original,'rating_group_size':8}, 'wait', 32, {}, evidence)
        ratings.update({key:c['after'] for key,c in changes.items()})
    assert evidence['a','wait'] == pytest.approx((1.,1.))
    # Online steps depend slightly on ordering, but eight rotations contribute
    # one sample, not eight independent wins.
    assert .8*one < ratings['a'] < 1.3*one
    loss = score(ratings, {**original,'elimination_ticks':[1,None]}, 'wait', 32, {}, evidence)
    assert loss['a']['after'] < ratings['a']


def test_new_outcomes_cannot_inherit_historical_refit_direction():
    ratings = {'a':500.,'b':0.,'idle':1000.}
    # Old history disagrees with the current ratings. A global/cumulative refit
    # would pull the winner down, which must never leak into this game's delta.
    evidence = {('a','b'):(1000.,500.), ('a','idle'):(500.,0.)}
    result = {'entrant_ids':['a','b'],'elimination_ticks':[None,1], 'rating_policy':POLICY}
    win = score(ratings, result, 'wait', 32, {}, dict(evidence))
    loss = score(ratings, {**result,'elimination_ticks':[1,None]}, 'wait', 32, {}, dict(evidence))
    assert set(win) == {'a','b'}
    assert win['a']['delta'] > 0 and win['b']['delta'] < 0
    assert loss['a']['delta'] < 0 and loss['b']['delta'] > 0
    probability = expected(ratings['a'],ratings['b'])
    for key in win:
        # Only the base online step has zero conditional expectation. The
        # bounded historical correction deliberately repairs earlier bias.
        assert probability*win[key]['online_delta']+(1-probability)*loss[key]['online_delta'] == pytest.approx(0, abs=1e-10)
        for change in (win[key], loss[key]):
            assert change['delta']*change['online_delta'] > 0
            assert abs(change['delta']) >= abs(change['online_delta'])-1e-8
            assert abs(change['delta']) <= change['update_limit']+1e-8
    assert ratings == {'a':500.,'b':0.,'idle':1000.}


def test_all_ranks_use_current_residual_and_late_upsets_stay_bounded():
    result = {'entrant_ids':['a','b','c'],'elimination_ticks':[None,2,1], 'rating_policy':POLICY}
    changes = score({'a':900.,'b':-100.,'c':300.}, result, 'wait', 32, {}, {})
    assert changes['a']['delta'] > 0 and changes['c']['delta'] < 0
    for change in changes.values():
        assert change['delta']*(change['expected_rank']-change['rank']) >= 0
    upset = {'entrant_ids':['a','wait'],'elimination_ticks':[1,None], 'rating_policy':POLICY}
    change = score({'a':2500.}, upset, 'wait', 32, {}, {('a','wait'):(1000.,1000.)})['a']
    # Even a badly miscalibrated historical rating cannot jump to the full
    # fitted estimate in one match. Total movement is capped at two base steps.
    assert -12 < change['delta'] < 0
    assert abs(change['delta']) <= change['update_limit']+1e-8


def test_late_sparse_reference_corrects_origin_after_many_relative_games():
    def run(policy):
        ratings, counts, evidence = {}, Counter(), {}

        def play(ids, ticks):
            changes = score(ratings, dict(entrant_ids=ids, elimination_ticks=ticks,
                                         rating_policy=policy), 'wait', 32, counts, evidence)
            assert set(changes) == set(ids)
            for key, change in changes.items():
                ratings[key], counts[key] = change['after'], change['rated_games']

        # No anchor information initially: 90% head-to-head fixes only the
        # difference. Many relative games must not freeze the absolute origin.
        for index in range(1000):
            play(['a','b'], [1,None] if index % 10 == 9 else [None,1])
        for _ in range(40):
            before = ratings['a']
            play(['b','wait'], [None,None])
            assert ratings['a'] == before  # No global translation on calibration.
            for index in range(50):
                play(['a','b'], [1,None] if index % 10 == 9 else [None,1])
        return ratings

    old, fixed = run(ONLINE_POLICY), run(POLICY)
    assert old['b'] < -200  # Reproduces the original zero-point freeze.
    assert fixed['wait'] == 0
    assert fixed['b'] == pytest.approx(0, abs=10)
    assert fixed['a'] == pytest.approx(500, abs=10)


def test_multiplayer_averages_each_opponent_strategy_and_is_seat_invariant():
    base = {'entrant_ids':['a','b','c'], 'elimination_ticks':[None,3,5], 'rating_policy':POLICY}
    duplicated = {'entrant_ids':['a','b','b','b','c'], 'elimination_ticks':[None,3,3,3,5], 'rating_policy':POLICY}
    ratings = {'a':90,'b':0,'c':20}
    expected = score(ratings, base, 'b', 32, {})
    actual = score(ratings, duplicated, 'b', 32, {})
    assert actual == expected
    reverse = {**duplicated,'entrant_ids':list(reversed(duplicated['entrant_ids'])),
               'elimination_ticks':list(reversed(duplicated['elimination_ticks']))}
    assert score(ratings, reverse, 'b', 32, {}) == expected
    self_play = {'entrant_ids':['a']*8,'elimination_ticks':[1,2,3,4,5,6,7,None],'rating_policy':POLICY}
    assert score(ratings, self_play, 'b', 32, {})['a']['after'] == ratings['a']


def test_automatic_duels_are_informative_and_new_rules_survive_rebuild(tmp_path):
    store = LeagueStore.create(tmp_path/'league', LeagueConfig(player_counts=(2,), max_ticks=1))
    a, _ = store.add({'name':'a','kind':'baseline','policy':'aggressive'}, games=2)
    for block in range(30):
        plans = eligible_schedule(store, {'entrant_id':a['entrant_id'],'id':'duels','done':block,'budget':40})
        assert all(len(set(p['entrant_ids'])) == 2 and p['rating_policy']==POLICY for p in plans)
    # Both versions can coexist in the same archive without retroactively
    # applying current weights to old snapshots or persisted pending plans.
    for game_id, policy in [('old',None),('new',POLICY)]:
        plan = {'game_id':game_id,'entrant_ids':[a['entrant_id'],store.anchor]}
        if policy:
            plan['rating_policy'] = policy
        store.insert_plans([plan])
        store.complete_result({**plan,'winner_seat':0,'elimination_ticks':[None,10],
                               'ticks':10,'kills':[1,0]}, b'')
    before = store.leaderboard()
    records = [store.replay(key) for key in ('old','new')]
    assert store.rebuild() == before
    assert [store.replay(key) for key in ('old','new')] == records
