"""Scalar and vector runners publish one match/replay protocol."""
import gzip
import json

import numpy as np
import pytest

from generals import MapConfig
from generals.protocol.rules import RULES_VERSION
from generals.core.mapgen import generate_map
from generals.core.recording import state_digest, state_to_dict
from generals.core.state import compute_stats
from generals_arena.matches.results import aggregate
from generals_arena.matches.results import build_result
from generals_arena.matches.replay import encode_replay
from generals_arena.matches.results import seat_outcome


def test_build_result_preserves_plan_and_owns_final_arrays():
    state = generate_map(MapConfig(25, 25, 2, 0, 0), 42)
    stats = compute_stats(state)
    initial = state_digest(state)
    plan = {"game_id": "game", "entrant_ids": ["a", "b"], "map_seed": 42,
            "evaluation_block": 7, "rating_group_size": 2}
    invalid = np.array([0, 1])
    result = build_result(plan, state=state, stats=stats, elimination_ticks=[None, None],
                          invalid_actions=invalid, initial_digest=initial)
    assert {key: result[key] for key in plan} == plan
    assert result["winner_seat"] is None and result["reason"] == "tick_limit"
    assert result["initial_digest"] == result["final_digest"] == initial
    assert result["final_stats"] == {name: getattr(stats, name).tolist()
                                     for name in ("army", "territory", "cities")}
    assert result["kills"] == [0, 0] and "execution_policy" not in result
    invalid[1] = 9
    state.alive[1] = False
    assert result["invalid_actions"] == [0, 1] and result["final_alive"] == [True, True]
    assert json.loads(json.dumps(result)) == result


def test_build_result_accepts_batched_stats_dictionary():
    state = generate_map(MapConfig(25, 25, 2, 0, 0), 42)
    state.terminated, state.winner_id, state.tick = True, 0, 5
    state.alive[1] = False
    state.end_reason = 1
    result = build_result({}, state=state, stats={name: np.array([3, 0])
                          for name in ("army", "territory", "cities")},
                          elimination_ticks=[None, np.int64(5)], invalid_actions=[0, 0],
                          initial_digest="initial")
    assert result["winner_seat"] == 0 and result["reason"] == "capture"
    assert result["elimination_ticks"] == [None, 5] and result["kills"] == [0, 0]


@pytest.mark.parametrize("forfeits", [(), [{"seat": 1, "tick": 2}]])
def test_replay_encoding_is_deterministic_and_pins_current_rules(forfeits):
    state = generate_map(MapConfig(25, 25, 2, 0, 0), 42)
    actions, digests = [[[0, 0, 0, 0], [0, 0, 0, 0]]], ["digest"]
    blob = encode_replay(state, actions, digests, forfeits)
    assert blob == encode_replay(state_to_dict(state), actions, digests, forfeits)
    trace = json.loads(gzip.decompress(blob))
    assert trace == {"schema": 3, "rules": RULES_VERSION, "initial": state_to_dict(state),
                     "actions": actions, "digests": digests,
                     **({"forfeits": list(forfeits)} if forfeits else {})}
    with pytest.raises(ValueError, match="same length"):
        encode_replay(state, actions, [])


def test_outcomes_distinguish_eliminated_players_at_tick_cap():
    row = {"entrant_ids": ["a", "b", "c"], "final_alive": [True, False, True],
           "winner_seat": None, "ticks": 20}
    assert [seat_outcome(row, seat) for seat in range(3)] == ["draw", "loss", "draw"]
    assert aggregate([row], "b") == {"games": 1, "wins": 0, "losses": 1, "draws": 0, "mean_ticks": 20.}
    row["winner_seat"] = 0
    assert [seat_outcome(row, seat) for seat in range(3)] == ["win", "loss", "loss"]


def test_aggregate_counts_matches_instead_of_duplicate_policy_seats():
    rows = [{"entrant_ids": ["a", "b", "a"], "final_alive": [True, False, False],
             "winner_seat": 0, "ticks": 10},
            {"entrant_ids": ["b", "c"], "final_alive": [True, False],
             "winner_seat": 0, "ticks": 20}]
    assert aggregate(rows, "a") == {"games": 1, "wins": 1, "losses": 0, "draws": 0, "mean_ticks": 10.}
    assert aggregate(rows, "b") == {"games": 2, "wins": 1, "losses": 1, "draws": 0, "mean_ticks": 15.}
    assert aggregate([], "missing")["games"] == 0
    with pytest.raises(ValueError, match="entrant or a seat"):
        aggregate(rows, "a", seat=0)
    with pytest.raises(ValueError, match="requires"):
        aggregate(rows)
    assert aggregate([rows[0]], seat=2)["losses"] == 1
    with pytest.raises(ValueError, match="requires"):
        aggregate([{**rows[0], "candidate_seat": 2, "outcome": "win"}])
