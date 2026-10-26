"""Joint decisions and elimination lifecycle through the active Arena executor."""
from dataclasses import asdict

import numpy as np
import pytest

from generals import Action, Direction, MapConfig
from generals.core.state import GameState, MapLayout
from generals_arena.matches.game import RecordedGame
from generals_arena.matches.replay import verified_replay
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.rating import ranks


class Probe:
    def __init__(self, action=None):
        self.action = action

    def reset(self, player_id, seed):
        self.player_id, self.seed = player_id, seed
        self.observations, self.moves = [], []

    def act(self, observation):
        self.observations.append(observation)
        return self.action(observation) if self.action is not None else Action.wait()

    def on_result(self, result):
        self.moves.append(result)


@pytest.fixture
def game_factory(monkeypatch):
    games = []

    def create(state, controllers, max_ticks):
        monkeypatch.setattr('generals_arena.matches.game.generate_map', lambda *args: state.copy())
        map_config = MapConfig(state.layout.width, state.layout.height, state.player_count, 0, 0)
        plan = {'game_id': 'fixture', 'entrant_ids': [str(seat) for seat in range(state.player_count)],
                'controller_seeds': list(range(10, 10 + state.player_count)),
                'map': asdict(map_config), 'map_seed': 17}
        config = LeagueConfig(player_counts=(state.player_count,), min_side=3, max_side=3,
                              side_cap=3, max_ticks=max_ticks)
        game = RecordedGame(plan, config, controllers)
        games.append(game)
        return game

    yield create
    for game in games:
        game.engine.close()


def adjacent_state():
    owner = np.full((3, 3), -1, dtype=np.int16)
    structure = np.zeros((3, 3), dtype=np.uint8)
    army = np.zeros((3, 3), dtype=np.int64)
    owner.flat[:2], structure.flat[:2], army.flat[:2] = [0, 1], 2, [10, 2]
    return GameState(MapLayout(np.zeros((3, 3), dtype=np.uint8)), structure, owner, army,
                     np.ones(2, dtype=np.bool_), np.array([0, 1], dtype=np.int32))


def four_player_state():
    return GameState(MapLayout(np.zeros((1, 4), dtype=np.uint8)),
                     np.full((1, 4), 2, dtype=np.uint8), np.arange(4, dtype=np.int16).reshape(1, 4),
                     np.array([[1, 1, 1, 30]], dtype=np.int64), np.ones(4, dtype=np.bool_),
                     np.arange(4, dtype=np.int32))


def test_joint_pre_growth_decisions_and_capture_at_tick_limit(game_factory):
    state = adjacent_state()
    attack, wait = Probe(lambda obs: Action.move(0, Direction.RIGHT)), Probe()
    game = game_factory(state, [attack, wait], 1)
    game.advance()
    result = game.result()
    assert game.done and game.state.terminated
    assert (result['winner_seat'], result['ticks'], result['reason']) == (0, 1, 'capture')
    assert [controller.observations[0].tick for controller in (attack, wait)] == [0, 0]
    assert wait.observations[0].army.flat[1] == 2
    assert attack.observations[0].stats.cities[1] == -1
    assert not attack.observations[0].owner.flags.writeable
    assert len(attack.moves) == 1 and not wait.moves
    assert state.tick == 0 and state.army.flat[0] == 10
    assert len(list(verified_replay(game.replay_blob(), result))) == 2


def test_tick_limit_and_reset_do_not_reuse_previous_game_memory(game_factory):
    controllers = [Probe(), Probe()]
    first = game_factory(adjacent_state(), controllers, 2)
    while not first.done:
        first.advance()
    first_result = first.result()
    second = game_factory(adjacent_state(), controllers, 2)
    while not second.done:
        second.advance()
    assert first_result == second.result()
    assert first_result['reason'] == 'tick_limit' and first_result['winner_seat'] is None
    assert len(controllers[0].observations) == 2
    assert [controller.seed for controller in controllers] == [10, 11]


def test_eliminated_players_stop_acting_and_high_seat_can_win(game_factory):
    controllers = [Probe(), Probe(), Probe(), Probe(lambda obs: Action.move(3 - obs.tick, Direction.LEFT))]
    game = game_factory(four_player_state(), controllers, 3)
    while not game.done:
        game.advance()
    result = game.result()
    assert game.state.terminated and result['winner_seat'] == 3
    assert result['elimination_ticks'] == [3, 2, 1, None]
    assert result['final_alive'] == [False, False, False, True]
    assert [len(controller.observations) for controller in controllers] == [3, 2, 1, 3]
    assert result['invalid_actions'] == [0, 0, 0, 0]
    assert len(list(verified_replay(game.replay_blob(), result))) == 4


def test_multiplayer_limit_preserves_elimination_order_for_current_rank_rating(game_factory):
    game = game_factory(four_player_state(),
                        [Probe(), Probe(), Probe(), Probe(lambda obs: Action.move(3 - obs.tick, Direction.LEFT))], 2)
    while not game.done:
        game.advance()
    result = game.result()
    assert not game.state.terminated and result['winner_seat'] is None
    assert result['reason'] == 'tick_limit'
    assert result['final_alive'] == [True, False, False, True]
    assert result['elimination_ticks'] == [None, 2, 1, None]
    assert ranks(result) == [1.5, 3, 4, 1.5]
