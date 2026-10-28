"""CUDA rules are checked against the independent CPU implementation.

The tests use explicit externally chosen actions, so a different backend random
number generator cannot conceal a rule difference. They intentionally require a
real CUDA device: installing only the CPU package must still run the CPU suite.
"""

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import unittest

import numpy as np

from generals.core.config import MapConfig
from generals.core.engine import CoreEngine
from generals.core.mapgen import generate_map
from generals.core.observation import legal_action_mask, observe
from generals.core.state import (
    Action, Direction, EMPTY, GameState, MapLayout, MoveMode,
    NEUTRAL, Structure, Terrain, compute_stats,
)

try:
    import torch
except ImportError:
    torch = None

_CUDA_AVAILABLE = (
    torch is not None
    and torch.cuda.is_available()
    and sys.platform == "linux"
)

if _CUDA_AVAILABLE:
    from generals.backends.gpu.kernel import compiler
    _CUDA_AVAILABLE = compiler() is not None


def make_state(height=3, width=7, generals=None, terrain=None):
    """Small hand-authored fixtures do not depend on the GPU map generator."""
    generals = [0, height * width - 1] if generals is None else generals
    terrain = np.zeros((height, width), dtype=np.uint8) if terrain is None else terrain
    structure = np.zeros((height, width), dtype=np.uint8)
    owner = np.full((height, width), EMPTY, dtype=np.int16)
    army = np.zeros((height, width), dtype=np.int64)
    for pid, cell in enumerate(generals):
        structure.flat[cell] = Structure.GENERAL
        owner.flat[cell] = pid
        army.flat[cell] = 1
    return GameState(MapLayout(terrain), structure, owner, army,
                     np.ones(len(generals), dtype=bool), np.array(generals, dtype=np.int32))


@unittest.skipUnless(_CUDA_AVAILABLE, "GPU parity tests require CUDA, PyTorch, and nvcc")
class GPUEnvParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # On a supported machine a broken GPU package must fail, not silently skip.
        from generals.backends.gpu import GPUEnv
        cls.GPUEnv = GPUEnv

    def encoded(self, actions):
        return torch.tensor([[action.encode() for action in lane] for lane in actions],
                            dtype=torch.int64, device="cuda")

    def assert_matches(self, env, states):
        env.check_errors()
        exported = env.to_states()
        self.assertEqual(len(exported), len(states))
        for lane, (actual, expected) in enumerate(zip(exported, states)):
            with self.subTest(lane=lane, tick=expected.tick):
                actual.validate()
                np.testing.assert_array_equal(actual.layout.terrain, expected.layout.terrain)
                for name in ("structure", "owner", "army", "alive", "general_pos"):
                    np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name),
                                                  err_msg=name)
                self.assertEqual(actual.tick, expected.tick)
                self.assertEqual(actual.terminated, expected.terminated)
                self.assertEqual(actual.winner_id, expected.winner_id)
                stats = compute_stats(expected)
                np.testing.assert_array_equal(
                    env.stats[lane].cpu().numpy(),
                    np.stack((stats.army, stats.territory, stats.cities), axis=-1),
                )

    def step_both(self, env, states, actions):
        env.step(self.encoded(actions))
        for state, lane_actions in zip(states, actions):
            if not state.terminated:
                CoreEngine(state, debug=True).step(lane_actions)
        self.assert_matches(env, states)

    def test_device_protocol_and_roundtrip_are_exact(self):
        states = [generate_map(MapConfig(width=25, height=25, players=3), seed)
                  for seed in (11, 37, 54)]
        env = self.GPUEnv.from_states(states)
        self.assert_matches(env, states)
        for name, dtype, shape in (
            ("terrain", torch.uint8, (3, 625)),
            ("structure", torch.uint8, (3, 625)),
            ("owner", torch.int16, (3, 625)),
            ("army", torch.int64, (3, 625)),
            ("alive", torch.bool, (3, 3)),
            ("general_pos", torch.int32, (3, 3)),
            ("tick", torch.int64, (3,)),
            ("terminated", torch.bool, (3,)),
            ("winner", torch.int32, (3,)),
        ):
            with self.subTest(field=name):
                value = env.state[name]
                self.assertEqual(value.device.type, "cuda")
                self.assertEqual(value.dtype, dtype)
                self.assertEqual(tuple(value.shape), shape)
        exported = env.to_states()
        exported[0].army.flat[states[0].general_pos[0]] = 1234
        self.assert_matches(env, states)

    def test_independent_lane_economic_cycles(self):
        states = []
        for tick in (0, 1, 24, 48, 49, 99):
            state = make_state()
            state.tick = tick
            state.owner.flat[1] = 0
            state.structure.flat[1] = Structure.CITY
            state.army.flat[1] = 7
            state.owner.flat[2] = 0  # Owned zero-army plains grow on tick 50.
            state.owner.flat[3] = NEUTRAL
            state.structure.flat[3] = Structure.CITY
            state.army.flat[3] = 17
            states.append(state)
        env = self.GPUEnv.from_states(states)
        self.step_both(env, states, [[Action.wait(), Action.wait()] for _ in states])
        for state in states:
            self.assertEqual(state.army.flat[1], 7 + (state.tick % 2 == 0) + (state.tick % 50 == 0))
            self.assertEqual(state.army.flat[2], int(state.tick % 50 == 0))
            self.assertEqual(state.army.flat[3], 17)

    def test_zero_army_moves_do_not_capture_or_wait(self):
        states, actions = [], []
        for count in (0, 1):
            for mode in MoveMode:
                for owner, defense in ((EMPTY, 0), (0, 0), (0, 7), (1, 0),
                                       (1, 7), (NEUTRAL, 0), (NEUTRAL, 40)):
                    state = make_state()
                    state.army.flat[0] = count
                    state.owner.flat[1], state.army.flat[1] = owner, defense
                    if owner == NEUTRAL:
                        state.structure.flat[1] = Structure.CITY
                    states.append(state)
                    actions.append([Action.move(0, Direction.RIGHT, mode), Action.wait()])
        env = self.GPUEnv.from_states(states)
        self.step_both(env, states, actions)
        self.assertTrue(all(state.tick == 1 for state in states))

    def test_half_moves_friendly_merges_ties_and_city_capture(self):
        states, actions = [], []
        for mode in MoveMode:
            for target_owner, defense in ((0, 3), (1, 0), (1, 5), (1, 40),
                                           (NEUTRAL, 0), (NEUTRAL, 40), (EMPTY, 0)):
                for count in (6, 9, 41, 42, 83):
                    state = make_state()
                    state.army.flat[0] = count
                    state.owner.flat[1], state.army.flat[1] = target_owner, defense
                    if target_owner == NEUTRAL:
                        state.structure.flat[1] = Structure.CITY
                    states.append(state)
                    actions.append([Action.move(0, Direction.RIGHT, mode), Action.wait()])
        env = self.GPUEnv.from_states(states)
        self.step_both(env, states, actions)

    def test_invalid_moves_and_mountain_edges(self):
        states, actions = [], []
        for action in (Action.move(21, Direction.LEFT), Action.move(0, Direction.UP),
                       Action.move(6, Direction.RIGHT), Action.move(20, Direction.LEFT),
                       Action.move(0, Direction.RIGHT)):
            terrain = np.zeros((3, 7), dtype=np.uint8)
            terrain.flat[1] = Terrain.MOUNTAIN
            state = make_state(terrain=terrain)
            state.army.flat[0] = 19
            states.append(state)
            actions.append([action, Action.wait()])
        env = self.GPUEnv.from_states(states)
        self.step_both(env, states, actions)
        self.assertTrue(all(state.army.flat[0] == 19 for state in states))

    def test_general_inheritance_and_immediate_terminal_lane_freeze(self):
        winning = make_state(2, 3, generals=[0, 1])
        winning.army.flat[0], winning.army.flat[1] = 10, 2
        for cell, count in zip((2, 3, 4, 5), (0, 1, 5, 8)):
            winning.owner.flat[cell] = 1
            winning.army.flat[cell] = count
        active = make_state(2, 3, generals=[0, 5])
        states = [winning, active]
        env = self.GPUEnv.from_states(states)
        self.step_both(env, states, [
            [Action.move(0, Direction.RIGHT), Action.move(1, Direction.LEFT)],
            [Action.wait(), Action.wait()],
        ])
        self.assertTrue(winning.terminated)
        self.assertEqual(winning.winner_id, 0)
        np.testing.assert_array_equal(winning.army.ravel()[2:], [0, 1, 3, 4])
        frozen = winning.copy()
        for _ in range(3):
            self.step_both(env, states, [[Action.wait(), Action.wait()] for _ in states])
            self.assertFalse(env.last_moves[0, :, 0].any().item(),
                             "frozen lanes must not repeat the previous capture event")
            self.assertFalse(env.last_moves[0, :, 8].any().item())  # SKIPPED reasons.
        self.assertEqual(winning.tick, frozen.tick)
        self.assertEqual(active.tick, 4)

    def test_sequential_eliminations_inherit_newly_acquired_land(self):
        state = make_state(2, 4, generals=[0, 1, 4, 5])
        state.army.flat[0], state.army.flat[4], state.army.flat[5] = 10, 30, 2
        state.owner.flat[2], state.army.flat[2] = 1, 5
        states = [state]
        env = self.GPUEnv.from_states(states)
        self.step_both(env, states, [[
            Action.move(0, Direction.RIGHT), Action.move(1, Direction.RIGHT),
            Action.move(4, Direction.UP), Action.move(5, Direction.LEFT),
        ]])
        np.testing.assert_array_equal(state.alive, [False, True, True, True])
        self.assertEqual(state.owner.flat[2], 1)
        self.assertEqual(state.army.flat[2], 5)
        self.step_both(env, states, [[Action.wait()] * 4])

    def test_random_external_actions_match_for_two_to_eight_players(self):
        rng = np.random.default_rng(731)
        for players in range(2, 9):
            with self.subTest(players=players):
                states = [generate_map(MapConfig(width=25, height=25, players=players), seed)
                          for seed in (10, 11, 12)]
                for lane, state in enumerate(states):
                    state.tick = (0, 47, 98)[lane]
                    # Enough troops to exercise fighting without thousands of idle ticks.
                    state.army.ravel()[state.general_pos] = np.arange(players) * 3 + 25
                env = self.GPUEnv.from_states(states)
                for _ in range(55):
                    actions = []
                    for state in states:
                        lane_actions = []
                        for pid in range(players):
                            if state.terminated or not state.alive[pid] or rng.random() < .1:
                                lane_actions.append(Action.wait())
                                continue
                            sources = np.flatnonzero(state.owner.ravel() == pid)
                            source = (int(rng.choice(sources)) if rng.random() < .9
                                      else int(rng.integers(state.layout.size + 2)))
                            lane_actions.append(Action.move(source, Direction(int(rng.integers(4))),
                                                            MoveMode(int(rng.integers(2)))))
                        actions.append(lane_actions)
                    self.step_both(env, states, actions)

    def test_local_and_full_observations_match_including_eliminated_player(self):
        active = make_state(3, 7)
        active.owner.flat[2], active.structure.flat[2], active.army.flat[2] = 0, Structure.CITY, 9
        active.owner.flat[17], active.structure.flat[17], active.army.flat[17] = 1, Structure.CITY, 12
        active.owner.flat[10], active.structure.flat[10], active.army.flat[10] = NEUTRAL, Structure.CITY, 40
        terminal = active.copy()
        terminal.structure.flat[0] = Structure.CITY
        terminal.owner[terminal.owner == 0] = 1
        terminal.alive[0] = False
        terminal.general_pos[0] = -1
        terminal.terminated, terminal.winner_id = True, 1
        states = [active, terminal]
        env = self.GPUEnv.from_states(states)
        for mode in ("local", "full"):
            for pid in range(2):
                actual = env.observe(pid, mode=mode)
                for lane, state in enumerate(states):
                    expected = observe(state, pid, mode)
                    with self.subTest(mode=mode, player=pid, lane=lane):
                        for name in ("terrain", "structure", "owner", "army", "visible", "alive"):
                            self.assertEqual(actual[name].device.type, "cuda")
                            np.testing.assert_array_equal(actual[name][lane].cpu().numpy(),
                                                          getattr(expected, name), err_msg=name)
                        for name in ("general_pos", "tick", "terminated"):
                            self.assertEqual(actual[name][lane].item(), getattr(expected, name))
                        self.assertEqual(actual["winner"][lane].item(),
                                         -1 if expected.winner_id is None else expected.winner_id)
                        for name in ("army", "territory", "cities"):
                            np.testing.assert_array_equal(actual["stats_" + name][lane].cpu().numpy(),
                                                          getattr(expected.stats, name))

    def test_reset_selects_lanes_and_restores_cached_initial_state(self):
        states = [generate_map(MapConfig(width=25, height=25, players=3), seed)
                  for seed in (3, 17, 24)]
        initial = [state.copy() for state in states]
        env = self.GPUEnv.from_states(states)
        for _ in range(4):
            self.step_both(env, states, [[Action.wait()] * 3 for _ in states])
        env.reset(torch.tensor([True, False, True], device="cuda"))
        states[0], states[2] = initial[0].copy(), initial[2].copy()
        self.assert_matches(env, states)
        env.reset(torch.zeros(3, dtype=torch.bool, device="cuda"))
        self.assert_matches(env, states)
        self.step_both(env, states, [[Action.wait()] * 3 for _ in states])
        env.reset()
        self.assert_matches(env, initial)

    def test_legal_action_masks_match_for_one_and_zero_army_sources(self):
        for height, width in ((3, 7), (1, 7), (7, 1)):
            terrain = np.zeros((height, width), dtype=np.uint8)
            terrain.flat[height * width // 2] = Terrain.MOUNTAIN
            active = make_state(height, width, terrain=terrain)
            active.owner.flat[1], active.army.flat[1] = 0, 0
            terminal = active.copy()
            terminal.structure.flat[0] = Structure.CITY
            terminal.owner[terminal.owner == 0] = 1
            terminal.alive[0] = False
            terminal.general_pos[0] = -1
            terminal.terminated, terminal.winner_id = True, 1
            states = [active, terminal]
            env = self.GPUEnv.from_states(states)
            for pid in range(2):
                moves, wait = env.legal_action_mask(pid)
                self.assertEqual(moves.dtype, torch.bool)
                self.assertEqual(wait.dtype, torch.bool)
                self.assertEqual(tuple(moves.shape), (2, height * width, 4, 2))
                for lane, state in enumerate(states):
                    with self.subTest(shape=(height, width), player=pid, lane=lane):
                        expected_moves, expected_wait = legal_action_mask(observe(state, pid))
                        np.testing.assert_array_equal(moves[lane].cpu().numpy(), expected_moves)
                        self.assertEqual(wait[lane].item(), expected_wait)
            moves, _ = env.legal_action_mask(0)
            self.assertFalse(moves[0, 0].any().item())
            self.assertFalse(moves[0, 1].any().item())

    def test_large_armies_preserve_int64_precision(self):
        states = []
        for mode in MoveMode:
            state = make_state()
            state.army.flat[0] = 2**54 + 5
            state.owner.flat[1], state.army.flat[1] = 0, 2**53 + 7
            states.append(state)
        env = self.GPUEnv.from_states(states)
        self.step_both(env, states, [[Action.move(0, Direction.RIGHT, mode), Action.wait()]
                                    for mode in MoveMode])
        self.step_both(env, states, [[Action.wait(), Action.wait()] for _ in states])

    def test_large_growth_merge_and_statistics_stay_exact(self):
        # GPU arithmetic has no overflow guards; retain precision tests within
        # the documented int64 range instead of expecting obsolete errors.
        states = [make_state() for _ in range(3)]
        for lane, state in enumerate(states):
            state.tick = 1
            state.army.flat[0] = 2**54 + lane + 3
            state.owner.flat[1] = 0
            state.army.flat[1] = 2**53 + 7
        env = self.GPUEnv.from_states(states)
        self.step_both(env, states, [
            [Action.wait(), Action.wait()],
            [Action.move(0, Direction.RIGHT), Action.wait()],
            [Action.move(0, Direction.RIGHT, MoveMode.HALF), Action.wait()],
        ])
        env.check_errors()
        self.assertFalse(env.errors.any().item())


@unittest.skipUnless(torch is not None, "GPU package import requires optional PyTorch dependency")
class GPUImportIsolationTests(unittest.TestCase):
    def test_core_import_does_not_load_cpu_implementation(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import generals.backends.gpu; "
             "assert not any(name == 'generals.core' or name.startswith('generals.core.') "
             "for name in sys.modules), [name for name in sys.modules if name.startswith('generals.core')]"],
            env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
