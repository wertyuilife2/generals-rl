"""Branch ownership and native CPU parity of the future search integration seam."""
import unittest

import numpy as np

from generals.backends.cpu import CPUEnv
from generals.core.batch import pack_states
from generals.core.state import Action, Direction, GameState, MapLayout, MoveMode, Structure
from generals_rl.search import JointTickSimulator


def make_state(height=3, width=4, generals=(0, 11)):
    shape = (height, width)
    structure = np.zeros(shape, dtype=np.uint8)
    owner = np.full(shape, -1, dtype=np.int16)
    army = np.zeros(shape, dtype=np.int64)
    for player_id, cell in enumerate(generals):
        structure.flat[cell] = Structure.GENERAL
        owner.flat[cell] = player_id
        army.flat[cell] = 1
    return GameState(
        MapLayout(np.zeros(shape, dtype=np.uint8)), structure, owner, army,
        np.ones(len(generals), dtype=bool), np.array(generals, dtype=np.int32),
    )


class JointTickSimulatorTests(unittest.TestCase):
    def assert_same_state(self, actual, expected):
        actual_arrays, actual_width, actual_height = pack_states([actual])
        expected_arrays, expected_width, expected_height = pack_states([expected])
        self.assertEqual((actual_width, actual_height), (expected_width, expected_height))
        for name in actual_arrays:
            with self.subTest(field=name):
                np.testing.assert_array_equal(actual_arrays[name], expected_arrays[name])

    def assert_native_parity(self, initial, joint_actions, *, episode_ticks=0):
        """Compare the independent Python rule path with the native CPU step."""
        env = CPUEnv.from_states(
            [initial], threads=1,
            auto_reset=False, episode_ticks=episode_ticks,
        )
        simulator = JointTickSimulator(episode_ticks=episode_ticks)
        parent = simulator.snapshot(initial)
        try:
            for submitted in joint_actions:
                previous = parent.copy()
                encoded = np.array([
                    [action.encode() if isinstance(action, Action) else action for action in submitted]
                ], dtype=np.int64)
                native = env.step(encoded)
                env.check_errors()
                child, info = simulator.advance(parent, encoded[0])
                self.assert_same_state(parent, previous)
                self.assert_same_state(child, env.to_states()[0])
                np.testing.assert_array_equal(native.done[0], [info.terminated, info.truncated])
                reference_stats = np.stack([
                    info.result.stats.army, info.result.stats.territory, info.result.stats.cities,
                ], axis=1)
                np.testing.assert_array_equal(native.stats[0], reference_stats)
                active_rows = native.moves[0][native.moves[0, :, 0].astype(bool)]
                self.assertEqual(len(active_rows), len(info.result.moves))
                for native_move, move in zip(active_rows, info.result.moves):
                    self.assertEqual(bool(native_move[1]), move.valid)
                    self.assertEqual(int(native_move[4]), move.moved)
                    self.assertEqual(bool(native_move[5]), move.captured)
                    self.assertEqual(int(native_move[6]), -1 if move.eliminated is None else move.eliminated)
                    self.assertEqual(bool(native_move[7]), move.target_owned)
                parent = child
        finally:
            env.close()

    def test_snapshots_and_sibling_branches_own_their_dynamic_arrays(self):
        state = make_state()
        state.army.flat[0] = 9
        simulator = JointTickSimulator()
        snapshot = simulator.snapshot(state)
        before = state.copy()
        left, _ = simulator.advance(snapshot, [Action.move(0, Direction.RIGHT), Action.wait()])
        right, _ = simulator.advance(snapshot, [Action.move(0, Direction.DOWN, MoveMode.HALF), Action.wait()])
        self.assert_same_state(snapshot, before)
        self.assert_same_state(state, before)
        self.assertEqual((left.army.flat[1], right.army.flat[4]), (8, 4))
        for name in ("structure", "owner", "army", "alive", "general_pos"):
            self.assertFalse(np.shares_memory(getattr(snapshot, name), getattr(state, name)))
            self.assertFalse(np.shares_memory(getattr(left, name), getattr(right, name)))
        self.assertIs(left.layout, snapshot.layout)
        left.army.flat[0] = 123
        self.assertEqual(right.army.flat[0], 5)
        self.assertEqual(snapshot.army.flat[0], 9)

    def test_pre_growth_joint_actions_and_growth_cycles_match_cpu(self):
        state = make_state()
        state.army.flat[0] = 1
        state.owner.flat[1] = 0
        state.structure.flat[1] = Structure.CITY
        state.army.flat[1] = 7
        state.owner.flat[2] = 0
        state.owner.flat[3] = -2
        state.structure.flat[3] = Structure.CITY
        state.army.flat[3] = 17
        # On tick 2, the general's growth follows selection, so this one-army move is discarded.
        sequence = [[0, 0], [Action.move(0, Direction.DOWN), 0]]
        sequence.extend([[0, 0]] * 49)
        self.assert_native_parity(state, sequence)
        second, _ = JointTickSimulator().advance(state, [0, 0])
        second, _ = JointTickSimulator().advance(second, sequence[1])
        self.assertEqual(second.army.flat[4], 0)
        self.assertEqual(second.owner.flat[4], -1)

    def test_ordered_capture_inheritance_and_dead_slots_match_cpu(self):
        state = make_state(generals=(0, 1, 11))
        state.army.flat[0] = 10
        state.army.flat[1] = 2
        state.owner.flat[2] = 1
        state.army.flat[2] = 5
        state.army.flat[11] = 4
        sequence = [
            [Action.move(0, Direction.RIGHT), Action.move(1, Direction.LEFT), Action.move(11, Direction.LEFT)],
            # Already eliminated slots are ignored even if they contain a MOVE.
            [0, Action.move(1, Direction.LEFT), 0],
        ]
        self.assert_native_parity(state, sequence)
        child, info = JointTickSimulator().advance(state, sequence[0])
        self.assertFalse(info.done)
        self.assertEqual(info.result.eliminated, (1,))
        self.assertEqual([move.player_id for move in info.result.moves], [0, 1, 2])
        self.assertEqual(child.army.flat[2], 3)
        self.assertEqual(child.owner.flat[2], 0)

    def test_natural_terminal_freeze_and_terminal_precedence_match_cpu(self):
        state = make_state(height=2, width=3, generals=(0, 1))
        state.army.flat[0] = 10
        state.army.flat[1] = 2
        capture = [Action.move(0, Direction.RIGHT), Action.move(1, Direction.LEFT)]
        self.assert_native_parity(state, [capture, [0, 0]], episode_ticks=1)
        child, info = JointTickSimulator(episode_ticks=1).advance(state, capture)
        self.assertTrue(info.terminated)
        self.assertFalse(info.truncated)
        frozen, info = JointTickSimulator(episode_ticks=1).advance(child, [0, 0])
        self.assert_same_state(frozen, child)
        self.assertTrue(info.done)
        self.assertFalse(info.advanced)

    def test_truncation_freezes_at_absolute_tick_without_autoreset(self):
        state = make_state()
        state.tick = 48
        sequence = [[0, 0]] * 4
        self.assert_native_parity(state, sequence, episode_ticks=50)
        simulator = JointTickSimulator(episode_ticks=50)
        child, info = simulator.advance(state, [0, 0])
        self.assertFalse(info.done)
        child, info = simulator.advance(child, [0, 0])
        self.assertEqual(child.tick, 50)
        self.assertTrue(info.truncated)
        self.assertFalse(info.terminated)
        frozen, info = simulator.advance(child, [0, 0])
        self.assert_same_state(frozen, child)
        self.assertFalse(info.advanced)
        resumed, info = JointTickSimulator().advance(child, [0, 0])
        self.assertEqual(resumed.tick, 51)
        self.assertTrue(info.advanced)

    def test_invalid_but_well_formed_moves_match_cpu(self):
        state = make_state()
        sequence = [
            [Action.move(0, Direction.UP), Action.move(0, Direction.RIGHT)],
            [Action.move(99, Direction.LEFT), Action.wait()],
            [Action.move(0, Direction.RIGHT, MoveMode.HALF), Action.wait()],
        ]
        self.assert_native_parity(state, sequence)

    def test_malformed_search_inputs_fail_without_changing_parent(self):
        state = make_state()
        before = state.copy()
        simulator = JointTickSimulator()
        for actions in ([0], [-1, 0], [True, 0], [1.5, 0], [2**63, 0]):
            with self.subTest(actions=actions), self.assertRaises((ValueError, TypeError)):
                simulator.advance(state, actions)
            self.assert_same_state(state, before)
        for episode_ticks in (-1, 2**63, True, 1.5):
            with self.subTest(episode_ticks=episode_ticks), self.assertRaises((ValueError, TypeError)):
                JointTickSimulator(episode_ticks=episode_ticks)

if __name__ == "__main__":
    unittest.main()
