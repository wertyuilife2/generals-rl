"""Array traces preserve canonical actions, state hashes and game-ID selection."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from generals.core.batch import pack_states
from generals.core.engine import CoreEngine
from generals.core.recording import replay, state_digest, state_to_dict
from generals.core.state import Action, Direction, GameState, MapLayout
from generals_rl.eval_trace import (
    decode_joint_actions, record_tick_arrays, state_digests_from_arrays,
)
from generals_rl.evaluate_builtin import GameSpec, SelectedGameTraces


def tiny_state():
    shape = (2, 3)
    structure = np.zeros(shape, dtype=np.uint8)
    owner = np.full(shape, -1, dtype=np.int16)
    army = np.zeros(shape, dtype=np.int64)
    for player, cell in enumerate((0, 1)):
        structure.flat[cell], owner.flat[cell], army.flat[cell] = 2, player, 1
    return GameState(MapLayout(np.zeros(shape, dtype=np.uint8)), structure, owner, army,
                     np.ones(2, dtype=bool), np.array([0, 1], dtype=np.int32))


def candidate(state, capacity=8):
    return {"initial_state": state_to_dict(state), "ticks": 0,
            "actions": np.empty((capacity, 2, 4), dtype=np.int64),
            "digests": np.empty((capacity, 32), dtype=np.uint8)}


class ScalarReferenceTraces(SelectedGameTraces):
    """Test-only scalar reference for the production array recorder."""

    def on_tick(self, spec, actions, state, moves, done):
        entry = self.candidates.get(spec.game_id)
        if entry is None:
            return
        index = entry["ticks"]
        if state.tick != index + 1 or index >= self.max_ticks:
            raise RuntimeError("selected trace skipped or repeated a tick")
        entry["actions"][index] = [[int(a.kind), a.source, int(a.direction), int(a.mode)] for a in actions]
        entry["digests"][index] = np.frombuffer(bytes.fromhex(state_digest(state)), dtype=np.uint8)
        entry["ticks"] += 1


class ArrayTraceTests(unittest.TestCase):
    def test_actions_match_scalar_decode_including_wait_and_large_source(self):
        encoded = np.array([[0, 1], [2, 8], [9, 72], [800_000_001, 800_000_008]])
        decoded = decode_joint_actions(encoded)
        expected = [[[int(a.kind), a.source, int(a.direction), int(a.mode)]
                     for a in map(Action.decode, row)] for row in encoded]
        np.testing.assert_array_equal(decoded, expected)
        for invalid in (np.array([0, 1]), np.zeros((2, 2), dtype=np.float32),
                        np.array([[0, -1]]), np.array([[0, 2**64 - 1]], dtype=np.uint64)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    decode_joint_actions(invalid)

    def test_hashes_match_core_for_backend_dtypes_and_endianness(self):
        first, second = tiny_state(), tiny_state()
        first.tick, second.tick = 1999, 2000
        first.army.flat[0] = 16_777_219
        second.army.flat[1] = 1_300_000_007
        # A different topology must contribute its own topology ID.
        terrain = second.layout.terrain.copy()
        terrain[1, 2] = 1
        second.layout = MapLayout(terrain)
        arrays, _, _ = pack_states([first, second])
        topologies = [second.layout.topology_id.encode(), first.layout.topology_id.encode()]
        expected = [state_digest(second), state_digest(first)]
        for dtype in (np.int32, np.int64, ">i8"):
            with self.subTest(dtype=dtype):
                arrays["army"] = arrays["army"].astype(dtype)
                # Also accept board-shaped, noncontiguous host views.
                for name in ("structure", "owner", "army"):
                    values = arrays[name].reshape(2, 2, 3)
                    backing = np.empty((2, 2, 6), dtype=values.dtype)
                    backing[..., ::2] = values
                    arrays[name] = backing[..., ::2]
                actual = state_digests_from_arrays(arrays, topologies, [1, 0])
                self.assertEqual([row.tobytes().hex() for row in actual], expected)

    def test_terminal_and_truncated_scalar_suffixes_match_core(self):
        terminal = tiny_state()
        engine = CoreEngine(terminal)
        terminal.army.flat[0] = 40
        engine.step([Action.move(0, Direction.RIGHT), Action.wait()])
        truncated = tiny_state()
        truncated.tick = 2000  # Time limit is not a game-rule termination.
        arrays, _, _ = pack_states([terminal, truncated])
        actual = state_digests_from_arrays(arrays,
                  [state.layout.topology_id.encode() for state in (terminal, truncated)], [0, 1])
        self.assertTrue(terminal.terminated)
        self.assertFalse(truncated.terminated)
        self.assertEqual([row.tobytes().hex() for row in actual],
                         [state_digest(terminal), state_digest(truncated)])
        self.assertEqual(state_digests_from_arrays(arrays, [], []).shape, (0, 32))

    def test_record_rejects_skipped_repeated_and_capacity_exhausted_ticks(self):
        state = tiny_state()
        spec = SimpleNamespace(game_id=12)
        entries = {12: candidate(state, capacity=1)}
        arrays, _, _ = pack_states([state])
        arrays["tick"][0] = 2
        with self.assertRaisesRegex(RuntimeError, "skipped or repeated"):
            record_tick_arrays(entries, [spec], np.zeros((1, 2), dtype=np.int64), arrays, [0])
        self.assertEqual(entries[12]["ticks"], 0)
        arrays["tick"][0] = 1
        record_tick_arrays(entries, [spec], np.zeros((1, 2), dtype=np.int64), arrays, [0])
        with self.assertRaisesRegex(RuntimeError, "skipped or repeated"):
            record_tick_arrays(entries, [spec], np.zeros((1, 2), dtype=np.int64), arrays, [0])
        arrays["tick"][0] = 2
        with self.assertRaisesRegex(RuntimeError, "skipped or repeated"):
            record_tick_arrays(entries, [spec], np.zeros((1, 2), dtype=np.int64), arrays, [0])

    def test_generated_trace_bytes_replay_and_first_game_id_match_original(self):
        # The lower-ID win finishes after the higher-ID win; completion order
        # must not change which actual game gets recorded.
        specs = [GameSpec(game_id, "aggressive", index, index + 1, index + 21, 0)
                 for index, game_id in enumerate((0, 7, 3))]
        states = [tiny_state() for _ in specs]
        states[0].army.flat[0] = states[1].army.flat[0] = 40
        states[2].army.flat[1] = 40
        engines = [CoreEngine(state) for state in states]
        with tempfile.TemporaryDirectory() as directory:
            metadata = {"checkpoint_sha256": "synthetic-checkpoint"}
            old = ScalarReferenceTraces(Path(directory) / "original", max_ticks=8,
                                       opponents=("aggressive",), metadata=metadata)
            new = SelectedGameTraces(Path(directory) / "arrays", max_ticks=8,
                                    opponents=("aggressive",), metadata=metadata)
            old.on_start(specs, states)
            new.on_start(specs, states)
            active, records = {0, 1, 2}, []
            for tick in range(1, 5):
                lanes = sorted(active)
                joint = np.zeros((3, 2), dtype=np.int64)
                for lane in lanes:
                    if tick == (4, 2, 3)[lane]:
                        seat = int(lane == 2)
                        joint[lane, seat] = Action.move(seat, Direction.LEFT if seat else Direction.RIGHT).encode()
                    actions = tuple(map(Action.decode, joint[lane]))
                    result = engines[lane].step(actions)
                    old.on_tick(specs[lane], actions, states[lane], None, None)
                    if result.terminated:
                        records.append({"game_id": specs[lane].game_id, "opponent": "aggressive",
                                        "outcome": "win" if result.winner_id == 0 else "loss",
                                        "learner_seat": 0, "ticks": result.tick})
                        active.remove(lane)
                arrays, _, _ = pack_states(states)
                arrays["army"] = arrays["army"].astype(np.int32)
                record_tick_arrays(new.candidates, specs, joint, arrays, lanes)
            self.assertEqual([row["game_id"] for row in records], [7, 3, 0])
            old.finish_batch(records)
            new.finish_batch(records)
            self.assertEqual(new.selected[("aggressive", "win")]["game_id"], 0)
            self.assertEqual(new.selected[("aggressive", "loss")]["game_id"], 3)
            for outcome, lane in (("win", 0), ("loss", 2)):
                old_path = Path(old.selected[("aggressive", outcome)]["path"])
                new_path = Path(new.selected[("aggressive", outcome)]["path"])
                self.assertEqual(new_path.read_bytes(), old_path.read_bytes())
                self.assertEqual(state_digest(replay(new_path)), state_digest(states[lane]))


if __name__ == "__main__":
    unittest.main()
