"""Regressions for interrupted recording, resumed accounting, and invalid traces."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from generals_env.recording import (
    RULES_VERSION, TraceWriter, replay, state_digest, state_from_dict, state_to_dict,
)
from generals_env.runner import Runner
from generals_env.state import Action, GameState, MapLayout, Structure


def initial_state():
    return GameState(
        MapLayout(np.zeros((1, 3), dtype=np.uint8)),
        np.array([[Structure.GENERAL, Structure.NONE, Structure.GENERAL]], dtype=np.uint8),
        np.array([[0, -1, 1]], dtype=np.int16),
        np.array([[1, 0, 1]], dtype=np.int64),
        np.ones(2, dtype=bool), np.array([0, 2], dtype=np.int32),
    )


class WaitController:
    def __init__(self):
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        return Action.wait()

    def on_result(self, result):
        pass


class FailingRecorder:
    def __init__(self):
        self.calls = 0

    def write_tick(self, result, actions, state):
        self.calls += 1
        raise OSError("simulated full disk")


class RunnerErrorTests(unittest.TestCase):
    def test_recording_failure_stops_runner_and_preserves_completed_tick(self):
        for limit in (1, 10):
            with self.subTest(max_ticks=limit):
                state = initial_state()
                controllers = [WaitController(), WaitController()]
                recorder = FailingRecorder()
                runner = Runner(state, controllers, max_ticks=limit, recorder=recorder)
                with patch("generals_env.runner.perf_counter", side_effect=[10.0, 12.0]):
                    with self.assertRaisesRegex(OSError, "simulated full disk"):
                        runner.tick()
                self.assertTrue(runner.stopped)
                self.assertEqual(runner.stop_reason, "recording_error")
                self.assertIn("simulated full disk", runner.error)
                self.assertEqual(state.tick, 1)
                self.assertEqual(runner.last_result.tick, 1)
                self.assertEqual(runner.elapsed_seconds, 2.0)
                self.assertEqual(runner.summary()["advanced_ticks"], 1)
                state.validate()
                digest = state_digest(state)
                with self.assertRaisesRegex(RuntimeError, "runner is stopped"):
                    runner.tick()
                self.assertEqual(state_digest(state), digest)
                self.assertEqual([controller.calls for controller in controllers], [1, 1])
                self.assertEqual(recorder.calls, 1)

    def test_resumed_throughput_counts_only_new_ticks(self):
        state = initial_state()
        state.tick = 5
        runner = Runner(state, [WaitController(), WaitController()], max_ticks=7)
        with patch("generals_env.runner.perf_counter", side_effect=[10.0, 10.5, 20.0, 20.5]):
            summary = runner.run()
        self.assertEqual(summary["ticks"], 7)
        self.assertEqual(summary["advanced_ticks"], 2)
        self.assertEqual(summary["elapsed_seconds"], 1.0)
        self.assertEqual(summary["ticks_per_second"], 2.0)
        self.assertEqual(summary["stop_reason"], "max_ticks")

    def test_already_stopped_resumed_run_has_zero_throughput(self):
        state = initial_state()
        state.tick = 5
        runner = Runner(state, [WaitController(), WaitController()], max_ticks=5)
        summary = runner.run()
        self.assertEqual(summary["advanced_ticks"], 0)
        self.assertEqual(summary["ticks_per_second"], 0.0)


class RecordedStateValidationTests(unittest.TestCase):
    def test_valid_state_roundtrip(self):
        state = initial_state()
        decoded = state_from_dict(state_to_dict(state))
        self.assertEqual(state_digest(decoded), state_digest(state))

    def test_lossy_numeric_or_boolean_coercion_is_rejected(self):
        changes = (
            ("tick", 0.5), ("tick", "0"), ("tick", False),
            ("terrain", [[0.1, 0, 0]]),
            ("army", [[1.5, 0, 1]]),
            ("terminated", "false"), ("terminated", 0),
            ("alive", ["true", "false"]), ("alive", [1, 1]),
            ("winner_id", False),
        )
        for field, value in changes:
            data = state_to_dict(initial_state())
            data[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                state_from_dict(data)

    def test_missing_state_fields_and_wrong_container_are_value_errors(self):
        for data in (None, [], {}, {"terrain": [[0, 0, 0]]}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                state_from_dict(data)


class TraceValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "trace.jsonl"
        state = initial_state()
        self.header = {"type": "header", "rules_version": RULES_VERSION,
                       "state": state_to_dict(state), "digest": state_digest(state)}
        self.tick = {"type": "tick", "tick": 1,
                     "actions": [[0, -1, 0, 0], [0, -1, 0, 0]], "digest": "bad"}

    def write_rows(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_empty_invalid_json_and_wrong_header_types(self):
        for content in ("", "\n", "{", "[]\n", "null\n", "{}\n"):
            self.path.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "invalid trace at line 1"):
                replay(self.path)

    def test_missing_header_fields_are_value_errors(self):
        for key in ("type", "rules_version", "state", "digest"):
            header = deepcopy(self.header)
            del header[key]
            self.write_rows([header])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "invalid trace at line 1"):
                replay(self.path)

    def test_malformed_tick_rows_are_value_errors_with_line_number(self):
        malformed = [None, [], {}, {"type": "unknown"}]
        for key in ("type", "tick", "actions", "digest"):
            row = deepcopy(self.tick)
            del row[key]
            malformed.append(row)
        for key, value in (("tick", True), ("tick", 1.5), ("actions", {}),
                           ("actions", None), ("actions", []),
                           ("actions", [None, None]),
                           ("actions", [[1, 0.5, 1, 0], [0, -1, 0, 0]]),
                           ("actions", [[1, 0, 9, 0], [0, -1, 0, 0]])):
            row = deepcopy(self.tick)
            row[key] = value
            malformed.append(row)
        for row in malformed:
            self.write_rows([self.header, row])
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "invalid trace at line 2"):
                replay(self.path)

    def test_valid_recording_replays_and_tampered_digest_fails(self):
        state = initial_state()
        with TraceWriter(self.path, state) as recorder:
            runner = Runner(state, [WaitController(), WaitController()], max_ticks=2, recorder=recorder)
            runner.run()
        self.assertEqual(state_digest(replay(self.path)), state_digest(state))
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        rows[-1]["digest"] = "tampered"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "state digest mismatch at tick 2"):
            replay(self.path)


if __name__ == "__main__":
    unittest.main()
