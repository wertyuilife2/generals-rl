"""CPU baseline-pool factory, lifecycle, persistence and cleanup contracts."""
from dataclasses import replace
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from generals.controllers import make_controllers
from generals.core.observation import Observation
from generals.core.state import Action, MoveResult, PlayerStats
from generals_rl.baseline_pool import BaselinePool, _feedback


def observation(player_id, tick=0):
    owner = np.full((3, 3), -1, dtype=np.int16)
    owner.flat[0], owner.flat[8] = 0, 1
    army = np.zeros((3, 3), dtype=np.int64)
    army.flat[0], army.flat[8] = 12, 12
    structure = np.zeros((3, 3), dtype=np.uint8)
    structure.flat[0], structure.flat[8] = 2, 2
    return Observation(player_id, tick, np.zeros((3, 3), dtype=np.uint8), structure, owner,
                       army, np.ones((3, 3), dtype=bool), np.ones(2, dtype=bool),
                       0 if player_id == 0 else 8,
                       PlayerStats(np.array([12, 12]), np.array([1, 1]), np.array([0, 0])))


def specs():
    return [SimpleNamespace(opponent=kind, learner_seat=seat, controller_seed=42 + index)
            for index, (kind, seat) in enumerate((
                ("aggressive", 0), ("expansion", 1), ("defensive", 0), ("random", 1)))]


class BaselinePoolTests(unittest.TestCase):
    def test_module_has_no_torch_dependency(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import generals_rl.baseline_pool; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_serial_uses_both_absolute_factory_slots_and_guards_lifecycle(self):
        configuration = specs()
        with patch("generals_rl.baseline_pool.make_controllers", wraps=make_controllers) as factory:
            with BaselinePool(configuration, 1) as pool:
                self.assertEqual(factory.call_count, 4)
                for call, spec in zip(factory.call_args_list, configuration, strict=True):
                    kinds = ["random", "random"]
                    kinds[1 - spec.learner_seat] = spec.opponent
                    self.assertEqual(call.args, (kinds,))
                    self.assertEqual(call.kwargs, {"seed": spec.controller_seed})
                with self.assertRaisesRegex(ValueError, "local Observation"):
                    pool.act({0: replace(observation(1), visibility="full")})
                actions = pool.act({0: observation(1)})
                self.assertIsInstance(actions[0], Action)
                with self.assertRaisesRegex(RuntimeError, "needs feedback"):
                    pool.act({0: observation(1)})
                with self.assertRaisesRegex(ValueError, "absolute seat order"):
                    pool.feedback({0: (MoveResult(1, actions[0], False, "NOT_OWNER"), None)})
                pool.feedback({0: (None, MoveResult(1, actions[0], False, "NOT_OWNER"))})
                self.assertEqual(pool._awaiting_feedback, set())
            pool.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pool.act({})

    def test_feedback_keeps_absolute_player_order_and_elimination_guard(self):
        events = []
        controller = SimpleNamespace(on_result=lambda move: events.append(("result", move.player_id)),
                                     eliminate=lambda: events.append(("eliminate",)))
        _feedback(controller, 1, (MoveResult(0, Action.wait(), True, "MOVED", eliminated=1),
                                  MoveResult(1, Action.wait(), True, "WAIT")))
        self.assertEqual(events, [("eliminate",), ("result", 1)])
        events.clear()
        _feedback(controller, 0, (MoveResult(0, Action.wait(), True, "WAIT"),
                                  MoveResult(1, Action.wait(), True, "MOVED", eliminated=0)))
        self.assertEqual(events, [("result", 0), ("eliminate",)])

    def test_spawn_retains_lane_rng_and_controllers_across_calls(self):
        with BaselinePool(specs(), 1) as serial, BaselinePool(specs(), 2) as parallel:
            processes = list(parallel._processes)
            pids = [process.pid for process in processes]
            for tick in range(5):
                inputs = {lane: observation(1 - spec.learner_seat, tick)
                          for lane, spec in enumerate(specs())}
                expected, actual = serial.act(inputs), parallel.act(inputs)
                self.assertEqual(actual, expected)
                feedback = {}
                for lane, action in expected.items():
                    seat = 1 - specs()[lane].learner_seat
                    moves = [None, None]
                    moves[seat] = MoveResult(seat, action, False, "NOT_OWNER")
                    feedback[lane] = tuple(moves)
                serial.feedback(feedback)
                parallel.feedback(feedback)
                self.assertEqual([process.pid for process in parallel._processes], pids)
                self.assertTrue(all(process.is_alive() for process in processes))
        self.assertTrue(all(not process.is_alive() and process.exitcode == 0 for process in processes))
        self.assertTrue(all(connection.closed for connection in parallel._connections))

    def test_reset_keeps_processes_but_recreates_factory_rng_for_new_and_smaller_batches(self):
        changed = [SimpleNamespace(opponent=spec.opponent, learner_seat=1 - spec.learner_seat,
                                   controller_seed=100 + spec.controller_seed) for spec in specs()]
        smaller = [SimpleNamespace(opponent="random", learner_seat=0, controller_seed=9876)]
        with BaselinePool(specs(), 1) as serial, BaselinePool(specs(), 2) as parallel:
            processes = list(parallel._processes)
            pids = [process.pid for process in processes]
            for batch_index, configuration in enumerate((specs(), changed, smaller, changed[:3])):
                if batch_index:
                    old_controller = serial._controllers[0]
                    with patch("generals_rl.baseline_pool.make_controllers", wraps=make_controllers) as factory:
                        serial.reset(configuration)
                        self.assertEqual(factory.call_count, len(configuration))
                    parallel.reset(configuration)
                    self.assertIsNot(serial._controllers[0], old_controller)
                references = []
                for spec in configuration:
                    kinds = ["random", "random"]
                    kinds[1 - spec.learner_seat] = spec.opponent
                    references.append(make_controllers(kinds, seed=spec.controller_seed)[1 - spec.learner_seat])
                for tick in range(4):
                    inputs = {lane: observation(1 - spec.learner_seat, tick)
                              for lane, spec in enumerate(configuration)}
                    expected = {lane: references[lane].act(obs) for lane, obs in inputs.items()}
                    self.assertEqual(serial.act(inputs), expected)
                    self.assertEqual(parallel.act(inputs), expected)
                    feedback = {}
                    for lane, action in expected.items():
                        seat = 1 - configuration[lane].learner_seat
                        move = MoveResult(seat, action, False, "NOT_OWNER")
                        references[lane].on_result(move)
                        feedback[lane] = (None, move) if seat else (move, None)
                    serial.feedback(feedback)
                    parallel.feedback(feedback)
                    for lane, reference in enumerate(references):
                        self.assertEqual(serial._controllers[lane].rng.bit_generator.state,
                                         reference.rng.bit_generator.state)
                    self.assertEqual([process.pid for process in parallel._processes], pids)
                    self.assertEqual(parallel.workers, 2)
                with self.assertRaisesRegex(ValueError, "unknown baseline lane"):
                    parallel.act({len(configuration): observation(1)})
        self.assertTrue(all(not process.is_alive() and process.exitcode == 0 for process in processes))

    def test_reset_rejects_lanes_with_pending_feedback(self):
        for workers in (1, 2):
            with self.subTest(workers=workers), BaselinePool(specs(), workers) as pool:
                pool.act({0: observation(1), 1: observation(0)})
                pool.feedback({0: (None, None)})
                with self.assertRaisesRegex(RuntimeError, "feedback to be complete"):
                    pool.reset(specs()[:1])
                self.assertEqual(set(pool._specs), {0, 1, 2, 3})
                pool.feedback({1: (None, None)})
                pool.reset(specs()[:1])
                self.assertEqual(set(pool._specs), {0})

    def test_worker_error_propagates_and_closes_all_workers(self):
        pool = BaselinePool(specs(), 2).__enter__()
        processes = list(pool._processes)
        malformed = replace(observation(1), visible=np.ones((2, 2), dtype=bool))
        with self.assertRaisesRegex(RuntimeError, "ValueError: invalid observation array: visible"):
            pool.act({0: malformed, 1: observation(0)})
        self.assertTrue(all(not process.is_alive() for process in processes))
        self.assertTrue(all(connection.closed for connection in pool._connections))
        pool.close()

    def test_serial_partial_act_or_feedback_failure_closes_pool(self):
        with BaselinePool(specs(), 1) as pool:
            malformed = replace(observation(0), visible=np.ones((2, 2), dtype=bool))
            with self.assertRaisesRegex(ValueError, "invalid observation array: visible"):
                pool.act({0: observation(1), 1: malformed})
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pool.act({0: observation(1)})
        with BaselinePool(specs(), 1) as pool:
            actions = pool.act({0: observation(1), 1: observation(0)})
            moves = {0: (None, MoveResult(1, actions[0], False, "NOT_OWNER")),
                     1: (MoveResult(0, actions[1], False, "NOT_OWNER"), None)}
            with patch.object(pool._controllers[1], "on_result", side_effect=RuntimeError("callback failure")):
                with self.assertRaisesRegex(RuntimeError, "callback failure"):
                    pool.feedback(moves)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pool.feedback(moves)

    def test_unexpected_worker_exit_does_not_leave_other_workers_alive(self):
        pool = BaselinePool(specs(), 2).__enter__()
        processes = list(pool._processes)
        processes[0].terminate()
        processes[0].join(timeout=5)
        with self.assertRaises((RuntimeError, OSError, EOFError)):
            pool.act({0: observation(1), 1: observation(0)})
        self.assertTrue(all(not process.is_alive() for process in processes))
        self.assertTrue(all(connection.closed for connection in pool._connections))

    def test_close_kills_worker_that_survives_terminate(self):
        pool = BaselinePool(specs(), 2)
        alive = [True]
        process = SimpleNamespace(pid=123, is_alive=lambda: alive[0], join=Mock(), terminate=Mock(),
                                  kill=Mock(side_effect=lambda: alive.__setitem__(0, False)))
        connection = Mock()
        pool._processes, pool._connections = [process], [connection]
        pool.close()
        pool.close()
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        connection.close.assert_called_once_with()
        self.assertFalse(process.is_alive())


if __name__ == "__main__":
    unittest.main()
