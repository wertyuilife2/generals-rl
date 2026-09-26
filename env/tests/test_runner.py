import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from generals_env import (Action, CoreEngine, Direction, GameState, MapConfig,
                          MapLayout, Structure, generate_map, observe)
from generals_env.controllers import make_controllers
from generals_env.recording import TraceWriter, replay, state_digest
from generals_env.runner import Runner, ControllerError


class Scripted:
    def __init__(self, action=None):
        self.action = action or Action.wait()
        self.observations = []
        self.results = []
    def act(self, obs):
        self.observations.append(obs)
        return self.action
    def on_result(self, result): self.results.append(result)


def duel():
    return GameState(MapLayout(np.zeros((1, 3), dtype=np.uint8)),
                     np.array([[2, 0, 2]], dtype=np.uint8),
                     np.array([[0, -1, 1]], dtype=np.int16),
                     np.array([[5, 0, 1]], dtype=np.int64),
                     np.ones(2, dtype=bool), np.array([0, 2], dtype=np.int32))


class RunnerTests(unittest.TestCase):
    def test_sequential_observations_and_growth(self):
        a, b = Scripted(Action.move(0, Direction.RIGHT)), Scripted()
        runner = Runner(duel(), [a, b], max_ticks=2)
        result = runner.tick()
        self.assertEqual(b.observations[0].owner[0, 1], 0)
        self.assertEqual(b.observations[0].army[0, 1], 4)
        self.assertEqual(result.stats.territory.tolist(), [2, 1])
        a.action = Action.wait()
        runner.tick()
        self.assertEqual(a.observations[-1].army[0, 0], 2)
        self.assertTrue(runner.truncated)
        self.assertFalse(runner.engine.state.terminated)
        self.assertIsNone(runner.engine.state.winner_id)
        with self.assertRaises(RuntimeError): runner.tick()

    def test_natural_terminal_precedes_tick_limit(self):
        state = duel()
        state.owner[0, 1], state.army[0, 1] = 0, 10
        a, b = Scripted(Action.move(1, Direction.RIGHT)), Scripted()
        runner = Runner(state, [a, b], max_ticks=1)
        runner.tick()
        self.assertTrue(runner.stopped)
        self.assertFalse(runner.truncated)
        self.assertEqual(runner.stop_reason, 'terminated')
        self.assertEqual(runner.engine.state.winner_id, 0)
        self.assertEqual(b.observations, [])

    def test_user_stop_does_not_change_state(self):
        state = duel()
        runner = Runner(state, [Scripted(), Scripted()])
        digest = state_digest(state)
        runner.stop()
        self.assertEqual(runner.stop_reason, 'user_stop')
        self.assertFalse(runner.truncated)
        self.assertEqual(state_digest(state), digest)

    def test_bad_controller_stops_coherently(self):
        class Bad(Scripted):
            def act(self, obs): raise ValueError('broken controller')
        runner = Runner(duel(), [Bad(), Scripted()])
        with self.assertRaisesRegex(ControllerError, 'broken controller'):
            runner.tick()
        self.assertEqual(runner.stop_reason, 'controller_error')
        self.assertIsNotNone(runner.last_result)
        self.assertIsNone(runner.engine.next_player)
        runner.engine.state.validate()
        with self.assertRaises(RuntimeError): runner.tick()

    def test_repeated_seed_and_frame_sampling_do_not_change_game(self):
        kinds = ['aggressive', 'expansion', 'defensive', 'random']
        one = Runner(generate_map(MapConfig(width=15, height=15), 77), make_controllers(kinds, 77), max_ticks=120)
        two = Runner(generate_map(MapConfig(width=15, height=15), 77), make_controllers(kinds, 77), max_ticks=120)
        while not one.stopped:
            one.tick()
            for _ in range(3): observe(two.engine.state, 0)
            two.tick()
        self.assertEqual(state_digest(one.engine.state), state_digest(two.engine.state))

    def test_transaction_and_batch_actions_match(self):
        initial = duel()
        actions = [Action.move(0, Direction.RIGHT), Action.wait()]
        runner = Runner(initial.copy(), [Scripted(a) for a in actions])
        engine = CoreEngine(initial.copy())
        runner.tick()
        engine.step(actions)
        self.assertEqual(state_digest(runner.engine.state), state_digest(engine.state))


class RecordingTests(unittest.TestCase):
    def test_round_trip_and_corruption_detection(self):
        kinds = ['aggressive', 'expansion']
        state = generate_map(MapConfig(15, 15, 2), 11)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'match.jsonl'
            with TraceWriter(path, state, {'seed': 11}) as writer:
                runner = Runner(state, make_controllers(kinds, 11), max_ticks=80, recorder=writer)
                runner.run()
            restored = replay(path)
            self.assertEqual(state_digest(restored), state_digest(runner.engine.state))
            lines = path.read_text().splitlines()
            row = json.loads(lines[-1])
            row['digest'] = 'wrong'
            lines[-1] = json.dumps(row)
            path.write_text('\n'.join(lines) + '\n')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                replay(path)

    def test_invalid_negative_action_is_replayable(self):
        state = duel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.jsonl'
            with TraceWriter(path, state) as writer:
                runner = Runner(state, [Scripted(Action.move(-1, 0)), Scripted()], max_ticks=1, recorder=writer)
                runner.run()
            self.assertEqual(state_digest(replay(path)), state_digest(state))

    def test_existing_recording_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'match.jsonl'
            path.write_text('keep')
            with self.assertRaises(FileExistsError): TraceWriter(path, duel())
            self.assertEqual(path.read_text(), 'keep')


if __name__ == '__main__': unittest.main()
