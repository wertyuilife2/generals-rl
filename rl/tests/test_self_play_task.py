"""Self-play episode, information-boundary, and history contracts."""
from dataclasses import dataclass
import unittest

import torch

from generals.core.state import Action, Direction
from generals.vector import VecEnv
from generals_rl.opponents import SelfPlayOpponent
from generals_rl.task import GeneralsTask, TaskConfig
from test_mdp import capture_state, example_state, hidden_world_variant


class LocalPolicy:
    def __init__(self):
        self.inputs = []

    def action_probabilities(self, observations, *, is_teacher=False):
        assert not is_teacher
        assert set(observations.keys()) == {"policy", "single_obs", "action_mask"}
        self.inputs.append({key: value.clone() for key, value in observations.items()})
        weights = observations["action_mask"].float()
        # Nontrivial stochastic decisions, without touching the global RNG.
        return weights / weights.sum(1, keepdim=True)

    def act_inference(self, observations):
        self.action_probabilities(observations)
        return torch.zeros((len(observations), 1), dtype=torch.long)


@dataclass
class Snapshot:
    snapshot_id: int
    model: LocalPolicy


class TestPool:
    __test__ = False

    def __init__(self):
        self.snapshots = [Snapshot(0, LocalPolicy())]
        self.requests = []

    @property
    def ids(self):
        return tuple(item.snapshot_id for item in self.snapshots)

    def sample(self, count, *, generator):
        self.requests.append(count)
        return [self.snapshots[-1]] * count

    def replace(self):
        item = Snapshot(self.snapshots[-1].snapshot_id + 1, LocalPolicy())
        self.snapshots = [item]
        return item


class SelfPlayTaskTests(unittest.TestCase):
    def task(self, states, **settings):
        cfg = dict(random_maps=False, map_pool_size=0, width=states[0].layout.width, height=states[0].layout.height,
                   players=states[0].player_count, num_envs=len(states), opponent="self_play",
                   history_length=3, max_episode_steps=16, rotate_seats=False)
        cfg.update(settings)
        config = TaskConfig(**cfg)
        backend = VecEnv.from_states(states, episode_ticks=config.max_episode_steps,
                                     auto_reset=False, threads=1)
        task = GeneralsTask(config, backend=backend)
        self.addCleanup(task.close)
        self.assertIsInstance(task.opponent, SelfPlayOpponent)
        return task

    def test_unbound_pool_fails_before_any_simulator_tick(self):
        task = self.task([capture_state()])
        with self.assertRaisesRegex(RuntimeError, "attach"):
            task.step(torch.tensor([0]))
        self.assertEqual(task.batch().state["tick"].item(), 0)

    def test_snapshot_is_frozen_until_episode_reset_despite_eviction(self):
        state = capture_state()
        task = self.task([state, state.copy()], rotate_seats=True)
        pool = TestPool()
        task.opponent.attach(pool, stochastic=False, seed=100)
        old = pool.snapshots[0]
        self.assertEqual(task.opponent.assignment_ids.tolist(), [[-1, 0], [0, -1]])
        new = pool.replace()
        # Lane zero captures the enemy general; lane one remains ongoing.
        result = task.step(torch.tensor([Action.move(3, Direction.RIGHT).encode(), 0]))
        self.assertEqual(result.terminated.tolist(), [True, False])
        self.assertEqual(task.opponent.assignment_ids.tolist(), [[1, -1], [0, -1]])
        self.assertEqual(task.opponent.snapshot_counts(), {0: 1, 1: 1})
        self.assertEqual(task.opponent.episode_assignments, 3)
        self.assertIs(task.opponent._snapshots[0], old)
        self.assertIs(task.opponent._snapshots[1], new)
        self.assertEqual(sum(len(x["single_obs"]) for x in old.model.inputs), 2)
        task.reset(torch.tensor([False, True]), rotate=True)
        self.assertNotIn(0, task.opponent._snapshots)
        self.assertEqual(task.opponent.snapshot_counts(), {1: 2})

    def test_history_pushes_once_and_partial_reset_clears_only_ended_lane(self):
        state = capture_state()
        task = self.task([state, state.copy()])
        task.opponent.attach(TestPool(), stochastic=False)
        initial = [history.frames.clone() for history in task.opponent.histories]
        task.step(torch.zeros(2, dtype=torch.long))
        first_tick = [history.frames.clone() for history in task.opponent.histories]
        for before, after in zip(initial, first_tick):
            torch.testing.assert_close(after[:, :2], before[:, 1:])
            self.assertFalse(torch.equal(after[:, -1], before[:, -1]))
        # Lane zero ends; lane one keeps exactly the last two ticks in order.
        result = task.step(torch.tensor([Action.move(3, Direction.RIGHT).encode(), 0]))
        self.assertEqual(result.terminated.tolist(), [True, False])
        for pid, history in enumerate(task.opponent.histories):
            torch.testing.assert_close(history.frames[1, :2], first_tick[pid][1, 1:])
            current = task.observation_encoder.encode(task.batch(), torch.full_like(task.player_ids, pid))
            torch.testing.assert_close(history.frames[0], current[0].expand(3, -1))

    def test_local_inputs_hide_world_and_stochastic_sampling_preserves_global_rng(self):
        first = example_state()
        second = hidden_world_variant(first)
        # Learner seat one makes historical opponents see the equivalent
        # player-zero local worlds, while their privileged worlds differ.
        task = self.task([first, second], learner_seat=1)
        pool = TestPool()
        task.opponent.attach(pool, seed=301)
        before_rng = torch.get_rng_state().clone()
        task.step(torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
        inputs = pool.snapshots[0].model.inputs[0]
        for key in ("policy", "single_obs", "action_mask"):
            torch.testing.assert_close(inputs[key][0], inputs[key][1], rtol=0, atol=0)
        self.assertFalse(task.opponent.histories[0].frames.requires_grad)

    def test_duel_infers_only_nonlearner_seats_and_supports_seeded_repeat(self):
        states = [capture_state() for _ in range(4)]
        task = self.task(states, rotate_seats=True)
        pool = TestPool()
        task.opponent.attach(pool, seed=123)
        batch = task.batch()
        rng = task.opponent.generator.get_state()
        first = task.opponent.joint_actions(batch, task.player_ids, task.action_space, task.observation_encoder)
        task.opponent.generator.set_state(rng)
        second = task.opponent.joint_actions(batch, task.player_ids, task.action_space, task.observation_encoder)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(task.opponent.snapshot_counts(), {0: 4})
        self.assertEqual(sum(len(obs["single_obs"]) for obs in pool.snapshots[0].model.inputs), 8)
        for pid in range(2):
            legal = task.action_space.mask(batch, torch.full_like(task.player_ids, pid))
            self.assertTrue(legal.gather(1, first[:, pid, None]).all())
        self.assertTrue((first[torch.arange(4), task.player_ids] == 0).all())


if __name__ == "__main__":
    unittest.main()
