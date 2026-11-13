"""Both player trajectories match independent single-seat MDPs and PPO targets."""
from copy import deepcopy
from pathlib import Path
import contextlib
import io
import tempfile
import unittest
import weakref

import torch

from generals.core.state import Action, Direction
from generals.vector import VecEnv
from generals_rl.cts import GeneralsCTSVecEnv
from generals_rl.mdp.rewards import RewardConfig
from generals_rl.task import GeneralsTask, TaskConfig
from generals_rl.train import train
from test_mdp import capture_state, example_state, hidden_world_variant
from rl_test_utils import small_training_config


torch.set_num_threads(1)


class JointActions:
    def __init__(self, actions):
        self.actions = actions

    def joint_actions(self, *args):
        return self.actions.clone()


class DualSelfPlayTests(unittest.TestCase):
    def task(self, states, *, dual=True, seat=0, horizon=8, opponent=None):
        config = TaskConfig(width=states[0].layout.width, height=states[0].layout.height,
            num_envs=len(states), random_maps=False, map_pool_size=0, history_length=4,
            max_episode_steps=horizon, opponent='current_self_play' if dual else 'wait',
            train_both_sides=dual, learner_seat=seat, rotate_seats=False,
            persistent_knowledge=True, combat_features=True, king_threat_distance=True,
            reward=RewardConfig(potential_gamma=.99))
        backend = VecEnv.from_states(states, auto_reset=False, episode_ticks=horizon, threads=1)
        task = GeneralsTask(config, backend=backend, opponent=opponent)
        self.addCleanup(task.close)
        return task

    def compare_rows(self, actual, expected, seat):
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            torch.testing.assert_close(actual[key][seat::2], expected[key], rtol=0, atol=0, msg=key)

    def test_observations_rewards_and_terminal_reset_match_both_independent_seats(self):
        states = [capture_state(), capture_state()]
        dual = self.task(states)
        actions = torch.tensor([[Action.move(3, Direction.RIGHT).encode(), 0], [0, 0]])
        singles = [self.task(states, dual=False, seat=seat, opponent=JointActions(actions)) for seat in (0, 1)]
        self.assertEqual((dual.num_games, dual.num_envs), (2, 4))
        self.assertIsNone(dual.opponent)
        self.assertEqual(dual.knowledge.terrain.shape[:2], (2, 2))
        for seat, single in enumerate(singles):
            self.compare_rows(dual.get_observations(), single.get_observations(), seat)
        initial_history = dual.history.frames.clone()
        result = dual.step(actions.flatten())
        for seat, single in enumerate(singles):
            reference = single.step(actions[:, seat])
            self.compare_rows(result.observations, reference.observations, seat)
            self.compare_rows(result.terminal_observations, reference.terminal_observations, seat)
            torch.testing.assert_close(result.reward[seat::2], reference.reward, rtol=0, atol=0)
            torch.testing.assert_close(result.terminated[seat::2], reference.terminated)
        self.assertEqual(result.terminated.tolist(), [True, True, False, False])
        self.assertEqual(dual.completed_episodes, 1)  # actual games, not two players
        torch.testing.assert_close(dual.history.frames[2:, :-1], initial_history[2:, 1:])
        self.assertEqual(dual.player_ids.tolist(), [0, 1, 0, 1])

    def test_timeout_keeps_final_observations_for_both_seats(self):
        states = [capture_state()]
        dual = self.task(states, horizon=1)
        env = GeneralsCTSVecEnv(dual)
        obs, rewards, done, extras = env.step(torch.zeros(2, dtype=torch.long))
        self.assertEqual(done.tolist(), [True, True])
        self.assertEqual(extras['time_outs'].tolist(), [True, True])
        self.assertEqual(dual.truncated_episodes, 1)
        self.assertTrue(torch.isfinite(rewards).all())
        # Both history clocks contain the pre-reset t=1 frame in the bootstrap
        # observation; the next episode starts again at t=0.
        self.assertTrue((extras['terminal_observations']['single_obs'][:, -5] > 0).all())
        self.assertTrue((obs['single_obs'][:, -5] == 0).all())

    def test_only_legal_local_information_enters_each_player_history(self):
        state = example_state()
        dual = self.task([state, hidden_world_variant(state)])
        observations = dual.get_observations()
        for key in ('policy', 'single_obs', 'knowledge', 'action_mask'):
            torch.testing.assert_close(observations[key][0], observations[key][2], rtol=0, atol=0)
        self.assertFalse(torch.equal(observations['critic'][0], observations['critic'][2]))
        self.assertFalse(torch.equal(observations['policy'][0], observations['policy'][1]))

    def test_pair_reset_contract_and_mode_validation(self):
        dual = self.task([capture_state()])
        with self.assertRaisesRegex(ValueError, 'both seats'):
            dual.reset(torch.tensor([True, False]))
        with self.assertRaisesRegex(ValueError, 'current_self_play'):
            TaskConfig(train_both_sides=True, opponent='self_play')

    def test_ppo_storage_and_gae_include_both_seats_and_correct_timeout_bootstrap(self):
        from rsl_rl.runners import OnPolicyRunnerCTS
        dual = self.task([capture_state(), capture_state()], horizon=1)
        env = GeneralsCTSVecEnv(dual)
        config = small_training_config(runner_overrides=dict(rollout_steps=2, learning_epochs=1, mini_batches=2))['runner']
        config['policy'].update(map_width=3, map_height=3)
        config['algorithm'].update(ppo_monitor=False, gamma=.99, schedule='fixed', desired_kl=None)
        with contextlib.redirect_stdout(io.StringIO()):
            runner = OnPolicyRunnerCTS(env, deepcopy(config), log_dir=None, device='cpu')
        obs = env.get_observations()
        recorded_rewards = []
        with torch.inference_mode():
            for _ in range(2):
                actions = runner.alg.act(obs)
                self.assertIs(runner.alg.transition.observations, obs)
                obs, reward, done, extras = env.step(actions)
                terminal_value = runner.alg.policy.evaluate(extras['terminal_observations']).flatten()
                recorded_rewards.append(reward + .99 * terminal_value * extras['time_outs'])
                runner.alg.process_env_step(obs, reward, done, extras)
            runner.alg.compute_returns(obs)
        storage = runner.alg.storage
        self.assertEqual(storage.num_envs, 4)
        torch.testing.assert_close(storage.rewards[:2, :, 0], torch.stack(recorded_rewards))
        # Every transition is an episode boundary. Its return is its own
        # correctly bootstrapped reward, with no carry from the other seat/game.
        torch.testing.assert_close(storage.returns[:2, :, 0], torch.stack(recorded_rewards))
        before = [p.clone() for p in runner.alg.policy.parameters()]
        losses = runner.alg.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, runner.alg.policy.parameters())))

    def test_training_checkpoint_resume_has_no_history_pool(self):
        config = small_training_config(task_overrides=dict(width=23, height=23, random_maps=True,
            map_pool_size=4, num_envs=2, max_episode_steps=2, opponent='current_self_play',
            train_both_sides=True))
        config['runner']['policy'].update(map_width=23, map_height=23)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / 'first'
            summary = train(config, output, iterations=2)
            self.assertEqual(summary['board_ticks'], 16)
            self.assertEqual(summary['player_transitions'], 32)
            self.assertFalse((output / 'history_pool').exists())
            self.assertEqual(summary['self_play']['pool_ids'], [])
            resumed = Path(directory) / 'resumed'
            result = train(config, resumed, iterations=1, resume=str(output / 'checkpoint.pt'))
            self.assertEqual(result['training_trajectories_per_rank'], 4)
            self.assertFalse((resumed / 'history_pool').exists())

    def test_runner_releases_consumed_terminal_history_before_update(self):
        from unittest.mock import patch
        from rsl_rl.runners import OnPolicyRunnerCTS
        task = self.task([capture_state()], horizon=1)
        config = small_training_config(runner_overrides=dict(rollout_steps=2, learning_epochs=1, mini_batches=1))['runner']
        config['policy'].update(map_width=3, map_height=3)
        config['release_terminal_observations'] = True
        config['algorithm'].update(ppo_monitor=False, schedule='fixed', desired_kl=None)
        with contextlib.redirect_stdout(io.StringIO()):
            runner = OnPolicyRunnerCTS(GeneralsCTSVecEnv(task), config, log_dir=None, device='cpu')
        references, checked = [], []
        process, update = runner.alg.process_env_step, runner.alg.update
        def observe(obs, reward, done, extras):
            references.append(weakref.ref(extras['terminal_observations']['policy']))
            return process(obs, reward, done, extras)
        def verify():
            self.assertIsNone(references[-1]())
            checked.append(True)
            return update()
        with patch.object(runner.alg, 'process_env_step', side_effect=observe), \
                patch.object(runner.alg, 'update', side_effect=verify), \
                contextlib.redirect_stdout(io.StringIO()):
            runner.learn(1)
        self.assertEqual(checked, [True])


if __name__ == '__main__':
    unittest.main()
