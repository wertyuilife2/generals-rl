"""Optional critic shift: real terminal/timeout PPO and default isolation."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from generals.core.state import Action, Direction
from generals.vector import VecEnv
from generals_rl.cts import GeneralsCTSVecEnv
from generals_rl.evaluate_builtin import evaluate_builtin
from generals_rl.mdp import RewardConfig, ShapedReward
from generals_rl.student_ppo import StudentPPO, StudentPPOPolicy
from generals_rl.task import GeneralsTask, TaskConfig
from generals_rl.train import train
from rsl_rl.storage import RolloutStorageCTS
from test_mdp import capture_state
from test_student_ppo import observation, student_config


def policy(obs, offset=False):
    with redirect_stdout(io.StringIO()):
        return StudentPPOPolicy(obs, {"policy": ["policy"], "critic": ["critic"]},
            obs['action_mask'].shape[-1], action_distribution='categorical', latent_dim=8,
            expert_num=2, actor_hidden_dims=[16], critic_hidden_dims=[16],
            teacher_encoder_hidden_dims=[16], student_encoder_hidden_dims=[16],
            value_potential_offset=offset)


class PotentialValueOffsetTests(unittest.TestCase):
    def task(self, shaped):
        reward = RewardConfig(potential_gamma=.995) if shaped else RewardConfig(territory=0, cities=0, army=0)
        cfg = TaskConfig(width=3, height=3, num_envs=4, max_episode_steps=2,
                         history_length=2, rotate_seats=False, opponent='wait', seed=47,
                         reward=reward, critic_potential=shaped)
        backend = VecEnv.from_states([capture_state() for _ in range(4)],
                                     episode_ticks=2, auto_reset=False, threads=1)
        task = GeneralsTask(cfg, backend=backend)
        self.addCleanup(task.close)
        return task

    def test_default_and_actor_exact_independence_and_clipped_value_gradient(self):
        torch.manual_seed(44)
        obs = observation()
        base = policy(obs)
        shifted = policy(obs, True)
        shifted.load_state_dict(base.state_dict())
        with self.assertRaisesRegex(ValueError, 'critic_potential'):
            shifted.evaluate(obs)
        self.assertEqual(shifted.evaluate(obs[:0], is_teacher=True).shape, (0, 1))
        obs['critic_potential'] = torch.linspace(.1, 2, len(obs))[:, None].requires_grad_()
        torch.testing.assert_close(base.action_probabilities(obs), shifted.action_probabilities(obs), rtol=0, atol=0)
        torch.testing.assert_close(shifted.evaluate(obs), base.evaluate(obs) - obs['critic_potential'], rtol=0, atol=0)
        plain_obs = obs.exclude('critic_potential')
        torch.testing.assert_close(base.evaluate(obs), base.evaluate(plain_obs), rtol=0, atol=0)
        # Force clipping to be active, then compare both losses and every critic gradient.
        with torch.no_grad():
            old = base.evaluate(obs).clone()
            base.critic.network[-1].bias.add_(.4)
            shifted.critic.network[-1].bias.add_(.4)
        target = old - .3
        value = base.evaluate(obs)
        shifted_value = shifted.evaluate(obs)
        phi = obs['critic_potential'].detach()
        def clipped(value, old, target):
            clipped_value = old + (value - old).clamp(-.2, .2)
            return torch.maximum((value - target).square(), (clipped_value - target).square()).mean()
        loss = clipped(value, old, target)
        shifted_loss = clipped(shifted_value, old - phi, target - phi)
        torch.testing.assert_close(loss, shifted_loss, rtol=0, atol=1e-7)
        loss.backward()
        shifted_loss.backward()
        for (name, left), (_, right) in zip(base.named_parameters(), shifted.named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad, name)
            else:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
        self.assertIsNone(obs['critic_potential'].grad)
        with self.assertRaises(TypeError):
            policy(obs, 1)
        malformed = obs.clone()
        malformed['critic_potential'] = torch.ones(len(obs), 2)
        with self.assertRaisesRegex(ValueError, 'shape'):
            shifted.evaluate(malformed)

    def test_real_terminal_timeout_td_gae_full_ppo_update_matches_base_task(self):
        torch.manual_seed(127)
        tasks = [self.task(False), self.task(True)]
        bridges = [GeneralsCTSVecEnv(task) for task in tasks]
        observations = [bridge.get_observations() for bridge in bridges]
        policies = [policy(obs, bool(i)) for i, obs in enumerate(observations)]
        policies[1].load_state_dict(policies[0].state_dict())
        algorithms = []
        for obs, model in zip(observations, policies):
            storage = RolloutStorageCTS('rl', 4, 0, 4, obs, [1], distribution_shape=[73],
                                       action_dtype=torch.long, store_action_mean=False, store_action_std=False)
            algorithms.append(StudentPPO(model, storage, 4, gamma=.995, num_learning_epochs=2,
                                         num_mini_batches=2, use_clipped_value_loss=True))
        saw_terminal = saw_timeout = False
        with torch.inference_mode():
            for tick in range(4):
                rng = torch.get_rng_state()
                for index, (task, bridge, alg) in enumerate(zip(tasks, bridges, algorithms)):
                    obs = bridge.get_observations()
                    # Force one real capture, one two-tick timeout; other lanes retain normal choices.
                    if tick == 0:
                        capture = Action.move(3, Direction.RIGHT).encode()
                        obs['action_mask'][0].zero_()
                        obs['action_mask'][0, capture] = True
                    if tick < 2:
                        obs['action_mask'][3].zero_()
                        obs['action_mask'][3, 0] = True
                    torch.set_rng_state(rng)
                    actions = alg.act(obs)
                    next_obs, reward, dones, extras = bridge.step(actions)
                    saw_timeout |= bool(extras['time_outs'].any())
                    saw_terminal |= bool((dones & ~extras['time_outs']).any())
                    if index == 1:
                        terminal_metrics = task.reward_model.potential(
                            task.batch().learner_stats(task.player_ids))
                        torch.testing.assert_close(next_obs['critic_potential'], terminal_metrics, rtol=0, atol=0)
                        if bool(extras['time_outs'].any()):
                            timed_out = extras['time_outs']
                            self.assertFalse(torch.equal(next_obs['critic_potential'][timed_out],
                                extras['terminal_observations']['critic_potential'][timed_out]))
                    alg.process_env_step(next_obs, reward, dones, extras)
            for bridge, alg in zip(bridges, algorithms):
                alg.compute_returns(bridge.get_observations())
        self.assertTrue(saw_terminal and saw_timeout)
        left, right = [alg.storage for alg in algorithms]
        phi = right.observations['critic_potential']
        for name in ('actions', 'actions_log_prob', 'dones'):
            torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0)
        torch.testing.assert_close(right.values + phi, left.values, rtol=0, atol=1e-7)
        torch.testing.assert_close(right.returns + phi, left.returns, rtol=0, atol=3e-7)
        torch.testing.assert_close(right.advantages, left.advantages, rtol=0, atol=3e-6)
        def residual(storage, alg, obs):
            final = alg.policy.evaluate(obs).detach().unsqueeze(0)
            next_values = torch.cat((storage.values[1:], final))
            return storage.rewards + .995 * (1 - storage.dones.float()) * next_values - storage.values
        torch.testing.assert_close(residual(left, algorithms[0], bridges[0].get_observations()),
                                   residual(right, algorithms[1], bridges[1].get_observations()), rtol=0, atol=2e-7)
        rng = torch.get_rng_state()
        loss_left = algorithms[0].update()
        final_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        loss_right = algorithms[1].update()
        torch.testing.assert_close(final_rng, torch.get_rng_state(), rtol=0, atol=0)
        for key in loss_left:
            self.assertAlmostEqual(loss_left[key], loss_right[key], places=6)
        for (name, p), (_, q) in zip(policies[0].named_parameters(), policies[1].named_parameters()):
            torch.testing.assert_close(p, q, rtol=1e-4, atol=2e-6, msg=name)
            if p.grad is None:
                self.assertIsNone(q.grad, name)
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=2e-6, msg=name)
        adam_left, adam_right = [alg.optimizer.state_dict() for alg in algorithms]
        self.assertEqual(adam_left['param_groups'], adam_right['param_groups'])
        self.assertEqual(adam_left['state'].keys(), adam_right['state'].keys())
        for parameter in adam_left['state']:
            for key in adam_left['state'][parameter]:
                torch.testing.assert_close(adam_left['state'][parameter][key],
                    adam_right['state'][parameter][key], rtol=1e-4, atol=2e-6)

    def test_real_self_play_pool_signature_and_local_evaluation_and_early_guards(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            config = student_config()
            config['task'].update(critic_potential=True)
            config['task']['reward']['potential_gamma'] = .995
            config['runner']['algorithm']['gamma'] = .995
            config['runner']['policy']['value_potential_offset'] = True
            config['evaluation']['enabled'] = False
            output = Path(temporary) / 'run'
            train(config, output, iterations=1)
            manifest = json.loads((output / 'history_pool/manifest.json').read_text())
            spec = manifest['policy_spec']
            self.assertTrue(spec['policy_config']['value_potential_offset'])
            self.assertEqual(spec['observation_shapes']['critic_potential'], [1])
            report = evaluate_builtin(output / 'checkpoint.pt', maps_per_ai=1, max_ticks=3,
                                      batch_size=8, cpu_threads=1)
            self.assertEqual(report['completed_games'], 8)
            for mutation in ('task', 'reward', 'policy'):
                invalid = deepcopy(config)
                if mutation == 'task':
                    invalid['task']['critic_potential'] = False
                elif mutation == 'reward':
                    invalid['task']['reward']['potential_gamma'] = None
                else:
                    invalid['runner']['policy']['class_name'] = 'ActorCriticMoECTS'
                with patch('generals_rl.train.seed_everything') as seed, patch('generals_rl.train.GeneralsTask') as task:
                    with self.assertRaisesRegex(ValueError, 'value_potential_offset'):
                        train(invalid, Path(temporary) / 'invalid', iterations=1)
                    seed.assert_not_called()
                    task.assert_not_called()
            with self.assertRaises(TypeError):
                TaskConfig(critic_potential=1)


if __name__ == '__main__':
    unittest.main()
