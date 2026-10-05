"""Optional potential reward contracts, real terminal/timeout and TD invariance."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from generals.core.engine import CoreEngine
from generals.core.state import Action, Direction
from generals.vector import VecEnv
from generals_rl.mdp import RewardConfig, ShapedReward
from generals_rl.task import GeneralsTask, TaskConfig
from generals_rl.train import train
from rl_test_utils import small_training_config
from test_mdp import capture_state, CaptureOpponent


class PotentialRewardTests(unittest.TestCase):
    def test_default_exact_old_reward(self):
        generator = torch.Generator().manual_seed(42)
        before = torch.randint(0, 500, (9, 3), generator=generator)
        after = torch.randint(0, 500, (9, 3), generator=generator)
        terminated = torch.tensor([0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=torch.bool)
        truncated = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.bool)
        alive = torch.tensor([1, 1, 0, 1, 1, 0, 0, 1, 1], dtype=torch.bool)
        pids = torch.arange(9) % 2
        winner = torch.tensor([-1, 1, 1, -1, -1, 0, -1, -1, -1])
        cfg = RewardConfig()
        actual = ShapedReward(cfg).evaluate(before, after, terminated, truncated, winner, alive, pids)
        delta = after.float() - before.float()
        original = {"territory": cfg.territory * delta[:, 1], "cities": cfg.cities * delta[:, 2],
                    "army": cfg.army * delta[:, 0],
                    "terminal": (terminated & (winner == pids)).float() * cfg.win + (~alive).float() * cfg.loss,
                    "tick": torch.full_like(delta[:, 0], cfg.tick)}
        self.assertIsNone(cfg.potential_gamma)
        for name, expected in original.items():
            torch.testing.assert_close(actual.terms[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.total, sum(original.values()), rtol=0, atol=0)

    def test_configuration_finite_and_gamma_range(self):
        for name in ("territory", "cities", "army", "win", "loss", "tick", "potential_gamma"):
            for invalid in (float('nan'), float('inf'), -float('inf'), True, '.995'):
                with self.subTest(name=name, value=invalid), self.assertRaises(ValueError):
                    RewardConfig(**{name: invalid})
        for invalid in (-.001, 1.001):
            with self.assertRaises(ValueError):
                RewardConfig(potential_gamma=invalid)
        for valid in (None, 0, .995, 1):
            self.assertEqual(RewardConfig(potential_gamma=valid).potential_gamma, valid)

    def task(self, state, **overrides):
        cfg = TaskConfig(width=3, height=3, players=state.player_count, num_envs=1,
                         threads=1, history_length=2, rotate_seats=False, opponent="wait",
                         reward=RewardConfig(potential_gamma=.995), **overrides)
        backend = VecEnv.from_states([state], episode_ticks=cfg.max_episode_steps, auto_reset=False, threads=1)
        task = GeneralsTask(cfg, backend=backend)
        self.addCleanup(task.close)
        return task

    def test_real_terminal_elimination_and_timeout_potential(self):
        for players, seat, action in ((2, 0, Action.move(3, Direction.RIGHT).encode()), (3, 1, 0)):
            state = capture_state(players)
            task = self.task(state, learner_seat=seat, max_episode_steps=1)
            if seat == 1:
                task.opponent = CaptureOpponent()
            before = task.batch().learner_stats(task.player_ids)
            result = task.step(torch.tensor([[action]]))
            self.assertTrue(result.terminated.item())
            self.assertFalse(result.truncated.item())
            for name, index, weight in (("army", 0, .001), ("territory", 1, .05), ("cities", 2, .2)):
                torch.testing.assert_close(result.metrics[f"Reward/{name}"], -weight * before[:, index].float(), rtol=0, atol=0)
            self.assertEqual(result.metrics["Reward/terminal"].item(), 1 if seat == 0 else -1)
        state = capture_state()
        task = self.task(state, learner_seat=0, max_episode_steps=2)
        task.step(torch.zeros((1, 1), dtype=torch.long))
        before = task.batch().learner_stats(task.player_ids)
        # Independent full Core tick computes the real pre-reset final stats.
        engine = CoreEngine(state.copy())
        engine.step([Action.wait(), Action.wait()])
        final = engine.step([Action.wait(), Action.wait()]).stats
        after = torch.tensor([[final.army[0], final.territory[0], final.cities[0]]])
        result = task.step(torch.zeros((1, 1), dtype=torch.long))
        self.assertFalse(result.terminated.item())
        self.assertTrue(result.truncated.item())
        for name, index, weight in (("army", 0, .001), ("territory", 1, .05), ("cities", 2, .2)):
            expected = weight * (.995 * after[:, index].float() - before[:, index].float())
            torch.testing.assert_close(result.metrics[f"Reward/{name}"], expected, rtol=0, atol=0)
        self.assertEqual(result.metrics["Reward/terminal"].item(), 0)

    def test_multistep_telescope_td_gae_and_timeout_bootstrap_invariance(self):
        metrics = torch.tensor([[30, 3, 1], [32, 4, 1], [28, 5, 2], [35, 6, 2]])
        weights = torch.tensor([.001, .05, .2], dtype=torch.float64)
        phi = metrics.double() @ weights
        # Compare proper shaping against terminal+tick base, not the old raw
        # difference reward (which is not a potential for gamma<1).
        for terminal in (False, True):
            for gamma in (.995, 1.):
                cfg = RewardConfig(potential_gamma=gamma)
                reward = ShapedReward(cfg)
                shaped, base = [], []
                for t in range(3):
                    done = terminal and t == 2
                    result = reward.evaluate(metrics[t:t+1], metrics[t+1:t+2],
                            torch.tensor([done]), torch.tensor([not terminal and t == 2]),
                            torch.tensor([0 if done else -1]), torch.tensor([True]), torch.tensor([0]))
                    shaped.append(result.total.double().item())
                    base.append(result.terms['terminal'].double().item() + result.terms['tick'].double().item())
                shaped, base = torch.tensor(shaped, dtype=torch.float64), torch.tensor(base, dtype=torch.float64)
                discount = gamma ** torch.arange(3, dtype=torch.float64)
                boundary_phi = 0 if terminal else phi[-1]
                torch.testing.assert_close((discount * (shaped - base)).sum(), -phi[0] + gamma**3 * boundary_phi,
                                           rtol=0, atol=2e-7)
                value = torch.tensor([.7, .8, .9, 1.], dtype=torch.float64)
                alive_next = torch.ones(3, dtype=torch.float64)
                if terminal:
                    alive_next[-1] = 0
                effective_phi = phi.clone()
                if terminal:
                    effective_phi[-1] = 0
                shifted = value - effective_phi
                delta = base + gamma * alive_next * value[1:] - value[:-1]
                shaped_delta = shaped + gamma * alive_next * shifted[1:] - shifted[:-1]
                torch.testing.assert_close(shaped_delta, delta, rtol=0, atol=2e-7)
                gae, shaped_gae = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
                for t in reversed(range(3)):
                    gae[t] = delta[t] + (gamma * .95 * gae[t+1] if t < 2 else 0)
                    shaped_gae[t] = shaped_delta[t] + (gamma * .95 * shaped_gae[t+1] if t < 2 else 0)
                torch.testing.assert_close(shaped_gae, gae, rtol=0, atol=4e-7)
                if not terminal:
                    # rsl marks timeout done after adding gamma*V(final obs).
                    torch.testing.assert_close(shaped[-1] + gamma * shifted[-1] - shifted[-2], delta[-1], rtol=0, atol=2e-7)

    def test_train_rejects_mismatched_or_implicit_gamma_before_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            for gamma in (.99, None, True, float('nan')):
                config = small_training_config()
                config['task']['reward']['potential_gamma'] = .995
                if gamma is None:
                    del config['runner']['algorithm']['gamma']
                else:
                    config['runner']['algorithm']['gamma'] = gamma
                output = Path(temporary) / 'invalid'
                with patch('generals_rl.train.seed_everything') as seed, patch('generals_rl.train.GeneralsTask') as task:
                    with self.assertRaisesRegex(ValueError, 'matching runner.algorithm.gamma'):
                        train(config, output, iterations=1)
                    seed.assert_not_called()
                    task.assert_not_called()
                    self.assertFalse(output.exists())
            config = small_training_config()
            config['task']['reward']['potential_gamma'] = .995
            config['runner']['algorithm']['gamma'] = .995
            config['evaluation'] = {'enabled': False}
            with contextlib.redirect_stdout(io.StringIO()):
                summary = train(config, Path(temporary) / 'matching', iterations=1)
            self.assertEqual(summary['steps_per_environment'], 4)
            self.assertEqual(summary['environment_steps'], 16)


if __name__ == '__main__':
    unittest.main()
