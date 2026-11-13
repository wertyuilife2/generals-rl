"""Real PPO training and continuation with a live, local current-policy opponent."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from generals_rl.self_play_training import prepare_self_play
from generals_rl.train import train
from rl_test_utils import small_training_config


class CurrentSelfPlayTrainingTests(unittest.TestCase):
    def config(self):
        config = small_training_config()
        config["task"].update(num_envs=8, backend="cpu", device="cpu",
                              max_episode_steps=8, map_pool_size=0,
                              opponent="current_self_play")
        config["runner"]["num_steps_per_env"] = 4
        config["runner"]["save_interval"] = 1
        config["runner"]["algorithm"].update(num_learning_epochs=1, num_mini_batches=2,
                                              ppo_probe_samples=8, ppo_probe_batch_size=4)
        config["evaluation"] = {"enabled": False}
        config["training_control"] = {}
        return config

    def run_training(self, config, output, **budget):
        observed = {}

        def capture(task, runner, *args, **kwargs):
            observed["runner"] = runner
            observed["initial_model"] = deepcopy(runner.alg.policy.state_dict())
            observed["initial_iteration"] = runner.current_learning_iteration
            session = prepare_self_play(task, runner, *args, **kwargs)
            self.assertIs(task.opponent.policy, runner.alg.policy)
            self.assertIs(task.opponent.knowledge, task.knowledge)
            observed["session"] = session
            return session

        with redirect_stdout(io.StringIO()), patch("generals_rl.train.prepare_self_play", capture):
            summary = train(config, output, **budget)
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.assertEqual(summary, json.loads((output / "summary.json").read_text()))
        return summary, checkpoint, observed

    def test_live_policy_updates_and_checkpoints_without_history_pool(self):
        config = self.config()
        original = deepcopy(config)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            summary, checkpoint, observed = self.run_training(config, output, iterations=3)
            self.assertEqual(config, original)
            self.assertEqual(summary["environment_steps"], 8 * 4 * 3)
            self.assertEqual(summary["iterations"], 3)
            self.assertEqual(summary["self_play"]["training_opponent"], "live_learner")
            self.assertFalse(summary["self_play"]["snapshots_used_for_training"])
            self.assertNotIn("active_snapshot_counts", summary["self_play"])
            self.assertEqual(checkpoint["iter"], 2)
            policy = observed["runner"].alg.policy
            for prefix in ("actor.", "critic."):
                self.assertTrue(any(not torch.equal(value, observed["initial_model"][name])
                                    for name, value in policy.named_parameters() if name.startswith(prefix)))
            session = observed["session"]
            self.assertIsNone(session)
            self.assertIs(observed['runner'].env.task.opponent.policy, policy)
            self.assertFalse((output / 'history_pool').exists())
            self.assertEqual(summary['self_play']['new_publications'], 0)
            self.assertEqual(summary['self_play']['pool_ids'], [])
            for name, value in policy.state_dict().items():
                torch.testing.assert_close(value, checkpoint["model_state_dict"][name], rtol=0, atol=0)
            with torch.inference_mode():
                local = observed["runner"].env.get_observations().exclude("critic")
                actions = policy.act_inference(local)
                self.assertTrue(bool(local["action_mask"].gather(1, actions).all()))

    def test_resume_binds_restored_live_policy_and_exact_partial_rollout_budget(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, saved, _ = self.run_training(config, root / "first", iterations=2)
            summary, checkpoint, observed = self.run_training(
                config, root / "resume", steps_per_env=6, resume=str(root / "first/checkpoint.pt"))
            self.assertEqual(observed["initial_iteration"], 2)
            for name, value in observed["initial_model"].items():
                torch.testing.assert_close(value, saved["model_state_dict"][name], rtol=0, atol=0)
            self.assertEqual(summary["iterations"], 2)
            self.assertEqual(summary["steps_per_environment"], 6)
            self.assertEqual(summary["environment_steps"], 8 * 6)
            self.assertEqual(checkpoint["iter"], 3)
            self.assertIsNone(observed['session'])
            self.assertIs(observed['runner'].env.task.opponent.policy, observed['runner'].alg.policy)
            self.assertFalse((root / 'resume/history_pool').exists())
            self.assertFalse(summary["self_play"]["snapshots_used_for_training"])


if __name__ == "__main__":
    unittest.main()
