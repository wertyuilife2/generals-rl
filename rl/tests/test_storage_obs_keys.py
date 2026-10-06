"""Storage-only observation selection preserves real Light PPO updates."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import tempfile
import unittest

import torch

from generals_rl.config import seed_everything, task_config
from generals_rl.cts import GeneralsCTSVecEnv
from generals_rl.mdp import RewardConfig
from generals_rl.self_play_training import prepare_self_play
from generals_rl.spatial_ppo import policy_config
from generals_rl.task import GeneralsTask
from rsl_rl.runners import OnPolicyRunnerCTS
from rl_test_utils import assert_nested_equal, small_training_config


STORAGE_KEYS = ["policy", "critic", "action_mask"]
FULL_KEYS = {*STORAGE_KEYS, "single_obs"}


def config():
    cfg = small_training_config(
        task_overrides={"width": 9, "height": 9, "history_length": 4,
                        "seed": 20261013, "combat_features": True,
                        "king_threat_distance": True, "productive_actions": True,
                        "map_pool_size": 8, "reward": RewardConfig(potential_gamma=.995)},
        runner_overrides={"learning_epochs": 1})
    cfg["runner"]["policy"] = policy_config(light_cnn=True)
    cfg["runner"]["policy"]["own_king_geometry"] = True
    cfg["runner"]["algorithm"].update(
        class_name="generals_rl.student_ppo.StudentPPO", teacher_env_ratio=0,
        schedule="fixed", desired_kl=None, load_balance_coef=0, gamma=.995)
    cfg["self_play"] = {"pool_size": 2, "snapshot_interval": 50, "seed": 917}
    return cfg


class StorageObsKeysTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def rollout(self, cfg):
        seed_everything(cfg["task"]["seed"])
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                GeneralsTask(task_config(cfg["task"])) as task:
            runner = OnPolicyRunnerCTS(GeneralsCTSVecEnv(task), deepcopy(cfg["runner"]), None, "cpu")
            prepare_self_play(task, runner, cfg, directory)
            alg, storage = runner.alg, runner.alg.storage
            initial = deepcopy(alg.policy.state_dict())
            obs = runner.env.get_observations()
            records, timeouts = [], 0
            with torch.inference_mode():
                for _ in range(cfg["runner"]["num_steps_per_env"]):
                    self.assertEqual(set(obs.keys()), FULL_KEYS)
                    actions = alg.act(obs)
                    self.assertEqual(set(alg.transition.observations.keys()), FULL_KEYS)
                    obs, rewards, dones, extras = runner.env.step(actions)
                    records.append((actions.clone(), rewards.clone(), dones.clone()))
                    timeouts += int(extras["time_outs"].sum())
                    alg.process_env_step(obs, rewards, dones, extras)
                alg.compute_returns(obs)
            self.assertEqual(timeouts, task.num_envs)
            self.assertEqual(set(obs.keys()), FULL_KEYS)
            stored = {key: storage.observations[key].clone() for key in STORAGE_KEYS}
            fields = {key: getattr(storage, key).clone() for key in
                      ("actions", "rewards", "dones", "values", "actions_log_prob", "returns", "advantages")}
            payload = sum(value.numel() * value.element_size() for value in storage.observations.values())
            keys = set(storage.observations.keys())
            losses = alg.update()
            self.assertTrue(any(not torch.equal(initial[key], value)
                                for key, value in alg.policy.state_dict().items()))
            self.assertTrue(alg.optimizer.state)
            return {"initial": initial, "records": records, "observations": stored,
                    "fields": fields, "losses": losses,
                    "model": deepcopy(alg.policy.state_dict()),
                    "adam": deepcopy(alg.optimizer.state_dict()), "rng": torch.get_rng_state().clone(),
                    "storage_keys": keys, "payload": payload}

    def test_subset_and_explicit_none_match_default_bitwise(self):
        cfg = config()
        reference = self.rollout(cfg)
        self.assertEqual(reference["storage_keys"], FULL_KEYS)
        for selected in (None, STORAGE_KEYS):
            with self.subTest(storage_obs_keys=selected):
                candidate_cfg = deepcopy(cfg)
                candidate_cfg["runner"]["storage_obs_keys"] = selected
                actual = self.rollout(candidate_cfg)
                for key in ("initial", "records", "observations", "fields", "losses", "model", "adam", "rng"):
                    with self.subTest(field=key):
                        assert_nested_equal(self, actual[key], reference[key])
                expected_keys = FULL_KEYS if selected is None else set(STORAGE_KEYS)
                self.assertEqual(actual["storage_keys"], expected_keys)
                saved_bytes = 0 if selected is None else 4 * 4 * 985 * 4
                self.assertEqual(reference["payload"] - actual["payload"], saved_bytes)

    def test_invalid_storage_keys_rejected(self):
        cfg = config()
        with redirect_stdout(io.StringIO()), GeneralsTask(task_config(cfg["task"])) as task:
            env = GeneralsCTSVecEnv(task)
            for invalid in ("policy", [], 4, ["policy", 1], ["policy", "policy"], ["missing"]):
                with self.subTest(storage_obs_keys=invalid), self.assertRaisesRegex(ValueError, "storage_obs_keys"):
                    options = deepcopy(cfg["runner"])
                    options["storage_obs_keys"] = invalid
                    OnPolicyRunnerCTS(env, options, None, "cpu")


if __name__ == "__main__":
    unittest.main()
