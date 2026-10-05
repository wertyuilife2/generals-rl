"""Read-only native teacher collection, shifted value labels and fresh CNN CLI init."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from generals_rl.cts import runner_config
from generals_rl.mdp import RewardConfig
from generals_rl.self_play_training import prepare_self_play
from generals_rl.student_ppo import StudentPPOPolicy
from generals_rl.task import GeneralsTask, TaskConfig
from rsl_rl.storage import RolloutStorageCTS
from rl_test_utils import assert_nested_equal


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("spatial_distill_tool", ROOT/"rl/tools/distill_spatial.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class SpatialDistillationTests(unittest.TestCase):
    def test_native_local_kl_preserves_source_and_shifted_value_then_cli_uses_fresh_adam_pool(self):
        torch.set_num_threads(1)
        cfg = {"task": asdict(TaskConfig(width=9, height=9, players=2, history_length=4,
                  num_envs=4, max_episode_steps=3, opponent="self_play", seed=83,
                  combat_features=True, productive_actions=True, critic_potential=True,
                  reward=RewardConfig(potential_gamma=.995))),
               "runner": runner_config(hidden_dim=16, expert_num=2, rollout_steps=2, mini_batches=2),
               "self_play": {"pool_size": 2, "snapshot_interval": 1},
               "evaluation": {"enabled": False}, "torch_threads": 1}
        cfg["runner"]["policy"].update(class_name="generals_rl.student_ppo.StudentPPOPolicy",
                                          value_potential_offset=True, latent_dim=8)
        cfg["runner"]["algorithm"]["gamma"] = .995
        with tempfile.TemporaryDirectory() as directory, GeneralsTask(TaskConfig(**{
                **cfg["task"], "reward": RewardConfig(**cfg["task"]["reward"])})) as task, redirect_stdout(io.StringIO()):
            root = Path(directory); source = root/"old"; source.mkdir()
            observations = TensorDict(task.get_observations(), [4])
            teacher = tool.teacher_factory(cfg, observations)()
            teacher.eval().requires_grad_(False)
            before = deepcopy(teacher.state_dict())
            class Forbidden(torch.nn.Module):
                def forward(self, *args, **kwargs):
                    raise AssertionError("privileged teacher branch is forbidden")
            original_encoder = teacher.teacher_encoder
            teacher.teacher_encoder = Forbidden()
            probabilities, values = tool.teacher_labels(teacher, observations)
            teacher.teacher_encoder = original_encoder
            self.assertFalse(probabilities.requires_grad or values.requires_grad)
            teacher.value_potential_offset = False
            torch.testing.assert_close(values, teacher.evaluate(observations) - observations["critic_potential"])
            teacher.value_potential_offset = True
            runner = SimpleNamespace(alg=SimpleNamespace(policy=teacher), cfg={"obs_groups": cfg["runner"]["obs_groups"]},
                                     env=SimpleNamespace(get_observations=lambda: observations))
            prepare_self_play(task, runner, cfg, source)
            torch.save({"model_state_dict": teacher.state_dict(), "optimizer_state_dict": {"sentinel": True}}, source/"checkpoint.pt")
            (source/"run_config.json").write_text(json.dumps(cfg))
            def hashes():
                return {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source.rglob("*") if p.is_file()}
            immutable = hashes()
            rng = torch.get_rng_state().clone()
            with patch.object(RolloutStorageCTS, "__init__", side_effect=AssertionError("distill cannot allocate PPO storage")):
                report = tool.distill(source/"checkpoint.pt", root/"cnn", num_envs=4, batch_size=2,
                                      max_teacher_models=1, ticks=4, time_budget=30, tensorboard=False)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            assert_nested_equal(self, teacher.state_dict(), before)
            self.assertEqual(hashes(), immutable)
            self.assertEqual((report["ticks"], report["samples"], report["Adam_steps"]), (4, 16, 8))
            self.assertTrue(report["value_potential_offset"])
            self.assertEqual(report["source"]["sha256"], immutable["checkpoint.pt"])
            self.assertEqual(report["source"]["config_sha256"], immutable["run_config.json"])
            provenance = json.loads((root/"cnn/source_provenance.json").read_text())
            self.assertEqual(provenance["run_id"], report["run_id"])
            self.assertEqual(provenance["source"], {k: report["source"][k] for k in provenance["source"]})
            self.assertEqual(report["limits"]["saved_datasets"], 0)
            self.assertEqual(len(list((root/"cnn").rglob("*.pt"))), 2)
            cnn = torch.load(root/"cnn/checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertEqual(set(cnn), {"model_state_dict", "migration"})
            output_config = json.loads((root/"cnn/run_config.json").read_text())
            self.assertEqual(output_config["self_play"], cfg["self_play"])
            self.assertEqual(output_config["run_id"], report["run_id"])
            self.assertTrue(report["run_id"].startswith("cnn_kl_"))
            manifest = json.loads((root/"cnn/history_pool/manifest.json").read_text())
            self.assertEqual(len(manifest["snapshots"]), 1)
            self.assertEqual(manifest["policy_spec"]["policy_class"], "generals_rl.spatial_ppo.SpatialPPOPolicy")
            result = subprocess.run([sys.executable, "-m", "generals_rl.train", "--config", str(root/"cnn/run_config.json"),
                                     "--replace-policy", "--warm-start", str(root/"cnn/checkpoint.pt"),
                                     "--iterations", "1", "--no-eval", "--output", str(root/"ppo")],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            seed = torch.load(root/"ppo/history_pool/snapshot_00000000.pt", weights_only=True, map_location="cpu")
            assert_nested_equal(self, seed["model_state_dict"], cnn["model_state_dict"])
            initialized = torch.load(root/"ppo/checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertEqual(initialized["iter"], 0)
            # E4 * two minibatches, with fresh Adam rather than eight prior distill steps.
            self.assertEqual({int(v["step"]) for v in initialized["optimizer_state_dict"]["state"].values()}, {8})
            self.assertIsNone(initialized["optimizer_stu_enc_state_dict"])

    def test_formal_ppo_preserves_source_cadence_and_loaded_source_bytes(self):
        cfg = {"task": {}, "runner": {"policy": {}, "algorithm": {"learning_rate": 1e-4}},
               "self_play": {"pool_size": 8, "snapshot_interval": 50, "seed": 83, "stochastic": True},
               "evaluation": {"every_n_snapshots": 2}}
        settings = TaskConfig(width=9, height=9, history_length=4, num_envs=4)
        generated = tool.new_run_config(cfg, settings)
        self.assertEqual(generated["self_play"], cfg["self_play"])
        self.assertEqual(generated["self_play"]["snapshot_interval"] * generated["evaluation"]["every_n_snapshots"], 100)
        # A path replacement after the initial read cannot change loaded labels.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"checkpoint.pt"
            reference = torch.nn.Linear(2,1)
            torch.save({"model_state_dict": reference.state_dict()}, path)
            frozen = tool.source_bytes(path, tool.MAX_SOURCE_BYTES)
            before_sha = hashlib.sha256(frozen).hexdigest()
            path.write_bytes(b"replacement not a checkpoint")
            loaded = tool.load_teacher(path, lambda: torch.nn.Linear(2,1), "cpu", content=frozen)
            assert_nested_equal(self, loaded.state_dict(), reference.state_dict())
            self.assertNotEqual(before_sha, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
