"""Stopping at update boundaries and retaining usable final/best checkpoints."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from generals_rl.self_play_training import prepare_self_play
from generals_rl.train import main, train
from generals_rl.training_control import TrainingControlConfig, TrainingController
from rl_test_utils import assert_nested_equal, small_training_config


class TrainingControlTests(unittest.TestCase):
    def test_plateau_needs_successful_points_minimum_time_and_meaningful_progress(self):
        controller = TrainingController(TrainingControlConfig(
            min_seconds=10, max_seconds=100, plateau_evaluations=3, min_improvement=0.02))
        point = {"selection_score": [0.4, 0.4, 0.6]}
        for _ in range(3):
            controller.on_evaluation(point)
        self.assertIsNone(controller.stop_reason(20))
        controller.on_evaluation(point)
        self.assertIsNone(controller.stop_reason(9))
        self.assertIsNone(controller.stop_reason(20, evaluation_healthy=False))
        self.assertEqual(controller.stop_reason(20), "evaluation_plateau")
        controller.on_evaluation({"selection_score": [0.4, 0.45, 0.6]})
        self.assertIsNone(controller.stop_reason(20))
        self.assertEqual(controller.stop_reason(100, evaluation_healthy=False), "max_seconds")

    def test_time_or_file_stop_saves_a_complete_update_and_does_not_require_the_full_tick_budget(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            for reason in ("max_seconds", "requested_stop"):
                with self.subTest(reason=reason):
                    config = small_training_config()
                    config["evaluation"] = {"enabled": False}
                    config["training_control"] = {"keep_checkpoints": 2}
                    output = Path(directory) / reason
                    if reason == "max_seconds":
                        config["training_control"]["max_seconds"] = 1e-9
                    else:
                        output.mkdir()
                        (output / "STOP").touch()
                    summary = train(config, output, steps_per_env=40)
                    checkpoint = torch.load(output / "checkpoint.pt", weights_only=False, map_location="cpu")
                    self.assertEqual(summary["stop_reason"], reason)
                    self.assertEqual(summary["steps_per_environment"], 4)
                    self.assertEqual(summary["iterations"], 1)
                    self.assertEqual(checkpoint["iter"], 0)
                    for key in ("optimizer_state_dict", "optimizer_stu_enc_state_dict"):
                        self.assertEqual({int(state["step"]) for state in checkpoint[key]["state"].values()}, {4})
                    self.assertEqual(json.loads((output / "summary.json").read_text()), summary)

    def test_runner_checkpoints_have_a_fixed_total_count(self):
        config = small_training_config()
        config["runner"]["save_interval"] = 1
        config["evaluation"] = {"enabled": False}
        config["training_control"] = {"keep_checkpoints": 2}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            train(config, output, iterations=5)
            self.assertEqual({path.name for path in output.glob("*.pt")}, {"model_3.pt", "checkpoint.pt"})

    def test_invalid_control_settings_fail_before_creating_output(self):
        cases = ({"max_seconds": 0}, {"max_seconds": float("nan")}, {"min_seconds": -1},
                 {"min_seconds": 2, "max_seconds": 1}, {"plateau_evaluations": True},
                 {"keep_checkpoints": 0}, {"min_improvement": -0.1})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            for settings in cases:
                config = small_training_config()
                config["training_control"] = settings
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    train(config, output, iterations=1)
                self.assertFalse(output.exists())

    def test_warm_start_loads_full_or_weights_only_checkpoints_with_fresh_optimizers_and_pool(self):
        config = small_training_config()
        config["evaluation"] = {"enabled": False}
        config["self_play"] = {"snapshot_interval": 1, "pool_size": 3}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source = Path(directory) / "source"
            train(config, source, iterations=2)
            manifest = source / "history_pool/manifest.json"
            original_manifest = manifest.read_bytes()
            # The evaluator's best.pt has the same weights-only pool format.
            newest = json.loads(original_manifest)["snapshots"][-1]["filename"]
            (source / "best.pt").write_bytes((source / "history_pool" / newest).read_bytes())
            config["runner"]["algorithm"].update(
                schedule="fixed", learning_rate=1e-4, student_encoder_learning_rate=2e-4)
            for filename in ("checkpoint.pt", "best.pt"):
                with self.subTest(source=filename):
                    config["task"]["productive_actions"] = filename == "best.pt"
                    checkpoint = source / filename
                    expected = torch.load(checkpoint, weights_only=False, map_location="cpu")["model_state_dict"]
                    observed = {}

                    def capture(task, runner, *args, **kwargs):
                        observed["iteration"] = runner.current_learning_iteration
                        observed["model"] = deepcopy(runner.alg.policy.state_dict())
                        observed["main_lr"] = runner.alg.learning_rate
                        observed["optimizers"] = {key: deepcopy(getattr(runner.alg, key).state_dict())
                                                  for key in ("optimizer", "optimizer_stu_enc")}
                        session = prepare_self_play(task, runner, *args, **kwargs)
                        observed["pool_ids"] = session.pool.ids
                        observed["seed"] = session.pool.get(0)
                        return session

                    output = Path(directory) / filename.removesuffix(".pt")
                    with patch("generals_rl.train.prepare_self_play", capture):
                        summary = train(config, output, iterations=1, warm_start=checkpoint)
                    self.assertEqual(observed["iteration"], 0)
                    self.assertEqual(observed["main_lr"], 1e-4)
                    assert_nested_equal(self, observed["model"], expected)
                    assert_nested_equal(self, observed["seed"].model.state_dict(), expected)
                    for key, lr in (("optimizer", 1e-4), ("optimizer_stu_enc", 2e-4)):
                        self.assertFalse(observed["optimizers"][key]["state"])
                        self.assertTrue(all(group["lr"] == lr for group in observed["optimizers"][key]["param_groups"]))
                    self.assertEqual(observed["pool_ids"], (0,))
                    pool_manifest = json.loads((output / "history_pool/manifest.json").read_text())
                    self.assertIsNone(pool_manifest["resume_source"])
                    mdp_spec = pool_manifest["policy_spec"]["mdp"]
                    if config["task"]["productive_actions"]:
                        self.assertTrue(mdp_spec["productive_actions"])
                    else:
                        self.assertNotIn("productive_actions", mdp_spec)
                    self.assertEqual(summary["warm_start"], {
                        "kind": "warm_start_weights", "path": str(checkpoint.resolve()),
                        "sha256": sha256(checkpoint.read_bytes()).hexdigest()})
                    self.assertEqual(observed["seed"].metadata["source"], summary["warm_start"])
                    self.assertEqual(manifest.read_bytes(), original_manifest)
                    final = torch.load(output / "checkpoint.pt", weights_only=False, map_location="cpu")
                    self.assertEqual(final["iter"], 0)
                    for key in ("optimizer_state_dict", "optimizer_stu_enc_state_dict"):
                        self.assertEqual({int(value["step"]) for value in final[key]["state"].values()}, {4})

    def test_warm_start_and_resume_are_exclusive_and_cli_uses_new_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaises(ValueError):
                train(small_training_config(), output, iterations=1, resume="old.pt", warm_start="best.pt")
            self.assertFalse(output.exists())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["--resume", "old.pt", "--warm-start", "best.pt"])
            self.assertEqual(error.exception.code, 2)
            new_config = Path(directory) / "new.json"
            new_config.write_text(json.dumps({"runner": {"algorithm": {"learning_rate": 1e-4}}}))
            # No checkpoint-side run_config.json is required for warm-start.
            with patch("generals_rl.train.train") as invoked:
                main(["--warm-start", "best.pt", "--config", str(new_config), "--output", str(output), "--iterations", "1"])
            self.assertEqual(invoked.call_args.kwargs["warm_start"], "best.pt")
            self.assertIsNone(invoked.call_args.kwargs["resume"])
            self.assertEqual(invoked.call_args.args[0]["runner"]["algorithm"]["learning_rate"], 1e-4)

    def test_matmul_precision_is_explicit_and_restored_after_success_or_failure(self):
        original_precision = torch.get_float32_matmul_precision()
        config = small_training_config()
        config["evaluation"] = {"enabled": False}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch("torch.set_float32_matmul_precision", wraps=torch.set_float32_matmul_precision) as setter:
                train(config, Path(directory) / "default", iterations=1)
                setter.assert_not_called()
            config["matmul_precision"] = "high"

            def capture(*args, **kwargs):
                self.assertEqual(torch.get_float32_matmul_precision(), "high")
                return prepare_self_play(*args, **kwargs)

            with patch("generals_rl.train.prepare_self_play", capture):
                train(config, Path(directory) / "high", iterations=1)
            self.assertEqual(torch.get_float32_matmul_precision(), original_precision)
            with patch("generals_rl.train.GeneralsTask", side_effect=RuntimeError("construction failed")), \
                    self.assertRaisesRegex(RuntimeError, "construction failed"):
                train(config, Path(directory) / "failed", iterations=1)
            self.assertEqual(torch.get_float32_matmul_precision(), original_precision)
            for invalid in ("low", True, {}):
                config["matmul_precision"] = invalid
                output = Path(directory) / "invalid"
                with self.subTest(precision=invalid), self.assertRaises(ValueError):
                    train(config, output, iterations=1)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
