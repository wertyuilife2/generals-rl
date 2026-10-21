"""Real CPU shard simulations check global IDs, aggregation and trace choice."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from generals_rl.eval_parallel import evaluate_parallel, merge_reports
from generals_rl.evaluate_builtin import evaluate_builtin
from generals_rl.task import TaskConfig


class UniformStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def action_probabilities(self, obs):
        mask = obs["action_mask"].float()
        return mask / mask.sum(-1, keepdim=True)


class ParallelEvaluationTests(unittest.TestCase):
    def test_shards_keep_global_games_and_merge_exact_statistics_and_traces(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            checkpoint = root / "snapshot.pt"
            checkpoint.write_bytes(b"immutable test snapshot")
            config = {"task": asdict(TaskConfig(width=5, height=5, mountain_density=0, city_density=0)),
                      "torch_threads": 1}
            (root / "run_config.json").write_text(json.dumps(config))
            options = dict(maps_per_ai=4, seed=73, max_ticks=64, stochastic=True)
            with patch("generals_rl.evaluate_builtin._load_student", side_effect=lambda *args: UniformStudent()):
                original = evaluate_builtin(checkpoint, trace_directory=root / "original", **options)
                shards = [evaluate_builtin(checkpoint, shard_index=index, num_shards=4,
                                           trace_directory=root / f"shard_{index}", **options) for index in range(4)]
            # Old no-op worker metadata must neither constrain nor leak into new reports.
            shards[0]["baseline_workers"] = 1
            shards[1]["baseline_workers"] = 4
            merged = merge_reports(shards, devices=["cpu"] * 4, elapsed=10, worker_reports=[])
            self.assertNotIn("baseline_workers", merged)
            self.assertEqual(merged["games"], original["games"])
            for name in ("overall", "by_ai", "by_seat", "by_ai_and_seat", "completed_games", "scheduled_games"):
                self.assertEqual(merged[name], original[name])
            actual, expected = merged["selected_game_traces"], original["selected_game_traces"]
            self.assertEqual(actual["missing"], expected["missing"])
            self.assertEqual([case["game_id"] for case in actual["cases"]],
                             [case["game_id"] for case in expected["cases"]])
            self.assertEqual(merged["timing"]["environment_ticks"], original["timing"]["environment_ticks"])
            self.assertEqual(merged["timing"]["environment_ticks_per_second"],
                             original["timing"]["environment_ticks"] / 10)
            for corruption in ("duplicate", "missing", "checkpoint"):
                broken = deepcopy(shards)
                if corruption == "duplicate":
                    broken[1]["games"][0]["game_id"] = broken[0]["games"][0]["game_id"]
                elif corruption == "missing":
                    broken[1]["games"].pop()
                else:
                    broken[1]["checkpoint_sha256"] = "different snapshot"
                with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                    merge_reports(broken, devices=["cpu"] * 4, elapsed=10, worker_reports=[])

    def test_invalid_devices_and_excess_shards_fail_before_launch(self):
        for devices in ([], ["cpu"], ["cuda"], ["cuda:0", "cuda:0"], [f"cuda:{i}" for i in range(9)]):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                evaluate_parallel("unused.pt", devices=devices, maps_per_ai=1)

    def test_legacy_workers_option_is_not_forwarded_to_evaluator(self):
        with patch("generals_rl.eval_parallel.evaluate_builtin", return_value={}) as evaluate:
            evaluate_parallel("unused.pt", devices=["cuda:0"], maps_per_ai=1, baseline_workers=4)
        self.assertNotIn("baseline_workers", evaluate.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
