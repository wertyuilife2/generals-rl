"""Rank configuration, collective control, timing and process-group ownership."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from generals_rl.distributed_training import DistributedConfig, distributed_process_group


def group_worker(rank, directory):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    config = DistributedConfig(rank=rank, local_rank=rank, world_size=2, device="cpu")
    results = {}
    with distributed_process_group(config, backend="gloo",
                                   init_method=f"file://{directory}/first_group", timeout_seconds=30):
        results["timings"] = config.max_timings(rank + 1., 3. - rank)
        results["counts"] = config.sum_counts(rank + 1, rank * 2)
        results["healthy"] = config.all_true(True)
        results["unhealthy"] = config.all_true(rank == 0)
        results["continue"] = config.broadcast_control(
            {"stop_reason": None, "next_rollout_steps": 3} if rank == 0 else {"ignored": True})
        results["stop"] = config.broadcast_control(
            {"stop_reason": "requested_stop", "next_rollout_steps": None} if rank == 0 else None)
        results["transitions"] = config.global_transitions(4, 3)
        config.wait_on_cpu()
        # Nested contexts borrow the group and must leave it alive.
        with distributed_process_group(config, backend="gloo"):
            assert dist.is_initialized()
        assert dist.is_initialized()
        other_rank = DistributedConfig(rank=1 - rank, local_rank=rank, world_size=2)
        try:
            with distributed_process_group(other_rank):
                raise AssertionError("a mismatched group was accepted")
        except ValueError:
            pass
        try:
            with distributed_process_group(config, backend="nccl"):
                raise AssertionError("a mismatched backend was accepted")
        except ValueError:
            pass
    assert not dist.is_initialized()
    try:
        with distributed_process_group(config, init_method=f"file://{directory}/second_group",
                                       timeout_seconds=30):
            assert config.all_true(True)
            raise RuntimeError("worker failure")
    except RuntimeError as error:
        assert str(error) == "worker failure"
    assert not dist.is_initialized()
    assert not torch.cuda.is_initialized()
    torch.save(results, Path(directory) / f"rank_{rank}.pt")


class DistributedTrainingTests(unittest.TestCase):
    def test_single_rank_preserves_device_seed_and_needs_no_group(self):
        config = DistributedConfig.from_env("cuda:3", environ={})
        self.assertEqual(config.device, torch.device("cuda:3"))
        self.assertTrue(config.is_main)
        self.assertFalse(config.is_distributed)
        self.assertIsNone(config.multi_gpu_cfg)
        self.assertEqual(config.rank_seed(42), 42)
        self.assertEqual(config.global_transitions(64, 32), 2048)
        self.assertEqual(config.max_timings(.5, 1.), (.5, 1.))
        self.assertEqual(config.sum_counts(2, 3), (2, 3))
        self.assertTrue(config.all_true(True))
        config.wait_on_cpu()
        control = {"stop_reason": None}
        self.assertIs(config.broadcast_control(control), control)
        with patch.object(dist, "is_initialized", return_value=False), \
             patch.object(dist, "init_process_group") as initialize, \
             patch.object(dist, "destroy_process_group") as destroy, \
             patch.object(torch.cuda, "set_device") as set_device:
            with distributed_process_group(config) as result:
                self.assertIs(result, config)
        initialize.assert_not_called()
        destroy.assert_not_called()
        set_device.assert_not_called()

    def test_torchrun_uses_local_device_and_global_rank_seed(self):
        config = DistributedConfig.from_env("cuda:0", {"WORLD_SIZE": "8", "RANK": "5", "LOCAL_RANK": "1"})
        self.assertEqual((config.rank, config.local_rank, config.world_size), (5, 1, 8))
        self.assertEqual(config.device, torch.device("cuda:1"))
        self.assertFalse(config.is_main)
        self.assertTrue(config.is_distributed)
        self.assertEqual(config.multi_gpu_cfg, {"global_rank": 5, "local_rank": 1, "world_size": 8})
        self.assertEqual(config.rank_seed(42), 5_000_057)
        self.assertEqual(config.global_transitions(64, 32), 16384)
        seeds = {DistributedConfig(rank=rank, world_size=8).rank_seed(42) for rank in range(8)}
        self.assertEqual(len(seeds), 8)
        cpu = DistributedConfig.from_env("cpu", {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"})
        self.assertEqual(cpu.device, torch.device("cpu"))

    def test_invalid_rank_configuration_and_uninitialized_collectives_fail(self):
        for environ in ({"WORLD_SIZE": "0"}, {"WORLD_SIZE": "2", "RANK": "2"},
                        {"WORLD_SIZE": "2", "LOCAL_RANK": "2"}, {"RANK": "-1"},
                        {"WORLD_SIZE": "many"}):
            with self.subTest(environ=environ), self.assertRaises(ValueError):
                DistributedConfig.from_env("cpu", environ)
        with self.assertRaises(ValueError):
            DistributedConfig(world_size=True)
        config = DistributedConfig(world_size=2)
        with patch.object(dist, "is_initialized", return_value=False):
            for operation in (lambda: config.max_timings(1.), lambda: config.sum_counts(1),
                              lambda: config.all_true(True), lambda: config.broadcast_control({})):
                with self.assertRaisesRegex(RuntimeError, "initialize"):
                    operation()
        with self.assertRaises(ValueError):
            config.rank_seed(-1)
        with self.assertRaises(ValueError):
            config.global_transitions(2.5, 4)

    def test_cleanup_failure_preserves_original_training_error(self):
        config = DistributedConfig(world_size=2)
        with patch.object(dist, "is_initialized", side_effect=[False, True]), \
             patch.object(dist, "init_process_group") as initialize, \
             patch.object(dist, "destroy_process_group", side_effect=RuntimeError("cleanup failure")) as destroy:
            with self.assertRaisesRegex(ValueError, "training failure"):
                with distributed_process_group(config, init_method="file:///unused"):
                    raise ValueError("training failure")
        self.assertEqual(initialize.call_args.kwargs["backend"], "gloo")
        destroy.assert_called_once()

    def test_nccl_final_wait_uses_gloo_and_cleans_up_after_errors(self):
        config = DistributedConfig(world_size=2, device="cuda:0")
        group = object()
        for error in (None, RuntimeError("wait failed")):
            with self.subTest(error=error), patch.object(DistributedConfig, "_verify_group"), \
                    patch.object(dist, "get_backend", return_value="nccl"), \
                    patch.object(dist, "new_group", return_value=group) as create, \
                    patch.object(dist, "barrier", side_effect=error) as barrier, \
                    patch.object(dist, "destroy_process_group") as destroy:
                if error is None:
                    config.wait_on_cpu()
                else:
                    with self.assertRaisesRegex(RuntimeError, "wait failed"):
                        config.wait_on_cpu()
                create.assert_called_once_with(backend="gloo")
                barrier.assert_called_once_with(group=group)
                destroy.assert_called_once_with(group)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo is unavailable")
    def test_two_ranks_share_control_reduce_timings_and_clean_up_groups(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "",
                                                                       "OMP_NUM_THREADS": "1"}):
            mp.spawn(group_worker, args=(directory,), nprocs=2, join=True)
            results = [torch.load(Path(directory) / f"rank_{rank}.pt", weights_only=True) for rank in range(2)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], {
            "timings": (2., 3.), "counts": (3, 2), "healthy": True, "unhealthy": False,
            "continue": {"stop_reason": None, "next_rollout_steps": 3},
            "stop": {"stop_reason": "requested_stop", "next_rollout_steps": None},
            "transitions": 24,
        })


if __name__ == "__main__":
    unittest.main()
