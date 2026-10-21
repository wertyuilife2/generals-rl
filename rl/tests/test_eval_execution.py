"""Boundary checks, trace pruning and evaluation-only CUDA graph conformance."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict

from generals import MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals_rl.evaluate_builtin import GameSpec, SelectedGameTraces, _local_inputs, _initial_knowledge
from generals_rl.eval_execution import FeaturePipeline, GraphStudent, pack_actions, unpack_actions
from generals_rl.mdp import GameBatch, ObservationHistory, make_action_space, make_observation_encoder
from generals_rl.task import TaskConfig


class EvaluationExecutionTests(unittest.TestCase):
    def test_packed_boundary_preserves_actions_and_errors(self):
        space = make_action_space(TaskConfig(width=5, height=5))
        mask = torch.ones((3, space.size), dtype=torch.bool)
        mask[1, 7] = False
        for values in ([0, 4, 11], [0, 7, 1], [0, -1, 1], [0, space.size, 1]):
            actions = torch.tensor(values)
            try:
                expected = space.validate(actions, mask).numpy()
            except ValueError as error:
                with self.assertRaisesRegex(ValueError, str(error)):
                    unpack_actions(pack_actions(actions, mask, space))
            else:
                np.testing.assert_array_equal(unpack_actions(pack_actions(actions, mask, space)), expected)
        with self.assertRaises(TypeError):
            pack_actions(torch.zeros(3), mask, space)
        with self.assertRaises(ValueError):
            pack_actions(torch.zeros((3, 2), dtype=torch.long), mask, space)

    def test_trace_pruning_keeps_lower_unfinished_ids_and_exact_minima(self):
        states = [generate_map(MapConfig(5, 5, 2), seed=19 + i) for i in range(6)]
        specs = [GameSpec(i, "random", i, 19 + i, 7, 0) for i in range(6)]
        with TemporaryDirectory() as directory:
            recorder = SelectedGameTraces(Path(directory), max_ticks=10,
                opponents=("random",), metadata={"checkpoint_sha256": "synthetic"})
            recorder.on_start(specs, states)
            def finish(gid, outcome):
                recorder.on_complete([{"game_id": gid, "opponent": "random", "outcome": outcome}])
            finish(2, "win")
            self.assertIn(5, recorder.candidates)  # Still a possible first loss.
            finish(3, "loss")
            self.assertEqual(set(recorder.candidates), {0, 1, 2, 3})
            finish(1, "win")
            self.assertEqual(set(recorder.candidates), {0, 1, 3})
            finish(0, "draw")
            self.assertEqual(set(recorder.candidates), {1, 3})

    def test_feature_pipeline_matches_original_history_knowledge_and_mask(self):
        self._feature_parity("cpu", capture=False)

    def _feature_parity(self, device, *, capture):
        config = TaskConfig(width=9, height=9, num_envs=4, history_length=4,
                            combat_features=True, king_threat_distance=True, persistent_knowledge=True)
        states = [generate_map(MapConfig(9, 9, 2), seed=19 + i) for i in range(4)]
        ids = torch.tensor([0, 1, 0, 1], device=device)
        encoder = make_observation_encoder(config)
        action_space = make_action_space(config)
        with VecEnv.from_states(states, device="cpu", threads=1, auto_reset=False, episode_ticks=32) as env:
            batch = GameBatch.from_backend(env, device)
            history, expected_history = ObservationHistory(4), ObservationHistory(4)
            current = encoder.encode(batch, ids)
            history.reset(current); expected_history.reset(current)
            memory = _initial_knowledge(batch, ids, config)
            expected_memory = memory.clone()
            with torch.inference_mode():
                pipeline = FeaturePipeline(batch, ids, encoder, history, action_space, memory, capture=capture)
                for tick in range(14):
                    original = _local_inputs(batch, ids, encoder, expected_history, action_space, expected_memory)
                    for name in original.keys():
                        torch.testing.assert_close(pipeline.observations[name], original[name], rtol=0, atol=0)
                    for name in memory._fields:
                        torch.testing.assert_close(getattr(memory, name), getattr(expected_memory, name), rtol=0, atol=0)
                    # Expand, attempt invalid moves and keep later lane memory frozen.
                    joint = np.full((4, 2), 1 + (tick % 81) * 8 + (tick % 4) * 2, dtype=np.int64)
                    env.step(joint)
                    batch = GameBatch.from_backend(env, device)
                    active = torch.tensor([True, tick < 3, tick < 8, True], device=device)
                    expected_history.push(encoder.encode(batch, ids))
                    expected_memory.update(batch, ids, mask=active)
                    pipeline.advance(batch, active)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graph conformance requires CUDA")
class CudaEvaluationExecutionTests(EvaluationExecutionTests):
    def test_feature_graph_replay_is_exact_and_initial_outputs_are_materialized(self):
        self._feature_parity("cuda:0", capture=True)

    def test_actor_graph_reuses_buckets_without_stale_rows_or_changing_weights(self):
        class Student(nn.Module):
            wait_move_factorization = False
            compile_forward = True
            def __init__(self):
                super().__init__()
                self.actor = nn.Linear(4, 9, bias=False)
                with torch.no_grad():
                    self.actor.weight.copy_(torch.arange(36).reshape(9, 4) / 32)
            def _forward(self, obs):
                return self.actor(obs["policy"]).float()
            def _probabilities_from_logits(self, logits, mask, *, validate_mask=True):
                return logits.masked_fill(~mask, -torch.inf).softmax(-1)
        policy = Student().eval().cuda()
        weights = policy.actor.weight.clone()
        graph = GraphStudent(policy)
        rng = torch.cuda.get_rng_state().clone()
        with torch.inference_mode():
            for count in (5, 6, 9, 10, 3, 4, 2, 1, 8, 5, 0):
                inputs = torch.arange(count * 4, device="cuda").reshape(count, 4).float() / 16
                mask = torch.ones((count, 9), device="cuda", dtype=torch.bool)
                mask[:, 2] = False
                obs = TensorDict({"policy": inputs, "action_mask": mask}, [count], device="cuda")
                expected = policy._probabilities_from_logits(policy._forward(obs), mask)
                torch.testing.assert_close(graph.action_probabilities(obs), expected, rtol=0, atol=0)
                torch.testing.assert_close(graph.act_inference(obs), expected.argmax(-1, keepdim=True), rtol=0, atol=0)
        self.assertEqual(len(graph.graphs), 5)
        self.assertTrue(policy.compile_forward)
        torch.testing.assert_close(policy.actor.weight, weights, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng, rtol=0, atol=0)
