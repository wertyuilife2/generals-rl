"""Portable real-checkpoint arena integration and local feature equivalence."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from generals import Action, MapConfig
from generals.core.batch import pack_states
from generals.core.engine import CoreEngine
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.state import compute_stats
from generals_arena.entrants import prepare_entrant
from generals_arena.tournament import rebuild, run_tournament
from generals_rl.arena_adapter import CTSController
from generals_rl.cts import runner_config
from generals_rl.inference import local_inputs
from generals_rl.mdp import EncodedActionSpace, GameBatch, LocalObservationEncoder, ObservationHistory
from generals_rl.task import TaskConfig
from rsl_rl.modules import ActorCriticMoECTS


def full_batch(state):
    arrays, width, height = pack_states([state])
    stats = compute_stats(state)
    return GameBatch({key: torch.tensor(value) for key, value in arrays.items()},
                     torch.tensor(np.stack([stats.army, stats.territory, stats.cities], axis=-1)[None]),
                     width, height)


def make_checkpoint(directory, seed=7, filename="checkpoint.pt"):
    config = {"task": asdict(TaskConfig(width=3, height=3, num_envs=4, max_episode_steps=32,
                                        history_length=2, mountain_density=0, city_density=0)),
              "runner": runner_config(hidden_dim=16, expert_num=2), "torch_threads": 1}
    state = generate_map(MapConfig(3, 3, 2, 0, 0), 0)
    batch, seats = full_batch(state), torch.tensor([0])
    encoder, history = LocalObservationEncoder(32), ObservationHistory(2)
    history.reset(encoder.encode(batch, seats))
    inputs = local_inputs(batch, seats, encoder, history, EncodedActionSpace(3, 3))
    inputs["critic"] = encoder.encode(batch, seats, privileged=True)
    policy_config = deepcopy(config["runner"]["policy"])
    policy_config.pop("class_name")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = ActorCriticMoECTS(inputs, config["runner"]["obs_groups"], 73, **policy_config)
    checkpoint = directory / filename
    torch.save({"model_state_dict": policy.state_dict()}, checkpoint)
    (directory / "run_config.json").write_text(json.dumps(config))
    return checkpoint, config


class ArenaAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_observation_only_features_and_masks_equal_training_inputs(self):
        encoder = LocalObservationEncoder(512)
        for width, height in ((3, 3), (9, 7)):
            action_space = EncodedActionSpace(width, height)
            state = generate_map(MapConfig(width, height, 2, 0.1, 0.1), seed=25)
            for tick in range(4):
                full = full_batch(state)
                for seat in (0, 1):
                    ids = torch.tensor([seat])
                    restricted = GameBatch.from_observation(observe(state, seat), "cpu")
                    actual = encoder.encode(restricted, ids)
                    torch.testing.assert_close(actual, encoder.encode(full, ids), rtol=0, atol=0)
                    torch.testing.assert_close(action_space.mask(restricted, ids), action_space.mask(full, ids))
                    self.assertTrue(torch.isfinite(actual).all())
                CoreEngine(state).step([Action.wait(), Action.wait()])

    def test_checkpoint_controller_resets_history_preserves_rng_and_never_receives_critic(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, config = make_checkpoint(Path(tmp))
            torch.manual_seed(1729)
            before = torch.random.get_rng_state().clone()
            controller = CTSController(checkpoint=checkpoint)
            torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
            seen = []
            original = controller.policy.act_inference

            def capture(inputs):
                self.assertEqual(set(inputs.keys()), {"policy", "single_obs", "action_mask"})
                seen.append(inputs.clone())
                return original(inputs)

            controller.policy.act_inference = capture
            state = generate_map(MapConfig(3, 3, 2, 0, 0), seed=18)
            encoder, history = LocalObservationEncoder(32), ObservationHistory(2)
            actions = []
            for tick in range(3):
                frame = encoder.encode(full_batch(state), torch.tensor([0]))
                history.reset(frame) if tick == 0 else history.push(frame)
                actions.append(controller.act(observe(state, 0)))
                torch.testing.assert_close(seen[-1]["policy"], history.flattened(), rtol=0, atol=0)
                self.assertEqual(float(seen[-1]["single_obs"][0, -5]), tick / 32)
                CoreEngine(state).step([Action.wait(), Action.wait()])
            controller.reset(0, 7)
            initial = generate_map(MapConfig(3, 3, 2, 0, 0), seed=18)
            self.assertEqual(controller.act(observe(initial, 0)), actions[0])
            torch.testing.assert_close(seen[-1]["policy"], seen[0]["policy"])
            controller.reset(1, 7)
            controller.act(observe(initial, 1))
            torch.testing.assert_close(seen[-1]["single_obs"][0, -2:], torch.tensor([0., 1.]))
            with self.assertRaises(ValueError):
                controller.validate_rules(MapConfig(9, 9, 2), "local")
            with self.assertRaises(ValueError):
                controller.validate_rules(MapConfig(3, 3, 2), "full")

    def test_two_real_checkpoint_versions_compete_with_a_baseline_and_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, _ = make_checkpoint(root, seed=7, filename="v1.pt")
            second, _ = make_checkpoint(root, seed=8, filename="v2.pt")
            config = {"rules": {"map": {"width": 3, "height": 3, "players": 2,
                                        "mountain_density": 0, "city_density": 0}, "max_ticks": 8},
                      "maps_per_pair": 1,
                      "entrants": [{"name": "v1", "kind": "rl", "checkpoint": str(first)},
                                   {"name": "v2", "kind": "rl", "checkpoint": str(second)},
                                   {"name": "baseline", "kind": "baseline", "policy": "aggressive"}]}
            result = run_tournament(config, root / "arena")
            self.assertEqual(result["completed_games"], 6)
            self.assertTrue(result["complete"])
            self.assertTrue(all(row["games"] == 4 for row in result["ranking"]))
            self.assertEqual(result, rebuild(root / "arena"))
            manifest = json.loads((root / "arena/entrants.json").read_text())
            self.assertNotEqual(manifest[0]["artifact_sha256"]["checkpoint"], manifest[1]["artifact_sha256"]["checkpoint"])
            # Renaming a path does not change the registered model version.
            moved = root / "renamed.pt"
            moved.write_bytes(first.read_bytes())
            relocated = {**config["entrants"][0], "checkpoint": str(moved)}
            self.assertEqual(prepare_entrant(relocated)["entrant_id"], manifest[0]["entrant_id"])
            # Overwriting the same checkpoint path cannot silently change an entrant.
            first.write_bytes(second.read_bytes())
            with self.assertRaisesRegex(ValueError, "existing entrants"):
                run_tournament(config, root / "arena", resume=True)


if __name__ == "__main__":
    unittest.main()
