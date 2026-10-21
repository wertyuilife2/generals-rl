"""Portable real-checkpoint arena integration and local feature equivalence."""
from copy import deepcopy
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
from generals_rl.config import task_config
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict
from rl_test_utils import small_training_config
from generals_rl.mdp import EncodedActionSpace, GameBatch, LocalObservationEncoder, ObservationHistory, make_observation_encoder
from generals_rl.mdp.knowledge import KnowledgeState
from generals_rl.task import GeneralsTask


def full_batch(state):
    arrays, width, height = pack_states([state])
    stats = compute_stats(state)
    return GameBatch({key: torch.tensor(value) for key, value in arrays.items()},
                     torch.tensor(np.stack([stats.army, stats.territory, stats.cities], axis=-1)[None]),
                     width, height)


def make_checkpoint(directory, seed=7, filename="checkpoint.pt", family="information", history_length=4):
    config = small_training_config(task_overrides={"opponent": "wait", "persistent_knowledge": True,
                                                   "max_episode_steps": 32, "history_length": history_length})
    config["runner"]["obs_groups"] = {"policy": ["policy", "knowledge"],
                                      "critic": ["critic", "policy", "knowledge"]}
    options = config["runner"]["policy"]
    options.update(history_length=history_length, knowledge_channels=11, critic_actor_information=True)
    if family == "optimized":
        options["class_name"] = "generals_rl.optimized_ppo.OptimizedInformationSpatialPPOPolicy"
    elif family == "scaled":
        options.pop("spatial_channels")
        options.update(class_name="generals_rl.scaled_ppo.ScaledInformationSpatialPPOPolicy",
                       backbone="vit", model_size="1m", spatial_mode="dense", critic_projection_channels=0)
    policy_options = deepcopy(options)
    policy_type = resolve_callable(policy_options.pop("class_name"))
    with torch.random.fork_rng(devices=[]), GeneralsTask(task_config(config["task"])) as task:
        torch.manual_seed(seed)
        inputs = TensorDict(task.get_observations(), [task.num_envs])
        policy = policy_type(inputs, config["runner"]["obs_groups"], 649, **policy_options)
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
        for width, height, players in ((3, 3, 2), (9, 7, 2), (9, 7, 4), (9, 7, 8)):
            action_space = EncodedActionSpace(width, height)
            state = generate_map(MapConfig(width, height, players, 0.1, 0.1), seed=25)
            for tick in range(4):
                full = full_batch(state)
                for seat in range(players):
                    ids = torch.tensor([seat])
                    restricted = GameBatch.from_observation(observe(state, seat), "cpu")
                    actual = encoder.encode(restricted, ids)
                    torch.testing.assert_close(actual, encoder.encode(full, ids), rtol=0, atol=0)
                    torch.testing.assert_close(action_space.mask(restricted, ids), action_space.mask(full, ids))
                    self.assertTrue(torch.isfinite(actual).all())
                CoreEngine(state).step([Action.wait()] * players)

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
                self.assertEqual(set(inputs.keys()), {"policy", "single_obs", "action_mask", "knowledge"})
                seen.append(inputs.clone())
                return original(inputs)

            controller.policy.act_inference = capture
            state = generate_map(MapConfig(9, 9, 2, 0, 0), seed=18)
            general_y, general_x = divmod(int(state.general_pos[0]), 9)
            outpost = next(cell for cell in range(81) if state.owner.flat[cell] == -1
                           and max(abs(cell // 9 - general_y), abs(cell % 9 - general_x)) > 2)
            state.owner.flat[outpost], state.army.flat[outpost] = 0, 2
            initial_state = state.copy()
            encoder, history = make_observation_encoder(task_config(config["task"])), ObservationHistory(4)
            knowledge = KnowledgeState(1, 9, 9, 2)
            actions = []
            for tick in range(3):
                batch = full_batch(state)
                frame = encoder.encode(batch, torch.tensor([0]))
                knowledge.update(batch, torch.tensor([0]))
                history.reset(frame) if tick == 0 else history.push(frame)
                actions.append(controller.act(observe(state, 0)))
                torch.testing.assert_close(seen[-1]["policy"], history.flattened(), rtol=0, atol=0)
                torch.testing.assert_close(seen[-1]["knowledge"], knowledge.encode(batch.state["tick"], torch.tensor([0])).flatten(1), rtol=0, atol=0)
                self.assertEqual(float(seen[-1]["single_obs"][0, -5]), tick / 32)
                if tick:
                    # The former outpost is now hidden; its last-seen age must
                    # still match training's 1024-tick knowledge scale.
                    self.assertAlmostEqual(float(seen[-1]["knowledge"].reshape(1, 11, 81)[0, -1, outpost]), tick / 1024)
                CoreEngine(state).step([Action.wait(), Action.wait()])
                if tick == 0:
                    state.owner.flat[outpost], state.army.flat[outpost] = -1, 0
            controller.reset(0, 7)
            initial = initial_state.copy()
            self.assertEqual(controller.act(observe(initial, 0)), actions[0])
            torch.testing.assert_close(seen[-1]["policy"], seen[0]["policy"])
            controller.reset(1, 7)
            controller.act(observe(initial, 1))
            torch.testing.assert_close(seen[-1]["single_obs"][0, -2], torch.tensor(0.))
            with self.assertRaises(ValueError):
                controller.validate_rules(MapConfig(7, 9, 2), "local")
            with self.assertRaises(ValueError):
                controller.validate_rules(MapConfig(9, 9, 2), "full")

    def test_two_real_checkpoint_versions_compete_with_a_baseline_and_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, _ = make_checkpoint(root, seed=7, filename="v1.pt")
            second, _ = make_checkpoint(root, seed=8, filename="v2.pt")
            config = {"rules": {"map": {"width": 9, "height": 9, "players": 2,
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

    def test_current_model_families_and_h8_load_without_privileged_inference(self):
        for family, history_length in (("information", 4), ("optimized", 4), ("scaled", 8)):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as tmp:
                checkpoint, _ = make_checkpoint(Path(tmp), family=family, history_length=history_length)
                controller = CTSController(checkpoint=checkpoint)
                state = generate_map(MapConfig(9, 9, 2, 0, 0), seed=18)
                action = controller.act(observe(state, 0))
                self.assertIsInstance(action, Action)
                self.assertEqual(controller.history.frames.shape, (1, history_length, 985))
                for players in (4, 8):
                    with self.assertRaises(ValueError):
                        controller.validate_rules(MapConfig(9, 9, players, 0, 0), "local")


if __name__ == "__main__":
    unittest.main()
