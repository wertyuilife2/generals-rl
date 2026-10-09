"""CTS student as an observation-only controller for any reference match runner."""
import json
from pathlib import Path

import torch

from generals import MapConfig
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.protocol.actions import Action
from .config import task_config
from .inference import load_student, local_inputs
from .mdp import EncodedActionSpace, GameBatch, LocalObservationEncoder, ObservationHistory


class CTSController:
    """Deterministic local student; cached model and independently reset history."""

    def __init__(self, *, checkpoint, config_path=None, device="cpu", player_id=0, seed=0):
        self.checkpoint = Path(checkpoint)
        config_path = Path(config_path) if config_path else self.checkpoint.with_name("run_config.json")
        self.config = json.loads(config_path.read_text())
        self.task = task_config(self.config["task"])
        if self.task.players != 2:
            raise ValueError("arena CTS checkpoints must be trained for two players")
        self.device = torch.device(device)
        self.encoder = LocalObservationEncoder(self.task.max_episode_steps)
        self.action_space = EncodedActionSpace(self.task.width, self.task.height)
        example = observe(generate_map(MapConfig(self.task.width, self.task.height, 2), seed=0), 0)
        batch = GameBatch.from_observation(example, self.device)
        seats = torch.zeros(1, dtype=torch.long, device=self.device)
        history = ObservationHistory(self.task.history_length)
        history.reset(self.encoder.encode(batch, seats))
        inputs = local_inputs(batch, seats, self.encoder, history, self.action_space)
        # Construct on CPU before moving weights; preserve the caller's RNG.
        # Eager loading lets the arena verify artifact hashes around the read.
        with torch.random.fork_rng(devices=[]):
            self.policy = load_student(self.checkpoint, self.config, inputs, self.device)
        self.reset(player_id, seed)

    def validate_rules(self, map_config, visibility):
        if (map_config.width, map_config.height, map_config.players) != (
                self.task.width, self.task.height, self.task.players):
            raise ValueError("arena board shape/player count does not match the CTS checkpoint")
        if visibility != "local":
            raise ValueError("CTS student evaluation requires local visibility")

    def reset(self, player_id, seed):
        if player_id not in (0, 1):
            raise ValueError("invalid CTS player seat")
        self.player_id = player_id
        self.history = ObservationHistory(self.task.history_length)
        self.player_ids = torch.tensor([player_id], device=self.device, dtype=torch.long)

    def act(self, observation):
        if observation.player_id != self.player_id:
            raise ValueError("observation belongs to another player")
        if observation.visibility != "local" or (observation.width, observation.height) != (
                self.task.width, self.task.height):
            raise ValueError("observation does not match the CTS local input contract")
        if observation.terminated or not observation.alive[self.player_id]:
            return Action.wait()
        batch = GameBatch.from_observation(observation, self.device)
        frame = self.encoder.encode(batch, self.player_ids)
        if self.history.frames is None:
            self.history.reset(frame)
        else:
            self.history.push(frame)
        inputs = local_inputs(batch, self.player_ids, self.encoder, self.history, self.action_space)
        with torch.inference_mode():
            selected = self.policy.act_inference(inputs)
            action = self.action_space.validate(selected, inputs["action_mask"])
        return Action.decode(int(action.item()))

    def on_result(self, result):
        pass
