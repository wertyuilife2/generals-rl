"""Retained RL checkpoints as observation-only Arena controllers."""
import json
from pathlib import Path

import torch

from generals import MapConfig
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.protocol.actions import Action
from .config import task_config
from .inference import load_student, local_inputs
from .mdp import (GameBatch, KnowledgeState, ObservationHistory,
                  make_action_space, make_observation_encoder)


class CTSController:
    """Greedy local actor with independently reset history and knowledge.

    The factory name remains stable for existing Arena entrant descriptors.
    """

    def __init__(self, *, checkpoint, config_path=None, device="cpu", player_id=0, seed=0):
        self.checkpoint = Path(checkpoint)
        config_path = Path(config_path) if config_path else self.checkpoint.with_name("run_config.json")
        self.config = json.loads(config_path.read_text())
        self.task = task_config(self.config["task"])
        self.device = torch.device(device)
        self.encoder = make_observation_encoder(self.task)
        self.action_space = make_action_space(self.task)
        self.reset(player_id, seed)
        example = observe(generate_map(MapConfig(self.task.width, self.task.height, self.task.players,
                                                self.task.mountain_density, self.task.city_density), seed=0), player_id)
        inputs = self._inputs(GameBatch.from_observation(example, self.device))
        with torch.random.fork_rng(devices=[]):
            self.policy = load_student(self.checkpoint, self.config, inputs, self.device)
        self.reset(player_id, seed)

    def validate_rules(self, map_config, visibility):
        if (map_config.width, map_config.height, map_config.players) != (
                self.task.width, self.task.height, self.task.players):
            raise ValueError("arena board shape/player count does not match the checkpoint")
        if visibility != "local":
            raise ValueError("student evaluation requires local visibility")

    def reset(self, player_id, seed):
        if type(player_id) is not int or not 0 <= player_id < self.task.players:
            raise ValueError("invalid player seat")
        self.player_id = player_id
        self.history = ObservationHistory(self.task.history_length)
        self.knowledge = None
        self.player_ids = torch.tensor([player_id], device=self.device, dtype=torch.long)

    def _inputs(self, batch):
        frame = self.encoder.encode(batch, self.player_ids)
        if self.history.frames is None:
            self.history.reset(frame)
        else:
            self.history.push(frame)
        if self.task.persistent_knowledge:
            if self.knowledge is None:
                self.knowledge = KnowledgeState(1, batch.width, batch.height, batch.players, self.device)
            self.knowledge.update(batch, self.player_ids)
        return local_inputs(batch, self.player_ids, self.encoder, self.history, self.action_space,
                            self.knowledge, current=frame)

    def act(self, observation):
        if observation.player_id != self.player_id:
            raise ValueError("observation belongs to another player")
        if observation.visibility != "local" or (observation.width, observation.height, len(observation.alive)) != (
                self.task.width, self.task.height, self.task.players):
            raise ValueError("observation does not match the local input contract")
        if observation.terminated or not observation.alive[self.player_id]:
            return Action.wait()
        inputs = self._inputs(GameBatch.from_observation(observation, self.device))
        with torch.inference_mode():
            selected = self.policy.act_inference(inputs)
            action = self.action_space.validate(selected, inputs["action_mask"])
        return Action.decode(int(action.item()))

    def on_result(self, result):
        pass
