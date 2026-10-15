"""One authoritative simulation for bots, browser play, and verified replays."""
import gzip
import json
import numpy as np

from generals.core.mapgen import generate_map
from generals.core.observation import snapshot
from generals.core.recording import state_digest, state_from_dict, state_to_dict
from generals.core.state import Action, compute_stats
from generals import MapConfig, make_engine

from .observation import local_observation, _kernel as observation_kernel


def board_view(observation):
    return {"width": observation.width, "height": observation.height,
            "tick": observation.tick, "player_id": observation.player_id,
            "general_pos": observation.general_pos,
            **{key: getattr(observation, key).tolist() for key in
               ("terrain", "structure", "owner", "army", "visible", "alive")},
            "stats": {key: getattr(observation.stats, key).tolist() for key in
                      ("army", "territory", "cities")}}


class RecordedGame:
    def __init__(self, plan, config, controllers):
        self.plan, self.config = plan, config
        self.controllers = controllers  # None denotes a human seat.
        count = len(plan["entrant_ids"])
        bots = [controller for controller in controllers if controller is not None]
        if (len(controllers) != count or not 2 <= count <= 8 or
                len({id(controller) for controller in bots}) != len(bots) or
                len(plan["controller_seeds"]) != count):
            raise ValueError("each seat requires an independent controller and RNG stream")
        self.map_config = MapConfig(**plan["map"])
        self.state = generate_map(self.map_config, plan["map_seed"])
        self.engine = make_engine(self.state)
        observation_kernel()  # Compilation belongs to setup, never a measured tick.
        self.stats = compute_stats(self.state)
        self.initial = state_to_dict(self.state)
        self.initial_digest = state_digest(self.state)
        self.actions, self.digests = [], []
        self.eliminations = [None] * count
        self.invalid = [0] * count
        for seat, controller in enumerate(controllers):
            if controller is not None:
                controller.reset(player_id=seat, seed=plan["controller_seeds"][seat])

    @property
    def done(self):
        return self.state.terminated or self.state.tick >= self.config.max_ticks

    def advance(self, human_actions=None):
        if self.done:
            raise ValueError("game already finished")
        human_actions = human_actions or {}
        actions = [Action.wait() if not self.state.alive[seat] else
                   human_actions.get(seat, Action.wait()) if controller is None else
                   controller.act(local_observation(self.state, seat, self.stats))
                   for seat, controller in enumerate(self.controllers)]
        if not all(isinstance(action, Action) for action in actions):
            raise TypeError("controller must return an Action")
        result = self.engine.step(actions)
        self.stats = result.stats
        self.actions.append([[int(a.kind), a.source, int(a.direction), int(a.mode)] for a in actions])
        self.digests.append(state_digest(self.state))
        for move in result.moves:
            self.invalid[move.player_id] += int(not move.valid)
            controller = self.controllers[move.player_id]
            if controller is not None:
                controller.on_result(move)
            if move.eliminated is not None:
                self.eliminations[move.eliminated] = self.state.tick
                eliminate = getattr(self.controllers[move.eliminated], "eliminate", None)
                if eliminate is not None:
                    eliminate()

    def result(self):
        if not self.done:
            raise ValueError("cannot score an unfinished game")
        stats = self.stats
        return {**self.plan, "winner_seat": self.state.winner_id,
                "ticks": self.state.tick, "reason": "capture" if self.state.terminated else "tick_limit",
                "elimination_ticks": self.eliminations, "invalid_actions": self.invalid,
                "final_alive": self.state.alive.tolist(), "initial_digest": self.initial_digest,
                "final_digest": state_digest(self.state),
                "final_stats": {key: getattr(stats, key).tolist() for key in ("army", "territory", "cities")}}

    def replay_blob(self):
        return gzip.compress(json.dumps({"schema": 1, "initial": self.initial,
                                        "actions": self.actions, "digests": self.digests},
                                       separators=(",", ":")).encode(), mtime=0)

    def view(self, seat=None):
        return board_view(snapshot(self.state, seat))


def replay_frames(blob, result, stride=1, compact=False):
    """Verify every state, even when only a subset of frames is requested."""
    trace = json.loads(gzip.decompress(blob))
    if trace["schema"] != 1 or len(trace["actions"]) != result["ticks"]:
        raise ValueError("invalid replay length or schema")
    state = state_from_dict(trace["initial"])
    if state_digest(state) != result["initial_digest"]:
        raise ValueError("replay initial digest mismatch")
    engine = make_engine(state)
    frames = [board_view(snapshot(state))]
    previous = np.stack([state.owner.ravel(), state.army.ravel(), state.structure.ravel()], axis=1).copy()
    if len(trace["digests"]) != len(trace["actions"]):
        raise ValueError("replay digest count mismatch")
    for actions, expected_digest in zip(trace["actions"], trace["digests"]):
        engine.step([Action(*values) for values in actions])
        if state_digest(state) != expected_digest:
            raise ValueError(f"replay diverged at tick {state.tick}")
        if state.tick % stride == 0 or state.tick == result["ticks"]:
            if compact:
                current = np.stack([state.owner.ravel(), state.army.ravel(), state.structure.ravel()], axis=1)
                changed = np.flatnonzero(np.any(current != previous, axis=1))
                stats = compute_stats(state)
                frames.append({"tick": state.tick,
                               "cells": [[int(i), *current[i].tolist()] for i in changed],
                               "alive": state.alive.tolist(),
                               "stats": {key: getattr(stats, key).tolist() for key in ("army", "territory", "cities")}})
                previous = current.copy()
            else:
                frames.append(board_view(snapshot(state)))
    if state_digest(state) != result["final_digest"]:
        raise ValueError("replay final digest mismatch")
    return frames
