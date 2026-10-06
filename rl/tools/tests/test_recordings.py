"""Recorded native joint ticks reproduce independently, including termination."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from generals.core.recording import replay, state_digest
from generals.core.state import Action, Direction, GameState, MapLayout
from generals_rl.evaluate_builtin import GameSpec, battle_batch
from generals_rl.task import TaskConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recordings import MatchRecorder


def tiny_state(adjacent):
    shape = (3, 3)
    positions = (0, 1) if adjacent else (0, 8)
    structure = np.zeros(shape, dtype=np.uint8)
    owner = np.full(shape, -1, dtype=np.int16)
    army = np.zeros(shape, dtype=np.int64)
    for pid, cell in enumerate(positions):
        structure.flat[cell], owner.flat[cell], army.flat[cell] = 2, pid, 1
    army.flat[0] = 20
    return GameState(MapLayout(np.zeros(shape, dtype=np.uint8)), structure, owner, army,
                     np.ones(2, dtype=bool), np.array(positions, dtype=np.int32))


class AttackStudent:
    def act_inference(self, observations):
        assert "critic" not in observations.keys()
        return torch.full((len(observations), 1), Action.move(0, Direction.RIGHT).encode(), dtype=torch.long)

    def action_probabilities(self, observations):
        return torch.zeros_like(observations["action_mask"], dtype=torch.float32).scatter_(
            1, self.act_inference(observations), 1)


class MatchRecordingTests(unittest.TestCase):
    def test_native_terminal_and_timeout_traces_match_core_and_record_every_state(self):
        specs = [GameSpec(0, "random", 0, 5, 6, 0), GameSpec(1, "random", 1, 7, 8, 0)]
        config = TaskConfig(width=3, height=3, num_envs=2, max_episode_steps=1024, history_length=4)
        with tempfile.TemporaryDirectory() as directory:
            with MatchRecorder(directory, names={0: "terminal", 1: "timeout"},
                               metadata={"snapshot_id": 340}) as recorder:
                games, _ = battle_batch(specs, [tiny_state(True), tiny_state(False)],
                                        AttackStudent(), config, max_ticks=3, recorder=recorder)
                completed = recorder.finish()
            self.assertEqual([g["outcome"] for g in games], ["win", "draw"])
            for entry, game in zip(completed, games, strict=True):
                with gzip.open(entry["frames"], "rt", encoding="utf-8") as stream:
                    data = json.load(stream)
                frames = data["frames"]
                self.assertEqual(len(frames), game["ticks"] + 1)
                self.assertEqual([f["state"]["tick"] for f in frames], list(range(game["ticks"] + 1)))
                self.assertIsNone(frames[0]["actions"])
                self.assertEqual(frames[-1]["done"], [game["terminated"], game["truncated"]])
                self.assertEqual(data["metadata"]["policy"], "student_local")
                self.assertEqual(data["metadata"]["snapshot_id"], 340)
                final = replay(entry["trace"], verify=True)
                self.assertEqual(final.winner_id, game["winner_seat"])
                self.assertEqual(state_digest(final), frames[-1]["digest"])
                self.assertTrue(entry["native_core_replay_verified"])


if __name__ == "__main__":
    unittest.main()
