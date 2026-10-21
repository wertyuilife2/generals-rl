"""Rectangular combat geometry, privacy and legacy numeric contracts."""
from hashlib import sha256
import unittest

import numpy as np
import torch

from generals import VecEnv
from generals.core.engine import CoreEngine
from generals.core.state import GameState, MapLayout
from generals_rl.mdp import GameBatch, LocalObservationEncoder, make_observation_encoder
from generals_rl.mdp.combat import _neighbor_max
from generals_rl.task import GeneralsTask, TaskConfig


torch.set_num_threads(1)


def geometry_state(width=7, height=11, tick=0):
    shape = (height, width)
    terrain = np.zeros(shape, np.uint8)
    structure = np.zeros(shape, np.uint8)
    owner = np.full(shape, -1, np.int16)
    army = np.zeros(shape, np.int64)
    center = (height // 2) * width + width // 2
    for cell, pid, troops, kind in (
        (0, 0, 3, 2), (width * height - 1, 1, 8, 2),
        (center, 0, 3, 0), (center - 1, 0, 9, 0),
        (center - width, 1, 4, 0), (center + 1, 1, 11, 1),
        (center + width, -2, 19, 1),
    ):
        owner.flat[cell], army.flat[cell], structure.flat[cell] = pid, troops, kind
    terrain.flat[center - width - 1] = 1
    state = GameState(MapLayout(terrain), structure, owner, army, np.ones(2, bool),
                      np.array([0, width * height - 1], np.int32), tick=tick)
    state.validate()
    return state


def visible_cells(state, pid):
    """Independent visibility reference using coordinates, not pooled planes."""
    visible = np.zeros(state.owner.shape, bool)
    height, width = visible.shape
    for y, x in np.argwhere(state.owner == pid):
        for yy in range(max(0, y - 1), min(height, y + 2)):
            for xx in range(max(0, x - 1), min(width, x + 2)):
                visible[yy, xx] = True
    return visible


def combat_reference(state, pid, privileged):
    """Native after-growth troops and explicit cardinal source enumeration."""
    engine = CoreEngine(state.copy())
    engine.begin_tick()
    after = engine.state
    visible = np.ones(after.owner.shape, bool) if privileged else visible_cells(state, pid)
    height, width = after.owner.shape
    risk, margin = np.zeros(after.owner.shape), np.zeros(after.owner.shape)
    for y in range(height):
        for x in range(width):
            if not visible[y, x] or after.layout.terrain[y, x] == 1:
                continue
            friendly, hostile = 0, 0
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if not (0 <= yy < height and 0 <= xx < width and visible[yy, xx]):
                    continue
                movable = max(0, int(after.army[yy, xx]) - 1)
                if after.owner[yy, xx] == pid:
                    friendly = max(friendly, movable)
                elif after.owner[yy, xx] == 1 - pid:
                    hostile = max(hostile, movable)
            risk[y, x] = hostile
            defender = int(after.army[y, x])
            margin[y, x] = (defender + friendly - hostile if after.owner[y, x] == pid
                            else friendly - defender)
    risk, margin = torch.tensor(risk, dtype=torch.float32), torch.tensor(margin, dtype=torch.float32)
    return (torch.log1p(risk).flatten() / 10,
            (margin.sign() * torch.log1p(margin.abs())).flatten() / 10)


class VariableGeometryTests(unittest.TestCase):
    def batch(self, states):
        env = VecEnv.from_states(states, device="cpu", auto_reset=False, threads=1)
        self.addCleanup(env.close)
        return GameBatch.from_backend(env, "cpu")

    def test_cardinal_neighbor_max_uses_rectangular_coordinates(self):
        for width, height in ((7, 11), (13, 13)):
            cells = width * height
            values = torch.arange(2 * 2 * cells).reshape(2, 2, cells).float()
            actual = _neighbor_max(values, width, height)
            expected = torch.empty_like(values)
            for y in range(height):
                for x in range(width):
                    neighbors = [yy * width + xx for yy, xx in
                                 ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                                 if 0 <= yy < height and 0 <= xx < width]
                    expected[:, :, y * width + x] = values[:, :, neighbors].amax(-1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        values = torch.arange(2 * 81).reshape(1, 2, 81).float()
        torch.testing.assert_close(_neighbor_max(values), _neighbor_max(values, 9, 9), rtol=0, atol=0)

    def test_rectangular_combat_matches_native_growth_and_manual_neighbors(self):
        for width, height in ((7, 11), (13, 13)):
            states = [geometry_state(width, height, tick) for tick in (0, 1, 49, 50)]
            batch = self.batch(states)
            ids = torch.tensor([0, 1, 0, 1])
            encoder = make_observation_encoder(TaskConfig(width=width, height=height, combat_features=True))
            cells = width * height
            for privileged in (False, True):
                encoded = encoder.encode(batch, ids, privileged=privileged)
                self.assertEqual(tuple(encoded.shape), (4, 12 * cells + 13))
                for lane, state in enumerate(states):
                    expected_risk, expected_margin = combat_reference(state, int(ids[lane]), privileged)
                    torch.testing.assert_close(encoded[lane, 10 * cells:11 * cells],
                                               expected_risk, rtol=0, atol=0)
                    torch.testing.assert_close(encoded[lane, 11 * cells:12 * cells],
                                               expected_margin, rtol=0, atol=0)

    def test_corner_king_distance_uses_map_diameter_and_each_seat(self):
        for width, height in ((7, 11), (13, 13)):
            shape = (height, width)
            owner, army = np.full(shape, -1, np.int16), np.zeros(shape, np.int64)
            structure, terrain = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
            for y, x, pid, troops, kind in ((0, 0, 0, 3, 2), (height - 1, width - 1, 1, 8, 2),
                    (height - 2, width - 2, 0, 2, 0), (height - 1, width - 2, 1, 2, 0)):
                owner[y, x], army[y, x], structure[y, x] = pid, troops, kind
            state = GameState(MapLayout(terrain), structure, owner, army, np.ones(2, bool),
                              np.array([0, width * height - 1], np.int32))
            state.validate()
            batch = self.batch([state, state.copy()])
            ids = torch.tensor([0, 1])
            diameter = width + height - 2
            for combat in (False, True):
                encoder = make_observation_encoder(TaskConfig(width=width, height=height,
                                   combat_features=combat, king_threat_distance=True))
                expected = torch.tensor([(diameter - 1) / diameter, 2 / diameter])
                for privileged in (False, True):
                    actual = encoder.encode(batch, ids, privileged=privileged)
                    torch.testing.assert_close(actual[:, -1], expected, rtol=0, atol=0)

    def test_rectangular_hidden_mutations_cannot_change_local_combat_or_distance(self):
        for width, height in ((7, 11), (13, 13)):
            first = geometry_state(width, height)
            second = first.copy()
            hidden_target, old_general = (height - 1) * width, width * height - 1
            for values in (second.owner, second.army, second.structure):
                values.flat[hidden_target], values.flat[old_general] = values.flat[old_general], values.flat[hidden_target]
            second.general_pos[1] = hidden_target
            terrain = second.layout.terrain.copy()
            terrain[0, width - 1] = 1
            second.layout = MapLayout(terrain)
            second.validate()
            batch = self.batch([first, second])
            ids = torch.zeros(2, dtype=torch.long)
            encoder = make_observation_encoder(TaskConfig(width=width, height=height,
                                          combat_features=True, king_threat_distance=True))
            local = encoder.encode(batch, ids)
            torch.testing.assert_close(local[0], local[1], rtol=0, atol=0)
            privileged = encoder.encode(batch, ids, privileged=True)
            self.assertFalse(torch.equal(privileged[0], privileged[1]))
            torch.testing.assert_close(encoder.encode(batch, ids), local, rtol=0, atol=0)

    def test_variable_task_shapes_and_two_player_guards(self):
        for width, height in ((7, 11), (13, 13)):
            config = TaskConfig(width=width, height=height, num_envs=1, combat_features=True,
                               king_threat_distance=True, persistent_knowledge=True,
                               rotate_seats=False, opponent="wait")
            env = VecEnv.from_states([geometry_state(width, height)], device="cpu",
                                     auto_reset=False, threads=1)
            task = GeneralsTask(config, backend=env)
            self.addCleanup(task.close)
            result = task.step(torch.zeros(1, dtype=torch.long))
            cells = width * height
            self.assertEqual(tuple(result.observations["policy"].shape), (1, 4 * (12 * cells + 13)))
            self.assertEqual(tuple(result.observations["critic"].shape), (1, 12 * cells + 13))
            self.assertEqual(tuple(result.observations["action_mask"].shape), (1, 1 + 8 * cells))
            self.assertEqual(tuple(result.observations["knowledge"].shape), (1, 11 * cells))
            self.assertIn("spatial_v1", task.observation_encoder.semantics)
        for flag in ("combat_features", "king_threat_distance"):
            with self.assertRaises(ValueError):
                TaskConfig(players=3, **{flag: True})

    def test_legacy_nine_by_nine_outputs_remain_bit_identical(self):
        # Captured from the original fixed-9x9 encoders before generalization.
        # These hash complete planes and public features across both seats,
        # growth phases, local/privileged views and both base feature layouts.
        expected = {
            (False, False): "2d596ebc7e227b4549b7f35b88bf555c1784453d858606ca879b3ce986d56941",
            (False, True): "2d8f84dcd4193e29bf40d701f0afb9427266bf4c1675535e482fc8a6de23c176",
            (True, False): "bc7d9b060e616ef6f67cbda0613cffa29997dac890ac3a67d2c4865e71c26f23",
            (True, True): "ee9ff1209cec1bcd1a16c9043626a1583fb61394b5b609424a3d60ec5c02e8af",
        }
        batch = self.batch([geometry_state(9, 9, tick) for tick in (0, 1, 49, 50)])
        ids = torch.tensor([0, 1, 0, 1])
        for (combat, privileged), digest in expected.items():
            encoder = make_observation_encoder(TaskConfig(combat_features=combat, king_threat_distance=True))
            prefix = "combat" if combat else "local"
            self.assertEqual(encoder.semantics, f"{prefix}_p2_985_v1_king_threat_distance_v1")
            encoded = encoder.encode(batch, ids, privileged=privileged)
            self.assertEqual(sha256(encoded.contiguous().numpy().tobytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
