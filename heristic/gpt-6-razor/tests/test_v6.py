"""V6 regressions: executable supply programs and observed tactical failures."""
from collections import deque
import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from generals.core.observation import Observation
from generals.core.state import PlayerStats
from generals.protocol.actions import ActionKind
from generals_arena.native import build_bundle, NativeController
import test_regressions as regression

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]


def observation(value):
    arrays = {k: np.array(value[k], dtype=dtype) for k, dtype in (
        ('terrain', np.uint8), ('structure', np.uint8), ('owner', np.int16),
        ('army', np.int64), ('visible', bool), ('alive', bool))}
    return Observation(player_id=value['player_id'], tick=value['tick'],
        general_pos=value['general_pos'], **arrays,
        stats=PlayerStats(**{k: np.array(v, dtype=np.int64)
                            for k, v in value['stats'].items()}))


class RazorV6Regressions(regression.RazorV5ReleaseRegressions):
    version = 'v6'

    def run_history(self, values, seat, seed=42):
        with tempfile.TemporaryDirectory() as tmp:
            source = ROOT.parent / self.version
            bundle = Path(tmp) / 'policy.zip'
            build_bundle(source / 'policy.cpp', bundle,
                         json.loads((source / 'params.json').read_text()))
            controller = NativeController(bundle=bundle)
            try:
                controller.reset(seat, seed)
                for value in values:
                    obs = observation(value)
                    action = controller.act(obs)
                self.assertEqual(controller.decision_metrics['timeouts'], 0)
                self.assertEqual(controller.decision_metrics['errors'], 0)
                return obs, action
            finally:
                controller.close()

    def test_city_purchase_preserves_margin_against_visible_merging_armies(self):
        original = json.loads(gzip.decompress(
            (ROOT / 'city_purchase_merge_observation.json.gz').read_bytes()))
        # Same live opponents, including a formerly crowded game: initial
        # player count must not disable the common tactical safety evaluator.
        for players in (2, 8):
            value = json.loads(json.dumps(original))
            value['alive'] += [False] * (players - 2)
            for field in ('army', 'territory', 'cities'):
                value['stats'][field] += [0] * (players - 2)
            obs, action = self.run_history([value], value['player_id'])
            if action.kind == ActionKind.MOVE:
                target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
                self.assertNotEqual((action.source, target), (73, 53))

    def test_route_does_not_postpone_a_known_counterattack_by_shuffling(self):
        fixture = json.loads(gzip.decompress(
            (ROOT / 'delayed_counterattack_history.json.gz').read_bytes()))
        obs, action = self.run_history(fixture['observations'], fixture['seat'], fixture['seed'])
        self.assertEqual(action.kind, ActionKind.MOVE)
        target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
        self.assertNotEqual([action.source, target], fixture['bad_transfer'])
        # A multi-order campaign may deploy a different column first. Require
        # concrete progress toward the observed front, rather than imposing
        # the old planner's particular one-step attack on the new architecture.
        owner = obs.owner.ravel()
        if owner[target] == obs.player_id:
            terrain = obs.terrain.ravel()
            distance = np.full(obs.size, obs.size + 1)
            queue = deque()
            for c in range(obs.size):
                if owner[c] >= 0 and owner[c] != obs.player_id:
                    distance[c] = 0
                    queue.append(c)
            while queue:
                c = queue.popleft()
                y, x = divmod(c, obs.width)
                for yy, xx in ((y - 1, x), (y, x + 1), (y + 1, x), (y, x - 1)):
                    if not (0 <= yy < obs.height and 0 <= xx < obs.width):
                        continue
                    v = yy * obs.width + xx
                    if terrain[v] in (1, 2) or distance[v] <= distance[c] + 1:
                        continue
                    distance[v] = distance[c] + 1
                    queue.append(v)
            self.assertLess(distance[target], distance[action.source])

    def test_supply_program_conserves_army_and_does_not_reward_pure_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'supply'
            subprocess.run(['c++', '-std=c++17', '-O2', '-I',
                str(PROJECT / 'arena/src/generals_arena/sdk'),
                str(ROOT / 'test_v6_supply.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
