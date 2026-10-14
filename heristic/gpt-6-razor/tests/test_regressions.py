"""Observed strategy failure cases, evaluated through Arena's guarded C ABI."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from generals.core.observation import Observation
from generals.core.state import PlayerStats
from generals.protocol.actions import ActionKind, MoveMode
from generals_arena.native import build_bundle, NativeController

ROOT = Path(__file__).resolve().parent


class RazorRegressions(unittest.TestCase):
    def test_front_city_launches_safe_half_attack_instead_of_shuffling_rear_army(self):
        value = json.loads((ROOT / 'city_sortie_observation.json').read_text())
        arrays = {key: np.array(value[key], dtype=dtype) for key, dtype in (
            ('terrain', np.uint8), ('structure', np.uint8), ('owner', np.int16),
            ('army', np.int64), ('visible', bool), ('alive', bool))}
        obs = Observation(player_id=value['player_id'], tick=value['tick'],
            general_pos=value['general_pos'], **arrays,
            stats=PlayerStats(**{k: np.array(v, dtype=np.int64) for k, v in value['stats'].items()}))
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / 'policy.zip'
            source = ROOT.parent / getattr(self, 'version', 'v3')
            build_bundle(source / 'policy.cpp', bundle, json.loads((source / 'params.json').read_text()))
            controller = NativeController(bundle=bundle)
            try:
                controller.reset(obs.player_id, 42)
                action = controller.act(obs)
                self.assertEqual(action.kind, ActionKind.MOVE)
                self.assertEqual(action.mode, MoveMode.HALF)
                s = action.source
                self.assertEqual(int(obs.structure.flat[s]), 1)
                target = s + (-obs.width, 1, obs.width, -1)[action.direction]
                self.assertGreaterEqual(int(obs.owner.flat[target]), 0)
                self.assertNotEqual(int(obs.owner.flat[target]), obs.player_id)
                ready = int(obs.army.flat[s])  # This even turn has no next-tick city growth.
                self.assertGreater(ready // 2, int(obs.army.flat[target]))
                for e in (s-obs.width, s+1, s+obs.width, s-1):
                    if 0 <= e < obs.size and obs.visible.flat[e] and obs.owner.flat[e] >= 0 and obs.owner.flat[e] != obs.player_id:
                        self.assertGreaterEqual(ready - ready // 2, int(obs.army.flat[e]))
                self.assertEqual(controller.decision_metrics['timeouts'], 0)
                self.assertEqual(controller.decision_metrics['errors'], 0)
            finally:
                controller.close()


class RazorV4Regressions(RazorRegressions):
    version = 'v4'

    def test_route_does_not_postpone_a_known_counterattack_by_shuffling(self):
        # This is the seat's observation history from a real Arena failure,
        # not the hidden replay board. Replay the history to restore fog memory.
        fixture = json.loads(gzip.decompress(
            (ROOT / 'delayed_counterattack_history.json.gz').read_bytes()))
        with tempfile.TemporaryDirectory() as tmp:
            source = ROOT.parent / self.version
            bundle = Path(tmp) / 'policy.zip'
            build_bundle(source / 'policy.cpp', bundle,
                         json.loads((source / 'params.json').read_text()))
            controller = NativeController(bundle=bundle)
            try:
                controller.reset(fixture['seat'], fixture['seed'])
                for value in fixture['observations']:
                    arrays = {key: np.array(value[key], dtype=dtype) for key, dtype in (
                        ('terrain', np.uint8), ('structure', np.uint8), ('owner', np.int16),
                        ('army', np.int64), ('visible', bool), ('alive', bool))}
                    obs = Observation(player_id=value['player_id'], tick=value['tick'],
                        general_pos=value['general_pos'], **arrays,
                        stats=PlayerStats(**{k: np.array(v, dtype=np.int64)
                                            for k, v in value['stats'].items()}))
                    action = controller.act(obs)
                self.assertEqual(action.kind, ActionKind.MOVE)
                target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
                self.assertNotEqual([action.source, target], fixture['bad_transfer'])
                self.assertNotEqual(int(obs.owner.flat[target]), obs.player_id)
                self.assertEqual(controller.decision_metrics['timeouts'], 0)
                self.assertEqual(controller.decision_metrics['errors'], 0)
            finally:
                controller.close()
