"""V8 preserves the observed tactical safety requirements of earlier releases."""
import unittest
import subprocess
import tempfile
import gzip
import json
from collections import deque
from pathlib import Path
import test_v6


class RazorV8Regressions(test_v6.RazorV6Regressions):
    version = 'v8'

    def test_does_not_route_a_recorded_field_column_into_a_visible_interception(self):
        fixture = json.loads(gzip.decompress(
            Path(__file__).with_name('human_failure_cases.json.gz').read_bytes()))
        case = next(x for x in fixture['cases']
                    if x['name'] == 'field_army_lost_in_reversal')
        values = [f['observation'] for f in case['frames']
                  if f['observation']['tick'] in (810, 811, 812)]
        self.assertEqual(len(values), 3)
        obs, action = self.run_history(values, case['seat'])
        if int(action.kind) and action.source == 271:
            target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
            # Both landing squares are one legal move from the visible 179
            # troops. Changing the old route to the other square is no fix.
            self.assertNotIn(target, (249, 272))

    def test_preserves_the_front_column_in_recorded_unsupported_city_purchase(self):
        fixture = json.loads(gzip.decompress(
            Path(__file__).with_name('human_failure_cases.json.gz').read_bytes()))
        case = next(x for x in fixture['cases']
                    if x['name'] == 'front_city_purchase_before_counterattack')
        values = [f['observation'] for f in case['frames']
                  if 369 <= f['observation']['tick'] <= 372]
        self.assertEqual(len(values), 4)
        obs, action = self.run_history(values, case['seat'])
        # Only the observations available BEFORE the 87-troop column appears
        # are supplied. This checks the spending decision under uncertainty;
        # it does not supply the next frame or assert a counterfactual win.
        if int(action.kind):
            target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
            self.assertNotEqual((action.source, target), (178, 179))

    def test_starts_reinforcing_before_a_recorded_human_column_reaches_tactical_radius(self):
        fixture = json.loads(gzip.decompress(
            Path(__file__).with_name('human_failure_cases.json.gz').read_bytes()))
        case = next(x for x in fixture['cases']
                    if x['name'] == 'larger_total_without_deliverable_defense')
        values = [f['observation'] for f in case['frames']
                  if f['observation']['tick'] in (1288, 1312)]
        self.assertEqual(len(values), 2)
        obs, action = self.run_history(values, case['seat'])
        # At 1312 the visible city holds 490 troops sixteen steps away.
        # This tests timely preparation, not a claim that this move saves
        # the historical game or that the opponent must choose that route.
        self.assert_reinforces_home(obs, action)

    def test_rebuilds_home_after_a_defeated_recorded_raid(self):
        fixture = json.loads(gzip.decompress(
            Path(__file__).with_name('exposed_home_recovery_history.json.gz').read_bytes()))
        obs, action = self.run_history(fixture['observations'], fixture['seat'])
        self.assert_reinforces_home(obs, action)

    def test_rebuilds_home_in_recorded_human_two_wave_failure(self):
        fixture = json.loads(gzip.decompress(
            Path(__file__).with_name('human_failure_cases.json.gz').read_bytes()))
        case = next(x for x in fixture['cases'] if x['name'] == 'exposed_home_between_raids')
        values = [f['observation'] for f in case['frames']
                  if f['observation']['tick'] in (128, 305)]
        self.assertEqual(len(values), 2)
        obs, action = self.run_history(values, case['seat'])
        self.assert_reinforces_home(obs, action)

    def assert_reinforces_home(self, obs, action):
        self.assertNotEqual(int(action.kind), 0)
        target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
        # In the recording the one-troop general was left behind while a
        # remote column continued raiding. A collection may start with a
        # more distant city; require actual progress along our return route.
        owner = obs.owner.ravel()
        distance = [obs.size + 1] * obs.size
        distance[obs.general_pos] = 0
        queue = deque([obs.general_pos])
        while queue:
            c = queue.popleft()
            y, x = divmod(c, obs.width)
            for yy, xx in ((y-1,x),(y,x+1),(y+1,x),(y,x-1)):
                if not (0 <= yy < obs.height and 0 <= xx < obs.width):
                    continue
                v = yy * obs.width + xx
                if owner[v] != obs.player_id or distance[v] <= distance[c]+1:
                    continue
                distance[v] = distance[c]+1
                queue.append(v)
        self.assertEqual(owner[target], obs.player_id)
        self.assertLess(distance[target], distance[action.source])
        self.assertLessEqual(distance[action.source], 12)
        self.assertGreater(int(obs.army.flat[action.source]), 10)

    def test_supply_program_conserves_army_and_does_not_reward_pure_relocation(self):
        # Compile against v8 rather than accidentally checking the frozen v6.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parent
            source = Path(tmp) / 'supply.cpp'
            old = (root / 'test_v6_supply.cpp').read_text()
            source.write_text(old.replace('../v6/policy.cpp', str(root.parent / 'v8/policy.cpp')))
            binary = Path(tmp) / 'supply'
            subprocess.run(['c++', '-std=c++17', '-O2', '-I',
                            str(test_v6.PROJECT / 'arena/src/generals_arena/sdk'),
                            str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_production_retention_and_reconnaissance_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'operations'
            subprocess.run(['c++', '-std=c++17', '-O2', '-I',
                            str(test_v6.PROJECT / 'arena/src/generals_arena/sdk'),
                            str(Path(__file__).with_name('test_v8_operations.cpp')),
                            '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
