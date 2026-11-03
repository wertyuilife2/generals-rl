"""The step benchmark is a measurement client, never another game engine."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from generals.benchmarks.contracts import BenchmarkCase
from generals.benchmarks.measurement import create_step_env, verify_final, by_tick_kind
from generals.benchmarks.step import measure_step_case
from generals.benchmarks.workload import prepare_workload, array_digest, snapshot
from generals.core.batch import unpack_states
from generals.core.engine import CoreEngine
from generals.core.state import Action, compute_stats
from generals import MapConfig


class StepWorkloadTests(unittest.TestCase):
    def case(self, **kwargs):
        return BenchmarkCase(MapConfig(25, 25, 4), batch_size=5, threads=2, episode_ticks=17, **kwargs)

    def test_configuration_and_bad_parameters(self):
        for kw in ({'batch_size': 0}, {'threads': -1}, {'episode_ticks': 0}, {'seed': -1}):
            with self.assertRaises(ValueError): BenchmarkCase(MapConfig(), **kw)
        for kw in ({'steps': 0}, {'warmup': -1}, {'repeats': 0}):
            with patch('generals.benchmarks.step.prepare_workload') as prepare:
                with self.assertRaises(ValueError): measure_step_case(self.case(), backend='cpu', **kw)
                prepare.assert_not_called()

    def test_policy_thread_independence_and_validation_prefix(self):
        first = prepare_workload(self.case(), 61, 2)
        second = prepare_workload(replace(self.case(), threads=1), 61, 70)
        np.testing.assert_array_equal(first.history, second.history[:61])
        self.assertEqual(first.fingerprints, second.fingerprints)
        self.assertIn('reset', first.tick_kinds)
        self.assertFalse(np.any(first.history[0]))
        self.assertGreater(np.count_nonzero(first.history), 0)

    def test_prepared_actions_match_independent_python_rules_every_tick(self):
        for config in (MapConfig(25, 25, 2), MapConfig(25, 25, 4), MapConfig(25, 25, 8)):
            case = BenchmarkCase(config, batch_size=3, threads=2, episode_ticks=17)
            work = prepare_workload(case, 61, 0)
            initial = unpack_states(work.initial, width=config.width, height=config.height)
            states = [state.copy() for state in initial]
            env, actions, _ = create_step_env(case, work.initial, work.history, backend='cpu')
            try:
                for row in actions:
                    for lane, state in enumerate(states):
                        decoded = [Action.decode(int(value)) for value in row[lane]]
                        # A batch policy acts on the PRE-TICK state, before earlier players can change it.
                        for pid, action in enumerate(decoded):
                            if int(row[lane, pid]):
                                self.assertEqual(state.owner.flat[action.source], pid)
                                self.assertGreater(state.army.flat[action.source], 1)
                        result = CoreEngine(state).step(decoded)
                        stats = result.stats
                        if result.terminated or state.tick >= case.episode_ticks:
                            states[lane] = initial[lane].copy()
                        # Save completed-tick statistics even when state resets immediately.
                        if lane == 0: expected_stats = []
                        expected_stats.append(np.stack((stats.army, stats.territory, stats.cities), axis=1))
                    env.step(row)
                    np.testing.assert_array_equal(env.stats, expected_stats)
                    for lane, state in enumerate(states):
                        for name in ('owner', 'army', 'structure', 'alive', 'general_pos'):
                            np.testing.assert_array_equal(env.state[name][lane], getattr(state, name).ravel())
                        self.assertEqual(env.state['tick'][lane], state.tick)
                verify_final(env, work.fingerprints['expected_final_state_sha256'])
            finally:
                env.close()

    def test_warmup_does_not_count_and_samples_reset(self):
        result = measure_step_case(self.case(), backend='cpu', steps=61, warmup=70, repeats=2)
        for sample in result['samples']:
            self.assertEqual(sample['counters']['env_ticks'], 61 * 5)
            self.assertEqual(len(sample['step_latency_ns']), 61)
            self.assertTrue(sample['verification']['rng_unchanged'])
            self.assertAlmostEqual(sample['env_steps_per_second'], 5e6/sample['mean_step_us'])
        self.assertEqual(result['samples'][0]['counters'], result['samples'][1]['counters'])

    def test_cleanup_on_failed_measurement_and_mismatch_detection(self):
        captured = []
        def create(*a, **kw):
            result = create_step_env(*a, **kw)
            captured.append(result[0])
            return result
        with patch('generals.benchmarks.step.create_step_env', side_effect=create), patch(
                'generals.benchmarks.step.measure_sample', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                measure_step_case(self.case(), backend='cpu', steps=3, warmup=0, repeats=1)
        self.assertTrue(captured[0].unwrapped._closed)
        work = prepare_workload(self.case(), 3, 0)
        env, _, _ = create_step_env(self.case(), work.initial, work.history, backend='cpu')
        try:
            with self.assertRaises(AssertionError): verify_final(env, work.fingerprints['expected_final_state_sha256'])
        finally: env.close()

    def test_tick_categories_account_for_all_time(self):
        result = by_tick_kind([1000, 3000, 6000], ['move_only', 'move_only', 'reset'])
        self.assertEqual(result['move_only']['mean_us'], 2)
        self.assertEqual(result['reset']['wall_fraction'], .6)
        with self.assertRaises(ValueError): by_tick_kind([1], [])


class StepCLITests(unittest.TestCase):
    def test_cpu_cli_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            cmd = [sys.executable, '-m', 'generals', 'benchmark', '--backend', 'cpu',
                   '--sizes', '25x25', '--players', '2', '4', '--batch-sizes', '2',
                   '--threads', '2', '--steps', '3', '--warmup', '1', '--repeats', '1', '--output', str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(path.read_text())
            self.assertEqual(len(data['results']), 2)
            self.assertEqual(data['workload'], 'pre_tick_owned_mobile_step_v2')
            previous = path.read_bytes()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(path.read_bytes(), previous)

    def test_source_entry_works_from_foreign_directory(self):
        script = Path(__file__).resolve().parents[1]/'benchmarks'/'benchmark_step.py'
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(script), '--backend', 'cpu', '--sizes', '25',
                '--players', '2', '--batch-sizes', '1', '--steps', '1', '--warmup', '0',
                '--repeats', '1', '--output', str(Path(directory)/'result.json')],
                cwd=directory, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
