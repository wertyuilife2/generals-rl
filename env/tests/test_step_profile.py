import json
from pathlib import Path
import tempfile
import unittest

from generals import MapConfig
from generals.benchmarks.contracts import BenchmarkCase
from generals.benchmarks.profiling import parse_cuda_trace, union_us, profile_case


class StepProfileTests(unittest.TestCase):
    def test_cpu_samples_real_step_and_preserves_every_output(self):
        result = profile_case(BenchmarkCase(MapConfig(9,7,4), batch_size=13, threads=3, episode_ticks=17),
                              backend='cpu', steps=61, warmup=2, repeats=1, cpu_sample_stride=3)
        profile = result['profile']
        self.assertTrue(profile['verification']['state_statistics_counters_match'])
        self.assertEqual(len(profile['native_rows']), 61)
        self.assertEqual(profile['sampled_lanes'], sum(1 for tick in range(61) for lane in range(13)
                                                     if lane % 3 == tick % 3))
        self.assertAlmostEqual(sum(v['fraction'] for v in profile['sampled_worker_phases'].values()), 1)
        self.assertGreater(profile['sampled_worker_phases']['autoreset']['total_us'], 0)
        self.assertGreater(profile['mean_native_region_us'], 0)
        self.assertIn('reset', profile['by_tick_kind'])
        self.assertGreaterEqual(profile['mean_team_entry_us'], 0)
        self.assertGreaterEqual(profile['mean_worker_start_spread_us'], 0)
        self.assertGreaterEqual(profile['mean_team_join_tail_us'], 0)

    def parse(self, events, steps=1):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.json'
            path.write_text(json.dumps({'traceEvents': events}))
            return parse_cuda_trace(path, steps)

    def event(self, name, start, duration, category):
        return {'ph': 'X', 'name': name, 'ts': start, 'dur': duration, 'cat': category}

    def test_trace_filters_step_boundaries_and_gpu_duplicate_annotation(self):
        e = self.event
        result = self.parse([e('init', 0, 3, 'kernel'), e('env_step/0', 10, 30, 'user_annotation'),
            e('env_step/0', 20, 10, 'gpu_user_annotation'), e('step', 20, 10, 'kernel'),
            e('overlap', 22, 3, 'kernel'), e('cudaDeviceSynchronize', 21, 12, 'cuda_runtime'),
            e('upload', 1, 2, 'gpu_memcpy'), e('late', 50, 1, 'kernel')])
        self.assertEqual(result['kernel_count'], 2)
        self.assertEqual(result['mean_kernel_us'], 10)
        self.assertEqual(result['mean_marker_outside_kernel_us'], 20)
        self.assertEqual(result['transfers'], [])
        self.assertEqual(result['cuda_api_calls']['cudaDeviceSynchronize']['total_us'], 12)

    def test_transfer_counts_and_multiple_steps(self):
        e = self.event
        result = self.parse([e('env_step/0', 0, 10, 'user_annotation'), e('a', 2, 2, 'kernel'),
            e('HtoD', 1, 1, 'gpu_memcpy'), e('env_step/1', 20, 10, 'user_annotation'),
            e('b', 22, 4, 'kernel'), e('clear', 21, 1, 'gpu_memset')], 2)
        self.assertEqual(result['kernel_union_us_per_step'], [2, 4])
        self.assertEqual(len(result['transfers']), 2)
        self.assertAlmostEqual(result['gpu_busy_fraction_of_marked_steps'], .3)

    def test_missing_or_duplicate_markers_and_no_kernels_are_errors(self):
        e = self.event
        with self.assertRaises(ValueError): self.parse([])
        with self.assertRaises(ValueError): self.parse([e('env_step/0', 0, 3, 'user_annotation')]*2)
        with self.assertRaises(RuntimeError): self.parse([e('env_step/0', 0, 3, 'user_annotation')])
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            self.parse([e('env_step/0', 0, 5, 'user_annotation'), e('a', 1, 1, 'kernel'),
                        e('env_step/1', 10, 5, 'user_annotation')], 2)
        self.assertEqual(union_us([(5, 10), (1, 3), (2, 6), (10, 12), (3, 4)]), 11)


if __name__ == '__main__': unittest.main()
