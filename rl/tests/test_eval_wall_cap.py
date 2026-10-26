"""CPU-only checks for shared-GPU evaluation wall limits and process cleanup."""
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import shutil
import signal
import subprocess
from unittest.mock import Mock, call, patch

import pytest

from generals_rl.training_evaluation import PeriodicEvaluator, TrainingEvalConfig


@pytest.fixture
def harness(tmp_path):
    value = Mock()
    value.clock, value.copy_seconds, value.cache_seconds = 100.0, 0.0, 0.0
    value.source = tmp_path / 'checkpoint.pt'
    value.source.write_bytes(b'immutable synthetic checkpoint, not a model')
    value.process = Mock(pid=123)
    value.process.returncode = None
    value.process.poll.return_value = None
    original_copy = shutil.copyfile

    def timed_copy(source, target):
        result = original_copy(source, target)
        value.clock += value.copy_seconds
        return result

    def timed_cache():
        value.clock += value.cache_seconds

    def export(checkpoint, output, **kwargs):
        Path(output).mkdir()
        return Path(output)

    with ExitStack() as stack:
        stack.enter_context(patch('generals_rl.training_evaluation.perf_counter', side_effect=lambda: value.clock))
        value.copy = stack.enter_context(patch('generals_rl.training_evaluation.shutil.copyfile', side_effect=timed_copy))
        value.export = stack.enter_context(patch('generals_rl.export.export_policy', side_effect=export))
        stack.enter_context(patch('generals_arena.policies.artifacts.validate_bundle',
                                  side_effect=lambda bundle: {'policy_id': sha256(str(bundle).encode()).hexdigest()}))
        value.popen = stack.enter_context(patch('generals_rl.training_evaluation.subprocess.Popen', return_value=value.process))
        value.killpg = stack.enter_context(patch('generals_rl.training_evaluation.os.killpg'))
        stack.enter_context(patch('generals_rl.training_evaluation.torch.cuda.is_initialized', return_value=True))
        value.synchronize = stack.enter_context(patch('generals_rl.training_evaluation.torch.cuda.synchronize'))
        value.empty_cache = stack.enter_context(patch('generals_rl.training_evaluation.torch.cuda.empty_cache', side_effect=timed_cache))

        def make(name='run', **kwargs):
            settings = TrainingEvalConfig(every_n_updates=1, final_top_k=0, **kwargs)
            config = {'task': {'width': 9, 'height': 9, 'players': 2}}
            return PeriodicEvaluator(settings, config, tmp_path / name, Mock())

        def schedule(evaluator, remaining, updates=1):
            with patch.object(evaluator, 'drain'):
                return evaluator.on_checkpoint(value.source, completed_updates=updates,
                    steps_per_environment=updates * 32, environment_steps=updates * 1024,
                    remaining_seconds=remaining)

        value.make, value.schedule = make, schedule
        yield value


def test_insufficient_budget_has_no_copy_export_job_or_worker(harness):
    evaluator = harness.make(device='cuda:0')
    index = (evaluator.directory / 'index.json').read_bytes()
    assert not harness.schedule(evaluator, 10)
    assert evaluator.budget_exhausted and not evaluator.jobs and not evaluator.pending
    assert (evaluator.directory / 'index.json').read_bytes() == index
    for function in (harness.copy, harness.export, harness.popen, harness.synchronize, harness.empty_cache):
        function.assert_not_called()


@pytest.mark.parametrize('configured,remaining,estimate,copy,cache,deadline,timeout,reason', [
    (30., 80., 30., 20., 3., 130., 30., 'evaluation_timeout'),
    (100., 50., 35., 3., 2., 140., 40., 'wall_budget'),
])
def test_absolute_deadline_includes_copy_export_and_cache(harness, configured, remaining, estimate,
                                                        copy, cache, deadline, timeout, reason):
    evaluator = harness.make(device='cuda:0', timeout_seconds=configured)
    harness.copy_seconds, harness.cache_seconds = copy, cache
    with patch.object(evaluator, '_estimated_worker_budget_seconds', return_value=estimate):
        assert harness.schedule(evaluator, remaining)
    job, process, started = evaluator.running
    assert job['deadline'] == deadline and job['timeout_seconds'] == timeout
    assert job['deadline_reason'] == reason
    assert started == 100 + copy + cache and started < deadline
    assert process is harness.process
    evaluator.running = None


@pytest.mark.parametrize('copy,cache,expected_cache_calls', [(41., 0., 0), (2., 39., 2)])
def test_copy_or_cache_exhaustion_cancels_without_worker_and_bounds_retention(harness, copy, cache, expected_cache_calls):
    evaluator = harness.make(device='cuda:0', timeout_seconds=100, keep_snapshots=1)
    harness.copy_seconds, harness.cache_seconds = copy, cache
    with patch.object(evaluator, '_estimated_worker_budget_seconds', return_value=35):
        for updates in (1, 2):
            assert harness.schedule(evaluator, 50, updates)
            assert evaluator.jobs[-1]['status'] == 'cancelled'
            assert evaluator.jobs[-1]['reason'] == 'wall_budget'
    first, second = evaluator.jobs
    assert not Path(first['checkpoint']).exists() and first['checkpoint_retained'] is False
    assert not Path(first['bundle']).exists()
    assert Path(second['checkpoint']).exists() and second['checkpoint_retained'] is True
    assert Path(second['bundle']).is_dir()
    assert not evaluator.pending and evaluator.running is None
    assert harness.empty_cache.call_count == expected_cache_calls
    harness.popen.assert_not_called()


def test_cpu_evaluation_ignores_learner_remaining_budget_and_runs_async(harness):
    evaluator = harness.make(device='cpu')
    assert harness.schedule(evaluator, 0)
    job, process, _ = evaluator.running
    assert job['deadline'] is None and job['timeout_seconds'] is None
    assert process is harness.process and not evaluator.budget_exhausted
    assert harness.popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
    harness.synchronize.assert_not_called()
    harness.empty_cache.assert_not_called()
    evaluator.running = None


def test_running_worker_is_reaped_at_absolute_deadline(harness):
    evaluator = harness.make(device='cuda:0', timeout_seconds=100)
    harness.copy_seconds, harness.cache_seconds = 38., 1.
    with patch.object(evaluator, '_estimated_worker_budget_seconds', return_value=35):
        assert harness.schedule(evaluator, 50)
    job, _, started = evaluator.running
    assert job['deadline'] == 140 and started == 139
    harness.clock = 141

    def wait(timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired('mock-worker', timeout)
        harness.process.returncode = -9
        return -9

    harness.process.wait.side_effect = wait
    evaluator.poll()
    assert evaluator.running is None and job['status'] == 'failed'
    assert 'TimeoutError' in job['error']
    assert (job['returncode'], job['worker_seconds']) == (-9, 2.)
    assert harness.killpg.call_args_list == [call(123, signal.SIGTERM), call(123, signal.SIGKILL)]
    assert evaluator.best_completed is None and evaluator.latest_completed is None


def test_admission_estimate_uses_recent_finite_successes_without_changing_best(harness):
    evaluator = harness.make(device='cuda:0')
    evaluator.jobs = [{'status': 'completed', 'stage': 'periodic', 'worker_seconds': value}
                      for value in (1000., 5., 8., 6.)]
    assert evaluator._estimated_worker_budget_seconds() == 40
    evaluator.best_completed = {'policy_id': 'selected', 'elo': 200}
    before = list(evaluator.jobs)
    assert not harness.schedule(evaluator, 46)
    assert evaluator.jobs == before and evaluator.best_completed['policy_id'] == 'selected'
    harness.copy.assert_not_called()
    harness.popen.assert_not_called()
