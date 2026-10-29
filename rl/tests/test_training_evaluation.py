"""Persistent Arena jobs, equal-budget Elo selection and learner isolation."""
from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from generals_arena.competitions.evaluation import expected_games

from generals_rl.train import train
from generals_rl.training_evaluation import PeriodicEvaluator, TrainingEvalConfig, _stop_worker
from rl_test_utils import assert_nested_equal, small_training_config


def small_config():
    config = small_training_config(runner_overrides={'learning_epochs': 1})
    config['self_play'] = {'pool_size': 2, 'snapshot_interval': 1}
    config['evaluation'] = {'enabled': True, 'every_n_updates': 2, 'maps_per_opponent': 1,
                            'max_ticks': 3, 'batch_size': 8, 'final_top_k': 0}
    return config


@pytest.fixture
def evaluator(tmp_path):
    return PeriodicEvaluator(TrainingEvalConfig(maps_per_opponent=1, max_ticks=3, final_top_k=0),
                             small_config(), tmp_path / 'run', None)


def enqueue(evaluator, updates, *, stage='periodic', profile=None):
    target = evaluator.directory / 'candidates' / f'synthetic_{updates}'
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / 'checkpoint.pt'
    checkpoint.write_bytes(f'synthetic full resume checkpoint {updates}'.encode())
    (target / 'run_config.json').write_text(json.dumps(evaluator.run_config))
    bundle = target / 'bundle'
    bundle.mkdir(exist_ok=True)
    return evaluator._enqueue(bundle, checkpoint, sha256(checkpoint.read_bytes()).hexdigest(), f'{updates:064x}',
                               profile or evaluator.profile, updates, updates * 4, updates * 16, stage=stage)


def report_for(job, elo=100):
    profile = json.loads(Path(job['profile_path']).read_text())
    opponents = [spec['name'] for spec in profile['opponents']]
    per_opponent = job['expected_games'] // len(opponents)
    return {'status': 'completed', 'policy_id': job['policy_id'], 'profile_id': job['profile_id'],
            'elo': elo, 'elo_ci': {'lower': elo - 20, 'upper': elo + 20, 'half_width': 20},
            'games': job['expected_games'], 'expected_games': job['expected_games'], 'errors': [],
            'device': job['device'], 'backend': job['backend'],
            'by_opponent': {name: {'games': per_opponent} for name in opponents},
            'results': [{'game_id': f'{name}-{index}', 'opponent': name}
                        for name in opponents for index in range(per_opponent)]}


def complete(evaluator, job, elo=100):
    job['worker_seconds'] = .1
    Path(job['report']).write_text(json.dumps(report_for(job, elo)))
    evaluator._complete(job)
    evaluator.pending = type(evaluator.pending)(j for j in evaluator.pending if j is not job)


def test_settings_use_one_based_updates_and_only_configure_evaluation_execution():
    value = TrainingEvalConfig(every_n_updates=3, maps_per_opponent=2, backend='cpu')
    assert value.every_n_updates == 3 and value.maps_per_opponent == 2
    assert [value.due(update) for update in (0, 1, 2, 3, 4, 6)] == [False, False, False, True, False, True]
    assert not TrainingEvalConfig(enabled=False, every_n_updates=3).due(3)
    assert 'inference' not in asdict(value)
    assert not value.synchronous and value.worker_timeout_seconds is None
    assert TrainingEvalConfig(device='cuda').synchronous
    assert TrainingEvalConfig(device='cuda').worker_timeout_seconds == 1800


@pytest.mark.parametrize('values', [
    {'enabled': 'yes'}, {'every_n_updates': 0}, {'maps_per_opponent': True}, {'batch_size': -1},
    {'max_ticks': 0}, {'seed': -1}, {'cpu_threads': 0}, {'final_top_k': -1},
    {'final_games_multiplier': 0}, {'keep_snapshots': 0}, {'backend': 'gpu'}, {'backend': 'other'},
    {'device': None}, {'device': 'invalid'}, {'device': 'mps'}, {'timeout_seconds': True},
    {'timeout_seconds': 0}, {'timeout_seconds': float('inf')},
    {'allow_shared_gpus': True}, {'allow_shared_gpus': 'yes'},
    {'device': 'cuda', 'cuda_visible_devices': ''}, {'device': 'cuda', 'cuda_visible_devices': '1,01'},
    {'device': 'cuda', 'cuda_visible_devices': '1,GPU-aaa'}, {'device': 'cuda:1', 'cuda_visible_devices': '2'},
    {'device': 'cuda', 'cuda_visible_devices': '6,7'},
    {'profile': []},
])
def test_invalid_settings_fail_early(values):
    with pytest.raises((TypeError, ValueError)):
        TrainingEvalConfig(**values)


def test_dedicated_gpu_requires_disjoint_training_visibility():
    settings = TrainingEvalConfig(device='cuda', cuda_visible_devices='7')
    assert settings.device == 'cuda:0' and not settings.synchronous
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6'}):
        settings.validate_training_devices('cuda:0')
    for visible in ('', '0,7', '0,07', 'GPU-aaaa'):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible}), pytest.raises(ValueError):
            settings.validate_training_devices('cuda:0')
    settings.validate_training_devices('cpu')
    shared = TrainingEvalConfig(device='cuda', cuda_visible_devices='0', allow_shared_gpus=True)
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}):
        shared.validate_training_devices('cuda:0')


@pytest.mark.parametrize('field,value', [
    ('policy_id', 'wrong'), ('profile_id', 'wrong'), ('status', 'partial'), ('games', 7),
    ('expected_games', 7), ('errors', ['failure']), ('device', 'cuda:0'), ('backend', 'gpu'),
    ('elo', float('nan')), ('elo', True), ('elo_ci', {'lower': 120, 'upper': 80}),
    ('results', []), ('by_opponent', {}),
])
def test_mismatched_reports_cannot_promote_best(evaluator, field, value):
    job = enqueue(evaluator, 2)
    report = report_for(job)
    report[field] = value
    Path(job['report']).write_text(json.dumps(report))
    with pytest.raises(RuntimeError):
        evaluator._complete(job)
    assert evaluator.best_completed is None
    assert not (evaluator.output / 'best_model.json').exists()


def test_best_uses_elo_and_retains_full_resume_reference(evaluator):
    first, second = enqueue(evaluator, 2), enqueue(evaluator, 4)
    complete(evaluator, first, 100)
    complete(evaluator, second, 80)
    best = json.loads((evaluator.output / 'best_model.json').read_text())
    assert best['policy_id'] == first['policy_id']
    assert best['elo'] == 100 and best['elo_ci']['lower'] == 80
    assert Path(best['checkpoint']).read_bytes().startswith(b'synthetic full resume')
    assert Path(best['bundle']).is_dir()
    assert not (evaluator.directory / 'best.pt').exists()


def test_persistent_key_deduplicates_same_model_profile(evaluator):
    job = enqueue(evaluator, 2)
    assert evaluator._enqueue(job['bundle'], job['checkpoint'], job['checkpoint_sha256'], job['policy_id'],
        evaluator.profile, 2, 8, 32, stage='periodic', final=True) is None
    assert len(evaluator.jobs) == 1 and job['final'] is True


def test_restart_recovers_queue_and_new_output_adopts_immutable_files(evaluator, tmp_path):
    job = enqueue(evaluator, 2)
    job.update(status='running', pid=99999999, deadline=1.0, timeout_seconds=.1)
    evaluator._save_index()
    old_index = (evaluator.directory / 'index.json').read_bytes()
    restarted = PeriodicEvaluator(evaluator.settings, evaluator.run_config, evaluator.output, None)
    assert restarted.jobs[0]['status'] == 'queued' and len(restarted.pending) == 1
    assert restarted.jobs[0]['deadline'] is None and restarted.jobs[0]['timeout_seconds'] is None
    resumed = PeriodicEvaluator(evaluator.settings, evaluator.run_config, tmp_path / 'resumed', None,
                                 resume=evaluator.output / 'checkpoint.pt')
    imported = resumed.jobs[0]
    assert imported['key'] == job['key'] and imported['status'] == 'queued'
    assert Path(imported['checkpoint']).is_relative_to(tmp_path / 'resumed')
    assert Path(imported['checkpoint']).read_bytes() == Path(job['checkpoint']).read_bytes()
    assert Path(imported['profile_path']).is_file() and Path(imported['bundle']).is_dir()
    # The first explicit restart may normalize the old running status; import
    # must subsequently leave that old run untouched.
    assert json.loads(old_index)['jobs'][0]['status'] == 'running'
    assert json.loads((evaluator.directory / 'index.json').read_text())['jobs'][0]['status'] == 'queued'


def test_completed_report_tampering_is_rejected_on_resume(evaluator):
    job = enqueue(evaluator, 2)
    complete(evaluator, job)
    Path(job['report']).write_text('{}')
    with pytest.raises(ValueError, match='report changed'):
        PeriodicEvaluator(evaluator.settings, evaluator.run_config, evaluator.output, None)


@pytest.mark.parametrize('keep', [None, 1])
def test_retention_preserves_pending_best_and_finalists(tmp_path, keep):
    evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_opponent=1, keep_snapshots=keep, final_top_k=2),
                                  small_config(), tmp_path / 'run', None)
    jobs = [enqueue(evaluator, value) for value in (1, 2, 3, 4, 5)]
    jobs[3]['status'] = 'running'
    for job in jobs:
        bundle = Path(job['bundle'])
        (bundle / 'actor.pt').write_bytes(b'opaque actor weights')
        (bundle / 'policy').mkdir()
        (bundle / 'policy' / 'runtime.py').write_text('# frozen runtime\n')
        Path(job['worker_log']).write_text('evaluation log\n')
    complete(evaluator, jobs[0], 100)
    complete(evaluator, jobs[1], 90)
    complete(evaluator, jobs[2], 20)
    assert all(Path(job['checkpoint']).exists() and Path(job['bundle']).is_dir() for job in jobs)
    complete(evaluator, jobs[3], 10)
    evaluator._prune_snapshots()  # Repeated cleanup also sees already removed pairs.
    for index, job in enumerate(jobs):
        retained = keep is None or index != 2
        assert Path(job['checkpoint']).exists() == retained
        assert Path(job['bundle']).exists() == retained
        assert job['checkpoint_retained'] == retained
        assert Path(job['profile_path']).is_file() and Path(job['worker_log']).is_file()
        if job['status'] == 'completed':
            assert sha256(Path(job['report']).read_bytes()).hexdigest() == job['report_sha256']


def test_earlier_resume_recovers_pruned_candidate_from_exact_resume_bytes(tmp_path):
    from generals_arena.policies.artifacts import validate_bundle
    from generals_rl.export import prepare_policy_source
    from rl_policy_test_utils import make_checkpoint

    root = tmp_path / 'run'
    root.mkdir()
    resume, config = make_checkpoint(root, filename='model_2.pt')
    source = prepare_policy_source(config, root)
    saved = torch.load(resume, weights_only=True)
    saved['infos'] = {**source, 'completed_ppo_updates': 2}
    torch.save(saved, resume)
    settings = TrainingEvalConfig(every_n_updates=2, maps_per_opponent=1, keep_snapshots=1, final_top_k=0)
    evaluator = PeriodicEvaluator(settings, config, root, None, source=source)
    with patch.object(evaluator, 'poll'):
        evaluator.on_checkpoint(resume, completed_updates=2, steps_per_environment=8, environment_steps=32)
    first = evaluator.jobs[0]
    original_manifest = validate_bundle(first['bundle'])['manifest']
    for job, elo in ((first, 100), (enqueue(evaluator, 4), 200), (enqueue(evaluator, 6), 300)):
        complete(evaluator, job, elo)
    assert not Path(first['checkpoint']).exists()
    assert not Path(first['bundle']).exists()
    resumed = PeriodicEvaluator(settings, config, tmp_path / 'branch', None,
                                resume=resume, resume_updates=2)
    best = json.loads((resumed.output / 'best_model.json').read_text())
    assert best['completed_ppo_updates'] == 2
    assert Path(best['checkpoint']).read_bytes() == resume.read_bytes()
    restored = validate_bundle(best['bundle'])
    assert restored['policy_id'] == first['policy_id']
    assert restored['manifest'] == original_manifest
    assert resumed.jobs[0]['checkpoint_retained']
    assert not Path(first['checkpoint']).exists()  # Do not alter the source run.
    assert not Path(first['bundle']).exists()


def test_unavailable_candidates_remain_in_curves_but_cannot_be_best_or_finalists(evaluator, tmp_path):
    first = enqueue(evaluator, 2)
    complete(evaluator, first, 200)
    Path(first['checkpoint']).unlink()
    # An unrelated resume file must never be substituted for this candidate.
    resume = evaluator.output / 'model_2.pt'
    resume.write_bytes(b'different checkpoint')
    restored = PeriodicEvaluator(evaluator.settings, evaluator.run_config, evaluator.output, None,
                                  resume=resume, resume_updates=2)
    assert len(restored.jobs) == 1 and restored.latest_completed['elo'] == 200
    assert restored.best_completed is None and not (restored.output / 'best_model.json').exists()
    complete(restored, enqueue(restored, 4), 100)
    assert restored.best_completed['completed_ppo_updates'] == 4
    assert [job['completed_ppo_updates'] for job in restored._candidates()] == [4]


def test_training_profile_freezes_paths_relative_to_profile_file(tmp_path, monkeypatch):
    from generals_arena.competitions.evaluation import make_profile, profile_id
    root = tmp_path / 'profiles'
    root.mkdir()
    (root / 'weights.dat').write_bytes(b'opaque weights')
    profile = make_profile({'width': 9, 'height': 9, 'players': 2}, opponents=[{
        'name': 'custom', 'kind': 'custom', 'version': 'test', 'factory': 'example:create_controller',
        'artifacts': {'weights': 'weights.dat'}}], max_ticks=2, maps_per_opponent=1)
    path = root / 'profile.json'
    path.write_text(json.dumps(profile))
    monkeypatch.chdir(tmp_path)
    evaluator = PeriodicEvaluator(TrainingEvalConfig(profile=str(path)), small_config(), tmp_path / 'run', None)
    assert evaluator.profile_id == profile_id(path)
    job = enqueue(evaluator, 2)
    assert profile_id(job['profile_path']) == evaluator.profile_id


def confirm_synthetic(evaluator, candidates, score):
    job = evaluator.confirmation
    Path(job['report']).parent.mkdir(parents=True, exist_ok=True)
    Path(job['report']).write_text('{}')
    job.update(status='completed', report_sha256=sha256(b'{}').hexdigest(), ratings=[
        dict(policy_id=candidate['policy_id'], elo=score(candidate), elo_ci=None,
             games=(len(job['profile']['opponents'])+len(candidates)-1)*4)
        for candidate in candidates])


def test_top_k_confirmation_is_one_joint_job_and_changes_best(tmp_path):
    evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_opponent=1, keep_snapshots=1, final_top_k=2),
                                  small_config(), tmp_path / 'run', None)
    for value, elo in ((2, 100), (4, 200), (6, 50)):
        complete(evaluator, enqueue(evaluator, value), elo)
    def confirm(candidates):
        confirm_synthetic(evaluator, candidates, lambda job: 300 if job['completed_ppo_updates']==2 else 100)
    with patch.object(evaluator, 'drain'), patch.object(evaluator, '_run_confirmation', confirm):
        evaluator.finish()
    final = evaluator.confirmation
    assert set(final['policies']) == {f'{2:064x}', f'{4:064x}'}
    assert final['profile_id'] != evaluator.profile_id
    assert final['expected_games'] == 44  # 2*5 reference pairs + one finalist pair, 2 maps*2 seats.
    assert evaluator.best_completed['completed_ppo_updates'] == 2
    assert evaluator.best_completed['stage'] == 'confirmation'
    assert evaluator.best_completed['report'] == final['report']
    assert json.loads((evaluator.output / 'best_model.json').read_text())['confirmation_complete']
    with patch.object(evaluator, 'drain'), patch.object(evaluator, '_run_confirmation') as repeat:
        evaluator.finish()
    repeat.assert_not_called()
    assert len(evaluator.jobs) == 3  # Periodic jobs; confirmation is a single shared report.
    restored = PeriodicEvaluator(evaluator.settings, small_config(), evaluator.output, None)
    assert restored.best_completed == evaluator.best_completed


@pytest.mark.parametrize('dedicated', [False, True])
def test_gpu_worker_routing_without_cuda_context(tmp_path, dedicated):
    settings = TrainingEvalConfig(device='cuda:0', cuda_visible_devices='7' if dedicated else None,
                                   maps_per_opponent=1, final_top_k=0)
    evaluator = PeriodicEvaluator(settings, small_config(), tmp_path / 'run', None)
    job = enqueue(evaluator, 2)
    process = SimpleNamespace(pid=123, poll=lambda: None)
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1', 'PYTHONPATH': 'preserved'}), \
         patch('generals_rl.training_evaluation.subprocess.Popen', return_value=process) as popen, \
         patch('generals_rl.training_evaluation.torch.cuda.is_initialized', return_value=True), \
         patch('generals_rl.training_evaluation.torch.cuda.synchronize') as sync, \
         patch('generals_rl.training_evaluation.torch.cuda.empty_cache') as empty:
        evaluator.poll()
    command = popen.call_args.args[0]
    assert command[command.index('-m') + 1:command.index('-m') + 3] == ['generals_arena', 'evaluate']
    assert command[command.index('--bundle') + 1] == job['bundle']
    assert popen.call_args.kwargs['start_new_session']
    env = popen.call_args.kwargs['env']
    assert env['CUDA_VISIBLE_DEVICES'] == ('7' if dedicated else '0,1')
    assert env['PYTHONPATH'] == 'preserved' and env['OPENBLAS_NUM_THREADS'] == '1'
    assert sync.call_count == empty.call_count == int(not dedicated)
    evaluator.running = None  # This is a routing stub, never a live process.


def test_real_training_evaluates_periodic_and_final_and_preserves_rng_and_optimizers(tmp_path):
    config = small_config()
    with redirect_stdout(io.StringIO()):
        summary = train(config, tmp_path / 'enabled', iterations=3)
        actual_rng = torch.get_rng_state().clone()
        config['evaluation']['enabled'] = False
        train(config, tmp_path / 'disabled', iterations=3)
    torch.testing.assert_close(torch.get_rng_state(), actual_rng, rtol=0, atol=0)
    evidence = summary['evaluation']
    assert (evidence['scheduled'], evidence['completed'], evidence['failed']) == (2, 2, 0)
    jobs = json.loads((tmp_path / 'enabled/evaluations/index.json').read_text())['jobs']
    assert [job['completed_ppo_updates'] for job in jobs] == [2, 3]
    assert jobs[-1]['final'] is True
    for job in jobs:
        saved = torch.load(job['checkpoint'], weights_only=True, map_location='cpu')
        assert saved['iter'] + 1 == job['completed_ppo_updates']
        assert saved['infos']['policy_spec'] and saved['infos']['source_id']
        assert saved['optimizer_state_dict']['state']
    actual = torch.load(tmp_path / 'enabled/checkpoint.pt', weights_only=True, map_location='cpu')
    expected = torch.load(tmp_path / 'disabled/checkpoint.pt', weights_only=True, map_location='cpu')
    for key in ('model_state_dict', 'optimizer_state_dict', 'optimizer_stu_enc_state_dict'):
        assert_nested_equal(unittest.TestCase(), actual[key], expected[key])
    events = EventAccumulator(str(tmp_path / 'enabled')).Reload()
    assert [point.step for point in events.Scalars('Arena/elo')] == [2, 3]
    best = json.loads((tmp_path / 'enabled/best_model.json').read_text())
    assert Path(best['bundle']).is_dir() and Path(best['checkpoint']).is_file()


def test_resume_into_new_output_recovers_pending_job_and_keeps_optimizer(tmp_path):
    config = small_config()
    config['evaluation']['enabled'] = False
    with redirect_stdout(io.StringIO()):
        train(config, tmp_path / 'first', iterations=2)
    checkpoint = tmp_path / 'first/checkpoint.pt'
    config['evaluation']['enabled'] = True
    settings = TrainingEvalConfig(**config['evaluation'])
    evaluator = PeriodicEvaluator(settings, config, tmp_path / 'first', None)
    with patch.object(evaluator, 'poll'):
        evaluator.on_checkpoint(checkpoint, completed_updates=2, steps_per_environment=8, environment_steps=32)
    evaluator.close()
    old_index = (evaluator.directory / 'index.json').read_bytes()
    relocated = tmp_path / 'relocated'
    (tmp_path / 'first').rename(relocated)
    checkpoint = relocated / 'checkpoint.pt'
    with redirect_stdout(io.StringIO()):
        summary = train(config, tmp_path / 'resumed', iterations=1, resume=str(checkpoint))
    assert summary['evaluation']['completed'] == 2 and summary['evaluation']['failed'] == 0
    jobs = json.loads((tmp_path / 'resumed/evaluations/index.json').read_text())['jobs']
    assert [job['completed_ppo_updates'] for job in jobs] == [2, 3]
    assert all(Path(job['checkpoint']).is_relative_to(tmp_path / 'resumed') for job in jobs)
    assert (relocated / 'evaluations/index.json').read_bytes() == old_index
    saved = torch.load(tmp_path / 'resumed/checkpoint.pt', weights_only=True, map_location='cpu')
    assert saved['iter'] == 2
    assert {int(value['step']) for value in saved['optimizer_state_dict']['state'].values()} == {6}


def test_resume_cannot_change_feature_contract_under_the_old_policy_spec(tmp_path):
    config = small_config()
    config['evaluation']['enabled'] = False
    with redirect_stdout(io.StringIO()):
        train(config, tmp_path / 'first', iterations=1)
    config['task']['combat_features'] = not config['task']['combat_features']
    with pytest.raises(ValueError, match='spec|Spec|config|warm.start'):
        train(config, tmp_path / 'incompatible', iterations=1, resume=str(tmp_path / 'first/checkpoint.pt'))


def test_continued_training_rebuilds_confirmation_finalists(tmp_path):
    evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_opponent=1, final_top_k=1),
                                  small_config(), tmp_path / 'run', None)
    complete(evaluator, enqueue(evaluator, 2), 100)
    def confirm(candidates):
        confirm_synthetic(evaluator, candidates, lambda job: job['completed_ppo_updates']*100)
    with patch.object(evaluator, 'drain'), patch.object(evaluator, '_run_confirmation', confirm):
        evaluator.finish()
    assert evaluator.confirmation['policies'] == [f'{2:064x}']
    complete(evaluator, enqueue(evaluator, 4), 300)
    assert evaluator.confirmation is None
    with patch.object(evaluator, 'drain'), patch.object(evaluator, '_run_confirmation', confirm):
        evaluator.finish()
    assert evaluator.confirmation['policies'] == [f'{4:064x}']
    assert evaluator.best_completed['completed_ppo_updates'] == 4


@pytest.mark.parametrize('changes', [{'batch_size': 64}, {'cpu_threads': 2}, {'final_top_k': 3}])
def test_resume_pins_deployment_and_confirmation_budget(evaluator, changes):
    settings = asdict(evaluator.settings)
    settings.update(changes)
    with pytest.raises(ValueError, match='deployment|selection budget'):
        PeriodicEvaluator(TrainingEvalConfig(**settings), evaluator.run_config, evaluator.output, None)


def test_resume_from_earlier_checkpoint_excludes_original_future_models(evaluator, tmp_path):
    complete(evaluator, enqueue(evaluator, 2), 100)
    complete(evaluator, enqueue(evaluator, 4), 200)
    enqueue(evaluator, 6)
    resumed = PeriodicEvaluator(evaluator.settings, evaluator.run_config, tmp_path / 'branch', None,
                                 resume=evaluator.output / 'checkpoint.pt', resume_updates=2)
    assert [job['completed_ppo_updates'] for job in resumed.jobs] == [2]
    assert resumed.best_completed['completed_ppo_updates'] == 2
    assert not resumed.pending


def test_confirmation_recovers_one_joint_job_after_interruption(tmp_path):
    settings = TrainingEvalConfig(maps_per_opponent=1, final_top_k=2)
    evaluator = PeriodicEvaluator(settings, small_config(), tmp_path / 'run', None)
    for update in (2, 4):
        complete(evaluator, enqueue(evaluator, update), update * 100)
    with patch.object(evaluator, 'drain'), patch.object(evaluator, '_run_confirmation', side_effect=RuntimeError('interrupted')):
        with pytest.raises(RuntimeError, match='interrupted'):
            evaluator.finish()
    evaluator.close()
    resumed = PeriodicEvaluator(settings, small_config(), evaluator.output, None)
    def confirm(candidates):
        confirm_synthetic(resumed, candidates, lambda job: job['completed_ppo_updates']*100)
    with patch.object(resumed, 'drain'), patch.object(resumed, '_run_confirmation', confirm):
        resumed.finish()
    assert resumed.confirmation['status'] == 'completed'
    assert len(resumed.confirmation['policies']) == 2
    assert resumed.best_completed['stage'] == 'confirmation'


def test_real_cpu_training_runs_joint_confirmation_and_resumes(tmp_path):
    config = small_config()
    config['evaluation'].update(every_n_updates=1, final_top_k=2)
    output = tmp_path/'joint_training'
    with redirect_stdout(io.StringIO()):
        train(config, iterations=2, output_dir=output)
    index = json.loads((output/'evaluations/index.json').read_text())
    report = json.loads(Path(index['confirmation']['report']).read_text())
    assert index['confirmation']['status'] == 'completed'
    assert len(report['candidates']) == 2 and report['games'] == 44
    a, b = [row['entrant_id'] for row in report['candidates']]
    assert report['win_matrix'][a][b]['games'] == 4
    assert json.loads((output/'best_model.json').read_text())['confirmation_complete']
    settings = TrainingEvalConfig(**config['evaluation'])
    restored = PeriodicEvaluator(settings, config, output, None)
    assert restored.confirmation['status'] == 'completed'
    restored.close()


def test_worker_group_cleanup_kills_surviving_child(tmp_path):
    ready = tmp_path / 'child_pid'
    child = ("import os,signal,sys,time;from pathlib import Path;"
             "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
             "Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)")
    parent = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);time.sleep(60)"
    process = subprocess.Popen([sys.executable, '-c', parent, child, str(ready)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        child_pid = int(ready.read_text())
        _stop_worker(process)
        assert process.poll() is not None
        stat = Path(f'/proc/{child_pid}/stat')
        deadline = time.monotonic() + 2
        while stat.exists() and stat.read_text().rsplit(') ', 1)[1].split()[0] not in ('Z', 'X'):
            assert time.monotonic() < deadline
            time.sleep(.01)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
