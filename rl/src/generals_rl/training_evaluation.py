"""Persistent Arena evaluation jobs outside the learner and environment."""
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from time import perf_counter

import torch


def _hash(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _gpu_selectors(value):
    if not isinstance(value, str):
        raise ValueError("CUDA_VISIBLE_DEVICES must be a comma-separated string of GPU ordinals or UUIDs")
    selectors = tuple(part.strip() for part in value.split(","))
    if not all(re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", part) for part in selectors):
        raise ValueError("CUDA_VISIBLE_DEVICES must use GPU ordinals or UUIDs")
    if len({part.isdecimal() for part in selectors}) != 1:
        raise ValueError("CUDA_VISIBLE_DEVICES cannot mix GPU ordinals and UUIDs")
    selectors = tuple(str(int(part)) if part.isdecimal() else "GPU-" + part[4:].lower() for part in selectors)
    for index, part in enumerate(selectors):
        if any(_same_gpu(part, other) for other in selectors[:index]):
            raise ValueError("CUDA_VISIBLE_DEVICES must not select the same GPU twice")
    return selectors


def _same_gpu(first, second):
    if first.isdecimal() and second.isdecimal():
        return int(first) == int(second)
    return first.lower().startswith(second.lower()) or second.lower().startswith(first.lower())


@dataclass(frozen=True)
class TrainingEvalConfig:
    enabled: bool = True
    every_n_updates: int = 14
    profile: str | dict | None = None
    maps_per_opponent: int = 100
    max_ticks: int = 2000
    seed: int = 1000042
    batch_size: int = 128
    cpu_threads: int = 1
    keep_snapshots: int | None = None
    device: str = 'cpu'
    cuda_visible_devices: str | None = None
    allow_shared_gpus: bool = False
    backend: str = 'cpu'
    timeout_seconds: float | None = None
    final_top_k: int = 3
    final_games_multiplier: int = 2

    def __post_init__(self):
        for name in ('enabled', 'allow_shared_gpus'):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f'evaluation.{name} must be boolean')
        for name in ('every_n_updates', 'maps_per_opponent', 'max_ticks', 'seed', 'batch_size',
                     'cpu_threads', 'final_top_k', 'final_games_multiplier'):
            value = getattr(self, name)
            minimum = 0 if name in ('seed', 'batch_size', 'final_top_k') else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'evaluation.{name} must be an integer >= {minimum}')
        if self.keep_snapshots is not None and (type(self.keep_snapshots) is not int or self.keep_snapshots < 1):
            raise ValueError('evaluation.keep_snapshots must be positive or null')
        if self.profile is not None and not isinstance(self.profile, (str, dict)):
            raise ValueError('evaluation.profile must be a JSON object or file path')
        if not isinstance(self.device, str):
            raise TypeError('evaluation.device must be a CPU or CUDA device string')
        try:
            device = torch.device(self.device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError('evaluation.device must be a CPU or CUDA device string') from exc
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('evaluation.device must be a CPU or CUDA device string')
        object.__setattr__(self, 'device', str(device))
        if self.backend not in ('cpu', 'gpu') or (self.backend == 'gpu' and device.type != 'cuda'):
            raise ValueError('evaluation.backend must be cpu or gpu; gpu requires a CUDA device')
        if self.timeout_seconds is not None and (isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError('evaluation.timeout_seconds must be positive and finite or null')
        if self.allow_shared_gpus and self.cuda_visible_devices is None:
            raise ValueError('evaluation.allow_shared_gpus requires explicit cuda_visible_devices')
        if self.cuda_visible_devices is not None:
            selectors = _gpu_selectors(self.cuda_visible_devices)
            if len(selectors) != 1:
                raise ValueError('Arena evaluation currently uses one device; reserve one CUDA GPU and use vector batches')
            object.__setattr__(self, 'cuda_visible_devices', selectors[0])
            if device.type != 'cuda' or device.index not in (None, 0):
                raise ValueError('a dedicated evaluation GPU requires evaluation.device=cuda:0')
            object.__setattr__(self, 'device', 'cuda:0')

    @property
    def synchronous(self):
        return torch.device(self.device).type == 'cuda' and self.cuda_visible_devices is None

    def due(self, completed_updates):
        return self.enabled and completed_updates > 0 and completed_updates % self.every_n_updates == 0

    @property
    def worker_timeout_seconds(self):
        return self.timeout_seconds if self.timeout_seconds is not None else (1800.0 if torch.device(self.device).type == 'cuda' else None)

    def validate_training_devices(self, device):
        if (not self.enabled or self.cuda_visible_devices is None or self.allow_shared_gpus
                or torch.device(device).type != 'cuda'):
            return
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible:
            raise ValueError('dedicated GPU evaluation requires explicit training CUDA_VISIBLE_DEVICES')
        selectors, evaluation = _gpu_selectors(visible), _gpu_selectors(self.cuda_visible_devices)
        if selectors[0].isdecimal() != evaluation[0].isdecimal():
            raise ValueError('training and evaluation CUDA_VISIBLE_DEVICES must both use ordinals or both use UUIDs')
        if any(_same_gpu(first, second) for first in selectors for second in evaluation):
            raise ValueError('the dedicated evaluation GPU must be excluded from training CUDA_VISIBLE_DEVICES')


def _stop_worker(process):
    # The evaluator owns a separate session, including its spawned controllers.
    # Reap the leader and kill surviving children even if the leader exits first.
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class PeriodicEvaluator:
    """One isolated worker; immutable deployments and full resume checkpoints.

    The persistent key is (policy_id, profile_id), never a transient history-pool
    index. CPU and dedicated GPU evaluation run asynchronously. Shared CUDA
    pauses between complete PPO updates. Only the parent writes training logs.
    """

    def __init__(self, settings, run_config, output_dir, writer, *, on_complete=None,
                 resume=None, resume_updates=None, source=None):
        from generals_arena.competitions.evaluation import make_profile
        from generals_arena.competitions.evaluation import profile_id
        self.settings, self.run_config = settings, run_config
        self.output = Path(output_dir).resolve()
        self.directory = self.output / 'evaluations'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.writer, self.on_complete, self.source = writer, on_complete, source
        self.jobs, self.pending, self.running = [], deque(), None
        self.latest_completed = self.best_completed = None
        self.budget_exhausted = False
        self.confirmation = None
        if settings.profile is None:
            task = run_config['task']
            map_config = {key: task[key] for key in ('width', 'height', 'players', 'mountain_density', 'city_density') if key in task}
            if task.get('random_maps', True):
                map_config.update(width=None, height=None)
            self.profile = make_profile(map_config, max_ticks=settings.max_ticks, seed=settings.seed,
                                        maps_per_opponent=settings.maps_per_opponent)
        else:
            from generals_arena.competitions.evaluation import load_profile
            self.profile = load_profile(settings.profile)
        self.profile_id = profile_id(self.profile)
        self._restore(resume, resume_updates)
        self._save_index()

    @property
    def pinned_checkpoints(self):
        return {Path(job['checkpoint']) for job in self.jobs
                if job['status'] in ('queued', 'running') and Path(job['checkpoint']).exists()}

    def _restore(self, resume, resume_updates):
        index = self.directory / 'index.json'
        imported = False
        if not index.exists() and resume is not None:
            for parent in Path(resume).resolve().parents:
                candidate = parent / 'evaluations' / 'index.json'
                if candidate.exists() and candidate != index:
                    index, imported = candidate, True
                    break
        if not index.exists():
            return
        saved = json.loads(index.read_text())
        if saved.get('schema_version') != 1:
            raise ValueError('unsupported training Arena evaluation index; preserve old reports in their original run')
        if saved['profile_id'] != self.profile_id:
            raise ValueError('resumed Arena evaluation profile differs; use the original fixed evaluation profile')
        expected_deployment = self._deployment()
        if saved.get('deployment') != expected_deployment:
            raise ValueError('resumed Arena inference deployment differs from the recorded evaluations')
        if saved.get('selection') != self._selection():
            raise ValueError('resumed Arena final selection budget differs; use the original top-K confirmation settings')
        self.confirmation = saved.get('confirmation')
        recorded_directory = Path(saved.get('directory', index.parent))
        resume_path = Path(resume).resolve() if resume is not None else None
        resume_hash = _hash(resume_path) if resume_path is not None and resume_path.is_file() else None
        for original in saved['jobs']:
            if resume_updates is not None and original['completed_ppo_updates'] > resume_updates:
                continue
            job = dict(original)
            if imported or recorded_directory != index.parent:
                original_checkpoint = None
                for key in ('checkpoint', 'bundle', 'report', 'worker_log', 'profile_path'):
                    path = Path(job[key])
                    relative = path.relative_to(recorded_directory)
                    relocated = index.parent / relative
                    path = relocated if relocated.exists() else path
                    destination = self.directory / relative if imported else relocated
                    if key == 'checkpoint':
                        original_checkpoint = path
                    if imported and path.exists() and not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if path.is_dir():
                            shutil.copytree(path, destination)
                        else:
                            shutil.copyfile(path, destination)
                    job[key] = str(destination)
                checkpoint_config = original_checkpoint.with_name('run_config.json')
                if imported and checkpoint_config.exists():
                    Path(job['checkpoint']).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(checkpoint_config, Path(job['checkpoint']).with_name('run_config.json'))
            checkpoint = Path(job['checkpoint'])
            if not checkpoint.is_file() and resume_hash == job['checkpoint_sha256']:
                # An earlier training checkpoint may outlive its evaluation
                # candidate. Recover exactly those bytes, never another update.
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(resume_path, checkpoint)
                if not checkpoint.with_name('run_config.json').is_file():
                    _write_json(checkpoint.with_name('run_config.json'), self.run_config)
            bundle = Path(job['bundle'])
            if checkpoint.is_file() and not bundle.exists():
                from .export import export_policy
                from generals_arena.policies.artifacts import validate_bundle
                export_policy(checkpoint, bundle, source=(self.source or {}).get('source_path'),
                              completed_updates=job['completed_ppo_updates'])
                if validate_bundle(bundle)['policy_id'] != job['policy_id']:
                    shutil.rmtree(bundle)
                    raise ValueError('restored bundle differs from the evaluated policy')
            job['checkpoint_retained'] = checkpoint.is_file()
            if job['status'] in ('queued', 'running') or (job['status'] == 'cancelled' and job.get('checkpoint_retained', True)):
                if not Path(job['checkpoint']).is_file() or not Path(job['bundle']).is_dir():
                    raise ValueError('queued Arena evaluation is missing its immutable checkpoint or bundle')
                job.update(status='queued', deadline=None, deadline_reason='evaluation_timeout',
                           timeout_seconds=self.settings.worker_timeout_seconds)
                job.pop('pid', None)
                _write_json(job['profile_path'], self.profile)
                self.pending.append(job)
            if job['status'] == 'completed':
                if not Path(job['report']).is_file() or _hash(job['report']) != job['report_sha256']:
                    raise ValueError('completed Arena evaluation report changed')
                self.latest_completed = self._point(job)
            self.jobs.append(job)
        if self.confirmation and not set(self.confirmation['policies']) <= {
                j['policy_id'] for j in self._candidates()}:
            self.confirmation = None
        if self.confirmation:
            self.confirmation['profile'] = {**self.profile, **{key: self.confirmation['profile'][key]
                                                            for key in ('seed', 'maps_per_opponent')}}
            for key in ('report', 'worker_log'):
                relative = Path(self.confirmation[key]).relative_to(recorded_directory)
                source = index.parent / relative
                destination = self.directory / relative
                if imported and source.exists() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                self.confirmation[key] = str(destination)
            if self.confirmation['status'] == 'completed':
                if _hash(self.confirmation['report']) != self.confirmation['report_sha256']:
                    raise ValueError('completed joint confirmation report changed')
            else:
                self.confirmation['status'] = 'queued'
        self._select_best()
        if self.on_complete is not None:
            for job in self.jobs:
                if job['status'] == 'completed' and job['stage'] == 'periodic':
                    self.on_complete(self._point(job))

    def on_checkpoint(self, checkpoint, *, completed_updates, steps_per_environment,
                      environment_steps, final=False, remaining_seconds=None):
        """Archive before enqueueing, so checkpoint retention cannot race a worker."""
        entered = perf_counter()
        if not self.settings.enabled or not (final or self.settings.due(completed_updates)):
            return False
        if type(completed_updates) is not int or completed_updates < 1:
            raise ValueError('completed_updates must be a positive integer')
        timeout = self.settings.worker_timeout_seconds
        deadline_reason = 'evaluation_timeout'
        if not final and self.settings.synchronous and remaining_seconds is not None:
            if (isinstance(remaining_seconds, bool) or not isinstance(remaining_seconds, (int, float))
                    or not math.isfinite(remaining_seconds)):
                raise ValueError('remaining_seconds must be finite or null')
            available = max(0.0, remaining_seconds - 10.0)
            estimate = self._estimated_worker_budget_seconds()
            if estimate is not None and available < estimate:
                self.budget_exhausted = True
                return False
            if timeout is None or available <= timeout:
                timeout, deadline_reason = available, 'wall_budget'
        deadline = entered + timeout if self.settings.synchronous and timeout is not None else None
        from .export import export_policy
        from generals_arena.policies.artifacts import validate_bundle
        checkpoint = Path(checkpoint).resolve()
        checksum = _hash(checkpoint)
        target = self.directory / 'candidates' / f'update_{completed_updates:08d}_{checksum[:16]}'
        frozen = target / 'checkpoint.pt'
        if not frozen.exists():
            target.mkdir(parents=True, exist_ok=True)
            temporary = frozen.with_suffix('.pt.tmp')
            shutil.copyfile(checkpoint, temporary)
            if _hash(temporary) != checksum:
                temporary.unlink()
                raise RuntimeError('checkpoint changed while preparing Arena evaluation')
            temporary.replace(frozen)
            _write_json(target / 'run_config.json', self.run_config)
        elif _hash(frozen) != checksum:
            raise RuntimeError('immutable evaluation checkpoint changed')
        export_started = perf_counter()
        bundle = target / 'bundle'
        if not bundle.exists():
            bundle = export_policy(frozen, bundle, run_config=self.run_config,
                                   source=(self.source or {}).get('source_path'), completed_updates=completed_updates)
        identity = validate_bundle(bundle)
        policy_id = identity['policy_id']
        job = self._enqueue(bundle, frozen, checksum, policy_id, self.profile, completed_updates,
                            steps_per_environment, environment_steps, stage='periodic', final=final,
                            deadline=deadline, timeout_seconds=timeout, deadline_reason=deadline_reason,
                            model_name=identity.get('name'))
        if job is None:
            return False
        job['export_seconds'] = perf_counter() - export_started
        if self.writer is not None:
            self.writer.add_scalar('Arena/export_seconds', job['export_seconds'], completed_updates)
        self._save_index()
        self.poll()
        if self.settings.synchronous:
            self.drain()
        return True

    def _estimated_worker_budget_seconds(self):
        durations = [j['worker_seconds'] for j in self.jobs if j['status'] == 'completed' and j['stage'] == 'periodic'
                     and isinstance(j.get('worker_seconds'), (int, float))
                     and not isinstance(j['worker_seconds'], bool)
                     and math.isfinite(j['worker_seconds']) and j['worker_seconds'] > 0]
        return 1.25 * max(durations[-3:]) + 30 if durations else self.settings.worker_timeout_seconds

    def _enqueue(self, bundle, checkpoint, checksum, policy_id, profile, completed_updates,
                 steps_per_environment, environment_steps, *, stage, final=False,
                 deadline=None, timeout_seconds=None, deadline_reason='evaluation_timeout', model_name=None):
        from generals_arena.competitions.evaluation import expected_games
        from generals_arena.competitions.evaluation import profile_id
        profile_key = profile_id(profile)
        existing = next((j for j in self.jobs if (j['policy_id'], j['profile_id']) == (policy_id, profile_key)), None)
        if existing is not None:
            existing['final'] = existing.get('final', False) or final
            self._save_index()
            return None
        if stage == 'periodic' and self.confirmation is not None:
            # Continuing training creates a new candidate set. Old held-out
            # results remain reusable, but must not freeze the previous top K.
            self.confirmation = None
        key = sha256(f'{policy_id}:{profile_key}'.encode()).hexdigest()
        target = self.directory / 'jobs' / key
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / 'profile.json', profile)
        count = expected_games(profile)
        job = {'key': key, 'policy_id': policy_id, 'profile_id': profile_key, 'model_name': model_name,
               'completed_ppo_updates': completed_updates, 'iteration': completed_updates - 1,
               'steps_per_environment': steps_per_environment, 'environment_steps': environment_steps,
               'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum, 'bundle': str(bundle),
               'report': str(target / 'report.json'), 'worker_log': str(target / 'worker.log'),
               'profile_path': str(target / 'profile.json'), 'expected_games': count,
               'stage': stage, 'final': final, 'status': 'queued', 'device': self.settings.device,
               'backend': self.settings.backend,
               'timeout_seconds': self.settings.worker_timeout_seconds if timeout_seconds is None else timeout_seconds,
               'deadline': deadline, 'deadline_reason': deadline_reason, 'checkpoint_retained': True}
        self.jobs.append(job)
        self.pending.append(job)
        self._save_index()
        print(f'[arena] queued updates={completed_updates} games={count} policy={policy_id[:12]} stage={stage}', flush=True)
        return job

    def poll(self):
        if self.running is not None:
            job, process, started = self.running
            code = process.poll()
            timeout = job['timeout_seconds']
            deadline = job.get('deadline')
            expired = (perf_counter() >= deadline if deadline is not None else
                       timeout is not None and perf_counter() - started >= timeout)
            if expired:
                _stop_worker(process)
                self.running = None
                job['worker_seconds'] = perf_counter() - started
                job['returncode'] = process.returncode
                self._fail(job, TimeoutError(f'Arena worker exceeded {timeout:g} seconds'))
            elif code is not None:
                self.running = None
                job['worker_seconds'], job['returncode'] = perf_counter() - started, code
                try:
                    if code != 0:
                        _stop_worker(process)
                        raise RuntimeError(f'Arena worker exited with code {code}')
                    self._complete(job)
                except Exception as exc:
                    self._fail(job, exc)
        while self.running is None and self.pending:
            job = self.pending.popleft()
            if job.get('deadline') is None and self.settings.synchronous and job['timeout_seconds'] is not None:
                job['deadline'] = perf_counter() + job['timeout_seconds']
            if job.get('deadline') is not None and perf_counter() >= job['deadline']:
                self._cancel(job)
                continue
            command = self._worker_command('--bundle', job['bundle'], job['profile_path'], job['report'])
            environment = self._worker_environment()
            if job.get('deadline') is not None and perf_counter() >= job['deadline']:
                self._cancel(job)
                continue
            try:
                with Path(job['worker_log']).open('w') as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               env=environment, start_new_session=True)
                job.update(status='running', pid=process.pid)
                self.running = job, process, perf_counter()
                self._save_index()
            except OSError as exc:
                self._fail(job, exc)

    def _worker_command(self, source_flag, source, profile, output):
        return [sys.executable, '-u', '-m', 'generals_arena', 'evaluate', source_flag, str(source),
                '--profile', str(profile), '--output', str(output), '--device', self.settings.device,
                '--batch-size', str(self.settings.batch_size), '--backend', self.settings.backend,
                '--cpu-threads', str(self.settings.cpu_threads)]

    def _worker_environment(self):
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        if torch.device(self.settings.device).type == 'cpu':
            environment['CUDA_VISIBLE_DEVICES'] = ''
        elif self.settings.cuda_visible_devices is not None:
            environment['CUDA_VISIBLE_DEVICES'] = self.settings.cuda_visible_devices
        elif torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return environment

    def _complete(self, job):
        from generals_arena.competitions.evaluation import validate_coverage
        report = json.loads(Path(job['report']).read_text())
        if (report.get('status') != 'completed' or report.get('policy_id') != job['policy_id']
                or report.get('profile_id') != job['profile_id'] or report.get('errors') != []
                or type(report.get('games')) is not int or report.get('games') != job['expected_games']
                or report.get('expected_games') != job['expected_games']
                or report.get('device') != job['device'] or report.get('backend') != job['backend']):
            raise RuntimeError('Arena report does not match the immutable policy/profile and complete budget')
        profile = json.loads(Path(job['profile_path']).read_text())
        validate_coverage(report, profile)
        value = report.get('elo')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError('Arena report requires a finite Elo')
        interval = report.get('elo_ci')
        if interval is not None:
            if not isinstance(interval, dict) or any(isinstance(interval.get(k), bool)
                    or not isinstance(interval.get(k), (int, float)) or not math.isfinite(interval[k])
                    for k in ('lower', 'upper')) or not interval['lower'] <= value <= interval['upper']:
                raise RuntimeError('Arena report contains an invalid Elo interval')
        if _hash(job['checkpoint']) != job['checkpoint_sha256']:
            raise RuntimeError('Arena resume checkpoint changed')
        job.update(status='completed', elo=float(value), elo_ci=interval, report_sha256=_hash(job['report']))
        self.latest_completed = self._point(job)
        self._select_best()
        if self.writer is not None:
            prefix = 'Arena' if job['stage'] == 'periodic' else 'ArenaConfirmation'
            step = job['completed_ppo_updates']
            self.writer.add_scalar(f'{prefix}/elo', value, step)
            if interval is not None:
                self.writer.add_scalar(f'{prefix}/elo_lower95', interval['lower'], step)
                self.writer.add_scalar(f'{prefix}/elo_upper95', interval['upper'], step)
            self.writer.add_scalar(f'{prefix}/games', job['expected_games'], step)
            self.writer.add_scalar(f'{prefix}/worker_seconds', job['worker_seconds'], step)
            self.writer.flush()
        if self.on_complete is not None and job['stage'] == 'periodic':
            self.on_complete(self._point(job))
        self._prune_snapshots()
        self._save_index()
        print(f"[arena] updates={job['completed_ppo_updates']} Elo={value:.1f} stage={job['stage']}", flush=True)

    def _point(self, job):
        return {'model_name': job.get('model_name'), **{key: job[key] for key in ('policy_id', 'profile_id', 'completed_ppo_updates', 'iteration',
                'steps_per_environment', 'environment_steps', 'checkpoint', 'checkpoint_sha256', 'bundle',
                'report', 'stage', 'elo', 'elo_ci', 'expected_games')}}

    def _candidates(self):
        """Reports remain historical evidence after resume weights are pruned."""
        return [job for job in self.jobs if job['status'] == 'completed' and job['stage'] == 'periodic'
                and Path(job['checkpoint']).is_file() and Path(job['bundle']).is_dir()]

    def _select_best(self):
        completed = [self._point(job) for job in self._candidates()]
        if self.confirmation and self.confirmation['status'] == 'completed':
            confirmation = self.confirmation
            ratings = {row['policy_id']: row for row in confirmation['ratings']}
            completed = [{**point, 'stage': 'confirmation', 'report': confirmation['report'],
                          'profile_id': confirmation['profile_id'], 'elo': ratings[point['policy_id']]['elo'],
                          'elo_ci': ratings[point['policy_id']]['elo_ci'],
                          'expected_games': ratings[point['policy_id']]['games']}
                         for point in completed if point['policy_id'] in ratings]
        if not completed:
            self.best_completed = None
            (self.directory / 'best_model.json').unlink(missing_ok=True)
            (self.output / 'best_model.json').unlink(missing_ok=True)
            return
        best = max(completed, key=lambda job: (job['elo'], -job['completed_ppo_updates'], job['policy_id']))
        self.best_completed = best
        document = {'schema_version': 1, **self.best_completed, 'selection_metric': 'elo',
                    'confirmation_complete': bool(self.confirmation and best['stage'] == 'confirmation'),
                    'resume_scope': {'checkpoint': 'model_and_optimizers',
                                     'historical_pool': 'only_if_available_beside_checkpoint',
                                     'environment_and_opponent_rng': 'fresh'}}
        _write_json(self.directory / 'best_model.json', document)
        _write_json(self.output / 'best_model.json', document)

    def _fail(self, job, error):
        job.update(status='failed', error=f'{type(error).__name__}: {error}')
        self._prune_snapshots()
        self._save_index()
        print(f"[arena] FAILED updates={job['completed_ppo_updates']}: {job['error']}; log={job['worker_log']}", flush=True)

    def _cancel(self, job):
        job.update(status='cancelled', reason=job['deadline_reason'])
        self._prune_snapshots()
        self._save_index()

    def _prune_snapshots(self):
        """Retain/delete each checkpoint and deployment together; keep reports."""
        keep = self.settings.keep_snapshots
        if keep is None:
            return
        completed = self._candidates()
        finished = [j for j in self.jobs if j['status'] in ('completed', 'failed', 'cancelled')]
        contenders = sorted(completed, key=lambda j: j['elo'], reverse=True)[:max(1, self.settings.final_top_k)]
        protected = {j['checkpoint'] for j in contenders + finished[-keep:]}
        protected.update(str(path) for path in self.pinned_checkpoints)
        if self.best_completed:
            protected.add(self.best_completed['checkpoint'])
        for job in self.jobs:
            if job['status'] in ('completed', 'failed', 'cancelled') and job['checkpoint'] not in protected:
                Path(job['checkpoint']).unlink(missing_ok=True)
                bundle = Path(job['bundle'])
                if bundle.is_dir():
                    shutil.rmtree(bundle)
                job['checkpoint_retained'] = False

    def _save_index(self):
        _write_json(self.directory / 'index.json', {'schema_version': 1, 'profile': self.profile,
                    'directory': str(self.directory),
                    'profile_id': self.profile_id,
                    'deployment': self._deployment(), 'selection': self._selection(),
                    'jobs': self.jobs, 'best': self.best_completed, 'confirmation': self.confirmation})

    def _deployment(self):
        return {key: getattr(self.settings, key) for key in ('device', 'backend', 'batch_size', 'cpu_threads')}

    def _selection(self):
        return {key: getattr(self.settings, key) for key in ('final_top_k', 'final_games_multiplier')}

    def describe(self):
        return {'scheduled': len(self.jobs),
                **{status: sum(j['status'] == status for j in self.jobs)
                   for status in ('queued', 'running', 'completed', 'failed', 'cancelled')},
                'profile_id': self.profile_id, 'latest_completed': self.latest_completed,
                'best_completed': self.best_completed, 'best_model': str(self.output / 'best_model.json') if self.best_completed else None,
                'confirmation': self.confirmation}

    def drain(self):
        self.poll()
        while self.running is not None or self.pending:
            if self.running is not None:
                try:
                    self.running[1].wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            self.poll()

    def finish(self):
        """Drain, then play top K against references and each other in one rating pool."""
        self.drain()
        if self.settings.final_top_k > 0:
            candidates = sorted(self._candidates(),
                                key=lambda j: (-j['elo'], j['completed_ppo_updates']))[:self.settings.final_top_k]
            if candidates:
                if self.confirmation is None:
                    from generals_arena.competitions.evaluation import profile_id, expected_games
                    profile = json.loads(json.dumps(self.profile))
                    profile['seed'] = self.profile['seed'] ^ 0x5A17F03
                    profile['maps_per_opponent'] *= self.settings.final_games_multiplier
                    identity = profile_id(profile)
                    policies = [j['policy_id'] for j in candidates]
                    key = sha256(json.dumps([identity, policies]).encode()).hexdigest()
                    directory = self.directory / 'confirmations' / key
                    self.confirmation = dict(profile=profile, profile_id=identity, policies=policies,
                        status='queued', report=str(directory/'report.json'), worker_log=str(directory/'worker.log'),
                        expected_games=expected_games(profile, len(candidates)))
                    self._save_index()
                else:
                    profile = self.confirmation['profile']
                    candidates = [j for j in self.jobs if j['stage'] == 'periodic'
                                  and j['policy_id'] in self.confirmation['policies']]
                if self.confirmation['status'] != 'completed':
                    self._run_confirmation(candidates)
        self._select_best()
        self._save_index()

    def _run_confirmation(self, candidates):
        from generals_arena.competitions.evaluation import validate_joint_coverage, validate_coverage
        job = self.confirmation
        directory = Path(job['report']).parent
        _write_json(directory/'profile.json', job['profile'])
        _write_json(directory/'candidates.json', [candidate['bundle'] for candidate in candidates])
        command = self._worker_command('--candidates', directory/'candidates.json', directory/'profile.json', job['report'])
        entered, process = perf_counter(), None
        try:
            with Path(job['worker_log']).open('w') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           env=self._worker_environment(), start_new_session=True)
            job.update(status='running', pid=process.pid)
            self._save_index()
            code = process.wait(timeout=self.settings.worker_timeout_seconds)
            if code != 0:
                raise RuntimeError(f'joint Arena confirmation exited with code {code}; log={job["worker_log"]}')
            report = json.loads(Path(job['report']).read_text())
            if (report['status'] != 'completed' or report['errors'] or report['profile_id'] != job['profile_id']
                    or report['device'] != self.settings.device or report['backend'] != self.settings.backend):
                raise RuntimeError('joint Arena confirmation differs from its requested profile or execution')
            if len(candidates) == 1:
                validate_coverage(report, job['profile'])
            else:
                validate_joint_coverage(report, job['profile'], job['policies'])
            if {row['policy_id'] for row in report['candidates']} != set(job['policies']):
                raise RuntimeError('joint Arena confirmation contains different candidates')
            for candidate in candidates:
                if _hash(candidate['checkpoint']) != candidate['checkpoint_sha256']:
                    raise RuntimeError('confirmation checkpoint changed')
            ratings = [{key: row[key] for key in ('policy_id', 'elo', 'elo_ci', 'games')} for row in report['candidates']]
            if any(isinstance(row['elo'], bool) or not math.isfinite(row['elo']) for row in ratings):
                raise RuntimeError('joint confirmation requires finite ratings')
            job.update(status='completed', ratings=ratings, report_sha256=_hash(job['report']))
            if self.writer is not None:
                updates = {candidate['policy_id']: candidate['completed_ppo_updates'] for candidate in candidates}
                for row in ratings:
                    self.writer.add_scalar('ArenaConfirmation/elo', row['elo'], updates[row['policy_id']])
                self.writer.flush()
        except BaseException as error:
            job.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if process is not None:
                _stop_worker(process)
            job['worker_seconds'] = perf_counter() - entered
            self._save_index()

    def close(self):
        if self.running is not None:
            job, process, _ = self.running
            _stop_worker(process)
            self.running = None
            job['status'] = 'cancelled'
        while self.pending:
            self.pending.popleft()['status'] = 'cancelled'
        self._save_index()
