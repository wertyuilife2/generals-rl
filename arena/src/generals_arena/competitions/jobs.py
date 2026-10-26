"""Resumable persistent jobs with ordered commits through the shared runner."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing
import os
import threading
import time
from types import SimpleNamespace

from generals_arena.util import integer, process_lock
from generals_arena.matches.runner import MatchRunner
from generals_arena.matches.worker import MatchWorker
from generals_arena.policies.native import PolicyError
from generals_arena.competitions.matchmaking import choose_player_count, make_block
from generals_arena.competitions.rating import MODES, POLICY, rating_mode
from generals_arena.competitions.store import CompetitionStore


def worker_count(value):
    value = min(4, os.cpu_count() or 1) if value is None else integer('workers', value, 1)
    if value > 32:
        raise ValueError('workers must be <= 32')
    return value


def eligible_schedule(store, job, *, coverage=None, **options):
    disabled = store.disabled_policy_ids()
    if job['entrant_id'] in disabled:
        raise ValueError('C++ policy is disqualified after a recorded violation')
    if not options.get('human') and options.get('players') is None:
        options['players'] = choose_player_count(store.config, job)
    # WAIT follows the same opponent selection and repeat rules as every bot.
    entries = {key: entry for key, entry in store.entrants().items() if key not in disabled}
    proxy = SimpleNamespace(config=store.config, manifest=store.manifest,
        entrants=lambda: entries, ratings=store.ratings,
        pair_counts=lambda target, mode=None: (coverage[mode] if coverage is not None
                                               else store.pair_counts(target, mode=mode)))
    plans = make_block(proxy, job, **options)
    for plan in plans:
        plan['rating_policy'] = POLICY
        plan['rating_mode'] = rating_mode(plan)
        plan['rating_group_size'] = len(plans)
    return plans


def prepare_batch(store, job):
    """Persist at least 32 games before execution, adapting at each window.

    This keeps workers busy across small rotation groups. The same window is
    used with one or many workers, so process count never changes a schedule.
    Already-persisted rotations always take precedence during a resume.
    """
    with store.connect() as db:
        plans = [json.loads(row[0]) for row in db.execute(
            "SELECT plan FROM games WHERE job_id=? AND status='pending' ORDER BY rowid", (job["id"],))]
    # Existing windows are immutable, including partially completed windows.
    if plans:
        return plans
    done = job["done"]
    coverage = {mode: store.pair_counts(job['entrant_id'], mode=mode) for mode in MODES}
    while len(plans) < 32 and done < job["budget"]:
        group = eligible_schedule(store, {**job, "done": done}, coverage=coverage)
        if not group:
            break
        plans.extend(group)
        done += len(group)
        for plan in group:
            for key in set(plan['entrant_ids']) - {job['entrant_id']}:
                mode = rating_mode(plan)
                coverage[mode][key] = coverage[mode].get(key, 0)+1
    if plans:
        store.insert_plans(plans, job["id"])
    return plans


def run_next(store, stop=None, job_id=None, cpp_worker=None):
    """Caller holds the worker lock. A failed game never becomes a rated loss."""
    for job in store.jobs():
        if job["status"] != "queued" or (job_id is not None and job['id'] != job_id):
            continue
        try:
            plans = prepare_batch(store, job)
            if not plans:
                continue

            def commit(result, blob):
                if 'policy_error' in result:
                    store.fail_game(result, blob)
                    raise PolicyError(result['policy_error'], details=result.get('error_detail'))
                store.complete_result(result, blob)

            with MatchRunner(store.entrants(), max_ticks=store.config.max_ticks,
                             execution="isolated", worker=cpp_worker) as runner:
                results, _ = runner.run(plans[:1], on_result=commit, stop=stop)
            return bool(results)
        except Exception as exc:
            store.fail(job["id"], exc)
            raise
    return False


def simulate(directory, plan):
    """Worker processes simulate only. The coordinator commits in schedule order."""
    store = CompetitionStore(directory)
    records = []
    start = time.monotonic()
    with MatchRunner(store.entrants(), max_ticks=store.config.max_ticks,
                     execution="isolated") as runner:
        runner.run([plan], on_result=lambda result, blob: records.append((result, blob)))
    result, blob = records[0]
    if 'policy_error' not in result:
        result = {**result, "simulation_seconds": time.monotonic() - start}
    return result, blob


def run_jobs(store, *, limit=None, retry=False, progress=None, workers=1, job_id=None):
    store.verify_execution()
    if limit is not None:
        integer("limit", limit, 1)
    workers = worker_count(workers)
    has_cpp = any(e['kind'] == 'cpp' for e in store.entrants().values())
    # Compile native libraries once before spawning, avoiding concurrent builds
    # on a cold cache. Workers still load their own library and RNG instances.
    from generals.backends.cpu.env import _library as cpu_library
    cpu_library()
    if any(e['kind'] == 'baseline' for e in store.entrants().values()):
        from generals_baselines.base import _library as baseline_library
        baseline_library()
    completed = 0
    with process_lock(store.directory, ".worker.lock"):
        if retry:
            store.retry()
        if workers > 1:
            return _run_parallel(store, workers, limit, progress, job_id)
        native_worker = MatchWorker() if has_cpp else None
        try:
            while limit is None or completed < limit:
                if not run_next(store, job_id=job_id, cpp_worker=native_worker):
                    break
                completed += 1
                if progress is not None:
                    progress(completed)
        finally:
            if native_worker is not None:
                native_worker.close()
    return completed


def _run_parallel(store, workers, limit, progress, job_id=None):
    completed = 0
    # Keep adaptation at the same persistent window as sequential execution.
    # Spawn works on macOS and does not inherit SQLite connections or locks.
    native = any(e['kind'] == 'cpp' for e in store.entrants().values())
    pool = NativeMatchPool(workers) if native else ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context('spawn'))
    with pool as executor:
        while limit is None or completed < limit:
            job = next((j for j in store.jobs() if j["status"] == "queued"
                        and (job_id is None or j['id'] == job_id)), None)
            if job is None:
                break
            plans = prepare_batch(store, job)
            if not plans:
                break
            if limit is not None:
                plans = plans[:limit - completed]
            directory = str(store.directory.resolve())
            futures = [executor.submit_native(directory, plan) if native else executor.submit(simulate, directory, plan)
                       for plan in plans]
            first_error = None
            for future in futures:
                if future.cancelled():
                    continue
                try:
                    result, blob = future.result()
                    if 'policy_error' in result:
                        store.fail_game(result, blob)
                        raise PolicyError(result['policy_error'], details=result.get('error_detail'))
                    store.complete_result(result, blob)
                    completed += 1
                    if progress is not None:
                        progress(completed)
                except Exception as exc:
                    first_error = first_error or exc
                    for queued in futures:
                        queued.cancel()
                    # Drain started games and preserve their results/violations.
                    # A failed fast worker must not erase another timeout.
            if first_error is not None:
                store.fail(job["id"], first_error)
                raise first_error
    return completed


class NativeMatchPool:
    """Coordinator threads reuse independent native match processes."""

    def __init__(self, workers):
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.local, self.workers = threading.local(), []

    def __enter__(self):
        return self

    def _run(self, directory, plan):
        if not hasattr(self.local, 'worker'):
            self.local.worker = MatchWorker()
            self.workers.append(self.local.worker)
        store = CompetitionStore(directory)
        completed = []
        with MatchRunner(store.entrants(), max_ticks=store.config.max_ticks, execution="isolated",
                         worker=self.local.worker) as runner:
            runner.run([plan], on_result=lambda result, blob: completed.append((result, blob)))
        return completed[0]

    def submit_native(self, directory, plan):
        return self.executor.submit(self._run, directory, plan)

    def __exit__(self, *exc):
        self.executor.shutdown(wait=True, cancel_futures=True)
        for worker in self.workers:
            worker.close()
