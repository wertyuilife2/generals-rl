"""Resumable persistent jobs with ordered commits through the shared runner."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import os
import threading
import time
from types import SimpleNamespace

from generals_arena.util import integer, process_lock
from generals_arena.matches.runner import MatchRunner, resolve_geometries, _batch_key
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
    opponents = options.pop('opponents', None)
    if opponents is not None:
        if not options.get('human') or not isinstance(opponents, list) or not 1 <= len(opponents) <= 7:
            raise ValueError('opponents must contain 1–7 bot IDs or null for random seats')
        if options.get('players') is None:
            options['players'] = len(opponents) + 1
        if len(opponents) != options['players'] - 1:
            raise ValueError('opponent count must equal players minus one')
    retired = store.retired_entrant_ids()
    if job["entrant_id"] in retired:
        raise ValueError("entrant is retired; restore it before scheduling")
    disabled = store.disabled_policy_ids() | retired
    if job['entrant_id'] in disabled:
        raise ValueError('C++ policy is disqualified after a recorded violation')
    if not options.get('human') and options.get('players') is None:
        options['players'] = choose_player_count(store.config, job)
    # WAIT follows the same opponent selection and repeat rules as every bot.
    entries = {key: entry for key, entry in store.entrants().items() if key not in disabled}
    if opponents is not None:
        for key in opponents:
            if key is not None and (not isinstance(key, str) or key not in entries or entries[key]['kind'] == 'human'):
                raise ValueError('opponent must be an active, eligible bot')
    proxy = SimpleNamespace(config=store.config, manifest=store.manifest,
        entrants=lambda: entries, ratings=store.ratings,
        pair_counts=lambda target, mode=None: (coverage[mode] if coverage is not None
                                               else store.pair_counts(target, mode=mode)))
    plans = make_block(proxy, job, **options)
    for plan in plans:
        if opponents is not None:
            seats = [seat for seat, key in enumerate(plan['entrant_ids']) if key != job['entrant_id']]
            for seat, key in zip(seats, opponents):
                if key is not None:
                    plan['entrant_ids'][seat] = key
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


def run_next(store, stop=None, job_id=None, cpp_worker=None, *, max_games=1, progress=None):
    """Caller holds the worker lock. A failed game never becomes a rated loss."""
    for job in store.jobs():
        if job["status"] != "queued" or (job_id is not None and job['id'] != job_id):
            continue
        try:
            plans = prepare_batch(store, job)
            if not plans:
                continue

            committed = 0
            def commit(result, blob):
                nonlocal committed
                if 'policy_error' in result:
                    store.fail_game(result, blob)
                    raise PolicyError(result['policy_error'], details=result.get('error_detail'))
                store.complete_result(result, blob)
                committed += 1
                if progress is not None:
                    progress(committed)

            with MatchRunner(store.entrants(), max_ticks=store.config.max_ticks,
                             execution="isolated", worker=cpp_worker, record_replays=True) as runner:
                results, _ = runner.run(plans[:max_games], on_result=commit, stop=stop)
            return len(results)
        except Exception as exc:
            store.fail(job["id"], exc)
            raise
    return False


def run_jobs(store, *, limit=None, retry=False, progress=None, workers=1, job_id=None):
    store.verify_execution()
    if limit is not None:
        integer("limit", limit, 1)
    workers = worker_count(workers)
    # Compile native libraries once before spawning, avoiding concurrent builds
    # on a cold cache. Workers still load their own library and RNG instances.
    from generals.backends.cpu.env import _library as cpu_library
    cpu_library()
    completed = 0
    with process_lock(store.directory, ".worker.lock"), store.session():
        if retry:
            store.retry()
        if workers > 1:
            return _run_parallel(store, workers, limit, progress, job_id)
        native_worker = MatchWorker()
        try:
            while limit is None or completed < limit:
                count = run_next(store, job_id=job_id, cpp_worker=native_worker,
                    max_games=None if limit is None else limit-completed,
                    progress=(lambda count: progress(completed+count)) if progress is not None else None)
                if not count:
                    break
                completed += count
        finally:
            if native_worker is not None:
                native_worker.close()
    return completed


def _run_parallel(store, workers, limit, progress, job_id=None):
    completed = 0
    # Keep adaptation at the same persistent window as sequential execution.
    # Spawn works on macOS and does not inherit SQLite connections or locks.
    with NativeMatchPool(workers, store.config.max_ticks) as executor:
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
            entries = store.entrants()
            # Small compatible batches amortize model inference and IPC while
            # retaining at least four tasks per worker for dynamic load balance.
            # The window and rating order remain independent of worker count.
            plans = resolve_geometries(plans)
            size = max(1, min(8, len(plans)//(workers*4)))
            buckets = defaultdict(list)
            for index, plan in enumerate(plans):
                buckets[_batch_key(plan)].append(index)
            chunks = [indices[start:start+size] for indices in buckets.values()
                      for start in range(0, len(indices), size)]
            chunks.sort(key=lambda indices: indices[0])
            futures = {executor.submit_native(
                {key: entries[key] for i in indices for key in plans[i]['entrant_ids']},
                [plans[i] for i in indices]): indices
                       for indices in chunks}
            first_error, ready, next_index = None, {}, 0
            for future in as_completed(futures):
                indices = futures[future]
                try:
                    if future.cancelled():
                        ready.update((index, None) for index in indices)
                    else:
                        records = future.result()
                        if len(records) != len(indices):
                            raise RuntimeError('match worker returned an incomplete task')
                        ready.update(zip(indices, records, strict=True))
                        for result, _ in records:
                            if 'policy_error' in result:
                                first_error = first_error or PolicyError(result['policy_error'], details=result.get('error_detail'))
                except Exception as exc:
                    first_error = first_error or exc
                    ready.update((index, None) for index in indices)
                # Completion order never becomes rating order. Commit any
                # contiguous ready prefix while the other processes keep going.
                while next_index in ready:
                    record = ready.pop(next_index)
                    next_index += 1
                    if record is None:
                        continue
                    result, blob = record
                    try:
                        if 'policy_error' in result:
                            store.fail_game(result, blob)
                            first_error = first_error or PolicyError(result['policy_error'], details=result.get('error_detail'))
                        else:
                            store.complete_result(result, blob)
                            completed += 1
                            if progress is not None:
                                progress(completed)
                    except Exception as exc:
                        first_error = first_error or exc
                if first_error is not None:
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

    def __init__(self, workers, max_ticks):
        self.max_ticks = max_ticks
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.local, self.workers = threading.local(), []

    def __enter__(self):
        return self

    def _run(self, entries, plans):
        if not hasattr(self.local, 'worker'):
            self.local.worker = MatchWorker()
            self.workers.append(self.local.worker)
        completed = []
        with MatchRunner(entries, max_ticks=self.max_ticks, execution="isolated",
                         worker=self.local.worker, record_replays=True) as runner:
            runner.run(plans, on_result=lambda result, blob: completed.append((result, blob)))
        return completed

    def submit_native(self, entries, plans):
        return self.executor.submit(self._run, entries, plans)

    def __exit__(self, *exc):
        self.executor.shutdown(wait=True, cancel_futures=True)
        for worker in self.workers:
            worker.close()
