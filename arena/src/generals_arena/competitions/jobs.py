"""Resumable jobs with rolling dispatch and completion-order rating updates."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
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
    if value is None or value == 'auto':
        from generals_arena.matches.resources import cpu_capacity
        available = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
        # Leave headroom for serial policy deadlines and heterogeneous CPU cores.
        return max(1, min(32, cpu_capacity(), available * 3 // 5))
    value = integer('workers', value, 1)
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


def prepare_batch(store, job, *, refill=False, size=32):
    """Keep persisted pending games and optionally top up a rolling queue.

    Scheduled games count towards the budget even while workers execute them.
    New matches use the latest committed ratings and include pending coverage.
    """
    with store.connect() as db:
        query = "SELECT plan FROM games WHERE job_id=? AND status='pending' ORDER BY rowid"
        args = (job['id'], size) if refill else (job['id'],)
        plans = [json.loads(row[0]) for row in db.execute(query + (" LIMIT ?" if refill else ""), args)]
    # Reuse persisted plans; only inspect coverage when new games are needed.
    if (plans and not refill) or len(plans) >= size:
        return plans
    existing = len(plans)
    done = job["done"] + existing
    coverage = {mode: store.pair_counts(job['entrant_id'], mode=mode) for mode in MODES}
    for plan in plans:
        for key in set(plan['entrant_ids']) - {job['entrant_id']}:
            mode = rating_mode(plan)
            coverage[mode][key] = coverage[mode].get(key, 0)+1
    while len(plans) < size and done < job["budget"]:
        group = eligible_schedule(store, {**job, "done": done}, coverage=coverage)
        if not group:
            break
        plans.extend(group)
        done += len(group)
        for plan in group:
            for key in set(plan['entrant_ids']) - {job['entrant_id']}:
                mode = rating_mode(plan)
                coverage[mode][key] = coverage[mode].get(key, 0)+1
    if len(plans) > existing:
        store.insert_plans(plans[existing:], job["id"])
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


def run_jobs(store, *, limit=None, retry=False, progress=None, workers="auto", job_id=None):
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
    completed, first_error = 0, None
    failed_jobs = {}
    # One match per task: native heuristic decisions are serial inside a worker.
    # A small lookahead keeps workers busy without fixing a rating window.
    with NativeMatchPool(workers, store.config.max_ticks) as executor:
        futures = {}
        while True:
            if first_error is None:
                capacity = workers * 2 - len(futures)
                if limit is not None:
                    capacity = min(capacity, limit - completed - len(futures))
                active = {plan['game_id'] for _, plan in futures.values()}
                entries = store.entrants() if capacity > 0 else None
                for job in store.jobs() if capacity > 0 else ():
                    if job['status'] != 'queued' or (job_id is not None and job['id'] != job_id):
                        continue
                    try:
                        plans = prepare_batch(store, job, refill=True,
                                              size=len(futures)+capacity)
                        for plan in plans:
                            if plan['game_id'] in active:
                                continue
                            future = executor.submit_native(
                                {key: entries[key] for key in plan['entrant_ids']}, [plan])
                            futures[future] = job['id'], plan
                            active.add(plan['game_id'])
                            capacity -= 1
                            if not capacity:
                                break
                    except Exception as exc:
                        failed_jobs[job['id']] = exc
                        first_error = exc
                        break
                    if not capacity:
                        break
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                current_job, plan = futures.pop(future)
                if future.cancelled():
                    continue
                try:
                    records = future.result()
                    if len(records) != 1:
                        raise RuntimeError('match worker returned an incomplete task')
                    result, blob = records[0]
                    if 'policy_error' in result:
                        store.fail_game(result, blob)
                        raise PolicyError(result['policy_error'], details=result.get('error_detail'))
                    store.complete_result(result, blob)
                    completed += 1
                    if progress is not None:
                        progress(completed)
                except Exception as exc:
                    failed_jobs[current_job] = exc
                    first_error = first_error or exc
            if first_error is not None:
                # Preserve results from started tasks; leave cancelled plans pending.
                for future in futures:
                    future.cancel()
    if first_error is not None:
        for failed_job, error in failed_jobs.items():
            store.fail(failed_job, error)
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
