"""Arena metadata, verification and warm-worker costs against a real league snapshot.

The source league is opened read-only and copied with SQLite backup into a
temporary directory. Migration/league setup is measured separately. Planning
does not insert games; history results, plans, choices, actions and replay frames
are compared across revisions. Run the same script with an earlier revision's
env/arena/baselines on PYTHONPATH for --output before.json, then --compare it.
This is not the environment step/s benchmark; compilation is excluded.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import sqlite3
from statistics import median
import tempfile
from time import perf_counter

from generals import MapConfig
from generals_arena.competitions.jobs import prepare_batch
from generals_arena.competitions.matchmaking import choose_opponents
from generals_arena.competitions.store import CompetitionStore
from generals_arena.matches.batch_worker import BatchJournal, BatchWorker
from generals_arena.matches.runner import MatchRunner
from generals_arena.policies.entries import prepare_entrant
from generals_arena.policies import native
from generals_arena.util import canonical, source_digest
from execution_pipeline import check


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def measure(name, fn, repeat, *, check_value=True):
    expected = fn()  # Warmup, outside all samples.
    samples = []
    for _ in range(repeat):
        start = perf_counter()
        value = fn()
        samples.append((perf_counter() - start) * 1000)
        if check_value:
            assert value == expected, f'{name} changed between identical requests'
    return dict(name=name, median_ms=median(samples), samples_ms=samples,
                checks=digest(expected) if check_value else None)


class PlanningStore(CompetitionStore):
    def insert_plans(self, plans, job_id=None, status='pending'):
        self.proposed = plans


def database_cases(args):
    with tempfile.TemporaryDirectory(prefix='arena-metadata-') as directory:
        source_path = args.league / 'league.sqlite3'
        source = sqlite3.connect(source_path.resolve().as_uri() + '?mode=ro', uri=True)
        destination = sqlite3.connect(Path(directory) / 'league.sqlite3')
        try:
            source.backup(destination)
            before_pages = destination.execute('PRAGMA page_count').fetchone()[0] - destination.execute('PRAGMA freelist_count').fetchone()[0]
        finally:
            source.close()
            destination.close()
        start = perf_counter()
        store = PlanningStore(directory)
        open_ms = (perf_counter() - start) * 1000
        entries = store.entrants()
        target = next(key for key, entry in entries.items() if entry['name'] == args.entrant or key == args.entrant)
        with store.connect() as db:
            inventory = dict(db.execute('SELECT status,count(*) FROM games GROUP BY status'))
            pages = db.execute('PRAGMA page_count').fetchone()[0] - db.execute('PRAGMA freelist_count').fetchone()[0]
            added_bytes = (pages-before_pages) * db.execute('PRAGMA page_size').fetchone()[0]
        cases = []
        with store.session():
            for mode in ('1v1', 'ffa'):
                cases.append(measure('pair-counts-' + mode, lambda: store.pair_counts(target, mode), args.repeat))
            job = dict(id='metadata-benchmark-only', entrant_id=target, done=1000, budget=1032)
            cases.append(measure('prepare-32-games', lambda: prepare_batch(store, job), args.repeat))
        filters = {'all': {}, 'include': dict(include=[target]), 'exclude': dict(exclude=[target]),
                   'include-ffa': dict(include=[target], mode='ffa')}
        for name, filters in filters.items():
            cases.append(measure('history-' + name,
                lambda: store.matches(limit=20, include_retired=False, **filters), args.repeat))
        # Per-game transaction overhead, including projection triggers on completion.
        for players in (2, 8):
            serial = 0
            def write():
                nonlocal serial
                serial += 1
                key = f'benchmark-{players}-{serial}'
                plan = canonical(dict(game_id=key, entrant_ids=list(entries)[:players]))
                with store.connect(write=True) as db:
                    db.execute("INSERT INTO games(id,status,plan,created) VALUES (?,'pending',?,0)", (key, plan))
                    db.execute("UPDATE games SET status='complete',finished=1 WHERE id=?", (key,))
            with store.session():
                cases.append(measure(f'persist-metadata-{players}players', write, args.repeat))
        return cases, dict(games=inventory, entrants=len(entries), open_and_migrate_ms=open_ms,
                           migration_added_bytes=added_bytes)


def local_cases(repeat):
    cases = []
    for size in (16, 128, 1024):
        candidates = [f'p{i:04}' for i in range(size)]
        ratings = {key: 1000+i for i, key in enumerate(candidates)}
        coverage = {key: i % 17 for i, key in enumerate(candidates)}
        def choose():
            rng = random.Random(7)
            return choose_opponents(candidates[0], candidates, 8, ratings, coverage, rng), rng.getstate()
        cases.append(measure(f'choose-8-from-{size}', choose, repeat))
    legacy_fingerprint = getattr(native, 'execution_fingerprint', None)
    if legacy_fingerprint is not None:
        cases.append(measure('execution-fingerprint', legacy_fingerprint, repeat, check_value=False))
    reads, original = [], Path.read_bytes
    def read(path):
        reads.append(str(path))
        return original(path)
    Path.read_bytes = read
    try:
        if legacy_fingerprint is not None:
            legacy_fingerprint()
    finally:
        Path.read_bytes = original
    worker = BatchWorker()
    def journal():
        if hasattr(worker, '_prepare_journal'):
            worker._prepare_journal(16, 128, 2)
        else:
            fd, path = tempfile.mkstemp(prefix='batch-benchmark-', dir=native._private_cache())
            os.close(fd)
            batch = BatchJournal(path, 16, 128, 2, create=True)
            batch.close()
            Path(path).unlink()
    try:
        cases.append(measure('batch-journal-preparation', journal, repeat))
    finally:
        worker.close()
    return cases, dict(warm_fingerprint_source_reads=len(reads))


def worker_case(repeat):
    entries = [prepare_entrant(dict(name=name, kind='baseline', policy=name)) for name in ('random', 'aggressive')]
    table = {entry['entrant_id']: entry for entry in entries}
    plans = [dict(game_id=str(i), map=asdict(MapConfig(18, 18, 2)), map_seed=1000+i//2,
                  entrant_ids=list(table), instance_ids=[0, 1], controller_seeds=[8000+i*2, 8001+i*2])
             for i in range(16)]
    samples, expected = [], None
    with MatchRunner(table, max_ticks=128, record_replays=True) as runner:
        for trial in range(repeat+1):
            records = []
            start = perf_counter()
            runner.run(plans, on_result=lambda result, blob: records.append((result, blob)))
            elapsed = (perf_counter()-start)*1000
            if trial:
                samples.append(elapsed)
            checks = [check(result, blob) for result, blob in records]
            if expected is not None:
                assert checks == expected
            expected = checks
    return dict(name='warm-runner-16games-replays', median_ms=median(samples), samples_ms=samples, checks=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--league', type=Path, required=True)
    parser.add_argument('--entrant', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--repeat', type=int, default=7)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    database, inventory = database_cases(args)
    local, verification = local_cases(args.repeat)
    report = dict(scope=__doc__, platform=platform.platform(), python=platform.python_version(),
                  sources={name: source_digest(name) for name in ('generals', 'generals_arena', 'generals_baselines')},
                  database=inventory, verification=verification, cases=database+local+[worker_case(args.repeat)])
    if args.compare:
        before = json.loads(args.compare.read_text())
        assert report['database']['games'] == before['database']['games'], 'league snapshot changed'
        for row, old in zip(report['cases'], before['cases'], strict=True):
            assert row['name'] == old['name'] and row['checks'] == old['checks'], row['name']
            row['speedup'] = old['median_ms'] / row['median_ms']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({row['name']: round(row['median_ms'], 4) for row in report['cases']}), flush=True)


if __name__ == '__main__':
    main()
