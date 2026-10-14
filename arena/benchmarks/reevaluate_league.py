"""Replace official match/rating data without a backup, then evaluate all bots.

Stop the web service before --reset. Policy artifacts, private trials and human
accounts are retained. Omit --reset to resume the new evaluation after a stop.
"""
import argparse
import json
import sqlite3
from pathlib import Path
import time
import uuid

from generals_arena.league.service import process_lock, run_jobs
from generals_arena.league.store import LeagueStore
from generals_arena.research import summarize, worker_count, refresh_publications, publication_ratings
from generals_arena.league.modes import split_budget
from generals_arena.util import canonical, integer, write_json


def reset(store, games):
    integer('games', games, 1)
    store.verify_execution()
    with process_lock(store, '.server.lock'), process_lock(store, '.worker.lock'), process_lock(store, '.registry.lock'):
        entrants = store.entrants()
        wait = next(e['entrant_id'] for e in entrants.values() if e['name'] == 'WAIT' and e['kind'] == 'cpp')
        jobs = {job['entrant_id']:job['id'] for job in reversed(store.jobs())}
        # All resets are one transaction; there is deliberately no data backup.
        with store.connect(write=True) as db:
            db.execute('DELETE FROM games')
            db.execute('DELETE FROM ratings')
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='rating_pairs'").fetchone():
                db.execute('DELETE FROM rating_pairs')
            db.execute('DELETE FROM jobs')
            for key in ('anchor', 'rating_reference'):
                db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, canonical(wait)))
            for i, entry in enumerate(entrants.values()):
                key = entry['entrant_id']
                if entry['kind'] != 'human' and key != wait:
                    job_id = jobs.get(key, uuid.uuid4().hex)
                    db.execute('INSERT INTO jobs(id,entrant_id,budget,created) VALUES (?,?,?,?)',
                               (job_id, key, games, time.time()+i/1000))
                    budgets = split_budget(games, store.config.player_counts) if games >= 2 and 2 in store.config.player_counts and len(store.config.player_counts) > 1 else {store.default_mode():games}
                    db.executemany('INSERT INTO job_modes(job_id,mode,budget) VALUES (?,?,?)',
                                   [(job_id, mode, amount) for mode, amount in budgets.items()])
        store.anchor = wait
        for suffix in ('', '-wal', '-shm'):
            (store.directory / ('statistics.sqlite3'+suffix)).unlink(missing_ok=True)
        for report in (store.directory/'commits').glob('*/publication.json'):
            report.unlink()
        for name in ('reevaluation.json', 'leaderboard.json', 'schedule_source.json',
                     'run.log', 'verification.json', 'rating_validation.json'):
            (store.directory/name).unlink(missing_ok=True)
        # Reclaim the deleted replay pages rather than retaining old recordings
        # in SQLite's freelist. This runs after the transaction has committed.
        with sqlite3.connect(store.path, isolation_level=None) as db:
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            db.execute('VACUUM')


def evaluate(store, workers):
    start, entries = time.monotonic(), store.entrants()
    for job in store.jobs():
        if job['status'] == 'complete':
            continue
        last = 0.
        def progress(done):
            nonlocal last
            now = time.monotonic()
            if done == 1 or now-last >= 10:
                print(f"{job['name']}: {job['done']+done}/{job['budget']}", flush=True)
                last = now
        run_jobs(store, workers=workers, job_id=job['id'], progress=progress)
        entry = entries[job['entrant_id']]
        report = {**summarize(store, entry['entrant_id'], job['id']), 'name':entry['name'],
                  'version':entry['version'], 'entrant_id':entry['entrant_id'],
                  **publication_ratings(store, entry['entrant_id']),
                  'publication_games':job['budget'], 'workers':workers,
                  'reevaluation':True, 'last_run_seconds':time.monotonic()-start}
        if entry['kind'] == 'cpp':
            directory = store.directory/'commits'/entry['version']
            path = directory/'commit.json'
            if path.exists():
                metadata = json.loads(path.read_text())
                report['preflight_report'] = metadata.get('preflight_report')
                report['report'] = str((directory/'publication.json').resolve())
                write_json(directory/'publication.json', report)
        print(f"{entry['name']}: completed, ratings {report['ratings']}", flush=True)
    boards = store.leaderboards()
    report = {'completed_games':sum(b['completed_games'] for b in boards.values()), 'workers':workers,
              'seconds':time.monotonic()-start, 'jobs':store.jobs(), 'leaderboards':boards}
    write_json(store.directory/'reevaluation.json', report)
    refresh_publications(store)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reset', action='store_true', help='delete official results and ratings, without backup')
    parser.add_argument('--games', type=int, default=1000, help='new target games for every non-WAIT bot')
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    store = LeagueStore(args.directory)
    workers = worker_count(args.workers)
    if args.reset:
        reset(store, args.games)
    evaluate(store, workers)


if __name__ == '__main__':
    main()
