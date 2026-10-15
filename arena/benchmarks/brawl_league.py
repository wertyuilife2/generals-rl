"""Replace old official games with a resumable brawl before one retained commit.

Preparation/simulation writes only NEW games to a staging league. Installation
is atomic and requires the official service to be stopped. No old-data backup
is made. Policy artifacts, registrations, accounts and private runs are kept.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

from generals_arena.cpp_match import NativeMatchPool
from generals_arena.league.modes import MODES, rating_mode
from generals_arena.league.service import prepare_batch, process_lock
from generals_arena.league.store import LeagueStore
from generals_arena.research import refresh_publications, summarize, publication_ratings, worker_count
from generals_arena.util import canonical, digest, integer, write_json


def preserved_digest(db, job):
    value = hashlib.sha256()
    for row in db.execute('SELECT id,plan,result,replay,created,finished FROM games WHERE job_id=? ORDER BY sequence', (job,)):
        result = json.loads(row['result'])
        for key in ('sequence', 'rating_changes'):
            result.pop(key, None)
        value.update(canonical([row['id'], row['plan'], result, row['created'], row['finished']]).encode())
        value.update(row['replay'])
    return value.hexdigest()


def source_revision(db):
    return digest([tuple(row) for row in db.execute('SELECT id,status,sequence FROM games ORDER BY id')])


def prepare(official, name, duels, ffa):
    official.verify_execution()
    for count in (duels, ffa):
        integer('games per mode', count, 1)
    entrants = official.entrants()
    preserved = next((e for e in entrants.values() if e['name'] == name), None)
    if preserved is None:
        raise ValueError('preserved entrant not found')
    jobs = [j for j in official.jobs() if j['entrant_id'] == preserved['entrant_id']]
    if len(jobs) != 1 or jobs[0]['status'] != 'complete' or jobs[0]['done'] != jobs[0]['budget']:
        raise ValueError('preserved commit must have one completed job')
    job = jobs[0]
    excluded = official.retired_entrant_ids() | official.disabled_policy_ids() | {preserved['entrant_id']}
    roster = sorted(k for k,e in entrants.items() if e['kind'] != 'human' and k not in excluded)
    if len(roster) < 3 or official.anchor not in roster:
        raise ValueError('brawl requires at least three active bots including the reference')
    if 2 not in official.config.player_counts or not any(p > 2 for p in official.config.player_counts):
        raise ValueError('both 1v1 and FFA must be enabled')
    with official.connect() as db:
        if db.execute("SELECT 1 FROM games WHERE status IN ('pending','live') LIMIT 1").fetchone():
            raise ValueError('finish active games before preparing a brawl')
        descriptor = dict(preserved_job=job['id'], preserved_entrant=preserved['entrant_id'],
            preserved_digest=preserved_digest(db,job['id']), source_revision=source_revision(db),
            roster=roster, budgets={'1v1':duels,'ffa':ffa}, rules_id=official.manifest['rules_id'])
    identity = digest(descriptor)
    directory = official.directory/'brawls'/identity
    if (directory/'league.sqlite3').exists():
        stage = LeagueStore(directory)
        with stage.connect() as db:
            if json.loads(db.execute("SELECT value FROM meta WHERE key='brawl'").fetchone()[0]) != descriptor:
                raise ValueError('staging league does not match the requested brawl')
        return stage, descriptor
    stage = LeagueStore.create(directory, official.config, reference='random')
    with stage.connect(write=True) as db:
        db.execute('DELETE FROM entrants')
        db.executemany('INSERT INTO entrants VALUES (?,?,?,?)',
            [(key,entrants[key]['name'],entrants[key]['kind'],canonical(entrants[key])) for key in roster])
        for key,value in (('anchor',official.anchor),('rating_reference',official.reference),('brawl',descriptor)):
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,canonical(value)))
        for index,key in enumerate(roster):
            # Spread target quotas evenly; offset remainders between pools.
            budgets = {mode:total//len(roster)+int((index+offset)%len(roster) < total%len(roster))
                       for offset,(mode,total) in enumerate(descriptor['budgets'].items())}
            if not sum(budgets.values()):
                continue
            job_id = digest([identity,key])
            db.execute('INSERT INTO jobs(id,entrant_id,budget,created) VALUES (?,?,?,?)',
                       (job_id,key,sum(budgets.values()),time.time()+index/1000))
            db.executemany('INSERT INTO job_modes(job_id,mode,budget) VALUES (?,?,?)',
                          [(job_id,m,n) for m,n in budgets.items() if n])
    return LeagueStore(directory), descriptor


def evaluate(stage, workers=4, progress=None):
    stage.verify_execution()
    from generals.backends.cpu.env import _library as cpu_library
    from generals_baselines.base import _library as baseline_library
    cpu_library(); baseline_library()
    with process_lock(stage,'.worker.lock'), NativeMatchPool(workers) as pool:
        while True:
            jobs = stage.jobs()
            if any(j['status'] == 'failed' for j in jobs):
                raise ValueError('brawl has a failed policy; inspect the recorded failure')
            pending = [j for j in jobs if j['status'] == 'queued']
            if not pending:
                return
            # Interleave targets by progress; do not finish one bot's entire
            # quota before allowing the others to establish their ratings.
            job = min(pending,key=lambda j:(j['done']/j['budget'],j['id']))
            plans = prepare_batch(stage,job)
            if not plans:
                raise ValueError('unfinished brawl job has no runnable plans')
            futures = [pool.submit_native(str(stage.directory.resolve()),plan) for plan in plans]
            errors = []
            for future in futures:
                try:
                    result,blob = future.result()
                    if 'policy_error' in result:
                        stage.fail_game(result,blob)
                        raise ValueError(result['policy_error'])
                    stage.complete_result(result,blob)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                stage.fail(job['id'],str(errors[0]))
                raise errors[0]
            if progress:
                progress(stage)


def install(official, stage, descriptor):
    official.verify_execution(); stage.verify_execution()
    keep = descriptor['preserved_job']
    with process_lock(official,'.server.lock'), process_lock(official,'.worker.lock'), process_lock(official,'.registry.lock'), process_lock(stage,'.worker.lock'):
        with stage.connect() as db:
            counts = Counter(rating_mode(json.loads(r[0])) for r in db.execute("SELECT result FROM games WHERE status='complete'"))
            if dict(counts) != descriptor['budgets'] or any(j['status'] != 'complete' for j in stage.jobs()):
                raise ValueError('finish the complete brawl before installation')
            for row in db.execute('SELECT plan FROM games'):
                if descriptor['preserved_entrant'] in json.loads(row[0])['entrant_ids']:
                    raise ValueError('excluded entrant appeared in the brawl')
        with official.connect(write=True) as db:
            if source_revision(db) != descriptor['source_revision']:
                raise ValueError('official games changed while the brawl ran; refusing to delete new games')
            if preserved_digest(db,keep) != descriptor['preserved_digest']:
                raise ValueError('preserved commit records changed')
            db.execute('ATTACH DATABASE ? AS brawl',(str(stage.path.resolve()),))
            retained = db.execute('SELECT id,result FROM games WHERE job_id=? ORDER BY sequence',(keep,)).fetchall()
            db.execute('DELETE FROM games WHERE job_id IS NULL OR job_id!=?',(keep,))
            db.execute('DELETE FROM jobs WHERE id!=?',(keep,))
            # Free unique sequence values before assigning the new prefix.
            db.execute('UPDATE games SET sequence=NULL')
            db.execute('INSERT INTO jobs SELECT * FROM brawl.jobs')
            db.execute('INSERT INTO job_modes SELECT * FROM brawl.job_modes')
            db.execute('INSERT INTO games SELECT * FROM brawl.games')
            for index,row in enumerate(retained,sum(counts.values())):
                result = json.loads(row['result']); result['sequence'] = index
                db.execute('UPDATE games SET sequence=?,result=? WHERE id=?',(index,canonical(result),row['id']))
            db.execute('DELETE FROM ratings'); db.execute('DELETE FROM rating_pairs')
            for row in db.execute("SELECT id,result FROM games WHERE status='complete' ORDER BY sequence").fetchall():
                result = json.loads(row['result'])
                official._score(db,result)
                db.execute('UPDATE games SET result=? WHERE id=?',(canonical(result),row['id']))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('rating_revision',?)",(canonical(uuid.uuid4().hex),))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('brawl',?)",(canonical(descriptor),))
            if preserved_digest(db,keep) != descriptor['preserved_digest']:
                raise ValueError('preserved map/actions/result changed during installation')
        for suffix in ('','-wal','-shm'):
            (official.directory/('statistics.sqlite3'+suffix)).unlink(missing_ok=True)
        # Delete obsolete score reports, not source snapshots or private runs.
        keep_version = official.entrants()[descriptor['preserved_entrant']]['version']
        for path in (official.directory/'commits').glob('*/publication.json'):
            if path.parent.name != keep_version:
                path.unlink()
        entries = official.entrants()
        replacements = {j['entrant_id']:j['id'] for j in stage.jobs()}
        for path in (official.directory/'commits').glob('*/commit.json'):
            record = json.loads(path.read_text())
            if record.get('entrant_id') == descriptor['preserved_entrant']:
                continue
            replacement = replacements.get(record.get('entrant_id'))
            if replacement:
                record.update(job_id=replacement,evaluation='brawl')
            else:
                record.pop('job_id',None)
                record['evaluation'] = 'removed'
            write_json(path,record)
        for job in stage.jobs():
            entry = entries[job['entrant_id']]
            path = official.directory/'commits'/entry['version']/'publication.json'
            if path.parent.exists():
                report = {**summarize(official,entry['entrant_id'],job['id']),
                          **publication_ratings(official,entry['entrant_id']), 'name':entry['name'],
                          'entrant_id':entry['entrant_id'], 'version':entry['version'],
                          'evaluation':'brawl', 'publication_games':job['budget'], 'report':str(path.resolve())}
                write_json(path,report)
        for name in ('rating_validation.json','verification.json','schedule_source.json','run.log'):
            (official.directory/name).unlink(missing_ok=True)
        write_json(official.directory/'reevaluation.json',dict(evaluation='brawl',preserved_commit=keep,
            brawl_games=dict(counts),completed_games=sum(counts.values())+len(retained),
            preserved_games=len(retained),preserved_recordings_sha256=descriptor['preserved_digest'],
            jobs=official.jobs(),leaderboards=official.leaderboards()))
        refresh_publications(official)
        with sqlite3.connect(official.path,isolation_level=None) as db:
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.execute('VACUUM')
    # The staging directory contains only the newly generated matches, now in
    # the official league. Remove it rather than retaining a second copy.
    shutil.rmtree(stage.directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--preserve-commit',required=True)
    parser.add_argument('--duels',type=int,default=5000)
    parser.add_argument('--ffa',type=int,default=5000)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--install',action='store_true',help='install completed staging results; stop the service first')
    args = parser.parse_args()
    official = LeagueStore(args.directory)
    stage,descriptor = prepare(official,args.preserve_commit,args.duels,args.ffa)
    if args.install:
        install(official,stage,descriptor)
        print('Installed brawl before preserved commit',flush=True)
        return
    last = 0.
    def progress(store):
        nonlocal last
        now = time.monotonic()
        if now-last >= 10:
            done = Counter()
            for job in store.jobs():
                for mode,value in job['modes'].items(): done[mode] += value['done']
            print('Brawl progress: '+json.dumps(dict(done)),flush=True); last = now
    print('Staging: '+str(stage.directory),flush=True)
    evaluate(stage,worker_count(args.workers),progress)
    print('Brawl complete; stop the service and rerun with --install',flush=True)


if __name__ == '__main__':
    main()
