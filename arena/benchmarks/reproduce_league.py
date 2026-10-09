"""Re-run an archived baseline schedule into a separate current-code league.

Map, seat, action RNG and completion order are preserved. New source versions
receive new entrant/game identities; original records and accounts are untouched.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

from generals_arena.league.service import run_jobs
from generals_arena.league.store import LeagueStore
from generals_arena.util import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    source = LeagueStore(args.source)
    entries = source.entrants()
    with source.connect() as db:
        records = [dict(row) for row in db.execute(
            "SELECT id,job_id,plan FROM games WHERE status='complete' AND json_extract(plan,'$.human')=0 ORDER BY sequence")]
    plans = [json.loads(r['plan']) for r in records]
    used = {key for plan in plans for key in plan['entrant_ids']}
    if not records or any(entries[key]['kind'] != 'baseline' for key in used):
        raise ValueError('requires completed baseline-only games')
    store = LeagueStore.create(args.destination, source.config)
    ids = {source.anchor: store.anchor}
    budgets = Counter(row['job_id'] for row in records)
    job_ids = {}
    for job in source.jobs():
        if not budgets[job['id']]:
            continue
        old = entries[job['entrant_id']]
        entry, new_job = store.add({'name': old['name'], 'version': old['version'],
            'kind':'baseline', 'policy':old['factory_kwargs']['kind']}, games=budgets[job['id']])
        ids[job['entrant_id']] = entry['entrant_id']
        job_ids[job['id']] = new_job
    mapping = []
    for record, plan in zip(records, plans):
        plan = {**plan, 'entrant_ids': [ids[key] for key in plan['entrant_ids']],
                'rules_id': store.manifest['rules_id'],
                'game_id': digest([store.manifest['rules_id'],record['id']])}
        store.insert_plans([plan], job_ids[record['job_id']])
        mapping.append((record['id'],plan['game_id']))
    write_json(store.directory/'schedule_source.json', {'source': str(args.source.resolve()), 'games':mapping})
    started = perf_counter()
    count = run_jobs(store, workers=args.workers,
        progress=lambda n: print(f'{n}/{len(records)} completed', flush=True) if n % 100 == 0 else None)
    seconds = perf_counter()-started
    assert count == len(records)
    board = store.leaderboard()
    assert store.rebuild() == board
    ticks = 0
    for previous, current in mapping:
        old_result, old_blob = source.replay(previous)
        new_result, new_blob = store.replay(current)
        assert hashlib.sha256(gzip.decompress(old_blob)).digest() == hashlib.sha256(gzip.decompress(new_blob)).digest(), previous
        for field in ('final_stats','elimination_ticks','invalid_actions','winner_seat','ticks'):
            assert old_result[field] == new_result[field], (previous,field)
        ticks += new_result['ticks']
    old_ranking = {row['name']:row for row in source.leaderboard()['ranking'] if row['kind']=='baseline'}
    for row in board['ranking']:
        for key in ('elo','games','rated_games','appearances','wins','losses','draws'):
            assert row[key] == old_ranking[row['name']][key], (row['name'],key)
    with store.connect() as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    report = {'completed_games': count, 'ticks':ticks, 'seconds':seconds,
              'workers': args.workers, 'all_actions_states_and_ratings_identical':True,
              'rules_id':store.manifest['rules_id'], 'schedule':'fixed archived baseline schedule'}
    write_json(store.directory/'leaderboard.json',board)
    write_json(store.directory/'verification.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
