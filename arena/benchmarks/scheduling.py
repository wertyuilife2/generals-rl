"""Replay a recorded workload through the league scheduler, without rating it."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import time

from generals_arena.competitions import jobs
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.store import CompetitionStore
from generals_arena.util import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    plans = fixture['plans']
    with tempfile.TemporaryDirectory(prefix='arena-scheduling-') as temporary:
        store = CompetitionStore.create(temporary, LeagueConfig(max_ticks=fixture['max_ticks']))
        specs = {}
        for entry in fixture['entries'].values():
            if entry['kind'] == 'human':
                continue
            spec = {key: entry[key] for key in ('name', 'kind', 'version')}
            if entry['kind'] == 'baseline':
                spec['policy'] = entry['factory_kwargs']['kind']
            elif entry['kind'] == 'cpp':
                spec['bundle'] = entry['artifact_paths']['bundle']
            elif entry['kind'] == 'custom':
                spec.update(factory=entry['factory'], kwargs=entry['factory_kwargs'])
            else:
                raise ValueError('this workload requires a native or baseline policy')
            assert store.register(spec)['entrant_id'] == entry['entrant_id']
            specs[entry['entrant_id']] = spec
        _, job = store.add(specs[plans[0]['entrant_ids'][0]], games=len(plans))
        # Fixed recorded matches isolate execution changes from adaptive matching.
        def prepared(target, current, *, refill=False, size=32):
            with target.connect() as db:
                pending = [json.loads(r[0]) for r in db.execute(
                    "SELECT plan FROM games WHERE job_id=? AND status='pending' ORDER BY rowid", (current['id'],))]
            if pending and not refill:
                return pending
            start = current['done'] + len(pending)
            group = plans[start:start + max(0, size-len(pending))]
            if group:
                target.insert_plans(group, current['id'])
            return pending + group
        jobs.prepare_batch = prepared
        started = time.perf_counter()
        completed = jobs.run_jobs(store, workers=args.workers)
        elapsed = time.perf_counter() - started
        checks = []
        with store.connect() as db:
            for row in db.execute("SELECT result,replay FROM games WHERE status='complete' ORDER BY sequence"):
                result = json.loads(row[0])
                result.pop('decision_timing', None)
                result.pop('simulation_seconds', None)
                result.pop('sequence', None)
                result.pop('rating_changes', None)
                checks.append({'game_id': result['game_id'], 'result_sha256': digest(result),
                    'replay_sha256': hashlib.sha256(gzip.decompress(row[1])).hexdigest()})
        checks.sort(key=lambda row: row["game_id"])
        report = dict(games=completed, seconds=elapsed, games_per_second=completed/elapsed,
                      workers=args.workers, fixture_sha256=digest(fixture), checks=checks)
        if args.compare:
            reference = json.loads(args.compare.read_text())
            assert reference['fixture_sha256'] == report['fixture_sha256']
            assert reference['checks'] == checks, 'Actions or outcomes changed'
            report.update(identical=True, speedup=reference['seconds']/elapsed)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='checks'}, indent=2))


if __name__ == '__main__':
    main()
