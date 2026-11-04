"""Real baseline matches: native execution, action logs, and persistent scheduling.

Run this same file with an earlier revision's env/arena/baselines on PYTHONPATH
to write --output before.json, then run normally with --compare before.json.
Warm compilation is excluded. Native timings include map lookup/generation,
binding, decisions, observations, stepping and result/replay serialization.
Scheduler timings additionally include worker startup, IPC and SQLite commits.
No random-action rollout, replay playback or league setup is timed.
"""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
import tempfile
from time import perf_counter

from generals import Action, MapConfig
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.jobs import run_jobs
from generals_arena.competitions.store import CompetitionStore
from generals_arena.matches.batch import execute_batch
from generals_arena.matches.replay import replay_frames
from generals_arena.policies.entries import prepare_entrant
from generals_arena.util import canonical, source_digest


def check(result, blob):
    state = {key: result[key] for key in ('ticks','winner_seat','reason','elimination_ticks',
        'invalid_actions','final_alive','initial_digest','final_digest','final_stats','kills','death_order','placements')}
    if 'rating_changes' in result:
        state['ratings'] = [result['rating_changes'][key] for key in result['entrant_ids']]
    if blob:
        trace = json.loads(gzip.decompress(blob))
        actions = trace['actions'] if trace.get('action_encoding') == 'generals-v1' else [
            [Action(*action).encode() for action in row] for row in trace['actions']]
        state.update(initial=trace['initial'], actions=actions, frames=replay_frames(blob, result, compact=True))
    return sha256(canonical(state).encode()).hexdigest()


def native_cases(repeat):
    entries = [prepare_entrant(dict(name=name, kind='baseline', policy=name))
               for name in ('random','expansion','defensive','aggressive')]
    table = {entry['entrant_id']:entry for entry in entries}
    ids = list(table)
    cases = []
    for count in (1, 32):
        plans = [dict(game_id=str(i), map=asdict(MapConfig(25,25,4)), map_seed=1000+i,
                      entrant_ids=ids, instance_ids=list(range(4)),
                      controller_seeds=[8000+4*i+seat for seat in range(4)]) for i in range(count)]
        for record in (False, True):
            cache, samples, phases = {}, [], []
            try:
                for trial in range(repeat+1):
                    start = perf_counter()
                    output, timing = execute_batch(table, plans, max_ticks=256, record_replays=record, runtime_cache=cache)
                    elapsed = perf_counter()-start
                    if trial:
                        samples.append(elapsed)
                        phases.append(timing)
                row = dict(name=f'native-b{count}-replay{record}', batch=count, record_replays=record,
                    seconds=samples, median_seconds=median(samples),
                    ticks=sum(result['ticks'] for result,_ in output),
                    replay_bytes=sum(len(blob) for _,blob in output if blob),
                    phases={key:median(item[key] for item in phases) for key in phases[0]},
                    checks=[check(result,blob) for result,blob in output])
                cases.append(row)
                print(row['name'], row['median_seconds'], flush=True)
            finally:
                for controller in cache.values():
                    close = getattr(controller, 'close', None)
                    if close:
                        close()
    return cases


def scheduler_case(repeat):
    samples, expected = [], None
    for trial in range(repeat+1):
        with tempfile.TemporaryDirectory(prefix='arena-pipeline-') as directory:
            store = CompetitionStore.create(directory, LeagueConfig(player_counts=(2,), min_side=18,
                max_side=18, side_cap=18, max_ticks=128), reference='random')
            candidate, job = store.add(dict(name='aggressive',kind='baseline',policy='aggressive'),games=16)
            ids = [candidate['entrant_id'], store.reference]
            plans = []
            for i in range(16):
                seats = [0,1] if i%2 == 0 else [1,0]
                plans.append(dict(game_id=str(i),map=asdict(MapConfig(18,18,2)),map_seed=1000+i//2,
                    entrant_ids=[ids[seat] for seat in seats],instance_ids=seats,
                    controller_seeds=[8000+(i//2)*2+seat for seat in seats],rating_group_size=2))
            store.insert_plans(plans,job)
            start = perf_counter()
            assert run_jobs(store,workers=2)==16
            elapsed = perf_counter()-start
            if trial:
                samples.append(elapsed)
            checks = [check(*store.replay(plan['game_id'])) for plan in plans]
            if expected is not None:
                assert checks == expected, 'scheduler changed the games or rating order'
            expected = checks
    return dict(name='scheduler-16games-2workers',seconds=samples,median_seconds=median(samples),checks=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',type=Path)
    parser.add_argument('--repeat',type=int,default=5)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    report = dict(platform=platform.platform(),python=platform.python_version(),scope=__doc__,
        sources={name:source_digest(name) for name in ('generals','generals_baselines','generals_arena')},
        cases=native_cases(args.repeat)+[scheduler_case(args.repeat)])
    if args.compare:
        previous = json.loads(args.compare.read_text())
        for row, before in zip(report['cases'],previous['cases'],strict=True):
            assert row['name']==before['name'] and row['checks']==before['checks'], 'actions, frames, results or ratings changed'
            row.update(identical=True,speedup=before['median_seconds']/row['median_seconds'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
