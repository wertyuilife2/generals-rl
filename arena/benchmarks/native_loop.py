"""Paired public/native policy bridges in the native Arena loop on identical real policy-driven matches.

Run from the repository root:
  .venv/bin/python arena/benchmarks/native_loop.py --output outputs/native-loop.json

Both executors use the same C++ CPU kernels, policies and independent lane
seeds. One untimed warmup per path excludes compilation. Timed calls include
map generation, binding, simulation, and final result/replay serialization;
they exclude entrant preparation, persistence and worker startup. This is an
Arena match benchmark, not the env step-only throughput benchmark.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
from time import perf_counter

from generals import MapConfig
from generals_arena.matches.batch import execute_batch
from generals_arena.policies.builtin import wait_spec
from generals_arena.policies.entries import prepare_entrant
from generals_arena.util import source_digest


def run_case(lineup, count, args):
    entries, ids = {}, []
    for name in lineup:
        spec = (wait_spec() if name == 'WAIT' else dict(name=name, kind='cpp', bundle=str(args.cpp_bundle))
                if name == 'cpp' else dict(name=name, kind='baseline', policy=name))
        entry = prepare_entrant(spec)
        entries[entry['entrant_id']] = entry
        ids.append(entry['entrant_id'])
    p = len(ids)
    plans = [dict(game_id=str(i), map=asdict(MapConfig(25, 25, p)), map_seed=1000+i,
                  entrant_ids=ids, instance_ids=list(range(p)),
                  controller_seeds=[8000+p*i+seat for seat in range(p)]) for i in range(count)]
    elapsed, phases = {False: [], True: []}, {False: [], True: []}
    caches = {False: {}, True: {}}
    try:
        for trial in range(args.repeat+1):
            outputs = {}
            # Alternate first runner to reduce ordering, cache and clock bias.
            for native in ((False, True) if trial % 2 == 0 else (True, False)):
                start = perf_counter()
                output, timing = execute_batch(entries, plans, max_ticks=args.ticks, native=native,
                    record_replays=args.replays, runtime_cache=caches[native])
                duration = perf_counter()-start
                outputs[native] = output
                if trial:
                    elapsed[native].append(duration)
                    phases[native].append(timing)
            assert [(dict((k,v) for k,v in r.items() if k != 'decision_timing'), b) for r,b in outputs[False]] == [(dict((k,v) for k,v in r.items() if k != 'decision_timing'), b) for r,b in outputs[True]], 'game results or replay bytes differ'
        steps = sum(record['ticks'] for record, _ in outputs[True])
        hashes = [sha256(blob).hexdigest() for _, blob in outputs[True]] if args.replays else None
        medians = {native: statistics.median(values) for native, values in elapsed.items()}
        return dict(lineup=lineup, batch=count, tick_limit=args.ticks, completed_env_ticks=steps,
            identical=True, replay_sha256=hashes, speedup=medians[False]/medians[True],
            **{name: dict(seconds=elapsed[native], median_seconds=medians[native],
                         env_ticks_per_second=steps/medians[native],
                         median_phases={key: statistics.median(row.get(key, 0.) for row in phases[native])
                                        for key in phases[native][0]})
               for native, name in ((False, 'python'), (True, 'native'))})
    finally:
        for cache in caches.values():
            for controller in cache.values():
                close = getattr(controller, 'close', None)
                if close is not None:
                    close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 32])
    parser.add_argument('--ticks', type=int, default=256)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--replays', action='store_true')
    parser.add_argument('--cpp-bundle', type=Path, help='add a mixed workload using this compiled policy bundle')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.batches, args.ticks, args.repeat) < 1:
        parser.error('batch sizes, ticks and repeat must be positive')
    workloads = [['random', 'expansion', 'defensive', 'aggressive'], ['WAIT']*4]
    if args.cpp_bundle:
        workloads.append(['cpp', 'random', 'expansion', 'WAIT'])
    report = dict(platform=platform.platform(), python=platform.python_version(),
                  scope=__doc__, replays=args.replays, cpu_threads=1,
                  cpp_bundle_sha256=sha256(args.cpp_bundle.read_bytes()).hexdigest() if args.cpp_bundle else None,
                  sources={name: source_digest(name) for name in ('generals', 'generals_baselines', 'generals_arena')},
                  cases=[])
    for lineup in workloads:
        for count in args.batches:
            row = run_case(lineup, count, args)
            report['cases'].append(row)
            print(f'{lineup}, B={count}: {row["python"]["median_seconds"]*1000:.2f} -> '
                  f'{row["native"]["median_seconds"]*1000:.2f} ms ({row["speedup"]:.2f}x)', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
