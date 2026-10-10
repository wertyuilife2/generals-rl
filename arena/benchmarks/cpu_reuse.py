"""Measure reuse of the existing C++/OpenMP backend through the shared env API.

python arena/benchmarks/cpu_reuse.py --league outputs/arena/native_1000 \
    --output outputs/arena/performance/cpu-reuse.json

Full games include AI, observations, callbacks and per-tick recording. Step-only
tests replay prepared actions in cloned lanes of one fixed geometry; they exclude
AI, observation construction, recording, state conversion, setup and compilation.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter

import numpy as np

from generals.backends.cpu import CPUEnv
from generals.core.engine import CoreEngine
from generals.core.recording import state_digest, state_from_dict
from generals.core.state import Action, compute_stats
from generals_arena.league.config import LeagueConfig
from generals_arena.league.game import RecordedGame
from generals_arena.league.store import LeagueStore
from generals_baselines import create_controller
from generals.backends.cpu.env import _library


def cases_from_league(directory):
    store = LeagueStore(directory)
    entries = store.entrants()
    with store.connect() as db:
        results = [json.loads(row[0]) for row in db.execute(
            "SELECT result FROM games WHERE status='complete' AND json_extract(plan,'$.human')=0 ORDER BY sequence")]
    cases = []
    for players in (2, 4, 8):
        group = sorted((r for r in results if len(r['entrant_ids']) == players), key=lambda r: r['ticks'])
        result = group[len(group)//2]  # Median-duration game, selected before timing.
        _, blob = store.replay(result['game_id'])
        policies = []
        for key in result['entrant_ids']:
            if entries[key]['kind'] != 'baseline':
                raise ValueError('this benchmark selects baseline-only leagues')
            policies.append(entries[key]['factory_kwargs']['kind'])
        cases.append({'name': result['game_id'], 'plan': result, 'policies': policies,
                      'ticks': store.config.max_ticks,
                      'trace_sha256': hashlib.sha256(gzip.decompress(blob)).hexdigest()})
    return cases


def new_game(case, native=False):
    game = RecordedGame(case['plan'], LeagueConfig(max_ticks=case['ticks']),
                        [create_controller(kind) for kind in case['policies']])
    if not native:
        game.engine.close()
        game.engine = CoreEngine(game.state)
    return game


def full_games(case, repeats):
    samples = {'core': [], 'native_adapter': []}
    trace = None
    # Interleave implementations to reduce thermal/order bias.
    for repeat in range(repeats):
        for native in ((False, True) if repeat % 2 == 0 else (True, False)):
            game = new_game(case, native)
            try:
                start = perf_counter()
                while not game.done:
                    game.advance()
                elapsed = perf_counter()-start
                blob = gzip.decompress(game.replay_blob())
                assert hashlib.sha256(blob).hexdigest() == case['trace_sha256'], 'full action/state trace mismatch'
                samples['native_adapter' if native else 'core'].append(elapsed)
                trace = json.loads(blob)
            finally:
                if native:
                    game.engine.close()
    medians = {key: median(values) for key, values in samples.items()}
    return {'players': len(case['policies']), 'map': case['plan']['map'], 'ticks': len(trace['actions']),
            'game': case.get('name', 'fixture'), 'samples_seconds': samples,
            'median_seconds': medians, 'speedup': medians['core']/medians['native_adapter'],
            'trace_sha256': case['trace_sha256'], 'every_action_and_state_identical': True}, trace


def step_only(trace, batch_sizes, threads, repeats, native_only_sizes=()):
    initial = state_from_dict(trace['initial'])
    actions = [[Action(*values) for values in row] for row in trace['actions']]
    codes = np.asarray([[action.encode() for action in row] for row in actions], dtype=np.int64)
    # Independent reference final statistics, outside all timed regions.
    expected = initial.copy()
    reference = CoreEngine(expected)
    for row in actions:
        reference.step(row)
    assert state_digest(expected) == trace['digests'][-1]
    stats = compute_stats(expected)
    expected_stats = np.stack([stats.army, stats.territory, stats.cities], axis=1)
    reports = []
    for size in [*batch_sizes, *native_only_sizes]:
        native_actions = list(np.repeat(codes[:, None, :], size, axis=1))
        samples = {f'omp_{t}': [] for t in threads if t <= size}
        if size not in native_only_sizes:
            samples['core'] = []
        for repeat in range(repeats):
            order = list(samples) if repeat % 2 else list(reversed(samples))
            for key in order:
                if key == 'core':
                    engines = [CoreEngine(initial.copy()) for _ in range(size)]
                    start = perf_counter()
                    for row in actions:
                        for engine in engines:
                            engine.step(row)
                    elapsed = perf_counter()-start
                    assert all(state_digest(engine.state) == trace['digests'][-1] for engine in engines)
                else:
                    env = CPUEnv.from_states([initial]*size, threads=int(key.split('_')[1]))
                    try:
                        # Warm runtime / thread team; reset before timing the exact trace.
                        env.step(native_actions[0])
                        env.reset()
                        start = perf_counter()
                        for row in native_actions:
                            env.step(row)
                            env.synchronize()
                        elapsed = perf_counter()-start
                        env.check_errors()
                        for name in ('terrain', 'structure', 'owner', 'army', 'alive', 'general_pos'):
                            value = expected.layout.terrain if name == 'terrain' else getattr(expected, name)
                            np.testing.assert_array_equal(env.state[name], np.broadcast_to(value.ravel(), env.state[name].shape))
                        np.testing.assert_array_equal(env.stats, np.broadcast_to(expected_stats, env.stats.shape))
                        assert np.all(env.state['tick'] == expected.tick)
                        assert np.all(env.state['terminated'] == expected.terminated)
                        assert np.all(env.state['winner'] == (-1 if expected.winner_id is None else expected.winner_id))
                    finally:
                        env.close()
                samples[key].append(elapsed)
        medians = {key: median(values) for key, values in samples.items()}
        reports.append({'batch': size, 'ticks_per_lane': len(actions), 'samples_seconds': samples,
                        'median_seconds': medians, 'all_lanes_final_state_and_stats_match': True,
                        'env_ticks_per_second': {key: size*len(actions)/seconds for key, seconds in medians.items()}})
        print(f'step-only batch={size}: {medians}', file=sys.stderr, flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--league', type=Path)
    parser.add_argument('--fixture', type=Path, default=Path(__file__).with_name('eight_player_35.json'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8, 32])
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--native-only-batch-sizes', type=int, nargs='*', default=[],
                        help='additional larger batches, compare OpenMP thread counts without Python timing')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.batch_sizes + args.threads + args.native_only_batch_sizes) < 1:
        parser.error('repeats, batch sizes and threads must be positive')
    if set(args.batch_sizes) & set(args.native_only_batch_sizes):
        parser.error('native-only batch sizes must not duplicate regular batch sizes')
    cases = cases_from_league(args.league) if args.league else [json.loads(args.fixture.read_text())]
    report = {'platform': platform.platform(), 'processor': platform.machine(), 'cpu_count': os.cpu_count(),
              'python': sys.version, 'numpy': np.__version__, 'repeats': args.repeats,
              'openmp_environment': {key: os.environ.get(key) for key in
                  ('OMP_WAIT_POLICY', 'OMP_PROC_BIND', 'OMP_PLACES', 'KMP_BLOCKTIME')},
              'protocol': {'full_games': 'advance includes AI, observations, callbacks and per-tick recording; excludes setup/compression',
                           'step_only': 'prepared actions, identical cloned lanes; includes synchronous step and OMP join; excludes AI/recording/setup',
                           'backend': 'env CPUEnv / make_engine'}}
    _, metadata = _library()
    report['native_build'] = metadata
    report['full_games'] = []
    for case in cases:
        result, trace = full_games(case, args.repeats)
        report['full_games'].append(result)
        print(f'full {result["players"]}p: {result["median_seconds"]}', file=sys.stderr, flush=True)
    report['step_only'] = step_only(trace, args.batch_sizes, args.threads, args.repeats,
                                   args.native_only_batch_sizes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'Saved verified benchmark: {args.output}')


if __name__ == '__main__':
    main()
