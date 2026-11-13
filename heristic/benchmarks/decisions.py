"""Compare policy actions and native decision time against a Git revision.

Runs both builds in lockstep on the same public CPU observations. Only the
reference's actions advance the game; any different candidate action fails.
No league data or registered artifacts are changed. Compilation, Python,
observation generation and simulation are outside the native decision timer.
"""
import argparse
import ctypes as C
import gzip
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import tempfile

import numpy as np

from generals import MapConfig, VecEnv
from generals_baselines import create_controller

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / 'arena/src/generals_arena/policies/sdk'
SOURCES = {f'kestrel-v{v}': f'heristic/gpt-6-kestrel-v{v}/policy.cpp' for v in range(1, 6)}
SOURCES.update({'razor-v1': 'heristic/gpt-6-razor/policy.cpp',
                'razor-v2': 'heristic/gpt-6-razor/v2/policy.cpp',
                'razor-v3': 'heristic/gpt-6-razor/v3-release/policy.cpp',
                'razor-v6': 'heristic/gpt-6-razor/v6/policy.cpp',
                'razor-v7': 'heristic/gpt-6-razor/v7/policy.cpp'})


class Observation(C.Structure):
    _fields_ = [(name, C.c_int) for name in ('width', 'height', 'players', 'player')]
    _fields_ += [('tick', C.c_int64), ('general_pos', C.c_int)]
    _fields_ += [(name, C.c_void_p) for name in (
        'terrain', 'structure', 'owner', 'army', 'visible', 'alive',
        'total_army', 'total_land', 'total_cities', 'fog_obstacles', 'has_kill',
        'warn_afk', 'neutralize_in', 'previous')]


def native_observation(raw, previous):
    state = raw['state']
    arrays = {k: np.ascontiguousarray(v) for k, v in state.items()}
    arrays['visible'] = np.ascontiguousarray(raw['visible'])
    for i, name in enumerate(('total_army', 'total_land', 'total_cities')):
        arrays[name] = np.ascontiguousarray(raw['stats'][0, :, i])
    obs = Observation(raw['width'], raw['height'], state['alive'].shape[1],
                      int(raw['player_ids'][0]), int(state['tick'][0]), int(raw['general_pos'][0]))
    for name, _ in Observation._fields_[6:-1]:
        setattr(obs, name, arrays[name].ctypes.data)
    obs.previous = previous.ctypes.data if previous is not None else None
    return obs, arrays


PROFILE = ('update', 'search', 'routePlan', 'weightedCampaign', 'campaignPrograms',
           'deploymentPrograms', 'gatherTree', 'scheduleTree', 'dangerAfter',
           'homeInfrastructure', 'attackPaths', 'preparePlanning')


def build(source, directory, revision=None, profile=False):
    source = Path(source)
    directory.mkdir()
    if revision:
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision,
                                         str(source.parent)], cwd=ROOT, text=True).splitlines()
        files = [Path(n) for n in names if Path(n).parent == source.parent]
        for path in files:
            if path.suffix in ('.cpp', '.h', '.json'):
                (directory/path.name).write_bytes(subprocess.check_output(
                    ['git', 'show', f'{revision}:{path}'], cwd=ROOT))
    else:
        for path in (ROOT/source.parent).iterdir():
            if path.is_file() and path.suffix in ('.cpp', '.h', '.json'):
                (directory/path.name).write_bytes(path.read_bytes())
    if profile:
        for path in directory.iterdir():
            if path.suffix not in ('.cpp', '.h'):
                continue
            code = path.read_text()
            for i, name in enumerate(PROFILE):
                code = re.sub(r'(\b(?:void|double|Candidate) '+name+r'\([^;{}]*\)(?:const)?\s*\{)',
                              rf'\1 ProfileScope profileScope({i});', code)
            path.write_text(code)
    wrapper = directory/'benchmark.cpp'
    wrapper.write_text('''#include <chrono>
#include <cstdint>
static std::uint64_t profileNs[32]{}, profileCalls[32]{};
struct ProfileScope {
 int i; std::chrono::steady_clock::time_point start;
 ProfileScope(int n):i(n),start(std::chrono::steady_clock::now()){}
 ~ProfileScope(){profileNs[i]+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();profileCalls[i]++;}
};
#include "policy.cpp"
extern "C" std::int64_t benchmark_act(void *p,const ArenaObservation *o,std::uint64_t *ns){
 auto start=std::chrono::steady_clock::now();auto action=arena_act(p,o);
 *ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();return action;
}
extern "C" std::uint64_t benchmark_profile(int i){return profileNs[i];}
extern "C" int benchmark_observation_size(){return sizeof(ArenaObservation);}
''')
    library = directory/'policy.so'
    subprocess.run(['c++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared', '-fPIC',
                    '-I', str(SDK), str(wrapper), '-o', str(library)], check=True)
    lib = C.CDLL(str(library))
    lib.arena_create.argtypes = [C.c_int, C.c_uint64, C.c_char_p]
    lib.arena_create.restype = C.c_void_p
    lib.arena_destroy.argtypes = [C.c_void_p]
    lib.benchmark_act.argtypes = [C.c_void_p, C.POINTER(Observation), C.POINTER(C.c_uint64)]
    lib.benchmark_act.restype = C.c_int64
    lib.benchmark_profile.restype = C.c_uint64
    assert lib.benchmark_observation_size() == C.sizeof(Observation)
    params = directory/'params.json'
    return lib, params.read_bytes() if params.exists() else b'{}'


class Pair:
    def __init__(self, builds, seat, seed):
        self.builds = builds
        self.handles = [lib.arena_create(seat, seed, params) for lib, params in builds]
        self.times = [[], []]
        self.actions = hashlib.sha256()

    def act(self, raw, previous=None):
        obs, keepalive = native_observation(raw, previous)
        actions = [None, None]
        # Alternate order to avoid consistently favouring the second build.
        for i in ((0, 1) if obs.tick % 2 else (1, 0)):
            ns = C.c_uint64()
            actions[i] = self.builds[i][0].benchmark_act(self.handles[i], C.byref(obs), C.byref(ns))
            self.times[i].append(ns.value)
        if actions[0] != actions[1]:
            raise AssertionError(f'action mismatch: player={obs.player}, tick={obs.tick}, actions={actions}')
        self.actions.update(int(actions[0]).to_bytes(8, 'little', signed=True))
        return actions[0]

    def close(self):
        for (lib, _), handle in zip(self.builds, self.handles):
            lib.arena_destroy(handle)


def game(builds, side, players, seed, ticks, selfplay, dense=False):
    pairs, opponents, histories = {}, {}, []
    times = [[], []]
    from generals.core.mapgen import generate_map
    initial = generate_map(MapConfig(side, side+2, players), seed)
    if dense:
        rng = np.random.default_rng(seed)
        yy, xx = np.indices(initial.owner.shape)
        gy, gx = np.divmod(initial.general_pos, side)
        distance = np.abs(yy[..., None]-gy)+np.abs(xx[..., None]-gx)
        # Large domains, border fragments, cities and armies immediately
        # exercise late-game planning; the state still satisfies env rules.
        owner = np.argmin(distance+rng.integers(0, 5, distance.shape), axis=-1)
        passable = initial.layout.terrain == 0
        initial.owner[passable] = owner[passable]
        initial.army[passable] = rng.integers(1, 250, int(passable.sum()))
        for p, general in enumerate(initial.general_pos):
            initial.owner.ravel()[general] = p
        initial.tick = 300
        initial.validate()
    with VecEnv.from_states([initial], threads=1, auto_reset=False) as env:
        observers = [env.observer([p]) for p in range(players)]
        for p in range(players):
            if selfplay or p == seed % players:
                pairs[p] = Pair(builds, p, seed*8+p)
            else:
                policy = create_controller(('aggressive', 'expansion', 'defensive')[p % 3])
                policy.reset_batch([p], [seed*8+p])
                opponents[p] = policy
        previous = None
        try:
            for tick in range(ticks):
                actions = np.zeros((1, players), np.int64)
                active = env.state['alive'][0].copy()
                for p in range(players):
                    if not active[p]:
                        continue
                    raw = observers[p]()
                    if p in pairs:
                        actions[0, p] = pairs[p].act(raw, None if previous is None else previous[p])
                    else:
                        actions[0, p] = opponents[p].act_batch(raw)[0]
                result = env.step(actions)
                previous = result.moves[0].copy()
                for p, policy in opponents.items():
                    policy.on_result_batch({'moves': result.moves[:, p].copy(),
                        'actions': actions[:, p].copy(), 'active': active[p:p+1]})
                if env.state['terminated'][0]:
                    break
            for pair in pairs.values():
                for i in range(2):
                    times[i].extend(pair.times[i])
                histories.append(pair.actions.hexdigest())
        finally:
            for pair in pairs.values():
                pair.close()
            for policy in opponents.values():
                policy.close()
    return times, histories


def fixtures(builds):
    """Historical combat observations include dense armies and stale fog memory."""
    count = 0
    for path in sorted((ROOT/'heristic/gpt-6-razor/tests').glob('*.json*')):
        text = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
        value = json.loads(text)
        values = value.get('observations', [value]) if isinstance(value, dict) else value
        if not values or not isinstance(values[0], dict) or 'terrain' not in values[0]:
            continue
        pair = Pair(builds, values[0]['player_id'], 42)
        try:
            for value in values:
                height, width = np.shape(value['terrain'])
                state = {k: np.array(value[k], dtype=dtype).reshape(1, -1) for k, dtype in (
                    ('terrain', np.uint8), ('structure', np.uint8), ('owner', np.int16),
                    ('army', np.int64), ('alive', np.bool_))}
                players = state['alive'].shape[1]
                for name, shape, dtype in (('fog_obstacles', (1,height*width), np.bool_),
                        ('has_kill', (1,players), np.bool_), ('warn_afk', (1,players), np.bool_),
                        ('neutralize_in', (1,players), np.float64)):
                    state[name] = np.asarray(value[name], dtype=dtype).reshape(shape) if name in value else np.zeros(shape, dtype)
                state['tick'] = np.array([value['tick']], np.int64)
                raw = dict(width=width, height=height, state=state,
                    player_ids=np.array([value['player_id']]), general_pos=np.array([value['general_pos']]),
                    visible=np.array(value['visible'], np.bool_).reshape(1,-1),
                    stats=np.array([[value['stats'][k] for k in ('army','territory','cities')]], np.int64).transpose(0,2,1))
                pair.act(raw)
                count += 1
        finally:
            pair.close()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--policy', choices=SOURCES, action='append')
    parser.add_argument('--ticks', type=int, default=500)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--players', type=int, nargs='+', choices=range(2, 9), default=[2, 4, 8])
    parser.add_argument('--dense', action='store_true', help='also compare late-game states')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.ticks < 1 or args.seeds < 1:
        parser.error('--ticks and --seeds must be positive')
    revision = subprocess.check_output(['git', 'rev-parse', args.reference], cwd=ROOT, text=True).strip()
    report = dict(reference=revision, ticks=args.ticks, seeds=args.seeds,
                  players=args.players, dense=args.dense,
                  platform=platform.platform(), machine=platform.machine(),
                  compiler_version=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
                  compiler='c++ -std=c++17 -O3 -ffp-contract=off', policies={})
    with tempfile.TemporaryDirectory(prefix='generals-policy-decisions-') as tmp:
        for name in args.policy or SOURCES:
            builds = [build(SOURCES[name], Path(tmp)/(name+'-before'), args.reference, args.profile),
                      build(SOURCES[name], Path(tmp)/(name+'-after'))]
            times, histories = [[], []], []
            for seed in range(args.seeds):
                for players in args.players:
                    side = 15 if players == 2 else 25 if players <= 5 else 33
                    for selfplay in (False, True):
                        sample, hashes = game(builds, side, players, 4100+seed, args.ticks, selfplay)
                        for i in range(2):
                            times[i].extend(sample[i])
                        histories.extend(hashes)
            dense_calls = 0
            if args.dense:
                for players in args.players:
                    sample, _ = game(builds, 33, players, 8100+players, min(120, args.ticks), True, True)
                    dense_calls += len(sample[0])
            calls = len(times[0])
            before, after = [sum(t)/calls/1000 for t in times]
            row = dict(source=SOURCES[name], games=args.seeds*len(args.players)*2,
                decisions=calls, dense_decisions=dense_calls,
                fixture_decisions=fixtures(builds), actions_identical=True,
                action_sha256=hashlib.sha256(''.join(histories).encode()).hexdigest(),
                before_mean_us=before, after_mean_us=after, speedup=before/after,
                before_p95_us=float(np.percentile(times[0],95)/1000),
                after_p95_us=float(np.percentile(times[1],95)/1000),
                before_p99_us=float(np.percentile(times[0],99)/1000),
                after_p99_us=float(np.percentile(times[1],99)/1000))
            row['candidate_files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((Path(tmp)/(name+'-after')).iterdir())
                if p.suffix in ('.cpp', '.h', '.json') and p.name != 'benchmark.cpp'}
            if args.profile:
                row['reference_inclusive_ms'] = {fn: builds[0][0].benchmark_profile(i)/1e6 for i,fn in enumerate(PROFILE)}
            report['policies'][name] = row
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(name, json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
