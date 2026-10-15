"""Explicit, replay-verified adoption of the October 2026 runtime fast paths.

This does not waive source checks. It records one exact runtime transition and
exact baseline package aliases after reproducing historical black-box actions.
Run with --apply only while the league server and workers are stopped.
"""
import argparse
from collections import Counter
from dataclasses import fields
from importlib.util import find_spec
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np

from .util import canonical, digest, file_digest, source_digest, write_json


ENV_FAST_PATHS = {'backends/_build.py', 'backends/cpu/engine.py',
                  'core/recording.py', 'core/state.py', 'protocol/actions.py'}
ARENA_FAST_PATHS = {'game.py', 'observation.py'}
SUFFIXES = {'.py', '.cpp', '.h', '.hpp', '.cu'}


def source_files(root):
    return {str(p.relative_to(root)): file_digest(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and p.suffix in SUFFIXES}


def installed_root(package):
    return Path(next(iter(find_spec(package).submodule_search_locations)))


def check_transition(previous, current, reference):
    """Protect the complete ruleset, not just the two changed Python wrappers."""
    reference = Path(reference)
    for key in previous.keys() | current.keys():
        if key not in {'engine_sha256', 'implementation', 'rules_id'}:
            if canonical(previous.get(key)) != canonical(current.get(key)):
                raise ValueError(f'runtime upgrade cannot change {key}')
    old_env, new_env = source_files(reference/'generals'), source_files(installed_root('generals'))
    if digest(old_env) != previous['engine_sha256']:
        raise ValueError('reference environment does not match the registered league')
    changed_env = {p for p in old_env.keys() | new_env.keys() if old_env.get(p) != new_env.get(p)}
    if changed_env - ENV_FAST_PATHS:
        raise ValueError(f'unreviewed environment changes: {sorted(changed_env - ENV_FAST_PATHS)}')
    if previous['implementation'].keys() != current['implementation'].keys():
        raise ValueError('runtime upgrade cannot change the manifest file set')
    changed_arena = []
    for name, expected in previous['implementation'].items():
        if file_digest(reference/'generals_arena'/'league'/name) != expected:
            raise ValueError(f'reference Arena does not match the registered league: {name}')
        if current['implementation'][name] != expected:
            if name not in ARENA_FAST_PATHS:
                raise ValueError(f'runtime upgrade cannot change {name}')
            changed_arena.append(name)
    # The timed C ABI, watchdog, SDK, and execution/forfeit semantics stay exact.
    old_arena = source_files(reference/'generals_arena')
    new_arena = source_files(installed_root('generals_arena'))
    protected = {'execution.py'} | {p for p in old_arena if p.startswith('sdk/')}
    for name in protected:
        if old_arena.get(name) != new_arena.get(name):
            raise ValueError(f'runtime upgrade cannot change the native boundary: {name}')
    return {'environment_files': sorted(changed_env), 'arena_files': sorted(changed_arena)}


def runtime_identity():
    return {p: source_digest(p) for p in ('generals', 'generals_arena', 'generals_baselines')}


def same_observation(actual, expected):
    for field in fields(actual):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(left, np.ndarray):
            if left.dtype != right.dtype or not np.array_equal(left, right) or left.flags.writeable:
                raise ValueError(f'observation changed: {field.name}')
        elif field.name == 'stats':
            for key in ('army', 'territory', 'cities'):
                a, b = getattr(left, key), getattr(right, key)
                if not np.array_equal(a, b) or a.flags.writeable:
                    raise ValueError(f'observation stats changed: {key}')
        elif left != right:
            raise ValueError(f'observation changed: {field.name}')


def verify_replays(store, per_count=8, per_baseline=12, progress=None):
    """Replay stored actions; call baselines only through their public controller API."""
    from generals import Action, MapConfig, make_engine
    from generals.core.mapgen import generate_map
    from generals.core.observation import observe
    from generals.core.recording import state_digest, state_from_dict
    from generals.core.state import compute_stats
    from .league.observation import local_observation
    from .registry import load_controller

    entries = store.entrants()
    baselines = {key: e for key, e in entries.items() if e['kind'] == 'baseline'}
    selected = {}
    with store.connect() as db:
        for count in store.config.player_counts:
            rows = db.execute("SELECT id FROM games WHERE status='complete' AND "
                "json_array_length(plan,'$.entrant_ids')=? AND "
                "COALESCE(json_array_length(result,'$.forfeits'),0)=0 ORDER BY sequence DESC LIMIT ?",
                (count, per_count)).fetchall()
            if len(rows) < per_count:
                raise ValueError(f'need {per_count} historical games for {count} players')
            selected.update((row['id'], None) for row in rows)
        for key in baselines:
            rows = db.execute("SELECT DISTINCT games.id FROM games,json_each(plan,'$.entrant_ids') p "
                "WHERE status='complete' AND p.value=? AND "
                "COALESCE(json_array_length(result,'$.forfeits'),0)=0 ORDER BY sequence DESC LIMIT ?",
                (key, per_baseline)).fetchall()
            if len(rows) < per_baseline:
                raise ValueError(f'need {per_baseline} historical games for {baselines[key]["name"]}')
            selected.update((row['id'], None) for row in rows)

    counts, decisions = Counter(), Counter()
    ticks = observations = 0
    for game_id in selected:
        with store.connect() as db:
            row = db.execute('SELECT plan,result,replay FROM games WHERE id=?', (game_id,)).fetchone()
        plan, result = json.loads(row['plan']), json.loads(row['result'])
        trace = json.loads(gzip.decompress(row['replay']))
        if trace['schema'] != 1 or len(trace['actions']) != result['ticks'] or len(trace['digests']) != result['ticks']:
            raise ValueError(f'invalid historical replay: {game_id}')
        state = state_from_dict(trace['initial'])
        if state_digest(state) != result['initial_digest']:
            raise ValueError(f'initial digest changed: {game_id}')
        generated = generate_map(MapConfig(**plan['map']), plan['map_seed'])
        if state_digest(generated) != result['initial_digest']:
            raise ValueError(f'map generation changed: {game_id}')
        engine = make_engine(state)
        controllers = {}
        try:
            for seat, key in enumerate(plan['entrant_ids']):
                if key in baselines:
                    controller = load_controller(baselines[key], SimpleNamespace(map=MapConfig(**plan['map']), visibility='local'))
                    controller.reset(player_id=seat, seed=plan['controller_seeds'][seat])
                    controllers[seat] = controller
            stats = compute_stats(state)
            for recorded, expected_digest in zip(trace['actions'], trace['digests']):
                actions = [Action(*values) for values in recorded]
                for seat in range(len(state.alive)):
                    obs = local_observation(state, seat, stats)
                    same_observation(obs, observe(state, seat))
                    observations += 1
                    if seat in controllers and state.alive[seat]:
                        action = controllers[seat].act(obs)
                        if action != actions[seat]:
                            raise ValueError(f'black-box action changed: game={game_id} seat={seat} tick={state.tick}')
                        decisions[plan['entrant_ids'][seat]] += 1
                outcome = engine.step(actions)
                stats = outcome.stats
                if state_digest(state) != expected_digest:
                    raise ValueError(f'state diverged: game={game_id} tick={state.tick}')
                for move in outcome.moves:
                    if move.player_id in controllers:
                        controllers[move.player_id].on_result(move)
                ticks += 1
            if state_digest(state) != result['final_digest']:
                raise ValueError(f'final digest changed: {game_id}')
            for key, value in result['final_stats'].items():
                if getattr(stats, key).tolist() != value:
                    raise ValueError(f'final stats changed: {game_id}/{key}')
            counts[len(state.alive)] += 1
        finally:
            engine.close()
            for controller in controllers.values():
                close = getattr(controller, 'close', None)
                if close is not None:
                    close()
        if progress:
            progress(sum(counts.values()), len(selected))
    if any(not decisions[key] for key in baselines):
        raise ValueError('every registered baseline must reproduce historical decisions')
    return {'games': list(selected), 'games_by_players': dict(counts), 'ticks': ticks,
            'observations': observations, 'baseline_decisions': dict(decisions)}


def compatible_implementation(store, entrant, package, expected, current):
    """Accept only a recorded, exact package transition for this entrant."""
    if current == expected:
        return True
    with store.connect() as db:
        row = db.execute("SELECT value FROM meta WHERE key='runtime_compatibility'").fetchone()
    for receipt in json.loads(row[0]) if row else []:
        if receipt['to_rules_id'] != store.manifest['rules_id']:
            continue
        for alias in receipt['package_aliases']:
            if alias == {'entrant_id': entrant['entrant_id'], 'package': package,
                         'from': expected, 'to': current}:
                return True
    return False


def install_verified(store, previous, current, report):
    """Caller holds all process locks; preserve IDs, accounts, ratings and games."""
    if runtime_identity() != report['runtime'] or canonical(store.config.manifest()) != canonical(current):
        raise ValueError('runtime changed during verification; verify again')
    backup_dir = store.directory/'runtime_migrations'
    backup_dir.mkdir(exist_ok=True)
    key = uuid.uuid4().hex
    backup = backup_dir/f'{key}.sqlite3'
    with store.connect() as db:
        with sqlite3.connect(backup) as target:
            db.backup(target)
    with store.connect(write=True) as db:
        saved = json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
        if canonical(saved) != canonical(previous):
            raise ValueError('league changed during verification')
        if (db.execute("SELECT 1 FROM games WHERE status IN ('live','pending') LIMIT 1").fetchone() or
            db.execute("SELECT 1 FROM jobs WHERE status='queued' AND done < budget LIMIT 1").fetchone()):
            raise ValueError('finish live, pending and queued games before upgrading runtime')
        aliases = []
        for row in db.execute("SELECT data FROM entrants WHERE kind='baseline'"):
            entrant = json.loads(row['data'])
            if not report['replay_validation']['baseline_decisions'].get(entrant['entrant_id']):
                raise ValueError('baseline was not verified')
            for package, expected in entrant['implementation'].items():
                actual = report['runtime'][package]
                if expected != actual:
                    aliases.append({'entrant_id': entrant['entrant_id'], 'package': package,
                                    'from': expected, 'to': actual})
        receipt = {'id': key, 'from_manifest': previous, 'to_manifest': current,
                   'to_rules_id': current['rules_id'], 'at': time.time(),
                   'reason': 'replay-verified performance-only runtime upgrade',
                   'backup': str(backup), 'package_aliases': aliases, 'evidence': report,
                   'next_sequence': db.execute('SELECT COALESCE(MAX(sequence),-1)+1 FROM games').fetchone()[0]}
        row = db.execute("SELECT value FROM meta WHERE key='runtime_compatibility'").fetchone()
        history = json.loads(row[0]) if row else []
        history.append(receipt)
        db.execute("INSERT OR REPLACE INTO meta VALUES ('runtime_compatibility',?)", (canonical(history),))
        db.execute("UPDATE meta SET value=? WHERE key='manifest'", (canonical(current),))
    store.manifest = current
    write_json(backup_dir/f'{key}.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reference-runtime', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    from .league.service import process_lock
    from .league.store import LeagueStore
    store = LeagueStore(args.directory)
    previous, current = store.manifest, store.config.manifest()
    report = {'from_rules_id': previous['rules_id'], 'to_rules_id': current['rules_id'],
              'runtime': runtime_identity(), 'source_review': check_transition(previous, current, args.reference_runtime)}
    def progress(done, total):
        if done % 8 == 0 or done == total:
            print(f'Runtime compatibility: {done}/{total} historical games verified', file=sys.stderr, flush=True)
    report['replay_validation'] = verify_replays(store, progress=progress)
    if runtime_identity() != report['runtime']:
        raise ValueError('runtime changed during verification; verify again')
    write_json(args.report, report)
    if args.apply:
        with process_lock(store, '.server.lock'), process_lock(store, '.worker.lock'), process_lock(store, '.registry.lock'):
            receipt = install_verified(store, previous, current, report)
        print(json.dumps({'ok': True, 'applied': True, 'receipt': receipt['id'], 'report': str(args.report)}))
    else:
        print(json.dumps({'ok': True, 'applied': False, 'report': str(args.report)}))


if __name__ == '__main__':
    main()
