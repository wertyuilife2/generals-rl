"""Explicit-seat private games and reusable admission trials."""
from dataclasses import replace
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import time

from generals_arena.policies.entries import prepare_entrant
from generals_arena.policies.artifacts import select_policy
from generals_arena.policies.native import DECISION_NS, decision_timeout
from generals_arena.matches.results import EXECUTION_POLICY
from generals_arena.competitions.config import DEFAULT_LEAGUE
from generals_arena.competitions.jobs import run_jobs, worker_count
from generals_arena.competitions.store import CompetitionStore
from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.schedules import fixed_seats
from generals_arena.util import canonical, digest, integer, process_lock, write_json


def package_profiles(entries):
    """Report the generic package identity without interpreting its configuration."""
    from generals_arena.policies.artifacts import validate_bundle
    return {entry['entrant_id']: {
        'policy_id': entry['version'],
        'manifest': validate_bundle(entry['factory_kwargs']['bundle'])['manifest'],
        'device': entry['factory_kwargs']['device'], 'decision_deadline': None,
    } for entry in entries if entry['kind'] == 'package'}


def _validate_package_size(entries, rules, size, players):
    """Check every possible private-game shape before persisting a run."""
    from generals_arena.policies.artifacts import capabilities_supports
    from generals_arena.policies.artifacts import validate_bundle
    for entry in entries:
        if entry['kind'] != 'package':
            continue
        caps = validate_bundle(entry['factory_kwargs']['bundle'])['manifest']['capabilities']
        if players not in caps['player_counts']:
            raise ValueError(f'{entry["name"]}: unsupported player count {players}')
        if size['mode'] == 'fixed':
            shapes = [(size['width'], size['height'])]
        else:
            low, high = ((size['min'], size['max']) if size['mode'] == 'range'
                         else rules.side_range(players))
            shapes = ((w, h) for w in range(low, high+1) for h in range(low, high+1))
        for width, height in shapes:
            if not capabilities_supports(caps, width, height, players):
                raise ValueError(f'{entry["name"]}: unsupported map size {width}x{height}')


def _size(value):
    if value == 'auto':
        return {'mode': 'auto'}
    if re.fullmatch(r'\d+[xX]\d+', value):
        width, height = map(int, re.split('[xX]', value))
        if not 3 <= min(width, height) <= max(width, height) <= 64:
            raise ValueError('fixed map sides must be between 3 and 64')
        return {'mode': 'fixed', 'width': width, 'height': height}
    if re.fullmatch(r'\d+:\d+', value):
        low, high = map(int, value.split(':'))
        if not 3 <= low <= high <= 64:
            raise ValueError('map range must satisfy 3 <= min <= max <= 64')
        return {'mode': 'range', 'min': low, 'max': high}
    raise ValueError('size must be auto, WIDTHxHEIGHT, or MIN:MAX (for example 15:35)')


def summarize(store, candidate, job_id):
    from generals_arena.competitions.rating import ranks
    with store.connect() as db:
        rows = list(db.execute('SELECT status,result FROM games WHERE job_id=? ORDER BY rowid', (job_id,)))
    groups, errors, forfeits, historical_timeouts = {}, [], [], []
    entrants = store.entrants()
    total = {'calls': 0, 'total_ns': 0, 'max_ns': 0, 'timeouts': 0, 'errors': 0}
    candidate_violations = 0
    for row in rows:
        if not row['result']:
            continue
        result = json.loads(row['result'])
        forfeits.extend({'game_id': result['game_id'], **event}
                        for event in result.get('forfeits', [])
                        if result['entrant_ids'][event['seat']] == candidate)
        for record in result.get('decision_timing', []):
            if record['entrant_id'] != candidate:
                continue
            for key in ('calls', 'total_ns', 'timeouts', 'errors'):
                total[key] += record[key]
            total['max_ns'] = max(total['max_ns'], record['max_ns'])
        if row['status'] == 'failed':
            if decision_timeout(result.get('error_detail', {})):
                historical_timeouts.append(result['game_id'])
                continue
            seat = result.get('error_detail', {}).get('player')
            offender = result['entrant_ids'][seat] if seat is not None else None
            errors.append({**{key: result.get(key) for key in ('game_id','ticks','policy_error','error_detail')},
                           'entrant_id': offender, 'version': entrants.get(offender, {}).get('version')})
            if seat is not None and result['entrant_ids'][seat] == candidate:
                candidate_violations += 1
            continue
        if row['status'] != 'complete':
            continue
        ids, places = result['entrant_ids'], ranks(result)
        own = [i for i, key in enumerate(ids) if key == candidate]
        if not own:
            continue
        other = [i for i, key in enumerate(ids) if key != candidate]
        score = (sum((places[i] < places[j])+.5*(places[i] == places[j]) for i in own for j in other)
                 / (len(own)*len(other))) if other else None
        group = groups.setdefault(len(ids), {'games': 0, 'game_wins': 0, 'rank_sum': 0.,
                                           'pair_scores': [], 'timeouts': 0, 'invalid_actions': 0})
        group['games'] += 1
        group['game_wins'] += int(result['winner_seat'] in own)
        group['rank_sum'] += sum(places[i] for i in own)/len(own)
        group['timeouts'] += int(result['reason'] == 'tick_limit')
        group['invalid_actions'] += sum(result['invalid_actions'][i] for i in own)
        if score is not None:
            group['pair_scores'].append(score)
    by_players = {}
    for players, g in sorted(groups.items()):
        by_players[players] = {'games': g['games'], 'game_win_rate': g['game_wins']/g['games'],
            'mean_rank': g['rank_sum']/g['games'], 'tick_limit_games': g['timeouts'],
            'invalid_actions': g['invalid_actions'], 'scored_games': len(g['pair_scores']),
            'pair_score': sum(g['pair_scores'])/len(g['pair_scores']) if g['pair_scores'] else None}
    values = [g['pair_score'] for g in by_players.values() if g['pair_score'] is not None]
    job = next(j for j in store.jobs() if j['id'] == job_id)
    return {'ok': job['status'] == 'complete' and not errors and not total['errors'],
            'job': job, 'completed_games': job['done'], 'by_players': by_players,
            'macro_pair_score': sum(values)/len(values) if values else None,
            'timing': {**total, 'limit_ns': DECISION_NS}, 'violations': errors,
            'candidate_violations': candidate_violations,
            'forfeits': forfeits, 'historical_timeout_games': historical_timeouts,
            'league_directory': str(store.directory.resolve())}


def run_trial(official, specs, rules, descriptor, planner, progress=None, workers=1):
    run_id = digest(descriptor)
    parent = official.directory/'runs'
    directory = parent/run_id
    with process_lock(official.directory, '.registry.lock', blocking=True):
        parent.mkdir(exist_ok=True)
        if not directory.exists():
            with tempfile.TemporaryDirectory(dir=parent) as temporary:
                scratch = Path(temporary)/'run'
                scratch.mkdir()
                store = CompetitionStore.create(scratch/'league', rules)
                for spec in specs:
                    store.register(spec)
                _, job_id = store.add(specs[0], games=descriptor['games'])
                store.insert_plans(planner(store, run_id), job_id)
                write_json(scratch/'request.json', {'request': descriptor, 'job_id': job_id})
                scratch.rename(directory)
        request = json.loads((directory/'request.json').read_text())
        if canonical(request['request']) != canonical(descriptor):
            raise ValueError('stored run request changed')
    store = CompetitionStore(directory/'league')
    store.verify_execution()
    with process_lock(store.directory, '.run.lock'):
        job = next(j for j in store.jobs() if j['id'] == request['job_id'])
        if job['status'] in ('complete', 'failed') and (directory/'report.json').exists():
            return json.loads((directory/'report.json').read_text())
        started, error = time.monotonic(), None
        try:
            run_jobs(store, workers=workers, job_id=request['job_id'], progress=progress)
        except Exception as exc:
            error = str(exc)
        entries = [prepare_entrant(spec) for spec in specs]
        players = []
        for entry in {e['entrant_id']: e for e in entries}.values():
            result = summarize(store, entry['entrant_id'], request['job_id'])
            players.append({**{k: entry[k] for k in ('entrant_id', 'name', 'kind', 'version')},
                **{k: result[k] for k in ('by_players', 'macro_pair_score', 'timing', 'candidate_violations', 'forfeits')},
                'seats': [i for i,e in enumerate(entries) if e['entrant_id'] == entry['entrant_id']]})
        timing = {'limit_ns': DECISION_NS, 'calls': 0, 'total_ns': 0, 'max_ns': 0, 'timeouts': 0, 'errors': 0}
        for player in players:
            for key in ('calls', 'total_ns', 'timeouts', 'errors'):
                timing[key] += player['timing'][key]
            timing['max_ns'] = max(timing['max_ns'], player['timing']['max_ns'])
        report = {'ok': result['ok'] and error is None, 'run_id': run_id,
            'request': descriptor, 'job': result['job'], 'completed_games': result['completed_games'],
            'rating_modes': store.rating_modes(), 'rating_policy': POLICY,
            'players': players, 'timing': timing, 'violations': result['violations'],
            'workers': workers,
            'league_directory': str(store.directory), 'report': str(directory/'report.json'),
            'last_run_seconds': time.monotonic()-started}
        if profiles := package_profiles(entries):
            report['policy_packages'] = profiles
        if error:
            report['execution_error'] = error
        report = json.loads(canonical(report))
        write_json(directory/'report.json', report)
    return report


def _export_target(official, output, run_id):
    if output is None:
        return None
    output = Path(output).resolve()
    if output.is_relative_to(official.directory) or official.directory.is_relative_to(output):
        raise ValueError('output must be outside the managed league directory')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        marker = output/'arena-export.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('run_id') != run_id:
            raise ValueError('output is not empty or belongs to a different run')
    return output


def _export(report, output):
    if output is None:
        return
    output.mkdir(parents=True, exist_ok=True)
    with process_lock(output, '.export.lock'):
        marker = output/'arena-export.json'
        existing = [p for p in output.iterdir() if p.name != '.export.lock']
        if existing and (not marker.is_file() or json.loads(marker.read_text()).get('run_id') != report['run_id']):
            raise ValueError('output belongs to a different run')
        write_json(marker, {'run_id': report['run_id'], 'managed_report': report['report']})
        # Both completed and failed partial recordings travel with the SQLite backup.
        with sqlite3.connect(Path(report['league_directory'])/'league.sqlite3') as source:
            with sqlite3.connect(output/'league.sqlite3.tmp') as target:
                source.backup(target)
        (output/'league.sqlite3.tmp').replace(output/'league.sqlite3')
        write_json(output/'report.json', report)


def run(players, *, league=DEFAULT_LEAGUE, size='auto', games=1, seed=42,
        output=None, parameters=None, max_ticks=None, progress=None, workers=1, device=None):
    workers = worker_count(workers)
    integer('games', games, 1)
    integer('seed', seed)
    if not 2 <= len(players) <= 8:
        raise ValueError('run requires 2–8 --player arguments, one per seat')
    dimensions = _size(size)
    official = CompetitionStore(Path(league).resolve())
    official.verify_execution()
    rules = replace(official.config, player_counts=(len(players),),
                    max_ticks=official.config.max_ticks if max_ticks is None else integer('max_ticks', max_ticks, 1))
    specs = [select_policy(official.directory, value, parameters, device=device, entries=official.entrants()) for value in players]
    for spec in specs:
        if spec['kind'] == 'package':
            official.validate_entrant_rules(spec)
    entries = [prepare_entrant(spec) for spec in specs]
    _validate_package_size(entries, rules, dimensions, len(players))
    descriptor = {'mode': 'run', 'players': [e['entrant_id'] for e in entries],
                  'games': games, 'seed': seed, 'size': dimensions, 'rules': rules.manifest(),
                  'rating_policy': POLICY, 'execution_policy': EXECUTION_POLICY}
    target = _export_target(official, output, digest(descriptor))
    report = run_trial(official, specs, rules, descriptor,
        lambda store, identity: fixed_seats(store.config, descriptor['players'], games, seed, identity, dimensions), progress, workers)
    _export(report, target)
    return report
