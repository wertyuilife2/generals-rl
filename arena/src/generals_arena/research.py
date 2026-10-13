"""Source-first Arena submissions and private games; all artifacts are Arena-owned."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import tempfile
import time
from types import SimpleNamespace
from zipfile import ZipFile

from .entrants import prepare_entrant, verify_artifacts
from .execution import EXECUTION_POLICY, decision_timeout
from .league.service import run_jobs, process_lock
from .league.store import LeagueStore
from .league.rating_policy import POLICY
from .native import DECISION_NS, build_bundle, source_identity
from .util import canonical, digest, file_digest, integer, write_json

DEFAULT_LEAGUE = Path('outputs/arena/native_1000')
PUBLICATION_GAMES = 1000
BASELINES = ('random', 'aggressive', 'expansion', 'defensive')


def worker_count(value):
    value = min(4, os.cpu_count() or 1) if value is None else integer('workers', value, 1)
    if value > 32:
        raise ValueError('workers must be <= 32')
    return value


@contextmanager
def _lock(directory):
    with (Path(directory)/'.registry.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _official(directory):
    store = LeagueStore(Path(directory).resolve())
    store.verify_execution()
    return store


def _alias(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 128:
        raise ValueError('name must contain 1–128 characters')
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError('name cannot contain control characters')
    return value.strip()


def _policy(store, version):
    if not re.fullmatch('[a-f0-9]{64}', version):
        raise ValueError('policy/ requires a full version hash from a run or commit report')
    directory = store.directory/'policies'/version
    metadata = json.loads((directory/'build.json').read_text())
    bundle = directory/'policy.zip'
    if metadata['version'] != version or file_digest(bundle) != metadata['bundle_sha256']:
        raise ValueError('archived policy artifact changed')
    return {'name': f'cpp-{version[:16]}', 'kind': 'cpp', 'version': version, 'bundle': str(bundle)}


def snapshot_policy(store, source, parameters=None):
    """Deduplicate by content, independent of source location and public alias."""
    snapshot = source_identity(source, parameters)
    version = snapshot[0]
    parent = store.directory/'policies'
    with _lock(store.directory):
        parent.mkdir(exist_ok=True)
        target = parent/version
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=parent) as temporary:
                scratch = Path(temporary)/'snapshot'
                scratch.mkdir()
                metadata = build_bundle(source, scratch/'policy.zip', parameters, snapshot=snapshot)
                write_json(scratch/'build.json', {**metadata, 'bundle_sha256': file_digest(scratch/'policy.zip')})
                with ZipFile(scratch/'policy.zip') as archive:
                    for name in archive.namelist():
                        if name.startswith(('source/', 'sdk/')) or name == 'params.json':
                            path = scratch/name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(archive.read(name))
                scratch.rename(target)
    return _policy(store, version)


def _selector(store, value, parameters=None):
    value = str(value)
    if value.startswith(('baseline/', 'baseline:')):
        kind = value[len('baseline/'):].removesuffix('.cpp')
        if kind == 'wait':
            from .reference import wait_spec
            return wait_spec(store)
        if kind not in BASELINES:
            raise ValueError(f'unknown baseline; choose from {", ".join((*BASELINES, "wait"))}')
        return {'name': 'random' if kind == 'random' else f'baseline/{kind}',
                'kind': 'baseline', 'policy': kind}
    if value.startswith('policy/'):
        return _policy(store, value[len('policy/'):])
    if value.startswith('@'):
        entry = next((e for e in store.entrants().values() if e['name'] == value[1:]), None)
        if entry is None or entry['kind'] != 'cpp':
            raise ValueError('@name must identify a committed C++ policy')
        verify_artifacts(entry)
        # Use the content identity in private games, even if selected via an alias.
        return {'name': f'cpp-{entry["version"][:16]}', 'kind': 'cpp',
                'version': entry['version'], 'bundle': entry['artifact_paths']['bundle']}
    return snapshot_policy(store, value, parameters)


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


def _fixed_plans(store, seats, games, seed, identity, size):
    """Every selected player participates; rotate seats on each shared map."""
    rng, plans = random.Random(seed), []
    count = len(seats)
    while len(plans) < games:
        map_seed = rng.getrandbits(63)
        config = asdict(store.config.sample_map(count, map_seed))
        if size['mode'] == 'fixed':
            config.update(width=size['width'], height=size['height'])
        elif size['mode'] == 'range':
            config.update(width=rng.randint(size['min'], size['max']), height=rng.randint(size['min'], size['max']))
        seeds = [rng.getrandbits(63) for _ in seats]
        rotations = min(count, games-len(plans))
        for rotation in range(rotations):
            order = list(range(rotation, count))+list(range(rotation))
            plans.append({'game_id': digest([identity, len(plans)]),
                'entrant_ids': [seats[i] for i in order], 'instance_ids': order,
                'controller_seeds': [seeds[i] for i in order], 'map_seed': map_seed,
                'map': config, 'rules_id': store.manifest['rules_id'], 'human': False,
                'rating_policy': POLICY, 'rating_group_size': rotations})
    return plans


def _plans(store, target, opponents, games, seed, identity):
    """Fixed independent schedule. Candidate code/name cannot change maps or RNG."""
    rng, plans, block = random.Random(seed), [], 0
    counts = store.config.player_counts
    while len(plans) < games:
        count = rng.choices(counts, weights=[1/p for p in counts])[0]
        instances = list(enumerate([target, *[rng.choice(opponents) for _ in range(count-1)]]))
        rng.shuffle(instances)
        map_seed = rng.getrandbits(63)
        map_config = asdict(store.config.sample_map(count, map_seed))
        seeds = {i: rng.getrandbits(63) for i, _ in instances}
        rotations = min(count, games-len(plans))
        for rotation in range(rotations):
            seats = instances[rotation:]+instances[:rotation]
            plans.append({'game_id': digest([identity, len(plans)]),
                'entrant_ids': [key for _, key in seats], 'instance_ids': [i for i, _ in seats],
                'controller_seeds': [seeds[i] for i, _ in seats], 'map_seed': map_seed,
                'map': map_config, 'rules_id': store.manifest['rules_id'], 'human': False,
                'evaluation_block': block, 'rating_policy': POLICY, 'rating_group_size': rotations})
        block += 1
    return plans


def summarize(store, candidate, job_id):
    from .league.rating import ranks
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


def _trial(official, specs, rules, descriptor, planner, progress=None, workers=1):
    run_id = digest(descriptor)
    parent = official.directory/'runs'
    directory = parent/run_id
    with _lock(official.directory):
        parent.mkdir(exist_ok=True)
        if not directory.exists():
            with tempfile.TemporaryDirectory(dir=parent) as temporary:
                scratch = Path(temporary)/'run'
                scratch.mkdir()
                store = LeagueStore.create(scratch/'league', rules, reference='random')
                if official.entrants()[official.reference]['kind'] == 'cpp':
                    # The scratch directory is renamed below; reuse the stable
                    # official bundle path rather than embedding scratch paths.
                    original = official.entrants()[official.reference]
                    reference = store.register({'name': original['name'], 'kind': 'cpp',
                                                'version': original['version'],
                                                'bundle': original['artifact_paths']['bundle']})
                    store.set_rating_reference(reference['entrant_id'])
                for spec in specs:
                    store.register(spec)
                _, job_id = store.add(specs[0], games=descriptor['games'])
                store.insert_plans(planner(store, run_id), job_id)
                write_json(scratch/'request.json', {'request': descriptor, 'job_id': job_id})
                scratch.rename(directory)
        request = json.loads((directory/'request.json').read_text())
        if canonical(request['request']) != canonical(descriptor):
            raise ValueError('stored run request changed')
    store = LeagueStore(directory/'league')
    store.verify_execution()
    with process_lock(store, '.run.lock'):
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
            'players': players, 'timing': timing, 'violations': result['violations'],
            'workers': workers,
            'league_directory': str(store.directory), 'report': str(directory/'report.json'),
            'last_run_seconds': time.monotonic()-started}
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
    with process_lock(SimpleNamespace(directory=output), '.export.lock'):
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
        output=None, parameters=None, max_ticks=None, progress=None, workers=1):
    workers = worker_count(workers)
    integer('games', games, 1)
    integer('seed', seed)
    if not 2 <= len(players) <= 8:
        raise ValueError('run requires 2–8 --player arguments, one per seat')
    dimensions = _size(size)
    official = _official(league)
    rules = replace(official.config, player_counts=(len(players),),
                    max_ticks=official.config.max_ticks if max_ticks is None else integer('max_ticks', max_ticks, 1))
    specs = [_selector(official, value, parameters) for value in players]
    entries = [prepare_entrant(spec) for spec in specs]
    descriptor = {'mode': 'run', 'players': [e['entrant_id'] for e in entries],
                  'games': games, 'seed': seed, 'size': dimensions, 'rules': rules.manifest(),
                  'rating_policy': POLICY, 'execution_policy': EXECUTION_POLICY}
    target = _export_target(official, output, digest(descriptor))
    report = _trial(official, specs, rules, descriptor,
        lambda store, identity: _fixed_plans(store, descriptor['players'], games, seed, identity, dimensions), progress, workers)
    _export(report, target)
    return report


def _preflight(official, spec, progress=None, workers=1):
    opponents = [_selector(official, 'baseline/'+kind) for kind in BASELINES]
    entries = [prepare_entrant(item) for item in [spec, *opponents]]
    descriptor = {'mode': 'preflight', 'players': [e['entrant_id'] for e in entries],
                  'games': 8, 'seed': 8675309, 'rules': official.manifest, 'rating_policy': POLICY,
                  'execution_policy': EXECUTION_POLICY}
    return _trial(official, [spec, *opponents], official.config, descriptor,
        lambda store, identity: _plans(store, entries[0]['entrant_id'], descriptor['players'][1:],
                                       8, 8675309, identity), progress, workers)


def _violations(official, version):
    """Inspect persisted failures, including failures as another player's opponent."""
    for directory in [official.directory, *(official.directory/'runs').glob('*/league')]:
        store = LeagueStore(directory)
        entries = store.entrants()
        with store.connect() as db:
            rows = db.execute("SELECT result FROM games WHERE status='failed' AND result IS NOT NULL")
            for row in rows:
                result = json.loads(row[0])
                if decision_timeout(result.get('error_detail', {})):
                    continue
                seat = result.get('error_detail', {}).get('player')
                if seat is not None and entries[result['entrant_ids'][seat]].get('version') == version:
                    return True
    return False


def commit(source, *, name=None, league=DEFAULT_LEAGUE, parameters=None, progress=None, workers=None):
    """One source version has one formal job, regardless of path or alias."""
    workers = worker_count(workers)
    if name is not None:
        name = _alias(name)
    official = _official(league)
    spec = snapshot_policy(official, source, parameters)
    version = spec['version']
    directory = official.directory/'commits'/version
    with _lock(official.directory):
        directory.parent.mkdir(exist_ok=True)
        if (directory/'commit.json').exists():
            record = json.loads((directory/'commit.json').read_text())
            if name is not None and name != record['name']:
                raise ValueError(f'this version already uses alias {record["name"]!r}; omit --name to resume')
        else:
            existing = next((e for e in official.entrants().values()
                             if e['kind'] == 'cpp' and e['version'] == version), None)
            alias = name or (existing['name'] if existing else f'{Path(source).stem[:100]}-{version[:12]}')
            if existing and alias != existing['name']:
                raise ValueError(f'this version is already committed as {existing["name"]!r}')
            claimed = [e for e in official.entrants().values() if e['name'] == alias]
            for path in directory.parent.glob('*/commit.json'):
                saved = json.loads(path.read_text())
                if saved['name'] == alias:
                    claimed.append(saved)
            if any(entry.get('version') != version for entry in claimed):
                raise ValueError('alias already identifies a different version; choose a new --name')
            record = {'name': _alias(alias), 'version': version, 'created': time.time()}
            directory.mkdir(exist_ok=True)
            write_json(directory/'commit.json', record)
    with process_lock(SimpleNamespace(directory=directory), '.commit.lock'):
        record = json.loads((directory/'commit.json').read_text())
        report_path = directory/'publication.json'
        jobs = {j['id']: j for j in official.jobs()}
        if record.get('job_id') in jobs and jobs[record['job_id']]['status'] == 'complete' and report_path.exists():
            saved = json.loads(report_path.read_text())
            if saved['ok']:
                return saved  # Keep the rating at completion, not today's rating.
        if _violations(official, version):
            raise ValueError('policy version has a recorded violation; commit a corrected new version')
        if record.get('job_id') in jobs and jobs[record['job_id']]['status'] == 'failed':
            official.retry(record['job_id'])
        if 'job_id' not in record:
            probe = _preflight(official, spec, progress, workers)
            if not probe['ok']:
                report = {'ok': False, 'phase': 'preflight', 'name': record['name'], 'version': version,
                    'completed_games': 0, 'publication_games': PUBLICATION_GAMES,
                    'timing': probe['timing'], 'violations': probe['violations'],
                    'preflight_report': probe['report'], 'report': str(report_path)}
                write_json(report_path, report)
                return report
            formal = {**spec, 'name': record['name']}
            with _lock(official.directory):
                entrant, job_id = official.add(formal, games=PUBLICATION_GAMES)
                record.update(job_id=job_id, entrant_id=entrant['entrant_id'], preflight_report=probe['report'])
                write_json(directory/'commit.json', record)
        started, error = time.monotonic(), None
        try:
            run_jobs(official, workers=workers, job_id=record['job_id'], progress=progress)
        except Exception as exc:
            error = str(exc)
        board = json.loads(canonical(official.leaderboard()))
        row = next(r for r in board['ranking'] if r['entrant_id'] == record['entrant_id'])
        report = {**summarize(official, record['entrant_id'], record['job_id']),
            'name': record['name'], 'version': version, 'entrant_id': record['entrant_id'],
            'rating': row['elo'], 'rank': 1+sum(r['elo'] > row['elo'] for r in board['ranking']),
            'leaderboard': board, 'publication_games': PUBLICATION_GAMES, 'workers': workers,
            'last_run_seconds': time.monotonic()-started, 'report': str(report_path),
            'preflight_report': record['preflight_report']}
        if error:
            report['ok'], report['execution_error'] = False, error
        report = json.loads(canonical(report))
        write_json(report_path, report)
        return report


def refresh_publications(store):
    """Explicit re-rating replaces published score snapshots, not simulations."""
    board = store.leaderboard()
    write_json(store.directory/'leaderboard.json', board)
    rows = {row['entrant_id']:row for row in board['ranking']}
    for path in (store.directory/'commits').glob('*/publication.json'):
        report = json.loads(path.read_text())
        row = rows.get(report.get('entrant_id'))
        if row is None:
            continue
        report.update(rating=row['elo'], rank=1+sum(r['elo'] > row['elo'] for r in rows.values()),
                      leaderboard=board, rating_policy=POLICY, rating_recomputed=True)
        write_json(path, report)
    path = store.directory/'reevaluation.json'
    if path.exists():
        report = json.loads(path.read_text())
        report.update(leaderboard=board, rating_policy=POLICY, rating_recomputed=True)
        write_json(path, report)


def status(league=DEFAULT_LEAGUE):
    official = _official(league)
    jobs = {j['id']: j for j in official.jobs()}
    commits = []
    for path in sorted((official.directory/'commits').glob('*/commit.json')):
        record = json.loads(path.read_text())
        commits.append({**record, 'job': jobs.get(record.get('job_id')),
                        'report': str(path.with_name('publication.json'))})
    return {'league_directory': str(official.directory), 'commits': commits,
            'runs': [str(p) for p in sorted((official.directory/'runs').glob('*/report.json'))]}


def add_cli(commands):
    submit = commands.add_parser('commit', help='compile a C++ source and run its official 1000-game evaluation')
    submit.add_argument('source', type=Path, help='.cpp file or source directory')
    submit.add_argument('--name', help='immutable display alias shown on the leaderboard')
    submit.add_argument('--params', type=Path, help='JSON object of fixed policy parameters')
    submit.add_argument('--league', type=Path, default=DEFAULT_LEAGUE)
    submit.add_argument('--workers', type=int, default=None, help='parallel games (default: up to 4 CPU workers)')
    play = commands.add_parser('run', help='private games between explicit seats; never updates official ratings')
    play.add_argument('--player', action='append', required=True,
                      help='baseline/KIND, source.cpp, source directory, policy/HASH, or @alias; repeat for each seat')
    play.add_argument('--size', default='auto', help='auto, WIDTHxHEIGHT, or MIN:MAX')
    play.add_argument('--games', type=int, default=1)
    play.add_argument('--seed', type=int, default=42)
    play.add_argument('--max-ticks', type=int, help='private-game tick limit; defaults to official rules')
    play.add_argument('--params', type=Path, help='fixed JSON parameters for all source-path players')
    play.add_argument('--output', type=Path, help='optional report and replay database export directory')
    play.add_argument('--league', type=Path, default=DEFAULT_LEAGUE)
    play.add_argument('--workers', type=int, default=1, help='parallel games (default: 1)')
    inspect = commands.add_parser('status', help='show managed submissions, jobs, and run report paths')
    inspect.add_argument('--league', type=Path, default=DEFAULT_LEAGUE)


def command(args):
    import sys
    last_progress = 0.
    def progress(done):
        nonlocal last_progress
        now = time.monotonic()
        if done == 1 or now-last_progress >= 2:
            print(f'Arena: {done} new games completed', file=sys.stderr, flush=True)
            last_progress = now
    parameters = json.loads(args.params.read_text()) if getattr(args, 'params', None) else None
    if args.command == 'commit':
        report = commit(args.source, name=args.name, league=args.league, parameters=parameters,
                        progress=progress, workers=args.workers)
    elif args.command == 'run':
        report = run(args.player, league=args.league, size=args.size, games=args.games, seed=args.seed,
                     output=args.output, parameters=parameters, max_ticks=args.max_ticks,
                     progress=progress, workers=args.workers)
    else:
        report = status(args.league)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report.get('ok', True) else 1
