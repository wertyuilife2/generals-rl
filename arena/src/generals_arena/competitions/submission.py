"""Immutable strategy submissions and publication evaluations."""
import json
from pathlib import Path
import time

from generals_arena.policies.artifacts import select_policy, snapshot_policy, snapshot_bundle
from generals_arena.policies.entries import prepare_entrant
from generals_arena.policies.native import decision_timeout
from generals_arena.matches.results import EXECUTION_POLICY
from generals_arena.competitions.config import DEFAULT_LEAGUE
from generals_arena.competitions.jobs import run_jobs, worker_count
from generals_arena.competitions.store import CompetitionStore
from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.schedules import candidate_panel
from generals_arena.competitions.private import run_trial, summarize, package_profiles
from generals_arena.util import canonical, digest, process_lock, write_json

PUBLICATION_GAMES = 1000
BASELINES = ('random', 'aggressive', 'expansion', 'defensive')


def _alias(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 128:
        raise ValueError('name must contain 1–128 characters')
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError('name cannot contain control characters')
    return value.strip()


def _preflight(official, spec, progress=None, workers=1):
    opponents = [select_policy(official.directory, 'baseline/'+kind, entries=official.entrants()) for kind in BASELINES]
    entries = [prepare_entrant(item) for item in [spec, *opponents]]
    descriptor = {'mode': 'preflight', 'players': [e['entrant_id'] for e in entries],
                  'games': 8, 'seed': 8675309, 'rules': official.manifest, 'rating_policy': POLICY,
                  'execution_policy': EXECUTION_POLICY}
    return run_trial(official, [spec, *opponents], official.config, descriptor,
        lambda store, identity: candidate_panel(store.config, entries[0]['entrant_id'], descriptor['players'][1:],
                                       8, 8675309, identity), progress, workers)


def _violations(official, version):
    """Inspect persisted failures, including failures as another player's opponent."""
    for directory in [official.directory, *(official.directory/'runs').glob('*/league')]:
        store = CompetitionStore(directory)
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


def publication_ratings(leaderboards, entrant_id):
    """Expose each independently rated pool without blending its reference."""
    scores = {}
    for mode, board in leaderboards.items():
        row = next(row for row in board['ranking'] if row['entrant_id'] == entrant_id)
        scores[mode] = {'rating': row['elo'], 'raw_rating': row['raw_elo'],
                        'rank': 1 + sum(other['elo'] > row['elo'] for other in board['ranking']),
                        'games': row['games'], 'rated_games': row['rated_games'],
                        'rating_model': board['rating_model']}
    return scores


def commit(source, *, name=None, league=DEFAULT_LEAGUE, parameters=None, progress=None,
           workers=None, device='cpu'):
    """One source version has one formal job, regardless of path or alias."""
    workers = worker_count(workers)
    if name is not None:
        name = _alias(name)
    official = CompetitionStore(Path(league).resolve())
    official.verify_execution()
    packaged = (Path(source)/'manifest.json').is_file()
    if packaged:
        if parameters is not None:
            raise ValueError('--params applies to native policies')
        spec = snapshot_bundle(official.directory, source, device=device)
        official.validate_entrant_rules(spec)
    else:
        if device != 'cpu':
            raise ValueError('--device applies to packaged policies')
        spec = snapshot_policy(official.directory, source, parameters)
    version = spec['version']
    entry = prepare_entrant(spec) if packaged else None
    execution_id = (digest({key: entry[key] for key in ('factory', 'kwargs', 'implementation')})
                    if packaged else None)
    directory = official.directory/'commits'/version
    with process_lock(official.directory, '.registry.lock', blocking=True):
        directory.parent.mkdir(exist_ok=True)
        if (directory/'commit.json').exists():
            record = json.loads((directory/'commit.json').read_text())
            if name is not None and name != record['name']:
                raise ValueError(f'this version already uses alias {record["name"]!r}; omit --name to resume')
            if packaged and record.get('execution_id') != execution_id:
                raise ValueError('this model was committed with a different inference deployment; use a separate league')
        else:
            existing = next((e for e in official.entrants().values()
                             if e['kind'] == spec['kind'] and e['version'] == version), None)
            default_alias = spec['name'] if packaged else f'{Path(source).stem[:100]}-{version[:12]}'
            alias = name or (existing['name'] if existing else default_alias)
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
            if packaged:
                record.update(kind='package', policy_id=version, execution_id=execution_id)
            directory.mkdir(exist_ok=True)
            write_json(directory/'commit.json', record)
    with process_lock(directory, '.commit.lock'):
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
            with process_lock(official.directory, '.registry.lock', blocking=True):
                entrant, job_id = official.add(formal, games=PUBLICATION_GAMES)
                record.update(job_id=job_id, entrant_id=entrant['entrant_id'], preflight_report=probe['report'])
                write_json(directory/'commit.json', record)
        started, error = time.monotonic(), None
        try:
            run_jobs(official, workers=workers, job_id=record['job_id'], progress=progress)
        except Exception as exc:
            error = str(exc)
        boards = json.loads(canonical(official.leaderboards()))
        report = {**summarize(official, record['entrant_id'], record['job_id']),
            'name': record['name'], 'version': version, 'entrant_id': record['entrant_id'],
            'ratings': publication_ratings(boards, record['entrant_id']), 'leaderboards': boards,
            'rating_policy': POLICY, 'publication_games': PUBLICATION_GAMES, 'workers': workers,
            'last_run_seconds': time.monotonic()-started, 'report': str(report_path),
            'preflight_report': record['preflight_report']}
        if packaged:
            report['policy_id'] = version
            report['policy_packages'] = package_profiles([official.entrants()[record['entrant_id']]])
        if error:
            report['ok'], report['execution_error'] = False, error
        report = json.loads(canonical(report))
        write_json(report_path, report)
        return report


def refresh_publications(store):
    """Explicit re-rating replaces published score snapshots, not simulations."""
    boards = store.leaderboards(include_retired=True)
    write_json(store.directory/'leaderboard.json', boards)
    entrants = store.entrants()
    for path in (store.directory/'commits').glob('*/publication.json'):
        report = json.loads(path.read_text())
        entrant_id = report.get('entrant_id')
        if entrant_id not in entrants:
            continue
        report.update(ratings=publication_ratings(boards, entrant_id), leaderboards=boards,
                      rating_policy=POLICY, rating_recomputed=True)
        write_json(path, report)
    path = store.directory/'reevaluation.json'
    if path.exists():
        report = json.loads(path.read_text())
        report.update(leaderboards=boards, rating_policy=POLICY, rating_recomputed=True)
        write_json(path, report)


def status(league=DEFAULT_LEAGUE):
    official = CompetitionStore(Path(league).resolve())
    official.verify_execution()
    jobs = {j['id']: j for j in official.jobs()}
    commits = []
    for path in sorted((official.directory/'commits').glob('*/commit.json')):
        record = json.loads(path.read_text())
        commits.append({**record, 'job': jobs.get(record.get('job_id')),
                        'report': str(path.with_name('publication.json'))})
    return {'league_directory': str(official.directory), 'commits': commits,
            'runs': [str(p) for p in sorted((official.directory/'runs').glob('*/report.json'))]}
