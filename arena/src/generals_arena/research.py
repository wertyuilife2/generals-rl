"""Public research service: immutable C++ registration, black-box trials, publication."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
from types import SimpleNamespace
import uuid

from .entrants import prepare_entrant, verify_artifacts
from .league.service import run_jobs, process_lock
from .league.store import LeagueStore
from .native import SDK, DECISION_NS, build_bundle, source_identity
from .util import canonical, digest, integer, write_json

PUBLICATION_GAMES = 1000
PUBLICATION_LIMIT = 3
DEFAULT_OPPONENTS = ['baseline:random', 'baseline:aggressive', 'baseline:expansion', 'baseline:defensive']


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('names must contain 1–64 letters, digits, underscores or hyphens')
    return value


@contextmanager
def _lock(directory):
    with (Path(directory)/'.research.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def session(directory):
    root = Path(directory).resolve()
    config = json.loads((root/'session.json').read_text())
    if config.get('schema') != 1:
        raise ValueError('unsupported research session')
    store = LeagueStore(config['league'])
    store.verify_execution()
    if store.manifest['rules_id'] != config['rules_id']:
        raise ValueError('research session league rules changed')
    return root, config, store


def initialize(directory, league, workspace=None):
    store = LeagueStore(league)
    store.verify_execution()
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root):
        if (root/'session.json').exists():
            raise ValueError('research session already exists; use research status')
        if workspace is not None:
            workspace = Path(workspace).resolve()
            if workspace.exists() and any(workspace.iterdir()):
                raise ValueError('workspace must be empty; existing source is never overwritten')
            workspace.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SDK/'arena_policy.h', workspace/'arena_policy.h')
            (workspace/'policy.cpp').write_text(
                '#include "arena_policy.h"\nstruct Policy {};\n'
                'extern "C" void *arena_create(int, std::uint64_t, const char *) { return new Policy; }\n'
                'extern "C" std::int64_t arena_act(void *, const ArenaObservation *) { return 0; }\n'
                'extern "C" void arena_destroy(void *p) { delete static_cast<Policy *>(p); }\n')
        for name in ('strategies', 'evaluations'):
            (root/name).mkdir(exist_ok=True)
        config = {'schema': 1, 'id': uuid.uuid4().hex[:12], 'created': time.time(),
                  'league': str(store.directory.resolve()), 'rules_id': store.manifest['rules_id'],
                  'workspace': str(workspace) if workspace is not None else None,
                  'decision_limit_ns': DECISION_NS, 'publication_games': PUBLICATION_GAMES,
                  'publication_limit': PUBLICATION_LIMIT}
        write_json(root/'publications.json', [])
        write_json(root/'session.json', config)
    return config


def registered(root, name):
    path = root/'strategies'/_name(name)/'entrant.json'
    spec = json.loads(path.read_text())
    entrant = prepare_entrant(spec, base_dir=path.parent)
    identity = json.loads((path.parent/'identity.json').read_text())
    if entrant['entrant_id'] != identity['entrant_id']:
        raise ValueError('registered policy artifact or specification changed')
    verify_artifacts(entrant)
    return spec, entrant


def register(directory, source, name, parameters=None):
    root, config, _ = session(directory)
    name = _name(name)
    key, _, _ = source_identity(source, parameters)
    with _lock(root):
        target = root/'strategies'/name
        if target.exists():
            spec, entrant = registered(root, name)
            if entrant['version'] != key:
                raise ValueError('registered name is immutable; use a new candidate name')
            return {'candidate': name, 'entrant': entrant, 'entrant_file': str(target/'entrant.json')}
        with tempfile.TemporaryDirectory(dir=root/'strategies') as temporary:
            scratch = Path(temporary)/'snapshot'
            scratch.mkdir()
            metadata = build_bundle(source, scratch/'policy.zip', parameters)
            spec = {'name': f'cpp-{config["id"]}-{name}', 'kind': 'cpp',
                    'version': metadata['version'], 'bundle': str(target/'policy.zip')}
            write_json(scratch/'entrant.json', spec)
            write_json(scratch/'build.json', metadata)
            identity = prepare_entrant({**spec, 'bundle': str(scratch/'policy.zip')})
            write_json(scratch/'identity.json', {'entrant_id': identity['entrant_id']})
            # Ship the readable source alongside the exact executable bundle.
            from zipfile import ZipFile
            with ZipFile(scratch/'policy.zip') as archive:
                for item in archive.namelist():
                    if item.startswith(('source/', 'sdk/')) or item == 'params.json':
                        output = scratch/item
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(archive.read(item))
            scratch.rename(target)
    _, entrant = registered(root, name)
    return {'candidate': name, 'entrant': entrant, 'entrant_file': str(target/'entrant.json')}


def _opponent(root, selector, candidate):
    if selector == 'self':
        return candidate
    if selector.startswith('baseline:'):
        kind = selector.partition(':')[2]
        if kind not in ('random', 'aggressive', 'expansion', 'defensive'):
            raise ValueError('unknown baseline opponent')
        return {'name': 'random' if kind == 'random' else f'baseline:{kind}',
                'kind': 'baseline', 'policy': kind}
    return registered(root, selector)[0]


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
        for rotation in range(min(count, games-len(plans))):
            seats = instances[rotation:]+instances[:rotation]
            plans.append({'game_id': digest([identity, len(plans)]),
                'entrant_ids': [key for _, key in seats], 'instance_ids': [i for i, _ in seats],
                'controller_seeds': [seeds[i] for i, _ in seats], 'map_seed': map_seed,
                'map': map_config, 'rules_id': store.manifest['rules_id'], 'human': False,
                'evaluation_block': block})
        block += 1
    return plans


def summarize(store, candidate, job_id):
    from .league.rating import ranks
    with store.connect() as db:
        rows = list(db.execute('SELECT status,result FROM games WHERE job_id=? ORDER BY rowid', (job_id,)))
    groups, errors = {}, []
    entrants = store.entrants()
    total = {'calls': 0, 'total_ns': 0, 'max_ns': 0, 'timeouts': 0, 'errors': 0}
    candidate_violations = 0
    for row in rows:
        if not row['result']:
            continue
        result = json.loads(row['result'])
        for record in result.get('decision_timing', []):
            if record['entrant_id'] != candidate:
                continue
            for key in ('calls', 'total_ns', 'timeouts', 'errors'):
                total[key] += record[key]
            total['max_ns'] = max(total['max_ns'], record['max_ns'])
        if row['status'] == 'failed':
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
    return {'ok': job['status'] == 'complete' and not errors and not total['timeouts'] and not total['errors'],
            'job': job, 'completed_games': job['done'], 'by_players': by_players,
            'macro_pair_score': sum(values)/len(values) if values else None,
            'timing': {**total, 'limit_ns': DECISION_NS}, 'violations': errors,
            'candidate_violations': candidate_violations,
            'league_directory': str(store.directory.resolve())}


def evaluate(directory, candidate, *, opponents=None, games=128, seed=42, workers=1,
             players=None, label=None):
    root, _, official = session(directory)
    integer('games', games, 1)
    integer('seed', seed)
    integer('workers', workers, 1)
    if workers != 1:
        raise ValueError('C++ research matches run serially; workers must be 1')
    spec, entrant = registered(root, candidate)
    opponents = DEFAULT_OPPONENTS if opponents is None else opponents
    if not opponents:
        raise ValueError('at least one opponent is required (self is allowed)')
    configs = [_opponent(root, selector, spec) for selector in opponents]
    prepared = [prepare_entrant(item) for item in configs]
    rules = official.config if players is None else replace(official.config, player_counts=tuple(players))
    if not set(rules.player_counts) <= set(official.config.player_counts):
        raise ValueError('player counts must be enabled in the official league')
    descriptor = {'candidate': candidate, 'entrant_id': entrant['entrant_id'],
                  'opponents': opponents, 'opponent_ids': [e['entrant_id'] for e in prepared],
                  'games': games, 'seed': seed, 'config': asdict(rules)}
    identity = digest(descriptor)
    label = _name(label or f'eval-{identity[:16]}')
    directory = root/'evaluations'/label
    with _lock(root):
        if directory.exists():
            saved = json.loads((directory/'evaluation.json').read_text())
            if canonical(saved['request']) != canonical(descriptor):
                raise ValueError('evaluation label already belongs to a different request')
            store = LeagueStore(directory/'league')
            job_id = saved['job_id']
        else:
            directory.mkdir()
            store = LeagueStore.create(directory/'league', rules)
            for item in configs:
                store.register(item)
            _, job_id = store.add(spec, games=games)
            plans = _plans(store, entrant['entrant_id'], descriptor['opponent_ids'], games, seed, identity)
            store.insert_plans(plans, job_id)
            write_json(directory/'evaluation.json', {'request': descriptor, 'job_id': job_id})
    with process_lock(store, '.evaluation.lock'):
        started = time.monotonic()
        error = None
        try:
            run_jobs(store, workers=workers, job_id=job_id)
        except Exception as exc:
            error = str(exc)
        report = {**summarize(store, entrant['entrant_id'], job_id), 'request': descriptor,
                  'workers': workers, 'last_run_seconds': time.monotonic()-started,
                  'report': str(directory/'report.json')}
        if error:
            report['ok'], report['execution_error'] = False, error
        write_json(directory/'report.json', report)
    return report


def publish(directory, candidate, *, workers=1):
    integer('workers', workers, 1)
    if workers != 1:
        raise ValueError('C++ research matches run serially; workers must be 1')
    root, _, _ = session(directory)
    _name(candidate)
    # Keep preflight, execution and the final report in one publication attempt.
    # A simultaneous caller must not overwrite a successful report with its
    # worker-lock error. Distinct candidates still share the league worker lock.
    with process_lock(SimpleNamespace(directory=root), f'.publish-{candidate}.lock'):
        return _publish(root, candidate, workers=workers)


def _publish(directory, candidate, *, workers):
    started = time.monotonic()
    root, _, official = session(directory)
    spec, entrant = registered(root, candidate)
    # A recorded violation by this candidate cannot be hidden by a later lucky run.
    for path in (root/'evaluations').glob('*/report.json'):
        report = json.loads(path.read_text())
        violation = any(v.get('version') == entrant['version'] for v in report['violations'])
        if violation or (report['request']['entrant_id'] == entrant['entrant_id'] and (
                report['candidate_violations'] or report['timing']['timeouts'] or report['timing']['errors'])):
            raise ValueError('candidate has a recorded violation; publish a corrected new version')
    with _lock(root):
        publications = json.loads((root/'publications.json').read_text())
        existing = next((p for p in publications if p['candidate'] == candidate), None)
        if existing is None:
            if len(publications) >= PUBLICATION_LIMIT:
                raise ValueError('research session has used all 3 checkpoint publication slots')
            if any(p['version'] == entrant['version'] for p in publications):
                raise ValueError('identical policy version has already been published')
    if existing is None:
        probe = evaluate(root, candidate, games=8, seed=8675309, workers=workers,
                         label=f'preflight-{candidate}')
        if not probe['ok']:
            raise ValueError(f'checkpoint preflight failed; inspect {probe["report"]}')
    with _lock(root):
        publications = json.loads((root/'publications.json').read_text())
        existing = next((p for p in publications if p['candidate'] == candidate), None)
        if existing is None:
            if len(publications) >= PUBLICATION_LIMIT:
                raise ValueError('research session has used all 3 checkpoint publication slots')
            if any(p['version'] == entrant['version'] for p in publications):
                raise ValueError('identical policy version has already been published')
            existing = {'candidate': candidate, 'version': entrant['version'],
                        'entrant_id': entrant['entrant_id'], 'created': time.time(),
                        'slot': len(publications)+1}
            publications.append(existing)
            # Reserve the slot before touching the official DB; restart resumes
            # this same candidate through the idempotent LeagueStore.add.
            write_json(root/'publications.json', publications)
        # The formal league owns its executable and source archive. Removing a
        # research workspace/session later must not break a published opponent.
        destination = official.directory/'policies'/entrant['entrant_id']
        destination.mkdir(parents=True, exist_ok=True)
        bundle = destination/'policy.zip'
        if not bundle.exists():
            temporary = destination/'policy.zip.tmp'
            shutil.copyfile(spec['bundle'], temporary)
            temporary.replace(bundle)
        archived_spec = {**spec, 'bundle': str(bundle.resolve())}
        if prepare_entrant(archived_spec)['entrant_id'] != entrant['entrant_id']:
            raise ValueError('published policy archive changed')
        write_json(destination/'entrant.json', archived_spec)
        _, job_id = official.add(archived_spec, games=PUBLICATION_GAMES)
        existing['job_id'] = job_id
        write_json(root/'publications.json', publications)
    error = None
    try:
        run_jobs(official, workers=workers, job_id=job_id)
    except Exception as exc:
        error = str(exc)
    report = {**summarize(official, entrant['entrant_id'], job_id),
              'candidate': candidate, 'slot': existing['slot'], 'version': entrant['version'],
              'leaderboard': official.leaderboard(), 'publication_games': PUBLICATION_GAMES,
              'workers': workers, 'last_run_seconds': time.monotonic()-started}
    if error:
        report['ok'], report['execution_error'] = False, error
    write_json(root/'strategies'/candidate/'publication.json', report)
    return report


def status(directory):
    root, config, official = session(directory)
    jobs = {j['id']: j for j in official.jobs()}
    publications = json.loads((root/'publications.json').read_text())
    return {'session': config,
            'candidates': [p.name for p in sorted((root/'strategies').iterdir()) if (p/'entrant.json').exists()],
            'evaluations': [str(p) for p in sorted((root/'evaluations').glob('*/report.json'))],
            'publications': [{**p, 'job': jobs.get(p.get('job_id'))} for p in publications],
            'remaining_publications': PUBLICATION_LIMIT-len(publications)}


def add_cli(commands):
    parser = commands.add_parser('research', help='Arena-managed C++ research and checkpoints')
    actions = parser.add_subparsers(dest='research_action', required=True)
    init = actions.add_parser('init', help='create a research session and optional C++ workspace')
    init.add_argument('directory', type=Path)
    init.add_argument('--league', type=Path, required=True)
    init.add_argument('--workspace', type=Path)
    reg = actions.add_parser('register', help='snapshot and compile an immutable C++ candidate')
    reg.add_argument('directory', type=Path)
    reg.add_argument('--source', type=Path, required=True)
    reg.add_argument('--name', required=True)
    reg.add_argument('--params', type=Path, help='optional JSON parameter object')
    test = actions.add_parser('evaluate', help='private black-box games; does not touch official ratings')
    test.add_argument('directory', type=Path)
    test.add_argument('--candidate', required=True)
    test.add_argument('--opponents', nargs='+', default=None, help='baseline:KIND, self, or registered candidate name')
    test.add_argument('--games', type=int, default=128)
    test.add_argument('--seed', type=int, default=42)
    test.add_argument('--players', type=int, nargs='+')
    test.add_argument('--workers', type=int, default=1)
    test.add_argument('--label', help='immutable experiment name; repeat command to resume')
    pub = actions.add_parser('publish', help='preflight and run one complete 1000-game checkpoint')
    pub.add_argument('directory', type=Path)
    pub.add_argument('--candidate', required=True)
    pub.add_argument('--workers', type=int, default=1)
    stat = actions.add_parser('status', help='list candidates, experiments and checkpoint progress')
    stat.add_argument('directory', type=Path)


def command(args):
    action = args.research_action
    if action == 'init':
        report = initialize(args.directory, args.league, args.workspace)
    elif action == 'register':
        parameters = json.loads(args.params.read_text()) if args.params else None
        report = register(args.directory, args.source, args.name, parameters)
    elif action == 'evaluate':
        report = evaluate(args.directory, args.candidate, opponents=args.opponents, games=args.games,
                          seed=args.seed, workers=args.workers, players=args.players, label=args.label)
    elif action == 'publish':
        report = publish(args.directory, args.candidate, workers=args.workers)
    else:
        report = status(args.directory)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report.get('ok', True) else 1
