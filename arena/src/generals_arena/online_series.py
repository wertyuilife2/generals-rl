"""Sequential online sampling with durable outcomes, including failed attempts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import monotonic, sleep

from generals import make_env
from generals.runtime.match import run_match
from .native import NativeController, file_digest


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _participants(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('event') == 'game_start':
                    data = row['data']
                    return {'player_id': data.get('playerIndex'),
                            'usernames': data.get('usernames'), 'replay_id': data.get('replay_id')}
    return {}


def run_series(bundle, options, *, games, output, seed=42, policy=None):
    """Stop on protocol/strategy errors; bound new attempts after network failures.

    A new directory prevents accidental overwrites. Queue failures never count
    as completed games. Every failed attempt stays in report.json and its log.
    """
    if isinstance(games, bool) or not isinstance(games, int) or games < 1:
        raise ValueError('games must be positive')
    if options.get('username') is not None:
        raise ValueError('register the account before running a series')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'policy': policy, 'bundle_sha256': file_digest(bundle),
              'requested_games': games, 'completed_games': 0, 'attempted_games': 0,
              'wins': 0, 'losses': 0, 'failures': 0, 'status': 'running',
              'server': options.get('server'), 'queue': options.get('queue'),
              'seed': seed, 'started_at': _now(), 'games': [],
              'report': str((output / 'report.json').resolve())}
    path = output / 'report.json'
    _write(path, report)
    consecutive_network_failures = 0
    try:
        while report['completed_games'] < games:
            index = report['attempted_games'] + 1
            record = output / f'game-{index:03d}.jsonl'
            row = {'attempt': index, 'seed': seed + index - 1,
                   'started_at': _now(), 'record': str(record.resolve())}
            report['active_game'] = row
            _write(path, report)
            controller, backend, retry = None, None, False
            before = monotonic()
            try:
                controller = NativeController(bundle=bundle)
                backend = make_env(**options, record=record)
                result = run_match(backend, controller, seed=row['seed'])
                row['result'] = result
                if result['done'] and result['reason'] in ('won', 'lost'):
                    report['completed_games'] += 1
                    report['wins' if result['reason'] == 'won' else 'losses'] += 1
                    consecutive_network_failures = 0
                else:
                    row['error'] = 'game ended without an authoritative won/lost result'
                    report['failures'] += 1
                    report['status'] = 'stopped'
            except BaseException as exc:
                if not isinstance(exc, (Exception, KeyboardInterrupt)):
                    raise
                secret = options.get('user_id', '')
                message = str(exc)
                row['error'] = message.replace(secret, '[REDACTED]') if secret else message
                row['error_type'] = type(exc).__name__
                report['failures'] += 1
                report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'stopped'
                if isinstance(exc, (TimeoutError, ConnectionError)):
                    row['failure_phase'] = ('in_game' if backend is not None and
                        backend.decoder.player_id is not None else 'before_game')
                    consecutive_network_failures += 1
                    retry = consecutive_network_failures < 3
                    if retry:
                        report['status'] = 'running'
            finally:
                if backend is not None:
                    backend.close()
                if controller is not None:
                    row['decision_timing'] = controller.decision_metrics
                    controller.close()
                row.update(_participants(record))
                row['elapsed_seconds'] = monotonic() - before
                row['finished_at'] = _now()
                report['attempted_games'] += 1
                report['games'].append(row)
                report.pop('active_game', None)
                _write(path, report)
                print(json.dumps({key: report[key] for key in (
                    'attempted_games', 'completed_games', 'wins', 'losses', 'failures', 'status')}) ,
                    file=sys.stderr, flush=True)
            if report['status'] != 'running':
                break
            if retry:
                sleep(30)
        if report['completed_games'] == games:
            report['status'] = 'complete'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
    finally:
        report['finished_at'] = _now()
        _write(path, report)
    return report
