"""Run a C++ policy online without registering or scoring a local league job."""
import json
import os
from pathlib import Path
import tempfile


def add_cli(commands):
    setup = commands.add_parser('online-account', help='create and register a saved Bot account without joining a game')
    setup.add_argument('--account', required=True, help='local account profile, e.g. razor')
    setup.add_argument('--username', required=True, help='public name, beginning with [Bot]')
    setup.add_argument('--account-dir', type=Path, help='private credential directory; defaults to config/secret (run from the project root)')
    setup.add_argument('--server', default='https://botws.generals.io')
    setup.add_argument('--timeout', type=float, default=15)
    parser = commands.add_parser('online', help='play one official-server game using a C++ policy')
    parser.add_argument('policy', help='C++ file/directory, policy.zip, @alias, or policy/<hash>')
    parser.add_argument('--params', type=Path, help='fixed JSON parameters for source policies')
    parser.add_argument('--league-directory', type=Path, default=Path('outputs/arena/native_1000'),
                        help='read-only alias lookup; online play never updates its ratings')
    parser.add_argument('--queue', choices=('custom', '1v1', 'ffa'), default='custom')
    parser.add_argument('--room', help='custom room id (required for the default custom queue)')
    parser.add_argument('--force-start', action='store_true', help='force start a custom room')
    parser.add_argument('--username', help='register a NEW [Bot] name once; omit on subsequent runs')
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument('--user-id-env', default='GENERALS_BOT_USER_ID', help='credential environment variable name')
    identity.add_argument('--account', help='saved account created by online-account')
    parser.add_argument('--account-dir', type=Path, help='private saved-account directory; defaults to config/secret (run from the project root)')
    parser.add_argument('--server', help='defaults to the saved account server or https://botws.generals.io')
    parser.add_argument('--seed', type=int, default=42, help='private controller RNG seed; never a map seed')
    parser.add_argument('--start-timeout', type=float, default=120)
    parser.add_argument('--update-timeout', type=float, default=15)
    parser.add_argument('--tick-offset', type=int, default=0, help='explicit server half-turn calibration, default unchanged')
    parser.add_argument('--record', type=Path, help='new online JSONL path; credentials are excluded')
    parser.add_argument('--max-decisions', type=int, help='optional local stop limit (leaves the game)')
    parser.add_argument('--games', type=int, default=1, help='number of completed games to collect sequentially')
    parser.add_argument('--output', type=Path, help='new directory for series results and per-game JSONL records')


def _bundle(args, scratch):
    from .native import build_bundle
    source = args.policy
    if source.startswith(('@', 'policy/')):
        if args.params:
            raise ValueError('saved policies retain their original parameters')
        from .league.store import LeagueStore
        from .research import _selector
        # For these two selectors _selector only reads existing artifacts.
        spec = _selector(LeagueStore(args.league_directory), source)
        return spec['bundle']
    path = Path(source)
    if path.suffix == '.zip':
        if args.params:
            raise ValueError('saved bundles retain their original parameters')
        return path
    parameters = json.loads(args.params.read_text()) if args.params else None
    destination = Path(scratch) / 'policy.zip'
    build_bundle(path, destination, parameters)
    return destination


def command(args):
    if args.command == 'online-account':
        from .online_accounts import initialize_account
        result = initialize_account(args.account, args.username, directory=args.account_dir,
                                    server=args.server, timeout=args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    from generals import make_env
    from generals.runtime.match import run_match
    from .native import NativeController
    from generals.backends.online.account import DEFAULT_SERVER, validate_server
    if args.games < 1:
        raise ValueError('--games must be positive')
    if args.games > 1 and args.output is None:
        raise ValueError('--games requires --output for durable results')
    if args.output is not None and (args.record is not None or args.max_decisions is not None or args.username is not None):
        raise ValueError('--output cannot be combined with --record, --max-decisions, or --username')
    if args.account:
        from .online_accounts import load_account
        account = load_account(args.account, args.account_dir)
        if args.username is not None:
            raise ValueError('saved accounts are already registered; omit --username')
        if args.server is not None and validate_server(args.server) != account['server']:
            raise ValueError('saved account is bound to a different server')
        user_id, server = account['user_id'], account['server']
    else:
        if args.account_dir is not None:
            raise ValueError('--account-dir requires --account')
        user_id, server = os.environ.get(args.user_id_env), args.server or DEFAULT_SERVER
        if not user_id:
            raise ValueError(f'set {args.user_id_env} or select a saved --account')
    options = dict(mode='online', user_id=user_id, room=args.room, queue=args.queue,
        username=args.username, server=server, force_start=args.force_start,
        start_timeout=args.start_timeout, update_timeout=args.update_timeout,
        tick_offset=args.tick_offset)
    if args.output is not None:
        from .online_series import run_series
        with tempfile.TemporaryDirectory(prefix='generals-online-') as scratch:
            report = run_series(_bundle(args, scratch), options, games=args.games,
                                output=args.output, seed=args.seed, policy=args.policy)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['completed_games'] == args.games else 1
    backend = make_env(**options, record=args.record)
    try:
        with tempfile.TemporaryDirectory(prefix='generals-online-') as scratch:
            controller = NativeController(bundle=_bundle(args, scratch))
            result = run_match(backend, controller, seed=args.seed, max_decisions=args.max_decisions)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['done'] else 1
    finally:
        backend.close()
