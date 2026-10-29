"""Commit C++ / RL policies, run private games, and manage the local Arena."""
import argparse
import json
from pathlib import Path
import sys
import time

from generals_arena.competitions.rating import MODES
from generals_arena.competitions.config import DEFAULT_LEAGUE


def league_command(args):
    from generals_arena.competitions.config import LeagueConfig
    from generals_arena.competitions.jobs import run_jobs
    from generals_arena.competitions.store import CompetitionStore
    if args.action == "init":
        config = json.loads(args.config.read_text()) if args.config else {}
        store = CompetitionStore.create(args.directory, LeagueConfig(**config))
        labels = {'1v1': '1v1 GoR relative to WAIT', 'ffa': 'FFA Codeforces mean 1000'}
        print(f"League initialized: {store.directory} ({'; '.join(labels[mode] for mode in store.rating_modes())})")
        return
    store = CompetitionStore(args.directory)
    if args.action == "resize":
        ranges = store.resize_maps(args.min_side, args.max_side, args.side_cap)
        print(f"Map ranges updated for future games: {json.dumps(ranges)}. Existing results and ratings retained.")
        return
    if args.action == "add":
        spec = json.loads(args.entrant.read_text(encoding="utf-8"))
        entrant, job = store.add(spec, base_dir=args.entrant.resolve().parent, games=args.games)
        print(f"Registered: {entrant['name']} / job={job}")
        if args.queue_only:
            return
    if args.action in ("add", "run"):
        count = run_jobs(store, limit=getattr(args, "limit", None), retry=getattr(args, "retry", False),
                         workers=getattr(args, "workers", 1),
                         progress=lambda done: print(f"League: {done} new games completed", file=sys.stderr))
        print(f"Completed {count} new games. Jobs: {json.dumps(store.jobs(), ensure_ascii=False)}")
    elif args.action == "rebuild":
        store.rebuild()
        from generals_arena.competitions.submission import refresh_publications
        refresh_publications(store)
        print("Ratings rebuilt from completed matches.")
    else:
        reports = {args.mode: store.leaderboard(args.mode)} if args.mode else store.leaderboards()
        if args.json:
            print(json.dumps(reports, ensure_ascii=False, indent=2))
        else:
            for mode, report in reports.items():
                reference = (next(row['name'] for row in report['ranking'] if row['anchor']) + ' = 0'
                             if mode == '1v1' else 'mean 1000')
                print(f"{mode} / {report['rating_model']} / {report['completed_games']} games / {reference}")
                for row in report["ranking"]:
                    print(f"{row['name']:<28} {row['elo']:>8.1f}  {row['games']:>6} games")


def match_mode_command(args):
    if args.command == "tournament":
        from generals import MapConfig
        from generals_arena.matches.config import MatchConfig
        from generals_arena.competitions.tournament import tournament
        config = json.loads(args.config.read_text()) if args.config else {}
        if "map" in config:
            config["map"] = MapConfig(**config["map"])
        roster = json.loads(args.entrants.read_text())
        report = tournament(roster, MatchConfig(**config), args.output,
                            maps_per_group=args.maps_per_group, seed=args.seed,
                            paired_seats=not args.single_seat,
                            base_dir=args.entrants.resolve().parent,
                            fit_ratings=args.calibrate_profile is not None,
                            **_execution_options(args))
        if args.calibrate_profile is not None:
            from generals_arena.competitions.evaluation import make_calibrated_profile
            from generals_arena.util import write_json
            profile = make_calibrated_profile(report, roster, maps_per_opponent=args.evaluation_maps,
                                               base_dir=args.entrants.resolve().parent)
            args.calibrate_profile.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.calibrate_profile, profile)
        print(json.dumps({key: report[key] for key in ("run_id", "rating_mode", "rating_model", "games", "ranking")}, indent=2))
    else:
        from generals_arena.competitions.evaluation import evaluate
        source_path = args.candidates or args.entrant
        candidates = json.loads(args.candidates.read_text()) if args.candidates else [
            args.bundle if args.bundle else json.loads(args.entrant.read_text())]
        report = evaluate(candidates, args.profile, args.output,
                          base_dir=source_path.resolve().parent if source_path else Path("."),
                          **_execution_options(args))
        print(json.dumps({key: report[key] for key in ("profile_id", "rating_mode", "rating_model", "candidates", "games")}, indent=2))
    return 0


def _execution_options(args):
    return {"backend": args.backend, "device": args.device, "batch_size": args.batch_size,
            "cpu_threads": args.cpu_threads, "record_replays": args.record_replays}


def _add_execution_options(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--device", default="cpu", help="policy inference device; gpu backend requires cuda:N")
    parser.add_argument("--batch-size", type=int, default=128, help="zero runs a whole compatible group")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--record-replays", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_submission_commands(commands)
    league = commands.add_parser("league", help="persistent random-map league (baseline, custom, human)")
    actions = league.add_subparsers(dest="action", required=True)
    init = actions.add_parser("init", help="create independent GoR 1v1 and Codeforces FFA rating pools")
    init.add_argument("directory", type=Path)
    init.add_argument("--config", type=Path, help="optional league distribution JSON")
    add = actions.add_parser("add", help="register a new version and schedule a bounded evaluation job")
    add.add_argument("directory", type=Path)
    add.add_argument("--entrant", type=Path, required=True)
    add.add_argument("--games", type=int, default=1000)
    add.add_argument("--queue-only", action="store_true", help="let serve/league run execute the job later")
    worker = actions.add_parser("run", help="resume pending league games")
    worker.add_argument("directory", type=Path)
    worker.add_argument("--limit", type=int)
    worker.add_argument("--retry", action="store_true", help="retry failed jobs after fixing their cause")
    worker.add_argument("--workers", type=int, default=1, help="parallel simulations within each persistent schedule window")
    ranking = actions.add_parser("leaderboard")
    ranking.add_argument("directory", type=Path)
    ranking.add_argument("--json", action="store_true")
    ranking.add_argument("--mode", choices=MODES, help="one rating pool; default shows every enabled pool")
    refresh = actions.add_parser("rebuild")
    refresh.add_argument("directory", type=Path)
    resize = actions.add_parser("resize", help="revise future map sizes, preserving accounts and recorded matches (stop server/workers first)")
    resize.add_argument("directory", type=Path)
    resize.add_argument("--min-side", type=int, required=True)
    resize.add_argument("--max-side", type=int, required=True)
    resize.add_argument("--side-cap", type=int, required=True)
    web = commands.add_parser("serve", help="serve rankings, verified replays, human games, and the job queue")
    web.add_argument("directory", type=Path)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8086)
    web.add_argument("--no-worker", action="store_true", help="view/play while a separate league run handles evaluation")
    tournament = commands.add_parser("tournament", help="fixed round-robin schedule mode")
    tournament.add_argument("--entrants", type=Path, required=True, help="JSON list of ordinary entrant specs")
    tournament.add_argument("--config", type=Path, help="MatchConfig JSON: map and max_ticks")
    tournament.add_argument("--maps-per-group", type=int, default=1)
    tournament.add_argument("--seed", type=int, default=42)
    tournament.add_argument("--single-seat", action="store_true", help="skip cyclic seat rotations")
    tournament.add_argument('--calibrate-profile', type=Path, help='fit GoR jointly and save a fixed-reference evaluation profile')
    tournament.add_argument('--evaluation-maps', type=int, default=100, help='maps per opponent in the saved evaluation profile')
    _add_execution_options(tournament)
    evaluation = commands.add_parser("evaluate", help="fixed-opponent training-evaluation schedule mode")
    source = evaluation.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--entrant", type=Path, help="ordinary Arena entrant JSON")
    source.add_argument('--candidates', type=Path, help='JSON list of candidate bundles or entrant specs; include their round robin')
    evaluation.add_argument("--profile", type=Path, required=True)
    _add_execution_options(evaluation)
    args = parser.parse_args(argv)
    try:
        if args.command in ('commit', 'run', 'status'):
            return submission_command(args)
        if args.command in ("tournament", "evaluate"):
            return match_mode_command(args)
        if args.command == "league":
            league_command(args)
            return 0
        if args.command == "serve":
            from generals_arena.web.server import serve
            serve(args.directory, args.host, args.port, run_worker=not args.no_worker)
            return 0
    except (ValueError, TypeError, KeyError, ImportError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; repeat the command to resume recorded progress.\n")
    return 0


def add_submission_commands(commands):
    submit = commands.add_parser('commit', help='snapshot a native source or policy package and run its 1000-game evaluation')
    submit.add_argument('source', type=Path, help='.cpp file, native source directory, or exported policy bundle')
    submit.add_argument('--name', help='immutable display alias shown on the leaderboard')
    submit.add_argument('--params', type=Path, help='JSON object of fixed policy parameters')
    submit.add_argument('--device', default='cpu', help='packaged policy device, e.g. cpu or cuda:0 (env remains CPU)')
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
    play.add_argument('--device', help='override packaged policy device; otherwise use registered device')
    play.add_argument('--output', type=Path, help='optional report and replay database export directory')
    play.add_argument('--league', type=Path, default=DEFAULT_LEAGUE)
    play.add_argument('--workers', type=int, default=1, help='parallel games (default: 1)')
    inspect = commands.add_parser('status', help='show managed submissions, jobs, and run report paths')
    inspect.add_argument('--league', type=Path, default=DEFAULT_LEAGUE)


def submission_command(args):
    from generals_arena.competitions.submission import commit, status
    from generals_arena.competitions.private import run
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
                        progress=progress, workers=args.workers, device=args.device)
    elif args.command == 'run':
        report = run(args.player, league=args.league, size=args.size, games=args.games, seed=args.seed,
                     output=args.output, parameters=parameters, max_ticks=args.max_ticks,
                     progress=progress, workers=args.workers, device=args.device)
    else:
        report = status(args.league)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report.get('ok', True) else 1
