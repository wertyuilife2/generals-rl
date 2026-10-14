"""Commit C++ policies, run private games, and manage the local Arena."""
import argparse
import json
from pathlib import Path
import sys


def league_command(args):
    from .league.config import LeagueConfig
    from .league.service import run_jobs
    from .league.store import LeagueStore
    if args.action == "init":
        config = json.loads(args.config.read_text()) if args.config else {}
        store = LeagueStore.create(args.directory, LeagueConfig(**config))
        print(f"League initialized: {store.directory} (WAIT = 0 Elo)")
        return
    store = LeagueStore(args.directory)
    if args.action == "resize":
        ranges = store.resize_maps(args.min_side, args.max_side, args.side_cap)
        print(f"Map ranges updated for future games: {json.dumps(ranges)}. Existing results and ratings retained.")
        return
    if args.action in ("retire", "restore"):
        result = store.set_retired(args.entrant, args.action == "retire", reason=args.reason)
        # Research leagues keep a static board beside the database. Refresh it
        # when present; publications and replay data remain untouched.
        if (store.directory / "commits").is_dir():
            from .research import refresh_publications
            refresh_publications(store)
        print(json.dumps(result, ensure_ascii=False))
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
        store.rebuild(upgrade=args.current_policy)
        print("Ratings rebuilt from completed matches.")
    else:
        report = store.leaderboard(include_retired=args.all)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            reference = next(row['name'] for row in report['ranking'] if row['anchor'])
            print(f"Unified rating / {report['completed_games']} games / {reference} = 0")
            for row in report["ranking"]:
                print(f"{row['name']:<28} {row['elo']:>8.1f}  {row['games']:>6} games")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    from .research import add_cli
    add_cli(commands)
    league = commands.add_parser("league", help="persistent random-map league (baseline, custom, human)")
    actions = league.add_subparsers(dest="action", required=True)
    init = actions.add_parser("init", help="create a league with a zero-rated WAIT reference")
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
    ranking.add_argument("--all", action="store_true", help="include retired entrants")
    for action, help_text in (("retire", "hide an entrant and exclude it from future matchmaking"),
                              ("restore", "return a retired entrant to the active roster")):
        roster = actions.add_parser(action, help=help_text)
        roster.add_argument("directory", type=Path)
        roster.add_argument("entrant", help="entrant name, @alias, or immutable entrant ID")
        roster.add_argument("--reason", help="optional audit note")
    refresh = actions.add_parser("rebuild")
    refresh.add_argument("directory", type=Path)
    refresh.add_argument('--current-policy', action='store_true',
                         help='re-score all recorded outcomes with current rules; stop server/workers first')
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
    args = parser.parse_args(argv)
    try:
        if args.command in ('commit', 'run', 'status'):
            from .research import command
            return command(args)
        if args.command == "league":
            league_command(args)
            return 0
        if args.command == "serve":
            from .server import serve
            serve(args.directory, args.host, args.port, run_worker=not args.no_worker)
            return 0
    except (ValueError, TypeError, KeyError, ImportError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; repeat the command to resume recorded progress.\n")
    return 0
