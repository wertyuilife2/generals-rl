"""Run a versioned tournament or rebuild its Elo leaderboard."""
import argparse
import json
from pathlib import Path
import sys

from .tournament import rebuild, run_tournament


def league_command(args):
    from .league.config import LeagueConfig
    from .league.service import run_jobs
    from .league.store import LeagueStore
    if args.action == "init":
        config = json.loads(args.config.read_text()) if args.config else {}
        store = LeagueStore.create(args.directory, LeagueConfig(**config))
        print(f"League initialized: {store.directory} (random = 0 Elo)")
        return
    store = LeagueStore(args.directory)
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
        print("Ratings rebuilt from completed matches.")
    else:
        report = store.leaderboard()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"Unified rating / {report['completed_games']} games / random = 0")
            for row in report["ranking"]:
                print(f"{row['name']:<28} {row['elo']:>8.1f}  {row['games']:>6} games")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    league = commands.add_parser("league", help="persistent random-map league (baseline, custom, human)")
    actions = league.add_subparsers(dest="action", required=True)
    init = actions.add_parser("init", help="create a league with a zero-rated random anchor")
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
    refresh = actions.add_parser("rebuild")
    refresh.add_argument("directory", type=Path)
    web = commands.add_parser("serve", help="serve rankings, verified replays, human games, and the job queue")
    web.add_argument("directory", type=Path)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--no-worker", action="store_true", help="view/play while a separate league run handles evaluation")
    run = commands.add_parser("run", help="run, resume, or extend a tournament")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    for command in ("leaderboard", "rebuild"):
        view = commands.add_parser(command, help="recompute Elo from the append-only match log")
        view.add_argument("directory", type=Path)
        view.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "league":
            league_command(args)
            return 0
        if args.command == "serve":
            from .server import serve
            serve(args.directory, args.host, args.port, run_worker=not args.no_worker)
            return 0
        if args.command == "run":
            config = json.loads(args.config.read_text(encoding="utf-8"))
            report = run_tournament(config, args.output, base_dir=args.config.resolve().parent,
                                    resume=args.resume,
                                    progress=lambda done, total: print(f"Arena: {done}/{total}", file=sys.stderr))
        else:
            report = rebuild(args.directory)
    except (ValueError, TypeError, KeyError, ImportError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(f"Games: {report['completed_games']}/{report['scheduled_games']}")
        print(f"{'Name':<28} {'Elo':>8} {'Games':>7} {'W':>6} {'L':>6} {'D':>6}")
        for row in report["ranking"]:
            print(f"{row['name']:<28} {row['elo']:>8.1f} {row['games']:>7} "
                  f"{row['wins']:>6} {row['losses']:>6} {row['draws']:>6}")
    return 0
