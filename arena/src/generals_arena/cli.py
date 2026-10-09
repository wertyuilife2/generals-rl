"""Run a versioned tournament or rebuild its Elo leaderboard."""
import argparse
import json
from pathlib import Path
import sys

from .tournament import rebuild, run_tournament


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
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
