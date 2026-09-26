"""Cross-platform command line. GUI imports are lazy."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .config import MapConfig
from .mapgen import generate_map
from .recording import TraceWriter, replay, state_digest
from .runner import Runner, ControllerError


def _parser():
    parser = argparse.ArgumentParser(prog="generals-env", description="CPU Generals environment: play, watch, simulate, replay")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("play", "watch", "simulate"):
        p = sub.add_parser(name)
        p.add_argument("--players", type=int, default=4)
        p.add_argument("--size", type=int, choices=(15, 25, 35), default=25)
        p.add_argument("--width", type=int)
        p.add_argument("--height", type=int)
        p.add_argument("--mountain-density", type=float, default=.2)
        p.add_argument("--city-density", type=float, default=.1)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--visibility", choices=("local", "full"), default="local")
        p.add_argument("--ais", help="comma-separated aggressive,expansion,defensive,random (one per AI seat)")
        if name == "simulate":
            p.add_argument("--games", type=int, default=1)
            p.add_argument("--max-ticks", type=int, default=20000)
            p.add_argument("--record", type=Path, help="JSONL path; requires --games 1 and a new file")
            p.add_argument("--output", type=Path, help="write JSON summary to a new file")
            p.add_argument("--debug", action="store_true")
        else:
            p.add_argument("--speed", type=float, choices=(1, 2, 5, 10), default=1)
            if name == "play": p.add_argument("--human-seat", type=int, default=0)
    p = sub.add_parser("replay", help="verify a JSONL action recording without calling AI")
    p.add_argument("path", type=Path)
    return parser


def _kinds(args):
    human = args.human_seat if args.command == "play" else None
    if human is not None and not 0 <= human < args.players:
        raise ValueError("human seat outside player range")
    count = args.players - (human is not None)
    cycle = ("aggressive", "expansion", "defensive", "random")
    ai = args.ais.split(",") if args.ais else [cycle[i % 4] for i in range(count)]
    if len(ai) != count or any(kind not in cycle for kind in ai):
        raise ValueError(f"--ais requires exactly {count} valid AI names")
    result = list(ai)
    if human is not None: result.insert(human, "human")
    return result


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            state = replay(args.path)
            print(json.dumps({"verified": True, "ticks": state.tick, "winner_id": state.winner_id,
                              "digest": state_digest(state)}, indent=2))
            return 0
        if args.seed < 0:
            raise ValueError("seed must be nonnegative")
        config = MapConfig(args.width if args.width is not None else args.size,
                           args.height if args.height is not None else args.size,
                           args.players, args.mountain_density, args.city_density)
        kinds = _kinds(args)
        if args.command in ("play", "watch"):
            from .gui.app import launch
            launch(config, seed=args.seed, mode=args.command,
                   human_seat=getattr(args, "human_seat", 0), visibility=args.visibility,
                   kinds=kinds, speed=args.speed)
            return 0
        if args.games < 1 or args.max_ticks < 1:
            raise ValueError("games and max-ticks must be positive")
        if args.record and args.games != 1:
            raise ValueError("recording requires --games 1")
        for path in (args.record, args.output):
            if path is not None and path.exists():
                raise FileExistsError(f"output already exists: {path}")
        if args.record and args.output and args.record.resolve() == args.output.resolve():
            raise ValueError("record and output must use different files")
        from .controllers import make_controllers
        summaries = []
        started = perf_counter()
        for game in range(args.games):
            seed = args.seed if game == 0 else int(np.random.SeedSequence([args.seed, game]).generate_state(1, dtype=np.uint64)[0])
            state = generate_map(config, seed)
            writer = TraceWriter(args.record, state, {"seed": seed, "map": asdict(config),
                                "controllers": kinds, "visibility": args.visibility}) if args.record else None
            try:
                runner = Runner(state, make_controllers(kinds, seed), seed=seed,
                                visibility=args.visibility, max_ticks=args.max_ticks,
                                debug=args.debug, recorder=writer)
                try:
                    runner.run()
                except ControllerError:
                    pass
                summaries.append(runner.summary())
            finally:
                if writer: writer.close()
        elapsed = perf_counter() - started
        ticks = sum(row["ticks"] for row in summaries)
        wins = [sum(row["winner_id"] == pid for row in summaries) for pid in range(args.players)]
        result = {"games": args.games, "map": asdict(config), "controllers": kinds,
                  "visibility": args.visibility, "wins_by_seat": wins,
                  "terminated": sum(row["terminated"] for row in summaries),
                  "truncated": sum(row["truncated"] for row in summaries),
                  "errors": sum(row["error"] is not None for row in summaries),
                  "total_ticks": ticks, "elapsed_seconds": elapsed,
                  "ticks_per_second": ticks / elapsed if elapsed else 0,
                  "runs": summaries}
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                file.write(rendered + "\n")
        print(rendered)
        return 1 if result["errors"] else 0
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except Exception as exc:
        # Tk may be unavailable on an otherwise fully functional headless machine.
        if type(exc).__name__ == "TclError":
            parser.exit(2, f"GUI display unavailable: {exc}. Use simulate on a headless machine.\n")
        raise
