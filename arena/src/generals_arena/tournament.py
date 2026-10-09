"""Run, resume, extend, and rebuild a fixed-rules rating pool."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path

from .config import ArenaRules
from .entrants import prepare_entrant
from .leaderboard import build_leaderboard
from .match import play_match
from .ratings.elo import EloConfig
from .registry import load_controller
from .results import append_match, read_matches, validate_matches
from .schedule import make_schedule, map_count
from .util import digest, write_json


@contextmanager
def _writer_lock(directory):
    lock = directory / ".arena.lock"
    try:
        stream = lock.open("x")
    except FileExistsError:
        raise RuntimeError("arena is locked; remove .arena.lock only after confirming no other writer is running") from None
    try:
        with stream:
            stream.write(str(os.getpid()) + "\n")
        yield
    finally:
        lock.unlink()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _schedule(config, entrants):
    rules = config["rules"]
    if rules.get("schema_version") != 1 or rules.get("rules_id") != digest(
            {key: value for key, value in rules.items() if key != "rules_id"}):
        raise ValueError("invalid rules manifest identity")
    names = set()
    for entrant in entrants:
        identity = {key: entrant[key] for key in ("name", "kind", "version", "factory", "kwargs", "implementation")}
        if entrant["name"] in names or entrant["entrant_id"] != digest(identity):
            raise ValueError("duplicate entrant name or invalid entrant identity")
        names.add(entrant["name"])
    return make_schedule([entry["entrant_id"] for entry in entrants],
                         rules_id=config["rules"]["rules_id"],
                         players=config["rules"]["map"]["players"],
                         maps_per_group=map_count(config), seed=config["rules"]["seed"])


def _report(directory, config, entrants, matches, schedule):
    validate_matches(matches, schedule, config["rules"])
    report = build_leaderboard(entrants, matches, elo=EloConfig(**config["elo"]),
                               rules_id=config["rules"]["rules_id"])
    report.update(scheduled_games=len(schedule), complete=len(matches) == len(schedule))
    write_json(directory / "leaderboard.json", report)
    return report


def rebuild(directory):
    """Recompute from recorded identities and results; no models are loaded."""
    directory = Path(directory).resolve()
    with _writer_lock(directory):
        config, entrants = _read(directory / "config.json"), _read(directory / "entrants.json")
        matches = read_matches(directory / "matches.jsonl")
        return _report(directory, config, entrants, matches, _schedule(config, entrants))


def run_tournament(config, output_dir, *, base_dir=Path("."), resume=False, progress=None):
    """Resume permits new names and more maps, preserving all existing versions.

    A controller/model is loaded once per entrant; reset owns all per-match
    memory and randomness. Matches run sequentially, and Elo follows log order.
    Controller failures abort the run and never become losses or draws.
    """
    if set(config) - {"rules", "maps_per_group", "maps_per_pair", "elo", "entrants"}:
        raise ValueError("unknown tournament configuration fields")
    rules = ArenaRules.from_dict(config.get("rules", {}))
    elo = EloConfig(**config.get("elo", {}))
    maps_per_group = map_count(config)
    entrants = [prepare_entrant(spec, base_dir=base_dir) for spec in config["entrants"]]
    names = [entry["name"] for entry in entrants]
    if len(names) < rules.map.players or len(set(names)) != len(names):
        raise ValueError("provide at least players uniquely named entrants")
    effective = {"rules": rules.manifest(), "maps_per_group": maps_per_group, "elo": asdict(elo)}
    schedule = _schedule(effective, entrants)
    directory = Path(output_dir).resolve()
    if not resume:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise FileNotFoundError("cannot resume an arena directory that does not exist")
    with _writer_lock(directory):
        if resume:
            previous = _read(directory / "config.json")
            old_entrants = _read(directory / "entrants.json")
            if previous["rules"] != effective["rules"] or previous["elo"] != effective["elo"]:
                raise ValueError("rules, engine implementation, and Elo settings must remain fixed; start a new arena")
            if maps_per_group < map_count(previous):
                raise ValueError("resume cannot reduce maps_per_group")
            current = {entry["name"]: entry["entrant_id"] for entry in entrants}
            if any(current.get(entry["name"]) != entry["entrant_id"] for entry in old_entrants):
                raise ValueError("existing entrants cannot be removed or changed; give new versions new names")
        matches = read_matches(directory / "matches.jsonl")
        seen = validate_matches(matches, schedule, effective["rules"])
        # Persist registration before loading optional dependencies so a failed
        # model load is resumable after the installation/configuration is fixed.
        write_json(directory / "config.json", effective)
        write_json(directory / "entrants.json", entrants)
        (directory / "matches.jsonl").touch(exist_ok=True)
        pending = [spec for spec in schedule if spec.game_id not in seen]
        needed = {key for spec in pending for key in spec.entrant_ids}
        controllers = {}
        try:
            for entry in entrants:
                if entry["entrant_id"] in needed:
                    controllers[entry["entrant_id"]] = load_controller(entry, rules)
            for spec in pending:
                try:
                    match = play_match(spec, rules, [controllers[key] for key in spec.entrant_ids])
                except Exception as exc:
                    write_json(directory / "failure.json", {"game_id": spec.game_id,
                               "error_type": type(exc).__name__, "error": str(exc)})
                    raise
                match.update(schema_version=1, rules_id=effective["rules"]["rules_id"], sequence=len(matches))
                validate_matches([dict(match, sequence=0)], [spec], effective["rules"])
                append_match(directory / "matches.jsonl", match)
                matches.append(match)
                if progress is not None:
                    progress(len(matches), len(schedule))
            (directory / "failure.json").unlink(missing_ok=True)
        finally:
            report = _report(directory, effective, entrants, matches, schedule)
        return report
