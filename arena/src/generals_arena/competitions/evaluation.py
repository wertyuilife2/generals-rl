"""Training evaluation: fixed policies/schedule, using the shared rating rules.

Execution, policy batching, game results and replays belong to matches, exactly
as for round-robin and persistent-league modes. No RL implementation is imported.
"""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from generals import MapConfig

from generals_arena.policies.entries import prepare_entrant
from generals_arena.policies.artifacts import RULES_PROTOCOL
from generals_arena.policies.artifacts import validate_bundle
from generals_arena.policies.builtin import with_wait_reference
from generals_arena.util import digest
from generals_arena.util import file_digest
from generals_arena.util import integer
from generals_arena.util import source_digest
from generals_arena.matches.runner import MatchRunner
from generals_arena.competitions.schedules import fixed_panel
from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.rating_session import score_results
from generals_arena.matches.results import aggregate
from generals_arena.matches.results import seat_outcome
from generals_arena.matches.reporting import output_path
from generals_arena.matches.reporting import replay_writer
from generals_arena.matches.reporting import write_report

def make_profile(map_config, *, max_ticks=2000, opponents=None, seed=1000042,
                 maps_per_opponent=50, paired_seats=True):
    """Build a serializable, model-independent fixed evaluation gauntlet."""
    map_config = MapConfig(**map_config) if isinstance(map_config, dict) else map_config
    if not isinstance(map_config, MapConfig):
        raise TypeError("map_config must be a MapConfig or its dictionary")
    integer("max_ticks", max_ticks, 1)
    integer("seed", seed, 0)
    integer("maps_per_opponent", maps_per_opponent, 1)
    if not isinstance(paired_seats, bool):
        raise TypeError("paired_seats must be boolean")
    if opponents is None:
        from generals_baselines import AI_KINDS
        opponents = [{"name": name, "kind": "baseline", "policy": name} for name in AI_KINDS]
    else:
        opponents = list(opponents)
    if not opponents:
        raise ValueError("provide at least one fixed opponent")
    mode = "1v1" if map_config.players == 2 else "ffa"
    opponents = with_wait_reference(opponents, map_config.players)
    names = [spec["name"] for spec in opponents]
    if len(set(names)) != len(names):
        raise ValueError("opponent names must be unique")
    return {"schema_version": 2, "rules": RULES_PROTOCOL, "map": asdict(map_config),
            "max_ticks": max_ticks, "opponents": opponents, "seed": seed,
            "maps_per_opponent": maps_per_opponent, "paired_seats": paired_seats,
            "rating_policy": POLICY, "rating_mode": mode,
            "reference": "WAIT" if mode == "1v1" else None}


def load_profile(profile):
    """Resolve opponent artifacts relative to the profile file, once at load."""
    base = Path.cwd()
    if isinstance(profile, (str, Path)):
        path = Path(profile).resolve()
        base, profile = path.parent, json.loads(path.read_text())
    else:
        profile = json.loads(json.dumps(profile))
    for spec in profile.get("opponents", []):
        if spec.get("kind") in ("package", "cpp"):
            spec["bundle"] = str((base / spec["bundle"]).resolve())
        elif spec.get("kind") == "custom" and "artifacts" in spec:
            spec["artifacts"] = {key: str((base / value).resolve())
                                 for key, value in spec["artifacts"].items()}
    return profile


def _profile(profile):
    profile = load_profile(profile)
    required = {"schema_version", "rules", "map", "max_ticks", "opponents", "seed",
                "maps_per_opponent", "paired_seats", "rating_policy", "rating_mode", "reference"}
    if set(profile) != required:
        raise ValueError("invalid evaluation profile fields")
    if profile["schema_version"] != 2 or profile["rules"] != RULES_PROTOCOL or profile["rating_policy"] != POLICY:
        raise ValueError("unsupported evaluation rules/profile/rating version")
    normalized = make_profile(profile["map"], max_ticks=profile["max_ticks"], opponents=profile["opponents"],
                              seed=profile["seed"], maps_per_opponent=profile["maps_per_opponent"],
                              paired_seats=profile["paired_seats"])
    if normalized != profile:
        raise ValueError("evaluation profile must match its map's rating mode and WAIT reference")
    entries = [prepare_entrant(spec) for spec in normalized["opponents"]]
    identity = {**normalized, "opponents": [{key: entry[key] for key in (
        "entrant_id", "name", "kind", "version", "factory", "kwargs", "implementation")} for entry in entries],
                "engine_sha256": source_digest("generals"),
                "evaluation_sha256": file_digest(Path(__file__)),
                "matches_sha256": source_digest("generals_arena.matches"),
                "competitions_sha256": source_digest("generals_arena.competitions"),
                "decision_timing": "joint_pre_growth_then_absolute_seat_order"}
    return normalized, entries, digest(identity)


def profile_id(profile):
    return _profile(profile)[2]


def expected_games(profile):
    """The evaluation-mode budget; callers need not reconstruct its schedule."""
    return len(profile["opponents"]) * profile["maps_per_opponent"] * (
        profile["map"]["players"] if profile["paired_seats"] else 1)


def validate_coverage(report, profile):
    """Check complete scheduled evidence before a training queue selects a model."""
    count = expected_games(profile)
    opponents = {spec["name"] for spec in profile["opponents"]}
    rows = report.get("results")
    per_opponent = count // len(opponents)
    if (report.get("games") != count or report.get("expected_games") != count
            or not isinstance(rows, list) or len(rows) != count
            or any(not isinstance(row, dict) or not isinstance(row.get("game_id"), str) for row in rows)
            or len({row["game_id"] for row in rows}) != count
            or set(report.get("by_opponent", {})) != opponents
            or any(sum(row.get("opponent") == opponent for row in rows) != per_opponent
                   or report["by_opponent"][opponent].get("games") != per_opponent for opponent in opponents)):
        raise RuntimeError("Arena report does not cover every scheduled opponent/game")


def evaluate(bundle, profile, output=None, *, device="cpu", batch_size=128, backend="cpu", cpu_threads=1,
             record_replays=False, scalar=False, base_dir=Path(".")):
    """Run the fixed-panel mode through the common match manager.

    A bundle directory or a normal Arena entrant specification is accepted.
    Batch size zero executes one compatible schedule group at a time in full.
    """
    started = perf_counter()
    profile, opponents, identity = _profile(profile)
    if isinstance(bundle, dict):
        candidate = prepare_entrant(bundle, base_dir=base_dir)
        policy_id = candidate["version"] if candidate["kind"] == "package" else candidate["entrant_id"]
    else:
        info = validate_bundle(bundle)
        candidate = prepare_entrant({"name": info["name"], "kind": "package", "bundle": str(bundle), "device": device})
        policy_id = info["policy_id"]
    entries = {entry["entrant_id"]: entry for entry in (candidate, *opponents)}
    if len(entries) != len(opponents) + 1:
        raise ValueError("candidate and opponents must be distinct policies")
    plans = [row for opponent in opponents for row in fixed_panel(profile, opponent, candidate)]
    output = output_path(output)
    sink = replay_writer(output, enabled=record_replays)
    with MatchRunner(entries, max_ticks=profile["max_ticks"], device=device, backend=backend,
                     batch_size=batch_size, cpu_threads=cpu_threads, record_replays=record_replays,
                     scalar=scalar) as runner:
        games, timings = runner.run(plans, on_result=sink)
    for game in games:
        game["outcome"] = seat_outcome(game, game["candidate_seat"])
    reference = next((entry["entrant_id"] for entry in opponents if entry["name"] == profile["reference"]), None)
    scoring = score_results(entries, games, mode=profile["rating_mode"], reference=reference)
    candidate_score = next(row for row in scoring["ranking"] if row["entrant_id"] == candidate["entrant_id"])
    for row in scoring["ranking"]:
        row["name"] = entries[row["entrant_id"]]["name"]
    report = {"schema_version": 2, "mode": "training_evaluation", "status": "completed",
              "policy_id": policy_id, "profile_id": identity, "profile": profile,
              **scoring, "elo": candidate_score["elo"], "raw_elo": candidate_score["raw_elo"],
              "elo_ci": candidate_score["elo_ci"], "reference": profile["reference"], **aggregate(games, candidate["entrant_id"]),
              "expected_games": expected_games(profile), "errors": [],
              "by_opponent": {entry["name"]: aggregate([game for game in games if game["opponent"] == entry["name"]], candidate["entrant_id"])
                              for entry in opponents},
              "results": games, "wall_seconds": perf_counter() - started, "timing": timings,
              "device": str(device), "backend": backend, "batch_size": batch_size,
              "cpu_threads": cpu_threads, "timing_basis": "host_dispatch_wall_with_cuda_sync_in_transfer",
              "execution": next(iter(runner.execution_modes)) if len(runner.execution_modes) == 1 else "mixed",
              "execution_modes": sorted(runner.execution_modes)}
    validate_coverage(report, profile)
    write_report(output, report)
    return report
