"""Fixed round-robin mode over Arena's shared execution and rating foundation."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from generals_arena.matches.config import MatchConfig
from generals_arena.policies.entries import prepare_entrant
from generals_arena.matches.reporting import output_path
from generals_arena.matches.reporting import replay_writer
from generals_arena.matches.reporting import write_report
from generals_arena.matches.runner import MatchRunner
from generals_arena.competitions.schedules import round_robin
from generals_arena.competitions.rating_session import score_results
from generals_arena.policies.builtin import with_wait_reference
from generals_arena.util import digest
from generals_arena.util import source_digest


def tournament(entrants, rules=None, output=None, *, maps_per_group=1, seed=42, paired_seats=True,
               device="cpu", backend="cpu", batch_size=128, cpu_threads=1,
               record_replays=False, scalar=False, base_dir=Path(".")):
    """Enumerate every P-policy group; use the same manager as other modes.

    ``entrants`` contains the normal Arena entrant specifications, including
    opaque exported bundles. Duels include WAIT as the display reference; FFA
    starts at mean 1000. Ratings evolve through the deterministic match order.
    """
    started = perf_counter()
    rules = rules or MatchConfig()
    specs = [{**spec, "device": spec.get("device", device)} if spec.get("kind") == "package" else spec
             for spec in entrants]
    mode = "1v1" if rules.map.players == 2 else "ffa"
    specs = with_wait_reference(specs, rules.map.players)
    prepared = [prepare_entrant(spec, base_dir=base_dir) for spec in specs]
    if len({entry["name"] for entry in prepared}) != len(prepared):
        raise ValueError("tournament entrant names must be unique")
    entries = {entry["entrant_id"]: entry for entry in prepared}
    plans = list(round_robin(entries, rules.map, maps_per_group=maps_per_group, seed=seed,
                             paired_seats=paired_seats))
    reference = next((key for key, entry in entries.items() if entry["name"] == "WAIT"), None) if mode == "1v1" else None
    request = {"mode": "tournament", "map": asdict(rules.map), "max_ticks": rules.max_ticks,
               "maps_per_group": maps_per_group, "seed": seed, "paired_seats": paired_seats,
               "entrant_ids": sorted(entries), "rating_mode": mode, "reference": reference,
               "engine_sha256": source_digest("generals"),
               "matches_sha256": source_digest("generals_arena.matches"),
               "competitions_sha256": source_digest("generals_arena.competitions")}
    output = output_path(output)
    with MatchRunner(entries, max_ticks=rules.max_ticks, device=device, backend=backend,
                     batch_size=batch_size, cpu_threads=cpu_threads, record_replays=record_replays,
                     scalar=scalar) as runner:
        results, timing = runner.run(plans, on_result=replay_writer(output, enabled=record_replays))
    scoring = score_results(sorted(entries), results, mode=mode, reference=reference)
    for row in scoring["ranking"]:
        row["name"] = entries[row["entrant_id"]]["name"]
    report = {"schema_version": 1, "status": "completed", "mode": "tournament",
              "run_id": digest(request), "request": request, **scoring,
              "expected_games": len(plans), "results": results, "errors": [],
              "wall_seconds": perf_counter() - started, "timing": timing,
              "backend": backend, "device": str(device), "batch_size": batch_size,
              "cpu_threads": cpu_threads, "execution": next(iter(runner.execution_modes)) if len(runner.execution_modes) == 1 else "mixed",
              "execution_modes": sorted(runner.execution_modes),
              "timing_basis": "host_dispatch_wall_with_cuda_sync_in_transfer"}
    write_report(output, report)
    return report
