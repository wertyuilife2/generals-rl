"""Shared ordered execution for persistent leagues and fixed match schedules.

Callers own schedules, persistence and scoring. A runner owns controller
lifetimes and emits the same completed result/replay pair for either executor.
"""
import json
import time
from dataclasses import replace
from types import SimpleNamespace

from generals import MapConfig

from generals_arena.policies.entries import load_controller
from generals_arena.policies.entries import verify_artifacts
from generals_arena.util import integer
from generals_arena.util import source_digest


def _verify_implementations(entries):
    sources = {}
    for entrant in entries:
        if entrant["kind"] == "human":
            continue
        for package, expected in entrant["implementation"].items():
            if package == "custom_version":
                continue
            if package not in sources:
                sources[package] = source_digest(package)
            if sources[package] != expected:
                raise ValueError(f"registered source changed for {entrant['name']}; restore the recorded version")


def build_controllers(entries, plan):
    """Create independent seat controllers against the planned map contract."""
    config = MapConfig(**plan["map"])
    if config.width is None or config.height is None:
        from generals.core.mapgen import generate_map
        geometry = plan.get('geometry')
        if geometry is None:
            layout = generate_map(config, plan['map_seed']).layout
            geometry = (layout.width, layout.height)
        config = replace(config, width=geometry[0], height=geometry[1])
    rules = SimpleNamespace(map=config, visibility="local")
    controllers = []
    try:
        _verify_implementations(entries[key] for key in set(plan["entrant_ids"]))
        for key in plan["entrant_ids"]:
            entrant = entries[key]
            if entrant["kind"] == "human":
                controllers.append(None)
                continue
            controllers.append(load_controller(entrant, rules))
        return controllers
    except BaseException:
        for controller in controllers:
            close = getattr(controller, "close", None)
            if close is not None:
                close()
        raise


def _batch_key(plan):
    """Rotation groups share instances even when their seat positions change."""
    return (json.dumps(plan["map"], sort_keys=True), tuple(plan.get('geometry', ())),
            tuple(sorted(zip(plan["instance_ids"], plan["entrant_ids"]))))


class MatchRunner:
    def __init__(self, entries, *, max_ticks, backend="cpu", device="cpu", batch_size=128,
                 cpu_threads=1, record_replays=False, scalar=False, execution="batch",
                 worker=None):
        integer("max_ticks", max_ticks, 1)
        integer("batch_size", batch_size)
        integer("cpu_threads", cpu_threads, 1)
        device = str(device)
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be cpu or gpu")
        if device != "cpu" and device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("device must be cpu, cuda, or cuda:N")
        if backend == "gpu" and device == "cpu":
            raise ValueError("GPU backend requires a CUDA device")
        if execution not in ("batch", "isolated"):
            raise ValueError("execution must be 'batch' or 'isolated'")
        self.entries, self.max_ticks = entries, max_ticks
        self.backend, self.device = backend, device
        self.batch_size, self.cpu_threads = batch_size, cpu_threads
        self.record_replays, self.scalar = record_replays, scalar
        self.execution = execution
        self.worker, self._owned_worker = worker, None
        self.runtime_cache = {}
        self.execution_modes = set()

    def _validate(self, plans):
        """Validate the complete schedule before any results can be emitted."""
        from generals_arena.policies.artifacts import capabilities_supports
        used = {key for plan in plans for key in plan["entrant_ids"]}
        if self.backend == "gpu" and any(self.entries[key]["kind"] == "cpp" for key in used):
            raise ValueError("C++ policies require isolated CPU execution; GPU batch execution is unsupported")
        _verify_implementations(self.entries[key] for key in used)
        capabilities = {}
        for key in used:
            entry = self.entries[key]
            if entry["kind"] == "human":
                continue
            info = verify_artifacts(entry)
            if entry["kind"] == "package":
                capabilities[key] = info["manifest"]["capabilities"]
        for plan in plans:
            integer("map seed", plan["map_seed"])
            config = MapConfig(**plan["map"])
            if 'geometry' in plan:
                config = replace(config, width=plan['geometry'][0], height=plan['geometry'][1])
            if (len(plan["entrant_ids"]) != config.players or len(plan["instance_ids"]) != config.players
                    or len(set(plan["instance_ids"])) != config.players
                    or len(plan["controller_seeds"]) != config.players):
                raise ValueError("each match seat requires an independent instance and controller seed")
            for seed in plan["controller_seeds"]:
                integer("controller seed", seed)
                if seed >= 2**64:
                    raise ValueError("controller seed must fit uint64")
            for key in set(plan["entrant_ids"]) & capabilities.keys():
                if not capabilities_supports(capabilities[key], config.width, config.height, config.players):
                    raise ValueError(f"policy {self.entries[key]['name']} does not support this match map/player count")
        return used

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        cache, self.runtime_cache = self.runtime_cache, {}
        try:
            for controller in cache.values():
                close = getattr(controller, "close", None)
                if close is not None:
                    close()
        finally:
            if self._owned_worker is not None:
                self._owned_worker.close()
                self.worker = self._owned_worker = None

    def _isolated(self, plan, stop):
        from generals_arena.matches.game import RecordedGame
        from generals_arena.matches.game import close_game
        from generals_arena.matches.game import failure
        from generals_arena.policies.native import PolicyError

        if any(self.entries[key]["kind"] == "cpp" for key in plan["entrant_ids"]):
            if self.worker is None:
                from generals_arena.matches.worker import MatchWorker
                self.worker = self._owned_worker = MatchWorker()
            # The match process owns native watchdogs, journal recovery and
            # decision-timeout forfeits; a generic batch call cannot replace it.
            return self.worker.run(self.entries, plan, max_ticks=self.max_ticks)

        controllers, game = [], None
        config = SimpleNamespace(max_ticks=self.max_ticks)
        try:
            controllers = build_controllers(self.entries, plan)
            game = RecordedGame(plan, config, controllers)
            while not game.done:
                if stop is not None and stop.is_set():
                    return None  # Leave a persistent plan pending for retry.
                game.advance()
            return game.result(), game.replay_blob()
        except PolicyError as exc:
            if game is None:
                game = RecordedGame(plan, config, [None] * len(plan["entrant_ids"]))
                game.controllers = controllers
            return failure(game, exc), game.replay_blob()
        finally:
            if game is not None:
                close_game(game)
            else:
                for controller in controllers:
                    close = getattr(controller, "close", None)
                    if close is not None:
                        close()

    def run(self, plans, on_result=None, *, stop=None):
        """Emit results in schedule order; callbacks may atomically commit them.

        Stopping leaves the un-emitted suffix untouched. Native C++ matches
        retain their existing whole-match watchdog boundary.
        """
        plans = list(plans)
        from generals.core.mapgen import generate_map
        geometries = {}
        plans = [dict(plan) for plan in plans]
        for plan in plans:
            config = MapConfig(**plan['map'])
            if config.width is None or config.height is None:
                key = (config, plan['map_seed'])
                if key not in geometries:
                    state = generate_map(config, plan['map_seed'])
                    geometries[key] = [state.layout.width, state.layout.height]
                plan['geometry'] = geometries[key]
        used = self._validate(plans)
        results, timing = [], {}
        ready = {}
        offset = 0
        while offset < len(plans):
            if stop is not None and stop.is_set():
                break
            if offset in ready:
                value, blob = ready.pop(offset)
                if on_result is not None:
                    on_result(value, blob)
                results.append(value)
                offset += 1
                continue
            isolated = self.execution == "isolated" or any(
                self.entries[key]["kind"] == "cpp" for key in plans[offset]["entrant_ids"])
            if isolated:
                self.execution_modes.add("isolated")
                start = time.monotonic()
                record = self._isolated(plans[offset], stop)
                timing["simulation_seconds"] = timing.get("simulation_seconds", 0.) + time.monotonic() - start
                if record is None:
                    break
                records = [record]
                offset += 1
            else:
                self.execution_modes.add("scalar_reference" if self.scalar else "policy_batch")
                from generals_arena.matches.batch import execute_batch
                key = _batch_key(plans[offset])
                limit = self.batch_size or len(plans)
                indices = [index for index in range(offset, len(plans))
                           if index not in ready and _batch_key(plans[index]) == key][:limit]
                records, elapsed = execute_batch(self.entries, [plans[index] for index in indices],
                    max_ticks=self.max_ticks, device=self.device, backend=self.backend,
                    cpu_threads=self.cpu_threads, record_replays=self.record_replays,
                    scalar=self.scalar, runtime_cache=self.runtime_cache)
                for name, seconds in elapsed.items():
                    timing[name] = timing.get(name, 0.) + seconds
                ready.update(zip(indices, records, strict=True))
                continue
            for value, blob in records:
                if on_result is not None:
                    on_result(value, blob)
                results.append(value)
        for key in used:
            if self.entries[key]["kind"] != "human":
                verify_artifacts(self.entries[key])
        return results, timing
