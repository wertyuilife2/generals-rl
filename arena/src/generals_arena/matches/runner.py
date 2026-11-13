"""Shared ordered execution for persistent leagues and fixed match schedules.

Callers own schedules, persistence and scoring. A runner owns controller
lifetimes and emits the same completed result/replay pair for either executor.
"""
import json
import time
from collections import defaultdict, deque
from dataclasses import replace
from types import SimpleNamespace

from generals import MapConfig

from generals_arena.policies.entries import load_controller
from generals_arena.util import integer
from generals_arena.matches.resources import cpu_slot
from generals_arena.matches.cache import initial_map


def build_controllers(entries, plan, *, resources=None):
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
        for key in plan["entrant_ids"]:
            entrant = entries[key]
            if entrant["kind"] == "human":
                controllers.append(None)
                continue
            controllers.append(load_controller(entrant, rules, resources=resources))
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


def resolve_geometries(plans):
    """Resolve a schedule's random dimensions once per map, before dispatch."""
    geometries, resolved = {}, []
    for original in plans:
        plan = dict(original)
        config = MapConfig(**plan['map'])
        if 'geometry' not in plan and (config.width is None or config.height is None):
            key = (config, plan['map_seed'])
            if key not in geometries:
                state = initial_map(config, plan['map_seed'])
                geometries[key] = [state.layout.width, state.layout.height]
            plan['geometry'] = geometries[key]
        resolved.append(plan)
    return resolved


class MatchRunner:
    def __init__(self, entries, *, max_ticks, backend="cpu", device="cpu", batch_size=128,
                 cpu_threads=1, record_replays=False, scalar=False, execution="batch",
                 worker=None):
        integer("max_ticks", max_ticks, 1)
        integer("batch_size", batch_size)
        integer("cpu_threads", cpu_threads, 1)
        device = str(device)
        if backend != 'cpu':
            raise ValueError('Arena requires native CPU execution; GPU/MPS training uses VecEnv')
        if device != "cpu" and device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("device must be cpu, cuda, or cuda:N")
        if execution not in ("batch", "isolated"):
            raise ValueError("execution must be 'batch' or 'isolated'")
        self.entries, self.max_ticks = entries, max_ticks
        self.backend, self.device = backend, device
        self.batch_size, self.cpu_threads = batch_size, cpu_threads
        self.record_replays, self.scalar = record_replays, scalar
        self.execution = execution
        self.worker, self._owned_worker = worker, None
        self._batch_worker = None
        self.execution_modes = set()

    def _validate(self, plans):
        """Validate the complete schedule before any results can be emitted."""
        from generals_arena.policies.artifacts import capabilities_supports
        used = {key for plan in plans for key in plan["entrant_ids"]}
        capabilities = {}
        for key in used:
            entry = self.entries[key]
            if entry["kind"] == "human":
                continue
            if entry["kind"] == "package":
                # Lightweight admission only; workers verify artifact bytes on load.
                from pathlib import Path
                from generals_arena.policies.artifacts import load_manifest
                manifest = load_manifest(Path(entry['factory_kwargs']['bundle'])/'manifest.json')
                capabilities[key] = manifest["capabilities"]
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
        if self._owned_worker is not None:
            self._owned_worker.close()
            self.worker = self._owned_worker = None
        if self._batch_worker is not None:
            self._batch_worker.close()
            self._batch_worker = None

    def _isolated(self, plan, stop):
        if self.worker is None:
            from .worker import MatchWorker
            self.worker = self._owned_worker = MatchWorker()
        try:
            with cpu_slot():
                return self.worker.run({key: self.entries[key] for key in plan['entrant_ids']},
                                       plan, max_ticks=self.max_ticks, stop=stop)
        except InterruptedError:
            return None  # Scheduler leaves the cancelled plan pending for retry.

    def run(self, plans, on_result=None, *, stop=None):
        """Emit results in schedule order; callbacks may atomically commit them.

        Stopping leaves the un-emitted suffix untouched. Every native policy
        uses the same worker and per-decision watchdog boundary.
        """
        plans = resolve_geometries(plans)
        self._validate(plans)
        results, timing = [], {}
        ready = {}
        buckets = defaultdict(deque)
        if not self.scalar and stop is None:
            for index, plan in enumerate(plans):
                buckets[_batch_key(plan)].append(index)
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
            if self.scalar or stop is not None:
                self.execution_modes.add("isolated")
                start = time.monotonic()
                record = self._isolated(plans[offset], stop)
                timing["simulation_seconds"] = timing.get("simulation_seconds", 0.) + time.monotonic() - start
                if record is None:
                    break
                records = [record]
                offset += 1
            else:
                self.execution_modes.add("isolated_batch")
                key = _batch_key(plans[offset])
                limit = self.batch_size or len(plans)
                bucket = buckets[key]
                indices = [bucket.popleft() for _ in range(min(limit, len(bucket)))]
                options = dict(max_ticks=self.max_ticks, device=self.device, backend=self.backend,
                    cpu_threads=self.cpu_threads, record_replays=self.record_replays, scalar=self.scalar)
                config = MapConfig(**plans[offset]['map'])
                options['initial_states'] = {(config, plans[index]['map_seed']):
                    initial_map(config, plans[index]['map_seed']) for index in indices}
                if self.worker is not None:
                    executor = self.worker.run_batch
                else:
                    if self._batch_worker is None:
                        from .batch_worker import BatchWorker
                        self._batch_worker = BatchWorker()
                    executor = self._batch_worker.run
                entries = {key: self.entries[key] for index in indices for key in plans[index]['entrant_ids']}
                records, elapsed = executor(entries,
                    [plans[index] for index in indices], **options)
                for name, seconds in elapsed.items():
                    timing[name] = timing.get(name, 0.) + seconds
                ready.update(zip(indices, records, strict=True))
                continue
            for value, blob in records:
                if on_result is not None:
                    on_result(value, blob)
                results.append(value)
        return results, timing
