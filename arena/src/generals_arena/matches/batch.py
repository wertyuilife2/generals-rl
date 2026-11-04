"""Joint-tick batch execution for explicit, independently seeded match plans."""
from dataclasses import replace

import numpy as np

from generals import Action, MapConfig, VecEnv
from generals.protocol import MOVE_REASONS
from .cache import initial_map as generate_map
from generals.core.observation import Observation, readonly
from generals.core.state import MoveResult, PlayerStats

from generals_arena.matches.config import MatchConfig
from generals_arena.policies.entries import load_controller
from generals_arena.matches.results import EXECUTION_POLICY
from generals_arena.matches.resources import cpu_bounded
from generals_arena.matches.native import bind_loop


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _host_observation(raw):
    """Transfer public arrays once per batch, never once per scalar lane."""
    if isinstance(raw['state']['army'], np.ndarray):
        return raw
    return {key: {name: _numpy(value) for name, value in item.items()} if key == 'state'
            else _numpy(item) if hasattr(item, 'shape') else item
            for key, item in raw.items()}


def _one_observation(batch, lane):
    """Scalar controllers see the same masked raw batch as policy packages."""
    state, width, height = batch["state"], batch["width"], batch["height"]
    pid = int(_numpy(batch["player_ids"])[lane])
    stats = _numpy(batch["stats"])[lane]
    return Observation(player_id=pid, tick=int(_numpy(state["tick"])[lane]),
        **{name: readonly(_numpy(state[name])[lane].reshape(height, width))
           for name in ("terrain", "structure", "owner", "army", "fog_obstacles")},
        visible=readonly(_numpy(batch["visible"])[lane].reshape(height, width)),
        alive=readonly(_numpy(state["alive"])[lane]),
        has_kill=readonly(_numpy(state["has_kill"])[lane]),
        warn_afk=readonly(_numpy(state["warn_afk"])[lane]),
        neutralize_in=readonly(_numpy(state["neutralize_in"])[lane]),
        general_pos=int(_numpy(batch["general_pos"])[lane]),
        stats=PlayerStats(*(readonly(stats[:, index]) for index in range(3))),
        visibility=batch.get("visibility", "local"), terminated=bool(_numpy(state["terminated"])[lane]),
        winner_id=None if int(_numpy(state["winner_id"])[lane]) < 0 else int(_numpy(state["winner_id"])[lane]))


def _move(row, pid, encoded):
    if not row[0]:
        return None
    return MoveResult(player_id=pid, action=Action.decode(int(encoded)), valid=bool(row[1]),
        reason=MOVE_REASONS[int(row[8])],
        source=int(row[2]), target=int(row[3]), moved=int(row[4]), captured=bool(row[5]),
        eliminated=None if row[6] < 0 else int(row[6]), target_owned=bool(row[7]))


class _PolicyGroup:
    def __init__(self, entry, rules, seats, seeds, *, scalar=False, cache=None, slot=None):
        self.entry, self.seats = entry, np.asarray(seats, dtype=np.int64)
        self.lanes = np.arange(len(self.seats))
        key = (entry["entrant_id"], slot, rules.map, rules.visibility)
        self.controller = cache[key] if cache is not None and key in cache else load_controller(entry, rules)
        if cache is not None and key in cache:
            if hasattr(cache, 'move_to_end'):
                cache.move_to_end(key)
        self.batch = (not scalar or callable(getattr(self.controller, 'native_batch', None))) and all(callable(getattr(self.controller, name, None))
                                        for name in ("reset_batch", "act_batch"))
        self.controllers = []
        self.cached = False
        try:
            if self.batch:
                self.controller.reset_batch(self.seats.tolist(), [int(seed) for seed in seeds])
                if cache is not None:
                    cache[key] = self.controller
                    self.cached = True
            else:
                for index, (seat, seed) in enumerate(zip(seats, seeds, strict=True)):
                    controller = self.controller if index == 0 else load_controller(entry, rules)
                    self.controllers.append(controller)
                    controller.reset(int(seat), int(seed))
        except BaseException:
            self.close()
            if cache is not None:
                cache.pop(key, None)
            raise

    def act(self, raw):
        if not self.batch or getattr(self.controller, 'batch_device', None) == 'cpu':
            raw = _host_observation(raw)
        if self.batch:
            actions = self.controller.act_batch(raw)
            if isinstance(actions, list) and all(isinstance(action, Action) for action in actions):
                actions = np.asarray([action.encode() for action in actions], dtype=np.int64)
            actions = actions.reshape(-1)
            if actions.shape != (len(self.seats),):
                raise ValueError("act_batch must return exactly one action per match")
            return actions
        active = _numpy(raw["active"])
        return np.asarray([controller.act(_one_observation(raw, lane)).encode() if active[lane] else 0
                           for lane, controller in enumerate(self.controllers)], dtype=np.int64)

    def feedback(self, host, joint, active):
        if self.batch:
            callback = getattr(self.controller, "on_result_batch", None)
            if callback is not None:
                own_moves = host["_moves"][self.lanes, self.seats]
                callback({"moves": own_moves,
                          "actions": joint[self.lanes, self.seats],
                          "active": active & own_moves[:, 0].astype(bool), "player_ids": self.seats.copy()})
        else:
            for lane in np.flatnonzero(active):
                seat = int(self.seats[lane])
                result = _move(host["_moves"][lane, seat], seat, joint[lane, seat])
                if result is not None:
                    self.controllers[lane].on_result(result)

    def close(self):
        if self.cached:
            release = getattr(self.controller, 'release_batch', None)
            if release is not None:
                release()
            return
        for controller in ([self.controller] if self.batch else self.controllers):
            close = getattr(controller, "close", None)
            if close is not None:
                close()


def _instances(entries, plans):
    """Validate the homogeneous bucket supplied by the schedule runner."""
    first = plans[0]
    config = MapConfig(**first["map"])
    mapping = dict(zip(first["instance_ids"], first["entrant_ids"], strict=True))
    if len(mapping) != config.players:
        raise ValueError("each match seat requires an independent instance_id")
    for plan in plans:
        if (plan["map"] != first["map"] or len(plan["controller_seeds"]) != config.players
                or len(plan["entrant_ids"]) != config.players
                or len(set(plan["instance_ids"])) != config.players
                or dict(zip(plan["instance_ids"], plan["entrant_ids"], strict=True)) != mapping):
            raise ValueError("execute_batch requires one map configuration and instance assignment")
    roles = []
    for instance, entrant in sorted(mapping.items()):
        entry = entries[entrant]
        seats = np.asarray([plan["instance_ids"].index(instance) for plan in plans], dtype=np.int64)
        seeds = [plan["controller_seeds"][int(seat)] for plan, seat in zip(plans, seats, strict=True)]
        roles.append((instance, entry, seats, seeds))
    return config, roles


@cpu_bounded
def execute_batch(entries, plans, *, max_ticks, device="cpu", backend="cpu", cpu_threads=1,
                  record_replays=False, scalar=False, runtime_cache=None, journal=None, native=True,
                  initial_states=None):
    """Schedule one homogeneous batch in the native Arena executor.

    native=False benchmarks the public policy bridge, not a second match loop.
    Environment training backends remain independent of Arena's CPU runtime.
    """
    if not plans:
        return [], {}
    if backend != 'cpu':
        raise ValueError('Arena requires the native CPU backend; GPU/MPS simulation is available through VecEnv')
    config, roles = _instances(entries, plans)
    maps = {plan['map_seed']: None for plan in plans}
    for seed in maps:
        maps[seed] = (initial_states[(config, seed)] if initial_states is not None and (config, seed) in initial_states
                      else generate_map(config, seed))
    states = [maps[plan['map_seed']] for plan in plans]
    config = replace(config, width=states[0].layout.width, height=states[0].layout.height)
    count, players = len(plans), config.players
    rules = MatchConfig(config, max_ticks=max_ticks)
    timing = dict(observation_seconds=0., policy_seconds=0., step_seconds=0., transfer_seconds=0.)
    groups, records = [], [None]*count
    try:
        for instance, entry, seats, seeds in roles:
            groups.append(_PolicyGroup(entry, rules, seats, seeds, scalar=scalar,
                                       cache=runtime_cache, slot=instance))
        with VecEnv.from_states(states, device='cpu', threads=cpu_threads,
                                auto_reset=False, episode_ticks=max_ticks) as env:
            active = np.ones(count, bool)
            joint, invalid = np.zeros((count, players), np.int64), np.zeros((count, players), np.int64)
            eliminations = np.full((count, players), -1, np.int64)
            with bind_loop(env, groups, joint, invalid, eliminations, active, journal,
                           native_policies=native, topologies=[s.layout.topology_id for s in states],
                           record=record_replays) as execution:
                while np.any(active):
                    execution.advance(max_ticks, timing)
                    for lane in np.flatnonzero(execution.ending):
                        record = {**plans[lane], **execution.query(0, lane), 'execution_policy': EXECUTION_POLICY}
                        metrics = execution.decision_timing(lane)
                        if metrics:
                            for item in metrics:
                                item['entrant_id'] = plans[lane]['entrant_ids'][item['seat']]
                            record['decision_timing'] = metrics
                        from .replay import encode_native_trace
                        blob = encode_native_trace(execution.query_bytes(1, lane)) if record_replays else None
                        records[lane] = record, blob
    finally:
        for group in groups:
            group.close()
    return records, timing
