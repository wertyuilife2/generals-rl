"""Joint-tick batch execution for explicit, independently seeded match plans."""
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from generals import Action, MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals.core.observation import Observation, readonly
from generals.core.recording import state_digest, state_to_dict
from generals.core.state import GameState, MoveResult, PlayerStats

from generals_arena.matches.config import MatchConfig
from generals_arena.policies.entries import load_controller
from generals_arena.matches.results import EXECUTION_POLICY
from generals_arena.matches.results import build_result
from generals_arena.matches.replay import encode_replay


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class _HostSnapshot:
    """One synchronization for all required CUDA host views per joint tick."""
    def __init__(self, env, joint):
        self.env = env
        self.sources = {**env.state, "_moves": env.unwrapped.last_moves, "_done": env.done, "_stats": env.stats,
                        "_joint": joint, "_errors": env.unwrapped.errors}
        self.cuda = hasattr(env.state["army"], "is_cuda")
        if self.cuda:
            import torch
            self.buffers = {name: torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
                            for name, value in self.sources.items()}
            self.arrays = {name: value.numpy() for name, value in self.buffers.items()}
        else:
            self.arrays = self.sources
        self.refresh()

    def refresh(self):
        if self.cuda:
            import torch
            for name, source in self.sources.items():
                self.buffers[name].copy_(source, non_blocking=True)
            torch.cuda.current_stream(self.env.device).synchronize()
        if np.any(self.arrays["_errors"]):
            raise RuntimeError("vector simulator reported a native error")
        return self.arrays


def _one_observation(batch, lane):
    """Scalar controllers see the same masked raw batch as policy packages."""
    state, width, height = batch["state"], batch["width"], batch["height"]
    pid = int(_numpy(batch["player_ids"])[lane])
    stats = _numpy(batch["stats"])[lane]
    return Observation(player_id=pid, tick=int(_numpy(state["tick"])[lane]),
        **{name: readonly(_numpy(state[name])[lane].reshape(height, width))
           for name in ("terrain", "structure", "owner", "army")},
        visible=readonly(_numpy(batch["visible"])[lane].reshape(height, width)),
        alive=readonly(_numpy(state["alive"])[lane]),
        general_pos=int(_numpy(batch["general_pos"])[lane]),
        stats=PlayerStats(*(readonly(stats[:, index]) for index in range(3))),
        visibility="local", terminated=bool(_numpy(state["terminated"])[lane]),
        winner_id=None if int(_numpy(state["winner_id"])[lane]) < 0 else int(_numpy(state["winner_id"])[lane]))


def _lane_state(host, lane, layout):
    shape = (layout.height, layout.width)
    winner = int(host["winner"][lane])
    return GameState(layout, *(host[name][lane].reshape(shape) for name in ("structure", "owner", "army")),
                     host["alive"][lane], host["general_pos"][lane], tick=int(host["tick"][lane]),
                     terminated=bool(host["terminated"][lane]), winner_id=None if winner < 0 else winner)


def _move(row, pid, encoded):
    if not row[0]:
        return None
    return MoveResult(player_id=pid, action=Action.decode(int(encoded)), valid=bool(row[1]),
        reason={1: "WAIT", 2: "MOVED", 3: "OUT_OF_BOUNDS", 4: "NOT_OWNER", 5: "MOUNTAIN"}[int(row[8])],
        source=int(row[2]), target=int(row[3]), moved=int(row[4]), captured=bool(row[5]),
        eliminated=None if row[6] < 0 else int(row[6]), target_owned=bool(row[7]))


class _PolicyGroup:
    def __init__(self, entry, rules, seats, seeds, *, scalar=False, cache=None, slot=None):
        self.entry, self.seats = entry, np.asarray(seats, dtype=np.int64)
        key = (entry["entrant_id"], slot, rules.map, rules.visibility)
        self.controller = cache[key] if cache is not None and key in cache else load_controller(entry, rules)
        self.batch = not scalar and all(callable(getattr(self.controller, name, None))
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
                lanes = np.arange(len(self.seats))
                own_moves = host["_moves"][lanes, self.seats].copy()
                callback({"moves": own_moves,
                          "actions": joint[lanes, self.seats].copy(),
                          "active": active & own_moves[:, 0].astype(bool), "player_ids": self.seats.copy()})
        else:
            for lane in np.flatnonzero(active):
                seat = int(self.seats[lane])
                result = _move(host["_moves"][lane, seat], seat, joint[lane, seat])
                if result is not None:
                    self.controllers[lane].on_result(result)

    def close(self):
        if self.cached:
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


def execute_batch(entries, plans, *, max_ticks, device="cpu", backend="cpu", cpu_threads=1,
                  record_replays=False, scalar=False, runtime_cache=None):
    """Return input-ordered (result, replay-or-None) pairs for explicit plans.

    Rows share map configuration and instance-to-entrant assignment; seats may
    rotate. The caller owns batch controllers cached by entrant, instance, map and visibility.
    Every controller_seed is final, including native P2 baseline seeds.
    """
    if not plans:
        return [], {}
    config, roles = _instances(entries, plans)
    if backend not in ("cpu", "gpu"):
        raise ValueError("backend must be cpu or gpu")
    states = [generate_map(config, plan["map_seed"]) for plan in plans]
    count, players = len(plans), config.players
    rules = MatchConfig(config, max_ticks=max_ticks)
    placement = {"device": device if str(device).startswith("cuda") else "cuda:0"} if backend == "gpu" else {
        "device": "cpu", "threads": cpu_threads}
    timing = dict(observation_seconds=0., policy_seconds=0., step_seconds=0., transfer_seconds=0.)
    groups, baselines, records = [], [], [None] * count
    eliminations = np.full((count, players), -1, dtype=np.int64)
    invalid = np.zeros((count, players), dtype=np.int64)
    kills = np.zeros((count, players), dtype=np.int64)
    traces = [[] for _ in plans] if record_replays else None
    digests = [[] for _ in plans] if record_replays else None
    initial_digests = [state_digest(state) for state in states]
    try:
        for instance, entry, seats, seeds in roles:
            if players == 2 and entry["kind"] == "baseline" and not scalar:
                from generals_baselines import BatchBaselineControllers
                specs = [SimpleNamespace(opponent=entry["factory_kwargs"]["kind"], learner_seat=1-int(seat))
                         for seat in seats]
                owner = BatchBaselineControllers(specs, height=config.height, width=config.width, seat_seeds=seeds)
                baselines.append((owner, seats))
            else:
                groups.append(_PolicyGroup(entry, rules, seats, seeds, scalar=scalar,
                                           cache=runtime_cache, slot=instance))
        with VecEnv.from_states(states, **placement, auto_reset=False, episode_ticks=max_ticks) as env:
            observers = [env.observer(group.seats) for group in groups]
            active = np.ones(count, dtype=np.bool_)
            if backend == "gpu":
                import torch
                joint = torch.zeros((count, players), dtype=torch.int64, device=env.device)
                lanes = torch.arange(count, device=env.device)
                group_seats = [torch.as_tensor(group.seats, device=env.device) for group in groups]
                baseline_seats = [torch.as_tensor(seats, device=env.device) for _, seats in baselines]
            else:
                joint = np.zeros((count, players), dtype=np.int64)
                lanes = np.arange(count)
                group_seats = [group.seats for group in groups]
                baseline_seats = [seats for _, seats in baselines]
            host_view = _HostSnapshot(env, joint)
            host = host_view.arrays
            while active.any():
                joint *= 0
                for observer, group, seats in zip(observers, groups, group_seats, strict=True):
                    entered = perf_counter()
                    raw = observer()
                    timing["observation_seconds"] += perf_counter() - entered
                    entered = perf_counter()
                    actions = group.act(raw)
                    joint[lanes, seats] = (torch.as_tensor(actions, device=env.device, dtype=torch.int64)
                                           if backend == "gpu" else _numpy(actions))
                    timing["policy_seconds"] += perf_counter() - entered
                for (owner, _), seats in zip(baselines, baseline_seats, strict=True):
                    entered = perf_counter()
                    actions = owner.act(host, active)
                    joint[lanes, seats] = (torch.as_tensor(actions, device=env.device)
                                           if backend == "gpu" else actions)
                    timing["policy_seconds"] += perf_counter() - entered
                entered = perf_counter()
                env.step(joint)
                timing["step_seconds"] += perf_counter() - entered
                entered = perf_counter()
                host = host_view.refresh()
                timing["transfer_seconds"] += perf_counter() - entered
                joint_host, moves = host["_joint"], host["_moves"]
                invalid += (moves[:, :, 0].astype(bool) & ~moves[:, :, 1].astype(bool)) * active[:, None]
                killed = moves[:, :, 0].astype(bool) & (moves[:, :, 6] >= 0) & active[:, None]
                killed_lanes, killed_seats = np.nonzero(killed)
                eliminations[killed_lanes, moves[killed_lanes, killed_seats, 6]] = host["tick"][killed_lanes]
                kills += killed
                for owner, _ in baselines:
                    owner.on_result(moves, joint_host, active)
                for group in groups:
                    group.feedback(host, joint_host, active)
                if record_replays:
                    for lane in np.flatnonzero(active):
                        traces[lane].append([[int(a.kind), a.source, int(a.direction), int(a.mode)]
                                             for a in (Action.decode(int(value)) for value in joint_host[lane])])
                        digests[lane].append(state_digest(_lane_state(host, lane, states[lane].layout)))
                ending = active & host["_done"].any(axis=1)
                for lane in np.flatnonzero(ending):
                    state = _lane_state(host, lane, states[lane].layout)
                    record = build_result(plans[lane], state=state,
                        stats={name: host["_stats"][lane, :, index] for index, name in enumerate(("army", "territory", "cities"))},
                        elimination_ticks=[None if value < 0 else int(value) for value in eliminations[lane]],
                        invalid_actions=invalid[lane].tolist(), initial_digest=initial_digests[lane], kills=kills[lane].tolist())
                    record["execution_policy"] = EXECUTION_POLICY
                    blob = encode_replay(state_to_dict(states[lane]), traces[lane], digests[lane]) if record_replays else None
                    records[lane] = record, blob
                active[ending] = False
    finally:
        for owner, _ in baselines:
            owner.close()
        for group in groups:
            group.close()
    return records, timing
