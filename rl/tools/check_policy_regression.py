"""Offline fixed-NN and greedy/stochastic checks using retained policy snapshots.

Run with PYTHONPATH pointing to the run's frozen packages. This tool does not
train, publish snapshots, or modify the historical opponent pool.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from generals import MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals_rl.config import task_config
from generals_rl.evaluate_builtin import (
    _load_student, _local_inputs, aggregate_games, battle_batch, make_schedule,
)
from generals_rl.mdp import (
    ActionMetrics, GameBatch, ObservationHistory, make_action_space,
    make_observation_encoder,
)


KINDS = ("aggressive", "expansion", "defensive", "random")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class OpeningWaits:
    """Count planned opening WAIT decisions without writing world recordings."""

    def on_start(self, specs, states):
        self.counts = {s.game_id: 0 for s in specs}
        self.opening = set(self.counts)

    def on_tick(self, spec, actions, state, moves, done):
        if spec.game_id in self.opening:
            if actions[spec.learner_seat].encode() == 0:
                self.counts[spec.game_id] += 1
            else:
                self.opening.remove(spec.game_id)


class SampledPolicy:
    def __init__(self, model, seed):
        self.model = model
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def act_inference(self, obs):
        return torch.multinomial(self.model.action_probabilities(obs), 1,
                                 generator=self.generator)


def load_policy(checkpoint, config, states, task):
    with VecEnv.from_states(states[:1], device="cpu", threads=1) as env:
        batch = GameBatch.from_backend(env, "cpu")
        ids = torch.zeros(1, dtype=torch.long)
        encoder = make_observation_encoder(task)
        history = ObservationHistory(task.history_length)
        history.reset(encoder.encode(batch, ids))
        obs = _local_inputs(batch, ids, encoder, history, make_action_space(task))
        return _load_student(checkpoint, config, obs, "cpu")


def nn_batch(specs, states, learner, opponent, task, *, max_ticks, seed):
    """Two local policies decide jointly before native growth/absolute seat order."""
    count = len(specs)
    ids = torch.tensor([s.learner_seat for s in specs], dtype=torch.long)
    encoder, space = make_observation_encoder(task), make_action_space(task)
    histories = [ObservationHistory(task.history_length) for _ in range(2)]
    metrics = ActionMetrics(count, "cpu")
    totals = {name: torch.zeros(count) for name in ActionMetrics.names}
    generators = [torch.Generator().manual_seed(seed + role) for role in range(2)]
    active = torch.ones(count, dtype=torch.bool)
    decisions, opening = torch.zeros(count, dtype=torch.long), torch.zeros(count, dtype=torch.long)
    opening_active = active.clone()
    invalid = torch.zeros(count, 2, dtype=torch.long)
    records = [None] * count
    started = perf_counter()
    with VecEnv.from_states(states, device="cpu", threads=1, auto_reset=False,
                            episode_ticks=max_ticks) as env, torch.inference_mode():
        batch = GameBatch.from_backend(env, "cpu")
        for role in range(2):
            histories[role].reset(encoder.encode(batch, ids if role == 0 else 1 - ids))
        while bool(active.any()):
            batch = GameBatch.from_backend(env, "cpu")
            rows = torch.where(active)[0]
            joint = torch.zeros(count, 2, dtype=torch.long)
            for role, model in enumerate((learner, opponent)):
                seats = ids if role == 0 else 1 - ids
                obs = _local_inputs(batch, seats, encoder, histories[role], space)[active]
                actions = torch.multinomial(model.action_probabilities(obs), 1,
                                            generator=generators[role])
                actions = space.validate(actions, obs["action_mask"]).reshape(-1)
                joint[rows, seats[rows]] = actions
                if role == 0:
                    waiting = torch.zeros(count, dtype=torch.bool)
                    waiting[rows] = actions == 0
                    opening += (opening_active & waiting).long()
                    opening_active &= waiting
            decisions += active.long()
            result = env.step(joint.numpy())
            env.check_errors()
            moves = torch.as_tensor(result.moves)
            invalid += ((moves[:, :, 0] != 0) & (moves[:, :, 1] == 0) & active[:, None]).long()
            for name, values in metrics.update(moves[torch.arange(count), ids]).items():
                totals[name] += values * active
            for lane in rows.tolist():
                terminal, truncated = map(bool, result.done[lane])
                if terminal or truncated:
                    seat = specs[lane].learner_seat
                    winner = int(env.state["winner"][lane])
                    records[lane] = {
                        **asdict(specs[lane]),
                        "outcome": "draw" if truncated else ("win" if winner == seat else "loss"),
                        "terminated": terminal, "truncated": truncated,
                        "winner_seat": winner if terminal else None,
                        "ticks": int(env.state["tick"][lane]),
                        "learner_invalid_actions": int(invalid[lane, seat]),
                        "opponent_invalid_actions": int(invalid[lane, 1 - seat]),
                        "opening_planned_wait_ticks": int(opening[lane]),
                        "action_metrics": {k: float(v[lane]) / int(decisions[lane]) for k, v in totals.items()},
                        "learner_final_stats": dict(zip(("army", "territory", "cities"), map(int, env.stats[lane, seat]))),
                        "opponent_final_stats": dict(zip(("army", "territory", "cities"), map(int, env.stats[lane, 1 - seat]))),
                    }
                    active[lane] = False
            batch = GameBatch.from_backend(env, "cpu")
            for role in range(2):
                histories[role].push(encoder.encode(batch, ids if role == 0 else 1 - ids))
    assert all(r is not None for r in records)
    return records, {"wall_seconds": perf_counter() - started,
                     "environment_ticks": sum(r["ticks"] for r in records)}


def summary(games):
    result = aggregate_games(games)
    if all("opening_planned_wait_ticks" in g for g in games):
        result["mean_opening_planned_wait_ticks"] = sum(g["opening_planned_wait_ticks"] for g in games) / len(games)
        result["opening_wait_at_least_100_fraction"] = sum(g["opening_planned_wait_ticks"] >= 100 for g in games) / len(games)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("history", "sampling"))
    parser.add_argument("--policy", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--opponent", nargs="+")
    parser.add_argument("--greedy-report", action="append", default=[], help="NAME=EXISTING_REPORT")
    parser.add_argument("--maps", type=int, default=100)
    parser.add_argument("--map-seed", type=int, default=271828)
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=[1001, 1002, 1003])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-ticks", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if min(args.maps, args.batch_size, args.max_ticks) < 1 or min(args.sampling_seeds) < 0:
        raise ValueError("positive sizes and nonnegative sampling seeds required")
    paths = {name: Path(path).resolve() for name, path in (v.split("=", 1) for v in args.policy)}
    greedy = {name: Path(path).resolve() for name, path in (v.split("=", 1) for v in args.greedy_report)}
    configs = {name: json.loads(path.with_name("run_config.json").read_text()) for name, path in paths.items()}
    first = next(iter(configs.values()))
    if any(c["task"] != first["task"] or c["runner"]["policy"] != first["runner"]["policy"] for c in configs.values()):
        raise ValueError("this check requires one frozen MDP and policy architecture")
    task = task_config(first["task"])
    config = MapConfig(task.width, task.height, 2, task.mountain_density, task.city_density)
    schedule = make_schedule(maps_per_ai=args.maps, seed=args.map_seed,
                             opponents=("aggressive",) if args.mode == "history" else KINDS)
    maps = {s.map_index: generate_map(config, s.map_seed)
            for s in {s.map_index: s for s in schedule}.values()}
    models = {name: load_policy(path, configs[name], list(maps.values()), task) for name, path in paths.items()}
    hashes = {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "protocol.json", {
        "mode": args.mode, "paths": {k: str(v) for k, v in paths.items()}, "checkpoint_sha256": hashes,
        "map_seed": args.map_seed, "maps": args.maps, "batch_size": args.batch_size,
        "max_ticks": args.max_ticks, "clock_normalization_ticks": task.max_episode_steps,
        "sampling_seeds": args.sampling_seeds, "device": "cpu", "cpu_threads": 1,
        "decision_timing": "joint_pre_growth_then_absolute_seat_order",
        "config_sha256": {k: sha256(p.with_name('run_config.json').read_bytes()).hexdigest() for k, p in paths.items()},
    })
    combinations = [(c, o, args.sampling_seeds[0]) for c in args.candidate for o in args.opponent] if args.mode == "history" else [
        (c, None, s) for c in args.candidate for s in args.sampling_seeds]
    if args.mode == "sampling":
        identity = ("game_id", "opponent", "map_index", "map_seed", "controller_seed", "learner_seat")
        for c in args.candidate:
            original = json.loads(greedy[c].read_text())
            assert not original["stochastic"] and original["checkpoint_sha256"] == hashes[c]
            assert original["max_ticks"] == args.max_ticks and original["batch_size"] == args.batch_size
            assert len(original["games"]) == len(schedule)
            assert all(tuple(g[k] for k in identity) == tuple(asdict(s)[k] for k in identity)
                       for g, s in zip(original["games"], schedule))
            write_json(args.output / f"{c}_greedy_reused.json", {
                "checkpoint_sha256": hashes[c], "source_report": str(greedy[c]),
                "source_report_sha256": sha256(greedy[c].read_bytes()).hexdigest(),
                "reused_existing_games": True, "new_games": 0, "by_ai": original["by_ai"],
                "overall": original["overall"], "games": original["games"],
            })
    for c, o, seed in combinations:
        name = f"{c}_vs_{o}" if o else f"{c}_stochastic_{seed}"
        destination = args.output / f"{name}.json"
        if destination.exists():
            raise FileExistsError(destination)
        games, timings = [], []
        for start in range(0, len(schedule), args.batch_size):
            specs = [replace(s, opponent=o) for s in schedule[start:start + args.batch_size]] if o else schedule[start:start + args.batch_size]
            states = [maps[s.map_index].copy() for s in specs]
            batch_seed = int(np.random.SeedSequence([seed, start]).generate_state(1, dtype=np.uint32)[0])
            if o:
                completed, timing = nn_batch(specs, states, models[c], models[o], task,
                                              max_ticks=args.max_ticks, seed=batch_seed)
            else:
                recorder = OpeningWaits()
                completed, timing = battle_batch(specs, states, SampledPolicy(models[c], batch_seed), task,
                                                  max_ticks=args.max_ticks, cpu_threads=1, recorder=recorder)
                for g in completed:
                    g["opening_planned_wait_ticks"] = recorder.counts[g["game_id"]]
            games.extend(completed)
            timings.append(timing)
            print(f"{name}: {len(games)}/{len(schedule)} completed", flush=True)
        assert len(games) == len(schedule) and len({g['game_id'] for g in games}) == len(schedule)
        # Legal joint decisions can become invalid after the earlier absolute
        # seat captures a source cell; preserve those native results in reports.
        report = {"candidate": c, "opponent": o or "builtin", "candidate_sha256": hashes[c],
                  "opponent_sha256": hashes.get(o), "stochastic": True,
                  "sampling_seed": seed, "scheduled_games": len(schedule), "completed_games": len(games),
                  "overall": summary(games), "timings": timings, "games": games}
        if not o:
            report["by_ai"] = {k: summary([g for g in games if g["opponent"] == k]) for k in KINDS}
        write_json(destination, report)
        print(json.dumps({"name": name, "overall": report["overall"]}), flush=True)


if __name__ == "__main__":
    main()
