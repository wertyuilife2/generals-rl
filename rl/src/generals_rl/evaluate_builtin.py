"""Balanced held-out 1v1 matches against the four original Generals controllers.

Both players choose from LOCAL observations before growth, then native CPUEnv
executes the joint tick in absolute seat order. This matches training; the GUI's
Runner instead observes/chooses separately after growth at each player's turn.
The original controllers from generals.controllers.make_controllers are used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tensordict import TensorDict

from generals import MapConfig, VecEnv
from generals.controllers import AI_KINDS, make_controllers
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.state import Action, MoveResult
from rsl_rl.modules import ActorCriticMoECTS

from .mdp import EncodedActionSpace, GameBatch, LocalObservationEncoder, ObservationHistory
from .config import seed_everything, task_config


@dataclass(frozen=True)
class GameSpec:
    game_id: int
    opponent: str
    map_index: int
    map_seed: int
    controller_seed: int
    learner_seat: int


def make_schedule(*, maps_per_ai=50, seed=1000042, opponents=AI_KINDS):
    """Pair the same held-out maps across both seats and all opponent kinds."""
    if maps_per_ai < 1 or seed < 0:
        raise ValueError("maps_per_ai must be positive and seed nonnegative")
    opponents = tuple(opponents)
    if not opponents or len(set(opponents)) != len(opponents) or any(k not in AI_KINDS for k in opponents):
        raise ValueError("opponents must be a nonempty unique subset of the four built-in AI kinds")
    schedule = []
    for kind in opponents:
        for index in range(maps_per_ai):
            map_seed = int(np.random.SeedSequence([seed, index, 0]).generate_state(1, dtype=np.uint64)[0])
            controller_seed = int(np.random.SeedSequence([seed, index, 1]).generate_state(1, dtype=np.uint64)[0])
            for seat in (0, 1):
                schedule.append(GameSpec(len(schedule), kind, index, map_seed, controller_seed, seat))
    return schedule


def decode_move(row, player_id, action):
    """Convert a borrowed native move row to the original controller's result type."""
    if not row[0]:
        return None
    reasons = {1: "WAIT", 2: "MOVED", 3: "OUT_OF_BOUNDS", 4: "NOT_OWNER", 5: "MOUNTAIN"}
    return MoveResult(
        player_id=player_id, action=action, valid=bool(row[1]),
        reason=reasons[int(row[8])], source=int(row[2]), target=int(row[3]),
        moved=int(row[4]), captured=bool(row[5]),
        eliminated=None if row[6] < 0 else int(row[6]), target_owned=bool(row[7]),
    )


def wilson_interval(wins, games):
    """Two-sided 95% Wilson interval; an empty decisive sample is undefined."""
    if not 0 <= wins <= games:
        raise ValueError("wins must be between zero and the number of games")
    if games == 0:
        return None
    z = 1.959963984540054
    fraction = wins / games
    denominator = 1 + z * z / games
    center = (fraction + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / games + z * z / (4 * games * games)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def aggregate_games(games):
    wins = sum(game["outcome"] == "win" for game in games)
    losses = sum(game["outcome"] == "loss" for game in games)
    draws = sum(game["outcome"] == "draw" for game in games)
    count, decisive = len(games), wins + losses
    return {
        "games": count, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / count if count else None,
        "decisive_win_rate": wins / decisive if decisive else None,
        "draw_rate": draws / count if count else None,
        "win_rate_wilson95": wilson_interval(wins, count),
        "decisive_win_rate_wilson95": wilson_interval(wins, decisive),
        "mean_ticks": sum(game["ticks"] for game in games) / count if count else None,
    }


def _local_inputs(batch, player_ids, encoder, history, action_space):
    # No privileged critic key is ever passed into the deployed student.
    return TensorDict({
        "policy": history.flattened(),
        "single_obs": encoder.encode(batch, player_ids),
        "action_mask": action_space.mask(batch, player_ids),
    }, batch_size=[batch.num_envs], device=batch.device)


def _load_student(checkpoint, config, example_obs, device):
    policy_config = deepcopy(config["runner"]["policy"])
    class_name = policy_config.pop("class_name")
    if class_name not in ("ActorCriticMoECTS", "rsl_rl.modules.ActorCriticMoECTS"):
        raise ValueError("built-in evaluation currently requires ActorCriticMoECTS")
    if policy_config.get("action_distribution") != "categorical":
        raise ValueError("Generals built-in evaluation requires categorical encoded actions")
    # Construction only inspects feature dimensions. Use an all-zero placeholder
    # for the critic shape; no true privileged map is needed to load the student.
    shape_obs = example_obs.clone()
    shape_obs["critic"] = torch.zeros_like(example_obs["single_obs"])
    policy = ActorCriticMoECTS(
        shape_obs, deepcopy(config["runner"]["obs_groups"]),
        example_obs["action_mask"].shape[-1], **policy_config,
    ).to(device)
    saved = torch.load(checkpoint, weights_only=False, map_location=device)
    policy.load_state_dict(saved["model_state_dict"])
    policy.eval()
    return policy


def battle_batch(specs, states, policy, training_task, *, max_ticks=2000,
                 device="cpu", stochastic=False, cpu_threads=1, recorder=None):
    """Complete every fixed lane once; no autorestart or first-finished sampling.

    ``policy`` may be a small injected student interface in tests. Normal callers
    pass the checkpoint's ActorCriticMoECTS. Controller factory and game rules
    are the original environment implementations.
    """
    if not specs or len(specs) != len(states) or max_ticks < 1:
        raise ValueError("provide matching nonempty specs/states and a positive tick limit")
    if any(state.player_count != 2 or state.tick != 0 or state.terminated for state in states):
        raise ValueError("built-in matches require fresh nonterminal two-player states")
    started = perf_counter()
    device = torch.device(device)
    timing = {"observation_seconds": 0.0, "policy_seconds": 0.0,
              "opponent_seconds": 0.0, "simulator_seconds": 0.0}
    count = len(specs)
    player_ids = torch.tensor([spec.learner_seat for spec in specs], dtype=torch.long, device=device)
    action_space = EncodedActionSpace(training_task.width, training_task.height)
    # The rollout limit does not change the model's trained clock normalization.
    encoder = LocalObservationEncoder(training_task.max_episode_steps)
    history = ObservationHistory(training_task.history_length)
    controllers = []
    for spec in specs:
        kinds = ["random", "random"]  # The learner's factory slot is unused.
        kinds[1 - spec.learner_seat] = spec.opponent
        controllers.append(make_controllers(kinds, seed=spec.controller_seed)[1 - spec.learner_seat])
    records = [None] * count
    invalid = np.zeros((count, 2), dtype=np.int64)
    decisions = np.zeros(count, dtype=np.int64)
    active = np.ones(count, dtype=bool)
    if recorder is not None:
        recorder.on_start(specs, states)
    with VecEnv.from_states(states, device="cpu", threads=cpu_threads, auto_reset=False, episode_ticks=max_ticks) as env:
        history.reset(encoder.encode(GameBatch.from_backend(env, device), player_ids))
        with torch.inference_mode():
            while active.any():
                tick_start = perf_counter()
                batch = GameBatch.from_backend(env, device)
                observations = _local_inputs(batch, player_ids, encoder, history, action_space)
                active_tensor = torch.as_tensor(active.copy(), device=device)
                selected_obs = observations[active_tensor]
                timing["observation_seconds"] += perf_counter() - tick_start
                tick_start = perf_counter()
                if stochastic:
                    probabilities = policy.action_probabilities(selected_obs)
                    selected_actions = torch.multinomial(probabilities, 1)
                else:
                    selected_actions = policy.act_inference(selected_obs)
                selected_actions = action_space.validate(selected_actions, selected_obs["action_mask"])
                learner_actions = selected_actions.cpu().numpy()
                timing["policy_seconds"] += perf_counter() - tick_start
                tick_start = perf_counter()
                current_states = env.to_states()
                joint = np.zeros((count, 2), dtype=np.int64)
                actions_by_lane = {}
                for offset, lane in enumerate(np.flatnonzero(active)):
                    seat = specs[lane].learner_seat
                    opponent_action = controllers[lane].act(observe(current_states[lane], 1 - seat, "local"))
                    if not isinstance(opponent_action, Action):
                        raise TypeError("built-in controller must return the original Action type")
                    learner_action = Action.decode(int(learner_actions[offset]))
                    actions = [Action.wait(), Action.wait()]
                    actions[seat], actions[1 - seat] = learner_action, opponent_action
                    joint[lane] = [action.encode() for action in actions]
                    actions_by_lane[lane] = actions
                    decisions[lane] += 1
                timing["opponent_seconds"] += perf_counter() - tick_start
                tick_start = perf_counter()
                result = env.step(joint)
                env.check_errors()
                timing["simulator_seconds"] += perf_counter() - tick_start
                tick_start = perf_counter()
                if recorder is not None:
                    recorded_states = env.to_states()
                    for lane in np.flatnonzero(active):
                        recorder.on_tick(specs[lane], tuple(actions_by_lane[lane]), recorded_states[lane],
                                         result.moves[lane].copy(), result.done[lane].copy())
                for lane in np.flatnonzero(active):
                    seat = specs[lane].learner_seat
                    opponent = controllers[lane]
                    for pid in (0, 1):
                        row = result.moves[lane, pid]
                        if row[0] and not row[1]:
                            invalid[lane, pid] += 1
                        move = decode_move(row, pid, actions_by_lane[lane][pid])
                        if move is not None and pid == 1 - seat:
                            opponent.on_result(move)
                        if move is not None and move.eliminated == 1 - seat and hasattr(opponent, "eliminate"):
                            opponent.eliminate()
                    terminal, truncated = map(bool, result.done[lane])
                    if terminal or truncated:
                        winner = int(env.state["winner"][lane])
                        outcome = "draw" if truncated else ("win" if winner == seat else "loss")
                        records[lane] = {
                            **asdict(specs[lane]), "outcome": outcome,
                            "terminated": terminal, "truncated": truncated,
                            "winner_seat": winner if terminal else None,
                            "ticks": int(env.state["tick"][lane]),
                            "learner_invalid_actions": int(invalid[lane, seat]),
                            "opponent_invalid_actions": int(invalid[lane, 1 - seat]),
                            "learner_final_stats": dict(zip(("army", "territory", "cities"), map(int, env.stats[lane, seat]))),
                            "opponent_final_stats": dict(zip(("army", "territory", "cities"), map(int, env.stats[lane, 1 - seat]))),
                        }
                        active[lane] = False
                timing["opponent_seconds"] += perf_counter() - tick_start
                tick_start = perf_counter()
                history.push(encoder.encode(GameBatch.from_backend(env, device), player_ids))
                timing["observation_seconds"] += perf_counter() - tick_start
    if any(record is None for record in records):
        raise RuntimeError("a scheduled built-in match did not finish")
    timing.update(wall_seconds=perf_counter() - started,
                  environment_ticks=sum(record["ticks"] for record in records),
                  student_decisions=int(decisions.sum()))
    return records, timing


def evaluate_builtin(checkpoint, *, maps_per_ai=50, seed=1000042, opponents=AI_KINDS,
                     batch_size=128, max_ticks=2000, device="cpu", stochastic=False,
                     cpu_threads=1, output=None):
    """Evaluate every scheduled game, paired by map and balanced by learner seat."""
    if batch_size < 1 or max_ticks < 1:
        raise ValueError("batch size and maximum ticks must be positive")
    opponents = tuple(opponents)
    started = perf_counter()
    checkpoint = Path(checkpoint).resolve()
    config = json.loads(checkpoint.with_name("run_config.json").read_text())
    training_task = task_config(config["task"])
    if training_task.players != 2:
        raise ValueError("1v1 evaluation requires a checkpoint trained with players=2")
    seed_everything(seed)
    torch.set_num_threads(config.get("torch_threads", 1))
    schedule = make_schedule(maps_per_ai=maps_per_ai, seed=seed, opponents=opponents)
    map_config = MapConfig(training_task.width, training_task.height, 2,
                           training_task.mountain_density, training_task.city_density)
    maps = {index: generate_map(map_config, spec.map_seed)
            for index, spec in {spec.map_index: spec for spec in schedule}.items()}
    games, timings = [], []
    policy = None
    for start in range(0, len(schedule), batch_size):
        specs = schedule[start:start + batch_size]
        states = [maps[spec.map_index].copy() for spec in specs]
        if policy is None:
            # Build feature shapes from the same encoder as deployment. The
            # temporary CPU environment is closed before the real matches run.
            with VecEnv.from_states(states, device="cpu", threads=cpu_threads) as example_env:
                batch = GameBatch.from_backend(example_env, device)
                pids = torch.tensor([spec.learner_seat for spec in specs], device=device)
                encoder = LocalObservationEncoder(training_task.max_episode_steps)
                history = ObservationHistory(training_task.history_length)
                history.reset(encoder.encode(batch, pids))
                obs = _local_inputs(batch, pids, encoder, history,
                                    EncodedActionSpace(training_task.width, training_task.height))
                policy = _load_student(checkpoint, config, obs, device)
        completed, timing = battle_batch(specs, states, policy, training_task,
                                        max_ticks=max_ticks, device=device,
                                        stochastic=stochastic, cpu_threads=cpu_threads)
        games.extend(completed)
        timings.append(timing)
        print(f"Built-in evaluation: {len(games)}/{len(schedule)} games completed", flush=True)
    if len(games) != len(schedule) or {game["game_id"] for game in games} != {spec.game_id for spec in schedule}:
        raise RuntimeError("completed games do not match the fixed schedule")
    totals = {key: sum(timing[key] for timing in timings) for key in timings[0]}
    totals["environment_ticks_per_second"] = totals["environment_ticks"] / totals["wall_seconds"]
    totals["student_decisions_per_second"] = totals["student_decisions"] / totals["wall_seconds"]
    totals["games_per_second"] = len(games) / totals["wall_seconds"]
    totals["total_seconds_including_setup"] = perf_counter() - started
    report = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "policy": "student_local", "opponent_implementation": "generals.controllers.make_controllers",
        "decision_timing": "joint_pre_growth_then_absolute_seat_order",
        "training_opponent": training_task.opponent,
        "seed": seed, "maps_per_ai": maps_per_ai, "scheduled_games": len(schedule),
        "completed_games": len(games), "batch_size": batch_size, "device": str(device),
        "cpu_threads": cpu_threads, "max_ticks": max_ticks,
        "clock_normalization_ticks": training_task.max_episode_steps,
        "stochastic": stochastic, "map_config": asdict(map_config),
        "confidence_interval": "95% Wilson binomial interval; games are paired by map, interval treats games as independent",
        "overall": aggregate_games(games),
        "by_ai": {kind: aggregate_games([g for g in games if g["opponent"] == kind]) for kind in opponents},
        "by_seat": {str(seat): aggregate_games([g for g in games if g["learner_seat"] == seat]) for seat in (0, 1)},
        "by_ai_and_seat": {kind: {str(seat): aggregate_games([
            g for g in games if g["opponent"] == kind and g["learner_seat"] == seat])
            for seat in (0, 1)} for kind in opponents},
        "timing": totals, "games": games,
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--maps-per-ai", type=int, default=50, help="each map is played from both learner seats")
    parser.add_argument("--seed", type=int, default=1000042)
    parser.add_argument("--opponents", nargs="+", choices=AI_KINDS, default=list(AI_KINDS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-ticks", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.checkpoint.with_name("builtin_evaluation.json")
    report = evaluate_builtin(args.checkpoint, maps_per_ai=args.maps_per_ai, seed=args.seed,
                              opponents=args.opponents, batch_size=args.batch_size,
                              max_ticks=args.max_ticks, device=args.device,
                              stochastic=args.stochastic, cpu_threads=args.cpu_threads, output=output)
    print(json.dumps({key: value for key, value in report.items() if key != "games"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
