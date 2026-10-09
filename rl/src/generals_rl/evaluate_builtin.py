"""Balanced held-out 1v1 matches against the four original Generals controllers.

Both players choose from LOCAL observations before growth, then native CPUEnv
executes the joint tick in absolute seat order. This matches training; the GUI's
Runner instead observes/chooses separately after growth at each player's turn.
The original controllers from generals_baselines.make_controllers are used.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from generals import MapConfig, VecEnv
from generals_baselines import AI_KINDS, make_controllers
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.recording import RULES_VERSION, state_digest, state_to_dict
from generals.core.state import Action, MoveResult

from .mdp import (ActionMetrics, GameBatch, ObservationHistory,
                  make_action_space, make_observation_encoder)
from .mdp.knowledge import KnowledgeState
from .config import seed_everything, task_config
from .baseline_pool import BaselinePool
from .inference import load_student as _load_student, local_inputs as _local_inputs


@dataclass(frozen=True)
class GameSpec:
    game_id: int
    opponent: str
    map_index: int
    map_seed: int
    controller_seed: int
    learner_seat: int


class SelectedGameTraces:
    """Keep one actual win/loss per AI, buffering only the current match batch.

    No full post-tick states or policy inputs are retained. Core-format traces
    contain the initial state, actual joint actions and native state digests.
    Selection is by game_id, not completion speed or a subsequent policy replay.
    """

    protocol = "recorded_joint_actions_core_v1"

    def __init__(self, directory, *, max_ticks, opponents, metadata):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_ticks = max_ticks
        self.opponents = tuple(opponents)
        self.metadata = dict(metadata)
        self.selected = {}
        self.candidates = {}

    def needs_batch(self, specs):
        return any((spec.opponent, outcome) not in self.selected
                   for spec in specs for outcome in ("win", "loss"))

    def on_start(self, specs, states):
        self.candidates = {}
        for spec, state in zip(specs, states, strict=True):
            if all((spec.opponent, outcome) in self.selected for outcome in ("win", "loss")):
                continue
            self.candidates[spec.game_id] = {
                "spec": spec, "initial_state": state_to_dict(state),
                "initial_digest": state_digest(state), "ticks": 0,
                "actions": np.empty((self.max_ticks, 2, 4), dtype=np.int64),
                "digests": np.empty((self.max_ticks, 32), dtype=np.uint8),
            }

    def on_tick(self, spec, actions, state, moves, done):
        entry = self.candidates.get(spec.game_id)
        if entry is None:
            return
        index = entry["ticks"]
        if state.tick != index + 1 or index >= self.max_ticks:
            raise RuntimeError("selected trace skipped or repeated a tick")
        entry["actions"][index] = [[int(a.kind), a.source, int(a.direction), int(a.mode)] for a in actions]
        entry["digests"][index] = np.frombuffer(bytes.fromhex(state_digest(state)), dtype=np.uint8)
        entry["ticks"] += 1

    def finish_batch(self, games):
        for game in sorted(games, key=lambda game: game["game_id"]):
            key = game["opponent"], game["outcome"]
            if key[1] not in ("win", "loss") or key in self.selected:
                continue
            entry = self.candidates[game["game_id"]]
            if entry["ticks"] != game["ticks"]:
                raise RuntimeError("selected trace does not cover the evaluated game")
            path = self.directory / f"{key[0]}_{key[1]}.trace.jsonl"
            metadata = {**self.metadata, "game": asdict(entry["spec"]), "match": game,
                        "trace_protocol": self.protocol,
                        "decision_timing": "joint_pre_growth_then_absolute_seat_order",
                        "spectator_full_state_is_not_policy_input": True}
            with path.open("x", encoding="utf-8") as stream:
                def write(row):
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                write({"type": "header", "rules_version": RULES_VERSION,
                       "state": entry["initial_state"], "digest": entry["initial_digest"],
                       "metadata": metadata})
                for index in range(entry["ticks"]):
                    write({"type": "tick", "tick": index + 1,
                           "actions": entry["actions"][index].tolist(),
                           "digest": entry["digests"][index].tobytes().hex()})
            self.selected[key] = {
                "game_id": game["game_id"], "opponent": key[0], "outcome": key[1],
                "learner_seat": game["learner_seat"], "ticks": game["ticks"],
                "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
                "checkpoint_sha256": self.metadata["checkpoint_sha256"], "protocol": self.protocol,
            }
        self.candidates.clear()

    def describe(self):
        keys = [(kind, outcome) for kind in self.opponents for outcome in ("loss", "win")]
        return {"protocol": self.protocol, "selection": "first_game_id_win_and_loss_per_ai",
                "cases": [self.selected[key] for key in keys if key in self.selected],
                "missing": [{"opponent": key[0], "outcome": key[1]} for key in keys if key not in self.selected]}


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
    result = {
        "games": count, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / count if count else None,
        "decisive_win_rate": wins / decisive if decisive else None,
        "draw_rate": draws / count if count else None,
        "win_rate_wilson95": wilson_interval(wins, count),
        "decisive_win_rate_wilson95": wilson_interval(wins, decisive),
        "mean_ticks": sum(game["ticks"] for game in games) / count if count else None,
    }
    if games and all("action_metrics" in game for game in games):
        ticks = sum(game["ticks"] for game in games)
        result["action_metrics"] = {
            name: sum(game["action_metrics"][name] * game["ticks"] for game in games) / ticks
            for name in ActionMetrics.names
        }
    return result



def _initial_knowledge(batch, player_ids, training_task):
    """One fresh player-local memory per fixed match lane, never a policy cache."""
    if not training_task.persistent_knowledge:
        return None
    knowledge = KnowledgeState(batch.num_envs, batch.width, batch.height,
                               batch.players, batch.device)
    knowledge.update(batch, player_ids)
    return knowledge



def _inference_metadata(policy, device):
    """Separate stored parameters from the CUDA forward's autocast precision.

    MapPPO returns float32 logits/probabilities after its mixed-precision forward.
    ``inference_dtype`` describes that forward, not every operation or output.
    """
    parameter_dtype = str(next(policy.parameters()).dtype).removeprefix("torch.")
    autocast_bfloat16 = getattr(policy, "amp_bfloat16", False) and torch.device(device).type == "cuda"
    return {"parameter_dtype": parameter_dtype,
            "inference_dtype": "bfloat16" if autocast_bfloat16 else parameter_dtype}


def battle_batch(specs, states, policy, training_task, *, max_ticks=2000,
                 device="cpu", stochastic=True, sampling_seed=1000042,
                 cpu_threads=1, baseline_workers=1, recorder=None, _baseline_pool=None):
    """Complete every fixed lane once; no autorestart or first-finished sampling.

    ``policy`` may be a small injected student interface in tests. Normal callers
    pass the checkpoint's ActorCriticMoECTS. Controller factory and game rules
    are the original environment implementations.
    """
    if not specs or len(specs) != len(states) or max_ticks < 1:
        raise ValueError("provide matching nonempty specs/states and a positive tick limit")
    if not isinstance(stochastic, bool):
        raise TypeError("stochastic must be boolean")
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int) or sampling_seed < 0:
        raise ValueError("sampling_seed must be a nonnegative integer")
    if isinstance(baseline_workers, bool) or not isinstance(baseline_workers, int) or baseline_workers < 1:
        raise ValueError("baseline_workers must be a positive integer")
    if any(state.player_count != 2 or state.tick != 0 or state.terminated for state in states):
        raise ValueError("built-in matches require fresh nonterminal two-player states")
    started = perf_counter()
    device = torch.device(device)
    timing = {"observation_seconds": 0.0, "policy_seconds": 0.0,
              "opponent_seconds": 0.0, "simulator_seconds": 0.0}
    count = len(specs)
    # A game's random stream is independent of batch size, other games ending,
    # policy construction and the global training RNG. Selected demos reuse it.
    game_seeds = [int(np.random.SeedSequence([sampling_seed, spec.game_id])
                      .generate_state(1, dtype=np.uint64)[0]) for spec in specs]
    generators = [np.random.default_rng(seed) for seed in game_seeds] if stochastic else None
    player_ids = torch.tensor([spec.learner_seat for spec in specs], dtype=torch.long, device=device)
    action_space = make_action_space(training_task)
    action_metrics = ActionMetrics(count, device)
    action_totals = {name: torch.zeros(count, device=device) for name in ActionMetrics.names}
    # The rollout limit does not change the model's trained clock normalization.
    encoder = make_observation_encoder(training_task)
    history = ObservationHistory(training_task.history_length)
    controllers = []
    for spec in specs if baseline_workers == 1 else ():
        kinds = ["random", "random"]  # The learner's factory slot is unused.
        kinds[1 - spec.learner_seat] = spec.opponent
        controllers.append(make_controllers(kinds, seed=spec.controller_seed)[1 - spec.learner_seat])
    records = [None] * count
    invalid = np.zeros((count, 2), dtype=np.int64)
    decisions = np.zeros(count, dtype=np.int64)
    active = np.ones(count, dtype=bool)
    if recorder is not None:
        recorder.on_start(specs, states)
    pool_context = (nullcontext(_baseline_pool) if _baseline_pool is not None else
                    BaselinePool(specs, workers=baseline_workers) if baseline_workers > 1 else nullcontext())
    with pool_context as baseline_pool, \
            VecEnv.from_states(states, device="cpu", threads=cpu_threads,
                               auto_reset=False, episode_ticks=max_ticks) as env:
        initial_batch = GameBatch.from_backend(env, device)
        history.reset(encoder.encode(initial_batch, player_ids))
        knowledge = _initial_knowledge(initial_batch, player_ids, training_task)
        with torch.inference_mode():
            while active.any():
                tick_start = perf_counter()
                batch = GameBatch.from_backend(env, device)
                observations = _local_inputs(batch, player_ids, encoder, history, action_space, knowledge)
                active_tensor = torch.as_tensor(active.copy(), device=device)
                selected_obs = observations[active_tensor]
                timing["observation_seconds"] += perf_counter() - tick_start
                tick_start = perf_counter()
                if stochastic:
                    probabilities = policy.action_probabilities(selected_obs)
                    uniforms = np.array([generators[lane].random() for lane in np.flatnonzero(active)])
                    # Inverse CDF samples the same categorical distribution while
                    # keeping one independent stream per lane. Double precision
                    # and normalization keep zero-mass masked actions unselected.
                    cdf = probabilities.double().cumsum(-1)
                    cdf = cdf / cdf[:, -1:]
                    selected_actions = torch.searchsorted(
                        cdf, torch.as_tensor(uniforms, device=device)[:, None], right=True)
                else:
                    selected_actions = policy.act_inference(selected_obs)
                selected_actions = action_space.validate(selected_actions, selected_obs["action_mask"])
                learner_actions = selected_actions.cpu().numpy()
                timing["policy_seconds"] += perf_counter() - tick_start
                tick_start = perf_counter()
                current_states = env.to_states()
                joint = np.zeros((count, 2), dtype=np.int64)
                actions_by_lane = {}
                opponent_actions = (baseline_pool.act({
                    lane: observe(current_states[lane], 1 - specs[lane].learner_seat, "local")
                    for lane in np.flatnonzero(active)}) if baseline_pool is not None else None)
                for offset, lane in enumerate(np.flatnonzero(active)):
                    seat = specs[lane].learner_seat
                    opponent_action = (opponent_actions[lane] if opponent_actions is not None else
                                       controllers[lane].act(observe(current_states[lane], 1 - seat, "local")))
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
                learner_moves = torch.as_tensor(result.moves, device=device)[
                    torch.arange(count, device=device), player_ids]
                for name, values in action_metrics.update(learner_moves).items():
                    action_totals[name] += values * active_tensor
                timing["simulator_seconds"] += perf_counter() - tick_start
                tick_start = perf_counter()
                if recorder is not None:
                    recorded_states = env.to_states()
                    for lane in np.flatnonzero(active):
                        recorder.on_tick(specs[lane], tuple(actions_by_lane[lane]), recorded_states[lane],
                                         result.moves[lane].copy(), result.done[lane].copy())
                feedback = {}
                for lane in np.flatnonzero(active):
                    seat = specs[lane].learner_seat
                    opponent = controllers[lane] if baseline_pool is None else None
                    moves = []
                    for pid in (0, 1):
                        row = result.moves[lane, pid]
                        if row[0] and not row[1]:
                            invalid[lane, pid] += 1
                        move = decode_move(row, pid, actions_by_lane[lane][pid])
                        moves.append(move)
                        if opponent is not None and move is not None and pid == 1 - seat:
                            opponent.on_result(move)
                        if opponent is not None and move is not None and move.eliminated == 1 - seat and hasattr(opponent, "eliminate"):
                            opponent.eliminate()
                    if baseline_pool is not None:
                        feedback[lane] = tuple(moves)
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
                            "action_metrics": {name: float(value[lane]) / decisions[lane]
                                               for name, value in action_totals.items()},
                            "learner_final_stats": dict(zip(("army", "territory", "cities"), map(int, env.stats[lane, seat]))),
                            "opponent_final_stats": dict(zip(("army", "territory", "cities"), map(int, env.stats[lane, 1 - seat]))),
                        }
                        if stochastic:
                            records[lane]["sampling_seed"] = game_seeds[lane]
                        active[lane] = False
                if baseline_pool is not None:
                    baseline_pool.feedback(feedback)
                timing["opponent_seconds"] += perf_counter() - tick_start
                tick_start = perf_counter()
                completed_batch = GameBatch.from_backend(env, device)
                history.push(encoder.encode(completed_batch, player_ids))
                if knowledge is not None:
                    # Include lanes ending on this tick, but do not re-observe
                    # previously finished lanes when other games keep running.
                    knowledge.update(completed_batch, player_ids, mask=active_tensor)
                timing["observation_seconds"] += perf_counter() - tick_start
    if any(record is None for record in records):
        raise RuntimeError("a scheduled built-in match did not finish")
    timing.update(wall_seconds=perf_counter() - started,
                  environment_ticks=sum(record["ticks"] for record in records),
                  student_decisions=int(decisions.sum()))
    return records, timing


def evaluate_builtin(checkpoint, *, maps_per_ai=50, seed=1000042, opponents=AI_KINDS,
                     batch_size=128, max_ticks=2000, device="cpu", stochastic=True,
                     sampling_seed=None, cpu_threads=1, baseline_workers=1,
                     output=None, trace_directory=None):
    """Evaluate every scheduled game, paired by map and balanced by learner seat."""
    if batch_size < 1 or max_ticks < 1:
        raise ValueError("batch size and maximum ticks must be positive")
    if isinstance(baseline_workers, bool) or not isinstance(baseline_workers, int) or baseline_workers < 1:
        raise ValueError("baseline_workers must be a positive integer")
    opponents = tuple(opponents)
    sampling_seed = seed if sampling_seed is None else sampling_seed
    started = perf_counter()
    checkpoint = Path(checkpoint).resolve()
    config = json.loads(checkpoint.with_name("run_config.json").read_text())
    cudnn_allow_tf32 = config.get("cudnn_allow_tf32")
    if cudnn_allow_tf32 is not None and not isinstance(cudnn_allow_tf32, bool):
        raise TypeError("cudnn_allow_tf32 must be boolean or null")
    cudnn_benchmark = config.get("cudnn_benchmark")
    if cudnn_benchmark is not None and not isinstance(cudnn_benchmark, bool):
        raise TypeError("cudnn_benchmark must be boolean or null")
    device = torch.device(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(config.get("matmul_precision") or "highest")
        if cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = cudnn_benchmark
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
    traces = None
    baseline_lifecycle_seconds = 0.0
    with ExitStack() as worker_lifetime:
        baseline_pool = None
        for start in range(0, len(schedule), batch_size):
            specs = schedule[start:start + batch_size]
            states = [maps[spec.map_index].copy() for spec in specs]
            if policy is None:
                # Build feature shapes from the same encoder as deployment. The
                # temporary CPU environment is closed before the real matches run.
                with VecEnv.from_states(states, device="cpu", threads=cpu_threads) as example_env:
                    batch = GameBatch.from_backend(example_env, device)
                    pids = torch.tensor([spec.learner_seat for spec in specs], device=device)
                    encoder = make_observation_encoder(training_task)
                    history = ObservationHistory(training_task.history_length)
                    history.reset(encoder.encode(batch, pids))
                    knowledge = _initial_knowledge(batch, pids, training_task)
                    obs = _local_inputs(batch, pids, encoder, history,
                                        make_action_space(training_task), knowledge)
                    policy = _load_student(checkpoint, config, obs, device)
                if trace_directory is not None:
                    traces = SelectedGameTraces(trace_directory, max_ticks=max_ticks, opponents=opponents,
                        metadata={"checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                                  "evaluation_seed": seed, "sampling_seed": sampling_seed if stochastic else None,
                                  "sampling_protocol": "per_game_numpy_uniform_cdf_v1" if stochastic else None,
                                  "stochastic": stochastic, "device": str(device),
                                  "matmul_precision": torch.get_float32_matmul_precision(),
                                  "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                  "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                  **_inference_metadata(policy, device), "batch_size": batch_size,
                                  "baseline_workers": baseline_workers, "max_ticks": max_ticks})
            if baseline_workers > 1:
                pool_started = perf_counter()
                if baseline_pool is None:
                    baseline_pool = worker_lifetime.enter_context(BaselinePool(specs, workers=baseline_workers))
                else:
                    baseline_pool.reset(specs)
                baseline_lifecycle_seconds += perf_counter() - pool_started
            completed, timing = battle_batch(specs, states, policy, training_task,
                                            max_ticks=max_ticks, device=device,
                                            stochastic=stochastic, sampling_seed=sampling_seed,
                                            cpu_threads=cpu_threads, baseline_workers=baseline_workers,
                                            _baseline_pool=baseline_pool,
                                            recorder=traces if traces is not None and traces.needs_batch(specs) else None)
            if traces is not None:
                traces.finish_batch(completed)
            games.extend(completed)
            timings.append(timing)
            print(f"Built-in evaluation: {len(games)}/{len(schedule)} games completed", flush=True)
        before_pool_close = perf_counter()
    if baseline_pool is not None:
        baseline_lifecycle_seconds += perf_counter() - before_pool_close
    if len(games) != len(schedule) or {game["game_id"] for game in games} != {spec.game_id for spec in schedule}:
        raise RuntimeError("completed games do not match the fixed schedule")
    totals = {key: sum(timing[key] for timing in timings) for key in timings[0]}
    totals["baseline_lifecycle_seconds"] = baseline_lifecycle_seconds
    totals["wall_seconds"] += baseline_lifecycle_seconds
    totals["environment_ticks_per_second"] = totals["environment_ticks"] / totals["wall_seconds"]
    totals["student_decisions_per_second"] = totals["student_decisions"] / totals["wall_seconds"]
    totals["games_per_second"] = len(games) / totals["wall_seconds"]
    totals["total_seconds_including_setup"] = perf_counter() - started
    report = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "policy": "student_local", "opponent_implementation": "generals_baselines.make_controllers",
        "decision_timing": "joint_pre_growth_then_absolute_seat_order",
        "training_opponent": training_task.opponent,
        "productive_actions": training_task.productive_actions,
        "combat_features": training_task.combat_features,
        "persistent_knowledge": training_task.persistent_knowledge,
        "seed": seed, "maps_per_ai": maps_per_ai, "scheduled_games": len(schedule),
        "completed_games": len(games), "batch_size": batch_size, "device": str(device),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        **_inference_metadata(policy, device),
        "cpu_threads": cpu_threads, "baseline_workers": baseline_workers, "max_ticks": max_ticks,
        "clock_normalization_ticks": training_task.max_episode_steps,
        "stochastic": stochastic, "map_config": asdict(map_config),
        "sampling_seed": sampling_seed if stochastic else None,
        "sampling_protocol": "per_game_numpy_uniform_cdf_v1" if stochastic else None,
        "confidence_interval": "95% Wilson binomial interval; games are paired by map, interval treats games as independent",
        "overall": aggregate_games(games),
        "by_ai": {kind: aggregate_games([g for g in games if g["opponent"] == kind]) for kind in opponents},
        "by_seat": {str(seat): aggregate_games([g for g in games if g["learner_seat"] == seat]) for seat in (0, 1)},
        "by_ai_and_seat": {kind: {str(seat): aggregate_games([
            g for g in games if g["opponent"] == kind and g["learner_seat"] == seat])
            for seat in (0, 1)} for kind in opponents},
        "timing": totals, "games": games,
    }
    if traces is not None:
        report["selected_game_traces"] = traces.describe()
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
    parser.add_argument("--baseline-workers", type=int, default=1,
                        help="persistent CPU processes for original baseline controllers; 1 is serial")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stochastic", action="store_true", help="sample categorical actions (default)")
    mode.add_argument("--greedy", dest="stochastic", action="store_false", help="explicit greedy diagnostic")
    parser.set_defaults(stochastic=True)
    parser.add_argument("--sampling-seed", type=int, help="action sampling seed; defaults to --seed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-directory", type=Path,
                        help="retain only the first evaluated win/loss action trace per opponent")
    args = parser.parse_args(argv)
    output = args.output or args.checkpoint.with_name("builtin_evaluation.json")
    report = evaluate_builtin(args.checkpoint, maps_per_ai=args.maps_per_ai, seed=args.seed,
                              opponents=args.opponents, batch_size=args.batch_size,
                              max_ticks=args.max_ticks, device=args.device,
                              stochastic=args.stochastic, sampling_seed=args.sampling_seed,
                              cpu_threads=args.cpu_threads, baseline_workers=args.baseline_workers,
                              output=output, trace_directory=args.trace_directory)
    print(json.dumps({key: value for key, value in report.items() if key != "games"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
