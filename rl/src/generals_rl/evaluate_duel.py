"""Balanced, batched matches between two frozen local-observation policies.

Each map is played with the candidate in both absolute seats. Each actor owns
its configured history and persistent knowledge; privileged critic observations
are never supplied to inference. Both decisions precede the joint native tick.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from generals import MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals.core.recording import RULES_VERSION

from .config import task_config
from .eval_execution import FeaturePipeline, GraphStudent, pack_actions, unpack_actions
from .evaluate_builtin import (
    _inference_metadata, _initial_knowledge, _load_student, aggregate_games,
)
from .mdp import GameBatch, ObservationHistory, make_action_space, make_observation_encoder
from .spatial_ppo import SpatialPPOPolicy


@dataclass(frozen=True)
class DuelSpec:
    game_id: int
    map_index: int
    map_seed: int
    candidate_seat: int


def make_duel_schedule(*, maps=400, seed=314159265):
    """The same map and absolute-seat random streams are used in each seat pair."""
    _integer("maps", maps, 1)
    _integer("seed", seed, 0)
    return [DuelSpec(2 * index + seat, index,
                     _derived_seed(seed, index, 0), seat)
            for index in range(maps) for seat in (0, 1)]


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _derived_seed(*parts):
    return int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint64)[0])


def _file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _features(batch, ids, task, *, capture):
    encoder = make_observation_encoder(task)
    history = ObservationHistory(task.history_length)
    history.reset(encoder.encode(batch, ids))
    return FeaturePipeline(batch, ids, encoder, history, make_action_space(task),
                           _initial_knowledge(batch, ids, task), capture=capture)


def _compatible_tasks(candidate, reference):
    # Each model retains its own normalization clock and observation/action
    # schema. History, knowledge and architecture deliberately need not match.
    fields = ("width", "height", "players", "mountain_density", "city_density")
    if any(getattr(candidate, name) != getattr(reference, name) for name in fields):
        raise ValueError("duel checkpoints must agree on map geometry and generation settings")
    if candidate.players != 2:
        raise ValueError("model duels require two-player checkpoints")


def _status(env):
    """One compact CUDA-to-host transfer; do not copy the full board per tick."""
    moves = env.unwrapped.last_moves
    if isinstance(moves, torch.Tensor):
        invalid = moves[:, :, 0].bool() & ~moves[:, :, 1].bool()
        return torch.cat((env.done.long(), env.state["tick"][:, None],
                          env.state["winner"][:, None].long(),
                          env.unwrapped.errors[:, None].long(), invalid.long()), 1).cpu().numpy()
    invalid = moves[:, :, 0].astype(bool) & ~moves[:, :, 1].astype(bool)
    return np.column_stack((env.done, env.state["tick"], env.state["winner"],
                            env.unwrapped.errors, invalid))


def duel_batch(specs, states, candidate, reference, candidate_task, reference_task, *,
               max_ticks=2000, device="cpu", env_backend="cpu", stochastic=True,
               sampling_seed=161803398, cpu_threads=1, reference_inference=False,
               recorder=None):
    """Finish fixed match lanes without auto-reset or replacement games.

    ``recorder(spec, joint_actions, tick)`` is an optional testing hook. Timing
    is full batch wall time; asynchronous per-phase timings are not reported.
    """
    _compatible_tasks(candidate_task, reference_task)
    _integer("max_ticks", max_ticks, 1)
    _integer("sampling_seed", sampling_seed, 0)
    _integer("cpu_threads", cpu_threads, 1)
    if not isinstance(stochastic, bool) or not isinstance(reference_inference, bool):
        raise TypeError("stochastic and reference_inference must be boolean")
    if not specs or len(specs) != len(states):
        raise ValueError("provide matching nonempty match specifications and states")
    if len({spec.game_id for spec in specs}) != len(specs):
        raise ValueError("game IDs must be unique")
    if any(spec.candidate_seat not in (0, 1) for spec in specs):
        raise ValueError("candidate seat must be zero or one")
    if any(state.player_count != 2 or state.tick != 0 or state.terminated or
           (state.layout.width, state.layout.height) != (candidate_task.width, candidate_task.height)
           for state in states):
        raise ValueError("duels require fresh two-player states with checkpoint geometry")
    device = torch.device(device)
    if env_backend not in ("cpu", "gpu") or (env_backend == "gpu" and device.type != "cuda"):
        raise ValueError("env_backend must be cpu or gpu; gpu requires a CUDA actor device")
    started = perf_counter()
    count = len(specs)
    seats = np.array([spec.candidate_seat for spec in specs], dtype=np.int64)
    # Common random numbers are tied to the map and ABSOLUTE seat, not the
    # policy identity or completion order. Identical policies therefore produce
    # complementary paired outcomes, including stochastic self-duels.
    seeds = [[_derived_seed(sampling_seed, spec.map_index, seat) for seat in (0, 1)]
             for spec in specs]
    rng = [[np.random.default_rng(seed) for seed in pair] for pair in seeds]
    active = np.ones(count, dtype=bool)
    invalid = np.zeros((count, 2), dtype=np.int64)
    joint = np.zeros((count, 2), dtype=np.int64)
    games = [None] * count
    placement = {"device": device} if env_backend == "gpu" else {"device": "cpu", "threads": cpu_threads}
    with VecEnv.from_states(states, **placement, auto_reset=False, episode_ticks=max_ticks) as env:
        batch = GameBatch.from_backend(env, device)
        ids = [torch.as_tensor(seats, device=device), torch.as_tensor(1 - seats, device=device)]
        with torch.inference_mode():
            features = [_features(batch, player_ids, task,
                                  capture=device.type == "cuda" and not reference_inference)
                        for player_ids, task in zip(ids, (candidate_task, reference_task), strict=True)]
            while active.any():
                lanes = np.flatnonzero(active)
                lane_ids = torch.as_tensor(lanes, device=device)
                active_before = torch.as_tensor(active.copy(), device=device)
                joint.fill(0)
                for role, policy in enumerate((candidate, reference)):
                    obs = features[role].observations[lane_ids]
                    if stochastic:
                        kwargs = ({"validate_mask": False} if isinstance(policy, SpatialPPOPolicy)
                                  or getattr(policy, "inference_execution", None) is not None else {})
                        probabilities = policy.action_probabilities(obs, **kwargs)
                        uniforms = np.array([rng[lane][int(seats[lane]) ^ role].random() for lane in lanes])
                        cdf = probabilities.double().cumsum(-1)
                        cdf = cdf / cdf[:, -1:]
                        actions = torch.searchsorted(cdf, torch.as_tensor(uniforms, device=device)[:, None], right=True)
                    else:
                        actions = policy.act_inference(obs)
                    encoded = unpack_actions(pack_actions(actions, obs["action_mask"], features[role].action_space))
                    joint[lanes, seats[lanes] ^ role] = encoded
                env.step(torch.as_tensor(joint, device=env.device) if env_backend == "gpu" else joint)
                status = _status(env)
                if status[:, 4].any():
                    lane = int(np.flatnonzero(status[:, 4])[0])
                    raise RuntimeError(f"native duel lane {lane}: error code {int(status[lane, 4])}")
                invalid += status[:, 5:7].astype(np.int64) * active[:, None]
                if recorder is not None:
                    for lane in lanes:
                        recorder(specs[lane], joint[lane].copy(), int(status[lane, 2]))
                ending = active & status[:, :2].astype(bool).any(axis=1)
                if ending.any():
                    stats = torch.as_tensor(env.stats).cpu().numpy()
                    for lane in np.flatnonzero(ending):
                        seat = int(seats[lane])
                        terminal, truncated = map(bool, status[lane, :2])
                        winner = int(status[lane, 3])
                        if terminal and winner not in (0, 1):
                            raise RuntimeError("a terminal two-player duel has no valid winner")
                        outcome = ("win" if winner == seat else "loss") if terminal else "draw"
                        games[lane] = {
                            **asdict(specs[lane]), "learner_seat": seat,
                            "outcome": outcome, "terminated": terminal, "truncated": truncated,
                            "winner_seat": winner if terminal else None, "ticks": int(status[lane, 2]),
                            "candidate_invalid_actions": int(invalid[lane, seat]),
                            "reference_invalid_actions": int(invalid[lane, 1 - seat]),
                            "candidate_final_stats": dict(zip(("army", "territory", "cities"), map(int, stats[lane, seat]))),
                            "reference_final_stats": dict(zip(("army", "territory", "cities"), map(int, stats[lane, 1 - seat]))),
                            "candidate_sampling_seed": seeds[lane][seat] if stochastic else None,
                            "reference_sampling_seed": seeds[lane][1 - seat] if stochastic else None,
                        }
                    active[ending] = False
                batch = GameBatch.from_backend(env, device)
                for feature in features:
                    feature.advance(batch, active_before)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    if any(game is None for game in games):
        raise RuntimeError("a scheduled duel did not finish")
    ticks = sum(game["ticks"] for game in games)
    wall = perf_counter() - started
    return games, {"wall_seconds": wall, "environment_ticks": ticks,
                   "actor_decisions": 2 * ticks, "games_per_second": count / wall,
                   "environment_ticks_per_second": ticks / wall}


def summarize_duels(games, *, bootstrap_seed=0):
    """Report draw-neutral score with a paired-map cluster bootstrap interval."""
    summary = aggregate_games(games)
    summary["score"] = ((summary["wins"] + .5 * summary["draws"]) / len(games)) if games else None
    by_map = {}
    for game in games:
        by_map.setdefault(game["map_index"], []).append(game)
    if not by_map:
        summary["score_paired_map_bootstrap95"] = None
        return summary
    if any(len(pair) != 2 or {game["candidate_seat"] for game in pair} != {0, 1}
           for pair in by_map.values()):
        raise ValueError("paired-map score interval requires both seats of every map")
    scores = np.array([np.mean([{"win": 1., "draw": .5, "loss": 0.}[game["outcome"]]
                               for game in pair]) for pair in by_map.values()])
    generator = np.random.default_rng(bootstrap_seed)
    # Bound temporary memory even for unusually large evaluation schedules.
    means = np.concatenate([scores[generator.integers(len(scores), size=(1000, len(scores)))].mean(1)
                            for _ in range(10)])
    summary["score_paired_map_bootstrap95"] = np.quantile(means, [.025, .975]).tolist()
    summary["paired_maps"] = len(scores)
    return summary


@contextmanager
def _evaluation_runtime(threads, device):
    previous = (torch.get_num_threads(), torch.get_float32_matmul_precision(),
                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        torch.set_num_threads(threads)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.set_num_threads(previous[0])
        torch.set_float32_matmul_precision(previous[1])
        torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = previous[2:]


def evaluate_duel(checkpoint, reference_checkpoint, *, config_path=None, reference_config_path=None,
                  maps=400, seed=314159265, sampling_seed=161803398, batch_size=0,
                  max_ticks=2000, device="cpu", env_backend="cpu", stochastic=True,
                  cpu_threads=1, reference_inference=False, output=None):
    """Evaluate immutable candidate/reference artifacts; default 400 maps =800 games."""
    _integer("batch_size", batch_size, 0)
    _integer("cpu_threads", cpu_threads, 1)
    _integer("max_ticks", max_ticks, 1)
    _integer("sampling_seed", sampling_seed, 0)
    if not isinstance(stochastic, bool) or not isinstance(reference_inference, bool):
        raise TypeError("stochastic and reference_inference must be boolean")
    device = torch.device(device)
    if env_backend not in ("cpu", "gpu") or (env_backend == "gpu" and device.type != "cuda"):
        raise ValueError("env_backend must be cpu or gpu; gpu requires a CUDA actor device")
    if output is not None and Path(output).exists():
        raise FileExistsError(f"duel output already exists: {output}")
    schedule = make_duel_schedule(maps=maps, seed=seed)
    checkpoints = [Path(checkpoint).resolve(), Path(reference_checkpoint).resolve()]
    config_paths = [Path(config_path).resolve() if config_path else checkpoints[0].with_name("run_config.json"),
                    Path(reference_config_path).resolve() if reference_config_path else checkpoints[1].with_name("run_config.json")]
    config_bytes = [path.read_bytes() for path in config_paths]
    configs = [json.loads(value) for value in config_bytes]
    tasks = [task_config(config["task"]) for config in configs]
    _compatible_tasks(*tasks)
    provenance = [{"checkpoint": str(path), "checkpoint_sha256": _file_sha(path),
                   "config_path": str(config_file), "config_sha256": sha256(raw_config).hexdigest(),
                   "policy_config": config["runner"]["policy"], "task_config": config["task"]}
                  for path, config_file, config, raw_config in
                  zip(checkpoints, config_paths, configs, config_bytes, strict=True)]
    map_config = MapConfig(tasks[0].width, tasks[0].height, 2,
                           tasks[0].mountain_density, tasks[0].city_density)
    parallel_games = batch_size or len(schedule)
    policies, games, timings = None, [], []
    started = perf_counter()
    with _evaluation_runtime(cpu_threads, device):
        for start in range(0, len(schedule), parallel_games):
            specs = schedule[start:start + parallel_games]
            map_states = {spec.map_index: generate_map(map_config, spec.map_seed) for spec in specs}
            states = [map_states[spec.map_index].copy() for spec in specs]
            if policies is None:
                with VecEnv.from_states(states[:1], device="cpu", threads=cpu_threads) as example_env:
                    example = GameBatch.from_backend(example_env, device)
                    policies = []
                    for path, config, task in zip(checkpoints, configs, tasks, strict=True):
                        features = _features(example, torch.zeros(1, dtype=torch.long, device=device), task, capture=False)
                        # Loading weights must not advance the caller's random stream.
                        with torch.random.fork_rng(devices=[]):
                            policy = _load_student(path, config, features.observations, device)
                        policy.requires_grad_(False)
                        if device.type == "cuda" and not reference_inference and hasattr(policy, "_forward"):
                            policy = GraphStudent(policy)
                        policies.append(policy)
            completed, timing = duel_batch(
                specs, states, *policies, *tasks, max_ticks=max_ticks, device=device,
                env_backend=env_backend, stochastic=stochastic, sampling_seed=sampling_seed,
                cpu_threads=cpu_threads, reference_inference=reference_inference)
            games.extend(completed)
            timings.append(timing)
            print(f"Model duel: {len(games)}/{len(schedule)} games completed", flush=True)
        if ({game["game_id"] for game in games} != {spec.game_id for spec in schedule}
                or len(games) != len(schedule)):
            raise RuntimeError("completed duels do not match the fixed schedule")
        for artifact, path, config_file in zip(provenance, checkpoints, config_paths, strict=True):
            if artifact["checkpoint_sha256"] != _file_sha(path) or artifact["config_sha256"] != _file_sha(config_file):
                raise RuntimeError("a checkpoint or config changed during the duel; discard this evaluation")
        wall = sum(timing["wall_seconds"] for timing in timings)
        ticks = sum(timing["environment_ticks"] for timing in timings)
        report = {
            "protocol": "paired_local_model_duel_v1", "candidate": provenance[0], "reference": provenance[1],
            "rules_version": RULES_VERSION, "evaluator_sha256": _file_sha(__file__),
            "torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
            "decision_timing": "joint_pre_growth_then_absolute_seat_order",
            "policy_inputs": "independent_local_history_and_optional_persistent_knowledge; no critic inputs",
            "maps": maps, "scheduled_games": len(schedule), "completed_games": len(games),
            "seed": seed, "sampling_seed": sampling_seed if stochastic else None, "stochastic": stochastic,
            "sampling_protocol": "paired_map_absolute_seat_numpy_uniform_cdf_v1" if stochastic else None,
            "confidence_interval": "95% Wilson raw/decisive win intervals treat games as independent; score uses 10000 paired-map cluster bootstrap replicates",
            "score_definition": "(wins + 0.5 * draws) / games; timeouts count as draws",
            "device": str(device), "env_backend": env_backend, "cpu_threads": cpu_threads,
            "batch_size": batch_size, "parallel_games": min(parallel_games, len(schedule)),
            "max_ticks": max_ticks, "map_config": asdict(map_config),
            "reference_inference": reference_inference,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "inference": [{**_inference_metadata(policy, device),
                           "execution": getattr(policy, "inference_execution", "original")} for policy in policies],
            "overall": summarize_duels(games, bootstrap_seed=seed),
            "by_seat": {str(seat): {**aggregate_games([game for game in games if game["candidate_seat"] == seat]),
                                    "score": np.mean([{"win": 1., "draw": .5, "loss": 0.}[game["outcome"]]
                                                      for game in games if game["candidate_seat"] == seat])}
                        for seat in (0, 1)},
            "timing": {"wall_seconds": wall, "total_seconds_including_setup": perf_counter() - started,
                       "environment_ticks": ticks, "actor_decisions": 2 * ticks,
                       "environment_ticks_per_second": ticks / wall, "games_per_second": len(games) / wall},
            "games": games,
        }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("reference_checkpoint", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--reference-config-path", type=Path)
    parser.add_argument("--maps", type=int, default=400, help="each map is played from both candidate seats")
    parser.add_argument("--seed", type=int, default=314159265)
    parser.add_argument("--sampling-seed", type=int, default=161803398)
    parser.add_argument("--batch-size", type=int, default=0, help="0 runs the full schedule concurrently")
    parser.add_argument("--max-ticks", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--reference-inference", action="store_true", help="disable evaluation CUDA graphs")
    parser.add_argument("--greedy", dest="stochastic", action="store_false", help="default samples categorical actions")
    parser.add_argument("--output", type=Path, required=True)
    report = evaluate_duel(**vars(parser.parse_args(argv)))
    print(json.dumps({key: value for key, value in report.items() if key != "games"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
