"""Native match observation sink, independent of policies and video rendering."""
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from generals import VecEnv
from generals.core.recording import (RULES_VERSION, TraceWriter, replay, state_digest,
                                    state_from_dict, state_to_dict)
from generals.core.state import Action


class MatchRecorder:
    """Record selected scheduled games, including every native post-tick state.

    The action trace is verified independently with CoreEngine, while the
    compressed frames let presentation tools operate without rerunning a policy.
    Full states belong to the recording sink, never to the deployed student.
    """

    def __init__(self, directory, *, names, metadata=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.names = dict(names)
        if len(set(self.names.values())) != len(self.names):
            raise ValueError("recording names must be unique")
        if any(Path(name).name != name or name in ("", ".", "..") for name in self.names.values()):
            raise ValueError("recording names must be simple filenames")
        self.metadata = dict(metadata or {})
        self.entries = {}
        self.completed = []

    def on_start(self, specs, states):
        for spec, state in zip(specs, states, strict=True):
            if spec.game_id not in self.names:
                continue
            if spec.game_id in self.entries:
                raise ValueError("a scheduled game was recorded twice")
            name = self.names[spec.game_id]
            metadata = {**self.metadata, "name": name, "game": asdict(spec),
                        "decision_timing": "joint_pre_growth_then_absolute_seat_order",
                        "policy": "student_local", "spectator_full_state_is_not_policy_input": True}
            path = self.directory / f"{name}.trace.jsonl"
            writer = TraceWriter(path, state, metadata)
            self.entries[spec.game_id] = {
                "name": name, "metadata": metadata, "writer": writer,
                "trace_path": path,
                "frames": [{"state": state_to_dict(state), "digest": state_digest(state),
                            "actions": None, "moves": None, "done": [False, False]}],
            }

    def on_tick(self, spec, actions, state, moves, done):
        entry = self.entries.get(spec.game_id)
        if entry is None:
            return
        if state.tick != len(entry["frames"]):
            raise RuntimeError("recording skipped or repeated a tick")
        entry["writer"].write_tick(SimpleNamespace(tick=state.tick), actions, state)
        entry["frames"].append({
            "state": state_to_dict(state), "digest": state_digest(state),
            "actions": [[int(a.kind), int(a.source), int(a.direction), int(a.mode)] for a in actions],
            "moves": moves.tolist(), "done": done.tolist(),
        })

    def finish(self):
        missing = set(self.names) - set(self.entries)
        if missing:
            raise RuntimeError(f"scheduled recordings were never started: {sorted(missing)}")
        for game_id, entry in self.entries.items():
            entry["writer"].close()
            # The native state digest for every tick must match the independent
            # rules interpreter with precisely the recorded joint actions.
            final = replay(entry["trace_path"], verify=True)
            if state_digest(final) != entry["frames"][-1]["digest"]:
                raise RuntimeError("native recording and independent replay differ")
            destination = self.directory / f"{entry['name']}.frames.json.gz"
            if destination.exists():
                raise FileExistsError(destination)
            with gzip.open(destination, "xt", encoding="utf-8") as stream:
                json.dump({"schema_version": 1, "metadata": entry["metadata"],
                           "frames": entry["frames"]}, stream, ensure_ascii=False, separators=(",", ":"))
            self.completed.append({"game_id": game_id, "name": entry["name"],
                                   "trace": str(entry["trace_path"].resolve()),
                                   "frames": str(destination.resolve()), "ticks": final.tick,
                                   "winner_seat": final.winner_id,
                                   "native_core_replay_verified": True})
        return self.completed

    def close(self):
        for entry in self.entries.values():
            entry["writer"].close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def replay_evaluation_trace(trace, match, directory, *, metadata):
    """Replay evaluated joint actions, without loading a policy or controller.

    Every CPU/native state is checked against the original evaluation digest;
    MatchRecorder then independently verifies every tick with CoreEngine.
    """
    from generals_rl.evaluate_builtin import GameSpec
    from generals_rl.mdp import ActionMetrics

    path = Path(trace["path"])
    if (trace["protocol"] != "recorded_joint_actions_core_v1"
            or trace["checkpoint_sha256"] != metadata["checkpoint_sha256"]
            or sha256(path.read_bytes()).hexdigest() != trace["sha256"]):
        raise ValueError("evaluated action trace identity/hash differs from the report")
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        source = header["metadata"]
        if (header["type"] != "header" or header["rules_version"] != RULES_VERSION
                or source["trace_protocol"] != trace["protocol"]
                or source["checkpoint_sha256"] != metadata["checkpoint_sha256"]
                or source["match"] != match):
            raise ValueError("evaluated action trace header differs from the report")
        spec = GameSpec(**source["game"])
        if (asdict(spec) != {key: match[key] for key in GameSpec.__dataclass_fields__}
                or trace["game_id"] != spec.game_id
                or trace["opponent"] != match["opponent"] or trace["outcome"] != match["outcome"]
                or trace["ticks"] != match["ticks"] or trace["learner_seat"] != match["learner_seat"]):
            raise ValueError("evaluated action trace game differs from the report")
        initial = state_from_dict(header["state"])
        if initial.tick != 0 or initial.terminated or state_digest(initial) != header["digest"]:
            raise ValueError("evaluated action trace initial state differs from its digest")
        name = f"{spec.opponent}_{match['outcome']}"
        recording_metadata = {**source, **metadata, "replay_mode": "recorded_joint_actions",
                              "replay_device": "cpu", "source_trace": str(path.resolve()),
                              "source_trace_sha256": trace["sha256"]}
        metrics = ActionMetrics(1, "cpu")
        totals = {name: torch.zeros(1) for name in ActionMetrics.names}
        invalid = np.zeros(2, dtype=np.int64)
        tick = 0
        with VecEnv.from_states([initial], device="cpu", threads=1, auto_reset=False,
                                episode_ticks=source["max_ticks"]) as env, \
                MatchRecorder(directory, names={spec.game_id: name}, metadata=recording_metadata) as recorder:
            recorder.on_start([spec], [initial])
            for row in map(json.loads, stream):
                if row["type"] != "tick" or row["tick"] != tick + 1 or tick >= match["ticks"]:
                    raise ValueError("evaluated action trace skipped or repeated a tick")
                actions = tuple(Action(*values) for values in row["actions"])
                if len(actions) != 2:
                    raise ValueError("evaluated action trace must contain both players' actions")
                result = env.step(np.array([[action.encode() for action in actions]], dtype=np.int64))
                env.check_errors()
                state = env.to_states()[0]
                tick += 1
                if state.tick != tick or state_digest(state) != row["digest"]:
                    raise ValueError(f"evaluated action trace native state mismatch at tick {tick}")
                recorder.on_tick(spec, actions, state, result.moves[0].copy(), result.done[0].copy())
                invalid += result.moves[0, :, 0].astype(bool) & ~result.moves[0, :, 1].astype(bool)
                moves = torch.as_tensor(result.moves[0, spec.learner_seat]).unsqueeze(0)
                for key, values in metrics.update(moves).items():
                    totals[key] += values
            if tick != match["ticks"] or tick == 0:
                raise ValueError("evaluated action trace did not cover the complete game")
            terminal, truncated = map(bool, result.done[0])
            if not terminal and not truncated:
                raise ValueError("evaluated action trace did not end a game")
            winner = int(env.state["winner"][0])
            seat = spec.learner_seat
            actual = {
                **asdict(spec), "outcome": "draw" if truncated else ("win" if winner == seat else "loss"),
                "terminated": terminal, "truncated": truncated, "winner_seat": winner if terminal else None,
                "ticks": tick, "learner_invalid_actions": int(invalid[seat]),
                "opponent_invalid_actions": int(invalid[1 - seat]),
                "action_metrics": {key: float(values[0]) / tick for key, values in totals.items()},
                "learner_final_stats": dict(zip(("army", "territory", "cities"), map(int, env.stats[0, seat]))),
                "opponent_final_stats": dict(zip(("army", "territory", "cities"), map(int, env.stats[0, 1 - seat]))),
            }
            if "sampling_seed" in match:
                actual["sampling_seed"] = match["sampling_seed"]
            if actual != match:
                raise ValueError("recorded-action replay did not reproduce the complete evaluated game")
            case = recorder.finish()[0]
    return {**case, "match": actual, "recorded_action_replay_verified": True,
            "source_trace": str(path.resolve()), "source_trace_sha256": trace["sha256"],
            "replay_mode": "recorded_joint_actions"}
