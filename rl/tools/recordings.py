"""Native match observation sink, independent of policies and video rendering."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from generals.core.recording import TraceWriter, replay, state_digest, state_to_dict


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
