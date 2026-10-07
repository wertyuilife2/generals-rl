"""Persistent CPU lane shards for the original built-in baseline controllers."""
from collections.abc import Mapping
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time
import traceback

from generals.controllers import AI_KINDS, make_controllers
from generals.core.observation import Observation
from generals.core.state import Action, MoveResult


def _parse_specs(specs):
    parsed = {}
    for lane, spec in enumerate(specs):
        get = spec.__getitem__ if isinstance(spec, Mapping) else lambda key: getattr(spec, key)
        kind, seat, seed = get("opponent"), int(get("learner_seat")), int(get("controller_seed"))
        if kind not in AI_KINDS or seat not in (0, 1) or seed < 0:
            raise ValueError("baseline specs require a built-in opponent, two-player seat and nonnegative seed")
        parsed[lane] = (kind, 1 - seat, seed)
    if not parsed:
        raise ValueError("baseline pool needs at least one lane")
    return parsed


def _make_controller(spec):
    kind, enemy_seat, seed = spec
    kinds = ["random", "random"]
    kinds[enemy_seat] = kind
    # Keep the unused learner factory slot: streams are spawned by absolute seat.
    return make_controllers(kinds, seed=seed)[enemy_seat]


def _feedback(controller, enemy_seat, moves):
    for pid, move in enumerate(moves):
        if move is not None and pid == enemy_seat:
            controller.on_result(move)
        if move is not None and move.eliminated == enemy_seat and hasattr(controller, "eliminate"):
            controller.eliminate()


def _worker(connection, specs):
    # No torch, policy or evaluator import. The caller's main module may still be
    # re-imported by Python spawn; controller work itself is strictly CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        controllers = {lane: _make_controller(spec) for lane, spec in specs.items()}
        connection.send(("ok", None))
        while True:
            command, requests = connection.recv()
            if command == "close":
                return
            if command == "act":
                answer = {}
                for lane, observation in requests.items():
                    action = controllers[lane].act(observation)
                    if not isinstance(action, Action):
                        raise TypeError("built-in controller must return the original Action type")
                    answer[lane] = action
            elif command == "reset":
                specs = requests
                controllers = {lane: _make_controller(spec) for lane, spec in specs.items()}
                answer = None
            elif command == "feedback":
                for lane, moves in requests.items():
                    _feedback(controllers[lane], specs[lane][1], moves)
                answer = None
            else:
                raise ValueError(f"unknown baseline command: {command}")
            connection.send(("ok", answer))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


class BaselinePool:
    """One fixed controller per lane, with act -> feedback barriers.

    Specs provide opponent, learner_seat and controller_seed (attributes or
    mapping keys). feedback values are the decoded moves for absolute seats
    (0, 1), with None for skipped native rows. workers=1 executes serially in
    the caller and preserves the original factory and controller lifecycle.
    reset(specs) starts a fresh match batch without restarting worker processes.
    """

    def __init__(self, specs, workers=1):
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError("baseline workers must be a positive integer")
        self._specs = _parse_specs(specs)
        self.workers = min(workers, len(self._specs))
        self._controllers = {}
        self._connections = []
        self._processes = []
        self._awaiting_feedback = set()
        self._entered = False
        self._closed = False

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("baseline pool cannot be reopened")
        self._entered = True
        try:
            if self.workers == 1:
                self._controllers = {lane: _make_controller(spec) for lane, spec in self._specs.items()}
            else:
                context = mp.get_context("spawn")
                for shard in range(self.workers):
                    parent, child = context.Pipe()
                    specs = {lane: spec for lane, spec in self._specs.items() if lane % self.workers == shard}
                    process = context.Process(target=_worker, args=(child, specs))
                    self._connections.append(parent)
                    self._processes.append(process)
                    try:
                        process.start()
                    finally:
                        child.close()
                for index in range(self.workers):
                    self._receive(index)
        except BaseException:
            self.close()
            raise
        return self

    def _ensure_open(self):
        if not self._entered or self._closed:
            raise RuntimeError("baseline pool is closed or not entered")

    def _receive(self, index):
        connection, process = self._connections[index], self._processes[index]
        ready = wait([connection, process.sentinel])
        if connection not in ready:
            raise RuntimeError(f"baseline worker {index} exited unexpectedly (exitcode={process.exitcode})")
        try:
            status, answer = connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"baseline worker {index} disconnected") from exc
        if status != "ok":
            raise RuntimeError(f"baseline worker {index} failed:\n{answer}")
        return answer

    def _dispatch(self, command, requests):
        groups = [{} for _ in range(self.workers)]
        for lane, value in requests.items():
            groups[lane % self.workers][lane] = value
        answer = {}
        try:
            for index, group in enumerate(groups):
                if group or command == "reset":
                    self._connections[index].send((command, group))
            for index, group in enumerate(groups):
                if group or command == "reset":
                    result = self._receive(index)
                    if command == "act":
                        answer.update(result)
        except BaseException:
            self.close()
            raise
        return {lane: answer[lane] for lane in requests} if command == "act" else None

    def reset(self, specs):
        self._ensure_open()
        if self._awaiting_feedback:
            raise RuntimeError("baseline reset requires all lanes' feedback to be complete")
        parsed = _parse_specs(specs)
        if self.workers == 1:
            try:
                self._controllers = {lane: _make_controller(spec) for lane, spec in parsed.items()}
            except BaseException:
                self.close()
                raise
        else:
            # Empty shards must also replace their dictionaries, releasing old
            # controller memory/RNG when the final batch has fewer lanes.
            self._dispatch("reset", parsed)
        self._specs = parsed

    def act(self, observations_by_lane):
        self._ensure_open()
        for lane, observation in observations_by_lane.items():
            if lane not in self._specs:
                raise ValueError(f"unknown baseline lane: {lane}")
            if lane in self._awaiting_feedback:
                raise RuntimeError(f"baseline lane {lane} needs feedback before its next act")
            if (not isinstance(observation, Observation) or observation.visibility != "local"
                    or observation.player_id != self._specs[lane][1]):
                raise ValueError("baseline act requires the lane's own local Observation")
        if self.workers == 1:
            try:
                answer = {lane: self._controllers[lane].act(observation)
                          for lane, observation in observations_by_lane.items()}
                if any(not isinstance(action, Action) for action in answer.values()):
                    raise TypeError("built-in controller must return the original Action type")
            except BaseException:
                self.close()
                raise
        else:
            answer = self._dispatch("act", observations_by_lane)
        self._awaiting_feedback.update(observations_by_lane)
        return answer

    def feedback(self, moves_by_lane):
        self._ensure_open()
        for lane, moves in moves_by_lane.items():
            if lane not in self._awaiting_feedback:
                raise RuntimeError(f"baseline lane {lane} has no act awaiting feedback")
            if len(moves) != 2 or any(move is not None and
                    (not isinstance(move, MoveResult) or move.player_id != pid)
                    for pid, move in enumerate(moves)):
                raise ValueError("baseline feedback needs decoded moves in absolute seat order (0, 1)")
        if self.workers == 1:
            try:
                for lane, moves in moves_by_lane.items():
                    _feedback(self._controllers[lane], self._specs[lane][1], moves)
            except BaseException:
                self.close()
                raise
        else:
            self._dispatch("feedback", moves_by_lane)
        self._awaiting_feedback.difference_update(moves_by_lane)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for connection, process in zip(self._connections, self._processes):
            if process.pid is not None and process.is_alive():
                try:
                    connection.send(("close", None))
                except (EOFError, OSError):
                    pass
        deadline = time.monotonic() + 2
        for process in self._processes:
            if process.pid is not None:
                process.join(timeout=max(0, deadline - time.monotonic()))
        for process in self._processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in self._processes:
            if process.pid is not None:
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
        for connection in self._connections:
            connection.close()
        self._controllers.clear()

    def __exit__(self, *args):
        self.close()
