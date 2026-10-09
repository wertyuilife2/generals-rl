"""Resumable bot workers and clock-driven human games."""
from contextlib import contextmanager
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import threading
import time
from types import SimpleNamespace
import uuid

from generals import MapConfig
from generals.core.state import Action

from ..registry import load_controller
from ..util import integer, source_digest
from .game import RecordedGame
from .scheduler import make_block


@contextmanager
def process_lock(store, name):
    """OS locks release on crashes; the file itself is safe to leave in place."""
    with (store.directory / name).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process holds {name}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def controllers_for(store, plan):
    entries = store.entrants()
    rules = SimpleNamespace(map=MapConfig(**plan["map"]), visibility="local")
    controllers = []
    for key in plan["entrant_ids"]:
        entrant = entries[key]
        if entrant["kind"] == "human":
            controllers.append(None)
            continue
        for package, expected in entrant["implementation"].items():
            if package != "custom_version" and source_digest(package) != expected:
                raise ValueError(f"registered source changed for {entrant['name']}; restore the recorded version")
        controllers.append(load_controller(entrant, rules))
    return controllers


def prepare_batch(store, job):
    """Persist at least 32 games before execution, adapting at each window.

    This keeps workers busy across small rotation groups. The same window is
    used with one or many workers, so process count never changes a schedule.
    Already-persisted rotations always take precedence during a resume.
    """
    with store.connect() as db:
        plans = [json.loads(row[0]) for row in db.execute(
            "SELECT plan FROM games WHERE job_id=? AND status='pending' ORDER BY rowid", (job["id"],))]
    # Existing windows are immutable, including partially completed windows.
    if plans:
        return plans
    done = job["done"]
    while len(plans) < 32 and done < job["budget"]:
        group = make_block(store, {**job, "done": done})
        if not group:
            break
        plans.extend(group)
        done += len(group)
    if plans:
        store.insert_plans(plans, job["id"])
    return plans


def run_next(store, stop=None):
    """Caller holds the worker lock. A failed game never becomes a rated loss."""
    for job in store.jobs():
        if job["status"] != "queued":
            continue
        try:
            plans = prepare_batch(store, job)
            if not plans:
                continue
            plan = plans[0]
            game = RecordedGame(plan, store.config, controllers_for(store, plan))
            while not game.done:
                if stop is not None and stop.is_set():
                    return False  # Keep the plan pending; deterministic retry.
                game.advance()
            store.complete(game)
            return True
        except Exception as exc:
            store.fail(job["id"], exc)
            raise
    return False


def simulate(directory, plan):
    """Worker processes simulate only. The coordinator commits in schedule order."""
    from .store import LeagueStore
    store = LeagueStore(directory)
    start = time.monotonic()
    game = RecordedGame(plan, store.config, controllers_for(store, plan))
    while not game.done:
        game.advance()
    return {**game.result(), "simulation_seconds": time.monotonic() - start}, game.replay_blob()


def run_jobs(store, *, limit=None, retry=False, progress=None, workers=1):
    store.verify_execution()
    if limit is not None:
        integer("limit", limit, 1)
    integer("workers", workers, 1)
    if workers > 32:
        raise ValueError("workers must be <= 32")
    completed = 0
    with process_lock(store, ".worker.lock"):
        if retry:
            store.retry()
        if workers > 1:
            return _run_parallel(store, workers, limit, progress)
        while limit is None or completed < limit:
            if not run_next(store):
                break
            completed += 1
            if progress is not None:
                progress(completed)
    return completed


def _run_parallel(store, workers, limit, progress):
    completed = 0
    # Keep adaptation at the same persistent window as sequential execution.
    # Spawn works on macOS and does not inherit SQLite connections or locks.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        while limit is None or completed < limit:
            job = next((j for j in store.jobs() if j["status"] == "queued"), None)
            if job is None:
                break
            plans = prepare_batch(store, job)
            if not plans:
                break
            if limit is not None:
                plans = plans[:limit - completed]
            futures = [executor.submit(simulate, str(store.directory.resolve()), plan) for plan in plans]
            try:
                for future in futures:
                    result, blob = future.result()
                    store.complete_result(result, blob)
                    completed += 1
                    if progress is not None:
                        progress(completed)
            except Exception as exc:
                store.fail(job["id"], exc)
                raise
    return completed


class LiveArena:
    def __init__(self, store):
        self.store = store
        self.store.verify_execution()
        self.games = {}
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.worker_error = None
        self.thread = threading.Thread(target=self._clock, daemon=True, name="arena-clock")
        self.worker = threading.Thread(target=self._worker, daemon=True, name="arena-worker")

    def start(self, run_worker=True):
        # Server process lock is held by the caller before this recovery step.
        self.store.abort_live()
        self.thread.start()
        if run_worker:
            self.worker.start()

    def close(self):
        self.stop.set()
        with self.changed:
            self.changed.notify_all()
        self.thread.join(timeout=10)
        if self.worker.is_alive():
            self.worker.join(timeout=10)
        self.store.abort_live()

    def create_game(self, human, players=None):
        if players is not None:
            integer("players", players, 2)
        with self.lock:
            now = time.monotonic()
            self.games = {key: value for key, value in self.games.items()
                          if not value["finished"] or now - value["finished"] < 600}
            for key, value in self.games.items():
                if value["human"] == human and not value["finished"]:
                    return self._view(key, value)
            if sum(not value["finished"] for value in self.games.values()) >= 16:
                raise ValueError("too many active human games; try again later")
            job = {"id": uuid.uuid4().hex, "entrant_id": human, "done": 0, "budget": 1}
            plans = make_block(self.store, job, human=True, players=players)
            if not plans:
                raise ValueError("register more bots before starting this game")
            plan = plans[0]
            game = RecordedGame(plan, self.store.config, controllers_for(self.store, plan))
            self.store.insert_plans([plan], status="live")
            value = {"game": game, "human": human, "seat": plan["entrant_ids"].index(human),
                     "action": Action.wait(), "queue": deque(), "last_command_id": 0,
                     "recent_commands": {}, "revision": 0, "board": None,
                     "next_tick": now + self.store.config.tick_seconds,
                     "finished": None, "error": None}
            self.games[plan["game_id"]] = value
            return self._view(plan["game_id"], value)

    def _owned(self, game_id, human):
        value = self.games.get(game_id)
        if value is None:
            raise ValueError("live game not found; check match history after a server restart")
        if value["human"] != human:
            raise PermissionError("this game belongs to another player")
        return value

    def _view(self, game_id, value):
        game = value["game"]
        if value["board"] is None:
            value["board"] = game.view(value["seat"])
        if "entrants" not in value:
            names = self.store.entrants()
            value["entrants"] = [{"name": names[key]["name"], "entrant_id": key} for key in game.plan["entrant_ids"]]
        return {"game_id": game_id, "board": value["board"],
                "revision": value["revision"], "last_command_id": value["last_command_id"],
                "queue": list(value["queue"]),
                "entrants": value["entrants"],
                "done": bool(value["finished"]), "error": value["error"],
                "result": game.result() if game.done and not value["error"] else None,
                "max_ticks": self.store.config.max_ticks}

    def view(self, game_id, human, after=None):
        if after is not None:
            integer("after", after)
        with self.changed:
            value = self._owned(game_id, human)
            if after is not None:
                self.changed.wait_for(lambda: value["revision"] > after or value["finished"] or self.stop.is_set(), timeout=15)
            return self._view(game_id, value)

    def command(self, game_id, human, command):
        """Ordered, idempotent input independent of the displayed server tick."""
        command_id = command.get("id")
        integer("command id", command_id, 1)
        op = command.get("op")
        if op not in ("append", "clear", "undo"):
            raise ValueError("unknown queue operation")
        with self.changed:
            value = self._owned(game_id, human)
            if command_id <= value["last_command_id"]:
                if value["recent_commands"].get(command_id) != command:
                    raise ValueError("command ID conflict; refresh this game")
                return self._view(game_id, value)
            if command_id != value["last_command_id"] + 1:
                raise ValueError("commands must arrive in order")
            game, seat = value["game"], value["seat"]
            if value["finished"] or not game.state.alive[seat]:
                raise ValueError("player is no longer active")
            queue = value["queue"]
            if op == "append":
                encoded = command.get("action")
                integer("action", encoded)
                if encoded > game.state.layout.size * 8:
                    raise ValueError("action is outside the map")
                if len(queue) >= 256:
                    raise ValueError("move queue is full (256 moves)")
                # Future sources may be captured by preceding queued moves.
                # Ownership and obstacles are checked by the engine at execution;
                # accepting intent reveals nothing about hidden terrain.
                queue.append({"id": command_id, "action": encoded})
            elif op == "clear":
                queue.clear()
                value["action"] = Action.wait()
            elif queue:
                queue.pop()
            value["last_command_id"] = command_id
            value["recent_commands"][command_id] = dict(command)
            if len(value["recent_commands"]) > 64:
                del value["recent_commands"][min(value["recent_commands"])]
            value["revision"] += 1
            self.changed.notify_all()
            return self._view(game_id, value)

    def active(self, human):
        with self.lock:
            for key, value in reversed(list(self.games.items())):
                if value["human"] == human:
                    return self._view(key, value)
        return None

    def action(self, game_id, human, encoded, tick):
        integer("action", encoded)
        integer("tick", tick)
        with self.lock:
            value = self._owned(game_id, human)
            game, seat = value["game"], value["seat"]
            if value["finished"] or not game.state.alive[seat]:
                raise ValueError("player is no longer active")
            if tick != game.state.tick:
                raise ValueError("stale turn; refresh the board and try again")
            if encoded > game.state.layout.size * 8:
                raise ValueError("action is outside the map")
            action = Action.decode(encoded)
            if encoded and game.state.owner.ravel()[action.source] != seat:
                raise ValueError("select one of your own tiles")
            value["action"] = action  # At most one move per clock tick; latest input wins.
            return {"queued_for_tick": tick + 1}

    def _clock(self):
        while not self.stop.wait(.02):
            with self.lock:
                for game_id, value in list(self.games.items()):
                    now = time.monotonic()
                    if value["finished"] or now < value["next_tick"]:
                        continue
                    try:
                        if not value["game"].state.alive[value["seat"]]:
                            value["queue"].clear()
                            value["action"] = Action.wait()
                        action = (Action.decode(value["queue"].popleft()["action"]) if value["queue"] else value["action"])
                        value["game"].advance({value["seat"]: action})
                        value["board"] = None
                        value["revision"] += 1
                        value["action"] = Action.wait()
                        # Do not let bursts of requests or lag accelerate gameplay.
                        value["next_tick"] += self.store.config.tick_seconds
                        if value["next_tick"] <= time.monotonic():
                            value["next_tick"] = time.monotonic() + self.store.config.tick_seconds
                        if value["game"].done:
                            self.store.complete(value["game"])
                            value["finished"] = time.monotonic()
                    except Exception as exc:
                        self.store.abort_live(game_id)
                        value["error"] = str(exc)
                        value["finished"] = time.monotonic()
                    self.changed.notify_all()

    def _worker(self):
        try:
            with process_lock(self.store, ".worker.lock"):
                while not self.stop.is_set():
                    try:
                        if not run_next(self.store, self.stop):
                            self.stop.wait(1)
                    except Exception as exc:
                        self.worker_error = str(exc)
                        self.stop.wait(1)
        except RuntimeError as exc:
            # A CLI worker may already be draining the queue; human play still works.
            self.worker_error = str(exc)
