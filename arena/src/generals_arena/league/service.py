"""Resumable bot workers and clock-driven human games."""
from contextlib import contextmanager
import fcntl
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


def run_next(store, stop=None):
    """Caller holds the worker lock. A failed game never becomes a rated loss."""
    for job in store.jobs():
        if job["status"] != "queued":
            continue
        try:
            plan = store.pending(job["id"])
            if plan is None:
                plans = make_block(store, job)
                if not plans:
                    continue  # Queue can wait for more entrants.
                store.insert_plans(plans, job["id"])
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


def run_jobs(store, *, limit=None, retry=False, progress=None):
    store.verify_execution()
    if limit is not None:
        integer("limit", limit, 1)
    completed = 0
    with process_lock(store, ".worker.lock"):
        if retry:
            store.retry()
        while limit is None or completed < limit:
            if not run_next(store):
                break
            completed += 1
            if progress is not None:
                progress(completed)
    return completed


class LiveArena:
    def __init__(self, store):
        self.store = store
        self.store.verify_execution()
        self.games = {}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.worker_error = None
        self.thread = threading.Thread(target=self._clock, daemon=True, name="arena-clock")
        self.worker = threading.Thread(target=self._worker, daemon=True, name="arena-worker")

    def start(self):
        # Server process lock is held by the caller before this recovery step.
        self.store.abort_live()
        self.thread.start()
        self.worker.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        self.worker.join(timeout=10)
        self.store.abort_live()

    def create_game(self, human, players):
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
                     "action": Action.wait(), "next_tick": now + self.store.config.tick_seconds,
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
        names = self.store.entrants()
        return {"game_id": game_id, "board": game.view(value["seat"]),
                "entrants": [{"name": names[key]["name"], "entrant_id": key} for key in game.plan["entrant_ids"]],
                "done": bool(value["finished"]), "error": value["error"],
                "result": game.result() if game.done and not value["error"] else None,
                "max_ticks": self.store.config.max_ticks}

    def view(self, game_id, human):
        with self.lock:
            return self._view(game_id, self._owned(game_id, human))

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
                        value["game"].advance({value["seat"]: value["action"]})
                        value["action"] = Action.wait()
                        # Do not let bursts of requests or lag accelerate gameplay.
                        value["next_tick"] = time.monotonic() + self.store.config.tick_seconds
                        if value["game"].done:
                            self.store.complete(value["game"])
                            value["finished"] = time.monotonic()
                    except Exception as exc:
                        self.store.abort_live(game_id)
                        value["error"] = str(exc)
                        value["finished"] = time.monotonic()

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
