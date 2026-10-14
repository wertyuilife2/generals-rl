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
from ..util import digest, integer, source_digest
from ..execution import RecordedGame, close_game, result as execution_result, failure as execution_failure
from ..native import PolicyError
from .scheduler import make_block
from .rating_policy import POLICY, player_count
from .views import human_board


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


def eligible_schedule(store, job, *, coverage=None, **options):
    retired = store.retired_entrant_ids()
    if job['entrant_id'] in retired:
        raise ValueError('entrant is retired from the active league')
    disabled = store.disabled_policy_ids() | retired
    if job['entrant_id'] in disabled:
        raise ValueError('C++ policy is disqualified after a recorded violation')
    reference = store.reference if store.entrants()[store.reference]['kind'] == 'cpp' else None
    coverage = store.pair_counts(job['entrant_id']) if coverage is None else coverage
    # Front-load two independent reference maps, then add one every 50 target
    # games. A 1000-game publication spends at most 40 games (4%) on WAIT.
    # Coverage includes plans reserved in this window, not just finished games.
    quota = min(2*(job['budget']//50), 4+2*(job['done']//50))
    calibration = coverage.get(reference, 0) < quota
    calibration = (reference is not None and reference not in disabled and not options.get('human')
                   and job['entrant_id'] != reference and calibration and 2 in store.config.player_counts
                   and options.get('players') in (None, 2))
    if not options.get('human') and options.get('players') is None:
        options['players'] = 2 if calibration or (job['entrant_id'] == reference and 2 in store.config.player_counts) else player_count(store.config, job)
    # Keep WAIT out of the ordinary opponent pool, including proximity and
    # underplayed weighting. Explicit runs still honor every requested seat.
    entries = {key: e for key,e in store.entrants().items() if key not in disabled and key != reference}
    if options.get('players') == 2 and any(key != job['entrant_id'] and e['kind'] != 'human'
                                         for key,e in entries.items()):
        entries.pop(job['entrant_id'], None)
    proxy = SimpleNamespace(config=store.config, anchor=store.anchor, manifest=store.manifest,
        entrants=lambda: entries,
        ratings=store.ratings, pair_counts=lambda target: coverage)
    plans = make_block(proxy, job, **options)
    # Each calibration rotation group contains just one WAIT seat.
    if calibration:
        for plan in plans:
            plan['entrant_ids'][plan['instance_ids'].index(1)] = reference
    for plan in plans:
        plan['rating_policy'] = POLICY
        plan['rating_group_size'] = len(plans)
    return plans


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
    coverage = store.pair_counts(job['entrant_id'])
    while len(plans) < 32 and done < job["budget"]:
        group = eligible_schedule(store, {**job, "done": done}, coverage=coverage)
        if not group:
            break
        plans.extend(group)
        done += len(group)
        for plan in group:
            for key in set(plan['entrant_ids']) - {job['entrant_id']}:
                coverage[key] = coverage.get(key, 0)+1
    if plans:
        store.insert_plans(plans, job["id"])
    return plans


def run_next(store, stop=None, job_id=None, cpp_worker=None):
    """Caller holds the worker lock. A failed game never becomes a rated loss."""
    for job in store.jobs():
        if job["status"] != "queued" or (job_id is not None and job['id'] != job_id):
            continue
        game = None
        controllers = []
        owned_worker = None
        try:
            plans = prepare_batch(store, job)
            if not plans:
                continue
            plan = plans[0]
            entries = store.entrants()
            if any(entries[key]['kind'] == 'cpp' for key in plan['entrant_ids']):
                from ..cpp_match import MatchWorker
                worker = cpp_worker
                if worker is None:
                    owned_worker = worker = MatchWorker()
                result, blob = worker.run(store, plan)
                if 'policy_error' in result:
                    store.fail_game(result, blob)
                    store.fail(job['id'], result['policy_error'])
                    raise PolicyError(result['policy_error'], details=result.get('error_detail'))
                store.complete_result(result, blob)
                return True
            controllers = controllers_for(store, plan)
            game = RecordedGame(plan, store.config, controllers)
            while not game.done:
                if stop is not None and stop.is_set():
                    return False  # Keep the plan pending; deterministic retry.
                game.advance()
            store.complete(game)
            return True
        except Exception as exc:
            if isinstance(exc, PolicyError) and controllers:
                if game is None:
                    game = RecordedGame(plan, store.config, [None]*len(plan['entrant_ids']))
                    game.controllers = controllers
                store.fail_game(execution_failure(game, exc), game.replay_blob())
            store.fail(job["id"], exc)
            raise
        finally:
            if owned_worker is not None:
                owned_worker.close()
            if game is not None:
                close_game(game)
            else:
                for controller in controllers:
                    close = getattr(controller, 'close', None)
                    if close is not None:
                        close()
    return False


def simulate(directory, plan):
    """Worker processes simulate only. The coordinator commits in schedule order."""
    from .store import LeagueStore
    store = LeagueStore(directory)
    start = time.monotonic()
    controllers = controllers_for(store, plan)
    game = None
    try:
        game = RecordedGame(plan, store.config, controllers)
        while not game.done:
            game.advance()
        return {**execution_result(game), "simulation_seconds": time.monotonic() - start}, game.replay_blob()
    except PolicyError as exc:
        if game is None:
            game = RecordedGame(plan, store.config, [None]*len(plan['entrant_ids']))
            game.controllers = controllers
        return execution_failure(game, exc), game.replay_blob()
    finally:
        if game is not None:
            close_game(game)
        else:
            for controller in controllers:
                close = getattr(controller, 'close', None)
                if close is not None:
                    close()


def run_jobs(store, *, limit=None, retry=False, progress=None, workers=1, job_id=None):
    store.verify_execution()
    if limit is not None:
        integer("limit", limit, 1)
    integer("workers", workers, 1)
    if workers > 32:
        raise ValueError("workers must be <= 32")
    has_cpp = any(e['kind'] == 'cpp' for e in store.entrants().values())
    # Compile native libraries once before spawning, avoiding concurrent builds
    # on a cold cache. Workers still load their own library and RNG instances.
    from generals.backends.cpu.env import _library as cpu_library
    cpu_library()
    if any(e['kind'] == 'baseline' for e in store.entrants().values()):
        from generals_baselines.base import _library as baseline_library
        baseline_library()
    completed = 0
    with process_lock(store, ".worker.lock"):
        if retry:
            store.retry()
        if workers > 1:
            return _run_parallel(store, workers, limit, progress, job_id)
        from ..cpp_match import MatchWorker
        native_worker = MatchWorker() if has_cpp else None
        try:
            while limit is None or completed < limit:
                if not run_next(store, job_id=job_id, cpp_worker=native_worker):
                    break
                completed += 1
                if progress is not None:
                    progress(completed)
        finally:
            if native_worker is not None:
                native_worker.close()
    return completed


def _run_parallel(store, workers, limit, progress, job_id=None):
    completed = 0
    # Keep adaptation at the same persistent window as sequential execution.
    # Spawn works on macOS and does not inherit SQLite connections or locks.
    from ..cpp_match import NativeMatchPool
    native = any(e['kind'] == 'cpp' for e in store.entrants().values())
    pool = NativeMatchPool(workers) if native else ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context('spawn'))
    with pool as executor:
        while limit is None or completed < limit:
            job = next((j for j in store.jobs() if j["status"] == "queued"
                        and (job_id is None or j['id'] == job_id)), None)
            if job is None:
                break
            plans = prepare_batch(store, job)
            if not plans:
                break
            if limit is not None:
                plans = plans[:limit - completed]
            directory = str(store.directory.resolve())
            futures = [executor.submit_native(directory, plan) if native else executor.submit(simulate, directory, plan)
                       for plan in plans]
            first_error = None
            for future in futures:
                if future.cancelled():
                    continue
                try:
                    result, blob = future.result()
                    if 'policy_error' in result:
                        store.fail_game(result, blob)
                        raise PolicyError(result['policy_error'], details=result.get('error_detail'))
                    store.complete_result(result, blob)
                    completed += 1
                    if progress is not None:
                        progress(completed)
                except Exception as exc:
                    first_error = first_error or exc
                    for queued in futures:
                        queued.cancel()
                    # Drain started games and preserve their results/violations.
                    # A failed fast worker must not erase another timeout.
            if first_error is not None:
                store.fail(job["id"], first_error)
                raise first_error
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
        with self.lock:
            for value in self.games.values():
                close_game(value['game'])
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
            plans = eligible_schedule(self.store, job, human=True, players=players)
            if not plans:
                raise ValueError("register more bots before starting this game")
            plan = plans[0]
            self.store.insert_plans([plan], status="live")
            controllers, game = [], None
            try:
                entries = self.store.entrants()
                if any(entries[key]['kind']=='cpp' for key in plan['entrant_ids']):
                    from ..cpp_match import HumanCppGame
                    game = HumanCppGame(self.store, plan)
                else:
                    controllers = controllers_for(self.store, plan)
                    game = RecordedGame(plan, self.store.config, controllers)
            except Exception as exc:
                if hasattr(exc, 'partial'):
                    self.store.fail_game(*exc.partial)
                elif isinstance(exc, PolicyError):
                    game = RecordedGame(plan, self.store.config, [None]*len(plan['entrant_ids']))
                    game.controllers = controllers
                    self.store.fail_game(execution_failure(game, exc), game.replay_blob())
                else:
                    self.store.abort_live(plan['game_id'])
                if game is not None:
                    close_game(game)
                else:
                    for controller in controllers:
                        close = getattr(controller, 'close', None)
                        if close is not None:
                            close()
                raise
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
            value["board"] = human_board(game.state, value["seat"])
            from ..execution import mark_forfeits
            mark_forfeits(value['board'], getattr(game, 'forfeits', []))
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
                            close_game(value['game'])
                    except Exception as exc:
                        if isinstance(exc, PolicyError):
                            self.store.fail_game(execution_failure(value['game'], exc), value['game'].replay_blob())
                        else:
                            self.store.abort_live(game_id)
                        value["error"] = str(exc)
                        value["finished"] = time.monotonic()
                        close_game(value['game'])
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
