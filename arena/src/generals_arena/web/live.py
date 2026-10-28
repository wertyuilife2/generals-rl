"""Clock-driven human sessions and fog presentation over shared matches."""
from collections import deque
import threading
import time
import uuid

from generals.core.observation import legal_action_mask, observe
from generals.core.state import Action
from generals_arena.util import integer, process_lock
from generals_arena.matches.game import RecordedGame, close_game, failure as execution_failure, mark_forfeits
from generals_arena.matches.runner import build_controllers
from generals_arena.matches.observation import board_view
from generals_arena.policies.native import PolicyError
from generals_arena.competitions.jobs import eligible_schedule, run_next


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
                    from generals_arena.matches.worker import HumanCppGame
                    game = HumanCppGame(entries, plan, max_ticks=self.store.config.max_ticks)
                else:
                    controllers = build_controllers(entries, plan)
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
                     "queue": deque(), "last_command_id": 0,
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
        if op not in ("append", "clear", "undo", "surrender"):
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
            if value["finished"] or not game.state.alive[seat] or game.state.leave_tick[seat] >= 0:
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
            elif op == "surrender":
                queue.clear()
                game.surrender(seat)
                value['board'] = None
                if game.done:
                    self.store.complete(game)
                    value['finished'] = time.monotonic()
                    close_game(game)
            elif op == "clear":
                queue.clear()
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
                        action = Action.wait()
                        if value['queue']:
                            legal, _ = legal_action_mask(observe(value['game'].state, value['seat']))
                            while value['queue']:
                                encoded = value['queue'].popleft()['action']
                                if encoded == 0 or legal.ravel()[encoded - 1]:
                                    action = Action.decode(encoded)
                                    break
                        value["game"].advance({value["seat"]: action})
                        value["board"] = None
                        value["revision"] += 1
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
            with process_lock(self.store.directory, ".worker.lock"):
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


def human_board(state, seat):
    return board_view(state, seat)
