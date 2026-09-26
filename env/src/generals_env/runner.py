"""Tick orchestration independent of rendering and wall-clock pacing."""
from time import perf_counter
from .config import RunConfig
from .engine import CoreEngine
from .observation import observe
from .state import Action


class ControllerError(RuntimeError):
    pass


class Runner:
    def __init__(self, state, controllers, seed=0, visibility="local", max_ticks=None,
                 debug=False, recorder=None):
        self.config = RunConfig(seed, max_ticks, visibility, debug)
        if len(controllers) != state.player_count:
            raise ValueError("one controller is required for each player slot")
        self.engine = CoreEngine(state, debug=debug)
        self.controllers = list(controllers)
        self.visibility = visibility
        self.recorder = recorder
        self.stopped = bool(state.terminated)
        self.stop_reason = "terminated" if state.terminated else None
        self.truncated = False
        self.error = None
        self.last_result = None
        self.elapsed_seconds = 0.0
        self.initial_tick = state.tick
        self._check_limit()

    def _check_limit(self):
        if self.engine.state.terminated:
            self.stopped, self.stop_reason = True, "terminated"
        elif self.config.max_ticks is not None and self.engine.state.tick >= self.config.max_ticks:
            self.stopped, self.truncated, self.stop_reason = True, True, "max_ticks"

    def tick(self):
        if self.stopped:
            raise RuntimeError(f"runner is stopped: {self.stop_reason}")
        started = perf_counter()
        self.engine.begin_tick()
        actions = [Action.wait() for _ in self.controllers]
        failure = None
        while self.engine.next_player is not None:
            pid = self.engine.next_player
            try:
                obs = observe(self.engine.state, pid, self.visibility)
                action = self.controllers[pid].act(obs)
                if not isinstance(action, Action):
                    raise TypeError("controller must return an Action")
                result = self.engine.apply_action(pid, action)
                actions[pid] = action
                self.controllers[pid].on_result(result)
                if result.eliminated is not None:
                    defeated = self.controllers[result.eliminated]
                    if hasattr(defeated, "eliminate"):
                        defeated.eliminate()
            except Exception as exc:
                failure = ControllerError(f"player {pid}: {type(exc).__name__}: {exc}")
                self.error = str(failure)
                # Finish the atomic tick with WAIT; no other policy is called.
                while self.engine.next_player is not None:
                    self.engine.apply_action(self.engine.next_player, Action.wait())
                break
        result = self.engine.finish_tick()
        self.last_result = result
        self._check_limit()
        if failure is not None:
            self.stopped, self.truncated, self.stop_reason = True, False, "controller_error"
        try:
            if self.recorder is not None:
                self.recorder.write_tick(result, actions, self.engine.state)
        except Exception as exc:
            self.stopped, self.stop_reason = True, "recording_error"
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.elapsed_seconds += perf_counter() - started
        if failure is not None:
            raise failure
        return result

    def stop(self, reason="user_stop"):
        if not self.stopped:
            self.stopped, self.stop_reason = True, reason

    def run(self):
        while not self.stopped:
            self.tick()
        return self.summary()

    def summary(self):
        state = self.engine.state
        return {"seed": self.config.seed, "ticks": state.tick,
                "advanced_ticks": state.tick - self.initial_tick,
                "terminated": bool(state.terminated), "truncated": self.truncated,
                "winner_id": state.winner_id, "stop_reason": self.stop_reason,
                "error": self.error, "elapsed_seconds": self.elapsed_seconds,
                "ticks_per_second": (state.tick - self.initial_tick) / self.elapsed_seconds if self.elapsed_seconds else 0.0}
