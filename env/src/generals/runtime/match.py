"""Run existing controllers against either match backend, without network callbacks."""
from collections.abc import Mapping
from time import perf_counter_ns

from generals.protocol.actions import Action
from generals.protocol.match import StaleFrameError


def run_match(backend, controllers, *, seed=42, max_decisions=None):
    """Own backend/controller cleanup. A single controller binds to the online seat.

    NativeController retains its existing hard decision watchdog. Transport waits
    are outside act(). No synthetic MoveResult is delivered when unavailable.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if max_decisions is not None and (isinstance(max_decisions, bool) or
            not isinstance(max_decisions, int) or max_decisions < 1):
        raise ValueError('max_decisions must be positive')
    supplied = dict(controllers) if isinstance(controllers, Mapping) else None
    owned = list(supplied.values()) if supplied is not None else [controllers]
    if len({id(c) for c in owned}) != len(owned):
        raise ValueError('each controlled seat requires its own controller')
    started, act_ns, calls, stale = perf_counter_ns(), 0, 0, 0
    bound, frame = {}, None
    stop_reason = None
    try:
        frame = backend.start()
        if not frame.done:
            if supplied is None:
                if len(frame.observations) != 1:
                    raise ValueError('multiple local seats require a controller mapping')
                supplied = {next(iter(frame.observations)): controllers}
            if set(supplied) != set(frame.observations):
                raise ValueError('controller mapping does not match controlled seats')
            for pid, controller in supplied.items():
                if getattr(controller, 'requires_move_results', False) and not backend.supports_move_results:
                    raise ValueError('controller requires authoritative move results unavailable online')
                controller.reset(player_id=pid, seed=seed + pid)
                bound[pid] = controller
        while not frame.done:
            if max_decisions is not None and calls >= max_decisions:
                stop_reason = 'decision_limit'
                break
            actions = {}
            for pid, observation in frame.observations.items():
                before = perf_counter_ns()
                action = bound[pid].act(observation)
                act_ns += perf_counter_ns() - before
                calls += 1
                if not isinstance(action, Action):
                    raise TypeError('controller must return an Action')
                actions[pid] = action
            try:
                frame = backend.step(actions, frame_id=frame.frame_id)
            except StaleFrameError as exc:
                stale += 1
                frame = exc.frame
                continue
            for result in frame.results:
                bound[result.player_id].on_result(result)
        winner = next((obs.winner_id for obs in frame.observations.values()
                       if obs.winner_id is not None), None)
        return {'backend': type(backend).__name__, 'done': frame.done,
                'reason': stop_reason or frame.reason, 'winner_id': winner,
                'tick': max((o.tick for o in frame.observations.values()), default=None),
                'decisions': calls, 'stale_decisions': stale, 'controller_ns': act_ns,
                'elapsed_seconds': (perf_counter_ns() - started) / 1e9,
                'replay_url': getattr(backend, 'replay_url', None),
                'decision_timing': {pid: c.decision_metrics for pid, c in bound.items()
                                    if hasattr(c, 'decision_metrics')}}
    finally:
        try:
            backend.close()
        finally:
            for controller in owned:
                close = getattr(controller, 'close', None)
                if close is not None:
                    close()
