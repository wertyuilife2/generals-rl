"""Drive one public-observation controller from the official server stream."""
from time import perf_counter_ns

from generals import Action
from generals.backends.online.backend import StaleFrameError


def run_online(backend, controller_factory, *, seed=42, max_decisions=None):
    """Build a controller from the first observation; own both lifecycles.

    Network updates may skip ticks or supersede an in-flight decision. No move
    receipts are fabricated: only the server knows which queued move executed.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if max_decisions is not None and (type(max_decisions) is not int or max_decisions < 1):
        raise ValueError('max_decisions must be positive')
    started, act_ns, calls, stale = perf_counter_ns(), 0, 0, 0
    controller, stop_reason = None, None
    try:
        frame = backend.start()
        if not frame.done:
            observation, = frame.observations.values()
            controller = controller_factory(observation)
            if getattr(controller, 'requires_move_results', False):
                raise ValueError('controller requires authoritative move results unavailable online')
            controller.reset(player_id=observation.player_id, seed=seed + observation.player_id)
        while not frame.done:
            if max_decisions is not None and calls >= max_decisions:
                stop_reason = 'decision_limit'
                break
            observation, = frame.observations.values()
            before = perf_counter_ns()
            action = controller.act(observation)
            act_ns += perf_counter_ns() - before
            calls += 1
            if not isinstance(action, Action):
                raise TypeError('controller must return an Action')
            try:
                frame = backend.step({observation.player_id: action}, frame_id=frame.frame_id)
            except StaleFrameError as exc:
                stale += 1
                frame = exc.frame
        observation = next(iter(frame.observations.values()), None)
        return {'backend': 'online', 'done': frame.done, 'reason': stop_reason or frame.reason,
                'winner_id': observation.winner_id if observation else None,
                'tick': observation.tick if observation else None,
                'decisions': calls, 'stale_decisions': stale, 'controller_ns': act_ns,
                'elapsed_seconds': (perf_counter_ns() - started) / 1e9,
                'replay_url': backend.replay_url,
                'decision_timing': getattr(controller, 'decision_metrics', None)}
    finally:
        try:
            backend.close()
        finally:
            if controller is not None:
                close = getattr(controller, 'close', None)
                if close is not None:
                    close()
