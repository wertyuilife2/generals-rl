"""Operational diagnostics and resource ownership outside the frozen game rules."""
from generals.core.recording import state_digest


def timing(game):
    return [{'seat': seat, 'entrant_id': game.plan['entrant_ids'][seat], **controller.decision_metrics}
            for seat, controller in enumerate(game.controllers)
            if hasattr(controller, 'decision_metrics')]


def result(game):
    value = game.result()
    metrics = timing(game)
    return {**value, 'decision_timing': metrics} if metrics else value


def failure(game, error):
    return {**game.plan, 'ticks': game.state.tick, 'initial_digest': game.initial_digest,
            'final_digest': state_digest(game.state), 'reason': 'policy_error',
            'policy_error': str(error), 'error_detail': getattr(error, 'details', {}),
            'decision_timing': timing(game)}


def close_game(game):
    for controller in game.controllers:
        close = getattr(controller, 'close', None)
        if close is not None:
            close()
    close = getattr(game.engine, 'close', None)
    if close is not None:
        close()
