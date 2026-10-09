"""Portable reference matches at the same joint decision boundary as training."""
from generals.core.engine import CoreEngine
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.recording import state_digest
from generals.core.state import Action, compute_stats


def play_match(spec, rules, controllers, *, state=None):
    if len(controllers) != 2 or controllers[0] is controllers[1]:
        raise ValueError("provide two independent controllers")
    state = generate_map(rules.map, spec.map_seed) if state is None else state.copy()
    if (state.player_count != 2 or state.tick != 0 or state.terminated or
            state.owner.shape != (rules.map.height, rules.map.width)):
        raise ValueError("matches require fresh two-player states matching the rules")
    engine = CoreEngine(state)
    for seat, controller in enumerate(controllers):
        controller.reset(player_id=seat, seed=spec.controller_seeds[seat])
    initial_digest = state_digest(state)
    invalid = [0, 0]
    while not state.terminated and state.tick < rules.max_ticks:
        # Both policies observe tick t before growth or any action at t+1.
        actions = [controller.act(observe(state, seat, rules.visibility))
                   for seat, controller in enumerate(controllers)]
        if not all(isinstance(action, Action) for action in actions):
            raise TypeError("controllers must return an Action")
        result = engine.step(actions)
        for move in result.moves:
            invalid[move.player_id] += int(not move.valid)
            controllers[move.player_id].on_result(move)
            if move.eliminated is not None:
                eliminate = getattr(controllers[move.eliminated], "eliminate", None)
                if eliminate is not None:
                    eliminate()
    stats = compute_stats(state)
    return {**spec.to_dict(), "winner_seat": state.winner_id if state.terminated else None,
            "terminated": state.terminated, "truncated": not state.terminated,
            "reason": "capture" if state.terminated else "tick_limit", "ticks": state.tick,
            "invalid_actions": invalid, "initial_digest": initial_digest,
            "final_digest": state_digest(state),
            "final_stats": [{key: int(getattr(stats, key)[seat]) for key in ("army", "territory", "cities")}
                            for seat in (0, 1)]}
