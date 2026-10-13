"""Matches using the environment's native CPU backend and joint decisions."""
from generals import make_engine
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.recording import state_digest
from generals.core.state import Action, compute_stats


def play_match(spec, rules, controllers, *, state=None):
    players = rules.map.players
    if (len(controllers) != players or len({id(controller) for controller in controllers}) != players or
            len(spec.entrant_ids) != players or len(set(spec.entrant_ids)) != players or
            len(spec.controller_seeds) != players):
        raise ValueError("provide one independent controller and a matching spec per player")
    state = generate_map(rules.map, spec.map_seed) if state is None else state.copy()
    if (state.player_count != players or state.tick != 0 or state.terminated or not state.alive.all() or
            state.owner.shape != (rules.map.height, rules.map.width)):
        raise ValueError("matches require fresh states matching the rules with every player alive")
    engine = make_engine(state)
    for seat, controller in enumerate(controllers):
        controller.reset(player_id=seat, seed=spec.controller_seeds[seat])
    initial_digest = state_digest(state)
    invalid = [0] * players
    elimination_ticks = [None] * players
    while not state.terminated and state.tick < rules.max_ticks:
        # Live policies observe tick t before growth or any action at t+1.
        actions = [controller.act(observe(state, seat, rules.visibility)) if state.alive[seat] else Action.wait()
                   for seat, controller in enumerate(controllers)]
        if not all(isinstance(action, Action) for action in actions):
            raise TypeError("controllers must return an Action")
        result = engine.step(actions)
        for move in result.moves:
            invalid[move.player_id] += int(not move.valid)
            controllers[move.player_id].on_result(move)
            if move.eliminated is not None:
                elimination_ticks[move.eliminated] = state.tick
                eliminate = getattr(controllers[move.eliminated], "eliminate", None)
                if eliminate is not None:
                    eliminate()
    stats = compute_stats(state)
    return {**spec.to_dict(), "winner_seat": state.winner_id if state.terminated else None,
            "terminated": state.terminated, "truncated": not state.terminated,
            "reason": "capture" if state.terminated else "tick_limit", "ticks": state.tick,
            "invalid_actions": invalid, "initial_digest": initial_digest,
            "final_digest": state_digest(state),
            "final_alive": [bool(alive) for alive in state.alive], "elimination_ticks": elimination_ticks,
            "final_stats": [{key: int(getattr(stats, key)[seat]) for key in ("army", "territory", "cities")}
                            for seat in range(players)]}
