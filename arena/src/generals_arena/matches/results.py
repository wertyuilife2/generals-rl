"""Physical match results and explicit participant outcome summaries."""

from generals.core.recording import state_digest
from generals.core.state import rank_players
from generals.protocol.rules import END_REASONS

EXECUTION_POLICY = "official-surrender-grace-v3"


def _list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def build_result(plan, *, state, stats, elimination_ticks, invalid_actions,
                 initial_digest):
    """Describe an authoritative final state without adding a scoring mode.

    The caller decides when its match has ended. Schedule fields remain intact,
    so seat rotations, policy identities and rating evidence travel together.
    """
    names = ("army", "territory", "cities")
    final_stats = {name: _list(stats[name] if isinstance(stats, dict) else getattr(stats, name))
                   for name in names}
    winner = state.winner_id
    value = {**plan, "winner_seat": None if winner is None else int(winner),
             "ticks": int(state.tick), "reason": END_REASONS[state.end_reason] if state.terminated else "tick_limit",
             "elimination_ticks": [None if tick is None else int(tick) for tick in elimination_ticks],
             "invalid_actions": [int(count) for count in invalid_actions],
             "final_alive": [bool(alive) for alive in state.alive],
             "initial_digest": initial_digest, "final_digest": state_digest(state),
             "final_stats": final_stats, "kills": state.kills.tolist(),
             "death_order": state.death_order.tolist()}
    # The official FFA leaderboard prioritizes having at least one kill, then
    # survival/death order. A bounded, unfinished duel remains a draw.
    if state.player_count == 2:
        value['placements'] = [1.5 if winner is None else 1 if p == winner else 2 for p in range(2)]
    else:
        order = rank_players(state, final_stats)
        value['placements'] = [order.index(p) + 1 for p in range(state.player_count)]
    return value


def seat_outcome(result, seat):
    """Eliminated players lose even when surviving players reach the tick cap."""
    winner = result["winner_seat"]
    if winner is not None:
        return "win" if winner == seat else "loss"
    return "draw" if result["final_alive"][seat] else "loss"


def _entrant_outcome(result, entrant_id):
    seats = [seat for seat, key in enumerate(result["entrant_ids"]) if key == entrant_id]
    if not seats:
        raise ValueError("entrant did not participate in this result")
    outcomes = [seat_outcome(result, seat) for seat in seats]
    return "win" if "win" in outcomes else "draw" if "draw" in outcomes else "loss"


def aggregate(results, entrant_id=None, *, seat=None):
    """Summarize explicit participation without counting duplicate seats as games."""
    if (entrant_id is None) == (seat is None):
        raise ValueError("aggregate requires an entrant or a seat")
    rows = [row for row in results if entrant_id is None or entrant_id in row["entrant_ids"]]
    outcomes = [_entrant_outcome(row, entrant_id) if entrant_id is not None else seat_outcome(row, seat)
                for row in rows]
    return {"games": len(rows), "wins": outcomes.count("win"),
            "losses": outcomes.count("loss"), "draws": outcomes.count("draw"),
            "mean_ticks": sum(row["ticks"] for row in rows) / len(rows) if rows else None}


def win_matrix(results):
    """Physical duel W/D/L, from each row policy's perspective; absent pairs stay absent."""
    matrix = {}
    for result in results:
        ids = result['entrant_ids']
        if len(ids) != 2 or ids[0] == ids[1]:
            continue
        for seat, key in enumerate(ids):
            cell = matrix.setdefault(key, {}).setdefault(ids[1-seat], dict(games=0, wins=0, draws=0, losses=0))
            cell['games'] += 1
            cell[{'win': 'wins', 'draw': 'draws', 'loss': 'losses'}[seat_outcome(result, seat)]] += 1
    for opponents in matrix.values():
        for cell in opponents.values():
            cell['win_rate'] = cell['wins'] / cell['games']
            cell['score_rate'] = (cell['wins'] + .5 * cell['draws']) / cell['games']
    return matrix
