"""Physical match results and explicit participant outcome summaries."""

EXECUTION_POLICY = "official-native-session-v4"


def build_result(plan, *, state, stats, elimination_ticks, invalid_actions, initial_digest):
    """Compatibility projection; all physical outcomes are computed natively."""
    from types import SimpleNamespace
    import numpy as np
    from .game import RecordedGame
    p = state.player_count
    context = dict(entrant_ids=[str(i) for i in range(p)], controller_seeds=[0]*p, map={}, map_seed=0)
    game = RecordedGame(context, SimpleNamespace(max_ticks=max(1,state.tick+1)), [None]*p,
                        initial=state, record=False)
    try:
        game._invalid[0] = invalid_actions
        game._eliminations[0] = [-1 if t is None else t for t in elimination_ticks]
        game.env.stats[0] = np.stack([stats[k] if isinstance(stats,dict) else getattr(stats,k)
                                     for k in ('army','territory','cities')],axis=-1)
        value = game.execution.query(0)
        value['initial_digest'] = initial_digest
        return {**plan, **value}
    finally:
        game.close()


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
