"""Physical match results and explicit participant outcome summaries."""

from generals.core.recording import state_digest

EXECUTION_POLICY = "decision-timeout-death-v2"


def _list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def build_result(plan, *, state, stats, elimination_ticks, invalid_actions,
                 initial_digest, kills=None):
    """Describe an authoritative final state without adding a scoring mode.

    The caller decides when its match has ended. Schedule fields remain intact,
    so seat rotations, policy identities and rating evidence travel together.
    """
    names = ("army", "territory", "cities")
    final_stats = {name: _list(stats[name] if isinstance(stats, dict) else getattr(stats, name))
                   for name in names}
    winner = state.winner_id
    value = {**plan, "winner_seat": None if winner is None else int(winner),
             "ticks": int(state.tick), "reason": "capture" if state.terminated else "tick_limit",
             "elimination_ticks": [None if tick is None else int(tick) for tick in elimination_ticks],
             "invalid_actions": [int(count) for count in invalid_actions],
             "final_alive": [bool(alive) for alive in state.alive],
             "initial_digest": initial_digest, "final_digest": state_digest(state),
             "final_stats": final_stats}
    if kills is not None:
        value["kills"] = [int(count) for count in kills]
    return value


def seat_outcome(result, seat):
    """Eliminated players lose even when surviving players reach the tick cap."""
    if not result["final_alive"][seat]:
        return "loss"
    winner = result["winner_seat"]
    return "draw" if winner is None else "win" if winner == seat else "loss"


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
