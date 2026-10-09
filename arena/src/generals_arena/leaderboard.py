"""Rebuildable Elo table and head-to-head results, in logged match order."""
from dataclasses import asdict

from .ratings.elo import EloConfig, update


def build_leaderboard(entrants, matches, *, elo=EloConfig(), rules_id=None):
    entries = {entry["entrant_id"]: entry for entry in entrants}
    ratings = {key: float(elo.initial) for key in entries}
    totals = {key: {"wins": 0, "losses": 0, "draws": 0} for key in entries}
    matrix = {key: {} for key in entries}
    seen = set()
    for sequence, match in enumerate(matches):
        if match["game_id"] in seen or match["sequence"] != sequence:
            raise ValueError("duplicate match or noncontiguous sequence")
        if rules_id is not None and match["rules_id"] != rules_id:
            raise ValueError("cannot combine different arena rules")
        seen.add(match["game_id"])
        left, right = match["entrant_ids"]
        if left == right or left not in entries or right not in entries:
            raise ValueError("invalid match participants")
        winner = match["winner_seat"]
        if winner not in (None, 0, 1):
            raise ValueError("invalid winner seat")
        score = 0.5 if winner is None else float(winner == 0)
        ratings[left], ratings[right] = update(ratings[left], ratings[right], score, elo)
        for seat, (player, opponent) in enumerate(((left, right), (right, left))):
            outcome = "draws" if winner is None else ("wins" if winner == seat else "losses")
            totals[player][outcome] += 1
            cell = matrix[player].setdefault(opponent, {"wins": 0, "losses": 0, "draws": 0})
            cell[outcome] += 1
    rows = []
    for key, entry in entries.items():
        counts = totals[key]
        played = sum(counts.values())
        rows.append({"entrant_id": key, "name": entry["name"], "elo": ratings[key],
                     "games": played, **counts,
                     "win_rate": counts["wins"] / played if played else None,
                     "score_rate": (counts["wins"] + 0.5 * counts["draws"]) / played if played else None})
    rows.sort(key=lambda row: (-row["elo"], row["name"]))
    return {"rules_id": rules_id, "elo": asdict(elo), "completed_games": len(matches),
            "ranking": rows, "head_to_head": matrix}
