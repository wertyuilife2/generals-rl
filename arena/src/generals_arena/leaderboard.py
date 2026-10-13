"""Rebuildable Elo table and head-to-head results, in logged match order."""
from dataclasses import asdict

from .ratings.elo import EloConfig, pair_score, update_multiplayer


def build_leaderboard(entrants, matches, *, elo=EloConfig(), rules_id=None):
    entries = {entry["entrant_id"]: entry for entry in entrants}
    ratings = {key: float(elo.initial) for key in entries}
    totals = {key: {"wins": 0, "losses": 0, "draws": 0} for key in entries}
    pair_points = dict.fromkeys(entries, 0.0)
    matrix = {key: {} for key in entries}
    seen = set()
    for sequence, match in enumerate(matches):
        if match["game_id"] in seen or match["sequence"] != sequence:
            raise ValueError("duplicate match or noncontiguous sequence")
        if rules_id is not None and match["rules_id"] != rules_id:
            raise ValueError("cannot combine different arena rules")
        seen.add(match["game_id"])
        players = match["entrant_ids"]
        count = len(players)
        if not 2 <= count <= 8 or len(set(players)) != count or any(key not in entries for key in players):
            raise ValueError("invalid match participants")
        winner = match["winner_seat"]
        updated = update_multiplayer([ratings[key] for key in players], winner, elo)
        ratings.update(zip(players, updated))
        for seat, player in enumerate(players):
            outcome = "draws" if winner is None else ("wins" if winner == seat else "losses")
            totals[player][outcome] += 1
            for other_seat, opponent in enumerate(players):
                if seat == other_seat:
                    continue
                score = pair_score(winner, seat, other_seat)
                pair_points[player] += score / (count - 1)
                comparison = {1.0: "wins", 0.5: "draws", 0.0: "losses"}[score]
                cell = matrix[player].setdefault(opponent, {"wins": 0, "losses": 0, "draws": 0})
                cell[comparison] += 1
    rows = []
    for key, entry in entries.items():
        counts = totals[key]
        played = sum(counts.values())
        rows.append({"entrant_id": key, "name": entry["name"], "elo": ratings[key],
                     "games": played, **counts,
                     "win_rate": counts["wins"] / played if played else None,
                     "score_rate": (counts["wins"] + 0.5 * counts["draws"]) / played if played else None,
                     "pairwise_score_rate": pair_points[key] / played if played else None})
    rows.sort(key=lambda row: (-row["elo"], row["name"]))
    return {"rules_id": rules_id, "elo": asdict(elo), "completed_games": len(matches),
            "ranking": rows, "head_to_head": matrix}
