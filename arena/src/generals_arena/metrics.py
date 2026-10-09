"""Shared win/loss/draw summaries for arena and training evaluations."""
import math


def wilson_interval(wins, games):
    """Two-sided 95% Wilson interval; an empty decisive sample is undefined."""
    if not 0 <= wins <= games:
        raise ValueError("wins must be between zero and the number of games")
    if games == 0:
        return None
    z = 1.959963984540054
    fraction = wins / games
    denominator = 1 + z * z / games
    center = (fraction + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / games + z * z / (4 * games * games)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def aggregate_games(games):
    wins = sum(game["outcome"] == "win" for game in games)
    losses = sum(game["outcome"] == "loss" for game in games)
    draws = sum(game["outcome"] == "draw" for game in games)
    count, decisive = len(games), wins + losses
    return {
        "games": count, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / count if count else None,
        "decisive_win_rate": wins / decisive if decisive else None,
        "draw_rate": draws / count if count else None,
        "win_rate_wilson95": wilson_interval(wins, count),
        "decisive_win_rate_wilson95": wilson_interval(wins, decisive),
        "mean_ticks": sum(game["ticks"] for game in games) / count if count else None,
    }
