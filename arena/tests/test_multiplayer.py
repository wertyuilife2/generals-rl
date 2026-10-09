"""Multiplayer seat balance, elimination lifecycle, and simultaneous Elo."""
from collections import defaultdict
import json
from math import comb
from pathlib import Path
import tempfile
import unittest

import numpy as np

from generals import Action, Direction, MapConfig
from generals.core.state import GameState, MapLayout
from generals_arena.config import ArenaRules
from generals_arena.leaderboard import build_leaderboard
from generals_arena.match import play_match
from generals_arena.ratings.elo import update, update_multiplayer
from generals_arena.schedule import make_schedule
from generals_arena.tournament import rebuild, run_tournament
from test_arena import WaitController, small_config


class MarchLeft(WaitController):
    def act(self, observation):
        super().act(observation)
        return Action.move(3 - observation.tick, Direction.LEFT)


def four_player_state():
    return GameState(MapLayout(np.zeros((1, 4), dtype=np.uint8)),
                     np.full((1, 4), 2, dtype=np.uint8), np.arange(4, dtype=np.int16).reshape(1, 4),
                     np.array([[1, 1, 1, 30]], dtype=np.int64), np.ones(4, dtype=bool),
                     np.arange(4, dtype=np.int32))


def multiplayer_config():
    config = small_config()
    config["rules"]["map"]["players"] = 4
    config.pop("maps_per_pair")
    config["maps_per_group"] = 1
    config["entrants"] = [{"name": kind, "kind": "baseline", "policy": kind}
                           for kind in ("aggressive", "expansion", "defensive", "random")]
    return config


class MultiplayerTests(unittest.TestCase):
    def test_group_combinations_rotate_every_player_through_every_seat(self):
        for players in (3, 4, 8):
            ids = [str(index) for index in range(players + 1)]
            schedule = make_schedule(ids, rules_id="r", players=players, maps_per_group=2)
            self.assertEqual(len(schedule), comb(len(ids), players) * players * 2)
            self.assertEqual(len({spec.game_id for spec in schedule}), len(schedule))
            groups = defaultdict(list)
            for spec in schedule:
                groups[spec.map_index, frozenset(spec.entrant_ids)].append(spec)
            for (_, group), games in groups.items():
                self.assertEqual(len(games), players)
                self.assertEqual(len({game.map_seed for game in games}), 1)
                for entrant in group:
                    seats = [game.entrant_ids.index(entrant) for game in games]
                    self.assertEqual(sorted(seats), list(range(players)))
                    self.assertEqual(len({game.controller_seeds[seat] for game, seat in zip(games, seats)}), 1)
            smaller = make_schedule(ids[:-1], rules_id="r", players=players, maps_per_group=1)
            self.assertTrue(set(smaller) <= set(schedule))

    def test_invalid_group_sizes_and_ambiguous_budgets_are_rejected(self):
        for options in ({"players": 1}, {"players": 9}, {"players": 5},
                        {"maps_per_pair": 1, "maps_per_group": 1}):
            with self.assertRaises(ValueError):
                make_schedule(["a", "b", "c", "d"], rules_id="r", **options)

    def test_eliminated_players_stop_acting_and_high_seat_can_win(self):
        rules = ArenaRules(MapConfig(4, 1, 4, 0, 0), max_ticks=3)
        spec = make_schedule(["a", "b", "c", "d"], rules_id="r", players=4, maps_per_group=1)[0]
        controllers = [WaitController(), WaitController(), WaitController(), MarchLeft()]
        result = play_match(spec, rules, controllers, state=four_player_state())
        self.assertEqual(result["winner_seat"], 3)
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["elimination_ticks"], [3, 2, 1, None])
        self.assertEqual(result["final_alive"], [False, False, False, True])
        self.assertEqual([len(controller.observations) for controller in controllers], [3, 2, 1, 3])
        self.assertEqual(result["invalid_actions"], [0, 0, 0, 0])
        self.assertEqual(len(result["final_stats"]), 4)

    def test_timeout_without_final_winner_draws_even_after_an_elimination(self):
        rules = ArenaRules(MapConfig(4, 1, 4, 0, 0), max_ticks=2)
        spec = make_schedule(["a", "b", "c", "d"], rules_id="r", players=4, maps_per_group=1)[0]
        result = play_match(spec, rules, [WaitController(), WaitController(), WaitController(), MarchLeft()],
                            state=four_player_state())
        self.assertIsNone(result["winner_seat"])
        self.assertTrue(result["truncated"])
        self.assertEqual(sum(result["final_alive"]), 2)
        result.update(sequence=0, rules_id="r")
        report = build_leaderboard([{"name": key, "entrant_id": key} for key in spec.entrant_ids], [result])
        self.assertTrue(all(row["elo"] == 1500 and row["draws"] == 1 for row in report["ranking"]))

    def test_multiplayer_elo_uses_pre_match_ratings_and_is_zero_sum_and_seat_invariant(self):
        equal = update_multiplayer([1500] * 4, 0)
        self.assertEqual(equal[0], 1516)
        for rating in equal[1:]:
            self.assertAlmostEqual(rating, 1500 - 16 / 3)
        for players in (3, 4, 8):
            ratings = [1250 + 80 * index for index in range(players)]
            for winner in (None, players - 1):
                result = update_multiplayer(ratings, winner)
                self.assertAlmostEqual(sum(result), sum(ratings))
                reverse = update_multiplayer(ratings[::-1], None if winner is None else players - 1 - winner)
                np.testing.assert_allclose(result, reverse[::-1], rtol=0, atol=1e-10)
        for winner, score in ((0, 1), (1, 0), (None, 0.5)):
            self.assertEqual(tuple(update_multiplayer([1600, 1400], winner)), update(1600, 1400, score))

    def test_game_losses_and_pairwise_ties_are_reported_separately(self):
        entrants = [{"name": key, "entrant_id": key} for key in ("a", "b", "c", "d")]
        match = {"game_id": "game", "sequence": 0, "rules_id": "r",
                 "entrant_ids": ["a", "b", "c", "d"], "winner_seat": 3}
        result = build_leaderboard(entrants, [match], rules_id="r")
        rows = {row["name"]: row for row in result["ranking"]}
        self.assertEqual(rows["d"]["wins"], 1)
        self.assertEqual(rows["a"]["losses"], 1)
        self.assertEqual(rows["a"]["games"], 1)
        self.assertEqual(rows["a"]["score_rate"], 0)
        self.assertAlmostEqual(rows["a"]["pairwise_score_rate"], 1 / 3)
        self.assertEqual(result["head_to_head"]["a"]["b"], {"wins": 0, "losses": 0, "draws": 1})
        self.assertEqual(result["head_to_head"]["a"]["d"], {"wins": 0, "losses": 1, "draws": 0})

    def test_multiplayer_resume_extend_and_validate_record_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "league"
            config = multiplayer_config()
            result = run_tournament(config, output)
            self.assertEqual(result["completed_games"], 4)
            self.assertEqual(result, rebuild(output))
            self.assertEqual(result, run_tournament(config, output, resume=True))
            original = (output / "matches.jsonl").read_bytes()
            config["entrants"].append({"name": "another-random", "kind": "baseline", "policy": "random"})
            self.assertEqual(run_tournament(config, output, resume=True)["completed_games"], 20)
            self.assertTrue((output / "matches.jsonl").read_bytes().startswith(original))
            path = output / "matches.jsonl"
            good = path.read_text()
            first, *rest = good.splitlines()
            for field, value in (("final_stats", []), ("winner_seat", 4), ("final_alive", [True] * 3)):
                bad = json.loads(first)
                bad[field] = value
                path.write_text("\n".join([json.dumps(bad), *rest]) + "\n")
                with self.assertRaises(ValueError):
                    rebuild(output)

    def test_old_two_player_budget_and_records_still_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "league"
            result = run_tournament(small_config(), output)
            config_path = output / "config.json"
            config = json.loads(config_path.read_text())
            config["maps_per_pair"] = config.pop("maps_per_group")
            config_path.write_text(json.dumps(config))
            path = output / "matches.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                record.pop("final_alive")
                record.pop("elimination_ticks")
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            self.assertEqual(result, rebuild(output))
