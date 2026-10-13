"""Tournament semantics, immutable versions, recovery, and rating evidence."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from generals import Action, Direction, MapConfig
from generals.core.state import GameState, MapLayout
from generals_arena.config import ArenaRules
from generals_arena.entrants import prepare_entrant, verify_artifacts
from generals_arena.leaderboard import build_leaderboard
from generals_arena.match import play_match
from generals_arena.ratings.elo import EloConfig, expected_score, update
from generals_arena.schedule import make_schedule
from generals_arena.tournament import rebuild, run_tournament


class WaitController:
    def __init__(self, player_id=0, seed=0, **kwargs):
        self.reset(player_id, seed)

    def reset(self, player_id, seed):
        self.player_id, self.seed = player_id, seed
        self.observations, self.moves = [], []

    def act(self, observation):
        self.observations.append(observation)
        return Action.wait()

    def on_result(self, result):
        self.moves.append(result)


class AttackController(WaitController):
    def act(self, observation):
        super().act(observation)
        return Action.move(0, Direction.RIGHT)


class BrokenController(WaitController):
    def act(self, observation):
        raise RuntimeError("policy failed")


def small_config():
    return {"rules": {"map": {"width": 3, "height": 3, "players": 2,
                               "mountain_density": 0, "city_density": 0}, "max_ticks": 8},
            "maps_per_pair": 2,
            "entrants": [{"name": kind, "kind": "baseline", "policy": kind}
                         for kind in ("aggressive", "random")]}


def adjacent_state():
    owner = np.full((3, 3), -1, dtype=np.int16)
    structure = np.zeros((3, 3), dtype=np.uint8)
    army = np.zeros((3, 3), dtype=np.int64)
    owner.flat[0], owner.flat[1] = 0, 1
    structure.flat[0] = structure.flat[1] = 2
    army.flat[0], army.flat[1] = 10, 2
    return GameState(MapLayout(np.zeros((3, 3), dtype=np.uint8)), structure, owner, army,
                     np.ones(2, dtype=bool), np.array([0, 1], dtype=np.int32))


class ScheduleAndMatchTests(unittest.TestCase):
    def test_partial_map_overrides_retain_two_player_arena_defaults(self):
        rules = ArenaRules.from_dict({"map": {"width": 3, "height": 3}})
        self.assertEqual(rules.map.players, 2)

    def test_round_robin_balances_seats_and_keeps_existing_pairings_on_extension(self):
        schedule = make_schedule(["a", "b", "c"], rules_id="rules", maps_per_pair=3, seed=42)
        self.assertEqual(len(schedule), 18)
        self.assertEqual(len({s.game_id for s in schedule}), 18)
        smaller = make_schedule(["b", "a"], rules_id="rules", maps_per_pair=2, seed=42)
        self.assertTrue(set(smaller) <= set(schedule))
        for index in range(3):
            same_map = [s for s in schedule if s.map_index == index]
            self.assertEqual(len({s.map_seed for s in same_map}), 1)
            for first, second in zip(same_map[::2], same_map[1::2]):
                self.assertEqual(first.entrant_ids, second.entrant_ids[::-1])
                self.assertEqual(first.controller_seeds, second.controller_seeds[::-1])
        different = make_schedule(["a", "b"], rules_id="other", maps_per_pair=2, seed=42)
        self.assertFalse({s.game_id for s in smaller} & {s.game_id for s in different})

    def test_joint_pre_growth_observations_and_natural_win_at_limit(self):
        rules = ArenaRules(MapConfig(3, 3, 2, 0, 0), max_ticks=1)
        spec = make_schedule(["a", "b"], rules_id="rules", maps_per_pair=1)[0]
        state = adjacent_state()
        attack, wait = AttackController(), WaitController()
        result = play_match(spec, rules, [attack, wait], state=state)
        self.assertEqual(result["winner_seat"], 0)
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["ticks"], 1)
        self.assertEqual([p.observations[0].tick for p in (attack, wait)], [0, 0])
        self.assertEqual(wait.observations[0].army.flat[1], 2)
        self.assertEqual(attack.observations[0].stats.cities[1], -1)
        self.assertEqual(len(attack.moves), 1)
        self.assertEqual(wait.moves, [])  # Eliminated before its execution slot.
        self.assertEqual(state.tick, 0)
        self.assertEqual(state.army.flat[0], 10)

    def test_timeout_is_a_draw_and_reset_removes_previous_match_memory(self):
        rules = ArenaRules(MapConfig(3, 3, 2, 0, 0), max_ticks=2)
        spec = make_schedule(["a", "b"], rules_id="rules", maps_per_pair=1)[0]
        players = [WaitController(), WaitController()]
        first = play_match(spec, rules, players)
        second = play_match(spec, rules, players)
        self.assertEqual(first, second)
        self.assertEqual(first["reason"], "tick_limit")
        self.assertIsNone(first["winner_seat"])
        self.assertEqual(len(players[0].observations), 2)


class RatingTests(unittest.TestCase):
    def test_known_elo_updates_draws_and_zero_sum(self):
        self.assertEqual(update(1500, 1500, 1), (1516, 1484))
        self.assertEqual(update(1500, 1500, 0.5), (1500, 1500))
        self.assertAlmostEqual(expected_score(1500, 1900), 1 / 11)
        left, right = update(1900, 1500, 0.5)
        self.assertLess(left, 1900)
        self.assertEqual(left + right, 3400)
        self.assertTrue(math.isfinite(expected_score(-1e9, 1e9)))
        for values in ({"k_factor": 0}, {"scale": -1}, {"initial": float("nan")}):
            with self.assertRaises(ValueError):
                EloConfig(**values)

    def test_ranking_preserves_draws_and_orients_head_to_head(self):
        entrants = [{"entrant_id": key, "name": key} for key in ("a", "b", "c")]
        matches = [{"game_id": str(i), "sequence": i, "rules_id": "r", "entrant_ids": ["a", "b"],
                    "winner_seat": winner} for i, winner in enumerate((0, None, 1))]
        result = build_leaderboard(entrants, matches, rules_id="r")
        self.assertEqual(result["head_to_head"]["a"]["b"], {"wins": 1, "losses": 1, "draws": 1})
        a = next(row for row in result["ranking"] if row["name"] == "a")
        self.assertEqual(a["win_rate"], 1 / 3)
        self.assertEqual(a["score_rate"], 0.5)
        self.assertAlmostEqual(sum(row["elo"] for row in result["ranking"]), 4500)


class TournamentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "run"

    def test_rebuild_resume_and_append_new_versions_without_replaying_games(self):
        config = small_config()
        result = run_tournament(config, self.output)
        before = (self.output / "matches.jsonl").read_bytes()
        self.assertEqual(result["completed_games"], 4)
        self.assertEqual(result, rebuild(self.output))
        self.assertEqual(result, run_tournament(config, self.output, resume=True))
        self.assertEqual(before, (self.output / "matches.jsonl").read_bytes())
        config["entrants"].append({"name": "defensive", "kind": "baseline", "policy": "defensive"})
        extended = run_tournament(config, self.output, resume=True)
        self.assertEqual(extended["completed_games"], 12)
        self.assertTrue((self.output / "matches.jsonl").read_bytes().startswith(before))
        config["maps_per_pair"] = 3
        self.assertEqual(run_tournament(config, self.output, resume=True)["completed_games"], 18)

    def test_changed_rules_versions_removals_and_overwrite_are_rejected(self):
        original = small_config()
        run_tournament(original, self.output)
        changes = []
        changed = deepcopy(original)
        changed["rules"]["max_ticks"] += 1
        changes.append(changed)
        changed = deepcopy(original)
        changed["entrants"][0]["policy"] = "expansion"
        changes.append(changed)
        changed = deepcopy(original)
        changed["elo"] = {"k_factor": 16}
        changes.append(changed)
        changed = deepcopy(original)
        changed["maps_per_pair"] = 1
        changes.append(changed)
        before = (self.output / "matches.jsonl").read_bytes()
        for changed in changes:
            with self.assertRaises(ValueError):
                run_tournament(changed, self.output, resume=True)
        with self.assertRaises(FileExistsError):
            run_tournament(original, self.output)
        self.assertEqual(before, (self.output / "matches.jsonl").read_bytes())

    def test_interrupted_match_does_not_affect_ratings_and_resume_finishes(self):
        config = small_config()
        from generals_arena.tournament import play_match as original
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("interrupted policy")
            return original(*args, **kwargs)

        with patch("generals_arena.tournament.play_match", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_tournament(config, self.output)
        report = rebuild(self.output)
        self.assertEqual(report["completed_games"], 1)
        self.assertFalse(report["complete"])
        before = (self.output / "matches.jsonl").read_bytes()
        result = run_tournament(config, self.output, resume=True)
        self.assertEqual(result["completed_games"], 4)
        self.assertTrue((self.output / "matches.jsonl").read_bytes().startswith(before))
        self.assertFalse((self.output / "failure.json").exists())

    def test_corrupt_duplicate_and_incompatible_records_are_not_ranked(self):
        run_tournament(small_config(), self.output)
        path = self.output / "matches.jsonl"
        original = path.read_text()
        first = original.splitlines()[0]
        bad = json.loads(first)
        bad["rules_id"] = "other"
        for content in (original + first + "\n", first[:-3], json.dumps(bad) + "\n"):
            path.write_text(content)
            with self.assertRaises(ValueError):
                rebuild(self.output)

    def test_custom_factory_and_artifact_identity(self):
        artifact = self.root / "weights.bin"
        artifact.write_bytes(b"model-v1")
        spec = {"name": "custom", "kind": "custom", "factory": "test_arena:WaitController",
                "version": "commit-a", "artifacts": {"weights": "weights.bin"}}
        entrant = prepare_entrant(spec, base_dir=self.root)
        artifact.write_bytes(b"model-v2")
        changed = prepare_entrant(spec, base_dir=self.root)
        self.assertNotEqual(entrant["entrant_id"], changed["entrant_id"])
        with self.assertRaises(ValueError):
            verify_artifacts(entrant)
        spec.pop("version")
        with self.assertRaises(ValueError):
            prepare_entrant(spec, base_dir=self.root)
        config = small_config()
        config["entrants"][1] = {"name": "wait", "kind": "custom", "factory": "test_arena:WaitController", "version": "1"}
        self.assertTrue(run_tournament(config, self.output)["complete"])

    def test_bad_plugin_aborts_without_fabricating_a_result(self):
        config = small_config()
        config["entrants"][1] = {"name": "broken", "kind": "custom",
                                  "factory": "test_arena:BrokenController", "version": "1"}
        with self.assertRaisesRegex(RuntimeError, "policy failed"):
            run_tournament(config, self.output)
        report = rebuild(self.output)
        self.assertEqual(report["completed_games"], 0)
        self.assertTrue(all(row["elo"] == 1500 and row["games"] == 0 for row in report["ranking"]))

    def test_reading_rankings_does_not_require_original_model_artifacts(self):
        artifact = self.root / "model.bin"
        artifact.write_bytes(b"version-1")
        config = small_config()
        config["entrants"][1] = {"name": "custom", "kind": "custom", "factory": "test_arena:WaitController",
                                  "version": "1", "artifacts": {"checkpoint": str(artifact)}}
        result = run_tournament(config, self.output)
        artifact.unlink()
        self.assertEqual(result, rebuild(self.output))

    def test_independent_runs_are_identical(self):
        first = run_tournament(small_config(), self.output)
        second = run_tournament(small_config(), self.root / "other")
        self.assertEqual(first, second)
        self.assertEqual((self.output / "matches.jsonl").read_bytes(), (self.root / "other/matches.jsonl").read_bytes())

    def test_cli_and_baseline_only_dependency_boundary(self):
        league_config = self.root / "league.json"
        league_config.write_text(json.dumps({"player_counts": [2], "max_ticks": 8}))
        entrant = self.root / "entrant.json"
        entrant.write_text(json.dumps({"name": "aggressive", "kind": "baseline", "policy": "aggressive"}))
        commands = [
            ["league", "init", str(self.output), "--config", str(league_config)],
            ["league", "add", str(self.output), "--entrant", str(entrant), "--games", "4"],
            ["league", "leaderboard", str(self.output), "--json"],
        ]
        for args in commands:
            command = subprocess.run([sys.executable, "-m", "generals_arena", *args],
                                     capture_output=True, text=True)
            self.assertEqual(command.returncode, 0, command.stderr)
        self.assertEqual(json.loads(command.stdout)["completed_games"], 4)

        # The Python compatibility API remains available to RL callers.
        config = self.root / "input.json"
        config.write_text(json.dumps(small_config()))
        code = f'''
import sys, json
from generals_arena.tournament import run_tournament
run_tournament(json.loads({json.dumps(config.read_text())}), {str(self.root / "clean")!r})
assert not any(n.startswith(("torch", "generals_rl", "rsl_rl", "tkinter")) for n in sys.modules)
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
