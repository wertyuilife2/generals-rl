"""Information, reset and search-branch contracts for persistent facts."""
import unittest

import torch

from generals_rl.mdp.knowledge import KnowledgeState, knowledge_channels
from generals_rl.mdp.observations import LocalObservationEncoder
from generals_rl.mdp.state import GameBatch


torch.set_num_threads(1)


def make_batch(*, num_envs=2, width=7, height=5, players=2, device="cpu"):
    cells = width * height
    owner = torch.full((num_envs, cells), -1, dtype=torch.int16, device=device)
    structure = torch.zeros((num_envs, cells), dtype=torch.uint8, device=device)
    army = torch.zeros((num_envs, cells), dtype=torch.int64, device=device)
    # Minimal tensor protocol deliberately omits general_pos: memory has no
    # access to the simulator's list of unseen general locations.
    for pid, position in enumerate(torch.linspace(0, cells - 1, players).long().tolist()):
        owner[:, position], structure[:, position], army[:, position] = pid, 2, 2
    state = {"owner": owner, "structure": structure, "army": army,
             "terrain": torch.zeros((num_envs, cells), dtype=torch.uint8, device=device),
             "alive": torch.ones((num_envs, players), dtype=torch.bool, device=device),
             "tick": torch.zeros(num_envs, dtype=torch.int64, device=device)}
    return GameBatch(state, torch.zeros((num_envs, players, 3), device=device), width, height)


def memory_for(batch, **kwargs):
    return KnowledgeState(batch.num_envs, batch.width, batch.height,
                          batch.players, batch.device, **kwargs)


def plane(memory, encoded, name):
    return encoded[:, memory.channels.index(name)].flatten(1)


class KnowledgeTests(unittest.TestCase):
    def test_unknown_schema_and_local_visibility_match_any_geometry(self):
        batch = make_batch(width=8, height=4, players=3)
        memory = memory_for(batch)
        ids = torch.tensor([0, 2])
        cold = memory.encode(batch.state["tick"], ids)
        self.assertEqual(tuple(cold.shape), (2, 12, 4, 8))
        self.assertEqual(memory.channels, knowledge_channels(3))
        self.assertEqual(int(cold.count_nonzero()), 0)
        memory.update(batch, ids)
        encoded = memory.encode(batch.state["tick"], ids)
        visible = LocalObservationEncoder(1024).visibility(batch, ids).float()
        torch.testing.assert_close(plane(memory, encoded, "explored"), visible)
        torch.testing.assert_close(plane(memory, encoded, "last_seen_owner_0"),
                                   (batch.state["owner"] == ids[:, None]).float())
        self.assertTrue(bool((plane(memory, encoded, "not_enemy_general") <= visible).all()))

    def test_fog_mutations_do_not_leak_and_known_general_survives_hiding(self):
        batch = make_batch()
        ids = torch.zeros(batch.num_envs, dtype=torch.long)
        # An owned outpost exposes the enemy general at [4, 6].
        batch.state["owner"][:, 27] = 0
        memory = memory_for(batch, age_scale=100)
        memory.update(batch, ids)
        first = memory.encode(batch.state["tick"], ids)
        self.assertTrue(bool(plane(memory, first, "enemy_general_landmark")[:, 34].all()))
        # Losing the outpost hides both its new owner and the enemy general.
        batch.state["owner"][:, 27] = 1
        batch.state["tick"][:] = 10
        memory.update(batch, ids)
        expected = memory.encode(batch.state["tick"], ids)
        changed = GameBatch({k: v.clone() for k, v in batch.state.items()},
                            batch.stats.clone(), batch.width, batch.height)
        changed.state["owner"][:, 34] = -2
        changed.state["structure"][:, 34] = 1
        changed.state["army"][:, 34] = 987654321
        changed.state["terrain"][:, 28] = 1
        changed.state["structure"][:, 30] = 2
        changed.state["owner"][:, 30] = 1
        branched = memory.clone()
        branched.update(changed, ids)
        torch.testing.assert_close(branched.encode(changed.state["tick"], ids), expected,
                                   rtol=0, atol=0)
        self.assertEqual(float(plane(memory, expected, "enemy_general_landmark")[0, 34]), 1)
        self.assertEqual(float(plane(memory, expected, "not_enemy_general")[0, 34]), 0)
        self.assertEqual(float(plane(memory, expected, "last_seen_owner_0")[0, 27]), 1)
        self.assertAlmostEqual(float(plane(memory, expected, "last_seen_age")[0, 27]), .1)
        self.assertEqual(float(plane(memory, expected, "last_seen_age")[0, 0]), 0)

    def test_owner_and_city_refresh_while_original_general_is_retained(self):
        batch = make_batch(num_envs=1, players=3)
        ids = torch.tensor([0])
        memory = memory_for(batch, age_scale=100)
        enemy_general = 17
        batch.state["owner"][:, 16] = 0  # Adjacent owned outpost.
        memory.update(batch, ids)
        before = memory.encode(batch.state["tick"], ids)
        self.assertEqual(float(plane(memory, before, "enemy_general_landmark")[0, enemy_general]), 1)
        # Another enemy captured this general. It is now a city, with a new
        # last-observed owner, while its original landmark remains known.
        batch.state["owner"][:, enemy_general] = 2
        batch.state["structure"][:, enemy_general] = 1
        batch.state["tick"][:] = 7
        memory.update(batch, ids)
        encoded = memory.encode(batch.state["tick"], ids)
        for name in ("last_seen_city", "general_landmark", "enemy_general_landmark",
                     "last_seen_owner_2"):
            self.assertEqual(float(plane(memory, encoded, name)[0, enemy_general]), 1)
        self.assertEqual(float(plane(memory, encoded, "last_seen_owner_1")[0, enemy_general]), 0)
        self.assertEqual(float(plane(memory, encoded, "not_enemy_general")[0, enemy_general]), 0)
        self.assertEqual(int(memory.general_owner[0, 0, enemy_general]), 1)
        self.assertEqual(float(plane(memory, encoded, "last_seen_age")[0, enemy_general]), 0)

    def test_relative_player_identity_and_independent_seats(self):
        batch = make_batch(num_envs=2, players=3)
        memory = memory_for(batch)
        # Seats 1 see an enemy seat 2 and a neutral city next to their general.
        batch.state["owner"][:, 18] = 2
        batch.state["owner"][:, 16] = -2
        batch.state["structure"][:, 16] = 1
        seat1 = torch.ones(2, dtype=torch.long)
        memory.update(batch, seat1)
        encoded = memory.encode(batch.state["tick"], seat1)
        self.assertEqual(float(plane(memory, encoded, "last_seen_owner_1")[0, 18]), 1)
        self.assertEqual(float(plane(memory, encoded, "last_seen_neutral")[0, 16]), 1)
        self.assertEqual(float(plane(memory, encoded, "last_seen_city")[0, 16]), 1)
        seat0 = torch.zeros(2, dtype=torch.long)
        self.assertEqual(int(memory.encode(batch.state["tick"], seat0).count_nonzero()), 0)
        memory.update(batch, seat0)
        other = memory.encode(batch.state["tick"], seat0)
        self.assertEqual(float(plane(memory, other, "explored")[0, 18]), 0)
        torch.testing.assert_close(memory.encode(batch.state["tick"], seat1), encoded,
                                   rtol=0, atol=0)

    def test_partial_reset_all_seats_clone_and_capped_age(self):
        batch = make_batch()
        memory = memory_for(batch, age_scale=10)
        ids = torch.tensor([0, 1])
        memory.update(batch, ids)
        memory.update(batch, 1 - ids)
        detached = memory.encode(batch.state["tick"], ids)
        branch = memory.clone()
        branch.reset(torch.tensor([True, False]))
        for seats in (ids, 1 - ids):
            original = memory.encode(batch.state["tick"], seats)
            cleared = branch.encode(batch.state["tick"], seats)
            self.assertEqual(int(cleared[0].count_nonzero()), 0)
            torch.testing.assert_close(cleared[1], original[1], rtol=0, atol=0)
        self.assertTrue(bool(detached[0].count_nonzero()))
        aged = memory.encode(torch.tensor([100, 100]), ids)
        torch.testing.assert_close(plane(memory, aged, "last_seen_age"),
                                   plane(memory, aged, "explored"))
        memory.reset()
        self.assertEqual(int(memory.encode(batch.state["tick"], ids).count_nonzero()), 0)
        # Encoded rollout frames own their storage across subsequent resets.
        self.assertTrue(bool(detached.count_nonzero()))

    def test_mountain_known_plain_and_non_general_certainty(self):
        batch = make_batch(num_envs=1)
        batch.state["terrain"][:, 1] = 1
        memory = memory_for(batch)
        ids = torch.tensor([0])
        memory.update(batch, ids)
        encoded = memory.encode(batch.state["tick"], ids)
        self.assertEqual(float(plane(memory, encoded, "mountain")[0, 1]), 1)
        for cell in (0, 1, 7, 8):
            self.assertEqual(float(plane(memory, encoded, "not_enemy_general")[0, cell]), 1)
        self.assertEqual(float(plane(memory, encoded, "not_enemy_general")[0, 34]), 0)
        known_plain = plane(memory, encoded, "explored") - plane(memory, encoded, "mountain")
        self.assertEqual(float(known_plain[0, 8]), 1)
        self.assertEqual(float(known_plain[0, 1]), 0)
        self.assertEqual(float(known_plain[0, 34]), 0)

    def test_update_mask_preserves_inactive_memory_despite_world_changes(self):
        batch = make_batch()
        memory = memory_for(batch)
        ids = torch.zeros(2, dtype=torch.long)
        memory.update(batch, ids)
        before = {name: getattr(memory, name).clone() for name in memory._fields}
        batch.state["owner"][:, 26] = 0
        batch.state["tick"][:] = 7
        memory.update(batch, ids, mask=torch.tensor([True, False]))
        for name in memory._fields:
            torch.testing.assert_close(getattr(memory, name)[1], before[name][1], rtol=0, atol=0)
        encoded = memory.encode(batch.state["tick"], ids)
        self.assertEqual(float(plane(memory, encoded, "explored")[0, 26]), 1)
        self.assertEqual(float(plane(memory, encoded, "explored")[1, 26]), 0)
        with self.assertRaises(ValueError):
            memory.update(batch, ids, mask=torch.tensor([1, 0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_cpu(self):
        cpu = make_batch(players=3)
        cuda = GameBatch({k: v.cuda() for k, v in cpu.state.items()}, cpu.stats.cuda(),
                         cpu.width, cpu.height)
        cpu_memory, cuda_memory = memory_for(cpu), memory_for(cuda)
        ids = torch.tensor([0, 2])
        for batch, memory, seats in ((cpu, cpu_memory, ids), (cuda, cuda_memory, ids.cuda())):
            memory.update(batch, seats)
        torch.testing.assert_close(cuda_memory.encode(cuda.state["tick"], ids.cuda()).cpu(),
                                   cpu_memory.encode(cpu.state["tick"], ids), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
