"""Behavioral contracts for the algorithm-independent Generals MDP."""

import unittest

import numpy as np
import torch

from generals.backends.cpu.env import CPUEnv
from generals.core.engine import CoreEngine
from generals.core.observation import legal_action_mask, observe
from generals.core.state import (
    Action, Direction, EMPTY, GameState, MapLayout, MoveMode, Structure, Terrain,
)
from generals.vector import VecEnv
from generals_rl.mdp import GameBatch, LocalObservationEncoder, RewardConfig, RewardResult, ShapedReward
from generals_rl.policy.actions import EncodedActionSpace
from generals_rl.policy.observations import ObservationHistory
from generals_rl.task import GeneralsTask, TaskConfig


def example_state():
    """Two separated generals, an own zero-army move, and a losing attack."""
    terrain = np.zeros((5, 5), dtype=np.uint8)
    terrain[1, 2] = Terrain.MOUNTAIN
    structure = np.zeros((5, 5), dtype=np.uint8)
    owner = np.full((5, 5), EMPTY, dtype=np.int16)
    army = np.zeros((5, 5), dtype=np.int64)
    structure[2, 2] = structure[4, 4] = Structure.GENERAL
    owner[2, 2], owner[4, 4] = 0, 1
    army[2, 2], army[4, 4] = 3, 50
    owner[0, 0], army[0, 0] = 0, 1
    owner[2, 3], army[2, 3] = 1, 99
    state = GameState(MapLayout(terrain), structure, owner, army,
                      np.array([True, True]), np.array([12, 24], dtype=np.int32))
    state.validate()
    return state


def hidden_world_variant(state):
    """Change hidden world details without changing public army/territory."""
    result = state.copy()
    result.structure[4, 4], result.owner[4, 4], result.army[4, 4] = Structure.NONE, EMPTY, 0
    result.structure[4, 0], result.owner[4, 0], result.army[4, 0] = Structure.GENERAL, 1, 50
    result.general_pos[1] = 20
    terrain = result.layout.terrain.copy()
    terrain[0, 4] = Terrain.MOUNTAIN
    result.layout = MapLayout(terrain)
    result.validate()
    return result


def capture_state(players=2):
    terrain = np.zeros((3, 3), dtype=np.uint8)
    structure = np.zeros_like(terrain)
    owner = np.full((3, 3), EMPTY, dtype=np.int16)
    army = np.zeros((3, 3), dtype=np.int64)
    positions = [3, 4, 8][:players]
    for pid, position in enumerate(positions):
        structure.flat[position] = Structure.GENERAL
        owner.flat[position] = pid
        army.flat[position] = [20, 1, 8][pid]
    state = GameState(MapLayout(terrain), structure, owner, army,
                      np.ones(players, dtype=bool), np.asarray(positions, dtype=np.int32))
    state.validate()
    return state


class CaptureOpponent:
    """Player zero attacks player one; the learner's own slot is overridden."""
    def act(self, batch, player_ids, action_space, encoder, generator):
        capture = Action.move(3, Direction.RIGHT).encode()
        return torch.where(player_ids == 0, capture, 0)


class MDPComponentTests(unittest.TestCase):
    def backend(self, states):
        env = CPUEnv.from_states(states, threads=1)
        self.addCleanup(env.close)
        return env

    def test_encoded_action_mask_matches_independent_core(self):
        state = example_state()
        env = self.backend([state, state.copy()])
        batch = GameBatch.from_backend(env, "cpu")
        action_space = EncodedActionSpace(5, 5)
        pids = torch.tensor([0, 1])
        actual = action_space.mask(batch, pids)
        self.assertEqual(actual.dtype, torch.bool)
        self.assertEqual(tuple(actual.shape), (2, 1 + 25 * 8))
        for lane, pid in enumerate(pids.tolist()):
            core_mask, wait = legal_action_mask(observe(state, pid))
            expected = np.concatenate(([wait], core_mask.reshape(-1)))
            np.testing.assert_array_equal(actual[lane].cpu().numpy(), expected)
        self.assertFalse(actual[0, Action.move(12, Direction.UP).encode()])
        self.assertTrue(actual[0, Action.move(12, Direction.RIGHT).encode()])
        self.assertTrue(actual[0, Action.move(0, Direction.RIGHT).encode()])

    def test_productive_mask_matches_independent_core_after_growth(self):
        actions = EncodedActionSpace(5, 5, productive_actions=True)
        states = []
        for tick in (0, 1, 48, 49, 50):
            for army in (0, 1, 2, 3):
                state = example_state()
                state.tick = tick
                state.owner[0, 1], state.structure[0, 1] = 0, Structure.CITY
                for source in (0, 1, 12):  # ordinary land, city, general
                    state.army.flat[source] = army
                states.append(state)
        masks = actions.mask(GameBatch.from_backend(self.backend(states), "cpu"),
                             torch.zeros(len(states), dtype=torch.long))
        for lane, state in enumerate(states):
            legal, _ = legal_action_mask(observe(state, 0))
            self.assertTrue(masks[lane, 0])
            for source, direction, mode in np.argwhere(legal):
                action = Action.move(int(source), Direction(int(direction)), MoveMode(int(mode)))
                engine = CoreEngine(state.copy())
                move = engine.step([action, Action.wait()]).moves[0]
                with self.subTest(tick=state.tick, source=source, mode=mode,
                                  army=int(state.army.flat[source])):
                    self.assertEqual(bool(masks[lane, action.encode()]), move.valid and move.moved > 0)
            self.assertFalse(bool((masks[lane, 1:] & ~torch.from_numpy(legal.reshape(-1))).any()))

    def test_local_observation_hides_terrain_and_opponent_positions(self):
        first = example_state()
        second = hidden_world_variant(first)
        env = self.backend([first, second])
        batch = GameBatch.from_backend(env, "cpu")
        encoder = LocalObservationEncoder(episode_ticks=64)
        pids = torch.zeros(2, dtype=torch.long)
        local = encoder.encode(batch, pids)
        full = encoder.encode(batch, pids, privileged=True)
        self.assertTrue(torch.isfinite(local).all())
        torch.testing.assert_close(local[0], local[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(full[0], full[1]))

    def test_hidden_opponent_city_totals_do_not_leak(self):
        first = example_state()
        first.owner[4, 3], first.army[4, 3] = 1, 8
        second = first.copy()
        second.structure[4, 3] = Structure.CITY
        first.validate()
        second.validate()
        env = self.backend([first, second])
        batch = GameBatch.from_backend(env, "cpu")
        encoder = LocalObservationEncoder(episode_ticks=64)
        pids = torch.zeros(2, dtype=torch.long)
        torch.testing.assert_close(encoder.encode(batch, pids)[0],
                                   encoder.encode(batch, pids)[1], rtol=0, atol=0)
        full = encoder.encode(batch, pids, privileged=True)
        self.assertFalse(torch.equal(full[0], full[1]))

    def test_eliminated_player_has_absorbing_wait_and_no_moves(self):
        state = example_state()
        state.structure[4, 4] = Structure.CITY
        state.owner[state.owner == 1] = 0
        state.alive[1] = False
        state.terminated, state.winner_id = True, 0
        state.validate()
        batch = GameBatch.from_backend(self.backend([state]), "cpu")
        mask = EncodedActionSpace(5, 5).mask(batch, torch.tensor([1]))
        self.assertTrue(mask[0, 0])
        self.assertEqual(mask.sum().item(), 1)

    def test_history_reset_clears_only_finished_lanes(self):
        history = ObservationHistory(length=3)
        first = torch.tensor([[1., 2.], [10., 20.]])
        history.reset(first)
        history.push(first + 1)
        history.push(first + 2)
        continuing = history.frames[1].clone()
        reset_frame = torch.tensor([[100., 200.], [11., 21.]])
        history.reset(reset_frame, mask=torch.tensor([True, False]))
        torch.testing.assert_close(history.frames[0], reset_frame[0].expand(3, 2))
        torch.testing.assert_close(history.frames[1], continuing)
        exported = history.flattened()
        history.push(first + 3)
        # Stored policy input cannot alias the next transition's history.
        torch.testing.assert_close(exported[0], reset_frame[0].repeat(3))

    def test_reward_distinguishes_win_loss_and_time_limit(self):
        config = RewardConfig(territory=.1, cities=.5, army=.01,
                              win=3., loss=-4., tick=-.02)
        reward = ShapedReward(config)
        before = torch.tensor([[10, 2, 0], [10, 2, 0], [10, 2, 0]])
        after = torch.tensor([[20, 3, 1], [0, 0, 0], [10, 2, 0]])
        result = reward.evaluate(before, after,
                                 terminated=torch.tensor([True, True, False]),
                                 truncated=torch.tensor([False, False, True]),
                                 winner=torch.tensor([0, 1, -1]),
                                 learner_alive=torch.tensor([True, False, True]),
                                 player_ids=torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(result.terms["terminal"], torch.tensor([3., -4., 0.]))
        torch.testing.assert_close(result.total, torch.tensor([3.68, -4.32, -.02]))
        torch.testing.assert_close(result.total, sum(result.terms.values()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_cpu_gpu_backends_share_mdp_observation_and_action_contract(self):
        from generals.backends.gpu.env import GPUEnv

        first, second = example_state(), hidden_world_variant(example_state())
        first.army.flat[12] = 1
        third, fourth = first.copy(), second.copy()
        third.tick = fourth.tick = 49
        states = [first, second, third, fourth]
        cpu = self.backend(states)
        gpu = GPUEnv.from_states(states, device="cuda")
        self.addCleanup(gpu.close)
        pids = torch.tensor([0, 1, 0, 1])
        cpu_batch = GameBatch.from_backend(cpu, "cpu")
        gpu_batch = GameBatch.from_backend(gpu, "cuda")
        encoder = LocalObservationEncoder(episode_ticks=64)
        for privileged in (False, True):
            torch.testing.assert_close(encoder.encode(cpu_batch, pids, privileged),
                                       encoder.encode(gpu_batch, pids.cuda(), privileged).cpu())
        for productive in (False, True):
            actions = EncodedActionSpace(5, 5, productive_actions=productive)
            torch.testing.assert_close(actions.mask(cpu_batch, pids),
                                       actions.mask(gpu_batch, pids.cuda()).cpu())


class TaskTests(unittest.TestCase):
    def task(self, states, **settings):
        defaults = dict(width=states[0].layout.width, height=states[0].layout.height,
                        players=states[0].player_count, num_envs=len(states),
                        threads=1, max_episode_steps=8, history_length=3,
                        rotate_seats=False, opponent="wait")
        defaults.update(settings)
        config = TaskConfig(**defaults)
        backend = VecEnv.from_states(states, episode_ticks=config.max_episode_steps,
                                     auto_reset=False, threads=1)
        task = GeneralsTask(config, backend=backend)
        self.addCleanup(task.close)
        return task

    def test_natural_win_takes_precedence_over_time_limit_and_resets(self):
        task = self.task([capture_state()], max_episode_steps=1, rotate_seats=True)
        result = task.step(torch.tensor([[Action.move(3, Direction.RIGHT).encode()]]))
        self.assertTrue(result.terminated.item())
        self.assertFalse(result.truncated.item())
        self.assertEqual(result.metrics["Reward/terminal"].item(), 1.)
        self.assertEqual(result.metrics["Episode/win"].item(), 1.)
        self.assertEqual(task.player_ids.item(), 1)
        self.assertEqual(task.batch().state["tick"].item(), 0)
        self.assertEqual(task.episode_length_buf.item(), 0)
        self.assertEqual(task.completed_episodes, 1)
        self.assertEqual(task.terminated_episodes, 1)
        self.assertFalse(torch.equal(result.terminal_observations["critic"],
                                     result.observations["critic"]))
        final_mask = result.terminal_observations["action_mask"]
        self.assertTrue(final_mask[0, 0])
        self.assertEqual(final_mask.sum().item(), 1)
        frames = result.observations["policy"].reshape(1, 3, -1)
        torch.testing.assert_close(frames[0], result.observations["single_obs"][0].expand(3, -1))

    def test_multiplayer_learner_elimination_is_terminal_even_when_game_continues(self):
        task = self.task([capture_state(players=3)], learner_seat=1, max_episode_steps=1)
        task.opponent = CaptureOpponent()
        result = task.step(torch.zeros((1, 1), dtype=torch.long))
        self.assertTrue(result.terminated.item())
        self.assertFalse(result.truncated.item())
        self.assertEqual(result.metrics["Reward/terminal"].item(), -1.)
        self.assertEqual(result.metrics["Episode/win"].item(), 0.)
        self.assertEqual(task.terminated_episodes, 1)
        self.assertEqual(task.truncated_episodes, 0)
        # Seat zero captured the learner before its submitted WAIT executed.
        self.assertEqual(result.metrics["Action/wait_rate"].item(), 0.)
        self.assertEqual(result.metrics["Action/moved_army"].item(), 0.)
        # No defeated-player moves survive in the saved transition.
        self.assertEqual(result.terminal_observations["action_mask"].sum().item(), 1)
        self.assertTrue(task.batch().state["alive"].all())
        self.assertFalse(task.batch().state["terminated"].item())

    def test_time_limit_preserves_terminal_observation_and_resets_history(self):
        task = self.task([capture_state()], max_episode_steps=2)
        task.step(torch.zeros((1, 1), dtype=torch.long))
        result = task.step(torch.zeros((1, 1), dtype=torch.long))
        self.assertFalse(result.terminated.item())
        self.assertTrue(result.truncated.item())
        self.assertEqual(result.metrics["Reward/terminal"].item(), 0.)
        self.assertEqual(result.metrics["Episode/truncated"].item(), 1.)
        self.assertEqual(result.metrics["Episode/length"].item(), 2.)
        terminal_frames = result.terminal_observations["policy"].reshape(1, 3, -1)
        torch.testing.assert_close(terminal_frames[:, -1], result.terminal_observations["single_obs"])
        reset_frames = result.observations["policy"].reshape(1, 3, -1)
        torch.testing.assert_close(reset_frames[0], result.observations["single_obs"][0].expand(3, -1))
        self.assertFalse(torch.equal(terminal_frames, reset_frames))
        # A retained terminal history remains isolated when the next episode steps.
        preserved = result.terminal_observations["policy"].clone()
        task.step(torch.zeros((1, 1), dtype=torch.long))
        torch.testing.assert_close(result.terminal_observations["policy"], preserved)

    def test_autoreset_preserves_unfinished_lanes_history_and_clock(self):
        state = capture_state()
        task = self.task([state, state.copy()])
        initial = task.get_observations()["single_obs"].clone()
        result = task.step(torch.tensor([[Action.move(3, Direction.RIGHT).encode()], [0]]))
        self.assertEqual(result.terminated.tolist(), [True, False])
        self.assertEqual(task.batch().state["tick"].tolist(), [0, 1])
        self.assertEqual(task.episode_length_buf.tolist(), [0, 1])
        frames = result.observations["policy"].reshape(2, 3, -1)
        torch.testing.assert_close(frames[0], result.observations["single_obs"][0].expand(3, -1))
        torch.testing.assert_close(frames[1, :2], initial[1].expand(2, -1))
        torch.testing.assert_close(frames[1, -1], result.observations["single_obs"][1])

    def test_custom_reward_component_controls_task_reward_and_metrics(self):
        class CustomReward:
            def evaluate(self, before, after, terminated, truncated, winner, alive, pids):
                value = torch.full((len(pids),), .125, device=pids.device)
                return RewardResult(value, {"custom": value})

        task = self.task([capture_state()])
        task.reward_model = CustomReward()
        result = task.step(torch.zeros((1, 1), dtype=torch.long))
        torch.testing.assert_close(result.reward, torch.tensor([.125]))
        torch.testing.assert_close(result.metrics["Reward/custom"], result.reward)

    def test_native_action_metrics_and_partial_reset(self):
        state = example_state()
        state.owner[0, 1], state.army[0, 1] = 0, 10
        state.army[0, 0] = 10
        task = self.task([state, state.copy()])
        forward = Action.move(0, Direction.RIGHT).encode()
        backward = Action.move(1, Direction.LEFT).encode()
        result = task.step(torch.tensor([[forward], [forward]]))
        torch.testing.assert_close(result.metrics["Action/moved_army"], torch.tensor([9., 9.]))
        torch.testing.assert_close(result.metrics["Action/backtrack_rate"], torch.zeros(2))
        task.reset(torch.tensor([True, False]))
        result = task.step(torch.tensor([[backward], [backward]]))
        torch.testing.assert_close(result.metrics["Action/backtrack_rate"], torch.tensor([0., 1.]))
        result = task.step(torch.zeros((2, 1), dtype=torch.long))
        torch.testing.assert_close(result.metrics["Action/wait_rate"], torch.ones(2))
        result = task.step(torch.tensor([[forward], [forward]]))
        torch.testing.assert_close(result.metrics["Action/backtrack_rate"], torch.zeros(2))

        # A native legal zero move is distinct from WAIT; metrics precede reset.
        task = self.task([example_state()], max_episode_steps=1)
        result = task.step(torch.tensor([[forward]]))
        self.assertTrue(result.truncated.item())
        self.assertEqual(result.metrics["Action/zero_moved_rate"].item(), 1.)
        self.assertEqual(result.metrics["Action/wait_rate"].item(), 0.)
        self.assertEqual(result.metrics["Action/moved_army"].item(), 0.)



if __name__ == "__main__":
    unittest.main()
