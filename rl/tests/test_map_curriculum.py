"""Map endpoints, geometric progression, partial resets and real PPO resume."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from generals import MapConfig, make_env
from generals.core.mapgen import generate_map
from generals.core.recording import state_digest
from generals_rl.config import default_training_config
from generals_rl.map_curriculum import MapCurriculumConfig, curriculum_maps, easy_map, map_metrics
from generals_rl.map_sampling import MapSampler
from generals_rl.train import train


class MapCurriculumTests(unittest.TestCase):
    settings = MapCurriculumConfig(32, 64)

    def test_short_spawn_reference_keeps_easy_distance_on_random_full_size_maps(self):
        settings = replace(self.settings, easy_spawn_reference_size=9)
        dimensions = set()
        for seed in range(32):
            maps = list(curriculum_maps(MapConfig(players=2), seed, settings))
            reference = easy_map(MapConfig(9, 9, 2), seed, settings)
            self.assertEqual(map_metrics(maps[0])['manhattan'], map_metrics(reference)['manhattan'])
            target = generate_map(MapConfig(players=2), seed)
            self.assertEqual(state_digest(maps[-1]), state_digest(target))
            self.assertEqual(maps[0].owner.shape, target.owner.shape)
            dimensions.add(target.owner.shape)
            for state in maps:
                state.validate()
            self.assertEqual((maps[0].structure == 1).sum(), 4)
        self.assertGreater(len(dimensions), 1)
        self.assertEqual(state_digest(easy_map(MapConfig(9, 9, 2), 42, settings)),
                         '7b365db0d84f1084712acd160dfb8a529adc74c13cfd5816c05d54f2f95a5652')

    def test_successful_easy_maps_match_archived_golden_and_final_maps_match_official(self):
        config = MapConfig(9, 9, 2)
        golden = {7: '009ee237c0b6bf9277600be40443754907db669470a547707e81697505fdb345',
                  42: '7b365db0d84f1084712acd160dfb8a529adc74c13cfd5816c05d54f2f95a5652'}
        for seed, digest in golden.items():
            self.assertEqual(state_digest(easy_map(config, seed, self.settings)), digest)
        for width, height in ((9, 9), (13, 17), (23, 23)):
            config = MapConfig(width, height, 2)
            for seed in (7, 42):
                maps = list(curriculum_maps(config, seed, self.settings))
                self.assertEqual(len(maps), self.settings.levels)
                self.assertEqual(state_digest(maps[-1]), state_digest(generate_map(config, seed)))
                self.assertEqual((maps[0].structure == 1).sum(), 4)
                for state in maps:
                    state.validate()
                    self.assertEqual(state.afk_turn, 0)
                    armies = state.army[state.structure == 1]
                    self.assertTrue(((armies >= 40) & (armies <= 50)).all())
                    self.assertGreaterEqual(map_metrics(state)['open_path'], 0)

    def test_population_difficulty_increases_and_global_rng_is_independent(self):
        before = np.random.get_state()
        values = [[], [], [], [], []]
        for seed in range(64):
            for level, state in enumerate(curriculum_maps(MapConfig(9, 9, 2), seed, self.settings)):
                values[level].append(map_metrics(state))
        for key in ('manhattan', 'mountain_fraction', 'cities', 'open_path'):
            means = [np.mean([row[key] for row in rows]) for rows in values]
            self.assertTrue((np.diff(means) > 0).all(), (key, means))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_schedule_changes_only_reset_lanes_and_final_bank_is_exact(self):
        config, size, seed = MapConfig(9, 9, 2), 8, 19
        with make_env(config, batch_size=8, auto_reset=False) as env:
            sampler = MapSampler(env, config, size, seed=seed, curriculum=self.settings)
            sampler.reset()
            previous = {key: value.copy() for key, value in env.state.items()}
            sampler.set_transitions(64)
            for key in previous:
                np.testing.assert_array_equal(previous[key], env.state[key])
            mask = np.arange(8) % 2 == 0
            sampler.reset(mask)
            np.testing.assert_array_equal(sampler.level_ids, np.where(mask, 4, 0))
            for key in previous:
                np.testing.assert_array_equal(previous[key][~mask], env.state[key][~mask])
            seeds = np.random.SeedSequence([seed, 0x4D4150]).generate_state(size, dtype=np.uint64)
            for index, map_seed in enumerate(seeds):
                state = generate_map(config, int(map_seed))
                row = 4 * size + index
                np.testing.assert_array_equal(sampler.arrays['army'][row], state.army.ravel())
                np.testing.assert_array_equal(sampler.arrays['owner'][row], state.owner.ravel())
                np.testing.assert_array_equal(sampler.arrays['structure'][row], state.structure.ravel())

    def test_real_training_logs_course_and_resume_uses_saved_transition_count(self):
        config = default_training_config()
        config['task'].update(width=9, height=9, random_maps=False, num_envs=8, device='cpu',
                              backend='cpu', max_episode_steps=2, map_pool_size=8,
                              map_curriculum=dict(warmup_transitions=32, end_transitions=64))
        config['runner'].update(num_steps_per_env=4, save_interval=1)
        config['runner']['policy'].update(map_width=9, map_height=9, compile_forward=False, amp_bfloat16=False)
        config['runner']['algorithm'].update(num_learning_epochs=1, num_mini_batches=2,
                                             physical_minibatch_size=16, amp_bfloat16=False,
                                             ppo_probe_samples=16, ppo_probe_batch_size=16)
        config['self_play'].update(pool_size=2, snapshot_interval=1)
        config['evaluation'] = dict(enabled=False)
        config['training_control'] = {}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            first, second = Path(directory) / 'first', Path(directory) / 'resumed'
            train(config, first, iterations=3)
            saved = torch.load(first / 'checkpoint.pt', weights_only=True, map_location='cpu')
            self.assertEqual(saved['infos']['map_curriculum_transitions'], 96)
            events = EventAccumulator(str(first)).Reload()
            positions = events.Scalars('Curriculum/map_position')
            self.assertEqual([value.value for value in positions], [0., 0., 1.])
            train(config, second, iterations=1, resume=str(first / 'checkpoint.pt'))
            saved = torch.load(second / 'checkpoint.pt', weights_only=True, map_location='cpu')
            self.assertEqual(saved['infos']['map_curriculum_transitions'], 128)
            events = EventAccumulator(str(second)).Reload()
            self.assertEqual(events.Scalars('Curriculum/map_position')[0].value, 1.)
            self.assertEqual(events.Scalars('Curriculum/map_final_fraction')[0].value, 1.)


if __name__ == '__main__':
    unittest.main()
