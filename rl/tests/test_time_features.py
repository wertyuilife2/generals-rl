"""Clock/age scales are portable model semantics, independent of game cutoff."""
from dataclasses import replace

import pytest
import torch

from generals import VecEnv
from generals_rl.config import default_training_config
from generals_rl.export import make_policy_spec
from generals_rl.mdp.state import GameBatch
from generals_rl.policy.config import FeatureConfig
from generals_rl.policy.observations import LocalObservationEncoder
from generals_rl.task import GeneralsTask, TaskConfig
from test_mdp import example_state


def test_fixed_clock_preserves_v32_within_horizon_and_stays_bounded_afterwards():
    with VecEnv.from_states([example_state()]) as env:
        batch = GameBatch.from_backend(env, 'cpu')
        seat = torch.zeros(1, dtype=torch.long)
        old = LocalObservationEncoder(2048)
        fixed = LocalObservationEncoder(4096, clock_scale_ticks=2048)
        for tick in (0, 1, 49, 50, 1024, 2048):
            batch.state['tick'].fill_(tick)
            torch.testing.assert_close(old.encode(batch, seat), fixed.encode(batch, seat), rtol=0, atol=0)
        batch.state['tick'].fill_(10001)
        clock = fixed.encode(batch, seat)[0, -5:-2]
        torch.testing.assert_close(clock, torch.tensor([1., .5, .02]))
        assert old.encode(batch, seat)[0, -5] > 1  # Explicit legacy behavior.


def test_task_and_export_preserve_custom_time_scales():
    config = default_training_config()
    config['task'].update(num_envs=1, map_pool_size=0, map_curriculum=None, backend='cpu', device='cpu',
        max_episode_steps=4096, clock_scale_ticks=2048, knowledge_age_scale=512)
    spec = make_policy_spec(config)
    features = FeatureConfig(**spec['features'])
    assert (features.max_episode_steps, features.clock_scale_ticks, features.knowledge_age_scale) == (4096, 2048, 512)
    from generals_rl.config import task_config
    with GeneralsTask(task_config(config['task'])) as task:
        assert task.knowledge.age_scale == features.knowledge_age_scale
    del config['task']['clock_scale_ticks']; del config['task']['knowledge_age_scale']
    legacy = FeatureConfig(**make_policy_spec(config)['features'])
    assert legacy.clock_scale_ticks is None and legacy.knowledge_age_scale == 2000


def test_exported_actor_observes_the_same_fixed_clock_and_age(tmp_path):
    from generals.core.observation import observe
    from generals_rl.config import task_config
    from generals_rl.export import export_policy
    from generals_rl.policy.runtime import create_controller
    from rl_policy_test_utils import make_checkpoint
    from test_policy_runtime import battle_state, capture_inputs
    checkpoint, config = make_checkpoint(tmp_path, family='scaled')
    config['task'].update(num_envs=1, learner_seat=0, rotate_seats=False,
        max_episode_steps=4096, clock_scale_ticks=2048, knowledge_age_scale=512)
    bundle = export_policy(checkpoint, tmp_path/'bundle', run_config=config, completed_updates=1)
    controller = create_controller(weights=bundle/'actor.pt', config_path=bundle/'model_config.json')
    captured = capture_inputs(controller)
    state = battle_state()
    state.tick = 5000
    backend = VecEnv.from_states([state], auto_reset=False, episode_ticks=100000)
    with GeneralsTask(task_config(config['task']), backend=backend) as task:
        for _ in range(3):
            controller.act(observe(task.backend.to_states()[0], 0))
            for key, value in captured[-1].items():
                torch.testing.assert_close(value, task.get_observations()[key], rtol=0, atol=0)
            assert captured[-1]['single_obs'][0, -5] == 1
            task.step(torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_invalid_scales_rejected(value):
    for key in ('clock_scale_ticks', 'knowledge_age_scale'):
        with pytest.raises(ValueError):
            replace(TaskConfig(), **{key: value})
