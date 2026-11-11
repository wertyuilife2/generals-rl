"""Compact local-information recipes through PPO and exported inference."""
from contextlib import redirect_stdout
import io
import json

import pytest
import torch

from generals_rl.config import configure_observations, task_config
from generals_rl.export import make_policy_spec
from generals_rl.policy.actions import EncodedActionSpace
from generals_rl.policy.features import FEATURE_SETS, feature_recipe
from generals_rl.policy.runtime import PolicyRuntime
from generals_rl.rollout_codec import CompactObservationCodec, DynamicObservationCodec
from generals_rl.task import GeneralsTask
from generals_rl.train import train
from test_shared_critic import model_for
from test_value_distribution import recipe as value_recipe, assert_train_resume_export


def recipe(features='stacked'):
    cfg = value_recipe()
    cfg['task']['feature_set'] = features
    return configure_observations(cfg)


@pytest.mark.parametrize('features', ('compact', 'temporal', 'temporal_dedup'))
def test_real_train_resume_export_and_scalar_rollout(tmp_path, features):
    assert_train_resume_export(tmp_path, recipe(features))


@pytest.mark.parametrize('source', FEATURE_SETS)
@pytest.mark.parametrize('target', FEATURE_SETS)
def test_switching_resolved_feature_recipe_rebuilds_derived_settings(source, target):
    cfg = recipe(source)
    cfg['task']['feature_set'] = target
    resolved = configure_observations(cfg)
    expected = recipe(target)
    assert resolved == expected
    assert configure_observations(resolved) == resolved


@pytest.mark.parametrize('codec', (None, 'custom.module.Codec'))
def test_feature_recipe_preserves_explicit_storage_execution_choices(codec):
    cfg = recipe('temporal')
    cfg['task']['feature_set'] = 'stacked'
    cfg['runner'].update(storage_codec=codec, storage_history={})
    resolved = configure_observations(cfg)
    assert resolved['runner']['storage_codec'] == codec
    assert resolved['runner']['storage_history'] == {}
    assert resolved['task']['history_length'] == feature_recipe('stacked').history_length


@pytest.mark.parametrize('source', ('compact', 'temporal'))
def test_switched_stacked_recipe_can_train(tmp_path, source):
    cfg = recipe(source)
    cfg['task']['feature_set'] = 'stacked'
    with redirect_stdout(io.StringIO()):
        result = train(cfg, tmp_path/'run', iterations=1)
    assert result['iterations'] == 1


@pytest.mark.parametrize('features', ('compact', 'temporal', 'temporal_dedup'))
def test_compact_storage_matches_rollout_and_keeps_combat(features):
    torch.set_num_threads(1)
    cfg = recipe(features)
    with GeneralsTask(task_config(cfg['task'])) as task:
        model, obs = model_for(task, cfg)
        codec = (CompactObservationCodec if features == 'compact' else DynamicObservationCodec)(obs)
        for key in ('policy', 'critic', 'knowledge', 'action_mask', 'valid_cells'):
            encoded = codec.encode(key, obs[key]).unsqueeze(0)
            result = codec.gather(key, encoded, torch.zeros(task.num_envs, dtype=torch.long), torch.arange(task.num_envs))
            assert torch.equal(result, obs[key]), key
        maps = model.actor.encoder.input_maps(obs['policy'], obs['knowledge'], obs['valid_cells'])
        assert maps.shape[1] == (28 if features == 'compact' else 40)
        frames = model.layout.history_maps(obs['policy']).reshape(task.num_envs, -1, 13, 9, 9)
        torch.testing.assert_close(maps[:, 9:11], frames[:, -1, 10:12], rtol=0, atol=0)
        # Actor logits cannot depend on privileged maps.
        altered = obs.clone()
        altered['critic'] = torch.randn_like(altered['critic'])
        with torch.no_grad():
            torch.testing.assert_close(model.action_probabilities(obs), model.action_probabilities(altered), rtol=0, atol=0)


@pytest.mark.parametrize('features', ('compact', 'temporal', 'temporal_dedup'))
def test_runtime_memory_matches_training_through_steps_and_reset(tmp_path, features):
    torch.set_num_threads(1)
    cfg = recipe(features)
    with GeneralsTask(task_config(cfg['task'])) as task:
        model, obs = model_for(task, cfg)
        weights, config_path = tmp_path/'weights.pt', tmp_path/'policy.json'
        torch.save({'actor_state_dict': model.actor.state_dict()}, weights)
        config_path.write_text(json.dumps(make_policy_spec(cfg)))
        runtime = PolicyRuntime(weights=weights, config_path=config_path)
        def reset():
            runtime.reset_batch(task.player_ids.tolist(), [4]*task.num_envs)
        reset()
        for step in range(5):
            batch = task.batch()
            # Internal boundary deliberately matches the same legal observation.
            inputs = runtime._inputs(batch, active=torch.ones(task.num_envs, dtype=torch.bool))
            for key in ('policy', 'knowledge', 'action_mask'):
                torch.testing.assert_close(inputs[key], task.get_observations()[key], rtol=0, atol=0)
            result = task.step(torch.zeros(task.num_envs, dtype=torch.long))
            if result.truncated.any() or result.terminated.any():
                reset()


def test_half_duplicate_mask_only_removes_equivalent_army_two():
    cfg = recipe('temporal_dedup')
    with GeneralsTask(task_config(cfg['task'])) as task:
        batch = task.batch()
        # Override independent batch view; engine actions still use native encoding.
        for troops in (2, 3, 5):
            batch.state['army'].fill_(troops)
            plain = EncodedActionSpace(9,9).mask(batch, task.player_ids)
            pruned = task.action_space.mask(batch, task.player_ids)
            assert torch.equal(plain[:, 0], pruned[:, 0])
            assert torch.equal(plain[:, 1::2], pruned[:, 1::2])
            if troops == 2:
                assert not pruned[:, 2::2].any()
            else:
                assert torch.equal(plain, pruned)


def test_dynamic_memory_is_observation_only_and_clears_stale_enemy():
    from generals_rl.policy.state import GameBatch
    cfg=recipe('temporal')
    with GeneralsTask(task_config(cfg['task'])) as task:
        batch=task.batch(); memory=task.knowledge
        visible=task.observation_encoder.visibility(batch,task.player_ids)
        # Same public scoreboard; unseen army/type changes cannot enter local memory.
        changed=GameBatch({k:v.clone() for k,v in batch.state.items()},batch.stats.clone(),9,9,valid=batch.valid_cells)
        changed.state['army'][~visible]=99999
        changed.state['structure'][~visible]=2
        copy=memory.clone()
        copy.update(changed,task.player_ids,lane_ids=task._game_rows,visible=visible)
        memory.update(batch,task.player_ids,lane_ids=task._game_rows,visible=visible)
        torch.testing.assert_close(copy.features(batch.state['tick'],task.player_ids,task._game_rows),
            memory.features(batch.state['tick'],task.player_ids,task._game_rows),rtol=0,atol=0)
        memory.reset()
        assert not memory.army.any() and not memory.score_ema.any()
        assert (memory.score_tick==-1).all() and (memory.last_seen_tick==-1).all()
