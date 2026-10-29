"""Exporter ownership, checkpoint pinning and immutable publication artifacts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from generals_arena.policies.artifacts import validate_bundle
from generals_rl.export import export_policy, prepare_policy_source
from rl_policy_test_utils import make_submission
from rl_policy_test_utils import make_checkpoint


def test_new_checkpoint_exports_pinned_source_and_actor_only(tmp_path):
    torch.set_num_threads(1)
    checkpoint, config = make_checkpoint(tmp_path, family='scaled')
    config['model'] = {'codename': 'nova-ViT1M'}
    deployment = prepare_policy_source(config, tmp_path)
    saved = torch.load(checkpoint, weights_only=True)
    saved['infos'] = {**deployment, 'completed_ppo_updates': 3}
    saved['optimizer_state_dict'] = {'example': torch.ones(128)}
    torch.save(saved, checkpoint)
    torch.manual_seed(100)
    rng = torch.random.get_rng_state().clone()
    bundle = export_policy(checkpoint, tmp_path/'bundle')
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    info = validate_bundle(bundle)
    assert info['name'] == 'nova-ViT1M-3'
    actor = torch.load(info['paths']['weights'], weights_only=True)
    assert set(actor) == {'actor_state_dict'}
    expected = {key[6:]: value for key, value in saved['model_state_dict'].items() if key.startswith('actor.')}
    assert actor['actor_state_dict'].keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actor['actor_state_dict'][key], expected[key], rtol=0, atol=0)
    model = json.loads(Path(info['paths']['config']).read_text())
    assert model['model']['completed_ppo_updates'] == 3
    assert model['model']['iteration'] == 2
    assert info['manifest']['artifacts']['code']['sha256'] == deployment['source_id']
    assert export_policy(checkpoint, bundle) == bundle
    # Existing actor cannot be silently replaced even with equal metadata.
    saved['model_state_dict'][next(key for key in saved['model_state_dict'] if key.startswith('actor.'))] += 1
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='different actor weights'):
        export_policy(checkpoint, bundle)


def test_source_drift_and_changed_resume_features_are_rejected(tmp_path):
    torch.set_num_threads(1)
    checkpoint, config = make_checkpoint(tmp_path, family='scaled')
    deployment = prepare_policy_source(config, tmp_path)
    modified = deepcopy(config)
    modified['task']['combat_features'] = not config['task']['combat_features']
    with pytest.raises(ValueError, match='specification changed'):
        prepare_policy_source(modified, tmp_path/'resume', policy_spec=deployment['policy_spec'])
    saved = torch.load(checkpoint, weights_only=True)
    saved['infos'] = {**deployment, 'completed_ppo_updates': 1}
    torch.save(saved, checkpoint)
    runtime = Path(deployment['source_path'])/'runtime.py'
    runtime.write_text(runtime.read_text()+'\n# changed after checkpoint\n')
    with pytest.raises(ValueError, match='recorded source_id'):
        export_policy(checkpoint, tmp_path/'bundle')
    with pytest.raises(ValueError, match='source differs'):
        prepare_policy_source(config, tmp_path/'resume', source=deployment['source_path'])
    assert not (tmp_path/'bundle').exists()


def test_bundle_identity_binds_code_config_weights_and_not_location(tmp_path):
    import shutil
    bundle, _ = make_submission(tmp_path/'first')
    before = validate_bundle(bundle)
    copied = tmp_path/'elsewhere'
    shutil.copytree(bundle, copied)
    assert validate_bundle(copied)['policy_id'] == before['policy_id']
    (copied/'model_config.json').write_text('{}')
    with pytest.raises(ValueError, match='changed|hash|SHA'):
        validate_bundle(copied)


def test_checkpoint_without_export_metadata_requires_configuration(tmp_path):
    checkpoint, _ = make_checkpoint(tmp_path, family='scaled')
    (tmp_path/'run_config.json').unlink()
    with pytest.raises(ValueError, match='run_config'):
        export_policy(checkpoint, tmp_path/'bundle')


def test_deployment_sampling_is_owned_by_model_and_default_remains_stochastic():
    from generals_rl.config import default_training_config
    from generals_rl.export import make_policy_spec
    config = default_training_config()
    assert make_policy_spec(config)['inference']['mode'] == 'stochastic'
    config['inference']['mode'] = 'greedy'
    assert make_policy_spec(config)['inference']['mode'] == 'greedy'


def test_task_memory_does_not_implicitly_add_an_actor_knowledge_input(tmp_path):
    from generals_rl.config import task_config
    from generals_rl.information_ppo import InformationSpatialPPOPolicy
    from generals_rl.task import GeneralsTask
    from rl_test_utils import small_training_config
    from tensordict import TensorDict
    config = small_training_config(task_overrides={'persistent_knowledge': True})
    options = dict(config['runner']['policy'])
    options.pop('class_name')
    assert 'knowledge_channels' not in options
    with GeneralsTask(task_config(config['task'])) as task:
        inputs = TensorDict(task.get_observations(), [task.num_envs])
        policy = InformationSpatialPPOPolicy(inputs, config['runner']['obs_groups'], 649, **options)
    checkpoint = tmp_path/'checkpoint.pt'
    torch.save({'model_state_dict': policy.state_dict(), 'iter': 0}, checkpoint)
    bundle = export_policy(checkpoint, tmp_path/'bundle', run_config=config)
    saved = json.loads((bundle/'model_config.json').read_text())
    assert saved['network']['knowledge_channels'] == 0


def test_moving_complete_run_keeps_pinned_export_and_resume_source(tmp_path):
    from generals_rl.export import resolve_policy_source
    root = tmp_path/'original'
    root.mkdir()
    checkpoint, config = make_checkpoint(root, family='scaled')
    deployment = prepare_policy_source(config, root)
    assert deployment['policy_spec']['model']['codename'].endswith('-original')
    saved = torch.load(checkpoint, weights_only=True)
    saved['infos'] = {**deployment, 'completed_ppo_updates': 3}
    torch.save(saved, checkpoint)
    moved = tmp_path/'moved'
    root.rename(moved)
    source = resolve_policy_source(moved/'checkpoint.pt', saved['infos'])
    assert source == moved/'policy_source'
    bundle = export_policy(moved/'checkpoint.pt', tmp_path/'bundle')
    assert validate_bundle(bundle)['manifest']['artifacts']['code']['sha256'] == deployment['source_id']
    resumed = prepare_policy_source(config, tmp_path/'resume', source=source,
                                    policy_spec=deployment['policy_spec'])
    assert resumed['policy_spec']['model']['codename'] == deployment['policy_spec']['model']['codename']
    import shutil
    shutil.rmtree(moved/'policy_source')
    with pytest.raises(FileNotFoundError, match='pinned policy_source'):
        resolve_policy_source(moved/'checkpoint.pt', saved['infos'])
