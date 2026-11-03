"""Official random geometry: training/deployment feature and action equivalence."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.utils import resolve_callable
from generals.core.observation import observe
from generals.core.state import MapLayout
from generals_rl.task import TaskConfig,GeneralsTask
from generals_rl.export import export_policy
from generals_rl.policy.runtime import PolicyRuntime


def unpad(state, valid):
    old_w=state.layout.width;valid=valid.reshape(state.owner.shape)
    h,w=int(valid.any(1).sum()),int(valid.any(0).sum())
    state.layout=MapLayout(state.layout.terrain[:h,:w])
    for name in ('structure','owner','army'):setattr(state,name,getattr(state,name)[:h,:w].copy())
    for name in ('general_pos','spawn_pos'):
        pos=getattr(state,name);setattr(state,name,np.where(pos>=0,pos//old_w*w+pos%old_w,-1).astype(np.int32))
    state.validate();return state


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_exported_actor_matches_random_shape_training(tmp_path,backend):
    if backend=='gpu' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    device='cuda:0' if backend=='gpu' else 'cpu'
    config=json.loads((Path(__file__).parents[1]/'src/generals_rl/default.json').read_text())
    config['task'].update(num_envs=3,backend=backend,device=device,map_pool_size=16,opponent='wait',seed=84)
    config['runner']['policy']['amp_bfloat16']=False
    tc={k:v for k,v in config['task'].items() if k!='reward'}
    with GeneralsTask(TaskConfig(**tc)) as task:
        options=deepcopy(config['runner']['policy']);cls=resolve_callable(options.pop('class_name'))
        policy=cls(TensorDict(task.get_observations(),[3]),config['runner']['obs_groups'],task.action_space.size,**options).to(device).eval()
        checkpoint=tmp_path/'model.pt';torch.save({'model_state_dict':policy.state_dict()},checkpoint)
        bundle=export_policy(checkpoint,tmp_path/'bundle',run_config=config,completed_updates=1,inference='greedy')
        runtime=PolicyRuntime(weights=bundle/'actor.pt',config_path=bundle/'model_config.json',device=device)
        runtime.reset_batch(task.player_ids.tolist(),[5,6,7])
        captured=[];original=runtime._inputs
        def inputs(*args,**kwargs):
            value=original(*args,**kwargs);captured.append({k:v.clone() for k,v in value.items()});return value
        runtime._inputs=inputs
        shapes=set()
        for tick in range(7):
            valid=task.map_sampler.valid
            if isinstance(valid,torch.Tensor):valid=valid.cpu().numpy()
            states=[unpad(s,valid[lane]) for lane,s in enumerate(task.backend.to_states())]
            shapes.update(s.owner.shape for s in states)
            observations=[observe(s,int(task.player_ids[lane])) for lane,s in enumerate(states)]
            selected=runtime.act_batch(observations)
            expected=task.get_observations()
            for name in ('policy','single_obs','knowledge','action_mask'):
                torch.testing.assert_close(captured[-1][name],expected[name],rtol=0,atol=0,msg=f'{backend} {tick} {name}')
            with torch.no_grad():encoded=policy.act_inference(TensorDict(expected,[3])).flatten()
            widths=torch.tensor([s.layout.width for s in states],device=device)
            assert [a.encode() for a in selected]==runtime._board_actions(encoded,widths).tolist()
            task.step(encoded)
            if tick==3:
                task.reset();runtime.reset_batch(task.player_ids.tolist(),[5,6,7])
        assert len(shapes)>1


def test_defaults_pin_2000_ticks_and_uncalibrated_official_profile():
    from generals_rl.training_evaluation import TrainingEvalConfig
    assert TaskConfig().max_episode_steps==TrainingEvalConfig().max_ticks==2000
    assert TaskConfig().random_maps and TrainingEvalConfig().profile is None


def test_light_runner_helper_accepts_official_task_inputs():
    from generals_rl.cts import runner_config
    config = TaskConfig(num_envs=2, map_pool_size=4, opponent='wait',
                        persistent_knowledge=True, combat_features=True, king_threat_distance=True)
    with GeneralsTask(config) as task:
        runner = runner_config(); options = runner['policy'].copy()
        policy = resolve_callable(options.pop('class_name'))(
            TensorDict(task.get_observations(), [2]), runner['obs_groups'], task.action_space.size, **options)
        with torch.no_grad():
            probabilities = policy.action_probabilities(TensorDict(task.get_observations(), [2]))
        assert probabilities.shape == (2, 1 + 8 * 23 * 23)
        torch.testing.assert_close(probabilities.sum(1), torch.ones(2))


def test_terminal_actor_features_publish_enemy_city_count():
    from generals import MapConfig, VecEnv
    from generals.core.mapgen import generate_map
    from generals.core.engine import surrender
    from generals_rl.policy.observations import LocalObservationEncoder
    from generals_rl.mdp.state import GameBatch
    state = generate_map(MapConfig(players=2), 42)
    city = np.flatnonzero(state.structure.ravel() == 1)[0]
    state.owner.flat[city] = 1; state.tick = 1
    surrender(state, 1)
    with VecEnv.from_states([state]) as backend:
        batch = GameBatch.from_backend(backend, 'cpu')
        features = LocalObservationEncoder(2000).encode(batch, torch.tensor([0]))
        assert features[0, 14 * state.layout.size + 6].item() == pytest.approx(1 / state.layout.size)


def test_without_map_bank_initial_lanes_are_independent_and_repeatable():
    config = TaskConfig(num_envs=3, map_pool_size=0, opponent='wait')
    with GeneralsTask(config) as task:
        before = task.backend.state['terrain'].copy()
        assert not np.array_equal(before[0], before[1])
        task.reset()
        np.testing.assert_array_equal(task.backend.state['terrain'], before)
    with GeneralsTask(config) as repeated:
        np.testing.assert_array_equal(repeated.backend.state['terrain'], before)


def test_short_training_saves_exports_and_evaluates_official_maps(tmp_path):
    from contextlib import redirect_stdout
    import io
    from generals_rl.train import train
    from rl_test_utils import small_training_config

    config = small_training_config(task_overrides=dict(
        width=23, height=23, random_maps=True, map_pool_size=8,
        max_episode_steps=2000, mountain_density=.5, city_density=.5,
        persistent_knowledge=True), runner_overrides={'learning_epochs': 1})
    config['runner']['policy'].update(map_width=23, map_height=23, knowledge_channels=11,
                                      critic_actor_information=True)
    config['runner']['obs_groups'] = {'policy': ['policy', 'knowledge'],
                                     'critic': ['critic', 'policy', 'knowledge']}
    config['self_play'] = {'pool_size': 2, 'snapshot_interval': 1}
    config['evaluation'] = {'enabled': True, 'every_n_updates': 2, 'maps_per_opponent': 1,
                            'max_ticks': 2000, 'batch_size': 8, 'final_top_k': 0}
    with redirect_stdout(io.StringIO()):
        summary = train(config, tmp_path/'run', iterations=2)
    assert summary['evaluation']['completed'] == 1 and summary['evaluation']['failed'] == 0
    best = json.loads((tmp_path/'run/best_model.json').read_text())
    assert Path(best['checkpoint']).is_file() and Path(best['bundle']).is_dir()
    report = json.loads(Path(best['report']).read_text())
    assert report['games'] == 10
    for result in report['results']:
        assert result['map']['width'] is None and result['map']['height'] is None
        assert result['ticks'] <= 2000


def test_cached_afk_winner_receives_win_reward_without_loss():
    from generals import MapConfig, VecEnv
    from generals.core.mapgen import generate_map
    state = generate_map(MapConfig(players=2), 42)
    state.tick = 70
    config = TaskConfig(width=state.layout.width, height=state.layout.height,
                        random_maps=False, map_pool_size=0, num_envs=2,
                        opponent='wait', rotate_seats=True)
    with VecEnv.from_states([state, state], episode_ticks=2000) as backend:
        with GeneralsTask(config, backend=backend) as task:
            result = task.step(torch.zeros(2, dtype=torch.long))
            assert result.terminated.tolist() == [True, True]
            assert result.truncated.tolist() == [False, False]
            assert result.metrics['Reward/terminal'].tolist() == [-1., 1.]
            assert result.metrics['Episode/win'].tolist() == [0., 1.]
