"""Terrain BFS, fog boundaries and frozen PPO feature snapshots."""
from contextlib import redirect_stdout
import io

import pytest
import torch
from tensordict import TensorDict

from generals_rl.config import task_config
from generals_rl.policy.paths import TerrainDistances
from generals_rl.rollout_codec import PathObservationCodec
from generals_rl.task import GeneralsTask
from test_sparse_features import recipe, test_runtime_memory_matches_training_through_steps_and_reset as check_runtime
from test_value_distribution import assert_train_resume_export
from test_shared_critic import model_for


def inputs(width=9, height=9, games=3):
    cells = width * height
    terrain = torch.zeros(games, 2, cells, dtype=torch.int8)
    generals = torch.full((games, 2, cells), -1, dtype=torch.int16)
    generals[:, :, 0] = 0
    generals[:, :, -1] = 1
    seats = torch.arange(2).repeat(games)
    rows = torch.arange(games).repeat_interleave(2)
    fog = torch.zeros(games*2, cells, dtype=torch.bool)
    valid = torch.ones_like(fog)
    active = torch.ones(len(seats), dtype=torch.bool)
    return terrain, generals, fog, valid, seats, rows, active


def test_distances_detour_unknown_city_and_discovery(monkeypatch):
    args = list(inputs(width=5, height=3, games=1))
    terrain, generals, fog, valid, seats, rows, active = args
    # Vertical wall with a bottom opening: (0,0)->(0,2) is six steps.
    terrain[:, :, [1, 6]] = 1
    generals[:, :, -1] = -1
    cache = TerrainDistances(1, 2, 5, 3, 'cpu')
    calls = []
    bfs = cache._bfs
    monkeypatch.setattr(cache, '_bfs', lambda *a: (calls.append(1), bfs(*a))[1])
    cache.update(*args)
    assert cache.distances[0, 0, 0, 2] == 6
    assert (cache.distances[0, 0, 1] == -2).all()
    assert cache.distances[0, 0, 0, 1] == -1
    before = len(calls)
    cache.update(*args)
    assert len(calls) == before
    # Fog hint cannot distinguish a mountain from a city. Observed city opens it.
    terrain[:, :, 11] = -1
    fog[:, 11] = True
    cache.update(*args)
    assert cache.distances[0, 0, 0, 2] == -1
    terrain[:, :, 11] = 0
    cache.update(*args)
    assert cache.distances[0, 0, 0, 2] == 6
    generals[:, :, 2] = 1
    cache.update(*args)
    assert cache.distances[0, 0, 1, 0] == 6
    assert cache.distances[0, 0, 1, 2] == 0
    assert cache.features(seats, rows)[0, 1, 2] == 0
    grown = TerrainDistances(4, 2, 5, 3, 'cpu')
    grown.copy_prefix(cache)
    for name in ('blocked', 'goals', 'distances'):
        assert torch.equal(getattr(grown, name)[:1], getattr(cache, name))
    assert (grown.distances[1:] == -2).all()


@pytest.mark.parametrize('width,height', [(9, 9), (23, 23), (7, 4)])
def test_cuda_matches_bfs_with_cache_masks_reset_and_long_corridor(width, height):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    args = list(inputs(width, height, games=13))
    terrain, generals, fog, valid, seats, rows, active = args
    rng = torch.Generator().manual_seed(971)
    terrain.copy_(torch.randint(-1, 2, terrain.shape, generator=rng, dtype=torch.int8))
    terrain[:, :, 0] = terrain[:, :, -1] = 0
    fog.copy_(torch.rand(fog.shape, generator=rng) < .3)
    valid[4:6, width-1::width] = False
    # Serpentine corridor defeats implementations bounded to width+height steps.
    board = terrain[0].reshape(2, height, width)
    board.fill_(1)
    for y in range(height):
        if y % 2 == 0:
            board[:, y, :] = 0
        else:
            board[:, y, -1 if (y//2)%2 == 0 else 0] = 0
    terrain[0, :, -1] = 0
    cpu = TerrainDistances(13, 2, width, height, 'cpu')
    gpu = TerrainDistances(13, 2, width, height, 'cuda')
    def compare():
        cpu.update(*args)
        gpu.update(*(x.cuda() for x in args))
        for name in ('blocked', 'goals', 'distances'):
            assert torch.equal(getattr(cpu, name), getattr(gpu, name).cpu()), name
    compare()
    compare()
    if width == height == 23:
        assert cpu.distances[0, 0, 0, -1] > width+height
    active[::3] = False
    generals[:, :, -1] = -1
    terrain[:, :, width] = 0
    compare()
    reset = torch.arange(13) % 2 == 0
    cpu.reset(reset)
    gpu.reset(reset.cuda())
    compare()


def test_fog_only_knowledge_and_ppo_snapshot():
    cfg = recipe('temporal_paths')
    assert cfg['runner']['storage_codec'].endswith('PathObservationCodec')
    with GeneralsTask(task_config(cfg['task'])) as task:
        obs = TensorDict(task.get_observations(), batch_size=[task.num_envs]).clone()
        codec = PathObservationCodec(obs)
        encoded = codec.encode('knowledge', obs['knowledge']).unsqueeze(0)
        original = obs['knowledge'].clone()
        # Change knowledge after collecting the transition; decode must not query it.
        task.knowledge.paths.distances.fill_(0)
        times = torch.zeros(task.num_envs, dtype=torch.long)
        lanes = torch.arange(task.num_envs)
        assert torch.equal(codec.gather('knowledge', encoded, times, lanes), original)
        task.knowledge.paths.goals.fill_(-2)
        # Same observed data/public hints; hidden mountain-vs-city identity cannot leak.
        batch = task.batch()
        visible = task.observation_encoder.visibility(batch, task.player_ids)
        from generals_rl.policy.state import GameBatch
        fog = batch.fog_obstacles(task.player_ids, visible)
        state = {k: v.clone() for k, v in batch.state.items()}
        state['fog_obstacles'] = fog
        public = GameBatch(state, batch.stats, batch.width, batch.height, valid=batch.valid_cells)
        copy = task.knowledge.clone()
        task.knowledge.update(public, task.player_ids, lane_ids=task._game_rows, visible=visible)
        state['terrain'][~visible] = 1
        state['structure'][~visible] = 2
        state['army'][~visible] = 999999
        copy.update(public, task.player_ids, lane_ids=task._game_rows, visible=visible)
        for name in ('blocked', 'goals', 'distances'):
            assert torch.equal(getattr(copy.paths, name), getattr(task.knowledge.paths, name))
        task.knowledge.reset()
        assert (task.knowledge.paths.distances == -2).all()
        assert not torch.equal(copy.paths.distances, task.knowledge.paths.distances)


def test_paths_train_resume_and_export(tmp_path):
    with redirect_stdout(io.StringIO()):
        assert_train_resume_export(tmp_path, recipe('temporal_paths'))


def test_new_channels_fill_padding_without_moving_existing_features():
    inputs, weights = [], []
    for features in ('temporal', 'temporal_paths'):
        cfg = recipe(features)
        with GeneralsTask(task_config(cfg['task'])) as task:
            torch.manual_seed(91)
            model, obs = model_for(task, cfg)
            inputs.append(model.actor.encoder.input_maps(obs['policy'], obs['knowledge'], obs['valid_cells']))
            weights.append({k:v.clone() for k,v in model.actor.state_dict().items()})
    assert inputs[0].shape == inputs[1].shape
    assert inputs[0].shape[1] == 40
    torch.testing.assert_close(inputs[0][:, :37], inputs[1][:, :37], rtol=0, atol=0)
    assert not inputs[0][:, 37:].any()
    assert inputs[1][:, 37:39].any()
    for key in weights[0]:
        assert torch.equal(weights[0][key], weights[1][key]), key


def test_paths_export_runtime_and_capacity(tmp_path):
    check_runtime(tmp_path, 'temporal_paths')
