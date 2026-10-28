"""Deployed actors preserve the real training task information boundary."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from generals import Action, Direction, MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.state import GameState, MapLayout

from generals_rl.policy.distributions import action_probabilities, select_action
from generals_rl.policy.precision import matmul_precision
from generals_rl.policy.runtime import create_controller
from generals_rl.policy.state import GameBatch
from rl_policy_test_utils import make_checkpoint
from generals_rl.config import task_config
from generals_rl.export import export_policy
from generals_rl.task import GeneralsTask
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict


@pytest.fixture
def policy_files(tmp_path):
    torch.set_num_threads(1)
    checkpoint, config = make_checkpoint(tmp_path, family="scaled")
    config["task"].update(max_episode_steps=1024)
    (tmp_path/"run_config.json").write_text(json.dumps(config))
    bundle = export_policy(checkpoint, tmp_path/"bundle", run_config=config,
                           completed_updates=1, inference="greedy")
    return bundle/"actor.pt", bundle/"model_config.json", config


def capture_inputs(controller):
    captured = []
    original = controller._inputs

    def capture(batch, *args, **kwargs):
        inputs = original(batch, *args, **kwargs)
        captured.append({key: value.clone() for key, value in inputs.items()})
        return inputs

    controller._inputs = capture
    return captured


def battle_state():
    """Visible combat, a wall, a neutral city and one-army growth boundaries."""
    terrain = np.zeros((9, 9), np.uint8)
    structure = np.zeros((9, 9), np.uint8)
    owner = np.full((9, 9), -1, np.int16)
    army = np.zeros((9, 9), np.int64)
    for cell, pid, troops, kind in (
        (0, 0, 1, 2), (80, 1, 1, 2), (40, 0, 1, 0), (41, 0, 7, 0),
        (39, 1, 5, 1), (31, 1, 1, 0), (49, -2, 40, 1),
    ):
        owner.flat[cell], army.flat[cell], structure.flat[cell] = pid, troops, kind
    terrain.flat[30] = 1
    state = GameState(MapLayout(terrain), structure, owner, army, np.ones(2, bool),
                      np.array([0, 80], np.int32))
    state.validate()
    return state


@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize("start_tick", [0, 48, 1023, 1999])
def test_policy_inputs_history_knowledge_and_actions_match_real_training_task(policy_files, seat, start_tick):
    weights, config_path, config = policy_files
    controller = create_controller(weights=weights, config_path=config_path)
    controller.reset(seat, 17)
    captured = capture_inputs(controller)
    config = deepcopy(config)
    config["task"].update(num_envs=1, learner_seat=seat, rotate_seats=False)
    state = battle_state()
    state.tick = start_tick
    enemy_general = int(state.general_pos[1-seat])
    outpost = 79 if seat == 0 else 1
    state.owner.flat[outpost], state.army.flat[outpost] = seat, 2
    state.army.flat[enemy_general] = 20
    recapture = Action.move(enemy_general, Direction.LEFT if seat == 0 else Direction.RIGHT).encode()

    class RecaptureOnce:
        def act(self, batch, player_ids, *args):
            return torch.full_like(player_ids, recapture if int(batch.state["tick"][0]) == start_tick else 0)

    # The backend horizon is independent of the model's normalization horizon,
    # as in Arena matches longer than the episodes used during training.
    backend = VecEnv.from_states([state], device="cpu", threads=1,
                                 auto_reset=False, episode_ticks=100000)
    with GeneralsTask(task_config(config["task"]), backend=backend, opponent=RecaptureOnce()) as task:
        options = deepcopy(config["runner"]["policy"])
        policy_type = resolve_callable(options.pop("class_name"))
        with torch.random.fork_rng(devices=[]):
            training = policy_type(TensorDict(task.get_observations(), [1]),
                                   config["runner"]["obs_groups"], 649, **options)
        saved = torch.load(weights.parent.parent/"checkpoint.pt", weights_only=True)
        training.load_state_dict(saved["model_state_dict"])
        training.eval()
        for step in range(4):
            current = task.backend.to_states()[0]
            selected = controller.act(observe(current, seat))
            expected = task.get_observations()
            actual = captured[-1]
            assert set(actual) == {"policy", "single_obs", "action_mask", "knowledge"}
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            torch.testing.assert_close(controller.history.frames, task.history.frames, rtol=0, atol=0)
            for name in task.knowledge._fields:
                torch.testing.assert_close(getattr(controller.knowledge, name), getattr(task.knowledge, name), rtol=0, atol=0)
            with torch.inference_mode():
                expected_action = training.act_inference(TensorDict(expected, [1]))
                logits = controller.actor(actual["policy"], actual["knowledge"]).float()
                trained_logits = training.actor(expected["policy"], expected["knowledge"]).float()
                torch.testing.assert_close(logits, trained_logits, rtol=0, atol=0)
            assert selected.encode() == int(expected_action.item())
            assert float(actual["single_obs"][0, -5]) == (start_tick + step)/1024
            assert torch.isfinite(actual["policy"]).all()
            if step:
                assert not observe(current, seat).visible.flat[enemy_general]
                age = actual["knowledge"].reshape(1, 11, 81)[0, -1, enemy_general]
                torch.testing.assert_close(age, torch.tensor(step / 2000, device=age.device))
            # A controlled action stream isolates visibility and clock boundaries.
            task.step(torch.zeros(1, dtype=torch.long))
    assert not any(key.startswith("critic") for key in controller.actor.state_dict())


def test_policy_greedy_preserves_wait_move_gate_instead_of_global_argmax():
    logits = torch.zeros(1, 649)
    logits[:, 0] = -0.1
    mask = torch.ones_like(logits, dtype=torch.bool)
    probabilities = action_probabilities(logits, mask)
    assert int(probabilities.argmax(-1).item()) == 0
    assert int(select_action(logits, mask).item()) == 1
    logits[:, 0] = 1
    assert int(select_action(logits, mask).item()) == 0
    mask[:, 1:] = False
    assert int(select_action(logits, mask).item()) == 0


def test_public_observation_conversion_contains_only_masked_board_and_public_stats():
    state = generate_map(MapConfig(9, 9, 2, .1, .05), seed=25)
    for seat in (0, 1):
        observation = observe(state, seat)
        batch = GameBatch.from_observation(observation, "cpu")
        hidden = torch.tensor(~observation.visible.ravel())
        assert "general_pos" not in batch.state
        assert (batch.state["owner"][0, hidden] == -3).all()
        assert (batch.state["army"][0, hidden] == 0).all()
        assert int(batch.stats[0, 1-seat, 2]) == -1
        torch.testing.assert_close(batch.stats[0, :, :2], torch.tensor(np.stack(
            (observation.stats.army, observation.stats.territory), axis=-1)), rtol=0, atol=0)


def test_policy_reset_rng_contract_and_rule_validation(policy_files):
    checkpoint, config_path, _ = policy_files
    torch.manual_seed(1729)
    before = torch.random.get_rng_state().clone()
    controller = create_controller(weights=checkpoint, config_path=config_path)
    torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
    state = generate_map(MapConfig(9, 9, 2, 0, 0), seed=18)
    first = controller.act(observe(state, 0))
    initial = controller.history.frames.clone()
    state.tick = 50
    controller.act(observe(state, 0))
    controller.reset(0, 999)
    torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
    assert not controller._initialized.any()
    assert (controller.knowledge.last_seen_tick == -1).all()
    state.tick = 0
    assert controller.act(observe(state, 0)) == first
    torch.testing.assert_close(initial, controller.history.frames, rtol=0, atol=0)
    controller.reset(1, 17)
    assert isinstance(controller.act(observe(state, 1)), Action)
    controller.validate_rules(MapConfig(9, 9, 2), "local")
    for rules in (MapConfig(7, 9, 2), MapConfig(9, 9, 4)):
        with pytest.raises(ValueError, match="shape|player"):
            controller.validate_rules(rules, "local")
    with pytest.raises(ValueError, match="local visibility"):
        controller.validate_rules(MapConfig(9, 9, 2), "full")


def test_explicit_actor_only_format_and_strict_loading(policy_files, tmp_path):
    checkpoint, config_path, _ = policy_files
    original = create_controller(weights=checkpoint, config_path=config_path)
    actor_path = tmp_path / "actor.pt"
    torch.save({"actor_state_dict": original.actor.state_dict()}, actor_path)
    actor_only = create_controller(weights=actor_path, config_path=config_path)
    for key, value in original.actor.state_dict().items():
        torch.testing.assert_close(actor_only.actor.state_dict()[key], value, rtol=0, atol=0)
    extra = dict(original.actor.state_dict())
    extra["unexpected"] = torch.zeros(1)
    torch.save({"actor_state_dict": extra}, actor_path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        create_controller(weights=actor_path, config_path=config_path)
    torch.save({"model_state_dict": {"critic.fake": torch.zeros(1)}}, actor_path)
    with pytest.raises(ValueError, match="actor_state_dict"):
        create_controller(weights=actor_path, config_path=config_path)


def test_inference_only_subprocess_forbids_live_training_imports(policy_files):
    checkpoint, config_path, _ = policy_files
    script = r'''
import importlib.abc
import sys

class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('generals_rl', 'rsl_rl'):
            raise AssertionError('policy imported training package: ' + fullname)

sys.meta_path.insert(0, BlockTraining())
import torch
from generals import MapConfig
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from pathlib import Path
import importlib.util
root = Path('rl/src/generals_rl/policy').resolve()
spec = importlib.util.spec_from_file_location('_exported_policy', root/'__init__.py', submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules['_exported_policy'] = module
spec.loader.exec_module(module)
from _exported_policy.runtime import create_controller
torch.set_num_threads(1)
controller = create_controller(weights=sys.argv[1], config_path=sys.argv[2])
action = controller.act(observe(generate_map(MapConfig(9, 9, 2, 0, 0), 18), 0))
assert action.encode() >= 0
assert not any(name.split('.')[0] in ('generals_rl', 'rsl_rl') for name in sys.modules)
print('frozen-only OK')
'''
    result = subprocess.run([sys.executable, "-c", script, str(checkpoint), str(config_path)],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "frozen-only OK" in result.stdout


def test_policy_configuration_rejects_unknown_mode_and_unsupported_shape(policy_files, tmp_path):
    checkpoint, config_path, _ = policy_files
    config = json.loads(config_path.read_text())
    other = deepcopy(config)
    other["inference"]["mode"] = "sample"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="greedy"):
        create_controller(weights=checkpoint, config_path=bad)
    other = deepcopy(config)
    other["features"].update(width=0, height=11)
    bad.write_text(json.dumps(other))
    with pytest.raises(ValueError):
        create_controller(weights=checkpoint, config_path=bad)


def test_unspecified_precision_never_reads_or_writes_global_flags(monkeypatch):
    def unexpected(*args):
        raise AssertionError("unspecified precision touched global flags")
    monkeypatch.setattr(torch, "get_float32_matmul_precision", unexpected)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", unexpected)
    with matmul_precision():
        pass


@pytest.mark.parametrize("fail", [False, True])
def test_precision_context_restores_flags_with_mocked_cpu_backend(monkeypatch, fail):
    state = {"precision": "highest"}
    backend = SimpleNamespace(allow_tf32=False, benchmark=False)
    monkeypatch.setattr(torch.backends, "cudnn", backend)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: state["precision"])
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda value: state.update(precision=value))
    try:
        with matmul_precision("high", True, True):
            assert state["precision"] == "high" and backend.allow_tf32 and backend.benchmark
            with matmul_precision("medium", False, False):
                assert state["precision"] == "medium" and not backend.allow_tf32 and not backend.benchmark
            assert state["precision"] == "high" and backend.allow_tf32 and backend.benchmark
            if fail:
                raise RuntimeError("forward failed")
    except RuntimeError as error:
        assert fail and str(error) == "forward failed"
    assert state["precision"] == "highest" and not backend.allow_tf32 and not backend.benchmark


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("fail", [False, True])
def test_runtime_scopes_precision_for_each_forward_and_restores_on_error(policy_files, monkeypatch, device, fail):
    checkpoint, config_path, _ = policy_files
    previous = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    requested = ("high" if previous[0] == "highest" else "highest", not previous[1], not previous[2])
    config = json.loads(config_path.read_text())
    config["inference"].update(dict(zip(("matmul_precision", "cudnn_allow_tf32", "cudnn_benchmark"), requested)))
    config_path.write_text(json.dumps(config))
    controller = create_controller(weights=checkpoint, config_path=config_path, device=device)
    original = controller.actor.forward
    called = []

    def forward(*args, **kwargs):
        assert (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark) == requested
        called.append(True)
        if fail:
            raise RuntimeError("forward failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(controller.actor, "forward", forward)
    observation = observe(battle_state(), 0)
    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            controller.act(observation)
    else:
        assert controller.act(observation).encode() >= 0
    assert called == [True]
    assert (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark) == previous


@pytest.mark.parametrize("key,value", [("matmul_precision", "invalid"), ("cudnn_allow_tf32", 1),
                                      ("cudnn_benchmark", "true")])
def test_runtime_rejects_invalid_precision_configuration(policy_files, key, value):
    checkpoint, config_path, _ = policy_files
    config = json.loads(config_path.read_text())
    config["inference"][key] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises((ValueError, TypeError), match=key):
        create_controller(weights=checkpoint, config_path=config_path)


def raw_batch(observations, *, device="cpu", active=None):
    batch = GameBatch.from_observations(observations, device)
    # Match the raw env contract: hidden troops are -1 until the runtime masks
    # them before log1p. All public feature code sees only these raw arrays.
    batch.state["army"] = torch.tensor(np.stack([obs.army.ravel() for obs in observations]), device=device)
    return {"state": batch.state, "stats": batch.stats, "width": batch.width, "height": batch.height,
            "visible": torch.tensor(np.stack([obs.visible.ravel() for obs in observations]), device=device),
            "player_ids": torch.tensor([obs.player_id for obs in observations], device=device),
            "visibility": "local", "active": torch.tensor(
                [True] * len(observations) if active is None else active, dtype=torch.bool, device=device)}


def subset_raw(raw, indices):
    return {key: {name: value[indices] for name, value in item.items()} if key == "state"
            else item[indices] if isinstance(item, torch.Tensor) else item for key, item in raw.items()}


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_runtime_reuses_public_visibility_without_mutating_it(policy_files, monkeypatch, device):
    weights, config, _ = policy_files
    controller = create_controller(weights=weights, config_path=config, device=device)
    controller.reset_batch([0, 1], [3, 7])
    raw = raw_batch([observe(battle_state(), seat) for seat in (0, 1)],
                    device=device, active=[True, False])
    visible = raw['visible'].clone()

    def recomputed(*args, **kwargs):
        pytest.fail('public visibility was recomputed in policy features or knowledge')

    monkeypatch.setattr(torch.nn.functional, 'max_pool2d', recomputed)
    actions = controller.act_batch(raw)
    assert actions.shape == (2,) and actions[1] == 0
    torch.testing.assert_close(raw['visible'], visible, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_tensor_batch_matches_scalar_memory_and_greedy_actions(policy_files, device):
    checkpoint, config_path, _ = policy_files
    batched = create_controller(weights=checkpoint, config_path=config_path, device=device)
    players, seeds = [0, 1, 0], [11, 23, 37]
    batched.reset_batch(np.asarray(players), np.asarray(seeds))
    scalars = [create_controller(weights=checkpoint, config_path=config_path, device=device,
                                 player_id=player, seed=seed) for player, seed in zip(players, seeds)]
    states = [battle_state() for _ in players]
    for tick in (0, 1, 49, 50, 1024):
        for state in states:
            state.tick = tick
        observations = [observe(state, player) for state, player in zip(states, players)]
        encoded = batched.act_batch(raw_batch(observations, device=device))
        assert encoded.dtype == torch.long and encoded.device.type == device
        assert encoded.tolist() == [controller.act(obs).encode() for controller, obs in zip(scalars, observations)]
        for lane, scalar in enumerate(scalars):
            torch.testing.assert_close(batched.history.frames[lane], scalar.history.frames[0], rtol=0, atol=0)
            for name in batched.knowledge._fields:
                torch.testing.assert_close(getattr(batched.knowledge, name)[lane],
                                           getattr(scalar.knowledge, name)[0], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_stochastic_streams_survive_batch_size_order_inactive_and_partial_reset(policy_files, tmp_path, device):
    checkpoint, config_path, _ = policy_files
    config = json.loads(config_path.read_text())
    config["inference"]["mode"] = "stochastic"
    config_path.write_text(json.dumps(config))
    initial = create_controller(weights=checkpoint, config_path=config_path)
    # A uniform actor isolates the sampling protocol from tiny batch-dependent
    # GEMM rounding differences; the action masks and state remain nontrivial.
    weights = tmp_path / "uniform.pt"
    torch.save({"actor_state_dict": {key: torch.zeros_like(value)
                                     for key, value in initial.actor.state_dict().items()}}, weights)
    full = create_controller(weights=weights, config_path=config_path, device=device)
    split = create_controller(weights=weights, config_path=config_path, device=device)
    players, seeds = [0, 1, 0], [11, 2**48 + 23, 37]
    full.reset_batch(players, seeds)
    split.reset_batch(players, seeds)
    scalars = [create_controller(weights=weights, config_path=config_path, device=device,
                                 player_id=player, seed=seed) for player, seed in zip(players, seeds)]
    observations = [observe(battle_state(), player) for player in players]
    for tick in range(12):
        active = [True, tick not in (3, 4, 5), True]
        raw = raw_batch(observations, device=device, active=active)
        raw["state"]["tick"].fill_(tick)
        previous_history = full.history.frames.clone()
        previous_memory = {name: getattr(full.knowledge, name).clone() for name in full.knowledge._fields}
        actual = full.act_batch(raw)
        reordered = torch.zeros_like(actual)
        for lanes in ([2, 0], [1]):
            ids = torch.tensor(lanes, device=device)
            reordered[ids] = split.act_batch(subset_raw(raw, ids), lane_ids=ids)
        torch.testing.assert_close(actual, reordered, rtol=0, atol=0)
        expected = []
        for lane, scalar in enumerate(scalars):
            lane_raw = subset_raw(raw, torch.tensor([lane], device=device))
            expected.append(int(scalar.act_batch(lane_raw).item()))
        assert actual.tolist() == expected
        torch.testing.assert_close(full._counter, split._counter, rtol=0, atol=0)
        if not active[1]:
            assert int(actual[1]) == 0
            torch.testing.assert_close(full.history.frames[1], previous_history[1], rtol=0, atol=0)
            for name in previous_memory:
                torch.testing.assert_close(getattr(full.knowledge, name)[1], previous_memory[name][1], rtol=0, atol=0)
    history = full.history.frames.clone()
    full.reset_batch([1], [99], lane_ids=[1])
    torch.testing.assert_close(full.history.frames[[0, 2]], history[[0, 2]], rtol=0, atol=0)
    assert int(full._counter[1]) == 0 and not bool(full._initialized[1])
    assert (full.knowledge.last_seen_tick[1] == -1).all()
