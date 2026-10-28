"""Independent official-client fixtures plus cross-backend lifecycle checks."""
from pathlib import Path
import json
import numpy as np
import pytest
from generals import VecEnv
from generals.core.engine import CoreEngine
from generals.core.state import GameState,MapLayout
from generals.core.batch import pack_states
from generals.core.recording import state_digest
from generals.protocol import Action

FIXTURES=Path(__file__).parent/'fixtures'/'official_31_4_5'
CASES=json.loads((FIXTURES/'edge_cases.json').read_text())


def initial(case):
    s=case['initial'];shape=(s['height'],s['width'])
    terrain=np.zeros(s['width']*s['height'],np.uint8);terrain[s['mountains']]=1
    return GameState(MapLayout(terrain.reshape(shape)),np.array(s['structure'],np.uint8).reshape(shape),
                     np.array(s['owner'],np.int16).reshape(shape),np.array(s['army'],np.int64).reshape(shape),
                     np.ones(len(s['generals']),bool),np.array(s['generals'],np.int32),tick=s['tick'])


@pytest.mark.parametrize('backend',['reference','cpu','gpu'])
def test_official_priority_combat_and_trades(backend):
    if backend=='gpu':
        torch=pytest.importorskip('torch')
        if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    for case in CASES:
        state=initial(case)
        if backend=='reference':
            CoreEngine(state,debug=True).step([Action.decode(a) for a in case['actions']])
        else:
            with VecEnv.from_states([state],backend=backend,device='cuda' if backend=='gpu' else 'cpu') as env:
                actions=np.array([case['actions']],np.int64)
                if backend=='gpu':actions=torch.as_tensor(actions,device='cuda')
                env.step(actions);env.check_errors();state=env.to_states()[0]
        for key,expected in case['expected'].items():
            np.testing.assert_array_equal(getattr(state,key).reshape(np.shape(expected)),expected,
                                          err_msg=f"{case['name']}: {backend} {key}")


@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('group',[8,32,64])
def test_full_native_metadata_reset_and_observation_parity(packed,group):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    states=[initial(c) for c in CASES if c['name'].startswith('joint_random_')][:17]
    with VecEnv.from_states(states,device='cpu',auto_reset=True,episode_ticks=53) as cpu, VecEnv.from_states(states,device='cuda',auto_reset=True,episode_ticks=53) as gpu:
        gpu.unwrapped.configure_step(group_threads=group,envs_per_block=1 if group>32 else 4,packed_metadata=packed)
        co=cpu.observer(np.arange(len(states))%4);go=gpu.observer(np.arange(len(states))%4)
        rng=np.random.default_rng(92)
        for tick in range(12):
            actions=rng.integers(0,201,size=(len(states),4),dtype=np.int64)
            a,b=cpu.step(actions),gpu.step(torch.as_tensor(actions,device='cuda'))
            for name in cpu.state:
                np.testing.assert_array_equal(cpu.state[name],gpu.state[name].cpu(),err_msg=f'{tick}: {name}')
            for name in ('stats','moves','done','errors'):
                np.testing.assert_array_equal(getattr(a,name),getattr(b,name).cpu())
            x,y=co(),go()
            for name in x['state']:
                np.testing.assert_array_equal(x['state'][name],y['state'][name].cpu())
            if tick==5:
                mask=np.arange(len(states))%2==0;cpu.reset(mask);gpu.reset(torch.as_tensor(mask,device='cuda'))


def test_departure_grace_credit_and_public_city_memory():
    from generals.core.observation import observe
    from generals.backends.cpu.engine import CPUEngine
    state=initial(CASES[1]);state.tick=1
    # Three seats keep the game running after a surrender.
    state.alive=np.ones(3,bool);state.general_pos=np.array([0,7,4],np.int32)
    state.structure.flat[4]=2;state.owner.flat[4]=2;state.army.flat[4]=1
    # Rebuild lifecycle arrays for the third seat.
    state=GameState(state.layout,state.structure,state.owner,state.army,state.alive,state.general_pos,tick=1)
    state.damage[2,1]=8
    engines=[CoreEngine(state.copy()),CPUEngine(state.copy())]
    try:
        for engine in engines:engine.surrender(1)
        assert not engines[0].state.alive[1] and engines[0].state.general_pos[1]==7
        for _ in range(50):
            for engine in engines:engine.step([Action.wait()]*3)
            assert state_digest(engines[0].state)==state_digest(engines[1].state)
        s=engines[0].state
        assert s.general_pos[1]==-1 and s.structure.flat[7]==1 and s.owner.flat[7]==-2
        assert s.kills[2]==1 and s.capture_mask[2]==2
        assert np.all(observe(s,0).neutralize_in==-1)
    finally:engines[1].close()


@pytest.mark.parametrize('backend', ['reference', 'cpu', 'gpu'])
def test_live_afk_rechecks_warning_and_retains_last_survivor(backend):
    """Live 31.4.5 checkAfks/killPlayer, unlike the replay UI, caches winners."""
    if backend == 'gpu':
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
    shape = (20, 20)
    owner = np.full(shape, -1, np.int16)
    army = np.zeros(shape, np.int64)
    structure = np.zeros(shape, np.uint8)
    for pid, cell in enumerate((0, 399)):
        owner.flat[cell], army.flat[cell], structure.flat[cell] = pid, 1, 2
    start = GameState(MapLayout(np.zeros(shape, np.uint8)), structure, owner, army,
                      np.ones(2, bool), np.array([0, 399], np.int32), afk_turn=71)

    def step(state):
        if backend == 'reference':
            CoreEngine(state, debug=True).step([Action.wait()] * 2)
            return state
        with VecEnv.from_states([state], device='cuda' if backend == 'gpu' else 'cpu') as env:
            actions = np.zeros((1, 2), np.int64)
            env.step(torch.as_tensor(actions, device='cuda') if backend == 'gpu' else actions)
            env.check_errors()
            result = env.to_states()[0]
            result.validate()
            return result

    start.tick = 48
    warned = step(start.copy())
    assert warned.warn_afk.tolist() == [1, 1]
    warned.owner.flat[:11] = 0
    warned.army.flat[:11] = 1
    safe = step(warned)
    assert safe.warn_afk.tolist() == [0, 1]
    start.tick = 70
    final = step(start.copy())
    assert final.terminated and final.winner_id == 1 and final.end_reason == 5
    assert final.alive.tolist() == [False, False]
    assert final.death_order.tolist() == [0, 1] and final.kills.tolist() == [0, 0]
    assert final.leave_tick.tolist() == [71, 71]
    assert final.warn_afk.tolist() == [0, 0]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('players', [2, 3])
@pytest.mark.parametrize('tick', [0, 1, 47])
def test_batched_surrender_matches_reference_through_grace(device, players, tick):
    from generals import MapConfig
    from generals.core.batch import generate_states
    if device == 'cuda':
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    states = generate_states(MapConfig(players=players), 3, seed=84)
    for state in states: state.tick = tick
    references = [CoreEngine(state.copy()) for state in states]
    with VecEnv.from_states(states, device=device, auto_reset=False) as env:
        mask = np.array([True, False, True])
        env.surrender(0, torch.as_tensor(mask, device=device) if device == 'cuda' else mask)
        for lane in (0, 2): references[lane].surrender(0)
        actions = np.zeros((3, players), np.int64)
        if device == 'cuda': actions = torch.as_tensor(actions, device=device)
        for step in range(52):
            actual = env.to_states()
            assert [state_digest(s) for s in actual] == [state_digest(e.state) for e in references], step
            if step == 50:
                env.surrender(players-1)
                for engine in references: engine.surrender(players-1)
            env.step(actions)
            for engine in references:
                if not engine.state.terminated: engine.step([Action.wait()] * players)
