"""Conformance against observed frames from the deployed official 31.4.5 client."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pytest
from generals.core.engine import CoreEngine, neutralize, surrender
from generals.core.observation import observe
from generals.core.state import GameState, MapLayout
from generals.protocol import Action, Direction, MoveMode, EMPTY, NEUTRAL

FIXTURES = Path(__file__).parent / 'fixtures' / 'official_31_4_5'


def replay_state(data):
    h, w = data['height'], data['width']
    terrain = np.zeros(h*w, dtype=np.uint8)
    structure = np.zeros_like(terrain)
    owner = np.full(h*w, EMPTY, dtype=np.int16)
    army = np.zeros(h*w, dtype=np.int64)
    terrain[data['mountains']] = 1
    for cell, count in zip(data['cities'], data['cityArmies']):
        structure[cell], owner[cell], army[cell] = 1, NEUTRAL, count
    for pid, cell in enumerate(data['generals']):
        structure[cell], owner[cell], army[cell] = 2, pid, 1
    return GameState(MapLayout(terrain.reshape(h,w)), structure.reshape(h,w),
                     owner.reshape(h,w), army.reshape(h,w),
                     np.ones(len(data['generals']), dtype=bool), np.array(data['generals']))


def canonical_digest(s):
    values = [np.where(s.owner.ravel() >= 0, s.owner.ravel(), -1).tolist(),
              s.army.ravel().tolist(), s.structure.ravel().tolist(),
              s.general_pos.tolist(), s.alive.tolist()]
    return sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def check_views(state, frame):
    for pid, expected in enumerate(frame.get('views', [])):
        obs = observe(state, pid)
        arrays = [getattr(obs, name).ravel().tolist()
                  for name in ('visible', 'fog_obstacles', 'terrain', 'structure', 'owner', 'army')]
        assert sha256(json.dumps(arrays, separators=(',', ':')).encode()).hexdigest() == expected, (state.tick, pid)


def decode_actions(frame, state):
    actions = [Action.wait() for _ in state.alive]
    for pid, source, target, half in frame['actions']:
        delta = target - source
        direction = {-state.layout.width: 0, 1: 1, state.layout.width: 2, -1: 3}[delta]
        actions[pid] = Action.move(source, Direction(direction), MoveMode.HALF if half else MoveMode.ALL_BUT_ONE)
    return actions


@pytest.mark.parametrize('path', sorted((FIXTURES / 'replays').glob('*.json')), ids=lambda p: p.stem)
@pytest.mark.parametrize('backend', ['reference', 'cpu'])
def test_official_replay(path, backend):
    data = json.loads(path.read_text())
    state = replay_state(data)
    if backend == 'cpu':
        from generals.backends.cpu.engine import CPUEngine
        engine = CPUEngine(state)
    else:
        engine = CoreEngine(state, debug=True)
    for frame in data['ticks']:
        for event in data['afks']:
            if event['turn'] == state.tick:
                if state.alive[event['index']]:
                    engine.surrender(event['index'])
                else:
                    neutralize(state, event['index'])
                    if backend == 'cpu':
                        engine.invalidate_cache()
        # Official replay UI advances one render frame after a terminal surrender;
        # the live server ends immediately at that boundary.
        if state.terminated:
            break
        engine.step(decode_actions(frame, state))
        assert canonical_digest(state) == frame['digest'], f'{path.stem}: tick {state.tick}'
        check_views(state, frame)
    assert state.winner_id == data['winner']
    assert sorted(np.flatnonzero(~state.alive), key=lambda p: state.death_order[p]) == data['deaths']


@pytest.mark.parametrize('name', ['IZSpNMqIS', 'WtLgBl5yB', 'twdUV3dVM'])
def test_official_replay_cuda(name):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from generals.backends.gpu.env import GPUEnv
    data = json.loads((FIXTURES / 'replays' / (name + '.json')).read_text())
    state = replay_state(data)
    env = GPUEnv.from_states([state])
    try:
        for frame in data['ticks']:
            actions = decode_actions(frame, state)
            env.step(torch.tensor([[a.encode() for a in actions]], device=env.device))
            state = env.to_states()[0]
            assert canonical_digest(state) == frame['digest'], f'{name}: tick {state.tick}'
            check_views(state, frame)
    finally:
        env.close()


def test_official_city_placement():
    from generals.core.cities import _DuelCities
    from generals.core.mapgen import _neighbors
    class RandomStream:
        def __init__(self, seed): self.value = seed
        def random(self):
            self.value = (self.value * 1664525 + 1013904223) & 0xffffffff
            return self.value / 4294967296
        def integers(self, count): return int(self.random() * count)
    data = json.loads((FIXTURES / 'city_placement.json').read_text())
    mountains = np.zeros(data['width'] * data['height'], dtype=bool)
    mountains[data['mountains']] = True
    for case in data['cases']:
        planner = _DuelCities(mountains, data['generals'], _neighbors(data['width'], data['height']), RandomStream(case['seed']))
        actual = planner.plan(10)
        assert [group[0] for group in actual] == [group['indexes'] for group in case['result']['groups']]
        assert planner.deficit == case['result']['fairnessDeficit']


def test_official_ffa_city_placement_and_fallback():
    from generals.core.cities import place_cities
    from generals.core.mapgen import _neighbors

    class RandomStream:
        def __init__(self, seed): self.value = seed
        def random(self):
            self.value = (self.value * 1664525 + 1013904223) & 0xffffffff
            return self.value / 4294967296
        def integers(self, count): return int(self.random() * count)
        def shuffle(self, values):
            for remaining in range(len(values), 0, -1):
                index = self.integers(remaining)
                values[remaining - 1], values[index] = values[index], values[remaining - 1]

    data = json.loads((FIXTURES / 'ffa_cities.json').read_text())
    mountains = np.zeros(data['width'] * data['height'], dtype=bool)
    mountains[data['mountains']] = True
    for case in data['cases']:
        groups = place_cities(mountains, case['generals'], data['neighbors'],
                              RandomStream(case['seed']), case['count'], data['width'], data['height'])
        actual = [[cell, army] for cells, army in groups for cell in cells]
        assert actual == case['cities'], (len(case['generals']), case['count'], case['seed'])
