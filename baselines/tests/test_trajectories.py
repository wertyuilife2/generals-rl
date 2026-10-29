"""All four policies retain deterministic decisions across official backends."""
import pytest
from generals import MapConfig, make_engine
from generals.core.engine import CoreEngine
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.recording import state_digest
from generals.protocol import Action
from generals_baselines import AI_KINDS, make_controllers


@pytest.mark.parametrize("players", [2, 4, 8])
@pytest.mark.parametrize("seed", [38, 715])
def test_official_trajectories_match_reference(players, seed):
    state = generate_map(MapConfig(players=players), seed)
    engines = [CoreEngine(state.copy()), make_engine(state.copy())]
    kinds = [AI_KINDS[p % 4] for p in range(players)]
    policies = [make_controllers(kinds, seed) for _ in engines]
    try:
        for _ in range(250):
            if engines[0].state.terminated:
                break
            submitted = []
            for engine, controllers in zip(engines, policies):
                actions = [c.act(observe(engine.state, p)) if engine.state.alive[p] else Action.wait()
                           for p, c in enumerate(controllers)]
                submitted.append(actions)
                result = engine.step(actions)
                for move in result.moves:
                    controllers[move.player_id].on_result(move)
            assert submitted[0] == submitted[1]
            assert state_digest(engines[0].state) == state_digest(engines[1].state)
            for a, b in zip(*policies):
                assert a.rng.bit_generator.state == b.rng.bit_generator.state
    finally:
        engines[1].close()
        for controllers in policies:
            for controller in controllers:
                controller.close()
