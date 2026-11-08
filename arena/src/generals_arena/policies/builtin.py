"""One trusted WAIT reference for scalar, vector and persistent competitions."""
from pathlib import Path
from contextlib import contextmanager
from functools import lru_cache


@lru_cache(maxsize=1)
def _native_library():
    from generals.backends._build import build_shared
    from generals.protocol.native import INCLUDE
    return build_shared(Path(__file__).with_name('wait.cpp'), name='arena-wait', include_dirs=[INCLUDE])[0]


class WaitController:
    """Trusted WAIT policy with the ordinary scalar and raw-batch protocol."""
    def reset(self, player_id, seed):
        pass

    def act(self, observation):
        from generals import Action
        return Action.wait()

    def on_result(self, result):
        pass

    def reset_batch(self, player_ids, seeds):
        pass

    def act_batch(self, raw):
        active = raw['active']
        if hasattr(active, 'device'):
            import torch
            return torch.zeros(active.shape, dtype=torch.int64, device=active.device)
        import numpy as np
        return np.zeros(active.shape, dtype=np.int64)

    @contextmanager
    def native_batch(self, height, width, players):
        from generals.protocol.native import ABI_VERSION, PolicyABI, PolicyBinding, address
        library = _native_library()
        yield PolicyBinding(PolicyABI(ABI_VERSION, None, address(library.arena_wait_act), None, address(library.arena_wait_one)), library, None)


def create_wait_controller(*, player_id=0, seed=0):
    return WaitController()


def wait_spec():
    """Return the same policy identity for every Arena match mode."""
    from generals_arena.util import digest, file_digest
    version = digest({name: file_digest(Path(__file__).with_name(name)) for name in ('builtin.py', 'wait.cpp')})
    return {'name': 'WAIT', 'kind': 'custom', 'version': version,
            'factory': 'generals_arena.policies.builtin:create_wait_controller'}


def with_wait_reference(specs, players):
    """Add the actual WAIT policy to a duel roster; reserve its reference name."""
    specs = list(specs)
    if players == 2:
        wait = wait_spec()
        named = [spec for spec in specs if spec['name'] == 'WAIT']
        if any(spec != wait for spec in named):
            raise ValueError('WAIT is reserved for the built-in display reference')
        if not named:
            specs.append(wait)
    return specs
