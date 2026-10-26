"""One trusted WAIT reference for scalar, vector and persistent competitions."""
from pathlib import Path


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


def create_wait_controller(*, player_id=0, seed=0):
    return WaitController()


def wait_spec():
    """Return the same policy identity for every Arena match mode."""
    from generals_arena.util import file_digest
    return {'name': 'WAIT', 'kind': 'custom', 'version': file_digest(Path(__file__)),
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
