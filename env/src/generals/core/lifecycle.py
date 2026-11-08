"""Device transfer adapter for infrequent external departure commands.

All departure rules execute in the native CPU kernel. Device tick kernels share
that same rules.h implementation; no Python lifecycle implementation exists.
"""
from generals.backends.cpu.env import CPUEnv


def surrender_batch(env, player_id, mask=None):
    import torch
    if mask is not None and (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.device != env.state['alive'].device or mask.shape != (env.batch_size,)):
        raise ValueError('surrender mask must be bool [batch] on the environment device')
    arrays = {name: value.detach().cpu().numpy() for name, value in env.state.items()}
    cpu = CPUEnv(arrays, width=env.width, height=env.height, episode_ticks=env.episode_ticks, threads=1)
    try:
        cpu.done[:] = env.done.detach().cpu().numpy()
        cpu.surrender(player_id, None if mask is None else mask.cpu().numpy())
        for name, value in env.state.items():
            value.copy_(torch.from_numpy(cpu.state[name]).to(value.device))
        env.done.copy_(torch.from_numpy(cpu.done).to(env.done.device))
        invalidate = getattr(env, 'invalidate_cache', None)
        if invalidate is not None:
            invalidate()
    finally:
        cpu.close()
