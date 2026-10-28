"""Batched external departures; ordinary tick rules stay in the native kernels."""
import numpy as np

from generals.protocol.config import _integer


def surrender_batch(env, player_id, mask=None):
    """Apply one seat's departure at the current boundary, on the owning device.

    Explicit departures do not advance time or auto-reset. The next step handles
    tick-zero departures and the 50-tick neutralization grace period.
    """
    player_id = _integer('player_id', player_id, 0)
    if player_id >= env.players:
        raise ValueError('invalid player id')
    state = env.state
    if isinstance(state['alive'], np.ndarray):
        xp = np
        cast = lambda value, dtype: value.astype(dtype)
        valid_mask = mask is None or isinstance(mask, np.ndarray) and mask.dtype == bool
    else:
        import torch
        xp = torch
        cast = lambda value, dtype: value.to(dtype)
        valid_mask = mask is None or (isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
                                      and mask.device == state['alive'].device)
    if not valid_mask or mask is not None and mask.shape != (env.batch_size,):
        raise ValueError('surrender mask must be bool [batch] on the environment device')
    active = state['alive'][:, player_id] & ~state['terminated'] & (state['leave_tick'][:, player_id] < 0)
    if env.episode_ticks:
        active &= state['tick'] < env.episode_ticks
    if mask is not None:
        active &= mask
    state['leave_tick'][:, player_id] = xp.where(active, xp.maximum(state['tick'], xp.ones_like(state['tick'])),
                                                state['leave_tick'][:, player_id])
    departed = active & (state['tick'] > 0)
    state['death_order'][:, player_id] = xp.where(departed, xp.amax(state['death_order'], axis=1) + 1,
                                                 state['death_order'][:, player_id])
    state['alive'][:, player_id] &= ~departed
    ended = departed & (state['alive'].sum(axis=1) == 1)
    winner = xp.argmax(cast(state['alive'], xp.int32), axis=1)
    state['terminated'] |= ended
    state['winner'][...] = xp.where(ended, winner, state['winner'])
    state['end_reason'][...] = xp.where(ended, 2, state['end_reason'])
    for seat in range(env.players):
        credit = ended & (winner == seat) & ((state['capture_mask'][:, seat] & (1 << player_id)) == 0)
        state['kills'][:, seat] += cast(credit, xp.int32)
        state['capture_mask'][:, seat] |= cast(credit, xp.uint8) << player_id
    env.done[:, 0] |= ended
    env.done[:, 1] &= ~ended
    env.invalidate_cache()
