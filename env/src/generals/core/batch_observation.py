"""Batched public observations; the same information boundary as observe()."""
import numpy as np


def public_obstacles(state, player_ids):
    """Public indistinguishable obstacle layer, before applying visibility."""
    gpu = not isinstance(state['owner'], np.ndarray)
    if gpu:
        import torch
        rows = torch.arange(state['owner'].shape[0], device=state['owner'].device)
    else:
        rows = np.arange(state['owner'].shape[0])
    obstacles = (state['terrain'] == 1) | (state['structure'] == 1)
    for slot in range(state['alive'].shape[1]):
        cell = state['spawn_pos'][:, slot]
        safe_cell = cell.clamp_min(0).long() if gpu else np.maximum(cell, 0)
        hidden = ((cell >= 0) & (state['general_pos'][:, slot] < 0)
                  & ((state['city_known'][rows, player_ids] & (1 << slot)) == 0))
        obstacles[rows, safe_cell] &= ~hidden
    return obstacles


def batch_observer(env, player_ids):
    """Bind fixed lane seats once; the returned callable keeps data on-device.

    Boards are [B, cells], stats [B, players, 3]. No enemy general positions or
    city totals are published. Arrays are independent of mutable backend state.
    This is game information, not policy features or an action mask.
    """
    gpu = not isinstance(env.state['owner'], np.ndarray)
    if gpu:
        import torch
        import torch.nn.functional as F
        host_ids = (player_ids.detach().cpu().numpy() if isinstance(player_ids, torch.Tensor)
                    else np.asarray(player_ids))
    else:
        host_ids = np.asarray(player_ids)
    if host_ids.ndim == 0:
        host_ids = np.full(env.num_envs, host_ids)
    if (host_ids.shape != (env.num_envs,) or host_ids.dtype.kind not in 'iu'
            or np.any(host_ids < 0) or np.any(host_ids >= env.players)):
        raise ValueError('one valid player id is required for every environment')
    if gpu:
        ids = torch.as_tensor(host_ids, device=env.state['owner'].device, dtype=torch.long)
        rows = torch.arange(env.num_envs, device=ids.device)
        public = torch.arange(env.players, device=ids.device)[None] == ids[:, None]
    else:
        ids = np.asarray(host_ids, dtype=np.int64).copy()
        rows = np.arange(env.num_envs)
        public = np.arange(env.players)[None] == ids[:, None]

    def observe():
        state = env.state
        obstacles = public_obstacles(state, ids)
        if gpu:
            owned = (state['owner'] == ids[:, None]).float().reshape(-1, 1, env.height, env.width)
            visible = F.max_pool2d(owned, 3, stride=1, padding=1).flatten(1).bool() | state["terminated"][:, None]
            boards = {key: torch.where(visible, state[key], fill)
                      for key, fill in (('terrain', 255), ('structure', 255), ('owner', -3), ('army', -1))}
            boards.update({key: state[key].clone() for key in ('alive', 'tick', 'terminated')})
            boards['winner_id'] = state['winner'].clone()
            boards['has_kill'] = state['kills'] > 0
            boards['warn_afk'] = state['warn_afk'].bool()
            boards['neutralize_in'] = torch.where((state['leave_tick'] >= 0) & (state['general_pos'] >= 0),
                ((state['tick'][:, None] - state['leave_tick']) / 50.).clamp(0, 1), -1.)
            boards['fog_obstacles'] = obstacles & ~visible
            stats = env.stats.clone()
            stats[..., 2] = torch.where(public | state["terminated"][:, None], stats[..., 2], -1)
            return {'state': boards, 'stats': stats, 'visible': visible, 'player_ids': ids.clone(),
                    'general_pos': state['general_pos'][rows, ids].clone(),
                    'active': ~env.done.any(dim=1) & state['alive'][rows, ids],
                    'width': env.width, 'height': env.height, 'visibility': 'local'}
        owned = (state['owner'] == ids[:, None]).reshape(-1, env.height, env.width)
        padded = np.pad(owned, ((0, 0), (1, 1), (1, 1)))
        visible = np.zeros_like(owned)
        for dy in range(3):
            for dx in range(3):
                visible |= padded[:, dy:dy+env.height, dx:dx+env.width]
        visible = visible.reshape(env.num_envs, -1) | state["terminated"][:, None]
        boards = {key: np.where(visible, state[key], fill).astype(state[key].dtype, copy=False)
                  for key, fill in (('terrain', 255), ('structure', 255), ('owner', -3), ('army', -1))}
        boards.update({key: state[key].copy() for key in ('alive', 'tick', 'terminated')})
        boards['winner_id'] = state['winner'].copy()
        boards['has_kill'] = state['kills'] > 0
        boards['warn_afk'] = state['warn_afk'].astype(bool)
        boards['neutralize_in'] = np.where((state['leave_tick'] >= 0) & (state['general_pos'] >= 0),
            np.clip((state['tick'][:, None] - state['leave_tick']) / 50., 0, 1), -1.)
        boards['fog_obstacles'] = obstacles & ~visible
        stats = env.stats.copy()
        stats[..., 2] = np.where(public | state["terminated"][:, None], stats[..., 2], -1)
        return {'state': boards, 'stats': stats, 'visible': visible, 'player_ids': ids.copy(),
                'general_pos': state['general_pos'][rows, ids].copy(),
                'active': ~env.done.any(axis=1) & state['alive'][rows, ids],
                'width': env.width, 'height': env.height, 'visibility': 'local'}
    return observe
