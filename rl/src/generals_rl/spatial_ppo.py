"""Spatial categorical actor and independent privileged critic for StudentPPO."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Categorical

from generals_rl.student_ppo import StudentPPOPolicy


MAP_PLANES, MAP_SIZE, PUBLIC_DIM, HISTORY = 12, 9, 13, 4
FRAME_DIM = MAP_PLANES * MAP_SIZE * MAP_SIZE + PUBLIC_DIM
NUM_ACTIONS = 1 + MAP_SIZE * MAP_SIZE * 8


class SpatialTrunk(nn.Module):
    def __init__(self, inputs, channels, blocks):
        super().__init__()
        self.projection = nn.Conv2d(inputs, channels, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                          nn.ELU(), nn.Conv2d(channels, channels, 1))
            for _ in range(blocks))

    def forward(self, maps):
        features = torch.nn.functional.elu(self.projection(maps))
        for block in self.blocks:
            features = torch.nn.functional.elu(features + block(features))
        return features


class SpatialActor(nn.Module):
    def __init__(self, channels, blocks, trunk_factory=SpatialTrunk):
        super().__init__()
        self.trunk = trunk_factory(HISTORY * MAP_PLANES, channels, blocks)
        self.context = nn.Sequential(nn.Linear(channels + PUBLIC_DIM, channels), nn.ELU())
        self.cell_head = nn.Conv2d(channels, 8, 1)
        self.wait_head = nn.Sequential(nn.Linear(2 * channels, channels), nn.ELU(), nn.Linear(channels, 1))

    def forward(self, history):
        frames = history.reshape(history.shape[0], HISTORY, FRAME_DIM)
        # History is oldest -> newest. Only the newest 13 public features enter
        # context; old public scalars and the redundant single_obs are ignored.
        maps = frames[..., :FRAME_DIM - PUBLIC_DIM].reshape(-1, HISTORY * MAP_PLANES, MAP_SIZE, MAP_SIZE)
        spatial = self.trunk(maps)
        pooled = spatial.mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, frames[:, -1, -PUBLIC_DIM:]), dim=-1))
        fused = torch.nn.functional.elu(spatial + context[..., None, None])
        cell_logits = self.cell_head(fused)
        wait = self.wait_head(torch.cat((pooled, context), dim=-1))
        # WAIT=0; move=1+source*8+direction*2+mode, source in row-major order.
        # A channel-major flatten here would silently scramble the action IDs.
        moves = cell_logits.permute(0, 2, 3, 1).reshape(history.shape[0], NUM_ACTIONS - 1)
        return torch.cat((wait, moves), dim=-1)


class PrivilegedCritic(nn.Module):
    """Its parameters/input are independent of the deployed local actor."""
    def __init__(self, channels, blocks, trunk_factory=SpatialTrunk):
        super().__init__()
        self.trunk = trunk_factory(MAP_PLANES, channels, blocks)
        self.context = nn.Sequential(nn.Linear(channels + PUBLIC_DIM, channels), nn.ELU())
        self.value_head = nn.Sequential(nn.Linear(2 * channels, channels), nn.ELU(), nn.Linear(channels, 1))

    def forward(self, privileged):
        maps = privileged[:, :-PUBLIC_DIM].reshape(-1, MAP_PLANES, MAP_SIZE, MAP_SIZE)
        pooled = self.trunk(maps).mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, privileged[:, -PUBLIC_DIM:]), dim=-1))
        return self.value_head(torch.cat((pooled, context), dim=-1))


class SpatialPPOPolicy(StudentPPOPolicy):
    """StudentPPO-compatible actor/value architecture without dense exact warm start.

    Inheritance only supplies the categorical lifecycle and the runner's type
    guards. The dense teacher/MoE constructor is never called. Observation
    normalization, BatchNorm, Dropout and recurrent state are unsupported here.
    """
    semantics = "spatial_p2_hist4_current_public13_v1"
    trunk_factory = staticmethod(SpatialTrunk)

    def __init__(self, obs, obs_groups, num_actions, *, spatial_channels=32,
                 residual_blocks=4, history_length=4, players=2, map_width=9,
                 map_height=9, map_planes=12, public_dim=13,
                 action_distribution="categorical", actor_obs_normalization=False,
                 critic_obs_normalization=False, value_potential_offset=False, **kwargs):
        nn.Module.__init__(self)
        if kwargs:
            raise ValueError(f"unsupported spatial policy options: {sorted(kwargs)}")
        if actor_obs_normalization is not False or critic_obs_normalization is not False:
            raise ValueError("spatial policy requires observation normalization disabled")
        if action_distribution != "categorical" or num_actions != NUM_ACTIONS:
            raise ValueError("spatial policy requires the 649-action categorical layout")
        geometry = (history_length, players, map_width, map_height, map_planes, public_dim)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in geometry) or geometry != (4, 2, 9, 9, 12, 13):
            raise ValueError("spatial policy supports only P2 9x9, H4, 12 map planes and 13 public features")
        for value in (spatial_channels, residual_blocks):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("spatial channels and residual blocks must be positive integers")
        if obs_groups != {"policy": ["policy"], "critic": ["critic"]}:
            raise ValueError("spatial policy requires exact local-policy/privileged-critic groups")
        for key, size in (("policy", HISTORY * FRAME_DIM), ("single_obs", FRAME_DIM), ("critic", FRAME_DIM)):
            if key not in obs or obs[key].ndim != 2 or obs[key].shape[-1] != size:
                raise ValueError(f"spatial policy requires {key} shape [B,{size}]")
        if not isinstance(value_potential_offset, bool):
            raise TypeError("value_potential_offset must be boolean")
        self.value_potential_offset = value_potential_offset
        self.obs_groups = deepcopy(obs_groups)
        self.num_actions, self.num_actor_obs, self.num_single_obs = NUM_ACTIONS, HISTORY * FRAME_DIM, FRAME_DIM
        self.num_critic_obs = FRAME_DIM
        self.action_distribution, self.state_dependent_std = "categorical", False
        self.actor_obs_normalization = self.critic_obs_normalization = False
        self.actor_obs_normalizer = nn.Identity()
        self.single_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()
        self.teacher_encoder = nn.Identity()  # No parameters, teacher lanes or teacher inference.
        self.actor = SpatialActor(spatial_channels, residual_blocks, self.trunk_factory)
        self.critic = PrivilegedCritic(spatial_channels, residual_blocks, self.trunk_factory)
        self.distribution = None

    def action_probabilities(self, obs, is_teacher=False):
        if is_teacher:
            raise ValueError("spatial student policy has no teacher inference branch")
        logits = self.actor(obs["policy"])
        mask = obs.get("action_mask")
        if mask is None:
            raise ValueError("categorical policies require action_mask")
        mask = mask.to(device=logits.device, dtype=torch.bool)
        if mask.shape != logits.shape or not mask.any(dim=-1).all():
            raise ValueError("action_mask must match logits and contain a legal action in every row")
        self.distribution = Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min), validate_args=False)
        return self.distribution.probs

    def act_inference(self, obs):
        return self.action_probabilities(obs).argmax(dim=-1, keepdim=True)

    def evaluate(self, obs, is_teacher=False, **kwargs):
        if obs.batch_size[0] == 0:
            return obs["critic"].new_empty((0, 1))
        if is_teacher:
            raise ValueError("spatial student policy has no nonempty teacher value branch")
        potential = None
        if self.value_potential_offset:
            potential = obs.get("critic_potential")
            if potential is None or potential.shape != (obs.batch_size[0], 1):
                raise ValueError("value_potential_offset requires obs['critic_potential'] with shape [N, 1]")
        value = self.critic(obs["critic"])
        return value if potential is None else value - potential.detach()


class LightSpatialPPOPolicy(SpatialPPOPolicy):
    """Two standard convolutions with optional dilation in the second layer."""
    semantics = "light_spatial_p2_hist4_current_public13_standard2_v1"

    def __init__(self, obs, obs_groups, num_actions, *, spatial_channels=16,
                 spatial_dilation=1, **kwargs):
        if "residual_blocks" in kwargs:
            raise ValueError("light spatial policy uses two standard convolutions, no residual_blocks option")
        if not isinstance(spatial_dilation, int) or isinstance(spatial_dilation, bool) or spatial_dilation < 1:
            raise ValueError("spatial_dilation must be a positive integer")
        self.spatial_dilation = spatial_dilation
        if spatial_dilation != 1:
            self.semantics = f"light_spatial_p2_hist4_current_public13_standard2_dilation{spatial_dilation}_v1"
        super().__init__(obs, obs_groups, num_actions, spatial_channels=spatial_channels,
                         residual_blocks=1, **kwargs)

    def trunk_factory(self, inputs, channels, _residual_blocks):
        return nn.Sequential(nn.Conv2d(inputs, channels, 3, padding=1), nn.ELU(),
                             nn.Conv2d(channels, channels, 3, padding=self.spatial_dilation,
                                       dilation=self.spatial_dilation), nn.ELU())


def policy_config(*, value_potential_offset=False, light_cnn=False):
    if not isinstance(light_cnn, bool):
        raise TypeError("light_cnn must be boolean")
    config = {"class_name": "generals_rl.spatial_ppo.SpatialPPOPolicy",
            "action_distribution": "categorical", "spatial_channels": 32, "residual_blocks": 4,
            "history_length": 4, "players": 2, "map_width": 9, "map_height": 9,
            "map_planes": 12, "public_dim": 13,
            "actor_obs_normalization": False, "critic_obs_normalization": False,
            "value_potential_offset": value_potential_offset}
    if light_cnn:
        config.update(class_name="generals_rl.spatial_ppo.LightSpatialPPOPolicy", spatial_channels=16)
        config.pop("residual_blocks")
    return config
