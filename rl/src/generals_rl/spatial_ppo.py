"""Spatial categorical actor and independent privileged critic for StudentPPO."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .actor_critic import CategoricalActorCritic
from .policy.cnn import (MAP_PLANES, MAP_SIZE, PUBLIC_DIM, FRAME_DIM, NUM_ACTIONS,
                         _validate_history_length, SpatialTrunk, SpatialActor,
                         light_trunk, add_own_king_geometry)
from .policy.distributions import make_distribution, greedy_actions

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

class SpatialPPOPolicy(CategoricalActorCritic):
    """StudentPPO-compatible actor/value architecture without dense exact warm start.

    The base supplies sampling, log probabilities and the runner lifecycle.
    Observation normalization, BatchNorm, Dropout and recurrence are unsupported.
    """
    semantics = "spatial_p2_hist4_current_public13_v1"
    trunk_factory = staticmethod(SpatialTrunk)

    def __init__(self, obs, obs_groups, num_actions, *, spatial_channels=32,
                 residual_blocks=4, history_length=4, players=2, map_width=9,
                 map_height=9, map_planes=12, public_dim=13,
                 action_distribution="categorical", actor_obs_normalization=False,
                 critic_obs_normalization=False,
                 wait_move_factorization=False, **kwargs):
        nn.Module.__init__(self)
        if kwargs:
            raise ValueError(f"unsupported spatial policy options: {sorted(kwargs)}")
        if actor_obs_normalization is not False or critic_obs_normalization is not False:
            raise ValueError("spatial policy requires observation normalization disabled")
        if action_distribution != "categorical" or num_actions != NUM_ACTIONS:
            raise ValueError("spatial policy requires the 649-action categorical layout")
        _validate_history_length(history_length)
        geometry = (players, map_width, map_height, map_planes, public_dim)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in geometry) or geometry != (2, 9, 9, 12, 13):
            raise ValueError("spatial policy supports only P2 9x9, 12 map planes and 13 public features")
        for value in (spatial_channels, residual_blocks):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("spatial channels and residual blocks must be positive integers")
        if obs_groups != {"policy": ["policy"], "critic": ["critic"]}:
            raise ValueError("spatial policy requires exact local-policy/privileged-critic groups")
        for key, size in (("policy", history_length * FRAME_DIM), ("single_obs", FRAME_DIM), ("critic", FRAME_DIM)):
            if key not in obs or obs[key].ndim != 2 or obs[key].shape[-1] != size:
                raise ValueError(f"spatial policy requires {key} shape [B,{size}]")
        if not isinstance(wait_move_factorization, bool):
            raise TypeError("wait_move_factorization must be boolean")
        self.wait_move_factorization = wait_move_factorization
        self.history_length = history_length
        self.semantics = self.semantics.replace("hist4", f"hist{history_length}")
        self.obs_groups = deepcopy(obs_groups)
        self.num_actions, self.num_actor_obs, self.num_single_obs = NUM_ACTIONS, history_length * FRAME_DIM, FRAME_DIM
        self.num_critic_obs = FRAME_DIM
        self.action_distribution, self.state_dependent_std = "categorical", False
        self.actor_obs_normalization = self.critic_obs_normalization = False
        self.actor = SpatialActor(spatial_channels, residual_blocks, self.trunk_factory,
                                  history_length=history_length)
        self.critic = PrivilegedCritic(spatial_channels, residual_blocks, self.trunk_factory)
        self.distribution = None

    def action_probabilities(self, obs, is_teacher=False, *, validate_mask=True):
        if is_teacher:
            raise ValueError("spatial student policy has no teacher inference branch")
        logits = (self.actor(obs["policy"], obs.get("knowledge"))
                  if getattr(self.actor, "knowledge_channels", 0) else self.actor(obs["policy"]))
        return self._probabilities_from_logits(logits, obs.get("action_mask"), validate_mask=validate_mask)

    def validate_action_mask(self, obs):
        """Check a complete logical PPO batch before splitting its forwards."""
        mask = obs.get("action_mask")
        if mask is None:
            raise ValueError("categorical policies require action_mask")
        mask = mask.to(device=obs["policy"].device, dtype=torch.bool)
        if mask.shape != (len(obs["policy"]), self.num_actions) or not mask.any(dim=-1).all():
            raise ValueError("action_mask must match logits and contain a legal action in every row")

    def _probabilities_from_logits(self, logits, mask, *, validate_mask=True):
        self.distribution = make_distribution(logits, mask, factorized=self.wait_move_factorization,
                                               validate_mask=validate_mask)
        return self.distribution.probs

    def act_inference(self, obs):
        return greedy_actions(self.action_probabilities(obs), factorized=self.wait_move_factorization)

    def evaluate(self, obs, is_teacher=False, **kwargs):
        if obs.batch_size[0] == 0:
            return obs["critic"].new_empty((0, 1))
        if is_teacher:
            raise ValueError("spatial student policy has no nonempty teacher value branch")
        return self.critic(obs["critic"])

class LightSpatialPPOPolicy(SpatialPPOPolicy):
    """Two standard convolutions, with optional own-general coordinates."""
    semantics = "light_spatial_p2_hist4_current_public13_standard2_v1"

    def __init__(self, obs, obs_groups, num_actions, *, spatial_channels=16,
                 own_king_geometry=False, **kwargs):
        if "residual_blocks" in kwargs:
            raise ValueError("light spatial policy uses two standard convolutions, no residual_blocks option")
        if not isinstance(own_king_geometry, bool):
            raise TypeError("own_king_geometry must be boolean")
        self.own_king_geometry = own_king_geometry
        super().__init__(obs, obs_groups, num_actions, spatial_channels=spatial_channels,
                         residual_blocks=1, **kwargs)
        if own_king_geometry:
            # Construct the baseline first to preserve its independent critic's
            # seeded initialization. Only this new actor input layer is replaced.
            add_own_king_geometry(self.actor)
            self.semantics += "_own_king_relative_xy_v1"

    trunk_factory = staticmethod(light_trunk)

