"""Spatial categorical actor and independent privileged critic for StudentPPO."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .actor_critic import CategoricalActorCritic
from .policy.layout import ObservationLayout
from .policy.cnn import (_validate_history_length, SpatialTrunk, SpatialActor,
                         light_trunk, add_own_king_geometry)
from .policy.distributions import make_distribution, greedy_actions

class PrivilegedCritic(nn.Module):
    """Its parameters/input are independent of the deployed local actor."""
    def __init__(self, channels, blocks, trunk_factory=SpatialTrunk, *, layout):
        super().__init__()
        self.layout = layout
        self.trunk = trunk_factory(layout.map_planes, channels, blocks)
        self.context = nn.Sequential(nn.Linear(channels + layout.public_dim, channels), nn.ELU())
        self.value_head = nn.Sequential(nn.Linear(2 * channels, channels), nn.ELU(), nn.Linear(channels, 1))

    def forward(self, privileged):
        maps = self.layout.current_maps(privileged)
        pooled = self.trunk(maps).mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, privileged[:, -self.layout.public_dim:]), dim=-1))
        return self.value_head(torch.cat((pooled, context), dim=-1))

class SpatialPPOPolicy(CategoricalActorCritic):
    """StudentPPO-compatible actor/value architecture without dense exact warm start.

    The base supplies sampling, log probabilities and the runner lifecycle.
    Observation normalization, BatchNorm, Dropout and recurrence are unsupported.
    """
    semantics = "spatial_p2_hist4_current_public13_v1"
    trunk_factory = staticmethod(SpatialTrunk)

    def __init__(self, obs, obs_groups, num_actions, *, spatial_channels=32,
                 residual_blocks=4, history_length=4, players=2, map_width=23,
                 map_height=23, map_planes=14, public_dim=13,
                 action_distribution="categorical", actor_obs_normalization=False,
                 critic_obs_normalization=False,
                 wait_move_factorization=False, **kwargs):
        nn.Module.__init__(self)
        if kwargs:
            raise ValueError(f"unsupported spatial policy options: {sorted(kwargs)}")
        if actor_obs_normalization is not False or critic_obs_normalization is not False:
            raise ValueError("spatial policy requires observation normalization disabled")
        self.layout = ObservationLayout(map_width, map_height, history_length, map_planes, public_dim)
        if action_distribution != "categorical" or num_actions != self.layout.num_actions:
            raise ValueError("categorical action vocabulary must match board geometry")
        _validate_history_length(history_length)
        geometry = (players, map_width, map_height, map_planes, public_dim)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in geometry) or (players, map_planes, public_dim) != (2, 14, 13):
            raise ValueError("spatial policy requires P2, 14 map planes and 13 public features")
        for value in (spatial_channels, residual_blocks):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("spatial channels and residual blocks must be positive integers")
        if obs_groups != {"policy": ["policy"], "critic": ["critic"]}:
            raise ValueError("spatial policy requires exact local-policy/privileged-critic groups")
        for key, size in (("policy", history_length * self.layout.frame_dim), ("single_obs", self.layout.frame_dim), ("critic", self.layout.frame_dim)):
            if key not in obs or obs[key].ndim != 2 or obs[key].shape[-1] != size:
                raise ValueError(f"spatial policy requires {key} shape [B,{size}]")
        if not isinstance(wait_move_factorization, bool):
            raise TypeError("wait_move_factorization must be boolean")
        self.wait_move_factorization = wait_move_factorization
        self.history_length = history_length
        self.semantics = self.semantics.replace("hist4", f"hist{history_length}")
        self.obs_groups = deepcopy(obs_groups)
        self.num_actions, self.num_actor_obs, self.num_single_obs = self.layout.num_actions, history_length * self.layout.frame_dim, self.layout.frame_dim
        self.num_critic_obs = self.layout.frame_dim
        self.action_distribution, self.state_dependent_std = "categorical", False
        self.actor_obs_normalization = self.critic_obs_normalization = False
        self.actor = SpatialActor(spatial_channels, residual_blocks, self.trunk_factory,
                                  layout=self.layout)
        self.critic = PrivilegedCritic(spatial_channels, residual_blocks, self.trunk_factory, layout=self.layout)
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

