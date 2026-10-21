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


def _validate_history_length(history_length):
    if not isinstance(history_length, int) or isinstance(history_length, bool) or history_length < 1:
        raise ValueError("history_length must be a positive integer")


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
    def __init__(self, channels, blocks, trunk_factory=SpatialTrunk, *, history_length=HISTORY):
        super().__init__()
        _validate_history_length(history_length)
        self.history_length = history_length
        self.trunk = trunk_factory(history_length * MAP_PLANES, channels, blocks)
        self.knowledge_channels = 0
        self.context = nn.Sequential(nn.Linear(channels + PUBLIC_DIM, channels), nn.ELU())
        self.cell_head = nn.Conv2d(channels, 8, 1)
        self.wait_head = nn.Sequential(nn.Linear(2 * channels, channels), nn.ELU(), nn.Linear(channels, 1))

    def forward(self, history, knowledge=None):
        if history.ndim != 2 or history.shape[-1] != self.history_length * FRAME_DIM:
            raise ValueError("actor history shape does not match history_length")
        frames = history.reshape(history.shape[0], self.history_length, FRAME_DIM)
        # History is oldest -> newest. Only the newest 13 public features enter
        # context; old public scalars and the redundant single_obs are ignored.
        maps = frames[..., :FRAME_DIM - PUBLIC_DIM].reshape(-1, self.history_length * MAP_PLANES, MAP_SIZE, MAP_SIZE)
        if self.knowledge_channels:
            if knowledge is None or knowledge.shape != (history.shape[0], self.knowledge_channels * MAP_SIZE * MAP_SIZE):
                raise ValueError("actor requires the configured non-stacked knowledge planes")
            maps = torch.cat((maps, knowledge.reshape(-1, self.knowledge_channels, MAP_SIZE, MAP_SIZE)), 1)
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
                 critic_obs_normalization=False, value_potential_offset=False,
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
        if not isinstance(value_potential_offset, bool):
            raise TypeError("value_potential_offset must be boolean")
        if value_potential_offset:
            raise ValueError("analytic value_potential_offset is retired; expected False")
        if not isinstance(wait_move_factorization, bool):
            raise TypeError("wait_move_factorization must be boolean")
        self.value_potential_offset = value_potential_offset
        self.wait_move_factorization = wait_move_factorization
        self.history_length = history_length
        self.semantics = self.semantics.replace("hist4", f"hist{history_length}")
        self.obs_groups = deepcopy(obs_groups)
        self.num_actions, self.num_actor_obs, self.num_single_obs = NUM_ACTIONS, history_length * FRAME_DIM, FRAME_DIM
        self.num_critic_obs = FRAME_DIM
        self.action_distribution, self.state_dependent_std = "categorical", False
        self.actor_obs_normalization = self.critic_obs_normalization = False
        self.actor_obs_normalizer = nn.Identity()
        self.single_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()
        self.teacher_encoder = nn.Identity()  # No parameters, teacher lanes or teacher inference.
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
        """One action boundary shared by legacy CNNs and dynamic-map heads."""
        if mask is None:
            raise ValueError("categorical policies require action_mask")
        mask = mask.to(device=logits.device, dtype=torch.bool)
        if mask.shape != logits.shape or (validate_mask and not mask.any(dim=-1).all()):
            raise ValueError("action_mask must match logits and contain a legal action in every row")
        if self.wait_move_factorization:
            has_move = mask[:, 1:].any(dim=-1, keepdim=True)
            moves = logits[:, 1:].masked_fill(~mask[:, 1:], -torch.inf)
            # Avoid log_softmax(all -inf), including its backward path.
            moves = torch.where(has_move, moves, torch.zeros_like(moves))
            log_moves = torch.nn.functional.log_softmax(moves, dim=-1)
            wait = logits[:, :1]
            log_wait = torch.where(has_move, torch.nn.functional.logsigmoid(wait),
                                   torch.zeros_like(wait))
            log_move_gate = torch.where(mask[:, :1], torch.nn.functional.logsigmoid(-wait),
                                        torch.zeros_like(wait))
            # Joint log probabilities retain the original Categorical lifecycle.
            logits = torch.cat((log_wait, log_move_gate + log_moves), dim=-1)
        self.distribution = Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min), validate_args=False)
        return self.distribution.probs

    def act_inference(self, obs):
        probabilities = self.action_probabilities(obs)
        if not self.wait_move_factorization:
            return probabilities.argmax(dim=-1, keepdim=True)
        choose_wait = probabilities[:, :1] >= probabilities[:, 1:].sum(dim=-1, keepdim=True)
        move = probabilities[:, 1:].argmax(dim=-1, keepdim=True) + 1
        return torch.where(choose_wait, torch.zeros_like(move), move)

    def evaluate(self, obs, is_teacher=False, **kwargs):
        if obs.batch_size[0] == 0:
            return obs["critic"].new_empty((0, 1))
        if is_teacher:
            raise ValueError("spatial student policy has no nonempty teacher value branch")
        return self.critic(obs["critic"])


class OwnKingRelativeGeometry(nn.Module):
    """Append relative coordinates of the latest locally observed own general."""

    def __init__(self, history_length=HISTORY):
        super().__init__()
        _validate_history_length(history_length)
        self.history_length = history_length
        self.knowledge_channels = 0
        self.register_buffer("row", torch.arange(MAP_SIZE, dtype=torch.float32)[:, None]
                             .expand(MAP_SIZE, MAP_SIZE), persistent=False)
        self.register_buffer("column", torch.arange(MAP_SIZE, dtype=torch.float32)[None, :]
                             .expand(MAP_SIZE, MAP_SIZE), persistent=False)

    def forward(self, maps):
        if maps.ndim != 4 or maps.shape[1:] != (self.history_length * MAP_PLANES + self.knowledge_channels, MAP_SIZE, MAP_SIZE):
            raise ValueError("own-king geometry expects history maps plus configured knowledge planes")
        if not maps.is_floating_point():
            raise TypeError("own-king geometry expects floating-point map features")
        current = maps[:, (self.history_length - 1) * MAP_PLANES:self.history_length * MAP_PLANES]
        own_king = (current[:, 4].bool() & current[:, 5].bool()).flatten(1)
        valid = own_king.sum(1).eq(1)
        king = own_king.long().argmax(1)
        dx = (self.column.to(dtype=maps.dtype) - (king % MAP_SIZE)[:, None, None]) / (MAP_SIZE - 1)
        dy = (self.row.to(dtype=maps.dtype) - (king // MAP_SIZE)[:, None, None]) / (MAP_SIZE - 1)
        geometry = torch.where(valid[:, None, None, None], torch.stack((dx, dy), 1), 0)
        return torch.cat((maps, geometry), 1)


class LightSpatialPPOPolicy(SpatialPPOPolicy):
    """Two standard convolutions, with optional own-general coordinates."""
    semantics = "light_spatial_p2_hist4_current_public13_standard2_v1"

    def __init__(self, obs, obs_groups, num_actions, *, spatial_channels=16,
                 spatial_dilation=1, own_king_geometry=False, own_king_context=False, **kwargs):
        if "residual_blocks" in kwargs:
            raise ValueError("light spatial policy uses two standard convolutions, no residual_blocks option")
        # Retain default-valued keys carried by v32 checkpoint metadata.
        if not isinstance(spatial_dilation, int) or isinstance(spatial_dilation, bool) or spatial_dilation != 1:
            raise ValueError("spatial_dilation experiments are retired; expected 1")
        if not isinstance(own_king_geometry, bool):
            raise TypeError("own_king_geometry must be boolean")
        if own_king_context is not False:
            raise ValueError("own_king_context experiments are retired; expected False")
        self.own_king_geometry = own_king_geometry
        super().__init__(obs, obs_groups, num_actions, spatial_channels=spatial_channels,
                         residual_blocks=1, **kwargs)
        if own_king_geometry:
            # Construct the baseline first to preserve its independent critic's
            # seeded initialization. Only this new actor input layer is replaced.
            first, *layers = self.actor.trunk.children()
            replacement = nn.Conv2d(first.in_channels + 2, first.out_channels, first.kernel_size,
                                    stride=first.stride, padding=first.padding,
                                    device=first.weight.device, dtype=first.weight.dtype)
            self.actor.trunk = nn.Sequential(OwnKingRelativeGeometry(self.history_length), replacement, *layers)
            self.semantics += "_own_king_relative_xy_v1"

    def trunk_factory(self, inputs, channels, _residual_blocks):
        return nn.Sequential(nn.Conv2d(inputs, channels, 3, padding=1), nn.ELU(),
                             nn.Conv2d(channels, channels, 3, padding=1), nn.ELU())

