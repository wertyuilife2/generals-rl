"""Shared visibility-masked features and episode-local history."""
import torch
import torch.nn.functional as F

from .state import GameBatch
from .features import feature_recipe


class LocalObservationEncoder:
    semantics = "official_31_4_5_local_cells13_v2"
    channels = ("visible", "plain", "mountain", "city", "general", "self",
                "enemy", "empty", "neutral", "log_army")

    def __init__(self, episode_ticks: int, *, clock_scale_ticks=None):
        self.episode_ticks = episode_ticks
        self.clock_scale_ticks = clock_scale_ticks
        if clock_scale_ticks is not None:
            if type(clock_scale_ticks) is not int or clock_scale_ticks < 1:
                raise ValueError('clock_scale_ticks must be a positive integer')
            self.semantics += '_fixed_bounded_clock_v1'

    def visibility(self, batch: GameBatch, player_ids: torch.Tensor):
        owned = (batch.state["owner"] == player_ids[:, None]).float()
        visible = F.max_pool2d(owned.reshape(-1, 1, batch.height, batch.width),
                            kernel_size=3, stride=1, padding=1).reshape(batch.num_envs, -1).bool()
        return (visible | batch.state["terminated"][:, None]) & batch.valid_cells

    def encode(self, batch: GameBatch, player_ids: torch.Tensor, *, visible=None, hide_enemy_cities=True):
        s, stats = batch.state, batch.stats
        visible = self.visibility(batch, player_ids) if visible is None else visible & batch.valid_cells
        owner, terrain, structure = s["owner"], s["terrain"], s["structure"]
        own = owner == player_ids[:, None]
        # Hidden ownership/army/type remain unknown; only the official combined
        # obstacle hint and the public board extent are available through fog.
        maps = torch.stack((visible, visible & (terrain == 0), visible & (terrain == 1),
                            visible & (structure == 1), visible & (structure == 2),
                            visible & own, visible & (owner >= 0) & ~own,
                            visible & (owner == -1), visible & (owner == -2)), 1).float()
        army = torch.log1p(s["army"].float()) / 10.0 * visible
        order = (player_ids[:, None] + torch.arange(batch.players, device=batch.device)) % batch.players
        # Relative identity channels retain the owner of visible enemy cells,
        # including in multiplayer games. Hidden identities stay zero.
        identities = (owner[:, None] == order[:, :, None]) & visible[:, None]
        maps = torch.cat((maps, army[:, None], identities.float(),
                          batch.fog_obstacles(player_ids, visible)[:, None].float()), 1).flatten(1)
        # Relative player order starts at the learner; all army/territory/alive
        # totals are public. Opponent city counts are unknown in local mode.
        relative = stats.gather(1, order[..., None].expand(-1, -1, 3)).float()
        cells = batch.valid_cells.sum(1).clamp_min(1)[:, None]
        city = relative[..., 2] / cells
        if hide_enemy_cities:
            city = torch.cat((city[:, :1], torch.where(s['terminated'][:, None], city[:, 1:],
                                                      -torch.ones_like(city[:, 1:]))), 1)
        alive = s["alive"].gather(1, order).float()
        public = torch.stack((torch.log1p(relative[..., 0]) / 10.0,
                              relative[..., 1] / cells, city, alive), -1).flatten(1)
        tick = s["tick"].float()
        elapsed = (tick / self.episode_ticks if self.clock_scale_ticks is None
                   else (tick / self.clock_scale_ticks).clamp(0, 1))
        clock = torch.stack((elapsed, (tick % 2) / 2, (tick % 50) / 50), -1)
        # Seat and clock expose the alternating base tie-break priority.
        seat = F.one_hot(player_ids, num_classes=batch.players).float()
        return torch.cat((maps, public, clock, seat), -1)


def make_observation_encoder(config, *, local_encoder=LocalObservationEncoder, combat_encoder=None, king_encoder=None):
    """Use one configured feature layout for learner, opponents and evaluation."""
    clock = getattr(config, 'clock_scale_ticks', None)
    options = {} if clock is None else {'clock_scale_ticks': clock}
    if config.combat_features:
        from .combat import CombatObservationEncoder
        encoder = (combat_encoder or CombatObservationEncoder)(config.max_episode_steps, **options)
    else:
        encoder = local_encoder(config.max_episode_steps, **options)
    encoder = (king_encoder or KingThreatDistanceEncoder)(encoder) if config.king_threat_distance else encoder
    return CompactPrecisionEncoder(encoder) if feature_recipe(getattr(config, "feature_set", "stacked")).half_precision else encoder


class CompactPrecisionEncoder:
    """Quantize normalized features before both rollout and stored PPO replay."""
    def __init__(self, base):
        self.base, self.channels = base, base.channels
        self.semantics = base.semantics + "_fp16_transport_v1"

    def visibility(self, batch, player_ids):
        return self.base.visibility(batch, player_ids)

    def encode(self, batch, player_ids, **options):
        return self.base.encode(batch, player_ids, **options).half().float()


class KingThreatDistanceEncoder:
    """Replace redundant seat-P1 with normalized nearest mobile enemy distance.

    Use current source armies at the decision boundary. This is geometric
    potential only: ignores walls, routes, joint moves and absolute seat order.
    Missing/dead/ambiguous own general and no mobile enemy use distance=1.
    """
    def __init__(self, base):
        self.base = base
        self.channels = base.channels
        self.semantics = base.semantics + "_king_threat_distance_v1"

    def visibility(self, batch, player_ids):
        return self.base.visibility(batch, player_ids)

    def encode(self, batch, player_ids, **visibility_options):
        if batch.players != 2:
            raise ValueError("king threat distance requires two players")
        result = self.base.encode(batch, player_ids, **visibility_options)
        cells_count = batch.width * batch.height
        planes = result[:, :12 * cells_count].reshape(batch.num_envs, 12, cells_count)
        visible, own, enemy = (planes[:, index].bool() for index in (0, 5, 6))
        own_general = planes[:, 4].bool() & own
        king = own_general.long().argmax(1)
        valid = own_general.sum(1).eq(1) & batch.state["alive"].gather(1, player_ids[:, None]).squeeze(1)
        # Mask raw hidden army before comparisons, and avoid int64 addition.
        army = torch.where(visible, batch.state["army"], 0)
        mobile = enemy & (army > 1)
        mobile &= batch.state["alive"].gather(1, (1 - player_ids)[:, None])
        cells = torch.arange(cells_count, device=batch.device)
        distance = ((cells[None] // batch.width - king[:, None] // batch.width).abs()
                    + (cells[None] % batch.width - king[:, None] % batch.width).abs()).float() / max(
                        1, batch.width + batch.height - 2)
        nearest = distance.masked_fill(~mobile, 1).amin(1)
        result[:, -1] = torch.where(valid, nearest, 1)
        return result


class ObservationHistory:
    def __init__(self, length: int):
        if length < 1:
            raise ValueError("history length must be positive")
        self.length = length
        self.frames = None

    def reset(self, current, mask=None, *, rows=None):
        repeated = current[:, None].expand(-1, self.length, -1)
        if rows is not None:
            if mask is not None:
                raise ValueError('choose reset rows or a reset mask')
            self.frames = self.frames.index_copy(0, rows, repeated)
        elif self.frames is None or mask is None:
            self.frames = repeated.clone()
        else:
            # Copy-on-write preserves terminal and pre-step observation views.
            self.frames = torch.where(mask[:, None, None], repeated, self.frames)

    def push(self, current, *, reset_rows=None):
        frames = torch.cat((self.frames[:, 1:], current[:, None]), 1)
        if reset_rows is not None:
            # Only the newly allocated history is mutable; prior observations
            # and stored timeout snapshots continue to own their old frames.
            frames[reset_rows] = current[reset_rows, None]
        self.frames = frames

    def flattened(self):
        return self.frames.flatten(1).clone()
