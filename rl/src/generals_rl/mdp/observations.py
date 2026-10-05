"""Fog-aware features, privileged features, and episode-local history."""
import torch
import torch.nn.functional as F

from .state import GameBatch


class LocalObservationEncoder:
    channels = ("visible", "plain", "mountain", "city", "general", "self",
                "enemy", "empty", "neutral", "log_army")

    def __init__(self, episode_ticks: int):
        self.episode_ticks = episode_ticks

    def visibility(self, batch: GameBatch, player_ids: torch.Tensor):
        owned = (batch.state["owner"] == player_ids[:, None]).float()
        return F.max_pool2d(owned.reshape(-1, 1, batch.height, batch.width),
                            kernel_size=3, stride=1, padding=1).reshape(batch.num_envs, -1).bool()

    def encode(self, batch: GameBatch, player_ids: torch.Tensor, privileged=False):
        s, stats = batch.state, batch.stats
        visible = torch.ones_like(s["owner"], dtype=torch.bool) if privileged else self.visibility(batch, player_ids)
        owner, terrain, structure = s["owner"], s["terrain"], s["structure"]
        own = owner == player_ids[:, None]
        # Every hidden cell feature is zero, including army and structure.
        maps = torch.stack((visible, visible & (terrain == 0), visible & (terrain == 1),
                            visible & (structure == 1), visible & (structure == 2),
                            visible & own, visible & (owner >= 0) & ~own,
                            visible & (owner == -1), visible & (owner == -2)), 1).float()
        army = torch.log1p(s["army"].float()) / 10.0 * visible
        order = (player_ids[:, None] + torch.arange(batch.players, device=batch.device)) % batch.players
        # Relative identity channels retain the owner of visible enemy cells,
        # including in multiplayer games. Hidden identities stay zero.
        identities = (owner[:, None] == order[:, :, None]) & visible[:, None]
        maps = torch.cat((maps, army[:, None], identities.float()), 1).flatten(1)
        # Relative player order starts at the learner; all army/territory/alive
        # totals are public. Opponent city counts are unknown in local mode.
        relative = stats.gather(1, order[..., None].expand(-1, -1, 3)).float()
        cells = batch.width * batch.height
        city = relative[..., 2] / cells
        if not privileged:
            city = torch.cat((city[:, :1], -torch.ones_like(city[:, 1:])), 1)
        alive = s["alive"].gather(1, order).float()
        public = torch.stack((torch.log1p(relative[..., 0]) / 10.0,
                              relative[..., 1] / cells, city, alive), -1).flatten(1)
        tick = s["tick"].float()
        clock = torch.stack((tick / self.episode_ticks, (tick % 2) / 2, (tick % 50) / 50), -1)
        # Rules execute absolute seat 0 before seat 1 etc.; retain the learner's
        # seat so rotating seats does not hide this public ordering information.
        seat = F.one_hot(player_ids, num_classes=batch.players).float()
        return torch.cat((maps, public, clock, seat), -1)


def make_observation_encoder(config):
    """Use one configured feature layout for learner, opponents and evaluation."""
    if config.combat_features:
        from .combat import CombatObservationEncoder
        encoder = CombatObservationEncoder(config.max_episode_steps)
    else:
        encoder = LocalObservationEncoder(config.max_episode_steps)
    return KingThreatDistanceEncoder(encoder) if config.king_threat_distance else encoder


class KingThreatDistanceEncoder:
    """Replace redundant seat-P1 with nearest observable mobile enemy / 16.

    Count public next-tick growth, not hypothetical actions. This is geometric
    potential only: ignores walls, routes, joint moves and absolute seat order.
    Missing/dead/ambiguous own general and no mobile enemy use distance=1.
    """
    def __init__(self, base):
        self.base = base
        self.channels = base.channels
        self.semantics = getattr(base, "semantics", "local_p2_985_v1") + "_king_threat_distance_v1"

    def visibility(self, batch, player_ids):
        return self.base.visibility(batch, player_ids)

    def encode(self, batch, player_ids, privileged=False):
        if (batch.players, batch.width, batch.height) != (2, 9, 9):
            raise ValueError("king threat distance requires two players on a 9x9 map")
        result = self.base.encode(batch, player_ids, privileged=privileged)
        planes = result[:, :972].reshape(batch.num_envs, 12, 81)
        visible, own, enemy = (planes[:, index].bool() for index in (0, 5, 6))
        own_general = planes[:, 4].bool() & own
        king = own_general.long().argmax(1)
        valid = own_general.sum(1).eq(1) & batch.state["alive"].gather(1, player_ids[:, None]).squeeze(1)
        # Mask raw hidden army before comparisons, and avoid int64 addition.
        army = torch.where(visible, batch.state["army"], 0)
        tick = batch.state["tick"][:, None] + 1
        grows = ((tick % 2 == 0) & (planes[:, 3].bool() | planes[:, 4].bool())) | (tick % 50 == 0)
        mobile = enemy & ((army > 1) | ((army == 1) & grows))
        mobile &= batch.state["alive"].gather(1, (1 - player_ids)[:, None])
        cells = torch.arange(81, device=batch.device)
        distance = ((cells[None] // 9 - king[:, None] // 9).abs()
                    + (cells[None] % 9 - king[:, None] % 9).abs()).float() / 16
        nearest = distance.masked_fill(~mobile, 1).amin(1)
        result[:, -1] = torch.where(valid, nearest, 1)
        return result


class ObservationHistory:
    def __init__(self, length: int):
        if length < 1:
            raise ValueError("history length must be positive")
        self.length = length
        self.frames = None

    def reset(self, current, mask=None):
        repeated = current[:, None].expand(-1, self.length, -1)
        if self.frames is None or mask is None:
            self.frames = repeated.clone()
        else:
            self.frames[mask] = repeated[mask]

    def push(self, current):
        self.frames = torch.cat((self.frames[:, 1:], current[:, None]), 1)

    def flattened(self):
        return self.frames.flatten(1).clone()
