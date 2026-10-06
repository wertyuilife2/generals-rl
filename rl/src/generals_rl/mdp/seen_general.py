"""Episode-local enemy-general landmark from current LOCAL features only."""
import torch


class SeenEnemyGeneralEncoder:
    def __init__(self, base):
        self.base = base
        self.channels = base.channels
        self.semantics = getattr(base, "semantics", "local_p2_v1") + "_seen_enemy_general_v1"
        self.positions = None

    def visibility(self, batch, player_ids):
        return self.base.visibility(batch, player_ids)

    def reset(self, batch, mask=None):
        if batch.players != 2 or batch.width * batch.height > 32767:
            raise ValueError("seen enemy general requires two players and int16 cell ids")
        if self.positions is None:
            self.positions = torch.full((batch.num_envs, 2), -1, dtype=torch.int16, device=batch.device)
        if self.positions.shape != (batch.num_envs, 2):
            raise ValueError("memory lanes must match the full observation context")
        if mask is None:
            self.positions.fill_(-1)
        else:
            self.positions[mask] = -1

    def observe(self, batch, player_ids, local):
        """Reuse the current local features, before any memory projection."""
        if self.positions is None:
            raise RuntimeError("reset observation context before local encode")
        if self.positions.shape != (batch.num_envs, 2):
            raise ValueError("memory lanes must match the full observation context")
        cells = batch.width * batch.height
        planes = local[:, :len(self.channels) * cells].reshape(batch.num_envs, len(self.channels), cells)
        # Do not use raw hidden structures, enemy general_pos, or critic.
        seen = planes[:, 0].bool() & planes[:, 4].bool() & planes[:, 6].bool()
        found = seen.sum(1).eq(1)
        columns = player_ids[:, None]
        prior = self.positions.gather(1, columns).squeeze(1)
        remembered = torch.where(found, seen.long().argmax(1).to(torch.int16), prior)
        # Alive flags are public. Capturing the enemy general ends 1v1.
        alive = batch.state["alive"].gather(1, 1-columns).squeeze(1)
        self.positions.scatter_(1, columns, torch.where(alive, remembered, -1)[:, None])

    def project(self, local, player_ids, cells):
        if self.positions is None:
            return local
        if self.positions.shape[0] != local.shape[0]:
            raise ValueError("memory lanes must match the full observation context")
        positions = self.positions.gather(1, player_ids[:, None]).squeeze(1).long()
        # Only landmark plane changes; unseen army/owner/visibility remain hidden.
        indices = (4 * cells + positions.clamp_min(0))[:, None]
        prior = local.gather(1, indices)
        local.scatter_(1, indices, torch.where(positions[:, None].ge(0), 1, prior))
        return local

    def encode(self, batch, player_ids, privileged=False):
        local = self.base.encode(batch, player_ids, privileged=privileged)
        if privileged:
            return local
        self.observe(batch, player_ids, local)
        return self.project(local, player_ids, batch.width * batch.height)

    def clone(self):
        """Fork observation memory for an information-state branch, not a model."""
        result = type(self)(self.base)  # Base is stateless; only memory is mutable.
        result.positions = None if self.positions is None else self.positions.clone()
        return result
