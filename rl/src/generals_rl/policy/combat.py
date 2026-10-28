"""Visible one-action combat potential for rectangular two-player maps."""
import torch
import torch.nn.functional as F

from .observations import LocalObservationEncoder


def _neighbor_max(values, width, height):
    grid = values.reshape(-1, 2, height, width)
    padded = F.pad(grid, (1, 1, 1, 1))
    vertical = torch.maximum(padded[..., :-2, 1:-1], padded[..., 2:, 1:-1])
    horizontal = torch.maximum(padded[..., 1:-1, :-2], padded[..., 1:-1, 2:])
    return torch.maximum(vertical, horizontal).flatten(-2)


class CombatObservationEncoder(LocalObservationEncoder):
    """Replace duplicate P2 identity planes with risk and signed combat margin.

    Risk uses current visible four-neighbor ALL_BUT_ONE armies. Own margin is
    garrison + best single reinforcement - risk; other margin is best friendly
    attack - the current visible defender.
    These are potential choices, not a prediction of joint actions or seat order.

    Float32 troop arithmetic is exact for the intermediate sums <=2**24. Native
    rules/debug boundaries own exceptional int64 overflow checks, avoiding host
    synchronization in every GPU encode.
    """
    semantics = "official_31_4_5_combat_p2_v1"
    channels = LocalObservationEncoder.channels + ("enemy_attack_risk", "combat_margin")

    def encode(self, batch, player_ids, **visibility_options):
        if batch.players != 2:
            raise ValueError("combat features require two players")
        result = super().encode(batch, player_ids, **visibility_options)
        cells = batch.width * batch.height
        planes = result[:, :12 * cells].reshape(batch.num_envs, 12, cells)
        visible = planes[:, 0].bool()
        own, enemy = planes[:, 5].bool(), planes[:, 6].bool()
        # Existing visible planes supply ownership/structures/terrain. Mask army
        # before arithmetic; hidden changes cannot enter growth, max or log.
        available = torch.where(visible, batch.state["army"], 0).float()
        movable = (available - 1).clamp_min(0)
        incoming = _neighbor_max(movable[:, None] * planes[:, 5:7], batch.width, batch.height)
        friendly, risk = incoming[:, 0], incoming[:, 1]
        margin = torch.where(own, available + friendly - risk, friendly - available)
        observable = visible & ~planes[:, 2].bool()
        risk = torch.where(observable, risk, 0.0)
        margin = torch.where(observable, margin, 0.0)
        result[:, 10 * cells:11 * cells] = torch.log1p(risk) / 10.0
        result[:, 11 * cells:12 * cells] = margin.sign() * torch.log1p(margin.abs()) / 10.0
        return result
