"""Training-only privileged visibility around shared public feature encoding."""
import torch
from ..policy.observations import (LocalObservationEncoder as PublicLocalEncoder,
                                  KingThreatDistanceEncoder as PublicKingEncoder,
                                  make_observation_encoder as shared_encoder)


class PrivilegedVisibility:
    def encode(self, batch, player_ids, privileged=False, **visibility_options):
        if privileged:
            visibility_options.update(visible=torch.ones_like(batch.state["owner"], dtype=torch.bool),
                                      hide_enemy_cities=False)
        return super().encode(batch, player_ids, **visibility_options)


class LocalObservationEncoder(PrivilegedVisibility, PublicLocalEncoder):
    pass


class KingThreatDistanceEncoder(PrivilegedVisibility, PublicKingEncoder):
    pass


def make_observation_encoder(config):
    from .combat import CombatObservationEncoder
    return shared_encoder(config, local_encoder=LocalObservationEncoder,
                          combat_encoder=CombatObservationEncoder, king_encoder=KingThreatDistanceEncoder)
