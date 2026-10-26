"""Training privileged-visibility adapter over the shared combat features."""
from ..policy.combat import CombatObservationEncoder as PublicCombatEncoder
from .observations import PrivilegedVisibility


class CombatObservationEncoder(PrivilegedVisibility, PublicCombatEncoder):
    pass
