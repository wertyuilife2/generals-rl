"""Named baseline factories with deterministic, independent seat streams."""
import numpy as np

from generals.protocol.controller import Controller
from .aggressive import AggressiveAI
from .expansion import ExpansionAI
from .defensive import DefensiveAI
from .random_policy import RandomAI

AI_KINDS = ("aggressive", "expansion", "defensive", "random")
_TYPES = dict(zip(AI_KINDS, (AggressiveAI, ExpansionAI, DefensiveAI, RandomAI)))


def create_controller(kind: str, player_id: int = 0, seed: int = 0) -> Controller:
    try:
        controller_type = _TYPES[kind.lower()]
    except KeyError:
        raise ValueError(f"Unknown baseline {kind!r}; choose from {', '.join(AI_KINDS)}") from None
    return controller_type(player_id=player_id, seed=seed)


def make_controllers(kinds: list[str] | tuple[str, ...], seed: int = 0) -> list[Controller]:
    """Use the same per-seat seeds as the interactive controller assembly."""
    streams = np.random.SeedSequence(seed).spawn(len(kinds) + 1)[1:]
    return [create_controller(kind, pid, int(stream.generate_state(1, dtype=np.uint64)[0]))
            for pid, (kind, stream) in enumerate(zip(kinds, streams))]
