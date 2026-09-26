"""Built-in controller factory. Importing this module never imports the GUI."""

import numpy as np

from .base import Controller
from .aggressive import AggressiveAI
from .expansion import ExpansionAI
from .defensive import DefensiveAI
from .random_policy import RandomAI
from .human import HumanController

AI_KINDS = ("aggressive", "expansion", "defensive", "random")
CONTROLLER_KINDS = (*AI_KINDS, "human")


def create_controller(kind: str, player_id: int = 0, seed: int = 0) -> Controller:
    types = dict(zip(CONTROLLER_KINDS, (AggressiveAI, ExpansionAI, DefensiveAI, RandomAI, HumanController)))
    try:
        controller_type = types[kind.lower()]
    except KeyError:
        raise ValueError(f"Unknown controller {kind!r}; choose from {', '.join(CONTROLLER_KINDS)}") from None
    return controller_type(player_id=player_id, seed=seed)


def make_controllers(kinds: list[str] | tuple[str, ...], seed: int = 0) -> list[Controller]:
    """Spawn streams by seat, reserving child zero for map generation."""
    streams = np.random.SeedSequence(seed).spawn(len(kinds) + 1)[1:]
    return [create_controller(kind, pid, int(stream.generate_state(1, dtype=np.uint64)[0]))
            for pid, (kind, stream) in enumerate(zip(kinds, streams))]


__all__ = ["Controller", "AggressiveAI", "ExpansionAI", "DefensiveAI", "RandomAI", "HumanController", "create_controller", "make_controllers", "AI_KINDS", "CONTROLLER_KINDS"]
