"""Observation/action boundary for local and remotely hosted matches.

Unlike VectorBackend, this interface does not expose an authoritative GameState.
An advance may span several server ticks and does not acknowledge an online move.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Protocol

if TYPE_CHECKING:
    from generals.core.observation import Observation
    from generals.core.state import MoveResult
    from .actions import Action


@dataclass(frozen=True)
class MatchFrame:
    frame_id: int
    observations: Mapping[int, "Observation"]
    results: tuple["MoveResult", ...] = ()
    done: bool = False
    reason: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'observations', MappingProxyType(dict(self.observations)))


class StaleFrameError(RuntimeError):
    """A newer observation arrived before submission; recompute the action."""
    def __init__(self, frame):
        super().__init__('a newer match frame is available; action was not sent')
        self.frame = frame


class MatchBackend(Protocol):
    supports_move_results: bool

    def start(self) -> MatchFrame: ...
    def step(self, actions: Mapping[int, "Action"], *, frame_id: int) -> MatchFrame: ...
    def close(self) -> None: ...
