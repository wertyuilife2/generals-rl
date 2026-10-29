"""Public lifecycle shared by human, heuristic, and learned players."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from generals.core.observation import Observation
    from generals.core.state import Action, MoveResult


@runtime_checkable
class Controller(Protocol):
    def reset(self, player_id: int, seed: int) -> None: ...
    def act(self, observation: Observation) -> Action: ...
    def on_result(self, result: MoveResult) -> None: ...
