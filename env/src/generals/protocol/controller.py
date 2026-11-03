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


@runtime_checkable
class BatchController(Protocol):
    """Independent lanes using VecEnv's public observations and encoded actions.

    Seats and uint64 seeds are final, one per lane. Inactive lanes return WAIT
    without advancing memory or RNG. Feedback contains [B,9] native move rows,
    [B] encoded actions, player_ids and an active receipt mask. on_result_batch
    is optional for policies that do not use move receipts. Batch routines
    run serially unless their implementation explicitly supports parallelism.
    """
    def reset_batch(self, player_ids, seeds) -> None: ...
    def act_batch(self, observations): ...
