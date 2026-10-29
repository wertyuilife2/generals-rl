"""Backend-neutral vector step contract. No NumPy, torch, or GUI imports."""
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BatchStep:
    """Borrowed CPU/GPU buffers; copy/clone them before the next step to retain.

    done is [B,2] (natural termination, truncation) before an automatic reset.
    stats describes the completed tick, even when state has already reset.
    """
    moves: Any
    stats: Any
    done: Any
    errors: Any


@runtime_checkable
class VectorBackend(Protocol):
    """Common execution contract, not a Gym reward/observation wrapper."""
    batch_size: int
    players: int
    state: dict[str, Any]
    stats: Any

    def step(self, actions: Any) -> BatchStep: ...
    def reset(self, mask: Any = None) -> dict[str, Any]: ...
    def surrender(self, player_id: int, mask: Any = None) -> None: ...
    def synchronize(self) -> None: ...
    def check_errors(self) -> None: ...
    def invalidate_cache(self) -> None: ...
    def to_states(self) -> list: ...
    def close(self) -> None: ...
