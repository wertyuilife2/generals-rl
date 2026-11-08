"""Simulation controller assembly; baseline implementations are optional."""

import numpy as np

from generals.protocol.controller import Controller


def create_controller(kind: str, player_id: int = 0, seed: int = 0) -> Controller:
    try:
        from generals_baselines import create_controller as create_baseline
    except ModuleNotFoundError as exc:
        if exc.name != "generals_baselines":
            raise
        raise ImportError("AI simulation requires generals-baselines; install with: pip install -e ./baselines") from exc
    return create_baseline(kind, player_id=player_id, seed=seed)


def make_controllers(kinds: list[str] | tuple[str, ...], seed: int = 0) -> list[Controller]:
    """Spawn streams by seat, reserving child zero for map generation."""
    streams = np.random.SeedSequence(seed).spawn(len(kinds) + 1)[1:]
    return [create_controller(kind, pid, int(stream.generate_state(1, dtype=np.uint64)[0]))
            for pid, (kind, stream) in enumerate(zip(kinds, streams))]


__all__ = ["Controller", "create_controller", "make_controllers"]
