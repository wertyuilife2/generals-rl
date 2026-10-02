"""Measurement contracts without importing a CPU or GPU implementation."""
from dataclasses import dataclass
from generals.protocol.config import MapConfig, _integer


@dataclass(frozen=True)
class BenchmarkCase:
    map_config: MapConfig
    batch_size: int = 1
    episode_ticks: int = 2000
    seed: int = 42
    threads: int = 0
    device: str = "cuda"

    def __post_init__(self):
        if not isinstance(self.map_config, MapConfig):
            raise TypeError("map_config must be MapConfig")
        for name, minimum in (("batch_size", 1), ("episode_ticks", 1), ("seed", 0), ("threads", 0)):
            object.__setattr__(self, name, _integer(name, getattr(self, name), minimum))
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")


@dataclass
class Counters:
    env_ticks: int = 0
    player_actions: int = 0
    move_actions: int = 0
    wait_actions: int = 0
    invalid_actions: int = 0
    positive_moves: int = 0
    captures: int = 0
    terminated_episodes: int = 0
    truncated_episodes: int = 0
