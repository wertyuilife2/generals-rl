"""Private feature configuration for an exported policy package."""
from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureConfig:
    width: int
    height: int
    players: int
    max_episode_steps: int
    history_length: int
    combat_features: bool
    king_threat_distance: bool
    persistent_knowledge: bool
    knowledge_age_scale: int = 2000
    random_maps: bool = True

    def __post_init__(self):
        for name in ("width", "height", "players", "max_episode_steps", "history_length", "knowledge_age_scale"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"features.{name} must be a positive integer")
        if not 2 <= self.players <= 8:
            raise ValueError("features.players must be between 2 and 8")
        for name in ("random_maps", "combat_features", "king_threat_distance", "persistent_knowledge"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"features.{name} must be boolean")
        if (self.combat_features or self.king_threat_distance) and self.players != 2:
            raise ValueError("combat and king-distance features require two players")
