"""Neural feature recipes, separate from lossless history transport/storage.

Compact inputs keep current local categories, signed log-army differences and
non-stacked facts. Dynamic inputs add eight-frame changes, last-seen enemy army
and public scoreboard EMA residuals. They never read the privileged board.
"""
from dataclasses import dataclass

import torch


SCORE_HALF_LIVES = (16, 64, 256)
SCORE_FEATURES = 2  # Public enemy army and territory.


@dataclass(frozen=True)
class FeatureRecipe:
    history_length: int
    knowledge_channels: int
    half_precision: bool = False
    dynamic_memory: bool = False
    deduplicate_half: bool = False
    terrain_distances: bool = False

    @property
    def public_extra_dim(self):
        return len(SCORE_HALF_LIVES) * SCORE_FEATURES if self.dynamic_memory else 0


FEATURE_RECIPES = {
    "stacked": FeatureRecipe(4, 11),
    "compact": FeatureRecipe(4, 11, half_precision=True),
    "temporal": FeatureRecipe(8, 12, half_precision=True, dynamic_memory=True),
    "temporal_dedup": FeatureRecipe(8, 12, half_precision=True, dynamic_memory=True, deduplicate_half=True),
    "temporal_paths": FeatureRecipe(8, 14, half_precision=True, dynamic_memory=True, terrain_distances=True),
}
FEATURE_SETS = tuple(FEATURE_RECIPES)
CURRENT_CHANNELS = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
MEMORY_CHANNELS = (0, 1, 2, 4, 6, 7, 9, 10)


def feature_recipe(name):
    if name not in FEATURE_RECIPES:
        raise ValueError(f"unsupported feature_set: {name!r}")
    return FEATURE_RECIPES[name]


def spatial_channels(layout):
    if layout.feature_set == "stacked":
        return layout.history_length * layout.map_planes + layout.knowledge_channels
    recipe = feature_recipe(layout.feature_set)
    return (len(CURRENT_CHANNELS) + 2 * (layout.history_length - 1) + len(MEMORY_CHANNELS)
            + int(recipe.dynamic_memory) + 2 * int(recipe.terrain_distances))


def spatial_features(layout, history, knowledge):
    maps = layout.history_maps(history)
    memory = layout.memory_maps(knowledge, len(history))
    if layout.feature_set == "stacked":
        return maps, memory
    frames = maps.reshape(len(history), layout.history_length, layout.map_planes, layout.height, layout.width)
    # Visibility and ownership changes can cause differences too; current
    # visibility and last-seen facts explicitly disambiguate their semantics.
    army = frames[:, :, 9:10] * frames[:, :, 5:7]
    differences = (army[:, 1:] - army[:, :-1]).flatten(1, 2)
    channels = MEMORY_CHANNELS + ((11,) if feature_recipe(layout.feature_set).dynamic_memory else ())
    if feature_recipe(layout.feature_set).terrain_distances:
        channels += (12, 13)
    return torch.cat((frames[:, -1, CURRENT_CHANNELS], differences), 1), memory[:, channels]


def public_features(layout, history, knowledge):
    current = layout.frames(history)[:, -1, -layout.public_dim:]
    extra = feature_recipe(layout.feature_set).public_extra_dim
    if extra:
        current = torch.cat((current, knowledge[:, -extra:]), -1)
    return current
