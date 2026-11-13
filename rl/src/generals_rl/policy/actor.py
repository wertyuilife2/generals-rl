"""Construct trainable actors without a PPO policy, critic or training package."""
from collections.abc import Mapping

from .cnn import (SpatialActor, SpatialTrunk, _PackedSpatialActor, add_actor_knowledge,
                  add_own_king_geometry, light_trunk)
from .layout import ObservationLayout
from .features import feature_recipe


def build_actor(network, features):
    """Return a pure, trainable module whose state keys match policy.actor."""
    if not isinstance(network, Mapping):
        raise TypeError("network must be a mapping")
    if isinstance(features, ObservationLayout):
        layout, players = features, 2
        feature_values = {"persistent_knowledge": bool(layout.knowledge_channels)}
    elif isinstance(features, Mapping):
        players = features.get("players", 2)
        feature_values = features
        channels = network.get("knowledge_channels", features.get(
            "knowledge_channels", feature_recipe(features.get("feature_set", "stacked")).knowledge_channels
            if features.get("persistent_knowledge", False) else 0))
        layout = ObservationLayout(features["width"], features["height"],
                                   history_length=features.get("history_length", 4),
                                   map_planes=features.get("map_planes", 13),
                                   public_dim=features.get("public_dim", 13), knowledge_channels=channels,
                                   feature_set=features.get("feature_set", "stacked"))
    else:
        raise TypeError("features must be a mapping or ObservationLayout")
    if type(players) is not int or (players, layout.map_planes, layout.public_dim) != (2, 13, 13):
        raise ValueError("actors require P2 thirteen-plane/public13 features")
    family = network.get("family")
    common = {"family", "amp_bfloat16", "wait_move_factorization"}
    if "amp_bfloat16" in network and type(network["amp_bfloat16"]) is not bool:
        raise TypeError("amp_bfloat16 must be boolean")
    if "wait_move_factorization" in network and type(network["wait_move_factorization"]) is not bool:
        raise TypeError("wait_move_factorization must be boolean")
    if family == "scaled":
        allowed = common | {"backbone", "model_size", "spatial_mode", "head_channels", "input_channel_alignment", "knowledge_channels", "mask_padding_tokens", "patch_size", "vit_config"}
        if set(network) - allowed:
            raise ValueError(f"unsupported scaled actor options: {sorted(set(network) - allowed)}")
        recipe = feature_recipe(layout.feature_set)
        if layout.history_length != recipe.history_length or layout.knowledge_channels != recipe.knowledge_channels or feature_values.get("persistent_knowledge", True) is not True:
            raise ValueError("scaled actor history and persistent memory must match its feature recipe")
        if network.get("wait_move_factorization", True) is not True:
            raise ValueError("scaled actors require WAIT/MOVE factorization")
        kind = network.get("backbone", "vit")
        if kind != "vit":
            raise ValueError("scaled backbone must be vit; use the original policy bundle for retired networks")
        size, mode = network.get("model_size", "1m"), network.get("spatial_mode", "dense")
        if size not in ("1m", "10m", "100m") or mode != "dense":
            raise ValueError("unsupported scaled model_size or spatial_mode")
        head, alignment = network.get("head_channels", 128), network.get("input_channel_alignment", 4)
        for name, value, minimum in (("head_channels", head, 1), ("input_channel_alignment", alignment, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        from .scaled import _ScaledInformationActor
        return _ScaledInformationActor(layout, size, alignment, head,
                                       network.get("mask_padding_tokens"), network.get("patch_size", 1),
                                       network.get("vit_config"))
    if family not in ("spatial", "lightcnn", "information", "optimized"):
        raise ValueError("unsupported actor family")
    allowed = common | {"spatial_channels", "residual_blocks", "own_king_geometry", "knowledge_channels", "input_channel_alignment"}
    if set(network) - allowed:
        raise ValueError(f"unsupported CNN actor options: {sorted(set(network) - allowed)}")
    channels, blocks = network.get("spatial_channels", 32 if family == "spatial" else 16), network.get("residual_blocks", 4)
    for name, value in (("spatial_channels", channels), ("residual_blocks", blocks)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    actor = SpatialActor(channels, blocks, SpatialTrunk if family == "spatial" else light_trunk,
                         layout=layout)
    geometry = network.get("own_king_geometry", False)
    if type(geometry) is not bool or family == "spatial" and geometry:
        raise ValueError("invalid own_king_geometry for actor family")
    if geometry:
        add_own_king_geometry(actor)
    if family in ("information", "optimized"):
        add_actor_knowledge(actor, layout.knowledge_channels)
    elif network.get("knowledge_channels", 0):
        raise ValueError("this actor family does not consume knowledge")
    if family == "optimized":
        alignment = network.get("input_channel_alignment", 4)
        if type(alignment) is not int or alignment < 0:
            raise ValueError("input_channel_alignment must be a nonnegative integer")
        actor = _PackedSpatialActor(actor, alignment)
    return actor
