"""Construct trainable actors without a PPO policy, critic or training package."""
from collections.abc import Mapping

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
    raise ValueError("unsupported actor family; use frozen source for retired networks")
