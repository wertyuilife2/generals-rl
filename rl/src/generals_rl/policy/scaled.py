"""Shared trainable scaled actor/backbones; no PPO, optimizer or critic."""
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import GroupNormAct
from timm.models.resnetv2 import ResNetV2
from timm.models.vision_transformer import VisionTransformer
from .backbones import RESNET_DEPTHS, ResNetBackbone, ViTBackbone
from .layout import own_king_coordinates
from .tokens import embed_tokens, patch_mask, prepare_tokens, spatial_features, valid_mean, valid_token_features


RESNET_TOTAL_PRESETS = {
    "1m": ((32, 64, 128, 272), 16),
    "10m": ((112, 224, 464, 928), 32),
    "100m": ((368, 736, 1472, 2944), 88),
}


VIT_TOTAL_PRESETS = {"1m": (96, 4, 3), "10m": (224, 8, 7), "100m": (640, 10, 10)}


def vit_configuration(size, shared=True):
    """Widen shared ViT1M to keep total policy capacity near one million."""
    return (128, 4, 4) if shared and size == "1m" else VIT_TOTAL_PRESETS[size]


class _TotalResNetBackbone(ResNetBackbone):
    """Balanced-budget ResNetV2 with optional coarse stages and a dense skip."""
    def __init__(self, in_channels, size, spatial_mode="dense"):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.spatial_mode = spatial_mode
        channels, stem_channels = RESNET_TOTAL_PRESETS[size]
        self.local_channels = channels[0]
        self.out_channels, self.depth = channels[-1], sum(RESNET_DEPTHS)
        self.encoder = ResNetV2(
            layers=list(RESNET_DEPTHS), channels=channels, in_chans=in_channels,
            num_classes=0, global_pool="", stem_type="deep", stem_chs=stem_channels,
            avg_down=False, norm_layer=partial(GroupNormAct, num_groups=4),
            output_stride=32, drop_rate=0.0, drop_path_rate=0.0, zero_init_last=False)
        self.encoder.stem.pool = nn.Identity()
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1, 1)
        for feature in self.encoder.feature_info:
            feature["reduction"] = 1
        if spatial_mode == "pyramid":
            for stage in list(self.encoder.stages)[1:]:
                stage.blocks[0].conv2.stride = (2, 2)
                stage.blocks[0].downsample.conv.stride = (2, 2)
            for feature in self.encoder.feature_info:
                if feature["module"].startswith("stages."):
                    feature["reduction"] = 2 ** int(feature["module"].rsplit(".", 1)[1])

    def forward_pyramid(self, x):
        """Return full-resolution stage0 and normalized coarse deep features."""
        x = self.encoder.stem(x)
        local = self.encoder.stages[0](x)
        x = local
        for stage in list(self.encoder.stages)[1:]:
            x = stage(x)
        return local, self.encoder.norm(x)


class _ViTBackbone(ViTBackbone):
    """Non-overlapping ViT with an explicit width/depth/head configuration."""
    def __init__(self, in_channels, configuration, spatial_mode="dense", map_shape=(23, 23), patch_size=1):
        nn.Module.__init__(self)
        if (len(configuration) != 3 or any(type(value) is not int or value < 1 for value in configuration)
                or configuration[0] % configuration[2]):
            raise ValueError("vit_config requires positive width/depth/heads with width divisible by heads")
        if spatial_mode not in ("dense", "pyramid"):
            raise ValueError("spatial_mode must be dense or pyramid")
        self.spatial_mode = spatial_mode
        self.in_channels = in_channels
        self.out_channels, self.depth, self.num_heads = configuration
        self.local_channels = self.out_channels
        canvas = tuple((side + patch_size - 1) // patch_size * patch_size for side in map_shape)
        self.transformer = VisionTransformer(
            img_size=canvas, patch_size=patch_size, in_chans=in_channels,
            num_classes=0, global_pool="", class_token=False,
            embed_dim=self.out_channels, depth=self.depth, num_heads=self.num_heads,
            mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=1e-4, dynamic_img_size=True, dynamic_img_pad=patch_size > 1,
            drop_rate=0.0, pos_drop_rate=0.0, patch_drop_rate=0.0,
            proj_drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0)

    def forward_pyramid(self, x):
        """Keep cell-resolution block zero, then pool spatial tokens twice.

        This changes spatial computation. It does not establish equal quality
        or measured efficiency. Dense forward remains the inherited ViT path.
        """
        if self.spatial_mode != "pyramid":
            raise ValueError("forward_pyramid requires spatial_mode=pyramid")
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("ViT pyramid requires packed NCHW inputs")
        transformer = self.transformer
        count, height, width = x.shape[0], x.shape[2], x.shape[3]
        tokens = transformer.patch_embed(x)
        # Add the configured board position table before either pool.
        tokens = transformer.norm_pre(transformer.patch_drop(transformer._pos_embed(tokens)))
        tokens = transformer.blocks[0](tokens)
        local = tokens.transpose(1, 2).reshape(count, self.out_channels, height, width)
        coarse = F.avg_pool2d(local, 3, stride=2, padding=1, count_include_pad=False)
        tokens = coarse.flatten(2).transpose(1, 2)
        middle_blocks = 1 if self.depth == 4 else 2
        for index in range(1, self.depth):
            if index == middle_blocks + 1:
                coarse = tokens.transpose(1, 2).reshape(count, self.out_channels, (height + 1) // 2, (width + 1) // 2)
                coarse = F.avg_pool2d(coarse, 3, stride=2, padding=1, count_include_pad=False)
                tokens = coarse.flatten(2).transpose(1, 2)
            tokens = transformer.blocks[index](tokens)
        deep = transformer.norm(tokens).transpose(1, 2).reshape(count, self.out_channels, (height + 3) // 4, (width + 3) // 4)
        return local, deep


def _build_total_backbone(kind, size, inputs, spatial_mode="dense", map_shape=(23, 23), patch_size=1,
                          vit_config=None):
    if type(patch_size) is not int or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    if patch_size != 1 and (kind != "vit" or spatial_mode != "dense"):
        raise ValueError("patch_size > 1 requires a dense ViT")
    if kind == "vit":
        configuration = vit_configuration(size) if vit_config is None else vit_config
        return _ViTBackbone(inputs, configuration, spatial_mode, map_shape, patch_size)
    return _TotalResNetBackbone(inputs, size, spatial_mode)


class _ScaledObservationEncoder(nn.Module):
    """The same local-information encoder serves policy and value learning."""
    def __init__(self, layout, kind, size, alignment, head_channels, spatial_mode="dense",
                 mask_padding_tokens=None, patch_size=1, vit_config=None):
        super().__init__()
        if mask_padding_tokens is None:
            mask_padding_tokens = kind == "vit" and spatial_mode == "dense"
        if type(mask_padding_tokens) is not bool or mask_padding_tokens and (kind != 'vit' or spatial_mode != 'dense'):
            raise ValueError('mask_padding_tokens requires a dense ViT')
        self.layout = layout
        self.spatial_mode = spatial_mode
        self.mask_padding_tokens = mask_padding_tokens
        self.patch_size = patch_size
        logical_inputs = layout.history_length * layout.map_planes + layout.knowledge_channels + 2
        padding = (-logical_inputs) % alignment if alignment else 0
        self.backbone = _build_total_backbone(kind, size, logical_inputs + padding, spatial_mode,
                                              (layout.height, layout.width), patch_size, vit_config)
        self.projection = nn.Conv2d(self.backbone.out_channels, head_channels, 1)
        self.register_buffer("_input_zero_padding", self.projection.weight.new_zeros(
            1, padding, layout.height, layout.width), persistent=False)
        if spatial_mode == "pyramid":
            self.local_skip = nn.Conv2d(self.backbone.local_channels, head_channels, 1)

    def input_maps(self, history, knowledge, valid_cells):
        layout, count = self.layout, history.shape[0]
        maps = layout.history_maps(history)
        parts = [maps, layout.memory_maps(knowledge, count), own_king_coordinates(maps[:, -layout.map_planes:], valid_cells)]
        if self._input_zero_padding.shape[1]:
            parts.append(self._input_zero_padding.expand(count, -1, -1, -1))
        inputs = torch.cat(parts, 1)
        if self.mask_padding_tokens and self.patch_size > 1:
            inputs = inputs.masked_fill(~valid_cells.reshape(count, 1, layout.height, layout.width), 0)
        return inputs

    def embed(self, history, knowledge, valid_cells, tokens):
        return embed_tokens(self.backbone.transformer, self.input_maps(history, knowledge, valid_cells), tokens)

    def project_tokens(self, x, tokens):
        transformer = self.backbone.transformer
        shape = transformer.patch_embed.dynamic_feat_size((self.layout.height, self.layout.width))
        spatial = F.silu(self.projection(spatial_features(transformer.norm(x), shape, tokens)))
        pooled = valid_mean(spatial, tokens[0]) if tokens is not None else spatial.mean((-2, -1))
        return spatial, pooled

    def forward(self, history, knowledge=None, valid_cells=None, tokens=None):
        layout = self.layout
        valid = None
        if self.mask_padding_tokens:
            if tokens is None:
                tokens = prepare_tokens(patch_mask(valid_cells, layout.height, layout.width, self.patch_size))
            valid = tokens[0]
        inputs = self.input_maps(history, knowledge, valid_cells)
        if self.spatial_mode == "pyramid":
            local, deep = self.backbone.forward_pyramid(inputs)
            coarse = self.projection(deep)
            if (coarse.is_cuda and torch.is_grad_enabled()
                    and isinstance(self.backbone, _TotalResNetBackbone)):
                coarse = coarse.contiguous(memory_format=torch.channels_last)
            spatial = F.silu(F.interpolate(coarse, size=(layout.height, layout.width),
                                          mode="bilinear", align_corners=False) + self.local_skip(local))
        elif self.mask_padding_tokens:
            features = valid_token_features(self.backbone.transformer, inputs, valid, tokens)
            spatial = F.silu(self.projection(features))
        else:
            spatial = F.silu(self.projection(self.backbone(inputs)))
        pooled = valid_mean(spatial, valid) if self.mask_padding_tokens else spatial.mean(dim=(-2, -1))
        return spatial, pooled


class _ScaledInformationActor(nn.Module):
    def __init__(self, layout, kind, size, alignment, head_channels, spatial_mode="dense",
                 mask_padding_tokens=None, patch_size=1, vit_config=None):
        super().__init__()
        self.layout = layout
        self.encoder = _ScaledObservationEncoder(layout, kind, size, alignment, head_channels,
                                                 spatial_mode, mask_padding_tokens, patch_size, vit_config)
        self.context = nn.Sequential(nn.Linear(head_channels + layout.public_dim, head_channels), nn.SiLU())
        self.cell_head = nn.Conv2d(head_channels, 8 * patch_size * patch_size, 1)
        self.wait_head = nn.Sequential(nn.Linear(2 * head_channels, head_channels), nn.SiLU(), nn.Linear(head_channels, 1))

    def decode(self, encoded, public):
        spatial, pooled = encoded
        layout, count = self.layout, spatial.shape[0]
        context = self.context(torch.cat((pooled, public), -1))
        fused = F.silu(spatial + context[..., None, None])
        moves = self.cell_head(fused)
        if self.encoder.patch_size > 1:
            moves = F.pixel_shuffle(moves, self.encoder.patch_size)[..., :layout.height, :layout.width]
        moves = moves.permute(0, 2, 3, 1).reshape(count, 8 * layout.cells)
        wait = self.wait_head(torch.cat((pooled, context), -1))
        return torch.cat((wait, moves), -1)

    def forward(self, history, knowledge=None, valid_cells=None, tokens=None):
        if not history.shape[0]:
            return history.new_empty((0, self.layout.num_actions))
        encoded = self.encoder(history, knowledge, valid_cells, tokens)
        public = self.layout.frames(history)[:, -1, -self.layout.public_dim:]
        return self.decode(encoded, public)
