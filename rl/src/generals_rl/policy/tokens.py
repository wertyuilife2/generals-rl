"""ViT-only padding exclusion, shared by training and exported actors.

CUDA mixed precision packs valid cells into PyTorch's variable-length Flash
Attention. CPU/FP32 uses the same attention mask on the full sequence. Neither
path changes cell positions or the environment's padded action vocabulary.
"""
import torch
from torch.nn import functional as F


def patch_mask(valid, height, width, patch_size=1):
    """Keep every patch containing a real cell, including partial edge patches."""
    if patch_size == 1:
        return valid
    return F.max_pool2d(valid.reshape(-1, 1, height, width).float(),
                        patch_size, stride=patch_size, ceil_mode=True).flatten(1).bool()


def prepare_tokens(valid):
    """Resolve the variable output shape before entering a compiled network."""
    if valid is None:
        raise ValueError("masked ViT requires valid_cells")
    if not valid.is_cuda:
        return valid, None, None
    indices = valid.flatten().nonzero().flatten()
    offsets = F.pad(valid.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
    # Frozen actors retain static batch buckets, but live maps and opponent
    # assignments change the number of valid tokens even within one bucket.
    torch._dynamo.mark_dynamic(indices, 0)
    return valid, indices, offsets


class _PackedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, offsets, maximum):
        output, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(
            query, key, value, offsets, offsets, maximum, maximum, 0., False, False)
        ctx.save_for_backward(query, key, value, output, lse, offsets, rng, unused)
        ctx.maximum = maximum
        return output

    @staticmethod
    def backward(ctx, gradient):
        query, key, value, output, lse, offsets, rng, unused = ctx.saved_tensors
        gradients = torch.ops.aten._flash_attention_backward(
            gradient.contiguous(), query, key, value, output, lse, offsets, offsets,
            ctx.maximum, ctx.maximum, 0., False, rng, unused)
        return *gradients, None, None


def valid_token_features(transformer, inputs, valid, tokens):
    count, _, height, width = inputs.shape
    height, width = transformer.patch_embed.dynamic_feat_size((height, width))
    x = transformer.patch_embed(inputs)
    # Keep the original canvas position table: resizing it to a smaller board
    # would change position semantics in addition to removing padding.
    x = transformer.norm_pre(transformer.patch_drop(transformer._pos_embed(x)))
    packed = x.is_cuda and (torch.is_autocast_enabled('cuda') or x.dtype in (torch.float16, torch.bfloat16))
    if packed:
        _, indices, offsets = tokens
        x = x.flatten(0, 1).index_select(0, indices)
        for block in transformer.blocks:
            attention = block.attn
            qkv = attention.qkv(block.norm1(x)).reshape(-1, 3, attention.num_heads, attention.head_dim)
            query, key, value = qkv.unbind(1)
            query, key = attention.q_norm(query), attention.k_norm(key)
            attended = _PackedAttention.apply(query, key, value, offsets, height * width)
            attended = attention.norm(attended.reshape(-1, attention.attn_dim))
            x = x + block.ls1(attention.proj(attended))
            x = x + block.ls2(block.mlp(block.norm2(x)))
        x = transformer.norm(x)
        x = x.new_zeros(count * height * width, x.shape[-1]).index_copy(0, indices, x)
        x = x.reshape(count, height * width, -1)
    else:
        mask = valid[:, None, None, :]
        for block in transformer.blocks:
            x = block(x, attn_mask=mask)
        x = transformer.norm(x)
    return x.transpose(1, 2).reshape(count, -1, height, width)


def valid_mean(spatial, valid):
    weights = valid.reshape(spatial.shape[0], 1, *spatial.shape[-2:]).to(spatial.dtype)
    total = (spatial * weights).sum((-2, -1), dtype=torch.float32)
    count = weights.sum((-2, -1), dtype=torch.float32).clamp_min(1)
    return (total / count).to(spatial.dtype)
