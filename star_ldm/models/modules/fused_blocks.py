import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

def exists(val):
    return val is not None

_metal_rmsnorm = None
_metal_attn = None
_metal_rmsnorm_checked = False
_metal_attn_checked = False

def _metal_disabled():
    return os.environ.get('STAR_DISABLE_METAL') == '1'

def _get_metal_rmsnorm():
    global _metal_rmsnorm, _metal_rmsnorm_checked
    if _metal_disabled():
        return None
    if not _metal_rmsnorm_checked:
        _metal_rmsnorm_checked = True
        from star_ldm.kernels import get_rmsnorm_film_kernel
        _metal_rmsnorm = get_rmsnorm_film_kernel()
    return _metal_rmsnorm

def _get_metal_attn():
    global _metal_attn, _metal_attn_checked
    if _metal_disabled():
        return None
    if not _metal_attn_checked:
        _metal_attn_checked = True
        from star_ldm.kernels import get_tiny_attention_kernel
        _metal_attn = get_tiny_attention_kernel()
    return _metal_attn

def metal_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift):
    mod = _get_metal_rmsnorm()
    if mod is None or not x.is_mps:
        return None
    return mod.rmsnorm_film(
        x.contiguous(), gamma.contiguous(), float(dim_scale),
        film_scale, film_shift,
    )

def metal_rmsnorm(x, gamma, dim_scale):
    mod = _get_metal_rmsnorm()
    if mod is None or not x.is_mps:
        return None
    return mod.rmsnorm(x.contiguous(), gamma.contiguous(), float(dim_scale))

def metal_tiny_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale):
    mod = _get_metal_attn()

    if mod is None or not q.is_mps or q.size(2) != 8:
        return None
    return mod.tiny_attention(
        q.contiguous(), k.contiguous(), v.contiguous(),
        q_gamma.contiguous(), k_gamma.contiguous(),
        float(dim_head_scale), float(attn_scale),
    )

@torch.jit.script
def _jit_fused_rmsnorm_film(
    x: Tensor, gamma: Tensor, dim_scale: float,
    film_scale: Tensor, film_shift: Tensor,
) -> Tensor:
    norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
    x_normed = x / norm * dim_scale
    return (x_normed * gamma) * (film_scale + 1.0) + film_shift

@torch.jit.script
def _jit_fused_rmsnorm(x: Tensor, gamma: Tensor, dim_scale: float) -> Tensor:
    norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
    return x / norm * dim_scale * gamma

def fused_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift):
    out = metal_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift)
    if out is not None:
        return out
    return _jit_fused_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift)

def fused_rmsnorm(x, gamma, dim_scale):
    out = metal_rmsnorm(x, gamma, dim_scale)
    if out is not None:
        return out
    return _jit_fused_rmsnorm(x, gamma, dim_scale)

@torch.jit.script
def _jit_fused_qknorm_attention(
    q: Tensor, k: Tensor, v: Tensor,
    q_gamma: Tensor, k_gamma: Tensor,
    dim_head_scale: float, attn_scale: float,
) -> Tensor:
    q_norm = torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
    q = q / q_norm * dim_head_scale * q_gamma
    k_norm = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-8)
    k = k / k_norm * dim_head_scale * k_gamma
    attn = torch.matmul(q, k.transpose(-2, -1)) * attn_scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)

def fused_qknorm_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale):
    out = metal_tiny_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale)
    if out is not None:
        return out
    return _jit_fused_qknorm_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale)

class FusedAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.heads = original.heads
        self.dropout = original.dropout
        self.causal = original.causal
        self.gamma = original.pre_norm.gamma
        self.dim_scale = original.pre_norm.scale
        self.to_qkv = original.to_qkv
        self.to_out = original.to_out
        self.q_gamma = original.q_norm.gamma
        self.k_gamma = original.k_norm.gamma
        self.dim_head_scale = original.q_norm.scale
        self.attn_scale = 1.0 / math.sqrt(self.q_gamma.shape[0])
        self.time_cond = getattr(original, 'time_cond', None)

    def forward(self, x, attn_bias, time_emb=None):
        batch = x.shape[0]
        if exists(time_emb) and exists(self.time_cond):
            film_scale, film_shift = self.time_cond(time_emb).chunk(2, dim=-1)
            x = fused_rmsnorm_film(x, self.gamma, self.dim_scale, film_scale, film_shift)
        else:
            x = fused_rmsnorm(x, self.gamma, self.dim_scale)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, 'b n (h d) -> b h n d', h=self.heads) for t in qkv)

        if exists(attn_bias) or self.causal:
            q_norm = torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
            q = q / q_norm * self.dim_head_scale * self.q_gamma
            k_norm = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-8)
            k = k / k_norm * self.dim_head_scale * self.k_gamma
            if exists(attn_bias):
                attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, self.heads, -1, -1)
                if self.causal:
                    from star_ldm.models.modules.blocks import create_causal_mask
                    mask_value = -torch.finfo(q.dtype).max
                    causal_mask = create_causal_mask(q.shape[-2], k.shape[-2], device=q.device)
                    attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.,
            )
        else:
            out = fused_qknorm_attention(
                q, k, v, self.q_gamma, self.k_gamma,
                self.dim_head_scale, self.attn_scale,
            )

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FusedFeedForward(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.gamma = original.pre_norm.gamma
        self.dim_scale = original.pre_norm.scale
        self.time_cond = original.time_cond
        self.net = original.net

    def forward(self, x, time_emb=None):
        if exists(self.time_cond) and exists(time_emb):
            film_scale, film_shift = self.time_cond(time_emb).chunk(2, dim=-1)
            x = fused_rmsnorm_film(x, self.gamma, self.dim_scale, film_scale, film_shift)
        else:
            x = fused_rmsnorm(x, self.gamma, self.dim_scale)
        return self.net(x)

def swap_to_fused_blocks(transformer_model: nn.Module):
    from star_ldm.models.modules.blocks import Attention, FeedForward
    for block in transformer_model.modules():
        if hasattr(block, 'attn') and isinstance(block.attn, Attention):
            block.attn = FusedAttention(block.attn)
        if hasattr(block, 'ff') and isinstance(block.ff, FeedForward):
            block.ff = FusedFeedForward(block.ff)
    return transformer_model
