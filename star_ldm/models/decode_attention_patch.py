
import math
import torch
import torch.nn.functional as F

_metal_decode_mod = None
_metal_decode_checked = False
_original_sdpa = F.scaled_dot_product_attention

def _get_metal_decode():
    global _metal_decode_mod, _metal_decode_checked
    if not _metal_decode_checked:
        _metal_decode_checked = True
        try:
            from star_ldm.kernels import get_decode_attention_kernel
            _metal_decode_mod = get_decode_attention_kernel()
        except Exception:
            _metal_decode_mod = None
    return _metal_decode_mod

def metal_decode_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, scale=None):
    mod = _get_metal_decode()

    if (mod is not None
        and query.is_mps
        and attn_mask is None
        and not is_causal
        and query.size(2) <= 16
        and key.size(2) <= 48
        and query.size(3) == 64
    ):
        if scale is None:
            scale = 1.0 / math.sqrt(query.size(-1))
        try:
            return mod.decode_attention(
                query.contiguous(), key.contiguous(), value.contiguous(), scale
            )
        except Exception:
            pass

    return _original_sdpa(
        query, key, value, attn_mask=attn_mask,
        dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )

def patch_gpt2_decode_attention(gpt2_model, max_kv_len=48):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    def make_patched_forward(original_attn):
        original_forward = original_attn.forward

        def patched_forward(hidden_states, *args, **kwargs):

            seq_len = hidden_states.size(1)
            use_metal = (
                seq_len <= 16
                and hidden_states.is_mps
                and _get_metal_decode() is not None
            )

            if use_metal:

                original_sdpa = F.scaled_dot_product_attention
                F.scaled_dot_product_attention = metal_decode_sdpa
                try:
                    return original_forward(hidden_states, *args, **kwargs)
                finally:
                    F.scaled_dot_product_attention = original_sdpa
            else:
                return original_forward(hidden_states, *args, **kwargs)

        return patched_forward

    patched_count = 0
    for module in gpt2_model.modules():
        if isinstance(module, GPT2Attention):
            module.forward = make_patched_forward(module)
            patched_count += 1

    return patched_count
