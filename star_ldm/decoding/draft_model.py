
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

@dataclass
class DraftModelConfig:
    vocab_size: int = 50257
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.1
    tie_embeddings: bool = True

class DraftRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight

class DraftAttention(nn.Module):
    def __init__(self, config: DraftModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=(past_key_value is None and T > 1),
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out, new_kv

class DraftFFN(nn.Module):
    def __init__(self, config: DraftModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(config.hidden_dim, 2 * config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.dropout(x)
        x = self.down(x)
        return x

class DraftTransformerBlock(nn.Module):
    def __init__(self, config: DraftModelConfig):
        super().__init__()
        self.attn_norm = DraftRMSNorm(config.hidden_dim)
        self.attn = DraftAttention(config)
        self.ffn_norm = DraftRMSNorm(config.hidden_dim)
        self.ffn = DraftFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x = self.attn_norm(x)
        attn_out, new_kv = self.attn(x, past_key_value=past_key_value, use_cache=use_cache)
        x = residual + attn_out

        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        return x, new_kv

class DraftTransformerLM(nn.Module):

    def __init__(self, config: Optional[DraftModelConfig] = None):
        super().__init__()
        if config is None:
            config = DraftModelConfig()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DraftTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = DraftRMSNorm(config.hidden_dim)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.transformer = _TransformerShim(self.tok_emb)

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        n_params_no_emb = n_params - self.tok_emb.weight.numel()
        self._n_params = n_params
        self._n_params_no_emb = n_params_no_emb

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self):
        return self._n_params

    @property
    def num_parameters_no_embedding(self):
        return self._n_params_no_emb

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> "_DraftModelOutput":
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.tok_emb(input_ids)

        B, T, _ = x.shape

        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]

        positions = torch.arange(past_len, past_len + T, device=x.device)
        x = x + self.pos_emb(positions)
        x = self.emb_dropout(x)

        new_kvs = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = layer(x, past_key_value=past_kv, use_cache=use_cache)
            new_kvs.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        return _DraftModelOutput(
            logits=logits,
            past_key_values=new_kvs if use_cache else None,
        )

class _TransformerShim:
    def __init__(self, wte):
        self.wte = wte

class _DraftModelOutput:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values

def load_draft_model(
    checkpoint_path: str,
    device: str = "cpu",
    config: Optional[DraftModelConfig] = None,
) -> DraftTransformerLM:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "config" in ckpt:
        config = DraftModelConfig(**ckpt["config"])
    elif config is None:
        config = DraftModelConfig()

    model = DraftTransformerLM(config)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model
