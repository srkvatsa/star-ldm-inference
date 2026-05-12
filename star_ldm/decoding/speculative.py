
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

_metal_verify = None

def _get_metal_verify():
    global _metal_verify
    if _metal_verify is None:
        try:
            from star_ldm.kernels import get_spec_verify_kernel
            _metal_verify = get_spec_verify_kernel()
        except Exception:
            _metal_verify = False
    return _metal_verify if _metal_verify is not False else None

def _verify_candidates_torch(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    rand_uniform: torch.Tensor,
) -> Tuple[int, torch.Tensor]:
    K, V = draft_logits.shape

    p_draft = F.softmax(draft_logits, dim=-1)
    p_target = F.softmax(target_logits, dim=-1)

    first_reject_idx = K

    for k in range(K):
        token = draft_tokens[k].item()
        pd = p_draft[k, token].item()
        pt = p_target[k, token].item()

        accept_prob = min(1.0, pt / pd) if pd > 0 else 0.0

        if rand_uniform[k].item() >= accept_prob:
            first_reject_idx = k
            break

    if first_reject_idx < K:
        adjusted = torch.clamp(p_target[first_reject_idx] - p_draft[first_reject_idx], min=0.0)
        adj_sum = adjusted.sum()
        if adj_sum > 0:
            adjusted = adjusted / adj_sum
        else:

            adjusted = p_target[first_reject_idx]
    else:
        adjusted = torch.zeros(V, device=draft_logits.device)

    return first_reject_idx, adjusted

def _verify_candidates(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    rand_uniform: torch.Tensor,
) -> Tuple[int, torch.Tensor]:
    metal_mod = _get_metal_verify()

    if metal_mod is not None and draft_logits.is_mps:
        try:
            first_reject_t, adjusted = metal_mod.speculative_verify(
                draft_logits.contiguous(),
                target_logits.contiguous(),
                draft_tokens.contiguous(),
                rand_uniform.contiguous(),
            )

            first_reject_idx = first_reject_t.item()
            return first_reject_idx, adjusted
        except Exception:
            pass

    return _verify_candidates_torch(draft_logits, target_logits, draft_tokens, rand_uniform)

@torch.no_grad()
def speculative_generate(
    draft_model,
    target_model,
    input_embeds: torch.Tensor,
    max_new_tokens: int = 32,
    K: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    input_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = input_embeds.device
    batch_size = input_embeds.size(0)
    assert batch_size == 1, "Speculative decoding currently supports batch_size=1"

    if hasattr(draft_model, 'transformer'):
        draft_embed_layer = draft_model.transformer.wte
    elif hasattr(draft_model, 'model'):
        draft_embed_layer = draft_model.model.embed_tokens
    else:
        raise ValueError("Cannot find embedding layer in draft model")

    if hasattr(target_model, 'transformer'):
        target_embed_layer = target_model.transformer.wte
    elif hasattr(target_model, 'model'):
        target_embed_layer = target_model.model.embed_tokens
    else:
        raise ValueError("Cannot find embedding layer in target model")

    draft_embed_dim = draft_embed_layer.weight.shape[1]
    target_embed_dim = target_embed_layer.weight.shape[1]

    if draft_embed_dim != target_embed_dim and input_ids is not None:
        draft_out = draft_model(input_ids=input_ids, use_cache=True)
    else:
        draft_out = draft_model(inputs_embeds=input_embeds, use_cache=True)
    draft_past = draft_out.past_key_values

    target_out = target_model(inputs_embeds=input_embeds, use_cache=True)
    target_past = target_out.past_key_values

    generated_tokens = []
    total_generated = 0
    total_accepted = 0
    total_rounds = 0

    draft_next_logits = draft_out.logits[:, -1, :]

    while total_generated < max_new_tokens:
        total_rounds += 1
        k_actual = min(K, max_new_tokens - total_generated)

        draft_tokens = []
        draft_logits_list = []

        current_draft_past = draft_past
        current_logits = draft_next_logits

        for i in range(k_actual):

            logits = _apply_sampling(
                current_logits, temperature, top_p, repetition_penalty,
                generated_tokens
            )
            draft_logits_list.append(logits.squeeze(0))

            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(token.squeeze())

            token_embed = draft_embed_layer(token)
            draft_step = draft_model(
                inputs_embeds=token_embed,
                past_key_values=current_draft_past,
                use_cache=True,
            )
            current_draft_past = draft_step.past_key_values
            current_logits = draft_step.logits[:, -1, :]

        draft_tokens_t = torch.stack(draft_tokens)
        draft_logits_t = torch.stack(draft_logits_list)

        all_tokens = draft_tokens_t.unsqueeze(0)
        all_embeds = target_embed_layer(all_tokens)

        target_step = target_model(
            inputs_embeds=all_embeds,
            past_key_values=target_past,
            use_cache=True,
        )

        target_logits_t = target_step.logits.squeeze(0)

        target_logits_processed = torch.stack([
            _apply_sampling(
                target_logits_t[i:i+1], temperature, top_p, repetition_penalty,
                generated_tokens + [t.item() for t in draft_tokens[:i]]
            ).squeeze(0)
            for i in range(k_actual)
        ])

        rand_uniform = torch.rand(k_actual, device=device)
        first_reject, adjusted_probs = _verify_candidates(
            draft_logits_t, target_logits_processed, draft_tokens_t, rand_uniform
        )

        n_accepted = first_reject
        for i in range(n_accepted):
            generated_tokens.append(draft_tokens[i].item())
            total_generated += 1

        total_accepted += n_accepted

        if first_reject < k_actual:

            replacement = torch.multinomial(adjusted_probs.unsqueeze(0), num_samples=1)
            generated_tokens.append(replacement.squeeze().item())
            total_generated += 1

            n_keep = first_reject + 1
            target_past = _truncate_kv_cache(target_step.past_key_values, k_actual, n_keep)

            accepted_plus_replacement = (
                [t.item() for t in draft_tokens[:first_reject]]
                + [replacement.squeeze().item()]
            )
            if accepted_plus_replacement:
                tokens_t = torch.tensor([accepted_plus_replacement], device=device)
                embeds = draft_embed_layer(tokens_t)
                draft_step = draft_model(
                    inputs_embeds=embeds,
                    past_key_values=draft_past,
                    use_cache=True,
                )
                draft_past = draft_step.past_key_values
                draft_next_logits = draft_step.logits[:, -1, :]
            else:

                token_t = replacement.view(1, 1)
                embeds = draft_embed_layer(token_t)
                draft_step = draft_model(
                    inputs_embeds=embeds,
                    past_key_values=draft_past,
                    use_cache=True,
                )
                draft_past = draft_step.past_key_values
                draft_next_logits = draft_step.logits[:, -1, :]
        else:

            bonus_logits = _apply_sampling(
                target_step.logits[:, -1, :], temperature, top_p,
                repetition_penalty, generated_tokens
            )
            bonus_probs = F.softmax(bonus_logits, dim=-1)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1)
            generated_tokens.append(bonus_token.squeeze().item())
            total_generated += 1

            target_past = target_step.past_key_values

            bonus_embed = target_embed_layer(bonus_token)
            target_bonus = target_model(
                inputs_embeds=bonus_embed,
                past_key_values=target_past,
                use_cache=True,
            )
            target_past = target_bonus.past_key_values

            all_accepted = [t.item() for t in draft_tokens] + [bonus_token.squeeze().item()]
            tokens_t = torch.tensor([all_accepted], device=device)
            embeds = draft_embed_layer(tokens_t)
            draft_step = draft_model(
                inputs_embeds=embeds,
                past_key_values=draft_past,
                use_cache=True,
            )
            draft_past = draft_step.past_key_values
            draft_next_logits = draft_step.logits[:, -1, :]

        if eos_token_id is not None and generated_tokens[-1] == eos_token_id:
            break

    acceptance_rate = total_accepted / (total_rounds * K) if total_rounds > 0 else 0.0

    result = torch.tensor([generated_tokens], device=device)
    return result, acceptance_rate

def _apply_sampling(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    prev_tokens: list,
) -> torch.Tensor:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if repetition_penalty != 1.0 and prev_tokens:
        prev_ids = torch.tensor(prev_tokens, device=logits.device).unique()
        for token_id in prev_ids:
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

    if temperature != 1.0:
        logits = logits / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    return logits

def _truncate_kv_cache(
    past_key_values,
    current_len: int,
    keep_len: int,
):
    if keep_len >= current_len:
        return past_key_values

    n_remove = current_len - keep_len

    if hasattr(past_key_values, 'crop'):
        seq_len = past_key_values.get_seq_length()
        past_key_values.crop(seq_len - n_remove)
        return past_key_values

    truncated = []
    for layer_kv in past_key_values:
        key = layer_kv[0]
        value = layer_kv[1]
        truncated.append((
            key[:, :, :-n_remove, :],
            value[:, :, :-n_remove, :],
        ))
    return tuple(truncated)
