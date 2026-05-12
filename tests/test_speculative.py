
import pytest
import torch
import torch.nn.functional as F

class TestApplySampling:

    def test_temperature_scaling(self):
        from star_ldm.decoding.speculative import _apply_sampling

        logits = torch.tensor([[1.0, 2.0, 3.0]])

        high_temp = _apply_sampling(logits, temperature=10.0, top_p=1.0, repetition_penalty=1.0, prev_tokens=[])
        probs_high = F.softmax(high_temp, dim=-1)

        low_temp = _apply_sampling(logits, temperature=0.1, top_p=1.0, repetition_penalty=1.0, prev_tokens=[])
        probs_low = F.softmax(low_temp, dim=-1)

        assert probs_high.max() < probs_low.max()

    def test_repetition_penalty(self):
        from star_ldm.decoding.speculative import _apply_sampling

        logits = torch.tensor([[5.0, 1.0, 1.0]])

        no_pen = _apply_sampling(logits, temperature=1.0, top_p=1.0, repetition_penalty=1.0, prev_tokens=[])

        with_pen = _apply_sampling(logits.clone(), temperature=1.0, top_p=1.0, repetition_penalty=2.0, prev_tokens=[0])

        assert with_pen[0, 0] < no_pen[0, 0]

    def test_top_p_filtering(self):
        from star_ldm.decoding.speculative import _apply_sampling

        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        result = _apply_sampling(logits, temperature=1.0, top_p=0.5, repetition_penalty=1.0, prev_tokens=[])

        probs = F.softmax(result, dim=-1)

        assert probs[0, 0].item() > 0.99

class TestTruncateKVCache:

    def test_truncate(self):
        from star_ldm.decoding.speculative import _truncate_kv_cache

        past = tuple(
            (torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4))
            for _ in range(2)
        )

        truncated = _truncate_kv_cache(past, current_len=4, keep_len=2)

        for key, value in truncated:
            assert key.shape[2] == 8
            assert value.shape[2] == 8

    def test_truncate_noop(self):
        from star_ldm.decoding.speculative import _truncate_kv_cache

        past = tuple(
            (torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4))
            for _ in range(2)
        )

        result = _truncate_kv_cache(past, current_len=4, keep_len=4)

        assert result is past

class TestVerifyCandidates:

    def test_certain_accept(self):
        from star_ldm.decoding.speculative import _verify_candidates_torch

        K, V = 4, 100
        logits = torch.randn(K, V)
        tokens = torch.argmax(logits, dim=-1)
        rand_uniform = torch.zeros(K)

        reject_idx, _ = _verify_candidates_torch(logits, logits, tokens, rand_uniform)
        assert reject_idx == K

    def test_certain_reject(self):
        from star_ldm.decoding.speculative import _verify_candidates_torch

        K, V = 4, 100
        draft_logits = torch.zeros(K, V)
        draft_logits[:, 0] = 100.0

        target_logits = torch.zeros(K, V)
        target_logits[:, 1] = 100.0

        tokens = torch.zeros(K, dtype=torch.long)
        rand_uniform = torch.ones(K) * 0.5

        reject_idx, adjusted = _verify_candidates_torch(
            draft_logits, target_logits, tokens, rand_uniform
        )
        assert reject_idx == 0

        assert adjusted[1].item() > 0.9
