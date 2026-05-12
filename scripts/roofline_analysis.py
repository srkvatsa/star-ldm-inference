
import argparse
import time
import math
import json
import torch
from contextlib import contextmanager
from omegaconf import OmegaConf

HW = {
    'name': 'Apple M4 Max (40-core GPU)',
    'peak_fp32_gflops': 14_000,
    'mem_bw_gbps': 546,
    'ridge_point': 14_000 / 546,
}

def sync():
    torch.mps.synchronize()

@contextmanager
def timed_us():
    sync()
    t0 = time.perf_counter()
    result = {}
    yield result
    sync()
    result['us'] = (time.perf_counter() - t0) * 1e6

def benchmark(fn, args, warmup=200, iterations=1000):
    for _ in range(warmup):
        fn(*args)
    sync()
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    sync()
    return (time.perf_counter() - t0) / iterations * 1e6

def roofline_entry(name, flops, bytes_accessed, time_us):
    ai = flops / bytes_accessed if bytes_accessed > 0 else 0
    achieved_gflops = flops / time_us / 1e3
    achieved_bw = bytes_accessed / time_us / 1e3

    roofline_ceiling = min(HW['peak_fp32_gflops'], ai * HW['mem_bw_gbps'])
    efficiency = achieved_gflops / roofline_ceiling * 100 if roofline_ceiling > 0 else 0

    return {
        'name': name,
        'time_us': time_us,
        'flops': flops,
        'bytes': bytes_accessed,
        'arithmetic_intensity': ai,
        'achieved_gflops': achieved_gflops,
        'achieved_bw_gbps': achieved_bw,
        'roofline_ceiling_gflops': roofline_ceiling,
        'efficiency_pct': efficiency,
        'bound': 'compute' if ai > HW['ridge_point'] else 'memory',
    }

def analyze_rmsnorm_film(D=1024, L=8, B=1):
    from star_ldm.models.modules.fused_blocks import (
        _jit_fused_rmsnorm_film, metal_rmsnorm_film,
    )

    x = torch.randn(B, L, D, device='mps')
    gamma = torch.randn(D, device='mps')
    dim_scale = math.sqrt(D)
    film_scale = torch.randn(B, 1, D, device='mps')
    film_shift = torch.randn(B, 1, D, device='mps')

    flops = B * L * D * 7

    bytes_rw = (B * L * D + D + B * D + B * D + B * L * D) * 4

    jit_us = benchmark(_jit_fused_rmsnorm_film, (x, gamma, dim_scale, film_scale, film_shift))
    metal_result = metal_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift)

    results = [roofline_entry('RMSNorm+FiLM (JIT)', flops, bytes_rw, jit_us)]
    if metal_result is not None:
        metal_us = benchmark(metal_rmsnorm_film, (x, gamma, dim_scale, film_scale, film_shift))
        results.append(roofline_entry('RMSNorm+FiLM (Metal)', flops, bytes_rw, metal_us))

    return results

def analyze_tiny_attention(B=1, H=16, S=8, D_head=64):
    from star_ldm.models.modules.fused_blocks import (
        _jit_fused_qknorm_attention, metal_tiny_attention,
    )

    q = torch.randn(B, H, S, D_head, device='mps')
    k = torch.randn(B, H, S, D_head, device='mps')
    v = torch.randn(B, H, S, D_head, device='mps')
    q_gamma = torch.randn(D_head, device='mps')
    k_gamma = torch.randn(D_head, device='mps')
    dim_head_scale = math.sqrt(D_head)
    attn_scale = 1.0 / math.sqrt(D_head)

    flops_norm = 2 * B * H * S * D_head * 2
    flops_qk = B * H * S * S * D_head * 2
    flops_softmax = B * H * S * S * 5
    flops_av = B * H * S * D_head * S * 2
    flops = flops_norm + flops_qk + flops_softmax + flops_av

    bytes_rw = (3 * B * H * S * D_head + 2 * D_head + B * H * S * D_head) * 4

    jit_us = benchmark(_jit_fused_qknorm_attention, (q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale))
    results = [roofline_entry('Tiny Attention (JIT)', flops, bytes_rw, jit_us)]

    metal_result = metal_tiny_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale)
    if metal_result is not None:
        metal_us = benchmark(metal_tiny_attention, (q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale))
        results.append(roofline_entry('Tiny Attention (Metal)', flops, bytes_rw, metal_us))

    return results

def analyze_ddpm_step(B=1, D=768):
    from star_ldm.diffusion.fused_ops import _jit_fused_ddpm_step, metal_ddpm_step

    z_t = torch.randn(B, D, device='mps')
    eps = torch.randn(B, D, device='mps')
    noise = torch.randn(B, D, device='mps')
    alpha2 = torch.rand(B, 1, device='mps') * 0.8 + 0.1
    alpha2_next = alpha2 * 0.5
    var_lambda = 0.2

    flops = B * D * 20

    bytes_rw = (3 * B * D + 2 * B + B * D) * 4

    jit_us = benchmark(_jit_fused_ddpm_step, (z_t, eps, noise, alpha2, alpha2_next, var_lambda))
    results = [roofline_entry('DDPM Step (JIT)', flops, bytes_rw, jit_us)]

    metal_result = metal_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda)
    if metal_result is not None:
        metal_us = benchmark(metal_ddpm_step, (z_t, eps, noise, alpha2, alpha2_next, var_lambda))
        results.append(roofline_entry('DDPM Step (Metal)', flops, bytes_rw, metal_us))

    return results

def analyze_gpt2_decode8(model, device):
    prefix_len = 64
    prefix_ids = torch.randint(0, 50257, (1, prefix_len), device=device)
    soft_prompt = torch.randn(1, 8, 1280, device=device, dtype=torch.float32)

    prefix_embed = model.lm_embedding(prefix_ids).to(torch.float32)
    with torch.no_grad():
        prefix_out = model.gpt2(inputs_embeds=prefix_embed, use_cache=True, output_hidden_states=False)
    cached_kv = prefix_out.past_key_values

    def forward_8_tokens():
        with torch.no_grad():
            model.gpt2(inputs_embeds=soft_prompt, past_key_values=cached_kv,
                       use_cache=False, output_hidden_states=True)

    n_layers = 36
    d_model = 1280
    seq_q = 8
    seq_kv = prefix_len + 8
    n_heads = 20
    d_head = d_model // n_heads

    flops_qkv_proj = 3 * seq_q * d_model * d_model * 2
    flops_attn = n_heads * seq_q * seq_kv * d_head * 2
    flops_attn_v = n_heads * seq_q * d_head * seq_kv * 2
    flops_out_proj = seq_q * d_model * d_model * 2
    flops_ffn = seq_q * d_model * 4 * d_model * 2 * 2
    flops_per_layer = flops_qkv_proj + flops_attn + flops_attn_v + flops_out_proj + flops_ffn
    flops = n_layers * flops_per_layer

    bytes_kv_read = n_layers * 2 * seq_kv * d_model * 4
    bytes_params = n_layers * (3 * d_model * d_model + d_model * d_model + 2 * d_model * 4 * d_model) * 4
    bytes_rw = bytes_kv_read + bytes_params

    time_us = benchmark(forward_8_tokens, (), warmup=50, iterations=200)
    return [roofline_entry(f'GPT-2 decode-8 (KV cached, prefix={prefix_len})', flops, bytes_rw, time_us)]

def analyze_soft_prompt_gen(model, device):
    z_t = torch.randn(1, 768, device=device)
    alpha2 = torch.tensor([[0.5]], device=device)

    def spg_forward():
        with torch.no_grad():
            model.soft_prompt_generator(z_t, alpha2)

    d = 1024
    s = 8
    n_layers = 6
    flops_per_layer = (3 * s * d * d * 2
                       + 16 * s * s * 64 * 2
                       + 16 * s * 64 * s * 2
                       + s * d * d * 2
                       + 2 * s * d * d * 4 * 2)
    flops_splicer = 768 * 768 * 4 * 2 + 8 * (768 * 4 // 8) * 1024 * 2
    flops = flops_splicer + n_layers * flops_per_layer

    bytes_rw = n_layers * (4 * d * d + 2 * d * 4 * d) * 4 + 768 * 4 * 4 + 8 * 1024 * 4

    time_us = benchmark(spg_forward, (), warmup=50, iterations=500)
    return [roofline_entry('SoftPromptGenerator (6-layer)', flops, bytes_rw, time_us)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dummy', action='store_true')
    args = parser.parse_args()

    device = torch.device('mps')
    print(f"{'='*70}")
    print(f"  STAR-LDM Roofline Analysis on {HW['name']}")
    print(f"  Peak FP32: {HW['peak_fp32_gflops']} GFLOPS")
    print(f"  Memory BW: {HW['mem_bw_gbps']} GB/s")
    print(f"  Ridge point: {HW['ridge_point']:.1f} FLOP/byte")
    print(f"{'='*70}\n")

    all_results = []

    print("── Kernel-level analysis ──")
    for results in [analyze_rmsnorm_film(), analyze_tiny_attention(), analyze_ddpm_step()]:
        all_results.extend(results)

    if args.model_path or args.dummy:
        print("\n── Model component analysis ──")
        if args.dummy:
            from scripts.bench_e2e_phase2 import make_model as create_dummy_model
            model = create_dummy_model(device)
        else:
            from star_ldm.interface import TransfusionGPTInterface
            interface = TransfusionGPTInterface(model_path=args.model_path, device=str(device))
            model = interface.model

        all_results.extend(analyze_gpt2_decode8(model, device))
        all_results.extend(analyze_soft_prompt_gen(model, device))

    print(f"\n{'='*110}")
    print(f"  {'Operation':<40} {'Time (us)':>10} {'AI (F/B)':>10} {'GFLOPS':>10} {'BW (GB/s)':>10} "
          f"{'Ceiling':>10} {'Eff %':>8} {'Bound':>8}")
    print(f"  {'-'*104}")
    for r in all_results:
        print(f"  {r['name']:<40} {r['time_us']:>10.1f} {r['arithmetic_intensity']:>10.2f} "
              f"{r['achieved_gflops']:>10.2f} {r['achieved_bw_gbps']:>10.2f} "
              f"{r['roofline_ceiling_gflops']:>10.1f} {r['efficiency_pct']:>7.1f}% "
              f"{r['bound']:>8}")
    print()

    print("── Dispatch overhead analysis ──")
    print(f"  At ~0.1ms dispatch overhead per Metal kernel:")
    print(f"    RMSNorm+FiLM: 1200 calls × 0.016ms Metal = {1200 * 0.016:.0f}ms total")
    print(f"    Tiny Attention: 600 calls × 0.042ms Metal = {600 * 0.042:.0f}ms total")
    print(f"    DDPM Step (JIT): 49 calls × 0.040ms JIT = {49 * 0.040:.0f}ms total")
    print(f"    Kernel savings vs JIT: {1200 * (0.027 - 0.016) + 600 * (0.099 - 0.042):.0f}ms")
    print()

    out = {'hardware': HW, 'results': all_results}
    with open('results/roofline.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Results saved to results/roofline.json")

if __name__ == '__main__':
    main()
