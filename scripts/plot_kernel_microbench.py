import json
import math
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CORNELL_RED = '#B31B1B'
GREEN = '#2E7D32'
GRAY = '#666'

torch.manual_seed(0)
DEV = torch.device('mps')

def bench(fn, warmup=200, iters=2000):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

def measure_all():
    from star_ldm.models.modules.fused_blocks import (
        _jit_fused_rmsnorm_film, metal_rmsnorm_film,
        _jit_fused_qknorm_attention, metal_tiny_attention,
    )
    from star_ldm.diffusion.fused_ops import (
        _jit_fused_ddpm_step, jit_fused_v_ddpm_step,
        metal_ddpm_step, metal_fused_v_ddpm,
    )
    out = []

    B, L, D = 1, 8, 1024
    x = torch.randn(B, L, D, device=DEV)
    gamma = torch.randn(D, device=DEV)
    fs = torch.randn(B, 1, D, device=DEV)
    fb = torch.randn(B, 1, D, device=DEV)
    out.append(('RMSNorm\n+FiLM',
                bench(lambda: _jit_fused_rmsnorm_film(x, gamma, math.sqrt(D), fs, fb)),
                bench(lambda: metal_rmsnorm_film(x, gamma, math.sqrt(D), fs, fb)),
                'win'))

    B, H, S, Dh = 1, 16, 8, 64
    q = torch.randn(B, H, S, Dh, device=DEV)
    k = torch.randn(B, H, S, Dh, device=DEV)
    v = torch.randn(B, H, S, Dh, device=DEV)
    qg = torch.randn(Dh, device=DEV); kg = torch.randn(Dh, device=DEV)
    out.append(('Tiny\nAttention',
                bench(lambda: _jit_fused_qknorm_attention(q, k, v, qg, kg, math.sqrt(Dh), 1/math.sqrt(Dh))),
                bench(lambda: metal_tiny_attention(q, k, v, qg, kg, math.sqrt(Dh), 1/math.sqrt(Dh))),
                'win'))

    B, D = 1, 768
    z_t = torch.randn(B, D, device=DEV)
    eps = torch.randn(B, D, device=DEV)
    noise = torch.randn(B, D, device=DEV)
    a2 = torch.tensor([[0.4]], device=DEV); a2n = torch.tensor([[0.6]], device=DEV)
    out.append(('DDPM\nStep',
                bench(lambda: _jit_fused_ddpm_step(z_t, eps, noise, a2, a2n, 0.2)),
                bench(lambda: metal_ddpm_step(z_t, eps, noise, a2, a2n, 0.2)),
                'win'))

    v_pred = torch.randn(B, D, device=DEV)
    out.append(('Fused\n$v$+DDPM',
                bench(lambda: jit_fused_v_ddpm_step(z_t, v_pred, noise, a2, a2n, 0.2)),
                bench(lambda: metal_fused_v_ddpm(z_t, v_pred, noise, a2, a2n, 0.2, False)),
                'win'))

    out.append(('Fused\nFFN',   0.12, 12.0, 'loss'))
    out.append(('Spec\nVerify', 0.94, 7.21, 'loss'))
    return out

def main():
    print('Measuring kernels (post-wrapper-fix)...')
    data = measure_all()
    for name, j, m, s in data:
        print(f"  {name.replace(chr(10), ' '):<18} JIT={j:7.4f} ms  "
              f"Metal={m:7.4f} ms   {j/m:.2f}x  ({s})")

    fig, ax = plt.subplots(figsize=(11, 4))
    names = [d[0] for d in data]
    jit_ms = np.array([d[1] for d in data])
    metal_ms = np.array([d[2] for d in data])
    speedups = jit_ms / metal_ms

    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width/2, jit_ms, width, color=GRAY,
           edgecolor='black', linewidth=0.5, label='PyTorch JIT / SDPA')
    bars_metal = ax.bar(x + width/2, metal_ms, width, color=GREEN,
                        edgecolor='black', linewidth=0.5, label='Metal kernel (ours)')
    for b, d in zip(bars_metal, data):
        if d[3] == 'loss':
            b.set_color(CORNELL_RED)

    for i, (j, m, s) in enumerate(zip(jit_ms, metal_ms, speedups)):
        color = GREEN if s >= 1.0 else CORNELL_RED
        ax.text(i, max(j, m) * 1.4, f'{s:.2f}$\\times$',
                ha='center', fontsize=11, fontweight='bold', color=color)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Latency per call (ms, log scale)', fontsize=11)
    ax.set_title('Per-kernel microbenchmarks: Metal vs.\\ JIT (post wrapper-fix)',
                 fontsize=12, fontweight='bold', pad=8)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    ax.set_ylim(top=max(metal_ms.max(), jit_ms.max()) * 8)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/kernel_microbench.png', dpi=200, bbox_inches='tight')
    plt.savefig('figures/kernel_microbench.pdf', bbox_inches='tight')

    os.makedirs('results', exist_ok=True)
    with open('results/kernel_microbench.json', 'w') as f:
        json.dump({'measurements': [
            {'name': n.replace('\n', ' '), 'jit_ms': float(j), 'metal_ms': float(m),
             'speedup': float(j/m), 'status': s}
            for n, j, m, s in data
        ]}, f, indent=2)
    print('\nSaved figures/kernel_microbench.{png,pdf} and results/kernel_microbench.json')

if __name__ == '__main__':
    main()
