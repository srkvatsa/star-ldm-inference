import json
import math
import os
import time

import numpy as np
import torch
from torch import Tensor

torch.manual_seed(0)
DEV = torch.device('mps')

def sync():
    torch.mps.synchronize()

def bench(fn, warmup=200, iters=2000):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e6

def calibrate_eager_op(D=768, ks=(1, 2, 4, 8, 16, 32, 64)):
    a = torch.randn(1, D, device=DEV)
    b = torch.randn(1, D, device=DEV)
    times = {}
    for K in ks:
        def f(K=K):
            x = a + b
            for _ in range(K - 1):
                x = x + b
            return x
        times[K] = bench(f, warmup=100, iters=1500)
    Ks = np.array(list(times.keys()), dtype=float)
    Ts = np.array([times[k] for k in ks], dtype=float)
    slope, intercept = np.polyfit(Ks, Ts, 1)
    return slope, intercept, times

@torch.jit.script
def _jit_chain_1(a: Tensor, b: Tensor) -> Tensor:
    return a + b

@torch.jit.script
def _jit_chain_2(a: Tensor, b: Tensor) -> Tensor:
    x = a + b; x = x + b
    return x

@torch.jit.script
def _jit_chain_4(a: Tensor, b: Tensor) -> Tensor:
    x = a + b; x = x + b; x = x + b; x = x + b
    return x

@torch.jit.script
def _jit_chain_8(a: Tensor, b: Tensor) -> Tensor:
    x = a + b
    x = x + b; x = x + b; x = x + b
    x = x + b; x = x + b; x = x + b; x = x + b
    return x

@torch.jit.script
def _jit_chain_16(a: Tensor, b: Tensor) -> Tensor:
    x = a + b
    for _ in range(15):
        x = x + b
    return x

@torch.jit.script
def _jit_chain_32(a: Tensor, b: Tensor) -> Tensor:
    x = a + b
    for _ in range(31):
        x = x + b
    return x

@torch.jit.script
def _jit_chain_64(a: Tensor, b: Tensor) -> Tensor:
    x = a + b
    for _ in range(63):
        x = x + b
    return x

_JIT_CHAINS = {1: _jit_chain_1, 2: _jit_chain_2, 4: _jit_chain_4, 8: _jit_chain_8,
               16: _jit_chain_16, 32: _jit_chain_32, 64: _jit_chain_64}

def calibrate_jit_op(D=768, ks=(1, 2, 4, 8, 16, 32, 64)):
    a = torch.randn(1, D, device=DEV)
    b = torch.randn(1, D, device=DEV)
    times = {}
    for K in ks:
        fn = _JIT_CHAINS[K]
        for _ in range(50):
            fn(a, b)
        times[K] = bench(lambda fn=fn: fn(a, b), warmup=100, iters=1500)
    Ks = np.array(list(times.keys()), dtype=float)
    Ts = np.array([times[k] for k in ks], dtype=float)
    slope, intercept = np.polyfit(Ks, Ts, 1)
    return slope, intercept, times

def calibrate_metal_launch():
    from star_ldm.kernels import get_rmsnorm_film_kernel
    mod = get_rmsnorm_film_kernel()
    if mod is None:
        return None
    D = 64
    x = torch.randn(1, 1, D, device=DEV)
    gamma = torch.randn(D, device=DEV)
    fs = torch.randn(1, 1, D, device=DEV)
    fb = torch.randn(1, 1, D, device=DEV)
    return bench(
        lambda: mod.rmsnorm_film(x.contiguous(), gamma.contiguous(), float(math.sqrt(D)), fs, fb),
        warmup=200, iters=2000,
    )

def measure_existing_kernels():
    from star_ldm.diffusion.fused_ops import _jit_fused_ddpm_step, jit_fused_v_ddpm_step
    from star_ldm.models.modules.fused_blocks import (
        _jit_fused_rmsnorm_film, metal_rmsnorm_film,
        _jit_fused_qknorm_attention, metal_tiny_attention,
    )
    from star_ldm.kernels import get_ddpm_step_kernel, get_fused_v_ddpm_kernel
    ddpm_metal = get_ddpm_step_kernel()
    v_ddpm_metal = get_fused_v_ddpm_kernel()

    out = {}

    B, L, D = 1, 8, 1024
    x = torch.randn(B, L, D, device=DEV)
    gamma = torch.randn(D, device=DEV)
    fs = torch.randn(B, 1, D, device=DEV)
    fb = torch.randn(B, 1, D, device=DEV)
    out['RMSNorm+FiLM'] = {
        'jit_us': bench(lambda: _jit_fused_rmsnorm_film(x, gamma, math.sqrt(D), fs, fb)),
        'metal_us': bench(lambda: metal_rmsnorm_film(x, gamma, math.sqrt(D), fs, fb)),
        'K_eff_unfused': 6, 'has_reduction': True,
        'M_bytes': 4 * (B*L*D + D + B*D + B*D + B*L*D),
    }

    B, H, S, Dh = 1, 16, 8, 64
    q = torch.randn(B, H, S, Dh, device=DEV)
    k = torch.randn(B, H, S, Dh, device=DEV)
    v = torch.randn(B, H, S, Dh, device=DEV)
    qg = torch.randn(Dh, device=DEV); kg = torch.randn(Dh, device=DEV)
    out['TinyAttention'] = {
        'jit_us': bench(lambda: _jit_fused_qknorm_attention(q, k, v, qg, kg, math.sqrt(Dh), 1/math.sqrt(Dh))),
        'metal_us': bench(lambda: metal_tiny_attention(q, k, v, qg, kg, math.sqrt(Dh), 1/math.sqrt(Dh))),
        'K_eff_unfused': 9, 'has_reduction': True,
        'M_bytes': 4 * (3*B*H*S*Dh + 2*Dh + B*H*S*Dh),
    }

    B, D = 1, 768
    z_t = torch.randn(B, D, device=DEV)
    eps = torch.randn(B, D, device=DEV)
    noise = torch.randn(B, D, device=DEV)
    a2 = torch.tensor([[0.4]], device=DEV)
    a2_n = torch.tensor([[0.6]], device=DEV)
    out['DDPMStep'] = {
        'jit_us': bench(lambda: _jit_fused_ddpm_step(z_t, eps, noise, a2, a2_n, 0.2)),
        'metal_us': bench(lambda: ddpm_metal.ddpm_step(
            z_t.contiguous(), eps.contiguous(), noise.contiguous(),
            a2.contiguous(), a2_n.contiguous(), 0.2)),
        'K_eff_unfused': 17, 'has_reduction': False,
        'M_bytes': 4 * (3*B*D + 2*B + B*D),
    }

    v = torch.randn(B, D, device=DEV)
    out['FusedV+DDPM'] = {
        'jit_us': bench(lambda: jit_fused_v_ddpm_step(z_t, v, noise, a2, a2_n, 0.2)),
        'metal_us': bench(lambda: v_ddpm_metal.fused_v_ddpm(
            z_t.contiguous(), v.contiguous(), noise.contiguous(),
            a2.contiguous(), a2_n.contiguous(), 0.2, False)),
        'K_eff_unfused': 23, 'has_reduction': False,
        'M_bytes': 4 * (3*B*D + 2*B + B*D),
    }

    return out

def plot(eager_times, jit_times, eager_slope, eager_intercept,
         jit_slope, jit_intercept, metal_launch, pred_records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    Ks_e = np.array(list(eager_times.keys()))
    Ts_e = np.array([eager_times[k] for k in Ks_e])
    Ks_j = np.array(list(jit_times.keys()))
    Ts_j = np.array([jit_times[k] for k in Ks_j])
    ax1.scatter(Ks_e, Ts_e, color='#B31B1B', s=70, label='eager', zorder=4)
    ax1.scatter(Ks_j, Ts_j, color='#1565C0', s=70, label='torch.jit.script', zorder=4)
    x_fit = np.linspace(0, max(Ks_e.max(), Ks_j.max()) + 4, 50)
    ax1.plot(x_fit, eager_intercept + eager_slope * x_fit, '--',
             color='#B31B1B', alpha=0.7, label=f'eager: {eager_slope:.1f}us·K + {eager_intercept:.1f}us')
    ax1.plot(x_fit, jit_intercept + jit_slope * x_fit, '--',
             color='#1565C0', alpha=0.7, label=f'jit:   {jit_slope:.1f}us·K + {jit_intercept:.1f}us')
    ax1.axhline(metal_launch, color='#2E7D32', linestyle=':', linewidth=2,
                label=f'Metal launch floor: {metal_launch:.0f}us')
    ax1.set_xlabel('Number of pointwise ops in chain (K)', fontsize=12)
    ax1.set_ylabel('Wall time per call (μs)', fontsize=12)
    ax1.set_title('M4 Max dispatch-cost calibration\n(D=768 floats, B=1)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(Ks_e.max(), Ks_j.max()) + 2)
    ax1.set_ylim(0, None)

    names = [r['kernel'] for r in pred_records]
    x_pos = np.arange(len(names))
    width = 0.2
    ax2.bar(x_pos - 1.5*width, [r['pred_jit_us'] for r in pred_records], width,
            color='#90CAF9', label='predicted JIT')
    ax2.bar(x_pos - 0.5*width, [r['obs_jit_us'] for r in pred_records], width,
            color='#1565C0', label='observed JIT')
    ax2.bar(x_pos + 0.5*width, [r['pred_metal_us'] for r in pred_records], width,
            color='#A5D6A7', label='predicted Metal')
    ax2.bar(x_pos + 1.5*width, [r['obs_metal_us'] for r in pred_records], width,
            color='#2E7D32', label='observed Metal')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax2.set_ylabel('Wall time per call (μs)', fontsize=12)
    ax2.set_title('Crossover model: predictions vs observations\n(✓ Metal wins where K_eff·T_jit > T_metal)',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.25, axis='y')

    for i, r in enumerate(pred_records):
        winner = 'Metal wins' if r['metal_wins'] else 'JIT wins'
        color = '#2E7D32' if r['metal_wins'] else '#1565C0'
        ax2.text(i, max(r['obs_jit_us'], r['obs_metal_us']) * 1.05, winner,
                 ha='center', fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/fusion_crossover.png', dpi=200, bbox_inches='tight')
    plt.savefig('figures/fusion_crossover.pdf', bbox_inches='tight')

def main():
    print('Calibrating eager op overhead...')
    eager_slope, eager_intercept, eager_times = calibrate_eager_op()
    print(f'  T_eager_op = {eager_slope:.2f} us/op   intercept = {eager_intercept:.2f} us')

    print('Calibrating JIT op overhead...')
    jit_slope, jit_intercept, jit_times = calibrate_jit_op()
    print(f'  T_jit_op   = {jit_slope:.2f} us/op   intercept = {jit_intercept:.2f} us')

    print('Calibrating Metal launch...')
    metal_launch = calibrate_metal_launch()
    print(f'  T_metal_launch = {metal_launch:.2f} us')

    print('\nMeasuring real kernels...')
    kdata = measure_existing_kernels()
    for n, d in kdata.items():
        print(f'  {n:<18} JIT={d["jit_us"]:6.1f} us  Metal={d["metal_us"]:6.1f} us  '
              f'K_eff={d["K_eff_unfused"]}  has_reduction={d["has_reduction"]}')

    pred_records = []
    print('\nFusion crossover predictions vs observation:')
    print(f'{"kernel":<18} {"K_eff":>5} {"reduction":>10} {"pred Metal":>12} '
          f'{"obs Metal":>11} {"pred JIT":>10} {"obs JIT":>9} {"M wins?":>9}')
    for n, d in kdata.items():
        K = d['K_eff_unfused']
        K_jit = K if d['has_reduction'] else 1
        pred_jit = jit_intercept + K_jit * jit_slope
        bw_us = (d['M_bytes'] / 1e9) / 546 * 1e6
        pred_metal = metal_launch + bw_us
        m_wins = d['metal_us'] < d['jit_us']
        print(f'{n:<18} {K:>5d} {str(d["has_reduction"]):>10} {pred_metal:>12.1f} '
              f'{d["metal_us"]:>11.1f} {pred_jit:>10.1f} {d["jit_us"]:>9.1f} '
              f'{"yes" if m_wins else "no":>9}')
        pred_records.append({
            'kernel': n, 'K_eff': K, 'has_reduction': d['has_reduction'],
            'pred_metal_us': pred_metal, 'obs_metal_us': d['metal_us'],
            'pred_jit_us': pred_jit, 'obs_jit_us': d['jit_us'],
            'metal_wins': m_wins,
        })

    os.makedirs('results', exist_ok=True)
    with open('results/fusion_crossover.json', 'w') as f:
        json.dump({
            'hardware': {'name': 'Apple M4 Max', 'peak_bw_GBps': 546, 'peak_fp32_GFLOPS': 14_000},
            'eager_slope_us': eager_slope, 'eager_intercept_us': eager_intercept,
            'jit_slope_us': jit_slope, 'jit_intercept_us': jit_intercept,
            'metal_launch_us': metal_launch,
            'eager_chain_times_us': eager_times,
            'jit_chain_times_us': jit_times,
            'kernels': kdata,
            'predictions': pred_records,
        }, f, indent=2)
    print('\nSaved results/fusion_crossover.json')

    plot(eager_times, jit_times, eager_slope, eager_intercept,
         jit_slope, jit_intercept, metal_launch, pred_records)
    print('Saved figures/fusion_crossover.{png,pdf}')

if __name__ == '__main__':
    main()
