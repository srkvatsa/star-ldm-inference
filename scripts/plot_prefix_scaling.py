import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CORNELL_RED = '#B31B1B'
GREEN = '#2E7D32'
GRAY = '#777777'

with open('results/phase2_prefix_scaling.json') as f:
    data = json.load(f)

prefixes = sorted(int(k) for k in data.keys())
A_mean = np.array([data[str(p)]['A_jit_only']['mean'] for p in prefixes])
A_std = np.array([data[str(p)]['A_jit_only']['std'] for p in prefixes])
B_mean = np.array([data[str(p)]['B_metal_kernels']['mean'] for p in prefixes])
B_std = np.array([data[str(p)]['B_metal_kernels']['std'] for p in prefixes])
C_mean = np.array([data[str(p)]['C_metal_plus_v_ddpm']['mean'] for p in prefixes])
C_std = np.array([data[str(p)]['C_metal_plus_v_ddpm']['std'] for p in prefixes])
prefixes = np.array(prefixes)

slope, intercept = np.polyfit(prefixes, A_mean, 1)
print(f'Linear fit on A (JIT-only): T(L) = {intercept:.0f} + {slope:.3f} * L  ms')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

ax1.errorbar(prefixes, A_mean, yerr=A_std, marker='o', capsize=3, lw=2,
             color=GRAY, label='A: JIT-only baseline', markersize=7)
ax1.errorbar(prefixes, B_mean, yerr=B_std, marker='s', capsize=3, lw=2,
             color=GREEN, label='B: + Metal kernels (post-wrapper-fix)', markersize=7)
ax1.errorbar(prefixes, C_mean, yerr=C_std, marker='^', capsize=3, lw=2,
             color=CORNELL_RED, label=r'C: + fused $v$+DDPM kernel', markersize=7)
xs = np.linspace(0, prefixes.max() * 1.05, 50)
ax1.plot(xs, intercept + slope * xs, '--', color=GRAY, alpha=0.5, lw=1.2,
         label=fr'Linear fit on A: $T(L)\approx{intercept:.0f}+{slope:.2f}\,L$ ms')
ax1.set_xlabel('Prefix length $L$ (tokens)', fontsize=12)
ax1.set_ylabel('End-to-end wall time (ms, 50 steps)', fontsize=12)
ax1.set_title('Wall time scales linearly with prefix length',
              fontsize=12, fontweight='bold', pad=8)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, prefixes.max() * 1.05)
ax1.set_ylim(700, max(A_mean.max(), B_mean.max(), C_mean.max()) * 1.05)

B_ratio = A_mean / B_mean
C_ratio = A_mean / C_mean
B_ratio_std = B_ratio * np.sqrt((A_std/A_mean)**2 + (B_std/B_mean)**2)
C_ratio_std = C_ratio * np.sqrt((A_std/A_mean)**2 + (C_std/C_mean)**2)

ax2.errorbar(prefixes, B_ratio, yerr=B_ratio_std, marker='s', capsize=3, lw=2,
             color=GREEN, label='B: + Metal kernels', markersize=7)
ax2.errorbar(prefixes, C_ratio, yerr=C_ratio_std, marker='^', capsize=3, lw=2,
             color=CORNELL_RED, label=r'C: + fused $v$+DDPM kernel', markersize=7)
ax2.axhline(1.0, color='black', linestyle='-', alpha=0.5, lw=0.8)
ax2.axhspan(1.02, 1.06, alpha=0.10, color=GREEN, label='Observed band: 1.02--1.06$\\times$')
ax2.set_xlabel('Prefix length $L$ (tokens)', fontsize=12)
ax2.set_ylabel('Speedup vs.\\ JIT-only (A)', fontsize=12)
ax2.set_title('Systems wins are constant in $L$, $\\sim$1.04$\\times$ on average',
              fontsize=12, fontweight='bold', pad=8)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, prefixes.max() * 1.05)
ax2.set_ylim(0.98, 1.10)

B_geomean = float(np.exp(np.mean(np.log(B_ratio))))
C_geomean = float(np.exp(np.mean(np.log(C_ratio))))
print(f'Geometric mean speedup B/A = {B_geomean:.3f}x')
print(f'Geometric mean speedup C/A = {C_geomean:.3f}x')

ax2.text(0.55, 1.085, f'Geom. mean: B={B_geomean:.3f}$\\times$, C={C_geomean:.3f}$\\times$',
         transform=ax2.transAxes, fontsize=10, color='#444',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8',
                   edgecolor='#ccc', alpha=0.9))

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/prefix_scaling.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/prefix_scaling.pdf', bbox_inches='tight')
print('Saved figures/prefix_scaling.{png,pdf}')
