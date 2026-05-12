import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CORNELL_RED = '#B31B1B'
GREEN = '#2E7D32'
BLUE = '#1565C0'

with open('results/all_results_dump.json') as f:
    data = json.load(f)
ps = data['canonical_prefix_scaling']

records_50 = sorted(
    (v for v in ps['results'].values() if v['steps'] == 50),
    key=lambda r: r['prefix_len'],
)
prefixes = np.array([r['prefix_len'] for r in records_50])
b_mean = np.array([r['baseline']['mean'] for r in records_50])
b_ci = np.array([r['baseline']['ci95'] for r in records_50])
o_mean = np.array([r['optimized']['mean'] for r in records_50])
o_ci = np.array([r['optimized']['ci95'] for r in records_50])
speedups = np.array([r['speedup'] for r in records_50])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.errorbar(prefixes, b_mean, yerr=b_ci, marker='o', capsize=3, lw=2,
             color=CORNELL_RED, label='Baseline (HuggingFace, no KV cache)', markersize=8)
ax1.errorbar(prefixes, o_mean, yerr=o_ci, marker='s', capsize=3, lw=2,
             color=GREEN, label='Optimized (KV cache + fused blocks + Metal kernels)', markersize=8)
ax1.fill_between(prefixes, o_mean, b_mean, alpha=0.10, color=GREEN)
ax1.set_xlabel('Prefix length $L$ (tokens)', fontsize=11)
ax1.set_ylabel('End-to-end wall time (ms, 50 steps)', fontsize=11)
ax1.set_title('Latency vs. prefix length', fontsize=12, fontweight='bold', pad=8)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, prefixes.max() + 30)
ax1.set_ylim(0, b_mean.max() * 1.08)
for x, b, o, s in zip(prefixes, b_mean, o_mean, speedups):
    ax1.annotate(f'{s:.2f}$\\times$', xy=(x, (b + o) / 2), fontsize=9,
                 color=BLUE, fontweight='bold', ha='center')

bar_colors = plt.cm.YlGn(np.linspace(0.35, 0.85, len(prefixes)))
bars = ax2.bar(range(len(prefixes)), speedups, color=bar_colors, edgecolor='black', linewidth=0.8)
ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax2.set_xticks(range(len(prefixes)))
ax2.set_xticklabels([str(p) for p in prefixes])
ax2.set_xlabel('Prefix length $L$ (tokens)', fontsize=11)
ax2.set_ylabel('Speedup vs.\\ baseline', fontsize=11)
ax2.set_title('Speedup grows with context length', fontsize=12, fontweight='bold', pad=8)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(speedups) * 1.15)
for bar, s in zip(bars, speedups):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
             f'{s:.2f}$\\times$', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/headline_scaling.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/headline_scaling.pdf', bbox_inches='tight')
print('Saved figures/headline_scaling.{png,pdf}')
print(f'Speedup range at 50 steps: {speedups.min():.2f}x (L={prefixes[0]}) '
      f'to {speedups.max():.2f}x (L={prefixes[-1]})')

fig2, ax = plt.subplots(figsize=(6, 3.6))
ax.errorbar(prefixes, b_mean, yerr=b_ci, marker='o', capsize=3, lw=2,
            color=CORNELL_RED, label='Baseline (HF, no KV cache)', markersize=8)
ax.errorbar(prefixes, o_mean, yerr=o_ci, marker='s', capsize=3, lw=2,
            color=GREEN, label='Optimized (KV + fused + Metal)', markersize=8)
ax.fill_between(prefixes, o_mean, b_mean, alpha=0.10, color=GREEN)
ax.set_xlabel('Prefix length $L$ (tokens)', fontsize=11)
ax.set_ylabel('End-to-end wall time (ms, 50 steps)', fontsize=11)
ax.set_title('Latency vs.\\ prefix length under the full optimization stack',
             fontsize=12, fontweight='bold', pad=8)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, prefixes.max() + 30)
ax.set_ylim(0, b_mean.max() * 1.08)
for x, b, o, s in zip(prefixes, b_mean, o_mean, speedups):
    ax.annotate(f'{s:.2f}$\\times$', xy=(x, (b + o) / 2), fontsize=9,
                color=BLUE, fontweight='bold', ha='center')
plt.tight_layout()
plt.savefig('figures/headline_scaling_single.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/headline_scaling_single.pdf', bbox_inches='tight')
print('Saved figures/headline_scaling_single.{png,pdf}')
