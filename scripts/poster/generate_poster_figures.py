import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('figures/poster', exist_ok=True)

plt.rcParams.update({
    'font.size': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.dpi': 300,
    'axes.linewidth': 1.5,
    'lines.linewidth': 3,
    'lines.markersize': 10,
})

CORNELL_RED = '#B31B1B'
GREEN = '#2E7D32'
BLUE = '#1565C0'
ORANGE = '#E65100'
GRAY = '#607D8B'
LIGHT_GREEN = '#C8E6C9'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.4, 1]})

prefixes = [16, 64, 128, 256, 512]
baseline = [1707, 2235, 2876, 4389, 7361]
optimized = [1235, 1227, 1281, 1347, 1492]
speedups = [b/o for b, o in zip(baseline, optimized)]
base_ci = [27, 34, 23, 26, 27]
opt_ci = [16, 8, 13, 10, 9]

ax1.errorbar(prefixes, baseline, yerr=base_ci, fmt='o-', color=CORNELL_RED,
             linewidth=3, markersize=12, capsize=6, capthick=2, label='Baseline (HuggingFace)', zorder=5)
ax1.errorbar(prefixes, optimized, yerr=opt_ci, fmt='s-', color=GREEN,
             linewidth=3, markersize=12, capsize=6, capthick=2, label='Optimized (this work)', zorder=5)
ax1.fill_between(prefixes, optimized, baseline, alpha=0.12, color=GREEN)

for x, b, o, s in zip(prefixes, baseline, optimized, speedups):
    ax1.annotate(f'{s:.1f}x', xy=(x, (b+o)/2), fontsize=16, ha='center',
                fontweight='bold', color=BLUE,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=BLUE, alpha=0.8))

ax1.set_xlabel('Prefix Length (tokens)', fontsize=22)
ax1.set_ylabel('End-to-End Latency (ms)', fontsize=22)
ax1.set_title('Latency vs. Prefix Length\n(50 diffusion steps, C4 validation)', fontsize=22)
ax1.legend(loc='upper left', fontsize=16, framealpha=0.9)
ax1.set_xlim(-20, 550)
ax1.set_ylim(0, 8500)
ax1.grid(True, alpha=0.25, linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

bars = ax2.bar(range(len(prefixes)), speedups, color=BLUE, alpha=0.85,
               edgecolor='white', linewidth=2, width=0.65)
ax2.set_xticks(range(len(prefixes)))
ax2.set_xticklabels([str(p) for p in prefixes], fontsize=16)
ax2.set_xlabel('Prefix Length (tokens)', fontsize=22)
ax2.set_ylabel('Speedup (×)', fontsize=22)
ax2.set_title('Speedup Scales with\nContext Length', fontsize=22)
ax2.set_ylim(0, 5.8)
ax2.grid(True, alpha=0.25, axis='y', linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for i, s in enumerate(speedups):
    ax2.text(i, s + 0.12, f'{s:.2f}×', ha='center', fontweight='bold', fontsize=18, color=BLUE)

plt.tight_layout(w_pad=3)
plt.savefig('figures/poster/prefix_scaling.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/poster/prefix_scaling.pdf', bbox_inches='tight')
plt.close()
print("Saved prefix_scaling")

fig, ax = plt.subplots(figsize=(10, 9))
labels = ['GPT-2 forward\n(diffusion loop)\n38.7%', 'GPT-2 generate\n(AR decode)\n38.5%',
          'Soft Prompt\nGenerator\n9.1%', 'Score Net\nHead\n9.2%', 'Other\n4.5%']
sizes = [38.7, 38.5, 9.1, 9.2, 4.5]
colors = [CORNELL_RED, '#E57373', BLUE, GREEN, '#BDBDBD']
explode = (0.04, 0.04, 0, 0, 0)
wedges, texts = ax.pie(sizes, labels=None, autopct=None,
                        colors=colors, explode=explode,
                        startangle=90, pctdistance=0.75,
                        wedgeprops=dict(linewidth=2, edgecolor='white'))

ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(0.85, 0.5),
         fontsize=14, frameon=False)

ax.set_title('Runtime Breakdown\n(50 C4 prompts, unoptimized)', fontsize=24, pad=20)

ax.text(0, 0, 'GPT-2\n77.2%', ha='center', va='center',
       fontsize=28, fontweight='bold', color=CORNELL_RED)

plt.tight_layout()
plt.savefig('figures/poster/profiling_pie.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/poster/profiling_pie.pdf', bbox_inches='tight')
plt.close()
print("Saved profiling_pie")

fig, ax = plt.subplots(figsize=(10, 7))

categories = ['HuggingFace\nGPT-2 Forward', 'Streamlined\nForward (ours)']
counts = [6185, 432]
colors_bar = [CORNELL_RED, GREEN]
bars = ax.bar(categories, counts, color=colors_bar, width=0.5,
              edgecolor='white', linewidth=2.5)

ax.set_ylabel('ATen Operator Dispatches\nper Call', fontsize=20)
ax.set_title('Framework Overhead:\nOperator Dispatch Count', fontsize=24)

ax.text(0, 6185 + 200, '6,185', ha='center', fontsize=28, fontweight='bold', color=CORNELL_RED)
ax.text(1, 432 + 200, '432', ha='center', fontsize=28, fontweight='bold', color=GREEN)

ax.annotate('', xy=(0.85, 3500), xytext=(0.15, 3500),
           arrowprops=dict(arrowstyle='->', color=BLUE, lw=3))
ax.text(0.5, 3800, '93% reduction', ha='center', fontsize=20,
       fontweight='bold', color=BLUE,
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor=BLUE, linewidth=2))

ax.set_ylim(0, 7800)
ax.grid(True, alpha=0.2, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/poster/dispatch_count.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/poster/dispatch_count.pdf', bbox_inches='tight')
plt.close()
print("Saved dispatch_count")

fig, ax = plt.subplots(figsize=(14, 9))

peak_gflops = 14000
mem_bw = 546
ridge = peak_gflops / mem_bw

ai_range = np.logspace(-2, 3, 500)
roofline = np.minimum(peak_gflops, ai_range * mem_bw)
ax.loglog(ai_range, roofline, 'k-', linewidth=3, label='M4 Max Roofline', zorder=3)
ax.axvline(x=ridge, color='gray', linestyle='--', alpha=0.4, linewidth=1)

points = [
    ('RMSNorm+FiLM (JIT)', 0.74, 2.1, GRAY, 'o'),
    ('RMSNorm+FiLM (Metal)', 0.74, 4.5, GREEN, 's'),
    ('Tiny Attn (JIT)', 2.28, 3.2, GRAY, 'o'),
    ('Tiny Attn (Metal)', 2.28, 7.4, GREEN, 's'),
    ('DDPM Step (JIT)', 1.25, 0.37, GRAY, 'o'),
    ('GPT-2 decode-8', 4.0, 120, CORNELL_RED, 'D'),
    ('SoftPromptGen', 4.04, 440, BLUE, '^'),
]

for name, ai, gflops, color, marker in points:
    ax.plot(ai, gflops, marker, color=color, markersize=16, markeredgecolor='white',
            markeredgewidth=2, zorder=5)
    offset_x, offset_y = 15, 5
    if gflops < 1: offset_y = -20
    if 'GPT' in name: offset_x = -15; offset_y = -25
    if 'Soft' in name: offset_x = 15; offset_y = 10
    ax.annotate(name, xy=(ai, gflops), xytext=(offset_x, offset_y),
               textcoords='offset points', fontsize=13, ha='left',
               fontweight='bold' if color != GRAY else 'normal')

ax.axhspan(0.01, 10, alpha=0.06, color='red', zorder=1)
ax.text(0.013, 0.08, 'Dispatch-latency-bound\nregime (<1% ceiling)',
        fontsize=14, fontstyle='italic', color=CORNELL_RED, fontweight='bold')

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=22)
ax.set_ylabel('Achieved Throughput (GFLOP/s)', fontsize=22)
ax.set_title('Roofline Analysis: STAR-LDM on M4 Max', fontsize=24)
ax.set_xlim(0.01, 1000)
ax.set_ylim(0.03, 20000)
ax.grid(True, alpha=0.15, which='both')

legend_elements = [
    mpatches.Patch(color=GREEN, label='Metal kernel'),
    mpatches.Patch(color=GRAY, label='JIT / PyTorch'),
    mpatches.Patch(color=BLUE, label='Full module'),
    mpatches.Patch(color=CORNELL_RED, label='GPT-2 backbone'),
    plt.Line2D([0],[0], color='k', linewidth=3, label='Hardware roofline'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=14, framealpha=0.9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/poster/roofline.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/poster/roofline.pdf', bbox_inches='tight')
plt.close()
print("Saved roofline")

fig, ax = plt.subplots(figsize=(14, 7))

kernels = ['RMSNorm\n+FiLM', 'Tiny\nAttention', 'DDPM\nStep', 'Decode-N\nAttn', 'Fused\nFFN', 'Spec\nVerify']
jit_ms = [0.027, 0.099, 0.040, 0.021, 0.124, 0.939]
metal_ms = [0.016, 0.042, 0.115, 0.011, 15.77, 7.214]
speedups_k = [j/m for j, m in zip(jit_ms, metal_ms)]
colors_k = [GREEN if s > 1 else CORNELL_RED for s in speedups_k]

x = np.arange(len(kernels))
w = 0.32
b1 = ax.bar(x - w/2, jit_ms, w, label='PyTorch JIT / SDPA', color=GRAY, edgecolor='white', linewidth=1.5)
b2 = ax.bar(x + w/2, metal_ms, w, label='Metal Kernel (ours)', color=colors_k, edgecolor='white', linewidth=1.5)

ax.set_ylabel('Latency (ms)', fontsize=20)
ax.set_title('Metal Kernel Microbenchmarks', fontsize=24)
ax.set_xticks(x)
ax.set_xticklabels(kernels, fontsize=14)
ax.legend(fontsize=16, loc='upper left')
ax.set_yscale('log')
ax.set_ylim(0.005, 30)
ax.grid(True, alpha=0.2, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, s in enumerate(speedups_k):
    color = GREEN if s > 1 else CORNELL_RED
    label = f'{s:.1f}×' if s >= 1 else f'{s:.2f}×'
    y_pos = max(jit_ms[i], metal_ms[i]) * 1.4
    ax.text(i, y_pos, label, ha='center', fontsize=15, fontweight='bold', color=color)

for i, s in enumerate(speedups_k):
    symbol = '✓' if s > 1 else '✗'
    color = GREEN if s > 1 else CORNELL_RED
    ax.text(i, 0.007, symbol, ha='center', fontsize=20, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('figures/poster/kernel_scorecard.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/poster/kernel_scorecard.pdf', bbox_inches='tight')
plt.close()
print("Saved kernel_scorecard")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

prefixes_20 = [16, 64, 128, 256, 512, 768, 1000]
base_20 = [898, 1102, 1464, 2161, 3428, 5100, 6558]
opt_20 = [740, 776, 831, 888, 1013, 1186, 1107]
sp_20 = [b/o for b, o in zip(base_20, opt_20)]

ax1.plot(prefixes_20, base_20, 'o-', color=CORNELL_RED, linewidth=3, markersize=10, label='Baseline')
ax1.plot(prefixes_20, opt_20, 's-', color=GREEN, linewidth=3, markersize=10, label='Optimized')
ax1.fill_between(prefixes_20, opt_20, base_20, alpha=0.12, color=GREEN)
for x, b, o, s in zip(prefixes_20, base_20, opt_20, sp_20):
    if x in [64, 256, 512, 1000]:
        ax1.annotate(f'{s:.1f}×', xy=(x, (b+o)/2), fontsize=14, ha='center',
                    fontweight='bold', color=BLUE,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=BLUE, alpha=0.8))
ax1.set_xlabel('Prefix Length (tokens)', fontsize=20)
ax1.set_ylabel('Latency (ms)', fontsize=20)
ax1.set_title('20 Diffusion Steps', fontsize=22)
ax1.legend(fontsize=14)
ax1.set_ylim(0, 7500)
ax1.grid(True, alpha=0.25)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

prefixes_50 = [16, 64, 128, 256, 512]
base_50 = [1707, 2235, 2876, 4389, 7361]
opt_50 = [1235, 1227, 1281, 1347, 1492]
sp_50 = [b/o for b, o in zip(base_50, opt_50)]

ax2.plot(prefixes_50, base_50, 'o-', color=CORNELL_RED, linewidth=3, markersize=10, label='Baseline')
ax2.plot(prefixes_50, opt_50, 's-', color=GREEN, linewidth=3, markersize=10, label='Optimized')
ax2.fill_between(prefixes_50, opt_50, base_50, alpha=0.12, color=GREEN)
for x, b, o, s in zip(prefixes_50, base_50, opt_50, sp_50):
    ax2.annotate(f'{s:.1f}×', xy=(x, (b+o)/2), fontsize=14, ha='center',
                fontweight='bold', color=BLUE,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=BLUE, alpha=0.8))
ax2.set_xlabel('Prefix Length (tokens)', fontsize=20)
ax2.set_ylabel('Latency (ms)', fontsize=20)
ax2.set_title('50 Diffusion Steps', fontsize=22)
ax2.legend(fontsize=14)
ax2.set_ylim(0, 8500)
ax2.grid(True, alpha=0.25)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('End-to-End Latency on Real C4 Validation Text\n(10 documents per bucket, 3 runs, 95% CI)',
            fontsize=22, y=1.02)
plt.tight_layout()
plt.savefig('figures/poster/prefix_scaling_both.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/poster/prefix_scaling_both.pdf', bbox_inches='tight')
plt.close()
print("Saved prefix_scaling_both")

print("\nAll poster figures saved to figures/poster/")
print("Files:", os.listdir('figures/poster'))
