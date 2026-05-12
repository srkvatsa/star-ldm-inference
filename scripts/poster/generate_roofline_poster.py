import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('figures/poster', exist_ok=True)

plt.rcParams.update({
    'font.size': 20,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
})

CORNELL_RED = '#B31B1B'
GREEN = '#2E7D32'
BLUE = '#1565C0'
GRAY = '#9E9E9E'

fig, ax = plt.subplots(figsize=(16, 10))

peak = 14000
bw = 546
ridge = peak / bw

ai = np.logspace(-1.5, 2.5, 500)
roof = np.minimum(peak, ai * bw)
ax.loglog(ai, roof, '-', color='#333', linewidth=4, zorder=3)

ax.fill_between(ai, roof, 20000, alpha=0.03, color='black')

ax.axvline(x=ridge, color=GRAY, linestyle=':', linewidth=1.5, alpha=0.5)
ax.text(ridge * 1.15, 18000, f'Ridge\n{ridge:.0f} F/B', fontsize=13,
        color=GRAY, ha='left', va='top')

ax.text(0.15, 150, 'Memory\nbandwidth\nlimited', fontsize=16,
        color='#555', ha='center', rotation=42, fontstyle='italic')
ax.text(100, 18000, 'Compute limited', fontsize=16,
        color='#555', ha='center', fontstyle='italic')

points = [

    ('DDPM Step', 1.25, 0.37, GRAY, 'o', 220),
    ('RMSNorm+FiLM', 0.74, 4.5, GREEN, 's', 280),
    ('Tiny Attention', 2.28, 7.4, GREEN, 's', 280),
    ('GPT-2\ndecode-8', 4.0, 120, CORNELL_RED, 'D', 350),
    ('Soft Prompt\nGenerator', 4.04, 440, BLUE, '^', 350),
]

for name, x, y, color, marker, size in points:
    ax.scatter(x, y, c=color, marker=marker, s=size, zorder=5,
              edgecolors='white', linewidths=2)

label_positions = {
    'DDPM Step': (0.25, 0.12, 'right'),
    'RMSNorm+FiLM': (0.18, 8, 'right'),
    'Tiny Attention': (5.5, 5, 'left'),
    'GPT-2\ndecode-8': (12, 80, 'left'),
    'Soft Prompt\nGenerator': (12, 600, 'left'),
}

for name, x, y, color, marker, size in points:
    lx, ly, ha = label_positions[name]

    ceiling = min(peak, x * bw)
    eff = y / ceiling * 100
    label = f'{name}\n{y:.0f} GFLOP/s ({eff:.1f}%)'
    if y < 1:
        label = f'{name}\n{y:.1f} GFLOP/s ({eff:.1f}%)'

    ax.annotate(label, xy=(x, y), xytext=(lx, ly),
               fontsize=14, fontweight='bold', color=color, ha=ha, va='center',
               arrowprops=dict(arrowstyle='-', color=color, lw=1.5, alpha=0.6),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=color, alpha=0.9, linewidth=1.5))

ax.axhspan(0.03, 10, alpha=0.07, color=CORNELL_RED, zorder=1)
ax.text(0.045, 0.06, 'Dispatch-latency-bound regime',
        fontsize=18, color=CORNELL_RED, fontweight='bold', fontstyle='italic')
ax.text(0.045, 0.038, 'GPU kernel launch cost > computation time',
        fontsize=14, color=CORNELL_RED, fontstyle='italic')

ax.set_xlabel('Arithmetic Intensity (FLOP / byte)', fontsize=24, labelpad=10)
ax.set_ylabel('Throughput (GFLOP/s)', fontsize=24, labelpad=10)
ax.set_title('Roofline Analysis: STAR-LDM Components on Apple M4 Max',
            fontsize=26, pad=15, fontweight='bold')

ax.set_xlim(0.03, 500)
ax.set_ylim(0.025, 25000)

ax.grid(True, alpha=0.12, which='major', linewidth=0.8)
ax.grid(True, alpha=0.06, which='minor', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_elements = [
    mpatches.Patch(color=GREEN, label='Metal kernel (micro-transformer)'),
    mpatches.Patch(color=GRAY, label='JIT fallback'),
    mpatches.Patch(color=CORNELL_RED, label='GPT-2 backbone'),
    mpatches.Patch(color=BLUE, label='Full module (6-layer transformer)'),
    plt.Line2D([0], [0], color='#333', linewidth=4, label=f'M4 Max roofline (14 TFLOP/s, 546 GB/s)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=14,
         framealpha=0.95, edgecolor='#ddd')

ax.text(250, 0.04, 'Apple M4 Max\n14 TFLOP/s FP32\n546 GB/s bandwidth\n128 GB unified memory',
       fontsize=12, color='#777', ha='right', va='bottom',
       bbox=dict(boxstyle='round', facecolor='#f8f8f8', edgecolor='#ddd'))

plt.tight_layout()
plt.savefig('figures/poster/roofline_clean.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/poster/roofline_clean.pdf', bbox_inches='tight')
plt.close()
print("Saved roofline_clean")
