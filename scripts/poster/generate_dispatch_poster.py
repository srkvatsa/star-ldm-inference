import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures/poster', exist_ok=True)
os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.size': 20,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
})

CORNELL_RED = '#B31B1B'
GREEN = '#2E7D32'
BLUE = '#1565C0'

fig, ax = plt.subplots(figsize=(12, 8))

categories = ['HuggingFace\nGPT-2 Forward', 'Streamlined\nForward (ours)']

hf_compute = 432
hf_mask = 806
hf_linear = 1440
hf_tensor = 3507

bottom = 0
bars_hf = []
segments = [
    (hf_compute, GREEN, 'Computation (432)'),
    (hf_mask, '#E57373', 'Mask construction (806)'),
    (hf_linear, '#FF8A65', 'Linear projections (1,440)'),
    (hf_tensor, '#FFAB91', 'Tensor management (3,507)'),
]
for val, color, label in segments:
    b = ax.bar(0, val, bottom=bottom, color=color, width=0.45,
               edgecolor='white', linewidth=1.5, label=label)
    bars_hf.append(b)
    bottom += val

ax.bar(1, 432, color=GREEN, width=0.45, edgecolor='white', linewidth=1.5)

ax.text(0, 6185 + 200, '6,185', ha='center', fontsize=32, fontweight='bold', color=CORNELL_RED)
ax.text(1, 432 + 200, '432', ha='center', fontsize=32, fontweight='bold', color=GREEN)

ax.annotate('', xy=(0.85, 3300), xytext=(0.15, 3300),
           arrowprops=dict(arrowstyle='->', color=BLUE, lw=3.5))
ax.text(0.5, 3650, '93% reduction', ha='center', fontsize=22,
       fontweight='bold', color=BLUE,
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor=BLUE, linewidth=2))

ax.text(0, 5100, 'Framework\noverhead', fontsize=16,
        color='#444', ha='center', va='center', fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#bbb', alpha=0.95, linewidth=1.2))

ax.set_xticks([0, 1])
ax.set_xticklabels(['HuggingFace\nGPT-2 Forward', 'Streamlined\nForward (ours)'], fontsize=20)
ax.set_ylabel('ATen Operator Dispatches per Call', fontsize=22)
ax.set_title('Framework Overhead: Operator Dispatch Count', fontsize=26, fontweight='bold', pad=15)

ax.set_ylim(0, 7500)
ax.set_xlim(-0.5, 1.5)
ax.grid(True, alpha=0.15, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper right', fontsize=14, framealpha=0.95, edgecolor='#ddd')

plt.tight_layout()
plt.savefig('figures/poster/dispatch_stacked.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/poster/dispatch_stacked.pdf', bbox_inches='tight')
plt.savefig('figures/dispatch_stacked.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/dispatch_stacked.pdf', bbox_inches='tight')
plt.close()
print("Saved dispatch_stacked to figures/ and figures/poster/")
