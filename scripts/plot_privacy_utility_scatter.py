"""
Privacy-Utility Trade-off Scatter Plot for ECCV paper Figure.
Plots AR (x-axis) vs RI (y-axis) for all methods.
Can run on login node — no GPU/model needed, just plotting numbers.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Results from verified experiments
methods = {
    'Raw Skeleton': {'sgn_ar': 89.1, 'sgn_ri': 75.4, 'mix_ar': 88.6, 'mix_ri': 72.8},
    'Gaussian\nNoise': {'sgn_ar': 80.3, 'sgn_ri': 67.8, 'mix_ar': 83.6, 'mix_ri': 70.7},
    'DMR': {'sgn_ar': 43.1, 'sgn_ri': 38.1, 'mix_ar': 45.3, 'mix_ri': 38.7},
    'PMR': {'sgn_ar': 19.9, 'sgn_ri': 24.2, 'mix_ar': 20.8, 'mix_ri': 25.5},
    'DisentangledTMR\n(Ours)': {'sgn_ar': 55.4, 'sgn_ri': 12.2, 'mix_ar': 55.7, 'mix_ri': 12.5},
}

# Error bars for Ours (from seed runs)
ours_err = {'sgn_ar': 0.7, 'sgn_ri': 0.2, 'mix_ar': 1.1, 'mix_ri': 0.4}

# Colors and markers
colors = {
    'Raw Skeleton': '#888888',
    'Gaussian\nNoise': '#aaaaaa',
    'DMR': '#e67e22',
    'PMR': '#e74c3c',
    'DisentangledTMR\n(Ours)': '#2980b9',
}
markers_sgn = 'o'  # circle for SGN
markers_mix = '^'  # triangle for MixFormer

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

for name, vals in methods.items():
    c = colors[name]
    ms = 120 if 'Ours' in name else 80
    zorder = 10 if 'Ours' in name else 5

    # SGN
    if 'Ours' in name:
        ax.errorbar(vals['sgn_ar'], vals['sgn_ri'],
                     xerr=ours_err['sgn_ar'], yerr=ours_err['sgn_ri'],
                     fmt='o', color=c, markersize=10, capsize=3, zorder=zorder,
                     markeredgecolor='black', markeredgewidth=0.5)
        ax.errorbar(vals['mix_ar'], vals['mix_ri'],
                     xerr=ours_err['mix_ar'], yerr=ours_err['mix_ri'],
                     fmt='^', color=c, markersize=10, capsize=3, zorder=zorder,
                     markeredgecolor='black', markeredgewidth=0.5)
    else:
        ax.scatter(vals['sgn_ar'], vals['sgn_ri'], c=c, marker='o', s=ms,
                   zorder=zorder, edgecolors='black', linewidths=0.5)
        ax.scatter(vals['mix_ar'], vals['mix_ri'], c=c, marker='^', s=ms,
                   zorder=zorder, edgecolors='black', linewidths=0.5)

# Random chance line
ax.axhline(y=2.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
ax.text(15, 3.5, 'Random chance RI (2.5%)', fontsize=7, color='gray')

# Ideal region annotation — star marks the ideal corner (high AR, low RI)
ax.plot(95, 2.5, marker='*', color='green', markersize=14, alpha=0.4, zorder=1)
ax.text(95, 5.5, 'Ideal', fontsize=7, color='green', alpha=0.6, ha='center', fontstyle='italic')

# Arrow annotation: 63pp RI reduction
ax.annotate('', xy=(55.4, 12.2), xytext=(89.1, 75.4),
            arrowprops=dict(arrowstyle='->', color='#2980b9', alpha=0.3,
                           lw=1.5, connectionstyle='arc3,rad=0.2'))

# Labels for each method
label_offsets = {
    'Raw Skeleton': (2, 3),
    'Gaussian\nNoise': (-15, 3),
    'DMR': (2, 3),
    'PMR': (-2, 3),
    'DisentangledTMR\n(Ours)': (-18, -22),
}
for name, vals in methods.items():
    mid_ar = (vals['sgn_ar'] + vals['mix_ar']) / 2
    mid_ri = (vals['sgn_ri'] + vals['mix_ri']) / 2
    dx, dy = label_offsets[name]
    fontweight = 'bold' if 'Ours' in name else 'normal'
    fontsize = 8 if 'Ours' in name else 7
    ax.annotate(name, (mid_ar, mid_ri), textcoords="offset points",
                xytext=(dx, dy), fontsize=fontsize, fontweight=fontweight,
                ha='left', va='bottom')

# Legend for evaluator markers
ax.scatter([], [], marker='o', c='gray', s=60, label='SGN evaluator', edgecolors='black', linewidths=0.5)
ax.scatter([], [], marker='^', c='gray', s=60, label='MixFormer evaluator', edgecolors='black', linewidths=0.5)
ax.legend(loc='upper left', fontsize=7, framealpha=0.8)

ax.set_xlabel('Action Recognition Accuracy (AR %) $\\rightarrow$', fontsize=9)
ax.set_ylabel('Re-Identification Accuracy (RI %) $\\leftarrow$', fontsize=9)
ax.set_xlim(10, 100)
ax.set_ylim(0, 85)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.2)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('paper/fig/privacy_utility_scatter.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/fig/privacy_utility_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: paper/fig/privacy_utility_scatter.pdf")
