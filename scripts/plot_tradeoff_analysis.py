"""
Trade-off Analysis: Privacy-Utility Operating Points Scatter Plot

Plots all existing AR/RI data points from ablation variants, stage variants,
baselines, and cross-dataset experiments to show the privacy-utility landscape.
Our full model is Pareto-optimal across all configurations.

Safe for login node — no GPU/model needed, just plotting numbers.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================
# All verified (SGN AR, SGN RI) operating points
# ======================================================================

# Baselines (NTU60)
baselines = {
    'Raw Skeleton':    (89.1, 75.4),
    'Gaussian Noise':  (80.3, 67.8),
    'DMR':             (43.1, 38.1),
    'PMR':             (19.9, 24.2),
}

# Our full model (NTU60, mean of 3 seeds)
ours = {
    'Full Model': (55.4, 12.2),
}
ours_err = {'ar': 0.7, 'ri': 0.2}

# Component ablations (NTU60, BS=32, single runs)
component_ablations = {
    'Static Identity':     (46.6, 14.6),
    'No Adversarial':      (44.9, 14.8),
    'Full-Seq Identity':   (44.5, 12.9),
    'No Orthogonality':    (43.5, 13.2),
    'No Action Backbone':  (40.2, 12.6),
    'No Temporal Convs':   (40.0, 13.8),
}

# Stage ablations removed per advisor request

# ======================================================================
# Plot
# ======================================================================
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

# Helper to plot a group
def plot_group(data, color, marker, label, size=50, alpha=0.85, annotate=True):
    for name, (ar, ri) in data.items():
        ax.scatter(ar, ri, c=color, marker=marker, s=size, alpha=alpha,
                   edgecolors='black', linewidths=0.4, zorder=5)
        if annotate:
            ax.annotate(name, (ar, ri), textcoords="offset points",
                        xytext=(4, 4), fontsize=5.5, alpha=0.75, ha='left')

# Baselines
for name, (ar, ri) in baselines.items():
    c = {'Raw Skeleton': '#888888', 'Gaussian Noise': '#aaaaaa',
         'DMR': '#e67e22', 'PMR': '#e74c3c'}[name]
    ax.scatter(ar, ri, c=c, marker='s', s=80, edgecolors='black',
               linewidths=0.5, zorder=6)
    offset = {'Raw Skeleton': (3, 3), 'Gaussian Noise': (-5, 4),
              'DMR': (4, 3), 'PMR': (-3, 4)}[name]
    ax.annotate(name, (ar, ri), textcoords="offset points",
                xytext=offset, fontsize=6.5, ha='left', va='bottom')

# Component ablations — use leader lines for the crowded cluster
ablation_labels = {
    'Static Identity':     (54, 19),
    'No Adversarial':      (54, 17),
    'Full-Seq Identity':   (26, 8),
    'No Orthogonality':    (26, 6),
    'No Action Backbone':  (26, 11),
    'No Temporal Convs':   (26, 13.8),
}
for name, (ar, ri) in component_ablations.items():
    ax.scatter(ar, ri, c='#7fb3d8', marker='D', s=45, alpha=0.85,
               edgecolors='black', linewidths=0.4, zorder=5)
    lx, ly = ablation_labels[name]
    ax.annotate(name, (ar, ri), xytext=(lx, ly),
                textcoords='data', fontsize=5.5, alpha=0.75, ha='left',
                arrowprops=dict(arrowstyle='-', color='#999999',
                                lw=0.5, alpha=0.5))

# Full model (star)
ax.errorbar(55.4, 12.2, xerr=0.7, yerr=0.2, fmt='*', color='#2980b9',
            markersize=16, capsize=3, zorder=10, markeredgecolor='black',
            markeredgewidth=0.5)
ax.annotate('Full Model\n(Ours)', (55.4, 12.2), textcoords="offset points",
            xytext=(6, -14), fontsize=7.5, fontweight='bold', ha='left',
            color='#2980b9')

# Random chance line
ax.axhline(y=2.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.text(18, 3.2, 'Random chance RI (2.5%)', fontsize=6, color='gray', alpha=0.6)

# Annotations for Pareto-optimal region
ax.annotate('', xy=(58, 10), xytext=(58, 16),
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.4, lw=1.5))
ax.annotate('', xy=(58, 10), xytext=(48, 10),
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.4, lw=1.5))
ax.text(58.5, 10, 'Better', fontsize=6, color='green', alpha=0.5,
        fontstyle='italic', va='top')

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',
               markersize=7, markeredgecolor='black', markeredgewidth=0.4,
               label='Baselines'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#7fb3d8',
               markersize=6, markeredgecolor='black', markeredgewidth=0.4,
               label='Component ablations'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#2980b9',
               markersize=12, markeredgecolor='black', markeredgewidth=0.4,
               label='Full Model (Ours)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=6.5,
          framealpha=0.85, edgecolor='#cccccc')

ax.set_xlabel('Action Recognition Accuracy (AR %) $\\rightarrow$', fontsize=9)
ax.set_ylabel('Re-Identification Accuracy (RI %) $\\leftarrow$', fontsize=9)
ax.set_xlim(15, 95)
ax.set_ylim(0, 80)
ax.tick_params(labelsize=7)
ax.grid(True, alpha=0.15)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('paper/fig/tradeoff_analysis.pdf', dpi=300, bbox_inches='tight')
print("Saved: paper/fig/tradeoff_analysis.pdf")
