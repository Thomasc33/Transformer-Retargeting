"""
Privacy-Utility Tradeoff Scatter Plot — Dissertation Figure 5.12
NTU-60, β=0.2, pre-trained SGN evaluator.

Safe for login node — no GPU/model needed.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_PATH = 'dissertation/figures/tmr/tradeoff_analysis.pdf'

# ── Data ──────────────────────────────────────────────────────────────────────

baselines = {
    'Raw Skeleton': (89.1, 75.4),
    'DMR':          (49.1, 25.7),
    'PMR':          (35.7,  7.8),
}

output_supervision = {
    'Base\n(no output supervision)':      (81.0, 43.4),
    '+ Output Action\nClassifier only':   (81.0, 42.2),
}

stage_ablations = {
    'Stage 3 only':  (82.6, 52.4),
    'Stages 2→3':    (82.5, 55.3),
    'Stages 1→3':    (82.3, 53.4),
    'Stages 1→2':    (81.1, 35.5),
    'Stage 1 only':  (57.3, 15.3),
}

reference = {
    'DisentangledTMR\n(ours, β=0.2)': (76.2, 14.9),
}

# ── Figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7.0, 5.0))

# Baselines — squares
BASELINE_COLORS = {
    'Raw Skeleton': '#888888',
    'DMR':          '#e67e22',
    'PMR':          '#e74c3c',
}
BASELINE_OFFSETS = {
    'Raw Skeleton': ( 4,  4),
    'DMR':          ( 4,  4),
    'PMR':          ( 4, -10),
}
for name, (ar, ri) in baselines.items():
    ax.scatter(ar, ri, c=BASELINE_COLORS[name], marker='s', s=80,
               edgecolors='black', linewidths=0.5, zorder=6)
    ox, oy = BASELINE_OFFSETS[name]
    ax.annotate(name, (ar, ri), textcoords='offset points',
                xytext=(ox, oy), fontsize=6.5, ha='left', va='bottom')

# Output supervision ablation — circles
for name, (ar, ri) in output_supervision.items():
    ax.scatter(ar, ri, c='#9b59b6', marker='o', s=55, alpha=0.85,
               edgecolors='black', linewidths=0.4, zorder=5)
    ax.annotate(name, (ar, ri), textcoords='offset points',
                xytext=(4, 4), fontsize=5.5, alpha=0.85, ha='left')

# Stage ablations — triangles
STAGE_COLORS = {
    'Stage 3 only': '#3498db',
    'Stages 2→3':   '#2980b9',
    'Stages 1→3':   '#1a6fa0',
    'Stages 1→2':   '#85c1e9',
    'Stage 1 only': '#aed6f1',
}
STAGE_OFFSETS = {
    'Stage 3 only': ( 4,  4),
    'Stages 2→3':   ( 4, -11),
    'Stages 1→3':   ( 4,  4),
    'Stages 1→2':   ( 4,  4),
    'Stage 1 only': ( 4,  4),
}
for name, (ar, ri) in stage_ablations.items():
    ax.scatter(ar, ri, c=STAGE_COLORS[name], marker='^', s=55, alpha=0.85,
               edgecolors='black', linewidths=0.4, zorder=5)
    ox, oy = STAGE_OFFSETS[name]
    ax.annotate(name, (ar, ri), textcoords='offset points',
                xytext=(ox, oy), fontsize=5.5, alpha=0.85, ha='left')

# DisentangledTMR reference — gold star, prominent
ref_ar, ref_ri = list(reference.values())[0]
ax.scatter(ref_ar, ref_ri, c='#f1c40f', marker='*', s=320,
           edgecolors='#b7950b', linewidths=0.8, zorder=10)
ax.annotate(list(reference.keys())[0], (ref_ar, ref_ri),
            textcoords='offset points', xytext=(7, -16),
            fontsize=7.5, fontweight='bold', ha='left', color='#9a7d0a')

# Random chance reference line
ax.axhline(y=2.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
ax.text(17, 3.3, 'Random chance (RI 2.5%)', fontsize=5.5, color='gray', alpha=0.5)

# "Better" direction arrow
ax.annotate('', xy=(93, 6), xytext=(93, 14),
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.35, lw=1.5))
ax.annotate('', xy=(93, 6), xytext=(85, 6),
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.35, lw=1.5))
ax.text(94, 6, 'Better', fontsize=6, color='green', alpha=0.45,
        fontstyle='italic', va='center')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',
               markersize=7, markeredgecolor='black', markeredgewidth=0.4,
               label='Baselines'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9b59b6',
               markersize=7, markeredgecolor='black', markeredgewidth=0.4,
               label='Output supervision ablation'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#3498db',
               markersize=7, markeredgecolor='black', markeredgewidth=0.4,
               label='Stage ablation (β=0.2)'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#f1c40f',
               markersize=12, markeredgecolor='#b7950b', markeredgewidth=0.6,
               label='DisentangledTMR (ours)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=6.5,
          framealpha=0.9, edgecolor='#cccccc')

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlabel('Action Recognition Accuracy (AR %) $\\rightarrow$', fontsize=9)
ax.set_ylabel('Re-Identification Accuracy (RI %) $\\downarrow$ lower is better', fontsize=9)
ax.set_xlim(15, 98)
ax.set_ylim(0, 82)
ax.tick_params(labelsize=7)
ax.grid(True, alpha=0.15)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight', format='pdf')
print(f"Saved: {OUT_PATH}")
