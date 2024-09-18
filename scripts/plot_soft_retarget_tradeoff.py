"""
Recreate soft_retarget_tradeoff figure with seaborn styling.
Panel (a): Frozen eval — AR (blue solid) + RI (red dashed)
Panel (b): From-scratch eval — AR only (green solid), no RI line

Uses the old dissertation data from frozen_ar_experiments/soft_a0_*.
Safe for login node — just plotting numbers.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
palette = sns.color_palette("muted")

# Data from dissertation/soft_retargeting_results.md
betas =      [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.70, 1.00]

frozen_ar =  [89.1, 84.9, 84.8, 83.8, 82.7, 80.0, 76.8, 38.8, 42.8, 27.7, 13.0,  2.0]
frozen_ri =  [75.4, 69.5, 61.4, 60.9, 52.0, 37.5, 17.3,  9.6, 14.3,  3.1,  8.2,  4.7]

fs_ar =      [89.1, 86.8, 87.8, 87.2, 87.3, 86.9, 87.1, 85.3, 85.4, 84.6, 81.9, 55.4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

# ── Panel (a): Frozen evaluation ──
ax1.plot(betas, frozen_ar, 'o-', color=palette[0], markersize=6, linewidth=2,
         label=r'Frozen AR ($\rightarrow$)', zorder=5)
ax1.plot(betas, frozen_ri, 's--', color=palette[3], markersize=5.5, linewidth=1.8,
         label=r'Frozen RI ($\leftarrow$)', zorder=5)

# Highlight β=0.2 sweet spot
ax1.axvspan(0.19, 0.21, color=sns.color_palette("pastel")[5], alpha=0.5, zorder=1)
ax1.annotate(
    r'$\beta$=0.2' + '\n76.8% AR\n17.3% RI',
    xy=(0.20, 76.8), xytext=(0.32, 78),
    fontsize=8, ha='left',
    arrowprops=dict(arrowstyle='-', color='black', lw=0.8),
)

ax1.set_title('(a) Frozen evaluation', fontsize=12, fontweight='bold')
ax1.set_xlabel(r'Retargeting ratio $\beta$', fontsize=10)
ax1.set_ylabel('Accuracy (%)', fontsize=10)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(0, 92)
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax1.tick_params(labelsize=8)

# ── Panel (b): From-scratch evaluation (AR only) ──
ax2.plot(betas, fs_ar, 'o-', color=palette[2], markersize=6, linewidth=2,
         label=r'FS AR ($\rightarrow$)', zorder=5)

# Highlight β=0.2
ax2.axvspan(0.19, 0.21, color=sns.color_palette("pastel")[5], alpha=0.5, zorder=1)

ax2.set_title('(b) From-scratch evaluation', fontsize=12, fontweight='bold')
ax2.set_xlabel(r'Retargeting ratio $\beta$', fontsize=10)
ax2.set_xlim(-0.02, 1.02)
ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax2.tick_params(labelsize=8)

sns.despine(fig=fig)
plt.tight_layout()
plt.savefig('paper/fig/soft_retarget_tradeoff.pdf', dpi=300, bbox_inches='tight')
print("Saved: paper/fig/soft_retarget_tradeoff.pdf")
