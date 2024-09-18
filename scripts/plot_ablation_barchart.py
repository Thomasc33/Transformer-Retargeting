"""
Ablation Bar Chart for ECCV paper.
Shows AR and RI for each ablation variant with PMR/DMR reference lines.
Can run on login node — no GPU needed.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Ablation results (SGN, BS=32)
variants = [
    ('Full Model', 45.3, 13.6),
    ('Static\nIdentity', 46.6, 14.6),
    ('No\nAdversarial', 44.9, 14.8),
    ('Full-Seq\nIdentity', 44.5, 12.9),
    ('No\nOrthogonality', 43.5, 13.2),
    ('No Action\nBackbone', 40.2, 12.6),
    ('No Temporal\nConvs', 40.0, 13.8),
]

names = [v[0] for v in variants]
ar_vals = [v[1] for v in variants]
ri_vals = [v[2] for v in variants]

x = np.arange(len(names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(7, 3.5))

# AR bars (blue)
bars_ar = ax1.bar(x - width/2, ar_vals, width, label='AR (%) $\\uparrow$',
                   color='#3498db', edgecolor='black', linewidth=0.5, alpha=0.85)
# RI bars (red)
bars_ri = ax1.bar(x + width/2, ri_vals, width, label='RI (%) $\\downarrow$',
                   color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.85)

# Reference lines
ax1.axhline(y=24.2, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.6)
ax1.text(len(names) - 0.5, 24.8, 'PMR RI (24.2%)', fontsize=7, color='#e74c3c',
         ha='right', alpha=0.8)

ax1.axhline(y=38.1, color='#e74c3c', linestyle=':', linewidth=1.0, alpha=0.4)
ax1.text(len(names) - 0.5, 38.7, 'DMR RI (38.1%)', fontsize=7, color='#e74c3c',
         ha='right', alpha=0.6)

ax1.axhline(y=2.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
ax1.text(len(names) - 0.5, 3.1, 'Random chance RI', fontsize=6, color='gray',
         ha='right', alpha=0.6)

# Grouping annotations
ax1.axvspan(-0.5, 0.5, alpha=0.05, color='blue')  # Full model
ax1.axvspan(0.5, 3.5, alpha=0.05, color='green')   # Architecture ablations
ax1.axvspan(3.5, 4.5, alpha=0.00, color='white')    # separator
ax1.axvspan(4.5, 6.5, alpha=0.05, color='orange')   # Component ablations

# Group labels
ax1.text(2.0, 48, 'Identity encoding\nvariants', fontsize=6, ha='center',
         color='green', alpha=0.7, style='italic')
ax1.text(5.5, 48, 'Component\nremoval', fontsize=6, ha='center',
         color='orange', alpha=0.7, style='italic')

ax1.set_ylabel('Accuracy (%)', fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=7)
ax1.set_ylim(0, 52)
ax1.legend(loc='upper right', fontsize=7, framealpha=0.8)
ax1.tick_params(labelsize=8)
ax1.grid(True, axis='y', alpha=0.2)
ax1.set_axisbelow(True)

# Key insight annotation
ax1.annotate('All variants below PMR RI\n→ architecture > losses',
             xy=(3, 16.6), xytext=(3, 32),
             fontsize=7, ha='center', style='italic',
             arrowprops=dict(arrowstyle='->', color='black', alpha=0.5, lw=1),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('paper/fig/ablation_barchart.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/fig/ablation_barchart.png', dpi=300, bbox_inches='tight')
print("Saved: paper/fig/ablation_barchart.pdf")
