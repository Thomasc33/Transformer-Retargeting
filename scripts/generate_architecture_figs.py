#!/usr/bin/env python3
"""Generate publication-quality architecture diagrams for the DisentangledTMR paper.

Creates two figures:
1. Encoder figure: Dual encoder (Action + Identity)
2. Decoder figure: Factorized decoder with adaptive gate

Uses matplotlib only (no GPU/model loading) - safe for login node.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------- Style constants ----------
# Colors
ACTION_COLOR = '#4477AA'        # Blue for action stream
IDENTITY_COLOR = '#CC6677'      # Rose for identity stream
GATE_COLOR = '#999933'          # Olive for gate/fusion
NEUTRAL_COLOR = '#BBBBBB'       # Grey for neutral blocks
INPUT_COLOR = '#66CCEE'         # Cyan for inputs
OUTPUT_COLOR = '#228833'        # Green for outputs
BG_ACTION = '#EEF3FA'           # Light blue background
BG_IDENTITY = '#FAEEF1'         # Light pink background

FONT_SIZE = 9
FONT_SIZE_SMALL = 7.5
FONT_SIZE_TINY = 6.5
FONT_FAMILY = 'serif'

plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': FONT_SIZE,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})


def rounded_box(ax, x, y, w, h, text, color='white', edgecolor='black',
                fontsize=FONT_SIZE, textcolor='black', linewidth=1.0,
                text_lines=None, bold=False, zorder=2):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=edgecolor,
        linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    if text_lines:
        for i, line in enumerate(text_lines):
            offset = (i - (len(text_lines)-1)/2) * fontsize * 0.015
            ax.text(x, y + offset, line, ha='center', va='center',
                    fontsize=fontsize, color=textcolor, weight=weight, zorder=zorder+1)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, weight=weight, zorder=zorder+1)
    return box


def arrow(ax, x1, y1, x2, y2, color='black', linewidth=1.0, style='->', zorder=3):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=linewidth, shrinkA=2, shrinkB=2),
                zorder=zorder)


def bracket_arrow(ax, x1, y1, x2, y2, color='black', linewidth=1.0, via_y=None):
    """Draw an L-shaped arrow."""
    if via_y is None:
        via_y = (y1 + y2) / 2
    ax.plot([x1, x1, x2], [y1, via_y, via_y], color=color, linewidth=linewidth,
            solid_capstyle='round', zorder=3)
    arrow(ax, x2, via_y, x2, y2, color=color, linewidth=linewidth)


def circle_op(ax, x, y, text, radius=0.12, color='white', edgecolor='black',
              fontsize=FONT_SIZE_SMALL, zorder=4):
    """Draw a circle with text (for operations like +, x, sigma)."""
    circle = plt.Circle((x, y), radius, facecolor=color, edgecolor=edgecolor,
                         linewidth=1.0, zorder=zorder)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            weight='bold', zorder=zorder+1)


# ==================== ENCODER FIGURE ====================
def create_encoder_figure():
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.8))
    ax.set_xlim(-0.3, 7.3)
    ax.set_ylim(-0.3, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Background regions ---
    # Action encoder background
    action_bg = FancyBboxPatch(
        (-0.15, 1.55), 7.3, 2.3,
        boxstyle="round,pad=0.1", facecolor=BG_ACTION, edgecolor=ACTION_COLOR,
        linewidth=1.5, linestyle='--', zorder=0, alpha=0.6
    )
    ax.add_patch(action_bg)
    ax.text(0.15, 3.65, 'Action Encoder  $E_A$', fontsize=FONT_SIZE+1, color=ACTION_COLOR,
            weight='bold', style='italic', zorder=1)

    # Identity encoder background
    id_bg = FancyBboxPatch(
        (-0.15, -0.2), 7.3, 1.55,
        boxstyle="round,pad=0.1", facecolor=BG_IDENTITY, edgecolor=IDENTITY_COLOR,
        linewidth=1.5, linestyle='--', zorder=0, alpha=0.6
    )
    ax.add_patch(id_bg)
    ax.text(0.15, 1.15, 'Identity Encoder  $E_I$', fontsize=FONT_SIZE+1, color=IDENTITY_COLOR,
            weight='bold', style='italic', zorder=1)

    # ---- ACTION ENCODER (top row, y=2.7) ----
    ya = 2.7  # main action row
    ya2 = 2.0  # MixFormer row

    # Input
    rounded_box(ax, 0.4, ya, 0.65, 0.45, '', color=INPUT_COLOR, edgecolor='#3399BB',
                fontsize=FONT_SIZE_SMALL, bold=True)
    ax.text(0.4, ya+0.08, 'Skeleton', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, weight='bold', zorder=3)
    ax.text(0.4, ya-0.1, r'$\mathbf{s} \in \mathbb{R}^{J \times T \times 3}$',
            ha='center', va='center', fontsize=FONT_SIZE_TINY, zorder=3)

    # Input processing
    rounded_box(ax, 1.5, ya, 0.9, 0.45, '', color='white', edgecolor=ACTION_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(1.5, ya+0.08, 'Input Processing', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(1.5, ya-0.1, 'pos + vel + acc', ha='center', va='center',
            fontsize=FONT_SIZE_TINY, color='#555', zorder=3)
    arrow(ax, 0.73, ya, 1.04, ya, color=ACTION_COLOR)

    # Multi-scale temporal conv
    rounded_box(ax, 2.8, ya, 1.05, 0.45, '', color='white', edgecolor=ACTION_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(2.8, ya+0.08, 'Multi-Scale', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(2.8, ya-0.10, 'Temporal Conv', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    arrow(ax, 1.96, ya, 2.27, ya, color=ACTION_COLOR)

    # Temporal attention
    rounded_box(ax, 4.1, ya, 0.9, 0.45, '', color='white', edgecolor=ACTION_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(4.1, ya+0.08, 'Temporal', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(4.1, ya-0.10, 'Attention', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    arrow(ax, 3.33, ya, 3.64, ya, color=ACTION_COLOR)

    # Linear projection
    rounded_box(ax, 5.15, ya, 0.7, 0.45, '', color='white', edgecolor=ACTION_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(5.15, ya+0.08, 'Linear', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(5.15, ya-0.10, r'$256 \to 768$', ha='center', va='center',
            fontsize=FONT_SIZE_TINY, color='#555', zorder=3)
    arrow(ax, 4.56, ya, 4.79, ya, color=ACTION_COLOR)

    # Gate fusion circle
    gx = 6.0
    circle_op(ax, gx, ya, r'$g$', radius=0.15, color=GATE_COLOR, edgecolor='#666600',
              fontsize=FONT_SIZE_SMALL)
    arrow(ax, 5.51, ya, gx - 0.15, ya, color=ACTION_COLOR)

    # MixFormer backbone (bottom sub-row)
    rounded_box(ax, 3.25, ya2, 1.3, 0.40, '', color='#F0F0FF', edgecolor=ACTION_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(3.25, ya2+0.06, 'Skeleton MixFormer', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(3.25, ya2-0.10, 'Backbone', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)

    # Arrow from input to MixFormer
    ax.annotate('', xy=(2.59, ya2), xytext=(0.73, ya),
                arrowprops=dict(arrowstyle='->', color=ACTION_COLOR,
                                lw=1.0, shrinkA=2, shrinkB=2,
                                connectionstyle='arc3,rad=0.2'),
                zorder=3)

    # Arrow from MixFormer to gate
    ax.plot([3.91, gx], [ya2, ya2], color=ACTION_COLOR, linewidth=1.0, zorder=3)
    arrow(ax, gx, ya2, gx, ya - 0.15, color=ACTION_COLOR)

    # Output H_action
    rounded_box(ax, 6.85, ya, 0.65, 0.45, '', color=ACTION_COLOR, edgecolor='#2255AA',
                fontsize=FONT_SIZE_SMALL, textcolor='white', bold=True)
    ax.text(6.85, ya+0.08, r'$\mathbf{H}_{\mathrm{action}}$', ha='center', va='center',
            fontsize=FONT_SIZE, color='white', weight='bold', zorder=3)
    ax.text(6.85, ya-0.12, r'$T \times 768$', ha='center', va='center',
            fontsize=FONT_SIZE_TINY, color='#ddd', zorder=3)
    arrow(ax, gx + 0.15, ya, 6.52, ya, color=ACTION_COLOR, linewidth=1.5)

    # Gate equation annotation
    ax.text(gx, ya + 0.40, r'$g \odot \mathbf{H}_{\mathrm{attn}} + (1{-}g) \odot \mathbf{H}_{\mathrm{MixF}}$',
            ha='center', va='bottom', fontsize=FONT_SIZE_TINY, color='#666600', style='italic', zorder=5)

    # ---- IDENTITY ENCODER (bottom row, y=0.4) ----
    yi = 0.4

    # Input
    rounded_box(ax, 0.4, yi, 0.65, 0.45, '', color=INPUT_COLOR, edgecolor='#3399BB',
                fontsize=FONT_SIZE_SMALL, bold=True)
    ax.text(0.4, yi+0.08, 'Skeleton', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, weight='bold', zorder=3)
    ax.text(0.4, yi-0.1, r'$\mathbf{s} \in \mathbb{R}^{J \times T \times 3}$',
            ha='center', va='center', fontsize=FONT_SIZE_TINY, zorder=3)

    # Static pose
    rounded_box(ax, 1.4, yi, 0.8, 0.45, '', color='white', edgecolor=IDENTITY_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(1.4, yi+0.08, 'Static Pose', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(1.4, yi-0.10, r'$\bar{\mathbf{s}} = \frac{1}{T}\sum_t \mathbf{s}_t$',
            ha='center', va='center', fontsize=FONT_SIZE_TINY, color='#555', zorder=3)
    arrow(ax, 0.73, yi, 0.99, yi, color=IDENTITY_COLOR)

    # Spatial GCN
    rounded_box(ax, 2.5, yi, 0.85, 0.45, '', color='white', edgecolor=IDENTITY_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(2.5, yi+0.08, 'Spatial GCN', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(2.5, yi-0.10, r'$64 \to 128 \to 256$', ha='center', va='center',
            fontsize=FONT_SIZE_TINY, color='#555', zorder=3)
    arrow(ax, 1.81, yi, 2.07, yi, color=IDENTITY_COLOR)

    # Spatial Attention
    rounded_box(ax, 3.6, yi, 0.85, 0.45, '', color='white', edgecolor=IDENTITY_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(3.6, yi+0.08, 'Spatial', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(3.6, yi-0.10, 'Attention', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    arrow(ax, 2.93, yi, 3.17, yi, color=IDENTITY_COLOR)

    # Global Pool + Bone Length branch
    # Upper: Global Avg Pool from Spatial Attention
    rounded_box(ax, 4.7, yi + 0.25, 0.75, 0.30, 'Global Pool',
                color='white', edgecolor=IDENTITY_COLOR,
                fontsize=FONT_SIZE_SMALL)
    arrow(ax, 4.03, yi + 0.05, 4.32, yi + 0.25, color=IDENTITY_COLOR)

    # Lower: Bone Length MLP from Static Pose
    rounded_box(ax, 4.7, yi - 0.25, 0.75, 0.30, '', color='white', edgecolor=IDENTITY_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(4.7, yi - 0.19, 'Bone', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(4.7, yi - 0.33, 'Length MLP', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    # Arrow from static pose to bone length (L-shaped)
    ax.plot([1.4, 1.4, 4.32], [yi - 0.23, yi - 0.55, yi - 0.55],
            color=IDENTITY_COLOR, linewidth=0.8, zorder=3)
    arrow(ax, 4.32, yi - 0.55, 4.32, yi - 0.40, color=IDENTITY_COLOR, linewidth=0.8)

    # Concat circle
    circle_op(ax, 5.35, yi, r'$\oplus$', radius=0.13, color='white', edgecolor=IDENTITY_COLOR,
              fontsize=FONT_SIZE)
    arrow(ax, 5.08, yi + 0.25, 5.22, yi + 0.05, color=IDENTITY_COLOR, linewidth=0.8)
    arrow(ax, 5.08, yi - 0.25, 5.22, yi - 0.05, color=IDENTITY_COLOR, linewidth=0.8)

    # Fusion MLP
    rounded_box(ax, 6.0, yi, 0.65, 0.45, '', color='white', edgecolor=IDENTITY_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(6.0, yi+0.08, 'Fusion', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(6.0, yi-0.10, 'MLP', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    arrow(ax, 5.48, yi, 5.67, yi, color=IDENTITY_COLOR)

    # Output H_identity
    rounded_box(ax, 6.85, yi, 0.65, 0.45, '', color=IDENTITY_COLOR, edgecolor='#AA4466',
                fontsize=FONT_SIZE_SMALL, textcolor='white', bold=True)
    ax.text(6.85, yi+0.08, r'$\mathbf{H}_{\mathrm{id}}$', ha='center', va='center',
            fontsize=FONT_SIZE, color='white', weight='bold', zorder=3)
    ax.text(6.85, yi-0.12, r'$256$', ha='center', va='center',
            fontsize=FONT_SIZE_TINY, color='#ddd', zorder=3)
    arrow(ax, 6.33, yi, 6.52, yi, color=IDENTITY_COLOR, linewidth=1.5)

    fig.tight_layout(pad=0.2)
    return fig


# ==================== DECODER FIGURE ====================
def create_decoder_figure():
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.6))
    ax.set_xlim(-0.3, 7.5)
    ax.set_ylim(-0.65, 3.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Main decoder layer background
    dec_bg = FancyBboxPatch(
        (0.7, -0.15), 5.65, 3.25,
        boxstyle="round,pad=0.1", facecolor='#F8F8F8', edgecolor='#888',
        linewidth=1.5, linestyle='--', zorder=0
    )
    ax.add_patch(dec_bg)
    ax.text(4.8, 2.90, 'Decoder Layer ($\\times 6$)', fontsize=FONT_SIZE+1,
            color='#555', weight='bold', style='italic', zorder=1, ha='center')

    yd = 1.4  # main decoder row

    # Input Frame[n]
    rounded_box(ax, 0.15, yd, 0.65, 0.50, '', color=INPUT_COLOR, edgecolor='#3399BB',
                fontsize=FONT_SIZE_SMALL, bold=True)
    ax.text(0.15, yd+0.10, 'Frame $n$', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, weight='bold', zorder=3)
    ax.text(0.15, yd-0.10, '(or zero)', ha='center', va='center',
            fontsize=FONT_SIZE_TINY, color='#555', zorder=3)

    # Causal Self-Attention
    rounded_box(ax, 1.25, yd, 0.75, 0.50, '', color='white', edgecolor='#555',
                fontsize=FONT_SIZE_SMALL)
    ax.text(1.25, yd+0.08, 'Causal', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    ax.text(1.25, yd-0.10, 'Self-Attn', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)
    arrow(ax, 0.48, yd, 0.87, yd, color='#555')

    # Residual + norm (small)
    circle_op(ax, 1.85, yd, '+', radius=0.10, color='#eee', edgecolor='#888',
              fontsize=FONT_SIZE_TINY)
    arrow(ax, 1.63, yd, 1.75, yd, color='#555', linewidth=0.8)
    # Skip connection (clean L-shape below)
    ax.plot([0.48, 0.48, 1.85], [yd - 0.25, yd - 0.52, yd - 0.52],
            color='#bbb', linewidth=0.7, zorder=2)
    arrow(ax, 1.85, yd - 0.52, 1.85, yd - 0.10, color='#bbb', linewidth=0.7)

    # ----- SPLIT: Action cross-attn (top) and Identity cross-attn (bottom) -----
    ya_cross = 2.15  # action cross-attn y
    yi_cross = 0.65  # identity cross-attn y

    # Action Cross-Attention
    rounded_box(ax, 2.85, ya_cross, 1.0, 0.50, '', color='white', edgecolor=ACTION_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(2.85, ya_cross+0.08, 'Action', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, color=ACTION_COLOR, weight='bold', zorder=3)
    ax.text(2.85, ya_cross-0.10, 'Cross-Attn', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)

    # Split arrow from residual to both cross-attns
    ax.plot([1.95, 2.2, 2.2], [yd, yd, ya_cross], color='#555', linewidth=1.0, zorder=3)
    arrow(ax, 2.2, ya_cross, 2.34, ya_cross, color=ACTION_COLOR)
    ax.plot([2.2, 2.2], [yd, yi_cross], color='#555', linewidth=1.0, zorder=3)
    arrow(ax, 2.2, yi_cross, 2.34, yi_cross, color=IDENTITY_COLOR)

    # H_action input (from top)
    rounded_box(ax, 2.85, 2.85, 0.75, 0.30, '', color=ACTION_COLOR, edgecolor='#2255AA',
                fontsize=FONT_SIZE_SMALL, textcolor='white')
    ax.text(2.85, 2.85, r'$\mathbf{H}_{\mathrm{action}}$', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, color='white', weight='bold', zorder=3)
    arrow(ax, 2.85, 2.70, 2.85, ya_cross + 0.25, color=ACTION_COLOR, linewidth=1.2)

    # Identity Cross-Attention
    rounded_box(ax, 2.85, yi_cross, 1.0, 0.50, '', color='white', edgecolor=IDENTITY_COLOR,
                fontsize=FONT_SIZE_SMALL)
    ax.text(2.85, yi_cross+0.08, 'Identity', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, color=IDENTITY_COLOR, weight='bold', zorder=3)
    ax.text(2.85, yi_cross-0.10, 'Cross-Attn', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, zorder=3)

    # H_identity input (from bottom)
    rounded_box(ax, 2.85, -0.05, 0.75, 0.30, '', color=IDENTITY_COLOR, edgecolor='#AA4466',
                fontsize=FONT_SIZE_SMALL, textcolor='white')
    ax.text(2.85, -0.05, r'$\mathbf{H}_{\mathrm{id}}$', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, color='white', weight='bold', zorder=3)
    arrow(ax, 2.85, 0.10, 2.85, yi_cross - 0.25, color=IDENTITY_COLOR, linewidth=1.2)

    # Labels Z_A and Z_I on output arrows
    ax.text(3.65, ya_cross + 0.12, r'$\mathbf{Z}_A$', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, color=ACTION_COLOR, weight='bold', zorder=5)
    ax.text(3.65, yi_cross + 0.12, r'$\mathbf{Z}_I$', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, color=IDENTITY_COLOR, weight='bold', zorder=5)

    # ---- ADAPTIVE GATE ----
    gx = 4.3
    gy = yd

    # Gate box (bigger, more prominent)
    rounded_box(ax, gx, gy, 0.95, 0.70, '', color=GATE_COLOR, edgecolor='#666600',
                fontsize=FONT_SIZE_SMALL, textcolor='white', bold=True, linewidth=1.5)
    ax.text(gx, gy+0.15, 'Adaptive', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, color='white', weight='bold', zorder=3)
    ax.text(gx, gy-0.05, 'Gate', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, color='white', weight='bold', zorder=3)
    ax.text(gx, gy-0.22, r'$\boldsymbol{\alpha}$', ha='center', va='center',
            fontsize=FONT_SIZE, color='#FFEE88', weight='bold', zorder=3)

    # Arrows from cross-attns to gate (L-shaped for cleanliness)
    ax.plot([3.36, 3.9], [ya_cross, ya_cross], color=ACTION_COLOR, linewidth=1.2, zorder=3)
    arrow(ax, 3.9, ya_cross, 3.9, gy + 0.35, color=ACTION_COLOR, linewidth=1.2)
    ax.plot([3.36, 3.9], [yi_cross, yi_cross], color=IDENTITY_COLOR, linewidth=1.2, zorder=3)
    arrow(ax, 3.9, yi_cross, 3.9, gy - 0.35, color=IDENTITY_COLOR, linewidth=1.2)

    # Gate equation annotation (below gate)
    ax.text(gx, gy - 0.58,
            r'$\boldsymbol{\alpha} \odot \mathbf{Z}_A + (1{-}\boldsymbol{\alpha}) \odot \mathbf{Z}_I$',
            ha='center', va='top', fontsize=FONT_SIZE_TINY, color='#666600',
            style='italic', zorder=5,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='#ccc', alpha=0.9))

    # Residual + after gate
    circle_op(ax, 5.05, yd, '+', radius=0.10, color='#eee', edgecolor='#888',
              fontsize=FONT_SIZE_TINY)
    arrow(ax, gx + 0.48, yd, 4.95, yd, color='#555')

    # Skip connection around cross-attention block (clean L-shape above)
    ax.plot([1.95, 1.95, 5.05], [yd + 0.30, yd + 0.65, yd + 0.65],
            color='#bbb', linewidth=0.7, zorder=2)
    arrow(ax, 5.05, yd + 0.65, 5.05, yd + 0.10, color='#bbb', linewidth=0.7)

    # FFN
    rounded_box(ax, 5.65, yd, 0.55, 0.50, 'FFN', color='white', edgecolor='#555',
                fontsize=FONT_SIZE_SMALL)
    arrow(ax, 5.15, yd, 5.37, yd, color='#555')

    # Residual + after FFN
    circle_op(ax, 6.15, yd, '+', radius=0.10, color='#eee', edgecolor='#888',
              fontsize=FONT_SIZE_TINY)
    arrow(ax, 5.93, yd, 6.05, yd, color='#555', linewidth=0.8)

    # Skip connection around FFN (clean L-shape below)
    ax.plot([5.15, 5.15, 6.15], [yd - 0.30, yd - 0.52, yd - 0.52],
            color='#bbb', linewidth=0.7, zorder=2)
    arrow(ax, 6.15, yd - 0.52, 6.15, yd - 0.10, color='#bbb', linewidth=0.7)

    # Output Frame[n+1]
    rounded_box(ax, 7.05, yd, 0.65, 0.50, '', color=OUTPUT_COLOR, edgecolor='#116622',
                fontsize=FONT_SIZE_SMALL, textcolor='white', bold=True)
    ax.text(7.05, yd+0.10, 'Frame', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, color='white', weight='bold', zorder=3)
    ax.text(7.05, yd-0.10, '$n{+}1$', ha='center', va='center',
            fontsize=FONT_SIZE_SMALL, color='white', weight='bold', zorder=3)
    arrow(ax, 6.25, yd, 6.72, yd, color='#555', linewidth=1.2)

    fig.tight_layout(pad=0.2)
    return fig


if __name__ == '__main__':
    import os
    outdir = 'paper/fig'
    os.makedirs(outdir, exist_ok=True)

    # Encoder figure
    print("Generating encoder architecture figure...")
    fig_enc = create_encoder_figure()
    for ext in ['pdf', 'png']:
        path = os.path.join(outdir, f'Encoder.{ext}')
        fig_enc.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"  Saved {path}")
    plt.close(fig_enc)

    # Decoder figure
    print("Generating decoder architecture figure...")
    fig_dec = create_decoder_figure()
    for ext in ['pdf', 'png']:
        path = os.path.join(outdir, f'Decoder.{ext}')
        fig_dec.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"  Saved {path}")
    plt.close(fig_dec)

    print("\nDone! Check paper/fig/Encoder.{pdf,png} and paper/fig/Decoder.{pdf,png}")
