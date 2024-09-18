#!/usr/bin/env python3
"""
Extra Overlay Visualization Variations (F-J) for Advisor Review

Companion script to visualize_overlay_variations.py (A-E).
Generates additional overlay layouts emphasizing different aspects of
motion retargeting quality.

Only uses STANDING actions (no sitting, bending, falling, picking up).

Variations:
  F) Large single-frame side-by-side pairs: 2x3 grid, each cell shows
     source (blue, semi-transparent) + retargeted (orange, bold) overlaid.
     3 actions, 2 frames each. Zoomed-in for joint detail inspection.
  G) Full method comparison filmstrip: 5 rows x 5 frames for one action.
     Rows: Source-only, Ours overlay, DMR overlay, PMR overlay, Gaussian overlay.
  H) Retargeting triplet: Source Motion | Target Identity | Retargeted overlay.
     3-4 rows for different actions. Shows the full retargeting pipeline visually.
  I) Bone-length comparison: Color-codes each bone by length change between
     source and retargeted. Green=preserved, red=changed. 4 frames in a row.
  J) Motion trajectory overlay: Plots joint trajectories (hands, feet, head)
     over all 64 frames. Source blue, retargeted orange.

Output: paper/fig/qualitative_overlay_v{F,G,H,I,J}.{pdf,png}
"""

import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import parse_file_name, load_data

# NTU bone connections (0-indexed)
NTU_BONES = [
    (1, 0), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

NTU_ACTIONS = {
    1: "Drink water", 7: "Throw", 10: "Clapping", 22: "Cheer up",
    23: "Hand waving", 24: "Kicking", 26: "Hopping", 27: "Jump up",
    28: "Phone call", 31: "Point to something", 34: "Rub two hands",
    38: "Salute", 39: "Put palms together", 40: "Cross hands in front",
    49: "Fan self",
}

# Standing actions only
STANDING_ACTIONS = [23, 10, 24, 27, 38, 31, 1, 22, 26, 34, 49, 40]

METHOD_COLORS = {
    'Source': '#2171b5',
    'Ours': '#d94801',
    'DMR': '#e67e22',
    'PMR': '#e74c3c',
    'Gaussian': '#7f8c8d',
}

# Key joints for trajectory plotting (J)
# 3=head, 7=left hand tip, 11=right hand tip, 15=left foot, 19=right foot
TRAJECTORY_JOINTS = {
    'Head': 3,
    'L. Hand': 7,
    'R. Hand': 11,
    'L. Foot': 15,
    'R. Foot': 19,
}

TRAJECTORY_COLORS = {
    'Head': ('#6baed6', '#fd8d3c'),       # blue/orange pair
    'L. Hand': ('#3182bd', '#e6550d'),
    'R. Hand': ('#08519c', '#a63603'),
    'L. Foot': ('#9ecae1', '#fdae6b'),
    'R. Foot': ('#c6dbef', '#fdd0a2'),
}



def center_at_hip(joints_3d):
    """Center skeleton at hip joint (joint 0) and return a copy."""
    centered = joints_3d.copy()
    hip = centered[0].copy()
    centered -= hip
    return centered


def rotate_to_view(joints_3d, elev_deg=15, azim_deg=45):
    """Rotate 3D joints to a 3/4 viewing angle.

    Applies Y-axis rotation (azimuth) then X-axis rotation (elevation).
    """
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    Ry = np.array([
        [np.cos(az), 0, np.sin(az)],
        [0, 1, 0],
        [-np.sin(az), 0, np.cos(az)],
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(el), -np.sin(el)],
        [0, np.sin(el), np.cos(el)],
    ])
    rotated = (Rx @ Ry @ joints_3d.T).T
    return rotated


def transform_frame(joints_3d):
    """Apply hip-centering and 3/4-view rotation to a single frame."""
    joints = center_at_hip(joints_3d)
    joints = rotate_to_view(joints)
    return joints

def seq_to_joints(seq):
    """Convert (T, V*C) to (T, V, 3)."""
    T = seq.shape[0]
    return seq.reshape(T, 25, 3)


def is_standing(joints):
    """Strict standing check — rejects sitting, crouching, and bent-leg poses."""
    hip_y = (joints[:, 12, 1].mean() + joints[:, 16, 1].mean()) / 2
    knee_y = (joints[:, 13, 1].mean() + joints[:, 17, 1].mean()) / 2
    ankle_y = (joints[:, 14, 1].mean() + joints[:, 18, 1].mean()) / 2
    if (hip_y - knee_y) < 0.08:
        return False
    if (knee_y - ankle_y) < 0.05:
        return False
    l_thigh_dy = np.abs(joints[:, 12, 1] - joints[:, 13, 1]).mean()
    l_thigh_dx = np.abs(joints[:, 12, 0] - joints[:, 13, 0]).mean()
    r_thigh_dy = np.abs(joints[:, 16, 1] - joints[:, 17, 1]).mean()
    r_thigh_dx = np.abs(joints[:, 16, 0] - joints[:, 17, 0]).mean()
    if (l_thigh_dy < 1.5 * l_thigh_dx) or (r_thigh_dy < 1.5 * r_thigh_dx):
        return False
    return True


def draw_skeleton(ax, joints_frame, color='#333333', alpha=1.0, linewidth=2.0,
                  joint_size=12, label=None, linestyle='-'):
    """Draw a 2D skeleton (front view: X horizontal, Y vertical)."""
    transformed = transform_frame(joints_frame)
    x, y = transformed[:, 0], transformed[:, 1]
    for (i, j) in NTU_BONES:
        ax.plot([x[i], x[j]], [y[i], y[j]], c=color, linewidth=linewidth,
                alpha=alpha, solid_capstyle='round', linestyle=linestyle)
    kw = dict(c=color, s=joint_size, zorder=5, alpha=alpha,
              edgecolors='white', linewidths=0.3)
    if label:
        ax.scatter(x, y, label=label, **kw)
    else:
        ax.scatter(x, y, **kw)


def draw_skeleton_colored_bones(ax, joints_frame, bone_colors, alpha=1.0,
                                linewidth=2.5, joint_size=12):
    """Draw skeleton with individually colored bones.

    Args:
        bone_colors: list of (r,g,b,a) tuples, one per bone in NTU_BONES order
    """
    transformed = transform_frame(joints_frame)
    x, y = transformed[:, 0], transformed[:, 1]
    for idx, (i, j) in enumerate(NTU_BONES):
        ax.plot([x[i], x[j]], [y[i], y[j]], c=bone_colors[idx],
                linewidth=linewidth, alpha=alpha, solid_capstyle='round')
    ax.scatter(x, y, c='#444444', s=joint_size, zorder=5, alpha=alpha,
               edgecolors='white', linewidths=0.3)


def setup_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal', adjustable='datalim')


def compute_limits(joints_list, padding=1.15):
    """Compute shared 2D limits after hip-centering and 3/4-view rotation."""
    all_x, all_y = [], []
    for j_seq in joints_list:
        pts = j_seq.reshape(-1, 25, 3)
        for t in range(pts.shape[0]):
            transformed = transform_frame(pts[t])
            all_x.append(transformed[:, 0])
            all_y.append(transformed[:, 1])
    if not all_x:
        return 0, 0, 1
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    cx = (all_x.max() + all_x.min()) / 2
    cy = (all_y.max() + all_y.min()) / 2
    half = max(np.abs(all_x - cx).max(), np.abs(all_y - cy).max()) * padding
    return cx, cy, half


def find_standing_samples(raw_data, methods, dataset, target_actions, n_per_action=2):
    """Find samples that are common across all methods, standing, and have good motion."""
    common_fnames = set(raw_data.keys())
    for data in methods.values():
        common_fnames &= set(data.keys())

    action_candidates = {}
    for fname in common_fnames:
        info = parse_file_name(fname, dataset)
        aid = info['A']
        if aid not in target_actions:
            continue
        joints = seq_to_joints(raw_data[fname])
        if not is_standing(joints):
            continue
        vel = np.diff(joints, axis=0)
        motion = np.sum(np.linalg.norm(vel, axis=-1))
        if aid not in action_candidates:
            action_candidates[aid] = []
        action_candidates[aid].append((fname, motion))

    selected = {}
    for aid in target_actions:
        if aid not in action_candidates:
            continue
        candidates = sorted(action_candidates[aid], key=lambda x: -x[1])
        selected[aid] = [fname for fname, _ in candidates[:n_per_action]]

    return selected


def find_different_identity_sample(raw_data, fname, dataset, target_action):
    """Find a sample with the same action but different person identity.

    Used by Variation H to show the 'target identity'.
    """
    info = parse_file_name(fname, dataset)
    src_person = info['P']
    src_action = info['A']

    candidates = []
    for other_fname in raw_data:
        other_info = parse_file_name(other_fname, dataset)
        if other_info['A'] == src_action and other_info['P'] != src_person:
            joints = seq_to_joints(raw_data[other_fname])
            if is_standing(joints):
                candidates.append(other_fname)
    if candidates:
        return candidates[0]
    return None


def bone_length(joints_frame, i, j):
    """Euclidean distance between two joints in a single frame."""
    return np.linalg.norm(joints_frame[i] - joints_frame[j])


# =========================================================================
# Variation F: Large single-frame side-by-side pairs
# 2 rows x 3 columns. Each cell shows ONE frame with source (blue dashed)
# and retargeted (orange bold) overlaid. 3 actions, 2 frames each.
# =========================================================================
def make_variation_f(raw_data, methods, selected, output_dir):
    print("\n=== Variation F: Large single-frame overlay pairs ===")

    # Pick 3 actions
    actions = sorted([a for a in selected if selected[a]])[:3]
    if len(actions) < 3:
        print("  Warning: fewer than 3 actions available, using what we have")

    # 2 frames per action: early (frame 8) and mid (frame 32)
    frame_pairs = [8, 32]

    n_rows = 2
    n_cols = len(actions)
    cell_size = 4.0  # large cells for detail

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_size, n_rows * cell_size * 1.1))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col, aid in enumerate(actions):
        fname = selected[aid][0]
        src_joints = seq_to_joints(raw_data[fname])
        ours_joints = seq_to_joints(methods['Ours'][fname])
        cx, cy, half = compute_limits([src_joints, ours_joints])
        action_name = NTU_ACTIONS.get(aid, f"A{aid}")

        for row, fi in enumerate(frame_pairs):
            ax = axes[row, col]
            setup_axis(ax)
            t = min(fi, src_joints.shape[0] - 1)

            # Source: semi-transparent blue dashed
            draw_skeleton(ax, src_joints[t], color='#2171b5', alpha=0.40,
                         linewidth=2.0, joint_size=20, linestyle='--',
                         label='Ground Truth' if (row == 0 and col == 0) else None)
            # Retargeted: bold orange solid
            draw_skeleton(ax, ours_joints[t], color='#d94801', alpha=0.95,
                         linewidth=3.0, joint_size=28,
                         label='Retargeted' if (row == 0 and col == 0) else None)

            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

            if row == 0:
                ax.set_title(action_name, fontsize=13, fontweight='bold', pad=8)
            if col == 0:
                ax.set_ylabel(f'Frame {fi}', fontsize=11, rotation=90, labelpad=10)

    axes[0, 0].legend(loc='upper left', fontsize=10, framealpha=0.8,
                      handlelength=1.8, borderpad=0.5, markerscale=0.6)
    plt.tight_layout(pad=0.6, h_pad=0.4, w_pad=0.4)

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_overlay_vF.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# Variation G: Full method comparison filmstrip
# 5 rows x 5 frames for one action.
# Row 0: Source only. Rows 1-4: source (light dashed) + method (bold).
# =========================================================================
def make_variation_g(raw_data, methods, selected, output_dir):
    print("\n=== Variation G: Full method comparison filmstrip ===")

    frame_indices = [8, 20, 32, 48, 60]
    n_frames = len(frame_indices)
    method_names = ['Ours', 'DMR', 'PMR', 'Gaussian']
    row_labels = ['Ground Truth'] + method_names

    # Pick hand waving or kicking
    target_aid = None
    for aid in [23, 24, 10, 27, 38]:
        if aid in selected and selected[aid]:
            target_aid = aid
            break
    if target_aid is None:
        print("  ERROR: No suitable action found")
        return

    fname = selected[target_aid][0]
    action_name = NTU_ACTIONS.get(target_aid, f"A{target_aid}")
    src_joints = seq_to_joints(raw_data[fname])

    # Collect all method joints for shared limits
    all_joints_list = [src_joints]
    method_joints = {}
    for mname in method_names:
        if fname in methods[mname]:
            mj = seq_to_joints(methods[mname][fname])
            method_joints[mname] = mj
            all_joints_list.append(mj)

    cx, cy, half = compute_limits(all_joints_list)

    n_rows = len(row_labels)
    fig, axes = plt.subplots(n_rows, n_frames,
                             figsize=(n_frames * 2.2, n_rows * 2.8))

    for col, fi in enumerate(frame_indices):
        t = min(fi, src_joints.shape[0] - 1)

        for row, rlabel in enumerate(row_labels):
            ax = axes[row, col]
            setup_axis(ax)

            if row == 0:
                # Source-only row
                draw_skeleton(ax, src_joints[t], color=METHOD_COLORS['Source'],
                             alpha=0.85, linewidth=2.0, joint_size=10,
                             label='Ground Truth' if col == 0 else None)
            else:
                # Source as light dashed underlay
                draw_skeleton(ax, src_joints[t], color=METHOD_COLORS['Source'],
                             alpha=0.25, linewidth=1.3, joint_size=4,
                             linestyle='--')
                # Method as bold overlay
                mname = rlabel
                if mname in method_joints:
                    mj = method_joints[mname]
                    mt = min(fi, mj.shape[0] - 1)
                    draw_skeleton(ax, mj[mt], color=METHOD_COLORS[mname],
                                 alpha=0.9, linewidth=2.2, joint_size=10,
                                 label=mname if col == 0 else None)

            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

            if row == 0:
                ax.set_title(f't={fi}', fontsize=9, pad=4)
            if col == 0:
                ax.set_ylabel(rlabel, fontsize=9, fontweight='bold',
                             rotation=90, labelpad=8,
                             color=METHOD_COLORS.get(rlabel, '#333333'))

    fig.suptitle(f'{action_name} -- Method Comparison Over Time',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.4, h_pad=0.2, w_pad=0.2)

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_overlay_vG.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# Variation H: Retargeting triplet
# 3 columns: "Source Motion" | "Target Identity" | "Retargeted (overlay)"
# 3-4 rows for different actions.
# Source Motion: 5 timesteps of the source person doing the action
# Target Identity: frame 0 (standing pose) of a different person
# Retargeted: source (dashed blue) + retargeted (bold orange) overlaid, 5 timesteps
# =========================================================================
def make_variation_h(raw_data, methods, selected, output_dir):
    print("\n=== Variation H: Retargeting triplet ===")

    actions = sorted([a for a in selected if selected[a]])[:4]
    n_actions = len(actions)

    # For source motion and retargeted columns, show 5 mini-frames inside one cell
    motion_frames = [8, 20, 32, 48, 60]

    fig_width = 14
    fig_height = n_actions * 3.2
    fig, axes = plt.subplots(n_actions, 3, figsize=(fig_width, fig_height))
    if n_actions == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['Source Motion', 'Target Identity', 'Retargeted (overlay on source)']

    for row, aid in enumerate(actions):
        fname = selected[aid][0]
        action_name = NTU_ACTIONS.get(aid, f"A{aid}")
        src_joints = seq_to_joints(raw_data[fname])
        ours_joints = seq_to_joints(methods['Ours'][fname])

        # Find a different-identity sample for the "target identity" column
        info = parse_file_name(fname, 'ntu')
        target_fname = find_different_identity_sample(raw_data, fname, 'ntu', aid)

        # Compute shared limits across source + retargeted
        cx, cy, half = compute_limits([src_joints, ours_joints])

        # --- Column 0: Source Motion (5 mini-frames spread horizontally) ---
        ax_src = axes[row, 0]
        setup_axis(ax_src)
        spread = half * 2.2  # horizontal spread between mini-frames
        total_width = spread * (len(motion_frames) - 1)
        for fi_idx, fi in enumerate(motion_frames):
            t = min(fi, src_joints.shape[0] - 1)
            frame_joints = src_joints[t].copy()
            # Offset horizontally
            x_offset = fi_idx * spread - total_width / 2
            frame_joints[:, 0] += x_offset
            draw_skeleton(ax_src, frame_joints, color=METHOD_COLORS['Source'],
                         alpha=0.7, linewidth=1.5, joint_size=5)
        # Set limits for the spread-out view
        ax_src.set_xlim(cx - total_width / 2 - half, cx + total_width / 2 + half)
        ax_src.set_ylim(cy - half, cy + half)
        ax_src.set_ylabel(action_name, fontsize=9, fontweight='bold',
                         rotation=90, labelpad=8)

        # --- Column 1: Target Identity (single standing pose, frame 8) ---
        ax_tgt = axes[row, 1]
        setup_axis(ax_tgt)
        if target_fname is not None:
            tgt_joints = seq_to_joints(raw_data[target_fname])
            tgt_cx, tgt_cy, tgt_half = compute_limits([tgt_joints])
            draw_skeleton(ax_tgt, tgt_joints[min(8, tgt_joints.shape[0]-1)], color='#2ca02c',
                         alpha=0.85, linewidth=2.0, joint_size=12,
                         label='Target ID' if row == 0 else None)
            ax_tgt.set_xlim(tgt_cx - tgt_half, tgt_cx + tgt_half)
            ax_tgt.set_ylim(tgt_cy - tgt_half, tgt_cy + tgt_half)
        else:
            # Fallback: show retargeted frame 8 as a proxy for target identity
            draw_skeleton(ax_tgt, ours_joints[min(8, ours_joints.shape[0]-1)], color='#2ca02c',
                         alpha=0.85, linewidth=2.0, joint_size=12,
                         label='Target ID' if row == 0 else None)
            ax_tgt.set_xlim(cx - half, cx + half)
            ax_tgt.set_ylim(cy - half, cy + half)

        # --- Column 2: Retargeted overlay (5 mini-frames, source dashed + ours bold) ---
        ax_ret = axes[row, 2]
        setup_axis(ax_ret)
        for fi_idx, fi in enumerate(motion_frames):
            t = min(fi, src_joints.shape[0] - 1)
            x_offset = fi_idx * spread - total_width / 2

            # Source dashed underlay
            src_frame = src_joints[t].copy()
            src_frame[:, 0] += x_offset
            draw_skeleton(ax_ret, src_frame, color=METHOD_COLORS['Source'],
                         alpha=0.30, linewidth=1.2, joint_size=3, linestyle='--')

            # Retargeted bold overlay
            ours_t = min(fi, ours_joints.shape[0] - 1)
            ours_frame = ours_joints[ours_t].copy()
            ours_frame[:, 0] += x_offset
            draw_skeleton(ax_ret, ours_frame, color=METHOD_COLORS['Ours'],
                         alpha=0.85, linewidth=1.8, joint_size=7)

        ax_ret.set_xlim(cx - total_width / 2 - half, cx + total_width / 2 + half)
        ax_ret.set_ylim(cy - half, cy + half)

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=8)

    plt.tight_layout(pad=0.5, h_pad=0.4, w_pad=0.3)

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_overlay_vH.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# Variation I: Bone-length comparison
# For one action, show the retargeted skeleton at 4 frames.
# Each bone is colored by how much its length changed vs the source.
# Green = same length, Red = large change.
# =========================================================================
def make_variation_i(raw_data, methods, selected, output_dir):
    print("\n=== Variation I: Bone-length comparison ===")

    frame_indices = [8, 24, 40, 60]
    n_frames = len(frame_indices)

    # Pick an action
    target_aid = None
    for aid in [23, 10, 24, 27, 38, 1]:
        if aid in selected and selected[aid]:
            target_aid = aid
            break
    if target_aid is None:
        print("  ERROR: No suitable action found")
        return

    fname = selected[target_aid][0]
    action_name = NTU_ACTIONS.get(target_aid, f"A{target_aid}")
    src_joints = seq_to_joints(raw_data[fname])
    ours_joints = seq_to_joints(methods['Ours'][fname])
    cx, cy, half = compute_limits([src_joints, ours_joints])

    # Compute bone length ratios across all frames to set color scale
    all_ratios = []
    for t in range(min(src_joints.shape[0], ours_joints.shape[0])):
        for (i, j) in NTU_BONES:
            src_len = bone_length(src_joints[t], i, j)
            ours_len = bone_length(ours_joints[t], i, j)
            if src_len > 1e-6:
                ratio = abs(ours_len - src_len) / src_len
                all_ratios.append(ratio)

    # Normalize: 0 = no change (green), max_ratio = max change (red)
    max_ratio = np.percentile(all_ratios, 95) if all_ratios else 0.5
    max_ratio = max(max_ratio, 0.01)  # avoid div by zero

    # Use RdYlGn_r: green=low, yellow=mid, red=high
    cmap = cm.get_cmap('RdYlGn_r')
    norm = Normalize(vmin=0, vmax=max_ratio)

    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 3.5, 4.5))

    for col, fi in enumerate(frame_indices):
        ax = axes[col]
        setup_axis(ax)
        t = min(fi, min(src_joints.shape[0], ours_joints.shape[0]) - 1)

        # Compute per-bone color
        bone_colors = []
        for (i, j) in NTU_BONES:
            src_len = bone_length(src_joints[t], i, j)
            ours_len = bone_length(ours_joints[t], i, j)
            if src_len > 1e-6:
                ratio = abs(ours_len - src_len) / src_len
            else:
                ratio = 0.0
            bone_colors.append(cmap(norm(ratio)))

        # Draw source as light gray reference
        draw_skeleton(ax, src_joints[t], color='#bbbbbb', alpha=0.35,
                     linewidth=1.5, joint_size=5, linestyle='--')

        # Draw retargeted with colored bones
        draw_skeleton_colored_bones(ax, ours_joints[t], bone_colors,
                                    alpha=0.95, linewidth=3.0, joint_size=14)

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(f'Frame {fi}', fontsize=10, pad=5)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.04, pad=0.12, aspect=40)
    cbar.set_label('Relative Bone Length Change', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f'{action_name} -- Bone Length Changes (Source=gray dashed, '
                 f'Retargeted=colored)',
                 fontsize=11, fontweight='bold', y=1.04)
    plt.tight_layout(pad=0.5)

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_overlay_vI.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# Variation J: Motion trajectory overlay
# For one action, plot the trajectory of key joints over all 64 frames.
# Source trajectories in blue shades, retargeted in orange shades.
# Single large plot.
# =========================================================================
def make_variation_j(raw_data, methods, selected, output_dir):
    print("\n=== Variation J: Motion trajectory overlay ===")

    # Pick an action with good visible motion
    target_aid = None
    for aid in [23, 24, 27, 10, 38, 1]:
        if aid in selected and selected[aid]:
            target_aid = aid
            break
    if target_aid is None:
        print("  ERROR: No suitable action found")
        return

    fname = selected[target_aid][0]
    action_name = NTU_ACTIONS.get(target_aid, f"A{target_aid}")
    src_joints = seq_to_joints(raw_data[fname])
    ours_joints = seq_to_joints(methods['Ours'][fname])

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Draw reference skeleton at frame 8 (very light, for spatial context)
    draw_skeleton(ax, src_joints[min(8, src_joints.shape[0]-1)], color='#cccccc', alpha=0.20,
                 linewidth=1.0, joint_size=3, linestyle=':')

    n_frames = min(src_joints.shape[0], ours_joints.shape[0])
    legend_handles = []

    for joint_name, joint_idx in TRAJECTORY_JOINTS.items():
        src_color, ours_color = TRAJECTORY_COLORS[joint_name]

        # Source trajectory
        src_traj_x = src_joints[:n_frames, joint_idx, 0]
        src_traj_y = src_joints[:n_frames, joint_idx, 1]
        h_src, = ax.plot(src_traj_x, src_traj_y, color=src_color, alpha=0.7,
                         linewidth=1.8, linestyle='-',
                         label=f'{joint_name} (Source)')

        # Retargeted trajectory
        ours_traj_x = ours_joints[:n_frames, joint_idx, 0]
        ours_traj_y = ours_joints[:n_frames, joint_idx, 1]
        h_ours, = ax.plot(ours_traj_x, ours_traj_y, color=ours_color, alpha=0.7,
                          linewidth=1.8, linestyle='--',
                          label=f'{joint_name} (Retargeted)')

        # Mark start (circle) and end (triangle) for each trajectory
        ax.scatter([src_traj_x[0]], [src_traj_y[0]], color=src_color,
                   marker='o', s=40, zorder=6, edgecolors='white', linewidths=0.5)
        ax.scatter([src_traj_x[-1]], [src_traj_y[-1]], color=src_color,
                   marker='^', s=40, zorder=6, edgecolors='white', linewidths=0.5)
        ax.scatter([ours_traj_x[0]], [ours_traj_y[0]], color=ours_color,
                   marker='o', s=40, zorder=6, edgecolors='white', linewidths=0.5)
        ax.scatter([ours_traj_x[-1]], [ours_traj_y[-1]], color=ours_color,
                   marker='^', s=40, zorder=6, edgecolors='white', linewidths=0.5)

        legend_handles.extend([h_src, h_ours])

    # Format axes
    ax.set_xlabel('X position', fontsize=10)
    ax.set_ylabel('Y position', fontsize=10)
    ax.set_title(f'{action_name} -- Joint Trajectories Over {n_frames} Frames\n'
                 f'(solid=Source, dashed=Retargeted; o=start, ^=end)',
                 fontsize=11, fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
              framealpha=0.8, ncol=2, borderpad=0.4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Light spines
    for spine in ax.spines.values():
        spine.set_alpha(0.3)

    plt.tight_layout(pad=0.8)

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_overlay_vJ.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Extra overlay visualization variations (F-J)')
    parser.add_argument('--raw_data_path', type=str,
                        default='data/ntu/ntu_cv_paired_10k.pt')
    parser.add_argument('--ours_pkl', type=str,
                        default='output/retargeted_data/disentangled_tmr_stable_retargeted.pkl')
    parser.add_argument('--dmr_pkl', type=str,
                        default='output/retargeted_data/dmr_ntu_cv_retargeted.pkl')
    parser.add_argument('--pmr_pkl', type=str,
                        default='output/retargeted_data/pmr_ntu_cv_retargeted.pkl')
    parser.add_argument('--noise_pkl', type=str,
                        default='output/retargeted_data/noise_baseline_retargeted.pkl')
    parser.add_argument('--output_dir', type=str, default='paper/fig')
    parser.add_argument('--dataset', type=str, default='ntu')
    parser.add_argument('--variations', type=str, nargs='+',
                        default=['F', 'G', 'H', 'I', 'J'],
                        help='Which variations to generate (F, G, H, I, J)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading raw data...")
    raw_data = load_data(args.dataset, T=64)

    print("Loading retargeted datasets...")
    methods = {}
    for name, path in [('Ours', args.ours_pkl), ('DMR', args.dmr_pkl),
                        ('PMR', args.pmr_pkl), ('Gaussian', args.noise_pkl)]:
        print(f"  Loading {name} from {path}...")
        with open(path, 'rb') as f:
            methods[name] = pickle.load(f)
        print(f"    {len(methods[name])} samples")

    print("\nFinding standing samples...")
    selected = find_standing_samples(raw_data, methods, args.dataset,
                                     STANDING_ACTIONS, n_per_action=2)
    for aid, fnames in sorted(selected.items()):
        action_name = NTU_ACTIONS.get(aid, f"A{aid}")
        print(f"  Action {aid} ({action_name}): {', '.join(fnames)}")

    if not selected:
        print("ERROR: No standing samples found!")
        return

    variation_funcs = {
        'F': make_variation_f,
        'G': make_variation_g,
        'H': make_variation_h,
        'I': make_variation_i,
        'J': make_variation_j,
    }

    for v in args.variations:
        v = v.upper()
        if v in variation_funcs:
            variation_funcs[v](raw_data, methods, selected, args.output_dir)
        else:
            print(f"Unknown variation: {v}")

    print("\nDone! All variations saved to", args.output_dir)


if __name__ == '__main__':
    main()
