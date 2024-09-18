#!/usr/bin/env python3
"""
Overlay Visualization Variations for Advisor Review

Generates multiple overlay layouts where the source/ground-truth skeleton
is drawn underneath the retargeted skeleton so differences are immediately visible.

Only uses STANDING actions (no sitting, pick up, falling, etc.).

Variations:
  A) Filmstrip overlay: rows=actions, columns=timesteps, source+ours overlaid per frame
  B) Method comparison overlay: rows=actions, cols=methods, source ghost + retargeted overlaid (single mid-frame)
  C) Dense filmstrip: one action, many frames, source+ours overlaid — shows motion preservation clearly
  D) Dual-skeleton filmstrip: source top row, retargeted bottom row, same frames — easiest direct comparison

Output: paper/fig/qualitative_overlay_v{A,B,C,D}.{pdf,png}
"""

import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

BODY_PART_BONES = {
    'torso': [(1, 0), (1, 20), (20, 2)],
    'head': [(2, 3)],
    'l_arm': [(20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21)],
    'r_arm': [(20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23)],
    'l_leg': [(0, 12), (12, 13), (13, 14), (14, 15)],
    'r_leg': [(0, 16), (16, 17), (17, 18), (18, 19)],
}

NTU_ACTIONS = {
    1: "Drink water", 7: "Throw", 10: "Clapping", 22: "Cheer up",
    23: "Hand waving", 24: "Kicking", 26: "Hopping", 27: "Jump up",
    28: "Phone call", 31: "Point to something", 34: "Rub two hands",
    38: "Salute", 39: "Put palms together", 40: "Cross hands in front",
    49: "Fan self",
}

# Standing actions only — no sitting, bending, falling, picking up
STANDING_ACTIONS = [23, 10, 24, 27, 38, 31, 1, 22, 26, 34, 49, 40]

METHOD_COLORS = {
    'Source': '#2171b5',
    'Ours': '#d94801',
    'DMR': '#e67e22',
    'PMR': '#e74c3c',
    'Gaussian': '#7f8c8d',
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
    """Strict standing check — rejects sitting, crouching, and bent-leg poses.

    Checks averaged over all frames:
    1. Hip must be well above knees (vertical thigh)
    2. Knees must be above ankles
    3. Thigh must be more vertical than horizontal (angle check)
    """
    # NTU joints: 0=base spine, 12=left hip, 13=left knee, 14=left ankle,
    #             16=right hip, 17=right knee, 18=right ankle
    hip_y = (joints[:, 12, 1].mean() + joints[:, 16, 1].mean()) / 2
    knee_y = (joints[:, 13, 1].mean() + joints[:, 17, 1].mean()) / 2
    ankle_y = (joints[:, 14, 1].mean() + joints[:, 18, 1].mean()) / 2

    # 1. Hip must be clearly above knees
    hip_knee_gap = hip_y - knee_y
    if hip_knee_gap < 0.08:
        return False

    # 2. Knees must be above ankles
    knee_ankle_gap = knee_y - ankle_y
    if knee_ankle_gap < 0.05:
        return False

    # 3. Thigh angle: vertical component should dominate horizontal
    # Average over frames
    l_thigh_dy = np.abs(joints[:, 12, 1] - joints[:, 13, 1]).mean()
    l_thigh_dx = np.abs(joints[:, 12, 0] - joints[:, 13, 0]).mean()
    r_thigh_dy = np.abs(joints[:, 16, 1] - joints[:, 17, 1]).mean()
    r_thigh_dx = np.abs(joints[:, 16, 0] - joints[:, 17, 0]).mean()
    # Vertical component should be at least 1.5x the horizontal
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


def find_standing_samples(raw_data, methods, dataset, target_actions, n_per_action=1):
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

    # Sort by motion, pick top n_per_action
    selected = {}
    for aid in target_actions:
        if aid not in action_candidates:
            continue
        candidates = sorted(action_candidates[aid], key=lambda x: -x[1])
        selected[aid] = [fname for fname, _ in candidates[:n_per_action]]

    return selected


# =========================================================================
# Variation A: Filmstrip overlay (rows=actions, cols=frames)
# Each cell: source skeleton (light) + ours (bold) overlaid at that frame
# =========================================================================
def make_variation_a(raw_data, methods, selected, output_dir):
    print("\n=== Variation A: Filmstrip overlay ===")
    frame_indices = [8, 16, 28, 40, 56]
    n_frames = len(frame_indices)
    actions = sorted([a for a in selected if selected[a]])[:4]
    n_actions = len(actions)

    fig, axes = plt.subplots(n_actions, n_frames, figsize=(n_frames * 2.2, n_actions * 2.8))
    if n_actions == 1:
        axes = axes.reshape(1, -1)

    for row, aid in enumerate(actions):
        fname = selected[aid][0]
        src_joints = seq_to_joints(raw_data[fname])
        ours_joints = seq_to_joints(methods['Ours'][fname])
        cx, cy, half = compute_limits([src_joints, ours_joints])

        for col, fi in enumerate(frame_indices):
            ax = axes[row, col]
            setup_axis(ax)
            t = min(fi, src_joints.shape[0] - 1)

            # Ground truth underneath (light dashed)
            draw_skeleton(ax, src_joints[t], color='#2171b5', alpha=0.35,
                         linewidth=1.5, joint_size=6, linestyle='--',
                         label='Ground Truth' if (row == 0 and col == 0) else None)
            # Ours on top (bold solid)
            draw_skeleton(ax, ours_joints[t], color='#d94801', alpha=0.9,
                         linewidth=2.2, joint_size=10,
                         label='Ours' if (row == 0 and col == 0) else None)

            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

            if row == 0:
                ax.set_title(f't={fi}', fontsize=9, pad=4)
            if col == 0:
                action_name = NTU_ACTIONS.get(aid, f"A{aid}")
                ax.set_ylabel(action_name, fontsize=8, rotation=90, labelpad=8)

    axes[0, 0].legend(loc='upper left', fontsize=7, framealpha=0.7,
                      handlelength=1.5, borderpad=0.3)
    plt.tight_layout(pad=0.4, h_pad=0.2, w_pad=0.2)

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_overlay_vA.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# Variation B: Method comparison overlay (rows=actions, cols=methods)
# Each cell: source (light) + method (bold) at a single representative frame
# =========================================================================
def make_variation_b(raw_data, methods, selected, output_dir):
    print("\n=== Variation B: Method comparison overlay ===")
    method_names = ['Ours', 'DMR', 'PMR', 'Gaussian']
    actions = sorted([a for a in selected if selected[a]])[:3]
    n_actions = len(actions)
    # Use frame near the middle of the action for most informative pose
    target_frame = 32

    fig, axes = plt.subplots(n_actions, len(method_names),
                             figsize=(len(method_names) * 2.5, n_actions * 3.0))
    if n_actions == 1:
        axes = axes.reshape(1, -1)

    for row, aid in enumerate(actions):
        fname = selected[aid][0]
        src_joints = seq_to_joints(raw_data[fname])
        all_joints = [src_joints]
        method_joints = {}
        for mname in method_names:
            if fname in methods[mname]:
                mj = seq_to_joints(methods[mname][fname])
                method_joints[mname] = mj
                all_joints.append(mj)

        cx, cy, half = compute_limits(all_joints)

        for col, mname in enumerate(method_names):
            ax = axes[row, col]
            setup_axis(ax)
            t = min(target_frame, src_joints.shape[0] - 1)

            # Source underneath
            draw_skeleton(ax, src_joints[t], color='#2171b5', alpha=0.30,
                         linewidth=1.5, joint_size=5, linestyle='--',
                         label='Ground Truth' if (row == 0 and col == 0) else None)

            # Method on top
            if mname in method_joints:
                mj = method_joints[mname]
                mt = min(target_frame, mj.shape[0] - 1)
                draw_skeleton(ax, mj[mt], color=METHOD_COLORS[mname], alpha=0.9,
                             linewidth=2.2, joint_size=10,
                             label=mname if row == 0 else None)

            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

            if row == 0:
                ax.set_title(mname, fontsize=10, fontweight='bold', pad=6)
            if col == 0:
                action_name = NTU_ACTIONS.get(aid, f"A{aid}")
                ax.set_ylabel(action_name, fontsize=8, rotation=90, labelpad=8)

    axes[0, 0].legend(loc='upper left', fontsize=7, framealpha=0.7,
                      handlelength=1.5, borderpad=0.3)
    plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3)

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_overlay_vB.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# Variation C: Dense filmstrip for one action (many frames, source+ours)
# Single row, 8 frames, very clear motion preservation
# =========================================================================
def make_variation_c(raw_data, methods, selected, output_dir):
    print("\n=== Variation C: Dense single-action filmstrip ===")
    frame_indices = [8, 16, 24, 32, 40, 48, 56, 62]
    n_frames = len(frame_indices)

    # Pick hand waving or first available standing action
    for aid in [23, 10, 27, 24, 38]:
        if aid in selected and selected[aid]:
            break

    fname = selected[aid][0]
    action_name = NTU_ACTIONS.get(aid, f"A{aid}")
    src_joints = seq_to_joints(raw_data[fname])
    ours_joints = seq_to_joints(methods['Ours'][fname])
    cx, cy, half = compute_limits([src_joints, ours_joints])

    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 1.8, 3.2))

    for col, fi in enumerate(frame_indices):
        ax = axes[col]
        setup_axis(ax)
        t = min(fi, src_joints.shape[0] - 1)

        draw_skeleton(ax, src_joints[t], color='#2171b5', alpha=0.30,
                     linewidth=1.5, joint_size=5, linestyle='--',
                     label='Ground Truth' if col == 0 else None)
        draw_skeleton(ax, ours_joints[t], color='#d94801', alpha=0.9,
                     linewidth=2.2, joint_size=10,
                     label='Retargeted' if col == 0 else None)

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(f't={fi}', fontsize=8, pad=3)

    axes[0].legend(loc='upper left', fontsize=7, framealpha=0.7,
                   handlelength=1.5, borderpad=0.3)
    fig.suptitle(action_name, fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout(pad=0.3, w_pad=0.2)

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_overlay_vC.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# Variation D: Two-row filmstrip (top=source, bottom=ours, same frames)
# Easiest side-by-side comparison — same scale, aligned vertically
# =========================================================================
def make_variation_d(raw_data, methods, selected, output_dir):
    print("\n=== Variation D: Dual-row filmstrip ===")
    frame_indices = [8, 16, 26, 36, 48, 58]
    n_frames = len(frame_indices)
    actions = sorted([a for a in selected if selected[a]])[:2]

    for aid in actions:
        fname = selected[aid][0]
        action_name = NTU_ACTIONS.get(aid, f"A{aid}")
        src_joints = seq_to_joints(raw_data[fname])
        ours_joints = seq_to_joints(methods['Ours'][fname])
        cx, cy, half = compute_limits([src_joints, ours_joints])

        fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 2.0, 5.5))

        for col, fi in enumerate(frame_indices):
            t = min(fi, src_joints.shape[0] - 1)

            # Top row: Source
            ax_top = axes[0, col]
            setup_axis(ax_top)
            draw_skeleton(ax_top, src_joints[t], color='#2171b5', alpha=0.85,
                         linewidth=2.0, joint_size=10)
            ax_top.set_xlim(cx - half, cx + half)
            ax_top.set_ylim(cy - half, cy + half)
            if col == 0:
                ax_top.set_ylabel('Ground Truth', fontsize=10, fontweight='bold',
                                 rotation=90, labelpad=10)
            ax_top.set_title(f't={fi}', fontsize=8, pad=3)

            # Bottom row: Ours (retargeted)
            ax_bot = axes[1, col]
            setup_axis(ax_bot)
            # Draw source as ghost underneath for reference
            draw_skeleton(ax_bot, src_joints[t], color='#2171b5', alpha=0.15,
                         linewidth=1.0, joint_size=3, linestyle='--')
            draw_skeleton(ax_bot, ours_joints[t], color='#d94801', alpha=0.85,
                         linewidth=2.0, joint_size=10)
            ax_bot.set_xlim(cx - half, cx + half)
            ax_bot.set_ylim(cy - half, cy + half)
            if col == 0:
                ax_bot.set_ylabel('Retargeted', fontsize=10, fontweight='bold',
                                 rotation=90, labelpad=10, color='#d94801')

        fig.suptitle(action_name, fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout(pad=0.4, h_pad=0.3, w_pad=0.2)

        safe_name = action_name.lower().replace(' ', '_')
        for ext in ['pdf']:
            path = os.path.join(output_dir, f'qualitative_overlay_vD_{safe_name}.{ext}')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {path}")
        plt.close()


# =========================================================================
# Variation E: Combined overlay + method comparison
# 2 actions x 6 frames, each frame overlays source+ours; below that,
# a single-frame all-methods comparison row
# =========================================================================
def make_variation_e(raw_data, methods, selected, output_dir):
    print("\n=== Variation E: Overlay filmstrip + method comparison ===")
    frame_indices = [8, 18, 28, 38, 48, 58]
    n_frames = len(frame_indices)
    method_names = ['Ours', 'DMR', 'PMR', 'Gaussian']

    actions = sorted([a for a in selected if selected[a]])[:2]
    n_actions = len(actions)
    # Layout: n_actions rows of filmstrip + 1 row of method comparison per action
    total_rows = n_actions * 2
    n_cols = max(n_frames, len(method_names))

    fig = plt.figure(figsize=(n_cols * 2.0, total_rows * 2.8))
    gs = fig.add_gridspec(total_rows, n_cols, hspace=0.4, wspace=0.15)

    for act_idx, aid in enumerate(actions):
        fname = selected[aid][0]
        action_name = NTU_ACTIONS.get(aid, f"A{aid}")
        src_joints = seq_to_joints(raw_data[fname])
        ours_joints = seq_to_joints(methods['Ours'][fname])

        all_j = [src_joints, ours_joints]
        for mname in method_names[1:]:
            if fname in methods[mname]:
                all_j.append(seq_to_joints(methods[mname][fname]))
        cx, cy, half = compute_limits(all_j)

        # Row 1: filmstrip overlay (source + ours)
        row_film = act_idx * 2
        for col, fi in enumerate(frame_indices):
            ax = fig.add_subplot(gs[row_film, col])
            setup_axis(ax)
            t = min(fi, src_joints.shape[0] - 1)
            draw_skeleton(ax, src_joints[t], color='#2171b5', alpha=0.30,
                         linewidth=1.3, joint_size=4, linestyle='--')
            draw_skeleton(ax, ours_joints[t], color='#d94801', alpha=0.9,
                         linewidth=2.0, joint_size=9)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            if col == 0:
                ax.set_ylabel(f'{action_name}\n(overlay)', fontsize=7,
                             rotation=90, labelpad=6)
            ax.set_title(f't={fi}', fontsize=7, pad=2)

        # Row 2: method comparison at mid-frame
        row_comp = act_idx * 2 + 1
        target_frame = 32
        # Center the method columns
        offset = (n_cols - len(method_names)) // 2
        for col, mname in enumerate(method_names):
            ax = fig.add_subplot(gs[row_comp, col + offset])
            setup_axis(ax)
            t = min(target_frame, src_joints.shape[0] - 1)
            draw_skeleton(ax, src_joints[t], color='#2171b5', alpha=0.25,
                         linewidth=1.3, joint_size=4, linestyle='--')
            if fname in methods[mname]:
                mj = seq_to_joints(methods[mname][fname])
                mt = min(target_frame, mj.shape[0] - 1)
                draw_skeleton(ax, mj[mt], color=METHOD_COLORS[mname], alpha=0.9,
                             linewidth=2.0, joint_size=9)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_title(mname, fontsize=8, fontweight='bold', pad=3)
            if col == 0:
                ax.set_ylabel('methods\n(t=32)', fontsize=7, rotation=90, labelpad=6)

    plt.tight_layout(pad=0.4)

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_overlay_vE.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
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
                        default=['A', 'B', 'C', 'D', 'E'],
                        help='Which variations to generate')
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
                                     STANDING_ACTIONS, n_per_action=1)
    for aid, fnames in sorted(selected.items()):
        action_name = NTU_ACTIONS.get(aid, f"A{aid}")
        print(f"  Action {aid} ({action_name}): {fnames[0]}")

    if not selected:
        print("ERROR: No standing samples found!")
        return

    variation_funcs = {
        'A': make_variation_a,
        'B': make_variation_b,
        'C': make_variation_c,
        'D': make_variation_d,
        'E': make_variation_e,
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
