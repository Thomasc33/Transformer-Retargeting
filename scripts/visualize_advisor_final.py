#!/usr/bin/env python3
"""
Advisor-requested qualitative figure.

Shows the full retargeting pipeline and method comparison:
  - Source Motion (P1 doing action A)
  - Target Identity (P2 standing)
  - Results: GT ghost + each method's output overlaid

Criteria:
  (1) Shows the retargeting task: P1A1 + P2 → P2A1
  (2) Our retargeted output clearly performs the action
  (3) Ours looks better than baselines
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

NTU_BONES = [
    (1, 0), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

NTU_ACTIONS = {
    1: "Drink water", 10: "Clapping", 22: "Cheer up",
    23: "Hand waving", 24: "Kicking", 26: "Hopping", 27: "Jump up",
    31: "Point to something", 34: "Rub two hands",
    38: "Salute", 40: "Cross hands in front", 49: "Fan self",
}

STANDING_ACTIONS = [23, 10, 24, 27, 38, 31, 1, 22, 26, 34, 49, 40]

METHOD_COLORS = {
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


def rotate_to_view(joints_3d, elev_deg=10, azim_deg=15):
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
    T = seq.shape[0]
    return seq.reshape(T, 25, 3)


def is_standing(joints):
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


def find_target_identity(raw_data, source_fname, dataset):
    """Find a standing sample from a DIFFERENT person than the source."""
    src_info = parse_file_name(source_fname, dataset)
    src_person = src_info['P']

    candidates = []
    for fname, seq in raw_data.items():
        info = parse_file_name(fname, dataset)
        if info['P'] == src_person:
            continue
        joints = seq_to_joints(seq)
        if is_standing(joints):
            candidates.append((fname, joints))
        if len(candidates) >= 20:
            break

    if not candidates:
        return None, None, None
    # Pick the one with the most different bone lengths (visually distinct)
    fname, joints = candidates[0]
    info = parse_file_name(fname, dataset)
    return fname, joints, info['P']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ntu')
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
    parser.add_argument('--actions', type=int, nargs='+', default=[23, 27, 24])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading raw data...")
    raw_data = load_data(args.dataset, T=64)

    print("Loading retargeted datasets...")
    methods = {}
    for name, path in [('Ours', args.ours_pkl), ('DMR', args.dmr_pkl),
                        ('PMR', args.pmr_pkl), ('Gaussian', args.noise_pkl)]:
        with open(path, 'rb') as f:
            methods[name] = pickle.load(f)
        print(f"  {name}: {len(methods[name])} samples")

    # Find standing samples for requested actions
    common_fnames = set(raw_data.keys())
    for mname in methods:
        common_fnames &= set(methods[mname].keys())

    method_names = ['Ours', 'DMR', 'PMR', 'Gaussian']
    frame_indices = [8, 24, 40, 56]
    n_frames = len(frame_indices)
    n_actions = len(args.actions)

    # Find one good standing sample per action
    selected = {}
    for aid in args.actions:
        for fname in common_fnames:
            info = parse_file_name(fname, args.dataset)
            if info['A'] != aid:
                continue
            joints = seq_to_joints(raw_data[fname])
            if is_standing(joints):
                selected[aid] = fname
                break

    actions = [a for a in args.actions if a in selected]
    n_actions = len(actions)
    print(f"Selected {n_actions} actions: {[NTU_ACTIONS.get(a, a) for a in actions]}")

    # =====================================================================
    # Figure: Full retargeting pipeline + method comparison
    # Columns: Source Motion (multi-frame) | Target ID | Ours | DMR | PMR | Gaussian
    # For method columns, show 4 time steps with GT ghost underneath
    # =====================================================================

    # Layout: rows = actions, columns grouped as:
    #   [source x n_frames] [target x 1] [gap] [ours x n_frames] [dmr x n_frames] [pmr x n_frames] [gauss x n_frames]
    # That's too wide. Instead:
    #   Row per action, two-row sub-layout:
    #     Top: Source Motion frames | arrow | Target ID
    #     Bottom: Ours frames | DMR frames | PMR frames | Gaussian frames (all overlaid on GT)
    # Still too wide. Let's do a clean grid:
    #
    # For each action, one wide row:
    #   Col 0: Source (t=24, single frame, blue)
    #   Col 1: Target ID (single frame, green)
    #   Col 2: Arrow/label
    #   Col 3-6: Ours at 4 timesteps (GT ghost + retargeted bold)
    #   Col 7-10: DMR at 4 timesteps
    #   ... too wide
    #
    # Simplest effective layout:
    #   For each action, columns = Source | Target ID | Ours | DMR | PMR | Gaussian
    #   Each "method" cell shows a representative mid-frame overlaid on GT ghost
    #   Plus small inset frames for temporal context

    # Version 1: Single representative frame per method
    fig_width = 14
    fig_height = n_actions * 3.0 + 0.8
    fig, axes = plt.subplots(n_actions, 6, figsize=(fig_width, fig_height))
    if n_actions == 1:
        axes = axes.reshape(1, -1)

    col_labels = ['Source\nMotion', 'Target\nIdentity', 'Ours', 'DMR', 'PMR', 'Gaussian']
    rep_frame = 32  # representative mid-sequence frame

    for row, aid in enumerate(actions):
        fname = selected[aid]
        gt_joints = seq_to_joints(raw_data[fname])
        action_name = NTU_ACTIONS.get(aid, f"A{aid}")
        src_info = parse_file_name(fname, args.dataset)

        # Find target identity
        tgt_fname, tgt_joints, tgt_person = find_target_identity(
            raw_data, fname, args.dataset)

        # Compute shared limits across all methods
        all_j = [gt_joints]
        for mname in method_names:
            if fname in methods[mname]:
                all_j.append(seq_to_joints(methods[mname][fname]))
        cx, cy, half = compute_limits(all_j)

        t = min(rep_frame, gt_joints.shape[0] - 1)

        # Col 0: Source Motion (original skeleton, blue)
        ax = axes[row, 0]
        setup_axis(ax)
        draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.9,
                     linewidth=2.2, joint_size=10)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_ylabel(action_name, fontsize=11, fontweight='bold',
                     rotation=90, labelpad=10)
        if row == 0:
            ax.set_title(col_labels[0], fontsize=9, fontweight='bold', pad=8)
        # Label person
        ax.text(0.5, -0.02, f'P{src_info["P"]}', transform=ax.transAxes,
                fontsize=7, ha='center', va='top', color='#2171b5')

        # Col 1: Target Identity (different person, green, standing pose)
        ax = axes[row, 1]
        setup_axis(ax)
        if tgt_joints is not None:
            tgt_t = min(8, tgt_joints.shape[0] - 1)
            tgt_cx, tgt_cy, tgt_half = compute_limits([tgt_joints])
            draw_skeleton(ax, tgt_joints[tgt_t], color='#2ca02c', alpha=0.9,
                         linewidth=2.2, joint_size=10)
            ax.set_xlim(tgt_cx - tgt_half, tgt_cx + tgt_half)
            ax.set_ylim(tgt_cy - tgt_half, tgt_cy + tgt_half)
            ax.text(0.5, -0.02, f'P{tgt_person}', transform=ax.transAxes,
                    fontsize=7, ha='center', va='top', color='#2ca02c')
        if row == 0:
            ax.set_title(col_labels[1], fontsize=9, fontweight='bold', pad=8)

        # Cols 2-5: Methods overlaid on GT ghost
        for col_idx, mname in enumerate(method_names):
            ax = axes[row, col_idx + 2]
            setup_axis(ax)

            # GT ghost
            draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.20,
                         linewidth=1.5, joint_size=4, linestyle='--')

            # Method result
            if fname in methods[mname]:
                mj = seq_to_joints(methods[mname][fname])
                mt = min(rep_frame, mj.shape[0] - 1)
                draw_skeleton(ax, mj[mt], color=METHOD_COLORS[mname], alpha=0.9,
                             linewidth=2.2, joint_size=10)

            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            if row == 0:
                ax.set_title(col_labels[col_idx + 2], fontsize=9,
                            fontweight='bold' if mname == 'Ours' else 'normal',
                            pad=8)

    # Add arrow between Source and Target columns
    # Add a conceptual "+" and "→" between columns
    for row in range(n_actions):
        # "+" between source and target
        mid_x = (axes[row, 0].get_position().x1 + axes[row, 1].get_position().x0) / 2
        mid_y = (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2
        fig.text(mid_x, mid_y, '+', fontsize=14, ha='center', va='center',
                fontweight='bold', color='#555555')

        # "→" between target and Ours
        mid_x = (axes[row, 1].get_position().x1 + axes[row, 2].get_position().x0) / 2
        fig.text(mid_x, mid_y, '→', fontsize=16, ha='center', va='center',
                fontweight='bold', color='#555555')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2171b5', linewidth=2, linestyle='--',
               alpha=0.3, label='Ground Truth'),
        Line2D([0], [0], color='#d94801', linewidth=2, label='Ours'),
        Line2D([0], [0], color='#e67e22', linewidth=2, label='DMR'),
        Line2D([0], [0], color='#e74c3c', linewidth=2, label='PMR'),
        Line2D([0], [0], color='#7f8c8d', linewidth=2, label='Gaussian'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              fontsize=8, framealpha=0.9, edgecolor='#cccccc',
              bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    path = os.path.join(args.output_dir, 'qualitative_final_single.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # =====================================================================
    # Version 2: Multi-frame filmstrip per method (more detailed)
    # Rows = actions, each action has sub-rows:
    #   Row A: Source motion across 4 frames | Target ID
    #   Row B: GT ghost + Ours across 4 frames
    #   Row C: GT ghost + DMR across 4 frames
    #   Row D: GT ghost + PMR across 4 frames
    #   Row E: GT ghost + Gaussian across 4 frames
    # =====================================================================
    n_method_rows = len(method_names)
    total_rows = n_actions * (1 + n_method_rows)  # 1 source row + 4 method rows per action
    n_cols = n_frames + 1  # frames + target identity column

    fig2, axes2 = plt.subplots(total_rows, n_cols,
                                figsize=(n_cols * 2.0, total_rows * 2.2))

    for action_idx, aid in enumerate(actions):
        fname = selected[aid]
        gt_joints = seq_to_joints(raw_data[fname])
        action_name = NTU_ACTIONS.get(aid, f"A{aid}")
        src_info = parse_file_name(fname, args.dataset)

        tgt_fname, tgt_joints, tgt_person = find_target_identity(
            raw_data, fname, args.dataset)

        all_j = [gt_joints]
        for mname in method_names:
            if fname in methods[mname]:
                all_j.append(seq_to_joints(methods[mname][fname]))
        cx, cy, half = compute_limits(all_j)

        base_row = action_idx * (1 + n_method_rows)

        # --- Source row ---
        for col, fi in enumerate(frame_indices):
            ax = axes2[base_row, col]
            setup_axis(ax)
            t = min(fi, gt_joints.shape[0] - 1)
            draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.9,
                         linewidth=2.0, joint_size=8)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            if action_idx == 0:
                ax.set_title(f't={fi}', fontsize=8, pad=4)
            if col == 0:
                ax.set_ylabel(f'{action_name}\nSource (P{src_info["P"]})',
                             fontsize=7, fontweight='bold', rotation=90, labelpad=8)

        # Target identity in last column of source row
        ax = axes2[base_row, n_frames]
        setup_axis(ax)
        if tgt_joints is not None:
            tgt_t = min(8, tgt_joints.shape[0] - 1)
            tgt_cx, tgt_cy, tgt_half = compute_limits([tgt_joints])
            draw_skeleton(ax, tgt_joints[tgt_t], color='#2ca02c', alpha=0.9,
                         linewidth=2.0, joint_size=8)
            ax.set_xlim(tgt_cx - tgt_half, tgt_cx + tgt_half)
            ax.set_ylim(tgt_cy - tgt_half, tgt_cy + tgt_half)
        if action_idx == 0:
            ax.set_title('Target\nIdentity', fontsize=8, pad=4)

        # --- Method rows (GT ghost + method bold) ---
        for m_idx, mname in enumerate(method_names):
            row = base_row + 1 + m_idx
            for col, fi in enumerate(frame_indices):
                ax = axes2[row, col]
                setup_axis(ax)
                t = min(fi, gt_joints.shape[0] - 1)

                # GT ghost
                draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.20,
                             linewidth=1.2, joint_size=3, linestyle='--')

                # Method result
                if fname in methods[mname]:
                    mj = seq_to_joints(methods[mname][fname])
                    mt = min(fi, mj.shape[0] - 1)
                    draw_skeleton(ax, mj[mt], color=METHOD_COLORS[mname],
                                 alpha=0.9, linewidth=2.0, joint_size=8)

                ax.set_xlim(cx - half, cx + half)
                ax.set_ylim(cy - half, cy + half)
                if col == 0:
                    weight = 'bold' if mname == 'Ours' else 'normal'
                    ax.set_ylabel(mname, fontsize=8, fontweight=weight,
                                 rotation=90, labelpad=8,
                                 color=METHOD_COLORS[mname])

            # Empty target column for method rows
            ax = axes2[row, n_frames]
            setup_axis(ax)

        # Separator line between action groups
        if action_idx < n_actions - 1:
            sep_row = base_row + n_method_rows
            for col in range(n_cols):
                axes2[sep_row, col].axhline(y=0, color='#cccccc',
                                             linewidth=0.5, alpha=0)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.08)

    path2 = os.path.join(args.output_dir, 'qualitative_final_filmstrip.pdf')
    fig2.savefig(path2, dpi=300, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
