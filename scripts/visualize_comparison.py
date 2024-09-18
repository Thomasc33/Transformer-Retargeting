#!/usr/bin/env python3
"""
Comparison Visualization: Ours vs DMR vs PMR vs Gaussian Noise

Generates publication-quality side-by-side skeleton figures comparing
retargeted outputs from all methods on the same source samples.

Layout (per row = one action):
  Source | Ours | DMR | PMR | Gaussian Noise

Shows 5 key frames per cell as ghost trail overlays.
Output: paper/fig/qualitative_comparison.pdf

Requires: retargeted .pkl files for all methods + raw NTU data.
Must run via SLURM (loads ~4GB of pickle data).
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

from src.data.datasets import datasets as DATASETS_CONFIG, parse_file_name, load_data

# NTU bone connections (0-indexed)
NTU_BONES = [
    (1, 0), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

BODY_PART_COLORS = {
    'torso': '#555555', 'l_arm': '#1f77b4', 'r_arm': '#ff7f0e',
    'l_leg': '#2ca02c', 'r_leg': '#d62728', 'head': '#9467bd',
}

def bone_to_part(i, j):
    joints = {i, j}
    if joints & {4, 5, 6, 7, 21, 22}: return 'l_arm'
    if joints & {8, 9, 10, 11, 23, 24}: return 'r_arm'
    if joints & {12, 13, 14, 15}: return 'l_leg'
    if joints & {16, 17, 18, 19}: return 'r_leg'
    if joints & {2, 3}: return 'head'
    return 'torso'



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


def draw_skeleton_2d(ax, joints_frame, color=None, alpha=1.0, linewidth=2.0,
                     joint_size=15, use_part_colors=False, label=None):
    """Draw skeleton on 2D axis (front view: X horizontal, Y vertical)."""
    transformed = transform_frame(joints_frame)
    x, y = transformed[:, 0], transformed[:, 1]
    for (i, j) in NTU_BONES:
        c = BODY_PART_COLORS[bone_to_part(i, j)] if use_part_colors else (color or '#333333')
        ax.plot([x[i], x[j]], [y[i], y[j]], c=c, linewidth=linewidth,
                alpha=alpha, solid_capstyle='round')
    kw = dict(c=color or '#333333', s=joint_size, zorder=5, alpha=alpha,
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


def compute_limits(joints_list):
    """Compute 2D limits after hip-centering and 3/4-view rotation."""
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
    half = max(np.abs(all_x - cx).max(), np.abs(all_y - cy).max()) * 1.1
    return cx, cy, half


def find_good_samples(raw_data, target_actions, dataset='ntu'):
    """Find filename for each target action with visually interesting motion."""
    NTU_ACTIONS = {
        1: "Drink water", 7: "Throw", 10: "Clapping", 23: "Hand waving",
        24: "Kicking", 26: "Hopping", 27: "Jump up", 43: "Falling down",
    }
    samples = {}
    for fname, seq in raw_data.items():
        info = parse_file_name(fname, dataset)
        aid = info['A']
        if aid in target_actions and aid not in samples:
            joints = seq_to_joints(seq)
            # Measure motion magnitude (total joint displacement)
            vel = np.diff(joints, axis=0)
            motion = np.sum(np.linalg.norm(vel, axis=-1))
            if aid not in samples or motion > samples[aid][1]:
                samples[aid] = (fname, motion)
    return {aid: fname for aid, (fname, _) in samples.items()}


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
    parser.add_argument('--actions', type=int, nargs='+',
                        default=[23, 27, 10, 24],
                        help='Action IDs to visualize')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Action labels
    NTU_ACTIONS = {
        1: "Drink water", 2: "Eat meal", 3: "Brush teeth", 4: "Brush hair",
        5: "Drop", 6: "Pick up", 7: "Throw", 8: "Sit down", 9: "Stand up",
        10: "Clapping", 11: "Reading", 12: "Writing", 13: "Tear up paper",
        14: "Wear jacket", 15: "Take off jacket", 16: "Wear shoe",
        17: "Take off shoe", 18: "Wear glasses", 19: "Take off glasses",
        20: "Put on hat", 21: "Take off hat", 22: "Cheer up", 23: "Hand waving",
        24: "Kicking", 25: "Reach into pocket", 26: "Hopping", 27: "Jump up",
        28: "Phone call", 29: "Play with phone", 30: "Type on keyboard",
        31: "Point to something", 32: "Take selfie", 33: "Check time",
        34: "Rub two hands", 35: "Nod head/bow", 36: "Shake head",
        37: "Wipe face", 38: "Salute", 39: "Put palms together",
        40: "Cross hands in front", 41: "Sneeze/cough", 42: "Staggering",
        43: "Falling down", 44: "Headache", 45: "Chest pain",
        46: "Back pain", 47: "Neck pain", 48: "Nausea/vomiting", 49: "Fan self",
    }

    # Load raw data
    print("Loading raw data...")
    raw_data = load_data(args.dataset, T=64)

    # Load retargeted datasets
    print("Loading retargeted datasets...")
    methods = {}
    for name, path in [('Ours', args.ours_pkl), ('DMR', args.dmr_pkl),
                        ('PMR', args.pmr_pkl), ('Gaussian', args.noise_pkl)]:
        print(f"  Loading {name} from {path}...")
        with open(path, 'rb') as f:
            methods[name] = pickle.load(f)
        print(f"    {len(methods[name])} samples")

    # Find good samples for each action
    print("Selecting samples...")
    # Use deterministic selection: find samples present in ALL methods
    common_fnames = set(raw_data.keys())
    for name, data in methods.items():
        common_fnames &= set(data.keys())
    print(f"  {len(common_fnames)} samples common to all methods")

    # Group by action
    action_fnames = {}
    for fname in common_fnames:
        info = parse_file_name(fname, args.dataset)
        aid = info['A']
        if aid in args.actions:
            if aid not in action_fnames:
                action_fnames[aid] = []
            action_fnames[aid].append(fname)

    # Pick the sample with most motion per action
    selected = {}
    for aid in args.actions:
        if aid not in action_fnames:
            print(f"  WARNING: No common samples for action {aid}")
            continue
        best_fname = None
        best_motion = -1
        for fname in action_fnames[aid]:
            joints = seq_to_joints(raw_data[fname])
            vel = np.diff(joints, axis=0)
            motion = np.sum(np.linalg.norm(vel, axis=-1))
            if motion > best_motion:
                best_motion = motion
                best_fname = fname
        selected[aid] = best_fname
        action_name = NTU_ACTIONS.get(aid, f"Action {aid}")
        print(f"  Action {aid} ({action_name}): {best_fname} (motion={best_motion:.1f})")

    if not selected:
        print("ERROR: No samples found for requested actions")
        return

    # ======================================================================
    # Figure: Comparison grid
    # Rows = actions, Columns = Source | Ours | DMR | PMR | Gaussian
    # Each cell shows 5 ghost frames overlaid
    # ======================================================================
    n_actions = len(selected)
    method_names = ['Source', 'Ours', 'DMR', 'PMR', 'Gaussian']
    method_colors = {
        'Source': '#2171b5',    # blue
        'Ours': '#d94801',      # orange
        'DMR': '#e67e22',       # amber
        'PMR': '#e74c3c',       # red
        'Gaussian': '#7f8c8d',  # gray
    }

    frame_indices = [4, 16, 32, 48, 60]  # 5 key frames spread across sequence
    n_frames = len(frame_indices)

    fig, axes = plt.subplots(n_actions, len(method_names),
                             figsize=(12, 2.8 * n_actions))
    if n_actions == 1:
        axes = axes.reshape(1, -1)

    for row_idx, aid in enumerate(sorted(selected.keys())):
        fname = selected[aid]
        action_name = NTU_ACTIONS.get(aid, f"Action {aid}")

        # Get raw source joints
        src_joints = seq_to_joints(raw_data[fname])

        # Get retargeted joints for each method
        method_joints = {'Source': src_joints}
        for mname in ['Ours', 'DMR', 'PMR', 'Gaussian']:
            if fname in methods[mname]:
                method_joints[mname] = seq_to_joints(methods[mname][fname])
            else:
                method_joints[mname] = None

        # Compute global limits across all methods for this action
        all_joints = [j for j in method_joints.values() if j is not None]
        cx, cy, half = compute_limits(all_joints)

        for col_idx, mname in enumerate(method_names):
            ax = axes[row_idx, col_idx]
            setup_axis(ax)

            joints = method_joints.get(mname)
            if joints is None:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
                continue

            color = method_colors[mname]
            T = joints.shape[0]

            # Draw ghost trail: earlier frames lighter, last frame darkest
            for fi, frame_idx in enumerate(frame_indices):
                if frame_idx >= T:
                    frame_idx = T - 1
                alpha = 0.15 + 0.17 * fi  # 0.15 to 0.83
                lw = 1.0 + 0.3 * fi       # thicker for later frames
                draw_skeleton_2d(ax, joints[frame_idx], color=color,
                                alpha=alpha, linewidth=lw, joint_size=8)

            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

            # Column headers
            if row_idx == 0:
                ax.set_title(mname, fontsize=10, fontweight='bold', pad=6)

            # Row labels (action name on left)
            if col_idx == 0:
                ax.set_ylabel(action_name, fontsize=8, rotation=90, labelpad=8)

    plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3)

    out_pdf = os.path.join(args.output_dir, 'qualitative_comparison.pdf')
    out_png = os.path.join(args.output_dir, 'qualitative_comparison.png')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close()

    # ======================================================================
    # Figure 2: Single-action detailed comparison (overlay style)
    # For one action, show source ghost + each method's retargeted ghost
    # side by side with the source overlaid for reference
    # ======================================================================
    # Pick the most dynamic action
    best_aid = sorted(selected.keys())[0]
    fname = selected[best_aid]
    action_name = NTU_ACTIONS.get(best_aid, f"Action {best_aid}")
    src_joints = seq_to_joints(raw_data[fname])

    fig2, axes2 = plt.subplots(1, 4, figsize=(12, 3.5))
    retarg_methods = ['Ours', 'DMR', 'PMR', 'Gaussian']

    for col_idx, mname in enumerate(retarg_methods):
        ax = axes2[col_idx]
        setup_axis(ax)

        retarg_joints = seq_to_joints(methods[mname][fname]) if fname in methods[mname] else None

        # Compute limits for this pair
        pair_joints = [src_joints]
        if retarg_joints is not None:
            pair_joints.append(retarg_joints)
        cx, cy, half = compute_limits(pair_joints)

        # Draw source as light ghost
        for fi, frame_idx in enumerate(frame_indices):
            fi_idx = min(frame_idx, src_joints.shape[0] - 1)
            draw_skeleton_2d(ax, src_joints[fi_idx], color='#2171b5',
                            alpha=0.15, linewidth=0.8, joint_size=4)

        # Draw retargeted on top
        if retarg_joints is not None:
            color = method_colors[mname]
            for fi, frame_idx in enumerate(frame_indices):
                fi_idx = min(frame_idx, retarg_joints.shape[0] - 1)
                alpha = 0.2 + 0.16 * fi
                lw = 1.2 + 0.3 * fi
                draw_skeleton_2d(ax, retarg_joints[fi_idx], color=color,
                                alpha=alpha, linewidth=lw, joint_size=8)

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(mname, fontsize=10, fontweight='bold', pad=4)

    fig2.suptitle(f'{action_name} — Source (blue ghost) + Retargeted (colored)',
                  fontsize=9, y=0.02, color='gray')
    plt.tight_layout(pad=0.5)

    out_pdf2 = os.path.join(args.output_dir, 'qualitative_overlay.pdf')
    out_png2 = os.path.join(args.output_dir, 'qualitative_overlay.png')
    plt.savefig(out_pdf2, dpi=300, bbox_inches='tight')
    plt.savefig(out_png2, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_pdf2}")
    print(f"Saved: {out_png2}")
    plt.close()


if __name__ == '__main__':
    main()
