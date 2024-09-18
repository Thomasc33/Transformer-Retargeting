#!/usr/bin/env python3
"""
Generate GIF visualizations for retargeted samples across model variants.

Two modes:
  1. --select_samples: Pick diverse good samples from raw data, save manifest.json
  2. --manifest: Use a pre-selected manifest to generate GIFs for a specific variant

Outputs GIFs showing GT ghost (blue dashed) + retargeted skeleton (orange solid).
"""

import argparse
import os
import sys
import pickle
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

NTU_ACTIONS_1INDEXED = {
    1: "Drink water", 2: "Eat meal", 3: "Brush teeth", 4: "Brush hair",
    5: "Drop", 6: "Pickup", 7: "Throw", 8: "Sit down", 9: "Stand up",
    10: "Clapping", 11: "Reading", 12: "Writing", 13: "Tear up paper",
    14: "Wear jacket", 15: "Take off jacket", 16: "Wear a shoe",
    17: "Take off a shoe", 18: "Wear on glasses", 19: "Take off glasses",
    20: "Put on a hat/cap", 21: "Take off a hat/cap", 22: "Cheer up",
    23: "Hand waving", 24: "Kicking something", 25: "Reach into pocket",
    26: "Hopping", 27: "Jump up", 28: "Make a phone call",
    29: "Playing with phone", 30: "Typing on keyboard",
    31: "Point to something", 32: "Taking a selfie", 33: "Check time",
    34: "Rub two hands", 35: "Nod head/bow", 36: "Shake head",
    37: "Wipe face", 38: "Salute", 39: "Put palms together",
    40: "Cross hands in front", 41: "Sneeze/cough", 42: "Staggering",
    43: "Falling", 44: "Touch head", 45: "Touch chest",
    46: "Touch back", 47: "Touch neck", 48: "Nausea/vomiting",
    49: "Use a fan",
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

def seq_to_joints(seq, center=True):
    T = seq.shape[0]
    joints = seq.reshape(T, 25, 3)
    if center:
        # Center each frame on SpineBase (joint 0) so skeletons align
        spine_base = joints[:, 0:1, :]  # (T, 1, 3)
        joints = joints - spine_base
    return joints


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


def motion_magnitude(joints):
    diffs = np.diff(joints, axis=0)
    per_frame = np.sqrt((diffs ** 2).sum(axis=-1)).mean(axis=-1)
    return per_frame.mean()


def draw_skeleton(ax, joints_frame, color='#333333', alpha=1.0, linewidth=2.0,
                  joint_size=12, linestyle='-'):
    transformed = transform_frame(joints_frame)
    x, y = transformed[:, 0], transformed[:, 1]
    for (i, j) in NTU_BONES:
        ax.plot([x[i], x[j]], [y[i], y[j]], c=color, linewidth=linewidth,
                alpha=alpha, solid_capstyle='round', linestyle=linestyle)
    ax.scatter(x, y, c=color, s=joint_size, zorder=5, alpha=alpha,
               edgecolors='white', linewidths=0.3)


def setup_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal', adjustable='datalim')


def generate_gif(gt_joints, ret_joints, fname, info, output_dir, variant_name="", fps=10):
    all_x, all_y = [], []
    for t in range(ret_joints.shape[0]):
        transformed = transform_frame(ret_joints[t])
        all_x.append(transformed[:, 0])
        all_y.append(transformed[:, 1])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    cx = (all_x.max() + all_x.min()) / 2
    cy = (all_y.max() + all_y.min()) / 2
    half = max(np.abs(all_x - cx).max(), np.abs(all_y - cy).max()) * 1.15

    action_name = NTU_ACTIONS_1INDEXED.get(info['A'], f"A{info['A']}")
    T = ret_joints.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    title_prefix = f"{variant_name} | " if variant_name else ""

    def animate(t):
        ax.clear()
        setup_axis(ax)
        draw_skeleton(ax, ret_joints[t], color='#d94801', alpha=0.9,
                      linewidth=2.0, joint_size=10)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(f'{title_prefix}{action_name} (P{info["P"]})  t={t}',
                      fontsize=9, fontweight='bold')
        return []

    frames = list(range(4, T))
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000 // fps, blit=True)

    safe_fname = fname.replace('/', '_').replace('.', '_')
    path = os.path.join(output_dir, f'anim_{safe_fname}.gif')
    anim.save(path, writer='pillow', fps=fps)
    plt.close()
    return path


def generate_still(gt_joints, ret_joints, fname, info, output_dir, variant_name="",
                   frame_indices=[8, 20, 32, 44, 56]):
    n_frames = len(frame_indices)
    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2.5, 3.0))
    if n_frames == 1:
        axes = [axes]

    all_x, all_y = [], []
    for seq in [gt_joints, ret_joints]:
        for t in range(seq.shape[0]):
            transformed = transform_frame(seq[t])
            all_x.append(transformed[:, 0])
            all_y.append(transformed[:, 1])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    cx = (all_x.max() + all_x.min()) / 2
    cy = (all_y.max() + all_y.min()) / 2
    half = max(np.abs(all_x - cx).max(), np.abs(all_y - cy).max()) * 1.15

    action_name = NTU_ACTIONS_1INDEXED.get(info['A'], f"A{info['A']}")
    title_prefix = f"{variant_name} | " if variant_name else ""

    for i, fi in enumerate(frame_indices):
        ax = axes[i]
        setup_axis(ax)
        t = min(fi, gt_joints.shape[0] - 1, ret_joints.shape[0] - 1)
        draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.25,
                      linewidth=1.2, joint_size=3, linestyle='--')
        draw_skeleton(ax, ret_joints[t], color='#d94801', alpha=0.9,
                      linewidth=2.0, joint_size=8)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(f't={fi}', fontsize=8, pad=4)

    fig.suptitle(f'{title_prefix}{action_name} (P{info["P"]})', fontsize=10,
                  fontweight='bold', y=1.02)
    plt.tight_layout()
    safe_fname = fname.replace('/', '_').replace('.', '_')
    path = os.path.join(output_dir, f'still_{safe_fname}.pdf')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def select_samples(raw_data, dataset, max_samples=30, min_motion=0.01):
    """Select diverse good samples from raw data based on motion and pose quality."""
    candidates = []
    for fname, seq in raw_data.items():
        joints = seq_to_joints(seq)
        if not is_standing(joints):
            continue
        motion = motion_magnitude(joints)
        if motion < min_motion:
            continue
        info = parse_file_name(fname, dataset)
        candidates.append({
            'fname': fname,
            'action': info['A'],
            'action_name': NTU_ACTIONS_1INDEXED.get(info['A'], f"A{info['A']}"),
            'person': info['P'],
            'motion': float(motion),
            'info': info,
        })

    # Sort by highest motion (most dynamic samples)
    candidates.sort(key=lambda x: -x['motion'])

    # Diverse selection: max 2 per action
    action_counts = {}
    selected = []
    for c in candidates:
        a = c['action']
        if action_counts.get(a, 0) >= 2:
            continue
        action_counts[a] = action_counts.get(a, 0) + 1
        selected.append(c)
        if len(selected) >= max_samples:
            break

    selected.sort(key=lambda x: (x['action'], -x['motion']))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ntu')
    parser.add_argument('--retargeted_pkl', type=str, default=None,
                        help='Path to retargeted pickle (for generating viz)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for GIFs/stills')
    parser.add_argument('--variant_name', type=str, default='',
                        help='Variant name to show in titles')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to manifest.json with pre-selected sample filenames')
    parser.add_argument('--select_samples', action='store_true',
                        help='Select samples from raw data and save manifest (no retargeted pkl needed)')
    parser.add_argument('--max_samples', type=int, default=30)
    parser.add_argument('--min_motion', type=float, default=0.01)
    parser.add_argument('--still_only', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw data
    print("Loading raw data...")
    raw_data = load_data(args.dataset, T=64)
    print(f"  Raw data: {len(raw_data)} samples")

    if args.select_samples:
        # Mode 1: Select samples and save manifest
        print("\nSelecting diverse samples from raw data...")
        selected = select_samples(raw_data, args.dataset, args.max_samples, args.min_motion)
        manifest = [{'fname': s['fname'], 'action': s['action'],
                      'action_name': s['action_name'], 'person': s['person'],
                      'motion': s['motion']} for s in selected]
        manifest_path = os.path.join(args.output_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  Selected {len(manifest)} samples")
        print(f"  Manifest saved: {manifest_path}")
        for i, s in enumerate(manifest):
            print(f"  [{i+1:3d}] A{s['action']:02d} {s['action_name']:<25s} P{s['person']:02d} motion={s['motion']:.4f} {s['fname']}")
        return

    # Mode 2: Generate visualizations
    if args.manifest is None or args.retargeted_pkl is None:
        print("ERROR: Need both --manifest and --retargeted_pkl for visualization mode")
        sys.exit(1)

    print(f"\nLoading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f"  {len(manifest)} samples to visualize")

    print(f"Loading retargeted data: {args.retargeted_pkl}")
    with open(args.retargeted_pkl, 'rb') as f:
        ret_data = pickle.load(f)
    print(f"  Retargeted: {len(ret_data)} samples")

    # Generate
    gif_count = 0
    still_count = 0
    missing = 0
    for i, entry in enumerate(manifest):
        fname = entry['fname']
        if fname not in ret_data:
            print(f"  [{i+1}] SKIP (not in retargeted data): {fname}")
            missing += 1
            continue
        if fname not in raw_data:
            print(f"  [{i+1}] SKIP (not in raw data): {fname}")
            missing += 1
            continue

        gt_joints = seq_to_joints(raw_data[fname])
        ret_joints = seq_to_joints(ret_data[fname])
        info = parse_file_name(fname, args.dataset)

        # Still
        spath = generate_still(gt_joints, ret_joints, fname, info, args.output_dir,
                                variant_name=args.variant_name)
        still_count += 1

        # GIF
        if not args.still_only:
            gpath = generate_gif(gt_joints, ret_joints, fname, info, args.output_dir,
                                  variant_name=args.variant_name)
            gif_count += 1

        print(f"  [{i+1}/{len(manifest)}] {entry.get('action_name', '')} done")

    print(f"\nDone: {still_count} stills, {gif_count} GIFs, {missing} skipped")


if __name__ == '__main__':
    main()
