#!/usr/bin/env python3
"""
Visualize the input skeletons (x1=P1,A1 and x2=P2,A2) for specific paired
quadruplets. Generates filmstrip-style figures showing the action source and
identity source that go into retargeting.

For each target quadruplet, produces a figure with:
  Row 0: x1 = (P1, A1) — action source (5 frames)
  Row 1: x2 = (P2, A2) — identity source (5 frames)

Must run via SLURM (loads .pt data).
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NTU_BONES = [
    (1, 0), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

NTU_ACTIONS = {
    1: "Drink water", 2: "Eat meal", 3: "Brush teeth", 4: "Brush hair",
    5: "Drop", 6: "Pick up", 7: "Throw", 8: "Sit down", 9: "Stand up",
    10: "Clapping", 11: "Reading", 12: "Writing", 13: "Tear up paper",
    14: "Put on jacket", 15: "Take off jacket", 16: "Put on shoe",
    17: "Take off shoe", 18: "Put on glasses", 19: "Take off glasses",
    20: "Put on hat", 21: "Take off hat", 22: "Cheer up",
    23: "Hand waving", 24: "Kicking", 25: "Reach into pocket",
    26: "Hopping", 27: "Jump up", 28: "Phone call",
    29: "Play with phone", 30: "Type on keyboard",
    31: "Point to something", 32: "Taking selfie", 33: "Check time",
    34: "Rub two hands", 35: "Nod head", 36: "Shake head",
    37: "Wipe face", 38: "Salute", 39: "Put palms together",
    40: "Cross hands in front", 41: "Sneeze", 42: "Staggering",
    43: "Falling down", 44: "Headache", 45: "Chest pain",
    46: "Back pain", 47: "Neck pain", 48: "Nausea",
    49: "Fan self",
}


def tensor_to_joints(t):
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    return np.transpose(t, (1, 2, 0))  # (C,T,V) -> (T,V,C)



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

def draw_skeleton(ax, joints_frame, color='#333333', alpha=1.0, linewidth=2.0,
                  joint_size=12):
    transformed = transform_frame(joints_frame)
    x, y = transformed[:, 0], transformed[:, 1]
    for (i, j) in NTU_BONES:
        ax.plot([x[i], x[j]], [y[i], y[j]], c=color, linewidth=linewidth,
                alpha=alpha, solid_capstyle='round')
    ax.scatter(x, y, c=color, s=joint_size, zorder=5, alpha=alpha,
              edgecolors='white', linewidths=0.3)


def setup_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal', adjustable='datalim')


def compute_limits(joints, padding=1.15):
    """Compute 2D limits after hip-centering and 3/4-view rotation."""
    pts = joints.reshape(-1, 25, 3)
    all_x, all_y = [], []
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


# Target quadruplets from the summary (figures 00, 04, 08)
TARGETS = [
    {'fig': '00', 'p1': 9,  'a1': 34, 'p2': 15, 'label': 'Rub two hands'},
    {'fig': '04', 'p1': 7,  'a1': 34, 'p2': 15, 'label': 'Rub two hands'},
    {'fig': '08', 'p1': 36, 'a1': 34, 'p2': 29, 'label': 'Rub two hands'},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired_data_path', type=str,
                        default='data/ntu/ntu_cv_paired_10k.pt')
    parser.add_argument('--output_dir', type=str, default='paper/fig/paired_qualitative')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading paired data from {args.paired_data_path}...")
    paired_data = torch.load(args.paired_data_path, weights_only=False)
    quads = paired_data['train']
    print(f"  {len(quads)} quadruplets")

    frame_indices = [8, 20, 32, 48, 60]

    for target in TARGETS:
        p1_want, a1_want, p2_want = target['p1'], target['a1'], target['p2']
        fig_id = target['fig']

        # Find matching quadruplet
        match = None
        for quad in quads:
            actors = quad[4]  # [p1, p2]
            actions = quad[5]  # [a1, a2]
            p1, p2 = int(actors[0]), int(actors[1])
            a1, a2 = int(actions[0]), int(actions[1])
            if p1 == p1_want and a1 == a1_want and p2 == p2_want:
                match = quad
                break

        if match is None:
            print(f"  WARNING: No match for fig {fig_id} (P{p1_want}->P{p2_want}, A{a1_want})")
            continue

        x1_ctv = match[0]  # (C, T, V) = action source (P1, A1)
        x2_ctv = match[1]  # (C, T, V) = identity source (P2, A2)
        y2_ctv = match[3]  # (C, T, V) = ground truth (P2, A1)
        actors = match[4]
        actions = match[5]
        p1, p2 = int(actors[0]), int(actors[1])
        a1, a2 = int(actions[0]), int(actions[1])

        x1_joints = tensor_to_joints(x1_ctv)
        x2_joints = tensor_to_joints(x2_ctv)
        y2_joints = tensor_to_joints(y2_ctv)

        a1_name = NTU_ACTIONS.get(a1, f"A{a1}")
        a2_name = NTU_ACTIONS.get(a2, f"A{a2}")

        print(f"\nFigure {fig_id}: x1=(P{p1}, A{a1} {a1_name}), "
              f"x2=(P{p2}, A{a2} {a2_name}), y2=(P{p2}, A{a1} {a1_name})")

        # --- Generate individual input figures ---
        n_frames = len(frame_indices)

        # x1 only
        cx1, cy1, h1 = compute_limits(x1_joints)
        fig_x1, axes_x1 = plt.subplots(1, n_frames, figsize=(n_frames * 2.2, 2.8))
        for col, fi in enumerate(frame_indices):
            ax = axes_x1[col]
            setup_axis(ax)
            t = min(fi, x1_joints.shape[0] - 1)
            draw_skeleton(ax, x1_joints[t], color='#2171b5', alpha=0.9,
                         linewidth=2.2, joint_size=10)
            ax.set_xlim(cx1 - h1, cx1 + h1)
            ax.set_ylim(cy1 - h1, cy1 + h1)
            ax.set_title(f't={fi}', fontsize=9, pad=4)
        fig_x1.suptitle(f'x1: P{p1} doing {a1_name} (Action Source)',
                        fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout(pad=0.3)
        path = os.path.join(args.output_dir, f'input_x1_fig{fig_id}.pdf')
        fig_x1.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close(fig_x1)

        # x2 only
        cx2, cy2, h2 = compute_limits(x2_joints)
        fig_x2, axes_x2 = plt.subplots(1, n_frames, figsize=(n_frames * 2.2, 2.8))
        for col, fi in enumerate(frame_indices):
            ax = axes_x2[col]
            setup_axis(ax)
            t = min(fi, x2_joints.shape[0] - 1)
            draw_skeleton(ax, x2_joints[t], color='#2ca02c', alpha=0.9,
                         linewidth=2.2, joint_size=10)
            ax.set_xlim(cx2 - h2, cx2 + h2)
            ax.set_ylim(cy2 - h2, cy2 + h2)
            ax.set_title(f't={fi}', fontsize=9, pad=4)
        fig_x2.suptitle(f'x2: P{p2} doing {a2_name} (Identity Source)',
                        fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout(pad=0.3)
        path = os.path.join(args.output_dir, f'input_x2_fig{fig_id}.pdf')
        fig_x2.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close(fig_x2)

        # Combined: 2 rows (x1, x2) x 5 frames
        all_joints = [x1_joints, x2_joints]
        all_x, all_y = [], []
        for j_seq in all_joints:
            for t in range(j_seq.shape[0]):
                transformed = transform_frame(j_seq[t])
                all_x.append(transformed[:, 0])
                all_y.append(transformed[:, 1])
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        cx = (all_x.max() + all_x.min()) / 2
        cy = (all_y.max() + all_y.min()) / 2
        half = max(np.abs(all_x - cx).max(), np.abs(all_y - cy).max()) * 1.15

        fig_both, axes_both = plt.subplots(2, n_frames,
                                            figsize=(n_frames * 2.2, 2 * 2.8))
        row_data = [
            (x1_joints, '#2171b5', f'x1: P{p1}, A{a1} ({a1_name})'),
            (x2_joints, '#2ca02c', f'x2: P{p2}, A{a2} ({a2_name})'),
        ]
        for row, (joints, color, ylabel) in enumerate(row_data):
            cxr, cyr, hr = compute_limits(joints)
            for col, fi in enumerate(frame_indices):
                ax = axes_both[row, col]
                setup_axis(ax)
                t = min(fi, joints.shape[0] - 1)
                draw_skeleton(ax, joints[t], color=color, alpha=0.9,
                             linewidth=2.2, joint_size=10)
                ax.set_xlim(cxr - hr, cxr + hr)
                ax.set_ylim(cyr - hr, cyr + hr)
                if row == 0:
                    ax.set_title(f't={fi}', fontsize=9, pad=4)
            axes_both[row, 0].set_ylabel(ylabel, fontsize=8, fontweight='bold',
                                          rotation=90, labelpad=8, color=color)

        fig_both.suptitle(f'Inputs for Figure {fig_id}: {a1_name} (P{p1} -> P{p2})',
                          fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout(pad=0.3, h_pad=0.3)
        path = os.path.join(args.output_dir, f'inputs_combined_fig{fig_id}.pdf')
        fig_both.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close(fig_both)

    print("\nDone!")


if __name__ == '__main__':
    main()
