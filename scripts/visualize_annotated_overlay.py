#!/usr/bin/env python3
"""
Annotated Overlay Visualization

Shows ground truth skeleton (ghost) overlaid with retargeted skeleton (bold),
annotated with:
  - SGN predicted action label (and whether it's correct)
  - Per-sample MSE between ground truth and retargeted

Ground truth = the source person doing the action (input to the model).
The retargeted output should preserve the action while changing identity.

Output: paper/fig/qualitative_annotated_*.{pdf,png}
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import parse_file_name, load_data
from src.model.sgn import SGN

# NTU bone connections (0-indexed)
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

# Only standing actions
STANDING_ACTIONS = [23, 10, 24, 27, 38, 31, 1, 22, 26, 34, 49, 40]

# Build remapped label list (drop two-person actions >49, keep 1-indexed as 0-indexed)
# SGN trained on retargeted NTU60 uses 49 single-person actions
# Labels are remapped: action IDs 1-49 → labels 0-48
ACTION_ID_TO_LABEL = {aid: aid - 1 for aid in range(1, 50)}
LABEL_TO_ACTION_ID = {v: k for k, v in ACTION_ID_TO_LABEL.items()}



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
    return seq.reshape(seq.shape[0], 25, 3)


def joints_to_sgn_input(joints):
    """Convert (T, V, 3) to SGN input format (1, T, V*3).
    SGN expects (batch, step, dim) where dim = num_joints * 3."""
    # (T, V, 3) -> (T, V*3) -> (1, T, V*3)
    T, V, C = joints.shape
    tensor = torch.from_numpy(joints).float().reshape(T, V * C).unsqueeze(0)
    return tensor


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


def compute_mse(gt_joints, retarg_joints):
    """Compute MSE between two (T, V, 3) arrays."""
    return np.mean((gt_joints - retarg_joints) ** 2)


def load_sgn_model(checkpoint_path, num_classes=49, device='cpu'):
    """Load trained SGN model."""
    model = SGN(num_classes=num_classes, dataset='ntu', seg=64)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    return model


def predict_action(model, joints, device='cpu'):
    """Run SGN on a single sample and return (predicted_label, confidence)."""
    inp = joints_to_sgn_input(joints).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
        pred_label = logits.argmax(dim=1).item()
        confidence = probs[0, pred_label].item()
    return pred_label, confidence


def find_standing_samples(raw_data, retarg_data, dataset, target_actions):
    """Find samples common to both, standing, with good motion."""
    common_fnames = set(raw_data.keys()) & set(retarg_data.keys())
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
        selected[aid] = candidates[0][0]  # highest motion
    return selected


# =========================================================================
# Variation K: Annotated filmstrip with MSE and SGN prediction
# Rows = actions, Cols = frames
# Ground truth (ghost) + retargeted (bold), annotated
# =========================================================================
def make_variation_k(raw_data, retarg_data, sgn_model, selected, output_dir, device):
    print("\n=== Variation K: Annotated filmstrip ===")
    frame_indices = [8, 16, 28, 40, 56]
    actions = sorted([a for a in selected])[:4]
    n_actions = len(actions)
    n_frames = len(frame_indices)

    fig, axes = plt.subplots(n_actions, n_frames, figsize=(n_frames * 2.4, n_actions * 3.2))
    if n_actions == 1:
        axes = axes.reshape(1, -1)

    for row, aid in enumerate(actions):
        fname = selected[aid]
        gt_joints = seq_to_joints(raw_data[fname])
        retarg_joints = seq_to_joints(retarg_data[fname])
        cx, cy, half = compute_limits([gt_joints, retarg_joints])

        # SGN prediction on retargeted
        pred_label, conf = predict_action(sgn_model, retarg_joints, device)
        pred_aid = LABEL_TO_ACTION_ID.get(pred_label, pred_label + 1)
        pred_name = NTU_ACTIONS.get(pred_aid, f"A{pred_aid}")
        correct = (pred_aid == aid)

        # MSE
        mse = compute_mse(gt_joints, retarg_joints)

        for col, fi in enumerate(frame_indices):
            ax = axes[row, col]
            setup_axis(ax)
            t = min(fi, gt_joints.shape[0] - 1)

            # Ground truth underneath (ghost)
            draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.30,
                         linewidth=1.5, joint_size=5, linestyle='--')
            # Retargeted on top (bold)
            draw_skeleton(ax, retarg_joints[t], color='#d94801', alpha=0.9,
                         linewidth=2.2, joint_size=10)

            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

            if row == 0:
                ax.set_title(f't={fi}', fontsize=9, pad=4)

            if col == 0:
                true_name = NTU_ACTIONS.get(aid, f"A{aid}")
                ax.set_ylabel(true_name, fontsize=8, rotation=90, labelpad=8)

        # Annotate the last column with SGN prediction + MSE
        ax_last = axes[row, -1]
        color = '#2ca02c' if correct else '#d62728'
        checkmark = '\u2713' if correct else '\u2717'
        ax_last.text(1.05, 0.7, f'SGN: {pred_name}',
                    transform=ax_last.transAxes, fontsize=6.5,
                    color=color, ha='left', va='center')
        ax_last.text(1.05, 0.5, f'{checkmark} {"Correct" if correct else "Wrong"}',
                    transform=ax_last.transAxes, fontsize=6.5,
                    color=color, ha='left', va='center', fontweight='bold')
        ax_last.text(1.05, 0.3, f'MSE: {mse:.4f}',
                    transform=ax_last.transAxes, fontsize=6.5,
                    color='#555555', ha='left', va='center')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2171b5', linewidth=1.5, linestyle='--',
               alpha=0.5, label='Ground Truth'),
        Line2D([0], [0], color='#d94801', linewidth=2.2, label='Retargeted'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
              fontsize=8, framealpha=0.8, bbox_to_anchor=(0.45, 1.0))
    plt.tight_layout(pad=0.4, h_pad=0.4, w_pad=0.3, rect=[0, 0, 0.88, 0.96])

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_annotated_filmstrip.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# Variation L: Method comparison with annotations
# Rows = actions, Cols = methods (Ours, DMR, PMR, Gaussian)
# Each cell: GT ghost + method bold, annotated with MSE
# Bottom text: SGN prediction per method
# =========================================================================
def make_variation_l(raw_data, methods, sgn_model, selected, output_dir, device):
    print("\n=== Variation L: Annotated method comparison ===")
    method_names = ['Ours', 'DMR', 'PMR', 'Gaussian']
    method_colors = {
        'Ours': '#d94801', 'DMR': '#e67e22', 'PMR': '#e74c3c', 'Gaussian': '#7f8c8d',
    }
    actions = sorted([a for a in selected])[:3]
    target_frame = 32

    fig, axes = plt.subplots(len(actions), len(method_names),
                             figsize=(len(method_names) * 3.0, len(actions) * 3.5))
    if len(actions) == 1:
        axes = axes.reshape(1, -1)

    for row, aid in enumerate(actions):
        fname = selected[aid]
        gt_joints = seq_to_joints(raw_data[fname])
        true_name = NTU_ACTIONS.get(aid, f"A{aid}")

        all_joints = [gt_joints]
        for mname in method_names:
            if fname in methods[mname]:
                all_joints.append(seq_to_joints(methods[mname][fname]))
        cx, cy, half = compute_limits(all_joints)

        for col, mname in enumerate(method_names):
            ax = axes[row, col]
            setup_axis(ax)
            t = min(target_frame, gt_joints.shape[0] - 1)

            # Ground truth ghost
            draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.25,
                         linewidth=1.5, joint_size=4, linestyle='--')

            if fname in methods[mname]:
                mj = seq_to_joints(methods[mname][fname])
                mt = min(target_frame, mj.shape[0] - 1)
                draw_skeleton(ax, mj[mt], color=method_colors[mname], alpha=0.9,
                             linewidth=2.2, joint_size=10)

                # MSE
                mse = compute_mse(gt_joints, mj)
                # SGN prediction
                pred_label, conf = predict_action(sgn_model, mj, device)
                pred_aid = LABEL_TO_ACTION_ID.get(pred_label, pred_label + 1)
                pred_name = NTU_ACTIONS.get(pred_aid, f"A{pred_aid}")
                correct = (pred_aid == aid)

                # Annotate
                check_color = '#2ca02c' if correct else '#d62728'
                checkmark = '\u2713' if correct else '\u2717'
                ax.text(0.5, -0.02, f'MSE: {mse:.4f}',
                       transform=ax.transAxes, fontsize=6.5, ha='center',
                       color='#555555')
                ax.text(0.5, -0.08, f'SGN: {pred_name} {checkmark}',
                       transform=ax.transAxes, fontsize=6, ha='center',
                       color=check_color)

            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

            if row == 0:
                ax.set_title(mname, fontsize=10, fontweight='bold', pad=6)
            if col == 0:
                ax.set_ylabel(true_name, fontsize=8, rotation=90, labelpad=8)

    plt.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.3)

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_annotated_comparison.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# Variation M: Dense annotated filmstrip (one action, all methods, 6 frames)
# 5 rows (GT only, Ours, DMR, PMR, Gaussian) × 6 frames
# GT ghost in all overlay rows, MSE per row
# =========================================================================
def make_variation_m(raw_data, methods, sgn_model, selected, output_dir, device):
    print("\n=== Variation M: Dense annotated multi-method filmstrip ===")
    frame_indices = [8, 18, 28, 38, 48, 58]
    method_names = ['Ours', 'DMR', 'PMR', 'Gaussian']
    method_colors = {
        'Ours': '#d94801', 'DMR': '#e67e22', 'PMR': '#e74c3c', 'Gaussian': '#7f8c8d',
    }

    # Pick best action
    for aid in [23, 10, 27, 24, 38]:
        if aid in selected:
            break

    fname = selected[aid]
    true_name = NTU_ACTIONS.get(aid, f"A{aid}")
    gt_joints = seq_to_joints(raw_data[fname])

    all_rows = ['Ground Truth'] + method_names
    n_rows = len(all_rows)
    n_frames = len(frame_indices)

    all_j = [gt_joints]
    for mname in method_names:
        if fname in methods[mname]:
            all_j.append(seq_to_joints(methods[mname][fname]))
    cx, cy, half = compute_limits(all_j)

    fig, axes = plt.subplots(n_rows, n_frames, figsize=(n_frames * 2.0, n_rows * 2.8))

    for row_idx, row_name in enumerate(all_rows):
        if row_name == 'Ground Truth':
            # Just show GT
            for col, fi in enumerate(frame_indices):
                ax = axes[row_idx, col]
                setup_axis(ax)
                t = min(fi, gt_joints.shape[0] - 1)
                draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.85,
                             linewidth=2.0, joint_size=10)
                ax.set_xlim(cx - half, cx + half)
                ax.set_ylim(cy - half, cy + half)
                if col == 0:
                    ax.set_ylabel('Ground Truth', fontsize=8, fontweight='bold',
                                 rotation=90, labelpad=8, color='#2171b5')
                ax.set_title(f't={fi}', fontsize=8, pad=3) if row_idx == 0 else None
        else:
            mname = row_name
            if fname not in methods[mname]:
                continue
            mj = seq_to_joints(methods[mname][fname])
            mse = compute_mse(gt_joints, mj)
            pred_label, conf = predict_action(sgn_model, mj, device)
            pred_aid = LABEL_TO_ACTION_ID.get(pred_label, pred_label + 1)
            pred_name = NTU_ACTIONS.get(pred_aid, f"A{pred_aid}")
            correct = (pred_aid == aid)

            for col, fi in enumerate(frame_indices):
                ax = axes[row_idx, col]
                setup_axis(ax)
                t = min(fi, gt_joints.shape[0] - 1)

                # GT ghost
                draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.20,
                             linewidth=1.2, joint_size=3, linestyle='--')
                # Method bold
                mt = min(fi, mj.shape[0] - 1)
                draw_skeleton(ax, mj[mt], color=method_colors[mname], alpha=0.9,
                             linewidth=2.0, joint_size=10)

                ax.set_xlim(cx - half, cx + half)
                ax.set_ylim(cy - half, cy + half)

                if col == 0:
                    ax.set_ylabel(mname, fontsize=8, fontweight='bold',
                                 rotation=90, labelpad=8,
                                 color=method_colors[mname])

            # Annotate right side
            ax_last = axes[row_idx, -1]
            check_color = '#2ca02c' if correct else '#d62728'
            checkmark = '\u2713' if correct else '\u2717'
            ax_last.text(1.05, 0.65, f'MSE: {mse:.4f}',
                        transform=ax_last.transAxes, fontsize=6.5,
                        color='#555555', ha='left')
            ax_last.text(1.05, 0.4, f'SGN: {pred_name}',
                        transform=ax_last.transAxes, fontsize=6,
                        color=check_color, ha='left')
            ax_last.text(1.05, 0.2, f'{checkmark} ({conf:.0%})',
                        transform=ax_last.transAxes, fontsize=6,
                        color=check_color, ha='left', fontweight='bold')

    fig.suptitle(f'{true_name}', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.4, h_pad=0.2, w_pad=0.2, rect=[0, 0, 0.87, 0.98])

    for ext in ['pdf']:
        path = os.path.join(output_dir, f'qualitative_annotated_multi_method.{ext}')
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
    parser.add_argument('--sgn_checkpoint', type=str,
                        default='output/downstream_disentangled_tmr_stable/ntu_sgn_ar_paired/model_best.pth.tar')
    parser.add_argument('--output_dir', type=str, default='paper/fig')
    parser.add_argument('--dataset', type=str, default='ntu')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

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

    print("Loading SGN model...")
    sgn_model = load_sgn_model(args.sgn_checkpoint, num_classes=49, device=device)
    print("  SGN loaded successfully")

    print("\nFinding standing samples...")
    selected = find_standing_samples(raw_data, methods['Ours'], args.dataset,
                                     STANDING_ACTIONS)
    for aid, fname in sorted(selected.items()):
        action_name = NTU_ACTIONS.get(aid, f"A{aid}")
        print(f"  Action {aid} ({action_name}): {fname}")

    if not selected:
        print("ERROR: No standing samples found!")
        return

    # Generate all annotated variations
    make_variation_k(raw_data, methods['Ours'], sgn_model, selected,
                     args.output_dir, device)
    make_variation_l(raw_data, methods, sgn_model, selected,
                     args.output_dir, device)
    make_variation_m(raw_data, methods, sgn_model, selected,
                     args.output_dir, device)

    print("\nDone! Annotated visualizations saved to", args.output_dir)


if __name__ == '__main__':
    main()
