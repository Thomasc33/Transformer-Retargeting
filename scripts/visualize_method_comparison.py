#!/usr/bin/env python3
"""
Generate multi-method comparison figures using paired dataset quadruplets.
Shows Input (P1A1), Dummy (P2A2), GT (P2A1), DMR, PMR, Ours side-by-side.
Uses actual paired data so Dummy and GT are consistent.
Shows SGN prediction for each method's output.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import (
    parse_file_name, sample_frames_fast, datasets as DATASETS_CONFIG,
)
from src.model.sgn import SGN

NTU_BONES = [
    (1, 0), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

NTU_ACTIONS = {
    1: "Drink water", 2: "Eat meal", 3: "Brush teeth", 4: "Brush hair",
    5: "Drop", 6: "Pickup", 7: "Throw", 8: "Sit down", 9: "Stand up",
    10: "Clapping", 11: "Reading", 12: "Writing", 13: "Tear up paper",
    14: "Wear jacket", 15: "Take off jacket", 16: "Wear a shoe",
    17: "Take off a shoe", 18: "Wear on glasses", 19: "Take off glasses",
    20: "Put on hat", 21: "Take off hat", 22: "Cheer up",
    23: "Hand waving", 24: "Kicking", 25: "Reach into pocket",
    26: "Hopping", 27: "Jump up", 28: "Phone call",
    29: "Play with phone", 30: "Typing", 31: "Point to something",
    32: "Selfie", 33: "Check time", 34: "Rub two hands",
    35: "Nod head/bow", 36: "Shake head", 37: "Wipe face",
    38: "Salute", 39: "Put palms together", 40: "Cross hands",
    41: "Sneeze/cough", 42: "Staggering", 43: "Falling",
    44: "Touch head", 45: "Touch chest", 46: "Touch back",
    47: "Touch neck", 48: "Nausea/vomiting", 49: "Use a fan",
}

# Actions exempt from sitting detection
SITTING_EXEMPT_ACTIONS = {8, 9}

FRAME_INDICES = [4, 16, 28, 40, 52, 60]


def is_sitting(joints):
    """Detect if person is sitting. joints: (T, 25, 3)."""
    mid_start = joints.shape[0] // 3
    mid_end = 2 * joints.shape[0] // 3
    mid_joints = joints[mid_start:mid_end]

    hip_y = mid_joints[:, 0, 1].mean()
    head_y = mid_joints[:, 3, 1].mean()
    left_foot_y = mid_joints[:, 15, 1].mean()
    right_foot_y = mid_joints[:, 19, 1].mean()
    foot_y = min(left_foot_y, right_foot_y)

    total_height = abs(head_y - foot_y)
    hip_height = abs(hip_y - foot_y)

    if total_height < 1e-6:
        return False

    return (hip_height / total_height) < 0.30



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
    T = seq.shape[0]
    return seq.reshape(T, 25, 3)


def motion_magnitude(joints):
    diffs = np.diff(joints, axis=0)
    per_frame = np.sqrt((diffs ** 2).sum(axis=-1)).mean(axis=-1)
    return per_frame.mean()


def draw_skeleton(ax, joints_frame, color='#333333', alpha=1.0, linewidth=2.0,
                  joint_size=10, linestyle='-'):
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


def compute_mse(a, b):
    min_len = min(a.shape[0], b.shape[0])
    return float(np.mean((a[:min_len] - b[:min_len]) ** 2))


def load_retargeted_data(pkl_path):
    """Load retargeted pkl and return dict: fname -> sequence (T, 75) numpy."""
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    result = {}
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict) and 'retargeted' in val:
                seq = val['retargeted']
            elif isinstance(val, np.ndarray):
                seq = val
            else:
                continue
            if hasattr(seq, 'numpy'):
                seq = seq.numpy()
            result[key] = seq
    print(f"  Loaded {len(result)} sequences")
    return result


def load_paired_data(data_path):
    """Load paired dataset. Returns (raw_X dict, sampled_data quadruplets)."""
    print(f"Loading paired data from {data_path}...")
    raw = torch.load(data_path, map_location='cpu')

    X = {}
    for split_name in ['train', 'test']:
        split_data = raw[split_name]
        if hasattr(split_data, 'X') and isinstance(split_data.X, dict):
            for key, val in split_data.X.items():
                if hasattr(val, 'numpy'):
                    val = val.numpy()
                X[key] = val

    # Use test split quadruplets
    quadruplets = raw['test'].sampled_data
    print(f"  Loaded {len(X)} sequences, {len(quadruplets)} quadruplets")
    return X, quadruplets


class SGNPredictor:
    """Runs SGN inference and returns predicted action name."""

    def __init__(self, checkpoint_path, dataset='ntu', seg=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seg = seg

        # Build action label map
        sorted_actions = sorted(range(1, 50))
        self.idx_to_action = {i: a for i, a in enumerate(sorted_actions)}
        self.action_to_idx = {a: i for i, a in enumerate(sorted_actions)}
        num_classes = len(sorted_actions)

        self.model = SGN(num_classes=num_classes, dataset='ntu', seg=seg).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in ckpt:
            self.model.load_state_dict(ckpt['state_dict'])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()

    def predict(self, seq):
        """Given (T, 75) numpy array, return (predicted_action_id, action_name, confidence)."""
        if hasattr(seq, 'numpy'):
            seq = seq.numpy()
        sampled = sample_frames_fast(seq, self.seg)
        x = torch.from_numpy(sampled).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            conf, pred_idx = probs.max(dim=-1)

        action_id = self.idx_to_action[pred_idx.item()]
        action_name = NTU_ACTIONS.get(action_id, f'A{action_id}')
        return action_id, action_name, conf.item()


def select_quadruplets(X, quadruplets, ours_data, sgn, max_per_action=3, max_samples=24):
    """Select diverse quadruplets with sitting detection and SGN filtering."""
    candidates = []

    for quad in quadruplets:
        p1, a1, fname_p1a1 = quad[0]
        p1_2, a2, fname_p1a2 = quad[1]
        p2, a1_2, fname_p2a1 = quad[2]
        p2_2, a2_2, fname_p2a2 = quad[3]

        # Skip if not in retargeted data
        if fname_p1a1 not in ours_data:
            continue

        # Skip if raw sequences missing
        for fn in [fname_p1a1, fname_p2a1, fname_p2a2]:
            if fn not in X:
                break
        else:
            pass  # all present
        if fname_p1a1 not in X or fname_p2a1 not in X or fname_p2a2 not in X:
            continue

        # Only single-person actions (1-49)
        if a1 < 1 or a1 > 49:
            continue

        input_joints = seq_to_joints(X[fname_p1a1])

        # Sitting detection on input
        if a1 not in SITTING_EXEMPT_ACTIONS and is_sitting(input_joints):
            continue

        # SGN prediction on our retargeted output
        pred_id, pred_name, conf = sgn.predict(ours_data[fname_p1a1])

        # Only keep if SGN correctly classifies our output
        if pred_id != a1:
            continue

        # MSE of our output vs GT (P2A1)
        ours_joints = seq_to_joints(ours_data[fname_p1a1])
        gt_joints = seq_to_joints(X[fname_p2a1])
        mse = compute_mse(gt_joints, ours_joints)

        # Pose variance
        pose_std = ours_joints.std(axis=0).mean()
        motion = motion_magnitude(ours_joints)

        candidates.append({
            'quad': quad,
            'fname_input': fname_p1a1,
            'fname_dummy': fname_p2a2,
            'fname_gt': fname_p2a1,
            'action': a1,
            'action_name': NTU_ACTIONS.get(a1, f'A{a1}'),
            'dummy_action': a2,
            'dummy_action_name': NTU_ACTIONS.get(a2, f'A{a2}'),
            'p1': p1, 'p2': p2,
            'mse': mse,
            'pose_std': pose_std,
            'motion': motion,
            'sgn_conf': conf,
        })

    # Sort by pose variance (desc), then MSE (asc)
    candidates.sort(key=lambda x: (-x['pose_std'], x['mse']))

    selected = []
    action_counts = {}
    for c in candidates:
        a = c['action']
        if action_counts.get(a, 0) >= max_per_action:
            continue
        action_counts[a] = action_counts.get(a, 0) + 1
        selected.append(c)
        if len(selected) >= max_samples:
            break

    print(f"Selected {len(selected)} samples across {len(action_counts)} actions")
    return selected


def generate_comparison_figure(sample, X, method_data, sgn, output_dir):
    """Generate a multi-method comparison figure for one quadruplet.

    Layout: rows = (Input P1A1, Dummy P2A2, GT P2A1, DMR, PMR, Ours), cols = timesteps.
    GT (P2A1) shown as ghost overlay on method rows.
    SGN prediction shown in row labels.
    """
    fname_input = sample['fname_input']
    fname_dummy = sample['fname_dummy']
    fname_gt = sample['fname_gt']

    colors = {
        'input': '#2171b5',
        'dummy': '#888888',
        'ground_truth': '#2ca02c',
        'dmr': '#e6550d',
        'pmr': '#31a354',
        'ours': '#6a3d9a',
    }

    # Load sequences
    sequences = {
        'input': seq_to_joints(X[fname_input]),
        'dummy': seq_to_joints(X[fname_dummy]),
        'ground_truth': seq_to_joints(X[fname_gt]),
    }

    # Method outputs + SGN predictions
    sgn_preds = {}
    for mk in ['dmr', 'pmr', 'ours']:
        if mk in method_data and fname_input in method_data[mk]:
            sequences[mk] = seq_to_joints(method_data[mk][fname_input])
            pred_id, pred_name, conf = sgn.predict(method_data[mk][fname_input])
            correct = (pred_id == sample['action'])
            sgn_preds[mk] = (pred_name, conf, correct)
        else:
            sequences[mk] = None
            sgn_preds[mk] = ('N/A', 0.0, False)

    action_name = sample['action_name']
    dummy_action_name = sample['dummy_action_name']
    p1, p2 = sample['p1'], sample['p2']

    # Build row labels with SGN predictions
    def sgn_label(mk):
        pred_name, conf, correct = sgn_preds[mk]
        mark = 'O' if correct else 'X'
        return f' [{mark}: {pred_name} {conf:.0%}]'

    row_specs = [
        ('input', f'Input (P{p1}, {action_name})'),
        ('dummy', f'Dummy (P{p2}, {dummy_action_name})'),
        ('ground_truth', f'GT (P{p2}, {action_name})'),
        ('dmr', f'DMR{sgn_label("dmr")}'),
        ('pmr', f'PMR{sgn_label("pmr")}'),
        ('ours', f'Ours{sgn_label("ours")}'),
    ]

    n_frames = len(FRAME_INDICES)
    n_rows = len(row_specs)

    fig, axes = plt.subplots(n_rows, n_frames, figsize=(n_frames * 2.2, n_rows * 2.5))

    # Compute bounds using 3/4-view transform (hip-centered + rotated)
    all_x, all_y = [], []
    for seq_name, seq_data in sequences.items():
        if seq_data is None:
            continue
        for t in range(seq_data.shape[0]):
            transformed = transform_frame(seq_data[t])
            all_x.append(transformed[:, 0])
            all_y.append(transformed[:, 1])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    half = max(np.abs(all_x).max(), np.abs(all_y).max()) * 1.2

    for row, (mk, method_name) in enumerate(row_specs):
        seq = sequences[mk]
        color = colors.get(mk, '#333333')

        for col, fi in enumerate(FRAME_INDICES):
            ax = axes[row, col]
            setup_axis(ax)
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)

            if seq is not None:
                t = min(fi, seq.shape[0] - 1)
                frame_joints = seq[t]

                # Draw GT ghost overlay on method rows
                if mk in ('dmr', 'pmr', 'ours'):
                    gt_t = min(fi, sequences['ground_truth'].shape[0] - 1)
                    gt_frame = sequences['ground_truth'][gt_t]
                    draw_skeleton(ax, gt_frame, color='#2ca02c',
                                alpha=0.2, linewidth=0.8, joint_size=2, linestyle=':')

                draw_skeleton(ax, frame_joints, color=color, alpha=0.9,
                            linewidth=2.0 if mk not in ('input', 'dummy', 'ground_truth') else 1.5,
                            joint_size=8 if mk not in ('input', 'dummy', 'ground_truth') else 6)
            else:
                ax.text(0, 0, 'N/A', ha='center', va='center', fontsize=8, color='gray')

            if row == 0:
                ax.set_title(f't={fi}', fontsize=8, pad=4)

        # Row label
        axes[row, 0].set_ylabel(method_name, fontsize=7, fontweight='bold',
                                rotation=90, labelpad=10)

    # MSE vs GT for subtitle
    mse_parts = []
    for mk, label in [('dmr', 'DMR'), ('pmr', 'PMR'), ('ours', 'Ours')]:
        if sequences[mk] is not None:
            mse = compute_mse(sequences['ground_truth'], sequences[mk])
            mse_parts.append(f'{label}: {mse:.4f}')
    mse_str = '  |  '.join(mse_parts) if mse_parts else ''

    fig.suptitle(f'{action_name}: P{p1} -> P{p2}\nMSE vs GT: {mse_str}',
                 fontsize=10, fontweight='bold', y=1.04)

    plt.tight_layout()

    safe_fname = fname_input.replace('/', '_').replace('.', '_')
    path = os.path.join(output_dir, f'comparison_{safe_fname}.pdf')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ours_pkl', default='output/retargeted_data/disentangled_tmr_stable_retargeted.pkl')
    parser.add_argument('--dmr_pkl', default='output/retargeted_data/dmr_ntu_cv_retargeted.pkl')
    parser.add_argument('--pmr_pkl', default='output/retargeted_data/pmr_ntu_cv_retargeted.pkl')
    parser.add_argument('--raw_data', default='data/ntu/ntu_cv_paired_10k.pt')
    parser.add_argument('--sgn_checkpoint',
                        default='output/downstream_disentangled_tmr_stable/ntu_sgn_ar_paired/model_best.pth.tar')
    parser.add_argument('--output_dir', default='paper/fig/method_comparison')
    parser.add_argument('--max_per_action', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=24)
    parser.add_argument('--seg', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load paired dataset (has quadruplets)
    X, quadruplets = load_paired_data(args.raw_data)

    # Load retargeted data
    ours_data = load_retargeted_data(args.ours_pkl)
    dmr_data = load_retargeted_data(args.dmr_pkl)
    pmr_data = load_retargeted_data(args.pmr_pkl)

    method_data = {
        'dmr': dmr_data,
        'pmr': pmr_data,
        'ours': ours_data,
    }

    # Load SGN for predictions
    print("Loading SGN model...")
    sgn = SGNPredictor(args.sgn_checkpoint, seg=args.seg)

    # Select quadruplets
    samples = select_quadruplets(X, quadruplets, ours_data, sgn,
                                  max_per_action=args.max_per_action,
                                  max_samples=args.max_samples)

    # Generate figures
    print(f"\nGenerating {len(samples)} comparison figures...")
    for s in samples:
        path = generate_comparison_figure(s, X, method_data, sgn, args.output_dir)
        dmr_pred = sgn.predict(method_data['dmr'][s['fname_input']])[1] if s['fname_input'] in method_data['dmr'] else 'N/A'
        pmr_pred = sgn.predict(method_data['pmr'][s['fname_input']])[1] if s['fname_input'] in method_data['pmr'] else 'N/A'
        print(f"  {s['action_name']:20s} P{s['p1']}->P{s['p2']}  mse={s['mse']:.4f}  SGN: DMR={dmr_pred}, PMR={pmr_pred}, Ours={s['action_name']}  -> {path}")

    print(f"\nDone. Generated {len(samples)} figures in {args.output_dir}")


if __name__ == '__main__':
    main()
