#!/usr/bin/env python3
"""
Find retargeted samples where SGN correctly predicts the action.
Then generate:
  1. Per-sample still images (PDF) showing GT ghost + retargeted overlay at multiple frames
  2. Per-sample GIF animations showing the retargeted skeleton over time

Outputs to paper/fig/good_samples/ with filenames that encode the sample info.
"""

import argparse
import os
import sys
import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import (
    parse_file_name, load_data, datasets as DATASETS_CONFIG,
    ActionRecognitionDataset, sample_frames_fast,
)
from src.model.sgn import SGN

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


def motion_magnitude(joints):
    """Compute average per-frame displacement across joints."""
    diffs = np.diff(joints, axis=0)  # (T-1, 25, 3)
    per_frame = np.sqrt((diffs ** 2).sum(axis=-1)).mean(axis=-1)  # (T-1,)
    return per_frame.mean()


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


def run_sgn_inference(model, data_dict, dataset, setting, seg, device):
    """Run SGN inference on all samples, return {filename: (predicted_class, gt_class, confidence)}."""
    ds = ActionRecognitionDataset(
        data_dict, dataset, setting,
        split='test', task='ar', seg=seg, augment=False,
        drop_two_person_actions=(dataset == 'ntu'),
    )

    # Build inverse label map: contiguous -> original 1-indexed action
    inv_action_map = None
    if hasattr(ds, 'action_label_map') and ds.action_label_map is not None:
        inv_action_map = {v: k for k, v in ds.action_label_map.items()}

    results = {}
    model.eval()
    with torch.no_grad():
        for idx in range(len(ds)):
            skeleton, label = ds[idx]
            fname = ds.samples[idx][0]
            x = skeleton.unsqueeze(0).to(device)  # (1, seg, 75)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            pred_class = logits.argmax(dim=-1).item()
            confidence = probs[0, pred_class].item()

            # Map back to 1-indexed action
            if inv_action_map:
                pred_action = inv_action_map.get(pred_class, pred_class)
                gt_action = inv_action_map.get(label, label)
            else:
                pred_action = pred_class + 1
                gt_action = label + 1

            results[fname] = {
                'pred': pred_action,
                'gt': gt_action,
                'correct': (pred_class == label),
                'confidence': confidence,
            }

    return results


def generate_still(gt_joints, ret_joints, fname, info, output_dir,
                   frame_indices=[8, 20, 32, 44, 56]):
    """Generate a single-row still image showing GT ghost + retargeted at multiple frames."""
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

    for i, fi in enumerate(frame_indices):
        ax = axes[i]
        setup_axis(ax)
        t = min(fi, gt_joints.shape[0] - 1, ret_joints.shape[0] - 1)

        # GT ghost
        draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.25,
                     linewidth=1.2, joint_size=3, linestyle='--')
        # Retargeted bold
        draw_skeleton(ax, ret_joints[t], color='#d94801', alpha=0.9,
                     linewidth=2.0, joint_size=8)

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(f't={fi}', fontsize=8, pad=4)

    fig.suptitle(f'{action_name} (P{info["P"]})', fontsize=10, fontweight='bold', y=1.02)

    plt.tight_layout()
    safe_fname = fname.replace('/', '_').replace('.', '_')
    path = os.path.join(output_dir, f'still_{safe_fname}.pdf')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def generate_gif(gt_joints, ret_joints, fname, info, output_dir, fps=10):
    """Generate an animated GIF showing GT ghost + retargeted skeleton over time."""
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
    T = min(gt_joints.shape[0], ret_joints.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    def animate(t):
        ax.clear()
        setup_axis(ax)
        # GT ghost
        draw_skeleton(ax, gt_joints[t], color='#2171b5', alpha=0.25,
                     linewidth=1.2, joint_size=3, linestyle='--')
        # Retargeted bold
        draw_skeleton(ax, ret_joints[t], color='#d94801', alpha=0.9,
                     linewidth=2.0, joint_size=10)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(f'{action_name} (P{info["P"]})  t={t}', fontsize=10, fontweight='bold')
        return []

    # Use frames starting from 4 to skip initial weirdness
    frames = list(range(4, T))
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000//fps, blit=True)

    safe_fname = fname.replace('/', '_').replace('.', '_')
    path = os.path.join(output_dir, f'anim_{safe_fname}.gif')
    anim.save(path, writer='pillow', fps=fps)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ntu')
    parser.add_argument('--setting', type=str, default='cv')
    parser.add_argument('--seg', type=int, default=64)
    parser.add_argument('--retargeted_pkl', type=str,
                        default='output/retargeted_data/disentangled_tmr_stable_retargeted.pkl')
    parser.add_argument('--sgn_checkpoint', type=str,
                        default='output/downstream_disentangled_tmr_stable/ntu_sgn_ar_paired/model_best.pth.tar')
    parser.add_argument('--output_dir', type=str, default='paper/fig/good_samples')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Max good samples to visualize')
    parser.add_argument('--min_motion', type=float, default=0.01,
                        help='Minimum average motion magnitude to filter static samples')
    parser.add_argument('--still_only', action='store_true',
                        help='Only generate stills, skip GIFs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load raw data
    print("Loading raw data...")
    raw_data = load_data(args.dataset, T=64)

    # Load retargeted data
    print("Loading retargeted data...")
    with open(args.retargeted_pkl, 'rb') as f:
        ret_data = pickle.load(f)
    print(f"  Retargeted: {len(ret_data)} samples")

    # Load SGN model
    print("Loading SGN model...")
    # Figure out num_classes from a quick dataset build
    ds_tmp = ActionRecognitionDataset(
        ret_data, args.dataset, args.setting,
        split='test', task='ar', seg=args.seg, augment=False,
        drop_two_person_actions=(args.dataset == 'ntu'),
    )
    num_classes = ds_tmp.num_classes
    print(f"  Num classes: {num_classes}")

    model = SGN(num_classes=num_classes, dataset=args.dataset, seg=args.seg, bias=True)
    ckpt = torch.load(args.sgn_checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    print("  Model loaded.")

    # Run inference on retargeted data
    print("\nRunning SGN inference on retargeted data...")
    results = run_sgn_inference(model, ret_data, args.dataset, args.setting, args.seg, device)

    correct = {k: v for k, v in results.items() if v['correct']}
    print(f"  Total test samples: {len(results)}")
    print(f"  Correctly predicted: {len(correct)} ({100*len(correct)/len(results):.1f}%)")

    # Filter: standing + sufficient motion + in both raw and retargeted
    good_samples = []
    for fname, res in correct.items():
        if fname not in raw_data or fname not in ret_data:
            continue
        gt_joints = seq_to_joints(raw_data[fname])
        ret_joints = seq_to_joints(ret_data[fname])

        # Check standing
        if not is_standing(gt_joints):
            continue

        # Check motion magnitude on retargeted
        motion = motion_magnitude(ret_joints)
        if motion < args.min_motion:
            continue

        # Compute MSE between retargeted and ground truth
        min_len = min(gt_joints.shape[0], ret_joints.shape[0])
        mse = float(np.mean((gt_joints[:min_len] - ret_joints[:min_len]) ** 2))

        info = parse_file_name(fname, args.dataset)
        good_samples.append({
            'fname': fname,
            'action': res['gt'],
            'action_name': NTU_ACTIONS_1INDEXED.get(res['gt'], f"A{res['gt']}"),
            'person': info['P'],
            'confidence': res['confidence'],
            'motion': motion,
            'mse': mse,
            'info': info,
        })

    # Sort by lowest MSE first (closest to GT), then highest confidence
    good_samples.sort(key=lambda x: (x['mse'], -x['confidence']))

    # Take diverse set: max 3 per action
    action_counts = {}
    diverse_samples = []
    for s in good_samples:
        a = s['action']
        if action_counts.get(a, 0) >= 5:
            continue
        action_counts[a] = action_counts.get(a, 0) + 1
        diverse_samples.append(s)
        if len(diverse_samples) >= args.max_samples:
            break

    print(f"\n  Good standing+motion samples: {len(good_samples)}")
    print(f"  Selected diverse samples: {len(diverse_samples)}")

    # Print manifest
    print("\n" + "=" * 80)
    print("SELECTED SAMPLES (sorted by action)")
    print("=" * 80)
    diverse_samples.sort(key=lambda x: (x['action'], -x['confidence']))

    manifest = []
    for i, s in enumerate(diverse_samples):
        line = (f"  [{i+1:3d}] {s['fname']:<45s} "
                f"A{s['action']:02d} {s['action_name']:<25s} "
                f"P{s['person']:02d}  conf={s['confidence']:.3f}  mse={s['mse']:.6f}  motion={s['motion']:.4f}")
        print(line)
        manifest.append({
            'index': i + 1,
            'fname': s['fname'],
            'action': s['action'],
            'action_name': s['action_name'],
            'person': s['person'],
            'confidence': float(s['confidence']),
            'mse': float(s['mse']),
            'motion': float(s['motion']),
        })

    # Save manifest
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    still_paths = []
    gif_paths = []

    for i, s in enumerate(diverse_samples):
        fname = s['fname']
        gt_joints = seq_to_joints(raw_data[fname])
        ret_joints = seq_to_joints(ret_data[fname])
        info = s['info']

        # Still
        spath = generate_still(gt_joints, ret_joints, fname, info, args.output_dir)
        still_paths.append(spath)
        print(f"  [{i+1}/{len(diverse_samples)}] Still: {spath}")

        # GIF
        if not args.still_only:
            gpath = generate_gif(gt_joints, ret_joints, fname, info, args.output_dir)
            gif_paths.append(gpath)
            print(f"  [{i+1}/{len(diverse_samples)}] GIF:   {gpath}")

    # Summary
    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"\nManifest: {manifest_path}")
    print(f"\nStills ({len(still_paths)}):")
    for p in still_paths:
        print(f"  {p}")
    if gif_paths:
        print(f"\nGIFs ({len(gif_paths)}):")
        for p in gif_paths:
            print(f"  {p}")

    print("\nDone!")


if __name__ == '__main__':
    main()
