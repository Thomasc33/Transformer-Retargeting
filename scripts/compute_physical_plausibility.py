#!/usr/bin/env python3
"""
Comprehensive physical plausibility metrics for retargeted skeleton motion.

Extends compute_reconstruction_quality.py with additional biomechanical metrics.
Compares multiple retargeted datasets against the raw data baseline.

Metrics computed:
  1. MPJPE (centered) -- mean per-joint position error vs. source action
  2. Bone length consistency -- std of bone lengths across frames (lower=better)
  3. Temporal smoothness (acceleration) -- mean |accel| (lower=better)
  4. Velocity MSE -- MSE of joint velocities vs. source (lower=better)
  5. Jerk magnitude -- mean |jerk| = |d^3 x / dt^3| (lower=smoother)
  6. Foot skating ratio -- fraction of frames where grounded foot moves (lower=better)
  7. Joint angle statistics -- mean/std of joint angles, % outside plausible range
  8. Motion FID -- Frechet distance between raw and retargeted motion feature distributions

Usage:
    python scripts/compute_physical_plausibility.py \\
        --retargeted_paths output/retargeted_data/disentangled_tmr_stable_retargeted.pkl \\
                           output/retargeted_data/dmr_ntu_cv_retargeted.pkl \\
        --method_names DisentangledTMR DMR \\
        --dataset ntu \\
        --output_path output/physical_plausibility_metrics.json
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import load_data as load_raw_data, parse_file_name

# ---------------------------------------------------------------------------
# NTU skeleton topology (0-indexed)
# ---------------------------------------------------------------------------

NTU_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),       # spine
    (20, 4), (4, 5), (5, 6), (6, 7),         # left arm
    (20, 8), (8, 9), (9, 10), (10, 11),      # right arm
    (0, 12), (12, 13), (13, 14), (14, 15),   # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),   # right leg
    (7, 21), (7, 22),                         # left hand tips
    (11, 23), (11, 24),                       # right hand tips
]

# Foot joints (0-indexed): left foot=15, right foot=19, left toe=14, right toe=18
FOOT_JOINTS = [14, 15, 18, 19]

# Joint triples (parent, joint, child) for angle computation.
# Each triple defines the angle at the middle joint.
JOINT_TRIPLES = [
    # Spine
    (0, 1, 20), (1, 20, 2), (20, 2, 3),
    # Left arm: shoulder, elbow, wrist
    (20, 4, 5), (4, 5, 6), (5, 6, 7),
    # Right arm: shoulder, elbow, wrist
    (20, 8, 9), (8, 9, 10), (9, 10, 11),
    # Left leg: hip, knee, ankle
    (0, 12, 13), (12, 13, 14), (13, 14, 15),
    # Right leg: hip, knee, ankle
    (0, 16, 17), (16, 17, 18), (17, 18, 19),
]

# Physiological angle limits (degrees) -- generous bounds
# Format: (min_deg, max_deg)
ANGLE_LIMITS = {
    # Spine angles
    (0, 1, 20): (90, 180),
    (1, 20, 2): (90, 180),
    (20, 2, 3): (90, 180),
    # Elbows -- can't hyperextend
    (4, 5, 6): (10, 180),
    (8, 9, 10): (10, 180),
    # Wrists
    (5, 6, 7): (30, 180),
    (9, 10, 11): (30, 180),
    # Shoulders -- very flexible
    (20, 4, 5): (10, 180),
    (20, 8, 9): (10, 180),
    # Knees -- can't hyperextend
    (12, 13, 14): (10, 180),
    (16, 17, 18): (10, 180),
    # Hips
    (0, 12, 13): (10, 180),
    (0, 16, 17): (10, 180),
    # Ankles
    (13, 14, 15): (30, 180),
    (17, 18, 19): (30, 180),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def to_skeleton(seq):
    """Ensure shape is (T, V=25, C=3)."""
    if seq.ndim == 2:
        T, VC = seq.shape
        return seq.reshape(T, 25, VC // 25)
    return seq


def compute_bone_lengths(skel, bones=NTU_BONES):
    """skel: (T, V, C). Returns (T, num_bones)."""
    lengths = np.zeros((skel.shape[0], len(bones)))
    for i, (p, c) in enumerate(bones):
        lengths[:, i] = np.linalg.norm(skel[:, c] - skel[:, p], axis=-1)
    return lengths


def compute_joint_angles(skel, triples=JOINT_TRIPLES):
    """Compute angles at middle joint for each triple.
    skel: (T, V, C). Returns (T, num_triples) in degrees.
    """
    T = skel.shape[0]
    angles = np.zeros((T, len(triples)))
    for i, (a, b, c) in enumerate(triples):
        v1 = skel[:, a] - skel[:, b]  # (T, C)
        v2 = skel[:, c] - skel[:, b]  # (T, C)
        # Compute cos(angle) via dot product
        dot = np.sum(v1 * v2, axis=-1)  # (T,)
        norm1 = np.linalg.norm(v1, axis=-1) + 1e-8
        norm2 = np.linalg.norm(v2, axis=-1) + 1e-8
        cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
        angles[:, i] = np.degrees(np.arccos(cos_angle))
    return angles


def compute_foot_skating(skel, foot_joints=FOOT_JOINTS, height_threshold=0.05,
                         velocity_threshold=0.01):
    """Estimate foot skating: fraction of foot frames that are grounded but moving.

    A foot is considered grounded when its y-coordinate (height) is below the
    ``height_threshold`` percentile of all foot heights for that sequence.
    Skating occurs when a grounded foot has velocity above ``velocity_threshold``.

    skel: (T, V, C). Returns float in [0, 1].
    """
    T = skel.shape[0]
    if T < 2:
        return 0.0

    foot_pos = skel[:, foot_joints, :]  # (T, num_feet, C)
    # Use y-axis (index 1) as height; NTU skeleton y is vertical
    foot_height = foot_pos[:, :, 1]  # (T, num_feet)

    # Height threshold: relative to this sequence's foot range
    h_thresh = np.percentile(foot_height, height_threshold * 100)

    # Velocity of each foot joint
    foot_vel = np.diff(foot_pos, axis=0)  # (T-1, num_feet, C)
    foot_speed = np.linalg.norm(foot_vel, axis=-1)  # (T-1, num_feet)

    # Grounded mask: foot below threshold at both frames of velocity interval
    grounded = (foot_height[:-1] <= h_thresh) | (foot_height[1:] <= h_thresh)  # (T-1, num_feet)

    # Skating: grounded AND fast
    skating = grounded & (foot_speed > velocity_threshold)

    # Ratio of skating events
    total_ground_frames = grounded.sum()
    if total_ground_frames == 0:
        return 0.0
    return float(skating.sum()) / float(total_ground_frames)


def compute_motion_features(skel):
    """Extract a simple motion feature vector for FID computation.
    Returns a 1D vector summarizing the sequence.
    skel: (T, V, C)
    """
    T, V, C = skel.shape
    # Center at spine base
    centered = skel - skel[:, 0:1, :]

    features = []

    # 1. Mean and std joint positions (V*C * 2)
    features.append(centered.mean(axis=0).flatten())
    features.append(centered.std(axis=0).flatten())

    # 2. Mean and std velocity (V*C * 2)
    if T >= 2:
        vel = np.diff(centered, axis=0)
        features.append(vel.mean(axis=0).flatten())
        features.append(vel.std(axis=0).flatten())
    else:
        features.append(np.zeros(V * C))
        features.append(np.zeros(V * C))

    # 3. Bone lengths mean/std (num_bones * 2)
    bl = compute_bone_lengths(skel)
    features.append(bl.mean(axis=0))
    features.append(bl.std(axis=0))

    return np.concatenate(features)


def compute_fid(features_real, features_gen):
    """Compute Frechet Inception Distance between two sets of feature vectors.
    Uses numpy only (no torch needed).
    features_real, features_gen: (N, D) arrays.
    """
    mu_r = features_real.mean(axis=0)
    mu_g = features_gen.mean(axis=0)
    sigma_r = np.cov(features_real, rowvar=False)
    sigma_g = np.cov(features_gen, rowvar=False)

    diff = mu_r - mu_g
    # Product of covariance matrices
    from scipy.linalg import sqrtm
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2 * np.trace(covmean)
    return float(fid)


# ---------------------------------------------------------------------------
# Per-sequence metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(retargeted_data, raw_data, dataset='ntu'):
    """Compute comprehensive metrics for a retargeted dataset.

    Returns a dict of aggregate statistics.
    """
    per_sample = {
        'mpjpe': [],
        'bone_consistency': [],
        'accel_magnitude': [],
        'velocity_mse': [],
        'jerk_magnitude': [],
        'foot_skating': [],
        'angle_violation_ratio': [],
        'mean_angle': [],
    }

    raw_features = []
    ret_features = []

    for fname in retargeted_data:
        if fname not in raw_data:
            continue

        src = to_skeleton(np.asarray(raw_data[fname], dtype=np.float32))
        ret = to_skeleton(np.asarray(retargeted_data[fname], dtype=np.float32))

        T = min(src.shape[0], ret.shape[0])
        src = src[:T]
        ret = ret[:T]

        if T < 4:
            continue

        # Center at spine base (joint 0)
        src_c = src - src[:, 0:1, :]
        ret_c = ret - ret[:, 0:1, :]

        # 1. MPJPE
        mpjpe = np.mean(np.sqrt(np.sum((src_c - ret_c) ** 2, axis=-1)))
        per_sample['mpjpe'].append(mpjpe)

        # 2. Bone length consistency (std across frames)
        bl = compute_bone_lengths(ret)
        per_sample['bone_consistency'].append(np.mean(np.std(bl, axis=0)))

        # 3. Acceleration magnitude (temporal smoothness)
        vel = np.diff(ret_c, axis=0)        # (T-1, V, C)
        accel = np.diff(vel, axis=0)         # (T-2, V, C)
        per_sample['accel_magnitude'].append(np.mean(np.linalg.norm(accel, axis=-1)))

        # 4. Velocity MSE
        src_vel = np.diff(src_c, axis=0)
        ret_vel = np.diff(ret_c, axis=0)
        per_sample['velocity_mse'].append(np.mean((src_vel - ret_vel) ** 2))

        # 5. Jerk magnitude (3rd derivative)
        if T >= 4:
            jerk = np.diff(accel, axis=0)    # (T-3, V, C)
            per_sample['jerk_magnitude'].append(np.mean(np.linalg.norm(jerk, axis=-1)))

        # 6. Foot skating
        per_sample['foot_skating'].append(compute_foot_skating(ret))

        # 7. Joint angles
        angles = compute_joint_angles(ret)  # (T, num_triples)
        per_sample['mean_angle'].append(np.mean(angles))

        # Count angle violations
        violations = 0
        total = 0
        for idx, triple in enumerate(JOINT_TRIPLES):
            if triple in ANGLE_LIMITS:
                lo, hi = ANGLE_LIMITS[triple]
                col = angles[:, idx]
                violations += np.sum((col < lo) | (col > hi))
                total += len(col)
        per_sample['angle_violation_ratio'].append(violations / max(1, total))

        # 8. Motion features for FID
        raw_features.append(compute_motion_features(src))
        ret_features.append(compute_motion_features(ret))

    # Aggregate
    results = {'num_samples': len(per_sample['mpjpe'])}
    for key, values in per_sample.items():
        if values:
            arr = np.array(values)
            results[f'{key}_mean'] = float(arr.mean())
            results[f'{key}_std'] = float(arr.std())
        else:
            results[f'{key}_mean'] = float('nan')
            results[f'{key}_std'] = float('nan')

    # Motion FID
    if len(raw_features) >= 10:
        raw_feat = np.stack(raw_features)
        ret_feat = np.stack(ret_features)
        try:
            results['motion_fid'] = compute_fid(raw_feat, ret_feat)
        except Exception as e:
            print(f"  FID computation failed: {e}")
            results['motion_fid'] = float('nan')
    else:
        results['motion_fid'] = float('nan')

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Comprehensive physical plausibility metrics"
    )
    p.add_argument(
        "--retargeted_paths",
        nargs="+",
        required=True,
        help="Paths to retargeted .pkl files",
    )
    p.add_argument(
        "--method_names",
        nargs="+",
        default=None,
        help="Method names for each retargeted path (same order). "
             "Defaults to filenames.",
    )
    p.add_argument("--dataset", default="ntu", choices=["ntu", "ntu120", "etri"])
    p.add_argument("--output_path", default=None, help="Path to save JSON results")
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Limit number of samples to evaluate (-1 = all)")
    return p.parse_args()


def main():
    args = parse_args()

    # Method names
    if args.method_names:
        assert len(args.method_names) == len(args.retargeted_paths), \
            "Number of method names must match number of retargeted paths"
        names = args.method_names
    else:
        names = [Path(p).stem for p in args.retargeted_paths]

    # Load raw data
    print(f"Loading raw {args.dataset} data...")
    raw_data = load_raw_data(args.dataset)
    print(f"  {len(raw_data)} raw samples")

    # Also compute metrics on raw data itself (as a reference baseline)
    all_results = {}

    print("\nComputing raw data self-metrics (reference)...")
    raw_self = compute_all_metrics(raw_data, raw_data, args.dataset)
    raw_self['method'] = 'Raw (self)'
    all_results['Raw (self)'] = raw_self
    print(f"  Done: {raw_self['num_samples']} samples")

    # Evaluate each retargeted dataset
    for path, name in zip(args.retargeted_paths, names):
        print(f"\nLoading {name} from {path}...")
        with open(path, 'rb') as f:
            ret_data = pickle.load(f)

        if args.max_samples > 0:
            keys = list(ret_data.keys())[:args.max_samples]
            ret_data = {k: ret_data[k] for k in keys}

        print(f"  {len(ret_data)} retargeted samples")
        print(f"  Computing metrics for {name}...")
        result = compute_all_metrics(ret_data, raw_data, args.dataset)
        result['method'] = name
        all_results[name] = result
        print(f"  Done: {result['num_samples']} samples")

    # Print comparison table
    print(f"\n{'='*100}")
    print("PHYSICAL PLAUSIBILITY METRICS COMPARISON")
    print(f"{'='*100}")

    header_fields = [
        ('Method', 20),
        ('MPJPE', 10),
        ('BoneCon', 10),
        ('Accel', 10),
        ('VelMSE', 10),
        ('Jerk', 10),
        ('FootSk', 10),
        ('AngViol', 10),
        ('FID', 10),
    ]
    header = ''.join(f'{name:<{w}}' for name, w in header_fields)
    print(header)
    print('-' * 100)

    for method_name, res in all_results.items():
        row = f"{method_name:<20}"
        row += f"{res.get('mpjpe_mean', float('nan')):>9.4f} "
        row += f"{res.get('bone_consistency_mean', float('nan')):>9.6f} "
        row += f"{res.get('accel_magnitude_mean', float('nan')):>9.6f} "
        row += f"{res.get('velocity_mse_mean', float('nan')):>9.6f} "
        row += f"{res.get('jerk_magnitude_mean', float('nan')):>9.6f} "
        row += f"{res.get('foot_skating_mean', float('nan')):>9.4f} "
        row += f"{res.get('angle_violation_ratio_mean', float('nan')):>9.4f} "
        row += f"{res.get('motion_fid', float('nan')):>9.2f} "
        print(row)

    # Save
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_path}")


if __name__ == '__main__':
    main()
