#!/usr/bin/env python3
"""
Retarget the same samples through multiple TMR models and generate
per-sample comparison figures (rows=models, cols=timesteps).

Picks a fixed set of GT sample pairs, runs each model on them, and
generates one PDF per sample so the user can compare model outputs.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import parse_file_name, sample_frames_fast, load_data
from src.model.disentangled_tmr import create_disentangled_tmr

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

# Broad set of actions — we want variety
TARGET_ACTIONS = list(range(1, 50))

FRAME_INDICES = [4, 16, 28, 40, 52, 60]


def load_model(checkpoint_path, device, dataset_name='ntu'):
    """Load a TMR model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", None)
    if isinstance(ckpt_args, dict):
        ckpt_args = argparse.Namespace(**ckpt_args)

    d_action = getattr(ckpt_args, "d_action", 768) if ckpt_args else 768
    d_identity = getattr(ckpt_args, "d_identity", 256) if ckpt_args else 256
    d_model = getattr(ckpt_args, "d_model", 320) if ckpt_args else 320
    tokenizer = getattr(ckpt_args, "tokenizer", None) if ckpt_args else None
    if tokenizer in ("none", "None"):
        tokenizer = None

    from src.data.datasets import datasets
    num_class = datasets[dataset_name]['num_class']

    model = create_disentangled_tmr(
        dataset=dataset_name,
        num_class=num_class,
        device=device,
        d_action=d_action,
        d_identity=d_identity,
        d_model=d_model,
        num_decoder_layers=getattr(ckpt_args, "num_decoder_layers", 6) if ckpt_args else 6,
        use_pretrained_action=getattr(ckpt_args, "use_action_backbone", True) if ckpt_args else True,
        use_temporal_convs=not getattr(ckpt_args, "no_temporal_convs", False) if ckpt_args else True,
        use_lstm=not getattr(ckpt_args, "no_lstm", False) if ckpt_args else True,
        identity_use_full_sequence=(getattr(ckpt_args, "identity_mode", "static") == "full_seq") if ckpt_args else False,
        tokenizer_type=tokenizer,
    )

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"  ERROR loading {checkpoint_path}: {e}")
        return None
    model.eval()
    return model


def prepare_input(raw_seq, seg=64):
    """(Frames, V*C) -> (1, C, T, V, 1) tensor."""
    seq = sample_frames_fast(raw_seq, seg)
    tensor = torch.from_numpy(seq).float()
    T, VC = tensor.shape
    tensor = tensor.reshape(T, 25, 3).permute(2, 0, 1)  # (C, T, V)
    return tensor.unsqueeze(0).unsqueeze(-1)  # (1, C, T, V, 1)


def retarget_sample(model, src_seq, tgt_seq, seg, device):
    """Retarget one sample, return (T, 25, 3) numpy array."""
    x1 = prepare_input(src_seq, seg).to(device)
    x2 = prepare_input(tgt_seq, seg).to(device)

    with torch.no_grad():
        output, _, _ = model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)

    first_frame = x2[:, :, 0:1, :, :]
    output_padded = torch.cat([first_frame, output], dim=2)
    out = output_padded.squeeze(-1).squeeze(0).permute(1, 2, 0).cpu().numpy()  # (T, V, C)
    return out



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
                  joint_size=10, linestyle='-'):
    transformed = transform_frame(joints_frame)
    x, y = transformed[:, 0], transformed[:, 1]
    for (i, j) in NTU_BONES:
        ax.plot([x[i], x[j]], [y[i], y[j]], c=color, linewidth=linewidth,
                alpha=alpha, solid_capstyle='round', linestyle=linestyle)
    ax.scatter(x, y, c=color, s=joint_size, zorder=5, alpha=alpha,
              edgecolors='white', linewidths=0.3)


def select_sample_pairs(raw_data, n_samples=10, seed=42):
    """Select diverse (source, target) pairs across actions."""
    rng = np.random.RandomState(seed)
    fnames = list(raw_data.keys())

    # Group by action
    by_action = {}
    for fname in fnames:
        info = parse_file_name(fname)
        if info is None:
            continue
        a = info.get('A', 0)
        if 1 <= a <= 49:
            by_action.setdefault(a, []).append(fname)

    pairs = []
    actions_used = set()
    # Try to get one sample per action
    action_list = sorted(by_action.keys())
    rng.shuffle(action_list)

    for action in action_list:
        if len(pairs) >= n_samples:
            break
        candidates = by_action[action]
        if len(candidates) < 2:
            continue

        # Pick source
        src_fname = rng.choice(candidates)
        src_info = parse_file_name(src_fname)

        # Pick target with different identity
        others = [f for f in fnames if parse_file_name(f) and parse_file_name(f)['P'] != src_info['P']]
        if not others:
            continue
        tgt_fname = rng.choice(others)

        pairs.append((src_fname, tgt_fname, action))
        actions_used.add(action)

    print(f"Selected {len(pairs)} sample pairs across {len(actions_used)} actions")
    return pairs


def generate_multi_model_figure(sample_idx, src_fname, gt_joints, model_outputs,
                                 model_names, output_dir, action_name):
    """Generate figure: rows=GT+models, cols=timesteps."""
    n_rows = 1 + len(model_names)  # GT + each model
    n_cols = len(FRAME_INDICES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Store sequences (no manual centering -- transform_frame handles it)
    all_sequences = {'Ground Truth': gt_joints}
    for name, joints in zip(model_names, model_outputs):
        all_sequences[name] = joints

    # Compute global axis limits using 3/4-view transform
    all_x, all_y = [], []
    for seq in all_sequences.values():
        for fi in FRAME_INDICES:
            t = min(fi, seq.shape[0] - 1)
            transformed = transform_frame(seq[t])
            all_x.extend(transformed[:, 0].tolist())
            all_y.extend(transformed[:, 1].tolist())
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    pad = max(xmax - xmin, ymax - ymin) * 0.15
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad

    colors = {
        'Ground Truth': '#2171b5',
    }
    # Assign distinct colors to models
    model_colors = ['#e6550d', '#31a354', '#6a3d9a', '#d62728', '#ff7f0e',
                    '#1f77b4', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    row_labels = ['Ground Truth'] + model_names
    for row_idx, label in enumerate(row_labels):
        if label == 'Ground Truth':
            color = '#2171b5'
        else:
            color = model_colors[(row_idx - 1) % len(model_colors)]

        seq = all_sequences[label]
        for col_idx, fi in enumerate(FRAME_INDICES):
            ax = axes[row_idx, col_idx]
            t = min(fi, seq.shape[0] - 1)
            draw_skeleton(ax, seq[t], color=color, linewidth=1.5, joint_size=6)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_aspect('equal', adjustable='datalim')

            if row_idx == 0:
                ax.set_title(f't={fi}', fontsize=9, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=8, fontweight='bold', rotation=90,
                             labelpad=10)

    fig.suptitle(f'{action_name} ({src_fname})', fontsize=10, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(output_dir, f'multimodel_{sample_idx:02d}_{src_fname}.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True,
                        help='checkpoint_path:label pairs, e.g. output/stable/ckpt.pth:Stable')
    parser.add_argument('--dataset', default='ntu')
    parser.add_argument('--seg', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--output_dir', default='paper/fig/multi_model_comparison')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse model specs
    model_specs = []
    for spec in args.models:
        parts = spec.split(':')
        path = parts[0]
        label = parts[1] if len(parts) > 1 else Path(path).parent.name
        model_specs.append((path, label))

    print(f"Will compare {len(model_specs)} models:")
    for path, label in model_specs:
        print(f"  {label}: {path}")

    # Load data
    raw_data = load_data(args.dataset, args.seg)
    print(f"Loaded {len(raw_data)} raw sequences")

    # Select sample pairs
    pairs = select_sample_pairs(raw_data, n_samples=args.n_samples, seed=args.seed)

    # For each model, retarget all pairs
    all_model_outputs = {}  # model_label -> list of (T,25,3) arrays
    model_names = []

    for ckpt_path, label in model_specs:
        print(f"\nLoading model: {label} ({ckpt_path})")
        model = load_model(ckpt_path, device, args.dataset)
        if model is None:
            print(f"  SKIPPING {label} (incompatible checkpoint)")
            continue
        model_names.append(label)

        outputs = []
        for src_fname, tgt_fname, action in pairs:
            src_seq = raw_data[src_fname]
            tgt_seq = raw_data[tgt_fname]
            ret_joints = retarget_sample(model, src_seq, tgt_seq, args.seg, device)
            outputs.append(ret_joints)
        all_model_outputs[label] = outputs

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Generate figures
    print(f"\nGenerating {len(pairs)} comparison figures...")
    for idx, (src_fname, tgt_fname, action) in enumerate(pairs):
        gt_seq = raw_data[src_fname]
        gt_joints = sample_frames_fast(gt_seq, args.seg)
        gt_joints = gt_joints.reshape(args.seg, 25, 3)

        model_outputs = [all_model_outputs[name][idx] for name in model_names]
        action_name = NTU_ACTIONS.get(action, f'Action {action}')
        generate_multi_model_figure(idx, src_fname, gt_joints, model_outputs,
                                     model_names, args.output_dir, action_name)

    print(f"\nDone! Generated {len(pairs)} PDFs in {args.output_dir}")


if __name__ == '__main__':
    main()
