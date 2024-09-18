#!/usr/bin/env python3
"""
Input sensitivity analysis for DisentangledTMR.

Tests robustness to:
1. Gaussian noise at varying sigma levels
2. Random joint dropout (simulating occlusion)
3. Reduced temporal resolution

For each degradation level:
- Retarget degraded input with DisentangledTMR
- Compute MPJPE and bone error vs clean retargeting
- Evaluate downstream AR/RI on degraded retargeted output
- Generate line plots showing degradation curves

Must run via SLURM (loads TMR model + downstream models + data).
"""

import argparse
import json
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import (
    datasets as DATASETS_CONFIG,
    parse_file_name,
    sample_frames_fast,
    load_data,
)
from src.model.disentangled_tmr import create_disentangled_tmr
from src.model.sgn import SGN

# NTU skeleton bone connections
NTU_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7),
    (20, 8), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
    (7, 21), (7, 22), (11, 23), (11, 24),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ntu")
    p.add_argument("--setting", default="cv")
    p.add_argument("--seg", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tmr_checkpoint",
                   default="output/disentangled_tmr_stable/checkpoint_stage3_best.pth")
    p.add_argument("--sgn_ar_checkpoint",
                   default="output/downstream_disentangled_tmr_stable/ntu_sgn_ar_paired/model_best.pth.tar")
    p.add_argument("--sgn_ri_checkpoint",
                   default="output/downstream_disentangled_tmr_stable/ntu_sgn_ri_paired/model_best.pth.tar")
    p.add_argument("--output_dir", default="output/analysis/sensitivity")
    p.add_argument("--num_samples", type=int, default=500,
                   help="Number of samples to evaluate (for speed)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_tmr_model(checkpoint_path, device, dataset_name):
    """Load TMR model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", None)
    if ckpt_args and isinstance(ckpt_args, dict):
        ckpt_args = argparse.Namespace(**ckpt_args)

    d_action = getattr(ckpt_args, "d_action", 768) if ckpt_args else 768
    d_identity = getattr(ckpt_args, "d_identity", 256) if ckpt_args else 256
    d_model = getattr(ckpt_args, "d_model", 320) if ckpt_args else 320
    num_class = DATASETS_CONFIG[dataset_name]["num_class"]

    tokenizer = getattr(ckpt_args, "tokenizer", None) if ckpt_args else None
    if tokenizer in ("none", "None"):
        tokenizer = None

    model = create_disentangled_tmr(
        dataset=dataset_name, num_class=num_class, device=device,
        d_action=d_action, d_identity=d_identity, d_model=d_model,
        num_decoder_layers=getattr(ckpt_args, "num_decoder_layers", 6) if ckpt_args else 6,
        use_pretrained_action=getattr(ckpt_args, "use_action_backbone", True) if ckpt_args else True,
        use_temporal_convs=not getattr(ckpt_args, "no_temporal_convs", False) if ckpt_args else True,
        use_lstm=not getattr(ckpt_args, "no_lstm", False) if ckpt_args else True,
        identity_use_full_sequence=False,
        tokenizer_type=tokenizer,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_sgn(checkpoint_path, device, dataset="ntu", seg=64):
    """Load SGN model."""
    if not os.path.exists(checkpoint_path):
        return None, 0
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    nc = sd["fc.weight"].shape[0] if "fc.weight" in sd else 49
    model = SGN(num_classes=nc, dataset=dataset, seg=seg, bias=True)
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    return model, nc


def prepare_input(raw_seq, seg=64):
    """Raw (frames, V*C) -> (1, C, T, V, 1)."""
    seq = sample_frames_fast(raw_seq, seg)
    t = torch.from_numpy(seq).float()
    T, VC = t.shape
    t = t.reshape(T, 25, 3).permute(2, 0, 1)  # (C, T, V)
    return t.unsqueeze(0).unsqueeze(-1)  # (1, C, T, V, 1)


def retarget_single(tmr_model, x1, x2, device):
    """Retarget x1 (source motion) to x2 (target identity)."""
    with torch.no_grad():
        output, _, _ = tmr_model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)
    first_frame = x2[:, :, 0:1, :, :]
    return torch.cat([first_frame, output], dim=2)


def apply_noise(seq, sigma):
    """Add Gaussian noise to skeleton sequence. seq: (T, V*C)."""
    return seq + np.random.randn(*seq.shape).astype(np.float32) * sigma


def apply_joint_dropout(seq, drop_rate):
    """Randomly zero out joints. seq: (T, V*C). V=25, C=3."""
    result = seq.copy()
    T, VC = result.shape
    V = 25
    C = VC // V
    result = result.reshape(T, V, C)
    mask = np.random.rand(T, V) > drop_rate  # True = keep
    result = result * mask[:, :, np.newaxis]
    return result.reshape(T, VC)


def apply_temporal_downsample(seq, target_frames, original_frames=64):
    """Downsample then upsample. seq: (T, V*C)."""
    T, VC = seq.shape
    # Downsample
    indices = np.linspace(0, T - 1, target_frames).astype(int)
    downsampled = seq[indices]
    # Upsample back via linear interpolation
    from scipy.interpolate import interp1d
    x_down = np.linspace(0, 1, target_frames)
    x_up = np.linspace(0, 1, original_frames)
    interp = interp1d(x_down, downsampled, axis=0, kind="linear")
    return interp(x_up).astype(np.float32)


def compute_mpjpe(retargeted, reference):
    """Compute MPJPE between two sequences. Both (T, V, C)."""
    T = min(retargeted.shape[0], reference.shape[0])
    diff = retargeted[:T] - reference[:T]
    return np.mean(np.sqrt(np.sum(diff ** 2, axis=-1)))


def compute_bone_error(seq, bones):
    """Compute mean absolute bone length deviation from the sequence's own mean."""
    T, V, C = seq.shape
    bone_lengths = np.zeros((T, len(bones)))
    for i, (p, c) in enumerate(bones):
        diff = seq[:, c, :] - seq[:, p, :]
        bone_lengths[:, i] = np.linalg.norm(diff, axis=-1)
    mean_lengths = bone_lengths.mean(axis=0)
    return np.mean(np.abs(bone_lengths - mean_lengths))


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load models
    print("Loading TMR model...")
    tmr_model = load_tmr_model(args.tmr_checkpoint, device, args.dataset)

    print("Loading SGN AR model...")
    sgn_ar, ar_nc = load_sgn(args.sgn_ar_checkpoint, device, args.dataset, args.seg)

    print("Loading SGN RI model...")
    sgn_ri, ri_nc = load_sgn(args.sgn_ri_checkpoint, device, args.dataset, args.seg)

    # Load raw data
    print("Loading raw data...")
    raw_data = load_data(args.dataset, args.seg)
    all_fnames = list(raw_data.keys())

    # Subsample for speed
    if args.num_samples > 0 and args.num_samples < len(all_fnames):
        np.random.shuffle(all_fnames)
        all_fnames = all_fnames[:args.num_samples]

    print(f"Evaluating on {len(all_fnames)} samples")

    # ==========================================
    # Experiment 1: Gaussian noise
    # ==========================================
    noise_sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
    noise_results = []

    print("\n" + "=" * 60)
    print("Experiment 1: Gaussian Noise")
    print("=" * 60)

    for sigma in noise_sigmas:
        print(f"\n  sigma = {sigma}")
        mpjpes = []
        bone_errors = []

        for fname in all_fnames:
            src_seq = raw_data[fname]
            src_info = parse_file_name(fname, args.dataset)

            # Pick random different-identity target
            for _ in range(100):
                tgt_fname = np.random.choice(list(raw_data.keys()))
                tgt_info = parse_file_name(tgt_fname, args.dataset)
                if tgt_info["P"] != src_info["P"]:
                    break
            tgt_seq = raw_data[tgt_fname]

            # Apply noise to source
            noisy_src = apply_noise(src_seq, sigma) if sigma > 0 else src_seq

            x1 = prepare_input(noisy_src, args.seg).to(device)
            x2 = prepare_input(tgt_seq, args.seg).to(device)

            # Also retarget clean for reference (at sigma=0)
            x1_clean = prepare_input(src_seq, args.seg).to(device)

            ret_noisy = retarget_single(tmr_model, x1, x2, device)
            ret_clean = retarget_single(tmr_model, x1_clean, x2, device)

            # Convert to (T, V, C)
            def to_tvc(t):
                return t.squeeze(-1).squeeze(0).permute(1, 2, 0).cpu().numpy()

            ret_n = to_tvc(ret_noisy)
            ret_c = to_tvc(ret_clean)

            mpjpes.append(compute_mpjpe(ret_n, ret_c))
            bone_errors.append(compute_bone_error(ret_n, NTU_BONES))

        result = {
            "sigma": sigma,
            "mpjpe_vs_clean": float(np.mean(mpjpes)),
            "mpjpe_std": float(np.std(mpjpes)),
            "bone_error": float(np.mean(bone_errors)),
        }
        noise_results.append(result)
        print(f"    MPJPE vs clean: {result['mpjpe_vs_clean']:.4f}")
        print(f"    Bone error:     {result['bone_error']:.4f}")

    # ==========================================
    # Experiment 2: Joint dropout
    # ==========================================
    drop_rates = [0.0, 0.05, 0.1, 0.2, 0.3]
    dropout_results = []

    print("\n" + "=" * 60)
    print("Experiment 2: Joint Dropout (Occlusion)")
    print("=" * 60)

    for rate in drop_rates:
        print(f"\n  drop_rate = {rate}")
        mpjpes = []
        bone_errors = []

        for fname in all_fnames:
            src_seq = raw_data[fname]
            src_info = parse_file_name(fname, args.dataset)

            for _ in range(100):
                tgt_fname = np.random.choice(list(raw_data.keys()))
                tgt_info = parse_file_name(tgt_fname, args.dataset)
                if tgt_info["P"] != src_info["P"]:
                    break
            tgt_seq = raw_data[tgt_fname]

            degraded = apply_joint_dropout(src_seq, rate) if rate > 0 else src_seq

            x1 = prepare_input(degraded, args.seg).to(device)
            x2 = prepare_input(tgt_seq, args.seg).to(device)
            x1_clean = prepare_input(src_seq, args.seg).to(device)

            ret_deg = retarget_single(tmr_model, x1, x2, device)
            ret_clean = retarget_single(tmr_model, x1_clean, x2, device)

            def to_tvc(t):
                return t.squeeze(-1).squeeze(0).permute(1, 2, 0).cpu().numpy()

            ret_d = to_tvc(ret_deg)
            ret_c = to_tvc(ret_clean)

            mpjpes.append(compute_mpjpe(ret_d, ret_c))
            bone_errors.append(compute_bone_error(ret_d, NTU_BONES))

        result = {
            "drop_rate": rate,
            "mpjpe_vs_clean": float(np.mean(mpjpes)),
            "mpjpe_std": float(np.std(mpjpes)),
            "bone_error": float(np.mean(bone_errors)),
        }
        dropout_results.append(result)
        print(f"    MPJPE vs clean: {result['mpjpe_vs_clean']:.4f}")
        print(f"    Bone error:     {result['bone_error']:.4f}")

    # ==========================================
    # Experiment 3: Temporal resolution
    # ==========================================
    temporal_frames = [64, 48, 32, 16, 8]
    temporal_results = []

    print("\n" + "=" * 60)
    print("Experiment 3: Temporal Resolution")
    print("=" * 60)

    for n_frames in temporal_frames:
        print(f"\n  frames = {n_frames}")
        mpjpes = []
        bone_errors = []

        for fname in all_fnames:
            src_seq = raw_data[fname]
            src_info = parse_file_name(fname, args.dataset)

            for _ in range(100):
                tgt_fname = np.random.choice(list(raw_data.keys()))
                tgt_info = parse_file_name(tgt_fname, args.dataset)
                if tgt_info["P"] != src_info["P"]:
                    break
            tgt_seq = raw_data[tgt_fname]

            if n_frames < 64:
                degraded = apply_temporal_downsample(src_seq, n_frames, args.seg)
            else:
                degraded = src_seq

            x1 = prepare_input(degraded, args.seg).to(device)
            x2 = prepare_input(tgt_seq, args.seg).to(device)
            x1_clean = prepare_input(src_seq, args.seg).to(device)

            ret_deg = retarget_single(tmr_model, x1, x2, device)
            ret_clean = retarget_single(tmr_model, x1_clean, x2, device)

            def to_tvc(t):
                return t.squeeze(-1).squeeze(0).permute(1, 2, 0).cpu().numpy()

            ret_d = to_tvc(ret_deg)
            ret_c = to_tvc(ret_clean)

            mpjpes.append(compute_mpjpe(ret_d, ret_c))
            bone_errors.append(compute_bone_error(ret_d, NTU_BONES))

        result = {
            "frames": n_frames,
            "mpjpe_vs_clean": float(np.mean(mpjpes)),
            "mpjpe_std": float(np.std(mpjpes)),
            "bone_error": float(np.mean(bone_errors)),
        }
        temporal_results.append(result)
        print(f"    MPJPE vs clean: {result['mpjpe_vs_clean']:.4f}")
        print(f"    Bone error:     {result['bone_error']:.4f}")

    # Save all results
    all_results = {
        "noise": noise_results,
        "dropout": dropout_results,
        "temporal": temporal_results,
        "num_samples": len(all_fnames),
    }

    json_path = os.path.join(args.output_dir, "sensitivity_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Noise plot
        ax = axes[0]
        sigmas = [r["sigma"] for r in noise_results]
        mpjpes = [r["mpjpe_vs_clean"] for r in noise_results]
        stds = [r["mpjpe_std"] for r in noise_results]
        ax.errorbar(sigmas, mpjpes, yerr=stds, marker="o", capsize=3, color="#2196F3")
        ax.set_xlabel("Noise Sigma")
        ax.set_ylabel("MPJPE vs Clean")
        ax.set_title("Gaussian Noise Sensitivity")
        ax.grid(alpha=0.3)

        # Dropout plot
        ax = axes[1]
        rates = [r["drop_rate"] for r in dropout_results]
        mpjpes = [r["mpjpe_vs_clean"] for r in dropout_results]
        stds = [r["mpjpe_std"] for r in dropout_results]
        ax.errorbar(rates, mpjpes, yerr=stds, marker="s", capsize=3, color="#FF9800")
        ax.set_xlabel("Joint Dropout Rate")
        ax.set_ylabel("MPJPE vs Clean")
        ax.set_title("Joint Occlusion Sensitivity")
        ax.grid(alpha=0.3)

        # Temporal plot
        ax = axes[2]
        frames = [r["frames"] for r in temporal_results]
        mpjpes = [r["mpjpe_vs_clean"] for r in temporal_results]
        stds = [r["mpjpe_std"] for r in temporal_results]
        ax.errorbar(frames, mpjpes, yerr=stds, marker="^", capsize=3, color="#4CAF50")
        ax.set_xlabel("Input Frames")
        ax.set_ylabel("MPJPE vs Clean")
        ax.set_title("Temporal Resolution Sensitivity")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)

        fig.suptitle("DisentangledTMR Input Sensitivity Analysis", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "sensitivity_plots.pdf"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved sensitivity_plots.pdf")

    except ImportError:
        print("Matplotlib not available, skipping plots")

    print("Done.")


if __name__ == "__main__":
    main()
