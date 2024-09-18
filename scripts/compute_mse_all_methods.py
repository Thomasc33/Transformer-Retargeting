#!/usr/bin/env python3
"""
Compute MSE(retargeted, ground_truth) and MSE(retargeted, action_input)
for all methods: DisentangledTMR, DMR, PMR, Noise.

Uses the same 10k paired quadruplets for a fair comparison.
All MSEs computed in (C, T=64, V=25) space.
"""

import argparse
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import datasets, Cross_Data, sample_frames_fast
from src.model.disentangled_tmr import create_disentangled_tmr


def load_ours(checkpoint_path, device, dataset_name):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", None)
    if isinstance(ckpt_args, dict):
        ckpt_args = argparse.Namespace(**ckpt_args)

    d_action = getattr(ckpt_args, "d_action", 768) if ckpt_args else 768
    d_identity = getattr(ckpt_args, "d_identity", 256) if ckpt_args else 256
    d_model = getattr(ckpt_args, "d_model", 320) if ckpt_args else 320
    num_class = datasets[dataset_name]['num_class']
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
        identity_use_full_sequence=(getattr(ckpt_args, "identity_mode", "static") == "full_seq") if ckpt_args else False,
        tokenizer_type=tokenizer,
    )
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_dmr_pmr(model_type, checkpoint_path, device):
    if model_type == 'dmr':
        from eval.dmr.dmr import DMR
        model = DMR(use_adv=False)
    else:
        from eval.pmr.pmr import AutoEncoder
        model = AutoEncoder(use_adv=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    core_prefixes = ('static_encoder.', 'dynamic_encoder.', 'decoder.')
    filtered = {k: v for k, v in state_dict.items()
                if any(k.startswith(p) for p in core_prefixes)}
    if filtered:
        state_dict = filtered

    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def resample_temporal(tensor, src_T, dst_T):
    """Resample (B, C, T, V) from src_T to dst_T frames."""
    if src_T == dst_T:
        return tensor
    # (B, C, T, V) -> (B, C*V, T) -> interpolate -> (B, C*V, dst_T) -> (B, C, dst_T, V)
    B, C, T, V = tensor.shape
    x = tensor.permute(0, 1, 3, 2).reshape(B, C * V, T)
    x = F.interpolate(x, size=dst_T, mode='linear', align_corners=True)
    return x.reshape(B, C, V, dst_T).permute(0, 1, 3, 2)


def compute_mse_ours(model, loader, device):
    """Compute MSE for DisentangledTMR."""
    mse_gt_list, mse_input_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ours"):
            x1 = batch[0].to(device)  # (B, C, T, V) — P1,A1
            x2 = batch[1].to(device)  # (B, C, T, V) — P2,A2
            y2 = batch[3].to(device)  # (B, C, T, V) — P2,A1

            x1_in = x1.unsqueeze(-1)  # (B, C, T, V, 1)
            x2_in = x2.unsqueeze(-1)
            output, _, _ = model(x1_in, x2_in, target_motion=None, teacher_forcing_ratio=0.0)
            first_frame = x2_in[:, :, 0:1, :, :]
            output_full = torch.cat([first_frame, output], dim=2).squeeze(-1)  # (B, C, T, V)

            B = x1.shape[0]
            mse_gt = ((output_full - y2) ** 2).reshape(B, -1).mean(dim=1)
            mse_input = ((output_full - x1) ** 2).reshape(B, -1).mean(dim=1)
            mse_gt_list.append(mse_gt.cpu())
            mse_input_list.append(mse_input.cpu())
    return torch.cat(mse_gt_list), torch.cat(mse_input_list)


def compute_mse_dmr_pmr(model, dataset, device, batch_size=64):
    """Compute MSE for DMR or PMR. They need T=75 input in (B, T, 25, 3) format."""
    MODEL_T = 75
    SEG = 64  # our standard frame count

    mse_gt_list, mse_input_list = [], []

    # Process in batches manually since we need custom data loading
    n = len(dataset)
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="DMR/PMR"):
            end = min(start + batch_size, n)

            x1_model_batch, x2_model_batch, y2_batch, x1_ref_batch = [], [], [], []
            for i in range(start, end):
                sample = dataset.sampled_data[i]
                # T=75 for model input
                x1_raw_75 = sample_frames_fast(dataset.X[sample[0][2]], MODEL_T)
                x2_raw_75 = sample_frames_fast(dataset.X[sample[3][2]], MODEL_T)
                # T=64 for ground truth comparison
                y2_raw_64 = sample_frames_fast(dataset.X[sample[2][2]], SEG)
                x1_raw_64 = sample_frames_fast(dataset.X[sample[0][2]], SEG)

                x1_model_batch.append(torch.from_numpy(x1_raw_75.reshape(MODEL_T, 25, 3)).float())
                x2_model_batch.append(torch.from_numpy(x2_raw_75.reshape(MODEL_T, 25, 3)).float())
                y2_batch.append(torch.from_numpy(y2_raw_64).float().reshape(SEG, 25, 3).permute(2, 0, 1))
                x1_ref_batch.append(torch.from_numpy(x1_raw_64).float().reshape(SEG, 25, 3).permute(2, 0, 1))

            x1_m = torch.stack(x1_model_batch).to(device)  # (B, 75, 25, 3)
            x2_m = torch.stack(x2_model_batch).to(device)
            y2_t = torch.stack(y2_batch).to(device)         # (B, C, 64, V)
            x1_ref = torch.stack(x1_ref_batch).to(device)

            # Forward: action from x1, identity from x2
            output_75 = model(x1_m, x2_m)  # (B, 75, 75)

            # Reshape output to (B, C, T=75, V=25)
            B_cur = output_75.shape[0]
            output_ctv = output_75.reshape(B_cur, MODEL_T, 25, 3).permute(0, 3, 1, 2)  # (B, C, 75, V)

            # Resample from T=75 to T=64
            output_64 = resample_temporal(output_ctv, MODEL_T, SEG)  # (B, C, 64, V)

            # MSE
            mse_gt = ((output_64 - y2_t) ** 2).reshape(B_cur, -1).mean(dim=1)
            mse_input = ((output_64 - x1_ref) ** 2).reshape(B_cur, -1).mean(dim=1)
            mse_gt_list.append(mse_gt.cpu())
            mse_input_list.append(mse_input.cpu())

    return torch.cat(mse_gt_list), torch.cat(mse_input_list)


def compute_mse_noise(dataset, sigma=0.05):
    """Compute MSE for Gaussian noise baseline (no model needed)."""
    SEG = 64
    mse_gt_list, mse_input_list = [], []

    torch.manual_seed(42)
    n = len(dataset)
    for i in tqdm(range(n), desc="Noise"):
        sample = dataset.sampled_data[i]
        x1_raw = sample_frames_fast(dataset.X[sample[0][2]], SEG)
        y2_raw = sample_frames_fast(dataset.X[sample[2][2]], SEG)

        x1 = torch.from_numpy(x1_raw).float().reshape(SEG, 25, 3).permute(2, 0, 1)  # (C, T, V)
        y2 = torch.from_numpy(y2_raw).float().reshape(SEG, 25, 3).permute(2, 0, 1)

        noise = torch.randn_like(x1) * sigma
        noisy = x1 + noise

        mse_gt = ((noisy - y2) ** 2).mean().item()
        mse_input = ((noisy - x1) ** 2).mean().item()
        mse_gt_list.append(mse_gt)
        mse_input_list.append(mse_input)

    return torch.tensor(mse_gt_list), torch.tensor(mse_input_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(ROOT / "data/ntu/ntu_cv_paired_10k.pt"))
    parser.add_argument("--dataset", default="ntu")
    parser.add_argument("--ours_checkpoint", default=str(ROOT / "output/disentangled_tmr_stable/checkpoint_stage3_best.pth"))
    parser.add_argument("--dmr_checkpoint", default=str(ROOT / "data/models/trained_models/dmr_ntu_cv_best.pth"))
    parser.add_argument("--pmr_checkpoint", default=str(ROOT / "data/models/trained_models/pmr_ntu_cv_best.pth"))
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load paired data
    print(f"Loading paired data from {args.data_path}...")
    paired_data = torch.load(args.data_path, map_location='cpu', weights_only=False)
    dataset = paired_data['train']
    dataset.augment = False
    print(f"  {len(dataset)} paired samples")

    results = {}

    # --- DisentangledTMR ---
    print("\n=== DisentangledTMR (Ours) ===")
    model_ours = load_ours(args.ours_checkpoint, args.device, args.dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    gt, inp = compute_mse_ours(model_ours, loader, args.device)
    results['ours'] = {
        'mse_gt': {'mean': float(gt.mean()), 'std': float(gt.std()), 'median': float(gt.median())},
        'mse_input': {'mean': float(inp.mean()), 'std': float(inp.std()), 'median': float(inp.median())},
    }
    print(f"  GT MSE: {gt.mean():.6f} | Input MSE: {inp.mean():.6f}")
    del model_ours
    torch.cuda.empty_cache()

    # --- DMR ---
    print("\n=== DMR ===")
    model_dmr = load_dmr_pmr('dmr', args.dmr_checkpoint, args.device)
    gt, inp = compute_mse_dmr_pmr(model_dmr, dataset, args.device, args.batch_size)
    results['dmr'] = {
        'mse_gt': {'mean': float(gt.mean()), 'std': float(gt.std()), 'median': float(gt.median())},
        'mse_input': {'mean': float(inp.mean()), 'std': float(inp.std()), 'median': float(inp.median())},
    }
    print(f"  GT MSE: {gt.mean():.6f} | Input MSE: {inp.mean():.6f}")
    del model_dmr
    torch.cuda.empty_cache()

    # --- PMR ---
    print("\n=== PMR ===")
    model_pmr = load_dmr_pmr('pmr', args.pmr_checkpoint, args.device)
    gt, inp = compute_mse_dmr_pmr(model_pmr, dataset, args.device, args.batch_size)
    results['pmr'] = {
        'mse_gt': {'mean': float(gt.mean()), 'std': float(gt.std()), 'median': float(gt.median())},
        'mse_input': {'mean': float(inp.mean()), 'std': float(inp.std()), 'median': float(inp.median())},
    }
    print(f"  GT MSE: {gt.mean():.6f} | Input MSE: {inp.mean():.6f}")
    del model_pmr
    torch.cuda.empty_cache()

    # --- Noise ---
    print("\n=== Noise (σ={}) ===".format(args.noise_sigma))
    gt, inp = compute_mse_noise(dataset, args.noise_sigma)
    results['noise'] = {
        'sigma': args.noise_sigma,
        'mse_gt': {'mean': float(gt.mean()), 'std': float(gt.std()), 'median': float(gt.median())},
        'mse_input': {'mean': float(inp.mean()), 'std': float(inp.std()), 'median': float(inp.median())},
    }
    print(f"  GT MSE: {gt.mean():.6f} | Input MSE: {inp.mean():.6f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'GT MSE':>10} {'Input MSE':>12} {'Ratio (I/G)':>12}")
    print("=" * 70)
    for name in ['noise', 'dmr', 'pmr', 'ours']:
        r = results[name]
        g = r['mse_gt']['mean']
        i = r['mse_input']['mean']
        ratio = i / g if g > 0 else float('inf')
        label = {'noise': f'Noise (σ={args.noise_sigma})', 'dmr': 'DMR', 'pmr': 'PMR', 'ours': 'DisentangledTMR'}[name]
        print(f"{label:<20} {g:>10.6f} {i:>12.6f} {ratio:>12.2f}")
    print("=" * 70)

    out_path = ROOT / "output" / "mse_all_methods.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
