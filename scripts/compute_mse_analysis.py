#!/usr/bin/env python3
"""
Compute MSE between:
  1. Ground truth (P2,A1) and retargeted output
  2. Action input (P1,A1) and retargeted output

Over the 10k paired samples.
"""

import argparse
import torch
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.disentangled_tmr import create_disentangled_tmr
from src.data.datasets import datasets, Cross_Data, sample_frames_fast


def load_model(checkpoint_path, device, dataset_name):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", None)
    if isinstance(ckpt_args, dict):
        import argparse as _ap
        ckpt_args = _ap.Namespace(**ckpt_args)

    d_action = getattr(ckpt_args, "d_action", 768) if ckpt_args else 768
    d_identity = getattr(ckpt_args, "d_identity", 256) if ckpt_args else 256
    d_model = getattr(ckpt_args, "d_model", 320) if ckpt_args else 320
    num_class = datasets[dataset_name]['num_class']
    tokenizer = getattr(ckpt_args, "tokenizer", None) if ckpt_args else None
    if tokenizer in ("none", "None"):
        tokenizer = None

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
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "output/disentangled_tmr_stable/checkpoint_stage3_best.pth"))
    parser.add_argument("--data_path", default=str(ROOT / "data/ntu/ntu_cv_paired_10k.pt"))
    parser.add_argument("--dataset", default="ntu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seg", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load paired data (stored as Cross_Data objects under 'train'/'test' keys)
    print(f"Loading paired data from {args.data_path}...")
    paired_data = torch.load(args.data_path, map_location='cpu', weights_only=False)
    print(f"  Keys: {list(paired_data.keys())}")
    dataset = paired_data['train']
    # Disable augmentation for deterministic results
    dataset.augment = False
    print(f"  Loaded {len(dataset)} paired samples")

    # Load model
    model = load_model(args.checkpoint, args.device, args.dataset)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    mse_gt_list = []      # MSE(output, ground_truth y2)
    mse_input_list = []   # MSE(output, action_input x1)
    n_samples = 0

    print("Computing MSE...")
    with torch.no_grad():
        for batch in tqdm(loader):
            # batch: x1, x2, y1, y2, actor_label, action_label
            x1 = batch[0].to(args.device)   # (B, C, T, V) — P1,A1 action source
            x2 = batch[1].to(args.device)   # (B, C, T, V) — P2,A2 identity ref
            y2 = batch[3].to(args.device)   # (B, C, T, V) — P2,A1 ground truth

            # Add person dim: (B, C, T, V) -> (B, C, T, V, 1)
            x1_in = x1.unsqueeze(-1)
            x2_in = x2.unsqueeze(-1)

            # Retarget: action from x1, identity from x2
            output, _, _ = model(x1_in, x2_in, target_motion=None, teacher_forcing_ratio=0.0)
            # output: (B, C, T-1, V, 1)

            # Prepend first frame from x2 (target identity structure)
            first_frame = x2_in[:, :, 0:1, :, :]
            output_full = torch.cat([first_frame, output], dim=2)  # (B, C, T, V, 1)
            output_full = output_full.squeeze(-1)  # (B, C, T, V)

            B = x1.shape[0]

            # MSE per sample: mean over (C, T, V)
            mse_gt = ((output_full - y2) ** 2).reshape(B, -1).mean(dim=1)
            mse_input = ((output_full - x1) ** 2).reshape(B, -1).mean(dim=1)

            mse_gt_list.append(mse_gt.cpu())
            mse_input_list.append(mse_input.cpu())
            n_samples += B

    mse_gt_all = torch.cat(mse_gt_list)
    mse_input_all = torch.cat(mse_input_list)

    print(f"\n{'='*60}")
    print(f"MSE Analysis over {n_samples} samples")
    print(f"{'='*60}")
    print(f"MSE(retargeted, ground_truth):  {mse_gt_all.mean().item():.6f}  (std: {mse_gt_all.std().item():.6f})")
    print(f"MSE(retargeted, action_input):  {mse_input_all.mean().item():.6f}  (std: {mse_input_all.std().item():.6f})")
    print(f"{'='*60}")
    print(f"\nInterpretation:")
    print(f"  - GT MSE measures reconstruction quality (lower = better action transfer)")
    print(f"  - Input MSE measures identity change (higher = more anonymization)")
    print(f"  - Ratio (input/GT): {mse_input_all.mean().item() / mse_gt_all.mean().item():.2f}")

    # Save results
    import json
    results = {
        "n_samples": n_samples,
        "mse_retargeted_vs_groundtruth": {
            "mean": float(mse_gt_all.mean()),
            "std": float(mse_gt_all.std()),
            "median": float(mse_gt_all.median()),
        },
        "mse_retargeted_vs_input": {
            "mean": float(mse_input_all.mean()),
            "std": float(mse_input_all.std()),
            "median": float(mse_input_all.median()),
        },
    }
    out_path = ROOT / "output" / "mse_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
