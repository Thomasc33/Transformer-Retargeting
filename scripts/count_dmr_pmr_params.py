#!/usr/bin/env python3
"""
Count parameters for DMR and PMR models.

Lightweight script safe for login node -- no GPU, no data loading, no checkpoints.
Just instantiates model architectures and counts parameters.

Results:
  DMR: 4.94M params (2x 2.20M encoders + 0.55M decoder)
  PMR: 0.99M params (2x 0.32M encoders + 0.35M decoder)
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

# Import model classes
from eval.dmr.dmr import DMR, DMR_Encoder2D, DMR_Decoder1D
from eval.pmr.pmr import AutoEncoder as PMR, Encoder2D as PMR_Encoder2D, Decoder1D as PMR_Decoder1D


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_module_params(module):
    return sum(p.numel() for p in module.parameters())


def layer_detail(model, prefix=""):
    """Print per-layer parameter details."""
    for name, param in model.named_parameters():
        print(f"  {prefix}{name}: {list(param.shape)} = {param.numel():,}")


def main():
    print("=" * 60)
    print("DMR (Disentangled Motion Retargeting)")
    print("  encoded_channels = (256, 32)")
    print("=" * 60)

    dmr = DMR(use_adv=False)
    total, trainable = count_params(dmr)
    print(f"\n  Total params:     {total:>12,}  ({total / 1e6:.2f}M)")
    print(f"  Trainable params: {trainable:>12,}  ({trainable / 1e6:.2f}M)")

    # Component breakdown
    se_params = count_module_params(dmr.static_encoder)
    de_params = count_module_params(dmr.dynamic_encoder)
    dec_params = count_module_params(dmr.decoder)
    print(f"\n  Components:")
    print(f"    static_encoder (action):   {se_params:>10,}  ({se_params / 1e6:.4f}M)")
    print(f"    dynamic_encoder (identity): {de_params:>10,}  ({de_params / 1e6:.4f}M)")
    print(f"    decoder:                    {dec_params:>10,}  ({dec_params / 1e6:.4f}M)")
    print(f"    sum of components:          {se_params + de_params + dec_params:>10,}")

    print(f"\n  Per-layer detail (static_encoder):")
    layer_detail(dmr.static_encoder, "    ")

    print(f"\n  Per-layer detail (decoder):")
    layer_detail(dmr.decoder, "    ")

    print("\n")
    print("=" * 60)
    print("PMR (Privacy-preserving Motion Retargeting)")
    print("  encoded_channels = (128, 16)")
    print("=" * 60)

    pmr = PMR(use_adv=False)
    total_pmr, trainable_pmr = count_params(pmr)
    print(f"\n  Total params:     {total_pmr:>12,}  ({total_pmr / 1e6:.2f}M)")
    print(f"  Trainable params: {trainable_pmr:>12,}  ({trainable_pmr / 1e6:.2f}M)")

    # Component breakdown
    se_params_pmr = count_module_params(pmr.static_encoder)
    de_params_pmr = count_module_params(pmr.dynamic_encoder)
    dec_params_pmr = count_module_params(pmr.decoder)
    print(f"\n  Components:")
    print(f"    static_encoder (action):   {se_params_pmr:>10,}  ({se_params_pmr / 1e6:.4f}M)")
    print(f"    dynamic_encoder (identity): {de_params_pmr:>10,}  ({de_params_pmr / 1e6:.4f}M)")
    print(f"    decoder:                    {dec_params_pmr:>10,}  ({dec_params_pmr / 1e6:.4f}M)")
    print(f"    sum of components:          {se_params_pmr + de_params_pmr + dec_params_pmr:>10,}")

    print(f"\n  Per-layer detail (static_encoder):")
    layer_detail(pmr.static_encoder, "    ")

    print(f"\n  Per-layer detail (decoder):")
    layer_detail(pmr.decoder, "    ")

    # Summary comparison
    print("\n\n")
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':25s} {'Params':>12s} {'Params (M)':>12s}")
    print("-" * 52)
    print(f"{'DMR':25s} {total:>12,} {total / 1e6:>12.2f}")
    print(f"{'PMR':25s} {total_pmr:>12,} {total_pmr / 1e6:>12.2f}")
    print(f"{'DisentangledTMR (known)':25s} {'22,509,575':>12s} {'22.51':>12s}")
    print(f"{'SGN (known)':25s} {'688,221':>12s} {'0.69':>12s}")
    print(f"{'MixFormer (known)':25s} {'2,057,517':>12s} {'2.06':>12s}")

    # Save results to JSON for reference
    results = {
        "DMR": {
            "params": {
                "total": total,
                "total_M": round(total / 1e6, 2),
                "trainable": trainable,
                "trainable_M": round(trainable / 1e6, 2),
            },
            "components": {
                "static_encoder": {"params": se_params, "params_M": round(se_params / 1e6, 2)},
                "dynamic_encoder": {"params": de_params, "params_M": round(de_params / 1e6, 2)},
                "decoder": {"params": dec_params, "params_M": round(dec_params / 1e6, 2)},
            },
            "architecture": "2x Conv2D encoders (256,32) + ConvTranspose1D decoder",
            "training_epochs": "225 (stages 1,2,5,6: 5+20+100+100)",
        },
        "PMR": {
            "params": {
                "total": total_pmr,
                "total_M": round(total_pmr / 1e6, 2),
                "trainable": trainable_pmr,
                "trainable_M": round(trainable_pmr / 1e6, 2),
            },
            "components": {
                "static_encoder": {"params": se_params_pmr, "params_M": round(se_params_pmr / 1e6, 2)},
                "dynamic_encoder": {"params": de_params_pmr, "params_M": round(de_params_pmr / 1e6, 2)},
                "decoder": {"params": dec_params_pmr, "params_M": round(dec_params_pmr / 1e6, 2)},
            },
            "architecture": "2x Conv2D encoders (128,16) + ConvTranspose1D decoder + adversarial heads",
            "training_epochs": "295 (stages 1-6: 5+20+20+50+100+100)",
        },
    }

    out_path = os.path.join(ROOT, "output", "analysis", "computational_cost", "dmr_pmr_params.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    main()
