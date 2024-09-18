#!/usr/bin/env python3
"""
Measure inference time for DMR and PMR models on GPU.

Must run via SLURM (requires GPU).

Reports per-sequence inference time (ms) at batch_size=1, with warm-up.
Also re-measures DisentangledTMR for consistent comparison on the same hardware.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.dmr.dmr import DMR
from eval.pmr.pmr import AutoEncoder as PMR
from src.model.disentangled_tmr import create_disentangled_tmr


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "total_M": round(total / 1e6, 2)}


def measure_inference_time(model, inputs, num_warmup=20, num_trials=100, device="cuda"):
    """Measure inference time in ms per forward pass."""
    model.eval()

    with torch.no_grad():
        for _ in range(num_warmup):
            if isinstance(inputs, tuple):
                model(*inputs)
            else:
                model(inputs)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_trials):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            if isinstance(inputs, tuple):
                model(*inputs)
            else:
                model(inputs)

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "mean_ms": round(np.mean(times), 2),
        "std_ms": round(np.std(times), 2),
        "min_ms": round(np.min(times), 2),
        "max_ms": round(np.max(times), 2),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    output_dir = ROOT / "output" / "analysis" / "computational_cost"
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # =========================================
    # DMR - input: (B, T=75, 25, 3)
    # =========================================
    print("\n" + "=" * 60)
    print("DMR (Disentangled Motion Retargeting)")
    print("=" * 60)

    dmr = DMR(use_adv=False).to(device).eval()
    dmr_params = count_params(dmr)
    print(f"  Params: {dmr_params['total_M']}M")

    x_dmr_action = torch.randn(1, 75, 25, 3).to(device)
    x_dmr_identity = torch.randn(1, 75, 25, 3).to(device)

    dmr_time = measure_inference_time(dmr, (x_dmr_action, x_dmr_identity), device=str(device))
    print(f"  Inference: {dmr_time['mean_ms']:.2f} +/- {dmr_time['std_ms']:.2f} ms/seq")

    results["DMR"] = {"params": dmr_params, "inference_time": dmr_time}

    del dmr, x_dmr_action, x_dmr_identity
    torch.cuda.empty_cache()

    # =========================================
    # PMR - input: (B, T=75, 25, 3)
    # =========================================
    print("\n" + "=" * 60)
    print("PMR (Privacy-preserving Motion Retargeting)")
    print("=" * 60)

    pmr = PMR(use_adv=False).to(device).eval()
    pmr_params = count_params(pmr)
    print(f"  Params: {pmr_params['total_M']}M")

    x_pmr_action = torch.randn(1, 75, 25, 3).to(device)
    x_pmr_identity = torch.randn(1, 75, 25, 3).to(device)

    pmr_time = measure_inference_time(pmr, (x_pmr_action, x_pmr_identity), device=str(device))
    print(f"  Inference: {pmr_time['mean_ms']:.2f} +/- {pmr_time['std_ms']:.2f} ms/seq")

    results["PMR"] = {"params": pmr_params, "inference_time": pmr_time}

    del pmr, x_pmr_action, x_pmr_identity
    torch.cuda.empty_cache()

    # =========================================
    # DisentangledTMR - for consistent comparison
    # =========================================
    print("\n" + "=" * 60)
    print("DisentangledTMR (Ours)")
    print("=" * 60)

    ckpt_path = ROOT / "output" / "disentangled_tmr_stable" / "checkpoint_stage3_best.pth"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_args = checkpoint.get("args", None)
        if ckpt_args and isinstance(ckpt_args, dict):
            ckpt_args = argparse.Namespace(**ckpt_args)
    else:
        ckpt_args = None
        checkpoint = None

    tmr = create_disentangled_tmr(
        dataset="ntu", num_class=60, device=device,
        d_action=getattr(ckpt_args, "d_action", 768) if ckpt_args else 768,
        d_identity=getattr(ckpt_args, "d_identity", 256) if ckpt_args else 256,
        d_model=getattr(ckpt_args, "d_model", 320) if ckpt_args else 320,
        num_decoder_layers=getattr(ckpt_args, "num_decoder_layers", 6) if ckpt_args else 6,
        use_pretrained_action=getattr(ckpt_args, "use_action_backbone", True) if ckpt_args else True,
        use_temporal_convs=not getattr(ckpt_args, "no_temporal_convs", False) if ckpt_args else True,
        use_lstm=not getattr(ckpt_args, "no_lstm", False) if ckpt_args else True,
        identity_use_full_sequence=False,
        tokenizer_type=None,
    )
    if checkpoint and "model_state_dict" in checkpoint:
        tmr.load_state_dict(checkpoint["model_state_dict"], strict=False)

    tmr_params = count_params(tmr)
    print(f"  Params: {tmr_params['total_M']}M")

    x1 = torch.randn(1, 3, 64, 25, 1).to(device)
    x2 = torch.randn(1, 3, 64, 25, 1).to(device)

    class TMRInference(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x1, x2):
            return self.model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)

    tmr_inf = TMRInference(tmr).eval()
    tmr_time = measure_inference_time(tmr_inf, (x1, x2), device=str(device))
    print(f"  Inference: {tmr_time['mean_ms']:.2f} +/- {tmr_time['std_ms']:.2f} ms/seq")

    results["DisentangledTMR"] = {"params": tmr_params, "inference_time": tmr_time}

    del tmr, tmr_inf
    torch.cuda.empty_cache()

    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':25s} {'Params (M)':>12s} {'Inference (ms)':>16s}")
    print("-" * 55)
    for name, info in results.items():
        pm = info["params"]["total_M"]
        ms = info["inference_time"]["mean_ms"]
        std = info["inference_time"]["std_ms"]
        print(f"{name:25s} {pm:12.2f} {ms:>9.2f} +/- {std:<5.2f}")

    # Speedup comparison
    ours_ms = results["DisentangledTMR"]["inference_time"]["mean_ms"]
    for name in ["DMR", "PMR"]:
        other_ms = results[name]["inference_time"]["mean_ms"]
        ratio = ours_ms / other_ms
        print(f"\n  DisentangledTMR / {name} = {ratio:.1f}x slower")

    # Save
    json_path = output_dir / "dmr_pmr_inference_timing.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()
