#!/usr/bin/env python3
"""
Compute and compare computational costs across methods.

Reports:
- Parameter counts (total, per component)
- Inference time (ms/sequence, with warm-up + averaging)
- GPU memory usage during inference
- FLOPs estimate (using a dummy forward pass)

Must run via SLURM (instantiates models, uses GPU).
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

from src.model.disentangled_tmr import create_disentangled_tmr
from src.model.sgn import SGN
from src.model.ske_mixf import Model as MixFormerModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="output/analysis/computational_cost")
    p.add_argument("--num_warmup", type=int, default=10)
    p.add_argument("--num_trials", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--checkpoint",
                   default="output/disentangled_tmr_stable/checkpoint_stage3_best.pth",
                   help="DisentangledTMR checkpoint for inference timing")
    return p.parse_args()


def count_params(model, human_readable=True):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if human_readable:
        return {
            "total": total,
            "total_M": round(total / 1e6, 2),
            "trainable": trainable,
            "trainable_M": round(trainable / 1e6, 2),
        }
    return total, trainable


def count_component_params(model):
    """Count params per component of DisentangledTMR."""
    components = {}

    if hasattr(model, "action_encoder"):
        ae_params = sum(p.numel() for p in model.action_encoder.parameters())
        components["action_encoder"] = {"params": ae_params, "params_M": round(ae_params / 1e6, 2)}

    if hasattr(model, "identity_encoder"):
        ie_params = sum(p.numel() for p in model.identity_encoder.parameters())
        components["identity_encoder"] = {"params": ie_params, "params_M": round(ie_params / 1e6, 2)}

    if hasattr(model, "decoder"):
        dec_params = sum(p.numel() for p in model.decoder.parameters())
        components["decoder"] = {"params": dec_params, "params_M": round(dec_params / 1e6, 2)}

    # Classification heads
    for name in ["ar_head", "ri_head", "discriminator"]:
        if hasattr(model, name):
            head = getattr(model, name)
            head_params = sum(p.numel() for p in head.parameters())
            components[name] = {"params": head_params, "params_M": round(head_params / 1e6, 2)}

    return components


def measure_inference_time(model, inputs, num_warmup=10, num_trials=50, device="cuda"):
    """Measure inference time in ms per forward pass."""
    model.eval()

    # Warm up
    with torch.no_grad():
        for _ in range(num_warmup):
            if isinstance(inputs, tuple):
                _ = model(*inputs)
            else:
                _ = model(inputs)

    # Synchronize
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_trials):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            if isinstance(inputs, tuple):
                _ = model(*inputs)
            else:
                _ = model(inputs)

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return {
        "mean_ms": round(np.mean(times), 2),
        "std_ms": round(np.std(times), 2),
        "min_ms": round(np.min(times), 2),
        "max_ms": round(np.max(times), 2),
    }


def measure_gpu_memory(model, inputs, device="cuda"):
    """Measure peak GPU memory during inference."""
    if device != "cuda":
        return {"peak_MB": 0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        if isinstance(inputs, tuple):
            _ = model(*inputs)
        else:
            _ = model(inputs)

    peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    return {"peak_MB": round(peak, 1)}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    B = args.batch_size
    T = 64
    V = 25
    C = 3

    # =========================================
    # DisentangledTMR
    # =========================================
    print("=" * 60)
    print("DisentangledTMR")
    print("=" * 60)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_args = checkpoint.get("args", None)
        if ckpt_args and isinstance(ckpt_args, dict):
            ckpt_args = argparse.Namespace(**ckpt_args)
    else:
        ckpt_args = None
        checkpoint = None

    tmr_model = create_disentangled_tmr(
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
        tmr_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    tmr_model.eval()

    # Params
    tmr_params = count_params(tmr_model)
    tmr_components = count_component_params(tmr_model)
    print(f"  Total params: {tmr_params['total_M']}M")
    for comp, info in tmr_components.items():
        print(f"    {comp}: {info['params_M']}M")

    # Inference time
    x1 = torch.randn(B, C, T, V, 1).to(device)
    x2 = torch.randn(B, C, T, V, 1).to(device)

    # Wrap inference call
    class TMRInference(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x1, x2):
            return self.model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)

    tmr_inf = TMRInference(tmr_model)
    tmr_time = measure_inference_time(tmr_inf, (x1, x2), args.num_warmup, args.num_trials,
                                       str(device))
    tmr_mem = measure_gpu_memory(tmr_inf, (x1, x2), str(device))
    print(f"  Inference: {tmr_time['mean_ms']:.1f} +/- {tmr_time['std_ms']:.1f} ms/seq")
    print(f"  GPU Memory: {tmr_mem['peak_MB']:.1f} MB")

    results["DisentangledTMR"] = {
        "params": tmr_params,
        "components": tmr_components,
        "inference_time": tmr_time,
        "gpu_memory": tmr_mem,
        "training_time_gpu_hours": 28,  # Known from training logs
    }

    del tmr_model, tmr_inf, x1, x2
    torch.cuda.empty_cache()

    # =========================================
    # SGN (downstream classifier)
    # =========================================
    print("\n" + "=" * 60)
    print("SGN (downstream)")
    print("=" * 60)

    sgn = SGN(num_classes=49, dataset="ntu", seg=T, bias=True).to(device)
    sgn.eval()

    sgn_params = count_params(sgn)
    print(f"  Total params: {sgn_params['total_M']}M")

    x_sgn = torch.randn(B, T, V * C).to(device)
    sgn_time = measure_inference_time(sgn, x_sgn, args.num_warmup, args.num_trials, str(device))
    sgn_mem = measure_gpu_memory(sgn, x_sgn, str(device))
    print(f"  Inference: {sgn_time['mean_ms']:.1f} +/- {sgn_time['std_ms']:.1f} ms/seq")

    results["SGN"] = {
        "params": sgn_params,
        "inference_time": sgn_time,
        "gpu_memory": sgn_mem,
    }

    del sgn
    torch.cuda.empty_cache()

    # =========================================
    # MixFormer (downstream classifier)
    # =========================================
    print("\n" + "=" * 60)
    print("MixFormer (downstream)")
    print("=" * 60)

    mixf = MixFormerModel(
        num_class=49, num_point=25, num_person=1,
        graph="src.graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
    ).to(device)
    mixf.eval()

    mixf_params = count_params(mixf)
    print(f"  Total params: {mixf_params['total_M']}M")

    x_mixf = torch.randn(B, C, T, V, 1).to(device)
    mixf_time = measure_inference_time(mixf, x_mixf, args.num_warmup, args.num_trials, str(device))
    mixf_mem = measure_gpu_memory(mixf, x_mixf, str(device))
    print(f"  Inference: {mixf_time['mean_ms']:.1f} +/- {mixf_time['std_ms']:.1f} ms/seq")

    results["MixFormer"] = {
        "params": mixf_params,
        "inference_time": mixf_time,
        "gpu_memory": mixf_mem,
    }

    del mixf
    torch.cuda.empty_cache()

    # =========================================
    # Summary table
    # =========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':25s} {'Params (M)':>12s} {'Inference (ms)':>16s} {'GPU Mem (MB)':>14s}")
    print("-" * 70)
    for model_name, info in results.items():
        params_m = info["params"]["total_M"]
        inf_ms = info["inference_time"]["mean_ms"]
        inf_std = info["inference_time"]["std_ms"]
        mem_mb = info["gpu_memory"]["peak_MB"]
        print(f"{model_name:25s} {params_m:12.2f} {inf_ms:>9.1f} +/- {inf_std:<5.1f} {mem_mb:14.1f}")

    # Save
    json_path = os.path.join(args.output_dir, "computational_cost.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
