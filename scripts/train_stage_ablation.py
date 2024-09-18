#!/usr/bin/env python3
"""
Stage ablation training for Disentangled TMR.

Trains partial-stage variants of the 3-stage pipeline to measure the
contribution of each training stage.  Every variant produces a checkpoint
at ``checkpoint_stage3_best.pth`` so that the downstream retargeting and
evaluation pipeline works unchanged.

Supported ablation variants (``--stages``):
    1       Stage 1 only (encoders pretrained, decoder random)
    12      Stage 1 + Stage 2 (no end-to-end fine-tuning)
    23      Stage 2 + Stage 3 (encoders random, decoder pretrained + fine-tuned)
    13      Stage 1 + Stage 3 (skip decoder pretraining)
    123     Full pipeline (baseline, same as train_disentangled_tmr.py)

Usage:
    python scripts/train_stage_ablation.py \\
        --stages 13 \\
        --data_path data/ntu_cv_paired_comprehensive.pt \\
        --dataset ntu \\
        --output_dir output/ablation_stage13

All arguments from train_disentangled_tmr.py are accepted.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# train_disentangled_tmr_stages is imported from sibling scripts/ dir
sys.path.insert(0, str(ROOT / "scripts"))

from src.model.disentangled_tmr import create_disentangled_tmr
from src.model.simple_classifiers import ActionClassifier, IdentityClassifier
from src.training.disentanglement_losses import DisentanglementLosses
from src.training.loss import Loss
from src.losses.physical_plausibility import PhysicalPlausibilityLoss

from train_disentangled_tmr import (
    load_data,
    create_models,
    create_optimizers,
    save_checkpoint,
    parse_args as _base_parse_args,
)
from train_disentangled_tmr_stages import train_stage1, train_stage2, train_stage3


def parse_args():
    """Extend the base argument parser with --stages."""
    # Start from the same parser to inherit all arguments
    parser = argparse.ArgumentParser(
        description="Stage ablation training for Disentangled TMR",
        parents=[],
    )

    # ---------- Ablation-specific ----------
    parser.add_argument(
        "--stages",
        type=str,
        required=True,
        choices=["1", "12", "23", "13", "123"],
        help="Which stages to run: 1, 12, 23, 13, or 123 (full pipeline)",
    )

    # ---------- Data ----------
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="ntu",
                        choices=["ntu", "ntu120", "etri", "ntu_smoke", "ntu_small"])
    parser.add_argument("--num_samples", type=int, default=-1)

    # ---------- Model ----------
    parser.add_argument("--d_action", type=int, default=768)
    parser.add_argument("--d_identity", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=320)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--use_action_backbone", action="store_true", default=True)
    parser.add_argument("--no_action_backbone", action="store_false", dest="use_action_backbone")
    parser.add_argument("--no_temporal_convs", action="store_true")
    parser.add_argument("--no_lstm", action="store_true")
    parser.add_argument("--identity_mode", type=str, default="static", choices=["static", "full_seq"])
    parser.add_argument("--tokenizer", type=str, default="none", choices=["none", "pos", "dynamics"])
    parser.add_argument("--tokenizer_dim", type=int, default=256)
    parser.add_argument("--token_fusion", type=str, default="add", choices=["add", "replace"])
    parser.add_argument("--use_codebook", action="store_true")
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--codebook_dim", type=int, default=256)
    parser.add_argument("--codebook_distance", type=str, default="euclidean")
    parser.add_argument("--vq_commitment_weight", type=float, default=0.25)
    parser.add_argument("--weight_vq", type=float, default=1.0)

    # ---------- Training ----------
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage2_epochs", type=int, default=15)
    parser.add_argument("--stage3_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_classifier", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=9.689e-05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--freeze_encoders_stage3", action="store_true")
    parser.add_argument("--stage2_encoder_lr_factor", type=float, default=0.01)

    # ---------- Loss weights ----------
    parser.add_argument("--weight_ar", type=float, default=3.0)
    parser.add_argument("--weight_ri", type=float, default=1.0)
    parser.add_argument("--weight_contrastive", type=float, default=1.0)
    parser.add_argument("--weight_adversarial", type=float, default=1.0)
    parser.add_argument("--weight_orthogonality", type=float, default=1.0)
    parser.add_argument("--weight_mutual_info", type=float, default=0.01)
    parser.add_argument("--weight_bone_length", type=float, default=0.5)
    parser.add_argument("--weight_temporal_smoothness", type=float, default=0.3)
    parser.add_argument("--weight_velocity", type=float, default=0.2)
    parser.add_argument("--weight_end_effector", type=float, default=1.0)
    parser.add_argument("--weight_action_preservation", type=float, default=1.0)
    parser.add_argument("--weight_feature_consistency", type=float, default=0.5)
    parser.add_argument("--weight_motion_dynamics", type=float, default=0.1)

    # Frozen SGN
    parser.add_argument("--use_frozen_sgn", action="store_true", default=False)
    parser.add_argument("--frozen_sgn_checkpoint", type=str,
                        default="output/ntu_sgn_ar_paired/model_best.pth.tar")
    parser.add_argument("--weight_frozen_sgn", type=float, default=0.2)

    # Teacher forcing
    parser.add_argument("--stage2_teacher_forcing_start", type=float, default=1.0)
    parser.add_argument("--stage2_teacher_forcing_end", type=float, default=0.5)
    parser.add_argument("--stage3_teacher_forcing_start", type=float, default=0.5)
    parser.add_argument("--stage3_teacher_forcing_end", type=float, default=0.3)

    # Optimization
    parser.add_argument("--use_gradient_clip", action="store_true")
    parser.add_argument("--gradient_clip_value", type=float, default=1.0)
    parser.add_argument("--use_lr_scheduler", action="store_true")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # ---------- Output / logging ----------
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="disentangled-tmr")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Early stopping / downstream eval
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--downstream_eval_freq", type=int, default=5)
    parser.add_argument("--use_downstream_early_stop", action="store_true")

    # Device / seed
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _copy_final_checkpoint(output_dir: str, from_stage: int):
    """
    Copy the best checkpoint from ``from_stage`` to ``checkpoint_stage3_best.pth``
    so that the downstream pipeline can consume it without changes.
    """
    src = os.path.join(output_dir, f"checkpoint_stage{from_stage}_best.pth")
    dst = os.path.join(output_dir, "checkpoint_stage3_best.pth")
    if not os.path.exists(src):
        # Fall back to latest if best doesn't exist
        src = os.path.join(output_dir, f"checkpoint_stage{from_stage}_latest.pth")
    if os.path.exists(src) and src != dst:
        import shutil
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst} (for downstream compatibility)")


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    run_stages = set(int(c) for c in args.stages)  # e.g. "13" -> {1, 3}

    # Wandb
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"ablation_stage{''.join(args.stages)}_{args.dataset}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args),
                   dir=args.output_dir)
    else:
        print("Wandb disabled")

    torch.autograd.set_detect_anomaly(True)

    # Data
    train_loader, val_loader = load_data(args)

    # Models
    (
        model,
        ar_classifier,
        ri_classifier,
        disentangle_losses,
        recon_loss,
        physical_loss,
        physical_weights,
        frozen_sgn_loss,
    ) = create_models(args)

    print(f"\n{'='*70}")
    print(f"STAGE ABLATION: running stages {args.stages}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------ #
    # Stage 1
    # ------------------------------------------------------------------ #
    if 1 in run_stages:
        print(f"\n{'='*70}")
        print("STAGE 1: ENCODER PRETRAINING")
        print(f"{'='*70}\n")

        optimizers = create_optimizers(
            model, ar_classifier, ri_classifier, disentangle_losses, args, stage=1
        )
        train_stage1(
            model,
            ar_classifier,
            ri_classifier,
            disentangle_losses,
            train_loader,
            val_loader,
            optimizers,
            args,
        )
    else:
        print("SKIPPING Stage 1 (encoders stay randomly initialized)")

    # ------------------------------------------------------------------ #
    # Stage 2
    # ------------------------------------------------------------------ #
    if 2 in run_stages:
        print(f"\n{'='*70}")
        print("STAGE 2: DECODER TRAINING")
        print(f"{'='*70}\n")

        optimizers = create_optimizers(
            model, ar_classifier, ri_classifier, disentangle_losses, args, stage=2
        )
        train_stage2(
            model,
            recon_loss,
            physical_loss,
            physical_weights,
            frozen_sgn_loss,
            train_loader,
            val_loader,
            optimizers,
            args,
            action_encoder=model.action_encoder if 1 in run_stages else None,
            ar_classifier=ar_classifier if 1 in run_stages else None,
        )
    else:
        print("SKIPPING Stage 2 (decoder stays random or Stage-1-only)")

    # ------------------------------------------------------------------ #
    # Stage 3
    # ------------------------------------------------------------------ #
    if 3 in run_stages:
        print(f"\n{'='*70}")
        print("STAGE 3: END-TO-END FINE-TUNING")
        print(f"{'='*70}\n")

        optimizers = create_optimizers(
            model, ar_classifier, ri_classifier, disentangle_losses, args, stage=3
        )
        train_stage3(
            model,
            ar_classifier,
            disentangle_losses,
            recon_loss,
            physical_loss,
            physical_weights,
            frozen_sgn_loss,
            train_loader,
            val_loader,
            optimizers,
            args,
            ri_classifier=ri_classifier,
        )
    else:
        print("SKIPPING Stage 3 (no end-to-end fine-tuning)")

    # ------------------------------------------------------------------ #
    # Ensure a stage3-compatible checkpoint exists for downstream pipeline
    # ------------------------------------------------------------------ #
    last_stage = max(run_stages)
    if last_stage != 3:
        _copy_final_checkpoint(args.output_dir, from_stage=last_stage)

    # Save ablation metadata
    import json
    meta = {
        "stages_run": args.stages,
        "dataset": args.dataset,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "stage1_epochs": args.stage1_epochs if 1 in run_stages else 0,
        "stage2_epochs": args.stage2_epochs if 2 in run_stages else 0,
        "stage3_epochs": args.stage3_epochs if 3 in run_stages else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "ablation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nStage ablation complete (stages={args.stages})")
    print(f"Output: {args.output_dir}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
