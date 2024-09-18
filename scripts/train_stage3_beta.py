#!/usr/bin/env python3
"""
Retrain TMR Stage 3 with beta residual blending.

Loads model weights from an existing checkpoint (Stage 3),
resets training from epoch 0 with the given --beta value.
"""
import argparse
import os
import sys
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from scripts.train_disentangled_tmr import (
    parse_args, set_seed, load_data, create_models, create_optimizers
)
from scripts.train_disentangled_tmr_stages import train_stage3


def main():
    args = parse_args()
    set_seed(args.seed)

    train_loader, val_loader = load_data(args)

    (model, ar_classifier, ri_classifier, disentangle_losses,
     recon_loss, physical_loss, physical_weights, frozen_sgn_loss) = create_models(args)

    # Load pretrained weights from checkpoint
    checkpoint_path = args.resume
    if checkpoint_path is None:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint_stage3_best.pth')

    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'ar_classifier_state_dict' in ckpt:
            ar_classifier.load_state_dict(ckpt['ar_classifier_state_dict'], strict=False)
        print(f"Loaded weights from stage {ckpt.get('stage')}, epoch {ckpt.get('epoch')}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}, starting from scratch")

    print(f"\n=== Training Stage 3 with beta={args.beta} ===\n")

    optimizers = create_optimizers(model, ar_classifier, ri_classifier, args, stage=3)

    args_stage3 = argparse.Namespace(**vars(args))
    args_stage3.stage3_epochs = args.stage3_epochs

    train_stage3(
        model, ar_classifier, disentangle_losses, recon_loss,
        physical_loss, physical_weights, frozen_sgn_loss,
        train_loader, val_loader, optimizers, args_stage3, None, None,
        ri_classifier=ri_classifier,
    )


if __name__ == '__main__':
    main()
