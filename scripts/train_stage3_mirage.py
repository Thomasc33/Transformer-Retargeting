#!/usr/bin/env python3
"""
Retrain TMR Stage 3 with MIRAGE-inspired losses.

Loads model weights from an existing checkpoint (Stage 2 or Stage 3),
resets Stage 3 training from epoch 0 with MIRAGE losses enabled.
"""
import argparse
import os
import sys
import torch

# Add project root to path
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from scripts.train_disentangled_tmr import (
    parse_args, set_seed, load_data, create_models, create_optimizers
)
from scripts.train_disentangled_tmr_stages import train_stage3


def main():
    args = parse_args()
    set_seed(args.seed)

    # Force MIRAGE losses on
    args.use_mirage_losses = True

    # Load data
    train_loader, val_loader = load_data(args)

    # Create models
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

    # Initialize MIRAGE losses
    from src.losses.mirage_inspired import MirageInspiredLosses

    dataset_configs = {
        'ntu': (49, 40), 'ntu_smoke': (49, 40), 'ntu_small': (49, 40),
        'ntu120': (94, 106), 'etri': (55, 100),
    }
    num_classes, num_identities = dataset_configs.get(args.dataset, (49, 40))

    # Load raw data for coordinate standardization if needed
    raw_data_dict = None
    lambda_motion_disc = getattr(args, 'lambda_motion_disc', 0.0)
    lambda_coord_std = getattr(args, 'lambda_coord_std', 0.0)
    if lambda_coord_std > 0:
        import pickle
        raw_stats_path = getattr(args, 'raw_stats_path', None)
        if raw_stats_path is None:
            from src.data.datasets import datasets as ds_configs
            raw_stats_path = ds_configs[args.dataset]['path']
        print(f"Loading raw data from {raw_stats_path} for coordinate standardization...")
        with open(raw_stats_path, 'rb') as f:
            raw_data_dict = pickle.load(f)
        print(f"  {len(raw_data_dict)} raw samples loaded for stats")

    mirage_losses = MirageInspiredLosses(
        num_classes=num_classes,
        num_identities=num_identities,
        device=args.device,
        lambda_dist_disc=args.lambda_dist_disc,
        lambda_output_act=args.lambda_output_act,
        lambda_output_id=args.lambda_output_id,
        lambda_output_contrastive=args.lambda_output_contrastive,
        lambda_ee_enhanced=args.lambda_ee_enhanced,
        lambda_motion_disc=lambda_motion_disc,
        lambda_coord_std=lambda_coord_std,
        raw_data_dict=raw_data_dict,
    )
    print(f"MIRAGE losses initialized: dist_disc={args.lambda_dist_disc}, "
          f"output_act={args.lambda_output_act}, output_id={args.lambda_output_id}, "
          f"contrastive={args.lambda_output_contrastive}, ee_enhanced={args.lambda_ee_enhanced}, "
          f"motion_disc={lambda_motion_disc}, coord_std={lambda_coord_std}")

    # Create optimizers for Stage 3
    optimizers = create_optimizers(model, ar_classifier, ri_classifier,
                                  disentangle_losses, args, stage=3)

    # Add MIRAGE optimizers
    mirage_lr = args.lr * 0.1
    optimizers['mirage_disc'] = torch.optim.Adam(
        mirage_losses.get_discriminator_params(), lr=mirage_lr, weight_decay=args.weight_decay
    )
    optimizers['mirage_act_cls'] = torch.optim.Adam(
        mirage_losses.get_classifier_params(), lr=mirage_lr, weight_decay=args.weight_decay
    )
    optimizers['mirage_contrastive'] = torch.optim.Adam(
        mirage_losses.get_contrastive_params(), lr=mirage_lr, weight_decay=args.weight_decay
    )

    # Initialize wandb
    import wandb
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"tmr_mirage_{args.dataset}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Run Stage 3
    print(f"\nStarting Stage 3 with MIRAGE losses for {args.stage3_epochs} epochs")
    best_ar, best_ri = train_stage3(
        model, ar_classifier, disentangle_losses, recon_loss,
        physical_loss, physical_weights, frozen_sgn_loss,
        train_loader, val_loader, optimizers, args,
        auxiliary_models=None, logger=None,
        ri_classifier=ri_classifier, mirage_losses=mirage_losses
    )

    print(f"\nTraining complete! Best AR: {best_ar:.4f}, Best RI: {best_ri:.4f}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
