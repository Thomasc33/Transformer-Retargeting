#!/usr/bin/env python3
"""
Training script for Disentangled TMR

Implements 3-stage training:
1. Stage 1: Encoder pretraining with disentanglement losses
2. Stage 2: Decoder training with reconstruction losses
3. Stage 3: End-to-end fine-tuning with all losses
"""

import os
import sys
import argparse
import time
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gc
import numpy as np
from pathlib import Path
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.disentangled_tmr import create_disentangled_tmr
from src.model.simple_classifiers import ActionClassifier, IdentityClassifier
from src.training.disentanglement_losses import DisentanglementLosses
from src.training.schedulers import get_scheduler
from src.training.loss import Loss
from src.data.datasets import Cross_Data
from src.losses.physical_plausibility import PhysicalPlausibilityLoss
from src.losses.frozen_sgn_loss import FrozenSGNLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train Disentangled TMR')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True, help='Path to paired data')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ntu120', 'etri', 'ntu_smoke', 'ntu_small'])
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples (-1 for all)')
    
    # Model (Aberman et al. 2019 style: larger action, smaller identity)
    parser.add_argument('--d_action', type=int, default=768, help='Action feature dimension (increased from 512 for better capacity)')
    parser.add_argument('--d_identity', type=int, default=256, help='Identity feature dimension (increased from 128 for better capacity)')
    parser.add_argument('--d_model', type=int, default=320, help='Decoder model dimension')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--use_action_backbone', action='store_true', default=True,
                       help='Use action-recognition backbone architecture (random init, no pretrained weights)')
    parser.add_argument('--no_action_backbone', action='store_false', dest='use_action_backbone',
                       help='Disable action-recognition backbone architecture')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Post-hoc retarget blending: output = beta*retargeted + (1-beta)*source. 1.0=full retarget, <1 partial.')
    parser.add_argument('--no_temporal_convs', action='store_true', help='Disable temporal convolution blocks in action encoder')
    parser.add_argument('--no_lstm', action='store_true', help='Disable LSTM in action encoder')
    parser.add_argument('--identity_mode', type=str, choices=['static', 'full_seq'], default='static',
                        help='Identity encoder mode: static pose (default) or full sequence pooling')
    parser.add_argument('--tokenizer', type=str, choices=['none', 'pos', 'dynamics'], default='none',
                        help='Tokenizer type for action encoder')
    parser.add_argument('--tokenizer_dim', type=int, default=256, help='Tokenizer output dimension before projection')
    parser.add_argument(
        '--token_fusion',
        type=str,
        choices=['add', 'replace'],
        default='add',
        help='Fuse tokenizer stream with baseline pos/vel/acc stream (additive by default)',
    )
    parser.add_argument('--use_codebook', action='store_true', help='Enable vector-quantized codebook on tokens')
    parser.add_argument('--codebook_size', type=int, default=256, help='Number of codebook entries')
    parser.add_argument('--codebook_dim', type=int, default=256, help='Codebook embedding dimension')
    parser.add_argument(
        '--codebook_distance',
        type=str,
        choices=['euclidean', 'cosine'],
        default='euclidean',
        help='Distance metric for nearest-neighbor codebook lookup',
    )
    parser.add_argument(
        '--vq_commitment_weight',
        type=float,
        default=0.25,
        help='VQ commitment loss weight (beta) used inside the VQ-VAE loss',
    )
    parser.add_argument(
        '--weight_vq',
        type=float,
        default=1.0,
        help='Overall weight applied to the VQ loss when codebook is enabled',
    )
    
    # Training
    parser.add_argument('--stage1_epochs', type=int, default=20, help='Stage 1 epochs (increased for better encoder learning)')
    parser.add_argument('--stage2_epochs', type=int, default=15, help='Stage 2 epochs')
    parser.add_argument('--stage3_epochs', type=int, default=20, help='Stage 3 epochs (increased for better fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (increased for better GPU utilization)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for encoders')
    parser.add_argument('--lr_classifier', type=float, default=1e-3, help='Learning rate for classifiers')
    parser.add_argument('--weight_decay', type=float, default=9.689e-05, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers (increased for faster data loading)')

    # Training strategy (hybrid approach)
    parser.add_argument('--freeze_encoders_stage3', action='store_true',
                       help='Freeze encoders in Stage 3 (hybrid strategy: train only decoder)')
    parser.add_argument('--stage2_encoder_lr_factor', type=float, default=0.01,
                       help='LR factor for encoder fine-tuning in Stage 2 (default: 0.01 = 1% of base LR)')
    
    # Loss weights
    parser.add_argument('--weight_ar', type=float, default=3.0, help='Weight for AR classification loss (prioritize action recognition)')
    parser.add_argument('--weight_ri', type=float, default=1.0, help='Weight for RI classification loss')
    parser.add_argument('--weight_contrastive', type=float, default=1.0)
    parser.add_argument('--weight_adversarial', type=float, default=1.0)
    parser.add_argument('--weight_orthogonality', type=float, default=1.0)
    parser.add_argument('--weight_mutual_info', type=float, default=0.01)  # Scaled down due to high magnitude

    # Reconstruction loss weights
    parser.add_argument('--weight_mse', type=float, default=1.0, help='Weight for MSE reconstruction loss')
    parser.add_argument('--weight_l1', type=float, default=0.0, help='Weight for L1 reconstruction loss')
    parser.add_argument('--weight_smoothl1', type=float, default=0.0, help='Weight for SmoothL1 reconstruction loss')
    parser.add_argument('--weight_ee', type=float, default=1.0, help='Weight for end-effector velocity loss')
    parser.add_argument('--weight_smoothing', type=float, default=1.0, help='Weight for temporal smoothing loss')
    parser.add_argument('--weight_bone', type=float, default=1.0, help='Weight for bone length loss')
    parser.add_argument('--weight_foot', type=float, default=1.0, help='Weight for foot contact loss')
    parser.add_argument('--weight_joint_limit', type=float, default=1.0, help='Weight for joint limit loss')
    parser.add_argument('--weight_fid_vel', type=float, default=1.0, help='Weight for FID velocity loss')

    # Physical plausibility loss weights
    parser.add_argument('--weight_bone_length', type=float, default=0.5, help='Weight for bone length consistency loss')
    parser.add_argument('--weight_temporal_smoothness', type=float, default=0.3, help='Weight for temporal smoothness loss')
    parser.add_argument('--weight_velocity', type=float, default=0.2, help='Weight for velocity consistency loss')

    # Frozen SGN auxiliary loss
    # parser.add_argument('--use_frozen_sgn', action='store_true', help='Use frozen SGN auxiliary loss')
    parser.add_argument('--use_frozen_sgn', action='store_true', default=False, help='Use frozen SGN auxiliary loss (Warning: requires pretrained weights)')
    parser.add_argument('--frozen_sgn_checkpoint', type=str, default='output/ntu_sgn_ar_paired/model_best.pth.tar',
                        help='Path to pre-trained SGN checkpoint')
    parser.add_argument('--weight_frozen_sgn', type=float, default=0.2, help='Weight for frozen SGN auxiliary loss (keep small to avoid adversarial perturbations)')

    # Teacher forcing schedules (Stage 2/3)
    parser.add_argument('--stage2_teacher_forcing_start', type=float, default=1.0, help='Stage 2 teacher forcing start ratio')
    parser.add_argument('--stage2_teacher_forcing_end', type=float, default=0.5, help='Stage 2 teacher forcing minimum ratio')
    parser.add_argument('--stage3_teacher_forcing_start', type=float, default=0.5, help='Stage 3 teacher forcing start ratio')
    parser.add_argument('--stage3_teacher_forcing_end', type=float, default=0.3, help='Stage 3 teacher forcing minimum ratio')

    # Optimization improvements
    parser.add_argument('--use_gradient_clip', action='store_true', help='Use gradient clipping')
    parser.add_argument('--gradient_clip_value', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', choices=['cosine', 'plateau'],
                        help='Type of LR scheduler')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision (enabled by default for CUDA)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps for larger effective batch size')
    parser.add_argument('--weight_action_preservation', type=float, default=1.0, help='Weight for action preservation loss')
    parser.add_argument('--weight_feature_consistency', type=float, default=0.5, help='Weight for feature consistency loss')
    parser.add_argument('--weight_motion_dynamics', type=float, default=0.1, help='Weight for motion dynamics loss')
    parser.add_argument('--weight_end_effector', type=float, default=1.0, help='Weight for end-effector position/velocity loss')
    

    # MIRAGE-inspired losses (Stage 3 only)
    parser.add_argument('--use_mirage_losses', action='store_true',
                       help='Enable all MIRAGE-inspired losses in Stage 3')
    parser.add_argument('--lambda_dist_disc', type=float, default=1.0,
                       help='Weight for distribution discriminator loss')
    parser.add_argument('--lambda_output_act', type=float, default=1.0,
                       help='Weight for output-level action classifier loss')
    parser.add_argument('--lambda_output_id', type=float, default=1.0,
                       help='Weight for output-level identity adversary loss')
    parser.add_argument('--lambda_output_contrastive', type=float, default=1.0,
                       help='Weight for output-level contrastive scattering loss')
    parser.add_argument('--lambda_ee_enhanced', type=float, default=1.0,
                       help='Weight for enhanced end-effector loss')
    parser.add_argument("--lambda_motion_disc", type=float, default=0.0,
                       help="Weight for motion-space distribution discriminator")
    parser.add_argument("--lambda_coord_std", type=float, default=0.0,
                       help="Weight for coordinate standardization loss")
    parser.add_argument("--soft_retarget_alpha", type=float, default=None,
                       help="Deprecated: use --beta instead.")
    parser.add_argument("--raw_stats_path", type=str, default=None,
                       help="Path to raw data pkl for coordinate standardization stats")
    # Output
    parser.add_argument('--output_dir', type=str, default='output/disentangled_tmr', help='Output directory')
    parser.add_argument('--save_freq', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--log_freq', type=int, default=10, help='Log every N batches')

    # Wandb
    parser.add_argument('--wandb_project', type=str, default='disentangled-tmr', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')

    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    parser.add_argument('--resume_stage', type=int, default=None, help='Resume from specific stage (auto-detect if None)')
    parser.add_argument('--auto_resume', action='store_true', help='Automatically resume from latest checkpoint if available')
    parser.add_argument('--resume_strict', action='store_true', default=True, help='Strict state dict loading for resume')

    # Enhanced logging and monitoring
    parser.add_argument('--no_progress_bars', action='store_true', help='Disable progress bars')
    parser.add_argument('--monitor_resources', action='store_true', default=True, help='Monitor system resources')
    parser.add_argument('--downstream_eval_freq', type=int, default=5, help='Downstream AR evaluation frequency (epochs)')
    parser.add_argument('--use_downstream_early_stop', action='store_true', help='Use downstream AR for early stopping')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience (epochs)')

    # Device
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # SLURM mode
    parser.add_argument('--slurm', action='store_true',
                       help='Generate SLURM script instead of training')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_data(args):
    """Load paired skeleton data"""
    print(f"Loading data from {args.data_path}...")

    data = torch.load(args.data_path, weights_only=False)

    # Extract datasets
    train_dataset = data['train']
    val_dataset = data['test']  # Note: file uses 'test' key for validation

    # Limit samples if specified
    if args.num_samples > 0 and args.num_samples < len(train_dataset):
        # If dataset is a list, just slice it
        if isinstance(train_dataset, list):
            train_dataset = train_dataset[:args.num_samples]
            val_samples = min(args.num_samples // 5, len(val_dataset))
            val_dataset = val_dataset[:val_samples]
        # If dataset is a Cross_Data object, slice its attributes
        elif hasattr(train_dataset, 'sampled_data'):
            train_dataset.sampled_data = train_dataset.sampled_data[:args.num_samples]
            train_dataset.actors = train_dataset.actors[:args.num_samples]
            train_dataset.actions = train_dataset.actions[:args.num_samples]

            val_samples = min(args.num_samples // 5, len(val_dataset))
            val_dataset.sampled_data = val_dataset.sampled_data[:val_samples]
            val_dataset.actors = val_dataset.actors[:val_samples]
            val_dataset.actions = val_dataset.actions[:val_samples]

    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    # Create dataloaders
    # Note: On CPU environments, setting pin_memory=True generates warnings.
    # However, if user explicitly requested CUDA, we should respect that intent
    # even if torch.cuda.is_available() is False (the script might fail later, but we tried).
    # But usually pin_memory is only useful if we have a GPU.
    pin_memory = args.device == 'cuda' and torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader


def create_models(args):
    """Create all models"""
    print("Creating models...")

    if args.use_codebook and args.tokenizer == 'none':
        raise ValueError("--use_codebook requires --tokenizer != none")
    
    # Get number of classes and identities from dataset
    if args.dataset in ['ntu', 'ntu_smoke', 'ntu_small']:
        num_class = 49  # Cross-view
        num_identities = 40
    elif args.dataset == 'ntu120':
        num_class = 94   # 120 - 26 two-person actions (50-60, 106-120)
        num_identities = 106
    elif args.dataset == 'etri':
        num_class = 55
        num_identities = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create main model
    model = create_disentangled_tmr(
        dataset=args.dataset,
        num_class=num_class,
        device=args.device,
        d_action=args.d_action,
        d_identity=args.d_identity,
        d_model=args.d_model,
        num_decoder_layers=args.num_decoder_layers,
        use_pretrained_action=args.use_action_backbone,
        use_temporal_convs=not args.no_temporal_convs,
        use_lstm=not args.no_lstm,
        identity_use_full_sequence=(args.identity_mode == 'full_seq'),
        tokenizer_type=None if args.tokenizer == 'none' else args.tokenizer,
        tokenizer_dim=args.tokenizer_dim,
        token_fusion=args.token_fusion,
        use_codebook=args.use_codebook,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        codebook_distance=args.codebook_distance,
        vq_commitment_weight=args.vq_commitment_weight,
    )
    
    # Create AR classifier (simple MLP for action features)
    ar_classifier = ActionClassifier(
        d_action=args.d_action,
        num_classes=num_class,
        dropout=0.5
    ).to(args.device)

    # Create RI classifier (simple MLP for identity features)
    ri_classifier = IdentityClassifier(
        d_identity=args.d_identity,
        num_identities=num_identities,
        dropout=0.5
    ).to(args.device)
    
    # Create disentanglement losses
    disentangle_losses = DisentanglementLosses(
        d_action=args.d_action,
        d_identity=args.d_identity,
        num_identities=num_identities,
        device=args.device
    )
    
    # Create reconstruction losses (weights configurable via CLI, skip zero-weight losses)
    loss_weights = {k: v for k, v in {
        'mse': args.weight_mse, 'l1': args.weight_l1, 'smoothl1': args.weight_smoothl1,
        'ee': args.weight_ee, 'smoothing': args.weight_smoothing,
        'bone': args.weight_bone, 'foot': args.weight_foot,
        'joint_limit': args.weight_joint_limit, 'fid_vel': args.weight_fid_vel
    }.items() if v > 0}
    print(f"  Active reconstruction losses: {loss_weights}")
    recon_loss = Loss(loss_weights, device=args.device, dataset=args.dataset)

    # Create physical plausibility losses
    physical_loss = PhysicalPlausibilityLoss(dataset=args.dataset, device=args.device)
    physical_weights = {
        'bone_length': args.weight_bone_length,
        'temporal_smoothness': args.weight_temporal_smoothness,
        'velocity': args.weight_velocity,
        'end_effector': args.weight_end_effector
    }

    # Create frozen SGN auxiliary loss (if enabled)
    frozen_sgn_loss = None
    if args.use_frozen_sgn:
        if not os.path.exists(args.frozen_sgn_checkpoint):
             print(f"\n{'='*80}")
             print("WARNING: FROZEN SGN CHECKPOINT NOT FOUND")
             print(f"Path: {args.frozen_sgn_checkpoint}")
             print("Skipping Frozen SGN Loss initialization.")
             print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print("INITIALIZING FROZEN SGN AUXILIARY LOSS")
            print(f"{'='*80}")
            print(f"⚠ WARNING: Using frozen SGN with weight {args.weight_frozen_sgn}")
            print(f"  This loss guides TMR to generate recognizable motion")
            print(f"  Safeguards against adversarial perturbations:")
            print(f"    1. SGN is completely frozen (no gradient updates)")
            print(f"    2. Loss weight is small ({args.weight_frozen_sgn})")
            print(f"    3. Combined with reconstruction and physical losses")
            print(f"{'='*80}\n")

            # Use 48 classes for frozen SGN (matches checkpoint)
            # Note: TMR uses 49 classes, but frozen SGN checkpoint has 48
            frozen_sgn_loss = FrozenSGNLoss(
                sgn_checkpoint_path=args.frozen_sgn_checkpoint,
                num_classes=48,  # Match SGN checkpoint, not dataset
                device=args.device
            )

    # Print model info
    params = model.get_num_params()
    print(f"\nModel Parameters:")
    print(f"  Action Encoder: {params['action_encoder']:,}")
    print(f"  Identity Encoder: {params['identity_encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Total: {params['total']:,}")
    
    ar_params = sum(p.numel() for p in ar_classifier.parameters())
    ri_params = sum(p.numel() for p in ri_classifier.parameters())
    print(f"  AR Classifier: {ar_params:,}")
    print(f"  RI Classifier: {ri_params:,}")

    return model, ar_classifier, ri_classifier, disentangle_losses, recon_loss, physical_loss, physical_weights, frozen_sgn_loss


def create_optimizers(model, ar_classifier, ri_classifier, disentangle_losses, args, stage):
    """Create optimizers for different training stages"""
    optimizers = {}
    
    if stage == 1:
        # Stage 1: Train encoders + classifiers + discriminator
        optimizers['action_encoder'] = torch.optim.Adam(
            model.action_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        optimizers['identity_encoder'] = torch.optim.Adam(
            model.identity_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        optimizers['ar_classifier'] = torch.optim.Adam(
            ar_classifier.parameters(), lr=args.lr_classifier, weight_decay=args.weight_decay
        )
        optimizers['ri_classifier'] = torch.optim.Adam(
            ri_classifier.parameters(), lr=args.lr_classifier, weight_decay=args.weight_decay
        )
        optimizers['discriminator'] = torch.optim.Adam(
            disentangle_losses.get_discriminator_params(), lr=args.lr
        )
        
    elif stage == 2:
        # Stage 2: Train decoder + fine-tune encoders with tiny LR
        optimizers['decoder'] = torch.optim.Adam(
            model.decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        # Fine-tune encoders with very small LR (default: 1% of base LR)
        encoder_lr = args.lr * args.stage2_encoder_lr_factor
        optimizers['action_encoder'] = torch.optim.Adam(
            model.action_encoder.parameters(), lr=encoder_lr, weight_decay=args.weight_decay
        )
        optimizers['identity_encoder'] = torch.optim.Adam(
            model.identity_encoder.parameters(), lr=encoder_lr, weight_decay=args.weight_decay
        )

    elif stage == 3:
        # Stage 3: Hybrid strategy - freeze encoders (optional), lower LR
        if args.freeze_encoders_stage3:
            print("\n" + "="*80)
            print("HYBRID STRATEGY: Freezing encoders and classifiers in Stage 3")
            print("Only training decoder with frozen SGN loss")
            print("="*80 + "\n")

            # Freeze encoders
            for param in model.action_encoder.parameters():
                param.requires_grad = False
            for param in model.identity_encoder.parameters():
                param.requires_grad = False

            # Train only decoder
            optimizers['decoder'] = torch.optim.Adam(
                model.decoder.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay
            )
        else:
            # Original strategy: Train everything
            optimizers['model'] = torch.optim.Adam(
                model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay
            )
            optimizers['ar_classifier'] = torch.optim.Adam(
                ar_classifier.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay
            )
            optimizers['discriminator'] = torch.optim.Adam(
                disentangle_losses.get_discriminator_params(), lr=args.lr * 0.1
            )
    
    return optimizers


def save_checkpoint(model, ar_classifier, ri_classifier, optimizers, epoch, stage, args, 
                   is_best=False, val_loss=None, best_metrics=None, training_state=None):
    """
    Save enhanced checkpoint with complete training state for resumption
    
    Args:
        model: TMR model
        ar_classifier: Action recognition classifier
        ri_classifier: Re-identification classifier  
        optimizers: Dictionary of optimizers
        epoch: Current epoch
        stage: Current training stage
        args: Training arguments
        is_best: Whether this is the best checkpoint
        val_loss: Validation loss (for Optuna)
        best_metrics: Dictionary of best metrics achieved so far
        training_state: Additional training state (epochs_without_improvement, etc.)
    """
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'stage': stage,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dicts': {k: v.state_dict() for k, v in optimizers.items()},
        'args': args,
        'timestamp': time.time(),
        'pytorch_version': torch.__version__,
    }

    # Save validation loss if provided (for Optuna)
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss

    # Save best metrics achieved so far
    if best_metrics is not None:
        checkpoint['best_metrics'] = best_metrics

    # Save additional training state for resumption
    if training_state is not None:
        checkpoint['training_state'] = training_state

    # Only save classifiers if they exist (Stage 1 and 3)
    if ar_classifier is not None:
        checkpoint['ar_classifier_state_dict'] = ar_classifier.state_dict()
    if ri_classifier is not None:
        checkpoint['ri_classifier_state_dict'] = ri_classifier.state_dict()

    # Save schedulers if they exist
    if hasattr(args, 'use_lr_scheduler') and args.use_lr_scheduler:
        scheduler_states = {}
        for name, optimizer in optimizers.items():
            scheduler_name = f'{name}_scheduler'
            if hasattr(args, scheduler_name):
                scheduler = getattr(args, scheduler_name)
                if scheduler is not None:
                    scheduler_states[scheduler_name] = scheduler.state_dict()
        if scheduler_states:
            checkpoint['scheduler_state_dicts'] = scheduler_states

    # Save latest checkpoint (overwrites previous)
    latest_path = os.path.join(args.output_dir, f'checkpoint_stage{stage}_latest.pth')
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved: {latest_path}")

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(args.output_dir, f'checkpoint_stage{stage}_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Best checkpoint saved: {best_path}")

    # Save checkpoint integrity verification
    try:
        # Verify checkpoint can be loaded
        test_checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
        required_keys = ['epoch', 'stage', 'model_state_dict', 'optimizer_state_dicts']
        for key in required_keys:
            if key not in test_checkpoint:
                raise ValueError(f"Missing required key: {key}")
        print(f"✓ Checkpoint integrity verified: {latest_path}")
    except Exception as e:
        print(f"⚠️ Checkpoint integrity check failed: {e}")


def load_checkpoint(checkpoint_path, model, ar_classifier=None, ri_classifier=None, 
                   optimizers=None, device='cuda', strict=True):
    """
    Load checkpoint and restore training state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: TMR model to load state into
        ar_classifier: Action recognition classifier (optional)
        ri_classifier: Re-identification classifier (optional)
        optimizers: Dictionary of optimizers (optional)
        device: Device to load checkpoint on
        strict: Whether to strictly match state dict keys
    
    Returns:
        dict: Loaded checkpoint with training state
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Verify checkpoint integrity
        required_keys = ['epoch', 'stage', 'model_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Invalid checkpoint: missing key '{key}'")
        
        # Load model state
        try:
            incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        except RuntimeError as e:
            if not strict:
                raise
            print(f"⚠️ Strict model load failed: {e}")
            print("⚠️ Retrying with strict=False (may drop new/old keys across code changes).")
            incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if hasattr(incompatible, "missing_keys") and (incompatible.missing_keys or incompatible.unexpected_keys):
            print(
                f"⚠️ Non-strict load details: missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )

        print(f"✓ Model state loaded from epoch {checkpoint['epoch']}, stage {checkpoint['stage']}")
        
        # Load classifier states
        if ar_classifier is not None and 'ar_classifier_state_dict' in checkpoint:
            try:
                ar_classifier.load_state_dict(checkpoint['ar_classifier_state_dict'], strict=strict)
            except RuntimeError as e:
                if not strict:
                    raise
                print(f"⚠️ Strict AR classifier load failed: {e}")
                print("⚠️ Retrying AR classifier load with strict=False.")
                ar_classifier.load_state_dict(checkpoint['ar_classifier_state_dict'], strict=False)
            print("✓ AR classifier state loaded")
        
        if ri_classifier is not None and 'ri_classifier_state_dict' in checkpoint:
            try:
                ri_classifier.load_state_dict(checkpoint['ri_classifier_state_dict'], strict=strict)
            except RuntimeError as e:
                if not strict:
                    raise
                print(f"⚠️ Strict RI classifier load failed: {e}")
                print("⚠️ Retrying RI classifier load with strict=False.")
                ri_classifier.load_state_dict(checkpoint['ri_classifier_state_dict'], strict=False)
            print("✓ RI classifier state loaded")
        
        # Load optimizer states
        if optimizers is not None and 'optimizer_state_dicts' in checkpoint:
            for name, optimizer in optimizers.items():
                if name in checkpoint['optimizer_state_dicts']:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dicts'][name])
                        print(f"✓ Optimizer '{name}' state loaded")
                    except (ValueError, RuntimeError) as e:
                        print(f"⚠️ Could not restore optimizer state '{name}': {e}")
                        print("⚠️ Continuing with freshly initialized optimizer state.")
        
        # Load scheduler states
        if 'scheduler_state_dicts' in checkpoint:
            print("✓ Scheduler states available in checkpoint")
        
        print(f"✓ Checkpoint loaded successfully from epoch {checkpoint['epoch']}")
        return checkpoint
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        raise


def find_latest_checkpoint(output_dir, stage=None):
    """
    Find the latest checkpoint in output directory
    
    Args:
        output_dir: Output directory to search
        stage: Specific stage to look for (optional)
    
    Returns:
        str: Path to latest checkpoint, or None if not found
    """
    if not os.path.exists(output_dir):
        return None
    
    checkpoint_files = []
    
    if stage is not None:
        # Look for specific stage
        patterns = [
            f'checkpoint_stage{stage}_latest.pth',
            f'checkpoint_stage{stage}_best.pth'
        ]
    else:
        # Look for any checkpoint
        patterns = [
            'checkpoint_stage*_latest.pth',
            'checkpoint_stage*_best.pth'
        ]
    
    import glob
    for pattern in patterns:
        files = glob.glob(os.path.join(output_dir, pattern))
        for file in files:
            if os.path.isfile(file):
                mtime = os.path.getmtime(file)
                checkpoint_files.append((file, mtime))
    
    if not checkpoint_files:
        return None
    
    # Return most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])[0]
    return latest_checkpoint


def train_disentangled_tmr(args, optuna_trial=None):
    """
    Train Disentangled TMR with given arguments

    This function can be called directly with an argparse.Namespace object,
    making it suitable for use with Optuna or other hyperparameter tuning frameworks.

    Args:
        args: argparse.Namespace with all training arguments
        optuna_trial: Optional optuna.Trial for intermediate reporting and pruning

    Returns:
        dict: Training results including best accuracies and losses
    """
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"disentangled_tmr_{args.dataset}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            dir=args.output_dir
        )
        print(f"✓ Wandb initialized: {args.wandb_project}/{run_name}")
    else:
        print("✓ Wandb disabled")

    # Anomaly detection disabled — causes crashes on transient NaN gradients
    # that the model normally recovers from. Enable only for debugging.
    # torch.autograd.set_detect_anomaly(True)

    # Load data
    train_loader, val_loader = load_data(args)

    # Create models
    model, ar_classifier, ri_classifier, disentangle_losses, recon_loss, physical_loss, physical_weights, frozen_sgn_loss = create_models(args)

    print("\n" + "="*80)
    print("DISENTANGLED TMR TRAINING - STARTING")
    print("="*80)
    print(f"Stage 1 epochs: {args.stage1_epochs}")
    print(f"Stage 2 epochs: {args.stage2_epochs}")
    print(f"Stage 3 epochs: {args.stage3_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Import training functions
    from train_disentangled_tmr_stages import train_stage1, train_stage2, train_stage3
    
    # Import enhanced logger
    try:
        from src.training.enhanced_logger import create_enhanced_logger
        enhanced_logger = create_enhanced_logger(args, f"disentangled_tmr_{args.dataset}")
        enhanced_logger.start_training(total_stages=3)
        print("✓ Enhanced logging enabled")
    except ImportError:
        print("Warning: Enhanced logger not available, using basic logging")
        enhanced_logger = None
    
    # Handle checkpoint resumption
    resume_checkpoint = None
    start_stage = 1
    start_epoch = 0
    
    if args.auto_resume or args.resume:
        if args.resume:
            checkpoint_path = args.resume
        else:
            checkpoint_path = find_latest_checkpoint(args.output_dir, args.resume_stage)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                resume_checkpoint = load_checkpoint(
                    checkpoint_path, model, ar_classifier, ri_classifier, 
                    device=args.device, strict=args.resume_strict
                )
                start_stage = resume_checkpoint['stage']
                start_epoch = resume_checkpoint['epoch'] + 1  # Resume from next epoch
                
                print(f"\n✓ Resuming from checkpoint: {checkpoint_path}")
                print(f"✓ Resume from stage {start_stage}, epoch {start_epoch}")
                
                # Restore best metrics if available
                if 'best_metrics' in resume_checkpoint:
                    print("✓ Best metrics restored from checkpoint:")
                    for key, value in resume_checkpoint['best_metrics'].items():
                        print(f"  {key}: {value:.4f}")
                
            except Exception as e:
                print(f"⚠️ Failed to resume from checkpoint: {e}")
                if args.resume:
                    raise
                print("Starting training from scratch...")
                resume_checkpoint = None
        elif args.resume:
            print(f"⚠️ Resume checkpoint not found: {args.resume}")
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    
    # Create auxiliary models for downstream AR evaluation if enabled
    auxiliary_models = None
    if getattr(args, 'use_downstream_early_stop', False):
        try:
            # This would need to be implemented based on available models
            print("Note: Downstream AR early stopping requested but auxiliary models not configured")
        except Exception as e:
            print(f"Warning: Could not create auxiliary models: {e}")
    
    # Stage 1: Encoder pretraining
    if start_stage <= 1:
        print("\n" + "="*80)
        print("STAGE 1: ENCODER PRETRAINING")
        if resume_checkpoint and start_stage == 1:
            print(f"RESUMING FROM EPOCH {start_epoch}")
        print("="*80 + "\n")

        optimizers = create_optimizers(model, ar_classifier, ri_classifier, disentangle_losses, args, stage=1)
        
        # Restore optimizer states if resuming
        if resume_checkpoint and start_stage == 1 and 'optimizer_state_dicts' in resume_checkpoint:
            for name, optimizer in optimizers.items():
                if name in resume_checkpoint['optimizer_state_dicts']:
                    try:
                        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dicts'][name])
                        print(f"✓ Restored optimizer state: {name}")
                    except (ValueError, RuntimeError) as e:
                        print(f"⚠️ Could not restore optimizer state '{name}': {e}")
                        print("⚠️ Continuing with freshly initialized optimizer state.")
        
        # Adjust stage 1 epochs if resuming
        remaining_epochs = max(0, args.stage1_epochs - start_epoch) if start_stage == 1 else args.stage1_epochs
        if remaining_epochs > 0:
            args_stage1 = argparse.Namespace(**vars(args))
            args_stage1.stage1_epochs = remaining_epochs
            # Create schedulers if requested
            schedulers = {}
            if args.use_lr_scheduler:
                warmup_epochs = max(1, args.stage1_epochs // 10)  # 10% of training for warmup
                for key, optimizer in optimizers.items():
                    schedulers[key] = get_scheduler(
                        optimizer, 
                        args.lr_scheduler_type, 
                        warmup_epochs, 
                        args.stage1_epochs,
                        min_lr=args.lr * 0.01
                    )
            
            # Train Stage 1
            best_ar_stage1 = train_stage1(model, ar_classifier, ri_classifier, disentangle_losses,
                          train_loader, val_loader, optimizers, args_stage1,
                          auxiliary_models=auxiliary_models, logger=enhanced_logger, schedulers=schedulers)
        else:
            print("Stage 1 already completed, skipping...")
            best_ar_stage1 = (resume_checkpoint or {}).get('best_metrics', {}).get('best_ar_accuracy', 0.0)

    # Optuna pruning after Stage 1: if AR is terrible, skip Stages 2-3
    if optuna_trial is not None:
        import optuna
        # Report Stage 1 AR as intermediate value (step=1)
        optuna_trial.report(best_ar_stage1, step=1)
        print(f"  [Optuna] Reported Stage 1 AR={best_ar_stage1:.4f} to pruner")
        if optuna_trial.should_prune():
            print(f"  [Optuna] Trial pruned after Stage 1 (AR={best_ar_stage1:.4f})")
            raise optuna.TrialPruned(f"Pruned after Stage 1: AR={best_ar_stage1:.4f}")

    # Stage 2: Decoder training
    if start_stage <= 2:
        # Free stage 1 memory (optimizers, grad buffers, cached activations)
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  GPU memory after cache clear: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated, {torch.cuda.memory_reserved()/1e9:.1f}GB reserved")

        print("\n" + "="*80)
        print("STAGE 2: DECODER TRAINING")
        if resume_checkpoint and start_stage == 2:
            print(f"RESUMING FROM EPOCH {start_epoch}")
        print("="*80 + "\n")

        optimizers = create_optimizers(model, ar_classifier, ri_classifier, disentangle_losses, args, stage=2)
        
        # Restore optimizer states if resuming
        if resume_checkpoint and start_stage == 2 and 'optimizer_state_dicts' in resume_checkpoint:
            for name, optimizer in optimizers.items():
                if name in resume_checkpoint['optimizer_state_dicts']:
                    try:
                        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dicts'][name])
                        print(f"✓ Restored optimizer state: {name}")
                    except (ValueError, RuntimeError) as e:
                        print(f"⚠️ Could not restore optimizer state '{name}': {e}")
                        print("⚠️ Continuing with freshly initialized optimizer state.")
        
        # Adjust stage 2 epochs if resuming
        remaining_epochs = max(0, args.stage2_epochs - start_epoch) if start_stage == 2 else args.stage2_epochs
        if remaining_epochs > 0:
            args_stage2 = argparse.Namespace(**vars(args))
            args_stage2.stage2_epochs = remaining_epochs
            train_stage2(model, recon_loss, physical_loss, physical_weights, frozen_sgn_loss,
                         train_loader, val_loader, optimizers, args_stage2, auxiliary_models, enhanced_logger,
                         action_encoder=model.action_encoder, ar_classifier=ar_classifier)
        else:
            print("Stage 2 already completed, skipping...")


    # Initialize MIRAGE-inspired losses if enabled
    mirage_losses = None
    if getattr(args, "use_mirage_losses", False):
        from src.losses.mirage_inspired import MirageInspiredLosses
        if args.dataset in ["ntu", "ntu_smoke", "ntu_small"]:
            _num_class_mirage = 49
            _num_id_mirage = 40
        elif args.dataset == "ntu120":
            _num_class_mirage = 94
            _num_id_mirage = 106
        elif args.dataset == "etri":
            _num_class_mirage = 55
            _num_id_mirage = 100
        else:
            _num_class_mirage = 49
            _num_id_mirage = 40
        mirage_losses = MirageInspiredLosses(
            num_classes=_num_class_mirage,
            num_identities=_num_id_mirage,
            device=args.device,
            lambda_dist_disc=args.lambda_dist_disc,
            lambda_output_act=args.lambda_output_act,
            lambda_output_id=args.lambda_output_id,
            lambda_output_contrastive=args.lambda_output_contrastive,
            lambda_ee_enhanced=args.lambda_ee_enhanced,
        )
        print("MIRAGE-inspired losses initialized")
        print(f"  Lambdas: dist_disc={args.lambda_dist_disc}, output_act={args.lambda_output_act}, "
              f"output_id={args.lambda_output_id}, contrastive={args.lambda_output_contrastive}, "
              f"ee_enhanced={args.lambda_ee_enhanced}")

    # Stage 3: End-to-end fine-tuning
    best_ar_stage3 = 0.0
    best_ri_stage3 = 1.0
    if start_stage <= 3:
        # Free stage 2 memory
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  GPU memory after cache clear: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated, {torch.cuda.memory_reserved()/1e9:.1f}GB reserved")

        print("\n" + "="*80)
        print("STAGE 3: END-TO-END FINE-TUNING")
        if resume_checkpoint and start_stage == 3:
            print(f"RESUMING FROM EPOCH {start_epoch}")
        print("="*80 + "\n")

        optimizers = create_optimizers(model, ar_classifier, ri_classifier, disentangle_losses, args, stage=3)

        # Add MIRAGE loss optimizers if enabled
        if mirage_losses is not None:
            mirage_lr = args.lr * 0.1  # Same as Stage 3 LR
            optimizers["mirage_disc"] = torch.optim.Adam(
                mirage_losses.get_discriminator_params(), lr=mirage_lr, weight_decay=args.weight_decay
            )
            optimizers["mirage_act_cls"] = torch.optim.Adam(
                mirage_losses.get_classifier_params(), lr=mirage_lr, weight_decay=args.weight_decay
            )
            optimizers["mirage_contrastive"] = torch.optim.Adam(
                mirage_losses.get_contrastive_params(), lr=mirage_lr, weight_decay=args.weight_decay
            )
            print("Added MIRAGE loss optimizers")
        
        # Restore optimizer states if resuming
        if resume_checkpoint and start_stage == 3 and 'optimizer_state_dicts' in resume_checkpoint:
            for name, optimizer in optimizers.items():
                if name in resume_checkpoint['optimizer_state_dicts']:
                    try:
                        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dicts'][name])
                        print(f"✓ Restored optimizer state: {name}")
                    except (ValueError, RuntimeError) as e:
                        print(f"⚠️ Could not restore optimizer state '{name}': {e}")
                        print("⚠️ Continuing with freshly initialized optimizer state.")
        
        # Adjust stage 3 epochs if resuming
        remaining_epochs = max(0, args.stage3_epochs - start_epoch) if start_stage == 3 else args.stage3_epochs
        if remaining_epochs > 0:
            args_stage3 = argparse.Namespace(**vars(args))
            args_stage3.stage3_epochs = remaining_epochs
            best_ar_stage3, best_ri_stage3 = train_stage3(model, ar_classifier, disentangle_losses, recon_loss,
                        physical_loss, physical_weights, frozen_sgn_loss,
                        train_loader, val_loader, optimizers, args_stage3, auxiliary_models, enhanced_logger,
                        ri_classifier=ri_classifier, mirage_losses=mirage_losses)
        else:
            print("Stage 3 already completed, skipping...")
            best_ar_stage3 = (resume_checkpoint or {}).get('best_metrics', {}).get('best_ar_accuracy', 0.0)
            best_ri_stage3 = (resume_checkpoint or {}).get('best_metrics', {}).get('best_ri_accuracy', 1.0)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

    # Finalize enhanced logging
    if enhanced_logger:
        final_metrics = {
            'best_ar_accuracy': best_ar_stage3,
            'best_ri_accuracy': best_ri_stage3,
        }
        enhanced_logger.end_training(final_metrics)
        enhanced_logger.close()
        print("✓ Enhanced logging finalized")

    # Finish wandb
    if not args.no_wandb:
        wandb.finish()
        print("✓ Wandb finished")

    # Return results for Optuna or other callers
    return {
        'best_ar_accuracy': best_ar_stage3,
        'best_ri_accuracy': best_ri_stage3,
    }


def generate_slurm_script(args):
    """Generate SLURM script for training"""
    # Extract dataset and setting from data_path if not explicitly provided
    data_path_parts = Path(args.data_path).stem.split('_')
    dataset = args.dataset
    setting = data_path_parts[1] if len(data_path_parts) > 1 else 'cv'

    job_name = f"tmr_{dataset}_{setting}"
    script_dir = Path("bash/train_eval/tmr")
    script_dir.mkdir(parents=True, exist_ok=True)

    script_path = script_dir / f"train_{job_name}.sbatch"

    # Build command with all arguments
    cmd_args = [
        f"--data_path {args.data_path}",
        f"--dataset {args.dataset}",
        f"--stage1_epochs {args.stage1_epochs}",
        f"--stage2_epochs {args.stage2_epochs}",
        f"--stage3_epochs {args.stage3_epochs}",
        f"--batch_size {args.batch_size}",
        f"--lr {args.lr}",
        f"--output_dir {args.output_dir}",
    ]

    if args.use_frozen_sgn:
        cmd_args.append(f"--use_frozen_sgn")
        cmd_args.append(f"--frozen_sgn_checkpoint {args.frozen_sgn_checkpoint}")

    if args.freeze_encoders_stage3:
        cmd_args.append("--freeze_encoders_stage3")

    # Join command args with backslash continuation
    cmd_args_str = ' \\\n    '.join(cmd_args)

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=GPU
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tcarr23@charlotte.edu

module load pytorch/2.3.0-cuda12.1
cd /users/tcarr23/Transformer-Retargeting
export PYTHONPATH=/users/tcarr23/Transformer-Retargeting:$PYTHONPATH

python scripts/train_disentangled_tmr.py \\
    {cmd_args_str}
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"✅ SLURM script saved: {script_path}")

    # Submit the job
    result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Job submitted: {result.stdout.strip()}")
    else:
        print(f"❌ Failed to submit job: {result.stderr}")


def main():
    """Main entry point when running as a script"""
    args = parse_args()

    if args.slurm:
        generate_slurm_script(args)
    else:
        train_disentangled_tmr(args)


if __name__ == "__main__":
    main()
