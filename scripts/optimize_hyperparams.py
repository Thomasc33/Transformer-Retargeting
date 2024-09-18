#!/usr/bin/env python3
"""
Hyperparameter optimization for Disentangled TMR.

Calls train_disentangled_tmr() directly (no subprocess) and uses
internal Stage 3 val AR/RI as the objective:
    score = 0.7 * val_AR - 0.3 * val_RI   (maximize)

Speedups over original (v1):
  - Default epochs reduced: 8/8/5 -> 3/3/2 (~4x fewer total epochs)
  - Data subsetting via --num_samples (default 5000, halves epoch time)
  - Optuna pruning after Stage 1 (kills bad trials early, saves ~60% time)
  - Log every 100 batches instead of 50 (minor I/O reduction)

Expected time per trial (V100, 5k samples, 3/3/2 epochs): ~2h
Expected trials per 24h per GPU: ~11
With 4 GPUs: ~44 trials total

Usage:
    python scripts/optimize_hyperparams.py --n_trials 10
    python scripts/optimize_hyperparams.py --n_trials 10 --resume  # continue study
"""

import argparse
import json
import os
import sys
import shutil
import time
import traceback
from pathlib import Path

import optuna

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_args(trial, cli_args):
    """Build training args namespace from Optuna trial suggestions + CLI overrides."""

    # -- Hyperparameters to tune --
    # Disentanglement loss weights
    w_contrastive = trial.suggest_float('weight_contrastive', 0.3, 2.0)
    w_adversarial = trial.suggest_float('weight_adversarial', 0.3, 2.0)
    w_orthogonality = trial.suggest_float('weight_orthogonality', 0.1, 1.5)
    w_mutual_info = trial.suggest_float('weight_mutual_info', 0.1, 1.5)

    # Classification head weights
    w_ar = trial.suggest_float('weight_ar', 0.3, 1.5)
    w_ri = trial.suggest_float('weight_ri', 0.3, 1.5)

    # Reconstruction loss weights
    w_mse = trial.suggest_float('weight_mse', 2.0, 10.0)
    w_bone = trial.suggest_float('weight_bone_length', 0.5, 3.0)
    w_ee = trial.suggest_float('weight_end_effector', 1.0, 5.0)
    w_smooth = trial.suggest_float('weight_temporal_smoothness', 0.05, 0.5)
    w_motion_dynamics = trial.suggest_float('weight_motion_dynamics', 0.05, 0.5)

    # Training hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # -- Fixed parameters --
    output_dir = str(ROOT / "output" / "optuna" / f"trial_{trial.number}")

    # Build a flat dict that matches train_disentangled_tmr's argparse
    args_dict = {
        # Data
        'data_path': cli_args.data_path,
        'dataset': 'ntu',
        'setting': 'cv',
        'seg': 64,
        'num_workers': 4,

        # Architecture (fixed -- validated by ablation study)
        'd_action': 768,
        'd_identity': 256,
        'd_model': 320,
        'num_decoder_layers': 6,
        'use_action_backbone': True,
        'no_temporal_convs': False,
        'no_lstm': True,
        'identity_mode': 'static',
        'tokenizer': None,
        'tokenizer_dim': 256,
        'token_fusion': 'add',
        'use_codebook': False,
        'codebook_size': 256,
        'codebook_dim': 256,
        'codebook_distance': 'euclidean',
        'vq_commitment_weight': 0.25,
        'weight_vq': 1.0,
        'num_samples': cli_args.num_samples,

        # Training
        'batch_size': 32,
        'lr': lr,
        'lr_classifier': lr * 10,
        'weight_decay': weight_decay,
        'device': 'cuda',
        'seed': 42,
        'no_amp': True,

        # Epochs (shortened for optimization)
        'stage1_epochs': cli_args.stage1_epochs,
        'stage2_epochs': cli_args.stage2_epochs,
        'stage3_epochs': cli_args.stage3_epochs,

        # Loss weights -- disentanglement
        'weight_contrastive': w_contrastive,
        'weight_adversarial': w_adversarial,
        'weight_orthogonality': w_orthogonality,
        'weight_mutual_info': w_mutual_info,
        'weight_ar': w_ar,
        'weight_ri': w_ri,

        # Loss weights -- reconstruction
        'weight_mse': w_mse,
        'weight_bone_length': w_bone,
        'weight_end_effector': w_ee,
        'weight_temporal_smoothness': w_smooth,
        'weight_motion_dynamics': w_motion_dynamics,
        'weight_velocity': 2.0,
        'weight_foot_contact': 0.5,
        'weight_joint_limit': 1.5,

        # Action preservation (Stage 2/3)
        'weight_action_preservation': 1.0,
        'weight_feature_consistency': 0.5,

        # Output
        'output_dir': output_dir,
        'save_freq': 999,  # Only save best checkpoints

        # Teacher forcing
        'stage2_teacher_forcing_start': 1.0,
        'stage2_teacher_forcing_end': 0.5,
        'stage3_teacher_forcing_start': 0.5,
        'stage3_teacher_forcing_end': 0.3,

        # Gradient
        'use_gradient_clip': False,
        'gradient_clip_value': 1.0,
        'gradient_accumulation_steps': 1,

        # Wandb
        'no_wandb': True,  # Disable wandb for Optuna trials
        'wandb_project': '',
        'wandb_run_name': None,

        # Resume
        'auto_resume': False,
        'resume': None,
        'resume_stage': None,
        'resume_strict': True,

        # Misc
        'log_freq': 100,  # Reduced from 50 for faster I/O
        'early_stop_patience': 999,  # Don't early stop in short trials
        'use_downstream_early_stop': False,
        'downstream_eval_freq': 999,
        'use_lr_scheduler': False,
        'lr_scheduler': 'none',
        'lr_step_size': 10,
        'lr_gamma': 0.5,
        'lr_patience': 5,
        'lr_min': 1e-6,
        'freeze_encoders_stage3': False,
        'stage2_encoder_lr_factor': 0.01,
        'no_progress_bars': True,
        'monitor_resources': False,
        'slurm': False,

        # Frozen SGN (disabled for speed)
        'use_frozen_sgn': False,
        'weight_frozen_sgn': 0.0,
        'frozen_sgn_checkpoint': '',
    }

    return argparse.Namespace(**args_dict)


def objective(trial, cli_args):
    """Optuna objective: train DisentangledTMR and return weighted AR/RI score."""
    from scripts.train_disentangled_tmr import train_disentangled_tmr

    args = build_args(trial, cli_args)
    output_dir = args.output_dir

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}  [{time.strftime('%H:%M:%S')}]")
    print(f"{'='*60}")
    print(f"  epochs={args.stage1_epochs}/{args.stage2_epochs}/{args.stage3_epochs}  samples={args.num_samples}")
    print(f"  lr={args.lr:.6f}  wd={args.weight_decay:.6f}")
    print(f"  w_contrastive={args.weight_contrastive:.3f}  w_adversarial={args.weight_adversarial:.3f}")
    print(f"  w_ortho={args.weight_orthogonality:.3f}  w_mi={args.weight_mutual_info:.3f}")
    print(f"  w_ar={args.weight_ar:.3f}  w_ri={args.weight_ri:.3f}")
    print(f"  w_mse={args.weight_mse:.3f}  w_bone={args.weight_bone_length:.3f}")
    print(f"  w_ee={args.weight_end_effector:.3f}  w_smooth={args.weight_temporal_smoothness:.3f}")

    trial_start = time.time()

    try:
        # Pass the Optuna trial for intermediate reporting and pruning
        results = train_disentangled_tmr(args, optuna_trial=trial)

        best_ar = results.get('best_ar_accuracy', 0.0)
        best_ri = results.get('best_ri_accuracy', 1.0)

        # Objective: higher AR is good, lower RI is good
        score = 0.7 * best_ar - 0.3 * best_ri

        elapsed = (time.time() - trial_start) / 60
        print(f"\n  Trial {trial.number} result ({elapsed:.1f} min):")
        print(f"    AR={best_ar:.4f}  RI={best_ri:.4f}  Score={score:.4f}")

        # Save trial results
        trial_results = {
            'trial': trial.number,
            'score': score,
            'best_ar': best_ar,
            'best_ri': best_ri,
            'elapsed_minutes': elapsed,
            'params': trial.params,
        }
        results_path = Path(output_dir) / "trial_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(trial_results, f, indent=2)

        return score

    except optuna.TrialPruned:
        # Re-raise pruned trials so Optuna handles them correctly
        elapsed = (time.time() - trial_start) / 60
        print(f"  Trial {trial.number} PRUNED after {elapsed:.1f} min")
        # Clean up pruned trial output to save disk
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise

    except Exception as e:
        elapsed = (time.time() - trial_start) / 60
        print(f"\n  Trial {trial.number} FAILED after {elapsed:.1f} min: {e}")
        traceback.print_exc()
        # Clean up failed trial output
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        return float('-inf')


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization for DisentangledTMR')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--study_name', default='tmr_optimization_v2', help='Study name')
    parser.add_argument('--storage', default=f'sqlite:///{ROOT}/output/optuna/optuna_v2.db', help='Optuna storage')
    parser.add_argument('--resume', action='store_true', help='Resume existing study')
    parser.add_argument('--data_path', default='data/ntu/ntu_cv_paired_10k.pt', help='Training data path')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of training samples per trial (-1 for all). '
                             'Default 5000 halves epoch time vs full 10k dataset.')
    parser.add_argument('--stage1_epochs', type=int, default=3, help='Stage 1 epochs per trial')
    parser.add_argument('--stage2_epochs', type=int, default=3, help='Stage 2 epochs per trial')
    parser.add_argument('--stage3_epochs', type=int, default=2, help='Stage 3 epochs per trial')
    cli_args = parser.parse_args()

    print("=" * 60)
    print("Disentangled TMR -- Optuna Hyperparameter Optimization v2")
    print("=" * 60)
    print(f"  Study:   {cli_args.study_name}")
    print(f"  Trials:  {cli_args.n_trials}")
    print(f"  Epochs:  {cli_args.stage1_epochs}/{cli_args.stage2_epochs}/{cli_args.stage3_epochs}")
    print(f"  Samples: {cli_args.num_samples}")
    print(f"  Data:    {cli_args.data_path}")
    print(f"  Objective: 0.7 * AR - 0.3 * RI (maximize)")
    print(f"  Pruning: MedianPruner after Stage 1 (3 startup trials)")

    # Ensure output dir exists for the SQLite DB
    os.makedirs(str(ROOT / "output" / "optuna"), exist_ok=True)

    # Stagger concurrent workers to avoid SQLite race conditions on NFS
    import time, random
    worker_delay = random.uniform(1, 15)
    print(f"  Worker stagger: sleeping {worker_delay:.1f}s to avoid DB race...")
    time.sleep(worker_delay)

    # Retry study creation to handle transient SQLite locking on NFS
    for attempt in range(5):
        try:
            study = optuna.create_study(
                study_name=cli_args.study_name,
                storage=cli_args.storage,
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=3,
                    n_warmup_steps=0,
                    interval_steps=1,
                ),
                load_if_exists=True,
            )
            break
        except Exception as e:
            if attempt < 4:
                wait = 5 * (attempt + 1) + random.uniform(0, 5)
                print(f"  DB access failed (attempt {attempt+1}/5): {e}")
                print(f"  Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
    existing = len(study.trials)
    if existing > 0:
        print(f"  Resumed study with {existing} existing trials")

    study.optimize(lambda trial: objective(trial, cli_args), n_trials=cli_args.n_trials)

    # Print summary
    print("\n" + "=" * 60)
    print("Optimization Complete")
    print("=" * 60)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials
              if t.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    print(f"  Completed: {len(completed)}  Pruned: {len(pruned)}  Failed: {len(failed)}")

    if completed:
        print(f"  Best trial:  {study.best_trial.number}")
        print(f"  Best score:  {study.best_value:.4f}")
        print(f"\n  Best parameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        # Save best params
        best_file = ROOT / "output" / "optuna" / "best_params.json"
        with open(best_file, 'w') as f:
            json.dump({
                'best_trial': study.best_trial.number,
                'best_score': study.best_value,
                'params': study.best_params,
            }, f, indent=2)
        print(f"\n  Saved to: {best_file}")
    else:
        print("  No completed trials!")


if __name__ == "__main__":
    main()
