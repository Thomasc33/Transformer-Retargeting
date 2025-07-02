import os
import sys
import argparse
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import pickle
import json
import numpy as np
import subprocess
from datetime import datetime
import logging

# Add parent directory to path to import from main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.autoencoder import Model
from train import Trainer
from data import get_cross_data, load_data
from util import init_seed

# Configure logging
def setup_logger(log_dir, name, trial_id=None):
    """Set up logger for the experiment."""
    if trial_id is not None:
        log_file = os.path.join(log_dir, f"{name}_trial_{trial_id}.log")
    else:
        log_file = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def objective(trial, args, logger):
    """Optuna objective function to minimize validation loss."""
    # Sample hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    decoder_dropout = trial.suggest_float("decoder_dropout", 0.0, 0.3)

    # All hyperparameters have been sampled

    # FIXED: Sample loss weights with numerically stable ranges
    # Updated ranges based on numerical stability analysis - joint limit was causing NaN gradients
    loss_mse = trial.suggest_float("loss_mse", 0.5, 2.0)  # Reduced from 1.0-10.0
    loss_ee = trial.suggest_float("loss_ee", 0.5, 2.0)   # Reduced from 1.0-10.0
    loss_smoothing = trial.suggest_float("loss_smoothing", 0.05, 0.5)  # Reduced from 0.01-5.0
    loss_inception = trial.suggest_float("loss_inception", 0.05, 0.5)  # Reduced from 0.01-5.0
    loss_fid_vel = trial.suggest_float("loss_fid_vel", 0.2, 1.0)  # Reduced from 0.1-10.0
    loss_bone = trial.suggest_float("loss_bone", 0.5, 2.0)  # Reduced from 1.0-15.0
    loss_foot = trial.suggest_float("loss_foot", 0.05, 0.2)  # Reduced from 0.5-5.0
    loss_joint_limit = trial.suggest_float("loss_joint_limit", 0.005, 0.05)  # DRASTICALLY REDUCED: was 0.1-3.0, now 0.005-0.05

    # Create a unique output directory for this trial
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # Save hyperparameters to JSON
    hyperparams = {
        "batch_size": batch_size,
        "lr": lr,
        "decoder_dropout": decoder_dropout,
        "loss_mse": loss_mse,
        "loss_ee": loss_ee,
        "loss_smoothing": loss_smoothing,
        "loss_inception": loss_inception,
        "loss_fid_vel": loss_fid_vel,
        "loss_bone": loss_bone,
        "loss_foot": loss_foot,
        "loss_joint_limit": loss_joint_limit,
        "trial_number": trial.number
    }

    with open(os.path.join(trial_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)

    # Log hyperparameters
    logger.info(f"Trial {trial.number} - Hyperparameters: {hyperparams}")

    # Set up model output path
    model_path = os.path.join(trial_dir, "model.pth")

    # Build command for training with these hyperparameters
    cmd = [
        "torchrun", "--nproc_per_node=4", "main.py",
        f"--dataset={args.dataset}",
        f"--setting={args.setting}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--epochs={args.epochs}",
        f"--train-samples={args.train_samples}",
        f"--test-samples={args.test_samples}",
        f"--teacher-forcing-ratio={args.teacher_forcing_ratio}",
        f"--teacher-forcing-decay={args.teacher_forcing_decay}",
        f"--loss-mse={loss_mse}",
        f"--loss-ee={loss_ee}",
        f"--loss-smoothing={loss_smoothing}",
        f"--loss-inception={loss_inception}",
        f"--loss-fid-vel={loss_fid_vel}",
        f"--loss-bone={loss_bone}",
        f"--loss-foot={loss_foot}",
        f"--loss-joint-limit={loss_joint_limit}",
        f"--decoder-dropout={decoder_dropout}",
        f"--output-model-path={model_path}",
        "--run-eval",  # Important: Run evaluation after training
        "--hpc"
    ]

    # Add data path if provided
    if args.data_path:
        cmd.append(f"--data-path={args.data_path}")

    # Add pretrained and freeze options
    if args.use_pretrained:
        cmd.append("--use-pretrained")
    else:
        cmd.append("--no-pretrained")

    if args.freeze_encoder:
        cmd.append("--freeze-encoder")
    else:
        cmd.append("--no-freeze-encoder")

    # Run the training process
    logger.info(f"Running command: {' '.join(cmd)}")

    # Create a log file for this specific trial
    trial_log_path = os.path.join(args.log_dir, f"trial_{trial.number}.log")
    with open(trial_log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        process.wait()

    # Check if training completed successfully
    if process.returncode != 0:
        logger.error(f"Trial {trial.number} failed with return code {process.returncode}")
        return float('inf')  # Return a large value to indicate failure

    # Create a metrics directory for this trial
    metrics_dir = os.path.join(trial_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Determine the correct model weights based on dataset and setting
    ar_model_weights = ""
    ri_model_weights = ""
    gc_model_weights = ""

    if args.dataset == "ntu":
        if args.setting == "cv":
            ar_model_weights = "eval/sgn/pretrained/ntu/cview_ar.pth"
            ri_model_weights = "eval/sgn/pretrained/ntu/cview_ri.pth"
            gc_model_weights = "output/ntu_gc_cview/NTU_gc_cview/model_best.pth.tar"
        else:  # cs
            ar_model_weights = "eval/sgn/pretrained/ntu/csub_ar.pth"
            ri_model_weights = "eval/sgn/pretrained/ntu/csub_ri.pth"
            gc_model_weights = "output/ntu_gc_csub/NTU_gc_csub/model_best.pth.tar"
    elif args.dataset == "ntu120":
        if args.setting == "cv":
            ar_model_weights = "eval/sgn/pretrained/ntu120/cview_ar.pth"
            ri_model_weights = "eval/sgn/pretrained/ntu120/cview_ri.pth"
            gc_model_weights = "output/ntu120_gc_cview/NTU120_gc_cview/model_best.pth.tar"
        else:  # cs
            ar_model_weights = "eval/sgn/pretrained/ntu120/csub_ar.pth"
            ri_model_weights = "eval/sgn/pretrained/ntu120/csub_ri.pth"
            gc_model_weights = "output/ntu120_gc_csub/NTU120_gc_csub/model_best.pth.tar"

    # Run evaluation to get metrics
    eval_cmd = [
        "python", "eval_model.py",
        f"--dataset={args.dataset}",
        f"--setting={args.setting}",
        f"--model_type=transformer",
        f"--transformer_model_path={model_path}",
        f"--eval_model=sgn",
        f"--test_samples={args.test_samples}",
        f"--output_dir={metrics_dir}",  # Save metrics to JSON file
        f"--ar_model_weights={ar_model_weights}",
        f"--ri_model_weights={ri_model_weights}",
        f"--gc_model_weights={gc_model_weights}"
    ]

    eval_log_path = os.path.join(args.log_dir, f"trial_{trial.number}_eval.log")
    with open(eval_log_path, "w") as log_file:
        eval_process = subprocess.Popen(
            eval_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        eval_process.wait()

    # Check if evaluation completed successfully
    if eval_process.returncode != 0:
        logger.error(f"Trial {trial.number} evaluation failed with return code {eval_process.returncode}")
        # Read the log to get more information about the failure
        with open(eval_log_path, "r") as f:
            eval_log_content = f.read()
            logger.error(f"Evaluation log: {eval_log_content}")
        return float('inf')  # Return a large value to indicate failure

    # Load metrics from the JSON file
    metrics_file = os.path.join(metrics_dir, "transformer_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        logger.info(f"Loaded metrics from {metrics_file}")
    else:
        logger.warning(f"Metrics file {metrics_file} not found, falling back to log parsing")
        # Parse evaluation results from log as fallback
        with open(eval_log_path, "r") as f:
            eval_log_content = f.read()

        # Initialize empty metrics dictionary
        metrics = {}

        # Action Recognition Accuracy (higher is better)
        ar_acc = None
        for line in eval_log_content.split('\n'):
            if "[SGN] Action Recognition Accuracy:" in line:
                try:
                    ar_acc = float(line.split("Accuracy:")[1].strip().split('%')[0])
                    metrics['action_recognition_accuracy'] = ar_acc
                except (ValueError, IndexError):
                    continue

        # Re-identification Accuracy (lower is better for anonymization)
        ri_acc = None
        for line in eval_log_content.split('\n'):
            if "[SGN] Re-identification Accuracy (Retargeted):" in line:
                try:
                    ri_acc = float(line.split("Retargeted):")[1].strip().split('%')[0])
                    metrics['reidentification_accuracy'] = ri_acc
                except (ValueError, IndexError):
                    continue

        # MSE with Ground Truth (lower is better)
        mse_gt = None
        for line in eval_log_content.split('\n'):
            if "[SGN] Average MSE (GT):" in line:
                try:
                    mse_gt = float(line.split("(GT):")[1].strip())
                    metrics['mse_gt'] = mse_gt
                except (ValueError, IndexError):
                    continue

        # Bone Length Consistency (lower is better)
        blc = None
        for line in eval_log_content.split('\n'):
            if "Bone Length Consistency:" in line and "(lower is better)" in line:
                try:
                    blc = float(line.split("Consistency:")[1].strip().split('(')[0])
                    metrics['bone_length_consistency'] = blc
                except (ValueError, IndexError):
                    continue

        # Joint Angle Limits (higher is better)
        jal = None
        for line in eval_log_content.split('\n'):
            if "Joint Angle Limits:" in line and "(higher is better)" in line:
                try:
                    jal = float(line.split("Limits:")[1].strip().split('%')[0])
                    metrics['joint_angle_limits'] = jal
                except (ValueError, IndexError):
                    continue

        # Temporal Smoothness (lower is better)
        ts = None
        for line in eval_log_content.split('\n'):
            if "Temporal Smoothness:" in line and "(lower is better)" in line:
                try:
                    ts = float(line.split("Smoothness:")[1].strip().split('(')[0])
                    metrics['temporal_smoothness'] = ts
                except (ValueError, IndexError):
                    continue

        # Velocity Consistency (higher is better)
        vc = None
        for line in eval_log_content.split('\n'):
            if "Velocity Consistency:" in line and "(higher is better)" in line:
                try:
                    vc = float(line.split("Consistency:")[1].strip().split('(')[0])
                    metrics['velocity_consistency'] = vc
                except (ValueError, IndexError):
                    continue

        # Foot Contact Consistency (higher is better)
        fcc = None
        for line in eval_log_content.split('\n'):
            if "Foot Contact Consistency:" in line and "(higher is better)" in line:
                try:
                    fcc = float(line.split("Consistency:")[1].strip().split('%')[0])
                    metrics['foot_contact_consistency'] = fcc
                except (ValueError, IndexError):
                    continue

    # Also get validation loss from training log
    with open(trial_log_path, "r") as f:
        train_log_content = f.read()

    val_loss = float('inf')  # Default to a high value
    for line in train_log_content.split('\n'):
        if "Validation Loss:" in line:
            try:
                val_loss = float(line.split("Validation Loss:")[1].strip().split()[0])
                metrics['validation_loss'] = val_loss
            except (ValueError, IndexError):
                continue

    # Save all metrics to the trial directory
    with open(os.path.join(trial_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Calculate combined score
    # We want to:
    # - Maximize: action_recognition_accuracy, joint_angle_limits, velocity_consistency, foot_contact_consistency
    # - Minimize: reidentification_accuracy, mse_gt, bone_length_consistency, temporal_smoothness, validation_loss

    # Extract metrics from the dictionary
    ar_acc = metrics.get('action_recognition_accuracy')
    ri_acc = metrics.get('reidentification_accuracy')
    mse_gt = metrics.get('mse_gt')
    blc = metrics.get('bone_length_consistency')
    jal = metrics.get('joint_angle_limits')
    ts = metrics.get('temporal_smoothness')
    vc = metrics.get('velocity_consistency')
    fcc = metrics.get('foot_contact_consistency')

    # Check if we have all the necessary metrics
    missing_metrics = []
    for metric_name, metric_value in [
        ("action_recognition_accuracy", ar_acc),
        ("reidentification_accuracy", ri_acc),
        ("mse_gt", mse_gt),
        ("bone_length_consistency", blc),
        ("joint_angle_limits", jal),
        ("temporal_smoothness", ts),
        ("velocity_consistency", vc),
        ("foot_contact_consistency", fcc),
        ("validation_loss", val_loss)
    ]:
        if metric_value is None:
            missing_metrics.append(metric_name)

    if missing_metrics:
        logger.warning(f"Trial {trial.number} missing metrics: {', '.join(missing_metrics)}")
        logger.warning(f"Using fallback score (validation loss: {val_loss})")
        # If any metric is missing, use validation loss as fallback
        combined_score = val_loss
    else:
        # Normalize metrics to 0-1 range with appropriate direction
        # For metrics we want to maximize, we use (1 - normalized_value)
        # For metrics we want to minimize, we use normalized_value directly

        # Action Recognition (maximize, range 0-100)
        ar_norm = 1.0 - (ar_acc / 100.0)

        # Re-identification (minimize, range 0-100)
        ri_norm = ri_acc / 100.0

        # MSE (minimize, typical range 0-1)
        # Cap at 1.0 to prevent extreme values from dominating
        mse_norm = min(mse_gt, 1.0)

        # Bone Length Consistency (minimize, typical range 0-0.1)
        # Scale to 0-1 range
        blc_norm = min(blc * 10, 1.0)

        # Joint Angle Limits (maximize, range 0-100)
        jal_norm = 1.0 - (jal / 100.0)

        # Temporal Smoothness (minimize, typical range 0-0.1)
        # Scale to 0-1 range
        ts_norm = min(ts * 10, 1.0)

        # Velocity Consistency (maximize, typical range 0-1)
        vc_norm = 1.0 - vc

        # Foot Contact Consistency (maximize, range 0-100)
        fcc_norm = 1.0 - (fcc / 100.0)

        # Validation Loss (minimize, typical range 0-10)
        # Scale to 0-1 range
        val_norm = min(val_loss / 10.0, 1.0)

        # Weighted sum of normalized metrics
        # Weights should sum to 1.0
        weights = {
            'ar_norm': 0.50,      # Action recognition (utility) - 50%
            'ri_norm': 0.20,      # Re-identification (privacy) - 20%
            'mse_norm': 0.05,     # MSE with ground truth
            'blc_norm': 0.05,     # Bone length consistency
            'jal_norm': 0.05,     # Joint angle limits
            'ts_norm': 0.05,      # Temporal smoothness
            'vc_norm': 0.05,      # Velocity consistency
            'fcc_norm': 0.03,     # Foot contact consistency
            'val_norm': 0.02      # Validation loss
        }

        combined_score = (
            weights['ar_norm'] * ar_norm +
            weights['ri_norm'] * ri_norm +
            weights['mse_norm'] * mse_norm +
            weights['blc_norm'] * blc_norm +
            weights['jal_norm'] * jal_norm +
            weights['ts_norm'] * ts_norm +
            weights['vc_norm'] * vc_norm +
            weights['fcc_norm'] * fcc_norm +
            weights['val_norm'] * val_norm
        )

    # Save the combined score
    metrics['combined_score'] = combined_score
    with open(os.path.join(trial_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Trial {trial.number} completed with combined score: {combined_score}")
    logger.info(f"Metrics: {metrics}")

    return combined_score

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--dataset", type=str, default="ntu", help="Dataset to use (ntu, ntu120, etri)")
    parser.add_argument("--setting", type=str, default="cv", help="Evaluation setting (cs or cv)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for each trial")
    parser.add_argument("--train-samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--test-samples", type=int, default=2000, help="Number of test samples")
    parser.add_argument("--teacher-forcing-ratio", type=float, default=1.0, help="Teacher forcing ratio")
    parser.add_argument("--teacher-forcing-decay", type=float, default=0.0, help="Teacher forcing decay rate")
    parser.add_argument("--data-path", type=str, default=None, help="Path to paired data file")
    parser.add_argument("--use-pretrained", action="store_true", help="Use pretrained encoder")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder parameters")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--output-dir", type=str, default="experiments/3_hyperparameter/results", help="Directory to save results")
    parser.add_argument("--log-dir", type=str, default="experiments/3_hyperparameter/logs", help="Directory to save logs")
    parser.add_argument("--study-name", type=str, default="hyperparameter_tuning", help="Name of the Optuna study")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    init_seed(args.seed)

    # Create output and log directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(args.log_dir, f"{args.study_name}_{timestamp}")

    # Log the start of the study
    logger.info(f"Starting Optuna study with {args.n_trials} trials")
    logger.info(f"Arguments: {args}")

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{args.study_name}_{timestamp}",
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args, logger),
        n_trials=args.n_trials
    )

    # Log best trial information
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {study.best_trial.params}")

    # Save study results
    study_results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "all_trials": [
            {
                "trial": trial.number,
                "value": trial.value,
                "params": trial.params
            }
            for trial in study.trials
        ]
    }

    with open(os.path.join(args.output_dir, f"study_results_{timestamp}.json"), "w") as f:
        json.dump(study_results, f, indent=4)

    # Create a script to reproduce the best trial
    best_trial_script = os.path.join(args.output_dir, f"reproduce_best_trial_{timestamp}.sh")
    with open(best_trial_script, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Script to reproduce the best trial from Optuna hyperparameter tuning\n\n")

        cmd = [
            "torchrun --nproc_per_node=4 main.py",
            f"--dataset={args.dataset}",
            f"--setting={args.setting}",
            f"--batch-size={study.best_trial.params['batch_size']}",
            f"--lr={study.best_trial.params['lr']}",
            f"--epochs={args.epochs}",
            f"--train-samples={args.train_samples}",
            f"--test-samples={args.test_samples}",
            f"--teacher-forcing-ratio={args.teacher_forcing_ratio}",
            f"--teacher-forcing-decay={args.teacher_forcing_decay}",
            f"--loss-mse={study.best_trial.params['loss_mse']}",
            f"--loss-ee={study.best_trial.params['loss_ee']}",
            f"--loss-smoothing={study.best_trial.params['loss_smoothing']}",
            f"--loss-inception={study.best_trial.params['loss_inception']}",
            f"--loss-fid-vel={study.best_trial.params['loss_fid_vel']}",
            f"--loss-bone={study.best_trial.params['loss_bone']}",
            f"--loss-foot={study.best_trial.params['loss_foot']}",
            f"--loss-joint-limit={study.best_trial.params['loss_joint_limit']}",
            f"--decoder-dropout={study.best_trial.params['decoder_dropout']}",
            f"--output-model-path=experiments/3_hyperparameter/results/best_model_{timestamp}.pth",
            "--run-eval",
            "--hpc"
        ]

        # Determine the correct model weights based on dataset and setting
        ar_model_weights = ""
        ri_model_weights = ""
        gc_model_weights = ""

        if args.dataset == "ntu":
            if args.setting == "cv":
                ar_model_weights = "eval/sgn/pretrained/ntu/cview_ar.pth"
                ri_model_weights = "eval/sgn/pretrained/ntu/cview_ri.pth"
                gc_model_weights = "output/ntu_gc_cview/NTU_gc_cview/model_best.pth.tar"
            else:  # cs
                ar_model_weights = "eval/sgn/pretrained/ntu/csub_ar.pth"
                ri_model_weights = "eval/sgn/pretrained/ntu/csub_ri.pth"
                gc_model_weights = "output/ntu_gc_csub/NTU_gc_csub/model_best.pth.tar"
        elif args.dataset == "ntu120":
            if args.setting == "cv":
                ar_model_weights = "eval/sgn/pretrained/ntu120/cview_ar.pth"
                ri_model_weights = "eval/sgn/pretrained/ntu120/cview_ri.pth"
                gc_model_weights = "output/ntu120_gc_cview/NTU120_gc_cview/model_best.pth.tar"
            else:  # cs
                ar_model_weights = "eval/sgn/pretrained/ntu120/csub_ar.pth"
                ri_model_weights = "eval/sgn/pretrained/ntu120/csub_ri.pth"
                gc_model_weights = "output/ntu120_gc_csub/NTU120_gc_csub/model_best.pth.tar"

        # Add model weights to the command
        cmd.append(f"--ar_model_weights={ar_model_weights}")
        cmd.append(f"--ri_model_weights={ri_model_weights}")
        cmd.append(f"--gc_model_weights={gc_model_weights}")

        if args.data_path:
            cmd.append(f"--data-path={args.data_path}")

        if args.use_pretrained:
            cmd.append("--use-pretrained")
        else:
            cmd.append("--no-pretrained")

        if args.freeze_encoder:
            cmd.append("--freeze-encoder")
        else:
            cmd.append("--no-freeze-encoder")

        f.write(" \\\n    ".join(cmd))
        f.write("\n")

    # Make the script executable
    os.chmod(best_trial_script, 0o755)

    logger.info(f"Study completed. Results saved to {args.output_dir}")
    logger.info(f"Best trial reproduction script saved to {best_trial_script}")

if __name__ == "__main__":
    main()
