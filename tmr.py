#!/usr/bin/env python3
"""
Disentangled Transformer Motion Retargeting (Study 3) - Unified CLI

This tool manages the entire lifecycle of the project:
1.  Environment Detection: Automatically detects if running on Windows (local) or Linux/HPC (SLURM).
2.  Job Submission: Runs jobs locally or auto-queues them to SLURM.
3.  Experiment Tracking: Tracks status of training, ablations, and baselines.
4.  Automation: 'run-all' command to launch everything needed.

Usage:
    python tmr.py status
    python tmr.py train --experiment stable
    python tmr.py train-baselines
    python tmr.py run-all
"""

import argparse
import datetime
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

# --- Constants & Configuration ---

ROOT = Path(__file__).parent.resolve()
SCRIPTS_DIR = ROOT / "scripts"
TRAIN_SCRIPT = SCRIPTS_DIR / "train_disentangled_tmr.py"
EVAL_SCRIPT = SCRIPTS_DIR / "evaluate_disentangled_tmr.py"
EVAL_SUITE_SCRIPT = SCRIPTS_DIR / "eval_tmr_ablation.py"
BASELINE_SCRIPT = SCRIPTS_DIR / "train_downstream_models.py"
GEN_SCRIPT = SCRIPTS_DIR / "generate_retargeted_dataset.py"

# Detect Environment
IS_WINDOWS = platform.system() == "Windows"
HAS_SLURM = shutil.which("sbatch") is not None
HOSTNAME = platform.node()

# Default SLURM Configuration (RedHat 9 HPC)
SLURM_DEFAULTS = {
    "partition": "GPU",
    "nodes": 1,
    "ntasks_per_node": 1,
    "gres": "gpu:1",
    "cpus_per_task": 16,
    "mem": "64G",
    "time": "24:00:00",
}

# --- Experiment Definitions ---

@dataclass
class Experiment:
    name: str
    script: Path
    args: List[str]
    output_dir: Path
    description: str
    required_files: List[str] = field(default_factory=lambda: ["model_best.pth.tar"])

    def get_status(self) -> str:
        """Check if experiment is completed, running, or pending."""
        if not self.output_dir.exists():
            return "Pending"
        
        # Check for completion markers
        all_files_exist = all((self.output_dir / f).exists() for f in self.required_files)
        if all_files_exist:
            return "Completed"
        
        # Check if recently modified (running)
        # Heuristic: if log file or dir modified in last 30 mins
        try:
            mtime = self.output_dir.stat().st_mtime
            if time.time() - mtime < 1800: # 30 mins
                return "Running"
        except FileNotFoundError:
            pass
            
        return "Incomplete/Failed"

# Define the Suite
EXPERIMENTS: Dict[str, Experiment] = {}

def register_experiment(name: str, script: Path, args: List[str], output_dir: str, desc: str, required_files=None):
    if required_files is None:
        required_files = ["model_best.pth.tar"]
    # Ensure output_dir is absolute or relative to ROOT
    out_path = Path(output_dir) if Path(output_dir).is_absolute() else ROOT / output_dir
    EXPERIMENTS[name] = Experiment(name, script, args, out_path, desc, required_files)

# 1. Baselines
register_experiment(
    "baselines_ar_ri",
    BASELINE_SCRIPT,
    ["--models", "sgn_ar", "sgn_ri", "mix_ar", "mix_ri", "--epochs", "60", "--batch_size", "128"],
    "output",
    "Train downstream AR/RI models (SGN, MixFormer) on real data",
    required_files=[
        "ntu_sgn_ar_paired/model_best.pth.tar",
        "ntu_sgn_ri_paired/model_best.pth.tar",
        "ntu_mixformer_ar_paired/model_best.pth.tar",
        "ntu_mixformer_ri_paired/model_best.pth.tar"
    ]
)

# 2. Stable TMR (Main Model)
# Note: Using --no_amp to avoid MixFormer NaN issues
register_experiment(
    "tmr_stable",
    TRAIN_SCRIPT,
    [
        "--data_path", "data/ntu/ntu_cv_paired_10k.pt",
        "--dataset", "ntu",
        "--stage1_epochs", "20",
        "--stage2_epochs", "15",
        "--stage3_epochs", "20",
        "--batch_size", "128", # Increased for A100/H200 utilization
        "--no_amp",
        "--weight_orthogonality", "0.1",
        "--weight_adversarial", "0.5",
        "--weight_end_effector", "2.0",
        "--weight_bone_length", "1.0",
        "--weight_motion_dynamics", "0.2",
        "--weight_temporal_smoothness", "0.1",
        "--output_dir", "output/disentangled_tmr_stable",
        "--wandb_project", "disentangled-tmr-stable"
    ],
    "output/disentangled_tmr_stable",
    "Main Disentangled TMR Model (Stable)",
    required_files=["checkpoint_stage3_best.pth"]
)

# 3. Ablations
ABLATIONS = [
    ("no_ortho", ["--weight_orthogonality", "0.0"], "Remove Orthogonality Loss"),
    ("no_grl", ["--weight_adversarial", "0.0"], "Remove Adversarial Loss"),
    ("no_action_backbone", ["--no_action_backbone"], "Remove Action Encoder Backbone"),
    # "no_lstm" removed because it is now the default
    ("no_temporal_convs", ["--no_temporal_convs"], "Remove Temporal Convs"),
    ("static_identity", ["--identity_mode", "static"], "Static Identity (Default)"), 
    # Note: static is default, but explicit ablation might check dynamic
    ("full_seq_identity", ["--identity_mode", "full_seq"], "Full Sequence Identity Pooling"),
]

for name, flags, desc in ABLATIONS:
    full_name = f"tmr_ablation_{name}"
    out_dir = f"output/disentangled_tmr_{name}"
    # Copy stable args and append ablation flags
    # We base ablations on the 'stable' config but override specific flags
    # For simplicity here, we'll just reconstruct the base args + flags
    base_args = [
        "--data_path", "data/ntu/ntu_cv_paired_10k.pt",
        "--dataset", "ntu",
        "--stage1_epochs", "20",
        "--stage2_epochs", "15",
        "--stage3_epochs", "20",
        "--batch_size", "32",  # Reduced from 64 to 32 to prevent OOM
        "--no_amp",
        "--no_lstm",  # Global default: No LSTM for all experiments
        "--output_dir", out_dir,
        "--wandb_project", "disentangled-tmr-ablation"
    ] + flags
    
    register_experiment(
        full_name,
        TRAIN_SCRIPT,
        base_args,
        out_dir,
        f"Ablation: {desc}",
        required_files=["checkpoint_stage3_best.pth"]
    )


# --- Helper Functions ---

def print_status_table():
    """Print a pretty table of experiment statuses."""
    print(f"\n{'Experiment Name':<30} | {'Status':<15} | {'Description'}")
    print("-" * 80)
    for name, exp in EXPERIMENTS.items():
        status = exp.get_status()
        print(f"{name:<30} | {status:<15} | {exp.description}")
    print("-" * 80)

def generate_sbatch(job_name: str, cmd: List[str], log_dir: str = "logs") -> str:
    """Generate SLURM script content."""
    cmd_str = " ".join(cmd)
    
    # Escape special characters if needed, but basic joining works for most args
    
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={SLURM_DEFAULTS['partition']}
#SBATCH --nodes={SLURM_DEFAULTS['nodes']}
#SBATCH --ntasks-per-node={SLURM_DEFAULTS['ntasks_per_node']}
#SBATCH --gres={SLURM_DEFAULTS['gres']}
#SBATCH --cpus-per-task={SLURM_DEFAULTS['cpus_per_task']}
#SBATCH --mem={SLURM_DEFAULTS['mem']}
#SBATCH --time={SLURM_DEFAULTS['time']}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err

# Auto-generated by tmr.py
mkdir -p {log_dir}
module load pytorch/2.3.0-cuda12.1

echo "=== Auto-Queued Job: {job_name} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Command: {cmd_str}"

{cmd_str}

echo "=== Done ==="
"""
    return script

def submit_job(cmd: List[str], job_name: str, use_slurm: bool = False):
    """Run a command locally or submit to SLURM."""
    if use_slurm:
        print(f"Queuing to SLURM: {job_name}")
        
        # Ensure log directory exists before submission
        # SLURM requires the output directory to exist
        os.makedirs("logs", exist_ok=True)
        
        script_content = generate_sbatch(job_name, cmd)
        
        # Write to temp file for debugging/record
        os.makedirs("sbatch_queue", exist_ok=True)
        sbatch_path = Path(f"sbatch_queue/{job_name}.sbatch")
        with open(sbatch_path, "w") as f:
            f.write(script_content)
            
        # Submit
        res = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  -> Submitted: {res.stdout.strip()}")
        else:
            print(f"  -> Error submitting: {res.stderr}")
            sys.exit(1)
    else:
        print(f"Running locally: {job_name}")
        print(f"  Command: {' '.join(cmd)}")
        # For local run, we might want to just run it and wait, or detach?
        # The user said "run the tmr.py and it queues whatever i need".
        # If running locally, we probably want to run sequentially or just one.
        # We'll run blocking for local to avoid spawning 10 processes.
        subprocess.run(cmd, check=True)

def submit_chained_jobs(cmds: List[List[str]], job_name: str, use_slurm: bool = False):
    """Run multiple commands sequentially."""
    cmd_strings = [" ".join(cmd) for cmd in cmds]
    full_cmd_str = "\n".join(cmd_strings)
    
    if use_slurm:
        print(f"Queuing chained job to SLURM: {job_name}")
        os.makedirs("logs", exist_ok=True)
        
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={SLURM_DEFAULTS['partition']}
#SBATCH --nodes={SLURM_DEFAULTS['nodes']}
#SBATCH --ntasks-per-node={SLURM_DEFAULTS['ntasks_per_node']}
#SBATCH --gres={SLURM_DEFAULTS['gres']}
#SBATCH --cpus-per-task={SLURM_DEFAULTS['cpus_per_task']}
#SBATCH --mem={SLURM_DEFAULTS['mem']}
#SBATCH --time={SLURM_DEFAULTS['time']}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err

# Auto-generated by tmr.py
mkdir -p logs
module load pytorch/2.3.0-cuda12.1

echo "=== Auto-Queued Job: {job_name} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

set -e # Stop on error

{full_cmd_str}

echo "=== Done ==="
"""
        os.makedirs("sbatch_queue", exist_ok=True)
        sbatch_path = Path(f"sbatch_queue/{job_name}.sbatch")
        with open(sbatch_path, "w") as f:
            f.write(script)
            
        res = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  -> Submitted: {res.stdout.strip()}")
        else:
            print(f"  -> Error submitting: {res.stderr}")
            sys.exit(1)
    else:
        print(f"Running locally: {job_name}")
        for cmd in cmds:
            print(f"  Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

# --- Commands ---

def cmd_status(args):
    print(f"Environment: {'SLURM / HPC' if HAS_SLURM else 'Local / Windows'}")
    print(f"Hostname: {HOSTNAME}")
    print_status_table()

def cmd_train(args):
    """Train a specific experiment or custom args."""
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Error: Unknown experiment '{args.experiment}'")
            print("Available:", list(EXPERIMENTS.keys()))
            sys.exit(1)
        
        exp = EXPERIMENTS[args.experiment]
        
        # Construct command
        cmd = [sys.executable, str(exp.script)] + exp.args + args.extra
        
        # Decide execution mode
        use_slurm = HAS_SLURM and not args.local
        
        submit_job(cmd, args.experiment, use_slurm)
    else:
        # Legacy/Custom training mode
        # Reconstruct standard training command
        if not args.data_path:
            print("Error: --data_path required for custom training")
            sys.exit(1)
            
        cmd = [sys.executable, str(TRAIN_SCRIPT)]
        # Add all parsed args that are relevant (simplified for now, relying on extra)
        # This part is tricky if we want to support full flags mapping.
        # Better to advise using registered experiments or just pass through.
        pass
        # Implementation of full flag mapping skipped for brevity in favor of experiment-based workflow

def cmd_run_all(args):
    """Run all incomplete experiments."""
    print("Checking for incomplete experiments...")
    
    # 1. Baselines first
    base_exp = EXPERIMENTS["baselines_ar_ri"]
    if base_exp.get_status() != "Completed":
        print(f"Launching Baselines: {base_exp.name}")
        cmd = [sys.executable, str(base_exp.script)] + base_exp.args
        submit_job(cmd, base_exp.name, use_slurm=HAS_SLURM and not args.local)
    else:
        print("Baselines already completed.")

    # 2. Main Model
    # Since --no_lstm is in base_args, this 'tmr_stable' will now be stable + no_lstm
    stable_exp = EXPERIMENTS["tmr_stable"]
    # Check stable experiment args to confirm no_lstm is present (sanity check)
    if "--no_lstm" not in stable_exp.args:
        stable_exp.args.append("--no_lstm")
        
    if stable_exp.get_status() != "Completed":
        print(f"Launching Main Model: {stable_exp.name}")
        cmd = [sys.executable, str(stable_exp.script)] + stable_exp.args
        submit_job(cmd, stable_exp.name, use_slurm=HAS_SLURM and not args.local)
    else:
        print("Main Model already completed.")

    # 3. Ablations
    for name, exp in EXPERIMENTS.items():
        if name.startswith("tmr_ablation_") and exp.get_status() != "Completed":
            print(f"Launching Ablation: {name}")
            cmd = [sys.executable, str(exp.script)] + exp.args
            submit_job(cmd, name, use_slurm=HAS_SLURM and not args.local)

def cmd_eval_all(args):
    """Run evaluation for all completed TMR experiments."""
    print("Launching evaluations for completed experiments...")
    
    for name, exp in EXPERIMENTS.items():
        # Skip baselines
        if name == "baselines_ar_ri":
            continue
            
        if exp.get_status() == "Completed":
            # Find checkpoint
            # We know required_files has the checkpoint name
            ckpt_name = exp.required_files[0] # e.g., "checkpoint_stage3_best.pth"
            ckpt_path = exp.output_dir / ckpt_name
            
            if not ckpt_path.exists():
                print(f"Warning: {name} marked completed but checkpoint not found at {ckpt_path}")
                continue
                
            print(f"Launching Eval: {name}")
            
            # Extract data_path and dataset from exp.args if possible
            # args is a list of strings
            eval_args = ["--checkpoint", str(ckpt_path)]
            
            # Simple arg parsing from exp.args
            if "--data_path" in exp.args:
                try:
                    idx = exp.args.index("--data_path")
                    eval_args.extend(["--data_path", exp.args[idx+1]])
                except IndexError:
                    pass
            
            if "--dataset" in exp.args:
                try:
                    idx = exp.args.index("--dataset")
                    eval_args.extend(["--dataset", exp.args[idx+1]])
                except IndexError:
                    pass
                
            # Add extra user args
            eval_args.extend(args.extra)
            
            cmd = [sys.executable, str(EVAL_SCRIPT)] + eval_args
            job_name = f"eval_{name}"
            submit_job(cmd, job_name, use_slurm=HAS_SLURM and not args.local)

def cmd_cleanup(args):
    """Clean up intermediate checkpoints."""
    print("Cleaning up intermediate checkpoints...")
    for name, exp in EXPERIMENTS.items():
        if not exp.output_dir.exists():
            continue
            
        print(f"Checking {name}...")
        # Delete checkpoint_stage[12]_*.pth
        for stage in [1, 2]:
            for pth in exp.output_dir.glob(f"checkpoint_stage{stage}_*.pth"):
                print(f"  Deleting {pth.name}")
                pth.unlink()
        
        # Delete *_latest.pth
        for pth in exp.output_dir.glob("*_latest.pth"):
            print(f"  Deleting {pth.name}")
            pth.unlink()
            
    print("Cleanup complete.")

def cmd_run_pipeline(args):
    """
    Automated pipeline: Retarget -> Train Downstream (AR/RI) -> Eval
    For each completed TMR experiment:
    1. Generate retargeted dataset (if not exists)
    2. Train SGN AR/RI on it (if not exists)
    """
    print("Running Retargeting Pipeline...")
    
    for name, exp in EXPERIMENTS.items():
        if name == "baselines_ar_ri":
            continue
            
        if exp.get_status() != "Completed":
            continue
            
        ckpt_name = exp.required_files[0]
        ckpt_path = exp.output_dir / ckpt_name
        
        # 1. Generate Retargeted Dataset
        retargeted_dir = Path("output/retargeted_data")
        retargeted_dir.mkdir(parents=True, exist_ok=True)
        retargeted_pkl = retargeted_dir / f"{name}_retargeted.pkl"
        
        downstream_root = Path(f"output/downstream_{name}")
        expected_ar = downstream_root / "ntu_sgn_ar_paired" / "model_best.pth.tar"
        
        if expected_ar.exists():
            print(f"[{name}] Pipeline already completed.")
            continue
            
        print(f"[{name}] Queuing Pipeline (Gen -> Train)...")
        
        cmds = []
        
        # Cmd 1: Generate (if needed)
        # Note: If running on SLURM, we always include it unless file exists NOW. 
        # But even if it exists, re-running might be safer or we check inside script?
        # For simplicity, if pickle doesn't exist, we add the command.
        if not retargeted_pkl.exists():
            cmd_gen = [
                sys.executable, str(GEN_SCRIPT),
                "--checkpoint", str(ckpt_path),
                "--output_path", str(retargeted_pkl),
                "--dataset", "ntu"
            ]
            cmds.append(cmd_gen)
            
        # Cmd 2: Train
        cmd_train = [
            sys.executable, str(BASELINE_SCRIPT),
            "--data_path", str(retargeted_pkl),
            "--output_root", str(downstream_root),
            "--models", "sgn_ar", "sgn_ri",
            "--epochs", "60"
        ]
        cmds.append(cmd_train)
        
        submit_chained_jobs(cmds, f"pipe_{name}", use_slurm=HAS_SLURM and not args.local)


def main():
    parser = argparse.ArgumentParser(description="Disentangled TMR Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Status
    subparsers.add_parser("status", help="Show experiment status")

    # Train
    train_parser = subparsers.add_parser("train", help="Run a training experiment")
    train_parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()), help="Named experiment to run")
    train_parser.add_argument("--local", action="store_true", help="Force local execution even if SLURM is present")
    train_parser.add_argument("--data_path", help="Override data path (for custom runs)")
    train_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra flags passed to script")

    # Run All
    run_all_parser = subparsers.add_parser("run-all", help="Queue all missing experiments")
    run_all_parser.add_argument("--local", action="store_true", help="Force local execution")

    # Eval (Wrapper)
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--local", action="store_true")
    eval_parser.add_argument("extra", nargs=argparse.REMAINDER)
    
    # Eval All
    eval_all_parser = subparsers.add_parser("eval-all", help="Run evaluation for all completed experiments")
    eval_all_parser.add_argument("--local", action="store_true")
    eval_all_parser.add_argument("extra", nargs=argparse.REMAINDER)
    
    # Cleanup
    subparsers.add_parser("clean", help="Clean up intermediate checkpoints")

    # Pipeline
    pipeline_parser = subparsers.add_parser("run-pipeline", help="Run Retarget -> Train Downstream pipeline")
    pipeline_parser.add_argument("--local", action="store_true")
    
    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    elif args.command == "eval":
        # Simple wrapper for now
        # Filter out '--' from extra args if present
        extra_args = [arg for arg in args.extra if arg != "--"]
        cmd = [sys.executable, str(EVAL_SCRIPT), "--checkpoint", args.checkpoint] + extra_args
        submit_job(cmd, "tmr_eval", use_slurm=HAS_SLURM and not args.local)
    elif args.command == "eval-all":
        cmd_eval_all(args)
    elif args.command == "clean":
        cmd_cleanup(args)
    elif args.command == "run-pipeline":
        cmd_run_pipeline(args)

if __name__ == "__main__":
    main()
