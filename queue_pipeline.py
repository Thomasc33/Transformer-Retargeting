#!/usr/bin/env python3
"""
Queue the complete evaluation and training pipeline with proper SLURM job dependencies.
"""

import subprocess
import sys
import time
from datetime import datetime

def submit_job(job_name, command, dependencies=None, gpus=1, memory="32GB", time_limit="12:00:00", cpus=2):
    """Submit a SLURM job and return the job ID."""
    
    # Build sbatch command
    sbatch_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "--nodes=1",
        f"--gres=gpu:{gpus}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        "--partition=GPU",
        "--mail-user=tcarr23@charlotte.edu",
        "--mail-type=FAIL",
        f"--output=/users/tcarr23/Transformer-Retargeting/logs/slurm/{job_name}-%j.out"
    ]
    
    # Add dependencies if specified
    if dependencies:
        if isinstance(dependencies, list):
            dep_str = ":".join(map(str, dependencies))
        else:
            dep_str = str(dependencies)
        sbatch_cmd.append(f"--dependency=afterok:{dep_str}")
    
    # Add the command to execute
    sbatch_cmd.append(f"--wrap={command}")
    
    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"✅ {job_name}: Job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to submit {job_name}: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return None

def main():
    print("🚀 Queuing Complete Evaluation and Training Pipeline")
    print("=" * 60)
    
    # Step 1: Submit all 16 evaluation jobs (1 GPU, 32GB each)
    print("\n📊 Step 1: Submitting 16 evaluation jobs...")
    
    evaluation_experiments = [
        "privacy_utility_sgn",
        "privacy_utility_mixformer", 
        "baseline_comparison",
        "physical_plausibility",
        "cross_dataset_validation",
        "loss_ablation",
        "loss_weight_sensitivity",
        "masking_configurations",
        "pretraining_approaches",
        "training_stability",
        "teacher_forcing_analysis",
        "per_class_analysis",
        "per_subject_analysis",
        "efficiency_analysis",
        "motion_visualizations",
        "attention_visualization"
    ]
    
    eval_job_ids = []
    for exp in evaluation_experiments:
        command = f"set -euo pipefail; module load pytorch/2.3.0-cuda12.1 || true; cd /users/tcarr23/Transformer-Retargeting; echo '[EVAL] {exp} start $(date)'; python tmr.py eval --one {exp}; echo '[EVAL] {exp} done $(date)'"
        
        job_id = submit_job(
            job_name=f"eval-{exp}",
            command=command,
            gpus=1,
            memory="32GB",
            time_limit="12:00:00",
            cpus=2
        )
        
        if job_id:
            eval_job_ids.append(job_id)
        else:
            print(f"❌ Failed to submit evaluation job for {exp}")
            return 1
    
    print(f"\n✅ Submitted {len(eval_job_ids)} evaluation jobs")
    print(f"   Job IDs: {', '.join(eval_job_ids)}")
    
    # Step 2: Submit dashboard update job (depends on all evaluations)
    print("\n📈 Step 2: Submitting dashboard update job...")
    
    dash_command = "set -euo pipefail; module load pytorch/2.3.0-cuda12.1 || true; cd /users/tcarr23/Transformer-Retargeting; echo '[DASH] Dashboard update start $(date)'; python tmr.py dash; echo '[DASH] Dashboard update done $(date)'"
    
    dash_job_id = submit_job(
        job_name="dash-post-eval",
        command=dash_command,
        dependencies=eval_job_ids,
        gpus=0,
        memory="16GB",
        time_limit="01:00:00",
        cpus=1
    )
    
    if not dash_job_id:
        print("❌ Failed to submit dashboard update job")
        return 1
    
    # Step 3: Submit training job (depends on dashboard update)
    print("\n🏋️ Step 3: Submitting training job...")
    
    train_command = "set -euo pipefail; module load pytorch/2.3.0-cuda12.1 || true; cd /users/tcarr23/Transformer-Retargeting; echo '[TRAIN] Training start $(date)'; python src/training/main.py --resume-from /users/tcarr23/Transformer-Retargeting/data/models_output/model.pth --output-model-path /users/tcarr23/Transformer-Retargeting/data/models_output/model_continued_$(date +%Y%m%d_%H%M%S).pth --teacher-forcing-ratio 0.0 --epochs 5 --validate-every 1; echo '[TRAIN] Training done $(date)'"
    
    train_job_id = submit_job(
        job_name="train-continue",
        command=train_command,
        dependencies=[dash_job_id],
        gpus=4,
        memory="128GB",
        time_limit="240:00:00",  # 10 days
        cpus=8
    )
    
    if not train_job_id:
        print("❌ Failed to submit training job")
        return 1
    
    # Step 4: Submit post-training evaluation (depends on training)
    print("\n📊 Step 4: Submitting post-training evaluation jobs...")
    
    post_eval_job_ids = []
    for exp in evaluation_experiments:
        command = f"set -euo pipefail; module load pytorch/2.3.0-cuda12.1 || true; cd /users/tcarr23/Transformer-Retargeting; echo '[POST-EVAL] {exp} start $(date)'; python tmr.py eval --one {exp}; echo '[POST-EVAL] {exp} done $(date)'"
        
        job_id = submit_job(
            job_name=f"post-eval-{exp}",
            command=command,
            dependencies=[train_job_id],
            gpus=1,
            memory="32GB",
            time_limit="12:00:00",
            cpus=2
        )
        
        if job_id:
            post_eval_job_ids.append(job_id)
        else:
            print(f"❌ Failed to submit post-training evaluation job for {exp}")
            return 1
    
    print(f"\n✅ Submitted {len(post_eval_job_ids)} post-training evaluation jobs")
    
    # Step 5: Submit final dashboard update (depends on all post-training evaluations)
    print("\n📈 Step 5: Submitting final dashboard update...")
    
    final_dash_command = "set -euo pipefail; module load pytorch/2.3.0-cuda12.1 || true; cd /users/tcarr23/Transformer-Retargeting; echo '[FINAL-DASH] Final dashboard update start $(date)'; python tmr.py dash; echo '[FINAL-DASH] Final dashboard update done $(date)'"
    
    final_dash_job_id = submit_job(
        job_name="dash-final",
        command=final_dash_command,
        dependencies=post_eval_job_ids,
        gpus=0,
        memory="16GB",
        time_limit="01:00:00",
        cpus=1
    )
    
    if not final_dash_job_id:
        print("❌ Failed to submit final dashboard update job")
        return 1
    
    # Summary
    print("\n🎉 PIPELINE SUCCESSFULLY QUEUED!")
    print("=" * 60)
    print(f"📊 Pre-training evaluations: {len(eval_job_ids)} jobs")
    print(f"📈 Dashboard update: 1 job ({dash_job_id})")
    print(f"🏋️ Training continuation: 1 job ({train_job_id})")
    print(f"📊 Post-training evaluations: {len(post_eval_job_ids)} jobs")
    print(f"📈 Final dashboard: 1 job ({final_dash_job_id})")
    print(f"\n📧 Email notifications: tcarr23@charlotte.edu (failures only)")
    print(f"📁 Logs: /users/tcarr23/Transformer-Retargeting/logs/slurm/")
    
    print("\n⏱️ ESTIMATED TIMELINE:")
    print("1. Pre-training evaluations: ~2-3 hours")
    print("2. Training continuation: ~10 days (5 epochs)")
    print("3. Post-training evaluations: ~2-3 hours")
    print("4. Final dashboard: ~5 minutes")
    print("\n🏖️ Everything will be ready when you return from your honeymoon!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
