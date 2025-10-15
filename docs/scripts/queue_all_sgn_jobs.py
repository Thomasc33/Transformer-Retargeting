#!/usr/bin/env python3
"""
Queue All SGN Training Jobs

This script generates and submits all required SGN training jobs:
- AR: NTU60 CS/CV, NTU120 CS/CV  
- RI: NTU60 CV, NTU120 CV (No CS for RI as requested)
- GC: NTU60 CS/CV, NTU120 CS/CV

Usage:
    python queue_all_sgn_jobs.py
"""

import os
import subprocess
import time

def run_command(cmd):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    """Generate and queue all SGN training jobs."""
    
    # Define all required jobs
    jobs = [
        # AR (Action Recognition) - All datasets and settings
        ('ntu', 'cv', 'ar'),      # NTU60 CV AR
        ('ntu120', 'cs', 'ar'),   # NTU120 CS AR  
        ('ntu120', 'cv', 'ar'),   # NTU120 CV AR
        
        # RI (Re-identification) - Only CV settings (No CS as requested)
        ('ntu', 'cv', 'ri'),      # NTU60 CV RI
        ('ntu120', 'cv', 'ri'),   # NTU120 CV RI
        
        # GC (Gender Classification) - All datasets and settings
        ('ntu', 'cs', 'gc'),      # NTU60 CS GC
        ('ntu', 'cv', 'gc'),      # NTU60 CV GC
        ('ntu120', 'cs', 'gc'),   # NTU120 CS GC
        ('ntu120', 'cv', 'gc'),   # NTU120 CV GC
    ]
    
    print("🚀 Queueing All SGN Training Jobs")
    print("=" * 50)
    print(f"Total jobs to create: {len(jobs)}")
    print("")
    
    successful_jobs = []
    failed_jobs = []
    
    for i, (dataset, setting, task) in enumerate(jobs, 1):
        print(f"📝 [{i}/{len(jobs)}] Creating SGN {task.upper()} job for {dataset.upper()} {setting.upper()}...")
        
        # Generate training script and SLURM job
        cmd = f"python train_sgn.py --dataset {dataset} --setting {setting} --task {task} --slurm"
        success, stdout, stderr = run_command(cmd)
        
        if not success:
            print(f"❌ Failed to generate job: {stderr}")
            failed_jobs.append((dataset, setting, task, "Generation failed"))
            continue
        
        # Submit the job
        job_script = f"train_sgn_{task}_{dataset}_{setting}.bash"
        if not os.path.exists(job_script):
            print(f"❌ Job script not found: {job_script}")
            failed_jobs.append((dataset, setting, task, "Script not found"))
            continue
        
        print(f"🎯 Submitting job: {job_script}")
        submit_cmd = f"sbatch {job_script}"
        success, stdout, stderr = run_command(submit_cmd)
        
        if success:
            job_id = stdout.strip().split()[-1] if stdout.strip() else "unknown"
            print(f"✅ Job submitted successfully: {job_id}")
            successful_jobs.append((dataset, setting, task, job_id))
        else:
            print(f"❌ Failed to submit job: {stderr}")
            failed_jobs.append((dataset, setting, task, f"Submission failed: {stderr}"))
        
        print("")
        
        # Small delay between submissions
        time.sleep(2)
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"✅ Successful jobs: {len(successful_jobs)}")
    print(f"❌ Failed jobs: {len(failed_jobs)}")
    print("")
    
    if successful_jobs:
        print("✅ Successfully submitted jobs:")
        for dataset, setting, task, job_id in successful_jobs:
            print(f"  - SGN {task.upper()} {dataset.upper()} {setting.upper()}: Job {job_id}")
        print("")
    
    if failed_jobs:
        print("❌ Failed jobs:")
        for dataset, setting, task, error in failed_jobs:
            print(f"  - SGN {task.upper()} {dataset.upper()} {setting.upper()}: {error}")
        print("")
    
    print("🎉 All jobs processed!")
    print("")
    print("💡 To check job status: squeue -u $USER")
    print("💡 To check logs: ls -la logs/sgn_*")

if __name__ == '__main__':
    main()
