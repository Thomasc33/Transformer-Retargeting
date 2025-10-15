"""
SLURM Job Management for TMR
Handles job submission, tracking, and dependency management
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .utils import (
    print_header, print_success, print_error, print_warning, print_info,
    get_root_dir, load_jobs, save_jobs, parse_slurm_output, get_slurm_status
)


class SlurmManager:
    """Manages SLURM job submission and tracking"""
    
    def __init__(self, user_email: str = "tcarr23@charlotte.edu"):
        self.root = get_root_dir()
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.user_email = user_email
        self.jobs_data = load_jobs()
    
    def generate_slurm_script(
        self,
        job_name: str,
        command: str,
        num_gpus: int = 1,
        time_hours: int = 24,
        mem_gb: int = 32,
        partition: str = "gpu",
        dependencies: Optional[List[str]] = None,
        array: Optional[str] = None
    ) -> str:
        """Generate SLURM batch script"""
        
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={self.logs_dir}/{job_name}_%j.out
#SBATCH --error={self.logs_dir}/{job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={time_hours}:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={self.user_email}
"""
        
        # Add dependencies if specified
        if dependencies:
            dep_str = ":".join(dependencies)
            script += f"#SBATCH --dependency=afterok:{dep_str}\n"
        
        # Add array if specified
        if array:
            script += f"#SBATCH --array={array}\n"
        
        script += f"""
# Load modules
module load pytorch/2.3.0-cuda12.1

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: {job_name}"
echo "Node: $SLURM_NODELIST"
echo "GPUs: {num_gpus}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Change to project directory
cd {self.root}

# Run command
{command}

# Print completion
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
"""
        
        return script
    
    def submit_job(
        self,
        job_name: str,
        command: str,
        num_gpus: int = 1,
        time_hours: int = 24,
        mem_gb: int = 32,
        partition: str = "gpu",
        dependencies: Optional[List[str]] = None,
        array: Optional[str] = None,
        dry_run: bool = False
    ) -> Optional[str]:
        """Submit a SLURM job"""
        
        # Generate script
        script = self.generate_slurm_script(
            job_name=job_name,
            command=command,
            num_gpus=num_gpus,
            time_hours=time_hours,
            mem_gb=mem_gb,
            partition=partition,
            dependencies=dependencies,
            array=array
        )
        
        # Save script
        script_path = self.logs_dir / f"{job_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        print_info(f"Generated SLURM script: {script_path}")
        
        if dry_run:
            print_warning("Dry run mode - not submitting job")
            print(script)
            return None
        
        # Submit job
        try:
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                job_id = parse_slurm_output(result.stdout)
                if job_id:
                    print_success(f"Submitted job {job_id}: {job_name}")
                    
                    # Track job
                    self._track_job(
                        job_id=job_id,
                        job_name=job_name,
                        command=command,
                        num_gpus=num_gpus,
                        dependencies=dependencies
                    )
                    
                    return job_id
                else:
                    print_error(f"Failed to parse job ID from: {result.stdout}")
                    return None
            else:
                print_error(f"Failed to submit job: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print_error("Job submission timed out")
            return None
        except FileNotFoundError:
            print_error("sbatch command not found - are you on an HPC system?")
            return None
    
    def _track_job(
        self,
        job_id: str,
        job_name: str,
        command: str,
        num_gpus: int,
        dependencies: Optional[List[str]] = None
    ):
        """Track a submitted job"""
        job_info = {
            "job_id": job_id,
            "job_name": job_name,
            "command": command,
            "num_gpus": num_gpus,
            "dependencies": dependencies or [],
            "submitted_at": datetime.now().isoformat(),
            "status": "submitted"
        }
        
        if "jobs" not in self.jobs_data:
            self.jobs_data["jobs"] = []
        
        self.jobs_data["jobs"].append(job_info)
        save_jobs(self.jobs_data)
    
    def check_job_status(self, job_id: str) -> Optional[str]:
        """Check status of a SLURM job"""
        return get_slurm_status(job_id)
    
    def list_jobs(self, status_filter: Optional[str] = None):
        """List tracked jobs"""
        jobs = self.jobs_data.get("jobs", [])
        
        if not jobs:
            print_info("No jobs tracked yet")
            return
        
        print_header("Tracked Jobs")
        
        for job in jobs:
            job_id = job["job_id"]
            job_name = job["job_name"]
            submitted_at = job["submitted_at"]
            
            # Check current status
            current_status = self.check_job_status(job_id)
            if current_status:
                status_str = f"RUNNING ({current_status})"
            else:
                status_str = "COMPLETED/FAILED"
            
            if status_filter and status_filter.lower() not in status_str.lower():
                continue
            
            print(f"  Job ID: {job_id}")
            print(f"  Name: {job_name}")
            print(f"  Status: {status_str}")
            print(f"  Submitted: {submitted_at}")
            print(f"  GPUs: {job['num_gpus']}")
            if job.get("dependencies"):
                print(f"  Dependencies: {', '.join(job['dependencies'])}")
            print()
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a SLURM job"""
        try:
            result = subprocess.run(
                ['scancel', job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print_success(f"Cancelled job {job_id}")
                return True
            else:
                print_error(f"Failed to cancel job: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print_error(f"Error cancelling job: {e}")
            return False
    
    def get_queue_status(self):
        """Get current queue status"""
        try:
            result = subprocess.run(
                ['squeue', '-u', os.environ.get('USER', 'tcarr23')],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print_header("Current Queue Status")
                print(result.stdout)
            else:
                print_error("Failed to get queue status")
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print_error(f"Error getting queue status: {e}")
    
    def submit_job_chain(
        self,
        jobs: List[Dict[str, Any]],
        dry_run: bool = False
    ) -> List[str]:
        """Submit a chain of dependent jobs"""
        job_ids = []
        
        print_header("Submitting Job Chain")
        print_info(f"Total jobs: {len(jobs)}")
        
        for i, job_config in enumerate(jobs, 1):
            print(f"\n[{i}/{len(jobs)}] Submitting: {job_config['job_name']}")
            
            # Add dependencies from previous jobs if specified
            if job_config.get("depends_on_previous", False) and job_ids:
                job_config["dependencies"] = [job_ids[-1]]
            
            job_id = self.submit_job(
                job_name=job_config["job_name"],
                command=job_config["command"],
                num_gpus=job_config.get("num_gpus", 1),
                time_hours=job_config.get("time_hours", 24),
                mem_gb=job_config.get("mem_gb", 32),
                partition=job_config.get("partition", "gpu"),
                dependencies=job_config.get("dependencies"),
                array=job_config.get("array"),
                dry_run=dry_run
            )
            
            if job_id:
                job_ids.append(job_id)
            else:
                print_error(f"Failed to submit job: {job_config['job_name']}")
                if not dry_run:
                    break
        
        if job_ids:
            print_success(f"\nSubmitted {len(job_ids)} jobs successfully")
            print_info(f"Job IDs: {', '.join(job_ids)}")
        
        return job_ids

