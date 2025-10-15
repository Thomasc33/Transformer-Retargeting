"""
Slurm job runner for HPC execution of experiments.
"""

import os
import subprocess
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


class SlurmRunner:
    """
    Manages Slurm job submission and monitoring for experiments.
    """

    def __init__(self, hpc_config: Dict[str, Any]):
        """
        Initialize Slurm runner.

        Args:
            hpc_config: HPC configuration dictionary
        """
        self.config = hpc_config
        self.logger = logging.getLogger(__name__)
        self.job_dir = Path("eval/suite/runners/jobs")
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def submit_experiment(self, experiment_name: str, experiment_config: Dict[str, Any]) -> Optional[str]:
        """
        Submit an experiment to Slurm.

        Args:
            experiment_name: Name of the experiment
            experiment_config: Experiment configuration

        Returns:
            Job ID if successful, None otherwise
        """
        try:
            # Generate job script
            job_script_path = self.generate_job_script(experiment_name, experiment_config)

            # Submit job
            cmd = ["sbatch", str(job_script_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Extract job ID from output
                job_id = result.stdout.strip().split()[-1]
                self.logger.info(f"Submitted job {job_id} for experiment {experiment_name}")

                # Save job info
                self.save_job_info(job_id, experiment_name, experiment_config, job_script_path)

                return job_id
            else:
                self.logger.error(f"Failed to submit job: {result.stderr}")
                return None

        except Exception as e:
            self.logger.error(f"Error submitting experiment {experiment_name}: {str(e)}")
            return None

    def generate_job_script(self, experiment_name: str, experiment_config: Dict[str, Any]) -> Path:
        """
        Generate Slurm job script for an experiment.

        Args:
            experiment_name: Name of the experiment
            experiment_config: Experiment configuration

        Returns:
            Path to the generated job script
        """
        # Determine job template based on experiment type
        job_template = self.get_job_template(experiment_config)

        # Generate script content
        script_content = self.generate_script_content(experiment_name, experiment_config, job_template)

        # Save script
        script_path = self.job_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sbatch"
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable
        os.chmod(script_path, 0o755)

        return script_path

    def get_job_template(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get appropriate job template based on experiment configuration."""
        # Estimate job requirements based on experiment
        estimated_time = experiment_config.get('estimated_time', '12:00:00')

        # Convert time estimate to template type
        if 'hour' in estimated_time:
            hours = int(estimated_time.split()[0])
            if hours <= 4:
                template_type = 'quick'
            elif hours <= 12:
                template_type = 'standard'
            else:
                template_type = 'long'
        else:
            template_type = 'standard'

        # Check for multi-seed experiments
        if 'seeds' in experiment_config or 'num_seeds' in experiment_config:
            template_type = 'multi_seed'

        return self.config.get('job_templates', {}).get(template_type, self.config.get('job_templates', {}).get('standard', {}))

    def generate_script_content(self, experiment_name: str, experiment_config: Dict[str, Any],
                              job_template: Dict[str, Any]) -> str:
        """Generate the actual Slurm script content."""

        # Default values based on train.bash
        defaults = {
            'partition': self.config.get('default_partition', 'GPU'),
            'time': self.config.get('default_time', '240:00:00'),
            'nodes': self.config.get('default_nodes', 1),
            'ntasks_per_node': self.config.get('default_ntasks_per_node', 1),
            'gres': self.config.get('default_gres', 'gpu:1'),
            'mem': self.config.get('default_mem', '64GB')
        }

        # Override with template values
        for key, value in job_template.items():
            if key in defaults:
                defaults[key] = value

        # Generate script
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={experiment_name}",
            f"#SBATCH --partition={defaults['partition']}",
            f"#SBATCH --time={defaults['time']}",
            f"#SBATCH --nodes={defaults['nodes']}",
            f"#SBATCH --ntasks-per-node={defaults['ntasks_per_node']}",
            f"#SBATCH --gres={defaults['gres']}",
            f"#SBATCH --mem={defaults['mem']}",
            f"#SBATCH --output=logs/slurm/{experiment_name}_%j.out",
            f"#SBATCH --error=logs/slurm/{experiment_name}_%j.err",
            "",
            "# Initialize module environment",
            "source /etc/profile || true",
            "if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi",
            "",
            "# Load modules and activate environment",
            "module purge || true",
            "module load pytorch/2.3.0-cuda12.1",
            "",
            "# Diagnostics",
            "which python || true",
            "python -V || true",
            "echo \"Is CUDA Available?\"",
            "python -c 'import torch, sys; print(sys.executable); print(torch.cuda.is_available())' || true",
            "echo \"\"",
            "echo \"nvidia-smi output:\"",
            "nvidia-smi || true",
            "",
            "# Set environment variables",
            "export OMP_NUM_THREADS=1",
            "export PYTHONUNBUFFERED=1",
            "",
            "# Change to project directory",
            "cd $SLURM_SUBMIT_DIR",
            "",
            "# Create output directories",
            "mkdir -p logs/slurm",
            "mkdir -p results/experiments",
            "",
            "# Run the experiment locally on the HPC node",
            f"python tmr.py eval --one {experiment_name} --local",
            "",
            "# Refresh dashboard",
            "python tmr.py dash || true",
            "",
            "echo \"Experiment completed at $(date)\""
        ]

        # Add array job configuration if needed
        if 'array' in job_template:
            script_lines.insert(10, f"#SBATCH --array={job_template['array']}")

        return "\n".join(script_lines)

    def save_job_info(self, job_id: str, experiment_name: str, experiment_config: Dict[str, Any],
                     script_path: Path):
        """Save job information for monitoring."""
        job_info = {
            'job_id': job_id,
            'experiment_name': experiment_name,
            'experiment_config': experiment_config,
            'script_path': str(script_path),
            'submit_time': datetime.now().isoformat(),
            'status': 'submitted'
        }

        job_info_path = self.job_dir / f"job_{job_id}.json"
        with open(job_info_path, 'w') as f:
            json.dump(job_info, f, indent=2, default=str)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a Slurm job."""
        try:
            cmd = ["squeue", "-j", job_id, "--format=%T,%R", "--noheader"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                status_line = result.stdout.strip()
                if ',' in status_line:
                    status, reason = status_line.split(',', 1)
                else:
                    status, reason = status_line, ""

                return {
                    'job_id': job_id,
                    'status': status.strip(),
                    'reason': reason.strip(),
                    'found': True
                }
            else:
                # Job not found in queue, check if completed
                return self.check_completed_job(job_id)

        except Exception as e:
            self.logger.error(f"Error checking job status for {job_id}: {str(e)}")
            return {'job_id': job_id, 'status': 'unknown', 'found': False}

    def check_completed_job(self, job_id: str) -> Dict[str, Any]:
        """Check if a job has completed using sacct."""
        try:
            cmd = ["sacct", "-j", job_id, "--format=State,ExitCode", "--noheader", "--parsable2"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if '|' in line:
                        state, exit_code = line.split('|', 1)
                        return {
                            'job_id': job_id,
                            'status': state.strip(),
                            'exit_code': exit_code.strip(),
                            'found': True
                        }

            return {'job_id': job_id, 'status': 'not_found', 'found': False}

        except Exception as e:
            self.logger.error(f"Error checking completed job {job_id}: {str(e)}")
            return {'job_id': job_id, 'status': 'unknown', 'found': False}

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a Slurm job."""
        try:
            cmd = ["scancel", job_id]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(f"Cancelled job {job_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel job {job_id}: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error cancelling job {job_id}: {str(e)}")
            return False

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs submitted by this runner."""
        jobs = []

        for job_file in self.job_dir.glob("job_*.json"):
            try:
                with open(job_file, 'r') as f:
                    job_info = json.load(f)

                # Get current status
                current_status = self.get_job_status(job_info['job_id'])
                job_info.update(current_status)

                jobs.append(job_info)

            except Exception as e:
                self.logger.warning(f"Error reading job file {job_file}: {str(e)}")

        return sorted(jobs, key=lambda x: x.get('submit_time', ''))

    def submit_experiment_set(self, set_name: str, experiments: List[str],
                            experiment_configs: Dict[str, Any]) -> List[str]:
        """Submit multiple experiments as a set."""
        job_ids = []

        self.logger.info(f"Submitting experiment set: {set_name}")

        for experiment_name in experiments:
            if experiment_name in experiment_configs:
                job_id = self.submit_experiment(experiment_name, experiment_configs[experiment_name])
                if job_id:
                    job_ids.append(job_id)

        self.logger.info(f"Submitted {len(job_ids)} jobs for experiment set {set_name}")
        return job_ids
