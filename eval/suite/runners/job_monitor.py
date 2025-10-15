"""
Job monitoring utilities for tracking experiment progress.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta


class JobMonitor:
    """
    Monitors job progress and provides status updates.
    """
    
    def __init__(self, slurm_runner=None):
        """Initialize job monitor."""
        self.slurm_runner = slurm_runner
        self.logger = logging.getLogger(__name__)
        self.job_dir = Path("evaluation_suite/runners/jobs")  # Keep job state here
        self.results_root = Path("results")
        
    def get_all_jobs_status(self) -> List[Dict[str, Any]]:
        """Get status of all jobs."""
        if self.slurm_runner:
            return self.slurm_runner.list_jobs()
        else:
            return self.get_local_jobs_status()
            
    def get_local_jobs_status(self) -> List[Dict[str, Any]]:
        """Get status of local jobs (non-Slurm)."""
        jobs = []
        
        # Check for completed experiments
        results_dir = self.results_root / "experiments"
        if results_dir.exists():
            for exp_dir in results_dir.iterdir():
                if exp_dir.is_dir():
                    # Check for recent results
                    result_files = list(exp_dir.glob("*/results.json"))
                    if result_files:
                        latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                        
                        try:
                            with open(latest_result, 'r') as f:
                                result_data = json.load(f)
                                
                            job_info = {
                                'experiment_name': exp_dir.name,
                                'status': 'COMPLETED',
                                'submit_time': result_data.get('timestamp', 'Unknown'),
                                'duration': result_data.get('duration', 0),
                                'found': True
                            }
                            jobs.append(job_info)
                            
                        except Exception as e:
                            self.logger.warning(f"Error reading result file {latest_result}: {e}")
                            
        return jobs
        
    def monitor_jobs(self, job_ids: List[str], check_interval: int = 60) -> Dict[str, str]:
        """
        Monitor a list of jobs until completion.
        
        Args:
            job_ids: List of job IDs to monitor
            check_interval: Time between status checks in seconds
            
        Returns:
            Final status of each job
        """
        if not self.slurm_runner:
            self.logger.warning("No Slurm runner available for job monitoring")
            return {}
            
        final_status = {}
        remaining_jobs = set(job_ids)
        
        self.logger.info(f"Monitoring {len(job_ids)} jobs...")
        
        while remaining_jobs:
            completed_jobs = set()
            
            for job_id in remaining_jobs:
                status_info = self.slurm_runner.get_job_status(job_id)
                status = status_info.get('status', 'UNKNOWN')
                
                if status in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                    final_status[job_id] = status
                    completed_jobs.add(job_id)
                    self.logger.info(f"Job {job_id} completed with status: {status}")
                elif status in ['RUNNING', 'PENDING']:
                    self.logger.debug(f"Job {job_id} status: {status}")
                else:
                    self.logger.warning(f"Job {job_id} has unknown status: {status}")
                    
            # Remove completed jobs from monitoring
            remaining_jobs -= completed_jobs
            
            if remaining_jobs:
                self.logger.info(f"Still monitoring {len(remaining_jobs)} jobs...")
                time.sleep(check_interval)
                
        self.logger.info("All jobs completed!")
        return final_status
        
    def get_job_summary(self) -> Dict[str, Any]:
        """Get summary of all jobs."""
        jobs = self.get_all_jobs_status()
        
        summary = {
            'total_jobs': len(jobs),
            'completed': 0,
            'running': 0,
            'pending': 0,
            'failed': 0,
            'unknown': 0,
            'experiments': {}
        }
        
        for job in jobs:
            status = job.get('status', 'UNKNOWN').upper()
            
            if status in ['COMPLETED', 'FINISHED']:
                summary['completed'] += 1
            elif status == 'RUNNING':
                summary['running'] += 1
            elif status == 'PENDING':
                summary['pending'] += 1
            elif status in ['FAILED', 'CANCELLED', 'TIMEOUT']:
                summary['failed'] += 1
            else:
                summary['unknown'] += 1
                
            # Track by experiment
            exp_name = job.get('experiment_name', 'unknown')
            if exp_name not in summary['experiments']:
                summary['experiments'][exp_name] = {
                    'status': status,
                    'submit_time': job.get('submit_time', 'Unknown'),
                    'duration': job.get('duration', 0)
                }
                
        return summary
        
    def print_status_report(self):
        """Print a formatted status report."""
        summary = self.get_job_summary()
        
        print("\n" + "="*60)
        print("📊 EXPERIMENT STATUS REPORT")
        print("="*60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Overall statistics
        print("📈 Overall Statistics:")
        print(f"  Total Jobs: {summary['total_jobs']}")
        print(f"  ✅ Completed: {summary['completed']}")
        print(f"  🏃 Running: {summary['running']}")
        print(f"  ⏳ Pending: {summary['pending']}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  ❓ Unknown: {summary['unknown']}")
        
        if summary['total_jobs'] > 0:
            completion_rate = (summary['completed'] / summary['total_jobs']) * 100
            print(f"  📊 Completion Rate: {completion_rate:.1f}%")
        
        print()
        
        # Experiment details
        if summary['experiments']:
            print("🧪 Experiment Details:")
            for exp_name, exp_info in summary['experiments'].items():
                status = exp_info['status']
                status_emoji = {
                    'COMPLETED': '✅',
                    'RUNNING': '🏃',
                    'PENDING': '⏳',
                    'FAILED': '❌',
                    'CANCELLED': '🚫',
                    'TIMEOUT': '⏰'
                }.get(status, '❓')
                
                print(f"  {status_emoji} {exp_name}: {status}")
                
                if exp_info.get('duration'):
                    duration = exp_info['duration']
                    if isinstance(duration, (int, float)):
                        duration_str = f"{duration:.1f}s"
                    else:
                        duration_str = str(duration)
                    print(f"    Duration: {duration_str}")
                    
        print("="*60)
        
    def wait_for_completion(self, experiment_names: List[str], 
                          timeout_hours: int = 24, check_interval: int = 300) -> bool:
        """
        Wait for specific experiments to complete.
        
        Args:
            experiment_names: List of experiment names to wait for
            timeout_hours: Maximum time to wait in hours
            check_interval: Time between checks in seconds
            
        Returns:
            True if all experiments completed successfully, False otherwise
        """
        start_time = datetime.now()
        timeout_time = start_time + timedelta(hours=timeout_hours)
        
        self.logger.info(f"Waiting for experiments to complete: {experiment_names}")
        self.logger.info(f"Timeout: {timeout_hours} hours")
        
        while datetime.now() < timeout_time:
            summary = self.get_job_summary()
            
            # Check if all target experiments are completed
            all_completed = True
            for exp_name in experiment_names:
                if exp_name in summary['experiments']:
                    status = summary['experiments'][exp_name]['status']
                    if status not in ['COMPLETED', 'FINISHED']:
                        all_completed = False
                        break
                else:
                    all_completed = False
                    break
                    
            if all_completed:
                self.logger.info("All target experiments completed!")
                return True
                
            # Print periodic status update
            elapsed = datetime.now() - start_time
            self.logger.info(f"Still waiting... (elapsed: {elapsed})")
            self.print_status_report()
            
            time.sleep(check_interval)
            
        self.logger.warning(f"Timeout reached after {timeout_hours} hours")
        return False
        
    def cleanup_old_jobs(self, days_old: int = 7):
        """Clean up old job files."""
        if not self.job_dir.exists():
            return
            
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        for job_file in self.job_dir.glob("job_*.json"):
            try:
                file_time = datetime.fromtimestamp(job_file.stat().st_mtime)
                if file_time < cutoff_time:
                    job_file.unlink()
                    self.logger.info(f"Cleaned up old job file: {job_file}")
            except Exception as e:
                self.logger.warning(f"Error cleaning up {job_file}: {e}")
                
    def export_job_history(self, output_path: Path):
        """Export job history to CSV file."""
        jobs = self.get_all_jobs_status()
        
        if not jobs:
            self.logger.warning("No job history to export")
            return
            
        # Prepare data for CSV
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['experiment_name', 'job_id', 'status', 'submit_time', 'duration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for job in jobs:
                writer.writerow({
                    'experiment_name': job.get('experiment_name', ''),
                    'job_id': job.get('job_id', ''),
                    'status': job.get('status', ''),
                    'submit_time': job.get('submit_time', ''),
                    'duration': job.get('duration', '')
                })
                
        self.logger.info(f"Exported job history to {output_path}")
