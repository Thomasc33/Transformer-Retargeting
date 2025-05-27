#!/usr/bin/env python3
"""
Job monitoring script for the evaluation suite.

Usage:
    python evaluation_suite/monitor_jobs.py --status
    python evaluation_suite/monitor_jobs.py --wait privacy_utility_sgn baseline_comparison
    python evaluation_suite/monitor_jobs.py --export job_history.csv
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evaluation_suite.runners.slurm_runner import SlurmRunner
from evaluation_suite.runners.job_monitor import JobMonitor


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Monitor evaluation suite jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current status
  python evaluation_suite/monitor_jobs.py --status
  
  # Wait for specific experiments to complete
  python evaluation_suite/monitor_jobs.py --wait privacy_utility_sgn baseline_comparison
  
  # Export job history
  python evaluation_suite/monitor_jobs.py --export job_history.csv
  
  # Clean up old job files
  python evaluation_suite/monitor_jobs.py --cleanup
        """
    )
    
    parser.add_argument('--status', action='store_true', help='Show current job status')
    parser.add_argument('--wait', nargs='+', help='Wait for specific experiments to complete')
    parser.add_argument('--export', type=str, help='Export job history to CSV file')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old job files')
    parser.add_argument('--timeout', type=int, default=24, help='Timeout in hours for waiting')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    parser.add_argument('--config', type=str, default='evaluation_suite/configs/experiments.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
        
    # Initialize components
    hpc_config = config.get('hpc', {})
    slurm_runner = SlurmRunner(hpc_config)
    job_monitor = JobMonitor(slurm_runner)
    
    # Handle different commands
    if args.status:
        job_monitor.print_status_report()
        
    elif args.wait:
        success = job_monitor.wait_for_completion(
            args.wait, 
            timeout_hours=args.timeout,
            check_interval=args.interval
        )
        sys.exit(0 if success else 1)
        
    elif args.export:
        output_path = Path(args.export)
        job_monitor.export_job_history(output_path)
        print(f"✅ Job history exported to {output_path}")
        
    elif args.cleanup:
        job_monitor.cleanup_old_jobs()
        print("✅ Old job files cleaned up")
        
    else:
        parser.print_help()
        print("\n💡 Tip: Use --status to see current job status!")


if __name__ == "__main__":
    main()
