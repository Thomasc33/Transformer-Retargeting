#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Tracking Script

This script tracks all running SLURM evaluation jobs and provides comprehensive
status updates, result file locations, and progress monitoring.

Usage:
    python scripts/track_evaluations.py
    python scripts/track_evaluations.py --detailed
    python scripts/track_evaluations.py --results-only
"""

import os
import sys
import subprocess
import json
import time
import glob
from pathlib import Path
from datetime import datetime
import argparse

def run_command(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def get_slurm_jobs():
    """Get current SLURM jobs for the user."""
    cmd = "squeue -u $USER --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R'"
    stdout, stderr, returncode = run_command(cmd)
    
    if returncode != 0:
        return []
    
    lines = stdout.split('\n')
    if len(lines) < 2:
        return []
    
    jobs = []
    for line in lines[1:]:  # Skip header
        if line.strip():
            parts = line.split()
            if len(parts) >= 9:
                job = {
                    'job_id': parts[0],
                    'partition': parts[1],
                    'name': parts[2],
                    'user': parts[3],
                    'state': parts[4],
                    'time': parts[5],
                    'time_limit': parts[6],
                    'nodes': parts[7],
                    'reason': ' '.join(parts[8:]) if len(parts) > 8 else ''
                }
                jobs.append(job)
    
    return jobs

def get_job_progress(job_id):
    """Get progress information for a specific job."""
    # Check log files for progress
    log_files = glob.glob(f"logs/*{job_id}*")
    
    progress_info = {
        'log_files': log_files,
        'progress': 'Unknown',
        'eta': 'Unknown',
        'status': 'Running'
    }
    
    # Look for progress in error files (where tqdm outputs)
    for log_file in log_files:
        if log_file.endswith('.err'):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Look for the last progress line
                for line in reversed(lines):
                    if 'Evaluating with' in line and '%' in line:
                        # Extract progress percentage
                        if '|' in line:
                            parts = line.split('|')
                            if len(parts) >= 2:
                                progress_part = parts[1].strip()
                                if '%' in progress_part:
                                    progress_info['progress'] = progress_part.split('%')[0] + '%'
                        
                        # Extract ETA
                        if '<' in line and '>' in line:
                            eta_part = line.split('<')[1].split('>')[0]
                            progress_info['eta'] = eta_part
                        
                        break
                        
            except Exception as e:
                continue
    
    return progress_info

def find_result_files():
    """Find all result files from evaluations."""
    result_patterns = [
        'results/**/*.json',
        'results/**/*.md',
        'results/**/*.png',
        'results/**/*.pdf',
        'logs/*.out',
        'logs/*.err'
    ]
    
    result_files = {}
    
    for pattern in result_patterns:
        files = glob.glob(pattern, recursive=True)
        category = pattern.split('/')[0]
        if category not in result_files:
            result_files[category] = []
        result_files[category].extend(files)
    
    return result_files

def get_recent_results(hours=24):
    """Get results from the last N hours."""
    import time
    cutoff_time = time.time() - (hours * 3600)
    
    recent_files = []
    
    # Check results directory
    for root, dirs, files in os.walk('results'):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                if os.path.getmtime(filepath) > cutoff_time:
                    recent_files.append({
                        'path': filepath,
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S'),
                        'size': os.path.getsize(filepath)
                    })
            except:
                continue
    
    # Check logs directory
    for root, dirs, files in os.walk('logs'):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                if os.path.getmtime(filepath) > cutoff_time:
                    recent_files.append({
                        'path': filepath,
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S'),
                        'size': os.path.getsize(filepath)
                    })
            except:
                continue
    
    # Sort by modification time
    recent_files.sort(key=lambda x: x['modified'], reverse=True)
    
    return recent_files

def print_header(title):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def print_section(title):
    """Print formatted section."""
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")

def main():
    """Main tracking function."""
    parser = argparse.ArgumentParser(description="Track evaluation jobs and results")
    parser.add_argument('--detailed', action='store_true', help='Show detailed progress information')
    parser.add_argument('--results-only', action='store_true', help='Show only result files')
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back for recent results')
    
    args = parser.parse_args()
    
    print_header("EVALUATION TRACKING DASHBOARD")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not args.results_only:
        # Get current SLURM jobs
        print_section("CURRENT SLURM JOBS")
        jobs = get_slurm_jobs()
        
        if not jobs:
            print("No SLURM jobs currently running.")
        else:
            # Filter for evaluation jobs
            eval_jobs = [job for job in jobs if 'eval' in job['name'].lower() or 'comprehensive' in job['name'].lower()]
            
            if eval_jobs:
                print(f"Found {len(eval_jobs)} evaluation jobs:")
                print()
                print(f"{'Job ID':<12} {'Name':<30} {'State':<10} {'Time':<10} {'Progress':<12} {'ETA':<15}")
                print("-" * 90)
                
                for job in eval_jobs:
                    progress_info = get_job_progress(job['job_id']) if args.detailed else {'progress': 'N/A', 'eta': 'N/A'}
                    
                    print(f"{job['job_id']:<12} {job['name'][:29]:<30} {job['state']:<10} {job['time']:<10} {progress_info['progress']:<12} {progress_info['eta']:<15}")
                    
                    if args.detailed and progress_info['log_files']:
                        print(f"  Log files: {', '.join([os.path.basename(f) for f in progress_info['log_files']])}")
            else:
                print("No evaluation jobs found in current SLURM queue.")
                
            # Show all jobs for context
            if len(jobs) > len(eval_jobs):
                print(f"\nOther jobs running: {len(jobs) - len(eval_jobs)}")
                for job in jobs:
                    if job not in eval_jobs:
                        print(f"  {job['job_id']}: {job['name']} ({job['state']})")
    
    # Show recent results
    print_section(f"RECENT RESULTS (Last {args.hours} hours)")
    recent_files = get_recent_results(args.hours)
    
    if not recent_files:
        print(f"No result files found in the last {args.hours} hours.")
    else:
        print(f"Found {len(recent_files)} recent files:")
        print()
        print(f"{'Modified':<20} {'Size':<10} {'Path'}")
        print("-" * 80)
        
        for file_info in recent_files[:20]:  # Show top 20
            size_str = f"{file_info['size']:,} B" if file_info['size'] < 1024 else f"{file_info['size']/1024:.1f} KB"
            print(f"{file_info['modified']:<20} {size_str:<10} {file_info['path']}")
        
        if len(recent_files) > 20:
            print(f"... and {len(recent_files) - 20} more files")
    
    # Show result file categories
    print_section("RESULT FILE CATEGORIES")
    result_files = find_result_files()
    
    for category, files in result_files.items():
        if files:
            print(f"\n{category.upper()}:")
            
            # Group by type
            file_types = {}
            for file in files:
                ext = os.path.splitext(file)[1] or 'no_ext'
                if ext not in file_types:
                    file_types[ext] = []
                file_types[ext].append(file)
            
            for ext, type_files in file_types.items():
                print(f"  {ext}: {len(type_files)} files")
                if args.detailed:
                    for file in sorted(type_files)[:5]:  # Show first 5
                        print(f"    - {file}")
                    if len(type_files) > 5:
                        print(f"    ... and {len(type_files) - 5} more")
    
    # Show key result directories
    print_section("KEY RESULT DIRECTORIES")
    key_dirs = [
        'results/comprehensive',
        'results/ntu_cv',
        'results/evaluation_suite',
        'logs'
    ]
    
    for dir_path in key_dirs:
        if os.path.exists(dir_path):
            try:
                file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                dir_count = len([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
                print(f"✅ {dir_path}: {file_count} files, {dir_count} subdirectories")
                
                if args.detailed:
                    # Show recent files in this directory
                    recent_in_dir = [f for f in recent_files if f['path'].startswith(dir_path)][:3]
                    for file_info in recent_in_dir:
                        print(f"    Recent: {os.path.basename(file_info['path'])} ({file_info['modified']})")
                        
            except Exception as e:
                print(f"❌ {dir_path}: Error accessing directory")
        else:
            print(f"❌ {dir_path}: Directory not found")
    
    # Show summary
    print_section("SUMMARY")
    
    # Count running jobs
    eval_jobs = [job for job in get_slurm_jobs() if 'eval' in job['name'].lower() or 'comprehensive' in job['name'].lower()]
    running_jobs = len([job for job in eval_jobs if job['state'] in ['RUNNING', 'R']])
    pending_jobs = len([job for job in eval_jobs if job['state'] in ['PENDING', 'PD']])
    
    print(f"📊 Evaluation Jobs: {running_jobs} running, {pending_jobs} pending")
    print(f"📁 Recent Files: {len(recent_files)} files in last {args.hours} hours")
    print(f"📈 Total Result Categories: {len([cat for cat, files in result_files.items() if files])}")
    
    # Show next steps
    if running_jobs > 0:
        print(f"\n🔄 {running_jobs} evaluation(s) currently running. Check back later for results.")
    elif pending_jobs > 0:
        print(f"\n⏳ {pending_jobs} evaluation(s) pending. Jobs will start when resources are available.")
    else:
        print(f"\n✅ No evaluation jobs currently running. All evaluations may be complete!")
    
    print(f"\n📋 To monitor specific jobs: squeue -u $USER")
    print(f"📋 To check job details: scontrol show job <job_id>")
    print(f"📋 To view logs: tail -f logs/<job_name>_<job_id>.out")

if __name__ == "__main__":
    main()
