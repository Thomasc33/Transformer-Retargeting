#!/usr/bin/env python3
"""
Training monitoring script to track progress and generate reports.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time


def load_training_log(log_file):
    """Load training log from JSONL file."""
    if not os.path.exists(log_file):
        return []
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return logs


def plot_training_progress(logs, output_dir):
    """Generate training progress plots."""
    if not logs:
        print("No training logs found.")
        return
    
    df = pd.DataFrame(logs)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Epoch time plot
    axes[0, 1].plot(df['epoch'], df['epoch_time'], label='Epoch Time', marker='o', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].set_title('Epoch Training Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate plot
    if 'learning_rate' in df.columns:
        axes[1, 0].plot(df['epoch'], df['learning_rate'], label='Learning Rate', marker='o', color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Memory usage plot
    if 'gpu_memory_allocated' in df.columns:
        axes[1, 1].plot(df['epoch'], df['gpu_memory_allocated'], label='GPU Memory (GB)', marker='o', color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Memory (GB)')
        axes[1, 1].set_title('GPU Memory Usage')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training progress plot saved to: {plot_path}")


def generate_training_report(logs, output_dir):
    """Generate a comprehensive training report."""
    if not logs:
        print("No training logs found.")
        return
    
    df = pd.DataFrame(logs)
    
    # Calculate statistics
    latest_epoch = df['epoch'].max()
    best_val_loss = df['val_loss'].min()
    best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    avg_epoch_time = df['epoch_time'].mean()
    total_training_time = df['epoch_time'].sum()
    
    # Estimate remaining time
    if len(df) > 1:
        recent_epoch_time = df['epoch_time'].tail(5).mean()  # Average of last 5 epochs
        # This would need to be updated based on total planned epochs
        estimated_remaining_epochs = max(0, 100 - latest_epoch)  # Assuming 100 total epochs
        estimated_remaining_time = recent_epoch_time * estimated_remaining_epochs
    else:
        estimated_remaining_time = 0
    
    # Generate report
    report = f"""
# Training Progress Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Latest Epoch**: {latest_epoch}
- **Best Validation Loss**: {best_val_loss:.6f} (Epoch {best_epoch})
- **Average Epoch Time**: {avg_epoch_time:.2f} seconds ({avg_epoch_time/60:.2f} minutes)
- **Total Training Time**: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)
- **Estimated Remaining Time**: {estimated_remaining_time:.2f} seconds ({estimated_remaining_time/3600:.2f} hours)

## Recent Performance (Last 5 Epochs)
"""
    
    if len(df) >= 5:
        recent_df = df.tail(5)
        report += f"""
- **Average Train Loss**: {recent_df['train_loss'].mean():.6f}
- **Average Val Loss**: {recent_df['val_loss'].mean():.6f}
- **Average Epoch Time**: {recent_df['epoch_time'].mean():.2f} seconds
"""
        
        if 'gpu_memory_allocated' in recent_df.columns:
            report += f"- **Average GPU Memory**: {recent_df['gpu_memory_allocated'].mean():.2f} GB\n"
    
    # Add detailed epoch information
    report += "\n## Detailed Epoch Information\n"
    report += "| Epoch | Train Loss | Val Loss | Time (min) | GPU Mem (GB) |\n"
    report += "|-------|------------|----------|------------|-------------|\n"
    
    for _, row in df.iterrows():
        gpu_mem = row.get('gpu_memory_allocated', 0)
        report += f"| {row['epoch']} | {row['train_loss']:.6f} | {row['val_loss']:.6f} | {row['epoch_time']/60:.2f} | {gpu_mem:.2f} |\n"
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'training_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Training report saved to: {report_path}")
    
    return report


def monitor_training_live(log_file, update_interval=60):
    """Monitor training progress in real-time."""
    print(f"Monitoring training log: {log_file}")
    print(f"Update interval: {update_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_size = 0
    
    try:
        while True:
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)
                if current_size > last_size:
                    logs = load_training_log(log_file)
                    if logs:
                        latest = logs[-1]
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Epoch {latest['epoch']}: "
                              f"Train Loss: {latest['train_loss']:.6f}, "
                              f"Val Loss: {latest['val_loss']:.6f}, "
                              f"Time: {latest['epoch_time']:.2f}s")
                    last_size = current_size
            
            time.sleep(update_interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--log-dir', type=str, default='logs/comprehensive_all',
                       help='Directory containing training logs')
    parser.add_argument('--output-dir', type=str, default='training_reports',
                       help='Directory to save reports and plots')
    parser.add_argument('--live', action='store_true',
                       help='Monitor training in real-time')
    parser.add_argument('--update-interval', type=int, default=60,
                       help='Update interval for live monitoring (seconds)')
    
    args = parser.parse_args()
    
    log_file = os.path.join(args.log_dir, 'checkpoints', 'training_log.jsonl')
    
    if args.live:
        monitor_training_live(log_file, args.update_interval)
    else:
        # Generate static report
        logs = load_training_log(log_file)
        
        if not logs:
            print(f"No training logs found at: {log_file}")
            return
        
        print(f"Found {len(logs)} training log entries")
        
        # Generate plots and report
        plot_training_progress(logs, args.output_dir)
        report = generate_training_report(logs, args.output_dir)
        
        # Print summary to console
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        if logs:
            latest = logs[-1]
            print(f"Latest Epoch: {latest['epoch']}")
            print(f"Train Loss: {latest['train_loss']:.6f}")
            print(f"Val Loss: {latest['val_loss']:.6f}")
            print(f"Epoch Time: {latest['epoch_time']:.2f}s")
            if 'gpu_memory_allocated' in latest:
                print(f"GPU Memory: {latest['gpu_memory_allocated']:.2f}GB")


if __name__ == '__main__':
    main()
