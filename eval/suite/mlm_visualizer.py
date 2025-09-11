#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLM Visualization Module for creating GIF animations of original vs reconstructed skeletons.

This module creates side-by-side visualizations of original and MLM-reconstructed skeleton sequences.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from tqdm import tqdm
import tempfile
import shutil

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# NTU skeleton connections (25 joints)
NTU_CONNECTIONS = [
    [0, 1], [1, 20], [20, 2], [2, 3], [20, 8], [8, 9], [9, 10], [10, 11],
    [11, 23], [11, 24], [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22],
    [0, 16], [16, 17], [17, 18], [18, 19], [0, 12], [12, 13], [13, 14], [14, 15]
]

# Color palette
COLORS = {
    'original': '#1f77b4',    # Blue
    'reconstructed': '#ff7f0e', # Orange
    'joints': '#2ca02c',      # Green
    'bones': '#d62728'        # Red
}


def plot_skeleton_frame(ax, skeleton, frame_idx, connections, color, alpha=1.0, label=None, joint_size=50):
    """
    Plot a single frame of a skeleton.

    Args:
        ax: Matplotlib 3D axis
        skeleton: Skeleton data of shape (T, J, 3)
        frame_idx: Index of frame to plot
        connections: List of joint connections
        color: Color for the skeleton
        alpha: Alpha value for transparency
        label: Label for the legend
        joint_size: Size of joint markers
    """
    # Get the frame
    frame = skeleton[frame_idx]

    # Plot joints with Y and Z swapped to match render.py orientation
    # render.py: x=x, y=y, z=z (skeleton stands upright)
    # matplotlib: need to swap Y and Z for proper orientation
    ax.scatter(frame[:, 0], frame[:, 2], frame[:, 1],
              color=color, alpha=alpha, s=joint_size, label=label)

    # Plot connections (bones) with Y and Z swapped
    for i, j in connections:
        if i < len(frame) and j < len(frame):  # Safety check
            ax.plot([frame[i, 0], frame[j, 0]],
                   [frame[i, 2], frame[j, 2]],  # Y and Z swapped
                   [frame[i, 1], frame[j, 1]],  # Y and Z swapped
                   color=color, alpha=alpha, linewidth=2)


def setup_3d_axis(ax, skeleton_data, title=""):
    """Setup 3D axis with proper limits and styling."""
    # Calculate bounds from all skeleton data
    all_data = np.concatenate([skeleton_data], axis=0) if len(skeleton_data.shape) == 3 else skeleton_data

    # Get min/max for each dimension with some padding
    # Account for Y and Z swap to match render.py orientation
    padding = 0.5
    x_min, x_max = all_data[:, :, 0].min() - padding, all_data[:, :, 0].max() + padding
    y_min, y_max = all_data[:, :, 2].min() - padding, all_data[:, :, 2].max() + padding  # Use Z as Y
    z_min, z_max = all_data[:, :, 1].min() - padding, all_data[:, :, 1].max() + padding  # Use Y as Z

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set viewing angle to match visualize/render.py
    # Plotly camera eye=(0,0,-0.9) means looking from negative Z towards origin
    # In matplotlib: elev=0 (horizontal), azim=0 (looking along +Z axis from -Z)
    # But we need to account for coordinate system differences
    ax.view_init(elev=10, azim=0)


def create_comparison_gif(original_skeleton, reconstructed_skeleton, output_path,
                         sample_info=None, duration=100, max_frames=None):
    """
    Create a GIF comparing original and reconstructed skeletons side by side.

    Args:
        original_skeleton: Original skeleton data (T, J, 3)
        reconstructed_skeleton: Reconstructed skeleton data (T, J, 3)
        output_path: Path to save the GIF
        sample_info: Dictionary with sample information (action, actor, etc.)
        duration: Duration per frame in milliseconds
        max_frames: Maximum number of frames to include (None for all)
    """
    # Ensure numpy arrays
    if torch.is_tensor(original_skeleton):
        original_skeleton = original_skeleton.cpu().numpy()
    if torch.is_tensor(reconstructed_skeleton):
        reconstructed_skeleton = reconstructed_skeleton.cpu().numpy()

    # Handle shape mismatches
    if len(original_skeleton.shape) == 2:  # (T, J*3)
        original_skeleton = original_skeleton.reshape(original_skeleton.shape[0], 25, 3)
    if len(reconstructed_skeleton.shape) == 2:  # (T, J*3)
        reconstructed_skeleton = reconstructed_skeleton.reshape(reconstructed_skeleton.shape[0], 25, 3)

    # Limit frames if specified
    num_frames = min(original_skeleton.shape[0], reconstructed_skeleton.shape[0])
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()

    try:
        # Create frames
        for frame_idx in tqdm(range(num_frames), desc="Generating frames"):
            fig = plt.figure(figsize=(16, 8))

            # Original skeleton subplot
            ax1 = fig.add_subplot(121, projection='3d')
            plot_skeleton_frame(ax1, original_skeleton, frame_idx, NTU_CONNECTIONS,
                              COLORS['original'], label='Original')
            setup_3d_axis(ax1, original_skeleton, 'Original Skeleton')
            ax1.legend()

            # Reconstructed skeleton subplot
            ax2 = fig.add_subplot(122, projection='3d')
            plot_skeleton_frame(ax2, reconstructed_skeleton, frame_idx, NTU_CONNECTIONS,
                              COLORS['reconstructed'], label='Reconstructed')
            setup_3d_axis(ax2, reconstructed_skeleton, 'MLM Reconstructed')
            ax2.legend()

            # Add overall title with comprehensive sample information
            title_parts = [f'Frame {frame_idx+1}/{num_frames}']
            if sample_info:
                if 'dataset' in sample_info and 'setting' in sample_info:
                    title_parts.append(f"Dataset: {sample_info['dataset']}_{sample_info['setting']}")
                if 'temporal_ratio' in sample_info and 'spatial_ratio' in sample_info:
                    title_parts.append(f"Masking: T{sample_info['temporal_ratio']}_S{sample_info['spatial_ratio']}")
                if 'action' in sample_info:
                    title_parts.append(f"Action: {sample_info['action']}")
                if 'actor' in sample_info:
                    title_parts.append(f"Actor: {sample_info['actor']}")
                if 'filename' in sample_info:
                    title_parts.append(f"File: {sample_info['filename']}")
                if 'model_type' in sample_info:
                    title_parts.append(f"Model: {sample_info['model_type']}")

            plt.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold')

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Save frame
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:04d}.png')
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close()

        # Create GIF from frames
        frames = []
        for frame_idx in range(num_frames):
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:04d}.png')
            frames.append(imageio.imread(frame_path))

        # Save GIF
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames, duration=duration, loop=0)

        print(f"GIF saved to: {output_path}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def create_overlay_gif(original_skeleton, reconstructed_skeleton, output_path,
                      sample_info=None, duration=100, max_frames=None):
    """
    Create a GIF with overlaid original and reconstructed skeletons.

    Args:
        original_skeleton: Original skeleton data (T, J, 3)
        reconstructed_skeleton: Reconstructed skeleton data (T, J, 3)
        output_path: Path to save the GIF
        sample_info: Dictionary with sample information
        duration: Duration per frame in milliseconds
        max_frames: Maximum number of frames to include
    """
    # Ensure numpy arrays
    if torch.is_tensor(original_skeleton):
        original_skeleton = original_skeleton.cpu().numpy()
    if torch.is_tensor(reconstructed_skeleton):
        reconstructed_skeleton = reconstructed_skeleton.cpu().numpy()

    # Handle shape mismatches
    if len(original_skeleton.shape) == 2:  # (T, J*3)
        original_skeleton = original_skeleton.reshape(original_skeleton.shape[0], 25, 3)
    if len(reconstructed_skeleton.shape) == 2:  # (T, J*3)
        reconstructed_skeleton = reconstructed_skeleton.reshape(reconstructed_skeleton.shape[0], 25, 3)

    # Limit frames if specified
    num_frames = min(original_skeleton.shape[0], reconstructed_skeleton.shape[0])
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()

    try:
        # Create frames
        for frame_idx in tqdm(range(num_frames), desc="Generating overlay frames"):
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot both skeletons with transparency
            plot_skeleton_frame(ax, original_skeleton, frame_idx, NTU_CONNECTIONS,
                              COLORS['original'], alpha=0.7, label='Original')
            plot_skeleton_frame(ax, reconstructed_skeleton, frame_idx, NTU_CONNECTIONS,
                              COLORS['reconstructed'], alpha=0.7, label='Reconstructed')

            # Setup axis
            combined_data = np.concatenate([original_skeleton, reconstructed_skeleton], axis=0)
            setup_3d_axis(ax, combined_data, 'Original vs MLM Reconstructed')
            ax.legend()

            # Add title with comprehensive sample information
            title_parts = [f'Frame {frame_idx+1}/{num_frames}']
            if sample_info:
                if 'dataset' in sample_info and 'setting' in sample_info:
                    title_parts.append(f"Dataset: {sample_info['dataset']}_{sample_info['setting']}")
                if 'temporal_ratio' in sample_info and 'spatial_ratio' in sample_info:
                    title_parts.append(f"Masking: T{sample_info['temporal_ratio']}_S{sample_info['spatial_ratio']}")
                if 'action' in sample_info:
                    title_parts.append(f"Action: {sample_info['action']}")
                if 'actor' in sample_info:
                    title_parts.append(f"Actor: {sample_info['actor']}")
                if 'filename' in sample_info:
                    title_parts.append(f"File: {sample_info['filename']}")
                if 'model_type' in sample_info:
                    title_parts.append(f"Model: {sample_info['model_type']}")

            plt.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold')

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Save frame
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:04d}.png')
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close()

        # Create GIF from frames
        frames = []
        for frame_idx in range(num_frames):
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:04d}.png')
            frames.append(imageio.imread(frame_path))

        # Save GIF
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames, duration=duration, loop=0)

        print(f"Overlay GIF saved to: {output_path}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def visualize_mlm_samples(mlm_model, data_loader, output_dir, num_samples=5,
                         max_frames=50, device='cuda', dataset='ntu', setting='cv',
                         temporal_ratio='unknown', spatial_ratio='unknown'):
    """
    Visualize MLM reconstruction results for multiple samples.

    Args:
        mlm_model: Trained MLM model
        data_loader: Data loader with samples
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        max_frames: Maximum frames per sample
        device: Device to run model on
    """
    os.makedirs(output_dir, exist_ok=True)

    mlm_model.eval()
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch_content in enumerate(data_loader):
            if sample_count >= num_samples:
                break

            # Extract data from Cross_Data format: (x1, x2, y1, y2, actors, actions)
            # Can be either tuple or list depending on PyTorch version/settings
            if isinstance(batch_content, (tuple, list)) and len(batch_content) == 6:
                x1, x2, y1, y2, actors, actions = batch_content
                x_data = x1  # Use x1 as primary skeleton data
            else:
                continue

            # Process data format
            if not isinstance(x_data, torch.Tensor):
                x_data = torch.tensor(x_data, dtype=torch.float32)

            # Handle different input shapes - Cross_Data returns (batch, frames, joints*channels)
            if len(x_data.shape) == 3 and x_data.shape[2] == 75:
                x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 25, 3)
            elif len(x_data.shape) == 2 and x_data.shape[1] == 75:
                x_data = x_data.reshape(1, x_data.shape[0], 25, 3)

            x_data = x_data.to(device)

            # Process each sample in the batch
            for i in range(min(x_data.shape[0], num_samples - sample_count)):
                try:
                    # Get single sample
                    sample_input = x_data[i:i+1]  # Keep batch dimension

                    # Forward pass through MLM
                    reconstructed = mlm_model(sample_input)

                    # Handle output format
                    if len(reconstructed.shape) == 5:  # (batch, frames, 1, joints, channels)
                        reconstructed = reconstructed.squeeze(2)

                    # Convert to numpy
                    original_np = sample_input[0].cpu().numpy()  # (frames, joints, channels)
                    reconstructed_np = reconstructed[0].cpu().numpy()

                    # Create sample info - handle tensor indexing safely
                    try:
                        if torch.is_tensor(actions):
                            action_val = actions[i].item() if actions[i].numel() == 1 else int(actions[i][0])
                        else:
                            action_val = actions[i] if hasattr(actions, '__getitem__') else actions

                        if torch.is_tensor(actors):
                            actor_val = actors[i].item() if actors[i].numel() == 1 else int(actors[i][0])
                        else:
                            actor_val = actors[i] if hasattr(actors, '__getitem__') else actors
                    except Exception as e:
                        print(f"Warning: Could not extract action/actor info: {e}")
                        action_val = "unknown"
                        actor_val = "unknown"

                    sample_info = {
                        'action': action_val,
                        'actor': actor_val,
                        'filename': f'sample_{sample_count+1}_batch_{batch_idx}_item_{i}',
                        'dataset': dataset,
                        'setting': setting,
                        'temporal_ratio': temporal_ratio,
                        'spatial_ratio': spatial_ratio,
                        'model_type': 'MLM_Autoencoder'
                    }

                    # Create comparison GIF with descriptive filename
                    comparison_filename = f'mlm_comparison_{dataset}_{setting}_t{temporal_ratio}_s{spatial_ratio}_sample_{sample_count+1:03d}.gif'
                    comparison_path = os.path.join(output_dir, comparison_filename)
                    create_comparison_gif(
                        original_np, reconstructed_np, comparison_path,
                        sample_info=sample_info, max_frames=max_frames
                    )

                    # Create overlay GIF with descriptive filename
                    overlay_filename = f'mlm_overlay_{dataset}_{setting}_t{temporal_ratio}_s{spatial_ratio}_sample_{sample_count+1:03d}.gif'
                    overlay_path = os.path.join(output_dir, overlay_filename)
                    create_overlay_gif(
                        original_np, reconstructed_np, overlay_path,
                        sample_info=sample_info, max_frames=max_frames
                    )

                    sample_count += 1
                    print(f"Created visualizations for sample {sample_count}")

                    if sample_count >= num_samples:
                        break

                except Exception as e:
                    print(f"Error visualizing sample {sample_count}: {e}")
                    continue

    print(f"Visualization complete! Created {sample_count} sample visualizations in {output_dir}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='MLM Visualization')
    parser.add_argument('--model-dir', type=str, required=True, help='MLM model directory')
    parser.add_argument('--dataset', type=str, default='ntu', help='Dataset name')
    parser.add_argument('--setting', type=str, default='cv', help='Evaluation setting')
    parser.add_argument('--output-dir', type=str, default='results/mlm_visualizations', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--max-frames', type=int, default=50, help='Maximum frames per sample')

    args = parser.parse_args()

    # This would need to be implemented with proper model loading and data loading
    print(f"MLM visualization script - model: {args.model_dir}, output: {args.output_dir}")
    print("Note: This is a standalone visualization module. Use it from the comprehensive evaluation script.")
