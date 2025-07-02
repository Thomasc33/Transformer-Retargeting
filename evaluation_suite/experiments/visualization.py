"""
Visualization and qualitative analysis experiments.

Based on Section 7 of experiments.md:
- Motion visualizations
- Attention visualization
- Enhanced skeleton animations with proper Kinect v2 view angles
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional

# Set matplotlib backend for HPC environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class VisualizationExperiments:
    """Visualization and qualitative analysis experiments."""

    # NTU skeleton connections (25 joints) - Kinect v2 format
    NTU_CONNECTIONS = [
        [0, 1], [1, 20], [20, 2], [2, 3], [20, 8], [8, 9], [9, 10], [10, 11],
        [11, 23], [11, 24], [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22],
        [0, 16], [16, 17], [17, 18], [18, 19], [0, 12], [12, 13], [13, 14], [14, 15]
    ]

    # Color palette for different visualization types
    COLORS = {
        'original': '#1f77b4',      # Blue
        'retargeted': '#ff7f0e',    # Orange
        'anonymized': '#2ca02c',    # Green
        'raw': '#d62728',           # Red
        'joints': '#9467bd',        # Purple
        'bones': '#8c564b',         # Brown
        'sensitive': '#e377c2',     # Pink
        'masked': '#7f7f7f'         # Gray
    }

    @staticmethod
    def get_experiment_configs() -> Dict[str, Any]:
        """Get all visualization experiment configurations."""
        return {
            'motion_visualizations': VisualizationExperiments.motion_visualizations(),
            'attention_visualization': VisualizationExperiments.attention_visualization(),
            'skeleton_animations': VisualizationExperiments.skeleton_animations(),
            'comparison_visualizations': VisualizationExperiments.comparison_visualizations(),
            'sensitivity_analysis': VisualizationExperiments.sensitivity_analysis(),
            'anonymization_showcase': VisualizationExperiments.anonymization_showcase(),
            'mlm_pretraining': VisualizationExperiments.mlm_pretraining(),
        }
    
    @staticmethod
    def motion_visualizations() -> Dict[str, Any]:
        """Motion visualization experiment."""
        return {
            'name': 'Motion Visualizations',
            'description': 'Create animated sequences and overlays',
            'evaluation_type': 'motion_visualization',
            'models': {
                'transformer': {
                    'type': 'transformer',
                    'path': 'model.pth'
                },
                'dmr': {
                    'type': 'dmr',
                    'path': 'trained_models/dmr_ntu_cv_best.pth'
                },
                'pmr': {
                    'type': 'pmr',
                    'path': 'trained_models/pmr_ntu_cv_best.pth'
                },
                'raw': {
                    'type': 'raw',
                    'path': 'raw'
                }
            },
            'data': {
                'ntu_cv_samples': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 1,
                    'test_samples': 20  # Small number for detailed visualization
                }
            },
            'visualizations': [
                'skeleton_animations',
                'side_by_side_comparison',
                'overlay_visualization',
                'trajectory_plots',
                'joint_velocity_plots',
                'bone_length_plots'
            ],
            'output_formats': ['mp4', 'gif', 'png_sequence'],
            'analysis': {
                'motion_quality_assessment': True,
                'visual_artifacts_detection': True,
                'motion_smoothness_visual': True
            },
            'output_dir': 'motion_visualizations'
        }
    
    @staticmethod
    def attention_visualization() -> Dict[str, Any]:
        """Attention visualization experiment."""
        return {
            'name': 'Attention Visualization',
            'description': 'Create attention heatmaps and analysis',
            'evaluation_type': 'attention_visualization',
            'models': {
                'transformer': {
                    'type': 'transformer',
                    'path': 'model.pth'
                }
            },
            'data': {
                'ntu_cv_attention': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 1,
                    'test_samples': 10  # Small number for detailed analysis
                }
            },
            'visualizations': [
                'encoder_attention_maps',
                'decoder_attention_maps',
                'cross_attention_maps',
                'attention_evolution_over_time',
                'joint_attention_patterns',
                'temporal_attention_patterns'
            ],
            'analysis': {
                'attention_pattern_analysis': True,
                'important_joints_identification': True,
                'temporal_focus_analysis': True,
                'attention_consistency_analysis': True
            },
            'output_dir': 'attention_visualization'
        }

    @staticmethod
    def skeleton_animations() -> Dict[str, Any]:
        """Enhanced skeleton animation experiment with proper Kinect v2 view angles."""
        return {
            'name': 'Skeleton Animations',
            'description': 'Create high-quality skeleton animations with proper view angles',
            'evaluation_type': 'skeleton_animation',
            'models': {
                'transformer': {
                    'type': 'transformer',
                    'path': 'model.pth'
                },
                'dmr': {
                    'type': 'dmr',
                    'path': 'trained_models/dmr_ntu_cv_best.pth'
                },
                'pmr': {
                    'type': 'pmr',
                    'path': 'trained_models/pmr_ntu_cv_best.pth'
                },
                'raw': {
                    'type': 'raw',
                    'path': 'raw'
                }
            },
            'data': {
                'ntu_cv_animation': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 1,
                    'test_samples': 10
                }
            },
            'animations': [
                'original_skeleton',
                'retargeted_skeleton',
                'dummy_skeleton',
                'combined_original_retargeted'
            ],
            'view_settings': {
                'camera_angle': {
                    'elevation': 10,
                    'azimuth': 0
                },
                'kinect_v2_view': True,
                'fixed_bounds': True,
                'aspect_ratio': 'equal'
            },
            'output_formats': ['gif', 'mp4'],
            'quality_settings': {
                'fps': 10,
                'duration_per_frame': 0.15,
                'resolution': (800, 600),
                'joint_size': 50,
                'bone_width': 2
            },
            'output_dir': 'skeleton_animations'
        }

    @staticmethod
    def comparison_visualizations() -> Dict[str, Any]:
        """Comparison visualization experiment for different anonymization methods."""
        return {
            'name': 'Comparison Visualizations',
            'description': 'Compare different anonymization methods side-by-side',
            'evaluation_type': 'comparison_visualization',
            'models': {
                'transformer': {
                    'type': 'transformer',
                    'path': 'model.pth'
                },
                'dmr': {
                    'type': 'dmr',
                    'path': 'trained_models/dmr_ntu_cv_best.pth'
                },
                'pmr': {
                    'type': 'pmr',
                    'path': 'trained_models/pmr_ntu_cv_best.pth'
                }
            },
            'data': {
                'ntu_cv_comparison': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 1,
                    'test_samples': 5
                }
            },
            'comparisons': [
                'original_vs_all_methods',
                'method_effectiveness_grid',
                'privacy_utility_tradeoff',
                'temporal_consistency_analysis',
                'joint_importance_heatmaps'
            ],
            'layout': {
                'grid_size': (2, 3),
                'subplot_titles': True,
                'shared_axes': True,
                'figure_size': (18, 12)
            },
            'analysis': {
                'quantitative_metrics_overlay': True,
                'visual_quality_assessment': True,
                'motion_preservation_analysis': True
            },
            'output_dir': 'comparison_visualizations'
        }

    @staticmethod
    def sensitivity_analysis() -> Dict[str, Any]:
        """Sensitivity analysis visualization experiment."""
        return {
            'name': 'Sensitivity Analysis',
            'description': 'Visualize model sensitivity to different inputs',
            'evaluation_type': 'sensitivity_analysis',
            'models': {
                'transformer': {
                    'type': 'transformer',
                    'path': 'model.pth'
                }
            },
            'animations': [
                'sensitivity_heatmap'
            ],
            'data_settings': {
                'max_samples': 3,
                'sample_strategy': 'diverse'
            },
            'quality_settings': {
                'max_frames': 30,
                'fps': 10,
                'resolution': (800, 600)
            },
            'analysis': {
                'input_perturbation': True,
                'gradient_analysis': True,
                'feature_importance': True
            },
            'output_dir': 'sensitivity_analysis'
        }

    @staticmethod
    def anonymization_showcase() -> Dict[str, Any]:
        """Anonymization showcase visualization experiment."""
        return {
            'name': 'Anonymization Showcase',
            'description': 'Showcase different anonymization techniques',
            'evaluation_type': 'anonymization_showcase',
            'models': {
                'transformer': {
                    'type': 'transformer',
                    'path': 'model.pth'
                },
                'dmr': {
                    'type': 'dmr',
                    'path': 'trained_models/dmr_ntu_cv_best.pth'
                },
                'pmr': {
                    'type': 'pmr',
                    'path': 'trained_models/pmr_ntu_cv_best.pth'
                },
                'raw': {
                    'type': 'raw',
                    'path': 'raw'
                }
            },
            'animations': [
                'original_skeleton',
                'retargeted_skeleton',
                'dummy_skeleton'
            ],
            'data_settings': {
                'max_samples': 5,
                'sample_strategy': 'representative'
            },
            'quality_settings': {
                'max_frames': 50,
                'fps': 15,
                'resolution': (1024, 768)
            },
            'analysis': {
                'anonymization_effectiveness': True,
                'motion_preservation': True,
                'visual_quality': True
            },
            'output_dir': 'anonymization_showcase'
        }

    @staticmethod
    def mlm_pretraining() -> Dict[str, Any]:
        """MLM pretraining visualization experiment."""
        return {
            'name': 'MLM Pretraining Visualization',
            'description': 'Visualize MLM pretraining reconstruction results',
            'evaluation_type': 'mlm_pretraining',
            'models': {
                'mlm_autoencoder': {
                    'type': 'mlm_autoencoder',
                    'base_dir': '.',  # Will be updated based on arguments
                    'temporal_ratio': 0.3,  # Will be updated based on arguments
                    'spatial_ratio': 0.3   # Will be updated based on arguments
                }
            },
            'data': {
                'mlm_samples': {
                    'type': 'mlm_masked',
                    'dataset': 'ntu',  # Will be updated
                    'setting': 'cv',   # Will be updated
                    'batch_size': 1,
                    'test_samples': 5,
                    'temporal_masking_ratio': 0.3,
                    'spatial_masking_ratio': 0.3
                }
            },
            'animations': [
                'original_skeleton',
                'masked_skeleton',
                'reconstructed_skeleton',
                'side_by_side_comparison',
                'overlay_comparison'
            ],
            'view_settings': {
                'camera_angle': {
                    'elevation': 10,
                    'azimuth': 0
                },
                'kinect_v2_view': True,
                'fixed_bounds': True,
                'aspect_ratio': 'equal'
            },
            'output_formats': ['gif'],
            'quality_settings': {
                'fps': 10,
                'duration_per_frame': 0.15,
                'resolution': (1200, 400),  # Wide format for side-by-side
                'joint_size': 50,
                'bone_width': 2,
                'max_frames': 50
            },
            'analysis': {
                'reconstruction_quality': True,
                'masking_effectiveness': True,
                'temporal_consistency': True
            },
            'output_dir': 'mlm_pretraining'
        }

    @staticmethod
    def create_skeleton_figure(figsize=(10, 8), kinect_v2_view=True):
        """Create a properly configured 3D figure for skeleton visualization."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if kinect_v2_view:
            # Set proper viewing angle to match visualize/render.py
            # Plotly camera eye=(0,0,-0.9) means looking from negative Z towards origin
            # In matplotlib: elev=10 (slight upward angle), azim=0 (looking along +Z axis from -Z)
            ax.view_init(elev=10, azim=0)

            # Configure axis properties for better skeleton visibility
            ax.set_xlabel('X (Left-Right)', fontsize=10)
            ax.set_ylabel('Y (Up-Down)', fontsize=10)
            ax.set_zlabel('Z (Forward-Back)', fontsize=10)

            # Remove grid and ticks for cleaner look
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Set background color
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # Make pane edges transparent
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')
            ax.xaxis.pane.set_alpha(0)
            ax.yaxis.pane.set_alpha(0)
            ax.zaxis.pane.set_alpha(0)

        return fig, ax

    @staticmethod
    def setup_3d_axis_kinect(ax, skeleton_data, title="", fixed_bounds=None):
        """Setup 3D axis with Kinect v2 optimal settings."""
        if fixed_bounds is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = fixed_bounds
        else:
            # Calculate bounds from skeleton data
            if len(skeleton_data.shape) == 3:
                all_data = skeleton_data.reshape(-1, 3)
            else:
                all_data = skeleton_data

            # Account for Y and Z swap to match render.py orientation
            padding = 0.3
            x_min, x_max = all_data[:, 0].min() - padding, all_data[:, 0].max() + padding
            y_min, y_max = all_data[:, 2].min() - padding, all_data[:, 2].max() + padding  # Use Z as Y
            z_min, z_max = all_data[:, 1].min() - padding, all_data[:, 1].max() + padding  # Use Y as Z

        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Set equal aspect ratio for proper skeleton proportions
        ax.set_box_aspect([1,1,1])

        # Set title
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')

    @staticmethod
    def calculate_global_bounds(skeleton_sequences):
        """Calculate global bounds for consistent scaling across all sequences."""
        all_points = []

        if isinstance(skeleton_sequences, (list, tuple)):
            for seq in skeleton_sequences:
                if len(seq.shape) == 3:
                    all_points.append(seq.reshape(-1, 3))
                else:
                    all_points.append(seq)
        else:
            if len(skeleton_sequences.shape) == 3:
                all_points.append(skeleton_sequences.reshape(-1, 3))
            else:
                all_points.append(skeleton_sequences)

        combined_points = np.concatenate(all_points, axis=0)

        # Account for Y and Z swap to match render.py orientation
        padding = 0.5
        bounds_3d = (
            combined_points[:, 0].min() - padding, combined_points[:, 0].max() + padding,
            combined_points[:, 2].min() - padding, combined_points[:, 2].max() + padding,  # Use Z as Y
            combined_points[:, 1].min() - padding, combined_points[:, 1].max() + padding   # Use Y as Z
        )

        return bounds_3d, combined_points.min(axis=0), combined_points.max(axis=0)

    @staticmethod
    def draw_skeleton_3d(ax, joints, connections=None, color='blue', alpha=1.0,
                        joint_size=50, bone_width=2, label=None, importance_scores=None,
                        masked_joints=None, colormap='plasma'):
        """Draw a 3D skeleton with enhanced visualization options."""
        if connections is None:
            connections = VisualizationExperiments.NTU_CONNECTIONS

        # Handle different joint data formats
        if isinstance(joints, torch.Tensor):
            joints = joints.detach().cpu().numpy()

        if len(joints.shape) == 1:
            # Reshape flat array to (25, 3)
            joints = joints.reshape(-1, 3)

        # Draw joints with Y and Z swapped to match render.py orientation
        # render.py: x=x, y=y, z=z (skeleton stands upright)
        # matplotlib: need to swap Y and Z for proper orientation
        if importance_scores is not None:
            # Color joints by importance
            scatter = ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1],
                               c=importance_scores, cmap=colormap, s=joint_size,
                               alpha=alpha, label=label)
        elif masked_joints is not None:
            # Color masked joints differently
            joint_colors = [VisualizationExperiments.COLORS['masked'] if i in masked_joints
                           else color for i in range(len(joints))]
            ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1],
                      c=joint_colors, s=joint_size, alpha=alpha, label=label)
        else:
            # Standard coloring
            ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1],
                      color=color, s=joint_size, alpha=alpha, label=label)

        # Draw bones (connections)
        for i, j in connections:
            if i < len(joints) and j < len(joints):
                bone_color = color
                bone_alpha = alpha

                # Modify bone appearance for masked joints
                if masked_joints is not None and (i in masked_joints or j in masked_joints):
                    bone_color = VisualizationExperiments.COLORS['masked']
                    bone_alpha = 0.3

                # Draw bones with Y and Z swapped to match render.py orientation
                ax.plot([joints[i, 0], joints[j, 0]],
                       [joints[i, 2], joints[j, 2]],  # Y and Z swapped
                       [joints[i, 1], joints[j, 1]],  # Y and Z swapped
                       color=bone_color, alpha=bone_alpha, linewidth=bone_width)

    @staticmethod
    def draw_motion_trails(ax, skeleton_sequence, current_frame, trail_length=10):
        """Draw motion trails for key joints."""
        # Key joints to show trails for (hands, feet, head)
        key_joints = [3, 7, 11, 15, 19]  # Head, left hand, right hand, left foot, right foot

        start_frame = max(0, current_frame - trail_length)

        for joint_idx in key_joints:
            if joint_idx < skeleton_sequence.shape[1]:
                # Extract trail for this joint
                trail_points = skeleton_sequence[start_frame:current_frame+1, joint_idx, :]

                if len(trail_points) > 1:
                    # Draw trail with fading alpha
                    for i in range(len(trail_points) - 1):
                        alpha = (i + 1) / len(trail_points) * 0.5  # Fading trail
                        # Draw motion trails with Y and Z swapped to match render.py orientation
                        ax.plot([trail_points[i, 0], trail_points[i+1, 0]],
                               [trail_points[i, 2], trail_points[i+1, 2]],  # Y and Z swapped
                               [trail_points[i, 1], trail_points[i+1, 1]],  # Y and Z swapped
                               color='red', alpha=alpha, linewidth=2)

    @staticmethod
    def extract_joints_from_frame(frame_data):
        """Extract joint positions from frame data."""
        if isinstance(frame_data, torch.Tensor):
            frame_data = frame_data.detach().cpu().numpy()

        # Handle different data formats
        if len(frame_data.shape) == 1:
            # Flat array - reshape to (25, 3)
            if len(frame_data) >= 75:  # 25 joints * 3 coordinates
                return frame_data[:75].reshape(25, 3)
            else:
                # Pad if necessary
                padded = np.zeros(75)
                padded[:len(frame_data)] = frame_data
                return padded.reshape(25, 3)
        elif len(frame_data.shape) == 2:
            # Already in (joints, coordinates) format
            return frame_data
        else:
            raise ValueError(f"Unexpected frame data shape: {frame_data.shape}")

    @staticmethod
    def reshape_skeleton(skeleton):
        """Reshape skeleton data to (frames, joints, coordinates) format."""
        if isinstance(skeleton, torch.Tensor):
            skeleton = skeleton.detach().cpu().numpy()

        if len(skeleton.shape) == 2:
            # (frames, features) -> (frames, joints, coordinates)
            frames, features = skeleton.shape
            if features >= 75:  # At least 25 joints * 3 coordinates
                return skeleton[:, :75].reshape(frames, 25, 3)
            else:
                # Pad if necessary
                padded = np.zeros((frames, 75))
                padded[:, :features] = skeleton
                return padded.reshape(frames, 25, 3)
        elif len(skeleton.shape) == 3:
            # Already in correct format
            return skeleton
        else:
            raise ValueError(f"Unexpected skeleton shape: {skeleton.shape}")

    @staticmethod
    def extract_filename_from_sample(sample):
        """Extract filename from sample data structure."""
        filename = None
        if isinstance(sample, (list, tuple)) and len(sample) >= 3:
            # Check if third element is a filename
            potential_filename = sample[2]
            if isinstance(potential_filename, str):
                filename = potential_filename
        return filename

    @staticmethod
    def create_title_with_filename(base_title, sample_filename=None, unique_id=None):
        """Create a title that includes filename information for reproducibility."""
        title_parts = [base_title]

        if unique_id:
            title_parts.append(f'ID: {unique_id}')
        elif sample_filename:
            # Clean up filename for display
            base_name = sample_filename.split('/')[-1].split('\\')[-1]
            base_name = base_name.replace('.skeleton', '').replace('.npy', '').replace('.avi', '')
            title_parts.append(f'File: {base_name}')

        return ' | '.join(title_parts)

    @staticmethod
    def create_skeleton_animation(samples, explanation=None, output_dir='visualizations',
                                 figure_type='original', max_frames=None, unique_id=None):
        """Create animated GIF of skeleton sequences with improved Kinect v2 visualization."""
        print(f"Creating {figure_type} animation...")

        if not samples:
            print(f"No samples provided for {figure_type} animation")
            return None

        os.makedirs(output_dir, exist_ok=True)

        try:
            sample = samples[0]
            # Extract filename for identification
            sample_filename = VisualizationExperiments.extract_filename_from_sample(sample)

            # Handle different sample formats safely
            if isinstance(sample, (list, tuple)):
                skeleton = sample[0]
                label = sample[1] if len(sample) > 1 else None
            else:
                skeleton = sample
                label = None

            # Convert to numpy if needed
            if hasattr(skeleton, 'numpy'):
                skeleton = skeleton.numpy()
            elif not isinstance(skeleton, np.ndarray):
                skeleton = np.array(skeleton)

            if skeleton is None or skeleton.size == 0:
                print(f"Empty skeleton data for {figure_type} animation")
                return None

            # Handle batch dimension - remove if present
            if len(skeleton.shape) == 3 and skeleton.shape[0] == 1:
                skeleton = skeleton[0]  # Remove batch dimension: (1, 64, 75) -> (64, 75)

            reshaped = VisualizationExperiments.reshape_skeleton(skeleton)

            # Calculate global bounds for consistent scaling
            bounds_3d, _, _ = VisualizationExperiments.calculate_global_bounds(reshaped)

            # Remove zero padding - fix the numpy.any call
            if len(skeleton.shape) == 2:
                non_zero_frames = np.any(skeleton[:, :75] != 0, axis=1)
            else:
                reshaped_for_check = skeleton.reshape(skeleton.shape[0], -1)
                non_zero_frames = np.any(reshaped_for_check != 0, axis=1)

            # Apply non-zero frame filtering to both skeleton and reshaped data
            skeleton = skeleton[non_zero_frames]
            reshaped = reshaped[non_zero_frames]
            num_frames = skeleton.shape[0] if max_frames is None else min(skeleton.shape[0], max_frames)

        except Exception as e:
            print(f"Error processing skeleton data for {figure_type}: {e}")
            return None

        frames = []

        if num_frames == 0:
            print(f"No valid frames found for {figure_type} animation")
            return None

        # Create frames with consistent styling
        try:
            for frame_idx in tqdm(range(num_frames), desc=f"Generating {figure_type} frames"):
                try:
                    # Check if matplotlib is available
                    if plt is None:
                        print(f"Matplotlib not available, cannot create {figure_type} animation")
                        return None

                    fig, ax = VisualizationExperiments.create_skeleton_figure(figsize=(8, 8))

                    # Extract joints for current frame
                    if frame_idx < len(reshaped):
                        frame_joints = reshaped[frame_idx]  # Already in (25, 3) format
                    else:
                        # For frames beyond reshaped data, extract directly from skeleton
                        frame_data = skeleton[frame_idx, :75] if len(skeleton.shape) == 2 else skeleton[frame_idx].flatten()[:75]
                        frame_joints = VisualizationExperiments.extract_joints_from_frame(frame_data)

                    if frame_joints is None or frame_joints.size == 0:
                        plt.close(fig)
                        continue

                except Exception as e:
                    if 'fig' in locals():
                        plt.close(fig)
                    continue

                # Draw skeleton based on figure type
                try:
                    if figure_type == 'original':
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, frame_joints,
                            color=VisualizationExperiments.COLORS['original'],
                            label=f'Original - Frame {frame_idx+1}'
                        )
                        # Enhanced title with file identification
                        base_title = f'Original Skeleton - Frame {frame_idx+1}/{num_frames}'
                        title = VisualizationExperiments.create_title_with_filename(
                            base_title, sample_filename, unique_id
                        )

                    elif figure_type == 'retargeted':
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, frame_joints,
                            color=VisualizationExperiments.COLORS['retargeted'],
                            label=f'Retargeted - Frame {frame_idx+1}'
                        )
                        # Enhanced title with file identification
                        base_title = f'Retargeted Skeleton - Frame {frame_idx+1}/{num_frames}'
                        title = VisualizationExperiments.create_title_with_filename(
                            base_title, sample_filename, unique_id
                        )

                    elif figure_type == 'motion_trail':
                        # Draw motion trails for key joints
                        VisualizationExperiments.draw_motion_trails(
                            ax, reshaped, frame_idx, trail_length=10
                        )
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, frame_joints,
                            color=VisualizationExperiments.COLORS['original'],
                            alpha=0.8,
                            label=f'Motion Trail - Frame {frame_idx+1}'
                        )
                        title = f'Motion Trail - Frame {frame_idx+1}/{num_frames}'

                    elif figure_type == 'attention':
                        # Draw attention heatmap (placeholder - would need actual attention weights)
                        # For now, highlight important joints (head, hands, feet)
                        important_joints = [3, 7, 11, 15, 19, 23]  # Head, hands, feet
                        importance_scores = np.zeros(25)
                        importance_scores[important_joints] = 1.0
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, frame_joints,
                            importance_scores=importance_scores,
                            colormap='hot',
                            label=f'Attention - Frame {frame_idx+1}'
                        )
                        title = f'Attention Heatmap - Frame {frame_idx+1}/{num_frames}'

                    elif figure_type == 'comparison':
                        # Draw side-by-side comparison (original vs processed)
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, frame_joints,
                            color=VisualizationExperiments.COLORS['original'],
                            alpha=0.7,
                            label=f'Original - Frame {frame_idx+1}'
                        )
                        # Add a slightly offset version to show comparison
                        offset_joints = frame_joints.copy()
                        offset_joints[:, 0] += 0.2  # Offset in X direction
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, offset_joints,
                            color=VisualizationExperiments.COLORS['retargeted'],
                            alpha=0.7,
                            label=f'Processed - Frame {frame_idx+1}'
                        )
                        title = f'Comparison - Frame {frame_idx+1}/{num_frames}'

                    elif figure_type == 'dummy':
                        # Draw dummy skeleton (minimal pose)
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, frame_joints,
                            color=VisualizationExperiments.COLORS['masked'],
                            alpha=0.6,
                            label=f'Dummy - Frame {frame_idx+1}'
                        )
                        title = f'Dummy Skeleton - Frame {frame_idx+1}/{num_frames}'

                    elif figure_type == 'combined_original_retargeted':
                        # Draw both original and retargeted skeletons together
                        # Original skeleton (blue)
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, frame_joints,
                            color=VisualizationExperiments.COLORS['original'],
                            alpha=0.7,
                            label=f'Original - Frame {frame_idx+1}'
                        )

                        # Get retargeted skeleton if available
                        # For now, create a slightly offset version to show the concept
                        # In a real implementation, this would be the actual retargeted data
                        retargeted_joints = frame_joints.copy()
                        retargeted_joints[:, 0] += 0.3  # Offset in X direction
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, retargeted_joints,
                            color=VisualizationExperiments.COLORS['retargeted'],
                            alpha=0.7,
                            label=f'Retargeted - Frame {frame_idx+1}'
                        )
                        # Enhanced title with file identification
                        base_title = f'Original vs Retargeted - Frame {frame_idx+1}/{num_frames}'
                        title = VisualizationExperiments.create_title_with_filename(
                            base_title, sample_filename, unique_id
                        )

                    elif figure_type == 'sensitivity' and explanation is not None:
                        # Convert label to int if needed
                        if isinstance(label, np.ndarray):
                            label = int(label.item())
                        elif isinstance(label, (np.int64, np.int32)):
                            label = int(label)

                        importance = explanation.importance_score(reshaped, label, is_action=True, alpha=0.9)
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, frame_joints,
                            importance_scores=importance,
                            colormap='plasma',
                            label=f'Sensitivity - Frame {frame_idx+1}'
                        )
                        title = f'Sensitivity Analysis - Frame {frame_idx+1}/{num_frames}'

                    else:
                        # Default to original
                        VisualizationExperiments.draw_skeleton_3d(
                            ax, frame_joints,
                            color=VisualizationExperiments.COLORS['original'],
                            label=f'Frame {frame_idx+1}'
                        )
                        # Enhanced title with file identification
                        base_title = f'Skeleton - Frame {frame_idx+1}/{num_frames}'
                        title = VisualizationExperiments.create_title_with_filename(
                            base_title, sample_filename, unique_id
                        )

                except Exception as e:
                    plt.close(fig)
                    continue

                # Setup axis with proper Kinect v2 view (for all figure types)
                try:
                    VisualizationExperiments.setup_3d_axis_kinect(ax, frame_joints, title, bounds_3d)

                    # Convert to image
                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(image)
                except Exception as e:
                    pass  # Silent fail for individual frames
                finally:
                    plt.close(fig)

            # Save as GIF with better quality
            if frames and imageio is not None:
                # Create filename using skeleton data name if available
                if unique_id:
                    gif_filename = f'{figure_type}_{unique_id}_animation.gif'
                elif sample_filename:
                    # Use the extracted filename
                    base_name = sample_filename.split('/')[-1].split('\\')[-1]  # Get filename from path
                    base_name = base_name.replace('.skeleton', '').replace('.npy', '').replace('.avi', '')
                    # Remove any remaining path separators and clean up
                    base_name = base_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
                    gif_filename = f'{base_name}_{figure_type}.gif'
                else:
                    import time
                    timestamp = int(time.time() * 1000) % 100000  # Last 5 digits of timestamp
                    gif_filename = f'{figure_type}_{timestamp}_animation.gif'

                gif_path = os.path.join(output_dir, gif_filename)
                imageio.mimsave(gif_path, frames, duration=0.15, loop=0)
                print(f"Saved animation: {gif_path}")
                return gif_path
            elif imageio is None:
                print(f"imageio not available, cannot save {figure_type} animation")
                return None
            else:
                print(f"No frames generated for {figure_type} animation")
                return None

        except Exception as e:
            print(f"Error creating {figure_type} animation: {e}")
            return None
