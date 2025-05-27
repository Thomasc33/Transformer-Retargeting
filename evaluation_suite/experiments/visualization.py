"""
Visualization and qualitative analysis experiments.

Based on Section 7 of experiments.md:
- Motion visualizations
- Attention visualization
"""

from typing import Dict, Any


class VisualizationExperiments:
    """Visualization and qualitative analysis experiments."""
    
    @staticmethod
    def get_experiment_configs() -> Dict[str, Any]:
        """Get all visualization experiment configurations."""
        return {
            'motion_visualizations': VisualizationExperiments.motion_visualizations(),
            'attention_visualization': VisualizationExperiments.attention_visualization(),
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
                    'path': 'trained_models/dmr_ntu_cv_final.pth'
                },
                'pmr': {
                    'type': 'pmr',
                    'path': 'trained_models/pmr_ntu_cv_final.pth'
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
