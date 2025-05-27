"""
Pretraining strategy analysis experiments.

Based on Section 2 of experiments.md:
- Masking configurations (temporal and spatial ratios)
- Pretraining approaches (freezing vs fine-tuning vs no pretraining)
"""

from typing import Dict, Any


class PretrainingExperiments:
    """Pretraining strategy analysis experiments."""
    
    @staticmethod
    def get_experiment_configs() -> Dict[str, Any]:
        """Get all pretraining experiment configurations."""
        return {
            'masking_configurations': PretrainingExperiments.masking_configurations(),
            'pretraining_approaches': PretrainingExperiments.pretraining_approaches(),
        }
    
    @staticmethod
    def masking_configurations() -> Dict[str, Any]:
        """Test different temporal and spatial masking ratios."""
        return {
            'name': 'Masking Configurations',
            'description': 'Test different temporal and spatial masking ratios during pretraining',
            'evaluation_type': 'masking_analysis',
            'completed': True,  # Already completed according to experiments.md
            'models': {
                'temporal_30': {
                    'type': 'transformer',
                    'path': 'output/ntu_temporal_30_cv/model_best.pth.tar',
                    'masking_config': {'temporal_ratio': 0.3, 'spatial_ratio': 0.5}
                },
                'temporal_50': {
                    'type': 'transformer',
                    'path': 'output/ntu_temporal_50_cv/model_best.pth.tar',
                    'masking_config': {'temporal_ratio': 0.5, 'spatial_ratio': 0.5}
                },
                'temporal_70': {
                    'type': 'transformer',
                    'path': 'output/ntu_temporal_70_cv/model_best.pth.tar',
                    'masking_config': {'temporal_ratio': 0.7, 'spatial_ratio': 0.5}
                },
                'spatial_30': {
                    'type': 'transformer',
                    'path': 'output/ntu_spatial_30_cv/model_best.pth.tar',
                    'masking_config': {'temporal_ratio': 0.5, 'spatial_ratio': 0.3}
                },
                'spatial_50': {
                    'type': 'transformer',
                    'path': 'output/ntu_spatial_50_cv/model_best.pth.tar',
                    'masking_config': {'temporal_ratio': 0.5, 'spatial_ratio': 0.5}
                },
                'spatial_70': {
                    'type': 'transformer',
                    'path': 'output/ntu_spatial_70_cv/model_best.pth.tar',
                    'masking_config': {'temporal_ratio': 0.5, 'spatial_ratio': 0.7}
                },
                'combined_30_30': {
                    'type': 'transformer',
                    'path': 'output/ntu_combined_30_30_cv/model_best.pth.tar',
                    'masking_config': {'temporal_ratio': 0.3, 'spatial_ratio': 0.3}
                },
                'combined_70_70': {
                    'type': 'transformer',
                    'path': 'output/ntu_combined_70_70_cv/model_best.pth.tar',
                    'masking_config': {'temporal_ratio': 0.7, 'spatial_ratio': 0.7}
                }
            },
            'eval_models': {
                'sgn_ar': {
                    'type': 'sgn',
                    'task': 'ar',
                    'path': 'output/ntu_ar_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                },
                'sgn_ri': {
                    'type': 'sgn',
                    'task': 'ri',
                    'path': 'output/ntu_ri_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                }
            },
            'data': {
                'ntu_cv': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 32,
                    'test_samples': 3000
                }
            },
            'metrics': [
                'action_recognition_accuracy',
                'reidentification_accuracy',
                'mse',
                'bone_length_consistency',
                'temporal_smoothness',
                'masking_effectiveness'
            ],
            'analysis': {
                'optimal_masking_ratio': True,
                'temporal_vs_spatial_impact': True,
                'combined_masking_effects': True
            },
            'output_dir': 'masking_configurations',
            'notes': 'This experiment was completed during model development. Results show optimal masking ratios for pretraining.'
        }
    
    @staticmethod
    def pretraining_approaches() -> Dict[str, Any]:
        """Compare different pretraining strategies."""
        return {
            'name': 'Pretraining Approaches',
            'description': 'Compare different pretraining strategies: freezing vs fine-tuning vs no pretraining',
            'evaluation_type': 'pretraining_comparison',
            'completed': True,  # Already completed according to experiments.md
            'models': {
                'no_pretraining': {
                    'type': 'transformer',
                    'path': 'output/ntu_no_pretrain_cv/model_best.pth.tar',
                    'pretraining_strategy': 'none'
                },
                'frozen_encoder': {
                    'type': 'transformer',
                    'path': 'output/ntu_frozen_encoder_cv/model_best.pth.tar',
                    'pretraining_strategy': 'frozen'
                },
                'fine_tuned_encoder': {
                    'type': 'transformer',
                    'path': 'model.pth',  # Current best model
                    'pretraining_strategy': 'fine_tuned'
                }
            },
            'eval_models': {
                'sgn_ar': {
                    'type': 'sgn',
                    'task': 'ar',
                    'path': 'output/ntu_ar_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                },
                'sgn_ri': {
                    'type': 'sgn',
                    'task': 'ri',
                    'path': 'output/ntu_ri_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                }
            },
            'data': {
                'ntu_cv': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 32,
                    'test_samples': 3000
                }
            },
            'metrics': [
                'action_recognition_accuracy',
                'reidentification_accuracy',
                'mse',
                'bone_length_consistency',
                'temporal_smoothness',
                'training_convergence_speed',
                'final_loss_values'
            ],
            'analysis': {
                'convergence_comparison': True,
                'final_performance_comparison': True,
                'training_efficiency': True
            },
            'output_dir': 'pretraining_approaches',
            'notes': 'This experiment was completed during model development. Results show fine-tuning approach is optimal.'
        }
