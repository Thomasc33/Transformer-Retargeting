"""
Ablation study experiments - Loss function analysis.

Based on Section 3 of experiments.md:
- Single loss component ablation studies
- Loss weight sensitivity analysis
"""

from typing import Dict, Any


class AblationExperiments:
    """Ablation study experiments for loss function analysis."""
    
    @staticmethod
    def get_experiment_configs() -> Dict[str, Any]:
        """Get all ablation experiment configurations."""
        return {
            'loss_ablation': AblationExperiments.loss_ablation(),
            'loss_weight_sensitivity': AblationExperiments.loss_weight_sensitivity(),
        }
    
    @staticmethod
    def loss_ablation() -> Dict[str, Any]:
        """Single loss component ablation studies."""
        return {
            'name': 'Loss Function Ablation',
            'description': 'Ablation study removing individual loss components',
            'evaluation_type': 'ablation',
            'models': {
                'baseline': {
                    'type': 'transformer',
                    'path': 'output/ntu_baseline_cv/model_best.pth.tar'
                },
                'no_bone_length': {
                    'type': 'transformer',
                    'path': 'output/ntu_no_bone_cv/model_best.pth.tar'
                },
                'no_foot_contact': {
                    'type': 'transformer',
                    'path': 'output/ntu_no_foot_cv/model_best.pth.tar'
                },
                'no_joint_limit': {
                    'type': 'transformer',
                    'path': 'output/ntu_no_joint_cv/model_best.pth.tar'
                },
                'no_fid_velocity': {
                    'type': 'transformer',
                    'path': 'output/ntu_no_fid_cv/model_best.pth.tar'
                },
                'no_end_effector': {
                    'type': 'transformer',
                    'path': 'output/ntu_no_ee_cv/model_best.pth.tar'
                },
                'no_smoothing': {
                    'type': 'transformer',
                    'path': 'output/ntu_no_smooth_cv/model_best.pth.tar'
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
                    'train_samples': 10000,  # Small training set for ablation retraining
                    'test_samples': 2000     # Small test set for ablation evaluation
                }
            },
            'metrics': [
                'action_recognition_accuracy',
                'reidentification_accuracy',
                'mse',
                'bone_length_consistency',
                'joint_angle_limits',
                'temporal_smoothness',
                'velocity_consistency',
                'foot_contact_consistency',
                'fid_score'
            ],
            'output_dir': 'loss_ablation'
        }
    
    @staticmethod
    def loss_weight_sensitivity() -> Dict[str, Any]:
        """Loss weight sensitivity analysis (already completed with Optuna)."""
        return {
            'name': 'Loss Weight Sensitivity Analysis',
            'description': 'Compare three loss weight configurations: (1) Optimal weights found via Optuna hyperparameter optimization, (2) Equal weights for all loss components, and (3) MSE-only training. This analysis shows how different loss weight balancing affects the privacy-utility tradeoff and physical plausibility of generated motions.',
            'evaluation_type': 'weight_sensitivity',
            'completed': False,  # Models not trained yet
            'models': {
                'optimal_weights': {
                    'type': 'transformer',
                    'path': 'model.pth',  # Current optimal model (Optuna-tuned)
                    'description': 'Model trained with Optuna-optimized loss weights'
                },
                'equal_weights': {
                    'type': 'transformer',
                    'path': 'output/ntu_equal_weights_cv/model_best.pth.tar',
                    'description': 'Model trained with equal weights for all loss components',
                    'status': 'needs_training'
                },
                'mse_only': {
                    'type': 'transformer',
                    'path': 'output/ntu_mse_only_cv/model_best.pth.tar',
                    'description': 'Model trained with MSE loss only (no physical constraints)',
                    'status': 'needs_training'
                }
            },
            'eval_models': {
                'sgn_ar': {
                    'type': 'sgn',
                    'task': 'ar',
                    'path': 'output/ntu_ar_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                }
            },
            'data': {
                'ntu_cv': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 32,
                    'train_samples': 10000,  # Small training set for ablation retraining
                    'test_samples': 2000     # Small test set for ablation evaluation
                }
            },
            'metrics': [
                'action_recognition_accuracy',
                'reidentification_accuracy',
                'mse',
                'physical_plausibility_score'
            ],
            'output_dir': 'loss_weight_sensitivity',
            'notes': 'This experiment was completed using Optuna optimization. Results show optimal weights for each loss component.'
        }
