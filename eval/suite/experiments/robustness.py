"""
Robustness analysis experiments.

Based on Section 5 of experiments.md:
- Training stability with multiple random seeds
- Teacher forcing analysis
- Per-class and per-subject analysis
"""

from typing import Dict, Any


class RobustnessExperiments:
    """Robustness analysis experiments."""
    
    @staticmethod
    def get_experiment_configs() -> Dict[str, Any]:
        """Get all robustness experiment configurations."""
        return {
            'training_stability': RobustnessExperiments.training_stability(),
            'teacher_forcing_analysis': RobustnessExperiments.teacher_forcing_analysis(),
            'per_class_analysis': RobustnessExperiments.per_class_analysis(),
            'per_subject_analysis': RobustnessExperiments.per_subject_analysis(),
        }
    
    @staticmethod
    def training_stability() -> Dict[str, Any]:
        """Training stability with multiple random seeds."""
        return {
            'name': 'Training Stability',
            'description': 'Repeat key experiments with different random seeds',
            'evaluation_type': 'stability',
            'models': {
                'seed_42': {
                    'type': 'transformer',
                    'path': 'output/ntu_seed42_cv/model_best.pth.tar'
                },
                'seed_123': {
                    'type': 'transformer',
                    'path': 'output/ntu_seed123_cv/model_best.pth.tar'
                },
                'seed_456': {
                    'type': 'transformer',
                    'path': 'output/ntu_seed456_cv/model_best.pth.tar'
                },
                'seed_789': {
                    'type': 'transformer',
                    'path': 'output/ntu_seed789_cv/model_best.pth.tar'
                },
                'seed_999': {
                    'type': 'transformer',
                    'path': 'output/ntu_seed999_cv/model_best.pth.tar'
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
                    'test_samples': 2000
                }
            },
            'metrics': [
                'action_recognition_accuracy',
                'reidentification_accuracy',
                'mse',
                'bone_length_consistency',
                'temporal_smoothness'
            ],
            'analysis': {
                'statistical_tests': ['anova', 'tukey_hsd'],
                'variance_analysis': True,
                'confidence_intervals': True
            },
            'output_dir': 'training_stability'
        }
    
    @staticmethod
    def teacher_forcing_analysis() -> Dict[str, Any]:
        """Teacher forcing decay schedule analysis."""
        return {
            'name': 'Teacher Forcing Analysis',
            'description': 'Compare different teacher forcing decay schedules',
            'evaluation_type': 'teacher_forcing',
            'models': {
                'linear_decay': {
                    'type': 'transformer',
                    'path': 'output/ntu_tf_linear_cv/model_best.pth.tar'
                },
                'exponential_decay': {
                    'type': 'transformer',
                    'path': 'output/ntu_tf_exp_cv/model_best.pth.tar'
                },
                'step_decay': {
                    'type': 'transformer',
                    'path': 'output/ntu_tf_step_cv/model_best.pth.tar'
                },
                'no_decay': {
                    'type': 'transformer',
                    'path': 'output/ntu_tf_none_cv/model_best.pth.tar'
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
                    'test_samples': 2000
                }
            },
            'metrics': [
                'action_recognition_accuracy',
                'mse',
                'temporal_smoothness',
                'training_convergence'
            ],
            'output_dir': 'teacher_forcing_analysis'
        }
    
    @staticmethod
    def per_class_analysis() -> Dict[str, Any]:
        """Per-class performance analysis."""
        return {
            'name': 'Per-Class Analysis',
            'description': 'Analyze performance for each action class',
            'evaluation_type': 'per_class',
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
                    'test_samples': 5000
                }
            },
            'metrics': [
                'per_class_accuracy',
                'per_class_anonymization',
                'class_confusion_matrix',
                'difficult_classes'
            ],
            'analysis': {
                'class_difficulty_ranking': True,
                'confusion_matrix_visualization': True,
                'error_analysis': True
            },
            'output_dir': 'per_class_analysis'
        }
    
    @staticmethod
    def per_subject_analysis() -> Dict[str, Any]:
        """Per-subject anonymization analysis."""
        return {
            'name': 'Per-Subject Analysis',
            'description': 'Analyze anonymization effectiveness per subject',
            'evaluation_type': 'per_subject',
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
                }
            },
            'eval_models': {
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
                    'test_samples': 5000
                }
            },
            'metrics': [
                'per_subject_anonymization',
                'subject_identification_rate',
                'anonymization_variance',
                'hard_to_anonymize_subjects'
            ],
            'analysis': {
                'subject_difficulty_ranking': True,
                'anonymization_distribution': True,
                'outlier_analysis': True
            },
            'output_dir': 'per_subject_analysis'
        }
