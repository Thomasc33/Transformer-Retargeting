"""
Primary evaluation experiments - Privacy vs. Utility analysis.

Based on Section 4 of experiments.md:
- Train baseline models (Raw, DMR, PMR)
- Task performance evaluation with SGN and MixFormer
- Physical plausibility metrics
"""

from typing import Dict, Any, List


class PrimaryExperiments:
    """Primary evaluation experiments for privacy vs utility analysis."""
    
    @staticmethod
    def get_experiment_configs() -> Dict[str, Any]:
        """Get all primary experiment configurations."""
        return {
            'privacy_utility_sgn': PrimaryExperiments.privacy_utility_sgn(),
            'privacy_utility_mixformer': PrimaryExperiments.privacy_utility_mixformer(),
            'baseline_comparison': PrimaryExperiments.baseline_comparison(),
            'physical_plausibility': PrimaryExperiments.physical_plausibility(),
            'cross_dataset_validation': PrimaryExperiments.cross_dataset_validation(),
        }
    
    @staticmethod
    def privacy_utility_sgn() -> Dict[str, Any]:
        """Privacy vs utility evaluation using SGN models."""
        return {
            'name': 'Privacy vs Utility - SGN',
            'description': 'Evaluate privacy-utility tradeoff using SGN for AR, RI, and GC tasks',
            'evaluation_type': 'privacy_utility',
            'models': {
                'raw': {
                    'type': 'raw',
                    'path': 'raw'
                },
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
                },
                'sgn_ri': {
                    'type': 'sgn',
                    'task': 'ri',
                    'path': 'output/ntu_ri_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                },
                'sgn_gc': {
                    'type': 'sgn',
                    'task': 'gc',
                    'path': 'output/ntu_gc_cview/model_best.pth.tar',
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
                },
                'ntu_cs': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cs',
                    'batch_size': 32,
                    'test_samples': 5000
                }
            },
            'metrics': [
                'action_recognition_accuracy',
                'reidentification_accuracy',
                'gender_classification_accuracy',
                'anonymization_rate',
                'privacy_utility_tradeoff'
            ],
            'output_dir': 'privacy_utility_sgn'
        }
    
    @staticmethod
    def privacy_utility_mixformer() -> Dict[str, Any]:
        """Privacy vs utility evaluation using MixFormer models."""
        return {
            'name': 'Privacy vs Utility - MixFormer',
            'description': 'Evaluate privacy-utility tradeoff using MixFormer for AR, RI, and GC tasks',
            'evaluation_type': 'privacy_utility',
            'models': {
                'raw': {
                    'type': 'raw',
                    'path': 'raw'
                },
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
                'mixformer_ar': {
                    'type': 'mixformer',
                    'task': 'ar',
                    'path': 'output/ntu_mixformer_ar_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                },
                'mixformer_ri': {
                    'type': 'mixformer',
                    'task': 'ri',
                    'path': 'output/ntu_mixformer_ri_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                },
                'mixformer_gc': {
                    'type': 'mixformer',
                    'task': 'gc',
                    'path': 'output/ntu_mixformer_gc_cview/model_best.pth.tar',
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
                },
                'ntu_cs': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cs',
                    'batch_size': 32,
                    'test_samples': 5000
                }
            },
            'metrics': [
                'action_recognition_accuracy',
                'reidentification_accuracy',
                'gender_classification_accuracy',
                'anonymization_rate',
                'privacy_utility_tradeoff'
            ],
            'output_dir': 'privacy_utility_mixformer'
        }
    
    @staticmethod
    def baseline_comparison() -> Dict[str, Any]:
        """Comprehensive baseline comparison across all models and tasks."""
        return {
            'name': 'Baseline Comparison',
            'description': 'Compare all baseline models (Raw, DMR, PMR, Transformer) across all tasks',
            'evaluation_type': 'baseline_comparison',
            'models': {
                'raw': {
                    'type': 'raw',
                    'path': 'raw'
                },
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
                },
                'sgn_ri': {
                    'type': 'sgn',
                    'task': 'ri',
                    'path': 'output/ntu_ri_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                },
                'mixformer_ar': {
                    'type': 'mixformer',
                    'task': 'ar',
                    'path': 'output/ntu_mixformer_ar_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                },
                'mixformer_ri': {
                    'type': 'mixformer',
                    'task': 'ri',
                    'path': 'output/ntu_mixformer_ri_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                }
            },
            'data': {
                'ntu_cv': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 32,
                    'test_samples': 10000
                }
            },
            'metrics': [
                'action_recognition_accuracy',
                'reidentification_accuracy',
                'mse',
                'bone_length_consistency',
                'temporal_smoothness',
                'velocity_consistency'
            ],
            'output_dir': 'baseline_comparison'
        }
    
    @staticmethod
    def physical_plausibility() -> Dict[str, Any]:
        """Comprehensive physical plausibility evaluation."""
        return {
            'name': 'Physical Plausibility',
            'description': 'Evaluate all physical plausibility metrics (BLC, JAL, TS, VC, FCC, FID)',
            'evaluation_type': 'physical_plausibility',
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
                    'batch_size': 16,
                    'test_samples': 2000
                }
            },
            'metrics': [
                'bone_length_consistency',
                'joint_angle_limits',
                'temporal_smoothness',
                'velocity_consistency',
                'foot_contact_consistency',
                'fid_score',
                'mse'
            ],
            'output_dir': 'physical_plausibility'
        }
    
    @staticmethod
    def cross_dataset_validation() -> Dict[str, Any]:
        """Cross-dataset validation experiment."""
        return {
            'name': 'Cross-Dataset Validation',
            'description': 'Train on NTU-60, test on NTU-120 and ETRI',
            'evaluation_type': 'cross_dataset',
            'models': {
                'transformer': {
                    'type': 'transformer',
                    'path': 'model.pth'
                }
            },
            'eval_models': {
                'sgn_ar_ntu120': {
                    'type': 'sgn',
                    'task': 'ar',
                    'path': 'output/ntu120_ar_cview/model_best.pth.tar',
                    'dataset': 'ntu120'
                },
                'sgn_ar_etri': {
                    'type': 'sgn',
                    'task': 'ar',
                    'path': 'output/etri_ar_cview/model_best.pth.tar',
                    'dataset': 'etri'
                }
            },
            'data': {
                'ntu120_cv': {
                    'type': 'paired',
                    'dataset': 'ntu120',
                    'setting': 'cv',
                    'batch_size': 32,
                    'test_samples': 3000
                },
                'etri': {
                    'type': 'paired',
                    'dataset': 'etri',
                    'setting': 'cv',
                    'batch_size': 32,
                    'test_samples': 1000
                }
            },
            'metrics': [
                'action_recognition_accuracy',
                'mse',
                'bone_length_consistency',
                'temporal_smoothness'
            ],
            'output_dir': 'cross_dataset_validation'
        }
