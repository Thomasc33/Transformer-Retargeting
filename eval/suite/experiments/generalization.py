"""
Generalization and efficiency experiments.

Based on Section 6 of experiments.md:
- Cross-dataset validation (NTU-60 → NTU-120, ETRI)
- Efficiency analysis (inference speed, memory, computational efficiency)
"""

from typing import Dict, Any


class GeneralizationExperiments:
    """Generalization and efficiency experiments."""
    
    @staticmethod
    def get_experiment_configs() -> Dict[str, Any]:
        """Get all generalization experiment configurations."""
        return {
            'cross_dataset_validation': GeneralizationExperiments.cross_dataset_validation(),
            'efficiency_analysis': GeneralizationExperiments.efficiency_analysis(),
        }
    
    @staticmethod
    def cross_dataset_validation() -> Dict[str, Any]:
        """Cross-dataset validation: Train on NTU-60, test on NTU-120 and ETRI."""
        return {
            'name': 'Cross-Dataset Validation',
            'description': 'Train on NTU-60, test on NTU-120 and ETRI to evaluate generalization',
            'evaluation_type': 'cross_dataset',
            'models': {
                'transformer_ntu60': {
                    'type': 'transformer',
                    'path': 'model.pth',  # Trained on NTU-60
                    'training_dataset': 'ntu60'
                },
                'dmr_ntu60': {
                    'type': 'dmr',
                    'path': 'trained_models/dmr_ntu_cv_final.pth',
                    'training_dataset': 'ntu60'
                },
                'pmr_ntu60': {
                    'type': 'pmr',
                    'path': 'trained_models/pmr_ntu_cv_final.pth',
                    'training_dataset': 'ntu60'
                }
            },
            'eval_models': {
                'sgn_ar_ntu120': {
                    'type': 'sgn',
                    'task': 'ar',
                    'path': 'output/ntu120_ar_cview/model_best.pth.tar',
                    'dataset': 'ntu120'
                },
                'sgn_ri_ntu120': {
                    'type': 'sgn',
                    'task': 'ri',
                    'path': 'output/ntu120_ri_cview/model_best.pth.tar',
                    'dataset': 'ntu120'
                },
                'sgn_ar_etri': {
                    'type': 'sgn',
                    'task': 'ar',
                    'path': 'output/etri_ar_cview/model_best.pth.tar',
                    'dataset': 'etri'
                },
                'sgn_ri_etri': {
                    'type': 'sgn',
                    'task': 'ri',
                    'path': 'output/etri_ri_cview/model_best.pth.tar',
                    'dataset': 'etri'
                },
                'mixformer_ar_ntu120': {
                    'type': 'mixformer',
                    'task': 'ar',
                    'path': 'output/ntu120_mixformer_ar_cview/model_best.pth.tar',
                    'dataset': 'ntu120'
                },
                'mixformer_ar_etri': {
                    'type': 'mixformer',
                    'task': 'ar',
                    'path': 'output/etri_mixformer_ar_cview/model_best.pth.tar',
                    'dataset': 'etri'
                }
            },
            'data': {
                'ntu120_cv': {
                    'type': 'paired',
                    'dataset': 'ntu120',
                    'setting': 'cv',
                    'batch_size': 32,
                    'test_samples': 5000
                },
                'etri': {
                    'type': 'paired',
                    'dataset': 'etri',
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
                'temporal_smoothness',
                'cross_dataset_transfer_score',
                'domain_adaptation_effectiveness'
            ],
            'analysis': {
                'performance_degradation_analysis': True,
                'dataset_similarity_analysis': True,
                'domain_gap_quantification': True,
                'generalization_capability_assessment': True
            },
            'output_dir': 'cross_dataset_validation'
        }
    
    @staticmethod
    def efficiency_analysis() -> Dict[str, Any]:
        """Efficiency analysis: inference speed, memory, computational efficiency."""
        return {
            'name': 'Efficiency Analysis',
            'description': 'Measure inference speed (FPS), memory requirements, and computational efficiency',
            'evaluation_type': 'efficiency',
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
                'ntu_cv_single': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 1,  # Single sample for precise timing
                    'test_samples': 100
                },
                'ntu_cv_batch_small': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 8,
                    'test_samples': 500
                },
                'ntu_cv_batch_large': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 32,
                    'test_samples': 1000
                }
            },
            'metrics': [
                'inference_time_per_sample_ms',
                'inference_time_per_batch_ms',
                'frames_per_second_fps',
                'memory_usage_peak_mb',
                'memory_usage_average_mb',
                'gpu_utilization_percent',
                'cpu_utilization_percent',
                'throughput_samples_per_second',
                'model_size_mb',
                'model_parameters_count',
                'flops_count',
                'energy_consumption_estimate'
            ],
            'benchmarks': {
                'single_sample_timing': {
                    'description': 'Measure inference time for single samples',
                    'iterations': 100,
                    'warmup_iterations': 10
                },
                'batch_processing_timing': {
                    'description': 'Measure inference time for different batch sizes',
                    'batch_sizes': [1, 4, 8, 16, 32],
                    'iterations': 50
                },
                'memory_profiling': {
                    'description': 'Profile memory usage during inference',
                    'profile_duration': 60,  # seconds
                    'sampling_interval': 0.1  # seconds
                },
                'gpu_profiling': {
                    'description': 'Profile GPU utilization and memory',
                    'profile_duration': 60,
                    'sampling_interval': 0.1
                },
                'scalability_analysis': {
                    'description': 'Test performance with increasing load',
                    'concurrent_requests': [1, 2, 4, 8, 16],
                    'duration_per_test': 30
                }
            },
            'hardware_configs': [
                {
                    'name': 'single_gpu_v100',
                    'description': 'Single V100 GPU configuration',
                    'gpus': 1,
                    'gpu_type': 'V100',
                    'batch_size': 32
                },
                {
                    'name': 'single_gpu_a100',
                    'description': 'Single A100 GPU configuration',
                    'gpus': 1,
                    'gpu_type': 'A100',
                    'batch_size': 64
                },
                {
                    'name': 'cpu_only',
                    'description': 'CPU-only configuration',
                    'gpus': 0,
                    'batch_size': 8
                }
            ],
            'analysis': {
                'performance_comparison': True,
                'scalability_analysis': True,
                'resource_efficiency': True,
                'real_time_capability': True,
                'cost_effectiveness': True
            },
            'output_dir': 'efficiency_analysis'
        }
