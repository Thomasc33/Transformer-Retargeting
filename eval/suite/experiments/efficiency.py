"""
Efficiency and generalization experiments.

Based on Section 6 of experiments.md:
- Cross-dataset validation
- Efficiency analysis (inference speed, memory usage)
"""

from typing import Dict, Any


class EfficiencyExperiments:
    """Efficiency and generalization experiments."""
    
    @staticmethod
    def get_experiment_configs() -> Dict[str, Any]:
        """Get all efficiency experiment configurations."""
        return {
            'cross_dataset_validation': EfficiencyExperiments.cross_dataset_validation(),
            'efficiency_analysis': EfficiencyExperiments.efficiency_analysis(),
        }
    
    @staticmethod
    def cross_dataset_validation() -> Dict[str, Any]:
        """Cross-dataset validation experiment."""
        return {
            'name': 'Cross-Dataset Validation',
            'description': 'Train on NTU-60, test on NTU-120 and ETRI',
            'evaluation_type': 'cross_dataset',
            'models': {
                'transformer_ntu60': {
                    'type': 'transformer',
                    'path': 'model.pth',  # Trained on NTU-60
                    'training_dataset': 'ntu'
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
                },
                'sgn_ri_ntu120': {
                    'type': 'sgn',
                    'task': 'ri',
                    'path': 'output/ntu120_ri_cview/model_best.pth.tar',
                    'dataset': 'ntu120'
                },
                'sgn_ri_etri': {
                    'type': 'sgn',
                    'task': 'ri',
                    'path': 'output/etri_ri_cview/model_best.pth.tar',
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
                'reidentification_accuracy',
                'mse',
                'bone_length_consistency',
                'temporal_smoothness',
                'cross_dataset_transfer_score'
            ],
            'analysis': {
                'domain_adaptation_analysis': True,
                'performance_degradation': True,
                'dataset_similarity_analysis': True
            },
            'output_dir': 'cross_dataset_validation'
        }
    
    @staticmethod
    def efficiency_analysis() -> Dict[str, Any]:
        """Efficiency analysis - inference speed and memory usage."""
        return {
            'name': 'Efficiency Analysis',
            'description': 'Measure inference speed and memory requirements',
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
            'eval_models': {
                'sgn_ar': {
                    'type': 'sgn',
                    'task': 'ar',
                    'path': 'output/ntu_ar_cview/model_best.pth.tar',
                    'dataset': 'ntu'
                }
            },
            'data': {
                'ntu_cv_small': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 1,  # Single sample for timing
                    'test_samples': 100
                },
                'ntu_cv_batch': {
                    'type': 'paired',
                    'dataset': 'ntu',
                    'setting': 'cv',
                    'batch_size': 32,  # Batch processing
                    'test_samples': 1000
                }
            },
            'metrics': [
                'inference_time_per_sample',
                'inference_time_per_batch',
                'memory_usage_peak',
                'memory_usage_average',
                'gpu_utilization',
                'throughput_samples_per_second',
                'model_size_mb',
                'flops_count'
            ],
            'benchmarks': {
                'single_sample_timing': True,
                'batch_processing_timing': True,
                'memory_profiling': True,
                'gpu_profiling': True,
                'model_complexity_analysis': True
            },
            'hardware_configs': [
                {
                    'name': 'single_gpu',
                    'gpus': 1,
                    'batch_size': 32
                },
                {
                    'name': 'cpu_only',
                    'gpus': 0,
                    'batch_size': 8
                }
            ],
            'output_dir': 'efficiency_analysis'
        }
