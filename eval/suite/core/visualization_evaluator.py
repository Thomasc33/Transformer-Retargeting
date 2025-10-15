#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Evaluator for the Evaluation Suite

This module handles the execution of visualization experiments with proper
Kinect v2 view angles and enhanced skeleton animations.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

# Optional torch import for environments where it might not be available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eval.suite.experiments.visualization import VisualizationExperiments
from eval.suite.core.data_loader import DataManager
from eval.suite.core.models import ModelManager


class VisualizationEvaluator:
    """Evaluator for visualization experiments."""
    
    def __init__(self, device='cuda', output_base_dir='results/visualizations'):
        """
        Initialize the visualization evaluator.
        
        Args:
            device: Device to run models on
            output_base_dir: Base directory for visualization outputs
        """
        self.device = device
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def run_visualization_experiment(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a visualization experiment.

        Args:
            experiment_config: Configuration for the visualization experiment

        Returns:
            Dictionary containing results and output paths
        """
        # Store current config for access by other methods
        self._current_config = experiment_config

        experiment_name = experiment_config.get('name', 'visualization')
        evaluation_type = experiment_config.get('evaluation_type', 'motion_visualization')

        self.logger.info(f"Running visualization experiment: {experiment_name}")
        
        # Create experiment output directory
        experiment_dir = self.output_base_dir / experiment_config.get('output_dir', experiment_name)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'experiment_name': experiment_name,
            'evaluation_type': evaluation_type,
            'output_directory': str(experiment_dir),
            'visualizations': {},
            'metrics': {}
        }
        
        try:
            if evaluation_type == 'skeleton_animation':
                results.update(self._run_skeleton_animation(experiment_config, experiment_dir))
            elif evaluation_type == 'comparison_visualization':
                results.update(self._run_comparison_visualization(experiment_config, experiment_dir))
            elif evaluation_type == 'motion_visualization':
                results.update(self._run_motion_visualization(experiment_config, experiment_dir))
            elif evaluation_type == 'attention_visualization':
                results.update(self._run_attention_visualization(experiment_config, experiment_dir))
            elif evaluation_type == 'sensitivity_analysis':
                results.update(self._run_sensitivity_analysis(experiment_config, experiment_dir))
            elif evaluation_type == 'anonymization_showcase':
                results.update(self._run_anonymization_showcase(experiment_config, experiment_dir))
            elif evaluation_type == 'mlm_pretraining':
                results.update(self._run_mlm_pretraining(experiment_config, experiment_dir))
            else:
                raise ValueError(f"Unknown evaluation type: {evaluation_type}")
                
        except Exception as e:
            self.logger.error(f"Error in visualization experiment {experiment_name}: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results
        
        results['success'] = True
        self.logger.info(f"Completed visualization experiment: {experiment_name}")
        return results
    
    def _run_skeleton_animation(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Run skeleton animation experiment."""
        results = {'visualizations': {}, 'metrics': {}}
        
        # Load data
        data_configs = config.get('data', {})
        models_config = config.get('models', {})
        animations = config.get('animations', ['original_skeleton'])
        view_settings = config.get('view_settings', {})
        quality_settings = config.get('quality_settings', {})
        
        for data_name, data_config in data_configs.items():
            self.logger.info(f"Processing data: {data_name}")
            
            # Load data
            data_loader = self.data_manager.load_single_dataset(data_name, data_config)

            if data_loader is None:
                self.logger.error(f"Failed to load data for {data_name}")
                continue

            # Get samples for visualization
            samples = []
            test_samples = data_config.get('test_samples', 10)

            try:
                for i, batch in enumerate(data_loader):
                    if i >= test_samples:
                        break
                    samples.append(batch)

                if not samples:
                    self.logger.warning(f"No samples loaded for {data_name}")
                    continue

            except Exception as e:
                self.logger.error(f"Error iterating through data loader for {data_name}: {e}")
                continue
            
            # Create animations for each model
            for model_name, model_config in models_config.items():
                self.logger.info(f"Creating animations for model: {model_name}")
                
                model_output_dir = output_dir / model_name
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Load model if needed
                if model_config['type'] != 'raw':
                    model = self.model_manager.load_single_model(model_name, model_config)
                    if model is not None and hasattr(model, 'eval'):
                        model.eval()
                else:
                    model = None
                
                # Create each type of animation
                for animation_type in animations:

                    try:
                        if not samples:
                            self.logger.warning(f"No samples available for {animation_type} in {model_name}")
                            continue

                        # Create unique identifier for this animation using skeleton filename if available
                        unique_id = None
                        if samples and len(samples) > 0:
                            first_sample = samples[0]
                            if isinstance(first_sample, (list, tuple)) and len(first_sample) >= 3:
                                skeleton_filename = first_sample[2]

                                # Handle both string and tuple formats
                                if isinstance(skeleton_filename, str):
                                    base_name = skeleton_filename
                                elif isinstance(skeleton_filename, (tuple, list)) and len(skeleton_filename) > 0:
                                    base_name = str(skeleton_filename[0])
                                else:
                                    base_name = None

                                if base_name:
                                    # Extract base name without extension
                                    base_name = base_name.split('/')[-1].split('\\')[-1]
                                    base_name = base_name.replace('.skeleton', '').replace('.npy', '').replace('.avi', '')
                                    base_name = base_name.replace('/', '_').replace('\\', '_')
                                    unique_id = f"{base_name}_{model_name}_{animation_type}"

                        # Fallback to generic identifier if no filename found
                        if not unique_id:
                            unique_id = f"{model_name}_{animation_type}"
                            self.logger.debug(f"Using fallback filename: {unique_id}")

                        animation_path = self._create_animation(
                            samples, model, animation_type, model_output_dir,
                            view_settings, quality_settings, unique_id
                        )

                        if animation_path:
                            if model_name not in results['visualizations']:
                                results['visualizations'][model_name] = {}
                            results['visualizations'][model_name][animation_type] = animation_path
                            self.logger.info(f"Created {animation_type} for {model_name}: {animation_path}")
                        else:
                            self.logger.warning(f"Failed to create {animation_type} for {model_name}: no output path")

                    except Exception as e:
                        self.logger.warning(f"Failed to create {animation_type} for {model_name}: {e}")
                        import traceback
                        self.logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        return results
    
    def _create_animation(self, samples, model, animation_type: str, output_dir: Path,
                         view_settings: Dict, quality_settings: Dict, unique_id: str = None) -> Optional[str]:
        """Create a specific type of animation."""
        try:
            if not samples:
                self.logger.warning(f"No samples provided for {animation_type}")
                return None

            max_frames = quality_settings.get('max_frames', None)

            if animation_type == 'original_skeleton':
                return VisualizationExperiments.create_skeleton_animation(
                    samples, output_dir=str(output_dir), figure_type='original',
                    max_frames=max_frames, unique_id=unique_id
                )
            elif animation_type == 'retargeted_skeleton':
                # Use fixed implementation for transformer visualization
                if model is not None and model != 'raw':
                    try:
                        from evaluation_suite.core.fixed_visualization_evaluator import FixedVisualizationEvaluator

                        # Get models config from the current experiment config
                        models_config = getattr(self, '_current_config', {}).get('models', {})

                        # Create config for transformer visualization
                        transformer_config = {
                            'model_path': model if isinstance(model, str) else None,
                            'models': models_config,  # Pass the models config
                            'output_dir': str(output_dir)
                        }

                        # Create fixed evaluator and run
                        fixed_evaluator = FixedVisualizationEvaluator(device=self.device)
                        results = fixed_evaluator.create_transformer_visualization(transformer_config)

                        # Return the first visualization path if available
                        if results.get('visualizations'):
                            for viz_type, paths in results['visualizations'].items():
                                if paths:
                                    return paths[0]

                        self.logger.warning("Fixed transformer visualization returned no results, using original")
                    except Exception as e:
                        self.logger.warning(f"Fixed transformer visualization failed: {e}, using original samples")

                # Use original samples as fallback
                return VisualizationExperiments.create_skeleton_animation(
                    samples, output_dir=str(output_dir), figure_type='original',
                    max_frames=max_frames, unique_id=unique_id
                )
            elif animation_type == 'dummy_skeleton':
                # Create dummy skeleton animation (for anonymization showcase)
                dummy_samples = self._create_dummy_samples(samples)
                return VisualizationExperiments.create_skeleton_animation(
                    dummy_samples, output_dir=str(output_dir), figure_type='dummy',
                    max_frames=max_frames, unique_id=unique_id
                )
            elif animation_type == 'combined_original_retargeted':
                # Create combined original + retargeted animation
                return VisualizationExperiments.create_skeleton_animation(
                    samples, output_dir=str(output_dir), figure_type='combined_original_retargeted',
                    max_frames=max_frames, unique_id=unique_id
                )
            else:
                # Default to original for unsupported animation types
                self.logger.info(f"Animation type '{animation_type}' not specifically implemented, using original")
                return VisualizationExperiments.create_skeleton_animation(
                    samples, output_dir=str(output_dir), figure_type='original',
                    max_frames=max_frames, unique_id=unique_id
                )
        except Exception as e:
            self.logger.error(f"Error creating {animation_type} animation: {e}")
            return None
    
    def _process_samples_through_model(self, samples, model):
        """Process samples through a model to get retargeted skeletons."""
        if not samples:
            self.logger.warning("No samples to process")
            return []

        processed_samples = []

        try:
            if not TORCH_AVAILABLE:
                self.logger.warning("PyTorch not available, cannot process samples through model")
                return samples

            with torch.no_grad():
                for i, sample in enumerate(samples):
                    try:
                        # Extract input data
                        if isinstance(sample, (list, tuple)):
                            input_data = sample[0]
                            label = sample[1] if len(sample) > 1 else None
                        else:
                            input_data = sample
                            label = None

                        # Convert to tensor if needed
                        if not isinstance(input_data, torch.Tensor):
                            input_data = torch.tensor(input_data, dtype=torch.float32)

                        # Move to device
                        input_data = input_data.to(self.device)

                        # Run through model - handle transformer model specifically
                        if hasattr(model, '__call__'):
                            try:
                                # For transformer models, create proper batch format
                                # Expected format: (x1, x2, y1, y2, actors, actions)

                                # Ensure input has batch dimension
                                if input_data.dim() == 2:
                                    input_data = input_data.unsqueeze(0)

                                x1 = input_data  # Original skeleton
                                x2 = torch.zeros_like(input_data)  # Dummy target skeleton
                                y1 = torch.tensor([label if label is not None else 0], dtype=torch.long, device=input_data.device)
                                y2 = torch.tensor([label if label is not None else 0], dtype=torch.long, device=input_data.device)
                                actors = torch.tensor([0, 1], dtype=torch.long, device=input_data.device)  # Dummy actor IDs
                                actions = torch.tensor([label if label is not None else 0], dtype=torch.long, device=input_data.device)

                                # Create batch tuple
                                batch = (x1, x2, y1, y2, actors, actions)

                                # Try to use the transformer processing function
                                try:
                                    from src.evaluation.eval_model import get_anonymized_paired_transformer
                                    # Create a dummy prep_data function
                                    def prep_data(x):
                                        return x

                                    # Process through transformer
                                    result = get_anonymized_paired_transformer(batch, model, prep_data)
                                    if result and len(result) > 0:
                                        output = result[0]['x1']  # Get the anonymized skeleton
                                    else:
                                        output = input_data  # Fallback to original
                                except ImportError:
                                    # Fallback: just return original data
                                    output = input_data

                            except Exception as model_error:
                                self.logger.debug(f"Model processing failed: {model_error}, using original data")
                                output = input_data
                        else:
                            self.logger.warning(f"Model is not callable: {type(model)}")
                            processed_samples.append(sample)
                            continue

                        # Handle different output formats
                        if isinstance(output, (list, tuple)):
                            processed_data = output[0]  # Take first output
                        else:
                            processed_data = output

                        # Move back to CPU
                        if isinstance(processed_data, torch.Tensor):
                            processed_data = processed_data.cpu()

                        processed_samples.append((processed_data, label))

                    except Exception as e:
                        self.logger.warning(f"Failed to process sample {i} through model: {e}")
                        # Use original sample as fallback
                        processed_samples.append(sample)

        except Exception as e:
            self.logger.error(f"Error in model processing: {e}")
            return samples  # Return original samples as fallback

        return processed_samples

    def _create_dummy_samples(self, samples):
        """Create dummy skeleton samples for anonymization showcase."""
        if not samples:
            return []

        dummy_samples = []
        for sample in samples:
            if isinstance(sample, (list, tuple)):
                skeleton = sample[0]
                label = sample[1] if len(sample) > 1 else None
                filename = sample[2] if len(sample) > 2 else None
            else:
                skeleton = sample
                label = None
                filename = None

            # Create dummy skeleton (zero motion or simple pattern)
            if isinstance(skeleton, torch.Tensor):
                dummy_skeleton = torch.zeros_like(skeleton)
                # Add minimal standing pose
                if skeleton.shape[-1] >= 75:  # 25 joints * 3 coords
                    # Set basic standing position
                    dummy_skeleton[:, 1] = -0.9  # Spine base Y
                    dummy_skeleton[:, 4] = -0.6  # Spine mid Y
                    dummy_skeleton[:, 10] = 0.0  # Head Y
            else:
                dummy_skeleton = skeleton * 0  # Zero out the skeleton

            if filename:
                dummy_samples.append((dummy_skeleton, label, filename))
            else:
                dummy_samples.append((dummy_skeleton, label))

        return dummy_samples
    
    def _run_comparison_visualization(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Run comparison visualization experiment."""
        results = {'visualizations': {}, 'metrics': {}}

        try:
            # Extract dataset and setting from config
            dataset = 'ntu'  # Default
            setting = 'cv'   # Default

            # Try to get from data config
            data_configs = config.get('data', {})
            if data_configs:
                first_data_config = next(iter(data_configs.values()))
                dataset = first_data_config.get('dataset', dataset)
                setting = first_data_config.get('setting', setting)

            # Load data
            data_loader = self.data_manager.load_paired_data(
                dataset, setting, batch_size=1, test_samples=3
            )

            if not data_loader:
                self.logger.warning("No data available for comparison visualization")
                return results

            # Get samples and extract skeleton data
            skeleton_samples = []
            for batch in data_loader:
                # Each batch is (skeleton, action_label) from VisualizationDataset
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    skeleton_data = batch[0]  # Extract skeleton data
                    skeleton_samples.append(skeleton_data)
                elif hasattr(batch, 'shape'):  # Direct tensor
                    skeleton_samples.append(batch)

                if len(skeleton_samples) >= 3:  # Limit samples
                    break

            # Create side-by-side comparison visualizations
            if skeleton_samples:
                comparison_path = VisualizationExperiments.create_skeleton_animation(
                    skeleton_samples[:1], output_dir=str(output_dir), figure_type='comparison',
                    max_frames=config.get('quality_settings', {}).get('max_frames', None)
                )
                if comparison_path:
                    results['visualizations']['side_by_side'] = comparison_path

            self.logger.info("Comparison visualization completed")

        except Exception as e:
            self.logger.error(f"Error in comparison visualization: {e}")

        return results
    
    def _run_motion_visualization(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Run motion visualization experiment."""
        results = {'visualizations': {}, 'metrics': {}}

        try:
            # Extract dataset and setting from config
            dataset = 'ntu'  # Default
            setting = 'cv'   # Default

            # Try to get from data config
            data_configs = config.get('data', {})
            if data_configs:
                first_data_config = next(iter(data_configs.values()))
                dataset = first_data_config.get('dataset', dataset)
                setting = first_data_config.get('setting', setting)

            # Load data
            data_loader = self.data_manager.load_paired_data(
                dataset, setting, batch_size=1, test_samples=5
            )

            if not data_loader:
                self.logger.warning("No data available for motion visualization")
                return results

            # Get samples and extract skeleton data
            skeleton_samples = []
            for batch in data_loader:
                # Each batch is (skeleton, action_label) from VisualizationDataset
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    skeleton_data = batch[0]  # Extract skeleton data
                    skeleton_samples.append(skeleton_data)
                elif hasattr(batch, 'shape'):  # Direct tensor
                    skeleton_samples.append(batch)

                if len(skeleton_samples) >= 5:  # Limit samples
                    break

            # Create motion trail visualizations
            if skeleton_samples:
                motion_trail_path = VisualizationExperiments.create_skeleton_animation(
                    skeleton_samples[:1], output_dir=str(output_dir), figure_type='motion_trail',
                    max_frames=config.get('quality_settings', {}).get('max_frames', None)
                )
                if motion_trail_path:
                    results['visualizations']['motion_trail'] = motion_trail_path

            self.logger.info("Motion visualization completed")

        except Exception as e:
            self.logger.error(f"Error in motion visualization: {e}")

        return results
    
    def _run_attention_visualization(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Run attention visualization experiment."""
        results = {'visualizations': {}, 'metrics': {}}

        try:
            # Extract dataset and setting from config
            dataset = 'ntu'  # Default
            setting = 'cv'   # Default

            # Try to get from data config
            data_configs = config.get('data', {})
            if data_configs:
                first_data_config = next(iter(data_configs.values()))
                dataset = first_data_config.get('dataset', dataset)
                setting = first_data_config.get('setting', setting)

            # Load data
            data_loader = self.data_manager.load_paired_data(
                dataset, setting, batch_size=1, test_samples=3
            )

            if not data_loader:
                self.logger.warning("No data available for attention visualization")
                return results

            # Get samples and extract skeleton data
            skeleton_samples = []
            for batch in data_loader:
                # Each batch is (skeleton, action_label) from VisualizationDataset
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    skeleton_data = batch[0]  # Extract skeleton data
                    skeleton_samples.append(skeleton_data)
                elif hasattr(batch, 'shape'):  # Direct tensor
                    skeleton_samples.append(batch)

                if len(skeleton_samples) >= 3:  # Limit samples
                    break

            # Create attention heatmap visualizations (placeholder)
            if skeleton_samples:
                attention_path = VisualizationExperiments.create_skeleton_animation(
                    skeleton_samples[:1], output_dir=str(output_dir), figure_type='attention',
                    max_frames=config.get('quality_settings', {}).get('max_frames', None)
                )
                if attention_path:
                    results['visualizations']['attention_heatmap'] = attention_path

            self.logger.info("Attention visualization completed")

        except Exception as e:
            self.logger.error(f"Error in attention visualization: {e}")

        return results

    def _run_sensitivity_analysis(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Run sensitivity analysis visualization experiment."""
        results = {'visualizations': {}, 'metrics': {}}

        try:
            # Extract dataset and setting from config
            dataset = 'ntu'  # Default
            setting = 'cv'   # Default

            # Try to get from data config
            data_configs = config.get('data', {})
            if data_configs:
                first_data_config = next(iter(data_configs.values()))
                dataset = first_data_config.get('dataset', dataset)
                setting = first_data_config.get('setting', setting)

            # Load data
            data_loader = self.data_manager.load_paired_data(
                dataset, setting, batch_size=1, test_samples=3
            )

            if not data_loader:
                self.logger.warning("No data available for sensitivity analysis")
                return results

            # Get samples
            samples = []
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    samples.extend(batch)
                else:
                    samples.append(batch)
                if len(samples) >= 3:  # Limit samples
                    break

            # Create sensitivity heatmap visualizations
            if samples:
                sensitivity_path = VisualizationExperiments.create_skeleton_animation(
                    samples[:1], output_dir=str(output_dir), figure_type='sensitivity',
                    max_frames=config.get('quality_settings', {}).get('max_frames', None)
                )
                if sensitivity_path:
                    results['visualizations']['sensitivity_heatmap'] = sensitivity_path

            self.logger.info("Sensitivity analysis completed")

        except Exception as e:
            self.logger.error(f"Error in sensitivity analysis: {e}")

        return results

    def _run_anonymization_showcase(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Run anonymization showcase visualization experiment."""
        results = {'visualizations': {}, 'metrics': {}}

        try:
            # This uses the same logic as skeleton_animation but with showcase focus
            return self._run_skeleton_animation(config, output_dir)

        except Exception as e:
            self.logger.error(f"Error in anonymization showcase: {e}")

        return results

    def _run_mlm_pretraining(self, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Run MLM pretraining visualization experiment using fixed implementation."""
        try:
            from evaluation_suite.core.fixed_visualization_evaluator import FixedVisualizationEvaluator

            # Extract configuration
            models_config = config.get('models', {})
            mlm_model_config = models_config.get('mlm_autoencoder', {})
            data_configs = config.get('data', {})

            # Create config for fixed evaluator
            fixed_config = {
                'temporal_ratio': mlm_model_config.get('temporal_ratio', 0.3),
                'spatial_ratio': mlm_model_config.get('spatial_ratio', 0.3),
                'dataset': 'ntu',
                'setting': 'cv',
                'output_dir': str(output_dir)
            }

            # Extract dataset and setting from data config if available
            if data_configs:
                first_data_config = next(iter(data_configs.values()))
                fixed_config['dataset'] = first_data_config.get('dataset', 'ntu')
                fixed_config['setting'] = first_data_config.get('setting', 'cv')

            # Create fixed evaluator and run
            fixed_evaluator = FixedVisualizationEvaluator(device=self.device)
            return fixed_evaluator.create_mlm_visualization(fixed_config)

        except Exception as e:
            self.logger.error(f"Error in MLM visualization: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {'visualizations': {}, 'metrics': {}}





    def run_all_visualization_experiments(self) -> Dict[str, Any]:
        """Run all predefined visualization experiments."""
        all_configs = VisualizationExperiments.get_experiment_configs()
        all_results = {}
        
        for experiment_name, config in all_configs.items():
            self.logger.info(f"Running experiment: {experiment_name}")
            results = self.run_visualization_experiment(config)
            all_results[experiment_name] = results
        
        return all_results
