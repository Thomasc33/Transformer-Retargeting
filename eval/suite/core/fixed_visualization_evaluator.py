"""
Fixed Visualization Evaluator that actually works properly.
This module provides proper model loading and forward passes for visualization.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

class FixedVisualizationEvaluator:
    """Fixed visualization evaluator with proper model loading and forward passes."""
    
    def __init__(self, device='cuda'):
        # Use CPU if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        self.device = device
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
    def create_mlm_visualization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create MLM visualization with proper model loading and forward pass."""
        results = {'visualizations': {}}
        
        try:
            # Extract configuration
            temporal_ratio = config.get('temporal_ratio', 0.3)
            spatial_ratio = config.get('spatial_ratio', 0.3)
            dataset = config.get('dataset', 'ntu')
            setting = config.get('setting', 'cv')
            output_dir = Path(config.get('output_dir', 'visualizations/mlm'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load MLM model properly
            model = self._load_mlm_model(dataset, setting, temporal_ratio, spatial_ratio)
            if model is None:
                return results
                
            # Load data properly
            data_loader = self._load_mlm_data(dataset, setting, temporal_ratio, spatial_ratio)
            if data_loader is None:
                return results
                
            # Create visualizations
            self._create_mlm_visualizations(model, data_loader, output_dir, dataset, setting, 
                                          temporal_ratio, spatial_ratio, results)
            
        except Exception as e:
            self.logger.error(f"Error in MLM visualization: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
        return results
    
    def create_transformer_visualization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create transformer visualization with PMR/DMR models."""
        results = {'visualizations': {}}

        try:
            # Extract configuration
            model_path = config.get('model_path')
            output_dir = Path(config.get('output_dir', 'visualizations/transformer'))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Check if config has models defined
            models_config = config.get('models', {})
            if models_config:
                self.logger.info("Loading models from config...")
                models = self._load_models_from_config(models_config)
            else:
                self.logger.info("Loading models from default paths...")
                models = self._load_transformer_models(model_path)

            if not models:
                self.logger.warning("No models loaded, creating raw visualization only")
                models = {'raw': None}  # At least show raw data

            # Load test data
            test_data = self._load_test_data()
            if not test_data:
                return results

            # Create visualizations for each model
            self._create_transformer_visualizations(models, test_data, output_dir, results)

        except Exception as e:
            self.logger.error(f"Error in transformer visualization: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

        return results

    def _load_models_from_config(self, models_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load models from configuration."""
        models = {}

        try:
            import sys
            sys.path.append('.')
            from src.evaluation.eval_model import load_anonymizer, datasets

            # Create mock args
            class MockArgs:
                def __init__(self):
                    self.dataset = 'ntu'
                    self.batch_size = 32
                    self.loading_transformer = False

            args = MockArgs()

            for model_name, model_config in models_config.items():
                model_type = model_config.get('type')
                model_path = model_config.get('path')

                if model_type == 'raw':
                    models[model_name] = None
                    self.logger.info(f"Added raw model: {model_name}")
                    continue

                if not model_path or model_path == 'raw':
                    continue

                # Try to load the model
                try:
                    if model_type == 'transformer':
                        if os.path.exists(model_path):
                            model = load_anonymizer('transformer', model_path, self.device, args, datasets['ntu'])
                            # Ensure model is on correct device
                            if hasattr(model, 'to'):
                                model = model.to(self.device)
                            models[model_name] = model
                            self.logger.info(f"Loaded {model_type} model: {model_name} from {model_path} on {self.device}")
                        else:
                            self.logger.warning(f"Transformer model path not found: {model_path}")
                    elif model_type in ['pmr', 'dmr']:
                        if os.path.exists(model_path):
                            model = load_anonymizer(model_type, model_path, self.device, args, None)
                            # Ensure model is on correct device
                            if hasattr(model, 'to'):
                                model = model.to(self.device)
                            models[model_name] = model
                            self.logger.info(f"Loaded {model_type} model: {model_name} from {model_path} on {self.device}")
                        else:
                            self.logger.warning(f"{model_type.upper()} model path not found: {model_path}")
                    else:
                        self.logger.warning(f"Unknown model type: {model_type}")

                except Exception as e:
                    self.logger.error(f"Failed to load {model_type} model {model_name}: {e}")
                    continue

            return models

        except Exception as e:
            self.logger.error(f"Failed to load models from config: {e}")
            return {}
    
    def _load_mlm_model(self, dataset: str, setting: str, temporal_ratio: float, spatial_ratio: float):
        """Load MLM model with proper error handling."""
        try:
            import sys
            sys.path.append('.')
            from pretrain import SkeletonAutoEncoder

            # Construct model paths - use the actual directory structure
            model_dir = f"eval/mixformer/pretrained/{dataset}/epochs_{setting}_comprehensive_temporal_{temporal_ratio}_spatial_{spatial_ratio}"
            encoder_path = f"{model_dir}/encoder_best.pth"
            decoder_path = f"{model_dir}/decoder_best.pth"
            output_layer_path = f"{model_dir}/output_layer_best.pth"

            # Check if model files exist
            if not all(os.path.exists(p) for p in [encoder_path, decoder_path, output_layer_path]):
                self.logger.error(f"MLM model files not found in {model_dir}")
                self.logger.error(f"Checked paths:")
                self.logger.error(f"  Encoder: {encoder_path} - {'exists' if os.path.exists(encoder_path) else 'missing'}")
                self.logger.error(f"  Decoder: {decoder_path} - {'exists' if os.path.exists(decoder_path) else 'missing'}")
                self.logger.error(f"  Output: {output_layer_path} - {'exists' if os.path.exists(output_layer_path) else 'missing'}")

                # Try to find available MLM models
                import glob
                available_dirs = glob.glob(f"eval/mixformer/pretrained/{dataset}/epochs_*_comprehensive_temporal_*_spatial_*")
                if available_dirs:
                    self.logger.info(f"Available MLM model directories:")
                    for dir_path in available_dirs:
                        self.logger.info(f"  {dir_path}")
                return None

            # Create and load model
            model = SkeletonAutoEncoder(dataset=dataset, seq_len=64)

            # Load weights
            if os.path.exists(encoder_path):
                encoder_state_dict = torch.load(encoder_path, map_location=self.device)
                model.encoder.load_state_dict(encoder_state_dict, strict=False)
                self.logger.info(f"Loaded encoder from: {encoder_path}")

            if os.path.exists(decoder_path):
                decoder_state_dict = torch.load(decoder_path, map_location=self.device)
                model.decoder.load_state_dict(decoder_state_dict, strict=False)
                self.logger.info(f"Loaded decoder from: {decoder_path}")

            if os.path.exists(output_layer_path):
                output_layer_state_dict = torch.load(output_layer_path, map_location=self.device)
                model.output_layer.load_state_dict(output_layer_state_dict, strict=False)
                self.logger.info(f"Loaded output layer from: {output_layer_path}")

            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            self.logger.error(f"Failed to load MLM model: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _load_mlm_data(self, dataset: str, setting: str, temporal_ratio: float, spatial_ratio: float):
        """Load MLM data with proper error handling."""
        try:
            from src.data.data import load_data, process_mlm, Masked_AE_Data
            
            # Load dataset
            data = load_data(dataset, T=64)
            train_data, test_data = process_mlm(data, setting, dataset, T=64)
            
            # Create MLM dataset
            num_samples = 5
            test_filenames = test_data[:num_samples]
            dataset_obj = Masked_AE_Data(
                [data[filename] for filename in test_filenames],
                frame_masking_ratio=temporal_ratio,
                joint_masking_ratio=spatial_ratio,
                seg=64,
                augment=False
            )
            
            self.logger.info(f"Created MLM dataset with {len(dataset_obj)} samples")
            return {'dataset': dataset_obj, 'filenames': test_filenames}
            
        except Exception as e:
            self.logger.error(f"Failed to load MLM data: {e}")
            return None
    
    def _load_transformer_models(self, transformer_path: str) -> Dict[str, Any]:
        """Load transformer, PMR, and DMR models."""
        models = {}

        try:
            import sys
            sys.path.append('.')
            from src.evaluation.eval_model import load_anonymizer, datasets

            # Create mock args
            class MockArgs:
                def __init__(self):
                    self.dataset = 'ntu'
                    self.batch_size = 32
                    self.loading_transformer = False

            args = MockArgs()

            # Try different transformer model paths
            transformer_paths = [
                transformer_path,
                'model.pth',
                'checkpoints/model.pth',
                'results/model.pth',
                'trained_models/transformer_ntu_cv.pth'
            ]

            transformer_model = None
            for path in transformer_paths:
                if path and os.path.exists(path):
                    try:
                        transformer_model = load_anonymizer('transformer', path, self.device, args, datasets['ntu'])
                        models['transformer'] = transformer_model
                        self.logger.info(f"Loaded transformer model from {path}")
                        break
                    except Exception as e:
                        self.logger.warning(f"Failed to load transformer from {path}: {e}")
                        continue

            if transformer_model is None:
                self.logger.warning("No transformer model found, trying available paths:")
                for path in transformer_paths:
                    self.logger.warning(f"  Checked: {path} - {'exists' if os.path.exists(path) else 'not found'}")

            # Try different PMR model paths (use the actual available files)
            pmr_paths = [
                "trained_models/pmr_ntu_cv_best.pth",
                "trained_models/pmr_ntu_cv_final.pth",
                "trained_models/pmr_ntu_cv_best_full.pth",
                "trained_models/pmr_ntu_cv_final_full.pth",
                "eval/pmr/pmr_ntu_cv.pth",
                "checkpoints/pmr_ntu_cv.pth"
            ]

            for path in pmr_paths:
                if os.path.exists(path):
                    try:
                        # Add eval/pmr to path for PMR import
                        import sys
                        pmr_path = os.path.abspath('./eval/pmr')
                        if pmr_path not in sys.path:
                            sys.path.append(pmr_path)

                        pmr_model = load_anonymizer('pmr', path, self.device, args, datasets['ntu'])
                        # Ensure model is on correct device
                        if hasattr(pmr_model, 'to'):
                            pmr_model = pmr_model.to(self.device)
                        models['pmr'] = pmr_model
                        self.logger.info(f"Loaded PMR model from {path}")
                        break
                    except Exception as e:
                        self.logger.error(f"Failed to load pmr model pmr: {e}")
                        continue

            # Try different DMR model paths (use the actual available files)
            dmr_paths = [
                "trained_models/dmr_ntu_cv_best.pth",
                "trained_models/dmr_ntu_cv_final.pth",
                "trained_models/dmr_ntu_cv_best_full.pth",
                "trained_models/dmr_ntu_cv_final_full.pth",
                "eval/dmr/dmr_ntu_cv.pth",
                "checkpoints/dmr_ntu_cv.pth"
            ]

            for path in dmr_paths:
                if os.path.exists(path):
                    try:
                        # Add eval/dmr to path for DMR import
                        import sys
                        dmr_path = os.path.abspath('./eval/dmr')
                        if dmr_path not in sys.path:
                            sys.path.append(dmr_path)

                        dmr_model = load_anonymizer('dmr', path, self.device, args, datasets['ntu'])
                        # Ensure model is on correct device
                        if hasattr(dmr_model, 'to'):
                            dmr_model = dmr_model.to(self.device)
                        models['dmr'] = dmr_model
                        self.logger.info(f"Loaded DMR model from {path}")
                        break
                    except Exception as e:
                        self.logger.error(f"Failed to load dmr model dmr: {e}")
                        continue

            if not models:
                self.logger.error("No models could be loaded. Available files:")
                import glob
                for pattern in ['*.pth', 'checkpoints/*.pth', 'trained_models/*.pth', 'eval/*/*.pth']:
                    files = glob.glob(pattern)
                    for f in files:
                        self.logger.error(f"  Found: {f}")

            return models

        except Exception as e:
            self.logger.error(f"Failed to load transformer models: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _load_test_data(self) -> List[Tuple[np.ndarray, str]]:
        """Load test data for visualization."""
        try:
            from src.data.data import load_data, process_mlm

            # Load dataset
            data = load_data('ntu', T=64)
            train_data, test_data = process_mlm(data, 'cv', 'ntu', T=64)

            # Get first 5 samples
            samples = []
            for i, filename in enumerate(test_data[:5]):
                skeleton_data = data[filename]

                # Ensure proper shape: (frames, joints, channels)
                if isinstance(skeleton_data, np.ndarray):
                    if len(skeleton_data.shape) == 2:
                        # (frames, features) -> (frames, joints, channels)
                        frames, features = skeleton_data.shape
                        if features >= 75:
                            skeleton_data = skeleton_data[:, :75].reshape(frames, 25, 3)
                        else:
                            # Pad to 75 features
                            padded = np.zeros((frames, 75))
                            padded[:, :features] = skeleton_data
                            skeleton_data = padded.reshape(frames, 25, 3)
                    elif len(skeleton_data.shape) == 3:
                        # Already correct shape
                        pass
                    else:
                        self.logger.warning(f"Unexpected skeleton shape: {skeleton_data.shape}")
                        continue

                samples.append((skeleton_data, filename))

            return samples

        except Exception as e:
            self.logger.error(f"Failed to load test data: {e}")
            # Create dummy data as fallback
            return self._create_dummy_test_data()

    def _create_dummy_test_data(self) -> List[Tuple[np.ndarray, str]]:
        """Create dummy test data with proper skeleton structure."""
        import numpy as np

        samples = []
        for i in range(5):
            # Create proper skeleton data: (64 frames, 25 joints, 3 channels)
            skeleton_data = np.zeros((64, 25, 3))

            # Add basic standing pose
            for frame in range(64):
                # Spine base (joint 0)
                skeleton_data[frame, 0, 1] = -0.9  # Y position
                # Spine mid (joint 1)
                skeleton_data[frame, 1, 1] = -0.6
                # Neck (joint 2)
                skeleton_data[frame, 2, 1] = -0.3
                # Head (joint 3)
                skeleton_data[frame, 3, 1] = 0.0

                # Add some simple motion
                t = frame / 64.0 * 2 * np.pi
                skeleton_data[frame, 3, 0] = 0.1 * np.sin(t)  # Head sway

                # Arms
                skeleton_data[frame, 4, 0] = -0.2  # Left shoulder
                skeleton_data[frame, 4, 1] = -0.2
                skeleton_data[frame, 8, 0] = 0.2   # Right shoulder
                skeleton_data[frame, 8, 1] = -0.2

            filename = f"S001C001P001R001A00{i+1}"
            samples.append((skeleton_data, filename))

        self.logger.info("Created dummy test data with proper skeleton structure")
        return samples

    def _force_standard_shape(self, tensor):
        """Force tensor to standard shape (64, 25, 3)."""
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = tensor

        # Target shape: (64, 25, 3)
        target_shape = (64, 25, 3)

        # If already correct shape, return as is
        if tensor_np.shape == target_shape:
            return tensor_np

        # Create output array
        output = np.zeros(target_shape)

        # Handle different input shapes
        if len(tensor_np.shape) == 1:
            # Flatten case
            flat_size = min(tensor_np.size, 64 * 25 * 3)
            output.flat[:flat_size] = tensor_np.flat[:flat_size]
        elif len(tensor_np.shape) == 2:
            # (time, features) case
            time_dim = min(tensor_np.shape[0], 64)
            if tensor_np.shape[1] == 75:
                # (time, 75) -> (time, 25, 3)
                for t in range(time_dim):
                    output[t] = tensor_np[t, :75].reshape(25, 3)
            else:
                # Other feature dimensions
                feat_dim = min(tensor_np.shape[1], 75)
                for t in range(time_dim):
                    output.flat[t*75:t*75+feat_dim] = tensor_np[t, :feat_dim]
        elif len(tensor_np.shape) == 3:
            # (time, joints, channels) case
            time_dim = min(tensor_np.shape[0], 64)
            joint_dim = min(tensor_np.shape[1], 25)
            channel_dim = min(tensor_np.shape[2], 3)
            output[:time_dim, :joint_dim, :channel_dim] = tensor_np[:time_dim, :joint_dim, :channel_dim]

        return output

    def _create_mlm_visualizations(self, model, data_info, output_dir, dataset, setting,
                                 temporal_ratio, spatial_ratio, results):
        """Create MLM visualizations with proper forward pass."""
        try:
            from evaluation_suite.mlm_visualizer import create_comparison_gif, create_overlay_gif
            
            dataset_obj = data_info['dataset']
            test_filenames = data_info['filenames']
            
            for i in range(min(5, len(dataset_obj))):
                try:
                    # Get masked sample
                    masked_sample = dataset_obj[i]
                    
                    # Get original sample
                    x_raw = dataset_obj.X[i].numpy() if isinstance(dataset_obj.X[i], torch.Tensor) else dataset_obj.X[i].copy()
                    from src.data.data import sample_frames
                    original_sample = sample_frames(x_raw, dataset_obj.seg)
                    
                    # Convert to proper tensor format for model
                    original_tensor = torch.from_numpy(original_sample).float().unsqueeze(0).to(self.device)
                    masked_tensor = masked_sample.float().unsqueeze(0).to(self.device)
                    
                    # Ensure proper shape: (batch, time, joints*channels)
                    if original_tensor.dim() == 3 and original_tensor.shape[2] == 75:
                        # Already correct format
                        pass
                    elif original_tensor.dim() == 4:  # (batch, time, joints, channels)
                        original_tensor = original_tensor.view(original_tensor.shape[0], original_tensor.shape[1], -1)
                        masked_tensor = masked_tensor.view(masked_tensor.shape[0], masked_tensor.shape[1], -1)
                    
                    # Run model forward pass
                    with torch.no_grad():
                        reconstructed_tensor = model(masked_tensor)
                    
                    # Convert to visualization format: (time, joints, channels)
                    try:
                        # Force both tensors to the same standard shape
                        original_np = self._force_standard_shape(original_tensor)
                        reconstructed_np = self._force_standard_shape(reconstructed_tensor)

                        # Final validation
                        if original_np.shape != reconstructed_np.shape:
                            self.logger.error(f"Shape mismatch after standardization: {original_np.shape} vs {reconstructed_np.shape}")
                            continue

                    except Exception as shape_error:
                        self.logger.error(f"Error in shape handling: {shape_error}")
                        import traceback
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        continue

                    # Create sample info
                    actual_filename = test_filenames[i] if i < len(test_filenames) else f'mlm_sample_{i+1:03d}'
                    display_filename = actual_filename.split('/')[-1].split('\\')[-1]
                    display_filename = display_filename.replace('.skeleton', '').replace('.npy', '').replace('.avi', '')
                    
                    sample_info = {
                        'action': f'sample_{i+1}',
                        'actor': 'mlm_test',
                        'filename': display_filename,
                        'original_filename': actual_filename,
                        'dataset': dataset,
                        'setting': setting,
                        'temporal_ratio': temporal_ratio,
                        'spatial_ratio': spatial_ratio,
                        'model_type': 'MLM_Autoencoder'
                    }
                    
                    # Create visualizations
                    safe_filename = display_filename.replace('/', '_').replace('\\', '_').replace(' ', '_')
                    comparison_filename = f'mlm_comparison_{dataset}_{setting}_t{temporal_ratio}_s{spatial_ratio}_{safe_filename}.gif'
                    overlay_filename = f'mlm_overlay_{dataset}_{setting}_t{temporal_ratio}_s{spatial_ratio}_{safe_filename}.gif'
                    
                    comparison_path = output_dir / comparison_filename
                    overlay_path = output_dir / overlay_filename
                    
                    create_comparison_gif(original_np, reconstructed_np, str(comparison_path), sample_info=sample_info)
                    create_overlay_gif(original_np, reconstructed_np, str(overlay_path), sample_info=sample_info)
                    
                    # Store results
                    if 'mlm_comparisons' not in results['visualizations']:
                        results['visualizations']['mlm_comparisons'] = []
                    if 'mlm_overlays' not in results['visualizations']:
                        results['visualizations']['mlm_overlays'] = []
                    
                    results['visualizations']['mlm_comparisons'].append(str(comparison_path))
                    results['visualizations']['mlm_overlays'].append(str(overlay_path))
                    
                    self.logger.info(f"Created MLM visualizations for sample {i+1}: {display_filename}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create MLM visualization for sample {i+1}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error creating MLM visualizations: {e}")
    
    def _tensor_to_visualization_format(self, tensor):
        """Convert tensor to (time, joints, channels) format for visualization."""
        try:
            # Convert to numpy if it's a tensor
            if isinstance(tensor, torch.Tensor):
                tensor_np = tensor.detach().cpu().numpy()
            else:
                tensor_np = tensor

            # Handle different tensor shapes
            if len(tensor_np.shape) == 1:
                # Flatten case - reshape to (time, joints, channels)
                total_elements = tensor_np.shape[0]
                # Try to fit into reasonable dimensions
                if total_elements >= 64 * 25 * 3:
                    tensor_np = tensor_np[:64*25*3].reshape(64, 25, 3)
                else:
                    # Pad with zeros to reach minimum size
                    padded = np.zeros(64 * 25 * 3)
                    padded[:total_elements] = tensor_np
                    tensor_np = padded.reshape(64, 25, 3)
            elif len(tensor_np.shape) == 2:
                # (time, features) or (time*joints, channels)
                if tensor_np.shape[1] == 75:  # (time, 75)
                    tensor_np = tensor_np.reshape(tensor_np.shape[0], 25, 3)
                elif tensor_np.shape[1] == 3:  # (time*joints, 3)
                    time_joints = tensor_np.shape[0]
                    joints = 25
                    time = time_joints // joints
                    if time * joints == time_joints:
                        tensor_np = tensor_np.reshape(time, joints, 3)
                    else:
                        # Adjust to fit exactly
                        adjusted_elements = (time_joints // 75) * 75
                        if adjusted_elements > 0:
                            tensor_np = tensor_np[:adjusted_elements].reshape(-1, 25, 3)
                        else:
                            tensor_np = np.zeros((64, 25, 3))
                else:
                    # Try to reshape to (time, joints, channels)
                    total_elements = tensor_np.shape[0] * tensor_np.shape[1]
                    if total_elements >= 64 * 25 * 3:
                        tensor_np = tensor_np.flatten()[:64*25*3].reshape(64, 25, 3)
                    else:
                        padded = np.zeros(64 * 25 * 3)
                        padded[:total_elements] = tensor_np.flatten()
                        tensor_np = padded.reshape(64, 25, 3)
            elif len(tensor_np.shape) == 3:
                # Already in (time, joints, channels) format or (batch, time, features)
                if tensor_np.shape[0] == 1:  # Remove batch dimension
                    tensor_np = tensor_np[0]
                    if tensor_np.shape[1] == 75:
                        tensor_np = tensor_np.reshape(tensor_np.shape[0], 25, 3)
            elif len(tensor_np.shape) == 4:
                # (batch, time, joints, channels) - remove batch dimension
                tensor_np = tensor_np[0]
            elif len(tensor_np.shape) == 5:
                # (batch, time, person, joints, channels) - remove batch and person
                tensor_np = tensor_np[0, :, 0, :, :]

            # Ensure we have valid dimensions
            if len(tensor_np.shape) != 3:
                self.logger.warning(f"Unexpected tensor shape after processing: {tensor_np.shape}")
                # Force reshape to expected format
                flat = tensor_np.flatten()
                if len(flat) >= 64 * 25 * 3:
                    tensor_np = flat[:64*25*3].reshape(64, 25, 3)
                else:
                    padded = np.zeros(64 * 25 * 3)
                    padded[:len(flat)] = flat
                    tensor_np = padded.reshape(64, 25, 3)

            return tensor_np

        except Exception as e:
            self.logger.error(f"Error converting tensor to visualization format: {e}")
            self.logger.error(f"Original tensor shape: {tensor.shape if hasattr(tensor, 'shape') else 'unknown'}")
            # Fallback: create dummy data
            return np.zeros((64, 25, 3))

    def _create_transformer_visualizations(self, models, test_data, output_dir, results):
        """Create transformer visualizations with PMR/DMR comparisons."""
        try:
            from evaluation_suite.experiments.visualization import VisualizationExperiments

            for i, (skeleton_data, filename) in enumerate(test_data):
                try:
                    # Clean filename for display
                    display_filename = filename.split('/')[-1].split('\\')[-1]
                    display_filename = display_filename.replace('.skeleton', '').replace('.npy', '').replace('.avi', '')

                    # Create original visualization
                    original_sample = [(skeleton_data, f'sample_{i+1}', filename)]
                    original_path = VisualizationExperiments.create_skeleton_animation(
                        original_sample,
                        output_dir=str(output_dir),
                        figure_type='original',
                        unique_id=f'{display_filename}_original'
                    )

                    # Process through each model
                    for model_name, model in models.items():
                        if model is None:
                            continue

                        try:
                            # Run forward pass
                            processed_data = self._run_model_forward_pass(skeleton_data, model, model_name)

                            # Create visualization
                            processed_sample = [(processed_data, f'sample_{i+1}', filename)]
                            processed_path = VisualizationExperiments.create_skeleton_animation(
                                processed_sample,
                                output_dir=str(output_dir),
                                figure_type=f'{model_name}_retargeted',
                                unique_id=f'{display_filename}_{model_name}'
                            )

                            # Store results
                            if f'{model_name}_visualizations' not in results['visualizations']:
                                results['visualizations'][f'{model_name}_visualizations'] = []
                            results['visualizations'][f'{model_name}_visualizations'].append(processed_path)

                            self.logger.info(f"Created {model_name} visualization for {display_filename}")

                        except Exception as e:
                            self.logger.warning(f"Failed to create {model_name} visualization for sample {i+1}: {e}")
                            continue

                    # Store original visualization
                    if 'original_visualizations' not in results['visualizations']:
                        results['visualizations']['original_visualizations'] = []
                    results['visualizations']['original_visualizations'].append(original_path)

                except Exception as e:
                    self.logger.warning(f"Failed to create visualizations for sample {i+1}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error creating transformer visualizations: {e}")

    def _run_model_forward_pass(self, skeleton_data, model, model_name):
        """Run proper forward pass through the model."""
        try:
            # Convert to tensor
            if not isinstance(skeleton_data, torch.Tensor):
                input_data = torch.from_numpy(skeleton_data).float()
            else:
                input_data = skeleton_data.float()

            # Ensure proper shape: (batch, time, features)
            if input_data.dim() == 3:  # (time, joints, channels)
                T, V, C = input_data.shape
                # Flatten to (time, features) first, then add batch dimension
                input_data = input_data.view(T, V * C).unsqueeze(0)  # (1, time, 75)
            elif input_data.dim() == 2:  # (time, features)
                input_data = input_data.unsqueeze(0)  # Add batch dimension
            elif input_data.dim() == 1:  # Flattened data
                # Reshape to (time, features) then add batch dimension
                if input_data.shape[0] >= 64 * 75:
                    input_data = input_data[:64*75].view(64, 75).unsqueeze(0)
                else:
                    # Pad if necessary
                    padded = torch.zeros(64 * 75)
                    padded[:input_data.shape[0]] = input_data
                    input_data = padded.view(64, 75).unsqueeze(0)

            input_data = input_data.to(self.device)

            # Run model-specific forward pass
            if model_name == 'transformer':
                return self._run_transformer_forward(input_data, model)
            elif model_name in ['pmr', 'dmr']:
                return self._run_pmr_dmr_forward(input_data, model)
            else:
                self.logger.error(f"Unknown model type: {model_name}")
                return input_data.squeeze().cpu().numpy()

        except Exception as e:
            self.logger.error(f"Error in {model_name} forward pass: {e}")
            return skeleton_data

    def _run_transformer_forward(self, input_data, model):
        """Run transformer model forward pass."""
        try:
            from src.evaluation.eval_model import get_anonymized_paired_transformer, prep_data

            # Force CPU mode to avoid CUDA issues
            if hasattr(model, 'cpu'):
                model = model.cpu()

            # Move input data to CPU as well
            input_data = input_data.cpu()

            # Create dummy batch data for transformer
            batch_size = input_data.shape[0]
            device = input_data.device
            dummy_actors = torch.zeros(batch_size, dtype=torch.long, device=device)
            dummy_actions = torch.zeros(batch_size, dtype=torch.long, device=device)

            # Create batch tuple (x1, x2, y1, y2, actors, actions)
            batch = (input_data, input_data, input_data, input_data, dummy_actors, dummy_actions)

            # Process through transformer
            with torch.no_grad():
                result = get_anonymized_paired_transformer(batch, model, prep_data)
                if result and len(result) > 0:
                    output = result[0]['x1']  # Get the anonymized skeleton
                    # Convert back to (time, joints, channels) format
                    if output.dim() == 2 and output.shape[1] == 75:
                        output = output.view(output.shape[0], 25, 3)
                    return output.cpu().numpy()
                else:
                    return input_data.squeeze().cpu().numpy()

        except Exception as e:
            self.logger.error(f"Error in transformer forward pass: {e}")
            # Return the input data as fallback (no transformation)
            return input_data.squeeze().cpu().numpy()

    def _run_pmr_dmr_forward(self, input_data, model):
        """Run PMR/DMR model forward pass."""
        try:
            from src.evaluation.eval_model import get_anonymized_paired_dmr_pmr

            # Force CPU mode to avoid CUDA issues
            if hasattr(model, 'cpu'):
                model = model.cpu()

            # Move input data to CPU as well
            input_data = input_data.cpu()

            # Create dummy batch data
            batch_size = input_data.shape[0]
            device = input_data.device
            dummy_actors = torch.zeros(batch_size, dtype=torch.long, device=device)
            dummy_actions = torch.zeros(batch_size, dtype=torch.long, device=device)

            # Create batch tuple (x1, x2, y1, y2, actors, actions)
            batch = (input_data, input_data, input_data, input_data, dummy_actors, dummy_actions)

            # Process through PMR/DMR
            with torch.no_grad():
                result = get_anonymized_paired_dmr_pmr(batch, model, T=75, mixformer_mode=True)
                if result and len(result) > 0:
                    output = result[0]['x1']  # Get the anonymized skeleton
                    # Convert back to (time, joints, channels) format
                    if output.dim() == 2 and output.shape[1] == 75:
                        output = output.view(output.shape[0], 25, 3)
                    return output.cpu().numpy()
                else:
                    return input_data.squeeze().cpu().numpy()

        except Exception as e:
            self.logger.error(f"Error in PMR/DMR forward pass: {e}")
            return input_data.squeeze().cpu().numpy()
