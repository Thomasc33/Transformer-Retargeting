"""
Data management utilities for loading and managing datasets.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Import existing data loading utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from data import load_data
except ImportError:
    load_data = None

try:
    from eval.eval_loader import Dataloaders
except ImportError:
    Dataloaders = None


class CrossDataWrapper:
    """Wrapper to convert Cross_Data output to simple (skeleton, label) format for visualization."""

    def __init__(self, cross_data):
        self.cross_data = cross_data

    def __len__(self):
        return len(self.cross_data)

    def __getitem__(self, index):
        try:
            # Get the Cross_Data output: (x1, x2, y1, y2, actors, actions)
            x1, x2, y1, y2, actors, actions = self.cross_data[index]

            # For visualization, we'll use x1 (first skeleton) and the first action
            skeleton = x1  # Shape should be (frames, features)
            action_label = int(actions[0]) if hasattr(actions, '__getitem__') else int(actions)

            return skeleton, action_label

        except Exception as e:
            # If there's an error, create dummy data
            import torch
            skeleton = torch.zeros(64, 75)  # 64 frames, 75 features
            action_label = 0
            return skeleton, action_label


class DataManager:
    """
    Manages loading and caching of different datasets used in evaluation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_cache = {}

    def load_data(self, data_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load all datasets specified in the configuration.

        Args:
            data_configs: Dictionary of data configurations

        Returns:
            Dictionary of loaded data loaders
        """
        data_loaders = {}

        for data_name, config in data_configs.items():
            self.logger.info(f"Loading dataset: {data_name}")

            try:
                data_loader = self.load_single_dataset(data_name, config)
                if data_loader is not None:
                    data_loaders[data_name] = data_loader
                    self.logger.info(f"Successfully loaded {data_name}")
                else:
                    self.logger.warning(f"Failed to load {data_name}")

            except Exception as e:
                self.logger.error(f"Error loading {data_name}: {str(e)}")

        return data_loaders

    def load_single_dataset(self, data_name: str, config: Dict[str, Any]) -> Optional[Any]:
        """
        Load a single dataset based on its configuration.

        Args:
            data_name: Name of the dataset
            config: Dataset configuration

        Returns:
            Loaded data loader or None if failed
        """
        dataset = config.get('dataset', 'ntu')
        setting = config.get('setting', 'cv')
        batch_size = config.get('batch_size', 32)
        test_samples = config.get('test_samples', None)

        # Check cache first
        cache_key = f"{data_name}_{dataset}_{setting}_{batch_size}_{test_samples}"
        if cache_key in self.data_cache:
            self.logger.info(f"Using cached dataset: {data_name}")
            return self.data_cache[cache_key]

        data_loader = None

        try:
            if config.get('type') == 'paired':
                data_loader = self.load_paired_data(dataset, setting, batch_size, test_samples)
            elif config.get('type') == 'cross':
                data_loader = self.load_cross_data(dataset, setting, batch_size)
            elif config.get('type') == 'eval':
                data_loader = self.load_eval_data(dataset, setting, config)
            else:
                # Default to paired data
                data_loader = self.load_paired_data(dataset, setting, batch_size, test_samples)

            # Cache the data loader
            if data_loader is not None:
                self.data_cache[cache_key] = data_loader

        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset}: {str(e)}")

        return data_loader

    def load_paired_data(self, dataset: str, setting: str, batch_size: int,
                        test_samples: Optional[int] = None) -> Optional[Any]:
        """Load paired data for evaluation."""
        try:
            # Construct data file path - try multiple options
            possible_files = []

            if dataset == 'ntu':
                if setting == 'cv':
                    possible_files = [
                        'data/ntu_cv_paired_10000_2000.pt',
                        'data/ntu_cv_paired_comprehensive.pt',
                        'data/ntu/pretraining_data_cv_comprehensive.pt',
                        'data/ntu/pretraining_data.pt'
                    ]
                else:  # cs
                    possible_files = [
                        'data/ntu_cs_paired_10000_2000.pt',
                        'data/ntu_cs_paired_comprehensive.pt',
                        'data/ntu/pretraining_data.pt'
                    ]
            elif dataset == 'ntu120':
                if setting == 'cv':
                    possible_files = [
                        'data/ntu120_cv_paired_10000_2000.pt',
                        'data/ntu120_cv_paired_comprehensive.pt'
                    ]
                else:  # cs
                    possible_files = [
                        'data/ntu120_cs_paired_10000_2000.pt',
                        'data/ntu120_cs_paired_comprehensive.pt'
                    ]
            elif dataset == 'etri':
                possible_files = ['data/etri_paired_data.pt', 'data/etri/pretraining_data.pt']
            else:
                self.logger.error(f"Unknown dataset: {dataset}")
                return None

            # Find the first existing file
            data_file = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    data_file = file_path
                    self.logger.info(f"Found data file: {data_file}")
                    break

            if data_file is None:
                self.logger.info(f"No data files found, using dummy data for visualization")
                return self._create_dummy_data_loader(batch_size, test_samples)

            # Try to load and use real data
            try:
                self.logger.info(f"Loading data from: {data_file}")
                data = torch.load(data_file, map_location='cpu')

                # Handle different data formats
                if isinstance(data, dict) and 'test' in data:
                    test_data = data['test']
                elif isinstance(data, list):
                    test_data = data
                else:
                    test_data = data

                # Check if this is Cross_Data (problematic format)
                if hasattr(test_data, '__class__') and 'Cross_Data' in str(test_data.__class__):
                    self.logger.info("Detected Cross_Data format, converting to simple format...")
                    # Try to convert Cross_Data to simple format
                    simple_data = self._convert_cross_data_to_simple(test_data, test_samples)
                    if simple_data:
                        test_data = simple_data
                    else:
                        self.logger.warning("Cross_Data conversion failed, using dummy data")
                        return self._create_dummy_data_loader(batch_size, test_samples)

                # Limit test samples if specified
                if test_samples is not None and len(test_data) > test_samples:
                    test_data = test_data[:test_samples]

                data_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                )

                self.logger.info(f"Successfully loaded real data with {len(test_data)} samples")
                return data_loader

            except Exception as e:
                self.logger.warning(f"Failed to load real data ({str(e)}), using dummy data")
                return self._create_dummy_data_loader(batch_size, test_samples)

        except Exception as e:
            self.logger.warning(f"Could not load real data ({str(e)}), using dummy data for visualization")
            # Create dummy data for visualization testing
            return self._create_dummy_data_loader(batch_size, test_samples)

    def _create_dummy_data_loader(self, batch_size: int, test_samples: Optional[int] = None) -> Optional[Any]:
        """Create dummy data loader for testing when real data is not available."""
        try:
            import torch

            self.logger.warning("Creating dummy data for visualization testing")

            # Create dummy skeleton data
            num_samples = test_samples if test_samples is not None else 10
            num_frames = 64
            num_features = 75  # 25 joints * 3 coordinates

            # Generate realistic skeleton data with proper NTU joint structure
            dummy_skeletons = []
            for i in range(num_samples):
                skeleton = self._create_realistic_skeleton_motion(num_frames, num_features, i)
                # Create a tuple with skeleton and dummy label
                dummy_skeletons.append((skeleton, i % 10))  # Dummy action labels 0-9

            # Create data loader
            data_loader = torch.utils.data.DataLoader(
                dummy_skeletons,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0  # Use 0 for dummy data to avoid multiprocessing issues
            )

            self.logger.info(f"Created dummy data loader with {num_samples} samples")
            return data_loader

        except Exception as e:
            self.logger.error(f"Error creating dummy data loader: {str(e)}")
            return None

    def _create_realistic_skeleton_motion(self, num_frames: int, num_features: int, motion_type: int):
        """Create realistic skeleton motion patterns for visualization."""
        import torch
        import math

        skeleton = torch.zeros(num_frames, num_features)

        # Define basic skeleton structure (NTU 25 joints)
        # Joint indices for key body parts
        spine_base = 0 * 3  # Joint 0: spine base
        spine_mid = 1 * 3   # Joint 1: spine mid
        neck = 2 * 3        # Joint 2: neck
        head = 3 * 3        # Joint 3: head

        left_shoulder = 4 * 3   # Joint 4: left shoulder
        left_elbow = 5 * 3      # Joint 5: left elbow
        left_wrist = 6 * 3      # Joint 6: left wrist
        left_hand = 7 * 3       # Joint 7: left hand

        right_shoulder = 8 * 3  # Joint 8: right shoulder
        right_elbow = 9 * 3     # Joint 9: right elbow
        right_wrist = 10 * 3    # Joint 10: right wrist
        right_hand = 11 * 3     # Joint 11: right hand

        left_hip = 12 * 3       # Joint 12: left hip
        left_knee = 13 * 3      # Joint 13: left knee
        left_ankle = 14 * 3     # Joint 14: left ankle
        left_foot = 15 * 3      # Joint 15: left foot

        right_hip = 16 * 3      # Joint 16: right hip
        right_knee = 17 * 3     # Joint 17: right knee
        right_ankle = 18 * 3    # Joint 18: right ankle
        right_foot = 19 * 3     # Joint 19: right foot

        # Create different motion patterns based on motion_type
        for frame in range(num_frames):
            t = frame / num_frames * 4 * math.pi  # Two full cycles

            # Base standing position
            skeleton[frame, spine_base + 1] = -0.9      # Y position (standing)
            skeleton[frame, spine_mid + 1] = -0.6       # Spine mid Y
            skeleton[frame, neck + 1] = -0.3            # Neck Y
            skeleton[frame, head + 1] = 0.0             # Head Y

            if motion_type % 4 == 0:  # Walking motion
                # Walking pattern
                step_phase = math.sin(t)

                # Hip movement (alternating)
                skeleton[frame, left_hip] = 0.1 * step_phase
                skeleton[frame, right_hip] = -0.1 * step_phase
                skeleton[frame, left_hip + 1] = -0.9
                skeleton[frame, right_hip + 1] = -0.9

                # Knee movement (bending during walk)
                skeleton[frame, left_knee] = 0.1 * step_phase
                skeleton[frame, left_knee + 1] = -0.6 + 0.1 * abs(step_phase)
                skeleton[frame, right_knee] = -0.1 * step_phase
                skeleton[frame, right_knee + 1] = -0.6 + 0.1 * abs(-step_phase)

                # Ankle movement
                skeleton[frame, left_ankle] = 0.1 * step_phase
                skeleton[frame, left_ankle + 1] = -0.3
                skeleton[frame, right_ankle] = -0.1 * step_phase
                skeleton[frame, right_ankle + 1] = -0.3

                # Arm swinging (opposite to legs)
                skeleton[frame, left_shoulder] = -0.2
                skeleton[frame, left_shoulder + 1] = -0.2
                skeleton[frame, left_elbow] = -0.2 + 0.1 * (-step_phase)
                skeleton[frame, left_elbow + 1] = -0.4

                skeleton[frame, right_shoulder] = 0.2
                skeleton[frame, right_shoulder + 1] = -0.2
                skeleton[frame, right_elbow] = 0.2 + 0.1 * step_phase
                skeleton[frame, right_elbow + 1] = -0.4

            elif motion_type % 4 == 1:  # Waving motion
                # Standing with arm waving
                wave = math.sin(t * 2)

                # Static lower body
                skeleton[frame, left_hip] = -0.1
                skeleton[frame, right_hip] = 0.1
                skeleton[frame, left_hip + 1] = skeleton[frame, right_hip + 1] = -0.9
                skeleton[frame, left_knee + 1] = skeleton[frame, right_knee + 1] = -0.6
                skeleton[frame, left_ankle + 1] = skeleton[frame, right_ankle + 1] = -0.3

                # Waving right arm
                skeleton[frame, right_shoulder] = 0.2
                skeleton[frame, right_shoulder + 1] = -0.2
                skeleton[frame, right_elbow] = 0.2 + 0.3 * wave
                skeleton[frame, right_elbow + 1] = -0.1 + 0.2 * abs(wave)
                skeleton[frame, right_wrist] = 0.2 + 0.4 * wave
                skeleton[frame, right_wrist + 1] = 0.1 + 0.3 * abs(wave)

                # Static left arm
                skeleton[frame, left_shoulder] = -0.2
                skeleton[frame, left_shoulder + 1] = -0.2
                skeleton[frame, left_elbow] = -0.2
                skeleton[frame, left_elbow + 1] = -0.4

            elif motion_type % 4 == 2:  # Jumping motion
                # Jumping pattern
                jump_phase = abs(math.sin(t))

                # Vertical movement for whole body
                y_offset = 0.2 * jump_phase
                skeleton[frame, spine_base + 1] = -0.9 + y_offset
                skeleton[frame, spine_mid + 1] = -0.6 + y_offset
                skeleton[frame, neck + 1] = -0.3 + y_offset
                skeleton[frame, head + 1] = 0.0 + y_offset

                # Legs bending during jump
                knee_bend = 0.2 * (1 - jump_phase)
                skeleton[frame, left_hip + 1] = -0.9 + y_offset
                skeleton[frame, right_hip + 1] = -0.9 + y_offset
                skeleton[frame, left_knee + 1] = -0.6 + y_offset - knee_bend
                skeleton[frame, right_knee + 1] = -0.6 + y_offset - knee_bend
                skeleton[frame, left_ankle + 1] = -0.3 + y_offset
                skeleton[frame, right_ankle + 1] = -0.3 + y_offset

                # Arms moving up during jump
                arm_raise = 0.3 * jump_phase
                skeleton[frame, left_shoulder] = -0.2
                skeleton[frame, left_shoulder + 1] = -0.2 + arm_raise
                skeleton[frame, right_shoulder] = 0.2
                skeleton[frame, right_shoulder + 1] = -0.2 + arm_raise
                skeleton[frame, left_elbow + 1] = -0.4 + arm_raise
                skeleton[frame, right_elbow + 1] = -0.4 + arm_raise

            else:  # Sitting motion
                # Sitting position
                sit_amount = 0.3
                skeleton[frame, spine_base + 1] = -0.9 + sit_amount
                skeleton[frame, spine_mid + 1] = -0.6 + sit_amount
                skeleton[frame, neck + 1] = -0.3 + sit_amount
                skeleton[frame, head + 1] = 0.0 + sit_amount

                # Bent legs for sitting
                skeleton[frame, left_hip + 1] = -0.9 + sit_amount
                skeleton[frame, right_hip + 1] = -0.9 + sit_amount
                skeleton[frame, left_knee + 1] = -0.3 + sit_amount
                skeleton[frame, right_knee + 1] = -0.3 + sit_amount
                skeleton[frame, left_ankle + 1] = -0.3 + sit_amount
                skeleton[frame, right_ankle + 1] = -0.3 + sit_amount

                # Relaxed arms
                skeleton[frame, left_shoulder] = -0.2
                skeleton[frame, left_shoulder + 1] = -0.2 + sit_amount
                skeleton[frame, right_shoulder] = 0.2
                skeleton[frame, right_shoulder + 1] = -0.2 + sit_amount
                skeleton[frame, left_elbow + 1] = -0.5 + sit_amount
                skeleton[frame, right_elbow + 1] = -0.5 + sit_amount

        return skeleton

    def _convert_cross_data_to_simple(self, cross_data, max_samples=None):
        """Convert Cross_Data to simple (skeleton, label) format safely."""
        try:
            simple_data = []
            max_samples = max_samples or min(len(cross_data), 10)  # Limit to avoid hanging

            self.logger.info(f"Converting Cross_Data to simple format (max {max_samples} samples)...")

            for i in range(min(max_samples, len(cross_data))):
                try:
                    # Try to get one sample from Cross_Data
                    sample = cross_data[i]

                    # Cross_Data returns (x1, x2, y1, y2, actors, actions)
                    if isinstance(sample, (list, tuple)) and len(sample) >= 6:
                        x1, x2, y1, y2, actors, actions = sample[:6]

                        # Use x1 as the skeleton and first action as label
                        skeleton = x1
                        action_label = int(actions[0]) if hasattr(actions, '__getitem__') else int(actions)

                        # Try to get filename information from Cross_Data
                        filename = None
                        if hasattr(cross_data, 'sampled_data') and i < len(cross_data.sampled_data):
                            sample_info = cross_data.sampled_data[i]
                            if isinstance(sample_info, (list, tuple)) and len(sample_info) >= 1:
                                # Each sample_info is a 4-tuple: [(p1,a1,fname1), (p1,a2,fname2), (p2,a1,fname3), (p2,a2,fname4)]
                                # Get the first filename
                                first_tuple = sample_info[0]
                                if isinstance(first_tuple, (list, tuple)) and len(first_tuple) >= 3:
                                    filename = first_tuple[2]

                        # Create sample with filename if available
                        if filename:
                            simple_data.append((skeleton, action_label, filename))
                        else:
                            simple_data.append((skeleton, action_label))

                    else:
                        # Unexpected format, skip this sample
                        continue

                except Exception as e:
                    # If any sample fails, skip it
                    self.logger.debug(f"Skipping sample {i}: {e}")
                    continue

            if len(simple_data) > 0:
                self.logger.info(f"Successfully converted {len(simple_data)} samples from Cross_Data")
                return simple_data
            else:
                self.logger.warning("No samples could be converted from Cross_Data")
                return None

        except Exception as e:
            self.logger.warning(f"Cross_Data conversion failed: {e}")
            return None

    def load_cross_data(self, dataset: str, setting: str, batch_size: int) -> Optional[Any]:
        """Load cross-validation data."""
        try:
            if 'get_cross_data' in globals():
                data_loader = get_cross_data(
                    dataset=dataset,
                    setting=setting,
                    batch_size=batch_size
                )
                return data_loader
            else:
                self.logger.warning("get_cross_data function not available")
                return None

        except Exception as e:
            self.logger.error(f"Error loading cross data: {str(e)}")
            return None

    def load_eval_data(self, dataset: str, setting: str, config: Dict[str, Any]) -> Optional[Any]:
        """Load evaluation data for specific tasks."""
        try:
            task = config.get('task', 'ar')
            batch_size = config.get('batch_size', 32)

            if 'get_eval_data' in globals():
                data_loader = get_eval_data(
                    dataset=dataset,
                    setting=setting,
                    task=task,
                    batch_size=batch_size
                )
                return data_loader
            else:
                self.logger.warning("get_eval_data function not available")
                return None

        except Exception as e:
            self.logger.error(f"Error loading eval data: {str(e)}")
            return None

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a loaded dataset."""
        info = {'name': dataset_name}

        for cache_key, data_loader in self.data_cache.items():
            if dataset_name in cache_key:
                if hasattr(data_loader, 'dataset'):
                    info['num_samples'] = len(data_loader.dataset)
                    info['batch_size'] = data_loader.batch_size
                    info['num_batches'] = len(data_loader)
                break

        return info

    def get_data_statistics(self, data_loader: Any) -> Dict[str, Any]:
        """Get statistics about a data loader."""
        stats = {}

        try:
            if hasattr(data_loader, 'dataset'):
                stats['num_samples'] = len(data_loader.dataset)
                stats['batch_size'] = data_loader.batch_size
                stats['num_batches'] = len(data_loader)

                # Get sample statistics
                sample_batch = next(iter(data_loader))
                if isinstance(sample_batch, (list, tuple)):
                    sample = sample_batch[0]
                else:
                    sample = sample_batch

                if isinstance(sample, torch.Tensor):
                    stats['sample_shape'] = list(sample.shape)
                    stats['sample_dtype'] = str(sample.dtype)
                elif isinstance(sample, dict):
                    stats['sample_keys'] = list(sample.keys())
                    if 'skeleton' in sample:
                        skel = sample['skeleton']
                        if isinstance(skel, torch.Tensor):
                            stats['skeleton_shape'] = list(skel.shape)

        except Exception as e:
            self.logger.warning(f"Could not get data statistics: {str(e)}")

        return stats

    def validate_data_config(self, config: Dict[str, Any]) -> bool:
        """Validate data configuration."""
        required_fields = ['dataset', 'setting']

        for field in required_fields:
            if field not in config:
                self.logger.error(f"Missing required field in data config: {field}")
                return False

        # Validate dataset
        valid_datasets = ['ntu', 'ntu120', 'etri']
        if config['dataset'] not in valid_datasets:
            self.logger.error(f"Invalid dataset: {config['dataset']}. Must be one of {valid_datasets}")
            return False

        # Validate setting
        valid_settings = ['cv', 'cs']
        if config['setting'] not in valid_settings:
            self.logger.error(f"Invalid setting: {config['setting']}. Must be one of {valid_settings}")
            return False

        return True

    def clear_cache(self):
        """Clear the data cache to free memory."""
        self.data_cache.clear()
        self.logger.info("Data cache cleared")

    def preload_common_datasets(self):
        """Preload commonly used datasets."""
        common_configs = [
            {'dataset': 'ntu', 'setting': 'cv', 'batch_size': 32, 'type': 'paired'},
            {'dataset': 'ntu', 'setting': 'cs', 'batch_size': 32, 'type': 'paired'},
            {'dataset': 'ntu120', 'setting': 'cv', 'batch_size': 32, 'type': 'paired'},
        ]

        for i, config in enumerate(common_configs):
            data_name = f"common_{i}"
            self.logger.info(f"Preloading common dataset: {data_name}")
            self.load_single_dataset(data_name, config)
