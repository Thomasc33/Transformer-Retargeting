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
            # Construct data file path
            if dataset == 'ntu':
                if setting == 'cv':
                    data_file = 'data/ntu_cv_paired_10000_2000.pt'
                else:  # cs
                    data_file = 'data/ntu_cs_paired_10000_2000.pt'
            elif dataset == 'ntu120':
                if setting == 'cv':
                    data_file = 'data/ntu120_cv_paired_10000_2000.pt'
                else:  # cs
                    data_file = 'data/ntu120_cs_paired_10000_2000.pt'
            elif dataset == 'etri':
                data_file = 'data/etri_paired_data.pt'
            else:
                self.logger.error(f"Unknown dataset: {dataset}")
                return None

            if not os.path.exists(data_file):
                self.logger.error(f"Data file does not exist: {data_file}")
                return None

            # Load the data
            data = torch.load(data_file)

            # Create data loader
            if isinstance(data, dict) and 'test' in data:
                test_data = data['test']
            else:
                test_data = data

            # Limit test samples if specified
            if test_samples is not None and len(test_data) > test_samples:
                test_data = test_data[:test_samples]

            data_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )

            return data_loader

        except Exception as e:
            self.logger.error(f"Error loading paired data: {str(e)}")
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
