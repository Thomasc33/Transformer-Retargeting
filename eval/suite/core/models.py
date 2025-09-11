"""
Model management utilities for loading and managing different models.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import existing model loading utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import from your existing eval_model.py
try:
    # Try different possible paths for eval_model.py
    import sys
    import os

    # Add src directory to path
    src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)

    # Add root directory to path
    root_path = os.path.join(os.path.dirname(__file__), '..', '..')
    if os.path.exists(root_path):
        sys.path.insert(0, root_path)

    # Try multiple import paths for eval_model functions
    eval_model_imported = False

    # Try src.evaluation.eval_model first
    try:
        src_eval_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'evaluation')
        if src_eval_path not in sys.path:
            sys.path.insert(0, src_eval_path)

        # Also add the root directory for data import
        root_path = os.path.join(os.path.dirname(__file__), '..', '..')
        if root_path not in sys.path:
            sys.path.insert(0, root_path)

        # Add src/utils for safe_model_loading
        utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'utils')
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)

        # Add eval directory for preprocess module
        eval_path = os.path.join(os.path.dirname(__file__), '..', '..', 'eval')
        if eval_path not in sys.path:
            sys.path.insert(0, eval_path)

        from eval_model import load_anonymizer
        from data import datasets
        logging.info("Successfully imported from src/evaluation/eval_model.py")
        eval_model_imported = True
    except ImportError as e:
        logging.debug(f"Failed to import from src/evaluation: {e}")
        pass

    # Try direct import from src
    if not eval_model_imported:
        try:
            from src.evaluation.eval_model import load_anonymizer
            from data import datasets
            logging.info("Successfully imported from src.evaluation.eval_model")
            eval_model_imported = True
        except ImportError:
            pass

    # Try evaluation.eval_model
    if not eval_model_imported:
        try:
            from evaluation.eval_model import load_anonymizer
            from data import datasets
            logging.info("Successfully imported from evaluation.eval_model")
            eval_model_imported = True
        except ImportError:
            pass

    # Try direct eval_model import
    if not eval_model_imported:
        try:
            from eval_model import load_anonymizer
            from data import datasets
            logging.info("Successfully imported from eval_model")
            eval_model_imported = True
        except ImportError:
            pass

    if not eval_model_imported:
        # Fallback - create dummy functions for visualization
        def dummy_load_anonymizer(model_type, model_path, device, args, ds=None):
            """Dummy anonymizer for visualization when real models aren't available."""
            if model_type == 'raw':
                return None  # Raw data, no model needed
            else:
                # Return a dummy model that just returns the input
                class DummyModel:
                    def __call__(self, x):
                        return x
                    def __repr__(self):
                        return f"DummyModel({model_type})"
                return DummyModel()

        load_anonymizer = dummy_load_anonymizer
        safe_load_model = None
        datasets = {'ntu': {'name': 'ntu'}}  # Provide basic dataset info
        logging.info("Using dummy models for visualization (real models not available)")

except ImportError as e:
    logging.warning(f"Could not import from eval_model.py: {e}")

# Ensure we always have fallback functions available
if 'load_anonymizer' not in locals() or load_anonymizer is None:
    def dummy_load_anonymizer(model_type, model_path, device, args, ds=None):
        """Dummy anonymizer for visualization when real models aren't available."""
        if model_type == 'raw':
            return None  # Raw data, no model needed
        else:
            # Return a dummy model that just returns the input
            class DummyModel:
                def __call__(self, x):
                    return x
                def __repr__(self):
                    return f"DummyModel({model_type})"
            return DummyModel()

    load_anonymizer = dummy_load_anonymizer
    safe_load_model = None
    datasets = {'ntu': {'name': 'ntu'}}  # Provide basic dataset info
    logging.info("Using fallback dummy models")

try:
    from src.model.autoencoder import Model as SkeletonAutoEncoder
except ImportError:
    SkeletonAutoEncoder = None

try:
    from src.model.sgn import SGN
except ImportError:
    SGN = None

try:
    from src.model.ske_mixf import SkeletonMixFormer
except ImportError:
    SkeletonMixFormer = None


class ModelManager:
    """
    Manages loading and caching of different models used in evaluation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_models(self, model_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load all models specified in the configuration.

        Args:
            model_configs: Dictionary of model configurations

        Returns:
            Dictionary of loaded models
        """
        models = {}

        for model_name, config in model_configs.items():
            self.logger.info(f"Loading model: {model_name}")

            try:
                model = self.load_single_model(model_name, config)
                if model is not None:
                    models[model_name] = model
                    self.logger.info(f"Successfully loaded {model_name}")
                else:
                    self.logger.warning(f"Failed to load {model_name}")

            except Exception as e:
                self.logger.error(f"Error loading {model_name}: {str(e)}")

        return models

    def load_single_model(self, model_name: str, config: Dict[str, Any]) -> Optional[Any]:
        """
        Load a single model based on its configuration.

        Args:
            model_name: Name of the model
            config: Model configuration

        Returns:
            Loaded model or None if failed
        """
        model_type = config.get('type', 'unknown')
        model_path = config.get('path', '')

        # Check cache first
        cache_key = f"{model_name}_{model_path}"
        if cache_key in self.model_cache:
            self.logger.info(f"Using cached model: {model_name}")
            return self.model_cache[cache_key]

        model = None

        try:
            if model_type == 'transformer':
                model = self.load_transformer_model(model_path, config)
            elif model_type == 'sgn':
                model = self.load_sgn_model(model_path, config)
            elif model_type == 'mixformer':
                model = self.load_mixformer_model(model_path, config)
            elif model_type == 'dmr':
                model = self.load_dmr_model(model_path, config)
            elif model_type == 'pmr':
                model = self.load_pmr_model(model_path, config)
            elif model_type == 'raw':
                model = 'raw'  # Special case for raw data
            else:
                self.logger.warning(f"Unknown model type: {model_type}")

            # Cache the model
            if model is not None:
                self.model_cache[cache_key] = model

        except Exception as e:
            self.logger.error(f"Error loading {model_type} model: {str(e)}")

        return model

    def load_transformer_model(self, model_path: str, config: Dict[str, Any]) -> Optional[Any]:
        """Load transformer (autoencoder) model using existing eval_model.py functions.
        Respects EVAL_DEFAULT_MODEL env var and repo default at data/models_output/model.pth.
        """
        # Resolve default path override
        env_default = os.environ.get('EVAL_DEFAULT_MODEL')
        repo_default = os.path.join(Path(__file__).resolve().parents[3], 'data', 'models_output', 'model.pth')
        resolved_path = model_path
        if not resolved_path or resolved_path == 'model.pth' or not os.path.exists(resolved_path):
            if env_default and os.path.exists(env_default):
                resolved_path = env_default
            elif os.path.exists(repo_default):
                resolved_path = repo_default

        # For dummy models, we don't need the file to exist
        using_dummy = hasattr(load_anonymizer, '__name__') and load_anonymizer.__name__ == 'dummy_load_anonymizer'
        if not os.path.exists(resolved_path) and not using_dummy:
            self.logger.error(f"Transformer model path does not exist: {resolved_path}")
            return None

        try:
            # Use your existing load_anonymizer function
            if load_anonymizer and datasets:
                # Create a mock args object with required attributes
                class MockArgs:
                    def __init__(self):
                        self.dataset = 'ntu'
                        self.batch_size = 32
                        self.loading_transformer = False

                args = MockArgs()
                ds = datasets['ntu']  # Get dataset config

                # Use your existing load_anonymizer function
                model = load_anonymizer('transformer', resolved_path, self.device, args, ds)
                return model
            else:
                self.logger.error("load_anonymizer function or datasets not available")
                return None

        except Exception as e:
            self.logger.error(f"Error loading transformer model: {str(e)}")
            return None

    def load_sgn_model(self, model_path: str, config: Dict[str, Any]) -> Optional[Any]:
        """Load SGN model."""
        if not os.path.exists(model_path):
            self.logger.error(f"SGN model path does not exist: {model_path}")
            return None

        try:
            # Determine task and dataset from config
            task = config.get('task', 'ar')
            dataset = config.get('dataset', 'ntu')

            # Get number of classes
            if task == 'ar':
                if dataset == 'ntu':
                    num_classes = 60
                elif dataset == 'ntu120':
                    num_classes = 120
                elif dataset == 'etri':
                    num_classes = 55
                else:
                    num_classes = 60
            elif task == 'ri':
                if dataset == 'ntu':
                    num_classes = 40
                elif dataset == 'ntu120':
                    num_classes = 106
                elif dataset == 'etri':
                    num_classes = 50
                else:
                    num_classes = 40
            elif task == 'gc':
                num_classes = 2
            else:
                num_classes = 60

            # Create SGN model
            model = SGN(
                num_class=num_classes,
                num_point=25,
                num_person=2,
                graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                edge_importance_weighting=True
            )

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            self.logger.error(f"Error loading SGN model: {str(e)}")
            return None

    def load_mixformer_model(self, model_path: str, config: Dict[str, Any]) -> Optional[Any]:
        """Load MixFormer model."""
        if not os.path.exists(model_path):
            self.logger.error(f"MixFormer model path does not exist: {model_path}")
            return None

        try:
            # Determine task and dataset from config
            task = config.get('task', 'ar')
            dataset = config.get('dataset', 'ntu')

            # Get number of classes (same logic as SGN)
            if task == 'ar':
                if dataset == 'ntu':
                    num_classes = 60
                elif dataset == 'ntu120':
                    num_classes = 120
                elif dataset == 'etri':
                    num_classes = 55
                else:
                    num_classes = 60
            elif task == 'ri':
                if dataset == 'ntu':
                    num_classes = 40
                elif dataset == 'ntu120':
                    num_classes = 106
                elif dataset == 'etri':
                    num_classes = 50
                else:
                    num_classes = 40
            elif task == 'gc':
                num_classes = 2
            else:
                num_classes = 60

            # Create MixFormer model
            model = SkeletonMixFormer(
                num_classes=num_classes,
                num_joints=25,
                num_frames=64,
                num_persons=2
            )

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            self.logger.error(f"Error loading MixFormer model: {str(e)}")
            return None

    def load_dmr_model(self, model_path: str, config: Dict[str, Any]) -> Optional[Any]:
        """Load DMR model using existing eval_model.py functions."""
        # For dummy models, we don't need the file to exist
        using_dummy = hasattr(load_anonymizer, '__name__') and load_anonymizer.__name__ == 'dummy_load_anonymizer'
        if not os.path.exists(model_path) and not using_dummy:
            self.logger.error(f"DMR model path does not exist: {model_path}")
            return None

        try:
            # Use your existing load_anonymizer function
            if load_anonymizer:
                # Create a mock args object with required attributes
                class MockArgs:
                    def __init__(self):
                        self.dataset = 'ntu'
                        self.batch_size = 32

                args = MockArgs()

                # Use your existing load_anonymizer function
                model = load_anonymizer('dmr', model_path, self.device, args, None)
                return model
            else:
                self.logger.error("load_anonymizer function not available")
                return None
        except Exception as e:
            self.logger.error(f"Error loading DMR model: {str(e)}")
            return None

    def load_pmr_model(self, model_path: str, config: Dict[str, Any]) -> Optional[Any]:
        """Load PMR model using existing eval_model.py functions."""
        # For dummy models, we don't need the file to exist
        using_dummy = hasattr(load_anonymizer, '__name__') and load_anonymizer.__name__ == 'dummy_load_anonymizer'
        if not os.path.exists(model_path) and not using_dummy:
            self.logger.error(f"PMR model path does not exist: {model_path}")
            return None

        try:
            # Use your existing load_anonymizer function
            if load_anonymizer:
                # Create a mock args object with required attributes
                class MockArgs:
                    def __init__(self):
                        self.dataset = 'ntu'
                        self.batch_size = 32

                args = MockArgs()

                # Use your existing load_anonymizer function
                model = load_anonymizer('pmr', model_path, self.device, args, None)
                return model
            else:
                self.logger.error("load_anonymizer function not available")
                return None
        except Exception as e:
            self.logger.error(f"Error loading PMR model: {str(e)}")
            return None

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        if model_name in self.model_cache:
            model = self.model_cache[model_name]
            if hasattr(model, 'parameters'):
                num_params = sum(p.numel() for p in model.parameters())
                return {
                    'name': model_name,
                    'type': type(model).__name__,
                    'num_parameters': num_params,
                    'device': str(next(model.parameters()).device) if hasattr(model, 'parameters') else 'N/A'
                }
        return {'name': model_name, 'status': 'not_loaded'}

    def clear_cache(self):
        """Clear the model cache to free memory."""
        self.model_cache.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.logger.info("Model cache cleared")
