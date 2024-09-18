"""
Base wrapper class for external SOTA models.

Provides a standard interface for all model wrappers to ensure consistency
and compatibility with our training/evaluation pipeline.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseModelWrapper(nn.Module, ABC):
    """
    Abstract base class for model wrappers.
    
    All wrapper classes should inherit from this class and implement
    the required abstract methods.
    
    Attributes:
        model_name: Name of the wrapped model
        external_repo_path: Path to the external repository
        num_class: Number of action classes
        num_point: Number of skeleton joints (default: 25 for NTU)
        num_person: Number of persons (default: 1 for single-actor)
        in_channels: Number of input channels (default: 3 for x,y,z)
    """
    
    def __init__(
        self,
        model_name: str,
        external_repo_name: str,
        num_class: int,
        num_point: int = 25,
        num_person: int = 1,
        in_channels: int = 3,
        **kwargs
    ):
        """
        Initialize the base wrapper.
        
        Args:
            model_name: Name of the model (e.g., 'CTR-GCN')
            external_repo_name: Name of the external repository directory
            num_class: Number of action classes
            num_point: Number of skeleton joints
            num_person: Number of persons in the sequence
            in_channels: Number of input channels (x, y, z coordinates)
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.external_repo_name = external_repo_name
        
        # Get path to external repository
        self.external_repo_path = self._get_external_repo_path(external_repo_name)
        
        # Add external repo to Python path with highest priority
        self._add_to_path(self.external_repo_path)
    
    @staticmethod
    def _get_external_repo_path(repo_name: str) -> Path:
        """
        Get the absolute path to an external repository.
        
        Args:
            repo_name: Name of the repository directory
            
        Returns:
            Path object pointing to the external repository
            
        Raises:
            FileNotFoundError: If the repository doesn't exist
        """
        # Get the project root (4 levels up from this file)
        project_root = Path(__file__).parent.parent.parent.parent
        repo_path = project_root / 'external_repo_for_reference' / repo_name
        
        if not repo_path.exists():
            raise FileNotFoundError(
                f"External repository not found: {repo_path}\n"
                f"Expected location: external_repo_for_reference/{repo_name}"
            )
        
        return repo_path
    
    @staticmethod
    def _add_to_path(path: Path):
        """
        Add a path to sys.path with highest priority.
        
        Args:
            path: Path to add to sys.path
        """
        path_str = str(path)
        # Remove if already present to re-add at front
        if path_str in sys.path:
            sys.path.remove(path_str)
        # Insert at position 0 for highest priority
        sys.path.insert(0, path_str)
    
    @staticmethod
    def _clean_conflicting_modules(repo_name: str):
        """
        Remove conflicting modules from sys.modules to avoid import conflicts.
        
        Only removes modules that don't belong to the current repo.
        
        Args:
            repo_name: Name of the repository being loaded
        """
        # Common conflicting module names
        conflicts = ['graph', 'model', 'feeders', 'utils']
        
        for conflict in conflicts:
            # Only remove if not from current repo
            to_remove = []
            
            # Check base module
            if conflict in sys.modules:
                module = sys.modules[conflict]
                if hasattr(module, '__file__') and module.__file__:
                    if repo_name not in str(module.__file__):
                        to_remove.append(conflict)
                else:
                    # No file info, safer to remove
                    to_remove.append(conflict)
            
            # Check submodules
            for key in list(sys.modules.keys()):
                if key.startswith(f"{conflict}."):
                    module = sys.modules[key]
                    if hasattr(module, '__file__') and module.__file__:
                        if repo_name not in str(module.__file__):
                            to_remove.append(key)
                    else:
                        to_remove.append(key)
            
            # Remove collected modules
            for key in to_remove:
                if key in sys.modules:
                    del sys.modules[key]
    
    @staticmethod
    def _import_from_path(module_path: str, class_name: str, repo_path: Path):
        """
        Dynamically import a class from an external repository.
        
        This method ensures that the entire package context is available,
        allowing internal imports within the external repo to work correctly.
        
        Args:
            module_path: Module path relative to repo (e.g., 'model.infogcn')
            class_name: Name of the class to import (e.g., 'InfoGCN')
            repo_path: Path to the external repository
            
        Returns:
            The imported class
            
        Raises:
            ImportError: If import fails
        """
        import importlib
        import importlib.util
        
        # Create a unique prefix for this repo to avoid conflicts
        repo_name = repo_path.name
        prefixed_module_path = f"_ext_{repo_name}_{module_path}"
        
        # Clean conflicting modules from previous imports
        BaseModelWrapper._clean_conflicting_modules(repo_name)
        
        # Add repo to path at the beginning to take precedence
        repo_str = str(repo_path)
        if repo_str in sys.path:
            sys.path.remove(repo_str)
        sys.path.insert(0, repo_str)
        
        # Load all parent packages first to establish context
        parts = module_path.split('.')
        for i in range(1, len(parts) + 1):
            partial_path = '.'.join(parts[:i])
            prefixed_partial = f"_ext_{repo_name}_{partial_path}"
            
            if prefixed_partial in sys.modules:
                continue
                
            if i < len(parts):
                # Parent package
                parent_dir = repo_path / '/'.join(parts[:i])
                init_file = parent_dir / '__init__.py'
                if init_file.exists():
                    spec = importlib.util.spec_from_file_location(prefixed_partial, init_file)
                else:
                    # Create a dummy package if __init__.py doesn't exist
                    spec = importlib.util.spec_from_loader(prefixed_partial, loader=None)
            else:
                # The actual module
                module_file = repo_path / (partial_path.replace('.', '/') + '.py')
                if not module_file.exists():
                    # Try without prefix (standard import)
                    try:
                        module = importlib.import_module(module_path)
                        return getattr(module, class_name)
                    except (ImportError, AttributeError) as e:
                        raise ImportError(
                            f"Could not find module file: {module_file}\n"
                            f"Original error: {e}"
                        )
                spec = importlib.util.spec_from_file_location(prefixed_partial, module_file)
            
            if spec:
                if spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[prefixed_partial] = module
                    # Also register without prefix for internal imports
                    sys.modules[partial_path] = module
                    if spec.loader:
                        spec.loader.exec_module(module)
                else:
                    # Dummy package
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[prefixed_partial] = module
                    sys.modules[partial_path] = module
        
        # Return the class from the loaded module
        final_module = sys.modules.get(prefixed_module_path) or sys.modules.get(module_path)
        if final_module:
            return getattr(final_module, class_name)
        
        raise ImportError(f"Failed to load {module_path} from {repo_path}")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (N, C, T, V, M)
               N: batch size
               C: number of channels (3 for x,y,z)
               T: number of frames (64)
               V: number of joints (25)
               M: number of persons (1)
        
        Returns:
            Output logits of shape (N, num_class)
        """
        pass
    
    @abstractmethod
    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        Load pre-trained weights from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce that the keys in checkpoint
                   match the keys in the model (default: False to allow
                   loading with different num_classes)
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        pass
    
    def get_num_params(self) -> int:
        """
        Get the total number of parameters in the model.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'model_name': self.model_name,
            'num_class': self.num_class,
            'num_point': self.num_point,
            'num_person': self.num_person,
            'in_channels': self.in_channels,
            'num_params': self.get_num_params(),
            'external_repo': str(self.external_repo_path),
        }
    
    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"num_class={self.num_class}, "
            f"num_point={self.num_point}, "
            f"num_person={self.num_person}, "
            f"num_params={self.get_num_params():,})"
        )
