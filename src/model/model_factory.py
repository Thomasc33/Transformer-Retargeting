"""
Unified model factory for loading action recognition models.

This module provides a centralized interface for creating and loading both
internal models (SGN, MixFormer) and external SOTA models (via wrappers).
"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn


class ModelFactory:
    """
    Factory class for creating and loading action recognition models.
    
    Provides a unified interface for:
    - Internal models: SGN, MixFormer (our implementations)
    - External models: CTR-GCN, InfoGCN, SkateFormer, HD-GCN, MAMP
    
    Usage:
        # Create a model
        model = ModelFactory.create_model('ctrgcn', num_class=40)
        
        # Load checkpoint
        model = ModelFactory.load_checkpoint(model, 'path/to/checkpoint.pth')
        
        # Get model info
        info = ModelFactory.get_model_info('infogcn')
    """
    
    # Registry of available models
    MODELS = {
        # Internal implementations (our code)
        'sgn': {
            'name': 'SGN',
            'description': 'Spatial-temporal Graph Neural Network',
            'type': 'internal',
            'class_path': 'src.model.sgn.SGN',
            'performance': '90% NTU60 X-View',
            'paper': 'Zhang et al., "Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition", CVPR 2020',
            'trained': True,
        },
        'mixformer': {
            'name': 'Skeleton-MixFormer',
            'description': 'Transformer-based action recognition with spatial-temporal mixing',
            'type': 'internal',
            'class_path': 'src.model.ske_mixf.Model',
            'performance': '94% NTU60 X-View',
            'paper': 'Based on MixFormer architecture',
            'trained': True,
        },
        
        # External implementations (via wrappers)
        'ctrgcn': {
            'name': 'CTR-GCN',
            'description': 'Channel-wise Topology Refinement Graph Convolution',
            'type': 'external',
            'wrapper_class': 'src.model.wrappers.ctrgcn_wrapper.CTRGCNWrapper',
            'repo': 'external_repo_for_reference/CTR-GCN',
            'performance': '89.9% NTU60 X-View, 88.9% NTU120 X-Set',
            'paper': 'Chen et al., "Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition", ICCV 2021',
            'trained': False,
        },
        'infogcn': {
            'name': 'InfoGCN',
            'description': 'Information Bottleneck Graph Convolution',
            'type': 'external',
            'wrapper_class': 'src.model.wrappers.infogcn_wrapper.InfoGCNWrapper',
            'repo': 'external_repo_for_reference/infogcn',
            'performance': '93.0% NTU60 X-View, 89.8% NTU120 X-Set',
            'paper': 'Chi et al., "Infogcn: Representation learning for human skeleton-based action recognition", CVPR 2022',
            'trained': False,
        },
        'skateformer': {
            'name': 'SkateFormer',
            'description': 'Skeletal-Temporal Transformer',
            'type': 'external',
            'wrapper_class': 'src.model.wrappers.skateformer_wrapper.SkateFormerWrapper',
            'repo': 'external_repo_for_reference/SkateFormer',
            'performance': '92.4% NTU60 X-View, 89.4% NTU120 X-Set',
            'paper': 'Duan et al., "Skateformer: Skeletal-temporal transformer for human action recognition", ECCV 2024',
            'trained': False,
        },
        'hdgcn': {
            'name': 'HD-GCN',
            'description': 'Hierarchically Decomposed Graph Convolution',
            'type': 'external',
            'wrapper_class': 'src.model.wrappers.hdgcn_wrapper.HDGCNWrapper',
            'repo': 'external_repo_for_reference/HD-GCN',
            'performance': '90.1% NTU60 X-View, 89.3% NTU120 X-Set',
            'paper': 'Lee et al., "Hierarchically Decomposed Graph Convolutional Networks for Skeleton-Based Action Recognition", ICCV 2023',
            'trained': False,
        },
        'mamp': {
            'name': 'MAMP',
            'description': 'Masked Motion Predictors',
            'type': 'external',
            'wrapper_class': 'src.model.wrappers.mamp_wrapper.MAMPWrapper',
            'repo': 'external_repo_for_reference/MAMP',
            'performance': '93.0% NTU60 X-Sub, 89.8% NTU120 X-Sub',
            'paper': 'Mao et al., "Masked Motion Predictors are Strong 3D Action Representation Learners", ICCV 2023',
            'trained': False,
        },
        # HGformer excluded due to severe bugs in external code
        # 'hgformer': {
        #     'name': 'HGformer',
        #     'description': 'Autoregressive Adaptive Hypergraph Transformer',
        #     'type': 'external',
        #     'wrapper_class': 'src.model.wrappers.hgformer_wrapper.HGformerWrapper',
        #     'repo': 'external_repo_for_reference/AutoregAd-HGformer',
        #     'performance': 'TBD (WACV 2025)',
        #     'paper': 'Autoregressive Adaptive Hypergraph Transformer for Skeleton-based Activity Recognition, WACV 2025',
        #     'trained': False,
        #     'note': 'EXCLUDED - Multiple bugs in external code, incompatible architecture',
        # },
    }
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict]:
        """
        Get information about all available models.
        
        Returns:
            Dictionary mapping model names to their metadata
        """
        return cls.MODELS.copy()
    
    @classmethod
    def get_model_info(cls, name: str) -> Dict:
        """
        Get information about a specific model.
        
        Args:
            name: Model name (e.g., 'sgn', 'ctrgcn')
            
        Returns:
            Dictionary containing model metadata
            
        Raises:
            ValueError: If model name is not recognized
        """
        if name not in cls.MODELS:
            available = ', '.join(cls.MODELS.keys())
            raise ValueError(
                f"Unknown model: {name}. "
                f"Available models: {available}"
            )
        
        return cls.MODELS[name].copy()
    
    @classmethod
    def create_model(
        cls,
        name: str,
        num_class: int,
        num_point: int = 25,
        num_person: int = 1,
        in_channels: int = 3,
        **kwargs
    ) -> nn.Module:
        """
        Create a model instance.
        
        Args:
            name: Model name from MODELS registry
            num_class: Number of action classes
            num_point: Number of joints (25 for NTU)
            num_person: Number of persons (1 for single-actor)
            in_channels: Input channels (3 for x,y,z)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is not recognized
            ImportError: If model class cannot be imported
        """
        if name not in cls.MODELS:
            available = ', '.join(cls.MODELS.keys())
            raise ValueError(
                f"Unknown model: {name}. "
                f"Available models: {available}"
            )
        
        model_info = cls.MODELS[name]
        model_type = model_info['type']
        
        if model_type == 'internal':
            # Load internal model (SGN, MixFormer)
            return cls._create_internal_model(
                name, model_info, num_class, num_point, num_person, in_channels, **kwargs
            )
        elif model_type == 'external':
            # Load external model via wrapper
            return cls._create_external_model(
                name, model_info, num_class, num_point, num_person, in_channels, **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @classmethod
    def _create_internal_model(
        cls,
        name: str,
        model_info: Dict,
        num_class: int,
        num_point: int,
        num_person: int,
        in_channels: int,
        **kwargs
    ) -> nn.Module:
        """Create an internal model (SGN, MixFormer)."""
        class_path = model_info['class_path']
        
        # Import the model class
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        
        # Create model instance with appropriate arguments
        if name == 'sgn':
            # SGN expects: num_classes, dataset, seg
            dataset = kwargs.get('dataset', 'ntu')
            seg = kwargs.get('seg', 64)  # number of frames
            
            # Create base model
            base_model = model_class(
                num_classes=num_class,
                dataset=dataset,
                seg=seg
            )
            
            # Wrap with input adapter for (N, C, T, V, M) -> (N, T, V*C) conversion
            class SGNAdapter(nn.Module):
                def __init__(self, sgn_model):
                    super().__init__()
                    self.sgn = sgn_model
                
                def forward(self, x):
                    # Input: (N, C, T, V, M)
                    # SGN expects: (N, T, V*C)
                    N, C, T, V, M = x.size()
                    # Reshape: (N, C, T, V, M) -> (N, T, V, C, M) -> (N, T, V*C)
                    x = x.permute(0, 2, 3, 1, 4).contiguous()  # (N, T, V, C, M)
                    x = x.view(N, T, V * C * M)  # (N, T, V*C*M)
                    return self.sgn(x)
                
                def load_state_dict(self, state_dict, strict=True):
                    return self.sgn.load_state_dict(state_dict, strict=strict)
                
                def state_dict(self):
                    return self.sgn.state_dict()
                
                def parameters(self):
                    return self.sgn.parameters()
            
            model = SGNAdapter(base_model)
        elif name == 'mixformer':
            # MixFormer expects: num_class, num_point, num_person, graph, graph_args, in_channels
            graph = kwargs.get('graph', 'graph.ntu_rgb_d.Graph')
            graph_args = kwargs.get('graph_args', {})
            model = model_class(
                num_class=num_class,
                num_point=num_point,
                num_person=num_person,
                graph=graph,
                graph_args=graph_args,
                in_channels=in_channels
            )
        else:
            raise ValueError(f"Unknown internal model: {name}")
        
        return model
    
    @classmethod
    def _create_external_model(
        cls,
        name: str,
        model_info: Dict,
        num_class: int,
        num_point: int,
        num_person: int,
        in_channels: int,
        **kwargs
    ) -> nn.Module:
        """Create an external model via wrapper."""
        wrapper_class_path = model_info['wrapper_class']
        
        # Import the wrapper class
        module_path, class_name = wrapper_class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        wrapper_class = getattr(module, class_name)
        
        # Create wrapper instance
        model = wrapper_class(
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            in_channels=in_channels,
            **kwargs
        )
        
        return model
    
    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        checkpoint_path: Union[str, Path],
        strict: bool = False,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        Load model weights from a checkpoint file.
        
        Handles both our checkpoints and external repo checkpoints with
        flexible format detection.
        
        Args:
            model: Model instance to load weights into
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce key matching (default: False)
            device: Device to load checkpoint to (default: 'cpu')
            
        Returns:
            Model with loaded weights
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Check if model has a load_pretrained method (wrappers)
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(str(checkpoint_path), strict=strict)
            return model
        
        # Otherwise, use standard PyTorch loading
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract state dict from different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {
            k.replace('module.', ''): v 
            for k, v in state_dict.items()
        }
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        
        return model
    
    @classmethod
    def print_model_info(cls, name: Optional[str] = None):
        """
        Print information about models.
        
        Args:
            name: Specific model name, or None to print all models
        """
        if name is not None:
            # Print info for specific model
            info = cls.get_model_info(name)
            print(f"\n{'='*80}")
            print(f"{info['name']} ({name})")
            print(f"{'='*80}")
            print(f"Description: {info['description']}")
            print(f"Type: {info['type']}")
            print(f"Performance: {info['performance']}")
            print(f"Paper: {info['paper']}")
            print(f"Trained: {'✅' if info['trained'] else '⏳ Needs training'}")
            if info['type'] == 'external':
                print(f"Repository: {info['repo']}")
        else:
            # Print info for all models
            print(f"\n{'='*80}")
            print("AVAILABLE ACTION RECOGNITION MODELS")
            print(f"{'='*80}\n")
            
            # Group by type
            internal_models = {k: v for k, v in cls.MODELS.items() if v['type'] == 'internal'}
            external_models = {k: v for k, v in cls.MODELS.items() if v['type'] == 'external'}
            
            print("Internal Models (Our Implementations):")
            print("-" * 80)
            for model_id, info in internal_models.items():
                status_icon = "✅" if info['trained'] else "⏳"
                print(f"{status_icon} {info['name']} ({model_id})")
                print(f"   {info['description']}")
                print(f"   Performance: {info['performance']}")
                print()
            
            print("\nExternal Models (Via Wrappers):")
            print("-" * 80)
            for model_id, info in external_models.items():
                status_icon = "✅" if info['trained'] else "⏳"
                print(f"{status_icon} {info['name']} ({model_id})")
                print(f"   {info['description']}")
                print(f"   Performance: {info['performance']}")
                print(f"   Repository: {info['repo']}")
                print()
