import torch
import os
import logging

logger = logging.getLogger(__name__)

def safe_load_model(model, model_path, device='cuda', strict=False):
    """
    Safely load a model with missing keys, printing warnings but not failing.
    
    Args:
        model: The model to load weights into
        model_path: Path to the model weights
        device: Device to load the model onto
        strict: Whether to strictly enforce all keys matching
        
    Returns:
        The loaded model
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return model
    
    try:
        # Load the state dict
        weights = torch.load(model_path, map_location=device)
        
        # Handle different formats
        if isinstance(weights, torch.nn.DataParallel):
            weights = weights.module.state_dict()
        elif not isinstance(weights, dict):
            weights = weights.state_dict()
        
        # Remove 'module.' prefix if it exists (from DataParallel)
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        
        # Get model state dict to check for missing keys
        model_dict = model.state_dict()
        
        # Find missing keys
        missing_keys = [k for k in model_dict.keys() if k not in weights]
        unexpected_keys = [k for k in weights.keys() if k not in model_dict]
        
        if missing_keys:
            logger.warning(f"Missing keys in state_dict: {missing_keys}")
        
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
        
        # Filter out missing keys if not strict
        if not strict:
            weights = {k: v for k, v in weights.items() if k in model_dict}
        
        # Load the filtered weights
        model.load_state_dict(weights, strict=strict)
        logger.info(f"Model loaded from {model_path}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return model

def fix_missing_buffers(model):
    """
    Fix missing buffers in the model by re-registering them.
    This is particularly useful for the A_SE buffer in Spatial_MixFormer.
    
    Args:
        model: The model to fix
        
    Returns:
        The fixed model
    """
    # Check if model has Spatial_MixFormer layers
    for name, module in model.named_modules():
        if 'spa_mixf' in name and not hasattr(module, 'A_SE'):
            # This is a Spatial_MixFormer without A_SE buffer
            logger.info(f"Fixing missing A_SE buffer in {name}")
            
            # Get the A tensor from the module
            if hasattr(module, 'A_GEME'):
                # Extract the shape and data from A_GEME
                A_shape = module.A_GEME.shape
                groups = module.groups if hasattr(module, 'groups') else 8
                
                # Create a new A_SE buffer with the same shape as A_GEME
                A_SE = module.A_GEME.detach().clone()
                
                # Register the buffer
                module.register_buffer('A_SE', A_SE)
                
                logger.info(f"Registered A_SE buffer with shape {A_SE.shape}")
    
    return model
