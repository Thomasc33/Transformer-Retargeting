import torch
import torch.nn as nn
import os

def safe_load_model(model_path, map_location=None):
    """Safe model loading function."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        if model_path.endswith('.tar'):
            checkpoint = torch.load(model_path, map_location=map_location)
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            else:
                return checkpoint
        else:
            return torch.load(model_path, map_location=map_location)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise

def fix_missing_buffers(model):
    """
    Fix missing buffers in PyTorch models.
    This function ensures that all required buffers are properly initialized.
    """
    if model is None:
        return model

    try:
        # Iterate through all modules and fix missing buffers
        for name, module in model.named_modules():
            # Handle BatchNorm layers
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if not hasattr(module, 'num_batches_tracked') or module.num_batches_tracked is None:
                    module.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

            # Handle other common modules that might need buffer fixes
            if hasattr(module, 'reset_parameters'):
                try:
                    # Only reset if the module seems to be missing required buffers
                    if hasattr(module, 'running_mean') and module.running_mean is None:
                        module.reset_parameters()
                    elif hasattr(module, 'running_var') and module.running_var is None:
                        module.reset_parameters()
                except Exception:
                    # If reset_parameters fails, continue silently
                    pass

        return model
    except Exception as e:
        print(f"Warning: Could not fix missing buffers: {e}")
        return model

